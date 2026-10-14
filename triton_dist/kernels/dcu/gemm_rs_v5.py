"""S13: windowed reduce-scatter v5/v3 (paper #1 RS design) on DCU planes.

C = sum_r a_r @ b_r^T; rank r KEEPS rows [r*M/R, (r+1)*M/R) of the sum
(tensor-parallel row-sharded output). Replaces the S6 RS op's
compute-everything -> scatter-everything -> 2x SymmBarrier structure with
the ticketed windowed pipeline: ZERO global barrier, chunked producer,
per-(dest, chunk) engine pushes, per-slot consumers that start as soon as
their chunk's arrivals land.

Source design (Tri-dist-hybird new_3rd_v5_frontier_windowed_panel_rsgemm /
new_3rd_v3_windowed_panel_rs):
- dual-phase frontier producer (PHASE_KIND=0 computes the first
  frontier_chunks rows of EVERY dest segment first, then the tails),
- banded (band, dest, chunk) readiness via an in-kernel atomic counter
  whose last tile fires a release store,
- v3 windowed engine: Lambda=active_chunk_window scatter slots,
  ticket = round_base + chunk*n_bands + band + 1, dest free_flag gates slot
  reuse, local_seed_direct, steady/tail reduce CTA budgets, comm lanes
  rotate (dest + chunk) % lanes.

What changed for DCU (S1-S11 planes, same substitutions as S12):
- the dual-phase frontier becomes the HOST enqueue ORDER (chunk-major:
  every segment's chunk 0 before any chunk 1 -- ONE FUSED GEMM launch per
  chunk covering all R segment panels; per-segment launches under-fill the
  CUs and per-segment streams hit S11-class multi-stream compute
  contention), and the in-kernel readiness
  counter becomes a host-issued 4 B engine write of the chunk's ticket
  STREAM-ORDERED after that GEMM launch. Same granularity, no kernel
  changes: the S12 panel GEMM kernel is reused verbatim.
- _wait_eq_cuda -> hipStreamWaitValue32 GTE on OWN cells (S12-probed: the
  stream-wait poll observes engine-written flags).
- _set_signal_cuda -> 4 B engine memcpy of the ticket from the arange
  staging buffer.
- free_flag direction FLIPPED (the consumer ACKs into every pusher's heap
  after finishing a slot), so every wait is local and every remote touch
  is an engine memcpy or an in-kernel atomic. arrival[src, slot] at MY heap
  is written by src after its push (identical to S12). The consumer's ack
  is a 1-CTA kernel launched right after the reduce kernel (release xchg
  into each pusher's heap): a host-side 4 B engine memcpy there
  deterministically WEDGES the DTK runtime's enqueue at the final shapes
  (device fully drained, host spinning inside hipMemcpyAsync), and an
  in-kernel last-CTA counter ack raced the peer's next push (weak loads
  still in flight) -- see kernel_ack_slot.
- local_seed_direct is the default AND only mode: the consumer reads my
  own partial rows straight from MY local gemm_out (never scattered).

Buffers:
  gemm_out   LOCAL torch [max_M, N]   my partial (NOT in the symmetric heap)
  scatter    symmetric [S, R, chunk_rows, N]  compact staging: S slots, not
             num_chunks slots (the v3 compaction)
  arrival    symmetric int32 [R, S] at MY heap (src signals "push landed")
  ack        symmetric int32 [R, S] at MY heap (src signals "slot consumed")
  ready      LOCAL int32 [num_chunks_max]  per-chunk GEMM readiness tickets
"""
import ctypes
import dataclasses
import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

from triton_dist.kernels.dcu.common_ops import (
    dcu_state,
    hip,
    init_dcu,
    memcpy_async,
)
from triton_dist.kernels.dcu.gemm_ar_rd import (
    MAX_TICKETS,
    HIP_WAIT_GTE,
    _wait_value_gte,
    _wait_ticket,
)


@triton.jit(do_not_specialize=["scatter_base", "local_base", "rows", "N",
                               "local_rank"])
def kernel_reduce_slot_with_local(
        scatter_base,  # symmetric-heap VA (int64) of scatter[slot]: [R, CR, N]
        local_base,  # int64 VA of my partial panel rows (local seed)
        out_ptr,  # [rows, N] output chunk view
        rows, N,
        local_rank,
        stride_local_m,
        stride_outm,
        NUM_SPLITS: tl.constexpr,
        SLOT_ROWS: tl.constexpr,  # chunk_rows (scatter region row stride)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    """out = local_seed + sum_{split != me} scatter[slot][split], fp32 math,
    grid-stride over tiles so the host can cap CTAs (steady/tail budgets)."""
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(rows, BLOCK_SIZE_M)
    total_tiles = num_pid_m * num_pid_n

    dtype = out_ptr.dtype.element_ty
    scatter = scatter_base.to(tl.pointer_type(dtype))
    local = local_base.to(tl.pointer_type(dtype))

    for tile_id in range(pid, total_tiles, num_pid):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask = (offs_m[:, None] < rows) & (offs_n[None, :] < N)

        acc = tl.load(local + offs_m[:, None] * stride_local_m + offs_n[None, :],
                      mask=mask, other=0.0).to(tl.float32)
        for split in range(NUM_SPLITS):
            if split != local_rank:
                src_rows = split * SLOT_ROWS + offs_m
                acc += tl.load(scatter + src_rows[:, None] * N
                               + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
        out_ptrs = (out_ptr + offs_m[:, None] * stride_outm
                    + offs_n[None, :])
        tl.store(out_ptrs, acc.to(dtype), mask=mask)


@triton.jit(do_not_specialize=["tab", "slot", "stride", "val", "nrows"])
def kernel_ack_slot(tab, slot, stride, val, nrows):
    """1 CTA: ack every pusher of this slot (release xchg into each peer
    heap). Launched on the slot stream DIRECTLY AFTER the reduce kernel: the
    launch boundary is the guarantee that ALL of the reduce's scatter reads
    retired, so the next pusher may reuse the slot. (Two earlier designs
    both failed at the final shapes: a host 4 B engine memcpy wedged the DTK
    runtime's enqueue, and an in-kernel last-CTA counter acked while weak
    loads were still in flight -- the peer's next push stomped one chunk.)"""
    tabp = tab.to(tl.pointer_type(tl.int64))
    for i in range(nrows):
        dst = tl.load(tabp + i * stride + slot)
        tl.atomic_xchg(dst.to(tl.pointer_type(tl.int32)), val)


_ack_ck_cache = {}


def _launch_ack_slot(ack_tab, slot, stage_slots, ticket, num_ranks):
    """kernel_ack_slot with the int64-VA warmup trick (heap VAs exceed
    int32; see _wait_ticket's spin-kernel note). Caller must already be in
    the slot-stream context."""
    ck = _ack_ck_cache.get("k")
    if ck is None:
        ck = kernel_ack_slot.warmup(1 << 40, 0, 4, 1, 1, grid=(1,),
                                    num_warps=1)
        ck._init_handles()
        _ack_ck_cache["k"] = ck
    ck[(1, 1, 1)](ack_tab.data_ptr(), slot, stage_slots, ticket,
                  num_ranks - 1)


@triton.jit(do_not_specialize=["chunk_row_base", "rows", "M",
                               "seg_row_stride", "tiles_per_seg"])
def kernel_gemm_banded_rd(
        a_ptr, b_ptr, c_ptr,  # c: full [max_M, N] gemm_out (strided panels)
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        seg_row_stride,  # m_per_rank: rows between segment starts
        chunk_row_base,  # ck * chunk_rows: chunk start within a segment
        rows,  # chunk rows (per segment)
        tiles_per_seg,  # cdiv(rows, BM) * cdiv(N, BN)
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """All R segments' chunk-ck panels in ONE launch: the frontier's
    (segment, chunk) work items co-scheduled the way the source persistent
    kernel did. Per-segment panel launches leave the CUs underfilled (S13
    bench: 2x [1024,N] launches = 0.34 ms vs this = 0.17 ms at 2 cards) and
    per-segment streams do not fix it (S11-class multi-stream compute
    contention). Same body as gemm_ar_rd.kernel_gemm_panel_rd otherwise."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    seg = tl.program_id(axis=0) // tiles_per_seg
    pid = tl.program_id(axis=0) % tiles_per_seg
    num_pid_m = tl.cdiv(rows, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    row0 = seg * seg_row_stride + chunk_row_base
    offs_am = (row0 + pid_m * BLOCK_SIZE_M
               + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    dtype = a_ptr.dtype.element_ty
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + (row0 + offs_cm)[:, None] * N
              + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < rows) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


@dataclasses.dataclass
class GemmRSv5Context:
    rank: int
    num_ranks: int
    max_M: int
    N: int
    chunk_rows: int  # within a dest segment
    num_chunks: int  # at max_M
    stage_slots: int
    comm_lanes: int
    elemsz: int
    output_dtype: torch.dtype
    scatter_ptr: int  # symmetric [slots, R, chunk_rows, N] elems
    arrival_ptr: int  # symmetric int32 [R, slots] (MY heap view)
    ack_ptr: int  # symmetric int32 [R, slots] (MY heap view)
    peer_scatter: List[int]
    peer_arrival: List[int]
    peer_ack: List[int]
    gemm_out: torch.Tensor  # LOCAL [max_M, N]
    ready: torch.Tensor  # LOCAL int32 [num_chunks_max]
    ack_tab: torch.Tensor  # LOCAL int64 [R-1, S]: peer ack-cell VAs
    tickets: torch.Tensor
    compute_stream: torch.cuda.Stream
    lane_streams: List[torch.cuda.Stream]
    slot_streams: List[torch.cuda.Stream]
    block_m: int
    steady_sms: int
    tail_sms: int
    tail_chunk_window: int
    round_base: int = 0
    round_stride: int = 0
    last_ticket_per_slot: List[int] = dataclasses.field(default_factory=list)
    wait_mode: str = "stream"
    _events: Optional[list] = None  # [fork, join streams...) raw hipEvents


def _round_up(x, align):
    return ((x + align - 1) // align) * align


# byte-aware panel cap (same rationale as gemm_ar_rd: LLaMA-scale N makes
# 512-row panels a 30-50 MB single engine op -- kill the window)
PANEL_BYTES_CAP = 8 << 20


def _panel_rows_cap(N, elemsz):
    rows = PANEL_BYTES_CAP // max(1, N * elemsz)
    rows = max(256, (rows + 255) // 256 * 256)
    return rows


def _auto_chunk_rows(m_per_rank, target_chunks_per_rank=2, min_chunk_rows=512,
                     align=256):
    rows = triton.cdiv(m_per_rank, target_chunks_per_rank)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, m_per_rank)
    rows = _round_up(rows, align)
    return min(rows, m_per_rank)


def create_gemm_rs_v5_context(max_M, N, output_dtype, rank, num_ranks, mpi,
                              comm, chunk_rows=0, stage_slots=4, comm_lanes=2,
                              steady_sms=0, tail_sms=0, tail_chunk_window=1,
                              wait_mode=None):
    """Symmetric scatter/arrival/ack + local gemm_out + ready flags + the
    stream set (compute / lanes / slots). mpi/comm from
    pydushmem.host.mpi_init() (BEFORE torch)."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert max_M % num_ranks == 0, "row-sharded output needs M % R == 0"
    assert N % 128 == 0
    m_per_rank = max_M // num_ranks
    assert m_per_rank % 64 == 0

    elemsz = torch.empty((), dtype=output_dtype).element_size()
    if chunk_rows <= 0:
        chunk_rows = _auto_chunk_rows(m_per_rank)
        chunk_rows = min(chunk_rows, _panel_rows_cap(N, elemsz))
    num_chunks = triton.cdiv(m_per_rank, chunk_rows)
    stage_slots = max(1, min(stage_slots, num_chunks))
    comm_lanes = max(1, min(comm_lanes, num_ranks))

    s = dcu_state()
    slot_elems = chunk_rows * N
    scatter_ptr = s["host"].malloc(stage_slots * num_ranks * slot_elems * elemsz)
    flags_bytes = 4 * num_ranks * stage_slots
    arrival_ptr = s["host"].malloc(flags_bytes)
    ack_ptr = s["host"].malloc(flags_bytes)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(arrival_ptr), 0, flags_bytes) == 0
    assert h.hipMemset(ctypes.c_void_p(ack_ptr), 0, flags_bytes) == 0

    host = s["host"]
    peer_scatter = [host.ptr(scatter_ptr, d) for d in range(num_ranks)]
    peer_arrival = [host.ptr(arrival_ptr, d) for d in range(num_ranks)]
    peer_ack = [host.ptr(ack_ptr, d) for d in range(num_ranks)]

    gemm_out = torch.empty(max_M, N, dtype=output_dtype, device="cuda")
    # one flag per CHUNK: the chunk's fused (all-segment) GEMM is one launch,
    # so every segment's panel completes together (frontier semantics)
    ready = torch.zeros(num_chunks, dtype=torch.int32, device="cuda")
    # per-chunk ack cell VAs, one row per pusher src != me, one col per slot:
    # kernel_ack_slot xchgs the ticket into each (see its docstring)
    ack_tab = torch.tensor(
        [[peer_ack[src] + (rank * stage_slots + slot) * 4
          for slot in range(stage_slots)]
         for src in range(num_ranks) if src != rank],
        dtype=torch.int64, device="cpu").cuda()
    tickets = torch.arange(MAX_TICKETS, dtype=torch.int32, device="cuda")

    h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    h.hipEventCreateWithFlags.restype = ctypes.c_int
    h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipEventRecord.restype = ctypes.c_int
    h.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_uint64]
    h.hipStreamWaitEvent.restype = ctypes.c_int
    n_streams = 1 + stage_slots + comm_lanes
    events = []
    for _ in range(1 + n_streams):
        ev = ctypes.c_void_p()
        rc = h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2)
        assert rc == 0, f"hipEventCreateWithFlags rc={rc}"
        events.append(ev)

    ctx = GemmRSv5Context(
        rank=rank, num_ranks=num_ranks, max_M=max_M, N=N,
        chunk_rows=chunk_rows, num_chunks=num_chunks, stage_slots=stage_slots,
        comm_lanes=comm_lanes, elemsz=elemsz, output_dtype=output_dtype,
        scatter_ptr=scatter_ptr, arrival_ptr=arrival_ptr, ack_ptr=ack_ptr,
        peer_scatter=peer_scatter, peer_arrival=peer_arrival,
        peer_ack=peer_ack, gemm_out=gemm_out, ready=ready, ack_tab=ack_tab,
        tickets=tickets,
        compute_stream=torch.cuda.Stream(),
        lane_streams=[torch.cuda.Stream() for _ in range(comm_lanes)],
        slot_streams=[torch.cuda.Stream() for _ in range(stage_slots)],
        # BM=128 for any panel the byte-cap allows (the S15 probe: BM64/s2
        # 38.2 ms vs BM128/s2 23.9 ms at the final shapes; see gemm_ar_rd)
        block_m=128 if chunk_rows >= 128 else 64,
        steady_sms=steady_sms, tail_sms=tail_sms,
        tail_chunk_window=tail_chunk_window,
        round_stride=num_chunks + 1,
        last_ticket_per_slot=[0] * stage_slots,
        wait_mode=wait_mode or os.environ.get("RD_WAIT_MODE", "stream"),
        _events=events,
    )

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ctx


def finalize_gemm_rs_v5_context(ctx):
    s = dcu_state()
    for p in (ctx.scatter_ptr, ctx.arrival_ptr, ctx.ack_ptr):
        if p:
            s["host"].free(p)
    ctx.scatter_ptr = ctx.arrival_ptr = ctx.ack_ptr = 0


def gemm_rs_v5_op(a, b, out, ctx, component="full"):
    """C = sum_r a_r @ b_r^T; rank r keeps rows [r*M/R, (r+1)*M/R).
    Everything is enqueued up front; compute/lane/slot streams self-pace on
    tickets. component="gemm": the frontier-ordered chunk GEMMs only."""
    M, K = a.shape
    N = b.shape[0]
    rank, R = ctx.rank, ctx.num_ranks
    CR, S, L = ctx.chunk_rows, ctx.stage_slots, ctx.comm_lanes
    assert b.shape[1] == K, "b must be [N, K]"
    assert M % R == 0 and M // R % 64 == 0 and M <= ctx.max_M
    assert N == ctx.N and K % 64 == 0
    assert a.dtype == ctx.output_dtype and b.dtype == ctx.output_dtype
    assert a.stride(1) == 1 and a.stride(0) == K
    assert b.stride(1) == 1 and b.stride(0) == K
    assert out.shape == (M // R, N)

    cur = torch.cuda.current_stream()
    h = hip()
    m_pr = M // R
    n_chunks = triton.cdiv(m_pr, CR)
    mode = ctx.wait_mode
    last = ctx.last_ticket_per_slot

    if component == "full":
        ctx.round_base += ctx.round_stride
        assert ctx.round_base + n_chunks < MAX_TICKETS
    tickets_va = ctx.tickets.data_ptr()

    def ticket(ck):
        return ctx.round_base + ck + 1

    # fork every op stream after the caller (gemm-only needs just the
    # producer stream -- forking the unused lanes/slots costs ~0.1 ms of
    # host-submit time that the device timeline absorbs as idle gaps)
    if component == "gemm":
        all_streams = [ctx.compute_stream]
    else:
        all_streams = ([ctx.compute_stream] + ctx.lane_streams
                       + ctx.slot_streams)
    ev_fork = ctx._events[0]
    rc = h.hipEventRecord(ev_fork, ctypes.c_void_p(cur.cuda_stream))
    assert rc == 0, f"fork hipEventRecord rc={rc}"
    for st in all_streams:
        rc = h.hipStreamWaitEvent(ctypes.c_void_p(st.cuda_stream),
                                  ev_fork, 0)
        assert rc == 0, f"fork hipStreamWaitEvent rc={rc}"

    cs = ctx.compute_stream
    # producer: CHUNK-MAJOR (the v5 dual-phase frontier collapsed into
    # enqueue order): chunk 0 of EVERY dest segment, then chunk 1, ... --
    # one FUSED launch per chunk covering all R segment panels.
    with torch.cuda.stream(cs):
        for ck in range(n_chunks):
            rows = min(CR, m_pr - ck * CR)
            tiles = (triton.cdiv(rows, ctx.block_m) * triton.cdiv(N, 128))
            kernel_gemm_banded_rd[(tiles * R,)](
                a, b, ctx.gemm_out, M, N, K,
                a.stride(0), a.stride(1),
                b.stride(1), b.stride(0),
                m_pr, ck * CR, rows, tiles,
                BLOCK_SIZE_M=ctx.block_m, BLOCK_SIZE_N=128,
                BLOCK_SIZE_K=64, GROUP_SIZE_M=8, num_warps=4,
                num_stages=1 if (CR > 512 and K < 8192) else 2)
            if component == "full":
                # readiness: 4 B engine write, stream-ordered after the
                # fused chunk GEMM (the v5 in-kernel last-tile release
                # store). Every segment's panel completes with this launch.
                memcpy_async(ctx.ready.data_ptr() + ck * 4,
                             tickets_va + 4 * ticket(ck), 4,
                             cs.cuda_stream)

    if component == "full" and R > 1:
        # comm engine: per (chunk, dest != me), rotated across lanes
        for ck in range(n_chunks):
            slot = ck % S
            prev_free = last[slot]
            last[slot] = ticket(ck)
            for step in range(1, R):
                d = (rank + step) % R
                lane = ctx.lane_streams[(d + ck) % L]
                rows = min(CR, m_pr - ck * CR)
                src_va = (ctx.gemm_out.data_ptr()
                          + (d * m_pr + ck * CR) * N * ctx.elemsz)
                with torch.cuda.stream(lane):
                    # slot-reuse gate: dest consumed the previous occupant
                    _wait_ticket(ctx.ack_ptr + (d * S + slot) * 4,
                                 prev_free, lane, mode)
                    # chunk ck's fused GEMM is done (all segments)
                    _wait_ticket(ctx.ready.data_ptr() + ck * 4,
                                 ticket(ck), lane, mode)
                    dst = ctx.peer_scatter[d] + ((slot * R + rank)
                                                 * CR * N) * ctx.elemsz
                    memcpy_async(dst, src_va, rows * N * ctx.elemsz,
                                 lane.cuda_stream)
                    memcpy_async(ctx.peer_arrival[d] + (rank * S + slot) * 4,
                                 tickets_va + 4 * ticket(ck), 4,
                                 lane.cuda_stream)

        # consumers: per chunk, my OWN segment shard
        for ck in range(n_chunks):
            slot = ck % S
            rows = min(CR, m_pr - ck * CR)
            sst = ctx.slot_streams[slot]
            out_chunk = out[ck * CR:ck * CR + rows]
            local_va = (ctx.gemm_out.data_ptr()
                        + (rank * m_pr + ck * CR) * N * ctx.elemsz)
            tiles = triton.cdiv(rows, 128) * triton.cdiv(N, 128)
            tail = ck >= max(0, n_chunks - ctx.tail_chunk_window)
            # steady chunks co-schedule with the producer GEMM: a small CTA
            # budget that scales with the chunk's tile count (fixed 6 was
            # tuned on M2048-era 32-tile chunks; at the final shapes a chunk
            # is 448 tiles and a 6-CTA drain trails the producer by ~15 ms).
            # tail chunks run alone -- tail_sms=0 means take the whole grid.
            if tail:
                budget = ctx.tail_sms
            elif ctx.steady_sms > 0:
                budget = ctx.steady_sms
            else:
                budget = max(6, min(24, tiles // 4))
            ctas = tiles if budget <= 0 else max(1, min(tiles, budget))
            with torch.cuda.stream(sst):
                # chunk ck's fused GEMM is done (my local seed is in place)
                _wait_ticket(ctx.ready.data_ptr() + ck * 4,
                             ticket(ck), sst, mode)
                for step in range(1, R):
                    src = (rank + step) % R
                    _wait_ticket(ctx.arrival_ptr + (src * S + slot) * 4,
                                 ticket(ck), sst, mode)
                scatter_va = ctx.scatter_ptr + (slot * R * CR * N) * ctx.elemsz
                kernel_reduce_slot_with_local[(ctas,)](
                    scatter_va, local_va, out_chunk, rows, N,
                    rank, N, out_chunk.stride(0),
                    NUM_SPLITS=R, SLOT_ROWS=CR,
                    BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, num_warps=8)
                # ack every pusher of this slot: a 1-CTA kernel AFTER the
                # reduce launch (the launch boundary retires the reads; the
                # memcpy version wedges the runtime, the in-kernel counter
                # version races the peer's next push -- see kernel_ack_slot)
                _launch_ack_slot(ctx.ack_tab, slot, S, ticket(ck), R)

    # join: caller orders after every op stream (covers gemm_out's and
    # out's cross-stream lifetime for the caching allocator)
    for si, st in enumerate(all_streams):
        ev = ctx._events[1 + si]
        rc = h.hipEventRecord(ev, ctypes.c_void_p(st.cuda_stream))
        assert rc == 0, f"join hipEventRecord[{si}] rc={rc}"
        rc = h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream), ev, 0)
        assert rc == 0, f"join hipStreamWaitEvent[{si}] rc={rc}"
    return out


def gemm_rs_v5(a, b, ctx):
    """GEMM Reduce-Scatter (TP row-sharded): rank r returns rows
    [r*M/R, (r+1)*M/R) of sum_r a_r @ b_r^T."""
    if ctx is None:
        raise RuntimeError("requires ctx")
    out = torch.empty((a.shape[0] // ctx.num_ranks, b.shape[0]),
                      dtype=ctx.output_dtype, device=a.device)
    gemm_rs_v5_op(a, b, out, ctx)
    return out


__all__ = [
    "GemmRSv5Context",
    "create_gemm_rs_v5_context",
    "finalize_gemm_rs_v5_context",
    "gemm_rs_v5_op",
    "gemm_rs_v5",
    "kernel_reduce_slot_with_local",
    "kernel_gemm_banded_rd",
]
