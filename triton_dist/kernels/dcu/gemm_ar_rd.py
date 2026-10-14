"""S12: windowed recursive-doubling panel GEMM-AllReduce (paper #1 AR v23
design, re-derived on the validated DCU planes).

C = sum_r a_r @ b_r^T with the FULL C on every rank, ZERO global barrier
inside the op, and log2(R) panel exchanges instead of the S6 full-mesh
fan-out (R=8: 3x panel bytes pushed vs 7x -- the 17-25 GB/s/rank fabric
wall is the thing this attacks).

Source design (Tri-dist-hybird new_windowed_panel_gemm_allreduce_v23.py,
producer_order="panel_recursive_doubling"): host enqueues the WHOLE op up
front as (panel -> slot) tasks; per task the slot stream seeds out=local
panel, then per XOR phase pushes the current out to the partner, waits the
partner's arrival flag, and accumulates in place; slot reuse across tasks
is guarded by monotonic tickets (arrival/free), never by barriers.

What changed for DCU (every substitution is an S1-S11-validated plane):
- one stream per STAGE SLOT carries the panel GEMM too (their producer ran
  on a separate stream + stripe_ready flags + done_event lookahead). With
  the GEMM on the slot stream the seed copy AND the ready flag disappear:
  the GEMM writes straight into the output panel, which IS the RD
  accumulator. Cross-slot GEMM/comm overlap still comes from slot
  parallelism; there is not a single cross-stream event inside the op.
- _wait_eq_cuda -> hipStreamWaitValue32 with flags=hipStreamWaitValueGte
  (0) on OWN-heap flag cells. NB: common_ops.wait_eq puts the mask in the
  flags slot (4-arg call; the 5th param is `mask`), which errors at first
  call -- it was never exercised, so it is left untouched and this module
  carries its own correct binding. If the stream-wait poll cannot observe
  PEER-ENGINE-written flags on this stack (probed by tests/test_gemm_ar_rd.py),
  set RD_WAIT_MODE=spin for the 1-CTA hybrid-spin fallback (the proven
  SymmBarrier wait plane).
- _set_signal_cuda -> 4 B engine memcpy of the ticket from a device-side
  arange staging buffer (the SymmBarrier arrive write verbatim: engine
  writes to a peer's symmetric VA are coherent at the peer, and they keep
  the S5.3 signal-transport cliff out of the RMA path). Cross-device
  hipStreamWriteValue32 is not assumed to exist.
- free-flag direction FLIPPED: instead of the sender reading the dest's
  free flag remotely (nvshmem makes that cheap; DCU remote reads are not),
  every rank ACKs into each partner's heap after finishing its task on a
  slot, so ALL waits are on OWN memory and ALL remote touches are engine
  memcpys. Zero remote reads, zero remote atomics, zero device-initiated
  dushmem ops in the whole op.
- cudaMemcpy2DAsync -> common_ops.memcpy_async (n_bands=1: every panel
  copy is one contiguous rows*N block).
- GEMM config: the S5-tuned grouped-tile kernel verbatim (128x128x64/w4,
  BM=64 for chunk heights <= 512; num_stages by the S10 rule).

Flag layout (all in the dushmem symmetric heap, int32, zeroed ONCE at
create; tickets are monotonic and never reset):
  scatter[slot, src, chunk_rows, N]   engine-pushed panel snapshots
  arrival[src, slot]   at MY heap: src signals "my push for (slot, ticket)
                       landed"; written by src's 4 B engine memcpy.
  ack[src, slot]       at MY heap: src signals "I finished consuming slot
                       (src pushed into it / I read it) for that ticket";
                       written by src's 4 B engine memcpy after its LAST
                       phase. The next occupant of the slot GTE-waits it.

Ticket math: ticket(task i) = round_base + i + 1; round_base advances by
(num_chunks_max + 1) per op regardless of the runtime task count, so
tickets never collide across ops of different M.
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

HIP_WAIT_GTE = 0  # hipStreamWaitValueGte: wait until (*ptr & mask) >= value
MAX_TICKETS = 1 << 20  # arange staging buffer; ~200k ops at 4 tasks/op
# cap a panel's push size at ~8 MB of engine work: the row heuristic alone
# (target_chunks x min 512 rows) was tuned for N~2-8k; at LLaMA-scale N
# (29568/49152) 512 rows is a 30-50 MB single engine op -- bad windowing
# and a symmetric-heap blowup.
PANEL_BYTES_CAP = 8 << 20


def _panel_rows_cap(N, elemsz):
    rows = PANEL_BYTES_CAP // max(1, N * elemsz)
    rows = max(256, (rows + 255) // 256 * 256)
    return rows


def rd_wait_mode():
    return os.environ.get("RD_WAIT_MODE", "stream")


def _wait_value_gte(ptr: int, val: int, stream):
    """hipStreamWaitValue32 with the CORRECT 5-arg signature (stream, ptr,
    value, flags, mask). common_ops.wait_eq passes the mask in the flags
    slot and omits the mask -- do not use it."""
    h = hip()
    h.hipStreamWaitValue32.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_uint32, ctypes.c_uint,
                                       ctypes.c_uint]
    h.hipStreamWaitValue32.restype = ctypes.c_int
    rc = h.hipStreamWaitValue32(stream.cuda_stream, ctypes.c_void_p(ptr),
                                 ctypes.c_uint32(val), HIP_WAIT_GTE,
                                 0xFFFFFFFF)
    assert rc == 0, f"hipStreamWaitValue32 rc={rc} (ptr=0x{ptr:x} val={val})"


@triton.jit(do_not_specialize=["flag_ptr", "expected"])
def kernel_wait_flag_ge(flag_ptr, expected):
    """RD_WAIT_MODE=spin fallback: 1-CTA hybrid spin on OWN memory (the
    barrier_spin_kernel pattern -- volatile batches as delay, rare atomic
    probe at the L2 coherence point for forward progress against
    engine-written flags). scope=sys: the cell can be written by a PEER
    (the I1 ack plane) -- the acquire must pair with the peer's sys-scope
    release xchg across the fabric (2026-10-14)."""
    fp = flag_ptr.to(tl.pointer_type(tl.int32))
    v = tl.atomic_add(fp, 0, scope="sys")
    while v < expected:
        for _ in range(128):
            v = tl.load(fp, volatile=True)
        if v < expected:
            v = tl.atomic_add(fp, 0, scope="sys")


@triton.jit(do_not_specialize=["dst", "val"])
def kernel_flag_xchg(dst, val):
    """1 CTA: release-xchg `val` into a 4 B flag cell (any heap). The
    DTK26.04 runtime can wedge the HOST enqueue of tiny 4 B hipMemcpyAsync
    flag writes at final-shape queue depths (device drained, host spinning
    in-driver; nondeterministic trigger, hits local cells too) -- so every
    flag plane write in the S13/S14 modules fires from this kernel instead,
    stream-ordered after whatever data op it flags (the launch boundary
    also retires that producer's memory ops).

    sem=release, scope=sys (2026-10-14): this kernel is the CROSS-DEVICE
    release leg of the I1 stage->arrive->pull message chain (the arrive
    ticket write into the PEER's heap). A gpu-scoped xchg does not flush
    the producer's ws stores across the fabric, leaving the peer's
    sys-acquire pull a ~2%/rank-round stale-read window at big shapes;
    4 B so the stronger scope is free. Same-device legs (local flag
    planes) are unaffected -- strictly stronger."""
    tl.atomic_xchg(dst.to(tl.pointer_type(tl.int32)), val,
                   sem="release", scope="sys")


_flag_ck_cache = {}


def _launch_flag_xchg(dst: int, val: int):
    """kernel_flag_xchg with the int64-VA warmup trick (heap VAs exceed
    int32; see _wait_ticket's spin-kernel note). Runs on the CALLER's
    current stream."""
    ck = _flag_ck_cache.get("k")
    if ck is None:
        ck = kernel_flag_xchg.warmup(1 << 40, 1, grid=(1,), num_warps=1)
        ck._init_handles()
        _flag_ck_cache["k"] = ck
    ck[(1, 1, 1)](dst, val)


_wait_ck_cache = {}


def _wait_ticket(ptr: int, val: int, stream, mode: str):
    if mode == "stream":
        _wait_value_gte(ptr, val, stream)
    else:
        ck = _wait_ck_cache.get("k")
        if ck is None:
            # warmup with a >2^31 value so `flag_ptr` types as int64 (heap
            # VAs exceed int32; a literal 0 would compile an i32 kernel and
            # the cached launch would truncate real addresses)
            ck = kernel_wait_flag_ge.warmup(1 << 40, 1, grid=(1,),
                                            num_warps=1)
            ck._init_handles()
            _wait_ck_cache["k"] = ck
        ck[(1, 1, 1)](ptr, val)


@triton.jit(do_not_specialize=["chunk_row_start", "rows", "M"])
def kernel_gemm_panel_rd(
        a_ptr, b_ptr, c_ptr,  # c_ptr: [rows, N] contiguous output PANEL view
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        chunk_row_start,  # global row of this panel within a/c
        rows,  # rows this panel (== c.shape[0])
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """Grouped-tile matmul of ONE row-panel of the partial product directly
    into the caller's output panel (which doubles as the RD accumulator).
    Same body as the S6 gemm_allreduce producer (loads wrap instead of
    masking; the store masks the ragged panel tail)."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(rows, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (chunk_row_start + pid_m * BLOCK_SIZE_M
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
    c_ptrs = c_ptr + offs_cm[:, None] * N + offs_cn[None, :]
    c_mask = (offs_cm[:, None] < rows) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


@triton.jit(do_not_specialize=["src_base", "rows"])
def kernel_accumulate_panel_rd(
        src_base,  # symmetric-heap VA (int64) of scatter[slot, partner]
        out_ptr,  # [rows, N] output panel view (the RD accumulator)
        rows, N,
        stride_om, stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    """out += src in fp32 (engine-written peer data is fully coherent for
    plain loads -- S4). The scatter region is contiguous [chunk_rows, N];
    `rows` masks the ragged tail."""
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (offs_m[:, None] < rows) & (offs_n[None, :] < N)

    dtype = out_ptr.dtype.element_ty
    src = src_base.to(tl.pointer_type(dtype))
    v = tl.load(src + offs_m[:, None] * N + offs_n[None, :],
                mask=mask, other=0.0).to(tl.float32)
    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptrs, (o + v).to(dtype), mask=mask)


@dataclasses.dataclass
class GemmARRDContext:
    rank: int
    num_ranks: int
    max_M: int
    N: int
    chunk_rows: int
    num_chunks: int  # at max_M
    stage_slots: int
    elemsz: int
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    scatter_ptr: int  # symmetric [slots, R, chunk_rows, N] elems
    arrival_ptr: int  # symmetric int32 [R, slots] (MY heap view)
    ack_ptr: int  # symmetric int32 [R, slots] (MY heap view)
    peer_scatter: List[int]
    peer_arrival: List[int]
    peer_ack: List[int]
    tickets: torch.Tensor  # int32 arange [MAX_TICKETS] (memcpy staging)
    slot_streams: List[torch.cuda.Stream]
    partners: List[int]
    block_m: int
    round_base: int = 0
    round_stride: int = 0
    last_ticket_per_slot: List[int] = dataclasses.field(default_factory=list)
    wait_mode: str = "stream"
    _events: Optional[list] = None  # [fork, join_0..join_S) raw hipEvents


def _partners_of(rank, num_ranks):
    return [rank ^ (1 << p) for p in range(num_ranks.bit_length() - 1)]


def _round_up(x, align):
    return ((x + align - 1) // align) * align


def create_gemm_ar_rd_context(max_M, N, input_dtype, output_dtype,
                              rank, num_ranks, mpi, comm,
                              chunk_rows=0, target_chunks=4,
                              min_chunk_rows=512, stage_slots=4,
                              wait_mode=None):
    """Symmetric scatter/arrival/ack planes + per-slot streams for the RD
    pipeline. mpi/comm from pydushmem.host.mpi_init() (BEFORE torch)."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert num_ranks & (num_ranks - 1) == 0, "recursive doubling needs 2^k ranks"
    assert input_dtype == output_dtype, "panels are exchanged in output dtype"
    assert N % 128 == 0 and max_M % 64 == 0

    elemsz = torch.empty((), dtype=output_dtype).element_size()
    if chunk_rows <= 0:
        chunk_rows = triton.cdiv(max_M, target_chunks)
        chunk_rows = max(chunk_rows, min_chunk_rows)
        chunk_rows = min(chunk_rows, max_M)
        chunk_rows = _round_up(chunk_rows, 256)
        chunk_rows = min(chunk_rows, max_M)
        chunk_rows = min(chunk_rows, _panel_rows_cap(N, elemsz))
    num_chunks = triton.cdiv(max_M, chunk_rows)
    stage_slots = max(1, min(stage_slots, num_chunks))

    s = dcu_state()
    slot_elems = chunk_rows * N
    scatter_ptr = s["host"].malloc(stage_slots * num_ranks * slot_elems * elemsz)
    flags_bytes = 4 * num_ranks * stage_slots
    arrival_ptr = s["host"].malloc(flags_bytes)
    ack_ptr = s["host"].malloc(flags_bytes)
    # flags gate reads of possibly-unwritten memory: garbage >= ticket would
    # skip a wait. hipMemset of OWN memory is safe (peers write through the
    # engine, which updates our L2 lines).
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(arrival_ptr), 0, flags_bytes) == 0
    assert h.hipMemset(ctypes.c_void_p(ack_ptr), 0, flags_bytes) == 0

    host = s["host"]
    peer_scatter = [host.ptr(scatter_ptr, d) for d in range(num_ranks)]
    peer_arrival = [host.ptr(arrival_ptr, d) for d in range(num_ranks)]
    peer_ack = [host.ptr(ack_ptr, d) for d in range(num_ranks)]

    tickets = torch.arange(MAX_TICKETS, dtype=torch.int32, device="cuda")

    h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    h.hipEventCreateWithFlags.restype = ctypes.c_int
    h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipEventRecord.restype = ctypes.c_int
    h.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_uint64]
    h.hipStreamWaitEvent.restype = ctypes.c_int
    events = []
    for _ in range(1 + stage_slots):
        ev = ctypes.c_void_p()
        rc = h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2)  # DISABLE_TIMING
        assert rc == 0, f"hipEventCreateWithFlags rc={rc}"
        events.append(ev)

    ctx = GemmARRDContext(
        rank=rank, num_ranks=num_ranks, max_M=max_M, N=N,
        chunk_rows=chunk_rows, num_chunks=num_chunks, stage_slots=stage_slots,
        elemsz=elemsz, input_dtype=input_dtype, output_dtype=output_dtype,
        scatter_ptr=scatter_ptr, arrival_ptr=arrival_ptr, ack_ptr=ack_ptr,
        peer_scatter=peer_scatter, peer_arrival=peer_arrival,
        peer_ack=peer_ack, tickets=tickets,
        slot_streams=[torch.cuda.Stream() for _ in range(stage_slots)],
        partners=_partners_of(rank, num_ranks),
        # BM=128 for any panel the byte-cap allows (>=256 rows): the S15
        # probe at M8192/N28672/K8192 measured BM64/s2 38.2 ms vs BM128/s2
        # 23.9 ms for the same panels (the old "<=512 -> 64" rule was the
        # S5 whole-matrix decode tuning, not a panel rule)
        block_m=128 if chunk_rows >= 128 else 64,
        round_stride=num_chunks + 1,
        last_ticket_per_slot=[0] * stage_slots,
        wait_mode=wait_mode or rd_wait_mode(),
        _events=events,
    )

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ctx


def finalize_gemm_ar_rd_context(ctx):
    """Free the symmetric planes (the 8-PE heap cap is ~1.8 GB cumulative)."""
    s = dcu_state()
    for p in (ctx.scatter_ptr, ctx.arrival_ptr, ctx.ack_ptr):
        if p:
            s["host"].free(p)
    ctx.scatter_ptr = ctx.arrival_ptr = ctx.ack_ptr = 0


def gemm_ar_rd_op(a, b, c, ctx, component="full"):
    """C = sum_r a_r @ b_r^T, full C on every rank, via the windowed
    recursive-doubling panel pipeline. Everything is enqueued up front;
    the slot streams self-pace on tickets. component="gemm": panels only
    (bench decomposition; no comm, tickets NOT advanced)."""
    M, K = a.shape
    N = b.shape[0]
    rank, R, S = ctx.rank, ctx.num_ranks, ctx.stage_slots
    CR = ctx.chunk_rows
    assert b.shape[1] == K and c.shape == (M, N) and N == ctx.N
    assert M <= ctx.max_M and M % 64 == 0 and K % 64 == 0
    assert a.dtype == ctx.input_dtype and b.dtype == ctx.input_dtype
    assert a.stride(1) == 1 and a.stride(0) == K
    assert b.stride(1) == 1 and b.stride(0) == K

    cur = torch.cuda.current_stream()
    h = hip()
    n_tasks = triton.cdiv(M, CR)
    partners = ctx.partners
    mode = ctx.wait_mode

    if component == "full":
        ctx.round_base += ctx.round_stride
        assert ctx.round_base + n_tasks < MAX_TICKETS
    tickets_va = ctx.tickets.data_ptr()

    # fork: every slot stream orders after the caller (a/b/c are ready).
    # RAW hipEvents -- torch wait_stream costs 10-40 us/call (S11 evtax).
    ev_fork = ctx._events[0]
    assert h.hipEventRecord(ev_fork, ctypes.c_void_p(cur.cuda_stream)) == 0
    for sst in ctx.slot_streams:
        assert h.hipStreamWaitEvent(ctypes.c_void_p(sst.cuda_stream),
                                    ev_fork, 0) == 0

    for i in range(n_tasks):
        r0 = i * CR
        rows = min(CR, M - r0)
        slot = i % S
        sst = ctx.slot_streams[slot]
        ticket = prev_free = 0
        if component == "full":
            ticket = ctx.round_base + i + 1
            prev_free = ctx.last_ticket_per_slot[slot]
            ctx.last_ticket_per_slot[slot] = ticket
        c_panel = c[r0:r0 + rows]

        with torch.cuda.stream(sst):
            grid = (triton.cdiv(rows, ctx.block_m) * triton.cdiv(N, 128),)
            kernel_gemm_panel_rd[grid](
                a, b, c_panel,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(1), b.stride(0),
                r0, rows,
                BLOCK_SIZE_M=ctx.block_m, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4,
                # S10 (tune_gemm2): s1 wins on big/mid shapes, deep-K keeps s2
                num_stages=1 if (CR > 512 and K < 8192) else 2)

            if component == "full" and R > 1:
                # slot-reuse gate: every partner acked the previous occupant
                # (all-local GTE waits; acks arrive as 4 B engine writes).
                for q in partners:
                    _wait_ticket(ctx.ack_ptr + (q * S + slot) * 4,
                                 prev_free, sst, mode)
                for q in partners:
                    # push the CURRENT accumulator snapshot to q's slot
                    dst = ctx.peer_scatter[q] + ((slot * R + rank)
                                                 * CR * N) * ctx.elemsz
                    memcpy_async(dst, c_panel.data_ptr(),
                                 rows * N * ctx.elemsz, sst.cuda_stream)
                    # arrival = 4 B engine write of this task's ticket
                    memcpy_async(ctx.peer_arrival[q] + (rank * S + slot) * 4,
                                 tickets_va + 4 * ticket, 4, sst.cuda_stream)
                    _wait_ticket(ctx.arrival_ptr + (q * S + slot) * 4,
                                 ticket, sst, mode)
                    src = ctx.scatter_ptr + (slot * R + q) * CR * N * ctx.elemsz
                    kernel_accumulate_panel_rd[
                        (triton.cdiv(rows, 128) * triton.cdiv(N, 128),)](
                        src, c_panel, rows, N,
                        c_panel.stride(0), c_panel.stride(1),
                        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, num_warps=8)
                # release: ack every partner (I consumed their pushes into
                # this slot, and they hold my snapshots) for this ticket.
                for q in partners:
                    memcpy_async(ctx.peer_ack[q] + (rank * S + slot) * 4,
                                 tickets_va + 4 * ticket, 4, sst.cuda_stream)

    # join: the caller orders after every slot stream (also covers c's
    # cross-stream lifetime for the caching allocator).
    for si, sst in enumerate(ctx.slot_streams):
        ev = ctx._events[1 + si]
        assert h.hipEventRecord(ev, ctypes.c_void_p(sst.cuda_stream)) == 0
        assert h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream), ev,
                                    0) == 0
    return c


def gemm_ar_rd(a, b, ctx):
    """GEMM + AllReduce (TP column-parallel linear): every rank returns the
    full c = sum_r a_r @ b_r^T. a: [M, K_local]; b: [N, K_local]."""
    if ctx is None:
        raise RuntimeError("requires ctx")
    c = torch.empty((a.shape[0], b.shape[0]), dtype=ctx.output_dtype,
                    device=a.device)
    gemm_ar_rd_op(a, b, c, ctx)
    return c


__all__ = [
    "GemmARRDContext",
    "create_gemm_ar_rd_context",
    "finalize_gemm_ar_rd_context",
    "gemm_ar_rd_op",
    "gemm_ar_rd",
    "kernel_gemm_panel_rd",
    "kernel_accumulate_panel_rd",
    "kernel_wait_flag_ge",
    "kernel_flag_xchg",
    "_launch_flag_xchg",
    "_wait_ticket",
    "_wait_value_gte",
    "rd_wait_mode",
]
