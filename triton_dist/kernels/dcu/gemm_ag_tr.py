"""S14: tile-ready chunk-major PULL allgather-GEMM (paper #1 AG design) on
DCU planes.

C[M, N_pr] = A_full[M, K] @ b[N_pr, K] on every rank, where A_full gathers
every rank's shard a_r [M_pr, K]. Replaces the S6 AG op's PUSH producer +
host-putmem signal transport with the paper's consumer-side PULL and
ALL-LOCAL readiness flags:

- consumer-side PULL: each rank's own comm lanes engine-copy the chunks it
  needs OUT of the sources' symmetric workspaces into its local one. All
  remote touches are engine reads; every flag the CONSUMER spins on lives
  in its own memory and is written by its own lane stream, stream-ordered
  after the data -- the S5.3 signal-transport cliff class is structurally
  gone (no host putmem, no remote flag write on the data path).
- the source's shard must be symmetric-visible: torch tensors are not, so
  each op first CHUNK-STAGES the local a into the gather workspace's own
  shard region on a self stream (engine own->own copies, ~385 GB/s probe)
  and fans out a 4 B ARRIVE ticket per (peer, chunk) -- the S12 arrival
  plane. Pulls for (src, ck) gate on that arrive at the PULLER's heap.
- chunk-major order (the paper's tile-ready frontier): chunk ck of every
  source before chunk ck+1 of any -- both the stage fan-out and the pull
  enqueue run chunk-major, so the consumer's first tiles across all ranks
  turn ready earliest.
- super-chunk sizing: chunk_rows = max(m_pr/2, 1024) aligned 256, capped so
  one chunk's bytes stay ~<= 8 MB (the gemm_ar_rd panel cap applied to the
  K-wide row); skip-local split: own-rank tiles read the local a directly
  (never staged-waited), the source's stage copy exists purely for peers.
- slot-reuse guard: the one unavoidable cross-round race is a source's
  round t+1 stage copy overwriting the ws rows a LAGGING puller of round t
  still wants. Round-granularity ACKs: after my consumer kernel completes,
  my consumer stream writes a done ticket into every source's ack row;
  each source's next stage waits all acks >= its last done ticket (the
  S12 ack plane -- ticketed engine writes + GTE stream-waits, no barrier
  op anywhere in the design).
- consumer GEMM: ONE full-M launch (the S13 lesson: per-segment launches
  under-fill the CUs and per-segment streams hit S11-class contention),
  S6 consumer tile geometry verbatim; per-tile hybrid spin (volatile
  batches + rare atomic probe at the L2 coherence point) on LOCAL int32
  flags carrying monotonic per-chunk tickets.

Flag layout (int32, zeroed once at create; tickets never reset):
  ws      symmetric [max_M, K] elems -- gather workspace AND stage target
  arrive  symmetric [R, n_chunks_max] at MY heap, row = source rank:
          src signals "my stage chunk ck is pullable" (4 B engine write)
  ack     symmetric [R] at MY heap, row = puller rank:
          puller signals "my round-t consumer is done" (4 B engine write)
  ready   LOCAL [R, n_chunks_max]: my lane's pull of (src, ck) landed --
          written by MY lane stream, spun by MY consumer kernel

Ticket math: ticket(ck) = round_base + ck + 1; round_base advances by
(n_chunks_max + 1) per full op; the consumer's expected value for global
chunk g = src * n_chunks + ck is round_base + (g % n_chunks) + 1.
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
    _launch_flag_xchg,
    _panel_rows_cap,
    _round_up,
    _wait_ticket,
)

# consumer tile geometry (S5/S10 sweep -- all_gather_gemm consumer config)
_CONSUMER_BLOCK_M = 128
_CONSUMER_BLOCK_N = 128
_CONSUMER_BLOCK_K = 64
_CONSUMER_GROUP_M = 8
_CONSUMER_WARPS = 4


def _auto_chunk_rows(m_per_rank, target_chunks_per_rank=2, min_chunk_rows=1024,
                     align=256):
    rows = triton.cdiv(m_per_rank, target_chunks_per_rank)
    rows = max(rows, min_chunk_rows)
    rows = min(rows, m_per_rank)
    rows = _round_up(rows, align)
    return min(rows, m_per_rank)


@triton.jit(do_not_specialize=["flags_base", "round", "rank", "world_size",
                               "m_chunk_num"])
def kernel_gemm_ag_tr_consumer(
        A,  # gathered workspace [max_M, K] (int64 VA, symmetric heap)
        localA,  # local A shard [M_per_rank, K] tensor
        B,  # [N_per_rank, K] tensor
        C,  # [M, N_per_rank] tensor
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_lam,
        stride_lak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        rank,
        world_size,
        m_chunk_num,  # runtime chunk count (M may be < max_M)
        flags_base,  # LOCAL int32 [num_ranks * m_chunk_num] ready tickets
        round,  # this op's round_base: expected = round + ck + 1
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        M_PER_CHUNK: tl.constexpr,
        NUM_SMS: tl.constexpr,  # == launched grid (loop step)
        EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    M_per_rank = M // world_size
    pid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    acc_dtype = tl.float32
    A_ptr = A.to(tl.pointer_type(localA.dtype.element_ty))
    flag_ptr = flags_base.to(tl.pointer_type(tl.int32))

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # rank swizzle (upstream): visit my own (already-local) rows first
        num_pid_m_per_copy_chunk = M_PER_CHUNK // BLOCK_SIZE_M
        chunk_offset = pid_m // (num_pid_m_per_copy_chunk * world_size)
        rank_offset = pid_m % (num_pid_m_per_copy_chunk * world_size) // num_pid_m_per_copy_chunk
        block_offset = pid_m % num_pid_m_per_copy_chunk

        rank_offset = (rank_offset + rank) % world_size
        pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_CHUNK +
                 block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M

        offs_am = pid_m * BLOCK_SIZE_M
        # global chunk id == source_rank * m_chunk_num + chunk: exactly the
        # slot my pull lane flags after the data landed (LOCAL memory)
        offs_sig = offs_am // M_PER_CHUNK
        offs_rank = pid_m // pid_m_per_rank

        if offs_rank != rank:
            # per-chunk ticket: expected = round_base + ck + 1 where
            # ck = offs_sig % m_chunk_num (offs_sig = src * nchunks + ck).
            # Hybrid spin on the LOCAL flag (volatile batches as delay,
            # rare atomic probe at the L2 coherence point for forward
            # progress -- the all_gather_gemm consumer plane; here the
            # flag is written by MY OWN lane's engine op, so it is fully
            # coherent at L2 for the atomic probe).
            expected = round + 1 + offs_sig % m_chunk_num
            fp = flag_ptr + offs_sig
            v = tl.atomic_add(fp, 0)
            while v < expected:
                for _ in range(128):
                    v = tl.load(fp, volatile=True)
                if v < expected:
                    v = tl.atomic_add(fp, 0)

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        if offs_rank == rank:
            rm = rm % M_per_rank
            A_BASE = localA + rm[:, None] * stride_lam + rk[None, :] * stride_lak
            a_kstride = stride_lak
        else:
            A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            a_kstride = stride_ak

        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * a_kstride
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            if offs_rank == rank:  # rm is already the local-shard mapping
                A_BASE = localA + rm[:, None] * stride_lam + rk[None, :] * stride_lak
            else:
                A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.dtype.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


@dataclasses.dataclass
class GemmAGTRContext:
    rank: int
    num_ranks: int
    max_M: int
    K: int
    chunk_rows: int
    num_chunks: int  # at max_M
    comm_lanes: int
    elemsz: int
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    ws_ptr: int  # symmetric [max_M, K] elems (stage target + gather)
    arrive_ptr: int  # symmetric int32 [R, num_chunks] (MY heap view)
    ack_ptr: int  # symmetric int32 [R] (MY heap view)
    peer_ws: List[int]
    peer_arrive: List[int]
    peer_ack: List[int]
    ready: torch.Tensor  # LOCAL int32 [R, num_chunks]
    tickets: torch.Tensor
    stage_stream: torch.cuda.Stream
    lane_streams: List[torch.cuda.Stream]
    consumer_stream: torch.cuda.Stream
    round_base: int = 0
    round_stride: int = 0
    last_done_ticket: int = 0
    wait_mode: str = "stream"
    _events: Optional[list] = None


def create_gemm_ag_tr_context(max_M, K, input_dtype, output_dtype, rank,
                              num_ranks, mpi, comm, chunk_rows=0,
                              comm_lanes=2, wait_mode=None):
    """Symmetric gather ws + arrive/ack planes + local ready flags + the
    stage/lane/consumer stream set. mpi/comm from pydushmem.host.mpi_init()
    (BEFORE `import torch`)."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert max_M % num_ranks == 0, "gather needs M % R == 0"
    m_per_rank = max_M // num_ranks
    assert m_per_rank % 64 == 0 and K % 16 == 0
    assert input_dtype == output_dtype

    elemsz = torch.empty((), dtype=input_dtype).element_size()
    if chunk_rows <= 0:
        chunk_rows = _auto_chunk_rows(m_per_rank)
        # cap one chunk's pull/stage bytes (K-wide rows) near ~8 MB, then
        # walk down to a divisor of m_per_rank (the flag/ticket math needs
        # chunk-aligned segments; e.g. K=12288 caps at 384 rows, which does
        # not divide a 4096-row shard)
        chunk_rows = min(chunk_rows, _panel_rows_cap(K, elemsz))
        while m_per_rank % chunk_rows:
            chunk_rows -= 64
        assert chunk_rows >= 64, "no 64-aligned chunk divides m_per_rank"
    num_chunks = triton.cdiv(m_per_rank, chunk_rows)
    comm_lanes = max(1, min(comm_lanes, num_ranks))

    s = dcu_state()
    ws_ptr = s["host"].malloc(max_M * K * elemsz)
    arrive_bytes = 4 * num_ranks * num_chunks
    arrive_ptr = s["host"].malloc(arrive_bytes)
    ack_ptr = s["host"].malloc(4 * num_ranks)
    # flags gate reads of possibly-unwritten memory: garbage >= ticket would
    # skip a wait (engine-written peers update our L2 lines -- hipMemset of
    # OWN memory is safe, the all_gather_gemm plane)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(arrive_ptr), 0, arrive_bytes) == 0
    assert h.hipMemset(ctypes.c_void_p(ack_ptr), 0, 4 * num_ranks) == 0

    host = s["host"]
    peer_ws = [host.ptr(ws_ptr, d) for d in range(num_ranks)]
    peer_arrive = [host.ptr(arrive_ptr, d) for d in range(num_ranks)]
    peer_ack = [host.ptr(ack_ptr, d) for d in range(num_ranks)]

    ready = torch.zeros(num_ranks, num_chunks, dtype=torch.int32,
                        device="cuda")
    tickets = torch.arange(MAX_TICKETS, dtype=torch.int32, device="cuda")

    h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    h.hipEventCreateWithFlags.restype = ctypes.c_int
    h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipEventRecord.restype = ctypes.c_int
    h.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_uint64]
    h.hipStreamWaitEvent.restype = ctypes.c_int
    n_streams = 2 + comm_lanes  # stage + consumer + lanes
    events = []
    for _ in range(1 + n_streams):
        ev = ctypes.c_void_p()
        rc = h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2)
        assert rc == 0, f"hipEventCreateWithFlags rc={rc}"
        events.append(ev)

    ctx = GemmAGTRContext(
        rank=rank, num_ranks=num_ranks, max_M=max_M, K=K,
        chunk_rows=chunk_rows, num_chunks=num_chunks, comm_lanes=comm_lanes,
        elemsz=elemsz, input_dtype=input_dtype, output_dtype=output_dtype,
        ws_ptr=ws_ptr, arrive_ptr=arrive_ptr, ack_ptr=ack_ptr,
        peer_ws=peer_ws, peer_arrive=peer_arrive, peer_ack=peer_ack,
        ready=ready, tickets=tickets,
        stage_stream=torch.cuda.Stream(),
        lane_streams=[torch.cuda.Stream() for _ in range(comm_lanes)],
        consumer_stream=torch.cuda.Stream(),
        round_stride=num_chunks + 1,
        wait_mode=wait_mode or os.environ.get("RD_WAIT_MODE", "stream"),
        _events=events,
    )

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ctx


def finalize_gemm_ag_tr_context(ctx):
    s = dcu_state()
    for p in (ctx.ws_ptr, ctx.arrive_ptr, ctx.ack_ptr):
        if p:
            s["host"].free(p)
    ctx.ws_ptr = ctx.arrive_ptr = ctx.ack_ptr = 0


def gemm_ag_tr_op(a, b, c, ctx, component="full"):
    """C[M, N_pr] = gather_r(a_r) @ b on every rank via the tile-ready
    chunk-major PULL pipeline. Everything is enqueued up front; the
    stage/lane/consumer streams self-pace on tickets. component="gemm":
    consumer GEMM only, against the workspace as the last full op left it
    (benchmarking decomposition; tickets NOT advanced)."""
    m_pr, K = a.shape
    N_pr = b.shape[0]
    rank, R = ctx.rank, ctx.num_ranks
    # runtime M may be smaller than the ctx's chunk: clamp (flag/ticket
    # indices only ever use ck < runtime n_chunks <= ctx.num_chunks)
    CR = min(ctx.chunk_rows, m_pr)
    L = ctx.comm_lanes
    M = m_pr * R
    assert K == ctx.K and M <= ctx.max_M
    assert a.dtype == ctx.input_dtype and b.dtype == ctx.input_dtype
    assert a.stride(1) == 1 and b.stride(1) == 1 and a.stride(0) == K
    assert b.stride(0) == K
    assert M % _CONSUMER_BLOCK_M == 0 and N_pr % _CONSUMER_BLOCK_N == 0
    assert m_pr % CR == 0

    cur = torch.cuda.current_stream()
    h = hip()
    n_chunks = m_pr // CR
    mode = ctx.wait_mode

    if component == "full":
        ctx.round_base += ctx.round_stride
        assert ctx.round_base + n_chunks < MAX_TICKETS
    tickets_va = ctx.tickets.data_ptr()

    def ticket(ck):
        return ctx.round_base + ck + 1

    # fork: stage + lanes + consumer order after the caller (a/b/c ready)
    all_streams = ([ctx.stage_stream] + ctx.lane_streams
                   + [ctx.consumer_stream])
    ev_fork = ctx._events[0]
    assert h.hipEventRecord(ev_fork, ctypes.c_void_p(cur.cuda_stream)) == 0
    for st in all_streams:
        assert h.hipStreamWaitEvent(ctypes.c_void_p(st.cuda_stream),
                                    ev_fork, 0) == 0

    my_off = rank * m_pr * K * ctx.elemsz

    if component == "full" and R > 1:
        # stage: chunk-major, own->own engine copies into MY ws shard region,
        # each chunk's ARRIVE fanned out as 4 B engine writes to every peer.
        # Round-granularity slot guard: every puller acked my last round
        # before I overwrite the ws rows its lagging pull may still want.
        with torch.cuda.stream(ctx.stage_stream):
            for d in range(R):
                if d != rank:
                    _wait_ticket(ctx.ack_ptr + d * 4,
                                 ctx.last_done_ticket,
                                 ctx.stage_stream, mode)
            for ck in range(n_chunks):
                memcpy_async(ctx.ws_ptr + my_off + ck * CR * K * ctx.elemsz,
                             a.data_ptr() + ck * CR * K * ctx.elemsz,
                             CR * K * ctx.elemsz, ctx.stage_stream.cuda_stream)
                for d in range(R):
                    if d != rank:
                        # kernel write (tiny 4B engine memcpys wedge the
                        # DTK runtime's enqueue nondeterministically at
                        # final-shape queue depths -- see kernel_flag_xchg)
                        _launch_flag_xchg(
                            ctx.peer_arrive[d]
                            + (rank * ctx.num_chunks + ck) * 4,
                            ticket(ck))

        # pulls: chunk-major -- chunk ck of EVERY source, then chunk ck+1.
        # Lanes rotate (src + ck) % L; the LOCAL ready flag rides the same
        # lane stream right after the data (stream order == data order).
        for ck in range(n_chunks):
            for step in range(1, R):
                src = (rank + step) % R
                lane = ctx.lane_streams[(src + ck) % L]
                # every rank stages its shard at the SAME global offset in
                # its own ws, so the pull is offset-preserving: the source
                # region I want (src's shard) sits at my destination offset
                src_off = (src * m_pr + ck * CR) * K * ctx.elemsz
                with torch.cuda.stream(lane):
                    _wait_ticket(ctx.arrive_ptr + (src * ctx.num_chunks + ck) * 4,
                                 ticket(ck), lane, mode)
                    memcpy_async(ctx.ws_ptr + src_off,
                                 ctx.peer_ws[src] + src_off,
                                 CR * K * ctx.elemsz, lane.cuda_stream)
                    # local ready rides the lane stream right after the
                    # data (kernel write; launch boundary retires the pull)
                    _launch_flag_xchg(
                        ctx.ready.data_ptr()
                        + (src * ctx.num_chunks + ck) * 4,
                        ticket(ck))

    # consumer: ONE full-M launch; per-tile spin on LOCAL ready tickets,
    # own-rank tiles read the local shard directly (skip-local split)
    with torch.cuda.stream(ctx.consumer_stream):
        tiles = (triton.cdiv(M, _CONSUMER_BLOCK_M)
                 * triton.cdiv(N_pr, _CONSUMER_BLOCK_N))
        kernel_gemm_ag_tr_consumer[(tiles,)](
            ctx.ws_ptr,
            a,
            b,
            c,
            M,
            N_pr,
            K,
            K, 1,  # gathered workspace [max_M, K] contiguous
            a.stride(0), a.stride(1),  # localA (a) strides
            b.stride(1), b.stride(0),  # b is [N, K]
            c.stride(0), c.stride(1),
            rank,
            R,
            n_chunks,
            ctx.ready.data_ptr(),
            ctx.round_base,
            BLOCK_SIZE_M=_CONSUMER_BLOCK_M,
            BLOCK_SIZE_N=_CONSUMER_BLOCK_N,
            BLOCK_SIZE_K=_CONSUMER_BLOCK_K,
            GROUP_SIZE_M=_CONSUMER_GROUP_M,
            M_PER_CHUNK=CR,
            NUM_SMS=tiles,
            EVEN_K=(K % _CONSUMER_BLOCK_K == 0),
            num_warps=_CONSUMER_WARPS,
            # S10 rule WITH the deep-K guard (gemm_ar_rd form): s1-on-deepK
            # is catastrophic (M8192/K8192 measured 45ms vs ~22 at s2)
            num_stages=1 if (M > 512 and K < 8192) else 2,
        )
        if component == "full" and R > 1:
            # release: my consumer is done with everyone's staged rows --
            # every source may overwrite its shard region next round
            done = ticket(n_chunks - 1)
            for d in range(R):
                if d != rank:
                    _launch_flag_xchg(ctx.peer_ack[d] + rank * 4, done)
            ctx.last_done_ticket = done

    # join: the caller orders after every op stream (also covers a/b/c
    # cross-stream lifetime for the caching allocator)
    for si, st in enumerate(all_streams):
        ev = ctx._events[1 + si]
        assert h.hipEventRecord(ev, ctypes.c_void_p(st.cuda_stream)) == 0
        assert h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream), ev,
                                    0) == 0
    a.record_stream(ctx.stage_stream)
    a.record_stream(ctx.lane_streams[0])
    b.record_stream(ctx.consumer_stream)
    c.record_stream(ctx.consumer_stream)
    return c


def gemm_ag_tr(a, b, ctx):
    """Allgather global matrix A and matmul with local matrix B.
    a: [M_per_rank, K]; b: [N_per_rank, K]; returns c: [M, N_per_rank]."""
    if ctx is None:
        raise RuntimeError("requires ctx")
    c = torch.empty(ctx.num_ranks * a.shape[0], b.shape[0],
                    dtype=ctx.output_dtype, device=a.device)
    gemm_ag_tr_op(a, b, c, ctx)
    return c


__all__ = [
    "GemmAGTRContext",
    "create_gemm_ag_tr_context",
    "finalize_gemm_ag_tr_context",
    "gemm_ag_tr_op",
    "gemm_ag_tr",
    "kernel_gemm_ag_tr_consumer",
]
