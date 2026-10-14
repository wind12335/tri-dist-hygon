"""DCU port of triton_dist/kernels/amd/all_gather_gemm.py.

Same architecture as upstream (producer pushes the local A shard into every
peer's gathered workspace; a persistent GEMM consumes chunks as they land,
reading the local shard directly), on the validated DCU planes:

- producer: HOST-side per-dest chunked hipMemcpyAsync into the peer's
  workspace (the peer VA from host.ptr), then one tiny triton kernel per
  dest issuing the dushmem uint64 flag atomics -- stream order guarantees
  the signals land only after that dest's data copies completed. This
  mirrors upstream's copy-engine producer: the S4 transport probe showed
  every DEVICE-initiated dushmem transfer on this stack runs at ~19 MB/s
  (putmem_signal/putmem_nbi, any chunking, proxy env on or off), while
  engine memcpys into the peer VA run at ~110 GB/s and stay L2-coherent
  for the consumer's plain loads (probe_memcpy_freshness).
- flags: symmetric-heap uint64 per (SOURCE rank, chunk), indexed by the
  GLOBAL chunk id `row // M_PER_CHUNK` (upstream: barrier_buffers[remote]
  [rank * chunk_num + c]). Written with the MONOTONIC op round via
  DUSHMEM_SIGNAL_SET -- never reset (upstream zeroes + waits ==1 each op;
  monotonic rounds make the reset unnecessary).
- consumer wait: spin on `tl.atomic_add(ptr, 0)` over the LOW int32 word of
  the u64 flag. Atomics operate at the L2 coherence point; plain/volatile
  loads can hit a stale L1/DRAM line and deadlock (see common_ops.SymmBarrier).
- own-rank tiles read localA directly (upstream behavior); no self-push,
  so the workspace's own slot is never written (nor read).
- op start: SymmBarrier -- a peer's next-round push must not overwrite a
  workspace slot that my still-running consumer is reading. (Upstream's op
  has no such guard; the monotonic flags alone do not protect the previous
  round's DATA.)
- SM budget: the producer is HOST-orchestrated engine memcpys (no SMs) --
  only the tiny staging fill needs one briefly, before the consumer is
  submitted. The consumer therefore launches one block per tile (grid ==
  tiles: the persistent loop degenerates to one iteration per block). The
  old n_sms-2 reservation dates from the SM-kernel producer era and cost
  ~2.3x on large shapes (S5 sweep: plain tile launch 0.323 ms vs capped
  persistent consumer 0.75 ms at M=4096,N=2048,K=2048).
- strides: b layout matches upstream ([N, K], stride_bk=b.stride(1)).
"""
import ctypes
import dataclasses
from typing import Optional

import torch
import triton
import triton.language as tl

from pydushmem import device as dushmem_dev
from triton_dist.kernels.dcu.common_ops import (
    SymmBarrier,
    dcu_state,
    fan_out_memcpy_async,
    hip,
    init_dcu,
    memcpy_async,
)

# Consumer GEMM tile config (S5 sweep, bench/results/tune_gemm.json):
# BLOCK_K=64/warps=4 beats the old 32/8 everywhere (+7% large shapes,
# +40% small-M / deep-K); best plain-kernel config on every shape.
_CONSUMER_BLOCK_M = 128
_CONSUMER_BLOCK_N = 128
_CONSUMER_BLOCK_K = 64
_CONSUMER_GROUP_M = 8
_CONSUMER_WARPS = 4

# Push signal transport: "host" (default) | "engine"
# "engine" (one engine memcpy of the staged flags, stream-ordered after the
# data) hits a ~1.6 s cliff on 1-4% of ops when chunks are small (MPC<=256,
# i.e. >=8 engine ops per dest; s5_signal_probe) -- the tiny flag write
# queues behind the engine's processing of the multi-MB data copies and the
# consumer's flag spin stalls for the whole drain. "host" event-syncs the
# data copies and writes the flags via host dushmem putmem (RMA path,
# updates the dest L2 directly): 0 stalls at every chunk size, max 2-3 ms,
# at the cost of one host event-sync (~+0.05 ms) per op.
SIGNAL_MODE = "host"


@triton.jit(do_not_specialize=["flags_base", "round", "my_rank", "dest",
                               "m_chunk_num"])
def kernel_ag_signal(
        flags_base,  # symmetric u64 flags [num_ranks, m_chunk_num] (int64 VA)
        round,  # monotonic op round (never reset)
        my_rank,
        dest,  # PE whose (my_rank, c) slots get set
        m_chunk_num,
):
    """UNUSED (kept for reference): device-initiated dushmem signal --
    occasionally stalls for tens of ms on this stack (S4 bench outliers).
    AGPushKernel.launch writes the flags with one engine memcpy instead
    (same L2 coherence the consumer's atomic spin already relies on)."""
    for c in range(m_chunk_num):
        dushmem_dev.dushmem_uint64_atomic_set(
            flags_base + (my_rank * m_chunk_num + c).to(tl.int64) * 8,
            round, dest)


@triton.jit
def kernel_gemm_ag_consumer(
        A,  # gathered workspace [max_M, K] (int64 VA)
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
        flags_base,  # int64 VA, u64 [num_ranks, m_chunk_num] (own flags)
        round,
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
    # remote signal is a u64; spin on its LOW int32 word (round < 2^31)
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
        # slot the source's producer signals
        offs_sig = offs_am // M_PER_CHUNK
        offs_rank = pid_m // pid_m_per_rank

        if offs_rank != rank:
            # wait for this source chunk's flag. The signal is an ENGINE
            # write that updates this PE's L2 -- but a tight atomic_add(0)
            # RMW spin keeps the line perpetually busy/dirty and can block
            # the incoming update for tens of ms (up to ~1 s; S4 outlier
            # probes: stalls land in the consumer phase only, after the
            # push phase completed). Hybrid spin: a batch of volatile
            # loads (pure delay + optimistic fast path; alone they can
            # read stale DRAM forever) interleaved with a rare atomic
            # probe at the L2 coherence point, which guarantees forward
            # progress while leaving us-wide gaps for the engine write.
            fp = flag_ptr + offs_sig * 2
            v = tl.atomic_add(fp, 0)
            while v < round:
                for _ in range(128):
                    v = tl.load(fp, volatile=True)
                if v < round:
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


class AGPushKernel:
    """Host-side producer (upstream-style copy-engine orchestration).

    Per dest (nearest-peer order, self skipped), all on the CURRENT torch
    stream (call inside `with torch.cuda.stream(ag_stream):`):
      1. chunked hipMemcpyAsync of my A shard into the peer's workspace
         slot (~110 GB/s, L2-coherent; device-initiated dushmem transfers
         are ~19 MB/s on this stack -- see common_ops.memcpy_async),
      2. flag write for all of dest's chunks at once. Default transport
         ("host", SIGNAL_MODE): event-sync the data copies, then host
         dushmem putmem of the staged u64 round values into the peer's
         flag slots (RMA path -- the peer's L2 updates directly; the
         consumer's atomic spin reads it at the L2 coherence point).
         "engine" (one stream-ordered engine memcpy) is kept for A/B: it
         hits a ~1.6 s cliff on 1-4% of ops at small chunk sizes (see
         SIGNAL_MODE above / bench/s5_signal_probe.py)."""

    def __init__(self, a_base, ws_base, flags_base, num_ranks, K,
                 M_per_rank, M_PER_CHUNK, elemsz):
        s = dcu_state()
        self._num_ranks = num_ranks
        self._K = K
        self._M_PER_CHUNK = M_PER_CHUNK
        self._elemsz = elemsz
        self._chunk_bytes = M_PER_CHUNK * K * elemsz
        self._peer_ws = [s["host"].ptr(ws_base, d) for d in range(num_ranks)]
        self._peer_flags = [s["host"].ptr(flags_base, d)
                            for d in range(num_ranks)]
        self._stage = torch.empty(M_per_rank // M_PER_CHUNK,
                                  dtype=torch.int64, device="cuda")
        self._sig_registered = False

    def launch(self, a_base, ws_base, flags_base, round, my_rank, M_per_rank):
        assert M_per_rank % self._M_PER_CHUNK == 0
        m_chunk_num = M_per_rank // self._M_PER_CHUNK
        shard_off = my_rank * M_per_rank * self._K * self._elemsz
        stream = torch.cuda.current_stream().cuda_stream
        self._stage.fill_(round)  # device fill, ordered on `stream`
        dests = [d for i in range(self._num_ranks)
                 for d in [(my_rank + i) % self._num_ranks]
                 if d != my_rank]  # nearest peers first
        ops = []
        for d in dests:
            dst0 = self._peer_ws[d] + shard_off
            for c in range(m_chunk_num):
                ops.append((dst0 + c * self._chunk_bytes,
                            a_base + c * self._chunk_bytes,
                            self._chunk_bytes))
        # S11: fan the (dest, chunk) copies out across the push-stream pool
        # (upstream cp_engine_*_multi_stream idea) -- forked from/joined to
        # the current (ag) stream, so both signal transports below stay
        # ordered behind ALL of them. No-op at 2 cards (<=4 ops of <=2MB);
        # the payoff is the 8-card (R-1)-dest fan-out.
        fan_out_memcpy_async(ops)
        # Signal transport (S5 tail-latency experiment; the flag write must
        # reach the peer AFTER its data copies). "engine" keeps the S4
        # default: one stream-ordered engine memcpy per dest. The observed
        # ~30 ms consumer tail (push phase long done -> the engine WRITE of
        # the flag is not yet VISIBLE at the peer) suggests the flag write
        # queues behind the engine's processing of the multi-MB data copies.
        # The alternates move only the tiny flag write off that path:
        #  "host":   event-sync the data copies, then host dushmem putmem
        #            of the staged flags (RMA path, updates dest L2)
        #  "device": stream-ordered kernel doing dushmem atomic_set per
        #            chunk (m_chunk_num * 8 B rides the ~19 MB/s slow path
        #            in ~microseconds)
        if SIGNAL_MODE == "engine":
            for d in dests:
                # fires all of dest d's chunk flags at once so the consumer
                # can start this dest while later dests are still in flight
                memcpy_async(self._peer_flags[d] + my_rank * m_chunk_num * 8,
                             self._stage.data_ptr(), m_chunk_num * 8, stream)
        elif SIGNAL_MODE == "host":
            ev = torch.cuda.Event()
            ev.record()
            ev.synchronize()  # data engine ops complete at the source
            host = dcu_state()["host"]
            for d in dests:
                host.putmem(flags_base + my_rank * m_chunk_num * 8,
                            self._stage.data_ptr(), m_chunk_num * 8, d)
        else:
            raise ValueError(f"unknown SIGNAL_MODE {SIGNAL_MODE!r}")


@dataclasses.dataclass
class AllGatherGEMMTensorParallelContext:
    rank: int
    num_ranks: int
    max_M: int
    K: int
    ws_ptr: int  # symmetric [max_M, K]
    flags_ptr: int  # symmetric u64 [num_ranks, m_chunk_num]
    barrier: SymmBarrier
    push: AGPushKernel
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    M_PER_CHUNK: int = 1024
    flag_round: int = 0
    ag_stream: Optional[torch.cuda.streams.Stream] = None
    gemm_stream: Optional[torch.cuda.streams.Stream] = None


def create_ag_gemm_intra_node_context(max_M, K, input_dtype, output_dtype,
                                      rank, num_ranks, mpi, comm,
                                      M_PER_CHUNK=1024, ag_stream=None,
                                      gemm_stream=None):
    """Workspace (symmetric heap) + flags + barrier for ag-gemm.

    mpi/comm from pydushmem.host.mpi_init() (BEFORE `import torch`)."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert max_M % num_ranks == 0
    M_per_rank = max_M // num_ranks
    assert M_per_rank % M_PER_CHUNK == 0
    assert M_PER_CHUNK % 128 == 0
    m_chunk_num = M_per_rank // M_PER_CHUNK

    s = dcu_state()
    elemsz = torch.empty((), dtype=input_dtype).element_size()
    ws_ptr = s["host"].malloc(max_M * K * elemsz)
    flags_bytes = 8 * num_ranks * m_chunk_num
    flags_ptr = s["host"].malloc(flags_bytes)
    # flags gate reads of a possibly-unwritten workspace: garbage >= round
    # would skip the wait. (Plain hipMemset of OWN memory is safe here --
    # peers write through dushmem RMA, which updates our L2 lines.)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(flags_ptr), 0, flags_bytes) == 0

    barrier = SymmBarrier(num_ranks)
    push = AGPushKernel(ws_ptr, ws_ptr, flags_ptr, num_ranks, K, M_per_rank,
                        M_PER_CHUNK, elemsz)

    current_stream = torch.cuda.current_stream()
    _ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    _gemm_stream = current_stream if gemm_stream is None else gemm_stream

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0

    return AllGatherGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        max_M=max_M,
        K=K,
        ws_ptr=ws_ptr,
        flags_ptr=flags_ptr,
        barrier=barrier,
        push=push,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        M_PER_CHUNK=M_PER_CHUNK,
        ag_stream=_ag_stream,
        gemm_stream=_gemm_stream,
    )


def ag_gemm_intra_node_op(a, b, c, ctx, skip_comm=False):
    """allgather A (each rank pushes its shard) then GEMM with local b.

    skip_comm=True: benchmarking mode -- consumer only, against the
    workspace as left by the last real op (flag_round is NOT incremented,
    so the flag spins never wait). GEMM-only component of the op
    decomposition; requires a real op to have run on this ctx first."""
    M_per_rank, K = a.shape
    N_per_rank = b.shape[0]
    num_ranks = ctx.num_ranks
    rank = ctx.rank
    M = M_per_rank * num_ranks
    assert K == ctx.K and M <= ctx.max_M
    assert a.dtype == ctx.input_dtype and b.dtype == ctx.input_dtype
    # layouts assumed by the in-kernel multiple_of hints, localA mapping,
    # and the raw-byte producer memcpys
    assert a.stride(1) == 1 and b.stride(1) == 1 and a.stride(0) == K
    assert M % _CONSUMER_BLOCK_M == 0 and N_per_rank % _CONSUMER_BLOCK_N == 0
    assert K % 16 == 0  # multiple_of hints; non-multiple-of-BLOCK_K uses the tail
    assert M_per_rank % ctx.M_PER_CHUNK == 0

    current_stream = torch.cuda.current_stream()
    if not skip_comm:
        ctx.flag_round += 1
        # my previous op's kernels are done, then every rank's are (a peer
        # must not overwrite a workspace slot my still-running consumer is
        # reading)
        current_stream.wait_stream(ctx.gemm_stream)
        current_stream.wait_stream(ctx.ag_stream)
        ctx.barrier.round_barrier(rank)

        # producer first: submission order + 2 reserved CUs keep the push
        # resident even while consumer blocks spin on un-landed chunks
        if ctx.ag_stream is not current_stream:
            ctx.ag_stream.wait_stream(current_stream)
        with torch.cuda.stream(ctx.ag_stream):
            ctx.push.launch(a.data_ptr(), ctx.ws_ptr, ctx.flags_ptr,
                            ctx.flag_round, rank, M_per_rank)

        # consumer on an INDEPENDENT stream (does NOT wait for the push):
        # the per-chunk flag spins provide the sync -- this is the overlap
        if ctx.gemm_stream is not current_stream:
            ctx.gemm_stream.wait_stream(current_stream)
    with torch.cuda.stream(ctx.gemm_stream):
        # one block per tile; the persistent loop degenerates to a single
        # iteration per block (NUM_SMS == grid). Producers are engine
        # memcpys -- no SM reservation needed anymore.
        tiles = (triton.cdiv(M, _CONSUMER_BLOCK_M)
                 * triton.cdiv(N_per_rank, _CONSUMER_BLOCK_N))
        kernel_gemm_ag_consumer[(tiles,)](
            ctx.ws_ptr,
            a,
            b,
            c,  #
            M,
            N_per_rank,
            K,  #
            K, 1,  # gathered workspace [max_M, K] contiguous
            a.stride(0), a.stride(1),  # localA (a) strides
            b.stride(1), b.stride(0),  # b is [N, K]
            c.stride(0), c.stride(1),  #
            rank,
            num_ranks,
            ctx.flags_ptr,
            ctx.flag_round,
            BLOCK_SIZE_M=_CONSUMER_BLOCK_M,
            BLOCK_SIZE_N=_CONSUMER_BLOCK_N,
            BLOCK_SIZE_K=_CONSUMER_BLOCK_K,
            GROUP_SIZE_M=_CONSUMER_GROUP_M,
            M_PER_CHUNK=ctx.M_PER_CHUNK,
            NUM_SMS=tiles,
            EVEN_K=(K % _CONSUMER_BLOCK_K == 0),
            num_warps=_CONSUMER_WARPS,
            # S10 (tune_gemm2 + s4 A/B): s1 wins 6-8% on big/mid shapes,
            # loses on decode (M<=256) -- same M-threshold as the plain
            # kernel in gemm_reduce_scatter.matmul_local. Deep-K guard from
            # the gemm_ar_rd form: s1-on-deepK (K>=8192) is catastrophic.
            num_stages=1 if (M > 512 and K < 8192) else 2,
        )

    current_stream.wait_stream(ctx.gemm_stream)
    # kernels on other streams may outlive this call
    a.record_stream(ctx.ag_stream)
    a.record_stream(ctx.gemm_stream)
    b.record_stream(ctx.gemm_stream)
    c.record_stream(ctx.gemm_stream)
    return c


def ag_gemm_intra_node(a, b, transe_b, ctx):
    """Allgather global matrix A and do matmul with local matrix B.

    a: [M_per_rank, K]; b: [N_per_rank, K]; returns c: [M, N_per_rank].
    """
    if ctx is None:
        raise RuntimeError("requires ctx")
    if transe_b:
        raise NotImplementedError("transpose weight is not yet supported")

    M_per_rank, K = a.shape
    N_per_rank = b.shape[0]
    c = torch.empty([ctx.num_ranks * M_per_rank, N_per_rank],
                    dtype=ctx.output_dtype, device=a.device)
    ag_gemm_intra_node_op(a, b, c, ctx)
    return c
