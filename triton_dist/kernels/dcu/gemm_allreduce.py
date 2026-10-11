"""DCU port of the newer upstream triton_dist/kernels/amd/gemm_allreduce.py
(triton-3.7 era): C = sum_r (a_r @ b_r^T) with the FULL C on every rank
(tensor-parallel column-parallel GEMM + allreduce).

Upstream architecture (amd): a persistent GEMM writes each finished tile
into the rank's OWN symmetric out-buffer and signals ALL ranks per tile
from inside the kernel (st ... semantic="release", scope="system");
owner-rank consumers (tile_id % world_size) dl.wait on every source's tile
signal, sum the remote partials, and fan the result tile out to every
rank's output buffer.

Why that does not transplant to this stack (S6 probes, 2x BW1000):
- the per-tile SIGNALS would be fine: tiny device dushmem atomics DO
  pipeline -- bench/s6_signal_concurrency.py measured ~1.3 us/op at
  512 concurrent CTAs (290 us single-op latency; 29 us/op serialized
  from one CTA).
- but the per-tile RESULT DATA is device-initiated remote traffic, and
  every device-initiated dushmem transfer is a ~19.5 MB/s byte-rate pipe
  regardless of op size or concurrency (32768 tiny atomics = 256 KB ->
  12.8 ms sits exactly on the S4 bulk ceiling). An 8 MB partial fan-out
  would take ~400 ms. Upstream's dl.wait / release-scoped stores are
  compiler-backend extensions (distributed_wait) anyway -- unavailable in
  stock triton 3.3.

This port re-derives the op on the validated S5 planes:
  compute:  per-M-chunk plain GEMM (S5-tuned 128x128x64 / w4 / s2; BM=64
            for chunk heights <= 512) writing MY partial chunk into MY
            symmetric workspace slot [num_ranks, max_M, N] -- local
            stores only, full GEMM speed.
  data:     HOST engine memcpys per (chunk x dest) of my partial chunk
            into every peer's ws slot for me (~110 GB/s, L2-coherent for
            plain loads).
  signal:   the op's whole flags row in ONE blocking host putmem (RMA
            path) after event-syncing the LAST chunk's data copies --
            the all_gather_gemm "host" transport verbatim (S5: zero
            stalls; the S5.3 finding keeps the signal off the engine
            queue). Per-chunk pipelined signals were tried and rejected:
            mutual 8 B blocking puts deadlock, nbi+quiet starves under SM
            load (see ARPushKernel.signal_all history note) -- per-chunk
            signal granularity is future work (device atomics are the
            measured-viable candidate: ~1.3 us/op pipelined).
  reduce:   per-chunk consumer on the caller's stream: hybrid-spins each
            remote source's chunk flag, sums every slot's chunk in fp32,
            writes the final C chunk. Own-slot data is guarded by a
            stream event, not a flag.
  overlap:  chunk c's data push runs under chunk c+1's GEMM (stream-
            ordered per chunk); the reduce starts once the flags row
            lands (after the last push). The only host-blocking step is
            the final event sync + row putmem.

Semantics preserved: full C on every rank, fresh-data correctness, and an
op-start SymmBarrier guarding the workspace against a peer's next-round
push while my consumer still reads it.
"""
import ctypes
import dataclasses
from typing import Optional

import torch
import triton
import triton.language as tl

from triton_dist.kernels.dcu.common_ops import (
    SymmBarrier,
    dcu_state,
    hip,
    init_dcu,
    memcpy_async,
)


@triton.jit(do_not_specialize=["c_base"])
def kernel_gemm_ar_producer(
        a_ptr, b_ptr,
        c_base,  # symmetric-heap VA (int64) of MY ws slot at the chunk's row 0
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """Grouped-tile matmul of ONE M-chunk of the partial product into the
    symmetric workspace.

    Same body as kernel_gemm_rs_producer_local, but the output lives in
    the dushmem symmetric heap (no torch tensor to pass as a pointer), so
    the base arrives as an int64 VA and is cast here."""
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
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
    c = accumulator.to(dtype)
    c_ptr = c_base.to(tl.pointer_type(dtype))

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(do_not_specialize=["ws_base", "flags_base", "round", "my_rank",
                               "total"])
def kernel_ar_consumer_reduce(
        ws_base,  # symmetric-heap VA (int64) of OWN [num_ranks, max_M, N]
        c_ptr,  # [M, N] output tensor
        flags_base,  # symmetric-heap VA (int64) u64 [num_ranks, m_chunk_num]
        round,  # monotonic op round (never reset)
        my_rank,
        total,  # M * N elements of the output
        num_ranks: tl.constexpr,
        m_chunk_num: tl.constexpr,
        SLOT_ELEMS: tl.constexpr,  # max_M * N
        CHUNK_ELEMS: tl.constexpr,  # M_PER_CHUNK * N
        BLOCK_SIZE: tl.constexpr,
):
    """Sum every source's partial in fp32 -> final C, ONE launch over all
    chunks (each block derives its chunk -- and thus its flag column --
    from pid; the spin condition stays scalar).

    Own slot (source == my_rank) is guarded by a stream event, not a
    flag. Remote slots wait that source's chunk flag with the same hybrid
    spin as the AG consumer: a batch of volatile loads as delay/optimistic
    fast path (alone they can read stale DRAM forever), interleaved with a
    rare atomic probe at the L2 coherence point, which guarantees forward
    progress without keeping the flag line perpetually busy."""
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inb = offs < total
    offs = tl.where(inb, offs, 0)
    chunk = pid * BLOCK_SIZE // CHUNK_ELEMS  # this block's flag column

    dtype = c_ptr.dtype.element_ty
    ws = ws_base.to(tl.pointer_type(dtype))
    # flags are u64; the round fits the LOW int32 word (round < 2^31)
    flag_ptr = flags_base.to(tl.pointer_type(tl.int32))

    accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in range(num_ranks):
        s = (i + my_rank) % num_ranks  # own slot first, then rotated peers
        if s != my_rank:
            fp = flag_ptr + (s * m_chunk_num + chunk) * 2
            v = tl.atomic_add(fp, 0)
            while v < round:
                for _ in range(128):
                    v = tl.load(fp, volatile=True)
                if v < round:
                    v = tl.atomic_add(fp, 0)
        accum += tl.load(ws + s * SLOT_ELEMS + offs).to(tl.float32)
    tl.store(c_ptr + offs, accum.to(dtype), mask=inb)


class ARPushKernel:
    """Host-side chunk producer: engine DATA + RMA SIGNALS, per chunk.

    Data: per dest, one hipMemcpyAsync of MY ws chunk into the peer's ws
    slot for me, on the push stream, stream-ordered after the chunk's GEMM
    (the caller records the GEMM event and makes the push stream wait it).

    Signal: host dushmem putmem of the staged u64 round into each peer's
    flag slot (source row = my_rank, column = chunk), issued only after
    the chunk's data copies completed -- the caller event-syncs the data
    before calling signal_chunk. See SIGNAL_MODE in all_gather_gemm for
    why the signal must not ride the engine queue (S5.3 cliff)."""

    def __init__(self, ws_base, flags_base, num_ranks, max_M, N, elemsz,
                 m_chunk_num):
        s = dcu_state()
        self._num_ranks = num_ranks
        self._elemsz = elemsz
        self._m_chunk_num = m_chunk_num
        self._slot_elems = max_M * N
        self._ws = ws_base
        self._flags = flags_base
        self._peer_ws = [s["host"].ptr(ws_base, d) for d in range(num_ranks)]
        self._stage = torch.empty(m_chunk_num, dtype=torch.int64,
                                  device="cuda")

    def stage_round(self, round):
        """Fill the flag staging buffer on the CURRENT stream -- call from
        the push stream BEFORE the first chunk's copies, so the fill lands
        before any ev_data event the signals are ordered behind."""
        self._stage.fill_(round)

    def push_chunk(self, c, my_rank, chunk_elems, stream_handle):
        """Engine data copies of chunk c into every peer's slot for me."""
        off = (my_rank * self._slot_elems + c * chunk_elems) * self._elemsz
        nbytes = chunk_elems * self._elemsz
        for d in range(self._num_ranks):
            if d == my_rank:
                continue
            memcpy_async(self._peer_ws[d] + off, self._ws + off, nbytes,
                         stream_handle)

    def signal_all(self, round, my_rank, m_chunk_num):
        """One blocking putmem of this op's whole flags row to every peer
        -- the exact all_gather_gemm "host" transport that measured
        zero-stall across shapes in S5. The caller guarantees all data
        copies landed (last chunk's event sync).

        History (S6, do not retry blind): per-chunk 8 B BLOCKING puts
        issued mutually by both ranks deadlocked (py-spy: both ranks
        inside dushmem_putmem, injection-credit style); per-chunk nbi
        puts + end-of-op quiet didn't deadlock but starved under SM load
        (~8 ms/op) and still hung on larger shapes. The row-putmem after
        one final event sync is the proven pattern."""
        host = dcu_state()["host"]
        nbytes = m_chunk_num * 8
        for d in range(self._num_ranks):
            if d == my_rank:
                continue
            host.putmem(self._flags + my_rank * self._m_chunk_num * 8,
                        self._stage.data_ptr(), nbytes, d)


@dataclasses.dataclass
class GEMMAllReduceTensorParallelContext:
    rank: int
    num_ranks: int
    max_M: int
    N: int
    ws_ptr: int  # symmetric [num_ranks, max_M, N]
    flags_ptr: int  # symmetric u64 [num_ranks, m_chunk_num]
    barrier: SymmBarrier
    push: ARPushKernel
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    M_PER_CHUNK: int = 512
    m_chunk_num: int = 0
    flag_round: int = 0
    gemm_stream: Optional[torch.cuda.streams.Stream] = None
    ar_stream: Optional[torch.cuda.streams.Stream] = None


def create_gemm_ar_intra_node_context(max_M, N, input_dtype, output_dtype,
                                      rank, num_ranks, mpi, comm,
                                      M_PER_CHUNK=512, gemm_stream=None,
                                      ar_stream=None):
    """Symmetric workspace + flags + barrier for gemm-allreduce.

    mpi/comm from pydushmem.host.mpi_init() (BEFORE `import torch`)."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert input_dtype == output_dtype, "partials are stored in output dtype"
    assert max_M % M_PER_CHUNK == 0 and M_PER_CHUNK % 64 == 0
    assert N % 128 == 0
    m_chunk_num = max_M // M_PER_CHUNK
    assert num_ranks * max_M * N < 2 ** 31  # i32 element offsets in consumer

    s = dcu_state()
    elemsz = torch.empty((), dtype=output_dtype).element_size()
    ws_ptr = s["host"].malloc(num_ranks * max_M * N * elemsz)
    flags_bytes = 8 * num_ranks * m_chunk_num
    flags_ptr = s["host"].malloc(flags_bytes)
    # flags gate reads of possibly-unwritten workspace: garbage >= round
    # would skip the wait. (hipMemset of OWN memory is safe -- peers write
    # through dushmem RMA / engine copies, which update our L2 lines.)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(flags_ptr), 0, flags_bytes) == 0

    barrier = SymmBarrier(num_ranks)
    push = ARPushKernel(ws_ptr, flags_ptr, num_ranks, max_M, N, elemsz,
                        m_chunk_num)

    current_stream = torch.cuda.current_stream()
    # the GEMM stream must differ from the caller's stream: the overlap is
    # chunk c+1's GEMM running under chunk c's push + reduce
    _gemm_stream = (torch.cuda.Stream() if gemm_stream is None
                    else gemm_stream)
    _ar_stream = torch.cuda.Stream() if ar_stream is None else ar_stream

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0

    return GEMMAllReduceTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        max_M=max_M,
        N=N,
        ws_ptr=ws_ptr,
        flags_ptr=flags_ptr,
        barrier=barrier,
        push=push,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
        M_PER_CHUNK=M_PER_CHUNK,
        m_chunk_num=m_chunk_num,
        gemm_stream=_gemm_stream,
        ar_stream=_ar_stream,
    )


def gemm_ar_intra_node_op(a, b, c, ctx, component="full"):
    """C = sum_r a_r @ b_r^T with the full C on every rank.

    component="gemm": benchmarking mode -- GEMM chunks into MY workspace
    slot only (no push/signal/consume; flag_round is NOT advanced, so the
    next full op's waits stay correct).

    a: [M, K_local]; b: [N, K_local]; c: [M, N] (caller-allocated)."""
    M, K = a.shape
    N = b.shape[0]
    rank, num_ranks = ctx.rank, ctx.num_ranks
    MPC = ctx.M_PER_CHUNK
    m_chunk_num = M // MPC
    assert b.shape[1] == K, "b must be [N, K]"
    assert M <= ctx.max_M and N == ctx.N and M % MPC == 0
    assert a.dtype == ctx.input_dtype and b.dtype == ctx.input_dtype
    # layouts assumed by the producer's loads and the raw-byte copies
    assert a.stride(1) == 1 and a.stride(0) == K
    assert b.stride(1) == 1 and b.stride(0) == K
    block_m = 64 if MPC <= 512 else 128
    assert MPC % block_m == 0 and N % 128 == 0 and K % 64 == 0

    cur = torch.cuda.current_stream()
    chunk_elems = MPC * N
    elemsz = torch.empty((), dtype=ctx.output_dtype).element_size()

    # my GEMM chunks overwrite ws[me]: after my previous op's consumer
    # (event wait across streams). Peer overwrites are excluded by the
    # op-start barrier -- a peer's push must not land in a slot my
    # still-running consumer reads.
    ctx.gemm_stream.wait_stream(cur)

    if component == "full":
        ctx.flag_round += 1
        ctx.barrier.round_barrier(rank)
        # the push stream must wait out the barrier (the arrive writes and
        # the spin kernel run on `cur`): a peer's push may not land in a ws
        # slot my previous op's consumer still reads -- the P6.1 hazard,
        # closed by stream order instead of timing luck
        ctx.ar_stream.wait_stream(cur)
        with torch.cuda.stream(ctx.ar_stream):
            ctx.push.stage_round(ctx.flag_round)

    evs_data = []
    ev_last = None
    for c_idx in range(m_chunk_num):
        c0 = c_idx * MPC
        a_c = a[c0:c0 + MPC]  # contiguous row slice
        my_slot_chunk = ctx.ws_ptr + (rank * ctx.max_M * N + c0 * N) * elemsz
        with torch.cuda.stream(ctx.gemm_stream):
            grid = (triton.cdiv(MPC, block_m) * triton.cdiv(N, 128),)
            kernel_gemm_ar_producer[grid](
                a_c, b,
                my_slot_chunk,
                MPC, N, K,
                a_c.stride(0), a_c.stride(1),
                b.stride(1), b.stride(0),
                N, 1,  # ws slot [max_M, N] contiguous
                BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4, num_stages=2)
            ev = torch.cuda.Event()
            ev.record(ctx.gemm_stream)
        ev_last = ev
        if component == "full":
            # chunk data push on the ar stream, stream-ordered after this
            # chunk's GEMM -- chunk c's push runs under chunk c+1's GEMM
            with torch.cuda.stream(ctx.ar_stream):
                ctx.ar_stream.wait_event(ev)
                ctx.push.push_chunk(c_idx, rank, chunk_elems,
                                    ctx.ar_stream.cuda_stream)
                ev_d = torch.cuda.Event()
                ev_d.record(ctx.ar_stream)
            evs_data.append(ev_d)

    if component == "full":
        # single batched signal after the LAST chunk's data copies
        # completed (see ARPushKernel.signal_all for the transport history)
        evs_data[-1].synchronize()
        ctx.push.signal_all(ctx.flag_round, rank, m_chunk_num)

        # Consumers launch only NOW: launching them per-chunk during the
        # GEMM was an 8 ms/op pathology -- spin-waiting consumer blocks
        # occupy every CU and starve the producer GEMM (the P9 SM-budget
        # rule from the findings doc; the AG op never hit it because its
        # consumer IS the GEMM). The flags are already set here, so the
        # spins pass immediately; the lost overlap is only the reduce's
        # tail, which the per-chunk-signal future work would recover.
        cur.wait_event(ev_last)  # own-slot chunks are complete
        assert chunk_elems % 2048 == 0  # block-aligned chunk boundaries
        kernel_ar_consumer_reduce[(triton.cdiv(M * N, 2048),)](
            ctx.ws_ptr, c, ctx.flags_ptr,
            ctx.flag_round, rank, M * N,
            num_ranks=num_ranks, m_chunk_num=ctx.m_chunk_num,
            SLOT_ELEMS=ctx.max_M * N, CHUNK_ELEMS=chunk_elems,
            BLOCK_SIZE=2048, num_warps=2)

    a.record_stream(ctx.gemm_stream)
    b.record_stream(ctx.gemm_stream)
    return c


def gemm_ar_intra_node(a, b, ctx):
    """GEMM + AllReduce (TP column-parallel linear): every rank returns
    the full c = sum_r a_r @ b_r^T.

    a: [M, K_local]; b: [N, K_local]."""
    if ctx is None:
        raise RuntimeError("requires ctx")
    c = torch.empty((a.shape[0], b.shape[0]), dtype=ctx.output_dtype,
                    device=a.device)
    gemm_ar_intra_node_op(a, b, c, ctx)
    return c
