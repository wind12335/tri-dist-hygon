"""DCU port of triton_dist/kernels/amd/gemm_reduce_scatter.py.

Data-plane note (measured on 2x BW1000): upstream scatters GEMM tiles with
direct tl.store into peer IPC-mapped buffers and reads them back with plain
tl.load. On this DCU stack, peer IPC stores reach DRAM but do not invalidate
the destination GPU's L2 -> stale reads (deterministic zeros on first touch
without memset, torn data later). The validated-coherent plane is dushmem
RMA, so this port splits the op differently:

  producer: plain grouped GEMM -> LOCAL scratch [M, N]
  scatter:  HOST-side hipMemcpyAsync per (dest rank) of my partial rows
            into a symmetric-heap buffer [num_ranks, M/num_ranks, N]
            (device-initiated dushmem puts run at ~19 MB/s on this stack;
            engine memcpys into the peer VA run at ~110 GB/s -- S4
            transport probe; the following same-stream barrier replaces
            the old in-kernel quiet)
  consumer: local reduce over the symmetric buffer's slices (all local)

Sync plane: SymmBarrier (dushmem atomic_set), as in common_ops.
"""
import ctypes
import dataclasses

import torch
import triton
import triton.language as tl

from pydushmem import device as dushmem_dev
from triton_dist.kernels.dcu.common_ops import (
    SymmBarrier,
    barrier_all_on_stream,
    dcu_state,
    fan_out_memcpy_async,
    init_dcu,
)


@triton.jit
def kernel_gemm_rs_producer_local(
        a_ptr, b_ptr, c_ptr,  # c = local scratch [M, N]
        M, N, K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr  #
):
    """Standard grouped-tile matmul C[M,N] = A[M,K] @ B[K,N] (local scratch)."""
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

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(do_not_specialize=["src_base", "dst_base", "my_rank", "total"])
def kernel_scatter_put(
        src_base,  # local scratch VA (int64), [M, N]
        dst_base,  # symmetric-heap VA (int64) of [num_ranks, M/num_ranks, N]
        my_rank,
        total,  # elements per shard = M_per_rank * N
        num_ranks: tl.constexpr,
        CHUNK_ELEMS: tl.constexpr,
        ELEMSZ: tl.constexpr,
):
    """UNUSED (kept for reference): device-initiated dushmem scatter.
    ~19 MB/s on this stack -- see ScatterPutKernel for the live path."""
    for d in range(num_ranks):
        for off in range(0, total, CHUNK_ELEMS):
            n = tl.minimum(CHUNK_ELEMS, total - off)
            src_off = ((d * total + off) * ELEMSZ).to(tl.int64)
            dst_off = ((my_rank * total + off) * ELEMSZ).to(tl.int64)
            dushmem_dev.dushmem_putmem_nbi(dst_base + dst_off, src_base + src_off,
                                           n * ELEMSZ, d)
    dushmem_dev.dushmem_quiet()


@triton.jit
def kernel_consumer_reduce(
        c_ptr_i64,  # symmetric-heap VA (int64) of OWN [num_ranks, Mpr, N]
        out_ptr,  # [M_per_rank, N]
        M_per_rank,
        N,
        rank,
        num_ranks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = tl.where(offs < M_per_rank * N, offs, 0)

    dtype = out_ptr.dtype.element_ty
    c_ptr = c_ptr_i64.to(tl.pointer_type(dtype))
    out_ptrs = out_ptr + offs
    accum = tl.zeros((BLOCK_SIZE, ), dtype=out_ptr.dtype.element_ty)
    for i in range(0, num_ranks):
        cur_rank = (i + rank + 1) % num_ranks
        c_ptrs = c_ptr + offs + cur_rank * M_per_rank * N
        data = tl.load(c_ptrs)
        accum += data
    tl.store(out_ptrs, accum)


def matmul_local(a, b, scratch, transpose_weight=False):
    if transpose_weight:
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        K, N = b.shape
        stride_bk, stride_bn = b.stride(0), b.stride(1)
    else:
        assert a.shape[1] == b.shape[1], "Incompatible dimensions"
        N, K = b.shape
        stride_bk, stride_bn = b.stride(1), b.stride(0)
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    max_M, scratch_N = scratch.shape
    assert scratch_N == N and M <= max_M

    alignment = 256
    assert M % alignment == 0 and N % alignment == 0 and K % alignment == 0

    # S5 sweep (bench/results/tune_gemm.json): BLOCK_K=64/warps=4 is best
    # on every shape (+7% large, +40% small-M); small M prefers BLOCK_M=64.
    # S10 sweep (bench/results/tune_gemm2.json): num_stages=1 wins 6-15% on
    # the big/mid shapes (M>512, K<8192); decode (M<=512) and deep-K keep s2.
    block_m = 64 if M <= 512 else 128
    stages = 1 if (M > 512 and K < 8192) else 2
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    compiled = kernel_gemm_rs_producer_local[grid](
        a, b, scratch,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        stride_bk, stride_bn,  #
        scratch.stride(0), scratch.stride(1),  #
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8,
        num_warps=4, num_stages=stages,
    )
    return compiled


class ScatterPutKernel:
    """Host-side scatter producer.

    Per dest rank d (self included -- my own slice feeds my consumer too):
    one hipMemcpyAsync of my partial rows [d*Mpr, (d+1)*Mpr) into rank d's
    symmetric-buffer slice `my_rank`, on the CURRENT torch stream. The
    barrier that follows on the same stream is the landing fence (stream
    order = memcpy completion), replacing the old in-kernel dushmem quiet.

    Engine memcpys into the peer VA: ~110 GB/s, L2-coherent for the
    consumer's plain loads; device-initiated dushmem puts: ~19 MB/s
    (S4 transport probe -- see common_ops.memcpy_async)."""

    def __init__(self, src_base, dst_base, num_ranks, elemsz):
        s = dcu_state()
        self._num_ranks = num_ranks
        self._elemsz = elemsz
        self._peer_dst = [s["host"].ptr(dst_base, d) for d in range(num_ranks)]

    def launch(self, src_base, dst_base, my_rank, total):
        nbytes = total * self._elemsz
        # S11: per-dest copies fanned out across the push-stream pool
        # (joined back to the current stream, so the landing barrier that
        # follows keeps its stream-order completion semantics)
        fan_out_memcpy_async([
            (self._peer_dst[d] + my_rank * nbytes,
             src_base + d * nbytes, nbytes)
            for d in range(self._num_ranks)])


def ring_reduce_after_scatter(rank, num_ranks, c_ptr_i64, M, N,
                              out_dtype, stream):
    M_per_rank = M // num_ranks
    output = torch.empty((M_per_rank, N), dtype=out_dtype, device="cuda")
    grid = lambda META: (triton.cdiv(M_per_rank * N, META["BLOCK_SIZE"]), )
    with torch.cuda.stream(stream):
        kernel_consumer_reduce[grid](
            c_ptr_i64,
            output,
            M_per_rank,
            N,
            rank=rank,
            num_ranks=num_ranks,
            BLOCK_SIZE=2048,
            num_warps=2,
        )
    return output


def gemm_rs_intra_node_op(a, b, output_dtype, rank, num_ranks, ctx):
    """gemm + reduce-scatter over dushmem planes; rank r keeps rows
    [r*M/num_ranks, (r+1)*M/num_ranks) of sum_r (a_r x b)."""
    M, local_K = a.shape
    N, weight_local_K = b.shape
    assert weight_local_K == local_K, "b must be [N, K]"
    assert a.dtype == b.dtype == ctx.scratch.dtype

    # round 1: previous readers of my buffer slice are done
    barrier_all_on_stream(rank, num_ranks, ctx.barrier)

    matmul_local(a, b, ctx.scratch)
    M_per_rank = M // num_ranks
    ctx.scatter_put.launch(ctx.scratch.data_ptr(), ctx.scatter_ptr, rank,
                           M_per_rank * N)

    # round 2: every rank's puts have landed (quiet is inside the kernel)
    barrier_all_on_stream(rank, num_ranks, ctx.barrier)
    return ring_reduce_after_scatter(rank, num_ranks, ctx.scatter_ptr, M, N,
                                     output_dtype, torch.cuda.current_stream())


def gemm_rs_intra_node(a, b, ctx):
    """GEMM Reduce-Scatter for Intra-Node (DCU planes).

    computes local GEMM (a x b) to generate partial results, followed by
    `reduce_scatter` to produce c; rank r keeps rows
    [r*M/world_size, (r+1)*M/world_size) of sum_r (a_r x b).
    """
    return gemm_rs_intra_node_op(
        a, b, ctx.output_dtype, ctx.rank, ctx.num_ranks, ctx)


@dataclasses.dataclass
class GEMMReduceScatterTensorParallelContext:
    rank: int
    num_ranks: int
    scratch: torch.Tensor  # local [max_M, N]
    scatter_ptr: int  # symmetric-heap VA of [num_ranks, max_M//num_ranks, N]
    barrier: SymmBarrier
    scatter_put: ScatterPutKernel
    output_dtype: torch.dtype


def create_gemm_rs_intra_node_context(max_M, N, output_dtype, rank, num_ranks,
                                      mpi, comm, fuse_scatter=True,
                                      transpose_weight=False):
    """Create local scratch + symmetric scatter buffer + barrier.

    mpi/comm come from pydushmem.host.mpi_init() (must have run BEFORE
    `import torch`)."""
    if not fuse_scatter or transpose_weight:
        raise NotImplementedError()

    init_dcu(mpi, comm, rank, num_ranks)
    elem = torch.empty((), dtype=output_dtype).element_size()
    s = dcu_state()
    scratch = torch.empty((max_M, N), dtype=output_dtype, device="cuda")
    scatter_ptr = s["host"].malloc(max_M * N * elem)  # symmetric on all PEs
    barrier = SymmBarrier(num_ranks)
    scatter_put = ScatterPutKernel(scratch.data_ptr(), scatter_ptr,
                                   num_ranks, elem)

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0

    return GEMMReduceScatterTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        scratch=scratch,
        scatter_ptr=scatter_ptr,
        barrier=barrier,
        scatter_put=scatter_put,
        output_dtype=output_dtype,
    )
