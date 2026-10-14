"""S13 probe (single GPU, no mpirun): why is one fused banded chunk-GEMM
launch slower than the equivalent whole-matrix matmul_local launch at the
same CTA count / tile config?  A/B the kernel-body deltas one at a time.

Env: export HSA_FORCE_FINE_GRAIN_PCIE=1
Run:  python3 bench/s13_probe_banded.py
"""
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache_s13probe")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu.gemm_reduce_scatter import (  # noqa: E402
    kernel_gemm_rs_producer_local,
)

torch.cuda.set_device(0)
DTYPE = torch.float16


# variant A: current kernel_gemm_banded_rd body (seg div/mod + hardcoded
# output row stride N)
@triton.jit(do_not_specialize=["chunk_row_base", "rows", "M",
                               "seg_row_stride", "tiles_per_seg"])
def kernel_banded_a(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        seg_row_stride, chunk_row_base, rows, tiles_per_seg,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
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
    c_ptrs = (c_ptr + (row0 + offs_cm)[:, None] * N + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < rows) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


# variant B: same but WITHOUT the seg mapping (chunk_row_base covers all
# rows; rows = M) -- isolates the seg div/mod + double-height grid cost
@triton.jit(do_not_specialize=["chunk_row_base", "rows", "M",
                               "seg_row_stride", "tiles_per_seg"])
def kernel_banded_b(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        seg_row_stride, chunk_row_base, rows, tiles_per_seg,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(rows, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    row0 = chunk_row_base
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
    c_ptrs = (c_ptr + (row0 + offs_cm)[:, None] * N + offs_cn[None, :])
    c_mask = (offs_cm[:, None] < rows) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


def tloop(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.fmean(sorted(ts)[len(ts) // 2:])


M, N, K = 2048, 2048, 2048
R = 2
CR = 1024
m_pr = M // R
a = (torch.randn(M, K, device="cuda") * 0.02).half()
b = (torch.randn(N, K, device="cuda") * 0.02).half()
c = torch.empty(M, N, dtype=DTYPE, device="cuda")

ref = a.float() @ b.float().t()

cfg = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
           GROUP_SIZE_M=8, num_warps=4, num_stages=1)

# matmul_local equivalent (kernel_gemm_rs_producer_local, M=2048)
grid_full = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
t_ref = tloop(lambda: kernel_gemm_rs_producer_local[grid_full](
    a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(1), b.stride(0),
    c.stride(0), c.stride(1), **cfg))
md = (c.float() - ref).abs().max().item()
print(f"producer_local [2048]      {t_ref:.3f} ms  md={md:.1e}", flush=True)

# fused banded (A): grid = tiles_per_seg * R
tiles = triton.cdiv(CR, 128) * triton.cdiv(N, 128)
grid_band = (tiles * R,)
t_a = tloop(lambda: kernel_banded_a[grid_band](
    a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(1), b.stride(0),
    m_pr, 0, CR, tiles, **cfg))
md = (c.float() - ref).abs().max().item()
print(f"banded_a fused 2x1024      {t_a:.3f} ms  md={md:.1e}", flush=True)

# variant B: no seg mapping, single chunk covering all M rows
tiles_b = triton.cdiv(M, 128) * triton.cdiv(N, 128)
t_b = tloop(lambda: kernel_banded_b[tiles_b,](
    a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(1), b.stride(0),
    m_pr, 0, M, tiles_b, **cfg))
md = (c.float() - ref).abs().max().item()
print(f"banded_b no-seg [2048]     {t_b:.3f} ms  md={md:.1e}", flush=True)

# A again but num_stages=2 (is the s1 rule wrong for this body?)
cfg2 = dict(cfg, num_stages=2)
t_a2 = tloop(lambda: kernel_banded_a[grid_band](
    a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(1), b.stride(0),
    m_pr, 0, CR, tiles, **cfg2))
print(f"banded_a fused s2          {t_a2:.3f} ms", flush=True)
t_ref2 = tloop(lambda: kernel_gemm_rs_producer_local[grid_full](
    a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(1), b.stride(0),
    c.stride(0), c.stride(1), **cfg2))
print(f"producer_local s2          {t_ref2:.3f} ms", flush=True)

# also: does the ORDER of grid dims matter? swap pid mapping: seg = pid % R
print("done", flush=True)
