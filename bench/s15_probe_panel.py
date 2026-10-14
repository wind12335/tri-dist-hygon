"""S15 probe (single GPU): panel GEMM config at the final shapes.
kernel_gemm_panel_rd (BM=64/s2 -- the current rule at CR=256) vs BM=128
variants vs torch.matmul, at M8192 N28672 K8192.

Env: export HSA_FORCE_FINE_GRAIN_PCIE=1
Run:  python3 bench/s15_probe_panel.py
"""
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache_s15probe")

import torch  # noqa: E402
import triton  # noqa: E402

from triton_dist.kernels.dcu.gemm_ar_rd import kernel_gemm_panel_rd  # noqa: E402

torch.cuda.set_device(0)
DTYPE = torch.float16

M, N, K, CR = 8192, 28672, 8192, 256


def tloop(fn, warmup=5, iters=10):
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


a = (torch.randn(M, K, device="cuda") * 0.02).half()
b = (torch.randn(N, K, device="cuda") * 0.02).half()
c = torch.empty(M, N, dtype=DTYPE, device="cuda")
ref = (a[:1024].float() @ b.float().t())

t_torch = tloop(lambda: torch.matmul(a, b.t(), out=c))
print(f"torch.matmul full        {t_torch:8.2f} ms", flush=True)

n_chunks = M // CR


def panels(bm, stages):
    def run():
        for i in range(n_chunks):
            r0 = i * CR
            panel = c[r0:r0 + CR]
            grid = (triton.cdiv(CR, bm) * triton.cdiv(N, 128),)
            kernel_gemm_panel_rd[grid](
                a, b, panel, M, N, K,
                a.stride(0), a.stride(1), b.stride(1), b.stride(0),
                r0, CR,
                BLOCK_SIZE_M=bm, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4, num_stages=stages)
    return run


for bm, st in ((64, 2), (128, 2), (128, 1), (256, 2)):
    try:
        fn = panels(bm, st)
        fn()
        torch.cuda.synchronize()
        md = (c[:1024].float() - ref).abs().max().item()
        t = tloop(fn)
        print(f"panels CR256 BM{bm:<3d} s{st}     {t:8.2f} ms  md={md:.1e}",
              flush=True)
    except Exception as e:
        print(f"panels CR256 BM{bm:<3d} s{st}     FAIL {e}", flush=True)

print("done", flush=True)

# CR sweep at the winning tile config (bigger panels = fewer, fuller
# launches; the 8MB panel cap may be too conservative for GEMM efficiency)
for CR2 in (1024, 2048):
    n_chunks = M // CR2

    def run2():
        for i in range(n_chunks):
            r0 = i * CR2
            panel = c[r0:r0 + CR2]
            grid = (triton.cdiv(CR2, 128) * triton.cdiv(N, 128),)
            kernel_gemm_panel_rd[grid](
                a, b, panel, M, N, K,
                a.stride(0), a.stride(1), b.stride(1), b.stride(0),
                r0, CR2,
                BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4, num_stages=2)
    run2()
    torch.cuda.synchronize()
    md = (c[:1024].float() - ref).abs().max().item()
    t = tloop(run2)
    print(f"panels CR{CR2:<5d} BM128 s2     {t:8.2f} ms  md={md:.1e}",
          flush=True)
print("done2", flush=True)
