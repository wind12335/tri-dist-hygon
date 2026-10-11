"""S5 GEMM autotune sweep (single GPU, no mpirun).

Sweep tile configs for kernel_gemm_rs_producer_local (the local-GEMM path
shared by gemm_reduce_scatter; the AG consumer uses the same tile geometry).
M/N/K are runtime args -> one compile per config, reused across shapes.

Env: export HSA_FORCE_FINE_GRAIN_PCIE=1
Run:  python3 bench/tune_gemm.py [--quick]   (results -> bench/results/)
"""
import os
import sys
import json
import time
import itertools
import statistics
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache_tune_gemm")

import torch  # noqa: E402
import triton  # noqa: E402

from triton_dist.kernels.dcu.gemm_reduce_scatter import (  # noqa: E402
    kernel_gemm_rs_producer_local,
)

torch.cuda.set_device(0)
DTYPE = torch.float16
SCALE = 0.02

# (M, N, K): RS producer geometry + AG consumer geometry (full M, per-rank N)
SHAPES = [
    (4096, 2048, 2048),  # RS big / AG validated shape
    (2048, 2048, 2048),  # AG validated shape
    (2048, 4096, 4096),  # RS bench shape
    (2048, 8192, 4096),  # AG wide-N
    (2048, 1024, 8192),  # AG deep-K
    (1024, 4096, 4096),  # AG mid
    (256, 4096, 4096),   # AG decode small-M
]

BMS = [64, 128, 256]
BNS = [64, 128, 256]
BKS = [32, 64, 128]
WARPS = [4, 8]
STAGES = [2, 3, 4, 5]
GROUP = 8

DEFAULT_CFG = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
                   GROUP_SIZE_M=GROUP, num_warps=8, num_stages=2)  # pre-S5


def cfgs(quick=False):
    """Yield (key, cfg). The shipped default is always first (baseline)."""
    yield "default", dict(DEFAULT_CFG)
    for bm, bn, bk, w, s in itertools.product(BMS, BNS, BKS, WARPS, STAGES):
        if (bm, bn, bk, w, s) == (128, 128, 32, 8, 2):
            continue  # == default
        if bm * bn < 8192:          # 64x64 and 64x128 too small to matter
            continue
        if bm * bn > 32768:         # 256x256 spills
            continue
        if bk == 128 and s >= 3:    # smem blowup
            continue
        if bm == 256 and bn == 256:
            continue
        if quick and not (s in (2, 3) and bk in (32, 64) and w == 8):
            continue
        yield f"{bm}x{bn}x{bk}_w{w}_s{s}", dict(
            BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=GROUP, num_warps=w, num_stages=s)


def bench_one(a, b, c, M, N, K, cfg, iters=20):
    grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
    args = (a, b, c, M, N, K, a.stride(0), a.stride(1),
            b.stride(1), b.stride(0), c.stride(0), c.stride(1))
    for _ in range(3):
        kernel_gemm_rs_producer_local[grid](*args, **cfg)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        kernel_gemm_rs_producer_local[grid](*args, **cfg)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]  # p50


def main():
    quick = "--quick" in sys.argv
    torch.manual_seed(9000)

    results = {"env": {"date": time.strftime("%Y-%m-%d %H:%M"),
                       "gpu": torch.cuda.get_device_name(0),
                       "dtype": "fp16", "quick": quick},
               "shapes": {}}

    for (M, N, K) in SHAPES:
        a = (torch.randn(M, K, device="cuda") * SCALE).half()
        b = (torch.randn(N, K, device="cuda") * SCALE).half()
        c = torch.empty(M, N, dtype=DTYPE, device="cuda")
        av, bv, cv = a, b, c

        with torch.no_grad():
            ref = av.float() @ bv.float().t()

        # vendor baseline
        ts = []
        for _ in range(20):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            torch.matmul(av, bv.t(), out=cv)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        t_vendor = statistics.median(ts)

        rows = []
        n_fail = 0
        for key, cfg in cfgs(quick):
            try:
                ms = bench_one(av, bv, cv, M, N, K, cfg)
                # correctness spot check
                cv.zero_()
                grid = (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)
                kernel_gemm_rs_producer_local[grid](
                    av, bv, cv, M, N, K, av.stride(0), av.stride(1),
                    bv.stride(1), bv.stride(0), cv.stride(0), cv.stride(1), **cfg)
                torch.cuda.synchronize()
                md = (cv.float() - ref).abs().max().item()
                if md > 1e-2:
                    print(f"[{M}x{N}x{K}] {key}: WRONG md={md:.1e}", flush=True)
                    continue
                rows.append({"cfg": key, "ms": ms, "md": md})
            except Exception as ex:
                n_fail += 1
                if "out of resource" not in str(ex).lower():
                    print(f"[{M}x{N}x{K}] {key}: {type(ex).__name__}: {ex}",
                          flush=True)
        rows.sort(key=lambda r: r["ms"])
        results["shapes"][f"{M}x{N}x{K}"] = {
            "vendor_matmul_ms": t_vendor, "rows": rows, "n_fail": n_fail}

        print(f"\n=== {M}x{N}x{K}  vendor={t_vendor:.3f} ms  "
              f"({n_fail} cfg failed)", flush=True)
        base = next((r["ms"] for r in rows if r["cfg"] == "default"), None)
        for r in rows[:6]:
            speedup = base / r["ms"] if base else 0
            print(f"  {r['cfg']:>22} {r['ms']:7.3f} ms  x{speedup:.2f} vs default"
                  f"  ({t_vendor / r['ms']:.2f}x vs vendor)", flush=True)
        if base:
            print(f"  {'default':>22} {base:7.3f} ms  (x1.00)", flush=True)

    out = os.path.join(REPO, "bench", "results", "tune_gemm.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nresults -> {out}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
