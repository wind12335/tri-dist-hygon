"""S19i: old-vs-new A/B of the base-AG-push consumer (full grid, s2).
run: export HSA_FORCE_FINE_GRAIN_PCIE=1; python3 bench/s19i_push_ab.py"""
import os
import sys
import importlib.util
import json
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache_s19")

import torch  # noqa: E402
import triton  # noqa: E402

from triton_dist.kernels.dcu.all_gather_gemm import (  # noqa: E402
    kernel_gemm_ag_consumer as k_new)

_spec = importlib.util.spec_from_file_location("old_agg", "/tmp/old_agg.py")
old_mod = importlib.util.module_from_spec(_spec)
sys.modules["old_agg"] = old_mod
_spec.loader.exec_module(old_mod)
k_old = old_mod.kernel_gemm_ag_consumer

M, N, K = 8192, 29568, 8192
R = 2
WARMUP, ITERS = 3, 7
ROUND = 7
OUT = os.path.join(REPO, "bench", "results", "s19i_push_ab.json")
RES = {"M": M, "N": N, "K": K, "arms": {}}

torch.manual_seed(42)
ws = (torch.randn(M, K, device="cuda") * 0.02).half()
a_loc = ws[:M // R].contiguous()
b = (torch.randn(N, K, device="cuda") * 0.02).half()
c = torch.empty(M, N, dtype=torch.float16, device="cuda")
n_chunks = 8
CR = (M // R) // n_chunks
flags = torch.full((R * n_chunks,), ROUND, dtype=torch.int64,
                   device="cuda")
tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)


def maxdiff():
    md = 0.0
    bf = b.float()
    for r0 in range(0, M, 2048):
        ref = (ws[r0:r0 + 2048].float() @ bf.t()).half()
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref.float()).abs().max().item())
    return md


def fn_old():
    k_old[(tiles,)](
        ws.data_ptr(), a_loc, b, c, M, N, K,
        K, 1, a_loc.stride(0), a_loc.stride(1),
        1, K, c.stride(0), c.stride(1),
        0, R, flags.data_ptr(), ROUND,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8, M_PER_CHUNK=CR, NUM_SMS=tiles,
        EVEN_K=True, num_warps=4, num_stages=2)


def fn_new():
    k_new[(tiles,)](
        ws.data_ptr(), a_loc, b, c, M, N, K,
        K, 1, K, c.stride(0), c.stride(1),
        0, R, flags.data_ptr(), ROUND,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8, M_PER_CHUNK=CR,
        EVEN_K=True, num_warps=4, num_stages=2)


def bench(fn, tag):
    fn()
    torch.cuda.synchronize()
    md = maxdiff()
    st = torch.cuda.current_stream()
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(st)
        fn()
        e.record(st)
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    p50 = statistics.median(ts)
    RES["arms"][tag] = {"p50": round(p50, 3), "md": md}
    print(f"[s19i] {tag:16s} p50 {p50:7.2f} ms  md {md:.1e}", flush=True)
    assert md < 1e-2, f"{tag} WRONG md={md}"


bench(fn_old, "old (HEAD)")
bench(fn_new, "new (S19c)")
bench(fn_old, "old again")
bench(fn_new, "new again")

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19i] wrote {OUT}", flush=True)
