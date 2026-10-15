"""S19h: single-card regression of the PATCHED base-AG-push consumer
(all_gather_gemm.py::kernel_gemm_ag_consumer, S19c restructure; same
recipe as gemm_ag_tr.py -- this kernel keeps its u64-flag hybrid spin
with expected == round).

Arms mirror s19d: full-shape perf+correctness (EVEN_K=True), small-shape
EVEN_K=False tail correctness.

run: export HSA_FORCE_FINE_GRAIN_PCIE=1; python3 bench/s19h_push_regression.py
"""
import os
import sys
import json
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache_s19")

import torch  # noqa: E402
import triton  # noqa: E402

from triton_dist.kernels.dcu.all_gather_gemm import (  # noqa: E402
    kernel_gemm_ag_consumer)

WARMUP, ITERS = 3, 7
OUT = os.path.join(REPO, "bench", "results", "s19h_push_regression.json")
RES = {"arms": {}}
ROUND = 7  # expected value the pre-armed u64 flags carry


def run_case(tag, M, N, K, R, bench_it):
    torch.manual_seed(42)
    ws = (torch.randn(M, K, device="cuda") * 0.02).half()
    a_loc = ws[:M // R]
    b = (torch.randn(N, K, device="cuda") * 0.02).half()
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    n_chunks = max(1, (M // R) // 512)
    CR = (M // R) // n_chunks
    flags = torch.full((R * n_chunks,), ROUND, dtype=torch.int64,
                       device="cuda")   # u64 slots, low word == ROUND
    tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)

    def fn():
        kernel_gemm_ag_consumer[(tiles,)](
            ws.data_ptr(), a_loc, b, c, M, N, K,
            K,                      # single A row stride
            1, K,                   # b is [N, K]
            c.stride(0), c.stride(1),
            0, R,
            flags.data_ptr(), ROUND,
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8, M_PER_CHUNK=CR,
            EVEN_K=(K % 64 == 0),
            num_warps=4,
            num_stages=1 if (M > 512 and K < 8192) else 2)

    fn()
    torch.cuda.synchronize()
    md = 0.0
    bf = b.float()
    for r0 in range(0, M, 2048):
        ref = (ws[r0:r0 + 2048].float() @ bf.t()).half()
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref.float()).abs().max().item())
    del bf
    torch.cuda.empty_cache()

    p50 = None
    if bench_it:
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
        p50 = round(statistics.median(ts), 3)

    RES["arms"][tag] = {"M": M, "N": N, "K": K, "R": R,
                        "p50": p50, "md": md}
    print(f"[s19h] {tag:12s} M{M} N{N} K{K} R{R}  "
          f"{'' if p50 is None else f'p50 {p50:7.2f} ms  '}md {md:.1e}",
          flush=True)
    assert md < 1e-2, f"{tag} WRONG md={md}"


run_case("full_r2", 8192, 29568, 8192, 2, bench_it=True)
run_case("tail_r2", 1024, 1152, 4104, 2, bench_it=False)

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19h] wrote {OUT}", flush=True)
