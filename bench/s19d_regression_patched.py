"""S19d: single-card regression of the PATCHED production consumer
(triton_dist/kernels/dcu/gemm_ag_tr.py::kernel_gemm_ag_tr_consumer,
S19c restructure: scalar-only dual-source branch, single addressing
form / single stride_k, fused accumulator, no tile loop, no store-side
modulo recompute).

Arms:
  full    M8192 N29568 K8192 (EVEN_K=True): perf vs the s19c reference
          numbers (k_cons_r 25.23ms) and correctness vs fp32 ref
  tail    small shape with K % 64 != 0 (EVEN_K=False): the tail branch
          is NEW in the patched kernel (probes only ran EVEN_K=True)
  m2      full shape, world_size=1 (own-path only, a_base=localA band)

run (single card, NO mpirun):
  cd /root/private_data/tri-dist-hygon && export HSA_FORCE_FINE_GRAIN_PCIE=1
  python3 bench/s19d_regression_patched.py
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

from triton_dist.kernels.dcu.gemm_ag_tr import (  # noqa: E402
    kernel_gemm_ag_tr_consumer)

WARMUP, ITERS = 3, 7
OUT = os.path.join(REPO, "bench", "results", "s19d_regression.json")
RES = {"arms": {}}


def run_case(tag, M, N, K, R, bench_it):
    torch.manual_seed(42)
    ws = (torch.randn(M, K, device="cuda") * 0.02).half()
    a_loc = ws[:M // R]
    b = (torch.randn(N, K, device="cuda") * 0.02).half()
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    n_chunks = max(1, (M // R) // 512)
    CR = (M // R) // n_chunks
    flags = torch.full((R * n_chunks,), 1 << 28, dtype=torch.int32,
                       device="cuda")
    tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)

    def fn():
        kernel_gemm_ag_tr_consumer[(tiles,)](
            ws.data_ptr(), a_loc, b, c, M, N, K,
            K,                      # single A row stride
            1, K,                   # b is [N, K]
            c.stride(0), c.stride(1),
            0, R, n_chunks,
            flags.data_ptr(), 0,
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
    print(f"[s19d] {tag:24s} M{M} N{N} K{K} R{R}  "
          f"{'' if p50 is None else f'p50 {p50:7.2f} ms  '}md {md:.1e}",
          flush=True)
    assert md < 1e-2, f"{tag} WRONG md={md}"


print(f"[s19d] dev {torch.cuda.get_device_name(0)} "
      f"count={torch.cuda.device_count()}", flush=True)
run_case("full_r2", 8192, 29568, 8192, 2, bench_it=True)   # EVEN_K=True
run_case("tail_r2", 1024, 1152, 4104, 2, bench_it=False)   # EVEN_K=False
run_case("full_r1", 8192, 29568, 8192, 1, bench_it=True)   # own-path band

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19d] wrote {OUT}", flush=True)
print("[s19d] reference: s19c k_cons_r 25.23ms / orig 30.72ms "
      "(same-process, single card)", flush=True)
