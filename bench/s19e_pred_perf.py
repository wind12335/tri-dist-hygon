"""S19e: single-card perf of the PATCHED pred consumer
(kernel_gemm_ag_tr_consumer_pred, S19c restructure + the S17 deep-K
num_stages guard that was never mirrored into ag_tr_pred.py -- the S18
i1 arms ran the degenerate M-only rule, i.e. s1 at K8192).

Production launch form: grid = min(tiles, 128) PERSISTENT, table-driven
mapping mode 0 (band-major own-first rotation == S14 behavior), flags
pre-armed (gate exits after one atomic).

run: export HSA_FORCE_FINE_GRAIN_PCIE=1; python3 bench/s19e_pred_perf.py
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

from triton_dist.kernels.dcu.ag_tr_pred import (  # noqa: E402
    kernel_gemm_ag_tr_consumer_pred,
    _CONSUMER_BLOCK_M, _CONSUMER_BLOCK_N, _CONSUMER_BLOCK_K,
    _CONSUMER_GROUP_M, _CONSUMER_WARPS, _CONSUMER_GRID_CAP)

M, N, K = 8192, 29568, 8192
R = 2
WARMUP, ITERS = 3, 7
OUT = os.path.join(REPO, "bench", "results", "s19e_pred_perf.json")
RES = {"M": M, "N": N, "K": K, "arms": {}}

torch.manual_seed(42)
ws = (torch.randn(M, K, device="cuda") * 0.02).half()
a_loc = ws[:M // R].contiguous()
b = (torch.randn(N, K, device="cuda") * 0.02).half()
c = torch.empty(M, N, dtype=torch.float16, device="cuda")

n_chunks = 8
CR = (M // R) // n_chunks
flags = torch.full((R * n_chunks,), 1 << 28, dtype=torch.int32,
                   device="cuda")
ctl = torch.zeros(4 + 2 * R, dtype=torch.int32, device="cuda")
ctl[2] = 0                      # mode 0 = S14 band-major own-first
ctl[4 + 0] = 0                  # order[0]
ctl[4 + 1] = 1                  # order[1]
ctl[4 + R + 0] = 128            # budget[0]
ctl[4 + R + 1] = 128            # budget[1]

print(f"[s19e] GRID_CAP={_CONSUMER_GRID_CAP} tensors ready", flush=True)


def maxdiff():
    md = 0.0
    bf = b.float()
    for r0 in range(0, M, 2048):
        ref = (ws[r0:r0 + 2048].float() @ bf.t()).half()
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref.float()).abs().max().item())
    return md


def launch(grid_c, stages):
    def fn():
        kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
            ws.data_ptr(), a_loc, b, c, M, N, K,
            K,                          # single A row stride
            1, K,                       # b is [N, K]
            c.stride(0), c.stride(1),
            0, R, n_chunks,
            flags.data_ptr(), ctl.data_ptr(), 0,
            diag_ptr=0,
            BLOCK_SIZE_M=_CONSUMER_BLOCK_M,
            BLOCK_SIZE_N=_CONSUMER_BLOCK_N,
            BLOCK_SIZE_K=_CONSUMER_BLOCK_K,
            GROUP_SIZE_M=_CONSUMER_GROUP_M,
            M_PER_CHUNK=CR, NUM_SMS=grid_c,
            EVEN_K=(K % _CONSUMER_BLOCK_K == 0),
            DIAG=False,
            num_warps=_CONSUMER_WARPS,
            num_stages=stages)
    return fn


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
    print(f"[s19e] {tag:40s} p50 {p50:7.2f} ms  md {md:.1e}", flush=True)
    assert md < 1e-2, f"{tag} WRONG md={md}"


tiles = triton.cdiv(M, _CONSUMER_BLOCK_M) * triton.cdiv(N, _CONSUMER_BLOCK_N)
bench(launch(min(tiles, _CONSUMER_GRID_CAP),
             1 if (M > 512 and K < 8192) else 2),
      "prod: grid128 persistent s2 (S17 rule)")
bench(launch(min(tiles, _CONSUMER_GRID_CAP), 1),
      "ref:   grid128 persistent s1 (S18 degenerate)")
bench(launch(tiles, 2),
      "ref:   full-grid one-tile s2")

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19e] wrote {OUT}", flush=True)
