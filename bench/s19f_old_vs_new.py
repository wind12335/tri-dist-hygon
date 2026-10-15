"""S19f: old-vs-new A/B of the PRED consumer in its PRODUCTION form.

The i1 op launches kernel_gemm_ag_tr_consumer_pred as a PERSISTENT grid
(min(tiles, 128)) to leave CUs for the pull lanes + observer. s19e found
that form prefers s1 even at deep K (31.5 vs 39.3ms at s2), so both arms
here run s1 -- the historical production setting.

Arms (same process, same tensors, M8192 N29568 K8192, flags pre-armed,
ctl mode 0):
  old   HEAD's kernel (dual addressing forms, acc +=, store modulo
        recompute), OLD 4-stride launch signature
  new   S19c-restructured kernel (scalar-only dual source, single form,
        fused acc), new single-stride_k signature

run: export HSA_FORCE_FINE_GRAIN_PCIE=1; python3 bench/s19f_old_vs_new.py
"""
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

from triton_dist.kernels.dcu.ag_tr_pred import (  # noqa: E402
    kernel_gemm_ag_tr_consumer_pred as k_new,
    _CONSUMER_BLOCK_M, _CONSUMER_BLOCK_N, _CONSUMER_BLOCK_K,
    _CONSUMER_GROUP_M, _CONSUMER_WARPS, _CONSUMER_GRID_CAP)

# HEAD's module under a distinct name (imports only constants/ctx helpers
# from the patched gemm_ag_tr -- its OLD kernel body is inline)
_spec = importlib.util.spec_from_file_location(
    "old_ag_tr_pred", "/tmp/old_ag_tr_pred.py")
old_mod = importlib.util.module_from_spec(_spec)
sys.modules["old_ag_tr_pred"] = old_mod
_spec.loader.exec_module(old_mod)
k_old = old_mod.kernel_gemm_ag_tr_consumer_pred

M, N, K = 8192, 29568, 8192
R = 2
WARMUP, ITERS = 3, 7
OUT = os.path.join(REPO, "bench", "results", "s19f_old_vs_new.json")
RES = {"M": M, "N": N, "K": K, "form": "grid128 persistent s1", "arms": {}}

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
ctl[2] = 0
ctl[4 + 0] = 0
ctl[4 + 1] = 1
ctl[4 + R + 0] = 128
ctl[4 + R + 1] = 128

tiles = triton.cdiv(M, _CONSUMER_BLOCK_M) * triton.cdiv(N, _CONSUMER_BLOCK_N)
grid_c = min(tiles, _CONSUMER_GRID_CAP)
STAGES = 1  # production M-only rule at M8192


def maxdiff():
    md = 0.0
    bf = b.float()
    for r0 in range(0, M, 2048):
        ref = (ws[r0:r0 + 2048].float() @ bf.t()).half()
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref.float()).abs().max().item())
    return md


def fn_old():
    k_old[(grid_c, 1, 1)](
        ws.data_ptr(), a_loc, b, c, M, N, K,
        K, 1,                      # gathered strides (OLD signature)
        a_loc.stride(0), a_loc.stride(1),
        1, K,
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
        num_stages=STAGES)


def fn_new():
    k_new[(grid_c, 1, 1)](
        ws.data_ptr(), a_loc, b, c, M, N, K,
        K,                        # single A row stride (NEW signature)
        1, K,
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
        num_stages=STAGES)


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
    print(f"[s19f] {tag:12s} p50 {p50:7.2f} ms  md {md:.1e}", flush=True)
    assert md < 1e-2, f"{tag} WRONG md={md}"


bench(fn_old, "old (HEAD)")
bench(fn_new, "new (S19c)")
bench(fn_old, "old (HEAD) again")
bench(fn_new, "new (S19c) again")

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19f] wrote {OUT}", flush=True)
