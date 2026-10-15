"""S19 single-card probe: decompose the consumer-form GEMM overhead.

S18 fact: at M8192 N29568 K8192 the fused consumer's gemm-only arm costs
39.6ms while plain triton with the IDENTICAL tile config (128x128x64 w4
s2 g8) runs 21.1ms -> 1.87x lives in the kernel CODE structure, not the
config. This probe peels the consumer extras apart on ONE GPU (no MPI):

  vendor    torch.matmul                          (~11ms ceiling)
  plain     standard grouped GEMM (tune_gemm2 base kernel body)
  cons_r1   consumer kernel, world_size=1: keeps rank-swizzle pid math,
            % indexing, max_contiguous hints, EVEN_K tail; drops the
            flag gate + dual-A-path (always localA)
  cons_r2   consumer kernel, world_size=2, flags PRE-ARMED huge so the
            spin exits after one atomic: adds per-tile flag atomic +
            dual-A-path branch + gate arithmetic
  sweep     config sweep over the same kernels (BN locked to 128: the
            consumer asserts N%BN==0 and 256 does not divide 29568)

decomposition: cons_r2 - cons_r1 = gate cost; cons_r1 - plain =
swizzle/index/tail structure cost.

run (single card, NO mpirun):
  cd /root/private_data/tri-dist-hygon && export HSA_FORCE_FINE_GRAIN_PCIE=1
  python3 bench/s19_probe_consumer_gemm.py            # all arms + sweep
  S19_ARMS=plain,cons_r2 python3 ...                  # subset
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
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu.gemm_ag_tr import (  # noqa: E402
    kernel_gemm_ag_tr_consumer)

M, N, K = 8192, 29568, 8192
DTYPE = torch.float16
WARMUP, ITERS = 3, 7
ARMS = [a for a in os.environ.get("S19_ARMS", "vendor,plain,cons_r1,cons_r2"
                                   ).split(",") if a]
OUT = os.path.join(REPO, "bench", "results", "s19_consumer_probe.json")
RES = {"M": M, "N": N, "K": K, "arms": {}, "sweep": {}}

torch.manual_seed(42)
ws = (torch.randn(M, K, device="cuda") * 0.02).half()   # gathered A stand-in
a_loc = ws[:M // 2]                                      # own shard view
b = (torch.randn(N, K, device="cuda") * 0.02).half()
c = torch.empty(M, N, dtype=DTYPE, device="cuda")
bf = b.float()
print(f"[s19] tensors ready ({torch.cuda.max_memory_allocated()/2**30:.1f}GB)",
      flush=True)

ref = torch.empty(M, N, dtype=DTYPE, device="cuda")
for r0 in range(0, M, 1024):
    ref[r0:r0 + 1024] = (ws[r0:r0 + 1024].float() @ bf.t()).half()
del bf
torch.cuda.empty_cache()
print("[s19] ref built", flush=True)


def maxdiff():
    md = 0.0
    for r0 in range(0, M, 2048):
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref[r0:r0 + 2048].float()).abs().max().item())
    return md


def tloop(fn, tag):
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
    print(f"[s19] {tag:34s} p50 {p50:7.2f} ms  "
          f"{2*M*N*K/1e9/p50:6.0f} GFLOP/ms", flush=True)
    return p50


# ------------------------------------------------------------ plain ----
@triton.jit
def k_gemm_plain(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                 BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                 EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
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

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def launch_plain(BM=128, BN=128, BK=64, G=8, W=4, S=2):
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)

    def fn():
        k_gemm_plain[(tiles,)](
            ws, b, c, M, N, K,
            K, 1, 1, K, N, 1,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=G, EVEN_K=(K % BK == 0),
            num_warps=W, num_stages=S)
    return fn


def launch_cons(R, BM=128, BN=128, BK=64, G=8, W=4, S=2):
    """world_size=R consumer launch, flags pre-armed (gate exits fast)."""
    m_pr = M // R
    n_chunks = 8  # chunk_rows 512 at K=8192 (auto rule)
    CR = m_pr // n_chunks
    flags = torch.full((R * n_chunks,), 1 << 28, dtype=torch.int32,
                       device="cuda")
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)

    def fn():
        kernel_gemm_ag_tr_consumer[(tiles,)](
            ws.data_ptr(),        # gathered A (int64 VA like the real op)
            a_loc, b, c,
            M, N, K,
            K, 1,                # gathered strides
            a_loc.stride(0), a_loc.stride(1),
            1, K,                # b is [N, K]: stride_bk, stride_bn
            c.stride(0), c.stride(1),
            0, R, n_chunks,
            flags.data_ptr(),
            0,
            BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK,
            GROUP_SIZE_M=G, M_PER_CHUNK=CR, NUM_SMS=tiles,
            EVEN_K=(K % BK == 0),
            num_warps=W, num_stages=S)
    return fn


def record(tag, p50, md):
    RES["arms"][tag] = {"p50": round(p50, 3), "md": md}
    assert md < 1e-2, f"{tag} WRONG md={md}"


if "vendor" in ARMS:
    p50 = tloop(lambda: torch.matmul(ws, b.t(), out=c), "vendor torch.matmul")
    record("vendor", p50, maxdiff())
if "plain" in ARMS:
    p50 = tloop(launch_plain(), "plain 128x128x64 w4 s2 g8")
    record("plain", p50, maxdiff())
if "cons_r1" in ARMS:
    p50 = tloop(launch_cons(1), "cons_r1 (no gate, keep swizzle/tail)")
    record("cons_r1", p50, maxdiff())
if "cons_r2" in ARMS:
    p50 = tloop(launch_cons(2), "cons_r2 (pre-armed gate + dual path)")
    record("cons_r2", p50, maxdiff())

# ----------------------------------------------------------- sweep -----
SWEEP = [
    # (kernel, R, BM, BN, BK, G, W, S)  -- BN=128 locked (256∤29568)
    ("plain", 0, 128, 128, 64, 8, 4, 3),
    ("plain", 0, 128, 128, 64, 8, 8, 2),
    ("plain", 0, 128, 128, 128, 8, 4, 2),
    ("plain", 0, 128, 128, 32, 8, 4, 3),
    ("plain", 0, 128, 128, 64, 4, 4, 2),
    ("plain", 0, 256, 128, 64, 8, 8, 2),
    ("cons", 2, 128, 128, 64, 8, 4, 3),
    ("cons", 2, 128, 128, 64, 8, 8, 2),
    ("cons", 2, 128, 128, 128, 8, 4, 2),
    ("cons", 2, 128, 128, 32, 8, 4, 3),
    ("cons", 2, 128, 128, 64, 4, 4, 2),
    ("cons", 2, 256, 128, 64, 8, 8, 2),
]
for kn, R, BM, BN, BK, G, W, S in SWEEP:
    tag = f"{kn}{R or ''}_{BM}x{BN}x{BK}_w{W}_s{S}_g{G}"
    try:
        fn = (launch_plain(BM, BN, BK, G, W, S) if kn == "plain"
              else launch_cons(R, BM, BN, BK, G, W, S))
        fn()
        torch.cuda.synchronize()
        md = maxdiff()
        p50 = tloop(fn, tag)
        RES["sweep"][tag] = {"p50": round(p50, 3), "md": md}
        assert md < 1e-2, f"{tag} WRONG md={md}"
    except Exception as ex:
        RES["sweep"][tag] = {"ERR": str(ex)[:200]}
        print(f"[s19] {tag} ERR {str(ex)[:120]}", flush=True)

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19] wrote {OUT}", flush=True)
print("[s19] decomposition: gate = cons_r2 - cons_r1; "
      "structure = cons_r1 - plain", flush=True)
