"""S19g: fold world_size/m_chunk_num/rank as constexpr in the
restructured consumer (they are shape/process-keyed, never per-round).
The swizzle div/mod chain divides by these; do_not_specialize keeps
them runtime -> never folded. Probe k_cons_g = k_cons_r + constexpr
folding. (flags_base/round MUST stay runtime -- they change per round.)

run: export HSA_FORCE_FINE_GRAIN_PCIE=1; python3 bench/s19g_fold_probe.py
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

M, N, K = 8192, 29568, 8192
R = 2
WARMUP, ITERS = 3, 7
OUT = os.path.join(REPO, "bench", "results", "s19g_fold_probe.json")
RES = {"M": M, "N": N, "K": K, "arms": {}}

torch.manual_seed(42)
ws = (torch.randn(M, K, device="cuda") * 0.02).half()
a_loc = ws[:M // 2]
b = (torch.randn(N, K, device="cuda") * 0.02).half()
c = torch.empty(M, N, dtype=torch.float16, device="cuda")
bf = b.float()
ref = torch.empty(M, N, dtype=torch.float16, device="cuda")
for r0 in range(0, M, 1024):
    ref[r0:r0 + 1024] = (ws[r0:r0 + 1024].float() @ bf.t()).half()
del bf
torch.cuda.empty_cache()
print("[s19g] tensors + ref ready", flush=True)


def maxdiff():
    md = 0.0
    for r0 in range(0, M, 2048):
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref[r0:r0 + 2048].float()).abs().max().item())
    return md


@triton.jit(do_not_specialize=["flags_base", "round"])
def k_cons_g(A, localA, B, C, M, N, K,
             stride_k, stride_bk, stride_bn, stride_cm, stride_cn,
             m_chunk_num,
             flags_base, round,
             BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
             BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
             M_PER_CHUNK: tl.constexpr, R_C: tl.constexpr,
             RANK_C: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = pid

    M_per_rank = M // R_C
    pid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    A_ptr = A.to(tl.pointer_type(localA.dtype.element_ty))
    flag_ptr = flags_base.to(tl.pointer_type(tl.int32))

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m

    num_pid_m_per_copy_chunk = M_PER_CHUNK // BLOCK_SIZE_M
    chunk_offset = pid_m // (num_pid_m_per_copy_chunk * R_C)
    rank_offset = (pid_m % (num_pid_m_per_copy_chunk * R_C)
                   // num_pid_m_per_copy_chunk)
    block_offset = pid_m % num_pid_m_per_copy_chunk
    rank_offset = (rank_offset + RANK_C) % R_C
    pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_CHUNK
             + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M

    offs_am = pid_m * BLOCK_SIZE_M
    offs_sig = offs_am // M_PER_CHUNK
    offs_rank = pid_m // pid_m_per_rank

    if offs_rank != RANK_C:
        expected = round + 1 + offs_sig % m_chunk_num
        fp = flag_ptr + offs_sig
        v = tl.atomic_add(fp, 0)
        while v < expected:
            for _ in range(128):
                v = tl.load(fp, volatile=True)
            if v < expected:
                v = tl.atomic_add(fp, 0)

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    if offs_rank == RANK_C:
        a_base = localA
        row_bias = RANK_C * M_per_rank
    else:
        a_base = A_ptr
        row_bias = 0
    A_BASE = (a_base + (rm - row_bias)[:, None] * stride_k + rk[None, :])
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
        b_ = tl.load(tl.multiple_of(B_BASE, (16, 1)))
        acc = tl.dot(a, b_, acc)
        A_BASE += BLOCK_SIZE_K
        B_BASE += BLOCK_SIZE_K * stride_bk

    cm = acc.to(C.dtype.element_ty)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_, cm, c_mask)


def launch():
    n_chunks = 8
    CR = (M // R) // n_chunks
    flags = torch.full((R * n_chunks,), 1 << 28, dtype=torch.int32,
                       device="cuda")
    tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)

    def fn():
        k_cons_g[(tiles,)](
            ws.data_ptr(), a_loc, b, c, M, N, K,
            K, 1, K, N, 1,
            n_chunks,
            flags.data_ptr(), 0,
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8, M_PER_CHUNK=CR, R_C=R, RANK_C=0,
            num_warps=4, num_stages=2)
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
    print(f"[s19g] {tag:34s} p50 {p50:7.2f} ms  md {md:.1e}", flush=True)
    assert md < 1e-2, f"{tag} WRONG md={md}"
    return p50


bench(launch(), "folded R/RANK constexpr")
bench(launch(), "folded again")

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19g] wrote {OUT}; ref: s19c k_cons_r 25.23, patched prod 26.18",
      flush=True)
