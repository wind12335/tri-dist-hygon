"""S19b: peel the consumer's 9.6ms structure cost piecewise.

S19 established (single card, M8192 N29568 K8192, config 128x128x64 w4
s2 g8): plain 21.14ms, consumer-with-prearmed-gate 30.74ms, gate itself
~0. The 45% lives in kernel CODE structure. This probe runs a
parametrized copy of the consumer (gate stripped) over the 2^3 toggle
space of the remaining suspects:

  ACC3   acc = tl.dot(a, b, acc) fused-accumulator form (vs acc += ...)
  NODUAL single A path (always ws; drops localA branch + merged stride)
  NOSWZ  direct pid_m (drops the rank-swizzle div/mod chain)

run: export HSA_FORCE_FINE_GRAIN_PCIE=1; python3 bench/s19b_probe_variants.py
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
DTYPE = torch.float16
WARMUP, ITERS = 3, 7
R = 2
OUT = os.path.join(REPO, "bench", "results", "s19b_variants.json")
RES = {"M": M, "N": N, "K": K, "config": "128x128x64 w4 s2 g8", "arms": {}}

torch.manual_seed(42)
ws = (torch.randn(M, K, device="cuda") * 0.02).half()
a_loc = ws[:M // 2]
b = (torch.randn(N, K, device="cuda") * 0.02).half()
c = torch.empty(M, N, dtype=DTYPE, device="cuda")
bf = b.float()
ref = torch.empty(M, N, dtype=DTYPE, device="cuda")
for r0 in range(0, M, 1024):
    ref[r0:r0 + 1024] = (ws[r0:r0 + 1024].float() @ bf.t()).half()
del bf
torch.cuda.empty_cache()
print("[s19b] tensors + ref ready", flush=True)


def maxdiff():
    md = 0.0
    for r0 in range(0, M, 2048):
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref[r0:r0 + 2048].float()).abs().max().item())
    return md


@triton.jit(do_not_specialize=["rank", "world_size", "m_chunk_num"])
def k_cons_v(A, localA, B, C, M, N, K,
             stride_am, stride_ak, stride_lam, stride_lak,
             stride_bk, stride_bn, stride_cm, stride_cn,
             rank, world_size, m_chunk_num,
             BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
             BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
             ACC3: tl.constexpr, NODUAL: tl.constexpr, NOSWZ: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    M_per_rank = M // world_size
    pid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    A_ptr = A.to(tl.pointer_type(localA.dtype.element_ty))

    for tile_id in range(pid, total_tiles, total_tiles):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        if NOSWZ:
            pass  # keep pid_m as-is
        else:
            num_pid_m_per_copy_chunk = M_PER_CHUNK_V // BLOCK_SIZE_M
            chunk_offset = pid_m // (num_pid_m_per_copy_chunk * world_size)
            rank_offset = pid_m % (num_pid_m_per_copy_chunk
                                   * world_size) // num_pid_m_per_copy_chunk
            block_offset = pid_m % num_pid_m_per_copy_chunk
            rank_offset = (rank_offset + rank) % world_size
            pid_m = (rank_offset * M_per_rank + chunk_offset * M_PER_CHUNK_V
                     + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M

        offs_rank = pid_m // pid_m_per_rank

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        if NODUAL:
            # single path: everything from the gathered workspace
            A_BASE = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
            a_kstride = stride_ak
        else:
            if offs_rank == rank:
                rml = rm % M_per_rank
                A_BASE = (localA + rml[:, None] * stride_lam
                          + rk[None, :] * stride_lak)
                a_kstride = stride_lak
            else:
                A_BASE = (A_ptr + rm[:, None] * stride_am
                          + rk[None, :] * stride_ak)
                a_kstride = stride_ak

        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b_ = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            if ACC3:
                acc = tl.dot(a, b_, acc)
            else:
                acc += tl.dot(a, b_)
            A_BASE += BLOCK_SIZE_K * a_kstride
            B_BASE += BLOCK_SIZE_K * stride_bk

        cm = acc.to(C.dtype.element_ty)
        rm2 = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn2 = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm2 = tl.max_contiguous(tl.multiple_of(rm2, BLOCK_SIZE_M),
                                BLOCK_SIZE_M)
        rn2 = tl.max_contiguous(tl.multiple_of(rn2, BLOCK_SIZE_N),
                                BLOCK_SIZE_N)
        c_mask = (rm2[:, None] < M) & (rn2[None, :] < N)
        C_ = C + rm2[:, None] * stride_cm + rn2[None, :] * stride_cn
        tl.store(C_, cm, c_mask)


M_PER_CHUNK_V = tl.constexpr(512)  # triton-capturable global


def launch(acc3, nodual, noswz):
    n_chunks = 8
    CR = (M // R) // n_chunks
    tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)

    def fn():
        k_cons_v[(tiles,)](
            ws.data_ptr(), a_loc, b, c, M, N, K,
            K, 1, K, 1, 1, K, N, 1,
            0, R, n_chunks,
            BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
            GROUP_SIZE_M=8,
            ACC3=acc3, NODUAL=nodual, NOSWZ=noswz,
            num_warps=4, num_stages=2)
    return fn


for acc3 in (False, True):
    for nodual in (False, True):
        for noswz in (False, True):
            tag = (f"acc3_{int(acc3)}_nodual_{int(nodual)}"
                   f"_noswz_{int(noswz)}")
            fn = launch(acc3, nodual, noswz)
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
            print(f"[s19b] {tag:28s} p50 {p50:7.2f} ms  md {md:.1e}",
                  flush=True)
            assert md < 1e-2, f"{tag} WRONG"

with open(OUT, "w") as f:
    json.dump(RES, f, indent=1)
print(f"[s19b] wrote {OUT}", flush=True)
