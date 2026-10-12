"""S10 GEMM autotune sweep v2 (single GPU, no mpirun) — NEW AXES.

S5 (tune_gemm.py) exhausted the plain tile-config space: 128x128x64/w4/s2
everywhere, deep-K at 0.89x vendor but wide-N/mid shapes stuck at 0.49-0.59x,
concluded "codegen quality, not config". This sweep attacks axes S5 never
touched:

  splitk     K-parallel split with fp32 atomic accumulation (+zero/cast
             overheads timed in) — for occupancy-starved skinny shapes
             (tile count < 2*NUM_CU).
  group      GROUP_SIZE_M L2-rasterization sweep on representative configs.
  persistent grid=NUM_CU tile loop (S5.2 said capped persistent loses for
             the AG consumer; one cell confirms for the plain kernel).
  cg/evict   cache_modifier=".cg" / eviction_policy="evict_first" on the
             streaming A/B loads.
  fp16acc    fp16 accumulator probe (mfma f16-acc) — perf datapoint only,
             loose tolerance.
  refine     tight re-grid around the S5 sweet spot incl. s=1 and the
             256-wide tiles that fit LDS.

Resumable: checkpoint per (shape, cell) into bench/results/tune_gemm2.json;
rerun skips done cells (pod restarts kill everything).

Env: export HSA_FORCE_FINE_GRAIN_PCIE=1
Run:  python3 bench/tune_gemm2.py
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
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tcache_tune_gemm2")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu.gemm_reduce_scatter import (  # noqa: E402
    kernel_gemm_rs_producer_local,
)

torch.cuda.set_device(0)
DTYPE = torch.float16
SCALE = 0.02
NUM_CU = torch.cuda.get_device_properties(0).multi_processor_count

SHAPES = [
    (4096, 2048, 2048),
    (2048, 2048, 2048),   # S8 k=2 skew shape / AR monolithic
    (2048, 4096, 4096),   # RS bench shape
    (2048, 8192, 4096),   # AG wide-N
    (2048, 1024, 8192),   # AG deep-K (S5: 0.89x vendor, near parity)
    (1024, 4096, 4096),   # worst relative gap in S5 (0.49x)
    (256, 4096, 4096),    # AG decode
    (128, 4096, 4096),    # S8 k=1 decode shape
    (8192, 29568, 8192),  # user-requested big prefill shape (N=231*128)
]

OUT = os.path.join(REPO, "bench", "results", "tune_gemm2.json")


# ------------------------------------------------------------------ kernels --
@triton.jit
def k_gemm_splitk(a_ptr, b_ptr, c32_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                  SPLITK: tl.constexpr):
    """C = A@B with the K range cut into SPLITK slices; partials accumulate
    via fp32 atomics into c32 (zeroed by the host each call), cast separately.
    Reuses the base kernel's group swizzle; K slices are BK-aligned."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    tile = pid % num_tiles
    kid = pid // num_tiles

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile % num_pid_in_group) % group_size_m)
    pid_n = (tile % num_pid_in_group) // group_size_m

    k_per = tl.cdiv(K, SPLITK * BLOCK_SIZE_K) * BLOCK_SIZE_K
    k0 = kid * k_per
    k1 = tl.minimum(k0 + k_per, K)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = k0 + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(k0, k1, BLOCK_SIZE_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    m = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.atomic_add(c32_ptr + stride_cm * offs_cm[:, None]
                  + stride_cn * offs_cn[None, :], accumulator, mask=m)


@triton.jit
def k_cast_f32f16(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=m).to(tl.float16),
             mask=m)


@triton.jit
def k_gemm_persistent(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn,
                      stride_cm, stride_cn,
                      BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                      BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                      GRID: tl.constexpr):
    """grid=GRID persistent tile loop. P7: the loop stride MUST equal the
    launched grid (GRID is constexpr == launch size)."""
    start = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    for tile in range(start, num_tiles, GRID):
        group_id = tile // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile % num_pid_in_group) % group_size_m)
        pid_n = (tile % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am
                          + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk
                          + offs_bn[None, :] * stride_bn)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        dtype = a_ptr.dtype.element_ty
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] \
            + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


@triton.jit
def k_gemm_cg(a_ptr, b_ptr, c_ptr, M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn,
              stride_cm, stride_cn,
              BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
              BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Base kernel with cache_modifier='.cg' on the streaming A/B loads."""
    pid = tl.program_id(axis=0)
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, cache_modifier=".cg")
        b = tl.load(b_ptrs, cache_modifier=".cg")
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    dtype = a_ptr.dtype.element_ty
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


@triton.jit
def k_gemm_evict(a_ptr, b_ptr, c_ptr, M, N, K,
                 stride_am, stride_ak, stride_bk, stride_bn,
                 stride_cm, stride_cn,
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                 BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Base kernel with eviction_policy='evict_first' on A/B loads."""
    pid = tl.program_id(axis=0)
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, eviction_policy="evict_first")
        b = tl.load(b_ptrs, eviction_policy="evict_first")
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    dtype = a_ptr.dtype.element_ty
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(dtype), mask=c_mask)


@triton.jit
def k_gemm_f16acc(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    """Base kernel with an fp16 accumulator (mfma f16-accumulate path).
    Perf datapoint only — K=4096 adds in fp16 drift; loose tolerance."""
    pid = tl.program_id(axis=0)
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                           dtype=a_ptr.dtype.element_ty)
    for _k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    dtype = a_ptr.dtype.element_ty
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ------------------------------------------------------------------- cells --
def tile_cells():
    """(key, variant, cfg) — the full v2 cell list (shape-independent)."""
    cells = []
    # shipped anchor: matmul_local's current choice rule (BM by M, at bench)
    cells.append(("refine_128x128x64_w4_s2_g8", "base",
                  dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                       GROUP_SIZE_M=8, num_warps=4, num_stages=2)))
    # refinement grid around the S5 sweet spot (+s1, +256-wide, +w8 pairs)
    for bm, bn, bk, w, s in itertools.product((64, 128, 256), (128, 256),
                                              (32, 64), (4, 8), (1, 2, 3)):
        if bm * bn > 32768 or (bm * bn < 8192):
            continue
        if (bm, bn, bk, w, s) == (128, 128, 64, 4, 2):
            continue  # == anchor
        cells.append((f"refine_{bm}x{bn}x{bk}_w{w}_s{s}_g8", "base",
                      dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                           GROUP_SIZE_M=8, num_warps=w, num_stages=s)))
    # GROUP rasterization on representative tiles
    for g in (1, 2, 4, 16):
        for bm, bn, bk, w in ((64, 128, 64, 4), (128, 128, 64, 4),
                              (128, 256, 32, 8)):
            cells.append((f"group_{bm}x{bn}x{bk}_w{w}_s2_g{g}", "base",
                          dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn,
                               BLOCK_SIZE_K=bk, GROUP_SIZE_M=g,
                               num_warps=w, num_stages=2)))
    # cache / eviction modifiers on the S5-best config
    best = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4, num_stages=2)
    cells.append(("cg_128x128x64_w4_s2", "cg", dict(best)))
    cells.append(("evict_128x128x64_w4_s2", "evict", dict(best)))
    cells.append(("f16acc_128x128x64_w4_s2", "f16acc", dict(best)))
    # persistent (GRID set per shape at bench time)
    for bm, bn, bk, w in ((64, 128, 64, 4), (128, 128, 64, 4)):
        cells.append((f"persist_{bm}x{bn}x{bk}_w{w}_s2", "persist",
                      dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                           GROUP_SIZE_M=8, num_warps=w, num_stages=2)))
    # split-K (only benched where tiles are scarce; see want_splitk)
    for sk, (bm, bn, bk, w) in itertools.product(
            (2, 4, 8), ((64, 128, 64, 4), (128, 128, 64, 4),
                        (128, 128, 32, 4))):
        cells.append((f"splitk{sk}_{bm}x{bn}x{bk}_w{w}_s2", "splitk",
                      dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                           GROUP_SIZE_M=8, num_warps=w, num_stages=2,
                           SPLITK=sk)))
    return cells


CELLS = tile_cells()
KERNELS = {"base": kernel_gemm_rs_producer_local, "cg": k_gemm_cg,
           "evict": k_gemm_evict, "f16acc": k_gemm_f16acc,
           "persist": k_gemm_persistent, "splitk": k_gemm_splitk}


def want_splitk(M, N, cfg):
    """Split-K pays only when the plain grid under-fills the device."""
    tiles = triton.cdiv(M, 128) * triton.cdiv(N, 128)
    return tiles < 2 * NUM_CU


def bench_cell(a, b, c, c32, M, N, K, variant, cfg):
    strides = (a.stride(0), a.stride(1), b.stride(1), b.stride(0),
               c.stride(0), c.stride(1))
    tol = 5e-2 if variant == "f16acc" else 2e-2

    if variant == "splitk":
        sk = cfg["SPLITK"]
        tiles = triton.cdiv(M, cfg["BLOCK_SIZE_M"]) \
            * triton.cdiv(N, cfg["BLOCK_SIZE_N"])
        grid = (tiles * sk,)
        cast_grid = (triton.cdiv(M * N, 8192),)
        n = M * N

        def run():
            c32.zero_()
            KERNELS["splitk"][grid](a, b, c32, c, M, N, K, *strides, **cfg)
            k_cast_f32f16[cast_grid](c32, c, n, BLOCK=8192)
    else:
        bm, bn = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"]
        tiles = triton.cdiv(M, bm) * triton.cdiv(N, bn)
        if variant == "persist":
            gridsize = min(NUM_CU, tiles)
            cfg = dict(cfg, GRID=gridsize)
            grid = (gridsize,)
        else:
            grid = (tiles,)
        run = lambda: KERNELS[variant][grid](a, b, c, M, N, K, *strides, **cfg)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        run()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    c.zero_()
    run()
    torch.cuda.synchronize()
    md = (c.float() - REF['v']).abs().max().item()
    return statistics.median(sorted(ts)), md, tol


REF = {}


def main():
    torch.manual_seed(9000)
    results = {}
    if os.path.exists(OUT):  # resume
        with open(OUT) as f:
            results = json.load(f)
    results.setdefault("env", {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "gpu": torch.cuda.get_device_name(0), "num_cu": NUM_CU,
        "dtype": "fp16"})
    shapes_done = results.setdefault("shapes", {})

    for (M, N, K) in SHAPES:
        skey = f"{M}x{N}x{K}"
        sa = shapes_done.setdefault(skey, {"rows": {}})
        a = (torch.randn(M, K, device="cuda") * SCALE).half()
        b = (torch.randn(N, K, device="cuda") * SCALE).half()
        c = torch.empty(M, N, dtype=DTYPE, device="cuda")
        c32 = torch.empty(M, N, dtype=torch.float32, device="cuda")
        with torch.no_grad():
            REF['v'] = a.float() @ b.float().t()

        # vendor + shipped-rule anchors (recomputed each resume, cheap)
        if "vendor" not in sa:
            ts = []
            for _ in range(20):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                torch.matmul(a, b.t(), out=c)
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            sa["vendor"] = statistics.median(ts)
            print(f"[{skey}] vendor {sa['vendor']:.3f} ms", flush=True)

        for key, variant, cfg in CELLS:
            if key in sa["rows"]:
                continue
            if variant == "splitk" and not want_splitk(M, N, cfg):
                continue
            try:
                ms, md, tol = bench_cell(a, b, c, c32, M, N, K, variant, cfg)
                if md > tol:
                    print(f"[{skey}] {key}: WRONG md={md:.1e}", flush=True)
                    sa["rows"][key] = {"ms": None, "md": md}
                else:
                    sa["rows"][key] = {"ms": ms, "md": md}
            except Exception as ex:
                if "out of resource" not in str(ex).lower():
                    print(f"[{skey}] {key}: {type(ex).__name__}: {ex}",
                          flush=True)
                sa["rows"][key] = {"ms": None, "err": str(ex)[:80]}
            with open(OUT, "w") as f:  # checkpoint every cell
                json.dump(results, f, indent=1)

        rows = sorted((v["ms"], k) for k, v in sa["rows"].items()
                      if v.get("ms"))
        print(f"\n=== {skey}  vendor={sa['vendor']:.3f} ms", flush=True)
        for ms, k in rows[:8]:
            print(f"  {k:>28} {ms:7.3f} ms  "
                  f"({sa['vendor'] / ms:.2f}x vs vendor)", flush=True)
        del a, b, c, c32
        REF.clear()
        torch.cuda.empty_cache()

    print(f"\nresults -> {OUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
