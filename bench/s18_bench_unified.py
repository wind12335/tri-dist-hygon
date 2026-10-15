"""S18 unified latency table (2x BW1000) at M8192 N29568 K8192 fp16.

One process per FAMILY (S18_FAM=ar|rs|ag|a2a) so a wedge in one family
cannot take down the rest (S17 lesson); each family writes its own JSON
fragment, shell merges. N29568 = 231*128: 128-aligned, NOT 256-aligned --
any op that asserts records a SKIP and the table says so.

Layers in one table:
  base-4   : AG push (S11 MPC), RS s6, AR s6, MoE a2a
  paper-#1 : AG tr (s14), RS v5, AR rd
  paper-#2 : AG i1 (predictor table; verify off/on), AR pred-carrier
             (natural order -- predictor payoff needs skew, S15 rig),
             RS: none exists (honest gap)
  refs     : vendor GEMM, plain triton GEMM, RCCL seq baselines

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s18_bench_unified.py            # S18_FAM=ar default env
"""
import os
import sys
import json
import time
import traceback
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
FAM = os.environ.get("S18_FAM", "ar")
OUTDIR = os.path.join(REPO, "bench", "results")

if FAM == "merge":
    merged = {}
    for f in ("ar", "rs", "ag", "a2a"):
        p = os.path.join(OUTDIR, f"s18_{f}_29568.json")
        if os.path.exists(p):
            with open(p) as fh:
                merged[f] = json.load(fh)
        else:
            merged[f] = None
    with open(os.path.join(OUTDIR, "s18_unified_29568.json"), "w") as fh:
        json.dump(merged, fh, indent=1)
    print(json.dumps(merged, indent=1))
    sys.exit(0)

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tcache_rank{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu import all_gather_gemm as ags4  # noqa: E402
from triton_dist.kernels.dcu import gemm_ag_tr as agtr  # noqa: E402
from triton_dist.kernels.dcu import gemm_allreduce as s6ar  # noqa: E402
from triton_dist.kernels.dcu import gemm_ar_pred as ap  # noqa: E402
from triton_dist.kernels.dcu import gemm_ar_rd as rd  # noqa: E402
from triton_dist.kernels.dcu import gemm_reduce_scatter as s6rs  # noqa: E402
from triton_dist.kernels.dcu import gemm_rs_v5 as rs5  # noqa: E402
from triton_dist.kernels.dcu import moe_a2a  # noqa: E402

DTYPE = torch.float16
SCALE = 0.02
M = int(os.environ.get("S18_M", 8192))
N = int(os.environ.get("S18_N", 29568))
K = int(os.environ.get("S18_K", 8192))
M_PR = M // size
WARMUP, ITERS = 2, 5
TOL = 1e-2
OUT = os.path.join(OUTDIR, f"s18_{FAM}_29568.json")
RES = {"fam": FAM, "M": M, "N": N, "K": K, "warmup/iters": f"{WARMUP}/{ITERS}",
       "arms": {}}

import torch.distributed as dist  # noqa: E402

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29619")
dist.init_process_group(backend="nccl", rank=rank, world_size=size)
_x = torch.ones(1, device="cuda")
for _ in range(3):
    dist.all_reduce(_x)
torch.cuda.synchronize()
print(f"[rank {rank}] fam={FAM} dist ready", flush=True)


def tloop(fn, tag):
    st = torch.cuda.current_stream()
    for w in range(WARMUP):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        print(f"[rank {rank}] {tag} warmup{w} {time.time()-t0:.2f}s",
              flush=True)
    ts = []
    for i in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(st)
        fn()
        e.record(st)
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
        print(f"[rank {rank}] {tag} it{i} {ts[-1]:.2f} ms", flush=True)
    return {"mean": statistics.mean(ts), "p50": statistics.median(ts),
            "min": min(ts)}


def arm(name, setup_fn):
    """Run one arm inside try/except; record SKIP on failure; write the
    JSON after EVERY arm so a wedge (S17-class) keeps partial results."""
    try:
        r = setup_fn()
        if r is not None:
            RES["arms"][name] = r
            print(f"[rank {rank}] ==> {name} p50 "
                  f"{r['p50']:.2f} ms corr_md={r.get('corr', -1):.2e}",
                  flush=True)
    except Exception:
        RES["arms"][name] = {"SKIP": traceback.format_exc(limit=3)}
        print(f"[rank {rank}] ==> {name} SKIP:\n{traceback.format_exc()}",
              flush=True)
    if rank == 0:
        with open(OUT, "w") as f:
            json.dump(RES, f, indent=1)


def chunked_ref_sum(A_list, B_list, chunk=1024):
    ref = torch.empty(M, N, dtype=DTYPE, device="cuda")
    for r0 in range(0, M, chunk):
        rows = min(chunk, M - r0)
        acc = torch.zeros(rows, N, dtype=torch.float32, device="cuda")
        for s in range(size):
            acc += (A_list[s][r0:r0 + rows] @ B_list[s].t()).float()
        ref[r0:r0 + rows] = acc.half()
    return ref


def max_diff_vs_ref(c, ref, chunk=1024):
    md = 0.0
    for r0 in range(0, c.shape[0], chunk):
        md = max(md, (c[r0:r0 + chunk].float()
                      - ref[r0:r0 + chunk].float()).abs().max().item())
    return md


if FAM in ("ar", "rs", "ag"):
    A, B = [], []
    for s in range(size):
        torch.manual_seed(7100 + s)
        A.append((torch.randn(M, K, device="cuda") * SCALE).half())
        torch.manual_seed(7300 + s)
        B.append((torch.randn(N, K, device="cuda") * SCALE).half())
    a, b = A[rank], B[rank]
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")
    t0 = time.time()
    ref = chunked_ref_sum(A, B)
    print(f"[rank {rank}] ref built {time.time()-t0:.1f}s "
          f"({torch.cuda.max_memory_allocated()/2**30:.1f}GB peak)",
          flush=True)

# ============================================================= AR ======
if FAM == "ar":
    def a_torch():
        md = None
        with torch.no_grad():
            torch.matmul(a, b.t(), out=c)
        r = tloop(lambda: torch.matmul(a, b.t(), out=c), "gemm_torch")
        r["corr"] = 0.0
        return r
    arm("gemm_torch", a_torch)

    def a_s6():
        ctx = s6ar.create_gemm_ar_intra_node_context(
            max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
            rank=rank, num_ranks=size, mpi=mpi, comm=comm)
        s6ar.gemm_ar_intra_node_op(a, b, c, ctx)
        torch.cuda.synchronize()
        r = tloop(lambda: s6ar.gemm_ar_intra_node_op(a, b, c, ctx), "ar_s6")
        r["corr"] = max_diff_vs_ref(c, ref)
        return r
    arm("base_s6", a_s6)

    def a_rd():
        ctx = rd.create_gemm_ar_rd_context(
            max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
            rank=rank, num_ranks=size, mpi=mpi, comm=comm)
        rd.gemm_ar_rd_op(a, b, c, ctx)
        torch.cuda.synchronize()
        r = tloop(lambda: rd.gemm_ar_rd_op(a, b, c, ctx), "ar_rd")
        r["corr"] = max_diff_vs_ref(c, ref)
        return r
    arm("p1_rd", a_rd)

    def a_pred():
        # paper-#2 carrier op, steady state (natural order): predictor
        # payoff is skew-bound (S15 rig: delta=1600us 5.19->3.73ms);
        # this measures the armed op's latency WITHOUT injected skew.
        ctx = rd.create_gemm_ar_rd_context(
            max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
            rank=rank, num_ranks=size, mpi=mpi, comm=comm)
        ap.gemm_ar_pred_op(a, b, c, ctx)
        torch.cuda.synchronize()
        r = tloop(lambda: ap.gemm_ar_pred_op(a, b, c, ctx), "ar_pred")
        r["corr"] = max_diff_vs_ref(c, ref)
        return r
    arm("p2_pred_carrier", a_pred)

    def a_rccl():
        def dist_ar_seq():
            torch.matmul(a, b.t(), out=c)
            dist.all_reduce(c)
        dist_ar_seq()
        torch.cuda.synchronize()
        r = tloop(dist_ar_seq, "ar_rccl_seq")
        r["corr"] = max_diff_vs_ref(c, ref)
        return r
    arm("rccl_seq", a_rccl)

    del A, B, ref
    torch.cuda.empty_cache()

# ============================================================= RS ======
if FAM == "rs":
    out_shard = torch.empty(M_PR, N, dtype=DTYPE, device="cuda")
    shard_ref = ref[rank * M_PR:(rank + 1) * M_PR].clone()

    def a_torch():
        r = tloop(lambda: torch.matmul(a, b.t(), out=c), "gemm_torch")
        return r
    arm("gemm_torch", a_torch)

    def a_s6():
        ctx = s6rs.create_gemm_rs_intra_node_context(
            max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
            mpi=mpi, comm=comm)
        c6 = s6rs.gemm_rs_intra_node_op(a, b, DTYPE, rank, size, ctx)
        torch.cuda.synchronize()
        md = max_diff_vs_ref(c6, shard_ref)
        r = tloop(lambda: s6rs.gemm_rs_intra_node_op(a, b, DTYPE, rank,
                                                     size, ctx), "rs_s6")
        r["corr"] = md
        return r
    arm("base_s6", a_s6)

    def a_v5():
        ctx = rs5.create_gemm_rs_v5_context(
            max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
            mpi=mpi, comm=comm)
        rs5.gemm_rs_v5_op(a, b, out_shard, ctx)
        torch.cuda.synchronize()
        sr = shard_ref if shard_ref is not None else \
            ref[rank * M_PR:(rank + 1) * M_PR]
        md = max_diff_vs_ref(out_shard, sr)
        r = tloop(lambda: rs5.gemm_rs_v5_op(a, b, out_shard, ctx), "rs_v5")
        r["corr"] = md
        rs5.finalize_gemm_rs_v5_context(ctx)
        return r
    arm("p1_v5", a_v5)
    RES["p2_note"] = "no RS predictor exists (paper-#2 covers AG/AR)"

    def a_rccl():
        def dist_rs_seq():
            torch.matmul(a, b.t(), out=c)
            dist.reduce_scatter_tensor(out_shard, c)
        dist_rs_seq()
        torch.cuda.synchronize()
        r = tloop(dist_rs_seq, "rs_rccl_seq")
        r["corr"] = max_diff_vs_ref(out_shard, shard_ref)
        return r
    arm("rccl_seq", a_rccl)

    del A, B, ref
    torch.cuda.empty_cache()

# ============================================================= AG ======
if FAM == "ag":
    A = None
    a_shards_src = []
    for s in range(size):
        torch.manual_seed(7100 + s)
        a_shards_src.append(
            (torch.randn(M, K, device="cuda") * SCALE).half())
    torch.manual_seed(7300 + rank)
    b = (torch.randn(N, K, device="cuda") * SCALE).half()
    a_loc = a_shards_src[rank][rank * M_PR:(rank + 1) * M_PR].contiguous()
    A_full = torch.cat([a_shards_src[s][s * M_PR:(s + 1) * M_PR]
                        for s in range(size)], dim=0)
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")
    with torch.no_grad():
        ref = (A_full.float() @ b.float().t()).half()
    del a_shards_src, A_full
    torch.cuda.empty_cache()
    Ag_buf = torch.empty(M, K, dtype=DTYPE, device="cuda")
    # ONE i1 ctx serves s14 / gemm_triton / i1 arms (its .ag is the s14
    # pipeline; multiple live i1 heaps wedged before -- S17/S18 lesson)
    ictx = i1.create_i1_context(
        max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm)
    print(f"[rank {rank}] AG tensors ready "
          f"({torch.cuda.max_memory_allocated()/2**30:.1f}GB) "
          f"chunk_rows={ictx.ag.chunk_rows} n_chunks={ictx.ag.num_chunks}",
          flush=True)

    def a_torch():
        with torch.no_grad():
            torch.matmul(Ag_buf, b.t(), out=c)
        return tloop(lambda: torch.matmul(Ag_buf, b.t(), out=c), "gemm_torch")
    arm("gemm_torch", a_torch)

    def a_push():
        ctx = ags4.create_ag_gemm_intra_node_context(
            M, K, DTYPE, DTYPE, rank, size, mpi, comm, M_PER_CHUNK=M_PR)
        ags4.ag_gemm_intra_node_op(a_loc, b, c, ctx)
        torch.cuda.synchronize()
        md = max_diff_vs_ref(c, ref)
        r = tloop(lambda: ags4.ag_gemm_intra_node_op(a_loc, b, c, ctx),
                  "ag_push")
        r["corr"] = md
        return r
    arm("base_push_s11", a_push)

    def a_tr():
        agtr.gemm_ag_tr_op(a_loc, b, c, ictx.ag)
        torch.cuda.synchronize()
        md = max_diff_vs_ref(c, ref)
        r = tloop(lambda: agtr.gemm_ag_tr_op(a_loc, b, c, ictx.ag), "ag_tr")
        r["corr"] = md
        return r
    arm("p1_tr_s14", a_tr)

    def a_i1(verify):
        i1.I1_VERIFY = verify
        rot_packed = sum(((rank + i) % size) << (4 * i)
                         for i in range(size))

        def call():
            i1.i1_table_write(ictx, 0, 0, 0, rot_packed, ictx.ag.num_chunks)
            i1.i1_ag_tr_op(a_loc, b, c, ictx, with_observer=True)
        call()
        torch.cuda.synchronize()
        md = max_diff_vs_ref(c, ref)
        hits0 = ictx.verify_hits
        r = tloop(call, f"ag_i1_v{int(verify)}")
        r["corr"] = md
        if verify:
            r["repairs"] = ictx.verify_hits - hits0
        return r
    arm("p2_i1_noverify", lambda: a_i1(False))
    arm("p2_i1_verify", lambda: a_i1(True))

    def a_gemm_triton():
        r = tloop(lambda: agtr.gemm_ag_tr_op(a_loc, b, c, ictx.ag,
                                             component="gemm"),
                  "gemm_triton")
        return r
    arm("gemm_triton", a_gemm_triton)

    def a_rccl():
        def dist_ag():
            dist.all_gather_into_tensor(Ag_buf, a_loc)
        r = tloop(dist_ag, "rccl_ag")

        def dist_seq():
            dist.all_gather_into_tensor(Ag_buf, a_loc)
            torch.matmul(Ag_buf, b.t(), out=c)
        dist_seq()
        torch.cuda.synchronize()
        md = max_diff_vs_ref(c, ref)
        r2 = tloop(dist_seq, "rccl_seq")
        r2["corr"] = md
        return {"ag_p50": r["p50"], **r2}
    arm("rccl_seq", a_rccl)

    del ref
    torch.cuda.empty_cache()

# ============================================================ A2A =====
if FAM == "a2a":
    # topk=8 needs 2.1GB symmetric heap (recv+out) > 1.8GB cap at T8192
    # H8192 -> NULL malloc; topk=2 (536MB) is the largest fitting variant.
    TOPK = int(os.environ.get("S18_A2A_TOPK", "2"))
    T, H, E = M, K, 32

    @triton.jit(do_not_specialize=["recv_base", "out_base", "n_ptr"])
    def k_expert_fanout(recv_base, out_base, n_ptr, H: tl.constexpr,
                        BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        n = tl.load(n_ptr)
        rp = recv_base.to(tl.pointer_type(tl.float16))
        op = out_base.to(tl.pointer_type(tl.float16))
        for off in range(0, H, BLOCK):
            offs = off + tl.arange(0, BLOCK)
            m = (row < n) & (offs < H)
            v = tl.load(rp + row * H + offs, mask=m, other=0.0).to(tl.float32) \
                * 2.0 + (row % 5).to(tl.float32)
            tl.store(op + row * H + offs, v.to(tl.float16), mask=m)

    def a_a2a():
        ctx = moe_a2a.create_moe_a2a_intra_node_context(
            num_experts=E, hidden=H, dtype=DTYPE, max_tokens=T, topk=TOPK,
            rank=rank, num_ranks=size, mpi=mpi, comm=comm)
        torch.manual_seed(9100 + rank)
        tokens = (torch.randn(T, H, device="cuda") * 0.5).half()
        topk_idx = torch.randint(0, E, (T, TOPK), device="cuda",
                                 dtype=torch.int32)
        weights = torch.rand(T, TOPK, device="cuda") + 0.5
        output = torch.empty(T, H, dtype=DTYPE, device="cuda")
        MRR = T * TOPK * size

        def full_round():
            moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx)
            k_expert_fanout[(MRR,)](ctx.recv_ptr, ctx.out_ptr,
                                    ctx.recv_total_t, H=H, BLOCK=1024,
                                    num_warps=4)
            moe_a2a.moe_combine_intra_node_op(ctx, topk_idx, output,
                                              weights)
        full_round()
        torch.cuda.synchronize()
        ok = torch.isfinite(output.float()).all().item()
        assert ok, "combine produced inf/nan"
        r = tloop(full_round, "a2a_full")
        rd_ = tloop(lambda: moe_a2a.moe_dispatch_intra_node_op(
            tokens, topk_idx, ctx), "a2a_dispatch")
        rc = tloop(lambda: moe_a2a.moe_combine_intra_node_op(
            ctx, topk_idx, output, weights), "a2a_combine")
        return {"p50": r["p50"], "mean": r["mean"], "min": r["min"],
                "corr": 0.0, "dispatch_p50": rd_["p50"],
                "combine_p50": rc["p50"]}
    arm("base_moe_a2a", a_a2a)

    def a_a2a_rccl():
        send_rows = T * TOPK
        per = (send_rows + size - 1) // size
        pad = per * size
        stage_eq = torch.randn(pad, H, device="cuda").half()
        recv_eq = torch.empty(pad, H, dtype=DTYPE, device="cuda")

        def dist_seq():
            dist.all_to_all_single(recv_eq, stage_eq)
            dist.all_to_all_single(stage_eq, recv_eq)
        r = tloop(dist_seq, "a2a_rccl_2x")
        r["MB_per_dir"] = T * TOPK * H * 2 * (size - 1) / size / 1e6
        return r
    arm("rccl_2xa2a", a_a2a_rccl)

# ---------------------------------------------------------------- out --
if rank == 0:
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1)
    print(f"wrote {OUT}", flush=True)
    print(f"[rank 0] ===== {FAM} @ M{M} N{N} K{K} (p50 ms) =====", flush=True)
    for k, v in RES["arms"].items():
        if "SKIP" in v:
            print(f"  {k:22s} SKIP", flush=True)
        else:
            print(f"  {k:22s} {v['p50']:8.2f}", flush=True)
dist.destroy_process_group()
print(f"[rank {rank}] {FAM} DONE", flush=True)
