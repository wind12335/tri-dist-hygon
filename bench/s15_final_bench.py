"""S15 FINAL benchmark (2x BW1000): the three migrated paper-#1 ops at the
user-specified LLaMA-scale shapes vs the old S6 fused ops (where their N%256
alignment allows) vs RCCL + vendor GEMM.

Shapes: M8192 K8192 N29568 and M8192 K12288 N49152 (fp16).
Ops per shape: AR (gemm_ar_rd), RS (gemm_rs_v5), AG (gemm_ag_tr).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s15_final_bench.py
"""
import os
import statistics
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s15final_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size in (2, 8)

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import all_gather_gemm as s6ag  # noqa: E402
from triton_dist.kernels.dcu import gemm_allreduce as s6ar  # noqa: E402
from triton_dist.kernels.dcu import gemm_ag_tr as agtr  # noqa: E402
from triton_dist.kernels.dcu import gemm_ar_rd as rd  # noqa: E402
from triton_dist.kernels.dcu import gemm_reduce_scatter as s6rs  # noqa: E402
from triton_dist.kernels.dcu import gemm_rs_v5 as rs5  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

DTYPE = torch.float16
SCALE = 0.02
WARMUP = 5
ITERS = 15
TOL = 1e-1  # loose fp16 tolerance -- the deliverable is speedup

# (M, K, N) -- the user-specified final shapes
SHAPES = [
    (8192, 8192, 28672),
    (8192, 12288, 49152),
]
# section selector (the S6 AR op alone needs a ~3.2 GB symmetric heap at
# these shapes; separate invocations keep the heap isolated):
# S15_OPS=ar,rs,ag  S15_SHAPES=0,1
OPS_SEL = [o for o in os.environ.get("S15_OPS", "ar,rs,ag").split(",") if o]
SHAPE_SEL = [int(i) for i in os.environ.get("S15_SHAPES", "0,1").split(",")]


def tloop(fn, warmup=WARMUP, iters=ITERS):
    st = torch.cuda.current_stream()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(st)
        fn()
        e.record(st)
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def stats(ts):
    ts = sorted(ts)
    return {"mean": statistics.fmean(ts), "p50": ts[len(ts) // 2],
            "min": ts[0]}


def fmt(d):
    return f"{d['mean']:9.2f}/{d['p50']:9.2f}/{d['min']:9.2f}"  # mean/p50/min


# ------------------------------------------- torch.distributed GPU backend --
dist = None
dist_backend = None
try:
    import torch.distributed as _dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29641")
    for _be in ("nccl", "rccl"):
        try:
            _dist.init_process_group(backend=_be, rank=rank, world_size=size)
            try:
                x = torch.ones(64, device="cuda") * (rank + 1)
                _dist.all_reduce(x)
                torch.cuda.synchronize()
                assert x[0].item() == 3.0, "verify failed"
                dist, dist_backend = _dist, _be
                break
            except Exception:
                _dist.destroy_process_group()
        except Exception:
            pass
except Exception:
    pass
print(f"[rank {rank}] torch.distributed GPU backend: "
      f"{dist_backend or 'UNAVAILABLE'}", flush=True)

RESULTS = {
    "env": {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "torch": torch.__version__,
        "gpus": torch.cuda.get_device_name(0),
        "num_ranks": size,
        "dtype": "fp16",
        "dist_backend": dist_backend,
        "warmup/iters": f"{WARMUP}/{ITERS}",
        "tol": TOL,
    },
    "final": {},
}


def chunked_ref_sum(A_list, B_list, M, N, chunk=1024):
    """ref[r0:r1] = sum_s A_s[r0:r1] @ B_s^T, fp16 GEMMs (fp32 accum
    inside), accumulated in fp32 per chunk row-block."""
    ref = torch.empty(M, N, dtype=DTYPE, device="cuda")
    for r0 in range(0, M, chunk):
        rows = min(chunk, M - r0)
        acc = torch.zeros(rows, N, dtype=torch.float32, device="cuda")
        for s in range(size):
            t = A_list[s][r0:r0 + rows] @ B_list[s].t()
            acc += t.float()
        ref[r0:r0 + rows] = acc.half()
    return ref


def max_diff_vs_ref(c, ref, chunk=1024):
    md = 0.0
    for r0 in range(0, c.shape[0], chunk):
        md = max(md, (c[r0:r0 + chunk].float()
                      - ref[r0:r0 + chunk].float()).abs().max().item())
    return md


def bench_shape(M, K, N):
    tag0 = f"M{M}_K{K}_N{N}"
    m_pr = M // size
    res_shape = {}

    A, B = [], []
    for s in range(size):
        torch.manual_seed(7100 + s)
        A.append((torch.randn(M, K, device="cuda") * SCALE).half())
        torch.manual_seed(7300 + s)
        B.append((torch.randn(N, K, device="cuda") * SCALE).half())
    a, b = A[rank], B[rank]
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")

    t0 = time.time()
    ref = chunked_ref_sum(A, B, M, N)
    if rank == 0:
        print(f"[rank 0] {tag0}: ref built in {time.time() - t0:.1f}s "
              f"({torch.cuda.max_memory_allocated() / 2**30:.1f}GB peak)",
              flush=True)
    res_shape["gemm_torch"] = stats(tloop(lambda: torch.matmul(a, b.t(),
                                                               out=c)))
    if "ar" not in OPS_SEL:
        RESULTS["final"][tag0] = res_shape
        del ref, A, B
        torch.cuda.empty_cache()
        return

    # ------------------------------------------------------------- AR ----
    rd_ctx = rd.create_gemm_ar_rd_context(
        max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm)
    if rank == 0:
        print(f"[rank 0] {tag0} AR rd ctx: CR={rd_ctx.chunk_rows} "
              f"chunks={rd_ctx.num_chunks} slots={rd_ctx.stage_slots}",
              flush=True)
    rd.gemm_ar_rd_op(a, b, c, rd_ctx)
    torch.cuda.synchronize()
    md = max_diff_vs_ref(c, ref)
    assert md < TOL, f"AR rd FAIL md={md}"

    res = {"corr": md}
    res["rd"] = stats(tloop(lambda: rd.gemm_ar_rd_op(a, b, c, rd_ctx)))
    res["rd_gemm"] = stats(
        tloop(lambda: rd.gemm_ar_rd_op(a, b, c, rd_ctx, component="gemm")))
    rd.finalize_gemm_ar_rd_context(rd_ctx)
    del rd_ctx

    try:
        s6ar_ctx = s6ar.create_gemm_ar_intra_node_context(
            max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
            rank=rank, num_ranks=size, mpi=mpi, comm=comm,
            M_PER_CHUNK=min(512, M))
        s6ar.gemm_ar_intra_node_op(a, b, c, s6ar_ctx)
        torch.cuda.synchronize()
        md6 = max_diff_vs_ref(c, ref)
        assert md6 < TOL, f"AR s6 FAIL md={md6}"
        res["s6"] = stats(
            tloop(lambda: s6ar.gemm_ar_intra_node_op(a, b, c, s6ar_ctx)))
        pass  # (no finalize in the S6 module; heap freed at process exit)
        del s6ar_ctx
    except Exception:
        if rank == 0:
            print(f"[rank 0] {tag0} AR s6 SKIPPED:", flush=True)
            traceback.print_exc()
    if dist is not None:
        def dist_ar_seq():
            torch.matmul(a, b.t(), out=c)
            dist.all_reduce(c)
        res["dist_seq"] = stats(tloop(dist_ar_seq))
    print(f"[rank {rank}] {tag0} AR corr={md:.1e}", flush=True)
    for k in ("rd", "s6", "dist_seq", "rd_gemm"):
        if k in res:
            print(f"[rank {rank}]   AR {k:9s} {fmt(res[k])}", flush=True)
    res_shape["ar"] = res
    del res
    if "rs" not in OPS_SEL:
        RESULTS["final"][tag0] = res_shape
        del ref, A, B
        torch.cuda.empty_cache()
        return

    # ------------------------------------------------------------- RS ----
    rs_ctx = rs5.create_gemm_rs_v5_context(
        max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
        mpi=mpi, comm=comm)
    if rank == 0:
        print(f"[rank 0] {tag0} RS v5 ctx: CR={rs_ctx.chunk_rows} "
              f"chunks={rs_ctx.num_chunks} slots={rs_ctx.stage_slots}",
              flush=True)
    out_shard = torch.empty(m_pr, N, dtype=DTYPE, device="cuda")
    rs5.gemm_rs_v5_op(a, b, out_shard, rs_ctx)
    torch.cuda.synchronize()
    shard_ref = ref[rank * m_pr:(rank + 1) * m_pr]
    md = max_diff_vs_ref(out_shard, shard_ref)
    assert md < TOL, f"RS v5 FAIL md={md}"

    res = {"corr": md}
    res["v5"] = stats(tloop(lambda: rs5.gemm_rs_v5_op(a, b, out_shard,
                                                      rs_ctx)))
    res["v5_gemm"] = stats(
        tloop(lambda: rs5.gemm_rs_v5_op(a, b, out_shard, rs_ctx,
                                        component="gemm")))
    rs5.finalize_gemm_rs_v5_context(rs_ctx)
    del rs_ctx

    try:
        s6rs_ctx = s6rs.create_gemm_rs_intra_node_context(
            max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
            mpi=mpi, comm=comm)
        c6 = s6rs.gemm_rs_intra_node_op(a, b, DTYPE, rank, size, s6rs_ctx)
        torch.cuda.synchronize()
        md6 = max_diff_vs_ref(c6, shard_ref)
        assert md6 < TOL, f"RS s6 FAIL md={md6}"
        del c6
        res["s6"] = stats(
            tloop(lambda: s6rs.gemm_rs_intra_node_op(a, b, DTYPE, rank,
                                                     size, s6rs_ctx)))
        pass
        del s6rs_ctx
    except Exception:
        if rank == 0:
            print(f"[rank 0] {tag0} RS s6 SKIPPED:", flush=True)
            traceback.print_exc()
    if dist is not None:
        def dist_rs_seq():
            torch.matmul(a, b.t(), out=c)
            dist.reduce_scatter_tensor(out_shard, c)
        res["dist_seq"] = stats(tloop(dist_rs_seq))
    print(f"[rank {rank}] {tag0} RS corr={md:.1e}", flush=True)
    for k in ("v5", "s6", "dist_seq", "v5_gemm"):
        if k in res:
            print(f"[rank {rank}]   RS {k:9s} {fmt(res[k])}", flush=True)
    res_shape["rs"] = res
    del res
    if "ag" not in OPS_SEL:
        RESULTS["final"][tag0] = res_shape
        del ref, A, B
        torch.cuda.empty_cache()
        return

    # ------------------------------------------------------------- AG ----
    a_shards = [A[s][s * m_pr:(s + 1) * m_pr] for s in range(size)]
    a_loc = a_shards[rank]
    # AG-GEMM gathers the ROW shards: C = cat_r(shard_r) @ b^T. A is
    # seeded per rank, so the ref must use the concatenated shards (the
    # first version used A[rank]'s full rows and "failed" with md~0.3)
    a_full = torch.cat(a_shards, dim=0)
    ref_ag = torch.empty(M, N, dtype=DTYPE, device="cuda")
    for r0 in range(0, M, 1024):
        rows = min(1024, M - r0)
        ref_ag[r0:r0 + rows] = a_full[r0:r0 + rows] @ b.t()
    del a_full

    ag_ctx = agtr.create_gemm_ag_tr_context(
        max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm)
    if rank == 0:
        print(f"[rank 0] {tag0} AG tr ctx: CR={ag_ctx.chunk_rows} "
              f"chunks={ag_ctx.num_chunks} lanes={ag_ctx.comm_lanes}",
              flush=True)
    agtr.gemm_ag_tr_op(a_loc, b, c, ag_ctx)
    torch.cuda.synchronize()
    md = max_diff_vs_ref(c, ref_ag)
    assert md < TOL, f"AG tr FAIL md={md}"

    res = {"corr": md}
    res["tr"] = stats(tloop(lambda: agtr.gemm_ag_tr_op(a_loc, b, c, ag_ctx)))
    res["tr_gemm"] = stats(
        tloop(lambda: agtr.gemm_ag_tr_op(a_loc, b, c, ag_ctx,
                                         component="gemm")))
    agtr.finalize_gemm_ag_tr_context(ag_ctx)
    del ag_ctx

    try:
        s6ag_ctx = s6ag.create_ag_gemm_intra_node_context(
            max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
            rank=rank, num_ranks=size, mpi=mpi, comm=comm,
            M_PER_CHUNK=min(1024, m_pr))
        c6 = s6ag.ag_gemm_intra_node(a_loc, b, False, s6ag_ctx)
        torch.cuda.synchronize()
        md6 = max_diff_vs_ref(c6, ref_ag)
        assert md6 < TOL, f"AG s6 FAIL md={md6}"
        del c6
        res["s6"] = stats(
            tloop(lambda: s6ag.ag_gemm_intra_node(a_loc, b, False,
                                                  s6ag_ctx)))
        pass
        del s6ag_ctx
    except Exception:
        if rank == 0:
            print(f"[rank 0] {tag0} AG s6 SKIPPED:", flush=True)
            traceback.print_exc()
    if dist is not None:
        # all_gather into a fresh buffer, then matmul (RCCL + vendor)
        buf = torch.empty(M, K, dtype=DTYPE, device="cuda")

        def dist_ag_seq2():
            dist.all_gather_into_tensor(buf, a_loc)
            torch.matmul(buf, b.t(), out=c)
        res["dist_seq"] = stats(tloop(dist_ag_seq2))
        del buf
    print(f"[rank {rank}] {tag0} AG corr={md:.1e}", flush=True)
    for k in ("tr", "s6", "dist_seq", "tr_gemm"):
        if k in res:
            print(f"[rank {rank}]   AG {k:9s} {fmt(res[k])}", flush=True)
    res_shape["ag"] = res

    RESULTS["final"][tag0] = res_shape
    del ref, ref_ag, A, B, a_shards
    torch.cuda.empty_cache()


for si in SHAPE_SEL:
    (M, K, N) = SHAPES[si]
    bench_shape(M, K, N)

if rank == 0:
    suffix = "-".join(OPS_SEL) + "_" + "-".join(str(i) for i in SHAPE_SEL)
    out = os.path.join(REPO, "bench", "results",
                       f"s15_final_{suffix}_2card_bw1000.json")
    with open(out, "w") as f:
        import json
        json.dump(RESULTS, f, indent=2)
    print(f"[rank 0] results -> {out}", flush=True)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
