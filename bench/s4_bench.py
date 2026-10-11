"""S4 performance benchmark (2x BW1000): triton-dist DCU kernels vs baselines.

Measures, per shape (fp16):
  AG-GEMM : fused op vs decomposition (push-only + consumer-only) vs
            torch.matmul, plus torch.distributed all_gather baseline if a
            GPU backend exists in this wheel. overlap_saved =
            (push + gemm_triton) - fused: the time the overlap hides.
  GEMM-RS : fused op vs (gemm_torch / gemm_triton + scatter + reduce),
            plus torch.distributed reduce_scatter baseline if available.
  barrier : SymmBarrier round cost alone.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 s4_bench.py
"""
import os
import sys
import json
import time
import traceback
import statistics

REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)
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

from triton_dist.kernels.dcu import all_gather_gemm as ag  # noqa: E402
from triton_dist.kernels.dcu import gemm_reduce_scatter as grs  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

DTYPE = torch.float16
SCALE = 0.02
WARMUP, ITERS = 3, 50
OUT_DIR = os.path.join(REPO, "bench", "results")

# ---------------------------------------------------------------- helpers --
LOG_STALLS = os.environ.get("BENCH_LOG_STALLS") == "1"


def tloop(fn, stream=None, warmup=WARMUP, iters=ITERS):
    """GPU-event timing (ms) of fn() enqueued on `stream` (default: current)."""
    st = stream if stream is not None else torch.cuda.current_stream()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for i in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(st)
        fn()
        e.record(st)
        torch.cuda.synchronize()
        t = s.elapsed_time(e)
        ts.append(t)
        if LOG_STALLS and t > 5:
            print(f"    [stall] iter {i}: {t:.1f} ms", flush=True)
    return ts


def stats(ts):
    ts = sorted(ts)
    return {"mean": statistics.fmean(ts), "p50": ts[len(ts) // 2],
            "min": ts[0]}


def fmt(d):
    return f"{d['mean']:8.3f}/{d['p50']:8.3f}/{d['min']:8.3f}"  # mean/p50/min


# ------------------------------------------- torch.distributed GPU backend --
dist = None
dist_backend = None
try:
    import torch.distributed as _dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29617")
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
        "num_sms": torch.cuda.get_device_properties(0).multi_processor_count,
        "num_ranks": size,
        "dtype": "fp16",
        "dist_backend": dist_backend,
        "warmup/iters": f"{WARMUP}/{ITERS}",
    },
    "ag": {},
    "rs": {},
    "ag_mpc": {},
}

# ----------------------------------------------------------------- AG-GEMM --
AG_SHAPES = [  # (M_per_rank, K, N_per_rank, M_PER_CHUNK)
    (128, 4096, 4096, 128),   # decode small batch (M=256 gathered)
    (512, 4096, 4096, 256),
    (1024, 4096, 8192, 256),
    (1024, 8192, 1024, 256),
    (2048, 2048, 2048, 256),  # validated shape
]


def bench_ag(M_pr, K, N, MPC):
    M = M_pr * size
    ctx = ag.create_ag_gemm_intra_node_context(
        M, K, DTYPE, DTYPE, rank, size, mpi, comm, M_PER_CHUNK=MPC)
    s_ag = ctx.ag_stream

    torch.manual_seed(7000)  # identical A_full on both ranks
    A_full = (torch.randn(M, K, device="cuda") * SCALE).half()
    a = A_full[rank * M_pr:(rank + 1) * M_pr]
    torch.manual_seed(7100 + rank)
    B = (torch.randn(N, K, device="cuda") * SCALE).half()
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")

    with torch.no_grad():
        ref = A_full.float() @ B.float().t()
    ag.ag_gemm_intra_node_op(a, B, c, ctx)
    torch.cuda.synchronize()
    md = (c.float() - ref).abs().max().item()
    assert md < 1e-2, f"AG correctness FAIL md={md}"

    res = {}
    res["ours"] = stats(tloop(lambda: ag.ag_gemm_intra_node_op(a, B, c, ctx)))
    res["barrier"] = stats(
        tloop(lambda: ctx.barrier.round_barrier(rank)))

    def push_only():
        with torch.cuda.stream(s_ag):
            ctx.push.launch(a.data_ptr(), ctx.ws_ptr, ctx.flags_ptr,
                            ctx.flag_round, rank, M_pr)
    res["push"] = stats(tloop(push_only, stream=s_ag))

    res["gemm_triton"] = stats(
        tloop(lambda: ag.ag_gemm_intra_node_op(a, B, c, ctx, skip_comm=True)))
    res["gemm_torch"] = stats(
        tloop(lambda: torch.matmul(A_full, B.t(), out=c)))

    res["seq_push_gemmT"] = res["push"]["mean"] + res["gemm_triton"]["mean"]
    res["overlap_saved_ms"] = res["seq_push_gemmT"] - res["ours"]["mean"]
    res["bytes_out_MB"] = M_pr * K * 2 * (size - 1) / 1e6
    res["push_GBps"] = res["bytes_out_MB"] / res["push"]["mean"]  # MB/ms = GB/s

    if dist is not None:
        Ag_buf = torch.empty(M, K, dtype=DTYPE, device="cuda")

        def dist_seq():
            dist.all_gather_into_tensor(Ag_buf, a)
            torch.matmul(Ag_buf, B.t(), out=c)
        res["dist_ag"] = stats(tloop(lambda: dist.all_gather_into_tensor(Ag_buf, a)))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"Mpr{M_pr}_K{K}_N{N}_mpc{MPC}"
    print(f"[rank {rank}] AG {tag} corr={md:.1e} bytes_out={res['bytes_out_MB']:.1f}MB",
          flush=True)
    print(f"[rank {rank}]   ours        {fmt(res['ours'])} ms (mean/p50/min)", flush=True)
    print(f"[rank {rank}]   push        {fmt(res['push'])} -> {res['push_GBps']:.2f} GB/s",
          flush=True)
    print(f"[rank {rank}]   gemm_triton {fmt(res['gemm_triton'])}", flush=True)
    print(f"[rank {rank}]   gemm_torch  {fmt(res['gemm_torch'])}", flush=True)
    print(f"[rank {rank}]   barrier     {fmt(res['barrier'])}", flush=True)
    print(f"[rank {rank}]   overlap_saved {res['overlap_saved_ms']:.3f} ms "
          f"(push+gemm_triton={res['seq_push_gemmT']:.3f} vs ours={res['ours']['mean']:.3f})",
          flush=True)
    if dist is not None:
        print(f"[rank {rank}]   dist({dist_backend})_ag {fmt(res['dist_ag'])} | "
              f"dist_ag+gemm_torch {fmt(res['dist_seq'])}", flush=True)
    return res


for shp in AG_SHAPES:
    try:
        RESULTS["ag"][f"Mpr{shp[0]}_K{shp[1]}_N{shp[2]}_mpc{shp[3]}"] = \
            bench_ag(*shp)
    except Exception:
        print(f"[rank {rank}] AG shape {shp} FAILED:", flush=True)
        traceback.print_exc()

# M_PER_CHUNK sensitivity on the validated shape
for MPC in (128, 256, 512, 1024):
    try:
        RESULTS["ag_mpc"][f"mpc{MPC}"] = bench_ag(2048, 2048, 2048, MPC)
    except Exception:
        print(f"[rank {rank}] AG mpc={MPC} FAILED:", flush=True)
        traceback.print_exc()

# ---------------------------------------------------------------- GEMM-RS --
RS_SHAPES = [  # (M, K, N): full M on each rank, output keeps M/size rows
    (512, 4096, 4096),
    (2048, 4096, 4096),
    (4096, 2048, 2048),
]


def bench_rs(M, K, N):
    Mpr = M // size
    ctx = grs.create_gemm_rs_intra_node_context(
        max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
        mpi=mpi, comm=comm)

    A_src = []
    for s in range(size):
        torch.manual_seed(7200 + s)
        A_src.append((torch.randn(M, K, device="cuda") * SCALE).half())
    # per-rank B (identical list on both ranks): op semantics are
    # out_r = sum_s (A_s @ B_s^T)[rows r] -- ref must pair each source's B
    B_src = []
    for s in range(size):
        torch.manual_seed(7300 + s)
        B_src.append((torch.randn(N, K, device="cuda") * SCALE).half())
    B = B_src[rank]
    with torch.no_grad():
        ref = sum((A_src[s].float() @ B_src[s].float().t() for s in range(size)))
    exp = ref[rank * Mpr:(rank + 1) * Mpr]

    out = grs.gemm_rs_intra_node(A_src[rank], B, ctx)
    torch.cuda.synchronize()
    md = (out.float() - exp).abs().max().item()
    assert md < 1e-2, f"RS correctness FAIL md={md}"

    res = {}
    res["ours"] = stats(tloop(lambda: grs.gemm_rs_intra_node(A_src[rank], B, ctx)))
    res["gemm_torch"] = stats(
        tloop(lambda: torch.matmul(A_src[rank], B.t(), out=ctx.scratch)))
    res["gemm_triton"] = stats(
        tloop(lambda: grs.matmul_local(A_src[rank], B, ctx.scratch)))
    res["scatter"] = stats(
        tloop(lambda: ctx.scatter_put.launch(ctx.scratch.data_ptr(),
                                             ctx.scatter_ptr, rank, Mpr * N)))
    res["reduce"] = stats(
        tloop(lambda: grs.ring_reduce_after_scatter(
            rank, size, ctx.scatter_ptr, M, N, DTYPE,
            torch.cuda.current_stream())))
    res["seq_triton"] = (res["gemm_triton"]["mean"] + res["scatter"]["mean"]
                         + res["reduce"]["mean"])
    res["bytes_out_MB"] = Mpr * N * 2 * (size - 1) / 1e6

    if dist is not None:
        rs_out = torch.empty(Mpr, N, dtype=DTYPE, device="cuda")

        def dist_seq():
            torch.matmul(A_src[rank], B.t(), out=ctx.scratch)
            dist.reduce_scatter_tensor(rs_out, ctx.scratch)
        res["dist_rs"] = stats(
            tloop(lambda: dist.reduce_scatter_tensor(rs_out, ctx.scratch)))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"M{M}_K{K}_N{N}"
    print(f"[rank {rank}] RS {tag} corr={md:.1e} bytes_out={res['bytes_out_MB']:.1f}MB",
          flush=True)
    print(f"[rank {rank}]   ours         {fmt(res['ours'])} ms", flush=True)
    print(f"[rank {rank}]   gemm_torch   {fmt(res['gemm_torch'])}", flush=True)
    print(f"[rank {rank}]   gemm_triton  {fmt(res['gemm_triton'])}", flush=True)
    print(f"[rank {rank}]   scatter      {fmt(res['scatter'])}", flush=True)
    print(f"[rank {rank}]   reduce       {fmt(res['reduce'])}", flush=True)
    print(f"[rank {rank}]   seq(g+s+r)_triton {res['seq_triton']:.3f} vs "
          f"ours {res['ours']['mean']:.3f}", flush=True)
    if dist is not None:
        print(f"[rank {rank}]   dist({dist_backend})_rs {fmt(res['dist_rs'])} | "
              f"gemm_torch+dist_rs {fmt(res['dist_seq'])}", flush=True)
    return res


for shp in RS_SHAPES:
    try:
        RESULTS["rs"][f"M{shp[0]}_K{shp[1]}_N{shp[2]}"] = bench_rs(*shp)
    except Exception:
        print(f"[rank {rank}] RS shape {shp} FAILED:", flush=True)
        traceback.print_exc()

# ------------------------------------------------------------------- dump --
if rank == 0:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "s4_2card_bw1000.json")
    with open(out_path, "w") as f:
        json.dump(RESULTS, f, indent=1)
    print(f"[rank 0] results -> {out_path}", flush=True)

print(f"[rank {rank}] S4 BENCH DONE", flush=True)

if dist is not None:
    dist.destroy_process_group()
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
