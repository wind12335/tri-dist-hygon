"""S6 benchmark: gemm_allreduce fused op vs RCCL allreduce + vendor GEMM
sequence (2x BW1000).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s6_bench_gemm_ar.py
"""
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s6ar_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import gemm_allreduce as gar  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

DTYPE = torch.float16
SCALE = 0.02
WARMUP = 10
ITERS = 50
LOG_STALLS = os.environ.get("BENCH_LOG_STALLS") == "1"


def tloop(fn, stream=None, warmup=WARMUP, iters=ITERS):
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
    os.environ.setdefault("MASTER_PORT", "29633")
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
    },
    "ar": {},
}

# ------------------------------------------------------------ GEMM-AllReduce --
AR_SHAPES = [  # (M, K_local, N, M_PER_CHUNK)
    (2048, 2048, 2048, 1024),  # MPC A/B: 2 chunks vs 4
    (2048, 2048, 2048, 512),
    (1024, 2048, 4096, 512),
    (2048, 8192, 1024, 1024),
    (2048, 8192, 1024, 512),
    (256, 4096, 4096, 256),   # small batch
    (128, 4096, 4096, 128),   # decode
]


def bench_ar(M, K, N, MPC):
    ctx = gar.create_gemm_ar_intra_node_context(
        max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm,
        M_PER_CHUNK=MPC)

    # deterministic per-rank inputs on every rank (reference needs them all)
    A, B = [], []
    for s in range(size):
        torch.manual_seed(7100 + s)
        A.append((torch.randn(M, K, device="cuda") * SCALE).half())
        torch.manual_seed(7300 + s)
        B.append((torch.randn(N, K, device="cuda") * SCALE).half())
    a, b = A[rank], B[rank]
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")

    with torch.no_grad():
        ref = A[0].float() @ B[0].float().t()
        for s in range(1, size):
            ref += A[s].float() @ B[s].float().t()
    gar.gemm_ar_intra_node_op(a, b, c, ctx)
    torch.cuda.synchronize()
    md = (c.float() - ref).abs().max().item()
    assert md < 1e-2, f"AR correctness FAIL md={md}"

    res = {}
    res["ours"] = stats(tloop(lambda: gar.gemm_ar_intra_node_op(a, b, c, ctx)))
    res["barrier"] = stats(
        tloop(lambda: ctx.barrier.round_barrier(rank)))
    res["gemm_triton"] = stats(
        tloop(lambda: gar.gemm_ar_intra_node_op(a, b, c, ctx,
                                                component="gemm")))
    # vendor GEMM cost of ONE rank's partial (all ranks identical shape)
    res["gemm_torch"] = stats(
        tloop(lambda: torch.matmul(a, b.t(), out=c)))
    res["bytes_out_MB"] = M * N * 2 * (size - 1) / 1e6

    if dist is not None:
        def dist_seq():
            torch.matmul(a, b.t(), out=c)
            dist.all_reduce(c)
        res["dist_ar"] = stats(tloop(lambda: dist.all_reduce(c)))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"M{M}_K{K}_N{N}_mpc{MPC}"
    print(f"[rank {rank}] AR {tag} corr={md:.1e} "
          f"bytes_out={res['bytes_out_MB']:.1f}MB", flush=True)
    print(f"[rank {rank}]   ours      {fmt(res['ours'])}", flush=True)
    if "dist_seq" in res:
        print(f"[rank {rank}]   dist_seq  {fmt(res['dist_seq'])} "
              f"(ar alone {fmt(res['dist_ar'])})", flush=True)
    print(f"[rank {rank}]   gemmT     {fmt(res['gemm_triton'])}  "
          f"gemm_torch {fmt(res['gemm_torch'])}  "
          f"barrier {fmt(res['barrier'])}", flush=True)
    RESULTS["ar"][tag] = res
    del ctx
    torch.cuda.empty_cache()


for (M, K, N, MPC) in AR_SHAPES:
    bench_ar(M, K, N, MPC)

if rank == 0:
    out = os.path.join(REPO, "bench", "results", "s6_2card_bw1000.json")
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
