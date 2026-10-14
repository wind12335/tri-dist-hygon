"""S13 benchmark: windowed frontier reduce-scatter v5 (gemm_rs_v5, paper #1
RS design on DCU planes) vs the S6 fused op vs RCCL reduce_scatter + vendor
GEMM (2x BW1000).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s13_bench_gemm_rs_v5.py
"""
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s13bench_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import gemm_reduce_scatter as grs  # noqa: E402
from triton_dist.kernels.dcu import gemm_rs_v5 as rs5  # noqa: E402
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
    os.environ.setdefault("MASTER_PORT", "29639")
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
        "wait_mode": rs5.__dict__.get("wait_mode", "stream"),
        "warmup/iters": f"{WARMUP}/{ITERS}",
    },
    "rs": {},
}

# (M, K_local, N, chunk_rows[0=auto], stage_slots)
RS_SHAPES = [
    (2048, 2048, 2048, 0, 4),    # auto chunk -> 512, 2 chunks/segment
    (2048, 2048, 2048, 1024, 4),
    (2048, 2048, 2048, 2048, 2),  # 1 chunk per segment: pure v3 window
    (2048, 8192, 1024, 0, 4),
    (1024, 2048, 4096, 512, 4),
    (256, 4096, 4096, 256, 2),   # small batch
]


def bench_rs(M, K, N, CR, SLOTS):
    v5_ctx = rs5.create_gemm_rs_v5_context(
        max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
        mpi=mpi, comm=comm, chunk_rows=CR, stage_slots=SLOTS)
    s6_ctx = grs.create_gemm_rs_intra_node_context(
        max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
        mpi=mpi, comm=comm)

    A, B = [], []
    for s in range(size):
        torch.manual_seed(7100 + s)
        A.append((torch.randn(M, K, device="cuda") * SCALE).half())
        torch.manual_seed(7300 + s)
        B.append((torch.randn(N, K, device="cuda") * SCALE).half())
    a, b = A[rank], B[rank]
    m_pr = M // size
    out = torch.empty(m_pr, N, dtype=DTYPE, device="cuda")
    full = torch.empty(M, N, dtype=DTYPE, device="cuda")

    with torch.no_grad():
        ref = A[0].float() @ B[0].float().t()
        for s in range(1, size):
            ref += A[s].float() @ B[s].float().t()
    shard_ref = ref[rank * m_pr:(rank + 1) * m_pr]

    rs5.gemm_rs_v5_op(a, b, out, v5_ctx)
    torch.cuda.synchronize()
    md = (out.float() - shard_ref).abs().max().item()
    assert md < 1e-2, f"v5 correctness FAIL md={md}"
    c6 = grs.gemm_rs_intra_node_op(a, b, DTYPE, rank, size, s6_ctx)
    torch.cuda.synchronize()
    md6 = (c6.float() - shard_ref).abs().max().item()
    assert md6 < 1e-2, f"s6 correctness FAIL md={md6}"

    res = {}
    res["v5"] = stats(tloop(lambda: rs5.gemm_rs_v5_op(a, b, out, v5_ctx)))
    res["v5_gemm"] = stats(
        tloop(lambda: rs5.gemm_rs_v5_op(a, b, out, v5_ctx,
                                        component="gemm")))
    res["s6"] = stats(
        tloop(lambda: grs.gemm_rs_intra_node_op(a, b, DTYPE, rank, size,
                                                s6_ctx)))
    res["s6_gemm"] = stats(
        tloop(lambda: grs.matmul_local(a, b, s6_ctx.scratch)))
    res["gemm_torch"] = stats(tloop(lambda: torch.matmul(a, b.t(), out=full)))
    # every rank pushes (R-1)/R of the full M*N product bytes
    res["bytes_out_MB"] = M * N * 2 * (size - 1) / size / 1e6

    if dist is not None:
        def dist_seq():
            torch.matmul(a, b.t(), out=full)
            dist.reduce_scatter_tensor(out, full)
        res["dist_rs"] = stats(tloop(lambda: dist.reduce_scatter_tensor(
            out, full)))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"M{M}_K{K}_N{N}_cr{v5_ctx.chunk_rows}_S{v5_ctx.stage_slots}"
    print(f"[rank {rank}] RS {tag} corr={md:.1e} "
          f"push={res['bytes_out_MB']:.1f}MB", flush=True)
    print(f"[rank {rank}]   v5        {fmt(res['v5'])}", flush=True)
    print(f"[rank {rank}]   s6        {fmt(res['s6'])}", flush=True)
    if "dist_seq" in res:
        print(f"[rank {rank}]   dist_seq  {fmt(res['dist_seq'])} "
              f"(rs alone {fmt(res['dist_rs'])})", flush=True)
    print(f"[rank {rank}]   v5_gemm   {fmt(res['v5_gemm'])}  "
          f"s6_gemm {fmt(res['s6_gemm'])}  "
          f"gemm_torch {fmt(res['gemm_torch'])}", flush=True)
    RESULTS["rs"][tag] = res
    rs5.finalize_gemm_rs_v5_context(v5_ctx)
    del v5_ctx, s6_ctx
    torch.cuda.empty_cache()


for (M, K, N, CR, SLOTS) in RS_SHAPES:
    bench_rs(M, K, N, CR, SLOTS)

if rank == 0:
    out = os.path.join(REPO, "bench", "results", "s13_2card_bw1000.json")
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
