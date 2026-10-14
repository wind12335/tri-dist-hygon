"""S12 benchmark: windowed recursive-doubling panel GEMM-AllReduce
(gemm_ar_rd, paper #1 AR v23 design on DCU planes) vs the S6 fused op
vs RCCL allreduce + vendor GEMM (2x BW1000).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s12_bench_gemm_ar_rd.py
"""
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s12bench_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import gemm_allreduce as gar  # noqa: E402
from triton_dist.kernels.dcu import gemm_ar_rd as rd  # noqa: E402
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
    os.environ.setdefault("MASTER_PORT", "29637")
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
        "wait_mode": rd.rd_wait_mode(),
        "warmup/iters": f"{WARMUP}/{ITERS}",
    },
    "ar": {},
}

# (M, K_local, N, chunk_rows[0=auto], stage_slots)
AR_SHAPES = [
    (2048, 2048, 2048, 0, 4),    # auto chunk -> 512, 4 panels/slots
    (2048, 2048, 2048, 1024, 4),
    (2048, 2048, 2048, 2048, 2),  # 1 panel: pure RD, no window
    (2048, 8192, 1024, 0, 4),
    (1024, 2048, 4096, 512, 4),
    (256, 4096, 4096, 256, 2),   # small batch
]


def bench_ar(M, K, N, CR, SLOTS):
    rd_ctx = rd.create_gemm_ar_rd_context(
        max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm,
        chunk_rows=CR, stage_slots=SLOTS)
    s6_ctx = gar.create_gemm_ar_intra_node_context(
        max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm,
        M_PER_CHUNK=min(512, M))

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

    rd.gemm_ar_rd_op(a, b, c, rd_ctx)
    torch.cuda.synchronize()
    md = (c.float() - ref).abs().max().item()
    assert md < 1e-2, f"RD correctness FAIL md={md}"

    res = {}
    res["rd"] = stats(tloop(lambda: rd.gemm_ar_rd_op(a, b, c, rd_ctx)))
    res["rd_gemm"] = stats(
        tloop(lambda: rd.gemm_ar_rd_op(a, b, c, rd_ctx,
                                       component="gemm")))
    res["s6"] = stats(tloop(lambda: gar.gemm_ar_intra_node_op(a, b, c, s6_ctx)))
    res["s6_gemm"] = stats(
        tloop(lambda: gar.gemm_ar_intra_node_op(a, b, c, s6_ctx,
                                                component="gemm")))
    res["gemm_torch"] = stats(tloop(lambda: torch.matmul(a, b.t(), out=c)))
    # RD pushes log2(R) panel copies per rank vs (R-1) for the full mesh
    n_phases = size.bit_length() - 1
    res["bytes_out_MB"] = M * N * 2 * n_phases / 1e6

    if dist is not None:
        def dist_seq():
            torch.matmul(a, b.t(), out=c)
            dist.all_reduce(c)
        res["dist_ar"] = stats(tloop(lambda: dist.all_reduce(c)))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"M{M}_K{K}_N{N}_cr{rd_ctx.chunk_rows}_S{rd_ctx.stage_slots}"
    print(f"[rank {rank}] AR {tag} corr={md:.1e} "
          f"rd_bytes={res['bytes_out_MB']:.1f}MB", flush=True)
    print(f"[rank {rank}]   rd        {fmt(res['rd'])}", flush=True)
    print(f"[rank {rank}]   s6        {fmt(res['s6'])}", flush=True)
    if "dist_seq" in res:
        print(f"[rank {rank}]   dist_seq  {fmt(res['dist_seq'])} "
              f"(ar alone {fmt(res['dist_ar'])})", flush=True)
    print(f"[rank {rank}]   rd_gemm   {fmt(res['rd_gemm'])}  "
          f"s6_gemm {fmt(res['s6_gemm'])}  "
          f"gemm_torch {fmt(res['gemm_torch'])}", flush=True)
    RESULTS["ar"][tag] = res
    rd.finalize_gemm_ar_rd_context(rd_ctx)
    del rd_ctx, s6_ctx
    torch.cuda.empty_cache()


for (M, K, N, CR, SLOTS) in AR_SHAPES:
    bench_ar(M, K, N, CR, SLOTS)

if rank == 0:
    out = os.path.join(REPO, "bench", "results", "s12_2card_bw1000.json")
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
