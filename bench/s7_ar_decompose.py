"""S7 AR tuning probe: decompose the gemm_allreduce residue (S6.5 open
item c) and sweep the consumer / chunk granularity.

Components at M=2048,K=2048,N=2048 (fp16):
  full(MPC)      = barrier + gemm + push/signal + consumer
  gemm           = chunk GEMMs only (component="gemm")
  push           = barrier + gemm + push + signal, NO consumer
                   (component="push", flags advance -> consumer re-run ok)
  reduce(B,W)    = consumer kernel alone, re-launched on a static, fully
                   signaled workspace (BLOCK_SIZE x num_warps sweep)
  monolithic     = MPC == M (one GEMM launch, one push per dest, no
                   per-chunk event/stream juggling)

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s7_ar_decompose.py
"""
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s7ard_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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
import triton  # noqa: E402

from triton_dist.kernels.dcu import gemm_allreduce as gar  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

torch.cuda.set_device(rank)
DTYPE = torch.float16
WARMUP, ITERS = 10, 50


def tloop(fn, warmup=WARMUP, iters=ITERS):
    st = torch.cuda.current_stream()
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
        ts.append(s.elapsed_time(e))
    return ts


def stats(ts):
    ts = sorted(ts)
    return {"mean": statistics.fmean(ts), "p50": ts[len(ts) // 2],
            "min": ts[0]}


def fmt(d):
    return f"{d['mean']:7.3f}/{d['p50']:7.3f}/{d['min']:7.3f}"


def make(M, K, N):
    torch.manual_seed(9500 + rank)
    a = (torch.randn(M, K, device="cuda") * 0.02).half()
    b = (torch.randn(N, K, device="cuda") * 0.02).half()
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")
    return a, b, c


def run_shape(M, K, N):
    print(f"\n[rank {rank}] ===== M{M} K{K} N{N} =====", flush=True)
    for MPC in [1024, M]:
        ctx = gar.create_gemm_ar_intra_node_context(
            max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
            rank=rank, num_ranks=size, mpi=mpi, comm=mpi and comm,
            M_PER_CHUNK=MPC)
        a, b, c = make(M, K, N)
        m_chunk = M // MPC
        chunk_elems = MPC * N

        res = {}
        res["full"] = stats(tloop(
            lambda: gar.gemm_ar_intra_node_op(a, b, c, ctx)))
        res["barrier"] = stats(tloop(lambda: ctx.barrier.round_barrier(rank)))
        res["gemm"] = stats(tloop(
            lambda: gar.gemm_ar_intra_node_op(a, b, c, ctx,
                                              component="gemm")))
        res["push"] = stats(tloop(
            lambda: gar.gemm_ar_intra_node_op(a, b, c, ctx,
                                              component="push")))
        # consumer alone on a static, fully-signaled workspace
        gar.gemm_ar_intra_node_op(a, b, c, ctx, component="push")
        torch.cuda.synchronize()

        def reduce_only(BLOCK, W):
            gar.kernel_ar_consumer_reduce[(triton.cdiv(M * N, BLOCK),)](
                ctx.ws_ptr, c, ctx.flags_ptr,
                ctx.flag_round, rank, M * N,
                num_ranks=size, m_chunk_num=ctx.m_chunk_num,
                SLOT_ELEMS=ctx.max_M * N, CHUNK_ELEMS=chunk_elems,
                BLOCK_SIZE=BLOCK, num_warps=W)

        for BLOCK, W in [(2048, 2), (2048, 4), (4096, 4), (4096, 8),
                         (8192, 8)]:
            res[f"reduce_B{BLOCK}_w{W}"] = stats(
                tloop(lambda: reduce_only(BLOCK, W), warmup=5, iters=30))

        tag = "monolithic" if MPC == M else f"mpc{MPC}"
        print(f"[rank {rank}] {tag}: full {fmt(res['full'])}  "
              f"gemm {fmt(res['gemm'])}  push {fmt(res['push'])}  "
              f"barrier {fmt(res['barrier'])}", flush=True)
        line = "  ".join(f"B{B}w{W} {fmt(res[f'reduce_B{B}_w{W}'])}"
                         for B, W in [(2048, 2), (2048, 4), (4096, 4),
                                      (4096, 8), (8192, 8)])
        print(f"[rank {rank}]   reduce: {line}", flush=True)
        del ctx
        torch.cuda.empty_cache()


run_shape(2048, 2048, 2048)
run_shape(2048, 8192, 1024)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
