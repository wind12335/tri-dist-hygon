"""S4 outlier diagnosis: the fused AG op occasionally absorbs a ~1s stall
(mean 32ms vs p50 1.3ms over 28 runs -- about ONE ~900ms event per ctx on
some shapes). Ladder per ctx, 40 iterations each, RAW per-iter times:

  L1 barrier+push   (no consumer -- no flag spins)
  L2 barrier+push+consumer   (= ours)

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 s4_outlier_probe.py
"""
import os
import sys

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

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import all_gather_gemm as ag  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

DTYPE = torch.float16
SCALE = 0.02
M_pr, K, N, MPC = 2048, 2048, 2048, 512
ITERS = 40


def run_ladder(tag):
    M = M_pr * size
    ctx = ag.create_ag_gemm_intra_node_context(
        M, K, DTYPE, DTYPE, rank, size, mpi, comm, M_PER_CHUNK=MPC)
    s_ag = ctx.ag_stream

    torch.manual_seed(7000)
    A_full = (torch.randn(M, K, device="cuda") * SCALE).half()
    a = A_full[rank * M_pr:(rank + 1) * M_pr]
    torch.manual_seed(7100 + rank)
    B = (torch.randn(N, K, device="cuda") * SCALE).half()
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")

    # one real op so flags/workspace are live (same as bench correctness)
    ag.ag_gemm_intra_node_op(a, B, c, ctx)
    torch.cuda.synchronize()

    def l1():
        ctx.flag_round += 1
        cur = torch.cuda.current_stream()
        cur.wait_stream(ctx.gemm_stream)
        cur.wait_stream(ctx.ag_stream)
        ctx.barrier.round_barrier(rank)
        ctx.ag_stream.wait_stream(cur)
        with torch.cuda.stream(s_ag):
            ctx.push.launch(a.data_ptr(), ctx.ws_ptr, ctx.flags_ptr,
                            ctx.flag_round, rank, M_pr)

    def l2():
        ag.ag_gemm_intra_node_op(a, B, c, ctx)

    def l2phased():
        """Same as l2 but with per-phase events: barrier on current,
        push on ag_stream, consumer on gemm_stream."""
        ctx.flag_round += 1
        cur = torch.cuda.current_stream()
        cur.wait_stream(ctx.gemm_stream)
        cur.wait_stream(ctx.ag_stream)
        eb = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        eb[0].record(cur)
        ctx.barrier.round_barrier(rank)
        eb[1].record(cur)
        ctx.ag_stream.wait_stream(cur)
        with torch.cuda.stream(s_ag):
            eb[2].record()
            ctx.push.launch(a.data_ptr(), ctx.ws_ptr, ctx.flags_ptr,
                            ctx.flag_round, rank, M_pr)
            eb[3].record()
        ctx.gemm_stream.wait_stream(cur)
        with torch.cuda.stream(ctx.gemm_stream):
            ec0 = torch.cuda.Event(enable_timing=True)
            ec1 = torch.cuda.Event(enable_timing=True)
            ec0.record()
            c2 = ag.ag_gemm_intra_node_op(a, B, c, ctx, skip_comm=True)
            ec1.record()
        torch.cuda.synchronize()
        print(f"[rank {rank}]   phase barrier={eb[0].elapsed_time(eb[1]):8.2f} "
              f"push={eb[2].elapsed_time(eb[3]):8.2f} "
              f"consumer={ec0.elapsed_time(ec1):8.2f}", flush=True)

    def l3():
        # consumer-only (no spins: flags already >= round from the last
        # real op) but WITH remote-workspace loads -- isolates the data
        # path from the flag path
        ag.ag_gemm_intra_node_op(a, B, c, ctx, skip_comm=True)

    for name, fn in (("L1_barrier_push", l1), ("L3_consumer_only", l3),
                     ("L2ph", l2phased), ("L2_ours", l2)):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ts = []
        for i in range(ITERS):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append((i, s.elapsed_time(e)))
        big = [(i, t) for i, t in ts if t > 5]
        line = " ".join(f"{t:.2f}" for _, t in ts)
        print(f"[rank {rank}] {tag} {name}: {line}", flush=True)
        print(f"[rank {rank}] {tag} {name}: >5ms at {big} "
              f"total={sum(t for _, t in ts):.1f}ms", flush=True)


run_ladder("ctxA")
run_ladder("ctxC")

print(f"[rank {rank}] OUTLIER PROBE DONE", flush=True)
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
