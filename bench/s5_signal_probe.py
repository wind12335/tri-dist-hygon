"""S5 tail-latency probe: push signal transport A/B/C.

Established (S4 + S5 bench): the fused AG op has a ~30 ms tail on nearly
every bench iteration (mean ~33 ms, p50 ~1 ms). The push phase INCLUDING
the signal engine memcpy completes in <1 ms -> the consumer's flag spin
waits ~30 ms for the signal write to become VISIBLE at the peer. Suspect:
the tiny flag engine-write queues behind the engine's processing of the
multi-MB data copies.

Variants (module SIGNAL_MODE in all_gather_gemm):
  engine : one engine memcpy of staged flags per dest (S4 default)
  host   : event-sync data copies, then host dushmem putmem of the flags

200 fused ops per variant on the validated shape; stalls (>5 ms) counted.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s5_signal_probe.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s5sig_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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
M_pr, K, N = 2048, 2048, 2048
MPCS = [512, 256, 128]  # bench (<=256) stalls ~every op; rearm (512) 1%
MODES = ["engine", "host"]
OPS = 100

M = M_pr * size

torch.manual_seed(7000)
A_full = (torch.randn(M, K, device="cuda") * SCALE).half()
a = A_full[rank * M_pr:(rank + 1) * M_pr]
torch.manual_seed(7100 + rank)
B = (torch.randn(N, K, device="cuda") * SCALE).half()
c = torch.empty(M, N, dtype=DTYPE, device="cuda")

with torch.no_grad():
    ref = A_full.float() @ B.float().t()


def run_variant(ctx, mode):
    ag.SIGNAL_MODE = mode
    # correctness under this transport first
    c.zero_()
    ag.ag_gemm_intra_node_op(a, B, c, ctx)
    torch.cuda.synchronize()
    md = (c.float() - ref).abs().max().item()
    assert md < 1e-2, f"{mode}: WRONG md={md}"

    stalls = []
    ts = []
    t0 = time.time()
    for i in range(OPS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        ag.ag_gemm_intra_node_op(a, B, c, ctx)
        e.record()
        torch.cuda.synchronize()
        t = s.elapsed_time(e)
        ts.append(t)
        if t > 5:
            stalls.append(t)
    wall = time.time() - t0
    ts.sort()
    n = len(ts)
    print(f"[rank {rank}] mpc{ctx.M_PER_CHUNK:<4} {mode:>7}: "
          f"mean {sum(ts)/n:7.2f}  p50 {ts[n//2]:6.2f}  "
          f"p90 {ts[int(n*0.9)]:6.2f}  max {ts[-1]:7.1f} ms | "
          f"stalls(>5ms) {len(stalls):3d}  wall {wall:5.1f}s  md={md:.1e}",
          flush=True)


for MPC in MPCS:
    ctx = ag.create_ag_gemm_intra_node_context(
        M, K, DTYPE, DTYPE, rank, size, mpi, comm, M_PER_CHUNK=MPC)
    ag.ag_gemm_intra_node_op(a, B, c, ctx)  # live ctx (JIT + flags)
    torch.cuda.synchronize()
    for mode in MODES:
        run_variant(ctx, mode)

print(f"[rank {rank}] S5 SIGNAL PROBE DONE", flush=True)
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
