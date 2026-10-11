"""S4 stall recurrence probe.

Established: the first flag-spin that actually WAITS for an in-flight
peer engine write stalls ~20-800 ms, once; later waits are fast. Two
questions:
  Q1 400 consecutive fused ops in one ctx: how many stalls?  (per-process
     one-time vs recurring)
  Q2 [10 fused ops + 1 NCCL all_gather] x 20 rounds: does NCCL re-trigger
     the stall each round?  (paper use-case: NCCL collectives interleaved
     with our fused ops)

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 s4_rearm_probe.py
"""
import os
import sys

REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)
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

# torch.distributed (nccl->rccl) for Q2
import torch.distributed as dist  # noqa: E402
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29719")
dist.init_process_group(backend="nccl", rank=rank, world_size=size)
x = torch.ones(64, device="cuda") * (rank + 1)
dist.all_reduce(x)
torch.cuda.synchronize()
assert x[0].item() == 3.0

M = M_pr * size
ctx = ag.create_ag_gemm_intra_node_context(
    M, K, DTYPE, DTYPE, rank, size, mpi, comm, M_PER_CHUNK=MPC)

torch.manual_seed(7000)
A_full = (torch.randn(M, K, device="cuda") * SCALE).half()
a = A_full[rank * M_pr:(rank + 1) * M_pr]
torch.manual_seed(7100 + rank)
B = (torch.randn(N, K, device="cuda") * SCALE).half()
c = torch.empty(M, N, dtype=DTYPE, device="cuda")
Ag_buf = torch.empty(M, K, dtype=DTYPE, device="cuda")

ag.ag_gemm_intra_node_op(a, B, c, ctx)  # live ctx
torch.cuda.synchronize()


def timed_op(i):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    ag.ag_gemm_intra_node_op(a, B, c, ctx)
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e)
    if t > 5:
        print(f"[rank {rank}]   STALL {t:.1f} ms", flush=True)
    return t


print(f"[rank {rank}] Q1: 400 consecutive fused ops", flush=True)
stalls = 0
tsum = 0.0
for i in range(400):
    t = timed_op(i)
    tsum += t
    stalls += t > 5
print(f"[rank {rank}] Q1: stalls={stalls} total={tsum:.0f}ms "
      f"(avg {tsum / 400:.2f}ms)", flush=True)

print(f"[rank {rank}] Q2: 20 rounds of [10 fused + 1 nccl all_gather]", flush=True)
stalls = 0
tsum = 0.0
for r in range(20):
    for i in range(10):
        t = timed_op(i)
        tsum += t
        stalls += t > 5
    dist.all_gather_into_tensor(Ag_buf, a)
    torch.cuda.synchronize()
print(f"[rank {rank}] Q2: stalls={stalls} total={tsum:.0f}ms "
      f"(avg {tsum / 200:.2f}ms)", flush=True)

print(f"[rank {rank}] REARM PROBE DONE", flush=True)
dist.destroy_process_group()
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
