"""S7 probe: decompose the a2a RMA metadata plane (0.48 ms/op measured,
v1 MPI host path was 0.43 -- WHERE does it go?).

Variants (all ranks run the SAME variant per loop, flags stay monotonic):
  sync_only : torch.cuda.synchronize() on an idle stream (host wall)
  A         : counters.zero_ + bincount + pack + sync#1
  B         : A + peer putmems + own-row engine copy   (host wall)
  C         : B + table kernel + plan D2H + sync#2     == component="meta"
              (flags are LONG SET by A/B of both ranks -> C measures the
              no-spin floor: kernel + copy + sync)
  C_live    : full component="meta" loop (real skew-wait spin included)

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s7_meta_decompose.py
"""
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s7md_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import moe_a2a  # noqa: E402
from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    dcu_state, memcpy_async)

torch.cuda.set_device(rank)

T, H, TOPK, E = 2048, 2048, 8, 32
WARM, N = 10, 100

ctx = moe_a2a.create_moe_a2a_intra_node_context(
    num_experts=E, hidden=H, dtype=torch.float16, max_tokens=T, topk=TOPK,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)
torch.manual_seed(9100 + rank)
tokens = (torch.randn(T, H, device="cuda") * 0.5).half()
topk_idx = torch.randint(0, E, (T, TOPK), device="cuda", dtype=torch.int32)
host = dcu_state()["host"]
R, epr = size, E // size
msg_bytes = (1 + E) * 4


def wall(fn, n=N):
    for _ in range(WARM):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def phase_ab(sync2=False, tables=False):
    ctx.meta_round += 1
    rd = ctx.meta_round
    ctx.counters.zero_()
    moe_a2a.kernel_pack_meta[(1,)](topk_idx, ctx.meta_msg, rd, T * TOPK,
                                   E=E, EZ=triton.next_power_of_2(E),
                                   BLOCK=1024, num_warps=4)
    torch.cuda.synchronize()
    for d in range(R):
        if d != rank:
            host.putmem(ctx.meta_ptr + rank * msg_bytes,
                        ctx.meta_msg.data_ptr(), msg_bytes, d)
    memcpy_async(ctx.meta_ptr + rank * msg_bytes, ctx.meta_msg.data_ptr(),
                 msg_bytes, torch.cuda.current_stream().cuda_stream)
    if tables:
        moe_a2a.kernel_meta_tables[(1,)](
            ctx.meta_ptr, ctx.dst_row_base, ctx.stage_row_base, ctx.recv_seg,
            ctx.host_plan, ctx.recv_total_t, rd, rank,
            E=E, EPR=epr, NUM_RANKS=R, num_warps=1)
        ctx.host_plan_pinned.copy_(ctx.host_plan, non_blocking=False)
    if sync2:
        torch.cuda.synchronize()


res = {}
res["sync_only"] = wall(lambda: torch.cuda.synchronize())
res["A_binpack_sync1"] = wall(lambda: phase_ab())
res["B_A+putmem"] = wall(lambda: phase_ab(sync2=True))
res["C_B+tables+D2H+sync2"] = wall(lambda: phase_ab(sync2=True, tables=True))
# real op: same as C but phases of the two ranks interleave naturally
res["C_live_component_meta"] = wall(
    lambda: moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx,
                                               component="meta"))
res["bincount_only"] = wall(
    lambda: torch.bincount(topk_idx.reshape(-1).long(), minlength=E))

for k, v in res.items():
    print(f"[rank {rank}] {k:26s} {v:7.3f} ms", flush=True)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
