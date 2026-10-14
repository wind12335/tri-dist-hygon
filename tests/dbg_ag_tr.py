"""S14 debug (2-card): one gemm_ag_tr op, then localize the wrong rows and
dump the flag planes. Not part of the acceptance suite.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/dbg_ag_tr.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s14dbg_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import gemm_ag_tr as agtr  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

m_pr, K, N = 4096, 8192, 28672
M = m_pr * size
DTYPE = torch.float16
SCALE = 0.02

torch.manual_seed(999 + rank)
b = (torch.randn(N, K, device="cuda") * SCALE).half()
A = []
for s in range(size):
    torch.manual_seed(5000 + s)
    A.append((torch.randn(m_pr, K, device="cuda") * SCALE).half())
a = A[rank]

ctx = agtr.create_gemm_ag_tr_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)
print(f"[rank {rank}] ctx: CR={ctx.chunk_rows} n_chunks={ctx.num_chunks}",
      flush=True)

c = torch.empty(M, N, dtype=DTYPE, device="cuda")
agtr.gemm_ag_tr_op(a, b, c, ctx)
torch.cuda.synchronize()

with torch.no_grad():
    ref = torch.cat(A, dim=0).float() @ b.float().t()
diff = (c.float() - ref).abs()
print(f"[rank {rank}] max_diff={diff.max().item():.5f}", flush=True)

# per 128-row block: how many elements exceed 1e-2
per_block = (diff > 1e-2).view(M // 128, 128, N).sum(dim=(1, 2))
bad_blocks = [(i, int(n)) for i, n in enumerate(per_block.tolist()) if n]
print(f"[rank {rank}] bad 128-row blocks (idx, count): "
      f"{bad_blocks[:20]}{'...' if len(bad_blocks) > 20 else ''} "
      f"total={len(bad_blocks)}/{M // 128}", flush=True)

# per COLUMN block too (is it rows or also columns?)
per_colblock = (diff > 1e-2).view(M, N // 128, 128).sum(dim=(0, 2))
bad_cols = [(i, int(n)) for i, n in enumerate(per_colblock.tolist()) if n]
print(f"[rank {rank}] bad 128-col blocks: {bad_cols[:8]} "
      f"total={len(bad_cols)}/{N // 128}", flush=True)

# dump flag planes
print(f"[rank {rank}] ready={ctx.ready.tolist()}", flush=True)
arr = torch.empty(size * ctx.num_chunks, dtype=torch.int32, device="cuda")
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int]
assert hip.hipMemcpy(arr.data_ptr(), ctypes.c_void_p(ctx.arrive_ptr),
                     4 * size * ctx.num_chunks, 2) == 0
print(f"[rank {rank}] arrive={arr.tolist()} round_base={ctx.round_base} "
      f"last_done={ctx.last_done_ticket}", flush=True)

# ws check: does my ws hold the correct gathered A_full?
ws = torch.empty(M, K, dtype=DTYPE, device="cuda")
assert hip.hipMemcpy(ws.data_ptr(), ctypes.c_void_p(ctx.ws_ptr),
                     M * K * 2, 2) == 0
A_full = torch.cat(A, dim=0)
wmd = (ws.float() - A_full.float()).abs().view(size, m_pr, K).amax(dim=(1, 2))
print(f"[rank {rank}] ws per-shard max_diff={[f'{v:.4f}' for v in wmd.tolist()]}",
      flush=True)

agtr.finalize_gemm_ag_tr_context(ctx)
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
