"""2-card DCU acceptance test: S14 tile-ready chunk-major PULL allgather-GEMM
(gemm_ag_tr -- the paper #1 AG design on DCU planes).

Verifies: C[M, N_pr] = gather_r(a_r) @ b_r on every rank, many-round reuse
(ticketed slot guard across rounds), partial-M schedules, and the
gemm-only decomposition (consumer GEMM against the workspace as the last
full op left it). The engine PULL direction and the local-flag plane were
probed by tests/probe_engine_pull.py.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_gemm_ag_tr.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s14ag_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import gemm_ag_tr as agtr  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

ok = True


def report(name, passed, detail=""):
    global ok
    if not passed:
        ok = False
    print(f"[rank {rank}] {name}: {'PASS' if passed else 'FAIL'} {detail}",
          flush=True)


M_per_rank = 1024
K = 2048
N = 2048
M = M_per_rank * size
DTYPE = torch.float16
SCALE = 0.02

torch.manual_seed(999 + rank)
b = (torch.randn(N, K, device="cuda") * SCALE).half()

ctx = agtr.create_gemm_ag_tr_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)
print(f"[rank {rank}] ctx: chunk_rows={ctx.chunk_rows} "
      f"num_chunks={ctx.num_chunks} lanes={ctx.comm_lanes} "
      f"wait_mode={ctx.wait_mode}", flush=True)

last_A = None


def run_round(it, m_pr_this, tag):
    global last_A
    A_it = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A_it.append((torch.randn(m_pr_this, K, device="cuda") * SCALE).half())
    a = A_it[rank]
    with torch.no_grad():
        A_full = torch.cat(A_it, dim=0).float()
        ref = A_full @ b.float().t()
    c = torch.empty(m_pr_this * size, N, dtype=DTYPE, device="cuda")
    agtr.gemm_ag_tr_op(a, b, c, ctx)
    torch.cuda.synchronize()
    diff = (c.float() - ref).abs()
    md = diff.max().item()
    report(f"{tag} iter {it}", md < 1e-2,
           f"max_diff={md:.5f} bad={(diff > 1e-2).sum().item()}")
    last_A = A_it


for it in range(3):
    run_round(it, M_per_rank, "full")
for it in range(8):
    run_round(100 + it, M_per_rank, "reuse")
for it in range(2):
    run_round(200 + it, 512, "M512shard")

# gemm-only decomposition: consumer GEMM against the workspace as the last
# full op left it -> same c as that op's reference
m_last = last_A[rank].shape[0] * size
c_g = torch.empty(m_last, N, dtype=DTYPE, device="cuda")
agtr.gemm_ag_tr_op(last_A[rank], b, c_g, ctx, component="gemm")
torch.cuda.synchronize()
with torch.no_grad():
    ref_g = torch.cat(last_A, dim=0).float() @ b.float().t()
md = (c_g.float() - ref_g).abs().max().item()
report("gemm-only", md < 1e-2, f"max_diff={md:.5f}")

agtr.finalize_gemm_ag_tr_context(ctx)

print(f"[rank {rank}] GEMM_AG_TR TEST {'PASS' if ok else 'FAIL'}", flush=True)
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
