"""2-card DCU acceptance test: S13 windowed reduce-scatter v5 (gemm_rs_v5,
paper #1 RS design on DCU planes).

Verifies: rank r keeps rows [r*M/R, (r+1)*M/R) of sum_r a_r @ b_r^T,
slot-reuse across many rounds, partial-M schedules, and the gemm-only
decomposition (gemm_out == my partial).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_gemm_rs_v5.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s13rs_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import gemm_rs_v5 as rs5  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

ok = True


def report(name, passed, detail=""):
    global ok
    if not passed:
        ok = False
    print(f"[rank {rank}] {name}: {'PASS' if passed else 'FAIL'} {detail}",
          flush=True)


M = N = K = 2048
DTYPE = torch.float16
SCALE = 0.02

B = []
for s in range(size):
    torch.manual_seed(999 + s)
    B.append((torch.randn(N, K, device="cuda") * SCALE).half())

ctx = rs5.create_gemm_rs_v5_context(
    max_M=M, N=N, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)
print(f"[rank {rank}] ctx: chunk_rows={ctx.chunk_rows} "
      f"num_chunks={ctx.num_chunks} slots={ctx.stage_slots} "
      f"lanes={ctx.comm_lanes} wait_mode={ctx.wait_mode}", flush=True)


def run_round(it, M_this, tag):
    A_it = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A_it.append((torch.randn(M_this, K, device="cuda") * SCALE).half())
    with torch.no_grad():
        ref = A_it[0].float() @ B[0].float().t()
        for s in range(1, size):
            ref += A_it[s].float() @ B[s].float().t()
    m_pr = M_this // size
    out = torch.empty(m_pr, N, dtype=DTYPE, device="cuda")
    rs5.gemm_rs_v5_op(A_it[rank], B[rank], out, ctx)
    torch.cuda.synchronize()
    shard = ref[rank * m_pr:(rank + 1) * m_pr]
    diff = (out.float() - shard).abs()
    md = diff.max().item()
    report(f"{tag} iter {it}", md < 1e-2,
           f"max_diff={md:.5f} bad={(diff > 1e-2).sum().item()}")


for it in range(3):
    run_round(it, M, "full")
for it in range(8):
    run_round(100 + it, M, "reuse")
for it in range(2):
    run_round(200 + it, 1024, "M1024")

# gemm-only decomposition: gemm_out == MY partial, no comm, no tickets
torch.manual_seed(7777)
a_g = (torch.randn(M, K, device="cuda") * SCALE).half()
dummy = torch.empty(M // size, N, dtype=DTYPE, device="cuda")
rs5.gemm_rs_v5_op(a_g, B[rank], dummy, ctx, component="gemm")
torch.cuda.synchronize()
with torch.no_grad():
    ref_g = a_g.float() @ B[rank].float().t()
md = (ctx.gemm_out.float() - ref_g).abs().max().item()
report("gemm-only", md < 1e-2, f"max_diff={md:.5f}")

rs5.finalize_gemm_rs_v5_context(ctx)

print(f"[rank {rank}] GEMM_RS_V5 TEST {'PASS' if ok else 'FAIL'}", flush=True)
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
