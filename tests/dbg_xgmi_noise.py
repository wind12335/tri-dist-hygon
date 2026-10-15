"""Cross-card traffic -> read-fault isolation (2026-10-14 v14).

Question: does CONCURRENT cross-card pull traffic corrupt the reads of an
UNRELATED, protocol-free GEMM on the pulling rank?

  rank 0: keeps its symmetric ws fresh (plain local copy into ws).
  rank 1: noise stream = back-to-back raw engine pulls peer-ws -> my-ws
               (NO flags, NO gates, NO protocol -- pure fabric traffic),
          main stream = the v13 solo consumer loop on its OWN private
               tensors (pre-staged, gates pre-satisfied, host-ordered).

If the solo consumer tears here, the AG tear's root cause is a transient
READ fault under concurrent fabric-write pressure -- independent of the
AG protocol, its gates, its flags and its staging forms. If it stays 0,
the fault needs the real per-round protocol overlap instead.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x XGMI_NOISE=1 python3 tests/dbg_xgmi_noise.py

Env: XGMI_NOISE=0 control (no pull stream), =1 traffic arm (default);
     XGMI_ITERS=1500; XGMI_PULLS=8 pulls per iteration.
"""
import ctypes
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_xgmi_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "2-card rig"

NOISE = os.environ.get("XGMI_NOISE", "1") == "1"
ITERS = int(os.environ.get("XGMI_ITERS", "1500"))
PULLS = int(os.environ.get("XGMI_PULLS", "8"))

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int


def barrier():
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


import torch  # noqa: E402
import triton  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu.gemm_ag_tr import memcpy_async  # noqa: E402

M_pr, K, N = 2048, 1024, 2048
R = 2
M = M_pr * R
CR = 1024
n_chunks = M_pr // CR
ROUND = 100
SCALE = 0.02

ictx = i1.create_i1_context(max_M=M, K=K, input_dtype=torch.float16,
                            output_dtype=torch.float16, rank=rank,
                            num_ranks=size, mpi=mpi, comm=comm)
ag = ictx.ag
copy_bytes = M * K * ag.elemsz
print(f"[rank {rank}] v14 noise={NOISE} pull={copy_bytes >> 20}MB "
      f"x{PULLS}/iter iters={ITERS}", flush=True)

# ---- the solo consumer's PRIVATE world (no protocol objects shared) ----
b = (torch.randn(N, K, device="cuda") * SCALE).half()
flags = torch.full((R * n_chunks,), ROUND + 1 + n_chunks,
                   dtype=torch.int32, device="cuda")
ctl = torch.zeros(4 + 2 * R, dtype=torch.int32, device="cuda")
ctl[2] = 0
ctl[4 + 1] = 1
ctl[4 + R] = 128
ctl[4 + R + 1] = 128

src_fresh = (torch.randn(M, K, device="cuda") * SCALE).half()

s_noise = torch.cuda.Stream()
grid_c = min(M // i1._CONSUMER_BLOCK_M * N // i1._CONSUMER_BLOCK_N,
             i1._CONSUMER_GRID_CAP)

barrier()
fails = 0
worst = 0.0
for it in range(ITERS):
    torch.manual_seed(5000 + 100 * (it % 97) + 4242)
    A_full = (torch.randn(M, K, device="cuda") * SCALE).half()
    a = A_full[:M_pr].contiguous()
    ws = A_full.contiguous()
    with torch.no_grad():
        ref = A_full.float() @ b.float().t()
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")

    # noise: rank0 refreshes its ws (local copy), rank1 pulls it across
    # the fabric on a side stream while the consumer below runs
    if NOISE:
        with torch.cuda.stream(s_noise):
            if rank == 0:
                memcpy_async(ag.ws_ptr, src_fresh.data_ptr(), copy_bytes,
                             s_noise.cuda_stream)
            else:
                for _ in range(PULLS):
                    memcpy_async(ag.ws_ptr, ag.peer_ws[0], copy_bytes,
                                 s_noise.cuda_stream)

    i1.kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
        ws, a, b, c,
        M, N, K,
        K,  # single A row stride
        b.stride(1), b.stride(0),
        c.stride(0), c.stride(1),
        0, R, n_chunks,
        flags.data_ptr(), ctl.data_ptr(), ROUND,
        diag_ptr=0,
        BLOCK_SIZE_M=i1._CONSUMER_BLOCK_M,
        BLOCK_SIZE_N=i1._CONSUMER_BLOCK_N,
        BLOCK_SIZE_K=i1._CONSUMER_BLOCK_K,
        GROUP_SIZE_M=i1._CONSUMER_GROUP_M,
        M_PER_CHUNK=CR,
        NUM_SMS=grid_c,
        EVEN_K=True,
        DIAG=False,
        num_warps=i1._CONSUMER_WARPS,
        num_stages=1,
    )
    torch.cuda.synchronize()
    d = (c.float() - ref).abs()
    md = d.max().item()
    worst = max(worst, md)
    if md >= 1e-2:
        fails += 1
        wrong = d > 1e-2
        rows = torch.nonzero(wrong.sum(dim=1)).flatten()
        cols = torch.nonzero(wrong.sum(dim=0)).flatten()
        ntiles = sorted(set((cols // 128).tolist()))[:6]
        print(f"[rank {rank}] FAIL it={it} md={md:.5f} "
              f"rows={rows[:6].tolist()}..{int(rows.numel())} "
              f"ntiles={ntiles}", flush=True)
    if it % 300 == 0:
        print(f"[rank {rank}] it={it} worst={worst:.5f} fails={fails}",
              flush=True)

print(f"[rank {rank}] V14 RESULT noise={NOISE}: fails={fails}/{ITERS} "
      f"worst={worst:.5f}", flush=True)
i1.finalize_i1_context(ictx)
print("V14DONE")
