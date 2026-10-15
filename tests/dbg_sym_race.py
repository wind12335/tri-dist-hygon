"""Symmetric-load noise-class repro (2026-10-14 v18/rig17).

Every isolation rig so far is clean (solo, +fabric noise, +local race,
+one-sided cross-card race = mixture class only) while the real
pipeline tears with content matching NOTHING. The ingredient none of
the rigs had: BOTH cards running the consumer GEMM while BOTH pull
each other's ws over the fabric (full-duplex fabric + both GPUs under
GEMM load).

This rig makes the pulls CONTENT-IDEMPOTENT: both ranks stage the same
seeded X_new into their own ws, then each pulls the PEER's ws (same
bytes) behind a flag while running the consumer. Mixtures of identical
content are no-ops, so ANY wrong element is the pure noise/transient
class -- the class the real pipeline shows.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x R17_ITERS=3000 python3 tests/dbg_sym_race.py
"""
import ctypes
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_rig17_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "2-card rig"

ITERS = int(os.environ.get("R17_ITERS", "3000"))

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int


def barrier():
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu.common_ops import memcpy_async  # noqa: E402

M_pr, K, N = 2048, 1024, 2048
R = 2
M = M_pr * R
CR = 1024
n_chunks = M_pr // CR
SCALE = 0.02

ictx = i1.create_i1_context(max_M=M, K=K, input_dtype=torch.float16,
                            output_dtype=torch.float16, rank=rank,
                            num_ranks=size, mpi=mpi, comm=comm)
ag = ictx.ag
copy_bytes = M * K * ag.elemsz
peer = 1 - rank


@triton.jit(do_not_specialize=["dst_ptr", "val"])
def _flag_xchg1(dst_ptr, val):
    p = dst_ptr.to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(p, val)


b = (torch.randn(N, K, device="cuda") * SCALE).half()
# both ranks consume in the rig16-validated rank0 perspective (band 0 =
# localA, band 1 = pulled ws) so the verdict math stays exact; the
# symmetric-load property (both consuming + both pulling at once) is
# unaffected by which band is local
ctl = torch.zeros(4 + 2 * R, dtype=torch.int32, device="cuda")
ctl[2] = 0
ctl[4 + 1] = 1
ctl[4 + R] = 128
ctl[4 + R + 1] = 128
s_noise = torch.cuda.Stream()
grid_c = min(M // i1._CONSUMER_BLOCK_M * N // i1._CONSUMER_BLOCK_N,
             i1._CONSUMER_GRID_CAP)

print(f"[rank {rank}] rig17 iters={ITERS} pull={copy_bytes >> 20}MB "
      f"idempotent", flush=True)
barrier()

fails = 0
worst = 0.0
for it in range(ITERS):
    ticket = it * 8 + 5
    ag.ready.fill_(0)
    torch.manual_seed(70000 + it % 211)
    X_new = (torch.randn(M, K, device="cuda") * SCALE).half()
    a = X_new[:M_pr].contiguous()
    # stage X_new into MY ws (the peer pulls this), then sync the pair
    memcpy_async(ag.ws_ptr, X_new.data_ptr(), copy_bytes,
                 torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    with torch.no_grad():
        ref = X_new.float() @ b.float().t()
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    barrier()
    with torch.cuda.stream(s_noise):
        memcpy_async(ag.ws_ptr, ag.peer_ws[peer], copy_bytes,
                     s_noise.cuda_stream)
        for cell in range(R * n_chunks):
            _flag_xchg1[(1,)](ag.ready.data_ptr() + cell * 4,
                              ticket + cell % n_chunks)
    i1.kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
        ag.ws_ptr, a, b, c,
        M, N, K,
        K,  # single A row stride
        b.stride(1), b.stride(0),
        c.stride(0), c.stride(1),
        0, R, n_chunks,
        ag.ready.data_ptr(), ctl.data_ptr(), ticket - 1,
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
              f"rows={int(rows.numel())} ntiles={ntiles} "
              f"localband={int(wrong[:M_pr].sum())} "
              f"wsband={int(wrong[M_pr:].sum())}",
              flush=True)
        torch.save({"it": it, "c": c.cpu(), "ref": ref.cpu(),
                    "X_new": X_new.cpu(), "b": b.cpu(), "rows": rows.cpu()},
                   f"/tmp/rig17fail_it{it}_r{rank}.pt")
    if it % 300 == 0:
        print(f"[rank {rank}] it={it} worst={worst:.5f} fails={fails}",
              flush=True)
    # end-of-iter barrier: neither rank may stage it+1 (overwriting its
    # ws) until BOTH ranks' iter-i pulls have completed
    barrier()

print(f"[rank {rank}] RIG17 RESULT: fails={fails}/{ITERS} "
      f"worst={worst:.5f}", flush=True)
i1.finalize_i1_context(ictx)
print("RIG17DONE")
