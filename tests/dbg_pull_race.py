"""Cross-card pull -> same-VA read race (2026-10-14 v16).

v14 (fabric traffic vs unrelated GEMM) was clean and rig15 (local copy
vs same-VA GEMM) tests the local write race. The real AG pipeline's
remaining combination is: a CROSS-CARD engine pull writes my-ws while
my consumer GEMM reads my-ws, gated by a same-stream flag -- with the
content of BOTH generations known to the checker:

  rank 0: each iter seeds X_new(it) (shared seed) and engine-copies it
          into ITS ws (the pull source).
  rank 1: pre-zero flags; noise stream: raw engine pull peer-ws ->
          my-ws (8 MB) then raise the ready cells (engine write or flag
          kernel -- same wire order as the real op); main stream: the
          solo consumer kernel gated on those cells, reading MY ws.

  Verdict per iter: c must equal X_new @ b exactly (the gate fired after
  the copy). Any deviation beyond fp-noise, checked against a per-64-line
  (X_prev - X_new) mixture decomposition, is the tear.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x R16_ARM=E python3 tests/dbg_pull_race.py
"""
import ctypes
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_rig16_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "2-card rig"

ITERS = int(os.environ.get("R16_ITERS", "3000"))
ARM = os.environ.get("R16_ARM", "E")

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
from triton_dist.kernels.dcu.common_ops import memcpy_async, set_signal  # noqa: E402

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
ctl = torch.zeros(4 + 2 * R, dtype=torch.int32, device="cuda")
ctl[2] = 0
ctl[4 + 1] = 1
ctl[4 + R] = 128
ctl[4 + R + 1] = 128
s_noise = torch.cuda.Stream()
grid_c = min(M // i1._CONSUMER_BLOCK_M * N // i1._CONSUMER_BLOCK_N,
             i1._CONSUMER_GRID_CAP)

# view of my ws for staging/refs
print(f"[rank {rank}] rig16 arm={ARM} iters={ITERS} pull="
      f"{copy_bytes >> 20}MB", flush=True)
barrier()

fails = 0
worst = 0.0
for it in range(ITERS):
    ticket = it * 8 + 5
    ag.ready.fill_(0)
    torch.cuda.synchronize()
    torch.manual_seed(70000 + it % 211)
    X_new = (torch.randn(M, K, device="cuda") * SCALE).half()
    a = X_new[:M_pr].contiguous()
    # rank0: X_new into MY ws (the pull source); rank1: ref + consumer
    if rank == 0:
        memcpy_async(ag.ws_ptr, X_new.data_ptr(), copy_bytes,
                     torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
    else:
        with torch.no_grad():
            ref_new = X_new.float() @ b.float().t()
        c = torch.empty(M, N, dtype=torch.float16, device="cuda")
        with torch.cuda.stream(s_noise):
            memcpy_async(ag.ws_ptr, ag.peer_ws[peer], copy_bytes,
                         s_noise.cuda_stream)
            for cell in range(R * n_chunks):
                v = ticket + cell % n_chunks
                if ARM == "E":
                    set_signal(ag.ready.data_ptr() + cell * 4, v, s_noise)
                else:
                    _flag_xchg1[(1,)](ag.ready.data_ptr() + cell * 4, v)
        # NOTE: consumer gated per-chunk exactly like the real op; the
        # own band reads `a` (fresh local shard), the pulled band reads ws
        i1.kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
            ag.ws_ptr,
            a, b, c,
            M, N, K,
            K, 1,
            a.stride(0), a.stride(1),
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
        d = (c.float() - ref_new).abs()
        md = d.max().item()
        worst = max(worst, md)
        if md >= 1e-2:
            fails += 1
            wrong = d > 1e-2
            rows = torch.nonzero(wrong.sum(dim=1)).flatten()
            cols = torch.nonzero(wrong.sum(dim=0)).flatten()
            ntiles = sorted(set((cols // 128).tolist()))[:6]
            # mixture check vs the KNOWN prior round content (last it's
            # X_new == this round's ws pre-pull content) on the first
            # damaged row
            i0 = int(rows[0])
            torch.manual_seed(70000 + (it - 1) % 211)
            Xp = (torch.randn(M, K, device="cuda") * SCALE).half()
            D = (Xp[i0] - X_new[i0]).float()
            e_row = (c.float() - ref_new)[i0]
            jc = cols[:128]
            y = e_row[jc]
            LB = 64
            nl = K // LB
            Xsub = torch.stack([(D[l*LB:(l+1)*LB][None, :]
                                 * b[jc][:, l*LB:(l+1)*LB].float()
                                 ).sum(dim=1) for l in range(nl)], dim=1)
            x = torch.linalg.lstsq(Xsub, y.unsqueeze(1)).solution.squeeze(1)
            r = (y - Xsub @ x).abs().median()
            print(f"[rank {rank}] FAIL it={it} md={md:.5f} "
                  f"rows={int(rows.numel())} ntiles={ntiles} "
                  f"row{i0} linefit_resid={float(r):.5f} "
                  f"base={float(y.abs().median()):.5f} "
                  f"x={[f'{float(v):.2f}' for v in x[:8]]}", flush=True)
            torch.save({"it": it, "c": c.cpu(), "ref_new": ref_new.cpu(),
                        "X_prev": Xp.cpu(), "X_new": X_new.cpu(),
                        "b": b.cpu(), "rows": rows.cpu()},
                       f"/tmp/rig16fail_it{it}_r{rank}.pt")
    if it % 300 == 0:
        print(f"[rank {rank}] it={it} worst={worst:.5f} fails={fails}",
              flush=True)
    barrier()

print(f"[rank {rank}] RIG16 RESULT arm={ARM}: fails={fails}/{ITERS} "
      f"worst={worst:.5f}", flush=True)
i1.finalize_i1_context(ictx)
print("RIG16DONE")
