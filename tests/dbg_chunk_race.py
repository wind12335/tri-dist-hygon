"""Per-chunk pull/read overlap race (2026-10-14 rig18).

rig17 (symmetric load, whole-pull-then-flags) was CLEAN 0/3000. The
remaining structural difference vs the real op: the real consumer is
gated PER CHUNK, so it starts reading chunk 0 of the symmetric heap
while the engine is still SDMA-writing chunk 1 into the SAME symmetric
heap allocation. Every prior rig raised flags only after the entire
copy landed, so the consumer never ran concurrent with an in-flight
pull.

This rig interleaves [pull ck, flag ck] pairs on a noise stream (the
real lane structure, minus ticket waits) and launches the consumer
right away, idempotent-content like rig17 (both ranks stage the same
seeded X_new, so mixtures are invisible and any wrong element is the
noise/transient class).

Fail forensics additionally re-reads MY ws after the round: torn ws =>
the pull WRITE landed garbage (write-path); clean ws + torn c => the
consumer READ transient (read-path). That decides the mitigation.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x R18_ITERS=3000 python3 tests/dbg_chunk_race.py
"""
import ctypes
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_rig18_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "2-card rig"

ITERS = int(os.environ.get("R18_ITERS", "3000"))

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
elemsz = ag.elemsz
chunk_bytes = CR * K * elemsz
copy_bytes = M * K * elemsz
peer = 1 - rank


@triton.jit(do_not_specialize=["dst_ptr", "val"])
def _flag_xchg1(dst_ptr, val):
    p = dst_ptr.to(tl.pointer_type(tl.int32))
    tl.atomic_xchg(p, val)


@triton.jit(do_not_specialize=["dst_ptr", "src_ptr", "n"])
def _kpeek(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    dp = dst_ptr.to(tl.pointer_type(tl.float16))
    sp = src_ptr.to(tl.pointer_type(tl.float16))
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dp + offs, tl.load(sp + offs, mask=m), mask=m)


b = (torch.randn(N, K, device="cuda") * SCALE).half()
# both ranks consume in the rig16/17-validated rank0 perspective (band 0
# = localA, band 1 = pulled ws); symmetric overlap is the property under
# test and is unaffected by which band is local
ctl = torch.zeros(4 + 2 * R, dtype=torch.int32, device="cuda")
ctl[2] = 0
ctl[4 + 1] = 1
ctl[4 + R] = 128
ctl[4 + R + 1] = 128
s_noise = torch.cuda.Stream()
grid_c = min(M // i1._CONSUMER_BLOCK_M * N // i1._CONSUMER_BLOCK_N,
             i1._CONSUMER_GRID_CAP)
ws_snap = torch.empty(M, K, dtype=torch.float16, device="cuda")
PEEK_BLOCK = 8192
peek_grid = ((M * K + PEEK_BLOCK - 1) // PEEK_BLOCK,)

print(f"[rank {rank}] rig18 iters={ITERS} chunk={chunk_bytes >> 10}KB "
      f"x{R * n_chunks} interleaved", flush=True)
barrier()

fails = 0
worst = 0.0
for it in range(ITERS):
    ticket = it * 8 + 5
    ag.ready.fill_(0)
    torch.manual_seed(70000 + it % 211)
    X_new = (torch.randn(M, K, device="cuda") * SCALE).half()
    a = X_new[:M_pr].contiguous()
    # stage X_new into MY ws (the peer's pull source), then sync the pair
    memcpy_async(ag.ws_ptr, X_new.data_ptr(), copy_bytes,
                 torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    with torch.no_grad():
        ref = X_new.float() @ b.float().t()
    c = torch.empty(M, N, dtype=torch.float16, device="cuda")
    barrier()
    # own-band cells (src=0) are instantly satisfied, as in the real op
    for ck in range(n_chunks):
        _flag_xchg1[(1,)](ag.ready.data_ptr() + ck * 4, ticket + ck)
    # interleaved per-chunk pulls + flags: the consumer eats chunk 0
    # while chunk 1's SDMA write is still in flight (the real overlap)
    with torch.cuda.stream(s_noise):
        for ck in range(n_chunks):
            row0 = peer * M_pr + ck * CR
            memcpy_async(ag.ws_ptr + row0 * K * elemsz,
                         ag.peer_ws[peer] + row0 * K * elemsz,
                         chunk_bytes, s_noise.cuda_stream)
            _flag_xchg1[(1,)](
                ag.ready.data_ptr() + (n_chunks + ck) * 4, ticket + ck)
    i1.kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
        ag.ws_ptr, a, b, c,
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
    d = (c.float() - ref).abs()
    md = d.max().item()
    worst = max(worst, md)
    if md >= 1e-2:
        fails += 1
        wrong = d > 1e-2
        rows = torch.nonzero(wrong.sum(dim=1)).flatten()
        cols = torch.nonzero(wrong.sum(dim=0)).flatten()
        ntiles = sorted(set((cols // 128).tolist()))[:6]
        # decisive: is MY ws itself torn, or did the consumer misread?
        _kpeek[peek_grid](ws_snap.data_ptr(), ag.ws_ptr, M * K,
                          BLOCK=PEEK_BLOCK)
        torch.cuda.synchronize()
        wd = (ws_snap.float() - X_new.float()).abs()
        wsmd = float(wd.max())
        wsn = int((wd > 1e-3).sum())
        wsb = [int((wd[s * M_pr:(s + 1) * M_pr] > 1e-3).sum())
               for s in range(R)]
        print(f"[rank {rank}] FAIL it={it} md={md:.5f} "
              f"rows={int(rows.numel())} ntiles={ntiles} "
              f"localband={int(wrong[:M_pr].sum())} "
              f"wsband={int(wrong[M_pr:].sum())} "
              f"| ws_torn md={wsmd:.4f} n={wsn} per_band={wsb}",
              flush=True)
        torch.save({"it": it, "c": c.cpu(), "ref": ref.cpu(),
                    "X_new": X_new.cpu(), "ws": ws_snap.cpu(),
                    "b": b.cpu(), "rows": rows.cpu()},
                   f"/tmp/rig18fail_it{it}_r{rank}.pt")
    if it % 300 == 0:
        print(f"[rank {rank}] it={it} worst={worst:.5f} fails={fails}",
              flush=True)
    # end-of-iter barrier: neither rank may stage it+1 (overwriting its
    # ws) until BOTH ranks' iter-i pulls have completed
    barrier()

print(f"[rank {rank}] RIG18 RESULT: fails={fails}/{ITERS} "
      f"worst={worst:.5f}", flush=True)
i1.finalize_i1_context(ictx)
print("RIG18DONE")
