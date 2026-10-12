"""2-card DCU acceptance test: all_gather_gemm (triton-dist port).

Rank r pushes A_full[r*Mpr:(r+1)*Mpr] to peers; consumer GEMM overlaps the
push (per-chunk flag spins). Reference: A_full @ B_r^T (fp32), A_full
generated identically on both ranks so no communication is needed for it.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 test_ag_gemm.py
"""
import os
import sys
import time

REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_agemm_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402
import numpy as np  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size in (2, 8)

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
hip.hipMemcpy.restype = ctypes.c_int
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int]
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import all_gather_gemm as ag  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

M_per_rank = 2048
K = N = 2048
M = M_per_rank * size
M_PER_CHUNK = 256  # 8 chunks/rank: exercises the (source, chunk) flag grid
DTYPE = torch.float16
SCALE = 0.02

s_ag = torch.cuda.Stream()
s_gemm = torch.cuda.Stream()

ctx = ag.create_ag_gemm_intra_node_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE, rank=rank,
    num_ranks=size, mpi=mpi, comm=comm, M_PER_CHUNK=M_PER_CHUNK,
    ag_stream=s_ag, gemm_stream=s_gemm)

m_chunk_num = M_per_rank // M_PER_CHUNK
flags_u64 = np.zeros(size * m_chunk_num, dtype=np.uint64)


def dump_flags():
    nbytes = flags_u64.nbytes
    hb = (ctypes.c_char * nbytes)()
    rc = hip.hipMemcpy(hb, ctypes.c_void_p(ctx.flags_ptr), nbytes, 2)
    assert rc == 0, f"hipMemcpy rc={rc}"
    flags_u64[:] = np.frombuffer(hb, dtype=np.uint64)


ok = True
for it in range(3):
    # fresh data every iteration: identical inputs would mask stale reads
    torch.manual_seed(5000 + 100 * it)
    A_full = (torch.randn(M, K, device="cuda") * SCALE).half()
    torch.manual_seed(9000 + 100 * it + rank)
    B = (torch.randn(N, K, device="cuda") * SCALE).half()
    a = A_full[rank * M_per_rank:(rank + 1) * M_per_rank]

    with torch.no_grad():
        ref = A_full.float() @ B.float().t()

    t0 = time.perf_counter()
    out = ag.ag_gemm_intra_node(a, B, False, ctx)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    dump_flags()
    diff = (out.float() - ref).abs()
    md = diff.max().item()
    status = "PASS" if md < 1e-2 else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"[rank {rank}] iter {it} max_diff={md:.5f} "
          f"bad(>1e-2)={(diff > 1e-2).sum().item()} "
          f"flags_min={flags_u64.min()} t={dt:.2f}s {status}", flush=True)
    if status == "FAIL":
        # locate worst diffs: which source rank / chunk should have fed them
        out_c = out.float().cpu()
        ref_c = ref.cpu()
        top = torch.topk(diff.flatten().cpu(), 4)
        for v, idx in zip(top.values.tolist(), top.indices.tolist()):
            i, j = idx // N, idx % N
            src = i // M_per_rank
            chunk = (i % M_per_rank) // M_PER_CHUNK
            sig = src * m_chunk_num + chunk
            print(f"[rank {rank}]   diff={v:.5f} at ({i},{j}) src={src} "
                  f"chunk={chunk} flag[{sig}]={flags_u64[sig]} "
                  f"out={out_c[i, j].item():.5f} exp={ref_c[i, j].item():.5f}",
                  flush=True)
        # workspace DRAM view per source slot (rows the consumer should have read)
        wsn = M * K * 2
        hb = (ctypes.c_char * wsn)()
        rc = hip.hipMemcpy(hb, ctypes.c_void_p(ctx.ws_ptr), wsn, 2)
        assert rc == 0, f"hipMemcpy rc={rc}"
        ws = torch.from_numpy(
            np.frombuffer(hb, dtype=np.float16).copy().astype(np.float32)
        ).reshape(M, K)
        for src in range(size):
            d = (ws[src * M_per_rank:(src + 1) * M_per_rank]
                 - A_full[src * M_per_rank:(src + 1) * M_per_rank].float()
                 ).abs().max().item()
            print(f"[rank {rank}]   ws slot src={src} vs A_full[src]: "
                  f"max_diff={d:.5f}", flush=True)

print(f"[rank {rank}] AG_GEMM TEST {'PASS' if ok else 'FAIL'}", flush=True)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
