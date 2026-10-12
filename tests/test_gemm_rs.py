"""2-card DCU acceptance test: gemm_reduce_scatter (triton-dist port).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 test_gemm_rs.py
"""
import os
import sys

REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_gemmrs_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

from triton_dist.kernels.dcu import gemm_reduce_scatter as grs  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

M = N = K = 2048
DTYPE = torch.float16
SCALE = 0.02
Mpr = M // size

torch.manual_seed(999)
B = (torch.randn(N, K, device="cuda") * SCALE).half()

ctx = grs.create_gemm_rs_intra_node_context(
    max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
    mpi=mpi, comm=comm)

ok = True
for it in range(3):
    # fresh data every iteration: identical inputs would mask stale reads
    A_it = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A_it.append((torch.randn(M, K, device="cuda") * SCALE).half())
    with torch.no_grad():
        partial_it = [A_it[s].float() @ B.float().t() for s in range(size)]
        ref_it = torch.zeros_like(partial_it[0])
        for _p in partial_it:
            ref_it = ref_it + _p
    exp_it = ref_it[rank * Mpr:(rank + 1) * Mpr]

    out = grs.gemm_rs_intra_node(A_it[rank], B, ctx)
    torch.cuda.synchronize()
    diff = (out.float() - exp_it).abs()
    md = diff.max().item()
    status = "PASS" if md < 1e-2 else "FAIL"
    if status == "FAIL":
        ok = False
    print(f"[rank {rank}] iter {it} max_diff={md:.5f} "
          f"bad(>1e-2)={(diff > 1e-2).sum().item()} {status}", flush=True)
    if status == "FAIL":
        # scratch (local GEMM output) vs own full partial
        sc = ctx.scratch.float()
        own_partial = partial_it[rank]
        print(f"[rank {rank}]   scratch vs own C: "
              f"max_diff={(sc - own_partial).abs().max().item():.5f}", flush=True)
        # symmetric buffer (DRAM view) vs per-source shards
        nbytes = size * Mpr * N * 2
        hb = (ctypes.c_char * nbytes)()
        rc = hip.hipMemcpy(hb, ctypes.c_void_p(ctx.scatter_ptr), nbytes, 2)
        assert rc == 0, f"hipMemcpy rc={rc}"
        buf = torch.from_numpy(
            np.frombuffer(hb, dtype=np.float16).copy().astype(np.float32)
        ).reshape(size, Mpr, N)
        for s in range(size):
            want = partial_it[s][rank * Mpr:(rank + 1) * Mpr].cpu()
            print(f"[rank {rank}]   symm slice src={s}: "
                  f"max_diff={(buf[s] - want).abs().max().item():.5f} "
                  f"absmax={buf[s].abs().max().item():.5f}", flush=True)
        out_c = out.float().cpu()
        exp_c = exp_it.cpu()
        top = torch.topk((out_c - exp_c).abs().flatten(), 2)
        for v, idx in zip(top.values.tolist(), top.indices.tolist()):
            i, j = idx // N, idx % N
            print(f"[rank {rank}]   diff={v:.5f} at ({i},{j}): "
                  f"out={out_c[i, j].item():.5f} exp={exp_c[i, j].item():.5f} "
                  f"buf0={buf[0, i, j].item():.5f} buf1={buf[1, i, j].item():.5f}",
                  flush=True)
    else:
        print(f"[rank {rank}] iter {it} PASS", flush=True)

print(f"[rank {rank}] GEMM_RS TEST {'PASS' if ok else 'FAIL'}", flush=True)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
