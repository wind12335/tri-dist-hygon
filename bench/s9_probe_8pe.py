"""S9 probe: 8-PE dushmem bootstrap + symmetric-heap capacity on the 8-card
node (gates the 8-card bench shapes for moe_a2a, whose recv+out buffers are
the largest symmetric allocations in the repo).

mpirun --allow-run-as-root -np 8 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s9_probe_8pe.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s9p_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
print(f"[rank {rank}] MPI up, size={size}", flush=True)

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

# peer-access matrix (dushmem topo requires it)
hip.hipDeviceCanAccessPeer.restype = ctypes.c_int
hip.hipDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int),
                                       ctypes.c_int, ctypes.c_int]
acc = ctypes.c_int(0)
if rank == 0:
    bad = []
    for d in range(size):
        assert hip.hipDeviceCanAccessPeer(ctypes.byref(acc), 0, d) == 0
        if not acc.value:
            bad.append(d)
    print(f"[rank 0] peer access from dev0: "
          f"{'ALL OK' if not bad else 'MISSING ' + str(bad)}", flush=True)

import torch  # noqa: E402

from triton_dist.kernels.dcu.common_ops import dcu_state, init_dcu  # noqa: E402

torch.cuda.set_device(rank)
init_dcu(mpi, comm, rank, size)
host = dcu_state()["host"]
print(f"[rank {rank}] dushmem up: PE {host.my_pe()}/{host.n_pes()}",
      flush=True)

# laddered malloc: keep the largest that succeeds, then free everything
kept = []
for mb in (64, 256, 512, 1024, 2048):
    try:
        p = host.malloc(mb * (1 << 20))
        kept.append((p, mb))
        print(f"[rank {rank}] malloc {mb} MB OK", flush=True)
    except Exception as e:
        print(f"[rank {rank}] malloc {mb} MB FAIL: {e}", flush=True)
        break
for p, mb in kept:
    host.free(p)
print(f"[rank {rank}] freed; largest={kept[-1][1] if kept else 0} MB", flush=True)

host.finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
