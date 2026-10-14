"""S14 stage-0 probe (2-card): engine PULL (dst=local, src=peer symmetric VA)
vs engine PUSH (dst=peer, src=local) throughput.

The paper #1 AG design is consumer-side PULL with all-local readiness flags;
S1-S11 only ever exercised the push direction. If pull runs at push speed,
the design ports cleanly.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/probe_engine_pull.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_pullprobe_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    dcu_state,
    hip as hip_fn,
    init_dcu,
    memcpy_async,
)

init_dcu(mpi, comm, rank, size)
h = hip_fn()
host = dcu_state()["host"]

MB = 8
BYTES = MB << 20
src_buf = host.malloc(BYTES)
dst_buf = host.malloc(BYTES)

st = torch.cuda.Stream()


def timed_dir(name, dst, src):
    for _ in range(5):
        memcpy_async(dst, src, BYTES, st.cuda_stream)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(st):
            ev0.record()
            memcpy_async(dst, src, BYTES, st.cuda_stream)
            ev1.record()
        torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    ts = sorted(ts)
    t = ts[len(ts) // 2]
    gbps = BYTES / (t * 1e-3) / 1e9
    print(f"[rank {rank}] {name}: {t:.3f} ms  {gbps:.1f} GB/s", flush=True)
    return gbps


# fill own src heap with a recognizable pattern, then sync
torch_fill = torch.full((BYTES // 4,), 0x01020300 + rank, dtype=torch.int32,
                        device="cuda")
h.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                        ctypes.c_int]
assert h.hipMemcpy(ctypes.c_void_p(src_buf), torch_fill.data_ptr(), BYTES,
                   2) == 0
torch.cuda.synchronize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0

peer = 1 - rank
# PUSH: my local fill -> peer's dst_buf
timed_dir("engine PUSH (local->peer)",
          host.ptr(dst_buf, peer), src_buf)
# PULL: peer's src_buf -> my dst_buf
timed_dir("engine PULL (peer->local)",
          dst_buf, host.ptr(src_buf, peer))

# verify pull data correctness (peer's pattern) BEFORE the local test
# overwrites dst_buf
got = torch.empty(BYTES // 4, dtype=torch.int32, device="cuda")
assert h.hipMemcpy(got.data_ptr(), ctypes.c_void_p(dst_buf), BYTES, 2) == 0
expect = 0x01020300 + peer
ok = bool((got == expect).all().item())
print(f"[rank {rank}] PULL data check: {'PASS' if ok else 'FAIL'}", flush=True)

# LOCAL: symmetric->own other buffer (the S14 stage copy)
timed_dir("engine LOCAL (own->own)",
          dst_buf, src_buf)

host.free(src_buf)
host.free(dst_buf)
dcu_state()["host"].finalize()
assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
