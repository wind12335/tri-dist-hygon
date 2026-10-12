"""S11 follow-up: how much does cross-stream EVENT CHOREOGRAPHY cost?

The multi-stream matrix (s11_probe_multistream.py) showed real engine
concurrency, but wiring it into the ops (fork via wait_stream before the
copies, join via wait_stream after) REGRESSED every 2-card cell -- the
regression implies ~25-35 us per wait_stream. This probe isolates that tax:

  serial  : k copies on ONE stream                (today's fan-out)
  streams : k copies on k streams, no events      (the matrix's winner)
  evfork  : + fork/join via torch wait_stream     (what fan_out does)
  rawev   : + fork/join via raw hipEvent calls    (torch overhead removed)

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s11_probe_evtax.py
"""
import ctypes
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s11e_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    dcu_state, init_dcu, memcpy_async)

torch.cuda.set_device(rank)
init_dcu(mpi, comm, rank, size)
host = dcu_state()["host"]
peer = 1 - rank

# raw hip event entry points (HIP_EVENT_DISABLE_TIMING = 0x2)
hip.hipEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                        ctypes.c_uint]
hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
hip.hipEventRecord.restype = ctypes.c_int
hip.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_uint64]
hip.hipStreamWaitEvent.restype = ctypes.c_int
hip.hipEventSynchronize.argtypes = [ctypes.c_void_p]
hip.hipEventSynchronize.restype = ctypes.c_int


def mpi_barrier():
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


BYTES = 8 * (1 << 20) * 8
dst = host.malloc(BYTES)
dst_peer = host.ptr(dst, peer)
src = torch.empty(BYTES, dtype=torch.uint8, device="cuda")
src.fill_(rank + 0x41)

MODES = ("serial", "streams", "evfork", "rawev")


def run_block(k, b, mode):
    streams = [torch.cuda.Stream() for _ in range(k)]
    cur = torch.cuda.current_stream()
    # events reused across reps (like the real op's per-launch events, but
    # created once here so rep-to-rep measures choreography, not creation)
    if mode == "rawev":
        ev_fork = ctypes.c_void_p()
        assert hip.hipEventCreateWithFlags(ctypes.byref(ev_fork), 0x2) == 0
        ev_join = [ctypes.c_void_p() for _ in range(k)]
        for e in ev_join:
            assert hip.hipEventCreateWithFlags(ctypes.byref(e), 0x2) == 0
    t0 = time.perf_counter()
    if mode == "evfork":
        for s in streams:
            s.wait_stream(cur)
    elif mode == "rawev":
        assert hip.hipEventRecord(ev_fork, cur.cuda_stream) == 0
        for s in streams:
            assert hip.hipStreamWaitEvent(s.cuda_stream, ev_fork, 0) == 0
    for i in range(k):
        s = streams[i if mode != "serial" else 0]
        memcpy_async(dst_peer + i * b, src.data_ptr() + i * b, b,
                     s.cuda_stream)
    if mode in ("evfork", "rawev"):
        for i, s in enumerate(streams):
            if mode == "evfork":
                cur.wait_stream(s)
            else:
                assert hip.hipEventRecord(ev_join[i], s.cuda_stream) == 0
                assert hip.hipStreamWaitEvent(cur.cuda_stream, ev_join[i],
                                              0) == 0
        if mode == "rawev":
            for e in ev_join:
                assert hip.hipEventSynchronize(e) == 0
        else:
            cur.synchronize()
    else:
        for s in streams:
            s.synchronize()
    return time.perf_counter() - t0


RES = {}
for b_mb in (2, 8):
    b = b_mb * (1 << 20)
    for k in (2, 8):
        cell = {}
        for mode in MODES:
            mpi_barrier()
            ts = [run_block(k, b, mode) for _ in range(4)]
            cell[mode] = {"ms_min": min(ts) * 1e3,
                          "gbps": k * b / min(ts) / 1e9}
        RES[f"B{b_mb}MB_k{k}"] = cell
        s, st, ev, rw = (cell[m]["gbps"] for m in MODES)
        print(f"[rank {rank}] B{b_mb}MB x k{k}: serial {s:6.1f} | "
              f"streams {st:6.1f} | evfork {ev:6.1f} | rawev {rw:6.1f} GB/s"
              f"   evtax~{(cell['streams']['ms_min']-cell['evfork']['ms_min'])*1e3:.0f}us",
              flush=True)

if rank == 0:
    out = os.path.join(REPO, "bench", "results", "s11_evtax.json")
    with open(out, "w") as f:
        json.dump(RES, f, indent=1)
    print(f"[rank 0] results -> {out}", flush=True)

host.free(dst)
host.finalize()
mpi_barrier()
mpi.MPI_Finalize()
os._exit(0)
