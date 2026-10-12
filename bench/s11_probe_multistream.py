"""S11 probe: do multiple copy-engine streams run CONCURRENTLY on DCU?

The new upstream's AG producer pushes the fan-out across MULTIPLE streams
(cp_engine_producer_all_gather_full_mesh_push_multi_stream). Our S9 8-card
wall (17-25 GB/s/rank effective) pushes (R-1) peers through ONE engine
stream -- if this stack has >=2 physical copy queues, a per-dest stream
pool multiplies fan-out bandwidth; if all streams serialize onto one
queue, the idea is dead. Decidable at 2 cards.

Matrix (per copy size B in {1, 8, 32} MB, fan-out k in {1, 2, 4, 8}):
  serial   : k copies of B on ONE stream   (today's AG push shape)
  streams  : k copies of B on k streams    (upstream's multi-stream shape)
  duplex   : both ranks do `streams` simultaneously (the real AG pattern)
Verdict: aggregate GB/s scaling with k under `streams`.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s11_probe_multistream.py
"""
import ctypes
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s11_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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

# async-engine count (corroborating evidence; hipDeviceAttributeAsyncEngineCount=16)
hip.hipDeviceGetAttribute.restype = ctypes.c_int
hip.hipDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int),
                                      ctypes.c_int, ctypes.c_int]
engines = ctypes.c_int(-1)
hip.hipDeviceGetAttribute(ctypes.byref(engines), 16, rank)
print(f"[rank {rank}] hipDeviceAttributeAsyncEngineCount = {engines.value}",
      flush=True)

init_dcu(mpi, comm, rank, size)
host = dcu_state()["host"]
peer = 1 - rank

BYTES = 32 * (1 << 20)      # per-copy cap
KMAX = 8
TOTAL = BYTES * KMAX        # 256 MB symmetric dest, disjoint regions per k

dst = host.malloc(TOTAL)
dst_peer = host.ptr(dst, peer)   # peer's VA for the same symmetric slot
src = torch.empty(TOTAL, dtype=torch.uint8, device="cuda")
src.fill_(rank + 0x41)


def mpi_barrier():
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


def run_block(k, b, nstreams):
    """k copies of b bytes, one per stream in `nstreams` distinct streams
    (nstreams=1 -> all on one stream). Disjoint peer regions. Host wall
    around issue+sync (cross-stream aggregate; host overhead ~us)."""
    streams = [torch.cuda.Stream() for _ in range(max(nstreams, 1))]
    for w in range(3):
        t0 = time.perf_counter()
        for i in range(k):
            s = streams[i % nstreams]
            with torch.cuda.stream(s):
                memcpy_async(dst_peer + i * b, src.data_ptr() + i * b, b,
                             s.cuda_stream)
        for s in streams:
            s.synchronize()
        t1 = time.perf_counter()
        if w == 2:
            return t1 - t0
    return None


RES = {"engines": engines.value, "cells": {}}
for b_mb in (2, 4):
    b = b_mb * (1 << 20)
    for k in (1, 2, 4, 8):
        if k * b > TOTAL:
            continue
        mpi_barrier()
        t_ser = run_block(k, b, 1)
        mpi_barrier()
        t_str = run_block(k, b, k)
        # full duplex: both ranks push concurrently
        mpi_barrier()
        t_dup = run_block(k, b, k)
        mpi_barrier()
        cell = {
            "serial_ms": t_ser * 1e3, "streams_ms": t_str * 1e3,
            "duplex_ms": t_dup * 1e3,
            "gbps_serial": k * b / t_ser / 1e9,
            "gbps_streams": k * b / t_str / 1e9,
            "gbps_duplex": k * b / t_dup / 1e9,
        }
        RES["cells"][f"B{b_mb}MB_k{k}"] = cell
        print(f"[rank {rank}] B{b_mb:>2}MB x k{k}: "
              f"serial {cell['gbps_serial']:6.1f} GB/s | "
              f"{k} streams {cell['gbps_streams']:6.1f} GB/s | "
              f"duplex {cell['gbps_duplex']:6.1f} GB/s", flush=True)

if rank == 0:
    out = os.path.join(REPO, "bench", "results",
                       "s11_multistream_2_4mb.json")
    with open(out, "w") as f:
        json.dump(RES, f, indent=1)
    print(f"[rank 0] results -> {out}", flush=True)

host.free(dst)
host.finalize()
mpi_barrier()
mpi.MPI_Finalize()
os._exit(0)
