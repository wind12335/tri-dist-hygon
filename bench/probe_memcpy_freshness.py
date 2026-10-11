"""Probe: after a COPY-ENGINE (hipMemcpyAsync) write into my symmetric
buffer, which local read variant sees fresh data when L2 already holds the
previous round's lines?

This gates the S4 producer redesign (host-side memcpy producer, upstream
style): if plain tl.load stays fresh, the AG/RS consumers need no change;
if not, consumers must read with .cg/volatile (DRAM-authoritative) and the
landing signal must come through a dushmem atomic (L2-coherent).

Protocol per round t (rank0 writes, rank1 reads):
  barrier -> rank0: pattern -> hipMemcpyAsync(peer_va) + sync
          -> barrier -> rank1: check variants V0 plain / V1 volatile / V2 .cg
  (pretouch before round 1 warms L2 with garbage = the stale condition)

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 probe_memcpy_freshness.py
"""
import os
import sys

REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_mf_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
hip.hipMemcpyAsync.restype = ctypes.c_int
hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    SymmBarrier, dcu_state, init_dcu)

NUM = 262144  # int32 per variant row (1 MiB x 3 rows = 3 MiB, fits L2)
NVAR = 3

init_dcu(mpi, comm, rank, size)
state = dcu_state()
host = state["host"]

sym = host.malloc(4 * NUM * NVAR)   # rank 1's rows under test
peer_va = host.ptr(sym, 1)          # rank 0's view of rank 1's buffer
barrier = SymmBarrier(size)


@triton.jit(do_not_specialize=["sym_base"])
def pretouch_kernel(sym_base, out_ptr, NUM_: tl.constexpr,
                    NVAR_: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    v = tl.program_id(axis=1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    p = (sym_base + (v * NUM_ + offs) * 4).to(tl.pointer_type(tl.int32))
    x = tl.load(p)
    tl.atomic_add(out_ptr + v, tl.sum((x == x).to(tl.int32), axis=0) & 0)


@triton.jit(do_not_specialize=["sym_base", "t"])
def check_kernel(sym_base, out_ptr, t, NUM_: tl.constexpr,
                 VARIANT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    exp = t * 1000000 + offs
    p = (sym_base + (VARIANT * NUM_ + offs) * 4).to(tl.pointer_type(tl.int32))
    if VARIANT == 0:
        x = tl.load(p)
    elif VARIANT == 1:
        x = tl.load(p, volatile=True)
    else:
        x = tl.load(p, cache_modifier=".cg")
    bad = tl.sum((x != exp).to(tl.int32), axis=0)
    tl.atomic_add(out_ptr + VARIANT, bad)


out = torch.zeros(NVAR, dtype=torch.int32, device="cuda")
pattern = torch.zeros(NVAR * NUM, dtype=torch.int32, device="cuda")

if rank == 1:
    pretouch_kernel[(NUM // 1024, NVAR)](sym, out, NUM_=NUM, NVAR_=NVAR,
                                         BLOCK=1024, num_warps=4)
torch.cuda.synchronize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))

for t in range(1, 6):
    barrier.round_barrier(rank)  # reader done with previous round
    if rank == 0:
        v = torch.arange(NUM * NVAR, dtype=torch.int64, device="cuda")
        pattern = (t * 1000000 + (v % NUM)).to(torch.int32).contiguous()
        torch.cuda.synchronize()
        rc = hip.hipMemcpyAsync(ctypes.c_void_p(peer_va),
                                ctypes.c_void_p(pattern.data_ptr()),
                                4 * NUM * NVAR, 3,
                                ctypes.c_void_p(torch.cuda.current_stream()
                                                .cuda_stream))
        assert rc == 0, f"hipMemcpyAsync rc={rc}"
        torch.cuda.synchronize()  # copy engine landed the data
    barrier.round_barrier(rank)  # atomic-path landing signal
    if rank == 1:
        out.zero_()
        for variant in range(3):
            check_kernel[(NUM // 1024,)](sym, out, t, NUM_=NUM,
                                         VARIANT=variant, BLOCK=1024,
                                         num_warps=4)
        torch.cuda.synchronize()
        counts = out.tolist()
        tag = " ".join(f"V{v}={c}" for v, c in enumerate(counts))
        names = ("plain", "volatile", ".cg")
        fresh = [names[i] for i, c in enumerate(counts) if c == 0]
        print(f"[rank 1] round {t}: {tag} -> fresh: "
              f"{fresh if fresh else 'NONE'}", flush=True)

barrier.round_barrier(rank)
torch.cuda.synchronize()
host.finalize()
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
