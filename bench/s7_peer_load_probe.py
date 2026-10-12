"""S7 decision probe: device-side PLAIN LOADS from the PEER's dushmem
symmetric heap (the upstream a2a combine's pull model).

Upstream combine reads expert outputs via dl.symm_at(input_buf, expert_rank)
+ plain ld -- remote reads of peer-written (locally-stored) data. On this
stack nothing has ever device-read the peer heap (all our planes push INTO
the local buffer). Measures: correctness, freshness across re-writes
(barrier between rounds), and bandwidth at 16 MB.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s7_peer_load_probe.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s7pl_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2
peer = 1 - rank

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu.common_ops import dcu_state, init_dcu  # noqa: E402

init_dcu(mpi, comm, rank, size)
host = dcu_state()["host"]
torch.cuda.set_device(rank)

N = 4 * 1024 * 1024  # int32 elems = 16 MB


@triton.jit(do_not_specialize=["base", "n", "seed"])
def k_fill(base, n, seed, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    p = base.to(tl.pointer_type(tl.int32))
    tl.store(p + offs, offs + seed, mask=m)


@triton.jit(do_not_specialize=["base", "n", "BLOCK: tl.constexpr"])
def k_sum(base, n, out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    p = base.to(tl.pointer_type(tl.int32))
    v = tl.load(p + offs, mask=m, other=0).to(tl.int64)
    tl.atomic_add(out_ptr, tl.sum(v))


sym = host.malloc(N * 4)
peer_va = host.ptr(sym, peer)
out = torch.zeros(1, dtype=torch.int64, device="cuda")


def hsync():
    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


grid = (triton.cdiv(N, 4096),)
ok = True
for rnd in range(3):
    seed = 1000 * peer + rnd  # peer re-writes between rounds
    hsync()  # peer fill starts together; both fill their OWN buffer
    k_fill[grid](sym, N, 1000 * rank + rnd, BLOCK=4096)
    hsync()  # fills done -> now read the PEER's buffer
    out.zero_()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    k_sum[grid](peer_va, N, out, BLOCK=4096)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)
    got = out.item()
    # expected sum of (i + seed) for i in [0, N)
    exp = N * (N - 1) // 2 + N * seed
    good = got == exp
    ok = ok and good
    print(f"[rank {rank}] round {rnd}: sum {'OK' if good else 'BAD'} "
          f"(got {got}, exp {exp})  {ms:.3f} ms  "
          f"{N * 4 / 1e9 / (ms / 1e3):.1f} GB/s", flush=True)

print(f"[rank {rank}] PEER-LOAD PROBE {'PASS' if ok else 'FAIL'}", flush=True)

host.finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
