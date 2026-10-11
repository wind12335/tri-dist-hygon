"""S6 decision probe: concurrency scaling of tiny device dushmem atomics.

The new-upstream gemm_allreduce signals per GEMM tile FROM the GEMM kernel
(device-initiated). The S4 verdict (~19 MB/s) was measured on BULK device
transfers from a single CTA -- the open question is whether many CTAs
issuing TINY atomics (8 B each) in parallel pipeline through the slow path.

Each variant launches G CTAs; CTA pid sets per_cta u64 slots in the PEER's
symmetric buffer to `round` (both ranks send simultaneously), then quiets.
Receiver verifies every slot afterwards.

Decision rule for the gemm_allreduce port:
  per-op latency ~< 2 us at (1024, 1)/(512, 2)  -> keep upstream's
  fine-grained per-tile signals;
  ~tens of us/op (serial slow path)             -> host-orchestrated chunks.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s6_signal_concurrency.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s6conc_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2
peer = 1 - rank

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
hip.hipMemcpy.restype = ctypes.c_int
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int]
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from pydushmem import device as dushmem_dev  # noqa: E402
from triton_dist.kernels.dcu.common_ops import init_dcu, dcu_state  # noqa: E402

init_dcu(mpi, comm, rank, size)
host = dcu_state()["host"]
extern_libs = dcu_state()["extern_libs"]
torch.cuda.set_device(rank)

SLOTS = 32768
VARIANTS = [  # (grid, per_cta) -> total 8B atomics per rank per round
    # decision-relevant concurrency variants first, stress last
    (1, 1), (1, 1024), (1024, 1), (512, 2), (64, 16), (512, 64),
]
ROUNDS = 20

sym = host.malloc(SLOTS * 8)  # symmetric u64 slots (own VA)
# zero OWN buffer (safe: peer writes ride RMA, which updates our L2)
hip.hipMemset(ctypes.c_void_p(sym), 0, SLOTS * 8)


@triton.jit(do_not_specialize=["dst", "pe", "round", "per_cta"])
def k_atomic_fan(dst, pe, round, per_cta):
    pid = tl.program_id(0)
    base = pid * per_cta
    for i in range(per_cta):
        dushmem_dev.dushmem_uint64_atomic_set(dst + (base + i) * 8, round, pe)
    dushmem_dev.dushmem_quiet()


@triton.jit(do_not_specialize=["round", "total"])
def k_check(dst, round, total, bad_ptr):
    pid = tl.program_id(0)
    p32 = dst.to(tl.pointer_type(tl.int32))  # low word; round < 2^31
    n = 0
    for i in range(pid, total, tl.num_programs(0)):
        v = tl.atomic_add(p32 + i * 2, 0)
        n += (v != round).to(tl.int32)
    # NOTE: no `if n > 0` guard -- scalar tl.atomic_add inside an scf.if
    # failed to compile on this triton; adding 0 is free.
    tl.atomic_add(bad_ptr, n)


def hsync():
    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


bad = torch.zeros(1, dtype=torch.int32, device="cuda")

# warmup with REAL VAs (slot-typing) + register the module
ck = k_atomic_fan.warmup(sym, peer, 7, 4, grid=(8,), extern_libs=extern_libs,
                         num_warps=4)
ck._init_handles()
host.cumodule_init(ck.module)
ckc = k_check.warmup(sym, 7, SLOTS, bad, grid=(64,))  # tensor -> pointer slot
ckc._init_handles()

hsync()

rnd = 0
for (g, per) in VARIANTS:
    total = g * per
    assert total <= SLOTS
    # cap rounds on bigger variants: if the slow path serializes at
    # ~ms/op, a 32768-op round alone can take minutes
    rounds = 20 if total <= 128 else (5 if total <= 2048 else 3)
    ts = []
    for _ in range(rounds):
        rnd += 1
        hsync()  # both ranks launch together (symmetric traffic)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        ck[(g, 1, 1)](sym, peer, rnd, per)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
        # clear for the next round: local memset of own slots
        hip.hipMemset(ctypes.c_void_p(sym), 0, total * 8)
        hsync()
    ts.sort()
    p50 = ts[len(ts) // 2]
    us_per_op = p50 * 1000.0 / total
    aggr_MBps = total * 8 / 1024 / (p50 / 1000.0) / 1024
    print(f"[rank {rank}] G={g:5d} x {per:3d} = {total:6d} ops: "
          f"p50 {p50:8.3f} ms  {us_per_op:8.2f} us/op  "
          f"aggregate {aggr_MBps:8.1f} MB/s", flush=True)
    # decision variants done; if even 512 concurrent CTAs don't pipeline,
    # skip the stress variants (minutes per round on the slow path)
    if (g, per) == (512, 2) and us_per_op > 50:
        print(f"[rank {rank}] >50 us/op at 512 CTAs x 2 -- skipping "
              f"stress variants", flush=True)
        break

# delivery correctness: one fan + remote check
rnd += 1
hsync()
ck[(512, 1, 1)](sym, peer, rnd, 8)  # 4096 slots
torch.cuda.synchronize()
hsync()
bad.zero_()
ckc[(64, 1, 1)](sym, rnd, 4096, bad)  # only the 512x8 delivered slots
torch.cuda.synchronize()
hsync()
n_bad = bad.item()
print(f"[rank {rank}] delivery check: bad={n_bad} "
      f"{'OK' if n_bad == 0 else 'FAIL'}", flush=True)

host.finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
