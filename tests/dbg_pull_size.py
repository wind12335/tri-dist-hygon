"""Pull-kernel size micro-repro v2 (2-card): kernel_i1_pull at big chunk
sizes, independent of the op's stream topology.

Fixes over v1 (which raced: both directions used the SAME ws slot, so
one rank's pull-source was concurrently overwritten by the peer's pull
-- one side always won, alternating with scheduling):
  - two DISJOINT slots (rank r stages at slot r; pull dst = my slot for
    the peer's id, src = peer's slot for its own id) -- exactly the op's
    (src, ck) slot discipline;
  - variant A: arrive PRE-SET before the pull launches (isolates the
    copy/counter path);
  - variant B: arrive set LATE (peer host-sleeps 300ms, then xchg) --
    isolates the in-kernel spin path at full size, the op's regime.

Sweep CR=K in {1024, 1536, 2048} -> chunk {2MB, 4.5MB, 8MB}. The op
hangs at >=4.5MB with kernel pulls while engine pulls pass at 8MB
(/tmp/l1_probe4_*); bisect already exonerated grid size, chunk count,
tile count.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x I1_PULL=kernel python3 tests/dbg_pull_size.py
"""
import ctypes
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_psz_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)

hipcd = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hipcd.hipSetDevice.restype = ctypes.c_int
assert hipcd.hipSetDevice(rank) == 0

import torch  # noqa: E402
import triton  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state, init_dcu  # noqa: E402

mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int


def align():
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


init_dcu(mpi, comm, rank, size)
s = dcu_state()
host = s["host"]
CHUNK_MAX = 2048 * 2048 * 2  # bytes per slot
SLOT = [0, CHUNK_MAX]  # rank r's data lives at slot r
ws_ptr = host.malloc(2 * CHUNK_MAX)
peer_ws = [host.ptr(ws_ptr, d) for d in range(size)]
arrive_ptr = host.malloc(64)
peer_arrive = [host.ptr(arrive_ptr, d) for d in range(size)]
hipcd.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
hipcd.hipMemset(ctypes.c_void_p(arrive_ptr), 0, 64)

ready = torch.zeros(8, dtype=torch.int32, device="cuda")
done = torch.zeros(8, dtype=torch.int32, device="cuda")
print(f"[rank {rank}] topology: ws={hex(ws_ptr)} "
      f"peer_ws={[hex(p) for p in peer_ws]}", flush=True)


def run_variant(tag, CR, K, late):
    """late=False: arrive pre-set (copy path). late=True: peer sleeps
    300ms then xchg's my cell (spin path)."""
    words = (CR * K * 2) // 4
    grid = triton.cdiv(words, i1._PULL_BLOCK)
    my_slot, peer_slot = SLOT[rank], SLOT[1 - rank]
    ready.zero_()
    done.zero_()
    i1._launch_flag_xchg(arrive_ptr + 0, 0)  # cell clean
    torch.cuda.synchronize()
    align()

    src_t = torch.full((words,), rank + 1, dtype=torch.int32, device="cuda")
    i1.kernel_i1_copy[(grid, 1, 1)](ws_ptr + my_slot, src_t.data_ptr(),
                                    words, BLOCK=i1._PULL_BLOCK)
    torch.cuda.synchronize()
    align()

    if not late:
        i1._launch_flag_xchg(peer_arrive[1 - rank] + 0, 1)
        torch.cuda.synchronize()
        align()
    t0 = time.perf_counter()
    i1.kernel_i1_pull[(grid, 1, 1)](
        ws_ptr + peer_slot, peer_ws[1 - rank] + peer_slot,
        arrive_ptr, ready.data_ptr(), done.data_ptr(),
        1, words, BLOCK=i1._PULL_BLOCK, GRID=grid, num_warps=4)
    if late:
        # release the peer's spin only after 300ms of host wall time.
        # The xchg MUST go on a side stream: the pull (spinning) sits on
        # the default stream, and a same-stream xchg would queue behind
        # its own waiter -- the self-deadlock the S14f bisect already
        # documented. In the real op the peer's GPU writes this cell.
        time.sleep(0.3 if rank == 0 else 0.05)
        xs = torch.cuda.Stream()
        with torch.cuda.stream(xs):
            i1._launch_flag_xchg(peer_arrive[1 - rank] + 0, 1)
        xs.synchronize()
    ev = torch.cuda.Event()
    ev.record()
    ev.synchronize()
    ms = (time.perf_counter() - t0) * 1e3

    got = torch.zeros(words, dtype=torch.int32, device="cuda")
    i1.kernel_i1_copy[(grid, 1, 1)](got.data_ptr(), ws_ptr + peer_slot,
                                    words, BLOCK=i1._PULL_BLOCK)
    torch.cuda.synchronize()
    good = bool((got == (1 - rank) + 1).all().item())
    mb = CR * K * 2 / 1e6
    print(f"[rank {rank}] {tag} CR={CR} chunk={mb:.1f}MB grid={grid} "
          f"ms={ms:.2f} verify={'OK' if good else 'BAD'} "
          f"ready={ready[0].item()} done={done[0].item()}", flush=True)
    align()
    return good


ok_all = True
for CR, K in [(1024, 1024), (1536, 1536), (2048, 2048)]:
    ok_all &= run_variant("A(pre)", CR, K, late=False)
for CR, K in [(1024, 1024), (1536, 1536), (2048, 2048)]:
    ok_all &= run_variant("B(late)", CR, K, late=True)

print(f"[rank {rank}] PULLSIZE {'PASS' if ok_all else 'FAIL'}", flush=True)
host.finalize()
align()
mpi.MPI_Finalize()
os._exit(0 if ok_all else 1)
