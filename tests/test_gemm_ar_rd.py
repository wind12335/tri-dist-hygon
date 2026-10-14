"""2-card DCU acceptance test: S12 windowed recursive-doubling panel
GEMM-AllReduce (gemm_ar_rd -- the paper #1 AR v23 design on DCU planes).

Stage 0 probes the ONE mechanism this port leans on that S1-S11 never
exercised: a PEER-ENGINE-written 4 B flag observed by hipStreamWaitValue32
(GTE) on OWN memory. If the stream-wait poll cannot see engine writes, the
same exchange is retried with the 1-CTA hybrid-spin fallback
(RD_WAIT_MODE=spin) and the whole test continues in that mode.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_gemm_ar_rd.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s12rd_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

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
hip.hipStreamQuery.restype = ctypes.c_int
hip.hipStreamQuery.argtypes = [ctypes.c_void_p]
assert hip.hipSetDevice(rank) == 0


def barrier():
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


def allreduce_fail(flag: bool) -> int:
    """Collective OR over ranks (the probe verdict is only meaningful on
    ranks 0/1 -- bystanders must adopt the same wait mode)."""
    mpi.MPI_Allreduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p, ctypes.c_void_p]
    mpi.MPI_Allreduce.restype = ctypes.c_int
    send = ctypes.c_int(1 if flag else 0)
    recv = ctypes.c_int(0)
    op_sum = pHost._sym_addr(mpi, "ompi_mpi_op_sum")
    assert mpi.MPI_Allreduce(ctypes.byref(send), ctypes.byref(recv), 1,
                             pHost.mpi_type_int(mpi), op_sum,
                             ctypes.c_void_p(comm)) == 0
    return recv.value


import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import gemm_ar_rd as rd  # noqa: E402
from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    dcu_state,
    hip as hip_fn,
    init_dcu,
    memcpy_async,
)

ok = True


def report(name, passed, detail=""):
    global ok
    if not passed:
        ok = False
    print(f"[rank {rank}] {name}: {'PASS' if passed else 'FAIL'} {detail}",
          flush=True)


def host_read_u32(va):
    buf = ctypes.c_uint32()
    assert hip.hipMemcpy(ctypes.byref(buf), ctypes.c_void_p(va), 4, 2) == 0
    return buf.value


def stream_drained(stream, timeout_s):
    """Host-side watchdog: poll hipStreamQuery until the wait command
    completes (rc==0) or the deadline passes (rc!=0 pending or sticky)."""
    deadline = time.time() + timeout_s
    while True:
        rc = hip.hipStreamQuery(ctypes.c_void_p(stream.cuda_stream))
        if rc == 0:
            return True
        if time.time() > deadline:
            return False
        time.sleep(0.005)


# ------------------------------------------------- stage 0: flag-plane probe --
init_dcu(mpi, comm, rank, size)
h = hip_fn()
host = dcu_state()["host"]
probe_flag = host.malloc(4)
h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
h.hipMemset.restype = ctypes.c_int
assert h.hipMemset(ctypes.c_void_p(probe_flag), 0, 4) == 0
probe_stage = torch.zeros(1, dtype=torch.int32, device="cuda")
torch.cuda.synchronize()
barrier()


def probe_exchange(val, mode):
    """rank0 engine-writes `val` into rank1's flag; rank1 waits for it.
    Returns (ok, timed_out). Other ranks are bystanders."""
    st = torch.cuda.Stream()
    if rank == 0:
        probe_stage.fill_(val)
        with torch.cuda.stream(st):
            memcpy_async(host.ptr(probe_flag, 1), probe_stage.data_ptr(),
                         4, st.cuda_stream)
        torch.cuda.synchronize()
    elif rank == 1:
        api_err = None
        with torch.cuda.stream(st):
            try:
                rd._wait_ticket(probe_flag, val, st, mode)
            except AssertionError as e:
                api_err = str(e)
        if api_err is not None:
            barrier()
            return False, False, api_err
        if not stream_drained(st, 10.0):
            barrier()
            return False, True, "wait never satisfied"
        got = host_read_u32(probe_flag)
        barrier()
        return got == val, False, f"flag={got}"
    barrier()
    return True, False, ""


mode = "stream"
r1, t1, d1 = probe_exchange(11111, "stream")
if rank in (0, 1):
    report("probe stream-wait vs engine write (0->1)", r1, d1)
if allreduce_fail(not r1):
    # engine->stream-wait coherence is broken (or the API rejects it):
    # verify the spin fallback on the same exchange.
    r2, t2, d2 = probe_exchange(22222, "spin")
    if rank in (0, 1):
        report("probe spin-kernel vs engine write (0->1)", r2, d2)
    if allreduce_fail(not r2):
        report("flag plane", False, "both stream and spin waits failed")
        dcu_state()["host"].finalize()
        barrier()
        mpi.MPI_Finalize()
        os._exit(1)
    mode = "spin"
    if rank == 0:
        print("[rank 0] NOTE: running the op tests with "
              "RD_WAIT_MODE=spin fallback", flush=True)
host.free(probe_flag)
barrier()

# ------------------------------------------------------- op correctness --
M = N = K = 2048
DTYPE = torch.float16
SCALE = 0.02

B = []
for s in range(size):
    torch.manual_seed(999 + s)
    B.append((torch.randn(N, K, device="cuda") * SCALE).half())

ctx = rd.create_gemm_ar_rd_context(
    max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm, wait_mode=mode)
print(f"[rank {rank}] ctx: chunk_rows={ctx.chunk_rows} "
      f"num_chunks={ctx.num_chunks} slots={ctx.stage_slots} "
      f"partners={ctx.partners} wait_mode={ctx.wait_mode}", flush=True)


def run_round(it, M_this, tag):
    A_it = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A_it.append((torch.randn(M_this, K, device="cuda") * SCALE).half())
    with torch.no_grad():
        ref = A_it[0].float() @ B[0].float().t()
        for s in range(1, size):
            ref += A_it[s].float() @ B[s].float().t()
    out = torch.empty(M_this, N, dtype=DTYPE, device="cuda")
    rd.gemm_ar_rd_op(A_it[rank], B[rank], out, ctx)
    torch.cuda.synchronize()
    diff = (out.float() - ref).abs()
    md = diff.max().item()
    report(f"{tag} iter {it}", md < 1e-2,
           f"max_diff={md:.5f} bad={(diff > 1e-2).sum().item()}")
    return md


for it in range(3):
    run_round(it, M, "full")
for it in range(8):
    run_round(100 + it, M, "reuse")
for it in range(2):
    run_round(200 + it, 1024, "M1024")

# gemm-only decomposition: c must equal MY rank's partial
torch.manual_seed(7777)
a_g = (torch.randn(M, K, device="cuda") * SCALE).half()
out_g = torch.empty(M, N, dtype=DTYPE, device="cuda")
rd.gemm_ar_rd_op(a_g, B[rank], out_g, ctx, component="gemm")
torch.cuda.synchronize()
with torch.no_grad():
    ref_g = a_g.float() @ B[rank].float().t()
md = (out_g.float() - ref_g).abs().max().item()
report("gemm-only", md < 1e-2, f"max_diff={md:.5f}")

rd.finalize_gemm_ar_rd_context(ctx)

print(f"[rank {rank}] GEMM_AR_RD TEST {'PASS' if ok else 'FAIL'} "
      f"(wait_mode={mode})", flush=True)
dcu_state()["host"].finalize()
barrier()
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
