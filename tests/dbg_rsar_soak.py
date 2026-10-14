"""RS-GEMM / AR-GEMM numerical-correctness soak: the control experiment
for the AG I1 consumer tear.

The AG+GEMM path tears at ~0.5-1%/rank-round with a precise signature:
contiguous tile-aligned row runs, ~2.4% of each damaged row's columns
over the 1e-2 threshold, per-element error std 0.0044 = exactly ONE
BLOCK_K=64 k-slice contribution, and the wrong values bit-match nothing
(same-round v9, prev-round v10). This soak runs the same long-round
regime on the two successfully ported paper-#1 overlaps to split the
hypothesis space:

  rs -- producer = plain LOCAL banded GEMM (local a/b tensors only) +
       engine-push slots + triton reduce reading OWN symmetric heap.
       k-slice-signature tear  => the plain local GEMM class itself
                                  occasionally miscomputes;
       dense/panel-signature   => slot-plane (engine write vs triton
                                  read) corruption.
  ar -- same producer class + engine push + triton accumulate reading
       OWN symmetric heap. Same signature reading.

The implied-delta-A forensics solve e = dA_j @ B_j^T per source rank j
(B_j square full-rank): a k-slice event gives a SPARSE (~64-wide,
64-aligned) dA for exactly one j; a panel/slot event gives DENSE dA for
every j with a much larger e std (~0.02 vs ~0.0044).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  -x DBG_RSAR=rs python3 tests/dbg_rsar_soak.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RANK_ENV = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_rsar_{RANK_ENV}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size in (2, 8)

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import gemm_ar_pred as ap  # noqa: E402
from triton_dist.kernels.dcu import gemm_ar_rd as rd  # noqa: E402
from triton_dist.kernels.dcu import gemm_rs_v5 as rs5  # noqa: E402
from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    dcu_state,
    hip as hip_fn,
    init_dcu,
    memcpy_async,
)

import threading  # noqa: E402

USE_PRED = os.environ.get("DBG_PREDOP", "0") == "1"
GATE = os.environ.get("DBG_GATE", "0") == "1"
REUSE_OUT = os.environ.get("DBG_REUSE", "0") == "1"
SHARED_OUT = []
rel_t = torch.zeros(64, dtype=torch.int32, device="cuda")
rel_va = rel_t.data_ptr()
rel_s = torch.cuda.Stream()

OP = os.environ.get("DBG_RSAR", "rs")
ROUNDS = int(os.environ.get("DBG_ROUNDS", "600"))
M = int(os.environ.get("DBG_M", "4096"))          # per-rank A rows
N = int(os.environ.get("DBG_N", "2048"))
K = int(os.environ.get("DBG_K", "2048"))
DTYPE = torch.float16
SCALE = 0.02
PRINT_FAILS = 6

print(f"[rank {rank}] soak OP={OP} ROUNDS={ROUNDS} "
      f"M={M} N={N} K={K}", flush=True)


def barrier():
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


def allreduce_fail(flag):
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


init_dcu(mpi, comm, rank, size)
h = hip_fn()
host = dcu_state()["host"]

# ------------------------------------------- AR only: flag-plane probe --
mode = "stream"
if OP == "ar":
    probe_flag = host.malloc(4)
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    assert h.hipMemset(ctypes.c_void_p(probe_flag), 0, 4) == 0
    probe_stage = torch.zeros(1, dtype=torch.int32, device="cuda")
    torch.cuda.synchronize()
    barrier()

    def host_read_u32(va):
        buf = ctypes.c_uint32()
        assert hip.hipMemcpy(ctypes.byref(buf), ctypes.c_void_p(va), 4, 2) == 0
        return buf.value

    def stream_drained(stream, timeout_s):
        deadline = time.time() + timeout_s
        while True:
            rc = hip.hipStreamQuery(ctypes.c_void_p(stream.cuda_stream))
            if rc == 0:
                return True
            if time.time() > deadline:
                return False
            time.sleep(0.005)

    def probe_exchange(val, wmode):
        st = torch.cuda.Stream()
        if rank == 0:
            probe_stage.fill_(val)
            with torch.cuda.stream(st):
                memcpy_async(host.ptr(probe_flag, 1), probe_stage.data_ptr(),
                             4, st.cuda_stream)
            torch.cuda.synchronize()
        elif rank == 1:
            err = None
            with torch.cuda.stream(st):
                try:
                    rd._wait_ticket(probe_flag, val, st, wmode)
                except AssertionError as e:
                    err = str(e)
            if err is None and not stream_drained(st, 10.0):
                err = "wait never satisfied"
            ok1 = err is None and host_read_u32(probe_flag) == val
            barrier()
            return ok1
        barrier()
        return True

    r1 = probe_exchange(11111, "stream")
    if allreduce_fail(not r1):
        r2 = probe_exchange(22222, "spin")
        assert allreduce_fail(not r2) == 0, "flag plane broken"
        mode = "spin"
    host.free(probe_flag)
    barrier()
    print(f"[rank {rank}] probe done wait_mode={mode}", flush=True)

# ------------------------------------------------------- contexts, B --
B = []
for s in range(size):
    torch.manual_seed(999 + s)
    B.append((torch.randn(N, K, device="cuda") * SCALE).half())

if OP == "rs":
    ctx = rs5.create_gemm_rs_v5_context(
        max_M=M, N=N, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm)
    print(f"[rank {rank}] rs ctx: chunk_rows={ctx.chunk_rows} "
          f"slots={ctx.stage_slots} lanes={ctx.comm_lanes} "
          f"wait_mode={ctx.wait_mode}", flush=True)
else:
    cr = int(os.environ.get("DBG_CR", "0"))
    ctx = rd.create_gemm_ar_rd_context(
        max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm, wait_mode=mode,
        chunk_rows=(cr if cr > 0 else 0))
    print(f"[rank {rank}] ar ctx: chunk_rows={ctx.chunk_rows} "
          f"slots={ctx.stage_slots} partners={ctx.partners} "
          f"wait_mode={ctx.wait_mode}", flush=True)

fails = 0
fails_printed = 0
worst = 0.0
_prev_A = None
m_pr = M // size
n_tasks_all = (M + ctx.chunk_rows - 1) // ctx.chunk_rows

with torch.no_grad():
    # pinv per source rank for the implied-delta-A solve
    PINV = [torch.linalg.pinv(B[s].float()).t() for s in range(size)]

for it in range(ROUNDS):
    A = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A.append((torch.randn(M, K, device="cuda") * SCALE).half())
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if OP == "rs":
        out = torch.empty(m_pr, N, dtype=DTYPE, device="cuda")
        rs5.gemm_rs_v5_op(A[rank], B[rank], out, ctx)
    elif USE_PRED:
        # bisect arm: my order/gate op variant (natural order; optional
        # victim-push gate on task 0 released by a host thread; optional
        # concurrent host observer polling the arrival flag words)
        # DBG_PERM=V -> victim V last every round. DBG_SEQ=1 -> replicate
        # the bench per-tick arm sequence [nopred(natural+observer),
        # pred(natural), oracle(V last), oracle(V last)] with the gate on
        # V in EVERY round -- the configuration that failed the smoke.
        pv = int(os.environ.get("DBG_PERM", "-1"))
        seq = os.environ.get("DBG_SEQ", "0") == "1"
        gt = int(os.environ.get("DBG_GT", str(pv if pv >= 0 else 0)))
        order = None
        do_obs = os.environ.get("DBG_OBS", "0") == "1"
        if pv >= 0:
            order = [i for i in range(n_tasks_all) if i != pv] + [pv]
            if rank == 0:
                print(f"[rank 0] perm order: victim {pv} last",
                      flush=True)
        if seq:
            ph = it % 4
            order = ([i for i in range(n_tasks_all) if i != gt] + [gt]
                     if ph >= 2 else None)
            do_obs = do_obs or (ph == 0)
            if rank == 0:
                print(f"[rank 0] seq arm {ph}: "
                      f"{'natural' if order is None else f'v{gt} last'}"
                      f"{' +obs' if do_obs else ''}", flush=True)
        gates = []
        if GATE and rank == 1:
            # the op assigns tickets by SEQUENCE position (base+p+1) --
            # a slot-occupant GTE chain stays monotone under permutation
            # only with position-keyed tickets (task-keyed tickets made
            # slot 3's cell go 8,12,16,4 and the victim's wait was
            # satisfied by task 15's stale 16)
            pos = gt if order is None else order.index(gt)
            tv = ctx.round_base + ctx.round_stride + pos + 1

            def _rel(tv=tv, gt=gt):
                time.sleep(float(os.environ.get("DBG_GDELAY", "0.0004")))
                with torch.cuda.stream(rel_s):
                    memcpy_async(rel_va + 4 * gt,
                                 ctx.tickets.data_ptr() + 4 * tv,
                                 4, rel_s.cuda_stream)
            threading.Thread(target=_rel).start()
            gates = [(gt, rel_va + 4 * gt, tv)]
        obth = None
        if do_obs and rank == 0:
            q0 = ctx.partners[0]
            fav = [ctx.arrival_ptr + (q0 * ctx.stage_slots + s) * 4
                   for s in range(ctx.stage_slots)]
            pbuf = torch.empty(ctx.stage_slots, dtype=torch.int32,
                               pin_memory=True)
            pstm = torch.cuda.Stream()

            def _obs(fav=fav, pbuf=pbuf, pstm=pstm):
                # same read pattern as the rig observer: async 4B D2H
                # from the symmetric heap on an independent stream
                t0o = time.perf_counter()
                n = 0
                while time.perf_counter() - t0o < 5.0:
                    for s, va in enumerate(fav):
                        hip.hipMemcpyAsync(
                            ctypes.c_void_p(int(pbuf.data_ptr()) + 4 * s),
                            ctypes.c_void_p(va), 4, 2,
                            ctypes.c_void_p(pstm.cuda_stream))
                    hip.hipStreamSynchronize(
                        ctypes.c_void_p(pstm.cuda_stream))
                    n += 1
                    time.sleep(0.00003)
            obth = threading.Thread(target=_obs)
            obth.start()
        if REUSE_OUT:
            if not SHARED_OUT:
                SHARED_OUT.append(
                    torch.empty(M, N, dtype=DTYPE, device="cuda"))
            out = SHARED_OUT[0]
        else:
            out = torch.empty(M, N, dtype=DTYPE, device="cuda")
        # position-keyed gate ticket (see comment above)
        ap.gemm_ar_pred_op(A[rank], B[rank], out, ctx, order=order,
                           gates=gates)
        if obth is not None:
            torch.cuda.synchronize()
            obth.join()
    else:
        out = torch.empty(M, N, dtype=DTYPE, device="cuda")
        rd.gemm_ar_rd_op(A[rank], B[rank], out, ctx)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1e3

    with torch.no_grad():
        ref = A[0].float() @ B[0].float().t()
        for s in range(1, size):
            ref += A[s].float() @ B[s].float().t()
        if OP == "rs":
            ref = ref[rank * m_pr:(rank + 1) * m_pr]
        d = (out.float() - ref).abs()
        md = d.max().item()
        wrong = d > 1e-2
        n_wrong = int(wrong.sum())
    worst = max(worst, md)
    if n_wrong == 0:
        if it % 50 == 0:
            print(f"[rank {rank}] it={it} ok md={md:.5f} ms={dt_ms:.2f}",
                  flush=True)
        _prev_A = A
        continue

    fails += 1
    print(f"[rank {rank}] FAIL it={it} md={md:.5f} n_wrong={n_wrong} "
          f"ms={dt_ms:.2f}", flush=True)
    if fails_printed < PRINT_FAILS:
        fails_printed += 1
        with torch.no_grad():
            # ---- damage shape: contiguous row runs + bad-col counts ----
            bad_rows = torch.nonzero(wrong.any(dim=1)).flatten()
            rowcnt = wrong.sum(dim=1)
            edge = bad_rows[1:] - bad_rows[:-1]
            brks = int((edge != 1).sum()) + 1
            runlen = bad_rows[-1] - bad_rows[0] + 1
            print(f"[rank {rank}]   shape: n_bad_rows={int(bad_rows.numel())}"
                  f" runs={brks} span={int(runlen)} "
                  f"first={int(bad_rows[0])} "
                  f"cols/row med={int(rowcnt[bad_rows].median())} "
                  f"e_std={float((out.float() - ref)[bad_rows].std()):.4f}",
                  flush=True)
            # ---- implied delta-A per source (sparse k-slice vs dense) --
            e = out.float() - ref
            for s in range(size):
                dA = (e @ PINV[s])                       # [rows, K]
                mag = dA.abs().max(dim=1).values
                dmg = torch.nonzero(mag > 0.004).flatten()
                if dmg.numel() == 0:
                    print(f"[rank {rank}]   dA(src{s}): none>0.004",
                          flush=True)
                    continue
                sups = []
                for i in dmg[:4].tolist():
                    m1 = dA[i].abs()
                    thr = 0.25 * float(m1.max())
                    ks = torch.nonzero(m1 > thr).flatten()
                    sups.append((int(ks[0]), int(ks[-1]), int(ks.numel())))
                med_supp = int(torch.tensor([x[2] for x in sups]).median())
                # implied wrong A content of the first damaged row vs
                # candidates: stale prev-round A, k-shifted A, row-shifted A
                i0 = int(dmg[0])
                m1 = dA[i0].abs()
                thr = 0.25 * float(m1.max())
                ks = torch.nonzero(m1 > thr).flatten()
                k0, k1 = int(ks[0]), int(ks[-1]) + 1
                grow = i0 + (rank * m_pr if OP == "rs" else 0)
                implied = (A[s].float()[grow, k0:k1] + dA[i0, k0:k1])
                hits = []
                if _prev_A is not None:
                    cv = _prev_A[s].float()[grow, k0:k1]
                    nh = int(((cv - implied).abs() < 1e-3).sum())
                    if nh:
                        hits.append(f"stale:{nh}/{cv.numel()}")
                for sh in (64, 128):
                    lo, hi = k0 + sh, k1 + sh
                    if hi <= K:
                        cv = A[s].float()[grow, lo:hi]
                        nh = int(((cv - implied).abs() < 1e-3).sum())
                        if nh:
                            hits.append(f"k+{sh}:{nh}/{cv.numel()}")
                    if k0 - sh >= 0:
                        cv = A[s].float()[grow, k0 - sh:k1 - sh]
                        nh = int(((cv - implied).abs() < 1e-3).sum())
                        if nh:
                            hits.append(f"k-{sh}:{nh}/{cv.numel()}")
                for rsh in (128, 256):
                    lo = grow + rsh
                    if lo < M:
                        cv = A[s].float()[lo, k0:k1]
                        nh = int(((cv - implied).abs() < 1e-3).sum())
                        if nh:
                            hits.append(f"r+{rsh}:{nh}/{cv.numel()}")
                    if grow - rsh >= 0:
                        cv = A[s].float()[grow - rsh, k0:k1]
                        nh = int(((cv - implied).abs() < 1e-3).sum())
                        if nh:
                            hits.append(f"r-{rsh}:{nh}/{cv.numel()}")
                print(f"[rank {rank}]   dA(src{s}): dmg_rows={int(dmg.numel())}"
                      f" med_supp={med_supp} supp(k0,k1,n)={sups}"
                      f" align64={all(x[0] % 64 == 0 for x in sups)}"
                      f" hits={hits or 'NONE'}", flush=True)
    _prev_A = A

if OP == "rs":
    rs5.finalize_gemm_rs_v5_context(ctx)
else:
    rd.finalize_gemm_ar_rd_context(ctx)

print(f"[rank {rank}] SOAK RESULT op={OP} fails={fails}/{ROUNDS} "
      f"worst={worst:.5f}", flush=True)
dcu_state()["host"].finalize()
barrier()
mpi.MPI_Finalize()
os._exit(0 if fails == 0 else 1)
