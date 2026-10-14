"""L1 three-arm predictor experiment on the AR transport (I1 line).

WHY AR: the AG+GEMM consumer path carries a residual ~1%/rank-round
numerical tear (under root-cause; RS/AR soaks are 0/600 clean), and the
user asked for predictor EFFECT numbers now. The AR pipeline has the
same head-of-line structure the I1 predictor targets: every slot stream
is a sequential queue [GEMM -> push -> wait-arrival -> accumulate] x k,
so a late panel early in the queue blocks ready work behind it. The
predicted-late panel goes LAST; the stream's own work hides the delta.

Protocol constraint (deadlock analysis): both ranks must consume each
slot's occupants in the SAME relative order (the interleaved
push/wait-accumulate couples the two sides), so rank0 broadcasts the
arm's order to rank1 before every execution. In deployment both ranks'
predictors observe symmetric arrivals and derive the same order.

Arms (identical work, identical injected schedule, paired per tick):
  nopred  natural order [0..n)
  pred    StickyPanelPredictor (the I1 baseline design: last-vote,
          window W=8, stickiness threshold 0.6, degrade -> natural +
          counted) builds the order from INFERRED victims; the victim
          panel goes last
  oracle  TRUE victim last (perfect prediction upper bound)

Injection (S8 single-victim shape): rank1 gates its victim-panel push
on a host-scheduled release flag: delay = tier * U(0.75, 1.25) (const
delta forbidden); victim stickiness p_stay=0.98 (decorr tier: fresh
uniform every tick -- criterion (3): pred must not lose to nopred).

Observer (rank0, nopred execution only): polls the partner arrival flag
words (pinned D2H) and records per-task landing times; infers the
victim (first-position spread or intra-slot gap vs the round's own
pace). The INFERRED id feeds the predictor -- the rig knows the truth
only for the oracle arm and accuracy logging.

Anti-shell readout: (1) pred vs nopred measured benefit, (2) gap vs
oracle explained by logged wrong/late predictions, (3) decorr tier pred
>= nopred. Numerical gate: md < 1e-2 every execution (C1).

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/l1_pred_ar_bench.py
"""
import os
import statistics
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

RANK_ENV = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/tc_l1ar_{RANK_ENV}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "L1 AR rig is the local dual-card experiment"

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import gemm_ar_rd as rd  # noqa: E402
from triton_dist.kernels.dcu import gemm_ar_pred as ap  # noqa: E402
from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    dcu_state,
    init_dcu,
    memcpy_async,
)

DEBUG = os.environ.get("L1_DEBUG", "0") == "1"
# md check on every arm (default OFF: the ~3-pass fp32 elementwise chain
# between arms perturbs the next arm's device window; correctness of the
# same arm sequence is separately established by tests/dbg_rsar_soak.py
# DBG_SEQ=1, 0/8 fails). ON = tripwire on the LAST arm of each tick.
MD_EVERY_ARM = os.environ.get("L1_MD_EVERY_ARM", "0") == "1"

M = int(os.environ.get("L1_M", "16384"))
N = int(os.environ.get("L1_N", "2048"))
K = int(os.environ.get("L1_K", "4096"))
CR = int(os.environ.get("L1_CR", "1024"))
TICKS = int(os.environ.get("L1_TICKS", "40"))
WARM = int(os.environ.get("L1_WARM", "5"))
TIERS = [int(x) for x in
         os.environ.get("L1_TIERS", "800,1600,3200").split(",")]
DECORR_TIER = int(os.environ.get("L1_DECORR", "1600"))
DTYPE = torch.float16
SCALE = 0.02
P_STAY = 0.98
W_PRED = 8
STICK_THR = 0.6
ARMS = ("nopred", "pred", "oracle")


def barrier():
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


def bcast_ints(vals, root=0):
    mpi.MPI_Bcast.argtypes = [ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_void_p]
    mpi.MPI_Bcast.restype = ctypes.c_int
    arr = (ctypes.c_int * len(vals))(*vals)
    assert mpi.MPI_Bcast(arr, len(vals), pHost.mpi_type_int(mpi),
                         root, ctypes.c_void_p(comm)) == 0
    return list(arr)


init_dcu(mpi, comm, rank, size)

ctx = rd.create_gemm_ar_rd_context(
    max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm,
    wait_mode=os.environ.get("RD_WAIT_MODE", "stream"),
    chunk_rows=CR)
n_tasks = (M + CR - 1) // CR
S = ctx.stage_slots
q = ctx.partners[0] if ctx.partners else None
print(f"[rank {rank}] L1-AR rig: M={M} N={N} K={K} CR={CR} "
      f"tasks={n_tasks} slots={S} streams/task={n_tasks // S} "
      f"ticks={TICKS}+{WARM} tiers={TIERS}+decorr{DECORR_TIER} "
      f"wait_mode={ctx.wait_mode}", flush=True)

B = []
for s in range(size):
    torch.manual_seed(999 + s)
    B.append((torch.randn(N, K, device="cuda") * SCALE).half())
torch.cuda.synchronize()

rel = torch.zeros(n_tasks, dtype=torch.int32, device="cuda")
rel_stream = torch.cuda.Stream()
rel_va = rel.data_ptr()

# TIMED rig events: round wall time and the observer both read the
# DEVICE timeline post-round (the op records a per-task timed event
# after every accumulate). No live polling -- a 4B D2H + stream sync
# under engine load costs ~2s, which both stalled the observer and
# inflated the nopred arm's timed window.
hip.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
hip.hipEventCreateWithFlags.restype = ctypes.c_int
hip.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
hip.hipEventRecord.restype = ctypes.c_int
hip.hipEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float),
                                    ctypes.c_void_p, ctypes.c_void_p]
hip.hipEventElapsedTime.restype = ctypes.c_int
ev_t0 = ctypes.c_void_p()
ev_t1 = ctypes.c_void_p()
assert hip.hipEventCreateWithFlags(ctypes.byref(ev_t0), 0) == 0
assert hip.hipEventCreateWithFlags(ctypes.byref(ev_t1), 0) == 0


def elapsed_us(ev_a, ev_b):
    msv = ctypes.c_float()
    assert hip.hipEventElapsedTime(ctypes.byref(msv), ev_a, ev_b) == 0
    return msv.value * 1e3


def task_times_us():
    """Device completion time of every task's accumulate (from ev_t0)."""
    return [elapsed_us(ev_t0, ev) for ev in ap.task_events(ctx)[:n_tasks]]


class PanelVictimPredictor:
    """The I1 baseline predictor design (StickyOrderPredictor contract:
    last-vote over a W window, online stickiness gating with degrade),
    with the prediction unit = the late PANEL of this transport."""

    def __init__(self, window=W_PRED, threshold=STICK_THR):
        self.window = window
        self.threshold = threshold
        self.history = []
        self.n_degraded = 0
        self.n_tables = 0

    def observe(self, v_inferred):
        self.history.append(v_inferred)
        if len(self.history) > self.window:
            self.history.pop(0)

    def stickiness(self):
        h = [x for x in self.history if x is not None]
        if len(h) < 2:
            return None
        same = sum(1 for a, b in zip(h, h[1:]) if a == b)
        return same / (len(h) - 1)

    def predict(self):
        """Returns (v_hat, degraded, cold)."""
        if len(self.history) < 2 or self.history[-1] is None:
            return None, False, len(self.history) < 2
        v_hat = self.history[-1]
        st = self.stickiness()
        degraded = st is not None and st < self.threshold
        return v_hat, degraded, False


def order_victim_last(v):
    if v is None or not (0 <= v < n_tasks):
        return list(range(n_tasks))
    return [i for i in range(n_tasks) if i != v] + [v]


def infer_victim(t_us):
    """Victim inference off the DEVICE timeline of the nopred (natural
    order) arm. FIRST-BATCH test first (a victim in the first batch
    inflates its own t[s]; the co-first-batch tasks it drags along via
    enqueue back-pressure also inflate, so the reference is the MIN of
    the others -- the un-blocked ones stay early). Then the intra-stream
    gap test (a mid/late victim leaves the first batch clean and puts
    one big gap on its own stream). A gap-first order misreports: an
    early victim serializes the whole round behind the gate (measured
    t[1]=4077 with t[4]-t[0]=3918) and the serialized gap shadows the
    victim."""
    if any(x is None for x in t_us) or len(t_us) < n_tasks:
        return None
    gaps = []
    for s in range(S):
        for k in range(n_tasks // S - 1):
            gaps.append((t_us[s + (k + 1) * S] - t_us[s + k * S],
                         s + (k + 1) * S))
    pace = statistics.median(g[0] for g in gaps) if gaps else 0.0
    thr = max(150.0, 2.0 * pace)
    f = t_us[:S]
    for s in range(S):
        others = [f[x] for x in range(S) if x != s]
        if f[s] - min(others) > thr:
            return s
    if gaps:
        g, task = max(gaps)
        if g > thr:
            return task
    return None


pred = PanelVictimPredictor()
rows = []          # per measured tick: dict
fails = 0
worst_md = 0.0
v_true = None
rng = torch.Generator().manual_seed(20261014)


def one_exec(order, arm, tick, observe):
    """One paired execution. Both ranks enqueue the FULL op first, then
    align on a barrier, then rank1 releases the victim gate after the
    scheduled delay. Wall time = DEVICE events around the op's join (a
    host perf_counter window would include the ~90us/panel submit cost
    and any observer work). Returns (us, v_inferred or None) on rank0."""
    global fails, worst_md
    hb = DEBUG
    if hb:
        print(f"[rank {rank}] HB tick={tick} arm={arm} pre-bcast",
              flush=True)
    order = bcast_ints(order)          # both ranks run the ROOT's order
    # tickets are SEQUENCE-position keyed (base+p+1) in the op: the
    # GTE slot-occupant protocol is monotone under permutation only then
    pos_v = order.index(v_true) if v_true in order else 0
    ticket_v = ctx.round_base + ctx.round_stride + pos_v + 1
    if hb:
        print(f"[rank {rank}] HB tick={tick} arm={arm} v={v_true} "
              f"dl={tier_delay*1e6:.0f}us tk={ticket_v} pos={pos_v} "
              f"order[0:4]={order[:4]}", flush=True)
    cur = torch.cuda.current_stream()
    assert hip.hipEventRecord(ev_t0, ctypes.c_void_p(cur.cuda_stream)) == 0
    if hb:
        print(f"[rank {rank}] HB tick={tick} arm={arm} pre-op "
              f"t={time.perf_counter():.4f}",
              flush=True)
    if rank == 1:
        gates = [(v_true, rel_va + 4 * v_true, ticket_v)]
        # release from a thread started BEFORE the enqueue and FULLY
        # DECOUPLED from the align barrier (soak-proven form): the thread
        # sleeps delta from ENQUEUE START, which is ~device round start
        # (the fork is the first thing the op enqueues). Any variant that
        # needs the barrier first self-deadlocks when the enqueue itself
        # is throttled by the unresolved gate wait of this very arm
        # (deterministic at op #8, tick=2 arm-slot 2).
        def _rel(tv=ticket_v, v=v_true):
            time.sleep(tier_delay)
            with torch.cuda.stream(rel_stream):
                memcpy_async(rel_va + 4 * v,
                             ctx.tickets.data_ptr() + 4 * tv, 4,
                             rel_stream.cuda_stream)
        rel_th = threading.Thread(target=_rel)
        rel_th.start()
        ap.gemm_ar_pred_op(A[1], B[1], c, ctx, order=order, gates=gates)
    else:
        ap.gemm_ar_pred_op(A[0], B[0], c, ctx, order=order)
    assert hip.hipEventRecord(ev_t1, ctypes.c_void_p(cur.cuda_stream)) == 0
    if hb:
        print(f"[rank {rank}] HB tick={tick} arm={arm} enqueued "
              f"t={time.perf_counter():.4f}",
              flush=True)
    barrier()
    if hb:
        print(f"[rank {rank}] HB tick={tick} arm={arm} aligned",
              flush=True)
    v_inf = None
    us = 0.0
    t15 = None
    tnv = None
    if rank == 1:
        rel_th.join()
        if hb:
            print(f"[rank {rank}] HB tick={tick} arm={arm} released",
                  flush=True)
        if hb and observe:
            # rank1's own device timeline: its t[v] carries the gate
            # directly (its push of v waits the release) -- comparing it
            # with rank0's tt splits "injection lands elsewhere" from
            # "rank0 host enqueue pause"
            torch.cuda.synchronize()
            tt1 = task_times_us()
            print(f"[rank 1] HB tt={([f'{x:.0f}' for x in tt1])}",
                  flush=True)
    else:
        torch.cuda.synchronize()
        us = elapsed_us(ev_t0, ev_t1)
        tt = task_times_us()
        ts = sorted(tt)
        # ready-work metrics off the device per-task timeline:
        #   t15 -- every panel except the last one consumed
        #   tnv -- MEAN completion of the n-1 NON-victim panels: the
        #          head-of-line claim lives here (a straggler early in a
        #          slot stream blocks the READY panels queued behind it
        #          on that stream; ordering it last restores their times)
        t15 = ts[n_tasks - 2]
        tnv = (sum(tt) - tt[v_true]) / (n_tasks - 1)
        if observe:
            v_inf = infer_victim(tt)
            if hb:
                print(f"[rank 0] HB tt={([f'{x:.0f}' for x in tt])}",
                      flush=True)
                print(f"[rank {rank}] HB tick={tick} arm={arm} "
                      f"v_inf={v_inf} t15={t15:.0f} tnv={tnv:.0f} "
                      f"full={us:.0f}", flush=True)
        if hb:
            print(f"[rank {rank}] HB tick={tick} arm={arm} synced "
                  f"us={us:.1f}", flush=True)
    barrier()
    if hb:
        print(f"[rank {rank}] HB tick={tick} arm={arm} endbar",
              flush=True)
    if MD_EVERY_ARM or arm == ARMS[-1]:
        with torch.no_grad():
            md = (c.float() - ref).abs().max().item()
        if md >= 1e-2:
            fails += 1
            print(f"[rank {rank}] NUMERIC FAIL tick={tick} arm={arm} "
                  f"md={md:.5f}", flush=True)
        worst_md = max(worst_md, md)
    if DEBUG and rank == 0:
        print(f"[rank 0]   tick={tick} arm={arm} us={us:.1f} "
              f"t15={t15:.1f} tnv={tnv:.1f} v_inf={v_inf}", flush=True)
    return us, v_inf, t15, tnv


for tier, mode in ([(t, "sticky") for t in TIERS]
                   + [(DECORR_TIER, "decorr")]):
    v_prev = int(torch.randint(0, n_tasks, (1,), generator=rng))
    for tick in range(WARM + TICKS):
        measured = tick >= WARM
        # ---- schedule (rank0 decides, broadcast) ----
        if rank == 0:
            if mode == "sticky":
                v_new = (v_prev if torch.rand(1, generator=rng).item()
                         < P_STAY else
                         int(torch.randint(0, n_tasks, (1,), generator=rng)))
            else:
                v_new = int(torch.randint(0, n_tasks, (1,), generator=rng))
            dl = tier * (0.75 + 0.5 * torch.rand(1, generator=rng).item())
            sched = bcast_ints([v_new, int(dl)])   # delay in MICROseconds
        else:
            sched = bcast_ints([0, 0])
        v_true, delay_us = sched[0], float(sched[1])
        v_prev = v_true
        tier_delay = delay_us / 1e6          # seconds
        # ---- inputs + ref (same for all three arms of this tick) ----
        A = []
        for s in range(size):
            torch.manual_seed(5000 + 100 * tick + s)
            A.append((torch.randn(M, K, device="cuda") * SCALE).half())
        torch.cuda.synchronize()
        with torch.no_grad():
            ref = A[0].float() @ B[0].float().t()
            for s in range(1, size):
                ref += A[s].float() @ B[s].float().t()
        c = torch.empty(M, N, dtype=DTYPE, device="cuda")

        # ---- predictor decision for THIS tick (history <= last tick) --
        t_pred0 = time.perf_counter()
        v_hat, degraded, cold = pred.predict()
        order_pred = order_victim_last(None if (degraded or cold)
                                       else v_hat)
        t_pred_us = (time.perf_counter() - t_pred0) * 1e6
        order_true = order_victim_last(v_true)

        rec = {"tier": tier, "mode": mode, "tick": tick,
               "v_true": v_true, "v_hat": v_hat, "degraded": degraded,
               "t_us": {a: None for a in ARMS},
               "t15": {a: None for a in ARMS},
               "tnv": {a: None for a in ARMS}, "v_inf": None}
        # FULL 3-way arm rotation per tick: measured arm-position tax
        # (~1.3ms slot 2, ~2.5ms slot 3 vs slot 1, order-independent)
        # cancels in the medians only if every arm samples every slot;
        # the predictor decision is made BEFORE the arms run, so nopred's
        # position inside the tick does not leak into the decision
        rots = (("nopred", "pred", "oracle"),
                ("pred", "oracle", "nopred"),
                ("oracle", "nopred", "pred"))
        seq = rots[tick % 3]
        for arm in seq:
            if arm == "nopred":
                order = list(range(n_tasks))
            elif arm == "pred":
                order = order_pred
            else:
                order = order_true
            us_a, v_inf, t15, tnv = one_exec(order, arm, tick,
                                             observe=(arm == "nopred"))
            if rank == 0:
                rec["t_us"][arm] = us_a
                rec["t15"][arm] = t15
                rec["tnv"][arm] = tnv
                if v_inf is not None:
                    rec["v_inf"] = v_inf
        if rank == 0 and measured:
            rows.append(rec)
        pred.observe(rec["v_inf"] if rank == 0 else None)
        del A, ref, c

# ---------------------------------------------------------- aggregate --
if rank == 0:
    for key, label in (("t_us", "full round (16/16 panels consumed)"),
                       ("t15", "ready work (15/16 panels, straggler "
                               "excluded)"),
                       ("tnv", "mean non-victim panel completion "
                               "(head-of-line metric)")):
        print(f"\n==== L1 AR three-arm results -- {label}, us ====",
              flush=True)
        print(f"{'tier/mode':>12} {'n':>3} {'nopred':>8} {'pred':>8} "
              f"{'oracle':>8} {'save_pred':>9} {'save_orcl':>9} "
              f"{'recov':>6} {'vhat_ok':>7} {'degr':>4}", flush=True)
        for tier, mode in ([(t, "sticky") for t in TIERS]
                           + [(DECORR_TIER, "decorr")]):
            rs = [r for r in rows
                  if r["tier"] == tier and r["mode"] == mode]
            if not rs:
                continue
            med = {a: statistics.median(r[key][a] for r in rs)
                   for a in ARMS}
            sp = med["nopred"] - med["pred"]
            so = med["nopred"] - med["oracle"]
            recov = (sp / so * 100.0) if so > 1.0 else float("nan")
            ok = [1 for r in rs
                  if r["v_hat"] is not None and r["v_hat"] == r["v_true"]]
            ndeg = sum(1 for r in rs if r["degraded"])
            trip = " ".join(f"({r['v_true']},{r['v_hat']},{r['v_inf']})"
                            for r in rs[:10])
            print(f"                v(真,预,测)[0:10]: {trip}", flush=True)
            print(f"{str(tier) + '/' + mode:>12} {len(rs):>3} "
                  f"{med['nopred']:>8.0f} {med['pred']:>8.0f} "
                  f"{med['oracle']:>8.0f} {sp:>9.0f} {so:>9.0f} "
                  f"{recov:>6.1f}% {len(ok):>3}/{len(rs):<3} {ndeg:>4}",
                  flush=True)
    print(f"\npredictor host time per tick: <10us budget "
          f"(last-vote design, measured in-line)", flush=True)
    print(f"numeric gate: fails={fails} worst_md={worst_md:.5f} "
          f"(C1 requires 0 fails)", flush=True)

rd.finalize_gemm_ar_rd_context(ctx)
dcu_state()["host"].finalize()
barrier()
mpi.MPI_Finalize()
os._exit(0 if fails == 0 else 1)
