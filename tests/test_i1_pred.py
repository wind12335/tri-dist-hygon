"""2-card DCU test: I1 predictor closed loop (ag_tr_pred + i1_predictor).

Verifies, in order:
1. host predictor unit behavior (cold start, sticky prediction, decorr
   auto-degradation) -- pure python, no GPU;
2. the table plane (writer kernel words via the peek kernel: seq/degraded/
   mode/k_eff/order/budget);
3. mode-0 and mode-1 consumers both produce the reference C (mode only
   permutes the tile->CTA mapping, C must be identical);
4. the full loop under straggler injection: observer reports the peer
   late exactly on the rounds the peer is the injected straggler; the
   predictor emits mode=1 on the non-straggler rank in the sticky phase
   and degrades in the alternating phase.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_i1_pred.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_i1_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "L1 loop is a 2-card mechanism test"

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu.i1_predictor import StickyOrderPredictor  # noqa: E402

ok = True


def report(name, passed, detail=""):
    global ok
    if not passed:
        ok = False
    print(f"[rank {rank}] {name}: {'PASS' if passed else 'FAIL'} {detail}",
          flush=True)


# ---------------------------------------------------------------- 1. predictor unit
def predictor_unit():
    good = True
    # cold start: no mode-1 before two observations
    p = StickyOrderPredictor(rank=0, num_ranks=2, window=4)
    t = p.table(k_eff=2)
    good &= (t["mode"] == 0 and t["cold"] and t["order"] == [0, 1])
    # sticky chain: peer (1) is the straggler round after round
    for _ in range(4):
        p.observe(1)
    t = p.table(k_eff=2)
    good &= (t["mode"] == 1 and t["order"] == [0, 1]
             and t["degraded"] == 0 and t["s_hat"] == 1)
    # own-rank predicted straggler: my consumer sees no skew -> mode 0
    q = StickyOrderPredictor(rank=1, num_ranks=2, window=4)
    for _ in range(4):
        q.observe(1)  # rank 1 itself
    t = q.table(k_eff=2)
    good &= (t["mode"] == 0 and t["s_hat"] == 1)
    # decorrelated chain: stickiness 1/3 < 0.6 -> degrade to mode 0
    r = StickyOrderPredictor(rank=0, num_ranks=2, window=4)
    for s in (1, 0, 1, 0):
        r.observe(s)
    t = r.table(k_eff=2)
    good &= (t["mode"] == 0 and t["degraded"] == 1
             and t["stickiness"] is not None and t["stickiness"] < 0.6)
    # order packing round trip
    good &= (t["order_packed"] == sum(o << (4 * i)
                                      for i, o in enumerate(t["order"])))
    return good


report("predictor unit", predictor_unit())

# ---------------------------------------------------------------- GPU setup
M_per_rank = 2048
K = 1024
N = 2048
M = M_per_rank * size
DTYPE = torch.float16
SCALE = 0.02
# 2 ms: with the align barrier moved to just-before-op, even 400us bites
# (mode-0 band-major hits the peer's chunk-0 band after ~1/(2*n_chunks)
# of the round ~= 0.15ms here), but the absolute margin over the round-ms
# jitter at this SMALL shape is thin (victim ~+0.25ms on a 0.6ms round).
# A tail-scale delta (p99-class, ms) keeps the inference check robust;
# the bench's big shape + {200..800}us grid exercises the marginal
# regime where the mechanism actually has to work.
DELTA_US = 2000.0

b = (torch.randn(N, K, device="cuda") * SCALE).half()
ictx = i1.create_i1_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)
ag = ictx.ag
print(f"[rank {rank}] ctx: chunk_rows={ag.chunk_rows} "
      f"num_chunks={ag.num_chunks} lanes={ag.comm_lanes} "
      f"wait_mode={ag.wait_mode} table_words={ictx.table_words}",
      flush=True)
assert ag.num_chunks >= 2, "the wave-order mechanism needs >= 2 chunks"

ev_round = torch.cuda.Event()
ev_t0 = torch.cuda.Event(enable_timing=True)
ev_t1 = torch.cuda.Event(enable_timing=True)


def make_round(it, m_pr):
    A_it = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A_it.append((torch.randn(m_pr, K, device="cuda") * SCALE).half())
    a = A_it[rank]
    with torch.no_grad():
        ref = torch.cat(A_it, dim=0).float() @ b.float().t()
    return a, ref


def peek_table():
    dst = torch.zeros(ictx.table_words, dtype=torch.int32, device="cuda")
    i1.kernel_i1_peek[(1,)](dst.data_ptr(), ictx.ctl_ptr,
                            NWORDS=ictx.table_words)
    return dst.cpu().tolist()


# ---------------------------------------------------------------- 2. table plane
rot_packed = sum(((rank + i) % size) << (4 * i) for i in range(size))
i1.i1_table_write(ictx, 7, 0, 1, rot_packed, ag.num_chunks)
torch.cuda.synchronize()
words = peek_table()
expect = [7, 0, 1, ag.num_chunks] + [rank, 1 - rank] + [128, 128]
report("table words", words == expect, f"got={words} expect={expect}")

mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int


def align():
    """Rig-only per-round host alignment (the OP itself stays barrier-
    free). Without it the two host loops drift apart and the non-straggler
    starts its round after the peer already staged -- the injected lateness
    evaporates and the observer counts collapse to 1 (run-2 fact)."""
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


def run_round(it, table, straggler, tag):
    """One full loop round: make_round (UNALIGNED -- its randn+ref-GEMM
    jitter is ~0.3ms and would absorb a 400us delta; the bench learned
    this the hard way) -> align -> table write -> [straggler HOST-sleeps
    delta] -> op -> round sync -> C check + observer read. With the
    barrier right before the op, the op-start skew == delta exactly.
    Injection is host-side (the bisect-validated form: device _sleep
    corrupts/wedges, and it must sit OUTSIDE the timed region so T
    measures the victim's wait, not the straggler's own sleep)."""
    a, ref = make_round(it, M_per_rank)
    align()
    if straggler is not None and rank == straggler:
        time.sleep(DELTA_US / 1e6)
    ev_t0.record()
    i1.i1_table_write(ictx, table["seq"], table["degraded"],
                      table["mode"], table["order_packed"],
                      table["k_eff"])
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")
    i1.i1_ag_tr_op(a, b, c, ictx)
    ev_t1.record()
    ev_round.record()
    ev_round.synchronize()
    md = (c.float() - ref).abs().max().item()
    obs = i1.i1_observer_read(ictx)
    ms = ev_t0.elapsed_time(ev_t1)
    report(f"{tag} iter {it} correct", md < 1e-2, f"max_diff={md:.5f}")
    return obs, ms


def peer_late(obs, peer, thresh):
    """Observer inference: True = peer's chunks arrived late this round.
    Marker < 0 (probe cap) -> None (no observation this round)."""
    if obs[0] < 0:
        return None
    return obs[3 + peer] > thresh


# ---------------------------------------------------------------- 3. modes
cal = i1.calibrate_sleep_cycles_per_us()
ictx.cycles_per_us = cal
print(f"[rank {rank}] cycles_per_us={cal:.1f}", flush=True)

base_mode0 = {"seq": 10, "degraded": 0, "mode": 0, "order_packed": rot_packed,
              "k_eff": ag.num_chunks}
# 2 warmup rounds are DISCARDED: round 0 walks every first-call compile path
# (consumer variants, flag kernels) -- the observer's probe counts during
# those are compile-time, not lateness (run-2: clean count 416756)
for it in range(2):
    run_round(it, base_mode0, None, "warmup")
# calibration rounds (compiles all done, aligned starts, no skew): these
# give BOTH the observer's clean count distribution AND the timing
# baseline for the 2-card lateness inference.
obs_counts_clean = []
calib_ms = []
for it in range(3):
    obs, ms = run_round(20 + it, base_mode0, None, "calib")
    obs_counts_clean.append(obs[3 + (1 - rank)])  # peer count slot
    calib_ms.append(ms)
BASE_MS = sorted(calib_ms)[1]
mode1_packed = rank | ((1 - rank) << 4)
base_mode1 = {"seq": 40, "degraded": 0, "mode": 1,
              "order_packed": mode1_packed, "k_eff": ag.num_chunks}
for it in range(2):
    run_round(100 + it, base_mode1, None, "mode1")

# ---------------------------------------------------------------- 4. injected loop
# 2-card lateness inference (2026-10-14 rig fact): the observer's probe
# COUNT signal collapsed under the all-kernel gated path (scheduling-
# mediated, victim ~74 vs own-view ~44 at delta=400) -- the robust
# binary signal is THIS rank's own round time: slow round == I waited
# for the peer == peer is the straggler; fast round == I am the
# straggler (or no skew). Separation measured 10x at delta=400us.
T_MARGIN_MS = 0.25
_base_ms = [1e9]


def fresh_base():
    """Two steady-state no-injection rounds; the min is this rank's
    no-skew round time. Early rounds are ~2x slow (JIT/cache residue),
    so the baseline must be taken right before each phase."""
    ms = []
    for j in range(2):
        _, m = run_round(240 + j, base_mode0, None, "base")
        ms.append(m)
    _base_ms[0] = min(ms)
    return _base_ms[0]


def infer_straggler(ms):
    """None = no usable signal; else the inferred straggler rank id."""
    return (1 - rank) if ms > _base_ms[0] * 1.6 + 0.1 else rank


THRESH = max(obs_counts_clean) * 4 + 32  # legacy count signal (recorded)
print(f"[rank {rank}] obs peer counts (no inj)={obs_counts_clean} "
      f"THRESH={THRESH} BASE_MS={BASE_MS:.3f}", flush=True)

# phase A: GLOBAL sticky straggler = rank 1 (the bench's Markov semantics:
# one straggler identity per round for the whole job -- run-2's `1 - rank`
# made both ranks delay themselves, i.e. no relative skew at all)
pred = StickyOrderPredictor(rank, size, window=4, stickiness_threshold=0.6)
sticky_straggler = 1
fresh_base()
mode_flags = []
for it in range(4):
    t = pred.table(k_eff=ag.num_chunks)
    mode_flags.append(t["mode"])
    obs, ms = run_round(200 + it, t, sticky_straggler, "stickyA")
    s_inf = infer_straggler(ms)
    pred.observe(s_inf)
    # BOTH ranks must infer the GLOBAL straggler identity: the victim
    # waits ~delta (slow round -> peer), the straggler itself runs a fast
    # round -> itself
    report(f"stickyA iter {it} inference", s_inf == sticky_straggler,
           f"ms={ms:.3f} count={obs[3 + (1 - rank)]} "
           f"round_marker={obs[0]}")
report("stickyA predictor mode1 reached",
       (mode_flags[-1] == 1) == (rank != sticky_straggler),
       f"modes={mode_flags}")

# phase B: alternating GLOBAL straggler -> stickiness 1/3 < 0.6 -> degraded
deg_seen = False
hit = 0
tot = 0
fresh_base()
for it in range(6):
    t = pred.table(k_eff=ag.num_chunks)
    straggler = it % 2
    obs, ms = run_round(300 + it, t, straggler, "altern")
    s_inf = infer_straggler(ms)
    tot += 1
    hit += int(s_inf == straggler)
    pred.observe(s_inf)
    deg_seen |= bool(t["degraded"])
report("altern degraded engaged", deg_seen,
       f"last_stickiness={pred.stickiness()}")
report("altern inference sane", tot >= 4 and hit == tot,
       f"hit={hit}/{tot}")

i1.finalize_i1_context(ictx)

print(f"[rank {rank}] I1_PRED TEST {'PASS' if ok else 'FAIL'}", flush=True)
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402
dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0 if ok else 1)
