"""L1 three-arm injection bench: I1 predictor closed loop on 2 cards.

One mpirun = ONE (arm, tier, delta) cell; the aggregator
(bench/l1_pred_aggregate.py) combines the JSONs. Environment:

  L1_ARM     nopred | pred | oracle   (default pred)
  L1_TIER    sticky | decorr          (p_stay 0.98 / 0.5; default sticky)
  L1_DELTA_US  nominal delta in us: 0 | 200 | 400 | 800 | 1600 (default
             400). mode-0's band-major layout meets the peer's chunk-0
             band after ~1/(2*n_chunks) of the round (~0.3ms at the
             default shape) and mode-1's source-major layout defers it to
             ~half the round (~2ms): the active recovery window is
             delta in (0.3ms, 2ms), so 1600 is the strong cell.
             delta>0 is JITTERED per round (lognormal, median=nominal,
             sigma .5, clipped [.25x, 4x]) -- CONSTANT delta is
             forbidden by the pre-registered protocol.
  L1_ROUNDS  measurement rounds       (default 40)
  L1_SHAPE   M_per_rank x K x N       (default 8192x2048x2048 ->
             M=16384, n_chunks=8, 1024 tiles: strong wave structure)
  L1_SEED    straggler-chain seed (SHARED by all arms/tiers so the
             aggregator pairs rounds 1:1; default 1234)

Rig (per round): make_round (UNALIGNED -- its randn+ref-GEMM jitter would
absorb sub-ms deltas) -> MPI align barrier -> arm's table -> [perf_
counter: predictor+table-write host cost] -> straggler host-sleep(delta)
-> I1 op (observer ON in every arm: identical stream topology, arms
differ ONLY in table content) -> round sync -> C check -> own-round-time
vs forced-delta-0 baseline (1-round stale inference for the pred arm;
the observer count signal is scheduling-mediated under the all-kernel
gated path and is recorded but not used).

Injection is HOST-side by design (2026-10-14 finding): the device
_sleep path both corrupts (flag-vs-data visibility race under skew --
see docs/I1_PREDICTOR_L1_20261013.md) and wedges with engine flag
fan-out; host delay reproduces the measured host-origin delta (p50
534us) exactly where it happens in production.

PRE-REGISTERED PASS CRITERIA (written before any run; the aggregator
checks them):
  C1  correctness: every round, every arm, max|C-ref| < 1e-2
  C2  host overhead: p95(predictor observe+table+write) < 10 us
  C3  recovery @ sticky, nominal delta >= 400us:
      (T_nopred - T_pred)/(T_nopred - T_oracle) >= 0.5 on victim rounds
  C4  decorr tier not worse: recovery >= -0.05 AND degrade_rate >= 0.5
  C5  delta=0 no tax: |T_pred - T_nopred|/T_nopred <= 0.02 (median)

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/l1_pred_bench.py
"""
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_l1_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "L1 is a 2-card local rig (order[16] needs vcjob)"

ARM = os.environ.get("L1_ARM", "pred")
TIER = os.environ.get("L1_TIER", "sticky")
DELTA_US = float(os.environ.get("L1_DELTA_US", "400"))
ROUNDS = int(os.environ.get("L1_ROUNDS", "40"))
M_PR, K, N = (int(x) for x in
              os.environ.get("L1_SHAPE", "8192x2048x2048").split("x"))
SEED = int(os.environ.get("L1_SEED", "1234"))
P_STAY = 0.98 if TIER == "sticky" else 0.5

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int


def align():
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu.i1_predictor import StickyOrderPredictor  # noqa: E402

M = M_PR * size
DTYPE = torch.float16
SCALE = 0.02
WARMUP = 3
# stream-isolation knobs (probe4/probe5, 2026-10-14): the op hangs at
# chunks=2 + chunk>=4.5MB with kernel pulls while the pull kernel itself
# is proven clean at 8.4MB incl. a 300ms spin (tests/dbg_pull_size.py)
LANES = int(os.environ.get("L1_LANES", "2"))
NO_OBS = os.environ.get("L1_NO_OBS", "") == "1"

b = (torch.randn(N, K, device="cuda") * SCALE).half()
ictx = i1.create_i1_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm, comm_lanes=LANES)
ag = ictx.ag
assert ag.num_chunks >= 2, "wave-order mechanism needs >= 2 chunks"
print(f"[rank {rank}] L1 arm={ARM} tier={TIER} delta={DELTA_US}us "
      f"rounds={ROUNDS} shape={M_PR}x{K}x{N} chunks={ag.num_chunks} "
      f"tiles={(M // 128) * (N // 256)} lanes={ag.comm_lanes} "
      f"arrive/ready engine flags={i1.I1_ARRIVE_ENGINE}/{i1.I1_READY_ENGINE} "
      f"pull={'kernel' if i1.I1_PULL_KERNEL else 'engine'}",
      flush=True)


# ---- straggler chain + delta schedule (seeded, shared across ranks AND
# arms so the aggregator pairs rounds 1:1) --------------------------------
def chain(rounds):
    import random
    rng = random.Random(SEED * 1000 + (0 if TIER == "sticky" else 1))
    s = 1
    out = []
    for it in range(rounds):
        if it > 0 and rng.random() > P_STAY:
            s = 1 - s
        if DELTA_US <= 0:
            d = 0.0
        else:
            d = math.exp(rng.gauss(math.log(DELTA_US), 0.5))
            d = min(max(d, 0.25 * DELTA_US), 4.0 * DELTA_US)
        out.append((s, d))
    return out


CH = chain(WARMUP + ROUNDS)

ev_round = torch.cuda.Event()
ev_t0 = torch.cuda.Event(enable_timing=True)
ev_t1 = torch.cuda.Event(enable_timing=True)
pred = StickyOrderPredictor(rank, size, window=8, stickiness_threshold=0.6)
rot_packed = sum(((rank + i) % size) << (4 * i) for i in range(size))


def make_round(it):
    A_it = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A_it.append((torch.randn(M_PR, K, device="cuda") * SCALE).half())
    a = A_it[rank]
    with torch.no_grad():
        ref = torch.cat(A_it, dim=0).float() @ b.float().t()
    return a, ref


def arm_table(it, straggler, delta):
    """The ONLY thing that differs between arms: the table content."""
    if ARM == "nopred":
        return {"seq": it, "degraded": 0, "mode": 0,
                "order_packed": rot_packed, "k_eff": ag.num_chunks}, None
    if ARM == "oracle":
        mode = 1 if (straggler != rank and delta > 0) else 0
        packed = (rank | ((1 - rank) << 4)) if mode else rot_packed
        return {"seq": it, "degraded": 0, "mode": mode,
                "order_packed": packed, "k_eff": ag.num_chunks}, None
    t = pred.table(k_eff=ag.num_chunks)
    return t, t


def run_round(it, straggler=None, delta=None):
    if straggler is None:
        straggler, delta = CH[it]
    a, ref = make_round(it)
    align()
    t0 = time.perf_counter()
    table, last = arm_table(it, straggler, delta)
    i1.i1_table_write(ictx, table["seq"], table["degraded"],
                      table["mode"], table["order_packed"],
                      table["k_eff"])
    host_us = (time.perf_counter() - t0) * 1e6
    if delta > 0 and rank == straggler:
        time.sleep(delta / 1e6)  # host-origin injection (rig header note)
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")
    ev_t0.record()
    i1.i1_ag_tr_op(a, b, c, ictx, with_observer=not NO_OBS)
    ev_t1.record()
    ev_round.record()
    ev_round.synchronize()
    md = (c.float() - ref).abs().max().item()
    obs = i1.i1_observer_read(ictx)
    ms = ev_t0.elapsed_time(ev_t1)
    rec = {"it": it, "straggler": straggler, "delta_us": round(delta, 1),
           "ms": round(ms, 4), "md": round(md, 6),
           "mode": table["mode"], "degraded": table.get("degraded", 0),
           "host_us": round(host_us, 2),
           "obs_marker": obs[0], "obs_peer_count": obs[3 + (1 - rank)]}
    if last is not None:
        rec["s_hat"] = last["s_hat"]
        rec["stickiness"] = last["stickiness"]
    return rec, obs


# warmup (compile paths; discarded), then a causal per-rank timing
# baseline, then measurement
for it in range(WARMUP):
    run_round(it)
align()
# 2026-10-14 rig fact: the observer probe COUNT collapsed under the
# all-kernel gated path (scheduling-mediated) -- the pred arm's lateness
# signal is THIS rank's own round time vs a forced-delta-0 baseline
# (slow round == I waited == peer is the straggler). Four delta-0 rounds
# right before measurement: early rounds run ~2x slow, so the baseline
# must not include the warmup chain deltas.
_base = [run_round(900 + j, straggler=1, delta=0.0)[0]["ms"]
         for j in range(4)]
BASE_MS = min(_base)
BASE_THR = BASE_MS + max(0.12, 0.04 * BASE_MS)
print(f"[rank {rank}] base rounds={[round(x, 3) for x in _base]} "
      f"BASE_MS={BASE_MS:.3f} thr={BASE_THR:.3f}", flush=True)

recs = []
for it in range(WARMUP, WARMUP + ROUNDS):
    rec, obs = run_round(it)
    recs.append(rec)
    # pred arm: feed THIS round's time-based inference to the predictor
    # for NEXT round's table (1-round stale by construction)
    if ARM == "pred":
        late = rec["ms"] > BASE_THR
        s_inf = (1 - rank) if late else rank
        # record the INFERENCE itself (not just next round's s_hat): the
        # design line's oracle-gap decomposition needs per-round
        # inferred / predicted / actual straggler to split identification
        # lag from prediction error (防套壳关②).
        rec["s_inf"] = s_inf
        pred.observe(s_inf)
    if (it - WARMUP) % 10 == 0:
        print(f"[rank {rank}] it={it} {rec}", flush=True)

i1.finalize_i1_context(ictx)
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402
dcu_state()["host"].finalize()
align()
mpi.MPI_Finalize()

out = {"arm": ARM, "tier": TIER, "delta_nominal_us": DELTA_US,
       "rounds": ROUNDS, "shape": [M_PR, K, N], "seed": SEED,
       "p_stay": P_STAY, "rank": rank, "records": recs,
       "base_ms": BASE_MS, "base_thr_ms": BASE_THR,
       "pred_stats": pred.stats() if ARM == "pred" else None}
tag = f"l1_{ARM}_{TIER}_{int(DELTA_US)}us_r{ROUNDS}"
with open(f"bench/results/{tag}_rank{rank}.json", "w") as f:
    json.dump(out, f, indent=1)
print(f"[rank {rank}] L1 DONE -> {tag}_rank{rank}.json "
      f"fails={sum(1 for r in recs if r['md'] >= 1e-2)}", flush=True)
os._exit(0 if all(r["md"] < 1e-2 for r in recs) else 1)
