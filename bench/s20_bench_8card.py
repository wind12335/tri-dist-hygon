"""S20: 8-card AG predictor NATURAL-SKEW bench (M8192 N29568 K8192).

The predictor question at R=8: does natural fabric skew exist, is it
round-to-round correlated (the predictor's fuel), and does pred beat
nopred? NO injection -- deltas come from the platform itself (barrier
release jitter, launch jitter, fabric congestion). Decision gates
(pre-registered, before any run):

  G1 skew exists   : per-round straggler probe spread p95 >= ~1 (raw
                     probe-count units; calibrated meaning = order-of
                     poll sweeps) -- i.e. peers do NOT arrive together
  G2 fuel exists   : P(straggler_t == straggler_{t-1}) >> 1/(R-1)
  G3 pred wins     : run-level median(pred) < median(nopred) outside
                     the nopred/nopred2 repeatability spread

If G1/G2 fail the predictor is rig-only on this platform -- a clean
negative result for the paper's platform analysis, stop there.

Arms (S20_ARMS, order matters; one ictx for ALL arms -- S18 wedge law):
  skew     nopred table, observer ON, full obs logged every round
           (natural-skew measurement, S20_ROUNDS_SKEW rounds)
  nopred   fixed rotation table, timing only
  pred     StickyOrderPredictor: R>=3 inference = LAST PEER in the
           previous round's observer order (1-round stale), mode-1
           source-major table, straggler last
  nopred2  repeatability control (drift detector for the A/B)
  s14      paper-#1 op on ictx.ag (reference + corr)
  rccl_seq all_gather_into_tensor + vendor matmul (reference + corr)

Per round: table write -> align barrier -> CUDA-event-timed i1 op
(observer ON in EVERY timing arm: identical stream topology) -> GPU md
check vs one-time ref -> obs D2H read. Data is FIXED across rounds
(steady-state loop, same as s18; natural skew does not need fresh
data). I1_VERIFY=0 by default (S20_VERIFY=1 to enable the tear guard).

Shape notes (R=8): m_pr=1024, chunk_rows=256 -> n_chunks=4 (finer
gating granularity for the predictor; engine pulls pinned -- the
chunks=2+8MB kernel-pull wedge shape does not apply).

run:
  bash bench/s20_run.sh            # mpirun -np 8, timeout 2400
  # single arm subset:
  S20_ARMS=skew,nopred,pred bash bench/s20_run.sh
"""
import json
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_s20_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert 2 <= size <= 16, "order packing is 4 bits/slot (R<=16)"

ARMS = [a for a in os.environ.get(
    "S20_ARMS", "skew,nopred,pred,nopred2,s14,rccl_seq").split(",") if a]
ROUNDS = int(os.environ.get("S20_ROUNDS", "400"))
ROUNDS_SKEW = int(os.environ.get("S20_ROUNDS_SKEW", "600"))
M = int(os.environ.get("S20_M", "8192"))
N = int(os.environ.get("S20_N", "29568"))
K = int(os.environ.get("S20_K", "8192"))
CHUNK_ROWS = int(os.environ.get("S20_CHUNK_ROWS", "256"))
LANES = int(os.environ.get("S20_LANES", "2"))
OUTDIR = os.path.join(REPO, "bench", "results")

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
from triton_dist.kernels.dcu import gemm_ag_tr as agtr  # noqa: E402
from triton_dist.kernels.dcu.i1_predictor import (  # noqa: E402
    StickyOrderPredictor)
import torch.distributed as dist  # noqa: E402

i1.I1_VERIFY = os.environ.get("S20_VERIFY", "0") == "1"

M_PR = M // size
DTYPE = torch.float16
SCALE = 0.02
WARMUP = 3

b = (torch.randn(N, K, device="cuda") * SCALE).half()
ictx = i1.create_i1_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm,
    chunk_rows=CHUNK_ROWS, comm_lanes=LANES)
ag = ictx.ag
assert ag.num_chunks >= 2, "wave-order mechanism needs >= 2 chunks"

# fixed data + one-time chunked-fp32 reference (s18 style)
shards = []
for s in range(size):
    torch.manual_seed(7100 + s)
    shards.append((torch.randn(M, K, device="cuda") * SCALE).half())
a_loc = shards[rank][rank * M_PR:(rank + 1) * M_PR].contiguous()
A_full = torch.cat([shards[s][s * M_PR:(s + 1) * M_PR] for s in range(size)],
                   dim=0)
del shards
c = torch.empty(M, N, dtype=DTYPE, device="cuda")
ref = torch.empty(M, N, dtype=DTYPE, device="cuda")
bf = b.float()
for r0 in range(0, M, 1024):
    ref[r0:r0 + 1024] = (A_full[r0:r0 + 1024].float() @ bf.t()).half()
del A_full, bf
torch.cuda.empty_cache()
Ag_buf = torch.empty(M, K, dtype=DTYPE, device="cuda")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29731")
dist.init_process_group(backend="nccl", rank=rank, world_size=size)

print(f"[rank {rank}] S20 ready: R={size} M{M} N{N} K{K} "
      f"chunk_rows={ag.chunk_rows} n_chunks={ag.num_chunks} "
      f"lanes={ag.comm_lanes} verify={i1.I1_VERIFY} arms={ARMS} "
      f"({torch.cuda.max_memory_allocated()/2**30:.1f}GB)", flush=True)


def maxdiff():
    md = 0.0
    for r0 in range(0, M, 2048):
        md = max(md, (c[r0:r0 + 2048].float()
                      - ref[r0:r0 + 2048].float()).abs().max().item())
    return md


def parse_obs(obs):
    """obs[0]=round marker (neg => capped/no-obs); obs[1:R]=peer finish
    ORDER; obs[R+1:2R+1]=per-source probe counts (-1 never)."""
    if obs[0] < 0:
        return None, None
    peers = obs[1:size]
    probes = obs[size + 1:2 * size + 1]
    if any(p < 0 for p in peers):
        return None, probes
    return peers, probes          # peers[-1] = LAST finisher = straggler


ev_t0 = torch.cuda.Event(enable_timing=True)
ev_t1 = torch.cuda.Event(enable_timing=True)
ev_done = torch.cuda.Event()
pred = StickyOrderPredictor(rank, size, window=8, stickiness_threshold=0.6)
rot_packed = sum(((rank + i) % size) << (4 * i) for i in range(size))


def write_rot_table(it):
    i1.i1_table_write(ictx, it, 0, 0, rot_packed, ag.num_chunks)


def run_pred_arm(tag, n_rounds, adaptive):
    """skew / nopred / nopred2 / pred share this loop; arms differ ONLY
    in the table policy (and what gets logged)."""
    recs = []
    obs_prev = None
    fails = 0
    for it in range(n_rounds):
        # --- table (host, before the round; pred uses LAST round's obs)
        t0 = time.perf_counter()
        if adaptive:
            peers_prev, _ = parse_obs(obs_prev) if obs_prev else (None, None)
            s_inf = peers_prev[-1] if peers_prev else None
            pred.observe(s_inf)
            table = pred.table(k_eff=ag.num_chunks)
            i1.i1_table_write(ictx, table["seq"], table["degraded"],
                              table["mode"], table["order_packed"],
                              table["k_eff"])
            host_us = (time.perf_counter() - t0) * 1e6
        else:
            write_rot_table(it)
            table = {"mode": 0, "degraded": 0}
            host_us = (time.perf_counter() - t0) * 1e6
        align()
        ev_t0.record()
        i1.i1_ag_tr_op(a_loc, b, c, ictx, with_observer=True)
        ev_t1.record()
        ev_done.record()
        ev_done.synchronize()
        ms = ev_t0.elapsed_time(ev_t1)
        md = (c.float() - ref).abs().max().item()
        if md >= 1e-2:
            fails += 1
        obs = i1.i1_observer_read(ictx)
        peers, probes = parse_obs(obs)
        rec = {"it": it, "ms": round(ms, 4), "md": round(md, 6),
               "mode": table["mode"], "deg": table["degraded"],
               "host_us": round(host_us, 1)}
        if peers is not None:
            rec["straggler"] = peers[-1]
            rec["order"] = peers
            rec["probes"] = probes
        if adaptive:
            rec["s_inf"] = s_inf
            rec["s_hat"] = table["s_hat"] if "s_hat" in table else None
            rec["stick"] = table["stickiness"] if "stickiness" in table \
                else None
        recs.append(rec)
        obs_prev = obs
        if it % 100 == 0:
            print(f"[rank {rank}] {tag} it={it} ms={ms:.2f} "
                  f"strag={rec.get('straggler')} fails={fails}", flush=True)
    return recs, fails


def summarize_rounds(recs, fails):
    mss = sorted(r["ms"] for r in recs)
    out = {"n": len(recs), "fails": fails,
           "p25": round(mss[len(mss) // 4], 3),
           "p50": round(mss[len(mss) // 2], 3),
           "p75": round(mss[3 * len(mss) // 4], 3),
           "worst_md": max(r["md"] for r in recs)}
    strag = [r["straggler"] for r in recs if "straggler" in r]
    if len(strag) >= 2:
        stay = sum(1 for a, b2 in zip(strag, strag[1:]) if a == b2)
        out["strag_stay_frac"] = round(stay / (len(strag) - 1), 4)
        out["strag_random_baseline"] = round(1.0 / (size - 1), 4)
        hist = {}
        for s in strag:
            hist[s] = hist.get(s, 0) + 1
        out["strag_hist"] = hist
        out["obs_rate"] = round(len(strag) / len(recs), 3)
    pr = [max(r["probes"]) - min(r["probes"]) for r in recs
          if "probes" in r]
    if pr:
        pr.sort()
        out["probe_spread_p50"] = pr[len(pr) // 2]
        out["probe_spread_p95"] = pr[int(len(pr) * 0.95)]
    return out


def run_ref_arm(kind):
    if kind == "s14":
        fn = lambda: agtr.gemm_ag_tr_op(a_loc, b, c, ictx.ag)
    else:
        def fn():
            dist.all_gather_into_tensor(Ag_buf, a_loc)
            torch.matmul(Ag_buf, b.t(), out=c)
    fn()
    torch.cuda.synchronize()
    md = maxdiff()
    st = torch.cuda.current_stream()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(30):
        ev_t0.record(st)
        fn()
        ev_t1.record(st)
        ev_done.record(st)
        ev_done.synchronize()
        ts.append(ev_t0.elapsed_time(ev_t1))
    ts.sort()
    return {"p50": round(ts[len(ts) // 2], 3), "corr": md}


for arm in ARMS:
    tag = f"s20_8card_{arm}"
    try:
        if arm in ("skew", "nopred", "nopred2", "pred"):
            n = ROUNDS_SKEW if arm == "skew" else ROUNDS
            recs, fails = run_pred_arm(arm, n, adaptive=(arm == "pred"))
            out = {"arm": arm, "rank": rank, "R": size,
                   "M": M, "N": N, "K": K, "chunk_rows": ag.chunk_rows,
                   "summary": summarize_rounds(recs, fails),
                   "pred_stats": pred.stats() if arm == "pred" else None,
                   "records": recs}
        elif arm in ("s14", "rccl_seq"):
            out = {"arm": arm, "rank": rank, "R": size,
                   "M": M, "N": N, "K": K,
                   "summary": run_ref_arm(arm)}
        else:
            raise ValueError(f"unknown arm {arm}")
        out["status"] = "ok"
    except Exception:
        out = {"arm": arm, "rank": rank, "status": "ERR",
               "tb": traceback.format_exc()[-2000:]}
        print(f"[rank {rank}] {arm} ERR:\n{out['tb']}", flush=True)
    with open(os.path.join(OUTDIR, f"{tag}_rank{rank}.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"[rank {rank}] ARM DONE {arm}: "
          f"{json.dumps(out.get('summary', {}))[:300]}", flush=True)

try:
    i1.finalize_i1_context(ictx)
except Exception:
    pass
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402
try:
    dcu_state()["host"].finalize()
except Exception:
    pass
align()
mpi.MPI_Finalize()
print(f"[rank {rank}] S20 DONE", flush=True)
os._exit(0)
