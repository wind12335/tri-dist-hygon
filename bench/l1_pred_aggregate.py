"""L1 three-arm aggregator (no GPU): reads the l1_pred_bench JSONs and
evaluates the PRE-REGISTERED criteria C1-C5 (see l1_pred_bench.py header,
written before any run):

  C1 correctness      every round, every arm, md < 1e-2
  C2 host overhead    p95(pred arm host_us) < 10 us
  C3 recovery         sticky & delta >= 400: (T_np - T_pd)/(T_np - T_or)
                      >= 0.5 on victim rounds (median over victim rounds
                      where the oracle actually helped)
  C4 decorr safety    recovery >= -0.05 AND pred degrade_rate >= 0.5
  C5 delta=0 tax      median |T_pd - T_np|/T_np <= 0.02

Victim round = rank != straggler and delta > 0 (the straggler's own round
never waits for itself). Rounds pair 1:1 across arms via the shared chain
seed, so per-(rank, it) triplets (nopred, pred, oracle) are exact pairs.

python3 bench/l1_pred_aggregate.py          # auto-discovers cells
python3 bench/l1_pred_aggregate.py 40       # restrict to ROUNDS=40
"""
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

NAME_RE = re.compile(
    r"l1_(nopred|pred|oracle)_(sticky|decorr)_(\d+)us_r(\d+)_rank(\d+)\.json$")


def load(rounds_filter=None):
    cells = {}  # (tier, delta, rounds) -> {arm -> {rank -> records}}
    for path in sorted(glob.glob(os.path.join(RESULTS, "l1_*.json"))):
        m = NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        arm, tier, delta, rounds, rank = m.groups()
        if rounds_filter and int(rounds) != rounds_filter:
            continue
        with open(path) as f:
            doc = json.load(f)
        key = (tier, int(delta), int(rounds))
        cells.setdefault(key, {}).setdefault(arm, {})[int(rank)] = doc
    return cells


def victim_pairs(docs_np, docs_pd, docs_or, rank):
    """Yield (it, t_np, t_pd, t_or) for this rank's victim rounds present
    in ALL three arms (paired by shared-chain it)."""
    by_it = {}
    for arm, docs in (("np", docs_np), ("pd", docs_pd), ("or", docs_or)):
        if rank not in docs or arm == "or" and rank not in docs:
            return
        for rec in docs[rank]["records"]:
            if rec["straggler"] != rank and rec["delta_us"] > 0:
                by_it.setdefault(rec["it"], {})[arm] = rec
    for it, d in sorted(by_it.items()):
        if len(d) == 3:
            yield it, d["np"]["ms"], d["pd"]["ms"], d["or"]["ms"]


def cell_report(tier, delta, rounds, arms, checks, verbose):
    """Print one (tier, delta) cell; feed the global checks dict."""
    R = sorted({r for docs in arms.values() for r in docs})
    line = f"--- tier={tier} delta={delta}us rounds={rounds} ranks={R}"
    print(line)

    # C1 correctness (all arms, all rounds -- not just victims)
    worst_md, n_rec = 0.0, 0
    for arm, docs in sorted(arms.items()):
        for r, doc in sorted(docs.items()):
            for rec in doc["records"]:
                worst_md = max(worst_md, rec["md"])
                n_rec += 1
    checks["C1_md_max"] = max(checks.get("C1_md_max", 0.0), worst_md)
    checks["C1_n_records"] = checks.get("C1_n_records", 0) + n_rec
    print(f"    C1  worst md = {worst_md:.2e} over {n_rec} records "
          f"({'OK' if worst_md < 1e-2 else 'FAIL'})")

    if "pred" in arms:  # C2 host overhead
        hus = [rec["host_us"] for docs in arms["pred"].values()
               for rec in docs["records"]]
        hus.sort()
        p95 = hus[max(0, int(0.95 * len(hus)) - 1)]
        checks["C2_p95_host_us"] = max(checks.get("C2_p95_host_us", 0.0),
                                       p95)
        print(f"    C2  pred host_us p95 = {p95:.1f} us (n={len(hus)}) "
              f"({'OK' if p95 < 10 else 'FAIL'})")

    if all(a in arms for a in ("nopred", "pred", "oracle")) and delta > 0:
        recs, helped = [], 0
        for rank in R:
            for it, t_np, t_pd, t_or in victim_pairs(
                    arms["nopred"], arms["pred"], arms["oracle"], rank):
                denom = t_np - t_or
                if denom <= 0:
                    continue  # oracle didn't help this round
                helped += 1
                recs.append((t_np - t_pd) / denom)
        if recs:
            med = statistics.median(recs)
            hurt = sum(1 for x in recs if x < 0) / len(recs)
            print(f"    recovery (victim, helped={helped}): "
                  f"median={med:.3f} mean={statistics.mean(recs):.3f} "
                  f"p25={sorted(recs)[len(recs)//4]:.3f} "
                  f"hurt_frac={hurt:.2f}")
            if delta >= 400:
                key = "C3_recovery_sticky" if tier == "sticky" \
                    else "C4_recovery_decorr"
                # conservative: the criterion must hold in EVERY active
                # cell, so keep the WORST (min) over delta>=400 cells
                prev = checks.get(key)
                checks[key] = med if prev is None else min(prev, med)
        else:
            print("    recovery: NO helped victim rounds (degenerate)")
        # context: per-arm victim medians
        for arm in ("nopred", "pred", "oracle"):
            vt = [rec["ms"] for rank in R
                  for rec in arms[arm][rank]["records"]
                  if rec["straggler"] != rank and rec["delta_us"] > 0]
            if vt:
                print(f"    T_{arm} victim median = "
                      f"{statistics.median(vt):.3f} ms")

    if "pred" in arms and tier == "decorr":  # C4 degrade gate
        for r, doc in sorted(arms["pred"].items()):
            st = doc.get("pred_stats") or {}
            if st:
                print(f"    C4  rank{r} degrade_rate = "
                      f"{st.get('degrade_rate', float('nan')):.2f} "
                      f"(n_tables={st.get('n_tables')})")
                checks["C4_degrade_rate_max"] = max(
                    checks.get("C4_degrade_rate_max", 0.0),
                    st.get("degrade_rate", 0.0))

    if delta == 0 and "nopred" in arms and "pred" in arms:  # C5 tax
        ratios = []
        for rank in R:
            np_by = {rec["it"]: rec["ms"]
                     for rec in arms["nopred"][rank]["records"]}
            for rec in arms["pred"][rank]["records"]:
                if rec["it"] in np_by and np_by[rec["it"]] > 0:
                    ratios.append(abs(rec["ms"] - np_by[rec["it"]])
                                  / np_by[rec["it"]])
        if ratios:
            med = statistics.median(ratios)
            checks["C5_tax_median"] = max(checks.get("C5_tax_median", 0.0),
                                          med)
            print(f"    C5  |T_pd-T_np|/T_np median = {med:.4f} "
                  f"({'OK' if med <= 0.02 else 'FAIL'})")


def main():
    rounds_filter = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cells = load(rounds_filter)
    if not cells:
        print(f"no l1_*.json found in {RESULTS}")
        sys.exit(2)
    checks, summary = {}, {}
    for (tier, delta, rounds), arms in sorted(cells.items()):
        cell_report(tier, delta, rounds, arms, checks, True)
        summary[f"{tier}_{delta}us"] = {
            arm: {f"rank{r}": {
                "victim_ms_median": _med([x["ms"] for x in d["records"]
                                          if x["straggler"] != r
                                          and x["delta_us"] > 0]),
                "straggler_ms_median": _med([x["ms"] for x in d["records"]
                                             if x["straggler"] == r
                                             and x["delta_us"] > 0]),
                "host_us_p95": _pctl([x["host_us"] for x in d["records"]],
                                     0.95),
            } for r, d in sorted(docs.items())}
            for arm, docs in sorted(arms.items())}
        if "pred" in arms:
            for r, doc in sorted(arms["pred"].items()):
                summary[f"{tier}_{delta}us"].setdefault(
                    "pred_stats", {})[f"rank{r}"] = doc.get("pred_stats")

    # final verdict against the pre-registered thresholds
    print("\n=== PRE-REGISTERED VERDICT ===")
    verdict = {}
    verdict["C1_correctness"] = checks.get("C1_md_max", 1.0) < 1e-2
    verdict["C2_host_overhead_p95_us<10"] = checks.get("C2_p95_host_us",
                                                       1e9) < 10
    c3 = checks.get("C3_recovery_sticky")
    verdict["C3_recovery_sticky>=0.5@d>=400"] = None if c3 is None else c3 >= 0.5
    c4r = checks.get("C4_recovery_decorr")
    c4d = checks.get("C4_degrade_rate_max", 0.0)
    verdict["C4_decorr_rec>=-0.05_and_degrade>=0.5"] = (
        None if c4r is None else (c4r >= -0.05 and c4d >= 0.5))
    verdict["C5_delta0_tax<=0.02"] = checks.get("C5_tax_median", 1e9) <= 0.02
    for k, v in verdict.items():
        val = c3 if k.startswith("C3") else (
            c4r if k.startswith("C4") and "rec" in k else None)
        print(f"  {k}: "
              f"{'PASS' if v else ('MISSING' if v is None else 'FAIL')}"
              + (f" ({val:.3f})" if k.startswith("C3") and c3 is not None
                 else "")
              + (f" (rec={c4r:.3f}, degrade={c4d:.2f})"
                 if k.startswith("C4") and c4r is not None else ""))
    out = {"checks": {k: checks.get(k) for k in sorted(checks)},
           "verdict": verdict, "cells": summary}
    with open(os.path.join(RESULTS, "l1_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {RESULTS}/l1_summary.json")


def _med(xs):
    return statistics.median(xs) if xs else None


def _pctl(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[max(0, int(q * len(xs)) - 1)]


if __name__ == "__main__":
    main()
