"""I1: host-side sticky arrival-order predictor (paper #2 I1 baseline).

Runs ONCE PER FORWARD on the host, plain python, zero dependencies (no
torch import -- the caller wires it to the table plane in ag_tr_pred).
Everything it encodes is a measured fact from the inference line:

- decode straggler stickiness p_stay ~= 0.98, hazard L10 stayP 0.979
  (msweep main table + straggler_predict.json runHazard) -> LAST-VOTE
  (predict the last observed straggler) is the right baseline family;
- ph7-class decorrelated segments stayIn 0.146-0.331 -> an ONLINE
  STICKINESS metric must gate the table, below threshold the table
  DEGRADES to the sequential/rotation form (never worse than no table);
- delta p50 534us is host-origin and M-independent -> the input signal is
  arrival ORDER / lateness marks, not message size;
- tail top-1% events are 26-39% of wait -> explicitly NOT handled here
  (that is I3); the predictor only needs to be right about the typical
  sticky majority.

Output contract (the I1 table format, shared with the inference line's
L0 offline replay -- see docs/I1_PREDICTOR_L1_20261013.md table spec):
  int32 words per forward: [seq, degraded, mode, k_eff, order[R],
  budget[R]] where mode 0 = band-major swizzle (S14 current behavior),
  mode 1 = source-major wave order (predicted-early source's ALL chunks
  first -- hides the late source's delta under the early source's
  compute). At R=2 the order[] content is degenerate by design (own
  source never waits -- skip-local split), so on 2 cards the whole
  signal rides on `mode`; order[] becomes the payload at R>=3.

The three divergence axes (predictor family / update granularity /
table form) are deliberately NOT here: this is the fixed BASELINE the
L1 three-arm experiment runs against.
"""


class StickyOrderPredictor:
    """Last-vote sticky predictor with online stickiness gating.

    Input per round: the straggler rank inferred from THIS rank's own
    observation (on R=2: peer-late flag from the observer plane + own
    staging knowledge; at R>=3: the peer arrival order). The input is
    one round stale by construction (observer D2H lands next round) --
  acceptable at p_stay 0.98.
    """

    def __init__(self, rank, num_ranks, window=8, stickiness_threshold=0.6):
        assert num_ranks <= 16, "order packing is 4 bits/slot (R<=16)"
        self.rank = rank
        self.num_ranks = num_ranks
        self.window = window
        self.stickiness_threshold = stickiness_threshold
        self.history = []          # inferred straggler per round (last W)
        self.seq = 0               # table sequence number (monotonic)
        self.n_tables = 0
        self.n_degraded = 0
        self.last = None           # last table() result (for logging)

    # ---- input -------------------------------------------------------
    def observe(self, straggler_inferred):
        """Feed this round's inference (int rank id, or None if the
        observer produced nothing). Call once per round, BEFORE table()."""
        self.history.append(straggler_inferred)
        if len(self.history) > self.window:
            self.history.pop(0)

    # ---- output ------------------------------------------------------
    def rotation(self):
        """S14's own-first rotation order (the no-prediction content)."""
        return [(self.rank + i) % self.num_ranks
                for i in range(self.num_ranks)]

    def stickiness(self):
        h = [x for x in self.history if x is not None]
        if len(h) < 2:
            return None
        same = sum(1 for a, b in zip(h, h[1:]) if a == b)
        return same / (len(h) - 1)

    def predict(self):
        """Returns (s_hat, degraded, cold). s_hat = predicted straggler
        rank for THIS round; degraded = fall back to mode-0 rotation."""
        if len(self.history) < 2 or self.history[-1] is None:
            return None, False, True
        s_hat = self.history[-1]              # last vote (p_stay family)
        st = self.stickiness()
        degraded = st is not None and st < self.stickiness_threshold
        return s_hat, degraded, False

    def table(self, k_eff, budget_default=128):
        """Build this forward's table words. Returns a dict; the caller
        launches the writer kernel from it (ag_tr_pred.i1_table_write).
        order rule: own rank first (own source never waits under the
        skip-local split), predicted straggler LAST, other peers in
        natural order (last-vote predicts only the straggler, not the
        full order -- full-order families are a divergence axis)."""
        s_hat, degraded, cold = self.predict()
        if cold or degraded or s_hat is None:
            mode, order = 0, self.rotation()
        elif s_hat == self.rank:
            # I am the predicted straggler: my own consumer sees no skew
            # (my peer staged on time) -> current behavior is optimal
            mode, order = 0, self.rotation()
        else:
            mode = 1
            order = ([self.rank]
                     + [p for p in range(self.num_ranks)
                        if p not in (self.rank, s_hat)]
                     + [s_hat])
        self.seq += 1
        self.n_tables += 1
        if degraded:
            self.n_degraded += 1
        out = {
            "seq": self.seq,
            "degraded": 1 if degraded else 0,
            "mode": mode,
            "k_eff": int(k_eff),
            "order": order,
            "order_packed": sum(int(o) << (4 * i)
                                for i, o in enumerate(order)),
            "budget": budget_default,
            "s_hat": s_hat,
            "cold": cold,
            "stickiness": self.stickiness(),
        }
        self.last = out
        return out

    def stats(self):
        return {
            "rank": self.rank,
            "n_tables": self.n_tables,
            "n_degraded": self.n_degraded,
            "degrade_rate": (self.n_degraded / self.n_tables
                             if self.n_tables else 0.0),
            "window": self.window,
            "stickiness_threshold": self.stickiness_threshold,
            "last_stickiness": self.stickiness(),
        }


__all__ = ["StickyOrderPredictor"]
