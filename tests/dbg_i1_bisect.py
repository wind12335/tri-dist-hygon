"""Bisect the one-round C corruption (max_diff 0.068 @ altern iter 303,
test_i1_pred run 2) and the observer-signal problems, one knob per mpirun:

  DBG_OBS=0/1     observer stream on/off                (default 1)
  DBG_INJ=0/1     alternating straggler injection 400us (default 0)
  DBG_MODE=static0|flip   table content: always mode-0 rotation,
                          or mode 0/1 flipping every round (default static0)
  DBG_ALIGN=0/1   per-round MPI_Barrier (rig alignment)  (default 0 =
                  the failing test's condition)
  DBG_ROUNDS=N    measurement rounds after warmup        (default 40)

Every round checks C against a GPU reference; failures are counted and
printed with the round index. The failing phase was mode-0 + injection +
observer + no alignment, i.e. DBG_OBS=1 DBG_INJ=1 DBG_MODE=static0.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/dbg_i1_bisect.py
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_i1dbg_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "bisect is a 2-card rig"

OBS = os.environ.get("DBG_OBS", "1") == "1"
INJ = os.environ.get("DBG_INJ", "0")  # 0 | 1 (device _sleep) | host
MODE = os.environ.get("DBG_MODE", "static0")
ALIGN = os.environ.get("DBG_ALIGN", "0") == "1"
OP = os.environ.get("DBG_OP", "i1")  # i1 = pred kernel | s14 = ORIGINAL op
ROUNDS = int(os.environ.get("DBG_ROUNDS", "40"))
DELTA_US = 400.0

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int


def barrier():
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu import gemm_ag_tr as agtr  # noqa: E402


@triton.jit(do_not_specialize=["dst_ptr", "src_ptr", "n"])
def _kpeek(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    """Post-round ws snapshot: plain loads (L2-coherent after sync) of the
    gathered workspace into a torch tensor -- decides whether a torn C
    came from a torn ws (pull/source side) or a clean ws read stale
    (consumer side)."""
    pid = tl.program_id(0)
    dp = dst_ptr.to(tl.pointer_type(tl.float16))
    sp = src_ptr.to(tl.pointer_type(tl.float16))
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dp + offs, tl.load(sp + offs, mask=m), mask=m)

M_per_rank, K, N = (int(x) for x in
                    os.environ.get("DBG_SHAPE", "2048x1024x2048").split("x"))
M = M_per_rank * size
DTYPE = torch.float16
SCALE = 0.02

b = (torch.randn(N, K, device="cuda") * SCALE).half()
ictx = i1.create_i1_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)
ag = ictx.ag
print(f"[rank {rank}] bisect OBS={OBS} INJ={INJ} MODE={MODE} "
      f"ALIGN={ALIGN} ROUNDS={ROUNDS} chunks={ag.num_chunks} "
      f"lanes={ag.comm_lanes}", flush=True)

ictx.cycles_per_us = i1.calibrate_sleep_cycles_per_us()
print(f"[rank {rank}] cycles_per_us={ictx.cycles_per_us:.1f}", flush=True)

rot_packed = sum(((rank + i) % size) << (4 * i) for i in range(size))
mode1_packed = rank | ((1 - rank) << 4)
ev_round = torch.cuda.Event()
ev_t0 = torch.cuda.Event(enable_timing=True)
ev_t1 = torch.cuda.Event(enable_timing=True)


# DBG_FRESHA=1: serve `a` from a 16-deep pool so consecutive rounds use
# DIFFERENT VAs (defeats allocator LIFO reuse). Discriminates the
# stale-DRAM-read mechanism: if the own-band damage is stale reads of
# localA, match_prev collapses (stale content is now 16 rounds old)
# while n_wrong stays; if the damage is a reuse race on the tensor,
# n_wrong itself collapses.
FRESHA = os.environ.get("DBG_FRESHA", "0") == "1"
_A_POOL = [None] * 16


def make_round_ref(it):
    """Returns (a, ref, A_full) -- caller keeps A_full for stale forensics."""
    A_it = []
    for s in range(size):
        torch.manual_seed(5000 + 100 * it + s)
        A_it.append((torch.randn(M_per_rank, K, device="cuda") * SCALE).half())
    a = A_it[rank]
    if FRESHA:
        # copy MY shard into a rotating slot; the ref still uses the
        # true content (identical values, different VA for localA)
        slot = _A_POOL[it % 16]
        if slot is None:
            slot = torch.empty_like(a)
            _A_POOL[it % 16] = slot
        slot.copy_(a)
        a = slot
    A_full = torch.cat(A_it, dim=0)
    with torch.no_grad():
        ref = A_full.float() @ b.float().t()
    return a, ref, A_full


_prev_A = None
_prev_ref = []  # last 3 rounds' refs (v10 bit-exact stale matching)


def one_round(it):
    global _prev_A
    a, ref, A_full = make_round_ref(it)
    a, ref, A_full = make_round_ref(it)
    mode = (it % 2) if MODE == "flip" else 0
    packed = mode1_packed if mode else rot_packed
    straggler = (it % 2) if INJ else None
    # host-inj: the straggler's HOST is delta late into the round -> its
    # staging lands delta late at the peer, same arrival skew as the
    # device sleep but with NO _sleep kernel in the stream (bisects
    # "_sleep artifact" vs "skew exposes a ticketed-protocol race")
    if INJ == "host" and rank == straggler:
        time.sleep(DELTA_US / 1e6)
    ev_t0.record()
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")
    if ictx.diag_buf is not None:
        ictx.diag_buf.fill_(1 << 30)
    if OP == "s14":
        agtr.gemm_ag_tr_op(a, b, c, ictx.ag)
    else:
        i1.i1_table_write(ictx, it, 0, mode, packed, ag.num_chunks)
        delay = (DELTA_US if rank == straggler else 0.0) if INJ == "1" else 0.0
        i1.i1_ag_tr_op(a, b, c, ictx, stage_delay_us=delay, with_observer=OBS)
    ev_t1.record()
    ev_round.record()
    ev_round.synchronize()
    md = (c.float() - ref).abs().max().item()
    ms = ev_t0.elapsed_time(ev_t1)
    peer_count = None
    if OBS and OP != "s14":
        obs = i1.i1_observer_read(ictx)
        peer_count = obs[3 + (1 - rank)]
        if obs[0] < 0:
            peer_count = -1  # probe-cap marker
    if md >= 1e-2:
        # forensics v6 (2026-10-14): RE-VERIFY after an L2-flushing dummy
        # pass. Gate/pull/ws/localA all proven correct while C "torn" and
        # 29% of damage matching the PREV round's C at the same VAs --
        # consistent with the VERIFY READ itself taking the
        # stale-DRAM-lags-L2 path (the S3 rule on the read side) under
        # L2 pressure at big shapes. If md2 collapses, the "tear" is a
        # verification-read artifact, not data corruption.
        _l2scrub = torch.empty(64 * 1024 * 1024 // 4, dtype=torch.float32,
                               device="cuda")
        _l2scrub.fill_(1.0)
        torch.cuda.synchronize()
        md2 = (c.float() - ref).abs().max().item()
        md3 = (c.float() - ref).abs().max().item()
        print(f"[rank {rank}]   reverify: md2={md2:.5f} md3={md3:.5f} "
              f"(md1={md:.5f})", flush=True)
        # forensics v2: wrong band -> wrong CHUNK band (CR rows) -> and
        # of the wrong elements, how many match the PREVIOUS round's values
        # (stale whole-round read) vs match neither (torn overwrite)?
        d = (c.float() - ref).abs()
        band = [float(d[s * M_per_rank:(s + 1) * M_per_rank].max())
                for s in range(size)]
        CR_rows = M // ag.num_chunks
        cb = [f"{float(d[k * CR_rows:(k + 1) * CR_rows].max()):.4f}"
              for k in range(ag.num_chunks)]
        wrong = d > 1e-2
        n_wrong = int(wrong.sum())
        stale = " n/a"
        if _prev_A is not None:
            with torch.no_grad():
                ref_prev = _prev_A.float() @ b.float().t()
            match_prev = int((((c.float() - ref_prev).abs() < 1e-2)
                              & wrong).sum())
            # per-band split (v5): band[] is a MAX -- it only proves >=1
            # bad element per band. The split shows where the wrong
            # elements actually live, and whether the own-source band
            # (read from the FRESH localA, stale-immune) participates.
            per = []
            for s in range(size):
                sl = slice(s * M_per_rank, (s + 1) * M_per_rank)
                nw = int(wrong[sl].sum())
                mp = int((((c.float() - ref_prev).abs() < 1e-2)
                          & wrong)[sl].sum())
                per.append(f"s{s}:n={nw}/prev={mp}")
            # displaced-data test: do the wrong elements match the CURRENT
            # round's C of the OTHER source's rows (tile mapping scramble)
            # rather than either round's own-position reference?
            dis = []
            for s in range(size):
                o = (s + 1) % size
                sl_s = slice(s * M_per_rank, (s + 1) * M_per_rank)
                sl_o = slice(o * M_per_rank, (o + 1) * M_per_rank)
                nd = int((((c.float()[sl_s] - ref[sl_o]).abs() < 1e-2)
                          & wrong[sl_s]).sum())
                dis.append(f"s{s}<-s{o}:{nd}")
            stale = (f" match_prev={match_prev}/{n_wrong}"
                     f" per_band={{{'; '.join(per)}}}"
                     f" displaced={{{'; '.join(dis)}}}")
        print(f"[rank {rank}] FAIL it={it} md={md:.5f} band={band} "
              f"chunkband={cb} n_wrong={n_wrong}{stale}", flush=True)
        # forensics v7: the damage is ~11 FULL C rows = ~11 scattered
        # wrong A elements (each contaminates its whole C row), not a
        # tile/gate event. (v5 correction: match_prev/displaced ratios
        # both sit at the |d|<1e-2 chance level ~28% -- noise, not
        # signal; the stale-content story was built on that noise.)
        # v7a: which rows, how many wrong cols each.
        rowcnt = wrong.sum(dim=1)
        bad_rows = torch.nonzero(rowcnt).flatten()
        rr = bad_rows[:12].tolist()
        cc = [int(rowcnt[i]) for i in rr]
        print(f"[rank {rank}]   v7 rows(n_bad_cols): "
              f"{list(zip(rr, cc))} of {int(bad_rows.numel())} rows",
              flush=True)
        # v7b: the actual wrong values -- huge garbage vs small perturb?
        dflat = d.flatten()
        topv = torch.topk(dflat, min(8, n_wrong))
        idxs = topv.indices.tolist()
        cv = [round(float(c.flatten()[i]), 4) for i in idxs]
        rv = [round(float(ref.flatten()[i]), 4) for i in idxs]
        print(f"[rank {rank}]   v7 worst (c, ref): "
              f"{list(zip(cv, rv))}", flush=True)
        # v7c: multi-path read-back of c (torch now, kpeek kernel copy,
        # cpu copy). All three agreeing => the wrong bytes are really in
        # c; a single disagreeing path => the verify read itself.
        c_snap = torch.empty_like(c)
        _kpeek[(triton.cdiv(c.numel(), 4096),)](
            c_snap.data_ptr(), c.data_ptr(), c.numel(), BLOCK=4096)
        torch.cuda.synchronize()
        md_kp = (c_snap.float() - ref).abs().max().item()
        md_cpu = (c.cpu().float() - ref.cpu()).abs().max().item()
        print(f"[rank {rank}]   v7 c readback: torch={md2:.5f} "
              f"kpeek={md_kp:.5f} cpu={md_cpu:.5f}", flush=True)
        # forensics v8: WHO is wrong -- c or the GPU-computed reference?
        # Recompute ref on the CPU from the same A_full/b and compare
        # ref_gpu vs ref_cpu at the wrong positions. A wrong ref element
        # => the whole "tear" is a verification-side phantom (the torch
        # GEMM miscomputing ~11 scattered elements) and C1-style md
        # checks must not trust single-path GPU references.
        with torch.no_grad():
            ref_cpu = (A_full.cpu().float() @ b.cpu().float().t())
        d_rc = (ref.cpu() - ref_cpu).abs()
        md_rc = float(d_rc.max())
        nw_rc = int((d_rc > 1e-2).sum())
        rows_rc = torch.nonzero((d_rc > 1e-2).sum(dim=1)).flatten()
        print(f"[rank {rank}]   v8 ref_gpu-vs-cpu: md={md_rc:.5f} "
              f"n_wrong={nw_rc} rows={rows_rc[:12].tolist()}",
              flush=True)
        d_cc = (c.cpu().float() - ref_cpu).abs()
        md_cc = float(d_cc.max())
        nw_cc = int((d_cc > 1e-2).sum())
        print(f"[rank {rank}]   v8 c-vs-ref_cpu: md={md_cc:.5f} "
              f"n_wrong={nw_cc}", flush=True)
        # forensics v9: exact-match search -- is a wrong c[i,j] EXACTLY
        # equal to some other element of the SAME round's ref (fp16 bit
        # equality = displaced placement, not corrupted bits)? Search
        # column displacement within the row and row displacement within
        # the column for the first bad rows.
        with torch.no_grad():
            cf = c.float()
            for i in bad_rows[:3].tolist():
                jw = torch.nonzero(wrong[i]).flatten()
                jw = jw[:16]
                if jw.numel() == 0:
                    continue
                vals = cf[i, jw]                       # (q,)
                # column displacement: vals vs ref[i, :]
                hitc = (vals[:, None] == ref[i][None, :]).any(dim=1)
                # row displacement: vals vs ref[:, jw]
                hitr = (cf[i, jw][None, :] == ref[:, jw]).any(dim=0)
                print(f"[rank {rank}]   v9 row {i}: n_j={int(jw.numel())}"
                      f" exact_col_match={int(hitc.sum())}"
                      f" exact_row_match={int(hitr.sum())}", flush=True)
        # forensics v10: bit-exact match of the wrong c values against
        # the PREVIOUS rounds' refs (the strict stale-read test -- v5's
        # |d|<1e-2 match_prev sat at the ~28% chance floor and proved
        # nothing). A stale ws/localA read yields values that are EXACT
        # fp16 copies of a prior round's C at the same VA.
        if _prev_ref:
            with torch.no_grad():
                for gen, pref in enumerate(_prev_ref):
                    hp = ((cf - pref).abs() < 1e-6) & wrong
                    nh = int(hp.sum())
                    if nh:
                        rows_h = torch.nonzero(hp.sum(dim=1)).flatten()
                        print(f"[rank {rank}]   v10 gen-{gen}: "
                              f"exact_prev_match={nh}/{n_wrong} "
                              f"rows~{rows_h[:6].tolist()}", flush=True)
                    else:
                        print(f"[rank {rank}]   v10 gen-{gen}: "
                              f"exact_prev_match=0/{n_wrong}", flush=True)
        # forensics v11: solve the IMPLIED delta-A. Error-magnitude
        # analysis says each wrong element carries the contribution of
        # ~ONE BLOCK_K=64 k-slice (std 0.0045 predicted vs 0.0044
        # measured; 2.4% of columns over threshold = exactly the
        # P(|N(0,0.0044)|>0.01) tail). e[M,N] = dA[M,K] @ Bt -> with B
        # known and invertible, dA = e @ pinv(Bt): if the damage is one
        # k-fragment load, dA's k-support must be a single 64-aligned
        # block, and its values identify the wrong content (bit-compare
        # against prev-round A and k-shifted current A).
        with torch.no_grad():
            P = torch.linalg.pinv(b.float())          # [K, N]
            e_all = (cf - ref)                         # [M, N]
            dA = (e_all @ P.t()).half().float()        # [M, K]
            arow_mag = dA.abs().max(dim=1).values
            # rows the dA is meaningful for (the damaged rows)
            dmg = torch.nonzero(arow_mag > 0.004).flatten()
            if dmg.numel():
                sup = []
                for i in dmg[:6].tolist():
                    mag = dA[i].abs()
                    thr = 0.25 * float(mag.max())
                    ks = torch.nonzero(mag > thr).flatten()
                    sup.append((i, int(ks[0]), int(ks[-1]),
                                int(ks.numel())))
                print(f"[rank {rank}]   v11 dA rows(dmg={int(dmg.numel())})"
                      f" (row,k0,k1,n_supp): {sup}", flush=True)
                # bit-compare the first damaged row's implied WRONG A
                # content against candidates. dmg rows are GLOBAL rows
                # (dA/ref live in A_full's row space: size*M_per_rank),
                # so the correct content comes from A_full, NOT the
                # local a (first capture crashed indexing a[3848] with
                # a having only M_per_rank rows). The observed signature
                # (s1-band only, dA DENSE over all K, c ~ -ref pairs,
                # no exact prev match) calls for a full-K fingerprint +
                # a cur/prev MIX count: a torn (partially-propagated)
                # tile matches each generation on part of its elements
                # and neither on the rest.
                i0 = int(dmg[0])
                K_ = A_full.shape[1]
                cur = A_full.float()[i0]              # correct content
                implied = cur + dA[i0]                # content actually used

                def _hits(cv):
                    return int(((cv - implied).abs() < 1e-3).sum())

                fp = [f"cur={_hits(cur)}/{K_}"]
                pv = _prev_A.float()[i0] if _prev_A is not None else None
                if pv is not None:
                    fp.append(f"prev={_hits(pv)}/{K_}")
                fp.append(f"neg={_hits(-cur)}/{K_}")
                fp.append(f"zero={int((implied.abs() < 1e-4).sum())}/{K_}")
                for nm, row in (("s0swp", i0 - M_per_rank),
                                ("r+CR", i0 + CR_rows),
                                ("r-CR", i0 - CR_rows)):
                    if 0 <= row < A_full.shape[0]:
                        fp.append(f"{nm}={_hits(A_full.float()[row])}/{K_}")
                print(f"[rank {rank}]   v11 implied-A fingerprint row={i0}:"
                      f" {' '.join(fp)}", flush=True)
                if pv is not None:
                    m_cur = (cur - implied).abs() < 1e-3
                    m_prev = (pv - implied).abs() < 1e-3
                    print(f"[rank {rank}]   v11 mix: "
                          f"cur_only={int((m_cur & ~m_prev).sum())} "
                          f"prev_only={int((m_prev & ~m_cur).sum())} "
                          f"neither={int((~m_cur & ~m_prev).sum())} "
                          f"both={int((m_cur & m_prev).sum())}", flush=True)
                # rowscan: does the implied content on the DAMAGED k's
                # of row i0 equal some OTHER row of current/prev A_full?
                # A true row/chunk displacement matches ~100% on the
                # candidate row; chance level is ~1-2% per element
                # (fp16 quantized randn*0.02 at 1e-3 tolerance).
                bad_k = (implied - cur).abs() > 1e-3
                if int(bad_k.sum()) > 64:
                    q = implied[None, :]
                    for nm, AF in (("curAF", A_full),
                                   ("prevAF", _prev_A
                                    if _prev_A is not None else A_full)):
                        sc = (((AF.float() - q).abs() < 1e-3)
                             [:, bad_k].float().mean(dim=1))
                        top = sc.topk(2)
                        print(f"[rank {rank}]   v11 rowscan {nm}: "
                              f"best_row={int(top.indices[0])} "
                              f"frac={float(top.values[0]):.2f} "
                              f"2nd={float(top.values[1]):.2f} "
                              f"(damaged_k={int(bad_k.sum())})",
                              flush=True)
                    seg = dA[i0][bad_k]
                    print(f"[rank {rank}]   v11 dA-dist: "
                          f"std={float(seg.std()):.4f} "
                          f"max={float(seg.abs().max()):.4f} "
                          f"frac_abs_gt_0.1="
                          f"{float((seg.abs() > 0.1).float().mean()):.3f} "
                          f"(swapped-A-content dA std~0.028; garbage "
                          f"shows max>>0.1)", flush=True)
                # oldgen scan: rowscan+exact tests kill every candidate
                # within reach (cur rows, prev gen, sign, zero, garbage
                # bits) while dA stays A-scale -- the strongest living
                # hypothesis is content from MANY rounds ago (v10 only
                # kept 3 ref gens). Round data is seed-deterministic, so
                # regenerate refs for it-31..it-1 and exact-match the
                # WRONG elements of the first damaged rows in C-space.
                # A partial-overwrite tear lights up exactly one old gen
                # with a large element fraction.
                try:
                    wrows = dmg[:8]
                    wmask = wrong[wrows]              # [8, N] bool
                    nw = int(wmask.sum())
                    if nw > 0:
                        hits = []
                        for g in range(1, 32):
                            git = it - g
                            if git < 0:
                                break
                            Af = []
                            for s in range(size):
                                torch.manual_seed(5000 + 100 * git + s)
                                Af.append((torch.randn(M_per_rank, K,
                                                       device="cuda")
                                           * SCALE).half())
                            ref_g = torch.cat(Af, dim=0).float() \
                                @ b.float().t()
                            nh = int(((c.float()[wrows] - ref_g[wrows])
                                      .abs() < 1e-3)[wmask].sum())
                            hits.append((g, nh))
                            del Af, ref_g
                        best = max(hits, key=lambda x: x[1])
                        hot = [(g, nh) for g, nh in hits if nh > nw * 0.1]
                        print(f"[rank {rank}]   v11 oldgen: wrong_el={nw} "
                              f"best={best} hot(>10%)={hot or 'NONE'}",
                              flush=True)
                except Exception as exc:      # noqa: BLE001
                    print(f"[rank {rank}]   v11 oldgen FAILED: {exc}",
                          flush=True)
                # N-tile banding of the damage (pull/tile addressing
                # fingerprint): which 128-wide N tiles are hit
                try:
                    badc = wrong[dmg[:1].item()]
                    tiles = badc.view(-1, 128).any(dim=1)
                    tids = torch.nonzero(tiles).flatten().tolist()
                    print(f"[rank {rank}]   v11 ntiles: {tids}",
                          flush=True)
                except Exception as exc:      # noqa: BLE001
                    print(f"[rank {rank}]   v11 ntiles FAILED: {exc}",
                          flush=True)
                # persist a snapshot so offline forensics never needs a
                # rerun: damaged C rows, ref rows, A gens, the solve
                try:
                    torch.save(
                        {"it": it, "rank": rank,
                         "dmg_rows": dmg[:200].cpu(),
                         "c_rows": cf[dmg[:200]].clone().cpu(),
                         "ref_rows": ref[dmg[:200]].clone().cpu(),
                         "A_full": A_full.cpu(),
                         "prev_A": (_prev_A.cpu()
                                    if _prev_A is not None else None),
                         "dA_rows": dA[dmg[:200]].clone().cpu(),
                         "b": b.cpu()},
                        f"/tmp/agfail_it{it}_r{rank}.pt")
                    print(f"[rank {rank}]   v11 snapshot -> "
                          f"/tmp/agfail_it{it}_r{rank}.pt", flush=True)
                except Exception as exc:      # noqa: BLE001
                    print(f"[rank {rank}]   v11 snapshot FAILED: {exc}",
                          flush=True)
                mag = dA[i0].abs()
                thr = 0.25 * float(mag.max())
                ks = torch.nonzero(mag > thr).flatten()
                k0, k1 = int(ks[0]), int(ks[-1]) + 1
                da_seg = implied[k0:k1]
                cands = []
                if _prev_A is not None:
                    cands.append(("prevA", _prev_A.float()[i0, k0:k1]))
                for sh in (64, 128, 256):
                    for sgn in (1, -1):
                        lo, hi = k0 + sgn * sh, k1 + sgn * sh
                        if 0 <= lo and hi <= K_:
                            cands.append((f"shift{sgn*sh:+d}",
                                          A_full.float()[i0, lo:hi]))
                for nm, cv in cands:
                    if cv.shape == da_seg.shape:
                        nm_hit = int(((cv - da_seg).abs() < 1e-3).sum())
                        if nm_hit > 0:
                            print(f"[rank {rank}]   v11 implied-A matches "
                                  f"{nm}: {nm_hit}/{cv.numel()}", flush=True)
            else:
                print(f"[rank {rank}]   v11 dA: no meaningful rows",
                      flush=True)
        # forensics v4: what did the gate ACTUALLY see this round? Per
        # peer cell: (src, ck, min loop-exit v, min post-acquire v,
        # expected). exit < expected => while-loop exited wrongly
        # (codegen class); acq < exit => a LOWER ticket landed on the
        # cell after the fact (cell aliasing); both >= expected and torn
        # data => the acquire/release ordering itself is broken.
        if ictx.diag_buf is not None:
            db = ictx.diag_buf.cpu().tolist()
            cells = []
            for s in range(size):
                if s == rank:
                    continue
                for ck in range(ag.num_chunks):
                    cell = s * ag.num_chunks + ck
                    cells.append((s, ck, db[2 * cell], db[2 * cell + 1],
                                  ag.round_base + 1 + ck))
            print(f"[rank {rank}]   diag(src,ck,exit_v,acq_v,expected): "
                  f"{cells}", flush=True)
        # forensics v4b: pull epilogue state -- ready xchg happens BEFORE
        # the done-counter self-clean store, so pull_done!=0 at FAIL time
        # means we caught a pull mid-epilogue (should be unreachable).
        pd = ictx.pull_done.cpu().tolist()
        print(f"[rank {rank}]   pull_done={pd} "
              f"ready={ag.ready.cpu().tolist()}", flush=True)
        # forensics v3: snapshot MY ws -- torn ws => pull/source side;
        # clean ws + torn C => consumer read stale. runs = contiguity of
        # the ws mismatch (1 = isolated elements, ~16/32 = line/sector).
        ws_snap = torch.empty(M, K, dtype=DTYPE, device="cuda")
        _kpeek[(triton.cdiv(M * K, 4096),)](ws_snap.data_ptr(), ag.ws_ptr,
                                            M * K, BLOCK=4096)
        torch.cuda.synchronize()
        for s in range(size):
            if s == rank:
                continue
            rows = slice(s * M_per_rank, (s + 1) * M_per_rank)
            wbad = (ws_snap[rows] != A_full[rows]).reshape(-1)
            n_ws = int(wbad.sum())
            if n_ws == 0:
                print(f"[rank {rank}]   ws src={s}: CLEAN (consumer-side)",
                      flush=True)
                continue
            edges = (wbad[1:] != wbad[:-1]).sum()
            runs = int(edges) + int(bool(wbad[0]))
            mx = 0
            cur = 0
            wb = wbad.cpu()
            for x in wb.tolist():
                cur = cur + 1 if x else 0
                mx = max(mx, cur)
            mpw = "-"
            if _prev_A is not None:
                mpw = str(int(((ws_snap[rows] == _prev_A[rows])
                               .reshape(-1) & wbad).sum()))
            print(f"[rank {rank}]   ws src={s}: TORN n={n_ws} "
                  f"runs={runs} maxrun={mx} match_prev_ws={mpw}/{n_ws}",
                  flush=True)
        # forensics v13 (2026-10-14): B-operand test. v12's kernel-pull
        # events damage ONE N-tile across MULTIPLE M-tiles in BOTH bands
        # -- the only operand those CTAs share is that pid_n's b[128,K]
        # tile. Solve the shared corruption directly, fully overdetermined:
        #   e[:, tile] = A_full @ dB.T     (dB [128, K] vs M=4096 rows)
        # A real B-side corruption collapses the residual AND the cross-
        # band validation (dB from band0 predicts band1's errors); an
        # A-side or accumulate error cannot. Also prints dB's k-support
        # (one BLOCK_K=64 slice predicted for the small single-tile
        # events) and its value scale vs b.
        try:
            e_full = cf - ref                       # [M, N]
            Af = A_full.float()
            tids13 = (torch.nonzero(wrong.any(dim=0)).flatten()
                      // 128).unique().tolist()
            for t13 in tids13[:2]:
                sl13 = slice(t13 * 128, (t13 + 1) * 128)
                Et = e_full[:, sl13]
                wt = wrong[:, sl13]
                if int(wt.sum()) < 32:
                    continue
                sol = torch.linalg.lstsq(Af, Et).solution      # [K,128]=dB.T
                r = (Et - Af @ sol).abs()
                expl = 1.0 - float(r[wt].norm() / Et[wt].norm())
                half = M // 2
                w0, w1 = wt[:half], wt[half:]
                if int(w0.sum()) > 256 and int(w1.sum()) > 256:
                    s0 = torch.linalg.lstsq(Af[:half], Et[:half]).solution
                    r1 = (Et[half:] - Af[half:] @ s0).abs()
                    xval = (1.0 - float(r1[w1].norm()
                                        / Et[half:][w1].norm()))
                else:
                    xval = float("nan")
                ke = sol.abs().amax(dim=1)            # per-k dB energy
                hot13 = torch.nonzero(ke > 0.1 * float(ke.max())).flatten()
                kk = hot13.tolist()
                span = (kk[0] // 64, kk[-1] // 64,
                        len(kk)) if kk else (-1, -1, 0)
                print(f"[rank {rank}]   v13 B-side tile{t13}: "
                      f"wrong={int(wt.sum())} explained={expl:.4f} "
                      f"xband={xval:.4f} "
                      f"dB_k_support(slices {span[0]}..{span[1]}, "
                      f"n={span[2]}) dB_std={float(sol.std()):.5f} "
                      f"(b_std={float(b.float().std()):.5f})", flush=True)
        except Exception as exc:          # noqa: BLE001
            print(f"[rank {rank}]   v13 B-side FAILED: {exc}", flush=True)
    _prev_A = A_full
    _prev_ref.insert(0, ref)
    del _prev_ref[3:]
    return md, ms, peer_count


# warmup (compile + calibrate everything, discarded)
for it in range(3):
    one_round(it)
barrier()

fails = 0
worst = 0.0
peer_counts = []
for it in range(ROUNDS):
    if ALIGN:
        barrier()
    md, ms, pc = one_round(100 + it)
    if pc is not None:
        peer_counts.append(pc)
    worst = max(worst, md)
    bad = md >= 1e-2
    if bad:
        fails += 1
    elif it % 10 == 0:
        print(f"[rank {rank}] it={it} ok md={md:.5f} ms={ms:.3f} "
              f"peer_count={pc}", flush=True)

print(f"[rank {rank}] BISECT RESULT fails={fails}/{ROUNDS} worst={worst:.5f} "
      f"peer_counts min/med/max="
      f"{(min(peer_counts), sorted(peer_counts)[len(peer_counts)//2], max(peer_counts)) if peer_counts else None}",
      flush=True)

i1.finalize_i1_context(ictx)
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402
dcu_state()["host"].finalize()
barrier()
mpi.MPI_Finalize()
os._exit(1 if fails else 0)
