"""I1: table-driven allgather-GEMM consumer + arrival observer (L1 loop).

Builds the I1 predictor closed loop ON TOP of the accepted S14 AG-tr op
(gemm_ag_tr.py is imported, never modified):

  observer (1 CTA, own obs stream)  ->  obs D2H next round (1-round stale)
  ->  host StickyOrderPredictor (i1_predictor.py, plain python)
  ->  table write (1-CTA kernel, on-stream, ~us) into a small symmetric-
      heap slot  ->  table-driven consumer kernel (tl.load at the exact
      spot Syncopate does its wave-table reads)

Table plane (int32 words, in the DUSHMEM symmetric heap -- same heap the
k-chunk slot pool needs; format shared with the inference line's L0
offline replay, see docs/I1_PREDICTOR_L1_20261013.md):
  [0] seq         forward sequence number (monotonic; v1 consumer does
                  not read it -- reserved for multi-slot/versioning)
  [1] degraded    1 = predictor fell back to the sequential form
  [2] mode        0 = band-major own-first (S14 current behavior),
                  1 = source-major wave order (early source's ALL chunks
                  first -- the late source's delta hides under the early
                  source's compute instead of stalling waves mid-grid)
  [3] k_eff       effective chunk budget for this forward
  [4 .. 4+R)      order[R]: order[g] = source rank consumed at slot g
                  (at R=2 this is degenerate BY DESIGN: own source never
                  waits under the skip-local split, so the whole 2-card
                  signal rides on `mode`; order[] is the payload at R>=3)
  [4+R .. 4+2R)   budget[R]: per-source spin budget (v1: constant 128,
                  the S14 hybrid-spin batch; budget-skip semantics need
                  associativity and belong to the AR side / I3)

Transport choice (on-stream kernel write, NOT host put / H2D memcpy):
a cached 1-CTA kernel launch is ~us of host time, is stream-ordered
after the previous round's consumer for free (same-stream
serialization), and keeps clear of the DTK26.04 tiny-hipMemcpyAsync
wedge class (S15) that a 28 B engine copy into the heap would re-enter.

Straggler injection (L1 rig): stage_delay_us delays ONLY the calling
rank's stage stream (device _sleep before the stage copies), so the
consumer fork + pulls proceed on time and the delay appears as pure
relative arrival skew of this rank's chunks at every puller.
"""
import ctypes
import dataclasses
import os
from typing import List, Optional

import torch
import triton
import triton.language as tl

from triton_dist.kernels.dcu.common_ops import (
    dcu_state,
    hip,
    memcpy_async,
    set_signal,
)
from triton_dist.kernels.dcu.gemm_ar_rd import (
    MAX_TICKETS,
    _launch_flag_xchg,
    _wait_ticket,
)
from triton_dist.kernels.dcu.gemm_ag_tr import (
    _CONSUMER_BLOCK_K,
    _CONSUMER_BLOCK_M,
    _CONSUMER_BLOCK_N,
    _CONSUMER_GROUP_M,
    _CONSUMER_WARPS,
    GemmAGTRContext,
    create_gemm_ag_tr_context,
    finalize_gemm_ag_tr_context,
)

TABLE_HEADER_WORDS = 4  # seq, degraded, mode, k_eff

# Arrive fan-out mechanism (I1 finding, 2026-10-14): the S14 form -- a
# KERNEL remote atomic_xchg right after the stage engine copy -- lets the
# flag beat the copy's data into remote visibility when a puller's GTE
# wait actually BLOCKS and then chases the flag at hardware speed
# (lockstep hides this: the wait passes at host-enqueue time, with the
# enqueue latency as slack). Symptom: torn peer-band data at ~400us skew,
# both stock S14 and the pred variant, both wait modes. The engine form
# fans the arrive out as 4 B same-stream D2D engine copies (the proven
# SymmBarrier arrive pattern): posted writes of one SDMA queue stay
# ordered on the wire, so the flag CANNOT become visible before the data.
I1_ARRIVE_ENGINE = os.environ.get("I1_ARRIVE", "kernel") == "engine"

# Ready-flag mechanism: same race, pull side. Under skew the consumer is
# already resident and spinning when the blocked pull GTE releases, so
# [copy -> ready-flag KERNEL -> consumer loads] runs at hardware speed
# with zero slack -- and the kernel's launch gate can beat the copy's
# data drain. An engine write on the SAME lane stream stays inside one
# SDMA queue behind the copy's posted writes.
#
# v16 (2026-10-14, tests/dbg_pull_race.py): the race is PROVEN and it is
# not a drain-lag nuance -- for a CROSS-CARD engine copy, a kernel launch
# on the same stream can run with the copy's data ENTIRELY unlanded
# (rig16 arm K: 8/3000 rounds, consumer's damaged tile fit x=1.00 on
# every 64-wide k-line = pure pre-copy content; local copies never do
# this, rig15 0/8000). hipStreamWriteValue32 is rc=709-broken on this
# heap, so the engine form writes the ready cell as a 4 B engine COPY
# from a staged ticket word -- same SDMA queue as the data copy, posted
# writes ordered on the wire (the proven I1_ARRIVE=engine pattern).
# I1_READY=engine selects this wire-ordered form.
I1_READY_ENGINE = os.environ.get("I1_READY", "kernel") == "engine"

# Slack knobs (diagnosis + mitigation for the skew-torn-read race): the
# peer's SDMA stage copy can be "complete" per stream semantics while its
# writes are still draining L2->HBM; a puller whose blocked GTE releases
# reads at hardware speed microseconds later and catches the drain window
# (lockstep hides it: host enqueue latency is the slack). Pull slack
# delays MY engine read after the wait; arrive slack delays the SOURCE's
# flag after its copy. Both measured in us, 0 = off.
I1_PULL_SLACK_US = float(os.environ.get("I1_PULL_SLACK_US", "0"))
I1_ARRIVE_SLACK_US = float(os.environ.get("I1_ARRIVE_SLACK_US", "0"))

# Verify+repair (2026-10-14, the residual tear): the consumer-side read
# transient is NOT source-catchable -- at fail time MY ws is bit-correct
# (47/47 forensics), the harness ref matches CPU ground truth exactly,
# and every structural rig (flags/observer/symmetric load/per-chunk
# overlap/barrier-free protocol) is clean, so the wrong bytes exist only
# in the consumer's in-flight loads (~0.3-0.7%/rank-round, one BLOCK_K
# slice per damaged tile, content matching nothing known). I1_VERIFY=1
# (default) makes the op verify its own result: per output tile, SPOT
# elements are recomputed in fp32 from the known-good gathered ws + b
# and compared against c (tear sigma ~4e-3 = ~340 fp16 ulp; tol 1e-3
# splits rounding from damage; P(miss) ~0.1% at SPOT=4), flagged tiles
# are recomputed from the same correct memory. Costs a tiny event sync
# + ~sub-ms per round; 0 disables.
I1_VERIFY = os.environ.get("I1_VERIFY", "1") == "1"
I1_VERIFY_SPOT = int(os.environ.get("I1_VERIFY_SPOT", "4"))
I1_VERIFY_TOL = float(os.environ.get("I1_VERIFY_TOL", "1e-3"))
I1_VERIFY_LOG = os.environ.get("I1_VERIFY_LOG", "0") == "1"

# Pull mechanism (I1 fix, 2026-10-14): the engine form -- lane engine copy
# + separate ready-flag kernel -- tears under real arrival skew (both wait
# modes, both flag styles; hipStreamWriteValue32 engine ready-writes also
# fail with rc=709 on this DTK, and 4 B engine copies to the PEER heap
# wedge). The kernel form moves the WHOLE pull into ONE device kernel per
# (src, ck): slice-copy peer->my ws via plain loads (P6 probe: 2 MB p50
# 30 us ~ 69 GB/s, 0/30 fails at 400 us skew -- same cleanliness class as
# the fabric probes), then a done-counter whose LAST CTA raises MY ready
# flag with atomic_xchg. Data and flag are produced by the same kernel and
# chained release/acquire through the counter: no engine/kernel launch
# boundary exists anywhere between the peer's data and my consumer's gate.
I1_PULL_KERNEL = os.environ.get("I1_PULL", "engine") == "kernel"
# env-tunable for the grid-threshold sweep (probe A/B/D 2026-10-14:
# pull_grid=512 hangs, 128 passes with chunks/tiles exonerated)
_PULL_BLOCK = int(os.environ.get("I1_PULL_BLOCK", "4096"))  # int32 words/CTA

# Consumer launch cap (required by I1_PULL=kernel, 2026-10-14): the
# consumer kernel is persistent-form (`for tile_id in range(pid, total,
# NUM_SMS)`), but S14 launched it at grid=tiles which packs the whole
# machine with (spinning) consumer CTAs -- fine while pulls were SDMA
# engine copies needing no SMs, but a DEADLOCK once pulls became SM
# kernels (S12a: 100%/100% both cards, round 1, no progress). Capping the
# grid keeps the spinners bounded and leaves warp slots for the pull CTAs;
# tile math, table mapping and output are unchanged (pure scheduling).
_CONSUMER_GRID_CAP = int(os.environ.get("I1_CONSUMER_GRID", "128"))

# Stage mechanism (I1, 2026-10-14): S12 left a 2-5%/round residual of the
# SAME torn-peer-band signature with kernel pulls -- placing the residual
# race on the SOURCE side: [engine copy own->own ws] -> [remote arrive
# flag kernel] lets the flag beat the SDMA data's visibility (kernel
# launch completion is NOT an ordering point for a preceding engine copy
# on this DTK). Kernel form stages the chunk with plain SM stores; the
# following flag-kernel launch then follows a REAL completion boundary.
# With I1_STAGE=kernel + I1_PULL=kernel the entire gated data path is
# SM-kernel-mediated with explicit release chains -- no SDMA engine copy
# anywhere a flag can outrun.
I1_STAGE_KERNEL = os.environ.get("I1_STAGE", "engine") == "kernel"


@triton.jit(do_not_specialize=["stage_ptr", "round_base", "n_chunks"])
def kernel_i1_arrive_stage(stage_ptr, round_base, n_chunks):
    """1 CTA: write this round's ticket values ticket(ck)=round_base+ck+1
    into the device staging words the engine fan-out copies from."""
    p = stage_ptr.to(tl.pointer_type(tl.int32))
    for ck in range(n_chunks):
        tl.store(p + ck, round_base + ck + 1)


@triton.jit(do_not_specialize=["dst_ptr", "src_ptr", "arr_ptr", "rdy_ptr",
                               "done_ptr", "ticket", "n"])
def kernel_i1_pull(dst_ptr, src_ptr, arr_ptr, rdy_ptr, done_ptr, ticket, n,
                   BLOCK: tl.constexpr, GRID: tl.constexpr):
    """Device-side pull of one (src, ck) chunk. Each CTA first spins on MY
    LOCAL arrive cell (hybrid volatile batch + periodic atomic probe,
    acquire-confirmed on exit) -- NOT a stream GTE: hipStreamWaitValue32
    blocked-release proved to be the unverified-acquire link in the skew
    tear chain (full-grid engine pulls ~27-30%/round torn through MANY
    consumer/flag mechanisms; probes without the op's concurrency were
    clean, and an in-consumer acquire changed nothing). Then the CTA
    copies its slice peer-ws -> my-ws with plain loads; the LAST CTA
    (done-counter) raises MY ready flag with atomic_xchg and self-cleans
    the counter. Stores -> atomic_add -> atomic_xchg form an explicit
    release chain, so the consumer's acquire probe on the ready cell can
    never see the flag ahead of the data."""
    pid = tl.program_id(0)
    ap = arr_ptr.to(tl.pointer_type(tl.int32))
    sp = src_ptr.to(tl.pointer_type(tl.int32))
    dp = dst_ptr.to(tl.pointer_type(tl.int32))
    rp = rdy_ptr.to(tl.pointer_type(tl.int32))
    cp = done_ptr.to(tl.pointer_type(tl.int32))
    # scope="sys" on the arrive spin (2026-10-14): the cell is written
    # REMOTELY (peer's stage-stream kernel_flag_xchg) and what the acquire
    # must order is the SUBSEQUENT REMOTE plain loads of peer-ws. Triton
    # atomics default to device (gpu) scope -- a gpu-scoped release on the
    # peer does not flush its ws stores across the fabric, so the big-shape
    # L1 matrix still tore ~2%/rank-round (md 0.10-0.13, spread over all
    # deltas incl. 0) after the same-device barrier fix; the HSA pairing
    # rule needs BOTH legs at system scope for cross-device message
    # passing.
    v = tl.atomic_add(ap, 0, scope="sys")
    while v < ticket:
        for _ in range(32):
            v = tl.load(ap, volatile=True)
        if v < ticket:
            v = tl.atomic_add(ap, 0, scope="sys")
    v = tl.atomic_add(ap, 0, scope="sys")
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    # PLAIN loads, never volatile: S15d/e A/B (2026-10-14) -- with
    # volatile (GLC) loads the consumer still read stale DRAM-side lines
    # of my-ws stores (4 tears / 4800 rank-rounds, ws forensically CLEAN
    # = consumer-side); plain loads = the coherent path, 0/4800. Matches
    # the S3 platform rule (volatile/.cg read DRAM, which lags L2).
    v = tl.load(sp + offs, mask=m)
    tl.store(dp + offs, v, mask=m)
    # CTA barrier BEFORE the done-counter (gate-soak 2026-10-14: 6/600
    # tears with the dispatch-order gate active, ws forensically CLEAN,
    # torn lines SPARSE and 37% matching the previous round -- a triton
    # scalar atomic is executed by ONE elected thread, so the OTHER
    # threads' data stores are not ordered before it by the release
    # alone; before the gate, launch latency sheltered this window
    # {0/4800}, the gate removed the slack and exposed it).
    tl.debug_barrier()
    prev = tl.atomic_add(cp, 1, sem="acq_rel")
    if prev == GRID - 1:
        tl.atomic_xchg(rp, ticket, sem="release")
        tl.store(cp, 0)  # self-clean for the next round


@triton.jit(do_not_specialize=["dst_ptr", "src_ptr", "n"])
def kernel_i1_copy(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    """Plain SM-store copy (int32 words), used for the I1_STAGE=kernel
    staging path: kernel completion is a real ordering point for the
    arrive flag kernels that follow on the same stream."""
    pid = tl.program_id(0)
    dp = dst_ptr.to(tl.pointer_type(tl.int32))
    sp = src_ptr.to(tl.pointer_type(tl.int32))
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(dp + offs, tl.load(sp + offs, mask=m), mask=m)


@triton.jit(do_not_specialize=["ctl_ptr", "seq", "deg", "mode", "k_eff",
                               "order_packed"])
def kernel_i1_table_write(ctl_ptr, seq, deg, mode, k_eff, order_packed,
                          R: tl.constexpr, BUDGET: tl.constexpr):
    """1 CTA: write this forward's table words into the symmetric-heap
    slot (order packed 4 bits/slot, R <= 16)."""
    p = ctl_ptr.to(tl.pointer_type(tl.int32))
    tl.store(p + 0, seq)
    tl.store(p + 1, deg)
    tl.store(p + 2, mode)
    tl.store(p + 3, k_eff)
    for i in tl.static_range(R):
        tl.store(p + 4 + i, ((order_packed >> (4 * i)) & 15).to(tl.int32))
        tl.store(p + 4 + R + i, BUDGET)


@triton.jit(do_not_specialize=["dst", "ctl_ptr"])
def kernel_i1_peek(dst, ctl_ptr, NWORDS: tl.constexpr):
    """Test helper: copy the table slot out to a readable device tensor."""
    p = ctl_ptr.to(tl.pointer_type(tl.int32))
    d = dst.to(tl.pointer_type(tl.int32))
    for i in tl.static_range(NWORDS):
        tl.store(d + i, tl.load(p + i))


@triton.jit(do_not_specialize=["ctl_ptr", "flags_base", "round", "rank",
                               "world_size", "m_chunk_num"])
def kernel_gemm_ag_tr_consumer_pred(
        A,  # gathered workspace [max_M, K] (int64 VA, symmetric heap)
        localA,  # local A shard [M_per_rank, K] tensor
        B,  # [N_per_rank, K] tensor
        C,  # [M, N_per_rank] tensor
        M,
        N,
        K,
        stride_k,  # single A row stride: ws and shard are both [*, K] contig
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        rank,
        world_size,
        m_chunk_num,
        flags_base,  # LOCAL int32 [num_ranks * m_chunk_num] ready tickets
        ctl_ptr,  # symmetric-heap int32 table slot (this file's format)
        round,  # this op's round_base: expected = round + ck + 1
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        M_PER_CHUNK: tl.constexpr,
        NUM_SMS: tl.constexpr,  # == launched grid (loop step)
        EVEN_K: tl.constexpr,
        diag_ptr,  # I1_CONSUMER_DIAG=1: int32 [2*R*nc] gate forensics
        DIAG: tl.constexpr,
):
    """S14 kernel_gemm_ag_tr_consumer with the rank swizzle replaced by a
    table lookup (mode/order/budget consumed exactly where Syncopate
    reads its wave tables). Tile math, skip-local split, hybrid spin and
    the K loop are verbatim S14 -- the produced C is identical in every
    mode (mapping only permutes which CTA computes which tile)."""
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_k > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    M_per_rank = M // world_size
    pid_m_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    acc_dtype = tl.float32
    A_ptr = A.to(tl.pointer_type(localA.dtype.element_ty))
    flag_ptr = flags_base.to(tl.pointer_type(tl.int32))
    ctl32 = ctl_ptr.to(tl.pointer_type(tl.int32))

    for tile_id in range(pid, total_tiles, NUM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        # ---- table-driven mapping (I1) --------------------------------
        # mode 0: S14's band-major own-first rotation (order[] carries
        #         the rotation content -> identical behavior);
        # mode 1: source-major transposition of the (band, source) tile
        #         order: the predicted-early source's ALL chunk bands
        #         before any band of the late source.
        tpc = M_PER_CHUNK // BLOCK_SIZE_M
        mode = tl.load(ctl32 + 2)
        if mode == 1:
            src_pos = pid_m // (tpc * m_chunk_num)
            band = (pid_m // tpc) % m_chunk_num
        else:
            band = pid_m // (tpc * world_size)
            src_pos = (pid_m // tpc) % world_size
        block_offset = pid_m % tpc
        src = tl.load(ctl32 + 4 + src_pos)
        pid_m = (src * M_per_rank + band * M_PER_CHUNK
                 + block_offset * BLOCK_SIZE_M) // BLOCK_SIZE_M
        # ----------------------------------------------------------------

        offs_am = pid_m * BLOCK_SIZE_M
        # global chunk id == source_rank * m_chunk_num + chunk: exactly
        # the slot my pull lane flags after the data landed (LOCAL memory)
        offs_sig = offs_am // M_PER_CHUNK
        offs_rank = pid_m // pid_m_per_rank

        if offs_rank != rank:
            # per-chunk ticket + hybrid spin verbatim S14, with the
            # volatile batch size taken from the budget table (v1: the
            # writer kernel stores the constant 128 -> bit-identical
            # behavior to S14)
            nb = tl.load(ctl32 + 4 + world_size + offs_rank)
            expected = round + 1 + offs_sig % m_chunk_num
            fp = flag_ptr + offs_sig
            # I1 fix (2026-10-14): every exit path must be an ACQUIRE.
            # S14's form could leave the loop straight off the volatile
            # batch -- a volatile load observes the flag WITHOUT ordering
            # the subsequent data loads after the producer's release, so
            # under real skew (flag lands while this CTA spins tight) the
            # tile read torn ws data (~25%/round at delta=400us stock,
            # 1-5% after the mechanism changes, lockstep always clean
            # because enqueue latency is the slack). The confirming
            # atomic_add(fp, 0) AFTER the loop makes the exit an acquire;
            # monotone tickets make it trivially pass. (Form note: putting
            # this atomic unconditionally INSIDE the while body hangs the
            # hcu backend's codegen -- same wait-state traffic as the
            # guarded form, so miscompile-class, not semantics. Keep the
            # loop body verbatim S14 and confirm once on the way out.)
            v = tl.atomic_add(fp, 0)
            while v < expected:
                for _ in range(nb):
                    v = tl.load(fp, volatile=True)
                if v < expected:
                    v = tl.atomic_add(fp, 0)
            # DIAG (dbg builds only): the v the loop EXITED on. If this is
            # ever < expected the while-loop itself exited wrongly (codegen
            # class); if it's >= expected while a LATER acquire reads a
            # LOWER value, the cell got a lower ticket written after the
            # fact (cell aliasing -- a different producer xchg'ing an old
            # ticket onto this cell).
            if DIAG:
                tl.atomic_min(diag_ptr.to(tl.pointer_type(tl.int32))
                              + offs_sig * 2, v)
            # acquire-confirmed exit + CTA barrier: the confirming atomic
            # orders only the elected thread's subsequent loads; the
            # tile's data loads happen in ALL threads, so a barrier must
            # sit between gate-exit and the first A-tile load (same class
            # as the pull kernel's pre-counter barrier; gate-soak
            # 2026-10-14)
            v = tl.atomic_add(fp, 0, sem="acquire")
            if DIAG:
                tl.atomic_min(diag_ptr.to(tl.pointer_type(tl.int32))
                              + offs_sig * 2 + 1, v)
            tl.debug_barrier()

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        # S19c restructure: dual source as a SCALAR-ONLY branch (base
        # pointer + row bias) so the k-loop compiles ONE addressing form
        if offs_rank == rank:
            a_base = localA
            row_bias = rank * M_per_rank  # shard row = rm - bias
        else:
            a_base = A_ptr
            row_bias = 0
        A_BASE = a_base + (rm - row_bias)[:, None] * stride_k + rk[None, :]

        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc = tl.dot(a, b, acc)
            A_BASE += BLOCK_SIZE_K
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = (a_base + (rm - row_bias)[:, None] * stride_k
                      + rk[None, :])
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.dtype.element_ty)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


@triton.jit(do_not_specialize=["ready_ptr", "out_ptr", "round", "rank",
                               "n_chunks_max", "n_chunks"])
def kernel_i1_observer(ready_ptr, out_ptr, round, rank, n_chunks_max,
                       n_chunks, R: tl.constexpr):
    """1 CTA on the observer stream: round-robin-poll MY ready plane and
    record the peer sources' completion ORDER + per-source probe counts
    (the count is the crude lateness clock at R=2; at R>=3 the order
    itself is the signal). Poll is the S3 hybrid form -- volatile sweep
    as delay + atomic sweep only to CONFIRM a volatile all-ready (a pure
    atomic RMW storm both crawls and contends with the consumer's own
    probes on the same lines). Terminates when every peer source's
    chunks of THIS round (tickets round+1+ck) are visible, or at
    PROBE_CAP (marker = -round, host treats as no-observation). Output:
      out[0] = round marker; out[1+i] = i-th finished source;
      out[R+1+s] = probe count for source s (-1 if never)."""
    PROBE_CAP = 1 << 26
    rp = ready_ptr.to(tl.pointer_type(tl.int32))
    op = out_ptr.to(tl.pointer_type(tl.int32))
    peers = ((1 << R) - 1) ^ (1 << rank)
    fin = 0
    order_idx = 0
    probes = 0
    capped = 0
    # NB: no `return` inside the loop (this triton build forbids it) --
    # cap forces the loop condition false and the marker goes negative
    while (fin != peers) & (capped == 0):
        probes += 1
        if probes > PROBE_CAP:
            capped = 1
            fin = peers
        if capped == 0:
            for s in tl.static_range(R):
                if (s != rank) and (((fin >> s) & 1) == 0):
                    base = rp + s * n_chunks_max
                    ok = 1
                    for ck in range(n_chunks):
                        v = tl.load(base + ck, volatile=True)
                        ok = ok & (v >= round + 1 + ck)
                    # volatile alone can read stale DRAM forever (S3/P8.2):
                    # every 16th pass take the atomic truth probe instead --
                    # 1/16 the RMW traffic of the pure-atomic form (which
                    # both crawled and fought the consumer's probes here)
                    periodic = (probes % 16) == 0
                    if ok == 0:
                        if periodic != 0:
                            ok = 1
                            for ck in range(n_chunks):
                                v = tl.atomic_add(base + ck, 0)
                                ok = ok & (v >= round + 1 + ck)
                    if ok != 0:
                        fin = fin | (1 << s)
                        tl.store(op + 1 + order_idx, s)
                        order_idx += 1
                        tl.store(op + R + 1 + s, probes)
                    else:
                        for _ in range(64):
                            v = tl.load(base, volatile=True)
    if capped != 0:
        tl.store(op + 0, -round)
    else:
        tl.store(op + 0, round)


@triton.jit(do_not_specialize=["c_ptr", "a_ptr", "b_ptr", "bad_ptr",
                               "M", "N", "K", "sc0", "sc1", "sbk",
                               "num_n", "tol"])
def kernel_i1_verify(c_ptr, a_ptr, b_ptr, bad_ptr, M, N, K,
                     sc0, sc1, sbk, num_n, tol,
                     SPOT: tl.constexpr, BM: tl.constexpr,
                     BN: tl.constexpr):
    """Per-tile spot verify: recompute SPOT elements of this output tile
    in fp32 from the gathered ws (post-round = the known-good memory;
    the transient lives only in the consumer's in-flight loads) and b,
    compare against c, flag the tile 1/0 (unconditional xchg -> no
    pre-zeroing). One CTA per tile."""
    pid = tl.program_id(0)
    pid_m = pid // num_n
    pid_n = pid % num_n
    cp = c_ptr.to(tl.pointer_type(tl.float16))
    ap = a_ptr.to(tl.pointer_type(tl.float16))
    bp = b_ptr.to(tl.pointer_type(tl.float16))
    xp = bad_ptr.to(tl.pointer_type(tl.int32))
    off_r = (17, 53, 91, 125)
    off_c = (29, 97, 61, 5)
    bad = 0
    for s in tl.static_range(SPOT):
        row = pid_m * BM + off_r[s]
        col = pid_n * BN + off_c[s]
        if (row < M) & (col < N):
            acc = 0.0
            for k0 in range(0, K, 128):
                offs = k0 + tl.arange(0, 128)
                m = offs < K
                av = tl.load(ap + row * K + offs, mask=m, other=0.0)
                bv = tl.load(bp + col * sbk + offs, mask=m, other=0.0)
                acc += tl.sum(av.to(tl.float32) * bv.to(tl.float32))
            cv = tl.load(cp + row * sc0 + col * sc1).to(tl.float32)
            if tl.abs(acc - cv) > tol:
                bad = 1
    tl.atomic_xchg(xp + pid, bad)


@triton.jit(do_not_specialize=["c_ptr", "a_ptr", "b_ptr", "list_ptr",
                               "num_n", "K", "sc0", "sc1", "sbk"])
def kernel_i1_tile_repair(c_ptr, a_ptr, b_ptr, list_ptr, num_n, K,
                          sc0, sc1, sbk,
                          BM: tl.constexpr, BN: tl.constexpr,
                          BK: tl.constexpr):
    """Recompute flagged output tiles from the known-good gathered ws +
    b (one CTA per bad tile, rare path). Whatever the transient was, the
    memory is correct, so a redo is correct."""
    pid = tl.program_id(0)
    lp = list_ptr.to(tl.pointer_type(tl.int32))
    cp = c_ptr.to(tl.pointer_type(tl.float16))
    ap = a_ptr.to(tl.pointer_type(tl.float16))
    bp = b_ptr.to(tl.pointer_type(tl.float16))
    t = tl.load(lp + pid)
    pid_m = t // num_n
    pid_n = t % num_n
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        a_t = tl.load(ap + offs_m[:, None] * K + offs_k[None, :])
        b_t = tl.load(bp + offs_n[None, :] * sbk + offs_k[:, None])
        acc = tl.dot(a_t, b_t, acc)
    tl.store(cp + offs_m[:, None] * sc0 + offs_n[None, :] * sc1,
             acc.to(tl.float16))


@dataclasses.dataclass
class I1PredContext:
    ag: GemmAGTRContext
    ctl_ptr: int  # symmetric int32 [4 + 2R] table slot (MY heap view)
    table_words: int
    budget: int
    obs: torch.Tensor  # LOCAL int32 [2R + 1] observer output
    obs_stream: torch.cuda.Stream
    events: Optional[list] = None  # [fork, join_stage, joins..., join_obs]
    cycles_per_us: float = 0.0  # _sleep calibration (bench sets it)
    arrive_stage: Optional[torch.Tensor] = None  # int32 [num_chunks] tickets
    ready_stage: Optional[torch.Tensor] = None  # int32 [num_chunks] tickets
    pull_done: Optional[torch.Tensor] = None  # int32 [R * num_chunks]
    # I1_CONSUMER_DIAG=1: int32 [2*R*nc], per cell the min loop-exit v and
    # the min post-acquire v seen at gate pass (reset by the dbg harness
    # each round; production leaves this None -> DIAG=False, dead code)
    diag_buf: Optional[torch.Tensor] = None
    _writer_ck: object = None
    _observer_ck: object = None
    _arrive_ck: object = None
    _rstage_ck: object = None
    _pull_ck: object = None  # ((words, BLOCK, GRID), cached kernel)
    _copy_ck: object = None  # ((words, BLOCK), cached kernel)
    verify_bad: Optional[torch.Tensor] = None  # int32 [n_tiles] flags
    _verify_ck: object = None
    _repair_ck: object = None
    verify_hits: int = 0  # tiles repaired over the context's lifetime


def create_i1_context(max_M, K, input_dtype, output_dtype, rank, num_ranks,
                      mpi, comm, chunk_rows=0, comm_lanes=2, wait_mode=None,
                      budget=128):
    """S14 AG-tr context + the I1 table slot, observer plane and writer/
    observer kernels. mpi/comm from pydushmem.host.mpi_init() (BEFORE
    `import torch`)."""
    ag = create_gemm_ag_tr_context(
        max_M=max_M, K=K, input_dtype=input_dtype,
        output_dtype=output_dtype, rank=rank, num_ranks=num_ranks,
        mpi=mpi, comm=comm, chunk_rows=chunk_rows, comm_lanes=comm_lanes,
        wait_mode=wait_mode)

    s = dcu_state()
    words = TABLE_HEADER_WORDS + 2 * num_ranks
    ctl_ptr = s["host"].malloc(4 * words)
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(ctl_ptr), 0, 4 * words) == 0

    obs = torch.full((2 * num_ranks + 1,), -1, dtype=torch.int32,
                     device="cuda")
    arrive_stage = torch.zeros(ag.num_chunks, dtype=torch.int32,
                               device="cuda")
    ready_stage = torch.zeros(ag.num_chunks, dtype=torch.int32,
                              device="cuda")
    pull_done = torch.zeros(num_ranks * ag.num_chunks, dtype=torch.int32,
                            device="cuda")

    h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    h.hipEventCreateWithFlags.restype = ctypes.c_int
    h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipEventRecord.restype = ctypes.c_int
    h.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_uint64]
    h.hipStreamWaitEvent.restype = ctypes.c_int
    L = ag.comm_lanes
    events = []
    # +1: ev_arrive (the pull-dispatch gate; see the pull section). The
    # join loop indexes events[1+si] over the L+3 op streams, so the
    # appended event is only ever touched by the pull section.
    for _ in range(1 + 2 + L + 1 + 1):  # fork + stage/consumer/obs + lanes
        ev = ctypes.c_void_p()
        rc = h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2)
        assert rc == 0, f"hipEventCreateWithFlags rc={rc}"
        events.append(ev)

    diag_buf = None
    if os.environ.get("I1_CONSUMER_DIAG", "") == "1":
        diag_buf = torch.full((num_ranks * ag.num_chunks * 2,),
                              1 << 30, dtype=torch.int32, device="cuda")
    ictx = I1PredContext(
        ag=ag, ctl_ptr=ctl_ptr, table_words=words, budget=budget, obs=obs,
        obs_stream=torch.cuda.Stream(), events=events,
        arrive_stage=arrive_stage, ready_stage=ready_stage,
        pull_done=pull_done, diag_buf=diag_buf)
    # initial table = S14 behavior (mode 0, own-first rotation, seq 0) so
    # any read before the first predictor write is safe
    rot = sum(int((rank + i) % num_ranks) << (4 * i)
              for i in range(num_ranks))
    i1_table_write(ictx, 0, 0, 0, rot, ag.num_chunks)
    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ictx


def finalize_i1_context(ictx):
    s = dcu_state()
    if ictx.ctl_ptr:
        s["host"].free(ictx.ctl_ptr)
    ictx.ctl_ptr = 0
    finalize_gemm_ag_tr_context(ictx.ag)


def i1_table_write(ictx, seq, deg, mode, order_packed, k_eff):
    """Cached 1-CTA writer-kernel launch on the CALLER's current stream
    (stream-ordered after the previous round's consumer join -- single
    slot is safe, no ring needed)."""
    ck = ictx._writer_ck
    if ck is None:
        # int64-VA warmup trick (heap VAs exceed int32; see
        # _launch_flag_xchg) -- order_packed warmed > 2^31 as well
        ck = kernel_i1_table_write.warmup(
            1 << 40, 1, 1, 1, 1, 1 << 40, R=ictx.ag.num_ranks,
            BUDGET=ictx.budget, grid=(1,), num_warps=1)
        ck._init_handles()
        ictx._writer_ck = ck
    # this triton build's cached launch takes the constexprs too (the
    # SymmBarrier ck-launch form)
    ck[(1, 1, 1)](ictx.ctl_ptr, seq, deg, mode, k_eff, order_packed,
                  ictx.ag.num_ranks, ictx.budget)


def i1_observer_read(ictx):
    """Host-side read of the observer plane AFTER this rank's round is
    device-complete (the caller syncs the round event first). Returns
    [round_marker, order..., counts...]."""
    return ictx.obs.cpu().tolist()


def i1_ag_tr_op(a, b, c, ictx, stage_delay_us=0.0, with_observer=True):
    """I1 pipeline = S14 gemm_ag_tr_op with (a) the table-driven consumer
    kernel, (b) the observer launch on its own stream, (c) optional
    straggler injection: stage_delay_us > 0 sleeps ONLY the calling
    rank's stage stream, turning the delay into pure arrival skew of
    this rank's chunks at every puller (the L1 rig)."""
    ag = ictx.ag
    m_pr, K = a.shape
    N_pr = b.shape[0]
    rank, R = ag.rank, ag.num_ranks
    CR = min(ag.chunk_rows, m_pr)
    L = ag.comm_lanes
    M = m_pr * R
    assert K == ag.K and M == ag.max_M, "v1: full-M rounds only"
    assert a.dtype == ag.input_dtype and b.dtype == ag.input_dtype
    assert a.stride(1) == 1 and b.stride(1) == 1 and a.stride(0) == K
    assert b.stride(0) == K
    assert M % _CONSUMER_BLOCK_M == 0 and N_pr % _CONSUMER_BLOCK_N == 0
    assert m_pr % CR == 0

    cur = torch.cuda.current_stream()
    h = hip()
    n_chunks = m_pr // CR
    assert n_chunks == ag.num_chunks  # full-M (flag/ticket layout)
    mode = ag.wait_mode

    ag.round_base += ag.round_stride
    assert ag.round_base + n_chunks < MAX_TICKETS

    # I1_READY=engine: stage this round's ready tickets ONCE on the
    # caller's stream -- the fork event below orders every lane behind
    # it, so each 4 B engine copy below reads a settled staging word
    # (v16: the ready cell must ride the SAME SDMA queue as the data
    # copy, as a posted 4 B copy -- kernel flags can beat cross-card
    # engine data entirely; hipStreamWriteValue32 is rc=709-broken).
    if R > 1 and I1_READY_ENGINE and not I1_PULL_KERNEL:
        ck_rs = ictx._rstage_ck
        if ck_rs is None:
            ck_rs = kernel_i1_arrive_stage.warmup(
                1 << 40, 1, 1, grid=(1,), num_warps=1)
            ck_rs._init_handles()
            ictx._rstage_ck = ck_rs
        ck_rs[(1, 1, 1)](ictx.ready_stage.data_ptr(),
                         ag.round_base, n_chunks)

    def ticket(ck):
        return ag.round_base + ck + 1

    # fork: stage + lanes + consumer + observer order after the caller
    # (a/b/c ready -- and the table write enqueued before this call)
    all_streams = ([ag.stage_stream] + ag.lane_streams
                   + [ag.consumer_stream, ictx.obs_stream])
    ev_fork = ictx.events[0]
    assert h.hipEventRecord(ev_fork, ctypes.c_void_p(cur.cuda_stream)) == 0
    for st in all_streams:
        assert h.hipStreamWaitEvent(ctypes.c_void_p(st.cuda_stream),
                                    ev_fork, 0) == 0

    my_off = rank * m_pr * K * ag.elemsz

    if R > 1:
        # stage: chunk-major, own->own engine copies into MY ws shard
        # region, each chunk's ARRIVE fanned out as 4 B kernel writes to
        # every peer. Round-granularity slot guard via the ack plane.
        with torch.cuda.stream(ag.stage_stream):
            if stage_delay_us > 0.0:
                assert ictx.cycles_per_us > 0, "calibrate_sleep first"
                torch.cuda._sleep(int(stage_delay_us * ictx.cycles_per_us))
            for d in range(R):
                if d != rank:
                    _wait_ticket(ag.ack_ptr + d * 4,
                                 ag.last_done_ticket,
                                 ag.stage_stream, mode)
            if I1_STAGE_KERNEL:
                st_words = (CR * K * ag.elemsz) // 4
                assert (CR * K * ag.elemsz) % 4 == 0, "stage needs 4B-align"
                st_grid = triton.cdiv(st_words, _PULL_BLOCK)
                skey = (st_words, _PULL_BLOCK, st_grid)
                ck_cp = ictx._copy_ck
                if ck_cp is None or ck_cp[0] != skey:
                    kc = kernel_i1_copy.warmup(
                        1 << 40, 1 << 40, 1, BLOCK=_PULL_BLOCK, grid=(1,),
                        num_warps=4)
                    kc._init_handles()
                    ictx._copy_ck = (skey, kc)
                    ck_cp = ictx._copy_ck
            for ck in range(n_chunks):
                if I1_ARRIVE_SLACK_US > 0:
                    torch.cuda._sleep(int(I1_ARRIVE_SLACK_US
                                           * ictx.cycles_per_us))
                if I1_STAGE_KERNEL:
                    ck_cp[1][(st_grid, 1, 1)](
                        ag.ws_ptr + my_off + ck * CR * K * ag.elemsz,
                        a.data_ptr() + ck * CR * K * ag.elemsz,
                        st_words, _PULL_BLOCK)
                else:
                    memcpy_async(ag.ws_ptr + my_off + ck * CR * K * ag.elemsz,
                                 a.data_ptr() + ck * CR * K * ag.elemsz,
                                 CR * K * ag.elemsz,
                                 ag.stage_stream.cuda_stream)
                if I1_ARRIVE_ENGINE:
                    # stage this round's tickets once, then 4 B engine
                    # copies per (peer, chunk): same SDMA queue as the
                    # data copy -> posted-write order on the wire, the
                    # flag cannot become remotely visible before the data
                    if ck == 0:
                        ck_st = ictx._arrive_ck
                        if ck_st is None:
                            ck_st = kernel_i1_arrive_stage.warmup(
                                1 << 40, 1, 1, grid=(1,), num_warps=1)
                            ck_st._init_handles()
                            ictx._arrive_ck = ck_st
                        ck_st[(1, 1, 1)](ictx.arrive_stage.data_ptr(),
                                         ag.round_base, n_chunks)
                    for d in range(R):
                        if d != rank:
                            memcpy_async(
                                ag.peer_arrive[d]
                                + (rank * ag.num_chunks + ck) * 4,
                                ictx.arrive_stage.data_ptr() + ck * 4,
                                4, ag.stage_stream.cuda_stream)
                else:
                    for d in range(R):
                        if d != rank:
                            _launch_flag_xchg(
                                ag.peer_arrive[d]
                                + (rank * ag.num_chunks + ck) * 4,
                                ticket(ck))

        # pulls: chunk-major, offset-preserving. Kernel form: ONE kernel
        # per (src, ck) does copy + done-counter + ready-flag (see
        # kernel_i1_pull / I1_PULL_KERNEL note); engine form: lane engine
        # copy with the ready flag riding the lane stream right after the
        # data (the known-corrupting-under-skew baseline, kept for A/B).
        if I1_PULL_KERNEL:
            # dispatch-order gate (probe5, 2026-10-14): two lane streams
            # launching BIG spinning pull kernels concurrently can fill
            # every SM slot before the peer's 1-CTA arrive-xchg kernel
            # gets one -- mutual starvation across ranks (my xchg
            # releases the peer's pulls, its xchg releases mine; measured
            # hang at chunk >= 4.5MB, pass at 2MB, pass with lanes=1).
            # Gating the lanes on an event recorded AFTER the stage+
            # arrive block retires makes the xchg dispatch into a machine
            # that is not yet full of spinners -- deterministically, on
            # both ranks. The arrive-after-data release semantics are
            # unchanged (same stream ordering as before); only dispatch
            # order is pinned.
            ev_arr = ictx.events[-1]
            assert h.hipEventRecord(
                ev_arr, ctypes.c_void_p(ag.stage_stream.cuda_stream)) == 0
            for st in ag.lane_streams:
                assert h.hipStreamWaitEvent(
                    ctypes.c_void_p(st.cuda_stream), ev_arr, 0) == 0
            pull_words = (CR * K * ag.elemsz) // 4
            assert (CR * K * ag.elemsz) % 4 == 0, "pull needs 4B-aligned"
            pull_grid = triton.cdiv(pull_words, _PULL_BLOCK)
            key = (pull_words, _PULL_BLOCK, pull_grid)
            ck_pull = ictx._pull_ck
            if ck_pull is None or ck_pull[0] != key:
                kp = kernel_i1_pull.warmup(
                    1 << 40, 1 << 40, 1 << 40, 1 << 40, 1 << 40, 1, 1,
                    BLOCK=_PULL_BLOCK, GRID=pull_grid, grid=(1,),
                    num_warps=4)
                kp._init_handles()
                ictx._pull_ck = (key, kp)
                ck_pull = ictx._pull_ck
            for ck in range(n_chunks):
                for step in range(1, R):
                    src = (rank + step) % R
                    lane = ag.lane_streams[(src + ck) % L]
                    src_off = (src * m_pr + ck * CR) * K * ag.elemsz
                    cell = (src * ag.num_chunks + ck)
                    with torch.cuda.stream(lane):
                        # no GTE: the kernel spins the LOCAL arrive cell
                        # itself (in-kernel acquire; see kernel docstring)
                        ck_pull[1][(pull_grid, 1, 1)](
                            ag.ws_ptr + src_off,
                            ag.peer_ws[src] + src_off,
                            ag.arrive_ptr + cell * 4,
                            ag.ready.data_ptr() + cell * 4,
                            ictx.pull_done.data_ptr() + cell * 4,
                            ticket(ck), pull_words,
                            _PULL_BLOCK, pull_grid)
        if not I1_PULL_KERNEL:
            for ck in range(n_chunks):
                for step in range(1, R):
                    src = (rank + step) % R
                    lane = ag.lane_streams[(src + ck) % L]
                    src_off = (src * m_pr + ck * CR) * K * ag.elemsz
                    with torch.cuda.stream(lane):
                        _wait_ticket(
                            ag.arrive_ptr + (src * ag.num_chunks + ck) * 4,
                            ticket(ck), lane, mode)
                        if I1_PULL_SLACK_US > 0:
                            torch.cuda._sleep(int(I1_PULL_SLACK_US
                                                   * ictx.cycles_per_us))
                        memcpy_async(ag.ws_ptr + src_off,
                                     ag.peer_ws[src] + src_off,
                                     CR * K * ag.elemsz, lane.cuda_stream)
                        # local ready rides the lane stream right after the
                        # data: engine write by default (one SDMA queue with
                        # the copy -> posted-write order; the kernel form's
                        # launch gate can beat the copy's data drain when a
                        # blocked GTE releases straight into it under skew)
                        if I1_READY_ENGINE:
                            # 4 B engine COPY from the staged ticket word
                            # (hipStreamWriteValue32 is rc=709-broken on
                            # this heap; a posted copy is wire-ordered
                            # behind the data copy in the same queue)
                            memcpy_async(
                                ag.ready.data_ptr()
                                + (src * ag.num_chunks + ck) * 4,
                                ictx.ready_stage.data_ptr() + ck * 4,
                                4, lane.cuda_stream)
                        else:
                            _launch_flag_xchg(
                                ag.ready.data_ptr()
                                + (src * ag.num_chunks + ck) * 4,
                                ticket(ck))

    # observer: 1 CTA on its own stream, forked with everything else
    # (with_observer=False is the bisect knob: same stream topology
    # otherwise, so A/B timings stay comparable)
    if with_observer:
        with torch.cuda.stream(ictx.obs_stream):
            ck_ob = ictx._observer_ck
            if ck_ob is None:
                ck_ob = kernel_i1_observer.warmup(
                    1 << 40, 1 << 40, 1, 0, 1, 1, R=R, grid=(1,), num_warps=1)
                ck_ob._init_handles()
                ictx._observer_ck = ck_ob
            ck_ob[(1, 1, 1)](ag.ready.data_ptr(), ictx.obs.data_ptr(),
                              ag.round_base, rank, ag.num_chunks, n_chunks, R)

    # consumer: ONE full-M launch; table-driven mapping; per-tile hybrid
    # spin on LOCAL ready tickets, own-rank tiles read the local shard
    # directly (skip-local split)
    with torch.cuda.stream(ag.consumer_stream):
        tiles = (triton.cdiv(M, _CONSUMER_BLOCK_M)
                 * triton.cdiv(N_pr, _CONSUMER_BLOCK_N))
        grid_c = min(tiles, _CONSUMER_GRID_CAP)
        # S19c single-form addressing (see gemm_ag_tr.py): one A row stride
        assert a.stride(0) == K and a.stride(1) == 1, \
            "localA must be a contiguous [m_pr, K] shard"
        kernel_gemm_ag_tr_consumer_pred[(grid_c, 1, 1)](
            ag.ws_ptr,
            a,
            b,
            c,
            M,
            N_pr,
            K,
            K,  # single A row stride (ws and shard both [*, K] contig)
            b.stride(1), b.stride(0),  # b is [N, K]
            c.stride(0), c.stride(1),
            rank,
            R,
            n_chunks,
            ag.ready.data_ptr(),
            ictx.ctl_ptr,
            ag.round_base,
            diag_ptr=(ictx.diag_buf.data_ptr()
                      if ictx.diag_buf is not None else 0),
            BLOCK_SIZE_M=_CONSUMER_BLOCK_M,
            BLOCK_SIZE_N=_CONSUMER_BLOCK_N,
            BLOCK_SIZE_K=_CONSUMER_BLOCK_K,
            GROUP_SIZE_M=_CONSUMER_GROUP_M,
            M_PER_CHUNK=CR,
            NUM_SMS=grid_c,
            EVEN_K=(K % _CONSUMER_BLOCK_K == 0),
            DIAG=(ictx.diag_buf is not None),
            num_warps=_CONSUMER_WARPS,
            # M-only s1 rule is CORRECT for this PERSISTENT launch (s19e
            # @M8192 N29568 K8192: grid128 s1 31.5ms vs s2 39.3ms -- at
            # 64KB/CTA only ~80 of 128 CTAs fit resident, 1.6 waves).
            # The S17 deep-K guard (s2 at K>=8192) applies to the
            # FULL-GRID consumer form (gemm_ag_tr.py); there tiles>grid
            # only at small M, which this rule already gives s2.
            num_stages=1 if M > 512 else 2,
        )
        # verify (stream-ordered behind the consumer): spot-recompute
        # elements of every tile from the known-good ws + b and flag
        # mismatches. R>1 only: at R=1 nothing stages into ws.
        if I1_VERIFY and R > 1:
            num_m = triton.cdiv(M, _CONSUMER_BLOCK_M)
            num_n = triton.cdiv(N_pr, _CONSUMER_BLOCK_N)
            ntile = num_m * num_n
            if ictx.verify_bad is None or ictx.verify_bad.numel() < ntile:
                ictx.verify_bad = torch.zeros(ntile, dtype=torch.int32,
                                              device="cuda")
            ck_v = ictx._verify_ck
            if ck_v is None:
                ck_v = kernel_i1_verify.warmup(
                    1 << 40, 1 << 40, 1 << 40, 1 << 40,
                    1, 1, 1, 1, 1, 1, 1, 1.0,
                    SPOT=I1_VERIFY_SPOT, BM=_CONSUMER_BLOCK_M,
                    BN=_CONSUMER_BLOCK_N, grid=(1,), num_warps=1)
                ck_v._init_handles()
                ictx._verify_ck = ck_v
            ck_v[(ntile, 1, 1)](c.data_ptr(), ag.ws_ptr, b.data_ptr(),
                                ictx.verify_bad.data_ptr(), M, N_pr, K,
                                c.stride(0), c.stride(1), b.stride(0),
                                num_n, I1_VERIFY_TOL,
                                I1_VERIFY_SPOT, _CONSUMER_BLOCK_M,
                                _CONSUMER_BLOCK_N)
        if R > 1:
            # release: my consumer is done with everyone's staged rows --
            # every source may overwrite its shard region next round.
            # FIX (2026-10-14, the ~0.3%/rank-round tear): this xchg used
            # to launch on the CALLER's stream, which is not ordered after
            # the consumer (the join below is what orders the caller) --
            # the ack fired while the consumer still read the peers' ws,
            # so a skew-fast peer restaged round r+1 straight into my
            # in-flight reads. The tear signature fits exactly: ws
            # snapshots 47/47 CLEAN post-round, error = one BLOCK_K=64
            # slice of NEXT-generation content (NOFIT vs every scanned
            # basis, which only spanned prev generations), rate = the
            # skew-tail probability. RS v5 / AR rd stay clean because
            # their acks are consumer-ordered. Launching the xchg ON the
            # consumer stream makes it fire on real completion.
            done = ticket(n_chunks - 1)
            with torch.cuda.stream(ag.consumer_stream):
                for d in range(R):
                    if d != rank:
                        _launch_flag_xchg(ag.peer_ack[d] + rank * 4, done)
            ag.last_done_ticket = done

    # verify verdict + repair (host): read the flag plane after the
    # verify kernel retires, recompute flagged tiles on the consumer
    # stream -- still before the join records it, so the caller orders
    # after the repair too. The ack above may already have fired: safe,
    # a peer restage only overwrites the PEER's ws while the repair
    # reads MY ws, b and c.
    if I1_VERIFY and R > 1:
        ev_v = torch.cuda.Event()
        ev_v.record(ag.consumer_stream)
        ev_v.synchronize()
        idx = torch.nonzero(ictx.verify_bad.cpu()).flatten()
        if idx.numel() > 0:
            num_n = triton.cdiv(N_pr, _CONSUMER_BLOCK_N)
            ck_r = ictx._repair_ck
            if ck_r is None:
                ck_r = kernel_i1_tile_repair.warmup(
                    1 << 40, 1 << 40, 1 << 40, 1 << 40, 1, 1, 1, 1, 1,
                    BM=_CONSUMER_BLOCK_M, BN=_CONSUMER_BLOCK_N,
                    BK=_CONSUMER_BLOCK_K, grid=(1,), num_warps=8)
                ck_r._init_handles()
                ictx._repair_ck = ck_r
            with torch.cuda.stream(ag.consumer_stream):
                # H2D inside the with: the list upload is stream-ordered
                # ahead of the repair kernel on this same stream
                lst = idx.to(torch.int32).cuda()
                ck_r[(int(idx.numel()), 1, 1)](
                    c.data_ptr(), ag.ws_ptr, b.data_ptr(), lst.data_ptr(),
                    num_n, K, c.stride(0), c.stride(1), b.stride(0),
                    _CONSUMER_BLOCK_M, _CONSUMER_BLOCK_N,
                    _CONSUMER_BLOCK_K)
            ictx.verify_hits += int(idx.numel())
            if I1_VERIFY_LOG:
                print(f"[i1] verify round={ag.round_base}: repaired "
                      f"{int(idx.numel())} tile(s)", flush=True)

    # join: the caller orders after every op stream incl. the observer
    # (the obs plane is complete when the consumer is, so the D2H read
    # next round is safe)
    for si, st in enumerate(all_streams):
        ev = ictx.events[1 + si]
        assert h.hipEventRecord(ev, ctypes.c_void_p(st.cuda_stream)) == 0
        assert h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream), ev,
                                    0) == 0
    a.record_stream(ag.stage_stream)
    a.record_stream(ag.lane_streams[0])
    b.record_stream(ag.consumer_stream)
    c.record_stream(ag.consumer_stream)
    return c


def i1_ag_tr(a, b, ictx):
    """gather_r(a_r) @ b with the I1 table-driven consumer; returns
    c: [R * M_per_rank, N_per_rank]."""
    c = torch.empty(ictx.ag.num_ranks * a.shape[0], b.shape[0],
                    dtype=ictx.ag.output_dtype, device=a.device)
    i1_ag_tr_op(a, b, c, ictx)
    return c


def calibrate_sleep_cycles_per_us(cycles=1 << 22, iters=5):
    """Event-time N _sleep cycles -> cycles/us (the S8 gpu_delay
    calibration, per rank, on the current device)."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    best = None
    for _ in range(iters):
        torch.cuda.synchronize()
        e0.record()
        torch.cuda._sleep(cycles)
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1e3
        cpu = cycles / us if us > 0 else 0.0
        if best is None or cpu < best:
            best = cpu
    return best


__all__ = [
    "I1PredContext",
    "create_i1_context",
    "finalize_i1_context",
    "i1_table_write",
    "i1_ag_tr_op",
    "i1_ag_tr",
    "i1_observer_read",
    "calibrate_sleep_cycles_per_us",
    "kernel_gemm_ag_tr_consumer_pred",
    "kernel_i1_observer",
    "kernel_i1_verify",
    "kernel_i1_tile_repair",
    "kernel_i1_table_write",
    "kernel_i1_peek",
    "kernel_i1_arrive_stage",
    "kernel_i1_pull",
    "kernel_i1_copy",
    "TABLE_HEADER_WORDS",
    "I1_ARRIVE_ENGINE",
    "I1_READY_ENGINE",
    "I1_PULL_KERNEL",
    "I1_STAGE_KERNEL",
    "_PULL_BLOCK",
    "_CONSUMER_GRID_CAP",
]
