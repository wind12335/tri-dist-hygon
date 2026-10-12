"""DCU port of the newer upstream triton_dist/kernels/amd/ep_a2a(_intra_node).py
(triton-3.7 era): MoE expert-parallel dispatch + combine (all-to-all).

Upstream architecture (amd, intra-node):
  dispatch:  per-(token, topk) warp-level SIMT kernel allocates a scatter
    index with atomic_add_per_warp and fires a DEVICE-INITIATED
    putmem_signal_warp of ONE token row (~H*dtype bytes, dynamically routed
    by expert) to the expert's rank; a metadata kernel all-gathers the
    per-expert splits with device putmem_signal_nbi_block + dl.wait +
    barrier_on_this_grid.
  combine:   the token owner re-reads each topk contribution from the
    remote rank via dl.symm_at(input_buf, expert_rank) + plain ld (pull).

Why only half of that transplants to this stack (S7):
- the dispatch DATA plane is token-granularity device-initiated traffic ->
    dead on the 19.5 MB/s device pipe (S4/S6 probes; a single token row at
    14 KB would be ~0.7 ms). Re-derived: ONE local staging kernel compacts
    my tokens per (dest, expert) segment (metadata pre-computes all
    segment bases), then the host engine-memcpys each dest's whole staged
    run (~110 GB/s) -- 1 push per dest instead of T*topk device puts.
- the metadata plane (splits allgather) is tiny -> a packed [flag|splits]
    message per dest through ONE host putmem (RMA), then a single-CTA
    device kernel spins the flags and computes every segment table on
    device (the v1 host path -- bincount, D2H, MPI_Allgather, numpy,
    H2D -- measured 0.43 ms/op, a third of dispatch; the RMA path drops
    the MPI round-trip, the wide torch.cuda.synchronize and the numpy
    table algebra; only two NARROW syncs remain, each waiting nothing
    but this op's own small kernels).
- the combine PULL model DOES transplant: bench/s7_peer_load_probe.py
    measured device-side plain loads from the PEER's symmetric heap
    correct + fresh across peer re-writes (barrier between rounds) at
    74-83 GB/s warm (first touch ~160 ms = one-time page-table build,
    paid at ctx create by k_touch). The owner kernel reads each topk row
    from the expert rank's symmetric expert_out buffer exactly like
    upstream's dl.symm_at + ld, with VAs pre-translated via host.ptr.

Skipped from upstream in v1 (semantics-preserving): the skipped-token
duplicate-rank detection / local pre-combine (each (t, j) is staged and
pulled independently -- correct, at most topk data amplification) and
weight forwarding (weights stay on the owner, applied at combine).

Phase protocol per round (2 SymmBarrier rounds, each guards a real
hazard; an op-start D.start barrier was removed -- the peer's next-round
push cannot reach my recv before my expert phase finished reading it,
because the peer must first pass the previous round's C.start):
  D.end    -- all dispatch data landed in every recv before any expert
              phase starts
  C.start  -- every expert_out written before any combine reads (remote
              loads); transitively also makes out buffers safe to
              overwrite next round
"""
import ctypes
import dataclasses
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl

from triton_dist.kernels.dcu.common_ops import (
    SymmBarrier,
    dcu_state,
    hip,
    init_dcu,
    memcpy_async,
)


@triton.jit(do_not_specialize=["base"])
def k_touch(base, sink_ptr):
    """Pay the one-time peer-VA page-table build (~160 ms at 16 MB in the
    S7 probe) at ctx create, not in the first combine."""
    p = base.to(tl.pointer_type(tl.float32))
    v = tl.load(p)
    tl.store(sink_ptr, v)


@triton.jit(do_not_specialize=["msg_ptr", "round", "n"])
def kernel_pack_meta(
        topk_ptr,  # [T, topk] int32 GLOBAL expert ids (read linearly)
        msg_ptr,  # i32 [1 + E] device staging: [round | splits]
        round,
        n,  # T * topk
        E: tl.constexpr,
        EZ: tl.constexpr,  # next_pow2(E): zero/store width for the row
        BLOCK: tl.constexpr,  # histogram scan width
):
    """Build this round's metadata message in ONE launch (single CTA):
    zero the splits row, histogram every (t, j) expert id into it with
    gather atomics, then store the round flag word. Same block ->
    program order; the host-side sync before the putmem covers the rest.

    Replaces the torch chain (reshape -> .long() -> bincount -> .to) --
    the s7_meta_decompose probe measured 0.13 ms of pure HOST launch
    time for those four ops, a quarter of the whole metadata plane."""
    z = tl.arange(0, EZ)
    tl.store(msg_ptr + 1 + z, tl.zeros((EZ,), tl.int32), mask=z < E)
    tl.debug_barrier()
    offs = tl.arange(0, BLOCK)
    for i in range(0, n, BLOCK):
        idx = i + offs
        m = idx < n
        e = tl.load(topk_ptr + idx, mask=m, other=0)
        tl.atomic_add(msg_ptr + 1 + e, 1, mask=m)
    tl.store(msg_ptr + tl.arange(0, 1), tl.full((1,), round, tl.int32))


@triton.jit(do_not_specialize=["meta_base", "round", "my_rank"])
def kernel_meta_tables(
        meta_base,  # i32 VA of MY symmetric meta_msg [R, 1 + E]
        dst_row_base_ptr,  # i32 [R, epr] out
        stage_row_base_ptr,  # i32 [R, epr] out
        recv_seg_ptr,  # i32 [R, epr] out
        host_plan_ptr,  # i32 [R, 4] out: per-dest [rows, dst_row0,
        #                stage_row0] + [recv_total, -, -] in row R-1's tail
        recv_total_ptr,  # i32 [1] out
        round,
        my_rank,
        E: tl.constexpr,
        EPR: tl.constexpr,
        NUM_RANKS: tl.constexpr,
):
    """Single CTA: hybrid-spin every rank's meta flag to `round` (flags
    arrive via host RMA putmem -> atomic probe at the L2 coherence point,
    the barrier_spin_kernel pattern), then compute all segment tables
    from the all-gathered splits rows (my own symmetric buffer -- the
    peers' putmems landed them there; plain local loads).

    host_plan carries everything the HOST needs for the engine pushes
    (per-dest bytes and both run bases) plus recv_total, so one small D2H
    replaces per-table .cpu() syncs. Table semantics (must match
    _segment_tables): my staging is ordered (dest, expert); dest d's recv
    is ordered (src, expert)."""
    meta = meta_base.to(tl.pointer_type(tl.int32))
    for r in range(NUM_RANKS):
        fp = meta + r * (1 + E)
        v = tl.atomic_add(fp, 0)
        while v < round:
            for _ in range(128):
                v = tl.load(fp, volatile=True)
            if v < round:
                v = tl.atomic_add(fp, 0)
    tl.debug_barrier()

    # my staging segments, ordered (d, k)
    run = 0
    for d in range(NUM_RANKS):
        tl.store(host_plan_ptr + d * 4 + 2, run)
        for k in range(EPR):
            tl.store(stage_row_base_ptr + d * EPR + k, run)
            run += tl.load(meta + my_rank * (1 + E) + 1 + d * EPR + k)
    # my run inside d's recv: after every earlier src's whole run
    for d in range(NUM_RANKS):
        base = 0
        for s in range(my_rank):
            for k in range(EPR):
                base += tl.load(meta + s * (1 + E) + 1 + d * EPR + k)
        tl.store(host_plan_ptr + d * 4 + 1, base)
        my_run = 0
        for k in range(EPR):
            tl.store(dst_row_base_ptr + d * EPR + k, base)
            v = tl.load(meta + my_rank * (1 + E) + 1 + d * EPR + k)
            base += v
            my_run += v
        tl.store(host_plan_ptr + d * 4 + 0, my_run)
    # my recv segments, ordered (src, k)
    base_r = 0
    for s in range(NUM_RANKS):
        for k in range(EPR):
            tl.store(recv_seg_ptr + s * EPR + k, base_r)
            base_r += tl.load(meta + s * (1 + E) + 1 + my_rank * EPR + k)
    tl.store(recv_total_ptr, base_r)
    tl.store(host_plan_ptr + (NUM_RANKS - 1) * 4 + 3, base_r)


@triton.jit(do_not_specialize=["recv_base", "my_rank", "row_bytes"])
def kernel_moe_stage_dispatch(
        tokens_ptr,  # [T, H] input rows
        topk_ptr,  # [T, topk] int32 GLOBAL expert ids
        scatter_ptr,  # [T, topk] int32 out: recv-row of each (t, j)
        recv_base,  # i64 VA of MY recv buffer row 0 (symmetric heap)
        stage_ptr,  # [T*topk, H] staging rows (torch, local)
        dst_row_base_ptr,  # i32 [R, epr]: my segment start row in d's recv
        stage_row_base_ptr,  # i32 [R, epr]: my segment start row in stage
        counters_ptr,  # i32 [R*epr] per-(dest, expert) running counts
        my_rank,
        row_bytes,  # H * elemsz
        H: tl.constexpr,
        EPR: tl.constexpr,  # experts per rank
        TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
):
    """One block per (token, topk-slot): allocate the segment position with
    a LOCAL atomic (the segment base comes from the metadata tables, so the
    position is globally unique in the dest's recv buffer), copy the token
    row, record the scatter index.

    Rows for MY OWN experts go straight into my recv buffer (no self-copy);
    everything else lands in the staging buffer for the per-dest engine
    push. Both stores are masked (complementary scalar masks) -- no runtime
    control flow around stores (the S6 trap list: scalar atomics inside an
    `if` do not compile on 3.3)."""
    pid = tl.program_id(0)  # pid = t * TOPK + j
    t = pid // TOPK

    e = tl.load(topk_ptr + pid).to(tl.int32)
    d = e // EPR
    k = e - d * EPR
    tab = d * EPR + k

    row_local = tl.atomic_add(counters_ptr + tab, 1)
    dst_row0 = tl.load(dst_row_base_ptr + tab).to(tl.int64)
    st_row0 = tl.load(stage_row_base_ptr + tab).to(tl.int64)
    tl.store(scatter_ptr + pid, (dst_row0 + row_local).to(tl.int32))

    is_local = d == my_rank
    recv_p = recv_base.to(tl.pointer_type(tokens_ptr.dtype.element_ty))
    for off in range(0, H, BLOCK_H):
        offs = off + tl.arange(0, BLOCK_H)
        m = offs < H
        v = tl.load(tokens_ptr + t.to(tl.int64) * H + offs, mask=m)
        r_st = (st_row0 + row_local).to(tl.int64) * H + offs
        r_rc = (dst_row0 + row_local).to(tl.int64) * H + offs
        tl.store(stage_ptr + r_st, v, mask=m & (d != my_rank))
        tl.store(recv_p + r_rc, v, mask=m & is_local)


@triton.jit(do_not_specialize=["H"])
def kernel_moe_pull_combine(
        out_va_ptr,  # i64 [R] row-0 VAs of every rank's expert_out buffer
        scatter_ptr,  # [T, topk] int32 recv-row of each (t, j)
        topk_ptr,  # [T, topk] int32 (recompute the expert's rank)
        weights_ptr,  # [T, topk] fp32 (unused if not HAS_WEIGHT)
        out_ptr,  # [T, H] result rows
        H,
        EPR: tl.constexpr,
        TOPK: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_WEIGHT: tl.constexpr,
):
    """One block per token: out[t] = sum_j w[t,j] * expert_out_{rank(e)}[row]
    -- remote plain loads at pre-translated peer VAs (the upstream
    dl.symm_at pull, S7-probe-validated). fp32 accumulation."""
    t = tl.program_id(0).to(tl.int64)
    dtype = out_ptr.dtype.element_ty
    for off in range(0, H, BLOCK_H):
        offs = off + tl.arange(0, BLOCK_H)
        m = offs < H
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for j in range(TOPK):
            pid = t * TOPK + j
            e = tl.load(topk_ptr + pid).to(tl.int32)
            d = e // EPR
            row = tl.load(scatter_ptr + pid).to(tl.int64)
            base = tl.load(out_va_ptr + d)
            p = base.to(tl.pointer_type(dtype))
            v = tl.load(p + row * H + offs, mask=m, other=0.0).to(tl.float32)
            if HAS_WEIGHT:
                acc += v * tl.load(weights_ptr + pid)
            else:
                acc += v
        tl.store(out_ptr + t * H + offs, acc.to(dtype), mask=m)


def _mpi_allgather_i32(state, send_t, recv_t):
    """Allgather int32 pinned buffers over the bootstrap MPI comm (tiny
    metadata plane; the dushmem control plane stays untouched)."""
    mpi, comm = state["mpi"], state["comm"]
    if not hasattr(mpi, "_tri_mpi_int"):
        mpi._tri_mpi_int = ctypes.addressof(
            ctypes.c_int.in_dll(mpi, "ompi_mpi_int"))
    mpi.MPI_Allgather.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p]
    mpi.MPI_Allgather.restype = ctypes.c_int
    n = send_t.numel()
    rc = mpi.MPI_Allgather(ctypes.c_void_p(send_t.data_ptr()), n,
                           ctypes.c_void_p(mpi._tri_mpi_int),
                           ctypes.c_void_p(recv_t.data_ptr()), n,
                           ctypes.c_void_p(mpi._tri_mpi_int),
                           ctypes.c_void_p(comm))
    assert rc == 0, f"MPI_Allgather rc={rc}"


@dataclasses.dataclass
class MoEA2AContext:
    rank: int
    num_ranks: int
    num_experts: int
    epr: int  # experts per rank
    hidden: int
    topk: int
    max_tokens: int
    max_recv_rows: int  # worst case: every (t, j) of EVERY src to one rank
    recv_ptr: int  # symmetric [max_recv_rows, H] dispatch destination
    out_ptr: int  # symmetric [max_recv_rows, H] expert outputs (1:1 rows)
    peer_recv_va: list  # host.ptr(recv_ptr, d)
    peer_meta_va: list  # host.ptr(meta_ptr, d)
    out_va_t: torch.Tensor  # i64 [R] row-0 VAs of expert_out buffers
    scatter_idx: torch.Tensor  # [max_tokens, topk] i32
    counters: torch.Tensor  # [R*epr] i32
    dst_row_base: torch.Tensor  # [R, epr] i32 device tables (per-op)
    stage_row_base: torch.Tensor
    recv_seg: torch.Tensor  # [R, epr] i32: MY recv segment start rows
    meta_ptr: int  # symmetric i32 [R, 1 + E]: [flag | splits] per src
    meta_msg: torch.Tensor  # i32 [1 + E] device staging of MY message
    host_plan: torch.Tensor  # i32 [R, 4] device: per-dest
    #                [rows, dst_row0, stage_row0] + recv_total in the tail
    host_plan_pinned: torch.Tensor  # i32 [R, 4] host pinned mirror
    recv_total_t: torch.Tensor  # i32 [1] device
    barrier: SymmBarrier
    input_dtype: torch.dtype
    row_bytes: int
    BLOCK_H: int
    meta_round: int = 0  # monotonic meta-flag rounds (never reset)
    # per-op state (filled by the last dispatch)
    stage: Optional[torch.Tensor] = None  # [T*topk, H] staging rows


def create_moe_a2a_intra_node_context(num_experts, hidden, dtype, max_tokens,
                                      topk, rank, num_ranks, mpi, comm,
                                      BLOCK_H=1024):
    """Symmetric recv/expert_out buffers + scatter metadata + barrier for
    MoE dispatch + combine.  H must be a multiple of BLOCK_H."""
    init_dcu(mpi, comm, rank, num_ranks)
    assert num_experts % num_ranks == 0
    assert hidden % BLOCK_H == 0
    s = dcu_state()

    elemsz = torch.empty((), dtype=dtype).element_size()
    row_bytes = hidden * elemsz
    max_recv_rows = max_tokens * topk * num_ranks

    recv_ptr = s["host"].malloc(max_recv_rows * row_bytes)
    out_ptr = s["host"].malloc(max_recv_rows * row_bytes)
    peer_recv_va = [s["host"].ptr(recv_ptr, d) for d in range(num_ranks)]
    out_va_t = torch.tensor([s["host"].ptr(out_ptr, d)
                             for d in range(num_ranks)],
                            dtype=torch.int64, device="cuda")

    # pay the peer-VA page-table build now, not in the first combine
    sink = torch.empty(1, dtype=torch.float32, device="cuda")
    for d in range(num_ranks):
        if d != rank:
            k_touch[(1,)](out_va_t[d], sink)

    epr = num_experts // num_ranks
    meta_bytes = num_ranks * (1 + num_experts) * 4
    meta_ptr = s["host"].malloc(meta_bytes)
    # flags gate table reads: garbage >= round would skip the wait
    h = hip()
    h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    h.hipMemset.restype = ctypes.c_int
    assert h.hipMemset(ctypes.c_void_p(meta_ptr), 0, meta_bytes) == 0

    ctx = MoEA2AContext(
        rank=rank, num_ranks=num_ranks, num_experts=num_experts, epr=epr,
        hidden=hidden, topk=topk, max_tokens=max_tokens,
        max_recv_rows=max_recv_rows, recv_ptr=recv_ptr, out_ptr=out_ptr,
        peer_recv_va=peer_recv_va,
        peer_meta_va=[s["host"].ptr(meta_ptr, d) for d in range(num_ranks)],
        out_va_t=out_va_t,
        scatter_idx=torch.zeros(max_tokens, topk, dtype=torch.int32,
                                device="cuda"),
        counters=torch.zeros(num_ranks * epr, dtype=torch.int32,
                             device="cuda"),
        dst_row_base=torch.zeros(num_ranks, epr, dtype=torch.int32,
                                 device="cuda"),
        stage_row_base=torch.zeros(num_ranks, epr, dtype=torch.int32,
                                   device="cuda"),
        recv_seg=torch.zeros(num_ranks, epr, dtype=torch.int32,
                             device="cuda"),
        meta_ptr=meta_ptr,
        meta_msg=torch.zeros(1 + num_experts, dtype=torch.int32,
                             device="cuda"),
        host_plan=torch.zeros(num_ranks, 4, dtype=torch.int32,
                              device="cuda"),
        host_plan_pinned=torch.zeros(num_ranks, 4, dtype=torch.int32,
                                     pin_memory=True),
        recv_total_t=torch.zeros(1, dtype=torch.int32, device="cuda"),
        barrier=SymmBarrier(num_ranks),
        input_dtype=dtype, row_bytes=row_bytes, BLOCK_H=BLOCK_H,
    )

    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0
    return ctx


def _segment_tables(splits_all, my_rank, num_ranks, epr):
    """Host-side segment algebra from the all-gathered splits [R, E].

    Returns (send_rows[d], stage_row_base[d,k], dst_row_base[d,k],
             recv_seg[src,k], recv_counts[src,k], recv_total) with the
    layouts the engine pushes and the kernels assume:
      - my staging buffer: segments ordered (d, expert-of-d) -> each
        dest's whole run is contiguous (ONE engine memcpy per dest)
      - dest d's recv buffer: segments ordered (src, expert-of-d) -> my
        run for d is contiguous at dst_row_base[d, 0]."""
    send_rows = np.array([splits_all[my_rank, d * epr:(d + 1) * epr].sum()
                          for d in range(num_ranks)], dtype=np.int64)

    stage_row_base = np.zeros((num_ranks, epr), dtype=np.int64)
    run = 0
    for d in range(num_ranks):
        for k in range(epr):
            stage_row_base[d, k] = run
            run += splits_all[my_rank, d * epr + k]

    # my run inside d's recv starts after all earlier srcs' whole runs
    dst_row_base = np.zeros((num_ranks, epr), dtype=np.int64)
    for d in range(num_ranks):
        base = sum(splits_all[s, d * epr:(d + 1) * epr].sum()
                   for s in range(my_rank))
        for k in range(epr):
            dst_row_base[d, k] = base
            base += splits_all[my_rank, d * epr + k]

    # MY recv side: segments ordered (src, k)
    recv_seg = np.zeros((num_ranks, epr), dtype=np.int64)
    recv_counts = np.zeros((num_ranks, epr), dtype=np.int64)
    base = 0
    for src in range(num_ranks):
        for k in range(epr):
            recv_seg[src, k] = base
            recv_counts[src, k] = splits_all[src, my_rank * epr + k]
            base += recv_counts[src, k]
    return send_rows, stage_row_base, dst_row_base, recv_seg, recv_counts, \
        int(base)


def moe_dispatch_intra_node_op(tokens, topk_idx, ctx, component="full"):
    """Route every (token, topk-slot) row to its expert's rank.

    tokens [T, H] (T <= ctx.max_tokens, H == ctx.hidden, ctx dtype);
    topk_idx [T, topk] int32 GLOBAL expert ids (validated by the caller /
    tests -- no per-op max/min D2H in this hot path).
    On return every rank's recv buffer holds all rows routed to it; the
    expert phase reads rows [0, recv_total) -- recv_total comes back as a
    DEVICE i32[1] tensor (ctx.recv_total_t) so no extra sync is forced;
    segment map in ctx.recv_seg. scatter_idx[t, j] gives the recv-row of
    every (t, j) for the later combine.

    component="meta": stop after the metadata plane (bincount, putmems,
    table kernel -- benchmarking mode; tables stay valid for a following
    stage/push pass)."""
    T, H = tokens.shape
    rank, R, epr = ctx.rank, ctx.num_ranks, ctx.epr
    assert T <= ctx.max_tokens and H == ctx.hidden
    assert tokens.dtype == ctx.input_dtype
    assert topk_idx.dtype == torch.int32 and topk_idx.shape == (T, ctx.topk)

    ctx.meta_round += 1
    round_ = ctx.meta_round

    # ---- metadata plane (RMA): count+pack [flag|splits] in ONE kernel,
    # then putmem. Nothing from the op's barriers is queued yet, so this
    # sync waits only my own small kernels, not a barrier round.
    ctx.counters.zero_()  # for the stage kernel's segment allocation
    kernel_pack_meta[(1,)](topk_idx, ctx.meta_msg, round_, T * ctx.topk,
                           E=ctx.num_experts,
                           EZ=triton.next_power_of_2(ctx.num_experts),
                           BLOCK=1024, num_warps=4)
    torch.cuda.synchronize()  # narrow: count+pack only
    host = dcu_state()["host"]
    msg_bytes = (1 + ctx.num_experts) * 4
    for d in range(R):
        if d != rank:
            # one putmem per dest carries flag AND splits: ordering inside
            # a single RMA message is guaranteed, so no separate flag wait
            host.putmem(ctx.meta_ptr + rank * msg_bytes,
                        ctx.meta_msg.data_ptr(), msg_bytes, d)
    # my OWN meta row: plain local copy, stream-ordered before the table
    # kernel -- it spins EVERY rank's flag, mine included
    memcpy_async(ctx.meta_ptr + rank * msg_bytes, ctx.meta_msg.data_ptr(),
                 msg_bytes, torch.cuda.current_stream().cuda_stream)

    # tables on device (spins every src's flag, then the segment algebra)
    kernel_meta_tables[(1,)](
        ctx.meta_ptr, ctx.dst_row_base, ctx.stage_row_base, ctx.recv_seg,
        ctx.host_plan, ctx.recv_total_t, round_, rank,
        E=ctx.num_experts, EPR=epr, NUM_RANKS=R, num_warps=1)
    # everything the host needs for the pushes, in ONE pinned D2H (the
    # single sync waits only the table kernel, not the barriers below)
    ctx.host_plan_pinned.copy_(ctx.host_plan, non_blocking=True)
    torch.cuda.synchronize()
    plan = ctx.host_plan_pinned  # [d] -> rows, dst_row0, stage_row0; tail rt
    assert int(plan[R - 1, 3]) <= ctx.max_recv_rows, "recv buffer overflow"
    if component == "meta":
        return ctx.recv_total_t

    # NOTE: no op-start barrier here. The hazard it would guard -- a
    # peer's round-(k+1) push overwriting my recv while my round-k expert
    # phase still reads it -- is already closed transitively: this API's
    # contract is dispatch -> expert -> combine per round, so a peer can
    # only reach this point after its combine(k) arrived at C.start(k),
    # and that barrier releases only when EVERY rank's expert phase k is
    # done. The first round is trivially safe (nothing reads recv yet).

    # ---- staging kernel: compact my routed rows + scatter indices ----
    stage = torch.empty(T * ctx.topk, H, dtype=ctx.input_dtype,
                        device="cuda")
    ctx.stage = stage
    kernel_moe_stage_dispatch[(T * ctx.topk,)](
        tokens, topk_idx, ctx.scatter_idx,
        ctx.recv_ptr, stage,
        ctx.dst_row_base, ctx.stage_row_base, ctx.counters,
        rank, ctx.row_bytes,
        H=H, EPR=epr, TOPK=ctx.topk, BLOCK_H=ctx.BLOCK_H,
        num_warps=4)

    # ---- data plane: ONE engine memcpy per dest of my whole staged run.
    # Same stream as the staging kernel -> ordered, no event needed.
    # Bases/sizes come from the pinned plan -- no further device sync.
    stream = torch.cuda.current_stream().cuda_stream
    for d in range(R):
        rows = int(plan[d, 0])
        if d == rank or rows == 0:
            continue
        dst = int(ctx.peer_recv_va[d] + int(plan[d, 1]) * ctx.row_bytes)
        src = int(stage.data_ptr() + int(plan[d, 2]) * ctx.row_bytes)
        memcpy_async(dst, src, rows * ctx.row_bytes, stream)

    # end guard: every rank's recv data landed before any expert phase
    ctx.barrier.round_barrier(rank)
    return ctx.recv_total_t


def moe_combine_intra_node_op(ctx, topk_idx, output, topk_weights=None):
    """out[t] = sum_j w[t,j] * expert_out[rank(e_j)][scatter_idx[t, j]].

    Call AFTER this rank's expert phase wrote ctx.recv_total rows into the
    expert_out buffer (row i <-> recv row i; symmetric VA ctx.out_ptr).
    topk_idx: the SAME tensor passed to dispatch; output [T, H]."""
    T = topk_idx.shape[0]
    assert output.shape == (T, ctx.hidden)
    assert output.dtype == ctx.input_dtype
    has_w = topk_weights is not None
    if has_w:
        assert topk_weights.shape == (T, ctx.topk) \
            and topk_weights.dtype == torch.float32
    # expert-done guard: all expert_out writes complete + visible
    ctx.barrier.round_barrier(ctx.rank)
    kernel_moe_pull_combine[(T,)](
        ctx.out_va_t, ctx.scatter_idx, topk_idx,
        topk_weights if has_w else ctx.counters,  # dummy ptr when unweighted
        output,
        ctx.hidden,
        EPR=ctx.epr, TOPK=ctx.topk, BLOCK_H=ctx.BLOCK_H,
        HAS_WEIGHT=has_w, num_warps=4)
    return output
