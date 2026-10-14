"""Order-driven GEMM-AllReduce op for the I1 L1 predictor experiment on
the AR transport (the numerically clean path: RS/AR soaks 0/600 fails
each while the AG consumer carries the residual ~1%/rank-round tear).

Same panel pipeline as gemm_ar_rd (local GEMM -> push snapshot -> wait
arrival -> accumulate), with the two knobs the three-arm rig needs:

  order -- the consumption sequence over the n_tasks panels. Sequence
           position p runs on slot stream p % S. Everything that
           touches PEER memory (scatter slot, arrival/ack flag cells,
           ticket values) stays keyed by the task's OWN id, so arms
           differ ONLY in the sequence. This is the host-enqueue
           equivalent of the I1 table's order[] field (the in-kernel
           table consumer is the AG variant, unit-tested separately).
  gates -- optional [(task, flag_va, expected_ticket)]: the stream
           waits flag >= expected BEFORE pushing that panel. The rig's
           straggler rank drives host-scheduled flag writes to delay
           chosen pushes (the injected delta, S8 single-victim shape).

The predictor's benefit mechanism on this structure: every slot stream
is a sequential queue [GEMM, push, wait-arrival, accumulate] x k, so a
late panel early in the queue blocks the ready work behind it
(head-of-line). Ordering the predicted-late panel LAST lets the
stream's own GEMM/accumulate work hide the straggler delta.
"""
import ctypes

import torch
import triton

from triton_dist.kernels.dcu.common_ops import hip, memcpy_async
from triton_dist.kernels.dcu.gemm_ar_rd import (
    _wait_ticket,
    kernel_accumulate_panel_rd,
    kernel_gemm_panel_rd,
)

MAX_TICKETS = 1 << 20

# per-ctx pool of TIMED task events (the ctx's own _events carry
# DISABLE_TIMING and cannot be used with hipEventElapsedTime). Event i
# completes when task i's accumulate+ack are done on its stream -- the
# device-side arrival timeline the L1 observer reads post-round.
_task_events = {}


def _get_task_events(ctx, h, n):
    evs = _task_events.get(id(ctx))
    if evs is None:
        h.hipEventCreateWithFlags.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        h.hipEventCreateWithFlags.restype = ctypes.c_int
        evs = []
        _task_events[id(ctx)] = evs
    while len(evs) < n:
        ev = ctypes.c_void_p()
        assert h.hipEventCreateWithFlags(ctypes.byref(ev), 0) == 0
        evs.append(ev)
    return evs


def task_events(ctx):
    """Timed per-task events of the LAST op (for post-round observers)."""
    return _task_events.get(id(ctx), [])


def gemm_ar_pred_op(a, b, c, ctx, order=None, gates=()):
    """C = sum_r a_r @ b_r^T with a chosen panel consumption order.

    order: permutation of range(n_tasks) (natural order if None).
    gates: iterable of (task_id, flag_va, expected_ticket) -- before
           pushing that panel the stream waits flag >= expected.
    """
    M, K = a.shape
    N = b.shape[0]
    rank, R, S = ctx.rank, ctx.num_ranks, ctx.stage_slots
    CR = ctx.chunk_rows
    assert b.shape[1] == K and c.shape == (M, N) and N == ctx.N
    assert M <= ctx.max_M and M % 64 == 0 and K % 64 == 0
    assert a.dtype == ctx.input_dtype and b.dtype == ctx.input_dtype
    assert a.stride(1) == 1 and a.stride(0) == K
    assert b.stride(1) == 1 and b.stride(0) == K

    cur = torch.cuda.current_stream()
    h = hip()
    n_tasks = triton.cdiv(M, CR)
    if order is None:
        order = list(range(n_tasks))
    assert sorted(order) == list(range(n_tasks)), "order must permute tasks"
    gates = {t: (va, exp) for t, va, exp in gates}
    partners = ctx.partners
    mode = ctx.wait_mode

    ctx.round_base += ctx.round_stride
    assert ctx.round_base + n_tasks < MAX_TICKETS
    tickets_va = ctx.tickets.data_ptr()

    ev_fork = ctx._events[0]
    assert h.hipEventRecord(ev_fork, ctypes.c_void_p(cur.cuda_stream)) == 0
    for sst in ctx.slot_streams:
        assert h.hipStreamWaitEvent(ctypes.c_void_p(sst.cuda_stream),
                                    ev_fork, 0) == 0
    tevs = _get_task_events(ctx, h, n_tasks)

    for p, i in enumerate(order):
        r0 = i * CR
        rows = min(CR, M - r0)
        # peer-memory key = SEQUENCE slot (p % S), same as the execution
        # stream: stream<->slot stay 1:1 under ANY order, so a stream's
        # waits all belong to its own slot chain (natural-keyed slots
        # under permutation spread one stream's tasks over several slots
        # and cross-stream ack coupling cost ~2-3ms/round, measured).
        # Both ranks run the same order, so both sides consume each
        # slot's occupants in the same relative sequence (deadlock rule),
        # and POSITION-keyed tickets keep each slot's GTE chain monotone
        # under permutation (task-keyed tickets made slot 3's arrival
        # cell go 8,12,16,4 under victim-last; the stale 16 satisfied the
        # victim's wait early -> stale/torn accumulate, e_std = sqrt(2) x
        # panel std, exactly-one-panel damage).
        s_nat = p % S
        sst = ctx.slot_streams[p % S]
        tev = tevs[i]
        ticket = ctx.round_base + p + 1
        prev_free = ctx.last_ticket_per_slot[s_nat]
        ctx.last_ticket_per_slot[s_nat] = ticket
        c_panel = c[r0:r0 + rows]

        with torch.cuda.stream(sst):
            grid = (triton.cdiv(rows, ctx.block_m) * triton.cdiv(N, 128),)
            kernel_gemm_panel_rd[grid](
                a, b, c_panel,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(1), b.stride(0),
                r0, rows,
                BLOCK_SIZE_M=ctx.block_m, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                GROUP_SIZE_M=8, num_warps=4,
                num_stages=1 if (CR > 512 and K < 8192) else 2)

            if R > 1:
                # slot-reuse gate: every partner acked the previous
                # occupant of scatter slot s_nat (in push sequence).
                for q in partners:
                    _wait_ticket(ctx.ack_ptr + (q * S + s_nat) * 4,
                                 prev_free, sst, mode)
                if i in gates:
                    fva, fexp = gates[i]
                    _wait_ticket(fva, fexp, sst, mode)
                for q in partners:
                    dst = ctx.peer_scatter[q] + ((s_nat * R + rank)
                                                 * CR * N) * ctx.elemsz
                    memcpy_async(dst, c_panel.data_ptr(),
                                 rows * N * ctx.elemsz, sst.cuda_stream)
                    memcpy_async(ctx.peer_arrival[q]
                                 + (rank * S + s_nat) * 4,
                                 tickets_va + 4 * ticket, 4, sst.cuda_stream)
                    _wait_ticket(ctx.arrival_ptr + (q * S + s_nat) * 4,
                                 ticket, sst, mode)
                    src = (ctx.scatter_ptr
                           + (s_nat * R + q) * CR * N * ctx.elemsz)
                    kernel_accumulate_panel_rd[
                        (triton.cdiv(rows, 128) * triton.cdiv(N, 128),)](
                        src, c_panel, rows, N,
                        c_panel.stride(0), c_panel.stride(1),
                        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, num_warps=8)
                for q in partners:
                    memcpy_async(ctx.peer_ack[q] + (rank * S + s_nat) * 4,
                                 tickets_va + 4 * ticket, 4, sst.cuda_stream)
            assert h.hipEventRecord(tev, ctypes.c_void_p(sst.cuda_stream)) == 0

    for si, sst in enumerate(ctx.slot_streams):
        ev = ctx._events[1 + si]
        assert h.hipEventRecord(ev, ctypes.c_void_p(sst.cuda_stream)) == 0
        assert h.hipStreamWaitEvent(ctypes.c_void_p(cur.cuda_stream), ev,
                                    0) == 0
    return c


__all__ = ["gemm_ar_pred_op", "task_events"]
