"""DCU (Hygon gfx936 / DTK dushmem) replacements for the amd common_ops.

Plane split (validated on 2x BW1000):
- DATA: triton tl.store / tl.load through IPC-opened raw peer pointers
  (pydushmem.ipc.IpcPtrList); direct stores work, but sys-scope
  tl.atomic_cas on remote IPC memory hangs -> no IPC atomics.
- SIGNAL: dushmem symmetric-heap atomics. Barrier = idempotent
  atomic_set of my flag to the round number on every PE + local volatile
  spin (monotonic rounds, no flag reset; set-not-add makes the
  all-threads uniform call harmless without a lane guard).
- HOST sync: hipStreamWaitValue32/64 + hipStreamWriteValue32/64 exist in
  DTK and are used exactly like the upstream hip-python bindings.

Requires the process set up exactly once via `init_dcu()`:
  mpi, comm = pydushmem.host.mpi_init()   # BEFORE import torch
  ...import torch...; hipSetDevice(rank)
  dcu_common.init_dcu(mpi, comm, rank, size)
"""
import ctypes
import os

import torch
import triton
import triton.language as tl

from pydushmem import device as dushmem_dev  # externs for jit bodies

_state = None


def init_dcu(mpi, comm, rank, size, host=None):
    """Initialize the dushmem control plane (MPI bootstrap)."""
    global _state
    import pydushmem
    if _state is None:
        if host is None:
            h = pydushmem.DushmemHost(preload_bootstrap=False)
            h.init_mpi(comm)
        else:
            h = host
        _state = {"mpi": mpi, "comm": comm, "rank": rank, "size": size,
                  "host": h, "extern_libs": pydushmem.extern_libs()}
    assert _state["rank"] == rank and _state["size"] == size
    return _state


def dcu_state():
    assert _state is not None, "call init_dcu() first"
    return _state


def hip():
    s = dcu_state()
    if "hip" not in s:
        s["hip"] = ctypes.CDLL(
            "/opt/dtk-26.04/lib/libamdhip64.so", mode=ctypes.RTLD_GLOBAL)
        for fn in ("hipStreamWaitValue32", "hipStreamWriteValue32",
                   "hipStreamWaitValue64", "hipStreamWriteValue64"):
            getattr(s["hip"], fn).restype = ctypes.c_int
    return s["hip"]


def memcpy_async(dst: int, src: int, nbytes: int, stream_handle: int):
    """hipMemcpyAsync (DeviceToDevice) with plain int VAs on the stream
    whose handle is `stream_handle` (torch_stream.cuda_stream).

    The S4 transport probe established this as THE fast cross-GPU path on
    BW1000/DTK26.04: ~110 GB/s single-op into a peer's symmetric-heap VA
    (host.ptr translation), fully coherent for peer plain loads, while
    every device-initiated dushmem transfer runs at ~19 MB/s."""
    h = hip()
    h.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.c_int,
                                 ctypes.c_void_p]
    rc = h.hipMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src),
                          nbytes, 3, ctypes.c_void_p(stream_handle))
    assert rc == 0, f"hipMemcpyAsync rc={rc}"


# S11 (bench/s11_probe_multistream.py + s11_probe_evtax.py): BW1000 runs
# MULTIPLE copy-engine queues concurrently -- aggregate bandwidth scales
# with streams for ops >=4MB (4MB x4: 54->63, 8MB x4: 99->111, 8MB x8:
# 108->125 GB/s), two big ops on ONE stream even degrade below single-op
# rate (32MB x2: 83 vs 107), and full duplex is free. BUT torch's
# cross-stream wait_stream choreography costs 10-40 us per call here and
# eats the whole gain; raw non-timing hipEvent record/wait cost 1-4 us and
# keep it (8MB x8: 114.5 serial -> 137.4 ideal -> 132.8 GB/s with raw
# events). <=2MB ops are per-op-overhead-bound (~21-33 GB/s at any stream
# count) and never benefit. So fan-outs go out through a side-stream pool
# with RAW-event fork/join, engaged only in the regime where it measured a
# win: >=4 ops of >=4MB each (i.e. the >=8-card (R-1)-dest fan-out).
# TD_PUSH_STREAMS=1 restores the single-stream fan-out everywhere (A/B).
PUSH_STREAMS = max(1, int(os.environ.get("TD_PUSH_STREAMS", "8")))
PUSH_OP_MIN_BYTES = 4 << 20
# TD_PUSH_MIN_OPS is a test knob: lower it to force the multi-stream path
# on 2 cards (correctness of the raw-event fork/join) -- production runs
# never need it, the default engages only the >=8-card fan-out shapes.
PUSH_OP_MIN_COUNT = max(1, int(os.environ.get("TD_PUSH_MIN_OPS", "4")))

_push_stream_pool_cache = None
_push_event_cache = None  # (fork_event, [join events per pool slot])


def _fan_out_events(h, n):
    global _push_event_cache
    if _push_event_cache is None or len(_push_event_cache[1]) < n:
        def mk():
            ev = ctypes.c_void_p()
            rc = h.hipEventCreateWithFlags(ctypes.byref(ev), 0x2)
            assert rc == 0, f"hipEventCreateWithFlags rc={rc}"
            return ev
        joins = [mk() for _ in range(max(n, PUSH_STREAMS))]
        _push_event_cache = (mk(), joins)
    return _push_event_cache[0], _push_event_cache[1][:n]


def fan_out_memcpy_async(ops):
    """Issue (dst, src, nbytes) engine copies round-robin across a side
    stream pool, forked from and joined back to the CURRENT torch stream
    with RAW hipEvents (torch wait_stream is 10-40x pricier per call --
    S11 evtax probe).

    On return the copies are stream-ordered complete on the current stream
    (still async on the host): any event recorded, engine write issued, or
    host putmem event-synced on the current stream afterwards sees ALL of
    them landed -- every signal-ordering contract is preserved. Outside
    the measured-win regime (<PUSH_OP_MIN_COUNT ops or any op below
    PUSH_OP_MIN_BYTES, TD_PUSH_STREAMS=1) this degrades to the plain
    single-stream path."""
    if (PUSH_STREAMS <= 1 or len(ops) < PUSH_OP_MIN_COUNT
            or min(o[2] for o in ops) < PUSH_OP_MIN_BYTES):
        stream = torch.cuda.current_stream().cuda_stream
        for dst, src, nbytes in ops:
            memcpy_async(dst, src, nbytes, stream)
        return
    global _push_stream_pool_cache
    if _push_stream_pool_cache is None:
        _push_stream_pool_cache = [torch.cuda.Stream()
                                   for _ in range(PUSH_STREAMS)]
    h = hip()
    h.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    h.hipStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                     ctypes.c_uint64]
    cur = torch.cuda.current_stream()
    pool = _push_stream_pool_cache[:min(PUSH_STREAMS, len(ops))]
    ev_fork, ev_join = _fan_out_events(h, len(pool))
    cur_h = ctypes.c_void_p(cur.cuda_stream)
    # fork: pool streams order after the current stream's dependencies
    assert h.hipEventRecord(ev_fork, cur_h) == 0
    for s in pool:
        assert h.hipStreamWaitEvent(ctypes.c_void_p(s.cuda_stream),
                                    ev_fork, 0) == 0
    n = len(pool)
    for i, (dst, src, nbytes) in enumerate(ops):
        memcpy_async(dst, src, nbytes, pool[i % n].cuda_stream)
    # join: the current stream now orders after every copy
    for i, s in enumerate(pool):
        assert h.hipEventRecord(ev_join[i], ctypes.c_void_p(s.cuda_stream)) == 0
        assert h.hipStreamWaitEvent(cur_h, ev_join[i], 0) == 0


HIP_WAIT_EQ = 0  # hipStreamWaitValueEq


def wait_eq(ptr: int, val: int, stream, require_i64=False):
    h = hip()
    if not require_i64:
        rc = h.hipStreamWaitValue32(stream.cuda_stream, ctypes.c_void_p(ptr),
                                    ctypes.c_uint(val), 0xFFFFFFFF)
    else:
        h.hipStreamWaitValue64.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_uint64, ctypes.c_uint]
        rc = h.hipStreamWaitValue64(stream.cuda_stream, ctypes.c_void_p(ptr),
                                    ctypes.c_uint64(val), 0xFFFFFFFF)
    assert rc == 0, f"hipStreamWaitValue rc={rc}"


def set_signal(ptr: int, val: int, stream, require_i64=False):
    h = hip()
    if not require_i64:
        rc = h.hipStreamWriteValue32(stream.cuda_stream, ctypes.c_void_p(ptr),
                                     ctypes.c_uint(val), 0)
    else:
        h.hipStreamWriteValue64.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.c_uint64, ctypes.c_uint]
        rc = h.hipStreamWriteValue64(stream.cuda_stream, ctypes.c_void_p(ptr),
                                     ctypes.c_uint64(val), 0)
    assert rc == 0, f"hipStreamWriteValue rc={rc}"


@triton.jit(do_not_specialize=["sync_ptr", "expected"])
def barrier_spin_kernel(sync_ptr, expected, n_ranks: tl.constexpr):
    # Spin until EVERY rank's slot reached `expected` (waiting only on
    # your own slot would be a no-op barrier). The ARRIVE writes are
    # host-issued engine memcpys (SymmBarrier.round_barrier) that precede
    # this kernel in stream order. Hybrid spin, same rationale as the AG
    # consumer flag wait: a tight atomic RMW spin keeps the line busy and
    # can block the incoming engine write for tens of ms, while volatile
    # loads alone can read stale DRAM forever. Volatile batches delay and
    # give the write us-wide commit windows; the rare atomic probe at the
    # L2 coherence point guarantees forward progress.
    for r in range(n_ranks):
        fr = (sync_ptr + r * 4).to(tl.pointer_type(tl.int32))
        v = tl.atomic_add(fr, 0)
        while v < expected:
            for _ in range(128):
                v = tl.load(fr, volatile=True)
            if v < expected:
                v = tl.atomic_add(fr, 0)
    tl.debug_barrier()


class SymmBarrier:
    """Monotonic device-side barrier over the dushmem symmetric heap.

    round_barrier() runs on the CURRENT torch stream (wrap the call in
    `with torch.cuda.stream(s):` like the upstream barrier_all_on_stream):
    host-issued engine memcpys carry the arrive value to every PE's slot,
    then a cached spin kernel (LOCAL atomics only) waits for all PEs.

    The arrive writes used to be device-initiated dushmem atomic_set ops
    inside one kernel; on this stack those occasionally stall for tens of
    ms (S4 bench outliers; same slow path as the ~19 MB/s device
    transfers), which is why they moved to the copy engine."""

    def __init__(self, n_ranks):
        s = dcu_state()
        self.n_ranks = n_ranks
        self.ptr = s["host"].malloc(4 * n_ranks)  # int32 flag per rank
        self.round = 0
        self._ck = None
        self._peer_slot = [s["host"].ptr(self.ptr, pe)
                           for pe in range(n_ranks)]
        self._stage = torch.empty(1, dtype=torch.int32, device="cuda")

    def _kernel(self):
        s = dcu_state()
        if self._ck is None:
            ck = barrier_spin_kernel.warmup(
                self.ptr, 1, self.n_ranks, grid=(1,), num_warps=1)
            ck._init_handles()
            self._ck = ck
        return self._ck

    def round_barrier(self, my_rank):
        """Must be called by ALL ranks; each call = one barrier round."""
        self.round += 1
        ck = self._kernel()
        stream = torch.cuda.current_stream().cuda_stream
        self._stage.fill_(self.round)  # device fill, ordered on `stream`
        for pe in range(self.n_ranks):
            memcpy_async(self._peer_slot[pe] + my_rank * 4,
                         self._stage.data_ptr(), 4, stream)
        ck[(1, 1, 1)](self.ptr, self.round, self.n_ranks)

    def sync(self):
        torch.cuda.synchronize()


def barrier_all_on_stream(rank, num_ranks, barrier: SymmBarrier):
    barrier.round_barrier(rank)
