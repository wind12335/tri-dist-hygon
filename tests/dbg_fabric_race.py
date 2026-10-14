"""Minimal fabric-race probe for the skew-torn-read bug (I1 bisect, Oct 14).

Replicates the S14/I1 pull path with everything else stripped away:

  rank0 (source), per iter: [fill staging tensor with iter pattern]
      -> [engine copy staging -> MY symmetric ws] -> [flag=it+1 to
      rank1's cell]           (FLAG_STYLE: kernel | engine)
  rank1 (puller), per iter: [GTE wait flag >= it+1]
      -> [engine copy rank0's ws -> MY local scratch]
      -> [checksum kernel: scratch == pattern(it)?]
      (PULL_STYLE: engine | kernel)

SKEW_US: rank0 host-sleeps before enqueueing (rank1's wait blocks and
chases the flag at hardware speed -- the condition under which the real
op corrupts; lockstep = the condition under which it never does).

mpirun --allow-run-as-root -np 2 ... -x FLAG_STYLE=kernel -x PULL_STYLE=engine -x SKEW_US=400
"""
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_fbr_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2

FLAG_STYLE = os.environ.get("FLAG_STYLE", "kernel")
PULL_STYLE = os.environ.get("PULL_STYLE", "engine")
SKEW_US = float(os.environ.get("SKEW_US", "400"))
ITERS = int(os.environ.get("ITERS", "30"))

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

from triton_dist.kernels.dcu.common_ops import (  # noqa: E402
    dcu_state, hip, init_dcu, memcpy_async, set_signal)
from triton_dist.kernels.dcu.gemm_ar_rd import (  # noqa: E402
    _launch_flag_xchg, _wait_ticket)

init_dcu(mpi, comm, rank, size)
host = dcu_state()["host"]

WORDS = 512 * 1024  # 2 MB fp32
ws_ptr = host.malloc(4 * WORDS)          # symmetric: data plane
flag_ptr = host.malloc(4)                # symmetric: one int32 cell
h = hip()
h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
h.hipMemset.restype = ctypes.c_int
assert h.hipMemset(ctypes.c_void_p(ws_ptr), 0, 4 * WORDS) == 0
assert h.hipMemset(ctypes.c_void_p(flag_ptr), 0, 4) == 0
peer_ws = host.ptr(ws_ptr, 1 - rank)     # peer's ws as I address it
peer_flag = host.ptr(flag_ptr, 1 - rank)

staging = torch.zeros(WORDS, dtype=torch.float32, device="cuda")
scratch = torch.zeros(WORDS, dtype=torch.float32, device="cuda")
bad_count = torch.zeros(1, dtype=torch.int32, device="cuda")
stream = torch.cuda.Stream()


@triton.jit(do_not_specialize=["x_ptr", "exp_ptr", "bad_ptr", "it", "n"])
def kernel_check(x_ptr, exp_ptr, bad_ptr, it, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    xp = x_ptr.to(tl.pointer_type(tl.float32))
    ep = exp_ptr.to(tl.pointer_type(tl.float32))
    bp = bad_ptr.to(tl.pointer_type(tl.int32))
    v = tl.load(xp + offs, mask=m)
    e = tl.load(ep + offs, mask=m)
    bad = tl.sum(((v != e) & m).to(tl.int32))
    tl.atomic_add(bp, bad)


@triton.jit(do_not_specialize=["dst_ptr", "src_ptr", "n"])
def kernel_pull(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    sp = src_ptr.to(tl.pointer_type(tl.float32))
    dp = dst_ptr.to(tl.pointer_type(tl.float32))
    v = tl.load(sp + offs, mask=m, volatile=True)
    tl.store(dp + offs, v, mask=m)


if rank == 0:
    print(f"[rank 0] fabric probe FLAG={FLAG_STYLE} PULL(peer)={PULL_STYLE} "
          f"SKEW={SKEW_US}us ITERS={ITERS}", flush=True)

fails = 0
t_pull_ms = []
for it in range(1, ITERS + 1):
    barrier()
    if rank == 0:
        if SKEW_US > 0:
            time.sleep(SKEW_US / 1e6)
        with torch.cuda.stream(stream):
            staging.fill_(float(it))        # pattern, on the same stream
            memcpy_async(ws_ptr, staging.data_ptr(), 4 * WORDS,
                         stream.cuda_stream)  # engine copy on stream
            if FLAG_STYLE == "kernel":
                _launch_flag_xchg(peer_flag, it)
            else:
                set_signal(peer_flag, it, stream)
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            _wait_ticket(flag_ptr, it, stream, "stream")
            if PULL_STYLE == "engine":
                memcpy_async(scratch.data_ptr(), peer_ws, 4 * WORDS,
                             stream.cuda_stream)
            else:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record(stream)
                kernel_pull[(WORDS // 4096,)](scratch.data_ptr(), peer_ws,
                                             WORDS, BLOCK=4096)
                ev1.record(stream)
            bad_count.zero_()
            staging.fill_(float(it))
            kernel_check[(WORDS // 4096,)](scratch.data_ptr(),
                                           staging.data_ptr(),
                                           bad_count.data_ptr(), it, WORDS,
                                           BLOCK=4096)
        torch.cuda.synchronize()
        bad = bad_count.item()
        if bad:
            fails += 1
            print(f"[rank 1] it={it} MISMATCH words={bad}", flush=True)
        if PULL_STYLE == "kernel":
            t_pull_ms.append(ev0.elapsed_time(ev1))

if PULL_STYLE == "kernel" and t_pull_ms:
    t = sorted(t_pull_ms)
    mbps = (4 * WORDS / 1e6) / (t[len(t) // 2] / 1e3)
    print(f"[rank 1] kernel-pull 2MB: p50={t[len(t)//2]:.3f}ms "
          f"-> {mbps:.1f} MB/s", flush=True)
print(f"[rank {rank}] PROBE RESULT fails={fails}/{ITERS}", flush=True)
host.finalize()
barrier()
mpi.MPI_Finalize()
os._exit(1 if fails else 0)
