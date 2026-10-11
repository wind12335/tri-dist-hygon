"""S4 transport micro-benchmark: why is the dushmem device push ~19 MB/s?

Paths (16 MB per direction, full duplex, 2 ranks):
  D1 putmem_signal (blocking) per chunk           -- current AG producer
  D2 putmem_nbi xN + one quiet                     -- gemm_rs scatter pattern
  D3 putmem_nbi + uint64_atomic_set per chunk, quiet at end -- candidate fix
  D4 getmem_nbi xN + one quiet (pull)              -- fallback design
  H1 host.putmem (blocking, host loop)
  H2 hipMemcpyAsync into host.ptr(peer_va)         -- upstream copy-engine path

Run twice and compare:
  default:                                                            (this file)
  DUSHMEM_DISABLE_LOCAL_ONLY_PROXY=1:
mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  [-x DUSHMEM_DISABLE_LOCAL_ONLY_PROXY=1] \
  python3 s4_transport_probe.py
"""
import os
import sys
import time
import statistics

REPO = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tcache_rank{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402
import numpy as np  # noqa: E402

import pydushmem.host as pHost  # noqa: E402
from pydushmem import device as dushmem_dev  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2
peer = rank ^ 1

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
hip.hipMemcpy.restype = ctypes.c_int
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int]
hip.hipMemcpyAsync.restype = ctypes.c_int
hip.hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu.common_ops import init_dcu, dcu_state  # noqa: E402

init_dcu(mpi, comm, rank, size)
host = dcu_state()["host"]
extern_libs = dcu_state()["extern_libs"]

TOTAL = 16 * 1024 * 1024  # bytes per direction
CHUNKS = [(TOTAL, 1), (4 * 1024 * 1024, 4), (1024 * 1024, 16),
          (256 * 1024, 64)]
ITERS = 8

sym = host.malloc(TOTAL)          # symmetric data buffer
flags = host.malloc(8 * 64)       # symmetric signal slots
peer_va = host.ptr(sym, peer)     # peer's copy of `sym`, PE-local VA

torch.manual_seed(1234 + rank)
src = (torch.randn(TOTAL // 2, device="cuda") * 0.02).half()
src_bytes = src.contiguous()


def hsync():
    torch.cuda.synchronize()
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


def verify(tag):
    """Check the 16MB in my sym buffer equals the PEER's src pattern."""
    chk = torch.empty(TOTAL // 2, dtype=torch.float16, device="cuda")
    assert hip.hipMemcpy(chk.data_ptr(), ctypes.c_void_p(sym), TOTAL, 3) == 0
    torch.manual_seed(1234 + peer)  # regenerate the PEER's GPU-side pattern
    want = (torch.randn(TOTAL // 2, device="cuda") * 0.02).half()
    md = (chk.float() - want.float()).abs().max().item()
    print(f"[rank {rank}]   verify[{tag}] max_diff={md:.4f} "
          f"{'OK' if md < 1e-3 else 'BAD'}", flush=True)


# ------------------------------------------------------------- triton path --
@triton.jit(do_not_specialize=["dst", "src", "flags", "pe", "round", "total",
                               "chunk_bytes"])
def k_signal(dst, src, flags, pe, round, total, chunk_bytes):
    for i in range(total):
        dushmem_dev.dushmem_putmem_signal(
            dst + i * chunk_bytes, src + i * chunk_bytes, chunk_bytes,
            flags, round, 9, pe)


@triton.jit(do_not_specialize=["dst", "src", "flags", "pe", "round", "total",
                               "chunk_bytes"])
def k_nbi_quiet(dst, src, flags, pe, round, total, chunk_bytes):
    for i in range(total):
        dushmem_dev.dushmem_putmem_nbi(dst + i * chunk_bytes,
                                       src + i * chunk_bytes, chunk_bytes, pe)
    dushmem_dev.dushmem_quiet()


@triton.jit(do_not_specialize=["dst", "src", "flags", "pe", "round", "total",
                               "chunk_bytes"])
def k_nbi_atomic(dst, src, flags, pe, round, total, chunk_bytes):
    for i in range(total):
        dushmem_dev.dushmem_putmem_nbi(dst + i * chunk_bytes,
                                       src + i *chunk_bytes, chunk_bytes, pe)
        dushmem_dev.dushmem_uint64_atomic_set(flags + i * 8, round, pe)
    dushmem_dev.dushmem_quiet()


@triton.jit(do_not_specialize=["dst", "src", "flags", "pe", "round", "total",
                               "chunk_bytes"])
def k_get_quiet(dst, src, flags, pe, round, total, chunk_bytes):
    # UNUSED: getmem_nbi VMFaults on this stack (see note at KS)
    for i in range(total):
        dushmem_dev.dushmem_getmem_nbi(dst + i * chunk_bytes,
                                       src + i * chunk_bytes, chunk_bytes, pe)
    dushmem_dev.dushmem_quiet()


def cached(fn, name):
    ck = fn.warmup(sym, src_bytes.data_ptr(), flags, 0, 3, 16, 1048576,
                   grid=(1,), extern_libs=extern_libs, num_warps=1)
    ck._init_handles()
    host.cumodule_init(ck.module)
    return ck


KS = {"D1_signal": cached(k_signal, "d1"),
      "D2_nbi_quiet": cached(k_nbi_quiet, "d2"),
      "D3_nbi_atomic": cached(k_nbi_atomic, "d3")}
# D4 getmem_nbi REMOVED: KERNEL VMFault on this stack (probe_default run 1) --
# the device-side pull path is unusable; do not revisit.


def bench_device(tag, ck, chunk, n):
    def fn():
        ck[(1, 1, 1)](sym, src_bytes.data_ptr(), flags, peer, 3, n, chunk)
    ts = []
    for _ in range(2):
        fn()
    hsync()
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    hsync()
    verify(tag)
    return statistics.median(ts)


print(f"[rank {rank}] env DUSHMEM_DISABLE_LOCAL_ONLY_PROXY="
      f"{os.environ.get('DUSHMEM_DISABLE_LOCAL_ONLY_PROXY', '(unset)')}",
      flush=True)

for tag, ck in KS.items():
    for chunk, n in CHUNKS:
        ms = bench_device(tag, ck, chunk, n)
        print(f"[rank {rank}] {tag:14s} chunk={chunk//1024:6d}KB n={n:3d} "
              f"median={ms:9.3f} ms  -> {TOTAL / ms / 1e3:8.2f} MB/s",
              flush=True)

# host blocking putmem
for chunk, n in CHUNKS:
    ts = []
    for _ in range(2):
        host.putmem(sym, src_bytes.data_ptr(), TOTAL, peer)
    hsync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        host.putmem(sym, src_bytes.data_ptr(), TOTAL, peer)
    hsync()
    ms = (time.perf_counter() - t0) / ITERS * 1e3
    verify("H1_putmem")
    print(f"[rank {rank}] H1_host_put   (single {TOTAL // 1024}KB blocking) "
          f"median={ms:9.3f} ms -> {TOTAL / ms / 1e3:8.2f} MB/s", flush=True)
    break  # host put is one blocking call; chunking N/A at this layer

# copy engine into peer VA
stream = torch.cuda.current_stream().cuda_stream
for chunk, n in CHUNKS:
    def fn():
        for i in range(n):
            assert hip.hipMemcpyAsync(ctypes.c_void_p(peer_va + i * chunk),
                                      ctypes.c_void_p(
                                          src_bytes.data_ptr() + i * chunk),
                                      chunk, 3, ctypes.c_void_p(stream)) == 0
    for _ in range(2):
        fn()
    hsync()
    ts = []
    for _ in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ms = statistics.median(ts)
    hsync()
    verify("H2_memcpy")
    print(f"[rank {rank}] H2_memcpyAsync chunk={chunk//1024:6d}KB n={n:3d} "
          f"median={ms:9.3f} ms -> {TOTAL / ms / 1e3:8.2f} MB/s", flush=True)

print(f"[rank {rank}] TRANSPORT PROBE DONE", flush=True)
host.finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
