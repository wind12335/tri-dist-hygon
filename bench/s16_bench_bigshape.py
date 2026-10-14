"""S16 big-shape A/B (2x BW1000): the user-named MLP shape on the AG path.

M8192 N28672 K8192 fp16 -> M_pr=4096/rank, N_pr=28672/rank, K=8192.
Per-rank GEMM = full gathered M x own N x K = 3.85 TFLOP (the S10.3 class).

Arms (same process, GPU-event timed, identical tensors):
  s14        gemm_ag_tr_op           -- the tuned pull pipeline, no I1 layers
  i1_v0      i1_ag_tr_op, verify OFF -- predictor path without the fix tax
  i1_v1      i1_ag_tr_op, verify ON  -- current full stack (fix + table)
  push_path  all_gather_gemm S11 op  -- single-chunk MPC push line (64MB ops)
  gemm_t     torch.matmul vendor GEMM alone (hipBLAS)
  dist_ag    RCCL all_gather_into_tensor alone
  dist_seq   RCCL all_gather + vendor GEMM serial  <-- the baseline to beat

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s16_bench_bigshape.py
"""
import os
import sys
import json
import time
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tcache_rank{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import ag_tr_pred as i1  # noqa: E402
from triton_dist.kernels.dcu import gemm_ag_tr as agtr  # noqa: E402
from triton_dist.kernels.dcu import all_gather_gemm as ags4  # noqa: E402

DTYPE = torch.float16
SCALE = 0.02
M_PR, K, N = 4096, 8192, 28672
M = M_PR * size
WARMUP, ITERS = 2, 5
OUT = os.path.join(REPO, "bench", "results", "s16_bigshape_2card.json")

import torch.distributed as dist  # noqa: E402

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29617")
dist.init_process_group(backend="nccl", rank=rank, world_size=size)
_x = torch.ones(1, device="cuda")
for _ in range(3):
    dist.all_reduce(_x)
torch.cuda.synchronize()
print(f"[rank {rank}] dist(nccl/rccl) ready", flush=True)


def tloop(fn, tag=""):
    st = torch.cuda.current_stream()
    for w in range(WARMUP):
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        print(f"[rank {rank}] {tag} warmup{w} {time.time()-t0:.2f}s", flush=True)
    ts = []
    for i in range(ITERS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(st)
        fn()
        e.record(st)
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
        print(f"[rank {rank}] {tag} it{i} {ts[-1]:.2f} ms", flush=True)
    return ts


def stats(ts):
    return {"mean": statistics.mean(ts), "p50": statistics.median(ts),
            "min": min(ts)}


# ------------------------------------------------------------ tensors --
A_shards = []
for s_ in range(size):
    torch.manual_seed(5000 + s_)
    A_shards.append((torch.randn(M_PR, K, device="cuda") * SCALE).half())
a = A_shards[rank]
A_full = torch.cat(A_shards, dim=0)
del A_shards
torch.manual_seed(7100 + rank)
b = (torch.randn(N, K, device="cuda") * SCALE).half()
c = torch.empty(M, N, dtype=DTYPE, device="cuda")
with torch.no_grad():
    ref = (A_full.float() @ b.float().t())
print(f"[rank {rank}] tensors ready (A {A_full.numel()*2/1e6:.0f}MB, "
      f"B {b.numel()*2/1e6:.0f}MB, C {c.numel()*2/1e6:.0f}MB)", flush=True)

RES = {"env": {"shape": "M8192_N28672_K8192_2card", "dtype": "fp16",
               "M_pr": M_PR, "K": K, "N": N,
               "warmup/iters": f"{WARMUP}/{ITERS}"}}


def check(tag, md_hist=None):
    torch.cuda.synchronize()
    md = (c.float() - ref).abs().max().item()
    assert md < 1e-2, f"{tag} correctness FAIL md={md}"
    print(f"[rank {rank}] {tag} corr md={md:.2e}", flush=True)
    return md


# ------------------------------------------------- tr-path contexts --
ictx = i1.create_i1_context(
    max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
    rank=rank, num_ranks=size, mpi=mpi, comm=comm)
ag = ictx.ag
rot_packed = sum(((rank + i) % size) << (4 * i) for i in range(size))
print(f"[rank {rank}] tr ctx: chunk_rows={ag.chunk_rows} "
      f"n_chunks={ag.num_chunks} lanes={ag.comm_lanes}", flush=True)

RES["chunk_rows"] = ag.chunk_rows
RES["n_chunks"] = ag.num_chunks

# --- arm: s14 base
agtr.gemm_ag_tr_op(a, b, c, ictx.ag)
check("s14")
RES["s14"] = stats(tloop(lambda: agtr.gemm_ag_tr_op(a, b, c, ictx.ag), "s14"))
print(f"[rank {rank}] s14   {RES['s14']['p50']:.2f} p50 ms", flush=True)

# --- arm: i1 verify OFF (module-global read at call time)
i1.I1_VERIFY = False


def call_i1():
    i1.i1_table_write(ictx, 0, 0, 0, rot_packed, ag.num_chunks)
    i1.i1_ag_tr_op(a, b, c, ictx, with_observer=True)


call_i1()
check("i1_v0")
RES["i1_v0"] = stats(tloop(call_i1, "i1_v0"))
print(f"[rank {rank}] i1_v0 {RES['i1_v0']['p50']:.2f} p50 ms", flush=True)

# --- arm: i1 verify ON (the landed fix, default posture)
i1.I1_VERIFY = True
hits0 = ictx.verify_hits
call_i1()
check("i1_v1")
RES["i1_v1"] = stats(tloop(call_i1, "i1_v1"))
RES["verify_hits"] = ictx.verify_hits - hits0
RES["verify_tax_ms"] = RES["i1_v1"]["p50"] - RES["i1_v0"]["p50"]
print(f"[rank {rank}] i1_v1 {RES['i1_v1']['p50']:.2f} p50 ms "
      f"(verify tax {RES['verify_tax_ms']:+.2f} ms, "
      f"repairs={RES['verify_hits']})", flush=True)

# --- arm: consumer-GEMM-only decomposition (component="gemm")
RES["gemm_triton"] = stats(
    tloop(lambda: agtr.gemm_ag_tr_op(a, b, c, ictx.ag, component="gemm"), "gemm_triton"))
print(f"[rank {rank}] gemm_triton {RES['gemm_triton']['p50']:.2f} p50 ms",
      flush=True)

# --- arm: vendor GEMM alone
RES["gemm_torch"] = stats(
    tloop(lambda: torch.matmul(A_full, b.t(), out=c), "gemm_torch"))
print(f"[rank {rank}] gemm_torch {RES['gemm_torch']['p50']:.2f} p50 ms",
      flush=True)
tf = 2 * M * N * K / 1e12
print(f"[rank {rank}] TFLOPS: triton {tf/RES['gemm_triton']['p50']:.0f} "
      f"vendor {tf/RES['gemm_torch']['p50']:.0f}", flush=True)

# --- RCCL arms
Ag_buf = torch.empty(M, K, dtype=DTYPE, device="cuda")
RES["dist_ag"] = stats(tloop(lambda: dist.all_gather_into_tensor(Ag_buf, a), "dist_ag"))
print(f"[rank {rank}] rccl_ag {RES['dist_ag']['p50']:.2f} p50 ms "
      f"({M_PR*K*2*(size-1)/1e6/RES['dist_ag']['p50']:.1f} GB/s out)",
      flush=True)


def dist_seq():
    dist.all_gather_into_tensor(Ag_buf, a)
    torch.matmul(Ag_buf, b.t(), out=c)


RES["dist_seq"] = stats(tloop(dist_seq, "dist_seq"))
check("dist_seq")
print(f"[rank {rank}] RCCL ag+vendorGEMM {RES['dist_seq']['p50']:.2f} p50 ms",
      flush=True)

# --- arm: the S11 push path (single-chunk MPC = M_pr, 64MB engine ops)
ctx4 = ags4.create_ag_gemm_intra_node_context(
    M, K, DTYPE, DTYPE, rank, size, mpi, comm, M_PER_CHUNK=M_PR)
s4c = torch.empty(M, N, dtype=DTYPE, device="cuda")


def call_s4():
    ags4.ag_gemm_intra_node_op(a, b, s4c, ctx4)


call_s4()
torch.cuda.synchronize()
md = (s4c.float() - ref).abs().max().item()
assert md < 1e-2, f"push_path FAIL md={md}"
RES["push_path"] = stats(tloop(call_s4, "push_path"))
print(f"[rank {rank}] push_path(S11,mpc=M_pr) {RES['push_path']['p50']:.2f} "
      f"p50 ms corr {md:.2e}", flush=True)

# --- summary
base = RES["dist_seq"]["p50"]
for k in ("s14", "i1_v0", "i1_v1", "push_path"):
    r = RES[k]
    r["vs_rccl_seq"] = base / r["p50"]
    r["p50"] = round(r["p50"], 3)
print(f"[rank {rank}] === M8192 N28672 K8192 (p50 ms, 2 ranks) ===",
      flush=True)
print(f"[rank {rank}] RCCL ag+vendorGEMM (baseline): {base:.2f}", flush=True)
for k in ("s14", "i1_v0", "i1_v1", "push_path"):
    print(f"[rank {rank}]   {k:10s} {RES[k]['p50']:7.2f}  "
          f"({base/RES[k]['p50']:.2f}x of baseline)", flush=True)

if rank == 0:
    with open(OUT, "w") as f:
        json.dump(RES, f, indent=1)
    print(f"wrote {OUT}", flush=True)
dist.destroy_process_group()
print(f"[rank {rank}] DONE", flush=True)
