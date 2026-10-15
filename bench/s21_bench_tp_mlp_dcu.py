"""S21: upstream-style TP-MLP bench on the DCU ops (credibility bench for
the 8-card session). Ports the SEMANTICS of triton-dist's
test/nvidia/test_tp_mlp.py (ag_rs / gemm_ar modes), benchmark/
bench_tp_mlp.py's decode->prefill M sweep (512*2^i), and
bench_allgather_gemm.py's total/ag/gemm decomposition -- on OUR op-level
APIs with OUR bootstrap. No HF / no layers/ on our side: shapes are
hardcoded from upstream test/utils.py LAYER_CONFIGS (I = intermediate,
K = hidden):

  LLaMA-7B        I=11008  K=4096    R=2 only (2I/R ragged at R=8)
  LLaMA-3.1-70B   I=28672  K=8192    128-aligned at R=2 AND R=8
  LLaMA-3.1-405B  I=53248  K=16384   128-aligned at R=2 AND R=8
  Qwen2-72B       I=29568  K=8192    R=2 only (2I/R=7392 ragged at R=8)

TP layouts (upstream semantics -- NOT our microbench B-replicated form;
B is truly N-sharded here):
  ag_rs:
    x_loc  [M/R, K]        M-sharded input
    Wgu_loc [2I/R, K]      gate_up weight shard: [I/R gate | I/R up]
    h   = AG-GEMM(x, Wgu)          -> [M, 2I/R]
    h2  = silu(gate)*up            -> [M, I/R]
    y   = RS-GEMM(h2, Wd_loc [K, I/R]) -> [M/R, K]
  gemm_ar:
    x_k  [M, K/R]          K-sharded input
    WguA [2I, K/R];  h = AR-GEMM -> [M, 2I] on EVERY rank (global layout)
    h2 = silu*up -> [M, I]; a2 = h2[:, rank slice] -> [M, I/R]
    y   = AR-GEMM(a2, WdA [K, I/R]) -> [M, K] on every rank
  (a2 needs a .contiguous() copy in ours -- our op asserts contiguous A;
   torch's matmul eats the strided view directly. ~0.1ms, documented.)

Fairness notes (documented deviations from upstream):
  * upstream CUDA-graph-captures BOTH arms; our host-recipe ops do
    per-round host work (tickets/flags) and cannot be captured -- BOTH
    arms run eager with per-iter event sync (the S18/S20 protocol).
  * mpirun/pydushmem bootstrap (symmetric-heap requirement), torch
    baseline on the nccl process group.
  * activation = torch silu-mul, IDENTICAL in both arms (the ops under
    test are the fused comm+GEMM pair, not the act).

Correctness: golden assembled once per (mode, model, M) via nccl
gathers + torch fp16 matmul (fp32 accum) + fp32 act, chunk-consistent
with the rank-blocked gate/up layout; both arms checked against it
(md, tol 2e-2 for two chained GEMMs). Contexts are created PER M
(chunk geometry is m_pr-dependent), finalized before the next cell --
one live ctx of each type at a time (S18 wedge law).

run:  bash bench/s21_run.sh
  S21_MODES=ag_rs,gemm_ar S21_MODELS=LLaMA-3.1-70B,LLaMA-3.1-405B \
  S21_MS=512,1024,2048,4096,8192 bash bench/s21_run.sh
"""
import json
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault(
    "TRITON_CACHE_DIR",
    f"/tmp/tc_s21_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert 2 <= size <= 32

MODES = [m for m in os.environ.get("S21_MODES", "ag_rs").split(",") if m]
MODELS = [m for m in os.environ.get(
    "S21_MODELS", "LLaMA-3.1-70B").split(",") if m]
MS = [int(x) for x in os.environ.get(
    "S21_MS", "512,1024,2048,4096,8192").split(",") if x]
ITERS = int(os.environ.get("S21_ITERS", "20"))
WARMUP = int(os.environ.get("S21_WARMUP", "5"))
AR_CHUNK = int(os.environ.get("S21_AR_CHUNK", "256"))
OUTDIR = os.path.join(REPO, "bench", "results")

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int


def align():
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.distributed as dist  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu.gemm_ag_tr import (  # noqa: E402
    create_gemm_ag_tr_context, finalize_gemm_ag_tr_context,
    gemm_ag_tr_op)
from triton_dist.kernels.dcu.gemm_rs_v5 import (  # noqa: E402
    create_gemm_rs_v5_context, finalize_gemm_rs_v5_context,
    gemm_rs_v5_op)
from triton_dist.kernels.dcu.gemm_ar_rd import (  # noqa: E402
    create_gemm_ar_rd_context, finalize_gemm_ar_rd_context,
    gemm_ar_rd_op)

DTYPE = torch.float16
SCALE = 0.02
TOL = 2e-2
LAYER_CONFIGS = {   # I = intermediate, K = hidden (upstream test/utils.py)
    "LLaMA-7B": {"I": 11008, "K": 4096},
    "LLaMA-3.1-70B": {"I": 28672, "K": 8192},
    "LLaMA-3.1-405B": {"I": 53248, "K": 16384},
    "Qwen2-72B": {"I": 29568, "K": 8192},
}

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29741")
dist.init_process_group(backend="nccl", rank=rank, world_size=size)

ev_t0 = torch.cuda.Event(enable_timing=True)
ev_t1 = torch.cuda.Event(enable_timing=True)
ev_done = torch.cuda.Event()


def timed(fn):
    """First call (compile/init) + warmup + event-timed p50, per-iter
    completion sync (S18/S20 protocol; both arms identical treatment)."""
    fn()
    torch.cuda.synchronize()
    st = torch.cuda.current_stream()
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        ev_t0.record(st)
        fn()
        ev_t1.record(st)
        ev_done.record(st)
        ev_done.synchronize()
        ts.append(ev_t0.elapsed_time(ev_t1))
    ts.sort()
    return round(ts[len(ts) // 2], 4)


def md_of(t, golden):
    return round((t.float() - golden.float()).abs().max().item(), 6)


def rand(*shape):
    return (torch.randn(*shape, device="cuda") * SCALE).half()


# ---------------------------------------------------------------- ag_rs
def run_ag_rs(model, I, K, M):
    R = size
    m_pr = M // R
    I_R = I // R

    torch.manual_seed(7300 + rank)
    x_loc = rand(m_pr, K)
    Wgu_loc = rand(2 * I_R, K)
    Wd_loc = rand(K, I_R)

    # ---- golden (nccl assembly; Wgu shards are rank-BLOCKED gate/up,
    # so the act splits EACH 2I/R block, not the global 2I)
    x_full = torch.empty(M, K, dtype=DTYPE, device="cuda")
    dist.all_gather_into_tensor(x_full, x_loc)
    Wgu_full = torch.empty(2 * I, K, dtype=DTYPE, device="cuda")
    dist.all_gather_into_tensor(Wgu_full, Wgu_loc)
    h1 = x_full @ Wgu_full.t()                       # [M, 2I]
    h2g = torch.empty(M, I, dtype=DTYPE, device="cuda")
    for r in range(R):
        b0 = r * 2 * I_R
        seg = h1[:, b0:b0 + 2 * I_R]
        h2g[:, r * I_R:(r + 1) * I_R] = (
            F.silu(seg[:, :I_R].float()) * seg[:, I_R:].float()).half()
    del h1
    Wd_list = [torch.empty(K, I_R, dtype=DTYPE, device="cuda")
               for _ in range(R)]
    dist.all_gather(Wd_list, Wd_loc)
    Wd_full = torch.cat(Wd_list, dim=1)              # [K, I]
    del Wd_list
    y_full = h2g @ Wd_full.t()                       # [M, K]
    del h2g, Wd_full, x_full, Wgu_full
    golden = y_full[rank * m_pr:(rank + 1) * m_pr].clone()
    del y_full
    torch.cuda.empty_cache()
    align()

    # ---- buffers + contexts (per-M; one live of each type)
    x_full = torch.empty(M, K, dtype=DTYPE, device="cuda")
    h_buf = torch.empty(M, 2 * I_R, dtype=DTYPE, device="cuda")
    partial = torch.empty(M, K, dtype=DTYPE, device="cuda")
    y_t = torch.empty(m_pr, K, dtype=DTYPE, device="cuda")
    y_o = torch.empty(m_pr, K, dtype=DTYPE, device="cuda")
    agctx = create_gemm_ag_tr_context(
        max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=R, mpi=mpi, comm=comm)
    rsctx = create_gemm_rs_v5_context(
        max_M=M, N=K, output_dtype=DTYPE, rank=rank, num_ranks=R,
        mpi=mpi, comm=comm)

    def act(c):
        return F.silu(c[:, :I_R]) * c[:, I_R:]

    def torch_e2e():
        dist.all_gather_into_tensor(x_full, x_loc)
        torch.matmul(x_full, Wgu_loc.t(), out=h_buf)
        h2 = act(h_buf)
        torch.matmul(h2, Wd_loc.t(), out=partial)
        dist.reduce_scatter_tensor(y_t, partial)

    def ours_e2e():
        gemm_ag_tr_op(x_loc, Wgu_loc, h_buf, agctx)
        h2 = act(h_buf)
        gemm_rs_v5_op(h2, Wd_loc, y_o, rsctx)

    ours_e2e()
    torch.cuda.synchronize()
    md = md_of(y_o, golden)
    torch_e2e()
    torch.cuda.synchronize()
    md_t = md_of(y_t, golden)

    # fixed RS input for the sub-op arms (identical for both arms)
    gemm_ag_tr_op(x_loc, Wgu_loc, h_buf, agctx)
    h2_static = act(h_buf).contiguous()
    torch.cuda.synchronize()

    def ag_torch():
        dist.all_gather_into_tensor(x_full, x_loc)
        torch.matmul(x_full, Wgu_loc.t(), out=h_buf)

    def rs_torch():
        torch.matmul(h2_static, Wd_loc.t(), out=partial)
        dist.reduce_scatter_tensor(y_t, partial)

    def ag_ours():
        gemm_ag_tr_op(x_loc, Wgu_loc, h_buf, agctx)

    def rs_ours():
        gemm_rs_v5_op(h2_static, Wd_loc, y_o, rsctx)

    res = {"md_ours": md, "md_torch": md_t, "arms": {}}
    for tag, fn in (("e2e_torch", torch_e2e), ("ag_torch", ag_torch),
                    ("rs_torch", rs_torch), ("e2e_ours", ours_e2e),
                    ("ag_ours", ag_ours), ("rs_ours", rs_ours)):
        align()
        res["arms"][tag] = timed(fn)
        print(f"[rank {rank}] ag_rs {model} M{M} {tag:10s} "
              f"{res['arms'][tag]:8.3f} ms", flush=True)

    finalize_gemm_ag_tr_context(agctx)
    finalize_gemm_rs_v5_context(rsctx)
    del x_full, h_buf, partial, h2_static, y_t, y_o, golden
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------- gemm_ar
def run_gemm_ar(model, I, K, M):
    R = size
    I_R = I // R
    K_R = K // R

    torch.manual_seed(7400 + rank)
    x_k = rand(M, K_R)
    WguA = rand(2 * I, K_R)
    WdA = rand(K, I_R)

    # ---- golden (K-sharded x and gate_up; global gate|up layout)
    xk = [torch.empty(M, K_R, dtype=DTYPE, device="cuda") for _ in range(R)]
    dist.all_gather(xk, x_k)
    x_full = torch.cat(xk, dim=1)
    del xk
    wg = [torch.empty(2 * I, K_R, dtype=DTYPE, device="cuda")
          for _ in range(R)]
    dist.all_gather(wg, WguA)
    WguF = torch.cat(wg, dim=1)
    del wg
    h1 = x_full @ WguF.t()                           # [M, 2I]
    del x_full, WguF
    h2g = (F.silu(h1[:, :I].float()) * h1[:, I:].float()).half()
    del h1
    wd = [torch.empty(K, I_R, dtype=DTYPE, device="cuda") for _ in range(R)]
    dist.all_gather(wd, WdA)
    WdF = torch.cat(wd, dim=1)
    del wd
    golden = h2g @ WdF.t()                           # [M, K] full
    del h2g, WdF
    torch.cuda.empty_cache()
    align()

    h1b = torch.empty(M, 2 * I, dtype=DTYPE, device="cuda")
    y_t = torch.empty(M, K, dtype=DTYPE, device="cuda")
    y_o = torch.empty(M, K, dtype=DTYPE, device="cuda")
    sl0, sl1 = rank * I_R, (rank + 1) * I_R
    arctx1 = create_gemm_ar_rd_context(
        max_M=M, N=2 * I, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=R, mpi=mpi, comm=comm, chunk_rows=AR_CHUNK)
    arctx2 = create_gemm_ar_rd_context(
        max_M=M, N=K, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=R, mpi=mpi, comm=comm, chunk_rows=AR_CHUNK)

    def act(c):
        return F.silu(c[:, :I]) * c[:, I:]

    def torch_e2e():
        p1 = x_k @ WguA.t()                          # [M, 2I] partial
        dist.all_reduce(p1)
        h2 = act(p1)
        p2 = h2[:, sl0:sl1] @ WdA.t()                # strided A ok for matmul
        dist.all_reduce(p2)
        y_t.copy_(p2)

    def ours_e2e():
        gemm_ar_rd_op(x_k, WguA, h1b, arctx1)
        h2 = act(h1b)
        a2 = h2[:, sl0:sl1].contiguous()             # op asserts contiguous A
        gemm_ar_rd_op(a2, WdA, y_o, arctx2)

    ours_e2e()
    torch.cuda.synchronize()
    md = md_of(y_o, golden)
    torch_e2e()
    torch.cuda.synchronize()
    md_t = md_of(y_t, golden)

    # fixed mid-activation input for the single-op arms
    gemm_ar_rd_op(x_k, WguA, h1b, arctx1)
    a2_static = act(h1b)[:, sl0:sl1].contiguous()
    torch.cuda.synchronize()

    def ar1_torch():
        p1 = x_k @ WguA.t()
        dist.all_reduce(p1)

    def ar1_ours():
        gemm_ar_rd_op(x_k, WguA, h1b, arctx1)

    def ar2_torch():
        p2 = a2_static @ WdA.t()
        dist.all_reduce(p2)

    def ar2_ours():
        gemm_ar_rd_op(a2_static, WdA, y_o, arctx2)

    res = {"md_ours": md, "md_torch": md_t, "arms": {}}
    for tag, fn in (("e2e_torch", torch_e2e), ("ar1_torch", ar1_torch),
                    ("ar2_torch", ar2_torch), ("e2e_ours", ours_e2e),
                    ("ar1_ours", ar1_ours), ("ar2_ours", ar2_ours)):
        align()
        res["arms"][tag] = timed(fn)
        print(f"[rank {rank}] gemm_ar {model} M{M} {tag:10s} "
              f"{res['arms'][tag]:8.3f} ms", flush=True)

    finalize_gemm_ar_rd_context(arctx1)
    finalize_gemm_ar_rd_context(arctx2)
    del h1b, y_t, y_o, a2_static, golden
    torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------- driver
def legal(mode, I, K, M):
    R = size
    if M % R or M % 128 or (M // R) % 64:
        return f"M{M} violates M%R / M%128 / (M/R)%64 at R={R}"
    if mode == "ag_rs":
        if (2 * I // R) % 128 or I // R % 64 or K % 128 or K % 16:
            return (f"2I/R={(2 * I // R)}%128 or I/R={I // R}%64 "
                    f"ragged at R={R}")
    else:
        if (2 * I) % 128 or K % 128 or K // R % 64 or I // R % 64:
            return (f"2I%128 or K/R={K // R}%64 or I/R={I // R}%64 "
                    f"ragged at R={R}")
        if R & (R - 1):
            return f"recursive doubling needs 2^k ranks, got {R}"
    return None


for mode in MODES:
    for model in MODELS:
        cfg = LAYER_CONFIGS[model]
        for M in MS:
            tag = f"s21_mlp_{mode}_{model.replace('.', '_')}_M{M}"
            why = legal(mode, cfg["I"], cfg["K"], M)
            out = {"mode": mode, "model": model, "M": M, "I": cfg["I"],
                   "K": cfg["K"], "rank": rank, "R": size}
            if why:
                out.update(status="SKIP", reason=why)
                print(f"[rank {rank}] SKIP {tag}: {why}", flush=True)
            else:
                try:
                    res = (run_ag_rs if mode == "ag_rs" else run_gemm_ar)(
                        model, cfg["I"], cfg["K"], M)
                    out.update(status="ok", **res)
                    a = res["arms"]
                    sp = a.get("e2e_torch", 0) / max(a.get("e2e_ours", 1), 1e-9)
                    print(f"[rank {rank}] CELL DONE {tag}: "
                          f"torch {a.get('e2e_torch')} vs ours "
                          f"{a.get('e2e_ours')} ms (x{sp:.2f}) "
                          f"md {res['md_ours']}", flush=True)
                except Exception:
                    out.update(status="ERR", tb=traceback.format_exc()[-2000:])
                    print(f"[rank {rank}] {tag} ERR:\n{out['tb']}", flush=True)
            with open(os.path.join(OUTDIR, f"{tag}_rank{rank}.json"), "w") as f:
                json.dump(out, f, indent=1)
            align()

align()
mpi.MPI_Finalize()
print(f"[rank {rank}] S21 DONE", flush=True)
os._exit(0)
