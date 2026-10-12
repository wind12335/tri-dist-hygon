"""S9 benchmark: all four fused comm-GEMM ops on the 8-card node vs RCCL
(same methodology as the 2-card s4/s6/s7 benches; world-size generic, run
with -np 8).

Sections / baselines ("dist_seq" = vendor GEMM + RCCL collective serial):
  ag   : C = allgather(A) @ B^T          vs all_gather + torch.matmul
  rs   : rank r keeps rows_r of sum a@b  vs torch.matmul + reduce_scatter
  ar   : C = sum_r a_r @ b_r^T           vs torch.matmul + all_reduce
         (monolithic MPC=M per S7)
  a2a  : MoE dispatch+combine round      vs equal-split all_to_all x2

Symmetric-heap discipline: the 8-PE probe measured ~1.8 GB usable per
rank; every shape FREES its big symmetric buffers behind an MPI barrier
before the next shape constructs its ctx.

mpirun --allow-run-as-root -np 8 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s9_bench_8card.py
"""
import ctypes
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s9b_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size in (2, 8)

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu import (  # noqa: E402
    all_gather_gemm as ag, gemm_allreduce as gar,
    gemm_reduce_scatter as grs, moe_a2a)
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

torch.cuda.set_device(rank)

DTYPE = torch.float16
WARMUP, ITERS = 10, 50


@triton.jit(do_not_specialize=["recv_base", "out_base", "n_ptr"])
def k_expert_fanout(recv_base, out_base, n_ptr, H: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(n_ptr)
    rp = recv_base.to(tl.pointer_type(tl.float16))
    op = out_base.to(tl.pointer_type(tl.float16))
    for off in range(0, H, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        m = (row < n) & (offs < H)
        v = tl.load(rp + row * H + offs, mask=m, other=0.0).to(tl.float32) \
            * 2.0 + (row % 5).to(tl.float32)
        tl.store(op + row * H + offs, v.to(tl.float16), mask=m)


def tloop(fn, warmup=WARMUP, iters=ITERS):
    st = torch.cuda.current_stream()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(st)
        fn()
        e.record(st)
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def stats(ts):
    ts = sorted(ts)
    return {"mean": statistics.fmean(ts), "p50": ts[len(ts) // 2],
            "min": ts[0]}


def fmt(d):
    return f"{d['mean']:8.3f}/{d['p50']:8.3f}/{d['min']:8.3f}"  # mean/p50/min


def mpi_barrier():
    mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
    mpi.MPI_Barrier.restype = ctypes.c_int
    assert mpi.MPI_Barrier(ctypes.c_void_p(comm)) == 0


def free_symm(*ptrs):
    """Free big symmetric buffers collectively (8-PE heap ~1.8 GB total)."""
    mpi_barrier()
    host = dcu_state()["host"]
    for p in ptrs:
        host.free(p)
    mpi_barrier()


# ------------------------------------------- torch.distributed GPU backend --
dist, dist_backend = None, None
try:
    import torch.distributed as _dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29755")
    for _be in ("nccl", "rccl"):
        try:
            _dist.init_process_group(backend=_be, rank=rank, world_size=size)
            try:
                x = torch.ones(64, device="cuda") * (rank + 1)
                _dist.all_reduce(x)
                torch.cuda.synchronize()
                assert x[0].item() == size * (size + 1) // 2, "verify failed"
                dist, dist_backend = _dist, _be
                break
            except Exception:
                _dist.destroy_process_group()
        except Exception:
            pass
except Exception:
    pass
print(f"[rank {rank}] torch.distributed GPU backend: "
      f"{dist_backend or 'UNAVAILABLE'}", flush=True)

RESULTS = {
    "env": {"date": time.strftime("%Y-%m-%d %H:%M"),
            "torch": torch.__version__, "gpus": torch.cuda.get_device_name(0),
            "num_ranks": size, "dtype": "fp16",
            "dist_backend": dist_backend, "warmup/iters": f"{WARMUP}/{ITERS}"},
    "ag": {}, "rs": {}, "ar": {}, "a2a": {},
}


# ------------------------------------------------------------- AG + GEMM --
def bench_ag(Mpr, K, Npr, MPC):
    M = Mpr * size
    ctx = ag.create_ag_gemm_intra_node_context(
        max_M=M, K=K, input_dtype=DTYPE, output_dtype=DTYPE, rank=rank,
        num_ranks=size, mpi=mpi, comm=comm, M_PER_CHUNK=MPC)
    torch.manual_seed(9100 + rank)
    a = (torch.randn(Mpr, K, device="cuda") * 0.02).half()
    B = (torch.randn(Npr, K, device="cuda") * 0.02).half()
    c = ag.ag_gemm_intra_node(a, B, False, ctx)
    torch.cuda.synchronize()
    assert torch.isfinite(c.float()).all()

    res = {}
    res["ours"] = stats(tloop(lambda: ag.ag_gemm_intra_node(a, B, False, ctx)))
    res["gemm_triton"] = stats(
        tloop(lambda: ag.ag_gemm_intra_node_op(a, B, c, ctx, skip_comm=True)))
    res["barrier"] = stats(tloop(lambda: ctx.barrier.round_barrier(rank)))

    if dist is not None:
        a_list = [torch.empty(Mpr, K, dtype=DTYPE, device="cuda")
                  for _ in range(size)]
        A_full = torch.empty(M, K, dtype=DTYPE, device="cuda")

        def dist_ag():
            dist.all_gather(a_list, a)

        def dist_seq():
            dist.all_gather(a_list, a)
            torch.matmul(A_full, B.t(), out=c)
        # A_full == allgather(a): fix once (inputs are static in the loop)
        with torch.no_grad():
            dist.all_gather(a_list, a)
            A_full.copy_(torch.cat(a_list, 0))
        res["dist_ag"] = stats(tloop(dist_ag))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"Mpr{Mpr}_K{K}_N{Npr}_mpc{MPC}"
    line = f"ours {fmt(res['ours'])}  gemm {fmt(res['gemm_triton'])}  " \
           f"barrier {fmt(res['barrier'])}"
    if dist is not None:
        line += f"  | dist_ag {fmt(res['dist_ag'])}  dist_seq {fmt(res['dist_seq'])}"
    print(f"[rank {rank}] AG  {tag}: {line}", flush=True)
    RESULTS["ag"][tag] = res
    free_symm(ctx.ws_ptr, ctx.flags_ptr)
    del ctx
    torch.cuda.empty_cache()


# ------------------------------------------------------------- GEMM + RS --
def bench_rs(M, K, N):
    ctx = grs.create_gemm_rs_intra_node_context(
        max_M=M, N=N, output_dtype=DTYPE, rank=rank, num_ranks=size,
        mpi=mpi, comm=comm)
    torch.manual_seed(9200 + rank)
    A = (torch.randn(M, K, device="cuda") * 0.02).half()
    B = (torch.randn(N, K, device="cuda") * 0.02).half()
    out = grs.gemm_rs_intra_node(A, B, ctx)
    torch.cuda.synchronize()
    assert torch.isfinite(out.float()).all()

    res = {}
    res["ours"] = stats(tloop(lambda: grs.gemm_rs_intra_node(A, B, ctx)))
    res["gemm_triton"] = stats(
        tloop(lambda: grs.matmul_local(A, B, ctx.scratch)))
    res["barrier"] = stats(tloop(lambda: ctx.barrier.round_barrier(rank)))

    if dist is not None:
        Mpr = M // size
        part = torch.empty(M, N, dtype=DTYPE, device="cuda")
        out_d = torch.empty(Mpr, N, dtype=DTYPE, device="cuda")

        def dist_rs():
            dist.reduce_scatter_tensor(out_d, part)

        def dist_seq():
            torch.matmul(A, B.t(), out=part)
            dist.reduce_scatter_tensor(out_d, part)
        res["dist_rs"] = stats(tloop(dist_rs))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"M{M}_K{K}_N{N}"
    line = f"ours {fmt(res['ours'])}  gemm {fmt(res['gemm_triton'])}  " \
           f"barrier {fmt(res['barrier'])}"
    if dist is not None:
        line += f"  | dist_rs {fmt(res['dist_rs'])}  dist_seq {fmt(res['dist_seq'])}"
    print(f"[rank {rank}] RS  {tag}: {line}", flush=True)
    RESULTS["rs"][tag] = res
    free_symm(ctx.scatter_ptr)
    del ctx
    torch.cuda.empty_cache()


# ------------------------------------------------------------- GEMM + AR --
def bench_ar(M, K, N, MPC):
    ctx = gar.create_gemm_ar_intra_node_context(
        max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm, M_PER_CHUNK=MPC)
    torch.manual_seed(9300 + rank)
    a = (torch.randn(M, K, device="cuda") * 0.02).half()
    b = (torch.randn(N, K, device="cuda") * 0.02).half()
    c = gar.gemm_ar_intra_node(a, b, ctx)
    torch.cuda.synchronize()
    assert torch.isfinite(c.float()).all()

    res = {}
    res["ours"] = stats(tloop(lambda: gar.gemm_ar_intra_node(a, b, ctx)))
    res["gemm_triton"] = stats(
        tloop(lambda: gar.gemm_ar_intra_node_op(a, b, c, ctx,
                                                component="gemm")))
    res["push"] = stats(
        tloop(lambda: gar.gemm_ar_intra_node_op(a, b, c, ctx,
                                                component="push")))
    res["barrier"] = stats(tloop(lambda: ctx.barrier.round_barrier(rank)))

    if dist is not None:
        part = torch.empty(M, N, dtype=DTYPE, device="cuda")

        def dist_ar():
            dist.all_reduce(part)

        def dist_seq():
            torch.matmul(a, b.t(), out=part)
            dist.all_reduce(part)
        res["dist_ar"] = stats(tloop(dist_ar))
        res["dist_seq"] = stats(tloop(dist_seq))

    tag = f"M{M}_K{K}_N{N}_mpc{MPC}"
    line = f"ours {fmt(res['ours'])}  gemm {fmt(res['gemm_triton'])}  " \
           f"push {fmt(res['push'])}  barrier {fmt(res['barrier'])}"
    if dist is not None:
        line += f"  | dist_ar {fmt(res['dist_ar'])}  dist_seq {fmt(res['dist_seq'])}"
    print(f"[rank {rank}] AR  {tag}: {line}", flush=True)
    RESULTS["ar"][tag] = res
    free_symm(ctx.ws_ptr, ctx.flags_ptr)
    del ctx
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- MoE a2a --
def bench_a2a(T, H, TOPK, E):
    ctx = moe_a2a.create_moe_a2a_intra_node_context(
        num_experts=E, hidden=H, dtype=DTYPE, max_tokens=T, topk=TOPK,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm)
    torch.manual_seed(9400 + rank)
    tokens = (torch.randn(T, H, device="cuda") * 0.5).half()
    topk_idx = torch.randint(0, E, (T, TOPK), device="cuda",
                             dtype=torch.int32)
    weights = torch.rand(T, TOPK, device="cuda") + 0.5
    output = torch.empty(T, H, dtype=DTYPE, device="cuda")
    MRR = T * TOPK * size

    def full_round():
        moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx)
        k_expert_fanout[(MRR,)](ctx.recv_ptr, ctx.out_ptr, ctx.recv_total_t,
                                H=H, BLOCK=1024, num_warps=4)
        moe_a2a.moe_combine_intra_node_op(ctx, topk_idx, output, weights)

    full_round()
    torch.cuda.synchronize()
    assert torch.isfinite(output.float()).all()

    res = {}
    res["ours_full"] = stats(tloop(full_round))
    res["dispatch"] = stats(
        tloop(lambda: moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx,
                                                          ctx)))
    res["combine"] = stats(
        tloop(lambda: moe_a2a.moe_combine_intra_node_op(ctx, topk_idx,
                                                        output, weights)))
    res["meta"] = stats(tloop(
        lambda: moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx,
                                                   component="meta"),
        warmup=5, iters=20))
    res["barrier"] = stats(tloop(lambda: ctx.barrier.round_barrier(rank)))

    if dist is not None:
        send_rows = T * TOPK
        per = (send_rows + size - 1) // size
        pad = per * size
        stage_eq = torch.randn(pad, H, device="cuda").half()
        recv_eq = torch.empty(pad, H, dtype=DTYPE, device="cuda")

        def dist_a2a():
            dist.all_to_all_single(recv_eq, stage_eq)

        res["dist_a2a_eq"] = stats(tloop(dist_a2a))
        res["MB_per_dir"] = T * TOPK * H * 2 * (size - 1) / size / 1e6

        def dist_seq():
            dist.all_to_all_single(recv_eq, stage_eq)
            dist.all_to_all_single(stage_eq, recv_eq)
        res["dist_seq_2xa2a"] = stats(tloop(dist_seq))

    tag = f"T{T}_H{H}_topk{TOPK}_E{E}"
    print(f"[rank {rank}] A2A {tag} "
          f"recv_total={int(ctx.recv_total_t.item())}", flush=True)
    print(f"[rank {rank}]   full   {fmt(res['ours_full'])}  "
          f"dispatch {fmt(res['dispatch'])}  combine {fmt(res['combine'])}",
          flush=True)
    if dist is not None:
        print(f"[rank {rank}]   dist   a2a {fmt(res['dist_a2a_eq'])} "
              f"({res['MB_per_dir']:.0f}MB/dir)  "
              f"2xa2a {fmt(res['dist_seq_2xa2a'])}", flush=True)
    print(f"[rank {rank}]   meta   {fmt(res['meta'])}  "
          f"barrier {fmt(res['barrier'])}", flush=True)
    RESULTS["a2a"][tag] = res
    free_symm(ctx.recv_ptr, ctx.out_ptr, ctx.meta_ptr)
    del ctx
    torch.cuda.empty_cache()


# ------------------------------------------------------------------ run all --
bench_ag(2048, 2048, 2048, 512)
bench_ag(1024, 4096, 8192, 512)
bench_ag(128, 4096, 4096, 128)      # decode-ish
bench_rs(4096, 2048, 2048)
bench_rs(2048, 4096, 4096)
bench_ar(2048, 2048, 2048, 2048)    # monolithic (S7)
bench_ar(2048, 8192, 1024, 2048)
bench_a2a(1024, 2048, 8, 32)
bench_a2a(1024, 4096, 8, 32)
bench_a2a(512, 2048, 8, 32)
bench_a2a(128, 2048, 8, 32)         # decode-ish

if rank == 0:
    out = os.path.join(REPO, "bench", "results", "s9_8card.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"[rank 0] results -> {out}", flush=True)

dcu_state()["host"].finalize()
mpi_barrier()
mpi.MPI_Finalize()
os._exit(0)
