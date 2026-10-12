"""S7 benchmark: MoE a2a (dispatch + combine) vs RCCL all_to_all reference
(2x BW1000).

"ours" is the full round dispatch -> fake expert -> combine (the expert
kernel is a trivial row transform, so the numbers isolate the a2a planes).
The RCCL reference is an EQUAL-split all_to_all of the same per-direction
volume (uniform-routing approximation -- real routing is dynamic; we keep
the routing fixed per shape so the allocator reaches steady state).
Components: dispatch / combine / metadata(host) / stage / barrier.

mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s7_bench_moe_a2a.py
"""
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s7a2ab_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

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
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from triton_dist.kernels.dcu import moe_a2a  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

torch.cuda.set_device(rank)

DTYPE = torch.float16
WARMUP, ITERS = 10, 50


@triton.jit(do_not_specialize=["recv_base", "out_base", "n_ptr"])
def k_expert_fanout(recv_base, out_base, n_ptr, H: tl.constexpr,
                    BLOCK: tl.constexpr):
    """Fake expert compute over [0, n) rows with n read from a DEVICE
    scalar -- the full round stays stream-async (no .item() sync for the
    grid); blocks past n exit via the mask."""
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
    for i in range(iters):
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


# ------------------------------------------- torch.distributed GPU backend --
dist = None
dist_backend = None
try:
    import torch.distributed as _dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29634")
    for _be in ("nccl", "rccl"):
        try:
            _dist.init_process_group(backend=_be, rank=rank, world_size=size)
            try:
                x = torch.ones(64, device="cuda") * (rank + 1)
                _dist.all_reduce(x)
                torch.cuda.synchronize()
                assert x[0].item() == 3.0, "verify failed"
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
    "env": {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "torch": torch.__version__,
        "gpus": torch.cuda.get_device_name(0),
        "num_ranks": size,
        "dtype": "fp16",
        "dist_backend": dist_backend,
        "warmup/iters": f"{WARMUP}/{ITERS}",
    },
    "a2a": {},
}

# T, H, topk, E -- routing FIXED per shape (steady-state allocator)
A2A_SHAPES = [
    (2048, 2048, 8, 32),
    (2048, 4096, 8, 32),
    (1024, 2048, 8, 32),
    (2048, 2048, 4, 32),
    (256, 2048, 8, 32),   # decode-ish
]


def bench_a2a(T, H, TOPK, E):
    ctx = moe_a2a.create_moe_a2a_intra_node_context(
        num_experts=E, hidden=H, dtype=DTYPE, max_tokens=T, topk=TOPK,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm)
    torch.manual_seed(9100 + rank)
    tokens = (torch.randn(T, H, device="cuda") * 0.5).half()
    topk_idx = torch.randint(0, E, (T, TOPK), device="cuda",
                             dtype=torch.int32)
    weights = torch.rand(T, TOPK, device="cuda") + 0.5
    output = torch.empty(T, H, dtype=DTYPE, device="cuda")

    # max grid; rows past recv_total exit via the device-n mask
    MRR = T * TOPK * size

    def full_round():
        moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx)
        k_expert_fanout[(MRR,)](ctx.recv_ptr, ctx.out_ptr, ctx.recv_total_t,
                                H=H, BLOCK=1024, num_warps=4)
        moe_a2a.moe_combine_intra_node_op(ctx, topk_idx, output, weights)

    # warm the exact path once, verify non-inf sanity
    full_round()
    torch.cuda.synchronize()
    assert torch.isfinite(output.float()).all(), "combine produced inf/nan"

    res = {}
    res["ours_full"] = stats(tloop(full_round))
    res["dispatch"] = stats(
        tloop(lambda: moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx,
                                                          ctx)))
    res["combine"] = stats(
        tloop(lambda: moe_a2a.moe_combine_intra_node_op(ctx, topk_idx,
                                                        output, weights)))
    res["barrier"] = stats(tloop(lambda: ctx.barrier.round_barrier(rank)))
    # metadata plane only: bincount -> pack -> putmem -> flag spin ->
    # table kernel -> plan D2H (component="meta")
    res["meta"] = stats(tloop(
        lambda: moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx,
                                                   component="meta"),
        warmup=5, iters=20))

    # equal-split RCCL all_to_all of the same per-direction volume
    if dist is not None:
        send_rows = T * TOPK
        per = (send_rows + size - 1) // size  # uniform approximation
        pad = per * size
        stage_eq = torch.randn(pad, H, device="cuda").half()
        recv_eq = torch.empty(pad, H, dtype=DTYPE, device="cuda")

        def dist_a2a():
            dist.all_to_all_single(recv_eq, stage_eq)

        res["dist_a2a_eq"] = stats(tloop(dist_a2a))
        vol = T * TOPK * H * 2 * (size - 1) / size / 1e6
        res["MB_per_dir"] = vol

        def dist_seq():
            dist.all_to_all_single(recv_eq, stage_eq)
            dist.all_to_all_single(stage_eq, recv_eq)
        res["dist_seq_2xa2a"] = stats(tloop(dist_seq))

    tag = f"T{T}_H{H}_topk{TOPK}_E{E}"
    print(f"[rank {rank}] A2A {tag} "
          f"recv_total={int(ctx.recv_total_t.item())}", flush=True)
    print(f"[rank {rank}]   full      {fmt(res['ours_full'])}", flush=True)
    print(f"[rank {rank}]   dispatch  {fmt(res['dispatch'])}  "
          f"combine {fmt(res['combine'])}", flush=True)
    if "dist_a2a_eq" in res:
        print(f"[rank {rank}]   dist_a2a  {fmt(res['dist_a2a_eq'])} "
              f"({res['MB_per_dir']:.0f}MB/dir)  "
              f"2xa2a {fmt(res['dist_seq_2xa2a'])}", flush=True)
    print(f"[rank {rank}]   meta      {fmt(res['meta'])}  "
          f"barrier {fmt(res['barrier'])}", flush=True)
    RESULTS["a2a"][tag] = res
    del ctx
    torch.cuda.empty_cache()


for (T, H, TOPK, E) in A2A_SHAPES:
    bench_a2a(T, H, TOPK, E)

if rank == 0:
    out = os.path.join(REPO, "bench", "results", "s7_2card_bw1000.json")
    with open(out, "w") as f:
        import json
        json.dump(RESULTS, f, indent=2)
    print(f"[rank 0] results -> {out}", flush=True)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
