"""S8 / 论文#2 v3 benchmark: straggler(δ) 注入下三种 AR 架构的 δ 响应斜率 (2x BW1000).

科学问题:decode serving 实测 δ≈400µs 到达散布下,集体语义 vs 解耦面(gemm_allreduce)
各自的 δ 响应斜率与反超点 δ*;fill 模式进一步测等待窗口的算力回收(I2)。

三种架构(同一算子 C = Σ_r a_r @ b_r^T,2 rank):
  mono    : matmul(整 M) + all_reduce(一次)          —— vLLM decode 现状(1 次宿主入口)
  chunked : k 次 [matmul(chunk) + all_reduce(chunk)]   —— E1 的 chunked-collective
            (python 循环 = k 次宿主入口,忠实于"今天用 collective 能拼出来的东西")
  fused   : gemm_ar_intra_node_op(内部 chunk 管道)     —— S6 解耦面(1 次宿主入口)

两种注入(straggler rank 的 GPU 流上、GEMM 之前,torch.cuda._sleep):
  offset : 整个 op 前睡 δ 一次   —— "rank 整体慢 δ"(阴性对照:任何架构都只付 1×δ)
  jitter : 每次宿主入口前睡 δ     —— "每次 collective 入口重抽宿主抖动"(decode 实测:
           每 AR=新 forward=新宿主往返)。入口数 mono=1, chunked=k, fused=1
           —— 差异不是偏袒,正是被测机制:解耦面没有逐 chunk 宿主重入。

fill 模式(S8_FILL=1,I2 判决):每个点额外测一次"op + 独立 GEMM 流"的合并时间:
  fill_launch() 在 fill 流上发 S8_FILL_COUNT 个独立 GEMM(与本 op 无数据依赖),
  再在计算流上跑 op —— 模拟"等待窗口里 GPU 有别的活干"。
  输出 hidden% = 1 − (t_combined − t_op_alone)/t_fill_alone。
  首轮(2026-10-12 _fill/_k4)判决:三架构全部 ≈ 精确加性(mono +18%/chunked +13%/
  fused −3%)—— t0 发出的 fill 与 op 的 GEMM 相时间分片,等待窗从未收到 fill,
  测的不是 I2。修正:S8_FILL_DELAY_US 在 fill 流上先 _sleep 再发 fill,把 fill
  排进等待窗内部;S8_FILL_PROBE=1 加流并发探针(compute 流只放 1-block _sleep,
  无 SM 压力):probe hidden≈100% → 流可并发,阻塞在 kernel SM 占用(RCCL 自旋/
  triton GEMM 满格/P9 消费者满格);≈0 → das 栈流全局串行,I2 多流路线不可行。

用法(迁移实例双卡):
  mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
    -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
    python3 bench/s8_bench_skew.py
  环境变量:
    S8_SHAPES="128:4096:4096:128,256:4096:4096:256,2048:2048:2048:1024"  # M:K:N:MPC
    S8_DELTAS_US="0,100,200,400,800"
    S8_MODES="offset,jitter"
    S8_RANK_STRAGGLER=1   S8_ITERS=30   S8_WARMUP=5
    S8_FILL=1                       # 开 fill 模式
    S8_FILL_SHAPE="1024:1024:1024"  # fill GEMM 的 M:K:N
    S8_FILL_COUNT=2                 # 每个 iter 发几个 fill GEMM
    S8_FILL_DELAY_US=600            # fill 流先睡再发(把 fill 排进等待窗;0=旧行为)
    S8_FILL_PROBE=1                 # 加流并发探针(1-block _sleep + fill)
    S8_FILL_PROBE_US=800            # 探针的 compute 流 _sleep 时长
    S8_TAG="_fill2"                 # 结果文件后缀(防覆盖):s8_skew_2card<S8_TAG>.json

  fill 修正轮(DELAY+PROBE): S8_FILL=1 S8_FILL_PROBE=1 S8_FILL_DELAY_US=600 \
      S8_TAG="_fill2" S8_SHAPES="128:4096:4096:128,2048:2048:2048:1024" mpirun ...
  k=4 + 越窗验证(已跑): S8_FILL=1 S8_TAG="_k4" S8_SHAPES="2048:2048:2048:512" \
                  S8_DELTAS_US="0,200,400,800,1600" mpirun ...
"""
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("TRITON_CACHE_DIR",
                      f"/tmp/tc_s8skew_{os.environ.get('OMPI_COMM_WORLD_RANK', '0')}")

import ctypes  # noqa: E402

import pydushmem.host as pHost  # noqa: E402

mpi, comm = pHost.mpi_init()
rank = pHost.mpi_comm_rank(mpi, comm)
size = pHost.mpi_comm_size(mpi, comm)
assert size == 2, "S8 设计为 2 rank(与 s6 一致);多 rank 需扩展 chunked 基线"

hip = ctypes.CDLL("/opt/dtk-26.04/lib/libamdhip64.so")
hip.hipSetDevice.restype = ctypes.c_int
assert hip.hipSetDevice(rank) == 0

import torch  # noqa: E402

torch.cuda.set_device(rank)

from triton_dist.kernels.dcu import gemm_allreduce as gar  # noqa: E402
from triton_dist.kernels.dcu.common_ops import dcu_state  # noqa: E402

DTYPE = torch.float16
SCALE = 0.02
STRAGGLER = int(os.environ.get("S8_RANK_STRAGGLER", "1"))
SHAPES = [tuple(int(x) for x in s.split(":"))
          for s in os.environ.get(
              "S8_SHAPES", "128:4096:4096:128,256:4096:4096:256,2048:2048:2048:1024"
              ).split(",") if s]
DELTAS = [int(x) for x in os.environ.get("S8_DELTAS_US", "0,100,200,400,800").split(",")]
MODES = [m for m in os.environ.get("S8_MODES", "offset,jitter").split(",") if m]
WARMUP = int(os.environ.get("S8_WARMUP", "5"))
ITERS = int(os.environ.get("S8_ITERS", "30"))
FILL = os.environ.get("S8_FILL", "0") == "1"
FILL_SHAPE = tuple(int(x) for x in os.environ.get("S8_FILL_SHAPE", "1024:1024:1024").split(":"))
FILL_COUNT = int(os.environ.get("S8_FILL_COUNT", "2"))
FILL_DELAY = int(os.environ.get("S8_FILL_DELAY_US", "0"))
PROBE = os.environ.get("S8_FILL_PROBE", "0") == "1"
PROBE_US = int(os.environ.get("S8_FILL_PROBE_US", "800"))
TAG = os.environ.get("S8_TAG", "")

# ---------------- torch.distributed GPU 后端(照抄 s6) ----------------
dist = None
dist_backend = None
try:
    import torch.distributed as _dist
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29637")
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
print(f"[rank {rank}] backend={dist_backend or 'UNAVAILABLE'}"
      + (f" fill={FILL_SHAPE}x{FILL_COUNT}" if FILL else ""), flush=True)
assert dist is not None, "需要 torch.distributed GPU 后端做 mono/chunked 基线"

# ---------------- GPU 延迟注入:torch.cuda._sleep + 标定 ----------------
sleep_fn = getattr(torch.cuda, "_sleep", None)
CYCLES_PER_US = None
if sleep_fn is not None:
    n_cal = 20_000_000
    s_ev, e_ev = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(3):
        sleep_fn(n_cal)
    torch.cuda.synchronize()
    s_ev.record()
    sleep_fn(n_cal)
    e_ev.record()
    torch.cuda.synchronize()
    CYCLES_PER_US = n_cal / max(s_ev.elapsed_time(e_ev) * 1e3, 1.0)
    print(f"[rank {rank}] _sleep calibrated: {CYCLES_PER_US:.0f} cycles/µs", flush=True)
else:
    print(f"[rank {rank}] WARN: torch.cuda._sleep 不可用,δ>0 的点将跳过", flush=True)


def gpu_delay(us):
    """straggler rank 的 GPU 流上睡 us 微秒(流序,在 GEMM 之前)。"""
    if us <= 0 or rank != STRAGGLER or sleep_fn is None:
        return
    sleep_fn(int(us * CYCLES_PER_US))


# ---------------- fill 模式:独立 GEMM 流(I2 等待窗口算力回收) ----------------
fill_stream = torch.cuda.Stream() if FILL else None
if FILL:
    torch.manual_seed(9900 + rank)
    _fa = (torch.randn(FILL_SHAPE[0], FILL_SHAPE[1], device="cuda") * 0.01).half()
    _fb = (torch.randn(FILL_SHAPE[2], FILL_SHAPE[1], device="cuda") * 0.01).half()
    _fc = torch.empty(FILL_SHAPE[0], FILL_SHAPE[2], dtype=DTYPE, device="cuda")


def fill_launch(delay_us=0):
    with torch.cuda.stream(fill_stream):
        if delay_us > 0 and sleep_fn is not None:
            sleep_fn(int(delay_us * CYCLES_PER_US))   # 把 fill 排进等待窗内部
        for _ in range(FILL_COUNT):
            torch.matmul(_fa, _fb.t(), out=_fc)


def probe_stream_concurrency():
    """流并发探针:compute 流只有 1-block _sleep(无 SM 压力)时 fill 能否并发。
    hidden≈100% → 流可并发,加性阻塞在 kernel SM 占用;≈0 → 流全局串行。"""
    st = torch.cuda.current_stream()

    def once():
        s, e_c, e_f = (torch.cuda.Event(True) for _ in range(3))
        s.record(st)
        fill_launch()
        sleep_fn(int(PROBE_US * CYCLES_PER_US))
        e_c.record(st)
        e_f.record(fill_stream)
        torch.cuda.synchronize()
        return max(s.elapsed_time(e_c), s.elapsed_time(e_f)) * 1e3

    def alone_sleep():
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(st)
        sleep_fn(int(PROBE_US * CYCLES_PER_US))
        e.record(st)
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3

    for _ in range(WARMUP):
        once()
    comb = statistics.fmean(once() for _ in range(ITERS))
    slp = statistics.fmean(alone_sleep() for _ in range(ITERS))

    def alone_fill():
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(fill_stream)
        fill_launch()
        e.record(fill_stream)
        torch.cuda.synchronize()
        return s.elapsed_time(e) * 1e3

    fil = statistics.fmean(alone_fill() for _ in range(ITERS))
    hidden = (1 - (comb - slp) / max(fil, 1e-9)) * 100
    return {"comb_us": comb, "sleep_alone_us": slp, "fill_alone_us": fil,
            "hiddenPct": hidden}


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


def tloop_fill(fn, warmup=WARMUP, iters=ITERS, delay=FILL_DELAY):
    """op(计算流)+ fill GEMM(fill 流)并发的合并耗时 = max(两流终点)。
    delay>0 时 fill 流先睡(把 fill 排进等待窗);fill_alone 基线须传 delay=0。"""
    st = torch.cuda.current_stream()
    for _ in range(warmup):
        fill_launch(delay)
        fn()
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e_c = torch.cuda.Event(enable_timing=True)
        e_f = torch.cuda.Event(enable_timing=True)
        s.record(st)
        fill_launch(delay)
        fn()
        e_c.record(st)
        e_f.record(fill_stream)
        torch.cuda.synchronize()
        ts.append(max(s.elapsed_time(e_c), s.elapsed_time(e_f)))
    return ts


def stats(ts):
    ts = sorted(ts)
    return {"mean": statistics.fmean(ts), "p50": ts[len(ts) // 2], "min": ts[0]}


def lsq_slope(xs, ys):
    """最小二乘斜率(y ms / x µs → 返回无量纲:每 1µs δ 付多少 µs)。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 1e3


RESULTS = {
    "env": {"date": time.strftime("%Y-%m-%d %H:%M"), "torch": torch.__version__,
            "gpus": torch.cuda.get_device_name(0), "num_ranks": size,
            "dist_backend": dist_backend, "straggler_rank": STRAGGLER,
            "cycles_per_us": CYCLES_PER_US, "warmup/iters": f"{WARMUP}/{ITERS}",
            "deltas_us": DELTAS, "modes": MODES,
            "fill": {"on": FILL, "shape": list(FILL_SHAPE), "count": FILL_COUNT}},
    "skew": {},
}

# 注入状态(tloop 的 fn 闭包读取;循环里总会先赋值)
PENDING_DELTA, PENDING_MODE = 0, "offset"

if FILL and PROBE and sleep_fn is not None:
    _pr = probe_stream_concurrency()
    RESULTS["env"]["fill_probe"] = _pr
    print(f"[rank {rank}] probe(1-block _sleep {_pr['sleep_alone_us']:.0f}µs + fill "
          f"{_pr['fill_alone_us']:.0f}µs): comb={_pr['comb_us']:.0f}µs "
          f"→ hidden {_pr['hiddenPct']:+.0f}%", flush=True)


def bench_shape(M, K, N, MPC):
    ctx = gar.create_gemm_ar_intra_node_context(
        max_M=M, N=N, input_dtype=DTYPE, output_dtype=DTYPE,
        rank=rank, num_ranks=size, mpi=mpi, comm=comm, M_PER_CHUNK=MPC)
    A, B = [], []
    for s_ in range(size):
        torch.manual_seed(7100 + s_)
        A.append((torch.randn(M, K, device="cuda") * SCALE).half())
        torch.manual_seed(7300 + s_)
        B.append((torch.randn(N, K, device="cuda") * SCALE).half())
    a, b = A[rank], B[rank]
    c = torch.empty(M, N, dtype=DTYPE, device="cuda")

    with torch.no_grad():
        ref = A[0].float() @ B[0].float().t()
        for s_ in range(1, size):
            ref += A[s_].float() @ B[s_].float().t()

    k = (M + MPC - 1) // MPC

    def mono():
        gpu_delay(PENDING_DELTA)
        torch.matmul(a, b.t(), out=c)
        dist.all_reduce(c)

    def chunked():
        for i in range(k):
            if PENDING_MODE == "jitter" or i == 0:
                gpu_delay(PENDING_DELTA)
            torch.matmul(a[i * MPC:(i + 1) * MPC], b.t(),
                         out=c[i * MPC:(i + 1) * MPC])
            dist.all_reduce(c[i * MPC:(i + 1) * MPC])

    def fused():
        # 单次宿主入口:两种模式都只注入一次 —— 入口数差异就是被测机制(见文件头)
        gpu_delay(PENDING_DELTA)
        gar.gemm_ar_intra_node_op(a, b, c, ctx)

    # 正确性(δ=0 路径)
    torch.matmul(a, b.t(), out=c)
    dist.all_reduce(c)
    torch.cuda.synchronize()
    md = (c.float() - ref).abs().max().item()
    assert md < 1e-2, f"mono correctness FAIL md={md}"
    gar.gemm_ar_intra_node_op(a, b, c, ctx)
    torch.cuda.synchronize()
    md2 = (c.float() - ref).abs().max().item()
    assert md2 < 1e-2, f"fused correctness FAIL md={md2}"
    print(f"[rank {rank}] shape M{M}_K{K}_N{N}_mpc{MPC} k={k} corr={md:.1e}/{md2:.1e}",
          flush=True)

    tag = f"M{M}_K{K}_N{N}_mpc{MPC}"
    RESULTS["skew"][tag] = {"k_chunks": k, "bytes_out_MB": M * N * 2 * (size - 1) / 1e6}

    fill_alone = None
    if FILL:
        fill_alone = stats(tloop_fill(lambda: None, delay=0))   # 基线不含延迟
        RESULTS["skew"][tag]["fill_alone"] = fill_alone
        print(f"[rank {rank}]   fill_alone {fill_alone['mean']*1e3:.0f}µs "
              f"({FILL_SHAPE}x{FILL_COUNT})", flush=True)

    for mode in MODES:
        for arch, fn in (("mono", mono), ("chunked", chunked), ("fused", fused)):
            pts = []
            for d in DELTAS:
                if d > 0 and sleep_fn is None:
                    continue
                globals()["PENDING_DELTA"], globals()["PENDING_MODE"] = d, mode
                st_ = stats(tloop(fn))
                pts.append((d, st_))
                entry = RESULTS["skew"][tag].setdefault(mode, {}).setdefault(arch, {})
                entry[str(d)] = st_
                if FILL:
                    comb = stats(tloop_fill(fn))
                    entry.setdefault("fill", {})[str(d)] = comb
                    hid = (1 - (comb["mean"] - st_["mean"])
                           / max(fill_alone["mean"], 1e-9)) * 100
                    entry.setdefault("fill_hiddenPct", {})[str(d)] = hid
            xs = [p[0] for p in pts]
            ys = [p[1]["mean"] for p in pts]
            slope = lsq_slope(xs, ys) if len(xs) > 1 else None
            if slope is not None:
                RESULTS["skew"][tag][mode][arch]["slope_us_per_us"] = slope
            line = " ".join(f"δ{d}={y['mean']*1e3:.0f}µs" for d, y in pts)
            sl = f" slope={slope:.2f}" if slope is not None else ""
            print(f"[rank {rank}]   {mode:6s} {arch:7s} {line}{sl}", flush=True)
            if FILL:
                fl = " ".join(
                    f"δ{d}={RESULTS['skew'][tag][mode][arch]['fill'][str(d)]['mean']*1e3:.0f}"
                    f"({RESULTS['skew'][tag][mode][arch]['fill_hiddenPct'][str(d)]:.0f}%)"
                    for d, _ in pts)
                print(f"[rank {rank}]   {mode:6s} {arch:7s} [fill] {fl}", flush=True)

    del ctx
    torch.cuda.empty_cache()


for (M, K, N, MPC) in SHAPES:
    bench_shape(M, K, N, MPC)

if rank == 0:
    import json
    out = os.path.join(REPO, "bench", "results", f"s8_skew_2card{TAG}.json")
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"[rank 0] results -> {out}", flush=True)

dcu_state()["host"].finalize()
mpi.MPI_Barrier.argtypes = [ctypes.c_void_p]
mpi.MPI_Barrier.restype = ctypes.c_int
mpi.MPI_Barrier(ctypes.c_void_p(comm))
mpi.MPI_Finalize()
os._exit(0)
