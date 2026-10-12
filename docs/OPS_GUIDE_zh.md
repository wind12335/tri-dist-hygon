# 四个通信-GEMM 融合算子：可用性与使用指南

> 平台：2×Hygon BW1000 (gfx936)，DTK 26.04（DUSHMEM），triton 3.3.0+das，torch 2.9.0+das，系统 OpenMPI 5.0.3。
> 全部数字为 2 卡实测 p50（50 iter，fp16；基准脚本自带 warmup 与 per-rank JIT 缓存目录）。
> 结论先行：**4/4 全部可用**——每个算子都通过了 fresh-data 多迭代正确性测试（每轮换数据、跨卡全量审计、干净退出）。性能上分两档：**MoE a2a 显著快于 RCCL（2.3-2.9×）**；三个 GEMM 融合算子功能完备、尾延迟已根治，但端到端仍落后 "vendor GEMM + RCCL 串行" 基线——差距主因是 triton GEMM 本地质量（对 vendor 1.7-3.1×），与分布式平面无关（分解见各节）。

## 0. 状态总览

| # | 算子 | 模块 | 语义 | 正确性 | 端到端 vs 基线 |
|---|---|---|---|---|---|
| 1 | AG+GEMM | `all_gather_gemm.py` | `C = allgather(A) @ B^T`，每 rank 得完整 `[M, N_pr]` | ✅ 3/3 | 1.12 vs 0.30 ms（落后，主因 gemm 质量） |
| 2 | GEMM+RS | `gemm_reduce_scatter.py` | rank r 留 `Σ_r a_r@b_r^T` 的第 r 行块 | ✅ 3/3 | 0.56 vs 0.32 ms（落后 1.7×，同因） |
| 3 | GEMM+AR | `gemm_allreduce.py` | `C = Σ_r a_r @ b_r^T`，每 rank 得完整 C | ✅ 3/3 | 0.83 vs 0.23 ms（S7 调优后 1.07→0.83） |
| 4 | MoE a2a | `moe_a2a.py` | expert-parallel dispatch/combine（动态路由 + topk 加权归并） | ✅ 3/3 | **1.31 vs 3.07 ms（快 2.34×；H4096 快 2.9×）** |

基线定义：#1-#3 = `torch.matmul`（vendor rocBLAS）+ 对应 RCCL 集合通信串行；#4 = RCCL `all_to_all_single` 等分近似 ×2（dispatch+combine 两趟）。RCCL 在本机由 torch.distributed（backend "nccl"→RCCL）提供。

三个 GEMM 算子的落后**不是通信面的失败**：组件分解显示 gemm_triton 单独就占差距的大头（例：AR 2048³ 中 gemm 0.28 vs vendor 0.09，3.1×），通信/信号/barrier 平面本身无停顿（S5 已根治尾延迟，mean≈p50）。a2a 不含本地 GEMM，纯 planes 优势直接兑现为对 RCCL 的 2.3×+。这套 planes + skew 韧性正是论文 #2（skew-aware 调度/重叠）的研究底座。

## 1. 通用运行环境（一次）

```bash
# 前置：构建设备位码 pydushmem/lib/pydushmem.bc（一次，见 README「构建」节）
cd /root/private_data/tri-dist-hygon   # 或任意 clone 处

# 统一启动模板（每个测试/基准都这么跑）：
ulimit -c 0   # 防 core dump 落盘（历史教训：双 rank 崩一次 ~4GB）
mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_ag_gemm.py        # <- 换成任意 tests/ 或 bench/ 脚本
```

- 所有 `tests/`、`bench/` 脚本**自带** per-rank `TRITON_CACHE_DIR` 与互不冲突的 `MASTER_PORT`，直接跑即可；
- 两条铁律：**MPI init + `hipSetDevice(rank)` 必须在 `import torch` 之前**（torch wheel 自带 OpenMPI 会 interpose 系统 libmpi）；`@triton.jit` 脚本落盘 `.py` 再跑；
- 自写调用方退出前：`host.finalize() → MPI_Barrier → MPI_Finalize → os._exit(0)`（析构竞争会 SEGV）；
- 目前验证范围：2 卡、fp16；每个 ctx 形状固定（对称堆按 max 尺寸预分配）。

---

## 2. AG+GEMM（all_gather_gemm.py）

**语义**：TP 行并行——每 rank 持 A 的一个行 shard `a=[M_pr, K]`，推全 A 到对称堆，本地 `C = A @ b^T`（`b=[N_pr, K]`），每 rank 得完整 `[M, N_pr]`。

**接口与约束**：

```python
from triton_dist.kernels.dcu import all_gather_gemm as ag
ctx = ag.create_ag_gemm_intra_node_context(
    max_M, K, torch.float16, torch.float16,      # 输入/输出 dtype
    rank, world_size, mpi, comm, M_PER_CHUNK=512)  # mpc512 是 AG 甜点（见下表）
c = ag.ag_gemm_intra_node(a, b, ctx)            # a [M_pr,K] b [N_pr,K] -> c [M,N_pr]
```
约束：`max_M % R == 0`、`M_pr % MPC == 0`、`MPC % 128 == 0`、`M % 128 == 0`、`N_pr % 128 == 0`、K 对齐 64；a/b 行主序连续。

**验证 / 基准**：

```bash
mpirun ... python3 tests/test_ag_gemm.py      # 3/3 PASS
mpirun ... python3 bench/s4_bench.py          # AG+RS 全套 -> bench/results/s4_2card_bw1000.json
```

**实测（p50 ms；dist_seq = vendor GEMM + RCCL allgather 串行）**：

| 形状 (M_pr,K,N_pr,MPC) | ours | dist_seq | 组件：gemm / push / barrier |
|---|---|---|---|
| 128, 4096, 4096, 128（decode） | 0.61 | 0.16 | 0.26 / 0.15 / 0.09 |
| 512, 4096, 4096, 256 | 0.91 | 0.24 | 0.51 / 0.22 / 0.09 |
| 1024, 4096, 8192, 256 | 1.95 | 0.62 | 1.43 / 0.32 / 0.09 |
| 2048, 2048, 2048, 256 | 1.12 | 0.30 | 0.50 / 0.43 / 0.09 |

MPC 甜点：512（mpc 扫参 128→1024：1.39/1.13/1.02/0.94）。状态注记：落后主因 = gemm_triton 对 vendor 3.1×；纯传输面 20 vs 49 GB/s（P5/S4）。优化史见 findings §6（S5）。

---

## 3. GEMM+RS（gemm_reduce_scatter.py）

**语义**：`C_r = (Σ_s a_s @ b_s^T)[rows_r]`——各 rank 本地 GEMM 到 scratch，dushmem scatter 各自部分和，环形归并后 rank r 留 `[M/R, N]` 行块。

**接口与约束**：

```python
from triton_dist.kernels.dcu import gemm_reduce_scatter as grs
ctx = grs.create_gemm_rs_intra_node_context(
    max_M, N, torch.float16, rank, world_size, mpi, comm)
c = grs.gemm_rs_intra_node(a, b, ctx)   # a [M,K_loc] b [N,K_loc] -> c [M/R, N]
```
约束：M、N、K 均 % 256 == 0（内核对齐断言）；行主序连续。

**验证 / 基准**：

```bash
mpirun ... python3 tests/test_gemm_rs.py
mpirun ... python3 bench/s4_bench.py    # rs 段 -> 同一 JSON
```

**实测（p50 ms；dist_seq = vendor GEMM + RCCL reduce_scatter 串行）**：

| 形状 (M,K,N) | ours | dist_seq | 组件：gemm / scatter / reduce |
|---|---|---|---|
| 512, 4096, 4096 | 0.48 | 0.22 | 0.24 / 0.09 / 0.10 |
| 2048, 4096, 4096 | 0.77 | 0.42 | 0.51 / 0.13 / 0.10 |
| 4096, 2048, 2048 | 0.56 | 0.32 | 0.30 / 0.13 / 0.10 |

状态注记：三算子中最接近基线（1.7×）；同样卡在 gemm 质量。优化史见 findings §6。

---

## 4. GEMM+AR（gemm_allreduce.py）

**语义**：TP 列并行——`C = Σ_r a_r @ b_r^T`，每 rank 得完整 `[M, N]`。各 rank 的部分积写进自己对称堆槽位，引擎 memcpy 扇出，整行 flags 一次 RMA putmem 信号，consumer 混合自旋后 fp32 归并。

**接口与约束**：

```python
from triton_dist.kernels.dcu import gemm_allreduce as gar
ctx = gar.create_gemm_ar_intra_node_context(
    max_M, N, torch.float16, torch.float16,
    rank, world_size, mpi, comm, M_PER_CHUNK=2048)  # S7 推荐 MPC=M（单块）
c = gar.gemm_ar_intra_node(a, b, ctx)   # a [M,K_loc] b [N,K_loc] -> c [M,N]
```
约束：`max_M % MPC == 0`、`MPC % 64 == 0`、`N % 128 == 0`、`K % 64 == 0`、`M ≤ max_M`；a/b 行主序连续。`component="gemm"/"push"` 为分解基准模式。

**验证 / 基准**：

```bash
mpirun ... python3 tests/test_gemm_ar.py
mpirun ... python3 bench/s6_bench_gemm_ar.py     # -> bench/results/s6_2card_bw1000.json
mpirun ... python3 bench/s7_ar_decompose.py      # S7 调参探针（consumer/monolithic 扫描）
```

**实测（p50 ms；dist_seq = vendor GEMM + RCCL allreduce 串行）**：

| 形状 (M,K,N) | ours（S7 调优后） | 调优前 | dist_seq |
|---|---|---|---|
| 2048, 2048, 2048 | **0.83**（MPC=2048） | 1.07（MPC=1024） | 0.23 |
| 2048, 8192, 1024 | **0.765**（MPC=2048） | 1.10 | 0.35 |
| 256, 4096, 4096 | 0.70（S6 数，S7 配置可再降，未复测） | — | 0.17 |
| 128, 4096, 4096 | 0.59（同上） | — | 0.14 |

S7 两项调优：consumer BLOCK 8192/w8（占用受限，2.2×）+ 单块 MPC=M。状态注记：与 #1/#2 同源——gemm_triton 0.28 vs vendor 0.09 是主因；逐 chunk 信号流水（恢复 GEMM/归约重叠）是下一步（设备原子 1.3µs/op 已探明可行）。优化史见 findings §7/S7.4。

---

## 5. MoE a2a（moe_a2a.py）——唯一超基线的算子

**语义**：expert 并行的 dispatch+combine。dispatch 按 `topk_idx[t,j]`（**全局** expert id，int32）把每个 (t,j) 行动态路由到 expert 所在 rank；expert 相在对端算完后，combine 由 token owner **拉回**各 topk 贡献并加权求和 `out[t] = Σ_j w[t,j]·expert_out_{rank(e_j)}[row]`。

**架构**（上游 per-token 设备 put 在 19.5MB/s 天花板下不可移植，按 DCU 平面重推导，findings S7）：
- dispatch 数据面：本地 staging kernel 按 (dest, expert) 段压拢（元数据预计算段基址 + 段内原子分配）→ 每 dest **一次**引擎 memcpy 推整段（~110GB/s）；
- 元数据面：单 CTA kernel 直方图计数+打包 `[flag|splits]` → 每 dest 一次 RMA putmem → 单 CTA kernel 自旋 flag 后**在设备上**算全部段表；
- combine：owner 侧 PULL kernel 远端 plain load（预翻译 peer VA；74-83 GB/s）。

**接口与调用协议**（**每轮必须 dispatch → expert → combine 依次调用**，combine 前必须写完 expert 输出）：

```python
from triton_dist.kernels.dcu import moe_a2a
ctx = moe_a2a.create_moe_a2a_intra_node_context(
    num_experts=32, hidden=2048, dtype=torch.float16,
    max_tokens=2048, topk=8, rank=rank, num_ranks=world_size, mpi=mpi, comm=comm)
# 约束：E % R == 0；H % 1024 == 0（BLOCK_H 默认 1024）；T <= max_tokens

recv_total = moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx)
#   recv_total: 设备 i32[1] 张量（ctx.recv_total_t）——需要宿主值时 .item()
# --- expert 相：读 ctx.recv_ptr 的前 recv_total 行（行序 = (src, expert) 段），
#     1:1 写 ctx.out_ptr 同行号（对称堆 VA，kernel 侧用）---
moe_a2a.moe_combine_intra_node_op(ctx, topk_idx, output, topk_weights)
#   topk_weights: 可选 fp32 [T, topk]；output: [T, H]
```

**验证 / 基准**：

```bash
mpirun ... python3 tests/test_moe_a2a.py        # 3/3 PASS（内容/双射/expert/归并四重审计）
mpirun ... python3 bench/s7_bench_moe_a2a.py    # -> bench/results/s7_2card_bw1000.json
mpirun ... python3 bench/s7_peer_load_probe.py  # PULL 决策探针（74-83 GB/s）
mpirun ... python3 bench/s7_meta_decompose.py   # 元数据面分解
```

**实测（p50 ms；RCCL 列 = 等分 all_to_all ×2 串行近似；单次 all_to_all 也列出）**：

| 形状 (T,H,topk,E) | ours 整轮 | dispatch / combine | RCCL 单次 a2a | RCCL 2×a2a | 倍数 |
|---|---|---|---|---|---|
| 2048, 2048, 8, 32 | **1.31** | 0.76 / 0.48 | 1.56 | 3.07 | **2.34×** |
| 2048, 4096, 8, 32 | **2.05** | 1.04 / 0.78 | 3.01 | 5.99 | **2.9×** |
| 1024, 2048, 8, 32 | **0.99** | 0.63 / 0.32 | 0.82 | 1.60 | 1.6× |
| 2048, 2048, 4, 32 | **0.92** | 0.60 / 0.30 | 0.83 | 1.61 | 1.7× |
| 256, 2048, 8, 32 | 0.74 | 0.55 / 0.18 | 0.27 | 0.49 | 0.66×（小包输，元数据地板） |

亮点：T2048/H2048 时**我们含 expert 相的整轮（1.31ms）比 RCCL 单次 all_to_all（1.56ms）还快**。带宽已贴物理上限：combine 0.48ms ≈ 33MB@74GB/s，push ≈ 34MB@110GB/s——数据面无剩余空间，后续收益在元数据面（0.29ms，宿主启动开销主导）与 push/stage 重叠。调优史见 findings §8（S7.3）。

---

## 6. 横向结论

- **能用的定义**：每个算子 = 正确性 3/3 PASS（fresh-data 多迭代 + 跨卡全量审计 + 干净退出）+ 尾延迟根治（mean≈p50）+ 分解基准可复现。
- **性能两档**：a2a（无本地 GEMM）纯 planes 优势兑现为 2.3-2.9×；三个 GEMM 融合算子落后基线，主因 triton GEMM 本地质量（1.7-3.1×）——追平 vendor GEMM 即可整体翻盘（通信平面额外开销 ≈ barrier 0.09 + push 已贴引擎带宽）。
- **已定型的三条传输平面**（所有算子共用）：数据 = 宿主引擎 memcpy ~110GB/s；信号/元数据 = 宿主 RMA putmem（16µs/条，消息内有序）；远端读 = 设备 plain load 74-83GB/s（L2 一致，混合自旋守卫）。设备发起的 dushmem 传输是 19.5MB/s 字节率天花板（P5），任何新算子先按这三条平面重推导。
- **下一步（按收益排序）**：gemm_triton 追 vendor（惠及 #1-#3）；逐 chunk 信号流水恢复 AR 的 GEMM/归约重叠；a2a push/stage 重叠与元数据面再砍半。

各算子完整问题定位与优化时间线：[DCU_PORT_FINDINGS_zh.md](DCU_PORT_FINDINGS_zh.md)（§5=S5 调优、§7=S6 AR 移植、§8=S7 a2a 移植+双算子调优）。
