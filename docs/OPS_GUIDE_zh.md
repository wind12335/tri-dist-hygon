# 四个通信-GEMM 融合算子：可用性与使用指南

> 平台：Hygon BW1000 (gfx936)，DTK 26.04（DUSHMEM），triton 3.3.0+das，torch 2.9.0+das，系统 OpenMPI 5.0.3。
> 数字为实测 p50（50 iter，fp16；基准脚本自带 warmup 与 per-rank JIT 缓存目录）；§2-§5 为 **2 卡**，§6 为 **8 卡**。
> 结论先行：**4/4 全部可用（2 卡与 8 卡均通过 fresh-data 多迭代正确性测试）**。性能分两档且**随卡数反转**：2 卡上 **MoE a2a 快于 RCCL 2.3-2.9×**、三个 GEMM 融合算子落后（主因 triton GEMM 本地质量 1.7-3.1×）；**8 卡上四算子全部落后 RCCL（2.4-7.5×，含 a2a 反转为慢 2.5-2.8×）**——根因是数据面的 (R-1) 单播引擎扇出在共享 fabric 上不扩展（有效出带宽 17-25GB/s/rank，S11 池 A/B 证实为 fabric 瓶颈）+ 宿主编排固定开销随 R 放大（barrier 0.09→0.21-0.26ms），分解见 §6 与 findings §9/§11。

## 0. 状态总览

| # | 算子 | 模块 | 语义 | 正确性 | 端到端 vs 基线 |
|---|---|---|---|---|---|
| 1 | AG+GEMM | `all_gather_gemm.py` | `C = allgather(A) @ B^T`，每 rank 得完整 `[M, N_pr]` | ✅ 3/3 | 0.91 vs 0.31 ms（S10 −6-8% + S11 单 chunk MPC −19%；主因 gemm 质量） |
| 2 | GEMM+RS | `gemm_reduce_scatter.py` | rank r 留 `Σ_r a_r@b_r^T` 的第 r 行块 | ✅ 3/3 | **0.51** vs 0.32 ms（S10: 0.56→0.51，落后 1.6×） |
| 3 | GEMM+AR | `gemm_allreduce.py` | `C = Σ_r a_r @ b_r^T`，每 rank 得完整 C | ✅ 3/3 | **0.75** vs 0.23 ms（S10: 0.83→0.75，gemm 组件 −21%） |
| 4 | MoE a2a | `moe_a2a.py` | expert-parallel dispatch/combine（动态路由 + topk 加权归并） | ✅ 3/3 | **1.31 vs 3.07 ms（快 2.34×；H4096 快 2.9×）** |

基线定义：#1-#3 = `torch.matmul`（vendor rocBLAS）+ 对应 RCCL 集合通信串行；#4 = RCCL `all_to_all_single` 等分近似 ×2（dispatch+combine 两趟）。RCCL 在本机由 torch.distributed（backend "nccl"→RCCL）提供。

三个 GEMM 算子的落后**不是通信面的失败**：组件分解显示 gemm_triton 单独就占差距的大头（例：AR 2048³ 中 gemm 0.165 vs vendor 0.095，1.7×，S10 后），通信/信号/barrier 平面本身无停顿（S5 已根治尾延迟，mean≈p50）。a2a 不含本地 GEMM，纯 planes 优势直接兑现为对 RCCL 的 2.3×+。这套 planes + skew 韧性正是论文 #2（skew-aware 调度/重叠）的研究底座。**上表为 2 卡数字；8 卡（S9）下胜负反转，见 §6。**

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
    rank, world_size, mpi, comm,                 # M_PER_CHUNK 缺省 1024；
    M_PER_CHUNK=max_M // world_size)  # S11: 每 rank 单 chunk 最优（见下表）
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
| 128, 4096, 4096, 128（decode） | 0.62 | 0.16 | 0.26 / 0.17 / 0.09 |
| 512, 4096, 4096, 512 | 0.90 | 0.24 | 0.50 / 0.18 / 0.09 |
| 1024, 4096, 8192, 1024 | 1.75 | 0.63 | 1.35 / 0.21 / 0.09 |
| 2048, 2048, 2048, 2048 | 0.91 | 0.31 | 0.49 / 0.21 / 0.09 |

**MPC 规则（S11）：MPC = M_per_rank（每 dest 单 chunk）最优**——post-S10 扫参单调（mpc128→2048 = 1.402/1.139/1.028/0.917/0.884），引擎单 op 带宽有台阶（2/4/8MB ≈ 25/48/70 GB/s），小 chunk 等于交每-op 开销税；旧"甜点 512"作废。例外：K8192 大块（16.8MB 单 op 实测仅 60 GB/s 且 consumer 重叠变差）收益收窄到 −2%。**S10**：consumer `num_stages=1`（M>512 时）。**S11**：push 多流池机制在库内但**默认关**（八卡 A/B 判定扇出墙是 fabric 瓶颈，池中性偏负，findings §11.5；`TD_PUSH_STREAMS=8` 可开）。状态注记：落后主因 = gemm_triton 对 vendor ~3×（0.49 vs 0.16）。优化史见 findings §6（S5）/§10（S10）/§11（S11）。

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
| 2048, 4096, 4096 | **0.69**（S10） | 0.42 | 0.43 / 0.13 / 0.10 |
| 4096, 2048, 2048 | **0.51**（S10） | 0.32 | 0.26 / 0.13 / 0.10 |

状态注记：三算子中最接近基线（**1.6×，S10 后**）；同样卡在 gemm 质量。**S10 更新**：`matmul_local` 按形状切 `num_stages`（M>512 且 K<8192 → s1，同场 A/B gemm 组件 −15/16%，端到端 −10/12%）。优化史见 findings §6（S5）/§10（S10）。

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

| 形状 (M,K,N) | ours（S7+S10 调优后） | 调优前 | dist_seq |
|---|---|---|---|
| 2048, 2048, 2048 | **0.752**（S10: 0.83→0.752，gemm 0.21→0.165） | 1.07（MPC=1024） | 0.23 |
| 2048, 8192, 1024 | **0.765**（深 K 保持 s2，数字不变） | 1.10 | 0.35 |
| 256, 4096, 4096 | 0.70（S6 数，S7 配置可再降，未复测） | — | 0.17 |
| 128, 4096, 4096 | 0.59（同上） | — | 0.14 |

S7 两项调优：consumer BLOCK 8192/w8（占用受限，2.2×）+ 单块 MPC=M。**S10 更新**：producer `num_stages=1`（MPC>512 且 K<8192）。状态注记：与 #1/#2 同源——gemm_triton 0.165 vs vendor 0.095 仍是主因（1.7×）；逐 chunk 信号流水（恢复 GEMM/归约重叠）是下一步（设备原子 1.3µs/op 已探明可行）。优化史见 findings §7（S7.4）/§10（S10）。

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

## 6. 八卡实测（S9）——胜负反转

节点换为 8×BW1000 后全量复测（正确性 4/4 PASS at `-np 8`；基准 `bench/s9_bench_8card.py` → `bench/results/s9_8card.json`；**S11 复测** AG 单 chunk MPC + 多流池 A/B → `bench/results/s11_8card_{pool,nopool}.json`）。**结论：8 卡上四个算子全部落后 RCCL，包括 2 卡时的赢家 a2a（反转为慢 2.5-2.8×）。S11 把 AG 从 3.4× 收窄到 2.6×（单 chunk MPC，−24%），多流池被 A/B 判定为中性偏负（fabric 瓶颈，findings §11.5）。**

| 算子（形状） | 2 卡 ours vs RCCL | **8 卡 ours（S11）** | 组件 | **8 卡 RCCL** | 8 卡倍数 |
|---|---|---|---|---|---|
| AG Mpr2048/K2048/N2048 | 0.91 vs 0.31（2.9×慢） | **3.15**（S9 4.15） | gemm 1.63 / barrier 0.23 | 1.20 | 2.6×慢 |
| AG Mpr1024/K4096/N8192 | 1.75 vs 0.63（2.8×慢） | **6.94**（S9 7.81） | gemm 5.45 | 2.21 | 3.1×慢 |
| AG Mpr128（decode） | 0.62 vs 0.16 | 1.49 | gemm 0.53 | 0.34 | 4.5×慢 |
| RS M4096/K2048/N2048 | 0.51 vs 0.32（1.6×慢） | **1.04** | gemm 0.26 / barrier 0.21 | 0.37 | 2.8×慢 |
| RS M2048/K4096/N4096 | 0.68 vs 0.46（1.5×慢） | **1.12** | gemm 0.43 | 0.46 | 2.4×慢 |
| AR M2048/K2048/N2048 | 0.75 vs 0.23（3.3×慢） | 2.33 | **push 2.34** / gemm 0.17 | 0.31 | **7.5×慢** |
| AR M2048/K8192/N1024 | 0.77 vs 0.35（2.2×慢） | 1.62 | push 1.68 / gemm 0.17 | 0.42 | 3.9×慢 |
| a2a T1024/H2048/topk8 | **0.99 vs 1.60（1.6×快）** | 2.07 | dispatch 1.34 / combine 0.57 | 0.74 | **2.8×慢** |
| a2a T1024/H4096 | 2.05 vs 5.97（2.9×快） | 3.38 | dispatch 2.34 / combine 0.85 | 1.33 | 2.5×慢 |
| a2a T512/H2048 | — | 1.74 | meta ~0.5 / barrier ~0.25 | 0.46 | 3.8×慢 |
| a2a T128/H2048 | — | 1.47 | | 0.21 | 7×慢 |

根因（完整分解见 findings §9.3，S11.5 有池 A/B 的直接证据）：

1. **传输面扇出不扩展（主因）**：数据面 = 每 rank (R-1) 次**单播**引擎 memcpy。2 卡 = 1 条专用点对点链路 110GB/s；8 卡 = 每 rank 出 7 份、8 rank 同时推，共享 fabric 竞争下有效出带宽实测 **17-25GB/s/rank**（AR 2048²：56MB/rank ÷ 2.34ms ≈ 24）。RCCL 的 ring 在同一 fabric 上摊销——8 卡单次 all_to_all（29MB/dir）0.39ms（75GB/s/dir），比 2 卡（0.82ms）还快。**S11 判决：墙是 fabric 瓶颈而非本地队列瓶颈——多流池在池触发形状上中性（AG −0.5%、AR2048² −1.1%）到负（AR K8192 4MB op −9.2%），已默认关闭。**
2. **固定开销随 R 放大**：SymmBarrier 0.09→0.21-0.26ms（每 op 2-3 轮 = 0.5-0.8ms 纯税）；a2a 元数据 0.29→0.5ms。
3. **AG 尾延迟**：S9 记录 mean(36.7)≫p50(4.15)；S11 复测下单 chunk MPC + 串行 push 的 AG mean≈p50（3.46/3.15，干净），但多流池打开时尾延迟复发（mean 25.5）——与并发到达/flag 信号交互相关，配置敏感开放项。

八卡运行注意事项：8-PE 对称堆累计上限 ~1.8GB/rank（`bench/s9_probe_8pe.py` 梯度实测）——大形状基准必须逐形状 `free_symm()`；测试侧 moe 用 T=256（rank0 blob allgather 内存 ∝ T·topk·R）。

```bash
mpirun --allow-run-as-root -np 8 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 bench/s9_bench_8card.py     # 四算子全套 vs RCCL
```

---

## 7. 横向结论

- **能用的定义**：每个算子 = 正确性 3/3 PASS（fresh-data 多迭代 + 跨卡全量审计 + 干净退出，2 卡与 8 卡均验）+ 尾延迟根治（mean≈p50，8 卡 AG 除外——开放项）+ 分解基准可复现。
- **性能结论按卡数分**：2 卡上 a2a（无本地 GEMM）纯 planes 优势兑现为 2.3-2.9×，三个 GEMM 融合算子落后基线（主因 triton GEMM 本地质量，S10 后对 vendor 1.6-1.9×）；8 卡上传输面 (R-1) 单播扇出不扩展 + 固定开销放大，四算子全部落后 2.6-7×（§6）。
- **已定型的传输平面**（所有算子共用）：数据 = 宿主引擎 memcpy ~110GB/s（**2 卡点对点**；8 卡共享 fabric 下 17-25GB/s/rank）；信号/元数据 = 宿主 RMA putmem（16µs/条，消息内有序）；远端读 = 设备 plain load 74-83GB/s（L2 一致，混合自旋守卫）；**跨流编排 = 裸 hipEvent 1-4µs/次，torch `wait_stream` 贵一个数量级（10-40µs，S11 事件税）——多流编排不得走 torch 封装**。引擎并发本身真实（≥4MB×≥2 op 多流聚合 +8-16%，双工免费，≤2MB 死区）。设备发起的 dushmem 传输是 19.5MB/s 字节率天花板（P5），任何新算子先按这些平面重推导。
- **下一步（按收益排序，S11 八卡复测后修订）**：8 卡数据面改**环形/分层扇出**（多流池已被 A/B 判负关闭——墙在 fabric，只有对齐 RCCL 的 ring 摊销才可能追上）；barrier 换设备原子路径（S6.2 已证 1.3µs/op）或合并轮次（八卡 0.21-0.26ms/轮 × 2-3 轮）；gemm_triton 追 vendor（S10 已收 −6-21%，剩余 1.6-1.9× 是代码生成质量）；AG 尾延迟与并发到达的交互（S11.5）；S8 skew 基准复测（GEMM 逆差收窄后赢区空缺预期显著变小）。

各算子完整问题定位与优化时间线：[DCU_PORT_FINDINGS_zh.md](DCU_PORT_FINDINGS_zh.md)（§6=S5 调优、§7=S6 AR 移植、§8=S7 a2a 移植+双算子调优、§9=S9 八卡复测与反转分析、§10=S10 GEMM 扫参 v2/num_stages=1）。
