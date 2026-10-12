# tri-dist-hygon

[triton-dist](https://github.com/bytedance/triton-dist) 的 Hygon DCU（海光）移植 —— 单机多卡 allgather+GEMM、GEMM+reduce-scatter、GEMM+allreduce 融合内核与 MoE all-to-all（dispatch/combine），基于 DTK 的 DUSHMEM（NVSHMEM 3.2.5 fork）单边通信。

这是**研究代码**，用于 skew-aware 调度 / 通信-计算重叠方向。完整的问题定位记录（传输带宽、缓存一致性模型、自旋病理学等）见 **[docs/DCU_PORT_FINDINGS_zh.md](docs/DCU_PORT_FINDINGS_zh.md)**；四个融合算子的可用性、用法、执行指令与实测数字见 **[docs/OPS_GUIDE_zh.md](docs/OPS_GUIDE_zh.md)**。

## 算子总览（2×BW1000 实测，fp16，p50 ms）

| 算子 | 模块 | 正确性 | ours | 基线¹ | 结论 |
|---|---|---|---|---|---|
| AG+GEMM | `all_gather_gemm.py` | ✅ 3/3 | 0.91 | 0.31 | 可用；S10+S11 调优 1.40→0.91，主因 triton GEMM 质量 |
| GEMM+RS | `gemm_reduce_scatter.py` | ✅ 3/3 | **0.51** | 0.32 | 可用；最接近基线（1.6×，S10: 0.56→0.51） |
| GEMM+AR | `gemm_allreduce.py` | ✅ 3/3 | **0.75** | 0.23 | 可用；S7+S10 调优 1.07→0.752 |
| MoE a2a | `moe_a2a.py` | ✅ 3/3 | **1.31** | 3.07 | **快 2.34×（H4096 快 2.9×）** |

¹ #1-#3：vendor GEMM + RCCL 集合通信串行；#4：RCCL all_to_all ×2。各算子的形状、组件分解与执行指令见 OPS_GUIDE。

**八卡（S9 + S11 复测）**：正确性 4/4 同样 PASS（`-np 8`），但性能胜负反转——8 卡上四算子全部落后 RCCL 2.4-7.5×（含 a2a 反转为慢 2.5-2.8×）。根因：数据面 (R-1) 单播引擎扇出在共享 fabric 上有效带宽掉到 17-25GB/s/rank（2 卡点对点 110GB/s）+ 宿主编排固定开销随 R 放大（barrier 0.09→0.21-0.26ms）。**S11 复测（findings §11.5）**：AG 单 chunk MPC 迁移成功（M2048 4.15→3.15，−24%，对 RCCL 3.4×→2.6×）；多流 push 池经同场 A/B 判定为**中性偏负**（AG/AR 8MB op 中性、AR 4MB op −9%，且 AG 尾延迟随池复发）——扇出墙是 fabric 瓶颈而非本地队列，池已默认关闭（`TD_PUSH_STREAMS=8` 可开），追 RCCL 只剩环形/分层扇出一条路。数字见 OPS_GUIDE §6；对称堆 8-PE 上限 ~1.8GB/rank（`bench/s9_probe_8pe.py`）。

## 平台要求

| 组件 | 版本（验证过的组合） |
|---|---|
| GPU | Hygon BW1000 (gfx936) ×2 或 ×8，驱动可 peers 访问（`hipDeviceCanAccessPeer=1`） |
| DTK | /opt/dtk-26.04（含 DUSHMEM、/opt/dtk/llvm 的 clang/opt/llvm-link） |
| triton | 3.3.0+das wheel（hcu 后端） |
| torch | 2.9.0+das |
| MPI | 系统 OpenMPI 5.0.3（/opt/mpi）—— dushmem MPI bootstrap 用 |

关键环境变量：GPU 进程一律 `export HSA_FORCE_FINE_GRAIN_PCIE=1`；每个 rank 独立 `TRITON_CACHE_DIR`（JIT 缓存竞争）。

## 构建（一次性）

仓库**不含**设备位码 `pydushmem/lib/pydushmem.bc`（由 DTK 位码派生，不入库）。从本机 DTK 提取 stock 位码后两步构建：

```bash
# 1) 从 DTK 安装中找到 dushmem 设备位码 init.bc / xfer.bc
#    （随 DTK 发布包分发，形如 dtk-26.04/zks-lib-dushmem-* 或 dushmem/lib/device/ 下）
mkdir -p dushbc && cp <你的>/init.bc <你的>/xfer.bc dushbc/

# 2) 剥 +sramecc target-features（triton llc 用裸 -mcpu=gfx936，属性冲突）
bash pydushmem/src/make_fixed_bc.sh dushbc

# 3) 编 wrapper + llvm-link 出 bundle
INIT_BC=dushbc/init_fixed.bc XFER_BC=dushbc/xfer_fixed.bc \
  bash pydushmem/src/build_bc.sh
# → pydushmem/lib/pydushmem.bc
```

## 使用

两条铁律（顺序错了就是 SEGV，见 findings 文档 P3）：

1. **MPI init + `hipSetDevice(rank)` 必须在 `import torch` 之前**（torch wheel 自带 OpenMPI 会 interpose 系统 libmpi）；
2. `@triton.jit` 的脚本落盘成 `.py` 再跑（`python -c` 因 inspect 拿不到源码）。

```python
from pydushmem import host as pHost
mpi, comm = pHost.mpi_init()          # 在 import torch 之前
# ... hipSetDevice(rank) ...
import torch
from triton_dist.kernels.dcu import all_gather_gemm as ag
ctx = ag.create_ag_gemm_intra_node_context(M, K, torch.float16, torch.float16,
                                           rank, world_size, mpi, comm)
ag.ag_gemm_intra_node_op(a, B, c, ctx)      # C = allgather(A) @ B^T

from triton_dist.kernels.dcu import gemm_allreduce as gar
ctx2 = gar.create_gemm_ar_intra_node_context(M, N, torch.float16, torch.float16,
                                             rank, world_size, mpi, comm,
                                             M_PER_CHUNK=1024)
c = gar.gemm_ar_intra_node(a, b, ctx2)      # C = sum_r a_r @ b_r^T（完整 C）

from triton_dist.kernels.dcu import moe_a2a
ctx3 = moe_a2a.create_moe_a2a_intra_node_context(
    num_experts=32, hidden=2048, dtype=torch.float16,
    max_tokens=2048, topk=8, rank=rank, num_ranks=world_size, mpi=mpi, comm=comm)
recv_total = moe_a2a.moe_dispatch_intra_node_op(tokens, topk_idx, ctx3)
# ... expert 相写 ctx3.out_ptr 的前 recv_total 行 ...
moe_a2a.moe_combine_intra_node_op(ctx3, topk_idx, output, topk_weights)
```

退出前必须 `host.finalize() → MPI_Barrier → MPI_Finalize → os._exit(0)`（析构竞争会 SEGV）。

## 测试 / 基准

```bash
# 正确性（-np 2 或 -np 8，脚本已 R 泛化）
mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_ag_gemm.py
mpirun ... python3 tests/test_gemm_rs.py
mpirun ... python3 tests/test_gemm_ar.py
mpirun ... python3 tests/test_moe_a2a.py

# 性能（含 torch.distributed RCCL 基线）→ bench/results/
mpirun ... python3 bench/s4_bench.py        # AG / RS（2 卡）
mpirun ... python3 bench/s6_bench_gemm_ar.py  # GEMM+AR（2 卡）
mpirun ... python3 bench/s7_bench_moe_a2a.py  # MoE a2a（2 卡，vs RCCL 2×all_to_all）
mpirun -np 8 ... python3 bench/s9_bench_8card.py  # 四算子全套（8 卡，vs RCCL）

# 诊断 probe：传输微基准 / 引擎写一致性 / 尾延迟 / 设备原子并发 / GEMM 扫参
mpirun ... python3 bench/s4_transport_probe.py
mpirun ... python3 bench/probe_memcpy_freshness.py
mpirun ... python3 bench/s4_rearm_probe.py
mpirun ... python3 bench/s6_signal_concurrency.py
mpirun ... python3 bench/s11_probe_multistream.py   # 引擎多流并发矩阵
mpirun ... python3 bench/s11_probe_evtax.py         # 跨流事件编排开销
python3 bench/tune_gemm2.py            # GEMM 扫参 v2（单卡，含 split-K/s=1 等新轴）
```

## 目录

```
triton_dist/kernels/dcu/   融合内核（上游 amd 层 1:1 移植 + DCU 专属 producer/barrier）
pydushmem/                 DUSHMEM ctypes 宿主绑定 + 设备 wrapper 位码 + triton 外链装置
tests/                     正确性验收（-np 2 / -np 8 均可）
bench/                     性能基准与诊断 probe（含 2× 与 8×BW1000 实测结果）
docs/                      移植问题全记录（findings）+ 算子使用指南（OPS_GUIDE）
```

## 路线与边界

- 路线：**纯 Python 层**——内核用 stock triton `extern_elementwise` 外链 DTK 设备位码，宿主侧 ctypes；不携带也不修改上游的 lib/include/plugin/csrc/dialect。
- 已移植：AG+GEMM、GEMM+RS（旧上游 amd 层 1:1）、**GEMM+AR（新版上游 gemm_allreduce，按 DCU 平面重推导——上游的 per-tile 设备信号/扇出架构依赖编译器扩展且撞 19.5MB/s 设备管道天花板，findings S6）**、**MoE a2a（新版上游 ep_a2a：dispatch 数据面重推导为 staging+整段 engine 推送，combine 的 PULL 模型直接移植，元数据走 RMA+设备段表，findings S7；2×BW1000 上比 RCCL 2×all_to_all 快 2.3-2.9×）**。
- 未移植：上游 NVIDIA 层与新版 amd 的 low-latency allgather / flash-decode / MoE rs 等（依赖 NVSHMEM 设备 API 完整面或编译器后端扩展）。
- 生产者（push/scatter/barrier arrive）是**宿主侧编排**：数据走引擎 memcpy（DCU 上 device 发起的 dushmem 传输实测 ~19 MB/s，宿主发起 ~110 GB/s，findings P5），**信号走整行 RMA putmem**（小引擎写混进大数据写会偶发 ~1.6s 队列排空悬崖，findings S5.3；逐 chunk 的 8B blocking put 会互相死锁，findings S6.4）——`all_gather_gemm.SIGNAL_MODE` 可切换。
- GEMM tile 配置经扫参定型（`128×128×64/w4`；**num_stages 按形状切**：M>512 且 K<8192 → s1（S10 发现，−6-21%），否则 s2——findings S5.1/S10）；AG consumer 每 tile 一个 block（findings S5.2）；AR 取单块 MPC=M（S7+S10：1.07→0.752ms），consumer BLOCK 8192/w8（占用受限，findings S7.4）。
- **push 扇出（S11）**：AG/RS/AR 的数据推送过 `common_ops.fan_out_memcpy_async` 侧流池（裸 hipEvent fork/join，≥4 op × 每个 ≥4MB 才启用 = 八卡扇出形状；`TD_PUSH_STREAMS=1` 关闭）；**AG 的 M_PER_CHUNK 取 M_per_rank（每 dest 单 chunk）最优**——引擎单 op 带宽有台阶（2/4/8MB ≈ 25/48/70 GB/s），2 卡 −3~19%（findings S11）。
- 自旋 consumer 不得与 producer GEMM 同时占满 SM（P9，AR 移植时复发过一次，findings S6.4）。

## License

MIT（继承上游 ByteDance triton-dist 许可，见 LICENSE；移植部分同样 MIT）。
