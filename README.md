# tri-dist-hygon

[triton-dist](https://github.com/bytedance/triton-dist) 的 Hygon DCU（海光）移植 —— 单机多卡 allgather+GEMM 与 GEMM+reduce-scatter 融合内核，基于 DTK 的 DUSHMEM（NVSHMEM 3.2.5 fork）单边通信。

这是**研究代码**，用于 skew-aware 调度 / 通信-计算重叠方向。完整的问题定位记录（传输带宽、缓存一致性模型、自旋病理学等）见 **[docs/DCU_PORT_FINDINGS_zh.md](docs/DCU_PORT_FINDINGS_zh.md)**。

## 平台要求

| 组件 | 版本（验证过的组合） |
|---|---|
| GPU | Hygon BW1000 (gfx936) ×2，驱动可 peers 访问（`hipDeviceCanAccessPeer=1`） |
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
```

退出前必须 `host.finalize() → MPI_Barrier → MPI_Finalize → os._exit(0)`（析构竞争会 SEGV）。

## 测试 / 基准

```bash
# 正确性（2 卡）
mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_ag_gemm.py
mpirun ... python3 tests/test_gemm_rs.py

# 性能（含 torch.distributed RCCL 基线）→ bench/results/
mpirun ... python3 bench/s4_bench.py

# 诊断 probe：传输微基准 / 引擎写一致性 / 尾延迟
mpirun ... python3 bench/s4_transport_probe.py
mpirun ... python3 bench/probe_memcpy_freshness.py
mpirun ... python3 bench/s4_rearm_probe.py
```

## 目录

```
triton_dist/kernels/dcu/   融合内核（上游 amd 层 1:1 移植 + DCU 专属 producer/barrier）
pydushmem/                 DUSHMEM ctypes 宿主绑定 + 设备 wrapper 位码 + triton 外链装置
tests/                     2 卡正确性验收
bench/                     性能基准与诊断 probe（含 2×BW1000 实测结果）
docs/                      移植问题全记录（findings，创新点素材）
```

## 路线与边界

- 路线：**纯 Python 层**——内核用 stock triton `extern_elementwise` 外链 DTK 设备位码，宿主侧 ctypes；不携带也不修改上游的 lib/include/plugin/csrc/dialect。
- 上游 NVIDIA 层（low-latency allgather、EP a2a、flash-decode 等）未移植。
- 生产者（push/scatter/barrier arrive）是**宿主侧引擎 memcpy**：DCU 上 device 发起的 dushmem 传输实测 ~19 MB/s，宿主发起 ~110 GB/s（findings P5）——这是与上游最大的架构差异。

## License

MIT（继承上游 ByteDance triton-dist 许可，见 LICENSE；移植部分同样 MIT）。
