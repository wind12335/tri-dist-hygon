# triton-dist Hygon DCU 移植：问题全记录与性能发现

> 平台：Hygon BW1000 (gfx936) ×2，DTK 26.04（DUSHMEM，NVSHMEM 3.2.5 fork），triton 3.3.0+das（hcu 后端），torch 2.9.0+das，系统 OpenMPI 5.0.3。
> 路线（已定）：纯 Python 层移植——内核用 stock triton `extern_elementwise` 调 DTK 设备端 dushmem 位码，宿主侧用 ctypes 绑定，不改动上游仓库的 lib/include/plugin/csrc/dialect/patches。
> 本文档按时间线记录每个卡点：现象 → 根因（实验/gdb 证据）→ 修复 → 对后续研究的启示。多数发现对 AMD/其他 NVSHMEM 系栈也有参考价值。

---

## 0. 功能覆盖现状

上游 `triton_dist/kernels/amd/` 共 3 个内核文件，本移植（`kernels/dcu/`）**1:1 全部覆盖**并在 2 卡验证正确（fresh-data 3 迭代，max_diff ≤ 1e-4，干净退出）；S6 又从新版上游（triton-3.7 时代的 `gemm_allreduce.py`）移植了 gemm+allreduce：

| 上游 (amd) | 本移植 (dcu) | 语义 | 验证 |
|---|---|---|---|
| `all_gather_gemm.py` | `all_gather_gemm.py` | allgather A + 本地 GEMM：`C = A_all @ B^T`，producer 推 shard、persistent consumer 按块到达顺序消费 | ✅ test_ag_gemm.py |
| `gemm_reduce_scatter.py` | `gemm_reduce_scatter.py` | reduce-scatter over GEMM：`C_r = Σ_s (A_s@B_s^T)[rows_r]` | ✅ test_gemm_rs.py |
| `common_ops.py` | `common_ops.py` | 初始化、对称堆 barrier、流式 wait/set 原语 | ✅（被上面两个测试覆盖） |
| （新版上游）`gemm_allreduce.py` | `gemm_allreduce.py` | TP 列并行 GEMM + allreduce：`C = Σ_r a_r @ b_r^T`，每 rank 得完整 C | ✅ test_gemm_ar.py |
| （新版上游）`ep_a2a(_intra_node).py` | `moe_a2a.py` | MoE expert-parallel dispatch + combine（all-to-all，动态路由 + topk 加权归并） | ✅ test_moe_a2a.py |

新增基础设施（上游没有，DCU 栈必需）：

- **pydushmem**：DUSHMEM 宿主 API 的 ctypes 绑定 + 设备端 wrapper 位码 + triton `extern_libs` 链接装置 + `llc` 直出修补。
- **性能基准套件**（`bench/`）：传输微基准、一致性 probe、离群值/复发 probe、端到端 S4 基准（含 RCCL 基线）、S6 信号并发 probe。

上游 NVIDIA 层与其余 amd 能力（low-latency allgather、flash-decode、MoE rs、perf model 等）**未移植**——它们依赖 NVSHMEM 设备 API 的完整面或编译器后端扩展，按需再评估（见 §7/S8 的移植性判断方法）。

## 1. 性能现状（2×BW1000，fp16，50 iter）

S4 → S5（见 §6）后的最终数字（mean/p50 ms；NCCL 列 = RCCL allgather/reduce_scatter + vendor GEMM 串行）：

| 算子 | 形状 | S4 | **S5** | NCCL 序列 | S5 差距分解 |
|---|---|---|---|---|---|
| AG+GEMM | M_pr=2048,K=N=2048 | 16.7/1.38 | **1.37/1.12** | 0.30 | gemm_triton 0.50 vs vendor 0.16（**3.1×，主因**）；barrier 0.09；push 暴露 ~0.4 |
| AG+GEMM | M_pr=1024,K=4096,N=8192 | 3.1/3.05 | **2.20/1.95** | 0.62 | gemm_triton 1.43 vs 0.48 |
| AG+GEMM | M_pr=128(decode) | 16.7/0.67 | **0.84/0.61** | 0.16 | 小 M 下 vendor GEMM 0.09 难追 |
| GEMM+RS | M=4096,K=N=2048 | 0.61 | **0.56** | 0.32 | gemm_triton 0.30 vs 0.16；scatter+reduce+2×barrier ~0.26 |
| 纯传输 | 8.4 MB/卡 | 0.42 ms | 0.42 ms ≈ 20 GB/s | allgather 0.17 ms ≈ 49 GB/s | 传输面 2.4× |

S5 之后：**尾部停顿已消除（mean≈p50）**，p50 提升 1.2-1.6×。当前与基线的全部剩余差距仍集中在 gemm_triton 本身（plain kernel 最优配置离 vendor 还有 1.7×，consumer 形态 2.9×）。

---

## 2. 问题全记录

### P1. triton 外链 DTK 位码的完整配方（S1-S2，最大工程量）

**现象**：`@triton.jit` 内调用 dushmem 设备函数，符号找不到（hipModuleLoad Code 500）→ 逐步解决后又在汇编/初始化各处爆炸。

**根因与修复**（每条都经 gdb/反汇编验证）：

1. **extern 库名必须前缀匹配**：`extern_libs=[("dushmem", bundle.bc)]` 的名字用于 `need_extern_lib(mod, name)` 判断——它检查模块中未解析符号是否以 `name` 或 `__name_` 开头。名字不匹配 ⇒ 库静默不链接 ⇒ 加载时 "named symbol not found"。
2. **指针参数以 int64 标量传入**：triton 的 `pointer_type` 实例不可哈希（不能作 extern 分派 dict 的 key）。kernel 侧 `core.cast(ptr, i64)`，C wrapper 收 `unsigned long` 再转 `void*`。
3. **wrapper 必须返回 int**：`extern_elementwise` 要求 `ret_type.to_ir`，void 返回直接崩。可见性 `__attribute__((visibility("default")))`，否则 hidden。
4. **O3 + GlobalDCE 删掉未引用全局**：`dushmemi_device_lib_version_d` 必须用 `__device__ __attribute__((used))` 的指针保活，否则 `dushmemx_cumodule_init` rc=7（内部是对版本符号的 hipModuleGetGlobal）。
5. **DTK 位码带 `+sramecc` target-features，triton 的 llc 裸 `-mcpu=gfx936`** ⇒ `.amdgcn_target gfx936:sramecc+` 与汇编器预期不符。用 `opt -S` + sed 剥属性再 link 回（`make_fixed_bc.sh`）。
6. **xfer.bc 的 `v_writelane_b32` 常量总线冲突**：DTK AsmParser 拒绝，但 `llc -filetype=obj` 直出可过 ⇒ `llc_fixup.py` monkeypatch `make_amdgcn/make_hsaco`（注意要 patch 注册表里的类，registry 用 spec_from_file_location 加载，普通 import 会 patch 到另一份副本）。
7. **懒初始化时序**：`dushmemx_hostlib_init_attr` 会把设备态镜像重置为 -1，只有 `dushmem_malloc` 才触发 `init_device_state` 填充 ⇒ 必须在 `dushmemx_cumodule_init` 之前先 malloc 一次。
8. **warmup 值特化**（见 P7）。

**启示**：这套配方意味着"任何 DTK 设备库（RCCL、hipBLASLt 元数据等）都能以纯 Python 层接入 triton"——对封闭栈上做内核研究是通用基础设施。

### P2. DUSHMEM uid bootstrap 死路（gdb 级定位）

**现象**：`dushmemx_set_attr_uniqueid_args` 路径下，rank0 绑 0.0.0.0:0（内核随机分配端口），对端永远拨不通；20000 次重试 rc=7 后 teardown double-free。

**根因**：uid 结构里的 AF_INET sockaddr `02 00 ...` 被**当 NUL 结尾字符串拷贝**，第 1 字节后截断。`DUSHMEM_BOOTSTRAP_UID_SESSION_ID=<ip>:<port>` 只修 connect 侧，不修 listen 侧。死路，不修。

**修复**：`DUSHMEMX_INIT_WITH_MPI_COMM (1<<1)` + `dushmem_bootstrap_mpi.so` + 系统 OpenMPI。注意 attr.mpi_comm 必须是**指向 MPI_Comm 变量的指针**（NVSHMEM 语义），不是句柄值。

**启示**：单机多卡场景 MPI bootstrap 反而最稳，而且 MPI 还能当宿主侧同步面用。

### P3. torch 自带 OpenMPI 与系统 libmpi 互踩

**现象**：先 `import torch` 再 MPI_Init ⇒ PMPI_Comm_size 内 NULL+0x10 SEGV。

**根因**：torch wheel 里的 libmpi（改了 soname，符号相同）被先加载，系统 libmpi 的内部 PLT 调用 interpose 到 torch 那份未初始化的副本上。

**修复（铁律）**：`mpi_init()` + `hipSetDevice(rank)` 必须在 `import torch` **之前**。

### P4. DCU 缓存一致性模型（实测，非文档推断）

用 probe 逐项测量（peer 写 → 本地各种读）：

| 写入方式 | plain load | volatile / .cg（绕 L2） | atomic 读（L2 一致性点） |
|---|---|---|---|
| dushmem put/atomic（RMA） | ✅ fresh | ❌ **stale（DRAM 滞后）** | ✅ fresh |
| 裸 IPC store（peer VA 直接 tl.store） | ❌ 不失效 dest L2，撕裂/全零 | — | — |
| **copy-engine（hipMemcpyAsync 写 peer VA）** | ✅ fresh | ✅ fresh（一次性读） | ✅ fresh |

**推论**（全部实测复核）：
- 持续 volatile **自旋**在 engine 写下仍会死锁（见 P8）——一次性读 fresh ≠ 自旋安全；
- upstream amd 的"裸 IPC store scatter + plain load"**不能**直移到 DCU——这就是我们把 RS 拆成 local-GEMM → 对称堆 scatter → 本地 reduce 的原因。

**启示**：这张表本身就是论文素材——三类写路径 × 三类读路径的可见性矩阵，且"engine 写全一致"与"RMA 写只更新 L2"的组合在公开资料里没见过明确描述。

### P5. 【最大发现】device 发起的 dushmem 传输整条路 ~19 MB/s

**现象**：S4 首轮基准，AG push 每 MB 数据 ~52ms（1MB→55ms，16.8MB→878ms，纯带宽线性）。

**定位实验**（s4_transport_probe.py，16MB/方向全双工）：

| 路径 | 耗时 | 带宽 |
|---|---|---|
| device 侧 putmem_signal / putmem_nbi+quiet / +atomic_set（任意分块 256KB-16MB） | ~878ms | **19.1 MB/s** |
| 同上 + `DUSHMEM_DISABLE_LOCAL_ONLY_PROXY=1` | ~878ms | 19.1（无效） |
| device 侧 getmem_nbi（pull） | — | **内核 VMFault**（pull 路径不可用） |
| 宿主 putmem | 0.154ms | 108 GB/s |
| **hipMemcpyAsync → host.ptr(peer)（copy engine）** | **0.150ms** | **111 GB/s** |

**结论**：慢的不是某个 proxy 开关，而是**从内核发起的 dushmem 调用整体**（~5700×）；宿主发起的 RMA 与引擎直写都是 100GB/s 级。这解释了上游 amd 层为何用宿主 hipMemcpyAsync producer——我们在 DCU 上用实验重新发现了这条架构约束。

**修复**：producer 全部反转成宿主侧引擎路径（AG：per-dest 分块 memcpy + 一次 8×chunk 数的 staging 信号 memcpy；RS：同型 scatter；stream 顺序 = 数据先落、信号后到）。push 878ms → 0.42ms。

**启示/创新点钩子**：
1. 19.1 MB/s ≈ 固定字节管道且与 op 无关——值得在 8 卡上复核是否同样成立（若成立，任何"设备侧 NVSHMEM 风格 one-sided 内核"在这代栈上都要绕行）；
2. copy-engine 有 ~30µs/op 开销（64×256KB 掉到 8.7GB/s）⇒ 大块传输；SM-kernel 搬运（coherent plain load/store + 引擎信号）可能绕开两者——留作创新点；
3. host 编排的 producer 在真实训练循环里会增加宿主入队延迟，8 卡时 per-dest 数量 ×7，需要评估（或回到 SM producer 但只用 engine 信号）。

### P6. barrier 的两个坑

1. **只等自己槽位的 barrier 是 no-op**：每个 PE 等自己 ⇒ 时序巧合就过 ⇒ ~0.7% 撕裂读。必须等**所有** rank 的槽位。
2. **自旋必须走 atomic**（P4 矩阵），且 arrive 从 dushmem atomic_set 改为 engine memcpy 后 barrier 0.27→0.09ms（arrive 的 atomic_set 也是 19MB/s 慢路径公民）。

### P7. triton 缓存内核启动的陷阱（改一次崩一次的来源）

- **值特化**：warmup 时 `total=1` 会把 `==1` 烘进 cubin（2 字节 put、barrier 从第 2 轮起 no-op）⇒ `do_not_specialize` + 有代表性的 warmup 值。
- **i32/i64 槽位由 warmup 值决定**：指针传 0 会把 launcher 槽位定型为 i32，launch 时大 VA 截断。**warmup 必须传真实大 VA**（do_not_specialize 只阻止烘焙，不改槽位类型）。
- **persistent kernel 的循环步长必须等于 launch grid**：`for tile_id in range(pid, total, NUM_SMS)` 的 NUM_SMS ≠ 实际 grid ⇒ 丢 tile（静默错结果）。
- **模块级常量**在 jit 体里必须 `tl.constexpr(...)`，裸 int 是 NameError。

### P8. 自旋病理学（三个阶段，最后一个未完全解决）

1. **atomic RMW 自旋挡住 engine 写**：RMW 每圈写脏 flag line，peer 的引擎写挤不进连续 RMW 之间，偶发 20-50ms（首例 ~860ms ≈ 16MB@19MB/s，疑同源慢路径）。已定位：停顿全部落在 consumer 相位、push 相位（含信号 memcpy）0.3ms 已完成；consumer-only（无等待）40 迭代零停顿。
2. **volatile 持续自旋死锁**：绕 L2 读 DRAM，engine 写的 L2 更新永远看不见（一次性读 fresh 是 probe 的假安全感——probe 的同步点是当时的 atomic barrier）。教训：**一次性读新鲜 ≠ 持续自旋安全**。
3. **现行方案（混合自旋）**：128 次 volatile（纯延迟+乐观快路径）+ 1 次 atomic 探测（L2 一致性点，保证前进）。把 RMW 占空比降 128×。
4. **残余 ~1% 尾延迟**（10-50ms）：等待"在途 engine 写对本地可见"的偶发延迟。已证与数据路径无关、**NCCL 交织不触发**（200 op 混跑 0 停顿）、首例有 ~800ms 放大。疑点：引擎写完成 ≠ 对端 L2 失效流完成，失效吞吐 ~200MB/s 量级时 4MB 数据的失效积压正好 ~20ms。**→ S5.3 已定位并根治**：真正机制是信号写排在引擎数据队列后面（小 chunk 时整队列排空 ~1.6s）；信号改走 RMA 路径后全部消失。

### P9. SM 预算与提交顺序

自旋的 persistent consumer 会占满全部 CU，把同为 SM kernel 的 producer 饿死。规则：consumer grid 预留 2 CU + producer 先提交。（producer 转 copy-engine 后此约束大半消失，但 per-dest 信号内核仍是 SM kernel，保留余量。）

### P10. 基准方法学的两个 bug（都改过自坑）

1. RS 的参考实现必须**逐源配对 B**：语义是 `out_r = Σ_s (A_s @ B_s^T)[rows_r]`，错用本卡 B 对全部源求和 ⇒ md~0.17 的假 FAIL。
2. **GPU 与 CPU 的 randn 是两条流**：用 CPU 重新生成"应答数据"来校验 GPU 生成的数据 ⇒ 恒 BAD。校验数据必须在同侧同种子重生成。

---

## 3. 八卡能打赢 NCCL 吗（量化判断）

先说结论：**卡数本身不改变胜负；两个前置条件满足后 8 卡反而是我们的优势放大器。**

分解（以 M_pr=2048,K=N=2048,fp16 为例，2 卡实测外推）：

```
ours   ≈ barrier + max(push, gemm_triton)   （融合：通信被计算掩盖）
NCCL 序列 ≈ dist_ag + vendor_gemm           （严格串行）
```

- **现状（2 卡）**：ours 1.36 vs 0.30。即使通信**完全**被掩盖，ours 下限 = gemm_triton = 0.75ms，仍输 2.3×。瓶颈纯粹是 GEMM 调优，与通信无关。
- **8 卡通信量**：每卡 push 字节 ×(7/1)，NCCL allgather 同样 ×(7/1)（ring 带宽最优）。通信项对两者等比放大，**比值不变**；但 ours 的通信可与计算重叠，NCCL 序列不能 ⇒ 通信占比越大，融合的**结构性优势**越大。M_pr 越大（如 4-8K 行的 prefill/长序列），GEMM 项越重、掩盖越充分。
- **打赢的两个前置条件**：
  1. **gemm_triton 调优到 vendor 的 1.2-1.5×以内**（现在 4.7×）。BLOCK/warps/stages 扫一遍 + 8 卡上 NUM_SMS 重标定，这是 S5 头号任务，2 卡上就能完成大部分；
  2. **传输面从 20GB/s 提到 ≥40GB/s**：现在 per-dest 信号粒度太粗（dest 的全部 chunk 一起 ready），改 per-chunk 流水（信号 memcpy 跟着每块数据走）+ 多引擎队列并行提交。8 卡若直连拓扑成立，7 个 dest 的 memcpy 可真正并发。
- **额外的 8 卡不确定性（要实测）**：(a) 19MB/s 慢路径是否随卡数恶化；(b) ~1% 尾延迟在 7 个入向流下的聚合概率（最坏 1-(0.99)^7≈7%/op）；(c) 引擎写失效吞吐在高并发下是否成为新瓶颈。这三条都是 vcjob 阶段的第一批实验。

**对研究定位的建议**：目标不必是"打赢 NCCL 的 collective"（RCCL 有厂商调优的拓扑适配），而是** fused 算子 vs (NCCL + vendor GEMM) 序列**在真实 TP decode（161 AR/step、skew 场景）下的端到端步时——这正是融合+调度研究的用武之地，且 8 卡上通信占比放大对我们有利。

## 4. 创新点候选（从上述问题直接推导）

1. **一致性感知的传输选择器**（P4 矩阵 + S5.3 实证）：运行时按（写路径 × 读模式 × 等待模式 × 负载大小）选 engine/RMA/atomic 组合——S5.3 的"数据走引擎、信号走 RMA"就是第一个实例（小写混进大写在引擎队列里会灾难性排队）。可做成 triton-dist 的通用后端抽象。
2. **引擎队列的混合负载建模**（S5.3）：小 op 与大 op 同队列时的悬崖行为（64B 信号排在 8×1MB 数据后偶发 1.6s 排空）——量化成模型可指导所有引擎编排系统的 op 分流。
3. **慢路径规避的 SM-producer**（P5）：用 coherent plain load/store 的 SM 搬运内核替代 device-dushmem（若 8 卡确认慢路径仍在），保留引擎只做数据。
4. **尾延迟感知的 fused 算子调度**（S5.3 残余 + 8 卡聚合）：每 ctx 一次的确定性 iter-11 停顿与 8 卡下 7 入向流的聚合概率，是 skew-aware 调度论文的天然素材。
5. **混合自旋原语**（P8.3）：volatile 批 + atomic 探测的可调占空比，可作为语言级 `wait` 原语贡献回 triton-dist。

---

## 5. 复现入口

```bash
# 验收（2 卡）
mpirun --allow-run-as-root -np 2 --oversubscribe --bind-to none \
  -mca coll ^hcoll -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
  python3 tests/test_ag_gemm.py
mpirun ... python3 tests/test_gemm_rs.py

# S4 基准（含 RCCL 基线）→ bench/results/
mpirun ... python3 bench/s4_bench.py

# 传输/一致性/尾延迟 probe
mpirun ... python3 bench/s4_transport_probe.py
mpirun ... python3 bench/probe_memcpy_freshness.py
mpirun ... python3 bench/s4_rearm_probe.py
```

位码构建（首次）：`pydushmem/src/make_fixed_bc.sh`（需 DTK stock init/xfer.bc，见脚本头注释）→ `pydushmem/src/build_bc.sh` → `pydushmem/lib/pydushmem.bc`。

---

## 6. S5 优化记录（GEMM 调优 + 网格放开 + 信号传输根治）

### S5.1 GEMM tile 扫参（bench/tune_gemm.py，单卡，~170 配置 × 7 形状）

- **统一最优 `128×128×64 / w4 / s2`**（原配置 128×128×32/w8/s2）：大 shape +7%，小 M/深 K +40%（K=8192 时已达 vendor 的 0.89×）。要点：BLOCK_K 32→64 是主要收益来源，warps 8→4 次之；stages≥3 无一胜出（gfx936 上 pipeliner 深度换不来收益）。
- 大 shape 剩余差距 1.7×（0.293 vs 0.173 ms @4096×2048×2048）——不再是配置问题，是 tl.dot 代码生成质量（vendor 是 hipBLASLt 手写汇编）。
- 已接入 `matmul_local`（BM=64 if M≤512）与 AG consumer。

### S5.2 AG consumer 网格放开（过时的 SM 预留）

- 旧 `grid = min(n_sms-2, tiles)` 预留 2 CU 是 **SM-kernel producer 时代**的遗产；S4 之后 producer 是宿主引擎 memcpy（不占 SM）。
- 同一数学量：plain tile launch 0.323 ms vs 封顶 persistent consumer 0.758 ms（**2.3×**）。放开为 grid=tiles（persistent 循环退化为每块一个 tile）后 consumer 0.498 ms。
- 剩余 0.498 vs 0.293 的差 = per-tile flag 检查 + rank swizzle 计算 + EVEN_K 尾部分支 + localA 间接寻址——后续可再瘦。

### S5.3 【S5 核心发现】引擎信号写的悬崖停顿与 RMA 信号根治

S4 遗留的 AG 尾部（bench mean≈33ms vs p50≈1ms）定位完毕（bench/s5_signal_probe.py，MPC×传输方式矩阵，每格 100 op）：

| MPC（chunk 数/dest） | engine 信号 memcpy | host putmem 信号 |
|---|---|---|
| 512（4 chunk） | 1-2 次停顿，max ~16ms | **0 停顿，max 3.4ms** |
| 256（8 chunk） | 1-3 次 **~1.6 秒悬崖**，mean 17-20ms | **0 停顿，max 2.0ms** |
| 128（16 chunk） | 1-3 次 **~1.6 秒悬崖**，mean 18-20ms | **0 停顿，max 1.9ms** |

- **机制**：数据走引擎 memcpy 后，信号（64B 引擎写）在同一引擎队列里排在多 MB 数据的处理后面；chunk 变多（≥8/dest）时偶发整队列排空级停顿（~1.6s ≈ 全部数据按慢速路径回退的量级）。consumer 的 flag 自旋在整个排空期间空转。
- **修复（已为默认）**：数据仍走引擎 memcpy；信号改为 **event-sync 后由宿主 `dushmem_putmem`（RMA 路径）写入**——RMA 直写对端 L2，不经过引擎队列。代价仅每次 op 一次宿主 event 同步（~+0.05ms）。
- 效果：bench 全形状 mean 33→0.84-2.2ms（**mean≈p50**），正确性 3/3 PASS 不变。
- 残余（开放问题）：每个 ctx 生命周期内约一次 ~13ms 停顿，**确定性地出现在第 11 次迭代**（跨形状、跨 ctx 可复现）——按 op 计数而非时间的系统性事件，疑似运行时/驱动的周期性维护；量级已无害（+0.26ms/50 iter 的 mean），留待 8 卡阶段观察是否放大。

### S5.4 M_PER_CHUNK 甜点

验证 shape（M_pr=2048）扫 MPC：128→1.37 / 256→1.13 / **512→1.02** / 1024→1.57 ms（p50）。chunk 太碎引擎 op 数量多、太大则 consumer 流水浅；512（4 chunk/dest，2MB/chunk）最优。

### S5.5 结论与 8 卡展望更新

- 融合 op 的通信部分已被计算充分掩盖（ours 1.12 ≈ barrier 0.09 + gemm 0.50 + push 暴露 ~0.4）；
- 与 NCCL 序列的差距全部集中在 **gemm_triton 质量**（consumer 2.9×、plain kernel 1.7× vs vendor）——这是纯本地优化问题，与分布式无关；
- 8 卡判断不变但更乐观：通信占比放大对融合结构有利；传输面 20GB/s 的第二优先级（per-chunk 流水/多队列）与 gemm 1.7× 的本地差距，两者都关闭后 8 卡有实胜算。

---

## 7. S6 记录：gemm_allreduce 移植（新版上游 → DCU）

目标：把新版上游（triton-3.7 时代，v0.0.2-rc-187）的 `kernels/amd/gemm_allreduce.py`（TP 列并行 GEMM + allreduce，`C = Σ_r a_r @ b_r^T`，每 rank 完整 C）迁到当前 hygon triton 3.3.0 / torch 2.9.0 栈。

### S6.1 上游兼容性判断（为什么不能直接搬）

- 上游核心原语 `dl.wait/consume_token`、`st(semantic="release", scope="system")` 全部路由到 `_ext().distributed_wait(...)` —— **编译器后端扩展**（他们打了补丁的 triton + rocSHMEM），stock triton 3.3 没有；任何一字不差的搬法都不存在。
- 但每个原语都有 DCU 已验证等价物：pyrocshmem↔pydushmem、dl.wait↔混合自旋、IPC barrier↔SymmBarrier、静态配置表↔S5 调优结果。唯一开放问题是信号粒度（见 S6.2 探针）。

### S6.2 决策探针：微型设备原子的并发流水性（bench/s6_signal_concurrency.py）

上游按 tile 从 GEMM kernel 内发信号（设备发起）。S4 的 ~19MB/s 结论测的是**单 CTA 大块传输**，多 CTA 微型原子未测过。探针（2 rank 对发 8B `dushmem_uint64_atomic_set`）：

| 形态 | 总 op | p50 | 每 op |
|---|---|---|---|
| 1 CTA × 1 op | 1 | 0.29ms | **290µs**（单 op 裸延迟） |
| 1 CTA × 1024（串行） | 1024 | 30.0ms | **29.3µs/op** |
| 1024 CTA × 1 | 1024 | 2.0ms | 1.95µs/op |
| 512 CTA × 2 | 1024 | 1.35ms | **1.32µs/op** ✅ |
| 512 CTA × 64（压力） | 32768 | 12.8ms | 0.39µs/op |

**结论**：(a) 微型设备原子**可以流水**（并发 ~25× 于单 CTA 串行）；(b) 但 19.5MB/s 是**字节率天花板**——32768×8B=256KB→12.8ms 正好压线，与 S4 大块传输同一条管道。上游架构真正的死穴不在信号而在**数据面**：其 consumer 是设备发起的结果扇出（每 rank 8MB 级），在这条管道上要 ~400ms。**上游 per-tile 架构整体不可移植，必须按 DCU 平面重推导。**

### S6.3 移植架构（kernels/dcu/gemm_allreduce.py）

compute=逐 M-chunk 纯 GEMM（S5 配置）写自己的对称槽位；data=宿主引擎 memcpy 逐 (chunk×dest)（流序跟在该 chunk GEMM 后 → chunk c 的推送与 chunk c+1 的 GEMM 重叠）；signal=整行 flags 一次宿主 putmem（S5 零停顿模式原样）；reduce=单次启动的 consumer：按 pid 推导 chunk → 混合自旋各源 flag → fp32 求和写 C。own 槽位用流事件守卫，不经 flag。

### S6.4 三个坑（都修了，都值得记住）

1. **互相阻塞的 8B blocking put 死锁**：逐 chunk 一次 8B `dushmem_putmem`，两 rank 同时互发 → py-spy 显示双方都卡在 putmem 内部（注入信用式互等）。AG 每次只发整行一次所以从没踩到。
2. **nbi+quiet 也死/慢**：改 `putmem_nbi`+末尾 `quiet` 不死锁了，但 SM 满载下 put 完成被饿死（~8ms/op，大 shape 直接挂）。→ 回到整行一次 blocking putmem（S5 验证模式），逐 chunk 信号粒度留作后续（候选：设备原子，探针已证可行）。
3. **P9 复发（自旋 consumer 饿死 producer）**：最初逐 chunk 提前启动 consumer，其自旋 block 占满全部 CU，producer GEMM 被饿到 8ms/op、大 shape 直接"假挂死"。findings §P9 写过这条规则，重写时又踩了。修复：consumer 在信号落地后启动（AG 不踩是因为它的 consumer 就是 GEMM 本身）。附带把 4 次启动合并成 1 次（按 pid 推导 chunk，自旋条件保持标量）。
4. **P6.1 复发（漏了 push 流等 barrier）**：AG 有 `ag_stream.wait_stream(current)`，我第一版没加——对端 push 可能早于我方 barrier 释放执行，靠时序侥幸通过测试。已补。

### S6.5 v1 性能（bench/s6_bench_gemm_ar.py，50 iter，p50 ms）

| 形状 (M,K,N,MPC) | ours | RCCL AR+vendor GEMM | gemm 组件 | vendor GEMM | barrier |
|---|---|---|---|---|---|
| 2048,2048,2048,**1024** | **1.07** | 0.24 | 0.28 | 0.09 | 0.09 |
| 2048,2048,2048,512 | 1.50 | 0.24 | 0.48 | 0.09 | 0.09 |
| 1024,2048,4096,512 | 1.07 | 0.24 | 0.28 | 0.09 | 0.09 |
| 2048,8192,1024,**1024** | **1.10** | 0.35 | 0.28 | 0.24 | 0.09 |
| 2048,8192,1024,512 | 1.45 | 0.35 | 0.48 | 0.24 | 0.09 |
| 256,4096,4096,256 | 0.70 | 0.18 | 0.17 | 0.09 | 0.09 |
| 128,4096,4096,128 | 0.59 | 0.15 | 0.17 | 0.07 | 0.09 |

- **MPC=1024 显著优于 512**（大 shape 1.50→1.07）：2 chunk×BM=128 的 GEMM 远快于 4 chunk×BM=64（0.28 vs 0.48）——AR 的 GEMM 粒度比 AG 的传输粒度更该往大取。
- 差距分解（以 1.07 vs 0.24 为例）：gemm_triton 0.28 vs vendor 0.09（**3.1×，主因，与 AG 完全同源的本地质量问题**）+ barrier 0.09 + ~0.45 残差（推送与 GEMM 的内存带宽竞争、宿主信号同步、consumer 尾部——S6.1 待分解）。
- 优化路线（未做，按预期收益排序）：(a) gemm_triton 追 vendor（AG S5 同题）；(b) 逐 chunk 信号恢复流水（设备原子，S6.2 已证 1.3µs/op）；(c) 残差分解定位 0.45ms；(d) consumer 与下一 op 的 barrier 重叠。

### S6.6 工程备忘

- 探针/基准一律 `ulimit -c 0`（曾因编译错误 + 崩溃一次落了 ~3.9GB core dump）。
- k_check 类校验 kernel：warmup 传 tensor（定指针槽位）而非 data_ptr()（会定型成 i64 标量，`tl.atomic_add(i64,·)` 编译错）；标量 atomic 放在 `if` 守卫内在 3.3 上编不过，去掉守卫无条件执行。

---

## 8. S7 记录：MoE a2a 移植 + AR/a2a 双算子调优

目标：新版上游的 `kernels/amd/ep_a2a(_intra_node).py`（MoE expert-parallel dispatch/combine）迁到本栈（S7.1-S7.4），并对 gemm_allreduce 与 a2a 做调优（S7.5）。

### S7.1 上游兼容性判断（与 S6 同方法：按平面拆）

- **dispatch 数据面不可移植**：上游每 (token, topk) 由 SIMT kernel 设备发起 `putmem_signal_warp` 推一行（~4-14KB 动态路由）。按 P5/S6.2 的 19.5MB/s 字节率天花板，一行 14KB ≈ 0.7ms，T=2048×topk=8 共 16384 行——整条路死刑。**必须重推导**：本地 staging kernel 按 (dest, expert) 段压拢（段基址由元数据预计算，段内原子分配位置），然后每 dest **一次** engine memcpy 推整段（~110GB/s）。
- **combine 的 PULL 模型可直接移植**：上游 `dl.symm_at(input_buf, expert_rank) + ld`（owner 侧拉远端 expert_out 行）。决策探针 `bench/s7_peer_load_probe.py`（k_fill 远端写 → k_sum int64 累加核 peer 读）：正确性、跨轮新数据、带宽全部达标——**warm 74-83 GB/s**；首触 ~160-335ms 是一次性远端 VA 页表构建，ctx create 时 `k_touch` 每个 peer 的 out VA 预付掉。VA 预翻译走 `host.ptr`（i64 设备张量）。

### S7.2 移植架构（kernels/dcu/moe_a2a.py）

段代数（两侧排序约定，宿主参考实现 `_segment_tables` 与设备版 `kernel_meta_tables` 必须一致）：我的 staging 按 (dest, expert) 排序 → 每 dest 的整段连续（单次 memcpy）；对端 recv 按 (src, expert) 排序 → 我在对端 d 的落点连续于 `dst_row_base[d,0]`。max_recv_rows = max_tokens·topk·num_ranks（最坏全指向一卡——首版按 T·topk 算少了 R 倍，越界断言炸出）。

**元数据面（本栈的关键重推导）**：上游用设备 `putmem_signal_nbi_block` + `dl.wait` + grid barrier 收集 splits；这里改为——单 CTA kernel `kernel_pack_meta` 直接从 topk_idx 直方图计数（gather atomic）+ 打包 `[round|splits]` 一条消息 → 每 dest 一次宿主 `putmem`（RMA 快路，消息内有序，flag 不用单独等）→ 单 CTA `kernel_meta_tables` 混合自旋（S5 模式）每源 flag 到 round 后**在设备上**算出全部段表 + `host_plan[R,4]`（每 dest 的 [行数, 对端基址, 本地基址] + recv_total）→ 一次 pinned D2H 给宿主推沿用。自己的 meta 行没人 put，必须补一次 engine 本地拷贝（首版漏了：table kernel 自旋包括自己那行 → 死等）。

**相协议（2 道 SymmBarrier）**：D.end（所有 dispatch 数据落齐才能进 expert 相）+ C.start（所有 expert_out 写完才能 combine 读）。原本还有 D.start（防 peer 下一轮 push 覆盖我 expert 相还在读的 recv），**已删**——本 API 契约下 peer 要到 D.start(k+1) 必先过 C.start(k)（全部 expert 相 k 已完），传递性已闭合；首轮平凡安全。跨轮元数据覆盖安全由 D.end 传递性同理证明。

跳过（语义保持）：上游的重复 rank 检测/本地预合并（每个 (t,j) 独立 staging+pull，最多 topk 数据放大）、权重转发（权重留在 owner，combine 时乘）。

### S7.3 a2a 调优记录（分解探针 bench/s7_meta_decompose.py，host wall 分段）

v1（bincount→pinned D2H→MPI_Allgather→numpy 表→H2D）meta 组件 0.43ms。分解：**RMA putmem 本身只 ~16µs**、自旋等待 ~20µs（两 rank 相位对称）、大头是**torch 算子链的宿主启动开销**——`reshape→.long()→bincount→.to` 四个 op 光启动就 0.13ms，加两次 sync 的排空。三步优化：

1. 元数据面换 RMA（去 MPI 往返 + 宽 sync + numpy 表代数）；
2. `kernel_pack_meta` 单 CTA 融合"清零+直方图+打包+flag"（gather atomic），替换整条 torch 链——meta 0.46→0.29ms；
3. 删 D.start barrier（上面的传递性证明）——dispatch −0.09ms。

| 组件 (T2048,H2048,topk8) | v1 | v3 |
|---|---|---|
| meta | 0.43 | **0.29**（启动开销 0.13 + sync×2 + putmem 0.016 + spin+表核+D2H 0.11） |
| dispatch | 1.28 | **0.76**（= meta 0.29 + barrier 0.09 + staging+push ~0.38，push 34MB@110GB/s≈0.31 已贴底） |
| combine | 0.47 | 0.48（33MB 远端读 @74GB/s ≈ 0.45，带宽饱和无余量） |
| **full round** | **1.87** | **1.31** |

各形状（p50 ms，`bench/s7_bench_moe_a2a.py`，RCCL 列 = 等分 all_to_all×2 串行近似）：

| 形状 | ours full | dispatch | combine | RCCL 2×a2a | 倍数 |
|---|---|---|---|---|---|
| T2048 H2048 topk8 | **1.31** | 0.76 | 0.48 | 3.07 | **2.34×** |
| T2048 H4096 topk8 | **2.05** | 1.03 | 0.78 | 5.97 | **2.9×** |
| T1024 H2048 topk8 | **0.99** | 0.63 | 0.32 | 1.61 | 1.6× |
| T2048 H2048 topk4 | **0.92** | 0.60 | 0.30 | 1.61 | 1.7× |
| T256 H2048 topk8 | 0.74 | 0.55 | 0.18 | 0.49 | 0.66×（元数据/延迟地板，RCCL 胜小包） |

### S7.4 gemm_allreduce 调优（bench/s7_ar_decompose.py）

- **consumer 归约是占用受限不是带宽受限**：BLOCK 2048/w2 → 8192/w8 提速 **2.2×**（0.30→0.14ms @M2048×N2048）。MPC·N 恒为 8192 倍数（MPC%64==0, N%128==0）→ 无跨 chunk 边界块，已断言。
- **单块 GEMM（MPC=M）整体优于分块**：一次 GEMM 启动 + 每 dest 一次 push，免掉逐 chunk 事件/流调度。2048,2048,2048：**1.07→0.83ms**；2048,8192,1024：**1.10→0.765ms**（mpc1024 同形状 0.93）。
- 残差分解确认 barrier 0.09、push/信号路径无异常停顿；剩余差距仍是 gemm_triton 对 vendor（3.1×，AG 同题）。

### S7.5 坑（都修了）

1. **`tl.full((1,),…)` 存标量指针编不过**（"Value cannot be block type if pointer is not a block"）——要 `tl.store(ptr + tl.arange(0,1), v)` 把指针变 block。
2. **自旋 kernel 等了没人写的行**：putmem 只发 peer，自己的 meta 行要 engine 本地拷贝补上（症状 = 无输出的静默死等，600s 超时）。
3. **ctypes 调 MPI_Allreduce 是六个参数**（sendbuf, recvbuf, count, datatype, op, comm）——按 Allgather 的八参布局传会把 int 落进 op 槽 → PMPI 内 `0x1+0x54` 解引用 → SEGV@0x55。用 gdb isolate 探针 + objdump PMPI_Allreduce+0x1ad 定位。OpenMPI 句柄一律 `&symbol`（BSS 里的 struct 变量，如 `ompi_mpi_op_max`），不是 Fortran 小整数。
4. **pinned 缓冲区 send/recv 不得重叠**（非 IN_PLACE 的 overlap 是 UB）：v1 用切片视图省内存 → rank1 的 splits 被踩 → 路由错位。
5. **测试变换受 fp16 动态范围约束**（max 65504）：假 expert 用 `+row`（行号到 16K）→ combine 和 ~100K → inf。改 `*2+row%5` 有界变换。
6. `recv_total` 走设备张量返回（避免强制同步）；benchmark 的 expert kernel 用设备标量 mask 守卫（max grid），全轮保持流异步。

### S7.6 复现入口

```bash
mpirun ... python3 tests/test_moe_a2a.py          # MoE a2a 正确性（3/3 PASS）
mpirun ... python3 bench/s7_peer_load_probe.py    # peer-load 决策探针
mpirun ... python3 bench/s7_bench_moe_a2a.py      # a2a 全套 → bench/results/s7_2card_bw1000.json
mpirun ... python3 bench/s7_meta_decompose.py     # 元数据面分解
mpirun ... python3 bench/s7_ar_decompose.py       # AR 组件/consumer 扫参
```

---

## 9. S9 记录：八卡验收 + 四算子 vs RCCL（结果反转）

硬件切换到 8×BW1000 节点（128 核）。先探针后基准（`bench/s9_probe_8pe.py`、`bench/s9_bench_8card.py`，结果 `bench/results/s9_8card.json`）。

### S9.1 8-PE 基础面（probe）

- **dushmem MPI bootstrap 8 PE 一次通过**（uid 死路的结论不变，MPI 路径 R 无关）。
- **对称堆累计 ~1.8 GB/rank 可用**（64+256+512+1024MB 同时持有成功，+2048MB 失败）——a2a 的 recv+out 是最大分配（T1024/H4096 时 1GB），基准必须逐形状 `host.free`（`free_symm()`，MPI barrier 前后夹住）。
- 四个正确性测试全部 R 泛化后 **8 卡 4/4 PASS**（改动只有：assert 放开、RS/AR 参考和的 [0]+[1] 折叠泛化、moe 测试 T 按 R 缩——blob allgather 内存 = MRR=T·topk·R·H·2）。

### S9.2 性能：四算子全部落后 RCCL（p50 ms，rank0，含 2 卡对照）

| 算子（形状） | 2 卡 ours vs RCCL | **8 卡 ours** | 组件 | **8 卡 RCCL** | 8 卡倍数 |
|---|---|---|---|---|---|
| AG Mpr2048/K2048/N2048 | 1.12 vs 0.30（3.7×慢） | 4.15 | gemm 1.72 / barrier 0.29 | 1.22 | 3.4×慢 |
| AG Mpr1024/K4096/N8192 | 1.95 vs 0.62（3.1×慢） | 7.81 | gemm 6.17 | 2.24 | 3.5×慢 |
| AG Mpr128（decode） | 0.61 vs 0.16（3.8×慢） | 1.56 | gemm 0.56 | 0.36 | 4.4×慢 |
| RS M4096/K2048/N2048 | 0.56 vs 0.32（1.7×慢） | 1.18 | gemm 0.31 / barrier 0.26 | 0.375 | 3.1×慢 |
| RS M2048/K4096/N4096 | 0.77 vs 0.42（1.8×慢） | 1.24 | gemm 0.52 | 0.46 | 2.7×慢 |
| AR M2048/K2048/N2048 | 0.83 vs 0.23（3.5×慢） | 2.25 | **push 2.25** / gemm 0.21 | 0.32 | **7.0×慢** |
| AR M2048/K8192/N1024 | 0.77 vs 0.35（2.2×慢） | 1.64 | push 1.63 / gemm 0.17 | 0.42 | 3.9×慢 |
| a2a T1024/H2048/topk8 | **0.99 vs 1.60（1.6×快）** | 2.20 | dispatch 1.47 / combine 0.62 | 0.75 | **2.9×慢** |
| a2a T1024/H4096 | 2.05 vs 5.97（2.9×快） | 3.44 | dispatch 2.44 / combine 0.86 | 1.34 | 2.6×慢 |
| a2a T512/H2048 | — | 1.76 | meta 0.52 / barrier 0.28 | 0.45 | 3.8×慢 |
| a2a T128/H2048 | — | 1.50 | | 0.21 | 7×慢 |

**核心反转**：2 卡上唯一超基线的 a2a，8 卡上落后 2.9×；三个 GEMM 算子的差距从 1.7-3.7× 恶化到 2.7-7×。

### S9.3 根因分解（为什么 R=8 翻转）

1. **传输面扇出不扩展（主因）**：我们的数据面 = 每 rank (R-1) 次**单播**引擎 memcpy。2 卡 = 1 条专用点对点链路 110GB/s；8 卡 = 每 rank 出 7 份、8 rank 同时推，共享 fabric 竞争下有效出带宽实测 **17-25GB/s/rank**（AR 2048²：56MB/rank ÷ 2.25ms ≈ 25；K8192：28MB ÷ 1.63 ≈ 17）。RCCL 用 ring 在同一 fabric 上摊销——8 卡时单次 all_to_all（29MB/dir）只要 0.39ms（75GB/s/dir），甚至比 2 卡（0.82ms）更快。
2. **固定开销随 R 线性放大**：SymmBarrier 0.09→0.28ms（(R-1)×4B 引擎写 + 8 槽自旋；每 op 2-3 轮 = 0.5-0.9ms 纯税）；a2a 元数据 0.29→0.52（7 次 putmem + 自旋 8 flag）。
3. **GEMM 本身**：单 rank 计算量不变时 gemm 组件不变（AR 0.21≈0.28），AG 因 M 总量 ×4 涨到 1.72——仍是对 RCCL 序列差距的次要项，8 卡下 push 反超成主项。
4. **AG 尾延迟复发**：8 卡 AG 的 mean(36.7)≫p50(4.15)（RS/AR/a2a 的 mean≈p50）——S5 根治的停顿在 8-way flag 自旋下重现（~30ms 级，待逐迭代直方图探针定位，S5.3 的队列排空悬崖候选）。

### S9.4 结论与下一步（按预期收益排序）

- **2 卡结论保持**（ planes 在小 R 下赢）；**8 卡结论**：单播引擎扇出 + 宿主编排固定开销在 R≥8 不可用，差距系统性。四算子作为**正确性已验证的 8 PE 研究底座**保留。
- (a) 数据面改**分层/环形**：引擎 memcpy 按 ring 分段（每 rank 只推邻居 + 本地中继），把 (R-1) 单播降为常数量长流水——对齐 RCCL 的摊销方式；
- (b) barrier 换已探明的设备原子路径（1.3µs/op 流水，S6.2）或合并轮次（a2a 已示范 3→2）；
- (c) AG 尾延迟定位（直方图探针）；
- (d) 逐 chunk 信号流水（S6 遗留）在 8 卡的收益重估。

---

## 10. S10 记录：GEMM 扫参 v2 —— num_stages=1 是 S5 漏掉的甜点

回到 2 卡，按 S8 指认的"GEMM 逆差是翻盘唯一钥匙"攻坚。S5 的 tile 空间已挖尽（结论：剩余差距是 tl.dot 代码生成质量），本轮扫 S5 没碰过的轴（`bench/tune_gemm2.py`，断点可续跑，结果 `bench/results/tune_gemm2.json`）：split-K（fp32 原子累加 + zero/cast 全计时）、GROUP 光栅化、persistent、cache_modifier(".cg")、eviction_policy、fp16 累加探针、以及含 s=1 的精调格。

### S10.1 主发现：num_stages=1（S5 的 STAGES 从 2 起，从未扫过 1）

单卡同场受控对照（plain kernel，p50）：

| 形状 | s2 | **s1** | 对 vendor 比 |
|---|---|---|---|
| 4096×2048×2048 | 0.293 | **0.247**（−16%） | 0.59→**0.66×** |
| 2048×2048×2048 | 0.180 | **0.167**（−13%） | 0.53→**0.60×** |
| 2048×4096×4096 | 0.485 | **0.426**（−15%） | 0.52→**0.59×** |
| 1024×4096×4096 | 0.282 | **0.266**（−6%） | 0.49→**0.52×** |
| 256/128（decode，M≤512） | 最优 | **劣 15%** | s2 保持 |
| 2048×1024×8192（深 K） | 0.275 | 0.316（劣） | s2 保持 |
| **8192×29568×8192（大 shape）** | **21.13** | **50.27（劣 2.4×！）** | s2 = 0.52× |

规律：**M>512 且 K<8192 → s1；decode（M≤512）与深 K（K≥8192）→ s2**。深 K 上 s1 灾难性（无双重缓冲 = 128 次 K 迭代全部裸露访存延迟）。大 shape 上 .cg 同样灾难（55ms）、persistent 劣 2×——三个负结果方向一致。

### S10.2 其余轴：全部负结果（一次性记录，别再扫）

- **split-K**：最好 0.971×（零头都吃不掉 zero+atomic+cast 的开销）——瘦形状本来就贴延迟地板；
- **persistent**：确认 S5.2 的结论在 plain kernel 上同样成立（med 1.73×）；
- **GROUP 光栅化**：g16 在宽 N/深 K 上 ~1% 量级（0.842→0.834），噪声级；
- **eviction_policy**：中性（med 0.998）；**fp16 累加**：本后端编译不过（tl.dot 不收 fp16 累加器）。

### S10.3 大 shape 实测（M8192 N29568 K8192，用户点名）

vendor（hipBLASLt）10.95ms = **362 TFLOPS**；triton 最优 21.13ms（仍是 128×128×64/w4/**s2**）= 172 TFLOPS（0.52×）。~4 TFLOP/call 的量级上宽 N 墙依旧——代码生成质量判决在大 shape 上不变（172 vs 362，差 1.9×）。

### S10.4 集成与端到端（同场 A/B，全部落库）

配置规则接入三处：`matmul_local`（RS）、AR producer（按 MPC）、AG consumer（按 M；与 plain kernel 同阈值）。s4/s7 同场 A/B（证据 `bench/results/s4_s10_{s2,s1,matmuls2}.json`）：

| 算子 | s2 | **s1** | gemm 组件 |
|---|---|---|---|
| RS M4096_K2048_N2048 | 0.565 | **0.511**（−10%） | 0.304→0.257（−15%） |
| RS M2048_K4096_N4096 | 0.776 | **0.685**（−12%） | 0.512→0.431（−16%） |
| AR 2048³（monolithic） | 0.83 | **0.752**（−9%） | 0.21→**0.165**（−21%） |
| AG M1024_K4096_N8192 | 2.017 | **1.864**（−8%） | 1.440→1.337 |
| AG M2048_K2048_N2048（mpc256） | 1.198 | **1.121**（−6%） | 0.504→0.468 |

正确性 3/3 PASS（ag/rs/ar）。**对 S8 的意义**：赢区空缺（153-255µs）里最大的 GEMM 逆差项显著收窄（AR 形状 gemm 对 vendor 从 2.2× 缩到 1.7×），skew 基准复测是下一步。
