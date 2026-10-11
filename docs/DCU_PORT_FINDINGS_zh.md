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

新增基础设施（上游没有，DCU 栈必需）：

- **pydushmem**：DUSHMEM 宿主 API 的 ctypes 绑定 + 设备端 wrapper 位码 + triton `extern_libs` 链接装置 + `llc` 直出修补。
- **性能基准套件**（`bench/`）：传输微基准、一致性 probe、离群值/复发 probe、端到端 S4 基准（含 RCCL 基线）、S6 信号并发 probe。

上游 NVIDIA 层与其余 amd 能力（low-latency allgather、EP a2a、flash-decode、MoE rs、perf model 等）**未移植**——它们依赖 NVSHMEM 设备 API 的完整面或编译器后端扩展，按需再评估（见 §7 的移植性判断方法）。

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
