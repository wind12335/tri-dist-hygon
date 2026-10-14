# AG-GEMM 撕裂：取证闭环与修复（S16，2026-10-14）

一句话：**撕裂已修复**。verify+repair 落地后 soak **0 fails / 7200
rank-rounds（v21 2×600 + v22 2×1200，worst md=0.00003）**，同台基线为
17/2400（v20）；校验器在现场捕获并修复了 32 次真实撕裂事件（修复率 ≈
历史撕裂率，证明正好接住原本漏出的损伤）。L1 偏斜矩阵（30 格 × 40 轮
× 2 rank，全 kernel 门控 = 历史最差路径）在修复路径上 **30/30 格
fails=0**（md_max 4.3e-5）【实测，2026-10-14】。开销：稳态每轮
+0.05-0.16ms（中位 0.63→0.68-0.79ms，+8-25%；一次性极值 839ms =
repair 内核首次编译）。

## 1. 两类事件（分类学，全部实测）

- **MIXTURE 类**：旗标先于数据落地 → consumer 读到整段/落地前缘的
  旧内容。rig16 复现（8/3000，x=1.00 纯旧内容 / 前缘部分混合，
  per-64-line lstsq resid=0.00000 完美拟合）。v17 的 wire-ordered
  旗标（4B 引擎复制同队列）针对此类。
- **NOISE 类**（真实管线残留）：单个 N-tile、每 damaged tile ≈ 一个
  BLOCK_K=64 k-slice 的贡献质量被同尺度随机内容替换；匹配**任何**
  已知基线均 NOFIT（prev 代、行混合、A/b 行、符号/缩放/量化、跨 M-tile
  共享 dB 联立求解）。幅度谱含 10× 事件（worst 1.296，内容 ≈ std-1
  原始 randn 中间态量级）→ 读错地址/读路径瞬态特征。

## 2. 取证闭环（谁有罪，逐项排除）

| 检验 | 结论 |
|---|---|
| agoff12：10 个失败快照离线 CPU 真值对比 | ref 侧逐位一致（0.00000），c 偏 0.07-0.12 → **op 有罪** |
| 全部 47 次真实 op 失败的 ws 快照（v3 forensics） | **47/47 CLEAN → 内存内容正确，瞬态只在 consumer 在飞读取** |
| agoff11：跨 M-tile 共享 dB 联立求解 | B 侧共享损坏否证（无突出 k-slice） |

隔离矩阵（全部干净，只有完整 op 撕裂 ~0.3-0.7%/rank-round）：
consumer solo 0/6000（3 臂）；无关 GEMM+并发 fabric 拉 0/1500；
本地同 VA 写竞态 0/8000；单侧跨卡拉取竞态 = 仅 MIXTURE 类 8/3000；
**对称负载 0/3000（rig17）**；**按块重叠 0/3000（rig18）**；
**无全局屏障协议完整复刻 0/3000（rig19：stage→远程 release arrive→
本地 GPU 自旋→pull→ready→ack 门控 restage，跨轮重叠）**；
observer 关闭（v19 OBS=0）仍 4/2400 → observer 无罪。

## 3. 顺带修掉的真 bug（保留，但非主因）

ack（"我的 consumer 已消费完各源行"）原在**调用方流**上入队 —— 它
并不排在 consumer 之后（join 才做该排序），即 ack 在 consumer 仍在读
时就发出，快对端可把 r+1 轮直接 restage 进我的在飞读取。v20 实测：
修复后 17/2400，撕裂仍在 → 该 bug 真实但非本撕裂主因；修复（ack 改在
consumer 流上入队）保留。教训：rig19 的 ack 天然按主机 sync 排序，故
干净 —— 复刻 rig 与真实 op 的差异要逐流核对入队序。

## 4. 修复：verify + repair（triton_dist/kernels/dcu/ag_tr_pred.py）

机制选择：瞬态不可在源头捕获（内存正确），故**验证结果本身**：
- `kernel_i1_verify`：每输出 tile 重算 SPOT=4 个元素的全程 fp32 点积
  （从已知的正确内存 ws+b）与 c 比较；tear σ≈4e-3 ≈ 340×fp16 ulp，
  tol=1e-3 分离舍入与损伤，P(漏检)≈0.1%/事件。无条件 xchg 1/0 写
  标志（免预清零）。
- host 读标志（小事件同步 + 2KB D2H），`kernel_i1_tile_repair` 仅对
  坏 tile 从同一正确内存重算（1 CTA/tile，罕见路径）。
- 排序：consumer → verify → repair → ack → join 全部在 consumer 流
  上保序（ack 先于 repair 也安全：对端 restage 只写对端 ws）。
- 旋钮：`I1_VERIFY`（默认 1）、`I1_VERIFY_SPOT`、`I1_VERIFY_TOL`、
  `I1_VERIFY_LOG`；计数 `ictx.verify_hits`。
- 注意：op 内含小事件同步，返回语义从"纯入队"变为"验证完成"（修复
  在返回前已入队，join 保证调用方排在修复后）。后续如需削尾延迟可
  异步化（把验证藏进下一轮 fork 之下）。

## 5. 复现实验资产

- tests/dbg_sym_race.py（rig17）、tests/dbg_chunk_race.py（rig18）、
  tests/dbg_nobar_race.py（rig19，注意平面方向：arrive/ack 均为
  **远程写入等待方平面**，写反 = 双方永久自旋）
- /tmp/run_v17/19/20/21/22.sh、/tmp/ag_v2*.log、/tmp/agoff11.py、
  /tmp/agoff12.py
