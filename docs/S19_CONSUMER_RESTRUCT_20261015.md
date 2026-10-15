# S19 consumer-form GEMM 重构 — 单卡 (2026-10-15)

Track-1：修我们自己的 consumer 形态 GEMM 结构开销（S18 结论：39.60ms vs plain triton
21.14 = 1.87×，差距 100% 在 GEMM 计算）。单卡分解（bench/s19{,b,c,d,e,f}*.py），
M8192 N29568 K8192 fp16，config 128×128×64 w4 g8。

## 分解链（单卡）

| 步骤 | 结论 |
|---|---|
| s19 | flag 门控（预置位）≈0ms；kernel CODE 结构 = 9.6ms（45%） |
| s19b | 2³ 消融：双 A 路径分支（两条活寻址形式穿过 k 循环）= **−5ms 主凶**；acc 形式 −1.2ms；swizzle ≈0（免费且必要） |
| s19c | 重构可部署形式（保真实语义：门控+swizzle+双源）30.72→25.23ms（−17.9%），A 用 int64 VA 比 tensor 传参更快 |
| s19d | 打补丁后的生产 kernel 回归：26.18ms md 1.2e-4；**EVEN_K=False 尾段 md 6.1e-5**（探针没覆盖过，新验证）；R=1 本地带 25.07 |
| s19e | pred kernel（i1 生产形态 = **grid128 持久**）：s2=39.33 / **s1=31.48** / 满格 s2=25.88 |
| s19f | 持久形态 old-vs-new（同进程）：**40.4 → 31.1/32.0 = −23%**，md 1.2e-4 |

## 落进生产的改动

`gemm_ag_tr.py::kernel_gemm_ag_tr_consumer`（满格 launch）与
`ag_tr_pred.py::kernel_gemm_ag_tr_consumer_pred`（持久 launch，保 tile 循环——
grid 封顶 128 是为了给 pull lanes/observer 留 CU）同法重构：

1. **双源分支只选标量**：`a_base`（localA 或 ws 指针）+ `row_bias`
   （own: rank*M_per_rank / peer: 0），k 循环只见一条寻址形式
   `a_base + (rm - row_bias)*stride_k + rk`（原版 own 路径 `rm % M_per_rank`
   与之严格相等——pid_m 带构造保证）
2. 单一 `stride_k` 参数（真实 launch 恒 stride_am==stride_lam==K、
   stride_ak==stride_lak==1，host 侧 assert），签名 4 个 stride 折叠为 1
3. `acc = tl.dot(a, b, acc)` 融合累加器
4. 去掉 store 侧 rm/rn 取模重算（row_bias 形式不再覆写 rm）
5. EVEN_K=False 尾段保留（同形式重算 A_BASE）；hybrid 门控、rank swizzle、
   表驱动映射、DIAG、acquire 语义逐字保留

## LAW：num_stages 分形态

- **满格 consumer**（gemm_ag_tr，tiles CTAs 无循环）：deep-K 必须 s2（S17：
  M8192/K8192 s1=45ms vs s2≈22）——S10 规则带 K<8192 guard。
- **持久 grid128 consumer**（ag_tr_pred，i1 形态）：**s1 更快**（s19e：
  31.5 vs s2 39.3，+25%）。机制：s2 = 64KB smem/CTA → 80 CU 只能驻留 ~80/128
  CTA → 1.6 波。M-only 规则（`1 if M > 512 else 2`）与形态恰好一致
  （tiles>128 ⟺ M 大），**保留 M-only 规则**——给 ag_tr_pred 加 S17 guard 是
  回归，已测已撤（今日踩坑）。

## 遗留

- s19g（constexpr 折叠 world_size/rank/m_chunk_num）：25.55/27.12 vs 参考 26.18，
  两相同臂自身差 1.6ms = 纯噪声，**无收益**——swizzle 除法链每 CTA 只跑一次，
  被 128 次 k 迭代摊薄。不改生产 kernel。
- 满格形态仍比 plain 慢 4-5ms（25.9 vs 21.1）——剩余 = 门控/swizzle 语义 +
  重构后剩余结构，待 2 卡 A/B 后再决定是否继续剥。
- 2 卡最终 A/B + 600 轮 soak（验收纪律）待卡。
- 持久形态满格 25.88 vs 持久 31.1：i1 若能让 consumer 与 pull 充分重叠，
  封顶本身可能是净赢——2 卡 soak 时用 I1_CONSUMER_GRID 扫一遍。

## 复现

```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1
python3 bench/s19_probe_consumer_gemm.py      # 分解（vendor/plain/r1/r2+sweep）
python3 bench/s19c_probe_restructured.py      # 可部署形式探针
python3 bench/s19d_regression_patched.py      # 生产 kernel 回归（含 EVEN_K 尾段）
python3 bench/s19e_pred_perf.py               # pred 持久形态 s1/s2/满格
python3 bench/s19f_old_vs_new.py              # old-vs-new A/B（需 /tmp/old_ag_tr_pred.py
                                               #  = git show HEAD:.../ag_tr_pred.py）
SOLO_ROUNDS=200 python3 tests/dbg_consumer_solo.py   # 0/600 三种 grid 形态
```
