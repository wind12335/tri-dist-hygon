# S20/S21 — 8 卡会话 runbook (2026-10-15)

单卡准备已完成的生产改动：S19 consumer 重构三连（gemm_ag_tr / ag_tr_pred /
all_gather_gemm，−15~−23%，commits 4080976 / 9d97e23 / 3dbfa95）。8 卡会话跑两个
脚本，互不依赖，先跑哪个都行（建议先 s21 出可信度数字，再 s20 出预测器结论；
卡时紧就只跑默认集）。

## 1) s21 — 上游口径 TP-MLP 基准（可信度实验）

```bash
bash bench/s21_run.sh          # 默认: ag_rs, LLaMA-3.1-70B, M{512..8192}, ~数分钟
# 卡时富余:
S21_MODES=ag_rs,gemm_ar S21_MODELS=LLaMA-3.1-70B,LLaMA-3.1-405B bash bench/s21_run.sh
```

移植自上游 `test_tp_mlp.py`（ag_rs/gemm_ar 语义）+ `bench_tp_mlp.py` 的
decode→prefill M 扫 + `bench_allgather_gemm.py` 的 total/ag/gemm 分解。
**真 TP 布局**（B 沿 N 切分），非我们微基准的 B 复制形态。形状取自上游
LAYER_CONFIGS（无 HF 依赖）。R=8 只有 LLaMA-3.1-70B / 405B 对齐合法，
Qwen2-72B、LLaMA-7B 自动 SKIP（2I/R ragged）——R=2 时它们合法。

产出：`bench/results/s21_mlp_{mode}_{model}_M{M}_rank{r}.json`，
每格 e2e_torch / ag_torch / rs_torch / e2e_ours / ag_ours / rs_ours 六臂 + md。

口径注意（已写进脚本 docstring）：无 CUDA graph（我们的 host-recipe op 不可捕
获，两臂同等 eager + per-iter event sync）；激活两臂同用 torch silu-mul。

## 2) s20 — 8 卡自然偏斜 + 预测器 A/B（机制实验）

```bash
bash bench/s20_run.sh          # 默认: skew,nopred,pred,nopred2,s14,rccl_seq @ M8192 N29568 K8192
S20_ARMS=skew,nopred,pred,nopred2 bash bench/s20_run.sh   # 卡时紧的缩减集
```

**预注册决策门（跑前锁定，防事后归因）**：
- G1 偏斜存在：probe_spread_p95 ≥ ~1（peer 到达不同步）
- G2 燃料存在：strag_stay_frac ≫ 1/(R−1)=0.143（滞后者 round 间相关）
- G3 预测器赢：run 级 median(pred) < median(nopred)，且在 nopred/nopred2 复现
  传播之外

G1/G2 失败 → 预测器在本平台是 rig-only，干净的负结果，到此为止。
`s20_8card_skew_rank*.json` 里的 order/probes 逐 round 记录是第一手数据。

## 风险与纪律

- 一切长任务可断点续跑：两脚本都是逐臂逐格增量落盘 JSON，pod 重启丢进度可控。
- s21 gemm_ar 模式：两个 AR rd ctx 同时存活（N=2I 与 N=K）是未测布局；
  AR_CHUNK=256 把对称堆压到 ~1.1GB（S18 定律 ~1.8GB 上限）。若 wedge，
  timeout 2400 兜底，ag_rs 结果已先落盘。
- s20 定律沿用：单 ictx 复用、I1_PULL=engine、chunk_rows=256。
- 验收线（S19 遗留）：2 卡 A/B + 600 轮 soak；若卡时允许，
  `SOLO_ROUNDS=200 python3 tests/dbg_consumer_solo.py` 在 8 卡 pod 里同样可跑。

## 当天可选加餐（卡时富余才做）

- `I1_CONSUMER_GRID` 扫一遍（s19e：持久 grid128 s1=31.5 vs 满格 s2=25.9——
  若 pull/consumer 重叠充分，封顶本身可能净赢）。
- S17 遗留：n_chunks=8 free-run 15 iters wedge 过一次——600 轮 soak 前先别开。
