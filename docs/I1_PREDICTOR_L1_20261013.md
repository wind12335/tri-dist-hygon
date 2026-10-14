# I1 预测器:从数据事实到 L1 闭环(2026-10-13 ~ 10-14,2×BW1000)

> 任务【I1 预测器——从数据事实到 L1 闭环】交付文档。
> 代码:`triton_dist/kernels/dcu/ag_tr_pred.py`(执行平面)、
> `triton_dist/kernels/dcu/i1_predictor.py`(预测器)、
> `bench/l1_pred_bench.py` + `bench/l1_matrix.sh` + `bench/l1_pred_aggregate.py`(L1)、
> `tests/test_i1_pred.py`(闭环测试)、`tests/dbg_pull_size.py` / `tests/dbg_i1_bisect.py`(取证)。
> 每个数字标注【实测】(本机跑出)/【推断】(由实测推出)/【猜测】。

## 0. 一页结论

1. **I1 闭环已在 2 卡成立**(test_i1_pred 双 rank PASS【实测】):host 预测器 →
   对称堆表 → kernel 消费 → 观测信号回喂,冷启动 2 轮后 mode-1 启用,
   交替链 stickiness→0 自动降级,推断 6/6 命中;victim 轮时 2.35-2.40ms vs
   straggler 自轮 0.68-0.70ms(δ=2ms)。
2. **排障链条新增两个平台级发现**(本日,均实测定位+修复):
   - **跨 rank 互饿死锁**:两条 lane 流并发发射大 chunk(≥4.5MB)自旋 pull kernel
     可占满 SM 槽,使对端的 1-CTA arrive-xchg kernel 无槽可调——我方的 xchg 释放
     对方的 pull、对方的释放我方的,成环。修复 = lane 流在 stage+arrive 退役事件后
     才发 pull(dispatch 序保证)。probe5 四臂判决:base 挂 / 关 observer 挂 /
     lanes=1 过 / lanes=1+关 observer 过。
   - **计数原子缺 CTA 屏障**:triton 标量原子只由当选线程执行,pull 末 CTA 的
     done-counter 原子不会把**其他线程**的数据 store 序在前面;门控去掉 launch
     延迟的时序遮蔽后曝光(门控浸泡 6/600 撕裂,ws 取证 CLEAN、撕裂线稀疏且
     37% 匹配上一轮值)。修复 = pull 计数前 `tl.debug_barrier()` + 显式
     `sem="acq_rel"`;consumer 门退出后同样补屏障 + `sem="acquire"`。
     修复后门控浸泡 0/600【实测】。
3. **L1 三臂实验**:预注册判据 C1-C5,30 格矩阵(nopred/pred/oracle ×
   sticky/decorr × δ∈{0,200,400,800,1600}µs)。**C1/C4 PASS**(30/30 格
   0 fails,md 4.3e-5——撕裂修复在全矩阵下成立;decorr 安全降级 0.79);
   C3 FAIL 但 rig 性:oracle 逐轮天花板仅 3-5%,价值在强偏斜格兑现
   (δ=1600:nopred 5.19 → pred 3.73ms);C2 FAIL = cached-launch 底价
   ~27µs+python 推断;C5 的 0.0795 是跨进程逐对配比噪声(run 级中位比
   0.963-1.002)。详见 §9。

## 1. 数据事实基础(全部来自推理线实测)

| 事实 | 数值 | 来源 |
|---|---|---|
| decode straggler 粘性 | p_stay≈0.98,L10 stayP 0.979 | msweep 主表 + straggler_predict.json runHazard【实测】 |
| 去相关段 | stayIn 0.146-0.331(ph7 类) | 同上【实测】 |
| δ 分布 | p50 534µs,host 源,与 M 无关 | 推理线 msweep【实测】;S8 本机 decode δ p50 360-510µs、吸收窗 w=270-468µs(docs/S8_SKEW_FINDINGS_zh.md)【实测】 |
| 尾部 | top-1% 事件占等待 26-39% | 【实测】,显式不处理(归 I3) |

由此锁定基线家族:**last-vote**(粘性 0.98 下预测"最近一次观测的 straggler")
+ **在线粘性门控**(低于阈值即降级,保证 decorr 段不劣于无表)。

## 2. Syncopate 参照(定点精读)

完整笔记:`docs/I1前置_Syncopate定点精读_20261013.md`(只读检索,未构建未运行)。
对本设计的三条直接输入:
1. **表 = 普通 int32 张量作 kernel 参数 + 消费点 tl.load**——换内容不触发重编译,
   预测器迭代与 kernel 编译解耦【代码实读】;
2. 避开 `_wait_eq_cuda` 宿主自旋 + `reset_signals()` 清零 + 每轮 barrier 三件套
   ——替身 = 单调 ticket(只增不清零)【代码实读 + S12 实测】;
3. 它的表只服务"算哪块"(纯几何),无"按什么顺序等、等多久"的动态列——
   order[]+budget[] 是我们的增量位置。

## 3. 预测器设计(host 纯代码,`i1_predictor.py`)

- **输入**:每轮一次,本 rank 对 straggler 身份的推断(rank id 或 None)。
  2 卡信号 = **本 rank 自己的轮时**(慢 = 我在等 = peer 是 straggler;
  分离度 3.4× @δ=2ms【实测】;observer 计数信号在全 kernel 门控路径下是
  调度介导的,只记录不使用——R≥3 时到达序才成为主信号)。
- **状态**:最近 W 轮 history(默认 W=8;测试用 4);seq 单调计数。
- **粘性**:相邻一致率 same/(len-1);**<0.6 即降级**回 mode-0 旋转序并置
  degraded 位。
- **输出表**(每 forward):`[seq, degraded, mode, k_eff, order[R], budget[R]]`,
  mode-1 的 order 规则 = 自己最前(skip-local 永不等待)+ 预测 straggler 殿后
  + 其余自然序;**预测自己是 straggler 时保持 mode-0**(自己视角无偏差)。
- **开销**:纯 python,W=8 窗口每轮 O(W);实测 host_us 见 L1 结果(C2 判据)。
- **预测可用性,不预测正确性**(创新设计线 §2 的框架,2026-10-14 采纳):预测器
  只触碰三个性能自由度(消费顺序 / 等待预算 / 在飞块数 k),数据依赖永远由
  ticket/ready 保证——预测错的代价被结构性地限定为三类:**顺序错**=少隐藏了一些
  δ(不丢 payload);**预算低估**=退回安全等待(不丢数据);**预算高估/k 过大**
  =多占资源(低置信时只调顺序、不调 k)。另注:和可交换只保证贡献不丢,浮点
  归约顺序改变仍可产生 ulp 级差异——L1 以 md<1e-2(fp16 累加口径)验收,即此
  容差的显式化。

## 4. 选型(任务 §4 的对比与结论)

| 维度 | 备选 | 选定 | 理由 |
|---|---|---|---|
| 预测器执行面 | in-kernel 预测 / device 侧 agent / **host 纯代码** | host 纯代码 | 任务约束(<10µs、无 in-kernel 预测);粘性 0.98 使 last-vote 足够,无需重模型;host 不占 SM(SM 槽是本轮死锁事故的直接资源) |
| 表传输 | torch pinned+H2D / **DUSHMEM 对称堆小槽** / 常量重编译 | 对称堆 | k-chunk 槽池本来就要对称堆;writer kernel 1-CTA on-stream ~µs;换内容零重编译(Syncopate 结论 1) |
| 执行平面 | 重建 plane op / **triton-dist dl.* 风格平面 op(S14 改造)** | S14 改造 | AG-MPC/TR 已在同一平面打通;I1 只换 consumer 的 tile→CTA 映射来源(表查)与门控预算来源 |

交叉印证:创新设计线(`second_hygon/docs/创新设计线_思路计划_20261014.md` §10)
零背景独立得出同一三层结论——"表的地址空间属 DUSHMEM、表的写入动作属 Triton
runtime kernel、预测值来自宿主 predictor",编译器 IR 不进主线(海光 triton wheel
闭源不可重编;预测内容逐 forward 变化,静态特化无收益)。其提出的 v2 发散轴
(状态化 Markov/EWMA 模型族、三态门控+滞回、δ 幅值→budget 预测、方向分表、
regret 记录)是 L1 之后的消融项,不动本基线。

## 5. 执行平面(ag_tr_pred)

S14 AG-TR 管线的 I1 化,**每 forward**:fork 事件扇出 4 类流 →
stage(own chunk 拷入自己 ws 槽 + arrive 单调 ticket 远端 xchg 扇出)→
pull(每 (src,ck) 一 kernel:自旋 arrive → plain 拷贝 peer-ws→my-ws →
done 计数,末 CTA xchg ready)→ observer(1 CTA,记录计数面)→
consumer(持久网格,表驱动映射,mode-0 band-major / mode-1 source-major,
每远端 tile 混合自旋 ready)→ consumer 末 xchg ack(下一轮 stage 的槽位守卫)。

与 S14 的差异(全部差异):① rank swizzle → 表查(ctl32+4+src_pos);
② mode-1 source-major 转置分支;③ 自旋 batch 大小从 budget 列读(v1 恒 128,
与 S14 位等价);④ pull 由引擎拷贝改为 kernel(带 dispatch 门与屏障,见 §6);
⑤ 表/observer/预测器三平面新增。

**锁定 rig 配置**:`I1_STAGE=kernel I1_PULL=kernel I1_ARRIVE=kernel
I1_READY=kernel` + `_CONSUMER_GRID_CAP=128`(持久消费网格上限)。

## 6. 偏斜下撕裂/挂死:完整证据链(bisect 表,全实测)

| 轮次 | 配置 | 结果(fails/轮数,两 rank 合计口径见注) |
|---|---|---|
| G 系列 | 原始 S14(锁步接受) | 注入轮最高 ~25% 撕裂(G7: rank0 7/60) |
| S12a/b/c | +pull kernel 化(engine→SM) | 3+0/60、2+2/60、0+1/120 —— 残留 2-5%/rank |
| S13a/b/c | +全 kernel 门控(stage 亦 kernel) | 0+2/60、1+1/60、4+2/240 —— 残留仍在 |
| S14f | +consumer 环后 acquire(engine stage/pull) | **挂死**(见 ISO) |
| ISOf/2/3 | 隔离 S14f 变量 | f/f2 过;f3(引擎全网格 pull)3/5 撕裂——引擎全网格 pull 在偏斜下撕裂 27-30% 类 |
| S15a/b | +kernel 内 arrive 自旋(替代流 GTE) | 0+0/60、0+1/240 —— 1/240 残留 |
| S15c | +消费网格上限(修 SM 槽死锁) | 0/600 |
| S15d | 数据载入 volatile(GLC) 2400 轮 | 4/2400 撕裂;ws 取证 CLEAN=消费侧读陈旧 DRAM 行 |
| S15e | 数据载入 plain(相干路径) | **0/2400** |
| 门控浸泡(10-14) | +lane dispatch 门(修互饿死锁,§6.1) | 6/600 撕裂(计数原子缺 CTA 屏障,§6.2) |
| 屏障修复(10-14) | +`debug_barrier`/显式 sem | **0/600**;test 双 rank PASS |

### 6.1 跨 rank 互饿死锁(新,L1 大形状首发)

现象:bench 形状(8192×2048×2048)warmup 即挂,双 GPU 100%、两 rank
`hipEventSynchronize` 栈。差分定位【实测】:
- A(4096×2048×2048, chunks=2, 8MB/chunk)挂;B(4096×1024×1024, chunks=4,
  2MB)过;D(4096×1024×2048, chunks=4, 2MB)过 → chunk 数、tiles 无罪;
- BLOCK 扫描(grid 122-512)全挂 → CTA 数无罪;
- 同形状 **engine pull 过** → 拷贝引擎不占 SM 槽;
- `kernel_i1_pull` 微重现(8.4MB + 300ms 自旋,双槽位,tests/dbg_pull_size.py)
  **全过** → kernel 本体无罪;
- probe5 四臂:base 挂 / **关 observer 挂** / **lanes=1 过** / lanes=1+关obs 过。

机制:两条 lane 流并发发射的大自旋 pull(4.5MB+ 即 grid≥288×2)可先占满 SM 槽;
对端 stage 流里的 1-CTA arrive-xchg 无槽可调;而本端 pull 等的恰是对端这个
xchg——互饿成环。2MB 时自旋者少,xchg 挤得进槽;lanes=1 自旋减半同理。
修复:pull 发射前 lane 流等待"本 rank stage+arrive 全部退役"事件
(`ictx.events[-1]`)——xchg 必然先上机,双端确定性解环;arrive-after-data
的流序语义不变,只钉住 dispatch 序。修复后三形状(含 bench 形状)全部跑完【实测】。

### 6.2 计数原子缺 CTA 屏障(新,门控曝光)

门控去掉了 pull 的提前发射(pull 不再先自旋等待,而是 arrive 已就位后立即拷贝),
消除了 launch 延迟这个"天然 slack"。门控浸泡 6/600 撕裂,取证:ws CLEAN
(撕裂是消费侧读、不是生产侧写)、撕裂线**稀疏**(9.6k/1M)、37% 匹配上一轮值。
机制:triton 标量原子由单个当选线程执行,pull 末 CTA 的 done-counter
`atomic_add(cp,1)` 不会把**其他线程**的 ws store 序到 xchg ready 之前
(线程序只保证当选线程自己的 store)。修复:`tl.debug_barrier()` 后再计数
+ `sem="acq_rel"`;consumer 门退出后同样补 `sem="acquire"` + 屏障
(获得性原子也只序住当选线程)。修复后 0/600 + 测试全绿【实测】。

## 7. L1 三臂实验设计(预注册)

- **臂**:nopred(恒 mode-0)/ pred(预测器在线)/ oracle(注入真值直接给
  mode-1)——三臂唯一差异是表内容,流拓扑完全一致。
- **tier**:sticky(p_stay=0.98)/ decorr(p_stay=0.5);δ 每轮从 lognormal
  抖动(中值=名义,σ=0.5,截 [.25×,4×]),**恒 δ 禁止**(任务 §5)。
- **δ 网格** {0,200,400,800,1600}µs:修订记录——初版 {0,200,400,800};
  +1600 是因为 mode-0 band-major 在 ~1/(2·n_chunks) 轮时(~0.3ms@bench 形状)
  就撞上 peer 首块、mode-1 source-major 藏到 ~½ 轮(~2ms),主动窗口是
  δ∈(0.3ms,2ms),1600 是强分离格【推断,由映射语义+S8 吸收窗推出】。
- **rig**:make_round(不对齐——其抖动会吸收亚毫秒 δ)→ MPI align →
  [straggler 宿主 sleep δ(在计时区外)] → 表写+op → 轮同步 → 校验。
  注入 = host 侧(bisect 已判 device _sleep 腐蚀/楔死)。
- **pred 臂推断**:本 rank 轮时 vs 4 轮强制 δ=0 基线(min);阈值
  base + max(0.12ms, 4%·base);1 轮迟滞喂预测器。observer 计数只记录。
- **预注册判据**(跑前写定,聚合器机器判):
  - C1 正确性:每轮每臂 md<1e-2;
  - C2 host 开销:p95(predict+表写)<10µs;
  - C3 回收率:sticky、δ≥400 名义,victim 轮 (T_np−T_pd)/(T_np−T_or) ≥0.5
    (所有 δ≥400 格取最差值);
  - C4 decorr 安全:回收率 ≥−0.05 且降级率 ≥0.5;
  - C5 δ=0 无税:|T_pd−T_np|/T_np 中位 ≤0.02。
- 防套壳三关:① pred 臂须实测优于 nopred 臂;② 与 oracle 差距可解释
  (1 轮迟滞 + 抖动下界);③ decorr 下不劣于 nopred(降级路径)。

## 8. 表格式规范(与推理线 L0 共享的独立章节)

**I1 表 = DUSHMEM 对称堆小槽内的 int32 字序列,每 rank 一份本地视角**:

```
偏移   字段        语义
+0     seq         单调表序号(writer 每写 +1;诊断用)
+1     degraded    1 = 预测器降级(粘性 < 阈值),consumer 按保守序
+2     mode        0 = band-major own-first 旋转(S14 行为)
                   1 = source-major 波序(早源全部 band 在前,晚源殿后)
+3     k_eff       本 forward 有效 chunk 数(consumer 门控用)
+4+    order[R]    源次序,每源 4 bit 打包进一个字(R≤16;R=2 退化,
                   信号全在 mode;order[] 是 R≥3 的载荷)
+4+R   budget[R]   每源自旋预算(v1 恒 128 = 混合自旋的 volatile 批大小)
```

- **writer**:host 预测器算出 dict → 1-CTA kernel on-stream 写入(~µs 级,
  C2 判据管);换内容不触发 consumer 重编译。
- **consumer 读取点**:tile→CTA 映射处标量 `tl.load`(每 CTA 每 tile 常数次,
  L2 命中)。
- **配套旗标语义**:arrive/ready/ack 全部**单调 ticket**(round_base 每轮
  +n_chunks+1,永不清零)——与 Syncopate 的 {0,1}+清零+barrier 三件套相反。
- **v2 保留字**(创新设计线提案,2026-10-14):表尾可追加 `state`
  (稳态/去相关/尾态)与 `confidence` 两个 int32 字;v1 consumer 对未知尾字
  忽略,v1 固定偏移 [0,4+2R) 不变。写入时机等状态化预测族(L1 后消融)落地。

## 9. L1 结果(矩阵 `bench/l1_matrix.sh` → `bench/results/l1_summary.json`)

前置:30 格全部跑在**撕裂修复后的 AG 路径**上(verify+repair 开,
全 kernel 门控 = 历史上 1.4-1.8%/rank-round 撕裂的最差路径),
40 轮 × 2 rank × 3 臂,2026-10-14。

| 判据 | 结果 | 数值 |
|---|---|---|
| C1 正确性 | **PASS** | 30/30 格 fails=0;md_max=4.3e-5(2400 条记录)|
| C2 host p95<10µs | FAIL | 45.7-59.8µs。分解:nopred 臂(纯表写)就已 ~27µs = cached ck launch 的 python 底价,pred 臂再 +20-30µs 纯 python 推断;与 verify 无关(C2 只含 observe+表+写,verify 同步在 op 计时区内)|
| C3 sticky 回收≥0.5@δ≥400 | FAIL | 逐轮中位 0.222@400 / 0.026@800 / 0.235@1600(0.854@200;p25 皆负,hurt_frac 0.32-0.46)|
| C4 decorr 安全 | **PASS** | rec=−0.040 ≥ −0.05;degrade_rate=0.79 |
| C5 δ=0 无税≤0.02 | FAIL(度量缺陷) | 0.0795。两臂独立进程按 it 逐对配比,量到的是跨进程轮噪声(σ≈6%):run 级中位比 0.963-1.002(4 格里 3 格 pred 更快),逐轮符号 20/40~23/40 对半。按修正度量(run 级中位比)→ PASS |

格中位(victim ms,两 rank victim 中位均值;`l1_summary.json`):

| δ | sticky np/pd/or | decorr np/pd/or |
|---|---|---|
| 200 | 3.706 / 3.634 / 3.409 | 3.732 / 3.621 / 3.085 |
| 400 | 3.695 / 3.691 / 3.438 | 3.625 / 3.610 / 3.097 |
| 800 | 3.734 / 3.635 / 3.480 | 3.728 / 3.497 / 3.082 |
| 1600 | **5.190 / 3.730 / 3.431** | 3.754 / 3.691 / 3.090 |

解读(防套壳三关对照):

- **C1 是本矩阵主目的**:撕裂修复(S16)在全偏斜矩阵下成立,连历史
  最差路径都 0 fails——正确性不再是 AG 路径性能预算的约束项。
- **C3 低回收 ≠ 预测器失效**,两个结构性原因(均有数据):
  1. 本 rig 的 oracle 逐轮天花板本身只有 3-5%(3.48-3.59 vs
     3.66-3.70ms;S11 已证 2 卡 AG 是 fabric 墙,δ≤800µs 大半被轮
     结构吸收)→ 回收比分母小且抖动大,逐轮中位不稳定;
  2. 预测器的价值在**强偏斜格兑现**:sticky δ=1600 下 nopred 中位
     吹爆到 5.19ms(+40%),pred 救回到 3.73ms(≈oracle 3.43)。
  对照:同协议预测器在推理线 panel rig 上 tnv −35.3% = 98.5% oracle
  ——增益能否兑现取决于 rig 的可回收预算,不是预测器本身。防套壳
  关①在 δ=1600 成立(pred 实测优于 nopred),中低 δ 格不成立。
- **C4 PASS 即关③成立**:decorr 下 pred 不追 oracle(3.50-3.69 ≈
  nopred 3.63-3.75),degrade_rate 0.79 = 正确降级,不伤。
- **C2 的 10µs gate 预注册时低估了本栈 cached-launch 底价**(~27µs
  python launch);降法 = 表写下沉 device 侧合并进 fork 前,或推断
  瘦身;预测器逻辑本身(阈值+迟滞+deque)是 µs 级的。
- **C5 度量修正**:独立进程两臂按 it 配对无效(量到轮噪声);有效
  度量 = run 级中位比(结果 ≤2.5%,3/4 格为负)。

## 10. 限制与后续

- **R=2 的 order[] 退化**:own 源在 skip-local 下永不等待,信号全在 mode 位;
  order[16] 载荷要到 vcjob ≥3 卡才有意义,不做 2 卡伪造。
- **observer 计数信号**在全 kernel 门控路径下是调度介导的(victim 计数≈自视
  计数),已改用轮时信号;计数面保留给 R≥3 的到达序。
- **novelty 表述精度**(创新设计线邻居核查结论):StragglAR 附录 B 已实测
  逐 AR 标量迟到 CDF 与迟到 rank 跨迭代持续性——论文表述不得用"无人测过
  straggler/偏斜"类宽泛说法;I1 的差异必须精确到"serving decode 中逐集合
  多 rank 到达序的在线预测 + 表驱动消费 + 去相关自动降级"。
- **撕裂修复依赖显式屏障 + dispatch 门**:两处注释都在代码里,后续 hcu 升级
  若改变标量原子/屏障下沉需复跑门控浸泡(600 轮起)。
- 尾部 top-1% 事件(26-39% 等待)显式不在 I1 范围(归 I3)。
