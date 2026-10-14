# I1 前置:Syncopate 定点精读一页笔记(2026-10-13)

> 任务【I1 预测器】§1 交付物。只读检索 `related_repos/syncopate`,未构建未运行。
> 全部内容标注【代码实读】+ 文件:行号;无一处运行数据(它没在本机跑过)。
> 回答四个指定问题:① 谁写 signal_offsets/wave_offset 表、何时、布局、kernel 何时读;
> ② dl.wait+consume_token 消费点在哪;③ 计划→调度→降级哪一步产表、能否变运行期输入;
> ④ 宿主侧 _wait_eq_cuda 调用形态(要避开的)。

## 表的数据流图(host 何时算 → 怎么传 → kernel 何时读)

```
【计划期:每次运行算一次,host,纯几何】
build_all_gather_plan_1d_swizzle(shape, dtype, axis, mesh_size, rank, ...)
    │  按 mesh_size 等切 axis → DevicePlan(IR:Period/MemorySegment/BufferRegion)
    │  【代码实读 communication/common_descriptors/all_gather.py:14-52】
    ▼
CommGenerator(device_plans).plan_signals()
    │  给每个 (wave, tile, peer) 分配信号槽位(全局唯一偏移)
    ▼
lower_comm_plan_to_raw_schedules(generator)[rank]["a"] → RawSchedule
    │  num_waves + block_infos[TileBlock(tile, signal, peer_rank, dst_offsets…)]
    │  【代码实读 interface/lowering.py:1-90、interface/tile_schedule.py 全文】
    ▼
schedule.gen_block_offset_lists() / gen_block_shape_lists() / gen_signal_lists()
    │  嵌套 python list;signal=dst_signal 槽位,无信号处 -1 哨兵
    ▼
torch.tensor(list, device="cuda", dtype=torch.int32)   ← 一次成型,之后只读
    │  三张 int32 设备张量:block_offsets(每 wave 行起点)/ block_shapes(每 wave
    │  行数×列宽)/ signal_offsets(每 wave 信号槽)【代码实读 tests/test_ag_gemm_hand.py:90-97】
    ▼ ──────────── 每迭代运行期 ────────────
【host 每 iter】reset_signals()(旗标 buf fill_(0) 清零)→ nvshmem_barrier_all_on_stream
    → comm_runtime.execute() 按 op 派发:
      IntraNodeSignal  → _set_signal_cuda(buf[offset], 1, stream)   1-CTA kernel 写 1
      WaitIntraNodeSig → _wait_eq_cuda(buf[offset], 1, stream)      1-CTA kernel 自旋等 1 ←【④ 要避开的形态】
      SMCopyOp         → persistent copy kernel(num_copy_sms,green ctx 按 SM 划分)
    【代码实读 communication/comm_runtime/communication_context.py(execute/reset/160 附近)】
【device 每 iter】gemm_consumer(a, b, c, signal_ptr, cur_wave_size, wave_offset, signal_offsets, …)
    persistent 循环,每个 wave 边界 3 次标量查表 + 1 次无条件等:
      cur_offset_m   = tl.load(wave_offset) // BM          ← 查表:本 wave 行起点
      cur_tiles      = tl.load(cur_wave_size + wave*2) // BM ← 查表:本 wave tile 数
      cur_signal_pos = tl.load(signal_offsets + wave)       ← 查表:本 wave 信号槽
      dl.wait(signal_ptr + cur_signal_pos, 1, "gpu", "acquire", waitValue=1)  ← 无条件等
      dl.consume_token(a_desc, token)
      然后算该 wave 的 GEMM tiles
    【代码实读 computation/gemm/template.py:82-88 及 wrapper;codegen 镜像
     computation/transform/transformer.py:616,1089-1141(wave_offsets/cur_wave_sizes/signal_offsets 同名参数)】
```

**四问直答**:① 表由**计划期**(CommGenerator + lowering + gen_*_lists)一次写死,
布局 = 每 wave 一行的 int32 设备张量(每 rank 各一份、内容互不相同),kernel 在 **wave 边界
tl.load**(每 CTA 每 wave 3 次标量读,L2 命中,成本可忽略)。② `dl.wait+consume_token` 在
**kernel 内 wave 循环里**(template.py:82-88),每 wave 一次、无条件;宿主侧对应物
`_wait_eq_cuda` 在 execute() 里(见④)。③ 产表在**降级(lowering)步**;能变运行期输入——
kernel 侧表只是指针参数+tl.load,把"一次成型的 torch.tensor"换成可更新缓冲即可,
kernel 编译产物零改动。④ 避开形态 = `_wait_eq_cuda(signal_buf[op.offset], 1, stream)`:
1-CTA 宿主入队 kernel 原地自旋等值 1,等的人占 SM 且 execute() 串行推进;配套
`reset_signals()` 每轮 fill_(0) 清零 + barrier——零值旗标+每轮清零+栅栏三件套,
与我们的单调 ticket(永不清零)+零 SM 流等待(S12 实测)正好相反。

## 三条结论

**照抄什么**:
1. **表 = 普通 int32 设备张量作 kernel 参数 + 消费点 tl.load**——无 constant/shared 依赖,
   表小、L2 命中。与八问作答 Q1 结论同型,OSDI'26 系统量产验证过该形态。
2. **表进 kernel 只是指针,换内容不触发重编译**——预测器迭代(host 侧)与 kernel
   编译解耦,这是"表驱动"最可贵的性质。
3. **计划期/运行期分离的分层**(plan→schedule→lists→tensors):几何部分
   (哪个 wave 覆盖哪些行、信号槽位分配)可整体照搬;我们只把第 4 步
   (一次成型)换成每 forward 更新。

**避开什么**:
1. `_wait_eq_cuda` 阻塞自旋 + `reset_signals()` 清零 + 每轮 barrier 三件套
   (零值旗标语义)——替身 = 单调 ticket + hipStreamWaitValue32 GTE 零 SM 流等待【S12 实测】。
2. **每个 wave 无条件 dl.wait**(wait 侧零作为):它的表只服务"算哪块",不服务
   "按什么顺序等、等多久"——这恰是我们的增量位置(顺序表+预算表)。
3. 表内容 = 纯静态几何,运行期只读(测试配置按模型手写,如 Qwen2-72B
   BM128/BN256/BK64/Stage3,全库无 autotune)——照搬即得不到动态到达序。

**换成动态到达序要改什么**(改造量清单,逐条对到我们的 I1 实现):
1. **表生成移运行期宿主**:保留几何列(wave/tile 划分、槽位),新增动态列
   (order[R] 到达序 + budget[R] 自旋预算)——表从【几何常量】变【几何+动态】两段。
2. **传输**:静态版 torch.tensor 一次成型 → 动态版 pinned 缓冲 + 每向前
   non_blocking H2D 落进 DUSHMEM 对称堆小槽(k-chunk 槽池反正要用对称堆);
   kernel 读法不变。
3. **kernel 消费点**:把"无条件按固定 swizzle 等"换成"按 order[] 定 tile→CTA
   映射、按 budget[] 定有界自旋"。这是 kernel 内的真改动,Syncopate 无对应物可抄。
4. **旗标值语义**:{0,1}+每轮清零 → 单调 ticket(只增不清零)。
5. **观测面**(它完全没有):到达序从哪来——1-CTA observer / 宿主时间戳,新增组件。

**附带发现**:Syncopate 自己把"查表 vs 闭式"当权衡维度——`Mod1DSchedule`
(priority 1,interface/tile_schedule.py)在均匀切分时走闭式数学**避免查表**。
动态到达序没有闭式表达 → 查表是唯一形态,反向确认我们选查表形态合理。
