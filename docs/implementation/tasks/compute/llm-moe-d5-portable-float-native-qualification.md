# D5-F45：CPU / Apple 官方低精度原语资格

<!-- MODULE REVIEW: AUDITED -->

- baseline：`55211caf1954f34a9322c08872ead3e0f3efd742`；Windows canonical。
- 属于[当前 D5](llm-moe-d5-real-fragment-acceptance.md)的剩余官方画像边界调查；
  本批产品 FIX 为零，不改变任何模块 owner 状态或提前增加完整画像通过格。
- 需求：GLM 官方 block128 FP8 与 DeepSeek 官方 FP8/FP4 的动态激活量化、
  权重解码、逐角色混合精度必须保留，不能用未量化 BF16 MatMul 代签。

## 已审计来源与有限动作

1. 固定 oneDNN `71094df451909e95ceeefc8f64c275cdcbbcdff5`：CPU reorder
   显式注册 F32↔E4M3 和 F32↔E2M1。使用现有静态库直接验证有限 codes、
   中点及相邻 F32 值、饱和范围内的正负输入，记录真实 implementation 名和误差。
   `float8.cpp` 的 F32→E4M3 reference 经 F16，E8M0 只截断指数；两者均不能
   未验证就当作原版直接 RNE / 向上取整。失败只约束该具名调用，不外推全部 CPU 库。
2. 固定 MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85`：公开
   `to_fp8/from_fp8`、`quantize/dequantize(mxfp4)` 与 `gather_qmm`。
   MacBook Air 使用现有 libmlx/metallib，在一个 C++ 原生批次检查全部有限 FP8 code、
   中点邻值、group32 FP4 往返，以及 group64/128 的原版 FP8 分组序列。
   分组序列只组合成熟 MLX 原语；向上指数取整直接对应已固定 DeepSeek
   `fast_round_scale/fast_log2_ceil/fast_pow2`，不写新转换或乘法 kernel。
3. 独立测试 oracle 用有限编码查表和 nearest-even；转换 code 精确，FP8 scale
   普通模式相对误差≤2e-6、power2模式精确，BF16 roundtrip 逐位比较。
   至少覆盖零组、不同组尺度、正负数、量化中点和小幅输入；原 FP4 的最小 scale
   与 MLX 原生最小 scale 的差异必须单列，不能隐藏为普通模型 tolerance。

预算为一次原生批次与一次明确的编译/ABI 接缝纠正；不调性能、不更新依赖、
不下载权重、不建大型 build tree。复用当前静态/动态成熟库，Windows 禁止 DLL 弹窗。
原生通过后才重新绑定具体 owner 并固定批量产品 FIX；完整 Direct/Graph、资源、
源码/静态/动态交付与 P1 仍由 D5-A5/A6 签署。任何失败保存原日志与适用范围。

## F45.1–F45.5：Apple FP8 批量接线（AUDITED）

[原生批次](../../../../evidence/correctness/llm-moe-d5-real-fragments/portable-float-native/README.md)
确认FP8边界；FP4极小值和oneDNN reference失败不进入本批产品范围。
现有 `compute_route_apple` / `graph_route_apple` 继续本轮AUDITED，重新绑定本卡。
共享语义与此前D5事实不重开，公开model_ops不增加字段。

1. F45.1：Apple InputSpec接纳精确canonical block128 activation/weight及grouped
   weight字节carrier；核对完整格式、bytes、rank和MLX int容量，不接受近似格式。
2. F45.2：一个Compute-owned冷lowering供Direct/Graph共同调用，连接原版MLX
   E4M3编解码、ReduceMax、Divide和DeepSeek原版power2 bit序列。支持既有预编码
   MatMul、Float8BlockLinear、Float8WeightDecode、GroupedFloat8Linear，以及
   group64/128 FP8 roundtrip；FP4 roundtrip明确拒绝，不能误接MXFP4。
3. F45.3：block GEMM按原模型保留各K128块F32 dot→两侧scale→F32累加；
   N尾块只在scale广播后裁剪。grouped只取选中专家的encoded bytes再解码，
   A=1/S按原语义广播；安全ID驱动Take，越界复用既有device status/Completion失败。
   不为全部未选专家展开F32，也不把已量化输入二次量化。
4. F45.4：MLX持有中间F32解码/分块乘积，容量和复制成本如实记录为非零；
   constant folding属于原生编译器，HYLE不增加私有权重缓存、不保证零热分配。
   默认规则在冷期由路线、精确操作/编码/shape绑定，无热期Provider搜索或计时。
5. F45.5：两入口批量验证真实dense/tail/grouped签名、变化/重复/非法ID、
   BF16/F32与两种scale、冷权重准备、普通GLM共享FFN及完整MoE，并运行Apple
   生命周期/索引回归；已验证共同片段身份不变时复用。完整GLM消费者随后批量接合，
   只有全部region通过才增加完整格。A5三交付、A6 P1继续独立待验。

预算：一次Apple低并行构建/串行验证，具体编译或数值失败最多一轮定向纠正。
默认策略和准确性固定后再做适用P1，不把这个扩域混入旧P1结果。
没有新数学kernel、Provider工厂、runtime组件或公共治理改动。

## F45.6：失败归因，回到测试合同审计

产品纠正预算在 `b0d724acce7b74d10fc1b51d8387a0c21f8b83eb` 用尽：
Direct 20/23、Graph 13/16，两入口同三个失败；完整画像仍148/176。
先固定该产品版本，回到 AUDITED 的测试合同审计，不继续试改产品数学。
原子抵消项的原生诊断已经确认所有 FP8 codes/scales 一致，F32关联差异约3.49e-10；
不能以此替另外两项完整 GLM 链路签PASS。初次/定向纠正原始日志均保留。

有限动作：补region、replay和实际数值到失败输出；只重跑三个失败用例一次，
并补跑原filter误写为Lifecycle而漏掉的AppleLifetime/AppleRegion及Graph资源回归。
必要时比较逐片段误差及实际状态反馈，原始逐元素门保持不变。
判断一ULP是否错误绑定某一FMA顺序时，只采用独立F32误差界；完整量化链必须单独
审计，不把原子误差例外传播为全链通行证。禁止搜索更宽tolerance来得到PASS。
本小批不建新build tree、不跑P1，新增任何产品FIX前重新固定具体原因和范围。
