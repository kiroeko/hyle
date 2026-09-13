# SIMD v7：Compute 驱动的普通 f32 row primitive

## Closure card

- component / owner：`simd` / SIMD；round=`simd-v7-compute-row-primitives`；
- baseline commit：`df54eef5352019e25ea1694fc5fec17583ae9d87`；上一 terminal candidate 为
  `e72089c196f6b38f897524087bc53201a0fd8df4`；
- trigger 与证据：Compute Native 的 18/19 executable checkpoint 已形成真实推理 consumer：
  [`elementwise.cc`](../../../../src/compute_provider_native/elementwise.cc) 对 dense f32 `Rsqrt`、`Minimum`、
  `Maximum` 和 `Select` 仍逐元素 scalar；[`reduction.cc`](../../../../src/compute_provider_native/reduction.cc)
  不能复用现有 extrema，因为当前 `simd::min_value/max_value` 对 `+0/-0` 不满足 Compute 已冻结的
  NaN propagation + signed-zero 合同。以上是缺失的通用连续 row 机制，不是 tensor/layout 或 benchmark
  名称；
- 正常 consumer 任务：对已验证的连续 f32 rows 执行 reciprocal-sqrt、IEEE `fmin/fmax` 风格逐 lane
  选择、byte-bool 条件选择和确定 extrema reduction；普通调用一次 pointer/count entry，fixed/fat
  Compute kernel 可在 target TU 内直接组合；
- 主产品纵切 / 具名来源：量化 decoder 的 elementwise、normalization/attention 前后处理；直接 consumer
  是 `compute_provider_native`；
- 正交反偏置纵切：批处理数值管线对连续 f32 arrays 执行相同四类操作，证明 API 不携带 Tensor、axis、
  broadcast、模型或 Graph policy；
- 跨模块 public journey 与每段 owner：SIMD 拥有连续 row 数值语义与 ISA route；Compute 拥有 dtype/layout/
  broadcast/alias validation、row partition 和 provider execution；Platform 继续提供 CPU facts；
- scope：只审计并交付 f32 reciprocal-sqrt、f32 pairwise `fmin/fmax`、byte-bool f32 select、以及现有 f32
  extrema reduction 的 signed-zero 合同；迁移 Compute dense row route 并保留 scalar strided/broadcast
  fallback；
- non-goals：重跑 v6 的 422-op 迁移、增加 f64/narrow/tensor broadcast API、模型 fused op、线程池、改变
  runtime decision authority、按微基准建立 HYLE/xsimd 例外，或修改未受影响 SIMD family；
- 直接依赖 / 直接消费者：依赖不变，仍为 Core + Platform 与锁定 Highway/xsimd build-only mechanism；
  直接消费者为 Compute Native，SIMD 不反向依赖 Compute；
- 性能 tier 与原因：`P1`。P0 先证明 O(n)/O(1)、零 allocation/lock、一次 ordinary dispatch 和 target body
  无逐元素分派；P1 只运行一个 Compute-real dense-row group，与同语义 scalar oracle及 direct locked-
  Highway body核对，固定 5% material non-inferiority margin；
- 适用机器 / 仪器 / material margin：W1-WIN 做普通/fixed codegen、correctness 与有界 P1；Linux 与两台
  Apple 做 target correctness/交付；5%；
- 调查与纠正预算：Highway/xsimd 各一次源码合同审计；公开 API/语义一轮；实现至多两轮；P1 一次完整
  qualification，若 preflight 不合格不靠重复采样找结论；
- 预声明验证：zero/null、tails、exact alias、NaN/inf/subnormal、`+0/-0`、非 0 bool bytes、scalar/
  selected differential、Compute dense/fallback differential、逐头、三平台、sanitizers、source/static/shared、
  selected target/codegen、文档、红队与有界 P1；
- stop / blocker 条件：若 Highway/xsimd 不能在不增加第二次 dispatch/临时 buffer 的情况下表达精确语义，
  保留最小 HYLE semantic adapter；若 API 必须吸收 tensor/broadcast policy，停止并把能力留在 Compute；
  material P1 失败最多纠正两轮，之后如实 `BLOCKED`。

## SCOPED 待审计项

1. public spelling、alias 与 special-value 语义，尤其 pairwise `fmin/fmax` 和 propagating reduction 不得混淆；
2. Highway/xsimd 对 rsqrt、NaN、signed zero、mask load/select 的精确合同与 target coverage；
3. existing Runtime/ops/search table 的最小增量，保证普通路径仍只有一次缓存间接调用；
4. Compute 能按完整 dense suffix 调用一次 row entry，strided/broadcast fallback 不被错误宣称加速；
5. 受影响的 source/static/shared export、用户指南与 v6 provider-route 账本增量。

AUDITED 前不修改 SIMD 产品代码。只允许读取 provider 实现、形成 capability map 和冻结有限 FIX。
