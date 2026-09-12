# Compute Provider Native v1：reference + HYLE SIMD 算子因子审计

## Closure card

- component / owner：`compute_provider_native` / Compute；round=`compute-provider-native-v1-factor`；
- baseline commit：`ec83514fe9d082698eb05018680cfaeda1822a5a`；
- trigger 与证据：当前 RMSNorm/MatMul/SiluAndMul 的 validation、reference route、SIMD whole-row kernel 和
  通用 provider/runtime 混在 `compute`；扩展算子目录会让 GPU-only runtime 强制取得 Platform/SIMD 和大量
  CPU kernels，也无法独立对 HYLE winner route 做资格化；
- 正常 consumer 任务：用户把 `native::provider()` 放入 CPU Context 的显式 provider sequence，冷路径
  prepare 一批 typed operations，在固定 tensor/storage 合同上重复 submit；不在热路选 ISA、推 shape、
  分配或静默 materialize；
- 主产品纵切：decoder 的 norm/linear/RoPE/attention/gated-MLP/top-k CPU fallback；正交纵切为通用
  elementwise/reduction/transform pipeline；
- scope：迁移已有 native validation/route/kernels，给 `compute_ops` 首批 `MUST` 目录提供正确 reference f32
  execution，并仅在已有 SIMD whole-row 或固定 kernel 符合同合同时接入；逐 op 登记 dtype/layout/alias/
  precision/workspace/capability，reference 与 optimized route 必须可差分；
- non-goals：拥有 Tensor/Operation schema、backend context/queue、oneDNN/ggml、GPU kernels、模型/KV/
  sampling/serving、为 benchmark 特设 API 或未经证据的微型 HYLE route；
- 直接依赖 / 消费者：候选公开依赖 Compute runtime，私有依赖 `compute_ops`、Math、Mem、Platform 与 SIMD；
  直接消费者为 direct CPU Compute、Graph fallback 和 provider qualification；
- 性能：`P0` 冻结每项复杂度、allocation/dispatch/copy/ISA 账单；只有真实高权重 op 进入 `P1`，与成熟
  peer 比较同 dtype/layout/threading/output/completion，5% 非劣门；reference 正确性 route 不冒充 SOTA；
- 验证：每个 operation 的 normal/zero/tail/shape/dtype/layout/stride/alias/overflow/special values，scalar
  oracle 与 SIMD differential，provider fallback、OOM、并发 Invocation、逐头、三平台、sanitizers、三交付、
  factor-only、P0/有限 P1、红队；
- 预算 / stop：每能力族至多两个强 peer、每个 FIX 两轮、每条 P1 一次资格化；若一个 op 需要 provider-
  specific packed artifact 或模型策略，留给对应 provider/引擎，不污染 native 通用面。

## SCOPED 待审计项

1. 当前三项 native implementation 的逐入口 `MOVE/KEEP/REPLACE/DELETE`；
2. 首批 operation family 哪些必须有 reference 才能对外宣称可执行，哪些明确推迟给 provider card；
3. generalized input/output arity 后 prepared/invocation metadata 的冷/热成本；
4. SIMD whole-row/fat-kernel 可直接复用的清单，以及没有 material 证据时保留 scalar/reference 的边界；
5. CPU product façade、可选 oneDNN 顺序与 source/static/shared 构建投影。

AUDITED 前不移动或增加 native kernel 产品代码。
