# G10 CPU 线程预算与实际消费

状态：**SCOPED，2026-10-17；尚未 AUDITED，产品 FIX 集未冻结，不写产品代码**。

本轮由当前 G07–G10 goal 明确授权，承接[原 G10 候选](../project/inference-single-machine-foundation-audit-2026-10.md)
和[消费适配](../project/inference-consumer-adaptation-plan-2026-10.md)。
重构 baseline 为 `0f852b231`；原 owner 为 `compute_route_cpu` 和 `compute_provider_native`。
G07/G08 已接受，G09 的产品与合格 CUDA P1 保留，三组 A/A 资格阻塞由
[独立工具卡](../tooling/inference-g09-native-workload-aa.md)负责。G10 不降低或代签 G09 的性能门。

## 真实触发与产品边界

单机 CPU 及 CPU+单 GPU 的引擎消费者需要协调 prefill/decode 与两个并发 invocation 的线程预算，
能分清 caller、借用 executor、原 ggml workers 和 Provider 私有池的控制范围。
当前 [CPU options](../../../../include/hyle/compute/cpu.h)明确 `native_threads` 仅约束 ggml，
executor concurrency 不隐式创建该池；[ATen runtime](../../../../src/compute_provider_native/aten_cpu_provider.cc)
明确 `uses_private_threads=true`。这两个现有合同并不能证明全进程单线程或总并行度上界。
本轮作为新的控制/可观测能力需求打开两 owner，不把原已声明的边界追认成旧产品缺陷。

复用原 Context/Program/Invocation、executor、Sched 与 Platform 的 affinity/NUMA 能力。
不新增引擎 scheduler、模型/request/token 身份、全局线程池框架或多 GPU/多机政策；
不以临时 process-global setter 冒充独立 context 的预算，不把未知私有线程写成 0。

## AUDITED 前的有限交付

1. 核现有 public option、runtime properties、实际原生线程路径和其进程/线程/context/invocation
   作用域；为可约束与只能报告的部分逐项给出证据，特别检查 ATen、ggml、oneDNN 及平台差异。
2. 沿已冻结的 SGLang `7224fce1`、vLLM `98f72111`、llama.cpp `cbe28bb3`，每族至多两处主要
   源机制加一个反偏置；不刷新依赖，不扩充完整引擎调研。
3. 固定一个 CPU-only 消费与一个 CPU+GPU 消费的公开调用草图，覆盖两个 phase contexts 与两个
   并发 invocation。先确认真实数学消费及预算语义，再决定是否需要极短反证；不从计时挑 API。
4. 固定 public 形状、owner/lifetime/失败/并发及常态成本账和有限 FIX；其他能力借用或明确拒绝。
   无兼容包袱，避免重复短文件；优先在现有自然 owner 中形成最小完整合同。

SCOPED 仅做以上有限审计与源事实收集。实现前必须完成 AUDITED、绑定受影响 owner、预声明风险与
selected gates；普通增量构建最多 4 jobs/4 logical、BelowNormal，所有平台重型工作串行。
仅对真实改变的 owner/直接消费者补证；原数学、包及 SDK 资格按身份复用，不做全局构建。
时间较长的同配置验证沿现有授权追加，不重置失败。新增 benchmark/仪器问题另卡，不在产品模块扩建。

本卡尚未把任何候选 API、全进程线程约束或性能结果写为完成。
