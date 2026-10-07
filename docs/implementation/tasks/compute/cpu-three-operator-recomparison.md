# CPU SDPA、Copy、RMSNorm 当前成熟实现复比

## 冻结范围

- baseline：`d283aae73b5547fdd2833ef1bf43ab3d00c729ec`，Windows 工作树干净。
- 用户触发：重新比较三个保留的自研算子；CPU-only ggml 已升级，旧 peer 的性能资格不能用于新版。
- 本轮是有界性能证据任务，owner 为 `compute_provider_native` 的 benchmark；产品 owner 状态和源码不变。
  测试/证据维护不伪造产品 reopen；若结果需要改产品，再独立绑定有限删除/路由 FIX。
- 主消费：Qwen3.5-2B hidden=2048 的 token/small-batch norm；正交消费为 plain tensor materialization。
  SDPA 同头子域是已准入 attention 能力的消费形状，不能冒称 Qwen 的完整 GQA/MRoPE 程序。
- mature：当前 CPU-only ggml `ffa4e8b8`（0.26）、oneDNN 3.13.2、LibTorch 2.8.0-cpu 的已安装静态 SDK，
  使用普通 CPU 冷绑定，记录每格实际 Provider。Native 用现有 expert factory，不改数学。

## AUDITED：同合同与成本

共同输入/输出、相同 dtype/epsilon/accumulation/mask/布局；禁止 alias；一个 caller、每库一线程。
普通无限预算请求，不要求某侧额外兑现严格预算或整个动态形状域；这两项只做能力对照。
准备、持久状态和 invocation 建立在 steady-state 外；每次 submit、检查、映射、转换、packing、
原库热分配、输出物化、真正 Completion wait 均计费；准备费用不由这批 steady-state 证明，不能假签端到端模型收益。

源码锚点：Native `native_provider.cc::build_plan/submit`、`attention.cc::prepare_attention`，
`transform.cc::submit_copy`、`normalization.cc`；成熟 `ggml_cpu_provider.cc::lower`
的 RMS/Copy/SDPA 分支及 `submit`，oneDNN `normalization_internal.cc::prepare_normalization`、
`onednn_provider.cc::submit`；低/宽 storage 的 ATen `aten_cpu_provider.cc::submit`。
两侧同步返回时已经写完输出并关闭映射；wait 无 deferred 数学。Native SDPA 是 tiled score/dot/value，
成熟 F32 SDPA 是原 ggml matmul/softmax/cont region；额外物化必须计费。RMS 已有 SIMD row 与 mature norm；
Copy 按实际 strides materialize，不能用 View 代替。Native workspace 由 caller 持有，额外 private=0，
仍有描述/状态开销；成熟 hot private 未知不是 math 唯一性。dense Copy 实际委托 Mem range-copy，
并无另一套自研 dense 复制算法；strided Copy 才走 Native 逐元素 materialization。

## 有限 FIX 与预算

1. 复用旧已审 benchmark 和公共统计，仅缩到三项：旧 18 格（Copy 五 floating storage、RMS 三 dtype、
   F32 SDPA，各 rows=1/32），加六个预先固定 holdout：F32 strided Copy、BF16 RMS epsilon=1e-6、
   F32 causal SDPA depth=256/head=8，各 rows=1/32；共 24 格。
2. 冻结 source/SDK/依赖身份，批量数值与当前 workload-matched A/A，覆盖两执行顺序与两分配顺序。
   复用原 5% factor-median / 10% gross 门；失败格保留原始记录，A/B 前明确 INCONCLUSIVE，不追加姿势。
3. 一次统一 P1：W1-WIN、单 P 核、above-normal、static、Release、LTO off；每 stratum 15 个独立进程，
   四 strata、5% practical 门，禁止 n31 升级或自研优化。通用 controls 与 workload A/A 都须通过。
4. 成熟整域/严格预算与 Native 的区别沿实际源码/已有资源资格复核；不把接缝未接自动写成库不支持。
5. 汇总合格 direction/CI 和去留建议；保留源码与是否优选分别写明。统一 LOC、文档检查、[skip ci] 提交推送，
   清理本轮 consumer 临时产物。

适用验证：24 格数值比较、SDK 及源码身份、fresh A/A、公共 P1 controls/manifest verify；
`validation-select.py plan`、closure/docs、diff check。产品无变化，旧 correctness/三交付资格复用，
不为本轮 benchmark 重建产品或重跑 GPU/WSL/全部算子；安装静态 consumer 实际重编。

停止：一批 preflight 与一次 P1；资格失败如实报告具体格和原因，预算用尽终止采样；
不把未知签成胜出，不把性能 LOSS 签成产品 correctness FAIL，不写新 SOTA 或跨平台结论。

## 实际前置结果

24/24 数值对照通过；两真实路线各四种执行/分配顺序的 fresh A/A 全部满足既有资格门。
完整原始前置与 loaded `.text`/DLL/SDK 身份固定在 benchmark 的 `preflight.json`。
Copy 的实际成熟 peer 是 oneDNN；F32 RMSNorm 是 oneDNN，F16/BF16 RMSNorm 与四个 SDPA 格是新版 ggml。
前置不是 A/B 性能结果；唯一完整 P1 尚待运行。无产品修改、无新优选规则。
