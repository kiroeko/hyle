# Compute v2：CPU 推理算子目录与多算子执行合同

## Closure card

- component / owner：`compute` / Compute；round=`compute-v2-cpu-operator-catalog`；
- baseline：`1b32fb8e792567b23d2aae15b1174ac52e0f8ffb`；previous terminal：
  [`compute-v1-clean-foundation`](compute-v1-clean-foundation.md)；
- 合法 trigger：v1 terminal 明确把完整算子目录留给后续卡；已接受的
  [`compute-design`](../../../architecture/compute-design.md)要求以 normalization → matmul → gated
  activation 的真实 CPU 片段验证 typed op/provider 合同；ggml/llama.cpp、vLLM、SGLang 与 ds4 都有
  对应稳定推理计算族；
- 正常用户任务：C++ 推理引擎准备固定 shape 的 RMSNorm、矩阵乘与 SiLU-gated multiply，随后在同一
  CPU Context/Queue 上重复提交，不在 steady path 做 shape inference、route search、packing 或 allocation；
- 正交反偏置任务：工程计算用户不构造 Graph，直接准备普通二维矩阵乘与逐元素 gated transform；公开
  类型不得泄漏 decoder/model/provider 专名；
- scope：审计并冻结 `MatMul` 与 `SiluMul`（最终名称由 AUDITED 形状决定）的 dtype、rank、layout、
  transpose、broadcast、alias、precision、failure 与 workspace 合同；实现本轮准入的 dense f32 CPU
  reference 与 HYLE-native prepared route；让现有 Operation/PreparedOperation 可自然承载多个 operation
  kind；为 Graph v2 暴露稳定描述和执行入口；
- non-goals：完整公共 op 超集、quantized matmul、attention/RoPE/embedding、模型/KV/batching/sampling、
  外部 ggml/oneDNN provider、CUDA/Metal/MLX/Vulkan/HIP、异步 queue、Native FFI；这些必须另开有限卡；
- 直接依赖：沿用 `core/error/mem` 公开边与 `math/simd/util_core` 实现边；不新增 Sched 或 Graph 反向边；
- 直接消费者：Graph v2 multi-op fragment 与独立 direct-Compute package sample；
- P0：逐 op 记录复杂度、读写量、SIMD route、packing/workspace、allocation、alias check 与调用边界；
- P1：候选后只对最终 shipping CPU route 做一次同合同测试。MatMul/SiluMul 选择至少一个当前成熟实现作
  peer，并把 dtype/layout/precision/threading/setup/completion 对齐；Graph 税另归 Graph 卡。固定门 5%，
  方向无论幅度都如实进入总账；
- 机器与验证：Windows/MSVC 为工程性能机；Windows、WSL/GCC 14、macOS/AppleClang 做 correctness/header；
  WSL ASan/UBSan/TSan 按风险；OOM、overflow、zero extent、wrong shape/layout/dtype、unsafe alias、move/close、
  reference differential、direct consumer、source/static/shared、red-team、closure/docs/diff；
- 调查预算：四个真实引擎与成熟 CPU provider 各扫描一遍；能力/API 纠正最多两轮；性能根因纠正最多
  两轮；每个正式 workload 最多一次 `n=15` qualification；
- stop：若标准 MatMul 形状必须依赖未建 provider/packing owner 才能达到可接受路线，或 tensor contract
  无法表达所需 layout/precision，则停止并形成精确依赖卡，不以 benchmark-only fast API 绕过。

## SCOPED 边界

本状态只授权能力和源码审计，尚未授权修改产品。AUDITED 前必须输出：

1. 四个真实引擎及 ggml/oneDNN/BLAS 类 peer 的调用形状和语义差异；
2. 每个现有与候选 Operation 入口的需求回映；
3. 有限 `KEEP/FIX/BORROW/DEFER/REJECT` ledger；
4. 自然 C++ 调用草图以及同合同性能 preflight 计划。
