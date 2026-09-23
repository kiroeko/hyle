# M2-B2 成熟 Provider 线性、量化线性与注意力终局

## 裁决

2026-09-24，M2-B2 在产品候选 `20763362ec063b9be38a08935f184595d82813e5` 上以
`PASS_WITH_RECORDED_RESIDUALS` 结束。已准入的 oneDNN、ggml CPU/CUDA/Vulkan、cuBLASLt、
Marlin、FlashInfer、MLX 与 IREE 能力已进入普通 Direct/Graph 路径；没有新增 HYLE 数学 kernel、
shader 或 SIMD 算子。准备阶段固定 Provider 与 native program，replay 不搜索、不计时、不重新裁决。

这里关闭的是 B2 需求批次。持续承载 M2 的组件 owner 保持 `AUDITED`，供 B3--B6 复用同一构建、
资源和交付接缝；下一批开始前仍须固定该批有限 FIX。精确格式或完整模型片段没有成熟承载的格保持
`UNPROVEN`，不能因为相近位宽或原子组成存在就写成覆盖。

## 已接成熟路径

| 路线 | 已绑定能力 | 原生执行和普通入口 |
|---|---|---|
| CPU | oneDNN F32 MatMul/FFN；ggml half/BF16 batch MatMul、GQA、RoPE、Hadamard | CPU program/Graph region 持有冷准备制品；Direct 与 Graph 共用 `oneDNN → ggml CPU → native` 固定顺序 |
| CUDA | cuBLASLt rank-2、Marlin Q4-S128；ggml CUDA batch MatMul、GQA、RoPE、Hadamard；既有 FlashInfer attention | 原生 CUDA buffer/stream/completion；Graph 最大 ggml region，重复执行不重建图 |
| Apple | MLX half/batch MatMul、Q4-A64、SDPA GQA、RoPE 原语组合、Hadamard | MLX compiled region 与 array/stream/completion；Graph 两节点可保留在一个 region |
| Vulkan | ggml Vulkan dense MatMul、GQA、RoPE、Hadamard；IREE 保留已资格化标准补位 | managed ggml graph、native queue/fence；Windows RTX 4070 与 Air MoltenVK 都实跑，无 CPU payload fallback |

## 13 行 × 四路线裁决

| 需求 | CPU | CUDA | Apple | Vulkan |
|---|---|---|---|---|
| F05.1 dense linear | BOUND | BOUND | BOUND | BOUND |
| F05.2 batch/broadcast GEMM | BOUND | BOUND | BOUND | BOUND |
| F05.3 transpose/stride | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| F05.4 完整 FFN region | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| F06.1 Q4-S128 | UNPROVEN | BOUND（Marlin） | UNPROVEN | UNPROVEN |
| F06.2 Q4-A64 | UNPROVEN | UNPROVEN | BOUND（MLX） | UNPROVEN |
| F06.3 Q8-B32 W8A16 | UNPROVEN | UNPROVEN | UNPROVEN | UNPROVEN |
| F07.1 causal GQA/MQA | BOUND | BOUND | BOUND | BOUND |
| F07.2 KV append/read | BOUND（复用 B1 state ports） | BOUND（复用 B1 state ports） | BOUND（复用 B1 state ports） | BOUND（复用 B1 state ports） |
| F07.3 gated attention 完整片段 | PARTIAL | PARTIAL | PARTIAL | PARTIAL |
| F08.1 partial/interleaved RoPE | BOUND | BOUND | BOUND | BOUND |
| F08.2 YaRN/NoPE | BOUND（调用者提供冻结 frequency；NoPE 为图省略） | BOUND | BOUND | BOUND |
| F08.3 normalized Hadamard | BOUND | BOUND | BOUND | BOUND（64/128/256/512） |

`PARTIAL` 只表示已经接通的成熟域可使用：F05.3 尚未证明所有真实 checkpoint stride 都无需整权重
repack，F05.4/F07.3 尚未把目标模型完整投影、norm、gate、residual 全部作为一个已验 region 收口。
这些格在 M3 汇总前保持 residual，B2 不用冻结自实现补齐。

## 聚合验证

| 风险 | 结果 |
|---|---|
| Windows 产品真相 | PASS：候选 selector 的 CUDA/Vulkan/Graph/provider/header 门与 Compute/Graph 三交付全部通过；CPU 候选的 Direct/Graph、两次 replay、错误域与旧结果门此前在同一 B2 卡通过 |
| Apple 原生 | PASS：MacBook Air shared 的 Apple Compute/Graph 4/4；Mac mini static 的同组 4/4；均从精确候选构建 |
| MoltenVK | PASS：MacBook Air M4、MoltenVK 1.4.1 的 Vulkan Compute/Graph/ggml/IREE 8/8；首次失败是 ICD 路径配置，改为登记安装路径后无需重建即全过 |
| source/static/shared | PASS：Windows selector 完成 Compute/Graph delivery smoke；Apple shared/static 与 Vulkan static 补足原生可移植性；未改变的 Linux CUDA/FlashInfer 接缝复用其不可变资格证据 |
| 资源/失败 | PASS（已接域）：native completion、两 invocation、变化输入、旧结果 lease、close/失败不发布与重复执行均由路线测试覆盖 |
| 性能 | P0 PASS：packing/repack、workspace、同步、half/BF16 边界与压缩驻留均显式保留。没有冻结的 B2 完整模型片段 P1 卡，因此 selector 按合同 SKIPPED；未把未运行 P1 写成 PASS |

## 有限 residual

- 四路线的 F05.3、F05.4、F07.3 仍需 M3 按真实模型片段逐格汇总，不能由部分 shape 或原子组成代签。
- CPU 没有与公共 Q4-S128、Q4-A64、Q8-B32 W8A16 物理合同相同的准入载体；oneDNN 在当前
  i9-13900K 上对精确 Q8 请求返回 `dnnl_unimplemented`。
- CUDA 只由 Marlin 证明 Q4-S128；Q4-A64、Q8-B32 与当前 ggml carrier 不等价。
- Apple 只由 MLX 证明 unsigned affine Q4/group64；Q4-S128、Q8-B32 不等价。
- Vulkan 的 ggml Q4_0/Q4_1/Q8_0 与三个公共格式均不同，IREE 也没有精确 B2 制品。

## 规模与下一步

统一 `loc-report.py` 在不可变产品候选上统计 Compute+Graph **59,043 实现行 / 251 文件**，
self-check PASS；排除文档、测试、bench、vendor、构建脚本和 scratch。B3 可以按
[批次合同](llm-moe-m2-batch-contracts.md#5-b3递归混合状态)启动，先接 ggml 已存在的
`ssm_conv` 与 `gated_delta_net`，没有成熟实现的域继续记 residual。
