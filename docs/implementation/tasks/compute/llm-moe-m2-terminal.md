# M2 四路线成熟 Provider 批量接入终局

<!-- MODULE REVIEW: CLOSED -->

## 裁决

2026-09-24，M2 的 B1--B6 六个批次全部按 Provider 原生 region/能力族方法结束，最终验证候选为
`68278c999b623edbd0c87f05d3926208ea84f6db`。M2 的退出结论是
`PASS_WITH_RECORDED_RESIDUALS`：所有已经准入的成熟入口均已接入普通 Direct/Graph 并完成本批
适用验证，未接通的已准入入口为 0；上游不支持、设备不接受、精确数值合同不相容或需要完整模型链
才能裁决的项均保留具名 residual，没有用冻结的 HYLE 自实现补齐或伪造 PASS。

组件 owner 保持 `AUDITED`，但本终局不授权继续修改产品。M3 必须按真实模型计算链重新绑定有限
范围后才可施工；依用户要求，本轮在 M2 完成后暂停，不进入 M3。

## 六个批次的实际结果

| 批次 | 已接成熟能力 | 留给 M3 的边界 |
|---|---|---|
| [B1](llm-moe-m2-b1-terminal.md)（含 [Apple 路线卡](llm-moe-m2-b1-apple.md)） | 四路线基础 tensor/state、普通 Direct/Graph、原生资源与批量 lowering | 上游未接受的 dtype/layout/index 组合逐格汇总 |
| [B2](llm-moe-m2-b2-terminal.md)（[施工卡](llm-moe-m2-b2-linear-attention.md)） | oneDNN/ggml/cuBLASLt/Marlin/FlashInfer/MLX/IREE 的 dense、有限量化线性、GQA/KV、RoPE/YaRN/Hadamard | 完整 FFN/gated-attention region 与未相容 Q4/Q8 格式 |
| [B3](llm-moe-m2-b3-terminal.md) | 四路线原版 causal conv 与 scalar/KDA GatedDeltaNet，显式 state 与多输出 | 24 层、chunk/decode、更多 dtype/state 形状 |
| [B4](llm-moe-m2-b4-terminal.md) | ggml 原版 grouped expert MatMul 与 DeepSeek mHC combine/pre/post | 完整 routing→expert→combine、SiTU、MTP 与量化 expert |
| [B5](llm-moe-m2-b5-terminal.md) | 四路线原版 Lightning Indexer；已有 MLA、pool/tail、compression 组合边界 | 完整 sparse/compression checkpoint 片段与跨层状态 |
| [B6](llm-moe-m2-b6-low-precision.md#终局裁决) | IREE Vulkan 精确 E4M3 block128/group32 dense 路径及 BF16 组合验收 | CPU/CUDA/Apple 设备资格、W4A8、grouped/index 与完整 2B FP8 链 |

M2 没有新增或优化 CPU/CUDA/Metal/Vulkan 数学 kernel、shader、排序、归约、scan 或量化算法。
接入层保留各路线自己的 program、buffer/array、stream/queue、completion 与失败回收；模型/config、
phase、量化档位、平台、硬件架构、驱动/SDK 和 Provider 版本在 prepare 阶段参与冷规则，replay 不搜索、
计时或重新选择 Provider。

## 最终验证

| 风险 | 结果 |
|---|---|
| B1--B5 | 各自终局已保存 Windows/WSL/Air/Mini 的适用正确性、资源、source/static/shared 与 P0/P1 裁决；未变接缝按 tree/contract/toolchain/machine 身份复用 |
| B6 Windows | PASS：候选 `68278c999`，RTX 4070，Vulkan SDK 1.4.357.0；selector configure、`factor:compute_route_vulkan`、`factor:graph_route_vulkan` 全部通过 |
| B6 Air static | PASS：Apple M4、MoltenVK 1.4.1；静态 Direct Vulkan 聚合测试 125/125，Graph Vulkan 聚合测试通过 |
| B6 资源与语义 | PASS：两种精确 FP8 scale contract、BF16 continuation、两次 replay、压缩输入驻留、旧结果与完成回收 |
| B6 性能 | `NOT_APPLICABLE`：本批没有冻结完整 F18 模型片段；P0 保留 packing、decode、workspace、同步与转换成本，完整链性能归 M3-06 |

B6 的机器与命令记录见[验收收据](../../../../evidence/correctness/compute-llm-moe-m2-b6/acceptance.json)，
Provider/格式探针见[资格收据](../../../../evidence/correctness/compute-llm-moe-m2-b6/qualification.json)。
统一 `loc-report.py --self-check` 在不可变候选上报告 Compute+Graph 实现 **59,749 行 / 252 文件**，
排除测试、文档、bench、vendor、构建产物和 scratch。

## 未完成项与下一阶段

M2 关闭不等于 256 个“需求行 × 路线”格已经全部 PASS。M3 仍需用本地 Qwen3.5-2B 与已登记的模型
机制片段完成真实链验收，逐格区分已覆盖、上游缺失、设备不接受、数值/状态不相容和未运行验证。
M3 完成 Provider 接入门后，M4 才能判断是否存在必须补写的最小算法缺口；冻结自实现继续保持不动。
