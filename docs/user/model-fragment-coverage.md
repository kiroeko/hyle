# 固定模型画像的 Compute / Graph 支持范围

本页先保留2026-10-05的[D5验收快照](../implementation/tasks/compute/llm-moe-d5-acceptance-ledger.json)，
Apple当前扩展与原FP4缺口的后续结果见文末[Apple当前资格](#apple-当前共同目录与官方fp4资格)。旧快照
说明可依赖的**模型片段支持范围**。每个表格单元同时对应普通Direct和Graph两个入口；
“通过”表示该固定画像的全部必需片段已运行，并通过原数值、状态反馈及资源/完成检查。
它不表示整模型推理、完整上下文容量或任意硬件已经验收。HYLE不负责模型加载、tokenizer或采样。

## D5画像与路线快照

画像链接固定到受审制品revision。`Q4`、`Q8`、官方版本都可能混合不同角色的编码，
不能把整模型设为一种dtype；例如Kimi和DeepSeek的GGUF画像仍含MXFP4专家，norm/state等另用F32。
逐角色格式、shape与region需求见[模型审计](../implementation/tasks/compute/llm-moe-model-weight-operator-audit.md)。

| 模型 | 固定制品画像 | CPU | CUDA | Apple | Vulkan |
|---|---|---|---|---|---|
| Qwen3.5-2B | [official-bf16](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.5-2B | [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.5-2B | [UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.8-27B | [official-bf16](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.8-27B | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.8-27B | [UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.8-2.4T-A95B | [official-bf16](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/tree/207bd685a7e3696cfaff12ded7c6a7ea0f88c996) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.8-2.4T-A95B | [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556) | 通过 | 通过 | 通过 | 通过 |
| Qwen3.8-2.4T-A95B | [Q8_0](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556) | 通过 | 通过 | 通过 | 通过 |
| Kimi K3 | [official-mxfp4](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06) | 通过 | 通过 | 通过 | 通过 |
| Kimi K3 | [UD-Q4_K_XL](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b) | 通过 | 通过 | 通过 | 通过 |
| Kimi K3 | [UD-Q8_K_XL](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b) | 通过 | 通过 | 通过 | 通过 |
| DeepSeek V4 Pro | [official-fp4-fp8](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1) | 通过¹ | 通过 | 准备拒绝 | 准备拒绝 |
| DeepSeek V4 Flash | [official-fp4-fp8](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1) | 通过¹ | 通过 | 准备拒绝 | 准备拒绝 |
| DeepSeek V4 Flash | [UD-Q4_K_XL](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/tree/e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0) | 通过 | 通过 | 通过 | 通过 |
| DeepSeek V4 Flash | [UD-Q8_K_XL](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/tree/e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0) | 通过 | 通过 | 通过 | 数值失败 |
| GLM-5.3 | [official-fp8](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42) | 通过¹ | 通过 | 通过 | 准备拒绝 |
| GLM-5.3 | [UD-Q4_K_XL](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771) | 通过 | 通过 | 通过 | 通过 |
| GLM-5.3 | [Q8_0](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771) | 通过 | 通过 | 通过 | 通过 |
| GLM-5.3-Flash | [official-fp8](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) | 通过¹ | 通过 | 通过 | 准备拒绝 |
| GLM-5.3-Flash | [UD-Q4_K_XL](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9) | 通过 | 通过 | 通过 | 通过 |
| GLM-5.3-Flash | [Q8_0](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9) | 通过 | 通过 | 通过 | 通过 |

¹ 这四个CPU官方画像要求Windows/Linux x86_64启用CPU-only LibTorch2.8.0+cpu，
见[构建与资源档位](compute.md#可选-cpu-fp8fp4-成熟程序)；无该SDK的构建仍拒绝对应域。
完整片段保留原A8、FP4/FP8、BF16/F32边界，不是硬件原生FP8 GEMM。

合计176个入口格：**162运行通过、12准备拒绝、2数值失败**，没有尚未执行验收的画像格。
CPU为44/44通过，CUDA为44/44，Apple为40/44，Vulkan为34/44。这个计数是片段正确性范围，
不是整个Compute/Graph目标的完成百分比；CPU、Apple、CUDA的代表片段完整成本性能验收已通过，
Vulkan仍因A/A资格失败而未完成。

## D5拒绝与失败记录

| 路线与画像 | 必需边界与结果 | 证据 |
|---|---|---|
| Apple：DeepSeek两个官方FP4/FP8画像 | 当时grouped FP4拒绝且原native有tiny差异；本轮对应原数学/实际形状已接入并资格化，见文末当前Apple结果 | [原完整拒绝](../../evidence/correctness/llm-moe-d5-real-fragments/apple-delivery/README.md)、[原生反例](../../evidence/correctness/llm-moe-d5-real-fragments/apple-fp4-native-boundary/README.md) |
| Vulkan：GLM两个官方FP8、DeepSeek两个官方FP4/FP8画像 | 完整MoE分别在grouped FP8/FP4冷准备或完整Graph编译拒绝 | [Vulkan边界](../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-delivery/README.md) |
| Vulkan：DeepSeek Flash `UD-Q8_K_XL` | 长位置attention/压缩片段未通过原数值门；同模型`UD-Q4_K_XL`逐画像通过 | [逐画像裁决](../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-delivery/README.md) |

“准备拒绝”表示当前已接入的路线不能执行该必需域，不证明所有成熟库都没有相应能力。
这些画像不能因其它片段通过、交付测试正确拒绝、换用GGUF或CPU回退而标成支持。
一次成功的`prepare/compile`也不能覆盖上表明确保留的数值失败域。

## 验证的硬件、数据与容量

- 当前D5主要验证环境：Windows i9-13900K（CPU）、RTX4070（CUDA SM89 / Vulkan），
  Apple Silicon M4 / MacBook Air（Apple）。每份收据绑定自己的工具链、Provider与驱动事实；
  不能用此表外推所有CPU/GPU。CUDA官方低精度原版AOT程序的当前资格要求SM89和Driver API13040，
  该API版本不是NVIDIA驱动安装包版本号。
- Qwen3.5-2B官方片段使用核对过的本地checkpoint。其它画像采用受固定header/config/源码约束的
  非零、格式合法测试payload；没有下载并运行所有巨型模型的完整权重。
- 保留模型内宽、逐角色物理编码、分组/block、状态维度、必需语义端口及完整词表行；
  batch、token、驻留专家数和KV容量按消费配方缩小。通过不等于27B/2.4T整模型能够在测试机常驻。
- recurrent完整片段只发布attended output和next state，按原门检查真实状态反馈。
  Provider内部gate不作为独立合格输出，普通Softplus的精度合同没有放宽。

## 交付与性能分别成立

| 路线 | source / static / shared代表消费者 | 本轮两个完整片段P1 |
|---|---|---|
| CPU | [原24/24](../../evidence/correctness/llm-moe-d5-real-fragments/cpu-delivery/README.md)及[官方低精度每种28项通过](../../evidence/correctness/llm-moe-d5-real-fragments/cpu-official-torch-binding/README.md) | [原批次](../../evidence/performance/graph-cpu-d5-fragments-p1/README.md)及[官方FP8/FP4完整MoE通过](../../evidence/performance/cpu-official-fragments-p1/README.md) |
| CUDA | [24/24通过](../../evidence/correctness/llm-moe-d5-real-fragments/cuda-delivery/README.md) | [已通过，原失败保留](../../evidence/performance/graph-cuda-d5-fragments-p1/README.md) |
| Apple | [36项通过，含预期拒绝](../../evidence/correctness/llm-moe-d5-real-fragments/apple-delivery/README.md) | [已通过](../../evidence/performance/graph-apple-d5-fragments-p1/README.md) |
| Vulkan | [36项通过，含预期拒绝](../../evidence/correctness/llm-moe-d5-real-fragments/vulkan-delivery/README.md) | [恢复A/A未合格，正式采样被阻止；P1未完成](../../evidence/performance/graph-vulkan-d5-fragments-recovery-preflight/README.md) |

P1比较普通Graph与同完成/资源合同的Direct完整片段，计入必要转换、同步与释放。
CPU/Apple/CUDA通过5%工程非劣门仅覆盖各自受测片段：共同BF16 FFN，CPU/Apple混合GGUF MoE、
CUDA官方FP4/FP8 MoE，以及本次CPU官方GLM FP8/DeepSeek FP4/FP8 MoE；
不表示每个算子都更快或整模型SOTA。
三交付验证也不为上表拒绝/失败的画像提供运行支持。

## 普通使用与证据

用户选择[CPU/CUDA Compute](compute.md)、[Graph](graph.md)、[Apple](apple-compute.md)或
[Vulkan](vulkan-compute.md)入口。已知制品时提供不可变`ModelArtifactProfile`与`WorkloadContext`，
实际每个端口仍用`TensorDesc`表达dtype、物理编码和shape；模型名或“Q4”标签不能替代这些描述。
准备期匹配硬件/软件、画像、operation/region及全部端口条件并绑定，重复执行不重新搜索或计时选路。

完整片段收据按固定批次复用：[Qwen九画像](../../evidence/correctness/llm-moe-d5-real-fragments/qwen-nine-profile-complete-gate/README.md)、
[Kimi三画像](../../evidence/correctness/llm-moe-d5-real-fragments/kimi-remaining-batch/README.md)、
[GLM四份GGUF](../../evidence/correctness/llm-moe-d5-real-fragments/glm-four-gguf-batch/README.md)、
[GLM官方CUDA Pro](../../evidence/correctness/llm-moe-d5-real-fragments/glm-official-dsa-cuda/README.md)/
[Flash](../../evidence/correctness/llm-moe-d5-real-fragments/glm-official-flash-cuda/README.md)/
[Apple](../../evidence/correctness/llm-moe-d5-real-fragments/apple-fp8-glm-complete/README.md)、
[DeepSeek官方CUDA](../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-official-cuda-complete/README.md)、
[CPU官方四画像](../../evidence/correctness/llm-moe-d5-real-fragments/cpu-official-torch-binding/receipt.json)、
[DeepSeek GGUF](../../evidence/correctness/llm-moe-d5-real-fragments/deepseek-gguf-three-route/README.md)。
Vulkan GGUF的后续逐画像裁决以上方当前收据为准。历史失败和撤回资格仍保留，不能覆盖当前ledger。

## CPU 热门量化与新版扩展缺口

面向更多 GGUF 档位、独立 MTP 和 DeepSeek Pro-0813/Flash-0731 DSpark 的
[具体缺口清单](../implementation/tasks/compute/cpu-popular-quant-mtp-gap-list.md)已单独盘点。
它区分公开格式接线缺失、已有格式的新制品资格、完整分支接线及身份对账，不改变本页当前通过数。
接入顺序和五个有限批次见[新的CPU扩覆盖计划](../implementation/tasks/compute/cpu-quant-mtp-coverage-expansion-plan.md)。
CPU扩展已经进入实施：11种新增GGUF格式和已准入NF4/AWQ/CT/MLX接缝已取得基础资格；
新版Pro-0813/Flash-0731主干与三阶段DSpark、Qwen混合Q3/Q4/Q6 MTP及两份Flash混合sidecar，
已通过Windows/Linux普通Direct/Graph计算链与安装入口，见
[当前C4资格](../implementation/tasks/compute/cpu-quant-mtp-c4-model-branches.md#本包交接资格)。
精确软件/制品身份与资源限制以对应收据为准；104个GGUF档位及全部准入非GGUF角色
仍待C5最终矩阵，不把基础格式接线计为这些新画像全部完成，也不改变本页旧ledger通过数。

## Apple 当前共同目录与官方FP4资格

当前Apple使用与CPU/CUDA相同的[模型算子目录](../architecture/model-compute-catalogue.md)和
[逐制品端口](../implementation/tasks/compute/cpu-final-demand-carriers.json)。上述D5旧快照
不会因扩展任务完成而回写；不能把其Apple40/44或旧FP4拒绝当成当前产品范围。

| 当前需求 | 普通Direct／Graph资格 | 证据 |
|---|---|---|
| 104 GGUF制品归并的559合同、34packed、45plain，八模型族 | 两入口实际geometry/编码/变化输入与replay通过；完整词表按分片、驻留专家有界 | [A2实际矩阵](../../evidence/correctness/apple-parity-a2/README.md) |
| Pro0813／Flash0731官方混合FP4/FP8主干 | 原group32 FP4精确BF16、原FP8 activation/专家、ratio0/4/128压缩attention/indexer和完整MoE通过 | [FP4资格](../implementation/tasks/compute/apple-ds4-fp4-source-qualification.md) |
| 必要MTP与DSpark | 两个官方三stage DSpark、Qwen混合Q3/Q4/Q6 sidecar完整MTP、Flash Q8/BF16 sidecars完整DSpark通过 | [当前Apple卡](../implementation/tasks/compute/apple-cpu-cuda-parity-closure.md) |
| source／static／shared | 同一公开consumer实际运行上述新FP4承载、重放/坏ID/恢复/lease；离线搬迁与资源失配检查通过 | [A4交付](../../evidence/correctness/apple-parity-a4/README.md) |

机器为当前MacBook Air M4/16GiB；资格有实际工具链/原source版本边界，不等于巨型整模型
驻留、loader或推理引擎。原native tiny失败保留；小段GPU位域桥接保持原数学与存储合同。
性能仍仅对具名完整成本域裁决，不能由这些正确性/交付结果宣称全部硬件SOTA。
