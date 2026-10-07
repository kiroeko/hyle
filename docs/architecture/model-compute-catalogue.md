# 目标模型、算子与实际量化格式共同目录

这是 CPU／CUDA／Apple／Vulkan **共同的需求目录**，从 CPU 扩覆盖轮已冻结的真实模型与制品事实派生。
各路线可采用不同成熟 Provider；相同模型／tensor／精度／状态合同应有相同能力目标。
本表不是四路线支持表，也不把 CPU PASS 复制给 GPU。路线验收仍看当前 owner 卡与实际收据。

需求真相为 [逐制品／端口目录](../implementation/tasks/compute/cpu-final-demand-carriers.json)，
真实 tensor 名／shape／原 storage 示例见 [完整 header 汇总](../../evidence/correctness/cpu-quant-mtp-c1-demand/actual-storage-matrix.json)。
本页是它们的人读视图，不另造版本队列。完整目录保留 104 GGUF、559 去重数值合同、80 参数类、
34 non-GGUF packed、45 non-GGUF plain、官方混合权重与 MTP／DSpark 分支。

文件名 Q4／Q8 不能代替下面的真实格式；storage、activation、accumulation、output、state 必须分别核对。
元数据证明权重需求，固定 forward／完整计算片段证明算子及数值／状态边界。

## 1. 模型与 GGUF 制品

| 模型 | 来源 | 档位数 | 固定档位（一个档位可混用多种 tensor 格式） |
|---|---|---|---|
| Qwen3.5-2B | [f6d5376b](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae) | 22 | BF16, IQ4_NL, IQ4_XS, Q3_K_M, Q3_K_S, Q4_0, Q4_1, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K, Q8_0, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_XXS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL, UD-Q8_K_XL |
| Qwen3.8-27B | [4ca72078](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | 25 | BF16, Q4_0, Q4_1, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_S, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_M, UD-Q4_K_S, UD-Q4_K_XL, UD-Q5_K_M, UD-Q5_K_S, UD-Q5_K_XL, UD-Q6_K, UD-Q6_K_L, UD-Q6_K_M, UD-Q6_K_XL, UD-Q8_K_L, UD-Q8_K_XL |
| Qwen3.8-2.4T-A95B | [567d3e6a](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556) | 9 | BF16, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_XS, UD-IQ2_XXS, UD-IQ3_XXS, UD-IQ4_XS, UD-Q1_0 |
| Kimi-K3 | [a0836360](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b) | 9 | UD-IQ1_M, UD-IQ1_S, UD-IQ2_XXS, UD-Q1_0, UD-Q2_K_XL, UD-Q4_K_XL, UD-Q8_K_XL, UD-TQ1_0, UD-TQ2_0 |
| GLM-5.3 | [346b3591](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771) | 12 | BF16, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ3_XXS, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL |
| GLM-5.3-Flash | [a38483c8](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/a38483c8cd5df544f53d70fb281afe97369d5ab6) | 12 | BF16, Q8_0, UD-IQ1_M, UD-IQ1_S, UD-IQ2_XXS, UD-IQ3_XXS, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL |
| DeepSeek-V4-Pro-0813 | [6d053616](https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF/tree/6d053616d152293da72569f56b861466f10ace7d) | 2 | UD-Q4_K_XL, UD-Q8_K_XL |
| DeepSeek-V4-Flash-0731 | [fbbb5b93](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/tree/fbbb5b93fb787c21338159b0af3318bb3f4d9768) | 13 | UD-IQ1_M, UD-IQ1_S, UD-IQ2_M, UD-IQ2_XXS, UD-IQ3_S, UD-IQ3_XXS, UD-IQ4_NL, UD-IQ4_XS, UD-Q2_K_XL, UD-Q3_K_M, UD-Q3_K_XL, UD-Q4_K_XL, UD-Q8_K_XL |

## 2. 算子／物理格式／所属模型／tensor 用途

下表逐真实数值权重角色归并；具体 K／N／expert 数、profile ID、全部原 tensor 示例保留在上述目录。
此处列出的格式是同角色跨已固定档位的并集，不能理解为单个档位或每一层都用全部格式。

| 模型 | 公共计算 | 实际权重 storage | 用途／角色 | 去重合同数 |
|---|---|---|---|---|
| Qwen3.5-2B | Gather | BF16, F16, Q5_K, Q6_K, Q8_0 | Embedding：词表／Markov lookup | 5 |
| Qwen3.5-2B | MatMul | BF16, F16, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 | Projection：attention／FFN／其它投影 | 82 |
| Qwen3.8-27B | Gather | BF16, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q6_K, Q8_0 | Embedding：词表／Markov lookup | 8 |
| Qwen3.8-27B | MatMul | BF16, F32, IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0 | Projection：attention／FFN／其它投影 | 129 |
| Qwen3.8-27B | MatMul | BF16, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 | Head：完整词表输出 | 6 |
| Qwen3.8-2.4T-A95B | Gather | BF16, Q4_K, Q8_0 | Embedding：词表／Markov lookup | 3 |
| Qwen3.8-2.4T-A95B | MatMul | BF16, F32, Q5_K, Q6_K, Q8_0 | Projection：attention／FFN／其它投影 | 27 |
| Qwen3.8-2.4T-A95B | GroupedMatMul | BF16, IQ1_M, IQ1_S, IQ1_XXXS, IQ2_XS, IQ2_XXS, IQ3_XXS, IQ4_XS, Q2_K, Q3_K, Q4_K, Q8_0 | Expert：按索引选择 expert | 24 |
| Qwen3.8-2.4T-A95B | MatMul | BF16, Q4_K, Q8_0 | Head：完整词表输出 | 3 |
| Kimi-K3 | Gather | BF16, Q6_K, Q8_0 | Embedding：词表／Markov lookup | 3 |
| Kimi-K3 | MatMul | BF16, F32, Q6_K, Q8_0 | Projection：attention／FFN／其它投影 | 41 |
| Kimi-K3 | GroupedMatMul | IQ1_M, IQ1_S, IQ1_XS, IQ1_XXS, IQ1_XXXS, IQ2_XS, IQ2_XXS, IQ3_XXS, MXFP4 | Expert：按索引选择 expert | 18 |
| Kimi-K3 | MatMul | BF16, Q6_K, Q8_0 | Head：完整词表输出 | 3 |
| Kimi-K3 | MatMul | BF16, Q6_K, Q8_0 | HeadGroup：分组输出／head | 5 |
| GLM-5.3 | Gather | BF16, Q4_K, Q5_K, Q6_K, Q8_0 | Embedding：词表／Markov lookup | 5 |
| GLM-5.3 | MatMul | BF16, F32, Q5_K, Q6_K, Q8_0 | Projection：attention／FFN／其它投影 | 30 |
| GLM-5.3 | GroupedMatMul | BF16, IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 | Expert：按索引选择 expert | 24 |
| GLM-5.3 | MatMul | BF16, Q4_K, Q6_K, Q8_0 | Head：完整词表输出 | 4 |
| GLM-5.3 | MatMul | BF16, Q8_0 | HeadGroup：分组输出／head | 4 |
| GLM-5.3-Flash | Gather | BF16, Q4_K, Q5_K, Q6_K, Q8_0 | Embedding：词表／Markov lookup | 5 |
| GLM-5.3-Flash | MatMul | BF16, F32, Q5_K, Q6_K, Q8_0 | Projection：attention／FFN／其它投影 | 46 |
| GLM-5.3-Flash | GroupedMatMul | BF16, IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 | Expert：按索引选择 expert | 24 |
| GLM-5.3-Flash | MatMul | BF16, Q4_K, Q6_K, Q8_0 | Head：完整词表输出 | 4 |
| GLM-5.3-Flash | MatMul | BF16, Q8_0 | HeadGroup：分组输出／head | 4 |
| GLM-5.3-Flash | Gather | F32 | Position：position 参数 lookup | 1 |
| DeepSeek-V4-Pro-0813 | Gather | BF16, Q8_0 | Embedding：词表／Markov lookup | 2 |
| DeepSeek-V4-Pro-0813 | MatMul | BF16, F32, Q8_0 | Projection：attention／FFN／其它投影 | 25 |
| DeepSeek-V4-Pro-0813 | GroupedMatMul | MXFP4 | Expert：按索引选择 expert | 2 |
| DeepSeek-V4-Pro-0813 | MatMul | BF16, Q8_0 | Head：完整词表输出 | 2 |
| DeepSeek-V4-Pro-0813 | Gather | F32 | Position：position 参数 lookup | 3 |
| DeepSeek-V4-Pro-0813 | Gather | I32 | Integer：整数表／索引消费 | 1 |
| DeepSeek-V4-Flash-0731 | Gather | BF16, Q4_K, Q5_K, Q6_K, Q8_0 | Embedding：词表／Markov lookup | 5 |
| DeepSeek-V4-Flash-0731 | MatMul | BF16, F32, Q5_K, Q6_K, Q8_0 | Projection：attention／FFN／其它投影 | 28 |
| DeepSeek-V4-Flash-0731 | GroupedMatMul | IQ1_M, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, MXFP4 | Expert：按索引选择 expert | 10 |
| DeepSeek-V4-Flash-0731 | MatMul | BF16, Q4_K, Q6_K, Q8_0 | Head：完整词表输出 | 4 |
| DeepSeek-V4-Flash-0731 | Gather | F32 | Position：position 参数 lookup | 3 |
| DeepSeek-V4-Flash-0731 | Gather | I32 | Integer：整数表／索引消费 | 1 |

## 3. 其它必要算子与完整 region

这部分不能仅从 MatMul 数量推出；完整 consumer、精度与状态来源见
[共同语义／实际 forward 审计](../implementation/tasks/compute/llm-moe-model-weight-operator-audit.md#算子模型位置与量化目标)
和 [当前逐族完整程序索引](../../evidence/correctness/cpu-quant-mtp-c5-required/family-witness-index.json)。
DeepSeek 新官方与 DSpark 以第 5 节的当前身份／程序为准，旧 preview 是回归来源。

| 算子／完整 region | 所属模型与用途 | 精度／状态要点 |
|---|---|---|
| RMSNorm／LayerNorm、Cast、残差 Add／Multiply、SiLU／SwiGLU／SiTU、稳定函数 | 全部；层 norm、门控、残差，Kimi 的 SiTU | norm／router／state 与 projection 权重量化分开；累加及输出 storage 按原合同 |
| GQA／QK norm、RoPE／YaRN、mask、Softmax／SDPA、KV 反馈 | 三个 Qwen 的 full attention | 原 QKV／position、cache dtype、F32 累加及输出边界 |
| CausalConv1d、RecurrentGatedDeltaNet、完整 gate／state 反馈 | 三个 Qwen 的 recurrent 层 | 权重可混合量化，conv／recurrent math／state 的 F32 合同保留 |
| KDA、三路 conv、decay／gate／recurrent state | Kimi K3、GLM-5.3-Flash | 原投影精度与 F32 state 独立，不用单次 attention 代签 |
| MLA、低 rank Q／KV、NoPE／RoPE、cache／output gate | Kimi、GLM 两款、DeepSeek 两款 | BF16／FP8／GGUF 投影；原中间 rounding 与 cache 反馈 |
| DSA／LightningIndexer、pool／compression、TopK／Gather、CSA／HCA、ragged tail | GLM 两款、DeepSeek 两款 | 原 index／compression ratio、window、跨调用 state；索引 I32 |
| Router／score／bias、稳定 TopK、Scan／Gather／Scatter／combine、grouped expert／shared expert | Qwen2.4T、Kimi、GLM 两款、DeepSeek 两款 | tie／空或偏斜 expert／重复 IDs／输出顺序，完整 activation／scale／combine |
| HyperConnection pre／combine／post、AttnRes、跨片段残差 | DeepSeek 两款、GLM Flash、Kimi | 原 small parameter／state 精度与 identity，不能只测小矩阵 |
| FP8 weight decode／dynamic activation quant／block roundtrip／grouped FP4／FP8 linear | 官方 Kimi／GLM／DeepSeek | 原 codes／scale／E8M0、group／block、rounding 与输出精度 |
| MTP／NextN、DSpark target_hidden／main projection／Markov／confidence／完整三 stage | 按第 5 节逐制品适用；Kimi 无 MTP | 独立 weights／norm／carry／cache；不能用主模型下一 token 输出代签 |
| 准备／绑定／资源复用、完成／失败、重复执行 | 每个以上合同 | 路线原生资源；准备时匹配，运行时不重选；失败／OOM／寿命按原合同 |

## 4. 四份 non-GGUF 制品

| 模型／固定制品 | 格式族 | 计算需求 | group | packed 合同数 |
|---|---|---|---|---|
| [techwithsergiu/Qwen3.5-2B-bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-2B-bnb-4bit/tree/97fb2bcc0255f1d5b36f07daa54b76cb29444df6) | Nf4 | ArtifactWeightDecode → 原精度 MatMul／Gather | 64 | 6 |
| [mattbucci/Qwen3.8-27B-AWQ](https://huggingface.co/mattbucci/Qwen3.8-27B-AWQ/tree/2cc4cffb391645c9be52f3fe50ff595ac7df3e38) | Awq | ArtifactWeightDecode → 原精度 MatMul／Gather | 128 | 7 |
| [soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ](https://huggingface.co/soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ/tree/a1b8f1570ecc09758c8431fd536c88e3b209674f) | Ct | ArtifactWeightDecode → 原精度 MatMul／Gather | 128 | 5 |
| [ToPo-ToPo/DeepSeek-V4-Flash-0731-mlx-4bit](https://huggingface.co/ToPo-ToPo/DeepSeek-V4-Flash-0731-mlx-4bit/tree/c8b8f3fe9112cc5aa3b34d2aec9f771b68622902) | Affine, Mx4, Mx8 | MlxQuantized linear／grouped／embedding | 32, 64 | 16 |

NF4 保留 codebook／absmax／nested 二级 scale／offset；AWQ 保留 GEMM 原 nibble permutation／zero point；
CT 保留 symmetric signed offset、实际 scale dtype；MLX affine 保留 scale／bias，MX4／MX8 保留 E8M0。
编码、FP16／BF16／F32 舍入和原 K group／selected expert 边界见公共 descriptor／共同 consumer，不能按“都是 4-bit”互换。
另 45 个 plain 端口同样在目录中，包含未量化 embedding／norm／conv／position／主干排除项。

## 5. 官方更新、MTP 与 DSpark

| 固定来源 | 实际 storage 并集 | 必需分支 | 共同 consumer／边界 |
|---|---|---|---|
| [deepseek-ai/DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/tree/72e1d3230f6c080a530b0a1d46f8eb4602340597) | BF16, F32, F8_E4M3, F8_E8M0, I64, I8 | 主干＋新版 DSpark：target hidden／main projection／Markov／confidence／三 stage | cpu_c4_model_branches.h 的两个当前 profile；原 header／forward 精度与状态不变 |
| [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062) | BF16, F32, F8_E4M3, F8_E8M0, I64, I8 | 主干＋新版 DSpark：target hidden／main projection／Markov／confidence／三 stage | cpu_c4_model_branches.h 的两个当前 profile；原 header／forward 精度与状态不变 |
| [MTP/mtp-Qwen3.8-27B-Q4_0.gguf](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/MTP/mtp-Qwen3.8-27B-Q4_0.gguf) | F32, Q3_K, Q4_K, Q6_K | mtp | 按完整 branch 原 tensor／weight／cache／head 资格，文件 Q 标签不作 dtype |
| [dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/fbbb5b93fb787c21338159b0af3318bb3f4d9768/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf) | BF16, F32, MXFP4, Q8_0 | dspark | 按完整 branch 原 tensor／weight／cache／head 资格，文件 Q 标签不作 dtype |
| [dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/fbbb5b93fb787c21338159b0af3318bb3f4d9768/dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf) | BF16, F32, MXFP4 | dspark | 按完整 branch 原 tensor／weight／cache／head 资格，文件 Q 标签不作 dtype |

Qwen 官方／其它六官方身份与 precision／shape 原要求保留在共同 JSON 的 `prior_official_profiles`；
该字段名中的 prior 表示证据出处，不撤回已准入能力。Qwen3.5 官方含 MTP，而固定 GGUF 语言主档不含其 MTP；
Qwen27B 有独立 sidecar；Kimi 明确没有 MTP。是否必需逐具体制品核对，不对所有文件一律加分支。

## 6. 路线验收如何引用

- [CPU 当前完整覆盖](../user/cpu-model-coverage.md)／[CPU 最终终审](../implementation/tasks/compute/cpu-final-f6-final-qualification.md)
  证明当前 CPU 的实际能力与限制。
- [CUDA 当前 U1 对账](../../evidence/correctness/cuda-final-u1/README.md)使用这份共同要求，
  区分成熟源码存在、HYLE hook、原编码／精度／shape／资源、普通 Direct／Graph 及运行资格；未运行不签 PASS。
- [旧四路线代表画像](../user/model-fragment-coverage.md)只签其原 22 画像范围，不签新增 104 GGUF／non-GGUF／新版 branch。

本页不增加模型 loader／tokenizer／采样／整模型推理承诺，不开放自研算子扩张或新一轮模型调查。
更新需求需先改变共同固定目录；各路线对同一版本接线，避免各自维护一套算子目标。
