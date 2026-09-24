# LLM/MoE 目标模型权重、量化与算子审计

## 结论

2026-09-24 对八个目标模型的固定 Hugging Face revision 逐分片读取了 safetensors header，
并对八个主流 Unsloth GGUF 仓库的 4-bit 部署档和高保真 Q8 档逐分片读取了 GGUF header。
读取只覆盖 metadata 和 tensor table，
没有下载权重 data section，也没有运行模型推理。

本审计得到四项会直接改变后续验收合同的事实：

1. **量化版本名不是全模型 dtype。** `UD-Q4_K_XL`、`UD-Q4_K_M`、`UD-IQ4_XS`
   都按张量角色混用多种格式；同一文件可同时出现 F32、F16、Q8、Q6、Q5、Q4、Q3、IQ 和
   MXFP4。HYLE 不得再用一个全局“Q4 block”描述整个模型。
2. **官方 checkpoint 也可能是混合格式。** Kimi K3 官方权重已经是 MXFP4 packed expert +
   BF16/F32；DeepSeek V4 是 packed FP4 expert、FP8、UE8M0 scale、BF16/F32 的组合；GLM 5.3
   是 FP8 block-128 与大量 BF16/F32 排除项的组合。
3. **存储、执行和状态精度是三个合同。** Qwen3.8-27B 的权重 header 全是 BF16，但配置把
   recurrent state/math 固定为 F32。将 BF16 checkpoint 直接交给只接受 F32 region 并被拒绝，
   不能证明模型链数值不相容；正确链路包含准备期转换和 F32 state。
4. **目标必须绑定模型 revision 和量化 profile。** DeepSeek V4 Pro preview 当前没有同 revision
   的 Unsloth GGUF；`DeepSeek-V4-Pro-0813-GGUF` 是后续 sibling，只能提供架构/格式证据，不能
   为 preview revision 代签正确性。

因此，本文件从现在起定义 M3/M4 使用的模型与量化事实输入。旧的全局
`B16/Q4-S128/Q4-A64/Q8-B32/F8-B128` 载体只能保留为历史实验，不再代表真实模型 profile。

## 审计方法与证据等级

| 等级 | 本文件中的含义 |
|---|---|
| `HEADER_OBSERVED` | 从固定 revision 的 safetensors/GGUF header 直接读得 tensor 名、shape 和 storage type |
| `CONFIG_DERIVED` | 从固定 revision 的 `config.json` 读得层型、宽度、专家、量化配置和精度声明 |
| `SOURCE_CONFIRMED` | 从固定源码的实际 forward/kernel 路径确认算子组合或运行时精度 |
| `CARD_PUBLISHED` | 模型卡或量化仓库的说明；不能单独代替 header 或执行验证 |

执行方法：

- 官方模型：先固定 repository SHA，读取 `config.json` 与 safetensors index；对每个分片仅请求
  8-byte header length 和随后的 JSON header。八个仓库共检查 641 个官方分片。
- GGUF：使用 `@huggingface/gguf 0.4.6` 从固定 SHA 的首分片发现完整 shard set，只读取
  metadata/tensor info。16 个选定 profile 共检查 247 个 GGUF 分片；其中首个部署档 105 个，
  高保真档 142 个。
- 本地 `C:\Users\kiro\Downloads\Qwen3.5-2B` 仍是 Qwen3.5-2B 的 canonical 可执行输入，
  source ID 为 `4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`；其
  repository/revision 与本表一致。
- 下文的 element count 是 header 中 shape 的乘积。packed U8/I8 是物理 payload 元素，
  GGUF MXFP4 是逻辑 tensor shape；两者不能直接比较“参数量”。
- 下载数和 likes 仅用于说明社区量化仓库的主流程度，观察后会变化，不进入产品路由条件。

## 固定的官方模型

| 模型与固定来源 | 语言主干结构 | 官方 header 的实际存储分布 | 运行时关键精度 |
|---|---|---|---|
| [Qwen3.5-2B @ `15852e8`](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) | 24 层：18 Gated DeltaNet + 6 full attention；dense FFN；1 MTP | 632 tensors：BF16 596 / 2,274,067,232 elements；F32 36 / 2,592。F32 是 18 个 `A_log` 与 18 个 recurrent norm；`dt_bias` 在该 checkpoint 为 BF16 | model activation/weights 以 BF16 为主；配置 `mamba_ssm_dtype=float32`，recurrent state/math 为 F32 |
| [Qwen3.8-27B @ `1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | 64 层：48 Gated DeltaNet + 16 full attention；dense FFN；1 MTP | 1,199 tensors，全部 BF16 / 27,781,427,952 elements | checkpoint 全 BF16 不改变 F32 recurrent state/math 合同 |
| [Qwen3.8-2.4T-A95B @ `207bd68`](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/tree/207bd685a7e3696cfaff12ded7c6a7ea0f88c996) | 92 层：69 Gated DeltaNet + 23 full attention；512 experts、top-10、shared expert；1 MTP | 1,609 tensors，全部 BF16 / 2,446,182,725,504 elements；routed experts 为按层堆叠 tensor | BF16 projections；F32 recurrent state/math；router/top-k 与 grouped expert 是独立 region |
| [Kimi K3 @ `f831ab6`](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06) | 93 层：69 KDA + 24 MLA/full attention；896 experts、top-16、2 shared；AttnRes、SiTU；无 MTP | 497,220 tensors：U8 494,592 / 1,446,456,066,048 packed elements，BF16 2,122 / 57,179,884,544，F32 506 / 11,122,432。专家 `weight_packed` 和 UE8M0 `weight_scale` 都以 U8 承载 | 官方 config 是 group-32 `mxfp4-pack-quantized`；KDA 的 `A_log/dt_bias`、三路 conv 与 norm 为 F32；模型卡声明 MXFP8 activation，部署时须绑定实际 provider profile |
| [DeepSeek V4 Pro preview @ `b5968e9`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1) | 61 层；MLA + DSA/indexer + compression；384 experts、top-6、1 shared；mHC；1 MTP | 145,116 tensors：packed I8 expert 71,424 / 786,381,668,352；F8_E8M0 scale 71,952 / 49,150,268,416；F8_E4M3 528 / 23,169,335,296；另有 BF16/F32/I64 | expert logical weight 是两 nibble/byte 的 FP4；其它量化 GEMM 采用 E4M3、UE8M0 scale、128×128 block；router/state/mHC/索引保留各自精度 |
| [DeepSeek V4 Flash preview @ `60d8d70`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1) | 43 层；MLA + DSA/indexer + compression；256 experts、top-6、1 shared；mHC；1 MTP | 69,187 tensors：packed I8 expert 33,792 / 141,733,920,768；F8_E8M0 34,167 / 8,858,737,664；F8_E4M3 375 / 6,023,020,544；另有 BF16/F32/I64 | 与 Pro 相同的角色化 FP4/FP8/BF16/F32 合同，shape 与层数不同 |
| [GLM-5.3 @ `aca966e`](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42) | 78 层 MLA/DSA；256 experts、top-8、1 shared；21 full + 57 shared indexer；1 MTP | 118,629 tensors：F8_E4M3 59,044 / 751,226,191,872；F32 scales/bias 59,120 / 45,872,560；BF16 exclusions 465 / 2,103,729,152 | dynamic FP8 E4M3、128×128 block；541 个 `modules_to_not_convert` 保留 BF16/F32 |
| [GLM-5.3-Flash @ `eb9eb20`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) | 45 层：34 KDA + 11 sparse attention；288 experts、top-8、1 shared；mHC；1 MTP | 76,108 tensors：F8_E4M3 37,338 / 314,396,639,232；F32 37,629 / 19,484,766；BF16 exclusions 1,141 / 6,926,096,640 | dynamic FP8 E4M3、128×128 block；1,509 个排除项；KDA state/conv、mHC 与 DSA indexer 有独立精度边界 |

## 固定的主流 GGUF profile

这些 profile 是后续 provider-first 接入和四路线覆盖的量化目标。选择的是一个主流、可部署的
约 4-bit 档；仓库中的其它 1/2/3/5/6/8-bit 档先保留为扩展目录，不在首轮同时展开。Qwen3.8
2.4T 仓库当前没有 Q4_K 档，因此选其已发布的 `UD-IQ4_XS`。Kimi/DeepSeek 的
`UD-Q4_K_XL` 保留原生 MXFP4 expert，并不把专家重新解释成 Q4_K。

| 模型 | 固定 GGUF profile 与来源 | header 观察 | 2026-09-24 社区信号 |
|---|---|---|---|
| Qwen3.5-2B | [`UD-Q4_K_XL` @ `f6d5376`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-UD-Q4_K_XL.gguf) | 1 shard，320 tensors，1,881,825,088 logical parameters；语言模型 GGUF，不含官方 checkpoint 的完整视觉 tensor 集 | repo 319,752 downloads/month、157 likes |
| Qwen3.8-27B | [`UD-Q4_K_M` @ `4ca7207`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | 1 shard，866 tensors，27,320,697,856 logical parameters | repo 7,134,167 downloads/month、4,563 likes |
| Qwen3.8-2.4T-A95B | [`UD-IQ4_XS` @ `567d3e6`](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556/UD-IQ4_XS) | 29 shards，1,702 tensors，2,446,182,725,504 logical parameters | repo 11,107 downloads/month、117 likes；同 repo 还发布 BF16/Q8 与 1--3-bit 档 |
| Kimi K3 | [`UD-Q4_K_XL` @ `a083636`](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b/UD-Q4_K_XL) | 32 shards，2,573 tensors，2,779,483,135,584 logical parameters | repo 457,281 downloads/month、386 likes |
| DeepSeek V4 Pro **0813 sibling** | [`UD-Q4_K_XL` @ `6d05361`](https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF/tree/6d053616d152293da72569f56b861466f10ace7d/UD-Q4_K_XL) | 20 shards，1,894 tensors，1,572,999,528,803 logical parameters；**不是 preview `b5968e9` 的精确量化版** | repo 145,537 downloads/month、65 likes；只作 sibling 格式证据 |
| DeepSeek V4 Flash | [`UD-Q4_K_XL` @ `e3aa0d6`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/tree/e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0/UD-Q4_K_XL) | 5 shards，1,328 tensors，284,334,567,511 logical parameters；base model 精确指向目标 Flash repo | repo 25,368 downloads/month、252 likes |
| GLM-5.3 | [`UD-Q4_K_XL` @ `346b359`](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q4_K_XL) | 11 shards，1,809 tensors，753,864,139,008 logical parameters | repo 562,261 downloads/month、87 likes |
| GLM-5.3-Flash | [`UD-Q4_K_XL` @ `621d456`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/UD-Q4_K_XL) | 6 shards，1,412 tensors，320,759,404,382 logical parameters | repo 779,565 downloads/month、425 likes |

### 高保真 GGUF profile

Q8 名称同样只是制品档位，不是“所有 tensor 都是 Q8”。Unsloth 的 `UD-Q8_K_XL` 会把部分
tensor 保留为 F16/BF16，并继续保留 Kimi/DeepSeek 的 MXFP4 expert；Qwen3.8-2.4T 和 GLM
仓库发布的是普通 `Q8_0` 档。它们为每个模型提供第二个实际部署 profile，也直接反证全局量化枚举。

| 模型 | 固定高保真 profile 与来源 | header 观察 | 主要按角色格式 |
|---|---|---|---|
| Qwen3.5-2B | [`UD-Q8_K_XL` @ `f6d5376`](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-UD-Q8_K_XL.gguf) | 1 shard，320 tensors，1,881,825,088 logical parameters | embedding F16；attention/FFN 按角色混合 F16/Q8_0；GDN projection F16，state/conv/norm F32 |
| Qwen3.8-27B | [`UD-Q8_K_XL` @ `4ca7207`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_XL.gguf) | 1 shard，866 tensors，27,320,697,856 logical parameters | embedding/FFN/GDN 主要 Q8_0；部分 attention、LM head、MTP 保留 BF16；state/conv/norm F32 |
| Qwen3.8-2.4T-A95B | [`Q8_0` @ `567d3e6`](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556/Q8_0) | 56 shards，1,702 tensors，2,446,182,725,504 logical parameters | projection、routed/shared expert、embedding/output 为 Q8_0；router、norm、GDN state/conv 为 F32 |
| Kimi K3 | [`UD-Q8_K_XL` @ `a083636`](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b/UD-Q8_K_XL) | 34 shards，2,573 tensors，2,779,483,135,584 logical parameters | routed expert 仍为 MXFP4；其它大矩阵保留 BF16；KDA state/conv、router、norm 为 F32 |
| DeepSeek V4 Pro **0813 sibling** | [`UD-Q8_K_XL` @ `6d05361`](https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF/tree/6d053616d152293da72569f56b861466f10ace7d/UD-Q8_K_XL) | 20 shards，1,894 tensors，1,572,999,528,803 logical parameters | routed expert 仍为 MXFP4；MLA/index/shared expert/embedding 为 BF16；mHC、norm、index 小 tensor 为 F32 |
| DeepSeek V4 Flash | [`UD-Q8_K_XL` @ `e3aa0d6`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/tree/e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0/UD-Q8_K_XL) | 5 shards，1,328 tensors，284,334,567,511 logical parameters | routed expert 仍为 MXFP4；MLA/index/shared expert/embedding 为 BF16；mHC、norm、index 小 tensor 为 F32 |
| GLM-5.3 | [`Q8_0` @ `346b359`](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771/Q8_0) | 17 shards，1,809 tensors，753,864,139,008 logical parameters | projection、routed/shared expert、embedding/output 为 Q8_0；router、norm、bias、index metadata 为 F32 |
| GLM-5.3-Flash | [`Q8_0` @ `621d456`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/Q8_0) | 8 shards，1,412 tensors，320,759,404,382 logical parameters | projection、routed/shared expert、embedding/output 为 Q8_0；KDA state/conv、mHC base/scale、norm/router 为 F32 |

### GGUF 的真实按角色格式

| Profile | embedding / output | attention / recurrent | FFN / MoE | 不量化或高精度部分 |
|---|---|---|---|---|
| Qwen3.5-2B `UD-Q4_K_XL` | embedding Q6_K；输出与 embedding 共享 | full attention 混合 Q4_K/Q5_K/Q6_K/IQ4_XS；GDN projection 为 F16/Q8_0，conv F32 | dense FFN 混合 Q4_K/Q5_K/Q6_K/IQ4_XS | norm、`A_log`、`dt_bias`、conv 为 F32 |
| Qwen3.8-27B `UD-Q4_K_M` | embedding Q4_K；output Q6_K | attention 混合 Q4_K/Q5_K/Q6_K/Q8_0/IQ4；GDN 混合 Q4--Q6、Q8、IQ4 | FFN 混合 Q3_K/Q4_K/Q5_K/Q6_K/IQ3/IQ4 | norm、recurrent state parameter、conv 为 F32 |
| Qwen3.8-2.4T `UD-IQ4_XS` | embedding/output Q4_K | attention Q5_K + F32；GDN output Q6_K，其余 GDN parameter/projection 与 conv 为 F32 | routed experts 276 个 IQ4_XS tensor + 3 个 Q4_K；shared experts Q5_K/Q6_K；router F32 | norm、router、recurrent state/conv 为 F32 |
| Kimi K3 `UD-Q4_K_XL` | embedding/output Q8_0 | KDA/MLA 大矩阵 Q8_0；KDA state、三路 conv、norm 为 F32 | routed experts 276 个 MXFP4 tensor；shared/dense expert Q8_0；router F32 | AttnRes score、norm、state/conv 与 router 为 F32 |
| DeepSeek V4 Pro-0813 / Flash `UD-Q4_K_XL` | embedding/output Q8_0 | MLA 与 index projection 主要 Q8_0；index/norm/mHC metadata F32 | routed experts MXFP4；shared expert Q8_0；router BF16 | mHC、norm、index 小 tensor 与 bias 为 F32 |
| GLM-5.3 `UD-Q4_K_XL` | embedding/output Q8_0 | MLA/DSA projection Q8_0；index 小 tensor F32/Q8_0 | routed expert 按 projection 混用 Q4_K/Q5_K/Q6_K；shared expert Q8_0；router F32 | norm、router、bias 与 index normalization 为 F32 |
| GLM-5.3-Flash `UD-Q4_K_XL` | embedding/output Q8_0 | KDA/MLA/DSA 大矩阵 Q8_0；state/conv/index metadata F32 | routed expert混用 Q4_K/Q5_K/Q6_K；shared expert Q8_0；router F32 | mHC base/scale、norm、router、state/conv 为 F32；mHC projection Q8_0 |

这张表也解释了接入策略：GGUF profile 应优先由 ggml/llama.cpp 的原生 tensor type、量化
matmul、grouped expert 和 region 承载。把它拆成 HYLE 自造的一个 Q4 decoder + 一个通用 GEMM，
既不能覆盖真实混合格式，也会破坏上游按 tensor role 选择格式的收益。

## 算子、模型位置与量化目标

下表中的“算子”指 HYLE 必须能通过一个成熟 Provider region 或自然 Direct/Graph 组合表达的
语义。Provider 可以融合整个 region；不要求 HYLE 把每一步拆成逐 kernel vtable。

| 算子/region | 哪个模型、在哪里使用 | 官方存储与运行时合同 | 首个量化验收链接/版本 |
|---|---|---|---|
| Embedding、Gather、LM head | 全部八个模型的词表入口/输出；Qwen3.5-2B GGUF 共享 output/embedding | 官方以 BF16 为主；Kimi/DeepSeek/GLM 按各自排除规则保留高精度 | 各模型上表 profile；实际为 Q4/Q6/Q8 等，不能按文件名推断 |
| RMSNorm/LayerNorm、Cast、残差 Add/Multiply、SiLU/SwiGLU/SiTU、稳定函数 | 每层 pre/post norm、attention/FFN gate；Kimi 使用 SiTU，DeepSeek/GLM 含受限 SwiGLU | reduction/activation accumulation 与输出 storage 分开；norm/router/state 常保留 F32/BF16 | 全部 profile；这些小 tensor 多为 F32，主 projection 才量化 |
| Dense/Batch/Strided MatMul | 所有 projection；Qwen3.5-2B/27B dense FFN；每个 attention/linear-attention projection | BF16/FP8/packed FP4 weight 依模型而定，activation 与 accumulation 按 provider 合同 | 全部 profile；Qwen 混合 K/IQ，Kimi/DeepSeek/GLM 多为 Q8 或角色化 expert 格式 |
| GQA、QK norm、RoPE/YaRN、causal mask、Softmax、KV cache | Qwen3.5-2B 6 层、Qwen3.8-27B 16 层、Qwen3.8-2.4T 23 层 full attention | Q/K/V/output weight BF16；KV/cache 采用模型运行 dtype，score/reduction 需合格累加精度 | [Qwen3.5 Q4 XL](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/f6d5376be1edb4d416d56da11e5397a961aca8ae/Qwen3.5-2B-UD-Q4_K_XL.gguf)、[Qwen3.8-27B Q4 M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf)、[2.4T IQ4](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556/UD-IQ4_XS) |
| Causal depthwise Conv1d + Gated DeltaNet | Qwen3.5-2B 18 层、Qwen3.8-27B 48 层、Qwen3.8-2.4T 69 层；kernel size 4 | projection 可为 BF16/量化；conv/state/math 为 F32。prepare 必须完成转换并固定 state layout | 同上三个 Qwen profile；真实 GDN tensor 分别混用 F16/Q8、Q4--Q8/IQ、Q6/F32 |
| KDA：三路 causal conv、gate/decay、recurrent update、state | Kimi K3 69 层；GLM-5.3-Flash 34 层 | 官方 Kimi/GLM Flash 的 `A_log/dt_bias`、conv 与 recurrent state 为 F32；projection 为 BF16/FP8 或 GGUF Q8 | [Kimi K3 Q4 XL](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b/UD-Q4_K_XL)、[GLM Flash Q4 XL](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/UD-Q4_K_XL) |
| MLA：low-rank Q/KV、NoPE/RoPE split、KV state、output gate | Kimi K3 的 24 full-attention 层；DeepSeek Pro/Flash 全层；GLM-5.3 全层；GLM Flash 11 sparse-attention 层 | 官方可能 BF16、FP8 或 packed profile；low-rank norm/state 保留各模型精度 | Kimi/DeepSeek/GLM 四组 profile，具体链接见上表 |
| DSA/Lightning Indexer：index projection、pool/compress、score、stable TopK、Gather | DeepSeek Pro top-1024、Flash top-512；GLM-5.3 top-2048、21 full + 57 shared indexer；GLM Flash top-2048、11 sparse layers | index 大矩阵可 FP8/Q8；norm、APE/gate 小参数与索引为 F32/I32；选择结果不能用浮点 index 代替 | [DeepSeek Flash Q4 XL](https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF/tree/e3aa0d6a5fa4f820d9e132ac1fd1d01e1b2b49e0/UD-Q4_K_XL)、[GLM 5.3 Q4 XL](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771/UD-Q4_K_XL)、[GLM Flash Q4 XL](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9/UD-Q4_K_XL) |
| Compression、CSA/HCA、跨层 index state、ragged tail | DeepSeek Pro/Flash 的 per-layer compression ratio 0/4/128；GLM shared/full indexer；长上下文 attention | 跨 invocation state、pool tail、window overlap 是 region 合同；不能由单次 dense attention 代签 | DeepSeek/GLM profile；其 GGUF header 已保留 compressor/indexer tensor |
| MoE router、score/bias、stable TopK、dispatch/scan/scatter/combine | Qwen 2.4T 512/top-10；Kimi 896/top-16；DeepSeek Pro 384/top-6、Flash 256/top-6；GLM 256/top-8、Flash 288/top-8 | router 常为 F32/BF16；index I32；tie、空/偏斜 expert、shared expert 与 token 重排必须保序 | [Qwen 2.4T IQ4](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556/UD-IQ4_XS)、Kimi/DeepSeek/GLM profile |
| Quantized grouped expert MatMul + activation | 上述全部 MoE 模型的 routed/shared expert | Qwen IQ4_XS/Q4/Q5/Q6；Kimi MXFP4；DeepSeek MXFP4 或官方 packed FP4/FP8；GLM Q4/Q5/Q6 或官方 FP8。转换、scale、activation 与 combine 必须一起计成本 | 各 MoE profile 的精确 tensor type 组合；不得用“任意 Q4”代签 |
| mHC / AttnRes：小矩阵、Sinkhorn/归一化、pre/post combine | DeepSeek Pro/Flash `hc=4`；GLM Flash `hc=4`；Kimi AttnRes block size 12 | base/scale/state 多为 F32，projection 可 BF16/Q8；必须保留跨 region residual identity | Kimi、DeepSeek、GLM Flash profile |
| MTP / NextN carry | Qwen 三个模型各 1 MTP；DeepSeek Pro/Flash 各 1；GLM/GLM Flash 各 1；Kimi K3 明确为 0 | MTP 具有独立 weight、norm 和 carry；普通 next-token logits 不能代签 MTP | 相应 profile；GGUF 是否包含/启用 MTP 必须由 tensor 与执行路径共同证明 |
| FP8 / MXFP4 decode、scale、动态 activation quant、量化 MatMul | Kimi 官方 MXFP4；DeepSeek 官方 packed FP4 + FP8/UE8M0；GLM 官方 FP8 E4M3 block-128 | 必须保留 codes/scales 身份、block/group shape、rounding 与 activation quant；不可先全量解成常驻 F32 再称原格式支持 | 官方 profile 是第一目标；GGUF profile 是第二个实际部署目标，两者分别验收 |
| 资源、准备、两次执行、失败不发布 | 上述每一个 model-region/profile 在 CPU、CUDA、Apple、Vulkan 四路线 | prepare 只匹配规则、转换/pack 并绑定程序；execute 不重新搜索/计时。缓存、workspace、stream/queue/completion 使用路线原生方式 | profile ID 必须进入冷路由条件；同模型可因硬件、驱动/toolkit、Provider 版本选择不同成熟 region |

## 对后续 M3/M4 的约束

1. **profile tuple 是验收最小身份**：
   `(model family, exact model/revision, model region, weight profile, activation/state profile)`。
   平台、硬件/架构、驱动/toolkit 与路线仍是路由条件，不能把其中任何一项藏在 benchmark 名里。
2. **每个模型至少验两套实际 profile**：固定官方 checkpoint profile；上表的固定 GGUF
   profile。DeepSeek V4 Pro preview 暂时只有官方 profile；0813 GGUF 只提供 sibling 接入证据。
3. **Provider-first**：先把 ggml/llama.cpp、oneDNN、cuBLASLt/FlashInfer/Marlin、MLX/Metal、
   IREE/Vulkan 在其真实 tensor type 和 region 域内接完，再统计精确缺口。在所有适用成熟接入
   完成前，不扩大自研算子，不重启冻结实现的性能对比。
4. **四路线语义齐全，承载可以不同**：同一个 profile 的某种 GGUF tensor type 若某路线的成熟
   Provider 不支持，可由该路线另一个成熟 Provider、受控准备期转换或完整 region 承载；转换成本、
   驻留与同步进入验收。不能用 CPU fallback 或另一条路线的 PASS 代签。
5. **直接 BF16 GDN 拒绝不是模型缺口**：Qwen 的正确 consumer 先按源码合同形成 F32
   recurrent inputs/state，再调用合格 region，公开输出回到模型 dtype。M3 必须按这条真实链重测。
6. **旧 64×4 矩阵需要 profile 维度**：同一算子在一个模型内就可能有 Q4/Q5/Q6/Q8/F32
   多个角色。后续矩阵按 model-region/profile 子用例聚合，不能把一个人工 Q4 carrier 扩写为整模
   `NUMERIC_INCOMPATIBLE` 或 `PASS`。
7. **vision 边界显式保留**：Qwen3.5-2B、Qwen3.8-27B、Kimi K3、GLM-5.3-Flash 官方
   checkpoint 中实际存在 vision tensors。当前目标仍是语言主干与本地 MoE；本审计记录这些
   tensors，但不把视觉 encoder 算子偷偷加入本轮完成门，也不宣称整套多模态模型已经完成。

## 路由表达审计与独立重构决定

### 现有合同可以保留的部分

现有 `WorkloadContext` 已经区分 model family/name/revision/configuration、prefill/decode 与
`RegionSemanticId`；`RouteRule` 已能匹配平台、四条 native route、硬件/驱动/toolkit 属性、Provider
版本、operation、有限输入输出 dtype、storage format 与单一量化描述。Graph Provider 在 proposal
阶段可以读取完整 graph，prepare 返回固定的 `ProviderRouteKey`，execute 不重新搜索。这些冷匹配、
热执行边界和四路线原生资源模型继续成立。

### 真实模型已经证明的缺口

1. 模型 identity 没有独立的**权重/部署 profile identity**，无法区分同一 model revision 的官方
   safetensors、Unsloth `UD-Q4_K_XL`、`Q8_0` 或准备期派生制品，也无法把固定 repository revision
   和 manifest digest 放进缓存身份。
2. `RouteQuantizationRule` 仍是单 operation 的一个 mode/group/weight/scale 描述，不能表达一个
   region 同时含 IQ4/Q4/Q5/Q6/Q8/F32，或 MXFP4 expert + BF16 projection + F32 state。
3. 公共 matcher 主要匹配两个输入和一个输出；MoE、KDA/GDN、MLA/DSA、mHC/MTP 都需要按语义端口
   描述 codes、scales、router、state、index、activation、accumulation、layout 和 shape 条件。
4. Provider 虽可临时遍历完整 graph，但没有跨 Provider 稳定的 canonical region signature；因此
   eligibility、诊断和 prepared-cache identity 会各自重复猜测 tensor role。

这些事实满足已关闭 Compute/Graph 路由轮次的 reopen trigger：真实 consumer 需要当前规则不能表达的
模型/量化组合、新冷资格维度和缓存合同。它们不能在 M3 测试中以临时字符串或 provider 私有分支绕过。

### 单独阶段的冻结方向

模型权重审计完成后、恢复 M3 之前，新建一轮 **Model / Quant Profile Routing Refactor**，独立走
`SCOPED -> AUDITED -> CANDIDATE -> CLOSED`。该轮只重构路由表达和迁移现有调用者，不接入新算子，
也不启动冻结自实现的性能比较。其设计边界为：

- `WorkloadContext` 增加 owned profile identity：publisher/repository、固定 revision、variant、
  manifest digest 和格式家族；模型加载仍属于消费者，HYLE 不读取模型目录。
- model/profile identity 负责复现、缓存失效和模型专用规则；Provider eligibility 必须同时匹配实际
  tensor/region signature，不能仅凭模型名或 `Q4` 标签跳过语义检查。
- Direct operation 继续以 `Operation + TensorDesc` 表达自然路径；Graph region 使用版本化
  `RegionSemanticId` 和 canonical 多端口 signature。端口采用模型中立的语义 ID，逐端口记录 dtype、
  `StorageFormatId`、group/block、layout/shape、activation、accumulation 与 mutable state 属性。
- platform/hardware/driver/toolkit/四路线中能在构建期确定的条件先裁剪候选；model/profile/region/shape
  在 prepare 期低成本匹配一次，结果与 provider/software identity 一起形成 prepared-cache key；
  submit 只执行已绑定程序。
- 普通用户选择 CPU/CUDA/Apple/Vulkan 路线并提供真实 workload facts；默认 best-of 规则由 HYLE
  开发期资格证据固定。普通用户不排序 Provider，不提供 timing profile，执行期不重新计时或选赢家。

例如 Qwen3.8-2.4T 的 `UD-IQ4_XS` MoE region 必须同时表达：routed expert 的 IQ4_XS 与少量 Q4_K、
shared expert 的 Q5_K/Q6_K、F32 router、activation/state/accumulation、模型与 profile revision，以及
CUDA SM/driver/toolkit 条件。全局 `quantization=Q4` 无法安全表示该组合。

## 直接来源

- 固定模型配置、模型卡与实现源码索引：
  [llm-moe-model-sources.json](llm-moe-model-sources.json)。
- Qwen 运行时实现：
  [Qwen3.5 dense](https://github.com/huggingface/transformers/blob/c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、
  [Qwen3.5 MoE](https://github.com/huggingface/transformers/blob/c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)。
- Kimi K3：
  [官方语言模型实现](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)。
- DeepSeek V4：
  [Pro reference](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)、
  [numeric kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/kernel.py)。
- GLM：
  [GLM-5.3 DSA](https://github.com/huggingface/transformers/blob/c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)、
  [GLM-5.3-Flash](https://github.com/huggingface/transformers/blob/c1b80cf3ee183f94d8b21dfa7e312d1eea4942ad/src/transformers/models/glm5_next/modeling_glm5_next.py)。
- GGUF 格式说明：[Hugging Face GGUF](https://huggingface.co/docs/hub/gguf)；Unsloth 的动态量化
  本身就是按 tensor role 混合格式，不能把文件标签解释成全模型 dtype：
  [Dynamic 2.0](https://unsloth.ai/blog/dynamic-v2)、
  [Unsloth quant mapping source](https://github.com/unslothai/unsloth/blob/main/unsloth/save.py)。

## 本文没有证明的事项

- header 审计证明的是实际存储和 shape，不证明 HYLE、ggml、llama.cpp 或任一硬件路线已正确执行。
- repo 下载量不证明数值质量；正式验收仍须使用固定输入、独立 oracle、两次执行、资源/失败与三交付。
- 除本地 Qwen3.5-2B 外，不要求在当前机器下载或运行大模型完整权重；应按真实 tensor header 和
  forward 形状构造容量缩小但宽度、dtype、state、codes/scales 不缩水的 region consumer。
- 没有精确 revision 的量化 sibling 不能为目标模型代签；它只能提前验证 loader/provider seam。
