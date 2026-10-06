# CPU 热门量化、MTP 与 DeepSeek 新版缺口清单

2026-10-06；baseline `653a91d1eb28f65dbeda380ebeb0a9c400162f80`，Windows canonical。
这是用户要求的需求盘点，**不创建 goal、不重新打开模块、不授权自研数学、不增加运行 PASS**。
当前状态仍以 [module-status](../../module-status.json) 为准；当前执行资格仍在
[D5 ledger](llm-moe-d5-acceptance-ledger.json)。机器可读逐版本/文件列表见
[本清单 JSON](cpu-popular-quant-mtp-gap-list.json)。

## 结论与统计口径

- **11 种明确缺少公开格式描述和 CPU 映射的 GGUF 物理格式**；应借用 ggml，不是新增11种数学算子。
- 既有 Q3_K/Q4_K/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS/MXFP4 已接，不按每个文件重写实现。
- 八个 Unsloth 仓库的元数据 API 快照共列出 **104 个主模型档位**；与现有22画像的精确
  repository/revision/variant 对账，**94个尚无对应当前资格或需 revision 身份对账**。
  它们不是94个缺少算子，也不是94个已经证明运行不了的模型。GLM Flash的新revision不自动复用旧资格，
  可先核对实际文件/header身份，未变数学不机械重测。
- 实际发布的独立 sidecar 为 **3份**：Qwen3.8-27B Q4_0 MTP、Flash-0731 Q8_0与BF16 DSpark。
  Pro-0813文件列表未见独立sidecar，**不能由此推断权重分片内没有DSpark**。
- 目前**没有证实必须自研新数学原语的项目**。新增模型分支/制品资格仍要完成，不能先承诺100%已覆盖。

以上由固定revision的公开仓库文件列表、当前源码及官方forward得到；没有下载大权重，也没有执行新模型。
新档位的所有tensor header尚未逐一审计，不能从UD-IQ2_M或Q4_XL等文件名推断每个tensor的dtype。
下载量只证明仓库使用规模，不提供每个量化文件的热门排名；本表列公开档位，不捏造逐档排名。
实际元数据响应、官方forward/config和类型定义的SHA及查询脚本保存在
[研究收据](../../../../evidence/research/cpu-popular-quant-mtp-inventory/receipt.json)与
[只读快照](../../../../evidence/research/cpu-popular-quant-mtp-inventory/snapshot.zip)。

## A. 明确的格式/算子接缝缺口

当前公开枚举见 [low_precision.h](../../../../include/hyle/compute/low_precision.h)；
CPU类型映射见 [ggml adapter](../../../../src/compute_provider_ggml/ggml_cpu_provider.cc)。
新增每个物理格式应一次覆盖：block描述/原字节导入、普通quant MatMul、模型需要时的grouped expert
MatMul，以及压缩embedding的Gather/GetRows。不自行反量化再展开全部常驻权重。

| ID | 缺少物理格式 | 目标制品或角色 | 承载 |
|---|---|---|---|
| Q01–Q02 | Q4_0、Q4_1 | Qwen3.5/3.8普通4-bit；Qwen3.8-27B Q4_0 MTP | ggml原版block/MatMul/GetRows |
| Q03 | Q2_K | 多模型UD-Q2_K_XL混合权重；具体tensor需header确认 | ggml |
| Q04–Q06 | IQ2_XXS、IQ2_XS、IQ2_S | Qwen/Kimi/GLM/Flash低内存档；UD-IQ2_M是混合档位名，不是一个独立IQ2_M物理类型 | ggml |
| Q07–Q08 | IQ1_S、IQ1_M | 多个大MoE模型公开的UD-IQ1档 | ggml |
| Q09 | Q1_0 | Qwen3.8-2.4T、Kimi的UD-Q1_0档 | ggml |
| Q10–Q11 | TQ1_0、TQ2_0 | Kimi发布的UD-TQ1_0/UD-TQ2_0档 | ggml |

[已核对的上游类型定义](https://github.com/ggml-org/ggml/blob/ffa4e8b80930029a35991f94e7c8a93cd67730ab/include/ggml.h)
是源码证据，不代替CPU backend实际运行资格。
Q5_0/Q5_1、Q2_0、NVFP4同样未在当前GGUF公开目录中，但这次目标仓库文件名未证明它们是必需物理类型；
放在条件候选，只有实际header出现才纳入，避免扩成全量ggml类型大全。
Q8_1/Q8_K等内部dot-product载体也不因上游enum存在就变成必须公开的权重格式。

## B. 已有格式，缺少具体制品资格

以下按实际仓库文件名归并；完整档位、分片路径与固定revision在JSON中。CPU接线可按物理类型/算子族
批量完成，再由同一metadata驱动consumer展开档位，禁止每个档位单开一轮编译和Provider竞速。

| 模型/新版 | 新增或待对账档位 | 当前边界 |
|---|---|---|
| [Qwen3.5-2B](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF) | Q4_0/1、普通Q3/Q4/Q5_K、Q6_K/Q8_0、IQ4、UD-Q2/Q3/Q5/Q6、UD-IQ2/IQ3、GGUF BF16 | 已验UD-Q4_K_XL与UD-Q8_K_XL；该固定GGUF没有MTP sidecar |
| [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 其余Q4档、IQ1/2/3/4、Q2/Q3、Q5/Q6各档、Q8_0/UD-Q8_K_L、BF16 | 已验UD-Q4_K_M与UD-Q8_K_XL；独立Q4_0 MTP待接 |
| [Qwen3.8-2.4T](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF) | UD-IQ1_S/M、IQ2_XS/XXS、IQ3_XXS、UD-Q1_0、GGUF BF16 | 已验UD-IQ4_XS与Q8_0；不能制造该仓库未发布的Q5/Q6版本 |
| [Kimi K3](https://huggingface.co/unsloth/Kimi-K3-GGUF) | UD-IQ1/2、UD-Q1_0、UD-Q2_K_XL、UD-TQ1_0/TQ2_0 | 已验UD-Q4_K_XL与UD-Q8_K_XL；原模型无MTP |
| [GLM-5.3](https://huggingface.co/unsloth/GLM-5.3-GGUF) | UD-IQ1/2/3/4、UD-Q2/Q3/Q5/Q6、GGUF BF16 | 已验UD-Q4_K_XL与Q8_0；其它档的MTP权重必须按角色复核 |
| [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | 同上，并对新版revision下的Q4/Q8身份对账 | 原621d456资格保留，不自动移给API观测的a38483c |
| [DeepSeek Pro-0813](https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF) | UD-Q4_K_XL、UD-Q8_K_XL，以及官方混合FP4/FP8版本 | 整个0813尚未纳入；不是旧Pro preview的另一个量化标签 |
| [DeepSeek Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) | UD-Q4_K_XL/Q8_K_XL、IQ1/2/3/4、Q2、Q3_K_M/XL，以及官方混合版本 | 新版主干与DSpark均需资格；旧Flash preview两GGUF不能代签 |

Hugging Face是分发平台，不是一种量化格式。本次具体盘点的是上述官方checkpoint和Unsloth GGUF。
AWQ/GPTQ/bitsandbytes NF4等**没有相同的已验覆盖承诺**；若要纳入，先确定这些目标模型实际发布的
对应制品/packing/runtime，单列格式接缝，不把GGUF通过当成它们通过，也不凭“4-bit”合并。

已找到以下实际发布的其它容器/packing候选；这里只登记缺少原制品接入资格，未完成header审计，
不把它们视为已证明缺少成熟CPU数学，也不按搜索排名签热门度：

| 候选族 | 目标来源例子 | 缺少的接缝/资格 |
|---|---|---|
| bitsandbytes NF4 / double quant | [Qwen3.5-2B bnb-4bit](https://huggingface.co/techwithsergiu/Qwen3.5-2B-bnb-4bit) | 非均匀codebook、absmax/二次量化元数据、混合BF16排除项及CPU成熟执行链 |
| AWQ / GPTQ / compressed-tensors W4A16 | [Qwen3.8-27B AWQ](https://huggingface.co/mattbucci/Qwen3.8-27B-AWQ)、[AWQ/GPTQ发布候选](https://huggingface.co/soyrsoyr/Qwen3.8-27B-W4A16-AWQ-GPTQ) | 原packing、group/zero-point/必要permutation、未量化模块、MTP是否保留；数学上的4-bit接口不能代签 |
| MLX int4制品 | [Flash-0731 MLX 4-bit](https://huggingface.co/ToPo-ToPo/DeepSeek-V4-Flash-0731-mlx-4bit) | 特定权重layout/scale/bias和CPU承载需审计；Apple路线提供MLX不等于CPU路线有该制品资格 |

这三族作为 **B09：其它主流量化制品接入审计**，先确认模型revision/实际tensor角色/成熟CPU
carrier，再冻结有限任务。当前104档位统计只计八个已查询的Unsloth GGUF仓库，不包含这三族。

## C. MTP 与 DeepSeek 新版的完整计算缺口

官方确认 [Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813) 和
[Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) 均为取代preview的正式新版，带DSpark。
已取两份固定revision的官方 `inference/model.py`，文件SHA相同；形状和config独立，不能互相代签。

| ID | 待接/待验完整计算 | 包含的实际运算与状态 | 预计成熟承载 |
|---|---|---|---|
| B01 | [Qwen3.8-27B Q4_0 MTP](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main/MTP) | 新编码projection、embedding/hidden merge、norm、decoder、head与carry；旧MTP公式资格不等于sidecar资格 | 原ggml算子/region组合 |
| B02–B03 | Pro-0813、Flash-0731新版主干 | 实际混合codes/scales、FP4专家/FP8 projection、MLA、压缩、hash路由、mHC、缓存状态 | ggml/ATen原程序；已有运算复用 |
| B04 | DSpark输入与target hidden投影 | 按target_layer_ids收集/拼接hidden，main_proj→main_norm；noise block/token embedding；HC扩展 | MatMul/RMSNorm/Gather/Concat/Repeat或原region |
| B05 | DSpark attention与独立cache | target KV prefill/窗口更新、draft block索引、带sink稀疏attention、尾维RoPE及逆RoPE、activation quant和output projection | 成熟attention/cache程序与原组合 |
| B06 | DSpark Markov与confidence head | Markov token embedding→词表投影→logit bias；hidden与Markov embedding拼接；BF16源confidence权重按官方F32计算边界处理 | Gather/MatMul/Add/Concat/Cast |
| B07 | DSpark完整block/head与重复执行 | HC/MoE、最终HC/RMS/head、logits/confidence输出；实际共享权重、独立slot、cache reset/反馈与失败 | 原程序/成熟region，不新增HYLE数学核 |
| B08 | Flash-0731两份DSpark sidecar | [Q8_0](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/main/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf)和[BF16](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/main/dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf)与主制品的版本/共享参数/格式匹配，完整DSpark输出资格 | 原ggml程序 |

DSpark源码位置：
[Pro固定forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/inference/model.py)、
[Flash固定forward](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py)。
普通preview NextN不代替 `DSparkAttention/MarkovHead/ConfidenceHead/Block`。
源码中DSpark权重仍使用`mtp.*` namespace，所以看到mtp前缀不能直接判定是旧NextN。
采样、draft token接受/回滚策略、模型加载与服务仍属于消费者；HYLE负责必要数学/状态/资源边界。

既有Qwen/GLM画像的MTP记录保留；新档位需核对是否有嵌入MTP或独立文件，不能一律写新增算子。
尤其不能因某量化制品缺失MTP权重而临时构造一个“支持MTP”的空目标。

## D. 更新ggml与默认规则的必需尾项

当前锁为 `456172ec733a135778adcd32d00e576a58232e45`。
只读查询得到候选ggml `ffa4e8b80930029a35991f94e7c8a93cd67730ab`、llama.cpp
`d7a695ef679138c13d86359b84c1731d36213d32`；**尚未更新或资格化**。

1. 定向审计旧资源/OOM/threadpool补丁、tensor/plan ABI、CPU基线和AVX2整程序后固定升级候选；
   保留成熟数学，不把补丁移植变成改写原核。
2. ggml源码与锁目前被CPU/CUDA/Apple/Vulkan共用。CPU优先升级需明确构建/ABI边界；不能静默
   更换共享源后继续签旧GPU资格，也不能为独立版本引入未隔离的符号/对象互传。
3. Native187旧WIN相对旧成熟peer。升级时给旧规则绑定peer版本或撤销不再适用的默认优选；
   不为扩大成熟覆盖提前重开所有自研竞争比较。

## E. 性能验收如何简化

用户提出只接成熟第三方，本阶段不自研新算法。这可以免去**逐算子、逐档位的HYLE与Provider赢家竞赛**。
单一成熟承载、没有优化/全局SOTA主张的新增接线，先做P0成本审计，不机械给每个档位开P1。

仍需要的工程检查：

- prepare固定程序和必要packing，replay不重复编译/搜索，不每次把整库专家或全部压缩权重展开；
- 保留原格式、融合、资源复用和完成边界；转换、复制、同步真实计入，不能以provider自身快代签adapter无额外成本；
- 批末在不可变候选上，用少数代表的**完整原程序**检查普通Direct/Graph的接入成本；
  只有当前施工卡明确需要时启动一份有预算的P1，不逐档位反复跑；完全相同身份的证据复用；
- 数值、状态反馈、边界/OOM/并发/close、source/static/shared必需，未运行不写PASS；
- 旧性能失败/失格原样保留，新scope不受其无限纠缠，也不把它们改名为PASS。

因此“只接第三方”意味着无需重新竞争谁的数学核更快，**不意味着可以忽略接线导致的慢、内存膨胀或错误转换**。
新任务应先SCOPED/AUDITED固定上述有限接缝，再批量实现、一次低并行构建/串行验收；
本清单不是新的施工授权或第二份当前模块状态。
