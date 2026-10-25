# 模型、量化 profile 与算子子图

本表是模型与计算能力的地图：模型／量化制品→权重角色与推理主链→基础算子和常见融合→实际后端实现。
正向用于找缺算子、缺融合和漏接线，反向用于判断哪些实现仍有消费者。融合本身也需要精确的格式、数值和资源合同；
图上的 region、批量 capture 和物理 kernel 融合分别记录，不把几种实现形状当成同一种能力。

这是当前模型与量化版本到权重角色、计算子图、Hyle custom descriptor 的现行人读入口。模型范围由[需求目录 D](tasks/compute/llm-moe-provider-demand-matrix.json) 的 `/current_scope` 定义：六个既有模型，加 DeepSeek V4 Pro-0813、Flash-0731 两个正式版，共 **8 个模型**；下表的 **24 个 profile 是本轮已锁基线，不是量化支持范围上限**。旧 `deepseek_v4_pro`、`deepseek_v4_flash` preview 及其量化衍生制品不再属于当前范围；正式版属于单独锁定的身份，不自动扩大到 V4.1。

[33项派生目录的逐合同复核](../evidence/research/inference-selected-profile-contracts-2026-10-25.json)按18类共享物理／数值合同回指真实consumer，区分已有reader、缺冷绑定、未决源解释及原值证据范围，不是整制品支持名单。已补[Q27 GGUF NVFP4冷映射](../evidence/correctness/qwen-gguf-nvfp4-binding-2026-10-25.json)：原type40／64值36字节沿现有Stored投影与Gather路径，不套HF group16数学；本次host元数据检查不代签原payload或整模型。另补[Q27 fidelity GGUF BF16冷映射](../evidence/correctness/qwen-gguf-bf16-binding-2026-10-25.json)，原type30按plain 1值2字节承载，计算路径与精度策略不变。

[反向接线复审](../evidence/research/inference-reverse-binding-review-2026-10-25.json)保留的实际连接包括17源跨度、20混合角色及CT selected bank；CT普通冷入口和有限组合的后续进展见下文。16的[原shard193复核](../evidence/research/quark-q24-excluded-roles-2026-10-25.json)已确认28个excluded普通权重为BF16，可复用现rank2 reader；该局部事实不推广到全部977项或未存header的head。它们不等于缺新算子，也不能全部改写成仅原payload未跑；自然提交前的logits政策输入边界另行设计。

22的[Kimi CT INT4独立源角色组合](../evidence/correctness/kimi-ct-int4-selected-source-2026-10-25.json)已接原Q／shape／BF16 SF合同、两代选中bank和现有grouped Graph。CPU单例通过，检查实际shape正文、旧bank与输出寿命，并以双项输入区分中间BF16权重存储和仅末端舍入。数值为生成fixture；不代表原payload、完整专家FFN或整模型通过。

17的[ModelOpt混合精度有限组合](../evidence/correctness/q24-modelopt-mixed-source-2026-10-25.json)将NVFP4专家down、逐步BF16合并／残差、F32归一化与后续静态FP8投影接在同一Graph；七个物理角色分别保留只读租约。两组生成fixture在CPU通过，保留首次BF16 Fill标量类型错误及修正。仅六个角色有已存原header，layer41 SA137仍缺精确跨度；不据此签原payload、完整FFN／GDN或整模型。

19的[原fused gate/up冷拆分与bank组合](../evidence/correctness/inferact-fused-mxfp4-source-2026-10-25.json)按原4096行的两个2048行半区，分别使用W与F32 SF的真实行步幅，各取前3行、保完整K8192。现有E8M0转换、四平面及global/local map同代发布、BF16-A grouped reader和独立旧结果在CPU通过；不量化A，不解码完整专家库。数值为生成fixture，现GRS/RNE转换不代表producer OAS或原生kernel逐位等价，也不签完整FFN／模型或吞吐。

CT BF16 block128已补[共享原参数冷图片段](../evidence/correctness/ct-block128-supplied-graph-2026-10-25.json)，随后接通[普通冷reader策略、角色与Recipe投影](../evidence/correctness/ct-block128-model-binding-2026-10-25.json)：源配置选独立BF16数学及local NVFP4的BF16预舍入，不按SF形状猜reader；原W／SF直接借入，固定与symbolic token行共用同体。20项Python、两组C++元数据／精度、原4项CPU回归及T1／T3投影通过。仅签普通冷入口及有限Recipe组件；18／21 selected bank、原数值及完整Model／MoE仍分开记录。

目标模型的 Unsloth 量化制品及 Hugging Face 上的主流量化制品也属于后续覆盖范围，不能用“每模型三个profile”排除其他实际发布的格式。新增制品按 D 的 `/current_scope/quantization_scope` 核定真实仓库、固定revision、对应GA base model、逐角色格式与数值合同后扩充本表；尚未核实的格式不写成某个模型已经发布或Hyle已经支持。

来源索引：D 的 `/profile_catalog` 与 `/current_scope/added_releases` 保存身份，`/demand_units` 保存 15 个区域；[需求明细 C](tasks/compute/llm-moe-coverage-matrix.json) 的 `/requirements` 保存 64 条历史需求；[原生映射 P](tasks/compute/llm-moe-provider-map.json) 的 `/requirements` 回指 C 的具体 domain。下面保留这些来源的反向关系，不继承旧状态、通过结论或 Provider 支持声明。

工作顺序是维护总表、核实并移除仅属旧 preview 的非通用实现、再逐量化精度检查融合与接入切割。**历史 64 行不是完整 GA 清单，也不是 24 个 profile 的逐角色量化与融合审查完成清单。** 这里的“保留”只表示当前需求或独立能力仍在、暂不删除；不表示现有 descriptor 已数值等价或已经支持 GA。

## 单一模型工况的数量示例

Qwen3.5-2B官方BF16、全CUDA、单序列R1的已有实际编译报告中，普通版和MTP均使用**17个公开算子族**；展开6种Unary和2种Binary后为**23种数学／支撑操作**，若把View的reshape／permute／slice也分别计数则为25种。BF16权重不表示所有状态与累加均为BF16，GDN等仍保其声明的F32合同。

| 口径 | 普通decode T1 | 普通prefill T15 |
| --- | ---: | ---: |
| 公开算子族／细分操作类型 | 17／23 | 17／23 |
| 逻辑图节点数 | 1870 | 1870 |
| 冷编译执行步骤数 | 1039 | 1105 |
| 按粗算子名称序列归并的多节点region模式 | 14 | 13 |

17族为Gather、Cast、Binary、RmsNorm、View、MatMul、Concat、Copy、CausalConv1D、Unary、Reduce、GatedDeltaNet、RoPE、PagedAppend、PagedAttention、DenseLinearF32和TopK。Unary实际使用Exp／Neg／Rsqrt／Sigmoid／SiLU／Softplus，Binary使用Add／Multiply；23包含布局、搬运及缓存支撑项，不是23种独立CUDA kernel。

普通decode实际选中4类具名专用融合：SiLU×Mul、双BF16 gate／up投影＋SiLU×Mul、保BF16舍入的Cast→SiLU、同输入双BF16投影；另有通用region lowering与其中的原生融合。专用融合是优化，不是功能正确性的必需算子，也不代表所有可做融合已经穷尽。一个region不保证只启动一个kernel，14／13种粗模式也不能解释为14／13种物理融合核。

所采MTP图仍为17族／23种，新增公开类型为0，但正常T2轮次为2115个节点，catchup T2为2117个；状态、接受前缀和供给接线仍须单独实现。以上是**2026-10-22有限R1／T1／T15诊断图的2026-10-23复核**，不是新测速，也不外推到全部上下文、量化与设备。逐项数量、模式、实现ID和原始报告身份见[数量记录](../evidence/research/qwen35-operator-counts-2026-10-23.json)。本表已有映射仍不等于所有模型／制品都有自动生成的完整数量账本。

## 模型与量化身份

`official`、`primary`、`fidelity` 分别对应目录键 `official`、`primary_gguf`、`fidelity_gguf`。身份 pointer 均相对 D；variant 链接指向已锁 revision 的 Hugging Face 制品目录。GA GGUF 的发布者与 `base_model` 已核对，固定revision的主文件解析目录已覆盖Pro两组各20分片／1,894个唯一角色、Flash两组各5分片／1,328个唯一角色。下表列实际混合类型，不由variant名推断，也不是Hyle支持白名单。

| 模型 | Profile | 具体 variant（HF） | 记录的 formats / 未核项 | 身份 pointer（D） |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B | official | [official-bf16](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) | BF16；F32 recurrent | `/profile_catalog/0/official` |
| Qwen3.5-2B | primary | [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae) | Q4_K；Q5_K；Q6_K；Q8_0；IQ4_XS；F16；F32 | `/profile_catalog/0/primary_gguf` |
| Qwen3.5-2B | fidelity | [UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/f6d5376be1edb4d416d56da11e5397a961aca8ae) | Q8_0；F16；F32 | `/profile_catalog/0/fidelity_gguf` |
| Qwen3.8-27B | official | [official-bf16](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) | BF16；F32 recurrent | `/profile_catalog/1/official` |
| Qwen3.8-27B | primary | [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | Q3_K；Q4_K；Q5_K；Q6_K；Q8_0；IQ3；IQ4；BF16；F32 | `/profile_catalog/1/primary_gguf` |
| Qwen3.8-27B | fidelity | [UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4ca720788d1e01f1bff70c033e0d0028fd02e502) | Q8_0；BF16；F32 | `/profile_catalog/1/fidelity_gguf` |
| Qwen3.8-2.4T-A95B | official | [official-bf16](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/tree/207bd685a7e3696cfaff12ded7c6a7ea0f88c996) | BF16；F32 recurrent/router | `/profile_catalog/2/official` |
| Qwen3.8-2.4T-A95B | primary | [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556) | IQ4_XS；Q4_K；Q5_K；Q6_K；F32 | `/profile_catalog/2/primary_gguf` |
| Qwen3.8-2.4T-A95B | fidelity | [Q8_0](https://huggingface.co/unsloth/Qwen3.8-2.4T-A95B-GGUF/tree/567d3e6ac26c5474b18311e619c04350fb9a5556) | Q8_0；F32 | `/profile_catalog/2/fidelity_gguf` |
| Kimi-K3 | official | [official-mxfp4](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06) | MXFP4 packed expert；UE8M0 scale；BF16；F32 | `/profile_catalog/3/official` |
| Kimi-K3 | primary | [UD-Q4_K_XL](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b) | MXFP4；Q8_0；F32 | `/profile_catalog/3/primary_gguf` |
| Kimi-K3 | fidelity | [UD-Q8_K_XL](https://huggingface.co/unsloth/Kimi-K3-GGUF/tree/a0836360ce58dfec088d966a97f2ddc8a606279b) | MXFP4；BF16；F32 | `/profile_catalog/3/fidelity_gguf` |
| GLM-5.3 | official | [official-fp8](https://huggingface.co/zai-org/GLM-5.3/tree/aca966e4e02791568aa6a4ced368624b3d897f42) | FP8 E4M3 block128；BF16；F32 | `/profile_catalog/6/official` |
| GLM-5.3 | primary | [UD-Q4_K_XL](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771) | Q4_K；Q5_K；Q6_K；Q8_0；F32 | `/profile_catalog/6/primary_gguf` |
| GLM-5.3 | fidelity | [Q8_0](https://huggingface.co/unsloth/GLM-5.3-GGUF/tree/346b3591c7f28d1a23716f97a065ecf12ec14771) | Q8_0；F32 | `/profile_catalog/6/fidelity_gguf` |
| GLM-5.3-Flash | official | [official-fp8](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a) | FP8 E4M3 block128；BF16；F32 | `/profile_catalog/7/official` |
| GLM-5.3-Flash | primary | [UD-Q4_K_XL](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9) | Q4_K；Q5_K；Q6_K；Q8_0；F32 | `/profile_catalog/7/primary_gguf` |
| GLM-5.3-Flash | fidelity | [Q8_0](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/621d456e93e926e4b52f85cff5f634358c1828f9) | Q8_0；F32 | `/profile_catalog/7/fidelity_gguf` |
| DeepSeek-V4-Pro-0813 | official | [official-fp4-fp8-dspark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/tree/72e1d3230f6c080a530b0a1d46f8eb4602340597) | packed FP4 expert；FP8 E4M3；UE8M0 scale；BF16；F32；I64 | `/current_scope/added_releases/0/official` |
| DeepSeek-V4-Pro-0813 | primary | [UD-Q4_K_XL](https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF/tree/6d053616d152293da72569f56b861466f10ace7d) | Q8_0 dense／词表；BF16 router；MXFP4 expert；F32；I32 | `/current_scope/added_releases/0/primary_gguf` |
| DeepSeek-V4-Pro-0813 | fidelity | [UD-Q8_K_XL](https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF/tree/6d053616d152293da72569f56b861466f10ace7d) | BF16 dense／词表／router；MXFP4 expert；F32；I32 | `/current_scope/added_releases/0/fidelity_gguf` |
| DeepSeek-V4-Flash-0731 | official | [official-fp4-fp8-dspark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/tree/7872f01b1d1fe23eabc4c98b48bffcef5a386062) | packed FP4 expert；FP8 E4M3；UE8M0 scale；BF16；F32；I64 | `/current_scope/added_releases/1/official` |
| DeepSeek-V4-Flash-0731 | primary | [UD-Q4_K_XL](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/tree/fbbb5b93fb787c21338159b0af3318bb3f4d9768) | Q8_0 dense／词表；BF16 router；MXFP4 expert；F32；I32 | `/current_scope/added_releases/1/primary_gguf` |
| DeepSeek-V4-Flash-0731 | fidelity | [UD-Q8_K_XL](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/tree/fbbb5b93fb787c21338159b0af3318bb3f4d9768) | BF16 dense／词表／router；MXFP4 expert；F32；I32 | `/current_scope/added_releases/1/fidelity_gguf` |

Variant 名称不表示所有权重采用同一精度。例如 Kimi 的 `UD-Q4_K_XL` 目录仍记录 MXFP4/Q8_0/F32；`IQ3`、`IQ4` 仍是宽泛名称，不能补写成已确定 codec。Qwen3.5-2B 两个 GGUF profile 明确排除 R13（MTP）。官方、GGUF、历史人工量化样例不能互相替代。

[本轮MTP制品核对](../evidence/research/mtp-artifact-scope-2026-10-22.json)确认固定官方header中，Qwen3.8-27B有15份BF16 `mtp.*`，Qwen3.8-2.4T-A95B有19份BF16 `mtp.*`，后者含512专家的合并gate/up及down bank、router和shared角色。它们与Qwen3.5的15份角色和几何分别记录；header存在不代表原payload已加载或完整MTP已运行。Kimi固定官方497220份tensor header没有`mtp.*`，config为`num_nextn_predict_layers=0`，不凭量化名称新增MTP任务。量化配置的`ignore/exclude mtp.*`只排除量化，不能单独当作制品不含MTP的证据。

[Q27／Q24角色核对](../evidence/correctness/qwen-original-mtp-input-2026-10-22.json)区分主GGUF与独立C4 sidecar：Q27主UD-Q4_K_M内嵌blk64 NextN，EH／Q／O／FFN为Q6_K、K／V为Q8_0；fidelity的EH／Q／K／V为BF16、O／FFN为Q8_0，norm均F32。Q24主UD-IQ4_XS的attention为Q5_K、专家Q4_K、shared gate／up Q5_K、down Q6_K、EH Q8_0；C5明确选primary54及fidelity48（Q8_0），不沿旧UD-IQ1_M推导。

[Q27完整原NextN与共享词表](../evidence/correctness/mtp-q27-original-nextn-2026-10-23.json)已接主UD-Q4_K_M全部15份blk64原参数及原Q4_K embedding／独立Q6_K head，共17份、2,109,114,368B，固定header差集为空。保持原GGUF压缩格式及F32激活，原norm不再加1；blk64 shared_head_norm优先于global output_norm。原QG物理每head为Q256＋gate256，无热重排；普通Graph复用QKV／GQA／dense FFN构图，原全V248320 head只需一次MatMul。两份词表各冷准备一次，完整原模式不生成占位大bank或整份F32权重副本。

CPU／CUDA各5次Graph保完整cap6 KV、normalized hidden、pending、cursor与position六资源；prefix0／1／2原子发布，各Head续算与旧结果通过。首轮独立检查312,832个投影值（含248,320全部分数），后续7行及实际Top4；参考明确CPU AQ8_K／Q8_0与CUDA AQ8_1舍入及误差界，Q4_K仅核真实embedding解码，不冒称额外MatMul格式已证。共享原输入SHA改为64B块＋128B tail，不复制整份词表做padding。旧Q27／Q24、DSpark与SHA标准向量共9项104次Graph通过；真实target／初始历史仍外部生成，官方BF16数学、其它量化、整模型质量、自然接受率和TPS未代签。

[Q24完整原NextN与独立词表](../evidence/correctness/mtp-q24-original-nextn-2026-10-23.json)已接全部20份blk92原参数及原embedding／head，共22份、17,268,803,584B。原K／V／K norm三份复用，其余19份只按实际第2／29分片header范围获取，未下载完整模型。两份词表虽同Q4_K也保持独立字节与owner；原Q5_K attention、Q8_0 EH、Q4_K专家、Q5_K shared gate／up、Q6_K shared down和F32 norms均保原合同，不额外加gamma1。

完整512域softmax Top10沿实际ID供给E10原压缩bank与I32 map，每槽283,117,568B，双槽；只有10个调度ID读回host，激活和状态保实际producer租约。一次冷图切分生成两份普通Graph，成功后六项状态原子发布。CPU／CUDA新用例各10次Graph，首轮各核435,201个投影值（含全V248320 head）；Q4／Q5参考区分实际M1 MMVQ与未用MMQ的minimum算术，不增加native参考提交。相关8项50次Graph无失败／跳过，实际反馈均为3,321,3,321,328；这是夹具输入的数值／状态结果，不是自然文本质量。声明槽容量不是物理峰；真实target／初始历史、其它量化、自然接受率及TPS未据此完成。旧C5两profile及[单次完整head](../evidence/correctness/mtp-full-packed-head-2026-10-23.json)依据保留。

Pro GA GGUF 固定为 [Unsloth Pro-0813][pro-gguf] `6d053616d152293da72569f56b861466f10ace7d`，两种 variant 各20分片；Flash GA GGUF 固定为 [Unsloth Flash-0731][flash-gguf] `fbbb5b93fb787c21338159b0af3318bb3f4d9768`，各5分片。[Flash GGUF DSpark 说明][flash-dspark]将其列为独立 optional add-on；不能把主 profile 自动解释成启用 DSpark，也不额外增加上述24个主 profile。已存主目录与发布者的MXFP4 passthrough、Q8_0／BF16转换声明相符：primary的普通dense、词表、attention、compressor、indexer与shared FFN为Q8_0，router为BF16；fidelity对应dense角色改为BF16，两者expert仍是MXFP4，另保F32及I32查表。四组目录均无显式MTP／DSpark角色名；独立sidecar仍另列，不据此主目录自动启用DSpark。

[四组GA主目录核定记录](../evidence/research/deepseek-ga-gguf-inventory-2026-10-25.json)保留固定身份、角色计数及来源哈希。Pro的来源为已固定身份的`.scratch/gguf-weight-audit.json`中两组`deepseek_v4_pro_0813_*`完整解析表，保name／shape／codec标签／shard，但该汇总未保原始头字节、tensor offset或逐片header范围。Flash的`.scratch/validation/cpu-quant-mtp-c1/headers.json`保逐片numeric type／offset／split与header extent，原始prefix已由当时的解析脚本删除。解析目录足以核定上述角色类型；缺失原始字节留存与原payload数值／完整模型未验证单列，不将它们写成角色codec未知。

[大packed词表head双行接线](../evidence/correctness/mtp-packed-head-m2-2026-10-23.json)已由Q27／Q24的两个实际normalized-hidden租约构成M2，一次普通MatMul读取同一份完整原Q6_K／Q4_K head；稳定TopK产生的两份独立I32 pending租约直接进入原embedding Gather。四项CPU／CUDA消费者保留原M1接受／状态续算，逐行全核原248320词表，共每项496,640个logits；加3项descriptor边界，共7项38次Graph，0失败／0跳过。

Provider在冷准备时按实际设备与原生选择器查询1～4行的MMVQ／MMQ能力，保canonical布局、K整512、I32逻辑索引／stride和编码范围。MMQ临时空间由调用方workspace承载，按实际stream复用最大需求，不按图节点累加或在热路径分配。该实现及4070上的MMQ数值、缓冲重叠保护、MMVQ尾行读取共7项检查见[当前记录](../evidence/correctness/mtp-packed-mmq-workspace-2026-10-24.json)；真实Orin仍未实测，不由4070结果外推。模型调用方不需知道设备分支，也不展开整份F32权重。

[Q27原target／NextN验证链](../evidence/correctness/mtp-qwen27-target-verifier-2026-10-23.json)另接blk63及global output_norm共12份原角色、282,318,848B；Q／O／gate为Q5_K，K／down／up为Q6_K，V为Q8_0。target与Q27／Q24 NextN共用纯attention构图，实际global norm之后的target tap与NextN独立hnorm／shared_head_norm分开。两个实际target tap经一次原全V248320 M2 head验证，直接保留纠正／bonus的I32租约。接受0保after-current；接受1从first草稿检查点用真实target tap重算proposal行，完整draft六项与second target六项同Head发布，下一轮按已接受target tap续算。

CPU／CUDA新项各23次Graph，分别含10次target、10次NextN和3次verifier；自然试验与实际Top1／Top2结果控制的接受／拒绝分开。每项首轮全核59,392个target投影值、496,640个verifier logits及248,320个NextN head值，取消、旧读者、过期发布与状态预算继续检查；加四项原Q27／Q24回归，共6项84次，0失败／0跳过。本轮为eager修复策略，显式保存完整lag／target hidden／token的延迟补算也合法。layer62输入与有限初始历史仍是fixture，未代签完整模型、任务质量、真实接受率、TPS或物理内存峰。

[Q24原target／NextN验证链](../evidence/correctness/mtp-qwen24-target-verifier-2026-10-23.json)新增blk91与global norm共17原角色、14,032,308,224B，已有原NextN92／词表复用。target的三专家源bank为IQ4_XS，NextN为Q4_K；共用供给按各自真实TensorDesc推导步长、E10输入和槽容量，分别267,388,928／283,117,568B，不把整份512专家bank上传或展开为F32。原64头／4KV头／RoPE64／theta10M、F32 router、softmax Top10和shared sigmoid数学保持；IQ4_XS独立检查非线性LUT、有符号6bit scale及CPU AQ8_K／CUDA MMVQ AQ8_1边界。

Q27／Q24共用attention、M2 head、K1接受／修复／12成员发布与纯数学参考，dense／MoE和原权重身份各自保留。新CPU／CUDA各43次Graph（target20、NextN20、verifier3），逻辑repair1各对应2次实际Graph；首轮各全核178,689个target投影值、496,640个verifier logits和248,320个NextN head值。加Q27 target与原Q24 NextN回归，共6项156次，0失败／0跳过。layer90／62输入与有限初始历史仍是fixture；自然试验与实际Top1／Top2控制分开，不据此宣称整模型、任务质量、真实接受率、TPS或物理内存峰。

## 续算状态与多会话接入

模型接入需声明完整续算状态，缓存目录、预算、COW、租约与退休机制共用；不为每个模型、量化仓库或后端重写缓存管理器。同一状态族可以按真实几何参数化，新的状态数学或存储格式仍须补相应原语，不能由名称推断支持。固定来源与逐项锚点见[八模型状态审查](../evidence/research/model-state-integration-2026-10-21.json)，原制品接线的当前完成与剩余范围以[MTP有限清单](../evidence/research/mtp-artifact-scope-2026-10-23.json)为准。

Kimi 两个所选混合量化制品的[原 router／bias 组件](../evidence/correctness/kimi-original-router-frontier-2026-10-25.json)已通过 CUDA 独立数值检查；两份角色逐字相同后复用同一图。
随后按实际相同 Top16 接入[原专家／latent／共享 FFN 组件](../evidence/correctness/kimi-original-working-set-component-2026-10-25.json)：A32 MXFP4 与 K512 W4A8 两种原编码、直接 gamma norm，以及独立 FP8 投影，两个输入共62阶段数值检查通过。
原七 bank／两代供给保留旧读者、忙槽拒绝和失败绑定后执行合同；独立参考逐阶段使用已核对的实际前驱，输入仍为生成 hidden，不代表整模型质量或吞吐通过。

KDA／NoPE MLA 已提供[外部参数构图入口](../evidence/correctness/kimi-supplied-parameters-2026-10-25.json)，与原测试权重共用数学体；BF16、非方阵 F32／GGUF 的参数边及原 CUDA 合成完整状态旅程通过。
原 layer0 KDA 的14角色、layer3 MLA 的9角色共1,352,207,360B，原header、分片身份与范围已核；
[冷绑定及真实租约检查](../evidence/correctness/kimi-original-attention-binding-2026-10-25.json)已接QKV conv／KV-b布局。
随后[原23角色CUDA组件](../evidence/correctness/kimi-original-attention-state-2026-10-25.json)通过两步数值、完整5资源＋Epoch提交及分叉／旧读者／未发布候选撤销检查。
这里明确采用llama/config的QK epsilon `1e-5`，与已查vLLM reader的`1e-6`分开；上游输入为fixture，跳过KDA0到MLA3之间的层，不代表整模型或其它量化payload已运行。

| 模型／状态族 | 同一快照须保留的内容 | 当前边界 |
| --- | --- | --- |
| 三个Qwen | 各attention层KV、各线性层conv历史／GDN状态、有效长度与位置 | Qwen3.5-2B已有完整分页状态和四会话淘汰重建；Q27／Q24原完整NextN／词表及六资源接受后续算已CPU／CUDA核，另接原末target63／91、全V verifier及target-conditioned缓存修复，12资源同Head发布；完整上游主干、初始历史及其它原制品仍保边界 |
| Kimi-K3 | MLA latent／位置历史、Q/K/V短conv历史、逐通道KDA状态与位置 | 原KDA0／MLA3两步及完整5资源＋Epoch已核，见上文；不能以scalar GDN替代KDA或据此签整模型与其它制品 |
| GLM-5.3 | CKV512＋旋转KPE64、非池化E4／F32scale index缓存、绝对位置／2048选中ID、待追赶journal及hidden／pending | [官方原NextN](../evidence/correctness/mtp-glm-pro-nextn-2026-10-23.json)及[原target77→NextN78](../evidence/correctness/mtp-glm-pro-target-verifier-2026-10-23.json)均已通过CPU／CUDA有限接线，含完整词表、接受修复、index追赶及同Head续算；原B09／B10转换GGUF也已完成。上游选择、layer76输入与有限初始历史仍是fixture |
| GLM-5.3-Flash | KDA／conv、NoPE MLA、index keys／gates、已pool缓存和未pool尾窗、偏移／位置 | [完整原NextN](../evidence/correctness/mtp-glm-original-moe-2026-10-23.json)已接layer45全部1760原角色及全V154880共享embedding／head，共1762份。原input／indexer／MLA／原288域MoE保真实存储边界；两阶段普通Graph与原17状态＋epoch、接受0／1／2、skip／catchup／epoch8 seed及实际ID反馈CPU／CUDA通过。默认ue8m0 cache及target43→44→NextN共34资源＋epoch组合保留；[原全词表R2验证](../evidence/correctness/mtp-glm-target-verifier-2026-10-23.json)已消费实际计算tap，实际I32 pending与接受0／1由同Head发布并续算。[原43→44→NextN](../evidence/correctness/mtp-glm-original-target43-2026-10-23.json)已使用各层独立原参数、原全V154880验证和真实I32 pending，同一34资源＋Epoch接受／续算；layer42输入与有限初始历史仍是fixture，本轮已确认含MTP的GLM量化有限流程已完成，完整主干仍未代签 |
| 两个正式DeepSeek V4 | SWA环、ratio4／128 compressor KV及score窗口、独立BF16-QAT index历史、环位置／压缩相位／长度／映射 | [Pro原58→59→60与三层draft](../evidence/correctness/mtp-dspark-pro-target-verifier-2026-10-23.json)、[Flash原40→41→42与正式FP8 draft](../evidence/correctness/mtp-dspark-formal-flash-2026-10-23.json)及两种选定GGUF sidecar组合已通过CPU／CUDA有限接线，共用全V K5、30成员状态、接受后修复和pending续算。layer57／39上游输入与有限历史仍是fixture，不代表完整模型或全部专家均已执行 |

前缀兼容键要覆盖精确模型／权重revision、adapter参数、实际token与位置、RoPE语义、cache dtype／codec／scale及产生状态的量化计算／舍入合同。权重量化标签、位宽或相同shape不能证明状态可共享。MoE权重驻留不是会话历史；AttnRes／mHC或跨层TopK只在原模型确实跨保存边界携带时计为续算状态。任意token截断、回滚及MTP接受前缀需要对应完整快照或明确计费的重算，不能只截KV。

GLM Flash [完整原NextN接线](../evidence/correctness/mtp-glm-original-moe-2026-10-23.json)与固定官方两份header逐角色差集为空：layer45为1760份、7,493,399,168B，另有全V154880 BF16 embedding／独立head两份，共1762份、10,030,953,088B。角色数包含量化scale，不是算子或kernel数；本轮只实际执行各次路由所选专家，不把完整288域等同288专家均已计算。

普通空／短历史已按[实际闭合池与尾窗](../evidence/correctness/glm-flash-ordinary-startup-2026-10-25.json)构图，
零池不运行评分／TopK；选择状态仍2051项，attention只读有效前缀。原layer43的16角色CPU组件已
从位置0真实续算到frontier5，保13资源同Head、未写区NaN、旧reader及失败发布合同。
MLX／GGUF九种位置已做构图检查，其原参数native短旅程、其它后端和完整层仍未在本批运行。

原indexer的wk／wq_b／compress_gate保BF16，norm／bias／APE／weights_proj精确扩宽；cache保PowerOfTwo／floor1e-4。原MLA与MoE投影保动态A128 Divide／floor1e-10及原E4／F32 block128权重。原kv_b只做一次普通冷Graph准备共享BF16 K／V，退掉原device临时bank；64头／NoPE256／latent512／value256、scale1/16不变。F32 cache载体只写BF16 latent精确扩宽；postnorm用实际F32残差和，另存BF16残差返回，见[原attention](../evidence/correctness/mtp-glm-original-attention-2026-10-23.json)。

原MoE为288域Top8：BF16输入／router→F32 logits→sigmoid；原F32 bias只影响选择，无bias分数除以sum＋1e-20并乘一次2.5。选定scale-in-topk合同，不宣称与所有vendor runner逐bit等价。原gate／up、clip10／SiLU activation、down后乘路由权重、sum8与shared merge保BF16存储边界。每代最多8个原专家的codes／scales原样组成三bank＋global map，CPU两代／CUDA两device与pinned槽每槽201,377,024B；同实际资源、owner及queue，只有8个I32 ID回读以安排供给，激活和状态通过实际租约传递。GLM与DSpark共用纯冷图切分；不复制执行器、逐token构图或展开整份F32专家权重。

[原43→44→NextN的K1续算](../evidence/correctness/mtp-glm-original-target43-2026-10-23.json)已接第43层独立HC、q_a／query norm、MLA／indexer和原288域MoE，向原44的KDA／conv／MoE及model.norm传递真实BF16四流，随后由原全V154880双行head验证并直接保留实际I32 pending租约。共同HC构图与数值参考保持既有torch／unfused顺序；层参数视图、只读owner、一次冷KV准备、有界专家供给及普通两阶段Graph共用，EH只在NextN外层。

CPU／CUDA新两项分别87／87次Graph，合计174次；原44验证链两项回归146次，共4项、320次，0失败／0跳过。完整34资源＋Epoch按接受0／1同Head发布，已接受journal追赶、fresh seed→skip0、旧结果只读及取消继续检查。前批[原target44](../evidence/correctness/mtp-glm-original-target44-2026-10-23.json)、[原全词表验证](../evidence/correctness/mtp-glm-target-verifier-2026-10-23.json)及[完整原NextN](../evidence/correctness/mtp-glm-original-moe-2026-10-23.json)保留历史依据。layer42输入与有限初始历史仍是fixture，不代签完整原模型、任务质量、真实任务自然接受率、TPS或物理显存峰。当前Qwen普通／MTP的单轮与真实两轮四臂性能另见[最新完整对比](../evidence/performance/qwen-mtp-single-and-multiturn-2026-10-24.json)。

[GLM Pro原NextN](../evidence/correctness/mtp-glm-pro-nextn-2026-10-23.json)与Flash共用普通Graph运行函数、冷图切分、几何化MoE供给及StateHead。官方原KV_b整块E4／F32scale先冷解码为BF16，再拆每head448行中的NoPE192／value256；scale不能在非128对齐的head边界重置。只保留两份紧凑只读BF16布局，退掉device原KV临时bank。Pro缓存整行576保CKV512与旋转KPE64，QK576和V512来自同一组选中全局行，再由原value投影输出256；F32载体仅为实际BF16值扩宽。

Pro非池化indexer使用原FP8wk／wq_b、BF16 LayerNorm系数精确扩宽、first64 interleaved RoPE、Hadamard128／BF16／UE8M0及原BF16 weights_proj的F32输出。skip复用原2048个绝对ID，不追加Flash当前尾窗；接受后仅追赶实际journal记录，fresh前index位置必须追平。原MoE保持256域、Top8、route2.5且无Flash clip10，同代三原codes／scales bank＋map每槽302,064,640B。CPU／CUDA各24次Graph，加受影响Flash回归共4项222次，0失败／0跳过；完整原target、自然任务质量／接受率及TPS仍未代签。

## 正式版 DeepSeek：角色、存储与数值边界

[Pro-0813][pro-card] 与 [Flash-0731][flash-card]均明确替代 preview 并附带 DSpark。官方固定 revision 分别为 `72e1d3230f6c080a530b0a1d46f8eb4602340597`、`7872f01b1d1fe23eabc4c98b48bffcef5a386062`。两份 config 的 `quant_method=fp8` 与仓库 FP8 标签都**不表示全 FP8**：`expert_dtype=fp4` 仍在。

下表的存储与计算分开记录：checkpoint 身份、转换后载体、参考实现 parameter dtype 不是同一件事。FP4/FP8 kernel 的 F32 累加由固定源码声明；`F.linear` / `einsum` 的物理执行算法与设备精度仍由具体后端核定。来源为两份固定 [Pro config][pro-config] / [Flash config][flash-config]、[Pro inference][pro-source] / [Flash inference][flash-source]及相同 revision 的 `inference/kernel.py`、`inference/convert.py`。

| 权重角色 / 子图 | Pro-0813 / Flash-0731 形状与模型参数 | 存储 / 转换 | Activation → accumulate → output；待核边界 |
| --- | --- | --- | --- |
| Embedding / LM head | vocab129280；hidden7168 / 4096 | checkpoint BF16；LM head 在参考程序持有 F32 parameter | embedding Gather→BF16；head 以 F32 operands 计算→F32 logits；完整词表不得用缩小 head 代替 |
| Q low-rank、Q/KV、输出 `wo_b`、shared expert 线性 | Q rank1536 / 1024；attention heads128 / 64，D512；输出 groups16 / 8、rank1024；FFN intermediate3072 / 2048 | 默认 E4M3FN `[N,K]`，128×128 UE8M0 scales；BF16/F32 例外逐角色处理 | BF16→动态 E4M3 group128；absmax floor1e-4、向上 power2 scale；FP8 GEMM F32累加→BF16 |
| `wo_a` 输出中间投影 | Pro / Flash 按各自 groups 与 output rank | checkpoint FP8 block128；官方 convert 解码并改为 BF16；参考层明确 BF16 | BF16 grouped `einsum("bsgd,grd->bsgr",...)`→BF16；不得把冷转换遗漏后直接套默认 FP8，也不把其物理累加算法写成已验证 |
| Routed expert gate/up/down | E384 / 256，Top6；`w1/w3: N3072,K7168 / N2048,K4096`，`w2` 方向相反 | packed E2M1 FP4 `[N,K/2]`，I8 容器；UE8M0 scales `[N,K/32]` | BF16→动态 FP8 group128→F32 kernel累加→BF16；gate/up转F32并按10裁剪、SiLU乘法，route weight 在 down 前施加；不能换成 Kimi 的 group32 activation 合同 |
| Router / hash table | E384 / 256、Top6、scale2.5 / 1.5；首3层 hash | `Gate.forward`以F32 activation／weight计算；原checkpoint存储dtype须另核；F32 selection bias；checkpoint API 有 I64，参考 `tid2eid` 为 I32 | F32 logits→sqrt(softplus)；普通层按带 bias 分数选 ID、无 bias 分数归一化；hash 按 token table 原序选 ID；GA `Gate.forward` 的归一化没有 `+1e-20` |
| mHC / norms / attention sink | HC4、Sinkhorn20、eps1e-6；每层 attention/FFN 与输出 head 分开 | HC fn/base/scale、norm weights、sink 为 F32 parameter；不能把整个 checkpoint 标成 F32 | F32 norm/projection/coefficients→HC pre/post；残差与返回的 BF16 边界必须保留；GLM Flash 仍共享 HC 数学需求 |
| CSA/HCA compressor 与缓存 | SWA128；ratio4（重叠）/128；Pro主干31×128+30×4，Flash主干20×128+21×4+2×0；两者另有3个 ratio0 DSpark stages | F32 compressor 计算/增量 KV 与 score state，随后 norm/转换；checkpoint各角色 dtype 与参考F32 parameter需区别 | BF16 hidden→F32 gated pooling/state→BF16；普通 NoPE KV 有 FP8/group64 simulation；ratio4 index compressor另走 Hadamard/FP4-QAT；GLM K4 pooling 不等同此合同 |
| Index projection / score / TopK | H64/D128；Pro Top1024，Flash Top512；causal 压缩候选 | `wq_b` 默认FP8；`weights_proj` BF16；参考 index cache 是 BF16 QAT 值 | RoPE→Hadamard→FP4/group32 QAT；BF16 dot/ReLU/weighted product/head sum边界后选TopK；Hyle物理68B行与paged metadata是接入载体，不是官方checkpoint格式 |
| DSpark `mtp.*` 新角色 | 3 stages；block_size5；Pro target layers58/59/60、Markov rank512；Flash40/41/42、rank256；noise token128799 | stage0 `main_proj/main_norm`；末stage Markov embedding/head、confidence projection与HC head；各角色按正式源 dtype核对 | target hidden拼接/投影→draft block/Markov logits→confidence；另核draft/verify、接受前缀、KV/state生命周期，不能承接旧单层MTP结论 |

Pro 的 `inference/kernel.py` 与旧 `b596…` 字节相同；两份 GA 的 Linear、Gate、Expert、MoE、Indexer、Compressor 主体及 HC pre/post 与各自 preview 保持相同计算。这证明需求仍在，不证明 Hyle 原有实现与 GA 精确等价。两份顶层 config 仍写 `num_nextn_predict_layers=1`，但固定 `inference/config.json` 写 `n_mtp_layers=3`，源码构造三个 `DSparkBlock`；总表保留这项来源差异，不能只取旧字段作结论。固定源码、元数据与差异对照见[本轮来源包](../evidence/research/deepseek-v4-ga-2026-10-19.zip)，其中没有权重payload。

GA typed mHC pre／post／head已用普通Graph复用四端原生实现：[4项检查、32次成功提交](../evidence/correctness/ga-mhc-reference-2026-10-21.json)，无跳过。
B2、S3／S1、HC4、D128各两代，保F32投影后乘RMS因子、Sinkhorn20、comb[source,destination]与三处BF16边界；有限BF16结果与独立参考逐字一致。
实际post租约进入下一代pre，系数／原残差／几何以同一StateVersion消费，旧结果和独立head租约保活。
分支输入是声明的BF16 attention／FFN输出边界，不代表整层或原checkpoint参数执行，也不外推GLM的其它精确角色、Linux、capture或吞吐。

原FP4编码bank现可经原`WeightSlots::upload_slices`接收显式U8物理片段，保原encoded描述及同代I32 global→local映射；修复共同校验误限plain bank的资源缺口。
[6项host与三GPU各2项](../evidence/correctness/encoded-bank-slices-reference-2026-10-21.json)通过：原N2048/K4096、7／256专家、T2／Top6、两槽三代及missing映射失败恢复，保plain NVFP4邻居；没有新数学或codec。
既有`moe_chain`的[完整正式GA FFN消费者](../evidence/correctness/ga-full-ffn-reference-2026-10-21.json)已四端通过：Flash D4096／H2048、E256／Top6、resident7，普通／hash路由各3组输入，共24次成功提交。
原FP4专家与FP8 shared bank同代绑定，非identity全局ID以原I32 Gather映射；核clip／SiLU、down前route权重、独立Invocation重放与旧结果寿命。
首投影、down、路由合并、shared down及最终BF16结果数值完全一致，首投影0ULP，F32阶段最大归一化误差小于1.6e-7；无新增产品数学／调度路径。
这覆盖正式来源的有限合成权重完整FFN，不代表原checkpoint、Pro几何、整模型、Linux或吞吐，也不承诺所有路由均capture。

普通/hash sqrt-softplus 路由现已按 GA 直接除以分数和，移除 carrier 的 epsilon/零和回退并更新 OpID；CUDA Direct/Graph 的极小分数、零和、表序、bounds、独立槽及OOM回归通过，旧内核对照能检出偏差。这不代表完整 GA 模型已执行，也不代表其它后端已有这两个 custom 入口。
`LightningIndexer` 的旧 F32 接口仍不等价于 GA 三处 BF16 舍入；公共 MatMul／Relu／Mul／Reduce 已能表达这套数学。
CUDA 现已为这段公共图接入 dense BF16 scorer，直接消费 K/W 的实际 View，支持 batch 前缀间距、可选 F16 mask 先窄为 BF16、BF16/F32 输出及空结果。
11 项 CUDA 回归通过，含独立舍入见证、双槽、旧结果、8 次强制 capture 与原 packed/paged 路径；小 trace 确认每 key tile 实际执行 Lt＋Relu/Mul/Reduce/publish 五颗 kernel。
见[源码、失败与验证记录](../evidence/correctness/ga-dense-indexer-cuda-2026-10-20.json)。这只补该 CUDA 复合承载；四后端 FP4 RowCache bank 仍与正式 BF16 cache 分开。
后续[四端BF16角色消费者](../evidence/correctness/ga-bf16-indexer-state-reference-2026-10-21.json)已核同一普通图：H64/D128、B2 active515/pitch520、Top512、三处BF16舍入、causal mask、I32 offset／-1映射与同代StateVersion／Head。
17项含12项新增与5项邻居，0跳过；共享Graph共52次成功提交。修TopK(0)、I32 GatherElements及Vulkan BF16 prefix准入；
CPU现ATen会在provider内打包／转换，Vulkan按实际stride在原lowerer整理唯一RHS，不展开共享T/H，均不称零内部复制。
后续[ratio4生产者](../evidence/correctness/ga-ratio4-producer-reference-2026-10-21.json)已把F32窗口／APE／pooling经明确BF16边界、suffix F32 RoPE、Hadamard及group32 FP4-QAT接入实际BF16缓存，沿原StateVersion消费评分／选择。
四端8项含4项新增与4项原RoPE邻居通过、0跳过，共60次提交；涵盖prefill3→p3…7、接受4token续算及一条仅发布状态的闭合emit图。
统一冷预建10代原Invocation保护仍被旧版本保留的输出；不承诺旧ReadLease在世时可重放同一个固定背存producer槽。
修三维RoPE准入、输出实际租约、Apple拥有租约的步长与Vulkan singleton行距；完整投影／attention／GA模型、prefill8、Pro Top1024整链、Linux、质量及吞吐未验证，未执行所选权重payload。

## 扩展量化制品

[量化制品明细](model-quantizations.json)另保存首批33条来源记录，覆盖八个目标模型，包含固定revision、
base-model声明、codec/packing/activation域、配置证据和未知项。除原UD-Q4/UD-Q8外，已发现AWQ产生的
compressed-tensors INT4、GPTQ、FP8、NVFP4/MXFP4、BNB NF4和MLX混合位宽等实际制品。
其中部分配置尚未完整读取，源模型commit通常也未由发布者锁定；这些是需求发现，不是后端支持结论，
也未穷尽HF的主流制品。Unsloth目录已发现的其他GGUF变体同样保留为需求，不把文件名当逐tensor格式。
[原始metadata与配置](../evidence/research/model-quantization-discovery-2026-10-19.zip)不含权重payload。

33条现均已补入 `provider_coverage` 的首轮静态合同核对。记录分别指出条件可达入口、
缺失格式和仍需逐tensor核实的部分，均为 `artifact_execution_verified=false`：CT INT4已补四后端普通图packed线性路径，
GPTQ-v1也已接原四端口decode及有界packed候选；小型算子回归不代表制品导入或整模型执行。
CT grouped另以相同原flat bank、精确reshape和GroupedMatMul接通四后端selected packed路径；
group32/128/512的[数值与资源寿命回归](../evidence/correctness/ct-packed-grouped-space-2026-10-21.json)保留实际scale与W舍入，
通用分片上传与同代W／SF／map原子绑定现已沿三GPU原WeightSlots／Invocation接通，见下表；各制品的精确publisher绑定仍未完成。
ModelOpt 存储 scale 的静态 FP8 也已接四后端普通图参考及有界冷候选，独立 SA/SW 不套动态 block128；
所选 mixed 制品的 payload、其它角色与整体执行仍未验。
MLX 的 affine4/5/6/8 基础入口及`A*W^T`／`A*W`两方向已接四端，物理K打包与分组保持不变。
[同bank head decode／prefill及寿命回归](../evidence/correctness/mlx-weight-orientation-reference-2026-10-21.json)已通过；
六份制品的完整header集合现已核对；GLM31首层两组head原tensor已[四端消费](../evidence/correctness/mlx-head-payload-reference-2026-10-21.json)，保BF16 metadata、独立attention边界和head顺序。
GLM32的[混合MLX普通MoE角色](../evidence/correctness/glm-flash-mixed-mlx-reference-2026-10-21.json)现已四端执行：D4096／H2048、E288／Top8／resident9，routed4／4／5与shared6、g64、原F16 metadata，BF16 A提升F32后保原clip／SiLU、sigmoid校正选择与down后权重。
每代92份分立片段、原双槽三代与同代I32映射，16次成功提交、4次预期失败后恢复；另有CUDA whole／sliced bank对齐回归，共5项通过、无跳过。CUDA原布局改为保256字节bank起点，容量／源字节／记账仍受原约束。
有限合成权重的F32归一化误差小于2.6e-7；不签原制品payload、全部129组bank、完整attention／mHC／模型、Linux或吞吐。Apple的[Graph常量metadata冷准备](../evidence/performance/apple-affine-constant-preparation-reference-2026-10-21.json)已沿现有prepared-cast owner接通，两入口保原F16／BF16逻辑类型、SSA去重、预算与逃逸结果寿命；4项实际检查通过。动态WeightSlots仍沿原Managed路径按真实就绪与代次转换；待执行AsType结点不能代签物理hot分配／dispatch成本，也不要求新增全局地址缓存。
其它角色／层的资源导入、producer舍入和完整执行仍须分别验证；
channel/token FP8 与 ModelOpt 静态 NVFP4 不能套现有 block128 动态合同；前者的固定CT BF16 QDQ参考
已接通四后端普通Graph；Apple/Vulkan本轮补普通Bool转换／选择及所需FP8存储桥，
[20项定向及邻居回归](../evidence/correctness/ct-channel-token-cross-backend-2026-10-20.json)通过，仍未验证整制品或部署端原生FP8点积。
[CPU](../evidence/correctness/mlx-affine-widths-cpu-2026-10-19.zip)、
[CUDA](../evidence/correctness/mlx-affine-widths-cuda-2026-10-19.zip)及
[Apple](../evidence/correctness/mlx-affine-widths-apple-2026-10-19.zip)另记录本轮算子回归。
这些记录没有完成各制品的完整资源导入、prefill/decode 物理融合或整模型运行；33条仍不是完整量化范围。
[来源与源码审查](../evidence/research/model-provider-coverage-2026-10-19.zip)保留配置缺失、root/text_config优先级、
精确packing和舍入未知项，不能把已审过的未知项当作已支持。

[2026-10-21固定来源复核](../evidence/research/fixed-role-binding-2026-10-21.json)更新其中16条：
MLX六制品164个header完整对齐；Kimi INT4实际为BF16 scale，FP8-BLOCK的已读角色是E4＋BF16 block128，
嵌套旧MXFP4配置不能覆盖物理dtype。ModelOpt Kimi注意力存F32四维block128 scale，不能套标量静态SA/SW；
[固定PB_WO部署reader与参考](../evidence/correctness/precise-role-consumers-reference-2026-10-21.json)另确认动态A128，不能从WO名称推出A16。原F32 scale冷导入与所选动态参考已四端执行，
QTensor先将scale转BF16的参考与原生vLLM后端逐位等价仍分开核定；
DeepSeek派生27实际存F32 block128 scale，25/26部分attention/shared才是真E8M0。
K512发布源码还确认另一条BF16 W4A8数学，已补四后端，见下表。Inferact样本则是U8 W＋F32 micro32 scale；
[固定部署reader](../evidence/research/inferact-micro32-contract-2026-10-21.json)确认F32 SF先数值转E8M0、低偶高奇码序与gate/up连续二半，
并支持BF16 A不QDQ的W4A16。Hyle已补该有限域参考；原exporter／OAS生成规则仍未取得，不能按名称套Quark W4A4。
CT BF16 block128普通图、grouped power2 A规则、普通分页E4 KV及E8M0 macroblock128冷导入已四端闭环，见下表；uniform compact MLA E4也已四端接通；具体制品的精确producer／scalar绑定仍分开记录。
Qwen 普通 CT 消费者已接显式 BF16 校准省略与 E4 原 K/V 校准：32个原 BF16 标量已取得，
按 attention 层与 K/V 分别冷绑定，真实行字节参与容量规划。Graph 新行编码、分页 append 与可追加页
共用原权限／退休机制；E4＋MTP 仍冷拒绝。有限生成小模型接线与整制品质量、吞吐分开判断，
下表四后端编码／读取通过不自动代签普通整模型执行。
[本批有限记录](../evidence/correctness/ordinary-e4-kv-2026-10-25.json)包含CPU／CUDA双层实际Model、
Apple原生追加组件和Vulkan严格host编译；未测原27B整模型质量／吞吐。

[compact MLA来源复核](../evidence/research/compact-mla-e4-contract-2026-10-21.json)分开记录producer QDQ与运行时缓存：
固定ModelOpt recipe只量化展开后的K/V，不能推出CKV/KPE物理格式。独立runtime源码及GLM Flash发布卡的TRTLLM选项
确认uniform E4 CKV512／CKV512+KPE64的需求，同一latent缓存共享一个scale；另一动态per128 scale＋BF16 KPE的
DSA布局为528／656字节，另行承载。Hyle uniform compact E4已接四后端原paged/ragged路径，见下表；不由此签收另一DSA格式或精确producer。

[GLM28／29边界复核](../evidence/research/glm-compact-cache-boundary-2026-10-25.json)进一步区分底层能力、引擎政策与制品资格：权重中的静态FP8 cache标签不规定compact字节ABI；28未选KV dtype／backend，29明确选TRTLLM raw E4。调用者可显式选择已有BF16或raw E4共享scale合同，不能借用expanded K/V校准量；scaled DSA528／656是另一个可选格式。精确producer、compact scalar及整制品数值等价仍未核定，不再笼统列为缺少基础compact原语。

## 模型到计算与融合子图

公共项包括布局/物化/转换、embedding/head、norm/逐点/残差、线性/FFN与状态资源复用（D：R01–R03、R15）；量化另查 R14。R 编号只是区域索引，不表示所有 dtype、role 或 fusion 已接入。

| 模型 | 主要计算与待核融合子图 | D / C 索引 |
| --- | --- | --- |
| Qwen3.5-2B / Qwen3.8-27B | Q/K/V→norm/RoPE→GQA→output gate→projection；causal conv→gate/decay→GDN→state；dense FFN；按profile核MTP | R04/R05/R13；F03.4、F04.3/4、F07–F10、F14.3 |
| Qwen3.8-2.4T-A95B | 上述attention/GDN；router→TopK→dispatch→grouped/shared experts→combine；MTP | R04/R05/R10/R11/R13；F07–F10、F12.1/2、F13.1/2、F14.3 |
| Kimi-K3 | KDA三路conv→gate/decay→recurrent；MLA；latent expert→SiTU/再量化→combine；AttnRes | R06/R07/R10–R12/R14；F09、F11–F13、F14.2、F15、F18.3 |
| GLM-5.3 | MLA/YaRN；index projection→score/TopK→sparse gather/attention；跨层选择复用；MoE；MTP index carry | R07–R11/R13/R14；F08.2、F12/13、F14.5、F15、F16.1/2、F18.1/5 |
| GLM-5.3-Flash | KDA/MLA；pooled index+未池化tail；跨层选择复用；MoE；mHC；MTP | R06–R14相应分支；F11–F16、F18.1/5 |
| Pro-0813 / Flash-0731 GA | MLA/CSA/HCA；H64 index；hash / sqrt-softplus router；mHC；FP4 expert+FP8 activation；DSpark | 对照R07–R14、旧F08/F12–F18；DSpark另列新合同 |

## 当前融合与量化接线缺口

以下是2026-10-19首批审查及后续修复，主看Qwen3.5/3.8 dense与MoE FFN，依据现有制品配置、Hyle admission/lowering和[锁定依赖](../../cmake/HyleDependencyLock.json)中的原生实现；未穷尽模型、格式、阶段或设备。表内区分已运行的修复与源码发现；制品发布、普通算子接通和物理融合分别核定。普通 `Unary SiLU→Multiply` 与 `GatedActivation` 是不同数值合同，不能互换来制造融合命中。

| 模型 / 格式域 | 已核源码事实 | 最小下一步 |
| --- | --- | --- |
| 分立expert的有界工作集／同代Graph绑定 | `WeightSlots::upload_slices`在写前校验完整覆盖、dtype、范围及别名，沿原上传和Completion保留不可变bank；`graph::{cuda,apple,vulkan}::bind`一次原子替换同批W／SF／map。三GPU已运行512专家域取4的ModelOpt NVFP4合成角色（N2048/K8192），另核CT asymmetric group128／F16 scale／oddN7显式zero重打包，含多代双槽、旧结果及失败计数。[原资源旅程](../evidence/correctness/expert-working-set-reference-2026-10-21.json)之后，GA原encoded bank和GLM mixed MLX分别由上文有限消费者续验 | 通用bank装配／映射／lease与同代绑定已具备；引擎决定选择、驻留与淘汰政策，不是底座新增功能。精确publisher payload、其它codec的具体旅程、任意大bank、整制品、Linux、质量或吞吐仍不由代表用例代签 |
| Qwen dense；F32/F16/BF16普通SiLU | [真实FFN](../../engine/models/qwen35/main.cc#L181)发普通SiLU再乘法。F32/BF16已有稳定SiLU→Mul，保留[stable-tail、`-Inf→+0`](../../third_party/sglang_activation/activation.cuh#L96)与SiLU输出RNE。CUDA BF16 单向量现另接 gate/up＋SiLU×Mul，直接读两份权重，保留 G/U/A/Y 四处RNE；真实Qwen decode每步24处物理命中 | F16、prefill/其他shape及其他精度的完整投影融合仍未接；F32点积归约树可不同于cuBLASLt，整模型差异与吞吐须单独判断 |
| GDN z门控；BF16来源／F32 SiLU与乘法／BF16结果 | CUDA已接闭合Cast→稳定SiLU→Mul→Cast的typed输入／存储，保F32两处舍入和末BF16 RNE；外部精确metadata View及两Mul顺序可用。13项相关回归及完整64向量通过，当前Qwen每步18处物理命中、少36个转换核；[实际证据](../evidence/performance/typed-gating-suffix-2026-10-20.json) | 中间值逃逸保原分解；内部插入View、非连续／广播及其它dtype保持原路径。未融合前置投影，不代签其它平台、模型或长期吞吐 |
| GDN单快照状态输出 | CUDA对闭合packed→score/state CONT段复用原GDN数学kernel，直接写独立状态输出；真实CONT可发布／多读，保原score拷贝、F32状态与BF16边界。13项CUDA及完整64向量通过；实际每步少18次、合计18MiB D2D，kernel仍596。[源码与实测](../evidence/performance/gdn-direct-state-output-2026-10-20.json) | 仅K=1且当前地址／owner／不重叠检查通过；私有中间值额外读者、其它snapshot或布局保原路径。packed分配保留，未证明吞吐或峰值改善 |
| 同A小行BF16双投影；Qwen GDN a/b等 | CUDA已接两个普通MatMul的多输出冷候选，直接读原BF16权重、各自F32累加与末BF16 RNE，两个结果均可发布／多读；完整FFN优先。5项新CUDA与7项邻居、8项共同CPU回归、完整64向量通过，实际Qwen每步18处命中、614→596核。[源码与实测](../evidence/performance/paired-small-projections-2026-10-20.json) | 目前M1/evenK、转置dense W、等N，cached NVIDIA/MMVF几何且N不超过SM数；其它shape/精度保原叶。两窗波动大，未证明稳定decode提速；不把局部热cache微基准推广成整模型收益 |
| Qwen dense；F16/BF16投影→GatedActivation | 原生gate/up向量融合要求F32 activation/output；Hyle [MatMul输出转换](../../src/compute_provider_ggml_cuda/tensor_program.cc#L239)及[BF16 gate输入扩宽](../../src/compute_provider_ggml_cuda/tensor_program.cc#L659)保留公开舍入边界 | 需要显式保留gate/up舍入的融合变体；普通SiLU recipe还须保留SiLU舍入。不能只删Cast或放宽matcher |
| Q24 plain/BF16与GGUF K/IQ/Q8/MXFP4专家 | [GroupedMatMul](../../include/hyle/compute/model_ops.h)使用 W `[E,N,K]`、F32 A `[T,A,K]`、IDs `[T,S]`→F32 `[T,S,N]`，A=1或S，共享可用真实零slot步幅View；重复ID保留各slot。NVIDIA plain及严格BF16W/F32A小T直接读GPU IDs，按实际元素宽度、pair步幅、grid及原F32计算偏好执行；BF16位扩宽后保F32累加，不留整bank F32副本。奇K／大T保F32 A，BF16仅临时转换单专家N×K。[16项CUDA回归与常量对照](../evidence/performance/bf16-grouped-bank-2026-10-20.json)通过，充分冷准备的原实现未被证明更慢，已确认空间节省；原F32-ID及Q8/MXFP4数学保持 | 大T及其他不合资格域仍有O(T×S×K+T×S×N)排序临时量、IDs读回／同步和原生池空间；严格mixed indexed GLU尚未融合。四后端共享布局／寿命见[原记录](../evidence/correctness/grouped-shared-activation-2026-10-20.json)；K/IQ、其他shape/平台、gate/up舍入与失败slot仍逐域核定，不据此签收整模型MoE或整体零复制 |
| Qwen官方block128 FP8；另有CT channel/token FP8 | [Float8BlockLinear](../../src/compute_provider_cuda/fp8_provider.cc)保精确E4M3/F32 block128合同；CUDA Graph已接同一激活、相同floor/scale rounding的两线性共享量化，保原两个GEMM及各自输出舍入。[16项相关回归与4→3核trace](../evidence/performance/fp8-shared-prologue-2026-10-20.json)通过，含F32、BF16、BF16 power2、捕获、双槽和旧结果，未执行所选FP8整制品 | CT固定BF16逐token QDQ／逐channel权重舍入已沿四后端普通图执行，[20项数值／寿命及邻居回归](../evidence/correctness/ct-channel-token-cross-backend-2026-10-20.json)通过，含三浮点型Bool与Select位保持；不承接Quark其它axis/F32 scale或block128数学。部署端原生FP8点积、更多分支与完整FFN另核；共享量化不等于两个GEMM物理融合，不据此推导当前BF16 Qwen提速或整个图的峰值工作区减半 |
| Kimi Quark attention逐token／逐channel FP8 | 普通图与原有界候选支持F32 SA标量／[M,1]、SW标量／[N]／[1,N]，冷记录SA→SW或SW→SA，保两个F32舍入；直接读原E4码。正式`f_a_proj` M1／M2、N128、K7168的有限人工码／SF消费者及原标量邻居[四端共28项](../evidence/correctness/fp8-outer-scale-reference-2026-10-21.json)通过，含双槽旧结果、动态prologue、发布中间码回退和F32顺序反例 | 所选reader把动态轴映为展平token；本参考明确max×F32(1/448)、strict reciprocal、SW→SA和BF16输出。后续[完整调用代价对照](../evidence/performance/outer-fp8-route-cost-2026-10-21.json)确认CPU／CUDA的3MiB工作区预算可自然选择有界候选；CPU M2变慢，CUDA短窗含每槽GGML迟初始化，故[本批保留默认](../evidence/performance/outer-fp8-submit-tail-2026-10-21.json)，不以少展开推定普遍提速。未签原payload／完整KDA／模型、AMD rcpf与所有GEMM逐位、Linux、质量或吞吐；block128采用下一行的独立规则 |
| GLM Quark block128动态FP8 | dense／grouped显式`Float8ScalePolicy::kMultiplyReciprocal`保F32 max×F32(1/448)、F32倒数、A×倒数三步舍入；原Divide／PowerOfTwo数学不变。四端[24项及CUDA共享量化3项](../evidence/correctness/block128-scale-policy-reference-2026-10-21.json)通过，无跳过；含固定layer45 expert0 gate_proj的M1／N2048／K4096、原E4／F32 block128轴、共享／独立输入、checked失败恢复和旧结果。CUDA同policy共享一次量化，异policy保独立 | 原SF floor域保持；有限人工码／SF不是原payload或完整MTP／模型，也未签Linux、AMD近似倒数／所有GEMM逐位、全F32输入域、质量或吞吐。Apple dense已与grouped统一紧凑SF索引，[8项定向及邻居检查](../evidence/correctness/apple-compact-fp8-scales-reference-2026-10-21.json)通过；显式全权重decode语义保持。物理分配／warm吞吐仍未测 |
| 独立动态FP8输出；GLM默认ue8m0 index cache | `float8_block_quantize(policy,floor)`将dense F32／BF16 `[M,K]`、K整128变为真实E4 codes及F32 `[M,K/128]` scales；沿原三种scale数学，默认Divide／1e-10，GLM显式PowerOfTwo／1e-4。[四端原语检查](../evidence/correctness/fp8-quantize-output-2026-10-23.json)保逐bit、NaN失败恢复、两输出无alias及旧结果。默认cache→query／scorer→TopK→MLA已进入[完整NextN与target43→44状态组合](../evidence/correctness/mtp-encoded-state-2026-10-23.json)，无第二套执行器或持久F32影子缓存 | 量化原语只核有限M2/K256及列明输入；GLM模型组件只核CPU／CUDA和逻辑E4＋F32scale，未签vendor页ABI、完整原模型／自然接受、Linux或吞吐 |
| Grouped FP8 power2 A；E4／F32 macroblock128 W | `grouped_float8_linear(kPowerOfTwo,1e-4F)`沿四端原执行体系保BF16输入、F32 absmax／乘倒数／ceil及原FP8点积；共享A一次量化，CUDA原BF16 colmajor AOT的scale直接用实际列步幅，无额外transpose。[9项新检查及10项邻居](../evidence/correctness/grouped-fp8-power2-reference-2026-10-21.json)通过，含双槽、旧结果、重复／非法IDs和None邻居 | 只签有界BF16参考；CUDA F32 power在prepare拒绝，未签整制品、Linux、质量或吞吐。E8M0 macroblock128冷导入另见下一行；本FP8专家的working-set consumer旅程仍待。Vulkan另修旧None scale常数除法的RN，未改power／normalize |
| E8M0 macroblock128权重冷导入；DeepSeek GA混合制品的attention/shared角色 | `import_float8_block128_weight(E4[N,K],U8[ceil(N/128),K/128],canonical)`沿compute runtime检查host映射、容量和别名，保原E4字节，仅扩宽scale表。4项host单测覆盖全部256码、stride／尾界、写前拒绝及映射错误；四后端各1项原dense power2 Graph检查通过，含CUDA捕获、双槽和旧结果。[实测](../evidence/correctness/e8m0-block128-import-reference-2026-10-21.json)共8项、无跳过 | host导入不暗中上传／发布，不做激活量化或整W浮点展开；SF0=`2^-127`，255保原NaN位。GPU见证仅有限普通scale，不代签特殊码执行、精确制品绑定、分立专家工作集、Linux、质量或吞吐；group32另合同 |
| stored-F32 macroblock128导入；ModelOpt Kimi PB_WO所选动态A128参考 | `import_float8_block128_weight`现也接原F32 scale表；四维singleton轴须显式折为二维，E4码与SF各bit原样保留。复用`Float8BlockLinear(kDivide,1e-4F)`，BF16 A刻意不落E4，反无AQDQ见证可区分；四端各双槽4次提交，CUDA实际capture。[本批14项](../evidence/correctness/precise-role-consumers-reference-2026-10-21.json)含6项新增与8项邻居、无跳过，另补Apple通用MoE | 这里只签所选动态128软件数值／资源参考；PB_WO名字不等于A16。QTensor先转BF16 scale的参考、原生vLLM累加／输出逐位、精确角色／payload／整模型、Linux、质量或吞吐未签；不展开NK浮点bank，也不改旧Float8WeightDecode数学 |
| 普通分页E4 KV；独立固定K/V scale | 原CacheEncode与CPU boxed／Vulkan FullF32／Apple Steel页读取器直接承载E4，并与既有CUDA核对；不展开整套浮点KV。[16项新检查及6项邻居](../evidence/correctness/paged-e4-kv-reference-2026-10-21.json)通过，含精确编码字节、GQA、引用页／尾界、双槽旧结果和Q1 warm/fresh增长 | 普通E4成对／独立CacheEncode→F2→pending页读取另有[4项新增与3项邻居](../evidence/correctness/e4-paired-pending-reference-2026-10-21.json)通过，含CPU Direct／Graph、CUDA Direct／GPU Graph／mixed及Apple／Vulkan native／mixed；其中新E4共9个成功旅程、54轮更新和18个预期失败旅程。修未提交copy槽的生产者退休检查，保原K→V绑定与非阻塞撤回。普通原校准绑定另见[当前E4接线](../evidence/correctness/ordinary-e4-kv-2026-10-25.json)：原BF16标量冷读、逐层K/V解释和物理字节规划已进入实际Model追加。新[Q8／F32整行追加](../evidence/correctness/ordinary-f32-append-2026-10-25.json)复用canonical carrier与原权限；各批原生／CPU范围分别记录，未签compact成对生产者、完整原27B、Linux、质量、吞吐或任意scale跨端逐bit |
| compact MLA E4；共享CKV512／KPE64与V前缀 | `MlaBatchCapacity`分开Q／输出类型和一个共享`CacheStorage`，沿四端原paged/ragged读者补uniform E4；V同根前512项，不展开V或整缓存。CUDA保Q64调度、E4 KV16与F32 scale后BF16 RNE；Apple保F32计算并缩小Q共享存储。[18项新增与10项邻居](../evidence/correctness/compact-mla-e4-reference-2026-10-21.json)通过，含KPE错省略见证、warm/fresh与真实旧结果 | BF16 Q／输出有界参考；CUDA KV增长沿原cold replan，native delegated ragged无page-report不签无重规划。[CPU编码追加组合](../evidence/correctness/compact-mla-e4-producer-2026-10-25.json)另接真实CacheEncode→单bank kMla追加→typed reader：原17→19行、576B stride、COW旧页／结果与NaN尾界通过；该一项不代签其它端。未签mixed DSA656／528、具体制品payload、Linux、质量、吞吐或新capture；独立Q／输出schema也不代表所有执行组合 |
| CT BF16 block128 W／动态group128 A | 普通Gather／View／Reduce／Binary／Cast／Select／MatMul保留原BF16 scale与逐节点舍入；共享A只QDQ每token一次，先选原E4专家W／BF16 SF再解码。[18项新检查及18项邻居](../evidence/correctness/ct-bf16-block128-reference-2026-10-21.json)分窗口通过，含四端双槽、旧结果、非法IDs恢复、tiny；CUDA原捕获与SiLU融合均保留 | 有限源激活、N/K整128、有界E3参考；未签payload／整模型、Linux、质量或吞吐。Vulkan一般非affine batch用原DAG冷拆MatMul再concat结果，仍有Y复制；CT M1保native广播。本BF16 block128专家的working-set旅程仍归交付3，不套F32-scale code-dot |
| FP8存储转换；F32/F16/BF16与E4M3FN/E5M2 | CUDA Native已统一五种浮点Cast并删旧managed分支；无Provider workspace/private result，支持dense、byte-stride、广播与外部capture。CPU BF16 NaN payload按SIMD对齐。[CPU14项与CUDA14项](../evidence/correctness/cuda-native-storage-cast-2026-10-20.json)通过；固定CT rank3 BF16 QDQ各双槽四次逐bit等于独立oracle，CUDA完整链RequireCapture | 8B元素只做descriptor冷准备；未签巨型payload、整制品、原生FP8 GEMM、质量或吞吐。CT逐channel BF16 scale不等于block128；[原生code-dot反例](../evidence/correctness/cuda-fp8-codedot-feasibility-2026-10-20.json)仍禁止把普通F32 MatMul静默替换成另一FP8点积 |
| ModelOpt 存储 scale 的静态 FP8 Linear | plain E4M3 `W[N,K]` 与独立 F32 标量 SA/SW；普通图保 `F32(A)*F32(1/SA)`→SATFINITE E4→F32 code-dot→先 SA 再 SW→末端 Cast。四后端已接精确闭合链的私有有界候选，直接读原 W、不留 NK F32 bank；公开中间值沿原分解。[数值／寿命与普通图红绿](../evidence/correctness/static-fp8-reference-2026-10-20.json) | 只签存储 scale 软件参考，不等同 fake-QDQ、动态 block128、CT token/channel 或原生 scaled_mm；固定 producer/header 研究不证明所选 mixed 制品 payload、KV/conv、整模型、质量或吞吐。保原默认偏好与设备范围。[准备器→角色→绑定→Recipe 冷接线](../evidence/correctness/qwen-static-fp8-cold-chain-2026-10-25.json)现已保原三个 tensor，18项 Python／383元数据断言及两次合成 CPU Graph 通过；该合成检查的CPU实际走分解。随后[原k_proj组件](../evidence/correctness/qwen-static-fp8-original-component-2026-10-25.json)以5,242,888B原E4／SA／SW通过CPU六步分解与CUDA一步packed各两槽、共8192值的解析误差界检查，并保旧输出／权重租约；工作区报告分别21,022,720B／10,240B，不据此推定TPS或物理峰值。所选mixed制品208组中仅85组有完整已存header，本批只检查一组原payload |
| 官方/CT grouped FP8；Q24 Top10及其他合法S | checked align/GEMM已按实际S承载，A=1共享或A=S独立行，不复制activation。两份F32/BF16 AOT在SM89的S3/6/10/16及Top8相邻回归通过，非法ID仍由Completion报告；[证据](../evidence/correctness/cuda-grouped-fp8-slots-2026-10-19.zip)。随后修复有限scale预乘使零权重产生NaN：dot先乘As、再乘Bs，两次F32 RNE后原顺序累加；实际PTX/SASS及[5项CUDA红绿／邻居回归](../evidence/correctness/grouped-fp8-scale-order-2026-10-20.json)通过，原dot、量化器、ABI/geometry和checked align保留。[锁定源码](../../third_party/sglang_activation/fp8/grouped/original_kernels.py#L327)虽有interleaved W13的`FUSE_SWIGLU`，当前两job仍关闭；各制品routed角色、packing及scale域尚须核定 | 核融合分支fast-math、BF16边界与GatedActivation合同，再接W13准备及region matcher；UE8M0与F32 scale等不同域不能混用，不以算子回归或F32累加命名签收普通F32点积等价、制品/整模型、质量、吞吐或其他平台 |
| Q35/Q27 AWQ命名制品、CT INT4 | [ArtifactWeightDecode](../../include/hyle/compute/model_ops.h#L229)提供NF4、legacy AWQ GEMM4与CT symmetric/asymmetric4冷decode。Q27 artifact9的399角色[header](../evidence/correctness/ct-asymmetric-int4-header-2026-10-20.zip)确认group32/BF16 scale/zero沿N。普通同图Decode[/Cast]→精确Transpose View→F32累加MatMul已接四后端packed候选，保三次舍入及逃逸原分解；[CPU三系统与CUDA](../evidence/correctness/ct-packed-cpu-portability-2026-10-20.json)、[Vulkan](../evidence/correctness/ct-packed-vulkan-2026-10-20.json)、[Apple](../evidence/correctness/ct-packed-apple-2026-10-20.json)各有实际算子／寿命回归。CT grouped另沿精确Reshape→GroupedMatMul直接读选中原Q/Z/SF，[四后端软件参考](../evidence/correctness/ct-packed-grouped-space-2026-10-21.json)覆盖group32/128/512及F32 scale，无ENK浮点bank | dense默认性能另核：CPU单向量、Vulkan已测单行BF16优先packed；Apple保原。grouped按空间合同优先、CPU普通默认已验，不据此声称吞吐提升。原flat EN行的zero布局不等于分立expert各自padding，通用working-set桥与oddN zero显式重打包已有上列实测；未验publisher binding／整模型、峰值或tensorcore／完整FFN融合。AWQ命名不能选择legacy packing；独立decode再绑dense constant不能恢复codec身份 |
| Q35/Q27 AutoRound GPTQ-v1 INT4 | `ArtifactWeightDecode{kGptqV1Int4}`接原I32 Q/Z、F16 scale、I32 g_idx；零nibble加一且15→16、F16乘法后转decode／operand，g_idx选组而非重排K。四后端原decode与普通闭合packed候选均已接通，其中packed候选直接读压缩bank、不留额外dense W；全部ID先验证再寻址。[48项数值／寿命及邻居回归](../evidence/correctness/gptq-reference-2026-10-20.json)均通过，无跳过，含CUDA实际捕获；同包保固定producer、Q35完整150组与Q27第二分片73组header | Q27 index共400组，完整模型各分片仍未穷尽；本轮[Q35原制品有限MTP](../evidence/correctness/mtp-qwen35-gptq-target-verifier-2026-10-24.json)与[Q27原制品有限MTP](../evidence/correctness/mtp-qwen27-gptq-target-verifier-2026-10-24.json)均已通过CPU／CUDA，完整上游主干／初始历史仍保fixture边界。新候选不代表默认更快、硬件INT4 GEMM、FFN／专家融合或其它GPTQ变体；独立发布decode仍物化dense W。Kimi名称含GPTQ但实际CT packing不归此合同 |
| GGUF NVFP4；Q27 混合制品 | 独立64值/36字节身份沿原MatMul/GroupedMatMul/Gather接CPU/CUDA/Vulkan/Apple；Apple保F32激活与累加，含共享激活、重复expert IDs及普通Graph的[数值／寿命验证](../evidence/correctness/apple-gguf-nvfp4-2026-10-20.json)。所选Q27固定header的866张量含374 NVFP4、122 Q5_K、9 Q6_K、1 Q8_0和360 F32，embedding为Q6_K、head为Q8_0；[原记录](../evidence/correctness/gguf-nvfp4-2026-10-20.json)保留header及CUDA MMQ修复 | 未读取所选制品权重payload、未运行27B/MTP整模型，也未验证该制品的融合、质量、吞吐或峰值内存；HF W4A4是另一合同 |
| HF MXFP4/NVFP4 | ModelOpt group16原packed W、E4M3 scale和两个F32 global接`Nvfp4Linear`软件参考；四后端直接读压缩bank，保非fast scale先乘和BF16操作数边界。[dense实测](../evidence/correctness/nvfp4-cross-backend-reference-2026-10-20.json)含N16/K2048、非零subnormal及失败后重建Invocation。`GroupedNvfp4Linear`另接末端IDs与每专家WG/IG，按选中专家分别QDQ，支持共享／独立／零slot步幅A、重复IDs，不展开整E×N×K bank；[CPU与三GPU的数值／寿命回归](../evidence/correctness/grouped-nvfp4-reference-2026-10-20.json)通过。QDQ本体分别为2×M×K／2×T×S×K字节，运行时及状态成本另计 | 未运行所选制品payload／整模型或Linux该入口；Q24专家0–7的[锁定header与最小合同](../evidence/research/nvfp4-grouped-expert-contract-2026-10-20.json)仅证明角色形状，未读取数值。硬件native W4A4、Marlin A16、其他producer和CT local-dynamic不能据此签等价，质量／吞吐未验。原[header/生产者研究](../evidence/research/nvfp4-channel-global-contract-2026-10-20.json)保留 |
| CT local-dynamic NVFP4；Q27/Q24 | 独立`LocalNvfp4Linear`／`GroupedLocalNvfp4Linear`复用group16几何，保原WG／IG直接除法及local零规则；`Fp4InputRounding`明确FP4阈值前F32／BF16边界，不套ModelOpt数学。CPU/CUDA caller激活工作区为2×R×K字节（R=M或T×S），Apple/Vulkan复用原native program并另计检查／状态成本，直接读压缩W。[27项回归](../evidence/correctness/ct-local-nvfp4-reference-2026-10-20.json)全部通过、0跳过：Windows CPU8／CUDA6／Vulkan6，macOS M4 Apple7 | [固定header与producer研究](../evidence/research/ct-local-nvfp4-contract-2026-10-20.json)核artifact8前56层的168个NVFP4 MLP角色，末8层MLP为FP8；artifact18所存首分片含1593组完整NVFP4角色，其中layer0为512专家加shared的1539组、layer1另54组。均为header范围，未读权重数值；该CT local数学与512份分立专家tensor的完整工作集旅程、任意大bank／整制品、Linux、硬件FP4等价、质量与吞吐均未签 |
| Quark MXFP4 W4A4；Q24／Kimi K3／GLM Flash | `Mxfp4Linear`／`GroupedMxfp4Linear`真3／4端口已接四后端软件参考，直接读原U8 `W[N,K/2]` 与E8M0 `SF[N,K/32]`；保BF16量化／重构边界、完整ID先验与selected SF校验。共享A1／零slot步幅只QDQ每token一次，CPU/CUDA本体工作区2TK，独立S为2TSK，不展开整专家浮点bank。[37项数值／寿命与旧FP4邻居回归](../evidence/correctness/quark-mxfp4-reference-2026-10-20.json)全部通过，无跳过 | [固定config、6个header与producer](../evidence/research/quark-mxfp4-contract-2026-10-20.json)仅核角色／来源；portable零／极值定义不冒充原HIP／Triton全域等价，SF0非零、SF255失败，合法有限算术溢出仍IEEE。Kimi独立896专家与外侧BF16投影分开，GLM混合MXFP4／FP8block／BF16；Kimi 的[两代 Top16 工作集组件](../evidence/correctness/kimi-mixed-selected-working-set-2026-10-25.json)已用原几何、生成原格式码通过 CUDA 七 bank 供给、SiTU／combine、旧 reader 及失败原子性；所选 publisher payload／整模型、Linux、原生硬件FP4、质量与吞吐仍未签 |
| MXFP4 W4A16；Inferact micro32所选reader | 同一`Mxfp4Linear`／`GroupedMxfp4Linear`显式选`kBFloat16`，四端直接读BF16 A和原U8 W，W逐值乘E8 scale并BF16舍入后再点积，无激活QDQ暂存或整W浮点bank；F32 SF由`encode_e8m0_scales`冷数值转换。[12项新增与14项邻居](../evidence/correctness/mxfp4-weight-only-reference-2026-10-21.json)通过，含tiny、反post-dot溢出见证、checked失败／重建、双槽旧结果及CUDA capture | [固定vLLM／PyTorch／Quark依赖reader](../evidence/research/inferact-micro32-contract-2026-10-21.json)明确所选有限域。原exporter／OAS、全部角色payload、专家working-set旅程、Linux、质量与吞吐仍未签；没有从macro128／mbs-off推导额外端口，也不替换现MLX post-dot合同 |
| Group512 INT4／INT8，逐组BF16累加；Kimi custom K512 | `BlockwiseW4A8Linear`／`GroupedBlockwiseW4A8Linear`沿四后端现执行体系读原I32低位先offset nibble和F32 scale；A每512元素按BF16 absmax×BF16(1/127)、BF16倒数／乘法后向零截断I8，I32点积及两次scale乘法分别转BF16，按K组顺序BF16累加。共享A只量化一次，不展开ENK浮点bank。[10项新路径与14项邻居](../evidence/correctness/blockwise-w4a8-reference-2026-10-21.json)分相关窗口通过，含CUDA真实capture、四布局、旧结果、tiny和失败重建 | 原source的NaN／越界float→I8没有可移植定义；本参考拒绝这些derived码，不造饱和或wrap。合法SF转BF16的0／Inf和后续IEEE结果保留。仅签有界软件数学／寿命，不签所选payload、整制品、Linux、硬件整数GEMM、质量或吞吐；[Kimi 两代 Top16 工作集组件](../evidence/correctness/kimi-mixed-selected-working-set-2026-10-25.json)已以原几何、生成原格式码在 CUDA 通过七 bank／共享 BF16 分支、SiTU／combine、旧 reader 及失败原子性，不代签 publisher payload。不能由CT packing相同套W4A16或GPTQ-v1 |
| MLX混合位宽；Qwen、GLM等 | `MlxQuantized{format,kind,bits,group_size,orientation}`已接affine4/5/6/8，连续U32行末轴`(K/32)*bits`。affine另支持`kWeightAsStored`：物理W[N,K]及K分组不变，A[...,N]→Y[...,K]；[四端同bank两方向](../evidence/correctness/mlx-weight-orientation-reference-2026-10-21.json)已验。Grouped支持`A[T,1,K]`、`A[T,S,K]`及真实零slot步幅／token行距View，IDs决定`[T,S,N]`并保重复槽，[四后端共享输入回归](../evidence/correctness/mlx-grouped-shared-activation-2026-10-20.json)通过。Vulkan mixed affine直接读原类型；CPU/CUDA/Vulkan的[完整普通MoE](../evidence/correctness/mlx-moe-typed-inputs-2026-10-20.json)与[本批Apple旅程](../evidence/correctness/precise-role-consumers-reference-2026-10-21.json)覆盖泛用5/6/4bit、g32、F32 A／BF16 metadata，未冒充GLM整层。六制品164个header已核，GLM Flash全部37338份scale/bias为F16；固定默认MLX-VLM加载保原F16，BF16 A＋F16 metadata按原MLX提升F32 | [GLM32有限角色／工作集](../evidence/correctness/glm-flash-mixed-mlx-reference-2026-10-21.json)已四端执行9个驻留专家的92片段／三代双槽；原37152份payload与129组bank未穷尽；本轮Q32 MTP有限实跑另见[原制品证据](../evidence/correctness/mtp-glm-mlx-target-verifier-2026-10-24.json)。GLM31的156个head-bank是64头投影；[首层embed_q／unembed_out六个原tensor](../evidence/correctness/mlx-head-payload-reference-2026-10-21.json)（7.875MiB）已四端消费，覆盖decode小query Copy、独立attention输入边界、prefill非转置／共享latent和head顺序，Apple补原生非连续只读结果投影。其余层／全部156个bank仍未验。原低类型QMM逐位、特殊short-block分支及嵌套lazy View未签；Vulkan合资格bank已[跨槽复用](../evidence/correctness/vulkan-mlx-constant-bank-reuse-2026-10-20.json)。GLM32的精确routing／shared已由上述有限消费者覆盖；完整mHC、其余payload／整制品、质量、吞吐或峰值仍未签；MXFP8不能替代integer affine8 |

F32 activation/output配合已接plain/GGUF权重，已有原生gate/up→GLU向量融合候选；它不含down projection/residual，也不覆盖任意prefill GEMM。[现有GGUF fixture](../../tests/graph_route_cuda/d5_qwen35_gguf_fragment_test.cc#L158)发布gate/up等全部中间值，[lowering会保护这些输出](../../src/compute_provider_ggml_cuda/ggml_cuda_provider.cc#L544)，因此不能用该case证明中间值已消除。物理命中须另用同数学、仅发布所需末端结果的case核定，保留诊断case与真实输出合同。

九处原生MMVF/MMVQ现将潜在融合的真实读集交给冷分配，防止未融合时可回收的输入覆盖融合仍需读取的值；
执行图和最终地址保护保留。[定向记录](../evidence/performance/qwen35-full-cuda-2026-10-19/README.md#融合读集进入冷分配)
已证明F32 raw普通完整GLU在dense/VIEW/NONE根的6组物理命中，以及公共Graph的内部输入、预算、双槽重绑；
这不补齐上表BF16舍入变体、grouped metadata拓扑或各种量化的实际接线，也未证明吞吐提升。

投影底座已补ggml CUDA rank2/batched矩阵末两轴transpose-dense RHS，并修正fast NVIDIA双BF16输入/F32累加被意外扩成整份F32输入的问题。能力准入与默认优先级分开：早先无条件扩大普通region造成49个投影改路由，因完整prefill退化而[撤回该默认选择](../evidence/performance/qwen35-full-cuda-2026-10-19/README.md#投影接线与输入累加精度)；现在冷发现读取实际叶提案，普通region不吞已知更优叶，真实融合独立提案。rank2能力已开放，默认及GGML400/cuBLAS300同分反例均保留cuBLAS投影；定向回归通过。新增单向量BF16完整FFN是单独融合候选，不改变普通投影叶偏好，也不构成跨平台覆盖声明。

真实Qwen的prefill与decode均已物理命中24个BF16 FFN及18个F32 SSM SiLU→Mul融合。此前prefill的18个F32候选被内部逆reshape切断，现已通过共同lowering的相邻同值折叠恢复；64份向量与旧路径相同。kernel数从原prefill1932/decode1794降为1890/1752，仍不代表完整FFN或decode提速；[实测与未完成项](../evidence/performance/qwen35-full-cuda-2026-10-19/README.md#相邻逆-reshape-与-prefill-融合)区分逻辑候选、物理覆盖与完整step吞吐。

原生CUDA的独立MMVF现可直接接BF16权重／activation／output，限定连续、无batch、偶数K、单向量、F32累加；
读取时精确扩宽，输出显式BF16 RNE，无整份输入转换。两种RHS布局、舍入中点、双槽换址及奇数K/F32原路径
回归通过。它尚无保留投影与SiLU逐节点舍入的完整gate/up epilogue；默认投影仍选cuBLAS。
三种真实尺寸的冷L2权重轮换配对筛选未显示足够的整体收益，不能用接通typed载体宣称模型提速；
[测量与边界](../evidence/performance/qwen35-full-cuda-2026-10-19/README.md#bf16-向量投影的直接承载)单独记录。

CUDA 普通常量表达式现沿真实 immutable roots 在冷准备中执行原生数学，保留逐节点 dtype/RNE；
同一片段多输出共享准备，相同只读 Cast 共享存储。完整 Qwen 的61组 norm scale Cast/Add、18组 dt bias Cast
和18组 A_log Exp/Neg 已移出热执行，并恢复48处已有 RMSNorm×scale 融合；warm decode 实测1528 kernels/步。
随后 Graph 补全 metadata View 的真实常量根、物理绑定与偏移，CUDA 冷准备接通剩余18组 conv weight Cast；
warm decode 实测1528→1510 kernels/步，64份hidden/logits逐byte不变。本轮未另测吞吐，不能由kernel数推导提速。
随后CUDA复用原RMS归约，在读取/写回中合并BF16扩宽/舍入；允许由归约同步和线程归属证明安全的完全同址连续映射，
保持部分重叠、不同映射、scale别名及公开读者保护。完整模型48处weighted和18处unscaled norm真实命中，
warm每步1510→1414→1378 kernels，64份hidden/logits逐byte不变。配对计时尚未证明整体提速。
BF16 GDN L2随后已补square→sum→epsilon→Rsqrt→multiply的五处舍入，保留原SUM_ROWS树、严格sqrt与fast div；
36处真实命中，warm1378→1054 kernels，19项CUDA与64份Qwen向量通过。冷matrix复用另消除prefill首窗72次512KiB D2D复制。
随后Provider按实际边补充13条Cast→RMS→Cast短链，冷scale保持独立；61处weighted及18处unscaled norm均typed读写，
warm1054→1028 kernels。20项CUDA及64份Qwen向量通过，配对计时未观察到本批decode收益。
随后CUDA单向量BF16 FFN补齐完整gate/up＋普通SiLU×Mul，正确识别两投影之间实际的只读权重TRANSPOSE，
不剥离数学节点或改变usecount；24处物理命中使warm1028→980 kernels，28项相关CUDA通过。
T1 workspace每槽2,103,424→2,054,272B，未测总显存峰值。同轮完整decode只观察到79.05→79.92、参考91.51 tokens/s，
仍低12.66%，不把少48个kernel当作稳定提速；F32点积树变化后的独立模型质量仍需单独判断。
显式频率RoPE的CUDA承载随后将位置转换、广播、三角函数与旋转合为单kernel，保留原F32运算及最终类型舍入。
F16/BF16/F32定向GPU对照及原回归合计30项通过；完整BF16 Qwen的64份向量逐byte不变，warm980→812 kernels，
其中BF16/F32独立转换223→199。其他后端仍沿原分解；本批不改变HF与当前RoPE的舍入差异。
随后F32首标量Divide／Rsqrt用真实只读零步幅View承载，不物化重复numerator；L2按实际私有读者融合最终扩宽，
保留五处BF16舍入与原归约树。32项相关CUDA及64份模型向量通过，warm812→740 kernels、转换199→163、Repeat36→0；
同窗warm98.30→98.50 tokens/s，尚未证明稳定提速。其他shape/精度的完整投影融合及剩余转换仍未完成；
普通二元kernel的广播支持不能推导旧bias融合的布局资格。随后CUDA九处旧MMVF/MMVQ融合已共用实际
bias/gate/scale读取布局、elided来源及restrict别名资格，并把量化尾行界改为真实N；35项相关CUDA通过，
旧实现的3个反例均可复现，64份模型向量保持。随后F16融合显式传原MatMul精度，OAI alpha/limit完整传至
普通、量化和MoE融合；38项相关CUDA与64份模型向量通过，小trace三条OAI路线各命中4次。未重测本批吞吐。
随后 GDN V/beta 已接通 BF16 typed load，原 F32 数学、state 与舍入边界保持。Provider 冷规划同时修复
跨输入前沿的 Cast 分段和互相占用候选问题，保留已有 RMS 融合；完整 Qwen 的18层实际命中，warm740→704 kernels，
转换163→127（73次扩宽、54次窄化）。3项CUDA含10个内部case检查实际融合选择与数值，完整64份向量逐byte相同。
该GDN窗口warm100.03→101.46 tokens/s，参考106.23，仍低4.50%；小窗口、样本重叠，不作为稳定收益或其他模型覆盖证明。
随后 CUDA SSM conv 已接通私有 BF16 sample 的真实 time/channel/batch strides；公开 F32 Conv、F32权重/MAC/输出、
输出 BF16 RNE、普通稳定 SiLU 与 history Copy 保持。共同长核补末块加载界，冷支持拒绝未实现 K 和非法跨度/geometry。
6项 CUDA（含原GDN邻居）零skip、完整64份向量逐byte相同；真实18层 typed conv命中，warm704→668核，
转换127→109（55扩宽＋54窄化）、18次F32实体转置消失。本窗warm101.6379→100.4901 tokens/s，回退1.13%，
不将更少物化或更小workspace写成速度成功；[证据与计时边界](../evidence/performance/qwen35-full-cuda-2026-10-19/README.md#conv-typed-load-与真实布局)。
随后私有Cast(F32→BF16)＋普通稳定SiLU已跨叶偏好接成原生单kernel，保两次BF16 RNE及未发布Cast的真实闭包；不修改原Conv数学或引擎。
5项CUDA（全BF16位型、F32 RNE边界、发布/读者/View/control保护与重绑）及完整64份向量通过；实际18处命中，668→650核、转换109→91。
同窗warm101.3918→101.5118、参考105.3241，差3.62%；完整decode反慢0.17%，不作稳定提速或其他模型覆盖证明。
[本轮实测](../evidence/performance/qwen35-full-cuda-2026-10-19/README.md#常量表达式冷准备与-rmsnorm-融合)不作为其他模型/量化/平台的覆盖证明。

## 64 条历史需求的反向索引

以下恰好逐行覆盖 C 的64个ID。`Q35/Q27/Q24/K/G/GF` 对应上文六模型；“六模型”即这六者。六模型列是 C 的 `models` 去重求交，保留其历史指向；它不是对当前所选每个量化profile的实际tensor存在声明。尤其旧 Q4-S128、Q4-A64、Q8-B32、人工FP8样例，不自动成为24个profile的格式身份。

GA列依据当前固定源码补反向需要；F14.4旧单层MTP与DSpark分开。精确分支不相同或来源尚不足时直接写待核，不由family范围推导独占或删除。ID链接中的 `/requirements/N` 是 C 的 JSON pointer，不引入旧 route status。

| C ID / pointer | 具体数值、形状或子图边界 | 六模型历史指向 | GA Pro / Flash 现行反向关系 |
| --- | --- | --- | --- |
| [F01.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/0) | View/stride/alias、边界与空tail | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F01.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/1) | Copy/Concat；显式连续物化 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F01.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/2) | Cast与索引转换；IEEE舍入边界 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F02.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/3) | Embedding/轴向Gather、重复及越界ID | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F02.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/4) | Scatter赋值/加法；重复目的地与F32端口 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F02.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/5) | causal/ragged/全mask；动态offset | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F02.4](tasks/compute/llm-moe-coverage-matrix.json#/requirements/6) | 计数、exclusive scan、稳定dispatch逆映射 | Q24、K、G、GF | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F03.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/7) | 逐点/残差/广播及strided tail | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F03.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/8) | 稳定exp/log/silu/softplus/sqrt/Clamp | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F03.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/9) | gate/up→SwiGLU→down；保留转换 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F03.4](tasks/compute/llm-moe-coverage-matrix.json#/requirements/10) | Qwen head-interleaved sigmoid output gate | Q35、Q27、Q24 | —；按六模型对应分支保留 |
| [F04.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/11) | RMSNorm末轴F32归约、eps/gamma | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F04.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/12) | LayerNorm mean/variance、gamma/beta | 六模型 | 未见GA独立LayerNorm需求；公共能力保留，不由历史全模型映射补算子 |
| [F04.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/13) | L2归一化epsilon在sqrt内 | Q35、Q27、Q24、K、GF | —；按六模型对应分支保留 |
| [F04.4](tasks/compute/llm-moe-coverage-matrix.json#/requirements/14) | norm→gamma/输出舍入→swish gate | Q35、Q27、Q24、K、GF | —；按六模型对应分支保留 |
| [F04.5](tasks/compute/llm-moe-coverage-matrix.json#/requirements/15) | Reduce/Softmax的F32归约与全mask | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F04.6](tasks/compute/llm-moe-coverage-matrix.json#/requirements/16) | TopK真实k/tie/NaN与ID顺序 | Q24、K、G、GF | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F05.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/17) | M=1/17/128、真实N/K、累加类型 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F05.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/18) | batched/broadcast QK/PV及leading strides | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F05.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/19) | checkpoint转置/slice、GEMV尾部 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F05.4](tasks/compute/llm-moe-coverage-matrix.json#/requirements/20) | gate/up→activation→down→residual、活中间值 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F06.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/21) | 历史本地132张量signed Q4/group128 | Q35（历史本地派生） | 不是GA官方格式；不得把历史派生样例冒作GA/GGUF |
| [F06.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/22) | unsigned affine Q4/group64、scale+bias | 六模型（历史载体，选定profile归属待核） | 不是GA官方格式；GA GGUF逐role codec待header核 |
| [F06.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/23) | 历史本地Q8/block32 W8A16，无A8替代 | Q35（历史本地派生） | 不是GA官方格式；GA GGUF逐role codec待header核 |
| [F07.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/24) | Qwen真实GQA/MQA ratio、head256 | Q35、Q27、Q24 | —；GA使用自身D512/MLA/CSA，不套Qwen head256合同 |
| [F07.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/25) | KV append/read、page ID、valid length | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F07.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/26) | 完整Qwen gated attention及旧state | Q35、Q27、Q24 | —；GA attention链另按sink/HC/compression核 |
| [F08.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/27) | Qwen partial/interleaved RoPE与非零offset | Q35、Q27、Q24 | —；GA RoPE/YaRN按F08.2与正式源 |
| [F08.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/28) | YaRN/NoPE、原context与参数 | G、GF | 需；GA YaRN/rope参数及NoPE转换按两份config |
| [F08.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/29) | normalized Hadamard、符号/layout及1/sqrt(D) | — | 需；GA index Hadamard仍在；通用Hadamard不删 |
| [F09.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/30) | causal depthwise conv K4、真实QKV宽 | Q35、Q27、Q24、K、GF | —；此行为Qwen/KDA卷积合同 |
| [F09.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/31) | causal conv update、非零history/reset | Q35、Q27、Q24、K、GF | —；此行为Qwen/KDA卷积合同 |
| [F10.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/32) | scalar-decay GDN chunk、真实K/V比与state | Q35、Q27、Q24、K、GF | —；K/GF在C的宽指向不表示其KDA等同scalar GDN |
| [F10.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/33) | scalar-decay GDN recurrent、state布局 | Q35、Q27、Q24、K、GF | —；K/GF的实际KDA另查F11 |
| [F11.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/34) | KDA逐channel decay/chunk、GLM下界-5 | K、GF | —；K/GF需求保留 |
| [F11.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/35) | KDA recurrent、tail/reset/双state | K、GF | —；K/GF需求保留 |
| [F12.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/36) | 按模型score函数；bias仅用于selection | Q24、K、G、GF | 需；E384/256、Top6、sqrtsoftplus、bias只选ID；保留router并核数值 |
| [F12.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/37) | 真实E/TopK/group、原分数归一化 | Q24、K、G、GF | 需；scale2.5/1.5；GA无epsilon的归一化/tie需精确对照 |
| [F12.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/38) | 早期hash层token table→原序expert IDs | — | 需；首3层hash；I64 checkpoint/I32参考表；不能以score TopK替代 |
| [F13.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/39) | dispatch稳定映射、空expert、grouped GEMM | Q24、K、G、GF | 需；E384/256与Top6；各expert dtype另核 |
| [F13.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/40) | 路由加权/共享expert、重复目的地累加 | Q24、K、G、GF | 需；GA route weight在down前；F32 sum→BF16边界单列 |
| [F13.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/41) | Kimi latent3584/expert3072、SiTU softcaps | K | —；不能把Kimi latent/SiTU当GA FFN |
| [F14.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/42) | mHC4、Sinkhorn20、F32系数及pre/post/head | GF | 需；GA HC4/Sinkhorn20，与GF共享数学但逐dtype核 |
| [F14.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/43) | Kimi block12 AttnRes、prior blockstate | K | —；Kimi AttnRes独立合同保留 |
| [F14.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/44) | Qwen MTP norm/concat/full attention | Q35、Q27、Q24 | —；Qwen MTP按各profile，不能转签DSpark |
| [F14.4](tasks/compute/llm-moe-coverage-matrix.json#/requirements/45) | 旧DeepSeek单nextn层e/h投影与HC head | — | 待DSpark对照：旧单层MTPBlock已替换；不继承旧合同，也非可删公共能力 |
| [F14.5](tasks/compute/llm-moe-coverage-matrix.json#/requirements/46) | GLM MTP BF16角色与跨层index carry | G、GF | —；GLM index carry不等同DSpark target hidden |
| [F15.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/47) | MLA low-rank/absorbed投影、NoPE/RoPE/V宽 | K、G、GF | 需；GA q-rank1536/1024、D512与output组投影，按各角色核 |
| [F15.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/48) | 模型各自gated MLA/NoPE分支 | K、G、GF | GA自身sink/HC边界另核；不继承Kimi gate或GLM NoPE分支 |
| [F16.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/49) | learned index score→TopK→sparse attention | G、GF | 需；GA H64/D128、Top1024/512、BF16逐步舍入；GLM是H32合同 |
| [F16.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/50) | GLM跨层index选择复用/versioned state | G、GF | —；DSpark取target hidden不能据此宣称同一index carry |
| [F16.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/51) | GLM Flash pool4、learned gate与unpooled tail | GF | —；GA ratio4 overlap不是GLM pooled/tail合同 |
| [F17.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/52) | CSA/HCA learned KV/gate、ratio4/128 | — | 需；GA learned compressor仍在，ratio4/128；GGUF角色另核 |
| [F17.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/53) | SWA128、ratio4 overlap及causal压缩位置 | — | 需；SWA128与重叠窗口/causal位置保留 |
| [F17.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/54) | compression增量state、ratio±1边界/reset | — | 需；prefill/chunk/decode state与边界；不以旧回执签GA |
| [F18.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/55) | E4M3FN weight128x128/A128、F32 scales | Q35、G、GF | GA官方是UE8M0分支，非本行F32-scale身份；GLM及旧派生另核 |
| [F18.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/56) | FP8 UE8M0 power2、group/饱和/舍入 | — | 需；GA dense/shared FP8与UE8M0，不表示全FP8模型 |
| [F18.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/57) | Kimi W4A8：weight32/A32 UE8M0、SiTU再量化 | K | —；Kimi A32数值域保留，不以GA A128替代 |
| [F18.4](tasks/compute/llm-moe-coverage-matrix.json#/requirements/58) | DeepSeek E2M1 weight32、dynamic FP8 A128 | — | 需；两GA官方仍FP4/A128；GroupedFP4保留，GGUF不能按名称套格式 |
| [F18.5](tasks/compute/llm-moe-coverage-matrix.json#/requirements/59) | 按源区分KV/index codec、FP4 QAT与score | G、GF | 需；GA FP4-QAT index仍在；物理68B Hyle载体/GLM codec分开 |
| [F19.1](tasks/compute/llm-moe-coverage-matrix.json#/requirements/60) | 冷shape/参数绑定、动态position/valid length | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F19.2](tasks/compute/llm-moe-coverage-matrix.json#/requirements/61) | state端口/读写范围、独立slot与旧结果 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F19.3](tasks/compute/llm-moe-coverage-matrix.json#/requirements/62) | immutable weights、packing/cache/workspace复用 | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |
| [F19.4](tasks/compute/llm-moe-coverage-matrix.json#/requirements/63) | Direct/Graph交付、OOM/accepted-failure/close | 六模型 | 需；正式版公共子图，实际role/dtype与逐profile接线另核 |

DSpark 新合同不塞入旧64行：三个stage的main-hidden抽取/concat、draft attention、Markov/confidence、proposal/verify/accepted-prefix及可选GGUF sidecar按实际角色分别记录；顶层nextn字段与inference构造差异仍保留。64行中标“需”只是现行需求存在，不是完整精度覆盖、实际kernel融合或端到端收益声明。

GLM接受状态分开保留MLA与indexer位置、BF16 raw key／score尾窗、E4 pool／F32 scale、未追赶接受记录和完整I32 selections；skip保旧indexer租约，fresh前仅追赶已接受记录。target43返回四stream租约，target44执行KDA／conv与末归一化，正常下一轮seed来自NextN draft-extend末接受位置，不借用target43 IDs。mHC不是跨token递归状态。当前原43／44层尾与NextN的34资源＋Epoch组合及上游fixture边界见[上方状态表](#续算状态与多会话接入)和[本轮证据](../evidence/correctness/mtp-glm-original-target43-2026-10-23.json)。

DSpark三stage仅把已验证main_x写入各自128行、584B编码ring；五行proposal KV保持临时，不能作为接受历史提交。实际所选token驱动后续Markov embedding／logits及对应confidence。GGUF target层号[41,42,43]对应官方[40,41,42]；GA raw F32 confidence与sidecar sigmoid概率分列。正式Flash原BF16全V embedding／head通过相同只读owner／lease借入draft，无F32大权重副本；原34份norm／sink／HC保F32，六份24路HC保20轮Sinkhorn，末head另走4路门控。原9份shared FFN与12份attention保canonical Q8_0；输出A为[8,1024,4096]独立grouped bank，保原query norm／RoPE／sink、clip／SwiGLU及BF16边界。此前[attention接线](../evidence/correctness/mtp-dspark-original-attention-2026-10-23.json)之后，[原专家供给](../evidence/correctness/mtp-dspark-original-experts-2026-10-23.json)已补齐15份router／bias／expert角色，合全部81份sidecar及两份正式BF16词表共83角色；真实256域ID、有界MXFP4供给与完整draft已有CPU／CUDA检查。现已接[正式原40→41→42 target及K5全词表验证](../evidence/correctness/mtp-dspark-target-verifier-2026-10-23.json)：保4／128／4压缩、独立末端HC／norm、实际main_x追赶及30成员原子接受／续算。CUDA FP4以原量化码的小块无损BF16展开承载F32内积，修复原FP8内积数值偏差，不放宽误差门或展开整份权重。GGUF MXFP4的Q8_1激活与官方E4-A128合同保持区分；layer39／初始历史仍有边界，有限自然trial和控制前缀不代表整模型质量、接受率或吞吐。

原数值与故障依据保留在[GLM原角色／FP8调查](../evidence/correctness/glm-mtp-state-2026-10-22.json)、[target与main角色](../evidence/correctness/mtp-target-state-2026-10-22.json)、[DSpark组合数值调查](../evidence/correctness/dspark-mtp-composition-2026-10-22.json)及[自然旅程](../evidence/correctness/mtp-natural-journey-2026-10-22.json)。CPU mHC旧constructor与AVX2 reader不匹配导致零输出的失败及修正worker身份保在原证据，不改变产品数学或误差门。本轮结果与验证边界见[最终工作快照](../evidence/planning/current-work-final-2026-10-24.md#mtp本轮结果与边界)，不维护第二份按实验轮次累积的待办。

## 25 个 custom descriptor 与实际反向关系

下表反查原25项公开描述符；当前新增量化描述符与精确角色见上文，不把这个历史索引当作当前描述符总数。源码链接定位入口；fixture链接说明已有消费者，GA源码链接说明计算需求。旧fixture仍使用旧profile身份时只作为recipe证据；后续正式GA来源的独立消费者按各自数值与资源范围提供执行证据。公共能力和固定模型组合分别判断，不能因为测试是消费者就永久保留一个废弃专用组合，也不能因移除preview就删除GA仍需的数学。

| Descriptor / 源码入口 | 现行需求、已有消费者或接线证据 | 保留边界 / 尚待核实 |
| --- | --- | --- |
| [`GatherElements`](../../include/hyle/compute/model_ops.h#L19) | [generic MoE routing](../../tests/compute_models/d5_moe_execution_fragments.h)：Q24/K/G/GF；GA selection也需逐元素Gather | 通用axis/bounds能力保留，已补I32原码与[同代GA候选映射](../evidence/correctness/ga-bf16-indexer-state-reference-2026-10-21.json)；GPU当前rank2末轴，B/T可元数据展平，完整模型仍未验 |
| [`RowCacheEncode`](../../include/hyle/compute/model_ops.h#L64) | [row-cache状态消费者](../../tests/compute_state/row_cache_consumer.cc)与typed batch缓存；GA量化index有逻辑需求 | 通用row-inline codec保留；不能将Hyle物理行格式称官方checkpoint |
| [`RowCacheGatherDecode`](../../include/hyle/compute/model_ops.h#L94) | [row-cache Graph fixture](../../tests/compute_models/row_cache_graph_fixture.h)与状态消费者 | 通用typed gather/decode保留；选定profile的codec另绑 |
| [`RowCacheGatherDecodePadded`](../../include/hyle/compute/model_ops.h#L134) | [typed paged reader](../../include/hyle/graph/batch.h)：padding/空history语义 | 通用-1 padding、mask与页权限保留，不随专用scorer一起删 |
| [`RowCacheIndexerScores`](../../include/hyle/compute/model_ops.h#L169) | 原H64行缓存[scorer fixture](../../tests/compute_models/row_cache_indexer_fixture.h)保留；正式GA的BF16 post-QAT cache／三处BF16 score边界、Top512与同代I32映射已由[四端普通图消费者](../evidence/correctness/ga-bf16-indexer-state-reference-2026-10-21.json)验证 | 原行codec／scorer描述符不等同正式BF16缓存或其生产者，也不是GLM H32通用indexer；正式ratio4生产者另有[有限闭环](../evidence/correctness/ga-ratio4-producer-reference-2026-10-21.json)，不签完整GA模型 |
| [`CacheEncode`](../../include/hyle/compute/model_ops.h#L213) | [公开cache codec](../../include/hyle/compute/model_ops.h#L205)面向追加的新行与native GGUF cache | 通用codec保留；GA cache实际codec不能由同名推断 |
| [`ArtifactWeightDecode`](../../include/hyle/compute/model_ops.h#L244) | [非GGUF载体fixture](../../tests/compute_models/cpu_c5_non_gguf_fixture.h)；NF4/compressed-tensors精确格式 | 保留独立通用artifact能力；24个profile的逐role使用与未使用仍需反查 |
| [`MlxQuantized`](../../include/hyle/compute/model_ops.h#L280) | [MLX packed fixture](../../tests/compute_models/cpu_c5_non_gguf_fixture.h)：affine/MXFP独立载体 | 保留通用MLX能力；不能将人工MLX样例签为所选GGUF/官方profile |
| [`BlockQuantizationRoundtrip`](../../include/hyle/compute/model_ops.h#L322) | [GA同族量化recipe](../../tests/compute_models/d5_deepseek_official_fragments.h)与正式源FP8/FP4 QAT | GA域仍需而保留；roundtrip不能替代完整量化MatMul证明 |
| [`Float8WeightDecode`](../../include/hyle/compute/model_ops.h#L359) | [GLM absorbed权重准备](../../tests/compute_models/d5_glm_remaining_fragments.h#L327)实际调用；GA wo_a仍有冷转换需求。[E8M0 macroblock128冷导入](../evidence/correctness/e8m0-block128-import-reference-2026-10-21.json)已保原E4码并将真实E8 scale网格接入canonical F32-scale载体，四端有限消费者通过 | 原scale解释与BF16 decode合同保留；有限合成导入不代签所选wo_a原payload、完整grouped导入或整模型 |
| [`Float8BlockLinear`](../../include/hyle/compute/model_ops.h#L395) | [动态FP8线性fixture](../../tests/compute_models/d5_dynamic_fp8_fragments.h)与所选GA／GLM角色；E8导入、[grouped power2](../evidence/correctness/grouped-fp8-power2-reference-2026-10-21.json)及[三种明确scale策略](../evidence/correctness/block128-scale-policy-reference-2026-10-21.json)已有有限回归 | 保留F32-scale carrier与各policy的独立舍入，不把已接通的有限参考扩成所有payload、原生GEMM逐位或整模型 |
| [`GroupedFloat8Linear`](../../include/hyle/compute/model_ops.h#L442) | [GLM grouped FP8 fixture](../../tests/compute_models/d5_grouped_fp8_fragments.h) | GLM域保留；不能把可选FP4→FP8转换当GA原始expert格式 |
| [`GroupedFloat4Linear`](../../include/hyle/compute/model_ops.h#L490) | 两GA仍是FP4 weight32/FP8 A128；[正式来源的完整FFN参考](../evidence/correctness/ga-full-ffn-reference-2026-10-21.json)已四端执行Flash D4096／H2048、E256／Top6／resident7的普通与hash组合，含同代原encoded bank和非identity ID映射 | 保留GA组合，不等同Kimi A32；有限合成码／scale不签原payload、Pro全部几何或整模型，旧preview记录不改写为GA通过 |
| [`Clamp`](../../include/hyle/compute/model_ops.h#L531) | GLM learned index ReLU、GA gate/up clip10；[MoE fixture](../../tests/compute_models/d5_moe_execution_fragments.h) | 通用min/max/clamp能力保留，Infinity/NaN规则按端口核 |
| [`SqrtSoftplusRouting`](../../include/hyle/compute/model_ops.h#L575) | 两GA E384/256、Top6、scale2.5/1.5仍需；[routing recipe](../../tests/compute_models/d5_moe_execution_fragments.h) | 保留；CUDA已核bias只选ID、无epsilon归一化、原生tie与零和；完整制品仍待核 |
| [`SqrtSoftplusHashRouting`](../../include/hyle/compute/model_ops.h#L622) | 两GA首3层token-table hash；[SGLang接线](../../src/compute_provider_cuda/router_provider.cc) | 保留；已改按真实分数和归一化，含零和NaN、表序/重复ID/bounds回归；完整制品仍待核 |
| [`SinkAttention`](../../include/hyle/compute/model_ops.h#L655) | GA learned sink/sparse attention；[状态消费者](../../tests/compute_state/deepseek_sparse_prefix_consumer.cc) | 保留；BF16逐节点转换、mask/sink分母和稀疏位置另核 |
| [`CausalConv1d`](../../include/hyle/compute/model_ops.h#L706) | Q35/Q27/Q24 GDN、K/GF KDA；[真实Qwen消费者](../../engine/models/qwen35/main.cc) | 共享通用conv/history能力保留；GA无需conv不构成删除理由 |
| [`RecurrentGatedDeltaNet`](../../include/hyle/compute/model_ops.h#L765) | 参数化scalar-gated GDN；[GDN recipe](../../tests/compute_models/d5_qwen38_24t_semantic_fragments.h) | 普通GDN推导QKV/状态/快照，backend另核真实能力；Kimi/GLM KDA的channel raw gate是另外合同，不能由宽family映射等同 |
| [`GroupedMatMul`](../../include/hyle/compute/model_ops.h#L817) | Q24/K/G/GF routed专家、GA分组投影；[Q24完整router](../../tests/compute_models/d5_qwen38_24t_full_router.h) | 通用stored-bank/ID能力保留；格式、A=1/S、累加/输出逐role核 |
| [`HyperConnectionCombine`](../../include/hyle/compute/model_ops.h#L863) | [GLM Flash](../../tests/compute_models/d5_glm_remaining_fragments.h#L211)实际接入共享HC core；[正式GA mHC](../evidence/correctness/ga-mhc-reference-2026-10-21.json)已四端验证HC4／Sinkhorn20的独立来源参考及同代状态旅程 | 共享GF与GA，保留；有限B2／S3和S1、D128参考保F32／BF16边界与epsilon，不代签其它profile或整模型 |
| [`HyperConnectionPre`](../../include/hyle/compute/model_ops.h#L904) | GLM Flash共享core与[正式GA collapse／head](../evidence/correctness/ga-mhc-reference-2026-10-21.json)有限消费者 | 共享能力保留；已核数学／寿命与完整attention／expert分开，旧单层MTP与DSpark角色仍不同 |
| [`HyperConnectionPost`](../../include/hyle/compute/model_ops.h#L939) | GLM Flash共享core与[正式GA expansion／residual](../evidence/correctness/ga-mhc-reference-2026-10-21.json)两代反馈 | 共享能力保留；已核有限combination方向、残差及转换边界，不外推所有profile |
| [`LightningIndexer`](../../include/hyle/compute/model_ops.h#L988) | 原[稀疏fixture](../../tests/sparse_fragment_test_support.h)是F32/F16-mask端口；正式GA三处BF16 score边界已沿[四端普通图](../evidence/correctness/ga-bf16-indexer-state-reference-2026-10-21.json)与CUDA复合承载执行 | 保留旧F32能力，不将其描述符改签为GA BF16 producer或GLM H32；完整模型仍未验 |
| [`DenseLinearF32`](../../include/hyle/compute/model_ops.h) | [Q24 BF16 router near-tie](../../tests/compute_models/d5_qwen38_24t_full_router.h)、KDA投影与 GGUF F16 词表；GA router也有F32 score需求 | 原浮点权重与 F32 分数合同，不等同普通低精度 MatMul 输出；Provider 按实际组合接纳 |

派生关系也必须保留在反向核对中：[detail::RowCachePagedIndexerScores](../../src/compute_batch/row_cache_map.h#L67) 与 [Graph PagedRowCacheIndexer](../../include/hyle/graph/batch.h#L81)依赖上述H64 scorer，另有页权限/metadata/empty-history合同；它们不是新增的公开custom描述符。不能只删schema而漏掉CUDA/Apple/Vulkan/CPU接线，也不能顺带删除generic row-cache map/gather/decode与typed paging能力。

[Hadamard](../../include/hyle/compute/operation.h#L390)是普通公开operation，不在25个custom描述符内；两GA仍需normalized Hadamard。带 `deepseek` 名称的[CUDA FP8 provider](../../src/compute_provider_cuda/deepseek_fp8_provider.cc)还承载GLM使用的decode与独立roundtrip能力；ggml `dsv4_hc_*` 也被GLM Flash使用。因此不能按路径/符号前缀或旧preview标签批量删除第三方依赖。

## 继续细化与融合审查边界

今后接入具体payload或扩展当前数值／布局域时，按实际角色核codec、block/group/axis、codes/scales/bias、pack/stride/orientation、activation量化、饱和/舍入、累加与输出。GA GGUF主header、DSpark可选sidecar及各backend原生域仍分别记录；全profile汇总formats不能代替逐role身份。这些未验证项不自动生成每模型／dtype／后端的新增底座任务。

每条实际recipe需连到Hyle admission/lowering、原生standalone/region入口和依赖版本，并区分prefill/chunk/decode、真实M/N/K、ragged/tail、KV/state有效范围、动态参数及设备容量。核对Cast、公开中间值、外部reader、alias/state顺序与融合被切断的位置；packing/cache、workspace所有者、rebind/replay/capture、completion/失败不发布和旧结果寿命同时保留。上列有限运行结论各有证据；它们不代表所有profile／几何的运行审查、原payload、融合命中或端到端收益。交付范围与旧R04–R12记录的[有限对照](../evidence/correctness/compute-graph-delivery-review-2026-10-21.json)保留preview失败身份和当前F32数学边界，不把旧记录改写为当前GA通过。

本轮没有确认可仅因撤销preview而删除上述router、GroupedFP4、HC、压缩或indexer实现。F14.4旧MTPBlock的变更应由DSpark角色对照解决；公共数学与仍有现行需求的载体继续保留。历史catalog、provider map、ledger及状态只提供来源，不签署当前支持。

[pro-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/README.md
[flash-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/README.md
[pro-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/config.json
[flash-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json
[pro-source]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/blob/72e1d3230f6c080a530b0a1d46f8eb4602340597/inference/model.py
[flash-source]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py
[pro-gguf]: https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF/tree/6d053616d152293da72569f56b861466f10ace7d
[flash-gguf]: https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/tree/fbbb5b93fb787c21338159b0af3318bb3f4d9768
[flash-dspark]: https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/blob/fbbb5b93fb787c21338159b0af3318bb3f4d9768/dspark/README.md
