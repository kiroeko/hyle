# 工作计划最终快照：2026-10-24

> 历史记录，不是当前待办或执行顺序。保留本轮MTP结果、验证边界和性能对照；后续工作仅按用户当前指令推进。
> 原页：`docs/developer/current-work.md`，归档前SHA256：`4f303bc8dea92559447971213d696fa4a96405ae2766dd65b24ade18001f586f`。相对链接已按归档位置调整。

更新：**2026-10-24**。进入“实际推理驱动的性能、功能与架构收敛”阶段。
范围仍是Windows／Linux／macOS的单机CPU、单GPU和CPU＋一张GPU；多模态限图片encoder／projector（mmproj）到文本。
Qwen3.5-2B是检验底座的真实测试消费者；架构还要承载其他已选模型、量化格式、算子和后端。

**本轮MTP专项已完成。** 已选25个原制品的有限流程、必要公共修复、Qwen3.5-2B真实消费者及最终四臂对照均已有证据。实际范围与验证边界见下文。

本轮MTP明确排除Kimi（用户2026-10-23确认）：已选原版及两主选GGUF均无原生MTP权重。Kimi普通推理、量化与通用算子支持仍在原范围内。

## 现在的位置

旧阶段8个能力包、3个交付包已完成各自列明范围，不代表架构已全量审完、全部模型已完整执行。现有冷编译、不可变计划、独立Invocation、状态版本、分页权限和真实CUDA文本推理基础，继续推进性能、功能与架构三条主线。
旧阶段见[归档](current-work-through-2026-10-21.md)；截至AY的数值调查、逐批回归及其限制见[本次归档](current-work-through-2026-10-22-ay.md)。源码、日志和数据保留在原证据中，不将历史实验自动重开为任务。

**最新完整性能见[页末2026-10-24单轮／真实两轮四臂窗口](#最新完整性能对比)**：单轮普通Hyle／llama.cpp decode98.093／101.998tok/s，MTP120.508／135.747；两轮普通98.082／101.590、MTP125.465／139.519，两轮综合分别92.967／93.534和114.806／123.383。普通公共余差和MTP增量分别调查；本窗不是隔离某次优化的收益实验。

Quark自身原权重的[target43／44、NextN与全词表验证](../correctness/mtp-glm-quark-target-verifier-2026-10-24.json)已通过CPU／CUDA各87次Graph，计入有限制品覆盖。手工验证程序曾混用CPU原生库，已按[同源运行库记录](../correctness/mtp-cpu-native-family-2026-10-24.json)纠正，不作为产品HC缺陷。

[Qwen NVFP4（Q03）原权重接线](../correctness/mtp-qwen35-nvfp4-target-verifier-2026-10-24.json)已通过：49原角色约1.17GB，CPU／CUDA各23次Graph，共46次；原始激活量化与BF16返回、同一词表、接受修复及完整续算均保留。上游尾层输入／历史仍是有限fixture，不代表完整模型质量或最终测速。

共用[F32激活×BF16权重批量投影](../correctness/mtp-batched-f32-bfloat-linear-2026-10-24.json)已补齐：Windows CPU／CUDA共6项检查通过，[Mac CPU／Metal另9项](../correctness/mtp-batched-f32-bfloat-apple-2026-10-24.json)及[Vulkan两项](../correctness/mtp-batched-f32-bfloat-vulkan-2026-10-24.json)也通过。后者以小型实际投影核对F32低位、批权重偏移和输入保持，另检查真实64头的两种冷描述符；不把BF16激活的公开合法域外推为Vulkan支持域。权重保持BF16，不展开整份F32，也不自动融合普通Cast权重图；MLX原制品有限执行另有下述独立证据。

GLM GGUF的F32状态与逐ID原字节供给已接入[同一GLM K1流程](../correctness/mtp-glm-gguf-shared-runtime-2026-10-24.json)，原Pro CPU46次、受影响Qwen BF16 MoE CPU43次Graph回归通过；B09的47静态角色约2.57GB及实际逐ID原专家通过[CPU／CUDA各44次Graph](../correctness/mtp-glm-gguf-b09-target-verifier-2026-10-24.json)，覆盖全词表验证、接受修复、pending续算与index追赶；上游层与初始历史仍是有限fixture。B12已通过[原85静态角色及实际专家的CPU／CUDA各87次Graph](../correctness/mtp-glm-gguf-b12-target-verifier-2026-10-24.json)，包括全词表K1、接受修复及pending续算；[B13原Q8_0制品](../correctness/mtp-glm-gguf-b13-target-verifier-2026-10-24.json)也已通过自身85静态角色及实际选中专家的CPU／CUDA各87次Graph。B09／B10／B12／B13转换GGUF有限接线均已完成，仍不代签完整模型或TPS。

[Q14 GGUF NVFP4原权重接线](../correctness/mtp-qwen27-gguf-nvfp4-target-verifier-2026-10-24.json)通过CPU／CUDA各23次Graph；自身29角色约2.90GB、原词表、接受修复与续算均保留。layer62输入及初始历史仍是fixture，不代表完整模型质量或TPS。

[B04 fidelity GGUF原权重接线](../correctness/mtp-qwen27-fidelity-gguf-target-verifier-2026-10-24.json)也通过CPU／CUDA各23次Graph：自身29角色4,927,328,256B、原Q8_0 embedding／BF16全词表head、target63／NextN64及12资源接受后修复、pending续算均保留。上游layer62输入与有限历史仍是fixture；该有限接线不衡量完整模型质量或TPS。

[Q01 GPTQ原权重接线](../correctness/mtp-qwen35-gptq-target-verifier-2026-10-24.json)通过CPU／CUDA各23次Graph；49原角色约1.17GB、同一词表、七组packed投影、接受修复与续算均通过，整权重展开为0。上游layer22与初始历史仍是fixture。

共用[GDN输出缩放](../correctness/mtp-gdn-output-scale-2026-10-24.json)已补齐，Windows CPU／CUDA各3项直接Compute检查通过，默认数学、输入状态和快照保留；[Apple CPU／直接Metal另各3项](../correctness/mtp-gdn-output-scale-apple-2026-10-24.json)通过。[默认Apple Graph桥接](../correctness/mtp-gdn-output-scale-apple-graph-2026-10-24.json)另通过1项代表检查：F32×BF16投影、普通GDN的default／raw缩放、两快照、输入保持，以及输出跨后续Invocation与原图销毁的寿命均通过。[Windows Vulkan直接两项](../correctness/mtp-gdn-output-scale-vulkan-2026-10-24.json)也通过，覆盖GDN／KDA／recurrent gates的default／raw缩放、快照与有限域；增量编译40个host和1个native源，复用SDK与全部着色器。MLX同Flash流程13源接线已合入，[自身原权重CPU／CUDA有限流程](../correctness/mtp-glm-mlx-target-verifier-2026-10-24.json)现已通过；以上各有明确的有限算子／Graph范围。

## MTP本轮结果与边界

**分开交付和报告：** Qwen3.5-2B完整消费者、其它模型／量化制品覆盖、最终性能收尾分别判断；局部尾部链不能代签完整模型，量化身份条目数也不代表独立执行器数量。完成按本轮列明范围及实际验证判断；只做受改动影响的必要检查，不追加全组合验证。

既定四臂长生成评估与普通T1／T2数值调查已完成，任务共同失败及浮点差异结论保留在原证据中，不以继续加题改善2B智商作为MTP门槛。普通自然多轮与llama.cpp对照驱动已在当前闭包完成真实两轮四臂功能检查：每臂8枚自然输出、309行实际prefill、最终缓存长度315；只追加前一轮pending与新增输入，不重算历史。当前单轮／真实两轮四臂速度及独立内存窗口均已实测完成，各12＋4进程，结果见页末；两轮功能检查另保有限状态证据。

本轮各组结果如下。每组复用已有加载、数学与状态机制，具体身份以[有限清单](../research/mtp-artifact-scope-2026-10-23.json)为准；25/25个原制品有限流程均已完成。

| 顺序 | 本轮工作组 | 当前状态／下一动作 |
| --- | --- | --- |
| 1 | DeepSeek BF16／MXFP4 sidecar：S03 | 已完成原权重有限状态接线：CPU／CUDA各87次Graph，受影响S02 CUDA87次通过，共261次；完整模型质量／TPS不在此证据范围 |
| 2 | Qwen派生BF16 NextN：Q03／Q09／Q01／Q10 | 四个原制品均通过CPU／CUDA各23次Graph；七组target投影均packed执行，无整权重展开；接线组已完成，完整模型质量与最终测速另列 |
| 3 | Qwen GGUF：B04／B07／Q14 | Q14与B04各以自身原29角色通过CPU／CUDA各23次Graph；[B07原33角色与自身Q8词表](../correctness/mtp-qwen24-fidelity-gguf-target-verifier-2026-10-24.json)通过两端各43次Graph，含三个场景与12资源修复续算；CPU大head准入已修，CUDA复用缓存，仅新增一个实际专家 |
| 4 | GLM转换GGUF：B09／B10／B12／B13 | 四个原制品有限状态接线均已完成；B13自身85必要角色、实际专家及完整接受后续算通过CPU／CUDA各87次Graph |
| 5 | Quark GLM Flash：Q30 | 自身100个静态角色、实际选中ID六平面源及target43／44／NextN的全词表接受后续算，通过CPU／CUDA各87次Graph；有限接线已完成 |
| 6 | MLX混合量化GLM Flash：Q32 | [原制品CPU／CUDA各90次Graph通过](../correctness/mtp-glm-mlx-target-verifier-2026-10-24.json)；原117静态角色、75个实际专家、32资源完整续算，有限接线已完成 |
| 收尾 | 质量、必要后端及最终性能 | Qwen完整生成评估、单轮／两轮四臂对比及本轮必要数值／续算／寿命和代表后端检查均已完成；保留共同任务失败、硬件未实测及性能余差，后续影响实际消费者的改动另做针对性复测 |

完成以实际原权重和完整接受后续算、必要能力修复、相关正确性证据及最新完整性能对照为准。自然接受率与控制接受试验分开，普通推理成本与MTP增量分开；有限尾层、硬件未实测边界及未确认制品继续如实标记。

覆盖范围已按固定元数据整理为[有限制品清单](../research/mtp-artifact-scope-2026-10-23.json)：25个确认含MTP的精确身份均已有原角色、实际词表／专家与完整有限状态实跑，物理schema和原供给接线均已完成。加载、计算和状态按实际schema复用，没有为各身份另建执行器；无MTP权重、待确认和已执行范围分开记录，不把身份数当完整模型覆盖率。

**最近已验证进展：** [Qwen共用投影接线整合回归](../correctness/mtp-qwen-projection-source-integration-2026-10-23.json)三项共69次Graph通过，覆盖S01 CPU／CUDA和官方BF16 CPU；tied词表、NVFP4叶与混合BF16投影源码已合入，该历史回归只覆盖共用源码；Q03、Q14与B04自身实跑现已另列。

[同版本Flash主GGUF原Q8词表组合](../correctness/mtp-dspark-main-gguf-q8-vocabulary-2026-10-23.json)已通过CPU／CUDA各87次Graph，受影响的正式BF16 CUDA回归45次也通过，三项共219次。保留原Q8／MXFP4三层draft、原target40→41→42、全V六行verifier及30成员K5状态；两份新增原Q8词表共约1.13GB。此项完成已有S02的词表组合，不增加身份完成数，不代签整个主UD-Q4模型或TPS。

[正式DeepSeek Flash原FP8三层draft／原target40→41→42／全词表K5](../correctness/mtp-dspark-formal-flash-2026-10-23.json)已通过CPU／CUDA，共用命名源后的Pro CPU／CUDA回归也通过，四项180次Graph。Flash新增97角色约595MB，借用已有target／词表；实际专家按选中ID获取，无整库或重复词表副本。上游layer39输入及初始历史仍是fixture。

[Qwen官方BF16 Q27 target63／NextN／全词表K1](../correctness/mtp-qwen-bfloat-target-verifier-2026-10-23.json)已通过CPU／CUDA，连同受影响的原GGUF F32回归共四项92次Graph；原29角色约6.68GB，含接受0／1、实际target tap修复草稿KV和12资源同Head续算。首轮失败是检查器读取已释放冷常量，已改读实际参数租约，计算及容差未改；上游输入／初始历史仍是fixture。

此前[DeepSeek Pro原target58／59／60＋正式三层draft／全词表K5](../correctness/mtp-dspark-pro-target-verifier-2026-10-23.json)，CPU／CUDA各45次Graph通过，含30成员同Head接受、pending续算、旧读者、取消及有界回收；共享执行代码的Flash CPU／CUDA回归各87次Graph也通过，合计四项264次Graph。layer57输入和有限初始历史仍是fixture，不代签完整61层模型、任务质量、自然接受率或TPS。此前[GLM Pro target77→NextN78及原78回归](../correctness/mtp-glm-pro-target-verifier-2026-10-23.json)的四项140次Graph及其上游边界继续保留。

**本轮已完成：** [Qwen原始BF16 MoE target91／NextN／全V K1](../correctness/mtp-qwen24-bfloat-target-verifier-2026-10-23.json)通过CPU和CPU词表＋CUDA主体两种放置，各43次Graph，合计86次，0失败／0跳过。保原33静态角色、真实512域Top10专家、接受／拒绝及12资源同Head续算；只取实际专家切片，两原词表留host，未下载整份约51.5GB专家bank。接线暴露的队列绑定、BF16上传描述和[公共Graph退休缺陷](../correctness/graph-segment-retirement-2026-10-23.json)均已修复。上游layer90和初始历史仍是fixture，无完整模型质量或新TPS结论。

**本轮新增完成：** [Q27独立MTP GGUF（S01）](../correctness/mtp-qwen27-sidecar-target-verifier-2026-10-23.json)的18原角色约1.36GB已通过CPU／CUDA，各23次Graph；受影响的主GGUF CPU／CUDA各23次及官方BF16 MoE CPU 43次也通过，共5项135次，0失败／0跳过。draft保自己的Q3_K词表，target保主GGUF原Q6_K head／global norm，共用K1与12资源状态；上游layer62和初始历史仍是fixture，没有完整模型质量或新TPS结论。

**本轮收尾结果：** 原制品有限流程25/25完成。最后B07的CPU／CUDA各43次Graph、自然／控制接受／控制拒绝三个场景均通过，0失败／跳过；共80个实际专家集合，CUDA仅新增一个，其余复用缓存。Q32两端共180次Graph也已完成。当前Qwen真实两轮功能、单轮／两轮各12＋4进程的性能窗口及有限候选决策均已完成；普通／MTP余差另见性能计划。MMQ的7项实际检查、Linux仅三处源码编译及Orin未实测的界限继续保留。

安全批宽属于Provider冷准备事实：大packed矩阵保留canonical、K%512、I32逻辑索引／stride及U32编码字节界。原生MMVQ已补尾行读取保护；新增冷查询按实际设备与原生选择器判定1～4行的MMVQ／MMQ资格，MMQ另保I32编码字节界并纳入调用方workspace预算，按实际stream复用最大临时缓冲。新增实现已通过4070上的独立MMQ数值、缓冲及尾部检查，Orin实机仍未验证，不能由4070结果外推。模型消费者无需知道这些设备分支。

Q27／Q24共用target attention、M2 head、K1状态流程及相应纯数学参考，模型各自保留dense／MoE、原参数和数值边界；未增加执行器、公共算子或模型白名单。[Q27验证链](../correctness/mtp-qwen27-target-verifier-2026-10-23.json)、[Q24验证链](../correctness/mtp-qwen24-target-verifier-2026-10-23.json)、[packed M2](../correctness/mtp-packed-head-m2-2026-10-23.json)与[GLM Pro原NextN](../correctness/mtp-glm-pro-nextn-2026-10-23.json)的实际范围保留在原证据中。

| 模型／制品 | 已验证的实际范围 | 剩余工作或验证边界 |
| --- | --- | --- |
| Qwen3.5-2B BF16 | 原335权重的greedy K1，含多会话／多轮／B2、完整前缀、有界重建和列明三平台取消／退休／预算恢复；[8项QKV差异](../correctness/qwen-mtp-quality-prefix-2026-10-22.json)指向普通单／双行稳定浮点差异 | 单轮／真实两轮当前性能复测与本轮候选决策已完成；保留decode余差，不将有限数值检查外推为所有形状或任务质量 |
| Q27主UD-Q4_K_M | [原17角色](../correctness/mtp-q27-original-nextn-2026-10-23.json)完整NextN／全V词表；本轮加原target63／global norm、实际tap的M2 verifier、接受0／1与target-conditioned草稿缓存修复，12资源同Head续算，CPU／CUDA通过 | 本制品有限接线已完成；完整上游主干／初始历史仍保边界 |
| Q27官方BF16 | [原29角色target63／NextN／两原词表](../correctness/mtp-qwen-bfloat-target-verifier-2026-10-23.json)，保原BF16 store／delta norm，实际M2全V及12资源K1接受修复／续算，CPU／CUDA通过 | 本制品有限接线已完成；上游layer62输入及初始历史仍保边界 |
| Q24主UD-IQ4_XS | [原22角色](../correctness/mtp-q24-original-nextn-2026-10-23.json)完整NextN／两原词表；本轮加原target91／global norm、真实512域Top10 IQ4_XS专家、M2 verifier及接受后缓存修复，12资源同Head续算，CPU／CUDA通过 | 本制品有限接线已完成；完整上游主干／初始历史仍保边界 |
| Q24官方BF16 | [原33静态角色、target91／NextN／两原词表](../correctness/mtp-qwen24-bfloat-target-verifier-2026-10-23.json)，真实512域Top10原BF16专家及12资源K1续算，CPU／CPU词表＋CUDA主体各43次Graph通过 | 本制品有限接线已完成；上游layer90及初始历史仍是fixture，混合结果不代签全CUDA或完整模型质量 |
| GLM Flash官方 | [完整原NextN1762角色](../correctness/mtp-glm-original-moe-2026-10-23.json)，以及[独立原43→44→NextN](../correctness/mtp-glm-original-target43-2026-10-23.json)、原全V双行验证、真实pending和34资源＋Epoch的K1接受／续算 | 已选Flash量化有限接线均已完成；layer42输入与有限初始历史仍是fixture |
| GLM Pro官方 | [真实原target77→NextN78](../correctness/mtp-glm-pro-target-verifier-2026-10-23.json)、独立model.norm与全V反馈，14资源＋Epoch的接受修复／index追赶／续算，CPU／CUDA通过；原78回归也通过 | 已选Pro量化有限接线均已完成；上游74选择、layer76输入及初始历史仍保边界 |
| 正式DeepSeek V4 Pro | [原58→59→60＋正式三层draft](../correctness/mtp-dspark-pro-target-verifier-2026-10-23.json)，216静态／词表角色、实际选中384域原专家、全V K5及30成员完整状态接受／pending续算，CPU／CUDA通过 | 已选适用制品有限接线均已完成；layer57输入及初始历史仍保边界 |
| 正式DeepSeek V4 Flash | [原40→41→42＋正式原FP8三层draft](../correctness/mtp-dspark-formal-flash-2026-10-23.json)、独立末端HC／norm、原全V K5及30成员接受／pending续算，CPU／CUDA通过；[S02同版本主UD-Q4原Q8词表组合](../correctness/mtp-dspark-main-gguf-q8-vocabulary-2026-10-23.json)CPU／CUDA也通过 | 已选适用制品有限接线均已完成；上游layer39输入／初始历史仍保边界 |
| Kimi原版及两主选GGUF | 原版NextN层数为0；Unsloth固定版本的UD-Q4_K_XL／UD-Q8_K_XL已分别清点完整2573个tensor，均无MTP／NextN角色，仓库文件清单也无独立draft／MTP制品；见[制品清单](../research/mtp-artifact-scope-2026-10-23.json) | 这三个固定制品均无原生MTP接线任务，普通推理的量化／算子支持仍在范围内；不外推其它版本 |

动态E4 codes＋F32 scales已有四端逐bit、失败恢复与寿命检查；indexer采用PowerOfTwo／floor1e-4，原MLA／MoE采用Divide／floor1e-10。CUDA公共I32→F32 Cast边界、状态、页权限、只读租约、预算和原子发布继续共用，范围见[总表](../../implementation/model-operators.md)。未执行的模型、系统、codec或完整质量不写成通过；角色数不作为完成百分比，最新完整性能已更新为页末2026-10-24两窗，未完成制品不由Qwen窗口代签。


## 执行顺序与近期进度

| 顺序 | 工作 | 状态与下一动作 |
| --- | --- | --- |
| 1 | 容量参数与初始化 | 消费者已接上下文、缓存会话数、活跃序列数与状态字节预算，支持auto和提前拒绝；状态预算不冒充全显存上限 |
| 2 | 多会话缓存与执行 | Radix／LRU、COW／独占尾权、共享执行资源、完全命中、有界恢复及真实B2已接。三平台有限完整前缀、取消／退休及额度恢复已有；跨形状数值／质量、实际性能和其它模型状态组合仍按真实范围推进 |
| 3 | MTP | 本轮完成：25/25原制品有限流程、必要公共修复、真实Qwen消费者、单轮／两轮四臂性能对照及有限候选决策均已交付，详细边界见上表 |
| 4 | 图片encoder／projector | 前三项列明范围已完成，完整图片链路待启动；已有297个BF16视觉张量、24层H1024 encoder及4096→4096→2048 merger身份。后续RGB／patch→视觉Graph→文本embedding、多轴位置／delta、图片前缀身份与寿命；原LayerNorm weight＋bias |
| 并行 | 已知性能慢项 | 普通／MTP decode差距、descriptor上传依赖、随历史增长的页组织、cached-prefill首用与合法拓扑变化继续按实际热点处理，见[性能计划](../../developer/performance/decode.md) |
| 贯穿 | 架构审查与实现收敛 | 保性能、职责、数值及寿命正确；检查其它已选模型／算子／后端，普通扩展不反复修改共同调度核心 |

用户顺序仍是多会话缓存／推理→MTP→图片；当前先收MTP已有明确剩项。单请求延迟、多人总吞吐、缓存命中和重建成本分别报告，不把一次CUDA结果外推到全部系统／制品，也不把完整KV以外状态丢掉后宣称可任意截断前缀。

**连续批处理调度扩展后置。** 参考llama-server可在线接纳新请求、按token预算混合兼容prompt／decode；当前消费者采用预定wave及T／role／lag桶。引擎负责调度政策，底座仍需承载可变活跃子集、变长计算、完整状态与有界资源复用；当前MTP任务不扩建server或新增多执行器。

## 性能：借鉴成熟实现，落到真实瓶颈

已有[本地7个引擎的优化研究](../research/local-inference-optimization-2026-10-21.json)，包括TensorRT-LLM、vLLM、SGLang、llama.cpp、ik_llama.cpp、DeepInfer和ds4；其他既有研究由旧计划保留链接。
先查当前准备／重录成本，再试实际热点的少量合法冷算法；量化链再检查保原舍入的融合、packed分块计算及多余转换／搬运。
匹配完整数学、设备和工作量后才采用。量化路线的改进不自动算成BF16 Qwen收益，源码启发也不算实测收益。

最新完整对比固定放在本页末尾；阶段归因与实验决策见[性能计划](../../developer/performance/decode.md#三轮共享缓存的四项基线2026-10-21)。
此前约22%改善属于一次cached16输入续算，不是整体decode提升。

每次完整对比固定报告**prefill、完整decode、显存变化、多轮综合速度**；加载／冷准备与热执行分列，
测速和显存旁采分开，保同输入、精度、缓存、状态、完成与计时边界，见[统一口径](../../developer/performance/README.md#推理对比固定报告)。
先排除工作量不等价和测量分辨率，再优化时间复杂度、显存复杂度与实际常数成本。

## 功能：引擎表达意图，底座承载机制

[容量参数研究](../research/engine-capacity-2026-10-21.json)提出下面四种主要意图，现已在[测试消费者](../../../tools/qwen35_text/README.md)提供相应参数及auto解析；它们不是Hyle库要求用户手填的内部字段。

| 用户可表达的意图 | 必须分清的含义 |
| --- | --- |
| 单序列上下文上限 | 一条序列的历史容量，不等于物理KV池 |
| 缓存会话数 | 保留的活跃＋空闲会话，不等于同时执行数 |
| 同时执行序列数 | 正在调度的序列并发，不等于执行槽或用户总数 |
| 缓存字节预算 | 各层KV、conv／GDN、分支写入及保留版本总成本，不等于全进程显存 |

总内存预算可选；步长、执行槽数和常用准备形状通常自动推导。启动报告实际容量、已准备域、已知／未知资源和超限行为。
参数支持预分配与准入，不能承诺固定tokens/s或控制驱动／其他进程的显存。
当前测试消费者按turn接纳活跃请求，空闲会话可按数量或状态字节限制淘汰；再使用时按原输入chunk重建完整状态。默认auto按本次输入与预算解析，没有固定的用户数要求。
上限不预先分配对应数量的会话，单用户decode直达自己的Session，不逐token查询LRU／预算。单序列冷准备由同时执行数表达，不把缓存数量限制误当加速开关。
缓存必须同时受会话数和字节预算约束；共享物理页只计一次，conv／GDN、COW暂存、旧版本和待退休页仍计实际占用，不能把移出目录当作显存已释放。
当前Qwen将12份arena backing及36份状态池统一预留、计费一次，包含最多R份批次候选版本与COW尾页；不再用旧dense每会话公式计算。容量256、R1保留1／2会话时分别预留42,418,224／65,101,872字节，N2／R2为84,836,400字节；IO、workspace与未知原生私有存储另计。保留旧前缀也消耗版本／页额度，缓存数量限制不是无界检查点许可。
先由引擎按最久未使用顺序淘汰空闲缓存；全部在用或在途退休后仍无空间时排队或拒绝，不覆盖活跃状态。Hyle提供预算与真实退休后的安全回收，不承担用户目录和淘汰策略。淘汰设备缓存不等于删除对话记录；重建所需token等输入由引擎保留。
用户的单台DGX Spark四人编码服务属于未来模型选型场景；当前开发验证继续使用已有验证机器。多会话机制支持可配置容量，缓存会话数另计；服务评估需看总吞吐、每路速度、首token和延迟分布，不由单请求速度相乘推算。

- **多会话缓存**：共享只读权重／计划，隔离完整KV、conv／GDN和位置状态；精确前缀匹配、只读共享、显式写时复制，取消／淘汰后仍保护在途与旧读者。
  会话目录、准入／淘汰和批次策略归引擎；状态、页权限、预准备、独立执行槽与退休归Hyle。不能按会话复制整份模型冒充复用，也不以仅有多序列布局算完成。
- **MTP**：模型格式、tokenizer、draft／verify及接受策略归消费者；Hyle提供计算、完整状态、页权限、搬运和真实退休。Qwen当前目标T2与草拟缓存沿同次Graph／Completion，逐成员选择接受边界后原子发布；已消费前缀、fresh预测与未消费pending分开保存。本轮结果与边界见[上表](#mtp本轮结果与边界)，长期合同见[生成记录与缓存重算](../../architecture/compute-design.md#生成记录与缓存重算)。
- **图片投影**：预处理和模型图片语义归消费者；Hyle补真实计算、布局、特征与文本绑定、搬运及资源寿命。先做图片，不扩展音频／视频。

既有自然轮次、EOS、取消／丢Completion、B2与有界重建、独立CPU／peer对照和数值调查保留在[AY前归档](current-work-through-2026-10-22-ay.md)及其原证据。已确认的完整状态、预算和退休能力继续复用；不同形状浮点差异、未验证的质量或物理资源峰不写成通过。

## 架构审查：性能成立，边界清楚，修改集中

审查目标是**实现是否满足这些边界**，不是再造一套抽象或统一所有原生执行细节。
共同点依据实际计算提取，允许后端保有原生算法、融合、布局和资源；引擎中间层不判断具体Provider／原生算子。

| 审查面 | 要确认并修复的内容 |
| --- | --- |
| 职责与扩展成本 | 模型表达数学／状态，Compute承载算子合同，Provider实现计算，Graph负责计划与依赖；新增算子／融合／后端不反复改共同调度、租约和提交核心，新模型尽量复用已有能力 |
| 冷规划与热执行 | 选路、融合、packing、容量和常见形状尽量冷生成；明确布局变化何时使原计划失效，只有实际依赖变化才重准备，避免隐藏搜索、重复分配／转换／同步 |
| 数值与布局合同 | dtype、量化与scale、累积／舍入、stride、alias、状态副作用可明确描述；融合覆盖真实DAG及读者，跨Provider边界明确import／copy与代价 |
| 不可变共享与可变隔离 | 代码／权重／常量可共享；Invocation、workspace、绑定和会话状态按实际并发要求隔离；分支、批次、旧结果不会相互覆盖 |
| 状态与异步寿命 | KV、recurrent／conv、位置和接受前缀属于完整续算状态；发布、失败、取消、关闭与真实native退休一致，旧版本、结果和设备在途使用都有owner |
| 资源与调度 | 冷／热内存、页池、执行槽、搬运和准备成本可解释；超限行为明确，未知私有内存不记零；混合CPU＋GPU不复制一套执行器，放置／同步按实际依赖处理 |
| 公共接口与可维护性 | 跨系统／设备调用形状一致，错误可定位到层和阶段；普通引擎调用不要求第三方类型、内部宏／Provider开关，明确的原生互操作入口保留；模块可独立升级，支持范围和实现保持一致 |

同时检查Direct与Graph共用合同、依赖和模块方向、缓存／制品身份及失效、能力发现与合法后备，避免隐式全局可变状态和第二套执行路径。
能力声明、路由与支持表应复用事实，减少重复名单和接线；设备或后端的真实限制在准备期说明，不强行抹平差异。
不追求“加算子零改动”：新语义可能需要新公共合同；普通扩展应集中在自己的实现和必要声明，新语义才修改共同合同。
用真实变更检验维护成本：换一个Provider实现、增加一种精度／融合、接入一种不同状态的模型，看改动是否局限于所属层，并检查完整推理速度与资源成本。
接线暴露的架构问题随当前任务逐步修复：Flash／Pro共用普通Graph执行与MoE几何／供给，Qwen共用attention和K1状态流程，专家上传按实际codec／stride派生布局，大packed head安全批宽归Provider冷准备。目标tap、递归草稿hidden及缓存补算的时序属于模型消费者，状态层只承载完整版本；数学参考与执行也分开。继续用其它正式制品核权重绑定和状态组合，保留模型数学与数值边界，持续推进MTP收尾。
长期设计依据在[Compute](../../architecture/compute-design.md)和[Graph](../../architecture/graph-design.md)；已选模型／量化／角色／融合范围以[总表](../../implementation/model-operators.md)为准，原制品未完整执行的条目继续如实标记。

## 工作方式

沿现有公共执行链补缺口，删除重复旧路径，不保兼容分支；不恢复治理、资格或审批流程。
只运行与改动相关的增量构建、数值／状态／寿命检查与真实消费者，复用三平台工具链、依赖和对象缓存，不因计划更新重跑全库。
保留必要源码、命令、失败和结果到[证据目录](../README.md)；新阶段是否完成按上述实际剩余工作判断，不按旧清单完成数推算总进度。

## 最新完整性能对比

**2026-10-24，当前源码的单轮与真实两轮四臂窗口。** 北京时间06:33:07～06:34:54；Qwen3.5-2B、RTX 4070、全CUDA、匹配原BF16权重，C1024／N1／R1、前缀关闭、replay auto、自然greedy并尊重EOS。两种负载各有12个交错测速新进程（每臂3个），另各4个独立进程旁采内存，合计32进程；窗口期间暂停下载和其它原生运行。两窗分别统计，历史样本不合池。

单轮沿原P154输入、64枚输出；两轮在同一已加载模型和会话上各追加相同154枚原输入、各生成64枚，第二轮只消费上一pending与新增输入，不重prefill历史。两轮是明确的原token追加负载，不外推聊天模板或任务质量。

| 负载／模式 | prefill tok/s | 完整decode tok/s | 综合输出 tok/s |
| --- | ---: | ---: | ---: |
| 单轮 Hyle普通 | 2285.823 | 98.093 | 90.373 |
| 单轮 Hyle MTP K1 | 2087.841 | 120.508 | 107.284 |
| 单轮 llama.cpp普通 | 1563.944 | 101.998 | 89.335 |
| 单轮 llama.cpp MTP K1 | 1461.629 | 135.747 | 112.387 |
| 两轮 Hyle普通 | 3301.684 | 98.082 | 92.967 |
| 两轮 Hyle MTP K1 | 2792.392 | 125.465 | 114.806 |
| 两轮 llama.cpp普通 | 2410.144 | 101.590 | 93.534 |
| 两轮 llama.cpp MTP K1 | 2245.021 | 139.519 | 123.383 |

单轮普通Hyle相对参考P／D／综合为+46.16%／−3.83%／+1.16%，MTP为+42.84%／−11.23%／−4.54%；两轮普通为+36.99%／−3.45%／−0.61%，MTP为+24.38%／−10.07%／−6.95%。两轮Hyle自身启用MTP使decode／综合+27.92%／+23.49%，参考自身为+37.34%／+31.91%。普通公共余差与MTP增量分开调查，这些差值不证明由某个控制函数或MMQ造成。

| 负载／模式 | prefill范围 | decode范围 | 综合范围 | 加载s | 入口至ready s | 入口至末输出s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单轮 Hyle普通 | 2258.12～2339.75 | 97.50～98.31 | 89.59～90.39 | 0.732 | 1.065 | 1.780 |
| 单轮 Hyle MTP K1 | 2078.92～2313.89 | 120.00～122.29 | 106.83～110.02 | 0.743 | 1.536 | 2.133 |
| 单轮 llama.cpp普通 | 1553.85～1583.80 | 101.22～102.06 | 88.78～89.52 | 1.587 | 1.615 | 2.336 |
| 单轮 llama.cpp MTP K1 | 1419.25～1469.62 | 135.55～136.25 | 111.64～112.84 | 1.687 | 1.721 | 2.288 |
| 两轮 Hyle普通 | 3265.39～3351.38 | 97.66～98.23 | 92.43～93.00 | 0.732 | 1.098 | 2.474 |
| 两轮 Hyle MTP K1 | 2738.33～2869.26 | 124.44～125.82 | 113.74～115.40 | 0.752 | 1.646 | 2.761 |
| 两轮 llama.cpp普通 | 2391.52～2416.73 | 101.45～102.02 | 93.35～93.91 | 1.628 | 1.657 | 3.020 |
| 两轮 llama.cpp MTP K1 | 2231.69～2302.49 | 139.50～140.26 | 122.99～123.46 | 1.696 | 1.729 | 2.766 |

表中速度及加载／ready／末输出是各臂3个独立进程的中位数，保留全部最小～最大范围，不作显著性结论。单轮decode分子为63；两轮prefill分子为**309行实际target输入（308枚新增用户输入＋1枚pending）**，decode为126枚新输出，综合为128输出／完整连续会话时间。顶层P＋D覆盖完整会话及轮间接纳；分轮span仅作说明。加载边界依消费者实现，入口不含OS装载DLL，末输出不含销毁，含启动总时长在原证据分列。

| 负载／模式 | GPU dedicated峰 GiB | GPU shared峰 MiB | 同点GPU总峰 GiB | CPU WS峰 GiB | CPU private commit峰 GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 单轮 Hyle普通 | 3.888 | 92.000 | 3.964 | 1.051 | 5.512 |
| 单轮 Hyle MTP K1 | 4.048 | 92.000 | 4.124 | 1.142 | 5.786 |
| 单轮 llama.cpp普通 | 3.728 | 82.000 | 3.808 | 4.071 | 4.673 |
| 单轮 llama.cpp MTP K1 | 3.896 | 94.000 | 3.987 | 4.220 | 5.081 |
| 两轮 Hyle普通 | 4.023 | 92.000 | 4.099 | 1.083 | 5.687 |
| 两轮 Hyle MTP K1 | 4.191 | 92.000 | 4.267 | 1.187 | 6.008 |
| 两轮 llama.cpp普通 | 3.728 | 82.000 | 3.808 | 4.075 | 4.676 |
| 两轮 llama.cpp MTP K1 | 3.896 | 94.000 | 3.987 | 4.224 | 5.087 |

内存是独立Windows PSAPI／PDH进程旁采，采样峰是下界，不是精确物理分配。dedicated／shared独立峰不相加，总峰来自同一有效采样点；WS包含映射／共享驻留页，private commit是承诺地址空间，均不能与GPU shared简单相加为物理RAM。请求间隔10ms，实际中位15.01～15.31ms；单轮最大24.72ms、两轮最大19.56ms。八臂ready基线均无样本，保留null；两轮prefill每段仅3～9个样本，阶段邻样差仍带边界年龄。单轮到两轮Hyle普通／MTP dedicated采样峰分别多138／146MiB，参考峰近同值；这是不同完整进程的观测差，不是精确单轮分配增量或已定位泄漏。

**实际工作：** 每臂3次ID、工作量和非计时决策稳定，全部达到64／128输出，未触发EOS。单轮Hyle MTP接受29／34个proposal、验证68行、draft decode62行；参考接受30／33、验证66行、draft decode62行。两轮Hyle接受60／66（90.91%），target验证132行、draft decode124行；参考接受61／64（95.31%），另有1次无proposal的单行尾步，共65次decode调用、target验证129行、draft decode125行。两者prefill target309／draft310行，额外catchup仍计成本，不计用户token。两轮target总计算Hyle441／参考438行，draft总计算434／435行，最终均提交435行。不能按每轮均摊把完整decode余差归于接受率。

**可比边界与后续：** 跨臂自然续文和实际工作不同，单轮从第19枚起分歧；Hyle中间BF16、末分数F32，参考保持原生计算／状态／host carry。报告真实自然生成吞吐，不称同token或数学完全一致，也不以可读性代签质量。本次更新当前源码闭包并接通真实两轮，未作隔离MMQ收益实验；与旧AV的跨窗变化不能归因某次优化。下一步仍是剩余制品接线与普通／MTP实际余差，保留既定质量失败与未知项。

[两窗24次计时、8次独立内存旁采、完整samples／命令／结果](../performance/qwen-mtp-single-and-multiturn-2026-10-24.json)；[自然两轮有限状态与构建证明](../correctness/mtp-qwen-natural-multiturn-2026-10-24.json)。此前[AV](../performance/qwen-mtp-assembly-2026-10-22.json)、[AU](../performance/qwen-capture-variants-2026-10-22.json)及其回链保留历史口径，不混算。
