# 旧阶段工作计划（截止2026-10-21）

这是转入新阶段时的历史快照，保留当时的范围、结论和未完成项，不作为当前任务指令。
本轮最终结果见[最终快照](current-work-final-2026-10-24.md)，本文不作为当前待办。原相对链接已按归档位置调整。

当前收敛 Compute/Graph 推理底座重构，范围是 Windows/Linux/macOS 上的单机 CPU、单 GPU、CPU＋一张 GPU。
**架构主干已落地，剩余工作按下面的能力、交付和性能包完成，不再用旧“约75%”估计进度。**

更新：**2026-10-21，北京时间**。**8个能力包、3个交付包已完成原清单范围；新增多会话缓存复用正在调研接线。D4已完成三轮prefill／decode／显存／综合速度基线，发现后续prefill约5ms准备成本与跨阶段重录capture，尚未收工。**
本轮剩余清单的结项数：能力包 **8／8**，交付包 **3／3**；这是剩余清单的计数，不是整个项目完成度。
每批实现或验证产生实际结果后更新本页；decode 的样本与决策只在[性能计划](../../developer/performance/decode.md)维护。
以后完整性能对比固定报告prefill、decode、显存变化及多轮会话综合速度，见[统一口径](../../developer/performance/README.md#推理对比固定报告)。
用户提供的DeepInfer／deepseek-recipe只读调研已完成，可复用机制已归入下文现有步骤；V4.1 Flash不加入支持范围。
以后确认适用的发现也说明来源、条件与下一动作；必需缺口与待验证优化分别标明。

## 当前重构主线

当前链路为模型语义图→冷编译→不可变计划→独立 Invocation→prepare/bind/layout→严格 submit→Completion／结果／状态发布。
共同核心处理依赖、权限、alias、预算与寿命；各后端处理自己的数学、布局和原生资源。
引擎中间层不判断 Provider 或原生算子，旧 CPU＋GPU 专属 Plan／Invocation／driver 已删除，不保兼容入口。
设计依据见[Compute 设计](../../architecture/compute-design.md#推理执行体系的重构形状)。

| 步骤 | 工作与当前状态 | 完成边界 |
| --- | --- | --- |
| 1 | 已完成本批 D1：定位并修复配置更新后重复绑定准备；相关 CPU/CUDA、数值与短长两窗已验证 | 保完整 step、当前权限／seal／旧结果门控；不由局部修复代签全部底座 |
| 2 | Top1 决策已完成：真实 head 后缀 3 个独立进程＋1 次结构 trace，预定配对方向不一致，默认保留 D1 的 F32 Top1；直接 BF16 消费者候选本轮不采用，已结项；详见 [decode 规划](../../developer/performance/decode.md) | 原始正／负结果和通用 BF16 Top1 测试保留，无挂起候选；不将空间减少或有利中位数代签吞吐收益 |
| 3 | **8／8能力包已完成本表范围**；最后GLM动态FP8的显式scale策略四端24项及CUDA共享量化3项通过、无跳过 | 按精确合同与真实角色／子图闭环，不按模型×dtype×后端或测试数量增加交付项；剩余资源／默认路由问题进入交付包 |
| 4 | **3／3交付包已完成本表范围** | 当前源码、安装消费者、真实资源寿命和公开范围相互一致 |
| 5 | **最终整机性能对比（D4）进行中**：同一缓存三轮256输入＋32 decode、实际上下文864／容量1024，三进程每臂与独立显存旁采完成；综合输出91.78对89.67 tok/s，后两轮prefill仍输，继续查准备与重录 | 不将整体或稳态较快代签所有阶段；具体数值、数学与采样边界见性能计划 |
| 6 | 实现带 MTP 的 Hyle Qwen3.5-2B 测试引擎，与 llama.cpp 的 MTP 路径对比 | 复用已有分支、状态版本与接受前缀能力；同权重／输入／生成策略测端到端 tokens/s、接受率和内存显存，保无 MTP 基线，并核正确性与回退 |
| 7 | 图像 encoder／projector（mmproj）与图像到文本链路 | 在当前纯文本底座收敛后推进 |

当前第 1–5 步是本轮收工范围；MTP 与 mmproj 随后分别推进。Qwen3.5-2B 是校正底座的测试消费者，不扩张为正式推理引擎产品。

D4是性能计划中“最终交付比较”的编号，不是第4个交付包；第4步的3个交付包已经完成各自列明范围。

**当前下一动作**：按用户新增要求，先调研引擎的会话／并行／上下文／缓存预算参数及Hyle预准备支持，再补多个会话的缓存复用。性能侧下一份有限诊断拆开`batch::prepare`与`prepare_replay`，记录六份既有BatchPlanReport，确认后续prefill准备与重录的真实原因；不先放宽复用条件。多轮消费者已接通，原64份向量逐字节回归通过；独立Torch同短计划25个ID一致，块／逐token和Torch向量差异保留，不代签质量。完整数据见[性能计划](../../developer/performance/decode.md)。Mac和Linux的producer／SDK继续增量复用。

**新增多会话缓存复用：未完成。** 目标是共享只读权重与计划、按会话保留完整KV／conv／GDN／位置状态，支持只读前缀复用、显式分支续写和安全回收。
先以交错双会话贯通实际消费者，再检验可配置容量；缓存会话数与同时执行数分别控制。用户目录、准入／淘汰和批次策略归引擎，Hyle承担资源／状态／布局及预准备机制。
[现有能力与缺口核查](../research/multi-session-cache-2026-10-21.json)明确：已有多序列分页、页权限与状态版本，不等于多用户服务已经交付。
7个本地引擎的[优化源码研究](../research/local-inference-optimization-2026-10-21.json)保留后续冷算法、保舍入量化prologue和packed4bit分块计算方向；量化路线不自动加速当前BF16 Qwen。
Apple dense／encoded-A FP8现在与grouped一样按列块直接读取原F32 SF；[8项实际检查](../correctness/apple-compact-fp8-scales-reference-2026-10-21.json)通过、无跳过，包括跨128列边界、显式全权重解码和原三种量化策略／旧结果。正式GLM几何的SF逻辑表示由262144B降至2048B；未测物理分配／流量／吞吐，不计为Qwen decode收益。本批Mac镜像／map已清理，保2个增量对象及共享缓存。
GLM Quark block128已加`Float8ScalePolicy::kMultiplyReciprocal`：F32乘预舍入的`1/448`生成scale，另取F32倒数并乘A，保三处舍入，原Divide／PowerOfTwo数学保持。[四端24项及CUDA共享量化3项](../correctness/block128-scale-policy-reference-2026-10-21.json)通过、无跳过；主24项有172次成功提交、8次预期失败。固定layer45 expert0 gate_proj的M1／N2048／K4096、原E4／F32 SF轴用有限人工字节验证；不代表原payload、完整MTP／模型、AMD硬件逐位、Linux或吞吐。旧26份Vulkan SPIR-V及其余6个VMFB逐字节未变；两个改动vendor与license记录的CMake检查通过。
证据归档后，本批10个Windows、2个Mac测试镜像／map／临时DLL链接已按根`test.ps1`删除，保14＋3个增量对象及共享缓存；[清理实录](../correctness/block128-scale-policy-reference-2026-10-21.cleanup.json)保精确目标，不删除SDK或依赖缓存。
Kimi FP8侧见[四端实测](../correctness/fp8-outer-scale-reference-2026-10-21.json)：16项新增、12项原标量邻居，252次成功提交、无跳过。沿原闭合图保SA行轴／SW列轴及两种显式F32乘法顺序；正式`f_a_proj`几何M1／M2、N128、K7168的有限人工E4／F32消费者通过，不代表原payload或AMD硬件逐位等价。
本批执行明确选择已有压缩候选。CPU／CUDA另做的无约束编译仍产生3.5MiB的F32权重workspace view及其转置alias；未测频率／流量／吞吐，默认选路与空间／速度代价归交付3。没有新的Qwen decode结论。
本批镜像／map与临时DLL链接已按根`test.ps1`清理，Windows约1.19GB逻辑文件、Mac约40MB；保22＋9个增量对象及共享缓存，[清理实录](../correctness/fp8-outer-scale-reference-2026-10-21.cleanup.json)。
不把33条整制品未执行扩大为33个完整模型任务。
[GLM Flash混合MLX角色](../correctness/glm-flash-mixed-mlx-reference-2026-10-21.json)四端各1项通过，另有CUDA对齐回归1项，共5项、无跳过；角色共16次成功提交、4次预期missing映射失败并恢复。
实际D4096／H2048、288专家域取9／Top8，routed4／4／5＋shared6、g64与F16 metadata；BF16输入提升F32，保sigmoid／correction选择、无偏概率归一化与down后route权重，不插BF16边界。
每代92份原格式分立片段，经原WeightSlots三代／双槽与同代I32映射消费；CPU用原host ReadLease，保旧结果及源销毁后的寿命。F32最大归一化误差小于2.6e-7，三种错误精度／路由反例均超原固定阈值。
实际CUDA首跑暴露小bank偏移问题：原装配16字节／scalar对齐不能满足所选原生端口256要求。修原whole与sliced布局，保固定容量、源字节与上传记账；260／259字节边界回归通过，不放宽校验或增加影子复制。
仅验证正式角色的有限合成权重组合，不代表原制品、全部129组bank、完整模型、Linux、capture或吞吐；Apple静态SF／bias的warm重复提升另列交付3待修。
[GA完整FFN组合](../correctness/ga-full-ffn-reference-2026-10-21.json)复用既有`moe_chain`与独立参考，CPU／CUDA／Vulkan／Metal四端各1项通过、无跳过，共24次成功提交。
正式Flash尺寸D4096／H2048、256专家域取7、Top6，普通／hash路由各3组输入；原FP4专家与FP8 shared bank、同代非identity I32映射沿原WeightBatch消费，保3个独立Invocation的旧结果及原句柄销毁后的结果租约。
核F32 router计算、clip／SiLU及down前route权重；首投影、down、合并与最终BF16结果均与独立参考数值完全一致，首投影0ULP，F32阶段最大归一化误差小于1.6e-7。
本批无产品数学或调度更改，只新增消费者；初次CPU读回回调误复制目标容器的测试失败已修并保留，未修改数学或放宽阈值。
范围是正式来源的有限合成权重完整FFN，不代表原checkpoint、完整模型、Linux、全部capture或吞吐；没有新增decode性能结论。
[编码bank分片承载](../correctness/encoded-bank-slices-reference-2026-10-21.json)已修：共同校验对plain bank保原scalar合同，encoded bank接受显式dense U8物理字节片段，保完整覆盖、范围、重叠、原bank对齐与固定容量。
原三GPU上传／InputSource／原子绑定直接复用；只刷新各自weights owner，不增codec、Graph路由或执行器。
6项host检查与三GPU各2项均通过、0跳过；含真实FP4 N2048/K4096、7／256专家、T2／Top6、同代非identity I32映射、两槽三代、missing映射失败与恢复，保plain NVFP4邻居。
实际18次成功Graph提交、3次预期失败；16个原host片段含1字节拆分，消费前销毁源，最终旧结果仍可读。
这是完整FFN的资源前置缺口；组合执行由上述完整FFN消费者另行验证，沿既有图消费down前route权重与shared FP8，未另写重复数学。
[GA mHC组合](../correctness/ga-mhc-reference-2026-10-21.json)四端共4项通过、0跳过，32次成功Graph提交。
B2、S3／S1、HC4、D128、各两代；F32投影后乘RMS因子、Sinkhorn20、BF16 pre／post／head边界与独立参考一致，BF16最大0ULP。
实际post结果租约反馈下一代pre，残差／系数／几何同代发布，核旧结果保护及原句柄销毁后的独立head租约。
复用既有原生算子，无产品更改；测试修正常量借用保活、最终物化输出布置与按实际输出预留arena，保留首失败记录。
只签正式来源的有限组合，attention／FFN是明确的BF16分支边界；未签原checkpoint参数、完整attention／expert／模型、Linux、capture或吞吐。
[ratio4生产者旅程](../correctness/ga-ratio4-producer-reference-2026-10-21.json)四端共8项通过、0跳过：4项新增、4项原四维RoPE邻居，60次成功Graph提交。
F32 KV／score窗口→BF16 pooling／norm／F32 suffix RoPE／Hadamard／FP4-QAT→实际BF16缓存→score／TopK／I32映射，
覆盖prefill3、两次压缩、无产出阶段与接受4个token后续算；每端一次emit只发布状态，保留普通闭合图。
修CUDA／Vulkan自然三维RoPE、共同输出租约误循可选alias、Apple拥有租约的实际stride、Vulkan singleton行距误选转换。
旧StateVersion仍占其背存；统一冷预建10个独立代Invocation，不绕过原生固定输出保护，也不承诺旧lease在世时同producer槽可重放。
有限F32状态与最终QAT／cache逐字检查，四处BF16中间最多1ULP；未验证完整投影／模型、prefill8分支、Linux、capture或吞吐。
本轮清理镜像并保增量对象／共享缓存的范围见[实录](../correctness/ga-ratio4-producer-reference-2026-10-21.cleanup.json)。
本批[GA角色边界与通用缺口](../correctness/ga-bf16-indexer-state-reference-2026-10-21.json)四端共17项通过、0跳过：12项新增、5项相邻检查；共享普通Graph共52次成功提交。
补TopK(0)、I32 GatherElements及CUDA准入／Vulkan原码恢复，Vulkan还补真实batch pitch的BF16 RHS内部布局适配。
B2/H64/D128、active515/pitch520、Top512，score保三处BF16边界、同代StateVersion／Head与旧结果；省略中间舍入会改变4104个参考值。
上述17项是post-QAT消费者证据；其前置F32 compressor另由本轮ratio4旅程覆盖。完整attention／模型、Pro Top1024整链、Linux与吞吐仍未验证。
清理本轮镜像并保增量对象／共享缓存的范围见[实录](../correctness/ga-bf16-indexer-state-reference-2026-10-21.cleanup.json)。
普通E4成对／独立生产者经原CopyBatchUpdate进入pending页读取的[四端旅程](../correctness/e4-paired-pending-reference-2026-10-21.json)已通过：4项新增、3项邻居，0跳过；其中新E4共9个成功旅程、54轮更新和18个预期失败旅程。
另修共同copy槽在尚未提交复制时漏观察已绑定生产者的非阻塞退休检查；pending／失败仍拒绝撤回，不增加成功路径host等待。CPU／CUDA旧孤立测试已登记常规CMake。
可重建测试镜像的删除与增量对象保留范围见[清理实录](../correctness/e4-paired-pending-reference-2026-10-21.cleanup.json)。
MLX head prefill所需的非转置affine乘法已沿原descriptor、冷规划和四端程序落地。
同一压缩W保物理K打包／分组，两方向覆盖H64/N512/K192、padded query显式小Copy与双槽旧结果；
[13项新增与13项邻居](../correctness/mlx-weight-orientation-reference-2026-10-21.json)通过、无跳过，CUDA要求并实际使用捕获。
另修CUDA原生MLX遗漏的捕获标记及Vulkan输出组初始累加污染，保原数学与逐字节检查；
后续已按固定首shard偏移取到GLM31首层embed_q／unembed_out六个原tensor，合计7.875MiB；
[真实权重消费者](../correctness/mlx-head-payload-reference-2026-10-21.json)核decode独立attention边界、prefill共享latent、head恢复与双槽旧结果。
Apple由此暴露并修复原生非连续结果的只读byte投影，保实际stride、完整span、借用／拥有权限及退休；不增加Copy。
实际原BF16 scale允许负数；提前将解码W舍入BF16会使17702个输出不同，四端仍保原F32合同。
只签首层两角色；其余77层、整MLA／模型、原生低dtype全域等价与吞吐仍未签。
本批[6项新增与8项邻居](../correctness/precise-role-consumers-reference-2026-10-21.json)通过、无跳过：原F32 scale逐位冷导入、四端动态FP8消费者和Apple完整通用mixed MoE。
固定vLLM把`FP8_PB_WO`选为动态A128，不能凭名字或缺少activation字段认作A16；ModelOpt另一个BF16解码参考先舍入SF，数学分别记录。
新F32消费直接读压缩carrier，不展开NK浮点权重；未签所选payload、原生vLLM逐位等价、整模型或吞吐。
通用分立expert→有界working-set桥已沿原WeightSlots／InputSource落地：CUDA、Vulkan、Apple均实际执行512专家域取4、
NVFP4真实N2048/K8192与CT奇数N7打包，权重／scale／map同代原子绑定，旧结果与失败恢复保留。
[18项新增与17项邻居](../correctness/expert-working-set-reference-2026-10-21.json)通过、无跳过；CPU验证共同装配／冷打包／绑定，三GPU验证实际资源旅程。
这是交付3的一部分，尚未签其它codec的精确角色旅程、整制品、Linux、质量或吞吐。
Inferact micro32的固定reader已核：F32 SF数值转E8M0，低偶高奇、gate/up连续二半，BF16 A不QDQ；
同一MXFP4入口新增W4A16模式，四端[12项新增与14项邻居](../correctness/mxfp4-weight-only-reference-2026-10-21.json)通过，无跳过。
原exporter／OAS与整制品payload未取得，所选reader参考、精确制品绑定和资源桥分别记录，不再把reader数学写成未知。
uniform compact MLA E4已接CPU、CUDA、Vulkan、Apple的原paged/ragged路径，缓存512／576字节一行、V同根前512项。
[18项新增与10项相邻回归](../correctness/compact-mla-e4-reference-2026-10-21.json)通过，无跳过；其中18项包含16个四端Graph、1个CUDA精确scale见证和1个CPU schema检查。
原ModelOpt QDQ与runtime缓存格式的归因保留在[来源复核](../research/compact-mla-e4-contract-2026-10-21.json)。
CUDA完整kernel在4070实跑，576维共享内存99KiB；Apple用无损原Q存储使共享tile降至27KiB。
原CUDA MLA增长KV仍需冷重规划；原生GPU delegated ragged无公开page-plan报告，只签warm/fresh与寿命，不宣称无冷重规划。
动态per128 scale＋BF16 KPE的656／528字节DSA另合同；本包未验新E4 capture、精确制品payload／全模型、Linux、质量或吞吐。
专门的 [decode 性能规划](../../developer/performance/decode.md)维护 D1–D4 的动作、对照和停止条件，本页只保留顺序与当前事实。
仍有必需功能或交付遗漏时，当前 goal 保持未完成；说明限制或取得一次局部性能收益不能代替补齐。

### 本轮收工含义

8／8只表示下表当前量化能力包完成，不能单独代表所有算子、融合和架构重构完成。
第1–5步收工还须把旧功能审查遗留与模型总表逐项对齐：目标范围内必需计算、状态、资源与失败行为有可执行路径；
已有Provider提供且数值等价的成熟算子／融合没有漏接。发现仍属目标范围的必需遗漏，就修复并归入相应能力或交付包，不能用当前计数排除它。
抽象与职责收口审实现：模型语义与生成策略留引擎，冷路由／融合／布局留规划，热执行与资源寿命走共同核心，
后端只处理自己的数学和原生资源；普通算子、融合或模型组件扩展不应反复修改共同调度、租约与结果发布。
最终消费者检查再核功能、接口和实际性能。预期结论是“目标范围内已知必需缺口清零，本轮重构完成”；
MTP／mmproj仍按第6／7步交付。新增物理融合、更多几何或硬件专用优化按明确的收益与取舍记录，不能由“能推理”推导已优化完。

步骤 3 的功能覆盖与当前 BF16 Qwen decode 吞吐分别报告。Grouped NVFP4 等专家／其它格式接线补齐后，
当前未使用这些格式的 Qwen 图不会自动提速；只有进入实际执行图的融合／搬运修复才能影响它。
低比特权重可另行比较带宽与质量，不能将不同量化条件的速度混入当前 BF16 基线。

## 8 个模型／量化能力包

范围保持 8 个正式模型、已锁定的 24 个官方／GGUF 基线配置及其 Unsloth/Hugging Face 主流量化；现有 33 条扩展制品来源是当前核查输入，
见[模型—制品—算子总表](../../implementation/model-operators.md)和[量化明细](../../implementation/model-quantizations.json)。身份表24条基线已附锁定revision的HF链接。
相同格式与计算规则的来源归并验证，不按仓库数量重复计数；确认属于正式模型主流需求的新格式或计算规则仍须补入并说明。
旧 DeepSeek preview 不回归范围。

| 能力包 | 状态 | 当前缺口与收工内容 |
| --- | --- | --- |
| GPTQ INT4／g_idx | **本包已完成** | 原四端口Q/Z/F16 scales/g_idx已接四后端decode及普通闭合packed候选，zero nibble加一且15→16，先F16乘法再转输出；全部ID先检查再寻址。[48项数值／寿命及邻居回归](../correctness/gptq-reference-2026-10-20.json)通过，含CUDA实际捕获。Q35完整150组、Q27第二分片73组header仅为角色依据；未签payload／整模型、Linux或吞吐 |
| 校准 static FP8 | **本包已完成** | [固定producer／真实header与数学](../research/static-fp8-stored-scale-contract-2026-10-20.json)及四后端普通FP8 Cast／有界冷候选已接：激活SATFINITE RNE、F32 code-dot、分别乘SA/SW、末端F32/F16/BF16 Cast，直接读原W而不留NK浮点bank。含原图逃逸、IEEE scale、捕获／寿命及邻居的[71项回归](../correctness/static-fp8-reference-2026-10-20.json)通过；修Apple原图标量FTZ与CUDA相同域默认选路。只签存储scale软件参考，未签payload／整模型、Linux、原生scaled_mm或吞吐；KV／conv仍归精确绑定包 |
| Grouped ModelOpt NVFP4 | **本包已完成** | 原 dense 前五角色 A/W/SF/WG/IG＋IDs 末端；按选中专家的 IG 分别 QDQ。CPU 8 项与三种 GPU 各 4 项通过，[数值／寿命实测](../correctness/grouped-nvfp4-reference-2026-10-20.json)共用检查，直接读压缩专家 bank，QDQ 本体为 2×T×S×K 字节；[设计与 header](../research/nvfp4-grouped-expert-contract-2026-10-20.json)只作角色依据，不代签全模型、Linux 或硬件 FP4 GEMM |
| CT local-dynamic NVFP4 | **本包已完成** | [固定header与producer](../research/ct-local-nvfp4-contract-2026-10-20.json)已核，两种FP4预舍入以数学字段区分；原WG/IG直接除法、BF16局部scale边界和zero→0.125保留。dense＋selected-grouped复用四后端原执行体系，[27项数值／寿命与邻居回归](../correctness/ct-local-nvfp4-reference-2026-10-20.json)通过。只签有界软件参考；分立expert到working-set的资源桥归交付3，未签整bank／制品、Linux、硬件FP4 GEMM或吞吐 |
| Quark MXFP4 W4A4 | **本包已完成** | [固定来源与数学](../research/quark-mxfp4-contract-2026-10-20.json)已落实为`Mxfp4Linear`／`GroupedMxfp4Linear`真3／4端口，原U8相邻nibble、E8M0 group32、BF16操作数与明确零／极值规则沿四后端原FP4执行体系承载；共享A1／零slot步幅只QDQ每token一次。含SF0、selected SF255／非法IDs、tiny BF16、双槽和失败重建的[37项数值／寿命及邻居回归](../correctness/quark-mxfp4-reference-2026-10-20.json)通过，无跳过。只签软件参考，不签原HIP／Triton全域等价、整制品、Linux、硬件FP4或吞吐；分立expert工作集桥仍归交付3 |
| CT channel/token QDQ 的其余底座接线 | **本包已完成** | Apple补普通F32/F16/BF16↔Bool及Select mask归一，Vulkan补同类Bool与BF16/F16↔E4存储转换；沿原普通图保BF16逐token QDQ、逐channel权重舍入后MatMul，不替换为另一FP8点积。[20项四后端数值／寿命及邻居回归](../correctness/ct-channel-token-cross-backend-2026-10-20.json)通过，含CUDA实际捕获、±0／tiny／NaN原位与广播。只签已核CT BF16合同，不签其它producer、整制品、Linux或吞吐 |
| CT packed grouped／expert 空间承载 | **本包已完成** | 精确Decode[/Cast]→Reshape[E,N,K][/Cast]→GroupedMatMul冷收缩复用四后端原执行体系，直接读选中Q/Z/SF，不展开ENK浮点bank；F32 A／输出、真实W舍入、完整IDs先验与三轴stride保留。含sym128、sym512/F32 scale和noCast的五组见证、双槽、失败恢复与发布decode逃逸，[四后端定向及邻居回归](../correctness/ct-packed-grouped-space-2026-10-21.json)通过。CPU普通默认即可选择；Apple另修只读borrow与Copy快照。只签原flat EN行Z布局的软件路径，不签各expert独立padding的盲拼；分立expert工作集桥仍归交付3 |
| 固定来源的精确角色绑定与资料收口 | **本包已完成** | [16条固定来源复核](../research/fixed-role-binding-2026-10-21.json)及后续有限消费者已补Kimi group512 W4A8、CT BF16 block128、grouped FP8 power2、普通分页E4 KV、uniform compact MLA E4、真实E8M0 macroblock128冷导入、Inferact reader W4A16、MLX head方向／原payload与mixed角色、GA压缩／indexer／mHC／FFN、Kimi token/channel FP8及GLM block128 scale策略；具体四端证据与边界见上文。MLX六制品164个header、Flash37338份F16 scale/bias及12387个5bit triple已核。原exporter／OAS、其余payload／整制品、具体expert工作集旅程及默认路由／热路径常量复用不由本包代签，仍进入下方交付包 |

已经落地的四后端 CT packed 线性、MLX affine4/5/6/8／共享A／mixed输入、dense ModelOpt NVFP4参考、
四后端 CT BF16 channel/token QDQ不重新作为缺功能计算。完整普通 mixed-width MoE现已四端执行；Apple本批补齐原通用5/6/4bit、BF16 metadata／F32 A组合，不能代签具体GLM整层。
完整FFN／expert单kernel、更多shape融合、CT tensorcore、NF4 packed和hardware-native FP8/FP4属于另列的优化候选；
不能把未物理融合写成普通图不能推理，也不能以另一量化数学替换现参考来制造覆盖。

33条 `artifact_execution_verified=false` 仍如实保留：这表示未执行这些整制品，不能解释成基础库完成度0%。
当前明确的完整测试引擎交付只有Qwen3.5-2B；其余模型以精确角色、数值、子图和资源旅程证明底座能力。
角色header、算子执行、整制品执行和质量／吞吐是不同证据。

## 3 个交付包

| 交付包 | 剩余工作 |
| --- | --- |
| 当前源码与安装消费者 | **本包已完成**：Linux当前闭包、232个安装头及迁移后独立CUDA／mixed消费者通过；Mac当前CMake producer构建／安装、迁移后两公共样例与affine参数回归共4项通过、无跳过；[Mac安装证据](../correctness/apple-current-sdk-reference-2026-10-21.json) |
| GPU 状态与结果寿命 | **本包已完成**：[三端原生256B消费者共15条旅程通过](../correctness/state-native-lifetimes-2026-10-21.json)，含成功clone、旧取消／失败control跨同槽复用隔离、确定的observer边界与原pending／close／退休邻居；不代表全MTP引擎 |
| 架构、能力域与接口收口 | **本包已完成**：[有限交叉审查](../correctness/compute-graph-delivery-review-2026-10-21.json)未确证新的必需primitive／共同executor／成熟等价融合漏接；总表与接口说明已同步，保精确几何、payload与未验证域边界，不新增模型名单或缓存框架 |

第三包已核对原R04–R12普通语义旅程：KDA／MLA、正式GA的BF16 indexer／压缩状态／mHC，以及普通MoE组合。
Apple完整通用mixed-width MoE、GA四端BF16 score／Top512、ratio4 F32压缩生产者、typed mHC、完整routed／shared FFN及GLM混合精确角色已补上述有限闭环；其边界由各自来源与实际消费者限定。
旧R09的两条失败属于已排除的preview制品，原数值失败保留，未冒称修复；当前正式GA与其身份／数学合同不同。原整链误差也不是当前普通F32 RoPE的隔离反例。33条整制品未执行不改为通过，精确payload与引擎驻留／淘汰政策不扩成新的底座必修清单。
Apple Graph常量的affine SF／bias已复用原prepared-cast owner冷提升，按来源去重；两入口共用实现，保原logical F16/BF16及输出，已知派生存储计预算并随真实数组owner释放。[4项检查](../performance/apple-affine-constant-preparation-reference-2026-10-21.json)全部通过，含16次成功提交、预算拒绝／回滚与旧结果；native CompileReport新增两个字段，实际布局使用者已重编。后续迁移后的当前安装SDK重链原两项参数回归，也已通过，未混入旧Hyle对象或诊断observer。
[原诊断](../performance/apple-affine-metadata-preparation-2026-10-21.json)的常量对照每次2个metadata pending转换现降至0，动态WeightSlots对照仍保每次12个的原managed路径；未测物理dispatch／分配／吞吐，不为此新增全局缓存。冷bind早于source-ready，不得提前读pending权重；动态准备优化必须先有真实不可变权限／producer代次和代价依据，不能按地址或allocation generation猜内容。
上述三GPU状态及两轮Apple测试封包后已按根`test.ps1`清理5个Windows、6个Mac镜像／map／临时DLL链接，保2＋8个对象、SDK和原始数据；[精确清理实录](../correctness/state-native-lifetimes-2026-10-21.cleanup.json)。
上述GLM block128 dense SF逻辑展开已由紧凑索引修复；显式全权重解码仍保原展开语义。物理分配频率／流量与吞吐仍未测。
Kimi FP8逐token／逐通道候选已验证，但CPU／CUDA默认编译未选择它，仍展开NK浮点权重；须结合真实角色的时间与空间代价审默认选路，不能以测试定向候选代签自动融合，也不未经测量全局调整Provider优先级。
本批[局部代价对照](../performance/outer-fp8-route-cost-2026-10-21.json)确认3MiB workspace预算能自然选到压缩候选：M1／M2显式工作区CPU约3.8MB→36／72KB、CUDA约3.8MB→43／86KB。CPU单进程M2变慢33.58%；CUDA三进程中位更低，但有尖峰，前两窗均值略慢。默认覆盖的具体原因是早起点GGML region占用候选内部Cast。本批不是Qwen decode、总私有内存或流量测量；原样本保留，5个镜像／map／临时DLL链接已清理，保2个对象及共享缓存。
[分段与一次API trace](../performance/outer-fp8-submit-tail-2026-10-21.json)确认slot0在计时前call2、slot1在warm样本1的call4触发原GGML迟capture／实例化，后续只launch；default／预算路线分别有5／2个GGML程序。原尖峰样本保留，不把删除初始化样本当新吞吐证据，也不将实例化API之和等同全部submit超额。本次尖峰区间未出现设备／锁页分配或事件池扩块，无需修改该机制；本批保留默认路由，CPU／CUDA资源预算路径已可用。归档后经根`test.ps1`清理3个诊断镜像／map／DLL链接约198MB逻辑文件，保3个对象、原trace与共享缓存；[清理实录](../performance/outer-fp8-submit-tail-2026-10-21.cleanup.json)。
Linux当前CPU＋CUDA的Compute／Graph／progress已完成真实构建、安装和单实例迁移；232个安装头与当前选定源一致，原独立CUDA及CPU＋CUDA混合消费者均实际退出0。[完整安装证据](../correctness/linux-current-package-install-2026-10-21.json)保留477个必要构建任务、GCC初始化首红与修复，以及WSL误选Windows nvcc后的标准Toolkit_ROOT修正；无额外下载。
Mac也已按当前CMake完成Compute／Graph／progress的实际producer、安装与单实例迁移；751份选定源一致，迁移后的公共Compute／Graph样例及原affine两项回归共4项通过、无跳过，含9次affine提交与预算拒绝。[完整Mac证据](../correctness/apple-current-sdk-reference-2026-10-21.json)保留必要的native重建、缺失container私有边与验证副本漏同步notice的首红，原路径已迁移，实际消费者只链接当前安装库；不将loose对象回归充作安装证明。两系统producer、SDK与消费者保留长期增量复用。
专家权重逐tensor分立时，强合512个expert可形成4GiB W与512MiB SF；现已用原catalog／WeightSlots／InputSource补通用碎片装配、
global/local ID映射、上传范围记账及双槽旧结果／失败退休。实际只分配4个专家的压缩bank，不分配完整global bank。
各codec／模型的精确资源消费者仍须分别闭环，不能从两种代表格式外推全部制品；原始buffer容量限制、U32寻址与保守INT32门限仍分别适用。
固定来源复核还确认GLM Flash的37152个专家碎片需129组有界bank绑定；GLM另有156个attention head-bank，
不能把rank3直接当成专家路由已接。固定GLM32默认loader保留F16 scale/bias，与BF16 A在原MLX中晋升F32；这与Hyle该mixed输入合同一致，
但不能外推显式dtype转换或同低dtype原生QMM逐位等价。MLX head的非转置packed prefill接口缺口已补，
H64/N512/K192同bank两方向及首层原payload的独立unembed_out角色已四端执行；GLM32混合routed／shared角色另由上述消费者覆盖，其余层、完整attention／mHC／模型仍须分别核定。
这些是交付范围，不要求扫描所有硬件组合。原生资源退休以实际Completion为准，Scheduler空闲不能替代GPU退休；
Vulkan固定backing、Apple独立生成Array和CPU/CUDA调用者输出保持各自公开ownership与旧结果保护。

### 新来源的落点

2026-10-20核查了两仓实际固定提交与源码，[调研及来源](../research/deepseek-runtime-inspiration-2026-10-20.json)保留具体实现、适用限制与下一动作。
下列是设计借鉴与待验证优化，不是新增已确证bug或吞吐收益；没有增加另一套执行器或支持模型。

| 来源机制 | 归入步骤与下一动作 |
| --- | --- |
| DeepInfer：上游直接产出后继所需布局，权重／scale冷准备 | 第3步static FP8／精确角色接线时检查一条真实producer→consumer的Cast／repack与常量复用；等价数学才沿原region/provider收缩 |
| DeepInfer：投影与residual直接落最终输出 | 第5步D4前有限检查一条真实链的等价epilogue与实际搬运；先保原BF16舍入、读者和alias，再按实测决定采用或明确关闭本轮候选 |
| DeepInfer：GPU保留先前token／接受长度并重叠CPU处理 | 第6步先完成真实Qwen MTP与对照，再看metadata往返与空隙；DSpark另需训练的ConfidenceHead，本地Qwen的15个MTP张量没有该头 |
| Recipe：协议／prompt／token流／图片预处理分层 | 第4／5／6步审消费者token-ID、stop／finish与异步关闭边界；第7步另做真实encoder／mmproj，HTTP与模型生成政策留消费者 |
| TensorRT-LLM：冷准备共享量化、按实际形状／设备选择kernel及稳定graph metadata | [固定源码研究](../research/tensorrt-llm-inspiration-2026-10-20.json)归第3／4／5步；已有FP8共享prologue、有限图族不重做，Quark共享A按实际source独立性复用。D4前只对当前consumer真实热点核选路与搬运；SM89 BF16普通投影仍须实测，不由SM100 low-M或FP8 CUDA-core入口推导收益 |
| TensorRT-LLM：GPU token依赖与CPU结果处理重叠 | 第6步完成真实MTP后核必要同步与状态发布；第5步仅按trace核当前host-token边界的实际空隙。保stop、失败、旧结果与Completion退休，不为重叠越过token依赖 |

有限图族、稳定metadata与小replay入口已与现架构对应，不另开重构；共享池也须保Hyle独立slot和旧结果寿命。
SM100多卡吞吐、Wide-EP及V4.1特定的近似SWA replay不外推为4070单卡Qwen收益。

## 当前事实与依据

- uniform compact MLA E4已沿四端原执行体系接通，Q／输出类型与一种共享cache解释分开；V保持CKV前512项的原行距view。补Apple／Vulkan原ragged单bank入口及共同facts过度限制，没有新executor。18项新增与10项相邻回归通过，含真实KPE敏感性、CUDA精确scale、双槽旧结果和warm/fresh；有限数值及资源边界见[源码与实测](../correctness/compact-mla-e4-reference-2026-10-21.json)。CUDA原KV长变化cold replan、delegated ragged report缺省、整制品／Linux／capture／吞吐未签均保留，不由测试数量推出目标完成。

- E8M0 macroblock128冷导入已加入compute runtime：原E4字节保留，仅扩宽scale网格，调用者负责host源与canonical输出；几何、容量及跨resource的host地址alias在写前检查，映射／flush／unmap错误显式返回。4项host单测核全部256码、SF0及raw NaN、stride和失败；四后端原dense power2 Graph各1项通过，合计16次成功提交，含CUDA真实捕获、双槽和旧结果。共8项、0跳过；无整W浮点展开。未签精确制品payload绑定、全模型、Linux、质量或吞吐，[源码、固定来源与实测](../correctness/e8m0-block128-import-reference-2026-10-21.json)。

- 普通分页E4 KV沿原CacheEncode／页读取器补齐CPU、Vulkan、Apple，并与既有CUDA同fixture核对；四端16项新检查、CPU／Vulkan／Apple共6项plain／Q8邻居均通过。原E4字节按引用页解码，K/V各自F32 scale，不展开整套缓存；固定Q1增长与fresh逐字同，双槽和旧结果寿命保留。另修mixed CPU补丁合成的锚点去重与固定计数，以及旧Q8测试先allocate后bind所需双缓冲。只签该批有界普通页读取；普通E4成对pending生产者链已由后续批次补验，compact生产者／具体制品scale绑定、Linux及吞吐仍未签；[源码、原始失败与实测](../correctness/paged-e4-kv-reference-2026-10-21.json)。
  封包后按根`test.ps1`回收Windows约1.23GiB、Mac约20MiB测试镜像／映射，保28＋8个增量对象及共享缓存；[清理实录](../correctness/paged-e4-kv-reference-2026-10-21.cleanup.json)。
- Grouped FP8 power2本批新增9项及邻居10项分实际窗口通过，0跳过；四端新Graph合计44次成功、8次预期失败。共同描述与dense同为显式rounding＋floor，不保旧4B入口；只刷新实际构造／抽取者及新增wrapper，复用SDK与CUDA原BF16 AOT。共享A每token只量化一次，CUDA scale列步幅直接交给原GEMM。另修Vulkan旧None scale的FDiv 1F32 ULP偏差，只改26份SPIR-V中的1份，原power2数学不变。CUDA F32 power在prepare明确拒绝；未签整制品、Linux或吞吐。[源码、独立数值与原始红绿](../correctness/grouped-fp8-power2-reference-2026-10-21.json)。
  封包后按根`test.ps1`回收Windows约974MiB、Mac约20MiB测试镜像／映射，保19＋3个增量对象、源码、结果和共享缓存；[清理实录](../correctness/grouped-fp8-power2-reference-2026-10-21.cleanup.json)。
- CT BF16 block128本批18项新检查及18项邻居分实际窗口通过，无跳过；四端新Graph合计116次成功、8次预期失败，另核CUDA三dtype两种广播的24次Direct提交。普通图先Gather选中的原E4 W／BF16 SF再还原，共享A每token只QDQ一次。补CUDA原Gather捕获与cuBLAS冷batch runs，保原SiLU融合并修typed BF16 FTZ；Apple修原BF16乘除tiny，Vulkan保原M1广播，非affine一般批量走原DAG MatMul＋结果concat，明确仍有结果复制。只签有限输入、N/K整128的软件数学／寿命，未测payload／整模型、Linux或吞吐；[源码、原始失败与实测](../correctness/ct-bf16-block128-reference-2026-10-21.json)。
  封包后按根`test.ps1`回收Windows约2.52GiB、Mac约58MiB镜像／映射，保28＋7个增量对象、源码、结果、PTX和共享缓存；[清理实录](../correctness/ct-bf16-block128-reference-2026-10-21.cleanup.json)。
- group512 W4A8本批10项新路径与14项邻居分相关窗口通过，无跳过；四后端新Graph各62次成功与20次预期失败，另保CPU Direct/schema。固定source、数值／寿命、初次错误及修复均在[证据包](../correctness/blockwise-w4a8-reference-2026-10-21.json)；没有新增BF16 Qwen吞吐或整制品结论。封包后按根`test.ps1`回收Windows约734MiB、Mac约19MiB测试镜像／映射，保17＋4增量对象、当前Qwen消费者和共享缓存，[清理实录](../correctness/blockwise-w4a8-reference-2026-10-21.cleanup.json)。
- 一份不可变有限图族已承载真实128/1及23/4/1请求，无热建图或padding；稳定调用者绑定跨layout保留，配置／投影权限仍重新准备。
  见[有限图族](../performance/qwen-finite-family-2026-10-20.json)、[绑定寿命](../correctness/layout-binding-lifetime-2026-10-20.json)和[Linux续验](../correctness/linux-layout-current-2026-10-20.json)。
- CT普通线性的四后端候选已接通，默认路线按实测保守选择：CPU单向量、Vulkan单行BF16可优先packed，Apple当前保原路线。
  源类型舍入、逃逸原分解与旧结果寿命见[CPU跨系统](../correctness/ct-packed-cpu-portability-2026-10-20.json)、[Vulkan](../correctness/ct-packed-vulkan-2026-10-20.json)、[Apple](../correctness/ct-packed-apple-2026-10-20.json)。
- MLX mixed链已避免额外整bank F32扩宽，合资格Vulkan不可变bank跨槽复用原存储；不合资格View／导入仍沿原复制路径。
  见[普通MoE与布局旅程](../correctness/mlx-moe-typed-inputs-2026-10-20.json)、[bank复用](../correctness/vulkan-mlx-constant-bank-reuse-2026-10-20.json)。
- dense NVFP4保留group16当前A块QDQ、scale先乘与BF16操作数边界；四后端参考路径已执行，不能代签native FP4 GEMM或其它producer。
  见[四后端参考](../correctness/nvfp4-cross-backend-reference-2026-10-20.json)。FP8转换／QDQ的最新CUDA capture续验见[原生storage Cast](../correctness/cuda-native-storage-cast-2026-10-20.json)。
- Grouped NVFP4本包已完成20项定向回归、0跳过，原dense邻居保留；封包后经根`test.ps1`回收本地约320MiB及Mac约39MiB的测试镜像／映射，保源码、对象、命令和结果，[清理实录](../correctness/grouped-nvfp4-reference-2026-10-20.cleanup.json)。
- GPTQ本批48项、0跳过；共用packed-artifact识别，prepared映射按实际端口数冷准备，热提交不分配元数据。
  原CT、FP8双输出与indexer邻居已续验。源码、命令、失败及最终结果见[四后端证据](../correctness/gptq-reference-2026-10-20.json)，不新增当前BF16 Qwen吞吐结论。
  封包核验后通过根`test.ps1`回收本地20项约453MiB与Mac两项约18MiB测试镜像／映射，保源码、增量对象与结果，[清理实录](../correctness/gptq-reference-2026-10-20.cleanup.json)。
- static FP8本批71项、0跳过；共同冷闭合证明与映射适配复用原执行体系，原图中间值发布仍保真实分解。
  Apple及CUDA标量原图红绿与同测试对象保留在[证据包](../correctness/static-fp8-reference-2026-10-20.json)；未新增当前BF16 Qwen吞吐结论。
  核验594个归档成员及54份当前源码后，通过根`test.ps1`回收本地40项约1.23GiB与Mac六项约56MiB镜像／映射，保源码、对象及当前Qwen程序，[清理实录](../correctness/static-fp8-reference-2026-10-20.cleanup.json)。
- CT local NVFP4本批27项、0跳过，含两种FP4预舍入、dense／selected-grouped、32次寿命提交与合法IEEE重建／失败退休。
  原ModelOpt邻居保留，Apple旧FP8 Cast拒绝断言更新为实际数值与非法算术检查；源码、首次失败和结果见[证据包](../correctness/ct-local-nvfp4-reference-2026-10-20.json)。
  独立核验352个归档成员／35份当前源码后，通过根`test.ps1`回收本地12项约327MiB与Mac四项约38MiB镜像／映射，保源码、对象、当前Qwen程序及SDK，[清理实录](../correctness/ct-local-nvfp4-reference-2026-10-20.cleanup.json)。

最新 D1 已消除通用配置更新后的重复物理 span 准备，保持当前 lease/seal、未知 identity hook 和原资源门控；
18 CPU＋6 CUDA 回归通过、64 份向量逐字节同。短／长两窗相对修改前都有约 1.1%–1.4% 的 decode 中位提升，
尚不签稳定全面胜出；精确基线、样本与比较边界只在[decode 规划](../../developer/performance/decode.md)维护。
[本批源码、复测和独立复算](../performance/qwen-prepare-bindings-2026-10-20.json)。
封包核验后通过根 `test.ps1` 回收 72 个诊断镜像／映射等文件和重复向量，共约 322 MiB；当前程序、对照、对象及 SDK 保留，[清理实录](../performance/qwen-prepare-bindings-2026-10-20.cleanup.json)。
[本窗原始数据与独立重算](../performance/qwen-decode-rebenchmark-2026-10-20.json)保留慢样本；比较边界与下一轮方法统一见[decode规划](../../developer/performance/decode.md)。
已有[首次准备诊断](../performance/qwen-first-prepare-2026-10-20.json)与[退休查询修复](../performance/qwen-retirement-query-reuse-2026-10-20.json)作为D1依据，不将它们替代每步准备的剩余归因。

此前独立内存观测中，Hyle／参考专用GPU峰约3895／4041MiB；Hyle private commit反而更高。
映射页、进程驻留集、commit和系统文件缓存不能相加，不能将较低进程驻留集解释为全机RAM节省。
[内存原始口径](../performance/qwen-memory-current-2026-10-20.json)保采样分辨率与加载方式差异；最终对照另按步骤5执行。

## 已确认范围与由来

2026-10-17的旧八条是本轮工作的由来。旧治理已删除，文档入口、构建配置和初始化脚本已经重整，
不再每轮重列为待办；日常使用[scripts初始化指南](../../developer/setup/README.md)、[文档导航](../../README.md)和[构建设计](../../architecture/build-system.md)。
MTP已具备状态版本、分支与接受前缀基础，见[状态版本](../../user/state-versions.md#mtp-接受前缀并继续计算)。
按用户2026-10-20补充，第6步须实现MTP版Qwen3.5-2B消费者并与llama.cpp的MTP推理比较。
本地Qwen3.5-2B配置声明一层MTP且共享embedding，实际safetensors/header与index一致，含15个`mtp.*`张量；
现纯文本消费者尚未使用这些权重。比较时核定双方实际MTP实现及草拟长度／接受规则，不把已有状态底座视作MTP引擎完成；mmproj按步骤7推进。
G04/G05按用户决定先视为解决，[公平实测与边界](../performance/g04-g05-bf16-partials-2026-10-17/same-fp8/README.md)保留。

“多机器”在本项目指多系统上的单机推理，不是多机协同。后端集合、执行模式与静态／动态交付分开表达，
普通用户不决定第三方算子Provider开关。精度、放置与预算由引擎选择；模型格式、tokenizer、生成／接受策略留在引擎，
可复用计算、搬运、执行、存储与状态能力归底座。CPU＋GPU及其它模型不能由一次Qwen单请求成功推导完成。

## 首个 Qwen3.5-2B 消费者的复用材料

[本地输入身份](../../implementation/tasks/compute/qwen35-2b-local-input.json)、[独立参考公式](../../../tests/compute_models/qwen35_reference.py)和
[早期参考条件](../correctness/qwen35-cli-preflight/README.md)仍可复用；当前消费者为纯文本、无MTP／视觉。

接线必须保留普通RMS的`1+weight`与GDN gated RMS区别、L2的`sum(x*x)+eps`、按head拆Q/gate及64维partial RoPE；
conv history、F32 recurrent state和KV实际长度逐步反馈，新请求真实重置。prefill、多token续算、decode分别核对，
不能用oracle state替换候选状态来掩盖累计误差。64份保留向量用于回归，不构成独立质量验收。
