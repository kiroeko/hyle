# 2026-10-22 AY之前的工作计划归档

> 原页：`docs/developer/current-work.md`。保留截至AY的逐批事实与限制，本轮最终结果见[最终快照](current-work-final-2026-10-24.md)，本文不作为当前待办。原页字节SHA256：`00517a70935d906e95a622c1fb58b452ff328233f008b78eba1db62321685add`。

更新：**2026-10-22**。进入“实际推理驱动的性能、功能与架构收敛”阶段。
范围仍是Windows／Linux／macOS的单机CPU、单GPU和CPU＋一张GPU；多模态限图片encoder／projector（mmproj）到文本。
Qwen3.5-2B是检验底座的真实测试消费者；架构还要承载其他已选模型、量化格式、算子和后端。

**当前优先级（用户2026-10-22再次确认）：先把MTP做完。** Qwen3.5-2B greedy K1列明主链、有限正确性检查及最终四臂对照已有，8项QKV调查已结束；普通decode回退已在AU恢复。**本窗普通／MTP decode为99.764／123.162，对参考105.224／139.532仍有差距，由性能主线分别调查普通共用成本与MTP增量，不据总TPS判为MTP逻辑错误。** 原GLM-Flash／Q27输入组件、Q24当前codec的生成参数NextN及DSpark投影target的两轮自然拒绝续算已核；其余适用原角色payload与完整状态组合仍未完成。具体剩项见下；连续批处理差距先记录，暂停独立调度扩展。图片encoder／projector完整链路仍等前三项完成后开始。

## 现在的位置

旧阶段的8个能力包、3个交付包已完成各自列明范围；**不代表架构已全量审完、所有模型已完整执行或全部推理功能已交付**。
旧清单、逐批结果与限制已[归档](current-work-through-2026-10-21.md)，不再按旧编号推进。
当前已有冷编译、不可变计划、独立Invocation、状态版本、分页权限与真实CUDA文本推理基础；下面三条主线均未完成。

**最近完整四臂对照见[页末](#最新完整性能对比)**：普通版Hyle／llama.cpp prefill为2349.076／1553.955tok/s，decode99.764／105.224；MTP为2137.322／1499.623、123.162／139.532。各3次中位数，独立内存旁采另列。两边都比AU更快，不能把Hyle约9%的窗口变化当成本批优化收益；两对续文从第19枚起不同，不称同token严格差距。

**当前MTP剩余项已按实际缺口拆开。** [前批同状态控制](../correctness/qwen-mtp-current-numerics-2026-10-22.json)保旧52份状态，并发现T1／T2的hidden／分数差异；真实quota1连续20输出与普通版相同。[最新局部对照与完整任务](../correctness/qwen-mtp-quality-prefix-2026-10-22.json)进一步逐字节复现P154整图的首层新QKV行：embedding和BF16归一化输出相同，归一化也与独立FP64参考舍入相同；普通cuBLASLt BF16投影的6144项中8项不同。首次差异已定位至这一步，早于conv／GDN及状态更新；不能由跨形状浮点差异直接判算子错误，L171及其它来源未由此局部实验代签。

按用户要求先分开重复性、MTP逻辑与数值合同：同一局部消费者共3个独立进程、6次Graph，每种形状的embedding／norm／QKV都逐字节复现。它不调用draft／verify／接受／回退或缓存状态，8项差异仍稳定出现，因此这组证据指向单／双行计算形状，未见同一路径随机变化。普通cuBLASLt仍为BF16输入／输出、F32累加，各精确形状独立选择冷算法。8项精确点积核对中，6项靠近BF16中点，另2项强抵消；不能笼统归为最后cast的tie，更不能据此宣称全部cuBLAS内部归约已被证明正确。peer自然普通／MTP输出从第19枚起分歧只是现象；对应投影的实际dtype与局部对照另核，不用自然续文替代同位置计算证据。

本批进一步核完：Hyle两路各6144项按原cuBLAS BF16回归式均0项超限，未修改阈值；BF16舍入区间也均与标准F32累加误差界相容，这只是必要相容检查，不证明厂商内部具体归约。固定peer原GGML前缀实跑T1／T2各两次，同形状全部字节相同；跨形状embedding／F32 norm相同，F32 QKV的6144项不同。单行结果更贴近自身F32输入点积，双行更贴近先将该输入舍入BF16后的点积，与固定源码的向量／矩阵选择相符。已核GGUF QKV原字节及norm转换与Hyle原权重一致；该局部图精确host展开embedding，未冒充完整llama调度／融合或同Snapshot复现。两边输出dtype不同，不能比较“8项对6144项”来排精度高低。

**本次8项调查结论：稳定的普通形状相关浮点差异，未发现越过既有数值判据的计算错误，也不是必须执行MTP逻辑才出现。** 不把强制逐位一致列为待修缺陷，不强改现有精度政策；后续若实际发现合同、状态或额外精度损失错误仍修。此结论不穷尽所有层／形状，也不声称已解释跨引擎续文分岔的全部因果链。

测试消费者已移除人为128枚输出上限，改按真实上下文和资源准入，以减法校验防止旅程长度溢出；3项早期拒绝检查通过。原P154完整编码任务使用相同2048输出额度、C4096及自然EOS，Hyle普通／MTP、llama.cpp普通／MTP分别输出1220／1211／1825／1265枚后正常结束。四者都违反禁止Markdown围栏的要求，静态审查也各有确定算法反例；未修补或执行生成代码，不称外部功能用例通过。用户确认小模型能力有限，这道题的四臂共同失败仅作辅助行为记录，不将写对复杂算法作为MTP完成门，也不继续加题刷通过率。MTP收尾以既有计算合同、draft／verify、接受／回退及完整状态正确性和实际速度／内存为依据。本批未改变精度政策或产品数学；后续当前四臂TPS和独立显存旁采已更新页末。

**[Apple／Vulkan普通Graph分页写入已实现，并完成本批原生检查](../correctness/native-paged-effects-2026-10-22.json)。** CPU既有追加回归通过；MacBook Air Metal与Windows Vulkan均实际通过原生Cast→双writer-only、Cast→写页→因果attention、缺显式依赖拒绝三例，核对COW、旧页完整字节与NaN尾部、双take成功门及过期提交。Apple实测暴露并修复了区域编译误用整图公开输出数量的检查；区域producer可为合法零输出effect图产出数据。实现复用原Runtime／Queue／页权限／退休链；本批原生数值范围为F16／page16／D64，未新增设备故障或在途取消注入，不能替代完整Qwen跨平台、其它dtype／形状及质量检查。已有F32末分数与GDN多快照不重复列缺；后续以真实消费者核其完整组合。

**[Qwen使用的四输入分页attention已补至Apple／Vulkan，并完成本批原生检查](../correctness/native-paged-capacity-2026-10-22.json)。** `compute::PagedAttention(capacity)`＋显式metadata复用原NodeLayout、一次metadata上传、原页权限及完整Graph成功门；固定容量事实不伪造成owned recipe。两系统各实际通过同一有限Graph的三个BF16／D256子形状：物理／活跃Q为1／1、2／2、2／1，含原生Cast→双append→因果reader、COW、旧NaN尾部及inactive输出逐位零。Vulkan首轮暴露创建时缓存的执行端点与后续冷准备不一致，且BF16转换真实走异步IREE；现由固定原生端点、冷准备的同队列约束和已有原队列转换解决，保原通用转换默认与权限门。没有全量重编或新TPS结果；这不是完整Qwen／MTP跨平台验收。

**Qwen3.5-2B greedy K1的MTP主链及列明有限检查已做，MTP总任务尚未完成。** 三平台已有多会话／多轮／真实B2及有界重建结果；[最新完整前缀检查](../correctness/qwen-native-protocol-2026-10-22.json)也已通过：CUDA、Apple、Vulkan各15次Graph、3次完整命中、534项向量对照逐字节一致、7次旧前缀全字节检查；负分支保14份事务，退休后原槽重试，最终版本数归零。这是各后端与自身相同输入分块参考的对照，不是跨后端张量逐位相同或完整质量通过。

已完成的Radix、LRU、COW、B2和完整前缀不再列缺。

**[本轮取消／退休及额度恢复已通过有限检查](../correctness/qwen-native-protocol-2026-10-22.json)。** 修复真实消费者遗漏原Vulkan Context.collect，以及原生IREE清洁取消被永久留在仅阻塞drain可推进的状态；现在取消时沿原队列、原Run fence提交一次真实终点，保全部回调及managed工作退休门，复用前恢复正常fence。CUDA与Vulkan各实际通过6次Graph的自然lag1／T2取消、丢Completion、14份权限保留、旧完整状态字节不变、原槽重建，以及成功terminal后的策略异常／重试。Apple同例通过，明确证明的是原Metal命令已Committed及原Runtime FIFO已接纳Model Job，不代表模型GPU命令已提交或抢占。其测试阻塞器原先在通知host前等待Scheduled，真实触发5秒Metal超时；现按真实Committed状态通知，仍等原生完成后释放。

Vulkan真实完成后再施加纯quota压力，finish拒绝且52份旧状态／ledger不变；释放后费用精确回到53,753,907B，原会话／槽／base重试与无压力参考全载荷逐字节一致。缓存额度不是原生私有或物理显存峰；本批没有设备故障注入、新TPS或全形状／全制品通过结论。曾怀疑的同delegated lane聚合候选经真实fixture检查不符合现编译器不变量，已撤回，不列为修复。

余项：

- **本次投影差异已定性**：不再把这8项或复杂编码题共同失败作为MTP阻塞项。已有有限计算／完整状态检查继续作为当前实现依据；实际暴露的draft／verify、接受／回退、状态或计算合同错误仍须修复，不要求跨形状逐位一致。
- **适用制品**：[本批Q27／Q24检查](../correctness/qwen-original-mtp-input-2026-10-22.json)已接Q27主GGUF原EH（Q6_K）／两份F32 norm到CPU／CUDA输入组件；Q24当前UD-IQ4_XS／Q8_0实际codec已对齐，生成参数的norm／carry和NextN链通过。此前[GLM原输入及DSpark投影target自然拒绝续算](../correctness/mtp-natural-journey-2026-10-22.json)保持列明范围，原43层target未运行仅作覆盖限制。**下一批接Q24原K／V（Q5_K）＋K norm，共11,535,360B，或等价必要原角色**；原payload及完整状态组合仍需补齐。Qwen BF16 K1不能代签其它制品；不含MTP或明确排除MTP的格式不凭空列缺，不为每个模型另造完整引擎。Qwen多于K1的草拟与非greedy策略不在本轮承诺范围；其它模型按实际结构接入。
- **实际性能与空间**：单槽捕获回退已恢复，最新四臂完整速度及独立内存旁采见页末；相对参考的普通共用decode差距与Vulkan原生私有资源成本归性能主线，按实际热点处理，不混作MTP状态／接受逻辑缺陷。

**[Apple多会话与保留结果预算机制本批已有实际结果](../correctness/native-state-ownership-2026-10-22.json)。** 复用上批原消费者，三个会话各两轮，N3/R1为19次Graph；N1/R1实际3次淘汰／重建，29次Graph中含10次恢复、64行target及draft重算，归一1行lag；N2/R2也实际通过26次Graph。三组各会话最终26枚输出相同，覆盖追赶、T1、EOS后下一轮及真实B2；未导出全状态，不能据此签跨形状质量。
库新增显式`GraphOutput::storage_lease(retention_budget)`，只对选中的保留状态准入；不改Provider vtable、MLX deleter或执行器。原完整backing在同scope／resource内按可证明的身份去重，原池费用转入共享记录，沿原租约及native use退账。6项CPU检查和Apple两个实际原生用例通过，含两个独立import共享256B只计一次、超限保原权限、128B生成结果脱离Graph句柄后保活。该段是库机制完成时的范围；消费者接线的后续实际结果见下一段，原生执行峰值与MLX私有成本仍未知。
**[Qwen保留状态预算已接入并完成本批实际检查](../correctness/qwen-retained-budget-2026-10-22.json)。** 36份conv／GDN及MTP carry／prior在整批任何页事务take和head发布前统一准入；共享池幂等计费，ID／logits／诊断不纳入缓存。CUDA普通／MTP各一短流程保ID／自然决策／工作量，Apple普通／MTP显式256MiB均通过；Apple三会话两轮的N1/R1显式额度、N2/R2 auto也与上批26枚会话输出及29／26次Graph、3次恢复相同。auto原生不设字节额度，仍受N／R／版本限制，显式额度按实际池和完成后保留的backing准入，不套CUDA公式推测原生可驻留数。各模式统一报告Model全局固定库存、额外保留、live／peak与额度，子结果不可相加。
真实Apple原生计算完成后再用纯quota reservation占满额度，原finish明确拒绝；旧完整快照52组件、14布局和已提交ledger不变，候选／原槽真实退休，费用回到压力前40,669,235B。释放压力后原会话／槽／base重试，与无压力参考全部逻辑载荷逐字节相同；不是完整质量、物理峰值或GPU故障注入。Vulkan整storage block轮换／静态descriptor冷重绑的后续结果见下一段；其它模型／量化、原生前缀／取消组合、数值质量和热点仍未完成。本批无新TPS对照，页末速度表保持原窗口。

**[Vulkan完整消费者装配与本批实际结果](../correctness/qwen-vulkan-assembly-2026-10-22.json)。** 沿原Context／ExecutionSource／文件管线接同一Qwen消费者，模型数学不变。完整owned native backing新增明确canonical身份，在同scope／resource内对独立whole／slice／repeat导出租约只计一次；计费token先于payload释放，不改变F1身份／可读范围／只读权限。device Buffer IO复用一个有界staging及原CopyRecording，完整owning allocation的2字节BF16标量可用私有padding；普通未对齐子范围仍明确拒绝。8项common检查及RTX4070三项原生检查通过，含原[整block三代复用](../correctness/vulkan-retained-output-reuse-2026-10-22.json)、预算超额保旧租约／释放后重试、跨1MiB块的2MiB＋2字节IO及2字节标量。

**[Vulkan BF16矩阵路径已修复并实际越过OOM](../correctness/vulkan-bf16-linear-2026-10-22.json)。** 原native MatMul已有BF16权重／F32累加及最终BF16舍入，现按真实资格优先走该路径，避免上层wrapper抢先升宽整权重。自然permute权重复用原payload；dense逻辑RHS需要的不可变冷转置保持原dtype，native／promote两类pack按不同key复用。没有改模型数学、公共API、Provider算法或其它BF16算子的后备。

新永久消费者实际通过T1／T15各两次不同输入，三输出共12,288元素：原stored[N,K]→permute与同义dense[K,N]常量的BF16 MatMul、原权重直接F32 scores；按decoded-BF16独立FP64点积核数值、BF16舍入、F32近tie及极端指数范围，原权重／输入全字节和只读权限保持。首轮fixture漏符号输出spec，构图失败后已修；未改数值阈值。只增量编译两产品TU与小测试，未重建SDK／shader。

上批BF16修复窗口中，以同原P15／4、C64／N1／R1及256MiB缓存额度，完整320份权重已加载、T15单槽和T1双槽已初始化；普通版实际prefill／decode1／decode2返回1328／5675／11，第三次decode准备报`Vulkan batch native output retained`。本批复用旧单TU诊断对象，不加产品日志：新2349个region私有分配累计198,830,080字节，阈值≥16MiB的大CPY／CONT为0；其中冷T15为132,346,240，两个T1块各32,824,128，动态reader另计。旧窗仅前573成功就达7,760,350,944字节并OOM，两者覆盖不同，这些累计值不能当物理显存峰或TPS收益。

**[分页／ragged输出复用已补齐并完成本批原生检查](../correctness/vulkan-retained-paged-2026-10-22.json)。** 冷布局编辑只在真实退休后撤执行资格；同一未发布输出epoch可重复准备／失败重试，完整bind及最终seal才建立新private baseline。两reader复用原整owned block轮换、Run／Program／endpoint；旧readonly storage及提交门不放宽。未完成的存储选择先保rearm guard，不能在重试时误认成功。RTX4070实际通过capacity-paged／owned-paged／ragged三case，各同Invocation三代、whole／offset-slice旧字节／只读、释放首代后第三代复用原spare、销Invocation后仍可读；原ordinary复用／canonical预算／device IO三case也通过。fixture首轮误撤typed KV端口，已按原合同改为核拒绝并只撤普通Q，未放宽产品权限。

**当前Vulkan完整普通／K1的R1短流程已通过。** 同335／320原权重，P15／4、C64／N1／R1、cache256MiB、prefix off、commands；当前镜像两模式4枚greedy ID均为1328／5675／11／9637，与CUDA／Apple既有同输入相同。K1实际先拒绝再接受，3次Graph、19target／17draft行，最终frontier17／lag1；短流程不代表质量或速度验收。

前批三会话两轮N2／R2的首B2 checked Gather缺失staging预算已修：实际4B状态字及回溯确认native allocate_values。消费者native执行预算按实际device／staging／system身份去重注册；随后出现的原生OOM由本批继续定位。

**[串行消费者单槽已实现并完成三平台实际检查](../correctness/qwen-serial-carrier-2026-10-22.json)。** 两个同负载诊断均到13,336个region，成功graph私有容量累计1,594,108,032B；失败位于实际submit的scratch分配，分别为1,024／33,024B，不能再称载体创建失败。Vulkan失败快照报告device heap使用12,442,644,480B／size12,593,397,760B，原ggml requested账约1.50GiB；这些不同口径不能相减当精确归因，也不是4096分配上限。

Model原先全程只准一个Graph batch，却按全局奇偶为各形状保留两套Invocation。本批删除该轮换及双槽使用计划，每个实际准备形状只留一个串行槽；保原预算、旧结果owner、捕获失效和真实退休，不设后端特判。Windows CUDA／Vulkan及MacBook Air实际完成同一N2／R2三会话两轮：26枚输出、26次Graph、3次恢复、64target／64draft重算均同既有参考。CUDA真实阻塞队列的取消、14份权限、36份旧状态与carry／prior、原槽重试及成功terminal后策略异常也通过；四个永久probe已同步更新并编译。

Vulkan另通过整前缀命中（后两请求各0Graph）和分波后的部分命中（各复用15行，只算后缀1／8行），显式256MiB缓存额度内。原三请求同波会先统一lookup再发布，实际0命中；分波仅重组已有输入，未为测试改发布时机。CLI没有导出完整旧状态，不替代COW全字节、质量或原生取消检查。每region独立NativeWork上下文、scratch及驱动内部成本仍需收敛；本批未引入上下文池，也无新完整TPS／显存旁采，页末速度表保持改动前窗口。

最新四臂的64token续文已[离线解码](../correctness/qwen-existing-output-text-2026-10-22.json)：都是可读的归并排序代码开头，分别采用`Any`／`TypeVar`；它们是截断前缀，尚未证明完整代码正确或数值差异可接受。当前将浮点路径扰动视为工作假设，继续保留状态及首异定位，不直接签质量完成。

**当前已接通CUDA真实多请求MTP、多轮、前缀和有界重建；本轮补完既有F32末分数合同的cuBLASLt承载与默认路由。** [本轮实现与实测](../performance/qwen-cublas-scores-2026-10-22.json)保留`bfloat-linear-f32`：BF16权重及激活、F32分数，中间linear不变。Provider按每端口dtype处理字节与布局，原权重只改描述符方向；默认策略在候选成功声明支持后选择矩阵实现，不加模型白名单，消费者源码及公共API均未改。11项相关测试和5个实际诊断覆盖普通文本、单会话MTP、双会话lag1新输入及N1三次重建。普通固定输入的4份hidden与993280个F32分数逐字节相同；98份原始向量有限、各行argmax匹配，选定分数另核原权重FP64点积；这些不是完整模型质量验收。
此前cuBLASLt实现窗口的Hyle前／后64枚自然输出、接受记录及计算量相同。prefill中位2027.62→2341.58tok/s（＋15.48%），综合107.88→109.84（＋1.82%）；decode121.78→122.19（＋0.33%）范围重叠，**decode仍按近持平处理**。dedicated旁采峰4.171→4.066GiB，少108MiB，不冒充精确分配归因。原llama.cpp本窗decode140.31tok/s，但第19枚起续文不同、工作量也不同，不签同token跨引擎胜负。四项与启动／采样限制见页末。capture额外重录已定因，本轮已完成不重录的MTP稳态分析；其它后端／量化、质量与图片仍有实现工作。

**[当前MTP额外10次重录已定位](../performance/qwen-capture-cause-2026-10-22.json)**：临时诊断副本跑原P154／64一次，输出、34轮决策及工作账与当前镜像相同，没有张量导出。18record＝4组首次role／slot各2段＋10次真实旧range失效，另50reuse；首个变化端口中9次为prior／carry hidden、1次为layer0 conv history，同一resource／handle但子范围offset不同。当前Graph必须按新物理参数重录，未见native-plan、Queue或权限撤回导致的失效，不放宽身份检查。状态池的普通空闲分配与接受前缀选择解释这些地址变化；首端口证据不等于穷尽整段后续所有差异。
这些失效集中于第4～15轮，之后不重录。涉及的7个调用冷准备桶合计18.1936ms，包含其它准备；不能称重录自耗、可回收收益或全部decode差距。当前不采用强留状态块、额外影子复制或绕过失效；底层参数更新也没有现成的通用合同，不据此先扩API。定因只增量编译4个临时host TU、重用191／195链接输入，产品执行源码不变。

**[MTP稳态热点及单一候选已完成有限排查](../performance/qwen-mtp-steady-profile-2026-10-22.json)**：普通当前镜像只采1次Nsight，64ID、34轮决策及工作量与原镜像记录一致。第16～34轮每轮复用2段；19段中位GPU完成锚区间14.044ms，kernel和13.004ms、kernel／copy／memset活动并集13.149ms，非活动区间0.886ms。矩阵投影占这些段的**累计kernel时长92.466%**；大头是小行数WMMA与两个末分数投影。锚区间不是消费者精确wall，非活动部分不是CPU函数自耗，软件trace使用CUPTI13.3采13.4驱动；这些值不混入TPS。
唯一候选将普通BF16两行MatMul内部描述成共享权重的两个单行批次，未改输入／输出dtype、F32累积、精度政策或搬运。12项相关测试和原64ID／决策／工作量通过；各1进程未插桩初筛的decode为122.62→93.52tok/s、综合110.46→86.59，prefill2348.10→2351.07。**本轮拒绝该候选，不改默认，不以更多重复寻找有利结果**；这是有限筛选，无显存旁采、统计普遍退化或完整质量结论。下一性能切片集中普通小行数投影的实际冷算法选择，保持原精度／布局／寿命；图片完整链路按上述新顺序后置。当前普通／MTP四臂完整测速与旁采已更新至页末，不将初筛候选数据混入。
[此前固定状态控制](../correctness/qwen-mtp-batch-numerics-2026-10-22.json)定位的是旧BF16末分数政策下首异A：同一旧Snapshot当步R1／R2目标状态与输出相同，分歧跟已有prefill状态走，draft K/V保真实差异；不直接沿用为新政策的质量结论。正常重建仍保实际已消费输入，已返回pending与fresh预测分开。旧窗缺首次失效step／port身份，本轮只对当前镜像取得了直接归因，不仅凭同为10次便反推历史原因。
当前每进程加载一份权重、按批宽编译有限图族；同一Queue上提交实际等T批次，不是每会话复制执行器。缓存数量／字节准入、空闲淘汰与按原chunk重算已验证；旧R1逐字节结果仅对应下面列明的原分块／批宽。[完整分页快照](../correctness/qwen-paged-snapshot-2026-10-21.json)同时发布12份KV帧、36份conv/GDN租约与位置，单会话18份、四会话淘汰重建72份向量逐字节一致。普通Graph写页与变长追加的原实测、旧dense容量／隔离检查由该证据回链保留；不能据此宣称共享前缀或GPU并发吞吐已完成。
选择性`Invocation::unbind`的4项CPU检查、Apple九个及Vulkan五个相关host翻译单元已通过；后两者不代表原生运行或分页写入验证。本批[实际前缀与分支](../correctness/qwen-prefix-reuse-2026-10-21.json)新增52份hidden／logits逐字节对照：两条请求各复用129token后分别执行T1／T16，原会话之后续写仍一致；完整命中0Graph。N2压力下，已命中分支被淘汰后按原5段／132token恢复并核后续输出，真实版本及共享backing均在预算内。原18／72份回归另通过；不是GPU并发、多用户吞吐、独占写或异步回收验收。
[本批真实异步回收](../correctness/qwen-native-retirement-2026-10-21.json)将原执行链拆成prepare／submit／finish，目录撤下会话后继续保留槽、页和额度。阻住实际CUDA队列后，取消及丢弃公开Completion均不能提前回收；原生工作退休后才收集，旧前缀可继续推理。N1保持一份committed＋一份串行candidate；未增加并发执行器。原18／72与前缀52份向量回归通过，其它系统完整消费者仍未验。
[此前重复准备及输出发布扫描修复](../performance/qwen-retirement-publication-2026-10-21.json)保留。[最新执行资源复用窗口](../performance/qwen-shared-execution-2026-10-21.json)单会话decode为101.25对参考105.38，仍差约3.92%；多会话共享前缀综合91.99对95.12，慢约3.29%，相对本批前中位改善6.80%。当前有慢进程、范围重叠，不作稳定普遍收益承诺；完整四项及口径变化见页末。
[原生拆分](../performance/qwen-paged-native-cost-2026-10-21.json)发现分页路径每步7次Graph launch，额外复制本身仅约23µs。[前批可录制写页](../performance/qwen-capturable-append-2026-10-21.json)沿原Graph、Queue和逐页权限实现设备descriptor；真实99个完成ID／96次decode均核对，每步已降为**1次Graph launch、596个kernel、0次录制外eager D2D**。90次decode不重录。CPU回退、CUDA动态映射／分裂／Invocation隔离检查及142份实际向量逐字节通过，在途取消仍不能提前回收。descriptor上传有真实IO存储和冷准备等待；不能把提交数量下降直接换算成TPS收益。其它系统原生写页路径尚未完成；独占尾权的最新结果见下。

[独占尾权与Radix本批结果](../performance/qwen-exclusive-radix-2026-10-21.json)：4项新尾权租约与13项原租约检查、4项Radix目录检查、CPU／CUDA真实Graph通过；真实Qwen单会话18、四会话72、前缀／压力52份向量合计142份逐字节一致。完整命中仍0Graph，缓存、分支、重建与预算统计未变。新回收探针直接阻塞CUDA队列，取消并丢弃公开Completion后仍保留42,418,224字节状态backing，旧KV guard的范围与全部字节保持不变；最后读者释放后整页才可重用。不是所有系统原生消费者或GPU多人吞吐已完成。

| 主线 | 当前事实 | 本阶段要交付的结果 |
| --- | --- | --- |
| 性能 | 成熟引擎调研、新阶段首轮完整对比和准备诊断已做；已区分槽首用、拓扑变化与capture成本 | 找到适用机制，定位并修复实际慢项，用完整消费者验证速度、空间收益及取舍 |
| 功能 | CUDA K1及真实多请求MTP、prompt前缀、有界重建和自然多轮已接通；跨批宽质量、其它系统原生覆盖和图片链路尚未完成 | 可用的底座支持与真实测试消费者，明确引擎策略和库机制的边界 |
| 架构 | 新执行主干已实现，仍有待审查、待实现及被真实用例暴露的不足 | 性能成立、职责清楚，新增模型／算子／后端及替换模块时改动集中，公共调用方式一致 |

## 执行顺序与近期进度

| 顺序 | 工作 | 状态与下一动作 |
| --- | --- | --- |
| 1 | 容量参数和多会话设计 | **测试消费者已接上下文、驻留会话数、活跃请求数和状态字节预算**，支持auto解析与提前拒绝；状态请求字节不冒充全显存上限 |
| 2 | 多会话缓存复用 | **真实Qwen前缀、分支、完全命中、有界恢复及CUDA异步回收已验证**。同645逻辑输入／102输出、258实际计算token完成资源复用后对照；CLI等T真实批次已接通；宽多行TopK1优化窗提升10.67%；能力接线后的最新窗168.15对186.50tok/s，仍慢9.84%。三平台完整前缀与有限取消／退休对照、Vulkan额度恢复已通过；数值质量及当前实现完整性能仍在收尾 |
| 2a | 独占续写 | **CPU／CUDA及真实Qwen已实现并实测**。旧前缀读范围不变，同页唯一尾权只写未提交后缀；失败退休后rollback，分支无尾权时COW。真实阻塞队列取消检查通过 |
| 2b | Radix前缀索引 | **有界压缩树已替换线性目录并经真实Qwen复用检查**。精确token／模型／数学身份，LRU淘汰及多读者pin；800次目录oracle检查通过。只命中完整KV／conv／GDN快照，结构分叉不冒充检查点；新增检查点仍受版本预算及冷shape域约束 |
| 2c | 执行资源复用 | **[已实现并完成本批检查](../performance/qwen-shared-execution-2026-10-21.json)**。Phase／Invocation／IO／workspace归Model，按原串行实际并发整体借用；只准备实际shape／slot，Session停止不关闭共享池。142份向量一致；取消／丢弃Completion保留借用至真实退休，失败槽重建、提交后trace错误及另一会话继续均实测通过 |
| 2d | 多序列批处理与其它后端 | **[typed组合／投影已完成](../correctness/paged-frame-reframe-2026-10-21.json)，[真实Qwen B2／T1及取消／退休已跑通](../correctness/qwen-multisequence-2026-10-22.json)**。同一次Graph处理两份不同长度历史，返回各自KV／36份状态及next ID，之后可分别B1续算；152组向量指标已保存，跨B数值并非逐字节相同，不以greedy ID相同代替质量验收。阻塞实际CUDA队列后整批取消并丢Completion，仍保两候选；退休后原R2槽重建并重试成功。[CLI等T调度、整批N份committed准入与完整四项对照已做](../performance/qwen-cli-batch-2026-10-22.json)：8项host检查、真实B2、T1／T16分支、0Graph完整命中与N2淘汰重算通过；48份原R1向量仍逐字节相同。跨B重算保真实差异，不能签质量完成。Apple／Vulkan原生写页、N2／R2消费者及完整前缀对照已通过；有限取消／退休也已通过；质量、当前实现完整性能仍待完成 |
| 2e | Apple／Vulkan同图分页读写 | **owned／capacity原生路径及三平台N2／R2真实消费者已通过列明范围**。整权重展开、输出复用和冗余双槽引起的既有OOM已修，26枚输出及工作量同参考。三平台完整前缀、有限取消／退休及Vulkan实际额度拒绝／恢复已核；继续数值／质量与最终性能；未知执行私有内存和物理峰另计 |
| 并行 | 已知性能慢项 | 可录制paged写入、尾权小分配、selected-slot扫描及执行资源重复创建已修。单会话分页decode仍慢，descriptor上传依赖、随历史增长的页组织、cached-prefill首用／合法拓扑变化仍查；性能与内存最新窗口见页末 |
| 3 | MTP真实消费者 | **Qwen3.5-2B greedy K1列明主链、有限检查及最终对照已有，8项QKV调查已结束。** 三平台已有多会话／多轮／B2、有界重建、完整前缀、有限原生取消及Vulkan额度恢复。GLM-Flash／Q27原输入、Q24当前codec生成参数NextN、DSpark投影target自然拒绝续算与接受前缀边界0～6已核；适用原角色payload和完整状态组合仍缺。普通共用性能差距另归性能主线，不要求另造八个正式引擎 |
| 4 | 图片encoder／projector | **等待前三项完成后再开工完整链路**。保留已有297个BF16视觉张量（632.13MiB）、24层H1024 encoder及4096→4096→2048 merger的身份与设计调研。后续范围为RGB／patch→视觉Graph→文本embedding、多轴位置／delta、图片前缀身份及寿命；视觉使用原LayerNorm weight＋bias |
| 贯穿 | 架构审查与实现收敛 | 随上述消费者修共同机制，并检查其他已选模型／算子类型；不以Qwen跑通代替公共架构完成 |

先处理已有证据明确的缺口；根据实际依赖调整后续顺序。不会等全部功能结束后才审架构，也不会只改设计文档而留下必要实现。
近期顺序：**收完多会话缓存与多会话推理 → 收完MTP（共用缺口一起处理）→ 图片完整链路**。前三项当前仍有原生寿命、预算／恢复接线、数值质量、适用模型／量化及性能工作；采用相关边界场景和真实消费者收口。单会话延迟、多人总吞吐与缓存命中收益分别报告。

页末只保留最新当前四臂窗口，历史单会话／多会话表在各自证据及本轮归档中保留，不混合工作量推算收益。当前Radix复用已有完整KV／conv／GDN检查点；任意token处分叉不能只切KV便宣称得到可续算状态。多序列CLI已实现列明范围；其它后端、MTP和图片仍有必要实现，不是只剩验证，也不以一个CUDA消费者通过代替全部模型／系统完成。
本批公共页帧有27项arena、7项row-cache检查及CPU／CUDA双请求实际Graph检查；只选来源允许的行和目标页，旧帧权限与字节不变。统一Qwen执行改造的原单会话／多会话及前缀回归142份向量全部逐字节一致，容量与Radix的5项host检查及R1／R2原生退休检查通过。B1基线与B2的MatMul形状／合法归约可能不同：四组hidden余弦约0.999941～0.999964、logits约0.999937～0.999966，greedy ID相同；151／152组向量非逐字节一致。跨B误差、模型质量和速度分别报告，尚未据此设定完整模型质量阈值或宣称提速。
每批有实际结果就更新本页状态；性能样本与实验决策集中在[性能计划](../../developer/performance/decode.md)，避免重复维护进度数字。

**连续批处理差距已登记，按用户要求后置。** llama-server与本消费者都共享模型、隔离会话状态并真实合批，常规单context调度按批推进；独立Graph重叠不是多请求成立的必要条件。llama-server能在空slot释放后接纳新请求，并按token预算组合兼容prompt／decode；本消费者目前用预定wave和T／role／lag桶，未交付同等在线动态调度。调度政策属于引擎，底座须审查可变活跃子集、变长计算、状态和有界资源复用的承载。该项只记录，不在当前MTP收尾中扩建server或引入多执行器。依据：本地`llama.cpp/tools/server/server-context.cpp`的3462～3474、3942～3985与slot释放／defer链；本项目`tools/qwen35_text/main.cc`的`run_mtp_sessions`及schedule wave分桶。

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
- **MTP**：本地15个`mtp.*`张量与320个目标张量共用加载器；CUDA greedy K1已接单会话、真实B2、初始prompt前缀、生成后重建及自然多轮。目标T2与草拟缓存沿同一次Graph／Completion，各成员选择接受边界后一次发布整批完整快照；取消及丢Completion不提前释放资源。缓存保存已消费前缀，fresh预测和引擎已返回但未消费的pending分别保存，见[架构合同](../../architecture/compute-design.md#生成记录与缓存重算)。P154最新四臂结果见页末；8项QKV数值调查已结束，三平台已有上文列明的有限协议／完整前缀检查，不能代签全层数值或完整质量。当前功能剩项是其它适用制品的原权重／量化角色与完整接受状态接线。
- **图片投影**：预处理和模型图片语义归消费者；Hyle补真实计算、布局、特征与文本绑定、搬运及资源寿命。先做图片，不扩展音频／视频。

MTP早期完整轮次见[轮次证据](../correctness/qwen-mtp-round-2026-10-22.json)，单层数学和指定接受边界的[前批证据](../correctness/qwen-mtp-target-2026-10-22.json)保留：

- Snapshot增加独立draft K/V、frontier、lag、catch-up token及owned target carry／prior；14份KV arena、N＋2R份候选状态和2(N＋R)份hidden均显式计费。草拟prefill只算所需K/V，不执行无用attention／FFN。
- 普通Graph现支持没有attention读者的真实分页写effect，仍走原页权限、Queue、Completion和whole-Graph成功门。原CPU回归及无MTP的90份向量结果随证据保留。
- 真实首15个prompt ID的有限流中，自然accept2后target17／draft16、lag1；追赶再拒绝后两者18、lag0。旧head重新分支得到同样T2输出；quota1用T1并维护完整draft。另一个完整23token prompt遇EOS仅提交1行。这里只得到2个自然轮次，不能据此推普遍接受率。
- T1与T2行0保留真实数值差异，不能因greedy相同签质量。前批探针的capture已实际录制，当时未据此证明热复用或MTP提速；新自然窗口单列实测，不回写旧结论。

[本批退休与独立对照](../correctness/qwen-mtp-retirement-2026-10-22.json)进一步补充：

- 实际阻塞CUDA队列后，stop／close及丢弃Completion均未提前释放14份写事务、候选、槽与80,240,691B预算（本探针C64／N2／R1）；原生退休后旧target／draft KV、36份conv/GDN与carry／prior全字节不变，同槽重建后可继续。本窗另以目标T1验证实际writer-only草拟KV分支，6次Graph、14份权限的取消及策略异常重试也通过。成功terminal后的策略异常也未发布head，14份脏页回收后原lag1前缀可重试。未注入GPU设备错误，普通退休模式本批未重跑。
- 独立CPU从原335权重生成自己的prefill与两轮草拟／验证，不读取候选中间值。完整token、接受／拒绝、位置与lag相同；3个目标hidden边界相对L2约1.78%～2.73%，logits约1.37%～2.28%，10份浮点边界均非逐字节相同。参考算术树与设备归约差异已列明，不据此签完整质量。
- 固定llama.cpp新增K1消费者，复用原库缓存；335张量制品的320份目标载荷与旧no-MTP制品逐字节相同，15份MTP转换均核原权重。实际两轮及EOS决策相同；目标2行／草拟1行GPU greedy、RS1／0已接。peer实际context最小256、完整草拟prefill和F32 host carry等差异保留。短样例计时与三点内存记录不冒充完整对照。

本次分歧已完成有限定位：真实控制的B165 live pending与同逻辑状态恢复后T1，36份R/S整blob及hidden／logits逐字节一致，未见该分支额外pending恢复误差。精确重放到C171的T2两行logits复现原peer round9；同一RS1状态下，T1[11906]的11与271均为23.793373107910156、选11，T2[11906,271]则为23.788818359375／23.805431365966797、选271。GPU greedy与实际logits一致，不是sampler行错位；Hyle相应BF16两项均23.75。此形状数值差足以翻转本次相近top2，不外推全面质量或所有rollback正确，也不据此改变dtype、容忍值或选择规则。后续功能顺序见下；额外10次capture失效已由当前镜像直接定因，不沿用旧数值窗反推原因。

[MTP真实批次与重建本批](../correctness/qwen-mtp-batch-2026-10-22.json)为当前边界；[旧严格R1恢复](../correctness/qwen-mtp-resume-2026-10-22.json)、[会话隔离](../correctness/qwen-mtp-sessions-2026-10-22.json)和[有界前缀](../correctness/qwen-mtp-prefix-2026-10-22.json)保留各自历史检查：

- **调用与事务**：自然多轮仍用prompt／generate；实际T／角色／lag分桶，单Graph承载多请求，每成员选择自己的接受边界。expected base全部核对后整批提交，任一成员取消则整批取消；真实退休后原槽可重试。
- **淘汰与重建**：只淘汰轮间空闲head。沿原实际已消费输入块重算，不重采样输出，draft归一至lag0；已返回但未消费的pending独立保存。fresh预测变化与额外行数单列，整次成功才公开新Session。严格保原role／quota的R1入口仅为诊断。
- **实际覆盖**：10个CLI包含R1／R2驻留及压力、前缀、原单入口和双会话lag1追赶。主场景三会话各两轮；R1 N3为19次Graph，N1为29＝19＋10；R2 N3为17次／20份成员参与，N2为26＝16＋10，其中恢复64target／64draft、归一1行lag并保留1次fresh预测变化。四请求完整命中场景仅1次B2 Graph，随后两命中0Graph。
- **状态与数值**：14次Graph探针自然接受[2,1]，阻塞实际CUDA队列后取消／丢Completion仍保候选，退休后原槽重试；同分块teacher-forced重建54项逐字节一致。跨B的108项中4份draft K/V有差异；CLI另有108项限定对照逐字节一致和84项跨形状指标，6份工作不同的对照明确排除。主样例A第6枚输出、warm B第5枚输出出现R1／R2分歧，质量未收口。
- **控制与后续**：旧BF16末分数政策下，同旧Snapshot的R1／R2目标状态和输出在该样例逐字节一致，差异只在两份draft K/V，首异A随prefill来源走。该旧控制当时的末层是BF16投影再cast F32，舍入可合并接近的候选分数；底座已有明确BF16输入／权重→F32输出合同，无需绕过MatMul dtype合同或再造精度接口。本轮已显式接入该能力，数值与完整性能分开记录；下一步处理其Provider／资源取舍。物理Graph每桶计一次，恢复工作单列，P＋D为一次global wall；最新测速见页末。其它后端／量化、图片和剩余架构工作继续保留。

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
长期设计依据在[Compute](../../architecture/compute-design.md)和[Graph](../../architecture/graph-design.md)；已选模型／量化／角色／融合范围以[总表](../../implementation/model-operators.md)为准，原制品未完整执行的条目继续如实标记。

## 工作方式

沿现有公共执行链补缺口，删除重复旧路径，不保兼容分支；不恢复治理、资格或审批流程。
只运行与改动相关的增量构建、数值／状态／寿命检查与真实消费者，复用三平台工具链、依赖和对象缓存，不因计划更新重跑全库。
保留必要源码、命令、失败和结果到[证据目录](../README.md)；新阶段是否完成按上述实际剩余工作判断，不按旧清单完成数推算总进度。

## 最新完整性能对比

**2026-10-22，MTP消费者接线与共用能力补齐后的完整四臂窗口（AV）。** 北京时间18:15:19～18:15:52；Qwen3.5-2B、RTX 4070、全CUDA、BF16权重，相同P154提示词、自然greedy输出64枚、C1024／N1／R1、前缀关闭、replay auto。每臂3个新进程交错测速，共12次；另各1个独立进程旁采内存，共4次。各臂沿用AU相应模型数学、权重与自然工作；历史窗口不合并。

| 当前模式 | prefill tok/s | 完整decode tok/s | 综合输出 tok/s |
| --- | ---: | ---: | ---: |
| Hyle普通 | 2349.076 | 99.764 | 91.816 |
| Hyle MTP K1 | 2137.322 | 123.162 | 109.669 |
| llama.cpp普通 | 1553.955 | 105.224 | 91.713 |
| llama.cpp MTP K1 | 1499.623 | 139.532 | 115.482 |

普通版Hyle相对参考prefill +51.17%，decode -5.19%，综合 +0.11%；MTP分别 +42.52%／-11.73%／-5.03%。综合为64输出／（完整prefill＋decode），不含加载；本窗单轮，不冒充多轮或多人吞吐。

| 当前模式 | prefill范围 tok/s | decode范围 tok/s | 综合范围 tok/s | 加载s | 入口至ready s | 入口至末输出s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hyle普通 | 2079.09～2407.99 | 99.24～100.37 | 90.28～92.54 | 0.841 | 1.187 | 1.884 |
| Hyle MTP K1 | 2049.64～2207.36 | 116.90～123.31 | 104.22～110.22 | 0.891 | 1.671 | 2.252 |
| llama.cpp普通 | 1545.17～1637.81 | 103.57～105.94 | 90.40～92.93 | 1.555 | 1.584 | 2.282 |
| llama.cpp MTP K1 | 1498.01～1508.31 | 135.85～139.91 | 112.97～115.86 | 1.672 | 1.706 | 2.260 |

速度为3个独立进程比率的中位数并保留最小～最大范围，加载／ready／末输出也各取进程中位数；样本少，不作显著性结论。decode计prefill后63枚新输出的完整host端点，包含准备／捕获／等待／结果处理。加载边界依消费者实现；入口不含OS装载DLL，末输出不含销毁，不将其称为同一加载算法的胜负。

| 当前模式 | GPU dedicated峰 GiB | GPU shared峰 MiB | 同采样GPU总峰 GiB | CPU WS峰 GiB | CPU private commit峰 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hyle普通 | 3.888 | 92.00 | 3.964 | 1.052 | 5.510 |
| Hyle MTP K1 | 4.048 | 92.00 | 4.124 | 1.142 | 5.784 |
| llama.cpp普通 | 3.728 | 82.00 | 3.808 | 4.071 | 4.673 |
| llama.cpp MTP K1 | 3.896 | 94.00 | 3.987 | 4.220 | 5.081 |

内存是独立Windows PSAPI／PDH旁采观测峰，非精确物理分配或TPS样本。dedicated与shared分列峰不可直接相加，总峰只在同一有效采样点相加；WS／commit／GPU shared也非互斥量。请求间隔10ms，实际间隔中位15.1～15.2ms、最大21.1ms；短prefill只有6～9个样本，ready基线无样本仍为null，GPU无实例时也不填零。四臂GPU采样峰与AU完全相同，不把它当精确分配不变的证明。

**实际工作与可比边界：** 各臂3次输出ID、实际工作及非计时轮次决策均稳定，与AU对应臂保持一致；全部输出64枚，未触发EOS。普通均计算154行prefill＋63行decode。Hyle MTP为34轮、29次接受（85.29%），验证68行、draft decode 62行；peer为33轮、30次接受（90.91%），验证66行、draft decode同为62行，两者最终提交decode均63行。两边MTP prefill各有154行target／draft，但Hyle只算所需draft K/V，peer走原完整MTP块。两对续文仍从第19枚起不同；Hyle中间BF16、末分数F32，peer原生算术／状态／host carry也不同，报告自然性能，不称同token或数学完全一致。

**窗口变化与性能主线：** AU已将普通／MTP decode恢复到91.44／112.29；本窗为99.764／123.162，但peer也从95.08／125.65升到105.224／139.532。两边都更快，不能把Hyle约9%的变化归因为本批优化收益。普通共用路径相对参考的差距继续按实际热点处理，与Qwen K1已列明的MTP状态／接受逻辑检查区分；其它适用MTP制品仍未完成。

**[AV共用能力与检查](../correctness/mtp-consumer-wiring-2026-10-22.json)：** 输入租约候选表仅在真实native输入需要时构造，原F1权限、alias、seal及退休门不变；F32 Add按实际标量carrier补4B对齐声明，保留V17夹具。owned alias与候选不相容时提前冷拒，保合法后续选路；29项相关回归0失败／0跳过。

**[本批Q27／Q24原角色与codec检查](../correctness/qwen-original-mtp-input-2026-10-22.json)：** 5项测试0失败／0跳过。Q27原三份参数共43,048,960B，用生成输入经CPU／CUDA各执行3次Graph；原GLM输入reader邻居复查通过。Q24当前primary54／fidelity48均通过生成参数的norm／carry和NextN算子链，含4步反馈与V248320分片head，未冒称原payload或完整接受状态。首轮Q8_0大表Gather因复用的provider编译命令漏CPU capability而拒绝；按生产CMake配置仅重编该TU后通过，未改产品门或数学。本批没有新TPS／内存窗口，AV四臂表不变。

[本轮12次计时、4次独立内存旁采及实际命令／结果](../performance/qwen-mtp-assembly-2026-10-22.json)。[AU有界捕获变体](../performance/qwen-capture-variants-2026-10-22.json)保留包含active在内的两变体、实际prepared资格与完整tuple／失效门，以及24项检查和15次Graph／534项向量回归；[AT诊断](../performance/qwen-capture-recording-cost-2026-10-22.json)、[AS可执行更新](../performance/qwen-capture-rebuild-2026-10-22.json)、[AR单槽回退](../performance/qwen-mtp-current-carrier-2026-10-22.json)、[AF单槽前窗口](../performance/qwen-mtp-steady-profile-2026-10-22.json)保留各自历史口径，不混算。
