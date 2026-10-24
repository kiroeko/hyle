# 非多模态推理底座全量审查

2026-10-18，按用户要求启动大 goal，当前审查进行中。范围是既定8个文本模型、22个官方/GGUF制品配置，
覆盖受支持系统上的CPU、单GPU、CPU＋单GPU。模型事实沿用[来源目录](../../../implementation/tasks/compute/llm-moe-model-sources.json)
和[制品目录](../../../implementation/tasks/compute/llm-moe-provider-demand-matrix.json)，旧表的流程状态不作为当前结论。
此前已完成的局部修复及其证据见[接线与高维批次报告](../model-route-completion-2026-10-18/README.md)，本轮在当前源码上继续检查。

完整Qwen3.5-2B消费者另完成[固定输入32阶段](qwen-fixed-decode.zip)：预先选定21-token提示及31个decode输入，
各自反馈真实KV/conv/GDN状态，可见长度达到52。Direct/Scheduler的64份决策向量逐字节一致，
最终CUDA/FI host更新后的64份回放亦相同。独立Torch先复现原150份三阶段结果，再运行同一序列；
eager的top1相同31/32，唯一差异步为最高两token同为20.25并列；logits相对L2均值1.166%、KL均值6.462e-4。
仅提高参考RoPE中间精度后top1为32/32。326份原始决策数据、固定输入、精确源和失败/命令边界均在包内。
输入来自既有参考生成，未作独立质量或公平吞吐验收，也不以参考的中间结果替换候选状态。

[实际capture审查](qwen-capture-current.zip)只改变普通重放政策与诊断输出：PreferCapture的32阶段仍全部为Commands，
64份决策向量不变。实际单段有686个命令步骤，其中427个不兼容外层capture；兼容步骤只能组成最长3步的短区间，
尚无证据支持为此细分Graph段。FI另有消费者尚未调用的原生slot capture，ggml也有内部capture，不能由外层报告推断其内部使用量。
后续一次[Nsight实际观测](qwen-native-graph-observation.zip)保留原8阶段输入：decode3/4各出现401条原ggml预热完成日志，
并各捕获、实例化、启动401个graph；decode5–7各重放401次，无新捕获。8个输出token不变。
工具使用legacy软件trace（driver13.4、CUPTI13.3），未按kernel节点归因，采集时间不作吞吐结论。
[计时边界修正](qwen-timing-boundary.zip)为消费者增加step_seconds，将输入上传、布局、绑定和重放准备纳入单步费用；
Direct/Scheduler各32阶段无trace烟测序列不变、计时次序成立。旧seconds只覆盖提交到token读回，不能用它与完整decode时间判输赢。
后续已完成正常greedy生成与21/128-token两组全CUDA固定输入的[实际性能对照](../../performance/qwen35-full-cuda-2026-10-19/README.md)：
每组3对独立进程，实测加载、完整step、decode和PID内存；128-token下Hyle首次prefill更快，加载、decode和显存有实际输项。
双方中间精度仍有差异，本次不作质量接受，也不以固定输入短程吞吐代替长期运行验收。
实测随后促成ggml CUDA普通区域临时寿命复用修复，沿原Invocation/gallocr删除逐tensor分配分支；
长链184→8 MiB，真实Qwen T128峰6.431→5.312 GiB，21/128两组128份决策向量逐字节相同。
回归、原负控阶段假定失败及修正见[原始记录](../../performance/qwen35-full-cuda-2026-10-19/ggml-cuda-liveness.zip)，未证明跨region workspace已复用或decode已改善。
后续已沿原caller workspace合同接通跨region HB复用，T128峰进一步5.312→3.972 GiB、128份决策向量不变；
真实两region各8 MiB临时量加8 MiB桥只占16 MiB，相关10项通过。[本阶段记录](../../performance/qwen35-full-cuda-2026-10-19/ggml-cuda-caller-workspace.zip)
不代签其余模型、多chunk或全部平台旅程，decode/加载问题仍在。
上述Qwen/GQA实验归档后，通过根目录test.ps1按精确路径、大小和SHA删除24个exe/map/lib/exp，
回收897,306,207B；9个可复用object、全部源码、原始数值、日志和SDK保留。[清理记录](qwen-current-reclaim.zip)

本轮续查修复Qwen3.8-2.4T实际QH64/KVH4/D256的CUDA原生批次拒绝：删除历史GQA比例名单，
沿原planner与runtime分组寻址，并将原uint32容量保护统一用于低维Q/output与ragged KV。
Windows默认Direct/普通Graph新3/3通过，含GQA16的24次及原GQA6的6次独立FP64校验、两槽旧结果和两项冷容量拒绝，
保持原0.007预算；同源旧产品两项控制均明确失败。只增量host/test TU，无CUDA内核或SDK重编，未验收整模型或吞吐。
[源码与记录](qwen38-gqa16-current.zip)保留修前拒绝和fixture错误；Windows数学校验在最终无行为换行前执行，两份精确源身份均保留。

原功能审查未完成的范围继续由架构重构承接，不以迁移完成代替功能补齐，也不保留旧 API 兼容层。
最新[变长Graph纵向消费](native-ragged-graph.zip)接通Apple/Vulkan原生GQA，沿原槽验证不等长/空请求、
同Q续KV、改Q冷重建、Graph生成QKV、两槽旧结果和容量零尾部；metadata共用冷owned版本与配置输入锁。
真实双CPU资源消费进一步修复跨域误共享镜像，并用稳定主机地址能力替代CPU搬运的system-resource名单，
保持未知映射与跨wrapper写重叠拒绝。Windows最终合跑19/19、CPU搬运6/6（共24个不同用例），Air最终11/11；
只增量编译host owner，SDK、数学库和shader未重建。早期宏编译、错误标量类型、CPU身份拒绝均保原记录。
固定容量补零/拼接和不同资源的必要原生metadata镜像有真实费用；本轮没有吞吐或整模型精度验收，
compact RaggedMLA与Linux现代码闭包也未因此补签。
Linux后续已用当前156个C++/3个ASM宿主、3个ggml接口overlay、12个CUDA owner及精确AVX2 companion增量链接，
唯一原CPU＋CUDA finite-family case在RTX4070实际exit0；[原始记录](linux-current-graph.zip)核190对象身份、源码漂移0，未用旧Hyle宿主对象或重建整SDK。
本批已把资源租约、结果和 batch/page/pending 控制收进原 Invocation，删除旧 Graph 租约与批次协调入口；
原 core 六项、CPU 租约重放、CPU pending 九模式、CUDA F16/BF16 真 gate→F2→MLA，以及两槽 CPU＋CUDA
控制通过。复用现有 native/SDK；旧缓存闭包和迁移漏冷准备造成的失败及其更正均保留。
[源码与原始记录](unified-invocation.zip)。generated-output、其余混合运行时及下文未完成的模型功能仍须继续。

后续批次将执行目标、region 与常量制品改为保有实际 `ExecutionSource`，删除原 Context 指针入口；
共同资源核心的 Gather 数学特判也已移到 typed 模块，由实际 singleton/融合制品证明受限读取。
共享源与原异步资源控制 13/13、迁移后的原 CPU materialized 完整消费者、真实 Vulkan 源与输出 backing 1/1 通过；
调用方按新接口迁移，未保留旧字段或转发入口。Vulkan 检查使用原生 Program/Invocation，不能代签统一 Graph 的原生结果链。
首次闭包 ABI 崩溃和新测试误选 native 未承载的 F32 Neg 均保留，分别复用匹配闭包、改用实际承载的 Exp 后通过。
[本批源码、命令与结果](execution-source.zip)。没有重建 SDK，也未运行多平台全矩阵。

统一 Invocation 后的 Qwen 三 phase 另有 150/150 trace 文件与冻结基线逐字节一致记录（同包保留），
它验证该次迁移没有改变既有输出；与独立 oracle 约 2% 的原数值差异仍未接受。本批 ExecutionSource 变更后未另跑完整模型。

原生生成结果随后已接通第一条普通 Graph 纵向：实际 Vulkan Source 通过 `graph::compile` 发现真实 region，
同一 Executable／Invocation 持原 Program、原结果 storage 与原 Completion；生成端口不分配影子 workspace。
Windows RTX 4070 上两个原始消费者通过，覆盖两槽、创建时／绑定后 backing、computed/input View、
旧 GraphOutput 和独立 lease 阻止覆盖、释放后复用、关闭后保活与无回指环，以及借用原生常量的两 phase。
其中一槽走统一 ResourceEpisode 的拥有式 ReadLease 绑定。共同异步寿命控制 13/13 和原 CPU materialized 消费者亦通过。
实际执行检出了已有原生结果附件与 cursor 的单附件冲突，现用原 CompletionRetention 组合，且在节点释放后才发布槽可复用；
失败日志与更正保留在[源码、命令与结果](native-generated-output.zip)。未重建 SDK／shader，也未重跑完整 Qwen。
这证明固定纯图链路，尚未证明 Apple 晚生成句柄、全部混合／跨片段路线、CPU→native 常量物化或完整模型／性能。

后续[冷常量搬运与能力准入](cold-constant-copy.zip)已把 CPU→Vulkan 的 `kMaterialize` 接入同一编译入口：
先校验实际资源和完整批次，再复用一份最大容量 staging／原 TransferSlot 上传；临时字节计入编译峰值，
成功后捕获目标常量，重放不再读取源。RTX 4070 上四项原生用例通过，含 16／4／3B 常量、释放源后两 phase、
预算预拒绝、外来资源整批拒绝、读取错误及一次真实 fence 观察 OOM 的原生退休与首错保留。
共同编译同时删除同 Device 总门和 `compile_cpu_gpu` 后端 ID 特例，仍保留实际 owner、资源、搬运与完成依赖检查；
相关常量／共享／失败／跨 Device 准入控制 26/26，原完整 CPU materialized 消费者通过。
测试使用匹配 host 对象和现有 SDK；旧对象引用已删私有签名的链接失败、测试预算漏 metadata 域的失败及修正均保留。
没有重建 SDK／shader；本批未重跑完整模型或实际 CPU＋CUDA 消费者，不代签其余混合路线、Apple 晚句柄、旧入口删除或性能。

后续[原生晚端口与有限形状](native-ports-finite.zip)已接通 Apple 同一 Runtime 的普通 Graph 原生结果链：
原槽／原 Job 解析真实端口，consumer 接纳后短期 pin producer，checked 失败不发布后继；指定 MacBook Air
原生四例通过。零公开输出的控制片段保留 Program 私有终点。只读输入、借用常量及结果导出按实际 Source
证明资源族；独占输入不额外捕获将升级回 writer 的 reader，最终相关控制 3/3 通过。
有限精确表现由共同冷编译承载，没有通用路线、热编译或独立执行器；形状／预算控制 11/11，
实际 Vulkan、Apple 及 CPU＋CUDA 消费者通过，三者旧形状表薄包装已删除。Apple／Vulkan 的其余混合入口仍在迁移。

同包记录了 Apple 拥有式 View 反馈的实际前后对照：同一 16B 子范围、64 轮，旧版存活元数据从
39块／51,152B 增至615块／245,712B；复用原 ReadLease 子范围后，各采样点均为23块／45,280B。
原生 Graph 原两例与反馈例 3/3 通过，含精确权限、原 owner、新计算结果、关闭后读取与最终释放。
这是元数据寿命修复，未测吞吐，未宣称少复制 payload。包保留定位失败、匹配对象身份、源码和命令，
没有 SDK／binary；本批未重跑完整 Qwen、Linux 或大矩阵，不代签跨 Runtime、全部混合路线与剩余模型功能。

后续[canonical 常量所有权](canonical-constant-owners.zip)改为逐项保有真实分配，按精确描述、实际资源与足够对齐
共享 ready owner；候选重试恢复原始来源，释放不兼容副本，family 只扣除真实共享字节。
三项隔离对照确认修复必要：旧重试留下4条对齐记录对应2实例并残留256；禁用实际偏移检查后，+4B/min64输入
被提交且改写哨兵；旧版允许将可写输出绑到借用常量。修复版最终13/13通过，包含+8B/min8合法双View链、
独立输出重复执行、只读常量/View及packing累计预算。此前family相关19/19、RTX4070原Vulkan四例4/4也保留各自
源码与对象身份；Vulkan四例发生在最后常量alias保护之前，不作为该保护的测试。初始fixture构造/资格错误均如实保留。
同计划跨存储常量副本的冷准备、CPU/native动态copy和分段发现仍未完成；本批未重跑整模型或新增跨平台验证。
归档786,133B，不含SDK或binary；随后经根目录`test.ps1`逐项核对并清理18个可重建文件，共275,013,775B，保留全部对象缓存。

后续[原生重放搬运与分段发现](native-replay-discovery.zip)将 CPU＋Vulkan 固定／有限形状数据链接入共同入口，
删除其旧形状表包装。实际 host、staging、device 域双向搬运；原 CopyRecording 仅冷绑定录制，同槽同址改内容后
直接重复 submit。控制失败保留 CPU 输出哨兵，原生待退休提交拒绝复用，已 settle 的旧 Completion 与新 Run 独立。
发现协议直接升级为22，按实际 target/Provider 许可和原承载资格生成片段；共享入口前沿算法为 O(V+E)，
后继新增外部前驱时切分，避免早期工作额外等待。零公开输出片段仍执行实际私有终点。
最终 Windows 相关控制16/16、原生 Vulkan及原 CopyProgram回归6/6、CPU→Vulkan→CPU 两个形状各两轮通过；
MacBook Air 最终原生5/5及 finite/control 消费者通过。原生冷适配同时保留真实来源校验和原 admission，
两实际片段交接、四类无效适配拒绝、旧结果和 checked 失败均有实测。初始测试对已 settle Completion 的错误期待、
融合后旧 region 数断言与旧适配门拒绝日志保留，未据此伪造产品性能缺陷。没有重建 SDK、shader或跑全矩阵。
同计划常量跨存储冷准备、其余混合／跨 Runtime、分页与 custom 迁移、完整模型数值及公平性能比较仍未完成。
归档1,706,069B，不含SDK／object／binary；经根目录`test.ps1`核对后清理 Windows 12个、Mac 10个
可重建binary/map/export文件，共314,322,548B，全部源码、日志、数据和对象缓存保留。

后续[冷常量物理副本](cold-constant-replicas.zip)把同计划跨存储的 Graph 常量根移到冷准备，
每个物理实例关联唯一 allocation owner，主借用政策保持不变；metadata View 共享完整目的根，再恢复具体偏移。
region 与 packing factory 使用实际本地副本；候选重试按真实 key 重新关联 ready owner，累计预算只扣一次。
主机11/11通过，含双存储不同偏移、目的域 View 发布、双槽直接重放/reset、有限形状共享、子路线寿命和精确预算边界。
Vulkan 最终9/9及原 CPU→Vulkan→CPU 消费者通过；冷上传、下载与对齐 D2D 复用原 Slot/Recording/Completion，
实际复制错误包含 Unimplemented 均不换路重试。逻辑分配长度与本次复制长度共同约束 padding，裁短 span 或短描述
不能借完整根的尾部权限。host staging 与 D2D 容量分别计算，纯 D2D payload 暂存为0，16B D2D＋3B host 批次为4B；
这是请求字节和实际分配参数的结构证明，不是 driver/RSS 峰值或吞吐测量。
Apple 原生源补齐无 CPU Context 的冷物化，真实共享 Metal 目标校验 device、范围及可写性；readonly 导入只能作源。
指定 Air 最终7/7及 native finite/control 通过，含16B权重跨两个形状只读源一次、源释放后双槽数学、odd3B和实际源错误。
初始测试遗漏 CPU caller 输出／HostResource map 的失败与更正均保留；共同层新别名曾提前检查 Provider 对齐，
现恢复在 prepare_replay 校验并通过原动态偏移回归。未重建 SDK、shader、跑矩阵或完整 Qwen；其余跨 Runtime、
分页/custom/旧执行链与继承的模型功能审查仍未完成。归档1,652,687B，不含SDK/object/binary；经根目录
`test.ps1`核对后清理 Windows12个、Mac8个可重建文件，共301,688,977B，保留源码、日志、数据和全部对象缓存。

后续[Apple 只读结果权限](apple-readonly-ports.zip)修复 ImportedArrayStorage 继承可写映射的问题，
并让原 Metal copy 的源／目标校验显式区分读写。真实 Graph 成功输出仍由原 ReadLease 与实际 resource 保活；
主机写映射、write/fill/copy 目标、普通与晚生成源 blit 的目标均拒绝，合法只读源和普通可写分配保持可用。
指定 Air 仅新编 memory、copy recording、copy program、native test 四个 TU，复用原 SDK 与最新共同图对象，
最终8/8通过，包含原融合、常量、拥有式结果、晚端口、epoch 与 checked 失败用例。
CPU＋Apple 的跨域晚端口冷选路与异步交接仍在迁移；本批没有实现该桥接或重跑整模型／性能比较。
归档后经根目录 `test.ps1` 精确清理本批 Mac binary／map 共14,171,868B，保留四个新对象及源码、日志、数据和SDK。

后续[端口承载与原生晚源 Copy](carrier-copy-contract.zip)将输入的原引用接纳、输出的可用时机及 Mem 投影承诺
放到同一冷提议中，最终制品按逻辑端口核对；共享／原生分类不再冒充引用能力。同 storage 也可明确准备 caller
副本，原 producer 保留给合法原生消费者；普通 Copy 不因 HostAccessible 就认未来句柄已存在。
Apple 复用原 InputSource／CopyRecording／FIFO Job／Completion，冷映射目标、每轮按新 epoch pin，
成功校验后才复制和 flush，失败／取消／拒绝 admission 均放 pin，实际接受后才消费 epoch；只读导出保持只读。
结果观察同时核实际 Source 的资源匹配与 Mem 投影承诺。Vulkan 的逐输出资源提议只存边界和所选 View 的稀疏事实，
不再为每个候选清整图大小的数组；这是结构复杂度改进，没有据此宣称吞吐收益。
最终 Windows host9/9、Vulkan9/9及原 CPU→Vulkan→CPU 有限形状／控制消费者通过。
指定 Air 最终11/11：真实默认 CPU→Apple→CPU 的两种 shape、双槽、两轮原绑定严格重放、旧结果保护与 checked
失败目标不变；原生所有权、常量和晚端口回归及新 Copy 的旧 epoch 拒绝、pending pin、旧 Completion 和失败退休通过。
新混合测试最初缺头／误用 Result 及错误地仅绑定不承载 F32 Abs/Neg 的参考 Provider，修正后使用默认 CPU 组合；
所需 CPU／batch host 对象及同456 pin当前锁定 native ops／allocator 差异按实际缓存准备，未混入另一 CPU-only pin。
当前 ContextRef 接口的两个旧缓存链接缺口也已用相应当前 TU 补齐，全部失败和 source／object 身份保留。
没有重建 Metal、shader 或整套 SDK，也没有跑整模型或性能比较。现有公开 submit 的显式 host-wait 仍可阻塞调用线程；
本批未实现统一 continuation，不把数据链通过等同于旧混合执行器已删除或完整异步迁移已完成。
归档共2,395,626B，SHA256 `3b2d90626b94a9e6dc02edc7a697c317b7c3184b8000ec9765862c75e2472aaf`。
归档后经根目录 `test.ps1` 精确清理 Windows 12个 image／map／导出文件共247,896,469B及 Mac 两个 image／map
共17,565,669B；保留本批 Windows 九个对象、Mac 13个主对象及15个 CPU overlay 对象，源码、日志、数据及SDK不删除。

后续[执行职责与有限形状收敛](execution-convergence.zip)将同一个提交游标、终点聚合与实际退休实现移至
`submission.cc`，冷 capture 和热提交仍复用原片段录制函数，未添加 Plan／Invocation 状态或调度器。
默认 CPU Provider 装配与 batch 冷编译拆开 TU；原生形状表模板和最后 CPU＋Apple 薄包装删除，实际 finite
消费者改用共同 family。通用 exact specialization／native analysis 回归保留，原模板的工厂测试由已有共同
family 的真实 finite、重复／非法点、部分分配失败与预算回归承接。Windows 六个相关 TU fresh、24/24通过；
指定 Air 六个相关 TU fresh、22/22及完整 Apple finite／control／旧结果消费者通过，原数值SDK／shader未重建。
Windows 另一个仅 create_context 的缩小链接仍未闭合：新 context 对象没有 graph_batch 编译直接引用，但
实际默认 Provider 仍带 native Batch／MLA 能力；补其既有缓存闭包时遇旧 operation 重复定义，三次失败均保留。
对象符号证明不等于整个默认路线已经剥离 batch。Queue 原状态、已接纳回调和原 Completion 的异步移交仍待贯通；
本批只改变职责归属与删除重复形状入口，没有宣称完整异步支持或整模型／性能接受。
归档共1,026,593B，SHA256 `2d1f0a19c15f8cffb52224ece82ea1627f23df9fbee0905851efec2866a0189d`。
归档后经根目录 `test.ps1` 精确清理 Windows 七个可重建产物55,635,606B和 Mac 四个 image／map 34,344,118B；
本批 Windows 七个对象、Mac 六个对象及全部复用缓存、历史源码、命令、日志和数据保留。

后续[原队列寿命与已接纳序列](admitted-queue-sequence.zip)将共同 Graph 的队列绑定改为持有原 QueueState，
原 QueueSequence 从接纳到 finish 保活同一状态；prepared／region／extension、CUDA stream／status、batch 和
custom-kernel 热回调均传原序列。删除回借公开 Queue 的旧内部入口，直接升级 Provider15、Region24、Batch5、MLA3。
公开句柄移动不改变绑定，close 拒绝新接纳，已接纳的原序列仍完成；没有会关闭原队列的第二个 Queue 包装。
两项新增回归使用真实 CPU backend 和 Graph 数学，覆盖移动／销毁／关闭、单 producer、精确状态释放与关闭后输出保护。
Windows 最终相关69/69通过；指定 Air 鲜编36个 Hyle host TU，24/24及原 Apple finite 消费者通过。
CUDA 的28个相关 host TU源码编译通过，RTX4070 上真实 native CUDA Provider 的原 finite 数学／非法点／配额／
旧结果消费者和两项 checked Gather 通过。该消费者仅在实验源码中显式装配真实 native Provider，原默认消费者未改。
默认 ggml CUDA 当前补丁的配套 native 缓存未找到；旧缓存头缺 IQ1 枚举，独立原始参考库也缺 Hyle resource 桥，
均未混用。tensor_program 使用已核同 pin 的公共头鲜编只证明源码编译，不证明当前 ggml CUDA native 闭包运行。
Mac 严格编译另修两个显式缺省初始化，以及原 DS4 header 未用函数的局部 Clang 护栏；失败命令、补闭包过程均保留。
数值库、CUDA kernel、Metal shader和整套SDK未重建。独立 batch/copy、旧混合协调器仍有公开句柄借用合同；
原 Completion 的异步移交、统一前沿和旧执行链删除仍未完成，本批未运行 Linux、完整模型或性能比较。
后续只读设计审查确认当前线性游标遇 HostWait 会挡住后续独立片段；A异步GPU、B等待A的CPU、C独立GPU copy
可冷排成A/B/C，B pending使C不能提交。这是源码推导，未运行新的吞吐实验。旧CPU＋CUDA已有并行CPU child；
下一步须在原计划依赖表和同游标上推进前沿，保CPU/GPU重叠、实际Queue prefix、工作回调与原生退休，
避免让BlockingPool进度任务等待同pool的排队child；本批未把它记作已修性能问题。
归档共2,305,406B，SHA256 `88fc632d81aac15f5c6120b914ab3758fc9f78c2b4c56d97e96ee062154a9260`。
归档后经根目录 `test.ps1` 核对并清理 Windows 12个可重建 image／map／导出文件186,229,915B、
Mac 四个 image／map 36,044,483B；保留本批 Windows 68个对象、Mac 36个对象及所有源码、日志、数据和SDK。

后续[原提交游标的就绪前沿](submission-frontier.zip)将最终依赖表冷转置为后继索引，原槽预留进度计数／队列；
HostWait 候选暂停时继续独立 lane，真实 Queue owner 分组限制 producer，最终 prefix 按实际接纳顺序选择。
CPU＋CUDA 删除自己的 JointPipelinePlan／frontier 及冷依赖复制，原 CPU child、回调交接和实际退休沿同游标。
Windows 鲜编8个相关host TU，原69＋新增三项72/72通过；指定 Air 同8TU、原24＋新增三项27/27及原finite消费者通过。
三项新控制覆盖等待旁路、非ordinal接纳后的真实最后prefix、claim到commit的独占与取消回收；固定图重复回放未新增所测元数据分配。
RTX4070 原 native CUDA finite消费者通过；CPU-first 实验在实际CUDA begin前等真实CPU executor gate进入，随后在CPU仍被持住时
观察到原CUDA event完成，放开后完整Abs＋Neg→Add、双槽变更输入、旧结果、取消／丢弃／drain通过。
插桩仅限当前CUDA backend宿主副本的受控begin与转发原event API，正常finite消费者使用未插桩产品backend；无数学／Completion替身。
最初GPU-first观察不能排除GPU先完成的解释，原源码／日志保留并降为数值及非阻塞证据；顺序纠正后才认定上述进度检查通过。
未重建数值库／SDK／CUDA kernel／Metal shader，未运行Linux、整模型或吞吐实验；默认ggml CUDA缺配套缓存的限制未变。
通用异步owner、真实Completion移交、其余旧原生协调器及剩余模型功能仍待迁移，不由局部前沿和消费者代签。
归档共695,936B，SHA256 `e7f278f47d2e1f42986123582122445145c607f0459c045ed4b65ae872cefe94`。
归档后经根目录 `test.ps1` 核对并清理 Windows/CUDA 12个 image／map／导出文件187,131,335B、
Mac 四个 image／map 38,683,690B；保留本批 Windows/CUDA 13个对象、Mac 八个对象及全部源码、日志、数据和SDK。

后续[Completion 异步移交与架构收敛](completion-handoff.zip)为原 RetainedCompletion 增加独立移交凭据，
持有原 observer State 的 driver 引用；公开 root 提前 reset／析构后，仍可交入真实最终 Completion。
没有删除 Invocation 原有的引用环清理，也没有新执行器或结果槽；安装阶段由一个互斥枚举表达。
Windows 四个当前 Compute／test TU 以 `/W4 /WX` 编译，小闭包 41/41 通过，包括六项新增取消／安装／
提前丢弃／未知退休／同步回调重入／root 析构与复用控制。首次 MSVC 转换歧义和未初始化告警及修正均保留。
反向审查同时取消 Graph 自行重置 Fiber 的设计前提，按最后一次槽访问区分 detached callback 与 embedded Job
的寿命；原 Scheduler 任务分配成本与严格无分配目标仍分别记录。共同退休提取、Graph 异步接线和旧驱动删除
尚未完成，本批未运行 GPU、Mac、Linux、整模型或性能实验。归档仅保留源码／命令／结果及缓存身份。
随后按根目录 `test.ps1 -ReclaimCompletionHandoff` 核对并删除两个可重建 image／map，共2,573,734B；
保留四个对象及全部源码、日志、结果和已有依赖缓存。

后续[共同提交寿命](submission-lifetime.zip)将原 ResourceEpisode 的退休节点、首错与计数搬到原 Invocation
的可选寿命附件，ordered admission 不再借栈；资源附件只管真实租约、权限、batch 和 payload。
原同步资源路径继续使用同一 Cursor／Completion，无资源普通路径不增加分配；资源路径多一个冷 owner。
`finish(true)` 零接纳失败现在等原 owner 归还 sentinel 后才释放槽及共享制品 gate，随后可重试且不发布输出。
Windows 八个相关 TU `/W4 /WX` 鲜编并链接匹配闭包，原72项＋新增零接纳／双槽控制73/73通过。
未接通完整 Graph 异步驱动，未重跑 Mac／Linux、原生 GPU、完整模型或性能；线程调用合同与旧混合执行链仍待收敛。
按根目录 `test.ps1 -ReclaimSubmissionLifetime` 精确清理四个 image／map／导出文件，共60,274,892B；
保留八个本批对象、复用对象及全部源码、日志、结果和依赖缓存。

后续[原Invocation的共同异步推进](generic-progress.zip)接通可选`prepare_progress`，仅驱动原Cursor、
原资源准入和原Completion。阻塞段按制品／实际transfer mode交BlockingPool，未引入第二份计划或结果槽。
修复准入失败的lease batch回滚、逐claim取消检查、ready通知窗口与完成成功早于结果可读的发布顺序。
Windows12个相关TU `/W4 /WX`鲜编，95/95通过；单worker真实CPU RMS、fiber drain、旧结果、拒绝、取消保活均覆盖。
原CPU＋CUDA pipeline消费者已迁同一Invocation并鲜编；实际expert CUDA finite家族及CPU RMS＋CUDA Neg→CPU Add
两槽两轮通过。受控原生事件在真实CPU gate放行前完成，证明进展，不是吞吐测量；两次测试门设置失败的日志保留。
任务／stack仍有原Scheduler分配成本，handoff尾部可能保守拒绝立即重放；旧WeightBatch／分页等专用执行链尚未全部删除。

本批还纠正此前“默认CUDA缓存缺桥接／IQ1”的判断：对原库逐源／符号审计后只增量编译当前主模块与allocator，
旧库只读复用，未重建kernel集合或复制SDK。四个当前host接入TU与匹配core链接后，完整Qwen24层的
23-token prefill→4-token续算→1-token decode真实执行成功；生成token与独立参考一致，150个tensor全finite。
独立logits relL2仍为2.1512%／1.9573%／2.4589%，数值未接受。新旧trace、1252对998报告步数及原生编译差异
分别留存；step不等于kernel，进程4.953s也不是稳态tokens/s。未重跑Mac／Linux、其余模型或公平吞吐对照。
单次T23诊断报告将1252步分解为518个metadata View、534个operation、200个region，Plan Copy为0；
原库fastmath unary与当前precise unary的差异是数值候选，尚未证明能解释全部累计误差。
证据包只含源码、命令、原始日志／结果和身份；本地原始tensor数据、对象和依赖缓存继续保留。
归档后经根目录`test.ps1 -ReclaimGenericProgress`逐项核对，回收16个可重建image／map／导出文件337,527,574B；
未删除源码、日志、tensor原始数据或对象缓存。

后续[权重输入与共同执行槽](weight-inputs.zip)将真实WeightBatch所有者和上传依赖接回原Invocation，
与分页payload独立，冷按实际读者规划、热按真实Queue首次接纳去重。三个旧CPU+CUDA运行时文件已归档删除，
公开header仅保留冷环境helper；没有兼容别名。共同resource_report覆盖任意StorageDomain。
相关core回归96/96，默认CPU/CUDA公共权重消费者、contracts、finite、sampling及真实pending分页成功／取消均通过。
分页测试检出零计算接纳失败跳过原源映射、无法取回已接纳CopyBatchUpdate控制的漏洞；修复后实际取消退休通过。
foreign Context负控按真实Buffer冷prepare隔离，未将普通compile的设备资格检查误当精确Queue准入。
原失败、修复源码、精确对象闭包及最终日志保留；未重建SDK／kernel，也未验证完整错误注入、多平台或模型性能。
Apple／Vulkan其余专用驱动及Qwen独立数值／吞吐验收仍未完成。
归档后经根目录`test.ps1 -ReclaimWeightInputs`逐项核对并回收24个image／map／导出文件，
共1,012,002,800B；所有源码、成功／失败日志、原始数据、对象和SDK缓存保留。

后续[原生输入与预算接线](native-inputs.zip)补通Vulkan immutable权重的HostWait／实际owner去重，
以及普通Source的冷custom lowering；CPU＋Vulkan sample／finite改用原Invocation与progress，完整Q8 FFN保持四节点融合。
发现并修复原生generated backing未传共享预算的遗漏：Region SPI25给出原budget／实际域，原native分配留其真实Reservation。
相关core96/96、11项Vulkan custom／source／预算、2项pending权重和两个实际消费者通过；原失败及修正均保留。
预算探针证明当前Unary需要16B外部输入适配＋16B输出；不足scope拒绝，32B成功，逃逸输出保留16B至最后owner释放。
没有因逻辑输出只有16B而放过实际分配，也未根据缓存猜测更改数学。Apple两TU仅定向编译，未跑当前共同core。
复用ggml缓存的证据限本次F32／Q8域；未宣称完整类型／模型、性能对照或旧native分页／驱动迁移完成。
归档后按根目录`test.ps1 -ReclaimNativeInputs`核对并回收20个image／map／导出文件448,515,932B，
保留全部对象、源码、成功／失败日志、数据和SDK缓存。

后续[原Invocation的原生ready分页](native-ready-pages.zip)由真实CPU→Vulkan page→CPU链校正抽象：
节点持有recipe／初始layout，所选prepared冷能力管理原slot与精确页权限，layout变化后由原replay准备重绑下游。
F16/D64/causal双槽、三layout、64轮及额外一次固定绑定直接重放通过原FP64对照与寿命检查；
owned Operation／Graph／schema相关61/61通过，Air四个相关TU仅语法检查通过。
保留libc++合同、缺私有头声明和消费者漏绑caller输出的失败与修正；未改误差预算、重建SDK或测吞吐。
native pending、空footprint、其余旧驱动和完整模型数值／性能仍未收完。
归档1,437,830B；按根目录`test.ps1 -ReclaimNativeReadyPages`核对后回收6个image／map／导出文件
100,314,412B，保留全部对象与SDK、源码、数据和成功／失败日志。

后续[原Invocation的native pending与空输入](native-pending-pages.zip)复用原资源附件、typed CopyOutcomeSource
及唯一DependentCompletion，不复制Graph执行槽或把可写更新套作不可变权重。Vulkan实际F16/D256双请求、
Q2/Q18各两槽、两page节点共享Update、CPU前后段与64轮增长历史通过两份输出的原FP64预算及寿命检查；
只复制196,608B新增行，保原generation。Q0/ref0真实空frame持arena身份但无页ReadLease，持全部页writer的
CPU→native→CPU空输出链通过；两项相关权限控制通过。
该真实双节点消费者发现了复制缓存ABA：第13轮第一节点正确、第二节点读旧零buffer。
保旧fastpath的临时诊断实证同Mem包装地址的VkBuffer key变化后仍跳重录；最终改冷绑定比较真实physical span，
原64轮和硬数值断言通过，热提交未增加解析或分配。保留原失败、诊断、源码和对象身份。
没有重建SDK／kernel或测吞吐；空路径、权限控制、pending的具体源码与链接批次分别列于包内。
未代签mixed ready/pending端口、accepted失败注入、Apple迁移或其余dtype。
另确认native Q容量接口导致上游多算／多传：18行容量、2行decode仍处理18行，而attention只处理活动行；
最小layout-aware描述修复设计已随包保存，尚未实现，不把9倍元素工作量写成实测时延。
旧驱动其余消费者及完整模型数值／公平性能对照仍未收完。
归档673,813B；按根目录`test.ps1 -ReclaimNativePendingPages`核对后回收12个image／map／导出文件
298,882,350B，保留全部对象、SDK、源码、数据与成功／失败日志。

后续[活动Q的共同描述](native-active-queries.zip)直接删除NativePaged无layout的`descriptions()`，
改为纯查询`input_count()`／`input_description(layout,port)`；Q取活动行，KV／页表保容量，
Direct、Graph、Apple/Vulkan冷计划与独立参考构建器使用同一事实，无兼容shape分支。
5项相关schema／所有权检查、ready和pending各64轮、同一空消费者内的零行Graph及Direct零lease通过。
pending D256/F16/H8的Q2每条搬运8192B，Q18为73728B；真实Host→staging→Device两跳各按活动范围计费，
进入device恰一次。
保留首轮统计把所有搬运误当一步的失败与修正；没有放宽数值预算或将元素／字节减少写成吞吐倍数。
Vulkan零逻辑输入仅在缺合法carrier时冷建最小物理载体，按原scope计费；现有足够backing继续共享。
Air两个相关TU仅语法检查通过，其他迁移编译范围与源码身份见包内；未重建SDK／kernel。
该批只修活动Q；数学描述与metadata端口的分歧随后按下一批收拢，完整模型数值／性能仍未完成。
归档848,997B；按根目录`test.ps1 -ReclaimNativeActiveQueries`核对后回收12个image／map／导出文件
290,878,207B，保留全部对象、SDK、源码、数据与成功／失败日志。

后续[同一owned-layout分页数学图](native-page-semantics.zip)删除旧native Graph描述／factory，
统一为layout＋纯数学选项、GQA三载荷／MLA两载荷；实际Source的批次能力由`batch::compile`
装配进原编译器，原Invocation和原生融合不变，Direct／expert的显式metadata容量合同保留。
同一个F16/Q8_0、D64、H8/KV2双请求Graph分别在CPU/Vulkan运行Q18/Q2/Q0，schema容量始终24。
两槽、layout／请求分配更新、旧输出门、strict同绑定重放、准确页权限和Q0持全部10个pagewriters通过；
独立FP64最大绝对误差6.48184e-5，预算仍为0.002。Vulkan两条Q搬运边合计分别36,864／4,096／0B。
实际调用补出了shared槽token被消费、空读缺少selected-slot证明及CPU同步数学错标非阻塞三处接缝，均已修正。
原ready／pending各64轮、零行Graph＋Direct控制和7项schema／所有权检查通过；18个相关TU增量编过，
其中CUDA host／包装与旧Vulkan联合编译器只编译，不能代签数值。保留两次编译问题和重复CPU Source fixture的失败及修正。
metadata共享只减少上传，各节点mirror/staging冷容量仍保留，Queue外壳仍遵守原冷准备借用合同。
本批未重建SDK／shader／kernel，未新增Apple/Linux运行或完整模型性能结论；旧混合driver和剩余功能继续迁移。
归档421,025B；按根目录`test.ps1 -ReclaimNativePageSemantics`核对后回收16个image／map／导出文件
388,072,075B，保留全部对象、SDK、源码和日志。旧私有编译声明已另经`test.ps1 -RetireBatchAdapter`归档删除，无兼容转发。

后续[Apple普通分页与晚端口](apple-page-common-invocation.zip)用原Invocation承载同一数学Graph：
Air Mac16,12上CPU、CPU→Apple page→CPU、Apple Abs/Cast→page→Cast/Neg三种放置的Q18/Q2/Q0均通过，
含双槽、容量Q24、请求重新分配、strict replay、旧输出门、foreign-layout拒绝和空批持全部10个page writers。
F16/Q8_0、D64/H8/KV2的独立FP64最大绝对误差6.48184e-5，沿用0.002预算；11项原生结果／晚输入／混合回归通过。
原生Array仍由原FIFO晚解析，未新增影子输出或第二运行槽。几何与partial事实归原PagedBatchAccess分析，
Graph只检查数学许可；Direct expert保原精度行为。变KV仍可能走原eager数组构建，plain two-pass F32 partial和Apple pending尚未补。
仅恢复旧复制缓存key的负控在换计划且目标buffer不变时失败：`Apple copy has no new producer submission`；
比较原CopyRecording持有的实际InputSource后通过。97个当前host owner与原SDK链接，map无旧joint或旧Graph归档成员。
真实编译另修了materialized helper的未限定compile歧义及64位页长度检查告警；未重建SDK／shader／kernel，未测吞吐或跑全模型。
[Vulkan MLA普通消费者](native-mla-common-graph.zip)另通过F16 D512/H64与D576/H96的Direct＋Graph原FP64和逃逸输出门，
仅刷新两个当前owner；其余旧Apple／Vulkan驱动及消费者仍须迁移，不能以此批局部结果宣称删除完成。
两包分别467,877B／63,766B；经根目录`test.ps1 -ReclaimApplePageInvocation`逐项核对回收Windows8个及Air4个
可重建image／map／导出文件，共230,693,188B；全部对象、SDK、源码、数据和日志保留。

[原生Source接入收敛](native-source-convergence.zip)补齐Apple冷custom配置、WeightBatch输入及typed pending页接线，
继续复用原Runtime、CopyBatchUpdate、controller、精确页权限和唯一DependentCompletion。
Air当前闭包通过22项相关gtests（原生11、Source4、编码追加4、前沿3），同一页图九种Q/placement组合、
原Q4 FFN／WeightBatch样例及65步sampling；Windows编码原3项、65步sampling和前沿3项通过。
实跑修出了共同同步前沿遗漏外部pending输入，以及新Apple pending适配误用原生Invocation而非CopyRecording资格检查；
原producer失败、已接纳独立工作及页不发布合同保留。测试的精确segment数、每source必poll及CPU存储必须host0等假设已纠正；
采样按真实CPU reader的physical端口核对，仍严格两条／24B，不把Vulkan staging中间跳重复计，也不强搬Apple共享存储。
完整失败、修正及最后七个Air程序的map均保留；无旧joint或旧Graph归档成员。只增量编相关host，未重建SDK／kernel／shader。
720,985B归档后按`test.ps1 -ReclaimNativeSourceConvergence`核对回收Windows16个、Air14个image／map／导出文件，
共517,244,697B；全部对象、SDK、数据与日志保留。source/static/shared交付、剩余旧合同／风险消费者、IO拆分、
Apple plain two-pass F32 partial、完整模型数值／吞吐仍未完成，不能以这批局部结果宣称架构收敛。

## 怎样审查

沿实际模型的prefill、cached continuation、decode、混合请求批次、MTP验证块与接受前缀恢复，
追踪公共API→Provider选择→原生计算→存储与状态反馈。只存在算子或测试源码，不代表整条调用链已验证。

| 部分 | 核查内容 |
|---|---|
| 融合与量化 | embedding/LM head、dense/batched投影、norm/gate/RoPE、MoE路由及expert/combine、GDN/KDA/conv；按实际dtype、block/scale、形状、布局及数值合同核对 |
| Attention与KV | 普通/分页/ragged、MLA共享latent、稀疏indexer/TopK/Gather、逐query可见性与sink、cache编码/追加/窗口/压缩 |
| Graph编译与执行 | 分区/融合边界、路由偏好、常量、别名、内存复用、动态形状、规划/重放与多queue/lane调度 |
| 驻留与搬运 | 权重和状态驻留、host/device桥接、增量上传、重复分配/解码/repack/contiguous/cast/实体转置及同步 |
| 状态与寿命 | GDN/conv/KV/压缩索引的共同版本、fork/COW/回收、接受前缀恢复、pending来源、失败/取消/关闭/排空及输出保留 |
| 接入完整性 | 普通Direct/Graph入口能否表达真实推理；内部布局/Provider细节是否不必要地交给消费者 |

结论分别标明缺功能、可组合但缺高效承载、实现错误、实际资源/数值限制及未验证。
以当前路由和实际执行成本判断，不以现有白名单缩小需求，也不把不同工作、显式参考成本或普通数值要求当作性能缺陷。
先修确定问题，围绕改动做小范围编译、数值/寿命或成本对照；复用现有工具链与依赖缓存，不反复全库重编。

当前已首轮核对模型主路径，并对下表事项深入修复；还不能称为全面完成，也不能用修复条数计算覆盖率。
仍需横向补审全部制品的实际默认路由、CPU＋GPU混合旅程、prefill→续算→decode→接受前缀的连续消费，
以及各路线的动态长度、规划/重放、分配与失败寿命。四后端的局部消费不等于每个模型在每条路线均已闭环；
完整bank个例与轻量缩形状也分别记录，避免将未实际运行的容量或组合计作通过。
这些未完范围继续由新架构承接：删除重复路径、收拢承载分析和权限合同，不等于补齐数学实现或证明模型整链。
完成状态按具体需求、路线和消费证据判断，不从修复条数或历史百分比推算；迁移目标不保留兼容层、旧入口薄包装或双执行体系。

## 模型域核对

以下为仓库选定制品的源码接线审查，不是22个完整模型的运行记录。普通provider仍按具体描述与资源
接纳；不因为模型名相同就假定算子、激活量化或存储格式相同。

| 模型（制品数） | 实际主路径 | 仍需区分的边界 |
|---|---|---|
| Qwen3.5-2B（3） | 官方BF16投影；GGUF的K/IQ4/Q8投影及Q6K或F16 tied vocabulary；GQA、conv、F32 GDN状态 | 官方BF16完整24层CUDA消费者已有三phase执行及token一致证据，精度接受仍未确定；不代签其他制品或完整引擎 |
| Qwen3.8-27B（3） | 同族投影与GQA24/4；实际vocabulary为Q4K→Q6K或Q8→BF16；F32 recurrent | 不把制品级混合dtype压成一个“Q4”标签 |
| Qwen3.8-2.4T-A95B（3） | BF16或IQ4_XS/K/Q8专家、F32 router/GDN；完整bank E512/N2048/K8192 | resident11与完整bank分开；CPU/CUDA压缩bank过度门已修，设备实际容量仍有限 |
| Kimi-K3（3） | 官方FP4无损冷repack为GGUF MXFP4，再以F32 activation作GroupedMatMul；GGUF同族专家，Q8/BF16 dense/vocabulary；KDA | 不误接为DeepSeek A8 FP4；缺3584↔3072的A8制品不构成此模型缺口 |
| DeepSeek-V4-Pro（1） | 官方block FP8 dense、原A8 FP4专家3072↔7168；compressor、Indexer、Sink、Hyperconnection | 无本轮选定GGUF；密集组合、分页原生承载、packed物理历史分别判断 |
| DeepSeek-V4-Flash（3） | 官方原A8 FP4专家2048↔4096及FP8；GGUF MXFP4专家、Q8/BF16 dense；原roundtrip数值边界 | ratio4及ratio128的多query/接受前缀CPU消费见后文，不外推其他后端整链 |
| GLM-5.3（3） | 官方block128 FP8；GGUF Q4K/Q5K/Q8及Q8 vocabulary；吸收后MLA H64、576→512 | CUDA官方E256单projection完整bank已实际验证；不以同dtype代签packed gate/up或全层 |
| GLM-5.3-Flash（3） | 同族FP8/GGUF及KDA；MLA512→512 | E288总bank已无上述CUDA误门，但本轮未另分配完整bank实跑；Vulkan typed BF16分页/紧凑MLA仍未承载，区别于普通dense BF16 SDPA的F32宽化组合 |

主路由为：plain dense由CPU oneDNN／CUDA cuBLAS与相应native浮点路线承载；GGUF投影/专家由各端
ggml承载；官方FP8/FP4由CPU ATen、CUDA SGLang/DeepSeek、Apple MLX、Vulkan IREE等原组合按域接纳。
直接源码锚点是各端`default_policy`、`torch_cpu_program.cc`、`apple_mature_float.cc`、`vulkan_official.cc`
与模型fixture中的实际storage descriptor。GPU GET_ROWS缺的三个IQ1 narrow变体不在所选embedding列表中，
仍作为公共catalog边界保留，不能通过白名单放开冒称已有native行解码。

## 已完成的修复与补全

以下是本轮已完成的局部修复，不代表全量审查结束。测试使用实际现有硬件和缓存原生库，重新编译改动的host/测试翻译单元。

| 事项 | 原因与修复 | 验证与边界 |
|---|---|---|
| CUDA FP8误拒兼容驱动 | 精确driver版本判断与SM89 AOT模块的真实加载能力无关；移除精确版本门槛，保留SM、模块/函数加载及错误处理 | 实际driver API 13040下2/2组件测试通过，覆盖F32/BF16、grouped、重复调用和寿命；13050只是host报告事实的对照，未声称测过另一台驱动机器。[包](fp8-driver.zip) |
| CCCL误拒新SDK/runtime | 删除历史toolkit/runtime上下限和固定CCCL版本选择器；保留实际环境、形状和native configure/launch检查 | 实际环境7/7通过；同一原生闭包报告13050时，旧代码7项prepare失败，新代码7/7通过。此对照不更换真实SDK。[包](cccl-qualification.zip) |
| FlashInfer非正QK scale错误 | 原生softmax实现不能承载非正scale：旧代码scale=0最大绝对误差0.45456，scale=-0.5所有128个输出为NaN；普通及batch入口现在提前拒绝该域 | 新规范测试1/1通过，正scale对照误差5.84e-5。普通SDPA可以选其他合格provider；专用batch明确返回不支持，不伪装存在自动fallback。[包](flashinfer-scale.zip) |
| 状态metadata重复分配 | 每次replace_metadata新建小buffer；现在在原StateStore预算及租约体系内复用空闲、唯一且实际可写的同尺寸snapshot，压力时优先释放空闲项 | StateVersions/StateClone 31/31通过；32步中前2步热身，后30步禁止metadata allocator后仍零新增调用。逃逸只读视图和native use保持不可覆盖；Store销毁后仍可读。未声称整模型延迟收益。[包](state-metadata.zip) |
| CUDA BF16专家常量每次转F32 | 保留F32激活计算合同，仅将Graph不可变专家权重放进现有ConstantArtifactAccess，冷准备一次；动态权重仍按输入处理 | 独立FP64数值/寿命回归1/1通过；实际native图计数为冷准备1次copy、两次常量调用均0次copy、动态调用每次1次copy。[包](cuda-constant-expert.zip) |
| Graph重复output bind扫描全部常量 | 所有后续绑定都已检查既有output范围，重复绑定相同resource/handle/offset/size无需重新扫描全部input/constant/output | 仍执行原绑定、发布及capture失效流程，变化的输入/输出继续检查。新回归及相关Invocation/GraphViews 32/32通过。[包](graph-rebind.zip) |
| CUDA/Vulkan多请求RoPE接线缺失 | 公共schema允许positions[B,S]与frequencies[S,pairs]，原ggml接线只允许共享向量；沿原原生算子补reshape/broadcast，并接受等价的sequence_axis=-3 | CUDA新旧2/2测试通过（F32/F16/BF16），Vulkan1/1通过（F32/F16）；覆盖两种旋转布局、两种位置、两种频率及两次变更输入。原phase F32及输出dtype不变，无新kernel/SDK构建。[包](batched-rope.zip) |
| CPU/CUDA Dense常量RHS每次转置 | 原native MatMul每个invocation构造contiguous转置并每步执行；沿既有ConstantArtifactAccess冷准备一份，动态RHS与原权重其他consumer不变 | CPU2/2、CUDA2/2组件数值/寿命测试通过；CUDA native图计数cold CONT=1、两constant调用各0、dynamic各1。CPU ggml也补齐同域RoPE，并通过F32/F16/BF16回归。未测完整roster/吞吐。[包](dense-cpu-cuda.zip) |
| Vulkan Dense常量RHS每次转置 | 沿原captured Tensor/Program持有和View/Copy冷准备，仅替换MatMul消费者；原权重仍可被Neg等其他节点使用，不新建cache框架 | public graph::vulkan::compile的F32/F16常量/动态、两slot回放数值及原RoPE共2/2通过；两dtype计数constant CONT各0/0、dynamic各1/1。BF16 widening不在此修复内。[包](vulkan-dense-constant.zip) |
| 紧凑MLA缺公共单cache承载 | 新typed MLA schema/独立extension、单kMla arena和2payload Direct/3port Graph；普通独立Q/K/V合同不变。CUDA接原专用CKV512/KPE64内核，避免普通576→512超SM89共享内存 | F16/H96、BF16/H64，两request Q1+Q3、KV31+33、非顺序/重复页/尾页，Direct/Graph及原Queue capture、重绑、预算/整数上界检查通过。单cache73728B，native float/int workspace约5.77/0.56MiB；并非总显存。无sink/mask/F32/encoded支持声明。[包](cuda-compact-mla.zip) |
| MLA H96 causal漏算可见tile | 实测首条输出错误，native scheduler已用ceil而kernel用floor求cluster的query上界；统一为ceil，保留逐element causal mask | 同输入/预算修前maxAbs0.012644，修后F16 maxAbs5.52e-5、BF16重绑最大4.69e-4；原独立Q/K/V consumer也通过。FlashInfer补丁lock/source/NOTICE身份已同步v17。[同包](cuda-compact-mla.zip) |
| CPU紧凑MLA承载 | 沿原flash paged/ragged计算，V为同一CKV buffer的只读strided前512项，保留576行跨度，无V拷贝 | 8组合通过：F16/H96、BF16/H64 × CKV512/CKV512+KPE64 × paged/ragged；每组Direct三episode＋Graph、default Context、Q1 native plan复用、完整FP64数值。最大误差2.44e-4；原Q/K/V三项相关回归3/3通过。[包](cpu-compact-mla.zip) |
| CUDA单节点BF16专家仍重复转换 | 先前cold-pack只在多节点region生效；现在仅对原ggml承载的不可变plain BF16专家权重接纳单节点region，不改变普通dense的oneDNN/cuBLAS选择 | 单节点、多消费者、dense共3/3回归通过；新增单节点cold Cast=1，两constant invocation热Cast=0/0，dynamic=1/1。[包](cuda-single-constant-expert.zip) |
| 同布局反复分配和排序页集合 | 将immutable layout的active页ID排序去重结果按需保留，由K/V及后续读取共享；每个frame/guard仍独立分配和取得实际权限 | 49项BatchLayout/PagedArena/PagedAdmission测试通过；32页同version的8对K/V读取48→32次metadata分配，新version首次读取6→5。OOM可重试、并发首次构造、旧版本与权限不驻留均覆盖。[包](paged-footprint.zip) |
| Vulkan BF16常量RHS仍热宽化 | 原BF16 carrier每个invocation重新Cast权重并CONT；沿原Cast/View/Copy冷准备一份F32 packed常量，仅替换该MatMul port | F32/F16/BF16×constant/dynamic、两slot、原权重Neg消费者及独立FP64通过；BF16冷weight Cast=1，热MatMul weight Cast/CONT各0/0，dynamic仍各1/1。Neg自己的宽化保持原语义，未泛称全图零Cast。[包](vulkan-bf16-constant.zip) |
| MLA默认partial精度合同 | 原生MLAParams曾把F32累加结果收窄为输出类型的split partial；改为F32 partial并按4B计费，直接从原register布局存储，无shared膨胀 | 真split BF16 Q1/KV8192，正负16384抵消、首row额外256，全32768输出的独立期望值0.03125，当前maxAbs=0；仅恢复旧partial16的对照明确失败。F16/BF16×512/576四组合Direct/Graph通过。[最终包](compact-mla-final.zip) |
| Apple/Vulkan紧凑MLA接线 | 公共typed单cache传到原native/controller与独立Graph，V只读别名CKV前512项、保持原512/576行跨度 | Apple四组合（F16/BF16×512/576）、Vulkan F16两组合，均Direct＋Graph→Neg并发布两结果，完整FP64通过。单cache65536/73728B，不含workspace；Vulkan BF16仍未承载。[同包](compact-mla-final.zip) |
| Vulkan page输出接下游时owner基线失配 | prepare_batch过早记录输出owner数，后续Graph内部GPU绑定增加合法引用；现在准备前原reuse_gate拒旧结果，内部绑定完成后专用reseal | MLA→Neg成功；保留旧OwnedResult或escaped Tensor仍拒绝prepare，实际释放后可重绑和复用；公共submit的owner equality未放宽。[同包](compact-mla-final.zip) |
| 双MLA节点的page-input越界比较 | 原NativePagedSlots共享页检查固定遍历三个tensor输入，MLA只有两个；改按各自真实port数 | 两纯host回归检出旧源码2个断言失败，新canonical TU共5/5通过；覆盖共享cache同layout、异layout与独立cache边界。[包](mla-slot-bounds.zip) |
| CUDA单cache pending追加未接 | 原公共PagedBatchReceiver直接拒绝MLA且只有独立K/V submit；新增单cache overload，与旧入口共用原source/Queue/页面权限及聚合退休逻辑 | F16/BF16真实gate producer→F2→MLA在gate释放前返回submit，完整FP64误差0，拒绝保Update、旧output阻复用、页writer恢复均过；旧CPU receiver九种风险/成功模式全部通过。[包](mla-pending.zip) |
| CUDA专家/量化embedding整ID回读 | 原ggml每次把全部I32 IDs读回host并stream同步；现在局部safe-ID leaf与原Queue device-status结合，整region公开结果只在校验成功时发布 | 最终9项组件/Graph回归通过，含旧18格式embedding、Grouped/BF16、非法ID与多输出、同ID不同extent、两slot及失败寿命；unknown private预算保持。未声称ggml内部TopK融合。[包](cuda-device-indices.zip) |
| 多token MoE路由产生T²中间结果 | 新独立GatherElements逐row选择[T,E]分数，保留旧Gather语义；模型fixture改为TopK自身合法IDs直接选择，只生成[T,K] | CPU/CUDA/Apple/Vulkan三浮点类型、Checked/Unchecked、K>E、重复IDs与变更回放均有独立参考；GPU公开Graph→Neg双结果也通过。真实默认CUDA为CCCL TopK→ggml两step，默认CPU三leaf；无整模型吞吐声明。[CPU](gather-elements-cpu.zip)、[CUDA](cuda-gather-elements.zip)、[Apple/Vulkan](gather-elements-native.zip) |
| MLA按最大工作表预留scratch | 原planner无split也预留约12 MiB partial；现在按实际work/partial项计费，保留16384工作项准入边界、cluster和merge完整容量及F32 partial | 四种小计划用16字节float＋4 KiB integer通过Direct/Graph/更新；真实BF16 split需求4,202,496/2,172字节，精确容量通过、任一少1字节prepare拒绝，敏感残差maxAbs=0。默认slot容量未自动缩小。[包](mla-active-workspace.zip) |
| CUDA完整FP8专家bank误拒 | 原adapter将整个E×N×K限制为I32；原AOT按I64计算专家基址，实际只有单expert的N×K需该限制，删除过度总量门槛 | GLM形状E256/N2048/K6144、codes＋scales共3,222,011,904字节，实际选expert255（codes偏移3,208,642,560）并重复数值验证；新旧组件2/2通过，非法ID仍返回原Completion错误。未分配更大的packed gate/up bank。[包](cuda-fp8-full-bank.zip) |
| CPU/CUDA完整GGUF专家bank误拒 | 仅对canonical rank3压缩Grouped权重按实际block寻址域接纳，保留单dim/单expert stride、压缩block索引及其他端口原限制，不全局删除I32门 | CPU3/3、CUDA5/5通过。E512/N2048/K8192 IQ4_XS共4,563,402,752字节，高expert511偏移4,554,489,856；CUDA实际整bank分配、仅初始化所需expert并两次核全部结果，CPU保留完整虚拟范围、仅commit8,912,896字节后两轮maxAbs=0。Q8仅prepare，Vulkan真实单buffer范围仍拒此bank。[包](gguf-full-bank.zip) |
| CPU＋CUDA权重每步重复排序上传 | WeightBatch的原upload尚未被观察时，即使native event已完成仍被当成pending；在保留Context和Queue admission检查后先非阻塞poll，成功ready时空sequence完成 | 实际CUDA同一ready batch八轮WaitEvent/EventRecord从8/8变为0/0；真pending仍1/1。两项canonical测试通过，另覆盖busy/closed/foreign及源late-error；错误保持sticky且不提交后继，未测整模型吞吐。[包](cuda-weight-order.zip) |
| 跨provider导致前后融合片段一起丢失 | CPU/CUDA ggml discovery只提一个全图eligible集合，经过TopK/FlashInfer等外部节点后整体不合法；改为沿数据与控制边线性切分合法prefix，独立穿插不切分 | 真实默认CPU/CUDA的数据、控制两种边界均由0 region/5 step变为2 region/3 step，独立TopK仍1 region/2 step；所有输出两轮精确对照。canonical CPU2/2、CUDA4/4相关测试通过。region不等于一个物理kernel，未测吞吐。[包](ggml-region-discovery.zip) |
| 同形reshape阻断原生双投影融合 | CUDA每个node恢复shape时即使ne相同也新建RESHAPE，双投影失去同一src1身份且native节点不连续；仅在原contiguous且四维ne完全相同时直接返回 | Q8_0同输入gate/up→SwiGLU，默认八provider图仍1region/1step，但native图8→3节点、5→0 RESHAPE；CUPTI原生kernel发射记录从5变2，单独发布gate/up仍5。两轮完整FP64 maxAbs2.70e-8，canonical3/3通过。非吞吐或CUDA Graph重放测量；对应GLM GGUF denseFFN同语义缩形状，不声称Kimi SiTU或完整GLM已融合。[包](cuda-native-shape-fusion.zip) |
| Apple FP8先展开全bank scale | Grouped路径先将全E的block scale repeat为每row，才take本次IDs；将codes/紧凑scale的选择移到repeat之前，保持原scale乘法及量化次序 | 实际managed图的展开expert维64→本次8；N257含尾block，两phase各2056结果、重复高ID，新旧全部4112个F32逐bit相同，原预算maxAbs=0；三种非法ID仍OutOfRange，旧结果保留。未测物理peak或吞吐。[包](apple-selected-fp8-scale.zip) |
| CPU先复制所选专家完整权重 | ATen grouped先index_select全部K，再逐block计算；改为先借用bank的当前K块，再选IDs，scale也先选当前block/IDs再展开FP8 row128 | canonical五项通过；T3/S4/E17/N129/K512，FP8 F32和FP4 BF16、共享/独立activation、重复高ID、两phase全部结果与独立FP64精确一致，越界仍拒且不写输出。实际selected codes最大物化量FP8 792,576→198,144B，FP4 396,288→24,768B；每block仍有decode/计算临时量，不声称总体峰值或吞吐。[包](cpu-grouped-block-selection.zip) |
| Apple完整FP8 bank受一维carrier限制 | 大GroupedFloat8权重改为U8[bytes/4,4]私有carrier，按4B行slice后显式reshape/view，不经总长度int flatten；只豁免该weight端口的整bank门，保留单轴/单expert及其他算子限制 | 小case旧1D与强制2D的4112个结果逐bit相同。实际GLM E256/N2048/K6144完整3,222,011,904B bank读取expert255，codes偏移3,208,642,560B；两phase各32768输出原预算下maxAbs=0。warm大buffer退回MLX cache后被实际bank复用；未测peak或吞吐，按token复制所选权重的问题另查。[包](apple-full-fp8-carrier.zip) |
| Apple按token复制完整FP8专家 | 沿同一个SIMDgroup kernel加私有INDEXED模板，直接用safe IDs及64位stride寻址原encoded bank，scale按column/128读compact bank；不再Take出每个token所选完整权重或展开scale | T3/S8/E64/N257/K256两phase共12,336输出及普通GLM linear对照12,288输出，新旧24,624个F32逐bit相同；三种非法ID和旧输出寿命不变。managed图47→39节点，权重kernel输入保持E64而非T×S24，Gather/Repeat权重均消失；未测总peak/吞吐，activation广播仍可能物化。[包](apple-indexed-fp8.zip) |
| Apple FP4先整份展开再计算 | 复用同一indexed kernel，以原32项分组直接解码bank中的低/高nibble；移除完整选中权重Take、E4M3转换、F32 decode与全K块dots，保原A128量化、A/W scale次序及BF16最终舍入。大canonical FP4 bank也使用二维私有carrier | E64/N257/K256、T3/S8、A1/AS双phase，24,672个BF16结果与独立FP64→BF16精确oracle、旧路径及强制小2D carrier均逐bit相同；原tiny/非法ID/旧输出寿命及FP8回归通过。Pro 5,284,823,040B仅核InputSpec，未prepare/分配；无峰值或吞吐声明。[包](apple-indexed-fp4.zip) |
| Vulkan不可变专家bank按slot重复驻留 | 仅为ready、直接owned的官方FP8/FP4 Graph常量建立只读borrowed HAL view，保真实native owner；各slot codes及对齐scale共享原buffer，动态及未知view仍复制 | 真实公开Graph两slot、三小case，数值/变更重放/输出保活/最终owner回收及三项非法参数通过，旧dynamic grouped也通过，共2/2。native对齐16、有效门64；人工收紧到256验证scale回退复制，非真实硬件反例。未松原寻址门、未跑完整bank/峰值/吞吐。[包](vulkan-official-bank.zip) |
| 每query不同KV只能拆成多个Sink调用 | 原SinkAttention仅B1；扩大为相同正batch的Q/K/V，支持共享[Q,S]或独立[B,Q,S] mask，保F32/H64或128/KV1/D512。三ggml端统一native mask轴及必要的tile打包，Apple短/长路径保持batch轴 | 四端8几何×两phase通过独立FP64；CPU/CUDA另实跑shared cache→Gather IDs[B,S]→一个Sink→Neg双publish。CPU/CUDA真实小tile、Apple128+1及Vulkan2+2+1强制小tile通过；两强制项不代表产品收益策略会选中。无分页/新融合kernel或吞吐声明。[合同](batched-sink-contract.zip)、[CPU/CUDA](batched-sink-ggml.zip)、[Apple](batched-sink-apple.zip)、[Vulkan](batched-sink-vulkan.zip) |
| Vulkan退化KV-head stride误触降型分支 | contiguous K从[D,1,S,B] permute后，单head的nb2仍是D行跨度；native的batch pitch判断不忽略该退化轴，B>1/Q3时误入缺失的F16 map并assert。仅在合法contiguous且退化stride需规范时same-shape reshape | 旧实际K ne[512,17,1,3]/nb[4,2048,2048,34816]，规范后nb2=nb3=34816；无CONT搬运，F32类型/精度参数均不变，四类组合的8几何及双publish数值通过。[同包](batched-sink-vulkan.zip) |
| CPU有限sink的合法全屏蔽行触发断言 | 原softmax在加入sink前检查普通分母为正；全屏蔽行此时为0，有限sink加入后才为正。将同一断言移到原sink加法之后，保留原计算与无sink检查 | 同一mixed456原生ops TU启用标准assert：旧S1/H64例在sum>0处真实退出，新例两phase全部32768输出为0，原Sink/普通attention共3项通过、预算不变。仅该TU前后各编一次，其余Release闭包复用；canonical FFA仅精确补丁重放与身份核对，未声明完整FFA Debug SDK运行。[包](cpu-sink-denominator-debug.zip) |
| CUDA Sink缺多节点分块region | 原Graph主动排除Sink；其Compute分块实现也使用不适合下游依赖的独立输出leaf。现仅接纳同数学的Sink region，以真实平衡Concat依赖连接各块，按node规划并给cold scalar/dense与safe-ID leaf独立存储，再复用原gallocr | 真实B8/H128/Q3/S2049小tile→Neg双publish通过；不同Q的双attention/两slot/冷常量，以及checked IDs失败保全部输出通过，另强制小tile两项核实际gallocr叶存储。普通SDPA继续独立排名以保FlashInfer/Torch融合路线。本阶段是组件公开Graph，非默认全roster/峰值/吞吐证明。[包](cuda-attention-regions.zip) |
| CPU＋CUDA同一上传依赖按port重复排序 | 同一不可变WeightBatch绑定多个ValueId时分别排入相同pending依赖；冷prepare按原WeightState总序排序，稳定列表仅线性is_sorted检查，提交线性合并同state，保每port引用及原prefix退休 | 持有真实上传gate至全部order完成，实际cudaStreamWaitEvent同batch两port由2→1、不同batch保持2；取消、ID交换、rebind/reset、OOM及原合同通过。统计shim仍调用原API，不是时延/吞吐测量，delegated host wait保留。[包](hybrid-weight-dependencies.zip) |
| Apple冷准备分配整份encoded bank | 官方Grouped FP8/FP4的隔离占位改为单U8零广播，原生reshape识别单实存元素以避免INT_MAX分组后整份复制；真实bind容量与scale CONT不变 | Air公开GLM FP8 3,222,011,904B规格prepare在128MiB native limit内成功，active24/cache956,693/peak956,713B；旧归档同阶段cache3,326,810,389B，未重跑旧大分配。5项canonical、非零/负stride控制及原FP8/FP4 bitwise数值/非法ID/寿命回归通过；未分配真实bank，Pro FP4 scales仍约1.057GB。[包](apple-prepare-placeholder.zip) |
| MLX清单版权字符损坏阻断依赖派生 | 两个native文件的四组整文件edit首行将©损坏为漏，导致原锁定源码匹配失败；只恢复8个before/after字符，原18文件最终native内容哈希不变 | 从原MLX1f8/0.32.2 archive逐项重放全部19文件及before/after SHA通过（含新增common修复），仅同步对应patch tree/NOTICE身份，不重编SDK或改变原数学。[同包](apple-prepare-placeholder.zip) |
| CPU＋Apple/Vulkan重复创建权重alias | 已绑定的不可变WeightBatch每次prepare仍创建native alias metadata；现在复用既有native输入，显式rebind及其他绑定路径仍清除旧alias并重新准备 | Apple、Vulkan完整合同通过；仅禁止WeightSlots metadata allocator后，变更activation的重复prepare/执行仍通过，显式weight rebind按预期OOM、恢复allocator后重试成功。原取消、输出保活和退休语义保留；不把metadata复用称为payload上传优化。Vulkan旧缓存adapter的私有Recipe ABI不匹配曾导致前置CPU段崩溃，只补编该host TU后原完整测试通过。[包](hybrid-weight-alias.zip) |
| 逐行压缩历史缺独立编码和选中行解码 | 新RowCacheEncoding与Encode/GatherDecode，主KV每行584B、index历史每行68B；CPU/CUDA只编码所给新行、只解码选中行，保原StateStore/F2及8B CacheStorage合同 | schema新旧7/7、CPU两项、CUDA组件及默认路由四项通过，双格式完整bytes/选中BF16与独立FP64 oracle一致；非法ID及非有限编码输入不写结果，含signed zero、极值、空输入及奇地址carrier。另有下文真实StateStore→Sink消费。布局是Hyle显式row-inline ABI，并非官方或原生分页布局。[合同](row-cache-contracts.zip)、[CPU](row-cache-cpu.zip)、[CUDA](row-cache-cuda.zip) |
| CUDA普通Checked Gather每次同步读回bounds | 原CCCL以两次Min/Max和16B D2H判界，再同步stream后才Gather；现复用原CCCL索引/字节布局，以同stream全ID校验、Queue device-status及门控Gather写入，去除这段回读/同步和临时workspace | Direct相关8项及现默认Graph原有2项通过；全部15 storage types、I32/I64、轴/非整齐stride/broadcast及raw bits不变，非法末ID保全输出、fresh slot恢复、空与Unchecked合同均过。真实producer gate下旧提交被阻塞，新valid/invalid均先返回pending，释放后分别成功/OutOfRange；仍为原managed runtime、不扩大外部capture支持，原Scan/TopK不改。非吞吐或完整Kimi执行证明。[包](cccl-device-gather.zip) |
| Apple scale视图仍实体化整区 | 原FP8/FP4 wrapper无条件对scale slice做CONT；即使冷codes占位已是1B，U8广播转F32也会整份复制。去掉scale CONT，并仅对单实存byte重复为完整word的情况，用原Scalar Copy构造一个真实4B word后作零stride视图；真实dense绑定沿原owner/offset共享 | 原View control失败、新canonical6/6及普通/transpose/reversed/empty控制通过；Pro FP4 5,284,823,040B规格公开prepare-only的native allocator peak268,509B，在128MiB cap内，无真实bank分配。原FP4 24,672 BF16、FP8 grouped/linear 24,624 F32与历史bytes一致，tiny/非法ID/旧输出寿命保持。20个MLX文件精确重放与lock身份通过；仅相关host TU，无新shader/SDK重编，不是整模型峰值声明。[包](apple-prepare-scale-view.zip) |
| 压缩行历史缺真实页几何与权限句柄 | 新typed RowCacheBatchSchema/LayoutVersion与PagedRowCacheRead，复用原layout owner、F1独立页、F2和初始化前缀，真实U8[P,page_size,584或68]，不伪造attention容量或whole-arena lease | 相关71/71及四个公共头独立编译通过，覆盖两编码、缺/多/外来/子范围页、旧分支、原native use阻回收、空历史及OOM重试。7个受影响host TU fresh；该包仅核页合同与Graph错误域，跨页decoder、pending及Sink的后续证据分阶段列于下文。[包](row-cache-pages.zip) |
| Apple逐行codec接线缺失 | 沿原MLX整数/bitcast/Take和private status补Encode/GatherDecode，编码新增行、先Gather选中压缩行再解码；保raw BF16尾部与signed zero，整数处理极小scale避免FTZ | Air Direct/Graph共4/4，双格式两phase逐bytes/BF16bits对齐共享独立oracle；两slot/重绑/保留旧结果、奇byte origin、非法ID/非有限输入、空/scalar及保留编码通过。只编相关4个host TU，无新SDK/kernel；原INT_MAX承载及普通U8冷bank占位仍在，不是分页reader或全模型证明。[包](apple-row-cache.zip) |
| Vulkan逐行codec接线缺失 | 一个原体系U32 residual shader，先全输入/ID校验再门控编码或仅选中行解码；整数scale/dyadic BF16重建保RNE/子正规数，无ID回读或整历史展开 | Direct及公开Graph共3/3，双格式、两phase/两slot/旧结果、全bytes/BF16bits、非法末ID/非有限源保全输出、奇地址和特殊编码通过。仅相关host TU与9个轻量residual shader重编；新shader仅Shader capability。保设备descriptor/U32寻址/grid限制，尚无分页联合消费/吞吐声明。[包](row-cache-vulkan.zip) |
| 常量BF16→F32中间值仍热转换 | 沿原artifact冷准备不可变rank2 BF16 Cast，供已选ggml多节点region外的消费者只读借用，不改GEMM选择；公开Cast/View链和动态输入仍走原路。复查发现的control→prepared Cast→data consumer丢边一并修正，两处producer表保逻辑生产者 | CPU/CUDA各1项六mode数值/两phase两slot/源释放寿命通过；实际GLM[24,16384]权重cold CAST1、两次hot CAST/COPY/CONT各0。控制边旧例明确失败，修后CPU/CUDA各1项含真实提交通过；最终调度修复后未重跑旧宽测/census。F32 artifact1,572,864B，原BF16786,432B仍保留，不是空间/吞吐声明。[最终包](derived-constant-cast.zip) |
| Apple typed行decoder冷准备整bank占位 | 仅typed Direct RowCacheGatherDecode在原私有RegionPlan标明operation，只有bank port用单U8广播占位；真实绑定检查、普通U8和opaque Graph保持原行为 | Air 153,092,096B bank规格、16MiB native cap下prepare-only峰值126,317B，无真实bank分配；原tiny realbank/Graph4/4重放通过，含变更真实输入、奇地址及旧结果。只fresh context/operations/probe host，无新SDK/native；此改动不解决opaque Graph整bank冷占位。[包](apple-row-cache-cold.zip) |
| 共同MLA Graph不能表达已有Direct窗口 | Paged/RaggedMlaAttention补同一window_left合同与选项校验，Graph provider按实际descriptor/program窗口匹配，移除强制全窗口门；descriptor payload版本升v2，无新native数学 | CPU公开consumer两例通过：BF16 H64/D576 paged window0与F16 H96/D512 ragged window7，每例三次Direct及一次Graph，两个request含Q1/Q3、Q1更新和页交换；独立完整FP64最大误差0/6.076e-5，原预算及Q1计划复用保持。只编wrapper/consumer两host TU，GPU数学资格不扩大。[包](mla-graph-window.zip) |
| Apple TopK专家ID仍重复CPU扫描 | 仅原TopK的I32索引输出、可经保值reshape、同SSA且selected轴长度恰等于consumer专家E时，冷计划证明范围；private native入口再核port2/E。外来ID、values、不同E、hash与Direct保持原检查 | Air counterfactual强制旧scan、计数candidate及无计数canonical均2/2；原CPU loop由12次/39ID降为6次/15ID，减少恰为6次同E调用，无新GPU检查/等待。小E4完整bank实测；历史MoE缩bank的E与完整router不同，不适用且未冒称全GGUF覆盖。原ready/native同步保留，无吞吐声明。[包](apple-topk-index-proof.zip) |
| Apple/Vulkan中间常量仍热宽化 | 两端原captured owner冷求一次rank2 BF16→F32，保原Node/PlanStep及control边界；Apple捕获ready叶，Vulkan在原RHS packing前传播derived View，不改变GEMM选择或精度。公开Cast/View、动态与原BF16消费者保原合同 | 两端新旧同6小控制各1/1，独立FP64/两slot/输入变更及旧输出保持。Apple实际hot AsType1→0，原Permute仍view；Vulkan实际hot Cast1/MatMul CONT1→0/0，TopK自己的CONT仍在。原BF16与派生F32按消费者需要保留，cold peak未知，不声称整模型/吞吐或Metal私有pack消除。[包](native-derived-constant.zip) |
| 压缩历史缺按请求跨页选中行读取 | 该阶段Direct PagedRowCacheGatherInvocation在原QueueSequence中将逻辑ID映射为physical row，再复用原checked codec；零query分组正确跳过，越请求长度置无效ID，保准确页/代际/初始化前缀，无whole-arena lease | 默认CPU1项/CUDA2项通过，双格式/两phase/换页、K0空footprint、逻辑越界但物理合法ID不写输出。CUDA gate下先返回pending，drop caller owner后backing保活，used页阻复用而unused页可用；mapping已接受后decoder拒绝仍返回实际prefix并保权限。metadata冷同步write显式计费；该包为ready Direct及I32物理行域，后续Graph、pending和其他后端证据另列，不能据本包代签。[包](row-cache-paged-reader.zip) |
| CPU dense FP8按token重复解码权重块 | 移除共享矩阵的token轴expand，保留单份encoded/scale，沿原ATen matmul及scale广播；原K128累加和输出精度不变 | canonical6/6；新增T3/N129/K256覆盖encoded、normal与power2动态激活，两phase/两slot与独立FP64完整结果精确一致。同一例实际24个block观察的F32 decoded payload从198,144降为66,048B/block，encoded从49,536降为16,512B。仅组件native context实跑，默认route另核源码，不是整过程峰值/吞吐。[包](cpu-dense-fp8-block.zip) |
| Graph生成的IDs不能直接消费typed压缩页 | 新薄PagedRowCacheGather宏展开原Map→checked codec，沿原LeasedInvocation保准确页权限及元数据；冷核同一source Graph、原CPU/CUDA普通provider、唯一容量reader，拒绝私有容量发布/alias/复制，无新executor或whole-arena lease | CPU6/6含原5项分页host回归、CUDA新项1/1；TopK→跨页FP8解码→Cast/View→单Sink→Neg，四输出、两phase换scores/query/cache/pages，独立FP64预算不变。FP4 K0空footprint、额外reader/容量发布、非空选择配空历史控制通过。ready frame限定，cold metadata同步write；未声称pending、混合route或整模型。[包](row-cache-paged-graph.zip) |
| Vulkan dense FP8按query展开当前权重块 | 原IREE三dense导出改为A[R,128]及单W[N,128]，沿同pin的原linalg.matmul转置索引计算；保持K128顺序、activation/weight scale次序及最终BF16。generator、MLIR、实际VMFB、manifest/embedding与lock均更新 | 原/新各4/4，含T3/N129/K256 F32/BF16×normal/power、两phase/两slot独立FP64精确结果及原encoded/grouped回归。实际compiled stream allocation的W块从198,144降为66,048B，含A为199,680→67,584B；原生VkDeviceMemory peak均134,288,448B，128MiB池掩盖物理峰值差，未声称峰值/吞吐胜出。25个math schedule全部保IEEE32 mode；后续grouped制品另记。[包](iree-dense-shared-block.zip) |
| Graph不能消费pending压缩行页 | 新typed PagedRowCachePendingRead复用原pending页身份/覆盖合同，原PendingInvocation在准确decoder lane等待CopyBatchUpdate后才准入ordered页读取；empty footprint仍沿ready frame，无新executor/whole-arena lease | CPU7/7、CUDA2/2，含原ready TopK→Sink及5项host回归。两编码×整2行新页/1行新尾页四episode，真实CUDA producer gate下submit先返pending，放行后BF16/CastF32/Neg全部raw bits一致。缺末byte的新增行拒收且保Update，原writer取回后补齐重试；输出副本阻控制回收。新增共享路径的两处页几何误取已改真实geometry。旧CUDA缓存fast-math导致的Neg FTZ按现有precise-unary设置仅补编1CU排除，不作为新产品缺陷。[包](row-cache-pending-graph.zip) |
| Apple/Vulkan缺typed压缩页Direct读取 | 两native namespace补同名薄reader，在一个原Program内Map→原checked codec，沿原OwnedResult/页Use保活；不伪装common Context。Apple非法ID的求值替代行只取准确授权footprint首行，避免误读未授权page0；普通codec保持原行为 | Air5/5、Vulkan4/4含原codec回归；两格式/两phase、zero-query/zero-KV段、重复高页、K0、逻辑非法ID及布局错配全部按独立oracle核BF16 bits。page0故意未初始化/无权限；释放caller reader/frame/arena/IDs后原native结果仍正确。每新version整metadata F2冷copy+wait、每次binding/result仍有费用，无Graph/pending/capture或零分配声明。[包](row-cache-paged-native.zip) |
| Vulkan官方完整bank受signed byte寻址限制 | 仅IREE grouped codes改私有32位word视图和取码，公共opaque格式/bytes/scale偏移不变，无重排；保持原workspace门，独立校验word数与真实descriptor字节上限。外层Graph只为精确格式和grouped weight port0放开逻辑I32门，普通producer/大U8不扩大 | 最终小例旧2/2、新2/2；GLM E256/N2048/K6144实际3,222,011,904B完整bank新1/1，两slot/两phase独立原oracle通过，expert255 codes偏移3,208,642,560B、scale偏移3,221,225,472B均实际消费。两slot借同VkBuffer，bad256拒绝、旧输出及owner回收保持。实际FP4只小例，未声称完整FP4 bank、吞吐或峰值；离线制品及全部IEEE源证同包。[包](vulkan-official-word-bank.zip) |
| CUDA紧凑MLA缺ragged承载 | 沿原MLA planner与24个已有kernel实例，以packed KV indptr直接定位单份512/576行；私有linear标记允许unit-row且16-bit时省去页表，普通paged仍拒绝缺失页表。V继续只读别名CKV前512项，无虚构页表或K/V展开 | 新1/1覆盖F16/BF16×H64/128×D512/576、两请求Q1/Q3及Q1/Q1、Direct/Graph共32次完整独立FP64核对；非零第二请求起点、split workspace不足、双slot旧输出保持通过。最大绝对误差F16 5.48e-5、BF16 4.25e-4；原paged/capture/换页及敏感F32 split回归通过。仅新编1个已有实例CU和相关host TU，不增加窗口、mask、sink或编码cache资格。[包](cuda-ragged-mla.zip) |
| Apple recurrent热调用无条件分配两个容器 | 将最多8个输入的CONT owner和最多9个BufferView改为固定栈数组；只有实际CONT时保留原async_eval所需vector。保留MLX等待、跨runtime就绪、checked状态及逃逸输出owner | Air旧/新计数版及无hook当前产品均5/5；15次实际调用中两个指定容器的分配共30→0。GDN T1/T17、KDA、strided CONT、取消/放弃及CONT接受后checked ID失败退休覆盖；未称整个run零分配、吞吐提升或可删除同步。[包](apple-recurrent-metadata.zip) |
| 压缩索引历史缺直接评分入口 | 新RowCacheIndexerScores显式接BF16 Q/权重、Hyle FP4行内bank与逐query半开范围，保dot/product/headsum的BF16边界。CPU沿原ATen以至多256行BF16 tile评分；keytile外层共享已解码行，只读所有合法range的并集，并跳过空gap，不解码整段历史 | 最终默认CPU4/4，含T3/L513两phase两slot独立oracle、三类非法末range全输出保护、T0/L0及T2/L8193两端各1行。v2实测decoder旧1023→新513次；C10最大请求65,536B，私有F32输出6,156B，无完整decoded bank请求；分配请求27→32，总bytes同470,662，真实OOM保护及fresh slot重试通过。v3仅补空gap跳跃及最终canonical，未重跑census；完整score初始化仍O(T*L)，未称时延/RSS收益。该包为CPU contiguous bank；其他后端与typed分页后续证据另列，不等于完整sparse模型链。[当前包](packed-indexer-cpu.zip)、[v2计数包](packed-indexer-cpu-v2.zip) |
| CUDA缺packed indexer普通承载 | 新四port操作沿原SharedQueue复合256-key BF16解码tile、cuBLASLt F32累加/BF16输出及已有ReLU/乘权/归约；所有range先在设备检查，原checked终态及accepted-prefix寿命保留，无新executor/逐tile主机等待/整历史解码 | 默认Direct/Graph新3项及原codec2项共5/5；T3/L513、256+256+1、严格BF16阶段oracle、两slot/两phase/同slot重放、变更输入、非法末range全输出不变、T0/L0与全空区间通过。实际caller workspace165,632B/slot，含64KiB keys与96KiB dots，Lt workspace0；vendor内部分配未知，无额外私有T*L输出。仅SM89实际验证，无独立capture/吞吐声明；当前GEMM仍遍历全部L的tile，不能声称计算仅随可见range增长。[包](cuda-row-cache-indexer.zip) |
| Apple/Vulkan缺packed indexer普通承载 | 四port接原native Direct/Graph与checked发布；完整range校验先于bank读取。Vulkan按可见query/key对直接评分；Apple改用共享256-key BF16 tile和原MLX matmul，保留显式BF16乘权、逐head F32累加及最终BF16边界 | Apple新3＋原codec4共7/7，Vulkan新3＋原codec3共6/6，含两phase/两slot/重放、变化origin、非法末range、旧结果、T0/L0及NaN/Inf。Apple最终矩阵版同7/7；固定T3/L513子域同进程9对AB/BA均更快，配对ratio中位数0.84786，不能外推整模型。跨进程漂移及全部异常值保留；resident dense不含预解码成本，不能据此判输赢。Apple实测active peak增量342,695B，保分块及拼接两份score payload，不称仅768B总空间。[功能包](row-cache-indexer-native.zip)、[Apple矩阵包](apple-row-indexer-matmul.zip)、[公平对照](apple-row-indexer-paired.zip) |
| CPU按token重复解码同一专家块 | validated IDs按expert分组，再按出现次数分桶，以原ATen BMM共享一次解码；保持K块、scale次序及原输出cast。共享activation先解码当前块再选row；无重复专家保持原batched路径，不按最大次数padding | native 6/6及普通default CPU route 2/2，T1/T3、FP8 F32/FP4 BF16、A1/A=S、尾N129、重复/高ID、变化输入、非法ID全输出保护和失败/并发原回归通过。精确产品函数T128/S4/E17/N129/K512同工9对AB/BA：A1 FP8中位26.25→2.56ms，FP4 122.74→15.62ms；C10最大请求33,816,576→990,720B和8,454,144→264,192B，更多小请求、累计bytes显著减少。T1无重复保持原C10请求，小样本时延有噪声；非整模型/RSS或vendor heap测量，Linux/macOS未重跑。[包](cpu-grouped-expert-reuse.zip) |
| CPU分页评分及首次prefill padding未闭环 | 新typed五port评分按request映射实际FP4行页，输出请求内score；每request最多256行tile共享解码并跳空gap。新独立padded decoder仅将-1视作无bank读取的BF16正零，宏产生0/-Inf mask，非法column/range仍失败；原checked合同不变 | 最终5/5，真实评分→TopK visible0/1<K3→FP8 sparse与raw未来窗口→B3/H64/D512 Sink→Neg，检查全部中间/输出；空历史零ReadLease footprint、C257跨tile、两slot不同Q分组、旧输出、非法输入sentinel和private/alias发布拒绝通过。保真实页权限，不造wholearena lease；未测吞吐/总peak，不含完整模型或pending评分。mask可能是accepted-prefix写，不承诺失败Graph原子回滚；CPU为首条闭环，其他后端资格独立。[包](paged-indexer-cpu.zip) |
| CUDA缺安全的padding行消费 | 独立padded codec沿原decode kernel默认false模板扩展，不改变旧Program/checked语义；private Map为-1或不可见合法列生成padding和mask，坏metadata/range/列生成-2使checked decoder失败 | 最终7/7（新5＋原codec2），默认Direct/Graph两format、odd bank、两phase/两slot、旧结果、非法末ID全decoded保护、L0全padding/K0以及真实typed Graph visible0/1<K通过；空history执行时同时持全部4个真实page writer，证明无页面读取权限/载荷访问，不等于零VRAM。无D2H IDs/逐tile hostwait，未测吞吐/peak；本包不含typed CUDA评分。失败mask仍按accepted-prefix不可消费。[包](cuda-row-cache-padded.zip) |
| 新row-cache接线破坏组件配置/头交付 | 三份public头未登记而误归compute；中性Graph为检查ID引用可选provider工厂头，graph_batch新row节点又缺直接compute_ops依赖。登记准确header owner、仅加私有schema边；两既有ProviderId移至同一中性来源，工厂alias与原权限检查数值不变 | 真实graph_batch最小factor configure原先直接失败；修后0.1s通过，三份header owner/所属安装列表均核对。仅四个core头的C++ TU在无可选工厂头下编译通过；没有新增CUDA/native factor依赖、vtable变化或SDK/全库重编。源码快照含独立施工中的cold-span代码，本包不代签其数学/完整状态。[包](row-cache-header-ownership.zip) |
| CUDA typed分页评分与请求跨度接线 | 同一原SharedQueue按真实request的Q span准备Lt；每request共享至多256行BF16 keytile，不按request重复全T计算。layout变化只在idle冷配置，失败使旧batch不可提交；准确页权限及原BF16阶段保持 | 默认CUDA分页4/4、原contiguous Direct/Graph2/2、共享CPU5/5通过，含首次prefill、空历史、C257跨tile、两slot与同slot跨度重配、OOM重试和非法range保护。C4/T3实际报告caller workspace33,557,504B，含32MiB Lt上限；capture=false，未测peak/吞吐。非空request仍逐完整declared history tile做GEMM，不声称稀疏计算复杂度已解决。[结果](../../../../.scratch/inference-foundation-review-2026-10-18/cuda-paged-indexer/summary.json) |
| IREE grouped逐slot解码权重临时量 | 将原FP8/FP4 word取码及decode嵌入同一原生F32 reduction，移除强制物化的W[R,N,G]及后继batchmatmul，保持K块、scale和BF16次序；实际generator/MLIR/VMFB/embedding/lock链均更新 | 当前完整制品Direct4/4、Graph2/2通过。独立单函数固定T128/S4/N1024/K128点的9对AB/BA中，两格式均较快，实际各Context peak为FP8 408,094,976→141,494,528B、FP4 272,354,336→139,054,112B；这些不是完整制品重测或模型收益。仍按slot读取/解码原bank，不等于重复expert共享解码，原容量门保持。[结果](../../../../.scratch/inference-foundation-review-2026-10-18/iree-grouped-inline-product/summary.json) |
| Apple/Vulkan padded首次prefill消费 | 两端独立padded codec仅-1返回BF16正零且不读取bank；ready页Map产生准确physical IDs及0/-Inf mask，再由原owner/页权限保活。原strict reader不放宽 | Apple8/8、Vulkan7/7通过，两格式、odd origin、两phase/slot、非法ID、空bank及实际持全部page writer的空history控制；选中输出和mask接普通native Graph Sink→Neg有完整oracle。未接native五port分页评分、page Graph/pending授权；零ReadLease footprint不等于零arena分配。[结果](../../../../.scratch/inference-foundation-review-2026-10-18/native-row-cache-padded/summary.json) |
| 资源投影依赖具体op/Provider判断 | 现由exact selected prepared的object-local能力描述真实physical input、空footprint及可选cold configure；typed模块证明逻辑访问链，core只核计划/实例/端口/alias/权限/退休，无新增替代白名单 | CUDA4/4、CPU6/6通过；同provider的checked/padded实现可声明能力，unchecked sibling仍拒绝。原pending、失效及旧结果门保持，无热submit新hook或数学变化；仅迁移ordinary singleton子域，不代签全部region或其他后端。[结果](../../../../.scratch/inference-architecture-refactor-2026-10-18/prepared-access/summary.json) |
| Apple/Vulkan承载分析重复名单 | Direct/Graph共用真实handler与native validator，删除重复custom/GGUF/multi-output前置名单；仅未解析custom可走显式cold expert resolver，不把它留到回放 | Vulkan9/9、Apple11/11通过，含原融合、callback冷调用次数、native/MLX状态组合、padded及旧输出owner。保持真实执行岛和结果表示，不新增kernel/SDK；此步未完成共同generated-output合同或所有公共入口迁移。[结果](../../../../.scratch/compute-execution-refactor-2026-10-18/carrier-shared-summary.json) |
| CPU＋Apple/Vulkan冷图生成与Plan重复 | 两端冷编译和Plan提取共用现有数据/控制依赖、常量、资源及物理图逻辑，平台adapter保真实native上传、prepare和输出规则 | CPU＋Vulkan原contracts及CPU＋Apple原contracts均通过，覆盖拒绝、OOM、旧输出、取消、abandon和accepted prefix；Apple保原MLX owner与contiguous输出要求。尚未将全部执行槽/进度合为共同实现，GPU-only旧joint入口仍要求CPU端点，不能称全量架构完成。[Vulkan日志](../../../../.scratch/architecture-refactor-2026-10-18/native-plan/06-contracts-run.stdout.log)、[Apple结果](../../../../.scratch/compute-execution-refactor-2026-10-18/joint-apple/summary.json) |
| 无覆盖节点触发路由前缀组合爆炸 | 按实际leaf/region全部成员建立incidence，增量维护兼容候选support，select/undo及单调reject剪去不可能覆盖的前缀；不扩大预算、不拆模型或增加名单 | 假provider预算/拒绝/重选/coverage等15/15通过。早期Qwen T23/1932节点图从耗尽65,536状态变为直接定位node300四port SDPA无承载；后续消费者以真实typed RaggedAttention执行完整三phase，见下文。当时普通F32 additive-mask SDPA候选洞未补齐，后续修复见下文。[回归](../../../../.scratch/architecture-refactor-2026-10-18/route-search/tests/summary.json)、[早期模型诊断](../../../../.scratch/architecture-refactor-2026-10-18/route-search/08-model-diagnostic.stderr.log) |

CUDA MLA首包是早期partial16阶段，工作区与小用例结果仅描述当时源码；最终F32 partial、
512/576四组合及跨后端数学结果以上表最终包为准；active workspace另见后续包。
FlashInfer lock/source/NOTICE已与最终数学及ragged补丁同步。
共享内存fragment的F32存储映射另做64×512坐标双射核对，非GPU数值证据；真正精度证据是敏感split对照。

后续native能力已在原Graph槽内冷准备：默认CUDA的16个组合通过Direct及两个共享metadata的attention节点完整oracle，
第二节点metadata staging为0，旧结果仍阻止重用；这一步不含row-cache/pending迁移或性能结论。[包与日志](../../../../.scratch/compute-execution-refactor-2026-10-18/graph-batch/source-results.zip)
原exact Context能力与实际workspace query/grow-only路径另通过CPU九项控制、canonical CUDA gated MLA及两个mock测试；
没有在submit新增scratch分配。[结果](../../../../.scratch/compute-execution-refactor-2026-10-18/batch-retained-workspace/summary.json)、[包](../../../../.scratch/compute-execution-refactor-2026-10-18/batch-retained-workspace/source-results.zip)
中性core adapter初始17项通过，修正同provider多能力共享原proposal上限后仅重跑三项、3/3通过；
保留原Context/实例及compile-only adapter寿命，不代表旧包装已经删除。[原始日志](../../../../.scratch/architecture-refactor-2026-10-18/region-adapter/tests/02-focused-tests.stdout.log)、[上限修正日志](../../../../.scratch/architecture-refactor-2026-10-18/region-adapter/tests/04-capability-tests.stdout.log)
页投影另已改为exact prepared artifact/物理端口证明，移除该处ProviderID判断；原CPU ready/pending、
CUDA低维pending及错误artifact/port拒绝通过，旧coordinator尚未删除。[迁移证据](../../../../.scratch/compute-execution-refactor-2026-10-18/projected-readers/source-results.zip)
实际workspace暴露并修复MLA零字节分配：no-split不构造float allocator，空指针不做偏移；原16B上限与
报告float=0断言保持，F16/BF16 Graph页消费通过；BF16真split的4,202,496B路径也通过。
只重编一个native CU。[零空间日志](../../../../.scratch/compute-execution-refactor-2026-10-18/mla-zero-workspace/02-graph-no-split-test.stdout.log)、[split日志](../../../../.scratch/compute-execution-refactor-2026-10-18/mla-zero-workspace/05-split-test.stdout.log)

GatherElements另有[公共schema及模型接线包](gather-elements-contract.zip)：两项schema测试通过；
Linux g++14的公共头、schema测试TU和ATen provider TU仅做syntax检查，三项通过，未声称Linux运行。
[低精度边界包](gather-elements-finite-edge.zip)中Apple、Vulkan各1/1通过，F16/BF16
子正规数、±0、最小正规数及±Inf重排后raw bits保持，不扩大为NaN payload承诺。

CUDA单cache的[Graph pending消费](mla-pending-graph.zip)另补F16/BF16实测：
同页先写负旧值，gate持有的producer再复制正新值；submit在gate释放前返回，释放后两dtype全32768输出
的FP64误差均为0。错layout/port拒绝、真实LeasedGraphOutput副本阻复用、独立caller-output lease跨reset
保持原值、页writer回收和producer guards归零均通过。此项未发现新的产品缺陷；只补消费者与条件测试目标，
证明组件Context的command replay，不扩大为全roster或新capture证明。

BF16专家cold-pack的空间成本明确计入资源报告：F32 packed payload是原BF16的2倍，原canonical常量仍保留，
不能在adapter内私自释放可能被其他consumer使用的源。旧实现每个invocation有一份F32 cast scratch，新实现由多个
invocation共享一份；单invocation没有降低总权重payload，多个invocation才减少重复驻留。此改动首先消除重复转换。

新增prepared常量使私有PlanState布局变化，直接消费者共6个TU：Graph内的executable、constant_artifact、
leased_invocation、lease_submission、region_access，以及CPU＋CUDA的invocation。上述普通两端测试fresh
了前5个；[混合消费者补证](hybrid-derived-control.zip)又fresh了第6个，并匹配当前前5个对象。
实际CPU TopK→control→CUDA prepared BF16 Cast→D2H→CPU TopK，以及同region的变更输入Neg，
两slot各两phase完整精确结果和report的传递依赖均通过，原混合合同也通过。旧调度core亦通过此跨存储例，
因此此项是新增调用链覆盖，不是原丢边的隔离复现或新增修复计数。后续仍不能混用旧私有ABI对象。

## 2026-10-19：移除重复混合运行时

普通Graph现已承接原CPU＋Apple/Vulkan的contracts、pipeline、effects、权重、ready/pending页与MLA消费者。
旧公开入口、两份Plan/Invocation/driver及仅服务旧路径的helper共20文件已按`test.ps1`归档删除，
不保兼容别名；仍被新测试使用的packed/sampling源码保留。原始220,293B源码见
[旧源码包](retired-native-runtimes.zip)；迁移源码、命令、原失败日志、当前对象/头身份及map成员表见
[本批结果包](native-runtime-retirement.zip)。包内未放模型、SDK、二进制或对象缓存。

| 实际检查 | 结果与边界 |
|---|---|
| contracts、publication、effects与pipeline | Windows/Vulkan和Air/Apple均有实际运行；保原数学、权重、旧结果、单worker、失败与关闭控制。pipeline选原4M元素/65步及resident双native用例，未跑全部性能矩阵。Air effects另核五个相关测试；初次全fixture暴露的四处旧假设已修，不宣称全194项重跑 |
| ready页风险 | Windows run090：compile85/create137/prepare11/atomic-bind1个OOM拒绝与六条生命周期路径通过。Air run265：79/95/10/1及适用五条生命周期通过；Apple不伪造Vulkan fence-unknown模式 |
| pending页风险 | Windows run084原11模式通过；Air run257原9模式通过。Windows run093与Air run259保两个真实原生producer、四个原F2控制与两个页节点，成功/第二来源取消/abandon均通过 |
| 原实际消费 | Air run249保D64 causal=1的F16/BF16各64步、双节点与旧结果，不扩称全部几何。run245保D512/H64和D576/H96两dtype的Direct/Graph完整FP64 oracle，并验证同cache/独立cache双MLA节点的layout拒绝与恢复；run247四项descriptor测试通过 |
| 冷错误分类与配置 | 29源码的33处真实host metadata分配失败不再吞成可换路由的拒绝；几何、配额及native资源策略仍按原能力。Windows九个CPU owner与20个CUDA host owner严格编译，Air六个CPU owner严格编译；CUDA未新链接/跑模型。原13项route配置和3项宏检查通过，无SDK构建 |

实际消费者修复了ready同布局帧原子替换、pending控制准备半提交、drain遗漏observer回调尾部，以及
已知关闭的native pending接收端在接纳前消费Update。关闭只停止接纳；取消显式请求；Scheduler空闲不证明GPU退休。
unknown注入按实际completion fence定位，保留原错误、不发布与guard退休检查，不恢复旧两次submit假设。

本轮Windows 36个image/map/导入库文件共877,814,161B、Air 26个image/map共236,685,112B已按冻结清单删除；
保留29个本批Windows对象、119个Air对象及原SDK缓存，CUDA host对象也保留。结果包3,406,970B，
SHA256 `166d5e8124588cda1c5a81a6d35b8aa904af23101bc2b5246ded702010c820b9`。
本批无吞吐结论、无Linux执行、未重跑完整Qwen；下列数值接受、原生成本及其余功能缺口继续保留。

可选文件流送已拆为显式 `compute_stream` 产品及四个薄桥接；默认Compute/Graph不再引入IO，
CPU/CUDA服务只要求实际copy backend。新CPU共享服务离线构建、安装及独立消费者通过，
未编数学SDK；文件与Mapping各512B往返复用同一512B pool，staging128B。
真实消费修复两处寿命问题：写回内部alias须先释放再判断读输出回收，采用固定两次线性扫描；
GPU copy退休后，取回mutable源须非阻塞观察实际接纳前缀使用的原producer，结算其host guard。
Windows原CPU14/14、Vulkan5/5、CUDA7/7及Air Apple6/6通过，原预算与错误/取消/abandon要求保留。
CPU共享交付与Vulkan/Apple file+mapping样例均通过；Vulkan另外只做共享导出owner编译，未重建Vulkan DLL包。
[源码与原始结果](optional-file-stream.zip)包含首次失败、修复、命令和实际链接身份；无Linux执行或性能结论。
按根目录`test.ps1`精确清理Windows319项733,039,297B、Air4项37,059,629B，
保留112个Windows对象、124个Air对象及完整CPU服务构建缓存；[归档与清理结果](optional-file-stream-results.json)。

CPU默认装配随后收敛为[一份原Provider roster](cpu-context-assembly.zip)，同一ggml实例承载ordinary与region。
定向6/6通过，实际核普通执行、融合和禁用region后的fallback均选同一instance，保原300/100/300偏好及数值；
原策略、转置、冷OOM及租约drain控制也通过。私有跨DLL入口有实际共享模式导出检查；未改数学、公共ABI或Plan/Invocation布局，
未重建SDK/内核、未整模型重跑或声称性能收益。[归档与精确清理结果](cpu-context-assembly-results.json)。

[冷规划与完整异步消费](cold-planning-progress.zip)继续复用同一Plan与Invocation：placement静态约束和region coverage
每compile初始化一次，实际资源权限及制品回滚逐attempt保留，成功后在report前释放临时事实。
原4项候选重试/硬预算/冷OOM控制通过；Air仅鲜编4个host TU，原CPU→Apple→CPU的F16/BF16双槽64轮通过。
Qwen使用现有`prepare_progress`接线，当前direct和scheduler两种模式各执行完整24层T23→4→1，
150个结果/状态彼此及与上一冻结direct均逐字节一致；next仍为20→248046→198。
显式逐槽drain后才结束runtimes和释放borrowed模型数据，保留共享状态顺序evaluate合同。
单次trace和提交返回时间不构成吞吐结论，也不推导同一Model的双槽可并发覆盖输入/状态。
首次模型启动缺CUDA Torch DLL和Air helper闭包误带IO的失败均保留；只修验证driver，没有以产品新依赖规避。
[归档与精确清理结果](cold-planning-progress-results.json)保留对象、完整SDK与本地raw trace。

[显式服务与普通mask接线](masked-attention-progress.zip)继续由真实消费者约束职责：`graph_progress`改为显式服务产品，
去除graph+sched交集的隐式装配；安装元数据静态/共享隔离夹具通过，shared/offline仅增量13个TU、2个DLL，
独立安装消费者实际导入该服务及公开Scheduler/BlockingPool成功。该小SDK只交付服务与bare Graph依赖，非完整数学SDK。
默认CUDA Direct/Graph的四口普通SDPA现接dense F32 additive mask，保原广播和tile真实stride，不收窄mask、不扫描内容。
新4项与原无mask F16/BF16 prefill/decode回归共5/5，含BF16 Q23/K32/H8/KV2/D256、两槽换址重放、
全score轴广播、F32 bias低位与NaN语义、257行的128/128/1分块及负scale/causal。
实际旧CUDA DIAG_MASK_INF由SASS确认减FLT_MAX，未来`+Inf`在同一原断言下导致NaN而非FP64预期；
仅原CU改为排除值直接写负无穷后，该用例及其余用例通过。复用原SDK，仅该CU与相关host增量，未改Graph核心/默认策略。
原F32宽化、score/softmax临时与managed阻塞事实保留，不称融合、零私有分配或吞吐优化。
同包保留当轮Apple F32 partial的只读最小设计；其后实现与验证见下文。
[归档与精确清理结果](masked-attention-progress-results.json)记录本轮映像回收；编译对象、缓存与薄安装元数据保留。

[Apple两阶段归约与重放](apple-two-pass-partials.zip)沿同一原生程序补齐默认精度：原MLX normal/GQA两pass均以
F32存取partial，最后才转回输出dtype；Q/K/V仍保原存储，额外载荷仅为`2*B*Hq*Q*blocks*Dv`字节，
未测总显存峰值或吞吐。另修GQA多batch KV基址遗漏，不引入新shader variant、Graph状态或公开开关。
删除Apple私有精度预测结构/架构后缀判断及Graph对应拒绝分支，共用原页/布局/边界validator。
Air `Mac16,12` 新4项通过：D64、H8/16、KVheads2、Q1/8、KV4096/8192的F16可表示结果溢出控制和BF16抵消残差，
普通Graph默认精度、两槽及旧结果保持，普通/ragged、短KV one-pass和Q16 full邻接；B2非零Q/K与逐行不同V的FP64
解析softmax同时区分两KV基址。原host+原metallib的两个数学控制均明确失败，原容差保留。
Graph首轮数学通过后，严格重放另检出page seal保留内部ResultStorageOwner导致错误拒绝；现同普通native region先释放
该已完成owner，再过原reuse_gate，最终同测试全过。没有放宽公共旧结果/source reader保护。
只增量编原MLX host和一个attention AIR、复用其余7 AIR重链，再编两个Hyle owner和测试；SDK、其余数学和core缓存保持。
ragged测试首次传lazy reshape而被ready-input合同拒绝的夹具错误、重放产品失败及各阶段源码/命令均保留。
[归档与精确清理结果](apple-two-pass-partials-results.json)保留匹配的新host/AIR/metallib与复用对象；无新整模型、其他设备或其他系统结论。

[物理别名与分页重放](graph-alias-lifecycle.zip)由真实计算和独立反例收敛共同内存合同。
旧可选原地写只看单个SSA最后使用，忽略View共享读者和发布后仍可读的私有View；
同输入精确数值控制分别出现`{3,5}`对`{1,1}`及`{2,3}`对`{-2,-3}`。现先建必须别名，
按存储组汇总寿命，再选择破坏性别名；union保留输出组未来读者。第二个独立控制证明跨lane早ordinal读者
并不happens-before写，第一版组修复仍选错alias；最终复用原有bounded segment clocks检查真实读者偏序。
多lane冷读者链在成功union后丢已消费历史，原同lane原地链仍只需一份16B载荷；workspace request根
改为复用已就绪父根，消除逐后代回溯。无新公共开关、Plan/Invocation字段或热分析；冷scratch有实际费用，未测编译时延或吞吐。
最终4项别名控制及原重试/预算/OOM4项共8/8通过。Vulkan页seal另检出与Apple同类的内部完成owner遗漏：
原strict submit自行reset隐藏了显式prepare_replay拒绝，修前真实失败保留；只在原seal前reset内部结果。
新公共core与两个fresh页owner/test合链，CPU/Vulkan Q18/Q2/Q0、两槽、旧GraphOutput拒绝和释放后显式重准备均通过，
原FP64预算0.002保留。只编相关host TU，未重建SDK/数学/shader；未新增Apple/Linux或完整模型验收。
[归档与清理结果](graph-alias-lifecycle-results.json)保留命令、失败控制、选中对象与源码身份，复用对象及缓存保留。

[BF16与真实输入旅程](bf16-source-journey.zip)同时保留下面的Vulkan接线、原槽InputSources临时索引空间复用和完整Qwen重放。
固定2个source、轮换普通输入/输出、两轮预热后的4轮prepare窗口，原每轮3次metadata分配（共12次）降为0；
仍逐次重核实际route、保原pending/失败与旧结果门，4/4通过；容量留至原槽销毁，不声称整次运行零分配。
新别名/输入核心只重链实际24层Qwen，prefill23→续算4→decode1及150个trace与上一冻结direct逐字节相同；
未重跑scheduler，也未接受此前独立oracle约2%误差或新吞吐。[归档与清理结果](bf16-source-journey-results.json)保留原trace，回收重复payload。

## 当前线索

以下未完范围由新架构继续承接，区分缺实现/接线、实际域限制及缺整链证据；已有局部成功不替代其余模型、制品或路线。
默认CUDA普通四port SDPA的dense F32 additive mask已按上文补齐；这项普通接线有独立实测，Qwen完整模型仍使用typed RaggedAttention。
GPU三个IQ1 narrow GET_ROWS等仍按真实实现边界保留。
2026-10-19早先只读复核时，普通dense BF16 SDPA已有F32宽化→ggml→IREE舍入承载，
typed页路线仍在cache domain、private PagedAttentionRead和native FullF32 host准入处拒绝BF16；
当时未修改或运行数学实现，既有F16页/紧凑MLA记录不代签BF16。
后续本轮沿同pin现有BF16/Q8 shader读取原KV页，private结果及request拼接保持F32，最终一次原IREE RNE Cast。
RTX4070新增两项通过：D256 BF16 plain分页/ragged及Q8_0 KV，D256与576→512单CKV的Direct/普通Graph
双槽、固定绑定重放及旧GraphOutput gate；两种Graph几何的独立FP64最大绝对误差为6.10e-5和1.22e-4。
原F16低维/高维与Q8三个控制通过。仅编native host和两个Hyle产品TU及两个test TU，原12份patched
header/shader逐SHA匹配且未重建SPIR-V。首轮fixture默认metadata cap0拒绝保留，改为既有4096B准备合同后通过。
当轮仍保plain cache与query同dtype、output与query同dtype；未据此验收pending精度或完整模型。
随后独立dtype片复用相同FullF32 native host/shader，只改两个包装层：成对F16/BF16 KV与F16/BF16 Q/output
解耦，private request结果统一F32并在拼接后一次转换；空结果也保持所选output。
RTX4070六项定向测试通过，含新D256 Direct/普通Graph三种组合：BF16 Q/F16 KV→BF16，
F16 Q/BF16 KV→F16及→BF16，独立FP64最大绝对误差分别6.1026e-5、7.628e-6、6.1025e-5。
双槽、旧结果门、固定绑定重放与空shape/dtype保留；原短F16、BF16/Q8及BF16单CKV正控同次通过。
公共BatchCapacity原F16/BF16轴范围、成对K/V格式、原深度/page16/设备限制不变；紧凑MLA schema仍只有一个data_type。
未新增F32 query、shader、executor或整KV影子，也未新增pending精度、完整模型或性能验收。
[布局与dtype轴归档](strided-layout-axes.zip)及[结果](strided-layout-axes-results.json)保留这6项定向测试。
同批按真实Q/K/V数据流补通用strided reshape：连续stride chunk可拆分或合并，跨gap flatten仍须Copy；
5新例在原对象上全失败，新实现连同4项既有控制9/9通过。完整Qwen三phase的150份trace与原冻结结果逐字节相同，
Copy节点85→25、报告步骤1252→1204，保留18份history、6份gate flatten及最终hidden的显式Copy。
Graph workspace每slot在T23/T4/T1分别增加46,592/16,064/3,968B；native仍可能有必要CONT，不能据Copy计数宣称空间或吞吐收益。
原始trace与复用objects保留；同字节的本轮重复trace和归档images按根目录test.ps1精确清理。
其余有成熟组合的需求先检查真实默认路由和连续消费，不以片段尚未实跑为由新造API，也不把设备硬限制当作可删白名单。

[CUDA队列原生接线](cuda-admitted-stream.zip)及[结果](cuda-admitted-stream-results.json)记录后续实际施工：
ggml在原Invocation借用已接纳stream，恢复原cuBLAS stream/workspace，移除逐片段双Record/双Wait与无条件私有root创建。
原EventSlot仅在未确认辅助流退休的失败路径附一次probe，根fence完成或观察错误均不提前释放原owner；原reaper接续abandon。
最终10/10定向检查通过，含两slot交替队列、实际native warm capture、受控pending下放弃两个公开owner并close、
外层capture拒绝后复用、原默认Direct/Graph mask及native退休观察错误控制。未注入真实辅助流drain/cuBLAS恢复故障。
最终完整Qwen三phase的150份trace仍与原冻结结果逐字节相同，图节点/route/workspace不变。
同权重、输入、精度、capacity32、direct、关闭trace，旧新各3次按ABBAAB执行；prefill/continuation/decode1/decode2
中位执行时间分别81.75→71.26、35.17→26.26、33.59→20.27、32.31→25.10ms，四组观察区间均分离。
计时含Completion及next-token读取，排除模型加载、冷图编译和layout准备；正常EOS结束于第二decode，不能推导持续warm吞吐。
独立数值接受、llama.cpp及其它平台的新旅程仍未完成；源码、失败fixture、命令、原始测量及缓存身份保留，精确清理本轮images和重复trace。

Qwen3.5-2B官方BF16完整24层CUDA消费者已执行prefill Q23/KV23→continuation Q4/KV27→decode Q1/KV28，
next token 20→248046→198与独立reference一致。当前1204 step含518个metadata View，不能与旧998步直接推断性能。
[数值定位](qwen-numerical-localization.zip)通过原官方CUDA单block与同输入算子重放，确认首个明显传播来自
post-attention RMSNorm→BF16的token21/channel619：Torch预cast恰在BF16中点，Hyle向上，与该点F64→F32控制一致。
只替该坐标即可逐bit复现全部47,104个Hyle layer0输出和18,432个layer1 conv元素；这只是诊断干预，不是候选运行喂oracle。
多处投影、残差与MLP逐位一致，原norm kernel未改。
后续[实际F32定位](qwen-rmsnorm-f32.zip)以原输入和权重逐bit复现47,104个BF16输出，
同点native预cast值为`0x3f568001`，Torch为中点`0x3f568000`；所有native F32→BF16符合RNE，差异已在Cast前存在。
ncu实际确认`rms_norm_f32<1024,true,false>`、grid23/block1024/shared128B，profile前后输出相同；
只复用原CUDA内核并新编5个host owner，不是换用其他norm近似。
后续[原归约诊断](qwen-norm-reduction.zip)只新编原norm单CU的控制/插桩各一次，两版全部47,104个F32和BF16
均与冻结Hyle逐bit相同，控制版与原目标kernel的SASS相同。实测mean`3b4ae0ec`比已保存Torch variance`3b4ae0ee`低2ULP；
原1024线程两级XOR拓扑的离线重演复现实测sum。后续F32两乘从native mean产生`3f568001`，
从已存Torch variance则产生中点`3f568000`，解释该点的归约与舍入传播。
观测仍有明确界限：写出mean使原FFMA变FMUL+FADD，本2048列精确缩放与全输出bit门成立，不能外推任意输入指令无扰动；
Torch scale/mul1未实际采样，后段是离线推演。没有修改产品数学或容差，不能代签完整模型误差全由此产生。
当前三phase logits相对L2为2.151%／1.957%／2.459%，cosine均大于0.99974，top1／top5一致；后者不构成精度接受标准。
该因果实验证明首层该点传播，不证明后续全部偏差同源。2,728,350B归档含脚本、小结果、命令与身份，无模型／权重／SDK／完整trace；
数值接受和公平吞吐对照仍未完成，不能称整模型数值PASS。

新的[首个full-attention同输入定位](qwen-first-attention.zip)只新编一个诊断消费者TU，执行一次完整24层prefill；
五个探针均为原选中步骤已有输出，默认/发布路由逐字节相同（1878节点、1204步骤），原50份结果共20,928,512B
仍逐字节相同。实际positions、frequencies、Q/K及cache活动V均保留，独立计算不向候选回填数据。
RoPE的Q/K最终BF16与F32及两种F64控制均零bit差，与官方eager BF16则分别差1604/389项；未旋转tail保持原bit。

| layer3 attention的同输入参考 | 参考精度边界 | 相对L2 | 参考投影BF16后的不同项/47104 |
|---|---|---|---|
| 官方eager | scores和归一化P均有BF16边界，结果BF16 | 0.470994% | 27666 |
| 独立整行数学 | scores、softmax、PV均FP64，结果未舍入 | 0.167083% | 6678 |
| 独立在线数学 | F32状态，每tile未归一化exp转BF16，最终结果BF16 | 0.005137% | 51 |

实际选中原生制品的源码按SM89资源规则推导KV tile16，故T23按16+7在线更新；**无split-K不代表只有一个KV tile**。
分母和PV都消费已舍入的BF16未归一化exp，旧分母/输出随max变化以F32缩放，最后使用原生approx reciprocal。
CPU控制保留这些边界但不模拟MMA、SFU、reciprocal及归约指令顺序；剩余51项未逐一分解，不将其直接判为功能bug。
本样本活动raw score为4.249…254.257，有限mask哨兵−50000不干扰有效key；不外推极端或全mask行。
探针保留使workspace每槽4,151,296→4,292,608B，本轮不是性能测量；没有产品数学、路由或容差修改，
也未据本层结果接受整个24层约2%的logits差异。[结构化结果](qwen-first-attention-results.json)包含命令、身份及明确限制。
归档443,574B，不含模型、SDK或重复完整trace；随后根目录test.ps1逐项核对并清理54个实验image/map和重复trace，
共170,197,711B，原50份trace、8份唯一探针和一个新编消费者object均保留。

后续[RoPE单因素诊断](qwen-rope-ablation.zip)先用同源权重、相同Torch和官方实现逐字节复现原150份三phase结果，
再重置独立cache、固定23／4／1输入，仅将参考的RoPE三角值与乘加中间保为F32，最终Q/K仍为BF16、未旋转tail保持原bit。
Hyle候选及产品数学未改，原eager attention和其他计算不变；每phase只比较最后位置的248,320项logits：

| phase | Hyle对原参考relL2 | Hyle对F32 RoPE参考relL2 |
|---|---|---|
| prefill | 2.151% | 1.940% |
| continuation | 1.957% | 1.490% |
| decode1 | 2.459% | 2.127% |

这支持RoPE中间舍入对本样本差异有部分因果贡献，不解释剩余误差，也不等于产品已修复。
三phase top1均一致，但不是所有指标都改善：continuation的参考到Hyle softmax KL从1.339×10⁻⁴增为2.912×10⁻⁴。
归档保留脚本、命令、环境/源身份、150份byte guard及两组最终logits/hidden小数据，无模型、权重、SDK或完整trace。
未改变精度预算，没有产品数学修改、吞吐测量或整模型精度接受。

新的[Apple双Runtime消费](apple-cross-runtime.zip)在Air Mac16,12/M4上通过1/1永久回归：两份真实Context、
MemoryResource和Source，一份普通Graph、两个原Invocation，A的Abs→B的Neg由冷编译选成两个native region和
A原复制录制的16B Copy。晚句柄留在producer FIFO解析，目的共享存储由B的实际能力准入；不要求它属于B的默认域。
HostWait Reject在冷prepare拒绝，Allow沿原frontier与可选progress服务执行。原A FIFO gate期间两Completion
均观察为Pending、输出不可见，释放后两槽各两轮不同输入与独立−abs公式逐bit一致；旧GraphOutput及导出ReadLease
跨另一槽重放和本槽reprepare仍保旧Array。B FIFO当时观察为空，本例未增加每次原生接纳trace或故障注入。
只新编当前core/input/ops和test四个C++ TU，实际map核对所选owner；SDK、原生数学与metallib全复用。
未新增产品执行机制，纠正了此前文档将跨Runtime复制桥笼统列为未实现的描述；真实copy/HostWait成本保留，未测吞吐。
[结构化结果](apple-cross-runtime-results.json)保留最初链接脚本重复operation owner及测试过度约束默认存储域的修正，
并明确压缩行分页的singleton限制、原生region能力和五口scorer仍待补齐。归档568,198B；根目录test.ps1随后逐项核对
并删除Air本轮image/map共18,487,004B，四个新object及原SDK/Metal资源保留。

新的[Vulkan行缓存原生region接线](row-cache-region-projection.zip)从真实Map→padded Decode→Cast/View→Sink计算
补齐普通Graph：共同core按实际选中prepared节点查询访问和冷配置能力，原生region以实际lowering与最终dispatch
证明受检bank读取，再映射回Graph边界端口。融合或分拆不改变页权限；空读须由当前slot实际配置证明，ready不授予pending。
原自有device metadata镜像经原lane Source的冷搬运上传，计入scratch预算并等成功退休后才publish；
相同已准备layout跳过重复配置与上传，变更layout仍有真实staging、控制分配和等待，不宣称零分配或零复制。
RTX4070最终6/6通过：新Vulkan三项、原padded Direct一项、CPU ready及pending两项。
实际融合/分拆、乱序页、同槽改变请求分组及页顺序、双槽、旧结果与导出lease、未选页writer、真空历史持有全部writer均已覆盖；
解码与mask逐bit核对，Sink沿用原独立FP64预算，额外bank读者和私有物理IDs发布拒绝。
原core对照4项中2项在首次typed prepare因singleton限制失败，原expert正控通过；
pending旧fixture补回显式prepare_replay，未放宽产品提交合同。仅新编15个相关C++ TU，原SDK/native数学/shader全复用。
没有Apple/Linux新运行、完整模型精度或吞吐验收；Apple projected row Graph、Vulkan projected row pending及
Apple/Vulkan五port分页scorer仍待补齐。[结构化结果](row-cache-region-projection-results.json)保留命令、源码与选中owner身份。
归档397,368B；根目录test.ps1核对路径、字节数及SHA后删除本轮8个image/map文件，共197,353,563B，
15个新object、原native host/shader及其余缓存保留。

后续[Vulkan pending输入接线](vulkan-row-pending.zip)将原生pending能力从Q/K/V分页程序中独立出来，
原Apple/Vulkan分页与Vulkan受检行读取共用实际bank端口、source工厂和准确frame验证/提交，旧专属pending字段删除。
共同payload仍唯一持有CopyOutcomeSource/DependentCompletion；原槽借frame/source，准确ready页与ordered writer
guard进入原Run退休。shared/native入口在take前互斥核对源协议，移除旧同lane命中授权；回收Update不把旧帧变成ready。
RTX4070最终9/9通过，原对象同测试9项中3项在pending bind明确拒绝。新FP8 D512覆盖融合/分拆及原FP64 Sink预算，
FP4 D128覆盖准确BF16行、mask与Cast/Neg；短copy覆盖不足拒绝保留Update，原来源及Graph实际观察Pending，
无结果/回收，退休后旧GraphOutput仍阻止回收，释放后取回原writer与copy invocation，已消费帧不可直接重放。
测试复用原fence观察门，暂将成功观察保留为NOT_READY；证明原Completion/权限寿命，不声称GPU物理仍阻塞。
迁移后的原native分页联合消费者在Windows与Air均通过活动Q2/18、双槽、两节点共一Update和原64步检查，
Air含F16/BF16。Windows/Air各新编8个host TU，所有SDK/native数学/shader复用；无新吞吐、峰值内存或整模型精度验收。
首次FP4测试误接D512-only Sink及一次测试report字段名编译错误均已修正，产品数学/容差未变。
该阶段Apple压缩行Graph及Apple/Vulkan五port分页scorer尚未完成；Apple ready后续进展见下段。
[结构化结果](vulkan-row-pending-results.json)保留当时限制和身份。
归档551,128B；根目录test.ps1逐项核对后删除Windows12项共296,062,878B、Air2项共17,881,851B，
两端各8个新object、原native host/metallib/shader及其他缓存保留。

后续[Apple ready行缓存接线](apple-row-projection.zip)沿同一普通Graph与原Invocation完成：实际lowering给出受检
bank事实和真实边界端口，Map/Decode保原融合或分拆。原共享buffer的地址导入不扩大页读取权限；同槽换layout
先过旧结果/在途gate，再释放旧输入读owner后更新metadata。plain checked decoder复用原padded选行kernel，
非法ID集合不读bank、不回退row0，保原OutOfRange和结果发布规则；codec数学及Sink容差未放宽。
MacBook Air Mac16,12/M4最终6/6通过：FP8 D512融合/分拆接原Sink、FP4 D128行/mask/Cast、两槽、同槽换请求/页序、
旧GraphOutput及独立只读Array、真空史全writer、plain仅页1/3且页0/2持writer、非法ID及额外reader/私有发布拒绝。
旧对象对照4项中3项在首次typed prepare明确失败。checked行bank冷warmup改用实际ordinal描述，避免按整个bank分配：
2,449,473,536B逻辑容量的融合prepare-only cold peak为3,900,296B，配置MLX limit为128MiB，未创建真实arena或Invocation。
同2.45GB旧控制因总字节索引门失败，不是OOM；仅改容量的1.22GB旧控制则实际分配到1,228,637,018B，超过配置limit，
说明该MLX设置未硬阻此次分配。两种容量不是配对性能测量，不据此报告吞吐或整模型峰值。
只增量编译7个产品host TU；实际map各选中一次，117份current源码及118份current/derived依赖无漂移。
SDK、原native host/metallib与padded Metal kernel源身份沿用，MLX原冷trace/JIT仍存在。
Direct独有的零selection和公开owner销毁夹具本轮未重跑；Apple行页pending、Apple/Vulkan五port分页scorer、
完整模型数值与吞吐仍未完成。[结构化结果](apple-row-projection-results.json)保留失败对照、身份与具体限制。
归档729,162B；根目录test.ps1核对路径、字节数及SHA后删除Air四个image/map共37,296,697B，
保留10个product/test object、原Metal资源及其他缓存。

随后[Apple行页pending与冷准备消重](apple-row-pending.zip)把原native页source代码提取给行缓存和分页attention共用，
来源仍由原CopyOutcomeSource唯一持有。每个实际槽保留自己的准确帧和预留页guard；原Run负责实际退休，已消费帧须reset。
Direct/Graph冷准备统一查询原数学实现的bank端口，删去warmup的decoder/indexer bool和组合reader的假operation注记；
冷描述不授予页权限，实际checked lowering与唯一reader证明仍独立。Air 17/17通过：新pending3项、原ready6项、
Direct codec4项、Direct paged1项、Direct/Graph/empty indexer3项。旧产品对相同新3项全部在pending绑定明确拒绝。
新用例核FP8融合/分拆与原FP64 Sink预算、FP4准确行/mask/Cast、短copy覆盖拒绝保Update、原FIFO实际Pending、
旧GraphOutput阻回收、原writer/copy恢复及已消费帧不能直接重放。原native分页联合消费者另通过F16/BF16各64步、
活动Q2/18、双槽、两page节点共享Update及原CPU前后片段。只新编8产品及4测试host TU，沿用原SDK/Metal资源。
144份源码快照、134份实际依赖无漂移；typed submit/take_update仅由新helper提供。
本阶段未注入接纳后来源取消，不能证明失败suffix未执行数学；该执行前检查由后续独立反例追查。
[结构化结果](apple-row-pending-results.json)保留这一限制。归档816,988B；根目录test.ps1精确清理Air四个image/map
共37,032,558B，保留12个新object及原缓存。

后续[Apple来源执行前检查](apple-source-preflight.zip)确认该失败边界：原FIFO只保证复制先执行，
复制取消/出错可以不poison Runtime；外层DependentCompletion虽挡发布，原decoder仍可成功执行。
两个原产品反例均明确失败，实际原生decoder返回成功Array；修复将全部真实CopyRecording的终态检查放在原Job
resolve_inputs/eval之前，来源失败直接finish该原Run。检查在原observations锁内读取原生状态，不poll/wait公开Completion；
借用原CopyOutcomeSource的稳定Update，原DependentCompletion保活，执行入口与enqueue拒绝均清借用字段，无新owner。
反例A在Graph接纳后只取消写页phase，原copy调用为0；反例B取消写页phase0，但独立phase1仍成功复制4B，
原copy调用为1，证明latest成功不能代签全部来源。两例预填有效但未publish的sentinel页，修复后原生decoder及外层均Cancelled、
无结果，Runtime仍健康，原失败来源控制可回收但页不可发布；未据此声称GPU物理load trace。
Air最终19/19及原native分页F16/BF16各64步通过，包含此前17项正控；Direct receiver已接同一原Job检查，未新增Direct晚失败注入。
只鲜编6产品及1测试host TU；122份当前源码/实际依赖无漂移，当前链接仅选中新leased对象，未抽旧SDK同member。
原leased控制来自已记录的缓存SDK member，不把当前Windows源码当作它当时的编译源码；SDK/native数学/metallib仍沿用。
[结构化结果](apple-source-preflight-results.json)保留具体命令、身份和限制。归档700,014B；根目录test.ps1逐项核对后
清理Air六个image/map共56,175,643B，保留本轮7个及上一轮12个object、原Metal资源。

随后[五输入分页评分](paged-row-scorer.zip)补通Apple/Vulkan普通Graph的FP4 bank1 scorer，
接原TopK→FP8 bank0 padded gather→D512 Sink→Neg，沿用原BF16数学边界与FP64 Sink预算。
两端同源新6项均通过，覆盖融合/强制拆分、双槽与同槽换布局、旧结果、空历史持全部writer、
C258跨256行tile、Rmax>T及空请求、非法末range、私有bank/metadata拒绝和真实pending页复制/回收。
Windows最终另有原四口3项与页读取9项共18/18；Air原19项先通过，最后两次局部修正后仅重跑新6项。
这条真实链发现部分placement下两端最大region共享未约束TopK而无法精确覆盖：native-only来源现用原固定writer
在最大候选后补单节点候选，仍由原lowerer实审并计原预算；无约束路径不增加候选或热分派。
Apple按request共享256列解码tile及原BF16 gather_mm；Vulkan复用原评分shader，仅新增分页寻址变体。
保留完整O(T×C) scores与分块临时载荷，没有流式TopK、原生sparse Sink融合、整模型或新性能验收。
Metal模板作用域及测试重复创建layout owner的修正、失败日志和最终同源旧产品控制均保留在
[结构化结果](paged-row-scorer-results.json)及归档；准确版本检查没有放宽。SDK/原数学缓存继续复用。
归档1,467,243B；根目录`test.ps1`逐项核对后清理20个image/map/lib/exp共438,150,693B，
保留15个Windows缓存项及Air8个新object与原数学制品。[清理记录](paged-row-scorer-cleanup.json)

| 事项 | 当前事实 | 本轮工作 |
|---|---|---|
| 紧凑MLA各route覆盖 | CPU paged/ragged、CUDA paged/ragged与Apple/Vulkan paged已有上表定向实测；CUDA单cache pending追加已验证。共同Graph window描述与CPU实际窗口消费现已补通 | GPU窗口、Apple/Vulkan单cache ragged及其他未验证域仍按实际native资格，不因共同schema可表达就伪装支持 |
| F32及编码高维KV | 共同BatchSchema的plain Q/output/cache以F16/BF16为域。DeepSeek官方源码实际存BF16 QAT，并未规定packed cache wire布局；四后端逐行codec及contiguous packed indexer、CPU/CUDA typed分页评分和状态追加→选中KV→Sink均已有上表定向实测 | 继续核分页/稀疏sink联合承载与模型数值需求。CUDA/Apple contiguous及CUDA paged scorer仍逐完整history tile做GEMM，只跳无效bank读；CPU可跳空gap。旧CacheStorage固定scale与新row-inline布局独立，不能按字节数假定原生page-footer兼容 |
| 稀疏attention联合链 | 四端typed分页评分→TopK→padded跨页Gather→Sink的ready首次prefill均已有上述实测；Apple/Vulkan另有原生pending评分整链。CPU/CUDA Graph pending行读取另有证据；Apple接纳后来源失败检查另有两个反例控制。通用paged batch仍没有逐query显式mask/sink端口 | 继续核完整模型与连续prefix旅程、流式TopK和原生稀疏Sink融合；当前仍有完整scores，保真实页权限，禁止whole-arena假lease |
| 首次prefill的稀疏padding消费 | 四端padded codec及ready选中行→Sink消费均有上述定向证据；CPU/CUDA另有typed评分联合链。旧prefix消费者从visible512/base2048开始，不能独自证明空历史全过程 | 仍需完整prefix从空状态连续推进、原始QAT→重编码等价性及其余后端整链；现例从post-QAT/encoded bank出发，不是完整模型。原官方参考同[评分包](packed-indexer-cpu.zip)保留 |
| 常量优化的路由覆盖 | CPU/CUDA上述packing适用于已选ggml region，CUDA另接不可变BF16专家单节点；普通单节点dense可能选oneDNN/cuBLAS。GLM mHC中间常量冷准备现四后端均有上表定向证据；ATen/IREE dense共享权重块重复解码已修，CUDA SGLang/DeepSeek沿原encoded weight pointer | 沿实际默认路由继续核热成本。保F32 activation精度、公开中间及其他消费者，不泛称所有常量表达式均已折叠 |
| MoE重复专家的原生成本 | CPU重复专家块共享解码与Vulkan IREE逐slot大权重临时量均已有上表实际修复/证据；CUDA原encoded pointer、Apple indexed读取及IREE inline reduction的成本不能直接类比 | IREE仍按slot取码/解码，并未共享重复expert结果。现固定点收益不代签所有T/S或decode；继续沿模型实际route核必要工作，不从逻辑tensor或不同算法的局部计数直接认定退化 |
| Apple区域同步 | 原生Metal recurrent桥需要就绪输入，MLX array wait还承担事件/错误状态退休，attention分块等待约束峰值内存；recurrent两个无条件容器分配已按上表移除 | 保留真实等待与逃逸输出寿命，不能用局部容器计数推导整个run零分配或吞吐收益；其余热成本继续审查 |
| 热调用资源成本 | 四端普通/混合消费者已迁移共同generated-output与执行槽，typed layout接入原Invocation，重复混合runtime已删除；可选文件服务独立 | 继续核默认Context装配、共同冷编译职责及每步分配/转换/上传/同步。Qwen整模型仍同步等待、直接驻留权重且非分页，不能代签progress/流式权重/分页整链，也不能以结构消重代替性能结果 |
| 混合路线的不可变权重准备 | CPU＋CUDA重复上传依赖、CPU＋Apple/Vulkan未变WeightBatch的alias metadata重复分配已修 | 后续仍核跨runtime顺序及寿命；原delegated host wait保留。reset后旧内部binding可能保留alias至后续bind/prepare/submit或析构，不声明reset立即释放全部旧WeightSlots |
| Apple冷准备与真实bank资源 | 官方codes/scale与typed row decoder的冷占位已窄修；实际checked行bank现按冷事实ordinal接入融合Graph，2.45GB逻辑描述prepare-only peak约3.9MB；普通U8/opaque Graph不因此获得相同优化 | 未包含真实bank、计算临时量或整模型峰值。旧1.22GB控制在128MiB配置limit下仍实际分配超过limit，不能称硬cap。继续按实际绑定、activation广播、checked状态及输出寿命核资源成本 |
| GPU路由ID接专家/embedding | CUDA ggml整ID回读及普通CCCL Checked Gather的16B bounds回读/stream同步已修；Apple同E TopK索引已可静态免扫，非Q4K Metal的外来/Direct IDs等仍CPU扫共享buffer，不能把它说成实测D2H copy | 区分整数组回读、共享memory CPU walk、终态少量观察及已有MLX/native ready等待。少量ID改为新增GPU归约/等待可能更贵，不据O(ID)本身断言性能输项，不删除实际边界检查 |
| 完整expert bank容量 | CUDA/Apple官方FP8、CPU/CUDA GGUF的上述过度门已修；本Qwen IQ4_XS bank的物理4,563,402,752字节超Vulkan单descriptor的uint32范围 | 区分真实设备限制和carrier设计；不能以resident11代签完整模型常驻；更大packed gate/up及其他bank按实际单buffer限制判断 |
| Vulkan官方bank寻址与副本 | 原IREE grouped signed byte寻址门已改word并更新实际VMFB/host/Graph接线，GLM完整3.22GB FP8有上表实测；直接owned常量共享已修，动态及lease/WeightBatch等view仍按slot分配和复制 | 原真实descriptor、word数和workspace限制保留，不能外推超过设备单buffer的bank。精确离线compiler已恢复到项目内工具缓存，[恢复记录](iree-compiler-recovery.zip)。其余输入的借用需明确owner/资源证明，不猜opaque根 |

缺整链证据的继承范围仍是8模型/22制品的实际embedding/LM-head、dense/batched、MoE、norm/RoPE/GDN/KDA/conv调用，
以及prefill→续算→decode→接受前缀后继续的状态反馈。普通默认Direct/Graph、CPU＋GPU旅程、动态长度/请求增删与布局变化、
两slot重绑、跨runtime权重/状态驻留和失败/取消/关闭退休都按各自现有证据补查；不要求无意义的format×op笛卡尔积，
也不以Qwen一个消费者代替其他模型、系统或后端的有关检查。完整bank只核过的容量与物理限制继续逐项保留。

DeepSeek的压缩/索引/接受前缀已有一条新的[公开CPU消费](deepseek-sparse-prefix.zip)：
H64/D512、index64×128、Top512/516、ratio4，8 token跨两个完成点，出现3种TopK集合，
每query的未来slot均不可见。接受k=0..8后再继续1 token，11项资源与cursor、IDs、Sink输出
与从同base只逐token执行接受前缀的独立分支完全相同；继续提交StateStore后的资源与metadata
逐字节相同，base未变。给定实际选中前驱的独立FP64 Sink最大误差6.1035e-5，沿用原BF16链预算。
另补[ratio128分支](deepseek-sparse-prefix-ratio128.zip)：此分支没有Indexer/TopK，采用64行半窗口，
从absolute position2302、当前半窗口62行、17个已发布压缩项开始，8 token在第2个token完成128边界，
可见项17→18，最终半窗口6行。APE、窗口滚动、raw更新和仅在完成点发布cache均有独立核对；
接受k=0..8后继续1 token，6项状态与metadata逐字节一致，配对Sink maxAbs=0，独立FP64 Sink
最大误差1.2207e-4，沿用原BF16链预算。参数化后的同一程序也重跑原ratio4分支，通过原几何及误差要求。
两分支均未发现新的产品缺陷。这证明投影后F32物理状态＋原显式低精度数值边界的密集组合可用；
未证明分页权限桥、packed物理历史、其他后端整链或原生融合，不能据此删掉上表稀疏attention线索。

新的[逐行压缩状态消费](row-cache-state.zip)在CPU与CUDA使用同一源码：base可见3行，
接受0时取消草稿；接受1或2时只编码新行，对固定5行chunk做COW，并用原F2精确追加新字节，
取回原整writer后才publish/complete，与可见长度4/5共同发布。主KV chunk为2,920B、index chunk为340B，
没有按整个历史长度做COW；旧base的bytes与metadata、未改prefix/suffix均保持，失败producer不发布。
主KV再经公开Graph的选中BF16解码→CastF32→View→单Sink（B2/H64/Q1/S3）→Neg双publish，
CPU最大FP64误差1.581e-7、CUDA2.176e-7，沿用原2e-5绝对＋2e-5相对预算。
可见性由该消费者显式保证IDs小于metadata长度；codec只检查物理容量，不隐式读取状态metadata。
这不是TopK/indexer整链、原生分页直读或完整MTP引擎证明；不得把固定chunk例子当成可无限增长的单bank方案。

另补[GLM KDA公开接受前缀消费](kda-prefix.zip)：投影后B1/H64/D128、F32 state、K4 conv，
3-token验证产生K3快照，接受k=0..3后各继续1 token。默认CPU/CUDA context中Direct与逐算子Graph
各8分支通过；实际conv→SiLU→Q/K L2→per-channel KDA反馈设备结果，完整state/scores/history
与同base逐token独立分支符合原3e-6预算。状态、history、cursor共同发布，旧base不变，取消不发布。
本项未发现新产品缺陷，不计新增修复；没有完整投影/模型、整链融合、原生故障注入或Apple/Vulkan新旅程证据。

完整模型加载、tokenizer、请求/生成/接受政策由引擎负责；本轮审查其所需的可复用底座。
整模型执行与跨平台交付未运行的部分保持明确，不以局部用例代签。

归档只含源码、脚本、命令、结果与缓存身份，不复制SDK、object或可执行文件。
本批归档后经根目录test.ps1按路径、字节数和SHA精确清理Windows 23个、Mac 10个实验binary，
共552,637,943B；源码、日志、数据、maps、product/test objects与SDK均保留。[清理记录](probe-cleanup-row-followup.zip)
本轮另按同一test.ps1精确清理6个已归档stage的36个实验exe/lib/exp，共547,277,745B；
所有object、原SDK及恢复的精确IREE compiler仍保留。[本轮清理](probe-cleanup-pending-followup.zip)
随后按同一根目录test.ps1精确清理3个已归档stage的14个实验exe/lib/exp，共568,680,588B；
全部object、maps、源码、日志、数据和SDK保留。[清理记录](probe-cleanup-current.zip)
上述评分、padding与CPU专家复用阶段归档后，又通过根目录`test.ps1`逐项核对并删除38个
exe/exp/lib，共436,566,645B；40个复用object、全部源码/日志/数据和SDK保留，当时尚在施工的CUDA分页评分目录未清理。
[清理记录](probe-cleanup-review.zip)
pending行测试发现的旧CUDA fast-math缓存差异及按现有canonical flags替换单个unary对象另留
[缓存身份记录](cuda-precise-unary-cache.zip)，没有修改Neg产品数学或放宽逐bit检查。

首扫还区分了两类容易误报的情况：混合route的allocate_host/allocate_boundary有已分配即返回的保护，
函数出现在prepare中不代表每步重复分配；CUDA best_of的精确环境条件选择特定测量域的优先级decorator，
域外仍有普通provider，不同于本轮已修的FP8/CCCL底层能力误拒。

## 2026-10-19 冷常量 packing 的失败退役

[本轮源码、命令、结果与数据](cold-constant-retirement.zip)将 CUDA packing 从栈上
Program/Invocation＋Status/drain 迁到原 QueueSequence/Completion：StorageOwner 移入真实资源，
ready lease 保留源 allocation，首次观察前保留 partial payload、原 source、allocator 和预算。
native 临时程序/workspace 与 transient charge 在实际退役时释放，成功后才缓存。
Region 扩展现为 v28；raw borrow 缺少保活资格时保留原 runtime 运算。

Windows 最终8项CPU用例与5项真实CUDA packing用例均通过、无skip。unknown wait用例涵盖
caller句柄全部释放、48B私有预约、真实退役后恰一次释放；加强版还在同一cache/constant/key下
退役首个失败prefix后再请求，证明未完成制品未缓存。另核custom allocator最后free及失败不消费Buffer。
CUDA用例覆盖矩阵转置、BF16专家单节点/region、冷Cast、其他消费者、动态权重与incoming control。

RTX4070（driver617.14）上的独立Qwen128prompt＋31fixed-decode与上一轮保留probe比较，
64份hidden/logits逐byte一致，32个ID及capture计数一致。这里只复测正确性，不给新的吞吐、
显存/内存、质量或llama.cpp结论；SDK/native kernels/tensor_program均未重编，Linux/macOS本轮未运行。
失败packing当前保留未发布Plan的全部ready常量/lease至退役，后续可缩小为实际用到的owner集合。
其他Status-only冷复制路径仍待迁移，本节不能作为所有冷复制或整体重构已完成的证明。

归档431项payload（含23份当前源码快照）共15,792,181B，SHA256
`8c3da3d91bf2d0bb9a3c22c63ba158fb8ec0c7aa5e762f838d9285e3d1e8dfc2`，CRC及逐项SHA核对通过。
随后经根目录test.ps1精确删除20个可重建image/map/lib/exp，回收615,728,995B；
object、SDK与raw vectors保留。[清理回执](cold-constant-retirement-reclaim.json)

## 2026-10-19 冷常量复制的失败退役

[本轮源码、命令与结果](cold-copy-retirement.zip)补齐 Compute 的 queued-direct canonical copy：
每个 target 的权重复制共用原 QueueSequence/Completion，首次复制前检查 ready lease 与实际源、
目的、metadata allocator 所有者；首次等待前挂入本批源与 canonical 根，不保留整张 Plan。
首错后结束原前缀，观察错误不授权释放或发布 ready，也不重投到其他路线。
raw queued source 在接纳前明确拒绝；同步 immediate copy 仍可借用原 view。

Windows 增量4个TU、5项CPU回归通过、无skip。双权重故障用例在unknown wait后释放全部caller句柄，
仍保留4个allocation及External32/Constant32预约；确认实际退役后恰一次释放资源、Context和预算。
原接受后失败、packing unknown与canonical cache回归亦通过。两份CUDA镜像仅重新链接缓存对象，
5项真实CUDA用例通过；RTX4070/driver617.14上Qwen128prompt＋31fixed-decode重放的64份
hidden/logits逐byte一致、32个ID及capture计数相同。没有重编SDK/native kernels或CUDA host TU。

源码核实：当前Vulkan原生Status冷搬运等待真实终态并复用最大staging，Apple路径为同步共享内存
访问；这两条路径没有同类异步返回缺口，本轮未改它们。上一节待查Status路径不能一概当成需要异步迁移。
失败copy前缀仍可保留本批已准入但未提交的后缀至退役；packing失败保留未发布Plan的成本仍在。
本轮不作新的吞吐、显存/内存、质量、llama.cpp或Linux/macOS运行结论，整体重构仍未完成。

归档180项payload共15,431,389B，SHA256
`74cde937882506b721f0ffba1350fe47bc211381c9abbd24d438ad1cc75651b2`，CRC和逐项SHA核对通过。
随后经根目录test.ps1精确删除12个可重建image/map/lib/exp，回收406,939,665B；
object、SDK与raw vectors保留。[清理回执](cold-copy-retirement-reclaim.json)

## 2026-10-19 参数化 recurrent GDN 与原生资格

公共`RecurrentGatedDeltaNet`不再限定B1、D128和若干模型头数。它将Q/K/V类型、B/T/Hk/Hv/Dk/Dv、
循环头映射、状态和快照推导委托普通GDN；自身保留raw a/b `[B,T,Hv,1]`和共享逐头参数`[1,1,Hv,1]`。
先在临时输出上完成全部检查，再发布两输出；不增加算子、参数注册、执行器或兼容入口。
可表达F16/BF16/F32、非方形状态和合法plain strides，不表示所有后端可以执行这些数学域。

CUDA原生GDN只实例化D16/32/64/128，其他D进入native abort，grid.y又承载B；旧共用能力检查未完整核对。
现于Direct/Graph共用承载分析增加真实D集合与B<=65535，同时修复普通GDN的同一缺口。
不故意执行旧abort作负控；D24/B65536仅做descriptor准备拒绝，不分配大batch payload。
CPU仍独立核dense F32/方形状态等实际限制；Apple/Vulkan已有各自资格，静态核对后未改代码或宣称扩大执行覆盖。

公共头包含内联schema，因此按实际依赖刷新28个已链接host owner，再加入完整CPU provider与ops fixture，
共30个host编译（29份不同源码），没有native数学/SDK/CMake重建。三镜像链接输入CPU114、CUDA133、Qwen125；
CUDA继承已验证的8个alias核心对象，避免混用私有PlanState布局。30份实际sourceDependencies均确认读取新公共头。
25份原未解析数学/分配对象按源include闭包排除header依赖，此分类不表示它们被重新编译。

Windows实际CPU11项、CUDA4项全部通过，零fail/error/skip，覆盖普通及组合GDN、原B1/H16/D128与新增B2/H2→6/D32、
快照时间序、rebind、非法域准备拒绝及推导失败原子性。新数学推导另覆盖B3/H3→9/D17→24、三种QKV类型和padded Q/K。
实际recurrent计算沿原独立FP64 oracle及3e-6容差，没有更改oracle或放宽预算。CPU、CUDA、Qwen各用自身链接环境。
一次fresh完整128prompt＋31固定输入，64份hidden/logits与改前逐byte相同、有限，预测和32×7 capture旅程一致。
新B2计算由Direct用例验证；完整Graph仍由当前Qwen旅程覆盖，不据此代签所有新shape的Graph或其他平台。
本批未测吞吐、kernel数、峰值内存、独立质量或全模型/量化覆盖，整体架构重构继续。
[源码、依赖闭包、编译及实际回归记录](recurrent-gdn-schema.zip)共602个payload，CRC与逐文件SHA已核。
归档后经根`test.ps1`精确清理12个可重建镜像及map/lib/exp，回收367,604,232B；
保留对象、配方、当前Qwen/CUDA与共享参考。[清理回执](recurrent-gdn-schema-reclaim.json)
