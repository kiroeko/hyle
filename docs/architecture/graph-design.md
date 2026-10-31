# Graph：逻辑计算与原生执行计划

Graph把模型无关的typed DAG编译为可重复执行的程序，拥有边界、依赖、放置和资源规划。
Compute拥有直接计算与原生执行，Provider拥有片段内部程序；compiled plan不是第三个公开产品。
项目内的引擎层负责把模型组成这份 DAG，并拥有 tokenizer、生成、会话与请求政策；
Graph 不依赖引擎或模型名单，也可由独立 C++ 程序使用。
实际用法见[用户Graph指南](../user/graph.md)，共同实现选择见[Compute冷路由](compute-design.md#冷路由)。

## 图、制品与运行状态

逻辑图描述多输入/输出、常量、operation/custom execution、View/Copy、alias、资源访问与控制顺序。
数值与物理编码是合同，不能将opaque/quantized bytes当普通标量，也不用vendor op枚举作公共语义。

不可变制品共享计划、代码、常量和合法packing；并发运行槽独占bindings、scratch、mutable native
state和结果。同一槽未完成不能重绑、覆盖或重放，排他状态持有到终态，不只保护enqueue。
共同Invocation接纳后失败会退休该槽，后续使用须重建；接纳前拒绝不等于槽已失效。
输出可由caller、Graph或Provider拥有，来源alias也保留真实owner；不为统一ABI制造多余复制。

会话状态与执行槽按各自寿命拥有：引擎保留完整Snapshot和历史，按实际执行并发准备Invocation、IO与workspace，
不按驻留会话数复制执行载体。串行请求可跨会话复用同一载体；借用须在写输入前取得，到真实退休及最后结果读取后归还。
若多个槽共用caller IO，就须整体排他，不能只凭不同Invocation授权覆盖；停止空闲会话不关闭他人的借用。
失败槽在真实退休后重建，成功发布后诊断读取失败不回滚已提交状态，仍须归还载体。

编译不偷执行用户输入或改变副作用。常量borrow/materialize/packing具有明确owner；兼容性未知时，
不能因地址或大小相同就共享。跨制品cache由消费者显式拥有，不建立隐式进程全局cache。

## 形状与副作用

图可表达有限exact形状与有界symbols/guards。flatten、concat、head split、broadcast等派生关系
由只读shape program维护，避免引擎重复维护中间相等关系。checked加乘、整除、广播与typed guards
足以覆盖这些需求，不引入通用符号求解器或模型脚本解释器。

接纳前核对overflow、元素数、stride、alias span、workspace及整个域。边界固定不能掩盖内部动态；
无法证明全域时缩窄域或选择exact形式，不以端点抽样代替证明。View默认保留实际stride/alias，
不能因没有kernel而自动丢弃资源、控制边或强制dense复制。

schema声明资源的read/write/read-write。调用者用data/control边表达同一canonical owner/subrange冲突的顺序，
图构建结束时验证；node添加顺序不是effect依赖，Graph不自动补边或用全局effect锁串行化无关工作。
可证明独立的资源可并行，未知副作用不能谎称pure。
跨Invocation共享可变状态由调用方约束。Provider须保存片段完整副作用，不能只删拒绝条件。

## 冷编译与有界规划

编译先验证语义/描述，一次发现单算子叶提案，再为region发现提供当前target、实际Provider实例槽的
借用偏好事实。最低排名仅来自已发现且接纳的叶提案，不保证prepare或预算成功；metadata View不虚构数学排名。
region是可收缩的node set，不必连续编号；例如SiLU与Mul可跨独立的另一输入producer形成候选，
但收缩不能产生离开后重入的新环，公开中间值、读者和数值边界仍须保留。只处理有限提案，不枚举所有子集。

硬约束先于性能选择。ggml CUDA普通组合只发现不覆盖已知更优叶的islands，非优选节点保留原leaf后备；
未知或region-only能力不伪造排名，也不额外保留完整普通大岛作为预算后备。真实融合有独立实现身份、
数值和成本事实，可以跨非优选单算子。Graph对copy/repack、同步、dispatch与workspace作有界判断，
不试跑争夺赢家；覆盖更多节点不证明成本更低，region、物理融合与capture也不能互相推证。

有界规划不能找到可行计划时明确拒绝；已找到合法计划但优化预算耗尽时可报告优化受限。
这些预算限制计划搜索的计算成本。只有接纳前、无副作用的能力/资源拒绝可尝试后备，
真实错误不吞掉，接纳后不换实现重算。

配额或实现能力不足可以拒绝候选；实际metadata分配失败是原始错误，不能吞成资源拒绝再换Provider。
计费记录本身分配失败也遵守这一区别，避免缺内存时意外尝试更贵的后备计划。

端口对齐也属于候选资格。资源落位后，Graph按当前实际alias链检查自有caller backing的静态偏移；
不相容时在制品准备前撤掉该候选，沿原硬约束重选。真实Copy或物化结果有独立根，不因旧逻辑View误拒。
动态偏移、外部／常量及原生生成载体仍保留各自的后续验证，未知地址不当作已知不对齐。
Provider须声明实际读写要求；分配器偏好的256字节边界不能自动变成每个标量算子的端口限制。

## 并发、内存与时间线

冷常量准备同样遵守接纳点。同步 factory 直接结束；需要 Queue 的 factory 在原 source Context
创建的冷 QueueSequence 中记录，Graph 在首次观察原 Completion 前接入保活，再等待准备结果。
观察错误不证明 native 退役，失败的 partial payload、canonical/read lease、实际 resource/allocator、
resident/transient charge 与 native workspace 都保留到原完成机制确认退役；成功后才发布或缓存。
临时程序和 transient charge 在准备退役时释放，packed payload 的 resident charge 随制品释放。
借用 MemoryResourceRef/TensorView 本身不能构成失败后保活；StorageOwner 移入真实 typed resource，
保留原 Context 和实际 metadata allocator，ready ReadLease 保留原 allocation。可选异步 packing
资格不足时保留原 runtime 运算，已经接纳的错误不能换实现重做。Compute 常量冷复制按实际
transfer mode 区分：immediate 可同步借用；queued-direct 要求源 ready lease、实际源／目的资源及
metadata owner，在接纳前检查。每个 target 的原 QueueSequence 承载其全部复制，原 Completion
只保留本批源与 canonical 根，不保留整张 Plan；首次观察前接入保活，成功后才将副本标记 ready。
当前 Vulkan 原生冷复制等待实际完成后返回并复用最大 staging，Apple 原生冷复制为同步共享内存
访问；不能把这些同步实现误当成缺少保活的异步路径，或只改返回类型便宣称支持异步资源所有权。

先根据data/control、lane顺序及跨lane依赖建立happens-before，再决定backing复用。
不能仅按拓扑编号区间忽略并发。默认保留并发，只有明确允许才增加可报告的串行依赖换内存。

最终Completion须覆盖本次每个已接纳segment的完成观察与资源退休；同Queue的后继事件即使覆盖了
较早的设备工作，也不自动释放较早Completion持有的租约。成功和部分接纳后失败共用此规则，
全部退休后才允许发布结果或复用Invocation；聚合容量按计划segment数冷准备，不按Queue数估算。

Graph给自身backing与必要转换调用Mem的lifetime/quota机制；Provider私有heap和生成输出由原owner规划。
报告区分编译峰值、唯一常驻权重/制品、有限variants、并发scratch/capture、外部借用和未知driver/private成本。
同一allocation不重复计费，未知不当零；仍有capture或异步引用就不回收。
完成结果可通过`GraphOutput::storage_lease(retention_budget)`显式接入保留预算；原LeaseDomain继续持有
backing和native-use权限，预算中的非拥有索引只合并已证明的共享backing身份。原池的费用转为同一共享记录，
不重新扣配额；失败不消费旧权限。该准入不替代Provider分配前的资源上界，也不把未知私有内存改记为零。

target表示可执行条件，storage表示数据域，lane表示顺序；多stream不制造多个假设备，
unified memory也不取消依赖。第三方时间线保留原生对象，不要求伪造Hyle Queue。

## 原生组合与重复执行

单路线及CPU＋GPU组合使用同一图语义，保留不同compiler、输入绑定和完成方式，见
[路线设计](compute-routes.md#cpu与gpu组合)。不以mixed SDK编译成功推导全部混合数学或硬件支持。

共同流程的边界由实际计算决定：引擎提供图、状态与请求政策；冷编译决定片段、lane、依赖及搬运；
Provider保存已选程序及原生执行事实；原Invocation保存绑定、workspace、结果和在途寿命。
prefill、续算和decode改变的是输入、合法形状及状态，不另造一套执行槽或完成协议。
新增普通算子沿schema与Provider准备能力接入，不向共同推进器添加模型、算子或Provider名单。

共同点从实际计算链提取。当前owned-layout分页与前后普通数学已使用同一图、原Invocation、页权限、
提交和结果寿命；Vulkan可冷绑定固定backing，Apple则要在原FIFO完成后投影生成Array，二者保留各自适配。
布局更新可能替换原生程序，冷复制缓存须比较真实来源，不能把外层Graph槽地址当作结果身份。
变长attention也由所选prepared实现提供同一冷布局能力，页权限只由真实分页recipe授予。
显式metadata端口由typed prepare绑定owned layout的只读租约；配置输入不能被普通bind覆盖。
布局重准备只撤销旧配置、准确页权限和来源准备信息，保留普通调用者绑定及其lease／InputSource owner；
显式`Invocation::reset`仍清空全部绑定。同一metadata端口必须使用同一owned版本，上传复用原Source与实际
输入storage，不在各后端再建一份公共metadata owner。Q/K/V仍可来自普通Graph节点。
已退休的固定投影可保留原物理span候选；重新绑定须逐端口核对完整TensorView和storage身份。
物理span不携带租约、代际、命令或Queue资格，每次准备仍核新权限并封存当前槽。
Commands失效也可保留上述几何事实，不能以没有capture身份为由重复构造整图View；
Source、别名、跨存储副本、原生私有载体及实际Queue更换保留完整准备路径。
Graph生成的普通Copy不含Provider私有调用参数，可以沿原录制参与复用；源和目标的完整描述、
resource、handle、offset、extent及实际Queue仍须命中同一记录。由Region实现的复制仍核Provider合同。
[CPU完整tuple与CUDA输出复制检查](../evidence/correctness/inference-capture-copy-2026-10-31.json)
分别验证这些范围；捕获资格不授予Source或投影读取权限，也不代表跨turn预热已经完成。
无别名投影使用编译时消费者索引核查实际读者和所选实现，页范围与权限仍逐次验证。
跨资源的原生数学镜像仍由实际provider保留；只在真实resource相同时共享镜像，逻辑输入使用一个
canonical只读绑定。CPU搬运按资源声明的稳定主机地址事实证明别名，不按内置resource身份放行。
容量型原生页节点编译时没有owned schema，保留不可变capacity／数学／端口事实；第一次NodeLayout准备
才使用其真实schema建立slot recipe，不能把它伪装成操作自带的recipe。显式metadata只取原配置租约中的
页ID子范围，不重复上传或分配镜像。Schema切换先过原退休／结果门，新冷准备成功后替换。
Graph在创建Invocation时缓存执行端点，因此该身份不能随NodeLayout改变。要求原Runtime队列顺序的
Vulkan组合在冷编译时约束执行域，采用已有同队列转换，并拒绝任何残留异步carrier；实际Invocation仍核
Runtime与原生队列能力。普通不带此约束的编译保留原默认选择，不靠放宽写页权限实现融合。
普通Paged/Ragged图冷冻结实际Q/output行数R，schema容量C只给上限，布局活跃行A必须满足A<=R<=C；
有限图的128/1入口可共享逻辑图与schema，各自保留真实大小端口和独立Invocation，不通过补齐查询统一形状。
输出未活跃尾部[A,R)为零；Apple/Vulkan按R-A构造，CPU/CUDA在原执行序列补零，CUDA清零属于原capture。
A=R不增加清零命令；TensorView本身已裁到实际端口范围，不以schema容量授权额外写入。
原生直接接口保留活跃行输出。相同query拓扑只冷更新KV布局，
query拓扑改变才冷重建原程序，共同submit不构图或规划。Apple原managed runtime对非初始KV长度仍有
原生array构造和求值费用，不代表热路径零分配。内核是否能承载GQA或compact MLA仍由实际数学决定。
内核几何、精度和完成方式由原生实现报告，Graph比较数学与资源合同；不在共同层复制内核分支，
也不为统一形状增加影子buffer、额外转换或第二执行槽。整模型暴露的新共同点继续按实际代价收拢。

失败原子性也沿实际提交点收敛：同布局 ready 页帧替换先完成资格与权限容量检查，
再由原绑定操作失效 replay 并提交新页权限；拒绝保留旧绑定。pending 来源有独立的在途
控制归还义务，不能套用 ready 替换。原生 import 可继续保有旧页，直到真实重绑或释放。
普通Graph的`batch::PagedAppend`是无tensor输出的外部写节点；`PagedAppendPages`持有旧帧与原脏页，
描述符只保留schema／role，行数来自各exact child的实际输入，同一symbolic Graph可编译不同输入长度。
带bank input的`bind_append`核对实际所选writer/reader制品及`depends_on(reader, append)`。
没有缓存读者时，先做`batch::prepare`，再调用不带bank input的四参数`bind_append`；
该入口只登记所选writer的真实旧页、脏页／尾权及控制资源，不伪造reader或输入。
两条入口都只接纳精确页写权限，不以整bank地址授予写权；外部写节点之间须有实际数据或控制依赖。
复制有效COW前缀、写新行和后继计算沿原Queue／Runtime、Invocation与Completion执行；原生执行源不要求调用方伪造Compute Queue。
真实工作退休后可用`take_append`取回事务并撤销旧绑定与执行准备；只有whole Graph成功才允许`ready()`发布新页帧，
失败或取消不能发布；已失败写页须在退休后显式释放writer lease并`discard()`，仅析构事务不代表页已恢复可用。
当前支持共享Queue的CPU/CUDA普通F16/BF16的fresh／COW及独占尾页续写。
后者接入原`PagedPageAppender`，旧只读快照保持有界，后缀写与其旧前缀读在原Graph中按root一次准入；
whole Graph成功且真实退休后才能扩大新帧的有效范围，失败后退休再rollback，不能扩大旧读者或提前回收。
Apple／Vulkan原生已接固定形状的普通F16/BF16行追加。实际所选writer声明有序写能力，冷绑定保原页／尾权；
原生reader按明确Graph边接同次写入，不制造ready帧或外部CopyBatchUpdate。Apple在原FIFO预检写结果，
Vulkan按同一硬件队列接纳已接受且无已知错误的写入；两者都在全部子工作真实退休且whole Graph成功后才允许发布。
本批F16/page16/D64的writer-only与COW→因果attention已实测；其它dtype／形状、原生在途故障及完整Qwen消费者另计。
encoded写页尚未接入；真实Qwen已有完整快照、共享前缀／分支及CUDA在途回收检查。
CUDA的metadata驱动写页可录入原Graph。描述符是本轮真实读控制资源，逐页权限仍按实际bank独立接纳；
仅所选writer能力与真实capture后端共同证明参数稳定时，取回事务才保留休眠录制。更新描述符、换发权限并重新seal前不能提交；
更换载体先销毁相关录制。不同Invocation不共享可变描述符，历史共享映射分裂时创建独立载体，不覆盖仍被其它writer使用的metadata。
同Invocation的selected写页槽在首次冷绑定建立非owning索引；后续只扫描这些实际槽，不反复扫描整图。索引不保存页权限、epoch或就绪事实；原绑定／退休／成功检查照常执行。

需要主机异步进展时，可在原Invocation上冷准备`prepare_progress`。可选Graph＋Sched装配只推进原
SubmissionCursor的ready frontier：根据准备完成的`blocks_host`和实际transfer mode，将阻塞段交给
BlockingPool，其余段保留原生提交。共同层不接管算子内部线程团队，也不把一个region当作一个kernel。
seal hook、Copy 步骤及 segment 的静态 `blocks_host` 基值随最终计划冷建索引，重排后重建，不在每次准备中重新分类全部步骤。
实际 hook、Copy 的资源/范围与 transfer mode、Queue/owner 以及本轮权限仍逐 episode 验证；静态索引不代替槽就绪或完成退休证明。
准备阶段保留完整端口描述/存储比较，相等的非拥有提交视图可省去重复赋值；租约、原生载体和 seal 检查独立于该比较结果。
Provider 的具体融合可跨普通叶偏好提出闭合候选，并在冷编译重核实际边界；共用原生程序和执行链，不要求引擎判断平台或新增共同执行器。
已由 Graph 承载的零命令 View 可直接作为复合计算的边界端口；后端按该端口的实际 shape／stride 准入。
例如 CUDA dense BF16 scorer 保留 K/W 的原 View，只覆盖数学节点与合法后缀，免费 metadata 的选取不会挡住后续复合计算。
依赖导入使用原Completion和实际端点；取消停止后续接纳，已接纳前缀保活到真实退休。
不使用此装配的Invocation继续直接提交。Scheduler任务／stack仍有原有分配成本；
原生完成对象缺少通知接口时，HostWait使用有界定时检查，这些成本不能宣传为零。

稳定prepared replay与声明有metadata/array等成本的managed执行分别表达；managed也不允许隐藏JIT/search/repack。
批次native planning属于提交前独立准备，capture能力与是否阻塞另行表达。形状或相关数值/资源合同改变，
须重新准备或使用预先有限的variant，不能在submit里重搜Provider或无限增长bucket。
分页布局准备和状态发布见[Compute设计](compute-design.md#分页计算为何需要独立准备尺度)。

原生批次若明确允许在共享Queue中录制，可直接加入原Graph的兼容区间，不再套一层执行器或捕获。
每次准备仍撤销旧请求的就绪事实并重新检查当前槽、布局、权限与Queue；普通TensorView地址不变也不能跳过槽检查。
只有原生计划、工作区、metadata载体与实际参数均保持时才复用物理录制；内容更新不等于地址变化。
更换这些资源前先销毁覆盖它们及共享metadata读者的录制，准备失败则保留未就绪状态。

参数化模型/资产版本变化时的通用增量失效层仍是未来需求，独立于当前typed DAG、有限variants和引擎KV策略。
若需要，应另外说明版本identity、依赖与重算边界，不给每次submit预埋dirty字段或提前制造新API。

取消停止后继，已接纳工作保活至真实终态。结束使用时先停止新调用，再显式等待/退休本owner已接纳工作；
不能假定名称叫close就会阻塞drain，也不把借用的Context、executor或allocator当作可销毁的私有资源。
