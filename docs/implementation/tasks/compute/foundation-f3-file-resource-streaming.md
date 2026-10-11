# F3：有界文件与计算资源双向管线

> 状态：六owner CANDIDATE，当前产品97858fbc8；原F3-06 suffix producer环已有限修复并复验，原FAIL保留；当前交付/终审待验。
> component / owner：新内部 factor compute_stream；compute仅补下述私有租约/退役接缝；
> native factory/adapter 归 compute_route_cpu/cuda/apple/vulkan；中央声明只投影这些owner的有限新边。
> IO/Sched/Mem 只消费既有能力；出现具体基础缺陷才另作定向 owner 修复。

## 共同执行合同

### 当前启动绑定

- 2026-10-12实际baseline：`a932606ee5796328742632e392d5c6c5595ce3fb`；
  Windows canonical工作树干净，[F2 terminal](foundation-f2-final-closure.md)及QUEUED身份已提交推送。
  compute_stream由QUEUED进入SCOPED，四个Compute route由各自旧terminal进入本卡SCOPED；
  trigger是已准入F3用户旅程需要新的自然native pipeline factory/adapter及SDK依赖边，
  不是重新审查旧数学或F1/F2缺陷。CUDA backend、Graph、IO/Sched/Mem和Provider保持CLOSED。
- SCOPED源码核对发现具体组合缺口：三个F2 controller的`output()`只验证当前replay的写范围，
  `reset()`销毁绑定，无法把一个大目标的独占写租约交回给下一小chunk；最后chunk不能代签整tensor。
  compute由其[F2 terminal](foundation-f2-final-closure.md)进入本卡SCOPED，仅为F3-03/05补私有
  actual-retirement后的writer交接及共享backing/最后reader观察；不重写F2 public输出合同或native copy。
- 本阶段只核对原F3-01至08的IO/Sched/F1/F2/native组合、source/sink拥有/借用、有限byte/slot/
  group/旧结果预算、取消/实际退役与两向错误；完成AUDITED并提交状态前不写产品。
- 形状审计输入见[源码与原模型头收据](../../../../evidence/correctness/foundation-f3-audit/inputs.json)。
  SCOPED分别已提交2971a8bd5/c82627ce5；当前AUDITED转移以c82627ce5作previous-ref，
  不跳过compute新接缝的SCOPED。审计只决定原八FIX；下面的局部实现证据不代签完整F3运行门。
- 遵守用户本轮明确要求：只验实际修改owner、真实直接消费者、新API与对应三交付；未受影响
  模块/成熟算子目录/其他facade不全局回放，复用已签且scope/合同/工具链/机器相符的F1/F2证据。
  原调查/纠正与单线程/内存预算不因新记录、目录或跨日重置。

- 最初设计研究基线：`edb2714384037b12a3d1e69839aa895fea17cec1`；不是本轮实施baseline，设计见
  [资源与流送设计](../../../architecture/compute-resource-streaming-design.md)。
- 本卡现为上述六owner的registry当前record；完整F3产品验证待运行，局部接缝证据见后。原设计研究收据作为需求输入，
  不把原子API已存在或F1/F2 PASS写成本轮F3通过。
- 主纵切：量化推理的外部状态/权重范围及真实成熟数学消费。
  正交纵切：渲染资产流送和分块矩阵管线；不加入 token/layer/KV policy。
- 复用 Mem/Util/Sched/IO/Compute/native 完成，不增加复制 kernel、数学算子、模型 loader 或第二套 runtime。
- 全部产品接线集中后冻结候选，批量执行适用检查；小修只运行影响门，同身份证据复用。
- 四路线资源/程序/完成独立。Windows 为唯一产品真相；CPU Win/WSL/Air，CUDA Win/WSL，
  Apple Air，Vulkan Win/Air；WSL Vulkan 按实际 GPU 资格，不以旧例外代签。
- 三交付：source/installed static/shared 普通 Direct+Graph 消费、SDK 迁址、ON/OFF 新头、路线缺席诊断、
  Windows DLL 同目录。不让普通用户添加内部 factor/provider/stream 开关。
- 默认 P0：有界资源、热分配/提交/完成计数与成本结构；没有 timed 性能赢家承诺。
  出现需性能裁决的具体反例才在稳定候选上按原统一仪器固定一次同合同有预算 P1，
  未资格/未运行如实记，不能无限换姿势或升级仪器。
- build1/test1、重型串行、启动前确认内存余量；复用源码缓存/SDK 构建树，完成后只清本批临时产物。
- 2026-10-12用户纠正后的当前执行策略：验证门与跨主机重型run仍串行，单门库构建按实际内存余量
  使用j4（余量≥12GiB）、j2（≥8GiB）或j1（≥4GiB），test1；本条取代上面的旧build1一刀切。
  不同配置只共享immutable source cache，不共享binary/subbuild目录；历史验证的并行度按原日志保留。
- 每轮 selected gates、`check-module-closure.py`、`check-docs.py`；
  registry 变化带 `--previous-ref`。文档/状态阶段不构建/P1。提交推送用 `[skip ci]`。
- 调研纠正预算：至多一个具体反例的一次补读、两次有限修复/验收；新公共语义或边界变化必须重新定界。
  所有 FIX/适用门完成则 terminal；不能把研究/文档/旧 CLOSED 计作本轮产品完成。

## 本轮SCOPED合同

- 主consumer：本地Qwen3.5-2B checked ranges的BF16投影和F32规范化权重，
  从原始文件分块进入真实资源，逐组Direct/Graph消费，再回写新临时文件并reload消费。
- 正交consumer：只按byte range/group/tag工作的渲染资产与分块矩阵；不解释模型名、层号或dtype策略。
- 已有原子：`io::File::read_some_at`/`write_some_at`及`read_exact_at`/`write_all_at`、
  `mem::FileMapping::bytes`、Sched Task/TaskScope/BlockingPool、F1 LeasePool/ReadLease/WriteLease、
  F2各路线CopyBatchProgram/Invocation/Submission。组合的请求、旧结果容量、取消与teardown缺口为SELF；
  原子与native拷贝为BORROW，文件原子发布及格式解释为APPLICATION。
- 依赖方向：compute_stream共同控制依赖compute、io_core、sched、mem及mem_vm；四route依赖
  compute_stream并拥有其原生factory/adapter。核心compute只提供私有数据/权限交接，
  不反向依赖stream、Graph或IO，不增加普通Lease/Completion对象常态字段。
  中央图、新factor CMake和四route本地source列表只作必需接线；不修改公共selector/closure/性能工具。
- 直接消费者：新增独立C++ file/resource consumer与F3风险测试；Graph只通过已存在lease binding组合。
  IO/Sched/Mem不改产品API；若出现具体基础缺陷，先另定该owner范围，不能顺手扩张。
- 形状审计有界：只核对上述现有原子与四native owner，首轮不超过45分钟；沿用已固定peer与原
  一次具体反例补读、两次有限纠正/验收，历史失败不因本卡状态/日志/日期重置；不重跑全库研究。
- 验证预声明：F3-02至06逐项正常/边界/OOM/错误/寿命/取消/关闭测试，F3-07真实四路线数学，
  F3-01/08只验新头和依赖接缝、普通source/static/shared、SDK迁址和路线缺席。
  CPU在Win/WSL/Air，CUDA在Win/WSL，Apple只Air，Vulkan在Win/Air；WSL Vulkan实际GPU资格单列。
  对共同控制执行ASan/UBSan及适用的跨线程释放/结果发布TSan；原F1/F2 native证据按身份匹配复用。
- 机器运行前读实际身份及可用内存≥4GiB，build1/test1且各主机重型串行；configure300秒、
  build1800秒、单测试120秒、独立consumer180秒。超时保存原始输出，不能换目录重置有限纠正。
- P0成本结构门：固定K个stage/inflight槽、有限请求/groups/ranges/result记录、共享bytes账单；
  确认分块上界和实际copy/native完成/host-wait，报告Task/IO/offload新增分配，不谎称全旅程零分配。
  不执行P1/P2，不承诺厂家私有内存上限或吞吐赢家；新反例需重定界，不能无限追加benchmark。
- 停止：八FIX及七门全部证明才CLOSED；适用门缺证写BLOCKED/INCONCLUSIVE。完成即停止F3，
  不追加完整模型、F4/F5或其他模块全局验证。

| Gate | 本轮所需证据 | 当前结果 |
|---|---|---|
| scope | 原八FIX、主/正交旅程、六owner及实际baseline | 固定范围已落实；不扩F4/F5 |
| architecture | API/生命周期/依赖审计、现有原子组合与有限ledger | 产品已实现；完整交付图与最终审查待验 |
| correctness | 02至07逐能力风险矩阵及四路线实际接缝 | 运行风险已齐；new installed headers尚待验，聚合INCONCLUSIVE |
| delivery | 01/08 source/static/shared、新头、迁址及缺席 | NOT_RUN |
| usability_docs | 普通C++自然工厂、双向与错误/close recipe | source指南/样例已运行；installed尚未验证 |
| red_team | 候选冻结后独立一遍反证检查 | NOT_RUN |
| engineering_performance | P0成本结构、有限容量与实际路径账单 | 源码/实际成本已记录；最终候选裁决待验 |

## 需求与来源

llama.cpp loader 1509、1690–1760 的 staging/event/尾块与排空，
SGLang weight_utils 951、1327 的 bounded window/stop，l2_transfer 的逐层 ready，
vLLM weight_utils 的成熟 ParallelLoader/source view 寿命，以及 tiering/fs/io 的短读/写与范围检查。
共同机制是显式文件范围↔显式计算资源的有界管线，不是模型加载器。

正常旅程包括只读 source/mapping、分批 ready、读入消费、已有 resource 回写、
部分成功、backpressure、取消和关闭；不能只做一次同步读整 tensor 然后 WeightSlots upload。

## API / usability freeze

- 冷创建 Pipeline：既有 IO/Scheduler、native endpoint、固定 slots/bytes/请求/group/metadata 预算。
  原生 factory 接收真实资源；共同层只编排 IO 生命周期，不执行数学或拥有万能 Queue。
- 请求包含已有 File/Mapping owner、offset/length、resource range/F1 leases、有限 group/tag。
  拥有/借用 source 两档的寿命写清；完整 typed 描述由 consumer/成熟格式 adapter 给出。
- admit/next_ready或可等 ticket/whole-request completion、cancel_request、
  stop_admission/drain/close 提供完整 natural C++ 路线。
  在实现 API freeze 时固定名称和一次结果/错误形状，不增加无需求的重复 callback/poll façade。
- group ready 只表示该组成功数据可用，后继计算按原 native dependency 排序；
  old ready/use lease 占容量，不能提早复用 stage。group 后续失败仍保护此前成功组。
- 读：positional buffered 与 immutable mapping，direct IO 明确 opt-in且受原IO capability限制。
  写：已有 writable File/range、write_all_at；文件发布/命名/淘汰复用已有 IO/引擎规则。
  不写用户原模型文件，不隐式 mmap whole file populate 或下载 HF。
- bytes+slots 双重预算，包括 native padding、全部在途、pointer/control 和旧结果；
  容量满明确拒绝或现有 Semaphore/Channel await，默认不自动丢弃。
- 普通 CPU 数据不全部 pin；仅传输边界有 staging，源大于 staging 仍正常 chunk。
  Apple 共享物理内存仍有 CPU/GPU ordering/cache/ready；不承诺所有平台零拷贝。
- 原IO平台不支持安全中断 in-kernel read时取消停止后续任务，已接受部分仍退役；不把 timeout当停止。
  无私有 poller/threadpool，所有真实 completion/first error/partial progress 可查询。
- 包中默认随 Compute/Graph route SDK闭包交付，不新增细粒度用户开关；
  compute_stream 是内部解依赖因子，不新造必须手选的公开“Stream模式”。

## 有限 FIX 集

### 源码审计与实现形状冻结（产品仍未运行）

| 能力族 | 层级与现有证据 | 当前覆盖 | 裁决 |
|---|---|---|---|
| 定位读写/短传输 | IO原子，file.h及file.cc的read_exact_at/write_all_at | COMPLETE原子，MISSING管线 | BORROW原IO，SELF有限编排 |
| 映射/源寿命 | Mem原子，mapping.h的FileMapping及checked bytes | COMPLETE原子，MISSING请求保活 | BORROW；共同层仅retain owner，不解析格式 |
| 分块native搬运 | F2复合程序，三个copy controller及原native recording | PARTIAL跨chunk组合 | BORROW原程序；FIX私有retired writer交回，不扩大public output coverage |
| 大目标逐组ready | F1权限原子及F2actual completion | MISSING文件组合 | SELF；完整组内全部声明范围成功才发布，旧组独立保留 |
| 有界背压/结果容量 | F1 backing/blocks与ResourceBudget | PARTIAL结果与请求 | SELF固定请求/group/slot记录；BORROW quota，按共享backing去重 |
| 取消/close/晚完成 | IO OperationOptions、Sched CancelToken/JoinHandle/BlockingPool、F2退休 | PARTIAL跨owner teardown | SELF停止后继与合并首错；BORROW实际等待/退休 |
| 格式/math消费 | 现有成熟Provider与checked local inventory | ROUTED数学，MISSING F3 sample | BORROW ggml/官方manifest；APPLICATION格式、发布命名、模型策略 |

现有公开入口全部回映为上述BORROW；F3新公开入口只回指双向文件旅程、请求/结果与显式终止。
不新增benchmark入口、raw native函数回调、公开executor或另一种Queue。

#### Owner与私有接缝

- `include/hyle/compute/file_stream.h`与`src/compute_stream/`归compute_stream，中央
  SPLIT_PUBLIC_HEADERS只登记此头；公开类型暴露compute/io/sched/mem时取PUBLIC边，纯实现依赖取PRIVATE。
  四route各自自然头声明工厂及自身实现，PUBLIC依赖compute_stream；Graph路线继承原Compute route闭包。
- compute仅允许在batch_transfer.h/controller增加私有friend/access、在现有LeaseAccess实现增加
  backing identity/capacity及最后reader观察；不增加任何普通对象字段。共享header归compute，
  Apple/Vulkan access实现归各自route。CUDA仍使用共同F2程序，不修改CUDA backend或vendor lock。
- 私有`CopyStreamAccess::take_mutable(invocation,id)`只在无submission owner、无actual native use且
  所有已提交phase成功时交回已有WriteLease；未提交绑定可撤回，未知/失败不会交回可发布writer。
  交回后该binding失效，必须重新绑定/prepare。F3累计成功logical范围并仅在整组成功后publish；
  不调用F2 output最后chunk扩大视图、不通过reinterpret/subview越过coverage gate。
- 私有backing观察只读取现有domain与refcount/use，保持原锁与权限规则；ready记录内部保留ReadLease。
  外部ReadyGroup、导出的ReadLease及native消费者任一保活都继续占请求/结果bytes；collect只有
  最后reader/use归还且结果handle已释放才复用记录。不用retain_resource_owner覆盖native原owner。

#### 固定公开对象与调用

- `FileStreamOptions`：allocator=system_allocator、可选ResourceBudget；默认max_requests=4、
  max_groups_per_request=16、max_ranges_per_group=16、in_flight_slots=2、staging_bytes_per_slot=1MiB、
  staging_alignment=256、max_active_backing_bytes=128MiB。所有乘加溢出/零必要容量在冷create拒绝；用户可明确调容量，
  不新增CMake用户开关。stage必须host可访问，CUDA只此传输stage用原pinned资源。
  direct File按原DirectIoAlignment检查实际stage pointer/offset/每chunk length，未满足在IO前拒绝；
  tail不降为buffered，必要alignment由consumer明确配置。
- `StreamFile`是可复制retained handle：`adopt(io::File&&, capacity_bytes)`或
  `borrow(io::File&, capacity_bytes)`；adopt失败不移动File。capacity是consumer明确的合法范围，
  不代表文件实际已含这些bytes，EOF仍从真实IO返回。借用File地址与IoContext必须活到所有请求实际退休。
- `StreamSource::file(StreamFile)`、`adopt(mem::FileMapping&&)`、`borrow(const mem::FileMapping&)`；
  mapping只读、稳定且不release/populate/并发改文件；mapping offset相对于已映射logical范围。
  通过原Mem checked bytes和stage BufferView::write复制有限chunk；映射page fault可能阻塞，
  此fill沿同BlockingPool offload并保留mapping owner，不制造copy kernel或whole-file缓存。
- `StreamReadRange`包含file_offset、目标WriteLease、resource_offset、logical bytes、显式physical_bytes
  和可选TensorDesc；`StreamReadGroup`是tag及range span。`StreamWriteRange`有两档：
  `ready(ReadLease,...)`，或`produced(WriteLease&&,Completion&&,...)`，都含resource_offset、
  file_offset与logical/physical bytes；write group同样只有tag/ranges。produced档在原F2 ordered mutable
  source/producer路径真正退休前绝不host读；首chunk成功后可把已生产source转成readonly供后续chunk。
  描述符由consumer解释，typed范围必须完整覆盖desc encoding；同backing别名/offset/对齐/尾padding
  在任何IO/native接受前验证，不允许padding覆盖邻块。
- 四路线固定自然工厂`cpu/cuda/apple/vulkan::create_file_pipeline`，参数为已有IoContext、原Context、
  CPU/CUDA原Queue（仅这两条）、stage resource、冷target resource specs及FileStreamOptions；
  返回共同move-only `FileResourcePipeline`，不暴露内部factor或执行回调。
- Pipeline的`read(source, mutable group span, OperationOptions)`与
  `write(StreamFile sink, mutable group span, OperationOptions)`同步admit并返回move-only StreamTicket。
  固定记录全部检查/接纳成功才移动WriteLease/Completion；容量满Unavailable，失败不消费输入。
  descriptors/tag冷复制入固定数组，caller数组可随即释放，source/sink retained owner保留到实际IO/native退休。
- `StreamTicket::next_ready()`非阻塞；`wait_next_ready()`按原Sched同步等待；
  `wait()`返回完整StreamResult（Status、实际read/copied/written bytes、ready组数与retirement_proven），
  `result()`观察同一快照，`request_cancel()`只停止本请求后继。pending返回Unavailable，
  已取完且成功结束返回OutOfRange，失败仍保留首错和partial bytes，不丢已有ready。
- `StreamReadyGroup`可复制，tag/range_count/`output(index)`提供覆盖范围的ReadLease；
  写回组报告实际write完成和bytes，并可返回原source的readonly lease，不新造GPU输出或复制source。
  结果记录与导出的lease寿命分开计费。
- `stop_admission()`、非阻塞`collect()`/`close()`、显式`drain()`；close重复安全，未退休返回Unavailable。
  drain停止接纳并等已接受工作，不等用户放弃旧ready；Pipeline/Ticket析构放弃观察及停止后继，
  不隐藏join或全设备wait，不关闭借用用户File/Queue/Context/全IO runtime。

#### 并发、进展与成本

- 共同行程只有有限slot工作任务，由已有IoContextAccess取得其原Scheduler/BlockingPool；
  不要求用户重复拼Scheduler身份或建新runtime pool。create冷保留K个原Mem guarded stack及stage/F2 slot，
  通过原同步原语等待新工作；native阻塞wait用同BlockingPool offload，不阻塞Scheduler worker。
  TaskScope要求析构前显式join；不把它藏进非阻塞Pipeline析构。源码核对JoinHandle在entry结果产生
  时唤醒，TaskDriver清理在稍后的finish hook，不能用join结果代签实际清理。
  每组在已有File.owner_worker上spawn_on，固定slot使用原Mem FiberStackPool冷保留并回收K个stack；
  不移动或重新绑定已经启动的Fiber。现有source-only src/sched/fiber_access.h仅借set_on_finish，
  不改Sched产品、不读写其runqueue、不向SDK用户暴露private hook。
  原Scheduler finish hook离栈后回收stack，再在同一owner锁下退掉worker引用并通知drain；
  析构只停止后继/放弃观察，accepted Fiber/native owner继续保活。没有第二runtime或私有poller。
  native wait/mapping fill复用BlockingPool的既有BlockingJob，一槽一个冷embedded cell；destroy边界
  后才通知文件Fiber，避免普通JoinHandle在job实际清理前发布。原IO read_exact_at/write_all_at继续复用，
  不复制短传输/deadline/cancel算法；io_calls记composed IO入口次数，不冒充底层syscall数。
- admission/状态快照与ready发布由原锁/ConditionVariable串行；Pipeline控制与同Ticket观察为
  thread-compatible，request_cancel和ReadyGroup/ReadLease复制释放跨线程沿原合同；
  同一个native slot只有一条工作任务修改，shared Context/Queue原host锁不绕过。
- 一个group可大于stage，按有限chunk循环；独立groups可占不同slot。只有该chunk的IO和native
  actual-retirement都已完成才复用stage，旧目标结果独立占backing与result记录，不能被后组复写。
  FIFO接纳不承诺物理重叠或完成顺序；组成功即通知，不等整个request再发布。
- Read：IO实际short progress→stage ready→F2原生copy→实际退役→交回writer→下一chunk；
  全组成功后F1 publish。Write：F2保留producer/readonly source→原D2H或共享内存可见→stage
  actual-ready→原write_all_at→计实际progress。write故障后的文件prefix不回滚，原子发布留给IO consumer。
- 取消沿原OperationOptions/CancelToken；无法安全中断的read/write及GPUcopy必须退役，停止后续chunk。
  F3统一选择cooperative chunk档：token/deadline在管线边界检查，不向原File声明单操作强取消；
  已接纳read_exact_at/write_all_at无伪造interrupt budget，返回可能晚于deadline，待实际IO退休后
  停后继。该档同样支持Windows adopted handle与POSIX blocking文件，不按OS猜某个File的native_async。
  unknown native observation保留slot/quarantine、retirement_proven=false、无ready/复用/假取消；
  collect/drain只调用原退休入口，不新增定时poller或无限重试；原sticky CUDA进程终止合同继承。
- 冷metadata为O(requests×groups×ranges + K×F2容量)，admit检查范围/别名/去重O(R log R)，
  chunk数量O(ceil(bytes/stage))，每chunk使用原IO/复制/退休；报告原Task/IO/offload allocation和
  host waits，任何新增临时allocation走已注入且受有限预算的allocator，不以replay零分配代签整个旅程。
  同backing只计一次，external/managed/native private unknown分列，不把caller整个文件读入或全部CPU数据pin。
  FileStreamReport的metadata_bytes计共同固定数组与copy-controller；原Mem租约/IO/Sched/offload
  的分配留在各自注入allocator账本，P0 fixture以既有StatsAllocator测全部新增分配及归零，不能猜为0。

七问裁决：第三方仍用原锁定native/成熟数学与consumer格式工具；MUST就是原八FIX与本表；
复用原IO/Mem/Sched/F1/F2，不新建runtime；P0只做成本结构；只C++三交付无FFI；中央依赖/头/安装
按新factor与四route投影；普通用户从四route工厂完成read→ready→math与write→reload，指南保留借用/取消/unknown坑。
以上为AUDITED前形状结论，不是产品PASS；实现若发现必须改变owner/API/常态成本，先退回审计再改卡。

#### 首项产品接缝进展

`e6c1d88f1039b237ef1ce412030c706bb4d23c10`实现共同私有CopyStreamAccess writer交回和
LeaseAccess原backing/最后reader观察；没有新增普通对象字段、native copy路径或核心反向依赖。
[Windows收据](../../../../evidence/correctness/foundation-f3-handoff-windows/receipt.json)绑定精确source SHA：
原CPU实际copy consumer五项PASS（含两项新接缝测试），受影响三个头独立编译PASS，原树build1仅十项增量。
16-byte stage四chunk填满64-byte目标，同writer/generation保留、最后才统一publish；持有submission
拒绝交回、非法/readonly binding不消费、共享backing去重、额外reader/use占容量及closed pool最后reader均验证。
这是F3-03/05的部分接缝，不是文件管线或完整F3验收；GPU pending/error/unknown、Apple/Vulkan交接、
文件请求/双向实际IO、Qwen数学与三交付仍UNPROVEN，六owner保持AUDITED。

#### 共同/CPU管线主体进展

`bc9122a0492a771de790ad59d9760fca4817f14e`实现StreamFile/StreamSource拥有及借用、固定请求/group/
range/stage/result记录、自然CPU工厂与中央新factor、原IO双向chunk、F2交接、逐组ready、背压、
取消/stop/collect/close/drain；IO源worker 1→目标worker 0→reload的真实旅程已验证。
[CPU主体收据](../../../../evidence/correctness/foundation-f3-cpu-subject/receipt.json)绑定全部实际源码：
Windows六项文件风险/正常旅程PASS、受影响五头独立编译PASS，每项运行后注入allocator balanced。
真实EOF保留138已读bytes/128已复制bytes与前组ready；旧导出ReadLease继续占request/backing容量。
首次文件运行的owner-worker fatal以及最初CMake/fixture编译接线失败保留在同一raw archive，
没有换目录重置预算；按原File affinity修复F3，未改IO/Sched/Mem产品。
CPU route正常链接所需oneDNN等依赖已构建，不因此执行其算子目录或全局/P1验证。
固定stack仅用既有Mem池，不是第二个线程池；drain等离栈及offload destroy确认，不等用户释放旧ready。
metadata_bytes是固定共同/copy控制数组账单，原Mem/IO/框架分配由注入allocator观测，不宣称全部0分配。
此项只证明共同/CPU主体首验，F3-01至08仍没有完整terminal：三GPU factory/producer/unknown/late，
WSL/Air、sanitizer、Qwen混合dtype和GGUF实际Direct+Graph、source/static/shared交付、完整P0与终审继续待验。

#### 三GPU工厂与文件首验进展

产品接线`8b35c527f57a91f89545a11748ff7be64e080de1`加入自然CUDA/Apple/Vulkan工厂，
共同冷目录/typed F2 slot适配与各自私有writer/geometry/不发布的退役观察；没有新增数学/复制kernel，
也没有修改IO/Sched或native CUDA backend。三route从中央图取得compute_stream与IO公共类型边。
CUDA actual双槽回写首次失败`ordered queue already has an active submission sequence`；
F3 adapter只锁bind/prepare/submit，提交结束即解锁，不把IO或实际GPU等待串行化；复验通过。
[Windows局部收据](../../../../evidence/correctness/foundation-f3-native-windows/receipt.json)签该产品内容的
CUDA/Vulkan各两项真实GPU文件测试、各48次native calls/phase completions。旧SDK测试配置拒绝、
旧Graph混合测试fragment缺席及Vulkan旧catalogue夹具缺头均保留；F3改用具名独立模块测试target，
不借此重写公共工具或无关算子。CUDA原shipping SDK配置已恢复，只恢复配置不代签新SDK交付。

Apple首验原错误`bounded file IO requires a natively cancellable backend`在原始日志及两次有限
test-only诊断中保留。`19bede70475d7e9fd719a217b2f8b57a1d1c8aa3`按上文原cooperative合同
修复共同F3 IO调用，不向blocking File伪造单操作取消，原IO产品不改。
[Air Apple局部收据](../../../../evidence/correctness/foundation-f3-apple-subject/receipt.json)绑定完整
canonical archive、Mac16,12/Clang21身份：两个真实Metal文件旅程、48次native calls/completions及
allocator balanced PASS；同产品Windows CPU六项回归PASS。原CUDA/Apple两个产品接缝纠正计入
本轮既有有限纠正记录，不因日期/目录/测试拆分重置。
[Air Vulkan局部收据](../../../../evidence/correctness/foundation-f3-air-vulkan-subject/receipt.json)同树
Mac16,12/实际Apple M4 GPU两个文件旅程及48次native calls/completions PASS；不代签WSL软件GPU。
cooperative IO新树的Windows GPU/WSL增量、GPU
producer/pending/unknown/late/cancel风险、实际Qwen混合dtype/GGUF Direct+Graph、三交付/P0/红队待验。
六owner保持AUDITED，局部bytes成功不是F3七门或完整数学验收。

#### F3-06已接纳producer生命周期接缝

`f3d931c47777f7fbfd35dedf1f5e0734fa28ef6c`完成本FIX剩余的admitted source producer保活：
每个已接纳async producer仅使用已有冷CompletionRetention，保留整个管线控制/stage直到原native
实际退休；取消/失败组放弃未绑定producer观察，交原owner退休，避免state→Completion→state循环。
pending producer计入retirement_proven与记录复用条件，零byte也不能绕过producer错误/完成。
已知hard error在接纳前原样返回、不消费输入；仍pending的零byte沿原BlockingJob等原Completion，
不制造强取消、第二runtime或新poller，也不改IO/Sched/CUDA backend或公开工具。
[生命周期局部收据](../../../../evidence/correctness/foundation-f3-producer-lifetime/receipt.json)绑定
Windows CUDA六项（正常两项、真实producer排序、取消、丢弃控制、normal/empty hard error）及CPU六项
回归PASS。整组先后TIMEOUT120、析构前still-use fatal、空范围错误ready与include构建失败均原样保留；
单项PASS没有覆盖原整组失败。新代码在CUDA整组连跑与原生uses/allocator归零后才签局部证据。
这属于原F3-06未完成的pre-candidate implementation/acceptance，不新增FIX或重置已有纠正/失败记录。
新producer节点的Apple/Vulkan、unknown/late/fault与sanitizer/跨平台仍未验；数学、三交付与七门未完成。

#### Windows真实数学consumer局部验收

新增独立普通Graph consumer `samples/package/graph/file_streaming`，沿同一F3 read→逐组ready→
原Direct/Graph数学→write→reload→数学路径；CPU/CUDA两lane、三次replay的BF16与已有GGUF Q8编码
输出均符合参考。初次样例Result lambda返回Status的编译失败保留，修为std::unexpected后整条旅程通过。
[数学局部收据](../../../../evidence/correctness/foundation-f3-math-windows/receipt.json)绑定精确样例SHA及
原始build/math日志，不代签安装SDK或全路线验收。

同收据签真实本地Qwen第0层三个完整BF16投影和F32 norm的CPU/CUDA原Direct及Graph输出：
官方Safetensors工具只核验并读取所声明原始范围，独立稀疏输入参考不靠HYLE算子生成；
实际F3读取75,497,984 bytes，stage 2,097,152 bytes，峰值两个slot，不缓存完整4.55GB checkpoint。
完整输入身份与范围见[checked input](../../../../evidence/correctness/foundation-f3-qwen-ranges/checked-input.json)。
这不是完整FFN/模型生成验收；该首份收据没有覆盖Apple/Vulkan或三交付，六owner保持AUDITED。

#### Windows原始权重完整往返与成熟GGUF消费

consumer `4a757351a707b17655d3aaf5471325c3dd3bb6f2`完成同一Qwen范围的双向旅程：
read→逐组Direct/Graph→写新临时文件→64KiB chunk逐字节对照原权重→reload→Direct/Graph再次对参考。
[往返收据](../../../../evidence/correctness/foundation-f3-math-roundtrip-windows/receipt.json)签CPU/CUDA/Vulkan
三个真实Windows路线；每路线logical payload 75,497,984 bytes、实际native copies 226,493,952 bytes，
stage仍为2MiB，峰值两个slot。消费者回收已释放write记录后再次collect原read记录，再从同一有限pool分配reload，
未放大容量掩盖旧结果保活。

同收据签成熟`gguf` writer/reader检查原始GGUF文件offset、BF16/F32/Q8编码/shape及bytes，
普通小consumer沿这些原始范围读入、两lane三replay数学/回写/重读。Apple使用其原affine Q4表示；
格式工具留在consumer，不增加产品parser/数学kernel。首次Vulkan device mapping错误、MSVC局部lambda编译错误
及首次reload容量错误全部保留；修正只在consumer。原F3产品仍是f3d931c47，不重置产品纠正预算。

Air验证使用校验过的稀疏验证副本：只传原header与四段约72MiB payload，保持原offset与capacity，
其余模型范围是hole并明确不在本卡验收范围；不传/缓存完整4.55GB模型，不反向覆盖Windows产品。
该Windows收据不签Air数学、新生命周期接缝、WSL/sanitizer、故障边界、三交付或最终七门。

#### Air四路线数学补齐后的证据边界

consumer `b890d9b61ad2334eba2be8dad0fe22dd0a1808eb`在Mac16,12/M4/macOS27.0.1/Clang21上，
普通shared source的CPU/Apple/Vulkan实际Qwen完整往返通过；每路线同样是四段75,497,984 bytes、
2MiB stage、峰值两slot与226,493,952实际native bytes。原header和四payload在验证机再次逐段SHA核对；
其他hole不读、不计资格。CPU/Vulkan成熟GGUF小样例及Apple原affine Q4小样例亦通过。
[Air数学收据](../../../../evidence/correctness/foundation-f3-math-air/receipt.json)绑定完整canonical archive、
实际机器/工具链、精确consumer SHA与全部原始失败/成功日志。

Apple原dense carrier拒绝外部转置lease后，consumer保留原始dense Metal权重，Direct用已有MLX
compiled region在内部做native transpose view→matmul，Graph用已有permute view→MatMul；
没有host权重转置buffer、产品API改动或新kernel。首个Graph View缺显式Operation wrapper的编译失败也保留。
这是原F3-07/08消费者接线，不重新打开F1/F2或重置产品预算；主体f3d931c47保持不变。

至此四路线实际数学往返均有局部证据；六owner仍AUDITED，不能据此写八FIX/七门整体PASS。
下一批只补本卡仍适用的source/sink/producer/fault风险、WSL/新控制sanitizer与安装static/shared、
迁址/新头/缺席门，最后按不可变candidate做P0与红队裁决，不运行无关模块或算子全目录。

#### F3文件风险的直接原生资格化

测试fixture `452eada9b770950d28f3b8e2d8ee3ac735d468ef`只补原F3-02/04/06风险，不改变产品。
[Windows Vulkan故障收据](../../../../evidence/correctness/foundation-f3-vulkan-file-faults/receipt.json)
签实际RTX4070/driver617.14五项：原两项文件旅程回归；原QueueSubmit在四个成功chunk后拒绝后缀，
read=320/copied=256、首组GPU-ready仍保留原bytes；实际Fence观察/等待/drain未知时的取消与丢弃不释放
真实uses或F3控制，恢复原生观察后保留首错且metadata归零；原native producer依赖使文件回写得到真实0x5d bytes。
故障边界复用F2的原生函数表，不新增运行时、copy kernel或公共验证工具。

本批还新增有限CPU描述/offset/backing拒绝、拥有source及caller描述寿命、原只读sink错误、已接纳hot OOM，
以及Apple原MLX blocked prefix后真正接纳Metal producer的正常/取消/丢弃/late error测试。
它们和Air Vulkan同fixture在此提交时尚未资格化，不以具名测试存在或Windows局部PASS代签。
本轮所有失败/预算仍保留；剩余跨平台、sanitizer与交付继续按上文原范围。

上述fixture的[第二批风险收据](../../../../evidence/correctness/foundation-f3-file-risk-acceptance/receipt.json)
进一步签Windows CPU十项、Air Apple六项、Air Vulkan五项实际测试。CPU新增四项拒绝/源与描述寿命/
readonly sink原错误/hot metadata OOM均通过；Apple真实GPU prefix被阻塞后接纳原Metal producer，
正常依赖、取消、放弃控制和真正late Internal均保持原uses到实际退休、不错误ready；Air Vulkan复验同套
unknown observation/后缀提交失败与producer边界。native uses归零与F3注入allocator balanced由实际断言证明，
不外推驱动私有物理内存。产品f3d931c47完全不改；该批没有用修源码换掉原错误。

原始正数short-write prefix与显式跨线程导出结果释放仍待直接资格化；WSL/sanitizer、安装交付、
新头/缺席/迁址及最终P0/红队仍未完成，不能把本批21项测试成功当完整F3关闭。

[WSL数学与导出结果收据](../../../../evidence/correctness/foundation-f3-wsl-math-exported-drop/receipt.json)
补签GNU14.2/实际RTX4070的普通CPU/CUDA static source：成熟GGUF及同四段原Qwen范围
read→Direct/Graph→write→逐byte检查→reload→Direct/Graph均通过；stage 2MiB、峰值两slot及
每路线226,493,952 actual native bytes。原锁定CUDA leaf的12/13链接warning保留，运行结果实际通过。
输入仍是核验过的四范围稀疏副本，不传完整模型。

fixture `c36c40bedfb960384020c117703c4eda599a9359`的Windows CPU十一项也通过：另一实际线程
持有ReadyGroup/ReadLease时不能collect后复用；该线程实际释放并join后，同一有限record/backing恢复可用。
POSIX正数short-write fixture已加入但尚未运行；本卡当前F3 ASan/UBSan/TSan与WSL CUDA native测试仍待验。
ASan初次本地参数错误将factor与product route选择混用而被现有CMake拒绝；只修验证argv并保留失败，
没有改公共工具/产品或重置预算。源码/输入已同步核验，原适用交付与最终门继续待验。

随后同一ASan树的factor-only配置在原300秒上限TIMEOUT，日志显示旧cache位于Windows挂载盘，
触发新的locked LibTorch CPU下载/准备；配置进程终止已核实，测试没有启动。只将验证配置指向
已由WSL数学consumer使用的原Linux cache `/root/.cache/hyle/dependencies/v1`，锁定SDK身份不变；
native-cache配置实际PASS，当前同一候选的instrumented依赖构建仍在运行，不能记成sanitizer测试PASS。
原互斥参数FAIL及TIMEOUT的日志保留，改变cache位置不重置原产品纠正或配置/build/test预算。

native-cache的首次ASan build随后实际用完1800秒，日志停在未改的SIMD依赖，12项测试未启动。
CMake/Ninja/已记录编译子进程均已结束。2026-10-12用户明确批准保留原TIMEOUT，追加一次1800秒
给同一配置与同一构建树的剩余依赖及测试，并随后要求库编译按内存并行。本次追加独立计账，
不删除原失败、不改变产品候选或重置原纠正预算；goal暂停期间未启动，恢复后才执行。

恢复后j4复用全部oneDNN产物，41秒内推进到IO代码；原未改`tcp.cc`的error coroutine在GCC14
`-O3`/ASan下触发`maybe-uninitialized`，旧sanitizer树继承的`HYLE_WERROR=ON`把它升级为编译失败。
该失败保留；不是F3测试失败，也不能据此证明TCP无缺陷。GCC14
[官方instrumentation说明](https://gcc.gnu.org/onlinedocs/gcc-14.1.0/gcc/Instrumentation-Options.html)
明确提醒sanitizer可能增加此类假阳性，并建议不与Werror组合。正常HYLE本地默认WERROR=OFF。
因此本批instrumented验证沿既有CMake选项使用WERROR=OFF，警告全数保留、`-O3`/sanitizer/原完整风险集不变，
不加`-Wno-*`、不改TCP/Sched或公共工具、不代签TCP正确性；普通source/交付门继续按其原配置。
恢复沿原追加预算的绝对截止时间，重配置/构建/测试共享剩余时间，绝不重新生成一笔1800秒预算。

[ASan实际收据](../../../../evidence/correctness/foundation-f3-asan/receipt.json)现签同一c36c40bed fixture
十二项FileStream测试全部PASS、零ASan/LeakSanitizer报告与实际allocator/uses退休断言。四个关键TU
（共同控制、CPU工厂、私有batch handoff、风险测试）的`-O3 -fsanitize=address`逐项核验。
真正POSIX short write保留written=80/copied=96/ready=0及原始prefix；原文件限制与信号action在诊断前恢复。
另一个实际线程释放导出结果后的有限record/backing回收亦通过。原FAIL/TIMEOUT/警告全在raw archive；
完成时间在原用户追加预算的绝对截止之前。产品仍为f3d931c47，未因编译器诊断修改IO/Sched或F3产品。
UBSan/TSan、WSL CUDA新接缝、安装交付/新头/缺席/迁址和最终P0/红队仍待验。

[UBSan实际收据](../../../../evidence/correctness/foundation-f3-ubsan/receipt.json)进一步签同一候选十二项
PASS、零runtime error、四个关键TU实际`-fsanitize=undefined -fno-sanitize-recover=all`，保持`-O3`和
完整原风险集；使用独立binary树与原Linux dependency cache，j4/build、test1，未覆盖ASan产物。
真正80-byte short-write prefix、跨线程导出结果释放及OOM/生命周期断言再次通过。
TSan、WSL CUDA新接缝、安装交付及最终门仍待验。

[TSan实际收据](../../../../evidence/correctness/foundation-f3-tsan/receipt.json)现签同一c36c40bed fixture
十二项PASS、零race报告，四个关键TU逐项带实际`-O3 -fsanitize=thread`。首次测试进程在任何用例
启动前因WSL地址布局冲突`unexpected memory mapping`退出66；原失败保留，只沿F2已验的方式执行
一次进程级`setarch x86_64 -R`恢复，不修改系统设置、不加suppression、不重复尝试。
共同控制的ASan/UBSan/TSan风险现均有实际证据；不代签GPU sanitizer或完整F3关闭。

WSL CUDA定向入口首次配置撞上原树继承的`HYLE_DEV_FACTORS`与product选择互斥，失败日志保留；
沿该维护树只选`compute_route_cuda;compute_stream`并清除product选择，重配置实际PASS。
当前仅构建独立CUDA文件测试target，不回放Graph/CPU Provider或无关算子目录；其六项运行、
安装static/shared、新头/缺席/迁址与最终P0/红队仍待验。

候选冻结前清理本轮instrumented编译揭示的两处共同getter误导缩进，以及两个测试中省略的
empty optional deadline初始化。共同TU全部非空白token与f3d931c47一致；测试只是将原aggregate
遗漏成员的空初始化显式写出，没有新语义/API/layout/算法/成本。原sanitizer与native收据保留其
精确旧source身份；另做受影响owner/直接route的增量正常门，不把旧SHA伪写成新candidate证据。

最终逐能力矩阵核对发现原F3-06的deadline及正常重复stop/close只由源码/其他取消旅程间接支持，
尚缺直接组合证据。补一个公开API fixture：真正过期的单调deadline在第一chunk前保留
DeadlineExceeded/零IO/零native/零ready；stop后接纳拒绝且输入writer不移动，重复close/drain/collect，
并验证借用Context/Queue/IoContext/Scheduler及原File仍可用。该测试属于原F3-06风险补证，
不新增FIX或修改产品；本提交时尚未运行，不写PASS。

[WSL CUDA与deadline收据](../../../../evidence/correctness/foundation-f3-wsl-cuda-deadline/receipt.json)
现签真实RTX4070/GNU14.2六项原生CUDA文件测试，以及当前产品微修后的Windows CPU十二项。
WSL当前完整c36c40bed source上的原producer排序、cancel、abandon、normal/empty hard error均实际通过，
首次configure互斥失败与所有warning保留；同树j4完成，不跑依赖算子目录或重新计产品纠正。
Windows新过期deadline、重复stop/close/drain、原用户资源仍可用组合实际PASS，增量仅编译/链接
共同TU和测试（五个Ninja动作），无oneDNN/全库重编。Air共同十三项和当前四route增量正在补证，
三交付、新头/缺席/迁址与最终红队仍待验。

| FIX | 必须实现 | 验收 |
|---|---|---|
| F3-01 | 中央 compute_stream 身份/依赖/普通 route 及三交付投影 | 无环，无第二名单/空 target；静态共享source一致，不拉无关 CPU Provider |
| F3-02 | create/prepare/admit，File/Mapping owner、range/desc/group 与有限预算 | 错描述、空请求、EOF/offset溢出、OOM、source大于stage、共享backing去重 |
| F3-03 | positional/mapping读→stage→F2/native-ready；多chunk多group并可提前消费 | 短读与尾块有错误/实际字节，不能整 shard 临时buffer；ready不是只完成host读 |
| F3-04 | producer依赖→resource→stage/host→write_all_at反向旅程 | 未完成GPU写不能读取；short write/部分进度；原子发布交已有owner |
| F3-05 | F1 leases、逐组/整请求结果与旧结果容量、独立在途槽 | N=2组/K=2槽以上；早ready旧租约不复用，无隐藏全设备等待 |
| F3-06 | request cancel、stop/drain/close、abandon/late completion、partial acceptance | 原错误不吞；在途内核/IO不假取消；关闭本pipeline不关闭用户所有资源 |
| F3-07 | 原生四路线与实际math consumer、确定输入与资源成本/路径报告 | BF16/F32/至少一种已有block编码保持；copy后Direct+Graph真实输出对参考 |
| F3-08 | 成熟GGUF/Safetensors消费示例、指南、SDK迁址/新头/包闭包与P0 | 解析器在consumer；本地Qwen混合dtype/ranges；DLL同目录，普通入口可发现 |

## 首个不可变Candidate（已被下述红队纠正替代）

- 产品与普通consumer固定提交：`3b7b9b9b8f0e99205de0c43ce29b4467bdae46d5`；
  tree：`7ffc5f2cedc973dae3093fed36dde2d30ba8ca46`。此前f3d931c47到本提交的产品差异只为
  共同getter两处换行，全部非空白token相同；算法、公共API/layout和成本不变。
- 原deadline/重复close补证fixture：`332026e43164e285ea890e92487bcfb9844aebab`，不改变产品。
  本轮状态转换previous-ref为`05ecad186`，六owner从AUDITED进入CANDIDATE，不跳级。
- 四路线Direct+Graph实际Qwen/GGUF或原Apple编码source旅程已迁移到普通样例；
  Win CPU十二项、Air CPU十三项以及当前Apple六/Vulkan五/WSL CUDA六增量已实际运行。
  原三sanitizer十二项已封存，新增风险在相同树增量运行，完整收据随后落盘。
- 候选之后只允许证据/文档/测试维护；产品新反例须按协议退回AUDITED。
  installed static/shared、SDK迁址、新头/缺席与最终红队尚未完成，CANDIDATE不等于CLOSED。

[当前风险收据](../../../../evidence/correctness/foundation-f3-current-risks/receipt.json)签Air CPU十三项、
当前Air Apple六/Vulkan五、Windows Vulkan五、WSL CUDA六，以及同一新fixture的ASan/UBSan/TSan
各十三项实际PASS。过期deadline/重复close、真正POSIX short-write与跨线程导出drop在Air和WSL
直接运行；所有旧失败保留。当前微修没有依赖源码重编：WSL sanitizer各四个Ninja动作，
Windows CPU五个动作、Air native仅共同库/route链接增量；没有新产品纠正或全库测试。
当前实际平台风险已齐，new installed头、普通static/shared/迁址和终审仍未完成。

候选红队读取`stream_worker`/`kick_locked`发现一个尚未直接反证的原F3-06组合：K=1时，
首组错误把尚未启动suffix标Failed；后组已接纳async producer的Completion仍在request目录中，
可能形成state→Completion→retention→state环。新增CUDA原生两组fixture，让第一组真实readonly
File写错、第二组原GPU producer阻塞；drop控制后放行并核对实际uses/allocator是否退休。
本提交仅增加测试，产品候选不改；未运行前保持UNPROVEN，不把源码疑点写成已证明FAIL。

该反证随后在实际WSL CUDA上FAIL：首组真实readonly File写错后，request complete但第二组
producer仍在未启动range里；drop后放行原GPU并drain，5秒内active uses仍为1、F3 allocator不balanced，
隔离子进程在原StatsAllocator析构断言abort6。源码证据为`stream_worker`/`kick_locked`只标记queued
group Failed，不销毁其中Completion，形成state→Completion→retention→state环。
因此只有共同owner compute_stream从CANDIDATE退回AUDITED，previous-ref为`bf59b57f0`，
其余五owner仍固定候选；不关闭、不伪造产品通过，也不修改IO/Sched/native backend。
这是原F3-06已接纳producer的未完成组合，固定为一次共同控制交接纠正及定向验收；旧两次形状
纠正、producer修正、原FAIL/TIMEOUT和追加ASan预算全部继续计账，不重新开始研究或扩新FIX。
允许的纠正仅在退出共同锁后放弃未启动producer观察，使原native owner在实际退休时释放原retention。
不得在锁内触发回调/释放，不隐藏等待或新增poller，未退休之前uses/stage仍须保活。

定向纠正新增`abandon_failed_producers`：每次只在共同锁内将Failed组的一个Completion移出，
锁外析构交原native owner；worker失败与spawn失败后的read/write/collect入口都覆盖。
固定目录扫描上界O(QGR)，无新数组、分配、线程或常态字段；原producer退休前retention仍保留
state/stage/uses。当前为AUDITED实施，复验尚未运行，旧candidate3b的交付只作历史局部证据。

[suffix退休收据](../../../../evidence/correctness/foundation-f3-suffix-retirement/receipt.json)现绑定
产品`97858fbc88a6b5a87f109290b956eeaeee632dd2`与原FAIL：WSL CUDA整组七项、共同
ASan/UBSan/TSan各十三、Air CPU十三/Apple六/Vulkan五及Windows Vulkan五项复验全部PASS。
原失败用例不再残留active use或metadata；原GPU真正退休前仍保活，没有提前释放或隐藏等待。
Windows CUDA新suffix直接运行、当前source/安装数学及新头/迁址仍待验；不代签全平台终态。

纠正后不可变candidate现为`97858fbc88a6b5a87f109290b956eeaeee632dd2`，tree
`6398479c76fdc9677206ee692943664814463ebd`。previous-ref `0fc9ce59d`：只有共同owner
compute_stream从AUDITED回到CANDIDATE，其余五owner产品未变但绑定同一完整树。
旧3b候选/静态SDK结果保留作历史；当前产品有限改动为上述原F3-06保活交接及O(QGR)扫描，
产品/样例随后不再改。Windows CUDA新fixture与完整交付仍继续，未运行不写PASS。

[Windows suffix复验](../../../../evidence/correctness/foundation-f3-windows-suffix-retirement/receipt.json)
现签当前产品97858fbc8的七项CUDA公开API风险整组PASS，包括原失败的readonly首组/未启动
producer后组。在普通shipping CPU/CUDA source树中添加独立fixture，不切换旧SDK profile，
原oneDNN产物复用；实际active uses与F3 allocator在native退休后归零。
当前原生与共同sanitizer风险已齐；当前Vulkan static安装交付运行中，其余交付/新头/终审仍待验。

[当前Windows Vulkan static交付](../../../../evidence/correctness/foundation-f3-sdk-windows-vulkan-static-c87521395/receipt.json)
绑定修正后97858fbc8产品：旧producer树完成87项增量，真实install后rename、原prefix消失，
普通独立C++项目仅find_package Graph且选择vulkan；原Direct/Graph小GGUF与同Qwen完整往返PASS，
六个新/changed头独立编译、CPU/CUDA/Apple头缺席及三条原package诊断PASS，运行库同目录。
新配置未要求任何内部factor/provider。共享版本正在运行，不代签其他交付格。
[旧CPU/CUDA static交付](../../../../evidence/correctness/foundation-f3-sdk-windows-cpu-cuda-static-8857b4459/receipt.json)
也已封存全部原运行与新头/缺席/迁址PASS，但它绑定红队纠正之前的3b产品，只作历史局部事实；
当前产品沿原树做增量接线与相同consumer，不能拿旧数学输出代签新候选。

[Windows Vulkan shared交付](../../../../evidence/correctness/foundation-f3-sdk-windows-vulkan-shared-7834ff2f9/receipt.json)
进一步签同一97858fbc8产品、独立shared artifact树：100项构建后install/rename，普通消费者实际
DLL同目录运行小GGUF与Qwen往返PASS，六头独立编译及CPU/CUDA/Apple缺席诊断PASS。
Windows Vulkan当前static/shared已齐；WSL CPU/CUDA static正在运行，其他当前安装格仍待验。

[WSL CPU/CUDA static交付](../../../../evidence/correctness/foundation-f3-sdk-wsl-cpu-cuda-static-97858fbc8/receipt.json)
现签同97858fbc8完整源码：原树87项增量产物，真正迁址后独立普通Graph C++消费者原样配置，
小GGUF与Qwen完整往返PASS、`ldd`无missing、新七头独立编译及Apple/Vulkan头/package请求缺席PASS。
原锁定CUDA leaf链接warning保留；未改算法或SDK依赖锁。shared版本正在运行，其他当前格仍未验。

[WSL CPU/CUDA shared交付](../../../../evidence/correctness/foundation-f3-sdk-wsl-cpu-cuda-shared-97858fbc8/receipt.json)
同样完成当前候选的安装/真实迁址、普通C++小GGUF与Qwen往返、新七头独立编译和两条缺席路线
诊断。`ldd`无missing且所有HYLE共享库实际来自relocated/lib；没有通过源码/旧build fallback运行。
WSL CPU/CUDA两种installed linkage现已齐，Air CPU/Apple static开始运行，其余当前交付待验。

Air CPU/Apple的[static](../../../../evidence/correctness/foundation-f3-sdk-air-cpu-apple-static-97858fbc8/receipt.json)
与[shared](../../../../evidence/correctness/foundation-f3-sdk-air-cpu-apple-shared-97858fbc8/receipt.json)
现签Mac16,12/Clang21上当前97858fbc8产品：真实install/rename、普通Graph独立C++项目、
CPU原GGUF/Apple原affine Q4小旅程、两路线同Qwen实际权重往返均PASS，新七头独立编译及
CUDA/Vulkan头与package请求缺席PASS。原Metal runtime asset同消费者接线、otool依赖记录保留；
实际dyld路径补证待运行，不用otool名字代签加载位置。Air Vulkan static正在运行。

Air CPU/Apple shared收据现补入真正`DYLD_PRINT_LIBRARIES`运行：同一普通consumer实际加载的
25条HYLE dylib全部来自relocated/lib，数学再次PASS；原首份日志及Git证据身份保留。
[Air Vulkan static交付](../../../../evidence/correctness/foundation-f3-sdk-air-vulkan-static-97858fbc8/receipt.json)
也已完成当前产品的真实M4/MoltenVK数学与Qwen往返、迁址、新六头、CPU/CUDA/Apple头与路线
请求缺席、原Vulkan/IREE闭包和consumer运行。shared版本正在运行，actual dyld路径同时记录。

[Air Vulkan shared交付](../../../../evidence/correctness/foundation-f3-sdk-air-vulkan-shared-97858fbc8/receipt.json)
同样完成当前产品的真实GPU数学/Qwen、迁址、新头和三条缺席路线诊断。
两个普通consumer在`DYLD_PRINT_LIBRARIES`下实际运行，HYLE共享库全部来自relocated/lib；
不靠otool名字猜测加载来源。Air CPU/Apple与Vulkan四installed格已齐，Windows CPU/CUDA
当前static在运行；其shared、WSL软件Vulkan适用交付和最终矩阵/红队仍待验。

[当前Windows CPU/CUDA static交付](../../../../evidence/correctness/foundation-f3-sdk-windows-cpu-cuda-static-f6db4d5b2/receipt.json)
现签修正后97858fbc8：原树44项增量并直接运行source小GGUF/Qwen数学PASS，再真实安装迁址，
普通独立find_package消费者实际CPU/CUDA小旅程与Qwen往返、七头独立编译、Apple/Vulkan头
及package请求缺席、DLL同目录与实际imports均PASS。没有再次重编oneDNN或切换SDK profile。
Windows shared正在运行，WSL软件Vulkan两格及最终source复验/矩阵/红队仍待验。

[当前Windows CPU/CUDA shared交付](../../../../evidence/correctness/foundation-f3-sdk-windows-cpu-cuda-shared-a5215036e/receipt.json)
完成独立shared tree的62项产物构建、真实install/rename、普通find_package消费者、当前CPU/CUDA
小GGUF与Qwen完整往返、七头及Apple/Vulkan缺席诊断，DLL同目录/最小PATH运行和实际imports均PASS。
五个合格平台/路线组合的十个installed static/shared profile格现已齐；WSL Vulkan已重新枚举只有llvmpipe，
其GPU数学NOT_APPLICABLE，当前static适用构建/头/缺席/ldd门正在运行。最终source/矩阵/红队待齐。

[WSL Vulkan static适用交付](../../../../evidence/correctness/foundation-f3-sdk-wsl-vulkan-static-97858fbc8/receipt.json)
现绑定本轮实际loader枚举仅llvmpipe/type4、qualified_gpu=false；不以旧例外代签native GPU数学。
当前产品真实安装迁址、普通Graph消费者两个target构建、六头独立编译、CPU/CUDA/Apple缺席及
`ldd`无missing均PASS，GPU运行明确NOT_APPLICABLE。shared适用门正在同一候选的独立树运行。

[WSL Vulkan shared适用交付](../../../../evidence/correctness/foundation-f3-sdk-wsl-vulkan-shared-97858fbc8/receipt.json)
也完成真实安装迁址、两个普通consumer构建、六头/三条缺席及`ldd`无missing，HYLE共享库实际
来自relocated/lib；原device枚举的qualified_gpu=false与未运行GPU数学一并写入收据。
全部十二个本轮installed profile/linkage格的适用部分已取证，十个合格GPU/CPU组合实际数学运行。
当前Air CPU/Apple source数学及Qwen复验PASS，Air Vulkan source运行中；Win Vulkan与WSL source
当前产品复验、最终矩阵/证据审查待验，不扩大算子目录或P1。

## 解析器与真实权重验收

格式解析不进入 pipeline。GGUF 用现有 ggml 格式核；本地 Qwen3.5-2B 原始 Safetensors
用官方格式工具生成 checked range manifest/fixture，独立 C++ consumer 消费它。
不能用文件 Q 标签猜所有 tensor 的 encoding，也不新增 HYLE parser/Torch依赖。
如 consumer 需要 C++ Safetensors loader，应独立选成熟 adapter，不制造未定版的产品依赖。

本地文件位置由用户提供，取头部身份/原始字节范围和实际tensor metadata；
读取数据仅限所声明 FFN/规范化矩阵/小片段，不重复全文件散列/同步4.55GB。
已读真实第0层三投影 BF16，linear_attn norm F32[128]；MTP分支也存在，不表示本卡实现MTP模型。
稀有量化可用既有目录的固定小artifact/明确bytes fixture，数学仍用已接成熟Provider。
完整模型、tokenizer和生成loop未实现，不能称“全模型推理验收”。

## 预声明验证与停止

候选反证、原suffix FAIL/一次有限纠正及尚未完成交付边界见[红队记录](foundation-f3-final-review.md)。
当前overall仍INCONCLUSIVE，不能因已通过若干安装格把F3写成CLOSED。

原八FIX的逐能力风险、实际证据和剩余平台/交付格见[当前矩阵](foundation-f3-verification-matrix.md)。
它明确新deadline/重复close fixture尚未运行、各installed gate尚未取证，不把局部PASS聚合为关闭。

原固定容量、分配/锁/IO/native路径及实际字节成本已整理于[P0结构核对](foundation-f3-p0-cost-review.md)。
该页明确已知控制数组、K个原fiber stack、热Task/IO/lease控制与vendor未知成本的不同边界，
不代签尚未完成的交付/终审，也没有启动P1。

正常 read→math、resource→file→reload→math、early group ready/late fail，
source/sink寿命、EOF/short transfer、cancel、retained result、quota与未知private bytes，四路线及三交付。
建立一套host workflow/相同fixtures集中运行；复用F1/F2已签同身份Native证据，
此卡只新增file/group/close接缝，不重复算子正确性全目录或性能全集。

P0检查logical/native/staging/metadata/old leases账单、峰值inflight、分块上限、热分配和host waits。
真实overlap若报告，必须用原native copy+成熟math的进展证据；只有异步函数名称/两queue不算通过。
可重叠执行不要求可复现吞吐领先；无需空跑P1争夺赢家。
所有FIX与适用门完成后停止，不追加GDS、持久化cache服务、F4/F5或完整引擎。
