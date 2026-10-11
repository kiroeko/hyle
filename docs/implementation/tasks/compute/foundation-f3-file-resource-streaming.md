# F3：有界文件与计算资源双向管线

> 状态：AUDITED；六owner与原八FIX、API/生命周期及有限改动集已冻结，私有共同接缝已首验，完整管线待实现；F1/F2原轮已CLOSED。
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
| scope | 原八FIX、主/正交旅程、六owner及实际baseline | 已定界，尚未产品验收 |
| architecture | API/生命周期/依赖审计、现有原子组合与有限ledger | AUDITED形状冻结，产品接线仍UNPROVEN |
| correctness | 02至07逐能力风险矩阵及四路线实际接缝 | NOT_RUN |
| delivery | 01/08 source/static/shared、新头、迁址及缺席 | NOT_RUN |
| usability_docs | 普通C++自然工厂、双向与错误/close recipe | NOT_RUN |
| red_team | 候选冻结后独立一遍反证检查 | NOT_RUN |
| engineering_performance | P0成本结构、有限容量与实际路径账单 | NOT_RUN |

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
  不要求用户重复拼Scheduler身份或建新pool。create冷保留K个原Sched Task/Fiber及stage/F2 slot，
  通过原同步原语等待新工作；native阻塞wait用同BlockingPool offload，不阻塞Scheduler worker。
  TaskScope要求析构前显式join；不把它藏进非阻塞Pipeline析构，使用原self-owned task/JoinHandle保活。
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
  unknown native observation保留slot/quarantine、retirement_proven=false、无ready/复用/假取消；
  collect/drain只调用原退休入口，不新增定时poller或无限重试；原sticky CUDA进程终止合同继承。
- 冷metadata为O(requests×groups×ranges + K×F2容量)，admit检查范围/别名/去重O(R log R)，
  chunk数量O(ceil(bytes/stage))，每chunk使用原IO/复制/退休；报告原Task/IO/offload allocation和
  host waits，任何新增临时allocation走已注入且受有限预算的allocator，不以replay零分配代签整个旅程。
  同backing只计一次，external/managed/native private unknown分列，不把caller整个文件读入或全部CPU数据pin。

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

正常 read→math、resource→file→reload→math、early group ready/late fail，
source/sink寿命、EOF/short transfer、cancel、retained result、quota与未知private bytes，四路线及三交付。
建立一套host workflow/相同fixtures集中运行；复用F1/F2已签同身份Native证据，
此卡只新增file/group/close接缝，不重复算子正确性全目录或性能全集。

P0检查logical/native/staging/metadata/old leases账单、峰值inflight、分块上限、热分配和host waits。
真实overlap若报告，必须用原native copy+成熟math的进展证据；只有异步函数名称/两queue不算通过。
可重叠执行不要求可复现吞吐领先；无需空跑P1争夺赢家。
所有FIX与适用门完成后停止，不追加GDS、持久化cache服务、F4/F5或完整引擎。
