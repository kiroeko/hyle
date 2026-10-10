# H2：CPU+MLX/Metal粗片段与offload基础机制

<!-- MODULE REVIEW: CANDIDATE -->

- baseline：`7f428d7a9`（H1 Windows/WSL完整合同CLOSED）；新factor身份登记`b842269f`。
- canonical：Windows；唯一Apple验收机MacBook Air `Mac16,12`，2026-10-10实读M4/16GiB、
  macOS27.0.1 build26A434、AppleClang21.0.0、CMake4.3.4、Ninja1.13.2。
- 当前owner：compute_route_apple、graph_route_apple、新graph_route_cpu_apple。
- reopen trigger：用户准入CPU+Apple联合数学、异步/重叠/专家片段及H1补齐的普通主存/预算/诊断。
  现单Apple执行只有原native数组/region，不能自然承担这些组合旅程；不以旧CLOSED拒绝新需求。
- 只借用现CPU Compute/Graph、ResourceBudget、Mem、Results和必要Sched；它们无具体缺陷不重开。
  需要修改额外owner时先记录具名源码反例并依法绑定，不在本卡顺手扩审算子或公共仪器。

## 已核实的缺口与承载

1. [Apple入口](../../../../include/hyle/compute/apple.h)的bind只接受成功ready的MLX数组；
   Context已有自己的提交线程及GPU stream，compile会追踪/预热MLX函数，不能在该追踪函数中
   执行CPU数学后假装CPU片段会在每次replay运行。
2. [原生计划](../../../../include/hyle/compute/detail/apple_plan.h)仅有MLX/ggml region；
   [执行器](../../../../src/compute_route_apple/apple_context.cc)在region边界检查同步。借用该原生
   owner/成熟编译region，不复制CUDA Queue、开启第二后台runtime或加入逐kernel调度。
3. [现Metal桥](../../../../src/compute_route_apple/apple_recurrent.cc)使用真正MLX allocator Buffer、
   shared views和MTLBuffer import；普通CPU resource不是MLX array，统一物理内存不取消owner
   或可见性合同。复用这条原生边界，必要copy/等待明确报告。
4. 当前[Apple Graph报告](../../../../include/hyle/graph/apple.h)只给logical constant bytes及未知
   私有存储，尚不能把CPU/全部槽/迁移和实际alias计入组合预算；复用H1 ResourceBudget和报告原则，
   不把MLX全局allocator/driver成本记零或当可预留物理内存。

## 有限审计与拟定FIX

本卡现已完成AUDITED；下节固定实际native承载、API职责与成本后开始产品实施。
调查上限：上述已具名接缝一遍、锁定MLX的buffer/stream/Metal completion源码一遍，至多一个
有界native buffer/copy/可见性实验。不能无限重选整体架构或重审模型算子目录。

| ID | 本包需要完成的机制 | 验收 |
|---|---|---|
| H2-01 | 自然CPU+Apple typed Graph入口、显式CPU/Apple粗片段及端口描述，沿现冷规则并保留完整MLX/ggml region；单Apple与CPU入口保持 | 两条真实多数学纵切、普通入口及原guard，不用mock/copy代签 |
| H2-02 | 普通CPU主存与有owner的MLX/Metal边界，必要有界copy/转换/可见性；CPU大权重不全量复制进GPU暂存 | CPU权重大于boundary容量仍真实计算；dtype/encoding/layout/alias/范围/OOM |
| H2-03 | 引擎给IDs/范围的固定容量两槽迁移、原始编码保留、ready/依赖及旧引用阻止覆盖 | IDs/目录/容量/overflow、两槽更新、独立迁移与计算及native实证 |
| H2-04 | 复用已有Apple原生owner及必要Sched进展，异步结果/错误/取消/close/abandonment保活 | 接纳前拒绝、接纳后排空、旧lease、失败不重算CPU，无第二runtime |
| H2-05 | 共享CPU/native/boundary requested-byte预算与冷诊断，alias去重、未知私有成本严格拒绝或显式partial；bind/reset准备失效 | 正确账单、共享/所有槽/旧lease占用、回滚、host-wait策略与实际执行一致 |
| H2-06 | 普通C++ Direct迁移与Graph示例/指南，source及installed static/shared集中验收 | 真机数学/资源、changed headers、native资产/搬迁、缺路线诊断和普通CPU/Apple回归 |

## 纵切与边界

主纵切：CPU-required normalization/projection等现成熟数学→完整Apple FFN/合法native region→
CPU消费，明确放置、CPU-resident权重及固定boundary容量。正交纵切：CPU分块变换→Apple矩阵/
归一化→CPU结果，双槽/旧结果/取消。具体算子仅从已有合格域挑选，不新增数学实现。
专家选择/缓存/预测、模型loader、KV、offload比例、GPU层数及请求循环属于引擎；多GPU不做。

## 预声明验收与停止

- API/算法稳定后一次冻结候选，批量native正确性/失败/生命周期/资源与三交付；只修实际失败半径。
- 正常/空/坏ID/shape/encoding/layout/范围/overflow、有限metadata与payload OOM；至少两槽、
  更新expert/payload不重选Provider、旧lease、accepted错误、取消、close、丢弃owner与晚到完成。
- 原生copy/compute进展与统一内存可见性实际取证；不以两个pending或host后台线程推断物理重叠。
- MacBook Air重型run串行、build1/test1；16GiB机器启动前读内存余量、监控自己的验证进程，
  使用已锁定依赖/对象，传完整可追溯候选并保持未变文件timestamp，不反向覆盖Windows。
- 本包P0，成熟数学/规则不比赛；若有具体完整成本反例才另定适用有预算P1，不制造性能仪器课题。
- 当前新能力UNPROVEN，H1证据只证明已完成CPU+CUDA及可复用下层，不代签Apple组合。
- docs/state轮仅docs/closure/diff；Queued身份config11项已通过，selector中央manifest给宽superset，
  该identity未添加target/dependency/public header，未重编/重跑全库或P1。不能把scope登记算新能力完成。

完成固定FIX及适用验收才CLOSED；超范围或native承载实质缺失要列明确问题与恢复条件，不能无限调查。

## AUDITED：有界native审计完成与FIX冻结

[机器身份](../../../../evidence/architecture/compute-entry-design/h2-native-preflight/machine.log)、
[锁定MLX源码](../../../../evidence/architecture/compute-entry-design/h2-native-preflight/mlx-source.log)和
[唯一buffer probe](../../../../evidence/architecture/compute-entry-design/h2-native-preflight/buffer-probe.json)
已经实际取证：MLX Metal allocator使用shared/untracked buffer；4KiB真正Metal blit、shared event、
CPU可见性与MLX alias保活通过，最后array deleter只执行一次。probe不签数学、H2完成或重叠。
它确认可借用原native allocation/event，GPU和CPU对shared存储仍必须正确排序。

### 固定实施形状

- H2-01：新graph::cpu_apple组合入口拥有冷partition和原value-ID映射；CPU连续片段由现CPU
  Graph编译，GPU连续片段由现Apple Graph编译，保留MLX/ggml成熟完整region。NodePlacement
  表达CPU-required/Apple位置，真实workload与全部port描述继续传入原规则；CPU/GPU不隐式互换。
- H2-02：Apple路线提供真正MLX-backed shared MemoryResource和有owner的array视图，复用Mem
  native handle/map/Buffer与H1配额；普通CPU内部存储仍system/host resource，只有边界分配native
  backing并做必要copy/repack。resource、外部payload借用规则显式，不造万能执行器/Queue。
- H2-03：Apple Compute拥有固定两槽WeightSlots/WeightBatch，IDs/TensorDesc/范围先硬校验，
  cold native staging/destination backing、实际Metal blit/shared event与native Completion保活；
  array只在合法ready后暴露，旧batch/Graph绑定/结果阻止槽复用，不转换原编码或管理专家策略。
- H2-04：组合Invocation借现Sched驱动CPU粗片段及原Apple Invocation；GPU仍在现Apple owner
  thread/native stream执行。现Completion poll与fiber-aware timed Event处理host进展，不增加线程池；
  native传输可独立提交并与GPU计算重叠，最终取证须actualGPU时间线，不靠poll命名推断。
- H2-05：已管理CPU Graph存储、native边界/两迁移槽、metadata及old lease共享ResourceBudget；
  MLX region的private allocation/cache仍未知，strict scope拒绝或用户显式partial。冷报告分别给
  known requested bytes、actual canonical array backing观察值、private upper/unknown与driver排除，
  不能直接重复相加。bind/reset使准备失效，host-wait拒绝覆盖native依赖和CPU边界。
- H2-06：普通Direct权重迁移与Graph两纵切、所有失败/寿命及Air source/shared/static集中验收；
  公开头和metallib/动态库/搬迁/缺路线同包完成，P0仍按上节固定成本边界，不填新数学。

普通公开入口分别位于hyle/compute/apple_weights.h和hyle/graph/cpu_apple.h，只暴露本路线native
MLX/Metal语义与CPU上下文。Direct链接compute，组合链接graph；单Apple路线不取得CPU/Sched。
compiled计划只读共享，每Invocation独立可变槽；所有输出以成功OwnedResult/lease发布，失败保活
并排空已接纳前缀，close只关本owner，外部Context/Queue/Scheduler及payload寿命由调用者保证。

冷成本：partition/shape校验与程序绑定，固定目录/两槽/边界backing、metadata与CPU workspace。
重放只沿固定粗片段与必要copy/等待；MLX managed replay允许其原allocator成本并报告未知，不称
recorded零分配。迁移按selected bytes及实际native DMA；不变payload不重选Provider。
三项owner已绑定，以上六FIX为本轮有限产品改动集；额外source/ABI/能力反例仍须重新定界。

### 首批产品接线

compute_route_apple新增真正MLX-backed shared Metal MemoryResource及apple_memory.h；通过现Mem
资源/Buffer/map接普通host-accessible视图，native array的deleter保留Buffer与配额到最后alias，
payload归还后才释放Reservation。控制块复用Mem StdAllocator及ResourceBudget metadata allocator，
不是mock resource、普通host地址冒充MTLBuffer或另一套allocator。未设置MLX全局限额/编译器模式。

中央public-header归属与Metal framework实际依赖已连接，owner为本卡AUDITED范围。11项路线配置、
closure/docs及selector授权通过；公开新头/Objective-C++尚待集中native编译/运行，不能写功能PASS。
剩余两槽迁移、Graph组合、报告与整包验收继续批量接线，不因本小片重跑旧H1或全库。

### 两槽原生迁移接线（待native资格）

新增apple_weights.h/apple_weights.mm：cold固定catalog与两组shared staging/destination backing，
shared ResourceBudget在分配前计入两端payload和受控metadata；upload硬校验ID、完整TensorDesc、
host可访问源、重复选择、容量与generation，再提交实际Metal blit和shared event。macOS blit的
四字节extent限制由已计账native尾部padding承担，公开tensor保留原精确编码，不进行量化转换。

每槽拥有原生transfer queue，可与现Apple Context stream独立提交；这尚不证明物理重叠。
binding只允许成功ready的array，array alias保留选中槽，batch/Completion/array存活禁止槽复用。
原生完成callback只保留backing，避免slot→command→slot循环；丢弃Completion的附加寿命在真实
native终结后回调，completed handler内不调用waitUntilCompleted，避免等待自身。close只排空本池。

当前为批量实施源码，不宣称原生编译、数学、资源或三交付PASS。Graph组合和整包固定候选仍待完成，
按原六FIX继续；本小片只运行工具/文档合同检查，不重跑H1或旧四路线性能。

完成回调在commit前注册，使用独立progress和backing；晚丢弃的Completion进入有界寿命链，在已注册
回调真实终结时释放，不尝试commit后追加Metal回调。MLX数组描述符的私有分配保留unknown owner；
显式partial预算才允许创建这些array，strict总账在native array分配前拒绝。已修正MemoryResource
set_data实参求值顺序，先读取handle再移动owner，避免由编译器实参顺序造成空指针。
路线配置11/11、closure59/schema4、docs898/4413和selector授权通过。native产品验收仍未运行。

### CPU/Apple粗片段产品接线（待native验收）

新增graph_route_cpu_apple真实target、cpu_apple.h及cold compiler/独立Invocation。现typed DAG
按显式NodePlacement分连续CPU/Apple片段并反向剔除dead pure节点；真实value-ID映射、全部端口
与workload进入原CPU Graph/Apple Graph规则，完整四节点affine-Q4 FFN留在一个原MLX region。
CPU片段仍由原Provider执行；边界布局归一化借用原CPU Copy和MLX contiguous lowering。
不新增数学、万能Queue、私有线程池或runtime；现Sched coarse fiber等待原Apple Completion，
CPU Queue检查exact Context，bind/reset使prepare与冷报告失效，旧结果/已接纳prefix保持owner。

普通CPU-resident输入不分配native backing；只给真实crossing value分配受限boundary。普通host存储、
CPU Graph workspace、metadata、原生边界和迁移共享ResourceBudget；MLX descriptor/region私有成本
仍unknown，严格总账拒绝，显式partial给known requested与canonical MTLBuffer去重观察，两者不能
相加冒充总显存。materialized constant拥有host bytes，Apple constants另有native owner；源可以
在compile返回后释放。native输入/动态weight以原编码绑定，cold prepare明确等待ready，不热搜索。

新增Direct weight sample及两条真实混合数学sample，source/static/shared使用相同公开入口。
本机路线配置原11项通过；新CPU+Apple/单Apple隔离项最初查出public/type/private声明漏项，已修
实际依赖边并仅定向重跑该项通过。当前所有数学/异常/重叠/三交付仍待Mac资格，不能以config通过
代签PASS。下一批先做固定changed TU/headers语法检查，再集中固定候选source与installed交付。

[固定Apple语法批次](../../../../evidence/correctness/compute-h2-apple-wiring-syntax/README.md)：
7个产品/consumer TU及4个公开头实际编译；初始两个类型错误修复后只补查这两个，最终11项均有
通过语法证据。仍不签link/数学/生命周期/重叠/三交付。6c415ae96为当前实施快照，owner仍AUDITED。
统一统计Compute+Graph 66,151实现行/289文件；全库自实现161,378行，不含测试。

[Air源码批次](../../../../evidence/correctness/compute-h2-apple-source/README.md)实际完成：普通CPU/Apple
Compute/Graph及新Direct/混合Graph四消费者4/4；新增合同target定向1/1。GGUF逻辑轴的is_dense
不等同native byte carrier的物理连续性，已保留原精确编码，provider继续原cold hard admission。
安装static/shared、搬迁/头/缺路线、物理重叠与终审仍待完成；不把source通过写成整包CLOSED。

1989b1fe3编码修复后统一脚本统计Compute+Graph 66,150实现行/289文件，全库自实现161,377行；
对应[当前LOC快照](../../../../evidence/architecture/compute-entry-design/loc-h2-source-qualified.json)。
本包仍未完成installed交付和native重叠，goal保持active，后续继续同一范围。

## 固定candidate与剩余资格

candidate：1989b1fe35b850d6dcc2da9d92e0b5056148a927。三owner由AUDITED进入CANDIDATE；六FIX的
API/算法、普通入口和消费方均已接线，原生source数学及异常/寿命/资源已验。上文各时间点UNPROVEN
仍为历史事实，不能据此重做已完成source阶段。产品变化须先依法退回AUDITED并列实际反例。

剩余固定批次：复用已锁定Air source与shared/static对象，仅增量构建/安装真实CPU+Apple SDK；
三普通旅程（Direct权重、混合Graph、合同consumer）及原入口共五项按linkage集中运行，独立安装头、
真正rename搬迁、原生dylib/metallib和缺CPU/Apple路线诊断同批核实。原生copy/compute重叠另一次
有界actual timeline取证；本包P0不启动数学性能比赛。终审核对cost、borrow、失败排空及全部gate。
文档/状态提交不更新产品candidate、不重建或重跑已匹配的source/P1。当前未完成项不能写PASS。

## 终审有限成本反例与回退

已核实drive_graph约223行按run.values.size扫描全部编译value，只为找Graph输入；完整provider
region已绑定仍需每次触碰全部内部value metadata，与重放只沿粗片段和必要边界的成本合同不符。
不是数学或benchmark输赢。只将graph_route_cpu_apple退回AUDITED：新增冷input ordinal目录，hot
仅处理固定的活跃输入。API、provider程序、布局/转换、资源及完成语义不变，不重开其它模块。

停止集就是这三个私有文件；冷metadata配额继续计账。固定新candidate后，仅对source/shared/static
混合Graph与合同consumer定向验证；原Direct权重、CPU/Apple原入口与公共头字节未变，复用已验
资格，不为这处成本修复重跑全集。SDK1989批次是原candidate真实证据，不据此伪签新candidate。

冷input目录修复仅九行新增/两行替换，固定新candidate37744b0ea。其余两owner源码/公开头完全
未变，与原candidate域匹配；统一组装记录指向新candidate但不重置资格预算。1989 shared/static
各5项消费者/4个安装头实际通过，旧install prefix已真正rename；剩余只补受影响的混合/合同，
不改接口、不测新数学/P1。任何新source反例仍须退回owner并固定有限修复。
