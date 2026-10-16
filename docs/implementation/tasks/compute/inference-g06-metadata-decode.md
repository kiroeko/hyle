# G06 metadata 与 decode 成本收口

状态：**G06 AUDITED；共同、CPU Q1与Apple/Vulkan runtime lengths source批通过，2026-10-16**。本卡沿[完整十项目标](../project/inference-gaps-goal-execution-2026-10.md)，
不缩减 CPU、单 GPU、CPU＋单 GPU、单/多请求或四原生路线。先完成四组共同审计和形状冻结，
再集中实施、批量检查、按实际失败定向修正；产品修改仅限下文已冻结且owner为AUDITED的批次。

## 初始 owner 与 baseline（后续批次逐段登记）

canonical baseline：`10129b90bbfc4d85c25e1595c823962b5a9c8146`；本轮前产品输入分别为：

| owner | 原产品 candidate | 当前阶段与边界 |
|---|---|---|
| compute_batch | `982676cb05a6e6244e66e0a596347aa96043617b` | 原 G04/G05 聚合 BLOCKED 后恢复到本卡 SCOPED；布局、上传与权限准入 |
| graph_batch | `982676cb05a6e6244e66e0a596347aa96043617b` | 同上；共同 metadata 与原 region 依赖 |
| CPU Provider/runtime、FlashInfer、Apple/Vulkan native controller 及三个 hybrid | 各 registry 的原 candidate/terminal | 本阶段只读，不自动转移施工权；找到具体必要修改后，在本卡固定范围并逐 owner 合法进入相应阶段 |

G04/G05 数学域与全部原失败、费用、未知、正确性/交付证据继续保留。
[CE28 成本检查](../../../../evidence/architecture/inference-g04-g05-cuda-ce28-producer-stage/consumer-cost-checkpoint.json)
已终止，没有新 FI material 双钥匙；原 CUDA prefill 非劣 UNRESOLVED 不冒充 PASS。
按总卡允许的真实依赖调整，G06 可先推进；共有费用降低不保证 CUDA 相对比率改善。

## 真实任务与有限范围

主消费者为 Qwen attention 的单请求与多请求 Q1 decode，以及已有混合 prefill/decode；
公开旅程为 `BatchSchema → BatchLayoutVersion → PagedArenaRead/原 pending carrier →
BatchInvocation/Graph coordinator → checked Completion → owning output → close/drain`。
请求选择、bucket、KV 管理与采样策略仍由引擎拥有。

正交消费者为两个真正 attention node 共享同一不可变布局、保留不同 native workspace 的
数值 DAG；改变请求顺序/加入/结束、shared physical pages 和旧结果存续反证特例化。

四组清单沿原总卡冻结，不新增第五组机制：

| ID | 已有缺口与承载裁决 | 实施前必须冻结的决定 |
|---|---|---|
| G06-M | 每版全容量分配/清零，每步全 metadata staging/upload。SELF，compute_batch | active snapshot 与原 fixed-capacity metadata 导出的关系；immutable generation、inactive 字节合同、changed ranges、失败保旧版、预算与线程安全。不能仅省 warm copy 而把全容量费用藏到每 token 的 version 创建 |
| G06-S | Graph 在逐 node 上传之后才共享 mirror。SELF，graph_batch/compute_batch，复用原 F2/G03 | 在首上传前按同 generation、schema、resource、queue/原依赖合格分组；各 node plan/workspace 独立，reader 活到真实退休；不以换队列后的旧 ready 推定新依赖 |
| G06-L | 每页普通 `LeaseUseBatch::read` 重扫已准入表，最坏 O(G²)。BORROW 原内部 `read_unique` | 按原 lease block identity/generation 去重，保 permission、poison、writer/reader 和 pending 顺序。`LeaseBackingInfo.identity` 是共享 backing，不能代替 lease identity；不得修改公共 F1 检查来签通过 |
| G06-D | 四路线实际 topology/runtime 与 Q1 planning/capture 成本。BORROW 数学、SELF 必要状态桥 | 核真正原生 plan key 和编译期长度，不能只放松 key 或因 PrefillPlan 名称判缺 decode；host-known 不暗中 D2H，device-produced 仍经 checked boundary；capture 单独取证 |

原 CUDA plan 只依赖 Q/KV indptr；page IDs/tail 不参与该计划是正确行为，应保留。
Apple/Vulkan 当前编译期切片保留布局 owner，不能在长度改变时复用含旧常量的 Program。
所有已关闭数学、F1/F2、scheduler 和 queue 继续复用，不新建 planner/线程池/通用 schema。

固定 peer 来源与精确文件 SHA 已在上述只读证据中重新核对：SGLang `7224fce1` 的
host-known plan/active buffers、vLLM `98f72111` 的 decode wrapper/tensor-core 默认、
llama.cpp `cbe28bb3` 的层外 attention input 创建与逐层消费。它们只证明成熟形状，
不代签 HYLE 的异步、capture 或速度资格，也不要求四路线使用相同 native 对象。

## 审计阶段与停止条件

当前 SCOPED 只做源代码、真实 consumer、有限 public 调用草图和成本反证；无产品编译/P1。
审计需一次性回答四组的 owner、依赖、公开对象/identity、ownership、失败/取消/关闭、
完成与并发、冷热分配/搬运/等待，再列唯一有限 KEEP/FIX/BORROW/REJECT 表进入 AUDITED。
普通 C++ 使用不要求用户组合 benchmark 私有 Provider；不保旧接口兼容。

调查、实现、验证各以 2h 为检查点，按用户常设授权有限追加且保原累计，不重置失败/未知。
到点必须记录已有结果、剩余工作与具体下一阶段；无新机制证据时停止该调查，不扩为
CUDA tile/阈值搜索。若真实依赖阻断某 owner，精确记录恢复条件，仍推进其余独立工作；
goal 不因单次构建慢或一个模块未决而停止。

## 预声明验收

P0 裁决 active/changed/uploaded bytes、上传次数、plan/replan、host waits、warm allocation、
页权限准入复杂度和 native workspace。验收用真实操作与原 report/allocator/native 观察，
不为某个 benchmark 行添加公共字段。P1 仅在 API/算法冻结后按实际必要成本问题定界，
不能把旧 G04 CUDA 合同通过视为 G06 完成，亦不为状态/说明变化重跑它。

- 保原 public single/multi、Direct/Graph、三联合和四 native 准入域；数学与 G03 证据仅按
  实际未变输入借用。空/容量/稳定 tail/跨页/请求重排/加入结束/同 shape 不同 layout 必测。
- 同 layout 两 attention node 检查一次共享上传、独立 workspace、队列依赖和真实退休；
  foreign schema/generation/resource、旧 output、关闭与未完成 ticket 不能错误复用。
- 完整 lease identity 重复/同 backing 不同 block/不同 generation、权限冲突、poison、
  producer 失败和 pending writer→reader 继续拒绝或传播原错误，不以去重绕过原 F1/F2。
- host-known 不发生隐式 D2H；device-produced 的未完成/失败事实不能发布 ready；
  OOM/fault injection 保旧版与已接受前缀，cancel/abandon/unknown 不假退休。
- 受影响 source/独立公开头/搬迁 static/shared/直接消费者与适用 ASan/UBSan/TSan；
  native 数学改变才扩相应设备风险，具体 target/命令在 AUDITED 时固定。Apple 只在实查
  身份的 MacBook Air 上，Windows/WSL/Air heavy 串行，最多 4 编译 worker。
- 使用 change-impact selector，保 selected/skipped 原因；不重写公共 checker、构建或性能核心。
  registry 每轮带实际 previous-ref；至少 docs/closure/diff 检查。未运行项不记 PASS。

七门、红队、LOC、完整剩余支持/负空间与最终性能方向在 immutable candidate 后实际裁决；
当前没有任何 G06 产品或运行 PASS，完成项仍为 G01/G02/G03。

## 共同基础批 AUDITED：M、S 与权限入口

本节基于 `04b52e09b909cb07f8bb240a599c354226cdea82` 的实际共同源码冻结；compute_batch、graph_batch 从 SCOPED 进入
AUDITED 后才实施。本批先落共同依赖，G06-D 与 native controller 的必要后续修改仍在
同一 G06 目标内，不能用共同批通过替代四路线低成本 decode/capture 裁决。

| FIX | 最终形状与成本 | 有限验收 |
|---|---|---|
| M1 active snapshot | `BatchSchema::version` 只分配 header＋实际四数组的紧凑 immutable backing。`host()` 返回其稳定 owned spans；每次成功 version 仍有独立 generation。固定容量 descriptor 是真实 State/Graph schema 需求，不删除；`metadata()` 在显式调用时才生成零填充全容量 blob，并缓存同一个 readonly lease | 新版本/旧快照/导出独立寿命、并发首次导出、空与 ragged、OOM 保旧、capacity 不扩大 active storage；原 F4 测试保持 |
| M2 delta upload | 内部五段只读视图映射到原固定 wire offsets。slot 冷分配并初始化零 staging/mirror；每次只比较旧/新有效段，对每段生成至多一个包含变化的 word range，缩小有效区的旧字节清零。完全同版本零上传/零上传等待；新 generation 如仅 header 改变也须上传对应字段 | 对独立 full snapshot oracle 核 grow/shrink/reorder/empty、同 shape 不同内容；旧镜像只在成功上传后更新身份。报告 actual bytes/waits；不谎报扫描量为 changed bytes |
| S1 prepare before sharing | 由原 Graph coordinator 按固定 binding 顺序找到同 metadata 输入的首节点；先准备首节点，其它节点通过内部 `prepare_shared` 消费它已 checked-retired 的同 generation/resource 镜像。各 native plan/workspace 独立；`run/enqueue` 仍核 exact metadata storage | 两节点首次一次上传、后续同版本零上传，各自必要 plan wait 仍计费；foreign/resource/旧 token/旧 output/失败前缀与跨 queue 原图依赖保持 |
| L1 common ready admission | 原 common slot 冷预留指针 scratch；收集 ordinary/Q、准确 K/V page guards 和 mirror 的完整 block identity/generation，排序去重后借原 `read_unique`，写权限仍走原检查。保留所有输入 owner 到 admission 完成，排序不保存跨 episode 裸指针 | 同 backing 不同 block 不合并；相同 block 多 subview 只准入一次；permission/poison/retirement/OOM 与页权限原测试保持。pending/native/Graph 组合的对应准入费用仍是本卡后续必验，L1 不代签整个 L |

本批无新 public 函数、Provider ABI、组件或依赖；只在既有 internal batch headers 增加
active segments/共享准备入口，替换旧“上传后共享”入口。`BatchInvocation` public
prepare/update/run/output/close 调用不变，但报告允许零上传/零上传等待。固定 blob 导出的
首次全容量 allocation/copy/zeroing 是显式费用，不发生于普通 version/Direct/Graph prepare。
immutable version 私有字段增加导出 cache 与 mutex；锁只用于显式首次/重复导出，不在
普通 active snapshot、warm upload 或 native run 中取得。导出失败不发布半成品，允许重试；
原 spans/source 在 version 期间不得并发修改，控制 slot 仍 externally serialized。

slot 另存“实际自身镜像版本”，不能以当前共享布局冒充自身镜像内容；切换 shared/own 时
据此产生正确 delta。source upload 已实际 wait 后才共享，跨 queue 不推定未完成 upload；
原图 math/producer/native retirement 依赖不改。旧读者和任意未退休前缀阻止覆写，失败
仍保留原 native owner/guard，无成功 suffix 掩盖错误。prepared ticket 保一次消费合同。

L1 借用的 private `read_unique` 最初由 F2 引入；本批扩展可信 batch 内部调用者须证明
全体只读 block 去重，函数原 domain lock、closed/permission/failure/refcount 检查完全
保留。不修改公共 F1、其实现或 writer admission，不以共享 backing/address/page ID 去重。
排序 scratch 冷 O(max guards) 指针空间，warm O(G log G) 比较＋O(G) 原权限操作，无 warm
分配。该局部用途不会把 general LeaseUseBatch 默认改成免检查入口。

本批明确未修改：CPU request PagedRead 的 retained page spans、FI q/kv indptr plan key、
Apple/Vulkan 的编译期长度；已检查它们确实属于 G06-D 的后续 owner 工作。不能只改
reuse key 使旧编译常量继续运行。四路线 D、pending L、native metadata banks 与三联合
验收仍列为未完成，不以此次有限基础批缩小最终成功条件。

共同 API/算法与本批消费者迁移集中完成后冻结一次候选，运行 selector 选中的两 factor
正确性/逐头和真实 CPU Direct＋两 node Graph；原 encoded/common、paged/ragged、
pending receiver/Graph 风险消费者按改变的共同机制取证。Windows cache
`.scratch/build-compute-v2-dev` 只增量构建相应目标、最多4worker；无 CUDA vendor 修改，
不重编 ggml CUDA。随后适用宿主三 sanitizer、搬迁 static/shared 和受影响 native/联合
输入逐配置串行；先复用未变数学/设备证据，仅在真实新增风险时补设备门。
M1/M2/S1/L1 的 P0 需独立数据/权限 oracle 与原 report/allocator/调用观测，源码说明本身
不签 PASS。当前全部这些新门均未运行；不得签 G06、G04/G05 完成或新的 P1 吞吐。

### 共同基础批集中实施（尚待验证）

M1/M2/S1/L1已按冻结形状接线，未改native数学、vendor、公共F1或构建系统。
F2只证明当次delta范围，不能用其output冒充全镜像覆盖；本实现待原上传实际成功退休后，
通过原take_mutable取回writer，以冷零初始化＋逐次delta证明完整内容，再按原F1 publish。
新增显式导出并发/OOM/独立固定blob oracle，原CPU64步复用补实际bytes/waits断言，
并新增真实CPU两个dependent attention node、反向布局输入顺序、同版本/新generation
三episode数值及一次共享上传断言。当前无运行PASS，owner仍AUDITED。
selector READY，选中两owner/公开头反向直接消费者及compute/graph交付，保130 skipped
reasons；先按卡执行Windows共同source与真实CPU消费，再串行适用风险/交付/native资格。

### Windows 共同 source 终态（096fb6b09）

[原始证据](../../../../evidence/correctness/inference-g06-common-source-windows-attempt-4/receipt.json)
实际PASS：compute_batch56、graph_batch6、原生CPU真实消费者24，共86项，零FAIL/SKIP。
新增双dependent attention node、反向NodeLayout输入顺序与三episode数值/一次共享上传
通过；独立完整wire oracle以64步覆盖空、增长、缩小、重排及inactive零清除；原64步
真实attention还核仅generation变化时至多8字节与同版本零上传/零准备等待。
active export并发、OOM重试、旧reader/F4及原CPU编码/producer/Direct/Graph/ragged
数学、预算/失败门一并通过；并非把CPU保留host布局的数学结果代签镜像正确性。
实际编译131.437s，source4外层176.062s含测试且与stage重叠；首次配置122.938s和
manifest0.187s另保，三个private setup失败/原观察缺口及raw全部封存，不记缺失成本为零。
仅增量构建三个目标的实际依赖闭包（包含旧SIMD生成依赖），没有CUDA编译/全局构建/P1。
owner仍AUDITED；本证据不签三san/SDK/header/native新资格、pending L或四路线D完成。
下步先补本G06依赖批后的native metadata/topology/runtime设计和实现，再对最终适用
不可变候选集中做风险、三交付与各直接消费者；不为每个helper重复整套验证。

## 原生与 pending 批 SCOPED

[本批源码反证](../../../../evidence/architecture/inference-g06-native-pending-intake/intake.json)
绑定baseline e533327918789edf60a474f4d6ad423ae57ae8e6。新增施工owner为原Apple route
（baseline3b51575320c3f823ad1a31d18e99b8c55523a117）及原CPU+Apple/CPU+Vulkan
Graph route（baseline11b2403e794b6eed2c89418d72b8c89aeaaac746）；三者合法CLOSED→SCOPED，
仅重开G06准备/metadata/准确页准入成本，不重做已关闭数学、原producer/退休机制。
原Vulkan route仍为旧CANDIDATE，进入本批AUDITED前不改其产品；CPU/FI/ggml runtime
继续只读，不由本批自动取得施工权。共同compute_batch/graph_batch维持AUDITED。

实际controller及两个hybrid traits都实例化同一MaterializedNativeState；每prepare原生
Program/Invocation重建、全容量索引快照与完整上传已由当前源码核实。原native程序的
常量只取Q/KV indptr及tail，page IDs经原device输入，给出了保完整key的有限复用入口。
范围仍为原G06四组；本批先固定lazy index导出、逐bank准确delta、原native对象复用和
pending/原生readonly页准入，不把长度变化假装静态复用。长度runtime、四路线decode
与实际capture裁决仍是原G06-D必做后续。主/正交消费者、预算、失败/关闭与预声明
source/三san/SDK/直接native及三联合验证沿本卡，新增风险见AUDITED有限表后才实施。

### 原生与 pending 批 AUDITED 有限表

本批沿上述源码身份定界；共同两 owner 保持 AUDITED，Apple/native 两 hybrid 由 SCOPED
进入 AUDITED，Vulkan route 从旧 CANDIDATE 合法回 AUDITED，原 candidate 见 intake。
只实现以下四处；CPU/FI/runtime 当前源码与产品状态不变，完整 G06-D 继续。

| FIX | 冻结形状与费用 | 不可破坏的验收 |
|---|---|---|
| M3 lazy native index | NativePagedBatchVersion 改为小型 intrusive immutable owner，保 program/layout；`indices()` 显式首次生成并缓存原 fixed-capacity zero-filled I32 blob。导出 lease 用独立 program/layout retainer，不回指含 cache 的版本，避免 owner cycle；并发导出用原样 cold mutex | 原导出字节/寿命/move/copy/空/预算/OOM；补创建版本不触发全容量 index 分配、并发首次导出与失败后重试 |
| M4 actual bank delta | PagedMetadataPolicy controller 直接读 version.layout 的有效 page IDs；两 device bank 与 staging 仅冷初始化为零，每 bank 保自己的准确 layout。相同 page IDs 复用已有 readonly bank；否则对实际目标 bank 计算一次 word range，清旧 suffix，只上传改变范围 | shrink/empty/page reorder/两 bank 切换、同 generation 与新 generation、真实 uploaded bytes/waits。F2 仅签 delta；成功实际退休后取原 writer，以冷初始化＋全部 delta 证明完整内容并发布，不伪造 full output coverage |
| D1 exact native reuse | 只在同 schema、Q/KV indptr 与 tail 完全相等时复用原 Program/Invocation；IDs 只影响 runtime table，不入编译 key。仍先原 reuse_gate 证明 output/worker/fence 释放，再过原 mirror writer 权限；长度变化仍冷重建，绝不复用旧常量 | 同内容新版本/仅ID重排无 native 重建；Q/tail/请求变化真实重建，原生错误/旧output/unknown/actual retirement继续阻止复用。NativePagedBatchReport 增加与 common 同名 `reused_native_plan`，其余 staging/device/backing字段仍是容量，不冒充本次搬运量 |
| L2 page-group admission | 复用原 ready/ordered read 原语；cold reserve 的 BatchPageReads 收集准确 ready lease 或已验证 CopyUpdateInput.writer，按完整 block/generation 排序去重，再执行原 domain/permission/failure/refcount 检查。共同 pending、native ready/pending 与同lane Graph page group 集中准入 | 同 backing 不同 block 不合并；同组 ready/ordered冲突拒绝；source dependency qualification、lane wait、covers/validate_page、首错误和聚合实际退休保持。无 warm allocation，O(G log G) 比较＋O(G) 原准入 |

L2 group 只借当前调用持有的 lease/update 指针，准入后立即清空；不缓存跨 generation 裸指针。
原 Graph/F1 的普通绑定已先准入，不复制它们的状态表。它们与 page group 的同一只读 block
可保留原来的平衡重复 guard，跨 lane 同样分别记录真实依赖；预留容量仍覆盖全部 guards。
这不扩大 mutable 权限：writer/ready 冲突仍由原检查拒绝，pending writer 只有通过本来已
验证的 original update、准确范围和 native ordering 才能调用 ordered_read_unique。
不得把页 ID、backing 或地址当权限 identity，也不声称全图所有 readonly guards 全局最少。

共享 controller 的 reference MaterializedMetadataPolicy 仍走原完整参考算法/metadata 合同；
新增 zero/delta 与 topology reuse 仅 PagedMetadataPolicy。native traits 与两个真正 hybrid
traits 都用原 CopyAccess::take_mutable 和原 invocation/reuse_gate，不引入第二 worker/queue。
程序复用若仍有 native 内部 reader 阻止 bank 覆写，原 try_write 必须拒绝；不得删除该门，
只有实测新事实才在本有限生命周期接缝修正。实际 Air 身份已读：Mac16,12、macOS27.0.1
26A434、arm64、Apple clang21.0.0、CMake4.3.4、Ninja1.13.2；执行验证前再核实际硬件。

预声明 source 先复用 Windows 三个既有 target 补 native index / group admission oracle 与
原 pending CPU风险；Apple/Vulkan各 native Direct与真实CPU+GPU Graph的64步、原错误/
alias/phase与旧结果条件串行验证。独立数学/页有效区 oracle、quant/精度门不放宽。
API和本批算法固定后一次候选集中跑适用san/header/static/shared及真实直接消费者；
数学/设备程序未变的原证据只在输入身份确实匹配时借用，不启动全库或未变CUDA P1。
该表不签长度runtime/Q1最优规划/capture的完成，也不将它们移出完整G06目标。

### 原生与 pending 批集中实施（等待验证）

M3/M4/D1/L2 已按有限表集中实现，六 owner 保持 AUDITED。Native version 保小型
immutable owner，显式导出缓存独立 retainer；四个实际 native/hybrid instantiator 共用
完整 topology key、逐 bank 有效 ID delta 与原 permission/retirement 门。原 reference
MaterializedMetadataPolicy 的完整准备算法保持。共同/原生/pending 与 Graph 同 lane 页组
使用同一 cold-reserved exact block/generation admission；普通 Graph 绑定原有 guard 不改。
测试补导出并发与失败重试、共享 backing 的不同 block、准确准入数量及 closed gate；
原 native 64 步拓扑变化各追加同布局 replay，并追加仅 ID 重排的独立 double 数值 oracle，
核真实上传与复用报告。真实联合 64 步核同表零上传，保原数学和生命周期门。
先冻结当前源快照，Windows 五个既有目标一次增量构建（最多4 worker），运行两 factor、
CPU实际消费者及 plain/encoded 两组原 pending 风险；再串行实际 Air native/联合。
本段仅记实施事实，当前新测试均未运行，不签 PASS、G06 完成或新性能资格。

### 原生/pending Windows source5 与 Air 首轮

源快照25fd231e4dd17af416f92321a2ea4766b5378da9的
[Windows source5证据](../../../../evidence/correctness/inference-g06-native-pending-windows-attempt-5/receipt.json)
为PASS：两factor60＋6、CPU实际消费者24，共90项；plain/encoded各9个原pending风险
全部通过。五个既有目标一次增量编译7.187s，外层52.047s含测试并与stage重叠。
原source4通过及setup1/2/3失败和费用不重置。未重复未变CUDA/P1或全库构建。
Air attempt1实际身份Mac16,12/M4/8核GPU/16GiB/macOS27.0.1/clang21；配置6.334s、
manifest0.076s成功，编译9.294s在CPU+Apple动态库链接时失败，测试尚未运行。
原因是本批首次跨route动态库使用原CopyStreamAccess::take_mutable，但原内部声明未导出。
有限修正只给Apple/Vulkan对应原声明加route export属性，不改变原退休/权限检查或算法；
使用同一source/build缓存续做原native/联合门，保留首轮失败和16.458s外层成本。

Air attempt2完成原Apple/Vulkan整批31个stage，外层108.541s（含stage且重叠），
终态FAIL。Apple Direct的64×2及ID重排、长页域、真实联合64步、reference、encoded
数值和原多数pending/生命周期门PASS；plain/encoded的cold OOM fixture仍用仅ID变化
期待旧cold allocation失败，pending双node fixture仍期待旧全量64字节，三项FAIL均保留。
Vulkan实际device allocation不可host-map，新冷清零在创建时失败，故其受影响native/
联合门FAIL；reference及不经过该controller的pending Graph风险PASS，不冒签整批。
有限修正：两device bank创建时均借原F2从冷零staging作完整初始化并实际wait，再取原
writer发布；显式冷费用是两次全容量copy＋完成等待，后续report仍只记本次prepare。
OOM改用真正改变tail编译key的版本，保原ResourceExhausted/旧generation可执行检查，
另核ID-only更新在禁止下一次controller allocation时成功且reuse；双node pending保
两真实paged/CPU fragment与独立数学、所有失败/旧结果门，改核实际delta上界和同表零上传。
不改数学/vendor/公共工具，六owner保持AUDITED；同一缓存串行复验本批native构造与消费者。

Air attempt3共31 stage、外层93.629s，Apple全部PASS；Vulkan Direct、plain/encoded原
风险、reference及真实pending联合64步等PASS，普通联合64步与encoded单/多请求联合
FAIL，统一错误为`invalid cold Vulkan composition output`。复用后的原Invocation已经
封存output bridge；原Graph却仍把每次prepare当新native owner再次取cold output/seal。
有限修正保留原CompositionAccess门：只在真正native替换后重建bridge；opt-in page Slot
持一位bridge-ready，在每个controller成功prepare且未复用时立即失效（含多node部分失败），
Vulkan桥全部成功seal后才置位。原old-result/retirement gate仍先执行，不重开原output。
另补真实原pending消费者要求的同topology prepare释放旧native输入import反证：实际结果
释放、prepare、移除caller frames后，准确旧页必须能沿原acquire/publish更新；不豁免F1
独占门。此项在attempt3未测，当前UNPROVEN，将随定向原native风险批核实。

Air attempt4定向两route各8个原联合/风险target，外层64.907s：普通联合64步、encoded
单/多请求、双node pending64与原联合错误/多source生命周期全部PASS。两route plain/
encoded各一项新增反证FAIL：同topology prepare仍有原Invocation input import持准确页
reader，阻止原acquire，不是权限门有误。有限修正只给可信内部PagedBatchAccess补
release_inputs，四traits在已通过reuse/output/退休门、原upload reset后调用；Apple清原
inputs/result/native临时arrays，Vulkan在原runtime mutex下把原input/alias/managed
ports置unbound，原root经已有old_inputs冷scratch移到锁外释放，保原计划与封存output。
Vulkan复用原Region::bind允许idle空port的合同；不改ggml/IREE/provider/vendor。
此私有操作无warm allocation，失败进入controller原failed状态；reference policy不调用。
两route实际Direct/联合/原pending与alias/phase/OOM/关闭门再次按相同缓存串行验证；
原90项Windows共同source产品输入未改，继续复用source5，不因native修正重跑CPU全套。

### 原生/pending source 批通过（42cae9837）

[Air source5终态](../../../../evidence/correctness/inference-g06-native-pending-air-attempt-5/receipt.json)
为PASS：实际M4 GPU及MoltenVK上的Apple/Vulkan共25次消费者可执行运行、6个配置/
manifest/build stage均通过。Direct每原64次拓扑变化增加同布局第二次replay及ID重排
独立double oracle；Apple12常规＋16长页域、Vulkan6常规、真实CPU/native/CPU联合
Apple12/Vulkan6组64步、encoded单/多请求和paired/separate生产者、原plain/encoded
六模式风险、pending Direct64/双node联合64和多source均通过。改后same-topology
释放旧import、禁止下一次allocation的ID-only复用、真正tail变化OOM保旧generation
也通过；数学精度、旧输出和actual retirement门未放宽。
此轮Apple增量build6.954s、Vulkan6.431s，外层96.970s包含并重叠所有stage；原四轮
失败与16.458/108.541/93.629/64.907s外层费用及raw完整封存，不把观察缺口记零。
Windows source5的90项＋18个pending场景按确实未变的共同产品输入和原artifact SHA
复用，匹配检查见receipt；不把CPU证据扩成native资格。owner仍AUDITED，无P1重跑。

M3/M4/D1/L2当前有限source批完成，未签完整G06：四路线runtime lengths、CPU/FI Q1
decode规划与实际capture裁决仍必做，最终适用三san/header/迁址static/shared及各真实
直接消费者待API/算法固定后集中验收。G04/G05原CUDA prefill非劣门与精确归因仍未解决。

## CPU runtime Q1 批：SCOPED / AUDITED

[定界源码与成本反证](../../../../evidence/architecture/inference-g06-cpu-runtime-q1-intake/intake.json)
绑定baseline ca9846561fae1641936bf88f9f30afb00c5f5be1，原CPU Provider candidate为
42a4c7cdcc0822a7b162efe575909622a906de1d。本批只使compute_provider_ggml从原CANDIDATE
合法回AUDITED并绑定本卡；compute_ggml_runtime仍保原candidate，只借现成原语，不改vendor。
真实阻碍是每次tail/页ID改变都会重建原请求/graph/pool；现有CPU native kernel已经有Q1
及长KV split档，planner workspace由D/heads/threads决定，当前Windows CPU-only及实际Air
两个mixed配置源码均核实不依赖logical rows/page refs，不因Prefill/Decode名称另造数学。

| FIX | 冻结形状与常态成本 | 原样保留的验收 |
|---|---|---|
| D2.1 CPU runtime页元数据 | 同schema/Q-indptr且每request为0或1 query时，先预检所有active request的原FLASH_ATTN_EXT节点；再更新其原I32 leaf指针/长度与n_kv_max，保原graph、pool与scratch。Slot持当前owned layout；不修改原PreparedBatch/API或原错误/退休门 | 单/多请求64步tail、跨页、长KV跨512原decode split、页重排/共享，原独立数值；无warm injected allocation，稳定Q拓扑workspace恒定；新Q拓扑仍真正cold replan |
| D2.2 immutable recipe寿命 | 每个原Request冷持首个compiled layout anchor和request index，防止Program.recipe的初始borrowed spans随Slot版本替换悬空；本次执行I32数据由当前Slot.layout保活。sizeof(Request)自动进入原metadata预算 | 保旧快照、两slot、OOM、无效版本、旧输出/关闭与原pending源、OOM不半提交。全体预检通过后才无失败地提交运行时指针 |
| D2.3 Q1零mask | 仅private paged且query_rows=1时省去causal全零mask；lower/prepare与预算一致。此时causal确实允许全部有效KV；window/boxed仍由原kernel计算，普通SDPA和prefill保原数学 | F16/BF16、plain/Q8、D64/128/256、D64/Dv128、window/noncausal与原普通attention回归；不放宽精度、不改native阈值/dispatch |

源码CPP/API、allocator、ownership、失败/并发和冷热边界已固定后实施；无需兼容旧私有形状。
主消费者为单/多请求Q1，正交消费者为现有真实两node attention及请求加入/结束/空request。
CPU-only及mixed core的原kernel/planner均复用，runtime/source preparer与构建系统不修改。
预声明Windows原缓存一次4worker构建Provider factor、原common实际消费以及两pending风险
target；factor排除已由common可执行完整覆盖的native suites，避免重复长数学。新独立Q1
oracle覆盖32组×64步，断言runtime复用/原始长KV档、无warm allocation与稳定scratch。
再串行Air实际CPU+Apple/CPU+Vulkan配置的同组CPU消费，保原lane/源错误与数值门。
最终适用san/header/迁址static/shared随完整G06不可变候选集中做。仍按2h检查点和用户
常设有限追加时间，原失败/未知不重置；不启动CUDA/P1，也不签其它G06-D完成。
ragged动态extent、Apple/Vulkan runtime lengths、FI Q1/capture及原G07–G10仍保完整目标。

### CPU runtime Q1 source 通过（bc3d8fadf）

[实际source证据](../../../../evidence/correctness/inference-g06-cpu-runtime-q1-source/receipt.json)
绑定bc3d8fadfb1c2139c3b7d12d3af00848fafa2bb3：Windows Provider普通回归62、真实
native/common/Graph25项及plain/encoded各9个原pending场景PASS；新32组×64步覆盖
F16/BF16、plain/Q8、D64/128/256和D64/Dv128、单/多请求、window/noncausal、tail
增长/缩短、跨页、共享/重排及512长度边界。稳定Q拓扑禁止下一次注入分配仍成功且
workspace不变；请求集合/空request拓扑变化实际重规划，旧owning output仍阻止更新。
原普通attention/混合prefill和ragged回归通过，不把Paged Q1 key推广到它们的动态extent。
原kernel的格式分档保留，F16 plain的长KV split与其它格式原路径不强求同一native档。

实际Air Mac16,12/M4下CPU+Apple、CPU+Vulkan构建配置各25项同组CPU测试全部PASS，
证明两个mixed core下的CPU运行时更新；没有用此测试冒充GPU数学或P1。Windows build
9.390s、外层55.157s；Air两build6.320/6.207s、外层87.837s包含并重叠各stage。
源码、命令、XML/raw、实际artifact SHA及原owner/baseline齐全；观察缺口仍UNKNOWN。
本轮没有产品失败/重试，没有runtime/vendor/构建工具修改，也没有CUDA/P1/全局构建。

CPU Provider保持AUDITED。剩余完整G06-D为ragged动态extent、Apple/Vulkan运行时长度、
FI Q1规划与实际capture资格；最终三san/header/迁址static/shared在完整算法固定后集中
验收。G04/G05 CUDA prefill未决与原失败不由本批代签，G07–G10仍保留。

## 原生 runtime lengths 批：SCOPED / AUDITED

[D3源码、owner与有限审计](../../../../evidence/architecture/inference-g06-native-runtime-lengths-intake/intake.json)
绑定2ed32d28520b89e26cdcb42f0c886df40ef9b99a。compute_provider_ggml_vulkan由原
CE15/16 CANDIDATE合法回AUDITED；原alias/phase范围、失败和全部资格不重置。其它
七个本卡owner已AUDITED。runtime、vendor、shader、数学和公共构建工具不修改。

| FIX | 冻结形状与生命周期 | 有限验收 |
|---|---|---|
| D3.1 原生共同key | 同schema、完整Q indptr及相同empty port域保Program/Invocation；KV indptr/tail由每Invocation更新，IDs沿原delta表。Q拓扑变化仍cold。原idle/output/permission门先于更新，接受后失败仍poison原slot | 两route固定混合Q拓扑64步KV增长/缩短、跨页、offset变化、重复/重排页、空request；same IDs零上传，独立旧slot输出不变 |
| D3.2 Apple runtime桥 | immutable tagged callable保初始layout与原cold-qualified compiled函数；仅paged派生Invocation存当前layout，普通Invocation无新增字段。初始长度走原compiled路径，其它长度用原纯paged builder及MLX fast原语，在原worker/取消/drain/completion内运行；不逐tail Context.compile | 原selected-stream拓扑由cold builder和固定Q证明；保CheckTrace，有device checks时用原reachable-status检查。两slot不共享可变长度；原pending/错误/旧output/OOM门保持 |
| D3.3 Vulkan runtime桥 | 私有host op_params slot11标request index；Region全量预检rows/offset/causal/table容量后，先使原NativeWork recording失效，再更新原参数。graph/gallocr/output storage保留；原admission mutex与checked退休不变 | 原provider及alias/phase source回归、完整Direct/联合/pending/encoded数学及退休；其它attention/ragged节点不打runtime tag |

Apple builder只组合选定stream的View/reshape/transpose/Concat和原native attention；没有
外来callback或数据相关checked索引进入可信桥。native内部allocation/图构造/driver费用仍
managed UNKNOWN，复用C++对象不能冒充capture或零分配。Vulkan长度变化需要原recording
重新编码，不称capture；host-known长度不暗中D2H。empty Q/无KV的一port与empty Q/有KV
的四port不能混用。共享Program保持immutable，每个Invocation独占当前layout owner。

两route及四trait消费者集中实施，冻结后一次Windows原共同source/pending与Air实际GPU
source；Vulkan额外原ce16_provider/ce16_public风险target。冷OOM反证改为真正Q拓扑变化，
不能继续把runtime tail变化错误当cold失败；原ID-only禁止下一次分配、旧页writer重取、
独立double oracle及精度不放宽。Win/Air heavy串行、max4，预算沿原2h检查点有限追加且
保累计失败/UNKNOWN。最终san/header/搬迁SDK在完整G06固定后集中做；当前无本批运行
PASS，ragged/FI Q1/capture及原完整十项不删减，不重跑未变CUDA/P1。

### 原生 runtime lengths source 通过（f53f88186）

[D3实际source证据](../../../../evidence/correctness/inference-g06-native-runtime-lengths-source/receipt.json)
绑定f53f88186e73eb81b9f6eee8bd315ea4a59c7dbc：Windows共同/Graph/CPU91项及原pending
18场景PASS；实际Air Apple/Vulkan共27次消费者执行、6个配置/manifest/build stage均PASS。
Apple12与Vulkan6常规域各新增64步固定mixed Q拓扑、KV增长/缩短/跨页/offset变化、重复/
重排页的独立double oracle；同IDs零上传/等待、保旧slot输出和原精度全部通过。Apple16
长页域保原64拓扑变化/第二次replay覆盖，不重复新增64步runtime序列。

原CPU/native/CPU64、encoded单/多请求与paired/separate、pending Direct/双node Graph64、
late/multi-source/失败/取消/abandon与准确页退休继续通过。真正Q拓扑变化的冷OOM保旧
generation、ID-only禁止下一次注入分配、旧页writer重取均保留。Vulkan额外provider10及
public alias/phase2项通过。原数学、vendor/runtime/shader、公共工具没有修改。

Windows build2.484s/外层47.516s；Air两build13.426/13.242s/外层121.073s，外层与stage
重叠。Windows启动器错误SHA在构建前被拒绝，Air过早启动被Windows尚未完成的guard拒绝；
两次setup失败及未观察费用UNKNOWN保留。Air attempt1无产品失败/重试，原各批失败不抹去。

D3有限source通过，不是完整G06或P1/capture。Apple变长度仍managed建图，Vulkan变参数
仍原重新编码；native/driver私有分配和费用仍UNKNOWN。owner继续AUDITED，下一批为原
ragged动态extent与FI Q1/capture定界；完整候选后再集中最终san/header/迁址static/shared。
G04/G05原CUDA非劣UNRESOLVED与全部失败/费用仍保留，完成项仍3/10。

## CPU ragged runtime Q1 批：SCOPED / AUDITED

[D4有限源码审计](../../../../evidence/architecture/inference-g06-cpu-ragged-runtime-intake/intake.json)
绑定1b68af80ba22307baf974cff1f659070df8b44b0，owner compute_provider_ggml已AUDITED。
主旅程为单/多请求连续Q1、packed KV增长/缩短及后续request起点移动；反偏置为请求加入/
结束/空中间request和独立旧slot结果。公共BatchProgram/Invocation与原queue/F1/F2不改。

| FIX | 冻结形状与成本 | 验收 |
|---|---|---|
| D4.1 Q1 ragged extents | 同schema/Q indptr、每request为0或1 query，逐request预检唯一原FA/Q1/无mask、直接KV root及其原permute view。全体通过后更新root/view的KV extent、stride、n_kv_max及Request输入desc/byte extent/packed offset；原graph/pool/scratch保留。各Request初始recipe anchor保持，Slot保当前owned layout | 32组单/多、F16/BF16、plain/Q8、D64/128/256、D64/Dv128、window/noncausal，各64步增长/缩短/512边界、packed起点移动；原numeric、下一次注入分配禁止、workspace恒定、旧output拒绝 |
| D4.2 原预算一致性 | private ragged Q1的零causal mask已由原lower省略，冷lower-bound预算同步省略；普通SDPA和非Q1不变。TensorDesc/required_bytes在全体预检阶段计算，提交阶段不引入可失败分配 | 原冷OOM、预算、接受前缀失败、取消/关闭/reader生命周期；普通attention与mixed prefill回归 |

本批采用原private batch直接调用的ProviderPreparedVTable；该入口仅检查port数，实际
storage范围由每次新TensorView及原map检查。Program.recipe保持初始immutable资格，Q及
output形状不变，runtime只改每Invocation借入KV root/view。原CpuExecution plan workspace
已由D2确认只依赖D/heads/threads，不依赖KV rows；不新写kernel或更改runtime/vendor。
control仍外部串行，原update gate先于mutation，未完成native work或旧结果不能更新。

集中实现后冻结候选，用原Windows CPU Provider普通factor＋真实native/common/Graph及
两pending风险组，再串行Air两个mixed配置中的同组CPU消费；max4，不构建GPU vendor。
最终san/header/static/shared沿完整G06集中验证。每2h检查点按用户授权有限追加，原失败/
UNKNOWN保留。此批只完成CPU ragged Q1；非Q1继续原冷规划，Apple/Vulkan ragged runtime、
FI Q1/capture和完整终验仍为原目标必做，不把本批当整个ragged或G06完成。
