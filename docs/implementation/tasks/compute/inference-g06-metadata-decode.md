# G06 metadata 与 decode 成本收口

状态：**共同基础批 AUDITED，2026-10-16**。本卡沿[完整十项目标](../project/inference-gaps-goal-execution-2026-10.md)，
不缩减 CPU、单 GPU、CPU＋单 GPU、单/多请求或四原生路线。先完成四组共同审计和形状冻结，
再集中实施、批量检查、按实际失败定向修正；仅下文已冻结的共同基础批允许产品修改。

## 当前 owner 与 baseline

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
