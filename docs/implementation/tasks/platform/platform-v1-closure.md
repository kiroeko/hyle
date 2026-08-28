# platform-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`SCOPED`。本文是 platform 当前轮唯一 closure card 与施工记录；既有 platform task、
> architecture、guide 和性能页只作审计输入，不继承旧 `SIGNED`、C ABI 或 SOTA 结论。权威机械状态见
> [`module-status.json`](../../module-status.json)。

## Closure card（`SCOPED`）

- component / owner / round ID：`platform` / platform owner / `platform-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`780bf5b69249cf17df0344b1e40ceb7cf8797c6a` /
  `4f77224063ec669b16caeb844b7079f55ab630ec`
- 进入原因与 demand/defect ID：`PLATFORM-CAP-01`（按基础库完整性标准重新枚举 OS/CPU 能力族）、
  `PLATFORM-NORTHSTAR-01`（量化模型加载、CPU dispatch、worker placement/idle 与 NUMA backing）、
  `PLATFORM-API-01`（C++23 owner/view、错误、默认值和可发现性）、`PLATFORM-ADOPT-01`（下游是否绕过
  platform 或误用 error/time）、`PLATFORM-PERF-01`（旧微基准未通过当前消费者与仪器资格）、
  `PLATFORM-DOC-01`（current 文档仍标 `NOT_REVIEWED_V2` 且夹有旧 C ABI/历史证书叙事）、
  `PLATFORM-DELIVERY-01`（当前中央组件图和 source/static/shared 交付复签）。
- 真实 consumer task / normal workload：
  - scheduler/condition-variable/GPU poller 的长寿命 worker：进程可用 CPU 查询、线程创建配置、
    token 不丢失的 park/unpark、有限 deadline park 和 shutdown/join；
  - 本地量化推理模型加载：只读持久 mapping、显式 prefetch/advice、随机页访问与正常 unmap；小型配置/
    pipeline cache 走 `FileHandle` 短读写；
  - mem 的大块 VM、guarded stack、huge-page/NUMA backing；simd/container 的一次 CPU/ISA/cache 事实读取；
  - 工程计算留出：大数据集持久映射和固定 worker 集合，不引入模型/token 专用策略。
- 范围：`include/hyle/platform/**`、`src/platform/**` 的 CPU/cache/topology、thread/affinity、address wait/
  Parker、anonymous VM、NUMA、filesystem/environment/file handle、mapped file、dynamic loader；全部 public
  surface 反向追踪；直接依赖和代表性直接消费者；platform-only C++ source/static/shared；当前
  architecture/user/performance 文档；统一仪器所需的窄 platform consumer contract。
- 非目标：scheduler/thread-pool/executor 策略、fiber/coroutine wait、模型/KV placement、async I/O、
  filesystem watcher/VFS/glob/path-string 算法、device/registered/pinned memory、跨进程 shared-memory 协议、
  Native FFI/C API、稳定 C++ ABI、Formal release certification，以及为了 benchmark 点值加入常态 metadata、
  cache、padding 或公共入口。
- 直接依赖 / 直接消费者：`PUBLIC core error time`；代表性直接消费者为 `mem`、`simd`、`container`、
  `sched`、`compute`，以及通过这些组件形成的 `io` 路径。本轮只验证直接 platform seam，不替未收口
  的下游签署整体结论。
- tier 与理由：`P1`；`Parker` 的 park/unpark/token path 位于 scheduler 同步与 worker idle 路径，持久
  mapping 是模型加载与随机访问的正常运行时机制。CPU/NUMA/VM/thread/fs/env/dynlib 的初始化、资源创建
  或 syscall 主导路径分别按 P0 结构/复杂度门审查，不能池化成一个“platform 总分”。
- material margin / resource budget：W1-WIN 同合同消费者 replay 使用 5% material/non-inferiority margin；
  Parker 无 heap/mutex、token 快路 O(1)，address wait 至多一次对应 native wait/wake；稳定 CPU/cache facts
  O(1) cached read，动态 process availability 与拓扑枚举同原生输入规模；VM/file/mapping owner 成功路径
  不增加隐藏 allocation/lock，普通一步 RW mapping 不多于一次 native mapping；不为未请求能力支付常态
  metadata 或 syscall。
- 总调查预算 / 性能预算：能力/API/依赖/实现审计最多 240 分钟；统一性能取证最多 90 分钟，只允许一次
  probe、一次双 order/适用 placement 正式采样，以及协议允许的一次 material 异常机制诊断与一次产品
  重跑。产品 `FIX` 集在 `AUDITED` 冻结，预计实现超过一个工作日时必须拆分或进入具名 blocker，不能
  默默扩面。
- 最多复现、机制诊断、产品修改次数：每个异常一次反序复现、一次 codegen/profile 机制检查、最多一个
  同时拥有机制和消费者证据的产品修改及重跑；不得在本轮发明第五种 benchmark 姿势或修改中央统计规则。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用 correctness、
  sanitizer、OS/backend 与交付复核，不跨平台池化性能。多 NUMA node 真机当前未登记，因此只允许结构/
  capability 结论，不产生跨 socket 带宽声明。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用
  `closure-correctness-v1` 与现有 component/package 合同。
- strong peers（每项 capability 最多两个，均为本地钉死源码）：
  - wait/Parker/thread：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的 Baton/ParkingLot/thread
    utilities，与 oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 的 address waiter/thread monitor；
  - CPU/topology/NUMA：oneTBB 同 commit 的 `info`/NUMA allocation，与 Folly 同 commit 的
    HardwareConcurrency；
  - file/mapping/VM：Folly 同 commit 的 `File`/`MemoryMapping`，与 Boost
    `df8eeb90846648464e3c8e378e312f0cec772e83` 的 Interprocess mapped-region/Filesystem；
  - dynlib：Boost.DLL 同 commit 与 oneTBB 同 commit 的 dynamic-link implementation。
  当前消费者形状另以 llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb` 和 ggml
  `d99724f2b141fed9e1a9f402213506c081433465` 的 model mmap/NUMA/thread paths 取证；它们不冒充
  已有 HYLE 生产消费者。
- 无同合同 peer 时的预算及禁止声明：NUMA selected-node/chunk 在单节点 W1 上只做复杂度、syscall/
  scratch、rollback 和 single-node floor；CPU hybrid rank、macOS QoS、filesystem UTF/path 与 unsupported
  OS operation 按真 backend correctness/结构裁决。不得声明“所有 platform 操作跨平台 SOTA”、物理 NUMA
  placement/bandwidth、或把 raw syscall 不同语义的点值写成产品输赢。
- correctness gates：normal；null/empty/zero；path/size/offset/alignment/overflow/native-limit；invalid UTF-8/
  flags/node/CPU/symbol；OS/OOM/permission/unsupported；move/release/close/join/destructor；unpark-before-park、
  timeout/wake race、spurious wake；environment/global-state concurrency 限制；recursive-delete symlink safety；
  property/differential/stress；ASan/UBSan/TSan（按合同适用）；Windows/Linux/macOS 真 backend 与 public-header
  self-containment。
- delivery gates：中央 `platform PUBLIC core error time` 及 include-owner 对账；platform-only source build；
  relocated static/shared C++23 SDK consumer；代表性 mem/simd/container/sched/compute seam；产物/安装树无
  C schema、bridge、Native FFI symbol 或伪 option。
- performance gates：冻结两个独立 family，不互相池化：`scheduler-worker-idle-handoff` 使用正常 Parker
  token/blocked lifecycle 对 faithful same-contract peer，通过 A/A、5%/10% delta、floor、双 order 和 5%
  NI；`model-loader-mapped-probe` 使用持久只读 mapping、固定页序列与完整 open/access/close 计费，对
  Folly/等价 raw mapping 同合同 replay 通过相同资格与 5% NI。Thread/VM/FileHandle 等 syscall/control
  paths若没有 material profile，只做结构与 raw-operation-count diagnostic，不进入 P1 aggregate。
- 预期输出：capability-family map、public-surface trace、有限 gap ledger 与冻结 `FIX`；immutable candidate；
  correctness、delivery、usability/docs、red-team 和 P1 evidence；四项 LOC、known limitations 与 reopen
  triggers。
- 停止条件 / 可接受限制：所有本轮 `MUST` 为 `COMPLETE/ROUTED`、适用门通过且 P1 两个 family 分别 PASS
  则 `CLOSED`；若 correctness/security、同合同 5% NI、交付或 material MUST 在冻结预算内无法裁决，则
  点名 blocker/owner/resume condition 进入 `BLOCKED`。无具名消费者的 watcher/shared-memory/loader flags、
  多节点真机证书、Native FFI 和上层 placement policy 是可接受 negative space，不因 peer API 数量扩面。

## `SCOPED` 审计输入与待证假设

- 当前实现约 6,264 行，公开面分为 CPU/cache、thread、wait/Parker、VM、NUMA、fs/env/FileHandle、
  MappedFile、DynLib 八个族；既有 architecture/user 文档已经描述很多能力，但仍是旧设计输入，不能直接
  抄成 `COMPLETE`。
- 语法消费已确认 sched/compute 使用 Parker/Thread，mem 使用 VM/NUMA/thread，simd/container 使用
  CPU facts，compute 使用同步 file API；本轮还要扫描语义重复和错误的 OS/error/time 绕行。
- 初查发现的假设包括：公开头仍有已删除 C adapter/FFI 和历史 milestone 注释；`for_each_entry` 的
  forwarding-reference/lvalue callable 形状需要编译验证；DynLib 的保留理由和完整 loader journey 需由
  当前基础需求而非旧 FFI 铺垫重裁决；MappedFile 的 const/view、range、flush 与 owner 旅程需同 peer
  对账。这些都只是待证假设，只有 `AUDITED` ledger 才能冻结为 `FIX/KEEP/ROUTED/REJECT`。

## Capability-family coverage map

进入 `AUDITED` 时填写；在此之前不得修改产品实现。

## Capability / gap ledger

进入 `AUDITED` 时填写并冻结有限 `FIX` 集。

## Candidate

进入 `CANDIDATE` 时填写。

## Verification manifest

候选冻结后按 closure card 的适用矩阵填写，不把未运行项写成 PASS。

## Closure decision

尚未裁决。

