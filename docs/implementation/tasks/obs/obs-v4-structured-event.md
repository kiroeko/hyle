# Obs v4 结构化事件与底层复用

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由旧 `obs-v3 CLOSED` 之后形成的新架构事实定向触发，不重做已经签署的调用点控制、同步 sink
> 生命周期或 Tracy seam。当前唯一状态真相仍是
> [`module-status.json`](../../module-status.json)。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`obs` / Obs / `obs-v4-structured-event` / `REOPEN`；
- baseline：`067c3a5060390a8c592fdde38dd095f04fd68603`；前轮：`obs-v3`；
- trigger：`OBS-STRUCTURED-EVENT-2026-09-05`；
- [`capability-factorization-v1`](../project/capability-factorization-v1.md) 在旧 v3 关闭后完成跨域
  A/B/C 冻结，并把 `OBS-A-02` 的 logger-neutral structured event 定为 Obs 自己的 `MUST`：现有
  `LogRecord` 只有 message/source/raw Unix-ns 与 string-only fields，缺稳定 event/category identity、
  typed value 和显式 correlation attributes；
- 同一冻结图把 Obs 保持为 `KEEP_PRIME`，而不是应用 logger 或 metrics 聚合器；Metrics 是另一个
  `NEW_PRIME`。Obs 的目标依赖投影是 public `core,error,time`、private `mem_reclaim`、build-only
  Tracy。该图本身不授权实现，因此本轮只把上述差距变成有限产品审计；
- 普通 consumer journey：宿主 logger 在一个同步 callback 中接收可路由的 event/category、typed
  scalar attributes、可选 correlation 与类型安全 wall timestamp，不再解析 message 文本；
- 推理 consumer journey：model load/decode/slot progress 将 stage/current/progress、slot/request/model/
  backend identity、token rate、队列深度与状态码等 typed values 交给宿主 adapter；异步复制、排队、
  持久化和 exporter 仍由宿主拥有；
- 内部复用新事实：`mem_reclaim` 已在本轮 baseline 前独立关闭，并在
  [`mem-reclaim-freeze-confirmation-2026-09-05`](../mem/mem-reclaim-freeze-confirmation-2026-09-05.md)
  通过新标准复检、取得 registry `FROZEN` 标记。现有 Obs registry 仍有自有双快照 reader epoch；必须
  按内部复用标准裁决为 `REUSE` 或留下同合同成本/生命周期不等价的 `LOCAL` 理由，不能因为历史实现
  已通过 benchmark 就自动保留；
- 初步受影响面：`LogRecord` / attribute vocabulary、Time 互操作、registry snapshot reclamation、
  Obs factor dependency、正确性、source/static/shared 交付、用户/架构文档与现有 P1 正常 workload；
- 暂定 tier：`P1`，material margin 5%。普通 one-adapter Abseil 路线必须重新资格化；registry mechanism
  变化以八 producer/一个 adapter 对 immutable pre-v4 HYLE 做回归门。structured event 增量按当前
  llama.cpp model-load callback 的完整记录与一个真实 adapter 消费成本裁决，不用孤立 getter/union
  微基准塑造 API；
- 暂定 non-goals：metrics/counter/histogram、异步 logger、队列/持久化/网络 exporter、category config
  tree、automatic TLS/task context propagation、GPU/lock/async-flow tracing、crash reporter、Native FFI/C
  API、稳定 C++ ABI、Compute/Graph、相邻模块重开、公共仪器重写和 Formal 认证。

当前仅登记 `REOPENED` 新事实。完整 closure card、capability ledger、API freeze 与有限 `FIX` 集尚未
进入 `SCOPED/AUDITED`，不授权产品代码修改。

`CLOSED -> REOPENED` 前的 registry 快照是
`C:\hyle-scratch\obs-v4-structured-event\transitions\module-status-before-reopened.json`，SHA-256
`0F4355E3D4885972FB8705CC8906A5141C64BDF28F1F5056B53FB2A074606462`。

## Closure card（`SCOPED`）

- component / owner / round / kind：`obs` / Obs / `obs-v4-structured-event` / `REOPEN`；
- baseline / trigger：`067c3a5060390a8c592fdde38dd095f04fd68603` /
  `OBS-STRUCTURED-EVENT-2026-09-05`；
- real journeys：普通服务把结构化诊断同步交给既有 logger；推理 runtime 把 request/slot/model/backend
  identity、token rate、queue depth、状态与可选 wall timestamp 交给唯一 host adapter。调用方只在同步
  callback 返回前借用 message、attribute 与 context；需要异步时 adapter 当场复制；
- scope：冻结一个 allocation-free、logger-neutral 的 structured event vocabulary，覆盖稳定
  category/event identity、string/signed/unsigned/double/bool value、普通 attributes、显式 correlation
  attributes、标准 source location 与可选 HYLE Time wall timestamp；迁移现有 string-only record；核对
  Error typed status 能否由该 vocabulary 自然表达；裁决 registry 的本地 snapshot epoch 与已关闭
  Mem-reclaim 是否同合同，只有等价时才复用；同步更新 factor manifest、测试、三交付、guide、architecture
  与受影响性能证据；
- non-goals：metrics/counter/histogram、异步 logger、queue/persistence/network exporter、category
  threshold/config tree、自动 TLS/fiber/task context capture/restore、GPU/lock/async-flow tracing、crash
  reporter、owning record、Native FFI/C API、稳定 C++ ABI、Compute/Graph、相邻模块重开、公共仪器重写与
  Formal；
- dependencies / ownership：Obs 保持一个质因数；Core 提供 compiler/assert/source 基础，Error 提供
  registry failure，Time 唯一拥有 typed Unix timestamp；Mem-reclaim 只是待裁决的 private mechanism，
  不进入公开 type。Tracy v0.14.1 仍是 trace-ON build-only backend；Metrics 保持独立未来 factor；
- tier / margin：`P1` / 5%。ordinary one-adapter 对当前 Abseil 是自然公开路线门；受影响 registry
  dispatch 以八 producer/一个 adapter 对 immutable pre-v4 HYLE 做同源回归。structured journey 使用
  当前 llama.cpp `load_progress_callback` 的三 stage、first/every-200-ms/final cadence 与字段，只测完整
  record -> one adapter -> typed consume，不为单个 variant/getter 建 benchmark，也不把异步 queue 或
  category tree 塞给 peer；
- normal metadata / cost：关闭 level 与 compile floor 仍不求值、不构造 record；普通 message emit 不得
  新增 allocation、clock read、global lookup、mutex 或 runtime type registry；typed record 只为显式选择的
  metadata 支付常数大小 borrowed descriptors，dispatch 保持 O(number of sinks + attributes consumed)；
  add/remove/flush 仍是冷路径；
- budgets：capability/API/实现 3 小时，正确性/交付/文档 2 小时，P1 资格 1 小时，红队 30 分钟；最多
  一次正常 workload qualification、一次 registry mechanism 对照和一次性能导向实现修正。无资格仪器
  或一次机制诊断后仍稳定 material FAIL 则 `BLOCKED`，不扩建公共性能系统或增加第三种测试姿势；
- machines / instruments：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 `closure-perf-v1`，correctness/P0
  `closure-correctness-v1`；点值只由 W1-WIN 裁决，其余平台验证 correctness/交付；
- strong peers / consumers：Abseil `9e15d3d6cc19ae4703f3ad7fa413a789580ec1fc`、Folly
  `dea9962e1a7d7cbbbeab5d6d4fe12d72b72ddabb`；llama.cpp
  `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd` 与 ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 是真实推理形状，不冒充 same-contract timing peer；
  Tracy shipping lock 保持 release v0.14.1 / `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`；
- correctness gates：空 event/category/context/attributes；exact byte string 与 embedded NUL；五种 value
  kind 和所有整数宽度；invalid enum/access misuse 的编译/断言边界；可选 timestamp 的 epoch/pre-epoch/
  invalid fraction；source capture；Status code/native/message 映射若准入；原 registry capacity、并发、
  recursion、自摘/跨摘、quiescence、flush/default sink；format failure/truncation；trace OFF no-evaluation
  与 ON regression；公开头自包含；WSL ASan/UBSan/TSan；
- delivery gates：精确 Obs factor 与 consumer-facing Obs product；source embedding、relocated installed
  static/shared C++23 consumers；Windows/Linux/macOS；trace OFF 与 locked Tracy ON；无 C/FFI artifact；
- outputs：capability map、public-surface reverse trace、内部复用账本、有限 `FIX`、冻结 API、实现与测试、
  架构/用户/性能文档、candidate、verification manifest、七问、独立红队、三项 LOC、terminal receipt 与
  新 `FROZEN` confirmation；
- stop：全部准入 `MUST` complete/routed，依赖图与三交付一致，correctness/docs/red-team/P1 gate 通过
  才 `CLOSED/FROZEN`；否则明确 `BLOCKED`。不因 GENERAL peer API 数或无 consumer 的 profiler knob
  继续扩张。

当前进入 `SCOPED`；上述范围、成本、机器、预算和停止条件冻结。产品实现仍须等待 public-surface
反向追踪、capability ledger、reuse decision 与 API shape 在 `AUDITED` 冻结。

`REOPENED -> SCOPED` 前的 registry 快照是
`C:\hyle-scratch\obs-v4-structured-event\transitions\module-status-before-scoped.json`，SHA-256
`BC567609600D9287D044FDCDC42CA53ABDFC865A84392B6360D6356523592AEA`。

## Capability-family coverage map（`AUDITED`）

### 使用原型与 peer 裁决

普通服务只注册一个同步 adapter。producer 构造 borrowed record，adapter 在 callback 返回前按类型读取
category/event、普通 attributes、correlation attributes、source 与可选 wall timestamp；若上层异步，
它在这一处复制。锁定推理原型是 `model.load_progress`：category=`inference.model`，普通 attributes
按原 consumer 顺序保留重复的 stage 列表、current stage 与 floating progress，host correlation 可附
model-load identity。decode slot 的 request/slot、model/backend、token rate、queue depth 与 status fields
使用同一 vocabulary 自然组合，但不冒充本轮锁定的计时来源。关闭 level 时不构造 record；开启时只为
caller 显式给出的 descriptors 付费。

- Abseil `9e15d3d6cc19ae4703f3ad7fa413a789580ec1fc` 的 `LogEntry` 是同步 borrowed sink record，带 typed
  timestamp/thread/source；其 structured 路线把 scalar 编码为尚未公开 schema 的 protobuf field-number
  stream。它证明 borrowed typed metadata 成熟，但 HYLE 不能把私有 protobuf 编号或序列化成本变成
  logger-neutral public schema；
- Folly `dea9962e1a7d7cbbbeab5d6d4fe12d72b72ddabb` 的 `LogMessage/LogHandler` 提供 category、typed
  timestamp/thread、同步 handler 与 flush，但 message/context 是 owning string。它证明 category +
  synchronous adapter 的正常形状，不证明所有 HYLE record 都应分配或捕获 thread/time；
- llama.cpp `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd` 的一般日志仍由全局 callback 输出 text，server 的
  `SLT_*` 宏把 slot/task identity 与 token/s 拼入字符串；但同一 server 的 `load_progress_callback`
  已直接向 host state callback 提交 `stages/current/value`，三个 stage 各自持有 `t_last`，首条、末条和
  每 200 ms 的中间条通过。ggml `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 仍只暴露 level/text
  log callback。它们给出真实字段、ownership 与 cadence，不是现成的 structured logging timing peer；
  HYLE record 让同一类 host adapter 不必解析日志文本。

### 能力图

| Family | normal journey / peer | current route | coverage | frozen decision / negative space |
|---|---|---|---|---|
| level + exact message | 普通诊断；Abseil/Folly/ggml | `LogLevel`、borrowed `string_view` | `COMPLETE` | `KEEP`；不改 format/gate 语义 |
| stable category/event identity | host routing、slot progress | 只能塞进 message/string field | `MISSING` | `FIX` 两个独立 borrowed byte views；空值合法，不建 category config tree |
| typed scalar attributes | backend/model/rate/depth/status；Abseil scalar structure | `LogField` 只有 string/string | `MISSING` | `FIX` string/signed/unsigned/double/bool tagged value；不引入 heap/schema registry |
| explicit correlation attributes | request/slot/trace identity | 与普通 fields 混在一起 | `MISSING` | `FIX` 独立 borrowed span，复用同一 attribute vocabulary；不做自动 TLS propagation |
| source location | C++ caller/foreign logger bridge | 三个松散 raw fields，无 column/exact slice | `PARTIAL` | `FIX` borrowed `LogSourceLocation` + `std::source_location` adapter；仍允许显式外来 source |
| wall timestamp | host ingestion、persisted event | raw `uint64_t ns`，epoch 0 被误作 absent | `PARTIAL` | `FIX` optional `time::UnixTimestamp`；不自动读钟，invalid canonical fraction fail-fast |
| Error status expression | failure boundary | 手写 string fields 会丢 code/native type | `PARTIAL` | `FIX/DOC` vocabulary 可无损表达 code-name/message/native-domain/native-code；不把 Status 常驻 record |
| synchronous sink lifecycle | 一个 host adapter；Abseil/Folly | typed sink、RAII、flush、quiescent remove | `COMPLETE` | `KEEP`；受影响回归，不增 async owner |
| immutable registry publication | producer fan-out | 自有全局 RMW reader epoch + active refs | `DUPLICATED` | `FIX/REUSE` Mem-reclaim guard 只保护 snapshot copy；active refs 继续保护 callback |
| call-site rate control | repeated warning/progress | explicit states + macros | `COMPLETE` | `KEEP`；v3 aggregate 回归 |
| trace seam | profiling build | Tracy native macros | `COMPLETE` | `KEEP`；只做 OFF/ON 受影响回归 |
| owning/async logging, metrics, task context | queue/export/dashboard | host / future Metrics factor | `ROUTED` | 不进入 Obs v4 record 或线程模型 |

### 现有 public-surface 反向追踪

- `log.h` 的 level/gate、message/format、sink/registration、default sink 与 registry queries 全部映射到
  上表；`log_control.h` 的 approximate/exact/first/power-of-two/interval 状态映射 call-site control；
  `trace.h` 全部映射 Tracy seam，没有无 owner symbol；
- `LogField` 与 raw `timestamp_ns` 是本轮明确替换对象，而不是兼容义务：项目未发布，不保留 string-only
  alias 或 0-sentinel；旧字段名不得以 deprecated wrapper 留在常态 record；
- sink 的 `void* + noexcept function pointer` 是零分配 C++ 类型擦除，`LogSink::from` 为正常 typed adapter；
  它没有 C linkage/schema/version，不能误标为 Native FFI；
- registry 固定容量、双 in-place snapshot 与 per-sink active ref 都有真实 lifecycle owner。迁移后前两者
  仍负责 storage/publication，只有重复的 reader grace-period protocol 下沉到 Mem-reclaim；
- public trace 没有 category hierarchy、per-sink filter、owned field、async queue、metrics registry、
  thread ID、automatic timestamp 或 task-local propagation。上述均无本轮真实 consumer，保持负空间。

## Capability / gap ledger 与冻结 `FIX`

| ID | class | owner / public shape | cost / semantics | action | verification |
|---|---|---|---|---|---|
| `OBS4-VALUE-01` | `MUST` | `LogValueKind` + `LogValue` | trivially copyable tagged scalar；borrowed exact string；整数归一到 64-bit | `FIX` | 五种 kind、全部整数宽度、embedded NUL、wrong-access death |
| `OBS4-ATTR-01` | `MUST` | `LogAttribute` + record spans | caller-owned ordered spans；duplicate/empty keys preserved | `FIX` | empty/duplicate/exact byte/order/lifetime |
| `OBS4-ID-01` | `MUST` | record `category` / `event` | two borrowed exact byte identities；empty means unspecified | `FIX` | empty/exact/control bytes/adapter routing |
| `OBS4-CORR-01` | `MUST` | `correlation_attributes` | same scalar vocabulary, distinct semantic span | `FIX` | request/slot inference record and empty path |
| `OBS4-SOURCE-01` | `MUST` | `LogSourceLocation` | borrowed file/function + line/column；macro captures standard source | `FIX` | macro callsite/custom source/empty source |
| `OBS4-TIME-01` | `MUST` | `optional<time::UnixTimestamp>` | no automatic clock; epoch/pre-epoch valid；invalid fraction fail-fast | `FIX` | absent/epoch/pre-epoch/invalid fraction |
| `OBS4-STATUS-01` | `MUST` | ordinary typed attributes | code name/message/native domain/code without owning Status | `FIX/DOC` | canonical/native/no-message recipe |
| `OBS4-RECLAIM-01` | `MUST` | private registry mechanism | guard covers snapshot copy only；per-sink active covers callback | `FIX/REUSE` | handoff race/self-remove/cross-remove/concurrency + mechanism check |
| `OBS4-DEFAULT-01` | `MUST` | default stderr/debugger rendering | typed values and identities escaped; numeric render allocation-free | `FIX` | controls/NUL/all kinds/timestamp/one physical line |
| `OBS4-DELIVERY-01` | `MUST` | Obs factor/product | public `core,error,time`; private `mem_reclaim`; C++ source/static/shared | `FIX` | exact closure/selected headers/relocated consumers/symbol scan |
| `OBS4-PERF-01` | `MUST` | normal one-adapter journeys | old plain/control no material regression；typed full-consume under budget | `FIX` | qualified P1 + one registry mechanism comparison |
| `OBS4-ASYNC-01` | `GENERAL` | owned queue/export | host adapter | `ROUTED/REJECT` | guide route only |
| `OBS4-METRIC-01` | `GENERAL` | counter/gauge/histogram | future Metrics factor | `ROUTED/DEFER` | independent component admission |

冻结 `FIX` 只有 `VALUE/ATTR/ID/CORR/SOURCE/TIME/STATUS/RECLAIM/DEFAULT/DELIVERY/PERF` 十一项。任何 async
logger、owning record、category config、task context、metrics、crash、trace 扩面、FFI、Compute/Graph 或
公共仪器变化均不进入候选。

## API / usability / cost freeze

- 用 `LogValueKind::{String,SignedInteger,UnsignedInteger,FloatingPoint,Boolean}` 与一个 standard-layout、
  trivially-copyable `LogValue` 表达 scalar。构造接受 exact `string_view`/string literal、≤64-bit signed/
  unsigned integer、`float/double` 与 `bool`；不接受隐式 enum 或 `long double` 丢精度。按 kind 调用
  `as_string/as_signed_integer/as_unsigned_integer/as_floating_point/as_boolean`；错 kind 是 programmer
  contract violation 并 fail-fast，不返回伪默认值；
- `LogAttribute { string_view key; LogValue value; }` 取代 `LogField`。`LogValue` 自然借用 string literal、
  `string_view` 和 `std::basic_string<char>` 左值，拒绝临时 string、raw pointer、enum 与 long double；
  两个 span 都是同步借用、保序且允许 duplicate/empty key；`attributes` 表示事件事实，
  `correlation_attributes` 表示跨记录关联 identity，HYLE 不按 key 建索引或复制；
- `LogSourceLocation { string_view file; string_view function; uint32_t line; uint32_t column; }` 提供
  `from(std::source_location)`；宏仍捕获 caller。显式 struct 让 bridge 可以表达外来 source，而不是被
  不可伪造的 `std::source_location` 构造限制；
- `LogRecord` 冻结字段顺序为 level、category、event、message、attributes、correlation_attributes、
  source、`optional<time::UnixTimestamp> timestamp`。timestamp absent 才表示未提供；Unix epoch 0 与
  pre-epoch 都合法。record、views、spans 与 string-valued attributes 只借到同步 callback 返回；
- Error 不进入 record layout。需要记录 `Status` 时按需构造 `status.code`（stable name）、
  `status.message`、`status.native_domain` 与 signed `status.native_code` attributes；这样保留 machine type，
  又不让每条非错误 record 支付 refcount/heap owner，也不替应用决定 message 是否可外发；
- 关闭 level 与 compile floor 的宏保持不求值；普通 message emit 不自动读 clock/thread/context，不分配，
  不做 runtime schema lookup。显式 typed record 只支付 O(attributes consumed + sinks) 的 borrowed descriptor
  traversal；default sink 的 numeric conversion 用固定栈缓冲；
- Obs public dependency 更新为 `core,error,time`，private dependency 为已经带新 `FROZEN` marker 的
  `mem_reclaim`。registry mutation 在锁内调用 `synchronize()` 后复用 inactive fixed snapshot，publish 后
  再等旧 snapshot readers 完成；emit guard 在取得全部 per-sink active refs 后立即退出，绝不覆盖任意
  callback；
- 普通 C++ recipe 仍是 aggregate record + `log_record` 或 `SinkRegistration::create(backend)`；不新增
  builder、runtime variant registry、`fast/light` API、C bridge 或 stable ABI。三种 C++ 交付都要求 matched
  headers/binary。

## Audit verdict

所有准入 `MUST` 均已有当前路线或上述有限 `FIX`；public names、ownership、failure、dependency、正常
成本、peer/consumer 与 P1 边界已经冻结，允许进入产品实现。Mem-reclaim 的新标准前置冻结证据是
[`mem-reclaim-freeze-confirmation-2026-09-05`](../mem/mem-reclaim-freeze-confirmation-2026-09-05.md)，
因此本轮不再存在“旧 CLOSED 冒充新 FROZEN”的依赖缺口。

`SCOPED -> AUDITED` 前的 registry 快照是
`C:\hyle-scratch\obs-v4-structured-event\transitions\module-status-before-audited.json`，SHA-256
`977328A23182F97888A9FB212424A32448C24FA4C07341C3D654D94BF7C1AB94`。

## 实现与性能输入纠正（candidate 前）

- 十一项冻结 `FIX` 已实现：公开 record 改为 typed borrowed event；Time timestamp 能区分 absent/epoch/
  pre-epoch；默认 sink 分配-free 地渲染并转义全部 scalar；Error status 有无损 recipe；registry 的短
  snapshot grace period 复用 `mem_reclaim`，callback quiescence 仍由 per-sink active ref 所有；
- `std::string` 易用性红队发现“左值也必须手写 string_view”不自然，因此在 `OBS4-VALUE-01` 内补入
  lvalue-only 借用重载，并在编译期拒绝 rvalue，既不放宽生命周期也不新增 benchmark 形状；
- Obs 获得后来登记的 `mem_reclaim` 边后，构建系统暴露出把 append-only component identity 顺序误当
  target 构造顺序的真实缺陷。修复由独立
  [`build-system-v12`](../build/build-system-v12-append-only-topological-construction.md) task 所有：中央图
  以登记顺序作稳定 tie-breaker 做 O(V+E) 拓扑解析，identity 历史不重排，target 与 closure 使用同一
  解析结果；component contract 已加入“旧 Obs 依赖后来低层 factor”的回归 fixture；
- v3 文档把 `16 slots x 256 ticks / 3-second` 合成 replay 写成 llama.cpp 当前 inference provenance；逐行
  检查锁定 commit 后确认这不成立，旧点值只保留为历史诊断，不能续签 v4。v4 重新锁定三份输入：
  当前 Abseil ordinary every-N；当前 llama.cpp 三 stage model-load（first/every 200 ms/final，字段
  `stages/current/value`）；八 producer/一个 host adapter 对 pre-v4 registry。4/16 独立 sink 只运行为
  capacity/contention 诊断，不得决定产品 gate；
- consumer manifest 是 `bench/obs/consumer_shapes_v2.json`，SHA-256
  `317d984ccddd0692e604bab733d5491fc7e5d597c36f55c229342f2eec67f652`。普通、structured 与 concurrent
  三份新合同的 immutable lock 分别为
  `5d2d7210782f1bae55c14bd42246f2485ce02ce6b74cf2e1f5bd8e20c8a33759`、
  `b35d0c0d81980ba39bc59a8c70048962fd9625fe179632ca801ed34f310594f0`、
  `3d527efed8d509b18207b80a7c0f556f908a11fb4ee9f23ffde0df000b32eb34`；资格结果只在 immutable
  candidate 后填写。

## Candidate

- product commit / tree：`14c82a18f8f00b892479ae16abba4d8320d5915c` /
  `7594a2f13271ba7956c3e685e24c6302a91f015b`；
- 公开合同变化：`LogField` 与 raw timestamp/source 字段被新的 `LogValue`、`LogAttribute`、
  category/event、correlation span、`LogSourceLocation` 与 optional `time::UnixTimestamp` 完整替换；项目
  未发布，不保留旧别名。普通 format/message/macro、rate-control 与 trace 表面保持；
- ownership / failure / concurrency：record 仍同步借用，string owner 左值可自然借用而 rvalue/raw pointer
  被编译拒绝；invalid timestamp/wrong value getter fail-fast；sink registration 失败仍走 Error。registry
  guard 只覆盖 snapshot-copy handoff，callback 继续由 per-sink active ref 排空；
- 算法 / 常态成本：disabled/plain message 不增加 clock、heap、schema lookup 或 owning metadata；显式
  structured record 支付固定 descriptor 与 adapter 实际遍历；registry read 从一个全局 RMW reader count
  改为已冻结 `mem_reclaim` 的 thread-route guard；
- 直接 consumer：package sample 已迁移到完整 typed event；当前 llama.cpp model-load 形状由新的
  structured replay 消费；ordinary 与 registry replay 分别锁定当前 Abseil 与 pre-v4 HYLE；
- candidate 前已通过：Windows/MSVC 19.50 与 WSL/GCC 14.2 selected shared Obs tests，各 3/3；WSL
  ASan/UBSan/TSan 各 3/3；Windows/WSL source + relocated shared/static consumer；Windows/WSL trace ON
  tests；component contract 与三份 benchmark semantic/contract validator；
- harness / product identity：benchmark、consumer manifest、validator 与 build-v12 task 在同一提交中，
  但性能 runner 会分别记录 contract/harness 与 product commit/tree；手工单次点值不进入裁决。

`AUDITED -> CANDIDATE` 前的 registry 快照是
`C:\hyle-scratch\obs-v4-structured-event\transitions\module-status-before-candidate.json`，SHA-256
`F1B3426FDB56BEB9DEF1C805D4ED0A3A50B8CB474BDF1935A6D7AD947C6F356E`。

## Verification manifest

验证全部绑定 immutable product candidate
`14c82a18f8f00b892479ae16abba4d8320d5915c` / tree
`7594a2f13271ba7956c3e685e24c6302a91f015b`。candidate 后 `include/hyle/obs`、
`src/obs`、Obs component edge 与 package sample 的 product diff 为零；随后提交只保存状态与证据。

| Risk / journey | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| selected Obs Release/shared tests | 3/3 | 3/3 | 3/3 | `PASS` |
| public-header self-contained | `PASS` | `PASS` | `PASS` | `PASS` |
| final candidate ASan / UBSan / TSan | n/a | 3/3 each | n/a | `PASS` |
| source-embedded consumer | 1/1 | 1/1 | 1/1 | `PASS` |
| relocated installed shared consumer | 1/1 | 1/1 | 1/1 | `PASS` |
| relocated installed static consumer | 1/1 | 1/1 | 1/1 | `PASS` |
| locked Tracy trace-ON Obs tests | 3/3 | 3/3 | 3/3 | `PASS` |
| complete component contract | `PASS` | `PASS` | `PASS` | `PASS` |

Windows 还从同一 candidate 通过 `HYLE_COMPONENTS=all`、tests/bench/install/trace/TLS/HTTP2/CUDA/ggml/
Vulkan 全 OFF 的完整 shared 构建；这只验证 Build v12 的全图构造，不替未冻结的 Compute/Graph 签收产品。
三平台 selected closure 都是 product `core,error,obs`、factor
`core,error,mem_reclaim,time,obs`。macOS 初次验证只发现机器没有 Ninja，随后用受支持的 Unix Makefiles
在新目录通过；没有为环境差异修改项目。

测试覆盖五种 scalar、全部准入整数宽度、exact bytes/NUL、wrong-kind 与 invalid timestamp fail-fast、
empty/duplicate attributes、source/correlation/status recipe、format failure/truncation、default rendering、
registry capacity/duplicate/not-found、并发 emit/mutation、自注销/跨注销、callback 排空、recursion、flush、
compiled/runtime gate 和 trace OFF no-evaluation。WSL sanitizer 是 candidate 最终 API（包括 string lvalue
借用与 rvalue 拒绝）的 fresh build，不继承 candidate 前的旧 object。

### W1-WIN P1 engineering qualification

三份 contract 都使用 clean candidate、fresh MSVC 19.50 Release/static、portable ISA、LTO OFF 和
`closure-perf-v1`；A/A、5%/10% known-delta、timer-floor、build/artifact identity 与适用 semantic/resource
validator 全部通过：

| Journey | Samples | Exact verdict | 5% NI | Conservative result |
|---|---:|---|---|---|
| ordinary every-N one adapter vs Abseil | 15 x 4 strata | `WIN` | `PASS` | ratio `5.5077–5.6112`；最低 CI lower `5.4030` |
| complete typed model-load progress vs source-derived llama gate | 31 x 4 strata | `PARITY` | `PASS` | ratio `0.9715–0.9821`；全部 CI 在等价带内 |
| 8 producers / 1 adapter throughput vs pre-v4 registry | 31 x 2 strata | `INCONCLUSIVE` | `PASS` | ratio `1.0519–1.0536`；不宣称确定胜出 |
| same registry replay, 256-emit chunk p99 rate | 31 x 2 strata | `INCONCLUSIVE` | `PASS` | ratio `1.0460–1.0475`；不宣称确定胜出 |

ordinary manifest/summary SHA-256 为
`54381e39ed8a390146aea4bd9761f51bfa33d6c4284c01430e93ade6a7ec0681` /
`f36d99bb3bf345aafc765e18b92a2553d927586d0ea7e8f3edbe83eb6e8d2af1`；structured 为
`22bb9d5416f0f0947318e4bd1ce045dfafc4ed6c48c228d6fbdb5d15e65e6d01` /
`7cbf305d6ba7603e9eb5e0921accd9ed8e77db9bcc18e671367962b2c4073475`；concurrent 为
`cd42b1879493007a0dc7a8027c3e882847dce06e89197f9c35a0a889710b0b0c` /
`8ca4ce843068d9e9a87538236ad7496bc2acca5fca6156e7d92323909d63fb14`。三者经
`hyle-perf verify` 返回 `QUALIFIED / EXACT / CURRENT`、`paths_verified=true`。完整 controls/
observations identity 与禁止外推见 [`benchmarks-obs`](../../evidence/performance/benchmarks-obs.md)。

结构化路线的 HYLE 点值约慢 source-derived manual limiter `1.8–2.85%`，但两侧构造并完整消费相同 typed
record，且四层 CI 均在预声明 5% 等价带内；按冻结停止规则不为这个非 material 差异修改 API/实现。
registry 点估计更快约 4.6–5.4%，但 exact CI 跨决策边界，只签 non-inferiority。两者都不是全面 SOTA
或端到端推理性能声明。

## Owner 七问终答

1. **第三方产品路径：`PASS`。**Trace OFF 的 Obs 产品闭包没有第三方实现；Trace ON 直接编译并链接锁定
   Tracy v0.14.1 / `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`，类别为 optional product/build backend，
   由中央 dependency lock、license 与单 client owner 管理。Abseil 是 benchmark peer，Folly 是 architecture
   reference，llama.cpp/ggml 是 consumer provenance，均不进入 Obs 产品。C++ standard library 与
   Windows/POSIX terminal/debugger API 是平台机制，不冒充第三方库。
2. **功能超集与推理消费：`PASS`。**Level/message、同步 sink、RAII/quiescent unregister、format、六类
   call-site control 与 Tracy seam 保持完整；v4 补齐 category/event、五种 typed scalar、普通/关联属性、
   source column、typed optional Unix timestamp 和 Error status recipe。当前 llama.cpp model-load 与
   slot/decode identity/rate/depth/status 可自然表达；当前 ggml level/text callback 也被普通路线覆盖。异步
   queue/export、metrics 与自动 task context 已完整路由给 host/future factor，未处置 `MUST` 为 0。
3. **内部复用与加速设施：`PASS`。**Core 的 compiler/environment/fatal contract、Error 的 Status/Result、
   Time 的 UnixTimestamp、Mem-reclaim 的短 snapshot grace period 均为 `REUSE`；Tracy 为 feature-ON
   `BORROW`。per-sink active ref 是 callback quiescence 的 Obs-owned `LOCAL`，与任意对象 reclamation 不同
   合同；固定 snapshot/slot arrays 避免常态 allocation，通用 Mem/Container 不适用。terminal/debugger
   输出直接借 OS API，因为冻结 Platform 没有同合同 console seam；不存在第二套 runtime clock、error、
   reclamation 或 trace authority，也没有为增加引用数量而制造依赖。
4. **拟真、公平性能：`PASS`。**普通公开路线对当前 Abseil 明确 `WIN`；完整 typed model-load consumer 为
   `PARITY/PASS`；受影响 registry 对 immutable pre-v4 为 `INCONCLUSIVE/PASS`。三者均比较相同语义、资源
   计费和 source/static 实际形状；4/16 sink 只作诊断。只签 W1-WIN P1 工程范围，不宣称所有操作、所有
   平台、异步 logger、end-to-end token latency 或 Formal SOTA。
5. **旧 C ABI / Native FFI：`CLEAN`。**产品树没有 C header/schema/bridge、managed sample、专用 option/
   target/package metadata或只为 FFI 支付的 layout/indirection。Windows DLL 的 24 个导出与 macOS dylib
   exports 全为 C++ decorated symbols；`detail::log_record_unchecked` 是 public header format template 必需的
   compiled C++ seam，不是未装饰 C ABI。
6. **构建与导入：`PASS`。**中央图唯一声明 public `core,error,time`、private `mem_reclaim`，稳定 O(V+E)
   拓扑构造允许旧 component identity 合法依赖后来拆出的低层 factor。三平台 exact selected closure、source
   embedding、relocated shared/static import 与 Trace ON 均通过；Windows 全产品 feature-OFF 和三平台完整
   component contract 通过。未选模块不会因这条边被构建。
7. **C++ API 与文档：`PASS`。**正常用户从 macro/message/structured record/typed RAII sink/explicit state/
   trace 五种任务进入；borrow、temporary rejection、callback concurrency、remove/flush、默认 sink、status
   脱敏与 async-copy 陷阱均可发现。API 没有 benchmark/internal mode、`fast/light` 分叉或 legacy alias。
   [`obs-design`](../../../architecture/obs-design.md) 只保存长期边界和原因，
   [`user/obs`](../../../user/obs.md) 按“什么情况用什么”给出 recipe，task/performance 分别保存施工事实与点值。

## 独立红队终审

`PASS`。红队先复核 comparison class，再攻击 product：

- 逐步检查 publish 前 inactive snapshot、publish 后旧 reader、active-reference handoff、external/self remove、
  cross-remove、last releaser、slot reuse 与 callback recursion；第二次 `snapshot_domain.synchronize()` 关闭旧
  snapshot 到 active ref 的窗口，guard 在 callback 前释放，不会让慢 callback 或自注销阻塞全域 grace period；
- `LogValue` 的 string owner 只接受左值，临时 string/raw pointer/enum/long double 在编译期拒绝；record 的
  span/view 仍明确同步借用。timestamp 能表示 absent、epoch 0 与 pre-epoch，非法 fraction fail-fast；
- 默认 sink 对 message/category/event/source/key/string value 的控制字节逐字转义，numeric 转换使用固定栈
  buffer；它仍是 I/O 冷后端，不把 mutex/syscall 成本藏进普通自定义 adapter 路线；
- candidate 后产品 diff 为零；三平台 sanitizer/交付、Trace ON、完整 component contract、DLL/dylib symbol、
  consumer semantic matrix 与三份 retained bundle 没有暴露合同或证据身份漂移；
- candidate 前发现并修复的三个真实问题——临时/左值 string 易用性边界、伪装成当前 llama.cpp 的旧 replay、
  component identity 顺序与 target 构造顺序耦合——均在冻结前解决，没有以文档解释替代实现或测试。

剩余 medium-or-higher finding、未处置 `MUST`、material 性能 FAIL、重复内部 authority 与交付 blocker 均为 0。

## Aggregate verification verdict

| Gate | Verdict | Boundary |
|---|---|---|
| scope | `PASS` | 十一项冻结 FIX 完成；candidate 后产品不变 |
| architecture | `PASS` | Obs 仍为同步 producer seam；metrics/async/task context owner 清楚 |
| correctness | `PASS` | 三平台、最终 WSL ASan/UBSan/TSan 与并发生命周期矩阵 |
| delivery | `PASS` | 三平台 source/relocated shared/relocated static、Trace OFF/ON |
| usability/docs | `PASS` | 自然 typed borrowed API、完整选型 recipe 与陷阱 |
| red team | `PASS` | ownership/reclamation/FFI/build/comparison/product-diff；MEDIUM+ 为 0 |
| engineering performance | `PASS` | 三份 W1-WIN P1 合同的所有 required strata 均通过 5% NI |

七项 gate 全部 `PASS`，所有准入 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，没有 blocker；Formal 独立保持
`PENDING`。`CANDIDATE -> VERIFIED` 使用前态快照
`C:\hyle-scratch\obs-v4-structured-event\transitions\module-status-before-verified.json`，SHA-256
`BF40E4536B8D00F4EE2E2B0AFBA47E2A45A14587821C2F3168BFE5BC09FDACFB`。
