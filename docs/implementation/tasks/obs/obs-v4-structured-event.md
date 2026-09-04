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
- 推理 consumer journey：model load/decode/slot progress 将 slot/request/model/backend identity 与
  token rate、队列深度、状态码等 typed values 交给宿主 adapter；异步复制、排队、持久化和 exporter
  仍由宿主拥有；
- 内部复用新事实：`mem_reclaim` 已在本轮 baseline 前独立关闭，并在
  [`mem-reclaim-freeze-confirmation-2026-09-05`](../mem/mem-reclaim-freeze-confirmation-2026-09-05.md)
  通过新标准复检、取得 registry `FROZEN` 标记。现有 Obs registry 仍有自有双快照 reader epoch；必须
  按内部复用标准裁决为 `REUSE` 或留下同合同成本/生命周期不等价的 `LOCAL` 理由，不能因为历史实现
  已通过 benchmark 就自动保留；
- 初步受影响面：`LogRecord` / attribute vocabulary、Time 互操作、registry snapshot reclamation、
  Obs factor dependency、正确性、source/static/shared 交付、用户/架构文档与现有 P1 正常 workload；
- 暂定 tier：`P1`，material margin 5%。旧 normal one-adapter 与 inference progress aggregate 必须保持；
  structured event 增量按一个真实 adapter 的完整消费成本裁决，不用孤立 getter/union 微基准塑造 API；
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
- tier / margin：`P1` / 5%。旧 v3 的 ordinary one-adapter Abseil aggregate 与 llama-shaped inference
  progress 是回归门；新增 structured journey 只测完整 record -> one adapter -> typed consume，不为
  单个 variant/getter 建 benchmark，也不把异步 queue 或 category tree 塞给 peer；
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
它在这一处复制。推理原型是 `slot.progress`：category=`inference.decode`，correlation 里是 request/slot，
普通 attributes 里是 model/backend、token rate、queue depth 与 status fields。关闭 level 时不构造
record；开启时只为 caller 显式给出的 descriptors 付费。

- Abseil `9e15d3d6cc19ae4703f3ad7fa413a789580ec1fc` 的 `LogEntry` 是同步 borrowed sink record，带 typed
  timestamp/thread/source；其 structured 路线把 scalar 编码为尚未公开 schema 的 protobuf field-number
  stream。它证明 borrowed typed metadata 成熟，但 HYLE 不能把私有 protobuf 编号或序列化成本变成
  logger-neutral public schema；
- Folly `dea9962e1a7d7cbbbeab5d6d4fe12d72b72ddabb` 的 `LogMessage/LogHandler` 提供 category、typed
  timestamp/thread、同步 handler 与 flush，但 message/context 是 owning string。它证明 category +
  synchronous adapter 的正常形状，不证明所有 HYLE record 都应分配或捕获 thread/time；
- llama.cpp `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd` 仍通过一个全局 callback 输出 text，server 的
  `SLT_*` 宏把 slot/task identity 与 token/s 拼入字符串；ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 同样只暴露 level/text callback。它们是真实字段和 cadence
  证据，不是结构化同合同 timing peer；HYLE 的价值是让 host 不再反向解析这些文本。

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
- `LogAttribute { string_view key; LogValue value; }` 取代 `LogField`。两个 span 都是同步借用、保序且允许
  duplicate/empty key；`attributes` 表示事件事实，`correlation_attributes` 表示跨记录关联 identity，
  HYLE 不按 key 建索引或复制；
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
