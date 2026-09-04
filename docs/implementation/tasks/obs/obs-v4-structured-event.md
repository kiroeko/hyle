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
- 内部复用新事实：`mem_reclaim` 已在本轮 baseline 前独立关闭。现有 Obs registry 仍有自有双快照
  reader epoch；必须按内部复用标准裁决为 `REUSE` 或留下同合同成本/生命周期不等价的 `LOCAL` 理由，
  不能因为历史实现已通过 benchmark 就自动保留；
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
