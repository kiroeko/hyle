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

