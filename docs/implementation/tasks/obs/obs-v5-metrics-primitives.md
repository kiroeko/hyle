# Obs v5：推理运行时指标聚合原语

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由已冻结模块的真实推理消费者回放触发。当前仅为 `REOPENED`：本页保存可复现的新产品事实与
> 最小影响面；在后续 `SCOPED/AUDITED` 分别冻结 closure card、能力账本和有限 `FIX` 之前，不修改
> Obs 产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`obs` / model-independent metrics aggregation /
  `obs-v5-metrics-primitives` / `REOPEN`；baseline
  `c51599be2fdcd4caeacc5b05aa0f61f725c62491`，tree
  `40e9c1d341548357bfdccc667d0c3af7d6ae0549`；
- trigger ID：`OBS-METRICS-PRIMITIVES-2026-09-06`。锁定 llama.cpp
  `73a43d1f69345aee8bb186ef4b3172cef892f2e5` 的
  `tools/server/server-common.h:server_metrics` 与
  `tools/server/server-task.cpp:server_task_result_metrics::to_metrics` 分别维护累计 counter、即时 gauge
  和窗口 bucket，并在 `/metrics` 才转换成 Prometheus 文本；
- 锁定 vLLM `808f8cd3ac6751278d8b7be97f6fde5295ca4653` 的
  `vllm/v1/metrics/loggers.py:PrometheusStatLogger.record` 按 scheduler iteration、token、request 和 KV block
  eviction 更新 counter/gauge/histogram；锁定 SGLang
  `5bebe7a033ab57751fac3c9d8db219da62ca8c72` 的
  `python/sglang/srt/observability/metrics_collector.py:SchedulerMetricsCollector` 以同三类原语记录 queue、
  token、cache、latency、transfer 和 storage 事实。三条独立生产旅程证明基础 producer/aggregation
  vocabulary 是通用 Obs 能力，不是模型、backend、HTTP 或 benchmark 形状；
- HYLE Obs v4 只有同步 structured log 与编译期 trace。把每个数值编码成 `LogRecord` 会支付 fan-out、
  format/sink 生命周期，并丢失 counter 单调累积、gauge last-value 和 histogram bucket 语义；Tracy plot
  又只服务 profiling build，不能组成 production scrape snapshot。因此现有路线不能自然承载上述任务；
- 成熟实现反证边界：Folly
  `folly/ThreadCachedInt.h` 与 `folly/stats/Histogram.h` 分离 update/snapshot、线程语义和固定 bucket；
  推理项目也都把 metric producer 与 Prometheus/exporter、label/cardinality policy 分开。HYLE 只考虑
  无名称注册表、无 exporter、无后台线程、无隐式 clock 的数值原语；
- 受影响范围只含 `include/hyle/obs` 下新的 metrics public surface、Obs tests/docs/sample、selected
  component 的 source/static/shared header delivery，以及新增路径的 P0/P1 成本裁决。现有 log registry、
  record layout、Tracy route、`core/error/time` 公共依赖和 private `mem_reclaim` 路线默认不改；
- 明确不做 Prometheus/OpenTelemetry wire format、metric name/description/labels、动态 registry、时间窗口、
  exporter、网络/持久化、后台 flush、自动 task context、GPU event timing、模型/KV/scheduler policy、
  Native FFI/C API、稳定 C++ ABI、公共性能仪器或相邻模块重构；
- `CLOSED -> REOPENED` 前 registry snapshot：
  `C:\\public\\project-hyle\\_validation\\obs-v5-metrics-primitives\\transitions\\before-reopened.json`，
  SHA-256 `8473C0DC32E5459DBE421603D04DB45A34EEB34E7F66BEBD866EC356056CE167`。

## 初步停止边界

- API 必须按 metric 语义命名并让 ownership、并发和 snapshot 一致性可见；不能用 `fast/light` 或
  benchmark 名称制造两套含糊入口；
- 普通更新不得分配、加锁、读取时钟、查名字/标签或调用 log sink；bucket 查找的复杂度必须显式；
- exporter 与 label/cardinality 生命周期继续由宿主拥有。若审计证明基本数值原语也必须依赖动态
  registry、Container、Mem 或异步 runtime，本轮停止并重新裁决 owner，不顺手扩大依赖图；
- 原 Obs v4 日志 P1 合同只做未受影响回放；新指标性能只比较同语义、同线程安全、同 snapshot 保证的
  路线，不能把 thread-confined plain integer 与 concurrent exact atomic 当成同合同 peer。

## Closure card（`SCOPED`）

- component / owner / round / kind：`obs` / model-independent metrics aggregation /
  `obs-v5-metrics-primitives` / `REOPEN`；baseline
  `c51599be2fdcd4caeacc5b05aa0f61f725c62491`，trigger
  `OBS-METRICS-PRIMITIVES-2026-09-06`；
- real consumer / normal workload：一个推理 scheduler owner 在每次 iteration 结束时累计 request/token/
  eviction counter，写入 queue/cache gauge，并把 batch、queue 与 decode latency 放入预先固定的 bucket；
  exporter 在 scrape 边界读取 caller-owned snapshot。多 worker 可以各自保有 owner-thread state，也可以
  显式选择共享 concurrent state；本轮不拥有名字、标签、HTTP 或 scrape cadence；
- scope：加入 model-independent owner-thread 与 concurrent counter/gauge/fixed-boundary histogram，
  包含构造/校验、update、snapshot、显式 quiescent reset、bucket 边界与 tail 语义；更新 public header、
  correctness/concurrency tests、Obs package sample、architecture/user guide 与 selected component 的
  source/static/shared header delivery。现有 logger、trace、record、sink 与 registry 产品字节默认不改；
- non-goals：metric names/descriptions/labels/cardinality registry、Prometheus/OpenTelemetry wire format、
  exporter、时间窗口/quantile sketch、后台线程、网络/持久化、自动 context、GPU event、模型/KV/scheduler
  policy、跨进程共享、Native FFI/C API、稳定 C++ ABI、相邻模块或公共性能仪器改造及 Formal 认证；
- dependencies / direct consumers：公开依赖保持 `core,error,time`，private `mem_reclaim`；新 metrics 头只
  需要 C++23 standard atomics/arrays/algorithms 与现有 Error 构造失败形状，不新增 Container/Mem/SIMD/
  Platform 边。当前具名消费者是未来 HYLE inference scheduler/serving adapter；现有 package sample 是
  交付 consumer，不伪装成生产 trace；
- reuse / acceleration candidates：C++23 `std::atomic`、`std::array`、`std::lower_bound` 为机制路线；
  Folly `65749da412d92fe27d87ae45a8ef670720712858` 的 `ThreadCachedInt` 与 `stats/Histogram` 只用于验证
  concurrency/snapshot 与 fixed-bucket 语义档位。当前 Container、Mem、reclamation、clock、SIMD/hash/
  text 对零分配数值更新均不等价或不适用；不得为了“复用”增加依赖；
- tier / margin：整体 Obs 保持 `P1` / 5%。新 metrics 先做 P0 复杂度、资源与 release codegen 门，再用一个
  完整 scheduler-iteration batch（counter/gauge/histogram 混合更新）对直接 standard mechanism 做
  `SAME_CONTRACT` P1 abstraction-overhead 门；owner-thread 与 concurrent 分开比较。原 Obs v4 三个冻结
  logger journey 只做产品未触碰与必要 smoke，不重新解释旧样本；
- resource contract：owner-thread counter/gauge 是 O(1) plain scalar；concurrent 对应每次一个 relaxed
  atomic update。任意固定边界 histogram update 为 O(log B)，一次 bucket count 加一次 sum 加；snapshot
  为 O(B)，concurrent snapshot 在并发 writer 下弱一致，在 quiescence 后精确。构造校验 O(B)；构造后
  update 不 allocation、不 mutex、不 clock、不 syscall、不名字/标签 lookup；reset 只用于 owner-thread
  或外部 quiescent 生命周期；
- budgets：capability/API audit 45 分钟，implementation/correctness/docs 90 分钟，delivery/P1 60 分钟，
  red team 30 分钟；最多一次资格化、一次机制/codegen 诊断、一个产品修正和一次重跑。稳定 material
  FAIL 或公共仪器不能裁决时本轮 `BLOCKED`，不增加第三种 benchmark 姿势；
- machines / instruments：`W1-WIN`、`W1-WSL`、`M1-MAC`；correctness/P0 使用
  `closure-correctness-v1`，P1 使用 `closure-perf-v1`。W1-WIN 裁决数值，WSL/macOS 负责适用的
  correctness、sanitizer 和交付；
- locked consumer evidence：llama.cpp `73a43d1f69345aee8bb186ef4b3172cef892f2e5`（bundled ggml tree
  `660745f60c65f44e39f8fc81f7a2fd3e29451a0d`）、standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`、ds4
  `9ab705347c1775e7599ede7eb81a6255ec7dccb5`、vLLM
  `808f8cd3ac6751278d8b7be97f6fde5295ca4653` 与 SGLang
  `5bebe7a033ab57751fac3c9d8db219da62ca8c72`，2026-09-06 均 clean；standalone ggml/ds4 没有独立
  production metrics owner，作为负空间而不是强行造需求；
- API archetype provenance：counter/gauge/histogram 家族来自三个 `REAL_CONSUMER`；owner-thread 与
  concurrent 档位由这些 scheduler/worker 生命周期及 Folly `MATURE_PEER` 共同支持。公开命名必须表达
  metric 与 concurrency 语义；不出现 `fast/light`、Prometheus 名称或 benchmark case；
- correctness gates：zero/negative/large gauge、counter/bucket boundary、tail、invalid/duplicate/NaN
  boundary、sum/count、move/reset lifecycle、并发 exact count、quiescent snapshot、弱一致声明、public
  header self-contained、compile-only misuse、WSL ASan/UBSan/TSan；不适用的 cancel/ISA/backend 必须
  明确写 N/A；
- delivery gates：selected Obs product 与 factor；source embedding、relocated installed static/shared
  C++ consumer；Windows/WSL/macOS；Trace OFF 与受影响最小 smoke；selected header inventory 与 C/FFI
  清洁度；
- outputs / stop：capability map、public trace、有限 `FIX`、冻结 API、实现/tests/sample、四类文档、
  candidate、三平台与三交付 manifest、公平 P1、七问、红队、LOC、terminal receipt 与新 FROZEN marker。
  仅当全部 `MUST` COMPLETE/ROUTED 且各适用门 PASS 才 `CLOSED`；否则具名 `BLOCKED`，不扩展 exporter
  或相邻模块。

当前进入 `SCOPED`；closure card 的范围、语义档位、资源账单、预算、机器、仪器和停止条件冻结。产品
实现仍须等待 capability/public-surface/reuse 审计和 API freeze 在 `AUDITED` 完成。

`REOPENED -> SCOPED` 前 registry snapshot：
`C:\\public\\project-hyle\\_validation\\obs-v5-metrics-primitives\\transitions\\before-scoped.json`，
SHA-256 `14F19674E9C68544EA108DE069F621ABE3450DA0CB2B1CBEE3B29A1B72AA9BA3`。
