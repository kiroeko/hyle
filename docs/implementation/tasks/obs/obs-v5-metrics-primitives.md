# Obs v5：推理运行时指标聚合原语

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由已冻结模块的真实推理消费者回放触发。closure card 已冻结；本页继续保存能力账本、公开面
> 反向追踪和有限 `FIX`。只有进入 `AUDITED` 后才修改 Obs 产品源码。

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

## Capability-family coverage map（`AUDITED`）

| Family | admitted normal journey | required semantics | current route | coverage | decision |
|---|---|---|---|---|---|
| synchronous structured log | caller emits borrowed typed event to one or more host sinks | level gate, typed fields, source/time, quiescent registration, recursion safety | Obs v4 `log.h`/`log.cc` | `COMPLETE` | `KEEP` unchanged |
| call-site diagnostic control | suppress repeated warning/progress without formatting suppressed records | approximate/exact count, first-N, power-of-two, caller- or clock-driven interval | Obs v4 `log_control.h` | `COMPLETE` | `KEEP` unchanged |
| profiling timeline | build-selectable zone/message/fiber/frame/memory/counter instrumentation | OFF zero evaluation; ON one process/image backend contract | native Tracy seam | `BORROWED` | `KEEP` unchanged |
| owner-thread counter | scheduler/worker accumulates request, token, byte, FLOP or duration totals | non-negative unsigned/floating delta, O(1), no synchronization or allocation | repeated plain fields in consumers | `MISSING` | `FIX` |
| owner-thread gauge | scheduler publishes queue/cache/utilization/current state | last value; arithmetic values including floating NaN sentinel; O(1) | repeated plain fields in consumers | `MISSING` | `FIX` |
| owner-thread fixed histogram | observe batch/latency/lifetime into fixed inclusive upper bounds | validated increasing finite bounds, implicit +Inf tail, repeated observation, sum and exclusive buckets | ad-hoc bucket arrays | `MISSING` | `FIX` |
| shared metric state | several workers deliberately update one aggregation object | lock-free relaxed update; weakly consistent running snapshot and exact quiescent snapshot | direct standard atomics | `MISSING` | `FIX` as explicit `Concurrent*` types |
| snapshot composition | exporter combines per-worker states without forcing a contended hot object | same-bound snapshot merge and total-count derivation, O(B) | consumer loops | `MISSING` | `FIX` |
| bucket percentile estimate | serving control/diagnostic finds a bucket containing p99 or another quantile | explicit bucket-upper-bound estimate; empty/invalid input distinguished | SGLang scans bucket counts; Folly histogram percentile | `MISSING` | `FIX` on snapshot |
| names, labels and dynamic registry | exporter owns schema/cardinality and metric discovery | owning strings, lifecycle, conflict/cardinality policy | host Prometheus/OpenTelemetry adapter | `ROUTED` | no HYLE product API |
| exporter, scrape cadence and time windows | host publishes/rotates metrics | protocol, background/network/time-window policy | application/runtime | `ROUTED` | `OUT_OF_SCOPE` |
| approximate quantile sketch | unbounded/distribution-dependent quantiles | error-budget and merge policy | application or specialized statistics library | `ROUTED` | fixed histogram is not a sketch |

### Public-surface reverse trace and duplicate scan

- `LogLevel`、`LogValue`、`LogAttribute`、`LogRecord`、`LogSink`、`SinkRegistration`、同步 emit/flush 和
  `HYLE_LOG_*` 宏都反向追到 v4 的 structured diagnostic journeys；本轮不改它们的 signature、layout
  或 registry 实现。
- `LogEveryNState`、`LogEveryNExactState`、`LogFirstNState`、`LogEveryPow2State` 与
  `LogIntervalLimiter` 是**是否产生日志**的 admission state，不表达业务数值聚合；复用它们会混淆
  reset、并发和 snapshot 语义。
- `HYLE_TRACE_COUNTER` 只把时间点送进 profiling backend，Trace OFF 时不存在，也没有 production
  snapshot；它不能替代 counter/gauge。
- Mem 的 allocator/domain stats、Util 的引用计数和 Core fatal observer 都是各域控制状态，不是可复用的
  metric primitive。反向扫描没有发现另一个 HYLE generic counter/gauge/histogram；因此新增实现不会
  复制已有质因数能力，也不需要引入 Container、Mem、SIMD、Platform 或 scheduler。
- llama.cpp 的 `server_metrics` 证明 owner-held cumulative/window state；vLLM
  `PrometheusStatLogger.record` 证明每次 scheduler iteration 混合更新三类指标；SGLang
  `SchedulerMetricsCollector` 还证明 floating counter、NaN gauge、一次重复 N 个同值 histogram sample
  与读取 p99 bucket 的正常形状。Folly `ThreadCachedInt`/`stats::Histogram` 只校准并发档、固定桶、
  repeated value 和 percentile 语义，不成为产品依赖。
- standalone ggml 与 ds4 没有独立通用 production metric owner；这是负空间。模型/backend/KV policy
  仍由未来推理 runtime 拥有，不因此塞进 Obs。

## Capability / gap ledger and frozen `FIX` set

| ID | class | exact action | complexity/resource contract | verification / negative space |
|---|---|---|---|---|
| `OBS5-VALUE-01` | `MUST` | add owner-thread `Counter<T>` and `Gauge<T>` value objects | O(1), zero allocation/lock/clock/syscall; counter accepts unsigned integral or floating values, gauge accepts non-bool arithmetic values | integer/floating, reset, invalid counter delta, gauge NaN; no names/labels |
| `OBS5-HIST-02` | `MUST` | add validated `FixedBuckets<N>` and owner-thread `FixedHistogram<N>` | construction O(B); observe O(log B), one bucket update plus one sum; snapshot/reset O(B); no post-construction allocation | invalid/duplicate/NaN bounds, exact boundary, +Inf tail, zero/repeated observation |
| `OBS5-CONCURRENT-03` | `MUST` | add explicit lock-free `ConcurrentCounter<T>`、`ConcurrentGauge<T>` and `ConcurrentFixedHistogram<N>` | relaxed atomics only; counter/common gauge update one RMW where arithmetic permits, checked integral gauge update may CAS; histogram one bucket RMW plus one sum RMW | exact quiescent totals, concurrent stress, weak-running-snapshot documentation; no hidden mutex |
| `OBS5-SNAPSHOT-04` | `MUST` | add copyable `HistogramSnapshot<N>` merge and quantile-bucket upper-bound query | merge/query O(B), zero allocation; merge rejects different bounds or unrepresentable bucket totals | per-worker merge, empty/invalid quantile, tail +Inf; no sketch accuracy claim |
| `OBS5-DELIVERY-05` | `MUST` | deliver `metrics.h` in selected source/static/shared Obs and exercise the natural package sample | header-only metric data plane, unchanged compiled Obs runtime and component graph | self-contained header, factor/product and relocated consumers; no C/FFI symbols |
| `OBS5-PERF-06` | `MUST` | qualify one full scheduler-iteration mixed update for each ownership grade against direct same-contract standard mechanisms | `P1_ENGINEERING`, 5%; no comparison between plain and atomic grades; P0 confirms release code/resource shape | one qualified run and at most one codegen diagnosis; old logging journeys only regression smoke |
| `OBS5-DOC-07` | `MUST` | update architecture, user selection/traps, package sample, P1 evidence and terminal record | separate production metrics from log/trace/exporter and state concurrency/lifecycle exactly | four document classes remain consistent |

The finite product batch is exactly `OBS5-VALUE-01..DOC-07`. It cannot add a registry, exporter, label/name storage,
window/sketch, background work, model policy, component edge, Native FFI, common-instrument rewrite or adjacent-module
change.

## API / usability freeze

- status：`FROZEN`；new public surface lives only in `<hyle/obs/metrics.h>` under `hyle::obs`. The normal path is
  `Counter<T = std::uint64_t>`、`Gauge<T = double>`、`FixedBuckets<N>`、`FixedHistogram<N>`；shared state is
  opt-in through the visibly named `ConcurrentCounter`、`ConcurrentGauge` and `ConcurrentFixedHistogram`.
- `Counter` accepts unsigned integral or IEEE floating storage. `add(delta)` requires a non-negative finite floating
  delta; default delta is one. A representable-lifetime precondition remains explicit rather than adding saturation,
  hidden sharding or wider metadata. `value()` observes and `reset()` begins a caller-controlled lifecycle.
- `Gauge` accepts standard non-`bool` arithmetic values. `set/value/reset` cover last-value publication; `add` and
  `subtract` provide natural inc/dec composition. Floating values deliberately preserve IEEE NaN/infinity because
  SGLang uses NaN as an unavailable gauge sentinel. Integral arithmetic checks representability instead of invoking
  signed overflow or silently wrapping.
- `FixedBuckets<N>::create(std::array<double,N>) -> Result<FixedBuckets<N>>` is the only raw-boundary constructor.
  Bounds must be finite and strictly increasing. Histogram buckets are exclusive counters selected by the first
  inclusive upper bound, plus an implicit +Inf tail. `observe(value, multiplicity)` returns false and changes nothing
  for non-finite samples; multiplicity zero is a successful no-op. It updates one bucket and the sum once.
- `snapshot()` returns a copyable `HistogramSnapshot<N>` with the exact finite bounds, `N+1` exclusive counts and
  sum; total count is derived. `merge(other)` requires identical bounds and fails recoverably on mismatch/count
  overflow. `quantile_upper_bound(q)` returns the containing bucket upper bound, including +Inf for the tail, and
  fails for empty data or q outside `[0,1]`; it is a fixed-bucket estimate, not an exact sample quantile.
- local objects are thread-compatible. Concurrent objects require a target where their selected `std::atomic<T>` is
  always lock-free, use relaxed operations, and do not synchronize application state. A running histogram snapshot
  may combine nearby update instants; after writers quiesce it is exact. Reset/move/destruction are lifecycle actions
  performed after quiescence. High-contention consumers should keep one local object per worker then merge snapshots;
  this is not hidden TLS registration and does not claim the stronger immediate-read contract beats Folly sharding.
- no metric type stores a name, label, timestamp, exporter pointer or registry membership. Existing log/trace public
  surface remains byte-for-byte outside this header. Freeze identity is this `AUDITED` ledger; implementation may vary
  only while exact semantics, complexity, resource cost and selected-header delivery remain unchanged.

## Owner 七问直答（`AUDITED`）

1. **Third-party product path:** existing Trace may borrow locked Tracy v0.14.1; the new metric path uses only C++23
   arrays/algorithms/atomics plus frozen Core/Error contracts. Folly and inference repositories are evidence only.
2. **Inference capability:** admitted production counter/gauge/fixed-histogram, repeated observation, worker merge and
   bucket-percentile journeys are covered by the finite FIX set. Export/schema/model policy remains correctly routed.
3. **Internal reuse/acceleration:** duplicate scan found no generic HYLE metric primitive. Core contracts and Error's
   recoverable factory/merge failures are reused; Container/Mem/SIMD/Platform/Time add no advantage to a fixed numeric
   update and would make the factor graph worse.
4. **Performance:** optimal admitted classes are O(1) scalar/atomic value updates, O(log B) histogram update and O(B)
   snapshot/merge/query. P1 will compare identical ownership and consistency contracts; no SOTA or end-to-end claim
   exists before qualification.
5. **C ABI/FFI:** Obs remains C++-only. No C header, dynamic C symbol, type-erased FFI handle or compatibility bridge is
   admitted; header templates do not pretend to be cross-language ABI.
6. **Build/delivery:** central graph and real target remain `obs PUBLIC core,error,time; PRIVATE mem_reclaim`.
   `metrics.h` adds no edge and must ship in source/static/shared selected Obs headers; validation is pending.
7. **C++ usability/docs:** names expose metric and concurrency semantics, common scheduler code needs no registry or
   exporter, and batch observe/worker merge avoid forced hot contention. Natural sample, choices and traps remain in
   the FIX set and are pending implementation.

`SCOPED -> AUDITED` 的前态快照保存在 canonical checkout 外：
`C:\\public\\project-hyle\\_validation\\obs-v5-metrics-primitives\\transitions\\before-audited.json`，SHA-256
`E59698C0B1928AADB9A71AC5F8F9354852AB293458199AF797908CA1AB3343C1`。

## 有限实现批次（`AUDITED`）

- `OBS5-VALUE-01..SNAPSHOT-04` 已实现于单一 header `<hyle/obs/metrics.h>`：owner-thread 与
  `Concurrent*` 档位不共享隐藏 registry/TLS，fixed bounds 在构造时验证，update/snapshot/merge/
  bucket-quantile 均遵守冻结语义；
- `OBS5-DELIVERY-05` 已把自然 metrics journey 接入 Obs package sample。W1-WIN / MSVC 19.50 的
  selected `obs` source embedding、relocated shared SDK 与 relocated static SDK consumer 均通过，安装
  header inventory 包含 `metrics.h`；
- correctness 覆盖 value/reset、invalid/duplicate/non-finite bounds、inclusive boundary、`+Inf` tail、
  multiplicity、merge 原子失败、empty/invalid quantile，以及四 writer 停止后的 concurrent exact state。
  W1-WIN fresh factor build 的四项 Obs tests（header self-contained、log、compiled-level、metrics）通过；
- P0 源码和 Release object 复核确认 owner path 只含 scalar update/binary search，shared path 只含匹配的
  relaxed atomic RMW/store/CAS；没有 allocation、mutex、clock、syscall、name/label lookup 或 metrics
  runtime symbol。初始实现曾让 MSVC 把 floating validation helper 留成 out-of-line call，形成真实
  10–13% wrapper cost；将该 API-boundary helper 设为强制内联后，正常路径恢复为直接比较/算术。这个
  改动修复 zero-overhead abstraction，而未改变 workload、公开 shape 或常态 metadata；
- `OBS5-PERF-06` 使用 `obs-metrics-p1-v1`：每次完整 scheduler iteration 更新 64 个 request 的 counters、
  gauges 与六个 runtime buckets；owner-thread 与 concurrent 分开和完全相同的 direct C++23
  scalar/relaxed-atomic mechanism 对比。两侧执行相同输入验证、runtime binary search、数值更新、
  ownership/consistency 和计费边界；Folly checkout 只锁定成熟语义来源，不冒充实际计时 API peer。
  semantic oracle 同时核对 owner/shared state，primary/placement artifact 必须不同；正式 P1 资格仍待
  clean immutable candidate；
- architecture/user guide 已增加三通路选型、local-first/shared opt-in、exclusive bucket、弱 snapshot、
  quiescent reset、overflow/lifetime 与 exporter/cardinality 边界。Native FFI/C API 仍为 `N/A`，中央
  component graph 和 compiled Obs runtime 不变。

本节只是候选前实现记录，未把 registry 状态越级写成 `CANDIDATE/VERIFIED`。candidate identity、三平台
sanitizer/交付、资格化 P1、红队结论与 terminal receipt 在后续合法状态转换中补入。

## 不可变候选（`CANDIDATE`）

- product commit：`47a9e03a6bfd4d2d299d26d1d32739bb84541166`；tree：
  `093836f0791f304deb521e86be8ba5a9c94b5913`；
- 冻结内容：`OBS5-VALUE-01..DOC-07` 的公开 API、算法/资源合同、consumer shapes、P1 contract、
  semantic validator、tests/sample 与 architecture/user/performance 文档。后续若改变这些产品事实，必须
  退回 `AUDITED`，不能在验证阶段偷偷调 workload 或 shape；
- `AUDITED -> CANDIDATE` 前 registry snapshot：
  `C:\\public\\project-hyle\\_validation\\obs-v5-metrics-primitives\\transitions\\before-candidate.json`，
  SHA-256 `4783F4F24F43E9D1883865750CBE6F05FC9B98539A06C4C2B268169EBDE3D6FA`。

registry 现进入 `CANDIDATE`。前一节的“候选前”措辞保留为发生顺序事实；当前状态只以 registry 为准。
