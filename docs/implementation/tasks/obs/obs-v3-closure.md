# obs v3 consumer-driven closure

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> This is the active `RESUME` record for `obs-v3`. The current product state is
> [`module-status.json`](../../module-status.json). The terminal
> [`obs-v2`](obs-v2-resume.md) remains historical evidence and is not edited by this round.

## Closure card (`CLOSED`)

- component / owner / round ID: `obs` / observability owner / `obs-v3`; task kind: `RESUME`.
- previous round / blocker / resume condition: `obs-v2` stopped at `BLOCKED` because a synthetic
  8-producer x 4/16-sink replay had no qualified product verdict. Its resume condition required a
  named inference consumer to establish the real fan-out and KPI, or a material product mechanism.
- owner rescope evidence: on 2026-08-29 the owner admitted two real product journeys: ordinary
  Abseil-shaped call-site control and llama.cpp-shaped long-running inference progress suppression.
  Local llama.cpp revision `7584430716ee229751771ed0d6bbcb780d105eeb` registers one process-global
  callback and performs application fan-out behind that adapter; its server progress path throttles
  reports to an approximately three-second cadence. Consequently one host adapter is the normal
  product shape; 4/16 independent atomic sinks remain capacity/contention diagnostics, not a normal
  inference performance gate.
- baseline commit: `1028b056501c4fc07e45924c5da8b8222cd86414`; the unchanged pre-round product
  candidate `ff925c0053e738fb48d98993e42750b593f3db22` is audit input, not the v3 candidate.
- demand / defect IDs:
  - `OBS-V3-CALLSITE-01`: common logging needs conditional logging, first-N, every-N,
    power-of-two and time-interval suppression without evaluating format arguments when suppressed;
  - `OBS-V3-INFERENCE-01`: a request/slot-owned interval gate is needed for long-running model load,
    decode progress and repeated-warning paths whose lifetime is not a single source expansion;
  - `OBS-V3-GUIDE-01`: installing a host logger adapter without disabling HYLE's default stderr sink
    duplicates output, but the current guide does not show the hand-off;
  - `OBS-V3-PERF-01`: replace the unsupported 4/16-sink product provenance with ordinary one-adapter
    and inference progress workloads while retaining high fan-out only as a non-gating diagnostic.
- real consumer / normal workload:
  - ordinary C++ service: a level-gated format call, one synchronous host adapter, repeated warning or
    progress call sites using count/time suppression;
  - inference server: token/decode scheduling repeatedly checks request-owned progress state, emits
    only at its interval boundary, and sends the resulting structured record through one application
    logger adapter. The consumer sources are llama.cpp `common/log.h` SHA-256
    `cd065a0799ffc9a8adcb96f5188f77aec139e2520cb76316bf091ca289bb5bcb` and
    `tools/server/server-context.cpp` SHA-256
    `a79cf9e1d4a8d7c1f0ee608aa781628db403e8f59e25e731f997d0952d230e47`.
- scope: add a compact, composable call-site-control surface and an explicit object-owned interval
  limiter; preserve compile/runtime level gating, borrowed records and the synchronous adapter;
  correct default-sink hand-off guidance; add correctness tests and one bounded P1 contract covering
  ordinary and inference shapes; update architecture, user, performance and task evidence.
- non-goals: category/config trees, async workers/queues, file rotation, network export, metrics/OTLP,
  fatal control flow, owning log records, printf/stream APIs, copying every Abseil/Folly macro
  permutation, changing registry layout/quiescence, treating 4/16 sinks as a product gate, Native
  FFI/C API, stable C++ ABI, downstream engine implementation and Formal certification.
- direct dependencies / consumers: retain `obs PUBLIC core error`; standard-library atomics and
  `steady_clock` are platform contracts, not third-party modules. The audit must reject a new component
  dependency unless it materially improves the admitted consumer. Trace consumers remain
  mem/sched/io/compute; runtime log output remains application-owned.
- tier and reason: `P1`; suppression is allowed inside repeated service/inference loops and the
  existing one-adapter dispatch is a runtime primitive. Material peer non-inferiority margin is 5%.
- normal performance/resource budgets:
  - compile/runtime-disabled calls do not evaluate conditions, limiter parameters or format arguments;
  - approximate count and exact every-N operations are O(1); first-N and interval admission are O(1) on
    the normal uncontended path and use relaxed CAS retry under contention, so per-call worst-case retry
    count is not bounded. HYLE adds no product mutex; the target standard library owns the atomic progress
    guarantee. All limiter operations are allocation-free and syscall-free when caller time is supplied;
    count suppression performs no stronger synchronization than its documented semantics;
  - a 4096-check, 16-slot inference progress replay is budgeted at <=100 microseconds total on W1-WIN
    (<=25 ns/check), excluding the deliberately emitted adapter records;
  - clock-owning interval convenience is budgeted at <=100 ns/suppressed check on W1-WIN;
  - retained one-adapter preformatted and formatted paths keep the existing <=100 ns and <=150 ns
    absolute budgets and 5% immutable-baseline non-inferiority where the contract is unchanged.
  Comparable call-site rows may claim only 5% non-inferiority to a bound peer; consumer-only rows may
  claim only their absolute/resource budget, not SOTA.
- total investigation / performance budget: four hours total, at most 75 minutes of P1 sampling and
  mechanism inspection. A larger design must split into a later task rather than extend this round.
- maximum experiments / fixes: one diagnostic probe, one codegen/complexity review, one bounded
  product implementation and one fresh qualification. A material failure gets one mechanism-directed
  correction and one rerun; otherwise stop `BLOCKED`.
- machine / instrument: W1-WIN / `closure-perf-v1` for P1; W1-WIN, W1-WSL and M1-MAC for applicable
  correctness and source/static/shared delivery. No cross-platform nanosecond pooling.
- strong peers:
  - Abseil `81c993af4fb88b3a51829a796df02251fa27102f`, `absl/log/log.h` and
    `absl/log/internal/conditions.{h,cc}`, for call-site condition/count/time semantics;
  - Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`, `folly/logging/xlog.h` and
    `folly/logging/RateLimiter.{h,cpp}`, for approximate/exact count trade-offs and explicit interval
    limiting. llama.cpp is the named consumer, not a performance peer.
- correctness gates: normal and suppressed calls; zero/one/large count boundaries and wrap policy;
  interval boundary/reset/clock regression policy; argument non-evaluation; independent explicit
  states; concurrency semantics; existing registry recursion/quiescence/default-sink behavior;
  public-header self-containment; trace OFF/ON regression. OOM/cancellation are N/A only if the final
  limiter remains fixed-state and synchronous.
- delivery gates: selected-component source embedding plus installed static/shared C++23 consumers on
  the applicable platforms; no new public C/FFI artifact, dependency leak or package option.
- performance gates: ordinary call-site control against a same-contract mature peer when comparable;
  deterministic inference progress against the frozen consumer budget; retained one-adapter dispatch
  regression and direct Tracy seam identity. 2/4/16-sink rows may be reported diagnostically but cannot
  decide the v3 product verdict.
- expected outputs: capability-family map, complete public-surface trace, finite `FIX` ledger, natural
  C++ API and guide, focused tests, frozen product candidate, qualified P1 evidence, owner six answers,
  terminal receipt and four LOC figures.
- stop / acceptable limits: close only with all admitted `MUST` capabilities complete or routed and all
  applicable gates passing. Stop `BLOCKED` on a reproducible material performance/correctness defect,
  an unavailable required platform/delivery journey, or an exhausted qualified performance contract;
  do not add macro families, logger services or benchmark shapes after audit merely to seek PASS.

The explicit pre-transition registry snapshot is
`C:\hyle-scratch\obs-v3\transitions\module-status-before-scoped.json`, SHA-256
`A04EC4868CC452C989855196081892653FE4751C4779AC9C2AC8271FC9240437`.

## Pre-audit inputs

These are findings to classify before `AUDITED`; they are not yet an implementation authorization.

| ID | observed fact | required audit decision |
|---|---|---|
| `OBS-V3-CALLSITE-01` | Abseil and Folly repeatedly expose conditional, every-N, first-N, power-of-two and interval controls; HYLE exposes only severity macros | freeze the smallest composable public surface and exact concurrency semantics |
| `OBS-V3-INFERENCE-01` | llama.cpp uses one process callback plus manually throttled per-slot progress; a hidden call-site static cannot represent multiple slot lifetimes | decide the explicit state type, injected-time seam and reset/lifetime contract |
| `OBS-V3-GUIDE-01` | `SinkRegistration::create(host)` leaves the default stderr membership enabled | fix the host hand-off journey and rollback/error handling |
| `OBS-V3-PERF-01` | old 4/16 atomic-sink provenance is self-referential; one callback plus an application logger is the named consumer topology | freeze new product workloads and reclassify high fan-out as diagnostic |
| `OBS-V3-DEPENDENCY-01` | time-based control can use the standard steady clock or pull the Time component and its public closure | retain a minimal dependency only if semantics and consumer cost justify it |

## Capability-family coverage map

| Family | admitted journeys / evidence | required variants and boundaries | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| level and conditional gating | ordinary debug/diagnostic call; llama macros avoid expensive argument evaluation | TU compile floor, process runtime level, explicit condition; disabled paths evaluate no condition/format/control argument | `HYLE_LOG_<LEVEL>` and `_IF`; floor/runtime short-circuit tests | `COMPLETE` | combined condition/frequency permutations compose with a normal `if` rather than multiplying macros |
| call-site repetition control | repeated warning/progress/error sites; Abseil/Folly consensus | first + every N, exact every N where cardinality matters, first N, powers of two, chrono interval; one state per expansion; thread-safe semantics must be named | header-only state objects plus all-level macro wrappers | `COMPLETE` | approximate count is intentional under contention; exact spelling remains explicit |
| request/slot progress control | llama.cpp slot timing report every approximately 3 seconds | object-owned state, injected steady time to share/hoist clock reads, convenience clock read, reset, independent lifetimes, exact at-most-one interval admission | `LogIntervalLimiter` raw-ns/chrono overloads | `COMPLETE` | consumer owns state lifetime and supplies only monotonic time |
| record production | preformatted bytes, structured fields, source metadata, fixed-buffer formatting | exact borrowed slices, optional timestamp, truncation/error policy, no allocation in native emit | `LogRecord`, `LogField`, `log_record`, `log_message`, `log_fmt*`, base macros | `COMPLETE` | owning async records remain adapter-owned |
| sink adapter and lifecycle | one process adapter into a host/application logger; optional default CLI stderr | typed/raw adapter, synchronous concurrent callback, flush, quiescent RAII unregister, recursion policy, default takeover | `LogSink`, `SinkRegistration`, registry/default-sink functions and takeover/rollback guide | `COMPLETE` | async ownership and host fan-out remain behind the adapter |
| profiling trace | HYLE hot paths and engine profiling build | OFF no-evaluation; ON Tracy zone/counter/fiber/frame/message/allocation/thread seams | `trace.h` macro seam | `COMPLETE` | viewer/storage and GPU policy remain outside Obs |
| logger service / telemetry backends | async file/network logging, metrics and OTLP | owning queue, backpressure, persistence, category/config, aggregation | application adapter / future named module | `ROUTED` | must not add ownership or category metadata to every base record |
| delivery and foreign boundary | C++ source/static/shared consumers | selected-component headers/target closure; no C schema/bridge/export | build/package system; Native FFI global policy `N/A` | `COMPLETE but affected` | candidate verification must re-run the three delivery journeys |

## Capability / gap ledger

| ID | class / demand | exact source and normal shape | owner / concurrency / complexity | state | decision / verification |
|---|---|---|---|---|---|
| `OBS-V3-CALLSITE-01` | `MUST` peer consensus | Abseil `LOG_IF`, every/first/pow2/time family; Folly approximate and exact count variants; ordinary repeated call site | Obs public header; one static state per expansion; normal path O(1), no heap/product mutex/syscall except the selected clock read | `COMPLETE` | `DONE`: all-level conditional/stateful macros preserve source location and short-circuit ordering |
| `OBS-V3-COUNT-SEM-01` | `MUST` semantic split | logging frequency normally tolerates lossy counts, while cardinality-sensitive users need exact admission | `LogEveryNState` and `LogEveryPow2State` use relaxed load/store and are explicitly approximate under contention; `LogEveryNExactState` uses relaxed RMW; `LogFirstNState` uses relaxed CAS retry and never admits more than N; its per-call retry count is contention-dependent and the standard library owns the atomic progress guarantee | `COMPLETE` | `DONE`: zero/negative N, reset/move, sequence and exact eight-thread cardinality tests pass |
| `OBS-V3-INTERVAL-01` | `MUST` llama consumer | each server slot owns last progress state and checks a 3-second steady interval | `LogIntervalLimiter`; exact relaxed-CAS admission, first check passes, duration <=0 passes, overflow-safe difference; normal uncontended path is O(1), while contended per-call retry is not bounded; caller-supplied steady time avoids duplicate clock reads | `COMPLETE` | `DONE`: raw/chrono boundary, reset, rollback, unsigned maximum and exact concurrency tests pass |
| `OBS-V3-MACRO-SURFACE-01` | `MUST` C++ usability | all five existing severities need the same discoverable operation order | `HYLE_LOG_<LEVEL>_IF`, `_EVERY_N`, `_EVERY_N_EXACT`, `_FIRST_N`, `_EVERY_POW_2`, `_EVERY`; compiled floor and runtime level precede condition/state/format evaluation | `COMPLETE` | `DONE`: one internal implementation family and thin wrappers; no combined permutation matrix |
| `OBS-V3-GUIDE-01` | `MUST` reproducible defect | host registers one adapter while default stderr remains a registry member | caller owns takeover sequence and error rollback | `COMPLETE` | `DONE`: guide covers registration, default disable, rollback, restoration and producer lifetime order |
| `OBS-V3-PERF-01` | `MUST` valid evidence | ordinary one-adapter log plus count suppression; 16 independent inference slots with deterministic injected time and one adapter | P1 W1-WIN; same semantics/resource accounting; 5% when comparable, otherwise frozen absolute/resource budget | `PARTIAL` | `IMPLEMENTED / VERIFY`: `obs-control-p1-v1` binds Abseil and llama.cpp; limiter-only sub-ns row is diagnostic, full one-adapter inference is the product gate |
| `OBS-V3-DEPENDENCY-01` | `MUST` modularity | standard `steady_clock` is sufficient; Time aliases that clock and exposes the same monotonic-ns domain | retain `obs PUBLIC core error`; no Time/container-intrusive closure or third-party product dependency | `COMPLETE` | `KEEP`: chrono and raw-ns overloads compose with Time without adding it to the component closure |
| `OBS-V3-RECORD-01` | `MUST` existing surface | structured and formatted one-adapter records | synchronous borrowed lifetime, O(S), S<=64 | `COMPLETE` | `KEEP + regression`: no record/layout/registry change |
| `OBS-V3-TRACE-01` | `MUST` existing consumers | mem/sched/io/compute Tracy seam | compile-time OFF or direct Tracy ON | `COMPLETE` | `KEEP + focused regression`; no new trace capability |
| `OBS-V3-FFI-01` | `MUST` global policy | current consumers are C++ source/static/shared | no public C header, bridge, undecorated export or managed sample | `COMPLETE` | `KEEP + scan` |

- existing public-entry reverse trace:
  - record/data: `LogLevel`, `LogField`, `LogRecord`;
  - sink/lifecycle: `LogSinkBackend`, `LogSink::from`, add/remove/count/capacity/flush,
    `SinkRegistration`, default-sink queries/setter;
  - gate/emission: set/query/`should_log`, record/message overloads, `log_fmt_at`, `log_fmt`, five
    `HYLE_LOG_<LEVEL>` macros; `detail::log_record_unchecked` remains an implementation seam;
  - trace: zone annotation, plot/counter, fiber/frame, message, allocation and thread-name macros.
- syntax/semantic consumer scan: HYLE product modules emit no unsolicited runtime logs; mem/sched/io/compute
  use only the trace seam. This remains the correct library policy. Existing fatal/error/benchmark output is
  not silently migrated to logging.
- public/internal/higher-layer decision: control states and formatting/record production are public Obs;
  registry mechanics stay internal; async ownership, logger configuration, metrics and persistence stay in
  the application or a future admitted module.
- frozen `FIX` set: exactly `OBS-V3-CALLSITE-01`, `OBS-V3-COUNT-SEM-01`,
  `OBS-V3-INTERVAL-01`, `OBS-V3-MACRO-SURFACE-01`, `OBS-V3-GUIDE-01` and
  `OBS-V3-PERF-01`. Product implementation is limited to a header-only control layer and its composition
  into `log.h`; registry, record layout, compiled log implementation, trace backend, component graph and
  build-system policy are excluded. A discovered need outside this set returns the round to `SCOPED`.

The `SCOPED -> AUDITED` pre-transition snapshot is
`C:\hyle-scratch\obs-v3\transitions\module-status-before-audited.json`, SHA-256
`3A77EFDC1650DCA6FAFA6FBA27AE79C4072BB5AA3F1883DDBF72261B6B506AB5`.

## Owner six answers

1. **第三方产品路径**：optional Tracy v0.14.1 是唯一产品路径第三方且只在 trace ON 时进入；Abseil
   `81c993af`、Folly `423fd7f8` 和 llama.cpp `75844307` 只是 peer/consumer evidence。Obs 的直接组件
   closure 仍精确为 `PUBLIC core error`，没有新增 Time 或 logger backend 依赖。
2. **推理与普通能力**：普通调用点的 conditional/every-N exact-or-approx/first-N/power-of-two/time
   suppression，以及 request/slot-owned injected-time interval state 都已实现；原有 borrowed structured
   record、同步 adapter、quiescent lifecycle 和 native trace seam 保持完整。异步 ownership、metrics、
   category/config、存储/网络和模型策略明确路由到应用或未来有真实 consumer 的模块，不是遗漏的 `MUST`。
3. **拟真公平性能**：W1-WIN `closure-perf-v1` 的 ordinary one-adapter Abseil 行和完整 16-slot
   llama.cpp-shaped progress replay 全部通过冻结的 5% P1 non-inferiority gate；绝对预算也通过。只裁决
   consumer aggregate，不把亚纳秒 limiter-only 排名或旧 4/16-sink stress 外推为全操作 SOTA。
4. **旧 C/Native FFI**：`CLEAN / N/A`。public Obs 树没有 C header/schema/bridge、`extern "C"`、managed
   sample 或 FFI option；Windows shared candidate 仍是与 v2 完全相同的 24 个 decorated C++ exports，
   没有 undecorated export。
5. **source/static/shared**：selected-Obs source embedding、relocated installed static 和 relocated installed
   shared consumers 在 W1-WIN、W1-WSL、M1-MAC 全部通过；trace OFF/Tracy ON 也在三平台各通过 3/3。
6. **C++ 易用性与文档**：宏提供最短普通路径，显式 state 支持 request/slot 生命周期和 hoisted clock；
   编译/runtime gate 保证 suppression 参数不求值。guide 已覆盖选型、default-sink hand-off/rollback、
   borrowed lifetime、并发近似/精确语义、monotonic clock、reset/move 同步和 callback 陷阱。

## Candidate

- candidate commit / tree: `21c07d15c4eac477c7b9a768b6475850af87f463` /
  `3db40042b12c25a5f9cf562d92de42caa077c057`;
- product delta: new header-only count/interval state types and all-level conditional/sampling macros; no change
  to `LogRecord`, sink registry, `src/obs`, component graph, Tracy seam or binary exports;
- consumer/performance adapter: [`consumer_shapes_v1.json`](../../../../bench/obs/consumer_shapes_v1.json)
  binds ordinary and llama.cpp-shaped topology; contract `obs-control-p1-v1` lock is
  `e97c70dc41b63dfc0c3e238db746fccd15cba43f4ecc45c03f4ac6945116e9cc` and its benchmark semantic
  validator binds SHA-256 `d5725d8bf3b50adae62428461c1676379ad5b82b1dcbb4e94482c9385743e004`;
- pre-freeze W1-WIN evidence: fresh selected-Obs static MSVC 19.50 trace OFF and Tracy v0.14.1 trace ON each
  pass 3/3 focused tests. The one-sample contract probe passed product evidence, semantic validator,
  timer-floor and placement controls; as designed it is `DIAGNOSTIC`, not qualification evidence.
- probe classification: the exact-atomic limiter-only mechanism changes relative rank under function-layout
  perturbation at roughly 0.5 ns/check, while the complete 80-record one-adapter inference replay remains
  non-inferior in all four one-sample strata. The former is therefore frozen as `NOT_COMPARABLE` diagnostic
  and judged only against the <=25 ns/check absolute budget; it cannot mutate the API or decide closure.

The `AUDITED -> CANDIDATE` pre-transition snapshot is
`C:\hyle-scratch\obs-v3\transitions\module-status-before-candidate.json`, SHA-256
`B29538199D68EEAEA03A65664F600EA2B480B2F7C46F5FDE65DC7591FA72C131`.

## Verification manifest

验证绑定 immutable candidate `21c07d15c4eac477c7b9a768b6475850af87f463` / tree
`3db40042b12c25a5f9cf562d92de42caa077c057`。当前 round 在 candidate 后只改证据和说明，不改变
`include/hyle/obs`、`src/obs`、component graph、公开 layout 或 binary implementation。

`CANDIDATE -> VERIFIED` 的 pre-transition registry snapshot 是
`C:\hyle-scratch\obs-v3\transitions\module-status-before-verified.json`，SHA-256
`778322724D2E8C9A9D615B93176E7A61289523B66B38827321F31FD5586AF419`。

### Correctness risk matrix

| Risk | verdict | command / evidence | limit / N/A reason |
|---|---|---|---|
| normal | `PASS` | selected `hyle_obs_tests`, `hyle_obs_compiled_level_tests`, `hyle_trace_tests`；三平台 trace OFF/ON 各 3/3 | 覆盖普通 record、全部新宏和显式 state |
| empty / zero | `PASS` | zero/negative N、zero/negative duration、no-sink 与 disabled-level tests | interval<=0 按合同表示不限制 |
| capacity / boundary / overflow | `PASS` | first/every/pow2 sequence、registry 64-slot capacity、clock rollback、interval edge、`UINT64_MAX` duplicate tests | 64-bit count wrap 明确开始新序列 |
| OOM / system failure / invalid input | `PASS` | control state 固定 8 bytes 且不分配；invalid sink/status 与 formatter failure retained tests | control hot path 没有 OOM/cancellation surface |
| move / destroy / lifetime | `PASS` | state move/reset implementation review、count move test、RAII registration move/reset/quiescence tests；guide lifecycle contract | active state move/reset/destroy 需要 consumer 外部同步 |
| concurrency / reentrancy | `PASS` | 8-thread exact cardinality、parallel registry churn/remove-waits/self-remove/cross-remove/nested-log tests | approximate counter 的 lossy 语义是公开合同 |
| cancellation / close | `NOT_APPLICABLE` | synchronous fixed-state control and callback seam | no owned worker/request/cancellable operation |
| property / differential / stress | `PASS` | deterministic sequence tests、consumer semantic/resource validator、registry concurrency stress | no parser/untrusted byte grammar requiring fuzz |
| ASan / UBSan / TSan | `PASS` | W1-WSL 三个 fresh sanitizer build 各 selected 3/3；TSan 包含 8-thread exact states | sanitizer runs are correctness, not timing evidence |
| real backend | `PASS` | Tracy v0.14.1 trace ON on W1-WIN/W1-WSL/M1-MAC, each 3/3 | Tracy build warnings on macOS are upstream deprecated `sprintf` only |

### Delivery and aggregate gates

| Gate | verdict | evidence | limit |
|---|---|---|---|
| correctness / sanitizer | `PASS` | 上表；MSVC 19.50、GCC 14.2、AppleClang 21 | cross-platform point performance is not pooled |
| component / source / static / shared | `PASS` | W1-WIN/W1-WSL/M1-MAC selected-Obs source + relocated static + relocated shared consumers | C++23 delivery only |
| direct consumer | `PASS` | ordinary one-adapter call site and 16-slot/4096-check/80-record inference replay | application owns adapter fan-out and persistence |
| P1 engineering performance | `PASS` | `obs-control-p1-v1` qualified bundle and retained unchanged emit/Tracy evidence | Formal remains `PENDING` |
| usability / docs | `PASS` | architecture/guide/performance/task split plus `check-docs.py` | no stable C++ ABI or Native FFI promise |
| independent red team | `PASS` | below | only frozen v3 scope attacked |

Windows shared symbol inspection used MSVC `dumpbin /exports` on the relocated candidate DLL: 24 names, all
decorated C++, exactly equal to the pre-v3 selected Obs DLL. Public/implementation scans found no `extern "C"`,
Native FFI option, C header or new binary symbol. `cmake/HyleComponents.cmake` still declares
`_hyle_declare_component(obs TRACE PUBLIC_DEPS core error)`.

P1 evidence is documented in
[`benchmarks-obs.md`](../../evidence/performance/benchmarks-obs.md). The frozen contract SHA-256 is
`e97c70dc41b63dfc0c3e238db746fccd15cba43f4ecc45c03f4ac6945116e9cc`; manifest/summary/controls/
observations SHA-256 are respectively
`77A4DBAC0E8883046759D36585B6CA8631E14517CA63D35FC27639D5072CE547`,
`0A5B04FFE639243AC0C5BB8646F1A88EF399339FBC02584E118AC40C7723CB9E`,
`CC92B4DFABB24D6A372E1F7471C67736FD0C385C985B22C517EEC73EEB4D6EAC` and
`017F3D7AAB47C7F63D6A5C2C0405DEE6B5EED243CC7493920BE0478FDFD67650`。deep verifier 返回
`QUALIFIED`、`paths_verified=true`、`provenance=EXACT`、`environment=CURRENT` 和
`current_certification_policy_eligible=true`；A/A、5%/10% known delta、timer floor、placement、
artifact/build、consumer provenance 及 semantic/resource validator 全部 `PASS`。

- ordinary HYLE 每 check 2.6794--2.7045 ns，Abseil 15.6439--16.137 ns，四个 order/placement strata
  ratio 5.8361--5.9789，5% NI 全 `PASS`；这是双方自然公开日志形状的 aggregate，不是原子指令排名；
- 完整 inference replay HYLE 1.3444--1.3741 ns/check，manual consumer 1.3481--1.4542 ns/check，四个
  strata ratio 1.0020--1.0635 且 5% NI 全 `PASS`；最慢完整 replay 约 5.63 us，含 80 条 adapter record，
  低于 100 us 冻结预算；
- limiter-only 0.5034--0.5536 ns/check 和 clock-owning 12.59--12.98 ns/check 都通过绝对预算，但前者
  因更强 exact-concurrency 语义和亚纳秒 placement 翻转预先归为 `NOT_COMPARABLE` diagnostic，不决定
  API、layout 或 performance gate。

Formal 状态为独立 `PENDING`；本轮工程结论不承诺跨平台固定纳秒、异步 logger、所有 sink fan-out 或
全部 Obs 操作全面 SOTA。

## Independent red team

2026-08-29，在冻结 candidate 和 qualification 后按 30 分钟预算复核 capability/contract、复杂度、
correctness/lifecycle、dependency/delivery 和性能有效性；没有新增产品实验或 benchmark 姿势。

| Finding | severity | disposition | evidence |
|---|---|---|---|
| 文档曾把 first-N/interval CAS 写成“有界 CAS/严格 O(1)” | material documentation | `fixed` | 实现保持 peer-standard CAS loop；architecture/guide/task/performance 现明确无竞争 O(1)，竞争重试次数不承诺有界，atomic progress guarantee 属目标标准库 |
| active explicit state 的 move/reset 生命周期容易被误读 | usability | `fixed` | guide/architecture 明确先停止或外部同步 check，再 move/reset/destroy |
| limiter-only 约 0.5 ns 的相对名次随 placement 翻转 | evidence validity | `accepted as diagnostic` | 合同预先标成 `NOT_COMPARABLE`；完整真实 consumer 四 strata 通过，产品未追逐亚纳秒点值 |
| 旧 4/16 independent sink stress 没有普通/llama consumer provenance | evidence validity | `out-of-scope` | consumer manifest 证明正常形状为一个 host adapter；保留为 capacity/contention fact，真实多 sink consumer 才重开 |
| 新 header-only API 可能意外产生 C/FFI 或 binary surface | delivery | `disproved` | source scan、component closure、三 delivery journey、24 decorated-export exact set |
| 第三方或 Time 可能泄漏到基础依赖 | dependency | `disproved` | product closure 仍 `core error`；Tracy optional，peer/consumer 不链接产品 |

红队没有发现未处置 `MUST`、可复现 correctness/security/交付缺陷或 material consumer regression。

## Closure decision

本轮 `CLOSED`。全部准入 `MUST` 已为 `COMPLETE` 或有明确上层 route；scope、architecture、correctness、
delivery、usability/docs、independent red-team 和 P1 engineering performance 七个 gate 都是 `PASS`。
没有 correctness/security、ownership、交付或 material performance blocker；Formal `PENDING` 是独立声明，
不阻塞本轮工程收口。

可接受限制：

- P1 `PASS` 只覆盖 W1-WIN、冻结的普通 one-adapter Abseil aggregate 与 llama.cpp-shaped inference
  progress replay；不签署跨平台固定纳秒、所有 Obs 操作、异步 logger 或全面 SOTA；
- approximate every-N/power-of-two 在并发下允许丢失/重复序号；first-N/interval 的无竞争路径为 O(1)，
  竞争 CAS 的单次重试次数不保证有界，底层 atomic progress guarantee 属目标标准库；
- record/view 和 sink state 仍是 synchronous borrowed ownership；异步 queue/copy、category/config、
  file/network、metrics/OTLP、durability 和模型策略归 host adapter 或未来准入模块；
- 4/16 independent sink 只保留 capacity/contention 诊断，没有 named normal consumer；Native FFI、稳定
  C++ ABI 和 C API 均不承诺；
- retained performance bundle 是本地可深重放工程证据，Formal 签名与长期制品托管仍独立待办。

重开条件：真实 consumer 暴露 correctness/security、borrowed lifetime、并发、API 易用性、交付或本边界
缺失能力；公开 API/layout、registry/control 算法、直接依赖、Tracy seam 或 source/static/shared 形状发生
material 变化；成熟同合同 peer 证明更优复杂度档或正常 consumer 上超过冻结 5% margin 的可复现回退；
真实多独立 sink、异步 ownership、telemetry/Native FFI 等新旅程被正式准入；性能 contract、peer、consumer
provenance、machine/instrument 或 retained bundle 变化而仍需续签声明。

终态 end/registry anchor 是 VERIFIED commit
`f44dc174b3a1d69e0b1e1ea29f2fc8b95ae5d9b0`；产品 candidate 仍是
`21c07d15c4eac477c7b9a768b6475850af87f463` / tree
`3db40042b12c25a5f9cf562d92de42caa077c057`，终态文档提交不冒充产品实现。`VERIFIED -> CLOSED` 的
pre-transition registry snapshot 是
`C:\hyle-scratch\obs-v3\transitions\module-status-before-closed.json`，SHA-256
`67629F2D98C97CEDFDE6639F1D66335C74F2345FD7FDBF857389F6E6B92A4E1E`。

四项 LOC 从 immutable candidate 执行
`python scripts/loc-report.py --module obs --commit 21c07d15c4eac477c7b9a768b6475850af87f463 --self-check --json`，
self-check `PASS`：

1. 全库代码：311,423；
2. 自实现包含测试：124,788；
3. 自实现不包含测试：83,467；
4. Obs 实现：1,235（`include/hyle/obs` + `src/obs`，不含 tests/bench/docs/CMake）。
