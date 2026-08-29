# obs v3 consumer-driven closure

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> This is the active `RESUME` record for `obs-v3`. The current product state is
> [`module-status.json`](../../module-status.json). The terminal
> [`obs-v2`](obs-v2-resume.md) remains historical evidence and is not edited by this round.

## Closure card (`SCOPED`)

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
  - every limiter operation is O(1), allocation-free, mutex-free and syscall-free when caller time is
    supplied; count suppression performs no stronger synchronization than its documented semantics;
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
| `OBS-V3-CALLSITE-01` | `MUST` peer consensus | Abseil `LOG_IF`, every/first/pow2/time family; Folly approximate and exact count variants; ordinary repeated call site | Obs public header; one static state per expansion; O(1), no heap/lock/syscall except the selected clock read | `COMPLETE` | `DONE`: all-level conditional/stateful macros preserve source location and short-circuit ordering |
| `OBS-V3-COUNT-SEM-01` | `MUST` semantic split | logging frequency normally tolerates lossy counts, while cardinality-sensitive users need exact admission | `LogEveryNState` and `LogEveryPow2State` use relaxed load/store and are explicitly approximate under contention; `LogEveryNExactState` uses relaxed RMW; `LogFirstNState` uses bounded CAS and never admits more than N | `COMPLETE` | `DONE`: zero/negative N, reset/move, sequence and exact eight-thread cardinality tests pass |
| `OBS-V3-INTERVAL-01` | `MUST` llama consumer | each server slot owns last progress state and checks a 3-second steady interval | `LogIntervalLimiter`; exact relaxed-CAS admission, first check passes, duration <=0 passes, overflow-safe difference, O(1); caller-supplied steady time avoids duplicate clock reads | `COMPLETE` | `DONE`: raw/chrono boundary, reset, rollback, unsigned maximum and exact concurrency tests pass |
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

Candidate answers; final verdicts still require verification:

1. optional Tracy v0.14.1 is the only product-path third party and only when trace is enabled; Abseil,
   Folly and llama.cpp are peer/consumer evidence, not product dependencies;
2. admitted ordinary/inference log and trace journeys are implemented; metrics, async logging and model
   policy remain explicitly routed above Obs;
3. P1 must be decided by the newly frozen one-adapter/count/inference contract, not the historical
   multi-sink stress result;
4. Native FFI is `N/A` and the product tree must remain free of public C artifacts;
5. new public headers require source, installed-static and installed-shared selected-Obs revalidation;
6. the C++ surface now exposes both hidden call-site and explicit object-owned state, and the guide documents
   the default-sink takeover and traps.

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

Not yet run. Unrun applicable gates are not `PASS`.

## Independent red team

Not yet run; budget 30 minutes after the candidate and evidence are frozen.

## Closure decision

Pending. This active record contains no terminal receipt.
