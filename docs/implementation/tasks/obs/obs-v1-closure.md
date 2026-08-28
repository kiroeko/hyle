# obs-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`AUDITED`。本文是 obs 当前轮唯一 closure card 与施工记录；既有 obs task、architecture、
> guide、旧 C/FFI 叙事和历史 benchmark 只作审计输入，不继承旧“已收口”或 SOTA 结论。机械状态只看
> [`module-status.json`](../../module-status.json)。在 capability map 与有限 `FIX` 集进入 `AUDITED`
> 前，不修改 obs 产品实现。

## Closure card（`SCOPED`）

- component / owner / round ID：`obs` / observability owner / `obs-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`f305b9d3fed87a065d300145e5d9e8de3f5a77a0` /
  `ee1d875638ec95e2228d0b5bb54e5e3dc1c54369`
- 进入原因与 demand/defect ID：`OBS-CAP-01`（按基础库完整性标准重新枚举 log/trace 能力族）、
  `OBS-NORTHSTAR-01`（本地推理引擎的请求/算子诊断与 allocation/fiber/io/device-poller 时间线）、
  `OBS-API-01`（同步 borrowed record、sink owner、format/trace 便利面的自然 C++ 形状）、
  `OBS-LIFETIME-01`（并发 fan-out、递归、quiescent unregister 与进程 teardown）、
  `OBS-PERF-01`（旧点值未进入当前统一仪器且完整 Abseil/Folly record 不是同合同 peer）、
  `OBS-DOC-01`（architecture/guide 仍标 `NOT_REVIEWED`，旧 task 仍叙述已删除的 C ABI）、
  `OBS-DELIVERY-01`（Tracy 0.14.1 升级后的 OFF/ON 与 source/static/shared 复签）。
- 真实 consumer task / normal workload：
  - 推理引擎把已存在的 request/operator/backend 诊断按同步 borrowed record 交给一个宿主 adapter
    sink；adapter 在 callback 内决定复制/排队，obs 不拥有异步线程、文件或上传策略；
  - 多 worker 同时产出预格式化 record，控制面偶尔注册/注销 adapter；注销成功后 host 可以立即
    卸载 callback state；
  - mem allocation、sched fiber/timer、io completion 与 compute GPU-poller 的当前
    `HYLE_TRACE_*` 埋点在 profiling build 汇入同一个 Tracy client；默认 build 完全消失；
  - 工程计算留出使用相同 log record/sink 与 zone/counter/memory/thread/fiber 机制，不引入模型、
    token、dashboard 或服务协议类型。
- 范围：`include/hyle/obs/**`、`src/obs/**` 的同步 structured-log producer、level/compile gate、
  default sink、sink registry/RAII/flush、native trace macro seam；全部 public surface 反向追踪；
  `obs PUBLIC core error` 与当前 trace consumers；Tracy owner/build/package 投影；obs-only
  source/static/shared C++23 delivery；architecture/user/performance 文档与有限统一 P1 replay。
- 非目标：异步 queue/thread、文件轮转、网络 export、聚合/查询/告警、metrics registry、dashboard、
  crash/signal-safe logging、fatal/check 控制流、完整 category/config tree、OpenTelemetry protocol、
  profiler viewer、GPU timestamp/context API、业务采样/命名策略、Native FFI/C API、稳定 C++ ABI、
  下游模块整体收口和 Formal 发布证书。
- 直接依赖 / 直接消费者：`PUBLIC core error`；log runtime 当前由外部 C++ 用户/sample 主动消费，
  HYLE 底层模块不擅自输出日常日志；trace 的真实源码消费者为 `mem`、`sched`、`io`、`compute`。
  trace 是受检的 header/CMake 横切 seam，不制造这些模块到 obs log runtime 的伪二进制依赖。
- tier 与理由：`P1`。runtime level gate、同步 emit/fan-out 与 trace event 可位于高频路径；sink mutation、
  default stderr、flush、配置和 teardown 是 P0 控制面。关闭态 trace/compile floor 是 codegen/交付门，
  不能制造“0.00 ns”计时排名。
- material margin / resource budget：W1-WIN 同合同 baseline/direct-Tracy replay 使用 5% NI；runtime
  gate 至多一次 relaxed threshold load且参数不求值；exact-byte emit 为 O(S)、S≤64，不做 heap
  allocation、不拿 mutation mutex、不扫描 message；mutation O(S) 且只在控制面；默认 sink 不因安全
  转义引入 heap；OFF trace 零参数求值、零符号、零 backend，ON seam 不比直接 Tracy 增加 material
  event 成本。吞吐不能抵偿 callback tail、unregister hang、stack/record 常态膨胀或多 singleton。
- 总调查预算 / 性能预算：从 `SCOPED` 起总计最多 120 分钟，其中统一性能取证最多 45 分钟、最终
  红队最多 20 分钟；到时只能 `CLOSED`、具名 `BLOCKED`，或发现产品变化后按协议退回阶段，不能继续
  发明实验。
- 最多复现、机制诊断、产品修改次数：每个 material 异常一次反序/复现、一次 codegen/profile/
  contention 机制检查、最多一个同时具备机制与 consumer 钥匙的产品修改及一次重跑；不得修改中央
  统计语义或发明第五种 benchmark 姿势。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用
  correctness/sanitizer/OS/default-sink/trace-on/交付复核，不跨平台池化点值。Apple 大小核只按已登记
  profile/QoS 观察，不用最好 placement 签胜负。
- instrument profile：P1 为 `closure-perf-v1`；correctness/delivery 为
  `closure-correctness-v1` 与当前 component/package contract。
- strong peers（绑定当前本地 clean source）：
  - synchronous log entry/sink/flush/recursion/process lifetime：Abseil
    `81c993af4fb88b3a51829a796df02251fa27102f` 与 Folly
    `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`；它们用于能力、lifetime、复杂度和负空间裁决，
    完整 category/config/LogEntry 的点值不冒充同合同排名；
  - compile-time profiling seam/backend lifecycle：Tracy v0.14.1
    `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`；相同 event 可作 direct same-contract peer。
- 无同合同 peer时的预算及禁止声明：logging 正常 replay 对 frozen baseline
  `f305b9d3` 做 5% regression NI，并守一 sink exact emit ≤100 ns、format emit ≤150 ns、8 producer
  ≤150 ns/emit 的 W1-WIN 预声明上限；同时以 strong-peer source 审计证明复杂度同档。该证据只支持
  “当前轻量同步合同无 material regression/已知复杂度缺口”，不得宣称比完整 Abseil/Folly 快若干倍、
  “obs 全面 SOTA”或跨平台固定纳秒数。
- correctness gates：normal；empty/zero/embedded-NUL/format truncation；invalid level/field/sink、容量
  与 waiter/refcount 边界；duplicate/absent/full/retiring；move/reset/destructor/static teardown；并发
  emit/mutation/flush、remove-waits、自摘/跨摘、nested record；default-sink control-byte/TTY/debugger；
  stress；ASan/UBSan/TSan；trace OFF 参数/codegen 与 ON 真 Tracy backend；public-header self-containment。
- delivery gates：中央 `obs PUBLIC core error` 与 include-owner/target 对账；Tracy pin/license/offline/
  shared-one-client/static-one-final-image；obs-only source embedding、relocated static/shared C++23
  consumer；trace-OFF package 无 Tracy，trace-ON package 与 mem/sched/io/compute seam；无旧 C schema、
  bridge、fixture 或 Native FFI symbol。
- performance gates：冻结四个互不池化的 family：
  1. `compile/runtime-gate`：compile floor/trace OFF 走 codegen O(0)，runtime drop 走一次 threshold load；
  2. `single-adapter-emit`：exact byte + structured 4-field + format macro 向一个无 I/O同步 adapter sink；
  3. `multi-producer-fanout`：8 producers 与 1/4/16 sink，报告 per emit/callback、资源与 callback tail，
     mutation 只作并发 correctness/control replay；
  4. `trace-native-seam`：counter 与 named zone 对直接 Tracy 同 event 的 codegen和5% NI。
- 预期输出：capability-family map、全部 public-surface trace、有限 FIX 集、owner/complexity/resource 表、
  correctness/stress/consumer证据、统一 P1 bundle、三种交付、更新 architecture/user/performance、独立
  红队与四项 LOC。
- 停止条件 / 可接受限制：全部 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，四个 family 的适用工程门
  `PASS`，无 correctness/security/delivery blocker 即停止；完整日志平台、GPU profiler、multi-image
  static singleton、Native FFI 与 Formal 可保持明确负空间。若功能 `MUST`、并发生命周期、统一仪器
  资格或 P1 NI 不能在预算内裁决，进入具名 `BLOCKED`，不无限研究。

## Capability-family coverage map（`AUDITED`）

本表先从真实使用任务和成熟问题域枚举能力族，再反向核对全部 public surface。`COMPLETE` 表示准入的
正常取得、使用、生命周期、失败和交付路线都存在，不表示 obs 要复制一个完整日志/遥测平台。

| Family | admitted user journeys / mature evidence | required coverage | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| failure / diagnostic separation | HYLE 内部 operational failure；Abseil/Folly sink 只消费已决定记录的事件 | `Status/Result` 保留控制流，log 由调用方主动产出，底层库不自动打印 | error + obs 两条独立路线 | `COMPLETE` | fatal invariant 仍归 core；obs 不吞错误或以日志代表成功 |
| level、compile gate 与 caller formatting | 普通应用诊断；Abseil compile/runtime gate；C++23 format/source-location | compile floor、一次 runtime gate、未通过不求值、source、bounded allocation-free format | `HYLE_LOG_*`、`should_log`、`log_fmt[_at]` | `PARTIAL` | 外来 custom formatter 可抛异常并穿透业务；需在最窄 adapter 边界转成静态 fallback |
| borrowed exact structured record | 已有 request/operator/backend 字段和 byte slice；Abseil `LogEntry` borrowed lifetime | exact message/field slices、source/timestamp、同步借用、零复制、无 C layout 假象 | message 已为 `string_view`，fields 仍是 `const char* + pointer/count` | `PARTIAL` | field 仍扫描 NUL、不能表达 slice/embedded NUL，并允许 pointer/count 错配 |
| native sink integration | 应用把 HYLE record 接进既有 logger/队列；Abseil `LogSink`、Folly `LogHandler` | 同步 callback、可选 flush、无分配类型擦除、自然 typed adapter、明确 copy boundary | `LogSink {void*, fn...}` | `PARTIAL` | 底层 seam 正确，但普通 C++ 用户必须手写 cast/static trampoline |
| sink ownership / concurrent fan-out | 多 worker emit、偶尔动态插件注册/卸载；Abseil recursion/process-lifetime registry | RAII、duplicate/full/absent、quiescent remove、递归与 self/cross-remove 规则、process teardown | fixed 64-slot immutable snapshot + active refs + `SinkRegistration` | `PARTIAL` | 算法覆盖完整；live-target move assignment、flush 自摘/并发注销仍缺直接回归证据 |
| safe default output | 无上层 sink 的 CLI/工程诊断 | stderr、TTY color、单物理行、控制字节转义、并发串行、显式开关 | default sink + Windows debugger mirror | `COMPLETE` | I/O、flush-per-record 与 4 KiB debugger 截断是默认冷 sink 的明确策略，不进入 adapter 热路 |
| compile-time trace/backend lifecycle | mem/sched/io/compute 热埋点；Tracy native integration | OFF 零求值/零符号/零依赖；ON 一个 backend client；source/static/shared 规则 | `HYLE_TRACE_*` + `hyle_trace` / Tracy 0.14.1 | `COMPLETE` | static 多 DLL/plugin 无法共享 singleton，明确改走 shared 交付 |
| scoped timeline + current-zone annotation | 推理 request/operator/kernel 动态名称、shape 与 work value；Tracy zones/text/name/value/color | RAII zone + copied dynamic text/name + value/color；OFF 全部消失 | 只有 zone 与 static named zone | `PARTIAL` | 已删除旧 runtime bridge 后没有 current-zone annotation，message 不能替代 zone 关联 |
| numeric/message/memory events | completion/queue counters、动态诊断、allocator attribution；Tracy plot/message/alloc | numeric normalization、exact bounded text view、alloc/free pairing、稳定 identity name | plot/counter/message/memory macros | `PARTIAL` | counter 对 full-width unsigned 静默 wrap；dynamic message 只接 C string 并重复扫描已知 slice |
| execution identity | scheduler fiber migration、worker thread、serving iteration | fiber enter/leave、thread name、frame boundary与稳定名称生命周期 | fiber/thread/frame macros | `COMPLETE` | 配对与 storage lifetime 由消费者维护；不制造 scheduler policy |
| category、sampling、async storage and operations | Abseil VLOG/rate macros、Folly category/config、应用日志平台 | 只有具名 consumer 才决定 category tree、sampling state、queue/drop/block、file/network/export | structured field + adapter/application owner | `ROUTED` | 不为未消费能力给每条 record 增加 category/thread/refcount/owning string 或后台线程 |
| metrics / distributed telemetry / GPU profiler | 未来服务、OpenTelemetry、device backend | registry/aggregation/protocol 与 GPU timestamp/context 由对应 owner 承担 | future metrics/compute/application layer | `OUT_OF_SCOPE` | obs v1 不冒充 metrics SDK、viewer、OTLP exporter 或 device profiler |
| delivery / foreign runtime | 首席消费者直接 C++ 源码；可选 static/shared SDK | `obs PUBLIC core error`、trace owner 对账、C++23 三交付；FFI 仅在具名需求后设计 | current component/package system | `COMPLETE` | Native FFI/C API 为 `N/A`，不保留伪 schema/符号/ABI 承诺 |

公开入口反向追踪覆盖：`LogLevel/LogField/LogRecord/LogSink`，registry/RAII/count/capacity/flush，
level/query，record/message/default sink，format helper 与五级宏；trace 的 zone/plot/counter/fiber/frame/
message/memory/thread-name 以及 build gate。没有入口落在表外，也没有把 benchmark helper 升成产品类型。

## Capability / gap ledger（`AUDITED`）

| ID | class / demand kind | exact source | normal shape / semantic-artifact-delivery owner | ownership / failure / concurrency | complexity / resource | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|
| `OBS-ROUTE-01` | MUST / project contract | lifecycle architecture + error/core consumers | error 传机器语义，调用方决定是否 log；error-error-obs | value error + explicit diagnostic | 不给失败热路增加 I/O/registry | `COMPLETE` | `KEEP` / N/A | 全树语义 scan；HYLE 产品代码不得为了 adoption 自动打印 |
| `OBS-LEVEL-01` | MUST / normal logging | C++23 + Abseil gate | five record levels + Off threshold、TU compile floor、一次 relaxed runtime load；obs-obs-build | process-global eventual config；invalid level fail-fast | dropped O(1)、零参数求值/分配 | `COMPLETE` | `KEEP` / N/A | compile-floor/codegen + gated throwing/side-effect argument |
| `OBS-RECORD-01` | MUST / source C++ consumer + legacy defect | request/operator journey；Abseil borrowed entry；当前 public trace | `LogField` key/value 用 `string_view`，`LogRecord::fields` 用 `span<const LogField>`；obs-obs-obs | 整条同步借用；sink 异步保存必须复制；invalid view 是 C++ 前置条件 | emit O(S)，不扫描 message/fields、不分配；record span 不增大小 | `PARTIAL` | `FIX` / BLOCKER | embedded-NUL/exact field slice、empty span、default sink escape；删除 pointer/count 文档与调用点 |
| `OBS-SINK-API-01` | MUST / usability | adapter sink journey；Abseil/Folly typed sink shape | 保留 ABI-free function-table seam并补 `LogSink::from(backend)`；obs-obs-obs | backend `write(record) noexcept`，可选 `flush() noexcept`；borrowed backend | header-inline static trampoline，零分配、零 virtual/extra metadata | `PARTIAL` | `FIX` / MATERIAL | typed sink with/without flush；manual seam 仍供高级用户 |
| `OBS-LIFETIME-01` | MUST / concurrent unload | Abseil sink registry + current immutable-snapshot algorithm | RAII/create/reset/move、quiescent add/remove、flush、自摘、跨摘拒绝、nested fallback；obs-obs-obs | callback concurrent/noexcept；state 活到 callback 返回；registry process-lifetime | emit O(S)、mutation O(S)、fixed S≤64；emit 无 mutex/heap | `PARTIAL` | `FIX` / BLOCKER | live-target move、create failure、flush self-remove/nested/concurrent-remove、stress；无新常态字段 |
| `OBS-DEFAULT-01` | GENERAL / CLI fallback | current Windows/POSIX sink + terminal safety | safe stderr + optional debugger mirror；obs-obs-obs | serialized, synchronous, controllable | O(bytes+fields)、零 heap；I/O/flush 冷路径 | `COMPLETE` | `KEEP` / N/A | control bytes/one line/concurrent setters；不作 async-signal-safe 承诺 |
| `OBS-FORMAT-01` | MUST / public exception contract | lifecycle architecture §exception boundary；custom `std::formatter` | compile-checked bounded format，外来异常在最窄边界变静态 `"[log formatting failed]"` record；obs-obs-obs | `noexcept`，不捕获 HYLE invariant；同步 borrowed fallback | 1 KiB caller stack、O(formatted bytes)、零 HYLE heap | `PARTIAL` | `FIX` / BLOCKER | throwing custom formatter、truncation/source、gate 不进入 formatter |
| `OBS-TRACE-GATE-01` | MUST / current consumers | mem/sched/io/compute + Tracy 0.14.1 | OFF no-eval/no-symbol；ON one native client；obs-build-build | compile-time policy；shared process singleton / static one final image | OFF O(0)；ON 等同 direct backend event | `COMPLETE` | `KEEP` / BLOCKER-at-verification | OFF object/symbol、ON build/link/viewer handshake、one-client audit |
| `OBS-TRACE-ZONE-01` | MUST / northstar + direct backend | request/operator/kernel dynamic annotation；Tracy `ZoneText/Name/Value/Color` | 补 current-zone copied text/name、uint64 value、color；obs-obs-obs | 仅在同 lexical scoped zone 内；text 同步复制；OFF 不求值 | direct macro seam，无 allocation/search/常态 HYLE metadata | `MISSING` | `FIX` / BLOCKER | string/string_view/long-bound、OFF side effect、direct Tracy codegen/NI；GPU/lock/callstack 仍在负空间 |
| `OBS-TRACE-EVENT-01` | MUST / current consumers | io/compute/sched uint64 counters；Tracy overload set | plot/counter 共享 numeric normalization；dynamic message 接 bounded `string_view`/C string；obs-obs-obs | stable plot name；message copied；alloc/free caller配对 | O(1) event + bounded text bytes；full uint64 走 backend double transport | `PARTIAL` | `FIX` / MATERIAL | int/float/uint64 compile + >INT64_MAX 不 wrap；exact slice/long text + OFF side effect |
| `OBS-TRACE-CONTEXT-01` | MUST / current consumers | scheduler fiber、worker、frame、mem allocation | fiber/frame/thread/memory native seam；obs-obs-consumer | stable identity pointer、explicit pairing | direct Tracy cost；OFF O(0) | `COMPLETE` | `KEEP` / N/A | current consumer syntax/semantic scan；命名/采样政策归 consumer |
| `OBS-ADOPT-01` | MUST / dependency audit | complete source scan | runtime obs 不被 HYLE 底层滥用；trace 消费保持 native seam；obs + consumers + build | error 不自动 log；fatal/bench output 保留原 owner | 无反向 runtime edge或热路 Status/log metadata | `COMPLETE` | `KEEP` / N/A | core/error/platform/mem/time/util/sched/io/compute syntax + semantic scan |
| `OBS-DELIVERY-01` | MUST / build-package contract | central graph + HyleTrace/HylePackage | `obs PUBLIC core error`；trace horizontal owner；source/static/shared C++23；build-build-build | exact Tracy pin/license/offline；one-client deployment contract | exact selected closure，无旧 FFI artifact | `COMPLETE` | `KEEP` / BLOCKER-at-verification | obs-only source/relocated SDK，trace OFF/ON，shared/static symbols与 representative consumers |
| `OBS-PERF-01` | MUST / evidence gap | frozen f305 baseline + direct Tracy | four frozen workload families，source-primary exact contract，W1-WIN 5% NI/ceilings；obs-perf-perf | fixed sink state/work，报告 per emit/callback与 tail；不池化完整 Abseil record | gate O(1)、emit O(S)、format O(bytes)、trace direct seam | `MISSING` | `FIX` / BLOCKER | A/A、reverse order、known delta/floor/identity；旧跨平台点值只作 historical diagnostic |
| `OBS-DOC-01` | MUST / current stale docs | architecture/user/performance/old task scan | architecture=长期决策，guide=选型/坑，performance=当前证据，task=事实；docs owners | N/A | 不复制易腐点值或伪 current state | `PARTIAL` | `FIX` / BLOCKER | 删除 NOT_REVIEWED、旧 C/runtime bridge、错误路径和 stale links；旧 task 显式 HISTORICAL |
| `OBS-CATEGORY-01` | GENERAL / two mature sources, no consumer | Abseil VLOG/rate；Folly category/config | category/filter/sample/route 归已有 application logger 或新具名需求 | policy/state owner 必须先确定；不能偷偷阻塞/drop | 避免每 record category lookup/metadata和 callsite state | `ROUTED` | `DEFER` / N/A | adapter 可用 structured field；真实 producer-side preformat filter 需求触发独立 scope |
| `OBS-SERVICE-01` | GENERAL / higher-layer systems | Folly/spdlog/OTel/operations | async queue、file/network、rotation、metrics/export/alert 归 application/专用 owner | 必须定义 capacity/backpressure/drop/shutdown | 由真实 service workload 裁决 | `OUT_OF_SCOPE` | `REJECT` / N/A | obs 不造第二日志服务或 telemetry protocol stack |
| `OBS-FFI-01` | GENERAL / accepted project policy | Native FFI policy + source-primary consumer | 当前仅 C++ source/static/shared；未来具名 foreign-runtime 走窄 runtime type erasure | 不承诺 C++ layout/callback ABI | 零伪 surface/symbol | `OUT_OF_SCOPE` | `REJECT` / N/A | 全树旧 C schema/fixture/symbol scan；不能把 C 头当 Java/C# 交付 |

审计结论：

- registry 的 fixed-capacity immutable snapshot + per-sink active reference 直接服务 quiescent unload；其
  O(S) fan-out 达到必须逐 sink callback 的渐进下界。Abseil/Folly 同样采用同步 sink/handler，未发现
  更低复杂度的同合同方案。不能为 benchmark 去掉 quiescence、递归保护或 process-lifetime 状态。
- `LogField const char*` 与 pointer/count 是旧 C ABI 留下的产品形状，不是性能优化。换成 view/span 同时
  改善正常 C++ 易用性、exact-byte 表达和字段热路复杂度；它不增加 `LogRecord` 大小，也不新增常态分支。
- Tracy 已直接提供 current-zone text/name/value/color；旧 runtime bridge 删除后 HYLE 的注释称其存在，
  真实 inference journey 却无自然路线。补齐的是零抽象税 native seam，不是为了制造 benchmark 项。
- Abseil/Folly 的 category、VLOG/rate、async/file/config 能力证明问题域存在，但当前 HYLE 既无 runtime
  日志消费者，也不拥有应用日志政策；本轮明确路由而不把完整 logger 的状态和成本强塞给基础 record。
- **本轮唯一冻结的八项 ledger `FIX`：**`OBS-RECORD-01`、`OBS-SINK-API-01`、
  `OBS-LIFETIME-01`、`OBS-FORMAT-01`、`OBS-TRACE-ZONE-01`、`OBS-TRACE-EVENT-01`、
  `OBS-PERF-01`、`OBS-DOC-01`（后两项为证据/文档批次；产品施工合并为 log API/lifetime 与 trace
  两批）。新发现项必须先退回 ledger 重裁决，不能边实现边扩展成日志平台。
- `SCOPED -> AUDITED` 前态 registry 保存于外部 scratch
  `C:\hyle-scratch\obs-v1\transitions\module-status-before-audited.json`，SHA-256 为
  `3D0D7D124249B0A11E8365E8EA04A311D3FD713412F52A1053DBA537943A375E`。

## Candidate

尚未冻结。

## Verification manifest

尚未执行。

## 独立红队

尚未执行。

## Closure decision

尚未裁决。
