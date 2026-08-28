# time-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> **当前阶段：`AUDITED`。** 本页是 `time-v1` 的唯一活动施工记录；既有 time task 与旧性能表只作
> 历史输入。权威状态见 [`module-status.json`](../../module-status.json)。本轮不继承旧的 C ABI、
> `SIGNED` 或 SOTA 结论；capability-family map 与有限 `FIX` 集已经冻结，尚未形成 product candidate。

## Closure card（`SCOPED`）

- component / owner / round ID：`time` / time owner / `time-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`cb46d1183b7ed11757ab0661db893cdfd98b9f65` /
  `19f1c58c3d897240818642b2ca65467e3ad999d8`
- 进入原因与 demand/defect ID：`TIME-CAP-01`（重新核对基础 time 能力及成熟实现的正常形状）、
  `TIME-CONSUMER-01`（以 sched/io/platform 的真实语法与语义消费确定边界）、`TIME-CORRECT-01`
  （时钟、转换、饱和算术和侵入 timer 的边界/生命周期复检）、`TIME-PERF-01`（旧点值与旧 C ABI
  workload 不具当前资格）、`TIME-API-01`（C++23 易用性及无伪 FFI 表面）、`TIME-DOC-01`
  （architecture/user/performance 仍带 provisional、旧 C ABI 或历史结论）、`TIME-DELIVERY-01`
  （build/package v3 后重新验证 source/static/shared 组件闭包）、`TIME-ADOPT-01`（核对底层直接消费者
  是否重复或误用 time 语义）。
- 真实 consumer task / normal workload：每 worker 的 scheduler 在一个 event-loop tick 中批量 arm、
  cancel、advance 和查询下一 deadline；sched timed wait 与 io timer-vs-completion 仲裁；platform/io
  文件 metadata 的 signed Unix timestamp；裸线程控制面的 monotonic deadline 与 blocking sleep。
- 范围：`clock.h`、`unix_timestamp.h`、`TimerEntry/TimerList`、`TimerQueue`、
  `CommonTimeoutQueue`、`clock.cc`；公开 ownership/failure/concurrency/overflow 语义；直接依赖与代表性
  直接消费者；time-only C++ source/static/shared delivery；当前 architecture、user、performance 文档；
  现有 tests/bench 与统一仪器的窄 time adapter/contract。
- 非目标：日历、时区、locale、日期格式化/解析；scheduler/fiber/executor、callback 执行上下文、周期
  timer 的 skip/catch-up 策略、跨线程 timer service；下游模块自身收口；Native FFI/C API；稳定 C++ ABI；
  Formal release certification；为了参考库 API 数量或 benchmark 点值增加常态 metadata。
- 直接依赖 / 直接消费者：`PUBLIC core container_intrusive`；直接消费者为 `platform`、`sched`、`io`。
  本轮验证其 time seam，但不替尚未收口的消费者签署完整结论。
- tier 与理由：`P1`；默认钟、deadline 算术和 `TimerQueue` 位于 scheduler/event-loop 高频路径。
  Unix/OS 转换、sleep、Common queue 及 opt-in profiling clock 没有相同消费频率，分别按适用的结构、
  正确性或 diagnostic 门审查，不能拿来凑一个“time 总分”。
- material margin / resource budget：W1-WIN 同合同 timer consumer replay 使用 5% material 与
  non-inferiority margin；`TimerQueue`/`CommonTimeoutQueue` 正常 arm/cancel 零分配，前者 O(1) arm/
  cancel、后者在单一 duration 合同下 O(1) arm/cancel；不增加 benchmark-only node metadata；默认
  monotonic/realtime typed 与 raw 入口不得比对应 `std::chrono` 多一层动态调用。
- 总调查预算 / 性能预算：功能、API、依赖和文档调查最多 120 分钟；性能只允许一次 probe、一次
  fresh n=15 `qualify-platform`，仅在统一仪器停止规则允许时进行一轮全新 n=31；远程正确性与交付
  只执行冻结矩阵，不扩展研究姿势。
- 最多复现、机制诊断、产品修改次数：一次正常运行；合同内双 order 与一种适用 placement；一次
  codegen/profile 机制诊断；最多一个同时取得机制钥匙和消费钥匙的性能产品假设及重跑；全部产品修复
  合并为一个冻结 batch。
- machine profile：`W1-WIN` 是 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 只承担适用 correctness、
  sanitizer、ISA 与 portability 复核，结果不跨平台池化。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用
  `closure-correctness-v1` 风险矩阵和现有 component/package 合同。
- strong peers（按能力最多两个）：默认 typed clock/elapsed 参考 C++23 `std::chrono` 与 oneTBB
  `e349c15263bbb1d6b48f2b514c06201e1b90eea3`；profiling counter 参考 Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`；任意 deadline queue 参考 Seastar
  `fd71a5b4c71b91abd4e93c9d20f88e1fd7465976` 的 `timer_set`；单一 duration queue 参考 libevent
  `e1f0335d2f3af1235421c939c62943bb99b3d652` 的 `common_timeout`。本轮只从
  `C:/public/project-hyle` 的已钉死本地源码取证。
- 无同合同 peer 时的预算及禁止声明：Unix/FILETIME/POSIX 转换按 checked/exact/round-trip、范围和
  零分配裁决；OS sleep 按不早醒、EINTR/平台降级及 syscall 结构裁决；profiling/coarse clock 若无具名
  产品消费者只保留绝对成本与 codegen diagnostic。不得声明跨平台“所有 time 操作 SOTA”，也不得把
  Common queue 的 peer-derived 一般能力伪装成已接入 sched/io。
- correctness gates：normal；zero/negative/past；全 `uint64_t`、signed epoch、2^53、ratio、NaN/Inf、
  overflow/underflow；EINTR/OS capability/fallback；timer reschedule/cancel/cancel_all/advance/FIFO 与
  wrong-owner；output-list/queue/entry lifetime；随机 oracle、million-churn、public-header self-containment；
  Debug misuse；ASan/UBSan/TSan（按 single-owner 合同适用）；MSVC/GCC/AppleClang 与 x86-64/ARM64 真 ISA。
- delivery gates：中央 component graph 与真实 target edge；time-only source build；relocated static/shared
  C++23 SDK consumer；代表性 platform/sched/io 编译/测试；产物与安装树没有 C schema、bridge、Native
  FFI symbol 或只为 FFI 存在的 option。
- performance gates：冻结 source-primary scheduler timer replay 的自然 tick 计费边界、node ownership、
  cancel/expiry 分布与同合同 faithful Seastar peer；W1-WIN 的 A/A、5%/10% delta、batch-matched floor、
  双 order 与适用 placement 全部通过后，required product strata 须通过 5% non-inferiority。clock/
  common/sleep 的 mechanism 行不进入该产品聚合。
- 预期输出：有限 capability/gap ledger 与唯一 `FIX` 集；immutable candidate commit/tree；correctness、
  delivery、usability/docs、red-team 与 P1 evidence；四项 LOC、known limitations 与 reopen triggers。
- 停止条件 / 可接受限制：全部适用工程门通过则进入 `CLOSED`；若真实产品缺口、同合同 P1 门或
  source/static/shared 交付在冻结预算内不能通过，则点名 blocker 和恢复条件进入 `BLOCKED`。无具名
  consumer 的 calendar/timezone、跨线程 service、periodic policy、Native FFI 和额外 timer family 是
  可接受 negative space，不因成熟库体积而自动扩面。

## 从 `SCOPED` 带入的审计输入

- 当前实现能力已反向枚举为 typed/raw/default/opt-in clocks、signed Unix interchange、饱和 deadline、
  blocking sleep、`Instant`、arbitrary radix timers 与 one-duration timers；每个公开入口仍须落入 ledger。
- 当前真实消费确认：sched 使用 `TimerQueue`、deadline/remaining 与 callback/data；io 使用
  `TimerEntry` 和 `UnixTimestamp`；platform 公开 `UnixTimestamp`。`CommonTimeoutQueue` 暂无产品消费者。
- 已发现但尚未冻结的候选缺口包括：当前性能页仍含已删除 C ABI 的 workload；公开 header 还留有
  “C adapter/boundary” 私有 seam 叙事；Windows timestamp 转换只有读向而没有确认是否应提供 checked
  反向转换。这些只是 scope 时的调查假设；下方 `AUDITED` map/ledger 才是裁决结果。

## Capability-family coverage map（`AUDITED`）

本表先按用户任务和成熟问题域列能力族，再反向核对当前全部 public surface。`COMPLETE` 不是“已有
函数”，而是该族已准入的正常取得、使用、互操作、边界和生命周期都有自然路线。

| Family | admitted user journeys / evidence | required coverage | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| precise clock domains | elapsed/deadline 与 wall timestamp；std::chrono、oneTBB、sched/io | typed + compact monotonic/realtime；两种 clock domain 不可混用；signed wall interchange 另走 timestamp | `monotonic_now/ns`、`realtime_now/ns/timestamp` | `COMPLETE` | `realtime_ns` 有意只表示非负 epoch；pre-epoch 走 `UnixTimestamp` |
| opt-in profiling counter | 高频局部 span；Folly hardware timestamp | capability query、unordered sample、ordered start/stop、语义不冒充 deadline、正确 fallback | x86 invariant TSC、ARM64 architected counter、其他 monotonic fallback | `COMPLETE` | 不提供 cycle-accurate、跨机同步或进程 suspend 后绝对连续性承诺 |
| coarse clocks | 高频 log stamp、rate-limit/event-loop cheap time；Folly coarse steady/system、libevent coarse monotonic | coarse realtime 与 monotonic 两个 domain、typed/compact route、native capability query、fallback 不改语义 | 只有 compact coarse realtime + availability | `PARTIAL` | 缺 coarse monotonic；已有 coarse realtime 也缺自然 typed/timestamp facade |
| CPU accounting clocks | stall/worker 利用率与 profile 归因；POSIX CPU clocks、Folly、Seastar | current-thread 与 process CPU elapsed；chrono-friendly + compact；绝不 fallback 成 wall time | 无公开路线 | `MISSING` | 不做任意外部线程 handle、per-core 或 child-process 聚合 |
| duration/deadline/instant | sched timed wait、io timeout、普通 elapsed；std::chrono/oneTBB | exact upward duration conversion、饱和 add、absolute deadline、remaining、signed/nonnegative elapsed | chrono/raw helpers + `Instant` | `COMPLETE` | calendar duration 与 realtime deadline 有意不接入 timeout 路径 |
| blocking sleep | 裸线程控制面 relative/absolute wait | zero/past、EINTR、overflow-safe absolute monotonic、never-early | raw + chrono `sleep_for/until` | `COMPLETE` | fiber/coroutine sleep 完整路由到 sched，不在此处执行 callback |
| signed Unix interchange | file metadata、OS boundary、pre-epoch；POSIX/Windows/chrono | canonical value、validation/comparison、chrono/time_t/timespec/FILETIME 双向 checked conversion | 除 timestamp→FILETIME 外均有；公开值仍带伪 ABI `reserved` | `PARTIAL` | 缺 checked reverse FILETIME、显式 100ns 精度策略与自然 value comparison/defaults |
| arbitrary timer container | sched worker 与 timed completion；Seastar timer_set | caller-owned stable node、arm/reschedule/cancel/advance/next/drain、full range、normal shutdown | radix `TimerQueue` + detached `TimerList` | `PARTIAL` | 正常 queue-first shutdown 仍 abort；跨线程 routing/回调执行不属于容器 |
| one-duration timer container | serving 中大量相同 idle timeout；libevent common timeout | fixed duration、monotonic append、transactional rejection、cancel/advance/next/drain、normal shutdown | `CommonTimeoutQueue` | `PARTIAL` | 同样缺 queue-first shutdown；当前没有产品 consumer，不进入 P1 聚合 |
| virtual/async/periodic policy | deterministic runtime、executor/coroutine timer、missed-tick policy | clock injection、execution context、cross-thread routing、periodic policy 由 runtime owner 决定 | queue 显式接受 `now`；其余路由 sched/io | `ROUTED` | time 不造第二 event loop，也不把 virtual dispatch 塞进默认 clock 热路 |
| civil/protocol time | RFC3339/RFC1123、timezone、locale | 标准/协议数据库、parse/format policy 由 std/专用库及 io/obs owner 承载 | 标准库/具名上层路线 | `ROUTED` | 不复制 timezone DB、locale DSL 或业务协议 |

现有公开入口反向追踪覆盖 `clock.h` 的所有 alias/read/query/sleep/arithmetic/`Instant`，
`unix_timestamp.h` 的 value/factory/全部 OS 与 chrono 转换，`TimerEntry/TimerList` 的 payload/callback/state，
以及两类 queue 的构造、arm/try/reschedule/cancel/advance/next/cancel-all/query。没有漏出本表之外的
benchmark-only public family；`CommonTimeoutQueueAccess` 是无人消费的旧 adapter seam，不是能力。

## Capability / gap ledger（`AUDITED`）

| ID | class / demand kind | exact source | normal shape / semantic-artifact-delivery owner | ownership / failure / concurrency | complexity / resource | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|
| `TIME-CLOCK-01` | MUST / current consumer + standard | sched deadline/io timeout；C++23 chrono；oneTBB pinned source | typed monotonic/realtime + compact ns；time-time-time | value types；no-fail；clock-domain separation | header-inline O(1)、零分配 | `COMPLETE` | `KEEP` / N/A | typed/raw epoch、header/codegen；pre-epoch wall value 走 timestamp，不自造 Duration |
| `TIME-PROFILE-01` | MUST / repeated foundation | Folly pinned hardware timestamp；profiling/trace journey | raw counter sample、ordered start/stop、availability；time-time-time | process-global calibration cache；no-fail fallback；近距离/固定线程测量 | 热读 O(1)，无 global clamp；首次 x86 校准不 busy-spin | `COMPLETE` | `KEEP` / N/A | x86 CPUID/fallback、ARM64 真 counter、ordered boundary；无 consumer 时只作 diagnostic |
| `TIME-COARSE-01` | MUST / repeated foundation | Folly coarse steady/system + libevent coarse monotonic pinned sources | 保留 coarse realtime；补同 epoch coarse monotonic、typed facade 与 availability；time-time-time | value read；native 缺失时语义等价 precise fallback；thread-safe | native O(1)、零分配；默认钟无额外分支/metadata | `PARTIAL` | `FIX` / BLOCKER | Linux native + Windows/macOS fallback；两 domain 不混用；cached-event-loop policy 留 sched/io |
| `TIME-CPU-01` | MUST / standard + repeated foundation | POSIX CPU clocks；Folly portability + Seastar stall detector pinned sources | chrono-friendly current-thread/process CPU clocks + compact ns；time-time-time | process/current-thread implicit owner；OS invariant failure fail-fast；thread-safe | O(1)、零分配、一次 native read；绝不 fallback 到 wall time | `MISSING` | `FIX` / BLOCKER | Windows/Linux/macOS busy-vs-sleep differential、单调性；不做任意 thread handle/per-core/child 聚合 |
| `TIME-DEADLINE-01` | MUST / current consumer + standard | sched sync/sleep、io timeout；C++23 chrono | exact/ceil/saturating duration、deadline/remaining、Instant、blocking sleep；time-time-time | value types；invalid numeric input clamps by contract；thread-safe | 常用 integral unit 常数乘除；generic ratio O(64) cold；sleep 一次或受 EINTR 重试 | `COMPLETE` | `KEEP` / N/A | 2^53/全范围/NaN/Inf、past/zero/EINTR；fiber sleep 完整路由 sched |
| `TIME-UNIX-01` | MUST / current consumer + platform standard | platform/io FileInfo；POSIX timespec/time_t、Windows FILETIME、chrono | canonical signed value + 双向 checked OS conversion；time-time-time | value-owned；invalid/range 失败为 optional；thread-safe | O(1)、零分配；窄精度策略必须显式 | `PARTIAL` | `FIX` / BLOCKER | 删除伪 ABI `reserved`，补 defaults/comparison；补 checked timestamp→FILETIME（100ns floor 明示、aligned round-trip、边界） |
| `TIME-TIMER-01` | MUST / current consumer + mature algorithm | sched worker/timed wait、io timed slot；Seastar timer_set pinned source | stable `TimerEntry` + 65-bucket radix + detached delivery list；time-time-time | caller owns node；single-owner thread；wrong owner/lifetime fail-fast；不执行 callback | O(1) arm/cancel/next；advance O(touched buckets + fired/rebucketed)；零分配 | `PARTIAL` | `FIX` / BLOCKER | 算法与 arm/cancel/delivery 已完整；`TIME-QUEUE-LIFE-01` 补正常 teardown；callback/data/owner 都有消费者 |
| `TIME-QUEUE-LIFE-01` | MUST / reproducible usability defect + peer | 两类 queue 的非空析构；Seastar timer_set destructor | owner destruction detach/cancel remaining nodes，不执行 callback；time-time-time | queue-first teardown 正常；entry-first 与 linked delivery misuse fail-fast；single-owner | cold O(n)、零分配、noexcept；entry 随后可复用/销毁 | `PARTIAL` | `FIX` / BLOCKER | 替换非空 queue 析构 abort；两类 queue、reuse 和 death tests |
| `TIME-COMMON-01` | GENERAL / mature peer + serving journey | libevent common_timeout pinned source；同-duration idle timeout | fixed duration、monotonic append、transactional reschedule、normal teardown；time-time-time | caller owns node；single-owner；ordering violation recoverable via try；不执行 callback | O(1) arm/cancel/next，advance O(fired)，零分配 | `PARTIAL` | `FIX` / MATERIAL | 算法/order/reject 已完整；补 queue-first teardown；无当前 product consumer，不进 P1 aggregate |
| `TIME-LEGACY-01` | MUST / build-package contract | accepted Native FFI=N/A 决策；public-surface trace | 只有 C++ source/static/shared；time-time-build/package | N/A | 零额外 surface/metadata | `PARTIAL` | `FIX` / CLEANUP | 删除无人引用 `CommonTimeoutQueueAccess` 与 C-adapter 文案；全树 stale scan |
| `TIME-ADOPT-01` | MUST / direct-consumer audit | sched/io/platform syntax + semantic scan at baseline | time 统一 deadline/queue/timestamp；OS wait/resource op 留 owner；time + consuming module + build/package | 各 owner 保持原 concurrency/failure contract | 不增加反向依赖，不给热 completion 强塞 Status | `COMPLETE` | `KEEP` / N/A | blocking CV/futex/touch-now 是正确 owner；候选后重跑受影响消费者 |
| `TIME-IO-ROUND-01` | MUST / reproducible direct-edge defect | `src/io/proactor.cc` Windows/epoll finite timeout | ceil ns→OS ms，infinite sentinel 保留；time semantic、io artifact/delivery | value conversion；no-fail；backend worker-owned | O(1)、无 overflow；最大有限值只能 clamp，不能 wrap 成 poll | `PARTIAL` | `FIX` / BLOCKER | 两处 add-before-divide 改 quotient/remainder helper + constexpr boundaries |
| `TIME-PERF-01` | MUST / current-consumer evidence gap | `src/sched/scheduler.cc` pinned source + Seastar peer | 64-step scattered nodes、约 70% cancel、逐 tick advance、最终 drain replay；time bench artifact、perf delivery | stable caller-owned nodes；single worker；同 callback/data/owner metadata | natural batch 64；W1-WIN 5% NI；双 execution/allocation order | `MISSING` | `FIX` / BLOCKER | 新 source-consumer contract/telemetry；A/A、5/10%、floor；旧 N sweep/clock/sleep 仅 diagnostic |
| `TIME-DELIVERY-01` | MUST / build-package contract | central component graph + build/package v3 | `time PUBLIC core container_intrusive`；time artifact、build/package delivery | C++ source/static/shared；无 stable ABI/FFI | exact selected closure；无伪 option/symbol | `COMPLETE` | `KEEP` / BLOCKER-at-verification | time-only header/build、relocated SDK、代表消费者；candidate 上三平台复核 |
| `TIME-DOC-01` | MUST / docs contract | current architecture/user/performance pages | architecture=决策，user=选型/pitfall，performance=当前证据，task=事实；time/doc owners | N/A | 不复制易腐点值或历史 current-state | `PARTIAL` | `FIX` / BLOCKER | 加 coarse/CPU/interop/teardown 选型，清 current `NOT_REVIEWED`，重写 stale performance；历史 task 不改写 |
| `TIME-SCHED-CLOCK-01` | GENERAL / peer-derived higher-layer | Abseil simulated clock、Seastar manual clock | queue 已接受 explicit now；完整 runtime clock injection 归 sched-sched-sched | runtime owns clock lifetime/concurrency；default time clock 无 virtual dispatch | 默认 clock 热路零成本 | `ROUTED` | `DEFER` / N/A | sched deterministic-runtime journey 触发；不是 time container 缺口 |
| `TIME-UTC-01` | GENERAL / peer + future protocol | Abseil civil/timezone；未来 io/obs protocol | RFC3339/RFC1123/timezone 归 std/专用库与 io/obs owner | database/locale ownership 不进入 time | 不复制 timezone DB/locale DSL | `ROUTED` | `DEFER` / N/A | 具名协议 consumer 触发独立提案；user guide 说明 route |
| `TIME-EXEC-01` | GENERAL / peer-derived higher-layer | Asio/Tokio/Seastar timer service | owning callback、async/periodic/cross-thread policy 归 sched/io | executor owns callback lifetime、cancel 与 routing concurrency | 由 runtime workload 决定 | `OUT_OF_SCOPE` | `REJECT` / N/A | time 只交付 data structure；不造第二 event loop 或含糊 `repeat=true` |

审计补充：

- 已有能力完整性审计明确发现三类“不是全新模块、而是既有能力族未做完整”的问题：coarse clock
  只有 realtime、Unix interchange 只有 FILETIME 读向、两类 timer 只有显式 `cancel_all` 而正常
  queue-first teardown 会 abort。CPU accounting 是成熟 runtime 重复出现但当前完全缺失的独立 time
  domain；它不会改变默认 clock 或 timer 热路径，因此按通用基础准入提升为 `MUST`。
- peer 源码身份采用 closure card 的四个 pinned commit。Seastar 的 queue 析构逐 timer cancel，直接否定了
  当前“非空 owner 析构必崩”是必需安全合同的假设；libevent common queue 仍支持特化能力，但没有
  HYLE product consumer，故保持 GENERAL 而不冒充产品收益。
- direct-consumer scan 证明 sched 的 queue/deadline、io 的 callback/Unix timestamp、platform 的
  FileInfo timestamp 边正确；发现的唯一 material adoption defect 是 io 两个有限大 timeout 的 ceil-add
  wrap。platform 的 wait backend 已使用安全 quotient/remainder，不能为了表面统一反向依赖 time。
- public/internal/higher-layer 边界：time 拥有 clock、timestamp、deadline math 和 timer containers；
  platform 拥有 OS blocking/resource mechanism；sched/io 拥有执行、仲裁、virtual-time/periodic policy；
  obs/协议 owner 拥有格式化、时区和 parse。
- 冻结性能合同：唯一 product workload 为 `scheduler-timer-lifecycle-64`；provenance 是
  `src/sched/scheduler.cc`，candidate 与 faithful Seastar port 均使用相同 stable scattered node、
  callback/data/owner metadata、同一 deadline/cancel 序列、同一 ready-list drain。`monotonic_ns`、
  profiling/coarse clock、Common queue、sleep overshoot、N=1 和大 N 算法 sweep 均只作 mechanism/
  diagnostic，不进入 P1 聚合。
- **本轮唯一冻结的十项 ledger `FIX`：**`TIME-COARSE-01`、`TIME-CPU-01`、`TIME-UNIX-01`、
  `TIME-TIMER-01`、`TIME-QUEUE-LIFE-01`、`TIME-COMMON-01`、`TIME-LEGACY-01`、
  `TIME-IO-ROUND-01`、`TIME-PERF-01`、`TIME-DOC-01`。其中 timer/common 两个 family 由同一个
  queue-lifecycle 实现批次补齐，因此实际是八个施工批次。实现只允许这一批；新发现项必须先回到
  ledger 重裁决，不能边施工扩面。
