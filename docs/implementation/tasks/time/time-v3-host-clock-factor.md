# Time v3：host-clock 质因数与可注入时间

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`time` / Time / `time-v3-host-clock-factor` / `REOPEN`；
- baseline：`25b13f5a18a0164a5fa81d01c4ef5d1a32f85cf4`；前轮：`time-v2`；
- trigger：`TIME-FACTORIZATION-2026-09-02`；
- `TIME3-R1`：已接受的 Stage C 证明 host clock/value 与 caller-owned timer node 是两个独立消费者、依赖
  和成本闭环。现有 `time -> container_intrusive` 迫使只需要 `UnixTimestamp` 的 Platform 取得 timer
  storage 依赖，满足“component dependency/delivery shape changed”的重开条件；
- `TIME3-R2`：能力账本 `TIME-A-17` 已由 Seastar manual clock、Abseil simulated clock 与确定性
  scheduler/timeout 测试共同准入，但当前 HYLE deadline helpers 固定读取全局 clock，只能给 queue 显式
  传 `now`，属于已确认 `MISSING`；
- `TIME3-R3`：time-v2 candidate 后，Time 实现已迁移到新的 Core environment contract，当前源码 identity
  不再等于旧 candidate；本轮以当前 baseline 重新绑定受影响的依赖、交付和 codegen，不伪装旧证据覆盖；
- 最新参考身份（2026-09-02 clean fast-forward）：llama.cpp
  `b96806d96061049a5b574269b049bf6241d63d46`（随仓 ggml tree
  `bc886da1a6c9d69660b0043d407200015667d609`）、standalone ggml
  `d4716378882593333721eb33f153144b6885caf2`、Seastar
  `991cf9a908482a3f11ca2fc34a1b3970a06d8622`、Abseil
  `cc05267c15127e41d497ddfc985328e59992ea53`；
- 初步受影响面：host clock/value/conversion/sleep 与 injectable/manual clock 的公开自然形状；从 Time
  迁出 timer headers/source ownership；Platform/Sched/IO 的直接边；`hyle::time` façade 的双 factor
  聚合；correctness、P1、三平台三交付、architecture/user 文档；
- 初步 tier：`P1`。monotonic/profiling/coarse reads 与 timer lifecycle 都可能处于 runtime 热路；本轮
  只在 API freeze 后分别冻结 host-clock 与 timer-queue 正常 workload，不用 clock-call 亚纳秒点值制造
  接口；
- 暂定非目标：device clock、timezone/tzdb/locale、协议文本、callback/executor、周期/跨线程 timer、
  cancellation propagation、Sched/IO 重写、Native FFI/C API、稳定 C++ ABI、公共仪器改造与 Formal。

当前只登记足以击穿旧结论的新事实；closure card、完整 public trace、有限 `FIX` 集和 API freeze 尚未
完成，因此不授权产品代码修改。

## Closure card（`SCOPED`）

- component / owner / round / kind：`time` / Time / `time-v3-host-clock-factor` / `REOPEN`；
- baseline：`25b13f5a18a0164a5fa81d01c4ef5d1a32f85cf4`；trigger：
  `TIME-FACTORIZATION-2026-09-02`，证据即上节 `TIME3-R1..R3`；
- 真实 consumer / normal workload：Platform 文件时间值、Sched/IO deadline 与批次读钟、Obs 时间戳；
  北极星旅程是 llama.cpp/ggml 在加载、采样和 backend 阶段的 elapsed 计费，以及不等待
  真实时间的 scheduler/timeout 确定性测试；
- 范围：host monotonic/realtime/coarse/profiling/CPU clocks，duration/deadline/`Instant`，Unix/OS
  conversion，blocking thread sleep；新增可编译期注入的 system/manual monotonic provider 与可选的
  小型借用型 runtime reference；从本 factor 移除 timer-node storage 所有权，但保持用户级
  `hyle::time` product facade；
- 非目标：timer callback/executor、periodic/fixed-rate 策略、跨线程 arm/cancel、fiber waiter 唤醒、
  device clocks/host-device correlation、tzdb/calendar/locale/协议文本、Sched/IO 状态机重写、Native FFI/C API、
  稳定 C++ ABI、公共性能仪器改造和 Formal certification；
- 直接依赖：完成后 `time -> PUBLIC core`，标准库/compiler builtin/OS clock API 是机制合同，
  不是第三方产品依赖；直接消费者：`timer_queue`、Platform、Sched、IO，Obs 只按值语义选择读钟而
  不强制建立产品边；
- tier：`P1`。默认 monotonic/profiling 读钟可位于 inference/runtime 高频路径；手动时钟和类型
  擦除只在显式选择时付费；
- normal performance shape：一个 phase/event-loop 批次仅采样一次 host monotonic time，完成
  begin/end elapsed 和饱和 deadline 算术；primary 对照直接 `std::chrono::steady_clock`/等价 OS 路线，
  ggml `ggml_time_us` 只作真实旅程来源而不把微秒精度冒充同合同 peer；holdout 为一次显式
  runtime clock-reference 采样加 deadline 计算；
- 常态成本合同：默认/templated system route 无 allocation、lock、virtual dispatch、registry 或新 metadata，
  不多于底层钟本身的读取；`ManualClock` 只有一个 lock-free 64-bit atomic state，
  `now()` 一次 load，`try_set()/advance()` 在无竞争时 O(1)，竞争时 CAS 重试次数随竞争而变，
  不拥有 waiter；显式 runtime reference 为两字大小、一次间接调用、无分配，生命周期由 owning
  runtime 保证；
- material margin / resource budget：W1-WIN 同合同 primary `5%`；默认路线代码形状不得多出
  间接调用/分支/常态存储，注入路线不得分配、系统调用或管理 waiter；
- 调查/性能预算：本 host-factor 轮总预算 3 小时，性能取证 45 分钟；异常后最多一次
  复现/反序、一次 codegen/profile 机制检查、一个产品修改与一次重跑；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 primary 固定 W1-WIN +
  `closure-perf-v1`，Formal 仅 `PENDING/formal-release-v1`；
- strong peers：Seastar `991cf9a908482a3f11ca2fc34a1b3970a06d8622`
  `manual_clock.hh/lowres_clock.hh`；Abseil `cc05267c15127e41d497ddfc985328e59992ea53`
  `clock_interface.h/simulated_clock.h`。默认 host clock 还受 C++23 chrono 和各 OS clock/sleep 合同约束；
- provenance：已有 clock/deadline 为 `REAL_CONSUMER`，ggml/llama phase timing 为
  `NORTH_STAR_JOURNEY`，injectable/manual clock 为两个成熟 peer 与确定性 runtime 共同形成的
  `PROVISIONAL_ARCHETYPE`；
- correctness gates：normal、zero/past、boundary/overflow、invalid programmer contract、move/destroy/lifetime、
  concurrent manual reads/advance、property/stress、ASan/UBSan；OOM 对零分配 clock value `NOT_APPLICABLE`，
  cancellation/close、real ISA/backend 对 host factor `NOT_APPLICABLE`；
- delivery gates：public-header self-containment/macro composition，Time 直接消费者编译，Windows/WSL/macOS，
  source embedding + relocated installed static/shared C++ consumer，旧 C/Native FFI surface 反证；
- 预期输出：只依赖 Core 的 host-time factor，完整且自然的 injectable/manual clock API，
  `timer_queue` 所有权迁出所需的已冻结 host API，P1/correctness/delivery 证据，同步 architecture/user/task；
- 停止条件：有限 `FIX` 完成后按 candidate 裁决；新 timezone/device/executor 需求只路由，
  相邻 Sched/IO 状态机问题记 backlog；仪器不合格则 `BLOCKED`，不在 Time 任务里扩建仪器。

`SCOPED` 只冻结上述合同；当前仍不授权产品实现。必须先完成 capability map、全部
public-surface 反向追踪、依赖/交付影响与有限 `FIX` 集，并进入 `AUDITED`。

## Capability-family coverage map（`AUDITED`）

| Family | admitted journeys / evidence | required variants and lifecycle | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| precise host clocks | Sched/IO deadline，ggml/llama phase elapsed | monotonic/realtime 分域，typed + compact，跨线程可比 | `monotonic/realtime_*` | `COMPLETE` | wall clock 不能用于 deadline |
| coarse clocks | Obs/rate-limit/event-loop 批次采样，Seastar lowres | monotonic/realtime 分域，availability，precise fallback | `coarse_*` | `COMPLETE` | cached reactor tick 的 update policy 属 owning runtime |
| profiling clock | inference phase/kernel instrumentation | raw 与 ordered pair，availability/calibration/fallback | profiling API | `COMPLETE` | device timestamp/correlation 属 Compute backend |
| CPU accounting | worker/process diagnosis | current thread/process typed clock，sleep 不累计 | CPU-clock API | `COMPLETE` | arbitrary thread/child process 未准入 |
| duration/deadline/elapsed | Sched/IO timeout，ordinary C++ | integral/floating 换算、向上取整、饱和、past/overflow | helpers + `Instant` | `PARTIAL` | 全部 convenience 固定读全局钟，无 provider overload |
| injectable monotonic time | deterministic scheduler/timeout/retry tests | system/manual，now/forward-set/advance，concurrent readers，borrowed runtime selection | 只能手动向 queue 传 now | `MISSING` | 缺 provider concept、manual value 和窄 runtime ref；waiter 不属 Time |
| Unix/OS interchange | Platform fs、IO metadata、协议边界 | pre-epoch、canonical ns、chrono/time_t/timespec/FILETIME 双向 checked | `UnixTimestamp` family | `COMPLETE` | tzdb/text formatting 路由标准/协议 owner |
| blocking host wait | tools/control thread/download retry | relative/absolute monotonic，zero/past/EINTR/oversleep | `sleep_for/until` | `COMPLETE` | fiber wait/cancel 属 Sched；manual clock 不伪造阻塞 sleep |
| timer-node data structure | Sched timer service、IO timeout node | caller-owned stable node，arbitrary/common duration，cancel/drain/teardown | 当前误由 `time` factor 携带 | `ROUTED` | 迁到 `timer_queue`；callback/wait/cross-thread policy 不迁入 |
| C++ delivery / legacy | source/static/shared users | header/compiled visibility，selected product facade，无伪 FFI | `hyle::time` + `hyle_time` | `PARTIAL` | host-only consumer 被 intrusive closure 污染；两个无消费的 unit sleep wrapper |

### Capability / gap ledger

| ID | class / demand | source | normal shape / owner | ownership / failure / concurrency | complexity/resource | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|
| `TIME3-CLOCK-01` | MUST / current consumers | Platform/Sched/IO + chrono/OS | precise/coarse/profiling/CPU clocks / Time | value API；native failure 按现有合同 fail closed/fail-fast | 默认 precise 无 HYLE 额外 hop | `COMPLETE` | `KEEP` | 现有 clock suite + P1 primary；不声称跨机器 domain |
| `TIME3-VALUE-01` | MUST / current consumers | Platform fs + IO | Unix/OS value conversion / Time | value-owned，invalid/range 用 `optional` | constexpr O(1)、零分配 | `COMPLETE` | `KEEP` | 边界/round-trip tests；不包 tzdb/parser |
| `TIME3-SLEEP-01` | MUST / ordinary thread | llama retry + OS/chrono | blocking `sleep_for/until` / Time | current OS thread，不可 cancel | one native wait，interrupt 重试 | `COMPLETE` | `KEEP` | 不早醒/zero/past；fiber 路由 Sched |
| `TIME3-INJECT-01` | MUST / admitted general | Seastar manual clock + Abseil simulated clock | `SystemMonotonicClock`、`ManualClock`、`MonotonicClockSource`、`MonotonicClockRef` / Time | manual identity noncopyable，relaxed atomic time value，只允许向前；ref 借用 source | typed route zero dispatch；ref 两字+一间接；零分配 | `MISSING` | `FIX` / blocker | compile-only generic/runtime journey，concurrent set/advance stress；不管 waiter |
| `TIME3-DEADLINE-01` | MUST / injection composition | Sched/timeout tests | provider-first `deadline_after[_ns]`、`remaining[_ns]`、`Instant::now/elapsed` overload | source 仅借用于一次调用，沿用饱和合同 | 直接模板可 inline；ref 只一次间接 now | `PARTIAL` | `FIX` / blocker | manual-clock 无真实 sleep 的 deadline/elapsed tests |
| `TIME3-API-01` | GENERAL / usability reverse trace | public header + zero consumers | 移除 `sleep_for_us/ms`，保留 chrono `sleep_for` 与 raw ns primitive | 无兼容承诺；测试改用 chrono | 无产品成本差 | `PARTIAL` | `FIX` / cleanup | stale-symbol scan；不删有真实 compact-state 用途的 `*_ns` |
| `TIME3-FACTOR-01` | MUST / accepted architecture | `B-F05/B-F06` | `time -> core`；timer headers 属 `timer_queue`；product facade 聚合两 factor | host artifact 不拥有 node；header-only queue 单 owner | host 工作集不携 65 list heads/hook 依赖 | `PARTIAL` | `FIX` / blocker | selected-factor graph、header ownership、Platform/Sched/IO direct-edge audit |
| `TIME3-POLICY-01` | GENERAL / north-star | llama server + Seastar runtime | periodic/cross-thread/wait/cancel/budget propagation / Sched/IO | executor 拥有 callback/waiter/cancel lifetime | 不给 clock/node 增 lock/owning callable | `ROUTED` | `DEFER` | Sched/IO 后续 ledger；本轮负面扫描 |
| `TIME3-DELIVERY-01` | MUST / package contract | source/static/shared consumer | `hyle::time` 对用户不变，developer factor 独立 | 无 C/FFI/stable ABI | header-only provider/queue 不造假 binary | `PARTIAL` | `FIX` / blocker | 三交付+重定位+Windows DLL stage |

### Public-surface 与依赖反向追踪

- `clock.h`：`Duration`、四个 clock/time-point alias；precise/coarse/profiling/CPU raw + typed 入口；
  duration/deadline/remaining/sleep 家族与 `Instant`。全部能映射到 `TIME-A-01..05,07..08,12`；
  只有 `sleep_for_us/ms` 无 consumer 且与 chrono API 重复；
- `unix_timestamp.h`：`UnixTimestamp/UnixTimePoint`、valid/factory，chrono/time_t/timespec/FILETIME 双向转换，
  映射 `TIME-A-02,10`；无单向 interop 缺口；
- `timer_entry.h/timer_queue.h/common_timeout_queue.h`：全部映射 `TIME-A-13..15`，语义保留但
  artifact/header owner 迁到 `timer_queue`，将在该 factor 自己的 `SCOPED/AUDITED` 卡中冻结精确修改；
- 直接消费：Platform 公开 `UnixTimestamp`；Sched 公开 monotonic value 并在实现使用 queue；IO 公开
  timestamp 并在 `net_timeout.h` 实体化 `TimerEntry`。因此迁移后 IO 对 `timer_queue` 是当前源码的
  真实 public edge；Sched 的 public forward declaration/private implementation edge 在本轮不改写 API，依安装
  include audit 与 direct-consumer 结果裁决可见性；
- 语义重复扫描：Obs 的 `std::chrono::steady_clock` 是无额外 HYLE 成本的同一标准类型，
  且接受显式 ns，不为“内部必须用 HYLE”强造依赖；Container 的 process hash seed 使用启动
  钟只是非密码乱数混入，不是 deadline 语义，留给 Container/Platform 后续 owner；Platform
  `touch` 的 native 当前文件时间是 filesystem mechanism，不重复 Time value policy。

本轮冻结 `FIX` 仅有：

1. `TIME3-INJECT-01`：实现 system/manual provider concept 和零分配 runtime ref；
2. `TIME3-DEADLINE-01`：让 deadline/remaining/Instant 自然组合显式 provider，不改默认热路；
3. `TIME3-API-01`：删除无消费且重复的 `sleep_for_us/ms`；
4. `TIME3-FACTOR-01/TIME3-DELIVERY-01`：与已登记 `timer_queue` factor 一起迁移 header/target/
   product/direct-edge 所有权，更新测试、三交付、架构与用户文档；
5. 在固定接口后创建 host-clock normal consumer P1 contract，回放已有 timer P1 时不混合两个
   factor 的结论。

相邻 backlog：Sched 后续选择 compile-time provider 或 `MonotonicClockRef` 并拥有 manual advance 后的
poll/wake；periodic/cancellation budget 仍属 Sched/IO。这些不是 Time 本轮未完成 `FIX`。

## API / usability freeze

- 状态：`FROZEN`；
- 新公开形状：`SystemMonotonicClock`、`ManualClock`、`MonotonicClockSource`、
  `MonotonicClockRef`；provider 均提供 `now() -> MonotonicTimePoint noexcept`；
- `ManualClock`：默认从 epoch 0 开始，可指定非负 initial point，不 copy/move，`now()` 线程安全，
  `try_set()` 只接受不早于当前值的 point，`advance(Duration)` 只接受非负 delta 并饱和；
  atomic 只发布时间值，不是用户数据的 synchronization fence；
- `MonotonicClockRef`：默认绑定 system route；只可从满足 concept 的 lvalue 构造，禁止 temporary，
  source 必须比 ref 长寿；两个 pointer word，无 heap/virtual base/registry；
- overload：`deadline_after[_ns](source, timeout)`、`remaining[_ns](source, deadline)`、
  `Instant::now(source)` 和 `instant.elapsed(source)`；现有无 source 入口原样保留并直接 inline 读 system clock；
- 不提供 manual `sleep_for/until`：“时间如何推进后唤醒哪些 waiter”是 owning runtime 状态机，
  Time 只提供值和取样 seam；
- compile journey：

  ```cpp
  hyle::time::ManualClock clock;
  const auto deadline = hyle::time::deadline_after(clock, 30s);
  clock.advance(25s);
  const auto left = hyle::time::remaining(clock, deadline);

  hyle::time::MonotonicClockRef selected(clock); // clock outlives selected
  const auto sampled = selected.now();            // one explicit indirect call
  ```

- 误用诊断：temporary 构造 ref 在编译期删除；manual 倒退 `try_set` 返回 false，负
  `advance` fail-fast；realtime/manual 类型不互换；用户 guide 必须说明默认/模板/ref 三种选择；
- 冻结后性能只可调整 provider 内部 codegen/atomic 实现或 clock mechanism，不得改名、
  加入 fast/light API、默认 dispatch 或常态 metadata。

## Owner 六问（`AUDITED` 直答）

1. **第三方：**Time 产品路径无直接或间接第三方实现；Core 是 HYLE 依赖。C++23
   chrono、compiler intrinsic 和 Windows/POSIX clock/sleep API 是标准/平台 mechanism。Seastar、Abseil、
   libevent、llama.cpp/ggml 本轮都只是 peer/journey 证据，不进入产品 artifact。
2. **推理能力：**llama.cpp `b96806d96061049a5b574269b049bf6241d63d46`、随仓 ggml tree
   `bc886da1a6c9d69660b0043d407200015667d609` 与 standalone ggml
   `d4716378882593333721eb33f153144b6885caf2` 已 clean fast-forward 确认。host elapsed/wall/retry/
   deadline 全部 `COMPLETE`，device clock/tz text/timer execution 都有完整 owner route；当前唯一未处置
   `MUST` 是 `TIME3-INJECT/DEADLINE` 两个已冻结 `FIX`。
3. **性能：**`P1`，当前只是 `PLANNED`。默认路线将以 source-equivalent chrono/OS 读钟与
   批次 elapsed/deadline consumer 资源计费验证 5% non-inferiority；在 candidate 和仪器资格化前
   不声称 SOTA，manual/ref 只签明确的 atomic/indirect-call 成本。
4. **C ABI/Native FFI：**`N/A/CLEAN`。没有 C header/schema/bridge、未修饰 C export、managed sample、
   FFI option/target/package metadata；private OS C API 与 shared C++ decorated symbol 不是产品 C ABI。
5. **三交付：**baseline 的 Time product source/static/shared 已存在，但候选因子图仍 `PARTIAL`；
   必须完成 host-only factor、`hyle::time` 双 factor facade、installed header ownership、Windows DLL stage 和
   relocated exact-package consumer 后才能 `PASS`。
6. **C++ API/文档：**typed/raw 域、饱和边界、checked OS conversion、sleep 与 timer 选型已自然；
   当前因缺 injectable composition 和存在只由测试使用的 unit wrappers 为 `FAIL`，按上述
   API freeze 修复。architecture 将分开 host/timer factor 决策，user guide 将补默认/注入选型和
   lifetime/atomic-not-a-fence 陷阱，performance 页只放取证事实。

## Candidate

- product commit / tree：`381eb4b195973ed35ed23ca63c05f9605a132f9b` /
  `a5d48a259161c1a60e19450e42b4a6eac9a1a712`；
- 能力/API：`SystemMonotonicClock`、`ManualClock`、`MonotonicClockSource` 与两字借用型
  `MonotonicClockRef` 已实现；deadline/remaining/Instant 已可组合 provider，默认 system route 不经
  type erasure；无消费者的 `sleep_for_us/ms` 已删除；
- correctness/resources：ManualClock 非 copy/move、非负 initial/delta、单调 set、饱和 advance、多线程
  read/set/advance 和 provider deadline/elapsed 均有测试；实现固定一个 always-lock-free 64-bit atomic state，
  无 allocation/lock/syscall/waiter/global registry；relaxed atomic 不是用户数据的 publication fence；
- factor/delivery：host `time -> core`，timer node/queue 全部迁出到 `timer_queue`；`hyle::time`
  仍是聚合两 factor 的 consumer-facing facade。W1-WIN MSVC 19.50 Release 的 Time/timer_queue
  focused tests 4/4、host-only selected-factor closure、中央 component contract 与 Sched/IO direct-consumer
  shared build 已通过；
- P1 contract：新 host normal-batch contract SHA-256 为
  `707029c26eb677dcecfc1478cca207d6a2a5d64b305eb04c43391f14d0cb158e`；一次未资格化 probe 的
  HYLE/std 差约 2%，只证明未见明显机制退化，不是 closure verdict；
- pending：W1-WIN 合格 P1 bundle、W1-WSL/M1-MAC correctness/sanitizer、最终 source/static/shared
  relocated journey、docs/red-team/aggregate gates。未运行项不写 `PASS`。

`AUDITED -> CANDIDATE` 的前态 registry 快照位于
`C:\hyle-scratch\time-v3-audited-before-candidate.json`，SHA-256
`53B2350D98372F57D5DAB398BC24F02408EB26B96FF80170B008E1A78CD4F75A`。candidate 后若改变公开合同、
产品实现、factor graph 或冻结 workload，必须退回 `AUDITED`；只补验证与事实记录不改变
candidate identity。

## Verification manifest

验证绑定 immutable product candidate `381eb4b195973ed35ed23ca63c05f9605a132f9b` / tree
`a5d48a259161c1a60e19450e42b4a6eac9a1a712`。candidate 后提交
`109cdfc4c` 只删除 Sched/IO 测试对已由产品 target 传递的 `hyle_time` 重复链接；
`include/hyle/time`、`src/time`、产品 CMake graph、公开 layout 和性能实现相对 candidate
零变化。

| Gate / risk | Verdict | Evidence / limit |
|---|---|---|
| normal/zero/past/max/conversion/provider | `PASS` | W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21 的 Time/timer_queue public-header + focused tests 各4/4 |
| ManualClock ownership/concurrency | `PASS` | noncopy/move、negative fail-fast、stale set、saturation、concurrent advance accumulation 与 read/set/advance non-regression；W1-WSL ASan/UBSan/TSan 各 2/2 |
| factor/header ownership | `PASS` | host-only factor 只解析 `core,time`；中央 component contract `PASS`；`timer_queue` 独立 owner，Container 不伪造不完整 Time facade |
| source/shared/static delivery | `PASS` | W1-WIN/W1-WSL/M1-MAC 的 selected `time` source + relocated shared + relocated static consumer 全部通过，新 ManualClock/provider sample 真运行 |
| Platform/Sched/IO direct consumers | `PASS` | 三平台 Sched/IO/timeout focused tests 各3/3；Mac 在 test-only 去重链接后重链 2/2，duplicate archive warning 消失 |
| P1 engineering performance | `PASS` | host `time-host-consumer-v1` 在4/4 strata 为 `PARITY` 且 5% NI 全通过；deep verifier `QUALIFIED/EXACT/CURRENT` |
| C ABI / stale surface | `PASS / N/A` | source/header/sample/CMake 对 `extern "C"`/Native FFI/`sleep_for_us/ms` 负向扫描为空；Windows DLL 12 个 export 全为 C++ decorated name |
| usability/docs | `PASS` | architecture 划分 host/timer factor，guide 提供选择表、system/manual/ref 旅程和 lifetime/atomic/waiter/sleep 陷阱；性能页已删除 v1 过时证书 |

P1 retained bundle 位于
`C:\hyle-scratch\time-v3-host-perf-evidence\manifest.json`；contract/manifest/summary SHA-256 分别是
`707029c26eb677dcecfc1478cca207d6a2a5d64b305eb04c43391f14d0cb158e`、
`F5A13CC5706F1E61FCC4E9B515416016FA34E0CAB7D368A1A1908E7F9209E65B`、
`A40742ED8829F988993A2552F15631A5534DB2B5F3BB000BE9E7077E8CE13745`。四个 strata 的 median
advantage ratio 为 0.9778--0.9799，99.26% CI lower 最低 0.9575，高于 5%
non-inferiority limit 0.95238；因此准确声明是同合同 `PARITY`，不是全面胜出。

## Independent red team

2026-09-02 在 candidate 和 qualification 后只攻击冻结 scope，不新增 API 或 benchmark 姿势。

| Finding | Disposition |
|---|---|
| closure card 曾把 ManualClock write 写成至多一次 atomic | 修正为一个 lock-free state；read 一次 load，write 无竞争 O(1)、竞争 CAS retry 不承诺有界；实现不变 |
| runtime ref 可能隐藏 owner/allocation/default dispatch | 反证：两 pointer word、temporary 编译期禁止、零 heap/registry；无参默认 API 不经 ref |
| ManualClock 可能被误当 user-data fence 或 waiter service | 代码注释、architecture/guide/sample 全部明示 relaxed-value-only 和 owner-polls-after-advance |
| 共享 `include/hyle/time` 让默认 LOC 边界把 timer 归入 host | 使用 `loc-report.py --module-path` 冻结 owned-header 清单；Time 1,192 + timer_queue 438 = 聚合 1,630，不修改公共工具 |
| Mac 直接消费测试重复显式链接 Time archive | test-only `109cdfc4c` 删除出产品 target 已传递的冗余 DEPS，重链无 warning、测试 2/2 |
| host 点值稳定约低 2% | 冻结 5% 正常 aggregate 的4/4 strata 均 `PARITY/PASS`，无机制级多读钟/间接调用；不用此差距扭曲 API/layout |

红队未发现未处置 `MUST`、correctness/security/lifetime/delivery 缺陷或越过冻结 5%
margin 的正常 consumer 回退。

`CANDIDATE -> VERIFIED` 的前态 registry 快照位于
`C:\hyle-scratch\time-v3\transitions\module-status-before-verified.json`，SHA-256
`A6FDF69D1C1E5FBF0E068ACD118B8F821CF5F5D3D472CDF25B26D55063348116`。

## Closure decision

本轮 `CLOSED`。capability/public-surface、architecture/dependency、correctness/sanitizer、
source/static/shared delivery、C++ usability/docs、independent red team 和 P1 engineering performance
七门全部 `PASS`；所有准入 `MUST` 已为 `COMPLETE` 或有明确上层 route。Formal 独立
`PENDING`，不阻塞工程收口。

可接受限制是：P1 只签 W1-WIN 的 64-phase host aggregate 对 source-equivalent precise
steady-clock route；ManualClock 只是 relaxed atomic time value，不是 fence/waiter/scheduler；
MonotonicClockRef 借用 source 并在显式选择时付一次间接调用；timezone/device/executor
政策仍属其 owner；无 Native FFI/C API 或稳定 C++ ABI 承诺。完整 limitation 与 reopen
trigger 已写入 `module-status.json`，没有以这些明示边界伪造未完成产品缺口。

终态 end/registry anchor 是 VERIFIED commit
`e8fb222bf975f76e45e777e5db3a7854db1d23e0`；product candidate 仍是
`381eb4b195973ed35ed23ca63c05f9605a132f9b` / tree
`a5d48a259161c1a60e19450e42b4a6eac9a1a712`，终态文档提交不冒充产品实现。
`VERIFIED -> CLOSED` 的前态 registry 快照是
`C:\hyle-scratch\time-v3\transitions\module-status-before-closed.json`，SHA-256
`3AEE61B58E39C9A8F51C5B0022A52E791DB76C9D4AFC3D960478CB203F06F412`。

四项 LOC 从 immutable candidate 执行：

```powershell
python scripts/loc-report.py --module time `
  --module-path include/hyle/time/clock.h `
  --module-path include/hyle/time/unix_timestamp.h `
  --module-path src/time `
  --commit 381eb4b195973ed35ed23ca63c05f9605a132f9b --self-check --json
```

`self_check=pass`；显式 owned-header 边界是因为 host 与 timer factor 为保持用户 include
形状共享 `include/hyle/time`：

1. 全库代码：420,468；
2. 自实现包含测试：101,897；
3. 自实现不包含测试：63,430；
4. Time host 实现：1,192。
