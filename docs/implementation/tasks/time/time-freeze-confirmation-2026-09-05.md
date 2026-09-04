# Time 冻结确认 — 2026-09-05

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` factor 的只读确认，不是无触发重开，不建立新 product candidate，也不替换
> [`time-v5-interoperability`](time-v5-interoperability.md) 的 terminal record。当前状态真相仍是
> [`module-status.json`](../../module-status.json)；其中的 `freeze_confirmation` 只把本次确认绑定到
> 明确仓库快照和本文证据。

## 绑定身份与范围

- confirmation：`time-freeze-2026-09-05`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `54ddb99fe43e4e45664f0150c4f1d84c95166b8e` /
  `19b813b2b999390f6677ce211d8ef88015b35e1b`；
- frozen Time v5 candidate / tree：
  `680c9941d7af787067afae904a8ba886c14ba851` /
  `b09b704304d943adb8d6442c3790fbe02a3048de`；
- candidate 到 reviewed commit 的 `clock.h`、`unix_timestamp.h`、`native_interop.h` 与 `src/time`
  product implementation 为 zero diff；后续只有 native interop 边界测试加固和中央 factor/product graph
  扩展触及 Time 验证闭包；
- 本次 confirmation 没有产品修改。

## 质因数与能力边界

`time` 是当前内部图中的 host-clock/value **质因数 factor**：直接且仅 `PUBLIC core`。精确选择解析为
`core,time`。把 precise/coarse/profiling/CPU clock、deadline arithmetic、blocking thread sleep、可注入
monotonic value 与 Unix/native value adapter 再拆成更多 target，既不解除依赖环，也不隔离第三方或重型
artifact；用户本来已经通过按需 include 支付头成本。caller-owned timer node/order 则已经正确拆到
`timer_queue`，避免 timestamp-only consumer 获得 intrusive closure。

普通用户选择 `hyle::time`；该 façade 聚合 `time` 与 `timer_queue`，但 façade 不是实现依赖。Platform
thread/fs、Sched、IO 和未来 Obs 可向下直接依赖 host `time`，不能让 Time 反向依赖这些策略 owner。

## 七问终答

### 1. 第三方产品路径

`PASS`。产品实现没有第三方源码、object、runtime 或第三方公开类型；直接闭包只有 Core。C++ chrono、
compiler intrinsic/inline assembly 与 Windows/POSIX clock、sleep、CPU-time API 是标准、工具链和 OS
mechanism，不冒充第三方模块。运行时 clock-domain 与 profiling-counter 决策只由 Time 持有。

2026-09-05 已刷新参考身份：Abseil
`9e15d3d6cc19ae4703f3ad7fa413a789580ec1fc`、Folly
`dea9962e1a7d7cbbbeab5d6d4fe12d72b72ddabb`、Boost remote commit
`cb94f33fcb13e90922905c2d8105508e71fa6db4`、Seastar
`143da84e51f8b5ac63a097056cef7d7e876223cf`、libevent
`d82464a277d0f42703702c4dfd9af6af38595a83`、llama.cpp
`8b4b3558f1459c13e4aa38d5c94d306a00dc6acd` 与 standalone ggml
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`。Boost 工作目录有本地改动，故只审查不可变
`origin/master` 对象；这些身份都只作能力/机制输入，不进入产品或替换冻结性能 peer。

相关增量中，Abseil `absl/time` 只有 CMake 行变化；Folly chrono/time、Seastar timer 和 libevent
common-timeout 路径相对冻结输入没有实现变化；llama.cpp/ggml 相对既有锁定输入没有新增 time/timeout
调用形状。因此没有第三方升级 trigger。

### 2. 功能超集与推理消费

`PASS`。当前自然能力路线为：

| 用户任务 | 当前路线 | 完整边界 |
|---|---|---|
| elapsed、timeout、deadline | typed/raw precise monotonic + 饱和 conversion/add/remaining | domain 分离，signed/unsigned/floating/NaN/inf/极值均定义 |
| wall metadata 与 Unix interchange | realtime + canonical `UnixTimestamp` | pre-epoch、signed s/ms/us/ns、chrono/time_t/timespec/timeval/FILETIME 双向 checked |
| 低成本非 deadline stamp | coarse monotonic/realtime + availability | 同 domain fallback，精度不冒充纳秒 |
| 高频局部 profiling | unordered sample + ordered start/stop pair | x86 invariant TSC、Arm virtual counter、fallback 与限制可查 |
| CPU accounting | thread/process chrono clock 与 raw ns | 与 wall elapsed 分型，不承担 arbitrary-thread/child policy |
| 确定性时间 | `SystemMonotonicClock`、`ManualClock`、concept、borrowed ref | compile-time 零分派或显式 runtime 一次间接调用；不自动唤醒 waiter |
| 裸线程等待 | saturating `sleep_for/until` | monotonic absolute domain、zero/past/interrupt 路线完整 |

ggml/llama.cpp 的 host timer、timeout、sleep、profiling 与 metadata 旅程都有以上自然路线。GPU/device
event 和 host-device correlation 由 Compute backend 拥有；fiber/coroutine wait、periodic、跨线程 arm、
cancel/deadline propagation 由 Sched/IO 拥有；calendar/timezone/RFC text 由标准库或协议 owner 完整承载。
这些路由不是 Time 半项 API。当前证据下未处置 `MUST` 为 0。

### 3. 现有模块与加速设施复用

`PASS`。

| Needed mechanism | Existing candidate | Verdict | Reason |
|---|---|---|---|
| compiler/environment/fatal invariant | Core | `REUSE` | 直接使用统一 environment、visibility、force-inline 与 fail-fast 合同 |
| clock/value vocabulary 与 arithmetic | C++ chrono | `REUSE` | typed clock/duration/time-point 保持标准互操作；HYLE 只补 domain、overflow 和平台边界 |
| native clock/sleep/CPU counter | owning OS API | `LOCAL` | Time 就是平台无关 clock semantic owner；转经 Platform 会形成错误反向边和第二 decision authority |
| recoverable native value conversion | Error | `NOT_APPLICABLE` | 纯值边界使用 `optional` 或 output+`bool`，无诊断 payload；OS clock primitive 与标准 `now()` 一样是 non-fallible public contract，环境不变量失败 fail-fast |
| timer ordering | `timer_queue` | `BORROW/ROUTED` | 已独立 factor；host Time 不携带 intrusive metadata |
| allocation/reclamation/container/SIMD/hash/topology | 相邻 HYLE modules | `NOT_APPLICABLE` | clock/value 路线零分配且无 bulk layout、membership、retirement、hash 或 topology policy |

全树反查只发现 Obs 的旧 `std::chrono` rate-control 路线和 Container 的 seed 路线；前者已由独立 Obs v4
SCOPED card 处理，后者属于尚未按本轮标准冻结的 Container owner。它们不证明 Time 自身重复机制，也不
授权本轮越界修改。冻结的低层 modules 没有自造第二套 clock authority。

### 4. 拟真、公平性能

`PASS`，但结论分层且不宣称 universal SOTA：

- 当前 v5 互操作轮为 `P0 STRUCTURAL PASS / timing peer comparison NOT_APPLICABLE`：fixed O(1) checked
  integer value route，零 allocation/lock/atomic/syscall/global state；header converters 可 constant-evaluate；
- 未改变的 host hot route 继续绑定 v4 W1-WIN P1 consumer：64 个 runtime phase，每个两次 precise read，
  加 elapsed、饱和 deadline 与 remaining；对同 TU `steady_clock` 等价路线的 exact 结果为 3 `PARITY` +
  1 `INCONCLUSIVE`，四项均通过 5% non-inferiority；
- `hyle-perf verify` 对保留 bundle 返回 `QUALIFIED / EXACT / CURRENT`、`paths_verified=true` 与当前策略
  eligible。current Scheduler provenance 仍为已登记的非 material
  `32ec50841311ca5db0a67f735582bcf1a49c4abdc605ae460b4d636808700ef8`。

coarse/profiling/CPU clock、sleep 和 interop 没有具名正常消费排名，只签已记录的复杂度、资源与
correctness；不能把它们外推成跨机器、全操作或 Formal 性能冠军。

### 5. 旧 C ABI / Native FFI

`CLEAN`。没有公开 C header/schema、bridge source、undecorated C export、managed sample、专用 option/
target/package metadata，亦没有只为 FFI 支付的 layout/indirection。POSIX/Win32 structs 是显式 opt-in
C++ native interop；shared library 的 decorated C++ symbols 不是 C ABI。

### 6. Source / static / shared 构建与导入

`PASS`。当前 Windows MSVC 19.50、WSL GCC 14.2 与 macOS AppleClang 21 都验证了精确 `time` factor；
三端的 `hyle::time` source embedding、relocated shared SDK consumer、relocated static SDK consumer
各自运行通过。中央全图 component contract 也通过。selected headers 同时交付 Core、
Container-intrusive、host Time 与 timer_queue 的真实闭包；只有 `hyle_time` 是编译 artifact，header-only
timer factor 不制造虚构 binary。

### 7. C++ API 易用性与文档

`PASS`。[`user/time`](../../../user/time.md) 先按 elapsed/wall/Unix/native/timeout/manual/coarse/
profiling/CPU/sleep/timer 场景选型，再就近说明 clock-domain、rounding、overflow、borrow、blocking 和
native-header 陷阱；[`time design`](../../../architecture/time-design.md) 记录长期边界与复杂度；性能页
只保存冻结命题。默认 typed chrono 路线自然，raw compact 与 profiling route 都是显式选择，没有
`fast/light/bench` 公共分叉、测试专用入口或 internal seam 泄漏。

## 当前快照验证

Windows 权威工作树中的无关既有 Container/performance 文档改动没有进入 reviewed commit；WSL/macOS
使用该提交的 disposable native copy，远端内容没有回写产品。

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact `time` factor | resolved=`core,time`；21/21 tests | same；20/20 | same；20/20 | `PASS` |
| Time product source/shared/static relocated | 1/1 each | 1/1 each | 1/1 each | `PASS` |
| current component graph | full component contract `PASS` | selected closure | selected closure | `PASS` |
| retained host P1 evidence | verifier `QUALIFIED/EXACT/CURRENT` | not a timing claim | not a timing claim | `PASS` |
| v5 P0 code/correctness | current codegen + tests `PASS` | retained sanitizer + current tests | current tests | `PASS` |

## 冻结裁决与 LOC

没有发现新的准入能力缺口、错误复杂度、material 性能缺陷、benchmark-shaped API、内部复用缺口、
易用性/文档错误、旧 C ABI 或 build/package integration 缺陷。`time` 保持 `CLOSED`，并在 reviewed
repository snapshot 上附加 `FROZEN` 标记；只有 registry 中既有 reopen trigger 的新事实才能解除。

`python scripts/loc-report.py ... --commit
54ddb99fe43e4e45664f0150c4f1d84c95166b8e --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 439,646 |
| 自实现不含测试 | 70,651 |
| `time` factor 实现 | 1,420 |
