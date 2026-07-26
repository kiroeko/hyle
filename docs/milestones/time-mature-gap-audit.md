# time 成熟库能力缺口复审（2026-07-26）

> 本页是复审记录，不是新的实现真源。当前设计、用户入口和性能点值仍分别见
> [`time-design`](../architecture/time-design.md)、[`guide/time`](../guide/time.md) /
> [`guide/c-time`](../guide/c-time.md) 与
> [`benchmarks-time`](../performance/benchmarks-time.md)。本轮只回答：既有 strict closure
> 是否仍成立、成熟库还提示了哪些真实需求、哪些能力应留在相邻模块。

实施状态（2026-07-26）：

- [x] integral duration 已改为精确 checked/saturating rational arithmetic（`52595a6`）；
- [x] 旧 `fast_monotonic` / `fast_realtime` 已分别改为语义名 `profiling_clock` /
  `coarse_realtime`，未保留未发布 ABI 兼容壳；
- [x] TSC fallback 已改为 sleep-based calibration，并由 fresh-process test 记录 wall/thread CPU；
- [x] 正常使用性能矩阵已加入（`bd29d57`），Windows n=15 已记录；
- [x] 合法的 `duration<bool, Period>` 已覆盖，typed facade hot helper 显式 inline（`9129001`）；
- [x] Windows/WSL/Apple 最终形状 correctness 与 normal-use 性能已复证；WSL GCC14
  ASan/UBSan 最终形状五门均为 5/5。

## 结论先行

`time` 的 timer 数据结构与模块边界是强的：任意 deadline 使用 exact-ns radix queue，固定
duration 使用 ordered intrusive queue；两者 single-owner、热路无分配、交付与执行分离，
复杂度处于 Seastar/libevent 同类实现的 SOTA 档。C++ 的 typed chrono 与紧凑整数入口并存，
C ABI 也覆盖了现有语言无关能力。`sched`/`io` 对 time 的消费方向正确，time 对 error/platform
的依赖没有倒置。

审计当时不能继续标为“彻底 strict 收口”，原因不是要机械补齐 Abseil/Folly/Tokio 的所有
功能，而是发现了三个发布前 P0：

1. `duration_ns_saturated` 在 MSVC 上会让大于 2^53 的 integral ns 丢失 1 ns，违反“正数向上
   取整、等待不变短”的公开合同；
2. `fast_monotonic_ns` 公开承认不严格 monotonic，`fast_realtime_ns` 实际是 coarse clock；名称
   仍是 benchmark 形状而非语义形状，趁 ABI 未冻结应改；
3. time 性能证书固定在 `14ee67e`，之后直接热路依赖 `core/intrusive_list.h` 已连续改变，且现有
   timer workload 只覆盖连续内存的 N=4096 批处理，没有证明正常入口、小 N、散布节点和 C ABI。

三个实现/仪器缺口现已修复。**功能骨架完整，算法选择正确，正确性和 API 语义已修，
三平台 Release/性能与 WSL ASan/UBSan 也已复证。**
日历/时区、通用格式语言、执行器、带锁跨线程 timer queue 仍不应进入 time core。

## 1. 当前实现清单与真实用户形状

| 能力 | 当前形状 | 真实使用场景 | 评价 |
|---|---|---|---|
| typed clock | `Duration`、steady/system time point、`*_now()` | 普通 C++ deadline、elapsed、墙钟记录 | 合理，优先推荐 |
| compact clock | `monotonic_ns`、`realtime_ns` | syscall/C ABI/运行时热路的统一 ns 表示 | 合理 |
| opt-in clock | TSC `profiling_clock`、OS `coarse_realtime` | 高频 profiling、日志/trace 打点 | 能力与语义名已对齐 |
| deadline arithmetic | 饱和 add、duration 转 ns、after/remaining | sched/io timeout 与 FFI 边界 | integral 路径已精确修复 |
| blocking sleep | relative/absolute、chrono/整数 | 普通 OS 线程；不用于 fiber worker | 能力完整 |
| elapsed snapshot | `Instant` | stopwatch/阶段耗时 | oneTBB 式小便利，保留 |
| arbitrary timers | `TimerEntry` + `TimerQueue` | 每 worker 大量 timeout，常 arm/cancel | 核心算法/复杂度正确 |
| one-duration timers | `CommonTimeoutQueue` | 海量同 idle timeout 的连接 | libevent 式真实特化；尚无生产 consumer |
| C ABI | opaque timer/general/common queue + Status + visitor | C/FFI 使用时钟与 timer 容器 | 现能力覆盖完整；创建策略仍可改善 |

生产消费不是 benchmark 自证：

- `src/sched/scheduler.cc` 每 worker 持有 `TimerQueue`，用 `monotonic_ns` 驱动 advance，fiber sleep
  与 timed wait 走 `deadline_after_ns`；
- `include/hyle/io/net_timeout.h` 把 `TimerEntry::FireCallback` 用于 timer-vs-I/O 仲裁；callback/data
  因而是真实 consumer metadata，不应为了缩节点而删除；
- `CommonTimeoutQueue` 暂无生产 consumer。它有成熟库和 serving 场景依据，但当前只能声明
  “结构能力完成”，不能把 application-level 收益写成已验证；
- `sched/blocking.cc` 的 condition-variable wait 是 OS blocking primitive；`platform/fs.cc` 的
  mtime 是 platform 资源语义。两者不应为了表面统一制造 `platform -> time` 反向依赖。

## 2. 成熟库对照矩阵

本轮只读对照工作区内的 `oneTBB`、`folly`、`seastar`、`libevent`、`tokio`、`asio`、
`abseil-cpp` 与标准库，不按已有 benchmark 反推 API。

| 成熟形状 | HYLE 当前 | 判断 | 归属/动作 |
|---|---|---|---|
| `std::chrono` typed duration/time point | 已有 | 足够，不自造 Duration 类 | time 保持 |
| oneTBB `tick_count`/interval | `Instant` + chrono 差值 | 足够 | time 保持 |
| Abseil cycle clock | `profiling_clock` TSC profiler | 语义已对齐 | time 保持 |
| Linux/Seastar low-resolution clock | `coarse_realtime` 是 coarse wall；无 cached steady | 不能混为一谈 | coarse wall 已正名；cached steady 由 sched profile 驱动 |
| Seastar radix `timer_set` | 65 桶 exact-ns radix | 同算法/复杂度档 | 保持，重证性能 |
| libevent common timeout | `CommonTimeoutQueue` | 能力覆盖 | 消费接入时补应用证书 |
| Folly owning `TimeoutQueue`/HHWheelTimer | 只有 intrusive delivery primitive | 不是底层缺口 | 易用 owning 层放 sched/executor |
| Tokio/Seastar interval | 无隐式 repeat | 底层不应猜 policy | sched 增 `Burst/Delay/Skip` 等显式策略时再设计 |
| Folly injectable clock / Seastar manual clock | queue 可显式 advance，但 sched 硬编码系统钟 | 有可测试性缺口 | **P1，跨 time/sched 设计 clock seam** |
| Tokio pause/advance | queue 层已可 deterministic advance | 部分覆盖 | runtime virtual time 属于 sched 测试设施 |
| Asio waitable timer async wait/cancel | time 不拥有 executor | 边界正确 | sched/io 提供异步等待 |
| Abseil CivilTime/TimeZone/Format/Parse | 无 | 不应整包复制 | 见 UTC interchange 决策 |
| time_t/timespec/timeval conversions | 仅原生 chrono/raw ns | C/OS 边界可能常用 | **P1，随实际协议/OS consumer 评估** |
| C 自定义 allocator / in-place timer | 每 timer 创建时 global `new` 一次 | 大规模 C consumer 有成本/策略缺口 | **P1，保持热路和布局 opaque** |

完整 Abseil time 连同 cctz 在本地约 17.8k code lines。复制 calendar/time-zone 数据库既会让
Tier 2 模块体积和维护面失控，也没有 HYLE 内部 consumer，不能把“成熟库提供”误写成“基础库
必须重做”。

### UTC interchange 的窄缺口

当前 architecture 说 calendar/timezone/format/parse 可交给“标准库/专用库”。边界总体正确，
但 guide 需要更准确：本地 GCC 14 C++23 探针可编译 chrono format 和时区 API，却没有
`std::chrono::parse`。纯 C 面也没有等价入口。因此对 HTTP Date、配置、日志导入导出等场景，
“外部专用 parser”目前才是可移植答案。

若后续 io/obs 出现明确 consumer，可评估一个独立、allocation-free 的 UTC interchange 小层：
RFC3339 与必要的 RFC1123 parse/format，加明确的 offset/leap-second/error 合同。它不应扩张为
locale format DSL、civil calendar 或自带 timezone database；未出现 consumer 前保持 P1 提案，
不立即加代码。

## 3. 已修 P0：正确性与 API 语义

### 3.1 integral duration 经浮点导致 deadline 提前（已修）

`include/hyle/time/clock.h` 的 `duration_ns_saturated` 无条件先转
`std::chrono::duration<long double, std::nano>`。MSVC 的 `long double` 与 `double` 同精度；
例如 `9007199254740993ns`（2^53+1，约 104.25 天）会变成 `9007199254740992`，结果提前 1ns。
这不是理论上的超大数：长租约、缓存期限、进程 keepalive 都能跨过该边界。

实现结果：

- integral `Rep` 走精确的 checked/saturating ratio arithmetic，不经过浮点；
- floating `Rep` 单列路径，保留 non-positive/NaN -> 0、正数向上取整、infinity/overflow -> max；
- 三平台增加 2^53±边界、`int64_t` max/min、非整 ns 比例、窄/宽有符号与无符号 Rep；
- `deadline_after`、`TimerQueue::schedule_after`、`CommonTimeoutQueue` chrono 构造都由同一门覆盖。

### 3.2 `fast_*` 是性能形容词，不是稳定语义（已修）

公开注释明确说 `fast_monotonic_ns` 可能跨核回退、resume/VM migration 后跳变，且“NOT
STRICTLY MONOTONIC”。这与函数名的 monotonic 合同冲突；成熟库通常把这类值命名为 cycle
clock/profiling clock，不让用户误接 timeout。`fast_realtime_ns` 实际合同则是 OS coarse wall
clock，粗粒度才是稳定语义，是否更快是平台证据。

HYLE 尚未发布且 C ABI 未冻结，本轮已同时改 C++/C 名称、availability、surface manifest、
tests 和两份 guide，最终采用 `profiling_clock_ns` 与 `coarse_realtime_ns`，没有保留兼容壳。

## 4. 复杂度与性能证据

### 4.1 算法评价

| 操作 | `TimerQueue` | `CommonTimeoutQueue` | 结论 |
|---|---:|---:|---|
| arm | O(1) | O(1) | SOTA-class |
| cancel | O(1) exact unlink | O(1) exact unlink | SOTA-class |
| next | O(1)，cancel-min 后可保守早醒 | O(1) front | 合理，用早醒换 cancel 热路不扫描 |
| advance | O(触碰桶 + fired/rebucketed) | O(fired) prefix | 与各自 workload 匹配 |
| allocation | 0（caller-owned stable node） | 0 | 适合基础 runtime |

不需要因为成熟库另有 heap/wheel 就新增第三个通用 timer：heap 的 eager cancel 是 O(log n)，
固定 tick wheel 会引入量化与更多 bucket heads；现 radix 是 mixed exact deadline 的更好默认。

### 4.2 正常使用性能证书扩展（三平台已完成）

审计发现 `benchmarks-time.md` 的旧证书钉在 `14ee67e`。其后直接 primitive
`include/hyle/core/intrusive_list.h` 在 `0f78eb5`、`5c3ad28`、`ed9988c`、`71dabf7` 中改变了
hook unlink、push/remove/clear 和 checked/unchecked 热路合同，time headers 也有调整。即使 core
自己的测试和 microbench 已通过，也不能替代 timer 组合路径的三平台复证。

原 `timer_queue_bench.cc` 全部是 N=4096、`std::vector<TimerEntry>`、预计算 deadline 的 warm
batch。`bd29d57` 保留该算法门，并新增：

- 常见小 N：1/8/64 与 N sweep；
- heap/对象内嵌的散布节点，而非只测连续 vector；
- event loop 每次少量 arm/cancel/advance 的增量路径；
- `schedule_after` 包含一次真实 `monotonic_ns()` 的端到端成本（当前 clock 约 12--14ns，可能
  大于已登记的 2--6ns queue primitive）；
- 长寿命 reschedule/churn、cancel-min 引发保守早醒的系统成本；
- C ABI status/wrapper/visitor 与每 timer 一次创建分配；
- sub-ms thread sleep 的 overshoot/分位数，尤其 Windows high-resolution waitable timer；
- Common 接入真实 fixed idle timeout router 后的 application-level 证书。

Windows、WSL/native Linux 与 Apple 两档 QoS 的 n=15 已在机器空闲时顺序完成。N=1 时
indexed heap 确实更快，N=8 起 HYLE 总体进入胜区；M4 N=8 arm+cancel 对 Seastar-shape
仍慢 7--12%，WSL reverse churn 慢约 6.3%。散布/连续 incremental path、真实
schedule_after、C ABI 常数税与 sleep overshoot 均公开记录。SOTA 的合格含义仍是算法/复杂度
不落后、同合同正常使用无系统性退化，不是让每个 microbenchmark 行都必须绿色。

### 4.3 calibration 冷路的 P1 性能/功耗问题（已修）

当 CPUID 0x15 不可用时，`measure_tsc_hz()` 以 1/2/4/.../128ms 窗口 busy-spin，最坏累计
255ms；`fast_monotonic_available()` 也会触发它。头注释却称为“few-ms”。本地 Abseil
`absl/base/internal/sysinfo.cc` 的对应 fallback 在采样间使用 `nanosleep`，并不 busy-spin。
invariant TSC 无需靠占满 core 才继续计数。

本轮保留 exact CPUID 快路，把 fallback 改为 sleep-based 抗抢占采样，并用 test-only
fresh-process gate 测 first-call wall/thread CPU。Windows wall median 3.703ms；WSL wall/CPU
median 3.118/0.030ms。release binary 不含 test env gate，已校准后的 rdtsc 热路不变。

## 5. C++、C ABI 与错误系统

### C++

- typed normal path 与 raw hot path分工清楚；queue 直接暴露具体合同，优于旧的伪
  `TimerSource` concept；
- intrusive stable-address node 是 runtime primitive 的正确形状，不能改成默认 heap owning；
- `TimerEntry` fail-fast 生命周期适合 HYLE 内部；更友好的 RAII auto-cancel/owning callback
  应作为 sched 上层便利，不给每个底层 node 增加常态元数据；
- `create_at`/显式 `advance(now)` 已使 queue 单测 deterministic，但 scheduler 仍硬编码系统钟，
  所以完整 virtual-time 测试能力尚未形成。

### C ABI

- 37 个动态符号 + 4 个 inline convenience 覆盖当前 clock/deadline/sleep/general/common 能力；
- opaque handle 隔离 C++ intrusive layout，Status 覆盖 recoverable misuse，visitor 前已摘队列且可
  re-arm，形状正确；
- 内部 C++ 消费者不绕 C ABI；`src/time/c_time.cc` 统一使用 error 的 `hyle::Status` /
  `hyle_status`，没有私造 TLS 错误；
- 对海量 C timer，每个 opaque timer 必须单独 global `new`，无法选择 allocator 或 bulk/in-place
  storage。这不是现有能力“无法表达”，但可能是 C 使用的 P1 性能/部署缺口。可评估 allocator
  callbacks、pool/bulk factory 或带 size/alignment 查询的 caller storage，同时保持 ABI opaque；
  不要为了它制造 time -> mem 硬依赖。

## 6. 文档审计

两份 guide 已能回答最重要的“什么情况用什么”：monotonic vs realtime、typed vs integer、
profiling/coarse 的风险、thread sleep vs fiber sleep、general vs common queue、生命周期与 C Status。
architecture 也正确区分 time 数据结构和 sched/io executor。

本轮修正状态：

1. architecture 已统一 Tier 2，guide 明确不同工具链 chrono parse 能力不一致；
2. 语义改名已同步 architecture、两份 guide、C header/surface、tests 与 performance；
3. time milestone 已重新打开 strict；全局 readiness 由主任务在三平台证据齐全后统一维护；
4. performance 只在专页维护点值，milestone 只留验收摘要。

## 7. 优先级与明确不做项

### P0：time 重新 strict 收口前必须完成

1. [x] 精确修复 integral duration conversion，补边界测试；
2. [x] 把两个 `fast_*` API 改为真实语义名，整套更新 C++/C/docs/surface；
3. [x] 扩展正常使用性能矩阵并重跑 Windows/WSL/Apple 两档 QoS；
4. [x] 同步 time architecture/guide/performance/milestone；全局 readiness 留给主任务统一更新。

### P1：需要 consumer 或独立设计批准

1. [x] TSC fallback 改 sleep-based，测冷启动 CPU time 与延迟；
2. time/sched 可注入 clock seam，用于 deterministic scheduler timeout 测试；
3. sched owning one-shot/periodic timer，periodic 必须显式 missed-tick policy；
4. 有 HTTP/obs consumer 后评估窄 UTC RFC3339/RFC1123 interchange；
5. 有大规模 C timer consumer 后评估 allocator/bulk/in-place 入口与 C ABI overhead。

### 明确不加入 time core

- 完整 civil calendar、locale formatter、timezone database 或 Abseil time clone；
- 不带 missed-tick 语义的 `repeat=true`；
- 内建 executor/callback event loop，或第二套 runtime；
- 带锁的跨线程全局 timer queue；
- 仅因别库存在而再加 fixed-tick wheel/heap 默认实现；
- 自造 Duration 类、用 C ABI 作为 HYLE 内部调用路径；
- 为 benchmark 删除 owner/lifecycle 检查，或给所有节点增加没有真实 consumer 的常态元数据。

## 8. 建议施工顺序

```text
exact duration correctness
  -> semantic clock rename/C ABI surface
  -> calibration cold path
  -> normal-use benchmark matrix
  -> Windows / WSL-or-native-Linux / Apple sequential recertification
  -> docs + strict readiness close

sched review (separate module)
  -> injectable clock / virtual time
  -> owning callback / periodic policy

consumer-driven proposals
  -> UTC interchange (io/obs proves demand)
  -> C timer allocator/bulk shape (FFI workload proves demand)
```

这条顺序先修合同，再测最终形状；把 runtime policy 留到 sched，把协议/FFI 的可选层留给真实
consumer 决定，避免 time 再次为了 benchmark 或“能力数量”无边界增长。

## 9. 实施验证

- 最终产品实现基点：`16c91b4`，最终 Apple 仪器：`ba61eb5`，Release shared，`HYLE_COMPONENTS=time`；
- Windows `hyle_time_test`、纯 C、C surface、header 与 fresh calibration：5/5 通过；
- Windows timer/normal-use benchmark n=15 已记录到 performance；
- WSL correctness 5/5 与 native-ext4 timer/normal n=15 已完成；Apple 最终 commit 五门 5/5，
  timer/normal interactive + utility 各 n=15 已完成；
- WSL GCC14 最终 `16c91b4`：ASan 五门 5/5，UBSan 五门 5/5。
