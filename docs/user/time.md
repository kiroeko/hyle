# HYLE time：什么情况选择什么

`time` 面向 C++23 source/static/shared 用户，提供时钟、deadline、Unix 时间互操作、OS 线程 sleep
和无分配 timer 容器。它不提供 C API/Native FFI，也不替 scheduler 执行 callback。

## 一页选择表

| 需求 | 选择 | 主要陷阱 |
|---|---|---|
| elapsed、timeout、deadline | `monotonic_now/ns` | 不要用 realtime |
| 日志、协议、Unix epoch | `realtime_now/timestamp` | 墙钟会因校时跳变 |
| 普通 C++ duration/deadline | typed chrono 入口 | 不要手写 ns 乘法或 `now + timeout` |
| 简单作用域耗时 | `Instant::now().elapsed()` | raw subtraction 可能为负 |
| 可接受粗分辨率的 rate-limit/event-loop time | `coarse_monotonic_*` | 先看 availability；多个 tick 可相同 |
| 高频粗粒度日志 stamp | `coarse_realtime_*` | 仍是会跳变的墙钟 |
| 高频局部 profiling sample | `profiling_clock_ns()` | 不能用于 timeout；raw read 不排序周围指令 |
| 精确 profiling 区间边界 | `profiling_measurement_start/stop_ns` | 不是同步或持久化 fence |
| 当前线程实际 CPU 消耗 | `ThreadCpuClock` / `current_thread_cpu_ns` | sleep/blocked 时间基本不累计 |
| 当前进程实际 CPU 消耗 | `ProcessCpuClock` / `process_cpu_ns` | 多线程 CPU 时间可快于 wall elapsed |
| 阻塞当前 OS 线程 | `sleep_for/until` | scheduler worker 上不要调用 |
| 挂起 HYLE fiber | `hyle::sched::sleep_*` | 不要阻塞整个 worker |
| 混合 deadline、大量 cancel | `TimerQueue` | caller 保证节点地址和 owner thread |
| 海量相同 duration timeout | `CommonTimeoutQueue` | arm deadline 必须非递减 |
| 日历、时区、RFC 文本 | 标准库或协议 owner | time 不带 timezone DB/parser |
| 周期任务 | executor 明示 skip/catch-up/fixed-delay | 不存在含糊的底层 repeating timer |

## Precise clock 与安全 deadline

日常 C++ 优先 typed 入口：

```cpp
#include <chrono>
#include "hyle/time/clock.h"

using namespace std::chrono_literals;

const auto started = hyle::time::Instant::now();
const auto deadline = hyle::time::deadline_after(25ms);
do_work();
const hyle::time::Duration elapsed = started.elapsed();
const hyle::time::Duration left = hyle::time::remaining(deadline);
```

系统边界或已采用 compact storage 的热路可用 ns：

```cpp
const std::uint64_t deadline = hyle::time::deadline_after_ns(timeout_ns);
const std::uint64_t left = hyle::time::remaining_ns(deadline);
```

不要写 `monotonic_ns() + timeout_ns`：大 timeout 会 wrap 到过去。HYLE 对 integral duration 做精确
ratio 换算，对正值向上取整到 ns 并饱和；NaN/非正值为 0，正无穷和超范围为最大值。这样转换不会
让请求的等待变短。

`Instant::elapsed()`/`until()` 是非负便利入口：反向为 0，过大的正差饱和。`later - earlier`
保留 `std::chrono` 的有符号语义，可能为负；对外部或极值 time point 优先用便利入口。

## Coarse、profiling 与 CPU time

coarse clock 的 value 与对应 precise clock 在同一 domain：

```cpp
if (hyle::time::coarse_monotonic_available()) {
  const auto event_loop_tick = hyle::time::coarse_monotonic_now();
}

const auto log_stamp = hyle::time::coarse_realtime_timestamp();
```

availability 为 false 时仍返回正确 domain 的 precise fallback；它只说明当前机器没有真正 coarse
快路。不要因为函数名带 ns 就假定分辨率是 1 ns，也不要把 coarse realtime 用作 elapsed。

profiling raw sample 追求最低读钟成本，有序 pair 才适合测一段代码：

```cpp
const auto start = hyle::time::profiling_measurement_start_ns();
hot_operation();
const auto stop = hyle::time::profiling_measurement_stop_ns();
if (stop >= start) record_sample(stop - start);
```

先用 `profiling_clock_available()` 决定是否值得采用。x86 上跨核迁移、suspend 或 VM migration
可能破坏局部差值；raw 入口也不阻止周围指令重排。任何情况下都不得把 profiling clock 接入 timer。

CPU accounting 用来判断“真的执行了多久”，不是“过去了多久”：

```cpp
const auto cpu_begin = hyle::time::ThreadCpuClock::now();
do_cpu_work();
const hyle::time::Duration cpu_used =
    hyle::time::ThreadCpuClock::now() - cpu_begin;

const std::uint64_t process_cpu = hyle::time::process_cpu_ns();
```

线程等待时 thread CPU time 基本停止；多线程进程可同时累计，所以 process CPU delta 可能大于 wall
elapsed。接口只计当前线程或当前进程，不接受外部 thread handle，也不聚合 child process。

## Unix 时间与 OS/持久化互操作

持久化、文件元数据或协议值使用 `UnixTimestamp{seconds, nanoseconds}`，不要压成无符号 epoch ns：
pre-1970 会 wrap，signed 64-bit ns 的范围也远窄于拆分表示。`nanoseconds` 必须在 `[0, 1e9)`，
所以 -1 ns 的 canonical 表示是 `{-1, 999999999}`。

```cpp
#include "hyle/time/unix_timestamp.h"

const hyle::time::UnixTimestamp stamp = hyle::time::realtime_timestamp();
const auto chrono_value = hyle::time::unix_time_point(stamp);
const auto posix = hyle::time::unix_timestamp_to_timespec(stamp);
const auto filetime_ticks =
    hyle::time::unix_timestamp_to_windows_filetime_ticks(stamp);
```

system-clock time point、`time_t`、POSIX `timespec` 和 raw Windows FILETIME ticks 都有双向转换。
不能表示或不 canonical 的输入返回 `std::nullopt`。FILETIME 只有 100 ns 分辨率：timestamp 转 ticks
会把纳秒部分向下取整到 100 ns，只有 100 ns 对齐值精确 round-trip。`platform::FileInfo` 与
`io::FileStat` 使用同一个 value type。

`realtime_ns()` 适合当前非负 epoch 的 compact 热路，不是通用 wire/storage format。日历、时区和
RFC3339/RFC1123 文本仍交给标准/协议层处理。

## sleep：先问“谁在睡”

```cpp
using namespace std::chrono_literals;

hyle::time::sleep_for(250us);  // 阻塞当前 OS thread
hyle::time::sleep_until(hyle::time::deadline_after(2ms));
```

这些入口用于裸线程、工具程序和 scheduler 外的控制面；保证 HYLE 不主动早于 deadline 返回，但 OS
仍可晚醒。0/past 立即返回，不承担 yield 语义。在 HYLE fiber 中使用 `hyle::sched::sleep_*`，否则
整个 worker 都会被阻塞。

## 任意 deadline：TimerQueue

```cpp
#include "hyle/time/timer_queue.h"

using namespace std::chrono_literals;

hyle::time::TimerQueue timers;
hyle::time::TimerEntry request_timer(&request, on_timeout);
timers.schedule_after(&request_timer, 30s);

hyle::time::TimerList due;
timers.advance(hyle::time::monotonic_now(), due);
due.consume_all([](hyle::time::TimerEntry* timer) noexcept {
  if (auto fire = timer->fire_callback()) fire(timer);
});
```

一个 event-loop tick 批量 arm 时只采一次 `now`：

```cpp
const auto now = hyle::time::monotonic_now();
for (auto* timer : request_timers) {
  timers.schedule_after(timer, 30s, now);
}
```

queue 不分配、不执行 callback。`TimerEntry` 的地址在 linked 期间必须稳定；一个节点同时最多属于
一个 queue。idle entry 的 `cancel` 返回 false，同 queue 再 schedule 是 reschedule，另一个 queue
操作是合同错误。`advance` 的 now 必须非递减。

`next_deadline()` 返回 optional typed time point；`next_deadline_ns()` 空时返回 `UINT64_MAX`。
cancel 当前 minimum 后允许保守 early wake 一次，这是保持 cancel O(1) 的取舍，不会漏发。跨 bucket
delivery 不保证总序；需要全序的消费者自行排序该批。

radix queue 针对正常 per-worker 多 timeout/high-cancel 负载；单 timer 时更小的 heap 常数可能更低。
不要根据一个微型 cardinality 给节点增加第二套索引 metadata。

## 单一 duration：CommonTimeoutQueue

```cpp
using namespace std::chrono_literals;

hyle::time::CommonTimeoutQueue idle_timeout(30s);
hyle::time::TimerEntry idle(&connection, close_idle_connection);
idle_timeout.schedule(&idle);
```

它适合“每个连接都是 30 秒 idle timeout”。批量 arm 时先采一次 `monotonic_now()`，再逐个
`try_schedule_after(entry, now)`。显式 deadline 用 `try_schedule`；返回 false 表示非递减顺序
不成立且旧 arm 未丢失，此时改用 `TimerQueue`。

多种固定 duration 建立多个 Common queue 并由消费者路由。该类型和 `TimerQueue` 都是
single-owner，不是带锁的跨线程 timeout service。

## Shutdown、生命周期与失败

正常 owner-first shutdown 有两种选择：

- 需要拿回节点：`cancel_all(out)` 后 drain `out`；
- 只需放弃剩余 timer：直接析构 queue；它会 O(n) detach，既不 fire callback，也不销毁 caller-owned
  entry，entry 随后可复用或销毁。

反过来的 entry-first shutdown 仍是 bug：armed entry 或仍链接在 delivery list 的 entry 不能先析构。
delivery 后必须先 pop/drain 再 re-arm 或 destroy。wrong owner、倒退的 advance 和违反这些 lifetime
合同会 fail-fast；不可信输入应由上层先验证，不能把程序员合同当 recoverable parsing error。

稳定设计见 [`time-design`](../architecture/time-design.md)，当前性能边界见
[`benchmarks-time`](../implementation/evidence/performance/benchmarks-time.md)。
