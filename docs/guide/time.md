# HYLE time：什么情况选择什么

`time` 提供时钟、deadline、OS 线程 sleep 和无分配 timer 数据结构。它不替 event loop 执行
callback，也不替业务决定周期 timer 的补发策略。纯 C/FFI 用户先在这里选能力，再到
[`c-time.md`](./c-time.md) 查 Status、opaque handle 与 visitor 合同。

## 一页选择表

| 需求 | 选择 | 不要选择 | 关键合同 |
|---|---|---|---|
| elapsed、timeout、deadline | `monotonic_now()` / `monotonic_ns()` | realtime | 单调、零点无意义 |
| 日志、协议、Unix epoch | `realtime_now()` / `realtime_timestamp()` | monotonic 当日期 | 可因校时跳变；跨边界优先分离秒/纳秒 |
| 普通 C++ duration/deadline | chrono typed 入口 | 手写 ns 乘法与加法 | 转换向上取整并饱和 |
| 简单作用域耗时 | `Instant::now()` + `elapsed()` | profiling clock 默认化 | 类型安全、差值可为负 |
| 高频低成本 profiling 采样 | `profiling_clock_ns()` | timeout / 精确边界 | 先查 availability；不序列化周围指令 |
| 精确 profiling 区间 | `profiling_measurement_start_ns()` / `stop_ns()` | 两次 raw sampling | 阻止编译器/CPU 跨边界重排；仍不是 deadline clock |
| 高频粗粒度日志时间戳 | `coarse_realtime_ns()` | 精确事件排序 | coarse 源可能一个 tick 内不变 |
| 阻塞当前 OS 线程 | `sleep_for` / `sleep_until` | sched worker 上调用 | 至少睡到 deadline；0 是 no-op |
| 挂起 HYLE fiber | `hyle::sched::sleep_*` | `time::sleep_*` | 不阻塞 worker |
| 混合/任意 deadline，大量 cancel | `TimerQueue` | heap / common queue | O(1) arm/cancel；single-owner |
| 海量相同 duration timeout | `CommonTimeoutQueue` | general queue 当最紧特化 | arm 顺序必须形成非递减 deadline |
| 多种固定 duration | 每种一个 Common queue + 业务路由 | 一个 queue 混装 | 路由策略属于消费者 |
| 周期 timer | 一次性 timer 到期后按业务策略 re-arm | 底层隐式 repeating | 明确 skip/catch-up/drift/cancel 语义 |

## 时钟和 deadline

日常 C++ 代码优先 typed 形状：

```cpp
#include "hyle/time/clock.h"

using namespace std::chrono_literals;
const auto started = hyle::time::Instant::now();
const auto deadline = hyle::time::deadline_after(25ms);
do_work();
const hyle::time::Duration elapsed = started.elapsed();
const hyle::time::Duration left = hyle::time::remaining(deadline);
```

系统边界、wire/FFI 或极热代码可用 `uint64_t` ns：

```cpp
const std::uint64_t deadline = hyle::time::deadline_after_ns(timeout_ns);
const std::uint64_t left = hyle::time::remaining_ns(deadline);
```

不要写 `monotonic_ns() + timeout_ns`，因为大 timeout 会 wrap 到过去。chrono integral duration
用精确 ratio 算术转换，包含 2^53 以上的 ns；浮点 duration 的 NaN/非正数转 0，正值向上取整到
ns，超范围饱和。`Instant` 的 `elapsed()`/`until()` 是非负
便利入口；`later - earlier` 与 `std::chrono` 一样保留方向，可能为负。

`profiling_clock_ns` 返回缩放到 ns 的硬件 counter 采样：x86 使用 invariant TSC，GCC/Clang
ARM64 使用 architected virtual counter。x86 跨核迁移、suspend 或 VM migration 可能让差值
失真，所以它只适合同一 pinned/近邻热路径；任何平台都不能把它写进 timeout。raw 入口追求
最低读取成本，不保证周围待测指令没有被重排。真正量一段代码时使用有序 pair：

```cpp
const auto start = hyle::time::profiling_measurement_start_ns();
hot_operation();
const auto stop = hyle::time::profiling_measurement_stop_ns();
if (stop >= start) record_sample(stop - start); // x86 跨核反向样本应丢弃
```

有序 pair 阻止编译器与 CPU 把普通指令/load 移过边界，但不会等待先前 store 全局可见；设备
I/O、持久化和跨核同步仍使用各自的 fence/protocol。availability 为 false 时 raw 与 pair 都
安全回退到 monotonic，只是不再有硬件快路。`coarse_realtime_ns` 是粗粒度日志时间戳；coarse
是稳定语义，是否更快只由当前平台性能证书说明。

### Unix 时间与 OS/FFI 互操作

持久化、文件元数据或 FFI 不要把 Unix 时间压成无符号纳秒。它会让 1970 年前的值 wrap，且
`int64_t` 纳秒也只能覆盖约 1677--2262。HYLE 使用 canonical `UnixTimestamp{seconds,
nanoseconds}`：seconds 有符号，nanoseconds 总在 `[0, 1e9)`，因此 `-1ns` 表示为
`{-1, 999999999}`。

```cpp
#include "hyle/time/unix_timestamp.h"

const hyle::time::UnixTimestamp stamp = hyle::time::realtime_timestamp();
const auto chrono_value = hyle::time::unix_time_point(stamp); // 超范围为 nullopt
const auto native = hyle::time::unix_timestamp_to_timespec(stamp);
```

提供与 `std::chrono::system_clock`、`time_t`、POSIX `timespec` 及 raw Windows FILETIME ticks
的 checked/exact 转换。`platform::FileInfo::modified_time` 和 `io::FileStat::modified_time`
复用同一表示，保留 pre-epoch 与亚秒精度。`realtime_ns()` 仍是当前时间的紧凑热路径入口，
不是通用存储格式。

## sleep：先问“谁在睡”

```cpp
hyle::time::sleep_for(250us);                 // 阻塞 OS thread
hyle::time::sleep_until(hyle::time::deadline_after(2ms));
```

这些入口用于裸线程、工具程序和不在 HYLE scheduler 上的控制面。fiber 中使用
`hyle::sched::sleep_for_ns/sleep_until_ns`，它把 `TimerEntry` arm 到 worker queue 后 suspend；
调用 `time::sleep_*` 会把整个 worker 阻塞。0 时长立即返回，不承担 yield 语义。

## 任意 deadline：TimerQueue

```cpp
#include "hyle/time/timer_queue.h"

hyle::time::TimerQueue timers;
hyle::time::TimerEntry request_timer(&request, on_timeout);
timers.schedule_after(&request_timer, 30s);

// event-loop owner thread
hyle::time::TimerList due;
timers.advance(hyle::time::monotonic_now(), due);
due.consume_all([](hyle::time::TimerEntry* timer) noexcept {
  if (auto fire = timer->fire_callback()) fire(timer);
});
```

一个 event-loop tick 批量 arm 时先采一次 `now`，再把它传给每个 `schedule_after`；这样所有
timer 属于同一时域并省去重复读钟，而不是缓存一个跨 tick 的旧时间：

```cpp
const auto now = hyle::time::monotonic_now();
for (auto* timer : request_timers) timers.schedule_after(timer, 30s, now);
```

queue 不分配、不执行 callback。`TimerEntry` 的地址在 linked 期间必须稳定；armed entry 必须先
cancel 才能销毁，delivery list 中的 entry 必须活到 list 被 drain 或销毁。list 析构会自动摘除
剩余节点，因此可以直接丢弃整个 delivery batch；节点随后可重新 arm。`cancel` 对 idle entry
返回 false；已在同 queue 的 entry 再 schedule 是 reschedule；另一个 queue 的 entry 是合同
错误。`advance` 时间必须非递减，仍在输出 list 中的节点不可直接 re-arm/destroy。

`next_deadline()` 返回 optional chrono time point，适合正常 C++ event loop；紧凑系统边界可用
`next_deadline_ns()`，空时为 `UINT64_MAX`。取消当前 minimum 后 next 允许保守地早醒一次，
这是用一次无害 early wake 换取 cancel O(1)，不是漏发。跨 bucket 的 delivery 不保证总序。

极小队列不必期待 radix 永远是最快常数：单 timer 时 indexed heap 可以更快；正常 per-worker
多 timeout 形状才会摊薄 radix 固定成本并体现 O(1) arm/cancel。HYLE 不为单 timer benchmark
给每个节点增加第二套索引元数据；具体 cardinality 边界只看性能证书。

## 单一 duration：CommonTimeoutQueue

```cpp
hyle::time::CommonTimeoutQueue idle_timeout(30s);
hyle::time::TimerEntry idle(&connection, close_idle_connection);
idle_timeout.schedule(&idle);  // 内部采 monotonic now + 30s
```

它适合“每个连接都是 30 秒 idle timeout”这类真实负载。若一个循环要批量 arm，先采一次
`monotonic_now()`，再对每个节点 `try_schedule_after(entry, now)`，既少读钟又保持相同 deadline
FIFO。显式 deadline 用 `try_schedule`；false 表示顺序不成立且旧 arm 未丢失，此时应改用
`TimerQueue`。

Common queue 同样 single-owner。它不是 thread-safe timeout service；线程安全/跨核命令、
callback 执行上下文、取消仲裁属于 sched/io。

## 周期 timer 的正确写法

底层故意不提供一个含糊的 `repeat=true`。例如固定速率任务需要先选策略：

- skip：下一次从“现在 + period”算，过载不补发；
- catch-up：从上一个目标 deadline 累加，可能连续补发；
- bounded catch-up：最多补 N 次再跳过；
- fixed delay：callback 完成后再等待 period。

选择后在 delivery callback 中重新 schedule 即可。HYLE scheduler 若将来提供周期任务，应在
executor API 暴露这些 policy，而不是改变 `TimerQueue`。

## shutdown 与失败

正常 shutdown 调 `cancel_all(out)`，drain 返回节点，再销毁 queue/entry。C++ 对程序员合同
错误 fail-fast；需要处理不可信输入或 FFI 时用 C API，它把 cross-source、顺序和非空 destroy
转成 `hyle_status`。

架构和维护不变量见 [`time-design.md`](../architecture/time-design.md)，唯一正式性能表见
[`benchmarks-time.md`](../performance/benchmarks-time.md)。
