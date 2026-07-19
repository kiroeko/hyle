# time 纯 C ABI 使用指南

`include/hyle/c/time.h` 是 time 模块的稳定 C/FFI 接缝。它只暴露
`uint64_t`、函数指针和 opaque handle；不会把 C++ 类、STL、模板或侵入链表
布局带过 ABI。

所有时间值均为纳秒。超时、deadline、elapsed 只能使用 monotonic 系列；
realtime 只用于日志和时间戳。

## 1. 时钟与 deadline

| 入口 | 用途 |
|---|---|
| `hyle_time_monotonic_ns` | 稳定单调钟；deadline/elapsed 的默认选择 |
| `hyle_time_fast_monotonic_ns` | x86 invariant-TSC 快速采样；不可用于 deadline |
| `hyle_time_fast_monotonic_available` | 快速单调钟当前是否真走 TSC，否则会安全回退 |
| `hyle_time_realtime_ns` | Unix epoch 高精度墙钟；允许被系统校时 |
| `hyle_time_fast_realtime_ns` | 低分辨率 coarse 墙钟，适合高频日志时间戳 |
| `hyle_time_sleep_for_ns` | 至少休眠给定时长；0 表示 yield |
| `hyle_time_sleep_for_us` / `hyle_time_sleep_for_ms` | header-inline 单位便利函数，乘法饱和 |
| `hyle_time_sleep_until_ns` | 睡到绝对 monotonic deadline；过去时间立即返回 |
| `hyle_time_deadline_after_ns` | `monotonic_now + timeout`，溢出时饱和到 `UINT64_MAX` |
| `hyle_time_remaining_ns` | deadline 距当前单调钟的剩余量，过期返回 0 |

```c
#include "hyle/c/time.h"

uint64_t deadline = hyle_time_deadline_after_ns(UINT64_C(50000000));
/* ... */
if (hyle_time_remaining_ns(deadline) == 0) {
  /* timed out */
}
```

`hyle_time_fast_monotonic_ns` 是 profiling 工具：休眠恢复、虚拟机迁移或跨核
迁移后可能出现不适合调度语义的毛刺。即使
`hyle_time_fast_monotonic_available()` 返回 1，也不要拿它生成 deadline。

## 2. timer 节点所有权

`hyle_time_timer` 是一次分配、可重复 arm 的 opaque 节点：

| 入口 | 用途 |
|---|---|
| `hyle_time_timer_create` / `hyle_time_timer_destroy` | 创建/销毁；销毁前必须未调度 |
| `hyle_time_timer_user_data` / `hyle_time_timer_set_user_data` | 存取调用方的裸上下文指针 |
| `hyle_time_timer_deadline_ns` | 最近一次成功 arm 的绝对 deadline |
| `hyle_time_timer_is_scheduled` | 当前是否属于某个 timer source |

一个 timer 同时最多属于一个 source。跨 source 重用会返回 0，并由
`hyle_time_last_error()` 给出线程局部诊断。必须先 cancel 或等它被
`advance` 取出，才能换 source 或 destroy。

timer、wheel、queue 都是单线程拥有的数据结构。跨线程操作应由调用方投递
命令给 owner 线程，而不是并发调用这些函数。

## 3. 任意 deadline：timer wheel

```c
static void on_due(hyle_time_timer* timer, void* ctx) {
  (void)ctx;
  void* job = hyle_time_timer_user_data(timer);
  /* timer 此时已脱离 wheel，可在这里重新 arm */
  run_job(job);
}

hyle_time_wheel* wheel =
    hyle_time_wheel_create(hyle_time_monotonic_ns());
hyle_time_timer* timer = hyle_time_timer_create(job);

if (!wheel || !timer ||
    !hyle_time_wheel_schedule_after(wheel, timer, UINT64_C(30000000))) {
  report(hyle_time_last_error());
}

uint64_t next = hyle_time_wheel_next_deadline_ns(wheel);
/* owner event loop park 到 next；醒来后： */
size_t fired =
    hyle_time_wheel_advance(wheel, hyle_time_monotonic_ns(), on_due, NULL);

hyle_time_wheel_cancel(wheel, timer); /* 未调度时也是安全 no-op */
hyle_time_timer_destroy(timer);
hyle_time_wheel_destroy(wheel);       /* destroy 时必须为空 */
```

导出入口：

| 入口 | 用途 |
|---|---|
| `hyle_time_wheel_create` / `hyle_time_wheel_destroy` | 创建 8×64、1ms 粒度的分层轮；销毁要求为空 |
| `hyle_time_wheel_schedule` | 按绝对 monotonic deadline arm/reschedule |
| `hyle_time_wheel_schedule_after` | header-inline 相对时长便利入口，饱和生成 deadline |
| `hyle_time_wheel_cancel` | O(1) 真正摘链；无 tombstone |
| `hyle_time_wheel_advance` | 把已到期节点交给同步 visitor，返回数量 |
| `hyle_time_wheel_next_deadline_ns` | 下一次需要唤醒/级联的绝对时间；空时 `UINT64_MAX` |
| `hyle_time_wheel_size` / `hyle_time_wheel_empty` | live timer 数量/空判断 |

wheel 的公开 deadline 是纳秒，但到期粒度为 1ms：永不早触发，最多晚一个
tick。`next_deadline_ns` 可能是较高层的级联边界，而不是最终 timer
deadline；event loop 应按它唤醒并再次 `advance`。

`hyle_time_wheel_advance(..., NULL, ...)` 仍会消费到期 timer，只是不调用
visitor。消费后 timer 可再次 arm 或销毁。

## 4. 相同时长：common timeout queue

当大量 timer 都采用同一固定时长（例如每个连接都是 30 秒 idle timeout）
时，使用 `hyle_time_common_queue`。相同 duration 加到单调 now 上后，deadline
天然非递减，因此 arm/expire/cancel 都是 O(1) 侵入链操作。

```c
hyle_time_common_queue* idle =
    hyle_time_common_queue_create(hyle_time_monotonic_ns(),
                                  UINT64_C(30000000000));
hyle_time_timer* conn_timer = hyle_time_timer_create(connection);

if (!hyle_time_common_queue_schedule_now(idle, conn_timer)) {
  report(hyle_time_last_error());
}

hyle_time_common_queue_advance(
    idle, hyle_time_monotonic_ns(), on_due, NULL);
```

导出入口：

| 入口 | 用途 |
|---|---|
| `hyle_time_common_queue_create` / `hyle_time_common_queue_destroy` | 创建固定 duration 队列；销毁要求为空 |
| `hyle_time_common_queue_schedule` | 显式绝对 deadline；必须保持非递减 |
| `hyle_time_common_queue_schedule_after` | 用调用方提供的 `now_ns + duration` arm |
| `hyle_time_common_queue_schedule_now` | header-inline 正常入口，内部读取 monotonic now |
| `hyle_time_common_queue_cancel` | O(1) 摘链 |
| `hyle_time_common_queue_advance` | 交付所有 `deadline <= now` 的节点 |
| `hyle_time_common_queue_next_deadline_ns` | 队首绝对 deadline；空时 `UINT64_MAX` |
| `hyle_time_common_queue_size` / `hyle_time_common_queue_empty` | live 数量/空判断 |
| `hyle_time_common_queue_duration_ns` | 查询该 queue 的固定 duration |

显式 schedule 若破坏顺序会返回 0。新 timer 保持未调度；失败的 reschedule
会保留原 arm，不会丢 timer。混合时长不要硬塞进 common queue，改用 wheel。

## 5. 失败、NULL 与回调规则

- `hyle_time_last_error` 返回当前线程最近一次可恢复边界错误；不要 free。
  成功的 fallible 调用会清除旧文本。
- create OOM 返回 `NULL`。schedule/cancel 的 source、timer 为 `NULL` 或跨
  source 所有权错误时返回 0。
- 查询函数对 `NULL` 返回安全默认值：size/deadline-data 为 0（next
  deadline 为 `UINT64_MAX`）、empty 为 1。
- 非空 source 被 destroy、仍调度或仍挂在 fired list 中的 timer 被 destroy，
  属于生命周期契约违反，会 fail-fast，而不是悄悄制造 UAF。
- visitor 在调用 `advance` 的同一线程同步执行。当前 timer 已 unscheduled，
  可以在 visitor 内重排或销毁；异常绝不能跨 C 回调边界。

## 6. 有意不导出的 C++ 面

| C++ 面 | C ABI 对应能力 |
|---|---|
| `Instant` | `hyle_time_monotonic_ns` + 调用方保存 `uint64_t` |
| `TimerEntry` / `TimerList` 的侵入布局 | `hyle_time_timer` opaque handle + visitor |
| `TimerSource` concept | 两组具体 opaque source 函数；C 不需要模板 concept |
| `TimerEntry::on_fire` | `advance` visitor，避免把内部节点布局或 C++ `noexcept` 函数类型暴露给 C |

这不是功能缺口：所有时钟、休眠、deadline、任意时长 timer 与固定时长
timer 能力均可从纯 C/FFI 到达。
