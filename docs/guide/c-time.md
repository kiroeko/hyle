# HYLE time：纯 C / FFI 指南

本页只解释 [`hyle/c/time.h`](../../include/hyle/c/time.h) 的 C ABI。先在
[`time.md`](./time.md) 按场景选能力；C++ 用户直接使用 `hyle::time`，不要在 HYLE 内部绕 C ABI。

```c
#include "hyle/c/time.h"
```

## 时钟、deadline 与 sleep

| 需求 | 入口 | 注意 |
|---|---|---|
| timeout / elapsed | `hyle_time_monotonic_ns` | 唯一可用于 deadline 的钟 |
| 日志 / 协议时间戳 | `hyle_time_realtime_ns` | Unix epoch；会受校时影响 |
| 高频 profiling | `hyle_time_fast_monotonic_ns` | 先查 available；禁止用于 deadline |
| 粗粒度高频日志 | `hyle_time_fast_realtime_ns` | 先查 available；可能多次返回同值 |
| 相对 timeout → deadline | `hyle_time_deadline_after_ns` | 饱和，不会 wrap 到过去 |
| deadline → 剩余时间 | `hyle_time_remaining_ns` | 已到期返回 0 |
| 阻塞 OS 线程 | `sleep_for_ns/us/ms`、`sleep_until_ns` | fiber 请用 sched；0 不等于 yield |

```c
uint64_t deadline = hyle_time_deadline_after_ns(UINT64_C(30000000));
while (hyle_time_remaining_ns(deadline) != 0) {
  /* poll work or pass the remaining duration to an OS wait */
}
```

不要写 `hyle_time_monotonic_ns() + timeout`；大值会溢出。不要用 realtime 做 timeout；NTP、
管理员调时或虚拟机时钟校正都会让它跳变。

## 任意 deadline：hyle_time_queue

timer 和 queue 都是 opaque handle。timer 创建时分配一次；之后的 arm/cancel/advance 不分配。

```c
static void on_due(hyle_time_timer* timer, void* ctx) {
  struct request* req = hyle_time_timer_user_data(timer);
  req->timed_out = 1;
  (void)ctx;
}

hyle_time_queue* queue = NULL;
hyle_time_timer* timer = NULL;
hyle_status st = hyle_time_queue_create(&queue);
if (!hyle_status_is_ok(st)) { /* read message, then free st */ }
hyle_status_free(&st);

st = hyle_time_timer_create(&timer, request);
if (!hyle_status_is_ok(st)) { /* handle */ }
hyle_status_free(&st);

st = hyle_time_queue_schedule_after(queue, timer, UINT64_C(30000000));
if (!hyle_status_is_ok(st)) { /* handle */ }
hyle_status_free(&st);

size_t fired = 0;
st = hyle_time_queue_advance(&fired, queue,
                             hyle_time_monotonic_ns(), on_due, NULL);
hyle_status_free(&st);

int cancelled = 0;
st = hyle_time_queue_cancel(&cancelled, queue, timer); /* idle 也成功 */
hyle_status_free(&st);

/* queue 必须先变空；timer 与 queue 都由创建它们的 HYLE 模块释放。 */
st = hyle_time_timer_destroy(timer); hyle_status_free(&st);
st = hyle_time_queue_destroy(queue); hyle_status_free(&st);
```

`hyle_time_queue_create_at` 只给 virtual clock、确定性测试或已经持有采样值的 event loop。
用它创建的 queue 不应调用 header-inline `schedule_after`，因为后者采样真实 monotonic；应自行
形成同一时域的绝对 deadline 后调用 `schedule`。

`advance` 的 `now_ns` 必须非递减。visitor 在 `advance` 调用线程同步执行，并且调用前 timer
已经 unarm，因此 visitor 内可以 re-arm。visitor 为 NULL 仍会消费到期 timer。不同 radix
bucket 的交付顺序不保证全序。

## 同一 duration：hyle_time_common_queue

大量连接共享一个固定 idle timeout 时，建立一个 common queue：

```c
hyle_time_common_queue* idle = NULL;
hyle_time_timer* timer = NULL;
hyle_status st = hyle_time_common_queue_create(
    &idle, UINT64_C(30000000000)); /* 30 s */
hyle_status_free(&st);
st = hyle_time_timer_create(&timer, connection);
hyle_status_free(&st);

st = hyle_time_common_queue_schedule_now(idle, timer);
hyle_status_free(&st);

size_t fired = 0;
st = hyle_time_common_queue_advance(
    &fired, idle, hyle_time_monotonic_ns(), on_due, NULL);
hyle_status_free(&st);
```

正常入口是 `schedule_now`，或者多个 arm 共用一次采样的 `schedule_after(queue,timer,now)`。
显式 `schedule(queue,timer,deadline)` 只接受非递减 deadline；违反时返回 `INVALID_ARGUMENT`，
若这是 reschedule，旧 arm 保持不变。混合 duration 用 general queue；不要为了通过顺序检查
给 deadline 排序，因为那会改变 timeout 语义。

## Surface 速查

| 分组 | 动态入口 | header-inline |
|---|---|---|
| clocks/deadline/sleep | `monotonic_ns`、`fast_monotonic_ns/available`、`realtime_ns`、`fast_realtime_ns/available`、`sleep_for_ns`、`sleep_until_ns`、`deadline_after_ns`、`remaining_ns` | `sleep_for_us/ms` |
| timer node | `timer_create/destroy`、`timer_user_data/set_user_data`、`timer_deadline_ns`、`timer_is_scheduled` | — |
| general queue | `queue_create/create_at/destroy`、`queue_schedule/cancel/advance/cancel_all`、`queue_next_deadline_ns/size/empty` | `queue_schedule_after` |
| common queue | `common_queue_create/destroy`、`common_queue_schedule/schedule_after/cancel/advance/cancel_all`、`common_queue_next_deadline_ns/size/empty/duration_ns` | `common_queue_schedule_now` |

表中短名都以 `hyle_time_` 为前缀。动态入口共 37 个，inline convenience 共 4 个；inline
仍只调用动态 ABI，不暴露 C++ layout。

## Status 与生命周期

所有返回 `hyle_status` 的调用都把 Status 所有权交给调用方；成功和失败都调用
`hyle_status_free(&status)`。常见错误：

| 情况 | 状态 |
|---|---|
| create 的 out 为 NULL、advance 缺少 count out | `INVALID_ARGUMENT` |
| 同一 timer 已被另一个 queue 拥有 | `FAILED_PRECONDITION` |
| common deadline 破坏非递减顺序 | `INVALID_ARGUMENT` |
| destroy 仍 scheduled 的 timer / 非空 queue | `FAILED_PRECONDITION` |
| opaque control 分配失败 | `RESOURCE_EXHAUSTED` |

`cancel_all` 是 shutdown/错误回滚入口：它 detach 全部 timer，并可经同步 visitor 把节点归还
业务 owner。`UINT64_MAX` 同时可是真实远期 deadline 和 empty sentinel；需要区分时调用
`*_empty()`。queue 是 single-owner；跨线程 arm/cancel 必须串行投递到 owner event loop。

完整动态 surface 由 `tests/time/c_surface_symbols.txt` 机械冻结；C 头自包含门同时用 C11 与
C++23 编译。性能与平台证据只维护在
[`benchmarks-time.md`](../performance/benchmarks-time.md)。
