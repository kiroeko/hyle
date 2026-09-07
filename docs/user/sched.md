# HYLE sched：什么情况选择什么

> 本页描述当前 Sched C++ 产品面，适用于 C++23 source/static/shared 三种交付；不描述 C API、
> Native FFI 或内部 worker-loop bridge。机械施工状态见
> [`module-status.json`](../implementation/module-status.json)。

`sched` 是单进程 CPU runtime：少量 OS worker 承载大量可协作挂起的 Fiber，并把 timer、I/O、GPU
completion 等事件统一成一次 `Waker` 唤醒。普通用户通常只需要 `Scheduler`、任务入口和同步/消息
原语；不要从 `Fiber`、fcontext 或 `sched/detail` 开始。

## 一页选择表

| 需求 | 首选形状 | 关键合同 |
|---|---|---|
| 很短、不挂起、无结果 callback | `Scheduler::schedule` | 直接在 worker stack 执行；不得 sleep/yield/block |
| 一批短小、不挂起的 indexed/chunk work，只等整批结果 | `bulk_execute` | caller-owned result slots；一个 batch 首错/取消/完成；无逐项 handle |
| 很短、不挂起、需要 typed result/join/cancel | `schedule_task` + `JoinHandle<T>` | worker stack + 一个 allocator cell；`request_cancel` 只发布协作请求 |
| 可挂起、无需结果的 fire-and-forget 工作 | `Scheduler::spawn_detached` | Scheduler 拥有 Fiber/stack；默认 stack tier |
| 可挂起且需要结果、join、cancel 或 fan-out | `Task<T>` + `spawn_task` + `JoinHandle<T>` | coroutine 首参数必须是 `AllocatorRef`；独立 guarded Fiber |
| 阻塞 syscall 或不可切片的长 CPU 工作 | `BlockingPool` + `spawn_blocking` | 在独立 OS thread 执行；body 内没有 fiber context |
| 手工拥有/恢复 stackful execution | `Fiber` | 仅低层 runtime owner；调用方负责完整生命周期 |
| 合作让出 worker | `Fiber::yield()` | 可能换 worker；不能跨 yield 缓存 thread-local 派生状态 |
| 不占用 worker 地等待时间 | `hyle::sched::sleep_for/until` | 只能在 Scheduler fiber 内调用；使用 monotonic typed time |
| 一个请求取消整棵子任务 | `CancelToken` / `CancelScope` | `is_cancel_requested()` 是检查点，不会强杀 stack |
| 等待任务结果 | `JoinHandle::join` | worker 中挂起 fiber，裸线程上阻塞线程；drop 等于 detach |
| 静态或动态 fan-out | `when_all` / `when_any` | span 版本不分配；`when_any` 不自动处理 loser |
| 一次性等待 N 个完成 | `Latch` | 计数只减不增 |
| 可增减的完成计数 | `WaitGroup` | 先 add 再发布任务，避免提前归零 |
| 一批参与者反复会合 | `Barrier` | 可复用 phase；退出成员用 `arrive_and_drop` |
| 单次/保持式信号 | `Event` | auto-reset 唤醒一个；manual-reset 保持并唤醒全部 |
| 限制并发量、公平 permits | `Semaphore` | FIFO；大请求会有意阻挡后续小请求 |
| 共享读、独占写 | `RwLock` | writer-fair；不要持读锁升级写锁 |
| 有界多生产/多消费队列 | `Channel<T>` | backpressure；send 失败会归还原 value |
| 单结果交付 | `make_oneshot<T>` | sender/receiver 都是 move-only |
| 只关心最新配置/状态 | `make_watch<T>` | 不积压历史；receiver 看到版本变化 |
| 每个订阅者都应看到每条消息 | `make_broadcast<T>` | 有界覆盖；慢消费者得到精确 lag 数量 |

## 建立 Scheduler 与提交工作

默认 worker **不绑核**，因此可以自然嵌入已有 runtime。普通构造在非法参数、OOM 或 thread 创建失败时
fail-fast；服务程序若要恢复，使用带 `Status*` 的构造并检查一次 `valid()`。`queue_capacity` 必须是
2 的幂。

```cpp
#include "hyle/mem/allocator.h"
#include "hyle/sched/scheduler.h"

auto alloc = hyle::mem::system_allocator();
hyle::sched::Scheduler scheduler(/*workers=*/4, alloc,
                                 /*queue_capacity=*/256);

auto status = scheduler.schedule([]() noexcept {
  // 短小、不会挂起或阻塞的工作。
});

status = scheduler.spawn_detached([]() noexcept {
  do_one_request();                 // 可以调用 fiber-aware wait/yield
});
```

捕获 lambda 入口要求 callable 可无异常 move 且调用为 `noexcept`，记录从 Scheduler 注入的 allocator
取得。已有 `fn(void*)` 的极限热路可以使用 raw overload，避免 callable record；这不是普通用户的
默认形状。`spawn_detached_on(worker, ...)`/`spawn_on` 只用于确实拥有 per-worker ring、scratch 或
shard state 的 consumer；普通工作保持可窃取。

需要显式 placement 时填 `SchedulerOptions::worker_cpu_order` 或 `worker_cpu_set`。默认
`ThreadConfigPolicy::kBestEffort` 允许平台降级；只有应用确实不能接受 affinity/priority 失败时才选
`kRequired`。`kLowLatency`、`kBalanced`、`kPowerSaving` 是延迟/功耗策略，不是 benchmark 档位。
worker start/stop hook 适合安装 runtime TLS、scratch 或 profiler state；hook 不得抛异常或无限阻塞
shutdown。

## 一批短任务只需要一次完成态

量化 tile、有限 graph level 或一组 caller-owned work records 通常只需要“全部结束以后继续”，不需要给
每个 item 建 `JoinHandle`。这时用 `bulk_execute`：body 接收半开 chunk `[begin,end)`，结果直接写进
调用者预先准备且互不重叠的 slot。调用同步返回，因此 callable、cancel token、输入和输出只需活到返回。

```cpp
#include "hyle/sched/bulk.h"
#include "hyle/sched/cancel.h"
#include "hyle/sched/scheduler.h"

std::vector<TileResult> output(tiles.size());
auto cancel = hyle::sched::CancelToken::root(alloc);

hyle::sched::BulkExecuteOptions options;
options.grain_size = 4;              // 每次 body 处理至多 4 个 tile
options.cancel_token = &cancel;      // 可省略

auto status = hyle::sched::bulk_execute(
    scheduler, 0, tiles.size(),
    [&](std::size_t begin, std::size_t end) noexcept -> hyle::Status {
      for (std::size_t i = begin; i < end; ++i) {
        auto item_status = decode_tile(tiles[i], output[i]);
        if (!item_status.ok()) return item_status;
      }
      return hyle::Status();
    },
    options);
if (!status.ok()) return status;
consume(output);
```

`grain_size` 是消费者的算法选择，不是隐藏 benchmark tuning：它决定一次 body 的数据局部性、callback
边界和 error/cancel 观察粒度。先从业务 kernel 自然处理的 tile 开始；工作高度不均匀或需要更快停止时用
较小 grain，callback 本身过轻时才适度合并。不要单纯为了躲内部 atomic 而放大 grain——runner 已经用
有界预领取摊销共享 cursor，并且仍逐 public grain 调 body。`max_concurrency=0` 使用 Scheduler worker 数，
也可以显式降低本批并发。外部线程和另一个 Scheduler 的 Fiber 只投递不超过 worker 数的 raw runner；
同 Scheduler 的调用 Fiber 自己参与，并只投递至多 `worker-1` 个 runner。它们共享一个 batch state，
不会为每个 item 分配、排队或创建 status/cancel/handle。同一个 callable/context 可能被多个 participant
并发调用；capture 中的可变共享状态必须自行同步，最自然的形状仍是按 chunk 写互不重叠的 slot。

body 返回的第一个并发可见 non-OK `Status` 会让各 runner 在下一次 body 调用前停止；当时已经运行的
body 会完成，所以输出可能包含一段或若干已完成 slot，失败时不能把整张输出假定为事务性回滚。内部
预领取但尚未开始的 grain 不会因此被强制执行。多个同时失败的 item 不保证按 index 选最小者。取消同样
是协作式：runner 在每次 body 调用前检查 token，并把 token 通过 `CancelScope` 暴露给 body 内的
`is_cancel_requested()`；它不会强杀正在执行的 kernel。

普通线程调用会阻塞该线程。同 Scheduler Fiber 调用会先在自己的 Fiber 栈上完成一份 chunk，只在其他
runner 尚未结束时挂起；另一个 Scheduler 的 Fiber 不会被借来替目标 Scheduler 计算。从
`schedule`/`schedule_task` 这类 worker-stack callback 中嵌套调用时，为避免单 worker 自死锁，整批在
当前 callback 上串行执行。body 无论运行在 caller Fiber 还是 worker-stack runner 上，都不得 yield、
fiber-wait、blocking syscall 或抛异常；栈较深的 kernel 还要把 caller Fiber 的 stack tier 计入预算。
动态添加的异构任务可组合 `schedule + WaitGroup`；需要逐项 result/join/cancel 时才选 `schedule_task`，
不要为了等待整批而制造一组 handles。

## 有结果任务与 fan-out

短小且确定不会挂起的 result-bearing work 直接使用 `schedule_task`。callable 与 result state 合并为一次
allocator allocation，并在 worker stack 执行；它不能调用 yield、sched sleep 或任何 fiber-blocking
wait：

```cpp
auto decoded = hyle::sched::schedule_task(
    scheduler, alloc, [tile]() noexcept { return decode_tile(tile); });
auto result = decoded.join();
if (!result) return result.error();
consume(*result);
```

只有任务确实会在 timer、I/O、channel 或其他 completion 上挂起时，才为它创建 coroutine 与 guarded
Fiber：

`Task<T>` coroutine frame 从函数的第一个 `mem::AllocatorRef` 参数取得 storage。漏掉这个首参数会编译
失败；frame OOM 和 driver/stack OOM 会成为 `JoinHandle::join()` 返回的错误。

```cpp
#include "hyle/sched/join.h"

hyle::sched::Task<int> twice(hyle::mem::AllocatorRef alloc, int value) noexcept {
  co_return value * 2;
}

auto handle = hyle::sched::spawn_task(
    scheduler, alloc, twice(alloc, 21));
auto result = handle.join();
if (!result) return result.error();
use(*result);                         // 42
```

两条路线返回相同的 `JoinHandle`，因此可以混合进入 `when_all/when_any`。它是 move-only result slot；
析构或 `detach()` 只放弃结果，任务继续运行且仍被
`Scheduler::drain/shutdown` 计数。`request_cancel()` 只设置协作取消标志，任务必须在循环或工作边界检查
`is_cancel_requested()` 并通过 RAII 自行退出。这个函数不会凭空唤醒已经挂在 timer、I/O 或同步 waiter
链上的 Fiber；这样做会绕开真正拥有底层操作和节点生命周期的 owner。

多个异构 handle 用参数包 `when_all(a,b,c)` / `when_any(a,b,c)`；运行时同类型集合用
`std::span<JoinHandle<T>>`。它们不消耗 handle，所以之后仍要 `join()`、`request_cancel()` 或明确
detach。`when_any` 只返回 winner index，不会猜测 loser 应继续、取消还是复用。如果 loser 正在等待
外部操作，应先通过该操作的 `CancelToken`/owner 撤销或完成它，再 `request_cancel()` 并 `join()`；只写
task flag 后立刻销毁 operation state 是悬挂回调风险。

## 阻塞工作不是 Fiber 工作

OS DNS、遗留 blocking I/O、同步模型加载或不可合作切片的长 CPU 函数应进入 `BlockingPool`，否则一个
调用会冻结整个 Scheduler worker：

```cpp
#include "hyle/sched/blocking.h"

hyle::sched::BlockingPool blocking(alloc, /*max_threads=*/32);
auto loaded = hyle::sched::spawn_blocking(
    blocking, alloc, [path]() noexcept { return load_model(path); });
auto model = loaded.join();
blocking.shutdown();
```

closure 在普通 OS thread 上执行，不能调用 `Fiber::yield`、sched sleep、current waker 或依赖 scheduler
worker TLS。pool shutdown 让已开始 job 完成，尚未开始的 job 以 `Cancelled` 结束；不能从该 pool 的
job 内 shutdown 自己。

## 取消与 shutdown

`CancelToken::root(alloc)` 建立可复制根句柄，`derive()` 建立子树；父取消向下传播，子取消不反向污染
父或 sibling。运营环境需要恢复 OOM 时使用 `try_create_root` / `try_derive`。`CancelScope` 把一个
token 叠加到当前 sched execution context 的活动取消源：Fiber 上随 Fiber 迁移，`schedule_task` 中只在
该 worker-stack callback 期间有效；嵌套 scope 与 `JoinHandle::request_cancel()` 组合，不互相覆盖。
主动取消已提交的 I/O 可用无分配 RAII `CancelCallback`，其 token copy 必须比 callback 长寿。典型顺序
是 caller 取消 token，operation owner 的 callback 请求 OS/backend 取消，正常 completion 路线裁决唯一
winner 并消费 Waker，最后 task 才从等待点返回；不要让 JoinHandle 越过 operation owner 强制恢复 Fiber。

短生命周期批处理直接 `shutdown()`：它等待有限工作归零、关闭 admission 并 join workers。长期服务先：

1. `begin_shutdown()` 关闭新 admission 并唤醒 `wait_for_shutdown()`；
2. handler 在协作边界退出，或 owner 取消/关闭其外部 I/O；
3. `drain_for()`/`drain_until()` 给出 typed monotonic budget；
4. 最终调用 `shutdown()` 完整 join。

stackful Fiber 无法安全地被外部强杀。超时后不能销毁仍在运行的 Scheduler；应继续驱动协作退出，或由
进程级 supervisor 终止整个进程。

## 同步原语如何选择

`ConditionVariable`、`WaitGroup`、`Latch`、`Event`、`Barrier`、`Semaphore` 和 `RwLock` 在 Scheduler
Fiber 中挂起当前 Fiber，在普通线程中阻塞当前线程。因此同一长等待原语可以跨控制线程与 worker
使用，但其 lifetime 必须覆盖所有 waiter。`Mutex` 是有意不同的短临界区锁：竞争时它会 park 当前 OS
worker，而不是切走 Fiber。绝不能持有它跨 `yield`、sched wait、blocking syscall 或长 kernel；需要等待
条件时用 `ConditionVariable`，让 Fiber 真正离开 run queue。

- 条件等待始终使用 predicate overload 或循环，且只把受保护状态放在对应 mutex 下；
- timed 入口接收 `time::Duration` / `MonotonicTimePoint`，不要手算 raw ns；
- 销毁带 waiter 的原语、递归锁普通 `Mutex`、读锁升级写锁都是程序错误；
- `Semaphore` 采用严格 FIFO，避免大 permit 请求饿死，代价是有意的 head-of-line blocking；acquire
  family 返回 `Result`，`Unavailable` 表示立即入口暂时拿不到，`DeadlineExceeded` 表示 timed wait 超时，
  `Cancelled` 表示 semaphore 已关闭；零 permit 在未关闭时立即成功；
- 同步原语保护短 metadata；不要拿它们包住 blocking syscall 或大段 kernel 计算。

限制并发量时优先用 RAII permit，并保留失败原因：

```cpp
auto permit = gate.try_acquire_guard();
if (!permit) {
  if (permit.error().IsUnavailable()) return apply_backpressure();
  return permit.error();                 // closed -> Cancelled
}
run_with_capacity();                     // scope exit 自动 release
```

## 消息与 ownership

`Channel<T>` 是有界 MPMC backpressure 队列。自然构造适合可信 capacity；用户输入或受控 OOM 用
`Channel(alloc, capacity, &status)` 并检查 `status`/`valid()`。`send`、`try_send`、`send_for` 的失败
臂是 `ChannelSendError<T>`：`reason()` 区分 full/timeout/closed/invalid，`value()` 把尚未入队的原值
交还给 producer。不要在 retry/reroute 前丢掉 error 对象。

```cpp
hyle::sched::Channel<Request> requests(alloc, 256);
auto sent = requests.try_send(std::move(request));
if (!sent) {
  const auto reason = sent.error().reason();
  request = std::move(sent.error()).value();
  reroute_or_reject(std::move(request), reason);
}

auto next = requests.recv();
if (!next) return next.error();
handle(std::move(*next));
```

`try_recv()` 的 `NotFound` 表示暂时为空，`recv_for()` 的 `DeadlineExceeded` 表示超时，`Cancelled` 表示
closed 且已 drain，recoverable 构造失败为 `ResourceExhausted`。复用 caller-owned slot 的 output
overload 是热路 adapter；默认优先 value-returning `Result<T>`。

其他消息形状：

- oneshot：sender 只能发送一次；receiver 提前 drop 时 send error 直接携带原 value；sender 未发送即
  drop 时 recv 返回 `Cancelled`；
- watch：只保留最新 value 和版本，适合配置、模型 generation 或 shutdown state；`borrow()` 返回 copy，
  不把锁跨 suspend；mutation callback 必须 `noexcept`；
- broadcast：bounded ring 不阻塞 sender；慢 receiver 得到 `BroadcastRecvError{Unavailable, missed}`，
  cursor 随后跳到仍保留的最老值；普通关闭是 `Cancelled`。

这些共享 state 都借用创建时的 allocator identity；所有 sender/receiver 必须在 backing 失效前销毁。

## Fiber、线程与常见坑

- Fiber 只在挂起状态迁移，但恢复后可能位于另一个 worker。不要把 thread-local 指针、当前 CPU、arena
  owner 或 lock ownership 跨 `yield/suspend/sleep/wait` 保存；恢复后重新读取，或显式 pinned。
- `schedule`/`schedule_task` callback 运行在 worker stack，不能调用任何会挂起 Fiber 的 API；无结果的
  可挂起工作选 `spawn_detached`，有结果的可挂起工作选 `spawn_task`。
- `bulk_execute` 是同步 batch barrier，不是异步 handle；传入的 input/output/callable/token 在它返回前
  不能销毁。不同 chunk 写同一非原子 slot 是用户数据竞争。同 Scheduler Fiber 可能直接执行 body；
  不要假设 `Fiber::current()==nullptr`，也不要在 body 内利用这一事实偷偷挂起。
- 不要在 worker 上调用 `hyle::time::sleep_*`，那会阻塞 OS thread；使用 `hyle::sched::sleep_*`。
- `Fiber::create` 与 raw `spawn(Fiber*)` 只适合实现 runtime integration 的 owner。`spawn` 不接管对象，
  caller 必须保持 Fiber 存活直到结束；started Fiber 不能 move。
- 默认 stack tier 适合普通工作。只有有实际深栈证据时选 small/large/custom stack；过小返回错误或
  fail-fast，不能靠关闭 guard page 赌性能。
- `arm_timeout/cancel_timeout` 是同 worker completion owner 的底层构件，不是普通 sleep API。
- `Waker` 是 move-only、一次消费的低层 completion handle。每次 suspend episode 都重新取得；`wake()`
  后自身失效。多个完成源必须先在 operation state 中裁决唯一 winner，只有 winner 可以消费 Waker；它不替
  source 决定 ownership、取消或关闭顺序。

## 构建与支持范围

源码用户在父工程中选择 `sched` 并链接公开 target：

```cmake
set(HYLE_COMPONENTS "sched")
add_subdirectory(path/to/hyle hyle-build)
target_link_libraries(my_runtime PRIVATE hyle::sched)
```

安装包用户用 `find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS sched)` 后同样链接
`hyle::sched`。fcontext backend 当前支持 Windows x86-64、Linux x86-64/AArch64 与 macOS
x86-64/AArch64；显式请求不支持的平台会在 configure 时失败。`sched/detail`、fcontext assembly 和
worker-loop integration 不安装，也不是 public API。static/shared 都是 C++ SDK，不提供 C ABI。

长期边界和复杂度预算见 [`sched-design`](../architecture/sched-design.md)。当前性能证据只在与候选
代码、机器和同合同 workload 绑定后有效；不要从旧 ticket 或 benchmark 名称推断全平台 SOTA。
