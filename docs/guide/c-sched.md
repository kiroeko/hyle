# sched 纯 C ABI 使用指南

`hyle/c/sched.h` 是 HYLE 调度器的稳定语言边界。它只暴露命名自由函数、
具名枚举、POD 配置、函数指针和不透明句柄；C++ fiber、模板、异常、
`std::vector` 和 coroutine frame 都不跨 ABI。

链接 `hyle_sched`；纯 C 消费者只需要：

```c
#include "hyle/c/sched.h"
```

所有返回空句柄或失败码的入口都会更新本线程的
`hyle_sched_last_error()`。成功调用会清除旧错误。返回的字符串是借用的
thread-local 存储，不要释放，也不要跨线程保存。

## 1. 三种执行入口

它们不是同一个成本/能力档，按任务是否需要挂起选择：

| 入口 | 运行位置 | 可以做什么 | 不可以做什么 |
|---|---|---|---|
| `hyle_sched_schedule` | worker 自身栈 | 极短、不可挂起的回调 | yield/suspend/sleep、fiber wait、长阻塞、抛异常 |
| `hyle_sched_spawn` | HYLE 自持的有栈 fiber | yield/sleep、同步原语、Waker、异步 I/O glue | 阻塞 syscall；应卸载到 blocking pool |
| `hyle_sched_spawn_on` | 指定 worker 的有栈 fiber | 与 spawn 相同，另加核亲和 | worker 下标越界 |

`schedule` 是最便宜的普通任务腿，适合计数、短 continuation 和
fork-join 小块；`spawn` 是完整 M:N fiber 腿。两者都只借用 `ctx`，调用方
必须让它和 callback 至少活到回调返回。

```c
#include <stdatomic.h>
#include "hyle/c/sched.h"

static void add_one(void* p) {
  atomic_fetch_add_explicit((atomic_int*)p, 1, memory_order_relaxed);
}

int main(void) {
  hyle_sched_options opts = hyle_sched_options_default();
  opts.num_workers = 0;                 /* 动态进程可用 CPU 数，不是整机总数 */
  opts.worker_cpu_order = HYLE_CPU_ORDER_PERFORMANCE_FIRST; /* 少量 worker 先占强核 */
  opts.has_worker_priority = 1;
  opts.worker_priority = HYLE_THREAD_PRIORITY_HIGH; /* macOS: QoS 软偏好 */
  hyle_sched_scheduler* s = hyle_sched_scheduler_create(&opts);
  if (!s) return 1;

  atomic_int count = 0;
  if (!hyle_sched_schedule(s, add_one, &count)) return 2;
  hyle_sched_drain(s);
  hyle_sched_scheduler_destroy(s);
  return atomic_load(&count) == 1 ? 0 : 3;
}
```

回调不得让异常越过 C 边界。外语绑定应在 trampoline 内捕获其语言异常并转成
本语言的错误通道。

`worker_cpu_order` 只是 worker pin 的顺序偏好，不减少可用 CPU。默认
`HYLE_CPU_ORDER_OS_DEFAULT` 保持系统顺序；`PERFORMANCE_FIRST` 适合延迟/吞吐优先，
`EFFICIENCY_FIRST` 适合后台或功耗敏感工作。平台无法可靠分类时两者都退化为 OS 顺序，
不会用 CPU 编号猜核型。macOS 没有硬 affinity，强核软倾向应结合线程 QoS。
设置 `has_worker_priority=1` 后，`worker_priority` 使用 platform 的五档
`hyle_thread_priority`；默认 `has_worker_priority=0`，不会改变 OS 优先级。Linux 提升优先级
可能需要额外权限，scheduler 的线程配置按 best-effort 处理。

## 2. 生命周期与优雅关闭

`hyle_sched_shutdown`/destroy 的普通批处理语义是：停止外部接纳之前，已接纳
任务可以继续派生后代；等待所有在途任务完成，再停止并 join worker。这避免
父任务在 shutdown 竞态中无法派生完成自身所需的子任务。

服务进程使用两阶段关闭：

```c
hyle_sched_begin_shutdown(s);       /* 拒绝新外部任务，广播关闭请求 */
if (!hyle_sched_drain_for(s, 5ULL * 1000 * 1000 * 1000)) {
  /* 超时升级：触发自己的 CancelToken / 关闭 fd；HYLE 不强杀 fiber。 */
}
hyle_sched_shutdown(s);
```

长寿命 fiber 在让出点调用 `hyle_sched_shutdown_requested()`，观察到请求后自然
退出。`hyle_sched_wait_for_shutdown()` 会挂起当前 fiber，直到请求到达。
有栈 fiber 不可安全抢占，超时不代表 HYLE 会销毁正在运行的栈。

允许多个外部线程同时调用 shutdown；每个调用都是同一个完成屏障。不得从
scheduler 自己的 worker 回调中销毁 scheduler。

## 3. Waker：外部完成源的桥

在 fiber 内 capture，把句柄交给 I/O、GPU 或自己的异步后端，然后 suspend：

```c
typedef struct operation {
  hyle_sched_waker* waker;
} operation;

static void waiter(void* p) {
  operation* op = (operation*)p;
  op->waker = hyle_sched_waker_capture((hyle_allocator){0});
  /* 先把 op 发布给后端，再挂起。wake-before-suspend 也不会丢信号。 */
  hyle_sched_suspend();
}

/* 完成线程：每个 waker 最多成功 wake 一次。 */
static void complete(operation* op) {
  (void)hyle_sched_waker_wake(op->waker);
  hyle_sched_waker_destroy(op->waker);
  op->waker = NULL;
}
```

`hyle_sched_waker_wake` 的 1 表示本调用消费了唤醒权；0 表示无效或已被消费。
销毁一个未 wake 的 Waker 不会恢复 fiber，因此调用方必须同时有取消/关闭方案。

`yield/suspend/sleep/current_worker/wait_for_shutdown` 是 current-fiber API，只能
在 `spawn`/`spawn_on` 的回调里调用；`schedule` 回调没有独立 fiber 栈。

## 4. 同步原语

以下句柄都可由 fiber 或裸线程使用。等待较久的原语会在 fiber 中让出调度器，
在线程中 park OS 线程；`Mutex` 是短临界区平台快锁，持锁期间不得 yield、
sleep 或做阻塞调用。

| 设施 | 核心入口 | 约定 |
|---|---|---|
| Mutex | lock/try_lock/unlock | 短临界区 |
| ConditionVariable | wait/wait_for/notify | wait 原子解锁并重新上锁；timed 返回 1=通知、0=超时 |
| WaitGroup | add/done/wait | C 输入错误可恢复，add/done 返回 0 并设 last-error |
| Latch | count_down/try_wait/wait | 一次性倒计时；下溢返回失败 |
| Semaphore | acquire/try_acquire/release/close | close 唤醒 waiter；溢出 release 返回失败 |
| RwLock | read/write lock/try/unlock | writer 通过 FIFO permit 避免饥饿 |

destroy 前必须确保没有线程/fiber 正在使用或等待对应句柄。外语的 RAII/`using`
包装应在同一个动态库分配域内调用配对 destroy，不要直接释放 opaque pointer。

## 5. 消息设施

所有 C 消息 API 只复制 `void*` 的数值，不拥有、解引用或释放 payload。
payload 的生命周期由调用方负责。

### 有界 MPMC Channel

`send/recv` 等待到成功或关闭；`try_*` 和 `*_for` 返回具名结果：

- `HYLE_SCHED_SEND_SENT/FULL/CLOSED`
- `HYLE_SCHED_RECV_VALUE/EMPTY/CLOSED`

timed send/recv 的超时也映射为 `FULL`/`EMPTY`。`close` 后已有元素仍可被接收，
队列耗尽后才返回 CLOSED。

### Oneshot

`hyle_sched_oneshot_create` 一次得到 sender/receiver。发送和接收能力各消费一次；
重复使用返回失败，不会越界访问已移动的 C++ 句柄。receiver drop 可被 sender
通过 `sender_is_closed`/`sender_wait_closed` 观察。

### Watch

watch 只保留最新值。sender/receiver 都能 clone；`changed` 等待版本变化，
`borrow_and_update` 同时把 receiver 游标推进到当前版本。borrow 返回的只是
payload 指针值，不是 HYLE 内部存储地址。

### Broadcast

sender 写入有界环，每个 receiver 有独立游标。慢 receiver 返回
`HYLE_SCHED_RECV_LAGGED` 并快进到仍保留的最旧消息；`try_recv` 不等待。
sender/receiver 可 clone 或 subscribe。所有 sender 释放后，消费者读完存量
消息得到 CLOSED。

## 6. 结构化取消

```c
hyle_sched_cancel_token* root =
    hyle_sched_cancel_root((hyle_allocator){0});
hyle_sched_cancel_token* child =
    hyle_sched_cancel_derive(root, (hyle_allocator){0});

/* 同一 owner worker 上管理树拓扑和 callback。 */
hyle_sched_cancel_callback* cb =
    hyle_sched_cancel_callback_create(child, on_cancel, ctx,
                                      (hyle_allocator){0});
hyle_sched_cancel(root);               /* 传播到 child，回调同步执行 */
hyle_sched_cancel_callback_destroy(cb);
hyle_sched_cancel_token_destroy(child);
hyle_sched_cancel_token_destroy(root);
```

derive/cancel/callback 注册与销毁必须串行地发生在同一个 owner worker；
`hyle_sched_is_cancelled` 是原子查询，可跨线程调用。callback 不得抛异常，且在
callback 句柄销毁前 token 节点必须仍然存在。

C++ 的 `CancelScope` 是把 token 注入当前 Fiber 的 RAII 类型，纯 C 无法安全复制
这个词法作用域语义，因此 C 面保留显式 token/callback，任务体在自己的取消点查
`hyle_sched_is_cancelled`。

## 7. 阻塞卸载池

```c
static void blocking_work(void* ctx) { /* getaddrinfo / blocking driver */ }
static void blocking_done(void* ctx, int cancelled) {
  /* 恰好调用一次；必须能在任意线程运行。 */
}

hyle_sched_blocking_pool* p =
    hyle_sched_blocking_pool_create((hyle_allocator){0}, 64,
                                    10ULL * 1000 * 1000 * 1000);
hyle_sched_blocking_submit(p, blocking_work, blocking_done, ctx);
hyle_sched_blocking_shutdown(p);
hyle_sched_blocking_pool_destroy(p);
```

shutdown 让正在执行的工作完成，取消尚未开始的队列项，并等待所有线程退出；
并发 shutdown 是完成屏障。不能从同一 pool 正在执行的 `task`/`done` 内调用
shutdown，因为 OS 线程无法 join 自己。`done(ctx, cancelled)` 总是在 pool
任务已开始时由 pool 线程调用；未开始的任务被 submit/shutdown 拒绝时由发起拒绝
的线程调用。它恰好调用一次，必须线程安全；ctx 必须活到 done 返回。

## 8. 分配器与错误

每个 create/clone/capture 接受 `hyle_allocator`：

- 全零表示 `hyle_system_allocator()`；
- 非零时必须同时提供 alloc/free/realloc；
- backing 的 state 和函数指针必须活到所有由它创建的句柄、共享状态和 fiber
  都销毁；
- shared oneshot/watch/broadcast/cancel 状态内部持有一个 allocator bridge，
  最后一个引用释放后才释放 bridge，不借用 create 调用栈上的结构体副本。

工厂失败返回 NULL 或 0，并设置 `hyle_sched_last_error()`。无效外部计数、容量、
worker 下标和 allocator 都走可恢复失败。真正无法继续的内部不变量仍按 HYLE
约定 fail-fast。

## 9. C++ 组合能力在 C 中怎么写

C ABI 的完整性按“能力是否可达”验收，不按“每个 C++ 模板是否有同名 C
函数”验收。无法跨 ABI 的只是模板、RAII 和 coroutine frame 的具体形状；
公开且实用的行为必须能由稳定 C 原语表达：

| C++ 形状 | C/FFI 组合 |
|---|---|
| `Task<void>` | `hyle_sched_spawn` + callback |
| `Task<T>` | 调用方拥有的 result POD/context + oneshot completion |
| `JoinHandle<T>::join()` | oneshot receiver；无结果批任务用 WaitGroup |
| `when_all()` | initial=N 的 WaitGroup，每个 callback 最后 `done` |
| `when_any()` | atomic winner + oneshot；loser 用 CancelToken 协作退出 |
| lambda capture | 普通函数指针 + `void* ctx` |
| RAII lock/permit guard | 显式 lock/acquire 与 unlock/release 配对 |
| `CancelScope` | 显式 token 查询和 callback |
| coroutine suspend/resume | capture Waker，先发布给后端，再 suspend |

### `Task<int>` + join recipe

结果对象和 callback context 由调用方拥有；oneshot 只传递它的地址并提供完成时序：

```c
typedef struct int_task {
  hyle_sched_oneshot_sender* done;
  int result;
} int_task;

static void compute_int(void* opaque) {
  int_task* task = (int_task*)opaque;
  task->result = calculate_int();
  (void)hyle_sched_oneshot_send(task->done, &task->result);
}

int_task task = {0};
hyle_sched_oneshot_receiver* receiver = NULL;
if (!hyle_sched_oneshot_create((hyle_allocator){0},
                               &task.done, &receiver)) {
  /* hyle_sched_last_error() */
}
if (!hyle_sched_spawn(scheduler, compute_int, &task)) {
  /* admission failure；销毁 sender/receiver */
}

void* result_ptr = NULL;
if (hyle_sched_oneshot_recv(receiver, &result_ptr) ==
    HYLE_SCHED_RECV_VALUE) {
  use_int(*(int*)result_ptr);
}
hyle_sched_oneshot_receiver_destroy(receiver);
hyle_sched_oneshot_sender_destroy(task.done);
```

`task` 必须活到 callback 返回；结果指针也只能在 `task` 生命周期内使用。外语绑定
通常把 `int_task` 放在自己的 pinned/GCHandle/Box 中，并在 completion 后释放。

### `when_all` / `when_any` recipe

`when_all` 不需要 future 数组：创建 initial=N 的 WaitGroup，把同一个句柄放进每个
任务 context，callback 的最后一步调用 `hyle_sched_wait_group_done()`，提交方调用
`wait()`。只有 wait 返回且所有 callback 不再访问 context 后才能 destroy。

`when_any` 使用一个 atomic winner（初值 -1）和一个 oneshot。每个 callback 完成时
先 CAS winner；只有 CAS 成功者写结果并 send，失败者不触碰 sender。若 loser
需要提前停止，winner 再 cancel 一个共享 token；loser 在自己的安全让出点查询
token。最后仍用 WaitGroup 收掉 loser，避免提前释放 context。

如果以后出现一种公开能力无法通过上述稳定原语表达，它就是 C ABI 覆盖缺口，
应新增 POD/opaque/callback 形状，而不能用“C 没有模板”作为不提供能力的理由。

## 10. C ABI 覆盖边界

当前 C 头包含 114 个调度入口，纯 C 测试逐符号引用全部入口，覆盖：

| C++ 能力 | C ABI |
|---|---|
| Scheduler、worker-stack callback、self-owned fiber、亲和派发 | 完整 |
| Waker、yield/suspend/sleep、优雅关闭 | 完整 |
| Mutex/CV/WaitGroup/Latch/Semaphore/RwLock | 完整 |
| Channel/oneshot/watch/broadcast | 以 `void*` 类型擦除完整 |
| CancelToken tree/callback | 完整；CancelScope 改为显式查询 |
| BlockingPool | 完整 callback 形状 |
| `Task<T>`、`JoinHandle<T>`、`when_all/when_any`、C++ RAII guards | 不直接导出 |

最后一行表示“形状不直接导出”，不是“能力不可用”：对应 recipe 见 §9。未来若
组合成本被真实绑定证明过高，可新增稳定 POD result + opaque future 便利层，
而不是把 C++ 模板伪装成 ABI。
