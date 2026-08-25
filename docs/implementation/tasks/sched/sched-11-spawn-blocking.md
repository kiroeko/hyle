# sched-11：spawn_blocking——阻塞卸载线程池

| 字段 | 值 |
|---|---|
| **阶段** | P4 |
| **模块** | sched/ |
| **前置** | 能力 C8 驱动 |
| **状态** | ✅ 完成（2026-07-03；双平台 82/82 sched 测试（新增 7 Blocking）+ WSL TSan 零 race）|

> **范围**：把阻塞 syscall / CPU-bound 闭包扔到**专属卸载线程池**（不在 scheduler worker 上跑，sched-design §2.4）。抄 tokio blocking::pool。
> **cloc**：生产 ~440（`blocking.h` + `blocking.cc`）。

## 为什么是它（阻塞 syscall 会卡死整个 worker）

M:N 调度器的 worker 线程数 == 核数，每个驱动多个 fiber。一个 fiber 做阻塞 syscall（DB 驱动、同步文件 io、getaddrinfo、长 CPU crunch）会**卡住整个 worker**——钉在那个核的每个其他 fiber 全停到 syscall 返回。每个真实 runtime 都发的修法（tokio spawn_blocking / marl blocking call / seastar thread）：把阻塞活移到一个**专门用来阻塞**的独立线程池，worker 保持自由跑本核其他 fiber。这是 sched-design §2.4 "阻塞系统调用会卡住 worker" 现实的显式卸载入口。

## 核心决策：抄 tokio blocking::pool，sysmon 不做

- **懒创建线程池**：LAZILY 生成，上限默认 512（远超核数：阻塞活主要在**等**，吃线程栈不吃核，过订即目的）；空闲线程超 keep-alive（默认 10s）自退，故一次 burst 不留 512 线程永久 park。
- **单共享 MPMC 队列 + 一个 condvar**：submit 唤醒一个空闲线程，无空闲且未达上限则生成新线程，否则 job 在队列等。
- **结果经现有 JoinState/JoinHandle（sched-10）回传**：池线程跑闭包、存结果，JoinState 里的 CV 唤醒 join 方——fiber-aware（若 joiner 是 worker 上的 fiber）/ thread-mode（裸线程）。**零新回传机制**。
- **sysmon 不做**：tokio 也没有——卸载是**显式**的（调用方决定什么会阻塞、主动 spawn_blocking）；自动检测卡住 worker 补线程是 Go runtime 风格，无消费者驱动前不投机（轴 C 纪律）。
- **类型擦除用 raw fn-ptr + ctx 不用 vtable**：对齐 Fiber::EntryFn/FinishFn 模块约定，生产代码零 virtual。job 用侵入 `next` 链 MPMC 队列，无 per-job 容器分配。

## API

- `spawn_blocking(pool, alloc, fn) → JoinHandle<R>`（R = `invoke_result_t<F&>`）：在池上跑 `fn`，返 JoinHandle。`join()` 阻塞（fiber-aware from worker fiber / thread-mode from 裸线程）到池线程完成。池在关闭则句柄解析为 Cancelled。闭包被 **MOVE** 到池上（跑在另一线程）——按值捕获，且其 body 跑在**任何 fiber 之外**（那里无 fiber-local API）。
- `BlockingPool(alloc, max_threads=512, keep_alive_ns=10s)` + `shutdown()`（两阶段：在跑的 finish、在途未启动 cancel→join 得 Cancelled）+ `live_threads()` 诊断。池用 std::mutex/condition_variable 管自己的线程（它们本就该阻塞、不是 fiber），区别于守 worker 的 fiber-aware 原语。

## 关闭两阶段

`shutdown()`（析构也调，幂等）：停止接受活、唤醒全线程、join 它们。**在跑的 job finish**；**在队列未启动的 job 被 drop**——其 JoinHandle 解析为 Cancelled（`cancel_impl` 发 Cancelled 状态）。num_notify 精确配对防虚假唤醒。

## 对抗式自审（挑适用类）

- **① 正确性**：双平台 82/82 sched 测试（新增 7 Blocking），含 **WorkerNotStalledByBlockingCall**——证单 worker 不被阻塞调用拖死（卸载真的让 worker 自由）。
- **③ 规范**：类型擦除 raw fn-ptr + ctx（零 virtual，对齐模块约定）；job/线程从注入 allocator。
- **④ TSan**：WSL TSan 零 race（池线程 MPMC 队列 + JoinState 跨线程回传）。
- **⑥ 过度复杂**：sysmon 明确不做（tokio 无，显式卸载）；结果复用 sched-10 JoinState 零新机制。

## 交付物

- `include/hyle/sched/blocking.h`（BlockingPool + spawn_blocking + detail::BlockingJob/BlockingJobImpl）
- `src/sched/blocking.cc`（Impl：懒创建线程 + MPMC 队列 + condvar + keep-alive 自退 + 两阶段 shutdown）
- `tests/sched/`（新增 7 Blocking，含 WorkerNotStalledByBlockingCall）

## 参考

- sched-design §2.4（阻塞 syscall 卡 worker → 显式 spawn_blocking，sysmon 不做）、全节总结（blocking 池已落地）。tokio blocking::pool（懒创建 / 512 上限 / 10s keep-alive / drop==detach）、marl blockingcall、seastar thread。sched-10（JoinState/JoinHandle 回传，fiber-aware CV 唤醒）。io-09（DNS 经 spawn_blocking 卸载 getaddrinfo 的消费者）。
