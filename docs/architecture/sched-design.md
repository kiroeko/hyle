# Sched：可协作挂起的单进程运行时

Sched 用少量 OS worker 承载大量可协作挂起的工作，将 timer、I/O 与外部完成事件接入同一条
suspend/wake 路线。它拥有执行进展和任务寿命，不拥有模型、tensor、网络协议或推理请求政策。
入口选择与代码示例见 [Sched 指南](../user/sched.md)，跨模块关闭原则见 [运行时总览](runtime-design.md)。

## 执行实体与完成事件

OS worker 是执行引擎，stackful Fiber 是可调度的栈；一个 worker 依次运行多个 Fiber。Fiber 可在
任意调用深度协作挂起，仅在 suspended 状态下迁移，绝不能由两条线程同时 resume 同一栈。
C++ Task 为需要结果的可挂起工作提供 typed 组合，不要求每个短 callback 都分配 coroutine。
进程是隔离边界，不能把它伪装成共享 Fiber 指针的另一个 worker。

统一的是完成事件：

```text
timer / IO / 外部设备完成 / channel
                 |
                 v
        operation owner 裁决唯一结果
                 |
                 v
       Waker -> ready queue -> Fiber
```

Waker 是 move-only、一次消费的唤醒权。suspend 与 wake 的 rendezvous 必须覆盖 wake-before-suspend：
完成先到或 worker 先离开 Fiber，都只能由一方把它重新置为 ready。多个完成来源要先在 operation
state 中裁决 winner，不能复制 Waker 让 timeout 与 I/O 分别唤醒同一栈。

## 为什么保留不同任务形状

| 入口 | 执行位置 | 必要成本与适用边界 |
|---|---|---|
| `schedule` | worker 自身栈 | 短、无挂起、无结果；raw fn+context 不分配，owning callable 需要其记录 |
| `schedule_task` | worker 自身栈 | 短、无挂起、有 typed result；callable/result 合并存储 |
| `bulk_execute` | worker runner；同 Scheduler Fiber 可参与 | 同步有限 batch，借用 input/output/callable，返回前全部参与者静默 |
| `spawn_detached` | 独立 guarded Fiber 栈 | 需要挂起但调用方不 join，runtime 负责回收 |
| `spawn_task` | 独立 Fiber 与 Task | 挂起、typed result、JoinHandle |
| `spawn_task_on` | 指定 worker 的 Fiber 与 Task | 与普通 task 相同，但保留 owner affinity，不可被偷 |
| `TaskScope` | 一个 Scheduler 上的 child 集 | 运行期接纳、close/cancel、首错与完整 drain |
| `spawn_blocking` | BlockingPool 的 OS thread | 不可协作的阻塞 syscall、外部同步函数或长 CPU 工作 |

不能在 worker-stack callback 中调用挂起接口。可挂起的工作也不能只因为想少分配就改成
`schedule_task`。raw `spawn(Fiber*)` 不接管 Fiber，适用于已拥有其寿命的 runtime integration；
普通消费者优先使用 owning task 入口。callback 跨越上下文切换 trampoline 时必须遵守 noexcept
合同，异常不能从 fcontext 汇编边界穿过。

`bulk_execute` 只分配 caller 已决定 grain 的半开区间，不提供 Graph 算法、adaptive partitioner
或 tensor tiling。它用 O(1) batch state 和 O(workers) runner 摊销 admission，不为每一项建立 handle。
同 Scheduler 的 root Fiber 可先投递 sibling，再直接做本地份额；外部线程或其它 Scheduler 的 Fiber
不参与目标执行。non-suspending worker callback 中的嵌套 bulk 串行执行，以避免单 worker 同时
等待自身提供进展。无论 body 在哪种参与者上运行，都不得挂起、阻塞或抛异常。

同步返回还需要真正的寿命交接：waiter 看见 Event signaled，不足以证明 signaler 已退出 signal
内部。最后通知者在不再访问 caller-stack state 后发布 terminal handoff，waiter acquire 后才能
销毁该 state。业务完成与通知实现退休是两个条件。

## TaskScope 为何保留每 child 完成记录

TaskScope 是结构化 task owner，不是 WaitGroup 的别名。admission 与 close 线性化；join 先停止
接纳，再等待所有已接纳 child。首错和取消只决定结果，不能提前释放其余任务。scope 的单一
终结 owner 调用一次 join/cancel-and-join，其它线程只请求取消。

只需统一拥有 void/Status child 时使用 `add`，需要 typed 或逐 child 结果才用 `spawn`。
两条路线复用普通 Task/Fiber 机制，不为 scope 增加 wrapper coroutine 或第二张 ready queue。
准备失败在 admission 前直接返回，不再计入 scope；这也不免除调用方 drain 更早 child 的责任。

成功短 child 若都争写一个 scope-wide completion counter，会把任务本身的局部工作变成共享
cache-line 瓶颈。当前实现让每个 child 保留与 driver 同次分配的完成记录；Task、Fiber 与 stack
释放后才发布完成，终结 owner 逐项观察并释放记录。publisher 自身也保留引用，waiter 立即销毁
scope 不会释放仍在 signal 中的记录。

这个选择用 O(N) 小记录保留换取局部成功发布：admission/completion 与 close 为 O(1)，完整 drain
为 O(N)，`remaining()` 也是 O(N) 控制面快照。它不向普通 task 增加追踪字段。曾接纳过任务的
scope 必须显式终结；析构不能静默 detach，也不能靠在所属 child 里 join 自己来完成寿命闭环。

## Worker、队列与优先级

worker 使用本地 FIFO、短期 run-next LIFO、明确的 pinned queue 与 timer；外部 producer 走分片
injection，idle worker 做有界批量 steal。run-next 保持刚唤醒 Fiber 的局部性，但有连续公平预算；
周期检查外部队列避免持续本地产生工作饿死外部请求。ready overflow 不能丢工作。

默认不绑核，让 OS 与宿主保留部署权。需要 per-core I/O、NUMA 或专用 worker 时显式配置 placement
和 best-effort/required policy。pinned Fiber 每次 yield、timer/I/O 或跨线程 wake 都回到 owner，
因此不能被偷；普通可迁移 Fiber 跨挂起点不能缓存 OS TLS 或当前 CPU 事实。

优先级是非抢占的 ready 选择提示。默认 normal-only 不分配 high/low 队列；显式 three-lane 才承担
sidecar 和固定三档选择成本。优先级随 execution identity 保留，同一 Scheduler child 默认继承，
外部或跨 Scheduler 提交默认 normal。持续 high 可使 low 饥饿，deadline、tenant 公平、请求接纳
及 prefill/decode 政策仍由引擎决定；硬隔离需要明确的 Scheduler/OS 资源。

balanced、low-latency 与 power-saving idle policy 在有限搜索与 park 之间取舍。防惊群需要有限
searcher 与唤醒接力，不能把某个固定自旋次数宣称为所有机器最优。

## 取消、时间与消息

公开 timeout 使用 Time 的 Duration/MonotonicTimePoint，不使用墙钟。每 worker 拥有 TimerQueue，
timer 只触发 Waker，不在容器内部执行任意用户 callback。

取消是协作请求：CancelToken 可派生子树，poll 是 O(1)；CancelScope 叠加到 execution context，
随 Fiber 迁移而不是依赖 OS TLS。JoinHandle 的 cancel flag 不会强制恢复仍挂在 I/O/timer/waiter
节点上的 Fiber。operation owner 必须先请求底层取消，正常完成路线裁决 winner、消费 Waker，
任务才安全退栈。取消等待不自动等于操作终止，更不等于资源已退休。

长等待原语在 Fiber 上挂起、在线程上 park。Mutex 是有意的短临界区例外：竞争会 park OS worker，
不能跨挂起、阻塞 syscall 或长工作持有。Semaphore 使用严格 FIFO，避免大 permit 请求饥饿，
代价是 head-of-line blocking；timeout waiter 通过侵入式节点 O(1) 摘除。

消息按语义分开：Channel 保留每条消息并提供有界背压，oneshot 是一次 reply，watch 只保留最新值，
broadcast 允许覆盖并向慢 reader 返回精确 missed count。失败发送归还尚未入队的原值；timeout、
closed、temporarily-full 不能压成一个 bool。锁内 copy 的异常不推进版本/cursor，mutation callback
要求 noexcept，避免半修改后无法说明是否发布。

## 依赖、栈与关闭

Boost.Context fcontext 只提供私有 ABI 切换汇编，不引入 Boost.Fiber scheduler；Fiber 语义归 Sched。
guarded stack 与常用 tier 池归 Mem。首次 resume 后 trampoline 已持有 Fiber 地址，不能再 move。
runtime metadata 使用显式 allocator，allocator backing 必须活过所有任务及完成记录。

Sched 不依赖 IO；IO 通过私有 integration 将 poll/wake 接入 worker loop。用户只链接 `hyle::sched`，
不用理解内部 Container/Time/Mem factor。当前 fcontext 交付范围与构建限制见用户指南，不从系统
名字自动推断任意 ABI 都可运行。

关闭先停止 admission 并通知 shutdown，再让外部 owner 取消/完成其操作，继续提供进展，最终
drain 并 join worker。stackful 工作无法安全抢占；预算到达不能销毁仍运行的栈。BlockingPool
也必须让已开始的 job 完成，而不是假装 deadline 能中断任意外部函数。

维护时优先检查分配、queue entry、唤醒、争用与 syscall 数，以及 wake-before-suspend 和通知退休。
不能用 stackless 与 stackful 的不同任务合同或不同计费边界，推导 API 或布局的性能优劣。
