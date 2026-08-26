# sched 模块强收口（2026-07-20）

## 结论

sched 的机制面、下游接线、对抗式并发修复、纯 C ABI 和用户入口已经形成一套
完整产品面。它不是只有 fcontext + queue 的内部骨架：

- 不可挂起的 worker-stack `schedule` 与完整 stackful
  `spawn_detached`/`spawn_task` 两条任务腿；
- Waker、timer、yield/suspend、亲和、work stealing、shutdown lifecycle；
- Mutex/CV/WaitGroup/Latch/Semaphore/RwLock；
- channel/oneshot/watch/broadcast、Join/when_all/when_any、取消树；
- blocking pool；
- 114 个纯 C ABI 入口和逐符号纯 C 测试。

冻结判定仍要以本页末尾三平台证书为准；新平台或真实性能回归可重新打开，而不是
继续按猜测无限扩接口。

## 1. 功能覆盖与下游消费

| 能力 | 状态 | 已有消费者/证据 |
|---|---|---|
| fcontext + guarded stack pool | 完整 | 所有 stackful task |
| M:N work stealing + affinity | 完整 | io shard-per-core、graph parallel |
| worker-stack fast callback | 本轮补齐 | 轻 continuation/fork-join |
| Waker completion spine | 完整 | io、GPU poller、timer、graph |
| fiber/thread 双模 timed wait | 完整 | sync、io timeout |
| structured Join/cancel | 完整 | graph、HTTP/io cancel |
| graceful shutdown | 本轮加固 | io signal/服务循环 |
| explicit blocking offload | 本轮加固 | DNS、ggml/compute |
| practical messaging | 本轮补 timed/try | C++ 与 C channel/broadcast |
| pure C/FFI | 本轮完成 | `hyle/c/sched.h` |

没有把 `std::execution`、actor/mailbox、distributed scheduler、抢占式 kill 或
自动 sysmon 塞入模块：前两者属于 graph/上层执行策略，跨进程属于 io 联邦；
有栈 fiber 不可安全强杀；显式 `spawn_blocking` 是当前选定合同。

## 2. 红队发现与修复

本轮不是只加导出。重点从“低概率竞态是否会让服务永久卡住”反推实现：

| 问题 | 后果 | 修复/回归 |
|---|---|---|
| spawn admission 与 shutdown 分离检查 | 已关闭 runtime 仍可漏进任务，或计数不归零 | Fiber 用 closed+count 合并 CAS；InlineJob 用 open→count++→recheck，drain 同时观察两者 |
| ordinary shutdown 过早拒绝后代 | 父任务等自己无法 spawn 的 child，死锁 | 外部关闭与 descendant admission 分语义 |
| `wait_for_shutdown` 轮询 | 额外 wake/延迟 | 真 waiter 链 |
| ready bounded queue 满时 worker 自旋 | worker 等自己清队列，永久死锁 | allocation-free intrusive overflow FIFO；pinned/unpinned 均覆盖 |
| wake flood/reinject 同样自旋 | CV notify_all / 外部 completion 可卡死 | 所有 ready 入口共用 bounded-scan + overflow |
| concurrent Scheduler shutdown | double-join / Thread handle race | 单 join owner + completion CV barrier |
| concurrent BlockingPool shutdown | 第二调用者提前返回，可能与析构竞态 | shutdown-complete barrier |
| BlockingPool idle-retired handle 不回收 | burst/idle 周期句柄线性泄漏 | submit 冷路径逐个 reap |
| BlockingPool 在自己的 job 内 shutdown | self-join 死锁/平台异常 | TLS contract check |
| timer callback 先 arm 后发布字段 | 立即到期读半初始化 slot | 数据先发布，再 arm |
| io timeout orphan drain 把一次空 reap 当终止 | 异步 cancel CQE 稍晚到达时 `TimedSlot` 永久漏 1 个 kernel ref | 只以 `in_flight()==0` 或 1024 次硬上限终止；Win/Linux 连续 10 次泄漏计数 + 三平台 io 套件 |
| timed fiber mode flag 在 enqueue 后重复写 | notifier 锁内读取与 waiter 写回构成 data race | mode 在发布前一次性写定；WSL TSan 抓获并修复 |
| Task/Join 的异常、OOM、optional result | terminate/UAF/错误值生命周期 | Status 回传、延迟析构、无效 allocator 可恢复 |
| oneshot sender 生命周期未计数 | receiver 不能可靠观察 closed | `tx_alive` |
| WatchSender copy 计数缺失 | sender_count/closed 错误 | 显式 copy retain |
| CancelScope 接无效 token | 暂时清掉父 scope，漏取消 | invalid scope 变真正 no-op |
| C allocator 指向 create 栈上副本 | shared state 析构时 UAF | refcounted allocator bridge |
| bridge realloc(NULL/0) 引用数错误 | bridge 泄漏/过早释放 | 按 allocation lease 修正 |
| FFI add/done/release 下溢/溢出 | 恶意输入触发 abort | `try_*` + last-error |
| worker-stack callback 每任务分配 owning record | M4 上 malloc/free 掩盖轻任务腿 | `{fn,ctx}` 两字按值入分片队列，零任务分配 |
| InlineJob 全局完成计数 | M4 上 producer/4 consumers 抢一条 cache line，短 callback 约 0.90× marl | per-worker 独占 cache-line 计数；job 带内部 shard tag；M4 回到统计平局 |
| 每个 inline callback 后完整扫 fiber 队列 | 短任务重复扫 pinned/local/global/steal | 最多 8 个有界 batch，之后强制服务 fiber/timer/io |
| ShardedQueue 热路径运行时取模 | ARM 每任务多次整数除法 | 2 的幂 shard 用 mask；pop 自增回绕；非 2 的幂回归保留 |
| POSIX Mutex 竞争解锁写两次状态 | Linux 短临界交接落后 std::mutex | 单次 `exchange(0)` 发布并按旧值决定 wake |
| marl benchmark 在主线程 `bind()` | 配置 4 worker 时 caller 也可执行，形成 5 对 4 | 直接 `Scheduler::enqueue`，caller 走 thread wait，严格 4 对 4 |

维护不变量和 ASCII 架构图见
[`sched-design.md`](../../../architecture/sched-design.md#2026-07-收口后的维护架构)。

## 3. 测试与性能

### 正确性面

Windows 当前：

- `hyle_sched_test`：137 项（含新 overflow wake flood、并发 shutdown、OOM/invalid
  allocator、timed channel、broadcast try-recv）；
- `hyle_sched_c_test`：纯 C，机械引用 `hyle/c/sched.h` 的 114/114 个函数；
- io downstream：基础 9 项 + cancel 2 项 + timeout 5 项，共 16/16。

WSL GCC 14.2 原生 `/root/hyle` ext4：

- Release：sched 137/137、纯 C ABI、io 基础 16 项 + cancel 2 项 +
  timeout 5 项，共 23/23；
- TSan RelWithDebInfo：sched 137/137 连续 3 遍，纯 C ABI 通过，零报告；
- TSan 在首轮实际抓到 `fiber_timed` 发布后重复写与 notifier 读取的 race，修复后
  才签字，因此不是只看功能断言的假绿。

Apple M4 / AppleClang 21：

- Release：sched 137/137、纯 C ABI、io 基础 4 项 + cancel 2 项 +
  timeout 5 项，共 11/11；
- 本轮把本地 marl 源码同步到远端，完成相同 4-worker、15 进程性能对拍；
- macOS 无可用 TSan 证书，竞态闸门由 WSL GCC TSan 承担。

特殊测试形状覆盖：

- queue capacity=2 的 pinned/unpinned 同 worker wake flood；
- shutdown 与多 producer/spawn descendant 竞态；
- notify-before-suspend、notify-vs-timeout、timer immediate expiry；
- blocking burst-retire-reburst、queued cancellation、并发 shutdown；
- allocator OOM/invalid、C handle/shared-state 最后引用泄漏计数；
- TSan 下 fcontext logical-fiber 切换注解。

### SOTA 性能

公平主标尺是 marl：它和 HYLE 都是 C++ 有栈/任务调度器。Highway thread pool 是
预知任务集合的 bulk parallel-for，合同不同，见 §4。

最终 4-worker 同合同结果，`bench-stat.py -n15`：

| 项 | Windows：HYLE / marl | WSL：HYLE / marl | M4：HYLE / marl | 结论 |
|---|---:|---:|---:|---|
| WaitGroup fork-join + worker-stack | 192.510 / 297.790 ns | 220.770 / 337.420 ns | 253.980 / 252.750 ns | 配对中位 Win **1.553×**、Lin **1.509×**；M4 **1.003×，统计平局** |
| 2000 noop + worker-stack | 192.600 / 297.530 ns | 218.440 / 327.030 ns | 252.120 / 252.720 ns | 配对中位 Win **1.534×**、Lin **1.480×**；M4 **1.007×，统计平局**（12/15 HYLE 更快） |
| contended Mutex | 18.687 / 24.372 ns | 28.179 / 28.283 ns | 12.819 / 24.560 ns | Win/M4 赢，Linux 统计平局 |

M4 原先约 10% 的同合同真落后已根治：不是异构核心/睿频，也不只是噪声，而是
InlineJob 全局完成计数的 cache-line ping-pong；运行时 shard 取模是次因。计数按
worker 分片并独占 cache line、有界 batch 与除法消除后，15 遍配对中位回到
1.007×，按测量纪律判统计平局。两次更差的替代实现（通用对象池、短锁 ring）
仍撤回。Windows/Linux 的明确领先来自无 owning record、分片队列与轻量完成
核算；不是弱基线。

完整有栈 `spawn_detached` 的绝对成本约为 Win 342 ns / Lin 355 ns / M4 396 ns；
`spawn_task` 约为 415 / 414 / 434 ns。它们多承担 guarded stack、yield/sleep/
fiber wait/coroutine Join 合同，故不与 marl 的 worker-stack callback 混成胜负。

这轮 ready overflow 在未饱和热路径只加一次 atomic hint 读取；InlineJob
生命周期计数使用 open→per-worker increment→recheck，保持关闭竞态正确性的
同时避开 stackful Fiber 的 CAS 循环和全局完成 cache line。

性能总账同步维护在
[`benchmarks-sched.md`](../../evidence/performance/benchmarks-sched.md)。

## 4. Highway thread pool 与 SIMD 审计

本地 Highway `hwy/contrib/thread_pool` 的合同是：

- 一次 Run 的任务范围已知；
- worker task 不在 pool 内递归 schedule；
- caller 可参与；
- barrier/fork-join 高频重复；
- 有 cache cluster/topology 与 spin/block 策略。

HYLE sched 的合同是动态异步 runtime：任务可派生、fiber 可挂起、外部 Waker 可
随时 reinject，且 timer/io/gpu/cancel/shutdown 共用 completion spine。直接复制
Highway pool 会丢失这些能力，也会制造第二套线程所有权。

吸收结论：

- Highway 的轻 job 思想已落 `schedule`，并在同合同 marl 对拍中获胜；
- caller-participating bulk barrier、cache-cluster partition 应落
  `graph::parallel_for`，由 graph workload 驱动；
- 更丰富 topology 只有在 NUMA/混合核 benchmark 显示当前逻辑核 pinning 失配时
  再补；
- sched 控制面不引 SIMD。指针队列、原子所有权、fcontext、futex/park 无可获益
  的数据并行 lane；SIMD 用在任务的数据 kernel 内。

## 5. C ABI 评审

实现：

- 纯 C 头 `hyle/c/sched.h`（该历史 schema 已在 BUILD-03 删除）；
- 单一 `.cc` 类型擦除层；
- 纯 C 编译/链接测试；
- 用户指南 [`c-sched.md`](../../../architecture/native-ffi-policy.md)。

遵循 close-01：

- 不导出 STL、模板、exception、C++ enum 或对象布局；
- 命名自由函数，不设模块级 API table/version gate；
- opaque handle 的 create/destroy 同分配域配对；
- POD options 以 `struct_size` 前缀兼容；
- callback + `void*` 类型擦除任务和 pointer payload；
- shared state 使用持久 allocator bridge；
- NULL/失败码配 thread-local last-error；
- 外部可控计数和容量软失败。

刻意不导出 `Task<T>`/`JoinHandle<T>`/RAII guards 的 C++ 形状：它们没有稳定 C
布局，但能力不能因此消失。guide 已给出 `Task<T>`→result context+oneshot、
join/when_all→oneshot/WaitGroup、when_any→atomic winner+oneshot+cancel 的可复制
recipe。以后若某项公开能力没有稳定组合通路，就重新判为 C ABI 缺口；若组合
成本被真实绑定证明过高，再设计 POD result + opaque future 便利层。

## 6. 冻结后触发器

以下事件可以重开 sched：

- 新 correctness/ABI/security 缺陷；
- Windows/Linux/mac 之外的平台、fcontext ABI 或 sanitizer 变化；
- 同方法论确认的 marl/Tokio/folly 类同合同回归；
- graph bulk workload 证明需要 caller participation/cache cluster；
- io/compute/graph 真实消费者需要新的 completion/sync 语义；
- close-01 install/export/外语冒烟要求调整句柄。

仅仅出现新的线程池库、亚纳秒波动、猜测 API 或“也许 SIMD 扫指针更快”不构成
重开依据。
