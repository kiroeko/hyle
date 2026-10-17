# 运行时：进展、取消与关闭

Sched、IO和Process组合任务与外部操作的生命周期，而非三个互不相干的runtime。
共同所有权和错误原则见[基础机制](foundation-design.md)；API使用见[用户文档](../user/README.md)。

## 调度与进展只有一个明确owner

Sched拥有任务的admission、ready、suspend/wake与drain。stackful Fiber服务同步调用栈中的挂起，
C++ Task保留typed异步组合；它们共享调度和完成机制，不因入口不同建立第二线程池。
第三方fcontext只提供私有ABI切换原语，guarded stack归Mem，不引入Boost.Fiber调度器。

一个Fiber同一时刻只能在一条线程运行，只有suspended后才可迁移。
Waker携带一次性唤醒权；完成、取消和timeout竞争同一个终态，不能复制handle重复ready。
跨worker执行context随Fiber/任务迁移，不能借当前OS线程TLS假装任务始终不动。

worker拥有本地FIFO、短期run-next和明确pinned工作；外部注入及steal保留不丢ready work的路径。
局部热度与公平性冲突，因此LIFO有连续使用上限、周期检查外部队列；这些是实现政策，不承诺所有机器最优。
阻塞工作走明确的blocking资源，不在IO worker偷偷阻塞，也不因嵌套等同一满池而死锁。

## scope为何需要独立完成记录

普通任务不为“将来可能被统一join”支付tracking。动态TaskScope只跟踪显式加入的child，
close与spawn线性化，join先停止接纳再等待全部child；首错或取消不允许提前释放其余child。
需要typed/per-child结果才返回相应handle，仅需scope owner的child无需额外wrapper coroutine。

scope共享completion counter会使短child争写同一cache line。独立child完成记录把这种争用换成
terminal owner观察前的O(N)小记录保留：admission/completion局部，drain遍历记录。
完成发布者也持内部引用，waiter看到完成后立刻销毁scope，不能释放仍在signal中的记录。
这是具体的空间/争用取舍，不是未来所有barrier都应采用相同形状。

live child下不能静默detach或假装异步析构已经drain。显式join/close与析构misuse边界比
析构中无限等待更可解释，尤其当析构发生在同一个调度worker上。

## 优先级与取消不是业务政策

execution priority是非抢占的ready选择提示，yield/wake及child继承保持同一执行身份。
默认normal-only不分配额外lane，opt-in多档才承担相关队列和选择成本。
持续高档可使低档饥饿；deadline、tenant share、prefill/decode与KV preemption由应用决定。
硬隔离需要明确的Scheduler/OS资源，而非把priority名称当保证。

取消是协作请求。等待取消、操作停止和资源退休是三个时刻：

```text
cancel请求 → 停止后续接纳/竞争结果 → 已接纳工作结束 → 最后引用归零 → 可回收
```

错误可以先返回，但仍在使用的payload、控制数据和Waker必须留到真实终态。
释放可见结果不等于底层工作已停；重复close、部分接纳和晚到完成不能制造双重唤醒或悬空owner。

## IO保留完成式与就绪式差异

IO拥有文件/网络/本地流操作及其取消、deadline、背压与完成，不拥有listener业务、路由、认证或服务调度。
IOCP/io_uring直接返回completion，epoll/kqueue先表示readiness；不能为统一外观给前者增加无用pending查表，
也不能把后者的一次ready当完成。

kernel可能在提交后继续读取参数结构、iovec和payload。取消后返回用户层前必须遵守该操作的实际退休合同。
普通文件的blocking fallback无法逐操作抢占时，deadline仅停止后继并等待已开始读写结束，不能承诺中断syscall。
每个方向的在途数量、worker affinity和borrow范围由实际流owner约束。

提交/收割尽量批量化并复用有限operation state，容量满产生明确背压而不是无界队列。
可选的io_uring multishot/provided-buffer通过真实extension取得，不在其它平台暴露恒失败的假portable入口。

## 协议与制品组合的关键边界

plain与TLS复用HTTP framing；TLS provider类型不进入公共头。
HTTP reader/writer以增量body为常态，消费上一块完成后才拉下一块，避免隐藏整body缓存。
file-body可以使用OS fast path，TLS或不适用路径走有界fallback，不声称同样zero-copy。

HTTP Upgrade移动transport时必须同时移交parser预读bytes：head和首帧可能在同一个网络包中，
丢prefix会破坏新协议。WebSocket在这个ownership seam上处理message/frame/control/close，
同方向并发约束、UTF-8与message limits由连接状态机承担，reconnect/auth/schema留给应用。

watch overflow、rename丢失或粗粒度事件明确要求rescan，不伪造无损日志。
recursive是显式模式，Linux按目录维护watch有真实容量成本；普通direct watch不建立全树metadata。
可靠制品传输保持有界buffer、source validator和resume边界，校验与原子发布成功前不覆盖旧目标。
模型解析、archive、token与仓库政策不进入IO。

## 子进程的owner与reap

Process组合OS spawn、child tree、IO pipe与退出通知，不建立每child一条阻塞等待线程。
逐项argv不默认通过shell；stdio传递/duplicate与caller file所有权分开。

ProcessRuntime拥有admission、退出通知与最终reap；ChildProcess拥有单child/tree及parent pipes。
丢弃活child或release-to-reaper仍须把最终OS生命周期交给明确runtime，不能留下zombie。
wait取消只取消等待，不自动杀进程；合作退出归应用协议，强制tree termination是不同能力。

Linux pidfd、macOS kqueue进程事件与Windows registered wait/IOCP有不同实现，但都避免轮询或占用每child一个worker。
spawn冷路径的argv/env复制与Windows environment sort有实际成本，不能拿raw syscall与完整owner直接排名。

## 关闭的顺序

先拒绝新任务/操作，再取消或完成受影响工作，继续提供所需进展，最后drain并销毁backing。
不能先关闭进展线程再等待依赖它的完成，也不能让plan关闭顺带销毁应用借出的全局Context。
异常/首错只决定返回结果，不免除其它已接纳工作的退休和释放责任。
