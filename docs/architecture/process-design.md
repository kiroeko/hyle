# Process：受管子进程与最终回收

Process 为需要本地故障隔离的消费者提供受管子进程生命周期。它组合 OS spawn、Sched 等待、
IO pipe、Mem allocator 与 Error，避免消费者各自维护退出线程和僵尸进程清理器。接口和示例见
[Process 指南](../user/process.md)，公开合同见 [process.h](../../include/hyle/process/process.h)。

## 两个 owner 各管什么

ProcessRuntime 拥有 admission、退出通知与最终 reap；move-only ChildProcess 拥有单个 child tree、
parent-side stdio pipe 与可观察的退出状态。runtime 是共享 OS 生命周期 owner，不是每 child
建立的一套 task/IO runtime。

启动不经过 shell，executable exact/PATH lookup 显式选择，argv 逐项传递并保留空参数。
环境可以继承后 set/unset，或完全 replacement。shell pipeline、PTY、sandbox、resource-limit、
readiness probe、restart/backoff 和日志轮转是不同政策，不通过给 spawn 增加模糊布尔开关隐式完成。

配置中的字符串/span 借用到同步 spawn 返回，实现先复制后续 OS 调用需要的内容。
`ProcessStdio::file()` 只快照 native value，caller 文件须保持打开到 spawn 返回；child 得到
独立 duplicate/dup2，不接管 caller owner。pipe 则成为 child-owned LocalStream，必须遵守其
owner worker、同方向单 in-flight、背压和取消合同。

## 观察退出不等于终止进程

try-wait 读取 cached 退出状态，wait 可以挂起 HYLE Fiber 或 park 普通线程。deadline/cancel 只
结束这次 wait，child 继续运行，之后仍可等待。不能把用户取消观察误写成“后台进程已停止”。

合作退出是应用协议。POSIX request-termination 向 process group 发送 SIGTERM；Windows 没有
可靠同义操作，明确返回 Unimplemented，应用可用 stdin 或自己的协议。跨平台强制结束使用
force-kill-tree，不能以“graceful”名义隐藏强杀。

丢弃仍活跃的 ChildProcess 会关闭 parent pipe、请求强制 tree kill，并把最终生命周期交给 runtime；
析构本身不等待。release-to-reaper 允许 child 继续运行，但仍由 runtime 观察与 reap，shutdown
仍会终止它；它不是 unmanaged native detach。

shutdown 先拒绝新 spawn，等待正在进入 OS 的启动离开 admission，再关闭 parent pipe、终止活跃
tree、收完退出通知并 reap，最后停止 carrier。借出的 LocalStream 必须已静默，不能一边关闭
runtime 一边让其它 worker 继续读写其 pipe。allocator backing 活过 runtime 及产生的所有 child。

## 后端为何使用一个退出 carrier

| 平台 | 启动与 tree 范围 | 退出观察 |
|---|---|---|
| Linux | posix_spawn[p]，process group | 每 child pidfd，runtime 共享 epoll/eventfd thread |
| macOS | posix_spawn[p]，process group | 每 child EVFILT_PROC，runtime 共享 kqueue thread |
| Windows | suspended CreateProcessW，restricted handle list；先加入 kill-on-close Job 再 resume | registered wait 投递到 runtime 共享 IOCP carrier |

每 runtime 只有一个非轮询 carrier，不为每 child 占用等待线程或 BlockingPool worker。
Windows 必须先纳入 Job 再 resume，避免 child 在控制建立前运行并产生后代的窗口；POSIX 的
tree 范围是 process group，不能把它描述成阻止主动逃离进程组的安全 sandbox。

## 成本与边界

argv 复制与总字节数线性。环境 edit 当前通过原项/编辑项比较，精确上界为
O(inherited entries × edits + edits² + copied bytes)，Windows 另有 O(items log items) environment
sort。它是 spawn 冷路径，不能把常见少量 edits 的观察写成无条件线性，也不应提前增加热路径索引。

单次退出通知与 cached try-wait 为 O(1)，常驻 metadata 随 child 数线性；pipe payload 直接进入
IO，没有第二层 adapter copy。raw posix_spawn/CreateProcessW 缺少 argv/env/stdio/tree/reap/wait
的完整 owner 合同，不能直接作为完整 façade 的同工作量性能对照。

Process 不提供服务 supervisor、远程进程或跨语言 ABI。引擎可用它隔离外部工具或 worker，但模型
请求重试、资源预算、崩溃恢复与业务 readiness 仍由拥有那些任务的上层决定。跨模块关闭顺序见
[运行时总览](runtime-design.md)。
