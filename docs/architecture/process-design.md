# Process 设计

Process 为需要隔离本地故障域的 C++ 消费者提供受管子进程生命周期。它组合 Platform 的 OS 文件机制、
Sched 的 thread/fiber 等待、IO 的异步本地字节流、Mem allocator 与 Error 状态，但不建立第二套 task 或
I/O runtime。

## 产品边界

一个 `ProcessRuntime` 是 admission、退出通知和最终 reap 的 owner；一个 move-only `ChildProcess` 是单个
child tree、parent-side stdio pipe 与操作状态的 owner。完整任务包括：

- 不经过 shell 的 exact/PATH executable 与逐项 argv；
- cwd，以及继承后 set/unset 或完全 replacement 的环境；
- inherit、null、已打开文件、异步 pipe 和 stderr 合并到 stdout；
- O(1) cached `try_wait`、普通线程或 HYLE fiber 上的 wait、deadline/cancel；
- POSIX process group 或 Windows Job Object 范围的强制 tree termination；
- owner abandonment、显式 release-to-reaper 和 runtime shutdown 后的可靠 reap。

Shell 语法/pipeline、PTY、sandbox、resource-limit policy、readiness probe、restart/backoff、日志轮转、远程
进程和服务 supervisor 属于消费层或独立产品，不伪装成缺失的低层参数。Process 不提供 Native FFI。

## 依赖与生命周期

公开配置借用字符串/span 到同步 `spawn()` 返回；实现先复制需要跨 OS 调用的内容。`ProcessStdio::file()`
快照 native value，但对应 native file 必须保持打开直到 `spawn()` 返回；child 得到独立 duplicate/`dup2`，
不接管 caller 的 `FileHandle`。pipe 成功后成为 child-owned `LocalStream`，沿既有 worker affinity、单方向单
in-flight、deadline/cancel 与 backpressure 合同运行。

Runtime 拒绝新 spawn 后，先等待正在进入 OS 的 spawn，再关闭 parent pipes、终止仍活跃的 tree、接收全部
退出通知并 reap，最后停止唯一 carrier。丢弃仍活跃的 `ChildProcess` 会关闭 pipes、请求强制 tree kill，
并把 OS 生命周期交给 runtime；析构本身不等待。`release_to_reaper()` 只允许 child 继续运行，仍不放弃
reap 和 runtime shutdown ownership。

wait 的取消或 deadline 只取消该次等待，不终止 child。合作退出属于应用协议：POSIX 的
`request_termination()` 明确发送 `SIGTERM`；Windows 没有可靠的同义操作并返回 `Unimplemented`。跨平台
强制停止使用 `force_kill_tree()`。

## 后端和成本结构

| 平台 | spawn / tree | 退出 carrier |
|---|---|---|
| Linux | `posix_spawn[p]` + process group | 每 child pidfd，runtime 单一 epoll/eventfd thread |
| macOS | `posix_spawn[p]` + process group | 每 child `EVFILT_PROC`，runtime 单一 kqueue thread |
| Windows | suspended `CreateProcessW` + restricted handle list，先加入 kill-on-close Job 再 resume | registered wait 投递到 runtime 的单一 IOCP carrier |

每个 runtime 只有一个非轮询 carrier thread，而不是每 child 一个等待线程或占用 BlockingPool worker。
spawn 为 O(argv bytes + environment bytes)，Windows replacement/inherited environment 为生成稳定 environment
block 所需的 O(n log n) sort；注册、一次退出通知、cached try-wait 和完成后的重复 wait 均为 O(1)。常驻
状态与 child 数线性，pipe payload 直接走 IO proactor，不增加 adapter copy。

Raw `posix_spawn`/`CreateProcessW` 不具备同一 argv/env/stdio/tree/reap/wait 合同，不能用来对完整 Process
façade 做性能排名。这个控制路径优先以 thread/handle/allocation/syscall 上限和复杂度作 P0 裁决；只有出现
完整同合同 peer 时才做 P1。
