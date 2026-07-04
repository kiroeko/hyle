# HYLE io/ 网络库选型

> 本文档定下 HYLE `io/` 模块的网络层选型。上游约束见 [`hyle-scope.md`](./hyle-scope.md)（`sched/` 统一调度 + fiber）。

## 两个调查事实

1. **slate 已自带 boost::asio**（`lib/windows_x64/boost/.../asio/`、`linux_x64` 均有完整 asio）——可借，但下面说明为何不借。
2. **proactor 跨平台天生不对称**：Windows IOCP 是真 proactor；Linux io_uring 现在也是真 proactor；**macOS/BSD 只有 kqueue（reactor），无任何 proactor**。这是跨平台高性能网络最硬的现实。

## 决定性约束：HYLE 已自带调度器

HYLE 的 `sched/` 是「线程/进程/纤程/协程统一调度 + fiber」。**任何自带 event loop / executor 的网络库（asio `io_context`、libuv loop、seastar reactor）都会和 HYLE 调度器抢线程管理权——双调度器冲突。**

与 `parallel/` 盘点结论同构：thread_pool 单全局锁架构长不出统一调度器、须基于 catomics 重写；网络层同理——**IO 必须长在 fiber 调度器上，不能 import 第二个调度器**。

## 编程模型：proactor（完成式），非 reactor（就绪式）

| | reactor（epoll/kqueue 原生）| proactor（IOCP/io_uring 原生）|
|---|---|---|
| 内核语义 | "可读了"，应用自己做 IO | 应用提交 IO，内核做完通知结果 |
| 配 fiber | 唤醒后还要自己 read，多一跳 | 提交即让出、完成即唤醒——**天然契合** |
| 零拷贝/批量 | 弱 | io_uring multishot + IOCP 均支持 |

fiber-blocking（`co_await async_read()` → 提交 → 挂起 fiber → 完成唤醒）与 proactor 一一对应。**HYLE 对上暴露 proactor 完成式 API**，reactor 后端内部模拟完成。方向与 C++ 标准网络一致（WG21 P4100R0《Coroutine-Native I/O for C++29》）。

## 后端矩阵（跨平台答案）

| 平台 | 首选后端 | 性质 | 降级 |
|---|---|---|---|
| Linux ≥5.1 | **io_uring** | 真 proactor，最高性能 | epoll（老内核 reactor 模拟）|
| Windows | **IOCP** | 真 proactor，OS 原生 | — |
| macOS / BSD | **kqueue** | 仅 reactor，须模拟完成 | — |

**诚实代价**：真·零开销 proactor 只在 Linux+Windows；mac/BSD 永远付"就绪→完成"模拟税（readiness 上内联做 IO 或丢线程池）。物理现实，asio 也这么扛。

## 借 vs 造：只借到 syscall 薄封装层

| 方案 | 借什么 | 判断 |
|---|---|---|
| asio（已带）| 全套 | ❌ 自带 `io_context` 调度器，与 HYLE fiber 冲突；默认 epoll，io_uring 后端不成熟 |
| libuv | 全套 | ❌ 自带 loop，双调度器；非 io_uring-first |
| seastar | 全框架 | ❌ share-nothing + 自有调度/内存模型，会"吞掉"HYLE；**仅 Linux**（实测无 IOCP/kqueue，后端链 io_uring→aio→epoll）。但其 `reactor_backend` 后端抽象（epoll/aio 也对上呈现 completion 语义）值得照镜 |
| **liburing(仅Linux) + IOCP/kqueue(OS原生)** | 只借 syscall shim | ✅ liburing 只是 io_uring 系统调用封装、**非调度器**；IOCP/kqueue 本就是 OS API。proactor 抽象 + fiber 绑定 HYLE 自造 |

**结论：方案 C。** 后端只借到"系统调用薄封装"这一层（liburing 在 Linux；IOCP/kqueue OS 原生无需库），**统一 proactor 抽象 + fiber 让出/唤醒由 HYLE 自写**，直接绑 `sched/`。既拿到 io_uring/IOCP 满血性能，又不引入第二调度器。

## 模块形状

```
io/
├─ backend/
│  ├─ io_uring.cc   (Linux 首选, 借 liburing)
│  ├─ epoll.cc      (Linux 降级, reactor→completion 模拟)
│  ├─ iocp.cc       (Windows, OS 原生)
│  └─ kqueue.cc     (mac/BSD, reactor→completion 模拟)
├─ proactor.h       (统一完成式抽象, 平台无关; socket+file op)
└─ net/             (socket/tcp/udp/file, co_await 风格, 遇 IO 让出 fiber
        ↓ 绑定       + dns 异步解析(卸载 spawn_blocking)+ signal_set)
     sched/         (fiber 提交即挂起, completion 即唤醒)
```

一句话：**对上一套 proactor 抽象，对下 4 个后端，中间只借 syscall 封装、调度自己管。**

## 已落地(io-01/02 实现纪要)

> 下面 3 项原为"待深挖"(JIT 延到开工),io-01/02 开工时已定并实现,记录当前真相:

1. **proactor.h 接口形状**:`Proactor{init/reap/post}` + `Completion{user_data,result,flags,buffer_id}`。收割式(reap 批量取完成),平台无关。**接口在写之前先研究 liburing v2.15 的 SOTA 机制**(multishot/registered buf/SQPOLL/linked SQE),确保不挡:`flags` 带 `kMore`(multishot 一提交多完成、token 不早回收)+`kBuffer`(零拷贝 provided buffer id);reap 批量=io_uring for_each_cqe(N 完成 1 次 release-store);timeout 走 EXT_ARG 不额外 syscall。submit 侧的具体 op(read/write/accept + link/fixed-file flag)留 io-04 net/。
2. **io_uring↔fiber 映射**:`user_data`(SQE)= Waker token,完成时原样回来(`cqe->user_data`)→ reap 交回 → sched wake。这正是 sched-design §3.3 的物理机制,IOCP 用 completion key 同构。
3. **只借 syscall shim 的力度**:实测比 io-design 预想更省——**连 liburing 都不链**,只用 3 个 io_uring syscall + `linux/io_uring.h` uapi 头 + 自写内联环算术(barrier 精确复刻 liburing v2.15 的 4 个 fast-path 点)。IOCP/kqueue 本就 OS 原生。**零第三方链依赖。**

**后端进度(诚实分层)**:io-01 **收割接缝**在 IOCP(Windows)+ io_uring(Linux)双后端均 ✅ 真测。io-02 **read/write 真 op 两后端全通 ✅**:io_uring 填 SQE(WSL 真测 write→read + bad-fd 负 errno;TSan 净);IOCP 用 `ReadFile/WriteFile` + per-op OVERLAPPED——proactor 侵入式在途链表**自持** OVERLAPPED 生命周期(OVERLAPPED 是 IoOp 首成员;reap cast `lpOverlapped` 回 IoOp;`post()` 保持 nullptr 以区分合成唤醒;dtor 先 CancelIoEx 再排空端口回收防关机 UAF)。**先前"IOCP 真 op 需随 net/ 定形"是过度谨慎——OVERLAPPED 纯 IOCP 内部细节 proactor 自持,net/ 不可见,故此处已落地。** io-03 **epoll 降级 + SOTA-first 后端选择 ✅**:`init()` 总先试 io_uring,仅真不可用(老内核/`io_uring_disabled` sysctl/容器 seccomp)才退 epoll——生产走 io_uring,epoll 保 io_uring 被禁环境仍可跑;epoll reactor→completion 模拟(记 pending op、就绪时内联 IO)。kqueue(mac)待做。**io-04 socket 层 + co_await + shard-per-core + Tcp 封装 ✅**:recv/send/accept/connect 三后端 + `net.h` co_await awaitable + **shard-per-core**(每 worker 一 proactor + fiber 亲和 `spawn_on` 绑核 + `bind_fd` 连接绑 owner 核,submit/reap/resume 单核 share-nothing,满足 IOCP 一 socket 一端口)+ `tcp.h` RAII `TcpListener/TcpStream`。真 loopback TCP echo + 32 连接分片跨核,双平台 + TSan 净。**worker 空闲已阻塞在 GETEVENTS reap**(有在途 op 时把阻塞交给 proactor 一次 `io_uring_enter` GETEVENTS,上限 1ms;无在途则 futex park 保即时跨线程唤醒)——这是正确性要求:io_uring 完成 task-work 只在本线程带 GETEVENTS 进内核才跑,否则 >ring 深度在途 op 的 completion 永不 surface(512 流 pipelined 负载揪出的 P0 hang,已修+双平台+TSan 验证)。**缺:udp;park 合并的纯延迟优化(eventfd 即时唤醒替 1ms 上限);kqueue(mac)。连接级 benchmark 已补(echo + pipelined,见 performance/benchmarks-io.md)。**

## 已落地 + 剩余缺口(2026-07-03 更新)

> proactor 收割接缝 + socket op + co_await 层 + shard-per-core(per-worker ring)+ 阻塞 GETEVENTS reap + TcpListener/Stream + UdpSocket + park 合并 **均已落地**。下面是**剩余缺口**,分两类:①JIT 延后的次要项;②参考库审计(liburing/asio/libevent 对照)揪出的真缺口,已登记为能力驱动的里程碑。

**① JIT 延后(次要):**
- reactor 后端(mac kqueue)如何把"就绪"模拟成"完成"(Linux epoll 降级已做,accept→::accept/connect→SO_ERROR/read/write 内联)。
- **park 合并 ✅ 已落地**(2026-07-03):worker 空闲以完整 timeout(timer deadline/INFINITE)阻塞 GETEVENTS reap,不再 1ms 轮询;跨线程唤醒经 `Proactor::wake()`(io_uring 专用 eventfd 上常驻 `IORING_OP_READ`,`write` 打断 reap,不碰单生产者 SQ;IOCP `PostQueuedCompletionStatus`;epoll level-triggered eventfd)。实测空闲 500ms 的 `io_uring_enter` 从 421→3(~140x),吞吐无变化。

**② 参考库审计真缺口(能力驱动,见 roadmap 轴 C / milestones):**
- **io-06 带超时 io(C2)**:recv/accept/connect 带 deadline + 取消在途内核 op(io_uring `ASYNC_CANCEL`/`LINK_TIMEOUT`、IOCP `CancelIoEx`)。几乎所有网络库的生存能力,慢/死连接不带超时会永久占资源。
- **io-07a multishot accept ✅(kMore 半成品补完)**:`submit_accept_multishot` 三后端(io_uring `IORING_ACCEPT_MULTISHOT` 原生;IOCP AcceptEx 循环 + epoll 重 arm 两路 emulation,对上同一 kMore 流)。一次 arm 持续吐连接,补完 `Completion.kMore` 当初留的接口。双平台 + TSan 净。
- **io-07b provided buffer ring + multishot recv ✅(kBuffer 半成品补完)**:`register_buf_ring`(IOU_PBUF_RING_MMAP,内核选 buffer 免预投递=零拷贝 recv)+ `submit_recv_multishot`(RECV_MULTISHOT + BUFFER_SELECT)+ `buf_ring_addr`/`buf_ring_return` 归还协议。io_uring only(IOCP/epoll 诚实 return false,net 回退预投递 recv)。双验(io_uring + TSan)。**至此 kMore(07a)+ kBuffer(07b)两个当初预留的 flag 全补完**。registered files/buffers + vectored readv/writev 是新增(非半成品),可后补。
- **io-08 TLS(C9)**:加密 socket over co_await stream(借 BoringSSL/OpenSSL shim)。面向公网服务器必需,roadmap 原本完全没有=真漏。
- **io-09 异步 DNS + signal ✅(C10)**:`resolve_addr`(tcp.h)只 `inet_pton` 解析 IPv4 字面量(连 getaddrinfo 都没有)——`dns.h/cc` 加 `resolve()` **经 sched-11 spawn_blocking 卸载 getaddrinfo**(抄 tokio 路线,不自造 DNS 协议),内部值拷贝 sockaddr + freeaddrinfo 不泄 addrinfo*,`JoinHandle<ResolveResult>` 回传;字面量 IP 走 getaddrinfo 同步快路径。**signal_set**(`signal.h/cc`)co_await `wait()`:POSIX self-pipe + sigaction(进程全局单例 + 引用计数;handler 只 async-signal-safe 写 pipe,免 signalfd 全线程 mask),Windows SetConsoleCtrlHandler(handler 独立线程 post 到 ring;CTRL_CLOSE/LOGOFF/SHUTDOWN handler 阻塞防提前杀,tokio 同款救命坑)。**单 waiter 模型**(优雅关闭单 fiber 用例,同 tokio Signal)。三后端真测(Win/io_uring/epoll)+ WSL TSan 净。signal 是 C4 优雅关闭的触发器。
- **io-05 文件/shmem(C5)✅(文件 io)**:`proactor` 加 `submit_open/close/statx/fsync/fallocate` 三后端——io_uring openat/close/fsync/fallocate 真异步 SQE(fallocate 字段映射反直觉:addr=len/len=mode/off=offset,照 liburing 核对),statx 同步 fstat + posted 合成;IOCP 同步执行 + kFileMeta 合成(open 返 64 位 handle 不截断,FILETIME→ns 换算);epoll regular-file 同步降级(epoll_ctl 返 EPERM→pwrite/pread 尊重 offset + posted,因 epoll 不能异步 regular file,tokio 同理用线程池)。`file.h/cc` File RAII + co_await read/write/stat/sync/allocate/close/open_file,shard-per-core 绑核。三后端真测 + WSL TSan 净。**shmem(mmap 跨进程)未做**,留 io-05 后续。

## 参考

- Why you should use io_uring for network I/O — Red Hat Developer
- Zero-Copy I/O and io_uring for High-Performance Async Servers
- From epoll to io_uring's Multishot Receives (r/programming, 2025)
- Coroutine-Native I/O for C++29 — WG21 P4100R0
