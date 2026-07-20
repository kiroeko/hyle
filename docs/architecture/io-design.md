# HYLE io/ 网络库选型

> 本文档定下 HYLE `io/` 模块的网络层选型。上游约束见 [`hyle-scope.md`](./hyle-scope.md)（`sched/` 统一调度 + fiber）。

## 2026-07-20 当前架构（维护入口）

```text
普通 C++ 用户
  TcpStream / UdpSocket / File / DNS / Signal / TLS / HTTP/1.1 + HTTP/2
       │ co_await；deadline / CancelToken；RAII
       ▼
IoContext（每 Scheduler worker 一个 Proactor + 文件 blocking pool）
       │ submit token                          │ completion token
       ▼                                       ▼
统一 Proactor ───────────────────────────────► sched::Waker
       │
       ├─ Windows IOCP：真完成；OVERLAPPED intrusive list + owner free-list
       ├─ Linux io_uring：真完成；不可用时自动降为 epoll
       ├─ Linux epoll：readiness→completion；HYLE SwissTable<fd, PendingOp>
       └─ macOS kqueue：readiness→completion；HYLE SwissTable<fd, PendingOp>

纯 C / FFI 用户
  hyle/c/io.h：opaque proactor/shmem/HTTP2 codec + POD completion + native handle
       └─ 自己组合 event loop / scheduler；不把 coroutine/STL/OpenSSL 类型过 ABI
```

当前不变量：

- socket/file shard 由一个 owner worker 提交和收割；IOCP handle 绑定后不可换
  completion port。
- 普通 C++ 文件腿依据 backend capability 自动选原生异步或 blocking pool；
  低层 Proactor 保持机制透明，调用者选择它就承担同步 fallback 语义。
- plain TCP 与已握手 TLS 经同一个 `HttpTransport` 进入 HTTP framing 核。
- HTTP/2 协议核固定为 nghttp2；它不拥有 IO/event loop。C++ connection 直接
  驱动 `HttpTransport`，C ABI 暴露 transport-neutral feed/next-output codec。
- TCP read/write/connect/accept 和 UDP recv/send 均有 base、deadline、
  CancelToken 腿；base 腿只用于确定能完成的受控路径。
- Linux-only multishot/provided-buffer/fixed-file 是可观察、可失败的快腿；
  portable one-shot 永远保留。
- epoll/kqueue pending 表使用 container/ 的 flat SwissTable。IOCP/io_uring
  不需要 fd readiness 映射，因此不会为统一外观平白引入 hash lookup。

实现与正确性已完成本轮收口，现有 8 benchmark 的三平台证书也已生成；已确认
的 M4 TLS/kqueue 多 worker 与 handshake 性能尾，以及尚缺的专项 peer 测试见
[`io-module-closure.md`](../milestones/io-module-closure.md) 与
[`benchmarks-io.md`](../performance/benchmarks-io.md)。下文“已落地纪要”保留
施工历史，若与本节冲突，以本节和代码为准。

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
| macOS / BSD | **kqueue** | 仅 reactor，须模拟完成 | ✅ 已做(镜像 epoll,真机全绿) |

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

```text
include/hyle/io/
├─ proactor.h                 统一完成式机制接口
├─ io_context.h + net*.h     sched/Waker、base/timed/cancel/multishot awaitable
├─ tcp.h / udp.h / file.h    普通 C++ RAII facade
├─ dns.h / signal.h          blocking resolver / shutdown source
├─ tls.h / http*.h           加密 transport 与 HTTP/1.1/2 产品层
└─ shared_memory.h           命名跨进程 mapping

src/io/
├─ proactor.cc               四后端按平台条件编译；不跨 TU 暴露 OS 类型
├─ io_context.cc + net impl  sched 集成
├─ file.cc / dns.cc          capability-aware blocking offload
├─ tls.cc / http*.cc         transport + picohttpparser/nghttp2 protocol core
├─ shared_memory.cc          Windows/POSIX mapping
└─ c_io.cc + http2.cc        稳定 C ABI shim + H2 codec facade
```

一句话：**对上一套 proactor 抽象，对下 4 个后端，中间只借 syscall 封装、调度自己管。**

## 已落地（io-01/02 历史实现纪要）

> 下面 3 项原为"待深挖"(JIT 延到开工),io-01/02 开工时已定并实现,记录当前真相:

1. **proactor.h 接口形状**:`Proactor{init/reap/post}` + `Completion{user_data,result,flags,buffer_id}`。收割式(reap 批量取完成),平台无关。**接口在写之前先研究 liburing v2.15 的 SOTA 机制**(multishot/registered buf/SQPOLL/linked SQE),确保不挡:`flags` 带 `kMore`(multishot 一提交多完成、token 不早回收)+`kBuffer`(零拷贝 provided buffer id);reap 批量=io_uring for_each_cqe(N 完成 1 次 release-store);timeout 走 EXT_ARG 不额外 syscall。submit 侧的具体 op(read/write/accept + link/fixed-file flag)留 io-04 net/。
2. **io_uring↔fiber 映射**:`user_data`(SQE)= Waker token,完成时原样回来(`cqe->user_data`)→ reap 交回 → sched wake。这正是 sched-design §3.3 的物理机制,IOCP 用 completion key 同构。
3. **只借 syscall shim 的力度**:实测比 io-design 预想更省——**连 liburing 都不链**,只用 3 个 io_uring syscall + `linux/io_uring.h` uapi 头 + 自写内联环算术(barrier 精确复刻 liburing v2.15 的 4 个 fast-path 点)。IOCP/kqueue 本就 OS 原生。**零第三方链依赖。**

**当时后端进度（2026-07-03 历史快照）**:io-01 **收割接缝**在 IOCP(Windows)+ io_uring(Linux)双后端均 ✅ 真测。io-02 **read/write 真 op 两后端全通 ✅**:io_uring 填 SQE(WSL 真测 write→read + bad-fd 负 errno;TSan 净);IOCP 用 `ReadFile/WriteFile` + per-op OVERLAPPED——proactor 侵入式在途链表**自持** OVERLAPPED 生命周期(OVERLAPPED 是 IoOp 首成员;reap cast `lpOverlapped` 回 IoOp;`post()` 保持 nullptr 以区分合成唤醒;dtor 先 CancelIoEx 再排空端口回收防关机 UAF)。**先前"IOCP 真 op 需随 net/ 定形"是过度谨慎——OVERLAPPED 纯 IOCP 内部细节 proactor 自持,net/ 不可见,故此处已落地。** io-03 **epoll 降级 + SOTA-first 后端选择 ✅**:`init()` 总先试 io_uring,仅真不可用(老内核/`io_uring_disabled` sysctl/容器 seccomp)才退 epoll——生产走 io_uring,epoll 保 io_uring 被禁环境仍可跑;epoll reactor→completion 模拟(记 pending op、就绪时内联 IO;**红队 L3 修 2026-07-05:pending 拆 in/out 双槽,同 fd 全双工 recv+send 并发不再互相覆盖孤儿**)。后续 kqueue、UDP 与 park 合并均已完成，见本页顶部当前架构。

## 已落地 + 剩余缺口(2026-07-05 更新)

> proactor 收割接缝 + socket op + co_await 层 + shard-per-core(per-worker ring)+ 阻塞 GETEVENTS reap + TcpListener/Stream + UdpSocket + park 合并 **均已落地**。下面是**剩余缺口**,分两类:①JIT 延后的次要项;②参考库审计(liburing/asio/libevent 对照)揪出的真缺口,已登记为能力驱动的里程碑。

**① JIT 延后(次要):**
- **reactor 后端(mac kqueue)✅ 已落地(2026-07-05,Apple Silicon 真机)**:`#elif HYLE_OS_MACOS` 块镜像 epoll reactor 把"就绪"合成"完成"——EVFILT_READ/WRITE+EV_ONESHOT 替 EPOLLIN/OUT+ONESHOT、EVFILT_USER+NOTE_TRIGGER 替 wake eventfd(无需 self-pipe)、dual-slot pending_in/out(L3)、spurious-EAGAIN re-arm(L4)、multishot-accept 模拟、submit_cancel 合成 -ECANCELED、常规文件 S_ISREG 检测走同步 pread/pwrite(kqueue 无 EPERM 信号)。macOS 差异:st_mtimespec/fsync(无 fdatasync)/F_PREALLOCATE(无 posix_fallocate)。io_uring 专属(buf_ring/fixed-files/direct-accept)诚实 stub false。真机 io/net/tcp/udp/file/signal/timeout/http 全绿,Linux 零回归。
- **park 合并 ✅ 已落地**(2026-07-03):worker 空闲以完整 timeout(timer deadline/INFINITE)阻塞 GETEVENTS reap,不再 1ms 轮询;跨线程唤醒经 `Proactor::wake()`(io_uring 专用 eventfd 上常驻 `IORING_OP_READ`,`write` 打断 reap,不碰单生产者 SQ;IOCP `PostQueuedCompletionStatus`;epoll level-triggered eventfd)。实测空闲 500ms 的 `io_uring_enter` 从 421→3(~140x),吞吐无变化。

**② 参考库审计真缺口(能力驱动,见 roadmap 轴 C / milestones):**
- **io-06 带超时 io(C2)✅**:TCP read/write/connect/accept 与 UDP recv/send 的 deadline/cancel 腿已完成；取消在途内核 op 用 io_uring `ASYNC_CANCEL`、IOCP `CancelIoEx`、epoll/kqueue 删除 readiness watch 并合成最终 completion。
- **io-07a multishot accept ✅(kMore 半成品补完)**:`submit_accept_multishot` 三后端(io_uring `IORING_ACCEPT_MULTISHOT` 原生;IOCP AcceptEx 循环 + epoll 重 arm 两路 emulation,对上同一 kMore 流)。一次 arm 持续吐连接,补完 `Completion.kMore` 当初留的接口。双平台 + TSan 净。
- **io-07b provided buffer ring + multishot recv ✅(kBuffer 半成品补完)**:`register_buf_ring`(IOU_PBUF_RING_MMAP,内核选 buffer 免预投递=零拷贝 recv)+ `submit_recv_multishot`(RECV_MULTISHOT + BUFFER_SELECT)+ `buf_ring_addr`/`buf_ring_return` 归还协议。io_uring only(IOCP/epoll 诚实 return false,net 回退预投递 recv)。双验(io_uring + TSan)。**至此 kMore(07a)+ kBuffer(07b)两个当初预留的 flag 全补完**。**co_await 层(2026-07-05,net_multishot.h)**:`MultishotRecvStream` + `co_await stream.next()` → RAII `RecvBuf`(析构自动 buf_ring_return),`TcpStream::read_multishot(bgid,alloc)` 暴露——把上面 proactor 能力提升为可 co_await 的流式 API(net.h 原只暴露 one-shot RecvOp,那零拷贝多发路径是 co_await 够不着的)。关键正确性:单 token 多完成用 slot 内完成队列(单槽会被后续完成 clobber)+ 只在 empty→non-empty 转移 wake(恰一次 wake_rendezvous)+ 堆槽 refcount{kernel,fiber} kernel-ref 撑到 kMore-clear 终止(取消安全,同 net_timeout 族)。**定位:已就位待承重**——正确性/不-回归已验(vs one-shot 同核对称负载统计平局,见 performance/benchmarks-io.md),但暂无真实消费者调用;兑现下一步=拿它改 HTTP body 流式读/proxy 转发。registered files/buffers + vectored readv/writev 是新增(非半成品),可后补。
- **io-07c registered/fixed files ✅第一步(2026-07-05)**:`register_file_table`(sparse `IORING_REGISTER_FILES2`)+ `assign_fixed_slot`/`release_fixed_slot`(线性找空槽 + `FILES_UPDATE` 填/清)+ `submit_recv_fixed`/`submit_send_fixed`(`IOSQE_FIXED_FILE` + 槽索引,内核免每 op fget/fput)。io_uring only(IOCP/epoll 透明 stub 回退裸 fd)。proactor 层 + proactor_test 验证。**性能实测边际(fixed vs raw fd loopback ping-pong 0.8-3.3% 更快,一致为正但小)**。**io-07c 第二步 direct-accept ✅原语已做+实测无收益,TcpStream 主动不改**:`submit_accept_direct`(accept 直落 registered-file 槽、CQE res=槽索引、连接从不是进程 fd)+ `submit_close_direct` + RecvOp/SendOp 加 fixed_slot。proactor_test.DirectAccept 验证全路径。**但 proactor 层(零 fiber)合并计时 direct 稳定慢 ~1%(6 跑,非噪声)——拆相追根后是 teardown 伪影,非 accept 回退**:预建连接分相计时显示 **accept 原语本身 direct 反而快 ~10-30ns/conn**(省内核 fget/fput,合理),那 ~1% 全来自 microbench 即建即关的 teardown 不对称(direct 的 `close_direct`+reap+`FILES_UPDATE` 2 syscall 其一冗余 vs normal `::close` 1 syscall),生产不 accept-完-立刻-close 故不出现。故**停在原语层,不做 TcpStream 拥有语义改造**(accept 机制赢面 ~10-30ns 被 TCP 握手 ~330µs 淹没,为它改核心 RAII = 纯风险);原语留供未来高建连速率/多核争 fd 表场景。详见 performance/benchmarks-io.md direct-accept 拆相节。
- **io-08 TLS(C9)✅**:OpenSSL 加密 socket over co_await stream，包含 SNI、DNS/IP identity verification、有界 shutdown；HTTP transport 已可持有已握手 TLS，真实 HTTPS loopback 覆盖。
- **io-09 异步 DNS + signal ✅(C10)**:`resolve_addr`(tcp.h)只 `inet_pton` 解析 IPv4 字面量(连 getaddrinfo 都没有)——`dns.h/cc` 加 `resolve()` **经 sched-11 spawn_blocking 卸载 getaddrinfo**(抄 tokio 路线,不自造 DNS 协议),内部值拷贝 sockaddr + freeaddrinfo 不泄 addrinfo*,`JoinHandle<ResolveResult>` 回传;字面量 IP 走 getaddrinfo 同步快路径。**signal_set**(`signal.h/cc`)co_await `wait()`:POSIX self-pipe + sigaction(进程全局单例 + 引用计数;handler 只 async-signal-safe 写 pipe,免 signalfd 全线程 mask),Windows SetConsoleCtrlHandler(handler 独立线程 post 到 ring;CTRL_CLOSE/LOGOFF/SHUTDOWN handler 阻塞防提前杀,tokio 同款救命坑)。**单 waiter 模型**(优雅关闭单 fiber 用例,同 tokio Signal)。三后端真测(Win/io_uring/epoll)+ WSL TSan 净。signal 是 C4 优雅关闭的触发器。
- **io-05 文件/shmem(C5)✅**:`proactor` 覆盖 open/close/stat/fsync/fallocate，`File` 按 capability 自动把非原生异步操作卸载到 `IoContext` blocking pool；命名 `SharedMemory` 已有 Windows file mapping / POSIX `shm_open` 的 C++ RAII 与 C opaque ABI。

## 参考

- Why you should use io_uring for network I/O — Red Hat Developer
- Zero-Copy I/O and io_uring for High-Performance Async Servers
- From epoll to io_uring's Multishot Receives (r/programming, 2025)
- Coroutine-Native I/O for C++29 — WG21 P4100R0
