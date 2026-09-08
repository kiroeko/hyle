# HYLE io/ 网络库选型

> **V2 状态：`NOT_REVIEWED`。** 本页暂作既有设计输入，不代表当前功能、性能或 API 证书。
> 当前只提供 C++23 source/static/shared，Native FFI 为 `N/A`。BUILD-03 已删除旧 C schema、
> bridge、动态导出和 fixture；正文的其他验收叙事仍待 io 轮次重裁决。

> 本文档定下 HYLE `io/` 模块的网络层选型。上游约束见 [`hyle-scope.md`](hyle-scope.md)（`sched/` 统一调度 + fiber）。

## 2026-09-08 当前架构（维护入口）

```text
普通 C++ 用户
  TcpStream / UdpSocket / File / DNS / Signal / TLS / HTTP/1.1 + HTTP/2
       │ co_await；deadline / CancelToken；RAII
       ▼
IoContext（每 Scheduler worker 一个 private completion engine + blocking pool）
       │ internal submit state                 │ internal completion state
       ▼                                       ▼
private backend engine ──────────────────────► sched::Waker
       │
       ├─ Windows IOCP：真完成；OVERLAPPED intrusive list + owner free-list
       ├─ Linux io_uring：真完成；不可用时自动降为 epoll
       ├─ Linux epoll：readiness→completion；HYLE FlatHashMap<fd, PendingOp>
       └─ macOS kqueue：readiness→completion；HYLE FlatHashMap<fd, PendingOp>

       Linux runtime 选中 io_uring 时
       └─ IoContext::io_uring()
           └─ typed provided-buffer multishot receive
              （显式公开快腿；fixed-file/direct-accept 仍是内部机制）
```

当前不变量：

- socket/file shard 由一个 owner worker 提交和收割；IOCP handle 绑定后不可换
  completion port。
- 普通 C++ 文件腿依据 backend capability 自动选原生异步或 blocking pool；
  用户不取得低层 engine、completion token 或同步 fallback seam。
- plain TCP 与已握手 TLS 经同一个 `HttpTransport` 进入 HTTP framing 核。
- TLS config/session/provider 由独立 `io_tls` factor 持有；OpenSSL 类型不进入
  公开面。client/server factory、handshake/read/write/shutdown 都返回共享 typed
  Error 形状，一个绝对 `OperationOptions` 贯穿 memory-BIO pump 的每次底层操作。
  默认验证 client peer、默认把缺少 close-notify 的 EOF 视为截断；通用 ALPN 与
  context-bound opaque session ticket 不含 HTTP 专有状态机。
  可选 client identity + server-side client CA 组成完整 mTLS；每条 stream 的
  16 KiB memory-BIO scratch 在 typed construction 时由 context 配置的 Mem allocator
  分配一次，allocator backing 生命周期是显式 context/stream 前置条件，operation
  allocator 只拥有 coroutine frame；provider session cache/ticket 默认关闭，只在
  client/server 配置显式启用。
- HTTP/2 协议核固定为 nghttp2；它不拥有 IO/event loop。C++ connection 直接
  驱动 `HttpTransport`。
- TCP read/write/connect/accept 和 UDP recv/send 均有 base、deadline、
  CancelToken 腿；base 腿只用于确定能完成的受控路径。
- Linux-only provided-buffer/multishot receive 只能经成功取得的
  `IoUringExtension` 调用；portable one-shot 永远保留。fixed-file/direct-accept
  目前只是 private backend capability，不能从 capability bit 推断公开操作存在。
- epoll/kqueue pending 表使用 Container 的 `FlatHashMap`。IOCP/io_uring
  不需要 fd readiness 映射，因此不会为统一外观平白引入 hash lookup。
- Native FFI 为 `N/A`；普通 io 修改不得恢复 BUILD-03 已删除的 C schema/bridge。若未来出现已命名
  foreign-runtime 任务，先按全局 [Native FFI 准入策略](native-ffi-policy.md)独立立项。

> **维护警戒线**：portable API 必须在所有支持后端真实实现、等价模拟或明确
> 返回 `StatusCode::Unimplemented`；该状态只是平台/实现不支持当前操作的
> 错误提示，不是 stub 实现。不得以恒定 `false`/成功 no-op 冒充完整能力。
> 现存例外和补全工单见
> [`io-capability-fill-list.md`](../implementation/tasks/io/io-capability-fill-list.md)。

IO 正在按 factor 重构；当前 `io_core` 仍为 `AUDITED`，不是已收口证书。当前轮与
唯一状态见 [`io-v1-core-factorization`](../implementation/tasks/io/io-v1-core-factorization.md)
和 [`module-status.json`](../implementation/module-status.json)。下文“已落地纪要”只保留
施工历史，若与本节、活动卡或代码冲突，以活动卡和代码为准。

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
├─ capabilities.h            typed backend/capability snapshot
├─ io_context.h              scheduler-bound runtime owner
├─ operation.h / buffer.h    deadline/cancel 与 borrowed buffer sequence
├─ tcp.h / udp.h / local.h   普通 C++ RAII transport facade
├─ file.h / dns.h / signal.h 文件、解析与关闭事件
├─ io_uring.h                Linux-only typed receive accelerator
├─ tls.h / http*.h           由独立协议 factor 拥有，仍保持统一用户 namespace
└─ detail/proactor.h + net*  不承诺的完成引擎与 awaitable 实现 seam


src/io_core/
├─ proactor.cc               四后端按平台条件编译；OS 类型不进入公开 API
├─ io_context.cc + operation.cc + io_uring.cc
└─ file.cc / dns.cc          capability-aware blocking offload

src/io_tls/                  OpenSSL provider/config/session/stream owner
src/io_http/                 HTTP/1 lifecycle + private picohttpparser
src/io_http2/                HTTP/2 lifecycle + private nghttp2
```

F09 已把协议源码、测试、benchmark、依赖与导出边分别迁入 `io_tls/io_http/io_http2`；
它们不再属于 `io_core`。对用户暴露的仍是统一 `hyle::io` 任务型 awaitable/RAII，对下仍是四种
private backend；调度和 completion lifetime 由 HYLE 自己管理。

## 已落地（io-01/02 历史实现纪要）

> 下面 3 项原为"待深挖"(JIT 延到开工),io-01/02 开工时已定并实现,记录当前真相:

1. **proactor.h 接口形状**:`Proactor{init/reap/post}` + `Completion{user_data,result,flags,buffer_id}`。收割式(reap 批量取完成),平台无关。**接口在写之前先研究 liburing v2.15 的 SOTA 机制**(multishot/registered buf/SQPOLL/linked SQE),确保不挡:`flags` 带 `kMore`(multishot 一提交多完成、token 不早回收)+`kBuffer`(零拷贝 provided buffer id);reap 批量=io_uring for_each_cqe(N 完成 1 次 release-store);timeout 走 EXT_ARG 不额外 syscall。submit 侧的具体 op(read/write/accept + link/fixed-file flag)留 io-04 net/。
2. **io_uring↔fiber 映射**:`user_data`(SQE)= Waker token,完成时原样回来(`cqe->user_data`)→ reap 交回 → sched wake。这正是 sched-design §3.3 的物理机制,IOCP 用 completion key 同构。
3. **只借 syscall shim 的力度**:实测比 io-design 预想更省——**连 liburing 都不链**,只用 3 个 io_uring syscall + `linux/io_uring.h` uapi 头 + 自写内联环算术(barrier 精确复刻 liburing v2.15 的 4 个 fast-path 点)。IOCP/kqueue 本就 OS 原生。**零第三方链依赖。**

**当时后端进度（2026-07-03 历史快照）**:io-01 **收割接缝**在 IOCP(Windows)+ io_uring(Linux)双后端均 ✅ 真测。io-02 **read/write 真 op 两后端全通 ✅**:io_uring 填 SQE(WSL 真测 write→read + bad-fd 负 errno;TSan 净);IOCP 用 `ReadFile/WriteFile` + per-op OVERLAPPED——proactor 侵入式在途链表**自持** OVERLAPPED 生命周期(OVERLAPPED 是 IoOp 首成员;reap cast `lpOverlapped` 回 IoOp;`post()` 保持 nullptr 以区分合成唤醒;dtor 先 CancelIoEx 再排空端口回收防关机 UAF)。**先前"IOCP 真 op 需随 net/ 定形"是过度谨慎——OVERLAPPED 纯 IOCP 内部细节 proactor 自持,net/ 不可见,故此处已落地。** io-03 **epoll 降级 + SOTA-first 后端选择 ✅**:`init()` 总先试 io_uring,仅真不可用(老内核/`io_uring_disabled` sysctl/容器 seccomp)才退 epoll——生产走 io_uring,epoll 保 io_uring 被禁环境仍可跑;epoll reactor→completion 模拟(记 pending op、就绪时内联 IO;**红队 L3 修 2026-07-05:pending 拆 in/out 双槽,同 fd 全双工 recv+send 并发不再互相覆盖孤儿**)。后续 kqueue、UDP 与 park 合并均已完成，见本页顶部当前架构。

## 2026-07-05 历史缺口快照

> 本节保留当时的施工判断；其中 io-07b/c 所述 portable/stub 债已在
> 2026-07-20 通过 `IoUringExtension` 收口，STATX 也已改为原生 SQE。
> 当前维护者应以本文顶部架构和 capability fill list 为准。

**① JIT 延后(次要):**
- **reactor 后端(mac kqueue)✅ 已落地(2026-07-05,Apple Silicon 真机)**：kqueue 镜像 epoll reactor 把 readiness 合成为 completion。2026-07-20 已把 io_uring 专属 buf-ring/fixed-files/direct-accept 从通用面移入显式 extension，kqueue 不再编译恒定失败的后端方法。真机 io/net/tcp/udp/file/signal/timeout/http 全绿。
- **park 合并 ✅ 已落地**(2026-07-03):worker 空闲以完整 timeout(timer deadline/INFINITE)阻塞 GETEVENTS reap,不再 1ms 轮询;跨线程唤醒经 `Proactor::wake()`(io_uring 专用 eventfd 上常驻 `IORING_OP_READ`,`write` 打断 reap,不碰单生产者 SQ;IOCP `PostQueuedCompletionStatus`;epoll level-triggered eventfd)。实测空闲 500ms 的 `io_uring_enter` 从 421→3(~140x),吞吐无变化。

**② 参考库审计真缺口(能力驱动,见 roadmap 轴 C / milestones):**
- **io-06 带超时 io(C2)✅**:TCP read/write/connect/accept 与 UDP recv/send 的 deadline/cancel 腿已完成；取消在途内核 op 用 io_uring `ASYNC_CANCEL`、IOCP `CancelIoEx`、epoll/kqueue 删除 readiness watch 并合成最终 completion。
- **io-07a multishot accept ✅(kMore 半成品补完)**:`submit_accept_multishot` 三后端(io_uring `IORING_ACCEPT_MULTISHOT` 原生;IOCP AcceptEx 循环 + epoll 重 arm 两路 emulation,对上同一 kMore 流)。一次 arm 持续吐连接,补完 `Completion.kMore` 当初留的接口。双平台 + TSan 净。
- **io-07b provided buffer ring + multishot recv ✅(io_uring 显式快腿)**：`IoUringExtension::register_buffer_ring` + `submit_recv_multishot` + `buffer_address/return_buffer`。`MultishotRecvStream` 和 `TcpStream::read_multishot` 都必须拿到有效 extension；非 io_uring 后端不再暴露首次 `next()` 才失败的伪 portable 入口。普通跨平台连续接收就是循环 `read()`。
- **io-07c registered/fixed files ✅第一步(2026-07-05)**:`register_file_table`(sparse `IORING_REGISTER_FILES2`)+ `assign_fixed_slot`/`release_fixed_slot`(线性找空槽 + `FILES_UPDATE` 填/清)+ `submit_recv_fixed`/`submit_send_fixed`(`IOSQE_FIXED_FILE` + 槽索引,内核免每 op fget/fput)。io_uring 原生实现已验证；IOCP/epoll 使用普通 fd 产品腿，但这些扩展方法本身仍需移出 portable Proactor，不能把恒定失败叫“透明 fallback”。**性能实测边际(fixed vs raw fd loopback ping-pong 0.8-3.3% 更快,一致为正但小)**。**io-07c 第二步 direct-accept ✅原语已做+实测无收益,TcpStream 主动不改**:`submit_accept_direct`(accept 直落 registered-file 槽、CQE res=槽索引、连接从不是进程 fd)+ `submit_close_direct` + RecvOp/SendOp 加 fixed_slot。proactor_test.DirectAccept 验证全路径。**但 proactor 层(零 fiber)合并计时 direct 稳定慢 ~1%(6 跑,非噪声)——拆相追根后是 teardown 伪影,非 accept 回退**:预建连接分相计时显示 **accept 原语本身 direct 反而快 ~10-30ns/conn**(省内核 fget/fput,合理),那 ~1% 全来自 microbench 即建即关的 teardown 不对称(direct 的 `close_direct`+reap+`FILES_UPDATE` 2 syscall 其一冗余 vs normal `::close` 1 syscall),生产不 accept-完-立刻-close 故不出现。故**停在原语层,不做 TcpStream 拥有语义改造**(accept 机制赢面 ~10-30ns 被 TCP 握手 ~330µs 淹没,为它改核心 RAII = 纯风险);原语留供未来高建连速率/多核争 fd 表场景。详见 performance/benchmarks-io.md direct-accept 拆相节。
- **io-07c 2026-07-20 收口补记**：register/assign/release、fixed
  recv/send、direct accept/close 全部位于 `IoUringExtension`；普通
  `RecvOp/SendOp` 的 `fixed_slot` 暗门已删除，IOCP/kqueue 不再有 stub
  定义。`release_fixed_file` 现在检查内核 `FILES_UPDATE` 结果。
- **io-08 TLS(C9) 历史落点**：最初实现了 OpenSSL memory-BIO over co_await
  stream、SNI 和 HTTPS loopback，但当时仍暴露 native provider pointer、invalid
  object 与 `int/long`。当前 `io_tls` v1 已用 typed config/session/result、统一
  deadline/cancel、通用 ALPN、严格 truncation 与显式 shutdown policy 取代这些
  旧形状；是否最终冻结只看活动 closure card 与 registry。
- **io-09 异步 DNS + signal ✅(C10，历史落点)**：当时首次用 blocking pool 卸载
  `getaddrinfo`，并建立 signal self-pipe/console-handler 路径。当前 resolver 已改为
  `resolve(IoContext&, ...).join(OperationOptions)`：host 与全部 IPv4/IPv6 endpoint
  都拥有所有权，deadline/cancel 可提前返回，而不可强停的 OS 查询通过引用状态在后台
  安全退休；数字地址由 `Endpoint::parse()` 处理。`SignalSet::create()` / `wait()`
  返回 typed result，维持进程级单 owner、顺序单 waiter 合同。
- **io-05 文件/shmem(C5)✅（历史）**：`proactor` 覆盖 open/close/stat/fsync/fallocate，`File` 按 capability 自动把非原生异步操作卸载到 `IoContext` blocking pool。旧 `io::SharedMemory` 曾实现 Windows file mapping / POSIX `shm_open`；Mem v2 已把 storage owner 迁为 `mem::SharedMapping`，Platform 保留 syscall mechanism，IO 只组合 IPC/async protocol，旧 C++ owner 与伪 C ABI 均不再是当前产品面。

## 参考

- Why you should use io_uring for network I/O — Red Hat Developer
- Zero-Copy I/O and io_uring for High-Performance Async Servers
- From epoll to io_uring's Multishot Receives (r/programming, 2025)
- Coroutine-Native I/O for C++29 — WG21 P4100R0
