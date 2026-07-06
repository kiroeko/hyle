# io-04：net——tcp/udp co_await socket 层 + shard-per-core

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-02（真 op）|
| **状态** | ✅ 完成（Win IOCP + Linux io_uring + forced-epoll 三后端；TCP echo + 32 连接分片跨 4 核 + UDP echo，双平台 + TSan 净）|

> **范围**：io-design「模块形状」的 `net/` 层——proactor 之上的 **co_await socket 层**。这是 io/ 遇见 sched/ 的接缝：一个 async op 挂起**fiber**（不只协程），完成时 completion 带 result 回来唤醒。含：proactor socket op（recv/send/accept/connect + udp recvfrom/sendto，三后端）、`net.h` co_await awaitable、**shard-per-core** 性能天花板架构、`TcpListener`/`TcpStream`/`UdpSocket` RAII 封装。
> **cloc**：io 支柱累计 1550（`net.h`/`io_context`/`tcp`/`udp` + proactor socket/datagram op；proactor 收割层 286 归 io-01/02/03）+ sched 亲和改动。**范围诚实标注见下。**

## 为什么是它
proactor 有了真 IO，但对上还是裸 token。io-04 把它提升成**能写服务器的东西**：`co_await stream.read(buf, len)`。fiber-blocking 与 proactor 一一对应（提交→挂起 fiber→完成唤醒），这正是 io-design 选 proactor 而非 reactor 的原因。这也是 io 支柱第一次有「真实并发连接负载」，从而第一次能做有意义的 benchmark。

## 核心决策
### co_await 层（`net.h`）
`IoOperation{waker, result, flags, completed}` slot 活在**挂起 fiber 的冻结栈**上，`user_data = &op`；完成时 reaper 写 `op.result` 并调 `op.waker.wake()`。`RecvOp`/`SendOp`/`AcceptOp`/`ConnectOp`/`RecvFromOp`/`SendToOp` 都是「submit + suspend」的 awaitable。`kSubmitFailed = -1000`（不与任何真 errno 撞——base op 早先用 -1 会撞 -EPERM，已修）。

**栈 slot 生命周期契约（红队 L5）**：base op 永不被取消，只被自己的 completion 唤醒，且挂起 fiber 一直算在 scheduler 的 `outstanding` 里——drain/shutdown 会阻塞到完成到达。唯一能打破的方式是「不 drain 到完成就销毁带在途 base op 的 fiber」→迟到 completion 写已释放栈 = UAF。故：completion 不保证到达的 op（可能永不发的 peer / 必须限时）**必须用 TIMED op（io-06）或 cancellable op（net_cancel）**，不能用裸 base op。这是契约，不是意外。

### shard-per-core（share-nothing 性能天花板，seastar/DPDK 式）
- sched 加 fiber 亲和 `spawn_on(worker, f)`（每 worker 一条专属 MPMC 队列不被偷 + 唤醒回原核 `unpark_specific`，**不碰防惊群 Dekker / steal CAS**）。
- `IoContext` 每 worker 一个 proactor，`bind_fd(worker, fd)` 把连接绑 owner 核（满足 IOCP「一 socket 一端口」），连接 fiber 钉该核 → submit/reap/resume 全程单核**无锁无跨核同步**。
- **sched↔io 解耦**：纯函数指针 `set_io_poll` hook——sched **不**依赖 io，反向 io→sched。`IoContext::proactor_for_current()` 让 awaitable 拿到「跑本 fiber 那个 worker 的 ring」（Plan A 不变量：op 必须在同一 worker submit + reap，跨 worker 的 ring 会 race reaper）。

### 封装
- `tcp.h`：`TcpStream`（RAII fd + co_await read/write/connect）、`TcpListener`（co_await accept→绑核 TcpStream，round-robin 分片跨核）、`tcp_listen`/`tcp_socket`/`resolve_addr`。
- `udp.h`：`UdpSocket`（recv_from/send_to，无连接/accept，shard-per-core 绑核）、`udp_bind`/`udp_socket`。UDP 走 io_uring RECVMSG/SENDMSG（msghdr+iovec 建在 awaitable 的 `scratch_` = 冻结 fiber 栈，跨过 async 完成，照 tokio-uring recv_from.rs 的 Box<msghdr> 生命周期契约）、IOCP WSARecvFrom/WSASendTo、epoll 内联 recvfrom/sendto。

## 硬点：pipelined 负载揪出的 P0 hang
512 流 pipelined 负载暴露：worker 空闲时的 `reap(0)` 空 CQ 只做 submit-only enter（无 GETEVENTS），io_uring 完成 task-work 从不在本线程跑 → 超过 ring 深度的在途 op 的 completion 永不 surface → 死等。**修**：worker park 边界有在途 op 时改**阻塞在 GETEVENTS reap**（`Proactor::in_flight()` 三后端 gate），非 futex park——这是正确性要求，不是优化。

## 对抗式自审（正确性 / 性能 / TSan）
- **① 正确性**：co_await 往返 + 两 fiber echo + 32 连接分片跨 4 核 + 真 loopback TCP echo（listen/accept/connect/read/write）+ loopback UDP echo，三后端双平台。
- **② 性能**（`bench/io/echo_bench.cc`，echo 64 连接×64 round-trip + pipelined 512 流×16）：submit 批量落地后单 worker 728K→798K（批量前恒 avg_batch=1.0）；reinject 同核短路 + COOP 再 -33% 到 ~748K/1337ns，n=15 within-run **平裸 epoll、单核赢 asio 1.25x/libuv 1.52x**、4w 赢 epoll 2.25x；shard-per-core 扩展 1w→4w ~2x；pipelined 高扇出 1w→8w **5.2x**。
- **④ TSan**：净（揪修 io_poll 跨线程发布 race→atomic；affinity 44/44 TSan）。

## 范围诚实标注
io-04 施工中吸收了 socket op + shard-per-core + 封装 + benchmark + park 合并 + 一个 P0 hang 修复共 6 件事，**已超出单里程碑「1-4k 行一步」的口径**。真实LOC 列填 io 支柱**累计** 1550 而非本步净增。追认它是「C1 并发 io 服务能力」的落地筐，不再回溯拆表——历史归 git。

## 交付物
- `include/hyle/io/net.h`（IoOperation + Recv/Send/Accept/Connect/RecvFrom/SendToOp awaitable + complete/to_token）
- `include/hyle/io/io_context.h` + `src/io/io_context.cc`（per-worker rings + poll hook + bind_fd + proactor_for_current）
- `include/hyle/io/tcp.h` + `src/io/tcp.cc`；`include/hyle/io/udp.h` + `src/io/udp.cc`
- `src/io/proactor.cc`（三后端 recv/send/accept/connect/recvfrom/sendto）
- `tests/io/net_test.cc` / `tcp_test.cc` / `udp_test.cc`；`bench/io/echo_bench.cc`

## 参考
- io-design「模块形状」「已落地 + 剩余缺口」；sched-design §3.2（每核一个 io_uring 实例）。seastar/DPDK（share-nothing）；tokio-uring recv_from.rs（msghdr 生命周期）。相关里程碑：io-02（真 op）、io-06（超时）、io-07（multishot）、sched-03（spawn_on/亲和）。详见 [performance/benchmarks-io.md](../performance/benchmarks-io.md)。
