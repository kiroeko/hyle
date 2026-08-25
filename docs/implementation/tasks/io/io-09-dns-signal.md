# io-09：异步 DNS 解析 + signal_set

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-04（net）|
| **状态** | ✅ 完成（能力 C10）。三后端真测（Win / io_uring / epoll）+ WSL TSan 净 |

> **范围**：两件 C10 能力。**异步 DNS**：`resolve()` 经 sched-11 spawn_blocking 卸载 getaddrinfo。**signal_set**：`SignalSet` co_await `wait()` 等 SIGTERM/SIGINT（C4 优雅关闭的触发器）。
> **cloc**：`dns.h`+`cc` 138 / `signal.h`+`cc` 340（+ `dns_test` 96 + `signal_test` 105）。

## 为什么是它
`resolve_addr`（tcp.h）实为 `inet_pton`——只认 IPv4 字面量，连 getaddrinfo 都没有（比审计说的「同步阻塞」更糟）。真实主机名解析必需。signal 则是 C4 优雅关闭的触发器：没有它 runtime 收不到 Ctrl-C。

## 核心决策
### 异步 DNS（抄 tokio net/addr.rs 路线）
- getaddrinfo 是**阻塞 syscall**（可命中 /etc/hosts、nsswitch、网络）；在 shard worker 上跑会 stall 钉在该核的**每一个** fiber——share-nothing 违规。
- 修：**不重实现 DNS 协议**（不走 libevent evdns / c-ares），而是 `resolve(pool, alloc, host, port)` **经 spawn_blocking 卸载 getaddrinfo**。pool 线程阻塞、worker 继续跑其他 fiber、结果经现有 `JoinState`/`JoinHandle` 回传（fiber-aware / thread join）。几乎零新机制——DNS 只是 spawn_blocking 的领域特化应用。
- 闭包内**值拷贝 sockaddr + freeaddrinfo**，`addrinfo*` 不逃出 pool 线程。`ResolvedAddr` 持 sockaddr_storage 大小的值拷贝；`ResolveResult` 最多 kMax=8 个地址（happy-eyeballs 式回退）。hints AF_UNSPEC + SOCK_STREAM（v4+v6 不重复）。字面量 IP 也走 getaddrinfo 同步快路径。host 字符串**拷进 job**，调用方 buffer 无需活过 await。

### signal_set
- **参考库共识**（tokio signal / libevent signal.c 通读）：**handler 里不干活**，只做一件 async-signal-safe 的事，event loop 把它变成普通 completion。
- **POSIX**：进程全局 sigaction handler 写**一个字节**（signo）到 self-pipe（async-signal-safe；线程无关——内核可在任意线程含 worker mid-fiber 跑 handler，裸 write 安全）。SignalSet arm 一个普通 proactor READ 在 pipe 读端；字节到达完成即唤醒 fiber。**用 self-pipe 而非 signalfd**——signalfd 要在每个线程一致地 block 信号，会跟已跑的 scheduler workers 打架；self-pipe 无需 mask surgery。进程全局单例 + per-signal 引用计数安装/卸载；pipe 两端 O_NONBLOCK。
- **Windows**：无 POSIX 信号。`SetConsoleCtrlHandler` 在**独立线程**跑 handler（故无 async-signal-safe 限制），该线程经线程安全的 `PostQueuedCompletionStatus`（proactor.post）post 到等待 SignalSet 的 ring。**`CTRL_CLOSE/LOGOFF/SHUTDOWN` handler `Sleep(INFINITE)` 防 OS 提前杀**（tokio 同款救命坑）；want `fetch_or` 累积。
- **单 waiter 模型**：一个 SignalSet 一次服务一个 co_await wait（同 tokio Signal 单流）。要 fan-out 多 waiter 用 sched watch/channel 广播。

## 对抗式自审（正确性 / TSan）
- **红队自审**修 self-pipe 读端阻塞 → nonblock + Win want 覆盖 → OR。
- **① 正确性**：DNS 端到端 + signal 三后端真测（Win 注册冒烟 / POSIX raise 端到端往返）。
- **④ TSan**：WSL 净。

## 交付物
- `include/hyle/io/dns.h`（ResolvedAddr / ResolveResult / resolve() + detail::resolve_blocking）+ `src/io/dns.cc`
- `include/hyle/io/signal.h`（SignalBits / SignalSet + WaitAwaitable）+ `src/io/signal.cc`（进程全局表 + self-pipe / SetConsoleCtrlHandler）
- `tests/io/dns_test.cc` / `tests/io/signal_test.cc`

## 参考
- io-design「② 参考库审计真缺口」io-09 节。tokio net/addr.rs（getaddrinfo 卸载）/ tokio signal / libevent signal.c。相关里程碑：sched-11（spawn_blocking / BlockingPool）、sched-10（JoinHandle）、sched-08（signal 触发的优雅关闭）、io-04（proactor READ）。
