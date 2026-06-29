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
├─ proactor.h       (统一完成式抽象, 平台无关)
└─ net/             (socket/tcp/udp, co_await 风格, 遇 IO 让出 fiber)
        ↓ 绑定
     sched/         (fiber 提交即挂起, completion 即唤醒)
```

一句话：**对上一套 proactor 抽象，对下 4 个后端，中间只借 syscall 封装、调度自己管。**

## 待深挖（io-01 开工时定，非现在）

> 选型决策（借谁/proactor/后端矩阵）已全部锁定——那是设计阶段必须拍的。下面 3 项是**实现机制**，按 JIT 原则**有意延到 io-01 开工时定**：它们依赖 sched completion 脊实现中学到的东西，现在写 = 空想。这是 io 作为 P5 未开工支柱的正常状态，非遗漏。

1. reactor 后端（mac kqueue / Linux epoll）如何把"就绪"模拟成"完成"而不丢性能。
2. io_uring SQ/CQ 如何与 fiber 挂起/唤醒对接（completion → 唤醒哪个 fiber 的映射）——直接复用 `sched-design.md` 第三节的 Waker 脊。
3. proactor.h 统一抽象的具体接口形状（提交/取消/超时/批量）。

## 参考

- Why you should use io_uring for network I/O — Red Hat Developer
- Zero-Copy I/O and io_uring for High-Performance Async Servers
- From epoll to io_uring's Multishot Receives (r/programming, 2025)
- Coroutine-Native I/O for C++29 — WG21 P4100R0
