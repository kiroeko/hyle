# io-03：epoll 降级后端 + SOTA-first 后端选择（+ kqueue mac）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-01（收割接缝）|
| **状态** | ✅ epoll 降级 ✅ + 后端选择 ✅ + kqueue(mac) 真机已做（Apple Silicon）。双平台 8/8 + TSan 净 |

> **范围**：给 proactor 补 **reactor→completion 模拟**后端——Linux epoll（io_uring 被禁环境的降级）+ macOS kqueue。核心是 `init()` 的 **SOTA-first 后端选择**：生产有 io_uring 就走 io_uring，epoll 只保住「io_uring 被禁的硬化容器/k8s」仍能跑。
> **cloc**：+130。

## 为什么是它
跨平台高性能网络最硬的现实（io-design「两个调查事实」）：**IOCP/io_uring 是真 proactor，epoll/kqueue 只是 reactor**（报「可读了」，应用自己做 IO）。HYLE 对上暴露统一的 proactor 完成式 API，reactor 后端内部**模拟完成**。这是让 mac/BSD、以及禁用 io_uring 的容器都能跑同一套 net/ 代码的必要层。

## 核心决策
- **SOTA-first `init()`**：总先试 io_uring，仅在**真不可用**才退 epoll——老内核 `ENOSYS` / `kernel.io_uring_disabled` sysctl / 容器 seccomp→EPERM。不是「优先兼容」而是「优先性能，兜底可用」。
- **epoll = reactor→completion 模拟**：epoll 只报就绪；proactor 记住每个 op（fd→{op,buf,len,token}，`EPOLLONESHOT`），就绪时**自己内联做 IO** 再合成 `Completion`。post 走软件唤醒队列。
- **kqueue 镜像 epoll**：`#elif HYLE_OS_MACOS` 块——`EVFILT_READ/WRITE + EV_ONESHOT` 替 EPOLLIN/OUT、`EVFILT_USER + NOTE_TRIGGER` 替 wake eventfd（无需 self-pipe）、dual-slot pending_in/out、spurious re-arm、multishot 模拟、常规文件 `S_ISREG` 走同步 fallback（kqueue 无 EPERM 信号）。

## 硬点：全双工双槽（红队 L3）
同一 fd 上 recv + send 并发时，pending op 若只有单槽会**互相覆盖成孤儿**。修法：pending 拆 **in/out 双槽**（`pending_in`/`pending_out`），全双工不再互踩。这个 bug 在 epoll（2026-07-05）和 kqueue 两边都按同一方案处理。

## 对抗式自审（正确性 / 平台 / 残留）
- **① 正确性**：WSL 强制 epoll 真测 pipe write/read 往返 + post 唤醒；双平台 8/8 + TSan 净。
- **平台真机**：kqueue 在 M4 Air 真机跑 io/net/tcp/udp/file/signal/timeout/http 全绿，Linux 零回归。
- **⑤ 残留**：io_uring 专属能力（buf_ring / fixed-files / direct-accept，见 io-07b/c）在 kqueue 上**诚实 stub false**，不假装支持。

## 交付物
- `include/hyle/io/proactor.h`（后端无关签名不变；`try_init_uring`/`try_init_epoll` 私有静态）
- `src/io/proactor.cc`（epoll reactor→completion 模拟 + `epoll_arm`/`epoll_arm_dgram`；kqueue `#elif HYLE_OS_MACOS` 块）
- `tests/io/proactor_test.cc`

## 参考
- io-design 后端矩阵、「借 vs 造」方案 C、「① JIT 延后」reactor 后端(mac kqueue) 节。seastar `reactor_backend`（epoll/aio 也对上呈现 completion）照镜。相关里程碑：io-01/02（收割 + 真 op）、io-04（socket 层消费三后端）。
