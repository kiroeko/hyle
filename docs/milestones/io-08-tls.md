# io-08：TLS（加密 socket over co_await stream）

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-04（net）|
| **状态** | ✅ 完成（三平台，能力 C9）。Linux io_uring+forced-epoll + WSL TSan 零 race / Windows IOCP（vcpkg openssl 3.6.3）1/1 PASS |

> **范围**：借 OpenSSL memory-BIO shim over co_await `TcpStream`——SSL 引擎只做握手 + 加解密（纯内存同步状态机），密文由自有的 co_await TcpStream read/write 承载。调研 seastar tls_openssl.cc / libevent bufferevent_ssl 定案。
> **cloc**：`tls.h` 72 + `tls.cc` 203 + `coro.h` Task awaiter（+ `tls_test` 210）。

## 为什么是它
面向公网服务器必需，roadmap 原本完全没有 = 真漏。北极星① 的推理服务对外要能上 HTTPS。

## 核心决策
- **memory-BIO 泵（asio ssl::stream / seastar tls_openssl / libevent 模型）**：`SSL_set_bio(ssl, rbio, wbio)` 挂两个 `BIO_s_mem`；外层 pump 循环在每次 `SSL_*` 调用后，按 `SSL_get_error` 的 WANT_READ/WANT_WRITE 决定 `co_await flush_wbio`（drain wbio→sock.write）或 `feed_rbio`（sock.read→BIO_write rbio）。SSL 只做纯内存活，密文进出走自有 TcpStream。
- **`TlsContext`（SSL_CTX client/server 工厂）+ `TlsStream`（TcpStream + SSL* + 2 BIO_s_mem）**：每 conn 一个 `SSL_new` 共享 CTX（cheap，共享已解析的 cert）。
- **自由函数而非成员协程**：成员协程 promise `operator new` 第一参是隐含 `this`、Allocator 落第二位不匹配 `PromiseAllocBase`。故 `tls_handshake`/`tls_read`/`tls_write`/`tls_shutdown` 是**自由函数（Allocator 首参）** + friend 访问私有——HYLE allocator-first 惯用法（同 dns::resolve / spawn_task）。
- **OpenSSL 可选依赖**：CMake `find_package(OpenSSL)` + `HYLE_IO_TLS` 门控——不装 OpenSSL 则整个头 inert、零依赖。

## 踩坑全抄（seastar/libevent 经验）
- `SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER`（部分写重试指针会变）。
- `SSL_call → SSL_get_error → ERR` 三连**不插 co_await**（有栈 fiber 每线程错误队列，每轮 `ERR_clear_error`）——被 co_await 割裂会读错线程的错误。
- `SSL_read` EOF 三分（ZERO_RETURN / SYSCALL+errno0 → EOF）。
- `SSL_set_bio` 接管 BIO 归 `SSL_free`，别双释。
- 单连接单 fiber 串行 = 线程安全（SSL 非线程安全，但一个 fiber 串行访问 + co_await 挂起时无他 fiber 碰它是安全的）。

## coro.h Task 加 awaiter（首个嵌套 Task 消费者）
TLS 泵是首个 `co_await` 子 Task 的消费者。有栈 fiber 下极简：`await_suspend` 循环 resume 子 Task，其内部 co_await 让整个 fiber yield 冻结父帧，无需 continuation 链。

## 热路径分配消除（2026-07-05 优化）
原本 flush/feed 每次 16KB 协程帧分配（那 16KB 局部数组活在 per-call 协程帧的 Allocator-backed operator new 里）。**已修**：16KB scratch 移进 `TlsStream` 懒分配一次全连接复用（`ensure_scratch`），pump helper 收 `char*` scratch，协程帧缩到几十字节（N 次读写 N 次 16KB → 1 次；TSan 净）。安全因一个 TlsStream 单 fiber 拥有、flush/feed 串行永不嵌套。

## 对抗式自审（生命周期 / 正确性 / 活锁 / 平台）
- **红队三大关键面**逐条推演均干净：BIO/SSL double-free、co_await 子 Task 帧 UAF、错误队列被 co_await 割裂。修 2 个：`kWriteMore` 空 wbio 潜在活锁 → 改 feed_rbio 保证 yield；feed_rbio BIO_write 短写检查。
- **① 正确性 / 平台**：loopback TLS1.2+ 握手 + 加密 echo（自签证书）——Linux 双后端 + WSL TSan 零 race / Windows IOCP（vcpkg `openssl:x64-windows` 3.6.3，`-DOPENSSL_ROOT_DIR` 指 vcpkg installed，`find_package` 自动找到，机制与平台无关一次编过）。

## 交付物
- `include/hyle/io/tls.h`（TlsContext + TlsStream + 4 个自由函数 tls_handshake/read/write/shutdown）
- `src/io/tls.cc`（memory-BIO 泵 + SSL 引擎驱动 + ensure_scratch）
- `include/hyle/sched/coro.h`（Task awaiter：co_await 子 Task）
- `tests/io/tls_test.cc`；CMake `HYLE_IO_TLS` 门控。开发环境见 [`docs/dev-setup.md`](../dev-setup.md)

## 参考
- io-design「② 参考库审计真缺口」io-08 节。OpenSSL memory-BIO；seastar tls_openssl.cc / libevent bufferevent_ssl / asio ssl::stream。相关里程碑：io-04（TcpStream 底座）、sched-06（coro.h Task）。install/DLL 分发归 close-01。详见 [performance/benchmarks-io.md](../performance/benchmarks-io.md) 热路径分配消除节。
