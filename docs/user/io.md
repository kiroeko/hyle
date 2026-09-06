# IO 使用指南

> **`provisional / NOT_REVIEWED_V2`。** IO 当前只提供 C++ 产品面。
> 本页适用于 C++23 source/static/shared 三种交付，不描述 C API 或跨语言调用层。

HYLE IO 有两层。普通 C++ 程序优先用 `IoContext` 与 RAII/`co_await`
封装；需要精确控制后端或自己管理状态机时才直接用 `Proactor`。

## 1. 运行时结构

```text
Scheduler worker 0 ── IoContext ring 0 ── socket/file shard 0
Scheduler worker 1 ── IoContext ring 1 ── socket/file shard 1
        ...                    ...
                         ┌─ Windows: IOCP
Proactor runtime select ─┼─ Linux: io_uring, unavailable时降为 epoll
                         └─ macOS: kqueue
```

一个 socket/file 绑定一个 owner worker。创建连接后，把处理它的 fiber 用
`Scheduler::spawn_on(owner)` 钉到相同 worker；不要在另一个 worker 上提交它的
操作。`Proactor::backend_kind()` 可观察实际后端，不能用操作系统名字猜 Linux
是否真的取得 io_uring。

IOCP/io_uring 是完成式后端，不做 fd hash lookup。epoll/kqueue 必须把 readiness
还原成 completion，内部用 HYLE `FlatHashMap<int, PendingOp>` 保存读写两侧；
这是实现细节，用户不需要提供 map，也不会因此把 SIMD 宏或 ISA flag 带进调用方。

## 2. TCP、超时与取消

```cpp
hyle::sched::Scheduler sched{4, hyle::mem::system_allocator(), 4096};
hyle::io::IoContext io{sched, 4};
auto listener = hyle::io::tcp_listen(io, 0, "::", 8080);

// 在 worker 0 的 fiber 中：
auto stream = co_await listener.accept(/*assign_owner=*/1);

// 在 worker 1 的连接 fiber 中：
char buffer[4096];
auto deadline = hyle::time::deadline_after_ns(5'000'000'000ull);
auto n = co_await stream.read_until(
    buffer, sizeof(buffer), hyle::mem::system_allocator(), deadline);
```

`read/write/connect/accept` 是无截止期的最薄腿，适合 completion 必然到达的
受控路径。面向网络对端应使用 `*_until` 或 `*_cancel`：TCP 的
read/write/connect/accept 与 UDP 的 recv/send 都有 deadline 和
`CancelToken` 变体。timeout 返回 `-110`，结构化取消统一返回
`hyle::io::kCancelled`，不暴露 IOCP 与 POSIX 的错误码差异。

buffer、sockaddr 和取消 token 必须活到 await 完成。不要在仍有裸 base op
挂起时强行销毁 fiber；需要有界退出时用 deadline/cancel 腿。

## 3. 文件：默认不会堵 scheduler worker

```cpp
auto file = co_await hyle::io::open_file(
    io, owner, path,
    hyle::io::Proactor::kRead | hyle::io::Proactor::kWrite);
auto n = co_await file.read(buffer, size, offset);
co_await file.close();
```

`File` 自动选择：

- io_uring 的 open/read/write/fsync/fallocate/close 走原生异步；
- IOCP read/write 走 OVERLAPPED；
- stat，以及 IOCP/epoll/kqueue 没有原生异步形式的文件操作，自动卸载到
  `IoContext` 自带的 blocking pool。

因此普通 C++ 用户无需先判断 backend。`File` 析构只作为同步防泄漏后备；
正常路径应 `co_await close()`。需要完全掌控 syscall/queue 形状的用户可直接
调用低层 `Proactor::submit_*`，并自行承担 capability 与阻塞语义。

命名共享 storage 属于 Mem：使用 `mem::SharedMapping::create/open/unlink`；IO 只负责建立在
它之上的 namespace/handle transfer、notification 或异步协议。跨进程数据结构只能保存
offset/POD，不能保存本进程裸指针。详见 [Mem 指南](mem.md)。

## 4. HTTP 与 HTTPS

HTTP/1.1 server/client 都接受 plain `TcpStream`。启用 OpenSSL 时也接受已经
完成握手的 `TlsStream`，同一解析与 framing 核直接运行在 TLS transport 上：

```cpp
hyle::io::TlsStream tls{std::move(tcp), client_ctx, false, "api.example.com"};
if (co_await hyle::io::tls_handshake(alloc, tls) != 0) co_return;

hyle::io::HttpClientConn client{alloc, std::move(tls)};
auto* response = co_await client.request(
    {"GET", 3}, {"/health", 7}, {"api.example.com", 15});
```

verifying client 必须给 server name；它同时用于 SNI 与证书 DNS/IP identity
校验。`HttpRequest`/`HttpResponseView` 的 string view 只活到同一连接的下一次
request。流式响应用 `HttpResponse::set_stream`；`add_header` 返回 false 表示
固定 header 槽已满。HTTP 会拒绝 TE+CL、冲突 Content-Length、非 bare chunked
和 CRLF/token 注入。

当前 HTTP convenience 尚未把整个 request 生命周期绑定到一个统一 deadline /
CancelToken；要求强制有界退出的服务应在连接层使用 cancellable/timed I/O，
并在 shutdown 阶段停止 accept 后排空连接。该项仍是 IO 的产品尾项。

HTTP/2 使用 `Http2ClientConn` / `Http2ServerConn`。它支持同连接多 stream、
逆序完成、HPACK、RST/GOAWAY 与流控；TLS 必须先调用
`TlsContext::enable_http2()` 并完成握手，构造函数会拒绝 ALPN 不是 `h2` 的
stream。当前 body 是最多 8 MiB 的完整 fixed buffer，不适合无限 token stream：

```cpp
hyle::io::Http2ClientConn h2{alloc, std::move(tls)};
auto id = co_await h2.submit_request(
    "POST", "https", "api.example.com", "/v1/infer",
    nullptr, 0, request_body);
auto* response = co_await h2.next_response();
```

可以连续 `submit_request` 后再取 response；必须按返回的 stream ID 关联，不要
假设 response 与提交顺序一致。连接仍是 single-owner，所有方法在同一个
transport owner fiber 上调用。

## 5. DNS、signal 与 Linux 专属快腿

- `resolve()` 把 `getaddrinfo` 卸载到 blocking pool，结果同时携带 IPv4/IPv6
  family；字面量地址可用 `resolve_addr()`。
- `SignalSet` 是进程级单 owner、单 waiter 的优雅关闭入口，析构会恢复宿主旧
  handler。
- provided-buffer multishot recv、fixed-file 与 direct accept 是 io_uring
  专属快腿。先查询/尝试注册，失败后回到 portable one-shot；不能把“不支持”
  当成一个极快 benchmark 样本。
