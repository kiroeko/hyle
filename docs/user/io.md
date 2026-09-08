# IO 使用指南

> **`provisional / NOT_REVIEWED_V2`。** IO 当前只提供 C++ 产品面。
> 本页适用于 C++23 source/static/shared 三种交付，不描述 C API 或跨语言调用层。

普通 C++ 程序使用 `IoContext` 与 RAII/`co_await` 封装。`Proactor`、native
completion token 和负错误码已经收进 `detail` 后端，不是公开产品面；不要 include
`<hyle/io/detail/...>`，也不要依赖共享库中为公开 inline awaitable 保留的内部符号。

## 1. 运行时结构

```text
Scheduler worker 0 ── IoContext ring 0 ── socket/file shard 0
Scheduler worker 1 ── IoContext ring 1 ── socket/file shard 1
        ...                    ...
                                  ┌─ Windows: IOCP
private completion engine select ─┼─ Linux: io_uring, unavailable时降为 epoll
                                  └─ macOS: kqueue
```

一个 socket/file 绑定一个 owner worker。创建连接后，把处理它的 fiber 用
`Scheduler::spawn_on(owner)` 钉到相同 worker；不要在另一个 worker 上提交它的
操作。后端能力通过 `IoContext` 的 typed capability 查询观察；不能用操作系统
名字猜 Linux 是否真的取得 io_uring。

IOCP/io_uring 是完成式后端，不做 fd hash lookup。epoll/kqueue 必须把 readiness
还原成 completion，内部用 HYLE `FlatHashMap<int, PendingOp>` 保存读写两侧；
这是实现细节，用户不需要提供 map，也不会因此把 SIMD 宏或 ISA flag 带进调用方。

## 2. TCP、超时与取消

```cpp
hyle::sched::Scheduler sched{4, hyle::mem::system_allocator(), 4096};
hyle::io::IoContext io{sched};
auto address = hyle::io::Endpoint::parse("::", 8080);
auto listener = hyle::io::TcpListener::bind(io, 0, *address);

// 在 worker 0 的 fiber 中：
auto accepted = co_await listener->accept(/*assign_owner=*/1);
if (!accepted) co_return;
auto stream = std::move(*accepted);

// 在 worker 1 的连接 fiber 中：
std::array<std::byte, 4096> buffer{};
hyle::io::OperationOptions operation;
operation.deadline = hyle::time::deadline_after(std::chrono::seconds(5));
auto read = co_await stream.read_some(buffer, operation);
if (!read.ok()) handle(read.status);
```

单 buffer 的无 options 腿不分配、只做一次 native submission；面向不可信网络
对端时，把 absolute monotonic deadline 和/或 `CancelToken` 放进同一个
`OperationOptions`。TCP read/write/connect/accept 与 UDP recv/send 都使用该
形状。`TransferResult` 同时保留 partial bytes、EOF 和 typed `Status`，用户不需要
解释 IOCP/POSIX 负错误码。

连接主机名使用一个总预算和有界错峰尝试，而不是让应用手写 resolver 循环：

```cpp
hyle::io::TcpConnectOptions connect_options;
connect_options.operation.deadline =
    hyle::time::deadline_after(std::chrono::seconds(5));
connect_options.max_parallel_attempts = 2;
auto connected = co_await hyle::io::connect_tcp(
    alloc, io, owner, "model-host.example", 443, connect_options);
```

主机名和 endpoint sequence 都在函数返回前复制。resolver 保留 OS 返回的完整
IPv4/IPv6 列表，按 family 交错并默认每 250 ms 启动下一次尝试；快速失败会立即
推进下一地址。成功或总预算结束前会关闭并排空全部 loser。portable
`getaddrinfo` 进入 OS 后不能强停，因此 deadline/cancel 可以让调用方立即返回，
后台状态仍会安全完成并自行释放。

长连接可用 `set_nodelay()`，以及 `TcpKeepaliveOptions` 设置 idle、interval 和
probe count；不填写的字段保留 OS 默认值。关闭 keepalive 时同时携带 tuning
字段属于错误，避免“配置被静默忽略”。

buffer 和取消 token 必须活到 await 完成。不要在操作仍挂起时强行销毁 fiber；
shutdown 应先停止 admission/cancel，再排空 completion。

## 3. 文件：默认不会堵 scheduler worker

```cpp
hyle::io::FileOpenOptions options;
options.access = hyle::io::FileAccess::kRead;
options.cache = hyle::io::FileCacheMode::kBuffered;
auto opened = co_await hyle::io::open_file(io, owner, path, options);
if (!opened) co_return;
auto file = std::move(*opened);
auto read = co_await file.read_some_at(offset, buffer, operation);
auto closed = co_await file.close();
```

`File` 自动选择：

- io_uring 的 open/read/write/fsync/fallocate/close 走原生异步；
- IOCP read/write 走 OVERLAPPED；
- stat，以及 IOCP/epoll/kqueue 没有原生异步形式的文件操作，自动卸载到
  `IoContext` 自带的 blocking pool。

因此普通 C++ 用户无需先判断 backend。`File` 析构只作为同步防泄漏后备；
正常路径应 `co_await close()`。模型文件要请求真实 direct/non-buffered IO 时显式
选择 `FileCacheMode::kDirect`，随后用 `direct_alignment().accepts(...)` 校验 buffer、
offset 和 length；平台无法兑现时 open 会诚实失败，不静默降为 buffered。

命名共享 storage 属于 Mem：使用 `mem::SharedMapping::create/open/unlink`；IO 只负责建立在
它之上的 namespace/handle transfer、notification 或异步协议。跨进程数据结构只能保存
offset/POD，不能保存本进程裸指针。详见 [Mem 指南](mem.md)。

## 4. HTTP 与 HTTPS

HTTP/1.1 server/client 都接受 plain `TcpStream`。启用 OpenSSL 时也接受已经
完成握手的 `TlsStream`，同一解析与 framing 核直接运行在 TLS transport 上：

```cpp
hyle::io::TlsClientOptions tls_options;
tls_options.alpn_protocols = hyle::io::tls_http2_protocols();
auto context = hyle::io::TlsContext::make_client(tls_options);
if (!context) co_return;

auto encrypted = hyle::io::TlsStream::connect(
    std::move(tcp), *context, {.server_name = "api.example.com"});
if (!encrypted) co_return;
auto tls = std::move(*encrypted);

hyle::io::OperationOptions budget;
budget.deadline =
    hyle::time::deadline_after(std::chrono::seconds(10));
if (!(co_await hyle::io::tls_handshake(alloc, tls, budget)).ok()) co_return;

hyle::io::HttpClientConn client{alloc, std::move(tls)};
auto* response = co_await client.request(
    {"GET", 3}, {"/health", 7}, {"api.example.com", 15});
```

client 默认验证系统信任链；verifying client 必须给 `server_name`，它同时用于
SNI 与证书 DNS/IP identity 校验。自签测试或明确的机会式加密必须显式设置
`TlsVerifyMode::kNone`，不再通过 OpenSSL native pointer 偷改验证。server 用
`TlsServerOptions` 提供证书链、私钥以及可选 mTLS CA；mTLS client 同时填写
`TlsClientOptions::certificate_chain_file/private_key_file`，只给其中一项会在
factory 返回 `InvalidArgument`。路径只在 factory 调用期间借用，provider/config
与编码后的通用 ALPN list 由 `TlsContext` 持有。

每条 stream 在构造时从 context 的 `connection_allocator` 取得一个可复用的 16 KiB
pump buffer；默认使用 `mem::system_allocator()`，record 热路不会反复分配。显式注入
allocator 时，其 backing 必须活过该 context 产生的全部 stream；分配失败由
`TlsStream::connect/accept` 返回 `ResourceExhausted`，不会产生半有效对象。传给
`tls_handshake/read/write/shutdown` 的首个 allocator 只负责对应 coroutine frame。

`tls_handshake/read/write/shutdown` 都接受同一个绝对 `OperationOptions`；一次
handshake 或 record pump 内不会重置 deadline。read/write 返回
`TransferResult`，因此 partial bytes、EOF、cancel/timeout 与协议失败不会再被
一个负整数混在一起。默认严格要求 peer `close_notify`：裸 TCP EOF 返回
`DataLoss`；只有已知会遇到不规范 peer 时才显式设置
`allow_truncated_eof=true`。默认 shutdown 只保证发送本端 close-notify；要求双向
关闭时选择 `TlsShutdownMode::kBidirectional` 并给出 deadline/cancel，避免静默
peer 永久占住 fiber。

客户端完成握手后可用 `take_session()` 取得 move-only opaque ticket，并在同一
`TlsContext` 的下一次 `TlsStream::connect` 中传给 `resume_session`。来自另一
context 的 ticket 会返回 `InvalidArgument`；`session_reused()` 可验证 provider
是否接受了复用。client/server 的 session cache 与 ticket 默认关闭；确实复用连接
握手时，两端都显式设置 `enable_session_resumption=true`。不要持有或转换 OpenSSL
类型，公开 API 不暴露 provider pointer。
TLS 1.3 ticket 可能在握手完成后的 post-handshake record 中才到达；若
`take_session()` 尚无可复用 ticket，它会返回 `FailedPrecondition`，调用方应在完成
一次正常读写后再取，而不是轮询 provider 状态。

`TlsStream` 是 single-owner transport；不得同时从多个 fiber 调用其操作。任一操作
返回非 OK 后，尤其是 deadline/cancel 或底层 partial write 后，关闭该 stream 并按
应用边界重连，不要把同一 plaintext buffer 从头盲重试。

`HttpRequest`/`HttpResponseView` 的 string view 只活到同一连接的下一次 request。
流式响应用 `HttpResponse::set_stream`；`add_header` 返回 false 表示固定 header
槽已满。HTTP 会拒绝 TE+CL、冲突 Content-Length、非 bare chunked 和 CRLF/token
注入。

当前 HTTP convenience 尚未把整个 request 生命周期绑定到一个统一 deadline /
CancelToken；要求强制有界退出的服务应在连接层使用 cancellable/timed I/O，
并在 shutdown 阶段停止 accept 后排空连接。该项仍是 IO 的产品尾项。

HTTP/2 使用 `Http2ClientConn` / `Http2ServerConn`。它支持同连接多 stream、
逆序完成、HPACK、RST/GOAWAY 与流控；TLS context 应把
`tls_http2_protocols()` 交给通用 `alpn_protocols` 并完成握手，构造函数会拒绝
ALPN 不是 `h2` 的 stream。当前 body 是最多 8 MiB 的完整 fixed buffer，不适合
无限 token stream：

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

- `resolve(io, allocator, host, port).join(operation)` 把 `getaddrinfo` 卸载到
  `IoContext` 的 blocking pool，复制 host、返回 owning 的完整 IPv4/IPv6 列表，
  并接受与 transport 相同的 deadline/cancel 合同；数字地址直接用
  `Endpoint::parse()`。
- `SignalSet` 是进程级单 owner、单 waiter 的优雅关闭入口。用
  `SignalSet::create()` 显式选择 `SignalInterest`，失败返回 typed `Status`；
  `co_await wait()` 返回 `Result<ProcessSignal>`，不需要解释 signal number 或负数。
  同类信号会合并，`Interrupt` 与 `Terminate` 两类 pending 事件不会互相覆盖。
  对象必须活到当前 `wait()` 完成，重复的顺序等待受支持；需要把关闭事件广播给
  多个任务时，在唯一 waiter 之后使用 Sched。析构会恢复嵌入宿主原有的 handler。
- Linux 上可以先用 `io.io_uring(owner)` 取得 typed extension；空 handle 表示运行时
  选择了 epoll。当前公开 expert route 只承诺 provided-buffer multishot receive：

  ```cpp
  auto uring = io.io_uring(owner);
  if (uring) {
    auto configured = uring.configure_receive_ring(
        {.group_id = 7, .entries = 256, .buffer_size = 16 * 1024});
    if (configured.ok()) {
      auto receives = stream.read_multishot(uring, 7, allocator);
      if (receives) {
        auto receive_stream = std::move(*receives);
        auto next = co_await receive_stream.next();
      }
    }
  }
  ```

  每个返回的 `IoUringReceiveBuffer` 都是 RAII 借用；析构/移动会把 buffer 归还
  ring，长期持有会以减少可用 buffer 的方式自然施加 backpressure。其他平台没有
  恒失败的同名 stub，普通跨平台路线始终是循环 `read_some()`。fixed-file 与
  direct-accept 仍是内部 backend 机制，不是用户可依赖的公开操作；capability
  snapshot 只用于诊断和选择已公开的 typed route。不能把“不支持”当成一个极快
  benchmark 样本。
