# IO 使用指南

> IO 当前只提供 C++ 产品面；六个内部
> factor 的终态以 `module-status.json` 为准。本页适用于 C++23 source/static/shared 三种交付，
> 不描述 C API 或跨语言调用层。

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

### 文件变化：观察后重读 authoritative state

证书、配置、model sidecar 或 shader 由另一个进程以 write/rename/atomic replace 发布时，使用
`FileWatcher`；不要轮询 mtime，也不要依赖“一次 write 必有一个 event”。watcher 只提供通用、
non-recursive 的变化流，解析证书、验证配置以及发布新对象仍由调用方或计划中的 `hyle-service` 拥有：

```cpp
#include <hyle/io/watch.h>

std::array targets{
    hyle::io::FileWatchTarget::file("tls/server.crt"),
    hyle::io::FileWatchTarget::file("tls/server.key"),
    hyle::io::FileWatchTarget::file("tls/ca.pem")};
auto made = hyle::io::FileWatcher::create(alloc, io, owner, targets);
if (!made) co_return;
auto watcher = std::move(*made);

while (!shutdown.is_cancelled()) {
  auto batch = co_await watcher.next(
      {.deadline = std::nullopt, .cancel = &shutdown});
  if (!batch) break;

  // Event 是失效通知，不是完整日志。先同时重读并验证整组文件，
  // 只有全部成功才替换当前 TLS context；失败时继续服务旧 context。
  if (batch->rescan_required() || !batch->empty())
    try_publish_reloaded_tls_context();
}
(void)watcher.close();  // 必须早于 IoContext/Scheduler shutdown
```

一个 exact-file target 可以在 `create()` 时尚不存在，但其 parent 必须存在；directory target 本身必须是
已有目录。输入 path 在 factory 返回前被复制；传入的 allocator 则是借用的资源 owner，必须活过 watcher
及其 `close()`。exact event 用稳定 `target_index` 标识，
`relative_path` 为空；directory event 的 path 只是一层 basename，深层 descendant 不会泄漏。
POSIX 根目录用 `realpath` 归一到现有目录身份；Windows 根目录使用绝对、分隔符归一的 lexical path，两个
指向同一目录的 Win32 alias 可能各自占用一个 native handle。子项按 native directory stream 返回的
UTF-8 拼写做 byte-exact 匹配。不要依赖大小写折叠、Unicode normalization 或 symlink alias 在三个平台上等价；对
exact-file target 使用目录实际保存/枚举的名称，或者观察其 parent directory 后重读 authoritative state。

返回的 `FileWatchBatchView` 借用 watcher 的双缓冲，到下一次 `next()`、`close()` 或析构即失效；需要跨
await 保存时复制。event 可能合并多个 change bit，rename correlation 只是尽力提供。任何 native/library
overflow、粗粒度通知或无法安全表示的名称都会设置 `rescan_required()`；此时枚举并重建 authoritative
state，不能猜丢了哪些事件。应用 debounce、内容 hash/diff、证书验证、reload retry 和 watcher service
不是 IO 的职责。

为避免逐事件 allocation、复制和跨 worker dispatch，`FileWatcher` 有意采用以下性能/生命周期限制：它是
single-owner 对象，`create/next/close` 以及 live-owner 的 move assignment 都在指定
Scheduler worker 上进行；只能有一个在途 `next()`。取消或 deadline 只结束该次等待，watcher 仍可复用；
`close()` 幂等，会唤醒 waiter 并排空 native completion。默认限制为 16 KiB native buffer、256 events
和 16 KiB path bytes，可按一批资源数量显式调整；容量耗尽不会静默丢失，而是请求 rescan。
从构造 `next()` 返回的 awaitable 到其 `await_resume()` 完成，awaitable 借用的是同一个 watcher 对象；
这段时间不得移动、赋值或析构 watcher。owner worker 上的另一 fiber 可以调用 `close()`，它会取消并
排空该等待，随后 awaitable 以取消状态恢复；watcher 的析构也属于 owner-worker 操作，不能把未等待的
awaitable 带出 watcher 的生命周期。

## 4. HTTP 与 HTTPS

HTTP/1.1 server/client 都接受 plain `TcpStream`。启用 OpenSSL 时也接受已经
完成握手的 `TlsStream`，同一解析与 framing 核直接运行在 TLS transport 上：

```cpp
hyle::io::TlsClientOptions tls_options;
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

auto made = hyle::io::HttpClientConnection::create(
    alloc, std::move(tls), hyle::io::HttpLimits{});
if (!made) co_return;
auto client = std::move(*made);

hyle::io::HttpRequestHead request;
request.method = "GET";
request.target = "/health";
request.authority = "api.example.com";
auto response = co_await client.request(
    request, std::span<const std::byte>{}, budget);
if (!response) co_return;
auto body = co_await response->body.read_all(64 * 1024, budget);
```

client 默认验证系统信任链；verifying client 必须给 `server_name`，它同时用于
SNI 与证书 DNS/IP identity 校验。自签测试或明确的机会式加密必须显式设置
`TlsVerifyMode::kNone`，不再通过 OpenSSL native pointer 偷改验证。server 用
`TlsServerOptions` 提供证书链、私钥以及可选 mTLS CA；mTLS client 同时填写
`TlsClientOptions::certificate_chain_file/private_key_file`，只给其中一项会在
factory 返回 `InvalidArgument`。路径只在 factory 调用期间借用，provider/config
与编码后的通用 ALPN list 由 `TlsContext` 持有。

当前 context 只能加载未加密的 PEM private-key file；没有 password/callback，也没有公开 provider native
handle 可安全绕行。需要 encrypted key 的部署当前应由外部 TLS terminator 承担，或等待已登记的 IOTLS
能力 trigger，不能把加载失败误诊为证书路径问题。自定义 cipher suite、内存证书和热 reload 同样不是
当前合同；安全默认以外的 provider policy 见
[观察清单](../architecture/inference-engine-capability-watchlist.md)。

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

`tls_write` 是 write-some：一次成功最多提交一个 provider record，并可能返回小于
输入 span 的 partial progress。发送完整 buffer 时像 TCP 一样按返回的 `bytes` 推进，
或使用 HTTP 等已经提供 write-all 组合语义的上层；不要假设一次调用必定耗尽输入。

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
应用边界重连，不要把同一 plaintext buffer 从头盲重试。HTTP/2 等流控协议造成底层
write backpressure 时，TLS 内部会有界地吸收已经到达的 peer ciphertext 并继续原操作；
单 stream 的这部分未处理 ciphertext 有 32 KiB 硬上限，
用户不需要、也不得另起并发 read/write 来手工驱动 memory-BIO pump。

HTTP/1 是连接级协议因子，不是应用 server framework。server 在 accept 后创建
`HttpServerConnection`，由自己的 route/handler/task owner 驱动循环：

```cpp
auto made = hyle::io::HttpServerConnection::create(
    alloc, std::move(accepted), limits);
if (!made) co_return;
auto connection = std::move(*made);

for (;;) {
  auto next = co_await connection.read_request(operation);
  if (!next || !next->has_value()) break;  // typed failure / clean EOF
  auto incoming = std::move(next->value());
  auto json = co_await incoming.body.read_all(256 * 1024, operation);
  if (!json) break;

  hyle::io::HttpResponseHead head;
  static constexpr hyle::io::HttpHeader headers[] = {
      {"Content-Type", "application/json"}};
  head.headers = headers;
  if (!(co_await connection.send_response(
            head, std::as_bytes(std::span("{}", 2)), operation)).ok()) {
    break;
  }
}
(void)co_await connection.close(operation);
```

head、header、body chunk 都是 connection-owned borrowed view，connection 必须活过这些
borrow 以及所有仍在途的 reader/writer operation。head/header 在同一连接的下一次消息操作
失效；body chunk 在该 reader 的下一次操作失效。移动 connection 不会移动其内部状态，但销毁或
`close()` 会让现有 reader/writer 失效；不要把未完成操作跨过 connection 的析构。
小 JSON 使用显式上限的 `read_all(max_bytes)`；大 request/response 循环
`body.read()`。要复用 keep-alive 连接，必须把 body 读完或调用 `discard()`；丢弃
未完成 reader 会把连接标为不可复用，但 server 仍可发送一个强制 close 的最终
响应。一个 connection 固定属于 transport 的 owner fiber，同一 reader/writer
只允许一个在途操作。

流式 token/SSE 用 `HttpBodySource::from(source)`。HTTP 在取得一块后先完整 await
transport 写入，再请求下一块，不建立隐藏队列，因此慢客户端会自然把压力传回
producer。`Source::next(AllocatorRef, OperationOptions)` 必须 `noexcept` 并返回
`Task<Result<HttpBodyChunk>>`；由于 HYLE coroutine frame 的 allocator-first 约定，
成员入口通常只是普通转发函数，真正的 coroutine 写成 allocator 为首参的 static
helper。source 和它借出的字节必须活到 `send_response` await 完成。

`HttpLimits` 同时限制 head/body/header 数、每次 body read，并限制异常大 head 与
`read_all` 聚合缓冲在下一消息后的保留量。默认 `Expect: 100-continue` 最多等待
1 秒，然后按 RFC 允许的行为继续发 body；可用 `expect_continue_wait` 调整，整体
更早的 `OperationOptions::deadline` 仍然优先。early 413/417 会作为正常 response
返回且不会发送 body。HTTP 拒绝 TE+CL、冲突 Content-Length、重复 Host、非 bare
chunked、obsolete folding 和 CRLF/token 注入；trailers 仅能随 chunked 的最终块
发送，调用方还应在 head 中声明相应 `Trailer` 字段。

HTTP/1 因子不提供协议接管：`CONNECT` tunnel、`101 Upgrade`/WebSocket 和取得底层
transport 都不在当前合同中。不要用通用 method/header 字段发起这些操作；若 peer 返回 `101`，
只能观察 response head，连接会变为不可复用并应关闭。需要 tunnel 或 upgraded protocol 时应先为
独立因子冻结 ownership、残留已读字节和 transport hand-off 合同，不能从当前 connection 私自取句柄。

所有 composed operation 复用同一个绝对 deadline/`CancelToken`，不会按 partial
read/write 重置预算。`close()` 幂等，在安全完成 TCP shutdown 或 TLS close-notify
发送后立即释放 transport；析构只作不阻塞的资源后备，不能替代需要 TLS
close-notify 的正常 close。listener accept loop、路由、中间件、认证、模型 schema、
handler admission/batching 与优雅排空策略属于应用 owner，HTTP 因子不偷偷选择。

HTTP/2 需要配置 `HYLE_IO_HTTP2=ON`。h2c 只支持 prior knowledge；HTTPS 则让 client/server
TLS context 都使用 `tls_http2_protocols(false)`、设置 `require_alpn=true`，完成握手后再把
`TlsStream` 移入 session。factory 会拒绝没有协商到 `h2` 的 TLS transport。nghttp2 是 private
framing/HPACK 机制，用户不取得它的 session 或 callback：

```cpp
#include <hyle/io/http2.h>

auto made = hyle::io::Http2ClientSession::create(
    alloc, std::move(tls), hyle::io::Http2Limits{});
if (!made) co_return;
auto session = std::move(*made);

hyle::io::Http2RequestHead head;
head.method = "POST";
head.scheme = "https";
head.authority = "model.example";
head.path = "/v1/infer";
auto id = co_await session.send_request(head, request_bytes, operation);
if (!id) co_return;

auto response = co_await session.read_response(*id, operation);
if (!response) co_return;
auto on_chunk = [&](const hyle::io::HttpBodyChunkView &chunk) noexcept
                    -> hyle::Status {
  return consume_tokens(chunk.bytes); // 不保存 chunk/view，也不重入 session
};
if (!(co_await response->body.consume(on_chunk, 16 * 1024, operation)).ok())
  co_return;
```

`send_request()`/`send_response()` 的 span 重载适合固定小 body；token stream 使用
`start_request()`/`start_response()` 得到 move-only `Http2BodyWriter`，逐块 `write()` 后
`finish()`。一次 write 完整穿过 transport boundary 后才返回，不存在隐藏的无界发送队列。
接收端可循环 `read()`，也可用上面的 `consume()` 让同步 borrowed sink 在一个 task 中处理完整
body；后者避免每 chunk 建立 task。sink 必须是活到 await 完成的 lvalue，`noexcept` 返回
`Status`，不得 suspend、保留 view 或重入同一 session；sink 返回错误会取消该 stream 并令 reader
失效，不能把它当成 chunk retry。只想释放数据与流控额度时使用 `discard()`。

每个 chunk view 在 reader 的下一次操作时失效；该次操作也才确认上一 chunk 并产生相应
WINDOW_UPDATE。丢弃未读完的 reader 会发送 `RST_STREAM(CANCEL)`、归还 connection credit，
不会毒化其他 stream。`next_response()` 按完成顺序取结果，`read_response(id)` 等待指定 stream；
不要把两者同时用于同一个 response。session 及所有 stream handle 固定属于 transport owner
worker，多个 stream 可以在该 worker 上协作推进，但同一 reader/writer 只能有一个在途操作。

`Http2Limits` 是每 session 的 header/body/concurrency/window 上限。一个低于 65,535 bytes 的
stream receive window 只有在 peer 收到 SETTINGS 后才生效；connection window 不能低于协议初值。
所有操作沿用同一个绝对 deadline/`CancelToken`。取消单 stream 用 `reset()`；停止接收新 stream、
等待既有 stream 并发送 GOAWAY 用 `drain()`，最后调用 `close()`。HTTP/1 Upgrade、server push、
gRPC schema/runtime、路由、认证与模型 batching 仍属于本因子之外。

## 5. 可靠复制与模型下载

当需求是“旧模型继续可用，完整的新制品校验成功后才替换它”时，使用
`TransferPlan`，不要在应用中重新拼 redirect、Range、临时文件和 rename：

```cpp
#include <hyle/io/transfer.h>

hyle::io::TransferOptions options;
options.limits.max_bytes = 16ull << 30;
// Optional: the 1 MiB default favors model-sized local and HTTP artifacts.
// Select a smaller block only when concurrent-transfer memory is tighter.
options.limits.buffer_bytes = 1 << 20;

auto plan = hyle::io::TransferPlan::http(
    alloc,
    {.url = "https://models.example/qwen/model.gguf"},
    {.path = "models/qwen.ready"}, options);
if (!plan) co_return;

hyle::io::TransferExecution execution;
execution.owner_worker = owner;
execution.operation.deadline =
    hyle::time::deadline_after(std::chrono::minutes(30));
execution.operation.cancel = &cancel;
execution.verifier = hyle::io::TransferVerifier::from(sha256_verifier);
execution.progress = hyle::io::TransferProgressObserver::from(progress);

auto report = co_await hyle::io::transfer(
    alloc, io, std::move(*plan), execution);
if (!report.published() && report.checkpoint.resumable()) {
  auto retry = hyle::io::TransferPlan::resume(
      std::move(report.checkpoint));
  // 在新的 transfer() attempt 中重新提供 deadline/cancel/callback。
}
```

这里的 `sha256_verifier` 是消费方拥有的 incremental adapter 示例，不是遗漏 include 的 HYLE 类型。可用
OpenSSL EVP SHA-2/3 或官方 BLAKE3 实现 `reset/update/finish`；不要用 Container 的 `hash_bytes` 代替，
后者只承诺进程内 hash-table 使用，不稳定、也不是 cryptographic digest。

本地文件用 `TransferPlan::local`，HTTP 与 HTTPS 用
`TransferPlan::http`。factory 在返回前复制 source、URL、destination 和 options；
factory 的 allocator 为 plan、checkpoint 和 kept-path value 提供存储，必须比这些值活得更久；
传给 `transfer()` 的 coroutine allocator 则必须活到返回的 task 完成。
`TransferExecution` 中的 verifier、observer、cancel token 与可选 TLS context 是 borrow，必须活到 task
完成。task 必须运行在 `owner_worker` 对应的 Scheduler worker。每次 read 完整 await
对应 write 后才继续取下一块，没有隐藏的全文件缓冲或无界队列；内存对制品大小为
O(1)。HTTP 正常路径复用 body chunk，本地复制以及 resume 前缀校验才使用显式 copy
buffer。默认 block 为 1 MiB；每个本地 resume 最多临时持有两块，因此需要大量并发下载且更重视
内存上限时，可显式选择更小的非零 `buffer_bytes`。较小 block 会增加每字节的 completion/syscall
摊销，不应仅为微基准随意修改。

成功路径在目标同目录创建已打开的随机 temporary file，校验完整 byte stream，sync
文件数据与 metadata，close 后才原子替换目标。因此失败、cancel、checksum mismatch、
超限或短读都不会破坏旧目标。`kFileDataAndMetadata` 只承诺文件本身已 sync；当前
Platform 没有可移植 directory-fsync 合同，所以它不声称掉电后目录项一定持久。该
workflow 使用 buffered temporary file：direct IO 的 alignment/尾块语义不适合 HTTP
chunk 与通用原子发布；需要 direct model-file reads 时应在发布后用第 3 节的显式
`FileCacheMode::kDirect`。

默认失败策略返回一个 move-only `TransferCheckpoint`。它拥有准确的 open partial
file、source identity/HTTP validator 和 options：

- `resume(std::move(checkpoint))` 继续原配置；重载可显式修改 limits 与 mismatch policy；
- 本地 resume 先比较 regular-file size/mtime，再把 source 与 retained partial 的
  完整前缀逐字节核对，并在复制结束再次核验元数据；即使时间戳碰巧相同也不会盲目
  append。POSIX 上 `/proc`/`sysfs` 一类报告 size=0 的 regular file 会读到真实 EOF，
  真正的空文件仍正常发布；HTTP 使用 strong ETag，缺少时退到 Last-Modified。server
  忽略 Range、validator/total 改变时默认安全
  truncate 并重启；`kFail` 则保留 partial 并返回 `FailedPrecondition`；
- `416` 只有在 `Content-Range: bytes */total` 与 checkpoint offset、已知 total，以及
  响应中出现的 validator 一致时，才把 partial 当作完整制品继续校验和发布；
- `discard()` 显式关闭并删除；checkpoint 析构也会 best-effort 同步 close/remove，故应
  在 control path 明确处理，不要把大量 checkpoint 的析构堆到 latency-sensitive worker；
- `keep_partial()` 返回 move-only `KeptTransferPartial`。它关闭 partial、解除 HYLE 的
  自动删除，并零额外分配地接管稳定路径存储；`path()`/`c_str()` 在该值存活期间有效。
  值析构后文件仍由调用方负责。它不会生成跨进程 resume manifest，调用方必须自行管理
  该诊断/移交文件。

HTTPS 默认验证系统信任链、只使用 HTTP/1.1 并禁止降级到 HTTP。需要企业 CA 或 mTLS 时，
先用 `<hyle/io/tls.h>` 构造 `TlsContext`，再令
`execution.tls_context = &client_context`；context 必须活到 task 完成，而且不得配置为
协商 `h2`，因为本 workflow 当前只组合已冻结的 HTTP/1 body contract。URL 中的 credentials/userinfo、fragment、
控制字符和非法 authority 会被拒绝。认证 token、代理、Hugging Face/repository layout、
archive、模型 schema 与 digest lookup 都属于上层。redirect 有显式上限，响应必须是
identity encoding；`max_bytes` 计算最终制品总量而非“本次剩余量”。progress/verifier
同步运行在 owner fiber，不得重入同一 transfer；它们返回非 OK `Status` 就按普通失败
策略停止。commit 在 atomic replace 开始后不可取消，避免出现调用方无法判断目标名字
指向哪一版的状态。

Transfer 内部 parser 只覆盖这条 HTTP(S) 下载路线，不是公共 URL 产品。其它 URL parse/resolve 用
Boost.URL 等成熟库；gzip/zstd/Brotli 或 archive 由对应 codec/libarchive 在上层组合。HTTP multipart、
router/middleware/request context 和 graceful listener drain 计划由尚未实现的 `hyle-service` 或其它完整
server framework 承载，不能要求 handler 手写无界 multipart parser，也不能把设计中的 service owner
算成今天已经交付的能力。

## 6. DNS、signal 与 Linux 专属快腿

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
