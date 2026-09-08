# IO HTTP/1 v1：有界 streaming client/server 因子化与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`AUDITED`）

- component / owner / round ID：`io_http` / IO maintainers / `io-http-v1`；task kind：`NEW`；
  baseline commit：`e1499bb404e1a69881fb1c4d91c60aa183d1d1a7`；
- demand：`IOHTTP-OWNERSHIP-01`、`IOHTTP-STREAM-02`、`IOHTTP-BUDGET-03`。现有 HTTP/1
  framing 与 picohttpparser 错归 `io_core`；request/response 以裸指针、bool、固定 64-header/8 MiB
  body 和 callback ctx 暴露，whole-request deadline/cancel/disconnect/backpressure 不完整；
- normal journey：推理服务在 plain TCP 或已握手 TLS 上接收 keep-alive REST 请求、限制 head/body、
  以 chunked/SSE 增量发送 token，并在慢客户端、断连、shutdown 与取消时停止生产和释放连接；client
  支持普通请求、streaming response 与连接复用；
- scope：HTTP/1.1 framing/parser integration、owning connection、borrowed message views、bounded
  head/body reader、streaming body reader/writer、client/server lifecycle、keep-alive、expect/continue、
  trailers、typed protocol/system failure、统一 request budget 与 backpressure；
- non-goals：router/middleware/auth/CORS/OpenAI schema、application handler pool/batching、WebSocket、
  HTTP/2/3、TLS provider、multipart convenience、cookie jar、Native FFI/C API；
- dependencies / consumers：候选 `PUBLIC io_core,error,mem,container_text,time,sched`，对 TLS 只允许
  一个经审计的 transport interop edge；picohttpparser 是 private parser mechanism；`io_transfer` 是直接消费者；
- tier：`P1`。normal 1 KiB request + bounded streaming response 与 holdout 64 KiB fragmented message；
  same-contract margin 5%。常态无每-chunk hidden heap allocation，输入 buffer 有界，慢消费者通过
  awaitable backpressure 而非无界排队；
- budget：总 8 engineering hours；审计/API/correctness 5 小时，P1 3 小时且至多两项机制诊断；
- machines / instruments：`W1-WIN/W1-WSL/M1-MAC`，`closure-correctness-v1` +
  `closure-perf-v1`；
- strong peers：Boost.Beast HTTP/1（本地 Boost）与 Seastar HTTP server/client
  `78f64e2317b19c7e34ba61f5e8ea85f083eca1da`；picohttpparser 只作 parser provider；
- provenance：llama.cpp REST/SSE、ds4 HTTP/SSE、vLLM streaming/disconnect、SGLang gateway
  pooling/timeout 是 north-star journeys；不复制其 route 或模型 schema；
- API freeze 前必须明确：connection 单 owner、view invalidation、body ownership、限额/OOM、
  request/response streaming、cancel/disconnect、keep-alive drain、handler continuation 与 teardown；
- gates：parser differential/fragmentation/smuggling/head-body limits/partial write/slow consumer/
  disconnect/cancel/close、plain+TLS route、三交付、P1、guide 与红队；
- stop：不在本卡增加应用 server framework；若 transport abstraction 会迫使 `io_core` 依赖高层或
  为可选 TLS 让普通 HTTP 付费，退回审计而不是复制两套 HTTP 实现。

## Completed audit inputs

- 反向追踪 `http.h/http.cc/http*_test.cc` 与 pico package owner；
- 建立 head、fixed/bounded body、streaming body、client/server、connection lifecycle 能力图；
- 比较四组推理真实旅程与 Beast/Seastar，先冻结 natural C++ journey 和负空间；
- 冻结有限 FIX 集后才允许迁移源码或改变公开 API。

## Capability-family coverage map（`AUDITED`）

| family | admitted journey / lifecycle | baseline | coverage / action |
|---|---|---|---|
| message head/view | request/status line、重复 header、lookup、view invalidation | fixed raw arrays + pointer-return | `PARTIAL / FIX` |
| framing security | CL/chunked/EOF framing、smuggling rejection、fragmentation、limits | pico integration mostly present | `PARTIAL / FIX`（typed errors + configurable limits） |
| request body input | buffered small JSON and bounded incremental large body | whole body only, 8 MiB constant | `MISSING / FIX` streaming route |
| response body output | fixed body and producer-paced chunk/SSE | synchronous raw callback; cannot await token producer | `PARTIAL / FIX` async body source/backpressure |
| server connection | keep-alive loop、one request at a time、drain/close | pointer/bool, no whole-request budget | `PARTIAL / FIX` |
| client connection | headers + fixed/streaming response、reuse、disconnect | one fixed-body convenience only | `PARTIAL / FIX` |
| TLS interop | already-handshaken stream without duplicating framing | compile-time variant embedded in union | `PARTIAL / FIX` transport seam |
| application server | listener accept、routing、middleware、auth、batching | thin callback loop mixes policy | `OUT_OF_SCOPE / REMOVE OR ROUTE` |
| WebSocket/multipart/cookies/proxy | separate protocol/application conveniences | absent | `GENERAL / DEFER` with named owner trigger |

llama.cpp, ds4, vLLM and SGLang all make bounded request parsing, streaming responses, disconnect and shutdown
first-class. Seastar and Beast both separate parser/body stream from routing. Therefore buffered-only request/response
and synchronous chunk callbacks are not complete routes even though historical tests pass.

### Public/source reverse trace and owner decision

- `HttpRequest*`/`HttpResponseView*` are borrowed but failure and EOF collapse into null; `send_response` bool collapses
  protocol/system/cancel; replace with value views inside typed read results and `Status`;
- response fixed header arrays and `add_header(bool)` are benchmark-shaped capacity exposure. The normal send route accepts
  a borrowed span/range of `HttpHeader`; an optional builder may own small storage but cannot be the only route;
- `NextChunkFn` is synchronous, so a token producer must busy-poll or buffer elsewhere. Replace with an awaitable source
  contract and preserve one-in-flight chunk backpressure; no per-chunk queue or hidden allocation;
- `http_serve` embeds handler invocation/request arena policy. The low-level connection loop is sufficient foundation;
  remove the policy wrapper unless a separate generic composed-operation sketch survives red team;
- semantic owner is `io_http`; picohttpparser owns syntax scanning only; io_core/io_tls own byte transports; application
  owns route, authentication, request scheduling and model response production.

## Frozen finite FIX set（`AUDITED`）

| ID | exact work | required gate |
|---|---|---|
| `IOHTTP-F01` | 建立 `io_http` target/header/source/test owner、独立 export macro 与 pico package owner；从 io_core 移除协议 | exact factor、source/static/shared、dependency/license/header scans |
| `IOHTTP-F02` | 建立 typed limits/error/message head/header span 和 value view；移除 null/bool/fixed public slots | parser fragmentation/smuggling/duplicate headers/limits/OOM/error-domain |
| `IOHTTP-F03` | 增加 request/response `HttpBodyReader` incremental route，small-body convenience 只是有界组合 | content-length/chunked/EOF/trailers/partial/cancel/disconnect |
| `IOHTTP-F04` | 增加 awaitable `HttpBodySource`/writer 与逐 chunk backpressure；固定 body 共用同一 send pipeline | delayed producer、slow client、zero/final chunk、no hidden queue/allocation |
| `IOHTTP-F05` | server/client connection 的 operation budget、keep-alive/drain/close 与 plain/TLS transport interop；删除 policy-shaped `http_serve` | total deadline/cancel、late completion、pipelining negative space、TLS optionality |
| `IOHTTP-F06` | guide/architecture、三平台 correctness/sanitizer、三交付、normal REST+SSE P1 与红队 | no apple-vs-orange、terminal receipt |

不保留旧 pointer/bool/fixed-array/callback API。Routing、middleware、auth、model schema、handler pool、
WebSocket 与 HTTP/2 不得追加进本轮。

## API / usability freeze draft

```cpp
auto server_result = HttpServerConnection::create(
    alloc, HttpTransport{std::move(stream)}, limits);
if (!server_result) co_return server_result.error();
auto server = std::move(*server_result);

auto incoming_result = co_await server.read_request(operation);
if (!incoming_result) co_return incoming_result.error();
if (!*incoming_result) co_return Status{};       // clean peer EOF
HttpIncomingRequest incoming = std::move(**incoming_result);
HttpRequestView request = incoming.head;         // borrowed until body retires
for (;;) {
  auto chunk = co_await incoming.body.read(0, operation);
  if (!chunk) co_return chunk.error();
  consume(chunk->bytes);
  if (chunk->end_stream) break;
}
Status sent = co_await server.send_response(head, fixed_body, operation);

auto client_result = HttpClientConnection::create(
    alloc, HttpTransport{std::move(client_stream)}, limits);
if (!client_result) co_return client_result.error();
auto client = std::move(*client_result);
auto response = co_await client.request(head, request_body, operation);
if (!response) co_return response.error();
for (;;) {
  auto chunk = co_await response->body.read(0, operation);
  if (!chunk) co_return chunk.error();
  consume(chunk->bytes);
  if (chunk->end_stream) break;
}
```

This API/algorithm shape is frozen for the immutable candidate. `HttpBodyReader` and
`HttpBodyWriter` are move-only, allow one outstanding operation, and borrow chunk bytes
until the next body operation. `HttpBodySource` is a non-owning, type-erased awaitable
producer; its callable and coroutine state must outlive `send_response()`/`request()`.
Trailers become visible only after end-of-body. P1 may start only after this shape has an
immutable candidate commit and the same-contract peer fixture is fixed.

## Owner 七问（AUDITED）

1. Third-party: picohttpparser is direct vendored mechanism and must move to io_http; inference engines/Beast/Seastar
   are references only.
2. Capability: seven admitted families have material gaps captured by F02--F05; application server/WebSocket remain
   explicit negative space.
3. Reuse: io_core transport/result/options, io_tls encrypted transport, Error, Time, Mem Region and Container text are
   candidates. Parser integration remains local because peers with event loops are not same-contract reusable runtimes.
4. Performance: P1 normal REST + SSE; historical raw/pico or sync callback microbenches do not gate. Compare same parsing,
   body ownership and backpressure or use natural-route budget without primitive SOTA claims.
5. FFI: `N/A`, no C schema/codec/exports.
6. Delivery: F01 ownership landed in `ea45d866`; source/static/shared/offline terminal journeys remain pending.
7. C++ usability/docs: currently `FAIL` due null/bool, fixed slots, sync callback and missing body/cancel selection guide;
   F02--F06 repair it.

## Current decision

`AUDITED / ACTIVE`。F01--F05 的实现、API/usability 与算法形状已形成候选并冻结，不再追加能力或
修改常态 metadata；F06 尚待 immutable candidate commit、三平台/三交付/sanitizer、同合同 P1、
guide/red-team terminal receipt。候选提交形成前禁止启动 P1。

## Implementation checkpoints

- `ea45d866`：`IOHTTP-F01` 的物理 owner 已建立；HTTP/1 source/header/test/benchmark、pico
  license/package owner 与 `HYLE_IO_HTTP_API` 均属于 `io_http`。HTTP/2 只经导出的 transport
  seam 复用本 factor；Windows shared correctness 与独立 header 已通过。
- 2026-09-08 candidate preparation：F02--F05 已实现。公开形状改为 typed
  `Result<optional<HttpIncomingRequest>>`/`Status`、move-only incremental body reader/writer、
  awaitable source 与 owning client/server connection；动态 limits、CL/TE smuggling rejection、
  fixed/chunked/EOF framing、trailers、keep-alive、Expect/Continue bounded fallback、whole-operation
  deadline/cancel、慢消费者 backpressure、断连停产、buffer high-water release 和幂等立即 close
  均有 Windows 回归覆盖。plain HTTP 3/3、TLS interop 1/1 通过，HTTP/2 transport seam 编译通过；
  smoke benchmark 只作分解诊断，不构成 P1 或 peer 结论。
