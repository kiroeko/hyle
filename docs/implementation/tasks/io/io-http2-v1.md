# IO HTTP/2 v1：多 stream 与流控因子化收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`AUDITED`）

- component / owner / round ID：`io_http2` / IO maintainers / `io-http2-v1`；task kind：`NEW`；
  baseline commit：`e1499bb404e1a69881fb1c4d91c60aa183d1d1a7`；
- demand：`IOH2-OWNERSHIP-01`、`IOH2-STREAMING-02`、`IOH2-LIFECYCLE-03`。现有
  nghttp2 session 与 HTTP/2 API 错归 `io_core`；固定 8 MiB whole-body 与 pointer/bool result
  不能自然承载 token stream、RST/GOAWAY/cancel 和 per-stream backpressure；
- normal journey：一个 h2c 或 ALPN `h2` 连接并发多个推理请求，按 stream ID 独立收发增量 body，
  遵守 connection/stream flow-control，在 reset/deadline/disconnect/GOAWAY 下正确退休；
- scope：client/server session、headers、incremental stream body、flow control、per-stream result/cancel、
  connection drain/GOAWAY、h2c prior knowledge 与 TLS ALPN interop；
- non-goals：HTTP/1 Upgrade、server push、gRPC schema/runtime、WebSocket、HTTP/3/QUIC、route/auth/model
  policy、Native FFI/C API；
- dependencies / consumers：候选 `PUBLIC io_core,io_http,error,mem,container,time,sched`；TLS 是
  optional interop edge；nghttp2 static core 是 private mechanism/delivery dependency；`io_transfer` 可消费；
- tier：`P1`。normal 两并发 streams 与 >64 KiB flow-control holdout，same-contract margin 5%；
  stream state 有界、无 whole-body 强制缓冲、每 chunk 不隐藏分配；
- budget：总 8 engineering hours，审计/API/correctness 5 小时，P1 3 小时；最多两项机制诊断；
- machines / instruments：`W1-WIN/W1-WSL/M1-MAC`，`closure-correctness-v1` +
  `closure-perf-v1`；
- strong peers：nghttp2 `85e300c79fb6dbcfa9c1013215c8710c1c2cd3d2` 定义协议机制；
  Boost.Beast 不提供 HTTP/2，故同合同产品 peer 若无法资格化，使用冻结 consumer budget而不伪造 SOTA；
- provenance：SGLang/vLLM gRPC/HTTP streaming 证明 multiplex + flow-control journey；llama.cpp/ds4
  主要为 HTTP/1，作为不强制默认启用 HTTP/2 的留出；
- API freeze 前明确 message view invalidation、stream owner、header/body limits、data consumption/
  WINDOW_UPDATE、cancel/reset、GOAWAY drain、TLS optionality 与 package discovery；
- gates：HPACK/frame differential、multi-stream reorder、large body/flow-control、RST/GOAWAY、invalid
  frame、OOM/limits、cancel/close、h2c+TLS、source/static/shared、offline nghttp2、P1、guide、红队；
- stop：缺可资格化同合同 peer 时只签 consumer budget；不能手写第二个 HPACK/frame state machine，
  不能用全 body buffering 伪装 streaming 完整。

## Completed audit inputs

- 反向追踪 `http2.h/http2.cc/http2_test.cc`、nghttp2 adapter 与 package metadata；
- 用四组推理旅程裁决 incremental body、cancel、GOAWAY、limits 和 gRPC owner；
- 冻结 feature-off/ON target 与 HTTP/TLS interop DAG；
- 冻结有限 FIX 集后才允许迁移或重写产品源码。

## Capability-family coverage map（`AUDITED`）

| family | admitted journey / lifecycle | baseline | coverage / action |
|---|---|---|---|
| session/provider owner | client/server SETTINGS、HPACK、frame/state machine | nghttp2 adapter in `io_core` | `PARTIAL / FIX` ownership |
| header blocks | pseudo fields、repeated headers、limits、informational/final response | fixed arrays + connection-borrowed pointer | `PARTIAL / FIX` |
| stream lifecycle | open/accept/half-close/close、completion order、typed stream identity | completed whole message only | `PARTIAL / FIX` |
| inbound body | bounded incremental DATA consumption + trailers | mandatory whole-body buffer, 8 MiB constant | `MISSING / FIX` |
| outbound body | awaitable producer、END_STREAM、flow-control backpressure | fixed body copied into per-stream buffer | `MISSING / FIX` |
| flow control | consume acknowledgement、connection/stream windows、slow peer | hidden behind whole-body completion | `PARTIAL / FIX` |
| cancel/reset | per-stream cancellation/RST with late frame retirement | bool reset; no operation budget | `PARTIAL / FIX` |
| graceful drain | GOAWAY last-stream/error/debug data、reject new streams、drain | bool connection shutdown only | `PARTIAL / FIX` |
| transport negotiation | h2c prior knowledge and already-negotiated TLS `h2` | constructor variants behind feature macros | `PARTIAL / FIX` seam |
| gRPC / push / upgrade / HTTP3 | application codec or separate protocol | absent | `OUT_OF_SCOPE / DEFER` |

The four inference references admit multiplexed request/response streams, cancellation and bounded flow control, but not a
gRPC schema/runtime in this factor. nghttp2 remains the sole HPACK/frame/state-machine authority. HYLE owns transport,
allocator, error translation and the stream-facing lifecycle; it must not duplicate protocol machinery to win a benchmark.

### Public/source reverse trace and owner decision

- `Http2MessageView*`, bool and positive/negative stream IDs collapse clean EOF, protocol error, cancellation and OOM;
  replace them with typed values/results and an opaque checked `Http2StreamId`;
- `body()` proves that DATA is retained until END_STREAM. The normal route must expose bounded incremental reads and call
  `nghttp2_session_consume_stream()` only when the consumer releases bytes, so WINDOW_UPDATE is real backpressure;
- outbound `StringView body` forces a whole-body copy. A one-in-flight awaitable source must feed nghttp2's deferred data
  provider; a fixed body is merely a convenience over that pipeline;
- session remains single-owner on the transport fiber. Per-stream handles may be move-only logical owners but cannot invoke
  nghttp2 concurrently; they schedule work through the same owner session without an unbounded hidden queue;
- `io_http2` owns HTTP/2 semantics, nghttp2 owns wire machinery, `io_core`/`io_tls` own bytes, Sched owns task/cancel primitives.
  If implementation proves a named missing Sched contract, reopen Sched lawfully; do not clone a scheduler in this factor.

## Frozen finite FIX set（`AUDITED`）

| ID | exact work | required gate |
|---|---|---|
| `IOH2-F01` | 建立 `io_http2` target/header/source/test owner、独立 export macro 与 nghttp2 package owner；feature OFF 不探测 | exact factor、OFF/ON、source/static/shared、offline/license/header scan |
| `IOH2-F02` | typed limits/error/head/header span、checked stream ID 与 value result；移除 pointer/bool/fixed public slots | invalid frame/HPACK/limits/OOM/typed error tests |
| `IOH2-F03` | 增加 per-stream incremental body reader，消费确认直接驱动 WINDOW_UPDATE，trailers 在 EOF 后可见 | fragmented/>64 KiB/slow reader/interleaving/flow-control |
| `IOH2-F04` | 增加 awaitable body source/writer，nghttp2 deferred provider 按 peer window 拉取且每 stream 至多一项在途 chunk | slow peer、zero/final chunk、reset race、无 whole-body/hidden queue |
| `IOH2-F05` | per-stream `OperationOptions`/RST、connection GOAWAY/drain、h2c 与已协商 TLS transport seam | cancel/deadline/late frame/GOAWAY/reject-new/ALPN tests |
| `IOH2-F06` | guide/architecture、三平台 sanitizer/correctness、三交付、multiplex+flow-control P1 与红队 | no apple-vs-orange、terminal receipt |

不保留旧 whole-message、pointer/bool/fixed-array API；不手写 HPACK/frame state machine。gRPC schema、server push、
HTTP/1 Upgrade、WebSocket 与 HTTP/3 不得追加到本轮。

## API / usability freeze（`AUDITED`）

```cpp
auto made = Http2ServerSession::create(alloc, std::move(transport), limits);
Http2ServerSession session = std::move(*made);
auto incoming = co_await session.accept(operation);
Http2StreamId id = (*incoming)->stream_id;
Http2RequestView head = (*incoming)->head;
Http2BodyReader body = std::move((*incoming)->body);

auto started = co_await session.start_response(
    id, Http2ResponseHead{.status = 200}, std::nullopt, operation);
Http2BodyWriter writer = std::move(*started);
co_await writer.write(token_bytes, operation);
co_await writer.finish({}, operation);
co_await session.drain(Http2Error::kNoError, operation);
```

The public shape is frozen as follows before product implementation begins:

- `Http2Limits` is a value contract for dynamic header count/bytes, per-stream body bytes, concurrent streams,
  connection/stream receive windows and ordinary read chunk size. There are no public compile-time HTTP/2 maxima.
- `Http2StreamId` is an opaque positive value. No API accepts or returns a raw signed stream ID as a success/error union.
- regular fields reuse `HttpHeader` / `HttpHeadersView`; `Http2RequestView` owns the HTTP/2 pseudo-field vocabulary and
  `Http2ResponseView` owns `:status`. Repeated regular fields retain order.
- `Http2IncomingRequest` / `Http2IncomingResponse` are values containing a head view, strong stream ID and move-only
  `Http2BodyReader`. Initial HEADERS make the value ready; END_STREAM is not required before delivery.
- `Http2BodyReader::read()` returns bounded `HttpBodyChunkView`; acknowledgement occurs on the next reader operation and
  directly calls `nghttp2_session_consume_stream()`. EOF exposes trailers. Dropping an unfinished reader submits RST and
  releases connection credit.
- `Http2BodyWriter` has one borrowed chunk in flight. `write()` does not return before bytes encoded from that chunk have
  crossed the transport write boundary; `finish()` supplies optional trailers. The nghttp2 provider returns
  `NGHTTP2_ERR_DEFERRED` when the producer has not supplied bytes, and `nghttp2_session_resume_data()` is the only resume
  mechanism. Fixed span and `HttpBodySource` overloads are conveniences over this same path.
- `Http2ServerSession` exposes `create`, `accept`, `start_response` / `send_response`, `reset`, `drain`, `close` and peer
  GOAWAY inspection. `Http2ClientSession` exposes `create`, `start_request` / `send_request`, completion-order
  `next_response`, ID-specific `read_response`, `reset`, `drain`, `close` and peer GOAWAY inspection.
- A session and every stream handle are bound to the transport owner worker. At most one read and one write may be active
  for an individual stream. Multiple stream fibers may operate cooperatively: a bounded internal driver serializes nghttp2
  mutation and at most one transport read/write per direction; it never creates a second scheduler or an unbounded request
  queue.
- Header views remain valid until the next header/session operation documented on that stream; a returned body view remains
  valid until the next operation on its reader. Session destruction invalidates every outstanding handle. Reader/writer
  destruction retires nghttp2 callbacks before allocator-backed stream state can be reclaimed.
- h2c means prior knowledge only. A TLS constructor accepts only an already-handshaken stream whose ALPN is exactly `h2`.
  CONNECT, HTTP/1 Upgrade, server push, WebSocket and gRPC schema/runtime remain named non-goals rather than half-working
  fields on the generic API.

No P1 starts before compile-only multiplexed/streaming journeys and the implementation freeze these contracts. Product
evidence may refine private layout and driver mechanics, but changing the public lifecycle requires returning this card to
`AUDITED` with a named consumer contradiction.

## Owner 七问（AUDITED）

1. Third-party: nghttp2 is the direct private protocol mechanism and build/package owner must move to io_http2; all inference
   engines are consumer evidence, not linked providers.
2. Capability: nine admitted families contain material gaps captured exactly by F02--F05; deferred protocol families have
   named future triggers and cannot block this round.
3. Reuse: io_core transport/options/result, io_http head vocabulary, optional io_tls transport, Error/Time/Mem/Container/Sched
   are reused. No peer event loop or duplicate scheduler is imported.
4. Performance: P1 multiplex + >64 KiB flow-control journeys; compare same body ownership/window semantics or use a frozen
   consumer budget. Whole-body versus streaming and single versus multiplexed are invalid comparisons.
5. FFI: `N/A`; no C schema/exports/managed sample.
6. Delivery: F01 ownership landed in `ea45d866`; source/static/shared/offline terminal journeys remain pending.
7. C++ usability/docs: currently `FAIL` due pointers/bools, mandatory body buffering and absent lifecycle guide; F02--F06 repair it.

## Current decision

`AUDITED / ACTIVE`。能力图、stream 生命周期、owner、API draft 与有限 F01--F06 已冻结；允许实现，
但 API/usability freeze 与 immutable candidate 尚未形成，禁止启动 P1。

## Implementation checkpoints

- `ea45d866`：`IOH2-F01` 的物理 owner 已建立；HTTP/2 source/header/test、nghttp2 package owner
  与 `HYLE_IO_HTTP2_API` 均属于 `io_http2`。feature OFF 只保留 capability query；Windows
  OFF/ON shared correctness、跨 DLL transport link 与独立 header 已通过。
