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
- `Http2BodyReader::consume()` is the allocation-bounded shape for a synchronous borrowed sink. It drives one body in one
  task, invokes the sink once per bounded `HttpBodyChunkView`, and acknowledges only after that callback returns. The sink
  may neither retain the view nor suspend/re-enter the session. `read()` remains the natural coroutine pull shape; this is
  not a whole-body buffer or benchmark-only entry point.
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

`AUDITED / ACTIVE`。`18d6d6b9` exact candidate 已完成 qualification；multiplex workload 为 `WIN`，但
`flow-control-holdout-4x256k` 在两种执行顺序分别只有约 `0.794` / `0.776` peer throughput，正式 P1
为 `FAIL`。控制项均通过，故不得把约 20--22% 固定差距解释为仪器噪声或签发 freeze。

机制诊断已用完并形成有限结论：256 KiB socket buffer 同时提高双方绝对吞吐、没有缩小比例；syscall
跟踪显示 HYLE 并未发送更多 wire calls；计数 allocator 则证明逐 chunk 的嵌套 `Task` 令 holdout 达到
约 554 次 allocation/stream。允许且只允许以下 consumer-proven correction：利用既有 stackful owner
fiber 去掉私有 transport-driver 嵌套 task，并增加上述单 task、borrowed synchronous sink 的
`consume()` 形状，使 HYLE 与 nghttp2/curl callback consumer 在 body ownership 上对等。不得调整窗口、
chunk、样本或 margin 来迎合结果；correctness/delivery 全通过并形成新 immutable candidate 后，才可执行
一次新的正式 P1。

该有限 correction 的工作树实现已经完成：raw TCP awaitable 直接借当前 owner fiber 挂起/恢复；一次
transport read 的 nghttp2 callback batch 只发布一次普通 DATA/HEADERS 状态变化；进入 proactor wait 前先
尝试现成 socket bytes，并只给同 owner runnable peer 一次 bounded cooperative yield，第二次 miss 必须
进入真实 kernel wait，不能 spin。TLS 没有 immediate socket seam，仍走原有单 task memory-BIO pump。
对照端的 stream lookup 也已从 benchmark 自建线性查找改为 nghttp2 stream user-data 的 O(1) 同合同形状。

尚未签作正式 P1 的方向诊断显示：`64x1KiB` multiplex 的 HYLE throughput 高约 41--51%；
`4x256KiB` flow-control holdout 在两种顺序下分别低约 3.9% / 2.4%，已进入预声明 5% noninferiority
margin。它只证明 correction 值得形成新 candidate，不替代 clean revision 上的一次正式测量。
Windows/Linux plain correctness 已重复通过；Windows/Linux TLS interop、ASan、UBSan 与 TSan 均通过。
WSL TSan 默认地址布局会在 main 前报 `unexpected memory mapping`，以 `setarch $(uname -m) -R` 运行
同一 instrumented binary 后主套件和强制 epoll 双工用例均通过。source/static/shared、macOS、guide/
red-team 与正式 P1 receipt 仍是 terminal gate，故本卡保持 `AUDITED`。

`consume()` 的 sink error 已定义为 stream-terminal：返回调用方原 `Status`，同时发送 CANCEL、归还
connection credit 并令 reader 失效。否则后续 reader operation 会确认并跳过被 sink 拒绝的 borrowed
chunk，形成虚假的 retry 形状。隔离用例同时验证该失败不会毒化后续 stream。测试 listener 改用 OS
分配端口，避免 Linux ephemeral `TIME_WAIT` 与固定测试端口相撞被误判为协议失败。

## Implementation checkpoints

- `ea45d866`：`IOH2-F01` 的物理 owner 已建立；HTTP/2 source/header/test、nghttp2 package owner
  与 `HYLE_IO_HTTP2_API` 均属于 `io_http2`。feature OFF 只保留 capability query；Windows
  OFF/ON shared correctness、跨 DLL transport link 与独立 header 已通过。
- `18d6d6b9`：首个 exact product candidate 的 W1-WSL P1 qualification/control 全通过；normal
  multiplex 为 `WIN`，大 body flow-control holdout 为固定 `LOSE`。该证据是上述有限 correction 的
  trigger，不是可挑选或合并的历史样本。
