# IO HTTP/1 v1：有界 streaming client/server 因子化与收口

<!-- MODULE CLOSURE RECORD: CLOSED -->

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

## Owner 七问（terminal candidate）

1. Third-party: picohttpparser is the direct private syntax mechanism owned and packaged by `io_http`; Boost.Beast and
   Nginx are evidence peers, while inference engines and Seastar are reference consumers/architectures only. No peer
   event loop or public provider type leaks into the product.
2. Capability: all seven admitted families are `COMPLETE`; bounded incremental client/server bodies, keep-alive,
   trailers, Expect/Continue, typed failures and lifecycle are present. Routing, middleware, auth/schema, CONNECT tunnel,
   protocol Upgrade/WebSocket, cookie jar and multipart convenience remain named negative space rather than silent claims.
3. Reuse: the factor reuses io_core transport/options/scatter-gather, io_tls encrypted transport, Error, Time, Mem and
   Container text along legal graph edges. HTTP syntax/framing remains local because none of those facilities has the same
   contract. No duplicate runtime, allocator, clock, cancel authority or normal-path queue remains.
4. Performance: the qualified W1-WSL natural REST/stream route is `PARITY` with Beast in both orders and the fragmented
   holdout is `WIN`; all four rows pass the predeclared 5% non-inferiority gate. This is engineering evidence for those
   two source-built routes, not an all-HTTP, TLS, Nginx-server, cross-platform or Formal SOTA claim.
5. FFI: `N/A`; the product tree has no C schema/header, bridge, C export, managed sample, FFI option or FFI-only layout.
   Shared artifacts export ordinary decorated C++23 symbols for compatible toolchains.
6. Delivery: the exact candidate passes direct source embedding plus relocated installed static/shared `io` product
   consumers on Windows, and a fresh fully-disconnected configure resolves its locked shared source cache without network.
   Linux/macOS shared correctness also exercises the same public factor and private provider boundary.
7. C++ usability/docs: typed RAII connection, move-only body handle, span/view, bounded convenience, streaming source,
   error, deadline/cancel and explicit `close()` form natural journeys. The guide states selection, borrow invalidation,
   owner-fiber/concurrency, backpressure, limit, teardown and unsupported protocol-takeover traps.

## Current decision

`CLOSED / FROZEN`。产品 candidate `400e015e6e8604b753d64cd27a16e21898670e91` 与全部适用工程
gate 已分别登记；本轮到此停止。后续只有 registry 中的具名 trigger 新事实才能依法重开。
F01--F05 的公开 API/usability 形状保持冻结。最终 immutable product candidate
`400e015e6e8604b753d64cd27a16e21898670e91` 已通过唯一允许的末次正式 P1：1 KiB normal workload
双执行顺序均为 `PARITY` 且通过 5% non-inferiority gate，64 KiB fragmented holdout 双顺序均为
`WIN`。性能调查到此停止，不再以 benchmark 为由改变产品。F06 的三平台、source/static/shared、
sanitizer、guide 与 red-team terminal inputs 已完成；registry 未走完合法转换前仍不得写成
`CLOSED/FROZEN`。

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
- 2026-09-08 instrument qualification：首先删除 `HttpBodySource` 对直接内存 writer 的
  apples-vs-oranges 对比，让 HYLE 与 Beast 都由应用按相同 chunk 大小写出，并锁定 persistent
  one-worker、同一 socketpair/缓冲、同一请求保留与响应增量校验合同。资格化暴露的真实机制是 HYLE
  chunk framing 原先把 prefix/payload/CRLF 拆为三个异步写，同时每条 head/ending 创建新的最小
  4 KiB grow buffer。修正后 plain TCP 复用已冻结的 io_core scatter/gather，连接复用 bounded output
  scratch，固定请求把 head/body 合并成一次 vectored route；TLS 小 chunk 在 retention budget 内复用
  staging，大 chunk 保留无额外大拷贝的顺序写，超过 portable vector-byte limit 的合法 fixed body 也
  自动退回顺序写。没有改变公开 API、layout 或常态 metadata。
- 资格化的 11-epoch 双向次序结果仅用于确认仪器有分辨率：1 KiB normal journey 中 HYLE 比 Beast
  慢约 `2.98%`（candidate-first）/`4.86%`（peer-first），均在预声明 5% non-inferiority margin 内；
  64 KiB fragmented holdout 中 HYLE 快约 `2.91x`/`2.97x`。这些不是 terminal evidence；新的
  immutable candidate 形成后只允许按 `bench/contracts/io-http-p1-v1.json` 正式执行一次证据包。
- `4a3cbf54` candidate 的正式 W1-WSL 包通过 artifact/build/host、A/A、5%/10% known-delta 与
  timer-floor 全部资格门；64 KiB holdout 在两个顺序分别约为 Beast 的 `2.90x`/`2.93x`，但 1 KiB
  normal 中位数比 Beast 慢 `4.46%`/`5.29%`，两侧 98.9% CI 均跨过预声明 5% non-inferiority
  limit，因此裁决严格为 `INCONCLUSIVE`，不得写成 PASS 或 FAIL。
- 第二项也是本卡最后一项机制诊断比较了实际调用链：每次 plain chunk 原为
  `http_body_write -> write_all_buffers -> http_transport_write_all -> io_core::write_all`。HYLE `Task`
  的每个 coroutine frame 都通过显式 `AllocatorRef` 分配，两个中间层只转发/翻译结果，却让 1 KiB
  workload 每个 response 重复支付 8 次；Beast 的相同行为由一个 `async_write` composed operation
  承担。最终修正使 plain transport dispatch 直接返回 io_core vectored task，并在 owner coroutine
  内同步翻译结果；fixed request 同样删除纯转发 frame。双顺序非正式机制检查的 normal 差距约
  `2.3%`，方向一致，Windows plain/TLS correctness 仍通过。该修正形成 `400e015e` candidate；其
  唯一正式 qualification 结果见下一项。此后不再增加诊断或改变 workload。
- `400e015e` 的最终 W1-WSL 正式包通过 source/peer/artifact/build/host provenance、A/A、5%/10%
  known-delta、timer-floor、qualification 与 performance 聚合门。1 KiB normal 的 candidate-first /
  peer-first 中位 advantage ratio 为 `0.9941` / `1.0042`，置信区间分别为
  `[0.9843, 1.0288]` / `[0.9808, 1.0288]`，均裁决 `PARITY`；64 KiB fragmented holdout 为
  `3.0297` / `3.0418`，均裁决 `WIN`。manifest / summary SHA-256 分别为
  `BBA0094D768F93CE9BBAA25B6F88F3400DA05567A22CD3B03C078A0CF2C043F3` /
  `31C919D09591F96E10A2B198B14EEA235B78EA92E3017B2157B5F186B1678C28`；contract SHA-256 为
  `4340D27B845DC83D1C2A86A02192FAD1DE0110992E583ECCF4A6723C1C716C9C`。这是 WSL 工程证据，
  不外推为所有网络、硬件、TLS 或完整服务器的普遍优势。
- Nginx `5c812c034e6cb10368f9e98cf05b6d0a01f27f1f` 有界实现对照确认：其 memory chain 同样以
  栈内 iovec、相邻内存段合并、partial-send cursor 和 write-ready backpressure 工作；HYLE 已有
  64 段原生 scatter/gather、游标续写和 awaitable backpressure，HTTP chunk 的 prefix/payload/CRLF
  本来就不相邻，因而相邻段合并不是本 workload 的缺陷。Nginx 的 `postpone_output`、rate limit 与
  `sendfile_max_chunk` 是服务器策略，不能作为低延迟 token stream 的隐藏默认缓冲。Nginx 另有
  memory/file mixed chain 与跨平台 `sendfile` 路线；HYLE 当前没有 file-to-socket 零拷贝 primitive，
  这是通用 artifact-serving 的命名候选，不属于本卡 HTTP/1 协议范围或已冻结 FIX 集。只有真实
  consumer 证明需要时，才依法重开其最低 owner（`io_core` 或独立 file-transfer factor）；不得为
  通过 benchmark 把静态文件服务器策略塞进 `io_http`。

## F06 terminal validation inputs

所有产品验证都绑定 clean、exact `400e015e6e8604b753d64cd27a16e21898670e91` checkout；文档与
registry 归档不改变该产品 identity：

| gate | environment / shape | result |
|---|---|---|
| correctness | W1-WIN，MSVC 19.50，Release/shared，TLS ON | `hyle_io_http_tests` + `hyle_io_tls_tests`，5/5 PASS |
| correctness | W1-WSL，GCC 14.2，Release/shared，TLS ON | 同一 5 tests，5/5 PASS |
| correctness | M1-MAC，AppleClang 21，Release/shared，TLS ON | 同一 5 tests，5/5 PASS |
| sanitizers | W1-WSL，GCC 14.2，RelWithDebInfo/static，ASan/UBSan/TSan | 三个独立 build 各 5/5 PASS；TSan 以禁用 ASLR 的受支持方式运行 |
| delivery | W1-WIN selected product `io` | source consumer、relocated installed shared consumer、relocated installed static consumer 全部 PASS |
| offline | W1-WIN，fresh configure，static `io`，tests/TLS/H2 off | `FETCHCONTENT_FULLY_DISCONNECTED=ON` PASS；无网络访问 |
| performance | W1-WSL，qualified P1，n=15 × 双顺序 | normal 2/2 PARITY、holdout 2/2 WIN、5% NI 4/4 PASS |

原始 WSL bundle 已用 `hyle-perf verify` 重放为 `QUALIFIED`、`paths_verified=true`、
`provenance_compatibility=EXACT`；仓库内 68-file/409,501-byte 镜像与原目录 `diff -qr` 为空。manifest
有意保留 live WSL artifact/toolchain 的绝对路径，Windows checkout 不能冒充可独立重放的 Formal archive。

macOS 链接器报告本机 26.6.2 与 Homebrew OpenSSL 26.0 deployment metadata 不同；两者都高于库要求且
correctness 全过，这只是验证环境 warning，不是产品或交付失败。Windows wrapper 若从另一个 current
directory 以绝对路径调用，会把 preset 作用到调用目录；本轮通过在 exact source checkout 内执行规避。
它不影响三种 consumer 结果，也不授权 IO task 顺手修改公共 build tooling；若要修复应作为独立
build/package finding 处理。

## Candidate red-team receipt

- 自顶向下按 llama.cpp/ggml、ds4、vLLM、SGLang 的 REST/SSE/disconnect 旅程复查，全部 admitted
  family 都有有界正常路线；应用路由、模型 schema、认证、batching/handler pool 没有下沉。
- 自底向上反查公开 header/source/export：没有遗留 pointer/bool/fixed-slot/callback server API、C ABI、
  provider-native handle、benchmark-only public entry 或第二套 deadline/cancel/allocator authority。
- RFC 9112 body-length 优先级要求 HEAD、1xx、204、304 在 header 后结束，不论收到什么 framing field；
  当前 client 的 no-body 路线因此是正确实现而非漏掉 CL/TE 检查。205 的发送端不生成 content，接收端
  同样不暴露 content。普通 response 的 TE+CL、冲突 CL、非 final/bare chunked 仍按安全边界拒绝。
- CONNECT 的 2xx 会转为 tunnel，101 会把 transport 交给 upgraded protocol；当前因子没有安全表达
  residual buffered bytes/transport hand-off 的类型，故两者保持明确负空间。guide 已禁止把通用
  method/header 当作 tunnel API；收到 101 后连接不可复用。WebSocket 非目标没有被写成“部分支持”。
- connection outlives borrowed head/chunk/reader/writer 和 pending operation 的要求已在 guide 明示；
  unfinished body 会取消复用，`discard()`、abandon/final-close、幂等 close 与 TLS close-notify 路径一致。
- 输入 head、header count、body、read chunk 与高水位保留有界；输出逐 chunk await transport，未引入
  hidden queue 或 per-chunk heap allocation。Nginx 对照没有发现当前 REST/SSE 路线遗漏更优复杂度；
  相邻 buffer 合并对非相邻 chunk prefix/body/CRLF 不适用，sendfile 留在未来 artifact-serving owner。

没有发现会推翻 candidate 的 correctness、安全、复杂度、自然 API、依赖或交付缺陷。

| gate | verdict | basis |
|---|---|---|
| scope / architecture | `PASS` | F01--F06 完成；协议 factor 与应用/transport/provider owner 分离 |
| correctness | `PASS` | 三平台 exact-candidate tests + WSL ASan/UBSan/TSan + adversarial framing/lifecycle cases |
| delivery | `PASS` | source、relocated installed static/shared consumer + fully disconnected configure |
| usability / docs | `PASS` | typed RAII/streaming journey、selection 与 lifetime/concurrency/teardown/negative-space 陷阱齐全 |
| reuse / red team | `PASS` | 合法复用既有 factor；RFC/Nginx/公开面反查无 material finding |
| engineering performance | `PASS` | qualified Beast same-contract 双顺序 4/4 5% NI PASS |

## Terminal receipt（`CLOSED / FROZEN`）

- freeze ID：`io-http-v1-freeze-2026-09-08`；完整复核点
  `bb55e3e1c4c15d6fa9cdc34879ef728273c75525` / tree
  `e4cf7f707dd9ec3b3ee45ef2583a664ad5987880`；产品 candidate/tree 保持
  `400e015e6e8604b753d64cd27a16e21898670e91` /
  `16b94922825e66373d4a33c76c22ece7156a4dde`；
- immutable candidate LOC：全库代码 `495109`；自实现不含测试 `85581`；io_http 实现 `2829`；
- scope、architecture、correctness、delivery、usability/docs、reuse/red-team 与 P1 全部 `PASS`；
  Formal 为 `NONE`；
- 限制：message/body view 与 reader/writer 借用 connection state，connection/source bytes 必须活过在途
  await，连接固定单 owner fiber。CONNECT/101 hand-off、WebSocket、HTTP2/3、路由/auth/schema、cookie、
  multipart 与 handler/batching policy 均为负空间。P1 只覆盖 W1-WSL plain、single-worker、persistent
  connection 的两个 source-build workload，不外推 TLS、完整服务器、WAN、跨平台或全操作 SOTA；
- reopen：只有真实 consumer 复现 framing/security/limit/disconnect/deadline/cancel/keep-alive/backpressure/
  teardown/ownership/API 缺陷，证明缺少通用 HTTP1/transport hand-off/artifact-serving primitive，产品
  API/layout/算法/正常成本/依赖/交付变化，provider/RFC/io_core/io_tls/OS/toolchain 语义漂移，或合格
  同合同证据证明 material regression/更优复杂度，才能依法重开。

三次 registry 转换前快照保存在 canonical checkout 外：

- `C:\public\project-hyle\_validation\io-http-v1\transitions\before-candidate.json`，SHA-256
  `B3052B93A9187145725256B1BCE78203D37A62549AF2769FC91A5966CD601A7B`；
- `C:\public\project-hyle\_validation\io-http-v1\transitions\before-verified.json`，SHA-256
  `0B2106FBC3167463B4D42FBA11F16DDCD5C89A2615A95A53D1A4A10D39EB035F`；
- `C:\public\project-hyle\_validation\io-http-v1\transitions\before-closed.json`，SHA-256
  `C238AB7591D0D7FF2BF6D9986FF56766697FDF64B3680E817D2ADF064F741ABE`。

本轮终止；HTTP/2、transfer 与 io_core 后续施工不得把本卡历史命令式文字当成继续修改 HTTP/1 的授权。
