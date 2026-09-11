# WebSocket v1：RFC 6455 双向消息连接

<!-- MODULE CLOSURE RECORD: CANDIDATE -->

## Closure card

- component / owner / round：`io_websocket` / WebSocket maintainers / `io-websocket-v1-rfc6455`；kind=`NEW`；
  canonical checkout=`C:\public\project-hyle\hyle`；baseline=`7a799620b4f8a25199ca37da434347531a7f6b50`。
  component identity 已登记为 `QUEUED`，尚无 target、产品 façade 或公开头。
- trigger 与证据：[基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)准入 G04；
  vLLM/SGLang 已有 realtime speech、多模态 binary frame 和 persistent request/response journey，ds4 自写了
  browser-devtools client，llama.cpp 的当前 vendored cpp-httplib 同时提供 client/server WebSocket。HYLE 已有
  HTTP/TCP/TLS/Sched 原子，但没有 RFC 6455 lifecycle owner。
- 正常 consumer 任务：从现有 HTTP listener 接受浏览器 `ws/wss` 连接，或从已连接 plain/TLS transport 发起
  client handshake；在单连接上流式收发 text/binary message，处理 fragmentation、ping/pong 与有类型的 close，
  deadline/cancel 后保持明确的连接状态和 bounded memory/backpressure。
- 主产品纵切 / 具名来源：SGLang `dcebe8c4733a5cb802fb679e0c8fbb886e2801b0` 与 vLLM
  `51da0ca66c8065619c79e35dff97aa99aeaf5644` 的 realtime text/binary sessions。正交纵切：ds4
  `f62ca29a308724cde5bc99134ede19104b2a3260` 的 browser-devtools client 和渲染 live telemetry，防止 API
  被 server/JSON-only shape 特化。
- strong peer / standard：RFC 6455、RFC 9110 Upgrade；Boost.Beast
  `cb94f33fcb13e90922905c2d8105508e71fa6db4` 与 llama.cpp/cpp-httplib
  `e71b80510c848c00175924ecf3c40333ccae8eb5`。参考只发现稳定语义，不引入 Asio 或 cpp-httplib runtime。
- 跨模块 public journey：`io_http` 解析/移交，`io_websocket` 做 handshake 与 protocol state；G01
  `container_text` 提供 Base64，`io_core/io_tls` 搬 bytes，Sched/Time/Error/Mem 提供 task、deadline/cancel、
  failure 和 allocator。route/auth/JSON/reconnect/rooms 留给 `hyle-service` 或应用。
- scope：client/server opening handshake；HTTP handoff 和 prefetched bytes；version 13、可选 Origin、caller
  subprotocol selection；text/binary、message/frame streaming、fragmentation；client masking/server rejection、
  ping/pong、close handshake/typed code；一读一写可并发、同方向并发拒绝；frame/message/buffer limits、UTF-8、
  deadline/cancel、bounded backpressure、plain/TLS 共用协议 core。
- non-goals：`permessage-deflate`/其它 extension、HTTP/2/3 WebSocket、proxy CONNECT、listener/router/auth/JSON、
  reconnect/heartbeat policy、rooms/pubsub、browser Origin policy、Native FFI、Compute/Graph。
- 候选依赖待 AUDITED 裁决：`core,error,io_http,mem,sched,container_text,time,util_core`；不引入第二 runtime。
  SHA-1 只允许一个私有、窄、可审来源，不建立通用 crypto 产品；mask/UTF-8 优先复用已关闭加速设施，但
  不为形式复用制造 SIMD/Container 依赖或 default metadata。
- 性能 tier：`P1`，material margin=`5%`。协议整体先做 P0 complexity/copy/allocation/syscall/metadata 账单；
  只有 mask/unmask bulk core 与 Boost.Beast 同语义 primitive 做双顺序 non-inferiority，端到端 HYLE-vs-Asio
  只作自然路线诊断，不能混入 runtime 差异签胜负。candidate 固定后只跑一次完整 P1。
- 调查与纠正预算：一次 RFC/consumer/API/peer 源码审计；至多四个产品批次（HTTP handoff、handshake/core、
  streaming/control、integration/docs）；P1 最多两次产品纠正，固定输先读源码/codegen，不增加样本找 PASS。
- 预声明验证：RFC handshake vectors、token/header/version/key/subprotocol/extension rejection；全部长度编码和
  canonical length、mask direction/key progression、fragment/control interleave、partial header/payload、UTF-8 跨
  fragment、message/frame/buffer cap、ping/pong、simultaneous/peer/local close、invalid code/reason、EOF、OOM、
  cancel/deadline、one-read+one-write、same-direction misuse、prefetched first frame、plain/TLS、source/static/shared、
  Windows/Linux/macOS、ASan/UBSan/适用 TSan、红队、P0/P1。
- stop / blocker：若无法在一个 HYLE IO runtime 上同时保证 prefetched handoff、一读一写、control interleave、
  bounded memory 与三平台一致 close，则 `BLOCKED`；不得降级成 whole-message-only、无 close owner 或忽略 mask/
  UTF-8 的 benchmark codec。

## SCOPED 待裁决问题

1. 公开 API 如何同时提供 message convenience 与真正的 partial read/write，而不暴露 frame-state misuse？
2. read 遇到 ping/close 时如何与唯一在途 application write 序列化 control reply，并让 deadline/cancel 生效？
3. UTF-8 validation 由 simdutf 借用还是窄自实现；跨 fragment partial code point 如何保留状态？
4. client masking key 使用哪个已冻结随机源，如何保证每帧新 key、无隐藏 allocation 且可确定性测试？
5. close API 如何区分 clean peer close、abnormal EOF、local close timeout 和 protocol failure？

本卡只冻结问题、范围和预算；`SCOPED -> AUDITED` 完成前不得写产品代码或 benchmark。

## 能力账本与 API freeze（`AUDITED`）

RFC、四个真实 consumer 与 Beast/cpp-httplib 源码回映后的有限公开面如下。它刻意以 message 为用户心智，
frame 只作为 partial progress/fragmentation 的内部协议事实：

```cpp
enum class WebSocketMessageType : uint8_t { kText, kBinary };
enum class WebSocketControlType : uint8_t { kPing, kPong };
struct WebSocketClose { uint16_t code; StringView reason; };
struct WebSocketReadResult {
  size_t bytes;
  WebSocketMessageType type;
  bool end_message;
  optional<WebSocketClose> close;
};
struct WebSocketLimits {
  uint64_t max_message_bytes;
  uint64_t max_frame_bytes;
  size_t write_fragment_bytes;
};

class WebSocketConnection {              // move-only owner
  static Task<Result<WebSocketConnection>>
  accept(AllocatorRef, HttpServerConnection&, HttpIncomingRequest&&,
         const WebSocketServerOptions&, OperationOptions = {});
  static Task<Result<WebSocketConnection>>
  connect(AllocatorRef, HttpTransport, const WebSocketClientOptions&,
          OperationOptions = {});
  Result<WebSocketMessageWriter> start_message(WebSocketMessageType);
  Task<Result<WebSocketReadResult>> read_some(span<byte>, OperationOptions = {});
  Task<Status> send_text(StringView, OperationOptions = {});
  Task<Status> send_binary(span<const byte>, OperationOptions = {});
  Task<Status> ping(span<const byte> = {}, OperationOptions = {});
  Task<Status> pong(span<const byte> = {}, OperationOptions = {});
  Task<Status> close(WebSocketClose = {}, OperationOptions = {});
  StringView selected_subprotocol() const noexcept;
};

class WebSocketMessageWriter {            // move-only connection borrow
  Task<Status> write(span<const byte>, bool end_message,
                     OperationOptions = {});
  Task<Status> finish(OperationOptions = {});
};
```

client options 拥有 authority/target/origin、caller headers 与 offered subprotocols；server options 只接受一个
caller 已选择且确实被请求的 subprotocol、额外 response headers、limits 与无分配 control callback。
连接保留 selected subprotocol 的必要小 ownership。caller 可在 `accept` 前用普通 `HttpRequestView` 做 route、
auth、Origin 与协议选择；这些策略不复制进 WebSocket。

读取把 text/binary message 流入 caller buffer，跨 frame 保留 type/UTF-8/message byte state；空 message 也以
`bytes=0,end_message=true` 返回。ping 自动串行发送 pong，ping/pong 可通知无分配 callback；peer close 自动
回复并返回有类型 close。`close` 是完整 local close handshake，要求没有另一 read 在途。允许一条 app read 与
一条 app write 同时在途；第二条同方向操作立即失败。内部 control write 用可取消/有 deadline 的 gate 等待
当前 frame 完成，不与 frame bytes 交错。

## 实现裁决与有限 FIX

| ID | 能力 / 已证缺口 | 实现路线 | 验收 |
|---|---|---|---|
| `G04-W01` | client/server RFC opening handshake | 私有 SHA-1 + 已冻结 Base64；secure_random 生成 16-byte key；HTTP handoff | RFC vector、duplicates/tokens/version/key/protocol/extensions |
| `G04-W02` | bounded streaming frame/message codec | caller-buffer read；header/payload incremental state；canonical lengths、role mask | partial/prefetch/all lengths/mask/fragment/caps/OOM |
| `G04-W03` | text 与 control lifecycle | incremental UTF-8；ping/pong；typed close/code/reason/EOF | split codepoint、interleave、simultaneous/local/peer close |
| `G04-W04` | duplex write serialization | one app writer generation + cancel/deadline control gate；server vector write | concurrent read/write、same-direction rejection、backpressure |
| `G04-W05` | client mask key 无 per-frame syscall | per-connection ChaCha20 stream，一次 `platform::secure_random` seed | 每 frame 新 key、deterministic core vector、零热路 allocation/syscall |
| `G04-W06` | 产品/交付/文档 | 新 `io_websocket` factor 与 `hyle::io` 聚合；guide/architecture/limitations | headers、source/static/shared、三平台、sanitizer、consumer |

SHA-1 和 ChaCha20 都是协议私有机制，不导出通用 crypto API。UTF-8 使用小型跨 chunk state machine；完整
contiguous handshake/close text 复用 `container_text` validation。server payload 以 header+caller payload
vector write；client 因 RFC masking 必须在 `write_fragment_bytes` 有界复用 scratch 中复制并 xor，不能虚假
宣称 zero-copy。读侧在 caller storage 原位 unmask，无 whole-message allocation。

## P0/P1 与 Owner 七问预裁决

- 每 frame parse O(header+payload)，每 payload byte 至多一次 copy（transport 必需）和 server 侧零/ client 侧
  一次 xor；每 message metadata O(1)，内存 O(write_fragment_bytes + 125-byte control + HTTP prefix)。
- random seed 每 client connection 一次 syscall；每 frame key 为 O(1) ChaCha word，不用弱 PRNG，也不逐 frame
  调 OS。server frame payload 无复制 vector write；client masking copy 是协议合同成本。
- P1 唯一同层计时是连续 bulk mask primitive 对 Beast scalar mask；同 role、相同 bytes、相同已生成 key、同
  completion，双顺序 5% non-inferiority。端到端 HYLE/Asio runtime 只作诊断，不签产品胜负。

七问：能力覆盖已准入 RFC 6455 core 且拒绝 service/extension；依赖全部是已关闭 HYLE 原子，不引入第二
runtime；message-first、caller-buffer、move-only owner 不是 benchmark seam；复杂度/常态成本如上；错误统一
使用 Status/Result，所有 limit/role/UTF-8/close/ownership failure 有确定状态迁移；独立 component 在 source/
static/shared 聚合一致；guide 将解释何时用 HTTP、WebSocket、raw stream，以及 masking copy、单方向并发、
borrow/deadline/close 陷阱。

审计裁决为 **PASS**。SCOPED 五问已固定为上述 API/state/cost 合同，只允许六项 FIX；产品测试、性能与红队
尚未运行，不能提前写 PASS。

## Immutable candidate

产品 candidate 为 `f231b6f41ca8e4599994c4d5fd71da6b80f6b2b7`。六项 ledger 已实现：协议无关
HTTP handoff、client/server handshake、caller-buffer message streaming、fragment/control/typed close、duplex
gate，以及每连接安全种子派生的 client mask keys。实现边界为公开 `websocket.h` 与私有 RFC state/SHA-1/
ChaCha/mask core，不引入 Asio、cpp-httplib 或第二 runtime。

进入 candidate 前的 Windows MSVC 19.50 shared 定向结果为：TLS-OFF WebSocket `8/8 PASS`、Upgrade
`3/3 PASS`；OpenSSL 3.6.3 TLS-ON WebSocket `9/9 PASS`。覆盖 RFC accept vector、同包首帧、fragment +
control interleave、126/65536 长度、mask/role/canonical length、UTF-8、frame/message caps、subprotocol/
extension rejection、close、同方向误用、OOM coroutine、deadline 与 cancel。模块实现代码为 **2,077 行**
（`scripts/loc-report.py --module io-websocket --commit f231b6f4... --self-check`）。完整三平台、三交付、
sanitizer、P1 与终局红队仍待候选验证，不能提前写 CLOSED。
