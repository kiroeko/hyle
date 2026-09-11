# HTTP v4：协议升级连接交接

<!-- MODULE CLOSURE RECORD: AUDITED -->

## Closure card

- component / owner / round：`io_http` / HTTP maintainers / `io-http-v4-upgrade-handoff`；kind=`REOPEN`；
  canonical checkout=`C:\public\project-hyle\hyle`；baseline=`7a799620b4f8a25199ca37da434347531a7f6b50`；
  previous terminal record=[io-http-v3](io-http-v3-file-body.md)，其 candidate 为
  `d72bd68aa4d23d14c66b31d5d4d1f4b9818dec6d`。
- trigger 与证据：[基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)准入 G04。
  当前 `http_server_read_request`/`http_client_read_response` 会把 head 之后已经读到的 bytes 留在私有
  connection wire buffer；101 又把 HTTP connection 标为不可复用，但没有同时移交 transport、预读 bytes
  和唯一 ownership 的路线。真实 WebSocket consumer 因此只能重写 HTTP parser 或丢字节。
- 正常 consumer 任务：HTTP server 解析、路由并鉴权一个 Upgrade request 后，把同一 plain/TLS transport
  连同 head 后的全部 prefetched bytes 一次性交给新协议；client 验证 101 后执行同一所有权迁移。旧 HTTP
  view/body borrow 立即失效，连接不能被两个协议 owner 同时使用。
- 主产品纵切 / 具名来源：vLLM/SGLang realtime speech 与多模态服务在 HTTP listener 上接受 WebSocket；
  llama.cpp vendored cpp-httplib 与 Boost.Beast 都从已解析 HTTP handshake 进入 WebSocket。正交纵切：渲染/
  资源工具在同一 TLS endpoint 上把控制连接升级为领域自有双向协议，验证 seam 不携带 WebSocket 策略。
- 跨模块 public journey：`io_http` 唯一拥有 HTTP parsing、101 和 handoff；`io_core`/`io_tls` 继续拥有 byte
  transport；`io_websocket` 消费 handoff 并拥有 RFC 6455。HTTP 不依赖 WebSocket，且不建立第二 TCP/TLS
  runtime。
- scope：冻结一条 server handoff 和一条 client-101 handoff；保持 residual bytes 顺序且不重复读取；拒绝
  active/unfinished request body、writer、非 101 response 和 stale borrow；迁移后源 connection 明确 invalid；
  plain/TLS 共用同一机制。
- non-goals：WebSocket header/SHA-1/framing、应用 route/auth、HTTP CONNECT、HTTP/2 extended CONNECT、
  任意协议注册表、Native FFI，以及改变普通 HTTP keep-alive/body 路线。
- 候选依赖 / 直接消费者：不预设新外部依赖或新 HYLE edge；直接消费者固定为 `io_websocket`。AUDITED
  阶段裁决 handoff 是公开 value、窄 detail seam，还是由 WebSocket public operation 隐藏的 move-only owner。
- 性能 tier：`P0`。handoff 每连接至多一次且只在 opt-in Upgrade 路线发生；必须证明 O(prefetched bytes)
  上界、至多一次必要搬运且普通 HTTP 对象布局/热路径不增加字段、分支、allocation 或 syscall。没有 raw
  socket 对完整 HTTP handoff 的伪排名。
- 调查与纠正预算：一次 HTTP ownership/source audit、一次 Boost.Beast handoff 对账；至多一个 HTTP 产品
  批次。若安全 handoff 必须重写 parser/runtime 或给每条普通 HTTP connection 增加不可隔离的 material 成本，
  本轮 `BLOCKED`，不降低为丢弃 prefetched bytes。
- 预声明验证：server/client、zero/one/multiple prefetched frames、head/body 同 read、active/stale body、非 101、
  OOM/搬运失败、plain/TLS、deadline/cancel、source/static/shared、逐头、三平台、sanitizer、直接 WebSocket
  consumer、红队和 P0。
- stop / blocker：一个所有权迁移接口、两个方向及失败原子性裁决后停止；不把其它 HTTP feature 捎带进本轮。

## SCOPED 待裁决问题

1. 如何同时迁移 `HttpTransport` 与私有 `GrowBuffer`，既不泄漏内部 buffer 类型，也不为后续每次读增加拷贝？
2. server handoff 是否必须先确认 request body 已完成；client 如何把 101 与其它 final response 明确区分？
3. failure 应保持原 HTTP owner 可关闭，还是消费并关闭连接；哪种形状不会制造两个 owner？
4. `io_websocket` 如何消费 seam 而不迫使 `io_http` 在公开面知道 WebSocket 类型？

本卡只冻结问题、范围和预算；registry 为 `SCOPED` 时只允许审计，不允许修改产品代码。

## 能力与公开面审计（`AUDITED`）

源码回映确认 server/client parser 都在消费 head 后把同一次 transport read 的余量保存在各自 `wire`，而
`HttpTransport` 已经统一 plain/TLS byte transport。缺口不是第二套 parser，而是一个拥有 transport 与这段
余量的协议无关 owner。最终 API 冻结为：

```cpp
class HttpUpgradeStream {                 // move-only owner
  bool valid() const noexcept;
  bool encrypted() const noexcept;
  StringView negotiated_protocol() const noexcept;
  unsigned owner_worker() const noexcept;
  Task<TransferResult> read_some(span<byte>, OperationOptions = {});
  Task<TransferResult> write_some(span<const byte>, OperationOptions = {});
  Task<BufferSequenceResult> write_all(span<const ConstBuffer>,
                                       OperationOptions = {});
  Task<Status> shutdown(OperationOptions = {});
};

Task<Result<HttpUpgradeStream>>
HttpServerConnection::accept_upgrade(HttpIncomingRequest&&, StringView protocol,
                                     span<const HttpHeader> response_headers,
                                     OperationOptions = {});
Task<Result<HttpIncomingResponse>>
HttpClientConnection::request_upgrade(const HttpRequestHead&, StringView protocol,
                                      OperationOptions = {});
Result<HttpUpgradeStream>
HttpClientConnection::take_upgrade(HttpIncomingResponse&&, StringView protocol);
```

server API 检查同 owner 的 completed body、HTTP/1.1 `Connection: Upgrade` 与请求的 protocol token，预留
handoff state 后才发送 101；写失败仍由原 HTTP owner 负责关闭，成功才令原 connection 和 borrow 失效。
client 的第一步返回普通 response，应用/协议层可检查 101 专用 headers；第二步只接受同 owner 的 101、
completed body 与匹配的 `Connection`/`Upgrade`，再做唯一 ownership 迁移。非 101 仍可按普通 HTTP body
读取/关闭，不因一次升级尝试丢失诊断。

`HttpUpgradeStream` 的私有 state O(1) 接管 `GrowBuffer` allocation，不复制 prefetched bytes；读取先顺序
排空余量，再直接读 transport。公开类型不泄漏 TCP/TLS variant、buffer allocator 或 WebSocket 状态。
普通 `HttpServerConnection`/`HttpClientConnection`、head POD、parser 分支与对象 layout 都不改变。

## 冻结有限改动集与 P0 账单

| ID | 已证问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `G04-H01` | transport 与 prefetched bytes 无统一 owner | `HttpUpgradeStream`，O(1) buffer ownership transfer | zero/one/multiple prefix、plain/TLS、move/destroy |
| `G04-H02` | server 不能合法发送 101 并原子交接 | 专用 `accept_upgrade`；生成 Connection/Upgrade，不放宽普通 header API | active body/stale borrow/write failure/success invalidation |
| `G04-H03` | client 不能发 Upgrade 或在 101 后交接 | `request_upgrade` + `take_upgrade` 两阶段 | rejection body 保留、101 headers、wrong owner/protocol |
| `G04-H04` | 公开/构建/使用文档无升级路线 | 定向测试、guide、component/package wiring | source/static/shared、直接 custom-protocol consumer |

P0：普通 HTTP 路线新增成本为 **零字段、零分支、零 allocation、零 syscall**。升级路线每连接一次：已有
HTTP head parse/write，加一次小 state allocation；transport、wire allocation 与 prefix 原地移动，handoff
O(1)，后续 prefix 每 byte 恰好读一次，随后与原 transport 相同。公开 vector write 是既有 transport helper
的 owner-safe 暴露，不新增 payload copy。不存在把 raw socket 与完整 HTTP handoff 排名的 P1。

## Owner 七问预裁决

1. 能力：只补 RFC 9110 通用 Upgrade ownership seam；WebSocket、CONNECT/H2/H3 与服务策略不下沉。
2. 依赖：继续只复用 Core/Error/IO Core/TLS/Mem/Sched/Time；HTTP 不依赖新协议。
3. 形状：专用 opt-in API，不给普通 head 增加 `upgrade` 字段或 benchmark 参数；两阶段 client 保留 rejection。
4. 性能：O(1) owner move、零 prefix copy；普通 HTTP 热路径和 layout 无变化。
5. 正确性：同-owner generation、body completion、header token、allocation-before-commit 与 move-after-write 保证
   不产生双 owner、悬空 reader 或丢失 prefetched bytes。
6. 交付：`HttpUpgradeStream` 属于既有 `io_http` artifact，无新 provider、option、第三方或 Native FFI。
7. 文档：用户指南明确 Upgrade 是 byte-stream 交接；RFC 6455 由 `io_websocket`，route/auth 由应用拥有。

审计裁决为 **PASS**，只允许上述四项进入实现；原 SCOPED 问题已全部落到明确 ownership、失败原子性和
成本合同。产品验证尚未运行，不能提前写 PASS。
