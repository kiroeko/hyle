# WebSocket v1：RFC 6455 双向消息连接

<!-- MODULE CLOSURE RECORD: SCOPED -->

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
