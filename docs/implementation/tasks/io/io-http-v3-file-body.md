# IO HTTP v3：文件区域响应体

<!-- MODULE CLOSURE RECORD: SCOPED -->

## Closure card

- component / owner / round：`io_http` / HTTP maintainers / `io-http-v3-file-body`；kind=`REOPEN`；
  canonical checkout=`C:\public\project-hyle\hyle`；baseline=
  `694cde21d5608e4569fb87a1b12d7c68ee133ea0`；前轮 terminal record 是
  [`io-http-v2-duplex-progress`](io-http-v2-duplex-progress.md)。
- trigger 与证据：[基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)已准入 G06。
  当前 fixed span 和 callback `HttpBodySource` 都要求应用拥有/填充用户态 bytes；即使已有 open File 与明确
  region，用户仍须手写 producer、offset/progress、EOF/error、Content-Length、keep-alive failure 和 TLS/
  plain 分流，plain TCP 也无法复用 io_core native fast path。
- 正常 consumer 任务：在已读完或放弃 request body 后，以一个已打开 File 的明确 region 发送 HTTP/1.1
  response；HEAD/无 body status 只发送合法 header；成功后连接按既有 framing 规则复用，失败后不会误复用。
- 主产品纵切 / 具名来源：推理 REST 服务发送模型 sidecar、tokenizer、adapter 或下载 range；参考
  vLLM/SGLang 的服务路径和 nginx/Boost.Beast 文件响应 shape。
- 正交反偏置纵切：渲染 asset server 发送 pack-file 中一段已知长度资源，验证 API 不依赖模型、route 或
  whole-file 假设。
- 跨模块 public journey 与 owner：io_http 拥有 response framing、request-body retirement、keep-alive 与
  transport dispatch；io_core 拥有 G06 file-region transfer；io_tls 拥有 encrypted stream；Mem 只提供有界
  fallback storage。应用拥有 path traversal 防护、range/conditional request、MIME/cache policy 与 routing。
- scope：增加明确 file-region HTTP response 入口；固定 Content-Length；plain TCP 复用 io_core native/
  portable route，TLS 走同语义 bounded read/write fallback；覆盖 HEAD、204/205/304、partial/EOF/error、
  deadline/cancel、backpressure、connection reuse 和完整 teardown。
- non-goals：router/static-file server、path 打开、目录索引、MIME、ETag/date/range 解析、multipart、压缩、
  sendfile over TLS 的虚假 zero-copy、HTTP/2 API、WebSocket、Native FFI 或 hyle-service policy。
- 直接依赖 / 直接消费者：不预设新增 component edge；依赖已存在 io_core/io_tls/mem/sched/error/time。
  直接消费者是公开 `hyle::io` façade 和后续服务/渲染 recipe；io_http2 不因名字相近自动重开。
- 性能 tier 与原因：`P0` 先审 framing、copy/allocation/syscall 与 plain/TLS dispatch；plain fixed-length
  response 若可建立完整 same-contract peer，则做一次 `P1`，固定 5% non-inferiority margin。TLS fallback
  不与 plain sendfile 排名。
- 适用机器 / 仪器：Windows、Linux、macOS correctness；TLS on/off、ASan/UBSan、适用 TSan；source/static/
  shared IO consumer。P1 必须在源码 preflight 后用统一仪器、同 request/response/bytes/backpressure/
  completion/teardown 和双顺序；不能用无 framing raw sendfile 对 HTTP façade。
- 调查与纠正预算：一次现有 HTTP writer/transport 与最多两个 peer 的源码审计；一个 API 方案和至多两个
  `FIX` 批次（public integration、plain/TLS routing）。完整 P1 最多一次；失败先做根因源码/汇编/系统调用
  归因，不增加样本或改 API 迎合点值。
- 预声明验证：fixed/zero region、HEAD、204/205/304、invalid length/overflow、short file、partial peer close、
  cancel/deadline、TLS fallback、body-source/response coexistence、request-body drain/abandon、keep-alive、OOM；
  io_http/io_core 直接测试、三平台、sanitizer、逐头、三交付、红队、P0 与适用 P1。
- stop / blocker：若 HTTP connection 无法在不泄漏 transport internals、不复制 writer 状态机且不破坏
  keep-alive/close 的前提下组合 io_core transfer，则退回 `SCOPED` 或 `BLOCKED`；不得为 fast path 新建第二
  HTTP API 宇宙，也不得把应用静态文件策略下沉。

## 审计待办

进入 `AUDITED` 前必须冻结 file-region 形状、transport dispatch、HEAD/status/Content-Length 与 failure 后
connection 状态，完成 capability map、公开面回映、依赖/交付审计和 P0 源码账单。
