# IO HTTP v3：文件区域响应体

<!-- MODULE CLOSURE RECORD: CANDIDATE -->

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

## 能力、API 与状态机审计（`AUDITED`）

| 能力族 | 当前公开面 | 完整目标 | 动作 |
|---|---|---|---|
| fixed bytes response | `send_response(head, span)` | 保持原形状和成本 | `KEEP` |
| generated streaming body | `start_response` / `HttpBodyWriter` / `HttpBodySource` | 保持 chunked/fixed/close-delimited 合同 | `KEEP` |
| opened file region response | 用户手写 source callback 与 offset | `send_response(head, file, region)` | `FIX` `G06-H01/H02` |
| HEAD / bodyless status | writer 已有 suppressed state | file overload 必须复用相同状态机 | `KEEP + FIX` |
| path/range/MIME/cache/conditional | 无 | service/router policy | `REJECT` |

公开面回映覆盖 server read/continue/start/fixed/source/close、client request/response、`HttpTransport` TCP/TLS
variant 与 `HttpBodyWriter` generation/busy/remaining/abandon。文件便利入口只增加一个现有名称的 overload：

```cpp
connection.send_response(head, file, FileRegion{offset, length}, options);
```

它先调用既有 `start_response(..., region.length)`，因此 request body retirement、header validation、HTTP/1.0/
1.1 framing、keep-alive 与 allocator failure没有第二份实现。HEAD 与 304 发送声明长度但不读 file；204/205
仍只接受零长度且不序列化 `Content-Length`。普通 fixed response：plain `TcpStream` 委托 io_core G06；TLS
只分配一块 64 KiB scratch，逐块 positional read 后走现有 TLS body write。成功恰好把 writer remaining 归零
并调用同一个 finish；partial、EOF、cancel、deadline、peer close 或 file error 都 abandon writer、令 connection
不可复用，并返回 typed status，不尝试在已发 header 后换一套 response。

## P0 时序与成本账单

| 路线 | 既有成本 | 本轮新增成本 | 裁决 |
|---|---|---|---|
| HEAD/204/205/304 | build/write one response head | 一次 overload dispatch；零 file IO、零 scratch | O(headers)，不污染默认 fixed/span 路线 |
| plain fixed file | response head + writer lifecycle | 一次 io_core Task；body 为 `TransmitFile/sendfile` chunk，零 payload userspace copy | O(headers + chunks)，无第二 HTTP parser/writer |
| TLS fixed file | response head + TLS record pump | 一次 64 KiB scratch；每块一 file read + 现有 TLS write；无逐块 buffer allocation | O(n)，加密本就必须读取 plaintext；不得与 plain zero-copy 排名 |
| failure | 已写出的 head/bytes 不可回滚 | 保留 lower progress 用于内部 state；writer abandonment 关闭复用 | O(progress)，不重试/不改 framing |

nginx `5c812c034e6c` 与 Beast `cb94f33fcb13` 证明 plain file response 的 kernel path；SGLang
`dcebe8c4733a` `FileResponse` 与 llama.cpp `e71b80510c84` static mount 证明 consumer task。Starlette/httplib
同时拥有 router/path/MIME，因此只借任务，不把其服务策略下沉。TLS 没有等价 zero-copy，使用同一
`OperationOptions` 贯穿 read/write；任何对 raw sendfile 与完整 HTTP/TLS façade 的排名都是苹果对橘子。

## 冻结有限改动集

| ID | 已证问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `G06-H01` | fixed/source 不能自然消费 open File | 一个 file-region overload、内部 transport/writer access，不复制 framing | overload/header/三交付 |
| `G06-H02` | plain/TLS 路由、partial failure 未统一提交 | plain 委托 io_core；TLS bounded fallback；统一 remaining/abandon/finish | HEAD/status/EOF/peer close/reuse/TLS |
| `G06-H03` | 用户不知道何时选 bytes/source/file | 更新 IO guide/architecture 与定向 tests | recipe、限制、无过时 owner 文案 |

不允许加入 static-file server、path open、Range/MIME/ETag、client upload、HTTP/2 或 WebSocket。新增公开头只
复用既有 `io_core` dependency；不增加 component edge、provider、开关或 binary。审计结论为 **PASS**，允许
进入三项有限实现；correctness、交付、红队与候选性能尚未运行。

## 候选签署（`CANDIDATE`）

- `G06-H01/H02/H03` 已落地：新增唯一 overload
  `send_response(head, File&, FileRegion, options)`；它复用原 response state machine。plain TCP 委托
  `io_core::send_file_region`，TLS 仅持有一块 64 KiB plaintext scratch；旧 span/source 热路径未改。
- pre-wire 同时校验 worker 与精确 `IoContext` identity；HEAD/204/205/304 复用 suppressed writer，零 file
  IO、零 scratch；header 发出后的短文件、取消、peer close 或 allocation failure 都 abandon connection，
  不把半响应误作 keep-alive success。direct-file alignment 只在确实发送 body 时检查。
- Windows/MSVC 19.50 shared Release 的 TLS OFF/ON 定向测试通过；Linux/GCC 14 shared Release 的 TLS
  OFF/ON 测试在默认 io_uring 与强制 epoll 下通过。已覆盖 GET region、同连接 HEAD、204/205/304、短文件、
  plain/TLS 跨 context 预 wire 拒绝，以及 TLS 64 KiB scratch OOM。
- 不可变候选是 `d72bd68aa4d23d14c66b31d5d4d1f4b9818dec6d`。产品实现最后改变于
  `65b7dc8c699cbedaf2a9311943241db600224dcd`；后续仅有测试协调与 optional-factor fixture 修正。
- Windows、Linux io_uring/epoll、两台 macOS 的 plain/TLS file-body correctness 均通过；WSL
  ASan/UBSan/TSan 各 `19/19 PASS`；Windows source/relocated shared/relocated static 的 `io` 直接消费
  全部通过，TLS ON/OFF 与 HTTP2 ON/OFF factor 组合没有被本轮破坏。
- 源码 preflight 未找到能与完整 HYLE framing、request retirement、backpressure、completion 和 teardown
  同合同的 peer，P1 为 `NOT_APPLICABLE`；P0 成本账单没有固定输项。terminal 红队与中央状态迁移仍须在
  下一提交签署，本节不提前声明 `CLOSED`。
