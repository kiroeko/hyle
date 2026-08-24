# io-11：HTTP/2（HPACK / 多路复用 / 流控）

> **历史施工记录：**本文保留落地时的上游 pin、源码行号与验收证据，不代表当前依赖版本。现行
> revision 以 [`cmake/HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json) 为准；本轮升级
> 影响与重新复证门见 [`dependency-upgrade-2026-08.md`](./dependency-upgrade-2026-08.md)。

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-10（HTTP/1.1）、io-08（TLS）|
| **状态** | ✅ Windows IOCP / Linux io_uring+epoll / macOS kqueue 真机验收 |

> **范围**：借 nghttp2 的 RFC 9113 framing、HPACK、stream state 与 flow
> control；HYLE 负责 allocator、资源上限、协程 transport、C ABI 和生命周期。
> 固定 body 上限 8 MiB；面向无限 token stream 的增量 body writer 仍是后续
> convenience，不伪装成已完成。

## 架构

```text
 C++ convenience                         pure C / FFI
 ┌──────────────────────────────┐        ┌──────────────────────────────┐
 │ Http2ClientConn/ServerConn   │        │ hyle_io_http2_session        │
 │ Task + stream-id + views     │        │ feed / next_output / message │
 └──────────────┬───────────────┘        └──────────────┬───────────────┘
                │ owns transport                         │ transport-neutral
                v                                        v
 ┌──────────────────────────────┐        caller's socket/TLS completion loop
 │ HttpTransport               │
 │ TcpStream or ALPN h2 TLS    │
 └──────────────┬───────────────┘
                └──────────────────┬─────────────────────┘
                                   v
                    ┌──────────────────────────────┐
                    │ HYLE session integration    │
                    │ SwissTable<stream_id,state> │
                    │ caps / views / lifecycle    │
                    └──────────────┬───────────────┘
                                   v
                    ┌──────────────────────────────┐
                    │ pinned nghttp2 protocol core│
                    │ HPACK / frames / flow ctrl  │
                    └──────────────────────────────┘
```

nghttp2 不接管 socket、线程、event loop 或 allocator。依赖固定到 v1.69.0
commit，应用/示例/HTTP3/HPACK tools 均不构建。

## 已实现

- client/server prior-knowledge h2c；TLS 构造仅接受 ALPN 已选中 `h2` 的
  `TlsStream`；
- 同一连接可先提交多个 request，按完成顺序取 response；server 可按任意
  stream 顺序回复；
- SETTINGS、HEADERS、DATA、RST_STREAM、GOAWAY、HPACK 与 connection/stream
  flow-control；
- 128 headers、64 KiB header list、8 MiB fixed body、100 concurrent streams，
  并限制 SETTINGS/CONTINUATION；
- HYLE allocator 贯穿 nghttp2 与 stream storage；stream map 使用
  `SwissTable`，无 STL ABI；
- C ABI 是 transport-neutral codec：C/外语 runtime 用现有 proactor/TLS 收发
  字节，`feed` 输入、`next_output` 取编码帧，不需要伪造 C++ coroutine；
- HTTP/2 关闭构建仍导出同一组 C 符号，`available()==0`，失败统一写
  `hyle_io_last_error()`；全零 allocator 在启用构建中选择 system allocator；
- reset/refused stream 会作为带 `stream_error` 的完成消息交给 client，不会让
  `next_response` 永久等待。

## 验收证据

三平台 Release（Windows MSVC/IOCP、WSL GCC 14/io_uring+forced epoll、
AppleClang/kqueue）：

- h2c 两个并发 POST，server 逆序 response，验证 stream ID、重复 header
  HPACK 路径与 body；
- 96 KiB request + 160 KiB response，双向跨越默认 64 KiB window，验证必须
  读取 WINDOW_UPDATE 才能继续发送；
- 纯 C client/server codec 互喂 SETTINGS/request/response，验证 header/body
  borrowed view 与 allocator；
- 真实 TLS loopback 在双方 ALPN 选择 `h2` 后跑完整 request/response。

三平台各 14/14 IO test 通过。macOS 组合测试还抓出 graceful close 与
`SIGPIPE` 尾部竞态：H2 shutdown 现在 flush GOAWAY 后执行 TLS close_notify /
TCP half-close，HYLE TCP socket 在 macOS 设置 `SO_NOSIGPIPE`；修复后 h2-over-
TLS 连续 100 次通过。

## 明确不在本里程碑

- HTTP/1.1 Upgrade 到 h2c（当前是 prior knowledge）；
- server push（RFC 9113 已弃用，client SETTINGS 明确关闭）；
- WebSocket、HTTP/3、gRPC；
- 无界/增量 request/response body。当前 fixed-body API 会完整保留 body，
  token/SSE 式无限流应在后续增加有背压的 chunk producer，不能绕开上限。

## 维护规则

协议语义优先交给 nghttp2，不在 HYLE 重写 HPACK/frame state machine。HYLE
改动集中在 transport、allocator、上限、stream 生命周期和两种外观；新增能力
必须同时考虑 C++ owner-fiber 腿与 C codec 腿。任何 flow-control 改动都要保留
“body 大于 64 KiB”的回归测试。
