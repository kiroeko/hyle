# io-10：HTTP/1.1 server + client（over 自有 io_uring/fiber）

> **历史记录 / 已被取代。** 本页只记录 2026-07 的首次实现，所列
> `HttpServerConn`、`HttpClientConn`、`http_serve`、fixed header slot 与同步
> `NextChunkFn` 均已由活动的 [`io-http-v1`](io-http-v1.md) 重构删除，不能作为当前
> API、状态或待办。当前真相只看 `module-status.json` 指向的 closure card 与用户指南。

| 字段 | 值 |
|---|---|
| **阶段** | P5 |
| **模块** | io/ |
| **前置** | io-04（net）、io-06（超时）、io-08（TLS）、io-09（DNS）|
| **状态** | ✅ 完成（三平台，能力 C11）。Windows IOCP + Linux io_uring + forced-epoll 三后端 × 常规 + WSL TSan 零 race 全绿 |

> **范围**：HTTP/1.1 skeleton over co_await `TcpStream`——类 llama.cpp 推理引擎必需的 REST（北极星① 一等能力）。借纯 parser 做协议解析，自写集成层（read loop→喂 parser→出 Request→组 Response→write）跑在 fiber/proactor 上。server + client 两侧。
> **cloc**：`http.h` 175 + `http.cc` 528（+ `http_test` 158）+ 借 picohttpparser 596（C，不计入自有）。

## 为什么是它
类 llama.cpp 推理引擎必需 REST = 北极星① 一等能力。价值 + 性能是「HTTP over 自有 io_uring/shard-per-core，无第二 runtime、无阻塞线程池」：字节从不离开 fiber/proactor。

## 核心决策：三层分工（docs scope「HTTP 最终结论」）
1. **借纯 parser**：picohttpparser vendored 于 `third_party/`（596 行 C，零 IO 零线程，`phr_parse_request/response` + `phr_decode_chunked`）做协议解析。
2. **自写集成层**：`co_await TcpStream.read`→喂 parser→出 Request→组 Response→write，跑在 fiber/proactor 上（HTTP over 完成模型，非阻塞线程池）。
3. **薄模型**：`HttpRequest`/`HttpResponse`（流式 body 一等公民）。
4. **路由/handler/业务端点留上层**——这是 skeleton，不是 framework。

### 两级 API
- **低层**：`HttpServerConn::next_request()`/`send_response()` 原语，调用方驱动 keep-alive 循环。
- **高层**：`http_serve(alloc, stream, handler, ctx)` 薄封装，handler = `Task<HttpResponse>(alloc, ctx, req)`，routing 在 handler 内（llama.cpp 风格）。
- **client**：`HttpClientConn::request()`（写请求 + `phr_parse_response`）。

### 流式响应 = chunked（LLM serving 命脉）
`HttpResponse::set_stream(NextChunkFn, ctx)` 生成器回调 = SSE / token 流，走 `Transfer-Encoding: chunked`。生成器是 type-erased raw fn-ptr + ctx（同 Fiber::EntryFn / spawn_blocking），避 std::function 于热路径。

### 成员协程 operator new 陷阱（同 tls/dns/file）
成员协程 promise operator new 首参是 `this`、Allocator 落第二位不匹配 `PromiseAllocBase`。故用**自由协程**（Allocator 首参：`http_server_next`/`http_server_send`/`http_client_request`）+ 成员转发。

## 硬点：红队揪修 3 真 bug（被「小 body 单读 + handler 只读 body」掩盖）
- **C1 head/body 视图悬垂**：method/path/header view 指向 buf，body 阶段 realloc/consume 移动或覆盖 buf → UAF。修：head 拷到 Impl 独立稳定缓冲重 parse + body 拷到独立缓冲，buf 随意 consume 都安全。
- **M2 pipelined 覆盖 body**：C1 的 body 独立缓冲修法一并解决。
- **M3 OOM null 解引用**：impl_ 分配失败 valid() 没查 → valid() 加 impl_ 检查。

## 硬点：chunked 跨平台 bug 两连
- **增量解码 bug**（Linux 一次读全掩盖，Windows IOCP 分片暴露）：`phr_decode_chunked` 返 -2 时消费全部输入，原写法重喂已压缩字节 → 重写 `decode_chunked`（每轮喂 buf、拷解码前缀到 out、-2 后 buf 清零读新）。
- **chunked keep-alive bug**（2026-07-05，fce90a6）：chunked 响应后同连接下个请求解析失败——根因 `phr_decode_chunked` 默认 `consume_trailer=0` 把终止 CRLF 当尾字节返回、污染下个 keep-alive 消息；修 = 设 `dec.consume_trailer=1`（server/client 两侧共用同修）。系统排查（纯解码探针 → 单请求网络探针 → 双 chunked keep-alive 探针三步隔离）+ 回归测试 GetKeepAlive 第 5 请求 chunked-after-chunked 双向验证。

## 热路径分配消除（2026-07-05）
原本 send 每 chunk 一个 GrowBuf 帧分配 → **已修**：复用 head 的 out（每轮 size=0），M chunk M 次分配 → 0（TSan 净）。

## 对抗式自审（正确性 / 生命周期 / 平台 / TSan / 残留）
- **① 正确性 / 平台**：loopback GET / keep-alive / POST body 往返 / chunked 流式——三后端 × 常规 + WSL TSan 零 race。
- **DoS 上限**：header 64KB（`kHttpMaxHeaderBytes`）/ body 8MB（`kHttpMaxBodyBytes`）/ 64 header 槽。
- **⑤ 残留（登记，骨架限制）**：SSE 心跳 ping / 断点续传 / multipart / CORS / 100-continue / trailer / TE+CL 走私严格拒绝 / `Connection: close` 头。
- **驱动 sched-07**（HTTP 请求 fan-out 取消，✅ 已做——见 net_cancel / http_cancel_test）。

## 交付物
- `include/hyle/io/http.h`（HttpRequest / HttpResponse / StreamChunk / HttpServerConn / HttpClientConn / HttpResponseView / http_serve / HttpHandlerFn）
- `src/io/http.cc`（read loop + 集成 picohttpparser + decode_chunked + write pump）
- `third_party/picohttpparser`（vendored，596 行 C）；`tests/io/http_test.cc`

## 参考
- io-design 参考节；docs scope「HTTP 最终结论 / HTTP 专项」。picohttpparser（借纯 parser）。相关里程碑：io-04（TcpStream）、io-06（超时）、io-08（TLS）、io-09（DNS）、sched-07（请求 fan-out 取消）、io-11（HTTP/2）。详见 [performance/benchmarks-io.md](../../evidence/performance/benchmarks-io.md)。
