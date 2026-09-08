# IO HTTP/1 v1：有界 streaming client/server 因子化与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

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

## SCOPED audit queue

- 反向追踪 `http.h/http.cc/http*_test.cc` 与 pico package owner；
- 建立 head、fixed/bounded body、streaming body、client/server、connection lifecycle 能力图；
- 比较四组推理真实旅程与 Beast/Seastar，先冻结 natural C++ journey 和负空间；
- 冻结有限 FIX 集后才允许迁移源码或改变公开 API。

## Current decision

`SCOPED / ACTIVE`。本卡只授权审计；尚未进入 `AUDITED`。
