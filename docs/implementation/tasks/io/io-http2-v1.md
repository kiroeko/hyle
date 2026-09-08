# IO HTTP/2 v1：多 stream 与流控因子化收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

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

## SCOPED audit queue

- 反向追踪 `http2.h/http2.cc/http2_test.cc`、nghttp2 adapter 与 package metadata；
- 用四组推理旅程裁决 incremental body、cancel、GOAWAY、limits 和 gRPC owner；
- 冻结 feature-off/ON target 与 HTTP/TLS interop DAG；
- 冻结有限 FIX 集后才允许迁移或重写产品源码。

## Current decision

`SCOPED / ACTIVE`。本卡只授权审计；尚未进入 `AUDITED`。
