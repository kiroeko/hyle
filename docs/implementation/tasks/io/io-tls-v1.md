# IO TLS v1：加密 transport 因子化与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round ID：`io_tls` / IO maintainers / `io-tls-v1`；task kind：`NEW`；
  baseline commit：`e1499bb404e1a69881fb1c4d91c60aa183d1d1a7`；
- demand：`IOTLS-OWNERSHIP-01`、`IOTLS-LIFECYCLE-02`、`IOTLS-BUDGET-03`。现有
  `TlsContext/TlsStream` 与 OpenSSL 依赖错误归 `io_core`，公开操作返回 `int/long`、
  无统一 `OperationOptions`，session/ALPN/config 生命周期不完整；
- normal journey：推理服务对外建立 verifying HTTPS client 或有证书的 TLS server，握手、
  有界读写、正常 close-notify，并在 disconnect/cancel/deadline 下排空底层 completion；
- scope：TLS provider/config/context/session、client/server handshake、SNI/identity verification、
  ALPN、session reuse、typed encrypted stream IO、deadline/cancel/shutdown 与 provider error mapping；
- non-goals：HTTP framing、route/auth/model schema、HTTP/2 stream、QUIC/DTLS、证书自动签发、
  system trust-store provisioning、Native FFI/C API、稳定 C++ ABI；
- direct dependencies / consumers：候选 `PUBLIC io_core,error,mem,container_text,time,sched`；
  OpenSSL 是 private mechanism；`io_http`/`io_http2` 是预期消费者。最终边只由 public/source trace 决定；
- tier：`P1`；normal TLS record stream 与 handshake resource budget，same-contract margin 5%。
  稳态不为每次 pump 分配 scratch，不复制已加密/解密字节超过 provider 必需次数；一个底层
  completion 至多一次 wake；
- budget：总 6 engineering hours，审计/API/correctness 4 小时，P1 与至多一次机制诊断 2 小时；
  candidate 只跑一次完整 P1 bundle；
- machines / instruments：`W1-WIN/W1-WSL/M1-MAC`，`closure-correctness-v1` +
  `closure-perf-v1`；TLS-enabled 三交付必须使用锁定 OpenSSL identity；
- strong peers：Asio SSL（Asio `8806a6803cde7054c3049d3666d3ec36786568c5`）与
  Seastar TLS（`78f64e2317b19c7e34ba61f5e8ea85f083eca1da`）；OpenSSL 是 provider，
  不是同合同 HYLE transport peer；
- provenance：llama.cpp HTTPS download/server、SGLang gateway TLS 是 north-star journey；
  ds4/vLLM 的 HTTPS/WSS 边界用于负空间，不把 Python server policy 下沉；
- API freeze 前必须明确：context/session owner、trust/cert input、buffer borrow、one-reader/
  one-writer 或 single-owner contract、handshake/read/write/shutdown 的 typed result、总 deadline/cancel、
  close-notify truncation policy、ALPN/session cache 的 opt-in 成本；
- gates：能力/public trace、header self-containment、TLS off 不探测、TLS on 三平台 correctness、
  invalid cert/host/truncation/cancel race/lifetime、source/static/shared + relocation、P1、guide、红队；
- stop：任何 provider identity、平台 TLS 环境或同合同 P1 无法在预算内资格化时明确 `BLOCKED`；
  不允许用无验证 stub、跳过证书验证或 benchmark-only buffer API取得绿色结果。

## SCOPED audit queue

- 反向追踪 `tls.h/tls.cc/tls_test.cc` 全部公开与 private provider seam；
- 对四组锁定推理证据、Asio/Seastar 的 config/session/stream/close 合同建立能力图；
- 裁决 feature-off header/target、OpenSSL package ownership、HTTP/2 ALPN 依赖方向；
- 冻结有限 `KEEP/FIX/BORROW/DEFER/REJECT` 集后才允许移动或重写产品源码。

## Current decision

`SCOPED / ACTIVE`。本卡只授权审计；尚未进入 `AUDITED`，不得修改 TLS 产品实现或物理 owner。
