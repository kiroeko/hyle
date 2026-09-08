# IO TLS v2：流控协议下的全双工进展修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen record（`REOPENED`）

- component / owner / round ID：`io_tls` / IO maintainers /
  `io-tls-v2-duplex-progress`；task kind：`REOPEN`；baseline：
  `79848f76946ac661abb221a537e71e4d70abdb1e`；
- trigger：`IOTLS-REOPEN-H2-EPOLL-DUPLEX-01`。`io_http2` 的确定性 TLS ALPN
  consumer 使用一条 worker、4 KiB socket buffers、65,535-byte connection receive
  window 和 256 KiB request/response。在 W1-WSL 默认 io_uring 下 12 ms 完成；强制
  epoll 时稳定在 server 已接受 request head、只交付 32 KiB body，而 client 尚未完成
  request submission；
- affected surface：OpenSSL memory-BIO 与 `TcpStream` 之间的 ciphertext pump，以及只供协议
  factor 使用的 immediate-progress detail seam。公开 TLS 配置、身份验证、ALPN、session reuse、
  shutdown 和普通 awaitable API 没有被该事实推翻；
- evidence：`tests/io_tls/tls_test.cc` 的
  `Http2TlsDuplexProgressSurvivesSmallSocketAndFlowControl`；连续复现的 syscall 末端是两端均已
  登记 `EPOLLOUT|EPOLLONESHOT`，client request ciphertext 占满一向缓冲，而 server 的
  WINDOW_UPDATE ciphertext 无法写出。小记录 TLS 和 HTTP/1 在同一 forced-epoll backend 通过，
  所以缺陷限定在需要读写交错才能前进的 TLS consumer；
- immediate decision：该事实命中 `io-tls-v1` 的第一个 reopen trigger。依法移除旧
  `FROZEN` confirmation 并进入 `REOPENED`；在完成新的 closure card 和有限审计前不修改
  TLS 产品实现。

## Historical boundary

`io-tls-v1` 对配置、验证、ALPN、session、deadline/cancel、truncation、普通小记录路线以及三种
C++ 交付的结论仍是历史证据。新轮不得借机增加 QUIC/DTLS、0-RTT、证书配置策略、公开 OpenSSL
handle、Native FFI/C API，或按 benchmark 发明第二套 TLS stream。
