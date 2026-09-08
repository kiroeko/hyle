# IO TLS v2：流控协议下的全双工进展修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

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
- scope：只修复 TLS ciphertext pump 在底层 write pressure 下无法吸收已到达 peer ciphertext 的
  进展缺陷，并让 `io_http` 的 internal transport seam 能在不泄漏 OpenSSL/native handle 的前提下
  消费这一能力；保持公开 TLS 配置、stream、结果和 operation-budget 形状不变；
- non-goals：公开 `try_read/try_write`，第二套 fast/light TLS API，后台 thread/scheduler，busy poll，
  无界 ciphertext/plaintext queue，HTTP/2 flow-control/parser 改写，或 QUIC/DTLS/0-RTT 扩面；
- direct dependencies / consumers：实现仅复用 `io_core` 的 typed `TcpStream`、统一 error/budget 和
  internal immediate transfer primitive；直接 consumer 是 `io_http` 的 type erasure，具名复现者是
  `io_http2`。OpenSSL 仍是唯一 provider authority；
- tier / cost：`P1`。P0 必须证明 single-owner、bounded scratch/BIO、无 busy loop、无新增普通路径
  allocation、deadline/cancel 不被 opportunistic progress 重置；P1 只回放冻结的 TLS small-record
  natural routes，不因本缺陷发明吞吐 benchmark；5% material margin 不变；
- correctness / delivery gates：确定性 TLS+h2 4 KiB/65,535/256 KiB 用例在默认 backend 与 Linux
  forced epoll 下通过；回放 handshake/read/write/shutdown、partial/EOF/truncation、deadline/cancel、
  session/ALPN、HTTP/1+HTTP/2 integration、ASan/UBSan/TSan，以及 W1-WIN/W1-WSL/M1-MAC 的
  source/static/shared exact-factor consumer；
- investigation / correction budget：最多 2 次机制诊断、1 次 TLS provider 产品纠正和 1 次相邻
  `io_core` correctness 修正；性能只允许一次冻结 P1 replay。当前 syscall trace 已计为第一次机制诊断；
- strong references：OpenSSL memory-BIO/`SSL_get_error` 合同与 Boost.Asio SSL engine pump；两者只用于
  验证 provider state-machine 与 buffer ownership，不把异步框架表面复制进 HYLE；
- stop：修复后上述确定性进展、旧 TLS 语义和冻结 P1 全部通过则进入 candidate；若必须改变公开 API、
  增加无界常态资源，或超过纠正预算仍不能维持进展，则按精确 blocker 终止。

## Decision invariants

- `TlsStream` 继续是 single-owner transport；同一时刻只有一个 TLS operation 使用共享 pump scratch。
  opportunistic receive 只能在底层 write 立即报告 would-block 时发生；
- 已接收 ciphertext 必须进入该 stream 的 read BIO，不能被 HTTP/2 owner旁路解析；pending write BIO
  数据在确认 native send 完成前不得被丢弃或重复；
- 每次 pressure 处理最多吸收一个 `kPumpBytes` chunk，随后重新尝试既有输出；若双向都 would-block，
  回到统一带 deadline/cancel 的 awaitable，不自旋；
- 普通无背压路线不得新增 heap allocation、公开字段、虚分派或 syscall。`io_core` 仍唯一负责 POSIX/
  WinSock error 与 SIGPIPE-safe socket send，TLS 不复制 OS 分支。

## Historical boundary

`io-tls-v1` 对配置、验证、ALPN、session、deadline/cancel、truncation、普通小记录路线以及三种
C++ 交付的结论仍是历史证据。新轮不得借机增加 QUIC/DTLS、0-RTT、证书配置策略、公开 OpenSSL
handle、Native FFI/C API，或按 benchmark 发明第二套 TLS stream。
