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
- 每次 pressure 处理最多吸收一个 `kPumpBytes` chunk，read BIO 的 opportunistic 未处理输入总量硬限为
  `2 * kPumpBytes`；随后重新尝试既有输出，若双向都 would-block，回到统一带 deadline/cancel 的
  awaitable，不自旋；
- `tls_write` 保持 `TransferResult` 的 write-some 语义：启用 provider partial-write 后一次成功只提交
  一个 record，完整发送由既有 HTTP/HTTP2 或调用者循环组合；不得在一次 operation 内循环全部输入并
  借机累积无界 peer ciphertext；
- 普通无背压路线不得新增 heap allocation、公开字段、虚分派或 syscall。`io_core` 仍唯一负责 POSIX/
  WinSock error 与 SIGPIPE-safe socket send，TLS 不复制 OS 分支。

## Capability and surface audit（`AUDITED`）

| family | current coverage | decision |
|---|---|---|
| context/config、peer identity、mTLS、ALPN | `COMPLETE`，本次复现未击穿 | `KEEP` 原 API/实现 |
| handshake、session reuse、clean/truncated shutdown | `COMPLETE`，小记录 forced-epoll 路线通过 | `KEEP` 并回放 |
| single-owner encrypted read/write + one absolute budget | 普通路线 `COMPLETE`；flow-controlled protocol 在 output pressure 下不能读入 peer control record | `PARTIAL / FIX` provider pump |
| public nonblocking TLS primitive | 没有独立 consumer；会泄漏 provider state 与误导普通调用者 | `REJECT`，只保留 internal composition |
| concurrent public reader + writer | 当前共享 scratch 和 user guide 均明确 single-owner | `OUT_OF_SCOPE`，不为修复一个 single-owner protocol 改对象模型 |
| HTTP/2 frame/window semantics | 由 `io_http2` + nghttp2 完整拥有 | `ROUTED`，TLS 只搬运 ciphertext |
| socket peer-close signal/error | forced-epoll 小 H2 teardown 还暴露 Linux scalar send 可产生 SIGPIPE | `ROUTED` 到仍在 `AUDITED` 的 `io_core` `IOCORE-F02/F05` |

公开面反向追踪覆盖 `TlsClientOptions`、`TlsServerOptions`、`TlsContext`、`TlsSession`、
`TlsConnectOptions`、`TlsStream` 与四个 operation。全部仍对应 v1 已准入的配置、owner、session 或
encrypted-operation journey；没有 benchmark-only symbol，也没有理由改变名称、默认值、layout、
failure 或 teardown。新能力只存在于 `src/io_tls` 与 factor 间 detail seam。

OpenSSL memory BIO 仍是唯一 provider state authority；pending ciphertext 在 write BIO 中拥有，peer
ciphertext 在 read BIO 中拥有。Asio 的 SSL engine 同样区分 `want_input` 与 `want_output` 并在跨
await 时保留 engine buffers；它通过可并发的独立 read/write operation 保证进展。HYLE 的公开合同是
single-owner，因此选择成本更小的 owner-local pump：只有 output 立即 would-block 时读入一块 peer
ciphertext，再重试原输出；不复制 Asio 的两个 timer/并发协调对象。

## Frozen finite FIX set（`AUDITED`）

| ID | exact authorized work | required gate |
|---|---|---|
| `IOTLS2-F01` | `flush_ciphertext` 在不先丢弃 write-BIO 数据的前提下 optimistic send；write pressure 时至多吸收一块 peer ciphertext 到 read BIO，再重试，双向均无即时进展则回到原 budgeted await | pending bytes 精确一次消费、partial/EOF/error、bounded BIO/scratch、无 spin/allocation |
| `IOTLS2-F02` | 让 `HttpTransport` 的 TLS fallback 消费修复后的普通 `tls_write`；不增加 public 或 internal TLS try API | 4 KiB TLS+h2 default/forced-epoll、HTTP/1/H2、deadline/cancel/teardown |
| `IOTLS2-F03` | 回放 TLS v1 correctness、sanitizer、source/static/shared、guide/red-team 与冻结 small-record P1 | 三平台适用门、5% NI、checker/receipt |
| `IOCORE-ADJ-01` | 在现有 `IOCORE-F02/F05` 内让 Linux scalar/vectored socket send 都以 typed EPIPE 返回而不向宿主投递 SIGPIPE | peer-close scalar/vectored，io_uring + forced epoll，不改 process signal policy |

FIX 集到此冻结。不得加入第二 scratch、公开 TLS pump 状态、HTTP/2 专用 flag、后台 progress task，
也不得因默认 io_uring 已通过而跳过 fallback correctness。

## Owner 七问（audit verdict）

1. **第三方：**产品直接使用锁定 OpenSSL；Asio 只作 peer。没有新增第三方或 provider owner。
2. **功能：**四组推理 consumer 需要 HTTPS/HTTP2 streaming；唯一被击穿的 MUST 是 TLS 下流控协议的
   single-owner progress，已进入 `F01/F02`；其余 v1 能力不扩面。
3. **内部复用：**Error/budget/TCP/allocator 均沿既有依赖复用；OS send 归 `io_core`，HTTP2 window
   归 `io_http2`，不在 TLS 重复实现。
4. **性能：**P1；普通路线保持一块连接级 scratch、无逐操作分配。candidate 后只回放已有同合同
   small-record gate，新大体量场景是 correctness，不产生 SOTA 声明。
5. **旧 ABI：**`N/A`；没有 C/Native FFI 产品面，OpenSSL C API 是 private provider boundary。
6. **构建交付：**factor/feature gate 不变；必须重放 source/static/shared 和 TLS ON/OFF exact-factor。
7. **易用性：**普通 C++ 调用不变；guide 继续说明 single-owner、context/session lifetime、budget 和
   truncation，补充流控协议无需用户手工并发 pump。

## Historical boundary

`io-tls-v1` 对配置、验证、ALPN、session、deadline/cancel、truncation、普通小记录路线以及三种
C++ 交付的结论仍是历史证据。新轮不得借机增加 QUIC/DTLS、0-RTT、证书配置策略、公开 OpenSSL
handle、Native FFI/C API，或按 benchmark 发明第二套 TLS stream。
