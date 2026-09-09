# IO TLS v2：流控协议下的全双工进展修复

<!-- MODULE CLOSURE RECORD: CLOSED -->

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

## Candidate readiness（`CANDIDATE`）

产品纠正在 `18bf62f81a91975e0776439d88fca3891f10e560` 完成，产品 tree 为
`98632557e0237bac0bed3ca5521e59420d89176e`；精确 Git archive 的 SHA-256 为
`329DA82D0438AA24ACB90A36DE397475A16A8FABAB65E6040F3DC7DABDE4E3DB`。本轮没有修改公开对象模型、
TLS 配置、provider 选择、factor 依赖或 benchmark 合同：

- `flush_ciphertext()` 每次循环先检查同一绝对 deadline/cancel budget；底层 write 立即 would-block 时，
  才能从 peer 吸收至多一个 16 KiB chunk；read BIO 中尚未被 provider 消费的 opportunistic ciphertext
  硬限为 32 KiB，达到上限便回到原 awaitable，不 spin、不新建后台 task；
- `tls_write()` 明确恢复 `TransferResult` 的 write-some 合同：一次 provider 成功只提交一个至多 16 KiB
  TLS record，HTTP/HTTP2 与普通调用者按返回的 `bytes` 推进。由此一个任意大的 plaintext span 不再让单次
  operation 反复吸收无界 peer ciphertext；
- scratch 仍为 stream 构造时一次分配的固定 16 KiB；普通无背压路线没有新增 allocation、虚分派、公开
  metadata 或 syscall。write BIO 的 ciphertext 只在 native send 已确认后消费，read BIO 只在
  `SSL_ERROR_WANT_READ` 且 provider 已耗尽既有输入后补充；
- 新增 64 KiB plaintext 的 partial-progress 回归：第一次成功写只返回一个 record，peer 精确解密返回的
  byte 数；原 small-socket HTTP/2 flow-control、pre-cancel 零 wire、partial/EOF/truncation、ALPN/session、
  deadline/cancel 与 shutdown 路线全部回放通过。

### Correctness、portability 与 delivery

| profile / shape | result |
|---|---|
| W1-WIN / MSVC 19.50 / shared | `io_tls`、`io_http2` 与新增 partial-progress 回归 `PASS` |
| W1-WSL / GCC 14.2 / exact archive shared | 默认 backend 与 `HYLE_IO_FORCE_EPOLL=1` 的 `io_tls`、`io_http2` 均 `PASS` |
| W1-WSL / GCC 14.2 / sanitizer | ASan、UBSan、TSan（既有 `setarch x86_64 -R` host mapping 修复）定向 TLS 套件均 `PASS` |
| MacBook Air `Mac16,12` / AppleClang 21 | exact commit shared `io_tls`、`io_http2` `PASS` |
| Mac mini `Mac16,10` / AppleClang 21 | exact archive shared `io_tls`、`io_http2` `PASS` |
| Windows exact archive / source embedding | downstream 只链接 `hyle::io`，configure/build/run `PASS` |
| Windows exact archive / installed shared + static | 两个 downstream 均只链接 `hyle::io`，configure/build/run `PASS`；static OpenSSL closure 由 package 传递发现 |
| W1-WSL / exact archive / `HYLE_IO_TLS=OFF` | 配置明确报告 dependency probe skipped，不生成 TLS binary；底层 `io_core` build/test `PASS` |

两台 Mac 只签 portability/correctness，不混入 P1 数值；Homebrew OpenSSL 的 deployment-target warning 属于
既有验证环境差异，真实 TLS runtime 全部通过。Windows、WSL 与 Mac 使用的产品内容均由同一 commit 或上述
同 SHA archive 得到；远程副本没有反向覆盖 canonical Windows checkout。

### Bounded red-team receipt

- 修复前的 duplex deadlock 是同一 owner operation 在底层 send pressure 下拒绝读取 peer flow-control
  ciphertext；修复后的 deterministic 4 KiB socket / 65,535-byte window / 256 KiB H2 用例在默认与 forced
  epoll 均前进并完成；
- 红队发现第一版 correction 会在一次巨大 `tls_write` 内随每次 output progress 重置 opportunistic-read
  机会，从而允许 dynamic read BIO 随输入增长。最终实现用 write-some + 32 KiB pending cap 消除了该隐藏
  无界状态；这属于产品成本结构修复，不是为 benchmark 加 metadata；
- pre-cancel 在第一次 wire effect 前仍返回 typed cancellation；provider 的 private state 可能已初始化，
  但回归证明 wire 为零，失败后的 stream 仍按公开合同关闭；
- deadline/cancel 检查位于 pump 每次迭代，而不是只在 blocking await 前检查，持续 immediate progress
  不能越过绝对 budget；双向均无进展时进入既有 proactor，未发现 lost wake、busy loop、第二 owner、
  ciphertext 丢失/重复或新的 high-severity finding；
- `validation-select.py` 因本卡 baseline 覆盖 IO factorization 的较长历史而保守选择了宽闭包；它不是本轮
  产品裁决器。依法只运行卡片预声明且与 TLS pump 改动相关的 gates，没有在模块任务中改写公共仪器。

## Unique P1 receipt

冻结合同 [`io-tls-p1-v1`](../../../../bench/contracts/io-tls-p1-v1.json) 在全新的 W1-WSL EXT4 checkout、
GCC 14.2、Release/static、portable ISA 上只运行一次。candidate 与 Asio peer 都是 clean/exact；双方使用
同一 host OpenSSL、persistent single-worker async runtime、verified localhost、禁用 session/ticket，并对每个
连接完整计入 TCP accept/connect、handshake、双向 `close_notify` 和 teardown。stream workload 另对称计入
512 次 256 B request/echo，没有苹果/橘子计费：

| workload / order | HYLE / Asio op/s | advantage / 99.26% CI | exact | 5% NI |
|---|---:|---:|---|---|
| verified handshake / candidate-first | 1,120.78 / 1,026.02 | 1.09879 / [1.06150, 1.15334] | `WIN` | `PASS` |
| verified handshake / peer-first | 1,175.95 / 1,061.78 | 1.10161 / [1.07436, 1.13795] | `WIN` | `PASS` |
| 512x256 B stream / candidate-first | 214.79 / 207.07 | 1.04147 / [0.99905, 1.06048] | `INCONCLUSIVE` | `PASS` |
| 512x256 B stream / peer-first | 223.71 / 214.05 | 1.05265 / [1.00579, 1.06548] | `INCONCLUSIVE` | `PASS` |

四行 5% non-inferiority 全部通过；握手双顺序为 material `WIN`，stream 点估计偏向 HYLE 但 CI 跨过
superiority 边界，所以不冒充确定胜出。A/A、5%/10% known-delta、timer floor、artifact/build/consumer、
source/peer provenance 与 host/input/artifact stability 全部 `PASS`；n=15 已裁决，没有触发 n=31。
68-file 精确 bundle 位于
[`io-tls-p1-v2-18bf62f8`](../../../../evidence/performance/io-tls-p1-v2-18bf62f8/)；manifest、summary、
controls SHA-256 依次为
`D6B9A542FB091D3CFC221D1620F51A0FBAA63BD6FACF30C468EF0802232B5751`、
`71D09A4D8DAD0904A40C6AFE939A313EB04FE101864831C255549BE459E33C91`、
`15DB83BC0737C3DD0773D187264A33A5A57B859B0E7FB1AF35D4AB259EB2313C`。保留 live build/toolchain 的
严格 verifier 返回 `QUALIFIED`、`paths_verified=true`、`provenance=EXACT` 和 environment policy
`CURRENT`；这是工程 P1，不是 Formal 或全 TLS 场景 SOTA 声明。

## Current decision

`VERIFIED`。scope、architecture、correctness、delivery、usability/docs、red-team 与 engineering
performance 七个 gate 全部 `PASS`；Formal 独立为 `NONE`。产品与证据不再修改，下一次状态提交只允许
写入 terminal limitation、reopen trigger、LOC 与 `CLOSED/FROZEN` receipt。

## Closure decision

`io_tls` v2 为 `CLOSED/FROZEN`。本轮没有为 HTTP/2 benchmark 发明公开 fast path；它修复的是任何
single-owner、双向流控协议都会遇到的真实 TLS pump 进展和有界状态问题。公开 C++ 面仍由 typed context、
session、stream、operation budget、`Status`/`TransferResult` 组成；用户不接触 OpenSSL handle、try-pump
seam、第二套 light API、C API 或 Native FFI。

- 单个 `TlsStream` 是 single-owner；operation buffer 在 await 完成前保持借用，`TlsContext` 及其 allocator
  backing 必须活过由它创建的 stream/session；
- `tls_write` 是 write-some：成功也可能只提交一个 record，调用者必须按 `bytes` 推进；`bytes + error`
  不能从原起点重试。HTTP/HTTP2 的现有循环已经按此合同组合 exact transfer；
- opportunistic peer ciphertext 只为解除 output pressure，未处理 pending 硬限 32 KiB；provider 自身的
  certificate/session/cipher 状态仍由 OpenSSL 决定。本轮不声称所有 provider memory 都是常数；
- 工程 P1 只覆盖 W1-WSL 上 verified connection 和 512x256 B small-record stream，对手是同 OpenSSL 的
  standalone Asio async SSL；不声明 WAN、多 worker、所有 cipher/provider、所有消息分布、Formal 或完整
  HTTPS server 的普遍 SOTA；
- certificate provisioning、system trust policy、OCSP、key logging、0-RTT、DTLS、QUIC、HTTP framing、
  路由、认证和模型 schema 仍在该 factor 边界外；source/static/shared 只承诺相容 C++23 工具链，不承诺
  Native FFI、C API 或跨工具链稳定 C++ ABI。

只有以下新事实可以重开：具名 HTTPS/HTTP2 consumer 复现 correctness/security/identity/truncation/
deadline/cancel/session/context lifetime/pump progress/partial-transfer/API 易用性缺陷；真实 consumer 证明缺少
通用 TLS 能力而非应用策略；公开 API、layout、algorithm、allocation/BIO bound、owner、dependency、OpenSSL
语义或 source/static/shared 交付发生变化；或者合格同合同证据证明 material regression、成熟 peer 证明更优
复杂度档。普通 benchmark 波动、未限定比较或想增加第二套 surface 都不是 reopen trigger。

候选 `18bf62f81a91975e0776439d88fca3891f10e560` 的
`loc-report.py --module io_tls --commit ... --self-check` 为 `PASS`：全库 522,049 行，自实现不含测试
89,877 行，`io_tls` 实现 987 行。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
