# IO HTTP v2：协议 transport 双向进展修复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`AUDITED`）

- component / owner / round ID：`io_http` / IO maintainers /
  `io-http-v2-duplex-progress`；task kind：`REOPEN`；baseline：
  `720c055b4e4bcf98556641acb77f82ad689c10cc`；
- trigger：`IOHTTP-REOPEN-H2-DUPLEX-01`。冻结在
  [`io-http2-p1-v1`](../../../../bench/contracts/io-http2-p1-v1.json) 的同合同
  4 KiB socket-buffer、双向 HTTP/2 flow-control holdout，在
  `HYLE_IO_FORCE_EPOLL=1 HYLE_BENCH_SMOKE=1` 下可重复停止进展；系统调用轨迹证明
  双端各有待写数据时，现有协议驱动只等待可写而没有继续吸收已到达输入；
- affected surface：`HttpTransport` 给上层协议使用的 transport-neutral 组合缝，以及它对
  `io_core`/可选 `io_tls` 的即时进展表达。HTTP/1 framing、message/body API、parser、路由和
  应用策略不在受影响范围；
- evidence：候选 `3bd9c9391efcacf63185c5f1c73d4d9351f51fdf` 加冻结 fixture
  `720c055b4e4bcf98556641acb77f82ad689c10cc`；W1-WSL 默认 io_uring 正确完成，强制 epoll
  在 holdout 稳定挂起，末端轨迹为两端 `EPOLLOUT|EPOLLONESHOT`，而对应 peer receive queue
  已有未消费字节。这满足旧 `io_http` 登记的“HTTP2 consumer 复现 backpressure/teardown
  defect” reopen trigger；
- scope：为协议因子增加一个 internal、transport-neutral 的 immediate-read attempt。它一次最多
  吸收一个已经到达的 bounded chunk，以独立 `would_block` 状态区分“现在无数据”和 EOF/error；
  不等待、不分配、不吞掉 deadline/cancel，也不公开 native/provider handle。`io_http2` 在准备阻塞
  写入前用它维持 full-duplex progress；plain TCP owner 在 `io_core`，TLS provider owner 若由确定性
  复现证明受影响则依法单独重开；
- non-goals：改变 HTTP/1 parser/framing/message/body API，给 ordinary 用户公开 reactor syscall，
  新建后台 scheduler/线程、无界读写队列，修改 nghttp2/flow-control 语义，或增加应用 server 策略；
- tier / gates：P0 先证明 nonblocking、EOF/error、单 reader/owner 与 no-allocation；correctness 固定覆盖
  4 KiB socket buffer + 双向 flow-control、默认 io_uring 与强制 epoll，并补 h2c/TLS 分类。P1 只回放
  已冻结 HTTP/1 两条 workload 与 HTTP/2 normal/holdout，不改变 peer、socket/window/body/order；
- budget：3 engineering hours；只允许一次 transport progress 产品机制纠正和一次受影响 provider
  纠正；性能最多一次机制检查。超过后按真实剩余问题 CLOSED 或 BLOCKED；
- machines / delivery：W1-WIN、W1-WSL、M1-MAC；source/static/shared。sanitizer 只跑受影响的
  io_core/io_tls/io_http/io_http2 exact factors；
- strong references：Boost.Asio、libuv/nginx 的 optimistic nonblocking receive + readiness fallback
  只作为机制参照；nghttp2 仍是协议 authority。不能把直接 busy polling 或 blocking raw loop 当成
  HYLE scheduler 路线；
- stop：确定性双向进展与旧 HTTP/1 语义/P1 回放全部通过后关闭；若 TLS 需要无法在本轮边界内表达的
  provider engine 重构，`io_http` 不冒充完成，登记精确 io_tls blocker/trigger。

## Decision invariants

- immediate attempt 只在 transport owner worker、且没有另一 read operation 在途时执行；上层仍保持
  每方向最多一项 native operation；
- `would_block` 不是错误，EOF 与部分 bytes + terminal status 不得丢失；任何已吸收 bytes 必须在本次
  调用返回前进入协议 parser；
- optimistic path 与普通 awaitable path 使用同一 native error 映射；Windows 不假设 adopted socket
  已处于 nonblocking mode；
- HTTP/2 只调用组合缝，不复制 POSIX/WinSock/OpenSSL 分支。API/layout/普通 HTTP/1 路线不因 benchmark
  增加 metadata。

## Capability / public-surface audit

| family | existing shape | finding / action |
|---|---|---|
| TCP immediate progress | `TcpStream::read_some()` 总是登记 completion 后 suspend | 协议 owner 无法在阻塞写前吸收已经到达的 bytes；增加 private detail helper，ordinary C++ API 不扩面 |
| TLS immediate progress | memory-BIO pump 只提供 awaitable read/write | 先以小缓冲 H2 TLS 确定性复现；未复现则本轮只记录 `UNSUPPORTED`，复现则由 `io_tls` 自己 reopen，禁止在 HTTP 复制 OpenSSL |
| transport erasure | `HttpTransport::read_some/write_some` | 缺 immediate outcome；增加 detail dispatch，不改变对象 layout、公开 message API 或 provider handle |
| HTTP/2 driver | `drive_until` 先完整 flush、再 blocking read | 双向 socket 背压可形成 write/write stall；在没有另一 read in flight 时先处理一个 bounded immediate read |
| HTTP/1 | 单 owner、顺序 request/response streaming | 没有新能力缺口；只做受影响 correctness/P1 replay，不重构 framing |

现有 public surface 没有 `try_*`、native provider 或 benchmark-only knob；本轮仍保持这一点。即时尝试是
factor 间 detail contract：TCP 系统调用与 error mapping 仍由 `io_core` 唯一实现，type erasure 仍由
`io_http` 实现，HTTP/2 只消费，不跨 owner 复制代码。

## Frozen finite FIX set（`AUDITED`）

| ID | exact work | required gate |
|---|---|---|
| `IOHTTP2P-F01` | 在 io_core detail 定义 allocation-free immediate receive outcome 与跨平台 TCP 实现；POSIX 使用 nonblocking recv，Windows 先零等待 readiness、不得假设 socket mode | ready/would-block/EOF/error/partial、owner/single-reader、三平台 |
| `IOHTTP2P-F02` | `HttpTransport` 增加 internal dispatch；HTTP/2 driver 在阻塞写前 bounded drain，并复用现有 parser/error/flow-control | 4 KiB h2c 双向 holdout，io_uring + forced epoll，无 busy loop/hidden queue |
| `IOHTTP2P-F03` | 增加确定性 regression，并对 TLS ALPN h2 路径做同形复现；若命中旧 io_tls trigger，独立 reopen 后再实现 provider owner 修复 | h2c/TLS、default/fallback、deadline/cancel/teardown |
| `IOHTTP2P-F04` | 回放 io_core/io_http/io_http2 correctness、sanitizer、source/static/shared，以及原 HTTP/1 与冻结 HTTP/2 P1；更新 guide/receipt/registry | 所有受影响门，无 workload 漂移 |

FIX 集到此冻结。不得把 public `try_read`、background thread、第二 scheduler、socket busy-poll、HTTP/2
专有字段或新 benchmark shape 加入本轮。

## Historical boundary

`io-http-v1` 的 HTTP/1 能力、三平台/三交付和 P1 receipt 仍是历史证据，不因重开而改写。
新一轮不得借机重做 parser、增加 server framework、公开 native provider handle，或把 HTTP/2
frame/flow-control 语义下沉到 `io_http`。

## Candidate readiness（`CANDIDATE`）

有限 FIX 集的产品形状在 `18bf62f81a91975e0776439d88fca3891f10e560` 完整，tree 为
`98632557e0237bac0bed3ca5521e59420d89176e`；精确 Git archive SHA-256 为
`329DA82D0438AA24ACB90A36DE397475A16A8FABAB65E6040F3DC7DABDE4E3DB`。本轮没有重写 HTTP/1
parser/message/body surface：

- `io_core` 的 internal TCP attempt 在 owner worker 上一次执行一个 allocation-free nonblocking
  `recv/send`；POSIX 使用 `MSG_DONTWAIT`，Windows 对可能由 host adopt 的 blocking socket 先做零等待
  `WSAPoll`，随后在 single-reader/single-writer 条件下调用 WinSock；
- `ImmediateTransferResult` 显式区分 ready、would-block 与 provider unsupported；ready 中继续完整保留
  partial bytes、clean EOF 和 typed terminal `Status`。invalid transport 仍是 typed
  `FailedPrecondition`，不会伪装成 would-block；
- `io_http` 只负责 `HttpTransport` 的 type-erased dispatch。plain TCP 走上述 immediate seam；TLS 不复制
  OpenSSL，而是返回 unsupported，由已冻结 `io_tls` 的普通 awaitable write/pump 保证双向进展；
- `io_http2` 在阻塞 write 前至多吸收一个已经到达的 bounded input chunk；一次 cooperative yield 后仍
  无即时进展便回到 proactor。parser、flow-control、deadline/cancel 和 owner 仍由原 authority 负责，
  没有 spin、后台 task、无界 queue 或常态 metadata；
- ordinary HTTP/1 read/write 不调用 immediate seam，因此没有新增 syscall、allocation、branch 或对象
  layout 成本；新增函数只存在于 `detail` factor-composition 边界，没有 public `try_*` 或第二套 light API。

### Correctness、sanitizer 与 delivery

| profile / shape | result |
|---|---|
| W1-WIN / MSVC 19.50 / shared | TCP immediate、HTTP/1 主/取消与 HTTP/2 套件 `PASS` |
| W1-WSL / GCC 14.2 / exact archive shared | 上述套件在默认 io_uring 与 forced epoll 均 `PASS`；独立 deterministic epoll duplex regression `PASS` |
| W1-WSL / exact archive | ASan、UBSan、TSan（`setarch x86_64 -R`）下 TCP/TLS/HTTP/HTTP2 五项均 `PASS` |
| MacBook Air `Mac16,12` / AppleClang 21 | exact commit TCP/HTTP/HTTP2 定向套件 `PASS` |
| Mac mini `Mac16,10` / AppleClang 21 | exact archive TCP/HTTP/HTTP2 定向套件 `PASS` |
| Windows exact archive / source embedding | downstream 只链接 `hyle::io` 并实际调用 typed HTTP factory，configure/build/run `PASS` |
| Windows exact archive / installed shared + static | 两个 downstream 只链接 `hyle::io` 并实际调用 typed HTTP factory，configure/build/run `PASS` |

`HYLE_IO_TLS=OFF` 的 HTTP source/static/shared 交付已在本轮 seam 首次落地的 `e201f86f` 精确矩阵通过；
该提交之后 `src/io_http` 零变化，`include/hyle/io/http.h` 只增加 `Http2SessionAccess` 的 private friend，
不改变 TLS-off API/layout/编译路线。当前 exact archive 的 TLS-on 三交付与两种 transport correctness 又全部
通过，故不重复制造一套无影响的 SDK build。

### Bounded red-team receipt

- Windows readiness→`recv/send` 之间没有 suspension point；detail 合同又禁止同方向第二 operation，因此
  不存在另一个 consumer 抢走 readiness 后令 adopted blocking socket 卡住的窗口；POSIX syscall 自身带
  `MSG_DONTWAIT`；
- empty buffer、would-block、EOF、partial+error 和 invalid transport 均有独立语义；HTTP/2 只在 owner
  校验后持有 direction claim 的区间调用，临时 claim 内没有 suspension；
- TLS 不支持 OS-level immediate attempt 是刻意的 provider boundary，而非能力缺口：已冻结 pump 在
  output pressure 下有界吸收 peer ciphertext。HTTP 不接触 native/OpenSSL handle，也不复制 deadline；
- `4060e217` 的 HTTP2 P1 候选到本候选之间，plain HTTP2 合同涉及的 `io_core/io_http/io_http2` 产品、fixture
  与合同逐字节无差异；其 normal/holdout 双顺序 5% P1 可以直接复用，不应重新采样；
- 未发现 lost bytes、重复消费、busy loop、unbounded state、owner race、公开 ABI 污染、伪 C ABI、
  HTTP/1 正常路线成本或其他 high-severity finding。

## Current decision

`VERIFIED`。有限 FIX、P0、三平台 correctness、exact sanitizer、source/static/shared、TLS/h2c 分类、
易用性/文档与红队均通过；plain HTTP2 的已冻结 P1 由逐字节等价直接继承。不可变候选在 W1-WSL clean
EXT4 上完成唯一一次 [`io-http-p1-v1`](../../../../bench/contracts/io-http-p1-v1.json) 回放：1 KiB
REST/stream 双顺序为 `PARITY`，64 KiB fragmented 双顺序为约 3.03x/3.07x `WIN`，所有 5% 非劣门与
controls 为 `PASS`。严格 verifier 返回 `QUALIFIED`、`paths_verified=true`、`provenance=EXACT`；完整
receipt 见 [`benchmarks-io.md`](../../evidence/performance/benchmarks-io.md) 和
[`io-http-p1-v2-18bf62f8`](../../../../evidence/performance/io-http-p1-v2-18bf62f8/)。七门已有可复核证据，
下一法定动作仅为终局限制、重开条件、LOC 与冻结确认，不再运行产品测试或 P1。
