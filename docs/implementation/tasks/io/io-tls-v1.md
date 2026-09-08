# IO TLS v1：加密 transport 因子化与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`AUDITED`）

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

## Completed audit inputs

- 反向追踪 `tls.h/tls.cc/tls_test.cc` 全部公开与 private provider seam；
- 对四组锁定推理证据、Asio/Seastar 的 config/session/stream/close 合同建立能力图；
- 裁决 feature-off header/target、OpenSSL package ownership、HTTP/2 ALPN 依赖方向；
- 冻结有限 `KEEP/FIX/BORROW/DEFER/REJECT` 集后才允许移动或重写产品源码。

## Capability-family coverage map（`AUDITED`）

| family | admitted journey / lifecycle | baseline | coverage / action |
|---|---|---|---|
| provider/config owner | client trust、server cert/key、protocol floor、verify policy | invalid-object factories；native pointer public | `PARTIAL / FIX` |
| encrypted stream construction | move TCP owner，SNI + hostname/IP identity，server/client role | constructor can only become invalid | `PARTIAL / FIX` |
| handshake/read/write | partial transfer、EOF、provider/native failure | `Task<int/long>`，无 operation budget | `PARTIAL / FIX` |
| deadline/cancel/close | whole operation budget、late completion retirement、close-notify/truncation | absent / best-effort int | `MISSING / FIX` |
| ALPN | generic protocol list/result，HTTP/2 convenience | only `enable_http2(bool)` | `PARTIAL / FIX` |
| session resumption | export/import reusable client session without exposing OpenSSL types | absent | `MISSING / FIX` |
| advanced provider controls | custom BIO、early data、OCSP stapling、keylog | no admitted normal journey | `GENERAL / DEFER` |

四组 inference references all require HTTPS transport but do not require certificate issuance or a TLS
framework. Asio and Seastar both expose config/session separate from socket ownership, async handshake/
shutdown, peer verification and cancellation. This admits the first six families; provider-specific knobs
remain explicit future triggers.

### Public/source reverse trace and owner decision

- `tls.h` exports raw `ssl_ctx_st* native()` and invalid factories, while `tls.cc` owns OpenSSL memory BIO;
  raw provider pointer has no current consumer and is internalized. A deliberate native interop view is deferred;
- `tls_handshake/read/write/shutdown` conflate EOF, protocol failure, syscall failure and byte count, and cannot
  inherit `OperationOptions`; all four are `PARTIAL`, not compatibility constraints;
- `TlsStream` correctly owns one `TcpStream` and one per-connection provider session; one owner fiber remains the
  concurrency contract. Scratch reuse is a valid local mechanism: OpenSSL record pumping needs contiguous mutable
  storage and Mem has no same-contract protocol buffer owner;
- semantic owner is `io_tls`; OpenSSL owns crypto/protocol mechanism; Build/package owns discovery/export/license;
  `io_core` owns TCP completion and Error/Time own result/deadline taxonomy. There is one backend decision authority.

## Frozen finite FIX set（`AUDITED`）

| ID | exact work | required gate |
|---|---|---|
| `IOTLS-F01` | 建立 `io_tls` target/header/source/test ownership、独立 export macro 与 OpenSSL package owner；feature OFF 不探测 | exact factor、source/static/shared、OFF/ON、symbol/header scan |
| `IOTLS-F02` | 用 typed client/server options 与 `Result<TlsContext>` factory 取代 invalid factory/native pointer；通用 ALPN list + HTTP/2 convenience | trust/cert/key/protocol/ALPN invalid-boundary tests |
| `IOTLS-F03` | 用 `Result<TlsStream>` construction、`Status` handshake/shutdown、`TransferResult` read/write 与统一 `OperationOptions` 取代 int/long | EOF/partial/provider+native domain/deadline/cancel/late completion |
| `IOTLS-F04` | 增加 move-only provider-opaque session ticket owner 与显式 client resume route；明确 close-notify truncation | resumed handshake、wrong context/host、move/destroy、session lifetime |
| `IOTLS-F05` | guide/architecture、三平台 TLS-on correctness/sanitizer、三交付、normal record+handshake P1 与红队 | no apple-vs-orange、terminal receipt |
| `IOTLS-F06` | 回放 Sched v5 completion-search 修复，重新冻结包含已修依赖的产品树；TLS API/provider/pump 不扩张 | Sched candidate identity、三平台定向 correctness、唯一正式 W1-WSL P1 重跑 |

不保留旧签名、`native()` 或伪 C ABI；HTTP framing、early data、OCSP/keylog/DTLS/QUIC 不得追加到本轮。

## API / usability freeze

```cpp
auto client = TlsContext::make_client(TlsClientOptions{...});
auto server = TlsContext::make_server(TlsServerOptions{...});
auto stream = TlsStream::connect(std::move(tcp), *client,
                                 {.server_name = "api.example.com"});
Status ready = co_await tls_handshake(alloc, *stream, operation);
TransferResult n = co_await tls_read(alloc, *stream, bytes, operation);
Status closed = co_await tls_shutdown(alloc, *stream, operation);
```

Options are copied at construction; certificate path/string ownership is documented per field. Buffers are borrowed
until await completion. Session tickets are owning values tied to the provider identity but expose no OpenSSL type.
状态：`FROZEN`。plain TLS、mTLS、ALPN、resume、deadline/cancel、strict truncation 与 move/lifetime
journey 均已进入候选；公开形状冻结于 `228c73a687efe34318afc6b399e45edd9acdd055`。context options
显式持有 per-connection scratch allocator，默认 system allocator；该 backing 必须活过 context 产生的
全部 stream。operation 首参 allocator 只拥有 coroutine frame，不会被 stream 隐式保存。session cache/
ticket 默认关闭，只有显式打开才产生 provider 保留成本。

## Owner 七问（AUDITED）

1. Third-party: direct product mechanism OpenSSL, exact installed identity supplied by the selected environment;
   Asio/Seastar and inference engines are peer/reference only.
2. Capability: six admitted families above have four material partial/missing edges, all in F01--F04; no unresolved
   `MUST` may remain at candidate.
3. Reuse: TCP/OperationOptions/Error/Time/Mem/Sched are reused; local memory-BIO pump and scratch remain `BORROW/LOCAL`
   because importing Asio/Seastar would add a second runtime and cannot share HYLE completion ownership.
4. Performance: P1; current historical TLS benchmark is diagnostic only. Frozen record-stream and handshake journeys
   need a same-provider comparison or consumer budget; no module-wide SOTA claim.
5. FFI: `N/A`; scan must remain free of C headers/exports/managed sample.
6. Delivery: F01 ownership landed in `ea45d866`; source/static/shared/offline terminal journeys remain pending.
7. C++ usability/docs: currently `FAIL` due invalid objects, raw integers, missing budget/session and stale guide;
   F02--F05 own the repair.

## Current decision

`AUDITED / ACTIVE`。原 `228c73a6` TLS API/provider 候选本身保持不变，但其产品树包含的 Sched
依赖已被合格 P1 证明有 material completion-latency 缺陷，故依法退回并冻结 `IOTLS-F06`。TLS 不得
借此扩张公开合同或修改 provider pump；只有 Sched v5 候选验证通过后才能形成新的整体 candidate。

## Implementation checkpoints

- `ea45d866`：`IOTLS-F01` 的物理 owner 已建立；TLS-enabled source、test、benchmark、OpenSSL
  package owner 与 `HYLE_IO_TLS_API` 均属于 `io_tls`。feature OFF 只保留 capability query，
  不暴露失败 operation；Windows OFF/ON shared correctness 与独立 header 已通过。
- `IOTLS-F02--F04 implementation candidate`：公开面已删除 invalid factory、raw
  `ssl_ctx_st* native()` 与 `int/long` operation；client/server typed options、
  `Result<TlsContext/TlsStream>`、`Status/TransferResult`、一个贯穿 pump 的
  `OperationOptions`、通用 ALPN、context-bound opaque session、显式 shutdown 与
  strict/opt-in-truncated EOF 已落地。context 以引用所有权活过 handle，stream scratch
  仍只在首次 pump 分配一次。Windows OpenSSL 3.6.3 shared TLS-on 10/10 通过，覆盖
  invalid config、verifying identity、deadline、pre-cancel、context lifetime、真实 TLS 1.2
  session reuse、wrong-context rejection、missing close-notify、HTTPS 与 H2；TLS-off target
  也通过。该结果尚不是 terminal receipt：F05 的三平台、source/static/shared、sanitizer、
  immutable-candidate P1、红队与最终文档仍待完成。
- `228c73a687efe34318afc6b399e45edd9acdd055`（tree
  `28b25ab2a0ada318fff9c8b89a2df591e594e78e`）：API/usability candidate 冻结。补齐 client
  certificate/private-key route，使 server-side client verification 成为完整 mTLS；session cache 改为
  显式 opt-in；每 stream 16 KiB scratch 在 typed construction 中从 context 的 allocator 分配一次，
  不再把 coroutine allocator 的生命周期偷偷延长；同时修复 `bytes + EOF` 最后一段 ciphertext 的处理。
  Windows shared/static、WSL shared、macOS shared exact-factor correctness 均通过；F05 后续门不再授权产品改动。
- `IOTLS-F05 delivery checkpoint`：TLS-enabled Windows installed-package consumer 已分别从
  `hyle::io` 的 shared 与 static 安装树完成 configure/build/run；sample 还实际构造 typed client
  context，因而不是只验证链接器能找到空壳 target。exact-factor source journey 已在 Windows、WSL
  与 macOS 覆盖；合格 P1、红队和最终 relocation receipt 仍待签发，不能据此提前写成
  `VERIFIED`。
- `IOTLS-F05 sanitizer checkpoint`：冻结 candidate 的 WSL GCC 14 static exact-factor 分别在
  ASan、UBSan 与 TSan 下通过全部 `io_tls` test。TSan 首次启动遇到 GCC/WSL 的
  `unexpected memory mapping` 环境失格；按既有 Linux 合同经 `setarch x86_64 -R` 关闭子进程
  ASLR 后复跑通过，未报告 data race，不能把首次环境失败隐去或误记成产品失败。
- `IOTLS-F05 P1 fixture freeze`：旧 blocking OpenSSL 行已明确降为 opt-in
  `MECHANISM_DIAGNOSTIC / NOT_COMPARABLE`；它不再默认执行，也不能签产品结论。新的
  [`io-tls-p1-v1`](../../../../bench/contracts/io-tls-p1-v1.json) 只比较持久单 worker runtime 上的
  HYLE public `spawn_task` + typed TLS 自然路线与 standalone Asio async SSL 自然路线；两侧使用同一
  OpenSSL、verified localhost、禁用 session/ticket，完整计 TCP connect/accept、handshake、精确
  512x256 B 双向 stream、`close_notify` 和 connection teardown。有限 5% margin、正反序与 workload
  identity 已在采样前冻结。正式 W1-WSL qualification 的 A/A、known delta、timer floor、artifact、
  build/product provenance 与正反序均通过，但 handshake advantage 为 `0.448/0.446`、encrypted-stream
  advantage 为 `0.0265/0.0259`，5% 非劣门全部失败。一次机制诊断将差距定位到冻结依赖 Sched 的
  Balanced idle search：同路线改用 `kPowerSaving` 后分别变为 `0.783 ms` 与 `3.94 ms`，同次 Asio
  为 `0.768 ms` 与 `4.26 ms`。因此这是合法的 Sched reopen trigger，不是 TLS benchmark 调参；
  `io_tls` 已退回 `AUDITED`，待新 Sched candidate 后只重跑原冻结 workload。

本次 registry 叙述校准的前态快照位于
`C:\public\project-hyle\_validation\io-tls-v1\registry-before-f02-f04.json`，SHA-256
`5F81C4D85139187BB2F1AAFCEE8BCD12B42B2EA6062200CF831A9D51DE482EDC`。
