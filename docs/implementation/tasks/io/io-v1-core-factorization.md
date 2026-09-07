# IO Core v1：异步传输因子化与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本卡是现有 `io_core` factor 的当前活动轮。其 round ID 在身份纠正前已冻结为
> `io-v1-core-factorization`；component 已通过
> [`io-core-factor-identity-migration-v1`](../build/io-core-factor-identity-migration-v1.md)
> 从含混的 `io` 纠正为 `io_core`。公共产品仍是 `hyle::io`，公开 namespace/include 仍是
> `hyle::io` / `<hyle/io/...>`。本卡先冻结最终 ownership 和审计预算；在进入 `AUDITED` 并
> 冻结有限 FIX 集以前，不修改产品源码。TLS、HTTP/1、HTTP/2 与 transfer 分别使用已登记的 factor ID
> 和独立 closure card，本卡不替它们签署功能或性能结论。

## Closure card（`SCOPED`）

- component / owner / round ID：`io_core` / IO maintainers / `io-v1-core-factorization`；
- task kind：`REFACTOR`；
- baseline commit：`2e4b0606b08f57ac15a3717f8b0e9be82db16123`；
- 进入原因与 demand/defect ID：`IO-V1-OWNERSHIP-01`、`IO-V1-CAPABILITY-02`、
  `IO-V1-NATURAL-API-03`。现有 target 把五个不同 lifecycle/可选成本族放在一起；公开 raw Proactor
  仍暴露 bool/negative errno/user-data/backend token 等内部机制形状；旧能力账本还显示 endpoint、
  cancellation、vectored/direct IO 与 local IPC 等待审缺口；
- 真实 consumer task / normal workload：北极星量化推理引擎以 bounded preallocated buffers 异步读取
  大模型文件，并通过多连接 TCP/local IPC 服务请求；普通 HYLE 服务端需要 DNS、signal-driven shutdown、
  deadline/cancel 与明确 backpressure。llama.cpp remote/server/model-load 路径只作真实形状输入，不把其
  同步 API 原样复制到本模块；
- 范围：把现有聚合实现最终收束为 scheduler-bound `io_core`，拥有 completion/backend 内部机制、
  `IoContext`、file、TCP/UDP、DNS、signal、local IPC 及其通用 endpoint/error/cancel/deadline/buffering
  合同；反向追踪全部现有公开入口；决定 raw Proactor 是完整 expert API 还是 internal seam；冻结自然
  C++ API、复杂度与成本；授权后续从旧 `io` target 移除由其他 IO factor 接管的源码/头/依赖；
- 非目标：本卡不设计或验证 TLS、HTTP/1、HTTP/2、remote/model-transfer 语义；不修改 Sched/Time/Mem/
  Platform 的已关闭产品，除非审计形成具名 correctness/security 触发；不修改 Compute/Graph；不建立
  Native FFI/C API；不按旧 benchmark 名称补入口；不承诺跨平台/全操作 SOTA；
- 直接依赖 / 直接消费者：当前 manifest 为 `core,mem,container,time,timer_queue,sched`；`error`、
  `platform_fs`、`platform_thread` 是否应成为直接边由 public/source trace 决定。当前 HYLE 直接消费者
  只有 Sched 的 build-only poll hook；未来 `io_tls/io_http/io_http2/io_transfer` 是预期直接消费者；
- tier 与理由：`P1`。submission、completion routing、buffer ownership 和 hot-file/network paths 可能
  进入 token 服务热路；系统调用本身不豁免错误复杂度、allocation/lock/wake 或额外 copy；
- material margin / resource budget：同合同 normal workload 采用 5% non-inferiority margin；同时冻结
  per-op allocation、copy、submission、syscall、wake/context-switch、queue depth 与 working-set 账本；
- 总调查预算 / 性能预算：总计最多 8 engineering hours；其中 peer/API/correctness 审计 5 小时，P1
  资格化与最多两次机制诊断 3 小时；
- 最多复现、机制诊断、产品修改次数：每个失败 shape 一次确认复现；全轮最多两项机制诊断、两次有
  normal-consumer material 证据的算法修改；超限进入 `BLOCKED/INCONCLUSIVE`，不扩建公共仪器；
- machine profile：correctness/delivery 为 `W1-WIN/W1-WSL/M1-MAC`；P1 primary 为 `W1-WSL`；Windows
  IOCP 与 macOS kqueue 是 backend correctness/cost holdout，除非另有合格同合同性能任务；
- instrument profile：`closure-correctness-v1` + `closure-perf-v1`；
- strong peers：Asio `8806a6803cde7054c3049d3666d3ec36786568c5` 与 Seastar
  `78f64e2317b19c7e34ba61f5e8ea85f083eca1da`；Linux kernel/io_uring、Windows IOCP 与 POSIX socket/file
  API 是定义性平台合同，不冒充第三方；
- 使用原型 provenance：primary model-file pipeline 与 loopback request path 是
  `NORTH_STAR_JOURNEY`；llama.cpp/ggml 是 `REAL_CONSUMER` reference；纯 submit/reap microcase 只能是
  `MECHANISM_DIAGNOSTIC`；
- API 语义档位与选择前提：`scheduler-bound async resources` 是普通路径；只有证明 raw completion API
  自身具备真实独立消费者、跨平台错误/lifetime 合同和三交付自然用法时才保留 public expert route，
  否则 internalize；不得创建只有“light/fast”含义的重复入口；
- 每档常态成本合同：普通 operation 借用 buffer 至最终 completion/cancel drain，不隐式复制；稳态无
  per-operation heap allocation 为目标，队列/slot 有界；一个 ready completion 至多一次 scheduler wake；
  batching 只在保持逐操作结果/cancel 语义时合并 syscall；expert seam 若保留不得增加虚分派；
- 适用 correctness / delivery / performance gates：normal/empty/boundary/OOM/invalid handle、短 IO、
  EOF、partial write、cancel-vs-complete、close-with-inflight、move/destroy、multi-worker、backend fallback、
  sanitizer、三平台、source/static/shared、直接 factor consumers、P1 primary+holdout、C++ guide、FFI
  清洁和红队；
- 预期输出：完整 capability-family map、自然 API freeze、最终 header/source ownership map、最小直接
  DAG、有限 FIX/DEFER/REJECT ledger、candidate 和 IO core terminal receipt；
- 停止条件 / 可接受限制：任何未完成适用风险门、缺真实 backend/peer、无法语义等价或超过上述预算即
  `BLOCKED/INCONCLUSIVE`。TLS/HTTP 等高层因子尚未关闭不允许本卡提前 `CLOSED`；但其协议功能不计入
  IO core 自身完整度。

## SCOPED ownership constraints

只读源码追踪已足以冻结以下施工边界，但不足以形成 `AUDITED` verdict：

| Existing family | final owner candidate | reason / next audit |
|---|---|---|
| `proactor.*`, capabilities, `io_context.*` | `io_core` | Proactor currently carries internal backend tokens and is consumed by IoContext; audit public-vs-internal status |
| `net*`, `tcp.*`, `udp.*`, `file.*`, `dns.*`, `signal.*` | `io_core` | same scheduler/cancel/completion lifecycle; audit endpoint, error, local IPC, vector/direct/bulk gaps |
| `tls.*` | `io_tls` | external provider/config/session lifecycle and feature cost are independent |
| `http.*` + picohttpparser integration | `io_http` | protocol/client/server/body lifecycle distinct from transport mechanism |
| `http2.*` + nghttp2 integration | `io_http2` | connection/stream/flow-control lifecycle and optional provider are distinct |
| reliable remote/model transfer (currently absent) | `io_transfer` | retry/resume/checksum/temp-file policy is a workflow above protocol transport |

本轮允许与当前正在提取的 factor 同时处于活动状态，仅为一次 ownership move：本卡授权从旧 `io`
ownership/target 删除，新 factor 的独立 `AUDITED` 卡授权新增 owner/API/target；不得让两个 factor 同时
声称同一实现或 performance authority。`io_core` 的 API/usability freeze 必须先完成，随后按
`io_tls → io_http → io_http2 → io_transfer` 处理；最后 `io_core` 才绑定 candidate 并裁决。

## Audit queue（进入 `AUDITED` 前必须清零）

- 全部 15 个公开头与 11 个实现文件双向 trace，删除 benchmark/history-only 形状；
- 对 llama.cpp、随仓/standalone ggml 的 file/network/server/remote journey 建 demand map；
- 对 Asio/Seastar 的 address/socket/file/cancel/buffer/backend tuning 逐族比较，不按 API 名求并集；
- 审计 bool/null/negative result、跨平台 native error、late completion 与 buffer/handle lifetime；
- 审计 typed endpoint、socket options、resolver cancel/Happy Eyeballs、vectored/bulk/direct IO、local stream、
  operation budget/backpressure 和 io_uring tuning；
- 冻结普通调用草图、guide 选型和 primary/holdout 的完整语义等价矩阵；
- 生成有限 `KEEP/FIX/BORROW/DEFER/REJECT` 集，未完成项不得进入实现。

## Resume reference lock（2026-09-08）

本轮在 canonical Windows checkout 重新锁定并扫描以下 clean reference tree；commit 只标识审计输入，
不构成 HYLE 产品依赖：

| reference | commit | 本卡使用的真实形状 |
|---|---|---|
| llama.cpp | `e71b80510c848c00175924ecf3c40333ccae8eb5` | model load/direct IO、RPC exact stream、HTTP server、SSE、Unix socket、download |
| standalone ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | framed RPC、remote transport、shared-memory/RDMA owner boundary |
| ds4 | `f62ca29a308724cde5bc99134ede19104b2a3260` | compact HTTP/SSE server、SSD/model-file access、browser WebSocket evidence |
| vLLM | `51da0ca66c8065619c79e35dff97aa99aeaf5644` | HTTP streaming/disconnect、realtime WebSocket、gRPC、KV offload/file IO |
| SGLang | `dcebe8c4733a5cb802fb679e0c8fbb886e2801b0` | HTTP/SSE disconnect、gateway pooling/timeouts/TLS、gRPC/local IPC |
| Asio | `8806a6803cde7054c3049d3666d3ec36786568c5` | endpoint、buffer sequence、resolver、socket/file cancellation contracts |
| Seastar | `78f64e2317b19c7e34ba61f5e8ea85f083eca1da` | shard-per-core transport、streams/backpressure、DMA file IO、Unix sockets |

这些消费者共同证明 HTTP(S) server/streaming 是 IO 产品必须承载的基础设施，但 application route、
OpenAI JSON schema、auth/batching/model policy 不属于 HYLE IO。它们也共同要求 base transport 先提供 typed
endpoint/error、deadline/cancel、scatter/gather、local IPC 与 bounded flow control；不能用 echo benchmark
替代这些合同。WebSocket 有 ds4/vLLM 的真实证据，但它是独立协议候选，不偷偷塞进 `io_core`。

## Capability-family coverage map（`AUDITED`）

本表把全局 `IO-A-01..54` 账本投影到 `io_core` owner；TLS/HTTP/H2/transfer 的缺口只标明路由，不由
本卡实现或签署。

| family | normal consumer task | baseline | verdict / owner |
|---|---|---|---|
| completion backend、per-worker shard、batched reap/park wake | scheduler-bound async runtime | IOCP/io_uring/epoll/kqueue 已有，稳态 completion 无逐项 heap | `COMPLETE / KEEP`，backend engine 变成 private seam |
| public backend/capability inspection | 调优、诊断与 fallback | 数据与 `Proactor` raw object 绑定 | `PARTIAL / FIX`，保留 typed snapshot，隐藏 submit token |
| owning IPv4/IPv6 endpoint | parse/format/compare/local-peer/DNS/connect/UDP | `void*+len`、128-byte array 与 bool parser 混用 | `MISSING / FIX` |
| portable failure and partial-transfer result | setup、IO、EOF、cancel、deadline、native diagnosis | invalid object/bool/null/negative errno 混用 | `MISSING / FIX`，普通面接 Error；raw integer 仅 private backend |
| one-shot TCP/UDP/File | borrowed buffer、partial、EOF、RAII owner | 四 backend 基础机制已有 | `COMPLETE / KEEP`，在新 typed facade 下复用 |
| one operation budget | base/deadline/cancel 与二者组合 | 每种操作复制三套入口，TLS/HTTP 无法继承 | `PARTIAL / FIX`；一个 options/budget 形状，默认腿无额外分配 |
| TCP setup/options/inspection | bind/listen/connect/half-close/keepalive/reuse-port | 基础 setup 完整，错误与选项过窄 | `PARTIAL / FIX` |
| async resolver + hostname connect | DNS、总期限、IPv6/IPv4 fallback | blocking pool + 固定 8 地址；无取消/Happy Eyeballs | `PARTIAL / FIX` |
| scatter/gather stream/datagram/file IO | framed RPC header+payload、分段 buffer | 单 buffer only | `MISSING / FIX` |
| local stream/listener | 同机 inference frontend/backend | 无 | `MISSING / FIX`，POSIX Unix domain + Windows named pipe/真实等价腿 |
| advanced Linux registered/provided-buffer route | 高吞吐 receive/fixed files | raw io_uring 扩展机制完整 | `COMPLETE / BORROW`，重塑为明确 expert extension；不污染 portable API |
| positional/direct model-file IO | large model read、alignment、queue depth | positional/metadata/fallback 有；direct/alignment 无，native owner 与 Platform 重复 | `PARTIAL / FIX`，adopt `platform_fs::FileHandle` owner，显式 direct policy |
| async file stream/read-ahead/watcher/pipe | 顺序流、reload、child-process pipe | 无 | `DEFER`：没有本轮 owner-ready dependency；分别由后续 consumer card 准入 |
| signal-driven shutdown | SIGINT/SIGTERM/Ctrl-C -> normal completion | single process waiter 已有 | `COMPLETE / KEEP`，补 typed failure/lifetime negative tests |
| plain file-to-socket/sendfile | static/model serving | 无 | `ROUTE io_transfer`，TLS fallback 与 transfer policy 不进 core |
| TLS / HTTP1 / HTTP2 / reliable download | HTTPS、REST/SSE、multiplex、model fetch | 当前代码仍错误同 target | `ROUTE io_tls/io_http/io_http2/io_transfer` |
| QUIC/gRPC/RDMA/WebSocket | future protocol/transport | 无统一 owner | `DEFER/REJECT`：只保留真实 consumer watchlist；不得在 core 手写协议 |

### Public/source reverse trace

| existing surface | audit verdict |
|---|---|
| `capabilities.h` | `KEEP/FIX`：只拥有产品与 backend typed facts |
| `io_context.h` | `KEEP/FIX`：自然 options、status 与 capability 查询；不返回 public `Proactor&` |
| `tcp.h`, `udp.h`, `file.h`, `dns.h`, `signal.h` | `KEEP/FIX`：成为普通 typed API；不泄 raw sockaddr、负 native code 或 backend slot |
| `proactor.h`, `net.h`, `net_cancel.h`, `net_timeout.h` | `INTERNALIZE`：completion token、tagged stack/heap slot 与 raw submit 是实现机制 |
| `net_multishot.h` | portable raw token 部分 `INTERNALIZE`；经 capability 成功取得的 io_uring accelerator 重塑为 explicit expert header |
| `coroutine_transport.h` | `REJECT` 独立用户概念：它是 package/target compatibility fact，不应要求普通用户选择一个机制 header |
| `tls.h`, `http.h`, `http2.h` | `MOVE` 到各自 factor；公开 namespace 仍为 `hyle::io`，用户不看内部 factor 名 |
| 11 个 `src/io_core/*.cc` 与 pico/nghttp2/OpenSSL | core 只留下 backend/context/file/net/DNS/signal；协议源码与第三方 owner 随 factor 移走 |

反向扫描未找到 HYLE 产品模块直接消费 raw Proactor/negative errno；当前直接 product consumer 只通过
Sched poll hook。因而 internalize 不会迫使已冻结下游迁移，也不是为 benchmark 删除正常能力。

## API/usability and cost freeze

- 用户只认识 `hyle::io` 产品、namespace 与 `<hyle/io/...>` 头；`io_core` 只在 component graph、target 和
  维护文档出现。以后所有内部子因子也遵守该规则，禁止再与产品根同名；
- 普通网络地址是 owning typed endpoint，支持 IPv4/IPv6 parse/format/equality/port；native sockaddr view
  只在 implementation boundary。local endpoint 是独立类型，不能用一个 family flag 假装 Windows pipe
  与 Unix path 生命周期相同；
- setup 返回 `Result<Resource>`；一次传输返回同时保留 bytes、EOF 与 `Status` 的 typed outcome。错误附带
  portable `StatusCode` 和正确的 Posix/Win32/WinSock native domain；用户不比较 magic negative integer；
- `OperationOptions` 统一 absolute monotonic deadline 与 optional `CancelToken`。无 options 的普通 awaitable
  保持 inline、无额外 heap/虚分派；带 deadline/cancel 的慢腿允许使用 IoContext 注入 allocator，但一个
  operation 仍至多一次 completion wake，late completion 不触碰已释放 frame；
- scatter/gather 接受 borrowed span-of-spans，返回跨 segment 的精确 progress；提交期间 descriptor 与 bytes
  均须存活，超过平台 segment/byte limit 返回 typed error而非截断；
- resolver 返回 owning endpoint sequence，不固定截断为 8；hostname connect 用一个总 budget 按 RFC 8305
  思路错峰尝试，失败关闭所有 loser，成功只留下一个 owner；
- File 继续是唯一 async owner，但从/向 `platform_fs::FileHandle` 显式 adopt/release，避免两个 RAII owner。
  Direct I/O 必须显式请求、查询/返回 alignment，不支持时诚实失败，绝不静默回 buffered；
- `Proactor`、`Completion`、raw fd/user_data 和 backend-specific negative result 变为 implementation detail。
  Linux acceleration 只经显式 capability/extension handle 暴露，portable 调用不存在恒失败 stub；
- P0 常态成本：single-buffer base op 无逐项 allocation/copy/lock；one ready completion 至多一次 wake；每
  worker 一个 owner ring；scatter/gather 一次 native submission；DNS blocking fallback 不占 worker；bounded
  queue/buffer pressure 必须可观察。P1 只在语义等价的 loopback/framed-RPC/model-file journeys 上裁决，
  不拿 raw submit 对完整 typed lifecycle，也不以地址/顺序翻转点值改 API。

## Owner 七问直答（`AUDITED`）

1. **第三方路径：** core backend 自实现 OS API；没有运行时第三方。pico、OpenSSL、nghttp2 必须随协议
   factor 移出。Asio/Seastar 与四组引擎只作证据。
2. **功能完整度：** 当前机制底座成立，但 typed error/endpoint、Happy Eyeballs、vectored/direct IO 与
   local IPC 是已证实的 `MUST`，未实现前不能关闭；高级 UDP、watcher、pipe/RDMA 暂无本轮准入。
3. **现有 HYLE 复用：** Error、Time、Platform FS、Mem、Container buffer 与 Sched cancellation/waker 应
   成为真实 direct edge；不得复制 native file owner、time scalar、queue 或 error taxonomy。
4. **性能：** backend 机制已有结构优势，但旧 microbench 不能证明产品 SOTA。先锁定零分配/copy/wake/
   syscall 账本，再对 framed RPC、model file 与 server loop 做公平 P1；系统调用延迟不是忽略用户态税的理由。
5. **Native FFI：** 当前无具名 Java/C# consumer，N/A；不恢复旧 C facade，也不导出 awaitable/template。
6. **Build/delivery：** factor 名已纠正，但物理 target 仍错误含三个协议和三类依赖；必须拆 target/option/
   package owner，并验证 source/static/shared 与 exact-factor consumer。
7. **易用性/文档：** 当前 guide 教用户理解 Proactor、raw sockaddr 和负数，是实现泄漏；需按“何时用
   TCP/UDP/local/file、何时加 budget、何时选择协议 factor”重写，并清除旧 C ABI/benchmark 结论。

## Frozen finite FIX set（`AUDITED`）

| ID | exact authorized work | required gate |
|---|---|---|
| `IOCORE-F01` | 把 raw completion/proactor/net slot internalize；公开 capability 经 IoContext，保留无恒失败 stub 的 Linux expert accelerator | header self-containment、raw symbol/include negative scan、base cost/codegen |
| `IOCORE-F02` | 接入 Error，建立 typed endpoint、native error mapping、setup/transfer outcome | IPv4/6/invalid/native-domain/EOF/partial/OOM tests |
| `IOCORE-F03` | 用一个 operation-options 合同覆盖 base/deadline/cancel 与组合 race；默认腿保持零分配 | cancel-before/during/after、deadline race、late completion、ASan/TSan |
| `IOCORE-F04` | 完成 socket option/endpoint inspection、owning resolver 与 Happy-Eyeballs hostname connect | dual-stack、port 0、reuse-port、all-fail cleanup、total budget |
| `IOCORE-F05` | 增加 portable scatter/gather TCP/UDP/File operation 与 progress cursor | segment/byte limits、partial progress、one native submit、three backends |
| `IOCORE-F06` | 增加 `LocalListener/LocalStream`；POSIX Unix socket 与 Windows named-pipe 是各自真实 lifecycle | path/cleanup/permissions/connect/cancel/close、三平台 |
| `IOCORE-F07` | File 与 Platform owner 互操作，增加显式 Direct I/O/alignment 与 large positional chunking | buffered/direct unsupported/alignment/tail/short IO、model-file journey |
| `IOCORE-F08` | 修正 IoContext/Signal teardown、allocator/queue depth/backpressure 与 invalid construction 状态 | init rollback、inflight destroy、repeated signal/move、resource ledger |
| `IOCORE-F09` | 建立真正的 `io_core` source/header/target；协议与依赖由各自合法 card 接走 | exact factor、source/static/shared、feature-off/no probe、offline package |
| `IOCORE-F10` | 重写 architecture/user/evidence，运行限定 correctness、sanitizer、P1 primary+holdout 和红队 | no apple-vs-orange、docs/closure checker、terminal receipt |

本轮产品修改只允许上述十项；HTTP body/router/server、TLS config、HTTP/2 stream 与 download 分别进入其
自己的 card。若某项需要重开已 FROZEN dependency、发明公共 scheduler select 或超过两次机制诊断，本卡
按合同进入 `BLOCKED`，不能在 IO 里偷偷实现第二份。

`SCOPED -> AUDITED` 的前态 registry snapshot 位于
`C:\public\project-hyle\_validation\io-core-v1\registry-before-audited.json`，SHA-256
`1C5F42AB5840889B9E943FC7E98ED5CBB97B4268F746F1690899FCA44580AB83`。

## Current decision

`AUDITED / ACTIVE`。identity、能力超集、自然 API、P0 成本、ownership 与有限 FIX 集已冻结。下一步只可
按 `IOCORE-F01..F10` 实施；任何新 protocol/API/benchmark idea 都不得追加进本候选。

## Implementation checkpoints

- `415e54ed`：完成 `io_core` 内部 factor 身份迁移，公共产品仍保持 `hyle::io`；
- `58ddd108`：完成 typed endpoint、portable/native error、transfer outcome 与统一
  deadline/cancel transport 合同；
- `8a68f175`：完成 File/Platform FS 单一 owner 互操作、typed positional/exact IO、
  显式 direct-I/O policy/alignment 与 preallocate 修正，并通过 Windows、Linux、macOS
  的 `io_core` correctness 回归；
- `ce2da848`：SignalSet 改为 typed factory/result，pending 按 signal kind 合并并
  保留不同种类；重复等待、move、重复 owner、非法输入与宿主 handler 恢复已在
  Windows IOCP、Linux io_uring/epoll 构建和 macOS kqueue 路径通过测试；
- `007c14ca`：`IoContext` 从 Scheduler 推导 worker 数量，移除用户可见的
  benchmark/backend knob，并把 ring owner table 改为注入 allocator 管理的连续
  存储；Windows 全 IO 与 Linux/macOS 定向回归通过；
- `30512b94`：resolver 改为 owning、可 deadline/cancel 的单状态 offload，
  hostname/endpoint-sequence connect 增加有界 RFC 8305-style 错峰、快速失败推进与
  loser drain；补齐 idle/interval/count keepalive、endpoint inspection、reuse-port
  的 typed 合同。DNS/TCP 定向测试在 Windows/MSVC、WSL/GCC 与 macOS/AppleClang
  全部通过；超时/取消使用被占满的单线程 blocking pool 确定性验证提前返回后状态
  安全退休。
- `ea1fb927`：完成 `IOCORE-F05` 的 portable buffer-sequence 形状与 TCP、UDP、
  positional File scatter/gather 路径。描述符在调用时复制，字节内存借用至原生
  completion 退休；64 段/2 GiB 边界返回 typed error，partial cursor 可精确续传。
  Windows socket 走单次 `WSARecv/WSASend`，文件走一次 allocator-backed aligned
  staging 加单次 `ReadFile/WriteFile`；Linux/macOS 分别走原生或 readiness-backed
  `readv/writev/preadv/pwritev/recvmsg/sendmsg`。UDP 现在显式报告报文尾部丢弃，
  bounded cancel/deadline 在返回用户前等待原操作退休，避免 late-completion 触碰
  已释放 buffer。buffer、TCP、UDP、File 与 timeout 定向测试已在 Windows IOCP、
  Linux io_uring 和 macOS kqueue 通过。

以上只证明对应 FIX 子项的实现 checkpoint，不是 candidate、P1 或 `CLOSED/FROZEN`
裁决；F01、F06、F08 的剩余部分以及 F09--F10 仍未完成，四个协议/传输 factor
仍处于 `QUEUED`。
