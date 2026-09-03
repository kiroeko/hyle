# IO v1：异步 IO core 因子化与收口

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本卡是现有 `io` component 的当前活动轮。它先冻结最终 ownership 和审计预算；在进入 `AUDITED` 并
> 冻结有限 FIX 集以前，不修改产品源码。TLS、HTTP/1、HTTP/2 与 transfer 分别使用已登记的 factor ID
> 和独立 closure card，本卡不替它们签署功能或性能结论。

## Closure card（`SCOPED`）

- component / owner / round ID：`io` / IO maintainers / `io-v1-core-factorization`；
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
- 范围：把现有 `io` 最终收束为 scheduler-bound async IO core，拥有 completion/backend 内部机制、
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
  `9b95b250cc64af11b84fa5ea3d677f76b37fdef0`；Linux kernel/io_uring、Windows IOCP 与 POSIX socket/file
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
| `proactor.*`, capabilities, `io_context.*` | `io` | Proactor currently carries internal backend tokens and is consumed by IoContext; audit public-vs-internal status |
| `net*`, `tcp.*`, `udp.*`, `file.*`, `dns.*`, `signal.*` | `io` | same scheduler/cancel/completion lifecycle; audit endpoint, error, local IPC, vector/direct/bulk gaps |
| `tls.*` | `io_tls` | external provider/config/session lifecycle and feature cost are independent |
| `http.*` + picohttpparser integration | `io_http` | protocol/client/server/body lifecycle distinct from transport mechanism |
| `http2.*` + nghttp2 integration | `io_http2` | connection/stream/flow-control lifecycle and optional provider are distinct |
| reliable remote/model transfer (currently absent) | `io_transfer` | retry/resume/checksum/temp-file policy is a workflow above protocol transport |

本轮允许与当前正在提取的 factor 同时处于活动状态，仅为一次 ownership move：本卡授权从旧 `io`
ownership/target 删除，新 factor 的独立 `AUDITED` 卡授权新增 owner/API/target；不得让两个 factor 同时
声称同一实现或 performance authority。`io` 的 API/usability freeze 必须先完成，随后按
`io_tls → io_http → io_http2 → io_transfer` 处理；最后 `io` 才绑定 candidate 并裁决。

## Audit queue（进入 `AUDITED` 前必须清零）

- 全部 15 个公开头与 11 个实现文件双向 trace，删除 benchmark/history-only 形状；
- 对 llama.cpp、随仓/standalone ggml 的 file/network/server/remote journey 建 demand map；
- 对 Asio/Seastar 的 address/socket/file/cancel/buffer/backend tuning 逐族比较，不按 API 名求并集；
- 审计 bool/null/negative result、跨平台 native error、late completion 与 buffer/handle lifetime；
- 审计 typed endpoint、socket options、resolver cancel/Happy Eyeballs、vectored/bulk/direct IO、local stream、
  operation budget/backpressure 和 io_uring tuning；
- 冻结普通调用草图、guide 选型和 primary/holdout 的完整语义等价矩阵；
- 生成有限 `KEEP/FIX/BORROW/DEFER/REJECT` 集，未完成项不得进入实现。

## Current decision

`SCOPED`。本卡尚未声明任何 capability `COMPLETE`、API `FROZEN`、candidate 或性能结论；下一步是只读
审计并进入 `AUDITED`，不是修改产品代码。
