# IO HTTP/2 v2：owner progress 常态成本恢复轮

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round：`io_http2` / IO maintainers /
  `io-http2-v2-owner-progress-cost`；kind=`RESUME`；baseline product candidate=
  `705b21112af34765227a132c1bd1a2e84a82e655`；
- previous terminal record：[`io-http2-v1`](io-http2-v1.md)。其 31-run P1 的 4x256 KiB
  candidate-first advantage 为 0.96586、98.93% CI `[0.95165, 0.97344]`，跨过固定 5% NI 下界
  0.95238，故为 `UNRESOLVED`；
- demand / normal journey：同一 owner worker 上的多个 HTTP/2 stream 通过有界 reader/writer 推进；没有同
  session waiter 的常态路径不应为不存在的竞争无条件支付 atomic monitor/notification，但 deadline、cancel、
  真并发 stream 和方向互斥不能丢 wake；
- scope：只审计并在结构证明成立时修正 `Http2SessionState` 的 owner-affine direction claim/release 与
  waiter notification；公开 API、layout contract、protocol state、nghttp2 路由和功能集合不变；
- non-goals：新 HTTP/2 能力，窗口/chunk/body ownership 改动，Sched/io_core 修改，公共验证工具、合同、
  margin、样本计划或 benchmark workload 修改，TLS/HTTP/transfer 重构，C/Native FFI；
- dependency / peer：产品依赖不变；nghttp2
  `85e300c79fb6dbcfa9c1013215c8710c1c2cd3d2` 仍是唯一同合同 peer；
- tier / machine / instrument：`P1`；`W1-WSL` + `closure-perf-v1` 为工程性能门；W1-WIN、W1-WSL、
  M1-MAC 承担受影响 correctness/portability；原 [`io-http2-p1-v1`](../../../../bench/contracts/io-http2-p1-v1.json)
  原样重放；
- budget：一次 source/P0 机制审计；证明成立时至多一项 private correction；一个 immutable candidate；
  一次正式 P1（含合同既有精度升级，而非人工重跑）。不再追加第三项机制假说；
- stop：机制不成立、correctness/red-team 不通过或唯一 P1 为 `FAIL/INCONCLUSIVE`，立即 `BLOCKED`；全部适用
  门通过才可 `VERIFIED -> CLOSED/FROZEN`。

## Frozen invariants

1. nghttp2 和 transport 只能由 transport owner worker 驱动；任一公开操作从错误 worker 进入仍返回 typed
   failure。
2. read/write direction 同时至多一个驱动者；真正等待方向释放、protocol state、deadline 或 cancel 的 fiber
   必须无 lost wake。
3. cancel callback 可以从非 owner 线程触发；不能把所有通知错误地当成 owner-local，也不能 busy-spin。
4. 无 waiter 的普通 progress 不新增 allocation、syscall、lock、context switch 或常态 public metadata。
5. v1 的 API、能力、错误、stream-terminal sink、GOAWAY/RST/flow-control 和交付合同保持逐字节语义。

## Audit queue（进入 `AUDITED` 前）

- 反向追踪 `claim_direction`、`release_direction`、`notify`、`wait_for_change` 与全部调用点，分别标出 owner-only
  和 cross-thread 路径；
- 给出 monitor/epoch/waiter 的 happens-before 与 missed-wake 证明，明确何时允许无锁/无通知 fast path；
- 确认 v1 peer 两侧在 body bytes、window credit、validation、teardown 和 owner lifetime 上仍同合同；不因结果
  接近边界改 comparison class；
- 冻结 `KEEP/FIX`：若无法证明一项通用私有修正，直接终止，不写产品代码。

## Predeclared validation

- correctness：HTTP/2 主套件、small-socket duplex、sink/reset/GOAWAY/deadline、TLS ALPN HTTP/2；
- race / memory：WSL ASan、UBSan、TSan（host mapping 需要时使用已有 `setarch -R` 环境修复）；
- portability：W1-WIN/MSVC、W1-WSL/GCC 14、M1-MAC/AppleClang 21 定向 factor；
- delivery：产品依赖/API 未变时复核 exact factor 与 source/static/shared link closure；不重跑无影响的全库矩阵；
- red-team：missed wake、cancel callback race、destroy/in-flight、false fast path、spin 和苹果/橘子计费；
- performance：candidate clean、peer clean、双 order、controls、固定 5% margin、唯一 P1 bundle。

## Current decision

`SCOPED`。本提交只建立有限恢复合同，不修改产品。下一步只可完成 audit queue；结构证明成立后才进入
`AUDITED` 并冻结唯一 FIX。
