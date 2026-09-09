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

## Source / P0 audit（`AUDITED`）

### 路径分类

- `validate_owner()` 要求所有 session、stream 和 body 操作运行在 transport owner worker；公开头与用户指南也把
  handle 的移动、查询、操作和销毁纳入同一 owner 契约。`claim_direction()` 的全部调用只来自已通过该检查的
  `drive_until()` / `flush_once()` / `read_once()` / `try_read_once()` 路径，而且 claim 自身到 busy publication
  之间没有 suspension point。
- 除 `wake_cancelled_wait()` 外，`notify()` 的所有调用都来自 owner worker：同步 nghttp2 recv/send callback、
  owner 上的 protocol submit/consume/reset、transport completion、body handle retire 或 session close。nghttp2
  callback 只在 owner 调用 `nghttp2_session_mem_recv2()` / `nghttp2_session_mem_send2()` 时同步执行，不是第二个
  执行线程。
- 唯一合法的 cross-thread notifier 是 `CancelCallback` 调用的 `wake_cancelled_wait()`。它不修改 session
  protocol state 或 `change_epoch`，只负责让对应 operation 重新检查外部 atomic cancel predicate。
- `read_busy`、`write_busy`、`change_epoch`、`fatal_status` 和 `peer_closed` 均只在 owner worker 读写；
  `references` 仍是独立 atomic lifetime 机制，不在本轮改动。

### happens-before / missed-wake 证明

1. 同一个 Scheduler worker 同时只执行一个 fiber。owner waiter 在检查 predicate 后，
   `ConditionVariable::prepare_wait_()` 会在仍持有 session monitor 时先发布其 waiter node，随后才释放 monitor
   并 suspend；在 node 发布前另一个 owner fiber不可能运行。owner progress 因而可以先修改 owner-local state /
   `change_epoch`，再直接调用 thread-safe `ConditionVariable::notify_all()`；无 waiter 时该 primitive 已用自己的
   atomic waiter count 返回，有 waiter 时 node 已经可见。wake-before-suspend 由既有 Fiber rendezvous 处理。
2. 对 cross-thread cancel，callback 仍先取得 session monitor。若 cancel 发生在注册前，
   `CancelCallback` 构造会同步调用 callback，随后 predicate 看见 cancelled；若发生在 predicate 检查前，检查直接
   停止等待；若发生在检查后，callback 必须等 waiter 在 monitor 下完成 publication 并释放 monitor，之后的
   notify 必然命中。故三种顺序均不丢 wake。
3. `wait_for_change()` 在释放 session monitor 后才析构 callback；`CancelCallback::unregister()` 与 callback
   执行由 cancel tree 的 recursive control lock 串行化，因此栈上的 `CancelWakeContext` 不会在返回后被调用。
4. owner progress 与 cancel 可以并发调用 `ConditionVariable::notify_all()`；该 primitive 自己以内部锁和 waiter
   winner 状态仲裁。cancel 不再伪造 `change_epoch`，只会让所有被唤醒 fiber 按各自 predicate 重检，属于 Mesa
   monitor 的既有允许语义。

### 同合同复核

v1 peer 两侧的 body byte 数、receive-window credit、固定 64x1 KiB 与 4x256 KiB workload、socket sizing、
validation、teardown、owner lifetime、双顺序、31-run precision 和固定 5% NI margin 全部不变。本轮不得通过
缩短 body、放宽 window、去掉 HYLE lifecycle 或替 peer 预分配额外状态把苹果与橘子重新比较。

### Frozen `KEEP/FIX`

- `KEEP-01`：保留 session monitor，专门关闭 cross-thread cancel 的 predicate-check / waiter-publication 窗口；
  保留 `ConditionVariable`、`change_epoch` 和所有 deadline/cancel/owner/public semantics。
- `FIX-01`（本轮唯一产品修正）：把现有通知拆成 owner-local progress 与 external-cancel wake。owner progress
  不再取得 session monitor；direction claim/release 和 epoch snapshot 改为 owner-local 访问，release 仍在需要时
  发布 epoch 并调用 CV。external cancel 仍在 session monitor 下 notify。不得添加 public 或 private waiter
  counter、allocation、syscall、spin、额外 yield 或第二项机制改写。

## Predeclared validation

- correctness：HTTP/2 主套件、small-socket duplex、sink/reset/GOAWAY/deadline、TLS ALPN HTTP/2；
- race / memory：WSL ASan、UBSan、TSan（host mapping 需要时使用已有 `setarch -R` 环境修复）；
- portability：W1-WIN/MSVC、W1-WSL/GCC 14、M1-MAC/AppleClang 21 定向 factor；
- delivery：产品依赖/API 未变时复核 exact factor 与 source/static/shared link closure；不重跑无影响的全库矩阵；
- red-team：missed wake、cancel callback race、destroy/in-flight、false fast path、spin 和苹果/橘子计费；
- performance：candidate clean、peer clean、双 order、controls、固定 5% margin、唯一 P1 bundle。

## Product correction and candidate freeze

`FIX-01` 已在 `4060e217c88d0ed35356b1ee21e5aefbe0c5b5d3` 实现，产品 tree 为
`cb91d5cba24706ddee390b51dab40fa12b2093ff`：

- `notify_owner()` 只更新 owner-affine epoch 并直接唤醒 CV；`claim_direction()`、`release_direction()` 和
  epoch snapshot 不再取得 session monitor；
- 唯一 cross-thread `CancelCallback` 改走 `notify_cancelled_wait()`，仍在 session monitor 下关闭
  predicate-check / waiter-publication 窗口，且不伪造 protocol epoch；
- 新增同 session、同 owner direction contender 被外部 OS thread cancel 的集成用例，连续 100 次通过；
  没有新增公开 API、layout、metadata、allocation、syscall、spin 或第二项机制修正。

精确候选 archive SHA-256 为
`73AE8B3A53797B41A841D176DFA7170138FE2B3C678B7A5F1387C7CF21CB1D2D`。受影响门结果：

- W1-WIN / MSVC 19.50：plain HTTP/2 主套件与新增 cancel contender 用例通过；本机未安装可发现的
  OpenSSL，TLS 路线不冒充已运行；
- W1-WSL / GCC 14.2：plain/TLS HTTP/2、forced-epoll duplex、ASan、UBSan、TSan（既有
  `setarch x86_64 -R` host mapping 修复）全部通过；
- MacBook Air `Mac16,12` 与 Mac mini `Mac16,10` / AppleClang 21：TLS 与 HTTP/2 定向套件全部通过；
  两台机器只作 portability/correctness，不混作性能样本；
- WSL exact factor 的 shared 与 static 测试均通过；source archive 在 WSL 与两台 Mac 的 hash 一致。
  Windows TLS 的环境缺口由 WSL 与两台 Mac 的真实 TLS 路线覆盖，不把 N/A 写成 PASS。

## Current decision

`CANDIDATE`。不可变产品身份如上；correctness、race/memory、portability 和 delivery 均已通过。下一步只可
保存并复核这一个候选已经产生的唯一 P1 bundle，再完成终局只读红队；不得修改产品、合同或追加样本。
