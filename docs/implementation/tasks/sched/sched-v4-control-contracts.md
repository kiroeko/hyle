# Sched v4 控制与结果合同

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前唯一状态真相是 [`module-status.json`](../../module-status.json)。本轮由只读
> [`Sched 当前产品回放`](sched-current-product-replay-2026-09.md) 的具体产品事实触发，不重做 v3 已经
> 关闭的 root participation 优化，也不把推理引擎策略下沉到通用调度器。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`sched` / Sched execution owner /
  `sched-v4-control-contracts` / `REOPEN`；
- baseline：`f4da6ed4a2953b183a99532f31686852147ca09e`；previous round：
  `sched-v3-root-participation`；
- trigger：`SCHED-V4-CONTROL-CONTRACT-2026-09-07`；
- reproducible facts：`JoinHandle::abort()` 只写 cooperative flag，parked task 必须由另一个 owner 唤醒；
  raw `DetachedFn/EntryFn` 接受可抛函数却在 `noexcept` trampoline 执行；Semaphore 的 `bool` 把 closed、
  unavailable 与 timeout 合并；用户/构建文档另有可定位的实现漂移；
- preliminary affected surface：`join/cancel/waker` 的控制合同、`Scheduler/Fiber` raw callback type、
  Semaphore acquire results、直接测试/consumer、public header closure、三种 C++ 交付和相关文档；
- preliminary non-goals：完整 scheduler/queue/fiber 重写；强制销毁 stack；绕开 operation owner 直接恢复
  仍挂链的 Fiber；external-caller SPMD/gang、模型/请求优先级、continuous batching、GPU/异构策略、
  Compute/Graph/IO 重构、Native FFI/C API、公共性能仪器修改、P2/Formal 和按 peer API 求并集；
- preliminary tier：`P1`，5% material margin。只有有限 `FIX`、API/usability 和 candidate 冻结后才定义
  并运行正常 workload；本状态不授权产品实现或性能采样。

下一合法状态是 `SCOPED`。届时必须冻结真实 journey、取消 ownership/teardown 语义、Waker 裁决、
直接依赖与 consumer、预算、机器/仪器、适用 gates 和停止条件；`AUDITED` 前不得修改产品源码。

## Closure card（`SCOPED`）

- component / owner / round / kind：`sched` / Sched execution owner /
  `sched-v4-control-contracts` / `REOPEN`；
- baseline / trigger：`f4da6ed4a2953b183a99532f31686852147ca09e` /
  `SCHED-V4-CONTROL-CONTRACT-2026-09-07`；
- normal journeys：
  1. 推理请求在一个 structured task 中等待 timer、channel 或 backend completion；owner 请求取消后，
     task、等待源和 join 的责任边界明确，不发生强制 stack unwind、悬挂 waiter 或无限误等；
  2. runtime owner 用 raw function pointer 提交非挂起 callback 或构造低层 Fiber；可抛入口在编译期被
     拒绝，而不是跨 `noexcept` trampoline terminate；
  3. bounded resource gate 的调用方能区分 acquired、temporarily unavailable、timeout 与 closed，RAII
     permit 不丢失原因；
- scope：反向追踪并冻结 cooperative task-cancel 与 operation-aware cancel 的分层；修正
  `JoinHandle`/token/wait recipe 中误导的名称或组合；让 raw callback 类型表达 `noexcept`；让 Semaphore
  acquire family 返回可判别结果并保持 FIFO/O(1) unlink；有限裁决 Waker one-shot 前置条件；迁移直接
  consumer；修正相关 architecture/user/performance/build 文档和三交付；
- non-goals：完整 scheduler/run queue/Fiber/context-switch 重写；强杀 stack；不经 operation owner 就恢复
  仍挂在 timer/IO/waiter 链的 Fiber；把所有同步原语改成自动取消；external-caller gang/SPMD、公开
  partitioner/auto-grain、worker resize、模型/请求 priority、continuous batching、GPU/异构策略；
  Compute/Graph/IO 语义重构；拆分整个 `sync.h`；Native FFI/C API；公共仪器修改；P2/Formal；
- direct dependencies：PUBLIC `core,container_intrusive,error,time,platform,platform_thread,mem,mem_vm,
  util_core`；PRIVATE `container_queue,timer_queue`；vendored Boost.Context fcontext 是 private product
  mechanism，Tracy 是条件 build-only mechanism；
- direct consumers：`io,compute,graph` 公开面使用 Sched；本轮只做被 public type change 强制要求的
  compile migration，不给这些未完成模块签收口结论，也不改变其产品策略；
- reuse candidates：Error `Status/Result` 表达 closed/timeout/unavailable；Time 提供 typed deadline；
  Container-intrusive 提供 callback/waiter 链；Util-core 提供短临界区锁；Platform-thread 提供 thread
  parking；Timer-queue 提供 deadline removal；现有 `CancelToken/CancelCallback` 和 Waker rendezvous 优先
  组合，不另造第二套取消树或 OS wait；
- strong peers：取消/任务合同使用 Tokio
  `060cc4e46aedc40b0a44dcbc31deb0df39abdb4e` 与 Folly
  `65749da412d92fe27d87ae45a8ef670720712858`；Semaphore 结果与 FIFO 使用 Tokio 同 commit 和 C++20
  `std::counting_semaphore` 的定义性边界；task/bulk negative space 使用 oneTBB
  `e00c68dcac0e83947c9b54f6d5304c0616c9639f` 与 ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` consumer shape；
- inference evidence：llama.cpp `3ad1ba7336986d98592d3e28cafd1a406715351f`、ds4
  `9ab705347c1775e7599ede7eb81a6255ec7dccb5`、vLLM
  `52358e6e192aeae73dd3764046c98fc4b156ea83`、SGLang
  `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110`；只提取 request completion、cancel/shutdown 与 CPU
  execution journey，不复制服务策略或 Python/GPU surface；
- provenance：已复现 API defect 为 `REAL_CONSUMER` 可达公共路径；推理请求等待/取消为
  `NORTH_STAR_JOURNEY`；peer 只验证通用合同；
- semantic tiers / selection：短、不挂起工作继续选择 `schedule/schedule_task`；可挂起 structured work
  选择 `spawn_task`，task cancellation 是 request/cooperative，in-flight operation cancellation 必须由
  operation owner 通过 token/callback 完成；Semaphore blocking、non-blocking 和 timed 入口共享一个
  可判别 outcome vocabulary；Waker 保持低层 completion seam，不成为普通用户取消 API；
- normal cost：既有未选择取消的 schedule/spawn/wake 路线不得新增 allocation、全局查找、syscall、
  generation word 或每次 wait CAS；显式 operation cancellation 可为 control-path token/callback 支付
  O(1) 注册/摘除和既有 token ownership；raw callback 修复是纯类型约束；Semaphore outcome 不新增
  allocation、waiter metadata、lock、wake 或复杂度，fast path 仍是一次短锁内检查；
- tier / margin：`P1` / 5%。primary 是 source-build 四 worker 的 typed task fan-out/join 与 cooperative
  cancel lifecycle，和 immutable baseline 做同产品回归；holdout 只复用 v3 已资格化的四 worker
  batch workload 对 Marl/oneTBB，确认无旁路回归。取消完成延迟只作有界 user-budget，不伪装成与
  stackless Tokio 的 primitive parity；
- budgets：capability/API 审计 90 分钟；实现与 correctness 120 分钟；delivery/docs 60 分钟；P1 45
  分钟；red team 30 分钟。P1 最多一次 primary qualification、一次 holdout qualification、一次机制
  diagnostic 和一次性能导向内部修正；fixture/tool/environment 纠正另记但不能改变产品范围；
- machines / instruments：`W1-WIN` 为 P1 裁决机，`closure-perf-v1`；`W1-WSL` 与 `M1-MAC` 做
  correctness/sanitizer/source-static-shared portability，`closure-correctness-v1`；
- correctness gates：取消 before-start/running/parked/completed 和 cancel-vs-completion race；operation
  owner teardown/late completion；when-any loser recipe；raw callback positive/negative compile；Semaphore
  zero/many/FIFO/close/timeout/race/result/RAII；Waker wake-before-suspend 与选定误用合同；shutdown；现有
  sched test suite；WSL ASan/UBSan/TSan；
- delivery gates：公开头自包含；selected `sched` source build；installed relocated static/shared C++23
  consumer；Windows/Linux/macOS；fcontext package/license；旧 C/Native FFI clean；direct IO/Compute/Graph
  compile smoke 仅在可构建范围内；
- outputs：capability map、public trace、reuse ledger、有限 `FIX`、API/usability freeze、实现/测试、
  architecture/user/build/performance 文档、candidate、三平台验证、独立红队、三项 LOC、terminal receipt
  与新 `FROZEN` confirmation；
- stop：`AUDITED` 后只实施冻结 `FIX`；一个 candidate 和上述有限 P1。全部 MUST complete/routed 且适用
  correctness/delivery/docs/reuse/red-team/P1 通过才 `CLOSED/FROZEN`；安全取消 ownership 无法定界、
  sanitizer/交付失败或正常 workload 稳定 material regression 则 `BLOCKED`，不增加测试姿势或扩建仪器。

当前进入 `SCOPED`；以上范围、非目标、成本、预算和停止条件冻结。产品实现必须等待 capability map、
public reverse trace、Waker/取消 ownership 裁决与有限 `FIX` 集在 `AUDITED` 完成。
