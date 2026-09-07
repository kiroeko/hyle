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

