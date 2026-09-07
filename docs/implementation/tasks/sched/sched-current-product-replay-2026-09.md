# Sched 当前产品回放 — 2026-09

<!-- READ-ONLY CAPABILITY REPLAY: PRODUCT_TRIGGER -->

> 本页只记录 product-non-mutating replay。它不授权修改公开 API、算法、布局、依赖或正常 workload；
> 产品施工必须进入独立的 `REOPENED -> SCOPED -> AUDITED` 轮次。

## 回放合同

- component：`sched`；reviewed round：`sched-v3-root-participation`；
- reviewed candidate / tree：`536d4abf6324e2b08ed7b3a80df34e83f955e60b` /
  `bc9b6baaa5dbf4f3d214047e85cb3ada472e45df`；
- replay commit / tree：`f4da6ed4a2953b183a99532f31686852147ca09e` /
  `aacd7e9e8bc1d9df075e2c52c93a2a5ab811ba45`；
- closure task：[`sched-v3-root-participation`](sched-v3-root-participation.md)；
- reviewer / date：Codex / 2026-09-07；
- budget：只读 public-surface、consumer/peer、复杂度、依赖和文档回放 90 分钟；性能计时预算为 0；
- owner question：在不把 Graph/Compute/推理策略塞进底层 runtime 的前提下，确认 Sched 的功能超集、
  自然 C++ 形状和冻结资格。

## 锁定输入

| Source | Clean identity | Role |
|---|---|---|
| HYLE | replay commit/tree 见上 | public surface、实现、测试和当前文档 |
| Tokio | `060cc4e46aedc40b0a44dcbc31deb0df39abdb4e` | task cancellation 与 semaphore 结果语义 strong peer |
| Folly | `65749da412d92fe27d87ae45a8ef670720712858` | callback-based cooperative cancellation strong peer |
| oneTBB | `e00c68dcac0e83947c9b54f6d5304c0616c9639f` | task/bulk execution 边界 strong peer |
| ggml | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` | CPU gang/threadpool foundation 差异输入 |
| llama.cpp | `3ad1ba7336986d98592d3e28cafd1a406715351f` | 推理 consumer journey |
| ds4 | `9ab705347c1775e7599ede7eb81a6255ec7dccb5` | 推理 consumer journey |
| vLLM | `52358e6e192aeae73dd3764046c98fc4b156ea83` | serving lifecycle/policy 边界输入 |
| SGLang | `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110` | serving lifecycle/policy 边界输入 |

上述 reference 在回放开始时均为 clean checkout。四组推理 reference 只证明消费形状，不冒充
same-contract timing peer。

## 结果

| ID | Evidence | Current consequence |
|---|---|---|
| `SCHED-R4-CANCEL-01` | `JoinHandle::abort()` 只写 `cancel_req`；`cancel_test.cc` 与 `join_test.cc` 都必须随后手工 `Waker::wake()` 才能让 parked task 观察取消。当前名称和 `when_any` loser recipe 容易被理解为能主动停止等待，而底层 operation owner 又不能被不安全地绕过。 | `PARTIAL / PRODUCT_TRIGGER`：冻结 request-only 与 operation-aware cancellation 的自然分层；不得靠强制恢复仍挂在 IO/timer/waiter 链中的 Fiber 修补。 |
| `SCHED-R4-NOEXCEPT-01` | `Scheduler::DetachedFn` 与 `Fiber::EntryFn` 是可抛的 `void(*)(void*)`，但执行 trampoline 是 `noexcept` 且文档要求回调不得抛；合法类型可编译后在边界直接 terminate。其他 Sched callback 已在类型上使用 `noexcept`。 | `PARTIAL / PRODUCT_TRIGGER`：公开 raw callback 类型必须表达真实异常合同，并有 compile-only 误用诊断。 |
| `SCHED-R4-SEMAPHORE-01` | `acquire/try_acquire/timed acquire` 都返回 `bool`；`false` 分别折叠 closed、no-permits 和 timeout。Tokio 同类 API 明确区分 `AcquireError` 与 `TryAcquireError::{Closed,NoPermits}`。 | `PARTIAL / PRODUCT_TRIGGER`：保留 O(1) fast path 和 FIFO 语义，同时给调用方可判别结果；不为 benchmark 增加状态。 |
| `SCHED-R4-DOC-01` | user guide 声称 `Mutex` 在 Scheduler Fiber 中挂起当前 Fiber，实际实现竞争时阻塞 OS worker；build-system/roadmap 对 Sched private Container factor 的陈述也不一致。 | 本轮必须修复的 usability/build documentation drift。 |
| `SCHED-R4-WAKER-01` | `Waker` 是可复制 raw pair，正确性依赖“每个 suspend episode 恰好调用一次”；重复或 stale wake 的成本与误用后果未在本轮回放内完成产品裁决。 | `AUDIT`：在新轮有限裁决为保留低层前置条件、无成本类型约束或具证据的修复；不得先加 generation/CAS 常态元数据。 |

Sched 的 M:N Fiber、work stealing、worker-stack/stackful/typed task、bulk、blocking pool、timer、同步与
消息原语、affinity 和 shutdown 主体保持 `KEEP`。external caller gang participation、continuous
batching、请求/模型优先级、KV/scratch、GPU/异构执行和 serving admission 属 Compute/Graph/未来推理
runtime；ggml、vLLM 与 SGLang 的这些策略差异为 `OUT_OF_SCOPE`，不构成本轮扩张理由。

性能回放只确认旧 v3 的一个四 worker bulk P1 结论范围很窄；没有据此发现新的固定 material regression，
也没有把旧结论外推为全操作 SOTA。新轮只有在 API/usability 与产品 candidate 冻结后，才可运行预声明的
一个正常任务 workload 和至多一个 holdout。

## Verdict

- replay verdict：**`PRODUCT_TRIGGER`**；
- trigger ID：`SCHED-V4-CONTROL-CONTRACT-2026-09-07`；
- 最小受影响面：task cancellation/control 名称与组合、raw callback type、Semaphore result、相关测试、
  user/architecture/build 文档以及这些变化触及的 source/static/shared journey；
- 下一步：建立 [`sched-v4-control-contracts`](sched-v4-control-contracts.md)，依法进入
  `CLOSED -> REOPENED -> SCOPED -> AUDITED`。在 `AUDITED` 冻结有限 `FIX` 集前不改产品代码。

