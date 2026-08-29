# Time 当前产品身份回放 — 2026-08-29

<!-- MODULE REPLAY RECORD: COMPLETE -->

> 本页是对已经 `CLOSED` 的 `time-v1` 所做的 product-non-mutating 回放。它只比较冻结候选与
> 当前产品事实；产品修复和重新签署由独立的 `time-v2` 轮承担。

## 回放合同

- component / replay：`time` / `time-current-identity-replay-2026-08`；
- reviewed state / round：`CLOSED` / `time-v1`；
- closure task：[`time-v1 closure`](time-v1-closure.md)；
- reviewed candidate / tree / end / registry anchor：
  `0ef2544ceece030bb522857854285d280e368d36` /
  `104a4b6dc904e5b261010ace820ec2c474374c31` /
  `85d4f1ec3c1a88eea8803271d518db8b8229f4e2` /
  `85d4f1ec3c1a88eea8803271d518db8b8229f4e2`；
- replay commit：`565b710fd6dad1a50588992d336dddb958326d99`；reviewer：Codex；日期：
  2026-08-29；调查预算：20 分钟；
- 范围：candidate 到 replay commit 的 `include/hyle/time`、`src/time`、Time tests、直接性能消费者
  provenance、组件边与三种交付身份；
- 非目标：修改产品、增加能力、重跑性能采样、重审 sched/platform 的其他能力或续签 Formal。

## 身份比较

`src/time`、timer 算法、对象 layout、直接组件边和 normal workload 均无变化。差异只有两类：

1. `3b759eb6a19b0f6e7137ec948b7298cf68c13c67` 把 libc++ 不提供的
   `std::chrono::is_clock_v` 测试改为直接验证标准 Clock 形状；这是 test portability，不是产品变化；
2. `5b8dab7012e9f7c6cdaf97f863c0b83b4e8a5064` 在 `clock.h` 和
   `common_timeout_queue.h` 把 `Duration::max()` 写成标准的 `(Duration::max)()`。该修改使先包含
   Windows SDK、且没有定义 `NOMINMAX` 的消费者不会被函数式 `max(a,b)` 宏破坏。数值语义、layout、
   分支和生成代码不变，但公开头的产品身份及受支持平台组合合同确实改变。

Time 的冻结 scheduler replay 另有 `src/sched/scheduler.cc` 整文件 hash 漂移。独立
[`performance provenance task`](../performance/perf-provenance-renewal-v1.md) 已证明变化只发生在
不计费的 Parker wait 调用，并登记精确、engineering-only 的 `NON_MATERIAL` 迁移；旧 manifest 未被
改写，Formal 不继承该迁移。这是仪器兼容结论，不消除上面的 Time 产品变化。

## Verdict

- replay verdict：**`PRODUCT_TRIGGER`**；
- trigger ID：`TIME-REPLAY-WINMAX-01`；
- trigger evidence：支持的 Windows C++ 消费者 include-order 曾能使 Time 公开头编译失败，而当前
  产品已经修复；它命中 time-v1 的“公开 API / supported platform primitive / delivery shape 改变”
  重开条件；
- 受影响切片：公开头宏可组合性、当前候选身份、Time 自有回归测试、Windows/Linux/macOS 当前
  source/static/shared 导入，以及旧 P1 工程证据在当前消费者 provenance 下的深回放；
- 未触发切片：能力族、clock/timer 算法、ownership、复杂度、常态 metadata、C++ API 语义与
  Native FFI policy；
- 下一动作：创建 `time-v2-platform-macro-reanchor`，只重新签署上述有限切片。不得借本 trigger
  增加功能或发明新 benchmark。
