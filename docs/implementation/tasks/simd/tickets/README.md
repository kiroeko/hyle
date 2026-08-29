# SIMD epic ticket 队列

本目录是 [`SIMD/Math epic`](../simd-epic-v1.md) 的顺序工作包。ticket 状态只用于施工追踪：

- `QUEUED`：依赖未完成；
- `READY`：依赖完成但尚未授权开始；
- `ACTIVE`：当前唯一执行票；
- `DONE`：验收事实和 output commit 已记录；
- `BLOCKED`：已停止并写明 owner/resume condition。

它们不替代 [`module-status.json`](../../../module-status.json) 的产品状态。产品票只有在当前 closure card
已经进入 `AUDITED` 并把该票的有限改动纳入冻结 `FIX` 集后才能执行。

顺序固定为 00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12。
