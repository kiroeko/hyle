# mem-06a：分配 stats + 泄漏追踪（debug 装饰器）

> **历史任务记录。**当前 Stats/LeakTracker 的 metadata 分域、并发 snapshot、dropped/complete
> 与 C ABI 合同见 [`mem-design`](../../../architecture/mem-design.md) 和
> [`guide/c-mem`](../../../architecture/native-ffi-policy.md)；下文保留最初切片的设计语境。

| 字段 | 值 |
|---|---|
| **阶段** | P2 / mem-06 切片 |
| **模块** | mem/ |
| **状态** | ✅ 完成（Win MSVC + Linux WSL：`hyle_mem_test` 全绿）|
| **范围** | 计数 + 泄漏图；**不含** NUMA/大页（仍 mem-06 余下）|

## 问题

clay `SystemAllocator::realloc` 用无符号 `(new_size - old_size)` 与
`(new?1:0)-(old?1:0)` 更新 stats → free 时 **下溢成巨大值**（资产表 bug #2）。

## 设计

1. **`AllocStats`**：64 位原子计数（`live_count/bytes`、`total_*`、`peak_live_bytes`、`reallocs`）。free / shrink **只做减法**，从不算 `0 - old`。
2. **`StatsAllocator`**：装饰任意 `Allocator`；`system_allocator` **不**自动挂 stats（热路径零税）。
3. **`LeakTracker`**（可选）：`ptr→size` 图；`outstanding` / `copy_leaks`。
4. 生产默认不碰；测试与 debug 工具显式
   `StatsAllocator::create(measured, track_leaks, metadata)`。当前收口 API 返回 `Result`；被测
   allocation 与诊断 metadata 分域，第三参数默认 system allocator，构造 OOM 不抛出。

## 交付

- `include/hyle/mem/stats.h`
- `src/mem/stats.cc`
- `tests/mem/stats_test.cc`（下溢回归 + 平衡 + 并发 + 泄漏）

## 验收

- free 大块后 `live_bytes == 0`（非 ~2^64）
- 平衡 alloc/free；realloc grow/shrink/to-zero
- 并发 4×500 无丢计数
- track_leaks 能报告 outstanding ptr
