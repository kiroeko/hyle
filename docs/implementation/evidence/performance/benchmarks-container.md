# Container v2 历史性能证据

<!-- PERFORMANCE EVIDENCE: HISTORICAL -->

> 本页只校准 Container v2 的历史五行 receipt，不是当前 Container 证据。当前状态和证据入口只看
> [`module-status.json`](../../module-status.json) 或其
> [人读投影](../../status.md)。当前 Container 轮的工程证据由 registry 直接指向 closure record。

Container v2 candidate `336793d63dd7cf08fde9c1be74c6439472335381` 的旧 W1-WIN bundle 在当时的
仪器门下通过，但其规模、allocator、payload 和消费 provenance 不能按当前方法冒充五个真实消费者的
同合同证书：

| 旧 workload | 校准后的类别 | 可保留的历史含义 | 不得外推 |
|---|---|---|---|
| mixed inline+spill `SmallVector` | `NATURAL_ROUTE` | 合成默认路线 lifecycle | Graph 的真实 `<uint32_t,64,mem::Allocator>` 或各子操作 NI |
| 32768-entry integral `FlatHashMap` lookup | `NATURAL_ROUTE` | 大表默认路线 lookup | 4--32 项 `StringView` Registry 或 probe/hash 算法胜负 |
| 1024-slot integer bounded handoff | mechanism candidate | 旧 queue 机制观察 | Sched 的 pointer/runtime-allocator/64-slot 消费证书 |
| 262144-bit count | mechanism candidate | 大 dense-mask count | 128-bit package journey或全部 bitset 操作 SOTA |
| 64-byte seeded hash | mechanism candidate | 该 key length 的 table-hash 观察 | 5--11 B key、全部长度或安全 hash 结论 |

旧点值和 artifact identity 可从 [`container-v2 task`](../../tasks/container/container-v2-rebuild.md)及 Git
历史追溯。它们不得续签到当前 candidate，也不得把五项合成一个 Container 分数。

2026-09-01 曾在未提交工作树中试作一个 queue-only contract；后来 Sched、package sample 和
`BoundedQueue` 三份 source provenance 全部变化，尚未形成 clean qualification。该临时 harness、合同、
validator 和 CMake 开关已删除，避免一个无法验证的 `PENDING` 实验被误认成当前产品工作。
