# Container 工程性能证据

<!-- PERFORMANCE EVIDENCE: BLOCKED RECORD -->

## Container v5：终局为 `INCONCLUSIVE`

当前 Container 产品候选是
`6fae3c11ebce9a0741f45780e63c9896e08bb2a2` / tree
`8025ee111380c2600f804cf979ed59d289a8be76`。本轮的功能、复杂度、正确性、交付、易用性、文档和
红队均已完成；P1 工程性能门没有通过，也没有确认产品性能失败。原因是第二候选的四组 peer
comparison 没有满足冻结合同的 `SAME_CONTRACT`，而第三候选不能继承被拒绝的数字。

第二候选 `a7ce0102413c783e26227c68645e0282b109fde4` 的 clean W1-WIN 诊断为：

| factor / workload | 自然路线点值 | 合同复核 | 当前可用结论 |
|---|---:|---|---|
| `container` / entry+weight LRU hit 与 insert+evict | HYLE/peer `1.1213–1.1665`，8/8 分层为 `WIN` | HYLE 是 recoverable OOM + rollback；Abseil/`std::list` peer 是 throwing，adapter 没有补齐失败与回滚语义 | 只保留 `NATURAL_ROUTE` 诊断；不得宣称实现级胜出或 SOTA |
| `container_concurrent` / read-mostly membership | `0.9893`、`0.9744`，置信区间未裁决 | 两侧 default allocator/resource 不同，且没有冻结 resource equality | `INCONCLUSIVE`；不得签 non-inferiority |
| `container_queue` / burst unbounded MPMC handoff | HYLE/peer `1.2296–1.2538` | HYLE 用 mimalloc-backed `SystemAllocator` 与 recoverable `try_push`；oneTBB 用 `cache_aligned_allocator` 与 throwing `push` | 只保留 `NATURAL_ROUTE` 诊断；不得宣称实现级胜出或 SOTA |
| `text` / safe UTF-8→UTF-16 caller buffer | HYLE/peer `0.9616–0.9731`；3/8 CI 穿越 5% 边界 | HYLE 返回 typed status/error offset/output count；peer adapter 对错误输入终止，未等价化结果语义 | 有效输入自然路线约慢 2.7–3.8%，但同合同门仍为 `INCONCLUSIVE` |

这正是“不得苹果与橘子对比”规则的应用：自然 API 路线可以回答用户默认得到什么，但不能自动回答某个
算法或实现是否在等价资源和语义下更快。即使自然路线数字对 HYLE 有利，也不能把 allocator、OOM、错误
结果和 rollback 的差异藏进一个 bool 后签发 `PASS`。

第三候选只修复 Unix shared visibility 和公开 injected-policy 并发前提，没有改变上述正常路径算法。
它已完成三平台 correctness/delivery 精确回放，但本轮预算和一次 performance-guided correction 均已用完，
因此没有临时改写 comparator 再追一轮点值。聚合工程 verdict 为 `INCONCLUSIVE`，四个 factor 均进入
`BLOCKED`，而不是伪造 `FROZEN`。

恢复时只允许新建有限 `RESUME` round，先为每个 cell 明示
`EQUAL`、`ADAPTER_EQUALIZED` 或 `INTENTIONALLY_DIFFERENT`：

- LRU 等价化 allocation failure、rollback 和返回结果；
- concurrent membership 使用同一 allocator/resource；
- queue 等价化 allocator、OOM、push result 和 producer-token/setup；
- Unicode 等价化 validation、错误结果、output-size 和 caller-buffer contract。

如果真正要衡量的是各库默认路线，则应另行预冻结 `NATURAL_ROUTE` consumer budget，并只对端到端默认
产品作结论，不能将其改名为 `SAME_CONTRACT`。两种命题都必须继续使用 5% material margin、双顺序、
placement、A/A、known-delta 和 immutable-candidate replay。

原始第二候选诊断：

- single-thread manifest：
  `C:\public\project-hyle\_validation\container-v5\qual-single-evidence-a7ce0102\manifest.json`，
  SHA-256 `93F91BA274FA1CA1BE2A7AF3014540F96046ADE8C848B4D8C2EED03D3484F4DF`；
- concurrent manifest：
  `C:\public\project-hyle\_validation\container-v5\qual-concurrent-evidence-a7ce0102\manifest.json`，
  SHA-256 `753246ACB14ACD834E143266DF24F10C247DD501A51E75ADBD6DDFEE7F3BAC11`。

这些是可追溯的 rejected-candidate 事实，不是当前候选的 P1 certificate。

## Container v2 历史性能证据

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
