# Container 工程性能证据

<!-- PERFORMANCE EVIDENCE: CANDIDATE -->

## Container v6 终态候选

当前证据由两个预冻结合同组成，二者都从干净不可变候选运行，没有在看到结果后改变 workload、peer、
margin、顺序或样本数：

| factor / workload | 同合同 peer | 当前结果 | 可声明边界 |
|---|---|---|---|
| `container` / entry+weight LRU mixed access | Abseil `38f8c1535de9ebc12c6340e5c1c8ebd72afde560` adapter；同 hash、allocator、recoverable OOM、rollback、typed result、per-entry/total weight 与 entry/weight limits | 8/8 strata `WIN`；median advantage `1.2243–1.2728`；8/8 NI `PASS` | 冻结 4096-entry model-cache 形状胜出，不外推所有 cache 或 hash workload |
| `container_text` / safe UTF-8→UTF-16 caller buffer | simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`；同 provider、validation-with-errors、exact sizing、endian-correct valid conversion、typed status/offset/count 与 caller buffer | 8/8 `PARITY`、8/8 NI `PASS`；median `0.9863–1.0102`；7/8 CI 跨 1，余下一项只证明 HYLE 小幅有利 | 同语义正常路径未证明固定劣化，不声明普遍更快 |
| `container_concurrent` / read-mostly membership | oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；同 hash/equality、system allocator、8 worker、分布、查询量、生命周期与 checksum | 双顺序 `PARITY`、NI `PASS`；median `0.9882/0.9777`，CI 均跨 1 | 冻结 membership 形状 5% 非劣，不声明 exact 胜出 |
| `container_queue` / unbounded typed MPMC handoff | 同一 oneTBB revision；同 allocator、recoverable OOM/retry、producer/consumer participation、setup/drain、payload 与 FIFO conservation | 双顺序 `WIN`、NI `PASS`；median `1.2888/1.2614` | 冻结 4P4C handoff 形状胜出，不外推其他 queue policy |

单线程/Unicode 终态 manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-single-v7-terminal-evidence\manifest.json`，SHA-256
`c4793256dfe376d82feb4beeb3b5d9789f90666bd9c981bd4a9323d108f4ea5a`；contract SHA-256
`3009a1294fdc7004b365e8af79c7a5d13f237e9ffaa540c722f105cc7b71d5ef`。它绑定
`container_text` commit `3ba50461ec7c8f387f4d953f1e499a1f6d449870`，qualification 与 performance gate 均
`PASS`，n=15 已给出终局裁决而未升级样本。

并发/queue manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-concurrent-v3-evidence\manifest.json`，SHA-256
`f16b135ff6a1b0974055ae63940e88df44f2ffe44c71841d6417930eb06eb196`；contract SHA-256
`0af5617ca1c4b978a8ad65ace62764efc2e792642939465110796c333f4e6389`，绑定 commit
`111c28eab99daff6257427473a2a1b11ac1cfd0c`。`container`、`container_concurrent` 与
`container_queue` 在 `111c28ea..3ba50461` 间没有产品字节变化；因此单线程 LRU 与并发结果可按 factor
identity 复用，不把 `container_text` 的后续修正错误归给其他 factor。

两合同的 A/A、5%/10% known delta、timer floor、placement/执行顺序、artifact identity 与未计时
semantic/resource oracle 均通过。这里的 `PASS/WIN` 只回答表中冻结的同合同 consumer 形状；不同
allocator、失败语义、错误结果或所有权的默认路线必须另立 `NATURAL_ROUTE` 合同，不能混入这些结论。

<!-- PERFORMANCE EVIDENCE: HISTORICAL / REJECTED CANDIDATE -->

## Container v6 首候选：终局红队撤销错误的 `PASS`

当时的不可变产品候选是
`6a78f6dfa7a711350c694c01d90d4301fc57ae4f` / tree
`53fd52ba931af02216b9686baf77de1a2b670acc`。下表保留当时的原始统计事实，但终局红队证明 LRU weight
语义尚未等价，并发 membership 又违反了样本升级顺序；所有 `PASS/WIN` 解释均已撤销，不是当前证书。

| factor / workload | 当时使用的 peer/adapter | candidate-bound 原始结果 | 红队后的合法结论 |
|---|---|---|---|
| `container` / entry+weight LRU hit、insert+evict | Abseil `38f8c1535de9ebc12c6340e5c1c8ebd72afde560`；hash、allocator、OOM、rollback 与 typed result 已对齐，但 peer 未维护 weight | 8/8 strata 数值有利，median ratio `1.1898–1.2061` | `NOT_COMPARABLE`；不得签胜出或 NI |
| `container_text` / safe UTF-8→UTF-16 caller buffer | simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`；同一 validation、exact length、endian-correct conversion、typed status/offset/count 与 caller buffer | 8/8 strata `PARITY`；median advantage `0.97197–0.99365`；8/8 NI `PASS` | 冻结 workload 上 5% 非劣，不声明普遍更快 |
| `container_concurrent` / read-mostly membership | oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；同一 hash/equality、system allocator backend、操作 mix、生命周期与 checksum | 首次合法 `n=31` 的 peer-first CI 下界 `0.94120`；后续反向缩至 `n=15` 才越线 | 首次 verdict `UNRESOLVED`；缩样本重跑无效 |
| `container_queue` / unbounded typed MPMC | 同一 oneTBB revision；同一 allocator backend、recoverable OOM/retry、producer/consumer participation、setup/drain 与 FIFO conservation | 最终双顺序 median `1.1784/1.4158`；2/2 NI `PASS`，其中一向为 `WIN` | 冻结 workload 上 5% 非劣，不声明所有 queue 形状胜出 |

单线程合同 `container-product-p1-v6` 的 manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-single-evidence\manifest.json`，SHA-256
`2d01defce2189c50ec3cc629832270c221dd4c943ef08eba838538da3401cc52`，contract SHA-256
`46adff1d372d43c2e6b403576d8117144cd1f1e3aa30acafa76488aed72f0390`；qualification 与 performance
gate 均为 `PASS`。A/A、5%/10% known delta、timer floor、placement 与语义/resource oracle 全部通过。

并发合同 `container-concurrent-product-p1-v2` 的首次 `n=31` qualification 合法，但 membership 的一个
99% CI 下界穿过 5% NI 边界，因此当次聚合正确记为 `INCONCLUSIVE`。唯一机制检查确认 affinity、工作量、
checksum、A/A、known delta 与 timer floor 均正常，没有固定产品损失或 correction 依据；按预冻结预算只
进行一次同候选、同合同重跑。最终 manifest 位于
`C:\public\project-hyle\_validation\container-v6\qual-concurrent-rerun-evidence\manifest.json`，SHA-256
`13e667c238c3268e6165f9e0792dd216baa6d998a90f7a9ddf47023d2bd6e371`，contract SHA-256
`c2eb52350b27c7663c6a761a6b3fc780f9ff315deff46652c9a2131c41138e15`。该 manifest 的机械 gate 虽为
`PASS`，但运行顺序违反只允许 `n=15 -> n=31` 提升功效、禁止跑到刚好通过的规则，不能覆盖首次
`n=31` 的 `UNRESOLVED`。

因此四个 factor 的聚合工程 P1 仍为 `INCONCLUSIVE`。本轮返回 `AUDITED`，只允许让 LRU peer 真正承担
weight 语义、把约 9ms 的 membership 进程窗口按预冻结规则提高到足够分辨率、修正 factor attribution，
再从唯一 `n=15` 开始形成新候选证据；不得复用本候选 verdict、改 margin 或挑样。

上述有限修复随后形成新的预采样合同 `container-product-p1-v7` 与
`container-concurrent-product-p1-v3`：peer 已真实执行 weighted-LRU 语义，membership 窗口提高到每 worker
4194304 次查询（W1-WIN smoke 约 134--137ms），合同以 Container 产品 façade 为根并记录逐 workload
factor owner。在该历史时间点，编译、未计时语义 oracle 和 smoke 已通过但正式 qualification 尚未运行，
因此当时保持 rejected/inconclusive，没有把预采样结果提前写成性能 `PASS`；当前终态结果见本页首节。

<!-- PERFORMANCE EVIDENCE: HISTORICAL -->

## Container v5：历史终局为 `INCONCLUSIVE`

该轮 Container 产品候选是
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
