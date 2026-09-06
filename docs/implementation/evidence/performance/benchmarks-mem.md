# Mem v1 历史性能证据

<!-- PERFORMANCE EVIDENCE: HISTORICAL -->

> 本页只校准 Mem v1 的历史 receipt，不是当前 Mem 证据。当前状态和当前证据入口只看
> [`module-status.json`](../../module-status.json) 或其
> [人读投影](../../status.md)。当前 Mem 产品证据是
> [`benchmarks-mem-v6.md`](benchmarks-mem-v6.md)，Mem-VM 证据是
> [`benchmarks-mem-vm-v2.md`](benchmarks-mem-vm-v2.md)。

Mem v1 candidate `2c71dfae9f4fb43ff0d68e5c1a4c55099fc0f101` 的旧 bundle 可以证明当时的
artifact、仪器和点值，但不能按后来建立的 comparison-class 规则把所有行统称为“同合同 P1”。旧数字
只保留下列历史解释：

| workload | 校准后的类别 | 历史事实 | 不得外推 |
|---|---|---|---|
| runtime `Allocator` / direct `SystemAllocator` | `MECHANISM_DIAGNOSTIC` | direct route 快约 11.1%--13.4% | allocator 产品门或 SOTA |
| runtime `Allocator` / oneTBB PMR 自然路线 | `NATURAL_ROUTE` | 冻结 64×64 B 任务上 HYLE 总路线约 1.45x | dispatch、mimalloc 或算法单独快 1.45x |
| `Region` / Boost monotonic 自然路线 | `NATURAL_ROUTE` | repeated epoch + 256 KiB retention budget 下约 1.62x | 两侧驻留/lifecycle 相同或 bump 算法快 1.62x |
| warmed guarded-stack pool / fresh guarded VM | `MECHANISM_DIAGNOSTIC` | 缓存复用相对每次新建 VM 约 8.0x | external-peer 或 Mem 实现级结论 |
| `ObjectPool` / Marl `BoundedPool` 自然路线 | `NATURAL_ROUTE` | 两库各自 scoped-pool 路线约 2.13--2.21x | move-only pool 与可复制/refcounted pool 同合同 |

这些差异混合了 backing、dispatch、retention、共享语义或缓存策略。旧 A/A、known-delta、timer-floor、
placement 和 raw hashes 只证明历史测量完整，不能消除语义差异。完整施工事实仍可从
[`mem-v1 closure`](../../tasks/mem/mem-v1-closure.md)和 Git 历史追溯。

以后需要复用旧数字时必须明确写 `HISTORICAL`，并继承本页的允许/禁止 claim；不得把该 receipt 续签到
新 candidate，也不得据此改变当前 API、layout 或常态 metadata。
