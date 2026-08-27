# 当前模块施工状态

> 更新时间：2026-08-27。机器真相是 [`module-status.json`](module-status.json)；本页的表格是受
> `python scripts/check-module-closure.py` 校验的人读投影。历史任务、architecture、user 或性能页中的
> `SIGNED/AUDITED/CANDIDATE/PRODUCT LOCKED` 字样都不是当前模块状态。

产品收口与 Formal 性能声明是两条轴：`CLOSED` 表示本轮工程上应停止，不表示取得发布级性能
证书；反之，“产品形状冻结”只说明 candidate 不再随意变化，不足以越过 P1 工程性能或最终红队门。

<!-- BEGIN GENERATED MODULE STATUS -->
| Component | Product | Tier | Engineering performance | Formal | Current record | Summary |
|---|---|---|---|---|---|---|
| core | CLOSED | P0 | PASS / STRUCTURAL | NONE | [task](tasks/core/core-rebuild.md) | P0 基础层已关闭；消费者性能声明由所属模块承担。 |
| container_intrusive | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/container_intrusive/container-intrusive-v3-resume.md) | v3 已关闭：真实消费者形状、typed membership transfer、correctness/交付/文档和 Boost 同合同 5% P1 工程非劣门均已签署；Formal 独立待办。 |
| util_core | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/util/util-core-v1-closure.md) | util-core-v1 CLOSED：五项 header-only primitive 的能力边界、异常/ownership、三平台 correctness/交付、C++ 易用性、依赖与 W1-WIN 同合同 5% P1 工程非劣门均已签署；util runtime 与 Formal 仍独立待办。 |
| error | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/error/error-v4-resume.md) | error-v4 CLOSED：未变化的 9595724a/a828fb4c 产品已绑定有效 source-consumer 合同；四个真实 workload 的全部 strata 通过 5% P1 工程非劣门，Formal 独立 PENDING。 |
| obs | QUEUED | — | — | — | — | 现有实现与历史审查只作输入。 |
| math | QUEUED | — | — | — | — | 现有实现只作输入；尚需裁决与 SIMD 的边界。 |
| time | QUEUED | — | — | — | — | 现有实现与历史审查只作输入。 |
| platform | QUEUED | — | — | — | — | 现有实现与历史审查只作输入。 |
| mem | QUEUED | — | — | — | — | 现有实现与历史审查只作输入。 |
| simd | QUEUED | — | — | — | — | 现有实现与历史审查只作输入。 |
| container | QUEUED | — | — | — | — | 完整容器模块仍未审查；不能继承 intrusive 子组件的候选结论。 |
| util | CLOSED | P0 | PASS / STRUCTURAL | NONE | [task](tasks/util/util-runtime-v1-closure.md) | util-runtime-v1 CLOSED：显式/no-throw/事务式 Registry 与诊断型 demangle 已通过三平台 correctness/sanitizer、source/static/shared 交付、文档、红队和 P0 成本结构门；无端到端 timing/SOTA 冒充。 |
| sched | QUEUED | — | — | — | — | 现有实现只作输入；特化 waiter 链仍归 scheduler。 |
| io | QUEUED | — | — | — | — | 现有实现只作输入；尚未启动 v1 收口。 |
| compute | QUEUED | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
| graph | QUEUED | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
<!-- END GENERATED MODULE STATUS -->

产品状态流固定为：

```text
normal:  QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> VERIFIED -> CLOSED
stop:    SCOPED | AUDITED | CANDIDATE | VERIFIED -> BLOCKED
resume:  BLOCKED -> SCOPED
reopen:  CLOSED -> REOPENED -> SCOPED
```

`container_intrusive` 已按旧 blocker 创建新的 `RESUME` round并进入 `CLOSED`。有限 ledger 的产品
修复、correctness、三平台可移植性和 source/static/shared 交付均绑定 immutable candidate；统一
`closure-perf-v1` 的 W1-WIN P1 工程证据通过 9×8 workload/strata 的冻结 5% non-inferiority gate，
并经独立重放审计。旧 v2 task 保持历史事实，不被回写。`formal-release-v1` 的跨平台证书仍为独立
`PENDING`，不能拿来替代或重复阻塞 P1 工程门。

`util_core` 已按同一状态机关闭：最终候选修复 ScopeGuard 失败安全和 `Ref` 标准指针全序，三平台
source/static/shared 交付通过；W1-WIN 的 Ref owning handoff 与短 SpinLock 同合同 workload 通过 5%
工程非劣门。该结论不覆盖仍为 `QUEUED` 的 util runtime，也不把 order-confounded 的 Mac 超短诊断
或 RNG diagnostic 冒充跨平台 SOTA 证书。

Build/package 是所有模块持续复用的交付 workstream，不是组件状态；Native FFI 仍按准入策略为
`N/A`，也不伪装成模块。完整进入/退出、停止和重开规则见
[模块重构与收口协议](../developer/review/module-closure-methodology.md)。
