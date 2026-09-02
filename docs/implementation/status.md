# 当前模块施工状态

> 更新时间：2026-09-02。机器真相是 [`module-status.json`](module-status.json)；本页的表格是受
> `python scripts/check-module-closure.py` 校验的人读投影。历史任务、architecture、user 或性能页中的
> `SIGNED/AUDITED/CANDIDATE/PRODUCT LOCKED` 字样都不是当前模块状态。

产品收口与 Formal 性能声明是两条轴：`CLOSED` 表示本轮工程上应停止，不表示取得发布级性能
证书；反之，“产品形状冻结”只说明 candidate 不再随意变化，不足以越过 P1 工程性能或最终红队门。

<!-- BEGIN GENERATED MODULE STATUS -->
| Component | Product | Tier | Engineering performance | Formal | Current record | Summary |
|---|---|---|---|---|---|---|
| core | CLOSED | P0 | PASS / STRUCTURAL | NONE | [task](tasks/core/core-v5-assert-macro-contract.md) | core-v5 CLOSED：断言宏卫生、Release/Debug 编译语义、字段边界和 observer 生命周期文档已修复；公开 API 不变，三平台正确性、三种 C++ 交付、文档/易用性、红队与 P0 结构性能全部通过。 |
| container_intrusive | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/container_intrusive/container-intrusive-v4-capability-extension.md) | container_intrusive v4 CLOSED：counted safe-list、一指针 cached-tail forward-list、auto/safe hlist 与稳定 sequence algorithms 已完成三平台正确性、三种 C++ 交付、直接 Sched 消费、文档/红队及 W1-WIN 四项 5% P1 工程门验证。 |
| util_core | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/util/util-core-v2-capability-completion.md) | util-core-v2 CLOSED：RISC-V pause、FunctionRef、NotNull 与 thread-confined RefCounted 已补齐；三平台/sanitizer/RISC-V、三种 C++ 交付、直接消费者、易用性/文档、红队和 W1-WIN 两项 5% P1 工程门全部通过。 |
| error | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/error/error-v5-capability-completion.md) | error-v5 CLOSED：WinSock、Result<T&>、bounded format_status 与 std::error_code adapter 已补齐；既有 Status/Result 无回归，三平台正确性和三种 C++ 交付、易用性/文档、红队及 W1-WIN 五个同合同 consumer 的 5% P1 非劣门全部通过。 |
| obs | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/obs/obs-v3-closure.md) | obs-v3 CLOSED：Abseil-shaped 普通日志控制与 llama.cpp-shaped 推理进度能力、三平台 correctness/sanitizer、三种 C++ 交付、文档/红队和 W1-WIN 5% P1 工程性能门全部通过；Formal 独立 PENDING。 |
| math | CLOSED | P0 | PASS / STRUCTURAL | NONE | [task](tasks/math/math-v2-checked-arithmetic.md) | math-v2 CLOSED：checked add/sub/mul 与既有 ceil/round 的功能/API、三平台正确性/UBSan、三交付、文档、红队和原生 overflow P0 code shape 全部签署；候选 1a9e680c/01032312。 |
| time | CLOSED | P0 | PASS / STRUCTURAL | PENDING | [task](tasks/time/time-v5-interoperability.md) | time-v5 CLOSED：signed Unix units、absolute/relative POSIX、timeout rounding、direct FILETIME 与 header isolation 已通过三平台正确性/WSL sanitizers、三种 C++ 交付、文档/红队和 P0 结构性能门；候选 680c9941/b09b7043 无产品漂移。 |
| timer_queue | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/timer_queue/timer-queue-v1-factor.md) | timer_queue v1 CLOSED：caller-owned radix/common-timeout 已成为独立 header-only factor，provider 组合与 Sched/IO 直接边正确；三平台正确性/sanitizer、三种 C++ 交付、文档/红队及对当前 Seastar 的 8/8 P1 WIN 全部通过。 |
| platform | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/platform/platform-v2-inference-residency.md) | platform-v2 CLOSED：推理驱动的 progressive prefix、Linux on-fault 与显式 quota-adjust 已由三平台正确性/sanitizer、三种 C++ 交付、文档/红队和 W1-WIN 5% P1 工程性能门全部签署；Formal 独立 PENDING。 |
| mem_reclaim | QUEUED | — | — | — | — | 已登记为 Mem 内部的并发回收质因数；尚未进入 SCOPED，当前 hyle::mem 产品门面不选择它。 |
| mem | AUDITED | P1 | PLANNED / P1_ENGINEERING / 5% | PENDING | [task](tasks/mem/mem-v2-rebuild.md) | mem-v2 能力超集、公开面反向追踪、语义所有权和 MEM2-01～09 有限 FIX 集已审定；下一步从产品 taxonomy 与无环 provider seam 开始实现。 |
| simd | CLOSED | P0 | PASS / STRUCTURAL | PENDING | [task](tasks/simd/simd-v4-expert-sdk.md) | simd-v4 CLOSED：统一 hyle::simd_expert 与 fixed/fat helper 已在 Windows/WSL/macOS 的 source、relocated shared/static 三种 C++ 交付全部通过；P0 结构成本、隔离、离线资产、文档和红队均 PASS，产品锁定 1e543dec/4e4e82e5。 |
| container | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | NONE | [task](tasks/container/container-v2-rebuild.md) | container-v2 CLOSED：exact overwrite ring、semantic composite hash、flat filtering、真实 StringView 与 const interner 已完成不兼容收束；三平台 correctness/sanitizer/ISA、直接消费者、source/static/shared、文档/红队和 W1-WIN 五项 5% P1 非劣门全部通过。 |
| util | CLOSED | P0 | PASS / STRUCTURAL | NONE | [task](tasks/util/util-runtime-v3-rebuild.md) | util-runtime-v3 CLOSED：唯一 lookup-only factory discovery 缺口已补齐；完整能力账本、三平台 correctness/sanitizer、source/static/shared、C++ 易用性、文档、FFI 清洁度、P0 结构成本与红队全部签署。 |
| sched | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/sched/sched-v3-root-participation.md) | sched-v3 CLOSED：same-Scheduler root participation 与 bounded sibling activation 消除了 v2 的 oneTBB batch gap；三平台 correctness/sanitizer、三种 C++ 交付、符号/FFI 清洁、直接消费者、文档/红队与 W1-WIN 5% P1 engineering gate 全部通过，Formal 保持 PENDING。 |
| io | QUEUED | — | — | — | — | 现有实现只作输入；尚未启动 v1 收口。 |
| compute | QUEUED | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
| graph | QUEUED | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
<!-- END GENERATED MODULE STATUS -->

新接受的能力完整性标准正在通过
[存量采用记录](tasks/project/capability-completeness-adoption-2026-08.md)回放到
`core/error/util_core/util/container_intrusive/build-package`。`core`、`error`、`util_core` 与
`container_intrusive` 的回放已经分别
[`CONFIRMED`](tasks/core/core-capability-replay-v2.md) /
[`CONFIRMED`](tasks/error/error-capability-replay-v1.md) /
[`CONFIRMED`](tasks/util/util-core-capability-replay-v1.md) /
[`CONFIRMED`](tasks/container_intrusive/container-intrusive-capability-replay-v1.md)，均无 API、算法或
布局改动。`util` 回放发现 owning-demangle `PRODUCT_TRIGGER` 后已经由 v2 修复并重新 `CLOSED`；
build/package 回放也已发现
[`PRODUCT_TRIGGER`](tasks/build/build-package-capability-replay-v1.md)：selected-root configure 会扫描并
受未选模块源码阻断；[`v5`](tasks/build/build-system-v5-selected-closure.md) 已将审计收束到 selected
closure，并经三平台 current contract 与 Core 三种交付复签后 `CLOSED`。
这项只读审计不会批量改写上表状态，只有发现真实 `PRODUCT_TRIGGER` 的单元才依法重开。`time` 已
直接使用新格式，`platform/mem/obs` 将在各自正常轮次使用，不补签历史结论。

产品状态流固定为：

```text
normal:  QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> VERIFIED -> CLOSED
stop:    SCOPED | AUDITED | CANDIDATE | VERIFIED -> BLOCKED
resume:  BLOCKED -> SCOPED
reopen:  CLOSED -> REOPENED -> SCOPED
```

`container_intrusive` v4 已因真实能力缺口重开并再次进入 `CLOSED`。counted safe-list、一指针
cached-tail forward-list、auto/safe hlist 与稳定 sequence algorithms 均绑定 immutable candidate；
三平台 correctness、三种 C++ 交付和 Sched 真实消费者通过，W1-WIN 四项正常 shape × 8 strata 全部
通过冻结的 5% non-inferiority gate。旧 v2/v3 task 保持历史事实，不被回写；`formal-release-v1` 的
跨平台证书仍为独立 `PENDING`。

`util_core` 已按同一状态机关闭：最终候选修复 ScopeGuard 失败安全和 `Ref` 标准指针全序，三平台
source/static/shared 交付通过；W1-WIN 的 Ref owning handoff 与短 SpinLock 同合同 workload 通过 5%
工程非劣门。util runtime 随后已按自己的 P0 startup/diagnostic 合同独立关闭；它没有真实 profile
证明进入 token/kernel 热路，因此不继承 util-core 的 P1 计时声明，也不把旧 lookup 微基准冒充当前
SOTA 证书。util-core 的新版能力回放已 `CONFIRMED`：五个 primitive 没有半项 `MUST`，只校准
throwing factory/TLS seed/type hash 等合同陷阱；同样不把 order-confounded 的 Mac 超短诊断或 RNG
diagnostic 冒充跨平台证书。

Build/package 是所有模块持续复用的交付 workstream，不是组件状态；Native FFI 仍按准入策略为
`N/A`，也不伪装成模块。完整进入/退出、停止和重开规则见
[模块重构与收口协议](../developer/review/module-closure-methodology.md)。
