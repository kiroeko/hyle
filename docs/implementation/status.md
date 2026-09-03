# 当前模块施工状态

> 更新时间：2026-09-03。机器真相是 [`module-status.json`](module-status.json)；本页的表格是受
> `python scripts/check-module-closure.py` 校验的人读投影。历史任务、architecture、user 或性能页中的
> `SIGNED/AUDITED/CANDIDATE/PRODUCT LOCKED` 字样都不是当前模块状态。

产品收口与 Formal 性能声明是两条轴：`CLOSED` 表示本轮工程上应停止，不表示取得发布级性能
证书；反之，“产品形状冻结”只说明 candidate 不再随意变化，不足以越过 P1 工程性能或最终红队门。

当前执行优先级是 **Mem 全产品只读 replay**：它复核 `mem/mem_vm/mem_reclaim` 的累积产品形状、
依赖、第三方输入、功能超集、C++ 易用性、交付与性能证据。Mem 在产生合法
`PRODUCT_TRIGGER` 前仍保持下表的 `CLOSED`；这不是第二套产品状态。IO 已冻结的 v1 scope 暂停在
`SCOPED`，尚未进入 `AUDITED` 或修改产品，不是当前施工对象。

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
| platform | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/platform/platform-v3-capability-factorization.md) | platform-v3 CLOSED：host/thread/fs 十一票能力与质因数边界完成；三平台、sanitizer、source/static/shared、direct consumers、文档/易用性、红队和三族同语义 P1 全部通过。 |
| platform_thread | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/platform_thread/platform-thread-v1-factor.md) | platform_thread v1 CLOSED：native thread/affinity/current ISA state/address wait 能力与独立因子完成；三平台、sanitizer、Platform 三交付、Sched consumer、文档/红队和 Parker P1 全部通过。 |
| platform_fs | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/platform_fs/platform-fs-v1-factor.md) | platform_fs v1 CLOSED：同步 namespace/FileHandle/temp/replace 能力与独立因子完成；三平台、sanitizer、Platform 三交付、Mem/IO consumers、文档/红队和同语义 filesystem P1 全部通过。 |
| mem_reclaim | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/mem/mem-reclaim-v2-inference-resume.md) | mem_reclaim v2 CLOSED：推理 read-mostly snapshot 回收、零分配 typed retire 与 Container 直接 consumer 完成；三平台正确性/sanitizer/三交付、文档/红队及官方 Folly 同合同 P1 全部通过。 |
| mem | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/mem/mem-v4-reclamation-facade.md) | mem-v4 CLOSED：read-mostly reclamation 已自然进入 Mem façade；三平台 source/shared/static、依赖闭包、package journey、用户选型、FFI 清洁与同 candidate P1 全部通过，allocator/VM 保持冻结。 |
| mem_vm | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/mem_vm/mem-vm-v1-factor.md) | mem_vm v1 CLOSED：mapping/residency/huge/shared/guarded storage 独立因子完成；三平台、sanitizer、Mem 三交付、direct consumers、文档/红队和两族同语义 P1 全部通过。 |
| simd | CLOSED | P0 | PASS / STRUCTURAL | PENDING | [task](tasks/simd/simd-v4-expert-sdk.md) | simd-v4 CLOSED：统一 hyle::simd_expert 与 fixed/fat helper 已在 Windows/WSL/macOS 的 source、relocated shared/static 三种 C++ 交付全部通过；P0 结构成本、隔离、离线资产、文档和红队均 PASS，产品锁定 1e543dec/4e4e82e5。 |
| container | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | NONE | [task](tasks/container/container-v3-node-table-reclamation.md) | container-v3 CLOSED：默认 retained 与显式 reclaiming table lifetime 完成；安全 hand-off、三平台正确性/sanitizer/三交付、依赖/文档/红队及 oneTBB natural-route P1 全部通过。 |
| util | CLOSED | P0 | PASS / STRUCTURAL | NONE | [task](tasks/util/util-runtime-v3-rebuild.md) | util-runtime-v3 CLOSED：唯一 lookup-only factory discovery 缺口已补齐；完整能力账本、三平台 correctness/sanitizer、source/static/shared、C++ 易用性、文档、FFI 清洁度、P0 结构成本与红队全部签署。 |
| sched | CLOSED | P1 | PASS / P1_ENGINEERING / 5% | PENDING | [task](tasks/sched/sched-v3-root-participation.md) | sched-v3 CLOSED：same-Scheduler root participation 与 bounded sibling activation 消除了 v2 的 oneTBB batch gap；三平台 correctness/sanitizer、三种 C++ 交付、符号/FFI 清洁、直接消费者、文档/红队与 W1-WIN 5% P1 engineering gate 全部通过，Formal 保持 PENDING。 |
| io | SCOPED | P1 | PLANNED / P1_ENGINEERING / 5% | PENDING | [task](tasks/io/io-v1-core-factorization.md) | io-v1 SCOPED 但已按 owner 决定暂停：scope、baseline 与预算已冻结，尚未进入 AUDITED、未修改产品；当前先执行 Mem 全产品只读 replay，IO 只有在 owner 恢复该顺序后继续。 |
| io_tls | QUEUED | — | — | — | — | IO/TLS 因子 ID 已登记；尚无独立源码 ownership、target、依赖或产品根，等待自己的 SCOPED/AUDITED 轮次。 |
| io_http | QUEUED | — | — | — | — | IO/HTTP 因子 ID 已登记；尚无独立源码 ownership、target、依赖或产品根，等待自己的 SCOPED/AUDITED 轮次。 |
| io_http2 | QUEUED | — | — | — | — | IO/HTTP2 因子 ID 已登记；尚无独立源码 ownership、target、依赖或产品根，等待自己的 SCOPED/AUDITED 轮次。 |
| io_transfer | QUEUED | — | — | — | — | IO/transfer 因子 ID 已登记；尚无独立源码 ownership、target、依赖或产品根，等待自己的 SCOPED/AUDITED 轮次。 |
| compute | QUEUED | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
| graph | QUEUED | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
<!-- END GENERATED MODULE STATUS -->

能力完整性标准最初通过
[存量采用记录](tasks/project/capability-completeness-adoption-2026-08.md)回放到
`core/error/util_core/util/container_intrusive/build-package`。`core`、`error`、`util_core` 与
`container_intrusive` 的回放分别
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
直接使用新格式；后续 Platform/Mem/Obs 等产品轮也留下了自己的 terminal 记录，但窄 reopen round
不能自动替整个累积产品回答今天的新问题。当前 Mem 因 owner 要求执行
[`全产品 replay`](tasks/mem/mem-current-product-replay-2026-09.md)，先裁决
`CONFIRMED/METHOD_GAP/PRODUCT_TRIGGER`，不预先改写其 `CLOSED` 结论。

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
