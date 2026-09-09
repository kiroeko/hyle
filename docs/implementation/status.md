# 当前模块施工状态

> 更新时间：2026-09-08。唯一机器真相是 [`module-status.json`](module-status.json)；下表由
> `python scripts/check-module-closure.py` 校验。architecture、user、历史 task、性能页标题或 summary
> 中的状态词都不能覆盖 registry。

`Product` 是本轮状态机；`Freeze` 是独立的当前方法复核确认。只有 registry 中存在完整
`freeze_confirmation` 的 `CLOSED` 模块才显示 `FROZEN`，旧 `CLOSED` 不会因文字描述被自动升级。
Engineering performance 的 `PASS` 直接链接本轮精确证据；它与 Formal 发布证书是两条轴。

<!-- BEGIN GENERATED MODULE STATUS -->
| Component | Product | Freeze | Tier | Engineering performance | Formal | Current record | Summary |
|---|---|---|---|---|---|---|---|
| core | CLOSED | FROZEN | P0 | [PASS](evidence/performance/benchmarks-core.md) / STRUCTURAL | NONE | [task](tasks/core/core-v5-assert-macro-contract.md) | core-v5 CLOSED：断言宏卫生、Release/Debug 编译语义、字段边界和 observer 生命周期文档已修复；公开 API 不变，三平台正确性、三种 C++ 交付、文档/易用性、红队与 P0 结构性能全部通过。 |
| container_intrusive | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-container-intrusive.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/container_intrusive/container-intrusive-v4-capability-extension.md) | container_intrusive v4 CLOSED：counted safe-list、一指针 cached-tail forward-list、auto/safe hlist 与稳定 sequence algorithms 已完成三平台正确性、三种 C++ 交付、直接 Sched 消费、文档/红队及 W1-WIN 四项 5% P1 工程门验证。 |
| util_core | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-util-core-v4.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/util/util-core-v4-resource-move-ctad.md) | Util-core v4 CLOSED/FROZEN：move-only deleter、失败安全、checked sentinel 完整不抛判定和无歧义 member CTAD 已完成三平台正确性、三种 C++ 交付、文档/易用性、红队、中央组件合同及固定 FunctionRef 5% P1 工程门验证。 |
| error | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-error.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/error/error-v5-capability-completion.md) | error-v5 CLOSED：WinSock、Result<T&>、bounded format_status 与 std::error_code adapter 已补齐；既有 Status/Result 无回归，三平台正确性和三种 C++ 交付、易用性/文档、红队及 W1-WIN 五个同合同 consumer 的 5% P1 非劣门全部通过。 |
| obs | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-obs.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/obs/obs-v5-metrics-primitives.md) | obs v5 CLOSED/FROZEN：生产级 counter、gauge、fixed histogram、batch observe、snapshot merge 与 bucket quantile 已完成三平台 correctness/sanitizer、三种 C++ 交付、易用性/docs、依赖和红队；W1-WIN owner 4/4 WIN、shared 4/4 PARITY 且 5% 非劣门全过。 |
| math | CLOSED | FROZEN | P0 | [PASS](evidence/performance/benchmarks-math-v4.md) / STRUCTURAL | NONE | [task](tasks/math/math-v4-checked-discrete-sizing.md) | math-v4 CLOSED/FROZEN：checked scalar lcm/bit-ceil 已补齐；三平台 correctness/UBSan/逐头/三交付、真实 package consumer、文档/红队与 P0 原生结构成本全部通过。 |
| time | CLOSED | FROZEN | P0 | [PASS](tasks/time/time-v5-interoperability.md) / STRUCTURAL | PENDING | [task](tasks/time/time-v5-interoperability.md) | time-v5 CLOSED：signed Unix units、absolute/relative POSIX、timeout rounding、direct FILETIME 与 header isolation 已通过三平台正确性/WSL sanitizers、三种 C++ 交付、文档/红队和 P0 结构性能门；候选 680c9941/b09b7043 无产品漂移。 |
| timer_queue | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-time.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/timer_queue/timer-queue-v1-factor.md) | timer_queue v1 CLOSED：caller-owned radix/common-timeout 已成为独立 header-only factor，provider 组合与 Sched/IO 直接边正确；三平台正确性/sanitizer、三种 C++ 交付、文档/红队及对当前 Seastar 的 8/8 P1 WIN 全部通过。 |
| platform | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-platform.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/platform/platform-v11-perf-control-resume.md) | platform v11 CLOSED/FROZEN：NUMA node memory fact 与原有 host facts 已完成三平台 correctness、三交付、API/docs、依赖和红队；唯一 fresh W1-WIN receipt 全 controls 与 4/4 host-planning NI PASS，未改 3c1ef239/63b58d10 产品重新冻结。 |
| platform_thread | CLOSED | FROZEN | P0 | [PASS](evidence/performance/benchmarks-platform.md) / STRUCTURAL | PENDING | [task](tasks/platform/platform-v8-os-resource-contract.md) | platform_thread v8 CLOSED/FROZEN：Parker 产品 bytes 未改，Rust MIT provenance/lock/license/package 完整；三平台正确性、三交付、直接消费者、易用性、红队与 P0 结构门全部通过。 |
| platform_fs | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-platform.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/platform_fs/platform-fs-v2-control-resume.md) | platform_fs v2 CLOSED/FROZEN：未变化的 v8 第二候选取得唯一 fresh W1-WIN receipt；31 个独立进程下 controls 全 PASS，8/8 同合同 warm-file strata 为 PARITY 且 5% NI PASS，无固定方向产品劣化。 |
| mem_reclaim | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-mem-reclamation.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/mem/mem-reclaim-v2-inference-resume.md) | mem_reclaim v2 CLOSED：推理 read-mostly snapshot 回收、零分配 typed retire 与 Container 直接 consumer 完成；三平台正确性/sanitizer/三交付、文档/红队及官方 Folly 同合同 P1 全部通过。 |
| mem | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-mem-v6.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/mem/mem-v6-comprehensive-rebuild.md) | mem v6 CLOSED/FROZEN：九族超集 COMPLETE/ROUTED，三平台正确性/sanitizer、三交付/direct-consumer、文档/易用性、红队与 W1-WIN 8/8 5% P1 非劣门全部通过。 |
| mem_vm | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-mem-vm-v2.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/mem_vm/mem-vm-v2-inference-mapping.md) | mem_vm v2 CLOSED/FROZEN：Platform 单一 OS authority、hole-aware FileMapping、headerless SharedMapping 与 LargeMapping 已完成三平台正确性/sanitizer、三交付/direct-consumer、文档/易用性、红队与 W1-WSL 4/4 5% P1 非劣门。 |
| simd | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-simd.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/simd/simd-v6-provider-consumer-replay.md) | simd-v6 CLOSED/FROZEN：当前推理能力缺口已补齐或路由，三平台正确性/ASan、三交付、387/422/0 路由、文档/易用性、红队和 W1-WIN 三行同合同 5% P1 非劣门全部通过。 |
| container | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-container.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/container/container-v6-identity-p1-resume.md) | container-v6 CLOSED/FROZEN：sequential 能力、三平台/三交付、文档、红队与同合同 weighted-LRU 8/8 WIN、5% P1 非劣门全部通过；候选冻结在 111c28ea。 |
| util_runtime | CLOSED | FROZEN | P0 | [PASS](evidence/performance/benchmarks-util.md) / STRUCTURAL | NONE | [task](tasks/util/util-runtime-v5-blocker-resolution.md) | util-runtime v5 CLOSED/FROZEN：三条 blocker 已关闭；三平台、三交付、WSL sanitizers、decorated C++ exports、易用性/文档、候选后红队与 P0 结构成本全部通过。 |
| sched | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-sched.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/sched/sched-v5-io-completion-search.md) | sched-v5 CLOSED/FROZEN：IO completion 空闲转换缺陷已修复；三平台定向正确性、WSL sanitizers、直接交付、红队及 W1-WSL 自然 TLS consumer 双顺序 5% P1 非劣门全部通过，无公开 API 或普通无 IO 路线扩张。 |
| io_core | AUDITED | — | P1 | PLANNED / P1_ENGINEERING / 5% | PENDING | [task](tasks/io/io-v1-core-factorization.md) | io_core v1 处于 AUDITED：F01--F08 已实现，TLS/HTTP/HTTP2 source/header/test/dependency 已在 ea45d866 迁入独立 factor；F09 交付矩阵、F10 candidate/P1/红队仍待完成。 |
| io_tls | VERIFIED | — | P1 | [PASS](evidence/performance/benchmarks-io.md) / P1_ENGINEERING / 5% | NONE | [task](tasks/io/io-tls-v2-duplex-progress.md) | io_tls v2 VERIFIED：有界 provider pump 与 write-some 修复通过三平台、sanitizer、三交付、终局红队；唯一同合同 P1 的握手双顺序 WIN、small-record stream 双顺序非劣，全部七门通过。 |
| io_http | AUDITED | — | P1 | PLANNED / P1_ENGINEERING / 5% | NONE | [task](tasks/io/io-http-v2-duplex-progress.md) | io_http v2 AUDITED：TCP/detail immediate receive、HttpTransport dispatch、HTTP/2 bounded drain、TLS 分类与受影响回放已冻结为唯一 F01--F04。 |
| io_http2 | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-io.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/io/io-http2-v2-owner-progress-cost.md) | io_http2 v2 CLOSED/FROZEN：4060e217 的 owner-progress 修正通过三平台、sanitizer、三交付与终局红队；唯一同合同 P1 中 64x1KiB 双顺序 WIN、4x256KiB 双顺序 PARITY，固定 5% 非劣门全部 PASS。 |
| io_transfer | AUDITED | — | P1 | PLANNED / P1_ENGINEERING / 5% | NONE | [task](tasks/io/io-transfer-v1.md) | io_transfer v1 已完成模型/制品 bounded transfer、resume validator、checksum、atomic publish、owner 与自然 API 审计并进入 AUDITED；IOTRANSFER-F01--F06 是唯一有限实现集。 |
| compute | QUEUED | — | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
| graph | QUEUED | — | — | — | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
| container_concurrent | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-container.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/container/container-v6-identity-p1-resume.md) | container_concurrent v2 CLOSED/FROZEN：并发表能力、三平台/三交付、文档、红队与约 134ms 同合同 membership 双顺序 PARITY、5% P1 非劣门全部通过；候选冻结在 111c28ea。 |
| container_queue | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-container.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/container/container-v6-identity-p1-resume.md) | container_queue v2 CLOSED/FROZEN：queue 能力、三平台/三交付、文档、红队与同合同 unbounded 4P4C handoff 双顺序 WIN、5% P1 非劣门全部通过；候选冻结在 111c28ea。 |
| container_text | CLOSED | FROZEN | P1 | [PASS](evidence/performance/benchmarks-container.md) / P1_ENGINEERING / 5% | PENDING | [task](tasks/container/container-v6-identity-p1-resume.md) | container_text v2 CLOSED/FROZEN：六种 safe conversion、三平台/三交付、文档、红队与同合同 Unicode 8/8 PARITY、5% P1 非劣门全部通过；候选冻结在 3ba50461。 |
<!-- END GENERATED MODULE STATUS -->

产品状态流固定为：

```text
normal:  QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> VERIFIED -> CLOSED
stop:    SCOPED | AUDITED | CANDIDATE | VERIFIED -> BLOCKED
resume:  BLOCKED -> SCOPED
reopen:  CLOSED -> REOPENED -> SCOPED
```

状态、`FROZEN` confirmation、回放与 reopen 的精确定义见
[模块重构与收口协议](../developer/review/module-closure-methodology.md)。历史施工原因从表中 `task` 进入；
不要在本页追加某一模块的阶段性叙事。
