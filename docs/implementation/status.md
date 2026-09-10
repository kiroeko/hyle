# 当前模块施工状态

> 更新时间：2026-09-09。唯一机器真相是 [`module-status.json`](module-status.json)；下表由
> `python scripts/check-module-closure.py` 校验并可用 `--write-status` 更新。architecture、user、历史 task、性能页标题或 summary
> 中的状态词都不能覆盖 registry。

`CLOSED` 就是当前方法下已经完成全部适用工程门、应停止日常施工并进入变更控制的唯一 terminal
状态。Engineering performance 的 `PASS` 直接链接本轮精确证据；它不是跨平台、跨 workload 的
发布级 SOTA 证书。范围、机器、限制、重开条件和 LOC 等完整事实保留在当前 record/evidence 中。

<!-- BEGIN GENERATED MODULE STATUS -->

| Component | State | Engineering performance | Current record | Summary |
|---|---|---|---|---|
| core | CLOSED | P0 / [PASS](evidence/performance/benchmarks-core.md) | [record](tasks/core/core-v5-assert-macro-contract.md) | core-v5 CLOSED：断言宏卫生、Release/Debug 编译语义、字段边界和 observer 生命周期文档已修复；公开 API 不变，三平台正确性、三种 C++ 交付、文档/易用性、红队与 P0 结构性能全部通过。 |
| container_intrusive | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container-intrusive.md) / 5% | [record](tasks/container_intrusive/container-intrusive-v4-capability-extension.md) | container_intrusive v4 CLOSED：counted safe-list、一指针 cached-tail forward-list、auto/safe hlist 与稳定 sequence algorithms 已完成三平台正确性、三种 C++ 交付、直接 Sched 消费、文档/红队及 W1-WIN 四项 5% P1 工程门验证。 |
| util_core | CLOSED | P1 / [PASS](evidence/performance/benchmarks-util-core-v4.md) / 5% | [record](tasks/util/util-core-v4-resource-move-ctad.md) | Util-core v4 CLOSED：move-only deleter、失败安全、checked sentinel 完整不抛判定和无歧义 member CTAD 已完成三平台正确性、三种 C++ 交付、文档/易用性、红队、中央组件合同及固定 FunctionRef 5% P1 工程门验证。 |
| error | CLOSED | P1 / [PASS](evidence/performance/benchmarks-error.md) / 5% | [record](tasks/error/error-v5-capability-completion.md) | error-v5 CLOSED：WinSock、Result<T&>、bounded format_status 与 std::error_code adapter 已补齐；既有 Status/Result 无回归，三平台正确性和三种 C++ 交付、易用性/文档、红队及 W1-WIN 五个同合同 consumer 的 5% P1 非劣门全部通过。 |
| obs | CLOSED | P1 / [PASS](evidence/performance/benchmarks-obs.md) / 5% | [record](tasks/obs/obs-v5-metrics-primitives.md) | obs v5 CLOSED：生产级 counter、gauge、fixed histogram、batch observe、snapshot merge 与 bucket quantile 已完成三平台 correctness/sanitizer、三种 C++ 交付、易用性/docs、依赖和红队；W1-WIN owner 4/4 WIN、shared 4/4 PARITY 且 5% 非劣门全过。 |
| math | CLOSED | P0 / [PASS](evidence/performance/benchmarks-math-v4.md) | [record](tasks/math/math-v4-checked-discrete-sizing.md) | math-v4 CLOSED：checked scalar lcm/bit-ceil 已补齐；三平台 correctness/UBSan/逐头/三交付、真实 package consumer、文档/红队与 P0 原生结构成本全部通过。 |
| time | CLOSED | P0 / [PASS](tasks/time/time-v5-interoperability.md) | [record](tasks/time/time-v5-interoperability.md) | time-v5 CLOSED：signed Unix units、absolute/relative POSIX、timeout rounding、direct FILETIME 与 header isolation 已通过三平台正确性/WSL sanitizers、三种 C++ 交付、文档/红队和 P0 结构性能门；候选 680c9941/b09b7043 无产品漂移。 |
| timer_queue | CLOSED | P1 / [PASS](evidence/performance/benchmarks-time.md) / 5% | [record](tasks/timer_queue/timer-queue-v1-factor.md) | timer_queue v1 CLOSED：caller-owned radix/common-timeout 已成为独立 header-only factor，provider 组合与 Sched/IO 直接边正确；三平台正确性/sanitizer、三种 C++ 交付、文档/红队及对当前 Seastar 的 8/8 P1 WIN 全部通过。 |
| platform | CLOSED | P1 / [PASS](evidence/performance/benchmarks-platform.md) / 5% | [record](tasks/platform/platform-v11-perf-control-resume.md) | platform v11 CLOSED：NUMA node memory fact 与原有 host facts 已完成三平台 correctness、三交付、API/docs、依赖和红队；唯一 fresh W1-WIN receipt 全 controls 与 4/4 host-planning NI PASS，未改 3c1ef239/63b58d10 产品候选重新确认。 |
| platform_thread | CLOSED | P0 / [PASS](evidence/performance/benchmarks-platform.md) | [record](tasks/platform/platform-v8-os-resource-contract.md) | platform_thread v8 CLOSED：Parker 产品 bytes 未改，Rust MIT provenance/lock/license/package 完整；三平台正确性、三交付、直接消费者、易用性、红队与 P0 结构门全部通过。 |
| platform_fs | CLOSED | P1 / [PASS](evidence/performance/benchmarks-platform.md) / 5% | [record](tasks/platform_fs/platform-fs-v2-control-resume.md) | platform_fs v2 CLOSED：未变化的 v8 第二候选取得唯一 fresh W1-WIN receipt；31 个独立进程下 controls 全 PASS，8/8 同合同 warm-file strata 为 PARITY 且 5% NI PASS，无固定方向产品劣化。 |
| mem_reclaim | CLOSED | P1 / [PASS](evidence/performance/benchmarks-mem-reclamation.md) / 5% | [record](tasks/mem/mem-reclaim-v2-inference-resume.md) | mem_reclaim v2 CLOSED：推理 read-mostly snapshot 回收、零分配 typed retire 与 Container 直接 consumer 完成；三平台正确性/sanitizer/三交付、文档/红队及官方 Folly 同合同 P1 全部通过。 |
| mem | CLOSED | P1 / [PASS](evidence/performance/benchmarks-mem-v7.md) / 5% | [record](tasks/mem/mem-v7-logical-range-suballocator.md) | mem v7 CLOSED：payload-free 64-bit aligned RangeAllocator、固定 metadata、RAII individual release、generation bulk reset 与 O(1) summary 已完成三平台正确性、source/shared/static 交付、文档/红队及 W1-WIN 同合同 8/8 strata 的 5% P1 非劣门。 |
| mem_vm | CLOSED | P1 / [PASS](evidence/performance/benchmarks-mem-vm-v2.md) / 5% | [record](tasks/mem_vm/mem-vm-v2-inference-mapping.md) | mem_vm v2 CLOSED：Platform 单一 OS authority、hole-aware FileMapping、headerless SharedMapping 与 LargeMapping 已完成三平台正确性/sanitizer、三交付/direct-consumer、文档/易用性、红队与 W1-WSL 4/4 5% P1 非劣门。 |
| simd | CLOSED | P1 / [PASS](evidence/performance/benchmarks-simd.md) / 5% | [record](tasks/simd/simd-v6-provider-consumer-replay.md) | simd-v6 CLOSED：当前推理能力缺口已补齐或路由，三平台正确性/ASan、三交付、387/422/0 路由、文档/易用性、红队和 W1-WIN 三行同合同 5% P1 非劣门全部通过。 |
| container | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container-v6 CLOSED：sequential 能力、三平台/三交付、文档、红队与同合同 weighted-LRU 8/8 WIN、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| util_runtime | CLOSED | P0 / [PASS](evidence/performance/benchmarks-util.md) | [record](tasks/util/util-runtime-v5-blocker-resolution.md) | util-runtime v5 CLOSED：三条 blocker 已关闭；三平台、三交付、WSL sanitizers、decorated C++ exports、易用性/文档、候选后红队与 P0 结构成本全部通过。 |
| sched | AUDITED | — | [record](tasks/sched/sched-v8-task-scope-distributed-completion.md) | sched v8 AUDITED：逐 child、driver 同分配的 completion record 与 publisher/scope 双引用已冻结；有限 S8-01..S8-05 只替换 TaskScope accounting，普通 task、G08、依赖图和性能合同不变。 |
| io_core | CLOSED | P0 / [PASS](tasks/io/io-core-v2-file-watch-submission.md) | [record](tasks/io/io-core-v2-file-watch-submission.md) | io_core v2 CLOSED：Windows IOCP directory-change seam 在 67177e41 冻结；三平台 io_core、三 sanitizer、header、source/static/shared/relocation、红队与零常态附加成本 P0 全部通过。 |
| io_tls | CLOSED | P0 / [PASS](tasks/io/io-tls-v3-credential-sources.md) | [record](tasks/io/io-tls-v3-credential-sources.md) | io_tls v3 CLOSED：file/borrowed PEM、encrypted-key borrowed password/FunctionRef callback、client/server identity 与 mTLS 固定在 4bd86adf；三平台、ASan/UBSan、直接消费者、TLS-on 三交付、TLS-off、红队与零连接热路附加成本 P0 全部通过。 |
| io_http | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-http-v2-duplex-progress.md) | io_http v2 CLOSED：双向 progress 组合修复通过三平台、exact sanitizer、三交付、易用性和红队；唯一 HTTP/1 P1 为 1 KiB 双顺序 PARITY、64 KiB 双顺序 WIN，固定 5% 非劣门全部通过。 |
| io_http2 | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-http2-v2-owner-progress-cost.md) | io_http2 v2 CLOSED：4060e217 的 owner-progress 修正通过三平台、sanitizer、三交付与终局红队；唯一同合同 P1 中 64x1KiB 双顺序 WIN、4x256KiB 双顺序 PARITY，固定 5% 非劣门全部 PASS。 |
| io_transfer | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-transfer-v1.md) | io_transfer v1 CLOSED：可靠 local/HTTP(S) transfer、resume、checksum 与 atomic publish 通过三平台、sanitizer、三交付、易用性和红队；P1 的 local publish 非劣、Range resume 双顺序 WIN，5% 门全过。 |
| compute | QUEUED | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
| graph | QUEUED | — | — | 实现尚不完整；前置与真实 consumer slice 建立前保持可变。 |
| container_concurrent | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_concurrent v2 CLOSED：并发表能力、三平台/三交付、文档、红队与约 134ms 同合同 membership 双顺序 PARITY、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_queue | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_queue v2 CLOSED：queue 能力、三平台/三交付、文档、红队与同合同 unbounded 4P4C handoff 双顺序 WIN、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_text | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container-text-base64.md) / 5% | [record](tasks/container/container-text-v3-base64.md) | container_text v3 CLOSED：Base64/Base64URL caller-buffer API、显式策略与 simdutf 私有 bulk core 通过三平台正确性、ASan/UBSan、Container 三交付、易用性和红队；W1-WIN 两个真实形状的 5% P1 非劣门全过。 |
| io_watch | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 10% | [record](tasks/io/io-watch-v1-file-change-stream.md) | io_watch v1 CLOSED：跨平台 bounded 文件变化流完成三平台正确性/sanitizer、公开 IO 三交付、易用性、红队和同合同 libuv P1；原预声明 10% 工程非劣门全过，严格 5% 附加诊断仅在 WSL exact-file 两格未决且无 LOSE。 |

<!-- END GENERATED MODULE STATUS -->

产品状态流固定为：

```text
normal:  QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> CLOSED
stop:    SCOPED | AUDITED | CANDIDATE -> BLOCKED
resume:  BLOCKED -> SCOPED
reopen:  CLOSED -> SCOPED（新 trigger + 新 record）
```

状态、回放与 reopen 的精确定义见
[模块重构与收口协议](../developer/review/module-closure-methodology.md)。历史施工原因从表中 `record` 进入；
不要在本页追加某一模块的阶段性叙事。
