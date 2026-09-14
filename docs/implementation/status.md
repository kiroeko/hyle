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
| mem | CLOSED | P1 / [PASS](evidence/performance/benchmarks-mem-v7.md) / 5% | [record](tasks/mem/mem-v8-resource-type-visibility.md) | mem v8 CLOSED：MemoryResource 类级 export 已修复 default-hidden shared 类型身份；三平台完整下游交付通过，API、layout、虚表、算法和运行时成本不变，v7 P1 证据继续有效。 |
| mem_vm | CLOSED | P1 / [PASS](evidence/performance/benchmarks-mem-vm-v2.md) / 5% | [record](tasks/mem_vm/mem-vm-v3-resource-visibility-replay.md) | mem-vm v3 CLOSED：源码无独立修改；继承 Mem 类型可见性修复后，三平台跨 factor source/static/shared 交付闭包通过，v2 P1 证据继续有效。 |
| simd | CLOSED | P1 / [PASS](evidence/performance/benchmarks-simd.md) / 5% | [record](tasks/simd/simd-v7-compute-row-primitives.md) | simd-v7 CLOSED：Compute 驱动的 exact reciprocal-sqrt、number min/max、byte-bool select 与确定 signed-zero extrema 已交付；三平台 correctness/sanitizer、source/static/shared、零额外 exact route codegen、终局红队和一次正式同合同 P1 全部通过，四组 row 均为 PARITY/5% NI PASS。 |
| container | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container-v6 CLOSED：sequential 能力、三平台/三交付、文档、红队与同合同 weighted-LRU 8/8 WIN、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| util_runtime | CLOSED | P0 / [PASS](evidence/performance/benchmarks-util.md) | [record](tasks/util/util-runtime-v5-blocker-resolution.md) | util-runtime v5 CLOSED：三条 blocker 已关闭；三平台、三交付、WSL sanitizers、decorated C++ exports、易用性/文档、候选后红队与 P0 结构成本全部通过。 |
| sched | CLOSED | P1 / [PASS](tasks/sched/sched-v10-execution-priority.md) / 5% | [record](tasks/sched/sched-v11-private-fcontext-export.md) | sched v11 CLOSED：三平台三交付与实际共享导出扫描通过；Windows raw fcontext 不再公开，依赖闭包的 Mem 类型身份已修复，v10 P1 证据继续有效。 |
| io_core | CLOSED | P0 / [PASS](tasks/io/io-core-v5-anonymous-stream-adopt.md) | [record](tasks/io/io-core-v5-anonymous-stream-adopt.md) | io_core v5 CLOSED：唯一 LocalStream::adopt_anonymous seam 与 Windows 同步 terminal completion 修正已通过四平台环境、sanitizer、三交付和终局红队；既有 transport/layout/普通 I/O 热路径不变。 |
| io_tls | CLOSED | P0 / [PASS](tasks/io/io-tls-v4-context-identity.md) | [record](tasks/io/io-tls-v4-context-identity.md) | io_tls v4 CLOSED：G06 HTTPS file-body 的 inline context identity seam 保持零字段、零符号和普通热路径零成本，通过三平台、sanitizer、三交付与红队。 |
| io_http | CLOSED | P0 / [PASS](tasks/io/io-http-v4-upgrade-handoff.md) | [record](tasks/io/io-http-v4-upgrade-handoff.md) | io_http v4 CLOSED：协议无关 Upgrade owner 以 O(1) move 安全移交 transport 与预读 bytes，普通 HTTP 零常态成本；三平台、sanitizer、三交付、直接 WebSocket consumer 与红队全部通过。 |
| io_http2 | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-http2-v2-owner-progress-cost.md) | io_http2 v2 CLOSED：4060e217 的 owner-progress 修正通过三平台、sanitizer、三交付与终局红队；唯一同合同 P1 中 64x1KiB 双顺序 WIN、4x256KiB 双顺序 PARITY，固定 5% 非劣门全部 PASS。 |
| io_transfer | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-transfer-v1.md) | io_transfer v1 CLOSED：可靠 local/HTTP(S) transfer、resume、checksum 与 atomic publish 通过三平台、sanitizer、三交付、易用性和红队；P1 的 local publish 非劣、Range resume 双顺序 WIN，5% 门全过。 |
| compute | AUDITED | — | [record](tasks/compute/compute-v7-typed-operation-facade.md) | compute v7 AUDITED：19-family typed prepare 与 Concat/TopK/masked-SDPA submit 缺口已对账；有限修正只用 rank-bounded stack descriptor/span 转发现有 generic runtime，删除冲突的早期 RMSNorm/Gated 显式-output 语义，不改 provider steady path。 |
| graph | AUDITED | — | [record](tasks/compute/graph-v8-explicit-allocator-sort.md) | graph v8 AUDITED：G82/G83 产品算法不变；ComputeOps O18 与 Compute v7 改变直接依赖候选身份，Graph 退回后只回放既有 allocator/selection 验收及批量 MatMul→separate gated 多节点纵切，再共同签发候选。 |
| container_concurrent | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_concurrent v2 CLOSED：并发表能力、三平台/三交付、文档、红队与约 134ms 同合同 membership 双顺序 PARITY、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_queue | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_queue v2 CLOSED：queue 能力、三平台/三交付、文档、红队与同合同 unbounded 4P4C handoff 双顺序 WIN、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_text | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container-text-base64.md) / 5% | [record](tasks/container/container-text-v3-base64.md) | container_text v3 CLOSED：Base64/Base64URL caller-buffer API、显式策略与 simdutf 私有 bulk core 通过三平台正确性、ASan/UBSan、Container 三交付、易用性和红队；W1-WIN 两个真实形状的 5% P1 非劣门全过。 |
| io_watch | CLOSED | P0 / [PASS](tasks/io/io-watch-v2-recursive-tree.md) | [record](tasks/io/io-watch-v2-recursive-tree.md) | io_watch v2 CLOSED：显式递归树、root-relative bounded stream、rescan/cap 与三平台 backend 通过三平台、sanitizer、三交付、红队；direct 默认成本隔离，Linux 建树 O(D)、普通 descriptor lookup O(1)。 |
| process | CLOSED | P0 / [PASS](tasks/process/process-v1-managed-child.md) | [record](tasks/process/process-v1-managed-child.md) | process v1 CLOSED：受管 child 的 argv/env/cwd/stdio、HYLE-native wait/pipe、三平台非轮询 reap 与 tree termination 已通过四平台环境、三 sanitizer、三交付和终局红队；P0 成本与平台限制已如实记录。 |
| io_websocket | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-websocket-v2-gcc-coroutine-allocator-use.md) | io_websocket v2 CLOSED：两个私有 coroutine 显式保留 frame allocator 使用；GCC 14.2 与 MSVC 19.50 shared/WERROR 定向编译和测试通过，产品 API/算法/成本及 v1 P1 结论不变。 |
| compute_provider_onednn | AUDITED | — | [record](tasks/compute/compute-provider-onednn-v4-region-kernel-seam.md) | compute_provider_onednn v4 AUDITED：OD40--OD44 私有 BRGEMM mechanism 已在 6306ab2f 抽取；公开 provider/API/route 不变，pair episode、OOM rollback、六 shape warm 零分配与新鲜 W1-WIN 5% P1 非劣门通过。真实 graph_provider_onednn 的 source/static/shared 消费与组合红队仍待完成，尚非 Candidate。 |
| compute_provider_ggml | BLOCKED | — | [record](tasks/compute/compute-provider-ggml-v1.md) | compute_provider_ggml v1 BLOCKED：HYLE block storage 与 Graph region 前置已经解除，但锁定/当前 ggml CPU runtime 仍在 context/threadpool 创建失败时 abort，且无持久 pool 时每次 submit 隐藏创建 disposable pool；空 factor 继续不进入产品。 |
| graph_provider_onednn | AUDITED | — | [record](tasks/compute/graph-provider-onednn-v3-sequence-spi.md) | graph_provider_onednn v3 AUDITED：OR60--OR62 冻结 region EnqueueResult、首次实际执行 acceptance 与 extension version 迁移；公开 factory/kernel/layout 不变，P0 消除每-region Completion，P1 等完整 Runtime/Graph candidate。 |
| compute_ops | AUDITED | — | [record](tasks/compute/compute-ops-v1-schema-and-catalog.md) | compute_ops v1 AUDITED：Native batch-broadcast MatMul 纵切证明 rank≥3 转置权重只能手写完整 permutation；有限 O18 将 transposed() 自然扩展为交换末两轴，rank-2 行为不变，不增加 material transpose 或 provider 策略。 |
| compute_backend_cpu | CLOSED | P0 / [PASS](tasks/compute/compute-backend-cpu-v1-factor.md) | [record](tasks/compute/compute-backend-cpu-v1-factor.md) | compute_backend_cpu v1 CLOSED：显式单 CPU Device/Context、caller-thread ordered Queue、system MemoryResource、copy/fill、Completion 与 typed host submit 通过 factor-only、三平台、sanitizer、三交付和红队；warm 路线零 allocation/lock/syscall/search。 |
| compute_backend_cuda | CLOSED | P1 / [PASS](evidence/performance/benchmarks-compute-graph.md) / 5% | [record](tasks/compute/compute-backend-cuda-v1-async-runtime.md) | compute_backend_cuda v1 CLOSED：CUDA 13.3 explicit-context runtime、stable resources、event Completion/reaper、typed interop、single-queue capture 与 CUDA-on 三交付通过正确性、易用性和红队；等价 Runtime copy/event P1 八格 PARITY/PASS。 |
| compute_provider_native | AUDITED | — | [record](tasks/compute/compute-provider-native-v1-factor.md) | compute_provider_native v1 AUDITED：有限 N18/N19 补 rank>=2 batch-broadcast MatMul 与 packed/separate 三种 GatedActivation；N20 证明 fat-kernel selector 直接消费 Platform CPU facts，故正确保留该边；不重开底层模块。 |
| compute_sched | QUEUED | — | — | compute_sched identity 已预留：未来审计 Platform process/topology 与 Sched persistent workers 到 Compute ParallelExecutorRef 的自然 CPU execution-pool bridge；尚无依赖、产品入口或实现。 |

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
