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
| mem | CLOSED | P0 / [PASS](tasks/mem/mem-v10-external-byte-reservation.md) | [record](tasks/mem/mem-v10-external-byte-reservation.md) | mem v10 CLOSED：M100/M101 外部 Reservation 与 Graph 预算尾项完成；复用 W1 不变实现与三交付，补齐 WSL/两 Mac、ASan/UBSan/TSan、红队和 P0；原 allocator 路径/布局无变化。 |
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
| compute | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| graph | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| container_concurrent | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_concurrent v2 CLOSED：并发表能力、三平台/三交付、文档、红队与约 134ms 同合同 membership 双顺序 PARITY、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_queue | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_queue v2 CLOSED：queue 能力、三平台/三交付、文档、红队与同合同 unbounded 4P4C handoff 双顺序 WIN、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_text | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container-text-base64.md) / 5% | [record](tasks/container/container-text-v3-base64.md) | container_text v3 CLOSED：Base64/Base64URL caller-buffer API、显式策略与 simdutf 私有 bulk core 通过三平台正确性、ASan/UBSan、Container 三交付、易用性和红队；W1-WIN 两个真实形状的 5% P1 非劣门全过。 |
| io_watch | CLOSED | P0 / [PASS](tasks/io/io-watch-v2-recursive-tree.md) | [record](tasks/io/io-watch-v2-recursive-tree.md) | io_watch v2 CLOSED：显式递归树、root-relative bounded stream、rescan/cap 与三平台 backend 通过三平台、sanitizer、三交付、红队；direct 默认成本隔离，Linux 建树 O(D)、普通 descriptor lookup O(1)。 |
| process | CLOSED | P0 / [PASS](tasks/process/process-v1-managed-child.md) | [record](tasks/process/process-v1-managed-child.md) | process v1 CLOSED：受管 child 的 argv/env/cwd/stdio、HYLE-native wait/pipe、三平台非轮询 reap 与 tree termination 已通过四平台环境、三 sanitizer、三交付和终局红队；P0 成本与平台限制已如实记录。 |
| io_websocket | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-websocket-v2-gcc-coroutine-allocator-use.md) | io_websocket v2 CLOSED：两个私有 coroutine 显式保留 frame allocator 使用；GCC 14.2 与 MSVC 19.50 shared/WERROR 定向编译和测试通过，产品 API/算法/成本及 v1 P1 结论不变。 |
| compute_provider_onednn | BLOCKED | — | [record](tasks/compute/llm-moe-c1-onednn-normalization.md) | C1-AA-Q2 BLOCKED：16个A/A进程与完整实例原记录保留，原门FAIL且产品采样0；Q2-D目录接线修复及12项Python回归通过，无native/build/P1重试。F1正确性/三交付不变，默认规则/淘汰未裁决，其他C1独立验收继续。 |
| compute_provider_ggml | BLOCKED | — | [record](tasks/compute/compute-provider-ggml-v1.md) | compute_provider_ggml v1 BLOCKED：HYLE block storage 与 Graph region 前置已经解除，但锁定/当前 ggml CPU runtime 仍在 context/threadpool 创建失败时 abort，且无持久 pool 时每次 submit 隐藏创建 disposable pool；空 factor 继续不进入产品。 |
| graph_provider_onednn | BLOCKED | — | [record](tasks/compute/llm-moe-c1-onednn-normalization.md) | C1-AA-Q2 BLOCKED：16个A/A进程与完整实例原记录保留，原门FAIL且产品采样0；Q2-D目录接线修复及12项Python回归通过，无native/build/P1重试。F1正确性/三交付不变，默认规则/淘汰未裁决，其他C1独立验收继续。 |
| compute_ops | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| compute_backend_cpu | CLOSED | P0 / [PASS](tasks/build/vulkan-r6-product-delivery.md) | [record](tasks/build/vulkan-r6-product-delivery.md) | R6 Vulkan 交付 CLOSED：D1–D3、Windows/Air 最终三交付、relocation/隐藏 producer/offline/DLL 邻置及全部 111 个 selected gates 完成；十三旧 owner 的 C++/API 不变，原运行时/P1、限定域与 CPU 例外继承。 |
| compute_backend_cuda | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| compute_provider_native | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| compute_sched | QUEUED | — | — | compute_sched identity 已预留：未来审计 Platform process/topology 与 Sched persistent workers 到 Compute ParallelExecutorRef 的自然 CPU execution-pool bridge；尚无依赖、产品入口或实现。 |
| compute_route_cpu | CLOSED | P0 / [PASS](tasks/build/vulkan-r6-product-delivery.md) | [record](tasks/build/vulkan-r6-product-delivery.md) | R6 Vulkan 交付 CLOSED：D1–D3、Windows/Air 最终三交付、relocation/隐藏 producer/offline/DLL 邻置及全部 111 个 selected gates 完成；十三旧 owner 的 C++/API 不变，原运行时/P1、限定域与 CPU 例外继承。 |
| graph_route_cpu | CLOSED | P0 / [PASS](tasks/build/vulkan-r6-product-delivery.md) | [record](tasks/build/vulkan-r6-product-delivery.md) | R6 Vulkan 交付 CLOSED：D1–D3、Windows/Air 最终三交付、relocation/隐藏 producer/offline/DLL 邻置及全部 111 个 selected gates 完成；十三旧 owner 的 C++/API 不变，原运行时/P1、限定域与 CPU 例外继承。 |
| compute_provider_cublas | CLOSED | P1 / [PASS](evidence/performance/benchmarks-compute-graph.md) / 5% | [record](tasks/compute/cuda-v1-four-route-foundation.md) | R4 CLOSED：自然 CUDA Direct/Graph、cuBLASLt/Marlin/SGLang 与 HYLE 保留域、原生资源/失败终态及冷默认已交付；Win/WSL 三交付与插桩、两 Mac OFF 隔离、最终默认5% P1及原生边界成本验收完成。资格限于已测 SM89/软件组合，不签普遍 SOTA；CPU 例外保留。 |
| compute_provider_marlin | CLOSED | P0 / [PASS](tasks/build/cuda-vendor-content-audit-v1.md) | [record](tasks/build/cuda-vendor-content-audit-v1.md) | CUDA vendor audit CLOSED：V1/V2 正规范摘要与 enabled 消费前校验完成；Windows/Air source 正负检查和最终 selector 通过。纯配置 P0，R4 运行代码、限定 P1 资格与方向继承，无 GPU 重测。 |
| compute_provider_cuda | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| compute_route_cuda | CLOSED | P0 / [PASS](tasks/build/vulkan-r6-product-delivery.md) | [record](tasks/build/vulkan-r6-product-delivery.md) | R6 Vulkan 交付 CLOSED：D1–D3、Windows/Air 最终三交付、relocation/隐藏 producer/offline/DLL 邻置及全部 111 个 selected gates 完成；十三旧 owner 的 C++/API 不变，原运行时/P1、限定域与 CPU 例外继承。 |
| graph_route_cuda | CLOSED | P0 / [PASS](tasks/build/vulkan-r6-product-delivery.md) | [record](tasks/build/vulkan-r6-product-delivery.md) | R6 Vulkan 交付 CLOSED：D1–D3、Windows/Air 最终三交付、relocation/隐藏 producer/offline/DLL 邻置及全部 111 个 selected gates 完成；十三旧 owner 的 C++/API 不变，原运行时/P1、限定域与 CPU 例外继承。 |
| compute_route_apple | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| graph_route_apple | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| compute_route_vulkan | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| graph_route_vulkan | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |
| compute_provider_ggml_vulkan | BLOCKED | — | [record](tasks/compute/llm-moe-self-operator-freeze.md) | 旧 C1 授权撤回：用户要求按全模型需求先接成熟 Provider，所有自实现冻结；已有局部证据保留，未签旧轮完成，等待新 M0 scope 移交。 |

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
