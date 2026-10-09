# 当前模块施工状态

> 唯一机器真相是 [`module-status.json`](module-status.json)；下表由
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
| platform | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-default-domain-routing/README.md) | [record](tasks/platform/cpu-identity-route-facts.md) | Platform I1 CLOSED：仅追加cached CpuIdentity事实；两平台CPU/header/实际普通C++三交付通过，非x86未知不宣称验收，P0无重复probe/heap。 |
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
| compute | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| graph | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| container_concurrent | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_concurrent v2 CLOSED：并发表能力、三平台/三交付、文档、红队与约 134ms 同合同 membership 双顺序 PARITY、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_queue | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_queue v2 CLOSED：queue 能力、三平台/三交付、文档、红队与同合同 unbounded 4P4C handoff 双顺序 WIN、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_text | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container-text-base64.md) / 5% | [record](tasks/container/container-text-v3-base64.md) | container_text v3 CLOSED：Base64/Base64URL caller-buffer API、显式策略与 simdutf 私有 bulk core 通过三平台正确性、ASan/UBSan、Container 三交付、易用性和红队；W1-WIN 两个真实形状的 5% P1 非劣门全过。 |
| io_watch | CLOSED | P0 / [PASS](tasks/io/io-watch-v2-recursive-tree.md) | [record](tasks/io/io-watch-v2-recursive-tree.md) | io_watch v2 CLOSED：显式递归树、root-relative bounded stream、rescan/cap 与三平台 backend 通过三平台、sanitizer、三交付、红队；direct 默认成本隔离，Linux 建树 O(D)、普通 descriptor lookup O(1)。 |
| process | CLOSED | P0 / [PASS](tasks/process/process-v1-managed-child.md) | [record](tasks/process/process-v1-managed-child.md) | process v1 CLOSED：受管 child 的 argv/env/cwd/stdio、HYLE-native wait/pipe、三平台非轮询 reap 与 tree termination 已通过四平台环境、三 sanitizer、三交付和终局红队；P0 成本与平台限制已如实记录。 |
| io_websocket | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-websocket-v2-gcc-coroutine-allocator-use.md) | io_websocket v2 CLOSED：两个私有 coroutine 显式保留 frame allocator 使用；GCC 14.2 与 MSVC 19.50 shared/WERROR 定向编译和测试通过，产品 API/算法/成本及 v1 P1 结论不变。 |
| compute_provider_onednn | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_provider_ggml | CLOSED | P0 / [PASS](../../evidence/correctness/compute-graph-macro-normalization/README.md) | [record](tasks/compute/compute-graph-macro-normalization.md) | 宏规范 CLOSED：目标facts/显式include/export/直接Core声明、三平台实际编译与三交付完成；原CPU数学、清退与43规则保持；不新签性能赢家。 |
| graph_provider_onednn | CLOSED | P0 / [PASS](tasks/compute/cpu-final-f1-build-contract.md) | [record](tasks/compute/cpu-final-f1-build-contract.md) | CPU最终F1 CLOSED：recipe/软件资格、同core原CPU端口、预编译程序与安全ISA/重放、逐合同成熟默认、oneDNN3.13.4和三平台单/具名混合三交付已验；原失败/skip保留。未签新的WIN。 |
| compute_ops | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-quant-mtp-c5-plain/cost-audit.md) | [record](tasks/compute/cpu-quant-mtp-coverage-terminal.md) | CPU热门量化/MTP/DSpark原准入合同CLOSED：固定104 GGUF及两新官方/三sidecar/四非GGUF typed+semantic/state联合资格、Windows/Linux三交付/真实失败并发资源、文档/独立终审与P0通过；不称整模型推理/SOTA、不改GPU旧状态。 |
| compute_backend_cpu | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_backend_cuda | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_provider_native | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_route_cpu | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| graph_route_cpu | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_provider_cublas | CLOSED | P0 / [PASS](../../evidence/correctness/llm-moe-d5-real-fragments/cuda-delivery/README.md) | [record](tasks/build/cuda-provider-runtime-exports-v1.md) | D5-A5 CLOSED：系统CUDA DLL身份与Provider私有runtime导出已补齐；三交付24项、459项模块测试（含一次旧名单断言纠正）、3组独立头通过。仅CMake修复，P0无热路径变化；不代签D5-A6。 |
| compute_provider_marlin | CLOSED | P0 / [PASS](../../evidence/correctness/llm-moe-d4-self-operator-retirement/README.md) | [record](tasks/compute/llm-moe-d4-self-operator-retirement.md) | M3-D4 CLOSED 91c2fb4d：无合格优势且由成熟 Provider 覆盖的 Marlin、CPU/CUDA/Apple/Vulkan 自有算子域已批量退场；四路线正确性、组件合同与 Compute/Graph 三交付通过。 |
| compute_provider_cuda | CLOSED | P1 / [PASS](../../evidence/performance/cuda-best-of-c2/README.md) / 5% | [record](tasks/compute/cuda-best-of-completion.md) | CLOSED：23精确CUDA叶优选、169成熟默认、14单承载；79重复程序实例清退。两OS三交付/Core63/严格profile资源回退及终审通过；原性能聚合和static批次FAIL保留，不外推全硬件/完整region。 |
| compute_route_cuda | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| graph_route_cuda | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_route_apple | AUDITED | — | [record](tasks/build/compute-mixed-shared-license-registration.md) | C3 CPU+Apple 真实配置发现 DS4 重复锁登记；已固定一个组件闭包条件修正，算子和路由不变。 |
| graph_route_apple | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_route_vulkan | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| graph_route_vulkan | CANDIDATE | — | [record](tasks/build/compute-unified-delivery.md) | 统一构建/冷路由候选已修正条件资产producer；Windows CPU/CUDA三交付消费者全部通过，其余平台/路线/mixed矩阵仍待验，H1–H3未实施。 |
| compute_provider_ggml_vulkan | CANDIDATE | — | [record](tasks/build/vulkan-macos-loader-discovery.md) | macOS Vulkan默认loader标准位置发现候选已冻结；不注入环境/安装driver，Air普通三交付待验。 |
| compute_ggml_runtime | CANDIDATE | — | [record](tasks/compute/vulkan-f32-matmul-admission-repair.md) | Vulkan F32同FIX限定完成已冻结；保留首次候选FAIL和原误差门，撤去显式F32的coop预转换，实际数值/交付待验。 |
| compute_provider_ggml_cuda | CLOSED | P0 / [PASS](tasks/compute/cuda-final-u6-audit.md) | [record](tasks/compute/cuda-final-u6-audit.md) | CUDA有限goal CLOSED：固定模型/量化/MTP完整原计算与状态、两OS CUDA-only/mixed三交付、cold成熟默认/真实能力门、六重复公式BORROW和具名unique保留、U5 42步/源码SDK增量及终审清理完成。AA原FAIL/性能INCONCLUSIVE无新WIN；按授权ENGINEERING_DEFAULT/P0关闭，不签全设备SOTA或推理engine。 |
| graph_provider_ggml_cuda | CLOSED | P0 / [PASS](tasks/compute/cuda-final-u6-audit.md) | [record](tasks/compute/cuda-final-u6-audit.md) | CUDA有限goal CLOSED：固定模型/量化/MTP完整原计算与状态、两OS CUDA-only/mixed三交付、cold成熟默认/真实能力门、六重复公式BORROW和具名unique保留、U5 42步/源码SDK增量及终审清理完成。AA原FAIL/性能INCONCLUSIVE无新WIN；按授权ENGINEERING_DEFAULT/P0关闭，不签全设备SOTA或推理engine。 |
| compute_provider_flashinfer | CLOSED | P0 / [PASS](tasks/compute/cuda-final-u6-audit.md) | [record](tasks/compute/cuda-final-u6-audit.md) | CUDA有限goal CLOSED：固定模型/量化/MTP完整原计算与状态、两OS CUDA-only/mixed三交付、cold成熟默认/真实能力门、六重复公式BORROW和具名unique保留、U5 42步/源码SDK增量及终审清理完成。AA原FAIL/性能INCONCLUSIVE无新WIN；按授权ENGINEERING_DEFAULT/P0关闭，不签全设备SOTA或推理engine。 |
| compute_provider_ggml_metal | CLOSED | P1 / [PASS](../../evidence/performance/apple-parity-overlap-p1/README.md) / 5% | [record](tasks/compute/apple-cpu-cuda-parity-closure.md) | Apple parity CLOSED: 559/34/45 carrier domains, current official FP4 main/DSpark and MTP, resources and three deliveries verified; four mature overlap WINs at n15, legal unknown-condition defaults, no redundant self kernel; Vulkan status is tracked separately. |
| compute_provider_iree_vulkan | CLOSED | P0 / [PASS](../../evidence/correctness/vulkan-parity-v4-delivery/final-audit.json) | [record](tasks/compute/vulkan-parity-final-audit.md) | Vulkan CLOSED: common Windows104/559/80/34/45/eight-family and current model/state scope, mature cold defaults, applicable Windows/Air/WSL three deliveries/OFF/headers/resources and final cleanup verified. No new self math or WIN; accepted P0 policy/P1 INCONCLUSIVE, WSL GPU unavailable and hardware/residency limits explicit. |
| compute_provider_torch_cuda | CLOSED | P0 / [PASS](tasks/compute/cuda-final-u6-audit.md) | [record](tasks/compute/cuda-final-u6-audit.md) | CUDA有限goal CLOSED：固定模型/量化/MTP完整原计算与状态、两OS CUDA-only/mixed三交付、cold成熟默认/真实能力门、六重复公式BORROW和具名unique保留、U5 42步/源码SDK增量及终审清理完成。AA原FAIL/性能INCONCLUSIVE无新WIN；按授权ENGINEERING_DEFAULT/P0关闭，不签全设备SOTA或推理engine。 |

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
