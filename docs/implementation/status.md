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
| compute | CLOSED | P0 / [PASS](../../evidence/correctness/model-profile-routing-refactor/README.md) | [record](tasks/compute/llm-moe-model-profile-routing-refactor.md) | MR CLOSED：compute 的真实模型制品画像、任意端口规则、确定性冷绑定和四路线单轨迁移已通过正确性、三交付、红队及 P0；M3 覆盖保持独立。 |
| graph | BLOCKED | — | [record](tasks/compute/llm-moe-d5-vulkan-performance-blocker.md) | D5 BLOCKED：总ledger162通过/12拒绝/2数值失败，四路线A5完成，CPU/Apple/CUDA代表片段P1通过。Vulkan恢复A/A失败、正式样本0，保留原中断数据；A6与D6未关闭，产品FIX零。 |
| container_concurrent | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_concurrent v2 CLOSED：并发表能力、三平台/三交付、文档、红队与约 134ms 同合同 membership 双顺序 PARITY、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_queue | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container.md) / 5% | [record](tasks/container/container-v6-identity-p1-resume.md) | container_queue v2 CLOSED：queue 能力、三平台/三交付、文档、红队与同合同 unbounded 4P4C handoff 双顺序 WIN、5% P1 非劣门全部通过；候选固定在 111c28ea。 |
| container_text | CLOSED | P1 / [PASS](evidence/performance/benchmarks-container-text-base64.md) / 5% | [record](tasks/container/container-text-v3-base64.md) | container_text v3 CLOSED：Base64/Base64URL caller-buffer API、显式策略与 simdutf 私有 bulk core 通过三平台正确性、ASan/UBSan、Container 三交付、易用性和红队；W1-WIN 两个真实形状的 5% P1 非劣门全过。 |
| io_watch | CLOSED | P0 / [PASS](tasks/io/io-watch-v2-recursive-tree.md) | [record](tasks/io/io-watch-v2-recursive-tree.md) | io_watch v2 CLOSED：显式递归树、root-relative bounded stream、rescan/cap 与三平台 backend 通过三平台、sanitizer、三交付、红队；direct 默认成本隔离，Linux 建树 O(D)、普通 descriptor lookup O(1)。 |
| process | CLOSED | P0 / [PASS](tasks/process/process-v1-managed-child.md) | [record](tasks/process/process-v1-managed-child.md) | process v1 CLOSED：受管 child 的 argv/env/cwd/stdio、HYLE-native wait/pipe、三平台非轮询 reap 与 tree termination 已通过四平台环境、三 sanitizer、三交付和终局红队；P0 成本与平台限制已如实记录。 |
| io_websocket | CLOSED | P1 / [PASS](evidence/performance/benchmarks-io.md) / 5% | [record](tasks/io/io-websocket-v2-gcc-coroutine-allocator-use.md) | io_websocket v2 CLOSED：两个私有 coroutine 显式保留 frame allocator 使用；GCC 14.2 与 MSVC 19.50 shared/WERROR 定向编译和测试通过，产品 API/算法/成本及 v1 P1 结论不变。 |
| compute_provider_onednn | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-mature-admission-batch-v1/README.md) | [record](tasks/compute/cpu-onednn-small-matmul-admission.md) | CPU小矩阵准入CLOSED：OM1/OM2原BRGEMM准入、正确性/重放、Windows/Linux三交付与P0已验。删除无资格M<16过滤，不签Native WIN；CPU逐域裁决仍SCOPED。 |
| compute_provider_ggml | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-aten-cast-stable-route/README.md) | [record](tasks/compute/cpu-aten-cast-stable-route.md) | TC1–4 CLOSED：五dtype原Cast与NaN storage边界、stable Sigmoid及generic必要图边界，两平台124/124+58/58、CPU151/7skip、Graph168/7skip、普通Direct/Graph三交付与P0通过。仅本批关闭，Native自研裁决和其它CPU接缝未完成。 |
| graph_provider_onednn | AUDITED | — | [record](tasks/compute/cpu-onednn-hybrid-ffn-retirement.md) | NX1–3 AUDITED：错误generic oneDNN/SIMD FFN route/artifact/执行已退役；保留有效norm/MatMul与SIMD本体。首批Windows3/3与selected Graph通过，Linux/三交付未验；自研性能冻结。 |
| compute_ops | AUDITED | — | [record](tasks/compute/llm-moe-d5-original-fp4-expert-binding.md) | D5 AUDITED：F44原版专家与资源、F43官方完整消费已验，CUDA正确性44/44；总ledger162通过/12拒绝/2数值失败。四路线A5和CPU/Apple/CUDA代表片段P1完成；Vulkan A6资格阻塞，残余覆盖与D6未关闭。 |
| compute_backend_cpu | CLOSED | P0 / [PASS](tasks/build/vulkan-r6-product-delivery.md) | [record](tasks/build/vulkan-r6-product-delivery.md) | R6 Vulkan 交付 CLOSED：D1–D3、Windows/Air 最终三交付、relocation/隐藏 producer/offline/DLL 邻置及全部 111 个 selected gates 完成；十三旧 owner 的 C++/API 不变，原运行时/P1、限定域与 CPU 例外继承。 |
| compute_backend_cuda | CLOSED | P0 / [PASS](../../evidence/correctness/llm-moe-d5-real-fragments/cuda-delivery/README.md) | [record](tasks/build/cuda-sdk-runtime-staging-v1.md) | D5-A5 CLOSED：系统CUDA DLL身份与Provider私有runtime导出已补齐；三交付24项、459项模块测试（含一次旧名单断言纠正）、3组独立头通过。仅CMake修复，P0无热路径变化；不代签D5-A6。 |
| compute_provider_native | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-aten-cast-stable-route/README.md) | [record](tasks/compute/cpu-aten-cast-stable-route.md) | TC1–4 CLOSED：五dtype原Cast与NaN storage边界、stable Sigmoid及generic必要图边界，两平台124/124+58/58、CPU151/7skip、Graph168/7skip、普通Direct/Graph三交付与P0通过。仅本批关闭，Native自研裁决和其它CPU接缝未完成。 |
| compute_sched | QUEUED | — | — | compute_sched identity 已预留：未来审计 Platform process/topology 与 Sched persistent workers 到 Compute ParallelExecutorRef 的自然 CPU execution-pool bridge；尚无依赖、产品入口或实现。 |
| compute_route_cpu | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-aten-storage-index/README.md) | [record](tasks/compute/cpu-aten-storage-index-admission.md) | SI1–4 CLOSED：原ATen六组plain存储/索引宽scalar、I32/I64、odd stride及duplicate Assign接线；两平台114/114、CPU12通过/1旧fixture跳过、Graph15/15、普通Direct/Graph三交付与P0通过。Native数学冻结，CPU全裁决未完成。 |
| graph_route_cpu | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-aten-storage-index/README.md) | [record](tasks/compute/cpu-aten-storage-index-admission.md) | SI1–4 CLOSED：原ATen六组plain存储/索引宽scalar、I32/I64、odd stride及duplicate Assign接线；两平台114/114、CPU12通过/1旧fixture跳过、Graph15/15、普通Direct/Graph三交付与P0通过。Native数学冻结，CPU全裁决未完成。 |
| compute_provider_cublas | CLOSED | P0 / [PASS](../../evidence/correctness/llm-moe-d5-real-fragments/cuda-delivery/README.md) | [record](tasks/build/cuda-provider-runtime-exports-v1.md) | D5-A5 CLOSED：系统CUDA DLL身份与Provider私有runtime导出已补齐；三交付24项、459项模块测试（含一次旧名单断言纠正）、3组独立头通过。仅CMake修复，P0无热路径变化；不代签D5-A6。 |
| compute_provider_marlin | CLOSED | P0 / [PASS](../../evidence/correctness/llm-moe-d4-self-operator-retirement/README.md) | [record](tasks/compute/llm-moe-d4-self-operator-retirement.md) | M3-D4 CLOSED 91c2fb4d：无合格优势且由成熟 Provider 覆盖的 Marlin、CPU/CUDA/Apple/Vulkan 自有算子域已批量退场；四路线正确性、组件合同与 Compute/Graph 三交付通过，D5 继续真实模型片段验收。 |
| compute_provider_cuda | AUDITED | — | [record](tasks/compute/llm-moe-d5-original-fp4-expert-binding.md) | D5 AUDITED：F44原版专家与资源、F43官方完整消费已验，CUDA正确性44/44且A5完成；CUDA完整BF16 FFN/官方FP4-FP8 MoE P1已在n15合格通过，旧A/A失败保留。D6未关闭。 |
| compute_route_cuda | AUDITED | — | [record](tasks/compute/llm-moe-d5-fp8-weight-preparation.md) | D5 AUDITED：F41冷准备和F43/F44官方完整消费已验，CUDA正确性44/44、三交付A5完成；CUDA两完整形状A6在n15 PARITY/非劣PASS，严格重放通过；D6未关闭。 |
| graph_route_cuda | CLOSED | P0 / [PASS](../../evidence/correctness/llm-moe-d4-self-operator-retirement/README.md) | [record](tasks/compute/llm-moe-d4-self-operator-retirement.md) | M3-D4 CLOSED 91c2fb4d：无合格优势且由成熟 Provider 覆盖的 Marlin、CPU/CUDA/Apple/Vulkan 自有算子域已批量退场；四路线正确性、组件合同与 Compute/Graph 三交付通过，D5 继续真实模型片段验收。 |
| compute_route_apple | AUDITED | — | [record](tasks/compute/llm-moe-d5-portable-float-native-qualification.md) | D5-F45 AUDITED：Apple40格运行通过、DeepSeek官方FP4四格准备拒绝。三交付与BF16 FFN/GGUF MoE代表片段P1已通过；未外推官方FP4或其它形状性能，残余能力与D6未关闭，无新数学。 |
| graph_route_apple | AUDITED | — | [record](tasks/compute/llm-moe-d5-portable-float-native-qualification.md) | D5-F45 AUDITED：Apple40格运行通过、DeepSeek官方FP4四格准备拒绝。三交付与BF16 FFN/GGUF MoE代表片段P1已通过；未外推官方FP4或其它形状性能，残余能力与D6未关闭，无新数学。 |
| compute_route_vulkan | BLOCKED | — | [record](tasks/compute/llm-moe-d5-vulkan-performance-blocker.md) | D5 BLOCKED：Vulkan两入口合计34通过/8冷拒绝/2数值失败，A5三交付完成。唯一A6恢复的自比较不合格，guard阻止正式采样，聚合INCONCLUSIVE；非产品性能FAIL。原Q4通过、Q8数值失败和官方低精度拒绝均保留。 |
| graph_route_vulkan | BLOCKED | — | [record](tasks/compute/llm-moe-d5-vulkan-performance-blocker.md) | D5 BLOCKED：Vulkan两入口合计34通过/8冷拒绝/2数值失败，A5三交付完成。唯一A6恢复的自比较不合格，guard阻止正式采样，聚合INCONCLUSIVE；非产品性能FAIL。原Q4通过、Q8数值失败和官方低精度拒绝均保留。 |
| compute_provider_ggml_vulkan | AUDITED | — | [record](tasks/compute/llm-moe-d5-real-fragment-acceptance.md) | D5-F30 AUDITED：Kimi MLA、GLM64-head及资源边界已验，Vulkan A5完成；当前34通过/8拒绝/2数值失败。A6原中断数据保留，唯一恢复A/A失败、正式样本0；路线A6阻塞，D6未关闭。 |
| compute_ggml_runtime | CLOSED | P0 / [PASS](../../evidence/correctness/cpu-ggml-isa-admission-v1/README.md) | [record](tasks/compute/cpu-ggml-isa-admission-v1.md) | CPU ISA CLOSED：CI1–4原版优化整程序、冷ISA绑定/失败、资源生命周期和Windows/Linux三交付/搬移/P0已验；不签Native WIN。CPU自研逐域与路由仍未完成，其它路线暂停，旧F35/GPU待验未解除。 |
| compute_provider_ggml_cuda | AUDITED | — | [record](tasks/compute/llm-moe-d5-real-fragment-acceptance.md) | D5-F30/F31 AUDITED：64-head压缩MLA、原生I32 Concat及拒绝/所有权边界已验，无新kernel。CUDA正确性44/44、三交付A5和两完整片段A6均通过；D6未关闭。 |
| graph_provider_ggml_cuda | CANDIDATE | — | [record](tasks/compute/llm-moe-d3-ggml-native-binding.md) | M3-D3 CANDIDATE 08b8e2f1：十种真实ggml/GGUF storage接缝、CUDA22画像两入口、三交付A5和两完整片段A6已有收据；D6残余覆盖和终审未完成，不提前CLOSED。 |
| compute_provider_flashinfer | CLOSED | P0 / [PASS](../../evidence/correctness/model-profile-routing-refactor/README.md) | [record](tasks/compute/llm-moe-model-profile-routing-refactor.md) | MR CLOSED：compute_provider_flashinfer 的真实模型制品画像、任意端口规则、确定性冷绑定和四路线单轨迁移已通过正确性、三交付、红队及 P0；M3 覆盖保持独立。 |
| compute_provider_ggml_metal | AUDITED | — | [record](tasks/compute/llm-moe-d5-real-fragment-acceptance.md) | D5-F24/MB AUDITED：原生Clamp、recurrent gate/GDN及原生buffer分配失败回滚已验，Apple A5和两个代表完整片段P1通过；官方DeepSeek FP4拒绝域保留，D6未关闭。 |
| compute_provider_iree_vulkan | CANDIDATE | — | [record](tasks/compute/llm-moe-d3-ggml-vulkan-native-binding.md) | M3-D3 CANDIDATE 2757ecc：checked Scan、BF16 conversion和具名FP8 MatMul限定域已有收据，Vulkan A5完成；不代签官方grouped FP8/FP4或Q8数值失败域。D5-A6及D6未关闭。 |

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
