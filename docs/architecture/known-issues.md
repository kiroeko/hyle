# Known Issues（已知问题 / 未解决缺陷）

> 诚实账本：已发现但**尚未解决**的真实缺陷。不藏、不淡化。每条带：现象、已验证的事实、已排除项、复现条件、下一步。解决后移除并在对应里程碑/提交记录。

## KI-4 [环境/工具限制]：多 CUDA context 并发时 libcuda 内部 memcpy 报 TSan 假 race（compute-05 发现）

> **现象**：`build-tsan` 跑 compute-05 `ggml_bridge_test` 的 `ConcurrentIndependentBackends`（6 fiber 各持一个 `GgmlBackend`=6 个独立 CUDA context，并发跑 matmul）报 **1 条** `data race ... in memcpy`。**测试仍 `[ OK ]` + PASSED**。
> **已验证根因 = libcuda 驱动内部跨 context 共享结构对 TSan 不可见（非 HYLE 逻辑 race）**：TSan 报告的两处冲突写 **都在 `libcuda.so.1.1` 内部的 memcpy**（栈帧 #2 `libcuda.so.1.1+0x2b3fc4`），由 ggml `graph_compute`（`ggml_bridge.cc:182` 的 `tensor_set`/compute H2D）在两个 `hyle-blocking` pool 线程上分别触发；**冲突内存是 `libcuda.so` 自己 malloc 的堆块**（`Location is heap block ... allocated by libcuda.so.1.1+0x131eb52`，32764 字节的驱动内部 scratch），不是 HYLE 内存。两个 pool 线程各持自己 context 的写锁（M1/M2）+ 共享读锁（M0，`cudaSetDevice` 建），但驱动对某全局区的同步 TSan 穿不透。
> **决定性实验（已排除 HYLE + 排除 mimalloc）**：**同一套 spawn_blocking + tensor_set memcpy 机制，强制走 CPU 后端（`CpuThreadModelSplit`，无 libcuda）→ TSan 零 race、PASSED**。race 仅在 libcuda 进入 memcpy 栈时出现 → 坐实是 CUDA 驱动内部、非 HYLE 数据竞争。它也不同于 mimalloc 2.x 曾出现、现已随 3.5 与完整 TSan instrumentation 解决的回收地址假 race：KI-4 的冲突内存属于 libcuda 自己的堆（`libcuda.so` 地址、libcuda memcpy 帧），且只在跨 context 并发出现。
> **影响**：**仅 TSan 构建 + 并发多 CUDA context**——compute-05 三平台 Release 全绿（Win/WSL RTX4070 4/4、mac 4/4），生产零影响。ggml 线程安全契约（每 fiber 一 backend、无共享 HYLE 状态）由 Release 并发稳定 + CPU-后端 TSan 零 race 旁证。
> **下一步（未做）**：TSan 下对 CUDA 并发路径加 libcuda 抑制（`called_from_lib:libcuda.so`）或单 context 串行验并发。低优先（非缺陷、非阻塞、第三方驱动限制）。

## KI-1 [已降级/重新定性]：fiber × mimalloc 跨线程 free 崩溃 —— 穷尽调查后判定为 9p/WSL 环境假象，非 HYLE 缺陷

> **状态变更(本轮深查)**:此条原为"P0 未解 HYLE 缺陷"。经一轮**穷尽、独立于崩溃时序**的调查,该定性**与证据矛盾**,现重新定性为**极可能是 9p/WSL demand-paging 环境假象**。完整实验账本(必要条件、已排除项、机制假说、复现尝试)见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md)。**不再声称是 HYLE bug,也不 100% 拍板;以证据说话。**
>
> **defect 级工具排除 HYLE 逻辑成因**:TSan(源码级 race)10万次迁移净;**MI_DEBUG=3** 全量堆不变量 + 65万 async churn + 48-worker 超额订阅 + 24G 内存压力**全净零断言**;代码无任何抑制。
> **忠实+激进复现全失败**:原始崩溃器本体 scheduler_bench + 原生 ninja 真饱和(cc1:9-17)**10/10 净**;`drop_caches` 拷打 9p/native **8/8+8/8**;真实 24G 内存压力(avail→4G 强制代码页驱逐)9p/native **6/6+6/6**。
> **关键线索**:① 早期"重负载复现"在 9p 上 `cc1plus=0`(IO-bound,负载没起来)——用户的 9p 洞察命中;② 原始崩溃含 **SIGILL** = 代码页问题(非堆损坏的数据 SIGSEGV),指向 9p 后端 .so 代码页 re-fault 不稳定。
> **mimalloc 保持出厂默认 ON**:核心 fiber 场景经 MI_DEBUG=3 全量校验证明无损坏,不需为此关 mimalloc。
> **⚠️ 勿与 sched-05 的 TSan×fcontext SEGV 混淆(不同 bug)**:sched-05 修过一个 `__tsan_switch_to_fiber` 缺失导致的 **TSan 构建内** SEGV(libtsan 影子栈跟踪,release 零影响、已修)。KI-1 是 **release -O3 + mimalloc TLS-on-fcontext栈** 的运行时崩溃,**与 TSan 无关**——两者机制不同,别因都是"WSL fiber 崩"而当成一回事。本会话 conditional-notify 大增 fiber 迁移 + 原生 ext4 重压(NoLostWakeup 100×/marl bench 多轮)**未复现 KI-1**,与"9p 假象"一致(非证实)。
> **诚实边界**:9p 上也未正面复现(同样净),故不 100% 断言"就是 9p";但旧"HYLE bug"框架已被证伪。scheduler_bench **已编入默认 build**(提交 2702df2 解封)——若该环境假象再现,直接正面查,不藏。
