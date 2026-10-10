# CPU + Vulkan 粗片段组合

选择 `HYLE_COMPUTE_ROUTES="cpu;vulkan"` 并链接 `hyle::graph`，组合 factor、Sched 和原生资产自动加入。
普通入口为 `hyle/graph/cpu_vulkan.h`。CPU 片段显式用 `kCpuTarget` 放置，其余片段调用原 Vulkan 编译器，
保留成熟 region。不能把 GPU 缺能力或执行失败当成 CPU fallback。

本例包含普通 CPU 主存 RMSNorm/投影→完整 GGUF Q8_0 FFN→CPU，以及 CPU abs→迁移权重 MatMul/
RMSNorm→CPU。CPU 权重 16 KiB，而首条管线 native 边界仅 512 字节；不要求全部 CPU 权重进入 staging。
原始编码和独立 scalar oracle 检查、两 Invocation 和多次 replay 都在例中。

调用顺序为 `compile → create_invocation(scheduler, cpu_queue) → bind → prepare_replay → submit → wait`。
资源包括普通 host、同 Context 的 device/staging MemoryResource；更改绑定会使准备报告失效。
`replay_report` 给出 CPU/GPU 片段、host copy/repack、完成等待及实际独立/计算队列上传数。
`HostWaitPolicy::kReject` 拒绝仍须 host completion 的组合；native 多队列没有消除这些依赖。

`ResourceBudget` 接纳可控 payload/metadata；provider 私有成本未知时 strict 拒绝，显式 partial
`allow_unattributed=true` 可接受并报告 unknown。原 VkDeviceMemory quota 单独计实际原生分配；
二者都不承诺驱动/其他进程的总显存。结果 lease、借用输入、资源、CPU Queue 和 Scheduler 按寿命合同保留。
引擎决定模型加载、offload 比例、KV 和专家缓存策略；这里提供固定放置、迁移、预算和完成机制。

源码：`cmake -S . -B build -DHYLE_SAMPLE_SOURCE=<hyle-source>`。
安装包：`cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk>`；构建使用 `--parallel 1`。
Windows/Air source/static/shared均已验收；WSL三交付编译/安装/链接已验，实GPU运行保持未资格。
