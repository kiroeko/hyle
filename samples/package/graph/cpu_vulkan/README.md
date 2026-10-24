# CPU + Vulkan 粗片段组合

选择 `HYLE_COMPUTE_ROUTES="cpu;vulkan"`、产品 `graph;graph_progress`，链接 `hyle::graph` 与 `hyle::graph_progress`。
进度服务公开提供所需Sched依赖。
样例把真实 CPU 和 Vulkan `ExecutionSource`、host/device/staging 资源装入普通 `graph::compile`。
CPU 与 Vulkan 节点均显式固定位置，完整 FFN 保留为原生 region；GPU 缺能力或执行失败不会转 CPU。

本例包含普通 CPU 主存 RMSNorm/投影→完整 GGUF Q8_0 FFN→CPU，以及 CPU abs→迁移权重 MatMul/
RMSNorm→CPU。CPU 权重 16 KiB；首条管线的两个 native 张量边界逻辑大小合计 320 字节，
不要求全部 CPU 权重进入 staging。样例另将 Graph device workspace 限为 512 字节；
这个限额不覆盖原生程序自己分配的 generated backing，不能当作完整 native 内存硬上限；
原生计划可知的 backing blocks 另记入同一 `ResourceBudget`，私有未知成本仍单独报告。
原始编码和独立 scalar oracle 检查、两 Invocation 和多次 replay 都在例中。样例额外发布已有的
FFN `down` 边界，读取其真实 VkBuffer owner；这会延长该值的寿命，不新增数学或切开 FFN。

调用顺序为 `compile → create_invocation → prepare_progress → bind → prepare_replay(queues) → submit(queues) → Completion.wait → result`。
`result` 返回原槽的 `GraphOutput`；其 `tensor()` 读取 CPU 结果，`storage_lease()` 可借原
Vulkan MemoryResource 导回真实 native Tensor。持有旧 GraphOutput 时禁止覆盖槽。
只为报告中需要 caller Queue 的 CPU lanes 提供 QueueBinding；Vulkan 使用自身原始提交和完成对象。
`graph::vulkan::bind(invocation, value, batch, id)` 保留迁移权重及其上传依赖，未准备好的字节不能被提前消费。

可选 `prepare_progress` 使用调用方的 Scheduler 和 BlockingPool 推进同一个 Invocation；池用于真实
阻塞片段。二者须活过已接纳工作、回调尾部及 `scheduler.drain()`。先结束 Graph，再 drain，最后销毁池；
不能在池 worker 中等待依赖同池推进的结果。普通 host、原生 device/staging 资源和借用输入也须保持原寿命。

CompileReport 描述实际节点覆盖、保留的 4 节点 native FFN 和物理 Copy；ReplayReport 给出真实 host-wait
依赖。`HostWaitPolicy::kReject` 拒绝这些必要等待。当前共同报告不提供每次上传使用独立还是计算 Queue 的
计数，此项为未提供，不能用设备具备独立队列的能力代替实际路由观察，也不表示 copy/compute 已重叠。

`ResourceBudget` 接纳可控 payload/metadata；provider 私有成本未知时 strict 拒绝，显式 partial
`allow_unattributed=true` 可接受并报告 unknown。原 VkDeviceMemory quota 单独计实际原生分配；
二者都不承诺驱动/其他进程的总显存。`Invocation::resource_report()` 按实际资源域报告共享预算，
provider-private 未知范围仍明确标记；不会把观察到的 VkBuffer 容量当作完整显存总量。
引擎决定模型加载、offload 比例、KV 和专家缓存策略；这里提供固定放置、迁移、预算和完成机制。

源码：`cmake -S . -B build -DHYLE_SAMPLE_SOURCE=<hyle-source>`。
安装包：`cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk>`；构建使用 `--parallel 1`。
完整的组合、寿命和结果租约说明见 [Graph 指南](../../../../docs/user/graph.md)。
