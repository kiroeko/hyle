# Vulkan 有界权重迁移

普通入口 `hyle/compute/vulkan_memory.h`、`hyle/compute/vulkan_weights.h`，链接 `hyle::compute`。
源码选择 `HYLE_COMPUTE_ROUTES=vulkan`；安装包消费者使用已含 Vulkan 的 SDK。

本例用普通 CPU 主存填充权重，固定两槽只暂存选中的 payload，然后以原生 Tensor 绑定成熟 MatMul。
调用顺序：`WeightSlots::create → upload(slot, fragments) → wait → binding(id)`；固定 catalog 描述完整
物理编码，不能只传“4 bit/8 bit”。不同 ID 选择或同描述 payload 更新不重新选择 Provider。
批次、Tensor 副本和 native owner 会保留槽；最后引用释放且原生退休后才可复用。

新 MemoryResource 缓冲允许原 compute/transfer 队列访问；有独立队列时上传以 semaphore 交接到
compute，ready 仍覆盖 compute ordered prefix。旧 Context/外部 buffer 与下载走原 compute 队列。
查询 queue capability 不等于硬件必然重叠。host staging 准备是同步的，GPU copy 接纳后异步执行。

源码：`cmake -S . -B build -DHYLE_SAMPLE_SOURCE=<hyle-source>`。
安装包：`cmake -S . -B build -DCMAKE_PREFIX_PATH=<sdk>`。构建 `cmake --build build --parallel 1`。
Context、资源、allocator/quota 必须活过在途工作及最后 native owner；`close` 排空该池，不关闭其它程序。
