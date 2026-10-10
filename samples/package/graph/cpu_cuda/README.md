# CPU+CUDA 组合

选择 `HYLE_COMPUTE_ROUTES="cpu;cuda"` 即取得组合、完整 CPU/CUDA Provider 和现有 Sched。
source、shared SDK、static SDK 使用相同 C++ 入口；安装消费仍链接 `hyle::graph`。
单路线 SDK 不包含组合，示例在 configure 时明确报告缺少 `cpu;cuda`。

```powershell
cmake -S samples/package/graph/cpu_cuda -B C:/hyle-demo -G Ninja `
  -DHYLE_SAMPLE_SOURCE=C:/public/project-hyle/hyle
cmake --build C:/hyle-demo --parallel 2
ctest --test-dir C:/hyle-demo --output-on-failure -j 1

# 已安装 SDK：省略 HYLE_SAMPLE_SOURCE，传 CMAKE_PREFIX_PATH。
```

`cpu_cuda::compile` 接受显式 CPU/CUDA Context 和同 CUDA Context 的 pinned/device 资源。
CPU-required 节点使用 `kCpuTarget`，普通数学使用 `kCudaTarget`；仍使用 Graph 原有 placement、
格式、模型/制品条件和预先固定的 Provider 规则。compile/prepare 绑定程序，submit 不重新选择。

`Invocation::create` 为一个可变槽准备固定状态，借用调用者 Scheduler 和原生 Queue。
创建独立槽与独立 Queue 可维持多个在途任务。bind 后 prepare_replay，submit 返回 OwnedResult；
wait/result 成功后取得 Results lease。任何旧结果或在途任务都会阻止该槽改绑和覆盖。
普通外部输入输出仍由应用保活；lease 保活 Graph workspace，不让外部可变存储自动不可变。

`WeightSlots::create` 冷分配两组固定容量 pinned/device backing 和实际 TensorDesc/ID 目录。
upload 选择 host-accessible tensor 片段，先检查 ID、格式、重复和总容量，再暂存原始编码字节并
执行真实 DMA。source TensorView 可以来自 Buffer subview，越界由 Mem/TensorView admission 拒绝。
一种矩阵可以是 FP16，另一种可以有不同 dtype/physical encoding；每片须匹配其登记描述。
该机制不量化或转换，也不负责模型加载、专家预测、缓存或淘汰。

用 `invocation.bind(value, batch, id)` 保活对应批次并建立依赖。共享 CUDA lane 记录原生 Queue wait；
delegated episode 无此原生端点时使用显式 host barrier。Direct 消费者可调用 batch.order(queue)，
并保留 batch 至其最终计算完成；若在 CPU 读数据，必须先 wait。
同槽的上传、批次或 Graph 绑定尚存时 upload 会立即拒绝复用；reset/rebind 也受结果 lease admission 保护。

取消只阻止尚未接纳的后继；已接纳原生前缀仍须完成退役。公有句柄析构不隐藏等待。
Invocation.close 只排空并关闭该槽；WeightSlots.close 只排空该池上传，不关闭调用者 Context/Queue。
Context、resource、Queue、Scheduler、allocator 和外部 tensor backing 的借用寿命见公开头。

示例包含 CPU 数值变换→CUDA 矩阵→CPU 结果消费，以及 CPU normalization/casts→CUDA gated FFN→CPU。
实际实现和适用运行验收状态见[当前 H1 卡](../../../../docs/implementation/tasks/compute/compute-h1-cpu-cuda-composition.md)。
