# CPU + CUDA 组合示例

选择 `HYLE_COMPUTE_ROUTES="cpu;cuda"`及`graph;graph_progress`产品，链接`hyle::graph`与`hyle::graph_progress`。
进度服务公开提供所需Sched依赖。
本例把普通 CPU 主存中的权重留在 CPU，用有界 staging 连接真正的 CPU 与 CUDA 数学，
并演示双槽权重迁移、两个普通`graph::Invocation`、GraphOutput和共享资源预算。
示例用`CompileEnvironment`冷装配CPU/CUDA、host/device/staging域；运行时只用共同Graph接口。

```powershell
cmake -S samples/package/graph/cpu_cuda -B .scratch/samples/cpu-cuda -G Ninja `
  "-DHYLE_SAMPLE_SOURCE=$PWD"
cmake --build .scratch/samples/cpu-cuda --parallel 1
ctest --test-dir .scratch/samples/cpu-cuda --output-on-failure -j 1
```

安装消费时省略 `HYLE_SAMPLE_SOURCE`，传 `CMAKE_PREFIX_PATH`；源码、static、shared 使用相同 C++。
单后端 SDK 不包含组合。直接 CMake 命令使用已初始化的工具链 shell，Windows Ninja 说明见
[环境指南](../../../../docs/developer/setup/windows.md)。

示例保留两个需要观察的边界：

- 16 KiB CPU 权重参与 MatMul，但图暂存上限仅 512 字节；另有 CPU norm/cast→CUDA gated FFN→CPU。
  staging 容量不是整模型显存上限，权重也不必全部进入 pinned backing。
- `WeightSlots` 上传原始编码，不量化、不决定模型加载或淘汰；同槽仍被上传、批次或 Graph 绑定持有时不能复用。
  Direct 用户同样可使用 `hyle/compute/cuda_weights.h`，不必依赖 Graph。

运行顺序是compile→`create_invocation()`→`prepare_progress()`→bind→`prepare_replay(queues)`→
`submit(queues)`→等待Completion→`result()`→释放GraphOutput。`queues`显式绑定原CPU/CUDA Queue；
`graph::cuda::bind(invocation, value, batch, id)`保留pending权重selection和实际上传依赖，
不使用专用CPU+CUDA Invocation、Results或RunState。

`input_source_report()`核对权重等待路线，`resource_report()`核对共享预算的实际StorageDomain usage；
搬运步从CompileReport读取。两个并发槽拥有独立可变workspace和输出；旧GraphOutput阻止reset／重放，
reset后才释放绑定的权重selection。取消只阻止后继，已接纳工作仍须退休；原Scheduler、BlockingPool、
Queue、资源和外部buffer须保活至drain及任务尾部完成。异步推进每轮仍有原Scheduler任务／stack成本。

本例是调用与寿命消费者，不代表完整Qwen推理或吞吐对照已验收。
预算、结果租约、host wait 与关闭的完整说明只在 [Graph 指南](../../../../docs/user/graph.md)和
[Compute 指南](../../../../docs/user/compute.md)维护。
