# CPU + CUDA 组合示例

选择 `HYLE_COMPUTE_ROUTES="cpu;cuda"`，链接 `hyle::graph`。
本例把普通 CPU 主存中的权重留在 CPU，用有界 staging 连接真正的 CPU 与 CUDA 数学，
并演示双槽权重迁移、两个 Invocation、结果租约和共享资源预算。

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

运行顺序是 compile→create invocation→bind→prepare_replay→submit→等待→释放结果。
并发使用独立槽/Queue。取消只阻止后继，已接纳工作仍须退休；旧结果与外部资源各自保有真实 owner。
预算、结果租约、host wait 与关闭的完整说明只在 [Graph 指南](../../../../docs/user/graph.md)和
[Compute 指南](../../../../docs/user/compute.md)维护。
