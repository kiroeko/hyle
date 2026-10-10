# CUDA Direct 权重片段

本例只链接 `hyle::compute`，选择 `HYLE_COMPUTE_ROUTES=cuda`；不需要 CPU、Graph 或 Sched。
使用 `hyle/compute/cuda_weights.h` 的双槽迁移，再记录原生 DMA 依赖并调用成熟 MatMul。
TensorDesc/ID 固定于冷创建，选择与源地址可以逐批改变，机制不转换量化编码。
保留 WeightBatch 到消费计算完成；Slot 拒绝在仍上传或被绑定/批次持有时复用。

```powershell
cmake -S samples/package/compute/cuda_weights -B C:/hyle-weights -G Ninja `
  -DHYLE_SAMPLE_SOURCE=C:/public/project-hyle/hyle
cmake --build C:/hyle-weights --parallel 2
ctest --test-dir C:/hyle-weights --output-on-failure -j 1
```

安装消费时省略 HYLE_SAMPLE_SOURCE 并传 CMAKE_PREFIX_PATH，source/static/shared 使用相同 C++。
当前实际验收边界见[H1 卡](../../../../docs/implementation/tasks/compute/compute-h1-cpu-cuda-composition.md)。
