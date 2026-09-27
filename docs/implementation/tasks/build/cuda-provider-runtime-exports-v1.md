# CUDA Provider 私有 SDK runtime 的导出

<!-- MODULE REVIEW: AUDITED -->

- baseline：`d32fa727`；Windows canonical。
- owner：`compute_provider_cublas`；`compute_provider_ggml_cuda`承接当前D5 AUDITED卡。
- trigger：D5-A5真实外部shared消费者仍缺少cuBLAS/cuBLASLt；backend的CUDART已被公开
  staging正确邻置。实际installed `HyleTargets-release.cmake`中的两个Provider没有相应
  `IMPORTED_LINK_DEPENDENT_LIBRARIES`，因为原SDK目标是UNKNOWN，补其interface不足以
  让私有依赖成为共享库的导出runtime依赖。收据：`C:/hv/d5cd/receipts/shared-sdk-runtime-final`。

## 固定范围与审计

| FIX | 允许改动 | 验收 |
|---|---|---|
| E1 | cuBLAS Provider在Windows私有链接backend已声明的真实cuBLASLt SHARED runtime叶目标 | installed共享目标保留私有DLL依赖；原SDK include/link合同不变 |
| E2 | ggml CUDA Provider私有链接同一机制的cuBLAS SHARED叶目标，其依赖包含cuBLASLt | source/static/shared公开staging得到两个DLL，不手工复制、不更改ggml数学 |
| E3 | 完成原D5-A5固定8项×三交付与selector适用门 | 移走producer、移除SDK PATH仍执行；原数值门不变 |

没有新C++、算子、公共API、运行时开销或依赖版本变化。P0仅检查构建/导出结构；原D5-A6
性能仍待完成。复用backend的私有config与既有package registry，不改变公共治理/验证驱动。
预算：上述两处有限CMake接线和原定批次；如仍缺失runtime，依据具体导出图停止裁决，不
启动随机全库构建或在测试里补DLL。仅在适用门完成后CANDIDATE/CLOSED。
