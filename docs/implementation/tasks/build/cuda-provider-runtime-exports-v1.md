# CUDA Provider 私有 SDK runtime 的导出

<!-- MODULE REVIEW: CANDIDATE -->

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

## 候选验收

`9273bc71017a1febb69db240b6ea8970b88dd77c`完成E1/E2；E3的三交付24项、受影响459个
模块测试（含一次旧名单断言纠正）、3组独立头和公共门全部完成，见
[原始收据与失败保留](../../../../evidence/correctness/llm-moe-d5-real-fragments/cuda-delivery/README.md)。
只有cuBLAS owner按本卡收口；ggml CUDA仍承接D5总卡的其它验收，不自动关闭整个Provider。

七门均满足当前有限范围：scope/architecture只补私有runtime导出；correctness/delivery见收据；
usability_docs保持公开staging用法；red_team验证来源hash、PE依赖与搬迁；性能P0无热路径改变。
复用SDK目标和既有backend helper，无新数学、FFI或依赖版本/许可变化。A6仍待验收。
统一实现60,373行/241文件；仅具体SDK/toolchain导出或加载回归可以重开本轮。
