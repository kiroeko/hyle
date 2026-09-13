# Build/package：CUDA system SDK 依赖投影

<!-- PROJECT UNIT REVIEW: AUDITED -->

> Build/package 是横向开发单元，不进入产品模块 registry。本任务只服务已进入 `AUDITED` 的
> `compute_backend_cuda` CU11；不修改 CUDA runtime/API/算法，也不借机重写 package registry。

## Closure card

- baseline：`cfed0e5d5`；
- trigger：enabled CUDA factor 的专用公开头自然包含 `cuda_runtime_api.h`，其 exported target 也传播
  `CUDA::cudart`。现有 `hyle_register_package_dependency()` 只有 `STATIC_FIND_DEPENDENCIES`，因此 installed
  shared SDK 会在载入 `HyleTargets.cmake` 前缺少 CUDA imported target/header path；这是
  build-system-v12 的“selected source/static/shared consumer 可复现断裂”reopen trigger；
- consumer：source embedding、installed static 与 installed shared 的 `hyle::compute` / exact CUDA factor；
- scope：给 package registry 增加通用 `FIND_DEPENDENCIES` 字段；enabled CUDA owner 登记
  `CUDAToolkit|13.3` 和 feature `compute_cuda`；`HyleConfig.cmake` 继续在 targets 前调用生成的
  `find_dependency`；增加 registry 正负合同和 CUDA OFF/ON installed smoke；
- non-goals：下载/复制 CUDA SDK，启用全局 CUDA language，改变其它依赖字段/lock/source cache，或把
  private vendor type暴露到普通 Compute 头；
- finite FIX：`CUDA-PKG-01` 通用字段与格式校验；`CUDA-PKG-02` CUDA owner 登记；
  `CUDA-PKG-03` source/static/shared + OFF/ON regression；`CUDA-PKG-04` 用户/离线/许可说明；
- cost：configure/package O(number of registered dependencies)，运行时零成本；
- stop：若 OFF 仍探测 Toolkit、未选 CUDA 污染 package、shared/static target 引用未定义 CUDA target，或
  需要第二套 dependency truth，则 `BLOCKED`。

## AUDITED 结论

`FIND_DEPENDENCIES` 是现有 `STATIC_FIND_DEPENDENCIES` 的严格正交补充：前者适用于任意 linkage，后者仅在
HYLE static archive 把实现依赖传给最终 linker 时追加。二者由同一 module-owned registry 合并、去重并在
`HyleTargets.cmake` 前解析，不增加 CUDA 特判。CUDA Toolkit 是机器 SDK，不属于 FetchContent lock 或
`HYLE_DEP_SOURCE_CACHE`；HYLE 不安装其 headers/runtime/license。只授权四项有限 FIX，完成即关闭本 task。
