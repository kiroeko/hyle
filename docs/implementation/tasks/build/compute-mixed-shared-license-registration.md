# C3 CPU+Apple 共享源码依赖登记修正

<!-- MODULE REVIEW: AUDITED -->

- owner：`compute_route_apple`；baseline `6e7af97eb`。
- 需求：普通 `HYLE_COMPUTE_ROUTES=cpu;apple` 的完整源码及静态/动态 SDK。
- 反例：Air 普通 source 配置在开始编译前失败：CPU 的 `ds4_quantization`
  与 Apple 的 `apple_ds4_quantization` 同时认领同一中央 lock。原配置失败单独保留，数学未运行。
- 本轮只将该 CANDIDATE owner 退回 AUDITED；CPU owner、锁和 registry 检查不改。

## 有限审计与 FIX

CPU native 声明无条件登记 DS4 的原许可证；Apple 的两项共享来源中只有 DS4 与 CPU 重复。
DeepSeek 来源仅 CUDA/Apple 使用，两路线宿主互斥；其余 CPU/Apple/CUDA/Vulkan 锁身份不同。
中央 registry 的一个锁一个登记 owner 正确，不能放宽。

唯一 FIX：Apple 登记 DS4 前检查最终 component 闭包是否选中 `compute_provider_native`。
选中时由 CPU owner 登记该锁及完整许可证一次；未选中仍由 Apple 登记。
使用已有 `HYLE_BUILD_COMPUTE_PROVIDER_NATIVE`，不依赖子目录创建顺序，不增加用户开关。
Apple 功能标记仍由必需 MLX 登记提供；数学、Provider 默认、资产和编译目标不变。

## 验证、预算和停止

P0：一次 CMake 条件选择，不增加运行期成本。恢复 Air 原 source-parent/source-build 配置；
随后在 C3 原批次完成 CPU 与 Apple 的真实 Direct/Graph 及 source/shared/static SDK。
保留旧失败和已通过单路线的实际身份；只配置改变的组合闭包，不重跑单路线数学或 P1。
运行 change-impact selector、docs、closure、diff 检查。预算为本次具名审计、一个登记修正、
一次组合交付批次；有新缺陷则报告具体项，不换目录重置失败预算。
本卡不实现 H2 联合执行或新算子；完成本项仍回到
[C3 统一交付](compute-unified-delivery.md)，不能代签其余路线/平台。
