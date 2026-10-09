# C3 CPU+Apple 共享源码依赖登记修正

<!-- MODULE REVIEW: CANDIDATE -->

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

产品候选：`530e230bbc11e4fe12218a6a021ca15b009337c3`。仅上述 component 闭包条件已实现，中央 registry 保持严格唯一 owner。
selector READY，docs/closure/diff 检查通过；Air 组合原目录配置和数学尚待恢复，不能称 PASS。

Air 原目录 configure 与 source Direct/Graph 均已实际 PASS；
[收据](../../../../evidence/correctness/compute-unified-c3/air-cpu-apple-partial.json)保留日志/候选/完整 archive 身份。
原 shared/static SDK 循环继续，未完成项仍 UNRUN；不重跑 source 或单路线。

同一候选的 shared/static relocated SDK Direct/Graph 四项均实际 PASS，原 SDK 循环 exit0；
与 source 合计六消费者通过。本项适用门完成，owner 暂留 CANDIDATE 等待 C3 全部交付终审，
不能把本组成功代签其它组合或 H2；旧配置 FAIL 保留。
