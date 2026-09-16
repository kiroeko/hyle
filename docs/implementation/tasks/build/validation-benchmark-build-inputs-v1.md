# Factor benchmark CMake 的影响分类

<!-- PROJECT UNIT REVIEW: AUDITED -->

- unit / owner：build-package TOOL / HYLE maintainers，不创建产品 component。
- baseline：`69f2d3a6`。R5 A6 添加两个 `bench/<factor>/CMakeLists.txt` 时，selector 把它们归为
  `factor_seam`，因此在 Windows 额外选中 Compute/Graph 三交付，尽管该平台不启用 Apple，
  普通交付的 `HYLE_BUILD_BENCH=OFF` 也完全不执行这些文件。
- 源码事实：顶层 `CMakeLists.txt` 只在 `HYLE_BUILD_BENCH` 分支进入 bench；现有 classifier
  在识别 factor 下的 benchmark 目录之前，先用 `.cmake/CMakeLists.txt` 后缀判断产品 seam。
  当前两个文件只登记所属 benchmark target，没有产品头/API/组件或 package 修改。
- 调查预算：10 分钟，具体分支顺序已经定位；固定以下一个 FIX，不扩影响选择或性能仪器。
- F1：已识别 owner 的 `bench/<factor>/` 下源码和 CMake 一致按 benchmark 输入分类。
  保留显式 frozen P1 card 责任；不自动采样。`src/<factor>/` 产品 CMake、全局
  `bench/CMakeLists.txt`、公共 cmake 和未知 ownership 的原门不变。
- 验证：factor benchmark CMake/嵌套 `.cmake` 不产生 header/delivery gate；同轮产品 seam
  仍保留这些门；全局 benchmark root 仍 broad；未声明 P1 不启动任何计时；完整 selector 工具回归。
  用本轮 R5 实际 diff 检查 corrected plan，并保留原误选与新 skip 原因。
- stop：上述有限分类与回归通过即 CLOSED；不借本卡减少已经启动的 `6f3ff7ca` 产品交付验收，
  不对任何未运行产品风险签 PASS，不改变状态方法、性能规则或现有原生平台适用性。

本卡是独立公共工具修正，先冻结再实现；不是 Compute/Graph 产品完成量。
