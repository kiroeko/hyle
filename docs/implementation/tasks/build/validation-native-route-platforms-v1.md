# Native route selector 的平台适用性

<!-- PROJECT UNIT REVIEW: AUDITED -->

- unit / owner：build-package TOOL / HYLE maintainers；不创建产品 component。
- baseline：`cbedb837791da15392019299f2652d9d3d70ec9d`。
- 具体反例：R5 shared build seam 的 broad plan 在 Windows 选择两个 Apple correctness gate，
  但 Apple OFF 只有逐头测试；`--no-tests=error -LE headers` 必然拒绝。反向在 Darwin 运行时，
  现有 policy 又强制六个 CUDA factor ON；这些平台本来不支持该原生路线。
- 同时 Apple factor 缺少真正的 ON 验证上下文；不能只补 OFF 测试来让空路线显示绿色。
- 调查预算：15 分钟，现有 policy/命令生成和 factor CMake 已逐一定位；以下有限 FIX 固定。
- F1：policy 显式声明八个 native factors 支持的 OS；两个 Apple factor 绑定已有
  `HYLE_COMPUTE_APPLE=ON`。不按是否装 SDK/有设备决定适用性；合格 OS 缺少 SDK 仍然失败。
- F2：配置仍保留反向闭包与 OFF 逐头测试；当前 OS 不支持的 correctness gate 明确 SKIPPED 并
  记录支持的 OS，相应 feature 为 OFF。P1 卡不得引用此类未选 correctness；不产生 PASS receipt。
- F3：工具回归覆盖 Windows/Linux/Darwin 混合闭包、仅 Apple 修改、无关模块、缺 SDK 的 ON
  责任和 P1 拒绝；用稳定候选运行适用 selector。
- 非目标：不改产品默认值、组件图、状态机、证据复用算法、统计/性能仪器或已测算法。
- 验证：工具回归、三个 OS 的计划、Windows 适用 broad gates、Mac 的真实 Apple native gate、
  docs/closure。此前 Air 原生与三交付证据已真实执行；skip 不代替它们。
- stop：有限平台投影与回归成立后 CLOSED；若需扩大到硬件探测/跨机自动调度则另行定界。

源码审计确认 `selected_factors` 同时承载 configure roots；因此不删除不支持的 root 以免连 OFF
公开头检查也消失。单独约束 correctness/P1，使用现有 cache 冲突检查，不改执行引擎。
