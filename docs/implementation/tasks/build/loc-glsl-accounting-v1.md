# 统一 LOC：GLSL 实现计入

<!-- PROJECT UNIT REVIEW: AUDITED -->

## 独立工具卡 / SCOPED → AUDITED

- owner：`/root`，仅 `scripts/loc-report.py` 的 native language/suffix 范围；baseline
  `8397126919a2b01f82669d1b3f7c0920fd0ea020`。
- trigger：C1-T 新增正式实现 `src/compute_route_vulkan/shaders/tensor.comp`。实际 cloc 2.08
  已识别 GLSL 为 186 行 / 1 文件，现有 native language 过滤却报告 0 / 0；registry-state
  选择也因缺少 `.comp` 后缀漏计。这满足已有 LOC 工具卡的“新语言静默漏计”具名触发条件。
- scope：只补已出现的 GLSL language / `.comp` suffix 与实际计数、归属和测试排除反例。
  不改变历史统计、产品、selector、状态协议、构建或性能仪器。
- 固定 FIX L1：两个白名单纳入上述语言与后缀。L2：现有工具测试覆盖真实 shader 的 cloc
  计数和 central owner/state 唯一选择，测试 shader 不进入实现统计。
- 预算：15 分钟，一次工具回归与同一不可变产品的完整统一统计；仅修正具体失败。
- 验证：`tests/tooling/loc_report_test.py`、`--commit 83971269 --self-check` 的 Compute + Graph
  报告、selector 的工具门、docs/closure/diff。无产品构建或 P1；未运行不签 PASS。
- P0：只影响离线源代码统计，无热路径、ABI、产品依赖或交付变化。
- stop：上述有限项成立即 CLOSED。后续只有新的正式实现语言被漏计才重开。

审计已复现两个静态范围缺口；复用现有 cloc、不可变快照和 central owner 机制。
