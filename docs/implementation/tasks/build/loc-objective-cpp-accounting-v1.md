# 统一 LOC：Objective-C++ 实现计入

<!-- PROJECT UNIT REVIEW: AUDITED -->

## 独立工具卡 / SCOPED → AUDITED

- owner：`scripts/loc-report.py` 的 native language/suffix 范围；baseline
  `d900a8cea8232f12b836a125ea7d3c9171704d72`。
- trigger：R5 正式源码出现 `src/compute_route_apple/apple_environment.mm`，统一报告的 module
  路径选中了它，但 `NATIVE_LANGUAGES` 没有 Objective-C++，静默漏计。旧报告的 self-check 只证明
  已选语言之和一致，不能证明“全部实现”。这不是产品缺陷或性能仪器问题。
- scope：只增加 Objective-C++ language / `.mm` native suffix，补实际 cloc 与 registry owner
  选择的反例；不修改算法、历史 LOC 数据、selector、状态协议或性能统计。
- 固定 FIX L1：两个白名单覆盖 Objective-C++。L2：回归证明该语言真实计入、测试文件仍排除。
- 预算：15 分钟，一次工具回归和同一不可变产品的完整统一统计。无产品构建或 P1。
- 验证：`loc_report_test.py`、cloc 对实际 `.mm` 的识别、`--commit d900a8ce --self-check`、
  docs/closure/diff；未运行不签 PASS。
- P0：仅离线统计；没有产品热路径、依赖、ABI、用户交付或 FFI 变化。
- stop：上述有限项成立后 CLOSED；只有新的正式 native 语言被静默漏计时重开。

审计已定位两个静态范围常量；现有 cloc 计数、不可变 Git snapshot 和 central owner 算法均可复用。
不借此更换统计器，也不通过改变产品文件扩展名迁就旧统计范围。

L2 同时把旧测试中固定“Compute/Graph 必须 QUEUED”的历史队列断言替换为显式 mixed-state fixture；
实际 registry 已变化，该断言不代表产品合同。只改测试输入，不更改 owner 或状态选择算法。
