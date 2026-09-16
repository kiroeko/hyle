# 统一 LOC：Objective-C++ 实现计入

<!-- PROJECT UNIT REVIEW: CLOSED -->

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

## CLOSED

工具候选 `9e41c15aab5a97206e82ebc70a0bee0d916e112f` 的两个范围常量修正后，工具回归、实际 cloc
识别与 docs/closure/diff 通过；修正前 `.mm` 的 AUDITED owner 选择稳定失败。selector 只选择
docs、closure 和该工具测试，其余 112 项跳过，没有构建产品或运行 P1。

同一不可变产品 `d900a8ce` 的完整统一统计为 Compute + Graph **33,815 行 / 116 文件**，
不含测试/bench/第三方，self-check PASS，见
[JSON](../../../../evidence/correctness/apple-r5-foundation-checkpoint/loc-d900a8ce.json)。原 33,792
漏掉 23 行 Objective-C++，不作为“全部实现”的报告。统计含新旧仍在树中的实现，不代表重构进度。
七问在本工具范围内均通过：有限范围、既有计数/owner 复用、真实反例、工具交付/文档、边界审阅和
离线 P0；本卡停止，不扩张为公共统计重构。
