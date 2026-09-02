# LOC 工具：container_intrusive v4 边界同步

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

## Finding

`container_intrusive` v4 的不可变 candidate
`58431ddbef6f54f9d4275b0fea2d317cba00b9f3` 已把实现拆为双链、forward-list、hlist 与共用
member-access 四个头；`scripts/loc-report.py` 的 canonical boundary 仍只选择旧双链头，因而把 terminal
module implementation 误报为 749 行。前三项 repository/self-written 指标不受影响。

这是 terminal LOC 工具边界缺口，不是产品 finding；模块保持 `VERIFIED`，不得把错误数字写入 receipt，
也不得借此修改候选 API、实现、性能合同或状态机。

## 有限修复与验收

- 只把 CMake 已归属 `container_intrusive` factor 的四个实现头登记进 canonical LOC boundary；Sched
  consumer、tests、bench、docs 与 CMake 不计入模块实现；
- `loc_report_test.py` 必须逐项验证四个头都被选择，`--self-check` 同步锁定相同边界；
- 用原 candidate commit 重放四项统计，确保 immutable-tree 语义不变；
- 运行 tooling test、closure/docs checks 与 `git diff --check` 后结束本工具任务。

本任务不改变 `module-status.json`，也不改变 candidate identity；修复提交只让 terminal receipt 能准确
描述既有不可变产品树。
