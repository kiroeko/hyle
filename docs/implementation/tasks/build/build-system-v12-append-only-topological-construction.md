# Build/package v12：只追加身份与拓扑构造解耦

<!-- PROJECT UNIT REVIEW: ACTIVE -->

> Build/package 是横向开发单元，不进入 `module-status.json`。本轮由 Obs v4 对已冻结
> `mem_reclaim` 的真实复用边触发；Obs 状态仍只由
> [`module-status.json`](../../module-status.json) 决定。

## Closure card

- unit / owner / round / kind：`build-package` / build-package maintainers /
  `build-v12-append-only-topological-construction` / `REOPEN`；
- baseline：`2bbbaed65c50c2b77a7dc7dd845af43b74a59ff4`；
- trigger：`BUILD-APPEND-ONLY-TOPOLOGY-2026-09-05`。Obs 的 identity 早于后来拆出的
  `mem_reclaim`，但 Obs v4 证明其 immutable snapshot 宽限期与该底层因子是真实同合同复用。
  旧构建系统又把不可重排的 component 登记顺序当作 target 构造顺序，使这条自然边
  除非改写历史否则无法表达；
- real consumers：新的底层 factor 拆出后复用它的旧模块，以及选择单个 product root 的
  source/static/shared C++ 父工程；
- scope：保持 `HYLE_KNOWN_COMPONENTS` 为只追加 identity ledger；在中央 manifest 上验证
  unknown/QUEUED/cycle，稳定拓扑解析 target 构造顺序；依然以单次反向拓扑传播解析
  selected closure；增加“旧 identity 依赖新底层 factor”正向回归；更新架构文档；
- non-goals：修改任何产品 API/算法/运行时 layout，改名或重排 component ID，新增 option/
  product façade/外部依赖，改写 module closure/performance 仪器，或宣称未收口模块已冻结；
- tier / cost：`P0` structural。Configure-time 仍为 `O(V+E)`；DFS 每个 factor/边至多访问
  一次，selected marks 仍只做一次反向拓扑传播。不产生任何运行时 target 元数据或热路径成本；
- finite FIX：`BUILD-V12-ORDER-01` 分离 ledger 与构造序；`BUILD-V12-CYCLE-02` 拓扑解析
  fail closed；`BUILD-V12-LIST-03` 提供 CMake list 形式的 resolved factors 给 target 构造；
  `BUILD-V12-REGRESSION-04` 锁定 Obs -> mem_reclaim 的向后 identity 边；
  `BUILD-V12-DOC-05` 清理“登记序即拓扑序”过时说法；
- required gates：current component contract；Obs selected source/shared/static relocated smoke；
  all-product feature-OFF configure/build；Windows/WSL/macOS 的 portable CMake 语义；docs/closure/diff；
- budget / stop：一个线性拓扑算法、一个正向 fixture、一轮受影响交付。若环检测不能
  fail closed，或者未选因子会被构建，则 `BLOCKED`；不增加第二套 graph truth。

## 待验证候选

`HyleComponents.cmake` 已将 identity ledger 与稳定 DFS 拓扑序分开，并将 list 形式保存为
`HYLE_RESOLVED_COMPONENT_LIST`；根 `CMakeLists.txt` 按该 list 构造 target。新 component-contract fixture
选择 Obs，要求精确闭包/target 顺序为 `core;error;mem_reclaim;time;obs`，因而无需改写
component 历史顺序。

当前状态：`ACTIVE`。只有上述适用门全部通过后才可写入 `CLOSED` 裁决。
