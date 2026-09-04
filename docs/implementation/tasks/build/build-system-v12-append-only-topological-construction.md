# Build/package v12：只追加身份与拓扑构造解耦

<!-- PROJECT UNIT REVIEW: CLOSED -->

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

## 候选

`HyleComponents.cmake` 已将 identity ledger 与稳定 DFS 拓扑序分开，并将 list 形式保存为
`HYLE_RESOLVED_COMPONENT_LIST`；根 `CMakeLists.txt` 按该 list 构造 target。新 component-contract fixture
选择 Obs，要求精确闭包/target 顺序为 `core;error;mem_reclaim;time;obs`，因而无需改写
component 历史顺序。

## Verification 与裁决

| Gate | Result | Evidence |
|---|---|---|
| current component contract | `PASS` | W1-WIN、W1-WSL/GCC 14.2、M1-MAC/AppleClang 21 的完整 fresh child-project 正负合同 |
| backward identity edge | `PASS` | 选择 Obs 精确得到 product `core,error,obs`、factor/target order `core,error,mem_reclaim,time,obs` |
| unknown / QUEUED / cycle | `PASS` | resolver fail closed；正向 graph fixture 与既有负向 fixture 全通过 |
| Obs source/shared/static | `PASS` | Windows、WSL、macOS 的 source embedding 与 relocated installed shared/static consumer |
| portable CMake semantics | `PASS` | Windows VS 18、WSL Ninja、macOS Unix Makefiles；没有依赖 generator-specific 排序副作用 |
| all-product feature-OFF | `PASS` | Windows MSVC 19.50 从 immutable Obs candidate 完整构造并链接 14 product / 21 implemented factor |

全产品构建只验证 target graph 可构造，不替未冻结的 Compute/Graph 或其他上层模块签发功能/性能结论。
稳定 DFS 在 configure time 每个 factor/edge 至多访问一次，仍为 O(V+E)，不生成运行时状态。append-only
`HYLE_KNOWN_COMPONENTS` 继续只是历史稳定 identity ledger；唯一构造 authority 是从同一 manifest 解析出的
`HYLE_RESOLVED_COMPONENT_LIST`，selected closure 与真实 target link 没有第二套顺序真相。

终态裁决：`CLOSED`。五项有限 FIX 全部完成且无产品运行时改动。只有中央 manifest 与真实 target 顺序
再次分叉、合法旧 identity→新低层 factor 边无法构造、环/QUEUED 未 fail closed，或 selected source/static/
shared consumer 出现可复现断裂时，才重开本 build/package task。
