# graph-01 (step-1)：任务依赖 DAG 增量重算核心

| 字段 | 值 |
|---|---|
| **阶段** | P6（北极星② 驱动，非纯收口）|
| **模块** | graph/（**支柱破土**——此前 0 行）|
| **前置** | error ✅ / mem ✅ / container(SmallVector) ✅ |
| **状态** | ✅ **step-1 完成（2026-07-10，WSL + MSVC 双平台 13/13 含 2 death test + 红队闭环）**。`TaskGraph`：任务依赖 DAG，**脏标记增量重算**——改一个输入只重算下游脏节点，干净节点复用缓存。串行 evaluate（纯 CPU，无 sched 依赖）。**step-2（并行执行绑 sched work-stealing）未做**。|

> **一句话**：graph 支柱破土。北极星② CAD/BIM/CAE 的核心热路径是"拖一个滑块只重算受影响的几何 / 改一个边界条件只重算相关网格块"——即**依赖 DAG 的增量重算**。clay taskgraph 是 one-shot 全图重跑、**零增量资产**（grep 实测），所以这一块从零造。step-1 先证增量语义在串行下正确，并行留 step-2。

## 为什么从零造（不是移植 clay）

设计依据见 [`graph-design.md`](../architecture/graph-design.md)。实测结论：clay `plugins/taskgraph`（3,378 code 行）是能跑的真并行 DAG 执行器，但 `dirty|incremental|recompute|invalidate` 全插件 **0 命中**——它每次从 source 全图重跑（`graph_executor.cc:41`）。增量重算（北极星② 的 hot path）clay 从来没有。HYLE 继承 DAG 建模 + 拓扑扫描的**思路**，增量引擎（脏传播 + 结果缓存 + 只跑脏节点）**新写**。

## 三个承重决策（step-1）

| # | 决策 | 理由 |
|---|---|---|
| ① | **节点 = fn-ptr + ctx 闭包**，图只管拓扑 + 脏/缓存状态 | 图**不**拥有/类型化数据流值系统（clay 的 data_type/socket/AsyncValue ~750 行正是 graph-design 要剥的应用层耦合）。节点的 `ComputeFn(void* ctx)` 透过用户自己的内存读输入写输出；"缓存"= "此节点自上次弄脏后已运行、输出在用户内存里有效",复用 = 跳过调用。沿用 HYLE 全库类型擦除惯例（Fiber::EntryFn / BlockingJob fn-ptr），无 std::function 无 vtable |
| ② | **边逆着创建序 → 插入序天然是拓扑序** | `add_node` 返回 NodeId,节点只能依赖已创建（更小 id）的节点。所以按 id 0..N-1 单趟扫描永远是合法拓扑序——**无需每轮拓扑排序、无需环检测**（环不可构造）。契合 CAD/CAE 自底向上构图（叶/输入先建,消费它们的 op 后建）,是 build-time DAG 的标准简化（Bazel action graph / Salsa·Adapton demand graph 同构建法）|
| ③ | **脏传播 eager（mark_dirty 时）,重算 lazy（evaluate 时）** | mark_dirty(n) 立即把脏标志下游泛洪（廉价,只置标志）;evaluate 单趟前向只对脏节点跑 ComputeFn（其依赖 id 更小已先跑=已新鲜）,清标志;干净节点整个跳过 = 增量的赢点。这是 demand-driven 增量模型（Salsa/Adapton）去掉 demand 追踪重连的简化版 |

**存储 = SoA + CSR**（struct-of-arrays + 压缩稀疏行邻接）：`container::SmallVector` 要求平凡可拷贝元素（memcpy 重定位）,所以节点不能是持有自己 deps SmallVector 的 struct。用 `fn_/ctx_/dirty_` 三个平行数组 + `dep_pool_`（扁平化全部 deps）+ `dep_off_`（CSR 行指针,节点 i 的 deps = `dep_pool_[dep_off_[i]..dep_off_[i+1])`）。既绕开平凡性限制,又是 evaluate 热扫描更好的布局。deps 建图时已知 + id 只增 → CSR 纯 append 构建,无回填。

## step-1 明确不做（step-2/step-3 去向）

| 不做 | 去向 |
|---|---|
| **并行执行**（同拓扑层的独立脏节点派发到 sched work-stealing,when_all 汇合）| **step-2**（graph-design §3:graph 是 sched+compute 之上的纯编排层;节点完成 = 又一种 co_await completion）|
| 数据并行层 parallel_for/reduce/scan | **graph-02**（归属已定案归 graph,消费 sched+compute）|
| "只重算某个输出所需"（demand 追踪）| 后续 refinement（step-1 重算所有脏节点,已是增量;按需 demand 是更细的第二层）|
| JSON 图加载 / kernel 反射 / 动态库插件 / 字符串 URL socket | ⛔ 剥离（clay 应用层耦合,graph-design 明确不做）|

## 验收

- **增量语义**（核心,clay 零资产的部分）：`IncrementalRecomputeOnlyDirtySubgraph`（改一个输入,只它 + 下游重算,兄弟节点跳过复用缓存）+ `UnaffectedBranchNotRecomputed`（无关分支不动）+ `DeepChainPropagatesToTail`（4 节点链,泛洪到尾,防"深度封顶"bug）+ `InteriorDirtyAndLateAdd`（内部节点弄脏 + 已评估图上加节点）。
- **拓扑正确性**：`DiamondFloodsAllPathsOnce`（钻石依赖,汇点每轮跑一次非每边一次）。
- **失败路径**：`FailingNodeAbortsAndStaysDirty`（失败中止 + 留脏重试）+ `DownstreamOfFailureDoesNotRun`。
- **生命周期**：`MoveTransfersState` + `MovedFromIsReusable`（moved-from = 有效空图,CSR seed 存活）。
- **契约 death test**：`AddNullFn`（fn 空）+ `DepNotSmaller`（依赖非更小 id → 破决策 ②）。
- **双平台**：WSL GCC14 + MSVC 各 13/13。

## 对抗式审查（2026-07-10 红队 subagent + 自审）

不跳 CHECKLIST 第 4 步。红队逐条 trace：mark_dirty 单趟前向泛洪的正确性（反证"深度封顶"不可能:任一入 j 的路径上的中间节点都是 j 的传递依赖故 < j 已先处理）、evaluate 单趟扫描 + 中止路径、CSR 边界（dep_off_ 有 N+1 项,`dep_off_[j+1]` 最大 `dep_off_[n_nodes]` 恰末位）、move-assign 的 `~TaskGraph()+placement-new` idiom、uint32↔size_t 收窄、环不可构造——**核心算法确认无懈**。揪出 1 项真缺陷 + 3 测试缺口：

| # | 类别 | 问题 | 修 |
|---|---|---|---|
| 1 | **正确性缺陷** | move 后**源对象 dep_off_ 被清空**（SmallVector move 把源 size 归 0,CSR seed `0` 丢失,违反"dep_off_ 有 num_nodes+1 项 ≥1"不变式）→ 在 moved-from 图上 add_node 会 OOB 读 | move-ctor 体内 `o.dep_off_.push_back(0u)` 重新播种（moved-from = 有效空图,HYLE 惯例）|
| 2 | 测试 | 无深度 ≥4 传递链（Diamond 只到深度 2,封顶 bug 会漏）| 补 `DeepChainPropagatesToTail` |
| 3 | 测试 | 无"内部节点弄脏 / 已评估图加节点" | 补 `InteriorDirtyAndLateAdd` |
| 4 | 测试 | moved-from 图无复用用例（#1 的 OOB 会漏检）| 补 `MovedFromIsReusable` |

**MSVC 额外揪出**（编译期,双平台价值）：C2487（类已 `HYLE_GRAPH_API` 成员再标）+ C4251（非 POD SmallVector 成员被类级 dllexport 波及）→ 按 conventions §4"类有非 POD 成员时宏放方法不放类"改。

## 交付物

- `include/hyle/graph/taskgraph.h`（`TaskGraph` + `NodeId`/`ComputeFn`;SoA+CSR 私有布局;导出宏放方法）
- `src/graph/taskgraph.cc`（add_node/mark_dirty/evaluate/move + CSR 构建）
- `tests/graph/taskgraph_test.cc`（11 功能 + 2 death test）
- `src/graph/CMakeLists.txt` + 顶层 CMake 挂载 + tests 注册
- **LOC（cloc 实测）**：43 header + 78 impl + ~290 test = **~410 code**（step-1 是 graph-01 3-4k 估算的增量核心第一片；建图/脏/缓存/CSR 全落地,并行留 step-2）

## 参考

- [`graph-design.md`](../architecture/graph-design.md)：graph 支柱选型、clay 继承面 vs 从零面、三承重决策、行数依据。
- [`sched-design.md`](../architecture/sched-design.md) + `sched/join.h`：step-2 并行执行要绑的 work-stealing + JoinHandle/when_all。
- clay `plugins/taskgraph`（`slate-render/iron/infra`）：DAG 建模 + 拓扑扫描思路来源（one-shot,无增量）。
- 对标：oneTBB `flow_graph`（DAG 编排）;增量模型 Salsa/Adapton（demand-driven 增量重算）。
