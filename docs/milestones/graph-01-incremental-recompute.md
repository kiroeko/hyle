# graph-01 (step-1)：任务依赖 DAG 增量重算核心

| 字段 | 值 |
|---|---|
| **阶段** | P6（北极星② 驱动，非纯收口）|
| **模块** | graph/（**支柱破土**——此前 0 行）|
| **前置** | error ✅ / mem ✅ / container(SmallVector) ✅ |
| **状态** | ✅ **step-1 + step-2 + step-3(demand + wavefront)全完成**。**step-1**（WSL+MSVC 13/13）：`TaskGraph` 脏标记增量重算核心，串行 evaluate（纯 CPU 无 sched）。**step-2**（WSL+ASan+TSan+MSVC 各 7/7）：`evaluate_parallel` 同拓扑层独立脏节点**并行跑 sched work-stealing**，逐层 fork-join。**step-3 demand**（WSL+MSVC 19/19 + ASan）：`evaluate_node(target)` 只重算 target 依赖锥。**step-3 wavefront**（2026-07-10，WSL+ASan+TSan+MSVC 各 14/14）：`evaluate_parallel_wavefront` 节点在自己脏 deps 一完成即起（不等整层），长短腿图缩短关键路径。graph-01 三步全落地。|

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

## 做了什么（step-2：并行执行绑 sched work-stealing）

`evaluate_parallel(g, sched, alloc)`（`include/hyle/graph/parallel.h` + `src/graph/parallel_eval.cc`）：把 step-1 串行 evaluate 里"逐个跑脏节点"改成**同拓扑层的独立脏节点并行跑在 sched workers 上**。这是 DAG 执行器的本义（graph-design §3：graph 是纯编排层,不拥有线程,把节点派发到 sched）。CAD 参数化重算 / CAE 网格求解每层有成千独立单元,串行是浪费机器。

**执行模型 = 逐层 fork-join**（最简正确的并行化,dataflow-wavefront 更细留 refinement）：
- 节点 **level = 1 + max(dep level)**,source=0。deps 恒更小 id（决策 ②）→ 单趟前向 id 扫描算全部 level。**同层两节点之间必无边**（有边就强制不同 level）→ 独立、可并发。
- 逐层：spawn_task 该层每个脏节点（在 workers 上扇出）→ join 全部（fiber-aware 屏障）→ 进下一层。节点的 deps 都在更低层、已完成、输入已新鲜。干净节点跳过（复用缓存），与串行一致。
- 层屏障比纯 dataflow wavefront 粗（节点本可在自己 deps 一完成就起,不必等整个上一层）,但简单、正确、契合 CAD/CAE"宽而浅"的层型。

**sched 隔离**（镜像 compute gpu_await 的拆分）：核心 `taskgraph.h` 只依赖 error/mem/container、**处处可编**；`parallel.h`/`parallel_eval.cc` 用 sched（spawn_task/join/fiber）→ 只在 `HYLE_HAS_FCONTEXT` 下编。核心 `TaskGraph` 加一组平凡内联 `*_（）` 私有 accessor 作最小 seam（非公开 API,非 friend,不耦合 sched 类型）。

**关键 UAF（ASan 揪出,开发中即修）**：node task 起初写成 `Task<Status>` 直接返回 Status。但 sched `JoinState<T>` 把返回值存在裸 `storage` 槽、take() 与 release() 各跑一次 `~T()`——对平凡 T 没事,但 **Status 是引用计数的（retain/release）,该值路径双重处理引用计数 → 堆消息 rep 的 use-after-free**（ASan 定位到 `status_retain`）。修=node 结果走**调用方自持的 per-node 槽**（`outcome[]` 数组,GpuOp 同款模式,gpu_await.h:73）、task 返回平凡 bool。`outcome[]` 活过 join+free。

## 验收（step-2）

- **对拍串行**：`DiamondMatchesSerial`（并行结果 == 串行 oracle 的 20）。
- **真并发证明**（非"跑了就算"）：`WideLevelRunsConcurrently`——8 个同层节点各在进入时 bump live-counter、观测 peak>1（4 workers 真重叠）。
- **并行下增量**：`IncrementalReRunsOnlyDirty`（改一个输入,只它+下游重算,兄弟复用缓存）。
- **失败传播**：`FailingNodeSurfacesAndStaysDirty` + `MidLevelFailureStopsLaterLevels`（中间层失败,后续层不跑=验证 break）。
- **混合/多层/空图**：`MixedDirtyCleanAndMultiLevel`（同层脏+干净混合,3 层链）+ `EmptyAndAllCleanAreNoOps`（n==0 早返 + 全干净 no-op）。
- **四配置**：WSL GCC14 + ASan + **TSan（并发代码必过,cheatsheet §3）** + MSVC 各 7/7。

## 对抗式审查（step-2，2026-07-10 红队 subagent + 自审）

红队逐条 trace 并**确认无懈**：level 单趟计算正确（deps 已先 level、同层必无边）、同层独立性、手动数组 handles/ran_ids/outcome 生命周期（n 是单层宽度上界、每个构造的 handle 恰析构一次、break 在 join 循环之后故无正在写的 task 撞 free）、outcome 槽的 join happens-before、bool-return 确实绕开 Status-in-JoinState 的 UAF。actionable：

| # | 类别 | 问题 | 修 |
|---|---|---|---|
| 1 | 文档（真契约缺口）| 同层节点并发,但**输出别名契约**未写明（两条无边同层节点若写重叠用户内存会 data-race 无诊断）| parallel.h 契约补"同层节点须写不相交内存" |
| 2 | 文档 | 失败路径调度与串行**略不同**（并行会跑完失败层已扇出的兄弟,串行在首个失败 id 即停）| parallel.h 补 caveat |
| 3 | 测试 | 中间层失败停后续层未验证 | 补 `MidLevelFailureStopsLaterLevels` |
| 4 | 测试 | 无多层链/同层脏干净混合/空图·全干净 | 补 `MixedDirtyCleanAndMultiLevel` + `EmptyAndAllCleanAreNoOps` |

## 做了什么（step-3 demand 半：`evaluate_node`）

`TaskGraph::evaluate_node(NodeId target)`（`taskgraph.cc`，串行，纯 CPU）：把 `evaluate()`"重算所有脏节点"收窄成"只重算 **target 的依赖锥**（target + 其**脏的**传递祖先），锥外脏节点**延后**"。这是完整的 demand-driven 增量模型（Salsa/Adapton）——调用方只关心一个输出时（CAD 视口只显示一个零件、CAE 探针只读一个场量），无关的脏子图可以等。返回 OK 后 target 干净、其值新鲜；锥外脏节点仍脏，等下次 `evaluate()` 或 demand 到它的 `evaluate_node` 再跑。

**算法（两趟,复用决策 ②"dep 恒更小 id"）**：① **锥标记 = 单趟后向扫描**：`needed[target]=true`,从 target 递减到 0,若 `needed[i]` 则标记 i 的全部 deps needed（deps 的 id < i,在递减扫描里仍在前方,必被标到）。一趟闭合全传递闭包,无递归无重访。② **前向运行 = 0..=target**：跑 `needed[i] && dirty_[i]` 的节点。needed 节点的 deps 全被标 needed（①无条件标全部 dep）→ needed 节点不可能有"脏但未标"的 dep → 输入必新鲜。`needed` 是本次调用 scratch（`SmallVector<bool>` from `alloc_`,尺寸 target+1,锥根是 target 故祖先 id 全 < target）。

## 做了什么（step-3 wavefront 半：`evaluate_parallel_wavefront`）

> **原判"缓"被纠正（用户 2026-07-10：HYLE 是通用基础库,CAD 只是预设消费者之一）**。之前用"CAD/CAE 宽浅层型收益小"给不做找理由——但 wavefront 是 DAG 执行器的**标准能力**(oneTBB flow_graph、clay 自己的执行器都是 wavefront),长短腿/窄深图是真实通用图型(构建系统/CI/任何非均匀 DAG)。拿"我预设的消费者够用"当免死金牌正是被纠正过的 infra-完整性反模式。做全。

`evaluate_parallel_wavefront`：节点在**自己的脏 deps 一完成即起**,不等整个拓扑层 join。step-2 层屏障让 level L 的每个节点等 level L-1 **最慢**的那个;wavefront 去掉这个——快节点的下游不被慢的同层兄弟拖住。长短腿 DAG(一条长关键路径旁挂短分支)缩短关键路径;宽浅 DAG 两者等价。

**机制**:每个脏节点一个 remaining-脏-dep 计数（只在驱动 fiber 单线程改,recv 之间,无需原子）;计数 0 即 spawn 到 sched;完成时经 Channel 把 id 投给驱动 fiber,驱动递减其脏后继的计数、归零即 spawn。反向后继邻接用 CSR 转置(2 趟:计数入 succ_off[dep+1]、前缀和、游标填充)。干净节点不 gate（已新鲜）也不跑。

**⚠️ 红队揪出 CRITICAL UAF（测试全绿却有真 bug,再次印证测试≠对抗式审查）**:node task 起初 `.detach()`,靠 Channel drain-to-zero 当"全完成"信号。但 **`Channel::send` 在 `mu_.unlock()` 之后才 `notify_one()` 触碰 `this`**——两个独立终端节点近乎同时完成时,驱动可能在 task B 的 unlock 与 notify_one 之间就 drain 到 0、返回、销毁栈上 `done` → task B 恢复后 notify_one 打到已释放内存。修=**保留全部 JoinHandle,drain loop 后 join 全部再返回**(join 等到 set_value,set_value 在 co_return 后、co_return 在 send 完全返回后 → join 全完成即保证无 task 还在碰 channel;JoinState 堆+引用计数独立于栈 channel)。红队同时揪出触发形状无测试(#2)→补 `WavefrontManyIndependentTerminalsNoUAF`(16 独立终端,ASan/TSan ×5 无错)。

## 验收（step-3 demand）

- **只跑锥内**：`EvaluateNodeRunsOnlyTargetCone`（两条独立链,demand 一条,另一条脏节点不跑、留脏）。
- **锥内深链 + 复用**：`EvaluateNodeDeepConeAndReuse`（脏祖先重算、干净兄弟复用）。
- **钻石锥**：`EvaluateNodeDiamondSharedAncestorRunsOnce`（共享祖先恰跑一次,非每路一次）。
- **延后不丢**：`DeferredNodePickedUpByFullEvaluate`（demand 延后的脏节点被后续 `evaluate()` 接住）。
- **全干净 no-op** + **锥内失败留脏重试**：`EvaluateNodeAllCleanNoOp` + `EvaluateNodeFailureStaysDirty`。
- **双平台 + ASan**：WSL GCC14 + MSVC 各 19/19；ASan 无泄漏（scratch `needed` 正确释放）。

## 对抗式审查（step-3，2026-07-10 红队 subagent + 自审）

红队逐条 trace 并**确认无懈**：锥标记后向扫描的传递闭包完整性（反证锥内节点不可漏——任一锥内 j 是某 needed i 的 dep 故 j<i,递减扫描里 i 先于 j 处理,j 必被标）、`needed[]` 越界安全（dep < i ≤ t < t+1）、前向运行输入新鲜（needed 节点的 dep 全 needed,无"脏但未标"dep）、延后正确性、失败路径、t=0 退化、scratch 分配无泄漏——**核心 1-7 类全清**。揪 2 测试缺口：钻石锥经 evaluate_node（原只 evaluate() 测过）、延后节点被 full evaluate 接住——均补（`EvaluateNodeDiamondSharedAncestorRunsOnce` + `DeferredNodePickedUpByFullEvaluate`）。

## 验收（step-3 wavefront）

- **对拍层屏障/串行 oracle**：`WavefrontDiamondMatchesSerial`（钻石结果 == fork-join 的 20,各节点跑一次）。
- **并行下增量**：`WavefrontIncrementalReRunsOnlyDirty`。
- **关键路径真缩短**（wavefront 的存在意义,非"跑了就算"）：`WavefrontShortLegNotBlockedBySlowSibling`——慢的 level-0 节点旁挂 fast0→fast1→fast2 快链,断言**快链尾（level-2）在慢的 level-0 兄弟之前完成**(逻辑时钟 finish_tick)。层屏障下 level-2 绝不可能早于某 level-0 → 只有 wavefront 能过。
- **混合脏干净 dep**：`WavefrontMixedDirtyCleanDeps`（只脏 dep gate,干净 dep 不 gate 不跑）。
- **失败停后继 + 空图/全干净**：`WavefrontFailureStopsSuccessors` + `WavefrontEmptyAndAllCleanNoOp`。
- **UAF 回归**：`WavefrontManyIndependentTerminalsNoUAF`（16 独立终端并发完成,ASan+TSan ×5 无错）。
- **四配置**：WSL GCC14 + ASan + TSan + MSVC 各 14/14。

## 对抗式审查（step-3 wavefront，2026-07-10 红队 subagent + 自审）

红队确认 remaining 计数模型/CSR 转置/channel 容量/drain 不变式/happens-before/单线程 remaining/self-schedule 无死锁全对,但揪出 **1 CRITICAL UAF**：栈 `Channel` 在 drain-to-zero 后即销毁,而 `Channel::send` 在 unlock 之后才 notify_one 触碰 this → 多独立终端并发完成时 notify 打到已释放内存。修=保留 JoinHandle、drain 后 join 全部再返回（join 等 set_value,在 send 完全返回之后）+ 补触发形状测试。**又一次坐实"测试全绿 ≠ 无 bug",对抗式审查是必经闸。**

## SOTA 校准（vs taskflow，2026-07-10）+ wavefront 去中心化重构

graph-01/02 完成后用新 clone 的 **taskflow**（经检验的现代 C++ DAG+数据并行框架,`C:\public\project-hyle\taskflow`）做双向校准（透镜:借设计不借运行时——HYLE 跑自己 sched,taskflow 有自己 Executor）。**多数决策确认对,几处超越 taskflow**（增量重算:taskflow 每 run() 全图重跑无增量;reduce 双档:taskflow 只非确定性且 std::mutex 串行 combine;Body 收 subrange:taskflow for_each 逐元素回调杀向量化）。

**揪出一个真 scalability 问题——wavefront 中心化**（这不是功能 bug,功能测试全绿,是吞吐瓶颈）：
- **taskflow 完全去中心化**：完成任务的**那个 worker 自己**原子递减后继的 join_counter,归零后继推进**自己的 WSQ**,还用 continuation 技巧接着跑下一个。无协调者。
- **HYLE 旧 wavefront 中心化**：每个完成经栈 `Channel` 送**单个驱动 fiber**,由它串行递减所有计数 + spawn 所有新就绪。宽 DAG（几千同层独立单元,HYLE 自己的目标）上驱动 fiber 是串行点,worker 空等被重 spawn。那个 send-after-free UAF 本身就是中心化 channel 的症状。

**重构 = 去中心化释放**（彻底修,非打补丁）：`remaining` 计数改**原子**,**完成的 task 自己**在 body 里递减后继计数、归零就 `spawn_task` 自推（`sched` 的 `spawn` 从 worker 内调用**已自动落该 worker 本地队列**,scheduler.cc:641——无需动 sched）。去掉 driver fiber + Channel + join-all-before-return（UAF 随中心化 channel 一并消失）。这也回应"三总目标第③:回看纠正历史选错"。

**明确不做/后置**（登记 backlog,非 graph-01 收尾）：
- **动态/条件任务**（taskflow subflow/condition）：CAE 定点迭代求解/自适应网格靠它表达,HYLE 静态 DAG 现表达不了。graph 后续能力,等消费者。
- **pipeline/data_pipeline**（有序阶段流水并行）：对应北极星① 推理 token 流水,HYLE 零覆盖。候选 graph-03。
- continuation cache（完成 task 直接跑下一就绪后继而非入队）:sched 未暴露"当前 fiber 转跑另一 fiber",基础去中心化不需要,评估后再定。

## 交付物

- `include/hyle/graph/taskgraph.h`（`TaskGraph` + `NodeId`/`ComputeFn`;SoA+CSR;导出宏放方法;内联 accessor seam + `dep_pool_size_` for wavefront 反向邻接）
- `include/hyle/graph/parallel.h`（`evaluate_parallel`[step-2] + `evaluate_parallel_wavefront`[step-3];输出别名/失败路径契约;sched-gated）
- `src/graph/taskgraph.cc`（add_node/mark_dirty/evaluate/**evaluate_node**[step-3 demand]/move + CSR 构建）
- `src/graph/parallel_eval.cc`（step-2 逐层 fork-join + step-3 wavefront:remaining 计数 + CSR 转置后继 + channel 驱动 + join-all 闭 UAF）
- `tests/graph/taskgraph_test.cc`（step-1:11 + 2 death;step-3 demand:+6）+ `tests/graph/parallel_test.cc`（step-2:7 + step-3 wavefront:+7）
- `src/graph/CMakeLists.txt`（核心处处编 / parallel_eval.cc gated on HYLE_HAS_FCONTEXT + hyle_sched）+ 顶层 CMake + tests 注册
- **LOC（cloc 实测）**：step-1 ~410 + step-2 ~281 + step-3 demand ~75 + step-3 wavefront ~292（impl ~103 + decl ~26 + test ~163）= **~1058 code**（graph-01 原估 3-4k 偏高,实测远低=精瘦非 stub,原因见 graph-design 行数修正）

## 参考

- [`graph-design.md`](../architecture/graph-design.md)：graph 支柱选型、clay 继承面 vs 从零面、三承重决策、行数依据。
- [`sched-design.md`](../architecture/sched-design.md) + `sched/join.h`：step-2 并行执行要绑的 work-stealing + JoinHandle/when_all。
- clay `plugins/taskgraph`（`slate-render/iron/infra`）：DAG 建模 + 拓扑扫描思路来源（one-shot,无增量）。
- 对标：oneTBB `flow_graph`（DAG 编排）;增量模型 Salsa/Adapton（demand-driven 增量重算）。
