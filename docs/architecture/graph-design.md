# HYLE graph/ 并行任务编排选型

> **V2 状态：`INCOMPLETE / NOT_REVIEWED`。** 本页是半成品阶段的调查与设计历史，不是冻结的
> 产品架构或性能证书。graph 的需求、归属及与 compute/sched 的边界要由真实消费者重新裁决。

> 本文档定下 HYLE `graph/` 支柱（并行任务编排：依赖 DAG 增量重算 + 数据并行）的设计。上游定位见 [`motivation.md`](motivation.md)（北极星② CAD/BIM/CAE）、[`hyle-scope.md`](hyle-scope.md)（DISCUSS-3 归属）；clay 资产盘点见 [`clay-asset-liability.md`](clay-asset-liability.md) 第六节；调度对接见 [`sched-design.md`](sched-design.md)；kernel 对接见 [`compute-design.md`](compute-design.md)。
>
> **一句话**：graph 是**北极星② 独占拉满的那根支柱**——推理引擎几乎不碰它。它承接 CAD/BIM/CAE 工程管线的两类图式计算：**① 参数化/仿真依赖 DAG 的增量重算**（改一个参数只重算下游脏节点，graph-01）、**② 数据并行算法层**（parallel_for/reduce/scan/sort，CAE 网格单元并行/场量归约，graph-02）。二者都跑在 `sched/` work-stealing 之上、几何/求解 kernel 走 `compute/`。

## 一、调查事实（真实数据，本机 cloc + grep 实测 2026-07-10）

clay 的 `plugins/taskgraph/`（iron/infra 版，`slate-render/iron/infra/{include,src}/plugins/taskgraph`）：

| | 文件 | code 行 | 含注释+空 |
|---|---|---|---|
| include（公开头）| 10 | 1,302 | ~2,006 |
| src（实现 + 内部头）| 23 | 2,076 | ~3,085 |
| **合计** | 33 | **3,378** | **~4,890**（≈ clay-asset-liability 表记的 4,890）|

> 口径澄清：clay-asset-liability.md:13 记的"4,890"是**含注释+空行**的总物理行；纯 code 是 **3,378**。两个数都对，只是口径不同。本文档估算一律用 **纯 code 行**。

**逐文件大头**（code 行）：`schema.h` 351（kernel 签名反射）、`data_type.h` 277（通用标量/向量/string 类型系统）、`task_graph.cc` 251（图构建 + 插件加载）、`graph_visitor.cc` 231（4 种 visitor）、`argument.h` 197、`schema.cc` 183、`kernels.h` 174、`graph_node.h` 130、`framebuffer.cc` 125、`graph_executor.cc` 109。

**三条决定性的实测结论（推翻"直接演化/照抄 4890 行"的直觉）**：

1. **clay taskgraph 零增量重算**。`grep -niE "dirty|incremental|recompute|invalidate|memoi|cache"` 在整个 taskgraph（include+src）**0 命中**。`graph_executor.cc:41` 每次执行都从 `ready_for_execution` 的全部 source 出发、`decrease_counter` 降后继入度、归零即投递（one-shot 数据流拓扑序）。**它是"每次全图重跑"的执行器，不是"改参数只重算下游"的增量引擎**。而增量重算恰是北极星② 的核心热路径（motivation.md:28「改一个参数，只重算下游受影响节点/单元」）——**这块 clay 无任何可继承资产，从零造**。

2. **clay taskgraph 焊死在 `std::future` + 裸线程池上**。`graph_executor.cc:66` 用 `context->enqueue(...)` 返回 `std::future<Status>`，`framedata_->wait_for_completion()` 阻塞等。这违反 HYLE 贯穿原则①（不引第二个调度器）——必须换成 HYLE `sched/` 的 fiber + `co_await`/`when_all`。clay-asset-liability.md:130 的处置早已点明：「`TaskQueue` 后端抽象化，对接 HYLE fiber 调度而非裸 `std::thread`」。

3. **clay taskgraph 一半是要剥掉的应用层耦合**。JSON 图加载（`graph_io` 142 行）、动态库插件 `load_plugin`（`task_graph.cc:145`）、`DATATYPE_PYTHON_OBJECT`（void* 外向耦合）、字符串 URL socket（`<node>/<port>` 运行时解析）、kernel 签名反射（`schema.h/.cc` 534 行）——这些是 clay 服务 slate-render 可视化节点编辑器的东西，**HYLE 底座不需要**。里程碑早写了「剥离特化」。

## 二、核心判断

### 1. graph 是"继承 1.5k 骨架 + 从零造 3–4k 增量引擎"，不是"移植 4890 行"

把 clay 3,378 code 行按"可继承 / 要剥离"实测切分：

| clay 部件 | code 行 | HYLE 处置 | 理由 |
|---|---|---|---|
| DAG 建模（node/edge 邻接表 `graph_node`）| ~213 | 🔧 **改后继承** | 通用拓扑结构，剥掉字符串 URL socket 换类型化端口 |
| 拓扑执行骨架（入度计数/source 发现/sink 收尾 `framebuffer`+`graph_executor`）| ~360 | 🔧 **改后继承** | 数据流调度对，但 `std::future`→HYLE fiber、加增量 |
| framebuffer 值容器（arena 化数据流值）| ~220 | 🔧 **改后继承** | arena 池化思路留用，改名避免与图形帧混淆 |
| **小计可继承骨架** | **~800–1,500** | | 剥离后实际留用面 |
| schema kernel 反射（`schema.h/.cc`）| ~534 | ⛔ **剥离** | 服务 JSON/可视化编辑器；HYLE kernel 走 C++ 类型化注册 |
| JSON graph_io | ~142 | ⛔ **剥离** | 图从代码构建，非 JSON 加载（序列化是 close-02 的事）|
| data_type 类型系统 + argument | ~474 | ⛔ **大部剥离** | 通用标量/向量/string + Python 耦合；HYLE 用 C++ 类型 + compute buffer |
| 动态库插件加载 | ~64 | ⛔ **剥离** | 应用层，不属底座 |
| visitor（4 种）| ~231 | 🟡 **按需** | 拓扑/环检测留用，dump/JSON visitor 剥 |

**结论**：clay 4,890（含注释）里真正流进 HYLE 的是 **~800–1,500 行骨架**，其余是要剥的应用层耦合 **或** HYLE 用不上的反射/JSON。**graph 最值钱的增量重算引擎（脏传播 + 结果缓存 + 只跑受影响子图 + 绑 fiber sched）是 clay 零资产、纯新写**。所以"clay taskgraph 占很多行 → HYLE graph 也要很多行"的推断**只对结论（行数相近）不对性质**：不是抄，是"继承小骨架 + 从零造增量引擎 + 换调度底座"。

### 2. graph-02（数据并行层）归 graph，不归 compute

这是 [`hyle-scope.md`](hyle-scope.md) DISCUSS-3 的悬空待决点。基于三条项目内既有约束，拍板 **归 graph**：

- **compute 自己把"铺多核"推给了 graph-02**。compute-design.md:341 立了铁律：「数据并行铺多核**不在后端自起线程**（scope 铁律"不引第二运行时"）——靠 HYLE sched 的 fork-join（graph-02 parallel_for），host_kernel_launch 保持单线程内联」。若把 graph-02 塞进 compute，等于让 compute 违反自己"单线程内联、不起线程池"的边界。
- **消费者与 graph-01 同源**。graph-02 的 parallel_for/reduce 的头号消费者是 CAE 的「网格单元并行 / 场量归约」（motivation.md:20），与 graph-01 的「仿真依赖 DAG 重算」是同一个北极星② 管线里的两种图式计算。放一个支柱，北极星② 的算力编排是一个内聚门面。
- **它消费 compute+simd，不拥有它们**。graph-02 是 oneTBB/oneDPL 那层"数据并行算法"——`parallel_for(range, kernel)` 里的 `kernel` 是 compute 的 host kernel 或 simd 核，`range` 的分块调度靠 sched workers。graph-02 是**编排者**（消费 sched 的并行 + compute/simd 的每块算力），compute 是**算力本体**。职责分明。

> DISCUSS-3 就此闭环：**数据并行层归 graph**（graph-02），消费 sched（workers 的 fork-join）+ compute/simd（每块 kernel）。

### 3. 与 sched / compute 的关系：graph 是纯编排层，零自有执行资源

graph 支柱**不拥有任何线程、stream 或 kernel**——它是站在 sched + compute 之上的编排：

```
graph/（北极星② 编排：依赖 DAG 重算 + 数据并行）
  ├─ graph-01 taskgraph：节点=任务、边=依赖，脏标记增量重算 + 结果缓存
  │     节点体 → 可以是 CPU 任务（sched fiber）或 GPU kernel（compute stream）
  │     执行 → 拓扑序投递到 sched work-stealing，co_await 完成
  └─ graph-02 数据并行：parallel_for/reduce/scan/sort over range
        分块 → sched workers fork-join；每块 → compute host kernel / simd 核
        ↑ 借鉴 oneTBB blocked_range/partitioner；不自起线程池
依赖：sched-03（work-stealing）✅ + compute-04（kernel launch）✅ + simd ✅
```

- **completion 同构**：graph 节点完成 = 又一种可 `co_await` 的 completion（fiber 完成 / GPU stream event），与 io/compute 走同一套 sched 唤醒脊（贯穿原则②）。graph 不新造调度。
- **增量重算的价值面**：graph-01 维护每节点的脏标记 + 上次结果缓存；改一个输入参数 → 标记该节点脏 → 脏沿依赖边向下游传播 → 只有脏节点重新投递执行、干净节点复用缓存。这是 CAD 参数化建模「拖一个滑块只重算受影响的几何」、CAE「改一个边界条件只重算相关网格块」的底座热路径。

## 三、里程碑拆分与行数估算

行数依据：测试按项目实测 test/prod ≈ 0.65。

| 里程碑 | 内容 | 原估 | **实测(cloc)** | 说明 |
|---|---|---|---|---|
| **graph-01** | taskgraph 增量重算：DAG 建模 + 脏标记传播 + 结果缓存 + 拓扑并行 + 绑 sched fiber | 3–4k | ✅ **216 prod / ~490 test（2026-07-10 完成 step-1+2）** | **远低于原估——是精瘦不是 stub（见下）** |
| **graph-02** | 数据并行层：parallel_for/reduce/scan/sort + blocked_range/partitioner | 1–2k | 未做（**设计已起草 [`graph-02-design.md`](graph-02-design.md)**）| 全从零（clay 无此层）；对标 oneTBB/oneDPL 设计（不借运行时）；消费 sched+simd。五决策已拍（递归 split / reduce 双档 / Body 收 subrange / scan v1 / 复用 graph-01 基建）。**参考 graph-01 的收敛，实际大概率也显著低于 1–2k** |

> **⚠️ 原估 3–4k 严重偏高，实测 graph-01 只有 216 生产行——已完成、功能完整（增量语义 + 并行，20 测试跨 WSL/ASan/TSan/MSVC 四配置全绿 + 两轮红队），非半成品。三个原因把估算压下来了：**
> 1. **两个设计决策消灭了整类代码**：「边只能指向更小 id → 插入序天然是拓扑序」这条约束，让**拓扑排序 + 环检测算法（通常数百行）完全不用写**；脏传播退化成一趟 for 循环。
> 2. **图不拥有数据流值系统**：原估把 clay 那套 data_type/socket/AsyncValue（~750 行）+ schema 反射（~534）+ JSON（~142）算进了"继承骨架"，但设计阶段就决定**全剥掉**（节点 = fn-ptr+ctx 闭包，值走用户内存）。原估的"继承 ~800–1,500 骨架"是错的——真正流进来的核心只有一两百行。
> 3. **step-2 复用 sched 现成 spawn_task/join**，不自造并行原语，逐层 fork-join 编排只 ~76 行 impl。
>
> **教训**：估算时把"clay 有多少行"当成"HYLE 要写多少行"的锚是错的——clay 的行数大头是被主动剥离的应用层耦合，不是增量引擎本身。**graph 支柱实际远小于当初的 ~4–6k 估算。**

## 四、开工顺序与前置

- **前置全部就绪**：sched-03 work-stealing ✅、sched-10 when_all/JoinHandle ✅、compute-04 kernel launch ✅、simd ✅、substrate（Chase-Lev deque，container-07/08）✅。**graph 无硬阻塞，可随时开工**。
- **建议顺序**：**graph-01 先**（北极星② 从 0 到 1 的唯一路径，且增量重算是最难、最有价值的从零部分）→ graph-02 后（数据并行层，可独立，CAE 求解器驱动）。
- **北极星视角的优先级**：北极星①（推理引擎）serving+算力两侧已基本齐（io/HTTP ✅ + compute 01~09 ✅）；北极星②（CAD/CAE）**graph 0 行 = 连骨架都不存在**。graph 是当前让"双北极星都立起来"的最高价值下一大块——且是 clay 血统直接背书的老本行（taskgraph/cycles）。

## 五、明确不做（防蔓延）

| 不做 | 理由 |
|---|---|
| JSON 图加载 / 可视化节点编辑器 | clay 服务 slate-render 的应用层；HYLE 图从 C++ 代码构建。图的序列化若需要归 close-02 |
| kernel 签名自动反射（schema）| 服务 JSON/动态注册；HYLE 用 C++ 类型化 kernel 注册，编译期已知 |
| 动态库插件加载 | 应用层，不属底座 |
| 几何核 / 网格布尔 / BVH / 光追 / FEA·CFD 求解器本身 | 这些是 graph 的**消费者**（L1/L2 领域），不是 graph 本身；graph 只提供依赖重算 + 数据并行的编排 |
| 分布式 / 跨机图执行 | 北极星② 的 HPC 分布式求解是更上层的事；graph 支柱先做单机多核 |

## 六、参考

- [`motivation.md`](motivation.md)：北极星② CAD/BIM/CAE 定位、graph 是独占拉满的支柱、clay taskgraph/cycles 血统背书。
- [`clay-asset-liability.md`](clay-asset-liability.md) 第六节：taskgraph 逐文件盘点 + 「改后继承」处置（剥离 TaskQueue 后端/Python 耦合/data_type 扩 tensor）。
- [`compute-design.md`](compute-design.md):341：数据并行铺多核归 graph-02 fork-join、host kernel 单线程内联的铁律。
- [`sched-design.md`](sched-design.md)：work-stealing + Waker completion 脊（graph 节点完成接同一唤醒机制）。
- [`hyle-scope.md`](hyle-scope.md) DISCUSS-3：数据并行层归属（本文档拍板归 graph）。
- clay taskgraph 源码：`slate-render/iron/infra/{include,src}/plugins/taskgraph`（`graph_executor.cc` one-shot 执行核心、`framebuffer.cc` 值容器、`graph_node.h` 邻接表）。
- 对标：oneTBB `flow_graph`（DAG 编排）+ `parallel_for`/`blocked_range`/`partitioner`（数据并行）；oneDPL。
