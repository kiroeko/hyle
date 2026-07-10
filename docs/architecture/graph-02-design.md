# HYLE graph-02 数据并行层设计

> 本文档定 graph 支柱第二个里程碑 **graph-02 数据并行层**（`parallel_for` / `reduce` / `scan` / `sort` over range）的设计。上游定位见 [`graph-design.md`](./graph-design.md)（graph 支柱选型 + graph-02 归属定案）；调度对接见 [`sched-design.md`](./sched-design.md)；kernel 对接见 [`compute-design.md`](./compute-design.md) + `simd`；对标源见本地 `oneTBB`（`include/oneapi/tbb/blocked_range.h`、`parallel_reduce.h`、`parallel_for.h`）。
>
> **一句话**：graph-01 编排"节点/边的依赖 DAG"（改一个参数重算下游）；graph-02 编排"一个大 range 切块并行跑同一 kernel"（一个网格 100 万单元铺满多核）。二者是 graph 支柱的两种正交图式计算，都**只编排、不拥有算力**——range 分块调度靠 sched work-stealing，每块的计算是 simd 核 / compute host kernel。references.md:105 的原话即此：「work-stealing 调度器做 fork-join / parallel-for——一个调度器同时干 IO 并发和 CPU 数据并行，这正是 work-stealing 的本职」。

## 一、定位与归属（已定案，本文只承接）

- **归 graph，不归 compute**（graph-design §2，2026-07-10 定案）：compute 立铁律「数据并行铺多核靠 graph-02 fork-join，host_kernel_launch 保持单线程内联」（compute-design:341），归 compute 会破其"不起线程池"边界。
- **不借 oneTBB/oneDPL/OpenMP**（references.md:99/114）：它们**自带 work-stealing 调度器 / task arena**，引入 = HYLE fiber 调度器 + TBB 调度器两套在同一批核上抢线程、过度订阅、cache 争抢——正是 HYLE 要消灭的双调度器阻抗失配。**借 oneTBB 的设计（blocked_range/partitioner/deterministic-reduce 的 SOTA 算法），不借它的运行时**。跑在 HYLE sched 上。
- **全从零**：clay 无数据并行层（clay taskgraph 是 DAG 编排，非 range 并行）。

## 二、API 面（对标 oneTBB 数据并行族，跑 HYLE sched）

```
graph/parallel.h（step-2/3 已有 DAG 并行；graph-02 加 range 并行）
  parallel_for  (Range, Body)                    每块 Body(subrange) 并行
  parallel_reduce(Range, identity, Body, Combine) 分块局部归约 + 树形合并
  parallel_scan (Range, identity, Body, Combine)  前缀和（两趟 up/down-sweep）
  parallel_sort (RandomIt, Compare)               并行归并/快排（借 simd vqsort 每块）
  BlockedRange<T>{begin, end, grainsize}          可递归 split 的一维区间
```

Body 收 **subrange `[begin,end)`**（不是逐元素回调）——这是与 simd 对接的关键（见决策③）。全部 `noexcept` + `Status` 错误模型；从 fiber 调用（复用 graph-01 的 sched 绑定 + 输出别名契约）。

## 三、五个待决点的决策（按"高性能 SOTA infra 库"愿景拍板）

### 决策① range 分块策略 = 递归二分 split + grainsize（work-stealing 友好），非静态均分

**选**：`BlockedRange{begin,end,grainsize}`，`is_divisible() = grainsize < size()`，`split` 构造把区间二分（oneTBB `blocked_range.h:44/64/67` 原样思路）。`parallel_for` 递归 split 到不可分，每个叶子 subrange spawn 一个 sched 任务。

**为什么不静态均分**（`N/num_workers` 每人一块）：静态均分在**负载不均**时垮掉——某块的单元恰好都是慢路径（CAE 里非线性单元、几何里退化面），持有它的 worker 干到死、其他 worker 空转。递归 split 产生的细粒度叶子任务喂给 HYLE work-stealing：慢块的兄弟任务被其他 worker 偷走，负载自动均衡。**这正是 HYLE 选 work-stealing 而非 share-nothing 的同一理由**（motivation.md：北极星是"唤醒突发且不可均分"）——静态均分是 share-nothing 的思路，与 HYLE 的调度哲学冲突。SOTA 库（oneTBB/Cilk）全用递归 split，不用静态均分。

**grainsize** = 停止 split 的阈值（一块 ≤ grainsize 就不再分，直接跑）。太小 → 任务调度开销淹没计算；太大 → 并行度不足、负载不均。默认由 `simple_partitioner` 语义给（grainsize 显式）；v1 提供 `auto` 档（对标 TBB auto_partitioner：按 worker 数 × 过采样因子推 grainsize，让任务数 ≈ 几倍 worker 数，够偷但不碎）。

### 决策② reduce 结合性 = **提供 fast + deterministic 双档**（不二选一）

**选**：`parallel_reduce` 默认走**非确定性树形归并**（最快，任意 combine 树形状），另提供 `parallel_reduce_deterministic`（固定归并树 → run-to-run 逐位一致）。这正是 oneTBB 的做法（`parallel_reduce.h` 有 `reduction_tree_node` 和独立的 `deterministic_reduction_tree_node`）。

**为什么双档而非单选**：SOTA infra 库不能替消费者做精度/速度的 trade：
- **fast 档**：combine 必须**数学可结合**（用户契约，文档写明）。浮点加法**非**严格可结合 → fast 档的浮点 reduce 结果随 worker 数/偷取时序**逐位漂移**（但在 combine 的数值稳定范围内）。AI 训练/推理的 loss 求和、大多数 CAE 残差范数能接受。
- **deterministic 档**：固定二分归并树，与 worker 数、偷取时序**无关** → run-to-run 逐位可复现。**回归测试、科学计算复现性、调试**必需。代价是不能自由 work-steal 合并顺序（略慢）。

**契约闸**（同 simd best-of 纪律：正确性优先于速度）：两档都要求 combine 满足单位元 + 可结合的**代数**契约；fast 档额外声明"浮点非严格结合 → 结果在数值误差内、非逐位确定"。文档明确 identity 必须是 combine 的真单位元（`combine(identity, x) == x`），否则 fatal。

### 决策③ kernel 接口 = **Body 收 subrange**（`[begin,end)`），非逐元素回调

**选**：`Body` 签名是 `Status(BlockedRange<T> subrange)`，用户在 Body 内自己写这一块的循环。**不**做逐元素 `fn(i)` 回调。

**为什么**：逐元素回调是**性能杀手 + 与 simd 对立**——每元素一次间接调用，编译器无法向量化、无法软件流水。收 subrange 让用户（或 graph-02 的 simd 特化）在块内写**紧循环 → 直接喂 simd `vec<T,N>`**（simd 支柱的整存整取），一块 = 一段连续内存 = simd 的主场。这是 oneTBB Body 的设计（收 `blocked_range`），也是数据并行能吃满带宽/向量单元的前提。graph-02 与 simd 的接缝就在这里：**分块靠 graph-02（sched work-stealing），块内靠 simd（向量化紧循环）**——两个支柱各司其职、零阻抗。

### 决策④ scan v1 就做（两趟 up/down-sweep），不推迟

**选**：`parallel_scan` v1 交付，用经典两趟（Blelloch）：① up-sweep 分块算局部 reduce ② 块间 exclusive scan ③ down-sweep 每块带前缀基值再 scan。

**为什么不推迟**：scan（前缀和）是**通用 infra 的基础数据并行原语**（流压缩、排序基数、稀疏结构构建、CAE 装配偏移），不是可选件。SOTA 数据并行层（oneTBB parallel_scan、Thrust、oneDPL）都含 scan。按"能力做全"纪律（feedback_infra_capability_completeness：验收看能力面完整非"够当前消费者"），scan 属于数据并行的定义性 API，缺它这层就不完整。它复用 parallel_reduce 的分块基建，增量成本低。

### 决策⑤ 复用 graph-01 的 sched 基建，不另起一套

**选**：graph-02 的 range 任务走 graph-01 step-2/3 已建的同一条路——`spawn_task` + fiber-aware join + 从 fiber 调用的契约 + outcome 槽避 Status-in-JoinState UAF（graph-01 step-2 血泪）+ 栈同步对象 join-all-before-return（wavefront UAF 教训）。`parallel_eval.cc` 已有的模式直接搬。

**为什么**：这些坑 graph-01 三轮红队全踩过、修过（move OOB / Status-in-JoinState UAF / wavefront 栈 Channel send-after-free）。另起一套 = 重踩同样的坑。递归 split 的 fork-join 与 graph-01 的节点 fan-out 是同构的 sched 用法。**唯一新增**是 BlockedRange 的递归二分逻辑 + reduce 的树形合并 + scan 的两趟，那是算法层，不碰 sched 绑定。

## 四、与其它支柱的接缝（一图厘清"谁拥有什么"）

```
graph-02 parallel_for(range, body)      ← 编排：递归 split + spawn 到 sched（不拥有线程）
   ├─ 分块调度  → sched work-stealing（graph-01 已绑定的同一套）
   └─ 每块计算  → body 内紧循环 → simd vec<T,N>（块内向量化）
                 或 → compute host kernel（compute-07 的 Kernel::HostFn）
借来的算子（ggml/oneDNN）自带并行 → 注入 HYLE 线程池（set_threadpool），不经 graph-02
```

**边界**：graph-02 只管"把 range 铺到多核 + 合并结果"；单块内怎么算是 simd/compute 的事；AI-GEMM 那种整算子并行是 ggml 自带并行（注入 HYLE 池，references.md:104），不重造。

## 五、里程碑拆分与验收

| step | 内容 | 依赖 |
|---|---|---|
| **step-1** | `BlockedRange` + `parallel_for`（递归 split + grainsize + auto 档）+ `parallel_reduce`（fast 档）| sched ✅ |
| **step-2** | `parallel_reduce_deterministic`（固定树）+ `parallel_scan`（两趟）| step-1 |
| **step-3** | `parallel_sort`（并行归并，叶子块借 simd `vqsort`）| step-1 + simd-09 vqsort ✅ |

**验收**（对标 SOTA，非只跑通）：
- **正确性**:parallel_for/reduce/scan/sort 对拍串行参考;reduce deterministic 档 run-to-run 逐位一致(多 worker 数 × 多次跑)、fast 档在数值误差内。
- **scaling**:parallel_for 加速比随 worker 数近线性(到内存带宽墙),负载不均场景(人为慢块)证 work-stealing 偷取生效(对比静态均分的落后)。
- **对标 oneTBB**:同 range/grainsize 下吞吐打平(bench-stat 统计法,不看单点);grainsize 扫描曲线合理。
- **并发**:WSL TSan 净(递归 split 的 fork-join 无 race);ASan 净。
- **双平台**:WSL GCC14 + MSVC;每 step 红队闭环(CHECKLIST 第 4 步,不跳)。

## 六、明确不做

| 不做 | 理由 |
|---|---|
| GPU 数据并行（range 铺到 GPU 线程）| v1 CPU 数据并行(sched workers + simd);GPU 并行是 compute 自造 kernel(04/06)的 grid,不同机制 |
| 借 oneTBB/oneDPL/OpenMP 运行时 | 双调度器红线(references.md:99) |
| 分布式/跨机数据并行 | 北极星② HPC 分布式是更上层;graph 先单机多核(同 graph-01 边界) |
| 重造 GEMM/BLAS 的并行 | 借 ggml/oneDNN 自带并行(注入 HYLE 池) |

## 七、参考

- [`graph-design.md`](./graph-design.md) §2：graph-02 归属定案（归 graph）。
- [`references.md`](./references.md):99-106：为何不借 oneTBB 运行时 + CPU 数据并行三层（自带并行注入 / HYLE 自有 fork-join / 指令级借核）。
- [`compute-design.md`](./compute-design.md):341：数据并行铺多核归 graph-02、host kernel 单线程内联的铁律。
- 本地 `oneTBB`（**设计对标源,不借运行时**）：`blocked_range.h`（grainsize + 递归 split）、`parallel_reduce.h`（reduction_tree_node + deterministic_reduction_tree_node 双档）、`parallel_scan.h`（两趟）。
- graph-01 `src/graph/parallel_eval.cc`：复用的 sched fork-join 基建 + 三个 UAF 教训。
- `simd`（simd-09 vqsort）：parallel_sort 叶子块的向量排序核。
