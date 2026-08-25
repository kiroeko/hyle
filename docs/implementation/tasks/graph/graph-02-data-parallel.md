# graph-02 数据并行层（step-1：parallel_for + parallel_reduce）

| 字段 | 值 |
|---|---|
| **阶段** | P6（北极星② CAD/BIM/CAE 驱动）|
| **模块** | graph/ |
| **前置** | sched(work-stealing + spawn_task/join) ✅ / graph-01 sched 基建 ✅ |
| **状态** | ✅ **step-1 完成（2026-07-10，WSL+ASan+TSan+MSVC 各 14/14）**。`BlockedRange` + `parallel_for`（递归二分 split + grainsize + auto 档）+ `parallel_reduce`（fast 档，左→右 fixed-tree fold）。递归 fork-join 跑 HYLE sched（借 oneTBB 设计,不借运行时）。**step-2（parallel_reduce_deterministic + parallel_scan）/ step-3（parallel_sort）未做**。|

> **一句话**：graph-01 编排"节点/边的依赖 DAG"；graph-02 切一个 range、同一 body 并行跑每块（一个网格百万单元铺满多核）。二者都只**编排**——分块调度靠 sched work-stealing,块内计算是用户 body（典型是 simd 紧循环）。references.md:105 原话："work-stealing 调度器做 fork-join / parallel-for——一个调度器同时干 IO 并发和 CPU 数据并行"。

## 做了什么（step-1）

- **`BlockedRange{begin,end,grainsize}`**：半开区间 + grainsize（oneTBB blocked_range 形,特化到 `size_t` 索引）。`kAutoGrain=0` = 让层自选 grain。
- **`parallel_for(sched, alloc, range, body, ctx)`**：递归二分 split 到 grainsize,每叶块 `body(ctx, b, e)` 并行跑。fork-join：split → spawn 右半 → `co_await` 左半（本 fiber 内联）→ join 右半。返回首个非 OK block Status。
- **`parallel_reduce(..., elem_size, body, combine, identity, ..., result)`（fast 档）**：同样递归 split,每叶归约成 partial,partials 沿 fixed 二分树**左→右** combine。
- **类型擦除 fn-ptr API**（body/combine/identity 都是 fn-ptr + ctx）：sched 机制留在 .cc,头零 sched 类型泄漏（sched-gated TU,同 graph-01/compute gpu_await）。

**5 决策落地**（graph-02-design §3）：① 递归 split + grainsize（work-stealing 友好,非静态均分）② reduce fast 档（fixed-grain 下**确定性**,combine 须结合）③ Body 收 subrange 非逐元素（喂 simd）④ scan 留 step-2 ⑤ 复用 graph-01 sched 基建 + 避 Status-in-JoinState UAF（task 返回 trivial bool,Status 走 ForState/RedState 的 CAS-gated first_err 槽）。

## 对抗式审查（2026-07-10 红队 subagent + 自审）

红队确认核心正确（fork-join 生命周期:root.join 保 ForState 存活;栈深度 log₂(n/grain) 有界≈20 帧,非线性;无 Status-in-JoinState UAF;first_err CAS 无竞争;非交换 combine 的左→右序正确;out 在 combine 前必初始化）。揪出并修:

| # | 类别 | 问题 | 修 |
|---|---|---|---|
| 1 | **真 bug(hang)** | `BlockedRange` ctor 把 `grainsize==0` 强制成 1 → **`kAutoGrain` sentinel 被销毁**,auto 档静默变 grain=1 → 大 range 上每元素一个 64KB fiber → OOM/hang（我的 auto-grain 大 range 测试暴露,小 range 旧测试掩盖了它）| ctor 保留 0,">=1 floor" 交给 `resolve_grain`（它对 explicit/auto 都返 ≥1）|
| 2 | 设计(fiber 成本) | 每次 split fork 一个满 64KB fiber 栈;auto-grain 若 bisect 到 1 会churn ~n 个 fiber | auto-grain 加 `kAutoGrainFloor=1024`,保 fork 粗;explicit grain 不 floor（调用方自担）|
| 3 | **文档误述** | 注释说 "combine 树随完成顺序 → 非确定"——**错**：树形由二分固定,fixed grain 下结果**确定**（float 也是）。step-2 deterministic 是消除跨 grain/worker 数的差异 | 改正头 + 决策② 注释 |
| 4 | 契约缺口 | `elem_size<=256` 是 fatal assert 但头契约只写 `>0` | 头契约补 `elem_size<=256` |
| 5 | 测试 | 无非交换 combine / elem_size>8 payload / 大 range 用例 | 补 `ReduceNonCommutativePreservesOrder`（字符串拼接,200 字节 payload,证左→右序）+ `ReduceLargeRange`(1M grain256)+ `ReduceLargeRangeAutoGrain` |

**教训**:auto-grain sentinel 被 ctor 的"0→1 防御"悄悄吃掉——**"防御性 coerce" 撞上 sentinel 语义是隐患**;小 range 测试掩盖了它,大 range 才暴露(呼应"验收看能力面,拿够本机 range 当无条件结论"的陷阱)。

## 验收

- **正确性**：parallel_for 填数组对拍串行 + 每 index 恰写一次;parallel_reduce 求和对拍闭式;非交换拼接证左→右序;负载不均（人为慢块）仍正确无 hang。
- **auto/grain=1/空/大 range** 全覆盖。body/reduce 错误传播（首个非 OK 返回 + 剪枝）。
- **四配置**：WSL GCC14 14/14 + ASan 14/14 + **TSan 5×14/14**（递归 fork-join 无 race）+ MSVC 14/14。

## step-2/3（未做）

| step | 内容 |
|---|---|
| step-2 | `parallel_reduce_deterministic`（跨 grain/worker 数固定树,bit 可复现）+ `parallel_scan`（两趟 Blelloch）|
| step-3 | `parallel_sort`（并行归并,叶子块借 simd `vqsort`）|

## 交付物

- `include/hyle/graph/data_parallel.h`（`BlockedRange`/`kAutoGrain` + parallel_for/parallel_reduce 声明 + fn-ptr 契约;零 sched 类型泄漏）
- `src/graph/data_parallel.cc`（递归 fork-join `pf_rec`/`pr_rec` + resolve_grain auto-floor + ForState/RedState CAS-gated first_err;gated on HYLE_HAS_FCONTEXT + hyle_sched）
- `tests/graph/data_parallel_test.cc`（14 用例）+ CMake 挂载
- **LOC（cloc 实测）**：39 header + ~120 impl + ~240 test = **~400 code**

## 参考

- [`graph-02-design.md`](../../../architecture/graph-02-design.md)：5 决策 + 归属 + 不做清单。
- 本地 `oneTBB`（设计对标,不借运行时）：`blocked_range.h`（grainsize + 递归 split）。
- graph-01 `src/graph/parallel_eval.cc`：复用的 sched fork-join 基建 + Status-in-JoinState UAF 教训。
