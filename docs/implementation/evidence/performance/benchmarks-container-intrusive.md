# container_intrusive 性能证据

> **状态：`HISTORICAL / UNQUALIFIED`。** `container_intrusive` 刚从 Core 中提取为正式的
> header-only 产品组件；算法本体未改变，但新的 owner、target、真实消费闭包与 candidate 还没有
> 取得 v2 性能资格。下列旧数据只用于回归和设计新 harness，不是当前 SOTA 证书。

## 产品合同

公开 target 为 `hyle::container_intrusive`，公开头为
`<hyle/container/intrusive_list.h>`。它提供 non-owning member hook 双向链表：节点地址必须稳定，
list 不分配也不拥有节点，hook 析构自动 unlink；单节点插入/移除和 whole-list splice 为 O(1)，
遍历/clear 为 O(n)，prefix scan 为 O(k)。time 公开使用这个轻量组件，完整 container 聚合它。

资格 workload 必须保持相同 ownership、hook policy、检查、teardown 和计费边界。拥有/分配节点的
`std::list` 只能作机制下界，不能充当同合同 SOTA peer；当前首选强 peer 是 Boost.Intrusive，真实
消费者是 time/scheduler 的 timer queue。

## 保留的 2026-08 历史运行

这些测量发生在类型仍由 Core 持有时，代码 identity 为：

| 项 | 历史 identity |
|---|---|
| product candidate | `a951f0a436336366df9729e663c2fccfd357cff2` / tree `714d660299bb67d2c855452e8cbdb14cd89ec841` |
| formal harness | `2707583f2fdd0d38f77f3302480be662dfade2f6` / tree `ac85153f7d3241472ace9fc8556afdb88311082a` |
| peer | Boost.Intrusive `df8eeb90846648464e3c8e378e312f0cec772e83` |
| nanobench | v4.6.0 `b709728dba4df0356d467abfb0ba20781a8031c0` |
| 旧 ratio/margin | Boost op/s ÷ HYLE op/s；3% non-inferiority margin |
| 旧 claim scale | N=4096 FIFO、arbitrary remove、traversal、auto-detach |

当时的汇总为：

| 平台 / QoS | FIFO | arbitrary remove | traversal | auto-detach | 旧 non-inferiority gate |
|---|---|---|---|---|---|
| Windows x64 / MSVC 19.50 | `INCONCLUSIVE` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |
| Linux x64 / GCC 14 | `INCONCLUSIVE` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |
| Apple M4 / interactive | `WIN` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |
| Apple M4 / utility | `WIN` | `EQUIVALENT` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |

这些标签只描述归档运行及其旧合同。不得把它们显示为当前绿色，也不能因为迁移前后算法看起来
相同就直接复用。Git 历史保留更早的 raw-file hash、命令和 7 月表格。

### 历史限制

- N=32 cancel 在语义等价的 runtime-bound harness 下翻转方向，只能记为
  placement/code-shape sensitive；
- whole-list append 的 O(1) 重链不随 N 增加计费，batch loop 是机制诊断，不是 scalable product
  benchmark；
- TimerQueue cancellation 还计入 size、owner、deadline 与 empty-transition，不能把全部差异归给
  intrusive list；
- 旧 `TIE`、`EQUIVALENT` 和 `INCONCLUSIVE` 来自不同统计时期，不能混用。

## V2 资格计划

1. 冻结新 owner/API、candidate source/tree、harness、peer、binary、compiler 与 runner identity；
2. 先复核复杂度、常态 metadata、member-pointer portability 与真实生命周期，再计时；
3. 使用 FIFO、arbitrary cancel、traversal、auto-unlink、batch splice 和 timer consumer，但不从现有
   benchmark case 反推产品 API；
4. 与同合同 Boost.Intrusive 对拍，并把 HYLE 的额外语义成本单独归因；
5. Windows、Linux、macOS 运行 default/peer-first、必要的 placement/code-shape perturbation、跨
   进程 raw samples、预声明 margin、exact 五态和独立 non-inferiority gate；
6. 调查每个 `LOSE`、`UNRESOLVED` 或方向不稳定结果，不为赢 benchmark 增加公开 metadata 或改变
   类型形状；
7. 用 time/scheduler 真实消费者确认提取 target 后的 Release codegen 和 end-to-end 机制没有回退。

完成前唯一有效结论是：**算法有一轮同合同三平台历史非劣输入；新的
`container_intrusive` 产品组件尚未取得 v2 SOTA 资格。**
