# Core 性能证据

> **状态：`HISTORICAL / UNQUALIFIED`。** Core 正处于 v2 `OPEN` 复审，没有当前性能证书，也不
> 宣称模块已达到 SOTA。下列历史测量仍可用于回归和设计新 harness，但不能证明当前产品边界。
> 当前施工只看 [`core-rebuild`](../../tasks/core/core-rebuild.md)。

## 哪些能力需要什么性能证据

Core 当前四个能力族需要不同证据形状：

| 能力族 | 合适的证据 | 当前资格 |
|---|---|---|
| 产品身份 | complexity/code-size 与冷路径正确性；不能制造 getter 竞速 | 源码上 O(1)，但 ownership 与 linked API 是否需要仍未裁决；无 SOTA claim |
| compiler/binary 合同 | toolchain codegen 加真实消费 kernel | 有旧 object inspection；未绑定 v2 candidate 或完整消费者账本 |
| cache layout 与 prefetch | footprint/contention 分析，加拥有该访问模式的 container/mem/sched/SIMD workload | 未决；单独计时 prefetch 不能给能力签证书 |
| intrusive list | 同合同强 peer workload 加真实 time/scheduler consumer | 有旧 Boost 对拍；边界、candidate 与 workload 合同重签前只算历史输入 |

当前 Core 产品复审的 Native FFI 为 `N/A`，因此没有 Core FFI 性能 claim。未来若把真实 loader/
bootstrap 任务分配给其他 owner，其动态调用成本归该产品旅程。

## 为什么撤销旧绿色证书

旧文档把 Core 标记为 `SIGNED`，并用 intrusive-list 结果描述整个模块已经性能收口。v2 不能继承
该状态，原因是：

- 尚未冻结 v2 candidate commit/tree；
- 四个能力族与根级文件的 ownership 未裁决；
- 语法消费者与语义重复账本未完成；
- compiler/cache 行为必须重新绑定当前真实消费者和当前工具链；
- list microbenchmark 不能证明 version、visibility、compiler 或 cache 合同；
- time/scheduler consumer 测量还计入自身 bookkeeping，必须单独归因；
- short-list 结果已经被证明对 code-shape/placement 敏感；
- 当前 Core 产品把 Native FFI 定为 `N/A`，旧证书却把三个 provisional C exports 和 managed smoke
  当作完成度的一部分。

降级不表示每个旧数字都错了；它表示旧证据不能证明当前 claim。

## 保留的 2026-08 历史运行

旧证据中最强的一轮登记了以下 identity：

| 项 | 历史 identity |
|---|---|
| product candidate | `a951f0a436336366df9729e663c2fccfd357cff2` / tree `714d660299bb67d2c855452e8cbdb14cd89ec841` |
| formal harness | `2707583f2fdd0d38f77f3302480be662dfade2f6` / tree `ac85153f7d3241472ace9fc8556afdb88311082a` |
| peer | Boost.Intrusive `df8eeb90846648464e3c8e378e312f0cec772e83` |
| nanobench | v4.6.0 `b709728dba4df0356d467abfb0ba20781a8031c0` |
| 旧 ratio/margin | Boost op/s ÷ HYLE op/s；3% non-inferiority margin |
| 旧 claim scale | N=4096 FIFO、arbitrary remove、traversal、auto-detach |

当时的历史汇总为：

| 平台 / QoS | FIFO | arbitrary remove | traversal | auto-detach | 旧 non-inferiority gate |
|---|---|---|---|---|---|
| Windows x64 / MSVC 19.50 | `INCONCLUSIVE` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |
| Linux x64 / GCC 14 | `INCONCLUSIVE` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |
| Apple M4 / interactive | `WIN` | `INCONCLUSIVE` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |
| Apple M4 / utility | `WIN` | `EQUIVALENT` | `EQUIVALENT` | `EQUIVALENT` | 4/4 pass |

这些标签只描述该归档运行及其当时合同；不得把它们显示为当前绿色、合成为“Core 已 SOTA”，也
不能只因 list 源码看起来相似就复用。Git 历史保留旧文档中的详细 CI、raw-file hash 与 7 月表格，
未来重建审计链时仍可取回。

### 必须保留的历史限制

- N=32 cancel 在语义等价的 runtime-bound harness 下翻转方向，因此只能记 placement/code-shape
  sensitive，不能签产品胜负；
- whole-list append 的 O(1) 重链不随 N 增加计费操作；这种 batch loop 是机制诊断，不是 scalable
  product benchmark；
- time `TimerQueue` cancellation 同时计入 size、owner、deadline 与 empty-transition 语义；合同更
  弱的 peer 无法隔离 Core list 成本；
- 旧 prefetch/compiler 证据在部分工具链看到预期指令或安全退化，但“发出了指令”不能证明
  consumer 加速，也不能证明 code-size/footprint 可接受；
- 更早的 7 月表格使用另一套统计解释，其中 `TIE` 不等于当前 `EQUIVALENT`。

## V2 资格计划

只有 ownership/API 审计完成并冻结 candidate 后，才开始性能签证：

1. 先复核算法复杂度和常态 metadata，再计时；
2. 从 Core 正常用户与 HYLE 真实消费者选择 workload，不从现有 benchmark case 反推需求；
3. 选择 ownership、检查、mutation、iteration、teardown 与计费边界相同的当前强 peer；
4. 把机制诊断、产品 claim 和 time/scheduler 自己拥有的语义成本分账；
5. 在 footprint/contention workload 中验证 cache 常量，在拥有访问模式的 kernel 中验证
   prefetch/compiler hint；
6. Windows、Linux、macOS 都运行 default/peer-first、必要的等价 code-shape/placement perturbation、
   至少 15 个独立进程样本、逐进程 raw JSON、预声明 margin、exact 五态与独立 non-inferiority gate；
7. 调查每个 `LOSE`、`UNRESOLVED` 或不稳定方向，不得为得到赢面修改 public API、object metadata
   或挑 benchmark shape；
8. 把 source、harness、peer、runner、binary、toolchain、command 与 log 绑定
   `core-rebuild` 记录的同一 candidate。

完成前唯一有效结论是：**Core 性能尚未取得资格；历史 intrusive-list 数据在旧合同的 N=4096
路径没有显示明显回归，但不是当前 SOTA 证书。**
