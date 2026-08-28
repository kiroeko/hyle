# obs-v1 工程性能证据（2026-08-28）

本页只描述当前 `obs-v1` 候选及其仍未通过的门。产品 identity 为
`2c784d9d6a925529995c6a5582d9bc373d91535d` / tree
`bff512b2fd644b3a88bce55be419dadad43c6a89`；基线为 `f305b9d3`。旧 C ABI、旧 runtime
trace bridge 和 2026-07 的三平台点值已经删除，不能用于当前裁决。

## 1. 合同与复杂度

| 操作 | 时间 / 资源合同 |
|---|---|
| 编译期 floor 以下的 `HYLE_LOG_*`、trace OFF | O(0)；参数不求值、无产品符号或 backend |
| `should_log` / runtime drop | O(1)；一次 relaxed threshold load |
| `log_record` / `log_message(string_view)` | O(S)，S 为同步 sink 数；不扫描 exact slice、不分配、不拿 mutation mutex |
| `log_message(const char*)` | O(M + S)，M 为 NUL 前消息长度 |
| format helper | O(formatted bytes + S)；调用方固定 1 KiB 栈缓冲，外来 formatter 异常转静态失败记录 |
| add/remove/flush | O(S)，S≤64；控制面 mutation mutex，remove 等待旧 reader/callback 排空 |
| trace ON native seam | 与相同 direct Tracy event 同阶；无 HYLE runtime adapter |

emit 的 O(S) 是必须逐 sink callback 的下界。registry 使用固定 64 槽、两份进程期 immutable
snapshot 和 per-sink active reference；热路没有 heap、retired list 或 mutation mutex。并发基准中的
线程、sample vector、barrier 和 sink 注册均在计时区外；每轮精确核对
`workers × iterations × sinks` 个 callback。

## 2. W1-WIN 单线程与 fan-out 支持证据

相同 benchmark 源码分别链接当前库和冻结基线；W1-WIN、MSVC 19.50、Release、trace OFF、static，
每个顺序 15 个独立配对进程。共享机器 profile、采样锁、affinity、priority 与
distribution-free process-median 统计均复用 `hyle_perf`。bundle：

`C:\hyle-scratch\obs-v1\logging-regression-20260828e\result.json`
（SHA-256 `E1B9DC2B2AFFF482D72A9312FF04C12CDC71AE012C5CD4CE098754A85B66104D`）。

| workload | 当前中位数（candidate-first / baseline-first） | 结果 |
|---|---:|---|
| runtime gate | 0.651 / 0.675 ns | 两顺序 5% baseline NI PASS |
| preformatted，1 sink | 42.616 / 43.837 ns/emit | NI PASS；≤100 ns PASS |
| structured 4 fields | 38.510 / 39.642 ns/emit | NI PASS |
| `std::format` macro | 93.802 / 97.817 ns/emit | NI PASS；≤150 ns PASS |
| fan-out 1 sink | 42.717 / 43.863 ns/callback | NI PASS |
| fan-out 4 sinks | 19.289 / 19.853 ns/callback | NI PASS |
| fan-out 16 sinks | 13.542 / 13.747 ns/callback | NI PASS |

这些行支持当前正常同步 producer 没有材料级 baseline 回退。它们不是完整日志平台与
Abseil/Folly 的点值排名，也不是 Formal 证书。原 bundle 中的 2/8 producer 行继承了单逻辑 CPU
process mask，不能证明真正并行，已明确排除。

## 3. 8-P-core 多生产者证据与 blocker

`hyle_obs_concurrent_bench` 使用真实 `log_message` + relaxed-atomic adapter callback，8 个 worker
分别固定到 W1 的逻辑 CPU `0,2,4,6,8,10,12,14`：它们来自 8 个不同、最高 efficiency class 的
P-core，排除了 SMT sibling。1/4/16 sink 使用相同源码、编译器和链接合同对拍当前/基线；每个顺序
15 个独立进程。分块 p99 以连续 256 次 emit 为一个 sample，避免让单次 clock read 主导热路；max
仅作 OS 中断诊断，不进入产品门。

证据：
`C:\hyle-scratch\obs-v1\concurrent-regression-20260828a\result-classified.json`
（SHA-256 `D81D444324909AE3B479C7F473F111B434D1481526E3CE81E1556B0C88080A06`）。

| sinks | 当前 throughput ns/emit | 当前 ns/callback | 当前 chunk p99 ns/emit | 5% baseline NI |
|---:|---:|---:|---:|---|
| 1 | 65.299 / 63.313 | 同左 | 668.750 / 666.406 | `UNRESOLVED` |
| 4 | 106.562 / 105.925 | 26.641 / 26.481 | 1080.078 / 1087.891 | `UNRESOLVED` |
| 16 | 224.434 / 225.502 | 14.027 / 14.094 | 2407.422 / 2269.922 | `UNRESOLVED` |

斜线两侧仍是 candidate-first / baseline-first。1-sink 当前 30 个进程的合并中位数为
64.339 ns/emit，预声明 ≤150 ns 的绝对门为 `PASS`；4/16 sink 必须同步执行更多 callback，未把
1-sink 的 per-emit 上限错误套到它们。所有 callback count、线程 singleton affinity、机器与锁门均
通过。

baseline NI 不能签署：全部主行的 99.26% process-median CI 穿过 5% 单边界，因此正确聚合结果是
`INCONCLUSIVE`，不是 `FAIL` 或 `PASS`。一次机制检查确认两版 snapshot/callback 指令逐条相同；
当前 `span` record 的空字段初始化在 MSVC 下比旧 pointer/count 入口多几条 store，但四种自然初始化
写法生成相同 codegen。破坏 exact `span` API、quiescence 或常态 layout 来迎合这一同步争用仪器
不成立。

当前统一仪器只登记单逻辑 CPU 与双核 pair，无法对 8-core 自定义结果执行同一套 A/A、5%/10%
positive control、timer-floor、n=15→31 停止规则与深重放。独立 blocker 见
[`perf-multicore-topology-v1`](../../tasks/performance/perf-multicore-topology-v1.md)。因此
`multi-producer-fanout` 工程门及 obs 聚合 P1 门保持 `INCONCLUSIVE`，模块不得进入 `VERIFIED`。

## 4. Trace native seam

clean fresh `qualify-platform` bundle：
`C:\hyle-scratch\obs-v1\perf-qualify-skip-drain-20260828d-output\summary.json`
（SHA-256 `1BF35D31C8F4F9E4817977D27E68D82A8F7CFA9215FC2CCE7D2B5D31AF2A0575`）。
W1-WIN、Tracy `30997d5c`、n=15 的 A/A、5%/10% delta、timer-floor、placement、build contract、
artifact identity 与深重放全部 PASS。counter、named zone 和 annotation diagnostic 共 12 个
strata 对 direct Tracy 同 event 均为 5% NI PASS；产品 performance gate 为 PASS。

该证据只说明 native macro seam 没有材料级 adapter tax。HYLE 宏直接展开为 Tracy event，不能从
微小点值方向宣称“HYLE 比 Tracy 更快”。

## 5. 当前裁决

- 复杂度和热路资源：同成熟同步 sink registry 档，未发现更优复杂度方案；
- single-adapter、format、单线程 fan-out：绝对门与冻结 baseline NI 均 PASS；
- trace native seam：统一 P1 qualification PASS；
- 8-producer × 1/4/16 sink：绝对 1-sink 门 PASS，但 baseline NI 与 p99 尾部为
  `INCONCLUSIVE`；
- obs-v1 聚合工程性能：`INCONCLUSIVE`。不能宣称“obs 全面 SOTA”、跨平台固定纳秒或已完成收口。

恢复后只允许在独立工具任务提供合格 8-core profile/controls 后重放同一 workload；不得改变 API、
margin、sink callback、分块大小或同步语义来追分。
