# obs-v2 工程性能证据（2026-08-29）

本页只描述当前 `obs-v2` candidate `ff925c0053e738fb48d98993e42750b593f3db22` / tree
`34c5903c9a3d19bfe8542ac1f050314147c73254` 的工程性能事实；冻结基线为 `f305b9d3`。v2 只改变
public callback 注释、CTest 环境和文档，`src/obs`、公开类型 token、机器码与 v1 产品候选相同，因此
下列合格 bundle 可按 exact product implementation identity 重新绑定。旧 C ABI、旧 runtime trace
bridge 和 2026-07 的三平台点值已经删除，不能用于当前裁决。

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

## 2. W1-WIN 单线程与 fan-out retained 支持证据

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

## 3. 8-P-core 多生产者合格证据

`hyle_obs_concurrent_bench` 使用真实 `log_message` + relaxed-atomic adapter callback。中央
`n-worker-physical-cores` profile 从 Windows CPU Sets 选择同 processor group、最高 efficiency class、
互不为 SMT sibling 的 8 个物理核；本机观察到逻辑 CPU `0,2,4,6,8,10,12,14`。每个 worker 的请求/
观察 CPU、singleton mask、work item 与 callback 数均由 structured row 和 deep replay 双重核对。
1/4/16 sink 使用相同源码、MSVC 19.50、Release/static 合同分别链接当前实现和 immutable
`f305b9d3` 基线；candidate-first / peer-first 双顺序从 n=15 机械升级并完整 fresh 重跑到 n=31。
分块 p99 以连续 256 次 emit 为一个 sample，避免单次 clock read 主导热路。

证据目录为 `C:\hyle-scratch\obs-nworker-qualified-evidence-8a704f7a`。contract SHA-256 是
`E981F348E960184AA08D4E1D9E61767B9F412B91D586BD7DA708545417AF556B`；manifest / summary /
controls / raw-log-index SHA-256 分别为
`3B4538580989A9665423E5C02E2C1B587E2FB75071D7223F9B71784731B05662`、
`02DD48AC65558B78C4C2F21961D62F2F9FE74671528F4CDA3BF46B6DABFFDC72`、
`80A8157E9BC3B6E0DF67F06BE63968EE3004A773DD36C9B7429E68169A92B758`、
`6788B19471669AA2E74118C9AB6FC0A2E7A871346248BB1CE3192E5FC6CC4275`。qualification、A/A、
5%/10% known delta、timer floor、host/topology、build/artifact/accounting 与 raw deep replay 全部
`PASS`。下表 ratio 大于 1 有利于当前实现，区间为冻结的 98.93% process-median CI：

| workload | candidate-first ratio / CI / NI | peer-first ratio / CI / NI | combined NI |
|---|---|---|---|
| fanout-1 throughput | 1.010 / [0.985, 1.044] / `PASS` | 0.996 / [0.968, 1.019] / `PASS` | `PASS` |
| fanout-1 chunk p99 | 1.014 / [0.958, 1.055] / `PASS` | 0.986 / [0.958, 1.040] / `PASS` | `PASS` |
| fanout-4 throughput | 1.001 / [0.942, 1.052] / `UNRESOLVED` | 1.002 / [0.960, 1.038] / `PASS` | `UNRESOLVED` |
| fanout-4 chunk p99 | 0.998 / [0.945, 1.046] / `UNRESOLVED` | 0.980 / [0.950, 1.029] / `UNRESOLVED` | `UNRESOLVED` |
| fanout-16 throughput | 0.986 / [0.810, 1.018] / `UNRESOLVED` | 1.007 / [0.968, 1.250] / `PASS` | `UNRESOLVED` |
| fanout-16 chunk p99 | 0.973 / [0.856, 1.005] / `UNRESOLVED` | 1.015 / [0.947, 1.192] / `UNRESOLVED` | `UNRESOLVED` |

1-sink throughput 与 tail 的两顺序 5% NI 均通过；4/16 sink 至少一个冻结 stratum 的 CI 穿过 0.95
下界，所以聚合 NI 必须是 `INCONCLUSIVE`，不能用接近 1 的 median 写成平局或 PASS。

唯一机制诊断确认两版 `SinkSnapshot`、active-reference、callback loop 与 release 路径算法一致；MSVC
反汇编只在当前 exact-view `LogRecord` 构造看到用于 optional fields 的额外 `xorps/movups` 零初始化。
这是一项固定 per-emit 成本，在 1-sink 已通过且随 sink 增多应被摊薄，不能解释 4/16-sink 方差扩大。
两个 artifact 的绝对分布都有相同快慢模态，relaxed-atomic adapter 的跨核 cache-line handoff 又是双方
共有成本；没有“可复现产品机制 + consumer materiality”双钥匙，故不改 API/layout/quiescence 或
benchmark 合同追分。中央工具缺口已经由
[`perf-multicore-topology-v1`](../../tasks/performance/perf-multicore-topology-v1.md) 完成，当前 blocker
只剩合格数据本身的统计未决。

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
- 8-producer × 1 sink：throughput/tail 两顺序 5% baseline NI `PASS`；
- 8-producer × 4/16 sink：合格 n=31 后仍有冻结 strata `UNRESOLVED`；
- obs-v2 聚合工程性能：`INCONCLUSIVE`。这不是已证明的产品回退，但不满足 P1 `CLOSED` 硬门；不能
  宣称“obs 全面 SOTA”、跨平台固定纳秒或已完成收口，也不能继续追加同合同样本追逐结论。
