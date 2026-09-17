# R3 CPU：比较边界与 decode 成本归因校准

2026-09-17；文档核对 baseline `647ff7feaba0f0b2849304a47657958c1f289a7c`。
用户要求核对先前子代理关于“苹果对橘子”的调查。本轮只核对冻结源码、原始结果与当前账本；
不修改 CPU 产品、不重开 CLOSED owner、不重建、不重新计时、不撤销用户当时的工程验收决定。

## 结论

`decode-fused-boundary-1x256x512` 测到的是完整 Graph API 相对手写融合调用的增量成本。
数学/精度/融合、输入输出映射和终态边界相同，**API 服务不等价**，不能作为同等能力的算法、
Sched 或引擎性能输赢。原测量有效回答这个包装成本问题，原 5% 是成本目标，不是同合同 peer NI。
先前调查已经指出这一点，当前机器账本却仍写 `SAME_CONTRACT_INTERNAL`，属于结论落账错误。

核对绑定产品 `1a3a8d0e40a2a17250119369331c712718a4d880` 与 runner
`7fa6dc0a8b3b857a1835525de0d0e7be5f8db5e9`；源码 blob、原分类和不变统计值见
[逐项记录](../../../../evidence/review/cpu-r3-comparison-boundary/identity.json)。原始
[P1 包](../../../../evidence/performance/graph-cpu-route-p1-v3-7fa6dc0a-w1-win/README.md)
及[用户验收卡](cpu-r3-acceptance-exception.md)保持历史事实，不改写成新的 PASS。

## 源码责任账单

以下锚点均对应冻结 runner，不用当前行号推断旧制品行为。

| 项目 | 完整 Graph | 直接融合参考 |
|---|---|---|
| 计算 | 同一 oneDNN BRGEMM pair、HYLE SIMD SwiGLU、down BRGEMM | 相同 |
| 准备 | JIT、分配、绑定、持久 workspace 映射在计时外 | 相同 |
| 边界数据 | 四输入、一输出的 map/close，完成后返回 | 相同 |
| 运行合同 | Invocation 排他、Context/Queue admission、关闭/错误协议、Completion | 直接持有 plan/指针，失败由 bench `require/fail` abort |
| decode 线程 | 调用线程串行；没有 executor | 相同；Sched 只在后面的 two-worker workload 启动 |

- `bench/graph_route_cpu/region_p1_bench.cc`：Graph `run` 305，直接融合 `run` 439，
  `fail/require` 73/92，`run_shapes` 669。它自身把这一组称为 `abstraction`。
- `src/graph/executable.cc:7510`：Invocation in-flight CAS，计划适用时的排他，以及完成状态管理。
- `src/compute/device.cc:895`：Context 与 Queue 两次 admission CAS、producer 原子标志；
  `end_submission` 释放两次 admission。
- `src/compute/context_state.h:29`：计数从 1 降到 0 就 `notify_all()`，不先判断是否存在等待者。

## 最可能的原因与确定性边界

1. **固定接口管理成本最可能。** 八个分层均偏慢；约 12 微秒上 2.6%–3.5% 是数百纳秒，
   与额外原子接纳、排他、分发和完成管理的结构一致。数值核没有因此变慢的证据。
2. **两次无等待者也会走的 notification 是具体嫌疑。** 先前只读调查在冻结对象中确认调用存在；
   本轮复核源码仍可确认这两个调用条件。没有分阶段测量，不能声称全部差额来自它们，也不能把
   `notify_all()` 的次数直接换算为固定 syscall 时间。
3. **统计波动影响最后的越界。** 原 median ratio `0.966131–0.974263`；一个 CI 下界
   `0.9517370300` 略低于 `1/1.05`，允许约 5.07% 额外耗时。不能归零为噪声，也不能据此
   把 5.00% 与 5.07% 解读为可稳定分辨的算法差距。
4. **Sched 不是该 decode 项的原因。** 它没有 executor；不能据此重开调度器。

测试使用缩小的 warm FFN，固定内容重复，数值在 warm-up 对照；不代表变化 token 轨迹或完整
模型吞吐。没有发现可否定原成本观察的数值/融合/同步偷换，问题是把成本诊断升级成同合同排名。

可能的未来优化是准确跟踪等待者，避免无等待者通知；不能简单改成“关闭后才通知”，因为开放
Context 也允许 `drain()`。没有真实消费者提出新预算或新的合格机制收益前，不修改等待协议。

## 当前账本更正

同一模板的四个 `fused-boundary` 与四个 `natural-sequence` 都是裸 direct 对 Graph，统一改为
`NATURAL_ROUTE / REJECTED_DIAGNOSTIC`，同合同排名与 NI 均为 `N/A`。八项的原方向、原工程门、
绝对值、ratio/CI 和用户取舍全部在逐项记录及各 claim limitations 中保留；对有利的 two-worker
natural `WIN` 同样撤销排名资格，不只重分类负向项。`REJECTED_DIAGNOSTIC` 表示排除排名，
不表示数值检查或测量采集失败。

四个 `native-sequence` 对照的两侧均为完整 `GraphRoute`，继续保留原同合同分类：1 LOSS、
1 WIN、2 INCONCLUSIVE，四项原 5% NI 通过。这项真实 LOSS 不被包装成本校准掩盖。
CPU Direct 四项本轮未重新审计，原记录不变。

因此当前应说：**用户当时接受了一个小片段的 Graph 接口成本预算取舍**；不能说用户豁免了
已经证明的同等服务算法退化。没有制造新的性能 PASS，也没有重新打开 CPU 优化队列。

统一脚本本轮统计全部 Compute + Graph **36,765 行 / 129 个代码文件**，不含测试；代码未改。
停止此次有界核对，继续 R7。后续需要重新比较时，先为参考补齐真实等责 owner/lifecycle，或
继续明确标作自然路线成本；不通过改 benchmark 命名获取排名。
