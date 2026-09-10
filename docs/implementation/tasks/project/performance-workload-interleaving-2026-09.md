# Workload-matched paired interleaving

## Task card

- trigger：Sched v8 的同一 paired executable 把整段 candidate measurement 全部跑完后才跑 peer；即使两行
  都替换为相同 TaskScope 或相同 manual route，first/second 仍覆盖 `0.917..1.052`，超过 5% 工程门；
- owner：HYLE performance-instrument owner；本任务只解除
  [`sched-v8-task-scope-distributed-completion`](../sched/sched-v8-task-scope-distributed-completion.md)
  登记的 workload-matched A/A blocker；
- scope：使用仓库已锁定 nanobench 的公开 `Bench::compare()`，让相同迭代数的真实 owner operation 以
  paired rounds 和循环拉丁方顺序交错；保持既有 runner schema、adapter、统计器、contract margin、
  Scheduler/allocator/helper/完成边界和双 baseline 顺序；现有严格 row parser 只允许按 nanobench 表头定位
  `op/s`，以接纳 `compare()` 增加的 `95% CI` 列，并补真实输出回归；
- non-goals：修改 TaskScope 或任何产品源码、发明新统计模型/adapter、放宽 5% 门、改样本数、把通用
  算术 control 冒充 workload A/A，或批量迁移历史 benchmark；
- 预算：一次 benchmark/合同重构、至多一次编译修正、一次现有 parser 表头修正、一组 bounded
  same-route A/A 和一组 bounded product A/B；不启动 qualified `n=15`；
- stop：candidate/candidate 或 peer/peer 在 primary/placement、双 baseline 顺序任一仍越过 5%，或
  `compare()` 输出不能被既有严格 parser 无歧义读取，则记录工具 blocker，不继续调参或扩样。

## 源码预审

锁定的 `third_party/nanobench/nanobench.h` 已直接提供所需机制：`Bench::compare()` 先为所有 alternative
校准同一个 iteration count，再逐 round 交错执行；每两个 round 使用随机选择的 cyclic Latin square，两个
alternative 时严格形成 ABBA 或 BAAB。其 `CompareResult` 同时保留每侧绝对 `ns/op`/`op/s`，现有
`scripts/hyle_perf/nanobench.py` 已识别带 `relative` 首列的旧表格，但首次真实运行证明它把列号写死，
会把 `compare()` 新增 `95% CI` 后的 `ns/op` 错读为 `op/s`。修正只按实际 header 查找 `op/s`，不读取
nanobench 的 relative/CI verdict，也不改变 HYLE 的 process-level estimator。因此不需要新 runner adapter
或 HYLE 自实现统计器。

G02 的一次 owner 已包含 32 个 runtime-sized helper 的 admission、yield、结果/首错、finish hook、stack
归还、`Scheduler::drain` 和 owner teardown。交错单位必须保持为完整 owner；不能把 child 原子拆出来。
两条 route 在同一个、已稳定 warmup 的 Scheduler 上交错，符合直接源码消费者只选择一个 Scheduler 的
真实形状，也避免两组 idle worker 互相干扰。两侧保持独立 RouteState/telemetry，验证工作量守恒。

## 验证与结果

benchmark 现在在一个正常、已 warm 的四 worker Scheduler 上运行完整 owner operation；4096-owner/route
warmup 自身交替，`Bench::compare()` 使用 52 paired rounds 和共同 iteration count。`HYLE_BENCH_AA_ROUTE`
只在 preflight 中把两个标记行路由到同一个实现，正式 runner 会按 controlled-environment 合同清除它。

一组固定、单样本 same-route preflight 得到以下 `candidate-label op/s ÷ peer-label op/s`；这里仅用于检查
harness 偏置，不是产品性能证据：

| same route | primary candidate baseline | primary peer baseline | placement candidate baseline | placement peer baseline |
|---|---:|---:|---:|---:|
| candidate/candidate | `0.983` | `1.003` | `0.978` | `1.000` |
| peer/peer | `0.989` | `1.018` | `0.969` | `0.998` |

8/8 位于预声明 `[0.95, 1.05]`，旧 sequential rows 的 `0.917..1.052` 越界不再出现，工具 stop 条件未
触发。随后唯一 bounded 产品 A/B 组的绝对 `op/s` ratio 为 `0.973, 0.896, 0.962, 0.985`；primary
peer-baseline 的 nanobench paired interval 把 manual 优势约束在 `1.04..1.11`。这仍只是 DIAGNOSTIC，
但足以要求恢复后的 Sched 卡先回到源码成本审计，而不是立即启动 qualified P1。

primary/placement executable 都为 `315,392` B，raw `.text` 都为 `0x36200`；placement 只增加 `0x50`
virtual bytes，SHA-256 分别为 `29A05481...` / `F5CE7816...`。contract validate 通过，hash 为
`5419b0483d7b2861fd6a02e27e6d040d377ad1a6ceb02c54daf59379f8d93435`；严格 parser 改为按 header
定位 `op/s`，真实 `relative | 95% CI | ns/op | op/s` 回归与全部 128 项 tooling tests 通过。

本工具 blocker 已解除；本任务不自行改变 `sched` 状态。下一步必须建立新的 Sched closure card，按
`BLOCKED -> SCOPED -> AUDITED` 恢复产品轮，并把这组同合同 signal 纳入有限源码调查。
