# Sched v5：IO completion 与空闲搜索协同

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen trigger

- component / owner / round：`sched` / Sched maintainers / `sched-v5-io-completion-search`；
- task kind：`REOPEN`；baseline：
  `d53cdadfb1184a57f5ed18880edfeaf719a4412f`；
- trigger ID：`SCHED-V5-IO-COMPLETION-SEARCH-2026-09-08`；
- 真实 consumer：`io_tls` 的持久单 worker TLS server/client。冻结的自然路线对 Asio SSL 的
  W1-WSL P1 中，verified handshake 两顺序仅为 peer 的 `0.448/0.446`，512x256 B encrypted
  stream 两顺序仅为 `0.0265/0.0259`，5% 非劣门全部失败；A/A、known delta、timer floor、
  artifact/build/product provenance 与正反序均通过；
- qualification bundle：WSL
  `/root/hyle-io-tls-p1-evidence-d53cdadf`；`summary.json` SHA-256
  `B49F7CEE77EF0EC812D21E036D92710FFA52089C07E6C9CB7578C37987819197`，`manifest.json`
  SHA-256 `2C62BC15DA17A8F5501C2CF93BF60AD087278D07988FBD623566ADCE53EB10AD`；
- mechanism reproduction：同一 HYLE TLS/TCP/allocator/OpenSSL 路线只把 scheduler idle policy
  从默认 `kBalanced` 改为公开的 `kPowerSaving`，handshake 从约 `1.94 ms` 降至 `0.783 ms`
  （Asio `0.768 ms`），encrypted-stream episode 从约 `165.4 ms` 降至 `3.94 ms`
  （Asio `4.26 ms`）。这不是 provider 或 TLS API 差异。

## Closure card（`AUDITED`，已冻结）

- scope：只修正 scheduler 已附着 IO driver 且该 worker 有 in-flight IO、run queue 已空时，默认
  idle search 在不 poll completion 的情况下先执行 4096 relax + 4 yield 所造成的 completion latency；
  验证 interruptible proactor wait、普通无 IO scheduler 路线、三平台 IO correctness 与 TLS consumer；
- non-goals：Sched 公共 API、队列/fiber/synchronization/cancellation 重构，idle policy 参数重新调优，
  TLS/provider 特化，多 worker IO 架构重写，公共性能仪器修改，Compute/Graph；
- direct dependencies / consumers：Sched 不新增依赖；`io_core` 的 private integration hook 和
  `io_tls` 是直接回放消费者；
- expected complexity/resource contract：无 IO hook 或无 in-flight IO 时机器路径不变；有 in-flight IO
  且无 runnable work 时，不进行与 completion 无关的固定 idle search，直接使用现有可被 foreign
  scheduler wake 打断的 proactor wait；不新增 allocation、常态 metadata、lock 或 syscall 类型；
- tier：`P1`，5% material margin；W1-WSL 的 TLS natural-route qualification 是触发与终端 consumer
  gate，Sched 自身既有无 IO P1 只做定向回归，不借此声明全操作 SOTA；
- machine / instruments：`W1-WIN`、`W1-WSL`、`M1-MAC`；`closure-correctness-v1`、
  `closure-perf-v1`；
- budget：产品机制纠正一次；机制诊断已经完成，修复与 correctness/delivery 最多 90 分钟，正式
  P1 只在新 immutable candidates 后运行一次；
- stop：若取消 idle search 后出现丢 wake、worker 无法被 foreign task 打断、三平台回归或 TLS
  两个 workload 仍稳定超出 5% margin，则明确 `BLOCKED`，不继续调 spin 常数。

## Capability / gap ledger（进入 `AUDITED` 前冻结）

| ID | demand | current mechanism | gap | action | verification |
|---|---|---|---|---|---|
| `SCHED5-F01` | low-concurrency async server completion latency | drained run queue 先 `poll(0)`，随后 Balanced search，最后才 blocking poll | SQE 已提交但稍后完成时，搜索阶段完全不收 CQ，给每次依赖链增加固定约 160 us | `FIX` candidate | deterministic integration/correctness + Linux natural TLS P1 |
| `SCHED5-F02` | foreign task/timer/shutdown 可打断 IO wait | existing `IoWakeFn`/ring wake token | 修复不得把 worker 困在 proactor | `KEEP` and replay | IO/scheduler shutdown、external wake、timer tests |
| `SCHED5-F03` | ordinary CPU scheduler throughput | existing searcher/anti-herd path | 无 IO 时不应改变 | `KEEP` | Sched focused tests and existing P1 holdout |

本轮没有公开 API 或功能超集扩张。若审计确认 `poll(0)` 的返回值已经可靠表达 pre-reap in-flight，
`FIX` 集只允许在 worker idle transition 中据此跳过无意义 search；不得添加 TLS benchmark flag 到产品。

审计已确认：`IoContext::poll()` 在 reap 前读取 `Proactor::in_flight()` 并把该值返回 scheduler；各平台
backend 都由现有 ring/port/kqueue wake hook 打断 blocking reap。run queue 在 non-blocking reap 后仍为空且
返回 true 时，继续 CPU search 不会执行用户工作、也不会收割 completion；直接进入既有 sleeper 注册、
二次 queue 检查和 interruptible proactor wait 不改变 ownership 或 wake 协议。因此本轮冻结产品改动仅为
`SCHED5-F01`；`F02/F03` 只回归，不得顺手调参或重写队列。

## Current decision

`AUDITED / ACTIVE`。有限 `FIX` 集已经冻结，允许实现 `SCHED5-F01`；公开 API、依赖图和普通无 IO
scheduler 路线必须保持不变。
