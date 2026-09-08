# Sched v5：IO completion 与空闲搜索协同

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

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

## Closure card（`CANDIDATE`，已冻结）

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

`CLOSED / FROZEN`。产品冻结于
`2f9ecc8168b4d303a17ae4fcad385b8150cdd072` / tree
`212a5c0305d9a756cdea91b2ed1270319264db8a`；公开 API、依赖图和普通无 IO scheduler 路线未变。

## Candidate checkpoint

- `SCHED5-F01`：drain-time `poll(0)` 的 pre-reap in-flight 返回值现在控制 idle transition；若其为 true
  且 completion 尚未产生 runnable work，worker 跳过不 poll IO 的 spin/yield search，直接进入既有
  sleeper publication、queue recheck 与 interruptible proactor wait；
- Windows MSVC 19.44 static 的 `hyle_sched_test`、`hyle_io_core_test`、`hyle_io_tls_test` 全过；
- WSL GCC 14 static 同三项全过；默认 Balanced TLS diagnostic 已从修复前约 `1.94/165.4 ms`
  降为约 `1.14/4.64 ms`，同次 Asio 为 `1.10/4.63 ms`。这些是机制 smoke，不替代正式 P1；
- candidate 后只允许验证、文档与 fixture 修正；任何产品变化必须退回 `AUDITED`。

## Candidate verification（`VERIFIED`）

- W1-WIN MSVC 19.44、W1-WSL GCC 14 与 M1-MAC AppleClang 21 的
  `hyle_sched_test`、`hyle_io_core_test`、`hyle_io_tls_test` 全部通过；
- WSL GCC 14 exact-factor 的 ASan、UBSan 与 TSan（`setarch x86_64 -R`）三组均通过；Windows installed
  `hyle::io` shared/static consumers 均通过；
- W1-WSL clean source `b38aa34cf75cd89b179f0ee32449bfeb4ff0cb59` 的正式
  `io-tls-p1-v1` qualification/performance gate、A/A、5%/10% known delta、timer floor、artifact/build/
  consumer provenance 与稳定性全部 `PASS`。handshake 双顺序 HYLE/Asio advantage 为
  `1.081/1.100`，encrypted stream 为 `1.069/1.075`，四项 5% NI 均通过；superiority 因置信区间跨
  practical margin 均为 `INCONCLUSIVE`；
- reopen 时 registry 的 `engineering_performance.machine_profile` 误沿用 v4 的 `W1-WIN`，但本卡从
  `REOPENED` 起冻结的终端 consumer gate 与 `io-tls-p1-v1` 合同始终明确为 `W1-WSL`；进入
  `VERIFIED` 时把投影纠正为实际冻结机器，不更换采样结果、阈值或测试合同；
- 原始 bundle `/root/hyle-io-tls-p1-evidence-b38aa34c`；manifest/summary/controls SHA-256 依次为
  `6D0049A792B83630B489EB59EB78B886AF731B2209BAB6022DB94834D2E341F9`、
  `BD4FDB499490DC080415C0DEDB781BDEDFEDAC1AB990C187CE7FF1ABE4A03143`、
  `49FE7239FA181D595013CE12BD78AEF811CB545B6522E34BCCCE654776E1BD49`；
- candidate 后红队复核了 searcher 计数配对、sleeper publication/recheck、foreign/pinned wake、timer
  bound、shutdown、IO detach 和 completion retirement。IO-in-flight 路线从未增加
  `num_searching_`，因此不会错误 `fetch_sub`；它注册 sleeper 后再查队列，proactor wake 可打断
  blocking wait。无 IO 路线仍成对 increment/decrement，未发现丢 wake、underflow、busy poll、API
  污染、TLS 特化或新增常态成本。

| gate | verdict | basis |
|---|---|---|
| scope / architecture | `PASS` | 唯一冻结 FIX 完成，无 API/依赖/idle-policy 参数扩张 |
| correctness | `PASS` | 三平台定向 suites + WSL ASan/UBSan/TSan |
| delivery | `PASS` | 直接 source journeys + Windows installed shared/static consumer |
| usability / docs | `PASS` | 无公开形状变化；触发、取舍、限制与自然 consumer 已记录 |
| reuse / red team | `PASS` | 复用既有 poll/wake/sleeper 协议，无第二套 IO 或等待 authority |
| engineering performance | `PASS` | qualified 双顺序 natural TLS consumer 四项 5% NI PASS |

## Terminal receipt（`CLOSED / FROZEN`）

- freeze ID：`sched-v5-freeze-2026-09-08`；完整复核点
  `c1b2a5b775fcc8100281bf3acc3117d09b764987` / tree
  `7bb2e83a57fa740d84625c1a9539d63e10ee351d`；产品 candidate/tree 保持
  `2f9ecc8168b4d303a17ae4fcad385b8150cdd072` /
  `212a5c0305d9a756cdea91b2ed1270319264db8a`；
- 不可变 candidate LOC：全库代码 `491234`；自实现不含测试 `83216`；Sched 实现 `5653`；
- 工程门：scope、architecture、correctness、delivery、usability/docs、reuse/red-team 与 P1 全部
  `PASS`；Formal 保持 `PENDING`；
- 限制：本轮数值只覆盖 single-worker natural TLS completion；普通无 IO batch 路线沿用 v4 证据，
  不声明全操作/跨平台 SOTA。IO integration 仍要求唯一 attached driver 正确报告 pre-reap in-flight，
  blocking wait 可被 scheduler wake 打断；取消仍由 operation owner 排空，stackful Fiber 不支持强杀，
  gang/SPMD、连续批处理和 GPU 调度仍属于上层；
- reopen：只有可复现的调度/同步/取消/ownership/自然 API 缺陷，IO hook 的 lost-wake、hang、busy-poll
  或错误 in-flight 合同，公开 API/layout/算法/依赖/交付/正常成本变化，支持平台/toolchain/fcontext
  语义变化，或合格同合同证据证明 material regression/更优复杂度，才能重新开启。

本轮到此终止；后续 IO 因子不得把历史命令式文字当成继续修改 Sched 的授权。
