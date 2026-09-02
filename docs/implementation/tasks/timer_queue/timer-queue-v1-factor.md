# Timer queue v1：Time 产品的 node/data-structure 因子

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round / kind：`timer_queue` / Time / `timer-queue-v1-factor` / `NEW`；
- baseline：`f342f8d92cebaae9f4870fe05e78afe6c6ca1391`；进入原因：已接受
  `B-F06 SPLIT_PRIME` 把 caller-owned timer node/data structure 与 host clock artifact 分开；
- 真实 consumer / normal workload：`src/sched/scheduler.cc` 每 worker 一个 queue，stable scattered nodes，
  mixed deadlines，大量 arm/cancel，park 前 advance 并 drain callback/data；IO `TimedSlot` 嵌入
  `TimerEntry` 参与 I/O-vs-timeout 仲裁；
- 范围：`TimerEntry/TimerList`、exact-ns radix `TimerQueue`、one-duration
  `CommonTimeoutQueue`；将现有公开 header/test/bench 所有权迁到独立 header-only factor；
  用已冻结 Time provider 补齐不改 layout 的 clock-composition overload；
- 非目标：callback 执行、owning callable、fiber/thread wait、cross-thread command、periodic policy、
  cancellation propagation、total-order guarantee、为 benchmark 增加 index/heap metadata、Sched/IO 状态机重写、
  C/Native FFI、稳定 C++ ABI、仪器改造与 Formal certification；
- 直接依赖：`core`、`time`、`container_intrusive` 全是 public header/type 边，无第三方
  产品实现；直接消费者：Sched implementation/public forward-type route、IO public `TimedSlot`，
  以及 consumer-facing `hyle::time` facade；
- tier：`P1`；normal primary 继续是当前 scheduler N=64，45/64 cancel，64 个 advance tick，
  19 个 delivery/callback observation 的完整 lifecycle replay，对照 Seastar `timer_set`；
- 语义档位：arbitrary mixed deadline 用 `TimerQueue`；一个单调 owner 上的单一 duration
  用 `CommonTimeoutQueue`；需要全序/indexed update 时组合 Container heap；执行/wait/cross-thread
  使用 Sched/IO service；
- 常态成本：queue/node 零分配、零 lock/atomic/syscall；`TimerEntry` 只保留 hook/deadline/
  payload/callback/exact owner；radix arm/cancel/next O(1)，advance O(touched buckets + delivered/
  rebucketed)；common arm/cancel/next O(1)，advance O(delivered)；provider overload 不增对象 metadata；
- margin / budget：W1-WIN `closure-perf-v1` 对同合同 Seastar 5% non-inferiority；总调查/
  取证预算 2 小时/性能 45 分钟；异常后最多一次复现/反序、一次 codegen/profile、
  一个产品修改和一次重跑；
- machines / instruments：`W1-WIN`、`W1-WSL`、`M1-MAC`；engineering
  `closure-perf-v1`，Formal `PENDING/formal-release-v1`；
- strong peers：Seastar `991cf9a908482a3f11ca2fc34a1b3970a06d8622`
  `timer-set.hh/timer.hh`（arbitrary deadline）；libevent
  `d82464a277d0f42703702c4dfd9af6af38595a83` `event.c common_timeout`（single duration）；
- provenance：Scheduler/IO 为 `REAL_CONSUMER`，libevent serving shape 为
  `PROVISIONAL_ARCHETYPE`，旧 synthetic microbench 只作 `DIAGNOSTIC_SYNTHETIC`；
- correctness gates：normal、empty/zero/past、full 64-bit/radix boundary/overflow、wrong owner/invalid order、
  reschedule/cancel/cancel-all、queue-first/entry-first/list lifetime、million-operation oracle，ASan/UBSan；OOM
  对 caller-owned zero-allocation factor `NOT_APPLICABLE`，concurrent operation 因 single-owner 合同 `NOT_APPLICABLE`；
- delivery gates：factor header self-containment/macro composition，`hyle::time` product source/static/shared，
  developer `HYLE_DEV_FACTORS=timer_queue`，Sched/IO direct consumer，三平台，无伪 binary/FFI 反证；
- 预期输出：独立 header-only target/test/bench owner，真实直接边，不变的用户
  `hyle::time` facade，provider 组合，重签 current scheduler P1 证据与同步文档；
- 停止条件：完成审计冻结的有限 `FIX` 后裁决；不研究亚纳秒单操作排名，
  正常 consumer gate 无资格或未越过则 `BLOCKED`，不修公共仪器。

`SCOPED` 只冻结任务合同；在 capability/public-surface 反向追踪和有限 `FIX` 集完成前，
不授权产品修改。
