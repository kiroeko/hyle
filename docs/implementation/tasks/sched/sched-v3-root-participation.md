# Sched v3 根参与批执行恢复记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `sched-v3` 的 `RESUME` 施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；已经终止的
> [`sched-v2`](sched-v2-batch-structured.md) 只保存候选、验证和失败事实，不在本轮回写。

## Closure card（`AUDITED`）

- component / owner / round ID：`sched` / Sched execution owner / `sched-v3-root-participation`；
  task kind：`RESUME`。
- 前一 round / blocker：`sched-v2-batch-structured` / `SCHED-V2-PERF-ONETBB-01`。最终候选在资格有效、
  同 consumer 合同的 W1-WIN replay 上没有通过 oneTBB 的 5% non-inferiority gate；peer-first advantage
  为 `0.916`，99.26% CI 为 `[0.777, 0.937]`。
- 前一 resume condition：出现一个有复杂度依据、能在不改变已准入 consumer 合同的前提下消除 material
  gap，并且可以用有限 proof plan 验证的 execution/partitioning mechanism。
- 恢复事实：正式中位数折算后，HYLE 相对 oneTBB 每个 16-round replay 多约 `75.7--104.8 us`，即每个
  batch 固定多约 `4.7--6.6 us`；Marl 的 enqueue/wait 路线与 HYLE 基本持平。oneTBB
  `simple_partitioner` 对 grain=1 一直拆到不可分 range，双方仍执行 128 个用户 callback；共同的
  `tile_digest` 是逐步依赖的 rotate/multiply recurrence，没有跨元素 SIMD 路线。当前最强机制假设是：
  oneTBB 已进入 arena 的 root task 直接完成一份 range，而 HYLE root Fiber 先投递全部 runner、挂起、
  再被 terminal Event 唤醒，支付了额外批级 handoff。
- baseline commit：`3a2bbca61038f5343729bf2dad66596bffc687b3`；产品输入仍是 v2 候选
  `22ec2382589656d1e45ff5a0665e384aba97dc4c`，本状态不把它冒充 v3 candidate。
- 范围：审计并冻结“同一个 Scheduler 上已经运行的 root Fiber 参与其同步 `bulk_execute`”机制；必要时
  增加仅供 Sched 实现使用的 current-owner identity seam；保持既有 shared first-error/cancel、partial
  admission、固定 public grain、caller-owned lifetime、O(N) work、O(worker) runner 与 O(1) batch state。
- 非目标：外部调用线程参与计算；跨 Scheduler Fiber 借用错误 worker；公开 partitioner/auto-grain；
  修改 benchmark tile、fanout、顺序、CPU 或 peer；Graph/Compute/IO 迁移；重写完整 scheduler、Event、
  Fiber 或 work-stealing；为结果删除错误/取消语义；Native FFI/C API；P2/Formal 声明。
- 真实 consumer / normal workload：在 Scheduler Fiber 内同步 fan-out 一批独立、non-suspending、
  caller-owned 输出 tile；固定为 v2 的 16×128×1024 replay，四个 pinned provider workers，外部线程只
  enqueue/wait。
- 直接依赖 / 消费者：产品依赖保持 v2 的 Core/Error/Time/Platform/Mem/Util-core 与 private Container
  queue；IO/Compute/Graph 只运行既有代表 smoke，不取得模块收口 verdict。
- tier / margin / machine / instrument：`P1` / 5% / `W1-WIN` / `closure-perf-v1`；`W1-WSL` 与
  `M1-MAC` 只做 correctness、sanitizer、portable codegen 和 delivery，不池化性能样本。
- strong peer：oneTBB `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` 的 natural
  `task_arena + parallel_for(blocked_range, simple_partitioner)`；Marl 只作机制诊断，不以它抵消 oneTBB
  hard fail。
- 总预算：机制归因与 audit 最多 60 分钟；至多一次非裁决 diagnostic、一个产品机制修改、一次
  v2 normal workload 的 fresh n=15 qualification（只有统一停止规则允许才 n=31）。不得发明新的产品
  benchmark 姿势或因结果继续调 reservation/tile。
- correctness 风险：same-Scheduler identity、单 worker、零/一 runner、partial admission、first error、
  mid-cancel、root/runner terminal race、跨 Scheduler 调用、shutdown 与 stack-borrow quiescence。
- delivery 风险：产品若改变则重跑 Windows/WSL/macOS 的 source、relocated static/shared Sched component
  journey、public headers、symbol/旧 FFI 清洁度和既有直接 consumer slice。
- 停止条件：`AUDITED` 前必须把当前假设裁决为一个有限 `FIX` 或否决并重新 `BLOCKED`；实施后只允许
  一个 candidate 和一次正式重跑。所有适用门和 oneTBB 5% NI 通过才能 `CLOSED`；稳定 material FAIL
  必须以新的具名 blocker 终止，不得继续换姿势。

## SCOPED audit ledger

| ID | question | evidence required before `AUDITED` | current |
|---|---|---|---|
| `SCHED-V3-EQUIV-01` | oneTBB 是否减少了用户 callback 数或使用 SIMD | peer source、callback 边界与共同 body codegen | `PASS` |
| `SCHED-V3-COST-01` | 差距是每 item 还是每 batch 固定成本 | v2 immutable summary 的 throughput→wall-time 分解；至多一个 diagnostic | `PASS`：现有正式证据已给出 4.7--6.6 us/batch，不消费额外姿势 |
| `SCHED-V3-ROOT-01` | root participation 是否保持同合同并消除一次 handoff | same-owner 判定、状态图、复杂度与失败/取消/lifetime proof | `PASS` |
| `SCHED-V3-FIX-01` | 应冻结的唯一产品修改 | 上述三项完成后形成有限 capability/gap ledger | `FIX`：见下节 |

## Audit result 与冻结 `FIX`

- callback 等价：oneTBB `simple_partition_type::execute` 在 `range.is_divisible()` 时持续 split，
  `blocked_range::is_divisible()` 是 `grainsize < size`；因此 `[0,128), grain=1` 的自然路线执行 128 个
  unit body。HYLE 同样按 128 个 public grain 调用 body，不存在 oneTBB 合并 callback 的语义便宜。
- body codegen 等价：v2 candidate 与 oneTBB peer object 中的 `tile_digest` 指令逐项相同，都是同一
  106-byte 标量 recurrence（load、shift、rotate、multiply）；没有 vector instruction，也没有不同的
  digest 实现。SIMD 不是本差距来源。
- 成本分解：按 `2048e9 / median throughput` 还原 wall time，candidate-first 总差约 `75.7 us`，
  peer-first 总差约 `104.8 us`；除以 16 batch 是 `4.7--6.6 us/batch`。这比一个稳定目标的间接调用或
  x86 acquire load 的预期成本阶高，并与当前每 batch 的四次 runner admission、root suspend/resume、
  terminal wake 相吻合。Marl 的 enqueue/wait 自然路线与 HYLE 持平，是独立的机制交叉检查。
- ownership 状态图：
  1. external caller 或另一个 Scheduler 的 Fiber：仍投递全部 `concurrency` runner 并等待；不得借用非
     owner worker；
  2. 同一 Scheduler 的 Fiber：投递 `concurrency-1` runner，caller 在原 Fiber 上调用同一个
     `run_with_cancel_scope` 消费共享 cursor，然后只在其他 runner 尚未 quiescent 时等待；
  3. 同一 Scheduler 的 worker-stack callback：保留现有 serial fallback，不能在 worker stack 上阻塞；
  4. submission guard 在 caller 参与和投递期间保持 `active>=1`，因此异步 runner 不能提前 signal 或
     释放 caller-stack state；caller 完成本地 work 后才释放 guard，并沿用 v2 的 notifier lifetime
     handoff。

冻结的唯一产品修改是 `FIX-01`：增加一个 Sched-private current-owner identity seam；在
`bulk_execute` 中仅对“`Fiber::current()!=nullptr` 且 current owner 是参数 Scheduler”的路径启用上述
root participation。它不新增 public declaration、symbol、option、layout、依赖或常态 metadata；
runner/queue entry 从最多 `concurrency` 降为最多 `concurrency-1`，work/state 复杂度不变，单 worker
Fiber 路线退化为直接执行而非一次无用 schedule+suspend+wake。公开注释和 guide 必须同步说明：该
participant 在 caller Fiber 栈上执行 body；non-suspending 合同不变，但 `Fiber::current()` 与 stack budget
是可观察的，不能继续写成“所有 body 都在裸 worker stack”。

验证冻结为：新增同 owner Fiber 确实在原 Fiber 内执行至少一个 grain、跨 Scheduler Fiber 不参与、
单 worker/error/cancel/exactly-once/lifetime/zero-allocation regressions；三平台 full Sched + sanitizer、
三交付与直接 consumer slice；最后只重放与 v2 相同 workload、peer、顺序和 5% margin 的 W1-WIN P1。
由于 v2 provenance 仍描述旧候选“root 只等待、至多四个 runner”，本轮使用
`consumer_shapes_v3.json` 与 `sched-batch-p1-v3.json` 重新绑定候选身份和自然执行拓扑；这不是新增
benchmark 姿势。除非 `FIX-01` 暴露 correctness 缺陷，本轮不再加入第二个产品假设。

## `FIX-01` implementation diagnostic（pre-candidate）

首个实现只把 runner 数从 `concurrency` 降到 `concurrency-1` 并让 root 消费 cursor；Windows correctness
通过，但同形状 raw replay 出现明显启动长尾。代码机制检查发现，普通 `Scheduler::schedule` 从 worker
内 burst 提交时故意复用当前 worker 的 accounting shard，只有该 shard 的首个 outstanding callback
执行一次 notify；这个策略适合普通短 callback 的 locality/anti-herd，却不能保证有限 bulk fork 的三个
sibling 立即激活。因此 `FIX-01` 的完整实现加入 Sched-private `SchedulerAccess`：root 的 sibling runner
轮转到其他 accounting shard，每个空→非空 shard 才触发一次既有 notify；普通公开 `schedule` 路径完全
不变。这是 root-participating fork 的 admission 完整性，不新增第二个用户能力或 benchmark 路线。

修正后消费本轮唯一 mechanism diagnostic：沿用相同 16×128×1024 workload、CPU `[0,2,4,6]`、parent
mask `0x55`、AboveNormal，15 次进程交替顺序；HYLE 中位 `2.056M callback/s`，oneTBB 中位
`1.968M callback/s`，中位比约 `1.045`。该 quick diagnostic 有明显共同系统长尾，未运行统一 A/A、
known-delta、timer-floor、artifact/provenance validator，因此只允许进入 candidate，绝不是 P1 verdict
或 SOTA 声明。Windows Release 的全部 Sched tests/public-header tests 已通过，新 same-owner/cross-owner
测试各重复 100 次通过。

## Candidate

- product commit：`38a5ca048b796e5f7e30283fef5b9c0c4f4875d5`；tree：
  `cf2fe3c28a606c787875b5c25cce869bcd860b5b`；工作树在冻结时 clean。
- public surface：没有新增 declaration、symbol、option、layout 或依赖；`bulk_execute` 的 callable、
  first-error、cancel、grain、borrow 与同步 quiescence 合同不变。公开注释、architecture 和 guide 现在准确
  说明同 Scheduler root Fiber 可在自己的 Fiber 栈执行 body，另一个 Scheduler 的 Fiber 不会参与。
- implementation：新增未安装的 `src/sched/scheduler_access.h`，只给 `bulk.cc` 做 current-owner identity 与
  bounded sibling admission。普通 `schedule` 仍复用当前 accounting shard；bulk root 的至多
  `concurrency-1` 个 sibling 轮转到其他 shard，root 自己消费同一 reservation cursor。若 sibling 已在
  root 结束前 quiescent，acquire 读取 submission guard 后直接返回；否则沿用 v2 的 Event + terminal
  notifier lifetime handoff。
- complexity/resource：O(N) body work、O(worker) participant/queue entries、O(1) caller-stack batch state；
  无 per-item allocation/status/cancel/handle，无新 Scheduler object metadata。相对 v2 只减少一个 runner，
  并把有限 sibling 的既有 accounting 分散到最多 worker 数个 shard。
- candidate tests：W1-WIN MSVC 19.50 Release build 通过；全部 Sched tests 与 10 个 public header
  self-contained tests 通过；same-owner participation 与 cross-owner rejection 各重复 100 次通过。quick
  diagnostic 只授权冻结该 candidate，工程性能仍是 `PLANNED`。

`AUDITED -> CANDIDATE` 前态快照保存在 canonical checkout 外：
`C:\hyle-scratch\sched-v2\module-status-before-sched-v3-candidate.json`。

## 首个 candidate 的交付 finding 与退回

首个 candidate 的 Windows/WSL/macOS correctness、sanitizer、source/relocated shared/static、六条直接
consumer slice 均通过；新合同 `sched-batch-p1-v3` 的 W1-WIN qualification 与 performance gate 也均为
`PASS`。但是最终 symbol scan 发现：Windows DLL 没有导出 `SchedulerAccess`，ELF 与 Mach-O shared
library 却因默认 visibility 各导出了
`detail::SchedulerAccess::is_current_scheduler` 和 `schedule_bulk_sibling`。这不改变运行语义或已测性能，
但违反冻结的“private seam、无新增公开 symbol”交付合同，因此不能把首个 candidate 提升为 VERIFIED。

本轮按协议从 `CANDIDATE -> AUDITED`，冻结 `FIX-01-VIS`：只给这两个 Sched-private declaration 加
`HYLE_HIDDEN`，不改变函数体、路由、layout、依赖、公开 header 或 benchmark。新 candidate 必须重跑受
影响的三平台 build/delivery/symbol、correctness smoke、直接 consumer slice 与同一正式 P1；首个 P1
bundle 只作诊断事实，不继承为最终 verdict。退回前 registry 快照位于 canonical checkout 外：
`C:\hyle-scratch\sched-v3\module-status-before-hidden-symbol-audit.json`。
