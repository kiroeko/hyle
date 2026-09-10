# Sched v9：TaskScope 源码成本审计

<!-- MODULE CLOSURE RECORD: AUDITED -->

## Closure card

- component / owner / kind：`sched` / Sched / `RESUME`；round=`sched-v9-task-scope-cost-audit`；
- baseline：`e2008c1b`；前轮 terminal record 是
  [`sched-v8-task-scope-distributed-completion`](sched-v8-task-scope-distributed-completion.md)；
- trigger 与恢复证据：独立仪器任务
  [`performance-workload-interleaving-2026-09`](../project/performance-workload-interleaving-2026-09.md)
  保持完整 32-child owner、Scheduler/allocator/四 worker 拓扑和双 baseline 顺序，以 nanobench
  `Bench::compare()` 逐 round 交错。candidate/candidate 与 peer/peer 在 primary/placement 的 8/8
  workload-matched A/A 都位于 `[0.95, 1.05]`，满足 v8 唯一恢复条件；同一有界产品 A/B 仍出现
  `0.896..0.985` 的 candidate/peer signal，因此恢复产品轮，但该信号不是 qualified P1 verdict；
- 正常 consumer：推理 request、model load 或 rendering frame 动态产生 32 个会挂起一次的 helper，owner
  停止 admission、保留首错、等待 finish-hook/stack cleanup 并完整 drain；
- 主/反偏置纵切：沿用 v8 的 inference/rendering 32-child trace及资源加载反偏置纵切，不新增 synthetic
  轻语义任务；
- public journey：`TaskScope::create` → concurrent `add/spawn` → `close`/`request_cancel` →
  `join/cancel_and_join`，与 public manual `spawn_task` + handles + `Scheduler::drain` 在同一完成边界比较；
- scope：先形成两条路线逐阶段时序及每 owner/per-child 的 allocation、retained bytes、atomic、lock、wake、
  deallocation 和 out-of-line call 静态账单；只检查账单中未决项的 MSVC codegen。只有源码能定位正常消费
  路径上的 material 可消除成本，才冻结有限产品修正；
- non-goals：扩样或 qualified `n=15`、换 peer/workload/margin、修改公开 API/layout 以迎合点值、G08
  priority、Scheduler queue/idle/steal、IO、Compute/Graph、再次修改公共仪器；
- 直接依赖/消费者：既有 `sched` 边不变；直接消费者仍为 IO 与后续 G03/G04；
- 性能：`P1`，material margin `5%`。本阶段只做 P0 源码/生成代码审计，不计时；API、算法和 correctness
  固定成新 candidate 后，才允许用已通过 A/A 的 v2 合同运行唯一一次 qualified P1；
- 调查与纠正预算：一次静态成本表、一次生成代码检查；至多冻结两个由源码直接证明、保持语义和 API 的
  产品修正。AUDITED 前不得改产品；修正后只允许一次 focused correctness/P0 preflight 和一次 bounded
  A/B 判定是否有资格签 candidate；
- 预声明验证：若形成产品修正，运行受影响 TaskScope focused correctness/death tests、产品 diff/P0/codegen；
  candidate 后才运行 v8 已声明的三平台、sanitizer、三交付、红队和唯一 qualified P1。无产品修正则直接
  记录现有取舍并按性能门裁决；
- stop：静态账单证明差异来自不可比较语义、没有有限可消除机制，或两次修正预算耗尽仍有 material
  同合同信号，则 `BLOCKED`；不得以继续采样代替根因。全部适用 gate 通过才 `CLOSED`。

## SCOPED 不变量

1. 完整 owner operation、完成边界、peer identity、四 worker topology、allocator、32-child fanout、双顺序、
   placement 和 5% margin 不变。
2. 普通 `schedule/spawn_task` 路径不能承担 TaskScope 成本；TaskScope 不新增第二 allocation、常态 metadata
   或 benchmark-only fast/light API。
3. 已修复的正常析构重复 Event 扫描和 prepublication locked ref increment 不得恢复。
4. 先用源码回答“慢在哪里、换取了什么”；本卡在 `AUDITED` 前不写产品代码，在 `CANDIDATE` 前不启动
   qualified performance run。

下一步仅执行静态成本与 codegen 审计，并据此冻结有限 `FIX/KEEP` 集。

## 静态成本与生成代码审计（`AUDITED`）

两侧共同支付：每 owner 一棵 root cancellation tree、32 个 coroutine frame、32 次标准 tier stack
acquire/recycle、32 次 Scheduler admission、一次 cooperative yield/re-injection、Status 结果和最终
`Scheduler::drain()`。candidate 的 scope state allocation 与 peer 的 reserved PMR handle-array allocation
都是每 owner 一次；两者都在 admission、worker completion 和 owner observation 各进入一次短锁语义。
因此这些共同成本不能解释固定方向，也不授权换 workload。

| 每个 child 的常态账单 | TaskScope candidate | public manual peer | 裁决 |
|---|---|---|---|
| driver/result storage | 一个 `288 B` 同分配 driver + record | `184 B` driver + `120 B` JoinState | candidate 少一次 allocation，合计 bytes 也未更大 |
| lifetime release | publisher/scope 两次 refcount decrement | task/handle 两次 refcount decrement | 同阶、同为两个 RMW |
| admission lock | scope control lock + closed check + list link | manual admission lock + closed check | 同一用户任务的 O(1) control cost |
| terminal publish | per-record Event signal；罕见失败再聚合首错 | JoinState result signal | 都是 per-child record，不存在 candidate 共享成功 completion line |
| owner observation | 逐 record `Event::wait()` | 逐 handle `JoinState::take()` | 都是 O(N)，但 candidate 未利用 publisher 已释放这一可证状态 |
| teardown | scope 遍历 32 records、同 owner sized-free | driver/JoinState 分布释放、handle buffer 回收 | 生命周期分布不同，但 candidate 没有更多 allocation 或渐近成本 |

MSVC 19.50 Release object 进一步把未决成本缩小到两项。公开 `TaskScope::add(Task<Status>)` 的正常路径
先调用自身 wrapper，再调用同 TU 但未内联的 `task_scope_add<Status>`；manual route 每 child 只有一次
`spawn_task_impl` 调用。`TaskScope::wait_` 的循环则无条件调用 out-of-line `Event::wait()`，即使
`TaskScopeCompletion::references == 1` 已经由 publisher 的 release RMW 证明 Fiber、Task、stack、publish
tail 全部结束。该 acquire observation 可以直接复用现有 refcount，不增加 layout/metadata；观察到 2 时
仍走原 Event slow path，保持 wake-before-wait、多个 waiter 与 final-publisher race 语义。

### 有限改动集

| ID | 已证实问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `S9-01` | 每次 `add` 多一层同 TU、未内联 wrapper→template call | 强制内联内部 `task_scope_add` 到两个既有公开 overload；不改变公开 inline/ABI、校验或错误路径 | Release codegen 的公开 overload 不再调用内部 template；产品 API/layout 不变 |
| `S9-02` | `wait_` 对 publisher 已释放的 record 仍进入 Event lock/wait | 以现有 refcount 的 acquire load 提供 `publisher_released()` 快路径；值为 2 时保持原 `Event::wait()` | 无新字段；已完成 record 不调用 Event，未完成/final-publisher pause/multiple waiter 行为不变 |

没有第三项候选。尤其不拆 completion allocation、不改多 waiter 合同、不引入 shared counter、不把 driver
释放移到远端 worker，也不增加 batch/fast API。实现上述两项后只跑 focused correctness、生成代码和一次
bounded interleaved A/B；若仍有 material signal，按卡停止，不继续猜测。

## API / usability freeze

公开调用和语义档位完全沿用 v8：无逐 child 结果时使用 `add`，需要 typed result/独立取消时使用
`spawn`，最后显式 `join` 或 `cancel_and_join`。同步 admission failure 不转移 ownership；已 admission 的
失败由 scope 在 full drain 后聚合。`remaining()` 仍是 O(N) control-plane snapshot。两项修正只移除编译
边界和已完成观察的冗余常态成本，不产生 benchmark-only 入口或 delivery 差异。
