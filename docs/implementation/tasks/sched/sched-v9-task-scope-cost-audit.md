# Sched v9：TaskScope 源码成本审计

<!-- MODULE CLOSURE RECORD: SCOPED -->

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
