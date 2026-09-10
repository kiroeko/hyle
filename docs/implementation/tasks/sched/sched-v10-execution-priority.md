# Sched v10：三档非抢占 execution priority

<!-- MODULE CLOSURE RECORD: SCOPED -->

## Closure card

- component / owner / kind：`sched` / Sched / `RESUME`；round=`sched-v10-execution-priority`；
- baseline：`4845e715099802ad59e498fed9311e65bce90efe`；前轮 terminal record 是
  [`sched-v9-task-scope-cost-audit`](sched-v9-task-scope-cost-audit.md)；
- trigger 与证据：[基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)已将 `G08`
  从 WATCH 提升为 low/normal/high opt-in primitive。Folly CPUThreadPoolExecutor 的多队列、oneTBB arena 与
  Non-Preemptive Priorities pattern 是两个独立成熟 peer；vLLM/SGLang 的 request priority 只证明推理 owner
  需要把应用策略映射到窄执行提示，不下沉其整数、tenant、KV 或 preemption policy；
- 正常 consumer：同一推理/渲染 runtime 中，owner 把交互控制/首 token 工作映射为 high，普通执行映射为
  normal，后台预取/清理映射为 low；runnable 选择优先处理更高 lane，已经运行的工作不被抢占；
- 主产品纵切：显式启用 priority 的四 worker runtime 接收三档可挂起 task；priority 在 cooperative yield、
  timer/IO/foreign-thread wake 后保持，high parent 的 child 默认继承，也可在 admission 时显式覆盖；
- 正交反偏置纵切：默认 Scheduler 的 all-normal schedule/spawn/wake/steal/pinned-IO 旅程保持既有 API、layout
  和热路径；一 worker deterministic backlog 验证选择顺序及“运行中不抢占”，防止吞吐 benchmark 把语义
  偷换成普通 FIFO；
- 跨模块 public journey：应用 owner 选择三档 execution hint → Sched admission/继承 → Fiber yield 或 IO/Timer
  保存和恢复 lane → JoinHandle/TaskScope 完整 drain。应用 request 排序仍归应用；OS thread priority 仍归
  PlatformThread；
- scope：审计并实现固定三档非抢占 runnable selection；覆盖 raw/owning `schedule`、`spawn_detached`、typed
  `schedule_task`/`spawn_task`、TaskScope add/spawn、pinned task 及 Waker 恢复；提供当前 execution priority
  的只读查询。先冻结独立 opt-in owner 或其它能严格隔离默认成本的架构，再写产品；
- non-goals：deadline/EDF、抢占、aging/share/tenant fairness、任意整数 priority、request/KV policy、OS thread
  priority、blocking pool priority、跨 Scheduler migration、Compute/Graph、G03--G06 或修改公共性能仪器；
- 直接依赖/消费者：直接依赖仍为已关闭的 Core/Error/Container-intrusive/Util-core/Time/Timer-queue/Mem/
  Platform；直接消费者为 IO 与未来 inference/rendering owner。本轮不得令任何已关闭下层反向依赖 Sched；
- 性能：`P1`，material margin `5%`。主工程门是默认 all-normal candidate 对 exact baseline 的同合同
  schedule/spawn/wake 组合非劣；priority-enabled 路线只与相同语义/完成边界的三 lane peer 或自身 normal
  control 比较，无法等价时降为 P0 成本账单而不制造输赢；
- 调查与纠正预算：一次 Folly/oneTBB 源码与 API 时序审计，一次 HYLE admission/yield/wake/pinned/steal
  全路径账单，一次候选架构裁决；AUDITED 后最多两个有限产品修正批次。计时前只允许一次 codegen/layout
  检查和一次短 A/A/双顺序 preflight，candidate 后只运行一次 qualified P1；
- 预声明验证：deterministic strict-order/non-preemption、yield/wake inheritance、explicit override、pinned/
  TaskScope/cancel/shutdown/full-queue/OOM；默认路径对象/layout/codegen/P0；受影响完整 Sched、三平台、
  ASan/UBSan/TSan、逐头、source/static/shared、IO direct consumer、候选后红队和 qualified P1；
- stop / blocker：若不能在不增加默认 Scheduler queue、常态 allocation/metadata 或 material 热路成本的前提
  下实现完整 yield/wake/inheritance 语义，则不得弱化为只给初次 enqueue 排序；退回独立 opt-in owner/factor
  设计。两次修正仍有同合同 material 默认回归、或 peer/consumer 不能支持冻结语义时，诚实 `BLOCKED`，
  不扩样、不换 workload、不为 benchmark 添加 fast/light API。

## SCOPED 不变量

1. 三档是 execution hint，不是应用请求值；higher-ready-first、非抢占和可能饿死 low 都是公开合同。
2. yield、wait/wake 和 child inheritance 是能力本体；只在首次 enqueue 排序不算完成。
3. 默认 all-normal 用户不能为未启用能力分配额外 lane，不能被迫携带 per-task 新 allocation 或常态
   priority context；是否能做到零新增 branch/layout 必须由 codegen/对象账单证明，不能靠点值猜测。
4. Folly/oneTBB 的多队列只证明形状，不自动证明 HYLE 应复制其整数 priority、thread pool 或 arena policy。
5. 在能力/API/成本账单进入 `AUDITED` 前不修改产品代码，不构造正式 benchmark。

下一步只做 peer/source、HYLE 全路径和 API 草图审计，形成有限 `KEEP/FIX/BORROW/REJECT` 集。
