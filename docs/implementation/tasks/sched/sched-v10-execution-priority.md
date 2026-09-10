# Sched v10：三档非抢占 execution priority

<!-- MODULE CLOSURE RECORD: AUDITED -->

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
- 性能：`P1`，material margin `5%`。默认 all-normal candidate 复用已资格化形状的 `sched-p1-v2`，在相同
  consumer contract 下与 Marl 比较 schedule 与 completion-wake；exact baseline 只承担 layout/codegen 的
  P0 差分，避免为了版本 A/B 人造第二套跨进程仪器。priority-enabled 路线只与相同语义/完成边界的三
  lane peer 比较；没有这种 peer 时降为 P0 成本账单而不拿较弱 normal-only 路线制造输赢；
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

## 能力、peer 与路径审计（`AUDITED`）

审计绑定本地受控参考：Folly `65749da412d9`、oneTBB `e00c68dcac0e`、vLLM `51da0ca66c80`、
SGLang `dcebe8c4733a`。Folly `Executor::addWithPriority` 以 0 为 normal，
`PriorityLifoSemMPMCQueue::nonBlockingTake` 从最高 lane 向下扫描；其文档明确线程本身不提权、运行中的低档
工作不会被抢占。oneTBB `task_arena::priority::{low,normal,high}` 及 Non-Preemptive Priorities pattern 同样
使用每档队列并在每个 work boundary 选择最高 ready lane。两者都不承诺低档免饥饿。

vLLM `v1/core/sched/request_queue.py::PriorityRequestQueue` 按 `(priority, arrival)` 排请求；SGLang
`srt/managers/schedule_policy.py` 还允许数值方向、retraction 和 preemption policy。它们证明上层需要映射
execution hint，但这些应用字段没有共同底层合同，保持 `OUT_OF_SCOPE`。

| 能力族 | 层级 | 自然语义 / owner | 当前路线 | 状态 | 动作 |
|---|---|---|---|---|---|
| low/normal/high vocabulary | 原子机制 | 三个固定 enum 值；应用 owner 负责映射 | 无 | MISSING | FIX |
| opt-in lane owner | 原子机制 | normal-only 默认；显式 three-lane Scheduler 承担额外 storage/scan | 单 lane | MISSING | FIX |
| admission 与显式 override | 原子机制 | 无 override 时只继承同一 Scheduler 当前 task；跨 Scheduler/外部线程为 normal | 单 lane | MISSING | FIX |
| yield / timer / IO wake 保持 | 原子机制 | priority 属 Fiber execution identity，不属于一次 enqueue | 无 | MISSING | FIX |
| structured child 继承 | 复合功能 | spawn/schedule 默认继承；TaskScope 可设 scope default 和 per-child override | 无 | MISSING | FIX |
| pinned + priority | 复合功能 | placement 与 lane 正交；wake 仍回 owner worker | 只有 normal pinned | PARTIAL | FIX |
| 当前 priority 查询 | 原子机制 | task 内返回当前档；runtime 外和 normal-only 返回 normal | 无 | MISSING | FIX |
| deadline/share/aging/preemption | 应用策略 | 独立 Scheduler/OS thread 才提供硬隔离 | 无 | OUT_OF_SCOPE | REJECT |

### HYLE 全路径账单

| 路径 | priority 必须保存/选择的位置 | 默认隔离裁决 |
|---|---|---|
| raw/owned `schedule`、`schedule_task` | inline item 入 lane；执行回调期间发布 TLS current priority | normal item/queue 形状不变 |
| raw Fiber、`spawn_detached`、`spawn_task` | admission 解析 inherit/override；Fiber 记录 route | 填入 `owner_worker_` 前既有 padding byte，不增 Fiber size |
| `Fiber::yield` | dispatch 后按 Fiber route 回原 lane | normal worker loop 走原 reschedule |
| Waker/Timer/IO | rendezvous 后按同一 Fiber route re-inject | 统一 Waker 必须做一次 sidecar-null mode gate |
| pinned | 每 worker 的 high/low pinned queue；只由 owner pop | sidecar 只在 three-lane mode 分配 |
| steal/global | high shared lane → normal local/global/steal → low shared lane | normal loop不扫描 priority lane |
| shutdown/drain/full | 既有 lifecycle/accounting 覆盖全部 lane；worker 递归满队列仍失败而不自锁 | 不复制 lifecycle authority |

逐任务新 allocation、第二 completion/cancel state、任意整数比较或 heap 都没有必要。three-lane sidecar 使用两个
固定档的 allocator-backed MPMC ready queues、每 worker high/low pinned queues 和现有 intrusive Fiber
overflow；每次选择只做固定档位和单队列 O(1) probe，不扫描 worker shards，完整 drain 仍为 O(N)。priority 模式在持续
normal/low backlog 下定期做非阻塞 IO reap，使已经完成的 high IO 有机会进入 ready lane；默认 loop 保持
既有 drain-time reap。

### 架构裁决

以下路线被拒绝：始终分配三档队列（默认用户付费）；单 FIFO 加 tag（不能越过队头 low）；每项 heap
（三档不需要 O(log N)）；只排序首次 enqueue（yield/wake 后失真）；复制 `PriorityScheduler + PriorityTask +
PriorityWaker + PriorityIO` 类型宇宙（破坏统一 completion spine，并形成第二 runtime）。

采用一个 Scheduler 的显式 `SchedulerPriorityMode::kThreeLanes` 与可空 sidecar。worker 启动时只选择一次
normal loop 或 priority loop，normal loop 没有 lane scan；公开 `Scheduler` 仍是一指针 pimpl，Fiber 的
priority 填入 `owner_worker_` 前既有 padding byte。统一 admission 和 Waker 边界必须各做一次可预测的 sidecar-null 判断，否则无法同时实现隐式
继承、IO wake 保持和一个公共 Waker；这是两处必要 routing seam，不伪称字面零指令。默认模式仍不得增加
队列、allocation、per-task size 或 run-loop scan，并由 exact-baseline codegen/size 与既有同合同 Sched
`5%` P1 非劣门共同裁决。

ready-order 合同是“每次 worker 选择下一项时，选择当时可观察到的最高 ready lane”。若 low 已开始运行后
high 才到达，或 high 与 dequeue 并发，它不被抢占；持续 high 可以无限推迟 low。用户需要硬隔离时创建独立
Scheduler，不用 priority hint 冒充 QoS。

### API / usability freeze

- `ExecutionPriority::{kLow,kNormal,kHigh}`；`SchedulerPriorityMode::{kNormalOnly,kThreeLanes}`；
  `SchedulerOptions::priority_mode` 默认 normal-only；
- 既有无 priority 参数的 `spawn/schedule/spawn_detached` 与 free `schedule_task/spawn_task` 在同一已启用
  Scheduler task 内继承，否则为 normal；显式路线使用聚合 `ExecutionOptions` 或 `FiberLaunchOptions`，让
  `scheduler.schedule(fn, ctx, {.priority = kHigh})` 和
  `spawn_task(scheduler, alloc, task(), {.priority = kHigh})` 可读；
- `current_execution_priority()` 只读且 noexcept；normal-only/outside runtime 返回 normal。
  `execution_priorities_enabled()` 让 owner 在接收应用配置时预检；
- 显式 low/high 交给 normal-only Scheduler 返回 typed `InvalidArgument`；失败不转移 raw Fiber/Task ownership。
  normal、inherit、closed、full、OOM 的既有 ownership 规则不变；
- `TaskScopeOptions::priority` 提供 scope default；每个 `add/spawn` 的 `FiberLaunchOptions::priority` 优先于
  scope default，二者都 absent 才继承 admission caller。scope 仍只有一个 cancellation/error/lifetime owner；
- priority 与 pinned 正交；explicit worker placement 不改变 lane，explicit lane 不解除 placement；
- 用户 guide 必须说明非抢占、饥饿、跨 Scheduler 不继承、IO completion visibility 与独立 Scheduler 的硬隔离。

### 有限改动集

| ID | 已证实问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| S10-01 | 无公开 vocabulary/options/继承 route | 增加上述 enum/options/query/overload；在现有 Fiber route word 与 priority-mode TLS 保存档位 | API compile、ownership/OOM、对象 size/codegen |
| S10-02 | 单 ready class 不能优先选择且 wake/yield 会丢档 | opt-in high/low sidecar、priority worker loop、pinned lanes、固定 lane scan 与 bounded IO reap | deterministic order、non-preemption、yield/wake/pinned/full/shutdown |
| S10-03 | TaskScope 和 typed helpers 无自然组合 | 统一 options resolution，scope default/per-child override，并补架构/user/build/delivery | inheritance/override/cancel/direct IO、三交付与 docs |

没有第四个产品改动集。先实现 S10-01/02 为一个 engine 批次，再实现 S10-03/文档为 integration 批次；之后
只做 correctness/P0 和一次短 preflight，不能边跑数边改变 lane、queue、inheritance 或默认值。

## AUDITED 实施与源码优先性能预审

两批有限产品修正已经完成，未增加第四批：

- `ExecutionPriority`、`ExecutionOptions`、`FiberLaunchOptions` 和显式 `SchedulerPriorityMode::kThreeLanes`
  覆盖 raw/owning callback、Fiber、typed task、pinned 与 TaskScope；absent priority 只在同一 Scheduler
  execution 中继承，跨 Scheduler/外部线程回到 normal；
- opt-in sidecar 使用 high/low 各一条 allocator-backed bounded MPMC queue，以及每 worker high/low pinned
  lane；正常 Fiber 既有 padding byte 保存 priority，没有增加 `Fiber` size；yield、Waker、timer、pinned、
  shutdown 和 cancellation 都回到同一 lane；
- `Scheduler::Impl` 构造把 raw sidecar allocation 移到所有可能抛出的 vector 建立之后，关闭构造异常时的
  隐藏泄漏；确定性 allocator seam 会让恰好第一个 priority-only allocation 失败并核对 allocation/free；
- 初版 high/low 曾错误使用 `ShardedQueue`，导致每次选择 normal 前的空 high probe 为 O(workers)。源码
  成本审计在计时前发现并改为每档单一 MPMC queue；现在 admission/selection 为固定 lane、常态 O(1)，
  不扫描 worker shard。该修正来自复杂度根因，不来自 benchmark 点值；
- three-lane mode 不能沿用 normal loop 的 8-callback batch：严格合同要求每个 runnable 完成边界重新观察
  high lane。这个 opt-in 固定 probe 是功能的必要成本；normal-only worker loop 仍保留原 batch 和队列。

exact `4845e715` MSVC Release P0 差分确认：`Fiber` 仍为 `136/8`，`Scheduler` 为 `8/8`，
`SchedulerOptions` 为 `56/8`；`spawn_normal`、`reinject_normal` 和 `schedule_with_hint` 的主体指令数与
baseline 分别保持 `167`、`183`、`121`，normal worker 只在进入旧 loop 前增加一次模式选择。公开 raw
admission 与统一 Waker 各保留一次 sidecar-null route seam；这是继承/恢复语义所需，未伪称字面零指令。

候选前唯一 `n=1` workload-matched preflight 使用 `sched-p1-v1` 原形状：A/A 双顺序 ratio 为
`0.9953/0.9816`，低于 5% bias；两个 comparable consumer rows 中 HYLE schedule 为 Marl 的
`1.3219/1.5908x`，completion-wake 为 `1.1289/1.0164x`。typed per-item ownership 仍按既有合同列为
`NOT_COMPARABLE`，不制造输赢。预飞的机械 FAIL 来自 `n=1` 不足以签统计 verdict，以及 IO component
路径迁移造成的 provenance hash 漂移；后者已独立审计为不改变实现/合同，并由只更新 provenance 的
[`provenance renewal`](../../evidence/performance/provenance-reviews/sched-v1-consumer-shapes-io-core-rename.md)
和 `sched-p1-v2` 承接终局 P1。不得为这些证据状态改产品或追加 diagnostic 样本。

## 第一候选撤回（`CANDIDATE -> AUDITED`）

首候选 `66dae758927a3d0f84cf11a716aa10ef50c592b1` 在 Windows MSVC Release/ASan、Linux GCC 14
Release 和 macOS AppleClang 21 Release 的 Sched/header correctness 均通过；Windows source、relocated
static 与 relocated shared consumer 也全部可执行。macOS 编译同时给出
`-Wtautological-constant-out-of-range-compare`：`PriorityState` 把 32 位 `unsigned workers` 与 64 位
`size_t` 上限直接比较，在当前目标上保护恒假，并会使预声明的 `HYLE_WERROR=ON` 交付失败。

这是源码与类型宽度已经充分解释的 portability 缺陷，不是性能回归，也不允许靠重复构建或采样寻找
PASS。状态依法退回 `AUDITED`；只允许在既有 `S10-02` 内先把 worker count 提升到 `size_t`，再用同宽
checked-size 计算得到 allocation bytes，并让 allocate/deallocate 共用该值。公开 API、lane 算法、对象
布局、默认成本、workload 和性能合同均不得改变。修正后签发新 candidate，只重跑受影响的编译/
correctness/交付/sanitizer；候选前 preflight 不重跑，正式 P1 仍保持唯一一次。
