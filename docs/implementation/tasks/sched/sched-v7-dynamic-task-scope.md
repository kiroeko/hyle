# Sched v7：动态 structured task scope

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round：`sched` / Sched / `sched-v7-dynamic-task-scope`；kind=`REOPEN`；
- baseline commit：`37b6246010fad38f36abcc5fed0cf3a5cc88ba2f`；前轮 terminal record 为
  [`sched-v6-pinned-structured-task`](sched-v6-pinned-structured-task.md)；
- trigger 与证据：已接受的基础能力扩展决策
  [`G02`](../../../architecture/foundation-capability-expansion.md#g02动态-structured-task-scope) 证明现有
  `JoinHandle` 销毁会 detach，`when_all/when_any` 只观察一个预先固定的 handle 集，`WaitGroup` 只提供裸计数；
  三者都不能拥有运行期增长的 child 集、关闭准入、聚合失败并在取消后完整 drain；
- 正常 consumer 任务：一个推理请求、连接、模型加载或渲染 frame 在运行期间按输入动态产生任意数量 helper
  child，owner 在停止 admission 后等待全部 child terminal，并可协作式取消整个 scope；
- 主产品纵切 / 具名来源：vLLM/SGLang 请求与 helper 生命周期、llama.cpp/ggml backend 工作单元以及 ds4
  推理 pipeline 的动态 fan-out；Folly `AsyncScope/CancellableAsyncScope` 是主要公开形状证据；
- 正交反偏置纵切：资源加载或渲染 frame 在发现依赖时动态 spawn decode/upload helper，并在 frame owner
  退出前显式 drain；它防止 API 被单一 HTTP request 或固定 batch benchmark 特化；
- 跨模块 public journey 与 owner：`sched` 拥有 task admission、Scheduler 绑定、completion 计数、scope
  cancellation 与 first-error 聚合；Error 承载结果，Mem 提供显式 allocator，Time/IO 只通过现有 suspend/wake
  组合；上层 request、route、model 或 frame policy 不进入本轮；
- scope：runtime-sized child admission；并发 spawn/close 线性化；显式 close、join 与 cancel-and-join；绑定一个
  Scheduler；`Task<void>` 与 `Task<Status>` 首错聚合；其它 typed result 继续由 `JoinHandle<T>` 拥有；析构
  live-child fail-fast；source/static/shared C++ 交付；
- non-goals：G08 priority/QoS/deadline、自动 deadline、preemption/强杀、跨 Scheduler 或外部 runtime task、
  detached 析构、隐式全局 allocator、task graph/DAG、Compute/Graph、C/Native FFI、普通 spawn/schedule
  metadata、调度队列/idle policy 或公共性能仪器重写；
- 直接依赖 / 直接消费者：保持现有 `sched` component edge；只复用 Core/Error/Container-intrusive/Util-core/
  Time/Platform/Mem/Mem-vm 与现有 Task/JoinHandle/CancelToken/Scheduler。直接消费者是后续 IO/WebSocket/process
  owner 和未来推理/渲染 orchestration；本轮不修改这些模块；
- 性能 tier 与原因：`P1`。P0 必须证明普通 `spawn_task/schedule` 的对象 layout、分配、原子、锁、分支和
  queue path 不变，只有 scoped child 支付 tracking/cancel/error 成本。P1 只比较相同动态 fan-out、错误、
  close/join ownership 的 HYLE 手写组合或成熟 peer；不得拿 detached、静态 handle 数组或不聚合错误的轻语义
  路线排名；material margin 5%；
- 适用机器 / 仪器：W1-WIN、W1-WSL、M1-MAC correctness/delivery；`closure-correctness-v1`；若 AUDITED
  后形成同合同 episode，正式 P1 使用 `closure-perf-v1` 的 W1-WIN，双顺序/placement、A/A 与 known-delta；
- 调查与纠正预算：一次 API/peer/现有生命周期审计、一个产品实现批次、最多两次 correctness/red-team
  产品纠正、一次性能机制归因和一次正式候选 P1；不为点值改公共 API、普通 task layout 或 queue policy；
- 预声明验证：public-header self-containment；0/1/动态 N child；void/Status/typed/move-only result；并发
  spawn-vs-close；closed admission；join/cancel-and-join 从 OS thread 与 Fiber；first error + full drain；
  child 自己继续 spawn；不同 Scheduler 不可注入；invalid allocator、coroutine/state/driver/stack/scope OOM；
  queue full、Scheduler shutdown、join 后复用/重复调用、live-child destructor death；W1-WIN/W1-WSL/M1-MAC；
  ASan/UBSan/TSan；Sched focused 与受影响 IO smoke；source/static/shared relocated consumer；P0/P1、终局红队；
- stop / blocker：若在不改变普通 task 成本的情况下无法安全线性化 admission/completion，或 cooperative
  cancellation 无法与现有 `CancelToken`/`JoinHandle` 组合，或同一 scope state 不能安全跨线程/Fiber drain，
  则以精确 blocker 终止；否则有限 FIX、全部适用门和文档通过即重新 `CLOSED`。

## SCOPED 不变量

1. `TaskScope` 是唯一新增 owner，不复制 Scheduler、Fiber、Task、JoinHandle、CancelToken、WaitGroup 或 IO
   runtime；任务只提交给构造时绑定的一个 Scheduler。
2. spawn 的 admission 先于 task publication 完成计数；与 close 竞态的调用要么完整进入并被 join，要么在
   创建 scope-owned driver/state/stack 前返回 typed rejection，不能出现半接纳、漏计或永不完成的 handle。
3. `join` 与 `cancel_and_join` 都关闭 admission 并等待所有已经接纳的 child terminal；取消只是 cooperative
   signal，首个失败不会提前返回或遗弃其它 child。
4. typed result 仍由显式 `JoinHandle<T>` 单次取得；scope 只聚合生命周期，以及至少 `void` 基础设施失败和
   `Status` 非 OK 结果。不得为了聚合复制任意 `T`。
5. scope state 必须比全部 child 活得久；析构有 live child 是明确 misuse 并 fail-fast，绝不采用
   `JoinHandle` 的 detach 语义。普通 task 路径没有 scope pointer、计数或取消字段。

## 进入 `AUDITED` 前的有限审计队列

- 锁定 Folly `AsyncScope/CancellableAsyncScope` 与 Seastar `gate` 的 add/close/join/cancel/destructor 边界，
  再映射到 HYLE 的 Status、CancelToken、Scheduler 和 blocking/fiber-aware wait；
- 追踪 `TaskDriver<T>`、`JoinState<T>`、Scheduler admission、WaitGroup/ConditionVariable 与取消树，选择只由
  scoped child 支付的 completion hook/driver/state 形状，证明普通 spawn 二进制路径不变；
- 冻结公开构造、spawn、close、join、cancel-and-join、remaining/closed 查询，以及 OOM/admission/重复调用
  的错误优先级；明确是否允许多个 joiner、join 是否隐式 close、child 内递归 spawn 的线性化规则；
- 用两条真实 consumer trace 冻结 `KEEP/FIX/BORROW/REJECT` ledger、correctness/failure matrix、P0 账本与
  同合同 P1 episode。完成这些之前不得修改产品代码。

## 能力、API 与机制审计（`AUDITED`）

锁定 peer 为 Folly `65749da412d92fe27d87ae45a8ef670720712858` 的
`folly/coro/AsyncScope.h` 与 Seastar `78f64e2317b19c7e34ba61f5e8ea85f083eca1da` 的
`include/seastar/core/gate.hh`。Folly 证明 runtime-sized add、显式 join、cooperative cancel 和 live-task
析构失败是成熟公开形状；Seastar 证明“先关闭准入，再等待已进入 owner 全部离开”的 gate/holder 生命周期。
二者分别绑定自己的 coroutine/executor 或 reactor/future，HYLE 只借语义，不引入第二 runtime。两份独立
证据已足够冻结本能力；不再为了扩大 peer 数量继续抄 API。

现有 HYLE 路径的只读追踪结论：`WaitGroup` 能 fiber/thread-aware 等待动态计数，却没有 close/error/cancel/
owner；`when_all/when_any` 要求调用前已经持有固定 handle 集；`JoinHandle` drop 明确 detach；`CancelToken`
已经能作为嵌套 cooperative source；`Scheduler::spawn`、stack pool、`TaskDriver<T>` 与 `JoinState<T>` 已经拥有
正确 admission、结果和 cleanup。故不增加新 module、queue 或 wake primitive，也不把 scope pointer 塞进
普通 `JoinState`。

### Public API freeze

```cpp
auto scope_result = hyle::sched::TaskScope::create(scheduler, allocator);
HYLE_ASSIGN_OR_RETURN(auto scope, std::move(scope_result));

if (Status admitted = scope.add(make_status_task(allocator)); !admitted.ok()) {
  (void)scope.cancel_and_join(); // drain any earlier children
  return admitted;
}
auto value = scope.spawn(make_value_task(allocator));
auto cancellation = scope.cancellation_token(); // pass to cancellable IO
HYLE_RETURN_IF_ERROR(scope.join());       // closes admission, then drains
// 或 scope.cancel_and_join();             // close + cooperative cancel + drain
```

公开面固定为：

- `static Result<TaskScope> create(Scheduler&, mem::AllocatorRef) noexcept`；owner 为 move-only，state 由所给
  allocator 分配，Scheduler 与 allocator backing 必须活过 scope 和全部已接纳 child；
- `spawn(Task<T>, stack_size=kDefaultStackSize) -> JoinHandle<T>`；不接收另一个 Scheduler。`void/Status`
  child 的失败由 scope 聚合，其它 `T` 只聚合 infrastructure failure，值仍只由返回的 handle 拥有；丢弃/
  detach handle 不影响 scope ownership；
- `add(Task<void/Status>, stack_size=kDefaultStackSize) -> Status` 是无需独立 child result/cancel handle 时的默认
  scope-owned 路线；它只同步返回 ownership 尚未转移时的 invalid/closed/task/stack rejection；一旦 admission
  CAS 成功就返回 OK，此后的 allocation/Scheduler/运行期错误统一由 `join` 聚合。循环中 direct rejection 后
  仍须 cancel/join 已有 child，不能裸 `HYLE_RETURN_IF_ERROR` 跳过 drain；该路线不制造随后立即丢弃的
  `JoinState/JoinHandle`；需要 typed value 或逐 child cancellation 时才使用 `spawn`；
- `close()` 幂等且与 spawn 原子线性化；`join()` 隐式 close，`cancel_and_join()` 依次 close、发布 cooperative
  cancellation、drain。二者可由 OS thread 或本 Scheduler 的 Fiber 调用，允许多个 waiter 并返回同一首错；
- `request_cancel()` 只发布 cancellation、不关闭准入；之后接纳的 child 立即继承已取消 token。查询面为
  `remaining()`、`is_closed()`、`is_cancel_requested()` 与 moved-from 检查用 `valid()`；
- `cancellation_token()` 返回可共享的 token，让 child 把同一个 scope cancellation source 显式传给 cancellable
  IO/等待；无效 scope 返回 invalid token，不新建 cancellation tree；
- moved-from owner 仅可析构/赋值/查询；其它操作返回 typed `InvalidArgument` 或安全 no-op。曾接纳 child 的
  scope 在析构前必须完成 `join/cancel_and_join`；live child 或未 join 的已用 scope 析构 fail-fast。

错误优先级固定为 invalid/moved-from scope → invalid Task/frame → unsafe stack size → closed/overflow admission
→ join-state（仅 `spawn`）/driver/stack allocation → Scheduler closed/full admission → child completion。close 后
的调用没有被接纳，`spawn` 错误由其 `JoinHandle`、`add` 的 pre-admission rejection 由返回 Status 表达；
try-admit 已成功后的资源或 Scheduler admission 失败只进入 scope 首错并将计数归零，`add` 仍返回 OK；
pre-admission rejection 后调用者必须先 drain 更早接纳的 child，任何路径都不能让 join 挂起。

### Internal mechanism freeze

- scope state 用一个 producer-written lifecycle word 表达 closed、ever-admitted、joined 与 admitted count；
  spawn/add 与 close 通过 CAS/fetch-or 给出单一线性化点。worker-written completed counter 独占另一 cache line，
  避免 admission 与短 child completion 在同一 line 上跨核来回争用；每个成功 completion 只做一次 fetch-add，
  仅首错和 close 后最后一个 completion 进入现有 fiber-aware ConditionVariable 的短控制锁；
- scoped child 使用专用 `ScopedTaskDriver<T>`，与现有 `spawn_task_impl` 共享相同的用户 coroutine frame、
  `JoinState<T>`、一个 driver allocation 和 scheduler stack pool。driver 只多 raw scope state pointer；不增加
  wrapper coroutine/frame，也不修改普通 `TaskDriver`、`JoinState`、Fiber、Scheduler queue 或 wake path；
- `add` 使用只拥有 coroutine/Fiber/scope pointer 的 scope-owned driver，直接把 void/Status completion 发布给
  scope；它不分配 `JoinState`、不构造 `JoinHandle`、不安装逐 child cancel flag，仍复用同一 stack pool、
  Scheduler admission、scope CancelToken 和 lifecycle 线性化点；
- driver 执行期间把 scope `CancelToken` 作为嵌套 `CancelScope`，与每个 JoinHandle 的既有 cancel flag 组合；
  在发布 scope completion 之前销毁该临时 scope，保证 join 返回后 state/token 可安全销毁；
- child 先完成自己的 `JoinState`，再递减 scope active count；因此 scope join 返回时所有保留的 handles 都已
  terminal。scope state 不保留 handle、任意 `T`、task node 列表或 unbounded container。

### 有限改动集

| ID | 已证实差距 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `S7-01` | 无动态 owner/API | 新增 `task_scope.h/.cc`，实现上述 create/spawn/close/join/cancel/query | API、生命周期与错误优先级测试 |
| `S7-02` | concurrent spawn/close 与 cleanup 尚无机制 | 原子 lifecycle + scoped-only driver；复用现有 CV/CancelToken/stack pool | race、OOM、shutdown、TSan；普通路径产品 diff 账本 |
| `S7-03` | void/Status 首错与 full drain 缺失 | completion 时只保存首个非 OK Status，全部 child 仍递减/drain | first-error、后续完成、detach/typed result |
| `S7-04` | 公开消费与交付缺证 | focused/header/source/static/shared、三平台、sanitizers、guide/architecture | 预声明矩阵全部诚实裁决 |
| `S7-05` | 新抽象成本未知 | opt-in v7 P1：动态 Status helper fan-out 的完整 scope lifecycle 对 pre-v7 公共手工组合；双向顺序/placement | qualified 5% NI；无同合同则明确阻塞，不换轻语义 peer |
| `S7-06` | 首候选强迫只需聚合的 helper 创建并立即 detach `JoinHandle`，且 admission/completion 争用一个计数 line、scope token 不能传给 cancellable IO | 增加 `add(Task<void/Status>)` scope-owned driver、拆分 admitted/completed cache line 与 `cancellation_token()`；保留 `spawn` 给 typed/per-child ownership | 无 unused JoinState/producer-worker line ping-pong；IO token 消费测试；同合同 P1 重新过 5% NI |

本表之后不加入 G08 priority、deadline、轻量 callback scope、DAG、IO/process/WebSocket 或普通 scheduler
优化；出现问题只能在 `S7-01..S7-06` 内纠正，否则依法回退/阻塞。

### 七问与红队预裁决

1. 第三方：运行时仍只依赖既有 vendored fcontext；Folly/Seastar 是设计证据，不成为依赖。
2. 完整性：动态 Task owner 的最小完整闭环是 admission/close/join/cancel/error/destructor；deadline/priority 与
   上层 request policy 已分离。
3. 内部复用：复用 Error、Mem/Mem-vm、CancelToken、ConditionVariable、Scheduler admission/stack pool 和
   JoinState；不重复容器、wake 或 cancellation tree。
4. 性能：所有操作 O(1)，普通 task 零产品改动；scope child 每次一次 producer-line admission CAS、一次隔离
   worker-line completion fetch-add，失败/末项才进入控制锁；scope-only helper 不再支付 JoinState/JoinHandle。
   P1 只测完整同合同 episode。
5. 交付：只承诺 C++ source/static/shared；模板 spawn 不增加 Native FFI/C ABI。
6. 构建：新 source 加入既有 `sched` target，中央依赖图不变；新 public header 由 sched owner 自动安装并
   自包含测试。
7. 易用：allocator/scheduler 在 create 时只写一次；`add` 对应 fire-within-scope，`spawn` 返回熟悉的
   JoinHandle，join 自动 close，token 可直接传给 IO；guide 明确取消合作性、late spawn、错误聚合、析构和
   owner lifetime 陷阱。

实现前红队没有发现需要扩大产品范围的 finding。高风险点已进入 S7-02/S7-03：last completion 与 waiter
publication 的失唤醒、close/admit 半接纳、cancel token 在 join 后 UAF、失败路径漏减计数、scope error 早返、
live-child 析构、普通 task metadata 污染。下一步只实施上述有限改动集。

## 首个 product candidate（已依法退回 `AUDITED`）

- immutable product commit / tree：
  `35f318a3fc33393b144b6576e272788b3cc0a7d0` /
  `1bf7c2d2c1f1750f5e019fc039a7c176ec16c9c6`；后续 correctness、delivery、sanitizer、P0/P1 与终局红队
  只能验证该候选，若产品/API/测试语义改变则依法退回 `AUDITED` 并签发新候选；
- 实现严格落在 `S7-01..S7-04`：新增独立 `TaskScope` state/driver/header/source/test，普通 `TaskDriver`、
  `JoinState`、Scheduler queue 与 admission 算法没有产品改动；`Fiber` 仅增加 scoped driver 的 private friend
  声明，中央 component edge 不变；
- W1-WIN 候选前检查：TaskScope 21/21、Sched 完整 205/205、sched public-header self-containment、closure
  checker、docs checker 与 `git diff --check` 均通过。它们是候选签发门，不替代后续精确候选三平台、三交付、
  sanitizer、P0/P1 和终局红队；
- `AUDITED -> CANDIDATE` 前态 registry snapshot：
  `C:\public\project-hyle\_validation\sched-v7-dynamic-task-scope\transitions\module-status-before-candidate.json`，
  SHA-256 `CE085F878A9717264BAF3D79DA6224BEE05BB0742801F365145FA060049E0BB9`。

精确候选随后完成 W1-WIN/W1-WSL/M1-MAC Sched correctness/header、source/static/shared relocated delivery、
Windows ASan 以及 WSL ASan/UBSan/TSan，均通过。W1-WIN v1 仪器的 15-run 诊断样本中 A/A、5%/10%
known-delta、timer-floor、placement 与 host/artifact identity 全部通过；但 `spawn(...).detach()` 候选在四个
顺序/布局层的 median advantage 分别为 `0.8552`、`0.8434`、`0.8785`、`0.8573`，两个 primary 层明确
`LOSE`。该 manifest 因错误使用 `--formal` 而缺少合同未声明的 formal toolchain pin，整体只作诊断，不能签
P1 verdict；但跨层固定约 12%–16% 的损失足以构成产品 finding。代码归因显示只需 scope 聚合的 Status child
仍为每个 helper 分配 `JoinState`、构造/释放 `JoinHandle` 并叠加 scope 计数，而 Folly AsyncScope 的正常
形状是直接 `add` void work。故不放宽门槛、不重复采样，按方法从 `CANDIDATE` 退回 `AUDITED`，只实施
`S7-06` 后签发新候选。退回前 registry snapshot 为
`C:\public\project-hyle\_validation\sched-v7-dynamic-task-scope\transitions\module-status-before-audit-return.json`，
SHA-256 `85BAE7921BA41E81B7966538BB6561CFEE2584AFE7638C07C13A87500BC19929`。
