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

