# Sched v8：TaskScope 分散完成记录

<!-- MODULE CLOSURE RECORD: SCOPED -->

## Closure card

- component / owner / kind：`sched` / Sched / `REOPEN`；round=`sched-v8-task-scope-distributed-completion`；
- baseline：`3ef5eed90f96ffda5edba6adf775a690026efc7b`；前轮 terminal record 是
  [`sched-v7-dynamic-task-scope`](sched-v7-dynamic-task-scope.md)；
- trigger 与恢复证据：v7 的唯一 qualified 同合同 P1 无法通过预声明 5% 非劣门，registry 要求先形成减少
  multi-worker shared completion-line 成本的有界方案。该方案已独立固定在
  [`sched-design`](../../../architecture/sched-design.md)：借鉴 Tokio `JoinSet` 的逐任务完成记录，但把记录
  与 HYLE 原本就需要的 scoped driver 同次分配；Folly `Barrier` 与 oneTBB `wait_context` 作为被明确否决的
  constant-state 取舍对照；
- 正常 consumer：推理 request、model load 或 rendering frame 动态产生数十个会挂起一次的 helper，停止
  admission 后保留首错并完整 drain；
- 主/反偏置纵切：沿用 v7 已冻结的 inference/rendering 32-child trace；资源加载 frame 是反偏置纵切，
  防止实现只适合 HTTP request 或 benchmark；
- public journey：`TaskScope::create` → concurrent `add/spawn` → `close`/`request_cancel` →
  `join/cancel_and_join`；Error、Mem、Scheduler、CancelToken 与 IO cancellation 的 owner 不变；
- scope：只替换 `TaskScope` 的内部 completion accounting/lifetime；保持动态 admission/close、共享取消、
  首错、多个 waiter、typed handle、full drain、live-child fail-fast 和三种 C++ 交付；
- non-goals：G08 priority、Scheduler queue/idle/steal、普通 Task/JoinState/Fiber layout、deadline、第二 runtime、
  C/Native FFI、Compute/Graph、公共性能仪器或 benchmark workload 重写；
- 直接依赖/消费者：中央 `sched` edge 不变；直接消费仍是 IO 与后续 G03/G04，不能要求其理解内部 record；
- 性能：`P1`，material margin `5%`。只运行 v7 已资格化的同语义 32-child dynamic owner 合同；同一
  Scheduler/allocator/body/yield/failure/output/cancel/close/drain，双顺序/placement、A/A、known-delta 与
  timer-floor；
- 调查与纠正预算：peer/机制审计已经结束；一次 API/实现审计、一个产品实现批次、至多一次 correctness
  修正、一次 P0 和一次 qualified P1。不能扩样、换轻语义 peer、放宽门、增加普通 task 成本或继续发明
  completion 算法；
- 预声明验证：TaskScope 0/1/N、void/Status/typed、并发 add-vs-close、recursive add、多个 waiter、首错/full
  drain、cancel token + IO、OOM/queue-full/shutdown、moved-from/live-child death；W1-WIN/W1-WSL/M1-MAC
  focused Sched + public header；Windows ASan、WSL ASan/UBSan/TSan；source/static/shared relocated consumer；
  产品 diff/P0、同合同 P1、终局红队以及固定治理检查；
- stop：correctness 或 lifetime 不能在一次修正内关闭，或唯一 qualified P1 仍不能通过 5% 非劣门，则诚实
  `BLOCKED`；全部适用门通过才 `CLOSED`。

## SCOPED 不变量

1. 普通 `schedule/spawn_task` 的字段、分配、分支、queue 和 completion 路线产品 diff 必须为空。
2. 每个 scope child 最多沿用原 driver 的一次 allocation；不能为 record 再分配一个对象，也不能按 worker
   数给每个 scope 预留 cache-line 数组。
3. admission 与 close 仍有单一线性化点。close 后 list 稳定；被接纳的 record 在 driver publication 之前
   已进入 scope ownership，未接纳的 record 完整回滚。
4. 成功 child 的 terminal 路线只写自己的 record。失败可进入共享首错锁；任何 child 都不得再写
   scope-wide completion counter。
5. driver 必须先离开 Fiber stack、归还 stack/JoinState task reference，再发布 record terminal。scope 与
   publisher 各持一个内部 record 引用，解决 waiter 醒来后立即 teardown 与 `signal()` 尾部并发的 UAF。
6. admission/completion 每 child O(1)，close O(1)；完整 drain/destruction O(N)，scope 生命周期保留 O(N)
   小记录。`remaining()` 若保留须诚实标为 O(N) snapshot，不能暗藏共享完成写来维持旧点复杂度。
7. public API 与 v7 的自然任务形状不因 P1 改名或增加 `fast/light` 变体；性能成立依赖由实现承担。

下一步仅进行一次实现审计，冻结有限 `FIX` ledger 和 record/cell 生命周期后进入 `AUDITED`；SCOPED 阶段
不得写产品代码。
