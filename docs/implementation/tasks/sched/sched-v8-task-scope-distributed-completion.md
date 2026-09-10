# Sched v8：TaskScope 分散完成记录

<!-- MODULE CLOSURE RECORD: AUDITED -->

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

SCOPED 阶段没有写产品代码；以下审计冻结实现范围后才进入 `AUDITED`。

## 能力与机制审计（`AUDITED`）

peer snapshot 已固定为 Folly `65749da4` 的 `coro/detail/Barrier.h`、oneTBB `e00c68dc` 的
`detail/_task.h::wait_context` 与 Tokio `060cc4e4` 的 `task/join_set.rs` + `util/idle_notified_set.rs`。前两者
都让每个 completion RMW 一个 owner-wide count；Tokio 则保留逐 task entry，让完成通知写该 entry。HYLE
不需要 Tokio 的 unordered result queue、task ID、abort handle 或第二 runtime，只借其“动态 owner 可以用
O(N) retained entry 换掉中央完成写争用”的机制边界。

现有 HYLE `Event` 已提供 manual-reset、多个 fiber/thread waiter 与 wake-before-wait 正确性；`JoinState`
已证明 publisher-side reference 可以保护 notification 尾部。故 record 不再发明 wait primitive：每个
scoped driver cell 内嵌 `TaskScopeCompletion`（intrusive next、manual Event、publisher/scope 两引用、dispose
seam），scope close 后按 admission 顺序等待这些稳定 record。运行期 non-OK 先在 scope 短锁中执行
`Status::Update`，再 signal 自己的 record；成功路径完全不写 scope state。

资源准备移到 admission 线性化之前：driver cell 和 Fiber stack 无法准备时，`add` 直接返回 typed error、
`spawn` 返回 ready failed handle，当前 task ownership 未转移；调用者仍须 drain 更早成功接纳的 child。
record 链入短 `SpinLock` 临界区是唯一 admission/close 顺序点，链入后才由 Scheduler 接管；链入后的 queue
full/shutdown 与运行期错误由 scope/handle 聚合。这避免为了兼容 v7 的“先 admission 再分配”时序而新增
in-flight admission counter，也不在 allocator/stack acquisition 期间持锁。

### 有限改动集

| ID | 已证实问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `S8-01` | 成功 child 争用 owner-wide completion line | 以 driver 同分配 `TaskScopeCompletion` 和稳定 intrusive record list 取代 admitted/completed counters | 产品路径无第二 allocation；成功 publish 只触碰自己的 record |
| `S8-02` | waiter 可在 `Event::signal()` 尾部前 teardown | record 初始 publisher ref，admission 增加 scope ref；signal 后 publisher 才 release，scope teardown 只 release 自己引用 | 确定性 final-publisher pause 下 join/destruct 安全；ASan/TSan |
| `S8-03` | close/admission/list 稳定性 | driver/stack 先准备；一个 scope SpinLock 原子完成 closed check + link；close 持同一锁 | concurrent add/spawn-vs-close、OOM、queue full、recursive add |
| `S8-04` | 旧计数 API/文档成本不真实 | `remaining()` 保留为 O(N) snapshot；guide/architecture 说明 record retention、同步 rejection 和 teardown | 公开 API 不增 benchmark 变体；header/guide/三交付 |
| `S8-05` | v7 P1 未过门 | 沿用同一 qualified contract/peer/5% threshold，只更新 candidate provenance | 一次 qualified P1 全 required strata NI PASS，否则 BLOCKED |

### API / usability freeze

公开方法名、参数和返回类型不变。`add` 的成功仍只表示 scope 已接管 task；资源准备、invalid/closed 属于
未接管的同步失败，queue admission/运行期失败属于 join 的聚合失败。`spawn` 继续把 typed value 与逐 child
cancel 放在 `JoinHandle<T>`；scope 不复制任意 `T`。`join/cancel_and_join` 支持多个 OS-thread/Fiber waiter，
都等待 driver cleanup 后的 record terminal；`remaining()` 只用于控制面观测，是 O(N) 瞬时快照，不应放在
热循环。每个 child 增加的是同一 driver allocation 内的一份 record 并保留到 scope teardown；普通 task、
没有 TaskScope 的 Scheduler 和 queue 不支付任何成本。

### 七问预裁决

1. 第三方只作设计证据；运行时依赖仍只有既有 fcontext，版本/许可证/构建不变。
2. G02 能力与 v7 public journey 不缩减；变化只是内部 accounting 和更自然的 ownership-transfer 错误边界。
3. 复用 Error Status、Mem AllocatorRef、Sched Event/SpinLock/CancelToken/Fiber/JoinState，不复制容器或 wait。
4. admission/completion 每 child O(1)，drain/destruction O(N)，无共享成功 completion RMW；P1 不预判结果。
5. 仍仅 C++ source/static/shared，没有 C/Native FFI 或新 ABI 面。
6. source/header 仍属于现有 `sched` target，中央依赖与 package component 不变。
7. 调用形状不变；guide 将明确同步 rejection 后也要 drain、record retention、O(N) remaining 和 child 内
   不得 join 自身 scope。

红队预审否决了固定 per-worker cache-line 数组（每 request O(workers) metadata）、close 后 polling（破坏
fiber/thread blocking）以及重新使用 JoinState（恢复 v7 已删除的额外 allocation/result state）。有限
`S8-01..S8-05` 已冻结，下一步只实现该集合。
