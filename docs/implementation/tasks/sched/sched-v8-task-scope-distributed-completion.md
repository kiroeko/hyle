# Sched v8：TaskScope 分散完成记录

<!-- MODULE CLOSURE RECORD: BLOCKED -->

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
- 调查与纠正预算：peer/机制审计已经结束；第一候选实现批次之后，只允许性能 preflight 从实际源码
  证明的 `S8-06/S8-07` 两项无 API/layout 纠正、至多一次 correctness 修正、一次 P0 和一次 qualified
  P1。不能扩样、换轻语义 peer、放宽门、增加普通 task 成本或继续发明 completion 算法；
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
   个同分配 completion cell 的 storage（Task、Fiber 与 stack 会在完成发布前释放）。`remaining()` 若保留须
   诚实标为 O(N) snapshot，不能暗藏共享完成写来维持旧点复杂度。
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
| `S8-06` | 成功 `join()` 后的合法析构仍逐 child 调用 `Event::is_signaled()`，为 32-child owner 增加 32 组无语义价值的锁；`joined` 只会在 `wait_` 已等待完整稳定 list 后置位 | `reset_` 只用 `ever_admitted && !joined` 保留 misuse fail-fast；正常 joined teardown 直接释放 record，不再重复证明 completion | 两项 death test 仍通过；不得改变 record、publication 或 public API |
| `S8-07` | `S8-06` 后的同拓扑短 probe 仍在 8/8 次中同向落后（diagnostic ratio `0.894..0.969`）。静态账单和 MSVC 生成代码把唯一额外的逐 child 同类成本定位为 admission 的 `references.fetch_add`（`lock inc`）：candidate 与 peer 都有 publisher/owner 两次 release RMW，但 peer 的 `JoinState` 直接以 2 refs 构造 | record 尚未进入 list、driver 尚未交给 Scheduler 时，以 relaxed initialization store 建立 scope reference；失败 admission 仍保留单一 publisher ref | admission 生成代码不再有额外 locked RMW；final-publisher race、concurrent add/close、OOM/queue failure 与 death tests 通过；一次短 paired probe 后立即决定是否有资格签 candidate，不再增加修正或样本 |

### API / usability freeze

公开方法名、参数和返回类型不变。`add` 的成功仍只表示 scope 已接管 task；资源准备、invalid/closed 属于
未接管的同步失败，queue admission/运行期失败属于 join 的聚合失败。`spawn` 继续把 typed value 与逐 child
cancel 放在 `JoinHandle<T>`；scope 不复制任意 `T`。`join/cancel_and_join` 支持多个 OS-thread/Fiber waiter，
都等待 driver cleanup 后的 record terminal；`remaining()` 只用于控制面观测，是 O(N) 瞬时快照，不应放在
热循环。每个 child 不增加第二次 allocation；Task、Fiber 与 stack 在完成发布前释放，但同一 driver
allocation 的 cell storage 会保留到 scope teardown。普通 task、没有 TaskScope 的 Scheduler 和 queue 不支付
任何成本。

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

## 第一候选与性能 preflight 退回（`AUDITED`）

- immutable product commit / tree：
  `cb4e892b6df85d779c7df03af7f1d4508981dfe2` /
  `68ae82483ffad1484e285eed24ba8dc69dbce43c`；公开 API 不增入口，算法、对象布局、测试语义与用户文档已经
  固定；
- 实现用每个 scoped driver 同分配的 manual-reset completion record 取代 owner-wide completion counter；
  Task、Fiber、stack 与 typed JoinState task reference 均在 record terminal 发布前释放；scope/publisher 双引用
  保护 signal 尾部与即时 teardown；
- admission/close 由一个短锁线性化；未 admission 的 driver/stack 准备失败直接返回，admission 后的 Scheduler
  与运行期失败进入首错/full-drain 聚合；成功 completion 不写 scope state；
- 候选前 W1-WIN instrumented Sched、公开头与 TaskScope 28 项通过，并发 admission/close 与即时 teardown
  三项回归重复 1000 轮通过；closure/docs/diff 检查通过。这些只是签发检查，不代替下述精确候选验证；
- `AUDITED -> CANDIDATE` 前态 registry snapshot：
  `C:\public\project-hyle\_validation\sched-v8-task-scope-distributed-completion\transitions\module-status-before-candidate.json`，
  SHA-256 `022EC2246247F5E31D4C73C6B633E79443C0BAE34F23EBB441B5F7C50681C3C8`。

该 candidate 的正确性证据仍是历史事实，但性能 preflight 发现下述有限产品问题后不再是活动 candidate。
`CANDIDATE -> AUDITED` 前态 registry snapshot 是
`C:\public\project-hyle\_validation\sched-v8-task-scope-distributed-completion\transitions\module-status-before-performance-return.json`，
SHA-256 `D26B0763ED68161ADBBA595A884703F098F87DA5844020CED8553E75BC2E4681`。`S8-06` 后的短 probe 在所有
八次换序/placement 诊断中仍同向落后，因而没有浪费正式 P1 样本；源码与反汇编随后证明 admission 比
同语义 peer 多一次逐 child locked refcount RMW。只允许再实现 `S8-07`、重跑其直接 correctness/P0/短
probe 后决定是否签发新 candidate；peer、用户任务、API、主算法和 5% 门槛不变。若短 probe 仍有
material 固定输，本轮立即 `BLOCKED`，不得继续扩充修正或样本。

## P1 完成边界预审纠正

第一次启动的 v7 合同运行在四个 product strata 各完成 31 个样本后被人工终止；它只完成了 31-sample
control 的 33/62 个进程，没有生成 manifest/summary，因此不是 qualified P1，也不进入产品裁决。停止不是
为了挑结果，而是源码复核证明合同把不同完成边界错误写成了 `SAME_CONTRACT`：

| 时序阶段 | v8 TaskScope candidate | 37b62460 public manual peer |
|---|---|---|
| user task / `Status` result 结束 | `ScopeOwnedTaskDriver::entry` 保存结果，但不 signal | `TaskDriver::entry` 调用 `JoinState::set_value` 并 signal |
| waiter 可返回 | 尚不可；record 未发布 | `JoinHandle::join` 此时即可返回 |
| Fiber finish hook、coroutine/driver 与 stack 清理 | `ScopeOwnedTaskDriver::finish` 在 record signal **之前**完成 | `TaskDriver::destroy` 在 result signal **之后**由 worker dispatch 执行 |
| owner 返回 | `TaskScope::join` 已包含上述 cleanup；scope 析构释放 retained cells | 原 v7 bench 可在 finish hook 尚未完成时返回并开始下一 owner |

源码锚点是候选 `src/sched/task_scope.cc::ScopeOwnedTaskDriver::entry/finish`、
`TaskScope::wait_/reset_`，以及 identity-locked 37b62460
`include/hyle/sched/join.h::JoinState::complete_/TaskDriver::entry/destroy`；Scheduler dispatch 在 finish hook
之后才 `finish_one()`，故公开 `Scheduler::drain()` 是能把 peer 推进到同一资源完成边界的自然 adapter。
原始 product 点值约为 peer 的 92.6%–96.6%，只能说明 stronger candidate route 与 result-visible peer route
不同，不能成为同合同输赢或继续改产品的依据。

对象/生成代码静态账单也不支持“候选必然更重”的猜测：MSVC 19.50 exact artifacts 中 candidate 的
scope-only Status driver 是 288 B、每 child 一次 allocation；manual peer 是 120 B `JoinState` + 184 B
`TaskDriver`、每 child 两次 allocation，另有 16 B handle slot。candidate 的 retained 288 B cell 与 peer
retained 120 B state + 已发布后异步清理的 184 B driver 只是生命周期分布不同。先对齐完成边界，才有资格
判断分散 record 是否非劣。

证据修正不改变产品 candidate。新合同
[`sched-v8-task-scope-cleanup-p1-v2`](../../../../bench/contracts/sched-v8-task-scope-cleanup-p1-v2.json)
在双方 owner 返回前都调用公开 `Scheduler::drain()`；candidate 路线该 fence 已满足，peer 路线等待
dispatch finish hook。随后两侧都在 lambda 返回前析构各自 retained owner records。旧 v7 executable/contract
保留为历史事实；v8 使用独立 target 与 consumer provenance，避免改写旧 bundle。

完整采样前的有界 diagnostic 又发现第二个仪器问题。cleanup-aligned 的四个 separate-process rows 在
1024-owner 聚合下随顺序在约 `0.94..1.12` 间反转；增至 4096 owner（每进程约 60 ms）后，candidate-first
约 `1.01..1.03`，peer-first 约 `0.94..0.96`。再把**同一个 candidate executable**放在 pair 两侧，五组
first/second ratio 仍为 `0.98, 1.04, 0.94, 0.97, 1.01`。这不是 TaskScope 固定劣化，而是第一个/第二个
独立进程的启动与 CPU 热状态差；公共 A/A 却在一个 control process 内跑两行，无法否证这种拓扑偏差。
因此 structured v2 草案没有进入正式采样，也没有生成证据。

HYLE 的首要消费者是同一 source/static/shared 进程内直接调用公开 C++ 路线。最终 v2 合同据此使用既有
`nanobench-paired-rows-v1`：candidate 和 exact pre-v7 public composition 位于同一个 source-built consumer，
各自拥有同配置、同 4096-owner（W1 约 60 ms）warmup 的持久 Scheduler，避免前一路线把 stack pool/
allocator 历史留给后一路线，也避免第一条路线只预热约 1 ms 而第二条路线继承近一秒热状态；
allocator/helper body 相同，每个计时 operation 都完成一个 32-child owner 的 admission、yield、
结果/首错、finish-hook/stack cleanup、`drain` 和 owner teardown。
双顺序与独立 placement target 仍保留，产品 comparison 现在使用同进程 paired rows；下述终局进一步
检查该 workload 自身的同路线 A/A。该修改只纠正证据形状，不改变 API、算法或 material margin。

## 终局：workload-matched A/A 阻塞（`BLOCKED`）

`S8-06/S8-07` 的实现提交是 `8488b6025443e2bc1d90dbcff03929e0178a2c2e`，但它没有被签成新的
candidate。Windows focused `TaskScope*` 30/30（含两项 death test）通过；MSVC 19.50 反汇编确认
admission 的 `lock inc` 已变为普通 `mov refs, 2`。这是源码证明成立的 P0 成本修正，不等于 P1 通过。

正式采样前只运行一次固定的短判别组。产品 A/B 在 primary 两个顺序为 `1.065/1.127`，在只增加 80 B
virtual `.text`、raw `.text` 同为 `0x32a00` 的 placement artifact 上为 `0.970/0.934`。随后在**同一个
executable**内把两个标记行都替换为完全相同的 candidate route 或完全相同的 peer route，保持各自独立、
同配置 Scheduler、4096-owner warmup、allocator、worker affinity 和 owner teardown；结果如下：

| same route | primary candidate-first | primary peer-first | placement candidate-first | placement peer-first |
|---|---:|---:|---:|---:|
| candidate/candidate first÷second | `0.917` | `1.044` | `1.012` | `0.972` |
| peer/peer first÷second | `0.974` | `1.052` | `1.010` | `1.052` |

同路线 A/A 自身覆盖 `0.917..1.052`，已经跨过 5% material margin；通用算术 control 没有相同
Scheduler/stack-pool/allocator/四 worker 热状态，不能否证该偏差。因此 primary/placement 的产品方向翻转
没有资格归因给 TaskScope，完整 `n=15` 未启动，也没有生成 qualified manifest/summary。增加样本、选择
primary 的 WIN、选择 placement 的 LOSE 或继续改产品都会让 benchmark 塑造实现，均被拒绝。

| 聚合 gate | Verdict |
|---|---|
| scope / architecture | `PASS` |
| correctness | `UNPROVEN`：本提交只按停止协议完成 Windows focused 30/30，未伪造三平台/三 sanitizer |
| delivery | `UNPROVEN`：没有在仪器 preflight 失败后浪费三交付回放 |
| usability_docs | `PASS`：公开 API 未变 |
| red_team | `UNPROVEN`：未签新 candidate，终局 candidate 红队不适用 |
| engineering_performance | `INCONCLUSIVE`：workload-matched A/A 未通过 5% preflight |
| product | **`BLOCKED`** |

blocker owner 是 **HYLE performance-instrument owner**。唯一恢复条件是独立、有界的仪器任务提供一个
保持当前真实 owner 语义、Scheduler/allocator/worker 拓扑和双顺序的测量方式，并在 primary/placement 上
先让 candidate/candidate 与 peer/peer 的 workload-matched A/A 全部落入预声明 5% band；随后为 `sched`
建立新 closure card，依法 `BLOCKED -> SCOPED -> AUDITED`，签发不可变 candidate，只运行一次 qualified
P1。重跑现有 harness、扩样、放宽门或继续修改 TaskScope 都不满足恢复条件。

LOC 绑定实现提交 `8488b602`：全库代码 `535,654`，自实现不含测试 `93,660`，Sched 实现 `6,256` 行；
`loc-report.py --self-check` 通过。
