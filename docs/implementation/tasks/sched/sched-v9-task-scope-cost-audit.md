# Sched v9：TaskScope 源码成本审计

<!-- MODULE CLOSURE RECORD: CANDIDATE -->

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
- non-goals：扩样或 qualified `n=15`、换 peer/workload/margin、为点值添加 benchmark-only API/layout、G08
  priority、Scheduler queue/idle/steal、IO、Compute/Graph、再次修改公共仪器；
- 直接依赖/消费者：既有 `sched` 边不变；直接消费者仍为 IO 与后续 G03/G04；
- 性能：`P1`，material margin `5%`。本阶段只做 P0 源码/生成代码审计，不计时；API、算法和 correctness
  固定成新 candidate 后，才允许用已通过 A/A 的 v2 合同运行唯一一次 qualified P1；
- 调查与纠正预算：一次静态成本表、一次生成代码检查、一次 peer/API 复核；至多冻结两个由源码和正常
  consumer 直接证明的产品修正。AUDITED 前不得改产品；修正后只允许一次 focused correctness/P0 preflight 和一次 bounded
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

候选已经固定；下一步只执行本卡预声明的 candidate 验证，不再改变 API、算法、layout 或 workload。

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
| teardown | terminal owner 观察后逐项 sized-free | driver/JoinState 分布释放、handle buffer 回收 | 都在 owner wait 中释放，不再延迟到 scope 析构 |

MSVC 19.50 Release object 进一步把未决成本缩小到两项。公开 `TaskScope::add(Task<Status>)` 的正常路径
先调用自身 wrapper，再调用同 TU 但未内联的 `task_scope_add<Status>`；manual route 每 child 只有一次
`spawn_task_impl` 调用。`TaskScope::wait_` 的循环则无条件调用 out-of-line `Event::wait()`，即使
`TaskScopeCompletion::references == 1` 已经由 publisher 的 release RMW 证明 Fiber、Task、stack、publish
tail 全部结束。该 acquire observation 可以直接复用现有 refcount，不增加 layout/metadata；观察到 2 时
仍走原 Event slow path，保持 wake-before-wait 与 final-publisher race 语义。

对这两个改动做了一次预声明的 bounded interleaved A/B 后，四格 candidate/peer ratio 仍为
`0.968 / 0.938 / 0.967 / 0.924`，因此没有启动 qualified `n=15`。codegen 同时证明强制内联虽消除 call，
却使 paired executable 增大 `1024 B`、`task_scope.obj` 增大 `3498 B`；它不是可靠的产品优化。

随后补做 peer/API 预审发现更早的合同错误：HYLE 文档与测试承诺重复、并发多个 `join()` waiter；
Folly `AsyncScope::joinAsync()` 明确只允许一次 join，Tokio `JoinSet` 的 join 需要独占 `&mut self`，当前
推理 request/model-load/render frame 纵切也只有一个 lifecycle owner。manual peer 同样只有一个 owner。
HYLE 的额外承诺使所有 completion record 必须保留到 scope 析构，而不能由 owner 在终止等待时逐项释放。
这不是同合同 peer 的必要成本，而是缺少消费证据的 API 过度设计；应在采样前由合同审计发现。

### 有限改动集

| ID | 已证实问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `S9-01` | 每次 `add` 多一层同 TU、未内联 wrapper→template call | `REJECT`：强制内联造成可测 code-size 膨胀，保留编译边界 | 不再为微观点值强制 inline |
| `S9-02` | `wait_` 对 publisher 已释放的 record 仍进入 Event lock/wait | 以现有 refcount 的 acquire load 提供 `publisher_released()` 快路径；值为 2 时保持原 `Event::wait()` | 无新字段；已完成 record 不调用 Event，未完成/final-publisher pause 行为不变 |
| `S9-03` | 无 consumer/peer 依据的重复/并发 join 承诺阻止 owner 及时释放 record | terminal join 改为 single-owner、single-shot；claim 后 detach 稳定 list，逐项等待并释放 scope reference；重复或并发 join 返回 typed `FailedPrecondition` | concurrent add/close/cancel、首错/full-drain/final-publisher race 不变；一次 owner join 后不保留全部 child record |

没有第四项候选。尤其不拆 completion allocation、不引入 shared counter、不把 driver
释放移到远端 worker，也不增加 batch/fast API。实现 `S9-02/S9-03` 后只跑 focused correctness、生成代码和一次
bounded interleaved A/B；若仍有 material signal，按卡停止，不继续猜测。

## API / usability freeze

公开调用名称沿用 v8：无逐 child 结果时使用 `add`，需要 typed result/独立取消时使用 `spawn`，最后由
唯一 lifecycle owner 显式调用一次 `join` 或 `cancel_and_join`。同步 admission failure 不转移 ownership；
已 admission 的失败由 scope 在 full drain 后聚合。重复或并发 terminal join 返回
`Status::FailedPrecondition`；concurrent admission/close/cancel 仍受支持。`remaining()` 仍是 O(N)
control-plane snapshot。修正不增加 benchmark-only 入口或常态 metadata。

## Candidate

- product candidate：`d32f4fc465edc4b6b6a450b1cd14cdc08383f5a7`；
- `S9-02` 复用已有 completion refcount 的 acquire observation，publisher 尚未释放时仍进入原 Event；
- `S9-03` 在 admission/close 同一控制锁上 claim single-shot terminal owner，detach 稳定 record list，并在
  观察每个 child 后立即释放 scope reference。重复或并发 terminal join 返回 `FailedPrecondition`；
- TaskScope focused correctness/death suite 为 `31/31 PASS`，其中新增 completed-record 即时释放与两个
  external joiner 只产生一个 terminal owner 的断言；final-publisher pause、concurrent admission/close、
  cancellation、首错/full-drain 与 fiber-aware join 继续通过；
- MSVC 19.50 Release codegen 在 wait loop 中形成 refcount acquire load → `cmp 1` → conditional
  `Event::wait()` → release RMW；没有强制内联 `task_scope_add`。两个 paired executable 均为
  `315392 B`，未重现 S9-01 的 `1024 B` executable 膨胀；
- 唯一 bounded interleaved preflight 的 HYLE/manual `op/s` ratio 为
  `1.056 / 1.018 / 1.095 / 1.051`（primary/placement × candidate-first/peer-first）。四格均不再固定输，
  因此允许签发 candidate；这些短样本仍是 `DIAGNOSTIC`，不冒充最终 qualified P1。
