# Sched v6：pinned、joinable、typed structured task

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner：`sched` / structured task admission and lifetime；round：
  `sched-v6-pinned-structured-task`；baseline：
  `cb87f5028b7f6bf8558054337de59538bb441f56`。前一轮当前范围由
  [Sched v5 terminal](sched-v5-io-completion-search.md) 与其
  [工程性能证据](../../evidence/performance/benchmarks-sched.md)固定；它们是本轮 baseline 事实，不授权
  重开 idle/IO-completion 工作。
- trigger：`SCHED-V6-PINNED-STRUCTURED-TASK`。跨产品公开 API 纵切已经证明
  `Scheduler::spawn_detached_on` 只能得到 pinned fire-and-forget，而 `spawn_task` 只能得到可 join 的
  typed task；两者没有交集。`samples/08_inference_service/main.cc` 因而手写 `launch_pinned`，重复处理
  coroutine frame failure、result/exception、错误上报和 detached lifetime。
- 正常 consumer 任务：把可能 suspend 的 `Task<T>` 提交到指定 scheduler worker，获得
  `JoinHandle<T>`；task 从首次 dispatch 到每次 timer/IO/foreign wake 都保持同一 owner，并能 join、
  detach 和 cooperative cancel。
- 主产品纵切：推理服务的 per-worker IO completion owner 与 arena/staging state。正交反偏置纵切：
  renderer 的 per-worker upload/resource queue。两者都要求稳定 worker ownership，但 result/error policy
  仍由调用者拥有。
- 锁定输入：HYLE inference-service sample 与现有 Sched/Fiber/JoinState；SGLang
  `dcebe8c4733a5cb802fb679e0c8fbb886e2801b0`、vLLM
  `51da0ca66c8065619c79e35dff97aa99aeaf5644`、ds4
  `f62ca29a308724cde5bc99134ede19104b2a3260`、llama.cpp
  `e71b80510c848c00175924ecf3c40333ccae8eb5` 用于真实 worker/backend 生命周期回放；Seastar、Folly、
  marl 与 oneTBB 只在 `AUDITED` 阶段锁定具体 API/机制证据。
- scope：审计并提供 `spawn_task_on(scheduler, worker, allocator, Task<T>, stack_size)` 或等价自然
  C++ 形状；复用现有 `TaskDriver<T>`、`JoinState<T>`、Fiber stack pool、admission accounting、pinned
  queue/reinject 和 `JoinHandle<T>`；覆盖 typed/void、首次运行与 suspend/wake owner、join/detach/cancel、
  invalid worker、OOM/stack/admission failure、shutdown 与 source/static/shared C++ 交付。
- non-goals：新 task lifecycle、priority/QoS/deadline lane、gang/SPMD、强杀、跨 scheduler migration、
  Compute/Graph、C/Native FFI、公共性能仪器、调度队列或 idle policy 重写，以及 F03 IO file watch。
- 直接依赖：现有 Sched 的 Core/Error/Container-intrusive/Util-core/Time/Platform/Mem/Mem-vm；不得新增
  dependency edge。直接消费者：IO per-worker operation owners、未来 Compute completion/staging owner、
  inference/rendering service；本轮不修改仍为 `QUEUED` 的 Compute/Graph。
- 性能 tier：暂定 `P1`。P0 必须先证明相对现有 `spawn_task` 只改变 admission route，不增加 task/cell/
  stack allocation、result copy、atomic、lock 或 syscall。P1 只允许同一 typed+suspending+pinned 生命周期
  的真实 consumer episode；若成熟 peer 无相同语义，则用现有手写组合路线作同合同迁移基线，不能拿
  unpinned、detached 或 non-suspending task 排名。material margin 5%。
- 仪器/机器：`closure-correctness-v1`；若审计后 P1 可比较，使用 `closure-perf-v1` 的已登记工程机器。
  correctness 覆盖 W1-WIN、W1-WSL 和一台 macOS；并发/唤醒风险需要 TSan，平台不支持时明确记录。
- 调查预算：能力/API/实现审计最多 90 分钟；至多两次产品纠正、一次性能机制归因和一次正式候选 P1。
  不为了点值调整公共 API、队列容量、pin policy 或常态 metadata。
- 预声明验证：public-header self-contained；typed/void/move-only result；invalid worker；frame/state/driver/
  stack OOM；closed/full admission；first-dispatch pin；timer/foreign wake 后仍 pin；join/detach/cancel；owner
  shutdown；Sched focused tests；受影响 IO integration；ASan/UBSan/TSan；source/static/shared public
  consumer；七问、分离红队、P0/P1 和固定治理命令。
- stop / blocker：`AUDITED` 后的有限 FIX 全部完成且七门通过则 `CLOSED`。若现有 Fiber owner/reinject
  不能安全承载 typed task、同合同性能或三平台 wake ownership 在预算内无法裁决，则 `BLOCKED`；不得
  通过 detached wrapper、弱化 result/lifetime 或无限 benchmark 姿势绕过。

下一步只做能力/API/peer/现状审计并冻结有限 `KEEP/FIX/DEFER/REJECT` 集；`SCOPED` 阶段不修改产品。
