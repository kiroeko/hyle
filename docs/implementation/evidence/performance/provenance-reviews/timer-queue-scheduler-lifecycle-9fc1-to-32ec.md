# Timer-queue scheduler consumer provenance renewal

<!-- PERFORMANCE PROVENANCE REVIEW: NON_MATERIAL -->

本记录只审定 `timer-queue-consumer-v1` 的既有 engineering replay 能否在 Sched consumer
发生非实质字节变化后继续表示同一个冻结消费命题。它不修改旧 manifest、contract、采样点值或
Timer-queue 产品 candidate，不替代 Formal qualification，也不重新裁决 Sched。

## 精确身份

- contract：`timer-queue-consumer-v1`；semantic SHA-256
  `48e636cb9e59d578b1148ef0af091cb04b6daeb38097581b6e315b9f1eb760fe`；
- workload：`scheduler-timer-lifecycle-64`；
- provenance path：`src/sched/scheduler.cc`；
- frozen SHA-256：`9fc1b77d51b70b701962418ca5b3003244b9828ac676fd5e9e33a49716842ba3`；
- reviewed current SHA-256：
  `32ec50841311ca5db0a67f735582bcf1a49c4abdc605ae460b4d636808700ef8`；
- Timer-queue product candidate：`381eb4b195973ed35ed23ca63c05f9605a132f9b` / tree
  `a5d48a259161c1a60e19450e42b4a6eac9a1a712`。

## Materiality review

对 candidate 到当前树执行：

```text
git diff --word-diff=porcelain \
  381eb4b195973ed35ed23ca63c05f9605a132f9b..HEAD -- \
  src/sched/scheduler.cc
```

结果只有 17 处同一词汇迁移：`mem::Allocator` 改名为 `mem::AllocatorRef`。构造、保存、传递和
deallocate 的 allocator 语义未变；没有修改任何 timer field、call site、deadline、callback、ready
list 或 worker loop。逐项核对冻结 workload 后，下列事实保持不变：

- 每个 worker 仍直接拥有一个 single-owner `time::TimerQueue`；
- 64 个 caller-owned、地址稳定的 `TimerEntry` 及 callback/data/owner metadata 形状未变；
- arm、45 次 cancel、64 次 monotonic advance、19-node detached ready-list drain 与 callback/data
  observation 的操作序列和全部 charged phases 未变；
- `fire_due_timers` 仍先检查 empty，以 `monotonic_ns()` advance，再在同一 worker drain ready list；
- sleep 和 timed-I/O 的 arm/cancel 仍在 owning worker 上执行；算法、锁、分配、atomic、syscall 和
  material-contribution 账单未变；
- Timer-queue 三个公开 header、测试、benchmark 和 contract 相对 candidate 零 diff；中央图中的
  `timer_queue -> PUBLIC {core,time,container_intrusive}`、Sched 的 private direct edge，以及
  `hyle::time -> {time,timer_queue}` façade root 均保持不变。后来新增的 Platform/Mem factor 边不改变
  Timer-queue delivery 或本 replay 的计费边界。

因此 `9fc1b77d... -> 32ec5084...` 对这个**精确 contract/workload**裁决为
`NON_MATERIAL / ENGINEERING_REPLAY_ONLY`。既有八个 strata 的统计结果仍对应同一 Timer-queue
consumer 命题，无需重新采样；这不产生当前 Sched 性能结论。

## 边界

- retained manifest 与 contract 保持原字节；账本只登记精确旧 hash 到新 hash 的兼容边；
- Formal evidence 不接受本 renewal；发布级证书仍需当前 provenance 的新 qualification；
- `scheduler.cc` 再次变化时，本边不再到达当前 hash，必须重新做有限 materiality review，或在
  变化涉及 ownership、charged phases、contribution、algorithm、delivery 时冻结新 contract 并重跑；
- 本裁决没有发现 Timer-queue 的 product reopen trigger，因此 registry 的 `CLOSED` 不变。
