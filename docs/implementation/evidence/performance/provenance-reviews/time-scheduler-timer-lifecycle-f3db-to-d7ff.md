# Time scheduler consumer provenance renewal

<!-- PERFORMANCE PROVENANCE REVIEW: NON_MATERIAL -->

本记录只审定 `time-source-consumer-v1` 的一个既有 engineering replay 能否在消费者源文件发生
**非实质字节变化**后继续深重放。它不修改旧 manifest、contract、点值或 Time 产品 candidate，也不
替代新的 Formal qualification。

## 精确身份

- contract：`time-source-consumer-v1`；SHA-256
  `ab4223b7066e8540f15bbc6ad586ef7e37cf678feb877f408162e7d08bec2356`；
- workload：`scheduler-timer-lifecycle-64`；
- provenance path：`src/sched/scheduler.cc`；
- frozen SHA-256：`f3db97bd536954082200cde1e86966083c3542daba832d72e629877213170934`；
- reviewed current SHA-256：
  `d7ffc29eb6051cae3e723d6dfc67c6a860dbea3b4310f3d154adb282404f19a2`；
- frozen Time product commit：`0ef2544ceece030bb522857854285d280e368d36`；
- replay review commit：`b044adeed05e362e2cf812b1ec0a9c3093c08dac`。

## Materiality review

`git diff --word-diff=porcelain 0ef2544ceece030bb522857854285d280e368d36..b044adeed05e362e2cf812b1ec0a9c3093c08dac -- src/sched/scheduler.cc`
只有三处 replacement：

1. 两处注释把 platform wait API 名从 `park_timeout` 改为 `park_for`；
2. 一处调用把 `w.parker.park_timeout(timeout)` 改为
   `w.parker.park_for(time::Duration(timeout))`。

冻结 workload 的计费阶段是 64 次 arm、45 次 cancel、64 次 monotonic queue advance、19-node
ready-list drain、callback/data/owner 观察及 queue state 观察。Parker 的 OS-thread wait 不在 replay
代码或计费阶段中。逐项核对当前消费者后，下列形成 workload 的事实均未变化：

- 每个 worker 仍直接拥有一个 single-owner `time::TimerQueue`；
- `fire_due_timers` 仍以 empty gate 后调用 `advance(monotonic_ns(), due)`，再 drain 同一个 detached
  ready list；
- fiber sleep 与 timed-I/O 仍使用 caller-owned、地址稳定的 `TimerEntry`；
- timeout arm/cancel 仍在 owning worker 上分别调用 `schedule` / `cancel`；
- callback、data、owner metadata，deadline/cancel 分布和 benchmark charged lifetime 均未改变。

因此这次 hash 变化对该**精确 contract/workload**裁决为 `NON_MATERIAL`。它是 platform Parker API
迁移，不改变 TimerQueue consumer shape、贡献预算、ownership topology、算法、产品交付或旧统计结果。

## 边界

- 只允许 engineering evidence replay；Formal 必须重新取得 byte-exact provenance；
- verifier 只能接受上述 frozen hash 到 reviewed hash 的精确迁移；不存在 path、contract、workload
  或 hash wildcard；
- `scheduler.cc` 再次变化时本迁移不再到达当前 hash，必须重新审定或重跑；
- Time 公开头在后续 platform 收口中的 Windows `min/max` 宏可组合性修复是独立产品身份事实，不由
  本 provenance renewal 裁决。
