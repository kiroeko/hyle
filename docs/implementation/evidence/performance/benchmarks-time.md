# Time / timer_queue 当前工程性能证据


> 当前状态与精确证据绑定只看 [`module-status.json`](../../module-status.json)。`time` 与
> `timer_queue` 是两个 factor：当前 Time v5 是 P0，证据直接位于
> [`time-v5 task`](../../tasks/time/time-v5-interoperability.md)；本页的 P1 数值只签
> `timer_queue` v1。旧 Time v4 host-clock P1 receipt 不续签成 Time v5 的当前 verdict。

## Time v5：P0 结构结论

signed Unix units、absolute/relative POSIX、timeout rounding、direct FILETIME 与 host-clock value
operation 都是 O(1)、零分配/锁/syscall 的纯转换或算术；native clock read/sleep 本身各落到相应 OS
机制。当前没有 profile 证明新增互操作进入热点，因此 timing peer comparison 为 `NOT_APPLICABLE`，
结构成本为 `PASS`。这不宣称 Time 比标准库或任意 peer 更快。

## timer_queue v1：同合同 P1 `WIN`

冻结合同 [`timer-queue-consumer-v1.json`](../../../../bench/contracts/timer-queue-consumer-v1.json) 使用
Scheduler N=64 完整 lifecycle：64 次 arm、45 次 cancel、64 个 advance tick、19 个 delivery 与 callback
state observation。candidate `381eb4b195973ed35ed23ca63c05f9605a132f9b` / tree
`a5d48a259161c1a60e19450e42b4a6eac9a1a712` 与 Seastar
`991cf9a908482a3f11ca2fc34a1b3970a06d8622` 的 faithful `timer_set` port 使用相同 caller-owned stable
node、ownership、输入、结果、计费和 teardown；不拿完整 cancel/drain 语义对比轻量单操作。

W1-WIN、MSVC 19.50、Release/static、portable ISA、LTO off，n=15 个独立进程已经裁决。八个
primary/placement × execution/allocation-order strata 的 median advantage ratio 为
`1.1815 / 1.1293 / 1.1338 / 1.1678 / 1.1754 / 1.1898 / 1.1758 / 1.1740`；最低 99.26% CI
lower 为 `1.0911`。八项均取得 exact `WIN` 和 5% non-inferiority `PASS`。

该结论只覆盖冻结 scheduler lifecycle：65-bucket radix queue 的 arm/cancel/next O(1)，advance 按
touched buckets + delivered/rebucketed nodes 计费；正常路径零分配/lock/atomic/syscall。它不外推为所有
timer cardinality、clock、sleep、`CommonTimeoutQueue` 或整个 Scheduler 的 SOTA。

## 证据身份

A/A、预埋 5%/10% known-delta、batch-matched timer floor、primary/placement、artifact/build、consumer
provenance 和 semantic/resource validator 全部 `PASS`。deep verifier 返回
`QUALIFIED / EXACT / CURRENT / paths_verified`。

当前 canonical Sched consumer 相对冻结 source 只发生过已审的 allocator 词汇迁移；timer ownership、
arm/cancel/advance/drain、计费、算法和交付没有变化。精确 renewal 见
[`provenance review`](provenance-reviews/timer-queue-scheduler-lifecycle-9fc1-to-32ec.md)，只续接同一工程
命题，不升级为 Sched 或 Formal 结论。

retained manifest：`C:\hyle-scratch\time-v3-timer-perf-evidence\manifest.json`。manifest、summary、
controls、observations SHA-256 分别为
`B7D23B40DB8EE3D5B77CB0CD6EC7B6F5F5810F4BBA609A1B92BF605FBCE0FB90`、
`2C2D986359FCB43725850EEBCF9F352D11465D7B4D9CDE3ADD1D22EDC33AF124`、
`9BA1CF2D4DC6EADB478A5943A2DCE27079E1995F4484D34098BA887B78DCD2D5`、
`5F36C2DBC4346F0765B2A9373313EA70F853E5ED773FEAF4C10BE8A445D2A52C`。

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/timer-queue-consumer-v1.json `
  --manifest C:\hyle-scratch\time-v3-timer-perf-evidence\manifest.json
```

`ManualClock`、runtime clock ref、coarse/profiling/CPU clock、blocking sleep、native conversion 与
`CommonTimeoutQueue` 仍按各自复杂度、资源、correctness 和交付门裁决。只有真实 profile、同合同强 peer
或正常用户任务暴露 material regression 时才建立新的 P1/P2；亚纳秒点值或顺序/地址翻转不改变产品。
