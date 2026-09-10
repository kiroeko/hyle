# Sched v1 consumer-shape provenance renewal

## Decision

`bench/sched/consumer_shapes_v1.json` 从
`ac0fa52f7f45203024b55c8ae78bae184852a7fff29535518f0cfa2975c8191a` 变为
`416f33964ae320212d5699cec7f9fb96eb1361cf1423158a3030a564ae090a5e`，裁决为
`NON_MATERIAL / ENGINEERING_REPLAY_ONLY`。

## Reviewed facts

- 唯一内容变化是 completion consumer 的源码路径从 `src/io/proactor.cc` 更新为模块拆分后的
  `src/io_core/proactor.cc`；被引用文件的 SHA-256、实现内容和 `Waker` completion spine 没有变化；
- `request-fanout-collect` 的 topology、owner、输入规模、charged phases 与源码引用逐字不变；
- `completion-wake` 的 4-worker pinned ownership、4096 episode、fresh Waker、外部 wake、resume 和
  acknowledgement 完成边界逐字不变；
- 该续签只允许既有 `sched-p1-v1` 工程回放使用，不把 IO/Core、Compute 或 Graph 状态带入 Sched 裁决。

因此这次路径迁移不改变两个 product workload 的比较类别、工作量、成本或可观察结果。
