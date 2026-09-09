# Mem v7：动态逻辑区间子分配

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner：`mem` / payload-free logical range ownership；round：`mem-v7-logical-range-suballocator`；
  baseline：`67893204aad888f8e801c59ec6c149432a8e779e`。
- trigger：`MEM-V7-CROSS-DOMAIN-LOGICAL-RANGE`。2026-09 多消费者纵切审计确认，现有
  `FixedHeap` 需要真实 host pointer 并把 block metadata 放在 payload backing 中，`LifetimePlanner`
  只处理预先已知的 interval；两者都不能管理 opaque backing 中运行时变化的 `[offset,size)`。
- 主纵切：异构渲染/资源流送。Cycles device sub-pointer/delayed-free、AMD VMA/D3D12MA virtual block
  都要求先在大块 device backing 中取得 aligned logical range，再由 Compute/provider 建资源或提交 copy。
- 正交纵切：推理 serving。SGLang `RangePageAllocator` 与 pinned-buffer buddy allocator、vLLM block pool
  都重复使用动态 page/byte offset 分配；KV hash、eviction 和 request policy 不进入 Mem。
- 当前锁定来源：SGLang `dcebe8c4733a`、vLLM `51da0ca66c80`、Cycles 无 Git identity 的本地快照仅作
  辅助；OffsetAllocator `3610a7377088b1e8c8f1525f458c96038a4e6fc0`、VMA
  `3aa921224c154a0d2c43912bc88e1c42ce1f7607`、D3D12MA
  `1d86c1130f61453634b1df85782e1fecfd59a525` 为成熟算法/API 参考。正式证据位置见
  [跨产品审计](../project/public-api-consumer-portfolio-audit-2026-09.md#f01--mem-动态逻辑区间子分配确认)。
- 正常任务：创建一个固定 logical capacity 的 metadata owner；按不同 power-of-two alignment 动态取得
  contiguous range；以不可伪造的 allocation identity 归还；观察常数时间摘要；整块 reset/reuse。
- scope：payload-free owner、64-bit capacity/offset/size、显式 metadata allocator 与容量、aligned
  allocate、individual free、bulk reset、stale/double/wrong-owner 诊断、O(1) summary stats、move/destroy、
  单 owner/caller-synchronized 合同，以及 Mem 产品/三种 C++ 交付和用户选型文档。
- non-goals：分配真实 host/device memory、device/stream/fence、buffer-image granularity、defrag/move
  callback、KV/texture/descriptor policy、并发 wrapper、linear/ring/double-stack strategy、C/Native FFI、
  Compute/Graph、公共性能仪器或现有 `FixedHeap/LifetimePlanner` 重写。
- 直接依赖：现有 Mem 的 Core/Error/Platform 与 `AllocatorRef`；不得为 metadata 反向依赖 Container。
  直接消费者：未来 Compute device heap、推理 cache/backing owner、渲染 resource allocator；本轮用独立
  public C++ probe 证明，不修改仍为 `QUEUED` 的 Compute/Graph。
- 性能 tier：`P1`。P0 先裁决复杂度、metadata bytes、allocation/copy/lock；P1 primary 是固定种子的
  mixed resource churn，在 alignment=1/32-bit 同合同交集对 OffsetAllocator，holdout 是变化 alignment
  与大于 4 GiB capacity 的 HYLE correctness/resource trace。默认 5% non-inferiority margin；VMA/
  D3D12MA 只在能隔离相同 virtual-block 合同和依赖身份时才进入计时，否则仅作算法/API 参考。
- 仪器/机器：`closure-correctness-v1`；P1 使用 `closure-perf-v1` 的 W1-WIN 锁定身份。正确性先覆盖
  W1-WIN；产品候选再覆盖 W1-WSL 与两台可用 macOS 中至少一台，另一台只在平台差异或前者不可用时
  使用。文档/状态改动不触发 P1。
- 调查预算：能力/API/算法最多 3 小时，性能调查最多 90 分钟；至多两次产品纠正、一次反序复现和一次
  codegen/profile 归因。范围变化退回 `SCOPED/AUDITED`，不能增加 benchmark-only 策略开关。
- 预声明验证：public-header self-contained；zero/invalid/overflow/OOM；alignment；fragment/coalesce/reset；
  stale/double/wrong-owner；metadata 上限；随机 model/property；move/destroy；ASan/UBSan，TSan 仅在实现
  引入共享状态时适用；Mem focused tests；source/static/shared public consumer；direct consumer probe；
  七问、独立红队、P1 和固定治理命令。
- stop / blocker：有限 `FIX` 全部完成且七门通过则重新 `CLOSED`。若 64-bit aligned range 的正确性、
  metadata bound 或同合同 P1 在预算内不能裁决，进入 `BLOCKED`；不能用弱化语义、无限 benchmark
  姿势或修改现有已验收 allocator 来绕过。

## 待完成的 `AUDITED` 输出

本阶段不授权产品实现。下一状态必须先给出：完整 capability/public-surface map、VMA/D3D12MA/
OffsetAllocator 的语义与算法取舍、冻结 C++ 调用草图、有限 `FIX` ledger、P0 成本上界和实际构建/测试
命令。F02/F03 不与本任务合并。
