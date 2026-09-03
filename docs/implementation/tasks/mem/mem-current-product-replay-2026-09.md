# Mem 当前全产品能力回放 — 2026-09

<!-- READ-ONLY CAPABILITY REPLAY: ACTIVE -->

> 本页是 product-non-mutating replay，不是第二套模块状态。Mem 在发现满足收口协议的具体
> `PRODUCT_TRIGGER` 前仍以 [`module-status.json`](../../module-status.json) 中的 `CLOSED` 为准；本页
> 不授权修改公开 API、算法、布局、依赖或正常 workload。

## 回放合同

- 对象：普通用户看到的 `hyle::mem` 产品，以及内部 `mem`、`mem_vm`、`mem_reclaim` 三个 factor 的
  累积合同；
- reviewed rounds：`mem-v2-rebuild`、`mem-v3-vm-factorization`、
  `mem-reclaim-v2-inference-resume`、`mem-v4-reclamation-facade`；
- reviewed current registry identity：Mem v4 candidate
  `0f6ef8e83ee290ab565c0d53081a0acb75a46675`，end
  `5a51b909031a3efd18288f84aa573549f5a4acaf`，anchor
  `b9fdec1a404aa05f0e63cd214896b05c09c59444`；
- replay baseline：HYLE `911e4ce61adc924e2261b622ca98461b2734a4d5`；
- reviewer / date：Codex / 2026-09-03；
- owner request：停止过早启动的 IO 审计，从 Mem 重新验证当前累积产品，而不是把历史窄
  `CLOSED` round 自动当成对全部 Mem 能力的今日回答；
- 总预算：只读能力/API/依赖/证据审计最多 4 engineering hours；性能计时预算为 0。若发现产品触发器，
  后续新 closure card 单独冻结实现与 P1 预算；
- verdict：`IN_PROGRESS`，终止时只允许 `CONFIRMED / METHOD_GAP / PRODUCT_TRIGGER`。

## 锁定输入

| Source | Clean identity at replay start | Role |
|---|---|---|
| llama.cpp | `c7bda030e7faee594dbe7550185e857351ad405d`；随仓 ggml tree `25e2241ed891c98cd26af5c47246c79b430a49cd` | 当前模型加载、buffer、KV/推理生命周期真实消费形状 |
| standalone ggml | `d4716378882593333721eb33f153144b6885caf2` | 当前基础 buffer/resource/allocation planning 能力 |
| mimalloc | HYLE 产品锁 `v3.5.0` / `18b08671c9302247bfb682286e6bf3cc1773f801`；上游 clean `c683b7c60c91cb2809977e37d3f4d084d7497180`，含 `v3.5.1` | host heap mechanism、版本影响与交付 |
| oneTBB | `843162db8cf4b2531a4a6147b6ea3fd860d44668` | scalable allocator、memory pool 与并发生命周期 peer |
| Folly | `80071a1f79e110b34a2cf7a3a4d01a59e823d697` | arena、mapping、reclamation peer |
| CCCL | `b7aaea69a2b07e50f09e67f2962da0243e0b7c5d` | property-bearing host/device/pinned/managed resource 与 buffer peer |

每个 capability family 最多使用上述两个 strong peers；llama.cpp/ggml 是强制 consumer/foundation
差异输入，不因“peer 数”被排除。Boost 工作树有未提交修改，本轮不把它登记为干净 peer identity。

## 必须回答的范围

1. **功能超集：**从最新 llama.cpp 真实 journey、standalone/随仓 ggml、定义性 C++/OS 能力和成熟 peer
   建立 family map；逐族审取得/构造、正常操作、组合/互操作、释放/trim/回收、边界、ownership、
   concurrency、failure、复杂度与交付。不得按 API 数量求并集。
2. **自然产品形状：**反向追踪全部公开 Mem 入口，识别 benchmark-only、半生命周期、错误 owner、
   命名/默认值/失败形状不自然和需要用户跨内部 factor 拼装的路线。
3. **质因数与依赖：**检查 `mem/mem_vm/mem_reclaim` 拆分是否足够小且无环，是否正确复用
   Core/Error/Platform/Platform-fs/Util-core，直接消费者是否绕过 Mem 私造 allocator、mapping、buffer、
   reclamation 或 placement 机制。
4. **第三方：**区分 HYLE semantic owner 与 mimalloc/OS/peer mechanism；检查 v3.5.1 相对产品锁是否
   改变实际源码、语义、算法、codegen、许可证或交付，不能因“有新版”自动升级或继承旧证据。
5. **性能：**先审复杂度、allocation/copy/syscall/atomic/lock、常态 metadata 和旧 comparison class。
   本 replay 不运行 benchmark；若有触发器，新轮先冻结 API/usability，再定义一个拟真 primary 与至多
   一个 holdout，同合同 P1 和 natural-route 分开。
6. **交付与文档：**核对 source、installed static/shared、selected roots、外部依赖锁/离线/许可证，清理
   旧 C ABI/Native FFI 叙事；architecture 解释边界，user guide 按场景告诉用户选什么，performance 只
   保留候选绑定的有限结论。

## 初始 capability-family 审计队列

- host system/aligned allocation、reallocation、STL/PMR adapter 与独立 growable heap；
- monotonic/stack/request scratch、fixed/variable/typed/concurrent pool、quota/stats/trim；
- virtual/file/shared mapping、reserve/commit/protect/advice/residency/huge/NUMA/guarded stack；
- resource properties、Buffer owner/view/import/composite、host/pinned/device/managed provider seam；
- sync/async copy/fill/map/flush/invalidate 的 owner 边界；
- transient pooling、capacity query、interval/lifetime packing 与 worst-case reserve；
- read-mostly publication/reclamation 及其与 Container 的直接 consumer；
- OOM、overflow、alignment、zero-size、move/destroy、跨线程 free、callback/quiescence 和 provider unload；
- 三种 C++ 交付、公开头闭包、第三方类型泄漏、旧 FFI 残留和用户选型可发现性。

## 停止与下一状态

- 全部当前 `MUST` 已自然 `COMPLETE/ROUTED`，产品 identity 与适用证据未被击穿：`CONFIRMED`；
- 没有产品缺陷，但旧任务/仪器不足以回答当前问题：`METHOD_GAP`，Mem 仍 `CLOSED`；
- 存在可复现缺失 journey、错误 owner/生命周期/复杂度/交付，或受影响第三方升级事实：
  `PRODUCT_TRIGGER`，记录 trigger ID 与最小影响面，然后才建立 `REOPENED -> SCOPED -> AUDITED` 新轮。

当前只允许完成 public-surface trace、consumer/peer 差异、依赖/交付扫描和 capability ledger；不得先写
产品代码，也不得因为旧 benchmark 某行输赢改变 API。
