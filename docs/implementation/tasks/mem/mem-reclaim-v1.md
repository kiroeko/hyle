# Mem reclamation v1：显式 domain 的 reader-safe 回收

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（SCOPED）

- component / product / round：`mem_reclaim` / Mem / `mem-reclaim-v1`；
- task kind：`NEW`；baseline commit：`31b86dd809ec61cd1fb7c42ed986e2697d20cb91`；
- trigger：全库能力因子化的 `MEM-A-32`。`ConcurrentNodeHashMap` 为避免 reader/table publication race
  把所有旧 bucket table 保留到析构，Obs 又私有实现 snapshot reader epoch；这两处说明 HYLE 缺少唯一的
  reader-safe lifetime owner；
- real consumers：read-mostly registry/config snapshot；并发 map 扩容后的旧 table；Obs sink snapshot。
  这些 consumer 都先发布 immutable 或 externally synchronized state，再在短 read region 后延迟回收；
- scope：显式 `ReclamationDomain`、move-only read guard、零分配 intrusive retire record、typed pointer
  protection helper、nonblocking collect、blocking grace-period synchronization、externally-quiescent drain、
  pending/reclaimed observation，以及当前直接消费者的适用迁移；
- non-goals：intrusive membership、容器/日志策略、lock-free 容器、Sched executor、自动为每次 retire 分配、
  完整 hazard-pointer family、GPU fence retirement、C/Native FFI、稳定 C++ ABI、Formal 或 universal SOTA；
- dependencies：`core` only。回收 callback 在触发回收的线程同步执行；factor 不依赖 Mem allocator、Platform、
  Sched 或 Container，避免内存生命周期底座反向成环；
- direct consumers：候选为 `container` 与 `obs`；只有迁移后仍保持其既有并发/回调合同且通过自己的直接
  测试，才建立真实 edge；不能为了证明本 factor 有用而强迫热路径付费；
- performance tier：`P1`。read guard 是潜在 registry/container 热路径；retire/collect/synchronize 是冷写侧；
- material margin：同合同 warmed read-side route 对锁定 Folly RCU 不劣于 5%；若 peer 无法在 HYLE 独立
  package 中形成可资格化 same-contract 构建，则以固定 API 的原子/分配/锁结构、HYLE direct-consumer
  holdout 和明确 `INCONCLUSIVE` 停止，不换 workload 追点值；
- resource budget：guard acquire/release 不分配、不锁、不 syscall；每个 guard 至多两次 reader-counter RMW
  加验证 load，嵌套 guard 每个独立计费；retire 不分配，常态一次短 queue lock；record metadata 只由 opt-in
  retire owner 支付；无 reader 时 collect 为 O(R)，有 reader 时 nonblocking 返回；
- investigation / modification budget：一个 API audit、一个实现、一次 correctness 修复；P1 最多一次
  qualification、一次机制诊断、一次产品重跑；禁止因亚纳秒、顺序或地址翻转改变 public layout；
- machines：`W1-WIN` correctness/P1，`W1-WSL` 与 `M1-MAC` correctness/sanitizer/交付；instrument：
  `closure-perf-v1`；
- strong peers：Folly `176c088840724715b2c3c6211b2b347611676002` 的 RCU/hazptr 与 concurrent-map
  consumption；oneTBB `b03a694ee0fe4b188bc77daa8a567551e3e3ebfc` 只作 concurrent-container lifetime
  反向检查；llama.cpp `b96806d96061049a5b574269b049bf6241d63d46` 与 ggml
  `d4716378882593333721eb33f153144b6885caf2` 用于确认推理 runtime 的实际并发 publication/retirement
  需求，不按名字复制 API；
- required gates：scope、architecture、correctness/sanitizer、source/static/shared C++ delivery、直接
  consumer、guide/误用诊断、red-team、P1 engineering；Native FFI `NOT_APPLICABLE`；
- stop：有限 FIX 全部完成才进入 candidate。若正确实现必须依赖未完成的 Platform/Mem split、同合同 P1
  无法资格化或迁移会让现有正常 consumer material 回退，则 `BLOCKED` 或保留显式 opt-in，不扩写通用
  并发框架。

## 初始能力分类

| family | current HYLE | normal consumer need | preliminary verdict |
|---|---|---|---|
| read-side lifetime guard | Obs 私有、Container 无通用 owner | 短 guard 后安全读已发布 pointer | `MUST / AUDIT` |
| intrusive deferred callback | 各模块私有保留/计数 | 不为每次 retire 分配，支持 allocator-aware owner callback | `MUST / AUDIT` |
| grace-period synchronize | 私有 drain 或保留到析构 | 等待调用开始前已存在 readers | `MUST / AUDIT` |
| nonblocking collection + pressure | 缺失 | writer 可尝试回收且观察 pending | `MUST / AUDIT` |
| default process domain | 缺失 | 普通小型 publication 的便捷 route | `MAY / AUDIT`，需检查 image/lifetime 成本 |
| hazard pointer slots/cohorts | 缺失 | stalled reader 独立回收单对象 | `MAY / DEFER`，当前 consumer 未证明必须 |
| async callback executor | 缺失 | 将析构移到指定线程 | `ROUTE` to Sched/owner composition |

SCOPED 阶段只冻结问题、预算和候选 family。进入 `AUDITED` 前必须完成 public-surface/direct-consumer 反向
追踪、peer borrow/reject 表、完整 API 草图、ownership/failure/concurrency/cost 合同和有限 FIX ledger；
此时不得写产品实现或运行 P1 计时。
