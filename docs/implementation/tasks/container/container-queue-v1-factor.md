# Container queue v1：并发传递存储因子

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

## Closure card（`SCOPED`）

- component / owner / round：`container_queue` / Container /
  `container-queue-v1-factor`；
- baseline：`b5568e929787ecd5bb222e436c7456c6517fae21`；触发与产品总合同见
  [Container v5](container-v5-product-factorization.md)；
- scope：接管 byte ring、SPSC、bounded MPMC、sharded、work-stealing 和 owner/thief FIFO queue 的
  header ownership、tests 与 Sched direct edge；裁决 unbounded typed handoff 与 priority storage；
  建立独立 target/façade；冻结 capacity/OOM/role/progress/quiescence 和 P1 正常 handoff；
- non-goals：condition variable、fiber suspension、deadline/cancel/close/backpressure service、request
  scheduling policy、IO completion、Native FFI/C API、稳定 C++ ABI、公共仪器和 Formal；
- candidate dependencies：PUBLIC `core,mem`；wait/clock/reclamation 只有真实算法合同需要时才可进入，
  不能让 try-only storage 反向取得 Sched；
- tier / machine / margin：P1，`closure-perf-v1`，W1-WIN，5%；W1-WSL/M1-MAC correctness 与交付；
- budget：审计 45 分钟、实现与测试 2.5 小时、性能 45 分钟、终态红队并入 Container suite；最多一次
  performance-guided product correction；
- next：完成全部 queue public trace、unbounded/priority owner 裁决、普通调用草图、语义等价矩阵和有限
  FIX ledger 后进入 `AUDITED`；此前不修改产品。

## Audit verdict（`AUDITED`）

- public trace：byte ring、SPSC、bounded MPMC、sharded queue、work-stealing deque 与 owner/thief FIFO
  headers/tests 归本 factor；`RingDeque/CircularBuffer` 是单线程 sequence，归 `container`；
- 现有各 queue：`COMPLETE/KEEP`。它们分别承诺 byte segments、single-role、bounded lock-free、sharded
  contention 与 owner/thief semantics；不能靠一个 `fast/light` 总开关合并；
- unbounded typed handoff：`MISSING/FIX`。普通形状是 `try_emplace`、callback `try_pop`、`empty` 和
  approximate size；allocation failure 可恢复，等待/close/cancel/deadline/backpressure 不在对象内；
- ordinary/priority：单线程 deque/heap `ROUTED` 到标准算法；concurrent priority `DEFER`，直到真实
  scheduler/engine 同时需要 parallel priority mutation；
- dependencies：PUBLIC `core,mem`。不依赖 `time/platform/mem_reclaim/container/sched`；Sched 反向以
  PRIVATE direct edge 使用本 factor；
- P0：bounded/SPSC/work-steal 保持既有 O(1) 与 steady no-allocation；unbounded enqueue/dequeue expected
  O(1)，增长只在 block boundary 分配，已出队 block 安全回收。P1 primary 为 burst MPMC handoff，先签
  capacity、ordering、allocation、progress、producer-token 与 setup 等价；5%；
- finite FIX：`CONQ-F01`（独立 target/header/test owner、unbounded MPMC、Sched/package edges）和
  `CONT5-F06`。不在本轮增加 blocking adapter 或 concurrent priority queue。

## Candidate（`CANDIDATE`）

- product commit / tree：`31c3ace558a821902d689d2d49c7bf8a102d5beb` /
  `b10d2261aecbcac34ad8d77a7fa8adc25e78bf9c`；
- `hyle_container_queue` 独立拥有全部 queue storage，PUBLIC closure 精确为 `core,mem`；新增
  `UnboundedQueue` 以固定 block 摊薄分配，分别发布 producer/consumer monitoring counts，等待、close、
  cancel 与 backpressure 仍不进入 storage；
- W1-WIN shared/static normal、empty、初始/增长 OOM、跨 block FIFO、nonassignable callback 与 4P/4C
  conservation stress 通过；四生产/四消费 oneTBB SAME_CONTRACT P1 等待 clean-candidate qualification。

## 第二 Candidate

首候选因共同产品的 Text provider 正常路径修正撤回；本 factor 的 API、layout 和算法
未变。新 product commit/tree 为 `a7ce0102413c783e26227c68645e0282b109fde4` /
`6ae52a087b8b2b9271bbd98acb9d7c62bb61d10c`，queue P1、correctness 和 delivery 必须在该身份上
重放。

## 第二候选撤回

共同 `hyle_container` shared artifact 泄漏 private bitset ISA symbols，因此四 factor 一并退回
`AUDITED`。终局红队确认 `UnboundedQueue` 的 producer allocation 与 consumer deallocation 可以重叠，
而公开合同没有要求 injected allocator 支持并发调用；有限修复只补 header/guide 合同。正式 queue 对拍中，HYLE 使用 mimalloc-backed `SystemAllocator` 与
recoverable `try_push`，oneTBB 使用 `cache_aligned_allocator` 与 throwing `push`；benchmark adapter
没有等价化 allocator、OOM 和 rollback，故 `1.23–1.25x WIN` 只是一条 `NATURAL_ROUTE` 观测，不能
签实现级 `SAME_CONTRACT` 或 SOTA。

## 第三 Candidate

共同 product commit/tree 为 `6fae3c11ebce9a0741f45780e63c9896e08bb2a2` /
`8025ee111380c2600f804cf979ed59d289a8be76`。本 factor 只补 `UnboundedQueue` injected allocator 的
producer/consumer 并发前提；实现、layout、默认 allocator 与正常路径成本不变。W1-WSL GCC 14
shared/WERROR tests/header tests 2/2 通过。

## Terminal verification and decision

精确共同候选 `6fae3c11ebce9a0741f45780e63c9896e08bb2a2` 已通过 Windows static/shared、WSL
ASan/UBSan/WERROR、macOS UBSan/WERROR，以及三平台 source/static/shared relocated consumer。终局红队
确认全部 role-specific queue、unbounded block lifetime/FIFO/conservation/OOM、公开 allocator 并发前提、
依赖和选择文档均通过，未发现新的产品 defect 或把 wait/service policy 塞回 storage。

本 factor 从 `CANDIDATE` 终止为 **`BLOCKED`**：scope、architecture、correctness、delivery、
usability/docs、red-team 均 `PASS`；engineering performance=`INCONCLUSIVE`。第二候选的自然路线曾显示
HYLE/oneTBB `1.2296–1.2538`，但 HYLE 使用 mimalloc-backed `SystemAllocator` 与 recoverable
`try_push`，peer 使用 `cache_aligned_allocator` 与 throwing `push`；这不是 `SAME_CONTRACT`，有利数字也
不能冒充 SOTA。

- blocker owner：`container-v5-p1-contract-evidence`；
- resume condition：新 `RESUME` round 等价化 allocator/resource、OOM、push result、setup 与
  producer-token 后，在 immutable candidate、W1-WIN、5% margin 下取得合格 verdict；或另行预冻结且只
  声称端到端默认路线的 `NATURAL_ROUTE` budget；
- known limitations：自定义 allocator 必须支持 producer allocation 与 consumer deallocation 并发；析构
  要求外部 quiescence；try-only storage 不提供 wait/close/cancel/deadline/backpressure；当前没有同合同
  P1、全面 SOTA 或 Formal 结论；
- reopen triggers：真实 consumer 暴露 correctness/security/lifetime/API/能力缺口；queue 的公开合同、
  layout、progress、allocator、依赖或三交付改变；强 peer 证明 material regression 或更优复杂度档。

LOC 绑定共同 immutable candidate：全库 464,718；自实现包含测试 121,614；自实现不包含测试 77,078；
`container_queue` 实现 1,101。

<!-- MODULE CLOSURE RECEIPT: BLOCKED -->
