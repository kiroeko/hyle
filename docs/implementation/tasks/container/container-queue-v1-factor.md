# Container queue v1：并发传递存储因子

<!-- MODULE CLOSURE RECORD: ACTIVE -->

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
