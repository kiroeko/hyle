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
