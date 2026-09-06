# Container concurrent v1：并发索引因子

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round：`container_concurrent` / Container /
  `container-concurrent-v1-factor`；
- baseline：`b5568e929787ecd5bb222e436c7456c6517fae21`；触发与产品总合同见
  [Container v5](container-v5-product-factorization.md)；
- scope：接管现有 concurrent flat/node hash map 与 concurrent vector 的公开 header ownership、tests 和
  direct consumers；裁决并实现已准入的 flat/node concurrent set；建立独立 target、准确 dependency 和
  Container façade 聚合；冻结 accessor/publication/reclamation/OOM 与 P1 正常 registry journey；
- non-goals：queue/wait/channel、sequential hash implementation、全局 registry policy、snapshot
  transaction、Native FFI/C API、稳定 C++ ABI、公共仪器和 Formal；
- candidate dependencies：PUBLIC `core,mem,mem_reclaim,container`；只有审计证明同合同才增加
  `util_core`，不得复制 reclamation、hash 或 CPU authority；
- tier / machine / margin：P1，`closure-perf-v1`，W1-WIN，5%；W1-WSL/M1-MAC correctness 与交付；
- budget：审计 45 分钟、实现与测试 2 小时、性能 30 分钟、终态红队并入 Container suite；最多一次
  performance-guided product correction；
- next：完整 public trace、set 的零 mapped-payload 证明、普通调用草图、复杂度/metadata 和有限 FIX
  ledger 完成后进入 `AUDITED`；此前不修改产品。

## Audit verdict（`AUDITED`）

- public trace：`concurrent_hash_map.h`/`concurrent_flat_hash_map.h`、`concurrent_node_hash_map.h`、
  `concurrent_vector.h` 及 adaptive lock implementation 归本 factor；sequential Swiss/hash 归 `container`，
  retirement 归 FROZEN `mem_reclaim`；
- existing map/vector：`COMPLETE/KEEP`。flat map 提供 compact sharded storage，node map 提供 accessor 内
  stable address 和 retained/reclaiming table 两档，vector 提供 segmented append；不把它们互相伪装；
- concurrent set：`MISSING/FIX`。提供 flat/node 两档，普通调用为 `contains/try_insert/erase/size/clear`；
  node set 的 key 只在 accessor 保护期稳定。empty policy 使用 `[[no_unique_address]]` 并以 entry/node
  `static_assert` 证明不支付 mapped payload；
- dependencies：PUBLIC `core,mem,mem_reclaim,container`，无 `util_core/time/platform/sched`；不得复制
  allocator、Swiss core、adaptive reclamation 或 wait service；
- P0：lookup/insert/erase expected O(1)，shard/bucket count 为构造期选择，正常操作不重复 topology 探测；
  P1 primary 为 read-mostly membership registry，SAME_CONTRACT peer 在 oneTBB/Folly 中择一，5%；
- finite FIX：`CONCC-F01`（独立 target/header/test owner、两种 set、direct/package edges）和 Container suite
  的 `CONT5-F06`。除此之外不改现有 map layout、shard heuristic 或 lock algorithm。
