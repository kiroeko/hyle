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
