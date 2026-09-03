# Container v3：并发 node map 的旧表寿命

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen trigger

- component / owner / round：`container` / Container / `container-v3-node-table-reclamation`；
- task kind：`REOPEN`；baseline：`13117568b9496b0b6db29235ece0bf6e35ae16db`；
- trigger ID：`CONTAINER-NODE-TABLE-RECLAIM-2026-09-03`；
- 新事实：`ConcurrentNodeHashMap` 每次 rehash 都把旧 bucket array 保留到 map 析构。几何增长把总
  bucket storage 限制在当前 array 的两倍以内，但长寿命、持续增长的推理 registry 仍无法在运行期
  归还旧表；项目 owner 已明确要求在该用途成立时完成 `mem_reclaim`；
- admitted journey：模型、backend 或请求路由表在服务期并发查询，启动后仍可能扩容；node/value 在
  accessor 锁内保持稳定，旧 bucket table 应在扩容前已进入 table hand-off 的 reader 全部离开后释放；
- peer/resume evidence：官方 vcpkg Folly `2026.02.23.00#2`（上游 tag
  `v2026.02.23.00`，commit `46afdbec8f58fc21445741278f14d53ec7144676`）已在 W1-WIN / MSVC
  19.50 构建并通过 `rcu_domain` + `rcu_obj_base` publication/retirement smoke；前一轮
  `mem_reclaim` 的 peer blocker 已消失；
- 受影响范围：只允许审计并修改 `ConcurrentNodeHashMap` 的 table load→bucket lock hand-off、rehash
  retirement、析构 quiescence、显式寿命档位、测试、直接组件边和对应指南；
- 非目标：重审其他 Container 能力、改变 node/accessor 语义、引入 lock-free map、改公共性能仪器、
  修改 Compute/Graph，或用 benchmark 命名第二套 API。

当前状态只进入 `REOPENED`。下一次 `SCOPED` 转换必须冻结普通 C++ 形状、默认寿命语义、P1
same-contract/normal-consumer workload、5% margin、实验预算及 source/static/shared gates；在
`AUDITED` 前不写产品代码。
