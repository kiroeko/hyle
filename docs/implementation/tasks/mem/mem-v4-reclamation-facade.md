# Mem v4：reclamation 产品聚合与交付

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen trigger

- component / owner / round：`mem` / Mem / `mem-v4-reclamation-facade`；
- task kind：`REOPEN`；baseline：`13117568b9496b0b6db29235ece0bf6e35ae16db`；
- trigger ID：`MEM-RECLAIM-FACADE-2026-09-03`；
- 新事实：项目 owner 已准入读极热、低频替换的推理 metadata/snapshot publication。独立
  `mem_reclaim` factor 的官方 Folly peer 也已能在锁定 W1-WIN 环境构建；若该 factor 完成恢复轮，
  它应通过普通 `hyle::mem` 产品被源码、静态和共享 C++ 用户自然取得，而不能只留在 maintainer
  `HYLE_DEV_FACTORS`；
- 受影响范围：Mem façade factor roots、selected headers、package/source delivery、Mem package sample
  和按场景选型的 user guide；必要时回放直接 Container consumer；
- 非目标：重审 allocator、PMR、pool、arena、buffer、NUMA、mapping 或 VM 算法，改变 mimalloc，
  Native FFI/C API、稳定 C++ ABI、Compute/Graph 或公共性能仪器。

当前状态只进入 `REOPENED`。本轮是窄的 semantic/delivery owner 回放；`SCOPED` 后仍必须让
`mem_reclaim` 自己承担 correctness 与 P1 实现证据，不能借 Mem v3 的历史绿灯代签。
