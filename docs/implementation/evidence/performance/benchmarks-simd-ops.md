# SIMD 旧逐操作矩阵说明

> `HISTORICAL / RETIRED`。本页过去保存旧 `vec.h`/backend 的大规模逐行 benchmark 摘要。旧寄存器栈已在
> SIMD v3 物理删除，旧矩阵的源码、peer pin、仪器和候选 identity 均不能证明当前产品性能。

当前机器真相是：

- 387-operation 冻结 inventory 与 422 个终态 route scope：`scripts/simd-op-routing.json`；
- ordinary family route：`scripts/simd-family-routing.json`；
- 当前工程性能与限制：[SIMD v3 工程性能证据](benchmarks-simd.md)；
- 当前施工/terminal record：[SIMD v3 closure card](../../tasks/simd/simd-v3-provider-routing.md)。

机械检查：

```text
python scripts/simd-op-coverage.py --check-provider-manifest --check-ledger \
  --require-route-decisions --check-provider-routes
```

当前账本为 387 个 operation、422 个 route scope：366 `ROUTE_HIGHWAY`、34 `KEEP_HYLE`、22
`REMOVE_REDUNDANT`、0 `PENDING`。这些数字描述迁移裁决，不等于 422 个独立性能冠军声明。

旧 `--check/--reports` 源码扫描、orderswap 大矩阵和历史“赢/输”表只可从 Git 历史追溯；不得恢复到当前
用户文档或作为重开产品的依据。新的真实 consumer、vendor 升级或新 ISA 证据只重签受影响 route group，
不再重跑开放的 operation × overload × size × ISA 笛卡尔积。
