# SIMD-01：Math 边界与文档权威

- ticket state：`QUEUED`
- kind：audit / default product-non-mutating
- depends on：SIMD-00
- budget：2 小时；性能预算 0

## 问题

确认顶层 `math` 已不再是 clay/Highway 缺口垃圾桶；机械对账其四个公开入口、真实/北极星 consumer、
Core 依赖和 user routing。分类 48 份 SIMD 相关文档，找出 architecture 中的阶段流水账、当前 guide、
维护合同、历史 task 和有效性能证据。

## 输出与停止

- `math -> scalar integer shape`、`simd -> lane/row`、`compute -> tensor/model` 的明确 verdict；
- 若无既有 reopen trigger，Math 保持 CLOSED，禁止改其产品；若有 trigger，停止并另建 math-v2；
- 当前 SIMD 文档 authority map、待重写清单与历史保留清单；
- 不因旧文档中的命令或“终态”扩大后续 FIX 集。
