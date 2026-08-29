# SIMD-01：Math 边界与文档权威

- ticket state：`DONE`
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

## 结果

- `math` 当前只有 59 行和四个 same-type non-bool integral 入口；产品语法 consumer 为 0，只有安装包
  fixture。其 shape 需求在 inference peer 中成立，但没有 correctness/usability/dependency/delivery 新事实
  击穿 `math-v1`，因此保持 `CLOSED`，本 epic 不改 math 产品。
- ownership 继续是：C++23 scalar → 标准库；integer shape → `hyle::math`；lane/row → `hyle::simd`；
  tensor/model → compute/consumer。Math 不再定义为“Highway 没有的 clay 剩余内容”。
- 当前 SIMD 相关文档 48 份：architecture 2、user 1、developer instrument 1、maintenance 1、debugging 3、
  performance evidence 3、implementation history/task 37。37 份 history 不进入新维护者默认阅读链。
- `simd-design.md` 含大量阶段 chronology、过时 LOC 与“产品零 vendor”说法；`simd-vendor-surface.md`
  把 integrated-source port 错写成全自造；`simd-usage.md` 仍显式 provisional；旧性能表只可作历史输入。
- verdict：`DONE`。文档权威分流进入 epic；vendor 事实由 SIMD-02 立即修正，完整 architecture/user rewrite
  由各产品票和 SIMD-12 按实际候选完成。
