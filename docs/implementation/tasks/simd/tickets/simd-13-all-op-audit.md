# SIMD-13：387-operation 全量 inventory 与审计

- ticket state：`DONE`
- kind：scope audit / machine contract
- budget：4 工程时；不运行产品 P1，不写产品实现
- depends on：simd-v3 closure card

## 输出

1. 从 baseline `77d22d11be31f3184e96f8379068fa9779ed33e0` 的 `vec.h` 与七 backend 机械提取
   387 个去重 operation，并保存 definition count、source/backend、现有 benchmark title 与历史分类；
2. 每项指定 `PERF_ROW / STRUCTURAL / SELECTOR / ALIAS / COMPOSITION / ALGORITHM_WORKLOAD`，列出适用
   type/target variants、Highway/xsimd/HYLE candidates 与 evidence group；
3. route 决策初始可为 `PENDING`，但 op 集、分类和待验证组不得缺失；校验器对新增、丢失、未知映射失败；
4. 完成自顶向下 inference/source/ordinary journey 与自底向上 387/public-family trace，冻结 SIMD-14～17
   的有限 FIX 后把模块推进到 `AUDITED`。

## 硬门

- baseline inventory 必须恰为 387/387；别名、constant、selector 和 whole-range algorithm 也不能静默排除；
- 同 implementation/codegen 的项目可以共享 evidence group，但每一项都必须显式引用；
- 不把 387 个名字自动承诺为终态公开 API；删除/合并 spelling 仍须保留 capability mapping；
- 本票无旧源删除权限。计数不等于完成；任何未映射 variant 或 candidate group 使本票停止为 `BLOCKED`。

## 当前审计结果

- 冻结 inventory：387/387 operation、1,797 个 definition site；不可变投影 SHA-256 为
  `5e330e8e91fc6d835a75db1298c3bdf82a95cf69b633ea6574a4f234f391b401`。每个 site 保存 header、line 与
  declaration，避免一个 operation 名掩盖各 backend/overload。
- 比较分类：314 `PERF_ROW`、26 `STRUCTURAL`、13 `SELECTOR`、10 `ALIAS`、21 `COMPOSITION`、
  3 `ALGORITHM_WORKLOAD`；总和 387，零未分类。
- 锁定源码发现：Highway 有 363 项 same-name source hit，24 项落到 selector/composition 审计；xsimd 有
  98 项 direct/semantic-token hit，289 项无 direct same-contract source token。source hit 只准入候选，
  不冒充语义等价或性能胜负。
- 全部 387 项已有非 `PENDING` public capability mapping、完整 legacy target/site 列表和 provider audit
  group；共形成 18 个有界 planned evidence bundle。route decision、variant correctness 和当前性能仍保持
  `PENDING/INVENTORIED`，留给 SIMD-15，终态 checker 因而按设计失败。
- 机器合同：`python scripts/simd-op-coverage.py --check-ledger --check` 为 `PASS`；
  `--require-route-decisions` 为 `FAIL (387 PENDING)`。这证明当前允许进入实现审计，但没有提前取得删除门。

terminal verdict：inventory、双向 capability/public trace、provider candidates、planned evidence groups 与
SIMD-14～17 有限 FIX 已冻结；本票没有产品修改或删除。模块可由 `SCOPED` 进入 `AUDITED`。

## 2026-08-30 parser 勘误

SIMD-15 的 variant 审计发现原正则会在 `HYLE_FORCE_INLINE` 后遇到第一个括号即停止，因而漏掉
`Vec128<T, (N / 4)> ExtractBlock(...)` 一类返回类型。词法嵌套解析重新扫描同一 baseline 后增加 23 个真实
definition site，operation 集仍严格为 387，零 site 被移除；7 个受影响 operation 当时全部仍是单一
`PENDING legacy-all` route，所以没有扩大任何已签结论。精确 delta、安全条件与新旧 digest 见
`evidence/simd-15-inventory-parser-erratum.json`。本节纠正本票的历史计数，不新增产品能力或删除权限。
