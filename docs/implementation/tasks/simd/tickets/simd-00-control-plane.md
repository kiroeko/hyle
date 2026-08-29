# SIMD-00：专题控制面与现有草稿归属

- ticket state：`ACTIVE`
- kind：program / product-non-mutating
- depends on：无
- budget：90 分钟；性能预算 0

## 范围

建立 epic、顺序 tickets、权威文档入口和硬停止规则；终止无法在预算内执行的 `simd-v1` 单体批次；
逐组归属当前未提交 SIMD 改动。不得修改 SIMD 产品语义、公共 API、构建图或性能仪器。

## 草稿裁决

| draft | decision | destination |
|---|---|---|
| revision 2 consumer audit | `KEEP` | 新 round audit 输入 |
| extrema-index search | `REWORK` | SIMD-09 |
| typed-scratch radix/argsort | `REWORK` | SIMD-11 |
| target-bound activation headers | `REWORK` | SIMD-08 |
| current integrated-source hash literal | `REGENERATE` | SIMD-03/11 |
| unrelated dirty files | `PRESERVE` | 原 owner |

## 验收

- epic、入口和 13 张顺序票落盘；
- 旧 closure card 有具名 blocker、owner、resume condition 和 terminal receipt；
- registry 使用明确 previous snapshot 验证合法 `AUDITED -> BLOCKED`；
- 新 `RESUME` round 另用 previous snapshot 进入 `SCOPED`，且只激活 SIMD-01；
- `check-module-closure.py` 与 `check-docs.py` 通过；
- 产品文件不进入本票提交。

## 结果

- output commit：`TBD`
- verdict：`ACTIVE`
