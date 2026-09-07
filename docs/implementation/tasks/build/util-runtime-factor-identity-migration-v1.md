# Util-runtime factor 身份迁移

<!-- PROJECT UNIT REVIEW: ACTIVE -->

## Scope

- owner 要求把语义上已经称作 util-runtime 的内部 factor 从含混的机器 ID `util` 彻底改为
  `util_runtime`；项目尚未发布，不保留旧 ID、target 或 visibility macro alias；
- baseline：`f21c05552b92a1c7ebeb15bf85233219845c8541`；迁移前 registry 快照 SHA-256 为
  `2395d8a0d140ffa00f1e3c61ef92449fe2fd162b2430c8be51db6054de89a005`；
- public product、namespace 与 include 形状保持 `hyle::util`、`hyle::util` 和
  `<hyle/util/...>`；只纠正维护者 factor 身份；
- non-goals：改变 Registry/demangle 行为、重开 `util_core`、新增 C/Native FFI、性能采样或修改
  closure/validation 工具。

## Decision and finite checks

中央 component ID、真实 target、源码/test owner 与 detail package target 分别成为
`util_runtime`、`hyle_util_runtime`、`src/util_runtime` / `tests/util_runtime` 与
`hyle::_factor_util_runtime`。Util 产品 façade 继续叫 `util`，聚合 `util_core + util_runtime`。
两个 runtime 公开头通过中央 `SPLIT_PUBLIC_HEADERS` 归属新 factor。

registry 永久记录一对一 `util -> util_runtime` migration，并在原 ledger 槽位延续原 BLOCKED
状态，不删除或重写历史 v4 事实。验证只包括：显式 previous transition、closure/docs/tooling
checks、`util_runtime` 精确 factor build/test/header，以及 Util source/static/shared package
consumer。产品行为修复属于后续 v5 resume，不混入本迁移。

## Result

待验证后填写。此项目单元不改变 util-runtime 的 BLOCKED 产品裁决，也不签发 FROZEN。
