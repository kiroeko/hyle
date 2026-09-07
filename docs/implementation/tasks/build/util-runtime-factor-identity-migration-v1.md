# Util-runtime factor 身份迁移

<!-- PROJECT UNIT REVIEW: COMPLETE -->

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

迁移实现 commit 为 `48b11387`。显式 previous transition、closure schema/transition tests、selector、
LOC owner test、docs checker 和 whitespace check 全部通过；新 `util_runtime` factor 在 Windows
独立配置，runtime library、普通 Registry/demangle executable 与两张公开头均成功构建。完整 Util
产品的源码嵌入、relocated shared SDK 与 relocated static SDK consumer 全部通过，安装导出只向用户
暴露 `hyle::util`，内部 detail target 为 `hyle::_factor_util_runtime`。

全仓当前合同扫描没有遗留 `hyle_util`、`HYLE_UTIL_BUILD`、`src/tests/bench util` 机器路径或旧
factor selector；历史 task/evidence 中的旧名仍作为当时事实保留。独立 fault fixture 和长平台符号
测试仍按 v4 已登记的 RT-02/03 失败，未被本身份任务改写成 PASS；Registry RT-01 也保持 BLOCKED，
由下一轮有限恢复处理。因此本项目单元 **COMPLETE**，但不改变 util-runtime 的产品裁决，也不签发
FROZEN。
