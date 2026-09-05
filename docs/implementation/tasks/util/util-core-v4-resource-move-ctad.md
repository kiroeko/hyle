# Util-core v4：UniqueResource move-only transfer 与 FunctionRef CTAD

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前只登记 `REOPENED`。完成 `SCOPED/AUDITED`、冻结有限 FIX 集以前不修改产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`util_core` / Util foundation /
  `util-core-v4-resource-move-ctad` / `REOPEN`；
- baseline：`66ac957109bff59262bc3ebdf7f2478fb7c67070`；前轮：
  `util-core-v3-foundation-utility-superset`；
- trigger：`UTILCORE-RESOURCE-MOVE-CTAD-2026-09-05`；只读证据见
  [Math / Util-core v3 post-closure replay](../project/math-util-core-v3-post-closure-replay-2026-09.md)；
- concrete defects：普通 move-only nothrow deleter 被 `UniqueResource` 拒绝；可唯一推出 signature 的
  nontype bound-member 不能 CTAD；checked factory 还丢失了成熟 shape 的 conditional `noexcept`；
- 初步范围：只修 deleter 的安全 move-or-copy transfer、owner move construction、factory exception
  specification、nontype bound callable deduction guides，以及相邻 tests/sample/docs/codegen；
- tier：`P1 / closure-perf-v1`。所有 resource/CTAD 修正本身为 P0 O(1)、zero-allocation；FunctionRef
  两字布局、调用机制和旧三项同合同 workload 不得变化，但当前策略要求对新 candidate 重放一次既有
  W1-WIN 5% non-inferiority gate；
- 非目标：nullable/owning/async callback、pointer-like UniqueResource operators、default owner、resource
  traits、typed Platform/Mem/IO owner、Util-runtime、下游冻结产品迁移、Native FFI、公共性能仪器或新的
  benchmark workload。

只读 replay verdict 为 `PRODUCT_TRIGGER`。合法下一步是 `REOPENED -> SCOPED`，不是直接补 patch。

