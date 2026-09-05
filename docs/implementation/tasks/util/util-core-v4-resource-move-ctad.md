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

## Closure card（`SCOPED`）

- normal journeys：
  - acquisition 返回 `int`/pointer-like token，cleanup 带 move-only 状态；owner 必须能从一个 nothrow-
    movable deleter 取得唯一责任，并在 owner move 后仍只 cleanup 一次；
  - synchronous decoder/visitor 把已知成员函数和 lvalue 对象交给 `FunctionRef`，当成员函数类型已经唯一
    给出 result、argument、cv 与 `noexcept` 时无需再手写 signature；
- dependency / delivery：继续 `util_core -> PUBLIC core`、header-only；用户仍通过 `hyle::util` product
  façade 获取，source/static/shared SDK 不新增 binary、第三方、C 或 Native FFI surface；
- API boundary：`UniqueResource<R,D>` 接受 nothrow-movable move-only deleter，也保留安全 copy fallback；
  transfer 选择只能由类型异常保证决定，不能新增 runtime branch、allocation 或状态字。checked factory 的
  `noexcept` 必须反映 comparison 与 owner construction。bound CTAD 只在 callable 类型能唯一推出 HYLE
  已支持 signature 时参与；不猜 ref-qualified/overloaded callable；
- performance：resource/CTAD 路线是 P0 O(1)、zero-allocation/lock/dispatch-addition。FunctionRef layout
  必须仍为两个机器字，bound route 仍只有一次 erased invocation；现有三项 4096-record workload、Abseil
  peer、5% margin、W1-WIN、`closure-perf-v1` 全部冻结，不得为了新 candidate 改测试形状；
- correctness / red team：direct/checked valid/invalid、move-only deleter、copy fallback、resource-copy throw、
  move construction success/failure、release/reset exactly-once、factory `noexcept` traits、member cv/noexcept/
  pointer/object/unsupported-ref-qualified deduction、no-exceptions compile；
- build gates：W1-WIN、W1-WSL、M1-MAC exact factor tests/header closure/code shape；Util source、搬迁后
  static/shared installed sample；中央 component/package contract；`check-module-closure.py` 与
  `check-docs.py`；
- budget：从 `SCOPED` 起 120 分钟，最多一个产品批次、一次 correctness/codegen correction、一次 P1
  qualification 和一次交付 correction；不修改公共仪器；
- stop：上述有限 API/安全缺口关闭且全部适用门通过则 `CLOSED`；若 move-only deleter 无法在不增加状态或
  破坏 acquisition safety 的条件下实现，具名 `BLOCKED`，不扩大成 resource traits 库。

进入 `SCOPED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-util-core-v4-scoped-1f8829c5.json`。
SHA-256：`BDD4366DD0F6FFEDA53BB0DBD6D79A3E1A4D7641ECF35D4789C134812841C4D8`。
