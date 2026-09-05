# Util-core v4：UniqueResource move-only transfer 与 FunctionRef CTAD

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前状态：`CANDIDATE`。红队纠正已经形成新的 immutable product commit/tree；只重放本卡既定门，
> 不增加能力或修改 benchmark。

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

## Capability 与 public-surface 审计（`AUDITED`）

| family / journey | current route | coverage | decision |
|---|---|---|---|
| value-like token + copyable nonthrow deleter | direct/checked `UniqueResource` | `COMPLETE` | `KEEP` |
| value-like token + move-only nonthrow deleter | class assertion 和 constructor constraint 拒绝 | `MISSING` | `FIX` |
| resource transfer 失败安全 | deleter 先安全复制，resource 再 move-or-copy | `PARTIAL` | 扩成 deleter 的 compile-time move-or-copy，不加 runtime state |
| checked factory exception contract | 运行语义正确，但 public function 未传播可证明的 `noexcept` | `PARTIAL` | `FIX` |
| runtime/compile-time free 与显式 member FunctionRef | 两字非 owning view | `COMPLETE` | `KEEP` |
| 可唯一推出 signature 的 bound-member CTAD | explicit signature 可用，CTAD 失败 | `PARTIAL` | `FIX` |
| nullable/owning/async callback、typed resource owner | higher semantic owners | `ROUTED/OUT_OF_SCOPE` | 不进入本轮 |

### Public API 与依赖反查

- move-only deleter 是普通 ownership 输入，不来自 benchmark；实现只放宽合法 constructor/move source，
  不增加对象字段、runtime mode 或“fast”入口；
- bound-member deduction 的 result/arguments/cv/`noexcept` 已完整编码在成员函数类型中，因此 CTAD 不需要
  猜测。overload set 和 ref-qualified member 保持显式 signature/不支持，避免悄悄选择错误合同；
- `FunctionRef` 的 4096-record benchmark 没有任何参数进入 public 类型。mutable-view 到 const-signature
  view 的 GCC hostile probe 与直接调用及 Abseil 都降为同一 indirect tail call，故不改该路线；
- 全库产品扫描没有 v3 新入口的既有语法 consumer，也没有因 v3 additive API 而失效的调用。Mem/Sched
  继续通过显式 `util_core` edge 使用既有 Ref/SpinLock；本轮没有 breaking rename/layout，故无需迁移冻结
  consumer；
- Math 新整数策略与 Time 的 wide rational conversion、Platform 的 OS parser/status policy、Obs 的 atomic
  sentinel transition 并非完整同合同 owner。若仅为隐藏一至两条 guard 而增加 public include/组件边，
  不会获得更好复杂度或 codegen；本 replay 将其记为 owner-local，而不以“采用率”重开冻结模块。未来某个
  owner 修改同一公式时必须重新比较 Math route。

## 冻结 FIX ledger 与 API freeze

| ID | exact action | cost / safety contract | verification |
|---|---|---|---|
| `UC4-UR-01` | deleter transfer 从“必须 nothrow copy”改为 compile-time “nothrow move 优先，否则 nothrow copy” | 对象 layout 不变；零 branch/allocation；resource construction 失败仍 exactly-once cleanup | move-only/copy-only、valid/invalid、throwing resource copy、owner move |
| `UC4-UR-02` | owner move 在潜在 resource-copy failure 时保留 source responsibility；若 deleter 必须先 move，则失败时由已构造 deleter cleanup 并 disarm source | 无 leak/double-free；成功 move 后 source inactive | adversarial state/counter tests |
| `UC4-UR-03` | checked factory 与 internal construction helper 传播条件 `noexcept` | 编译期属性，无 runtime cost | positive/negative static assertions |
| `UC4-FR-04` | 为受支持、非 ref-qualified member-function type 增加 bound-member deduction guide | 只改变合法 overload resolution；仍走现有 nontype invoker和两字 layout | mutable/const × throwing/noexcept、object/pointer、negative compile traits |
| `UC4-DOC-05` | 修正文档、installed sample 与 P0 probe | 不引入 peer/benchmark 词汇到用户 API | choice/pitfall、三交付 |
| `UC4-P1-06` | 在不改 contract/workload/peer/margin 的条件下对新 candidate 重跑现有 FunctionRef P1 | 只裁决原三条正常同步 traversal | fresh qualification + verifier |

API freeze：继续使用同一 `UniqueResource`、checked factory、`FunctionRef` 和 `nontype` 名称；不新增对象
metadata、动态 dispatch、nullable state、pointer convenience 或 benchmark-only overload。Util-core 保持
Core-only header-only factor。所有 `MUST` 已落入上述六项有限 FIX，才允许进入实现。

`SCOPED -> AUDITED` 的前态快照为
`C:\public\project-hyle\_validation\registry-before-util-core-v4-audited-10962013.json`，SHA-256
`778A6D46F874DD69AD01B9C3B326E639B818688BBD5AE47D2720A1429BA490E2`。

## 第一产品候选（已由红队退回）

- candidate commit / tree：
  `2f53239c0cc2a3f15b717f7910e6b809d42f42b8` /
  `26cc24958d7cc917c9e740152ba72a8bf7592f62`；
- `UniqueResource` 继续保持 deleter/resource/ownership-bit 的既有 layout。构造期和 owner move 的
  deleter 路线由异常属性在编译期选择：nothrow move 优先，可能抛的 resource transfer 且存在 nothrow
  deleter copy 时先保留 source deleter；只有 move-only deleter 时，resource transfer 失败由已转移的
  deleter exactly-once 回收 source token 并 disarm source；
- checked factory 现在精确传播 owner construction 的条件 `noexcept`；valid/invalid、direct/checked、
  copy fallback、两类 owner-move failure 和 no-exceptions 路径都有独立测试；
- `FunctionRef{nontype<&Owner::member>, owner}` 只对现有四种非 ref-qualified signature 做 CTAD，直接复用
  原 bound invoker；两个机器字布局和旧三项 benchmark 路线没有改变；
- installed sample 同时编译使用 move-only deleter 与 bound-member CTAD；用户/架构文档补充了选择边界和
  resource-copy failure 陷阱。

进入 `CANDIDATE` 的前态 registry 快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-util-core-v4-candidate-2f53239c.json`，SHA-256
`3F4A003A1518F31E306030EAB347D7ABF5A57E6627CAF8BA23F5C59BBC83DB01`。

## 候选红队纠正（退回 `AUDITED`）

- candidate `2f53239c` 的 direct/checked ownership、move-only deleter 与 FunctionRef 固定门均通过；冻结的
  FunctionRef P1 qualification 也在 exact source/peer 上通过，但这些结果只属于旧 candidate；
- 敌意 sentinel 类型可以让 `resource == invalid` 本身为 `noexcept`，却让结果 proxy 到 `bool` 的转换
  抛异常。旧 requires 接受该类型，factory 的 `noexcept` 又只计算 owner construction，实际执行
  `!(resource == invalid)` 时会 terminate；
- 该 finding 直接属于既有 `UC4-UR-03`“checked factory 精确传播 comparison 与 owner construction
  合同”，不扩展 FIX 集、能力族或 benchmark。纠正必须让完整布尔判定具备不抛合同并以 compile-negative
  probe 固定；随后为新 commit/tree 重放全部既定门。

`CANDIDATE -> AUDITED` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-util-core-v4-candidate-correction-c8eee384.json`，
SHA-256 `62C93ED70BA1E5B4FE85C38009D323983449D1A819E3F88B1E491282B8876633`。

## 纠正后产品候选（`CANDIDATE`）

- candidate commit / tree：
  `af2e432f6c9025b6d398f0d4ba207005392cc8aa` /
  `d009cc1945f6d77622622df0c8a6c5ac8e19f21c`；
- `NothrowSentinelComparable` 继续接受 bool-like equality result，但把完整的显式 `bool` 转换纳入
  compile-time nothrow requirement；factory 不新增 catch、runtime mode、branch 或 metadata；
- throwing-conversion proxy 由 compile-negative test 固定。W1-WIN fresh factor 在该产品树等价工作树上
  20/20 通过；其余 exact-candidate、三平台、交付和 P1 门待本段重放。

进入纠正后 `CANDIDATE` 的前态 registry 快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-util-core-v4-candidate-correction-af2e432f.json`，
SHA-256 `DD57F5A64B0451381002E8436DCC969DA82876645B7FE6CDB3DCE745661B9BF2`。
