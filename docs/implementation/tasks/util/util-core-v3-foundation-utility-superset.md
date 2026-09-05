# Util-core v3：基础 callable、scope 与 resource 工具补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只扩充 Core-only、header-only、去掉模型名后仍成立的基础工具。推理 serving、请求指标、
> backend 错误聚合、调度策略和容器能力不属于 Util-core；`SCOPED/AUDITED` 前不修改产品源码。

## Reopen evidence（`REOPENED`）

- baseline：`e3202ae42d30062da049938c1f267bde83e47324`；Util-core v2 的冻结产品身份仍是
  `d42d992cf99232669f60422f30fb3b50ec9ec4eb` / tree
  `9944f099679503df632b6335e69154656164b30b`；
- trigger ID：`UTILCORE-FOUNDATION-UTILITY-SUPERSET-2026-09-05`；
- owner 在 2026-09-05 正式准入 Math 与 Util-core 的通用能力扩展，同时明确 Obs/Error 的推理引擎
  策略不下沉。这命中 v2 的 missing-capability/API-usability reopen trigger；
- 当前 `FunctionRef` 已覆盖非空同步 borrow 与 `noexcept`，但没有标准 C++26 `function_ref`
  （WG21 P0792R14）和当前 Abseil
  `536d30bfea05a09c7561cbba749ca7f738d46821` 共同提供的 const-qualified callable contract 与
  compile-time nontype function/member binding。现有用户只能额外制造 lambda/member-pointer storage，
  不能从类型表达“不修改 callable”；
- 当前 `ScopeGuard` 只表达 all-exit cleanup。用户指南把 exception-only success/failure cleanup 路由到
  “工具链可用时的 `<scope>`”，但当前支持的 MSVC 19.50 标准库没有该 header。Folly
  `65749da412d92fe27d87ae45a8ef670720712858`、Boost.Scope gitlink
  `6e13b8e2b539ce8c23c282c54e8704af176fd40c` 与 Library Fundamentals v3 TS 都提供完整
  exit/fail/success 族；当前跨工具链路线因此不完整；
- Library Fundamentals v3 与 Boost.Scope 还共同提供 generic `unique_resource` 任务：拥有任意
  value-like native token、显式 invalid sentinel、deleter、release/reset/get。它适合没有领域 RAII owner
  的低层资源，但不应替代 Platform/Mem/IO 已有 typed owner；本轮审计其是否作为 `MUST`，而不是从
  peer symbol 直接抄 API；
- C++23 已有 `std::move_only_function`，普通 owning callable 继续 `ROUTED`；Sched/IO 的异步
  fn+ctx record 有自己的 cancellation/allocation/lifetime 合同，不被本轮强行迁移。

## 初步能力边界

| family | current coverage | initial disposition |
|---|---|---|
| synchronous borrowed callable | `PARTIAL` | `MUST`；审计 const/noexcept qualifier matrix 与 nontype binding |
| all-exit scope cleanup | `COMPLETE` | `KEEP`；保留现有 ScopeGuard/HYLE_DEFER 的自然 route |
| exception success/failure scope cleanup | `MISSING` | `MUST`；完整支持或给出所有 shipping toolchain 可用的完整 route |
| generic value-like unique resource owner | `MISSING` | `GENERAL -> audit`；只有证明 typed owner 不适用且形状成熟才提升 `MUST` |
| owning/asynchronous/fixed-capacity callable | standard or owning module | `ROUTED/DEFER`；没有具名资源合同不造第二套 scheduler callback |
| weak ownership, parking/fair locks, secure entropy, containers | other owners/standard | `OUT_OF_SCOPE` |

## 初步成本与停止边界

- round 初定 `P1`，因为 `FunctionRef` 是潜在热调用边；scope/resource 能力按 P0 结构验证；
- borrowed call 必须维持两个机器字、零分配、零 ownership、一次 erased call；静态 binding 不得引入
  per-call allocation/lock/branch。scope/resource 必须只存必要 callable/resource/active state；
- P1 只允许在 API freeze 后对一个普通 callback traversal 和一个 holdout 调用形状做同合同验证，
  peer、5% margin、W1-WIN 与 `closure-perf-v1` 在 `SCOPED` 前冻结；
- 本轮不扩建公共仪器，不凭微基准发明 `fast/light` API；`AUDITED` 前不写产品实现。
