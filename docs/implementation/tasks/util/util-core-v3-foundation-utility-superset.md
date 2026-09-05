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

## Closure card（`SCOPED`）

- component / owner / round ID：`util_core` / Core-only generic utility owner /
  `util-core-v3-foundation-utility-superset`；task kind：`REOPEN`；
- baseline commit：`e3202ae42d30062da049938c1f267bde83e47324`；
- demand IDs：
  - `UTILCORE-V3-FREF-01`：同步 borrowed callable 必须表达 mutable/const 与 throwing/noexcept 四种合同，并
    允许 free/member function 的 compile-time nontype binding；
  - `UTILCORE-V3-SCOPE-02`：exception success/failure cleanup 在全部 shipping C++23 工具链上都必须有
    可发现 route，不能指向 MSVC 19.50 不存在的 `<scope>`；
  - `UTILCORE-V3-RESOURCE-03`：审计并在准入后提供任意 value-like native token + deleter 的独占 RAII
    owner，覆盖 invalid/release/reset/move/construction-failure，而不替代已有领域 owner；
- normal workload：推理/工程图遍历在一次同步调用期间对一批 node/record 调用 borrowed predicate，
  callable 可能是 const state、mutable state、free function 或 member function；ggml 的 graph eval/
  compare callback + context 是北极星任务证据，但 HYLE 形状保持 C++、不复制 C ABI。scope/resource
  workload 是资源取得后的多早退 rollback、exception-only commit/fail cleanup，以及没有领域 wrapper 的
  单个 fd/HANDLE-like token；
- direct dependencies / consumers：factor 继续只 `PUBLIC core`。中央直接消费者为 `mem_vm`、`sched`
  （PUBLIC）与 `graph`（PRIVATE），当前分别消费其它 Util-core 头；普通用户由 `hyle::util` façade 取得。
  本轮不强制迁移已冻结消费者，也不修改 util-runtime；
- reuse candidates：现有 `FunctionRef`/`ScopeGuard` exception-safe storage、`HYLE_NO_UNIQUE_ADDRESS`、
  `std::invoke`、`std::uncaught_exceptions`、C++23 `std::move_only_function` owning route，以及标准/peer 的
  nontype、scope condition 和 unique-resource 生命周期。allocator、Container、Platform、Error、SIMD、
  wait/parking 均不属于这些 header-only 值对象；
- tier：整体 `P1`，只因为 `FunctionRef` 可处于 per-node 调用边；scope/resource 为 `P0 STRUCTURAL`；
- material/resource budget：5% non-inferiority margin；FunctionRef 保持两个机器字、零分配/锁/ownership，
  每次调用一个 erased invoker，nontype binding 不增加第二次 runtime function-pointer dispatch；scope/
  resource O(1)、零动态资源，只存必要 callable/resource/condition state；
- budget：从 `SCOPED` 起总计 120 分钟，其中能力/API 审计 35 分钟、实现与 correctness 45 分钟、P1/
  交付/红队 40 分钟；性能预算 30 分钟。最多一次复现/反序、一次 codegen/profile 诊断、一个产品实现
  批次与一次产品重跑，不扩建公共仪器；
- machine / instrument：correctness 与交付为 `W1-WIN`、`W1-WSL`、`M1-MAC`；P1 只在 `W1-WIN` 使用
  `closure-perf-v1`；Formal `PENDING / formal-release-v1`；
- strong peers（每项 capability 最多两个）：FunctionRef 使用 C++26 P0792R14 与当前 Abseil
  `989c696cd131fdaba300c44eb86e1c00fd3a4b56`；conditional scope 使用 Library Fundamentals v3
  N4948 与 Folly `65749da412d92fe27d87ae45a8ef670720712858`；generic resource 使用 N4948 与
  Boost.Scope gitlink `6e13b8e2b539ce8c23c282c54e8704af176fd40c`；
- provenance：`NORTH_STAR_JOURNEY + STANDARD + PROVISIONAL_ARCHETYPE`；当前 HYLE product syntax
  consumer 只证明已有 ordinary FunctionRef/ScopeGuard 可交付，不冒充新 qualifier/resource 的生产采用；
- frozen P1 workload：primary 是 4096 个预建 record 的同步只读 traversal，两侧通过相同 noinline
  call boundary 接收 const-qualified borrowed callable、执行相同 predicate/checksum；holdout 是预绑定
  free/member function 的相同 traversal。candidate 为 HYLE，peer 为上述锁定 Abseil，header-only、
  Release、LTO off、setup/teardown 均不计费；comparison class 预定 `SAME_CONTRACT`，在采样前必须完成
  等价矩阵和独立 reviewer；
- API 草图：`FunctionRef<R(Args...) [const] [noexcept]>` 与 HYLE-owned nontype tag；现有 all-exit owner、
  新 conditional guards、generic resource owner 都按 PascalCase 值类型 + CTAD/factory 组织。精确名称、
  constructor constraints、invalid-state policy 与 exception guarantee 在 `AUDITED` 冻结；
- correctness gates：callable qualifier/nothrow/return/member/free/null/lifetime compile matrix，scope normal/
  exception/nested exception/move/dismiss/invoke/construction failure，resource valid/invalid/release/reset/move/
  deleter/storage failure，header self-contained、warnings、WSL UBSan/TSan 适用子集；无共享状态的条目不
  伪造 concurrency PASS；
- delivery gates：`HYLE_DEV_FACTORS=util_core` factor-only tests；`hyle::util` source embedding、relocated
  static/shared installed consumer；header-only factor 不制造独立 binary；中央 edge/header inventory
  保持一致；
- usability/docs gates：同步 borrow vs owning async、all-exit vs success/failure、typed owner vs one-off
  generic resource 的选择表和可复制 recipe；lifetime、exception、deleter、invalid token 陷阱；无旧 C ABI/
  Native FFI、benchmark seam 或第三方 public-type leak；
- expected output：完整双向 capability map/public trace、冻结 API 与有限 `FIX`，一个 product candidate，
  correctness/P0 structure/P1 callback/三交付证据、七问、红队、限制和 terminal receipt；
- stop：只修冻结缺口。P1 若在一次机制诊断和一次产品重跑后仍 material fail/inconclusive，则以具名
  blocker 停止；不能改 margin、增加第五种姿势或用旧 Ref/SpinLock 结果替新 FunctionRef 签证。

`REOPENED -> SCOPED` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-math-util-scoped-68398f1e.json`，SHA-256
`FEFC5C89EBC242AC9447EFC87342137B1E7B37E37D75751290CE19E130CABE5D`。

## Capability-family coverage map（`AUDITED`）

| family | admitted normal journey | variants/lifetime contract | current route | coverage | decision |
|---|---|---|---|---|---|
| synchronous borrowed callable | per-node visitor/predicate within one stack call | mutable/const × throwing/noexcept; runtime callable/function and compile-time free/member binding | two mutable-signature `FunctionRef` forms | `PARTIAL` | `FIX` qualifier matrix + nontype binding |
| all-exit lexical cleanup | rollback and C API cleanup across early returns | once, move, dismiss/invoke, constructor failure | `ScopeGuard` / `HYLE_DEFER` | `COMPLETE` | `KEEP` |
| exception-conditioned cleanup | transaction commit/failure action | run only on a newly escaping exception or only when none is added; once/move/dismiss | user guide points at unavailable MSVC `<scope>` | `MISSING` | `FIX` `ScopeFail` / `ScopeSuccess` |
| generic value-like exclusive resource | one-off fd/HANDLE-like token lacking a domain owner | explicit ownership bit, invalid checked factory, move, get/deleter, release/reset, acquisition safety | ad-hoc guard/capture | `MISSING` | promote to `MUST`; `FIX` bounded `UniqueResource` |
| owning/nullable/asynchronous/fixed-capacity callable | stored/cancelled work | owner controls storage, allocation, cancellation and lifetime | `std::move_only_function` or Sched/IO owner | `ROUTED` | no second callable family |
| typed resources and memory ownership | files, mappings, allocations, sockets, threads | domain errors and teardown | Platform/Mem/IO typed RAII | `ROUTED` | generic owner is fallback, never preferred replacement |
| containers, secure entropy, parking/fair locks, weak ownership | different semantic owners | domain-specific | existing factors/standard | `OUT_OF_SCOPE` | no Util-core union |

### Public-surface, consumer and duplicate trace

- Existing `borrow.h`, `random.h`, `ref_counted.h`, `spin_lock.h`, `type_name.h` and their contracts are unaffected.
  Existing `ScopeGuard` remains the all-exit class; conditional guards do not replace it or add a hidden branch to it.
- Ordinary `FunctionRef<R(Args...)>` invokes an object as mutable `F&`; its const-signature counterpart accepts/invokes
  only `const F&`. Both throwing and `noexcept` forms remain non-null, trivially copyable two-word views. A temporary is
  legal only through the containing full-expression synchronous call; storage/return/async use remains a lifetime bug.
- `NontypeTag<F>` and the `nontype<F>` variable template bind a compile-time free/member function. An unbound callable
  stores no runtime function pointer; a bound member stores only the object address. Both still cross exactly one erased
  invoker and do not add a second indirect function call.
- `ScopeFail` runs only when `std::uncaught_exceptions()` increased since construction; `ScopeSuccess` runs otherwise.
  Both are move-only, dismissible and always terminate if a destructor-time action throws, matching HYLE's existing
  destructor safety policy. Failure while storing a `ScopeFail` action invokes the original cleanup; failed
  `ScopeSuccess` construction does not run a success action.
- `UniqueResource<Resource,Deleter>` is a small fallback owner with a separate ownership bit. Direct construction owns
  even a sentinel-shaped value; `make_unique_resource_checked(resource, invalid, deleter)` is the route for acquisition
  APIs with an invalid result. `release()` only disarms, `reset()` deletes at most once, and `get()` lets a caller copy a
  value-like token before release. It does not surface resource errors or replace typed owners.
- Current `mem_vm`/Sched/Graph dependencies consume other Util-core headers. No frozen consumer is migrated merely to
  manufacture adoption, and no new component edge is required.

## Capability / gap ledger and frozen `FIX` set

| ID | class | exact action | cost/complexity contract | verification / negative space |
|---|---|---|---|---|
| `UC3-FREF-QUAL-01` | `MUST` | add const and const-noexcept signature specializations | two words, trivial copy, O(1), one erased invocation, zero ownership/allocation/lock | compile matrix and ordinary/const/mutable rejection |
| `UC3-FREF-NTTP-02` | `MUST` | add HYLE tag and compile-time free/unbound/bound-member constructors | no stored runtime function pointer for static target; no second indirect call | behavior, null object, CTAD where unambiguous, optimized probe |
| `UC3-SCOPE-COND-03` | `MUST` | add `ScopeFail` and `ScopeSuccess` in `scope_guard.h` | O(1), callable + one count/active state, zero allocation/lock | normal/throw/nested/move/dismiss/construction failure/no-exception compile |
| `UC3-RESOURCE-04` | `MUST` | add `unique_resource.h`, `UniqueResource`, checked factory | O(1), resource+deleter+ownership state, zero allocation/dispatch | valid/invalid/move/release/reset/deleter/construction and throwing paths |
| `UC3-P1-05` | `MUST` | add only the frozen Abseil same-contract traversal and nontype holdout | 4096 prebuilt records, setup excluded, 5% W1-WIN non-inferiority | qualification matrix, reverse order, one diagnosis/rerun maximum |
| `UC3-DELIVERY-06` | `MUST` | add the new header to factor inventory and recheck Util source/static/shared | header-only factor, Core-only real edge | exact-factor/header/relocated SDK; no binary/FFI artifact |
| `UC3-DOC-07` | `MUST` | update architecture and user selection/pitfall guide | current behavior only | borrow/own, exit/fail/success and typed/generic decisions |

These seven entries are the entire product batch. Weak references, optional borrowed callables, async storage, generic
error-carrying teardown, resource traits/pointer syntax/swap, consumer migration and common-instrument changes are not
authorized.

## API / usability freeze

- Public callable forms are `FunctionRef<R(Args...)>`, `FunctionRef<R(Args...) noexcept>`,
  `FunctionRef<R(Args...) const>` and `FunctionRef<R(Args...) const noexcept>`. Nontype use is explicit at the call
  boundary, e.g. `FunctionRef<int(int)>{nontype<&transform>}` or a bound member form with an lvalue/object pointer.
- All four FunctionRef forms stay non-defaultable and non-null. Const is a target-invocation contract, not a claim that
  the two-word view itself was previously mutable. Nontype binding does not introduce ownership or extend object life.
- Conditional scope types use class CTAD and the same discoverable `dismiss()` / `active()` control as `ScopeGuard`;
  no macro-only API or dependency on experimental `<scope>` is introduced.
- `UniqueResource` requires a nothrow-copy-constructible deleter and a resource that is nothrow-movable or copyable so failed
  transfer cannot leak or silently steal ownership. Move assignment/reset-to-new-resource are available only when their
  storage operations satisfy the required nonthrow transfer. The deleter is expected to be nonthrowing; throwing during
  destruction terminates.
- The generic resource minimum is `allocated`/explicit bool, `get`, `get_deleter`, `release`, `reset()` and checked
  construction. Dereference operators, resource traits, default invalid construction, swap and error propagation remain
  outside this batch; use a domain owner where those semantics matter.
- Util-core remains header-only, `PUBLIC core`, selected directly by maintainers and aggregated for users by
  `hyle::util`. There is no C API, Native FFI or stable cross-toolchain C++ ABI.

## Owner seven questions（`AUDITED`）

1. **Third-party product path:** none. Abseil/Folly/Boost and WG21 documents are pinned evidence and benchmark-only
   peers, never public/product dependencies.
2. **Capability completeness:** callable qualifiers/static binding, conditional scope and bounded generic resource are
   the admitted gaps. Serving policy, async work and typed resource errors remain with higher owners.
3. **Internal reuse/acceleration:** implementation reuses Core macros, `std::invoke`, existing ScopeGuard storage rules
   and C++ exception count. No lower factor or SIMD/container facility improves these value objects; no fake consumer
   migration is permitted.
4. **Performance:** FunctionRef remains P1 and unverified until the frozen same-contract Abseil workloads run. Scope and
   resource are P0 O(1)/zero-allocation structural claims; no universal SOTA claim is made.
5. **C ABI/build:** no old C/Native FFI exists. The new header must enter the split-header inventory; factor stays
   INTERFACE and source/static/shared Util products must expose identical C++ behavior without a fake archive.
6. **C++ usability:** names expose borrow, exit condition and ownership; the guide must make forbidden lifetimes,
   invalid sentinels and throwing deleters explicit rather than requiring users to infer them from templates.
7. **Docs/red team:** architecture/user/performance/task updates, cross-toolchain behavior, package relocation and an
   independent red-team pass remain candidate gates and are not predeclared PASS.

`SCOPED -> AUDITED` uses the prior registry snapshot
`C:\public\project-hyle\_validation\registry-before-math-util-audited-ff6b5daf.json`, SHA-256
`27C669D4B93877DB3F5E07909374948388CB66E92FF491EB9C77BDBFDFA4F6D4`.
