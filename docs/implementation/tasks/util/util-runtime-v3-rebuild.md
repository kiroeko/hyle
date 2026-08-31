# util-runtime-v3 重审与重构记录

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> 当前机械状态只看 [`module-status.json`](../../module-status.json)。本轮由已关闭候选之后发生的
> Container 公开字符串/索引合同替换触发；在进入 `AUDITED` 并冻结有限 `FIX` 集前不得修改产品实现。

## Reopen intake

- component / owner / round：`util` / runtime utility owner / `util-runtime-v3`；task kind：`REOPEN`；
- trigger ID：`UTIL-V3-CONTAINER-PUBLIC-CONTRACT-01`；
- baseline：`8eb8926a7b15b879d8c96628b1f2c35c804d27b9`；
- trigger evidence：v2 candidate `f3b341cb2ae81154f81bc45052a651e773c5e2b3` 之后，Container 删除
  `string_ref`/旧 Swiss surface，Registry 的全部 key/help、hash/index、tests 与公开 include 已迁移到
  `StringView`/`FlatHashMap`。这满足 v2 的“公开合同、算法、布局或直接依赖变化”重开条件；
- affected scope：完整 Registry capability/public surface、Container/Error/Mem/Core 依赖合同、复杂度与
  source/static/shared 交付；demangle 作为同一 runtime component 的未变能力做双向回放；`util_core`
  保持独立、已关闭且不在本轮；
- provisional functional finding：当前指南要求按名解析后缓存 typed factory/handle，但 Registry 没有
  expected-O(1) 的 lookup-only 入口，只有 lookup+invoke 的 `create()` 或 O(n) 枚举。该项由当前
  llama.cpp/ggml 的 backend-reg-by-name 真实旅程支持，必须在 `AUDITED` 前裁决，尚不预先决定 API；
- north-star identities（2026-08-31 remote check）：llama.cpp
  `9723942adc518b43c4b95dc4dce6906903eb5e09`；standalone ggml
  `36da57138425487184aa1da2eee2cde155909c6f`；llama.cpp 随仓 ggml 与前者的 registry shape 对照；
- provisional tier / profiles / instrument：`P0`；`W1-WIN`、`W1-WSL`、`M1-MAC`；
  `closure-correctness-v1`；Formal `NONE`。Registry 是启动/服务发现，demangle 是诊断冷路；没有 profile
  证明二者进入 token/kernel 热路径；
- next：冻结 closure card 进入 `SCOPED`，再形成 capability-family map、完整 public trace、六问与有限
  gap ledger。动态库 discover/load/unload/quiescence、backend/device 领域选择和 proc-address 先作为
  owner-boundary 输入，不自动下沉到通用 Registry。

`CLOSED -> REOPENED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\util-runtime-v3\module-status-before-reopened.json`，SHA-256
`10B756BFFF43C9D8A9BEC6519738A24381AD93B0E19CED66BC2FBACB13C8AC1F`。

## Closure card（`SCOPED`）

- component / owner / round ID：`util` / runtime utility owner / `util-runtime-v3`；task kind：`REOPEN`；
- baseline commit：`8eb8926a7b15b879d8c96628b1f2c35c804d27b9`；该提交已包含 Container v2
  迁移后的 Registry 当前实现，后续 reopen/scoped 提交只有任务与状态；
- demand / defect：
  - `UTIL-V3-DEP-01`：Registry 的全部字符串/索引公开合同已从已签 v2 identity 漂移，必须按当前
    `StringView` 语义重查长度、ownership、hash、OOM、复杂度和 delivery；
  - `UTIL-V3-FIND-02`：启动期登记后，普通 backend/plugin consumer 需要按名解析并缓存 factory，
    当前只有 lookup+invoke 或 O(n) 枚举，指南与公开能力不闭环；
  - `UTIL-V3-INFER-03`：必须把当前 llama.cpp/ggml 的 backend registration、reg/device enumeration、
    name/type selection、dynamic load/unload 和 proc-address 逐项映射到 util、platform、compute/module
    owner，不能把“参考过 ggml”替代真实路线；
  - `UTIL-V3-SURFACE-04`：Registry 与 demangle 的全部公开入口、C++ 易用性、Native FFI 清洁度和
    source/static/shared 交付按六问重签；
- normal consumer workload：startup owner 显式登记约 4--32 个内建或已加载 backend factory，检查
  collision/OOM，按名字取得 factory 或直接构造，把解析出的 typed descriptor/object 缓存到服务对象；
  初始化结束后可冻结并枚举帮助信息。日志/backtrace 在正常线程把 C++ symbol 写入 owning string 或
  caller buffer；二者都不进入 token/kernel 内循环；
- scope：`include/hyle/util/registry.h`、`include/hyle/util/demangle.h`、`src/util/demangle.cc`，当前
  `hyle::util` component 的 public/internal 边界，直接 dependency/consumer，C++ API、ownership/
  lifetime/failure/concurrency/复杂度，tests/package journey，source/static/shared delivery，以及
  architecture/user/implementation/performance 文档；
- non-goals：重开 `util_core` 或依赖模块；把 backend/device 类型、capabilities、按类型/最佳设备策略、
  proc-address schema、动态库搜索/评分/卸载/quiescence 下沉到通用 Registry；async-signal-safe parser；
  captured arbitrary callable；模型/token/KV/serving policy；Native FFI/C API；稳定 C++ ABI；P1 timing、
  all-operation SOTA、Formal 证书或下游 compute/graph 收口；
- public/internal/higher-layer initial boundary：Registry 只拥有通用显式名字→factory 元数据、启动期
  mutation 与 immutable publication；platform `DynLib` 只拥有 loader handle；backend descriptor、设备
  集合、选择与插件资源排空由未来 compute/module owner 组合。demangle 的 sink 仍是 shared C++ SDK
  internal seam，不是公开 callback 或 FFI；
- direct dependencies / consumers：`PUBLIC core,error,mem,container`；没有直接第三方产品依赖。
  仓内真实产品 consumer 为 0，selected package sample 是交付 fixture；未来推理引擎是准入 demand，
  llama.cpp/ggml 是真实 domain evidence，不冒充 HYLE 已完成 engine；
- performance tier：`P0`。Registry 登记为 O(key+help)+expected-amortized O(1)，find/create 为
  O(key)+expected O(1)，freeze 为 O(1)/零 allocation，enumeration O(n)；demangle 为
  O(input+output)，允许平台 ABI 的临时 allocation/serialization。不得新增 lookup timing 专用状态，
  不运行纳秒 benchmark，不继承旧 N=256 数字，不声明端到端或跨平台 SOTA；
- resource/layout budget：lookup-only 路线不得 allocation/lock（frozen）或增加每项常态 metadata；
  mutable lookup 可持现有 mutex 但 callback/factory 不在锁内；新增公开入口优先返回现有
  `EntryView` value snapshot，不引入 `std::function`、owning string 或第二张 frozen table；
- total budget：审计 60 分钟、一个产品批次 45 分钟、三平台/三交付/红队 75 分钟；最多一次产品
  批次、一次机制纠正和一次受影响重跑；不得顺手修改 closure checker、构建系统或公共性能仪器；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC` / `closure-correctness-v1`；Formal `NONE`；
- pinned strong peers / domain inputs：
  - inference lifecycle：llama.cpp `9723942adc518b43c4b95dc4dce6906903eb5e09` 与 standalone ggml
    `36da57138425487184aa1da2eee2cde155909c6f` 的 backend registry；
  - index lower bound：Abseil `2c004366e983c5be8334ac1ea3d4420e8fbcbea7` flat hash；
  - demangle：Folly `23ebefdf96c116700d3e134f04d95a2b60a509de` 与 Boost
    `78505eab9b203e3e26e83893d0685f160133fa0f`；
- correctness gates：normal、empty/zero、key/help 长度上限、OOM/invalid input、duplicate/priority、
  lookup/create 等价、move-only args、view/owner lifetime、mutable/freeze race、creator/visitor reentry、
  stress、ASan/UBSan/TSan、三编译器/平台 ABI 与 public-header self-containment；同步 utility 的
  cancellation 为语义不适用，不能写虚假 PASS；
- delivery gates：isolated selected `util` build；source embedding、relocated installed static/shared
  C++ package consumer；中央 component graph、真实 target links、headers、Windows import/export 与 Unix
  loader identity；不制造 C/Native FFI journey；
- expected output：capability-family map、完整 public trace、有限 gap ledger/FIX、一个 candidate、三平台
  correctness/sanitizer 与三交付证据、P0 cost structure、六问、文档/红队、限制/reopen triggers、receipt
  与四项 LOC；
- stop condition：全部 `MUST` 为 `COMPLETE/ROUTED` 且所有适用门有真实 verdict则 `CLOSED`；否则在
  预算内以具名 blocker 停为 `BLOCKED`。动态 module lifecycle 只有 owner+完整路线才可写 `ROUTED`，
  不能用“未来再说”掩盖缺口。

`REOPENED -> SCOPED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\util-runtime-v3\module-status-before-scoped.json`，SHA-256
`5396D080BA423700C4E421F8E42E76B01878C95E4A1DE8041E4DF5B09619B483`。

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | Coverage | missing edge / negative space |
|---|---|---|---|---|---|---|
| C++ symbol presentation | ordinary-thread logging, backtrace and diagnostic UI; Folly/Boost demangle | owning string, caller buffer, `type_info`, plain-name fallback, truncation/required-size | caller owns output; implementation may allocate/serialize inside the call; overlap and ABI failure are defined | `demangle()` overloads | `COMPLETE` | async-signal-safe and Rust-name parsing are not admitted without a named foreign-runtime/crash-handler consumer |
| explicit named factory registration | startup composition of 4--32 built-in or already-loaded inference backends | nullable or `Result<T>` factory, optional help, three priorities | Registry owns key/help; creator is a no-capture `noexcept` function pointer; registration reports invalid/OOM/conflict | `register_creator()` / `reserve()` | `COMPLETE` | captured arbitrary callables would add ownership and type-erasure cost and are not admitted |
| lookup-only factory discovery | llama.cpp/ggml `reg_by_name`: resolve a descriptor/factory once and cache the typed result | mutable and frozen lookup; present/missing; factory plus help/priority metadata | result is a value snapshot; key/help remain borrowed from Registry; mutable lookup is synchronized and frozen lookup is lock/allocation-free | `Registry::find()` / `FrozenView::find()` | `COMPLETE` | dynamic module image lifetime remains with its owner, not the name table |
| lookup-and-create with failure | direct named construction during startup | nullable missing or `NotFound` plus creator `Status` | creator executes outside Registry mutex and may re-enter; Args carry explicit state | mutable/frozen `create()` | `COMPLETE` | Registry does not own the returned object |
| collision and precedence | deterministic composition independent of link/registration order | lower/equal/higher priority and replacement | equal priority fails, lower is no-op, higher atomically replaces callable/help while old owned bytes remain valid | `RegistryPriority` / `register_creator()` | `COMPLETE` | no dependency solver or implicit global policy |
| metadata and enumeration | startup diagnostics and help listing | size, existence, help, entry/key visitor, early stop | mutable iteration takes an O(n) snapshot so callback runs unlocked; frozen iteration borrows immutable storage without allocation | `size()` / `has()` / `help()` / `for_each()` / `for_each_key()` | `COMPLETE` | ordering is deliberately unspecified; lookup-only edge is tracked separately |
| startup-to-service publication | mutate during owner-controlled startup, then publish immutable registry | idempotent freeze and cacheable frozen view | Registry outlives views and in-flight operations; registration/reserve after freeze fail; no unfreeze | `freeze()` / `frozen()` / `FrozenView` | `COMPLETE` | unregister and module quiescence conflict with this one-way lifecycle |
| backend descriptor and device catalogue | ggml enumerate backend registrations, then enumerate/name/type/query devices and capabilities | backend descriptor, devices, selection and typed capability queries | domain types and object/resource lifetime belong to compute/backend owner | factory result naturally returns a typed descriptor; remaining route is future compute/module work | `OUT_OF_SCOPE` | current unfinished Compute is not claimed complete or routed by this Util round |
| dynamic module discovery and unload | ggml search/load/score modules and retain their image while factories/objects exist | loader handle, symbol contract, duplicate load, unload/quiescence | `platform::DynLib` owns the OS handle mechanism; module manager owns image/object quiescence | loader mechanism exists in Platform; full lifecycle is future compute/module policy | `OUT_OF_SCOPE` | ggml itself records unsafe-unload concerns; a generic Registry must not pretend to solve them |
| source/static/shared C++ delivery | direct-source engine plus installed static/shared consumers | selected component closure and shared C++ import/export | no stable cross-toolchain ABI and no Native FFI | current `hyle::util` component and package sample on Windows/WSL/macOS | `COMPLETE` | all three C++ shapes pass; foreign-language ABI remains deliberately absent |

## Capability / gap ledger

| ID | requirement class | demand kind | exact consumer / peer / standard source | normal shape | semantic / artifact / delivery owner | ownership/failure/concurrency | complexity/resource target | coverage state | Action | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `UTIL-V3-DEMANGLE-01` | `MUST` | existing public contract + mature peers | HYLE diagnostics; Folly `23ebefdf...`; Boost `78505eab...` | cold ordinary-thread symbol formatting | Util / Util+platform ABI / build-package | owning or caller buffer; fallback and required size; platform calls may allocate/serialize | O(input+output); no hot-path promise | `COMPLETE` | `KEEP` | — | replay plain/mangled/fallback/truncation/overlap and three delivery shapes |
| `UTIL-V3-REG-02` | `MUST` | north-star + ggml foundation | llama.cpp `9723942...`; ggml `36da5713...` explicit backend registration | register 4--32 typed factories and handle conflict/OOM | Util / Util+Container+Mem / build-package | Registry owns strings; no-capture creator; mutable operations synchronized | copy O(key+help), expected-amortized O(1) insertion; no extra metadata | `COMPLETE` | `KEEP` | — | invalid, OOM, priorities, replacement, reentry and stress |
| `UTIL-V3-FIND-03` | `MUST` | north-star + ggml real consumer | `ggml_backend_reg_by_name()` and llama backend selection cache | resolve named typed factory once without constructing | Util / Util header / build-package | return value snapshot; missing is `nullopt`; borrowed key/help; no callback under lock | O(key)+expected O(1); frozen zero allocation/lock; no new per-entry state | `COMPLETE` | `FIX` | high | present/missing, metadata/callable identity, mutable/frozen, embedded NUL and TSan all pass |
| `UTIL-V3-CREATE-04` | `MUST` | existing public contract + north-star | direct startup construction with nullable or `Result<T>` failure | lookup and invoke once | Util / Util header / build-package | creator outside lock; Registry does not own result | O(key)+expected O(1) plus creator; frozen no allocation/lock | `COMPLETE` | `KEEP` | — | find/create equivalence, missing/creator failure, move-only args |
| `UTIL-V3-PUBLISH-05` | `MUST` | definition + service lifecycle | immutable service discovery after engine initialization | freeze once, cache view, read/enumerate | Util / Util header / build-package | one-way publication; Registry outlives view/in-flight calls | freeze O(1)/zero allocation; frozen lookup expected O(1) | `COMPLETE` | `KEEP` | — | freeze race, post-freeze failures, visitor/creator reentry and TSan |
| `UTIL-V3-CONTAINER-06` | `MUST` | direct dependency contract change | Container v2 `StringView`/`FlatHashMap` replacement | owned strings indexed through non-owning lookup views | Util semantic / Container mechanism / build-package | hash/equality length-aware; returned views live to Registry destruction | expected O(1), no fallback `std::unordered_map` | `COMPLETE` | `KEEP` | — | embedded-NUL/non-NUL query, long key/help, allocator failure and package closure |
| `UTIL-V3-DOC-07` | `MUST` | C++ usability defect | user guide already says resolve/cache but exposed no lookup-only route | choose register/find-or-create/freeze and cache correctly | Util / docs / docs | borrowed fields, factory/image lifetime, concurrency and failure are explicit | no runtime cost | `COMPLETE` | `FIX` | medium | copyable guide journey and stale API scan pass |
| `UTIL-V3-DELIVERY-08` | `MUST` | build/package contract | source-primary engine plus installed static/shared sample | consume lookup-only and compiled demangle from selected `util` root | Util / Util+dependencies / build-package | exact headers/artifacts/runtime staging; C++ only | no accidental extra component or FFI layer | `COMPLETE` | `FIX` | high | source/static/shared selected-component smoke passes on three platforms |
| `UTIL-V3-CAPTURED-09` | `GENERAL` | Folly peer difference | Folly domain registries can own polymorphic factories | register stateful callable | possible future owner / no artifact / no delivery | requires allocator, destruction, exception and capture lifetime policy | would add type erasure/allocation or another type family | `OUT_OF_SCOPE` | `DEFER` | — | trigger: two real consumers cannot express state through explicit Args/owner |
| `UTIL-V3-SIGNAL-10` | `GENERAL` | Folly demangle difference | optional libiberty signal-safe/Rust parsing | crash-handler symbol rendering | future crash/symbol owner | async-signal safety and platform parity required | fixed caller storage; no allocation/lock | `OUT_OF_SCOPE` | `DEFER` | — | trigger: named crash or foreign-runtime consumer with a cross-platform contract |
| `UTIL-V3-MODULE-11` | `GENERAL` | ggml domain difference | load/score/best/unload/proc-address/device APIs | compose dynamic backend modules | compute/module semantic / Platform loader mechanism / build-package | image must outlive factory/object/thread; safe quiescence required | startup path; no generic Registry tax | `OUT_OF_SCOPE` | `REJECT` | — | belongs to future Compute/module task; Util reopen only if that owner proves a generic primitive is missing |

- 现有公开入口反向追踪：`RegistryPriority`、`Registry::Creator`、`EntryView`、`FrozenView`，以及
  `register_creator`、`reserve`、`freeze`、`frozen`、`has`、`create`、`help`、`size`、`for_each`、
  `for_each_key` 和 FrozenView 的 `create/has/help/size/for_each` 均映射到上表；`demangle` 的
  `const char*` buffer、owning `std::string`、`type_info` 与固定数组 overload 均映射到 symbol
  presentation。`detail::RegistryResultTraits` 和 `detail::demangle_visit_internal` 是实现 seam，不是用户
  能力。
- 已有能力完整性检查：取得、登记、冲突、构造、枚举、冻结、失败、ownership、并发与销毁均有完整
  路线；唯一半项是“指南要求缓存 factory，但 API 只能立即构造或 O(n) 枚举”。
- 语法消费者 / 语义重复扫描：仓内没有 Util Registry 产品消费者；package sample 是交付 fixture。
  Compute 当前固定 `DeviceTable` 是未完成领域实现，不在本轮迁移或拿来证明 Util 完整。Obs 的动态 sink
  registry 具有 unregister/quiescence 语义，不能错误替成启动期单向冻结 Registry。
- public/internal/higher-layer 裁决：名字→factory 元数据与 publication 属公开 Util；demangle sink 属
  internal；backend/device/select/load/unload/proc-address 属 Platform 机制与未来 Compute/module policy。
- 本轮冻结 `FIX` 集（唯一产品批次）：
  1. 在 mutable `Registry` 与 `FrozenView` 增加 `find(StringView) -> std::optional<EntryView>`；返回值复制
     creator/priority 并借用 Registry-owned key/help，保持 expected-O(1)，frozen 无锁/无 allocation，且不加
     常态 metadata；
  2. 增加 present/missing、优先级替换、creator identity、mutable/frozen、并发与生命周期测试；
  3. package sample 与 architecture/user/performance 文档改为真实“find once, cache typed factory”旅程，
     再验证三种 C++ 交付。
- 相邻模块 backlog：未来 Compute/module task 负责 backend descriptor/device catalogue、动态 module 搜索、
  score/best、proc-address、image/object/thread quiescence；这不是本轮可假称 `ROUTED` 的已完成产品。

## Owner 六问直答

### 1. 第三方产品路径

| Dependency / locked identity | direct / indirect | class | semantic / mechanism / decision / delivery owner | public leak / trigger |
|---|---|---|---|---|
| Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` | indirect through Container | product | Util owns Registry semantics; Container/Highway own affected string/bit kernels; Container dispatch decides those kernels; build/package delivers | no vendor type leaks; Container dependency/codegen change triggers the affected lookup route |
| xxHash `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53` | indirect through Container | product | Container owns hash mechanism/decision; build/package delivers | no vendor type leaks; hash semantic/codegen change triggers review |
| wyhash `e4764a0b637d34d3421a7760affada9288b625a8` | indirect through Container | product | Container owns hash mechanism/decision; build/package delivers | no vendor type leaks; hash semantic/codegen change triggers review |
| simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b` | indirect through Container closure but unused by Registry's normal lookup | product | Container owns UTF-8 semantics/dispatch; build/package delivers | no vendor type leaks; only a selected-component/delivery concern here |
| llama.cpp / ggml / Abseil / Folly / Boost locked above | no product link | peer/domain evidence | their projects own their artifacts; HYLE Util owns this decision | no public leak; peer release alone does not reopen |

- Util has no direct third-party product implementation. Standard library mutex/atomic/optional/string and compiler ABI or Windows
  DbgHelp are platform contracts, not third-party HYLE modules. Itanium demangle may use `__cxa_demangle`; Windows uses
  DbgHelp under an internal mutex/retry loop.

### 2. 推理引擎能力完整度

- llama.cpp clean reference: `9723942adc518b43c4b95dc4dce6906903eb5e09`, remote checked 2026-08-31.
- standalone ggml clean reference: `36da57138425487184aa1da2eee2cde155909c6f`, remote checked 2026-08-31.
- llama.cpp vendored ggml: registry shape compared separately; differences observed are outside this Util capability family.

| real journey / foundation difference | demand | HYLE route / owner | audit verdict |
|---|---|---|---|
| explicit backend registration and failure | admitted | `register_creator`; Util | `COMPLETE` |
| registration count/get/enumeration | admitted | `size/for_each`; Util | `COMPLETE` |
| backend registration/factory by name | admitted | `find`; Util | `COMPLETE` |
| construct selected backend and propagate failure | admitted | nullable/`Result<T>` `create` or cached creator; Util | `COMPLETE` |
| device enumeration/name/type/caps | not a generic Registry demand | typed backend descriptor; future Compute/backend owner | `OUT_OF_SCOPE` |
| module load/search/score/best/unload | not a generic Registry demand | Platform `DynLib` mechanism plus future Compute/module lifecycle | `OUT_OF_SCOPE` |
| proc-address | domain extension schema | typed backend descriptor/module owner | `OUT_OF_SCOPE` |

- 当前证据下未处置 `MUST`：`NO`。`UTIL-V3-FIND-03` 已完成；不按 ggml API 数求并集。

### 3. 拟真、公平性能

- Tier `P0`；正常任务是启动期 4--32 项登记/解析/冻结和冷路 demangle，source/static/shared 都验证，
  但没有 profile 证明 token/kernel 热路径。
- 最终 verdict：`P0 STRUCTURAL PASS`；timing peer comparison `NOT_APPLICABLE`。candidate 的 find 直接
  复用同一 `FlatHashMap::find`，没有新 table/field/allocation/binary symbol；freeze 仍为 O(1)/零
  allocation，frozen find 无 mutex 和 lifecycle atomic。三平台真实交付与 sanitizer 均通过。
- 允许的最终 claim 只会是 P0 cost structure；禁止声称 timing SOTA、全平台更快或所有操作不劣。

### 4. 旧 C ABI / Native FFI 清洁度

- 全局 policy 为 `N/A`，本轮无已准入 FFI slice。
- Util 产品树没有公开 C header/schema、bridge source、未装饰 C export、C/managed sample、专用 option/
  target/package metadata。compiled demangle 只产生 C++ decorated symbol；平台 ABI 调用保持 private。
- 最终 verdict：`CLEAN`。Windows `dumpbin`、Linux `nm -D`、macOS `nm -gU` 均只见既有 buffer
  demangle 与 internal sink 两个 decorated C++ symbol；header-only find 没有新增 export。

### 5. Source / static / shared 构建与导入

| Journey | audit state | required candidate evidence | limit |
|---|---|---|---|
| source embedding | `PASS` | Windows/WSL/macOS selected `util` source consumer executes find/cache/invoke and demangle | C++23 only |
| installed static | `PASS` | three-platform relocated installed consumer and complete transitive closure | same-toolchain C++ contract |
| installed shared | `PASS` | three-platform relocated consumer; Windows staging and Unix loader-relative lookup | no stable cross-toolchain C++ ABI claim |

- Central graph is `util -> PUBLIC core,error,mem,container`; selected headers exclude `util_core`. No build-system change
  is admitted unless candidate validation proves a concrete contract defect.

### 6. C++ API 易用性与文档

| User journey | current route | ownership/failure/concurrency | audit verdict |
|---|---|---|---|
| register an explicit typed factory | natural `register_creator` | Registry owns strings; `Status`; startup synchronization | `PASS` |
| resolve and cache a factory without construction | `find()` then copy typed Creator | value snapshot; borrowed strings/module lifetime documented | `PASS` |
| construct directly | natural nullable/`Result<T>` `create` | creator outside lock; caller owns result | `PASS` |
| publish and enumerate service registry | natural `freeze/FrozenView/for_each` | borrowed view; one-way lifecycle documented | `PASS` |
| demangle into owning or bounded output | natural overloads | caller owns output; fallback/truncation documented | `PASS` |

- 默认路径不引入 static-init macro、`std::function` 或动态 module policy；显式 reserve/freeze 是 startup opt-in。
- user guide 现在给出 register/find-or-create/freeze 的可复制旅程，architecture 解释 owner 与负空间，
  performance 只记录 P0 结构事实，implementation 保存本轮审计；总体 verdict：`PASS`。

## Candidate

- product commit：`b2b7bd679293d748d996662414c01c4cbcefcbda`；tree：
  `1a95077cccbcabe4701a50873143297739adcb59`；
- public contract：mutable `Registry` 与 `FrozenView` 新增
  `find(StringView) -> std::optional<EntryView>`；missing 为 `nullopt`，不调用 factory；其余公开入口与
  demangle contract 不变；
- ownership / lifetime / concurrency / failure：返回值复制 creator/priority，key/help 借用
  Registry-owned storage；旧 snapshot 跨 priority replacement 可读但不得活过 Registry。mutable find 在
  现有 mutex 内取得 snapshot，frozen find 无锁；缓存 Creator 仍要求 module image 存活。lookup missing
  不属于运营错误，因此使用 optional，而 create 的 nullable/`Result<T>` 失败合同不变；
- algorithm / complexity / metadata：复用同一 `FlatHashMap` 和 `IndexValue`；没有新 table、entry 字段、
  allocation、indirection、virtual dispatch 或 binary symbol。mutable/frozen find 均为 O(|key|)+expected
  O(1)，frozen 零 allocation/lock；普通 Registry 只保留既有 lifecycle acquire；
- direct consumer：package sample 真实执行 `find -> cache Creator -> invoke`，同时继续覆盖 direct
  `create` failure/missing 与 owning/buffer demangle；Compute/Obs 未迁移，原因见 owner boundary；
- pre-candidate Windows evidence：MSVC 19.50 Release selected `util` closure build PASS；Registry 21/21、
  Util aggregate、public-header self-containment PASS；source embedding、relocated installed shared/static
  consumer PASS。全 selected-closure ctest 36/37，唯一失败是独立 build/package `loc_report_test` 的 synthetic
  fixture 未包含其 canonical boundary 新增的 `scripts/hyle-unix.sh`，与 Util product/path 无关；本轮不越界
  修改公共工具，终止前把它作为 build/package method gap 单独裁决；
- harness/tool identity：`closure-correctness-v1`；scratch
  `C:\hyle-scratch\util-runtime-v3\build-win-candidate` 与
  `C:\hyle-scratch\util-runtime-v3\smoke-win-candidate`；candidate 身份与 harness 分离。

`AUDITED -> CANDIDATE` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\util-runtime-v3\module-status-before-candidate.json`，SHA-256
`4656C98A0AF31A0A41EA6213118028DDE24B7F00F85DAC329ED9F63F73E37AEA`。

## Verification manifest

### Correctness risk matrix

| Risk | Verdict | command / evidence | limit |
|---|---|---|---|
| normal | `PASS` | W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21 Release `hyle_util_test` 均 34/34；public-header self-contained 三平台通过 | HYLE 尚无已完成推理引擎，不把 package fixture 冒充生产 engine |
| empty / zero | `PASS` | missing find/nullopt、missing nullable/Result create、empty/null symbol、zero buffer 全部通过 | empty help 与 absent help 都是 empty view，存在性用 find/has 区分 |
| capacity / boundary / overflow | `PASS` | maximum StringView length widening、long help OOM、truncation/required-size/overlap，以及 current harness `a81d3689542ff3dcb35718ced95a390ba7bc7acb` 的 embedded-NUL exact/prefix lookup | StringView 公共上限仍为 `uint32_t`，恶意 hash collision 最坏界不改变 |
| OOM / system failure / invalid input | `PASS` | allocator fail-at-every-step、failed priority replacement、invalid key/creator/priority、demangler fallback；find 在 fault allocator 冻结后不增加 allocation call | standard string/demangler 自身 OOM 服从既有合同 |
| move / destroy / lifetime | `PASS` | Registry 不可 move；owned key/help、旧 EntryView 跨 replacement 稳定、FrozenView/EntryView 借用与 Creator module lifetime 均测试或写明 | caller 必须在 Registry 析构前 quiesce；无 generic unload |
| concurrency / reentrancy | `PASS` | mutable registration/find/read stress、creator/visitor reentry、8-reader frozen find；W1-WSL TSan 无 report | mutable path 是 mutex-based，不声称 lock-free |
| cancellation / close | `NOT_APPLICABLE` | synchronous lookup/demangle and one-way startup publication have no asynchronous operation | dynamic module drain/unload belongs to future module owner |
| property / differential / fuzz / stress | `PASS` | 128 length-aware named entries with concurrent writers/readers, priority transition, mutable/frozen equivalence and three-platform ABI fallback | no coverage-guided fuzz campaign is claimed |
| ASan / UBSan / TSan | `PASS` | W1-WSL fresh GCC 14 ASan/UBSan/TSan candidate suite 33/33, then embedded-NUL harness case under all three and full current Registry under TSan | mimalloc TSan aligned-OS-memory fallback warning is not a sanitizer finding |
| real ISA / backend | `NOT_APPLICABLE` | Registry has no ISA/backend execution; real MSVC/Itanium/Apple ABI demangle routes ran on their platforms | backend/device completeness is not claimed by Util |

### Aggregate gates

| Gate | Verdict | evidence | limit |
|---|---|---|---|
| scope / architecture | `PASS` | finite three-item FIX completed; public trace and util/platform/future-compute owners explicit | Compute/module implementation remains future work |
| correctness / sanitizer | `PASS` | risk matrix above; three compilers/platforms and WSL ASan/UBSan/TSan | no Formal campaign |
| component / source / static / shared | `PASS` | selected component smoke on W1-WIN, W1-WSL and M1-MAC; relocated source/shared/static consumers each run | C++23 same-toolchain SDK contract |
| direct consumer | `PASS` | package sample performs register/freeze/find/cache/invoke, direct create failure/missing and both demangle shapes | north-star engine remains demand evidence, not a finished HYLE consumer |
| P0 engineering performance | `PASS` | O(|key|)+expected O(1), frozen no lock/allocation, no new metadata/export; `benchmarks-util.md` | timing comparison is N/A; no SOTA claim |
| usability / docs | `PASS` | architecture/user/implementation/performance roles, copyable journey, ownership/concurrency/unload traps and stale API scan | no Native FFI |
| independent red team | `PASS` | findings below all fixed/disproved/accepted/out-of-scope | bounded to frozen capability and candidate |

Windows selected-closure ctest also ran all 37 configured tests: 36 passed and the unrelated
`hyle_loc_report_test` failed because its build/package synthetic fixture omits the now-canonical
`scripts/hyle-unix.sh` boundary path. Direct immutable candidate LOC reporting passes with `self_check=pass`; the
fixture defect neither reads Util product code nor affects selected component/package journeys, so it is recorded as a
build/package method gap rather than misreported as a Util failure. No closure checker/build-system edit was made here.

## 独立红队

- reviewer / date / budget：candidate 后独立第二遍 contract review / 2026-08-31 / 30 分钟；
- capability / contract / complexity：逐项攻击 lookup-only 是否其实需要 value registry、unregister、captured
  callable 或第二张表；均无当前准入证据。find 使用现有 index，复杂度档与资源合同成立；
- correctness / lifetime / concurrency：检查 freeze race、mutable replacement、borrow invalidation、module
  unload、embedded NUL、OOM 与 callback-under-lock；
- dependency / build / delivery：检查 Container length/hash 语义、中央 PUBLIC 闭包、三种交付、三平台
  decorated exports 和旧 C/FFI 残留；
- engineering performance validity：没有运行 timing 或用旧 N=256 点值塑造 API；只签 P0 结构成本。

| Finding | severity | decision | evidence |
|---|---|---|---|
| EntryView 若直接借 Entry 字段，会被 priority replacement 并发改写 | high | `disproved` | candidate 在 mutex 下复制 creator/priority/help view；旧 help arena bytes 不回收，replacement snapshot test + TSan 通过 |
| cached Creator 可能在 shared module unload 后悬空 | high | `accepted/out-of-scope` | user/architecture 明确 module owner 必须先排空对象、线程、reader，再销毁 Registry/unload；通用 Registry 不提供假安全 unregister |
| Container 迁移可能把 embedded NUL key 当 C-string prefix | medium | `fixed in harness` | `a81d3689...` 增加 exact length/prefix negative test，三平台与三 sanitizer 通过；product 已按 length hash/copy |
| lookup-only 可能暗中 allocation、lock frozen phase 或扩大 Entry | high | `disproved` | `entry_view()` 只构造 optional value；FrozenView 直接现有 table find；无新 member/table/symbol，fault allocator call count 不变 |
| new public method 可能扩大 shared C++ export 或恢复伪 C ABI | high | `disproved` | 三平台 symbol scan 仍只有两个既有 decorated demangle symbols；无 extern-C/header/schema/option/sample |
| backend/device/load/best/proc-address 未复制 ggml | medium | `out-of-scope` | typed factory 可返回 descriptor；设备与 module policy 分别归未来 Compute/module 和 Platform mechanism，不能下沉到通用 Registry |
| full selected ctest 的 loc-report fixture failure | low for Util / high for build tooling | `out-of-scope method gap` | direct `loc-report.py --module util --commit ... --self-check` PASS；fixture只缺 build/package synthetic path，另轮处理 |

## Closure decision

- 当前各适用 gate 全部 `PASS`，无 Util product blocker；
- known limits：无 generic dynamic module manager/unregister/quiescence；factory 是 no-capture `noexcept`
  function pointer；EntryView/FrozenView 借用 Registry；P0 不给 timing/SOTA/Formal 声明；当前 HYLE 没有
  已完成推理引擎消费者；
- reopen triggers：真实 consumer 证明 captured-state/unregister/lookup 语义缺失；Registry API/layout/
  algorithm/Container direct dependency/delivery 改变；可复现 correctness/security/lifetime defect；profile 将
  lookup 提升为 P1/P2；或成熟同合同 peer 证明更好复杂度/material consumer advantage；
- candidate LOC（immutable `b2b7bd679293d748d996662414c01c4cbcefcbda`，self-check PASS）：全库
  `404969`，自实现含测试 `97015`，自实现不含测试 `60362`，Util canonical implementation `1267`；
- raw local evidence：W1-WIN `C:\hyle-scratch\util-runtime-v3\{build-win-candidate,smoke-win-candidate}`；
  W1-WSL `/root/hyle-scratch/util-runtime-v3/{build-release,build-asan,build-ubsan,build-tsan,smoke-linux}`；
  M1-MAC `/Users/kiro/hyle-scratch/util-runtime-v3/{build-release,smoke-mac}`。这些是可丢弃 replay artifact，
  不是 Formal bundle。

`CANDIDATE -> VERIFIED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\util-runtime-v3\module-status-before-verified.json`，SHA-256
`CAD782DA73F6D6C63ABB0D7E39A04CA659A8E98AD8C824C5C2F40054E778EFC7`。

- terminal receipt：`CLOSED`。产品 candidate 保持
  `b2b7bd679293d748d996662414c01c4cbcefcbda` /
  `1a95077cccbcabe4701a50873143297739adcb59`；七项 gate 全部 `PASS`，没有 blocker。receipt 所在提交
  产生后由后续 registry 提交记录完整 end/anchor identity；
- 建议机械状态：`CLOSED`。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
