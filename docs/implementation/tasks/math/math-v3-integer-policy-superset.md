# Math v3：通用整数策略能力补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由 owner 明确要求扩充通用基础能力。它不把模型算子、tensor 类型或 SIMD 数学下沉到
> scalar Math；`SCOPED/AUDITED` 前不修改产品源码。

## Reopen evidence（`REOPENED`）

- baseline：`e3202ae42d30062da049938c1f267bde83e47324`；Math v2 的冻结产品身份仍是
  `1a9e680cb5a25ccc0c20786f4635625eef2d7ab0` / tree
  `010323120bc6eb07b5a07e7d3e6de1725968d314`；
- trigger ID：`MATH-INTEGER-POLICY-SUPERSET-2026-09-05`；
- owner 在 2026-09-05 正式准入 Math 与 Util-core 的通用基础能力扩展，并明确拒绝把 Obs/Error 的
  推理引擎策略误下沉到基础模块。这命中 Math v2 已签 reopen trigger：“mixed-result、saturation 或
  bulk-product journey 被正式准入”；
- 当前 Math 只有同类型非 bool 整数的 checked add/subtract/multiply 与 ceil/round。冻结记录已经明确
  mixed signed/unsigned conversion、checked divide/remainder、compound/bulk arithmetic 和 saturating
  policy 不在 v2 合同内；它们现在成为本轮待审问题域；
- HYLE 自身在 `mem/lifetime_plan.cc`、`mem/composite_buffer.cc`、`container/pod_vector.cc`、
  `platform/cpu.cc`、`platform/numa.cc` 等多处重复手写 size product/sum、十进制 multiply-add、窄化和
  overflow 判断。这里只把这些作为通用需求证据；已冻结模块不在本轮被静默迁移；
- 北极星形状是由 signed tensor extents、quant block size、alignment 和 backend narrow fields 导出
  element/row/byte/workgroup counts。当前锁定 standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 与 llama.cpp
  `4d9176092d00586775af140581bb0b558ddc4389` 都反复执行 product、round-up 和 narrow conversion；
  这是模型无关的整数边界任务，不是让 Math 拥有 tensor；
- Folly `65749da412d92fe27d87ae45a8ef670720712858` 的 `folly/lang/CheckedMath.h` 提供 checked
  divide/modulo、compound add 和 multiply-add，并在 container/memory/IO 中直接消费；C++26
  saturation arithmetic（WG21 P0543R3，2026 rename direction P4052R0）证明 saturating operation/cast
  是独立、成熟的整数策略族。它们是发现输入，不授权机械复制名字或不完整边界。

## 初步能力边界

| family | current coverage | initial disposition |
|---|---|---|
| representability-checked integral conversion | `MISSING` | `MUST`；明确 signed/unsigned 与 bool/enum 负空间 |
| checked divide/remainder | `MISSING` | `MUST`；零除数和 signed minimum / `-1` 必须无 UB |
| checked compound and extent aggregation | `PARTIAL` | `MUST`；审计 multiply-add、sum/product 与 empty identity 的自然形状 |
| saturating integral operation/conversion | `MISSING` | `MUST`；按当前成熟语义冻结 HYLE 名称，不冒充未来标准 ABI |
| floating transcendental, activation, quant/dequant, vector/tensor math | other owners | `OUT_OF_SCOPE`；分别属于 standard/SIMD/Compute/consumer |
| arbitrary precision, wide/carry chains and decimal parsing | separate domains | `GENERAL`；没有当前准入，不借本轮扩成数值大全 |

## 初步成本与停止边界

- tier 初定 `P0 / closure-correctness-v1`：单次 scalar operation 必须 constexpr、O(1)、零动态资源；
  span aggregate 必须 O(n)、单遍、零动态资源；shipping runtime 优先 compiler builtin/native lowering；
- 不建立亚纳秒排名，不因 benchmark 增加 public overload、常态 metadata 或动态 dispatch；
- 本轮只在 `SCOPED` 冻结用户任务、API 草图、最多两个 strong peers、correctness/codegen/交付矩阵和
  有限预算；`AUDITED` 才冻结精确 `FIX` 集。

## Closure card（`SCOPED`）

- component / owner / round ID：`math` / scalar integral control-plane owner /
  `math-v3-integer-policy-superset`；task kind：`REOPEN`；
- baseline commit：`e3202ae42d30062da049938c1f267bde83e47324`；本轮登记提交不改变该
  baseline 的 Math 产品实现；
- demand IDs：
  - `MATH-V3-CONVERT-01`：把外部 signed/unsigned width 转成调用者指定整数类型时，必须一次返回精确值
    或不可表示，而不是让每个 parser 重复 `in_range + cast`；
  - `MATH-V3-DIVREM-02`：checked quotient/remainder 必须同时处理零除数与 signed minimum / `-1`，不执行
    undefined behavior；
  - `MATH-V3-COMPOUND-03`：extent、row bytes 与 range end 需要完整的 checked multiply-add 和有界
    sum/product route，不能要求用户保存 unchecked intermediate；
  - `MATH-V3-SATURATE-04`：标准化的 fixed-width saturating add/subtract/multiply/divide/cast 是独立整数
    policy，不能用 wrapped 或 checked-result API 冒充；
- normal workload：读取不可信的模型/文件 shape，依次把 signed extent 转成 `size_t`、求 element product、
  计算 `base + count * stride` 与 padded count，任何不可表示步骤立即失败；工程留出是 fixed-width counter/
  像素或信号值采用饱和而非回绕的单值运算。Math 不拥有 tensor、文件格式或 DSP policy；
- direct dependencies / consumers：保持 `PUBLIC core`，不新增第三方产品依赖。当前产品语法消费者为 0；
  `mem/platform/container/time` 中的重复 guard 与锁定 llama.cpp/ggml 只是 demand evidence，冻结模块不在
  本轮静默迁移；`hyle::math` façade 仍直接聚合本 factor；
- reuse candidates：C++23 `std::in_range`、现有 `try_add/try_subtract/try_multiply`、compiler overflow
  builtin/MSVC wide multiply、当前 C++ saturation 语义。SIMD、Mem alignment、Error 与动态 dispatch
  均不适用；
- tier：`P0 / STRUCTURAL`。所有 scalar primitive 必须 `constexpr noexcept`、O(1)、零 allocation/lock/
  syscall/state/dispatch；aggregate 必须 O(n)、单遍、零动态资源，调用者提供 view；timing peer comparison
  明确 `NOT_APPLICABLE`；
- material margin / resource budget：timing margin `N/A`；每个 scalar 结果只携带返回值及 checked route 的
  success state，aggregate 不复制输入、不分配；
- budget：从 `SCOPED` 起总计 120 分钟，其中审计/API 35 分钟、实现 40 分钟、codegen 与三平台/交付
  验证 45 分钟；最多一个产品实现批次、一次 codegen 纠正和一次受影响重跑，不改公共仪器；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC` / `closure-correctness-v1`；Formal `NONE`；
- strong inputs（每族不超过两个）：checked/compound 使用 Folly
  `65749da412d92fe27d87ae45a8ef670720712858` `folly/lang/CheckedMath.h` 与 WG21 P3161R6 作为
  未入标准的边界发现；saturation 使用已进入 C++ 工作草案的 P0543R3 语义与 2026 P4052R0/current-draft
  命名；conversion 使用 C++23 `std::in_range` 与同一 current-draft saturation cast。锁定需求输入是
  llama.cpp `4d9176092d00586775af140581bb0b558ddc4389`、其 ggml tree
  `f4ed259b5708475fb9a853998591f2d4d2624c4f` 和 standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`；
- provenance：`NORTH_STAR_JOURNEY + STANDARD + PROVISIONAL_ARCHETYPE`；不冒充已有 HYLE 推理引擎 trace；
- API 语义档位草图：checked exact route 返回 `optional<T>`；saturating route 返回离精确数学结果最近的
  representable `T`，零除不是 saturation；aggregate 明确 empty identity。精确 public 名称与 overload
  集在 `AUDITED` 冻结；
- correctness gates：全部支持整数宽度和 signedness、zero/one/min/max、每个 overflow 方向、mixed
  representability、zero divisor、min/-1、empty/singleton span、constexpr、small-domain differential、
  WSL UBSan、三工具链 release code shape、header self-contained；无 ownership/concurrency/cancel/OOM 的
  项写 `NOT_APPLICABLE`；
- delivery gates：factor-only configure/test；`hyle::math` source embedding、relocated installed static 和
  shared consumer；header-only factor 不制造 archive/export symbol；
- usability/docs gates：普通 C++ journey、checked/saturating/trusted 选择表、类型与 empty/zero 陷阱；
  architecture/user/performance/task 各守职责；无 C/Native FFI surface；
- expected output：双向 capability map、public trace、冻结 API/`FIX` ledger，一个 product candidate，
  三平台 correctness/codegen、三种交付、七问、红队、限制与 terminal receipt；
- stop：`AUDITED` 后只执行有限 `FIX`。全部 `MUST` 为 `COMPLETE/ROUTED` 且适用门真实通过则关闭；任何
  correctness/codegen/交付 blocker 在预算内无法消除则具名 `BLOCKED`，不继续发明能力或 benchmark。

`REOPENED -> SCOPED` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-math-util-scoped-68398f1e.json`，SHA-256
`FEFC5C89EBC242AC9447EFC87342137B1E7B37E37D75751290CE19E130CABE5D`。

## Capability-family coverage map（`AUDITED`）

| family | admitted normal journey | required semantics | current route | coverage | decision |
|---|---|---|---|---|---|
| exact checked binary arithmetic | untrusted extent/offset/byte composition | same-type non-bool integer；no UB；exact value or failure | `try_add/subtract/multiply` | `COMPLETE` | `KEEP` |
| representability-checked conversion | signed model/file dimensions to `size_t` or backend-width fields | source and destination are non-bool integers；no implicit promotion；exact value or failure | caller repeats `std::in_range` + cast | `MISSING` | `FIX` as `try_integer_cast<To>(value)` |
| checked division/remainder | block counts、strides and partition calculations | zero rejected；signed minimum / `-1` never evaluated with built-in `/` or `%` | no Math route | `MISSING` | `FIX` as `try_divide` / `try_remainder` |
| checked compound/aggregate arithmetic | `base + count * stride` and shape product/sum | left-to-right exact steps；first unrepresentable step fails；empty sum/product identities are 0/1 | callers compose helpers or hand-write loops | `PARTIAL` | `FIX` as multiply-add plus span sum/product |
| saturating integer policy | fixed-width counters/signal-like values whose contract is clamp, not reject or wrap | nearest representable value；division by zero remains a violated precondition；signed min / `-1` clamps high | no route before C++26 library availability | `MISSING` | `FIX` as five explicitly named helpers |
| bit operations, ordinary trusted scalar arithmetic | normal language/standard-library work | standard semantics | C++23 | `ROUTED` | do not wrap |
| wide/carry arithmetic、parsing、floating、activation、quant/tensor/SIMD math | different precision, syntax or backend contracts | domain-owned | standard/SIMD/Compute/consumer | `ROUTED/OUT_OF_SCOPE` | no Math-v3 API |

### Public-surface reverse trace and duplicate scan

- The seven existing entries in `scalar.h` map only to checked same-type add/subtract/multiply and non-negative
  ceil/round. They remain natural, have no benchmark-only parameters or metadata, and receive no semantic change.
- `std::in_range` is the primitive used by the new conversion helper, not a complete user journey: repeating the
  predicate and cast lets the checked branch accidentally retain the original value or choose a different target type.
  A named optional-returning conversion therefore closes a real composition boundary instead of wrapping for branding.
- Folly's multiply-add and checked divide/modulo establish the compound family, but HYLE does not copy its arity
  overload union. `try_multiply_add(a,b,c)` means checked `a*b` followed by checked addition of `c`; an intermediate
  outside `T` fails even if later cancellation could make the mathematical final value representable.
- `try_remainder(min,-1)` returns exact zero without executing `%`; `try_divide(min,-1)` fails. Both reject a zero
  divisor. This distinguishes an exactly representable remainder from an unrepresentable quotient.
- `try_sum(span)` and `try_product(span)` are single-pass views over caller storage, short-circuit on the first failure,
  and define empty identities as `T{0}` and `T{1}`. Initializer-list ownership, variadic folds and tensor shape types are
  deliberately absent.
- Saturation uses HYLE's existing readable verb convention:
  `saturating_add/subtract/multiply/divide/cast`. It follows current C++ saturation semantics but does not claim a
  standard ABI or import the standard's abbreviated spelling. Untrusted divisors must use `try_divide`; the trusted
  saturating divide entry fail-fast checks nonzero.

## Capability / gap ledger and frozen `FIX` set

| ID | class | exact action | complexity/resource contract | verification / negative space |
|---|---|---|---|---|
| `MATH3-CONVERT-01` | `MUST` | add `try_integer_cast<To>(From)` | constexpr O(1), zero resource; use `std::in_range` | all signedness/width boundaries; bool and enum rejected |
| `MATH3-DIVREM-02` | `MUST` | add checked quotient and remainder | constexpr O(1), no UB or dispatch | zero and min/-1 matrix; remainder exact-zero case |
| `MATH3-COMPOUND-03` | `MUST` | add checked multiply-add, span sum and span product | scalar O(1); aggregate O(n), one pass, early exit, no allocation | empty/single/boundary/property/constexpr; no arbitrary folds |
| `MATH3-SATURATE-04` | `MUST` | add saturating add/subtract/multiply/divide/cast | constexpr O(1), zero resource; compiler-native lowering where available | exhaustive small domains + boundaries; zero divisor is fail-fast |
| `MATH3-CODEGEN-05` | `MUST` | extend release probes for representative 32/64-bit paths | no allocation, exception, indirect call, runtime dispatch or division fallback for add/sub/cast | MSVC/GCC/AppleClang inspection; timing is N/A |
| `MATH3-DELIVERY-06` | `MUST` | expose the unchanged header-only factor through source/static/shared Math journeys | Core-only INTERFACE component | exact factor and relocated SDK smoke; no binary/FFI symbol |
| `MATH3-DOC-07` | `MUST` | update architecture and user choice/pitfall guide | current behavior only | checked vs saturating vs trusted and type/zero/empty traps |

The finite product batch is exactly these seven entries. It may not change Core, downstream frozen products, component
edges, common instruments, or add heterogeneous arithmetic, parsing, big integers, model operators or SIMD kernels.

## API / usability freeze

- All public templates accept only standard non-`bool` integral types. Conversion is the only heterogeneous entry and
  names its result type explicitly as `try_integer_cast<To>(from)` / `saturating_cast<To>(from)`.
- Checked operations return `std::optional<T>` and never return a wrapped, partial or saturated value. Saturating
  operations return `T` and never use `optional`; trusted built-in arithmetic remains the shortest route.
- `try_divide` and `try_remainder` are for untrusted divisors. `saturating_divide` has a nonzero precondition enforced by
  the existing Core fatal assertion, because there is no mathematically meaningful clamp for division by zero.
- Aggregate APIs accept `std::span<const T>` by value, allocate nothing and do not retain it. Empty identities are part
  of the contract, not an implementation accident.
- The factor remains `<hyle/math/scalar.h>` plus `hyle::math`, header-only and `PUBLIC core`. There is no C API, Native
  FFI, stable cross-toolchain C++ ABI or peer type in the public surface.

## Owner seven questions（`AUDITED`）

1. **Third-party product path:** none; standards, Folly and ggml/llama.cpp are evidence only.
2. **Capability completeness:** the four admitted generic integer-policy families are the only concrete gaps; model,
   vector, text and arbitrary-precision work stays with its semantic owner.
3. **Internal reuse/acceleration:** all new compound/saturation helpers reuse existing checked primitives or C++23
   primitives; no lower frozen HYLE module has a faster missing route. Downstream migration is owner-controlled.
4. **Performance:** P0 only. Best possible complexity is O(1) scalar and O(n) aggregate; the candidate still owes
   three-toolchain code shape and must not claim timing SOTA.
5. **C ABI/build:** no legacy C/Native FFI surface exists. The component graph and actual target both remain Core-only
   INTERFACE; source/static/shared users get the same templates from matched headers.
6. **C++ usability:** policy is visible in the verb and return type; no out-parameter, hidden promotion, tensor wrapper
   or benchmark seam is admitted.
7. **Docs/red team:** architecture/user/performance/task roles and boundary/UB traps are frozen but not yet updated or
   verified, so no gate is pre-marked PASS.

`SCOPED -> AUDITED` uses the prior registry snapshot
`C:\public\project-hyle\_validation\registry-before-math-util-audited-ff6b5daf.json`, SHA-256
`27C669D4B93877DB3F5E07909374948388CB66E92FF491EB9C77BDBFDFA4F6D4`.
