# Math v4：checked discrete sizing

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只补齐两个由当前推理参考路径和 owner 明确准入的 scalar control-plane 边界。它不把 Math
> 扩成 SIMD、tensor 或一般数值库；进入 `AUDITED` 前不修改产品源码。

## Reopen evidence（`REOPENED`）

- baseline commit：`cc2bb5b648cfa43dc5b72a32db318eca5d4a80e7`；Math v3 产品源码、测试和
  构建边界相对冻结候选无漂移；
- trigger ID：`MATH-CHECKED-DISCRETE-SIZING-2026-09-06`；
- owner 明确决定：若 HYLE SIMD、Highway 或 xsimd 没有同合同能力，则在 Math 增加
  checked scalar `lcm` 与 `bit_ceil`。公开面扫描确认三者都没有这种“精确结果或不可表示”的
  scalar 合同；SIMD 的 `Ceil/CeilInt` 是浮点 lane rounding，不是同一能力；
- 当前锁定 llama.cpp `4d9176092d00586775af140581bb0b558ddc4389` 的 model-load split
  granularity 路径直接使用 `std::lcm`；standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 的 argsort/queue sizing 路径具有
  next-power-of-two 计算。两者都由模型或工作尺寸导出控制值；
- `std::lcm` 和 `std::bit_ceil` 在结果不可表示时不提供 checked failure contract。HYLE v3 已以
  `std::optional<T>` 表达同类 untrusted integer policy，因此缺口属于既有 Math 边界，而不是新的
  模块或 benchmark 形状；
- 这命中 v3 的 reopen trigger：“locked llama.cpp/ggml journey 证明新的 model-independent scalar
  control-plane MUST”以及“missing common scalar integer policy”。

## 初步边界

| Family | current coverage | initial disposition |
|---|---|---|
| checked least common multiple | `MISSING` | `MUST`；零、负数、signed minimum 与结果溢出必须无 UB |
| checked next power of two | `MISSING` | `MUST`；负值与不可表示的下一幂返回失败，0/1 明确定义为 1 |
| trusted scalar GCD/LCM/bit operations | standard route | `ROUTED`；继续直接使用标准库，不做品牌包装 |
| lane/buffer discrete math | no admitted consumer | `OUT_OF_SCOPE`；有真实批量旅程时由 SIMD owner 独立准入 |

## 初步成本与停止边界

- tier 初定 `P0 / closure-correctness-v1`：`try_bit_ceil` 必须 constexpr O(1)，并降为原生位计数
  形状；`try_lcm` 必须 constexpr O(log(min(|a|,|b|)))，采用无符号 magnitude 与标准 GCD，二者
  都零 allocation/lock/syscall/HYLE runtime state or dispatch；portable 标准库允许使用其有界 ISA
  bit-selection helper；
- 本轮最多两个 public API、一个实现批次和一次 codegen 纠正；不修改 Core、SIMD、下游冻结模块、
  组件依赖或公共仪器；
- 亚纳秒 timing、批量 SIMD、公开 `gcd` wrapper、任意精度、模型算子、Native FFI 和稳定 C++ ABI
  均不是本轮工作。

`CLOSED -> REOPENED` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-math-v4-reopened-cc2bb5b6.json`，SHA-256
`B19D70C122F190BE868E7129CCB958D4D8FDEAD7A09ACE4B70EF5D8638070CF5`。

## Closure card（`SCOPED`）

- component / owner / round ID：`math` / scalar integral control-plane owner /
  `math-v4-checked-discrete-sizing`；task kind：`REOPEN`；
- baseline commit：`cc2bb5b648cfa43dc5b72a32db318eca5d4a80e7`；
- demand IDs：
  - `MATH4-LCM-01`：由不可信 model/work dimensions 求 non-negative LCM，零和 signed minimum 不触发
    UB，结果超出目标类型时返回失败；
  - `MATH4-BITCEIL-02`：由不可信 count/capacity 求最小的 representable power of two，负值与结果溢出
    返回失败，0 和 1 都得到 1；
- normal workload：模型加载时合并 split/tensor granularity，以及为排序或有界队列规划 power-of-two
  capacity。输入是单个 fixed-width scalar 控制值；不是对数据 buffer 做逐 lane 数值计算；
- 范围 / 非目标：只新增 `try_lcm(T,T)` 与 `try_bit_ceil(T)`，并补齐测试、codegen、架构/用户/
  P0 证据；不新增 trusted wrapper、`gcd`、SIMD overload、tensor/model 类型或下游迁移；
- 直接依赖 / 直接消费者：保持 `math PUBLIC core` 和 header-only INTERFACE。llama.cpp/ggml 是锁定
  north-star evidence，不成为产品依赖；本轮没有已冻结 HYLE consumer migration；
- 待复用机制：`std::gcd`、`std::bit_ceil/std::bit_floor`、无符号模运算 magnitude、现有
  `NonBooleanIntegral` 与 `optional<T>` policy。SIMD/Highway/xsimd 扫描为非同合同，不接入；
- tier：`P0 / STRUCTURAL`。`try_bit_ceil` 为 constexpr O(1)，`try_lcm` 为 constexpr
  O(log(min(|a|,|b|)))；二者零 allocation/lock/syscall/HYLE state or dispatch；portable standard-library
  helper 可以按目标选择 scalar bit instruction；timing peer comparison
  `NOT_APPLICABLE`；
- resource budget：返回值只含 `optional<T>` 的值与 success state，不增加常态 metadata 或全局状态；
- 总预算：90 分钟，其中审计/API 20、实现/单测 25、三平台/codegen/交付 35、文档/红队/收尾 10；
  最多一个产品实现批次、一次 codegen 纠正、一次受影响重跑，不改公共仪器；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC` / `closure-correctness-v1`；Formal
  `NONE`；
- strong inputs：C++23 `<numeric>/<bit>` 的定义性语义；locked llama.cpp
  `4d9176092d00586775af140581bb0b558ddc4389` model-load LCM journey；locked standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` argsort/queue pow2 journey。Highway/xsimd 只用于
  路由反证，不作为 timing peer；
- provenance：`NORTH_STAR_JOURNEY + STANDARD`；没有 HYLE inference runtime trace，不声称端到端收益；
- API 语义草图：保持 v3 的 same-type non-`bool` integral 与 `std::optional<T>` failure route；LCM
  允许负 operand 但结果非负，bit-ceil 拒绝负值。精确边界和名称在 `AUDITED` 冻结；
- correctness gates：zero/one/min/max、全部 overflow 方向、signedness、small-domain differential、
  constexpr、WSL UBSan、三工具链 release code shape、header self-contained；ownership/OOM/concurrency/
  cancellation 为 `NOT_APPLICABLE`；
- delivery gates：factor-only configure/test，`hyle::math` source embedding、relocated installed static/shared
  consumer；header-only factor 不制造 binary journey 或 C/Native FFI symbol；
- usability/docs gates：checked/trusted/SIMD 选择、signed/zero/overflow 陷阱和自然 compile-only journey；
- expected output：有限 capability ledger、冻结 API、一个 product candidate、三平台 correctness/codegen、
  三种 C++ 交付、七问、独立红队、P0 证据和 terminal receipt；
- stop：`AUDITED` 后只实现两个 FIX。适用门全部通过则 `CLOSED`；correctness/codegen/交付 blocker 在
  预算内无法消除则具名 `BLOCKED`，不扩成一般 number-theory 或 SIMD 研究。

`REOPENED -> SCOPED` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-math-v4-scoped.json`，SHA-256
`72B3552D9A92323A5DDA5EEB1E8FAF679A9BF30EC3029A2EC1A9228FB25D5CB8`。

## Capability-family coverage map（`AUDITED`）

| Family | admitted user journey | required semantics | current route | coverage | decision |
|---|---|---|---|---|---|
| exact checked arithmetic/conversion/division/aggregate | untrusted scalar sizing | exact result or failure | Math v3 helpers | `COMPLETE` | `KEEP` |
| checked least common multiple | combine model split, block and tensor granularity | same-type non-bool integral；non-negative mathematical result；zero identity；no UB；unrepresentable result fails | raw `std::lcm` or local guards | `MISSING` | `FIX` as `try_lcm` |
| checked next power of two | capacity/padding planning from untrusted count | same-type non-bool integral；0/1 -> 1；negative or unrepresentable result fails | local loop or guarded `std::bit_ceil` | `MISSING` | `FIX` as `try_bit_ceil` |
| trusted GCD/LCM/bit operations | prevalidated scalar input | standard semantics | C++23 standard library | `ROUTED` | no HYLE wrapper |
| lane/buffer discrete sizing | batched independent inputs | SIMD-owned lane semantics and mask/failure representation | no admitted consumer | `OUT_OF_SCOPE` | defer until real SIMD journey |
| floating/transcendental/activation/quant/tensor/model math | data plane or model policy | domain-owned | standard/SIMD/Compute/consumer | `ROUTED/OUT_OF_SCOPE` | no Math API |

### Reverse trace and duplicate scan

- Existing public Math entries all serve checked/saturating scalar integer policy or trusted ceil/round convenience;
  none exposes a benchmark-only parameter, hidden global state, allocation or dynamic dispatch.
- HYLE SIMD's `Ceil/CeilInt` are floating lane rounding and its row/buffer kernels have no checked scalar LCM or
  next-power-of-two contract. Locked Highway `63f4b16cda024e2adff8a18dee81e067f8a87d61` and xsimd
  `3b3151ff665514a00a437cc985f47b7de99b772b` likewise provide no same-contract public route. Pulling SIMD into
  Math would add ISA/backend machinery without accelerating a single scalar value.
- HYLE Container's current `std::bit_ceil` call sites already establish their own representability guard, so silently
  migrating a frozen component would add dependency churn without fixing a defect. SIMD's packed-size formula is a
  domain-specific checked decomposition and remains with SIMD.
- A public `try_gcd` is rejected: standard GCD is always representable for the unsigned magnitude domain needed by
  this implementation, while no admitted consumer needs a separately checked result.

## Capability / gap ledger and frozen `FIX` set

| ID | class | exact action | complexity/resource contract | verification / negative space |
|---|---|---|---|---|
| `MATH4-LCM-01` | `MUST` | add same-type `try_lcm(T,T) -> optional<T>` | constexpr O(log min magnitude), zero resource; unsigned magnitude + `std::gcd`, divide before checked multiply | signs, zero, min/max, exhaustive 8-bit oracle; bool/mixed rejected |
| `MATH4-BITCEIL-02` | `MUST` | add `try_bit_ceil(T) -> optional<T>` | constexpr O(1), zero resource; guard representability before `std::bit_ceil` | 0/1, powers/between powers, signed negative, highest representable power, overflow; bool rejected |
| `MATH4-CODEGEN-03` | `MUST` | add representative signed/unsigned release probes | no allocation, exception, lock, syscall or HYLE runtime dispatch; direct bounded standard helper is permitted and bit-ceil must reach native bit-count shape | MSVC/GCC/AppleClang object inspection; timing N/A |
| `MATH4-DELIVERY-04` | `MUST` | verify exact header-only factor through source/static/shared Math journeys | unchanged Core-only INTERFACE closure | factor/header and relocated package smokes; no binary/FFI export |
| `MATH4-DOC-05` | `MUST` | update architecture/user/P0 evidence and terminal record | describe current contract, selection and traps only | checked vs trusted vs SIMD choice and complexity |

The finite product batch is exactly these five entries. It cannot modify Core, SIMD, downstream components, common
instruments, component edges or introduce a third-party product dependency.

## API / usability freeze

- status：`FROZEN`；public surface is exactly `try_lcm(T,T)` and `try_bit_ceil(T)` in
  `<hyle/math/scalar.h>` and the existing `hyle::math` façade;
- both accept one deduced standard non-`bool` integral type and return `std::optional<T>`. Mixed operands do not
  compile; character integer types retain the existing Math domain;
- `try_lcm` follows the mathematical/standard sign convention: negative operands are accepted, the successful result
  is non-negative, and either zero operand yields zero even when the other is signed minimum. A positive magnitude not
  representable by signed `T` fails;
- `try_bit_ceil(0)` and `(1)` return 1; signed negative input fails; values above the largest representable power of
  two fail before calling the standard primitive, so its representability precondition is never violated;
- trusted, prevalidated scalar code continues to call `std::lcm`/`std::bit_ceil` directly. Future batch/lane demand
  gets a distinct SIMD API with an explicit per-lane failure shape; it will not make scalar Math depend on SIMD;
- freeze identity：this `AUDITED` ledger / owner decision of 2026-09-06. Internal implementation may change only if
  names, exact semantics, complexity, state/resource cost and Core-only delivery remain fixed.

## Owner 七问直答（`AUDITED`）

1. **Third-party product path:** none. C++23 primitives and compiler lowering are the implementation contract;
   llama.cpp, ggml, Highway and xsimd are evidence only.
2. **Inference capability:** the two admitted north-star scalar sizing journeys become complete with the two FIXes;
   tensor/model policy and lane/buffer kernels remain correctly routed. No other unhandled `MUST` was found.
3. **Internal reuse/acceleration:** reuse existing Math constraint/failure policy and standard bit/GCD primitives.
   SIMD is semantically inapplicable to one scalar input; no lower HYLE facility offers a better route.
4. **Performance:** P0 structural only. Best admitted complexity is O(1) bit-ceil and logarithmic standard-GCD LCM; timing,
   end-to-end inference and universal SOTA claims are forbidden.
5. **C ABI/FFI:** Math has no admitted Native FFI slice, C header, bridge source or undecorated installed export.
   Test-only `extern "C"` probes are not product ABI.
6. **Build/delivery:** central graph and actual target are both `math PUBLIC core`, header-only. Source/static/shared
   package consumers receive the same templates through selected headers; verification remains pending.
7. **C++ usability/docs:** optional failure matches all existing checked helpers; trusted and SIMD alternatives are
   explicit and no benchmark seam enters the signature. User/architecture/P0 evidence updates remain pending.

`SCOPED -> AUDITED` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-math-v4-audited.json`，SHA-256
`E4FBA77F1E6095B9859734FA11B64CA18DCF3B651E13B2700FD2AEA61EAAB37D`。

## Product candidate（`CANDIDATE`）

- candidate commit：`686ec1c0215fa7a808ee470fe103ac62a1050759`；tree：
  `1cb4bb81451ba016e09aecce563b3b591648d7a0`；
- `try_lcm` 使用无符号 magnitude、`std::gcd`、先约分和既有 `try_multiply`，安全覆盖 signed minimum、
  zero identity、目标正值边界和 full-width product overflow；
- `try_bit_ceil` 在调用 `std::bit_ceil` 前按 `T` 的最大 representable power 做 guard，因此没有触发
  标准 primitive 的不可表示前置条件；
- API、类型约束和五项 FIX 均与 `AUDITED` freeze 一致；没有 component edge、binary artifact、第三方
  product path、C/Native FFI surface、常态 metadata 或下游迁移；
- red-team delivery review found that the first implementation commit's installed consumer still exercised only v3
  entries. The replacement candidate adds both v4 calls to the existing natural package sample; public implementation
  bytes are unchanged from `a57194d8f33611d899bbb4aafb7df994317d7340`；
- pre-candidate W1-WIN MSVC 19.50 Release/WERROR 与 W1-WSL GCC 14.2 Release/WERROR+UBSan 已通过 Math
  unit/header tests。GCC optimized object 的 bit-ceil 是 bounded guard + `bsr`，LCM 是 inlined standard
  binary-GCD shape + checked full-width multiply；
- M1-MAC、三种交付、candidate-identity replay、完整 P0 evidence 与独立红队仍待执行；上述初步结果
  不构成 `VERIFIED/CLOSED`。

`AUDITED -> CANDIDATE` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-math-v4-candidate-a57194d8.json`，SHA-256
`1177B91178A414088996ADD2B31FD29F0B94F52EC90E1E60753B935807A16B3E`。

候选身份的同态刷新（`CANDIDATE -> CANDIDATE`）使用前态快照
`C:\public\project-hyle\_validation\registry-before-math-v4-candidate-686ec1c0.json`，SHA-256
`328431FC18BC837B8CDCE0A30336E9E3080F68981A4DAB13D148E4A2428F0124`。

## Verification（`VERIFIED`）

Candidate identity remains `686ec1c0215fa7a808ee470fe103ac62a1050759` / tree
`1cb4bb81451ba016e09aecce563b3b591648d7a0`. Candidate-to-HEAD path diff is empty for the Math public
header, tests, probe, CMake and package consumer.

### Correctness risk matrix

| risk | verdict | evidence / boundary |
|---|---|---|
| normal and sign convention | `PASS` | positive/negative LCM and power/exact-between-power bit-ceil on three compilers |
| empty / zero / one | `PASS` | either-zero LCM including signed-min peer；bit-ceil 0/1 -> 1 |
| min/max / overflow / standard preconditions | `PASS` | signed minimum magnitude, largest representable powers, next-power/product overflow all fail before invalid primitive evaluation |
| type-domain / invalid input | `PASS` | same-type non-bool concepts, mixed/bool rejection, signed-negative bit-ceil, character integral route |
| property / differential | `PASS` | exhaustive signed and unsigned 8-bit LCM plus unsigned/signed 8-bit bit-ceil oracle |
| constexpr / header closure | `PASS` | compile-time boundaries and per-header test on MSVC/GCC/AppleClang |
| sanitizer | `PASS` | exact-candidate W1-WSL GCC 14.2 Release + UBSan |
| ownership / OOM / concurrency / cancellation | `NOT_APPLICABLE` | value-only synchronous templates own and allocate nothing |

### Aggregate gates

| gate | verdict | evidence / boundary |
|---|---|---|
| scope / architecture | `PASS` | exactly two frozen scalar APIs；Math remains Core-only and independent of SIMD |
| correctness / sanitizer | `PASS` | all three unit/header runs plus WSL UBSan matrix above |
| source / static / shared | `PASS` | selected-component smoke on W1-WIN, W1-WSL and M1-MAC；relocated consumers execute both APIs |
| direct/north-star consumer | `PASS` | package model-sizing journey plus locked llama.cpp LCM and ggml pow2 evidence；no fake HYLE inference claim |
| reuse / acceleration | `PASS` | existing `try_multiply`, standard GCD/bit primitive and compiler ISA lowering used；SIMD correctly N/A |
| P0 engineering performance | `PASS` | [`benchmarks-math-v4`](../../evidence/performance/benchmarks-math-v4.md) |
| usability / docs | `PASS` | same checked optional vocabulary, guide choice/traps, architecture boundary and package sample |
| independent red team | `PASS` | separate checklist pass below; one delivery-test gap fixed before replacement candidate |

### Validation receipts

- W1-WIN：MSVC 19.50 Release/WERROR Math tests + header isolation；selected `math` source/relocated
  shared/static smoke；codegen object SHA-256
  `F524BEEE609975E06D130EC20C76245824FB9C532C85924BF6451AB87994949F`；
- W1-WSL：GCC 14.2 exact-candidate Release/WERROR+UBSan tests/header；source/relocated shared/static smoke；
  non-sanitized codegen object SHA-256
  `CAE89783BFE56F74172B70F77120E0343DC6939540EAB9F3819747D3BD359B48`；
- M1-MAC：candidate tar SHA-256
  `7297D51DE9DF17B958606AFB901E97D6A694C1E7A9550B69C179D970CA96F6C2`；AppleClang 21
  tests/header and source/relocated shared/static smoke；codegen object SHA-256
  `40DB7278DFECB91C475B3B24B54C755A68ED481C1CF50F068F74E3188A32EAC9`；
- all runs selected only the declared `core,math` closure. No general all-module test or timing benchmark was used.

## 独立红队

- reviewer / date / budget：Codex separate evidence-led pass / 2026-09-06 / 10 minutes；不是外部人类审计；
- capability/API：checked gap is backed by standard preconditions and two locked inference journeys. Public names have
  no benchmark knobs; trusted and batch routes remain explicit negative space；
- correctness：reviewed signed-min magnitude, zero short-circuit, multiply overflow, bit-ceil representability and
  character types; exhaustive oracles and UBSan found no remaining defect；
- dependency/build：central graph and target both remain `math -> PUBLIC core`; installed surfaces have no stale C ABI,
  runtime library or third-party leak；
- performance：comparison class remains P0 structural. MSVC's bounded standard bit helper is recorded rather than
  falsely called fully inline; no timing/SOTA inference is made；
- finding：the first implementation candidate's package sample did not call the new APIs (`medium`, fixed). Candidate
  `686ec1c0` adds the natural calls and all three exact-candidate delivery journeys pass. No unresolved issue remains.

## Closure decision（verification stage）

- all seven aggregate gates are `PASS`; no blocker or unhandled `MUST` remains；
- known limitations and reopen triggers are prepared for the terminal registry but do not broaden this round；
- terminal receipt has not yet been committed, so this section recommends mechanical state `VERIFIED`, not `CLOSED`；
- LOC snapshot at candidate `686ec1c0215fa7a808ee470fe103ac62a1050759`：repository code 452,775；
  self-written with tests 117,111；self-written without tests 74,346；Math implementation 420；self-check `PASS`。

`CANDIDATE -> VERIFIED` 的前态快照保存在 canonical checkout 外：
`C:\public\project-hyle\_validation\registry-before-math-v4-verified.json`，SHA-256
`BDEBF806622B341E61976ABB5282585C04054907E45E5B57D6B34D40F5C0433B`。
