# math-v1 边界重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `math` 的当前 closure card。旧
> [`math-01-scalar.md`](math-01-scalar.md) 只记录 clay 迁移史，不证明当前四个公开函数的需求、
> 边界或正确性。机械状态只看
> [`module-status.json`](../../module-status.json)。本轮先裁决 scalar math 与 SIMD 的边界；
> capability map 和有限 `FIX` 集进入 `AUDITED` 前不改产品代码。

## Closure card（`CANDIDATE`，scope 不变）

- component / owner / round ID：`math` / scalar numeric primitive owner / `math-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`8652aa53c46ee28ff002bf301906c5ab4a148765` /
  `1d95449146887a73931e21d1687e2500feb436e6`
- 进入原因与 demand/defect ID：
  - `MATH-BOUNDARY-01`：现有 target 是 clay math 迁移剩余，没有从用户任务正向定义 scalar
    numeric 与 `hyle::simd` 的分工；
  - `MATH-INT-01`：本地 CCCL 与推理 shape/tile 任务都证明非负整数 ceil-division / round-up
    是通用 primitive，但现有 `round_up_to_multiple` 在结果不可表示时无提示回绕；
  - `MATH-FLOAT-01`：`sgn` 与 `approx_eq` 没有产品消费者，NaN/无穷/近零语义与
    Eigen 等成熟形状不能等价，不能因为已有测试就保留。
- 真实 consumer task / normal workload：
  - 当前 HYLE 产品源对 `hyle::math` 的语法消费为 0；这是需求分类事实，不是自动删除理由；
  - 北极星推理任务使用 element count / lane width / block size 计算 chunk 数与可表示的
    padded extent；工程留出是二维 tiling / chunk partition；
  - 单值标准数学函数直接路由 C++23，lane/vector 与 whole-buffer 数值计算路由
    `hyle::simd`。
- 范围：`include/hyle/math/**`、`src/math/**`、`tests/math/**`、math component/package journey、
  scalar-vs-SIMD 边界与 architecture/user 文档。
- 非目标：SIMD lane/backend/batch 实现、线性代数、FFT、几何、随机、特殊函数库、模型算子、
  C++23 标准库镜像、Native FFI/C API、稳定 C++ ABI 与 Formal 证书。
- 直接依赖 / 直接消费者：当前产品依赖为空，产品消费者为空；测试是唯一语法消费者。
  本轮只对边界做决定，不倒逼 SIMD 依赖 math，也不为保留 target 制造假边。
- tier 与理由：`P0`。当前函数是 header-only `constexpr` scalar primitive，未被 profile 证明为
  独立热点；本轮守复杂度、可表示性与 release codegen，不跑纳秒胜负。
- material margin / resource budget：P1 margin 不适用。保留的 primitive 必须零分配、零常态
  metadata/lock/syscall；ceil division 不得比同语义成熟形状多于一次整数除法。
- 总调查预算 / 性能预算：90 分钟 / 15 分钟 P0 codegen。若删除整个组件的交付半径超出预算，
  以具名 build/package blocker 终止，不扩大到 SIMD 本体。
- 最多复现、机制诊断、产品修改次数：一次边界/溢出复现，一次 release codegen 检查，
  一个有限产品修改批次及一次重跑。
- machine profile：`W1-WIN`、`W1-WSL`、`M1-MAC`。
- instrument profile：`closure-correctness-v1`。
- strong peers（每族最多两个）：
  - 标准可替代面：C++23 `<bit>/<cmath>/<numbers>/<limits>` 和 HYLE 当前所支持标准；
  - integer ceil/round：CCCL `132670eba92ca5bbddb2d479e76045184ed885a5` 的
    `cuda::ceil_div` / `cuda::round_up`；
  - fuzzy floating comparison：Eigen `26f009dbbe1f8520439e6b1b6c564d897e80ace6` 的
    scalar `isApprox` / `isMuchSmallerThan`；只用于语义对照，不借用 Eigen 为产品依赖。
- 无同合同 peer 时的预算及禁止声明：只做复杂度、分配与 codegen 结构门；不声称 scalar
  math 或 SIMD 全面 SOTA，不为无消费者的浮点近似语义制造 benchmark。
- 适用 gates：public-header self-contained；normal/zero/max/divisor/overflow 与保留浮点的
  NaN/infinity/signed-zero 语义；`constexpr` oracle/property；MSVC/GCC/AppleClang；source 与 installed
  static/shared SDK 中真实的 INTERFACE target journey；P0 release codegen；用户选型文档；独立红队。
- 预期输出：capability-family map；逐 symbol public trace；保留/路由/删除裁决；有限 `FIX`
  集；candidate 与三平台/交付/P0 证据；architecture/user 指南；红队、LOC 与 terminal receipt。
- 停止条件 / 可接受限制：每个公开符号有准入来源、完整语义或明确路由；不存在
  `PARTIAL + KEEP`；SIMD/math 选型不再要求用户猜测。无产品消费者、header-only 无独立
  binary 及不提供 Native FFI 都可接受，但不得伪造 journey 或证书。

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|---|
| C++ scalar constants, bit and elementary math | C++23 defines the ordinary scalar route | integral/floating overloads and standard exceptional values | value-only; standard library owns semantics and implementation | `<bit>/<cmath>/<numbers>/<limits>` | `ROUTED` | HYLE must not rename or mirror this surface |
| Non-negative integral ceiling division | tensor/block partition; CCCL; 76 ggml SYCL call sites plus a signed llama shape use | same-type signed or unsigned, excluding bool; checked input and fail-fast normal route | no state; divisor must be positive and signed dividend non-negative | current unsigned `div_ceil` | `PARTIAL` | zero divisor is undiscoverable; signed shape requires ad-hoc helper/cast; name diverges from selected peer |
| Integral round-up to a representable multiple | padded extents, storage/block sizing; CCCL | same-type signed/unsigned; zero divisor and unrepresentable result | no state; checked and fail-fast routes; one division, no allocation | current unsigned `round_up_to_multiple` | `PARTIAL` | multiplication silently wraps although its comment calls the path overflow-safe |
| Threshold sign classification | no admitted product consumer; threshold is domain policy | NaN, signed zero and tolerance policy would be required | consumer owns tolerance and invalid-value policy | current `sgn(x, eps)` | `PARTIAL` | NaN becomes zero; arbitrary default epsilon; route to explicit consumer comparison |
| Approximate floating comparison | existing public symbol only; Eigen demonstrates one of several incompatible policies | absolute/relative/ULP choice, zero, NaN, infinity and tolerance validation | numerical algorithm/test owner chooses error model | current multiplicative `approx_eq` | `PARTIAL` | name hides zero behavior; equal infinities compare false; no consumer justifies choosing this policy globally |
| Lane/vector/whole-buffer math | SIMD user journeys | fixed/scalable vector, runtime/exact dispatch, dtype and accuracy tier | `simd` owns implementation and delivery | `hyle::simd` | `OUT_OF_SCOPE` | completeness is adjudicated in simd-v1, not copied into scalar math |
| Linear algebra, FFT, geometry, special functions and model operators | domain libraries and upper layers | domain-specific | mature dependency, compute or consumer owns policy | outside math | `OUT_OF_SCOPE` | no generic wrapper or umbrella module |

### Public-surface trace

| Current symbol | reverse demand | audit verdict |
|---|---|---|
| `div_ceil<T>(a,b)` | `MATH-INT-01` | `PARTIAL -> FIX` as `ceil_div` plus checked route; no compatibility alias because the project is pre-release and has no consumer |
| `round_up_to_multiple<T>(n,m)` | `MATH-ROUND-01` | `PARTIAL -> FIX`; add checked overflow/invalid-input route and fail-fast normal route |
| `sgn<T>(x,eps)` | clay migration only | `REJECT`; threshold classification remains explicit consumer policy |
| `approx_eq<T>(a,b,tol)` | clay migration only | `REJECT`; route to the consumer's absolute/relative/ULP contract or a domain library |

Repository syntax scan at baseline found no product include or call of `hyle::math`; only
`tests/math/scalar_test.cc` consumes the header. By contrast, the pinned llama/ggml source contains repeated
ceil-division in tensor, block and launch-shape calculations. This is north-star and mature-domain evidence, not a
claim that HYLE already has a production caller.

## Capability / gap ledger

| ID | class | demand kind / exact source | normal shape | owner / failure | complexity/resource | coverage | action / severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|
| `MATH-STD-01` | `MUST` | language standard: C++23 scalar numeric facilities | user calls the standard facility directly | standard library owns semantics/artifact; math docs own routing | standard complexity; zero HYLE cost | `ROUTED` | `KEEP` route | stale-wrapper scan; do not mirror standard names |
| `MATH-INT-01` | `MUST` | north-star shape partition; CCCL `132670eb`; llama/ggml `75844307` | `ceil_div(value, divisor)` and `try_ceil_div` for untrusted values | math header; normal invalid input is invariant/fail-fast, checked route returns `optional` | O(1), one quotient/remainder operation, zero metadata | `PARTIAL` | `FIX` / correctness+usability | zero/max/signed/property/codegen; no mixed-type or negative-rounding policy |
| `MATH-ROUND-01` | `MUST` | padded extent/block sizing; CCCL `round_up` | `round_up_to_multiple` plus checked `try_` | math header; checked route rejects zero/negative divisor, negative value and unrepresentable result | O(1), one quotient/remainder operation, zero metadata | `PARTIAL` | `FIX` / correctness | max-boundary/property/codegen; arbitrary alignment ownership remains mem when power-of-two address alignment is intended |
| `MATH-FLOAT-01` | `GENERAL` | existing clay-derived public surface; Eigen `26f009db` semantic counterexample | consumer chooses explicit threshold/tolerance model | consumer/domain library | O(1), but no common correctness contract | `PARTIAL` | `REJECT` / API correctness | remove both symbols/tests; user guide explains std/domain/SIMD routes |
| `MATH-SIMD-01` | `GENERAL` | module boundary and existing SIMD product | lane or batch task names its accuracy/dispatch contract | simd semantic/artifact/delivery owner | vector/backend-specific | `OUT_OF_SCOPE` | `REJECT` from math | no math-to-simd dependency; simd-v1 audits completeness |
| `MATH-DELIVERY-01` | `MUST` | central component/package contract | select `math`, include public header, link `hyle::math` | math target; build/package delivers selected closure | header-only, no fabricated archive | `PARTIAL` | `FIX` / delivery | add real sample; source and relocated static/shared SDK journeys; declare real `PUBLIC core` edge |

- **Existing capability completeness.** The two integer tasks are admitted but incomplete at invalid/overflow
  boundaries. The two floating helpers are not incomplete requirements to expand: their only source is old public
  code, and the correct action is removal because tolerance policy has no universal owner.
- **Syntax consumers / semantic duplicates.** Product syntax consumers: none. Standard replacements are already
  documented; `mem::align_up` remains the power-of-two address-alignment owner and is not a substitute for arbitrary
  numeric multiple rounding. SIMD owns vectors and buffer kernels.
- **Public/internal/higher-layer decision.** Retained integer functions stay public scalar value primitives. No hidden
  internal implementation or compiled artifact is needed. Floating threshold/comparison policy moves upward; vector
  math never moves downward.
- **Adjacent backlog.** SIMD completeness and accuracy remain wholly in simd-v1. No core product change is requested;
  math merely acquires a real `PUBLIC core` target edge to use the already-closed invariant primitive.

### Frozen `FIX` set

Exactly one product batch is authorized:

1. replace `div_ceil` with same-type non-bool integral `ceil_div` and `try_ceil_div`;
2. make `round_up_to_multiple` fail-fast on invalid/unrepresentable input and add
   `try_round_up_to_multiple` without overflow;
3. remove `sgn` and `approx_eq` with no compatibility aliases;
4. declare `math -> PUBLIC core`, rewrite correctness/property tests, add the installed consumer and current
   architecture/user guidance.

No SIMD source, backend, benchmark, central instrument or unrelated build option may change in this round. A new
capability request returns the task to scope; it is not appended to this batch.

## Candidate

- product commit / tree：`753ad5f304f52727c3ba925c9aac0a30f941502e` /
  `93f7bf1a996d6beabebe2b42801b086c483a18e6`。
- 公开合同与 user journey：删除 `div_ceil/sgn/approx_eq`；新增同类型非 bool integral
  `ceil_div/try_ceil_div`，保留并补全 `round_up_to_multiple`，新增
  `try_round_up_to_multiple`。普通入口用于已验证程序不变量，checked 入口处理不可信
  shape/count 并返回 `std::optional<T>`。
- ownership / lifetime / concurrency / failure：全部是 stateless `constexpr` value function，没有 borrow、
  allocation 或共享状态。负 signed value、非正 divisor/multiple 与不可表示 round-up 在
  `try_` 路线返回 `nullopt`，普通路线使用 Core release-surviving fail-fast。
- 算法 / 复杂度 / 常态 metadata：O(1)；quotient/remainder 避免 `a+b-1` 先行溢出，round-up
  在加法前比较 `max-increment`。MSVC 19.50 `/O2` consumer 的 ceil、round 与 checked-round
  每条路径均只有一次 `div`，零 heap/lock/indirect dispatch；普通路径只保留到 cold fail-fast
  sink 的失败分支。
- 直接消费者 / 依赖：产品消费者仍为 0；新 package sample 是交付 fixture。公开头直接
  include Core assert，中央图与真实 target 同时投影 `math -> PUBLIC core`；没有 math-to-simd 边。
- candidate tests：W1-WIN MSVC 19.50 Release + WERROR 组件闭包配置/构建通过；math normal +
  header-self-contained 2/2；selected source/shared/static relocated consumer 各1/1；全中央组件图合同
  `PASS`。WSL/Mac 与候选后最终门尚未在本节写成通过。
- harness / tool identity：`closure-correctness-v1`；Windows build root
  `C:\hyle-scratch\math-v1-win-60811bab`，selected/package root
  `C:\hyle-scratch\math-v1-selected-smoke-60811bab`；一次性 codegen probe/source SHA-256
  `c6dacce2afe6f31fc31951a7f18910a3e5841739f0ab9f8fcec33be58380cc58`，assembly SHA-256
  `15f7f0f6dda07b74fd316158f6d870212d32411ff06a0bd94d33c886dd20f244`。这些 harness 事实不改变
  product identity。

## Verification manifest

只使用 closure card 预声明的 gates；未运行项不记 `PASS`。

## 独立红队

在 candidate 冻结后执行，30 分钟上限。

## Closure decision

尚未裁决。
