# math-v2 checked arithmetic 收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> **当前状态：`CANDIDATE`。** 当前产品候选为
> `1a9e680cb5a25ccc0c20786f4635625eef2d7ab0` / tree
> `010323120bc6eb07b5a07e7d3e6de1725968d314`；当前机械状态只看
> [`module-status.json`](../../module-status.json)。本轮由
> [`math capability replay`](math-capability-replay-v2.md) 的真实 `PRODUCT_TRIGGER` 启动。在完成
> `AUDITED` 并冻结有限 `FIX` 集前不得修改产品实现。

## Reopen intake（历史入口）

- component / owner / round：`math` / scalar numeric primitive owner / `math-v2-checked-arithmetic`；
- trigger：`MATH-REPLAY-CHECKED-ARITH-01`；
- evidence：CCCL 与 Folly 都提供 checked integral arithmetic，CCCL mdspan 与 llama.cpp/ggml 推理
  backend 又实际用它保护 extent、stride、element count 和 byte size；Math 当前没有公共 route；
- affected scope：same-type non-bool integral checked add/sub/mul，及其 tests、component/package journey、
  architecture/user/performance documentation；
- unaffected conclusions：现有 ceil/round 四个入口、`math -> PUBLIC core`、header-only delivery 与
  scalar-vs-SIMD 边界保持原结论；
- provisional baseline：`57ea0ba90c8472c938c4e3c6360ae4e0a6c69250`；
- tier / profiles / instrument：`P0`；`W1-WIN`、`W1-WSL`、`M1-MAC`；
  `closure-correctness-v1`；Formal `NONE`；
- immediate next action：冻结 closure card 后进入 `SCOPED`，再完成双向能力审计并冻结有限 FIX 集。

本状态只确认新事实击穿旧能力完整度；它不预先决定函数名、返回形状、compiler lowering 或最终
`FIX`。审计超出 checked add/sub/mul 时必须重新 scope，不得把 Math 扩成泛用算法集合。

## Closure card（`SCOPED`）

- component / owner / round ID：`math` / scalar numeric primitive owner /
  `math-v2-checked-arithmetic`；task kind：`REOPEN`；
- baseline commit：`57ea0ba90c8472c938c4e3c6360ae4e0a6c69250`。该提交的 math product 与
  `math-v1` candidate 相同；后续只把 trigger 与状态登记为文档；
- demand/defect：
  - `MATH-CHECKED-01`：真实 shape/stride/byte-count 必须在运算前发现 add/sub/mul overflow，当前
    Math 无公共路线；
  - `MATH-API-02`：接口必须与 math-v1 的 checked C++ 风格自然组合，同时避免要求可信算术都换成
    fail-fast wrapper；
  - `MATH-P0-03`：实现必须在支持工具链上使用最佳 overflow lowering，并为 `constexpr` 提供无 UB
    fallback，不产生动态资源或运行时路由；
- normal consumer workload：模型/tensor 元数据解析取得 integral extent、stride、offset、element size，
  以 checked add/sub/mul 逐步构造可表示的元素数与 byte range；成功后再分配、映射或 launch。普通
  已验证算术继续使用 `+/-/*`；
- north-star journey：量化推理引擎的 tensor layout、workspace sizing 和 backend boundary validation；
  当前 HYLE 产品消费者为 0，llama.cpp/ggml 只作已钉死的真实 domain evidence，不冒充仓内消费；
- scope：`include/hyle/math/scalar.h` 的 same-type non-bool integral checked add/sub/mul；对应 unit/
  property/constexpr/codegen tests；math package sample；architecture/user/performance 文档；中央状态与本 task；
- non-goals：saturating arithmetic、mixed input/result type、任意参数 fold/mul-add convenience、arbitrary
  precision、safe cast、负数 rounding、floating point、SIMD/model ops、下游 owner 迁移、C/Native FFI、
  stable C++ ABI、通用性能仪器与 timing SOTA；
- public/internal/higher-layer initial boundary：能力是 public header-only scalar value primitive；compiler
  适配可以是 header internal detail；不新增 binary、target、第三方运行依赖或 math-to-SIMD 边；
- direct dependencies / consumers：保持 `PUBLIC core`，不新增依赖；当前真实产品 consumer 为空，
  selected package sample 仅验证 C++ source/static/shared 用户旅程；
- performance tier：`P0`。目标为 O(1)，零 allocation/lock/syscall/常态 metadata/indirect dispatch；每次
  helper 只有一个 arithmetic operation 和 overflow predicate。未被 profile 证明是独立热点，不运行
  纳秒 benchmark，不声明 all-workload SOTA；
- budget：总调查与实现 120 分钟，其中 peer/API 审计 35 分钟、P0 codegen 25 分钟、验证 60 分钟；
  最多一个产品实现批次、一次 codegen 纠正与一次受影响重跑，不修改公共仪器；
- machine profiles / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC` /
  `closure-correctness-v1`；Formal `NONE`；
- pinned strong peers：CCCL `132670eba92ca5bbddb2d479e76045184ed885a5` 的 overflow numeric
  primitives 与 mdspan 消费；Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的
  `CheckedMath`。north-star 对照为 llama.cpp/ggml
  `7584430716ee229751771ed0d6bbcb780d105eeb`；
- correctness gates：signed/unsigned 8/16/32/64-bit，zero/one/min/max 与每个 overflow 方向；成功值精确、
  overflow 可发现且无 UB；`constexpr`；property/oracle；public-header self-contained；warnings-as-errors；
  WSL UBSan；纯值函数的 OOM/lifetime/cancellation/TSan 为语义不适用，不伪造 PASS；
- delivery gates：isolated selected `math` build；source、relocated static、relocated shared C++ package
  consumer；header-only target 不制造 binary/symbol journey；
- documentation/usability gates：architecture 解释模块所有权与 trusted/untrusted 选择；user guide 给出
  shape/byte-size 组合示例、overflow 与类型陷阱；performance evidence 只签复杂度/资源/code shape；
- expected output：双向 capability map、public trace、冻结 FIX、一个 product candidate、三平台与三交付
  证据、红队、六问、限制/reopen triggers、terminal receipt 和四项 LOC；
- stop condition：本轮全部 `MUST` 为 `COMPLETE/ROUTED`，有限 FIX 完成，全部适用 gate 有真实 verdict，
  无未命名 blocker则 `CLOSED`；否则在预算内以一个具名 blocker 停为 `BLOCKED`，不“继续研究”。

`REOPENED -> SCOPED` 的前态 registry 快照保存在 canonical checkout 外：
`C:\hyle-scratch\math-v2-checked-arithmetic\module-status-before-scoped.json`，SHA-256
`7418E3318981398F83CAD28BB34249BF108BA2F3E869F13E0F4D318073970AE0`。

## Capability-family map（`AUDITED`）

| Family | admitted journeys / source | required semantic variants and boundary | current route | coverage | action / negative space |
|---|---|---|---|---|---|
| checked integral add | tensor offset、range end、allocation header+payload；CCCL mdspan、Folly、仓内多个 owner | same-type non-bool signed/unsigned；成功为数学和，overflow 无 UB；`constexpr noexcept` | owner-local guard；Math 无入口 | `MISSING` | `FIX` as `try_add`；不加 variadic fold |
| checked integral subtract | range length、offset delta；CCCL；与完整基本算术族对称 | 同上；同时覆盖 signed positive/negative overflow 与 unsigned underflow | owner-local comparison；Math 无入口 | `MISSING` | `FIX` as `try_subtract`；不定义 saturating/clamped 结果 |
| checked integral multiply | element-count/stride/byte size；CCCL mdspan、Folly、ggml | 同上；zero、min×-1、符号四象限和 64-bit 边界 | owner-local `max / rhs` 或 backend helper；Math 无入口 | `MISSING` | `FIX` as `try_multiply`；不加 arbitrary arity/mul-add |
| non-negative ceil division / round-up | math-v1 shape/block journey | same-type non-bool、invalid/unrepresentable 可发现 | 现有四个入口 | `COMPLETE` | `KEEP`，产品实现不改 |
| standard bit/scalar math and safe range query | C++23 | 标准合同 | standard library | `ROUTED` | `KEEP` route；`bit_ceil/in_range` 等不重包 |
| saturation / mixed-result checked arithmetic | CCCL 有扩展 surface，但当前无具名 HYLE journey | result-type conversion、clamp policy、heterogeneous promotion 需要独立合同 | 无 | `OUT_OF_SCOPE` | `DEFER` only on real trigger；不能由 peer API 数量扩面 |
| floating / SIMD / model operators | domain-specific | accuracy、dtype、backend、dispatch | standard/domain/`hyle::simd`/consumer | `OUT_OF_SCOPE` | 不建立依赖或 umbrella API |

## Public-surface trace 与 API 裁决

| Entry / candidate shape | reverse demand | verdict |
|---|---|---|
| `try_ceil_div<T>(T,T)` | `MATH-INT-01`，非负 shape partition | `KEEP`；零改动 |
| `ceil_div<T>(T,T)` | 可信 invariant convenience | `KEEP`；零改动 |
| `try_round_up_to_multiple<T>(T,T)` | `MATH-ROUND-01`，可表示 padded extent | `KEEP`；零改动 |
| `round_up_to_multiple<T>(T,T)` | 可信 invariant convenience | `KEEP`；零改动 |
| `try_add<T>(T,T) -> optional<T>` | `MATH-CHECKED-01` | `FIX`；untrusted value composition |
| `try_subtract<T>(T,T) -> optional<T>` | `MATH-CHECKED-01` | `FIX`；名称不缩成含糊的 `sub` |
| `try_multiply<T>(T,T) -> optional<T>` | `MATH-CHECKED-01` | `FIX`；名称不缩成含糊的 `mul` |

选择 `try_ + optional<T>` 而不是照抄 peer 的 `bool + T*` 或公开 `overflow_result<T>`：它与 math-v1
现有错误可发现路线一致，没有 out-parameter/null contract，能够直接 `if (!value)` 或链式赋值；优化后
仍是 value+success flag。CCCL 的 mixed-result struct 是 GPU/heterogeneous surface 的合理形状，但会把
本轮 promotion 语义扩入公共 API。可信算术不新增 `add/subtract/multiply` fail-fast wrapper：语言 operator
最自然，也没有失败分支；只有来自模型、文件、网络或不可信尺寸的数据走 `try_`。

三个新函数保持同类型、非 `bool` integral constraint。`char` 等标准 integral 仍按自身 signedness 工作；
调用者必须先显式选择共同 result type，避免 implicit promotion 或窄化。成功结果精确，失败返回
`nullopt`，不提供 wrapped/clamped value。

## Capability / gap ledger（冻结）

| ID | kind / exact source | normal shape | owner / failure / resources | coverage | action / severity / verification |
|---|---|---|---|---|---|
| `MATH-CHECKED-ADD-01` | `MUST`；CCCL add_overflow + mdspan、Folly checked_add | `try_add(a,b)` | public math header；overflow→`nullopt`；O(1)，零动态资源 | `MISSING` | `FIX / high`；signed/unsigned exhaustive-small + boundary/property + codegen |
| `MATH-CHECKED-SUB-02` | `MUST`；CCCL sub_overflow；完整基础族 | `try_subtract(a,b)` | 同上，unsigned underflow 也失败 | `MISSING` | `FIX / high`；四个符号方向与 min/max/codegen |
| `MATH-CHECKED-MUL-03` | `MUST`；CCCL/Folly；ggml shape bytes | `try_multiply(a,b)` | 同上；zero 成功，min×-1 失败；runtime 用 toolchain overflow primitive/高半积，constexpr 用无 UB guard | `MISSING` | `FIX / high`；符号四象限、8..64 bit、UBSan/codegen |
| `MATH-API-04` | `MUST`；现有 HYLE checked convention | 三个 optional-returning binary helper | caller owns type selection；没有 binary ABI | `MISSING` | `FIX / high`；constraint compile tests、package sample、guide |
| `MATH-P0-05` | `MUST`；P0 structural contract | source caller observable bool+value | inline/constexpr；一个 arithmetic op + predicate；零 allocation/lock/syscall/indirect | `MISSING` for new family | `FIX / high`；MSVC/GCC/AppleClang optimized object inspection，不计时 |
| `MATH-DOC-06` | `MUST`；当前 docs 仍称 ceil/round 是唯一族 | trusted operator vs untrusted checked shape | architecture/user/perf owners | `PARTIAL` | `FIX / medium`；选择、组合、类型与 overflow 陷阱 |
| `MATH-DELIVERY-07` | `MUST`；C++ source/static/shared SDK | select math、include scalar header、link `hyle::math` | header-only INTERFACE；保持 `PUBLIC core` | `PARTIAL` for new journey | `FIX / medium`；sample + isolated/relocated three shapes |
| `MATH-ADOPT-08` | `GENERAL`；仓内 owner-local guards | 各模块未来按自身变更采用 helper | downstream owner | `OUT_OF_SCOPE` | `DEFER`；本轮不重开 error/platform/container/sched/io/compute |

覆盖状态与施工动作分离：三个新 `MUST` 当前仍是 `MISSING`，只有候选完成有限 FIX 后才能写
`COMPLETE`。没有 `PARTIAL + KEEP`。

## 实现策略冻结

- add/sub 使用 unsigned modulo arithmetic 与符号位 overflow predicate，成功时 `std::bit_cast` 回原类型；
  unsigned 分别检查 carry/borrow。这个表达式本身 `constexpr`、无 signed UB，并让支持工具链生成一次
  add/sub 与条件；
- multiply 在 GCC/Clang runtime/constexpr 可用路径使用 compiler overflow builtin；MSVC 64-bit runtime
  在 x64/ARM64 使用高半积 intrinsic，小类型使用无溢出的 wider intermediate；constant evaluation 和
  未知工具链使用先比较后乘的 portable guard；
- compiler-specific 内容只在 header `detail`，public signature/semantics 不暴露 compiler；不能新增
  runtime dispatch、第三方依赖、异常或常态 metadata；
- 若 MSVC/GCC/AppleClang 任一 optimized probe 出现除法型 runtime multiply fallback、out-of-line helper
  或重复 arithmetic，允许一次实现纠正；不得因此改变 public API 或建立 timing benchmark。

## 冻结 `FIX` 集

只授权一个产品批次：

1. 在 `scalar.h` 实现 `try_add/try_subtract/try_multiply` 及最小 header-internal overflow lowering；
2. 扩充 unit/constexpr/property/UBSan 与 test-only codegen probe，覆盖 signed/unsigned 8/16/32/64 bit；
3. 扩充 math package sample，验证正常结果与 overflow 失败；
4. 更新 math architecture、user guide/user index 与 P0 evidence，写清模块边界、接口选择、资源和陷阱。

禁止修改现有 ceil/round 合同、Core、其他产品模块、中央 component graph、公共性能仪器或构建开关。
任何额外 capability 必须退回 `SCOPED/AUDITED`，不得附加到本批次。

## 施工前六问直答

1. **第三方依赖：**产品只依赖 Core 与 C++23 standard library；CCCL/Folly/ggml 是 pinned evidence，
   不成为 product dependency。GoogleTest 仅测试。
2. **推理底座能力：**ceil/round 已完整；checked shape/stride/byte arithmetic 是唯一已证明缺口，冻结
   三个 FIX 后本轮范围完整。linear algebra/SIMD/model op 有明确 owner，不冒充 Math 缺口。
3. **性能：**现有四函数的 v1 P0 结论保持；新函数目标是最佳 O(1)/零资源/code shape，当前尚未验证，
   所以模块仍为 `PLANNED`，不提前写 SOTA/PASS。
4. **C ABI / FFI：**仓库中没有 Math 产品 C symbol、旧伪 C ABI 或导出；header-only template 不适合作
   无消费者 FFI。test-only `extern "C"` 仅稳定反汇编符号名，不是交付 API。
5. **构建/交付：**中央 `math -> PUBLIC core` 与 target/link 一致，INTERFACE component 形状正确；新 API
   的 source/static/shared installed journey 尚待候选验证，不伪造 binary artifact。
6. **C++ 易用性/文档：**现有 ceil/round 自然；新 optional API 已冻结为同一风格。architecture/user/perf
   仍过时，属于有限 FIX，候选前必须校准。

`SCOPED -> AUDITED` 的前态 registry 快照位于
`C:\hyle-scratch\math-v2-checked-arithmetic\module-status-before-audited.json`，SHA-256
`9A7BD73D6F022BE277A089366ECFF5AA254F203F82F7EE6A91703197F7625593`。

## Candidate

- product commit / tree：`1a9e680cb5a25ccc0c20786f4635625eef2d7ab0` /
  `010323120bc6eb07b5a07e7d3e6de1725968d314`；
- public contract：新增同类型非 `bool` integral 的 `try_add`、`try_subtract`、`try_multiply`，均为
  `constexpr noexcept -> optional<T>`；成功为精确值，overflow/underflow 为 `nullopt`，无 wrapped/
  saturated result。现有 ceil/round 四入口零改动；
- implementation：add/sub 在 unsigned bit domain 判定 carry/sign overflow；GCC/Clang runtime 使用
  overflow builtin，MSVC x64/ARM64 64-bit multiply 使用 signed/unsigned 高半积，小类型使用 wider
  intermediate，constant evaluation/unknown toolchain 使用无 UB guard；
- algorithm/resources：每项 O(1)，零 allocation/lock/syscall/常态 metadata/exception/runtime dispatch；
  public header 仍只依赖 Core/std，不引入 CCCL/Folly/ggml；
- tests：8-bit signed/unsigned operand 全空间 oracle，8/16/32/64-bit min/max/zero/sign 边界，constexpr、
  constraint、public-header；test-only probe 新增六个 observable 64-bit entry；package sample 覆盖组合与
  overflow；
- W1-WIN candidate smoke：MSVC 19.50 Release/Werror，math unit/header 2/2；selected source、relocated
  shared、relocated static consumer 各 1/1；optimized probe 的 add/sub 各一次目标运算，multiply 各一次
  `mul/imul`，零 call/divide/indirect dispatch；
- pending：W1-WSL GCC 14 + UBSan、M1-MAC AppleClang、两平台 code shape、最终 docs/red-team/aggregate
  gates。未运行项不写 `PASS`。

`AUDITED -> CANDIDATE` 的前态 registry 快照位于
`C:\hyle-scratch\math-v2-checked-arithmetic\module-status-before-candidate.json`，SHA-256
`9A06A2F848BE35DC5D8C5BC3025594FE61E46F29EA0C2FD5ADAF90FEBE19A95E`。候选后公开合同、算法、
layout 或正常 workload 若变化，必须退回 `AUDITED`；只补验证与事实记录不改变候选 identity。
