# math-v2 checked arithmetic 收口记录

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> **终态：`CLOSED`。** 当前产品候选为
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

## Verification manifest

验证绑定 product candidate `1a9e680cb5a25ccc0c20786f4635625eef2d7ab0` / tree
`010323120bc6eb07b5a07e7d3e6de1725968d314`。candidate 后 test-only commits
`a8f5844e08c19d758f76482850181a21f54b4d12`、
`cf93514e88057cb8d05e7ab3337a33a16027b1be` 只为 GCC/AppleClang `-Werror` 校准 oracle fixture；从
candidate 到验证提交的 `include/hyle/math`、package sample 与当前 architecture/user 产品合同均零 diff。

### Correctness risk matrix

| Risk | evidence | verdict / limitation |
|---|---|---|
| normal / zero / min / max | 三平台 math unit；8/16/32/64 signed/unsigned boundary；zero 与符号四象限 | `PASS` |
| overflow / underflow / invalid intermediate | 8-bit signed/unsigned operand 全空间精确 oracle；max+1、min-1、unsigned 0-1、max×2、min×-1 | `PASS`；不返回 wrapped/clamped value |
| constexpr / constraints | valid/overflow compile-time assertions；bool 与 mixed-type 不可调用 | `PASS` |
| property / differential | 131,072 个 8-bit operand pair × add/sub/multiply 与 wide exact oracle 对拍 | `PASS` |
| UB / sanitizer | W1-WSL GCC 14.2 RelWithDebInfo UBSan math/header 2/2 | `PASS`；signed operation 只在先证明可表示后执行 |
| allocation / OOM | 纯 inline value function，无 allocation/owner | `NOT_APPLICABLE`；结构扫描确认零动态资源 |
| lifetime / move / cancellation | 无对象、borrow、挂起工作或 teardown | `NOT_APPLICABLE` |
| concurrency / reentrancy | 无共享或 thread-local state | `PASS`；任意线程独立调用 |
| ASan / TSan | 无 memory ownership、共享状态或同步协议 | `NOT_APPLICABLE`；不把未运行写成 PASS |
| compiler / ISA | MSVC 19.50 x64、GCC 14.2 x86-64、AppleClang 21 arm64 真编译运行 | `PASS` |
| public header | 三平台 `hyle_math_header_self_contained_test` 各 1/1 | `PASS` |
| dependency / third party | product include/link scan；selected closure 仅 `core,math` | `PASS`；CCCL/Folly/ggml 只作 evidence |
| source/static/shared delivery | W1-WIN、W1-WSL、M1-MAC 的 selected source、relocated shared、relocated static consumer 各 1/1 | `PASS`；INTERFACE target 不虚构 archive |
| P0 code shape | 三工具链 object/hash/反汇编见 performance evidence | `PASS`；不产生 timing SOTA 声明 |

### Aggregate gates

| Gate | verdict | evidence / limitation |
|---|---|---|
| scope | `PASS` | 冻结四项 FIX 完成；ceil/round、其他模块、build/instrument 零产品改动 |
| architecture | `PASS` | checked scalar shape arithmetic 归 Math；SIMD/model/domain/standard routes 不变；`PUBLIC core` 真实 |
| correctness | `PASS` | 三平台 unit/header，WSL UBSan，boundary/constexpr/全空间 oracle |
| delivery | `PASS` | 三平台三种 C++ journey 共 9/9；header-only 无 binary/Native FFI 假入口 |
| usability / docs | `PASS` | architecture 与 guide 写清 trusted operator、untrusted `try_`、组合、same-type、unsigned subtract 和 alignment 陷阱 |
| red team | `PASS` | 见下节；无未处理 high finding |
| engineering performance | `PASS` | P0 O(1)/零资源；三工具链原生 overflow lowering；不运行纳秒 benchmark |

### Reproduction summary

- W1-WIN：MSVC 19.50.35728，Ninja Release/Werror，math/header 2/2；selected source/shared/static
  各 1/1；scratch `C:\hyle-scratch\math-v2-checked-arithmetic\{win-build,win-delivery}`；
- W1-WSL：GCC 14.2 native ext4，Release 与 UBSan 各 2/2；source/shared/static 各 1/1；scratch
  `/root/hyle-scratch/math-v2-{1a9e680c,build-release-1a9e680c,build-ubsan-1a9e680c,delivery-cf93514e}`；
- M1-MAC：AppleClang 21 arm64 / Unix Makefiles / deployment target 14.4，Release 2/2；source/shared/
  static 各 1/1；scratch `/Users/kiro/hyle-scratch/math-v2-{a8f5844e,build-a8f5844e,delivery-cf93514e}`；
- exact code shape、object hashes 和 ISA 解释见
  [`benchmarks-math-v2`](../../evidence/performance/benchmarks-math-v2.md)。这些是本地工程 artifact，
  不冒充长期签名 Formal bundle。

## 独立红队

| Attack | severity | verdict / evidence |
|---|---|---|
| signed add/sub/mul 先执行 UB 再检查 | high | `disproved`：add/sub 在 unsigned domain；mul builtin/intrinsic 或先判界；UBSan + min/max oracle 通过 |
| `min * -1`、negative×negative 或 unsigned borrow 漏判 | high | `disproved`：具名 boundary + exhaustive int8 三平台通过 |
| optional API 隐藏 allocation/exception/ABI seam | high | `disproved`：inline value-only；反汇编无 call；组件无 math binary symbol，三交付同一头 |
| MSVC fallback 用 divide 进入正常 runtime multiply | high | `disproved`：x64 `mul/imul`；ARM64 路径使用 `__umulh/__mulh`；portable divide 只属 constant-evaluation/unknown compiler fallback |
| AArch64 两条 multiply 被误判为 wrapper regression | medium | `disproved`：ISA 分离 low/high；AppleClang builtin 直接形成 `mul+umulh/smulh`，无 HYLE seam |
| mixed signed/unsigned 自动 promotion 产生意外结果 | high | `fixed by contract`：same-type deduction，compile-negative test，guide 要求显式选择 result type |
| peer API union 扩出 saturation、variadic 或 mixed result | medium | `rejected`：无当前 admitted journey；保持 standard/SIMD/domain route |
| test-only `extern "C"` 被当成公开 Native FFI | medium | `disproved`：probe object 不安装，package 只有 header/INTERFACE target，文档明确无 C ABI |
| 为减少仓内手写 guard 越界迁移已关闭/未审模块 | medium | `rejected`：candidate diff 只触及 Math 产品/fixture/docs，下游 adoption 由 owner 触发 |
| 文档仍称 ceil/round 是唯一能力族 | medium | `fixed`：architecture、user guide/index 与当前 API 对账，docs checker 通过 |

红队未发现未处理 HIGH、未声明资源、错误复杂度或需退回 `AUDITED` 的产品事实。

## 验证后六问

1. **直接/间接第三方：**Math 产品只有 Core/std；没有 vendored/runtime third party。GoogleTest 仅测试，
   CCCL/Folly/ggml 仅为 pinned 设计证据。
2. **推理底座功能：**scalar shape/stride/offset/byte sizing 现在有 checked add/sub/mul 与 ceil/round；本
   边界全部 `MUST` 为 `COMPLETE`。SIMD、linear algebra、model ops、mixed/saturation 都有明确 owner/
   trigger，不是半实现。
3. **性能：**复杂度和资源达到最佳档；三工具链正常 runtime 采用原生 overflow lowering，无 call/
   divide/dispatch。这里只签 P0 structural，不伪称端到端 inference 或纳秒 SOTA。
4. **C ABI 清理：**产品没有 Math C ABI、exported symbol 或旧伪 C surface；test-only C linkage 不交付。
5. **build/package：**中央 component graph 与真实 `hyle::math -> PUBLIC core` 一致；三平台 source/
   static/shared 用户均通过；header-only 不制造 archive。
6. **C++ 易用性/文档：**可信算术用 operator，不可信值用同风格 optional `try_`；same-type、链式检查、
   underflow、alignment/rounding 等陷阱已写入当前 guide，architecture/performance 也已对账。

`CANDIDATE -> VERIFIED` 的前态 registry 快照位于
`C:\hyle-scratch\math-v2-checked-arithmetic\module-status-before-verified.json`，SHA-256
`D98E053468D6318B73AE8699A03595E57D36E1E9B42BCC2102CDC37708563FB1`。全部适用 aggregate gate
有真实 `PASS`，Formal 为 `NONE`；下一步只允许 terminal receipt、LOC 与 registry anchor，不再改产品。

## Closure decision 与 terminal receipt

本轮 terminal verdict 为 **`CLOSED`**：三个新增 `MUST` 已从 `MISSING` 变为 `COMPLETE`，旧 ceil/round
仍为 `COMPLETE`，standard/SIMD/domain 路线保持完整；scope、architecture、correctness、delivery、
usability/docs、red-team、engineering-performance 七门全部 `PASS`，没有 `PARTIAL + KEEP`、
`INCONCLUSIVE`、未命名 blocker 或未执行的适用门。Formal 为 `NONE`。

已知限制：

- 只接受同类型、非 `bool` native integral；调用者负责先验证/转换 mixed-width 与 signed/unsigned
  数据，不提供 mixed-result promotion；
- 失败只返回 `nullopt`，不提供 wrapped、saturated、clamped、variadic product 或 mul-add；binary helper
  可组合但每个中间结果都应检查；
- Math 是 header-only C++ component，没有 stable C++ ABI、C/Native FFI 或独立 binary artifact；
- P0 只签 O(1)、资源与三工具链 code shape；没有端到端推理 profile、纳秒胜负或 Formal SOTA 证书；
- 当前仓内没有 HYLE product syntax consumer；package fixture 和 llama.cpp/ggml north-star 证明交付与需求，
  但不冒充已经完成的 HYLE inference engine；
- 未知/非冻结编译器的 portable multiply constant-evaluation fallback 可以使用 division；三个 shipping
  工具链的正常 runtime 都已证明原生 lowering。

重新打开只允许以下新事实：

1. 真实 consumer 暴露 checked scalar arithmetic 的 correctness、security、易用性或缺失能力；
2. public API、overflow 算法、Core 依赖、compiler/ISA semantics 或 source/static/shared delivery 变化；
3. 支持工具链的正常 runtime 出现 divide、out-of-line helper、动态资源或 material consumer regression；
4. 新标准设施可完整取代本能力，或成熟同合同 peer 证明更好复杂度档/material consumer advantage；
5. 真实 mixed-result、saturation 或 bulk-product journey 被正式准入。单纯 peer API 数、benchmark 点值或
   下游 owner 选择采用现有 helper 不重开 Math。

候选的 canonical LOC self-check（snapshot
`1a9e680cb5a25ccc0c20786f4635625eef2d7ab0`）：全库代码 **403,849**；自实现包含测试 **96,313**；
自实现不包含测试 **60,061**；当前 Math 实现 **222** 行。命令为
`python scripts/loc-report.py --module math --commit 1a9e680c... --self-check --json`，结果
`self_check=pass`；LOC 只记录规模，不证明完成度。

Terminal receipt：

- product candidate / tree：`1a9e680cb5a25ccc0c20786f4635625eef2d7ab0` /
  `010323120bc6eb07b5a07e7d3e6de1725968d314`；
- final verification evidence commit：`8e2777d3bca8784f63e197483f919b16b2f0d504`；
- raw engineering artifacts：W1-WIN、W1-WSL、M1-MAC 路径见 reproduction summary；
- end receipt commit：`0ab83f780087802699bae9cb2a330c74b832922a`，这是首次保存完整 terminal
  decision 的提交；
- registry anchor commit：`a1c771d7674bb5c7313ee27781ce0f09da48487e`，这是首个实际保存本
  `CLOSED` state/candidate、全部 terminal 字段与人读状态投影的中央 registry 快照；本次紧随的
  evidence-only 提交只校准该非自引用 identity，产品、状态和 verdict 均未变化；
- 后续只按本节 reopen triggers 重开；math-v1 与本轮 replay/audit 只作历史事实，不再授权施工。

`VERIFIED -> CLOSED` 的前态 registry 快照位于
`C:\hyle-scratch\math-v2-checked-arithmetic\module-status-before-closed.json`，SHA-256
`2251E6377CE8544958F9FCAB3A085DC582660EA63C591FCC31FFA1736BB26F99`。

Anchor 校准前态 snapshot：
`C:\hyle-scratch\math-v2-checked-arithmetic\module-status-before-anchor.json`，SHA-256
`59AEA5A0A1DB46CFC7AEC417097319431DB2F2902FB248CB078F765EA0EC9AA2`。
