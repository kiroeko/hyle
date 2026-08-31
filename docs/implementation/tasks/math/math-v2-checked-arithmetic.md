# math-v2 checked arithmetic 收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> **当前状态：`SCOPED`。** `math-v1` 仍是上一轮有效历史候选；当前机械状态只看
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
