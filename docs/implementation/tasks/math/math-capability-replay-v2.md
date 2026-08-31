# Math 能力完整性回放 — 2026-08

<!-- CAPABILITY REPLAY RECORD: COMPLETE -->

> 本页把当前能力完整性标准回放到已经 `CLOSED` 的 `math-v1`。回放本身不修改产品 API、算法、
> layout、依赖或交付；它只判断旧结论能否继续成立。机械状态只看
> [`module-status.json`](../../module-status.json)。

## 回放合同

- component / replay：`math` / `math-capability-replay-v2`；
- reviewed state / round：`CLOSED` / `math-v1`；
- reviewed closure task：[`math-v1 closure`](math-v1-closure.md)；
- candidate / tree / end / registry anchor：
  `753ad5f304f52727c3ba925c9aac0a30f941502e` /
  `93f7bf1a996d6beabebe2b42801b086c483a18e6` /
  `8b7c55e6661994be08661eb13022fde9627cb6bb` /
  `8b7c55e6661994be08661eb13022fde9627cb6bb`；
- replay base：`fb8dd115982aa60e6eef20d67e97d030174ce778`；evidence commit：
  `57ea0ba90c8472c938c4e3c6360ae4e0a6c69250`；reviewer：Codex；日期：2026-08-31；
- normal consumer：C++23 用户从不可信 tensor shape、stride、element count 和 element size 计算可表示的
  offset、元素总数与 byte size；可信且已验证的普通算术仍直接使用语言运算符；
- north-star journey：量化推理引擎解析模型或建立 tensor layout 时，在分配、映射或 kernel launch 前
  拒绝溢出的 shape/stride/byte-product；
- tier / instrument：`P0 / closure-correctness-v1`。该族只裁决 O(1)、零动态资源、无 UB 与 release
  code shape，不运行纳秒 benchmark；
- 调查预算：45 分钟；每个能力族最多两个 pinned strong peers；
- strong peers：CCCL `132670eba92ca5bbddb2d479e76045184ed885a5` 的
  `add_overflow/sub_overflow/mul_overflow` 与 mdspan 消费；Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的 `CheckedMath`。north-star 对照使用
  llama.cpp/ggml `7584430716ee229751771ed0d6bbcb780d105eeb` 的 OpenVINO/Hexagon size arithmetic；
- 范围：scalar integral arithmetic 的能力族、当前公开 surface、复杂度/资源、易用性和三种 C++
  交付；
- 非目标：在回放中写产品代码，或扩到 saturating arithmetic、mixed-type conversion、负数 rounding、
  arbitrary precision、浮点、SIMD、模型算子、Native FFI/C API 与 timing SOTA。

候选到 replay base 的 `include/hyle/math/**` 产品路径没有公开 API、算法、依赖或 layout 漂移；期间只发生
SIMD 与构建交付演进。当前 package/source/static/shared 回放均能消费 math-v1 的四个现有函数，因此本次
finding 不是旧候选失效或交付损坏。

## 自顶向下 capability-family map

| Family | admitted journey / exact evidence | required semantics | current route | coverage / action | missing edge / negative space |
|---|---|---|---|---|---|
| non-negative ceil division | shape/block partition；math-v1、CCCL、ggml | same-type non-bool integral；invalid input 可发现；O(1) | `try_ceil_div` + `ceil_div` | `COMPLETE / KEEP` | mixed type 和负数 rounding 仍由 caller 处理 |
| representable round-up | padded extent/block sizing；math-v1、CCCL | same-type non-bool integral；invalid/unrepresentable 可发现；O(1) | `try_round_up_to_multiple` + fail-fast route | `COMPLETE / KEEP` | power-of-two address alignment 仍归 Mem |
| checked integral add/sub/mul | tensor offset、shape product、allocation byte count；CCCL mdspan、Folly、ggml | signed/unsigned same-type non-bool；成功返回精确值，overflow 不执行 UB；`constexpr noexcept`；O(1) | 各 owner 私有手写 guard，Math 无公共 route | **`MISSING / PRODUCT_TRIGGER`** | saturating、mixed result type 与 arbitrary precision 不自动准入 |
| standard scalar numeric facilities | C++23 | 标准规定的 ordinary scalar contract | `<bit>/<cmath>/<numbers>/<limits>` | `ROUTED / KEEP` | 不在 HYLE 重包标准库 |
| lane/vector/buffer math | SIMD consumer | backend/accuracy/dispatch contract | `hyle::simd` | `OUT_OF_SCOPE` | 不建立 math-to-simd 依赖 |

checked add/sub/mul 同时满足三项准入：两个成熟基础库提供稳定 primitive；CCCL 的 mdspan 直接用它们保护
extent/stride/index，ggml 又在真实推理 backend 中重复私有实现；能力能够保持 value-only、零状态和 O(1)，
不会给无关用户增加 metadata、依赖或运行时路由。它因此是 Math 边界内的 `MUST`，不是为了某个 benchmark
增加的表面。

## 自底向上 public-surface 与仓内重复实现

| Public entry | mapped family | verdict |
|---|---|---|
| `try_ceil_div` / `ceil_div` | non-negative ceil division | `KEEP`；现有语义与复杂度完整 |
| `try_round_up_to_multiple` / `round_up_to_multiple` | representable round-up | `KEEP`；现有语义与复杂度完整 |
| checked add/sub/mul | 当前不存在 | `MISSING / PRODUCT_TRIGGER` |

仓库扫描同时发现 `error::Status`、`platform` 映射/文件大小、`container` capacity/slab、`sched` permit、
`io` mapping/ring 与半成品 `compute` 各自存在 `max - x`、`max / x` 或 builtin overflow guard。这些调用点
证明问题普遍，但不授权本轮把所有已关闭或尚未审查的模块一并迁移；是否采用公共 helper 由各 owner 的
后续真实修改决定。Math 只先提供无依赖的正确 primitive。

## 复杂度、资源、接口与交付裁决

- add/sub/mul 的最佳复杂度档都是 O(1)，零 allocation、零 lock、零 syscall、零常态 metadata；GCC/
  Clang 的 overflow builtin 与 MSVC/portable fallback 都必须生成单个普通 arithmetic operation 加条件
  判定，而不是通用大整数或异常路径。
- 对不可信 shape/count，`std::optional<T>` 与 math-v1 既有 `try_` 约定能直接组合；对已经验证的
  普通算术，语言 `+/-/*` 是自然入口，不应再造 fail-fast 包装来替换每次 operator。
- Math 保持 header-only `INTERFACE` C++ component；不存在应被导出的 binary symbol，也不制造 C API、
  Native FFI 或稳定 C++ ABI。source/static/shared 用户都通过同一 public header 取得能力。
- 当前架构与用户文档把 ceil/round 写成 Math 唯一能力族，因而在产品修复后必须同步改为“checked scalar
  shape arithmetic + non-negative rounding”，并写清 trusted operator 与 untrusted `try_` 的选择。

## 回放裁决

- product verdict：**`PRODUCT_TRIGGER`**；trigger ID：`MATH-REPLAY-CHECKED-ARITH-01`；
- trigger evidence：两个 pinned strong peers 都提供 checked add/mul（CCCL 还提供 sub），CCCL mdspan 与
  llama.cpp/ggml 真实 tensor/backend 代码都用该能力防止 shape/stride/byte-size overflow；HYLE Math 没有
  公共 route，且多个 owner 重复手写不完整子集。这击中 math-v1 的 missing-capability reopen trigger；
- 受影响范围：只重开 scalar checked add/sub/mul、相邻 tests/package/docs 和 P0 code shape；ceil/round
  结论保持，SIMD 与其他 owner 不重开；
- 下一轮先冻结 API 和 signed/unsigned semantics，再实现一个有限 batch。若审计不能证明跨 MSVC/GCC/
  AppleClang 的无 UB constexpr 形状，则进入具名 blocker，不扩大性能仪器；
- Formal 仍为 `NONE`，不产生 timing SOTA 声明。
