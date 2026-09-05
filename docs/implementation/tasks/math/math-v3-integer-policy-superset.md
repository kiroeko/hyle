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
