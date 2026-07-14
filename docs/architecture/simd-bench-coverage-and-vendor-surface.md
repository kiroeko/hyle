# SIMD benchmark 覆盖率 & vendor 借用面（现状盘点）

> 本文回答两个问题，作为后续处理的依据（快照日期 2026-07-14）：
> 1. **SIMD op benchmark 覆盖率达到多少？**
> 2. **有哪些地方我们直接调用 Highway / xsimd（而非 HYLE 自造后端）？**
>
> 数字随代码演进，引用处均给出权威来源文件，核对时以当前代码/文档为准。

---

## 1. op benchmark 覆盖率（机械口径，见 `scripts/simd-op-coverage.py`）

**权威留档**：数字由 `scripts/simd-op-coverage.py` 机械产出（可复现 + 自校验 + `--check` 可进 CI）；
verdict 权威见 [`../milestones/simd-orderswap-matrix.md`](../milestones/simd-orderswap-matrix.md)。

> **⚠️ 为什么旧"96%（245/255）"不可信**：手工/一次性 grep 统计反复出错（已测的 `Lt128/IsInf`
> 被误报"没测"，虚增缺口；分母还混入 `detail::` helper 且被人为缩小）。**根因是没有自校验**。
> 新脚本的闭环：**已测集必须 ⊆ 提取的全集**；任何已测 op 不在全集里 = 提取规则有盲区 → 报错重跑。
> 开发时这个自校验抓出 10 个提取盲区（复合标题 `Load/Store`、`Mask And` 前缀、自定义 helper
> `run_reduce/cmp128`、跨行字符串合并、大写宏泄漏），逐个修正后数字才可信。

### 机械口径三档（快照 2026-07-14）

| 指标 | 数值 | 口径 |
|---|---|---|
| 公开 op 名（去重 8 头，剔除 `detail::`/大写宏/小写 helper） | **380** | `extract_public_ops` |
| 不可测（constexpr 谓词/常量/introspection/单 lane 存取/tag helper，逐个带理由） | 26 | `NOT_BENCHABLE` |
| 内部选择器变体（`*Native`/`*Emulated`，由伞 op 覆盖） | 10 | `SELECTOR_SUBSTR` |
| 别名（`Rol`→RotateLeft 等，目标已测才算覆盖） | 4 | `ALIASES` |
| **可测全集** | **340** | benchable |
| 已有 bench（op_parity + crypto + gather） | 303 | benched |

### ⭐ SOTA 状态三分（"有 bench" ≠ "已证 SOTA"）

打平/反超 Highway（SOTA）的判据是**校准仪器出了 measured verdict 并落进矩阵**，不是"有 bench"。据此：

| 档 | 数量 | 下一步动作 |
|---|---|---|
| **已证 SOTA**（矩阵有 measured verdict 行，TIE/WIN） | **269** | 已闭环，防回归监控 |
| **有 bench 未测**（bench 在、还没跑校准仪器落矩阵） | **30** | 跑双序校准仪器 → 回填矩阵 |
| **无 bench**（连性能测试都没有） | **41** | 先按 methodology 加 bench，再测 |
| **合计"未证 SOTA"** | **71**（30 待测 + 41 待加测） | 这 71 是"要证明 ≥ Highway"的工作队列 |

- **30 有 bench 未测** = 本会话新增 14（MulEvenAdd/MulOddAdd/U8FromU32/MaskedMulAdd 族/位三元/掩码
  gather-scatter/CopySignToAbs/IfVecThenElse）+ 16 既存未回填矩阵（AES×6/CLMul×2/ReduceAnd/Mul/Or/
  ReciprocalNewton/SqrtNewton/Store/StoreU/StoreN）。
- **41 无 bench** 分三层:①结构性原语（VecFromMask/MaskFromVec/SignBit/BitsFromMask——CMP kernel 里
  当 sink/输入被间接跑，但无独立 parity 行）②InRange 快变体（PromoteInRange*/DemoteInRangeTo/
  ConvertInRangeTo/DemoteToNearestIntInRange）③真缺口（TableLookupBytesInRange/TableLookupBytesOr0/
  MaskedSet/MaskedSetOr/InterleaveEvenBlocks/OddBlocks/UpperBlocks/SlideUpBlocks/SlideDownBlocks/
  Lt128Vec/SumsOfShuffledQuadAbsDiff/OrderedTruncate2To/TruncateStore/Shuffle01/RearrangeToOddPlusEven/
  SafeCopyN/SafeFillN/Stream/CompressBits/CompressBitsStore/CompressBlocksNot/MaskedIsNaN/
  IfNegativeThenNegOrUndefIfZero/LoadMaskBits/StoreMaskBits/MaskFalse/MaskFromNormalizedVec/
  Lower·UpperHalfOfMask）。
- **完整实时清单跑 `python scripts/simd-op-coverage.py`**（本表是快照，脚本是真相）。

### op-parity 的三/四个文件

`bench/simd/op_parity_*.cc`，用校准仪器（`op_parity_engine.h`：pin+replicate+stagger+锚点闸门+双序换向）：

| 文件 | 覆盖面 |
|---|---|
| `op_parity_full_bench.cc` | 主 op 面（UNARY/BINARY/TERNARY/CMP 宏 + 具名 title） |
| `op_parity_bench.cc` | mask 域、gather/scatter-N、tail load/store、bit ops 等 |
| `op_parity_widen_bench.cc` | 变宽 promote/demote/madd 族 |
| `crypto_bench.cc` | AES 轮 + CLMul |

双平台真编真跑：WSL GCC14 AVX2 + Win MSVC /arch:AVX2。

### 旧"有据跳过 10 个"复审（4 个站不住）

| op | 旧理由 | 复审判定 |
|---|---|---|
| `DivIsPow2` | constexpr bool | ✅ 真不可测（标量谓词），有效 |
| `CompressBlocksNot` | block 变体 | ✅ 是 `CompressNot` 的 u64 别名，本体已测（op_parity_bench），有效 |
| `Bf16EvenToF32` / `Bf16OddToF32` | bf16 域 | ⚠️ 是 `detail::` 内部 helper——**不应进公开分母**（分母污染），非"跳过" |
| `MulEvenAdd` / `MulOddAdd` | "由 dtype_bench 覆盖" | ❌ **假称**：dtype_bench 只测整-buffer 转换批，从不调它们；这是 bf16 点积 FMA，Highway 有同名 op，**零性能覆盖** → 待补 |
| `U8FromU32` | "由 dtype_bench 覆盖" | ❌ **假称**：同上，dtype_bench 不碰它 → 待补 |
| `IfNegativeThenNegOrUndefIfZero` | Undef 语义难对拍 | ❌ 借口：吞吐可测（Undef 只在零 lane），Highway 有同名 op → 待补 |
| `CompressBits` / `CompressBitsStore` | 需特殊 harness | ⚠️ 半真：需预打包 byte-mask buffer，但 Compress/CompressStore 本体已测，只是 bitfield 前导额外——harness 成本待评估 |

### 分母漏项（~15 个真 op，有 Highway 同名/等价，却不在任何 op_parity 也不在上面 10-skip 列）

按藏真损失风险排序（`vec.h` 行号）：

| 优先 | op | 位置 | Highway 等价 | 为何要测 |
|---|---|---|---|---|
| P0 | `MaskedMulAdd`/`MaskedNegMulAdd`/`MaskedMulAddOr` | vec.h:477/483/480 | 同名 | AI 热点 masked FMA；HYLE 组合 IfThenElseZero+MulAdd，**avx512 后端无原生 maskz-fmadd**，Highway 用单条 `_mm512_maskz_fmadd`——avx512 真差距候选（AVX2 两边都组合应 TIE） |
| P0 | `MulEvenAdd`/`MulOddAdd` | vec.h:1010/1015 | 同名 | bf16 点积 FMA，多指令(PromoteEven/Odd+ShiftLeft<16>+MulAdd)；旧文档假称已覆盖 |
| P1 | `XorAndNot`/`AndXor`/`BitwiseIfThenElse` | vec.h:644/645/648 | 同名/generic | AVX-512 可融单条 `vpternlog`，HYLE 组合两条（同 Xor3/Or3 已知"512 上可能真差"类） |
| P1 | `MaskedGatherIndex`/`MaskedGatherIndexOr`/`MaskedScatterIndex` | vec.h:873/854/877 | 同名 | HYLE 标量回退 vs Highway 原生 masked gather，分歧大 |
| P1 | `U8FromU32` | avx2.h:1363 | 同名 | 跨步 pack，非平凡；旧文档假称已覆盖 |
| P2 | `CopySignToAbs`/`IfVecThenElse` | vec.h:681/1483 | 同名/generic | 便宜，低风险 |
| P2 | `OrderedTruncate2To`/`TruncateStore`/`PromoteInRange{Lower,Upper,Even,Odd}To` | vec.h:1771/1782/1794-1806 | 同名 | 变宽 InRange 族，低风险 |

（变宽 MAC 家族 ReorderWidenMulAccumulate/WidenMulAccumulate/SatWidenMul*/SumOfMulQuadAccumulate 已在 op_parity_widen 覆盖，无漏。）

### op-parity 面之外另有专项 bench（不计入"未覆盖"）

op-parity 只覆盖 `vec.h` 核心 op 面。以下能力面由独立 bench 兜底，**不应计入"未覆盖"**：

| bench | 覆盖能力 |
|---|---|
| `math_bench.cc` | 超越函数 exp/log/sin/cos/tan/tanh/erf/cbrt/hypot/asinh…（f32+f64），best-of vs Sleef/xsimd |
| `dtype_bench.cc` | f16/bf16/f8e4m3/f8e5m2 转换批 |
| `activation_bench.cc` | softmax/log_softmax/RNG |
| `sort_bench.cc` | vqsort/partial/top-k tier |
| `bitpack_bench.cc` | bit pack/unpack u8/u16/u32 |
| `crypto_bench.cc` | AES 轮 + CLMul（8 op） |
| `gather_bench.cc` | GatherIndex / Compress（原生 vs store+reload A/B） |
| `simd_bench.cc` | dot / saxpy / reduce kernel |

**结论**：vec.h 核心 op 面对拍覆盖 **96%**，10 个缺口全有据；dtype/crypto/math/sort/bitpack 家族由专项 bench 覆盖。

> 各 op 的 HYLE vs Highway/xsimd **输赢**不在本文范围——权威见 [`../milestones/simd-orderswap-matrix.md`](../milestones/simd-orderswap-matrix.md)。本文只回答"覆盖率多少"与"借用面在哪"。

---

## 2. 直接调用 Highway / xsimd 的地方

**关键前提**：`include/` 里**零** vendor 调用——所有头文件里的 `hwy`/`xsimd` 字样都是注释（契约对齐说明或性能标注）。公开 API 只吃裸指针，vendor include 全部关在 `src/simd/**` 内部 TU（+ 一个 container 文件），**vendor 类型永不泄漏到 include**。

实际借用共 **4 处专项 kernel**，全是薄借（vendor 做真活，HYLE 只写编排 + 尾部）+ 运行期/编译期门控：

### 2.1 超越函数 / math（最大借用，薄借，逐 op best-of）

- **dispatch**：`src/simd/math/math_dispatch.cc`；decls `src/simd/math/kernels.h`
- **借核 TU**：
  - x86：`transcendental_xsimd.cc` / `transcendental_highway.cc` / `transcendental_highway_avx512.cc` / `transcendental_xsimd_avx512.cc`
  - arm64：`transcendental_highway_neon.cc` / `transcendental_xsimd_neon.cc`
  - 复数 SoA：`complex_hyle-inl.h`（cexp/clog/csqrt 借 xsimd AVX2 核；csin/ccos/ctanh/cpow 是 HYLE 编排 xsimd 原语）
  - scalar 底：`transcendental_scalar.cc`
- **借什么**：Highway contrib/math 的 `hn::Exp/Log/Tanh/…` 和 `xsimd::exp/tanh/…` 做真向量化；HYLE 只写 batch 循环 + libm 尾。
- **winner（逐 op 逐精度 bench 定）**：f32 多为 xsimd（exp/tanh/exp2/erf/tgamma + xsimd 独有 erfc/exp10/fmod/fdim）；Highway 赢 log/sin/cos/log2/atan/cbrt/expm1/log1p/asinh/acosh/… 及大部分。**f64 几乎全 Highway**（xsimd 只赢 erf + tgamma）。`lgamma`/`remainder`/`nextafter` 留 scalar（vendor bug）。
- **门控**：运行期 `cpu_features().avx2/avx512`；NEON arm64 无条件。AVX-512 Highway 在 MSVC 关闭（`!defined(_MSC_VER)`，HWY_BROKEN AVX3）。

### 2.2 vqsort（薄借，纯 Highway）

- **位置**：`src/simd/sort/vqsort_body.inc` + `vqsort_highway_{avx2,avx512,neon}.cc`；dispatch `sort_dispatch.cc`；scalar `vqsort_scalar.cc`
- **借什么**：`hn::VQSortStatic / VQPartialSortStatic / VQSelectStatic`（header-only，`VQSORT_ONLY_STATIC=1`）。sort/partial-sort/select 覆盖 u32/i32/u64/i64/f32/f64 + KV32/KV64 top-k（reinterpret 到 Highway K32V32/K64V64）。
- **门控**：AVX2/AVX512/NEON 运行期派发。

### 2.3 bitpack（多数薄借，一处 HYLE 自造绕 bug）

- **位置**：`src/simd/bitpack/bitpack_body.inc` + `bitpack_highway_{avx2,avx512,neon}.cc`；`bitpack_dispatch.cc`
- **借什么**：`hn::Pack8/16/32<kBits>` 做 block 打包；HYLE 写 scalar-bitstream 尾（`bitstream.h`）。
- **窄例外**：`bitpack_hyle_u16-inl.h` 是 HYLE 自造 u16 11/13-bit packer（用 `hn::` 原语），绕 Highway 满位宽 round-trip bug。

### 2.4 UTF-8 计数（薄借，编译期可选）

- **位置**：`src/container/string_algo.cc`
- **借什么**：`simdutf::validate_utf8 + count_utf8`；`-DHYLE_SIMD_SIMDUTF` 门控，否则 scalar。simdutf 自派发 ISA。
- 注：未定义该宏时不 include simdutf，故常规构建的 vendor include 列表里不出现。

### 不借、全自造的部分

`vec.h`/`ops.h` + 五后端（avx2/sse4/neon/scalar/scalable）的 ~217 op 核心面、RNG（xoshiro256）、cmp128、swizzle、quantize、crypto（AES/CLMul）、fp8（`cvt_avx10_2.cc` 手写 `_mm512_*`，无 Highway/xsimd）—— 100% HYLE 自己的 intrinsic。无 intdiv/RNG 借用。

**一句话**：HYLE 自造整个核心 SIMD 面（~217 op）+ fp8；只对四个"难自己超越"的招牌 kernel（超越函数 / vqsort / bitpack / 可选 UTF-8）做薄借，全关在 `src/simd/**` 内部 TU，vendor 类型不入 `include/`。

---

## 后续处理备忘（TODO）

> 供后面决策，本文只盘点不改代码。

- **覆盖率 96% 是否收口**：剩 10 个未覆盖已全部有据；若要 100%，需为 CompressBits 类做特殊 byte-mask harness、为 Undef 语义 op 定对拍口径——收益 vs 工作量待评估。
- **vendor 借用面是否要收窄**：超越函数是最大借用面且逐 op best-of，短期难自造超越；vqsort/bitpack 是招牌算法件，薄借合理。若未来要"零 vendor"需重估各自 ROI。
- **借核对拍闸门**：借核同样要过 round-trip/oracle（已抓到 Highway Pack16 + xsimd lgamma 两个 vendor bug），维持"借不等于信"。
