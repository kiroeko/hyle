# SIMD benchmark 覆盖率 & vendor 借用面（现状盘点）

> 本文回答两个问题，作为后续处理的依据（快照日期 2026-07-14）：
> 1. **SIMD op benchmark 覆盖率达到多少？**
> 2. **有哪些地方我们直接调用 Highway / xsimd（而非 HYLE 自造后端）？**
>
> 数字随代码演进，引用处均给出权威来源文件，核对时以当前代码/文档为准。

---

## 1. op benchmark 覆盖率 = 96%（245/255 真算子）

**权威留档**：[`../performance/benchmarks-simd.md`](../performance/benchmarks-simd.md)（"op-parity 覆盖率"节）。

| 指标 | 数值 | 说明 |
|---|---|---|
| 公开 op 名（headline） | ~228 | 泛型模板一名多型，对齐 Highway ~370 typed 变体（[`simd-design.md`](./simd-design.md)） |
| 覆盖率分母（真算子集） | 255 | 公开 op 面剔除 tag/trait/构造 helper + alias/纯直通-单指令-block 变体后 |
| **op-parity 已覆盖** | **245 / 255 = 96%** | HYLE 自造 op vs Highway/xsimd 同 op 同 ISA 对拍 |
| op-parity 未覆盖 | 10 | 全部有据（见下），非遗漏 |

### op-parity 的三/四个文件

`bench/simd/op_parity_*.cc`，用校准仪器（`op_parity_engine.h`：pin+replicate+stagger+锚点闸门+双序换向）：

| 文件 | bench 行 | 覆盖面 |
|---|---|---|
| `op_parity_full_bench.cc` | ~190 | 主 op 面（UNARY/BINARY/TERNARY/CMP 宏 + 具名 title） |
| `op_parity_bench.cc` | ~61 | mask 域、gather/scatter-N、tail load/store、bit ops 等 |
| `op_parity_widen_bench.cc` | ~22 | 变宽 promote/demote/madd 族 |
| `crypto_bench.cc` | 8 | AES 轮 + CLMul |

双平台真编真跑：WSL GCC14 AVX2 + Win MSVC /arch:AVX2。

### 未被 op-parity 覆盖的 10 个（有据跳过）

| op | 原因 |
|---|---|
| `DivIsPow2` | constexpr bool，非向量 op |
| `IfNegativeThenNegOrUndefIfZero` | Undef 语义难对拍 |
| `CompressBits` / `CompressBitsStore` | 预打包 byte-mask 指针，需特殊 harness |
| `CompressBlocksNot` | block 变体 |
| `Bf16EvenToF32` / `Bf16OddToF32` / `MulEvenAdd` / `MulOddAdd` / `U8FromU32` | bf16/dtype 域，由 `dtype_bench` 覆盖 |

### 覆盖率不被低估：op-parity 面之外另有专项 bench

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
