# SIMD vendor 借用面（现状盘点）

> 本文只回答一个问题：**有哪些地方我们直接调用 Highway / xsimd / simdutf（而非 HYLE 自造后端）？**
>
> **覆盖率 / op 计数 / SOTA 判决不在本文** —— 唯一数据源是
> [`../milestones/simd-orderswap-matrix.md`](../milestones/simd-orderswap-matrix.md)（权威计数树 380→340→299→291、
> 全量表、独有测试表、41 待补 bench、语义缺口）。数字随代码演进，跑 `scripts/simd-op-coverage.py` 取实时值。

**关键前提**：`include/` 里**零** vendor 调用——所有头文件里的 `hwy`/`xsimd` 字样都是注释（契约对齐说明或性能标注）。公开 API 只吃裸指针，vendor include 全部关在 `src/simd/**` 内部 TU（+ 一个 container 文件），**vendor 类型永不泄漏到 include**。

实际借用共 **4 处专项 kernel**，全是薄借（vendor 做真活，HYLE 只写编排 + 尾部）+ 运行期/编译期门控：

## 1. 超越函数 / math（最大借用，薄借，逐 op best-of）

- **dispatch**：`src/simd/math/math_dispatch.cc`；decls `src/simd/math/kernels.h`
- **借核 TU**：
  - x86：`transcendental_xsimd.cc` / `transcendental_highway.cc` / `transcendental_highway_avx512.cc` / `transcendental_xsimd_avx512.cc`
  - arm64：`transcendental_highway_neon.cc` / `transcendental_xsimd_neon.cc`
  - 复数 SoA：`complex_hyle-inl.h`（cexp/clog/csqrt 借 xsimd AVX2 核；csin/ccos/ctanh/cpow 是 HYLE 编排 xsimd 原语）
  - scalar 底：`transcendental_scalar.cc`
- **借什么**：Highway contrib/math 的 `hn::Exp/Log/Tanh/…` 和 `xsimd::exp/tanh/…` 做真向量化；HYLE 只写 batch 循环 + libm 尾。
- **winner（逐 op 逐精度 bench 定）**：f32 多为 xsimd（exp/tanh/exp2/erf/tgamma + xsimd 独有 erfc/exp10/fmod/fdim）；Highway 赢 log/sin/cos/log2/atan/cbrt/expm1/log1p/asinh/acosh/… 及大部分。**f64 几乎全 Highway**（xsimd 只赢 erf + tgamma）。`lgamma`/`remainder`/`nextafter` 留 scalar（vendor bug）。
- **门控**：运行期 `cpu_features().avx2/avx512`；NEON arm64 无条件。AVX-512 Highway 在 MSVC 关闭（`!defined(_MSC_VER)`，HWY_BROKEN AVX3）。

## 2. vqsort（薄借，纯 Highway）

- **位置**：`src/simd/sort/vqsort_body.inc` + `vqsort_highway_{avx2,avx512,neon}.cc`；dispatch `sort_dispatch.cc`；scalar `vqsort_scalar.cc`
- **借什么**：`hn::VQSortStatic / VQPartialSortStatic / VQSelectStatic`（header-only，`VQSORT_ONLY_STATIC=1`）。sort/partial-sort/select 覆盖 u32/i32/u64/i64/f32/f64 + KV32/KV64 top-k（reinterpret 到 Highway K32V32/K64V64）。
- **门控**：AVX2/AVX512/NEON 运行期派发。

## 3. bitpack（多数薄借，一处 HYLE 自造绕 bug）

- **位置**：`src/simd/bitpack/bitpack_body.inc` + `bitpack_highway_{avx2,avx512,neon}.cc`；`bitpack_dispatch.cc`
- **借什么**：`hn::Pack8/16/32<kBits>` 做 block 打包；HYLE 写 scalar-bitstream 尾（`bitstream.h`）。
- **窄例外**：`bitpack_hyle_u16-inl.h` 是 HYLE 自造 u16 11/13-bit packer（用 `hn::` 原语），绕 Highway 满位宽 round-trip bug。

## 4. UTF-8 计数（薄借，编译期可选）

- **位置**：`src/container/string_algo.cc`
- **借什么**：`simdutf::validate_utf8 + count_utf8`；`-DHYLE_SIMD_SIMDUTF` 门控，否则 scalar。simdutf 自派发 ISA。
- 注：未定义该宏时不 include simdutf，故常规构建的 vendor include 列表里不出现。

## 不借、全自造的部分

`vec.h`/`ops.h` + 五后端（avx2/sse4/neon/scalar/scalable）的核心 op 面、RNG（xoshiro256）、cmp128、swizzle、quantize、crypto（AES/CLMul）、fp8（`cvt_avx10_2.cc` 手写 `_mm512_*`，无 Highway/xsimd）—— 100% HYLE 自己的 intrinsic。无 intdiv/RNG 借用。

**一句话**：HYLE 自造整个核心 SIMD 面 + fp8；只对四个"难自己超越"的招牌 kernel（超越函数 / vqsort / bitpack / 可选 UTF-8）做薄借，全关在 `src/simd/**` 内部 TU，vendor 类型不入 `include/`。

## vendor 收窄备忘（TODO）

- **是否收窄借用面**：超越函数是最大借用面且逐 op best-of，短期难自造超越；vqsort/bitpack 是招牌算法件，薄借合理。若未来要"零 vendor"需重估各自 ROI。
- **借核对拍闸门**：借核同样要过 round-trip/oracle（已抓到 Highway Pack16 + xsimd lgamma 两个 vendor bug），维持"借不等于信"。
