# SIMD vendor 借用面（现状盘点）

> 本文只回答一个问题：**有哪些地方我们直接调用 Highway / xsimd / simdutf（而非 HYLE 自造后端）？**
>
> **覆盖率 / op 计数 / SOTA 判决不在本文** —— 唯一数据源是
> [`../milestones/simd-orderswap-matrix.md`](../milestones/simd-orderswap-matrix.md)（权威计数树 380→340→299→291、
> 全量表、独有测试表、41 待补 bench、语义缺口）。数字随代码演进，跑 `scripts/simd-op-coverage.py` 取实时值。

**关键前提**：`include/` 里**零** vendor 调用——所有头文件里的 `hwy`/`xsimd` 字样都是注释（契约对齐说明或性能标注）。公开 API 只吃裸指针，vendor include 全部关在 `src/**` 内部 TU（+ 一个 container 文件），**vendor 类型永不泄漏到 include**。

实际借用共 **1 处专项 kernel**（可选 UTF-8）。math / bitpack / vqsort 已是 HYLE 实现。

## 1. 超越函数 / math（HYLE 实现，best-of 保留）

- **dispatch**：`src/simd/math/math_dispatch.cc`；decls `src/simd/math/kernels.h`
- **实现**：
  - Highway 算法侧：`math_hyle-inl.h`（contrib/math 语义移植）+ `table_f64.h`（P2-1 N=128 f64 exp/log）
  - xsimd 算法侧：`math_xsimd_hyle-inl.h` + `math_xsimd_body.inc`
  - ISA TU：`transcendental_{highway,xsimd}_{avx2,avx512,neon}.cc`（符号名保留 `*_highway_*` / `*_xsimd_*` 供 best-of 接线）
  - 复数 SoA：`complex_hyle-inl.h`（纯 HYLE 编排，依赖 port 后的 Exp/Log/SinCos/…）
  - scalar 底：`transcendental_scalar.cc`
- **winner 表不变**（编译期 per-op）：f32 多为 xsimd 算法路径（exp/tanh/exp2/erf/tgamma + erfc/exp10/fmod/fdim）；Highway 算法路径赢 log/sin/cos/…；f64 几乎全 Highway 算法路径（xsimd 路径赢 erf + tgamma）。`lgamma`/`remainder`/`nextafter` 留 scalar。
- **门控**：运行期 `cpu_features().avx2/avx512`；NEON arm64 无条件。产品 math 不再 `#include` highway/xsimd。

## 2. vqsort（HYLE 实现）

- **位置**：`src/simd/sort/vqsort_hyle/`（`vqsort-inl.h` + networks/traits/ops）+ `vqsort_hyle_{avx2,avx512,neon}.cc`；dispatch `sort_dispatch.cc`；scalar `vqsort_scalar.cc`
- **实现**：Highway `contrib/sort` 语义移植为 HYLE D-tag；`VQSortStatic` / `VQPartialSortStatic` / `VQSelectStatic` 覆盖 u32/i32/u64/i64/f32/f64 + KV32/KV64 top-k。
- **门控**：AVX2/AVX512/NEON 运行期派发。产品 sort 不再 `#include` highway。

## 3. bitpack（HYLE SIMD 实现）

- **位置**：`src/simd/bitpack/bitpack_hyle-inl.h` + `bitpack_hyle_{avx2,avx512,neon}.cc`；`bitpack_dispatch.cc`
- **实现**：HYLE D-tag `Pack8/16/32<kBits>` 做完整 block 打包；`bitstream.h` 只处理不足一个 SIMD block 的尾。
- **u16 例外布局**：`bitpack_hyle_u16-inl.h` 保留 11/13-bit 的已验证 SIMD 修正布局，涵盖 13-bit rawF scatter mask 与两个 inverse 的向量选择。

## 4. UTF-8 计数（薄借，编译期可选）

- **位置**：`src/container/string_algo.cc`
- **借什么**：`simdutf::validate_utf8 + count_utf8`；`-DHYLE_SIMD_SIMDUTF` 门控，否则 scalar。simdutf 自派发 ISA。
- 注：未定义该宏时不 include simdutf，故常规构建的 vendor include 列表里不出现。

## 不借、全自造的部分

`vec.h`/`ops.h` + 五后端（avx2/sse4/neon/scalar/scalable）的核心 op 面、RNG（xoshiro256）、cmp128、swizzle、quantize、crypto（AES/CLMul）、fp8、**math**、**bitpack**、**vqsort**—— 100% HYLE 自己的 intrinsic / 移植核。

**一句话**：HYLE 自造核心 SIMD 面 + fp8 + bitpack + math + vqsort；只对可选 UTF-8 做薄借，全关在 `src/**` 内部 TU，vendor 类型不入 `include/`。

## vendor 摘除状态

simd-20：X0 bench 外部 pin、X1 bitpack、X2 math、**X3 vqsort** 已落地。产品 `hyle_simd` **不再** include `third_party/{highway,xsimd}`。剩余 **X4** 删仓库内 `third_party/{highway,xsimd}` 与 `check-no-vendor.sh` 闸门收口。`simdutf` 不在本轮范围。

移植核过 round-trip/oracle（已抓到 Highway Pack16 + xsimd lgamma 两个 vendor bug；X3 另修 SSE4 partial-tag 全宽 LoadU 堆损），保持"借不等于信"。完整阶段与验证见 [`../milestones/simd-20-vendor-extraction.md`](../milestones/simd-20-vendor-extraction.md)。
