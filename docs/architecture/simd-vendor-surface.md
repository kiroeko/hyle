# SIMD vendor 借用面（现状盘点）

> **V2 状态：`NOT_REVIEWED`。** 本页盘点既有 vendor 边界，不是当前 SIMD 功能或性能证书；
> 具体计数与“终态”字样要在 SIMD 轮次重新生成和裁决。

> 本文只回答一个问题：**有哪些地方我们直接调用 Highway / xsimd / simdutf（而非 HYLE 自造后端）？**
>
> **覆盖率 / op 计数 / SOTA 判决不在本文** —— 唯一数据源是
> [历史 SIMD closure](../implementation/tasks/simd/simd-module-closure.md)。数字随代码演进，跑 `scripts/simd-op-coverage.py` 取实时值。

**关键前提**：`include/` 里**零** vendor 调用。公开 API 只吃裸指针。

## 终态（simd-20 X4）

**库零 Highway/xsimd**：`include/` + `src/` 不 `#include` `hwy/` 或 `xsimd/`；仓库内**无** `third_party/{highway,xsimd}`。闸门：`scripts/check-no-vendor.sh`。

**bench 外部对拍**：`HYLE_BUILD_BENCH=ON` 时经 `HYLE_BENCH_VENDOR_DIR` + `cmake/vendor-versions.cmake` pin 的外部 checkout 引用 Highway/xsimd；`scripts/fetch-bench-vendors.{sh,ps1}` 按 SHA 拉取。产品 build 不读该目录。

## 产品实现面（全 HYLE）

| 面 | 位置 | 说明 |
|----|------|------|
| math | `math_hyle-inl.h` / `math_xsimd_hyle-inl.h` + ISA TU | best-of 符号名保留 `*_highway_*` / `*_xsimd_*` |
| bitpack | `bitpack_hyle-inl.h` + ISA TU | Pack8/16/32 D-tag |
| vqsort | `vqsort_hyle/` + ISA TU | sort/partial/select + KV |
| 核心 op | `vec.h` + scalar/x86/ARM/RISC-V 七后端 | intrinsic / 自造 |

这些产品 kernel 遵循同一双轨边界：ops/random/scan 的 per-ISA wrapper 与 `native::` 复用同一 D-tag body；
math/bitpack/vqsort 虽保留各自算法 dispatcher，也只通过 pointer/POD batch API 暴露。vendor 摘除与
runtime/native/exact 派发彼此正交，详见 [`simd-dispatch-guide.md`](../developer/maintenance/simd-dispatch-guide.md)。

## container 直接借用的成熟 SIMD 核

**UTF-8 validate/count**（`src/container/string_algo.cc`）：vendored simdutf v9.1 amalgamation 无条件编入
`hyle_container`；`utf8_validate` 直接借单遍 `validate_utf8`，`utf8_length` 借
`validate_utf8 + count_utf8`，并复用其 runtime ISA dispatch。
公开 container 头不泄露 simdutf 类型；scalar decoder 保留为 constexpr/reference fallback。

**一句话**：HYLE 自造核心 SIMD + math + bitpack + vqsort；仓库无 highway/xsimd；bench 用外部 pin 对拍；container 的 UTF-8 validate/count 直接借 vendored simdutf。

完整阶段史见 [`../milestones/simd-20-vendor-extraction.md`](../implementation/tasks/simd/simd-20-vendor-extraction.md)。
