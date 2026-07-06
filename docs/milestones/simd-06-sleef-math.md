# simd-06：transcendental 数学核（exp/log/tanh）

> **当前状态（2026-06，SLEEF 退役后回填）**：本里程碑交付的向量化数学核**现走 Highway + xsimd**（均 vendored 在 `third_party/`、无条件默认编进生产、逐 op best-of-breed 选最快核）。**最初的 SLEEF 后端已完全删除**（代码 + `HYLE_SIMD_SLEEF`/`HYLE_SLEEF_DIR` 开关 + `transcendental_sleef.cc`）——理由：SLEEF 非 header-only（需预编库），且 dispatcher 三个 op 都把它排第三，xsimd+Highway 都在时永不被选，纯死代码。下文保留 SLEEF 时期的实现/纠错记录作历史；"借核"现读作"借 Highway+xsimd（已 vendored）"。最新性能数字见 [`../performance/benchmarks-simd.md`](../performance/benchmarks-simd.md) 数学核节。

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 1-2k |
| **骨架?** | 否 |
| **状态** | ✅ 完成（scalar libm 双平台实跑；向量化核 Highway+xsimd vendored 无条件默认开，exp/log/tanh 对拍 libm + 同 ISA SOTA） |

## 前置依赖

- 硬前置：`simd-04`（dispatch 模型）。软：`simd-02`（AVX2 后端结构）。
- 决策依据：[`simd-design.md`](../architecture/simd-design.md)（数学核借验证核不手搓 + 逐 op best-of-breed）、[`references.md`](../architecture/references.md)（Highway/xsimd=借）。

## 目的

AI 热路径（softmax/gelu/attention）被 exp/log/tanh 主导。手搓多项式逼近易错——**借验证过的向量化核**（simd-design 铁律"借核不手搓"），不自己写多项式。

## 借核选型：Highway + xsimd（vendored header-only,逐 op best-of-breed）

> **为何不是 SLEEF（承重决策历史）**：最初选 SLEEF,但 SLEEF **非 header-only**（需预编 libsleef,MSVC 要 winbuild-msvc 产 sleef.lib）,且实测在 dispatcher 里三个 op 都被 xsimd/Highway 排到第三、永不被选——纯死代码。故 **SLEEF 后端完全删除**,改借同样验证过、但 **header-only 可 vendored** 的 Highway（contrib/math）+ xsimd。逐 op 选最快:**exp/tanh→xsimd、log→Highway**（同 ISA bench 结论,见 performance）。header-only 意味着"不可能缺失"、无预编库、无 opt-in 开关——一个高性能库默认就该带最快路径。

## 具体事务

- [x] `include/hyle/simd/math.h`：`exp_f32_batch`/`log_f32_batch`/`tanh_f32_batch` + `active_math_variant`。
- [x] `src/simd/math/`：scalar libm 核（`std::exp/log/tanh`,**永远编、跨平台、ULP oracle + 非 x86/arm fallback floor**）+ **Highway/xsimd 借核**（`transcendental_highway.cc` log、`transcendental_xsimd.cc` exp/tanh,AVX2;`*_avx512.cc` 512b 变体;`transcendental_highway_neon.cc` arm64 全三 op）+ baseline dispatcher（`math_dispatch.cc`:x86 按 best_isa 选 avx512/avx2 借核否则 libm;arm64 走 Highway NEON;非 x86/arm 走 libm）。
- [x] CMake：Highway+xsimd include dir 无条件（header-only vendored,不可缺失,无 opt-in 门）;x86 借核 TU 按 tier 设 `-mavx2`/`-mavx512*`（`hyle_set_highway_isa`/`hyle_set_avx512_math_isa`）;arm64 NEON TU 无额外 flag（ARMv8 baseline）。
- [x] `tests/simd/math_test.cc`：exp/log/tanh 批量对拍 libm（收紧到 `4e-7`≈3.4 ULP）+ tail + active 报告（arch-aware:x86 kAvx2/kAvx512、arm64 kNeon、else kScalar）。
- [x] **明确不做**：gelu/softmax(组合核,后续)；double/其他宽。

## 验收标准

- **编译**：✅ 三平台——Win MSVC 19.50 + Linux GCC14.2 + macOS clang21,零告警（Highway AVX3 在 MSVC 会发 ymm 已 `#if !_MSC_VER` 排除,退 AVX2）。
- **正确性**：✅ exp/log/tanh 向量化核对拍 libm 全过——x86 AVX2/AVX512（`active_math_variant()`=kAvx2/kAvx512 证借核真生效）+ **arm64 NEON**（Highway lower 到 NEON,`=kNeon`,真机 M4 Air 验）+ scalar libm 双平台。收紧容差 `4e-7`(实测 worst 5x 余量,坏核会 fail)后所有路径仍过。
- **借非造**：向量化核是 Highway `hn::Exp/Log/Tanh` + xsimd `_u10`(1ULP)cores,HYLE 零多项式手搓——adapter 只喂 `vec.raw` 等价的 `__m256`/NEON vector。
- **诚实边界**：① gelu/softmax 组合核待后续。② AVX512 借核在无 AVX512 机/SDE 上未真跑（GCC objdump 验 zmm）。③ **arm64 用 Highway 跑全三 op**;x86 的 xsimd(exp/tanh)/Highway(log) per-op 拆分是速度调优,arm64 的 xsimd-NEON 逐 op 调优待实测（Highway-NEON 已够用）。

## 参考

- **Highway** `hwy/contrib/math/math-inl.h`（`hn::Exp/Log/Tanh`,跨 ISA:AVX2/AVX512/NEON 同一核）;**xsimd** `xsimd::exp/log/tanh`（AVX2/AVX512 多项式最快）。均 header-only vendored 在 `third_party/`。
- mem `mimalloc_static`（vendored 依赖默认带最快路径、无 opt-in 的先例）。
- simd-design 数学核决策；performance/benchmarks-simd 数学核节（同 ISA SOTA 对标数字）。
