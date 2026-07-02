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
- 决策依据：[`simd-design.md`](../architecture/simd-design.md)（数学核 vendor SLEEF 不手搓 + xsimd 仅参考）、[`references.md`](../architecture/references.md)（SLEEF=借）。

## 目的

AI 热路径（softmax/gelu/attention）被 exp/log/tanh 主导。手搓多项式逼近易错——**借 SLEEF 验证过的核**（simd-design 铁律"借 sleef 不手搓"），不自己写多项式。

## 关键决策：消费预构建 libsleef（库路径），非内联头（GCC-only）

> **纠错记录**：原想走 SLEEF 的"内联头"（header-only，零链接依赖）。**实测内联头 GCC-only**——MSVC 编不过（`__builtin_inf`/GCC 复合字面量）。但 **SLEEF 库本身支持 MSVC**（`winbuild-msvc.bat`），其**声明头 `sleef.h` 纯 extern、零 GCC builtin、可移植**。故走**库路径**：消费预构建 libsleef（链 `libsleef.a`/`sleef.lib` + include 纯声明 `sleef.h`，喂 `vec.raw`）——与 mem 链 mimalloc_static 同款模式。是为双平台正解。

## 具体事务

- [x] `include/hyle/simd/math.h`：`exp_f32_batch`/`log_f32_batch`/`tanh_f32_batch` + `active_math_variant`。
- [x] `src/simd/math/`：scalar libm 核（`std::exp/log/tanh`，**永远编、双平台、ULP oracle**）+ SLEEF AVX2 核（`Sleef_expf8_u10avx2` 等，`_u10`=1ULP，**opt-in**，gate 在 `HYLE_SIMD_SLEEF`）+ baseline dispatcher（按 `HYLE_SIMD_SLEEF && cpu_features().avx2` 选 SLEEF 否则 libm）。
- [x] CMake：`HYLE_SIMD_SLEEF` 选项(默认 OFF)+ `HYLE_SLEEF_DIR` 指向预构建 SLEEF；find_library 链 libsleef + include 纯声明头；SLEEF TU 编 `-mavx2 -mfma`。
- [x] `tests/simd/math_test.cc`：exp/log/tanh 批量对拍 libm（相对容差容 1-ULP SLEEF + f32 舍入）+ tail + active 报告。
- [x] **明确不做**：gelu/softmax(组合核,后续)；double/其他宽；**MSVC 上 build libsleef + 向量化路径**(需先跑 winbuild-msvc 产 sleef.lib,登记待补)。

## 验收标准

- **编译**：✅ Win MSVC 19.50（默认 scalar 路径，/W4 零告警）+ Linux GCC14.2（默认 + SLEEF-wired 均零告警）。
- **正确性**：✅ **双平台默认 scalar libm 路径**：Win+WSL `hyle_simd_test` 全过、全量 ctest 各 13/13。**WSL SLEEF-wired 路径**：exp/log/tanh 向量化(`Sleef_*f8_u10avx2`)对拍 libm 全过(5/5 Math),`active_math_variant()`=kAvx2 证 SLEEF 真生效。
- **批量红队补审（2026-06）**：原测试相对容差 `1e-5`(≈84 ULP f32)对 broken 多项式无防护。实测两路径真实误差——scalar 0.5-0.7 ULP、SLEEF `_u10` 0.5 ULP——后收紧到 `4e-7`(≈3.4 ULP,实测 worst 5x 余量,4 ULP 起的坏核会 fail);双平台 + SLEEF-wired 两路径收紧后仍过。另补 log/tanh 的 tail 测试(原只 exp)。
- **借非造**：向量化核是 SLEEF 的 `_u10`(1ULP)cores,HYLE 零多项式手搓——adapter 只喂 `vec.raw` 等价的 `__m256`。
- **诚实边界**：① SLEEF 向量化 opt-in(默认 OFF,需 `-DHYLE_SIMD_SLEEF=ON -DHYLE_SLEEF_DIR=<built sleef>`);默认构建 math 经 libm 正确但非向量化。② MSVC 向量化未跑(本轮只在 WSL 链 libsleef.a;MSVC 需 winbuild-msvc 产 sleef.lib——登记待补,设计上可行因 sleef.h 可移植+winbuild-msvc 在)。

## 参考

- SLEEF：`winbuild-msvc.bat`(MSVC 库构建)、生成的 `sleef.h`(纯声明)、`Sleef_expf8_u10avx2` 等 `_u10` cores。
- mem `mimalloc_static`（链 vendored 静态库 + 符号内部化的先例）。
- simd-design 数学核决策；bench-absl/highway(指向预构建 sibling 的 opt-in 模式)。
