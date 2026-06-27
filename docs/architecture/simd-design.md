# HYLE simd/ 设计

> SIMD 抽象层（Tier 1，性能焦点 #1）。clay 这块基本空白，HYLE 净新 ~10-16k 行。上游约束见 [`hyle-scope.md`](./hyle-scope.md)、[`engineering-charter.md`](./engineering-charter.md)。

## clay 现状：近乎空白

- `simd.h` 全文 **43 行**，只做编译期指令集探测（`__SSE2__`→emmintrin、`__SSE4_2__`→nmmintrin、ARM→sse2neon），**只定义 `CLAY_HAVE_SSE2/SSE4` 两个宏，无 vector 类型、无 AVX/AVX512、无运行时 dispatch**。
- `half.h` 是纯软件查表转换（512+2048 项表），无 F16C 硬件路径。
- 结论：**等于从零造**，但可继承 `math/utils.h` 的标量工具、`host_device.h` 宏、`CLAY_ALIGNMENT 64`。

## 2026 格局

| 库 | 性质 | 对 HYLE |
|---|---|---|
| **std::simd（C++26 P1928）** | 定宽、无运行时 dispatch | ❌ 不能当核心——"表达不了 90% 真实 SIMD 代码"（gather/scatter/mask/swizzle 弱）；仅可作平凡循环的薄互操作 |
| **Highway（Google）** | 可移植 + 运行时 dispatch + scalable vector（AVX512/NEON/SVE/RVV/WASM）| 金标准，但 C++ idiom 重、头文件庞大，整体 vendor 与"贴硬件/不过度抽象"有张力 |
| **xsimd** | 定宽 + 优化数学函数（exp/log/trig）| 数学核可借 |
| **sleef** | 高精度 SIMD transcendental 核 | exp/tanh 等的多项式核可借（别手搓，易错） |

## 核心决策：自造精简抽象 + 运行时 dispatch + 借数学核

与 io/（借 liburing 不借 asio）、compute/（借 ggml 算子不借框架）同构——**借难且已解决的、造贴自己需求的**：

| 部件 | 造/借 | 理由 |
|---|---|---|
| 可移植 vector 抽象 `vec<T,N>` + 多 ISA 后端 | 🔧 **自造（精简）** | 焦点 #1；不 vendor 整个 Highway（idiom 重、违背贴硬件）；参考其 scalable-vector + dispatch 理念造精简版 |
| 运行时 dispatch（CPUID + 多版本）| 🔧 **自造** | clay 完全没有，是核心缺口 |
| transcendental 数学核（exp/log/sin）| ✅ **借 sleef-style 多项式核** | 手推多项式逼近易错，已解决问题 |
| dtype 转换（fp16/bf16/int8/fp8）| 🔧 自造（硬件路径 + 软件 fallback）| 替换 clay 纯软件 half 表 |
| AI 数学函数（gelu/softmax/tanh）| 🔧 自造（组合数学核）| HYLE 特定，AI 热路径 |
| std::simd 互操作 | ⚠️ 薄封装 | 平凡循环可转发给它，非核心 |

## 分层结构

```
simd/
├─ vec.h          可移植 vector 类型 vec<T, Width>（load/store/算术/fma/compare/mask/shuffle）
├─ backend/       per-ISA 实现（编译多版本）
│  ├─ sse42.h  avx2.h  avx512.h     (x86)
│  ├─ neon.h   sve.h                 (arm)
│  └─ scalar.h                       (无 SIMD fallback)
├─ dispatch.h     运行时 CPUID 探测 + 函数指针表选最优后端
├─ dtype/         fp16/bf16/int8/fp8 ↔ fp32（F16C/AVX512-FP16/BF16 硬件 + 软件 fallback）
├─ math/          exp/log/tanh/gelu/softmax/erf（借 sleef 核 + 组合）
└─ ops/           供 container/compute 用的 SIMD 原语（如 SwissTable 探测、reduce、memcpy）
```

## 运行时 dispatch（clay 最大缺口）

clay 是编译期固定指令集，**跑在 AVX512 机器上却只用 SSE2**。HYLE 必须运行时选最优：

1. **多版本编译**：热函数对每个 ISA 各编一份（`avx512_kernel` / `avx2_kernel` / `neon_kernel`…）。
2. **CPUID 探测**（启动一次）：查 CPU 支持的最高指令集。
3. **函数指针表**：首次调用解析到对应版本，之后直调（或 IFUNC/全局指针）。

这与 `compute/` 的后端 dispatch（CPU/CUDA/Vulkan 运行时选）是同一哲学——**运行时按硬件能力选实现**，只是粒度从"设备"到"指令集"。

## dtype 层（AI 数值刚需，scope 完整性评估补的缺口）

| 类型 | 硬件路径 | fallback |
|---|---|---|
| fp16 | F16C（`_mm_cvtph_ps`）/ AVX512-FP16 | 软件查表（继承 clay half）|
| bf16 | AVX512-BF16 / 截断转换 | 移位截断（bf16 就是 fp32 高 16 位）|
| int8/fp8 | AVX512-VNNI（int8 点积）| 标量 |

bf16 尤其重要（AI 训练/推理主力格式）；clay 完全没有。

## 与 mem/container 的耦合 + 一个 tier 倒置的澄清

- **对齐**：vector load/store 要求 `CLAY_ALIGNMENT=64` 对齐，与 `mem/` 分配器对齐契约一致（AVX512 一条 512-bit load 要 64B 对齐）。
- **SwissTable SIMD 探测的 tier 问题**：`container/`（Tier 0）的 SwissTable 要 16 路并行探测 control word，看似依赖 `simd/`（Tier 1）——**tier 倒置**。澄清：探测只需一小段自包含的 SSE2/NEON 片段（16 字节比较 + movemask），**container/ 内联这一小段、不依赖整个 simd/ 模块**。保持 T0 不依赖 T1。`simd/` 成熟后 container 可选择复用其 `ops/`，但不强耦合。

## 落地清单

1. `vec<T,N>` 抽象 + scalar fallback 后端（先能跑、再加速）。
2. SSE4.2 / AVX2 / AVX512 / NEON 后端 + 运行时 dispatch。
3. dtype 转换（先 fp16/bf16 硬件路径）。
4. 数学函数（借 sleef 核：exp/tanh → 组合出 gelu/softmax）。
5. `ops/`：reduce、对齐 memcpy、SwissTable 探测片段。
6. 单测：每后端对拍 scalar 参考值、跨 ISA CI、数学函数精度 ULP 测试、benchmark 对标 Highway/手写 intrinsics。

## 待决（需主拍板）

1. **vector 抽象用定宽 `vec<T,N>` 还是 scalable（length-agnostic）**？**✅ 已决：scalable-friendly 接口 + 先实现定宽后端**——接口按 length-agnostic 设计（参考 Highway，将来加 SVE/RVV 不改 API），但首版只实现定宽后端（x86 SSE4.2/AVX2/AVX512 + ARM NEON，它们本就定宽）。对冲：不堵未来又不先背复杂度。
2. **数学核 vendor sleef 还是自抄多项式**？建议 vendor sleef（Apache/BSL，精度经考验），只在 adapter 里封——同 ggml 解耦思路。
3. **要不要 SVE/RVV 进首版**？建议 v1 只做 x86(SSE4.2/AVX2/AVX512) + ARM NEON，SVE/RVV 留接口、缓实现。

## 参考

- [C++26 Shipped a SIMD Library Nobody Asked For（std::simd 表达力批判）](https://lucisqr.substack.com/p/c26-shipped-a-simd-library-nobody)
- [Highway：可移植 + 运行时 dispatch SIMD](https://google.github.io/highway)
- [xsimd：SIMD wrapper + 优化数学函数](https://github.com/xtensor-stack/xsimd)
- [Evaluation of C++ SIMD Libraries（AVX2/AVX512/SVE/NEON 基准）](https://www.mnm-team.org/pub/Fopras/rock23/PDF-Version/rock23.pdf)
