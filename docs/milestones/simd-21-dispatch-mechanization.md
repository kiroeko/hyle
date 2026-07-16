# simd-21: 双腿分发机制化

> **状态**:已落地，并由 [simd-22](./simd-22-dual-track.md) 扩展/修正。本文记录 variant TU
> 机械化的历史设计；新的公共 `Runtime`/`native` 双轨接口、能力边界与迁移状态以
> [`../architecture/simd-dispatch-guide.md`](../architecture/simd-dispatch-guide.md) 为准。

## 一、两条腿与边界

| 腿 | 适用面 | 形态 | 运行时成本 |
|---|---|---|---|
| 静态宏腿 | `vec.h` 的寄存器/D-tag 原语 | 消费者 TU 定义 `HYLE_SIMD_TARGET_*`，编译器选择 backend 与 lane 类型 | 零；所有操作可内联 |
| 动态 CPUID 腿 | 稳定 ABI 的 whole-buffer/batch API | baseline dispatcher 选 per-ISA kernel 函数指针并缓存 | 每 batch 一次可预测间接调用 |

宏腿不是漏了 CPUID：寄存器类型、lane 数与返回类型本身依赖 ISA，不能安全地伪装成一个运行时 API。
只有能定义 POD/指针/opaque-state batch 契约的语义操作才进入动态腿。本文当时所称“没有缺口”
只统计了已有 batch 声明，遗漏了 `VectorXoshiro` 这类高层静态-only 能力，也没有为源码消费者提供
公共的精确目标入口；simd-22 已用 runtime-safe RNG 和 `native` 路径补上第一条竖切。

## 二、已落地的机械化范围

`cmake/HyleDispatch.cmake` 提供：

```cmake
hyle_add_dispatched_kernel(
  TARGET hyle_simd
  NAME ops
  IMPL "src/simd/ops/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon)
```

它在**构建目录**生成薄 variant TU，复用 `hyle_set_isa` 的唯一 flag 映射，并按架构过滤：x64 为
`scalar/sse4/avx2/avx512`，arm64 为 `scalar/neon`，未知架构仅 scalar。`ops` 与 `scan` 已迁移；
手写的十个薄 TU 已删除。

这项机制只覆盖“一个目标无关 `kernel_impl.h` 被多个 ISA 编译”的真实公因式。生成 TU 在 include 前
定义 `HYLE_SIMD_TARGET_<ISA>` 与 `HYLE_SIMD_VARIANT_SUFFIX`，include 后立即取消定义；没有生成源码
进入仓库。

`src/simd/dispatch_internal.h` 提供完整 canonical ISA 阶梯的 `default_dispatched_isa()` 与
`resolve_default_variant()`。`ops` 使用它消除了六份手写 tier switch；`scan` 使用它但保留
Linux x64 的纯标量性能裁决。`cloc` 实测机制本体为 128 code LOC：`HyleDispatch.cmake` 86、
`dispatch_internal.h` 42；其余改动复用既有 kernel body 与模块语义。

## 三、刻意不泛化的部分

不生成 C++ dispatcher，也不在公共 `dispatch.h` 放注册模板。函数签名、特殊 CPU feature、fallback
与 ABI 是模块语义，而非 CMake 元数据：

- `dtype` 按 F16C、AVX512-BF16、AVX10 等精确 feature gate 选择；
- `math` 按操作选择不同供应商核、特殊值策略与 MSVC AVX-512 限制；
- `sort`/`bitpack` 还含算法前置分流与 Highway 静态 target 约束；
- `scan` 的 Linux x64 纯标量路径是实测性能策略。

这些模块继续手写 dispatcher，避免把 `FNS` 字符串代码生成变成无法审计的宏系统。将来只有新增模块同时
满足 shared impl header、完整 ISA 阶梯、无额外 feature/算法策略时，才使用 CMake helper。

## 四、宏与 ISA 契约

1. `kernel_impl.h` 是刻意多重 include 的私有实现头；入口只使用
   `HYLE_SIMD_VARIANT_SUFFIX`。所有 token-paste 辅助宏以模块前缀命名并在头尾 `#undef`，不污染同一
   TU 的后续 include。
2. variant TU 只生成内部符号；导出 API 仍在 baseline dispatcher，以固定 POD/指针 ABI 表达。
3. dispatcher/probe 保持 baseline 编译，`hyle_simd` 强制 LTO 关闭；宽 ISA 指令只能出现在 per-ISA TU。
4. AVX-512 `SlideUpLanes`/`SlideDownLanes` 的 `valign` offset 是 intrinsic immediate。实现显式展开
   有界 lane offset，修复了 MSVC 因运行时 offset 拒绝编译而暴露的跨编译器缺陷。

## 五、验收

- Windows MSVC：生成 x64 的八个 `ops`/`scan` TU；`hyle_simd_test` 331 PASS、1 个无硬件 BF16 SKIP，
  `hyle_simd_avx2_test` 221 PASS。
- WSL GCC14：`hyle_simd_test` 331 PASS、1 个无硬件 BF16 SKIP，`hyle_simd_avx2_test` 221 PASS；
  SDE `-future` 的 `hyle_simd_avx512_test` 221 PASS。
- Mac Apple Silicon：只生成 `scalar/neon` 的四个 TU；`hyle_simd_test` 328 PASS、1 个无硬件 BF16 SKIP。
- `bench/simd/simd_bench.cc` 新增 same-active-ISA 的 direct/public dispatch batch-size sweep。它把小 batch
  的固定间接调用税与大 batch 内核吞吐分开；没有同 ISA direct anchor 的 tier 明确省略该对照，不伪造结果。
