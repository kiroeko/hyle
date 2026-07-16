# SIMD dispatch 指南

SIMD 有两条刻意分开的腿。选择错误会在热路径引入间接调用，或把 ISA 专属类型暴露到未知 CPU 的二进制中。

## 选择哪条腿

| 调用形态 | 使用 | 原因 |
|---|---|---|
| 消费者与部署 CPU 都已知，循环操作的是一个或几个寄存器 | `vec.h` 的 `hs::` API + 本 TU 的 `HYLE_SIMD_TARGET_*` | 类型、lane 数与 backend 都是编译期常量；调用可完全内联，没有运行时分发。 |
| 库导出的 whole-buffer/batch 操作，二进制需要发布到未知 CPU | 模块的 batch API + per-ISA kernel TU + CPUID dispatcher | CPUID 只在首次调用解析一次；每批仅一次可预测间接调用，内核内部仍是静态 `hs::` 全内联。 |

`vec.h` 的寄存器级 API 不是“漏了 CPUID”的语义缺口：返回类型和 lane 数本身依赖目标 ISA，不能安全地在一个运行时 API 中混用。只有能定义稳定的 POD/指针 batch 契约的操作才进入动态腿。

当前动态面是 `ops`、`math`、`dtype`、`scan`、`sort` 与 `bitpack`。个别标量实现（例如 math 中没有可部署向量核的函数）是正确性或实测性能裁决，不代表遗漏 dispatcher。

## 机械生成 variant TU

`cmake/HyleDispatch.cmake` 的 `hyle_add_dispatched_kernel` 只适用于“一个目标无关 impl header 被多个 ISA 薄 TU 编译”的形态。它生成构建目录内的 TU、按架构过滤目标，并通过 `hyle_set_isa` 取得唯一的编译器 flag 映射：

```cmake
hyle_add_dispatched_kernel(
  TARGET hyle_simd
  NAME example
  IMPL "src/simd/example/kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon)
```

x64 生成 `scalar/sse4/avx2/avx512`，arm64 生成 `scalar/neon`，其它架构仅生成 scalar。生成 TU 在 include 前定义 `HYLE_SIMD_TARGET_<ISA>` 与 `HYLE_SIMD_VARIANT_SUFFIX`，include 后立即取消定义；它们不进入源仓库。

`kernel_impl.h` 的契约：

- 仅以 `hs::` D-tag 泛型 API 写 kernel body；不写 ISA `#if` 或 intrinsic。
- 用 `HYLE_SIMD_VARIANT_SUFFIX` 为内部 entry point 命名；本地 token-paste 辅助宏必须在 header 尾部 `#undef`。
- 只定义内部函数；导出 ABI 仍由 baseline dispatcher 提供固定 POD/指针签名。

CMake 不生成 C++ dispatcher。不同模块的 ISA 选择还可能含 Linux 性能裁决、F16C/BF16/AVX10 特征组合、供应商核限制或算法分流；把这些语义压进 `FNS` 字符串会制造无法审计的构建时代码生成。

## dispatcher 契约

普通完整 ISA 阶梯的 baseline dispatcher 可使用 `src/simd/dispatch_internal.h`：

```cpp
return detail::resolve_default_variant<Fn>(
    &detail::kernel_scalar, &detail::kernel_sse4,
    &detail::kernel_avx2, &detail::kernel_avx512);
```

该 helper 只处理已编译的 canonical ISA 梯阶。任何额外特征、平台性能分流或特殊值策略必须留在模块 dispatcher；scalar fallback 永远存在。dispatcher/probe 必须保持 baseline 编译且模块关闭 LTO，避免高 ISA 指令泄入未知 CPU 最先执行的代码。

新增一个动态 batch kernel 的最小改动是：定义稳定 batch 契约、在符合条件时写入现有 impl header、在内部声明/模块 dispatcher 绑定 variant，并加 scalar oracle 与目标 ISA 测试。若没有 stable batch 契约，应保持 header-only 静态 API，而不是人为创建 CPUID 包装。

## 验收

1. Win/WSL 构建必须展示由 CMake 生成的正确 ISA TU；arm64 构建只能出现 NEON + scalar。
2. 每个 batch API 对拍 scalar oracle；active-tier 测试验证 CPUID 选择的已编译变体。
3. 变体 TU 用目标 ISA 编译，dispatcher/probe 保持 baseline 且 LTO 关闭。
4. direct 与 public dispatch 的对照只能在同一 active ISA 上解释；小批量固定税与大批量内核吞吐分开记录。
