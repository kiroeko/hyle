# simd-04：运行期变体 dispatch（best_isa 真正接上 kernel）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 1-2k |
| **骨架?** | 否 |
| **状态** | 完成 |

## 前置依赖

- 硬前置：`simd-01`（best_isa 探测）+ `simd-02`（scalar/sse4/avx2 后端）。
- 决策依据：[`coding-conventions.md` §10](../architecture/coding-conventions.md)（能力 dispatch 三层 + 两种运行期模型）；[`simd-design.md`](../architecture/simd-design.md) dispatch 表。

## 目的（为什么做这一步）

simd-01 造了 CPUID 探测器（`best_isa()`），simd-02 造了各 ISA 后端——但**两者没接上**：simd-01 红队审计明确指出 `best_isa()` 是**死代码**，除测试外无任何生产代码消费它选 kernel。本里程碑接上这条线：`dot_f32()` 在 AVX2 机上跑 AVX2 kernel、老机上跑 SSE4/scalar，**首次调用时按 `best_isa()` 选定一次**。这是 simd 支柱"一个二进制吃任意 CPU"承诺的兑现点。

## 在架构中的位置

`simd/ops/` 子层。fn-ptr 表模型（像 Highway `HWY_DYNAMIC_DISPATCH`）：每 ISA kernel 独立 `-march` TU，baseline dispatcher 取 fn-ptr。变体-DLL 已复审为错配复杂度：当前公开面全是 whole-buffer batch API，一次间接调用被内核循环摊销；部署选择与边界见 [`../architecture/simd-dispatch-guide.md`](../architecture/simd-dispatch-guide.md)。

## 具体事务（做什么）

- [x] `include/hyle/simd/ops.h`：公开派发入口 `dot`/`saxpy`/`add` 的 f32/f64 API。最初的 `active_variant` 已在 simd-22 被统一 `runtime().ops` 元数据取代；每调用一次指针间接，内核循环保持静态全内联。
- [x] `src/simd/ops/kernels.h`：内部变体符号声明 `detail::*_{scalar,sse4,avx2,avx512,neon}`。
- [x] `src/simd/ops/kernel_impl.h`：共享 kernel 体（ScalableTag 写法，以 `HYLE_SIMD_VARIANT_SUFFIX` 命名），由构建生成的 TU 多次编译。
- [x] `cmake/HyleDispatch.cmake`：按架构生成 scalar/x64/arm64 变体 TU，并以 `hyle_set_isa` 施加唯一的 per-source flag。
- [x] `src/simd/dispatch_internal.h` + `dispatch_ops.cc`：baseline dispatcher 使用 canonical ISA resolver；公开入口经 Meyers static 缓存转发。
- [x] `tests/simd/ops_test.cc`：派发结果对拍 double 参考、`runtime().ops` tier、tail/n=0 边界；完整机制验收见 `simd-21/22`。
- [x] **明确不做**：变体-DLL 零间接模型；没有稳定 batch 契约的寄存器级 `vec.h` 原语保持静态宏腿。

## 交付物

- `ops.h`（公开）+ `kernels.h`/`kernel_impl.h`/`dispatch_ops.cc`（内部）+ `HyleDispatch.cmake` 的构建期 variant TU + ops_test。动态 dispatch 机制的当前总面与验证见 `simd-21`。

## 验收标准

- **编译/正确性**：当前三平台的生成 TU、公开 batch API 与 active-tier 测试均通过；精确测试矩阵和计数见 `simd-21`。
- **SIGILL 安全**：baseline dispatcher + LTO off + 独立宽 ISA TU 保证未知 CPU 先执行的代码不含宽指令；AVX2 仅在 CPUID feature group 满足时解析。特殊特征（F16C/BF16/AVX10）不使用 canonical resolver，而由对应模块精确 gate。
- **对抗自审保留的教训**：跨 ISA TU 的 `kNativeVectorBytes` 不能是外部链接 inline 变量；各宽度不同的常量须为内部链接 `static constexpr`，避免同一 DLL 内 ODR 违规。

## 参考

- conventions §10 两种运行期 dispatch 模型（fn-ptr 表 vs 变体 DLL）。
- Highway `HWY_DYNAMIC_DISPATCH`（fn-ptr 表范本）、llama.cpp `GGML_BACKEND_DL`（变体 DLL 范本）。
