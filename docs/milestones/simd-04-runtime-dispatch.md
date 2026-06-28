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

`simd/ops/` 子层。fn-ptr 表模型（像 Highway `HWY_DYNAMIC_DISPATCH`）：每 ISA kernel 独立 `-march` TU，baseline dispatcher 取 fn-ptr。**零间接的变体-DLL 模型**（每 ISA 独立 .so + dlopen，调用全直接）是更重的后续里程碑——conventions §10 记两种皆一等公民，按部署选。

## 具体事务（做什么）

- [x] `include/hyle/simd/ops.h`：公开派发入口 `dot_f32`/`saxpy_f32`/`active_variant`（HYLE_SIMD_API 导出）。注释标清 fn-ptr 表模型 = 每调用一次指针间接（诚实，非零间接）。
- [x] `src/simd/ops/kernels.h`：内部变体符号声明 `detail::dot_f32_{scalar,sse4,avx2}` 等。
- [x] `src/simd/ops/kernel_impl.h`：共享 kernel 体（ScalableTag 写法，HYLE_KERNEL_SUFFIX 命名），编 3 次。
- [x] `kernels_{scalar,sse4,avx2}.cc`：3 个变体 TU,各 `HYLE_SIMD_TARGET_*` + `-march`。
- [x] `src/simd/ops/dispatch_ops.cc`：baseline dispatcher，`selected_tier()` 映射 best_isa→{scalar,sse42,avx2}，`resolve_*` 选 fn-ptr，公开入口经 Meyers static 缓存转发。
- [x] CMake：变体 TU per-source `-march`，dispatcher 保持 baseline，整模块 LTO off。
- [x] `tests/simd/ops_test.cc`：派发结果对拍 double 参考 + `active_variant()==best_isa 映射 tier` + tail/n=0 边界。
- [x] **明确不做**：变体-DLL 零间接模型；avx512/neon 变体（待 simd-03 后端）；dot/saxpy 之外的 op（按需加）。

## 交付物

- ops.h（公开）+ 5 个内部文件（kernels.h/kernel_impl.h/3 变体 TU/dispatch_ops.cc）+ ops_test。cloc：生产 111、测试 67。这是**派发接线**非新算法，故精瘦合理。

## 验收标准

- **编译**：✅ Win MSVC 19.50（/W4 零告警）+ Linux GCC14.2 Release（零告警）。
- **正确性**：✅ 双平台 `hyle_simd_test` 36/36（31 既有 + 5 派发）。派发结果对拍 double 精度参考（相对容差，容 fp reorder 但抓逻辑 bug——精确 per-op 数学在 vec_test 整数 oracle 钉死）；`active_variant()` 本机 = kAvx2（证 best_isa 真被消费选 kernel）。全量 ctest 13/13 无回归。
- **SIGILL 安全（红队代码分析核实，未在非 AVX2 机实跑——本机即 AVX2,无老硬件可验,同 NEON"仅未跑"口径）**：老 CPU（无 AVX2）→ `selected_tier()` 返 kSse42/kScalar → `&dot_f32_avx2` **永不取址、永不调用**；avx2 符号在二进制中但不执行=安全；baseline dispatcher + LTO off + 独立 TU = 无跨 TU 内联。**FMA gating 闭合 AVX2-without-FMA 漏洞**（kAvx2 要求 fma 位，故 AVX2 kernel 的 `_mm256_fmadd` 永不在无 FMA CPU 跑）。**待补真验**：在禁 AVX2 的环境（如 QEMU `-cpu` 限 SSE4.2、或 `sde -snb`）实跑确认不 SIGILL，登记为后续。
- **对抗自审修掉的真 bug**：`kNativeVectorBytes` 原 `inline constexpr`（外部链接）但**各 TU 值不同**（avx2 TU=32，其余=16）——inline 变量须各 TU 定义一致,divergent = IFNDR。simd-02 时各变体在独立测试 exe 故潜伏；simd-04 把 scalar+sse4+avx2 kernel TU 链进**同一 hyle_simd.dll**,变真 ODR 雷。改 `static constexpr`(内部链接,各 TU 自有正确值,仅常量折叠入模板参数不 odr-use)。

## 参考

- conventions §10 两种运行期 dispatch 模型（fn-ptr 表 vs 变体 DLL）。
- Highway `HWY_DYNAMIC_DISPATCH`（fn-ptr 表范本）、llama.cpp `GGML_BACKEND_DL`（变体 DLL 范本）。
