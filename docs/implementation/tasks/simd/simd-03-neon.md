# simd-03：ARM64 NEON + AVX-512 后端（均已执行验证）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | ✅ 完成。NEON 全路径已在 Apple Silicon M4 Air 真机验证；AVX512 后端及 ops/dtype/math 派发已接线，Intel SDE `vec_test` 169/169 并确认运行时选择 kAvx512。SDE 仅作正确性与派发证据，性能仍待 AVX512 真机 benchmark |

## 前置依赖

- 硬前置：`simd-01`（vec D-tag 契约）+ `simd-02`（后端结构 + op 契约模板）。
- 决策依据：[`simd-design.md`](../../../architecture/simd-design.md) 落地清单 #2 与 AVX512 执行边界；`reference_wsl_linux_tsan.md`（aarch64 交叉编译验证法）。

## 目的

补齐 aarch64 NEON 与 x86 AVX512 后端，让同一份 D-tag 源码覆盖 128 位 ARM 向量和 512 位 x86 向量；AVX512 使用后端自有 k-register mask，不改公开 op 签名。

## 具体事务

- [x] `include/hyle/simd/backend/neon.h`：128 位 NEON 后端，lane f32/f64/i32/u32，op 全集匹配 scalar/sse4/avx2 契约。intrinsic 对照 Highway arm_neon-inl.h：`vdupq/vld1q/vst1q`、`vaddq/vmulq/vdivq(aarch64)/vminq/vmaxq`、`vfmaq`(FMA)/`vfmsq`、`vbicq`(AndNot 需 swap 成 `~a&b`)、`vceqq/vcgtq/...`(返 mask 向量)、`vbslq`(IfThenElse)、`vaddvq/vminvq/vmaxvq`(横向 reduce)、mask pack `{1,2,4,8}+vaddvq`+popcnt/ctz。mask=全1/全0 unsigned 向量(同 SSE/AVX2 契约)。
- [x] vec.h 的 `HYLE_SIMD_TARGET_NEON` ladder hook 已就位（simd-01 留），无需改。
- [x] `include/hyle/simd/backend/avx512.h`：512 位 AVX-512 F+BW+VL+DQ 后端与 k-register mask。

## 验收标准

- **编译**：✅ `aarch64-linux-gnu-g++-14 -std=c++23 -O3 -c` 干净（零 error/warning）。**真实捕到 bug**：`typename RawNeonT<T>`（别名模板 id 前误加 typename）交叉编译即报错并修掉——证明交叉编译是真验证非走过场。
- **指令验证（objdump）**：✅ `aarch64-linux-gnu-objdump -d` 确认真发 NEON：dot kernel → `fmla v31.4s`(FMA)+`faddp v31.4s`(横向和)；relu(Gt+IfThenElse) → `fcmgt v30.4s,#0.0`；int sum → `addv s31,v30.4s`(横向整数和)，全在 `v` 寄存器。
- **✅ 真机已跑（2026-07-05，Apple-Silicon M4 Air / macOS / Apple clang 21）**：NEON 结构与 SSE/AVX2 同构、vec_test 宽度无关的推断被真机证实——`hyle_simd_test` **49/49 全绿**(vec 对拍 + dot/saxpy NEON kernel)，派发诊断后来在 simd-22 统一为 `runtime().ops.isa==kNeon`；`hyle_container_test` **158/158**(SwissTable GroupNeon 真跑)。**双平台无回归**:同批 CMake 改动下 WSL x86 simd 56/56 + container 160/160。
- **真机暴露的真缺口(交叉编译从没碰到)**：NEON 此前只**单编 neon.h 一个头**,从没走完整 CMake 构建——真机一编发现 **simd/container 的 CMake 是 x86 写死**(无条件编 x86 TU + 无条件 `-mavx2`,arm64 clang 报 "unsupported option")。补:①`hyle_isa_flags` 加 NEON tier(ARMv8 baseline 空 flag);②simd/container CMakeLists 按 `CMAKE_SYSTEM_PROCESSOR` 分支(x86 编 x86 TU、arm64 编 neon TU/走 baseline);③`kernels_neon.cc` 薄 TU(复用 neon.h)+ `kernels.h`/`dispatch_ops.cc` 加 `HYLE_ARCH_ARM64`→kNeon 分支。另修一个 macOS 真 bug:`pthread_t` 是 opaque 指针(glibc 是整数),`thread.cc` 的 static_cast 非法→改 memcpy。**教训:"交叉编译过"≠"该架构能构建"——只编了一个后端头,不是编了模块。**
- **✅ dtype + math NEON 已补全(2026-07-05,真机)**：dtype fp16↔f32 用 Apple Silicon 原生 ARMv8.2 `vcvt`（finite 位精确、NaN by-class），bf16↔f32 用 NEON 整数位运算；math exp/log/tanh 借 Highway contrib/math 核并对拍 libm。AVX512 后续由 Intel SDE 通过正确性与派发门；两者的性能结论均以对应真机 benchmark 为准。

## AVX-512 后端（已实现并由 Intel SDE 执行验证）

> AVX512 使用后端自有 `Mask512` 包装 k-register；公开 D-tag/op 签名不变。Intel SDE 负责无真机时的执行正确性与 dispatch 闸门，性能数据仍必须来自真实 AVX512 硬件。

- [x] `include/hyle/simd/backend/avx512.h`：512 位 AVX-512(F+BW+VL+DQ),lane f32/f64/i32/u32(16/8/16 lane),op 全集。**自定义 `Mask512` 包 `__mmask16/8`（k 寄存器）**——比较直接写 k(`_mm512_cmp_*_mask`/`vpcmp*`)、`IfThenElse`→`_mm512_mask_blend_*`、`CountTrue`→k 本就是位掩码 popcnt。unsigned 比较有原生 epu32 mask(无需 SSE/AVX2 的 xor-MSB)。
- **验证**：`g++-14 -mavx512f -mavx512bw -mavx512vl -mavx512dq -c` 干净；objdump 实证真发 zmm/k-reg；Intel SDE `vec_test` 169/169，并由 tier 探针确认 dispatch 选择 kAvx512。
- **设计结论(重要)**：写 AVX512 **证明了 vec.h 的 per-backend mask 抽象够用**——`Mask` 本就每后端自定义(scalar=bool[]/SSE=向量/AVX512=k-reg),op 签名不变,vec.h 和测试零改动。先前"需 mask 改造"是误判。
- **执行边界**：AVX512 后端及 ops/dtype/math 运行时派发均已启用，具备完整 ISA/OS-state 的主机会选择 kAvx512；Intel SDE 已验证正确性与 tier 选择。SDE 不提供可信性能时序，性能仍待真机 benchmark。
- **边界对拍**：`Ne(float/double)` 使用 `_CMP_NEQ_UQ` 保持 NaN 无序语义，`Neg(float/double)` 用 xor 符号位保持 `Neg(+0.0)=-0.0`；这些行为现已包含在 Intel SDE 执行对拍中。

## 参考

- Highway `hwy/ops/arm_neon-inl.h`（intrinsic 逐个对照）。
- `container/probe.h` GroupNeon（mask pack 折位 idiom 来源 + "写+编+指令验未跑"先例）。
- simd-design 落地清单 + AVX512 执行边界。
