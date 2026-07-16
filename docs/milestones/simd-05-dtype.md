# simd-05：dtype fp16 / bf16 <-> fp32 转换（AI 数值刚需）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | 完成 |

## 前置依赖

- 硬前置：`simd-02`（F16C 走 AVX 路径 + 变体 TU 模型）+ `simd-01`（dispatch 探测 cpu_features().f16c）。
- 决策依据：[`simd-design.md`](../architecture/simd-design.md) dtype 层（"bf16 尤其重要——AI 训练/推理主力，clay 完全没有"）；[`coding-conventions.md` §10](../architecture/coding-conventions.md)（能力 dispatch + 探测器隔离）。

## 目的（为什么做这一步）

AI infra 数值刚需：fp16/bf16 是混合精度训练/推理 + KV-cache 的主力存储格式。clay 的 fp16 是 6923 行纯软件查表、**无 bf16**。这步落地两种 dtype 的存储类型 + fp32 双向转换：fp16 走 **F16C 硬件指令**（本机可跑可测），bf16 走移位（promote=高 16 位，demote=RNE）。

## 在架构中的位置

`simd/dtype/` 子层。fp16 批量转换沿用 ops/ 的 fn-ptr 派发 + 变体 TU 模型（scalar 兜底 + F16C 硬件），bf16 标量核（无 AVX2-tier 硬件，AVX-512-BF16 是后续）。

## 具体事务（做什么）

- [x] `dtype.h`：`float16_t`/`bfloat16_t`（uint16 背衬，像 Highway wrapper）+ 标量转换核 `f16_to_f32`/`f32_to_f16`/`bf16_to_f32`/`f32_to_bf16`，**逐行照搬 Highway base.h**（F32FromF16/F16FromF32/F32FromBF16/BF16FromF32）——处理 subnormal/inf/nan/RNE，非查表非手推。
- [x] `dtype_ops.h` + `dtype/`：批量转换派发 API（`f16_to_f32_batch` 等；当前层由统一 `runtime().dtype` 报告）。fp16 走 F16C 变体 TU（`cvt_f16c.cc`，`_mm256_cvtph_ps`/`_mm256_cvtps_ph` + `_MM_FROUND_NO_EXC`）；scalar 兜底 TU；baseline dispatcher 按 `f16c && avx` 门控。
- [x] CMake：F16C TU 编 `-mf16c -mavx`（**非 avx2**，见下安全修复）。
- [x] `tests/simd/dtype_test.cc`：已知位模式 + 全 65536 fp16 往返 + RNE tie-to-even oracle + 全 65536 硬件-vs-标量对拍 + tail + bf16 RNE/NaN。
- [x] **明确不做**：AVX-512-FP16 原生算术、AVX-512-BF16 `vcvtne2ps`、int8/fp8（后续）；bf16 显式 AVX2 8-wide kernel（deferred，标量 auto-vec 4-wide 兜着）。

## 交付物

- 2 公开头（dtype.h/dtype_ops.h）+ 3 内部 .cc（scalar/f16c/dispatch）+ kernels.h + 测试。cloc：生产 187、测试 151。

## 验收标准

- **编译**：✅ Win MSVC 19.50（/W4 零告警，C4556 false-positive 局部抑制如 Highway）+ Linux GCC14.2（零告警）。
- **正确性**：✅ 双平台 `hyle_simd_test` 47/47（含 dtype 9）。**全 65536 fp16 位模式硬件(F16C `vcvtph2ps`)-vs-标量逐位对拍**（finite 位等 / NaN 按类）+ 全 65536 fp16→f32→fp16 往返 + RNE tie-to-even hand-oracle。`runtime().dtype` 本机报告 kAvx2 + F16C。全量 ctest 13/13 无回归。
- **对照 Highway**：标量转换逐行对照 base.h；F16C intrinsic `_MM_FROUND_NO_EXC` 对照（Highway 同此立 immediate=8，C4556 同款抑制）。

## 对抗式自审（红队揪出 2 个本机测不到的 SIGILL 真 bug）

闭环后派红队审计（dtype 是稠密位运算，正是 bug 藏身处）。**标量数学逐行核对无误**（RNE/subnormal/inf/nan/snan-guard 全对）。但揪出 **2 个本机(AVX2 机)测不到的 SIGILL 真 bug**——正是"代码分析验证 ≠ 老硬件实跑"的缺口：

1. **F16C 内核用 256 位 YMM(`_mm256_cvtph_ps`)但 dispatch 只门控裸 CPUID f16c 位、缺 OS-YMM 门**（不像 `f.avx`/`f.avx2` 有 os_ymm）。stale OS 不存 YMM → 状态损坏。**修**：门控改 `f16c && avx`（avx 已 XGETBV os_ymm 门控）。
2. **F16C TU 编 `-mavx2`，但 F16C 在 Ivy Bridge 无 AVX2**。f16c-only CPU 上 dispatch 进该 TU，自动向量化的标量 tail 可能发 AVX2 整数指令 → Ivy Bridge SIGILL。**修**：TU 降到 `-mf16c -mavx`（F16C+AVX = Ivy Bridge 级,封顶不发 AVX2）。

**附带修测试漏洞（红队指认诚实）**：① 原"穷举往返"只喂精确可表示值,RNE 路径从未触发 → 加 RNE tie-to-even hand-oracle(标量路径,每机跑非 tautology);② 非 F16C 机上硬件对拍退化为 scalar-vs-scalar tautology → 加 `GTEST_SKIP` 诚实跳过;③ bf16 "无需 AVX2 路径" 措辞修正为"deferred"(标量 auto-vec 4-wide 兜,显式 8-wide 是真未来工作)。**待补真验**：Ivy Bridge / QEMU `-cpu` 限 F16C-no-AVX2 实跑确认不 SIGILL（登记，本机无此硬件）。

## 参考

- Highway `base.h`：F32FromF16/F16FromF32/F32FromBF16/BF16FromF32（标量核逐行对照）、`x86_128/256-inl.h` F16C DemoteTo（`_MM_FROUND_NO_EXC` + C4556 抑制）。
- simd-design dtype 层表。
