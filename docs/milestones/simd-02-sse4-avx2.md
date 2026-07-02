# simd-02：SSE4.2 + AVX2 实后端（首个 intrinsic 落地）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 2-3k |
| **骨架?** | 否（simd-01 已立脊，这是第一根真 intrinsic 后端） |
| **状态** | 完成 |

## 前置依赖

- 硬前置：`simd-01`（vec.h D-tag 契约 + scalar 地板 + dispatch 探测骨架 + backend ladder）。
- 决策依据：[`simd-design.md`](../architecture/simd-design.md) 落地清单 #2；[`performance/benchmarks-simd.md`](../performance/benchmarks-simd.md)（SIMD vs scalar 4-16x、SOTA 标尺=Highway）；[`references.md`](../architecture/references.md)（Highway=参考不 vendor）。

## 目的（为什么做这一步）

simd-01 只有 scalar 地板（能跑、不快）。这步落第一批**真 intrinsic 后端**——SSE4.2(128 位) + AVX2(256 位)——让 `vec.h` 的 op 契约真正吃到向量硬件。这是 SIMD 支柱"4-16x over scalar"价值的兑现点，也是对标 Highway（SOTA）的第一次实测。

## 在架构中的位置

`simd/` 子树：simd-01 的 vec.h 定契约，本里程碑填两个 x86 后端。走 `HYLE_SIMD_TARGET_*` ladder（一后端一 TU，变体-DLL 模型）——同一份 kernel/test 源码按宏 + `-march` 选后端，backends 永不在一个 TU 共存（无符号冲突）。NEON/AVX512 → simd-03，运行期变体选择 → simd-04。

## 具体事务（做什么）

- [x] `backend/sse4.h`：128 位 SSE4.2 后端。`Vec128<T>` 经 `Raw128` trait 选 `__m128/__m128d/__m128i`。op 全集（Set/Zero/Load[U]/Store[U]/Add/Sub/Mul/Div/Min/Max/Neg/Abs/Sqrt/MulAdd/NegMulAdd/And/Or/Xor/AndNot/Eq/Ne/Lt/Le/Gt/Ge/IfThenElse[Zero]/AllTrue/AllFalse/CountTrue/FindFirstTrue/ReduceSum/Min/Max/GetLane）。
- [x] `backend/avx2.h`：256 位 AVX2 后端，同结构 `_mm256_*`。真 FMA3、float 比较谓词、reduction 跨 128 位。
- [x] Lane 类型 **f32/f64/i32/u32**（数值热路径集）。**native 全宽**（SSE 16B / AVX2 32B）。
- [x] `vec_test.cc` 改宽度无关（ScalableTag + 动态 Lanes + 朴素参考）；`tests/CMakeLists` 编 3 份（scalar/sse4/avx2，各配 `-march`/`/arch`）；`HyleModule.cmake` 的 `hyle_add_test` 加 `NAME/DEFINES/OPTIONS`。
- [x] `bench/simd/`：dot + saxpy，scalar-autovec / HYLE sse4 / HYLE avx2 / Highway(SOTA，gated)。
- [x] scalar.h 补 `Sqrt`（simd-01 漏，对齐契约）。
- [x] **明确不做**：i8/u8/i16/u16（→ simd-05 dtype/量化）；i64/u64（mul/min/max 需 AVX-512 或繁琐 partial-product fallback，待消费者需要）；partial/Half 宽向量；NEON/AVX512（simd-03）；运行期变体 DLL 选择（simd-04）；数学核（simd-06 借 sleef）。

## 交付物

- 2 后端头（sse4.h 366 + avx2.h 364 ≈ 730）+ scalar.h Sqrt + 宽度无关 vec_test + 5 bench TU（161 行）+ CMake。
- cloc：后端 730、bench 161。每后端 ~365 行，**8x 低于 simd-design "<~2k 行/后端" 上限**——精瘦达标。

## 验收标准

- **编译**：✅ Win MSVC 19.50（/W4 零告警，scalar+sse4 默认 + avx2 `/arch:AVX2`）+ Linux GCC14.2 Release（零告警，`-msse4.2` / `-mavx2 -mfma -mf16c`）。
- **正确性（对拍 scalar）**：✅ 双平台三后端各 24/24 vec 测试，每 op 逐 lane 对拍朴素标量参考——含 AVX2 跨 128 位 reduction、u32 xor-MSB 无符号比较、blendv select、FMA（epsilon 容差）。全量 ctest 13/13（11→13，+sse4/avx2 变体，无回归）。
- **性能（对标 SOTA，WSL 原生 ext4，4096 f32 L1 常驻）**：
  | kernel | scalar auto-vec | HYLE sse4 | HYLE avx2 | Highway avx2(SOTA) |
  |---|---|---|---|---|
  | dot | 2.73 G op/s | 11.3 G (4.14x) | **12.5 G (4.58x)** | 10.7 G |
  | saxpy | 17.6 G | 16.5 G | **25.0 G (1.42x)** | 20.0 G |
  **结论：追平 Highway**（dot 名义 +17% 在微基准噪声内，内循环同编 `vfmadd231ps`）；4.6x over auto-vec（落 perf 文档区间）。Win MSVC 复测一致（4.56x/1.56x）。
- **对照 Highway 契约**：op 名/语义与 Highway 一致（AndNot=~a&b、IfThenElse blendv、ReduceSum shuffle cascade）；intrinsic 逐个对照 `x86_128-inl.h`/`x86_256-inl.h`（unsigned 比较 xor-MSB、AVX2 reduction `permute`/extract 跨 128、SSE4.2 tier 无 FMA3 故 sse4 用 mul+add fallback、avx2 用真 fmadd）。

## 对抗式自审（简化审查 + 边界诚实）

- **"能否手写更少"（用户问）**：sse4/avx2 各 ~365 行高度并行，但**故意保持显式**——这正是 Highway 自己的选择（x86_128-inl.h 与 x86_256-inl.h 分开各万行级，不宏合并）。宏合并 SSE+AVX 可省 ~300 行但：① 违 conventions §3 宏纪律（大代码生成宏正是其警告对象）；② 两后端真有差异（float 比较谓词 / FMA 真假 / reduction 跨 128）；③ 伤硬件代码可读/可调。**已做的清理**：AndNot 折进 And/Or/Xor 的 bitop 宏（同形，省冗余）。**该借不自写的**：transcendental 数学核（exp/log/tanh）登记 simd-06 vendor sleef，不手搓多项式。结论：per-backend 已近最简，共享逻辑已在 vec.h。
- **诚实边界**：lane 类型只 f32/f64/i32/u32；native 全宽（无 partial）；sse4 saxpy 0.94x（128 位带宽受限，无加速空间，非 bug，留账）。
- **批量红队补审（2026-06，整支柱批次）**：揪出 AVX2 `Ne(float/double)` 用有序谓词 `_CMP_NEQ_OQ`（NaN 返 false），与 scalar(`!=`)/SSE4(`_mm_cmpneq`,无序)/NEON(`~Eq`) 发散——后果 `Eq`/`Ne` 在 NaN 上非互补，可移植 NaN 检测 `IfThenElse(Ne(x,x),..)` 静默失效。改 `_CMP_NEQ_UQ`(无序)。原 `CheckCompareBlend` 只喂有限值看不到，已加 `FloatEdgeCases`(NaN 互补 + Neg 符号零) 覆盖，AVX2 真机过。

## 参考

- `performance.md` simd 实测表（本里程碑回填）。
- Highway `hwy/ops/x86_128-inl.h` / `x86_256-inl.h`（intrinsic 映射逐个核对）、`generic_ops-inl.h`（reduction cascade 模式）。
- `simd-design.md` 落地清单 #2、"<~2k 行/后端"上限。
