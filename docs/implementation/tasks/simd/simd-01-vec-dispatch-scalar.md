# simd-01：vec D-tag 抽象 + scalar 后端 + 运行期 CPUID dispatch 骨架

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **Tier** | 1 |
| **LOC 估** | 2-3k |
| **骨架?** | 是（simd 支柱第一刀，纵切立脊） |
| **状态** | 完成 |

## 前置依赖

- 硬前置：`mem-01`（对齐契约——load/store 要求 64B 对齐；本里程碑 load/store 暂用裸指针，对齐消费留 ops/ 里程碑）。
- 决策依据：[`simd-design.md`](../../../architecture/simd-design.md)（落地清单 #1/#2、待决 #1 已决 scalable-friendly 接口+定宽后端）、[`coding-conventions.md` §10](../../../developer/standards/coding-conventions.md)（能力 dispatch 三层模型 + 探测器隔离铁律 + 前置结论"硬件特性代码不全用编译选项"）、[`roadmap.md`](../../../architecture/roadmap.md)（simd 决策"用户拍板"行）。

## 目的（为什么做这一步）

simd 支柱开张第一刀。clay 这块近乎空白（`simd.h` 仅 43 行编译期探测，无 vector 类型/无 AVX/无运行时 dispatch）。这步立两根桩：① **运行期 CPUID dispatch 骨架**——用户铁律"simd 一开张就搭 `dispatch.h`，不要只堆 `#ifdef`"，因为 HYLE 分发的是 C-ABI `.so`，`-march=native` 编出的 `.so` 在老 CPU 上会 SIGILL，必须运行期选 ISA；② **可移植 vec D-tag 抽象 + scalar 地板后端**——照搬 Highway 接口契约，给后续各 ISA 后端定下统一的 op 形状与对拍参照。

## 在架构中的位置

依赖图 `{container, simd} → sched` 一层，simd 只依赖 mem（对齐契约），**不在通往 sched 的关键路径上**，可与 container 并行。本里程碑是 simd 子树的根：vec.h 的 op 契约被后续 SSE2/AVX2/AVX512/NEON 后端实现，dispatch.h 的探测被 simd-04 的变体 DLL 选择器消费。

## 具体事务（做什么）

- [x] `include/hyle/simd/dispatch.h` + `src/simd/dispatch.cc`：`CpuFeatures`（逐特性位 SSE2..AVX512BF16/FP16/VNNI + LZCNT/BMI + NEON/SVE）、`Isa` 打分枚举（按 arch gap-spaced）、`detect()`（CPUID leaf 1 / 7,0 / 7,1 / 0x80000001(LZCNT) + OSXSAVE 后 XGETBV 查 XCR0，防 OS 没存 YMM/ZMM 误判）、`best_isa()` 打分（tier 真·对标 Highway group：AVX2 tier 需 avx2+fma+f16c+lzcnt+bmi1+bmi2+sse42=kGroupAVX2，AVX512 tier 在其上叠 F+BW+VL+DQ+CD 并 subsume AVX2 group，故"高 tier 蕴含低 tier"是真不变量）、`supports()`/`isa_name()`。Meyers 单例缓存。MSVC(`__cpuidex`/`_xgetbv`)+GCC(`__cpuid_count`/asm `.byte` xgetbv) 双 ABI。
- [x] `include/hyle/simd/vec.h`：照搬 Highway D-tag 契约——`Simd<T,N,kPow2>` 零大小标签、`TFromD`/`VFromD<D>=decltype(Zero(D()))`、`MaxLanes`/`Lanes(d)`、`Half`/`Twice`、`ScalableTag`/`CappedTag`/`FixedTag` 别名。`Lanes(d)` 经 `HYLE_SIMD_HAVE_SCALABLE` 宏设计成可运行期求值（v1 定宽=constexpr，将来 SVE 翻宏+后端供运行期 Lanes，调用点不改）。`kNativeVectorBytes` 按 baseline ISA 选（构建期层）。
- [x] `include/hyle/simd/backend/scalar.h`：逐 lane 朴素 C++ 实现核心 op 全集——Set/Zero/Load[U]/Store[U]/Add/Sub/Mul/Div/Min/Max/Neg/Abs/MulAdd/NegMulAdd/And/Or/Xor/AndNot/Eq/Ne/Lt/Le/Gt/Ge/IfThenElse[Zero]/AllTrue/AllFalse/CountTrue/FindFirstTrue/ReduceSum/Min/Max/GetLane。作所有 SIMD 后端的对拍地板。
- [x] `src/simd/CMakeLists.txt`：`hyle_add_module(simd ...)` + `INTERPROCEDURAL_OPTIMIZATION OFF`（探测器隔离）。接入顶层 + tests。
- [x] **明确不做**（留后续）：SSE2/AVX2/AVX512/NEON 实后端（simd-02/03）；运行期变体 DLL 编译 + dlopen 选择（simd-04，本里程碑只做"探测"半，"选择执行"半留下）；dtype fp16/bf16（simd-05）；sleef 数学核（simd-06）；ops/（对齐 memcpy/SwissTable 探测复用）。

## 交付物

- 3 头（dispatch.h/vec.h/scalar.h）+ 1 实现（dispatch.cc）+ 2 测试（dispatch_test/vec_test）+ CMake。
- cloc 实测：生产 496（头 337 + dispatch.cc 159）+ 测试 210 = 706 行。骨架精瘦合理：scalar 后端逐 lane 朴素实现密度低，真重活（各 ISA intrinsic 后端）在 simd-02/03。

## 验收标准

- **编译**：✅ Win MSVC 19.50（/W4 零告警）+ Linux GCC14.2 Release（-Wall -Wextra -Wpedantic 零告警）。
- **测试**：✅ 双平台各 22/22。dispatch（探测不崩/缓存稳定/x64 必有 SSE2/tier 蕴含特性组/AVX512 扩展蕴含基/supports 单调）+ vec（D-tag 契约 Lanes/MaxLanes/tags + scalar 每 op 对拍朴素参考 + ScalableTag 写法的 dot product 与宽度无关）。
- **探测器隔离**：✅ dispatch.cc 独立 TU + `INTERPROCEDURAL_OPTIMIZATION OFF`；探测在两平台真实 CPU 跑过不 SIGILL（DetectDoesNotCrash 过）。
- **全量无回归**：✅ Win 全量 ctest 11/11 目标过（10→11，simd 为净增，测试数上升证非假 PASS）。
- **对照 Highway 契约**：D-tag 接口形状（Simd/TFromD/VFromD/Lanes/Half/Twice/三 tag 别名）与 Highway `shared-inl.h` 一致；CPUID+XGETBV 探测逻辑对照 `targets.cc` FlagsFromCPUID/ReadXCR0。**只照搬契约不 vendor**：vec.h+scalar.h 共 ~340 行 << Highway 单后端 ~3k 行，守住"不 vendor 的理由"。

## 对抗式自审（3 红队 agent 并行审计 + 独立 oracle 核账，2026-06-29）

闭环后派 3 个独立 agent 挑刺（dispatch 正确性 / vec-scalar 契约 / 构建+探测 oracle 核账）。**核账铁证**：WSL 上写独立 oracle 程序，把 `cpu_features()` 每个 bool 与 `__builtin_cpu_supports` + `/proc/cpuinfo` 逐项对拍——**零 silent false positive**（无"HYLE 报有、硬件没有"），best_isa=avx2 与硬件吻合，过 `-Werror`。CPUID 每个 bit 位对照 Highway `FlagsFromCPUID` 逐位核对无误，XGETBV 门控正确（`read_xcr0` 仅在 OSXSAVE 后调，YMM=bit1&2 / ZMM=bit5,6,7）。

**审计揪出并已修的硬伤（本轮，非略过）**：
1. **【文档撒谎→改成真】** 里程碑文档原称 detect() 覆盖 `0x80000001`，但代码从未调该 leaf（Highway 用它取 LZCNT）。**修法不是删claim而是补实现**：真加 `0x80000001` 探测 LZCNT，并把 LZCNT+BMI1+BMI2 折进 AVX2 tier，使其**真·对标** Highway kGroupAVX2（原注释"mirror kGroupAVX2"是假的，只查 avx2+fma+f16c，材料性偏松）。
2. **【tier 不变量没坐实】** 原 kAvx512 tier 不要求 AVX2 base，"高 tier 蕴含低 tier"形同虚设。改为 AVX512 group 在 AVX2 group 之上叠加，真 subsume。测试同步加严（TierImplies 检全组 + AVX512 蕴含 AVX2）。
3. **【死代码】** `f.sse2 = bit(edx1,26) || f.sse2`（f.sse2 已 true，RHS 恒真）删除；`edx1` 随之删。
4. **【过度承诺的注释→诚实化】** `kNativeVectorBytes` 的 `HYLE_SIMD_TARGET_*` ladder 当前全是死分支（宏无处定义、宽度恒 16）。注释原称"-march=native build 自动加宽"似已接线。改为明示：simd-01 只接 scalar(#else)，非 scalar 分支是 simd-02/03 激活的前向接缝，**故意 inert 而非接错**。backend include 也改成同款 ladder（scalar 在 #else），simd-02 加 `#elif` 即可、无需重构（红队点出的 simd-02 前置）。

**审计确认solid（非虚报）**：探测 bit 精确、OS 门控正确、`Lanes()` 的 SVE constexpr 接缝是真接缝、op 正确性（AndNot 序、bit_binop 类型选择、MulAdd 双舍入注释+epsilon 测试）干净、`DFromV` 硬编码 kPow2=0 与 Highway emu128 一致（红队误判为偏离,实为对齐参考）。

**诚实边界（未做，非缺陷）**：这是 dispatch 的**探测半 + 选择接缝**，不是工作的选择器——变体 DLL 编译 + load 时 score + dlopen 选优是 simd-04；今天除测试外**无生产代码消费 best_isa()**（选择半留下）。ARM SVE 仅留接口未探测（无 ARM 机）。LTO-off 探测隔离今天无全局 LTO 可防，是防御将来的前向保险。

## 参考

- `simd-design.md` 落地清单 #1/#2、dispatch 三层表、待决 #1（已决）。
- `coding-conventions.md` §10 能力 dispatch + 探测器隔离 + 前置结论。
- Highway：`hwy/ops/shared-inl.h`（Simd D-tag + Lanes + tag 别名契约）、`hwy/ops/emu128-inl.h`（定宽标量后端 Vec128/Set/Zero/VFromD 形状）、`hwy/targets.cc:185` FlagsFromCPUID + `:121` ReadXCR0、`hwy/x86_cpuid.h`（Cpuid 双 ABI 包装）。
