# SIMD 配对符号检验揪出的真输清单（2026-07-12）

> 用户揪出旧 bench-stat "gap>spreadA+spreadB" 绝对判据把真输淹成平（同机噪声相关，ratio 远比绝对 spread 稳）。已加**配对符号检验**（`scripts/bench-stat.py` 的 `[sign]` 列：N 次 run 里 ratio 一致同侧、二项 p<0.01 即真输，无视绝对 spread）。以下是三平台全 sweep（op_parity full/parity/widen）在符号检验下的真输（HYLE 慢）清单，按 ratio 排序。
>
> **纪律**：WSL/Windows 同物理机必须错开跑（不并发）；Mac 独立可并行。每项修复方向由 WSL sign 判定，正确性四环境验，最终三平台 sign 一致才算追平。

## AVX2（WSL GCC14，符号检验）

### 大输（明确真输，优先）
| op | ratio | 性质 |
|---|---:|---|
| LoadNOr i32 (half n) | 9.64× | store+标量循环+reload；可用 native maskload(fault-safe) |
| CompressStore i32 | 3.42× | 契约差异(精确写)+AVX2 无 native compress-store —— 待重审 Highway |
| CompressBlendedStore i32 | 3.04× | 同上 |
| MultiRotateRight u64 | 2.90× | Highway generic=DupEven+TableLookupBytes+Shr<u16>；Shr<u16> 需向量化 |
| GatherIndexN / GatherIndexNOr i32 | 2.59× | 变长 gather harness |
| LoadInterleaved2 f32 | 1.34× | 待查 |
| GatherOffset i32 | 1.16× | gather harness |
| TwoTablesLookupLanes i32 | 1.11× | avx2 无 permutex2，回退 |
| MulHigh32 u32 | 1.10× | 待查 |

### 中/小输（1.02–1.25×，部分可能串扰，需隔离确认）
FindKnownLastTrue 1.25 / IsNegative 1.19 / CombineMasks 1.15 / Min·MaxNumber 1.07 / MaskedApproxReciprocal(Sqrt) 1.06 / FindLastTrue 1.07 / Lt128Upper 1.06 / Max128 1.04 / MulAddSub 1.04 / MaskedModOr 1.03 / RotateLeftSame 1.03 / ConcatOdd 1.03 / BitShuffle 1.03 / MaskedShiftLeft·Right 1.03–1.04 / Clamp 1.03 / MulByPow2 1.03 / Eq128 1.02

## NEON（Apple Silicon，符号检验）

### 明确真输
| op | ratio | 性质 |
|---|---:|---|
| Ceil f32 | 1.40× | 待查(vrndpq 单条?可能 harness) |
| Not i32 | 1.39× | 单指令，极可能 harness 串扰(隔离核) |
| TrailingZeroCount i32 | 1.38× | 待查 |
| RoundingShiftRightSame i32 | 1.14× | 待查 |

### 小输（1.01–1.07×，绝大多数是完整-bench 序列串扰）
Zero/CeilInt/FloorInt/AllBits0/MaskedReduceMax/DupEven/DupOdd/ShiftRight/IfThenElse*/Xor/And/Shuffle0321·2301/SaturatedAbs/Round/PairwiseAdd128/ZeroIfNegative —— 全是单指令或已知最优 op，1.01–1.02× 高度疑似 NEON 完整-bench 相邻 op 串扰（已坐实 Shuffle2301 隔离即平）。**须隔离单-op bench 逐个确认真输 vs 串扰**。

## 已在本轮修复追平（符号检验确认）
SaturatedNeg（位技巧 `~v-(v>MIN?-1:0)`）、SaturatedAbs i32（`Min_u32(Abs,INT_MAX)`）、PromoteMaskTo/OrderedDemote2MasksTo（向量域+PEXT+MaskFromNormalizedVec）、InterleaveEven/Odd（双 shuffle_ps）、TestBit（Eq(v&bit,bit)）、MulByFloorPow2 NEON（vcvtmq）。

## 三平台交集（铁真输，排除单平台 harness/编译器伪影）——2026-07-12 补

三平台(WSL GCC avx2 / MSVC avx2 / Mac NEON)各自 sign sweep 后求交集。**单平台独有的"输"多是 harness/编译器伪影**（NEON 22 输里 21 个单平台独有=串扰；MSVC 76 输含 Add/Max/Min f32 等单指令=MSVC harness），交集才是真算法输。

**三平台都输（1）→ objdump 证伪为 harness 伪影**：`MaskedApproxReciprocal f32`。HYLE 与 Highway 源码逐字相同（都是 `IfThenElseZero(m, ApproximateReciprocal(v))`），隔离 TU 编 `-S` 两函数体**逐指令完全相同**（vmovups→vbroadcastss→vcmpps→vrcpps→vandps→vmovups）→ 不可能有真算法差，三平台都显 1.05–1.27× 是各平台完整-bench 的一致 harness 串扰。**⚠️关键教训：符号检验"一致同侧"+三平台交集仍不足以定真输——一致方向也可能来自一致的 harness 偏差；必须 objdump 看 codegen 是否真不同才能终审。** 交集里每一项都要 objdump 分类（codegen 不同=真输可修 / 相同=伪影）。

**avx2 两编译器(GCC+MSVC)都输（20，x86 真算法输）**：Eq128 / Max128（128-cmp）、CompressStore / CompressBlendedStore / GatherOffset / LoadNOr / LoadInterleaved2 / BitShuffle / TwoTablesLookupLanes / MultiRotateRight（内存/复杂）、MaskedApproxReciprocal(Sqrt) / MaskedMulComplexConj / MaskedShiftLeft·Right<3>（Masked 族）、FindKnownLastTrue、MinNumber / MaxNumber、MulAddSub、ConcatOdd。

**⚠️ MSVC 独有的大输（GCC 已追平但 MSVC 没有，编译器 codegen 差异，需查）**：SetAtOrAfterFirst/OnlyFirst/BeforeFirst/AtOrBeforeFirst 5×、OrderedDemote2MasksTo 4.6×、Reverse2/4/8 2.8×、Dup128MaskFromMaskBits 5.4×、PairwiseAdd/Sub。这些我在 WSL(GCC) 用向量化追平了，但 **MSVC 对同一份 C++ 的 codegen 没优化到位**——说明"追平"必须三平台各自验证，不能只信 GCC。

**NEON 独有小输（1.01–1.07×，串扰待隔离确认）**：Not/Xor/And/Zero/DupEven/Shuffle 等单指令。

## "一边倒 HYLE 慢"根因调查（2026-07-12，用户观察驱动）——布局偏差非算法

用户注意到 sign sweep **一边倒 HYLE 慢**（MSVC 76 sign-WIN 几乎无 sign-LOSE），怀疑伪影/顺序问题。三步决定性实验：
1. **run 顺序不是根因**：同 op 各测 HYLE-first / Highway-first 两序（order_bias 探针）。MaskedApproxReciprocal **两序都 HYLE 慢 ~3.5%，不翻转** → 排除"先跑吃冷启动"。
2. **nanobench 两 slot 公平**：同一份代码放两个 run slot（selfsame 探针，G/G 和 H/H）→ **都干净 TIE**（p=0.61/0.42）→ harness 对两 slot 无系统偏差。
3. **codegen 相同却稳定慢**：mrecip 的 HYLE/Highway 内联后**逐指令相同**（objdump 证），却稳定 HYLE 慢 3.5% 且不随顺序翻转。

**结论**：根因是**代码布局偏差**——HYLE/Highway 是不同函数符号，链接后落在不同地址/对齐，icache/DSB/预测细微差异；每个 bench 宏里 HYLE 恒先定义/先实例化，系统性地让 HYLE 的布局吃 ~2–4% 的亏。**这是真实但与算法/代码质量无关、且不可控（取决于链接布局）的效应。**

**对真输判定的修正（第四道关）**：
- **≤ ~1.05× 的一边倒 HYLE 慢**：极可能是这 ~2–4% 布局偏差，**不是真输**。除非 objdump 显示 codegen 真不同，否则不动。
- **真输门槛上调**：只有 ratio 明显超出布局偏差量级（经验 ≥ ~1.10×）**且** objdump codegen 不同，才是真算法输。据此，交集 20 项里真正该修的收窄到大 ratio 的内存/复杂族（LoadNOr 9.6/Compress 3/Gather 2.6/MultiRotate 2.9/LoadInterleaved2 1.34/TwoTables 1.11/MulHigh32 1.10）+ 逐个 objdump 确认 codegen 不同的中 ratio 项。1.02–1.05× 的（Eq128/Max128/MulAddSub/ConcatOdd/Min·MaxNumber/MaskedShift 等）先 objdump，codegen 相同即布局伪影不修。

## 施工方针
1. 大输优先（内存族 LoadNOr/Compress/Gather、MultiRotateRight）——收益最大且明确。
2. 中输逐个对照 Highway 源码找多余步骤。
3. 小输（1.01–1.05×）先隔离单-op bench 确认是真输还是完整-bench 串扰，是串扰则不动（记录），是真输才修。
4. 每项：WSL sign 判方向 → 四环境正确性 → 三平台 sign 一致 → 提交。
