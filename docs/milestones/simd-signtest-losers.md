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

## 施工方针
1. 大输优先（内存族 LoadNOr/Compress/Gather、MultiRotateRight）——收益最大且明确。
2. 中输逐个对照 Highway 源码找多余步骤。
3. 小输（1.01–1.05×）先隔离单-op bench 确认是真输还是完整-bench 串扰，是串扰则不动（记录），是真输才修。
4. 每项：WSL sign 判方向 → 四环境正确性 → 三平台 sign 一致 → 提交。
