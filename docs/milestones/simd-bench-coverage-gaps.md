# SIMD test/benchmark 覆盖缺口清单（grok 批量施工蓝图）

> **目的**：op 面 ~228/后端且正确性测试基本完整（vec_test 220 TEST），但 op_parity 性能背书只有 ~65 行——大量 op 有正确性测试却无"对拍 Highway/xsimd 是否 SOTA"的性能证据。这份清单是 grok 批量补 bench 的施工单。2026-07-12 用 general-purpose agent 逐 bench 文件 diff op 面建立并 spot-verify。
>
> **方法纪律（必守）**：① grep op 名有假阴（后端宏定义），标"无 bench"前必 grep 实际 bench 文件确认；② 每 bench 行必须公平对拍（objdump 验对照没静默降 128b、用库的生产实现非教学写法、两边同等内联；见 `benchmarks-simd.md` 顶部两条承重前置）；③ 统计法判定（`scripts/bench-stat.py -n15`，gap>两变体 spread 和才判赢/输，否则平局）；④ 纯直通单指令 op（add/and/min）不必补 parity（保证平局、低价值），已抽样 Add/MulAdd/Min-i32 代表该类足够。

## 已有 parity 覆盖（~65 行，勿重复）

- **op_parity_bench.cc（20）**：Add(+xsimd)、MulAdd、ReduceSum(+xsimd)、Reverse、Min(i32)、PopulationCount、Compress(+xsimd)、GatherIndex、SaturatedAdd(i16)、TableLookupBytes、LeadingZeroCount、Expand、ConcatEven、InterleaveLower(+xsimd)、ReverseLaneBytes、PromoteLowerTo、DemoteToNearestInt(+InRange)、DemoteMaskTo、LoadInterleaved3/StoreInterleaved4、LoadInterleaved4(u8)。
- **op_parity_full_bench.cc（~38）**：Abs/Neg/BroadcastSignBit/DupEven/DupOdd/ReverseBits/TrailingZeroCount/ShiftLeft<>/ShiftRight<>/Sub/Mul/Max/AbsDiff/OddEven/InterleaveUpper/ConcatOdd/Shl/Mod/MaskedModOr/MulEven/MulOdd/Sqrt/ApproxRecip(+Sqrt)/Eq/Lt/Gt/Masked{Shl,ShlOr,Shr,ShrOr,ShiftLeft/Or,ShiftRight/Or,ReduceSum/Min/Max,ApproxRecip/Sqrt,LeadingZeroCount,ConvertTo,MulFixedPoint15,WidenMulPairwiseAdd,SatAddOr,SatSubOr,MulComplex/Or/Conj/Add/ConjAdd}。
- **op_parity_widen_bench.cc（7）**：PromoteUpperTo/DemoteTo(2→1 sat)/ReorderDemote2To/WidenMulPairwiseAdd/MulFixedPoint15/SumsOf8/GetExponent。

## 施工优先级（grok 按此顺序）

### ⭐ Tier 1 — 高价值（真算法内容 + Highway/xsimd 有同 op + 零 bench）

新建 `bench/simd/crypto_bench.cc`（唯一连 vs-scalar 都没有的域）：
- **Crypto**（对照 Highway）：`AESRound`/`AESLastRound`/`AESRoundInv`/`AESLastRoundInv`/`AESInvMixColumns`/`AESKeyGenAssist`/`CLMulLower`/`CLMulUpper`。HYLE 有 soft+x86+NEON 三套,codegen 可能与 Highway 异,最高优先。

补进 `op_parity_bench.cc` 或 `op_parity_full_bench.cc`：
- **标量 complex**（对照 Highway MulComplex 族）：裸 `MulComplex`/`MulComplexConj`/`MulComplexAdd`/`MulComplexConjAdd`/`ComplexConj`（现在只有 Masked 包装被测,裸的热形式没测）。
- **128-cmp**（对照 Highway）：`Lt128`/`Lt128Upper`/`Eq128`/`Ne128`/`Min128`/`Max128`（HYLE 从 Reverse2/DupOdd 合成,与 Highway 手法异）。
- **跨 lane permute**（对照 Highway）：`TableLookupLanes`/`TwoTablesLookupLanes`/`SetTableIndices`。
- **加宽 MAC**（对照 Highway）：`ReorderWidenMulAccumulate`/`SatWidenMulPairwiseAdd`/`SumOfMulQuadAccumulate`/`SumsOfAdjQuadAbsDiff`/`WidenMulAccumulate`/`SatWidenMulPairwiseAccumulate`/`SatWidenMulAccumFixedPoint`。
- **gather/scatter/masked-mem**（对照 Highway）：`ScatterIndex`/`CompressStore`/`MaskedLoad`/`MaskedLoadOr`/`BlendedStore`/`LoadN`/`StoreN`。
- **位操作**（对照 Highway）：`MulHigh`(i16/u16)/`MulHigh32`/`BitShuffle`/`MultiRotateRight`/`Reverse2`/`Reverse4`/`Reverse8`。
- **reductions**（对照 Highway）：`ReduceMin`/`ReduceMax`/`ReduceMul`/`ReduceAnd`/`ReduceOr`（现只有 ReduceSum + Masked 变体）。

### Tier 2 — 中价值（多指令但较简单,有 vendor 同 op）
`AverageRound`/`SaturatedAbs`/`SaturatedNeg`/`RoundingShr`/`MulRound`/`MulByPow2`/`MulByFloorPow2`/`GetBiasedExponent`/`Clamp`/`Slide1Up/Down`/`SlideUp/DownLanes`/`RotateLeft/Right<>`/`Shuffle{0123,1032,2301,...}`/`Per4LaneBlockShuffle`/`CombineShiftRightBytes/Lanes`/`Broadcast`/`PairwiseAdd128/Sub/Sub128`/`AddSub`/`ZeroExtendVector`/`TruncateTo`/`OrderedDemote2To`。

### Tier 3 — HYLE-only（无 vendor 同 op → 只能 vs scalar,非 parity）
`ReciprocalNewton`/`ReciprocalSqrtNewton`（章程标 HYLE 超越点——值得 vs-Div + vs 裸估计的精度/吞吐 bench 背书那个 SOTA 声称）、`GatherOffset`/`ScatterOffset` store+reload 泛型、`hypot_in_range` 快路径。

### 跳过（纯直通单指令,保证平局,低价值）
`And/Or/Xor/AndNot/Not/Min(f)/Max(f)/CopySign/Set/Zero/Load/Store/IfThenElse/BitCast/Ne/Le/Ge/RoundToInt/Floor/Ceil/Round/ShiftLeftSame/ShiftRightSame`。

## 非-parity bench 缺口（域级零 bench,补 vs scalar 或 vs vendor）
- Crypto AES/CLMul：**零 bench**（见 Tier 1,建 crypto_bench.cc）。
- 标量 complex 族：零 bench（只有 Masked-complex）。
- 128-cmp：零 bench。
- mask 域 op：`PromoteMaskTo`/`OrderedDemote2MasksTo`/`CombineMasks`/`LowerHalfOfMask`/`UpperHalfOfMask`/`SlideMask1Up/Down`/`SlideMaskUp/DownLanes`/`RebindMask`/`SetMask`/`Dup128MaskFromMaskBits`——只有 DemoteMaskTo(§18)被测。
- 单向量 Promote/Demote（新）：`PromoteTo(dw,v)`/`DemoteTo(dn,v)` 1-arg 形 + `ShiftRightAndDemoteTo`/`RoundingShiftRightAndDemoteTo`——零 bench（widen bench 只覆盖 2-arg 形）。
- fp8：`f16_to_f8e4m3/e5m2_batch` 无 bench（f32 源方向已在 dtype_bench）。
- 交织（新）：`LoadInterleaved2`/`StoreInterleaved2`/`StoreInterleaved3` + u16 宽度无 bench（f32/u8 的 Load3/4+Store4 已 §19-20）。

## 正确性测试缺口（vec_test 220 TEST,基本完整）
Spot-check 确认 crypto/complex/128-cmp/mask 域/单向量 promote-demote/Newton/widen-MAC/permute/interleave 均有专门 TEST。**无高置信度缺失项**。唯一待确认：`TableLookupLanes`/`SetTableIndices` 是否有独立 TEST（可能在 `SwizzleCompletion`/`HighwayParityWrappers` 内）——grok 补 bench 时顺手确认。

## 施工纪律提醒（每补一个 bench）
1. 对照公平性两前置（补全 ISA baseline `hyle_set_highway_isa`、用库生产实现）——否则平局/输赢都是假的。
2. 统计法判定,不看单点 ratio。
3. bench 常揪出真事（本项目历史:f64 winner 无留档、RNG 后端配置坑、fp8 无向量化收益、log Linux 出厂 128b bug）——补 bench 不只是背书,是发现工具,认真看异常。
4. 五平台构建过（补 bench 别只在一个 tier 编）。
