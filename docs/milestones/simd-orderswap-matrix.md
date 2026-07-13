# SIMD op parity 三平台矩阵(2026-07-14 校准仪器 v2;TZC/HSB 家族已修)

> 本表由重建后的测量引擎产出(`bench/simd/op_parity_engine.h` + `scripts/op-parity-report.py`),
> 取代旧的"原始读数+人工伪影标签"体系。旧 op_parity 的数字被证明携带三层测量伪影(P/E 调度、
> 代码放置运气、闭包栈 4K 别名,合计可达 ±30~40%),单条 `vpsubd` 的 `Neg i32` 曾被它测成
> 0.75~1.49 之间的任意值。完整证据链与重建过程:[`../debug/op-parity-harness-artifacts-hunt.md`](../debug/op-parity-harness-artifacts-hunt.md)。

## 方法论(数字为何可信)
- **测量**:钉一颗 P 核+高优先级;每 op 每侧 4 个独立实例化的代码副本(采样放置运气,防 ICF 折叠,
  且带确定性相位错位 stagger —— v2 修复:同侧副本若以 64 倍数步长排布会共享同一相位,
  使 best-of-replicas 偏向另一侧达 ~10%,同文本对拍曾读 0.898,修后读 1.000);
  H/G 交替轮次;kernel 指针外提(热循环零闭包访问);宏形状循环 4x 展开。
- **聚合**:best-of-replicas(放置理想值)+ median + 逐轮配对比值三估计器;判决需三者在每个输入
  运行中一致越线,且线宽随该 op 实测副本散布自适应(>= ±3%)。每平台喂双序(HYLE 先/Highway 先)。
- **校准闸门**:5 个源码级验证"两侧同指令"的锚点 op(真值=1.00)必须全部落在闸门内,否则整表拒绝输出。
  本表锚点读数:MSVC/WSL 0.99~1.03(闸门 ±4%),Mac 0.95~1.05(闸门 ±6%,无亲和性 API,列分辨率较低,
  已用 4 次运行合并 + 8x epoch 补偿)。
- **判读**:数字 = HYLE/Highway 配对中位比值(>1=HYLE 快);W=WIN,L=LOSE,==TIE,?=UNSTABLE(多运行判决不一致),
  ·=该平台无此 op。UNSTABLE 一律不算输赢。Mac 列只覆盖 full 形状集(NEON 无 parity/widen 目标)。

## 真输清单(校准仪器判决,按比值升序)
### MSVC(AVX2): 11 个
- DemoteToNearestInt f64->i32 (4096): **0.39** —— 已知正确性取舍:HYLE safe 版做 emu128 谱系 NaN/饱和(Highway 原生只 Min+cvt);InRange 快路径同表 TIE
- DemoteMaskTo i32->i16 (4096): **0.74** —— mask 宽度桥接冷 op,残余为寄存器搬运差(源注释已记录,接受)
- AddSub i32: **0.86** —— 仅 MSVC 0.86,GCC TIE
- CompressNot i32 (4096, 50%): **0.86** —— 仅 MSVC 0.86
- RoundingShiftRightAndDemoteTo<3> i32->i16 (4096): **0.89**
- BroadcastLane<3> i32: **0.92** —— 已定论保留 vpermd:MSVC 0.92 是 bench 形状特定差距(Highway 的 vbroadcasti128 仅在输入直接来自内存时被折叠成 load 口免费复制;寄存器输入下该序列被 MSVC 编成栈 spill(实测 0.47),而 vpermd 单 uop 更优,GCC 上 vpermd 反赢 1.13)
- SumOfLanes i32: **0.92** —— 仅 MSVC(GCC 反赢);已从 0.78 改善到 0.91(in-vector broadcast-reduce),残余为 MSVC 水平归约 codegen
- Mod i32 invariant: **0.93**
- CompressStore i32 (4096, 50%): **0.94**
- SatWidenMulPairwiseAccumulate i16->i32: **0.94**
- MinOfLanes i32: **0.95**

### WSL/GCC(AVX2): 11 个
- DemoteToNearestInt f64->i32 (4096): **0.26** —— 已知正确性取舍:HYLE safe 版做 emu128 谱系 NaN/饱和(Highway 原生只 Min+cvt);InRange 快路径同表 TIE
- DemoteMaskTo i32->i16 (4096): **0.33** —— mask 宽度桥接冷 op,残余为寄存器搬运差(源注释已记录,接受)
- LoadNOr i32 (half n): **0.55** —— 仅 GCC 输 0.55(旧文档曾据有偏复核"证伪不是真输",校准仪器推翻该免罪)
- LoadN/StoreN i32 (half lanes): **0.66** —— 仅 GCC 输 0.66,与 LoadNOr 同族(partial load/store 路径)
- CombineMasks i32 (4096): **0.84** —— 仅 GCC 0.84
- ConcatLowerLower i32: **0.88** —— 仅 GCC 0.88
- TableLookupBytes u8 (4096): **0.89** —— HYLE 多一条设计内的 vpand 索引掩码(安全语义,非缺陷)
- RoundingShr i32: **0.94**
- FindLastTrue i32: **0.95**
- InterleaveLowerBlocks i32: **0.95**
- PairwiseSub128 i32: **0.96**

### Mac/NEON: 3 个
- RoundingShiftRightSame i32 bits=5: **0.81** —— 仅 NEON 0.81
- MulAddSub f32: **0.90** —— 仅 NEON 0.90
- MulSubAdd f32: **0.90** —— 仅 NEON 0.90

### x86 双编译器交集(最高优先级): 2 个
- DemoteMaskTo i32->i16 (4096): MSVC 0.74 / WSL 0.33
- DemoteToNearestInt f64->i32 (4096): MSVC 0.39 / WSL 0.26

### 与旧报告的对照(演进:旧 harness -> v1 校准仪器 -> 修复 + v2 stagger)
- 旧 op_parity 报 ~40-50 个 0.xx(MSVC 列);v1 校准仪器判 MSVC 21 / WSL 11 / Mac 3;
  经 5 组后端修复(ReverseLaneBytes 0.05->TIE、Broadcast<0>->TIE、TruncateTo->TIE、
  SumOfLanes 0.78->0.91、Min/MaxOfLanes->TIE)+ v2 仪器修相位相关后,当前为上表数字。
- v1 的假 LOSE 被 v2 消除的例子:Not i32 / TestBit / RebindMask / IsNegative(GCC)——同侧副本
  相位相关曾让它们稳定读 0.83~0.96,stagger 后回到 TIE/UNSTABLE。
- 旧"隔离复核"的 7 真输里,BitShuffle/SetAtOrAfterFirst/Lt128Upper/Ne128/MultiRotateRight 已被此前的
  修复收回(本表均 TIE/WIN,修复有效性经校准仪器确认);LoadNOr 的"证伪免罪"被推翻(WSL 0.55 真输)。
- **2026-07-14 TZC/HSB 浮点指数家族修复(MSVC 13→11 / WSL 13→11 / Mac 4→3,x86 双编译器交集 4→2)**:
  ① TrailingZeroCount(三平台唯一全输 0.86/0.87/0.69)——avx2/avx512/sse4 去掉共享 `u32_to_f32_biased_exp`
  里对单 bit 输入无用的 round-down 归一、改 Min(raw_exp,158) 修 2^31 signed-cvt;NEON 改
  `LeadingZeroCount(ReverseBits)`。三平台 TIE(1.00/1.00/0.98)。② HighestSetBitIndex(x86 双编译器 0.82/0.85)
  ——加 native 直算 `biased_exp-127`(Highway u32 shape),跳过 `31-LeadingZeroCount` 往返里 defined 输入下
  恒为 no-op 的 Min(.,32)+额外 Sub。双编译器 TIE(0.99/1.00)。③ **顺带揪出真正的正确性 bug**:该共享 helper
  漏了 Highway 的 Min(.,158) 钳制,导致 bit-31-set 输入(0x80000000..0xFFFFFFFF)signed cvtepi32_ps 变负、
  sign bit 泄漏到 >>23 → LZCnt/HSB 全错(LZCnt(2^31) 读 32 而非 0);gtest 因取值 <2^18 从未覆盖。
  已补 Min(158) 修正 + gtest 加全 bit 位扫。avx512 由 Intel SDE 真跑 221 全绿(非仅编译验)。

## 全量表

| op | MSVC(AVX2) | WSL GCC(AVX2) | Mac(NEON) |
|---|---|---|---|
| Abs i32 | 1.00= | 0.99= | 1.01= |
| AbsDiff i32 | 1.00= | 1.01= | 1.01= |
| Add f32 | 1.01= | 1.01= | 1.03= |
| Add f32 (4096) | 1.00= | 0.99= | · |
| AddSub i32 | 0.86L | 0.97= | 1.00= |
| AllBits0 i32 | 1.00= | 0.99= | 1.00= |
| AllBits1 i32 | 1.00= | 1.00= | 1.00= |
| AllFalse i32 | 0.98= | 1.01= | 1.00= |
| AllTrue i32 | 0.99= | 1.00= | 0.99= |
| And i32 | 1.01= | 1.00= | 1.00= |
| AndNot i32 | 1.01= | 0.99= | 1.01= |
| ApproximateReciprocal f32 | 1.01= | 1.01= | 1.00= |
| ApproximateReciprocalSqrt f32 | 1.01= | 1.00= | 1.00= |
| AverageRound u16 | 1.01= | 1.00= | 1.00= |
| BitCast f32->i32 | 1.01= | 1.02= | 1.00= |
| BitCast i32->f32 | 0.97= | 1.00= | 1.00= |
| BitShuffle u64 (4096) | 1.06W | 1.12W | · |
| BlendedStore i32 (4096, 50%) | 0.99= | 1.01= | · |
| Broadcast<0> i32 | 1.00= | 1.00= | 1.00= |
| BroadcastBlock<1> i32 | 1.02? | 1.00= | · |
| BroadcastLane<3> i32 | 0.92L | 1.13W | 0.96= |
| BroadcastSignBit i32 | 1.00= | 1.00= | 0.99= |
| Ceil f32 | 1.00= | 0.98= | 0.99= |
| CeilInt f32->i32 | 1.48W | 1.56W | 0.99= |
| Clamp i32 | 1.00= | 0.99= | 0.99= |
| CombineMasks i32 (4096) | 1.01= | 0.84L | · |
| CombineShiftRightBytes4 i32 | 0.99= | 1.00= | 1.00= |
| CombineShiftRightLanes1 i32 | 1.01= | 1.00= | 1.00= |
| ComplexConj f32 | 1.25W | 1.01= | 1.01= |
| Compress i32 (4096, 50% dense) | 0.98= | 1.04? | · |
| CompressBlendedStore i32 (4096, 50%) | 1.00= | 1.22W | · |
| CompressNot i32 (4096, 50%) | 0.86L | 0.97? | · |
| CompressStore i32 (4096, 50%) | 0.94L | 1.03= | · |
| ConcatEven i32 (4096) | 0.99= | 1.00= | · |
| ConcatLowerLower i32 | 0.95= | 0.88L | · |
| ConcatLowerUpper i32 | 0.95= | 0.99= | · |
| ConcatOdd i32 | 1.00= | 0.99= | 1.00= |
| ConcatUpperLower i32 | 1.04= | 0.99= | · |
| ConcatUpperUpper i32 | 1.00= | 1.01= | · |
| ConvertTo f32->i32 | 1.38W | 1.12? | 0.99= |
| ConvertTo i32->f32 | 1.00= | 0.99= | 1.00= |
| CopySign f32 | 1.00= | 1.00= | 1.01= |
| CountTrue i32 | 1.04= | 1.00= | 1.00= |
| DemoteMaskTo i32->i16 (4096) | 0.74L | 0.33L | · |
| DemoteTo i32->i16 (sat) | 1.88W | 1.84W | · |
| DemoteTo i32->i16 1-arg (4096) | 1.00= | 1.00= | · |
| DemoteToNearestInt f64->i32 (4096) | 0.39L | 0.26L | · |
| Div f32 | 1.00= | 1.00= | 1.00= |
| Dup128MaskFromMaskBits i32 (4096) | 1.05? | 1.18W | · |
| DupEven i32 | 1.00= | 0.99= | 0.99? |
| DupOdd i32 | 0.99= | 1.00= | 0.97= |
| Eq i32 | 1.00= | 1.00= | 0.99= |
| Eq128 u64 | 1.00= | 1.01= | 1.04? |
| ExclusiveNeither i32 | 1.00= | 1.01= | 1.01= |
| Expand i32 (4096, 50%) | 2.22W | 2.14W | · |
| FindFirstTrue i32 | 0.99= | 1.01= | 1.06= |
| FindKnownFirstTrue i32 | 0.97= | 1.00= | 1.18W |
| FindKnownLastTrue i32 | 1.07W | 1.00= | 1.12W |
| FindLastTrue i32 | 1.02= | 0.95L | 1.19W |
| Floor f32 | 1.00= | 0.99= | 0.94= |
| FloorInt f32->i32 | 1.48W | 1.57W | 0.99= |
| GatherIndex i32 (4096) | 1.00= | 1.02= | · |
| GatherIndexN i32 (half n) | 1.00= | 0.98= | · |
| GatherIndexNOr i32 (half n) | 1.01= | 1.00= | · |
| GatherOffset i32 (4096) | 1.00= | 1.02= | · |
| Ge i32 | 0.99= | 1.00= | 0.98= |
| GetBiasedExponent f32 | 1.00= | 1.02= | 1.00= |
| GetExponent f32 | 0.98= | 1.01= | · |
| Gt i32 | 1.01= | 1.02= | 1.04= |
| HighestSetBitIndex u32 | 0.99= | 1.00= | 1.00= |
| IfNegativeThenElse f32 | 1.05W | 1.07W | 1.00= |
| IfNegativeThenElseZero f32 | 1.36W | 1.08W | 1.03? |
| IfNegativeThenZeroElse f32 | 1.31W | 1.08W | 1.02= |
| IfThenElse i32 | 1.00= | 0.99= | 1.00= |
| IfThenElseZero i32 | 0.97? | 1.00= | 1.02= |
| IfThenZeroElse i32 | 0.97? | 1.00= | 1.03= |
| InterleaveEven i32 | 1.01= | 0.99= | 1.01= |
| InterleaveLower i32 (4096) | 1.00= | 1.01= | · |
| InterleaveLowerBlocks i32 | 0.91= | 0.95L | · |
| InterleaveOdd i32 | 1.01= | 1.00= | 1.02? |
| InterleaveUpper i32 | 1.00= | 1.00= | 1.03= |
| InterleaveWholeLower i32 | 1.00= | 1.00= | 1.03= |
| InterleaveWholeUpper i32 | 1.00= | 0.99= | 1.05= |
| IsEitherNaN f32 | 1.00= | 1.00= | 0.99= |
| IsFinite f32 | 0.99= | 1.31W | 1.00= |
| IsInf f32 | 0.94? | 1.11? | 1.00= |
| IsNaN f32 | 1.00= | 0.98= | 1.00= |
| IsNegative i32 | 0.98= | 0.94? | 1.00= |
| Le i32 | 0.99= | 1.01= | 1.00= |
| LeadingZeroCount u32 (4096) | 1.05= | 1.17W | · |
| Load/Store i32 aligned | 1.00= | 0.99= | 0.99= |
| LoadExpand i32 (4096, 50%) | 2.24W | 2.15W | · |
| LoadInterleaved2 f32 (4096) | 1.00= | 1.00= | · |
| LoadInterleaved3 f32 (4096) | 0.99= | 1.00= | · |
| LoadInterleaved4 u8 (4096, RGBA) | 1.00= | 0.98= | · |
| LoadN/StoreN i32 (half lanes) | 1.09? | 0.66L | · |
| LoadNOr i32 (half n) | 0.84? | 0.55L | · |
| LoadU/StoreU i32 | 1.00= | 0.99= | 0.99= |
| Lt i32 | 1.00= | 0.99= | 0.98= |
| Lt128 u64 | 1.00= | 0.99= | 0.99= |
| Lt128Upper u64 | 1.00= | 1.00= | 0.95= |
| Mask And i32 | 0.99= | 1.00= | 1.00= |
| Mask AndNot i32 | 1.00= | 0.99= | 1.00= |
| Mask Not i32 | 1.00= | 1.00= | 1.00= |
| Mask Or i32 | 0.99= | 1.00= | 1.00= |
| Mask Xor i32 | 1.01= | 1.00= | 1.00= |
| MaskedApproxReciprocal f32 | 1.02= | 1.01= | 1.00= |
| MaskedApproxReciprocalSqrt f32 | 1.00= | 1.00= | 1.00= |
| MaskedConvertTo f32-i32 | 2.04W | 1.51W | 1.01= |
| MaskedLeadingZeroCount i32 | 1.13W | 1.12W | 0.97= |
| MaskedLoad i32 (4096, 50%) | 0.97= | 1.01= | · |
| MaskedLoadOr i32 (4096) | 0.98? | 1.01= | · |
| MaskedModOr i32 invariant | 0.98= | 0.99= | 1.06W |
| MaskedMulComplex f32 | 1.19W | 1.08W | 1.00= |
| MaskedMulComplexAdd f32 | 1.13W | 1.02= | 1.00= |
| MaskedMulComplexConj f32 | 1.21W | 1.10W | 1.00= |
| MaskedMulComplexConjAdd f32 | 1.13W | 1.03= | 1.00= |
| MaskedMulComplexOr f32 | 1.16W | 1.05W | 1.00= |
| MaskedMulFixedPoint15 i16 | 1.02= | 0.99= | 1.00= |
| MaskedReduceMax i32 | 1.00= | 0.98= | 1.00= |
| MaskedReduceMin i32 | 1.00= | 1.00= | 1.00= |
| MaskedReduceSum i32 | 1.00= | 0.99= | 1.00= |
| MaskedSatAddOr i16 | 0.99= | 0.99= | 1.00= |
| MaskedSatSubOr i16 | 1.00= | 1.01= | 1.00= |
| MaskedShiftLeft<3> i32 | 0.99= | 1.00= | 1.01= |
| MaskedShiftLeftOr<3> i32 | 1.00= | 0.99= | 1.00= |
| MaskedShiftRight<3> i32 | 0.99= | 0.99= | 1.02= |
| MaskedShiftRightOr<3> i32 | 1.00= | 1.00= | 1.01= |
| MaskedShl i32 | 1.02= | 1.00= | 1.02= |
| MaskedShlOr i32 | 1.00= | 0.98? | 1.00= |
| MaskedShr i32 | 1.02= | 1.01= | 0.99= |
| MaskedShrOr i32 | 1.00= | 0.97= | 1.00= |
| MaskedWidenMulPairwiseAdd i16-i32 | 1.00= | 1.00= | 0.99= |
| Max f32 | 1.01= | 0.97= | 1.01? |
| Max i32 | 0.99= | 1.00= | 0.97= |
| Max128 u64 | 0.99= | 1.01= | 1.00= |
| MaxMagnitude f32 | 1.00= | 1.01= | 1.00= |
| MaxNumber f32 | 1.01= | 1.00= | 1.00= |
| MaxOfLanes i32 | 0.97? | 1.16W | 0.98= |
| Min f32 | 1.00= | 1.02= | 0.98= |
| Min i32 (4096) | 1.00= | 1.02= | · |
| Min128 u64 | 0.99= | 0.99= | 1.00= |
| MinMagnitude f32 | 1.01= | 1.01= | 1.00= |
| MinNumber f32 | 1.00= | 1.02= | 0.99= |
| MinOfLanes i32 | 0.95L | 1.17W | 0.98= |
| Mod i32 invariant | 0.93L | 1.00= | 1.07W |
| Mul f32 | 0.99= | 1.01= | 1.02= |
| Mul i32 | 1.01= | 0.99= | 1.00= |
| MulAdd f32 (4096) | 1.00= | 1.00= | · |
| MulAddSub f32 | 1.01= | 1.00= | 0.90L |
| MulByFloorPow2 f32 | 1.06? | 0.98= | 1.00= |
| MulByPow2 f32 | 1.00= | 1.01= | 1.00= |
| MulComplex f32 | 1.42W | 1.22W | 1.00= |
| MulComplexAdd f32 | 1.02= | 1.03= | 1.00= |
| MulComplexConj f32 | 1.43W | 1.20W | 1.00= |
| MulComplexConjAdd f32 | 1.02= | 1.03= | 1.00= |
| MulEven i32-i64 | 1.00= | 1.01= | 1.00= |
| MulFixedPoint15 i16 | 1.00= | 0.99= | · |
| MulHigh i16 (4096) | 1.00= | 1.01= | · |
| MulHigh u16 (4096) | 1.00= | 1.01? | · |
| MulHigh32 u32 (4096) | 0.98= | 1.02= | · |
| MulOdd i32-i64 | 1.01= | 0.95= | 1.00= |
| MulRound f32 | 0.98= | 1.00= | 0.98? |
| MulSub f32 | 1.00= | 1.00= | 1.01= |
| MulSubAdd f32 | 1.02= | 0.97= | 0.90L |
| MultiRotateRight u64 (4096) | 1.00= | 1.00= | · |
| Ne i32 | 1.00= | 1.00= | 1.03= |
| Ne128 u64 | 1.00= | 1.01= | 1.01= |
| NearestInt f32->i32 | 1.35W | 1.10W | 1.00= |
| Neg i32 | 1.00= | 0.99= | 0.96= |
| NegMulAdd f32 | 1.00= | 1.00= | 1.00? |
| NegMulSub f32 | 1.00= | 0.99= | 0.98= |
| Not i32 | 0.99= | 0.99= | 0.99= |
| OddEven i32 | 1.00= | 1.01= | 0.99= |
| OddEvenBlocks i32 | 0.98= | 0.99= | · |
| Or i32 | 1.01= | 1.00= | 0.97= |
| Or3 i32 | 1.00= | 1.01= | 0.99= |
| OrAnd i32 | 1.00= | 1.00= | 1.00= |
| OrderedDemote2MasksTo i32->i16 (4096) | 1.00= | 1.01= | · |
| OrderedDemote2To i32->i16 | 0.96= | 1.02? | · |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.13W | 1.24W | · |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 0.99= | 0.99? | · |
| PairwiseAdd i32 | 1.09W | 1.02= | 0.99= |
| PairwiseAdd128 i32 | 1.00= | 0.99= | 1.00= |
| PairwiseSub i32 | 1.10W | 1.02= | 1.00= |
| PairwiseSub128 i32 | 1.10W | 0.96L | 1.00= |
| Per4LaneBlockShuffle 3210 i32 | 0.86= | 0.98= | 1.00= |
| PopulationCount u32 (4096) | 1.48W | 1.52W | · |
| PromoteEvenTo i8->i16 | 0.99= | 1.00= | · |
| PromoteLowerTo i16->i32 (4096) | 0.99= | 0.98= | · |
| PromoteMaskTo i16->i32 (4096) | 0.98= | 1.00= | · |
| PromoteOddTo i8->i16 | 0.99= | 1.00= | · |
| PromoteTo i16->i32 1-arg (4096) | 1.01= | 0.99= | · |
| PromoteUpperTo i16->i32 | 1.01= | 1.01= | · |
| RebindMask i32->f32 (4096) | 0.97? | 1.01= | · |
| ReduceMax i32 | 1.18W | 1.02= | 1.00= |
| ReduceMin i32 | 1.18W | 1.03= | 1.00= |
| ReduceSum f32 (4096) | 0.99= | 0.99= | · |
| ReorderDemote2To i32->i16 | 0.99= | 1.04? | · |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.04= | 1.17W | · |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 1.00= | 1.03= | · |
| ReorderWidenMulAccumulate i16->i32 | 1.00= | 1.01= | · |
| Reverse i32 (4096) | 1.00= | 1.01= | · |
| Reverse2 i32 | 1.00= | 1.00= | 0.99= |
| Reverse4 i32 | 1.00= | 0.98= | 0.99= |
| Reverse8 i32 (4096) | 1.00= | 0.99= | · |
| ReverseBits i32 | 2.10W | 1.97W | 1.01= |
| ReverseBlocks i32 | 1.05= | 1.00= | · |
| ReverseLaneBytes u32 (4096) | 0.99= | 0.99= | · |
| RotateLeft<3> i32 | 0.98= | 1.00= | 1.02= |
| RotateLeftSame u32 bits=5 | 0.99= | 1.01= | 0.98= |
| RotateRight<3> i32 | 0.99= | 1.01= | 1.01? |
| RotateRightSame u32 bits=5 | 1.00= | 1.01= | 1.01= |
| Round f32 | 1.00= | 1.01= | 0.98= |
| RoundToInt f32->i32 | 1.43W | 1.11? | 1.00= |
| RoundingShiftRight<2> i32 | 1.09= | 0.99= | 0.98= |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.89L | 1.12W | · |
| RoundingShiftRightSame i32 bits=5 | 1.09W | 1.00= | 0.81L |
| RoundingShr i32 | 1.04W | 0.94L | 1.00= |
| SatWidenMulAccumFixedPoint i16->i32 | 1.90W | 1.77W | · |
| SatWidenMulPairwiseAccumulate i16->i32 | 0.94L | 1.00= | · |
| SatWidenMulPairwiseAdd u8*i8->i16 | 1.01= | 1.00= | · |
| SaturatedAbs i32 | 1.00= | 0.99= | 0.97= |
| SaturatedAdd i16 (4096) | 1.00= | 1.00= | · |
| SaturatedNeg i32 | 1.00= | 0.98= | 1.00= |
| SaturatedSub i16 | 1.00= | 1.00= | 1.00= |
| ScatterIndex i32 (4096) | 0.98= | 0.98= | · |
| ScatterIndexN i32 (half n) | 1.23W | 1.37W | · |
| ScatterOffset i32 (4096) | 1.02= | 1.02= | · |
| Set i32 | 1.00= | 1.02= | 1.00= |
| SetAtOrAfterFirst i32 | 1.00= | 1.00= | 1.00= |
| SetAtOrBeforeFirst i32 | 0.99= | 1.15W | 1.00= |
| SetBeforeFirst i32 | 1.00= | 1.23W | 1.00= |
| SetMask i32 (4096) | 0.99= | 0.99= | · |
| SetOnlyFirst i32 | 0.99= | 1.61W | 1.00= |
| ShiftLeft<3> i32 | 1.01= | 1.01= | 1.00= |
| ShiftLeftBytes<4> i32 | 1.00= | 0.99= | 1.00= |
| ShiftLeftLanes<1> i32 | 1.00= | 1.00= | 1.01= |
| ShiftLeftSame i32 bits=3 | 1.00= | 0.99= | 1.92W |
| ShiftRight<3> i32 | 1.00= | 0.99= | 0.97= |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.99= | 1.01= | · |
| ShiftRightBytes<4> i32 | 1.00= | 1.00= | 1.01= |
| ShiftRightLanes<1> i32 | 1.00= | 1.01= | 1.00= |
| ShiftRightSame i32 bits=3 | 1.00= | 1.00= | 1.22W |
| Shl i32 | 1.01= | 0.99= | 1.02= |
| Shr u32 (variable) | 1.00= | 1.02= | 1.00= |
| Shuffle0123 i32 | 1.00= | 0.97= | 1.05= |
| Shuffle0321 i32 | 1.00= | 1.01= | 1.00= |
| Shuffle1032 i32 | 1.00= | 0.99= | 1.00? |
| Shuffle2103 i32 | 1.00= | 0.99= | 0.98= |
| Shuffle2301 i32 | 1.00= | 0.99= | 1.00= |
| Slide1Down i32 | 1.01= | 0.99= | 0.99= |
| Slide1Up i32 | 1.00= | 1.00= | 1.02= |
| SlideDownLanes amt=2 i32 | 0.99= | 0.99= | 1.00= |
| SlideMask1Down i32 (4096) | 0.99= | 1.02= | · |
| SlideMask1Up i32 (4096) | 1.00= | 0.99= | · |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.10W | 1.01= | · |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.06? | 1.00= | · |
| SlideUpLanes amt=2 i32 | 1.00= | 1.00= | 1.00= |
| Sqrt f32 | 1.00= | 1.00= | 1.00= |
| StoreInterleaved2 f32 (4096) | 0.99= | 1.00= | · |
| StoreInterleaved3 f32 (4096) | 1.04? | 1.04? | · |
| StoreInterleaved4 f32 (4096) | 1.49W | 1.27? | · |
| Sub f32 | 0.99= | 1.01= | 0.97= |
| Sub i32 | 1.00= | 0.99= | 1.02= |
| SumOfLanes i32 | 0.92L | 1.14W | 1.01= |
| SumOfMulQuadAccumulate i8->i32 | 0.99= | 1.02= | · |
| SumsOf2 u8->u16 | 1.01= | 1.00= | · |
| SumsOf4 u8->u32 | 1.00= | 0.99= | · |
| SumsOf8 u8->u64 | 1.00? | 1.03? | · |
| SumsOf8AbsDiff u8->u64 | 0.99= | 1.01= | · |
| SumsOfAdjQuadAbsDiff u8->u16 | 0.98= | 1.00= | · |
| SwapAdjacentBlocks i32 | 1.06= | 1.00= | · |
| TableLookupBytes u8 (4096) | 0.95? | 0.89L | · |
| TableLookupLanes i32 (4096) | 1.00= | 0.99= | · |
| TestBit i32 | 1.04= | 0.99= | 1.00= |
| TrailingZeroCount i32 | 1.00= | 1.00= | 0.98= |
| Trunc f32 | 1.00= | 1.00= | 1.04? |
| TruncateTo u32->u16 | 1.00= | 1.00= | 1.00= |
| TwoTablesLookupLanes i32 (4096) | 0.99= | 1.00= | · |
| WidenMulAccumulate i16->i32 | 1.00= | 1.00= | · |
| WidenMulPairwiseAdd i16->i32 | 1.00= | 1.00= | · |
| Xor i32 | 1.00= | 1.01? | 1.01= |
| Xor3 i32 | 1.63W | 1.00= | 1.02= |
| Zero i32 | 1.00= | 1.00= | 1.04= |
| ZeroExtendVector i16 half->full | 0.96= | 1.00= | · |
| ZeroIfNegative f32 | 1.04? | 1.09W | 1.00= |

## 复现
```
# 每平台:构建 hyle_op_parity{,_full,_widen}_bench_<isa>,双序各跑一遍(Mac 建议 4 遍),然后
python scripts/op-parity-report.py out_hf.txt out_hwf.txt          # x86(闸门 ±4%)
python scripts/op-parity-report.py --gate 0.06 mac_*.txt           # Mac(无亲和性,±6%)
# 报告首段为锚点校准;CALIBRATION PASS 之后的判决才可引用。
```
