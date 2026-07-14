# SIMD op parity 三平台矩阵(2026-07-13 校准仪器 v2 全量重测)

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
- DemoteToNearestInt f64->i32 (4096): **0.61** —— 正确性取舍待定稿:HYLE 现版(cvt-first)仍保 sign-based NaN 饱和;决策=改随同 target 的 Highway 语义(x86 NaN->INT_MIN / NEON NaN->0)即追平,严格版无消费者、删
- AddSub i32: **0.85** —— MSVC 0.86 / GCC 0.94(GCC 侧v10新进临界区),真实小差距
- CompressNot i32 (4096, 50%): **0.88** —— 仅 MSVC 0.88
- SetBeforeFirst i32: **0.89** —— 仅 MSVC 0.89(本轮新进,GCC 1.3 反赢;borderline 待复核)
- SumOfLanes i32: **0.91** —— 仅 MSVC(GCC 反赢);已从 0.78 改善到 0.91(in-vector broadcast-reduce),残余为 MSVC 水平归约 codegen
- SatWidenMulPairwiseAccumulate i16->i32: **0.93**
- BroadcastLane<3> i32: **0.93** —— 已定论保留 vpermd:MSVC 0.93 是 bench 形状特定差距(Highway 的内存源 vbroadcasti128 折叠红利;寄存器输入下 vpermd 更优,GCC 反赢 1.13)
- IsInf f32: **0.93** —— 仅 MSVC 0.93
- MaxOfLanes i32: **0.95** —— 仅 MSVC 0.95(已从 0.86 改善,边缘)
- Not i32: **0.96** —— 仅 MSVC 0.96(阈值边缘徘徊型)
- MinOfLanes i32: **0.97** —— 仅 MSVC 0.97(已从 0.86 改善,边缘)

### WSL/GCC(AVX2): 5 个
- DemoteToNearestInt f64->i32 (4096): **0.44** —— 正确性取舍待定稿:HYLE 现版(cvt-first)仍保 sign-based NaN 饱和;决策=改随同 target 的 Highway 语义(x86 NaN->INT_MIN / NEON NaN->0)即追平,严格版无消费者、删
- TableLookupBytes u8 (4096): **0.88** —— 语义代价非缺陷:scalar oracle 固化 &15 全定义语义(比 Highway 的"越界=实现定义"承诺更强),vpand 是契约本体,保留
- AddSub i32: **0.94** —— MSVC 0.86 / GCC 0.94(GCC 侧v10新进临界区),真实小差距
- FindLastTrue i32: **0.94** —— 仅 GCC 0.94;objdump 与 Highway 逐指令等价 -> 仪器单 op 残余分辨率边缘,非算法输
- RoundingShr i32: **0.95** —— 仅 GCC 0.95;已试 Highway 分支型(MSVC 平/GCC 反退),判 GCC codegen 地板

### Mac/NEON: 3 个
- RoundingShiftRightSame i32 bits=5: **0.82** —— 仅 NEON 0.82
- MulAddSub f32: **0.88** —— 仅 NEON 0.88
- MulSubAdd f32: **0.90** —— 仅 NEON 0.90

### x86 双编译器交集(最高优先级): 2 个
- AddSub i32: MSVC 0.85 / WSL 0.94
- DemoteToNearestInt f64->i32 (4096): MSVC 0.61 / WSL 0.44

### 与旧报告的对照(演进:旧 harness -> v1 校准仪器 -> 修复 + v2 stagger)
- 旧 op_parity 报 ~40-50 个 0.xx(MSVC 列);v1 校准仪器判 MSVC 21 / WSL 11 / Mac 3;两轮后端修复
  (第一轮:ReverseLaneBytes 0.05->TIE、Broadcast<0>、TruncateTo、SumOfLanes、Min/MaxOfLanes;
  第二轮:TZC 三平台追平、HSBI、DemoteMaskTo 0.33->1.4W、LoadN/StoreN 与 LoadNOr 弃 vpmaskmovd
  ->WIN、CombineMasks、ConcatLowerLower)+ v2 stagger 仪器修相位相关后,收敛为上表数字。
- v1 的假 LOSE 被 v2 消除的例子:TestBit / RebindMask / IsNegative(GCC)——同侧副本相位相关
  曾让它们稳定读 0.83~0.96,stagger 后回到 TIE/UNSTABLE。
- 本轮回归教训:ConcatLowerLower 的 vinsert 形态 GCC 赢但 MSVC 在复合上下文(Slide1Up)编成
  extract+栈 spill 掉到 0.099(单独 kernel 却 TIE)——复合上下文回归只有全表重测才抓得住,
  已按编译器分叉(MSVC=vperm2i128 / GCC=vinsert)修复,Slide1Up/SlideMask1Up 复位 TIE/WIN。

## 全量表

| op | MSVC(AVX2) | WSL GCC(AVX2) | Mac(NEON) |
|---|---|---|---|
| Abs i32 | 1.00= | 1.02= | 0.97= |
| AbsDiff i32 | 1.01= | 1.01= | 1.01? |
| Add f32 | 1.00= | 0.99= | 0.99= |
| Add f32 (4096) | 1.00= | 1.00= | · |
| AddSub i32 | 0.85L | 0.94L | 1.01= |
| AllBits0 i32 | 1.01= | 1.01= | 1.00= |
| AllBits1 i32 | 1.00= | 1.00= | 1.00= |
| AllFalse i32 | 1.00= | 0.99= | 1.00= |
| AllTrue i32 | 0.99= | 1.00= | 1.00= |
| And i32 | 1.00= | 1.02= | 1.03= |
| AndNot i32 | 0.99= | 1.00= | 0.98= |
| ApproximateReciprocal f32 | 1.00= | 1.00= | 1.00= |
| ApproximateReciprocalSqrt f32 | 1.00= | 0.98= | 1.00= |
| AverageRound u16 | 0.99= | 1.01= | 1.00= |
| BitCast f32->i32 | 1.00= | 1.01= | 1.00= |
| BitCast i32->f32 | 1.00? | 0.99= | 1.00= |
| BitShuffle u64 (4096) | 1.06W | 1.11W | · |
| BlendedStore i32 (4096, 50%) | 0.99= | 1.01= | · |
| Broadcast<0> i32 | 1.00= | 1.00= | 0.97= |
| BroadcastBlock<1> i32 | 0.98= | 0.98= | · |
| BroadcastLane<3> i32 | 0.93L | 1.10W | 0.97? |
| BroadcastSignBit i32 | 1.00= | 1.02= | 0.96= |
| Ceil f32 | 1.00= | 1.00= | 1.01= |
| CeilInt f32->i32 | 1.46W | 1.54W | 1.00= |
| Clamp i32 | 1.00= | 0.99= | 1.01= |
| CombineMasks i32 (4096) | 0.95= | 1.00= | · |
| CombineShiftRightBytes4 i32 | 0.99= | 0.98? | 1.01= |
| CombineShiftRightLanes1 i32 | 0.97= | 0.99= | 1.00= |
| ComplexConj f32 | 1.24W | 1.00= | 0.98? |
| Compress i32 (4096, 50% dense) | 0.98= | 1.03= | · |
| CompressBlendedStore i32 (4096, 50%) | 0.99= | 1.18W | · |
| CompressNot i32 (4096, 50%) | 0.88L | 1.00= | · |
| CompressStore i32 (4096, 50%) | 0.93? | 0.98= | · |
| ConcatEven i32 (4096) | 0.97= | 1.01= | · |
| ConcatLowerLower i32 | 1.01= | 1.00= | · |
| ConcatLowerUpper i32 | 1.00= | 0.98= | · |
| ConcatOdd i32 | 1.00= | 0.99= | 0.99= |
| ConcatUpperLower i32 | 1.01= | 1.00= | · |
| ConcatUpperUpper i32 | 0.95= | 0.97? | · |
| ConvertTo f32->i32 | 1.41W | 1.15W | 1.00= |
| ConvertTo i32->f32 | 1.00= | 0.99= | 1.00= |
| CopySign f32 | 1.00= | 0.98= | 0.96? |
| CountTrue i32 | 1.02= | 0.99= | 1.00= |
| DemoteMaskTo i32->i16 (4096) | 1.02= | 1.41W | · |
| DemoteTo i32->i16 (sat) | 1.88W | 1.89W | · |
| DemoteTo i32->i16 1-arg (4096) | 1.03= | 1.00= | · |
| DemoteToNearestInt f64->i32 (4096) | 0.61L | 0.44L | · |
| Div f32 | 1.00= | 1.00= | 1.00= |
| Dup128MaskFromMaskBits i32 (4096) | 1.06? | 1.17W | · |
| DupEven i32 | 0.99= | 1.00= | 0.99= |
| DupOdd i32 | 1.00= | 1.01= | 1.02? |
| Eq i32 | 1.00= | 1.01= | 1.03? |
| Eq128 u64 | 1.01= | 1.00= | 1.03= |
| ExclusiveNeither i32 | 1.03= | 1.01= | 1.00= |
| Expand i32 (4096, 50%) | 1.76W | 2.14W | · |
| FindFirstTrue i32 | 0.99= | 1.00= | 1.05= |
| FindKnownFirstTrue i32 | 0.99= | 0.97= | 1.19W |
| FindKnownLastTrue i32 | 1.05? | 1.00= | 1.11? |
| FindLastTrue i32 | 1.02= | 0.94L | 1.15? |
| Floor f32 | 1.00= | 0.97= | 1.02= |
| FloorInt f32->i32 | 1.44W | 1.55W | 1.01= |
| GatherIndex i32 (4096) | 1.00= | 1.01= | · |
| GatherIndexN i32 (half n) | 1.00= | 1.01= | · |
| GatherIndexNOr i32 (half n) | 1.00= | 1.00= | · |
| GatherOffset i32 (4096) | 0.99= | 0.98= | · |
| Ge i32 | 0.98= | 1.00= | 0.99= |
| GetBiasedExponent f32 | 1.01= | 1.03= | 1.00= |
| GetExponent f32 | 1.00= | 0.99= | · |
| Gt i32 | 0.99= | 1.01= | 0.98= |
| HighestSetBitIndex u32 | 0.98= | 0.99= | 1.00= |
| IfNegativeThenElse f32 | 1.05W | 1.07W | 1.00= |
| IfNegativeThenElseZero f32 | 1.30W | 1.10W | 1.02= |
| IfNegativeThenZeroElse f32 | 1.36W | 1.11W | 0.99= |
| IfThenElse i32 | 1.00= | 1.01= | 1.00= |
| IfThenElseZero i32 | 0.97? | 1.00= | 1.04= |
| IfThenZeroElse i32 | 0.97? | 1.01= | 0.99? |
| InterleaveEven i32 | 1.00= | 1.00= | 1.01? |
| InterleaveLower i32 (4096) | 1.00= | 1.01= | · |
| InterleaveLowerBlocks i32 | 0.99= | 0.98? | · |
| InterleaveOdd i32 | 1.00= | 1.00= | 0.98? |
| InterleaveUpper i32 | 1.00= | 1.01= | 0.98= |
| InterleaveWholeLower i32 | 1.01= | 0.99= | 0.97= |
| InterleaveWholeUpper i32 | 1.00= | 1.00= | 1.00? |
| IsEitherNaN f32 | 0.98= | 1.01= | 1.00= |
| IsFinite f32 | 0.99= | 1.25W | 1.00= |
| IsInf f32 | 0.93L | 1.13W | 1.00= |
| IsNaN f32 | 1.00= | 1.02? | 1.00= |
| IsNegative i32 | 0.97= | 0.94? | 1.00= |
| Le i32 | 0.99= | 0.98? | 0.98= |
| LeadingZeroCount u32 (4096) | 0.95= | 1.01= | · |
| Load/Store i32 aligned | 0.99= | 1.01= | 0.99= |
| LoadExpand i32 (4096, 50%) | 1.75W | 2.12W | · |
| LoadInterleaved2 f32 (4096) | 1.00= | 1.00= | · |
| LoadInterleaved3 f32 (4096) | 0.99= | 0.99= | · |
| LoadInterleaved4 u8 (4096, RGBA) | 1.01= | 1.02= | · |
| LoadN/StoreN i32 (half lanes) | 1.58W | 1.03= | · |
| LoadNOr i32 (half n) | 1.15W | 1.06W | · |
| LoadU/StoreU i32 | 1.00= | 1.01= | 1.00= |
| Lt i32 | 0.99= | 0.99= | 1.00= |
| Lt128 u64 | 1.00= | 1.01= | 1.00= |
| Lt128Upper u64 | 1.00= | 1.00= | 0.97? |
| Mask And i32 | 1.01= | 1.01= | 1.00= |
| Mask AndNot i32 | 1.00= | 1.01= | 1.00= |
| Mask Not i32 | 1.00= | 0.99= | 1.00= |
| Mask Or i32 | 0.99= | 1.00= | 1.00= |
| Mask Xor i32 | 1.00= | 0.98= | 1.00= |
| MaskedApproxReciprocal f32 | 1.02= | 1.01= | 1.00= |
| MaskedApproxReciprocalSqrt f32 | 0.99= | 0.99= | 1.00= |
| MaskedConvertTo f32-i32 | 2.06W | 1.50W | 1.00= |
| MaskedLeadingZeroCount i32 | 1.00= | 1.01= | 1.02= |
| MaskedLoad i32 (4096, 50%) | 0.98= | 1.01= | · |
| MaskedLoadOr i32 (4096) | 1.00= | 1.02= | · |
| MaskedModOr i32 invariant | 0.98= | 0.98= | 1.06W |
| MaskedMulComplex f32 | 1.19W | 1.09W | 1.00= |
| MaskedMulComplexAdd f32 | 1.14W | 1.04= | 1.00= |
| MaskedMulComplexConj f32 | 1.21W | 1.08? | 1.00= |
| MaskedMulComplexConjAdd f32 | 1.14W | 1.01= | 1.00= |
| MaskedMulComplexOr f32 | 1.17W | 1.06W | 1.00= |
| MaskedMulFixedPoint15 i16 | 0.99= | 1.04? | 1.00= |
| MaskedReduceMax i32 | 1.00= | 0.99= | 1.00= |
| MaskedReduceMin i32 | 1.00= | 0.98= | 1.00= |
| MaskedReduceSum i32 | 1.00= | 0.96? | 1.00= |
| MaskedSatAddOr i16 | 1.00= | 0.99= | 1.00= |
| MaskedSatSubOr i16 | 0.99= | 0.98= | 1.00= |
| MaskedShiftLeft<3> i32 | 1.00= | 1.01= | 1.06= |
| MaskedShiftLeftOr<3> i32 | 1.00= | 0.99= | 1.00= |
| MaskedShiftRight<3> i32 | 0.99= | 1.00= | 1.01= |
| MaskedShiftRightOr<3> i32 | 1.00= | 1.02= | 1.00= |
| MaskedShl i32 | 0.99= | 1.02= | 0.99= |
| MaskedShlOr i32 | 1.00= | 1.02= | 1.00= |
| MaskedShr i32 | 0.99= | 0.99= | 0.99= |
| MaskedShrOr i32 | 1.01= | 1.00= | 1.00= |
| MaskedWidenMulPairwiseAdd i16-i32 | 1.00= | 0.96? | 1.00= |
| Max f32 | 1.01= | 1.00= | 1.00? |
| Max i32 | 1.00= | 1.00= | 0.99= |
| Max128 u64 | 0.99= | 0.99= | 1.00= |
| MaxMagnitude f32 | 0.99= | 0.99= | 1.00= |
| MaxNumber f32 | 0.99= | 1.01= | 0.99= |
| MaxOfLanes i32 | 0.95L | 1.17W | 0.92= |
| Min f32 | 1.00= | 0.99= | 1.03? |
| Min i32 (4096) | 1.00= | 1.00= | · |
| Min128 u64 | 1.00= | 1.01= | 1.00= |
| MinMagnitude f32 | 1.00= | 1.00= | 1.00= |
| MinNumber f32 | 1.01= | 1.01= | 0.95? |
| MinOfLanes i32 | 0.97L | 1.15W | 0.96= |
| Mod i32 invariant | 0.94? | 0.98= | 1.07W |
| Mul f32 | 1.00= | 0.99= | 1.06= |
| Mul i32 | 1.00= | 1.00= | 1.04= |
| MulAdd f32 (4096) | 1.00= | 0.99= | · |
| MulAddSub f32 | 1.00= | 1.01= | 0.88L |
| MulByFloorPow2 f32 | 1.06W | 0.97= | 1.00= |
| MulByPow2 f32 | 1.00= | 0.99= | 1.00= |
| MulComplex f32 | 1.44W | 1.19W | 1.00= |
| MulComplexAdd f32 | 1.02= | 1.03= | 1.00= |
| MulComplexConj f32 | 1.46W | 1.20W | 1.00= |
| MulComplexConjAdd f32 | 1.02= | 1.03? | 1.00= |
| MulEven i32-i64 | 1.00= | 1.06= | 1.00= |
| MulFixedPoint15 i16 | 1.00= | 1.00= | · |
| MulHigh i16 (4096) | 1.00= | 1.02= | · |
| MulHigh u16 (4096) | 1.00= | 1.00= | · |
| MulHigh32 u32 (4096) | 1.01= | 1.02= | · |
| MulOdd i32-i64 | 0.97= | 0.97= | 1.00= |
| MulRound f32 | 1.00= | 1.01= | 1.02= |
| MulSub f32 | 1.00= | 1.00= | 0.98= |
| MulSubAdd f32 | 1.01= | 1.00= | 0.90L |
| MultiRotateRight u64 (4096) | 1.00= | 1.00= | · |
| Ne i32 | 0.99= | 1.00= | 1.00= |
| Ne128 u64 | 1.00= | 1.00= | 0.97= |
| NearestInt f32->i32 | 1.43W | 1.10W | 1.00= |
| Neg i32 | 1.00= | 1.01= | 0.99? |
| NegMulAdd f32 | 1.01= | 1.00= | 0.98= |
| NegMulSub f32 | 1.01= | 0.99= | 1.01= |
| Not i32 | 0.96L | 1.00= | 1.02= |
| OddEven i32 | 1.00= | 1.00= | 1.02= |
| OddEvenBlocks i32 | 1.00= | 1.01? | · |
| Or i32 | 1.00= | 1.03= | 0.97= |
| Or3 i32 | 1.00= | 0.99= | 1.00= |
| OrAnd i32 | 1.00= | 1.02= | 1.00= |
| OrderedDemote2MasksTo i32->i16 (4096) | 0.99= | 0.97= | · |
| OrderedDemote2To i32->i16 | 0.99= | 1.00= | · |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.14W | 1.22W | · |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 1.00= | 0.99= | · |
| PairwiseAdd i32 | 1.10W | 1.01= | 0.99= |
| PairwiseAdd128 i32 | 0.99= | 0.99= | 1.00= |
| PairwiseSub i32 | 1.10W | 0.99= | 0.99= |
| PairwiseSub128 i32 | 1.09W | 0.96? | 1.00= |
| Per4LaneBlockShuffle 3210 i32 | 0.90? | 1.00= | 1.00= |
| PopulationCount u32 (4096) | 1.49W | 1.51W | · |
| PromoteEvenTo i8->i16 | 0.96= | 0.99= | · |
| PromoteLowerTo i16->i32 (4096) | 1.00= | 0.98= | · |
| PromoteMaskTo i16->i32 (4096) | 1.01= | 1.00= | · |
| PromoteOddTo i8->i16 | 0.99= | 0.99= | · |
| PromoteTo i16->i32 1-arg (4096) | 1.00= | 1.00= | · |
| PromoteUpperTo i16->i32 | 1.01= | 0.99= | · |
| RebindMask i32->f32 (4096) | 0.93? | 1.02= | · |
| ReduceMax i32 | 1.18W | 1.03= | 1.00= |
| ReduceMin i32 | 1.18W | 1.03= | 1.00= |
| ReduceSum f32 (4096) | 0.99= | 1.00= | · |
| ReorderDemote2To i32->i16 | 1.00= | 0.99= | · |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.05? | 1.17W | · |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 1.00= | 1.00= | · |
| ReorderWidenMulAccumulate i16->i32 | 1.00= | 1.01= | · |
| Reverse i32 (4096) | 1.00= | 1.02= | · |
| Reverse2 i32 | 1.00= | 1.00= | 1.04? |
| Reverse4 i32 | 1.00= | 1.00= | 1.10= |
| Reverse8 i32 (4096) | 0.99= | 0.99= | · |
| ReverseBits i32 | 2.09W | 1.94W | 1.04= |
| ReverseBlocks i32 | 0.96= | 0.99= | · |
| ReverseLaneBytes u32 (4096) | 1.02= | 1.01= | · |
| RotateLeft<3> i32 | 1.00= | 1.01? | 1.00= |
| RotateLeftSame u32 bits=5 | 0.99= | 1.01= | 0.97= |
| RotateRight<3> i32 | 0.99= | 0.99= | 1.05= |
| RotateRightSame u32 bits=5 | 1.00= | 0.99= | 1.01= |
| Round f32 | 1.00= | 1.01= | 0.99= |
| RoundToInt f32->i32 | 1.40W | 1.14? | 1.00= |
| RoundingShiftRight<2> i32 | 1.11? | 0.98= | 0.99= |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.99? | 1.11W | · |
| RoundingShiftRightSame i32 bits=5 | 1.09W | 1.00= | 0.82L |
| RoundingShr i32 | 1.05W | 0.95L | 1.00? |
| SatWidenMulAccumFixedPoint i16->i32 | 1.90W | 1.78W | · |
| SatWidenMulPairwiseAccumulate i16->i32 | 0.93L | 1.00= | · |
| SatWidenMulPairwiseAdd u8*i8->i16 | 0.97= | 1.00= | · |
| SaturatedAbs i32 | 1.02= | 0.99= | 1.03? |
| SaturatedAdd i16 (4096) | 1.00= | 0.98= | · |
| SaturatedNeg i32 | 1.01= | 0.99= | 1.08? |
| SaturatedSub i16 | 1.00= | 1.02= | 1.00= |
| ScatterIndex i32 (4096) | 1.02= | 1.01= | · |
| ScatterIndexN i32 (half n) | 1.27W | 1.02= | · |
| ScatterOffset i32 (4096) | 0.99? | 0.98= | · |
| Set i32 | 1.00= | 1.00= | 1.00= |
| SetAtOrAfterFirst i32 | 1.00= | 0.99= | 1.00= |
| SetAtOrBeforeFirst i32 | 0.88? | 1.18W | 1.00= |
| SetBeforeFirst i32 | 0.89L | 1.27W | 1.00= |
| SetMask i32 (4096) | 0.98= | 1.00= | · |
| SetOnlyFirst i32 | 0.95= | 1.65W | 1.00= |
| ShiftLeft<3> i32 | 1.00= | 0.99= | 1.03= |
| ShiftLeftBytes<4> i32 | 0.99= | 1.00= | 0.99= |
| ShiftLeftLanes<1> i32 | 1.00= | 1.00= | 1.00= |
| ShiftLeftSame i32 bits=3 | 0.98= | 1.00= | 1.91W |
| ShiftRight<3> i32 | 1.00= | 1.01= | 1.02= |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 1.01= | 1.00= | · |
| ShiftRightBytes<4> i32 | 1.01= | 1.00= | 0.99= |
| ShiftRightLanes<1> i32 | 1.02= | 0.99= | 1.00= |
| ShiftRightSame i32 bits=3 | 1.01= | 0.99= | 1.22W |
| Shl i32 | 0.99= | 1.00= | 1.03? |
| Shr u32 (variable) | 1.00= | 1.01= | 1.00= |
| Shuffle0123 i32 | 1.00= | 0.99= | 1.02= |
| Shuffle0321 i32 | 1.01= | 0.99= | 1.01= |
| Shuffle1032 i32 | 1.00= | 1.00= | 0.98= |
| Shuffle2103 i32 | 1.00= | 1.01= | 1.02= |
| Shuffle2301 i32 | 1.00= | 0.99= | 0.98= |
| Slide1Down i32 | 1.01= | 0.99= | 0.99= |
| Slide1Up i32 | 1.00= | 1.00= | 0.96? |
| SlideDownLanes amt=2 i32 | 1.00= | 1.01= | 1.00= |
| SlideMask1Down i32 (4096) | 1.10? | 1.02= | · |
| SlideMask1Up i32 (4096) | 1.08W | 0.99= | · |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.09W | 0.99= | · |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.09W | 1.04W | · |
| SlideUpLanes amt=2 i32 | 1.00= | 1.00= | 1.00= |
| Sqrt f32 | 1.00= | 1.00= | 1.00= |
| StoreInterleaved2 f32 (4096) | 1.00= | 1.03? | · |
| StoreInterleaved3 f32 (4096) | 1.03? | 1.15W | · |
| StoreInterleaved4 f32 (4096) | 1.49W | 1.26W | · |
| Sub f32 | 1.00= | 0.99= | 0.98= |
| Sub i32 | 1.00= | 1.03= | 1.01? |
| SumOfLanes i32 | 0.91L | 1.16W | 0.99= |
| SumOfMulQuadAccumulate i8->i32 | 1.00= | 1.02= | · |
| SumsOf2 u8->u16 | 0.98= | 1.00= | · |
| SumsOf4 u8->u32 | 0.99= | 0.98= | · |
| SumsOf8 u8->u64 | 1.00= | 1.01= | · |
| SumsOf8AbsDiff u8->u64 | 1.05? | 0.99= | · |
| SumsOfAdjQuadAbsDiff u8->u16 | 1.00= | 1.00= | · |
| SwapAdjacentBlocks i32 | 0.95= | 1.01= | · |
| TableLookupBytes u8 (4096) | 0.96= | 0.88L | · |
| TableLookupLanes i32 (4096) | 1.00= | 1.00= | · |
| TestBit i32 | 1.06? | 1.00= | 1.00= |
| TrailingZeroCount i32 | 1.00= | 1.00= | 1.02? |
| Trunc f32 | 1.00= | 1.01= | 0.98= |
| TruncateTo u32->u16 | 1.01= | 0.98= | 1.00= |
| TwoTablesLookupLanes i32 (4096) | 0.99= | 1.02= | · |
| WidenMulAccumulate i16->i32 | 1.00= | 0.99= | · |
| WidenMulPairwiseAdd i16->i32 | 1.00= | 0.98= | · |
| Xor i32 | 0.99= | 0.98= | 0.96= |
| Xor3 i32 | 1.61W | 0.98= | 1.01= |
| Zero i32 | 1.00= | 1.00= | 1.06= |
| ZeroExtendVector i16 half->full | 1.02= | 1.00= | · |
| ZeroIfNegative f32 | 1.04W | 1.06? | 0.97? |

## 复现
```
# 每平台:构建 hyle_op_parity{,_full,_widen}_bench_<isa>,双序各跑一遍(Mac 建议 4 遍),然后
python scripts/op-parity-report.py out_hf.txt out_hwf.txt          # x86(闸门 ±4%)
python scripts/op-parity-report.py --gate 0.06 mac_*.txt           # Mac(无亲和性,±6%)
# 报告首段为锚点校准;CALIBRATION PASS 之后的判决才可引用。
```
