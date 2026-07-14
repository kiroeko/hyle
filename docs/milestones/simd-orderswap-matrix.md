# SIMD op parity 三平台矩阵(2026-07-14 校准仪器 v2;四轮修复后 x86 双编译器交集清零)

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
### MSVC(AVX2): 13 个
- AddSub i32: **0.85** —— MSVC 0.86 / GCC 0.94(GCC 侧v10新进临界区),真实小差距
- SetAtOrBeforeFirst i32: **0.90**
- SumOfLanes i32: **0.91** —— 仅 MSVC(GCC 反赢);已从 0.78 改善到 0.91(in-vector broadcast-reduce),残余为 MSVC 水平归约 codegen
- SetBeforeFirst i32: **0.91** —— 仅 MSVC 0.89(本轮新进,GCC 1.3 反赢;borderline 待复核)
- RoundingShiftRightAndDemoteTo<3> i32->i16 (4096): **0.91**
- RebindMask i32->f32 (4096): **0.92**
- SatWidenMulPairwiseAccumulate i16->i32: **0.93**
- BroadcastLane<3> i32: **0.93** —— 已定论保留 vpermd:MSVC 0.93 是 bench 形状特定差距(Highway 的内存源 vbroadcasti128 折叠红利;寄存器输入下 vpermd 更优,GCC 反赢 1.13)
- IsInf f32: **0.93** —— 仅 MSVC 0.93
- Mod i32 invariant: **0.93**
- MaxOfLanes i32: **0.95** —— 仅 MSVC 0.95(已从 0.86 改善,边缘)
- CompressNot i32 (4096, 50%): **0.93** —— 已消向量 Not(bits 域 mbits^0xFF,commit 7c856fa,dumpbin=Compress+单条 `not eax`);独立双序复核(2026-07-14):MSVC 0.93 与 `Compress i32` 同轮 0.945 同速,GCC 1.02 ≥TIE——修复确认,残余=公共 compress 路径(table+srlv+vpermd)的 MSVC codegen,逐指令同 Highway,非 CompressNot 特有,疑绝对地址/table 布局级仪器残余,要啃是 Compress 的活
- CompressStore i32 (4096, 50%): **0.97** —— 已消重复 movemask(提一次 mbits 共享 compress+count,commit 647d254);MSVC 0.91→0.97 且双序 UNSTABLE(进打平抖动),GCC TIE。残余同 CompressNot=公共 compress32_by_bits 核(table+srlv+vpermd)的 MSVC codegen,非 CompressStore 特有

### WSL/GCC(AVX2): 1 个
- FindLastTrue i32: **0.94** —— 仅 GCC 0.94;objdump:两侧指令相同且循环顶同为 64B 对齐(GCC .p2align 吸收 stagger)-> 绝对地址级仪器残余,HYLE 代码无可修,非算法输
- (CompressNot i32 已于 commit 7c856fa 移出:bits 域补码消向量 Not,GCC 0.94→1.02 ≥TIE,独立双序复核确认)

### Mac/NEON: 2 个
- MulSubAdd f32: **0.87** —— 仅 NEON 0.90
- MulAddSub f32: **0.88** —— 仅 NEON 0.88

### x86 双编译器交集(最高优先级): 0 个
- (曾唯余 CompressNot i32,MSVC 0.81/GCC 0.94;commit 7c856fa 消向量 Not 后 GCC→1.02 ≥TIE、MSVC→0.93 且残余归 Compress 公共路径(同轮 Compress 本身 0.945),交集清零。独立双序复核确认。)
- 注:MSVC `Compress i32` 0.94 与 CompressNot 同源,但 GCC 侧 Compress TIE,故不构成双编译器交集。

### 与旧报告的对照(演进:旧 harness -> v1 校准仪器 -> 修复 + v2 stagger)
- 旧 op_parity 报 ~40-50 个 0.xx(MSVC 列);v1 校准仪器判 MSVC 21 / WSL 11 / Mac 3;两轮后端修复
  (第一轮:ReverseLaneBytes 0.05->TIE、Broadcast<0>、TruncateTo、SumOfLanes、Min/MaxOfLanes;
  第二轮:TZC 三平台追平、HSBI、DemoteMaskTo 0.33->1.4W、LoadN/StoreN 与 LoadNOr 弃 vpmaskmovd
  ->WIN、CombineMasks、ConcatLowerLower)+ v2 stagger 仪器修相位相关后,收敛为上表数字。
- v1 的假 LOSE 被 v2 消除的例子:TestBit / RebindMask / IsNegative(GCC)——同侧副本相位相关
  曾让它们稳定读 0.83~0.96,stagger 后回到 TIE/UNSTABLE。
- 第三轮(2026-07-14):RoundingShr 改 4-op bias=(1<<s)>>1(负 s 收窄 unspecified)MSVC 1.06W/GCC
  0.95->1.15W;NEON RoundingShiftRightSame 转发 vrshl(0.82->见上表);TableLookupBytesInRange 落地
  五后端+sve/rvv(InRange 家族惯例),对拍改为契约对等的 InRange-vs-Highway。
- DemoteToNearestInt 语义决策落地(2026-07-14):默认版弃 emu128 sign-based NaN(Highway 自己
  在任何硬件 target 都不遵守的参考语义,2.5x 代价无消费者),改为与 Highway x86 native 逐字节一致
  (Min+cvt;NaN 双符号->MAX、±溢出饱和,仍是全后端统一契约)。MSVC 0.61->1.00 / GCC 0.44->1.01 TIE;
  InRange 快路径不变(命名即 Highway 官方词汇)。7 个正确性闸门全绿(含 SDE avx512)。
- 本轮回归教训:ConcatLowerLower 的 vinsert 形态 GCC 赢但 MSVC 在复合上下文(Slide1Up)编成
  extract+栈 spill 掉到 0.099(单独 kernel 却 TIE)——复合上下文回归只有全表重测才抓得住,
  已按编译器分叉(MSVC=vperm2i128 / GCC=vinsert)修复,Slide1Up/SlideMask1Up 复位 TIE/WIN。
- 第四轮(2026-07-14,commit 7c856fa):CompressNot i32(曾唯一 x86 双编译器交集,MSVC 0.81/GCC 0.94)
  弃泛型 `Compress(v,Not(m))` 的向量 Not,改 Highway 的 bits 域补码(`mbits^0xFF` 复用同一 compress
  索引表,即 Highway IndicesFromNotBits);dumpbin 坐实 = Compress + 单条 `not eax`。GCC→TIE、
  MSVC→0.95(追到与 Compress 本身同速)。**x86 双编译器交集清零。** 副产:MSVC `Compress i32`
  本身 0.94(公共 table+srlv+vpermd 路径 codegen,逐指令同 Highway,GCC 侧 TIE)= CompressNot MSVC
  残余的真源,非 CompressNot 特有——要再啃是 Compress 的活,疑绝对地址/table 布局级仪器残余。
- 第四轮续(commit 647d254):CompressStore i32 消重复 movemask —— 原 `StoreU(Compress(v,m))+CountTrue(m)`
  对同一 mask 双提取,改提一次 mbits 共享 compress+count(Highway 同款单提取)。MSVC 0.91→0.97 UNSTABLE、
  GCC TIE。至此 Compress 三兄弟(Compress/CompressNot/CompressStore)可修部分(向量 Not、重复 movemask)
  全消,同进打平抖动区;MSVC 残余统归共享 compress32_by_bits 核的绝对地址/布局级仪器抖动,非算法输。

## 全量表

| op | MSVC(AVX2) | WSL GCC(AVX2) | Mac(NEON) |
|---|---|---|---|
| Abs i32 | 1.01= | 0.98= | 1.02= |
| AbsDiff i32 | 1.00= | 1.00= | 1.02= |
| Add f32 | 1.00= | 1.00= | 0.99= |
| Add f32 (4096) | 1.00= | 0.99= | · |
| AddSub i32 | 0.85L | 0.97? | 1.00? |
| AllBits0 i32 | 1.02= | 1.00= | 1.00= |
| AllBits1 i32 | 0.98= | 0.98= | 1.00= |
| AllFalse i32 | 0.99? | 1.01= | 1.00= |
| AllTrue i32 | 0.99= | 0.99= | 1.00? |
| And i32 | 1.00= | 0.99= | 0.97= |
| AndNot i32 | 1.00= | 1.01= | 1.02= |
| ApproximateReciprocal f32 | 1.00= | 1.01= | 1.00= |
| ApproximateReciprocalSqrt f32 | 1.00= | 0.99= | 1.00= |
| AverageRound u16 | 1.00= | 1.01= | 1.00= |
| BitCast f32->i32 | 0.99= | 1.00= | 1.00= |
| BitCast i32->f32 | 1.01= | 1.00= | 1.00= |
| BitShuffle u64 (4096) | 1.07W | 1.11W | · |
| BlendedStore i32 (4096, 50%) | 1.00= | 0.99= | · |
| Broadcast<0> i32 | 0.99= | 0.99= | 1.05= |
| BroadcastBlock<1> i32 | 1.03= | 1.01= | · |
| BroadcastLane<3> i32 | 0.93L | 1.14W | 1.00= |
| BroadcastSignBit i32 | 1.01= | 1.00= | 0.99= |
| Ceil f32 | 1.01= | 0.99= | 0.99= |
| CeilInt f32->i32 | 1.50W | 1.54W | 0.99= |
| Clamp i32 | 1.00= | 1.03? | 1.01= |
| CombineMasks i32 (4096) | 1.00= | 1.01= | · |
| CombineShiftRightBytes4 i32 | 1.04= | 0.99= | 1.00= |
| CombineShiftRightLanes1 i32 | 0.96= | 0.99= | 1.00= |
| ComplexConj f32 | 1.25W | 0.99= | 1.01= |
| Compress i32 (4096, 50% dense) | 0.94L | 1.00= | · |
| CompressBlendedStore i32 (4096, 50%) | 0.98? | 1.15W | · |
| CompressNot i32 (4096, 50%) | 0.93L | 1.02? | · |
| CompressStore i32 (4096, 50%) | 0.97? | 1.00= | · |
| ConcatEven i32 (4096) | 1.02= | 1.00? | · |
| ConcatLowerLower i32 | 0.97= | 0.99= | · |
| ConcatLowerUpper i32 | 0.92= | 0.98= | · |
| ConcatOdd i32 | 1.01= | 1.01= | 1.02= |
| ConcatUpperLower i32 | 1.04= | 1.01= | · |
| ConcatUpperUpper i32 | 0.96= | 0.99= | · |
| ConvertTo f32->i32 | 1.42W | 1.11W | 1.01= |
| ConvertTo i32->f32 | 0.99= | 0.99= | 1.01= |
| CopySign f32 | 1.00= | 1.00= | 0.98= |
| CountTrue i32 | 1.04= | 1.00= | 1.00= |
| DemoteMaskTo i32->i16 (4096) | 1.04= | 1.42W | · |
| DemoteTo i32->i16 (sat) | 1.88W | 1.89W | · |
| DemoteTo i32->i16 1-arg (4096) | 1.00= | 1.00= | · |
| DemoteToNearestInt f64->i32 (4096) | 1.01= | 0.99= | · |
| Div f32 | 1.00= | 1.01= | 1.00= |
| Dup128MaskFromMaskBits i32 (4096) | 1.02= | 1.19W | · |
| DupEven i32 | 1.00= | 1.00= | 0.99= |
| DupOdd i32 | 1.00= | 1.01= | 1.07= |
| Eq i32 | 0.99= | 1.02= | 1.04= |
| Eq128 u64 | 1.01= | 1.00= | 0.99= |
| ExclusiveNeither i32 | 1.01= | 0.98= | 1.00= |
| Expand i32 (4096, 50%) | 1.99W | 2.13W | · |
| FindFirstTrue i32 | 0.99= | 1.00= | 1.05= |
| FindKnownFirstTrue i32 | 0.97= | 1.00= | 1.18W |
| FindKnownLastTrue i32 | 1.08W | 1.01= | 1.12W |
| FindLastTrue i32 | 1.02= | 0.94L | 1.16W |
| Floor f32 | 1.01= | 1.00= | 1.01= |
| FloorInt f32->i32 | 1.49W | 1.55W | 0.99= |
| GatherIndex i32 (4096) | 1.00= | 1.01= | · |
| GatherIndexN i32 (half n) | 1.00= | 0.97= | · |
| GatherIndexNOr i32 (half n) | 1.00= | 1.00= | · |
| GatherOffset i32 (4096) | 1.00= | 1.00= | · |
| Ge i32 | 0.99= | 1.00= | 0.98= |
| GetBiasedExponent f32 | 1.00= | 1.03= | 1.00= |
| GetExponent f32 | 1.00= | 0.99= | · |
| Gt i32 | 1.00= | 1.02= | 1.01= |
| HighestSetBitIndex u32 | 1.02= | 1.01= | 1.00= |
| IfNegativeThenElse f32 | 1.05W | 1.10W | 1.01= |
| IfNegativeThenElseZero f32 | 1.35W | 1.08W | 0.97= |
| IfNegativeThenZeroElse f32 | 1.34W | 1.09W | 1.00= |
| IfThenElse i32 | 1.00= | 1.02= | 1.00= |
| IfThenElseZero i32 | 0.97? | 0.99= | 1.00? |
| IfThenZeroElse i32 | 0.97= | 1.01= | 0.95= |
| InterleaveEven i32 | 1.00= | 1.01= | 1.03? |
| InterleaveLower i32 (4096) | 1.00= | 1.00= | · |
| InterleaveLowerBlocks i32 | 0.92= | 0.99= | · |
| InterleaveOdd i32 | 1.01= | 1.00= | 0.98= |
| InterleaveUpper i32 | 1.00= | 1.02= | 1.00= |
| InterleaveWholeLower i32 | 1.00= | 1.00= | 0.98? |
| InterleaveWholeUpper i32 | 0.99= | 0.98= | 1.02= |
| IsEitherNaN f32 | 1.00= | 0.99= | 1.00= |
| IsFinite f32 | 1.00= | 1.33W | 1.00= |
| IsInf f32 | 0.93L | 1.14W | 1.00= |
| IsNaN f32 | 1.01= | 1.00= | 1.00? |
| IsNegative i32 | 0.97? | 0.94? | 1.00= |
| Le i32 | 1.00= | 1.00= | 0.98= |
| LeadingZeroCount u32 (4096) | 0.95? | 1.01= | · |
| Load/Store i32 aligned | 0.99= | 1.00= | 0.99= |
| LoadExpand i32 (4096, 50%) | 2.00W | 2.12W | · |
| LoadInterleaved2 f32 (4096) | 1.00= | 1.00= | · |
| LoadInterleaved3 f32 (4096) | 1.00= | 0.99= | · |
| LoadInterleaved4 u8 (4096, RGBA) | 1.00= | 1.00= | · |
| LoadN/StoreN i32 (half lanes) | 1.59W | 1.06W | · |
| LoadNOr i32 (half n) | 1.15? | 1.05? | · |
| LoadU/StoreU i32 | 1.01= | 0.99= | 1.00? |
| Lt i32 | 0.99= | 1.02= | 0.99= |
| Lt128 u64 | 0.99= | 1.00= | 1.00= |
| Lt128Upper u64 | 1.01= | 1.01= | 0.95? |
| Mask And i32 | 1.00= | 1.00= | 1.00= |
| Mask AndNot i32 | 1.00= | 1.01= | 1.00= |
| Mask Not i32 | 1.00= | 0.98= | 1.00= |
| Mask Or i32 | 1.00= | 1.00= | 1.00= |
| Mask Xor i32 | 1.00= | 1.00= | 1.00= |
| MaskedApproxReciprocal f32 | 1.00= | 1.00= | 1.00= |
| MaskedApproxReciprocalSqrt f32 | 1.00= | 0.99= | 1.00= |
| MaskedConvertTo f32-i32 | 2.06W | 1.50W | 0.99= |
| MaskedLeadingZeroCount i32 | 1.00= | 1.01= | 1.01= |
| MaskedLoad i32 (4096, 50%) | 0.98= | 0.99= | · |
| MaskedLoadOr i32 (4096) | 1.01= | 1.00= | · |
| MaskedModOr i32 invariant | 0.97= | 0.98= | 1.06W |
| MaskedMulComplex f32 | 1.20W | 1.09W | 1.00= |
| MaskedMulComplexAdd f32 | 1.14W | 1.02= | 1.00= |
| MaskedMulComplexConj f32 | 1.21W | 1.08W | 1.00= |
| MaskedMulComplexConjAdd f32 | 1.14W | 1.00= | 1.00= |
| MaskedMulComplexOr f32 | 1.16W | 1.07W | 1.00= |
| MaskedMulFixedPoint15 i16 | 0.99= | 1.00= | 1.00= |
| MaskedReduceMax i32 | 1.00= | 1.00= | 1.00= |
| MaskedReduceMin i32 | 1.00= | 1.00= | 1.00= |
| MaskedReduceSum i32 | 1.00= | 1.00= | 1.00= |
| MaskedSatAddOr i16 | 1.00= | 1.01= | 1.00= |
| MaskedSatSubOr i16 | 1.00= | 1.00= | 1.00= |
| MaskedShiftLeft<3> i32 | 0.99= | 0.99= | 0.99= |
| MaskedShiftLeftOr<3> i32 | 1.02= | 0.99= | 1.00? |
| MaskedShiftRight<3> i32 | 0.99= | 0.99= | 0.99= |
| MaskedShiftRightOr<3> i32 | 1.00= | 0.99= | 1.00? |
| MaskedShl i32 | 1.00= | 0.99= | 0.99? |
| MaskedShlOr i32 | 1.00= | 0.99= | 1.00= |
| MaskedShr i32 | 1.00= | 1.00= | 1.01= |
| MaskedShrOr i32 | 0.99= | 1.01= | 1.00= |
| MaskedWidenMulPairwiseAdd i16-i32 | 0.98= | 1.00= | 1.00= |
| Max f32 | 1.00= | 1.00= | 1.06? |
| Max i32 | 1.00= | 1.01= | 1.03= |
| Max128 u64 | 0.99= | 0.97= | 1.00= |
| MaxMagnitude f32 | 1.00= | 0.99= | 1.00= |
| MaxNumber f32 | 0.99= | 1.00= | 0.98= |
| MaxOfLanes i32 | 0.95L | 1.19W | 0.97? |
| Min f32 | 1.01= | 1.02= | 1.05= |
| Min i32 (4096) | 1.00= | 1.01= | · |
| Min128 u64 | 0.98= | 1.02= | 1.00= |
| MinMagnitude f32 | 1.00= | 1.00= | 1.00= |
| MinNumber f32 | 1.02= | 1.00= | 0.99= |
| MinOfLanes i32 | 0.96? | 1.16W | 1.07= |
| Mod i32 invariant | 0.93L | 0.99= | 1.07W |
| Mul f32 | 1.01= | 0.99= | 1.01= |
| Mul i32 | 0.99= | 1.02= | 0.98= |
| MulAdd f32 (4096) | 1.00= | 1.01= | · |
| MulAddSub f32 | 0.99= | 0.99= | 0.88L |
| MulByFloorPow2 f32 | 1.06W | 0.98= | 1.00= |
| MulByPow2 f32 | 1.00= | 1.00= | 1.00= |
| MulComplex f32 | 1.44W | 1.19W | 1.00= |
| MulComplexAdd f32 | 1.02= | 1.01= | 1.00= |
| MulComplexConj f32 | 1.46W | 1.20W | 1.00= |
| MulComplexConjAdd f32 | 1.02= | 1.03= | 1.00= |
| MulEven i32-i64 | 1.00= | 1.02= | 1.00= |
| MulFixedPoint15 i16 | 1.00= | 1.00= | · |
| MulHigh i16 (4096) | 1.00= | 0.99= | · |
| MulHigh u16 (4096) | 1.00= | 1.01= | · |
| MulHigh32 u32 (4096) | 1.00= | 1.01= | · |
| MulOdd i32-i64 | 0.97= | 0.91= | 1.00= |
| MulRound f32 | 1.00= | 0.99= | 1.01? |
| MulSub f32 | 1.00= | 0.99= | 0.98? |
| MulSubAdd f32 | 1.02= | 1.00= | 0.87L |
| MultiRotateRight u64 (4096) | 0.99= | 1.00= | · |
| Ne i32 | 1.00= | 1.00= | 1.00= |
| Ne128 u64 | 0.99= | 1.00= | 0.98? |
| NearestInt f32->i32 | 1.35W | 1.09? | 0.99= |
| Neg i32 | 0.99= | 0.99= | 1.05? |
| NegMulAdd f32 | 1.00= | 0.99= | 0.99? |
| NegMulSub f32 | 1.00= | 0.98= | 0.98? |
| Not i32 | 0.99= | 1.00= | 1.00= |
| OddEven i32 | 0.99= | 1.01? | 1.01= |
| OddEvenBlocks i32 | 1.02= | 0.99= | · |
| Or i32 | 0.99= | 1.03= | 1.01= |
| Or3 i32 | 0.99= | 1.00= | 1.00= |
| OrAnd i32 | 1.00= | 0.99= | 1.00= |
| OrderedDemote2MasksTo i32->i16 (4096) | 0.96? | 1.00= | · |
| OrderedDemote2To i32->i16 | 0.99= | 1.00= | · |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.14W | 1.22W | · |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 1.00= | 0.99= | · |
| PairwiseAdd i32 | 1.10W | 1.00= | 1.00= |
| PairwiseAdd128 i32 | 1.00= | 1.00= | 1.02= |
| PairwiseSub i32 | 1.11W | 0.98? | 1.01= |
| PairwiseSub128 i32 | 1.10W | 1.00= | 1.01= |
| Per4LaneBlockShuffle 3210 i32 | 0.87? | 1.00= | 1.00= |
| PopulationCount u32 (4096) | 1.48W | 1.49W | · |
| PromoteEvenTo i8->i16 | 0.96= | 0.99= | · |
| PromoteLowerTo i16->i32 (4096) | 1.00= | 1.00= | · |
| PromoteMaskTo i16->i32 (4096) | 1.00= | 1.00= | · |
| PromoteOddTo i8->i16 | 0.99= | 0.99= | · |
| PromoteTo i16->i32 1-arg (4096) | 0.99= | 0.99= | · |
| PromoteUpperTo i16->i32 | 1.01= | 0.99= | · |
| RebindMask i32->f32 (4096) | 0.92L | 0.99= | · |
| ReduceMax i32 | 1.17W | 1.02= | 1.00= |
| ReduceMin i32 | 1.17W | 1.03? | 1.00= |
| ReduceSum f32 (4096) | 0.98= | 1.00= | · |
| ReorderDemote2To i32->i16 | 1.00= | 0.99= | · |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.05? | 1.17W | · |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 1.00= | 1.00= | · |
| ReorderWidenMulAccumulate i16->i32 | 1.00= | 1.01= | · |
| Reverse i32 (4096) | 1.00= | 1.00= | · |
| Reverse2 i32 | 1.00= | 1.01= | 1.01? |
| Reverse4 i32 | 1.00= | 1.01= | 1.00= |
| Reverse8 i32 (4096) | 1.00= | 1.01= | · |
| ReverseBits i32 | 2.09W | 1.95W | 1.03= |
| ReverseBlocks i32 | 0.99= | 0.98= | · |
| ReverseLaneBytes u32 (4096) | 1.02= | 1.00= | · |
| RotateLeft<3> i32 | 0.99= | 1.00= | 1.02= |
| RotateLeftSame u32 bits=5 | 1.00= | 0.99= | 0.97= |
| RotateRight<3> i32 | 0.98= | 1.01= | 0.94= |
| RotateRightSame u32 bits=5 | 1.00= | 1.00= | 1.01= |
| Round f32 | 1.00= | 1.00= | 1.04= |
| RoundToInt f32->i32 | 1.38W | 1.14W | 1.00= |
| RoundingShiftRight<2> i32 | 1.08? | 0.99= | 0.99= |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.91L | 1.12W | · |
| RoundingShiftRightSame i32 bits=5 | 1.10W | 1.00= | 1.46W |
| RoundingShr i32 | 1.48W | 1.15W | 1.00? |
| SatWidenMulAccumFixedPoint i16->i32 | 1.90W | 1.78W | · |
| SatWidenMulPairwiseAccumulate i16->i32 | 0.93L | 1.00= | · |
| SatWidenMulPairwiseAdd u8*i8->i16 | 0.97= | 1.00= | · |
| SaturatedAbs i32 | 1.00= | 1.00= | 1.01= |
| SaturatedAdd i16 (4096) | 1.00= | 1.00= | · |
| SaturatedNeg i32 | 1.00= | 1.02= | 0.99= |
| SaturatedSub i16 | 1.00= | 1.00= | 1.00= |
| ScatterIndex i32 (4096) | 0.99? | 1.00= | · |
| ScatterIndexN i32 (half n) | 1.20W | 1.31W | · |
| ScatterOffset i32 (4096) | 0.99= | 0.99= | · |
| Set i32 | 1.00= | 1.00= | 1.00? |
| SetAtOrAfterFirst i32 | 1.02= | 0.99= | 1.00= |
| SetAtOrBeforeFirst i32 | 0.90L | 1.16W | 1.00= |
| SetBeforeFirst i32 | 0.91L | 1.26W | 1.00= |
| SetMask i32 (4096) | 1.00= | 1.01= | · |
| SetOnlyFirst i32 | 0.92= | 1.65W | 1.00? |
| ShiftLeft<3> i32 | 1.01= | 0.99= | 1.02= |
| ShiftLeftBytes<4> i32 | 1.02= | 0.97? | 1.00= |
| ShiftLeftLanes<1> i32 | 1.00= | 0.99= | 1.00= |
| ShiftLeftSame i32 bits=3 | 1.00= | 1.02= | 1.92W |
| ShiftRight<3> i32 | 1.00= | 0.99= | 0.97= |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 1.00= | 0.99= | · |
| ShiftRightBytes<4> i32 | 0.99= | 1.01= | 1.00= |
| ShiftRightLanes<1> i32 | 1.02= | 1.00= | 1.00= |
| ShiftRightSame i32 bits=3 | 1.00= | 1.02= | 1.21W |
| Shl i32 | 1.00= | 1.02= | 0.99= |
| Shr u32 (variable) | 1.00= | 1.01= | 1.00= |
| Shuffle0123 i32 | 1.00= | 0.98= | 1.04= |
| Shuffle0321 i32 | 1.00= | 1.00= | 0.97= |
| Shuffle1032 i32 | 1.00= | 0.98= | 1.02= |
| Shuffle2103 i32 | 1.01= | 1.01= | 1.09? |
| Shuffle2301 i32 | 1.00= | 1.00= | 0.96? |
| Slide1Down i32 | 1.00= | 0.98? | 1.02? |
| Slide1Up i32 | 1.00= | 0.99= | 1.02= |
| SlideDownLanes amt=2 i32 | 0.95= | 1.01= | 1.00= |
| SlideMask1Down i32 (4096) | 1.08W | 1.01= | · |
| SlideMask1Up i32 (4096) | 1.06W | 1.00= | · |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.11W | 0.99= | · |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.10W | 1.00= | · |
| SlideUpLanes amt=2 i32 | 1.00= | 0.98= | 0.99= |
| Sqrt f32 | 1.00= | 1.00= | 1.00= |
| StoreInterleaved2 f32 (4096) | 1.00= | 1.01= | · |
| StoreInterleaved3 f32 (4096) | 1.03= | 0.91? | · |
| StoreInterleaved4 f32 (4096) | 1.48W | 1.01? | · |
| Sub f32 | 1.00= | 1.00= | 1.02? |
| Sub i32 | 1.01= | 1.03= | 1.03= |
| SumOfLanes i32 | 0.91L | 1.16W | 0.97= |
| SumOfMulQuadAccumulate i8->i32 | 1.00= | 1.02= | · |
| SumsOf2 u8->u16 | 0.98= | 1.00= | · |
| SumsOf4 u8->u32 | 0.99= | 0.98= | · |
| SumsOf8 u8->u64 | 1.00= | 1.01= | · |
| SumsOf8AbsDiff u8->u64 | 1.05? | 0.99= | · |
| SumsOfAdjQuadAbsDiff u8->u16 | 1.00= | 1.00= | · |
| SwapAdjacentBlocks i32 | 0.98= | 1.01= | · |
| TableLookupBytes u8 (4096) | 1.00= | 1.00= | · |
| TableLookupLanes i32 (4096) | 0.99= | 1.01= | · |
| TestBit i32 | 1.05W | 1.00= | 1.00? |
| TrailingZeroCount i32 | 0.99= | 1.00= | 1.04= |
| Trunc f32 | 1.00= | 1.01= | 1.07? |
| TruncateTo u32->u16 | 1.01= | 1.01= | 1.00= |
| TwoTablesLookupLanes i32 (4096) | 1.01= | 0.98= | · |
| WidenMulAccumulate i16->i32 | 1.00= | 0.99= | · |
| WidenMulPairwiseAdd i16->i32 | 1.00= | 0.98= | · |
| Xor i32 | 0.99= | 1.01= | 1.02= |
| Xor3 i32 | 1.63W | 1.00= | 1.04? |
| Zero i32 | 1.00= | 1.00= | 0.95= |
| ZeroExtendVector i16 half->full | 1.02= | 1.00= | · |
| ZeroIfNegative f32 | 1.02? | 1.09W | 1.01= |

## 复现
```
# 每平台:构建 hyle_op_parity{,_full,_widen}_bench_<isa>,双序各跑一遍(Mac 建议 4 遍),然后
python scripts/op-parity-report.py out_hf.txt out_hwf.txt          # x86(闸门 ±4%)
python scripts/op-parity-report.py --gate 0.06 mac_*.txt           # Mac(无亲和性,±6%)
# 报告首段为锚点校准;CALIBRATION PASS 之后的判决才可引用。
```
