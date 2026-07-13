# SIMD op parity 三平台矩阵(2026-07-13 校准仪器全量重测)

> 本表由重建后的测量引擎产出(`bench/simd/op_parity_engine.h` + `scripts/op-parity-report.py`),
> 取代旧的"原始读数+人工伪影标签"体系。旧 op_parity 的数字被证明携带三层测量伪影(P/E 调度、
> 代码放置运气、闭包栈 4K 别名,合计可达 ±30~40%),单条 `vpsubd` 的 `Neg i32` 曾被它测成
> 0.75~1.49 之间的任意值。完整证据链与重建过程:[`../debug/op-parity-harness-artifacts-hunt.md`](../debug/op-parity-harness-artifacts-hunt.md)。

## 方法论(数字为何可信)
- **测量**:钉一颗 P 核+高优先级;每 op 每侧 4 个独立实例化的代码副本(采样放置运气,防 ICF 折叠);
  H/G 交替轮次;kernel 指针外提(热循环零闭包访问);宏形状循环 4x 展开。
- **聚合**:best-of-replicas(放置理想值)+ median + 逐轮配对比值三估计器;判决需三者在每个输入
  运行中一致越线,且线宽随该 op 实测副本散布自适应(>= ±3%)。每平台喂双序(HYLE 先/Highway 先)。
- **校准闸门**:5 个源码级验证"两侧同指令"的锚点 op(真值=1.00)必须全部落在闸门内,否则整表拒绝输出。
  本表锚点读数:MSVC/WSL 0.99~1.03(闸门 ±4%),Mac 0.95~1.05(闸门 ±6%,无亲和性 API,列分辨率较低,
  已用 4 次运行合并 + 8x epoch 补偿)。
- **判读**:数字 = HYLE/Highway 配对中位比值(>1=HYLE 快);W=WIN,L=LOSE,==TIE,?=UNSTABLE(多运行判决不一致),
  ·=该平台无此 op。UNSTABLE 一律不算输赢。Mac 列只覆盖 full 形状集(NEON 无 parity/widen 目标)。

## 真输清单(校准仪器判决,按比值升序)
### MSVC(AVX2): 14 个真输(本轮又修 4)+ BroadcastLane<3>(放置伪影,不计)
> 原 21 → 已修/移出:Broadcast<0>(vpbroadcastd)、TruncateTo u32->u16(2-shuffle)、**ReverseLaneBytes
> (vpshufb 0.05→1.00)、MaxOfLanes/MinOfLanes(in-vector reduce 0.86→0.96 TIE)**;BroadcastLane<3> 判放置
> 伪影不计;SumOfLanes 0.78→0.90(改善,仍列)。**= 14 个真算法输**(SumOfLanes 含在内,余见下)。
- ~~ReverseLaneBytes u32: 0.05~~ → **已修 1.00**(92f83f1):scalar store+byte-loop+reload 改单 vpshufb(复用 ReverseBits 字节反转表),MSVC 0.05→0.999 全平台 TIE。
- DemoteToNearestInt f64->i32 (4096): **0.38** —— 已知正确性取舍:HYLE safe 版做 emu128 谱系 NaN/饱和(Highway 原生只 Min+cvt);InRange 快路径同表 TIE
- ~~Broadcast<0> i32: 0.52~~ → **已修 1.00**(5b98ff8):store+reload 兜底改 native;lane 0 用 vpbroadcastd,全平台 TIE。
- BroadcastLane<3> i32: **0.90(MSVC full_bench,判为放置伪影非指令输 —— 见下调研)** —— d5f0f60 store+reload
  兜底改 native 单 vpermd 全局 lane-splat(WSL 1.14-1.17 赢)。MSVC full_bench 仍 0.90 LOSE(钉核+双序+
  CALIBRATION PASS)。**深入调研定位(2026-07-13)**:dumpbin 显示 HYLE 循环 = 4×(vpermd 折 256b load + store),
  最小指令、索引常量已提循环外、无 spill;Highway = 4×(vbroadcasti128 128b-load + vpshufd + store)。
  **决定性实验**:把 HYLE 改成与 Highway 逐字节相同的 vbroadcasti128+vpshufd(变体 C),MSVC full_bench 仍
  **0.898 LOSE**(几乎不动)——即**同一指令序列在 HYLE 的 full_bench 位置 vs Highway 的位置就差 ~10%**,
  证明这不是指令/算法差,而是猎人日志记的 **SELF=2 代码放置伪影**(同文本不同地址 ±10-15%,4-副本采样对个别
  op 未完全抵消)。⚠️ 隔离 TU 无法复核(每次锚点 Neg 都 OUT-OF-GATE 1.3-1.46=小 TU 每-op 放置方差过大)。
  **结论:vpermd 是全局广播最小指令,无更快指令可换;残留 0.90 = full_bench 位置噪声,非真算法输**。保留单
  vpermd(最简)。Broadcast<0> 用 vpbroadcastd 全平台 TIE 不受影响。bench 对照已改 hn::BroadcastLane(全局等价)。
- DemoteMaskTo i32->i16 (4096): **0.70** —— mask 宽度桥接冷 op,残余为寄存器搬运差(源注释已记录,接受)
- SumOfLanes i32: 0.78 → **0.90**(5dff3b6 去 GPR round-trip,in-vector broadcast-reduce):残留同 BroadcastLane<3> 放置噪声类
- HighestSetBitIndex u32: **0.84** —— 同 TZC 家族,x86 双编译器 0.84/0.87
- AddSub i32: **0.84**
- TruncateTo u32->u16: **0.85** —— 源码级确认:correctness-first store+reload(avx2.h 注释自认"native pack 是 profiling upgrade")
- ~~MaxOfLanes i32: 0.86~~ → **0.96 TIE**(5dff3b6 同 SumOfLanes)
- ~~MinOfLanes i32: 0.86~~ → **0.96 TIE**(5dff3b6 同 SumOfLanes)
- TrailingZeroCount i32: **0.87** —— 三平台唯一全输 op(0.87/0.87/0.68):HYLE 浮点指数技巧序列比 Highway 的对应序列贵
- CompressNot i32 (4096, 50%): **0.88**
- Per4LaneBlockShuffle 3210 i32: **0.89**
- TestBit i32: **0.92**
- RoundingShiftRightAndDemoteTo<3> i32->i16 (4096): **0.93**
- SatWidenMulPairwiseAccumulate i16->i32: **0.93**
- IsInf f32: **0.93**
- CompressStore i32 (4096, 50%): **0.94**
- RebindMask i32->f32 (4096): **0.95**
- Not i32: **0.96**

### WSL/GCC(AVX2): 11 个
- DemoteToNearestInt f64->i32 (4096): **0.26** —— 已知正确性取舍:HYLE safe 版做 emu128 谱系 NaN/饱和(Highway 原生只 Min+cvt);InRange 快路径同表 TIE
- DemoteMaskTo i32->i16 (4096): **0.32** —— mask 宽度桥接冷 op,残余为寄存器搬运差(源注释已记录,接受)
- LoadNOr i32 (half n): **0.54** —— 仅 GCC 输 0.54(旧文档曾据有偏复核"证伪不是真输",校准仪器推翻该免罪)
- LoadN/StoreN i32 (half lanes): **0.65** —— 仅 GCC 输 0.65,与 LoadNOr 同族(partial load/store 路径)
- IsNegative i32: **0.83**
- TruncateTo u32->u16: **0.83** —— 源码级确认:correctness-first store+reload(avx2.h 注释自认"native pack 是 profiling upgrade")
- TableLookupBytes u8 (4096): **0.85** —— HYLE 多一条设计内的 vpand 索引掩码(安全语义,非缺陷);曾被误当校准锚点,已剔除
- CombineMasks i32 (4096): **0.86**
- HighestSetBitIndex u32: **0.87** —— 同 TZC 家族,x86 双编译器 0.84/0.87
- TrailingZeroCount i32: **0.87** —— 三平台唯一全输 op(0.87/0.87/0.68):HYLE 浮点指数技巧序列比 Highway 的对应序列贵
- StoreInterleaved3 f32 (4096): **0.90**

### Mac/NEON: 3 个
- TrailingZeroCount i32: **0.68** —— 三平台唯一全输 op(0.87/0.87/0.68):HYLE 浮点指数技巧序列比 Highway 的对应序列贵
- RoundingShiftRightSame i32 bits=5: **0.82**
- MulSubAdd f32: **0.90**

### x86 双编译器交集(最高优先级): 5 个
- DemoteMaskTo i32->i16 (4096): MSVC 0.70 / WSL 0.32
- DemoteToNearestInt f64->i32 (4096): MSVC 0.38 / WSL 0.26
- HighestSetBitIndex u32: MSVC 0.84 / WSL 0.87
- TrailingZeroCount i32: MSVC 0.87 / WSL 0.87 (Mac 也输 0.68 -> 三平台全输)
- TruncateTo u32->u16: MSVC 0.85 / WSL 0.83

### 与旧报告的对照
- 旧 op_parity 报 ~40-50 个 0.xx(MSVC 列)-> 校准后 MSVC 真输 21、WSL 11、Mac 3;其余全是伪影。
- 旧"隔离复核"的 7 真输里,BitShuffle/SetAtOrAfterFirst/Lt128Upper/Ne128/MultiRotateRight 已被此前的
  修复收回(本表均 TIE/WIN,修复有效性经校准仪器确认);LoadNOr 的"证伪免罪"被推翻(WSL 0.54 真输);
  InterleaveLowerBlocks/TestBit 降为临界(0.92~0.97,UNSTABLE/边缘)。

## 全量表

| op | MSVC(AVX2) | WSL GCC(AVX2) | Mac(NEON) |
|---|---|---|---|
| Abs i32 | 1.00= | 0.97= | 1.02= |
| AbsDiff i32 | 1.00= | 1.00= | 0.99= |
| Add f32 | 1.00= | 0.99= | 1.01= |
| Add f32 (4096) | 1.00= | 1.01= | · |
| AddSub i32 | 0.84L | 1.02? | 1.00= |
| AllBits0 i32 | 1.01= | 1.00= | 1.00= |
| AllBits1 i32 | 1.01= | 1.00= | 1.00= |
| AllFalse i32 | 0.99= | 1.00= | 1.00= |
| AllTrue i32 | 0.99= | 1.00= | 1.00= |
| And i32 | 0.99= | 0.99= | 1.01= |
| AndNot i32 | 0.98= | 0.97= | 0.97? |
| ApproximateReciprocal f32 | 1.00= | 1.00= | 1.00= |
| ApproximateReciprocalSqrt f32 | 1.00= | 1.00= | 1.00= |
| AverageRound u16 | 1.00= | 1.00= | 1.00= |
| BitCast f32->i32 | 1.00= | 0.99= | 1.00= |
| BitCast i32->f32 | 1.00= | 0.99= | 1.00= |
| BitShuffle u64 (4096) | 1.07W | 1.10W | · |
| BlendedStore i32 (4096, 50%) | 0.99= | 1.00= | · |
| Broadcast<0> i32 | 1.00= | 1.00= | 1.02= |
| BroadcastBlock<1> i32 | 0.98= | 1.00= | · |
| BroadcastLane<3> i32 | 0.91L | 1.14W | 0.98= |
| BroadcastSignBit i32 | 1.00= | 1.01= | 0.98= |
| Ceil f32 | 1.00= | 1.00= | 1.00= |
| CeilInt f32->i32 | 1.50W | 1.55W | 1.01= |
| Clamp i32 | 1.00= | 0.99= | 0.98= |
| CombineMasks i32 (4096) | 1.04? | 0.86L | · |
| CombineShiftRightBytes4 i32 | 1.00= | 1.01= | 0.99= |
| CombineShiftRightLanes1 i32 | 1.00= | 1.00= | 1.00= |
| ComplexConj f32 | 1.25W | 1.01= | 1.02? |
| Compress i32 (4096, 50% dense) | 0.98= | 1.00= | · |
| CompressBlendedStore i32 (4096, 50%) | 1.01= | 1.23W | · |
| CompressNot i32 (4096, 50%) | 0.88L | 0.97= | · |
| CompressStore i32 (4096, 50%) | 0.94L | 1.00= | · |
| ConcatEven i32 (4096) | 0.99= | 1.01? | · |
| ConcatLowerLower i32 | 1.02= | 0.95? | · |
| ConcatLowerUpper i32 | 1.00= | 1.00= | · |
| ConcatOdd i32 | 1.00= | 0.99= | 0.98= |
| ConcatUpperLower i32 | 1.00= | 1.01= | · |
| ConcatUpperUpper i32 | 1.01= | 1.00= | · |
| ConvertTo f32->i32 | 1.40W | 1.11W | 1.00= |
| ConvertTo i32->f32 | 1.00= | 1.00= | 1.00= |
| CopySign f32 | 1.00= | 0.99= | 1.02= |
| CountTrue i32 | 1.00= | 1.00= | 1.00= |
| DemoteMaskTo i32->i16 (4096) | 0.70L | 0.32L | · |
| DemoteTo i32->i16 (sat) | 1.79W | 1.90W | · |
| DemoteTo i32->i16 1-arg (4096) | 1.00= | 1.02= | · |
| DemoteToNearestInt f64->i32 (4096) | 0.38L | 0.26L | · |
| Div f32 | 1.00= | 1.03? | 0.99= |
| Dup128MaskFromMaskBits i32 (4096) | 1.06? | 1.19W | · |
| DupEven i32 | 1.00= | 1.01= | 1.03= |
| DupOdd i32 | 1.00= | 1.00= | 0.99= |
| Eq i32 | 0.99= | 1.00= | 1.00= |
| Eq128 u64 | 1.00= | 1.00= | 1.00? |
| ExclusiveNeither i32 | 1.00= | 0.97= | 1.00= |
| Expand i32 (4096, 50%) | 1.78W | 2.12W | · |
| FindFirstTrue i32 | 1.03? | 1.00= | 1.12= |
| FindKnownFirstTrue i32 | 1.02= | 1.00= | 1.29W |
| FindKnownLastTrue i32 | 1.08W | 1.00= | 1.23W |
| FindLastTrue i32 | 1.01= | 0.94? | 1.23? |
| Floor f32 | 1.00= | 0.98= | 1.02= |
| FloorInt f32->i32 | 1.50W | 1.58W | 0.98= |
| GatherIndex i32 (4096) | 1.00= | 1.00= | · |
| GatherIndexN i32 (half n) | 1.00= | 0.99= | · |
| GatherIndexNOr i32 (half n) | 1.00= | 1.00= | · |
| GatherOffset i32 (4096) | 1.00= | 0.99= | · |
| Ge i32 | 0.99= | 1.00= | 1.01= |
| GetBiasedExponent f32 | 1.00= | 1.01= | 1.00= |
| GetExponent f32 | 1.00= | 1.02? | · |
| Gt i32 | 0.99= | 0.99= | 0.99? |
| HighestSetBitIndex u32 | 0.84L | 0.87L | 1.00= |
| IfNegativeThenElse f32 | 1.04W | 1.09W | 0.99= |
| IfNegativeThenElseZero f32 | 1.37W | 1.08W | 0.99= |
| IfNegativeThenZeroElse f32 | 1.37W | 1.08W | 0.97= |
| IfThenElse i32 | 1.00= | 0.99= | 1.00= |
| IfThenElseZero i32 | 0.97? | 1.00= | 0.98= |
| IfThenZeroElse i32 | 0.97? | 0.99= | 1.01= |
| InterleaveEven i32 | 1.01= | 1.01= | 0.99= |
| InterleaveLower i32 (4096) | 1.00= | 1.00= | · |
| InterleaveLowerBlocks i32 | 0.96= | 0.97? | · |
| InterleaveOdd i32 | 1.01= | 0.99= | 1.01= |
| InterleaveUpper i32 | 0.99= | 0.98= | 0.99= |
| InterleaveWholeLower i32 | 1.00= | 1.01= | 1.00? |
| InterleaveWholeUpper i32 | 1.00= | 1.00= | 1.01= |
| IsEitherNaN f32 | 1.00= | 1.00= | 1.00= |
| IsFinite f32 | 0.99= | 1.32W | 1.00= |
| IsInf f32 | 0.93L | 1.14W | 1.00= |
| IsNaN f32 | 1.01= | 0.99= | 1.00= |
| IsNegative i32 | 1.00= | 0.83L | 1.00= |
| Le i32 | 0.99= | 0.99= | 1.01= |
| LeadingZeroCount u32 (4096) | 1.05? | 1.20W | · |
| Load/Store i32 aligned | 1.00= | 1.00= | 1.00= |
| LoadExpand i32 (4096, 50%) | 1.77W | 2.11W | · |
| LoadInterleaved2 f32 (4096) | 1.00= | 1.00= | · |
| LoadInterleaved3 f32 (4096) | 1.00= | 1.01= | · |
| LoadInterleaved4 u8 (4096, RGBA) | 0.99= | 1.01= | · |
| LoadN/StoreN i32 (half lanes) | 1.10W | 0.65L | · |
| LoadNOr i32 (half n) | 0.95? | 0.54L | · |
| LoadU/StoreU i32 | 0.99= | 1.02= | 1.00= |
| Lt i32 | 1.00= | 1.01= | 0.96= |
| Lt128 u64 | 1.00= | 1.00= | 1.00= |
| Lt128Upper u64 | 1.00= | 1.00= | 0.99= |
| Mask And i32 | 1.00= | 1.01= | 1.00= |
| Mask AndNot i32 | 1.00= | 0.99= | 1.00= |
| Mask Not i32 | 1.00= | 1.00= | 1.00= |
| Mask Or i32 | 1.00= | 1.01= | 1.00= |
| Mask Xor i32 | 1.00= | 1.00= | 1.00= |
| MaskedApproxReciprocal f32 | 1.00= | 1.00= | 1.00= |
| MaskedApproxReciprocalSqrt f32 | 1.00= | 1.00= | 1.00= |
| MaskedConvertTo f32-i32 | 2.07W | 1.49W | 0.99= |
| MaskedLeadingZeroCount i32 | 1.13W | 1.11W | 1.01= |
| MaskedLoad i32 (4096, 50%) | 0.99= | 1.01= | · |
| MaskedLoadOr i32 (4096) | 1.00= | 1.04= | · |
| MaskedModOr i32 invariant | 0.96= | 0.99= | 1.06W |
| MaskedMulComplex f32 | 1.21W | 1.09W | 1.00= |
| MaskedMulComplexAdd f32 | 1.14W | 1.01= | 1.00= |
| MaskedMulComplexConj f32 | 1.21W | 1.11W | 1.00= |
| MaskedMulComplexConjAdd f32 | 1.13W | 1.01= | 1.00= |
| MaskedMulComplexOr f32 | 1.18W | 1.05W | 1.00= |
| MaskedMulFixedPoint15 i16 | 1.00= | 1.00= | 1.00= |
| MaskedReduceMax i32 | 1.00= | 1.00= | 1.01= |
| MaskedReduceMin i32 | 1.00= | 1.00= | 1.00= |
| MaskedReduceSum i32 | 1.01= | 0.99= | 1.00= |
| MaskedSatAddOr i16 | 1.00= | 1.00= | 1.00= |
| MaskedSatSubOr i16 | 1.00= | 1.01= | 1.00= |
| MaskedShiftLeft<3> i32 | 1.00= | 1.00= | 1.04= |
| MaskedShiftLeftOr<3> i32 | 1.00= | 1.00= | 1.00= |
| MaskedShiftRight<3> i32 | 1.00= | 1.00= | 1.00= |
| MaskedShiftRightOr<3> i32 | 0.99= | 0.99= | 1.00= |
| MaskedShl i32 | 1.01= | 0.99= | 0.99= |
| MaskedShlOr i32 | 1.00= | 1.00= | 1.00= |
| MaskedShr i32 | 1.01= | 1.00= | 0.99= |
| MaskedShrOr i32 | 1.00= | 1.00= | 1.00= |
| MaskedWidenMulPairwiseAdd i16-i32 | 1.00= | 1.01= | 1.00= |
| Max f32 | 1.00= | 1.00= | 1.03= |
| Max i32 | 1.00= | 1.00= | 1.02= |
| Max128 u64 | 1.00= | 0.99= | 1.00= |
| MaxMagnitude f32 | 0.99= | 1.00= | 1.00= |
| MaxNumber f32 | 0.99= | 1.00= | 1.04= |
| MaxOfLanes i32 | 0.86L | 1.17W | 1.01= |
| Min f32 | 1.00= | 1.01= | 1.01= |
| Min i32 (4096) | 1.00= | 1.00= | · |
| Min128 u64 | 1.00= | 0.99= | 1.00= |
| MinMagnitude f32 | 1.01= | 1.01= | 1.00= |
| MinNumber f32 | 1.00= | 1.01= | 1.03= |
| MinOfLanes i32 | 0.86L | 1.18W | 1.02= |
| Mod i32 invariant | 0.96? | 0.98= | 1.07W |
| Mul f32 | 1.01= | 1.00= | 1.00= |
| Mul i32 | 1.00= | 1.01= | 1.03= |
| MulAdd f32 (4096) | 1.00= | 1.00= | · |
| MulAddSub f32 | 1.00= | 1.01= | 0.89? |
| MulByFloorPow2 f32 | 1.07W | 0.98= | 1.00= |
| MulByPow2 f32 | 1.00= | 1.01= | 1.00= |
| MulComplex f32 | 1.42W | 1.20W | 1.00= |
| MulComplexAdd f32 | 1.04= | 1.03? | 1.00= |
| MulComplexConj f32 | 1.43W | 1.21W | 1.00= |
| MulComplexConjAdd f32 | 1.01= | 1.05W | 1.00= |
| MulEven i32-i64 | 1.00= | 0.99= | 1.00= |
| MulFixedPoint15 i16 | 1.00= | 0.99= | · |
| MulHigh i16 (4096) | 1.00= | 1.01= | · |
| MulHigh u16 (4096) | 1.00= | 1.00= | · |
| MulHigh32 u32 (4096) | 1.01= | 1.01= | · |
| MulOdd i32-i64 | 0.99= | 1.00? | 1.00= |
| MulRound f32 | 1.00= | 1.00= | 1.02= |
| MulSub f32 | 1.00= | 1.00= | 1.01= |
| MulSubAdd f32 | 1.00= | 1.01= | 0.90L |
| MultiRotateRight u64 (4096) | 1.00= | 0.99= | · |
| Ne i32 | 0.99= | 1.00= | 1.00= |
| Ne128 u64 | 1.00= | 1.00= | 1.01= |
| NearestInt f32->i32 | 1.40W | 1.07W | 1.01= |
| Neg i32 | 1.00= | 1.00= | 1.01= |
| NegMulAdd f32 | 1.00= | 1.00= | 1.00= |
| NegMulSub f32 | 1.00= | 0.99= | 0.98= |
| Not i32 | 0.96L | 1.00= | 1.03? |
| OddEven i32 | 1.00= | 0.99= | 1.02= |
| OddEvenBlocks i32 | 0.99= | 1.01= | · |
| Or i32 | 0.99= | 1.00= | 0.99= |
| Or3 i32 | 1.00= | 1.00= | 0.98= |
| OrAnd i32 | 1.00= | 1.00= | 0.99= |
| OrderedDemote2MasksTo i32->i16 (4096) | 1.02? | 1.03= | · |
| OrderedDemote2To i32->i16 | 1.00= | 1.00= | · |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.08W | 1.18W | · |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 1.00= | 0.99= | · |
| PairwiseAdd i32 | 1.09W | 1.01? | 1.00= |
| PairwiseAdd128 i32 | 1.00= | 1.01= | 1.00= |
| PairwiseSub i32 | 1.09W | 1.00= | 1.00= |
| PairwiseSub128 i32 | 1.08W | 1.03? | 0.99= |
| Per4LaneBlockShuffle 3210 i32 | 0.89L | 1.00= | 1.00= |
| PopulationCount u32 (4096) | 1.82W | 1.51W | · |
| PromoteEvenTo i8->i16 | 0.98= | 0.97= | · |
| PromoteLowerTo i16->i32 (4096) | 1.01= | 1.00= | · |
| PromoteMaskTo i16->i32 (4096) | 0.99= | 1.00= | · |
| PromoteOddTo i8->i16 | 0.99= | 1.00= | · |
| PromoteTo i16->i32 1-arg (4096) | 0.99= | 0.97= | · |
| PromoteUpperTo i16->i32 | 1.00= | 1.00= | · |
| RebindMask i32->f32 (4096) | 0.95L | 1.01= | · |
| ReduceMax i32 | 1.17W | 1.03= | 1.00= |
| ReduceMin i32 | 1.19W | 1.02= | 1.00= |
| ReduceSum f32 (4096) | 0.99= | 1.00= | · |
| ReorderDemote2To i32->i16 | 1.00= | 0.99= | · |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.05= | 1.18W | · |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 1.00= | 0.98? | · |
| ReorderWidenMulAccumulate i16->i32 | 1.00= | 1.00= | · |
| Reverse i32 (4096) | 1.00= | 1.00= | · |
| Reverse2 i32 | 1.00= | 0.99= | 1.02= |
| Reverse4 i32 | 1.00= | 1.00= | 0.99= |
| Reverse8 i32 (4096) | 1.00= | 0.99= | · |
| ReverseBits i32 | 2.09W | 1.97W | 0.99? |
| ReverseBlocks i32 | 1.00= | 1.02= | · |
| ReverseLaneBytes u32 (4096) | 0.05L | 1.00= | · |
| RotateLeft<3> i32 | 0.99= | 1.02? | 1.01= |
| RotateLeftSame u32 bits=5 | 0.99= | 1.01= | 1.01= |
| RotateRight<3> i32 | 0.98= | 1.02= | 1.02= |
| RotateRightSame u32 bits=5 | 1.00= | 1.00= | 1.01= |
| Round f32 | 1.00= | 1.00= | 0.98= |
| RoundToInt f32->i32 | 1.39W | 1.08W | 1.00= |
| RoundingShiftRight<2> i32 | 1.08W | 0.98= | 1.00= |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.93L | 1.12W | · |
| RoundingShiftRightSame i32 bits=5 | 1.09W | 1.01= | 0.82L |
| RoundingShr i32 | 1.06W | 0.95? | 0.98= |
| SatWidenMulAccumFixedPoint i16->i32 | 1.92W | 1.77W | · |
| SatWidenMulPairwiseAccumulate i16->i32 | 0.93L | 0.98= | · |
| SatWidenMulPairwiseAdd u8*i8->i16 | 1.00= | 1.02= | · |
| SaturatedAbs i32 | 1.00= | 1.00= | 0.97= |
| SaturatedAdd i16 (4096) | 0.99= | 0.99= | · |
| SaturatedNeg i32 | 1.00= | 1.01= | 1.03= |
| SaturatedSub i16 | 1.00= | 0.99= | 1.00= |
| ScatterIndex i32 (4096) | 1.00= | 1.01? | · |
| ScatterIndexN i32 (half n) | 1.22W | 1.02? | · |
| ScatterOffset i32 (4096) | 0.99= | 1.03= | · |
| Set i32 | 1.00= | 1.00= | 1.00= |
| SetAtOrAfterFirst i32 | 1.00= | 1.00= | 1.00= |
| SetAtOrBeforeFirst i32 | 0.98? | 1.18W | 1.00= |
| SetBeforeFirst i32 | 0.99= | 1.31W | 1.00= |
| SetMask i32 (4096) | 1.02= | 1.00= | · |
| SetOnlyFirst i32 | 0.98= | 1.69W | 1.00= |
| ShiftLeft<3> i32 | 1.00= | 1.00= | 0.98= |
| ShiftLeftBytes<4> i32 | 1.00= | 1.00= | 1.00= |
| ShiftLeftLanes<1> i32 | 1.00= | 1.00= | 1.01= |
| ShiftLeftSame i32 bits=3 | 1.00= | 1.00= | 1.96W |
| ShiftRight<3> i32 | 1.01= | 1.00= | 1.00= |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 1.00= | 1.00= | · |
| ShiftRightBytes<4> i32 | 1.00= | 0.98= | 1.00= |
| ShiftRightLanes<1> i32 | 1.00= | 1.00= | 1.00= |
| ShiftRightSame i32 bits=3 | 1.00= | 1.00= | 1.22W |
| Shl i32 | 0.99= | 1.01= | 1.01= |
| Shr u32 (variable) | 1.00= | 0.99= | 1.00= |
| Shuffle0123 i32 | 1.00= | 1.00= | 1.01? |
| Shuffle0321 i32 | 1.00= | 1.00= | 0.97= |
| Shuffle1032 i32 | 1.00= | 1.00= | 1.01? |
| Shuffle2103 i32 | 1.00= | 1.01= | 1.02= |
| Shuffle2301 i32 | 1.00= | 0.99= | 1.00= |
| Slide1Down i32 | 1.00= | 0.99= | 1.03? |
| Slide1Up i32 | 1.00= | 0.99= | 1.01? |
| SlideDownLanes amt=2 i32 | 0.98= | 0.99= | 1.00= |
| SlideMask1Down i32 (4096) | 0.96? | 1.00= | · |
| SlideMask1Up i32 (4096) | 0.95? | 0.99= | · |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.09W | 1.00= | · |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.07? | 1.01= | · |
| SlideUpLanes amt=2 i32 | 0.96? | 1.00= | 0.98= |
| Sqrt f32 | 1.00= | 1.00= | 1.00= |
| StoreInterleaved2 f32 (4096) | 1.00= | 0.99= | · |
| StoreInterleaved3 f32 (4096) | 1.02= | 0.90L | · |
| StoreInterleaved4 f32 (4096) | 1.48W | 1.45W | · |
| Sub f32 | 1.00= | 0.99= | 0.99= |
| Sub i32 | 1.00= | 0.99= | 0.99= |
| SumOfLanes i32 | 0.78L | 1.20W | 0.98= |
| SumOfMulQuadAccumulate i8->i32 | 1.00= | 1.02= | · |
| SumsOf2 u8->u16 | 1.00= | 0.98= | · |
| SumsOf4 u8->u32 | 0.99= | 0.99= | · |
| SumsOf8 u8->u64 | 1.00= | 1.02= | · |
| SumsOf8AbsDiff u8->u64 | 1.01= | 1.00= | · |
| SumsOfAdjQuadAbsDiff u8->u16 | 0.99= | 0.98= | · |
| SwapAdjacentBlocks i32 | 1.00= | 0.99= | · |
| TableLookupBytes u8 (4096) | 0.90? | 0.85L | · |
| TableLookupLanes i32 (4096) | 1.00= | 1.00= | · |
| TestBit i32 | 0.92L | 1.01= | 1.00= |
| TrailingZeroCount i32 | 0.87L | 0.87L | 0.68L |
| Trunc f32 | 1.00= | 1.01= | 0.99= |
| TruncateTo u32->u16 | 0.85L | 0.83L | 1.00= |
| TwoTablesLookupLanes i32 (4096) | 1.01= | 1.00= | · |
| WidenMulAccumulate i16->i32 | 1.00= | 1.01= | · |
| WidenMulPairwiseAdd i16->i32 | 0.99= | 1.01= | · |
| Xor i32 | 1.00= | 0.98= | 0.97= |
| Xor3 i32 | 1.63W | 1.00= | 1.03= |
| Zero i32 | 1.00= | 1.00= | 1.02= |
| ZeroExtendVector i16 half->full | 1.00= | 0.99= | · |
| ZeroIfNegative f32 | 1.02= | 1.10W | 1.01= |

## 复现
```
# 每平台:构建 hyle_op_parity{,_full,_widen}_bench_<isa>,双序各跑一遍(Mac 建议 4 遍),然后
python scripts/op-parity-report.py out_hf.txt out_hwf.txt          # x86(闸门 ±4%)
python scripts/op-parity-report.py --gate 0.06 mac_*.txt           # Mac(无亲和性,±6%)
# 报告首段为锚点校准;CALIBRATION PASS 之后的判决才可引用。
```
