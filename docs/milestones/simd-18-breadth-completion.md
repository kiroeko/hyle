# simd-18 广度顶格补全 + 性能盲区

| 项 | 内容 |
|---|---|
| 状态 | 🔴 施工中（opus 裁决 + 施工）|
| 触发 | 用户拍板:目标顶格——广度做全(不再"缓做/等消费者/偏门"豁免,只有**重新核实的真 ISA/工具链墙**才免),再补性能测试盲区 |
| 方法 | Highway HWY_API 全集 vs HYLE 机械差集(106 候选)→ 4 组 subagent 逐个对源码核实 → opus 裁决 |

## 0. 裁决总表(源码核实,非凭记忆)

差集 106 候选,裁决:
- **真公开面缺口(做): ~55 op**
- **命名差异/已有(不做,登记): 30 Masked* + 若干**
- **Highway 内部 detail::(不做,非公开面): 9**
- **真 ISA 墙(登记不做): 2**

### 已有,别做(命名相同,宏生成——agent 组1 核实)
27 个 `Masked{Add,Sub,Mul,Div,Max,Min,Eq,Ne,Lt,Le,Gt,Ge,Or,Xor,Abs,Sqrt,SaturatedAdd/Sub}` + `Or` 形:vec.h:392-430 三个宏 `HYLE_SIMD_MASKED_{BINARY,CMP,UNARY}` 已全生成,七后端覆盖。`WidenMulEven`=HYLE `MulEven`(vec.h:843)。

### Highway 内部 detail::(非公开 op,不做)
- `ShuffleTwo1230/2301/3012`:全在 `namespace detail`(generic_ops:1604/x86_256:5172/...),是 LoadInterleaved3 内部件。HYLE 已有 LoadInterleaved3 消费者。
- `GaloisAffine`:`detail::` GFNI 原语,服务 ReverseBits/RotateRight(HYLE 已实现)。
- `NativeGather256`/`NativeMaskedGatherOr256`/`LoadDupFull128`:内部 impl 选择器,公开 GatherIndex/LoadDup128 已覆盖。
- `*TransposedBlocks*`×5(LoadTransposedBlocks3/4/StoreTransposedBlocks2/3/4):Load/StoreInterleaved{2,3,4} 的内部 plumbing,HYLE 已有那些公开 op。
- `PFalse`:SVE detail,== 公开 `MaskFalse`。

### 真 ISA 墙(登记不做,坐实)
- `PerBlock2x2MatMul`:Highway 仅 NEON/SVE2 经 `svmmla`(需 `__ARM_FEATURE_SVE_MATMUL_INT8`/i8mm),x86 无对应指令。ARM-matmul 专属,非通用面。
- `CompressBlocksNot`:AVX3-only + uint64 block 粒度 + 需 `CompressNot`;窄。

## 1. 施工批次(TIER A 真缺口,~55 op)

### 批A:泛型 mask 族(14 op,全 vec.h 一处定义,七后端自动覆盖)
`RebindMask`(前置,跨同宽 lane 重解释 mask)、`CombineMasks`、`LowerHalfOfMask`、`UpperHalfOfMask`、`SlideMask1Up`、`SlideMask1Down`、`SlideMaskUpLanes`、`SlideMaskDownLanes`、`MaskFalse`、`SetMask`、`Dup128MaskFromMaskBits`。
手法:同宽 mask 族全是 `MaskFromVec(<vec-op>(VecFromMask(m)))` 薄封装(Highway generic_ops:242-469 同款),HYLE 已有全部底层 vec-op。
**宽度转换 mask 3 op**(`PromoteMaskTo`/`DemoteMaskTo`/`OrderedDemote2MasksTo`)不走向量域(HYLE 无单向量 PromoteTo + AVX-512 mask=k-reg 桥接昂贵),改 **bitfield round-trip**(`BitsFromMask`→`LoadMaskBits`):promote/demote 保持 lane 数,五后端 1-bit/lane 位域与 lane 宽无关,重打标签即得(c45c6af,@opus 裁决)。

### 批B:泛型 vector 杂项 + gather/scatter-N(10 op)
`IfVecThenElse`(vec 选择子,非 mask)、`IfNegativeThenNegOrUndefIfZero`(float=CopySign(v,Xor(m,v));int=IfNeg(m,Neg,v))、`HighestSetBitIndex`((bits-1)-LZCnt)、`Inf`/`NaN`(常量生成器)、`Dup128VecFromValues`(立即数建 128-block 广播,需 native)、`TwoTablesLookupLanes`(双源 permute,native fast-path)、`GatherIndexN`/`GatherIndexNOr`/`ScatterIndexN`(runtime-N,= Masked{Gather,Scatter}(FirstN))。

### 批C:demote/truncate/shift-demote/InRange(17 op)
- 平凡组合:`Blocks`(constexpr)、`TruncateStore`、`OrderedTruncate2To`(ConcatEven)、`OrderedDemote2To`(avx2/512 需 permute 修序,sse4/neon/scalar=Reorder 别名)、`Reorder{,Rounding}ShiftRightAndDemote2To`、`Ordered{,Rounding}ShiftRightAndDemote2To`(均 2-arg demote2 形;单向量 `ShiftRightAndDemoteTo` 不做——HYLE 无单向量 DemoteTo 实体)。
- `DemoteToNearestInt`(f64→i32 round-nearest-even + 溢出/NaN 按符号饱和,c45c6af @opus):**非** DemoteTo∘Round 平凡组合——全 f64 域决策(两 INT32 端点精确 double),`in_range?Round(v):sign_sat` 后单次 `DemoteInRangeTo` 避 cvttpd 越界 UB;NaN 用 `MaskFromVec` 符号位(=emu128 ScalarSignBit)。objdump avx2=vroundpd+vcmp+2×vblendv+vcvttpd2dq 全原生。
- InRange 快路(caller 保证域内,跳饱和 fixup,x86 raw cvt 真更快=性能目标):`ConvertInRangeTo`、`DemoteInRangeTo`、`PromoteInRangeTo` + 派生 `PromoteInRange{Lower,Upper,Even,Odd}To`。

### 批D:WidenMul accumulate(3 op,泛型组合)
`WidenMulAccumulate`(双宽累加器 FMA)、`SatWidenMulPairwiseAccumulate`(i16→i32 饱和累加)、`SatWidenMulAccumFixedPoint`(Q15 饱和 MAC,需 InterleaveWholeLower)。

### 批E:128-bit block 操作(12 op,avx2=2/avx512=4 block,sse4/scalar=identity)
`InsertBlock`(ExtractBlock 逆)、`ReverseBlocks`、`SwapAdjacentBlocks`、`OddEvenBlocks`、`SlideUpBlocks`、`SlideDownBlocks`、`Interleave{Even,Odd,Lower,Upper}Blocks`、`InterleaveWhole{Lower,Upper}`(≠HYLE 现有 per-block InterleaveLower/Upper,是整向量连续)、`Per4LaneBlockShuffle`(模板 4-idx block 内 permute)。

### 批F:SAD 视频(2 op)
`SumsOfAdjQuadAbsDiff`(AVX2 `mpsadbw` 基线,非门控;portable 用 InterleaveLower+CombineShiftRight+SumsOf2)、`SumsOfShuffledQuadAbsDiff`(AVX512BW `dbsad`,SDE 可验非墙;portable 用 Per4LaneBlockShuffle)。

### 批G:数学面(dispatch 布线,借 Highway/xsimd best-of)
- 两库都缺,HYLE 也缺(自造/借 xsimd):`erfc`、`exp10`、`fmod`、`remainder`、`nextafter`、`fdim`。
- Highway 有 HYLE 缺(借 Highway):`acos`、`asin`、`atan2`、`sinh`、`cosh`、`tan`、`atanh`、`log10`、`pow`、`sincos`。

### 批H:性能盲区补全(第二目标)
新增 op 进 `op_parity_bench`;审已上架但无 bench 背书的 op;best-of 逐精度/逐 ISA(bench-stat.py -n15,同 TU 同等内联)。

### 别名(登记,薄封装)
`ZeroExtendResizeBitCast`(=HYLE ResizeBitCast,已零高位)、`SetTableIndices`(=IndicesFromVec∘LoadU)。

## 2. 施工纪律(逐 op)
按 gap-inventory §0.5 决策树:两库都有→bench 取优;正确性闸(scalar oracle 对拍);五平台真跑(WSL scalar/sse4/avx2 + AVX512 SDE + AVX512_DL SDE where native + mac NEON + scalable qemu);ASan 查 store+reload scratch 栈宽。
