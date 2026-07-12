# SIMD test/benchmark 覆盖缺口清单

> **状态（2026-07-12）**：施工完成——**禁止抽样，直通单指令也全补**。原 ~65 行 → `op_parity_full` 单文件 **139 行** + crypto 8 + parity_bench 扩 + widen 16 + dtype f16→fp8。CMake 已接 crypto/widen。WSL GCC14 AVX2 真编真跑全绿。
>
> **方法纪律（必守）**：① 每 bench 行必须公平对拍（full baseline `-maes -mpclmul` 等，两边 header-inline）；② 统计法判定（`bench-stat.py -n15`）；③ **每个公开 op 都要有 bench 行，不许"抽样代表"**；④ Tier3 HYLE-only 只 vs scalar/裸估计。

## 已落地覆盖

| 文件 | 行/域 | 状态 |
|---|---|---|
| `crypto_bench.cc` | AESRound/Last/Inv/LastInv/InvMix/KeyGenAssist + CLMulLower/Upper | ✅ 新建；AES/CLMul **全平 Highway** |
| `op_parity_full_bench.cc` | 原 Masked* + 裸 complex 5 + ReduceMin/Max + ReduceMul/And/Or(vs scalar) + 128-cmp 6 + Tier2(Saturated*/Reverse2/4/Slide1/Rotate/Broadcast/AddSub/Clamp/RoundingShr/AverageRound/MulByPow2/Floor/GetBiasedExp) + Newton 2 | ✅ |
| `op_parity_bench.cc` | 原 20 + TableLookupLanes/TwoTables + Scatter/CompressStore/MaskedLoad(Or)/BlendedStore/LoadN·StoreN + MulHigh i16/u16/32 + Reverse8 + BitShuffle/MultiRotateRight + Interleave2/Store2/3 | ✅ |
| `op_parity_widen_bench.cc` | 原 7 + ReorderWidenMulAccum + SatWidenMulPairwiseAdd + SumOfMulQuad + WidenMulAccum + SatWidenMulPairwiseAccum + SatWidenMulAccumFixedPoint + SumsOfAdjQuadAbsDiff + OrderedDemote2To + ZeroExtendVector | ✅ CMake 接线 |

## 冒烟揪出的输家 + opus 校对/修复处置（2026-07-12）

opus 逐项 objdump 校对（区分"真输"vs"bench 伪影"）并修复。下表为处置后状态：

| op | 单遍 gap | 校对结论 | 处置 → 现状 |
|---|---|---|---|
| **mask 全族**（SetMask/Dup128/CombineMasks/PromoteMaskTo/OrderedDemote2Masks…） | 3–22x | **主因是 bench 伪影**：`detail::mask_bits` 对 i32/i64 用 `movemask_epi8`+20 条标量压位，Highway 用一条 `movemask_ps/pd`。sink 拖慢所有 mask 测量，非各 op 本身 | ✅ **已修** mask_bits 用宽度匹配 movemask（i32→ps/i64→pd）→ SetMask/Dup128/CombineMasks **平**；命中 CountTrue/FindFirstTrue/Compress 全域 |
| **MulByPow2 / MulByFloorPow2 f32** | 8–19x | **真输**：HYLE store+`std::ldexp`+reload | ✅ **已修** 抄 Highway 3 段指数域分解（五后端 avx2/sse4/neon/rvv 向量化，avx512 MulByFloorPow2 改原生 vscalef）→ **平**；五平台真机/模拟器验含 overflow/denormal/±inf |
| **Slide1Up / Slide1Down i32** | 6x | **真输**：store+reload | ✅ **已修** 抄 CombineShiftRightBytes/vext（avx2/sse4/neon）→ **平**；连带 SlideMask1Up/Down 12x→**平** |
| **SumOfMulQuadAccumulate i8** | 13x | **真输**（-n15 坐实 gap 1193%»spread，非伪影）。第一次改（PromoteEven/Odd+2×vpmaddwd）只消了标量乘，**真根因是 PromoteEvenTo/OddTo 本身 store+reload**——objdump 见 vpmaddwd 但 promote 仍标量 gather | ✅ **已修（根因）** 见下条 PromoteEvenTo；SumOfMulQuad 随之 0.22→0.02ns **追平**；sse4/neon 同步 |
| **PromoteEvenTo / PromoteOddTo（承重根因）** | — | store+reload 标量 gather，拖累所有调用者（SumOfMulQuad/WidenMul 族） | ✅ **已修** 三后端寄存器内移位：avx2/sse4 `slli+srai`(signed)/`srli+mask`(unsigned) 6 组合(i8→i16/u8→u16/i16→i32/u16→u32/i32→i64/u32→u64)，neon `vuzp1/2q+vmovl`；i32→i64 signed 无 srai_epi64 用 srai32 sign-fill；全组合 probe 0-mismatch + x86 全 test 过 |
| **TwoTablesLookupLanes i32** | 4.6x | **真输**：store+reload | ✅ **已修** avx2 抄 Highway 回退（2×TableLookupLanes+idx&8 选）→ 近平 1.18x |
| **BlendedStore i32** | 4.1x | **真输**：store+reload；Highway 用原生 maskstore | ✅ **已修** 抄 `_mm256_maskstore_epi32/64/ps/pd`（8/16-bit 无 maskstore 保留 scalar，同 Highway）→ **赢 1.12x** |
| **PairwiseAdd/Sub（whole）** | 5x | **真输**：store+reload。但 HYLE 契约（分段 [a-sums \| b-sums]）≠ Highway（交织）——**不能直接抄 Highway，用正确 building block 达最优复杂度** | ✅ **已修** `Add(ConcatEven(b,a), ConcatOdd(b,a))`（ConcatEven/Odd 已 vpermd 向量化，保持 HYLE 分段契约）→ **平** |
| **PairwiseAdd128/Sub128** | 5–7x | 真输：store+reload；Highway 用 hadd | ✅ **已修** 抄 `_mm256_hadd/hsub_{ps,pd,epi32,epi16}`（i16/i32/f32/f64）→ **平/赢** |
| **SlideUpLanes/DownLanes（runtime amt）** | 8x | **真输**（不接受"runtime 无向量路径"借口）：32-bit 可用 vpermd+range-mask 做 runtime-amt 向量化 | ✅ **已修** avx2 32-bit 走 `vpermd(iota±amt)+cmp-mask`（runtime amt 一条 permute，比 Highway 只 constant-amt 向量化更通用）→ **平**；连带 SlideMaskUp/DownLanes 15x→**平**；exhaustive amt 0..8 f32+i32 验 |
| **SumOfMulQuadAccumulate i8**（sse4/neon） | 10x | 真输：store+reload | ✅ **已修** sse4 挪定义到 PromoteEven/Odd 后 + 同 avx2 手法；neon 用 `vmull_s8+vpaddlq_s16+vpaddq_s32`（qemu 200k fuzz 验）→ 四后端全向量 |
| **SaturatedNeg/Abs i32**（avx2/sse4） | 1.14–1.5x | store+reload（i32/i64 无 native satsub） | ✅ **已修** 纯向量 `IfThenElse(a==MIN, MAX, Neg(a))`；NEON 本就原生 `vqnegq/vqabsq`（残留 gap 是亚纳秒噪声，统计平） |
| **MultiRotateRight u64** | 3x | **真硬件墙**（agent objdump 坐实）：Highway generic 回退核心是 `_mm256_srlv_epi16`（u16 变量右移=AVX512BW+VL），纯 AVX2 无；照搬会因内部 `Shr<u16>` 本身 store+reload 而回归 | ⚠️ **保留 store+reload**（avx512 走原生 multishift）；缺的指令明确、非借口 |
| **CompressStore i32** | 3.4x | **契约差异非缺陷**：HYLE 精确写 p[count..) 不触碰，比 Highway 写满向量的弱契约强。AVX2 无 native compress-store，精确写只能 scalar/maskstore；实测 scalar 最快（Compress+maskstore(FirstN) 更慢，FirstN 自身 store+reload） | ⚠️ **保留 scalar 精确写**（256 种 mask 模式 + tail-untouched 验）；想要 Highway 速度用 `StoreU(Compress(v,m))` |
| **SatWidenMulAccumFixedPoint** | harness 栈拷贝失真，已改半宽 ZeroExtend | — | ✅ 复测平 |

## 多数打平 / 赢（冒烟）

- **Crypto 全 8 行**：AES* / CLMul **统计平**（ratio ~0.94–1.03）。
- **裸 complex 5 / 128-cmp 6 / ReduceMin·Max / MulHigh / TableLookupLanes / Scatter / MaskedLoad* / Interleave2·3 / 多数 widen MAC**：平。
- **ReduceAnd/Or vs scalar fold**：HYLE ~2.2x 赢标量。
- **ReciprocalNewton**：比裸 ApproxRecip 慢 ~1.3x（付 1 步 Newton，预期），比 `Div(1,x)` 快 ~1.1x；Sqrt 变体比 `Div(1,Sqrt)` **~2.1x**——SOTA 声称有吞吐背书。

## 低优先已补（2026-07-12 续）

| 域 | 文件 | 行 |
|---|---|---|
| mask 全族 | `op_parity_bench` §25 | PromoteMaskTo / OrderedDemote2MasksTo / CombineMasks / SlideMask1Up·Down / SlideMaskUpLanes / RebindMask / SetMask / Dup128MaskFromMaskBits |
| 单向量 Promote/Demote | `op_parity_bench` §26 | PromoteTo 1-arg / DemoteTo 1-arg / ShiftRightAndDemoteTo / RoundingShiftRightAndDemoteTo |
| Shuffle / Pairwise / SlideLanes | `op_parity_full` | Shuffle0123/1032/2301/0321/2103 / Per4LaneBlockShuffle / PairwiseAdd·Sub·Add128·Sub128 / SlideUp·DownLanes / MulRound / CombineShiftRightBytes·Lanes / TruncateTo |
| f16→fp8 | `dtype_bench` | f16_to_f8e4m3 / f16_to_f8e5m2 vs via-f32 两跳 |

**低优先冒烟新输家**（单遍，待 `-n15`）：
- **PairwiseAdd/Sub/128 族 ~6–7×**（HYLE 注释即 store+reload）
- **SlideUp/DownLanes ~8–9×**（同 Slide1 族）
- **mask 族多数 4–20×**（bitfield 往返；文档已承认冷路径；RebindMask 平）
- **TruncateTo ~2.5×**（中等）
- Shuffle* / MulRound / CombineShift* / 单向量 Promote·Demote **平**

**直通单指令已全补（禁止抽样）**：And/Or/Xor/AndNot/Not、Min/Max f32、Div、CopySign、Floor/Ceil/Round/Trunc、Ne/Le/Ge、IfThenElse*、ShiftLeftSame/RightSame、RoundToInt、ConvertTo 双向、BitCast 双向、Set/Zero、LoadU/StoreU、Load/Store(对齐缓冲)、Mask And/Or/Xor/AndNot/Not、Add/Sub/Mul f32。冒烟：**向量直通全平**；Mask 位运算 HYLE 慢 ~3×（BitsFromMask 路径，冷）；RoundToInt/ConvertTo f32→i32 微慢 ~1.2×（待 -n15）。

## 正确性测试

vec_test 含 crypto/complex/128-cmp/Newton/widen-MAC/permute/interleave 专门 TEST。`SetTableIndices` 在 `HighwayParityWrappers`。**无高置信度正确性缺口**。
