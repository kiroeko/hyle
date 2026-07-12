# SIMD 缺口清单

> **这份文档回答:HYLE simd 到底还差什么。** 2026-07-10 用机械差集 + 逐源验证建立，并随施工保持当前状态。覆盖四个信源:①消费者驱动延后项 ②标量/非标量 + 定长/变宽对称性漏项 ③xsimd/Highway 有而 HYLE 无的非偏门 op ④文档明确登记的缺口。
>
> **方法纪律(必读)**:原始 `grep` op 名差集**有噪音**——各后端宏风格不同(scalar 用 `HYLE_SIMD_SCALAR_BINOP(Add,...)`、avx2 用 `HYLE_SIMD_AVX2_ROUND(...)`,裸 grep 会漏报"缺 Add/Round")。**本清单每一项都经后端源码核实**,分类为:🔴 REAL-GAP(施工目标)/ 🟰 NAMING-DIFF(已有,别做)/ 🔒 HW-GATED(硬件门控,登记不做)/ ⚪ OBSCURE(偏门,跳过)。下个 session 施工前,对将做的每一项**再核一次当前源码**(可能已被后续 commit 补上)。

## 0. 一句话结论

- **定宽四后端(sse4/avx2/avx512/neon)op 面严格一致**(各 ~223-228 distinct op 名,avx2/avx512 并列最高——P3/P4/simd-17/18 补 op 后从初版 ~166-168 增长)——**这块没问题**。
- **scalar 完整**(是 oracle;"缺 Add/Mul" 是宏定义的 grep 假阳)——**没问题**。
- **定稿时的真问题 = 变宽(SVE/RVV)落后定宽 ~72 op**；当前 P0-P5 已全闭环：scalable 量化/点积/结构 op 已对称，定宽 Masked* 与 Mul/Mod/Magnitude 已补齐，P5 平凡合成 + mask 前缀 + MulByPow2/Expand/TZCnt 已落地。
- vec.h 泛型 complex / 128-cmp / Newton 曾因 scalable TU 未实例化而掩盖底层原语缺口；P0 已用 SVE/RVV 直接实例化测试消除该潜在编译断裂。`Expand` 已在 P5 落地（SVE store+reload / RVV viota+vrgather）。
- **主施工闭集 P0-P5 软件面 100%**。2026-07-12 directive2 复核后,先前登记的"硬件/工具门控"项**全部翻案做掉**(见下条);连最后的 mask-bits VL>64 也用 byte-buffer API 解掉(dbf88c0)——**已无真墙**(fp8-HW 是有意的正确性取舍,非墙)。
- **公开 op 面对 Highway `HWY_API` 已完备**(2026-07-12 d6b1d13 收尾):机械差集审计发现的 4 个薄封装缺口(`SetTableIndices`/`CompressBlocksNot`/`ZeroExtendResizeBitCast`/`Dup128MaskFromMaskBits`)已补,五平台真跑对拍底层原语。
- **又翻案两处我自己下过的"有意排除"(2026-07-12,用户点探)**:①单向量 `PromoteTo`/`DemoteTo`/`ShiftRightAndDemoteTo`(Rebind-tag 形)—— 复核后是**过保守,已补**(d488813,现有原语薄封装,五平台+ASan);②**f32→fp8 硬件方向 —— 已上架 bit-identical**(3fe8407):双舍入用 round-to-odd 中间步骤消除(Boldo-Melquiond,同 Highway F64→F16 手法),clang-20+SDE 真跑 + 486.6M 穷举 0 mismatch。**剩余仅真有意排除**:`GaloisAffine*`(Highway `detail::` 内部 GFNI 原语,非公开面);单向量 `PromoteTo`/`DemoteTo` 的**全寄存器 half-at-a-time 主路径**仍是 HYLE 原生首选(单向量形是便利封装,返回 partial 向量,读回需 `DFromV`)。
- **[`simd-16`](./simd-16-hw-gated-reclaim.md) + directive2 已翻案 RVV Zvbb/Zvbc + SVE2 全套**：RVV `LeadingZeroCount`/`TrailingZeroCount`/`ReverseBits`/native `Rotate`/`CLMulLower`/`CLMulUpper` **+ `PopulationCount`(2026-07-12 b0ad404,inline-asm vcpop.v)**；**SVE2 `AES*`/`CLMul` 8 op 全套(2026-07-12 8c466f8,native svaese/svaesmc/svpmull)**——全 qemu 4 VLEN/VL + objdump 真指令。**先前"PopulationCount 工具链缺口 / SVE2 只能编译验"结论已推翻**(是当时测试 harness bug,非真墙;qemu 8.2.2 本就能跑)。

## 0.5 每项 op 的执行决策流程(施工规程,逐 op 机械跑)

清单里每个 🔴 REAL-GAP op,**照这棵决策树走**(固化 simd-08/09 沉淀 + backlog-map best-of 判据,不用每次重推):

```
对一个待补 op:
├─ 两个参考库(Highway/xsimd)都没有?
│    └─ 是 → 【自己实现】(泛型层组合已有 op,或后端原生 intrinsic)。scalar 是 oracle。
├─ 只有一个库有?
│    └─ 是 → 【默认它最优,借/抄它】(单一来源无从比,直接采用其手法/契约)。
└─ 两个库都有?
     └─ benchmark 取优(op_parity_bench,统计法 -n15,gap>两变体 spread 和才判赢),
        选出"最优的那个",再对它判断:
        ├─ 你觉得能超过它? → 【自己写 + benchmark 确认真超过】(超过才留,没超过就退回抄最优;
        │                      如 simd-09 rsqrt/rcp Newton 精修反超两库=自造超越)
        ├─ 感觉能自己写但不必超越? → 【抄】(抄最优的契约/指令序列,自己写进后端,如 simd-05~08
        │                            per-op intrinsic 序列;不 vendor)
        └─ 太复杂写不了(数千行/多项式逼近/整套算法)? → 【借】(vendor 薄封装:公开层吃裸指针
                                                        不 include 库、借核 TU 内部用库 D-tag,
                                                        如 vqsort/bit_pack/数学核 exp_f32_batch)
```

**三条铁前置(踩过坑,别省)**:
1. **契约内 correctness 是闸**:任何"更快"实现必先在其公开输入域内过 libm/scalar-oracle / round-trip 自洽对拍；若完整语义与显著加速冲突，提供显式 safe/fast 双入口而非静默换语义（`hypot_f32_batch` overflow-safe，`hypot_in_range_f32_batch` 约快2倍但要求 direct-formula 输入域）。精度不达标或契约内错误的实现没资格入选；Highway Pack16 / xsimd lgamma vendor bug说明**借不等于信，借核同样过对拍闸**。
2. **best-of 只在"有真实算法内容 + 三家有差异 + 不贴单一硬件指令"时才有意义**:纯直通指令(add/min/shuffle/位计数)三家编成同一条,不必 bench;数学函数(多项式逼近)是 best-of 主场;结构 op(gather/compress store+reload)库也无硬件指令,真加速是自造 native 非借库。
3. **winner 不可外推,必须逐精度/逐 ISA bench**:f64 winner 与 f32 大面积相反(exp/tanh/exp2 翻转);SDE 验正确性可信、验性能不可信;bench 两边必须同等内联能力(同 TU / 都 header-only,否则跨 .so 不可内联=假 gap)。

**每 op 收口**:四 ISA 真跑(SSE4/AVX2/AVX512-SDE/NEON-qemu-or-mac + scalable qemu 多 VL)对拍 scalar oracle;进 op_parity_bench 背书(自造件尤其,不是"抄了契约"就默认不亏);过 ASan(store+reload 的 scratch 栈宽是 TruncateTo 类单平台陷阱重灾区)。

## 1. 信源①②:变宽 vs 定宽对称性缺口(最大块)

本清单定稿时的精确差集(宏感知、去 crypto + Emulated)是 avx2 有而 **SVE 缺 72 / RVV 缺 74**。P0-P2 已从 scalable 差集中消除；P3 是已完成的定宽 Highway parity 批，剩余 scalable 软件项集中在 P5，硬件门控项单独管理。

### ✅ P0 — 解锁已发货的泛型层(正确性,非仅覆盖)
原先这些 vec.h 泛型 op 依赖 scalable 后端未定义的底层原语；现在 SVE/RVV 测试 TU 已直接实例化消费者，编译断裂已消除：

| 泛型 op(vec.h)| 已补的 SVE/RVV 原语 |
|---|---|
| `MulComplex/MulComplexAdd/MulComplexConj*` | `DupEven`, `DupOdd`（直接 FMA 友好公式） |
| `Lt128/Eq128/Ne128/Min128/Max128` | `Reverse2`, `DupOdd` |
| `ReciprocalNewton/ReciprocalSqrtNewton` | `ApproximateReciprocal`, `ApproximateReciprocalSqrt` |
| `Rol/Ror` | `RotateLeft/RotateRight` |
| `CompressBits/LoadExpand` | `LoadMaskBits` / `Expand` |

**P0 实现(8 op × 2 后端,原生或平凡合成)**:`AddSub`(IfThenElse+even_mask,两后端已有 even_mask)、`DupEven/DupOdd`(SVE svtrn1/2(v,v);RVV vrgather (i&~1)/(i|1))、`Reverse2`(RVV vrgather idx^1;SVE svtbl)、`ApproximateReciprocal(+Sqrt)`(SVE svrecpe/svrsqrte;RVV vfrec7/vfrsqrt7)、`RotateLeft/Right`(shift-or 合成)。complex 最终采用 Highway 的 `DupEven/DupOdd + MulAdd/Sub + OddEven` 直接公式，避免 `ComplexConj + Reverse2 + AddSub` 中间重排。

### ✅ P1 — 变宽 swizzle 族(高价值,ARM/RISC-V CAE/图形;原生 intrinsic,18 op)
| op | approach(SVE / RVV)|
|---|---|
| `TableLookupBytes(Or0)`, `TableLookupLanes` | `svtbl`(+svsel)/ `vrgather` |
| `Slide1Up/Down`, `SlideUp/DownLanes` | `svtbl` 运行期索引 / `vslideup`/`vslidedown`，partial 尾部显式清零 |
| `LoadInterleaved2/3/4` + `StoreInterleaved2/3/4` | `svld2/3/4`+`svst2/3/4` / `vlseg2/3/4e`+`vsseg`(**原生分段,AoS↔SoA 大赢**)|
| `InterleaveEven/Odd` | `svtrn1/2` / vrgather+even-odd mask |
| `Reverse4/8`, `ReverseLaneBytes` | `svtbl` 组反索引 / `vrgather` |

### ✅ P2 — 变宽 widen/quantize + reduce(15 op)
`PromoteEvenTo/OddTo`、`TruncateTo`、`U8FromU32`、`ReorderDemote2To`、`ReorderWidenMulAccumulate`、`SatWidenMulPairwiseAdd`、`SumsOf2/4/8`、`ReduceMul`、`SumOfMulQuadAccumulate`、`MulFixedPoint15`、`SaturatedAbs/Neg` 均已落地。SVE i8 quad-dot 走 `svdot`；RVV 用两级 widen pairwise；ReduceMul 用支持非 2 次幂 lane 数的上升步长 Mul+slide。Q15 两后端采用同一 high/low 分解，明确保持 HYLE scalar/x86 的 `MIN*MIN -> MIN` 契约，而不是 RVV `vsmul` 的饱和 `MAX`。

**验收**：本轮 cloc 净增 **+911 生产 / +688 测试 / +5 benchmark = +1,604 code LOC**。SVE 15 tests × VL 128/256/384/512、RVV 12 tests × VLEN 128/256/512/1024 全绿；SVE VL2048 的 P2/partial、Win/WSL fixed SIMD、WSL fixed-backend ASan、Apple-Silicon NEON 169/169、AVX512 SDE 169/169 复验通过；NEON 主/full/widen/Q15/compact 真机 bench 修复并追平 u8/i16/i32/i64 `Compress/Expand`。运行期 VL 断言确认每个 SVE gate 的真实宽度；对抗复审另修 partial masked/gather/scatter、partial PromoteUpper/Demote 与 NEON Q15 极值分歧。

### ✅ P5 — 变宽平凡合成 + mask 操作(易,15 op；[`simd-15`](./simd-15-scalable-p5.md) 已完成)
`BroadcastSignBit`、`ExclusiveNeither`、`IsEitherNaN`、`FindKnownFirst/LastTrue`、`MulRound`、`BroadcastLane`(svdup_lane / vrgather)、`MulByPow2`(SVE 原生 svscale；RVV store+reload ldexp)、`GetExponent`(指数域位运算)、`SetBeforeFirst/OnlyFirst/AtOrBefore/AtOrAfterFirst`(SVE svbrk 系原生;RVV vfirst+iota 合成)、`Expand`(SVE store+reload；RVV viota+vrgather)、`TrailingZeroCount`(SVE svrbit+svclz；**RVV 归 Zvbb 门控不做**)。SVE 18 tests / RVV 15 tests × 4 VL 全绿。

### ✅ 变宽硬件门控翻案 —— [`simd-16`](./simd-16-hw-gated-reclaim.md)（2026-07-11）
> 纪律纠错(`feedback_infra_capability_completeness`)：qemu-user 实测**支持 Zvbb/Zvbc**,"无硬件不做"对这些站不住；已翻案补做+真验。
- **RVV `LeadingZeroCount`/`TrailingZeroCount`/`ReverseBits`/`Rotate`**:✅ **已落地**——`#if defined(__riscv_zvbb)` + `-march=rv64gcv_zvbb` + `qemu -cpu rv64,v=true,zvbb=true`；objdump 见 `vclz.v`/`vctz.v`/`vbrev.v`/`vror.vi`/`vrol.vx`。
- **RVV `CLMulLower`/`CLMulUpper`**:✅ **已落地**——`#if defined(__riscv_zvbc)`；`DupEven/DupOdd`+`vclmul`/`vclmulh`+`OddEven` 对齐 128-bit block 语义；objdump 见 `vclmul.vv`/`vclmulh.vv`。
- `LoadMaskBits`/`StoreMaskBits`/`BitsFromMask`:✅ **已做**(2026-07-12 dbf88c0)——SVE/RVV 用 **byte-buffer** API(`Store/LoadMaskBits(d, uint8_t*)`,1 bit/lane,同 Highway)绕开 uint64 装不下 256 lane;`BitsFromMask` 保留 uint64 便利版(Lanes≤64,否则 FATAL_ASSERT 引导用 byte 版)。SVE VL=2048(i8 256 lane)+ RVV VLEN=1024(i8 128 lane)真跑 round-trip。
- **RVV `PopulationCount`**:✅ **已做**(2026-07-12,b0ad404)——翻案:Zvbb `vcpop.v` qemu 8.2.2 就正确执行(之前"no-op"是我 vmv.x.s+exit-code harness 读错);GCC-14 缺 C wrapper → inline-asm。rvv_test 4 VLEN 对拍 std::popcount + objdump 原生 vcpop.v。
- **SVE2 `AES`/`CLMul`**:✅ **已做**(2026-07-12,8c466f8)——翻案:qemu 8.2.2 `-cpu max` 就执行(之前"Property not found"是用错 `sve2-aes=on` + 探针栈溢出误判)。8 op 全套 native svaese/svaesmc/svpmull;sve_test crypto tier 4 VL + objdump 原生指令。
- **fp8 硬件 convert(AVX10.2)**:✅ **f16 源 + f32 源两方向都已上架 bit-identical**(2026-07-12 69efa5f + 3fe8407)。f16 源:f16→f32 无损→`f32_to_f8(f16_to_f32)` 只舍入一次=与 AVX10.2 硬件同款单舍入=**bit-identical**。`f16_to_f8e4m3_batch`/`f16_to_f8e5m2_batch` 公开 API;scalar 默认(任意平台 bit-exact)+ AVX10.2 硬件核(clang-20 `-mavx10.2-512`,CMake `check_cxx_compiler_flag` 门控 HYLE_HAVE_AVX10_2,GCC-14/MSVC 无 flag 走 scalar)+ 运行期 `cpu_features().avx10_2`(CPUID leaf 0x24)门控。**穷举全 65536 f16 实测:E4M3 0 差;E5M2 硬件核补 inf/NaN mask(Intel 饱和 vs OCP 保 inf)后 0 差** → 透明 dispatch 目标(结果不随 CPU 变)。SDE -dmr 真跑 tier=20 + 0 mismatch。**f32→fp8 方向也已上架 bit-identical(2026-07-12 3fe8407)**:无 f32 直转 fp8 指令,走 f32→f16→fp8,双舍入用 **round-to-odd** 中间步骤消除(Boldo-Melquiond;f16 10 尾数位 ≥ fp8 2-3+2;同 Highway F64→F16 DemoteTo 手法:清 13 丢弃位、有余则 sticky 置保留 LSB)+ f32 源 inf/NaN fixup。可移植 scalar 穷举 486.6M 值 0 mismatch;clang-20+SDE 真跑 HW 核(llvm-objdump 坐实 vcvtps2phx+vcvtph2hf8s/bf8s,native avx10_2=0/SDE=1 双向坐实)0 mismatch。**先前"f32 源必双舍入不上架"结论已推翻**。
- **Per4LaneBlockShuffle 8/16-bit**:✅ **已做**(2026-07-12,1ec8d87)——纯软件无硬件墙,directive2 复核后补:avx2/avx512 vpshufb 字节索引、sse4/neon/scalar store+reload 组循环泛化。五平台真跑。
- scalable 后端**无 ASan 通道**:qemu-user + `-fsanitize=address` shadow-memory 段错误(实测),store+reload op 靠逻辑正确而非 ASan 根治,真边界。
### ⚪ 变宽不该做(定宽专属语义)
- `ConcatLowerLower/LowerUpper/UpperLower/UpperUpper`:⚠️ **文档定性待复核**——原判"128-bit-block 相对,scalable 无意义"可能有误:Highway 对 SVE(arm_sve Splice+MaskLowerHalf)和 RVV(Get/Set)**均实现了整向量半-concat**(与 128-bit block 无关)。待逐语义核对后决定翻案(见下方"待拍板"),暂留此处。
- `CompressEmulated`/`GatherIndexEmulated`:avx2-only A/B bench 变体(真不做)。

## 2. 信源③:Highway/xsimd 有、HYLE 无(去噪后,非偏门)

P3 前机械快照为258个公开 op；P3 新增24个公开名后为282个，P4 再新增12个后为294个。以下经源码核实。

### ✅ P3 — 定宽 Masked* 家族补全(24 op)

已补齐24个可独立落地的公开名：

- 8个 shift：`MaskedShl/MaskedShlOr/MaskedShr/MaskedShrOr`、`MaskedShiftLeft/Right` 及各自 `Or` 形；
- 3个 reduce：`MaskedReduceSum/Min/Max`，全假 mask identity 分别为0、类型上界、类型下界；
- 4个转换/近似/位计数：`MaskedApproximateReciprocal(+Sqrt)`、`MaskedLeadingZeroCount`、`MaskedConvertTo`；
- 2个定点/加宽乘：`MaskedMulFixedPoint15`、`MaskedWidenMulPairwiseAdd`；
- 5个 complex：`MaskedMulComplex`、`MaskedMulComplexOr`、`MaskedMulComplexConj`、`MaskedMulComplexAdd`、`MaskedMulComplexConjAdd`；
- 2个 Highway 短名 parity alias：`MaskedSatAddOr/MaskedSatSubOr`。

非 `Or` 形 false lane 为0，`Or` 形 false lane 取 `no`；HYLE compute-family 参数顺序保持 `(mask, no, ...)`。`MaskedModOr` 已在 P4 基于 i32/u32 invariant `Divisor` 的 `Mod` 落地；`Divisor(0)` fail-fast，mask false lane 只返回 `no`。

**性能闭环**：同TU/同ISA `bench-stat.py -n15`逐个覆盖24/24公开名；其中Highway没有同名zeroing `MaskedMulComplex/MaskedMulComplexAdd`，对照使用其等价`IfThenElseZero(MulComplex*)`组合，其余22项直接对同名API。AVX2与Apple NEON全部统计打平。NEON首轮`MaskedMulComplexConjAdd`明确慢18.5%（0.249 vs 0.203 ns/op）；把底层complex从组合式改为Highway的FMA友好直接公式后，完整复测为0.204 vs 0.204，ratio 1.000、合并spread 6.9%，回到统计打平。

**验收**：`cloc --diff`实测净增 **+107 production / +140 tests / +83 bench = +330 code LOC**；Windows/WSL scalar 239 passed+1硬件skip，SSE4/AVX2/SSE4-AES/AVX512-SDE与Apple NEON各173/173，WSL ASan三变体全绿；complex底层改动另由SVE 15 tests×4 VL、RVV 12 tests×4 VLEN回归。

### ✅ P4 — 定宽 Mul/Mod/Magnitude(12 op)
- FMA 符号变体：`MulSub`、`NegMulSub`、`MulAddSub`、`MulSubAdd`。浮点复用 FMA 保持单次舍入；整数经无符号位表示保持模算术。
- 加宽偶奇乘：`MulEven/MulOdd` 覆盖 i/u8→16、i/u16→32、i/u32→64；`MulEvenAdd/MulOddAdd` 覆盖 packed-bf16 even/odd→f32 fused accumulate。
- `Mod` 与 `MaskedModOr` 复用已有 Granlund-Montgomery `Divisor/Div`，范围明确为 invariant i32/u32；`INT_MIN / -1` 余数为零。
- `MinMagnitude/MaxMagnitude` 覆盖 signed/unsigned 与浮点；`INT_MIN` magnitude 正确，浮点相等 magnitude 与 signed-zero tie 对齐同 ISA Highway。

**性能闭环**：Windows AVX2 三组独立`N=15`中，`Mod`、`MaskedModOr`、`MulEven`、`MulOdd`每组均为统计 TIE。Apple NEON 初版 `MulHigh32` 有额外解交织/窄化，`Mod`慢21.9%、`MaskedModOr`慢18.1%；改为 Highway 同形 `vmull[_high] + vuzp2` 后，三组独立`N=15`均复现前两项为 HYLE 1.073x、1.058x WIN，两个 widen multiply 均为 TIE。

### ✅ 消费驱动 / 低优 —— **已在 simd-17 批5 / simd-18 批E 补齐**
字节移位族(`ShiftLeftBytes/RightBytes`、`CombineShiftRightBytes/Lanes`)、`Combine/LowerHalf/UpperHalf/ZeroExtendVector/InsertIntoUpper`(半向量合并族)、`PairwiseAdd/Sub(128)`、`Shuffle2301/1032/...`(定 permute)、`Stream`、`LoadDup128`、`ResizeBitCast` **全已实现**(avx2.h/vec.h;vec_test 有 `ShiftBytesAndLanes`/`CombineShiftRightBytes`/`PairwiseAddSub`/`ShuffleFixedPatterns32`/`Shuffle01_64`/`StreamAndLoadDup128`/`HalfVectorFamily` 对拍)。

### ✅ 公开 op 面差集审计(2026-07-12,d6b1d13)——4 个薄封装补齐
对齐 Highway `HWY_API` 公开 op 面的机械差集复核,发现 4 个公开 op 在 HYLE 泛型层缺一个**具名入口**,底层原语全已存在,补薄封装(`vec.h` "Highway public-op-face parity" 块):
- `SetTableIndices(d, ptr)` = `IndicesFromVec(d, LoadU(d, ptr))`——从指针加载 TableLookupLanes 索引;
- `CompressBlocksNot(v, m)` = `CompressNot`(Highway 的 u64-only 别名);
- `ZeroExtendResizeBitCast(dTo, dFrom, v)` = `ResizeBitCast`(HYLE 的 ResizeBitCast 已零填充多余目标字节,直接转发);
- `Dup128MaskFromMaskBits(d, bits)`:取低 128-block 位复制到每个 128-bit block 后 `LoadMaskBits`(单 block=恒等)。

`vec_test` `HighwayParityWrappers` 各自对拍底层原语,五平台真跑通过(scalar/sse4/avx2 GCC-14 + avx512 SDE + MSVC + NEON mac),全 217 Vec 测试无回归。

**~~同批曾登记的"有意排除"已翻案(2026-07-12 d488813)~~**:单向量 `PromoteTo`/`DemoteTo`/`ShiftRightAndDemoteTo`/`RoundingShiftRightAndDemoteTo`(Highway Rebind-tag 形,输入输出同 lane 数、在半寄存器里)**已补**——之前"HYLE 无半宽 DemoteTo 单实体"是过保守:它们是现有原语的薄封装(`DemoteTo(dn,v)=LowerHalf(DemoteTo(Twice<dn>,v,v))`、`PromoteTo(dw,v)=PromoteLowerTo(ZeroExtendVector(v))`),整数饱和+f64→f32 都覆盖。单向量形其实是 Highway 最常用的 promote/demote 写法。返回 partial 向量(kPow2=-1),HYLE Store 只吃 `Simd<T,N,0>`,读回用 `DFromV<decltype(r)>`;全寄存器 half-at-a-time 仍是原生首选主路径。`SingleVectorPromoteDemote` 对拍,五平台+ASan。

### 🟰 NAMING-DIFF(已有,别当缺口)
- `Round/Ceil/Floor/Trunc`:`avx2.h` 宏 `HYLE_SIMD_AVX2_ROUND` 实例化,**已有**(grep 假阴)。
- `PromoteTo`/`DemoteTo`(单向量一次性):HYLE 原生首选仍是 half-at-a-time `PromoteLowerTo/UpperTo`(`simd-design.md` 批7),但 Highway 单向量便利形**已补为薄封装**(2026-07-12 d488813,见上"翻案"条)——曾登记"有意差异非缺口"是过保守。
- 128-cmp 全家族 `Lt128/Lt128Upper/Eq128/Ne128/Min128/Max128`:`vec.h` **全有**(op 名搜索;行号勿硬引会漂移)。
- Complex 算术 `ComplexConj/MulComplex/MulComplexConj/MulComplexAdd/MulComplexConjAdd`:`vec.h` **定宽已有**。

### ⚪ OBSCURE(跳过) — simd-17 批5 复核修正
初版把 `BitShuffle`/`MultiRotateRight` 混进本类是**错分类**(误判为 AVX512-only 门控)。复核 Highway 源:二者是 `generic_ops-inl.h` 的**公开 `HWY_API` 通用 op**(每 target 有可移植 fallback),**已在 simd-17 批5 实现**(五后端 fallback + AVX3_DL native VPSHUFBITQMB/VPMULTISHIFTQB,SDE 双路对拍 oracle + objdump 坐实)。`SumsOf8AbsDiff`/`BroadcastBlock`/`ExtractBlock` 同批已做。
仍跳过(坐实):
- `SwapAdjacentBlocks`/`InterleaveWholeLower/Upper`/`LoadTransposedBlocks*` 等 `*Blocks`:Highway 512-bit block-lane 内部编排机制,越 HYLE"贴硬件不过度抽象"scope,无消费者。
- `GaloisAffine`/`GaloisAffineTranspose`:Highway `namespace detail` **内部 GFNI 原语**(非公开 op),其消费者 ReverseBits/RotateRight 已在 HYLE 实现。**注:不做的理由是"非公开面缺口",非"无法验证"——SDE 探针实测 GFNI `-future` 正确执行**。
- ~~`*InRange*`/`Ordered*Demote2*`/`SumsOf*QuadAbsDiff`~~ **已在 simd-18 实现**(批C InRange+OrderedDemote2To、批F dbsad;`OrderedDemote2MasksTo` 随 @opus 裁决 c45c6af)。
- `Dup128VecFromValues`:**已实现**(simd-18 d418ebf,泛型层四 lane 宽重载调 LoadDup128 广播,五平台真跑)。
- `Per4LaneBlockShuffle`:✅ **全宽已实现**(1/2/4/8-byte,五后端)——32/64-bit 用 native shuffle/permute,8/16-bit(2026-07-12,1ec8d87)用 vpshufb 字节索引(avx2/avx512)/store+reload 组循环(sse4/neon/scalar)。`vec_test` Per4LaneBlockShuffle8And16 逐 block 逐组对拍。

## 3. 信源④:文档登记缺口(交叉核对,含纠错)

- **gelu/softmax**：已完成（simd-09 #67，`math_test` Activations/Softmax）。
- **fp8 硬件 convert**:✅ **f16 源 + f32 源两个方向都已上架 bit-identical**(AVX10.2,clang-20 门控;69efa5f + 3fe8407);scalar 位实现是任意平台 bit-exact 默认。
- **复数超越**：`csin/ccos/ctanh/cpow` 已在 AVX2/NEON 上架 HYLE SoA 编排；`ctanh/cpow` 分别消除复三角/双曲重复计算和 complex-pow 的重复 `log/exp`，异常 lane 逐 lane 使用 `std::complex`。`cexp/clog/csqrt`：x86 借 xsimd AVX2，**ARM 已在 simd-17 批5 补 `cexp/clog/csqrt_f32_hyle_neon`（xsimd batch<complex> NEON 核）**，dispatch ARM64 从 scalar 改指 hyle_neon。**ARM 复数缺口已闭**（待 mac 真机复验）。
- **lgamma**:退 scalar libm(xsimd 向量 lane-mixing vendor bug);tgamma 已向量化。无 action。
- **ReciprocalNewton**:HYLE 超越两库的点(它们只出裸估计)——保持。
- complex/128-cmp/Newton 已由 scalable P0 直接实例化，不再是定宽 only。

## 4. 优先级施工单

| 优先 | 内容 | op 数 | 价值 |
|---|---|---|---|
| **P0 ✅** | 解锁 scalable 泛型层:AddSub/DupEven/DupOdd/Reverse2/ApproxRecip(+Sqrt)/Rotate × SVE+RVV | 8 | 已完成 |
| **P1 ✅** | scalable swizzle 族(TableLookup/Slide/Interleaved load-store/InterleaveEven-Odd/Reverse4-8)| 18 | 已完成 |
| **P2 ✅** | scalable widen/quantize/reduce(Promote-Even-Odd/Truncate/Demote2/SumsOf/ReduceMul/svdot/Q15/SatAbs-Neg)| 15 | 已完成 |
| **P3 ✅** | 定宽 Masked* 补全(Shl/Shr/Reduce/ApproxRecip/Complex/... +Or 形)| 24 | 已完成，性能无输家 |
| **simd-12 ✅** | 自研热点核竞争 + math/sort/bit-pack小数组调用税收口 | — | 已完成；不改变op完成数。11/13 bit-pack、csin/ccos、small-K std路由均已裁决并上架 |
| **P4 ✅** | 定宽 Mul/Mod/Magnitude(MulSub/AddSub/EvenOdd、Mod、Min/MaxMagnitude)| 12 | 已完成；同 ISA 性能无输家 |
| **P5 ✅** | scalable 平凡合成 + mask 操作(BroadcastSignBit/ExclusiveNeither/IsEitherNaN/MulRound/GetExponent/BroadcastLane/Set*First/FindKnown*/MulByPow2/Expand/TZCnt)| 15 | 已完成([`simd-15`](./simd-15-scalable-p5.md))；RVV TZCnt 归 Zvbb 不做 |
| **simd-16 ✅** | 翻案 RVV Zvbb/Zvbc：LZCnt/TZCnt/ReverseBits/Rotate + CLMul | ~6-7 | 已完成([`simd-16`](./simd-16-hw-gated-reclaim.md))；qemu 三档×4 VLEN + objdump 真指令 |
| **simd-17 ✅** | 广度补全:批1 Shuffle/字节移位、批2 *Same/Pairwise/Stream、批3 scalable Concat、批4 半向量合并族、批5 复数ARM+SAD+BitShuffle/MultiRotateRight+Block提取 | — | 已完成([批5 doc](./simd-17-breadth-batch5.md));批4/5 待 Win MSVC + mac 平台复验 |
| ✅ 2026-07-12 翻案做掉 | RVV PopulationCount(b0ad404)、SVE2 AES/CLMul(8c466f8)、Per4Lane 8/16(1ec8d87) | — | 均"工具墙"复核推翻(我的 harness 错),qemu 8.2.2 本就够 |
| ✅ 2026-07-12 上架(f16+f32 源) | fp8-HW-convert:f16/f32_to_f8e4m3/e5m2_batch + AVX10.2 硬件核(69efa5f + 3fe8407) | — | 两方向都 bit-identical;f16 源单舍入,f32 源 round-to-odd 消双舍入(Boldo-Melquiond);SDE 0 mismatch + 486.6M 穷举 |
| ✅ 2026-07-12 补(用户点探) | 单向量 `PromoteTo`/`DemoteTo`/`ShiftRightAndDemoteTo`(Highway Rebind-tag 形,d488813) | 4 | 现有原语薄封装(LowerHalf∘DemoteTo2 / PromoteLowerTo∘ZeroExtend);曾登记"有意排除"是过保守 |
| ✅ 2026-07-12 做掉 | mask-bits VL>64(SVE/RVV byte-buffer StoreMaskBits/LoadMaskBits,dbf88c0) | — | 曾以为真墙,byte-buffer API 解掉;已无真墙 |
| ⚪不做 | 其余 *Blocks(SwapAdjacent等) / GaloisAffine(Highway内部detail非公开面) | — | 偏门/越 scope/非公开面 |

P0-P5、[`simd-12`](./simd-12-vendor-displacement.md)、[`simd-13`](./simd-13-vendor-displacement-2.md)、[`simd-14`](./simd-14-fixed-width-p4.md)、[`simd-15`](./simd-15-scalable-p5.md)与[`simd-16`](./simd-16-hw-gated-reclaim.md)已全部完成。**主施工闭集软件面 100%**；simd-16 翻案 RVV Zvbb/Zvbc（TZCnt/LZCnt/ReverseBits/Rotate/CLMul），directive2(2026-07-12)再翻案 **RVV PopulationCount + SVE2 AES/CLMul 全套 + Per4Lane 8/16**(先前"工具墙"是 harness bug)。directive2 又用 byte-buffer 解掉 mask-bits VL>64(dbf88c0)。两轮 vendor displacement 不新增公开 op；P4 新增 12 个、P5 新增 15 个公开 op。**已无真墙,能力面全覆盖**;fp8-HW-convert 的 **f16 源 + f32 源两个方向都已上架 bit-identical**(69efa5f + 3fe8407;f32 源用 round-to-odd 消双舍入)。2026-07-12 又补单向量 Promote/Demote(d488813)——翻案两处此前我自己下过的过保守"有意排除"。**剩余真有意排除仅 `GaloisAffine*`**(Highway `detail::` 内部 GFNI 原语非公开面)。


## 5. 复现方法

```bash
# 每后端 op 集(宏感知需谨慎——见方法纪律)
for be in sse4 avx2 avx512 neon sve rvv scalar; do
  grep -hE "HYLE_FORCE_INLINE" include/hyle/simd/backend/$be.h \
    | grep -oE "[A-Z][A-Za-z0-9_]*\(" | sed 's/(//' | sort -u > /tmp/ops_$be.txt
done
comm -23 /tmp/ops_avx2.txt /tmp/ops_sve.txt   # 定宽有、SVE 无
# ⚠️ 结果必逐项对 sve.h/rvv.h 源码核实(宏定义会假阴)
```
op 集快照存 `project-hyle/simd_gap_scratch/`(本次生成)。

## 参考
- `include/hyle/simd/vec.h`(泛型层)+ `backend/{sve,rvv,avx2}.h`(逐 op 核实源)。
- `/c/public/project-hyle/highway/hwy/ops/{generic_ops,x86_256}-inl.h`(Highway op 全集,HWY_API)。
- `docs/architecture/simd-design.md` + `docs/milestones/simd-0{5,7,8,9}-*.md`(信源④)。
- memory `project_simd_int_lanes`(op 补全全史)+ `project_simd_sve_rvv_feasibility`(scalable 可行性)。
