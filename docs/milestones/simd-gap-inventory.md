# SIMD 缺口清单（下一个 session 的施工蓝图）

> **这份文档回答:HYLE simd 到底还差什么。** 2026-07-10 用机械差集 + subagent 逐源验证编撰,作为**下一个 session 补 simd 的施工检查单**。覆盖四个信源:①消费者驱动延后项 ②标量/非标量 + 定长/变宽对称性漏项(本应对称)③xsimd/highway 有而 HYLE 无的非偏门 op ④文档明确登记的缺口。
>
> **方法纪律(必读)**:原始 `grep` op 名差集**有噪音**——各后端宏风格不同(scalar 用 `HYLE_SIMD_SCALAR_BINOP(Add,...)`、avx2 用 `HYLE_SIMD_AVX2_ROUND(...)`,裸 grep 会漏报"缺 Add/Round")。**本清单每一项都经后端源码核实**,分类为:🔴 REAL-GAP(施工目标)/ 🟰 NAMING-DIFF(已有,别做)/ 🔒 HW-GATED(硬件门控,登记不做)/ ⚪ OBSCURE(偏门,跳过)。下个 session 施工前,对将做的每一项**再核一次当前源码**(可能已被后续 commit 补上)。

## 0. 一句话结论

- **定宽四后端(sse4/avx2/avx512/neon)op 面严格一致**(各 ~166-168,仅 avx2 多 2 个 A/B bench 变体)——**这块没问题**。
- **scalar 完整**(是 oracle;"缺 Add/Mul" 是宏定义的 grep 假阳)——**没问题**。
- **真问题 = 变宽(SVE/RVV)显著落后定宽 ~72 op**,且大多不是硬件借口。
- **⚠️ 最要命的发现(headline)**:vec.h 的"泛型层一份 covers 五后端"对 scalable **是假的**——已发货的 complex / 128-cmp / Newton 泛型 op **依赖 SVE/RVV 没定义的底层原语**,今天只因 scalable TU 从没实例化它们才没崩;一旦 ARM/RISC-V 消费者(FFT/复数、vqsort-128)碰到就编译失败。**这不是覆盖问题,是潜在正确性/可构建性问题**,应作为下轮第一优先。

## 1. 信源①②:变宽 vs 定宽对称性缺口(最大块)

精确差集(宏感知提取,已去 crypto + Emulated):avx2 有而 **SVE 缺 72 / RVV 缺 74**(RVV 另缺 `PopulationCount`/`LeadingZeroCount`)。每项已对 `sve.h`/`rvv.h` 源码核实确缺。

### ⚠️ P0 — 解锁已发货的泛型层(正确性,非仅覆盖)
vec.h 这些**已发货**的泛型 op 组合在 SVE/RVV **未定义**的底层原语上(subagent `grep -c`=0 坐实),现在编译过只因 scalable TU 不实例化;消费者一碰即断:

| 泛型 op(vec.h)| 依赖的、SVE/RVV 缺的原语 |
|---|---|
| `MulComplex/MulComplexAdd/MulComplexConj*`(596-619)| `AddSub`, `DupEven`, `DupOdd`, `Reverse2` |
| `Lt128/Eq128/Ne128/Min128/Max128`(777-812)| `Reverse2`, `DupOdd` |
| `ReciprocalNewton/ReciprocalSqrtNewton`(438-448)| `ApproximateReciprocal`, `ApproximateReciprocalSqrt` |
| `Rol/Ror`(470-471)| `RotateLeft/RotateRight` |
| `CompressBits/LoadExpand`(500-519)| `LoadMaskBits` / `Expand` |

**P0 施工项(7 op × 2 后端,全有原生或平凡合成)**:`AddSub`(IfThenElse+even_mask,两后端已有 even_mask)、`DupEven/DupOdd`(SVE svtrn1/2(v,v);RVV vrgather (i&~1)/(i|1))、`Reverse2`(RVV vrgather idx^1;SVE svtbl)、`ApproximateReciprocal(+Sqrt)`(SVE svrecpe/svrsqrte;RVV vfrec7/vfrsqrt7)、`RotateLeft/Right`(shift-or 合成)。

### 🔴 P1 — 变宽 swizzle 族(高价值,ARM/RISC-V CAE/图形;原生 intrinsic 存在,~20 op)
| op | approach(SVE / RVV)|
|---|---|
| `TableLookupBytes(Or0)`, `TableLookupLanes` | `svtbl`(+svsel)/ `vrgather` |
| `Slide1Up/Down`, `SlideUp/DownLanes` | `svext`/`svsplice` / `vslideup`/`vslidedown`(原生) |
| `LoadInterleaved2/3/4` + `StoreInterleaved2/3/4` | `svld2/3/4`+`svst2/3/4` / `vlseg2/3/4e`+`vsseg`(**原生分段,AoS↔SoA 大赢**)|
| `InterleaveEven/Odd` | `svtrn1/2` / vrgather+even-odd mask(rvv.h 已有 helper)|
| `Reverse4/8`, `ReverseLaneBytes` | svtbl 组反索引 / vrgather;SVE `svrevb` |

### 🔴 P2 — 变宽 widen/quantize + reduce(~12 op)
`PromoteEvenTo/OddTo`、`TruncateTo`(带 ASan 栈-OOB 单平台陷阱教训)、`U8FromU32`、`ReorderDemote2To`(lane 序未定义契约,scalable 实现须注明其序)、`ReorderWidenMulAccumulate`、`SatWidenMulPairwiseAdd`、`SumsOf2/4/8`、`ReduceMul`(SVE/RVV 都无积归约原语 → log-step Mul+slide 级联合成)、`SumOfMulQuadAccumulate`(SVE `svdot` 原生 i8 点积)、`MulFixedPoint15`(svqrdmulh/vsmul)、`SaturatedAbs/Neg`(svqabs/svqneg 原生)。

### 🔴 P5 — 变宽平凡合成 + mask 操作(易,~12 op)
`BroadcastSignBit`、`ExclusiveNeither`、`IsEitherNaN`、`FindKnownFirst/LastTrue`、`MulRound`、`BroadcastLane`(svdup_lane / vrgather)、`MulByPow2`、`GetExponent`(指数域位运算)、`SetBeforeFirst/OnlyFirst/AtOrBefore/AtOrAfterFirst`(SVE svbrk 系原生;RVV vfirst+iota 合成)、`Expand`、`TrailingZeroCount`(SVE svrbit+svclz)。

### 🔒 变宽硬件门控 / 语义受限(登记不做)
- Crypto `AES*`/`CLMul`:SVE2 `+sve2-aes` / RVV Zvbc(backlog 已登记)。
- RVV `PopulationCount`/`LeadingZeroCount`/`TrailingZeroCount`/`ReverseBits`/`Rotate`:base rv64gcv 无,需 Zvbb(`vcpop/vclz/vctz/vbrev/vror`)。
- `LoadMaskBits`/`StoreMaskBits`/`BitsFromMask`:scalable VL 可达 2048-bit(i8=256 lane)超 uint64 位域容量,语义受限。
### ⚪ 变宽不该做(定宽专属语义)
- `ConcatLowerLower/LowerUpper/UpperLower/UpperUpper`:128-bit-block 相对操作,scalable 无意义(sve.h:530 明确不提供);**scalable 提供的是整向量 `ConcatEven/ConcatOdd`**。
- `CompressEmulated`/`GatherIndexEmulated`:avx2-only A/B bench 变体。

## 2. 信源③:Highway/xsimd 有、HYLE 无(去噪后,非偏门)

HYLE 全集(所有后端+vec.h 泛型+宏展开 Masked=258 op)。以下经源码核实。

### 🔴 P3 — 定宽 Masked* 家族补全(便宜,一个宏块;兑现 Highway parity 声称)
HYLE 实有 **35 个** Masked*(非文档说的 ~40)。Highway 有而 HYLE 缺、且便宜:
`MaskedShl/Shr/ShiftLeft/ShiftRight(+Or)`(~8,Shl/Shr/ShiftLeft/ShiftRight 都在)、`MaskedReduceSum/Min/Max`、`MaskedApproximateReciprocal(+Sqrt)`、`MaskedLeadingZeroCount`、`MaskedConvertTo`、`MaskedMulComplex(Conj)(Add)`、`MaskedMulFixedPoint15`、`MaskedWidenMulPairwiseAdd`、`MaskedModOr/SatAddOr/SatSubOr`。约 25 个,全包已有 op。

### 🔴 P4 — 定宽 Mul/Mod/Magnitude(中价值中成本)
- FMA 符号变体(易,FMA+Neg 合成):`MulSub`、`MulAddSub`、`MulSubAdd`、`NegMulSub`(HYLE 只有 MulAdd/NegMulAdd)——复数/DSP 好用。
- 加宽偶奇乘:`MulEven/MulOdd/MulEvenAdd/MulOddAdd`(定点/大数)。
- `Mod`(+`MaskedModOr`):向量整数取模,扩展已有 Divisor/Div(Granlund-Montgomery)机制。高价值中成本。
- `MinMagnitude/MaxMagnitude`(易,IfThenElse(Lt(Abs,Abs)))。

### 🔴 消费驱动 / 低优(按需)
字节移位族(`ShiftLeftBytes/RightBytes`、`CombineShiftRightBytes/Lanes`,palignr,字符串/解析)、`Combine/LowerHalf/UpperHalf/ZeroExtendVector/InsertIntoUpper`(半向量合并族,窄化管线才需)、`PairwiseAdd/Sub(128)`(hadd,DSP)、`Shuffle2301/1032/...`(定 permute 模式,32-bit lane 有用)、`Stream`(非临时存)、`LoadDup128`、`ResizeBitCast`。

### 🟰 NAMING-DIFF(已有,别当缺口)
- `Round/Ceil/Floor/Trunc`:avx2.h:472 宏定义,**已有**(grep 假阴)。
- `PromoteTo`/`DemoteTo`(单向量一次性):HYLE 刻意用 half-at-a-time `PromoteLowerTo/UpperTo`(simd-design:147 设计决策),Highway 单向量版是便利包装。**登记为有意差异,非缺口**。
- 128-cmp 全家族 `Lt128/Lt128Upper/Eq128/Ne128/Min128/Max128`:vec.h:777 **全有**。
- Complex 算术 `ComplexConj/MulComplex/MulComplexConj/MulComplexAdd/MulComplexConjAdd`:vec.h:596 **定宽已有**(文档 Source④ 说"未做"是**过期信息**)。

### ⚪ OBSCURE(跳过)
所有 `*Blocks` 族(BroadcastBlock/ExtractBlock/InterleaveWholeLower/SwapAdjacentBlocks/LoadTransposedBlocks* 等,Highway 512-bit block-lane 机制,越 HYLE"贴硬件不过度抽象"scope)、`*InRange*`(UB 规避变体)、`GaloisAffine`/`BitShuffle`/`MultiRotateRight`/`Per4LaneBlockShuffle`/`Ordered*Demote2*`/`SumsOf*QuadAbsDiff`(GFNI/视频专用,HW-gated 或极窄)。

## 3. 信源④:文档登记缺口(交叉核对,含纠错)

- **gelu/softmax**:文档一处说"消费驱动缓"——**过期,已做**(simd-09 #67,math_test Activations/Softmax)。
- **fp8 硬件 convert**:AVX10.2/GPU 专属;**scalar 位实现已完整**,非缓做。
- **复数超越** `batch<complex>` exp/log/pow/sqrt/sin/cos/tanh:**定宽已借**(simd-09 线①-d);**但 ARM 是真缺口**(Highway 无复数路径,arm64 曾回落 scalar,xsimd-NEON 复数未验)。
- **lgamma**:退 scalar libm(xsimd 向量 lane-mixing vendor bug);tgamma 已向量化。无 action。
- **ReciprocalNewton**:HYLE 超越两库的点(它们只出裸估计)——保持。
- **⚠️ 文档普遍把 complex/128-cmp/Newton 当"全局已做",实为定宽 only**(见 §1 P0)——本清单纠正之。

## 4. 优先级施工单(下 session 检查单)

| 优先 | 内容 | op 数 | 价值 |
|---|---|---|---|
| **P0** | 解锁 scalable 泛型层:AddSub/DupEven/DupOdd/Reverse2/ApproxRecip(+Sqrt)/Rotate × SVE+RVV | ~7×2 | **最高**(修潜在断裂,非仅覆盖)|
| **P1** | scalable swizzle 族(TableLookup/Slide/Interleaved load-store/InterleaveEven-Odd/Reverse4-8)| ~20 | 高(ARM/RISC-V CAE/图形,原生 intrinsic)|
| **P2** | scalable widen/quantize/reduce(Promote-Even-Odd/Truncate/Demote2/SumsOf/ReduceMul/svdot/Q15/SatAbs-Neg)| ~12 | 高(量化/点积)|
| **P3** | 定宽 Masked* 补全(Shl/Shr/Reduce/ApproxRecip/Complex/... +Or 形)| ~25 | 中(兑现 Highway parity,一个宏块)|
| **P4** | 定宽 Mul/Mod/Magnitude(MulSub/AddSub/EvenOdd、Mod、Min/MaxMagnitude)| ~10 | 中 |
| **P5** | scalable 平凡合成 + mask 操作 | ~12 | 中低(易)|
| 按需 | 字节移位/半向量合并/Pairwise/Shuffle/Stream/xsimd-NEON 复数 | — | 低 |
| 🔒登记不做 | SVE2 crypto/RVV Zvbb/mask-bits VL>64/fp8-HW-convert | — | 硬件门控 |
| ⚪不做 | 所有 *Blocks / *InRange / GaloisAffine / 视频专用 | — | 偏门/越 scope |

**总施工量(P0-P5,不含按需/门控/偏门)≈ 86 op**(变宽约占 63,定宽 Masked/Mul/Mod 约 35 里去重)。**建议顺序 P0→P1→P2 先把 scalable 拉到定宽对称(这是"对不上"的主体),再 P3-P4 补定宽对 Highway 的零头。**

## 5. 复现方法(下 session 起手可重跑)

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
