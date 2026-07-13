# SIMD 对拍全量换向统计（2026-07-13 复跑，当前代码）

> 每个 op × 每平台 × 每序的输赢一目了然。benchmark 本身内建 order-swap（env `HYLE_BENCH_HWYFIRST`）：
> 每 op 在"HYLE 先跑"(hf)和"Highway 先跑"(hwf)两序各测。列 **win=MSVC(Windows AVX2)、
> wsl=GCC AVX2、mac=NEON**；每列两子列 = 两序的 HYLE/Highway op/s 比值（>1=HYLE 快）+ 标签
> **W**(赢>2%)/**L**(输>2%)/**=**(平±2%)/**·**(该平台无此 op)。N=6 中位。

## 判定口径（真输的定义）
- **某平台"两序都 L" = 该平台真输**：换向后 HYLE 仍慢，排除了"第一槽冷启动"harness 伪影。
- **x86(avx2/sse4) 与 NEON 是各自独立的后端实现**，一个 op 完全可能"x86 真输、NEON 赢"（反之亦然）。
  因此**真输按平台判**——x86 上的真输不因它在 NEON 上赢而免责，反过来也一样。
- 不存在"必须三平台全慢才算真输"的门槛；只要**任一平台两序都慢**就是那个平台需修的真输。

## 权威统计（当前代码，本会话全部修复已并入）
- 总 benched op = **266**
- **真输 op 数（至少一个平台两序都慢）= 64**
  - **win (MSVC AVX2) = 48**   |   **wsl (GCC AVX2) = 23**   |   **mac (NEON) = 5**
  - 三平台同时真输 = **0**（NEON 独立实现，x86 输的项在 NEON 上大多 W）
- 其余：平/赢 = 125，HYLE 全序赢 = 16，混合(仅单序慢/布局伪影) = 61

### x86 双序真输（win & wsl 都两序慢，10 个）——最硬核心，优先修
| op | win | wsl | 初判根因 |
|---|---|---|---|
| BitShuffle u64 | 0.59/0.59 | 0.97/0.96 | 标量; 8-blend byte-rotate 向量化在 MSVC 反降(已回退),需 Highway 精确 u8-Rol |
| LoadN/StoreN i32 | 0.59/0.61 | 0.68/0.72 | sub-128 partial load/store 基础设施缺失 |
| MultiRotateRight u64 | 0.90/0.90 | 0.91/0.91 | 已向量化仍落后,需再核 |
| MaskedApproxReciprocal f32 | 0.86/0.86 | 0.96/0.91 | **IfThenElseZero=blendv, Highway=vpand** |
| MaskedApproxReciprocalSqrt f32 | 0.83/0.81 | 0.95/0.96 | 同上(Masked 族公因) |
| MaskedMulComplex f32 | 0.86/0.86 | 0.98/0.96 | 同上 |
| MaskedShiftLeft<3> i32 | 0.94/0.92 | 0.97/0.93 | 同上(此前"objdump 同 Highway"需复核) |
| MulByFloorPow2 f32 | 0.94/0.96 | 0.96/0.97 | 待 objdump |
| CompressStore i32 | 0.95/0.89 | 0.90/0.96 | 已弃精确写契约,内存-bound 残留 |
| Eq128 u64 | 0.96/0.97 | 0.92/0.93 | 128-cmp 族,内存-bound 残留 |

### mac (NEON) 真输（5 个）
| op | mac | 备注 |
|---|---|---|
| FindFirstTrue i32 | 0.89/0.83 | NEON 无 movemask,需核 Highway aarch64 手法 |
| CeilInt f32->i32 | 0.95/0.95 | |
| MaskedShiftRight<3> i32 | 0.95/0.95 | Masked 族(NEON 侧) |
| Lt128Upper u64 | 0.93/0.93 | 128-cmp(NEON 侧) |
| RoundingShiftRightSame i32 bits=5 | 0.98/0.98 | 临界 |

> ⚠️ 两点必读：
> (1) 此 harness 是**内存-bound**(Load+op+Store)。个别"真输"项在 compute-bound 链式或 realistic
>     4-累加器 streaming kernel 下会翻正(如 128-cmp 族 Max128 已 compute-bound 1.04×)。按用户口径
>     **realistic kernel 换向仍输才是真输**；本表是内存-bound harness 下的"真输上界",逐项修时需再跑
>     realistic kernel 复核。
> (2) MSVC 真输数(48)远高于 GCC(23)：MSVC 对 store+reload/标量展开 codegen 优化弱于 GCC,第一槽惩罚更强。
>     **同一 VS2026 编译, Highway 快 = HYLE 源码写法问题,非 VC++ 版本问题。**

---

## 全量表（真输在前，然后混合，最后平/赢；同组内按 op 名排序）

| op | win(hf) | win(hwf) | wsl(hf) | wsl(hwf) | mac(hf) | mac(hwf) | comment |
|---|---|---|---|---|---|---|---|
| BitShuffle u64 (4096) | 0.59L | 0.59L | 0.97L | 0.96L | · | · | 真输@win,wsl |
| CompressStore i32 (4096, 50%) | 0.95L | 0.89L | 0.90L | 0.96L | · | · | 真输@win,wsl |
| Eq128 u64 | 0.96L | 0.97L | 0.92L | 0.93L | 1.05W | 1.05W | 真输@win,wsl |
| LoadN/StoreN i32 (half lanes) | 0.59L | 0.61L | 0.68L | 0.72L | · | · | 真输@win,wsl |
| Lt128Upper u64 | 0.96L | 1.00= | 0.93L | 0.95L | 0.93L | 0.93L | 真输@wsl,mac |
| MaskedApproxReciprocal f32 | 0.86L | 0.86L | 0.96L | 0.91L | 1.12W | 1.11W | 真输@win,wsl |
| MaskedApproxReciprocalSqrt f32 | 0.83L | 0.81L | 0.95L | 0.96L | 1.14W | 1.14W | 真输@win,wsl |
| MaskedMulComplex f32 | 0.86L | 0.86L | 0.98L | 0.96L | 1.00= | 1.00= | 真输@win,wsl |
| MaskedShiftLeft<3> i32 | 0.94L | 0.92L | 0.97L | 0.93L | 1.00= | 1.00= | 真输@win,wsl |
| MaskedShiftRight<3> i32 | 0.95L | 0.95L | 1.10W | 0.96L | 0.95L | 0.95L | 真输@win,mac |
| MulByFloorPow2 f32 | 0.94L | 0.96L | 0.96L | 0.97L | 1.00= | 1.00= | 真输@win,wsl |
| MultiRotateRight u64 (4096) | 0.90L | 0.90L | 0.91L | 0.91L | · | · | 真输@win,wsl |
| AllBits0 i32 | 0.96L | 0.95L | 1.00= | 1.00= | 1.00= | 1.00= | 真输@win |
| BitCast i32->f32 | 0.89L | 0.92L | 1.01= | 0.96L | 0.99= | 0.99= | 真输@win |
| BroadcastLane<3> i32 | 0.92L | 0.91L | 1.29W | 1.26W | 1.16W | 1.16W | 真输@win |
| CeilInt f32->i32 | 1.51W | 1.53W | 1.47W | 1.45W | 0.95L | 0.95L | 真输@mac |
| CombineMasks i32 (4096) | 1.06W | 1.08W | 0.88L | 0.87L | · | · | 真输@wsl |
| ConcatLowerLower i32 | 1.05W | 1.03W | 0.94L | 0.94L | · | · | 真输@wsl |
| FindFirstTrue i32 | 1.04W | 1.06W | 1.01= | 0.98= | 0.89L | 0.83L | 真输@mac |
| FindKnownFirstTrue i32 | 0.92L | 0.90L | 1.00= | 1.01= | 1.18W | 1.08W | 真输@win |
| FindLastTrue i32 | 1.06W | 1.06W | 0.96L | 0.94L | 1.05W | 1.05W | 真输@wsl |
| HighestSetBitIndex u32 | 0.88L | 0.98= | 0.87L | 0.96L | 1.00= | 1.00= | 真输@wsl |
| IsNegative i32 | 0.96L | 0.99= | 0.84L | 0.83L | 1.00= | 1.00= | 真输@wsl |
| LoadNOr i32 (half n) | 0.73L | 1.05W | 0.67L | 0.66L | · | · | 真输@wsl |
| Lt128 u64 | 0.90L | 0.98= | 0.97L | 0.98L | 1.05W | 1.05W | 真输@wsl |
| MaskedLeadingZeroCount i32 | 0.96L | 0.95L | 1.08W | 1.10W | 1.15W | 1.14W | 真输@win |
| MaskedLoad i32 (4096, 50%) | 0.94L | 0.95L | 1.02= | 1.01= | · | · | 真输@win |
| MaskedModOr i32 invariant | 1.00= | 1.00= | 0.97L | 0.97L | 1.06W | 1.06W | 真输@wsl |
| MaskedMulComplexConj f32 | 0.87L | 0.84L | 0.96L | 0.98= | 1.00= | 1.00= | 真输@win |
| MaskedMulComplexConjAdd f32 | 0.93L | 0.95L | 0.99= | 1.01= | 1.11W | 1.11W | 真输@win |
| MaskedShiftLeftOr<3> i32 | 0.97L | 0.92L | 1.01= | 1.01= | 1.00= | 1.00= | 真输@win |
| MaskedShiftRightOr<3> i32 | 0.95L | 0.96L | 1.01= | 1.01= | 1.00= | 1.00= | 真输@win |
| MaskedShl i32 | 0.87L | 0.88L | 1.02W | 1.03W | 1.11W | 1.11W | 真输@win |
| MaskedShr i32 | 0.92L | 0.89L | 1.04W | 1.03W | 1.09W | 1.09W | 真输@win |
| MaskedWidenMulPairwiseAdd i16-i32 | 0.84L | 0.81L | 1.04W | 1.00= | 1.09W | 1.09W | 真输@win |
| Min f32 | 1.01= | 1.04W | 0.97L | 0.97L | 1.11W | 1.11W | 真输@wsl |
| MulComplexAdd f32 | 0.96L | 0.98L | 1.02W | 1.02= | 1.12W | 1.12W | 真输@win |
| MulComplexConjAdd f32 | 0.95L | 0.96L | 1.00= | 0.98L | 1.12W | 1.12W | 真输@win |
| Ne128 u64 | 0.95L | 0.88L | 1.00= | 0.98= | 1.05W | 1.05W | 真输@win |
| Neg i32 | 0.91L | 0.93L | 1.00= | 0.98= | 1.38W | 1.39W | 真输@win |
| OddEvenBlocks i32 | 0.97L | 0.98L | 0.97L | 1.02W | · | · | 真输@win |
| OrderedDemote2To i32->i16 | 0.91L | 0.92L | 1.00= | 0.99= | · | · | 真输@win |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 0.96L | 0.96L | 1.01= | 1.00= | · | · | 真输@win |
| Per4LaneBlockShuffle 3210 i32 | 0.90L | 0.92L | 0.99= | 1.02= | 1.14W | 1.14W | 真输@win |
| ReorderDemote2To i32->i16 | 0.92L | 0.91L | 0.97L | 1.00= | · | · | 真输@win |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 0.91L | 0.93L | 1.00= | 1.05W | · | · | 真输@win |
| RoundingShiftRight<2> i32 | 0.85L | 0.85L | 1.09W | 1.07W | 1.12W | 1.12W | 真输@win |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.91L | 0.90L | 1.12W | 1.12W | · | · | 真输@win |
| RoundingShiftRightSame i32 bits=5 | 1.18W | 1.20W | 1.08W | 1.04W | 0.98L | 0.98L | 真输@mac |
| RoundingShr i32 | 1.04W | 1.01= | 0.97L | 0.97L | 1.10W | 1.10W | 真输@wsl |
| SatWidenMulPairwiseAccumulate i16->i32 | 0.77L | 0.77L | 0.99= | 0.99= | · | · | 真输@win |
| SetAtOrAfterFirst i32 | 0.86L | 0.84L | 1.06W | 1.02= | 1.00= | 1.00= | 真输@win |
| SetOnlyFirst i32 | 0.85L | 0.85L | 1.64W | 1.58W | 1.00= | 1.00= | 真输@win |
| ShiftLeft<3> i32 | 0.95L | 0.98= | 0.96L | 0.95L | 1.16W | 1.16W | 真输@wsl |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.98L | 0.98L | 0.99= | 0.98= | · | · | 真输@win |
| Shuffle0321 i32 | 0.96L | 0.97L | 0.98L | 1.01= | 1.13W | 1.13W | 真输@win |
| SlideMask1Down i32 (4096) | 0.90L | 0.93L | 0.99= | 1.01= | · | · | 真输@win |
| SlideMask1Up i32 (4096) | 0.93L | 0.90L | 1.00= | 0.99= | · | · | 真输@win |
| StoreInterleaved3 f32 (4096) | 0.91L | 0.97L | 1.03W | 1.09W | · | · | 真输@win |
| SumOfLanes i32 | 0.96L | 0.93L | 1.30W | 1.29W | 1.11W | 1.12W | 真输@win |
| TestBit i32 | 0.94L | 0.96L | 1.01= | 1.02= | 1.00= | 1.00= | 真输@win |
| TrailingZeroCount i32 | 1.05W | 1.04W | 0.95L | 0.89L | 1.09W | 1.10W | 真输@wsl |
| TruncateTo u32->u16 | 0.87L | 0.87L | 1.01= | 1.06W | 1.16W | 1.16W | 真输@win |
| ZeroIfNegative f32 | 0.88L | 0.93L | 1.13W | 1.10W | 1.14W | 1.16W | 真输@win |
| AllBits1 i32 | 1.04W | 1.04W | 0.97L | 0.99= | 1.00= | 1.00= | 混合(单序/布局) |
| BlendedStore i32 (4096, 50%) | 0.99= | 0.93L | 1.02= | 1.00= | · | · | 混合(单序/布局) |
| BroadcastBlock<1> i32 | 1.05W | 1.10W | 0.96L | 1.02W | · | · | 混合(单序/布局) |
| Ceil f32 | 0.94L | 1.00= | 1.01= | 1.01= | 1.12W | 1.12W | 混合(单序/布局) |
| CombineShiftRightLanes1 i32 | 0.98L | 1.00= | 1.01= | 0.99= | 1.09W | 1.09W | 混合(单序/布局) |
| CompressNot i32 (4096, 50%) | 0.90L | 1.00= | 0.99= | 0.97L | · | · | 混合(单序/布局) |
| ConcatLowerUpper i32 | 0.98= | 0.97L | 0.98= | 1.02= | · | · | 混合(单序/布局) |
| ConcatUpperLower i32 | 0.99= | 0.96L | 1.00= | 1.01= | · | · | 混合(单序/布局) |
| ConvertTo i32->f32 | 1.02W | 0.99= | 1.02W | 0.94L | 1.01= | 1.01= | 混合(单序/布局) |
| CountTrue i32 | 0.97L | 0.99= | 1.04W | 1.00= | 1.01= | 1.01= | 混合(单序/布局) |
| DemoteTo i32->i16 1-arg (4096) | 0.98L | 0.99= | 0.98= | 1.02W | · | · | 混合(单序/布局) |
| DupOdd i32 | 0.99= | 0.98L | 1.00= | 0.98= | 1.12W | 1.12W | 混合(单序/布局) |
| Eq i32 | 0.99= | 0.99= | 0.99= | 0.97L | 1.10W | 1.11W | 混合(单序/布局) |
| FindKnownLastTrue i32 | 0.97L | 1.00= | 1.00= | 1.01= | 1.22W | 1.16W | 混合(单序/布局) |
| GatherIndexN i32 (half n) | 1.02W | 0.95L | 1.00= | 0.99= | · | · | 混合(单序/布局) |
| GatherOffset i32 (4096) | 0.99= | 0.96L | 1.01= | 0.99= | · | · | 混合(单序/布局) |
| Ge i32 | 1.04W | 1.02= | 0.96L | 0.99= | 1.07W | 1.07W | 混合(单序/布局) |
| GetExponent f32 | 1.01= | 1.00= | 0.97L | 1.02W | · | · | 混合(单序/布局) |
| Gt i32 | 0.98L | 1.01= | 1.02W | 1.02= | 1.11W | 1.11W | 混合(单序/布局) |
| IfNegativeThenElseZero f32 | 1.01= | 0.98L | 1.10W | 1.05W | 1.11W | 1.12W | 混合(单序/布局) |
| InterleaveOdd i32 | 0.99= | 0.99= | 0.98L | 1.00= | 1.11W | 1.12W | 混合(单序/布局) |
| IsEitherNaN f32 | 1.01= | 1.00= | 0.97L | 1.02= | 1.00= | 1.00= | 混合(单序/布局) |
| IsNaN f32 | 0.92L | 1.01= | 1.02W | 1.02= | 1.00= | 1.00= | 混合(单序/布局) |
| Load/Store i32 aligned | 1.01= | 1.01= | 1.00= | 1.01= | 1.00= | 0.90L | 混合(单序/布局) |
| LoadInterleaved2 f32 (4096) | 1.01= | 1.00= | 1.00= | 0.96L | · | · | 混合(单序/布局) |
| Mask Xor i32 | 1.03W | 1.00= | 1.00= | 0.97L | 1.00= | 1.00= | 混合(单序/布局) |
| MaskedMulComplexAdd f32 | 0.98L | 0.99= | 0.95L | 0.99= | 1.04W | 1.04W | 混合(单序/布局) |
| MaskedMulFixedPoint15 i16 | 0.96L | 1.00= | 1.08W | 1.04W | 1.12W | 1.12W | 混合(单序/布局) |
| MaskedSatAddOr i16 | 0.95L | 1.00= | 0.98= | 1.02= | 1.09W | 1.09W | 混合(单序/布局) |
| MaskedShlOr i32 | 0.97L | 1.00= | 1.00= | 1.00= | 1.09W | 1.09W | 混合(单序/布局) |
| MaskedShrOr i32 | 0.97L | 0.99= | 1.00= | 1.01= | 1.09W | 1.09W | 混合(单序/布局) |
| Max f32 | 1.01= | 1.01= | 1.00= | 0.98L | 1.13W | 1.13W | 混合(单序/布局) |
| Max128 u64 | 1.00= | 0.98= | 0.94L | 1.03W | 1.14W | 1.14W | 混合(单序/布局) |
| MaxMagnitude f32 | 1.04W | 1.02W | 1.00= | 0.97L | 1.14W | 1.14W | 混合(单序/布局) |
| MaxOfLanes i32 | 0.98L | 0.98= | 1.28W | 1.35W | 1.15W | 1.15W | 混合(单序/布局) |
| Min128 u64 | 1.02= | 1.01= | 0.98= | 0.97L | 1.14W | 1.14W | 混合(单序/布局) |
| Mod i32 invariant | 0.99= | 1.00= | 1.00= | 0.97L | 1.07W | 1.07W | 混合(单序/布局) |
| Mul i32 | 0.98L | 1.00= | 0.97L | 1.01= | 1.11W | 1.11W | 混合(单序/布局) |
| MulAddSub f32 | 1.02W | 1.01= | 0.99= | 0.96L | 1.13W | 1.13W | 混合(单序/布局) |
| MulFixedPoint15 i16 | 1.03W | 1.03W | 0.96L | 1.00= | · | · | 混合(单序/布局) |
| MulHigh i16 (4096) | 0.99= | 1.02= | 0.97L | 0.99= | · | · | 混合(单序/布局) |
| MulHigh32 u32 (4096) | 0.99= | 0.96L | 0.99= | 1.00= | · | · | 混合(单序/布局) |
| MulRound f32 | 1.01= | 0.98= | 0.98= | 0.97L | 1.13W | 1.13W | 混合(单序/布局) |
| Not i32 | 1.01= | 1.03W | 0.97L | 1.02W | 1.13W | 1.13W | 混合(单序/布局) |
| OrAnd i32 | 0.96L | 1.01= | 1.00= | 1.00= | 1.00= | 1.00= | 混合(单序/布局) |
| PromoteEvenTo i8->i16 | 0.97L | 0.98= | 1.00= | 1.01= | · | · | 混合(单序/布局) |
| PromoteTo i16->i32 1-arg (4096) | 1.05W | 1.04W | 1.00= | 0.97L | · | · | 混合(单序/布局) |
| RebindMask i32->f32 (4096) | 1.12W | 1.12W | 1.03W | 0.96L | · | · | 混合(单序/布局) |
| Reverse8 i32 (4096) | 0.99= | 0.88L | 1.01= | 0.97L | · | · | 混合(单序/布局) |
| RotateLeftSame u32 bits=5 | 0.98= | 0.98= | 0.99= | 0.98L | 1.10W | 1.10W | 混合(单序/布局) |
| SaturatedAbs i32 | 0.95L | 0.98= | 1.04W | 1.00= | 1.12W | 1.12W | 混合(单序/布局) |
| ScatterIndex i32 (4096) | 1.01= | 1.01= | 1.00= | 0.97L | · | · | 混合(单序/布局) |
| ScatterOffset i32 (4096) | 1.08W | 1.06W | 0.97L | 1.01= | · | · | 混合(单序/布局) |
| SetBeforeFirst i32 | 0.98L | 0.99= | 1.21W | 1.24W | 1.00= | 1.00= | 混合(单序/布局) |
| ShiftLeftBytes<4> i32 | 1.00= | 0.99= | 0.98L | 1.05W | 1.16W | 1.15W | 混合(单序/布局) |
| ShiftRightSame i32 bits=3 | 0.98L | 1.02= | 1.00= | 1.00= | 1.10W | 1.10W | 混合(单序/布局) |
| SumsOf2 u8->u16 | 1.07W | 1.17W | 0.98= | 0.96L | · | · | 混合(单序/布局) |
| SumsOf8 u8->u64 | 0.81L | 1.00= | 1.00= | 0.98= | · | · | 混合(单序/布局) |
| SumsOf8AbsDiff u8->u64 | 0.91L | 1.02W | 0.98= | 0.95L | · | · | 混合(单序/布局) |
| SumsOfAdjQuadAbsDiff u8->u16 | 0.71L | 1.03W | 0.99= | 0.99= | · | · | 混合(单序/布局) |
| TableLookupLanes i32 (4096) | 0.97L | 1.01= | 0.99= | 1.00= | · | · | 混合(单序/布局) |
| Abs i32 | 0.99= | 1.00= | 0.99= | 1.01= | 1.12W | 1.10W | 平/赢 |
| AbsDiff i32 | 1.00= | 0.99= | 1.03W | 1.00= | 1.10W | 1.10W | 平/赢 |
| Add f32 | 1.00= | 0.99= | 1.01= | 1.00= | 1.14W | 1.14W | 平/赢 |
| AddSub i32 | 1.02= | 1.01= | 1.01= | 1.02= | 1.10W | 1.10W | 平/赢 |
| AllFalse i32 | 1.08W | 1.11W | 1.01= | 0.98= | 1.00= | 1.00= | 平/赢 |
| AllTrue i32 | 1.07W | 1.06W | 1.00= | 1.03W | 1.01= | 1.01= | 平/赢 |
| And i32 | 0.99= | 0.99= | 1.01= | 1.00= | 1.08W | 1.08W | 平/赢 |
| AndNot i32 | 1.00= | 1.04W | 0.99= | 1.03W | 1.09W | 1.09W | 平/赢 |
| ApproximateReciprocal f32 | 1.01= | 0.99= | 1.00= | 0.98= | 1.15W | 1.16W | 平/赢 |
| ApproximateReciprocalSqrt f32 | 1.00= | 1.01= | 1.02W | 1.00= | 1.15W | 1.16W | 平/赢 |
| AverageRound u16 | 1.02W | 1.02W | 1.01= | 1.02= | 1.14W | 1.14W | 平/赢 |
| BitCast f32->i32 | 1.01= | 0.99= | 1.02= | 0.99= | 1.00= | 1.00= | 平/赢 |
| Broadcast<0> i32 | 1.03W | 1.06W | 1.14W | 1.12W | 1.15W | 1.15W | HYLE赢(所有序) |
| BroadcastSignBit i32 | 1.00= | 1.00= | 1.00= | 1.00= | 1.12W | 1.12W | 平/赢 |
| Clamp i32 | 1.00= | 1.02W | 1.00= | 1.07W | 1.09W | 1.09W | 平/赢 |
| CombineShiftRightBytes4 i32 | 1.00= | 1.01= | 1.00= | 1.00= | 1.11W | 1.11W | 平/赢 |
| ComplexConj f32 | 1.01= | 1.00= | 1.00= | 0.99= | 1.15W | 1.15W | 平/赢 |
| CompressBlendedStore i32 (4096, 50%) | 1.10W | 0.99= | 1.04W | 1.10W | · | · | 平/赢 |
| ConcatOdd i32 | 1.02= | 1.02= | 1.00= | 0.99= | 1.11W | 1.11W | 平/赢 |
| ConcatUpperUpper i32 | 0.99= | 1.00= | 1.00= | 1.01= | · | · | 平/赢 |
| ConvertTo f32->i32 | 1.28W | 1.31W | 1.18W | 1.22W | 1.01= | 1.01= | 平/赢 |
| CopySign f32 | 1.05W | 1.03W | 1.02= | 0.99= | 1.13W | 1.13W | 平/赢 |
| DemoteTo i32->i16 (sat) | 1.61W | 1.59W | 1.20W | 1.18W | · | · | HYLE赢(所有序) |
| Div f32 | 1.02= | 1.00= | 1.00= | 0.99= | 1.13W | 1.13W | 平/赢 |
| Dup128MaskFromMaskBits i32 (4096) | 1.01= | 1.02= | 1.19W | 1.21W | · | · | 平/赢 |
| DupEven i32 | 1.00= | 0.99= | 1.00= | 1.01= | 1.12W | 1.12W | 平/赢 |
| ExclusiveNeither i32 | 1.16W | 1.16W | 1.00= | 0.99= | 1.00= | 1.00= | 平/赢 |
| Floor f32 | 1.06W | 1.02W | 1.01= | 1.00= | 1.11W | 1.12W | 平/赢 |
| FloorInt f32->i32 | 1.55W | 1.53W | 1.42W | 1.45W | 1.27W | 1.27W | HYLE赢(所有序) |
| GatherIndexNOr i32 (half n) | 1.02= | 0.99= | 1.00= | 0.99= | · | · | 平/赢 |
| GetBiasedExponent f32 | 1.00= | 1.02W | 1.16W | 1.14W | 1.00= | 1.00= | 平/赢 |
| IfNegativeThenElse f32 | 0.98= | 0.99= | 1.01= | 1.00= | 1.11W | 1.11W | 平/赢 |
| IfNegativeThenZeroElse f32 | 1.02W | 1.02= | 1.12W | 1.05W | 1.12W | 1.12W | 平/赢 |
| IfThenElse i32 | 1.02W | 1.03W | 1.01= | 1.01= | 1.11W | 1.11W | 平/赢 |
| IfThenElseZero i32 | 1.05W | 1.08W | 1.09W | 1.09W | 1.15W | 1.16W | HYLE赢(所有序) |
| IfThenZeroElse i32 | 1.00= | 1.02W | 1.12W | 1.06W | 1.15W | 1.15W | 平/赢 |
| InterleaveEven i32 | 1.02= | 0.99= | 1.01= | 1.02= | 1.10W | 1.10W | 平/赢 |
| InterleaveLowerBlocks i32 | 1.00= | 1.02W | 1.01= | 1.01= | · | · | 平/赢 |
| InterleaveUpper i32 | 1.02= | 0.99= | 1.01= | 0.99= | 1.10W | 1.10W | 平/赢 |
| InterleaveWholeLower i32 | 1.00= | 1.00= | 1.00= | 1.02= | 1.09W | 1.09W | 平/赢 |
| InterleaveWholeUpper i32 | 1.01= | 0.99= | 1.04W | 1.03W | 1.10W | 1.10W | 平/赢 |
| IsFinite f32 | 1.03W | 1.03W | 1.36W | 1.40W | 1.00= | 1.00= | 平/赢 |
| IsInf f32 | 1.10W | 1.11W | 1.12W | 1.20W | 1.00= | 1.00= | 平/赢 |
| Le i32 | 1.03W | 1.03W | 1.00= | 1.02= | 1.09W | 1.09W | 平/赢 |
| LoadExpand i32 (4096, 50%) | 2.12W | 2.30W | 2.18W | 2.16W | · | · | HYLE赢(所有序) |
| LoadU/StoreU i32 | 1.00= | 1.01= | 1.01= | 1.02W | 1.13W | 1.13W | 平/赢 |
| Lt i32 | 1.01= | 1.00= | 1.00= | 1.02= | 1.10W | 1.10W | 平/赢 |
| Mask And i32 | 1.01= | 1.02= | 0.99= | 0.99= | 1.00= | 1.00= | 平/赢 |
| Mask AndNot i32 | 1.02= | 1.01= | 1.00= | 0.99= | 1.00= | 1.00= | 平/赢 |
| Mask Not i32 | 1.01= | 1.01= | 0.99= | 0.99= | 1.00= | 1.00= | 平/赢 |
| Mask Or i32 | 1.04W | 1.04W | 1.00= | 1.00= | 1.00= | 1.00= | 平/赢 |
| MaskedConvertTo f32-i32 | 1.41W | 1.41W | 1.37W | 1.35W | 1.05W | 1.05W | HYLE赢(所有序) |
| MaskedLoadOr i32 (4096) | 1.00= | 1.02W | 0.99= | 1.01= | · | · | 平/赢 |
| MaskedMulComplexOr f32 | 1.01= | 0.99= | 1.07W | 1.01= | 1.00= | 1.00= | 平/赢 |
| MaskedReduceMax i32 | 0.99= | 1.00= | 0.99= | 1.00= | 1.00= | 1.00= | 平/赢 |
| MaskedReduceMin i32 | 1.00= | 0.99= | 1.00= | 1.00= | 0.99= | 0.99= | 平/赢 |
| MaskedReduceSum i32 | 1.11W | 1.11W | 1.09W | 1.09W | 0.99= | 0.99= | 平/赢 |
| MaskedSatSubOr i16 | 0.99= | 1.00= | 1.01= | 1.02= | 1.11W | 1.11W | 平/赢 |
| Max i32 | 0.99= | 1.00= | 0.99= | 1.02= | 1.08W | 1.08W | 平/赢 |
| MaxNumber f32 | 0.99= | 1.03W | 1.01= | 0.99= | 1.13W | 1.13W | 平/赢 |
| MinMagnitude f32 | 1.02= | 1.03W | 1.01= | 0.98= | 1.13W | 1.13W | 平/赢 |
| MinNumber f32 | 1.01= | 1.03W | 0.99= | 0.99= | 1.11W | 1.11W | 平/赢 |
| MinOfLanes i32 | 0.99= | 1.00= | 1.33W | 1.34W | 1.12W | 1.12W | 平/赢 |
| Mul f32 | 0.99= | 1.02= | 1.00= | 0.99= | 1.13W | 1.13W | 平/赢 |
| MulByPow2 f32 | 0.98= | 1.00= | 1.01= | 0.99= | 1.00= | 1.00= | 平/赢 |
| MulComplex f32 | 1.00= | 1.01= | 1.07W | 1.01= | 1.02= | 1.02= | 平/赢 |
| MulComplexConj f32 | 1.01= | 1.01= | 1.11W | 1.08W | 1.11W | 1.11W | 平/赢 |
| MulEven i32-i64 | 0.98= | 1.00= | 1.02= | 1.01= | 0.99= | 1.01= | 平/赢 |
| MulHigh u16 (4096) | 1.01= | 1.01= | 1.00= | 1.00= | · | · | 平/赢 |
| MulOdd i32-i64 | 0.99= | 0.99= | 1.00= | 1.03W | 0.99= | 1.00= | 平/赢 |
| MulSub f32 | 1.01= | 1.00= | 1.00= | 1.01= | 1.12W | 1.12W | 平/赢 |
| MulSubAdd f32 | 1.06W | 1.03W | 1.03W | 1.02= | 1.12W | 1.12W | 平/赢 |
| Ne i32 | 1.03W | 1.01= | 0.99= | 1.00= | 1.11W | 1.11W | 平/赢 |
| NearestInt f32->i32 | 1.23W | 1.29W | 1.20W | 1.21W | 1.00= | 1.00= | 平/赢 |
| NegMulAdd f32 | 0.99= | 1.01= | 1.01= | 1.00= | 1.11W | 1.11W | 平/赢 |
| NegMulSub f32 | 1.01= | 0.99= | 1.00= | 0.98= | 1.12W | 1.12W | 平/赢 |
| OddEven i32 | 1.00= | 0.99= | 0.99= | 1.02= | 1.10W | 1.10W | 平/赢 |
| Or i32 | 1.02W | 1.02W | 1.04W | 1.02= | 1.10W | 1.10W | 平/赢 |
| Or3 i32 | 1.01= | 1.01= | 1.01= | 1.02= | 1.08W | 1.08W | 平/赢 |
| OrderedDemote2MasksTo i32->i16 (4096) | 1.11W | 1.00= | 0.99= | 1.00= | · | · | 平/赢 |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.21W | 1.19W | 1.16W | 1.12W | · | · | HYLE赢(所有序) |
| PairwiseAdd i32 | 1.09W | 1.10W | 1.11W | 1.12W | 1.11W | 1.11W | HYLE赢(所有序) |
| PairwiseAdd128 i32 | 1.01= | 1.01= | 0.99= | 1.01= | 1.07W | 1.07W | 平/赢 |
| PairwiseSub i32 | 1.08W | 1.12W | 1.14W | 1.13W | 1.11W | 1.11W | HYLE赢(所有序) |
| PairwiseSub128 i32 | 1.08W | 1.10W | 1.07W | 1.08W | 1.11W | 1.11W | HYLE赢(所有序) |
| PromoteMaskTo i16->i32 (4096) | 0.99= | 0.99= | 1.01= | 1.00= | · | · | 平/赢 |
| PromoteOddTo i8->i16 | 1.01= | 1.00= | 1.04W | 1.00= | · | · | 平/赢 |
| PromoteUpperTo i16->i32 | 0.99= | 0.99= | 1.02= | 1.01= | · | · | 平/赢 |
| ReduceMax i32 | 1.16W | 1.17W | 1.02= | 1.01= | 1.00= | 1.00= | 平/赢 |
| ReduceMin i32 | 1.20W | 1.16W | 1.03W | 1.02W | 1.00= | 1.00= | 平/赢 |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.09W | 1.10W | 1.22W | 1.20W | · | · | HYLE赢(所有序) |
| ReorderWidenMulAccumulate i16->i32 | 1.01= | 1.02= | 1.00= | 1.01= | · | · | 平/赢 |
| Reverse2 i32 | 1.01= | 1.00= | 1.00= | 1.01= | 1.12W | 1.12W | 平/赢 |
| Reverse4 i32 | 0.98= | 0.98= | 0.98= | 1.00= | 1.12W | 1.12W | 平/赢 |
| ReverseBits i32 | 1.71W | 1.76W | 1.59W | 1.58W | 1.15W | 1.15W | HYLE赢(所有序) |
| ReverseBlocks i32 | 1.03W | 1.07W | 1.04W | 1.01= | · | · | 平/赢 |
| RotateLeft<3> i32 | 1.08W | 1.08W | 1.00= | 0.98= | 1.12W | 1.12W | 平/赢 |
| RotateRight<3> i32 | 1.07W | 1.07W | 1.02W | 1.01= | 1.12W | 1.12W | 平/赢 |
| RotateRightSame u32 bits=5 | 1.01= | 1.03W | 1.00= | 0.98= | 1.09W | 1.09W | 平/赢 |
| Round f32 | 1.00= | 1.03W | 1.02W | 0.99= | 1.12W | 1.12W | 平/赢 |
| RoundToInt f32->i32 | 1.29W | 1.27W | 1.17W | 1.20W | 1.25W | 1.26W | HYLE赢(所有序) |
| SatWidenMulAccumFixedPoint i16->i32 | 1.68W | 1.69W | 2.21W | 2.21W | · | · | HYLE赢(所有序) |
| SatWidenMulPairwiseAdd u8*i8->i16 | 1.00= | 1.02= | 1.01= | 1.02W | · | · | 平/赢 |
| SaturatedNeg i32 | 1.04W | 1.04W | 1.00= | 1.01= | 1.12W | 1.12W | 平/赢 |
| SaturatedSub i16 | 1.00= | 1.00= | 0.99= | 1.09W | 1.14W | 1.14W | 平/赢 |
| ScatterIndexN i32 (half n) | 1.09W | 1.16W | 1.14W | 1.27W | · | · | HYLE赢(所有序) |
| Set i32 | 1.00= | 1.01= | 1.00= | 1.07W | 1.00= | 1.00= | 平/赢 |
| SetAtOrBeforeFirst i32 | 1.10W | 1.09W | 1.19W | 1.17W | 1.00= | 1.00= | 平/赢 |
| SetMask i32 (4096) | 1.01= | 1.00= | 1.01= | 1.00= | · | · | 平/赢 |
| ShiftLeftLanes<1> i32 | 1.03W | 1.00= | 1.01= | 1.01= | 1.12W | 1.12W | 平/赢 |
| ShiftLeftSame i32 bits=3 | 1.03W | 1.00= | 1.02= | 1.00= | 1.11W | 1.11W | 平/赢 |
| ShiftRight<3> i32 | 1.03W | 1.03W | 1.00= | 0.98= | 1.16W | 1.15W | 平/赢 |
| ShiftRightBytes<4> i32 | 1.01= | 1.00= | 1.00= | 1.00= | 1.12W | 1.12W | 平/赢 |
| ShiftRightLanes<1> i32 | 0.99= | 0.99= | 1.00= | 1.02= | 1.15W | 1.15W | 平/赢 |
| Shl i32 | 1.01= | 0.99= | 0.99= | 1.02= | 1.08W | 1.08W | 平/赢 |
| Shr u32 (variable) | 0.99= | 1.02= | 1.03W | 1.00= | 1.15W | 1.15W | 平/赢 |
| Shuffle0123 i32 | 1.00= | 0.99= | 1.00= | 1.02W | 1.15W | 1.15W | 平/赢 |
| Shuffle1032 i32 | 1.01= | 0.99= | 1.01= | 1.03W | 1.13W | 1.13W | 平/赢 |
| Shuffle2103 i32 | 1.01= | 0.99= | 1.02= | 1.03W | 1.13W | 1.13W | 平/赢 |
| Shuffle2301 i32 | 1.00= | 1.00= | 1.00= | 1.00= | 1.13W | 1.13W | 平/赢 |
| Slide1Down i32 | 1.00= | 1.00= | 1.04W | 1.02= | 1.13W | 1.13W | 平/赢 |
| Slide1Up i32 | 1.04W | 1.07W | 1.02W | 1.00= | 1.15W | 1.15W | 平/赢 |
| SlideDownLanes amt=2 i32 | 1.17W | 1.17W | 1.00= | 1.03W | 1.12W | 1.12W | 平/赢 |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.01= | 1.02= | 1.02= | 1.00= | · | · | 平/赢 |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.00= | 0.99= | 1.00= | 0.99= | · | · | 平/赢 |
| SlideUpLanes amt=2 i32 | 1.06W | 1.08W | 1.00= | 1.01= | 1.14W | 1.14W | 平/赢 |
| Sqrt f32 | 0.99= | 1.01= | 1.01= | 0.99= | 1.00= | 1.00= | 平/赢 |
| StoreInterleaved2 f32 (4096) | 1.01= | 1.01= | 1.05W | 1.16W | · | · | 平/赢 |
| Sub f32 | 1.02W | 0.99= | 1.00= | 0.99= | 1.12W | 1.12W | 平/赢 |
| Sub i32 | 1.03W | 1.00= | 1.04W | 1.01= | 1.10W | 1.10W | 平/赢 |
| SumOfMulQuadAccumulate i8->i32 | 1.00= | 1.02W | 1.00= | 0.99= | · | · | 平/赢 |
| SumsOf4 u8->u32 | 1.02= | 1.02W | 1.01= | 1.02= | · | · | 平/赢 |
| SwapAdjacentBlocks i32 | 1.07W | 1.07W | 1.01= | 1.04W | · | · | 平/赢 |
| Trunc f32 | 1.00= | 1.02= | 1.01= | 1.03W | 1.12W | 1.12W | 平/赢 |
| TwoTablesLookupLanes i32 (4096) | 0.99= | 1.00= | 0.99= | 1.06W | · | · | 平/赢 |
| WidenMulAccumulate i16->i32 | 0.99= | 1.01= | 1.03W | 1.02= | · | · | 平/赢 |
| WidenMulPairwiseAdd i16->i32 | 1.00= | 0.99= | 1.00= | 1.01= | · | · | 平/赢 |
| Xor i32 | 0.99= | 0.99= | 1.01= | 1.00= | 1.11W | 1.11W | 平/赢 |
| Xor3 i32 | 1.68W | 1.64W | 1.05W | 1.02W | 1.94W | 1.95W | HYLE赢(所有序) |
| Zero i32 | 1.02W | 1.00= | 1.00= | 1.06W | 1.00= | 1.02= | 平/赢 |
| ZeroExtendVector i16 half->full | 1.05W | 1.04W | 1.01= | 1.00= | · | · | 平/赢 |
