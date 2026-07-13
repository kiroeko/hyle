# SIMD 对拍全量换向统计（2026-07-13 复跑，Masked+LoadN 修复后）

> 每个 op × 每平台 × 每序的输赢一目了然。benchmark 内建 order-swap（env `HYLE_BENCH_HWYFIRST`）：
> 每 op 在"HYLE 先跑"(hf)和"Highway 先跑"(hwf)两序各测。列 **win=MSVC(Windows AVX2)、
> wsl=GCC AVX2、mac=NEON**；每列两子列 = 两序 HYLE/Highway op/s 比值（>1=HYLE 快）+ 标签
> **W**(赢>2%)/**L**(输>2%)/**=**(平±2%)/**·**(该平台无此 op)。N=6 中位。

## 判定口径（真输的定义）
- **某平台"两序都 L" = 该平台真输**：换向后仍慢，排除第一槽冷启动伪影。
- **x86(avx2/sse4) 与 NEON 独立后端**，一个 op 可"x86 真输、NEON 赢"，真输**按平台判**。
- 只要**任一平台两序都慢**就是那个平台需修的真输。
- **⚠️ 本 op_parity harness 是内存-bound(Load+op+Store)**。计算密集 op 会被内存墙淹没判成"输"，
  但在 **realistic 消费者 kernel**（4-累加器 streaming / 真实 transform）下 HYLE 反赢。逐个"真输"必须
  再跑 realistic kernel 复核才能定性——见下方 §已复核反转清单。

## 权威统计（含 IfThenElseZero vpand 修复 f32b7ba + LoadN/StoreN maskload 修复 6176285）
- 总 benched op = **266**
- **op_parity 内存-bound 真输 = 49**（Masked 修复前 64 → 修 Masked 后 50 → 修 LoadN 后 49）
  - **win (MSVC) = 34**   |   **wsl (GCC) = 15**   |   **mac (NEON) = 4**
- 其余：平/赢 135，HYLE 全序赢 14，混合(单序/布局) 68

### §已复核反转清单（op_parity 判"输"但 realistic kernel 实测 HYLE 赢/平 = 内存-bound 伪判）
| op | op_parity | realistic kernel(HYLE/Highway) | 结论 |
|---|---|---|---|
| **HighestSetBitIndex u32** | win 0.89/wsl 0.88 | WSL 1.40-1.48×、MSVC 1.02× | **两平台赢/平,非真输**(LZCnt float-exp trick) |
| **MultiRotateRight u64** | 0.91 both | **WSL 1.82-1.83× 赢**；MSVC 0.91× 慢 | WSL 伪判(赢);**MSVC 真慢**(同算法 MSVC codegen 差) |

### 真正剩余的 x86 硬核真输（realistic 复核后仍慢）
| op | 平台 | 现状 | 根因 |
|---|---|---|---|
| **BitShuffle u64** | MSVC 0.59 | 真慢 | 标量; 8-blend byte-rotate 向量化 MSVC 反降(已回退),需 Highway 精确 u8-Rol |
| **MultiRotateRight u64** | MSVC 0.91 | 真慢(仅 MSVC) | 同泛型算法 GCC 编译后 1.82× 赢, MSVC codegen 差 → 需 MSVC 专项手调或 VBMI multishift 门控 |
| **LoadNOr i32** | WSL 0.68 | 部分改善 | LoadN 已 maskload 修好;LoadNOr 残留需 Highway 递归半宽二分(sub-128 partial-load infra) |

### mac (NEON) 真输（4 个，多为内存-bound 临界）
| op | mac | 备注 |
|---|---|---|
| CeilInt f32->i32 | 0.95/0.96 | 待 realistic 复核 |
| Lt128Upper u64 | 0.91/0.96 | 128-cmp(NEON 侧) |
| MaskedShiftRight<3> i32 | 0.95/0.97 | vshrq_n_s32+vandq=2 指令已最简,内存-bound 噪声 |
| RoundingShiftRightSame i32 bits=5 | 0.98/0.96 | 临界 |

> MSVC 真输数(34)高于 GCC(15)：MSVC 对 store+reload/标量展开/向量序列调度 codegen 弱于 GCC。
> **同一 VS2026 编译 Highway 快 = HYLE 源码写法问题,非 VC++ 版本问题。**

---

## 全量表（真输在前，然后混合，最后平/赢；同组内按 op 名排序）

| op | win(hf) | win(hwf) | wsl(hf) | wsl(hwf) | mac(hf) | mac(hwf) | comment |
|---|---|---|---|---|---|---|---|
| BitShuffle u64 (4096) | 0.59L | 0.59L | 0.98L | 0.97L | · | · | 真输@win,wsl |
| HighestSetBitIndex u32 | 0.97L | 0.92L | 0.87L | 0.91L | 1.00= | 0.97L | 真输@win,wsl |
| Lt128Upper u64 | 0.96L | 1.00= | 0.93L | 0.94L | 0.91L | 0.96L | 真输@wsl,mac |
| MultiRotateRight u64 (4096) | 0.91L | 0.91L | 0.91L | 0.92L | · | · | 真输@win,wsl |
| BroadcastLane<3> i32 | 0.82L | 0.82L | 1.25W | 1.15W | 1.16W | 1.19W | 真输@win |
| CeilInt f32->i32 | 1.43W | 1.69W | 1.39W | 1.26W | 0.95L | 0.96L | 真输@mac |
| CombineMasks i32 (4096) | 1.09W | 1.09W | 0.86L | 0.88L | · | · | 真输@wsl |
| CompressStore i32 (4096, 50%) | 1.00= | 1.16W | 0.98L | 0.98L | · | · | 真输@wsl |
| ConcatLowerLower i32 | 1.03W | 1.02W | 0.92L | 0.94L | · | · | 真输@wsl |
| Eq128 u64 | 0.94L | 0.98= | 0.92L | 0.94L | 1.05W | 1.03W | 真输@wsl |
| FindKnownFirstTrue i32 | 0.85L | 0.85L | 1.00= | 1.01= | 1.17W | 1.25W | 真输@win |
| FindKnownLastTrue i32 | 0.96L | 0.96L | 0.98= | 1.00= | 1.10W | 1.10W | 真输@win |
| FindLastTrue i32 | 1.06W | 1.06W | 0.93L | 0.95L | 1.05W | 1.09W | 真输@wsl |
| InterleaveLowerBlocks i32 | 1.03W | 1.03W | 0.98L | 0.97L | · | · | 真输@wsl |
| IsNaN f32 | 0.97L | 0.97L | 1.00= | 1.00= | 1.00= | 1.01= | 真输@win |
| IsNegative i32 | 0.97L | 1.02= | 0.82L | 0.85L | 1.00= | 0.99= | 真输@wsl |
| LoadN/StoreN i32 (half lanes) | 1.14W | 0.96L | 0.91L | 0.89L | · | · | 真输@wsl |
| LoadNOr i32 (half n) | 0.75L | 1.11W | 0.69L | 0.66L | · | · | 真输@wsl |
| Lt128 u64 | 0.94L | 0.94L | 1.02= | 0.99= | 1.08W | 1.04W | 真输@win |
| MaskedApproxReciprocalSqrt f32 | 0.96L | 0.97L | 0.96L | 1.01= | 1.14W | 1.10W | 真输@win |
| MaskedMulComplexConjAdd f32 | 0.98L | 0.97L | 0.99= | 1.01= | 1.12W | 1.07W | 真输@win |
| MaskedShiftLeftOr<3> i32 | 0.94L | 0.95L | 1.01= | 1.01= | 1.00= | 1.01= | 真输@win |
| MaskedShiftRight<3> i32 | 1.05W | 1.06W | 1.01= | 0.99= | 0.95L | 0.97L | 真输@mac |
| MaskedShiftRightOr<3> i32 | 0.93L | 0.96L | 1.01= | 1.02W | 1.00= | 0.98L | 真输@win |
| MaskedShrOr i32 | 0.96L | 0.96L | 0.99= | 0.99= | 1.09W | 1.08W | 真输@win |
| MaskedWidenMulPairwiseAdd i16-i32 | 0.82L | 0.84L | 1.00= | 1.02W | 1.09W | 1.08W | 真输@win |
| MulByFloorPow2 f32 | 1.01= | 1.00= | 0.96L | 0.97L | 1.00= | 1.00= | 真输@wsl |
| MulComplexAdd f32 | 0.97L | 0.98L | 0.99= | 1.00= | 1.13W | 1.10W | 真输@win |
| MulComplexConjAdd f32 | 0.95L | 0.95L | 0.99= | 0.99= | 1.11W | 1.09W | 真输@win |
| Ne128 u64 | 0.92L | 0.87L | 0.99= | 0.98L | 1.08W | 1.07W | 真输@win |
| Neg i32 | 0.96L | 0.91L | 1.02W | 1.00= | 1.38W | 1.33W | 真输@win |
| OrderedDemote2To i32->i16 | 0.90L | 0.91L | 1.01= | 0.99= | · | · | 真输@win |
| Per4LaneBlockShuffle 3210 i32 | 0.92L | 0.91L | 0.99= | 1.00= | 1.14W | 1.11W | 真输@win |
| PromoteEvenTo i8->i16 | 0.97L | 0.97L | 1.02= | 1.00= | · | · | 真输@win |
| ReorderDemote2To i32->i16 | 0.91L | 0.89L | 1.00= | 0.96L | · | · | 真输@win |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 0.91L | 0.92L | 0.98= | 1.00= | · | · | 真输@win |
| RoundingShiftRight<2> i32 | 0.87L | 0.85L | 1.10W | 1.03W | 1.12W | 1.08W | 真输@win |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.90L | 0.94L | 1.09W | 1.09W | · | · | 真输@win |
| RoundingShiftRightSame i32 bits=5 | 1.19W | 1.19W | 1.05W | 1.06W | 0.98L | 0.96L | 真输@mac |
| SatWidenMulPairwiseAccumulate i16->i32 | 0.77L | 0.74L | 1.01= | 1.01= | · | · | 真输@win |
| SetAtOrAfterFirst i32 | 0.84L | 0.86L | 1.01= | 1.02= | 1.00= | 1.00= | 真输@win |
| SetOnlyFirst i32 | 0.83L | 0.83L | 1.65W | 1.63W | 1.00= | 1.00= | 真输@win |
| ShiftLeft<3> i32 | 0.97L | 0.97L | 1.00= | 1.00= | 1.16W | 1.18W | 真输@win |
| SlideMask1Down i32 (4096) | 0.93L | 0.95L | 1.01= | 1.01= | · | · | 真输@win |
| SlideMask1Up i32 (4096) | 0.90L | 0.91L | 1.01= | 1.02= | · | · | 真输@win |
| TestBit i32 | 0.94L | 0.94L | 1.04W | 0.99= | 1.00= | 0.99= | 真输@win |
| TrailingZeroCount i32 | 1.05W | 1.03W | 0.93L | 0.94L | 1.10W | 1.07W | 真输@wsl |
| TruncateTo u32->u16 | 0.88L | 0.88L | 0.94L | 0.99= | 1.16W | 1.12W | 真输@win |
| ZeroIfNegative f32 | 0.95L | 0.96L | 1.10W | 1.12W | 1.15W | 1.13W | 真输@win |
| Abs i32 | 0.98L | 1.00= | 0.99= | 1.00= | 1.11W | 1.08W | 混合(单序/布局) |
| AbsDiff i32 | 0.99= | 0.98L | 0.99= | 1.00= | 1.10W | 1.10W | 混合(单序/布局) |
| AddSub i32 | 1.03W | 1.01= | 0.99= | 0.97L | 1.10W | 1.10W | 混合(单序/布局) |
| AllBits0 i32 | 0.96L | 0.98= | 1.02= | 1.00= | 1.00= | 1.04W | 混合(单序/布局) |
| AllFalse i32 | 1.03W | 1.07W | 1.00= | 1.00= | 1.00= | 0.97L | 混合(单序/布局) |
| And i32 | 1.03W | 0.99= | 0.98L | 1.21W | 1.08W | 1.10W | 混合(单序/布局) |
| BitCast f32->i32 | 1.02W | 1.00= | 0.99= | 0.98L | 1.02W | 1.00= | 混合(单序/布局) |
| BlendedStore i32 (4096, 50%) | 0.98= | 0.91L | 1.03W | 0.99= | · | · | 混合(单序/布局) |
| CompressNot i32 (4096, 50%) | 0.86L | 1.06W | 0.98= | 0.98= | · | · | 混合(单序/布局) |
| ConvertTo i32->f32 | 1.01= | 0.97L | 1.00= | 1.00= | 1.01= | 0.97L | 混合(单序/布局) |
| CountTrue i32 | 0.99= | 0.99= | 0.97L | 0.99= | 1.02= | 0.98L | 混合(单序/布局) |
| Dup128MaskFromMaskBits i32 (4096) | 0.96L | 1.02= | 1.19W | 1.20W | · | · | 混合(单序/布局) |
| DupOdd i32 | 0.97L | 0.98= | 1.00= | 1.02W | 1.12W | 1.12W | 混合(单序/布局) |
| ExclusiveNeither i32 | 1.18W | 1.16W | 0.96L | 1.03W | 1.01= | 0.99= | 混合(单序/布局) |
| FindFirstTrue i32 | 1.04W | 1.03W | 0.99= | 0.99= | 0.87L | 1.24W | 混合(单序/布局) |
| Floor f32 | 1.01= | 1.01= | 1.02W | 0.98L | 1.13W | 1.11W | 混合(单序/布局) |
| FloorInt f32->i32 | 1.51W | 1.73W | 1.45W | 0.90L | 1.26W | 1.28W | 混合(单序/布局) |
| GatherOffset i32 (4096) | 1.01= | 0.97L | 1.02= | 1.01= | · | · | 混合(单序/布局) |
| InterleaveOdd i32 | 1.02W | 0.99= | 1.04W | 0.97L | 1.11W | 1.08W | 混合(单序/布局) |
| InterleaveWholeUpper i32 | 1.00= | 0.97L | 1.01= | 0.97L | 1.10W | 1.09W | 混合(单序/布局) |
| IsFinite f32 | 0.99= | 0.97L | 1.35W | 1.40W | 1.00= | 0.99= | 混合(单序/布局) |
| Load/Store i32 aligned | 1.01= | 1.00= | 1.02W | 1.02W | 1.07W | 0.87L | 混合(单序/布局) |
| LoadU/StoreU i32 | 1.01= | 0.98L | 1.01= | 0.99= | 1.13W | 1.14W | 混合(单序/布局) |
| Lt i32 | 1.03W | 1.00= | 0.98L | 1.01= | 1.10W | 1.14W | 混合(单序/布局) |
| MaskedLoadOr i32 (4096) | 1.02= | 0.99= | 0.97L | 1.02= | · | · | 混合(单序/布局) |
| MaskedMulComplex f32 | 0.98= | 1.00= | 1.03W | 1.02W | 0.97L | 1.01= | 混合(单序/布局) |
| MaskedMulComplexConj f32 | 0.97L | 0.99= | 1.00= | 1.04W | 1.01= | 1.07W | 混合(单序/布局) |
| MaskedMulComplexOr f32 | 1.00= | 1.01= | 1.03W | 1.02= | 1.00= | 0.97L | 混合(单序/布局) |
| MaskedShl i32 | 0.99= | 0.97L | 1.01= | 1.00= | 1.10W | 1.10W | 混合(单序/布局) |
| Max f32 | 1.02= | 1.01= | 1.00= | 0.97L | 1.12W | 1.08W | 混合(单序/布局) |
| Max128 u64 | 0.97L | 0.98= | 0.99= | 1.01= | 1.11W | 1.10W | 混合(单序/布局) |
| Min f32 | 0.98L | 1.02W | 0.99= | 1.01= | 1.12W | 1.08W | 混合(单序/布局) |
| MinOfLanes i32 | 0.98L | 1.01= | 1.35W | 1.08W | 1.12W | 1.10W | 混合(单序/布局) |
| Mul f32 | 1.00= | 0.97L | 1.01= | 1.00= | 1.13W | 1.08W | 混合(单序/布局) |
| MulHigh i16 (4096) | 1.02= | 0.98L | 0.96L | 0.99= | · | · | 混合(单序/布局) |
| MulHigh u16 (4096) | 0.98L | 1.02W | 1.00= | 0.97L | · | · | 混合(单序/布局) |
| NegMulAdd f32 | 0.99= | 1.01= | 0.98L | 1.00= | 1.11W | 1.14W | 混合(单序/布局) |
| OddEvenBlocks i32 | 1.01= | 1.02W | 0.98L | 1.00= | · | · | 混合(单序/布局) |
| OrderedDemote2MasksTo i32->i16 (4096) | 1.05W | 1.06W | 0.97L | 1.00= | · | · | 混合(单序/布局) |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 0.97L | 0.98= | 0.94L | 1.00= | · | · | 混合(单序/布局) |
| PairwiseAdd128 i32 | 1.00= | 0.99= | 0.98L | 1.01= | 1.07W | 1.05W | 混合(单序/布局) |
| ReduceMax i32 | 1.09W | 1.12W | 1.02W | 1.02= | 1.00= | 0.95L | 混合(单序/布局) |
| Reverse8 i32 (4096) | 1.05W | 0.88L | 1.00= | 0.99= | · | · | 混合(单序/布局) |
| ReverseBlocks i32 | 1.04W | 1.09W | 0.98L | 1.00= | · | · | 混合(单序/布局) |
| RotateLeftSame u32 bits=5 | 0.99= | 1.00= | 0.96L | 1.01= | 1.10W | 1.11W | 混合(单序/布局) |
| RotateRight<3> i32 | 1.08W | 1.06W | 0.97L | 1.00= | 1.12W | 1.07W | 混合(单序/布局) |
| SatWidenMulPairwiseAdd u8*i8->i16 | 1.03W | 0.99= | 1.01= | 0.96L | · | · | 混合(单序/布局) |
| SaturatedAbs i32 | 1.00= | 0.96L | 0.99= | 0.99= | 1.12W | 1.10W | 混合(单序/布局) |
| SaturatedSub i16 | 1.03W | 0.97L | 1.00= | 1.01= | 1.14W | 1.11W | 混合(单序/布局) |
| ScatterIndex i32 (4096) | 1.01= | 0.99= | 0.98L | 0.98= | · | · | 混合(单序/布局) |
| ScatterIndexN i32 (half n) | 0.88L | 1.14W | 1.05W | 1.07W | · | · | 混合(单序/布局) |
| SetBeforeFirst i32 | 0.98= | 0.97L | 1.24W | 1.24W | 1.00= | 0.99= | 混合(单序/布局) |
| ShiftLeftLanes<1> i32 | 1.02W | 1.01= | 0.98L | 1.00= | 1.12W | 1.09W | 混合(单序/布局) |
| ShiftLeftSame i32 bits=3 | 0.98L | 1.01= | 0.99= | 1.01= | 1.10W | 1.17W | 混合(单序/布局) |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.96L | 0.99= | 1.00= | 1.00= | · | · | 混合(单序/布局) |
| Shl i32 | 1.02W | 0.98L | 1.02= | 0.99= | 1.08W | 1.08W | 混合(单序/布局) |
| Shuffle1032 i32 | 1.01= | 0.98L | 1.02= | 1.00= | 1.13W | 1.12W | 混合(单序/布局) |
| Shuffle2103 i32 | 1.00= | 0.97L | 1.00= | 1.01= | 1.13W | 1.12W | 混合(单序/布局) |
| Slide1Down i32 | 1.00= | 0.99= | 1.04W | 0.98L | 1.13W | 1.15W | 混合(单序/布局) |
| SlideUpLanes amt=2 i32 | 1.07W | 1.06W | 1.03W | 0.98L | 1.14W | 1.09W | 混合(单序/布局) |
| StoreInterleaved2 f32 (4096) | 1.04W | 1.00= | 0.97L | 1.03W | · | · | 混合(单序/布局) |
| StoreInterleaved3 f32 (4096) | 0.96L | 0.99= | 0.96L | 1.03W | · | · | 混合(单序/布局) |
| Sub i32 | 1.00= | 1.00= | 1.00= | 0.98L | 1.10W | 1.06W | 混合(单序/布局) |
| Trunc f32 | 1.02W | 0.98= | 0.98L | 1.01= | 1.10W | 1.12W | 混合(单序/布局) |
| WidenMulPairwiseAdd i16->i32 | 0.88L | 1.00= | 0.98L | 1.01= | · | · | 混合(单序/布局) |
| Xor i32 | 0.99= | 1.01= | 0.98L | 0.98= | 1.10W | 1.06W | 混合(单序/布局) |
| Zero i32 | 1.00= | 1.01= | 0.98= | 0.93L | 1.00= | 1.01= | 混合(单序/布局) |
| ZeroExtendVector i16 half->full | 1.08W | 1.06W | 0.97L | 1.01= | · | · | 混合(单序/布局) |
| Add f32 | 1.01= | 1.00= | 1.02= | 0.99= | 1.13W | 1.13W | 平/赢 |
| AllBits1 i32 | 1.05W | 1.05W | 1.02= | 1.00= | 1.00= | 0.98= | 平/赢 |
| AllTrue i32 | 1.04W | 1.03W | 1.00= | 1.00= | 1.01= | 1.04W | 平/赢 |
| AndNot i32 | 1.02= | 1.00= | 1.00= | 1.00= | 1.09W | 1.12W | 平/赢 |
| ApproximateReciprocal f32 | 0.99= | 1.00= | 0.99= | 0.99= | 1.16W | 1.11W | 平/赢 |
| ApproximateReciprocalSqrt f32 | 1.02= | 1.01= | 1.02= | 0.99= | 1.16W | 1.16W | 平/赢 |
| AverageRound u16 | 1.03W | 1.01= | 0.99= | 0.99= | 1.15W | 1.18W | 平/赢 |
| BitCast i32->f32 | 1.05W | 1.05W | 0.99= | 1.00= | 1.00= | 1.02W | 平/赢 |
| Broadcast<0> i32 | 1.04W | 1.03W | 1.17W | 1.08W | 1.14W | 1.19W | HYLE赢(所有序) |
| BroadcastBlock<1> i32 | 1.03W | 1.09W | 1.02W | 0.99= | · | · | 平/赢 |
| BroadcastSignBit i32 | 0.99= | 1.01= | 1.02= | 0.98= | 1.11W | 1.10W | 平/赢 |
| Ceil f32 | 0.99= | 0.99= | 1.02= | 0.98= | 1.14W | 1.09W | 平/赢 |
| Clamp i32 | 1.01= | 1.01= | 0.99= | 1.04W | 1.07W | 1.03W | 平/赢 |
| CombineShiftRightBytes4 i32 | 1.01= | 1.00= | 1.00= | 1.02W | 1.11W | 1.15W | 平/赢 |
| CombineShiftRightLanes1 i32 | 1.03W | 1.01= | 0.99= | 0.99= | 1.09W | 1.06W | 平/赢 |
| ComplexConj f32 | 0.99= | 1.01= | 1.00= | 1.00= | 1.15W | 1.17W | 平/赢 |
| CompressBlendedStore i32 (4096, 50%) | 1.09W | 1.23W | 1.01= | 1.07W | · | · | 平/赢 |
| ConcatLowerUpper i32 | 1.01= | 1.00= | 1.01= | 1.00= | · | · | 平/赢 |
| ConcatOdd i32 | 1.00= | 1.00= | 0.99= | 1.00= | 1.11W | 1.10W | 平/赢 |
| ConcatUpperLower i32 | 1.01= | 1.03W | 1.02= | 1.01= | · | · | 平/赢 |
| ConcatUpperUpper i32 | 1.01= | 1.01= | 1.02= | 1.00= | · | · | 平/赢 |
| ConvertTo f32->i32 | 1.27W | 1.32W | 1.19W | 1.26W | 1.03W | 0.99= | 平/赢 |
| CopySign f32 | 1.04W | 1.03W | 1.01= | 1.03W | 1.15W | 1.13W | 平/赢 |
| DemoteTo i32->i16 (sat) | 1.61W | 1.55W | 1.18W | 1.16W | · | · | HYLE赢(所有序) |
| DemoteTo i32->i16 1-arg (4096) | 0.98= | 1.00= | 1.00= | 0.98= | · | · | 平/赢 |
| Div f32 | 1.00= | 0.99= | 1.03W | 0.99= | 1.12W | 1.14W | 平/赢 |
| DupEven i32 | 0.99= | 0.98= | 1.00= | 0.99= | 1.12W | 1.11W | 平/赢 |
| Eq i32 | 1.01= | 1.01= | 1.03W | 1.02W | 1.10W | 1.08W | 平/赢 |
| GatherIndexN i32 (half n) | 1.00= | 0.98= | 0.99= | 1.00= | · | · | 平/赢 |
| GatherIndexNOr i32 (half n) | 0.99= | 0.99= | 1.00= | 1.02= | · | · | 平/赢 |
| Ge i32 | 0.99= | 0.99= | 1.01= | 1.01= | 1.07W | 1.06W | 平/赢 |
| GetBiasedExponent f32 | 1.01= | 1.00= | 1.16W | 1.17W | 1.00= | 0.98= | 平/赢 |
| GetExponent f32 | 1.01= | 1.04W | 1.00= | 1.02W | · | · | 平/赢 |
| Gt i32 | 1.00= | 1.00= | 0.99= | 1.01= | 1.12W | 1.06W | 平/赢 |
| IfNegativeThenElse f32 | 1.00= | 0.98= | 1.03W | 1.01= | 1.11W | 1.10W | 平/赢 |
| IfNegativeThenElseZero f32 | 1.05W | 1.03W | 1.08W | 1.07W | 1.09W | 1.08W | HYLE赢(所有序) |
| IfNegativeThenZeroElse f32 | 1.05W | 1.04W | 1.11W | 1.10W | 1.12W | 1.11W | HYLE赢(所有序) |
| IfThenElse i32 | 1.00= | 1.01= | 1.00= | 0.99= | 1.10W | 1.12W | 平/赢 |
| IfThenElseZero i32 | 1.09W | 1.08W | 1.02W | 1.00= | 1.15W | 1.14W | 平/赢 |
| IfThenZeroElse i32 | 1.08W | 1.08W | 1.01= | 1.01= | 1.15W | 1.14W | 平/赢 |
| InterleaveEven i32 | 0.99= | 1.02= | 0.99= | 1.04W | 1.11W | 1.05W | 平/赢 |
| InterleaveUpper i32 | 1.00= | 0.99= | 1.02= | 0.99= | 1.10W | 1.12W | 平/赢 |
| InterleaveWholeLower i32 | 0.99= | 0.99= | 0.98= | 1.00= | 1.10W | 1.06W | 平/赢 |
| IsEitherNaN f32 | 0.99= | 1.00= | 1.03W | 0.98= | 1.00= | 1.00= | 平/赢 |
| IsInf f32 | 1.10W | 1.15W | 1.17W | 1.18W | 1.00= | 1.01= | 平/赢 |
| Le i32 | 1.02W | 1.02= | 0.99= | 1.00= | 1.09W | 1.11W | 平/赢 |
| LoadExpand i32 (4096, 50%) | 2.19W | 2.14W | 2.16W | 2.18W | · | · | HYLE赢(所有序) |
| LoadInterleaved2 f32 (4096) | 1.00= | 1.01= | 1.04W | 0.98= | · | · | 平/赢 |
| Mask And i32 | 1.02W | 1.05W | 1.00= | 1.00= | 1.00= | 1.02W | 平/赢 |
| Mask AndNot i32 | 1.00= | 1.02W | 1.00= | 1.00= | 1.02W | 1.01= | 平/赢 |
| Mask Not i32 | 1.03W | 0.99= | 1.00= | 0.99= | 0.99= | 0.98= | 平/赢 |
| Mask Or i32 | 1.04W | 1.04W | 1.00= | 0.99= | 1.03W | 1.06W | 平/赢 |
| Mask Xor i32 | 1.01= | 0.99= | 0.99= | 1.01= | 1.01= | 0.98= | 平/赢 |
| MaskedApproxReciprocal f32 | 1.01= | 1.03W | 0.98= | 1.01= | 1.12W | 1.02= | 平/赢 |
| MaskedConvertTo f32-i32 | 1.98W | 1.97W | 1.38W | 1.22W | 1.05W | 1.05W | HYLE赢(所有序) |
| MaskedLeadingZeroCount i32 | 1.06W | 0.99= | 1.11W | 1.07W | 1.18W | 1.15W | 平/赢 |
| MaskedLoad i32 (4096, 50%) | 1.00= | 1.15W | 0.99= | 1.01= | · | · | 平/赢 |
| MaskedModOr i32 invariant | 0.99= | 1.00= | 0.98= | 0.99= | 1.07W | 1.06W | 平/赢 |
| MaskedMulComplexAdd f32 | 1.03W | 1.01= | 0.98= | 0.99= | 1.03W | 1.01= | 平/赢 |
| MaskedMulFixedPoint15 i16 | 0.99= | 1.01= | 0.99= | 1.04W | 1.12W | 1.11W | 平/赢 |
| MaskedReduceMax i32 | 1.01= | 1.00= | 1.01= | 0.99= | 1.00= | 1.00= | 平/赢 |
| MaskedReduceMin i32 | 1.01= | 1.02= | 0.99= | 0.99= | 1.00= | 1.00= | 平/赢 |
| MaskedReduceSum i32 | 0.99= | 1.00= | 1.00= | 0.99= | 0.99= | 1.01= | 平/赢 |
| MaskedSatAddOr i16 | 1.00= | 0.99= | 1.00= | 0.99= | 1.08W | 1.07W | 平/赢 |
| MaskedSatSubOr i16 | 1.02= | 1.00= | 1.02= | 1.01= | 1.11W | 1.07W | 平/赢 |
| MaskedShiftLeft<3> i32 | 1.05W | 1.08W | 1.00= | 1.01= | 1.01= | 0.98= | 平/赢 |
| MaskedShlOr i32 | 0.99= | 1.01= | 1.05W | 0.98= | 1.10W | 1.07W | 平/赢 |
| MaskedShr i32 | 0.98= | 1.00= | 1.01= | 1.01= | 1.07W | 1.08W | 平/赢 |
| Max i32 | 0.99= | 1.00= | 1.00= | 1.03W | 1.08W | 1.06W | 平/赢 |
| MaxMagnitude f32 | 1.03W | 1.03W | 1.01= | 0.99= | 1.10W | 1.16W | 平/赢 |
| MaxNumber f32 | 1.00= | 1.00= | 1.00= | 1.01= | 1.13W | 1.12W | 平/赢 |
| MaxOfLanes i32 | 0.99= | 1.00= | 1.33W | 1.09W | 1.14W | 1.12W | 平/赢 |
| Min128 u64 | 0.99= | 1.02= | 1.03W | 1.00= | 1.14W | 1.03W | 平/赢 |
| MinMagnitude f32 | 1.03W | 0.99= | 1.00= | 1.01= | 1.13W | 1.10W | 平/赢 |
| MinNumber f32 | 1.01= | 1.00= | 0.99= | 1.01= | 1.11W | 1.13W | 平/赢 |
| Mod i32 invariant | 1.00= | 1.02= | 1.00= | 0.99= | 1.07W | 1.07W | 平/赢 |
| Mul i32 | 1.00= | 1.01= | 1.03W | 1.01= | 1.11W | 1.16W | 平/赢 |
| MulAddSub f32 | 1.01= | 1.03W | 0.99= | 1.00= | 1.13W | 1.14W | 平/赢 |
| MulByPow2 f32 | 1.02= | 0.99= | 0.99= | 1.01= | 1.00= | 1.03W | 平/赢 |
| MulComplex f32 | 1.00= | 1.01= | 1.09W | 1.11W | 1.02= | 1.02W | 平/赢 |
| MulComplexConj f32 | 1.01= | 1.01= | 1.12W | 1.10W | 1.12W | 1.08W | 平/赢 |
| MulEven i32-i64 | 1.00= | 1.00= | 1.01= | 1.00= | 0.99= | 1.00= | 平/赢 |
| MulFixedPoint15 i16 | 1.04W | 1.01= | 1.02= | 1.00= | · | · | 平/赢 |
| MulHigh32 u32 (4096) | 0.98= | 1.01= | 0.98= | 1.01= | · | · | 平/赢 |
| MulOdd i32-i64 | 1.01= | 0.98= | 0.99= | 1.00= | 1.01= | 1.01= | 平/赢 |
| MulRound f32 | 1.02= | 1.01= | 1.03W | 1.01= | 1.13W | 1.11W | 平/赢 |
| MulSub f32 | 0.99= | 1.01= | 1.01= | 1.02W | 1.12W | 1.17W | 平/赢 |
| MulSubAdd f32 | 1.03W | 1.06W | 1.03W | 1.03W | 1.11W | 1.10W | HYLE赢(所有序) |
| Ne i32 | 0.99= | 1.00= | 1.00= | 1.00= | 1.10W | 1.11W | 平/赢 |
| NearestInt f32->i32 | 1.27W | 1.27W | 1.20W | 1.19W | 1.00= | 0.99= | 平/赢 |
| NegMulSub f32 | 0.98= | 0.98= | 0.99= | 1.00= | 1.12W | 1.13W | 平/赢 |
| Not i32 | 1.04W | 1.03W | 1.00= | 1.03W | 1.12W | 1.12W | 平/赢 |
| OddEven i32 | 0.99= | 0.99= | 0.99= | 1.00= | 1.11W | 1.13W | 平/赢 |
| Or i32 | 1.05W | 1.04W | 0.99= | 1.01= | 1.10W | 1.09W | 平/赢 |
| Or3 i32 | 1.02= | 1.04W | 1.01= | 0.98= | 1.08W | 1.07W | 平/赢 |
| OrAnd i32 | 0.98= | 0.99= | 1.00= | 0.98= | 1.00= | 1.01= | 平/赢 |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.19W | 1.15W | 1.14W | 1.16W | · | · | HYLE赢(所有序) |
| PairwiseAdd i32 | 1.11W | 1.12W | 1.10W | 1.06W | 1.11W | 1.12W | HYLE赢(所有序) |
| PairwiseSub i32 | 1.12W | 1.13W | 1.08W | 1.15W | 1.11W | 1.08W | HYLE赢(所有序) |
| PairwiseSub128 i32 | 1.03W | 1.02= | 1.05W | 1.06W | 1.11W | 1.09W | 平/赢 |
| PromoteMaskTo i16->i32 (4096) | 1.00= | 1.00= | 0.99= | 1.02W | · | · | 平/赢 |
| PromoteOddTo i8->i16 | 1.00= | 0.99= | 0.99= | 1.00= | · | · | 平/赢 |
| PromoteTo i16->i32 1-arg (4096) | 1.03W | 1.03W | 1.02W | 1.00= | · | · | 平/赢 |
| PromoteUpperTo i16->i32 | 1.01= | 1.03W | 0.99= | 1.00= | · | · | 平/赢 |
| RebindMask i32->f32 (4096) | 1.10W | 1.09W | 1.00= | 1.00= | · | · | 平/赢 |
| ReduceMin i32 | 1.10W | 1.11W | 1.03W | 1.02= | 1.00= | 0.99= | 平/赢 |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.10W | 1.10W | 1.19W | 1.23W | · | · | HYLE赢(所有序) |
| ReorderWidenMulAccumulate i16->i32 | 0.98= | 1.02= | 1.02= | 0.99= | · | · | 平/赢 |
| Reverse2 i32 | 1.00= | 0.99= | 1.00= | 1.00= | 1.12W | 1.12W | 平/赢 |
| Reverse4 i32 | 0.99= | 1.00= | 1.03W | 1.00= | 1.12W | 1.11W | 平/赢 |
| ReverseBits i32 | 1.72W | 1.77W | 1.57W | 1.33W | 1.15W | 1.17W | HYLE赢(所有序) |
| RotateLeft<3> i32 | 1.10W | 1.09W | 1.00= | 0.98= | 1.12W | 1.10W | 平/赢 |
| RotateRightSame u32 bits=5 | 1.01= | 1.02W | 1.00= | 1.00= | 1.09W | 1.14W | 平/赢 |
| Round f32 | 1.03W | 1.00= | 1.00= | 1.01= | 1.15W | 1.05W | 平/赢 |
| RoundToInt f32->i32 | 1.27W | 1.32W | 1.20W | 1.86W | 1.26W | 1.19W | HYLE赢(所有序) |
| RoundingShr i32 | 1.05W | 1.05W | 0.99= | 1.01= | 1.10W | 1.12W | 平/赢 |
| SatWidenMulAccumFixedPoint i16->i32 | 1.64W | 1.68W | 2.16W | 2.15W | · | · | HYLE赢(所有序) |
| SaturatedNeg i32 | 1.06W | 1.03W | 0.99= | 1.00= | 1.14W | 1.13W | 平/赢 |
| ScatterOffset i32 (4096) | 1.23W | 1.06W | 0.98= | 1.03W | · | · | 平/赢 |
| Set i32 | 0.98= | 0.99= | 1.00= | 1.03W | 0.99= | 1.01= | 平/赢 |
| SetAtOrBeforeFirst i32 | 1.10W | 1.10W | 1.20W | 1.16W | 0.99= | 1.00= | 平/赢 |
| SetMask i32 (4096) | 0.99= | 0.99= | 1.02= | 1.00= | · | · | 平/赢 |
| ShiftLeftBytes<4> i32 | 0.99= | 1.00= | 0.99= | 1.01= | 1.16W | 1.13W | 平/赢 |
| ShiftRight<3> i32 | 1.01= | 1.00= | 1.01= | 0.99= | 1.15W | 1.10W | 平/赢 |
| ShiftRightBytes<4> i32 | 0.99= | 0.99= | 1.09W | 1.00= | 1.11W | 1.10W | 平/赢 |
| ShiftRightLanes<1> i32 | 1.00= | 1.00= | 1.00= | 1.00= | 1.15W | 1.13W | 平/赢 |
| ShiftRightSame i32 bits=3 | 1.01= | 1.00= | 1.00= | 0.98= | 1.10W | 1.08W | 平/赢 |
| Shr u32 (variable) | 1.00= | 1.01= | 1.00= | 0.99= | 1.14W | 1.13W | 平/赢 |
| Shuffle0123 i32 | 1.00= | 1.01= | 0.99= | 1.00= | 1.15W | 1.09W | 平/赢 |
| Shuffle0321 i32 | 1.00= | 1.00= | 1.00= | 0.99= | 1.13W | 1.11W | 平/赢 |
| Shuffle2301 i32 | 1.01= | 1.00= | 1.00= | 1.00= | 1.13W | 1.12W | 平/赢 |
| Slide1Up i32 | 1.06W | 1.08W | 1.03W | 0.99= | 1.11W | 1.15W | 平/赢 |
| SlideDownLanes amt=2 i32 | 1.07W | 1.07W | 1.02= | 1.01= | 1.12W | 1.14W | 平/赢 |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.08W | 1.05W | 1.02= | 0.99= | · | · | 平/赢 |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.05W | 1.00= | 1.02W | 1.02= | · | · | 平/赢 |
| Sqrt f32 | 1.00= | 1.02= | 1.00= | 1.02= | 1.00= | 1.03W | 平/赢 |
| Sub f32 | 1.01= | 1.00= | 0.98= | 1.04W | 1.12W | 1.11W | 平/赢 |
| SumOfLanes i32 | 0.98= | 0.99= | 1.32W | 1.08W | 1.12W | 1.10W | 平/赢 |
| SumOfMulQuadAccumulate i8->i32 | 0.99= | 1.00= | 1.00= | 0.99= | · | · | 平/赢 |
| SumsOf2 u8->u16 | 1.14W | 1.06W | 0.99= | 1.05W | · | · | 平/赢 |
| SumsOf4 u8->u32 | 1.00= | 1.02W | 1.01= | 1.01= | · | · | 平/赢 |
| SumsOf8 u8->u64 | 1.02W | 1.01= | 1.02= | 1.00= | · | · | 平/赢 |
| SumsOf8AbsDiff u8->u64 | 1.01= | 1.00= | 1.00= | 0.99= | · | · | 平/赢 |
| SumsOfAdjQuadAbsDiff u8->u16 | 1.01= | 1.02= | 1.00= | 1.02= | · | · | 平/赢 |
| SwapAdjacentBlocks i32 | 1.06W | 1.09W | 1.00= | 0.99= | · | · | 平/赢 |
| TableLookupLanes i32 (4096) | 0.99= | 1.04W | 1.01= | 1.00= | · | · | 平/赢 |
| TwoTablesLookupLanes i32 (4096) | 1.03W | 1.03W | 1.00= | 1.00= | · | · | 平/赢 |
| WidenMulAccumulate i16->i32 | 0.99= | 1.00= | 1.02= | 0.98= | · | · | 平/赢 |
| Xor3 i32 | 1.59W | 1.58W | 1.01= | 0.99= | 1.95W | 1.88W | 平/赢 |
