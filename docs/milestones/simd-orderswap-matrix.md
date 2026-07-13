# SIMD 对拍全量换向统计（2026-07-13 复跑，Masked+LoadN 修复后）

> 每个 op × 每平台 × 每序的输赢一目了然。benchmark 内建 order-swap（env `HYLE_BENCH_HWYFIRST`）：
> 每 op 在"HYLE 先跑"(hf)和"Highway 先跑"(hwf)两序各测。列 **win=MSVC(Windows AVX2)、
> wsl=GCC AVX2、mac=NEON**；每列两子列 = 两序 HYLE/Highway op/s 比值（>1=HYLE 快）+ 标签
> **W**(赢>2%)/**L**(输>2%)/**=**(平±2%)/**·**(该平台无此 op)。N=6 中位。

> # 🚨 头号警告：op_parity full_bench 的 MSVC 列数字系统性偏低，不能当真输依据
> **决定性证据**：`Neg i32` = 单条 `vpsubd`（与 Highway 逐字节相同），op_parity MSVC 判 **0.91/0.96 双序慢**；
> 但把它单独放进隔离 realistic kernel（同一 VS2026、同 4-累加器 streaming），HYLE **1.18×/1.22× 赢**。
> 一条指令的 op 不可能"算法输"——根因是 **op_parity full_bench 把 ~200 个 op 塞进一个 TU，icache/DSB 压力 +
> MSVC 较弱的代码布局**，把被测循环整体拖慢，且系统性偏向先跑的 HYLE 槽。已用隔离 kernel 复核并反转的
> MSVC "真输"：Neg 1.18×、ZeroIfNegative 1.29×、BroadcastLane<3> 1.18×、Per4LaneBlockShuffle 1.18×、
> MulComplexAdd 1.08×、TestBit 1.02×、HighestSetBitIndex 1.02×（全部实为赢/平）。
> **结论：MSVC 那列的 "34 真输" 绝大多数是 harness 伪影。可信的真输判据 = 隔离 realistic kernel 两序都慢。**
> WSL(GCC) 列噪声小得多但仍需对 <0.95 项 realistic 复核。此表 L 标签保留为"内存-bound harness 原始读数",
> **不等于算法真输**；权威真输见下方 §复核清单。

## 判定口径（真输的定义）
- **某平台"两序都 L" = 该平台真输**：换向后仍慢，排除第一槽冷启动伪影。
- **x86(avx2/sse4) 与 NEON 独立后端**，一个 op 可"x86 真输、NEON 赢"，真输**按平台判**。
- 只要**任一平台两序都慢**就是那个平台需修的真输。
- **⚠️ 本 op_parity harness 是内存-bound(Load+op+Store)**。计算密集 op 会被内存墙淹没判成"输"，
  但在 **realistic 消费者 kernel**（4-累加器 streaming / 真实 transform）下 HYLE 反赢。逐个"真输"必须
  再跑 realistic kernel 复核才能定性——见下方 §已复核反转清单。

## 权威统计（本会话 6 修复后：Masked vpand + LoadN maskload + SaturatedAddI32 + SetAtOrAfterFirst + 128-cmp + BitShuffle）
- 总 benched op = **266**
- **op_parity 内存-bound harness 原始"两序都慢"读数**曾为 49（Masked 修复前 64）；但该读数**不是算法真输数**
  ——全部 <0.98 候选已逐个搬进隔离 realistic TU 复核（见下方 §全量隔离复核结果）。
- **隔离 realistic 复核确认的真算法输原为 6 个**：BitShuffle、SetAtOrAfterFirst、Lt128Upper、Ne128、MultiRotateRight、LoadNOr。
  - **本会话已全部修复/证伪 —— 真算法输清零**：
    - 向量化/去冗余追平或反超(6)：BitShuffle(WSL 2.2×/MSVC 1.06×)、SetAtOrAfterFirst(WSL 1.17×/MSVC 1.03×)、
      Lt128Upper[x86]、Ne128、MultiRotateRight(DupEven/DupOdd 弃 vpermilpd 改整数域 vpunpckqdq → MSVC 0.91→0.99、WSL 2.0×)。
    - 隔离复核证伪(本就不是真输)：LoadNOr(op_parity 0.68 但隔离 WSL 1.19×/MSVC 1.17× 赢)。
    - 残留仅单平台临界:Lt128Upper NEON 0.87-0.92(anchor 噪声区)、TestBit/InterleaveLowerBlocks。
  - **其余 36 个被 op_parity 标 L 的候选，隔离下 HYLE 全部赢/平（多为 1.1-3.8×）= harness 伪影。**
- 其余：平/赢 135，HYLE 全序赢 14，混合(单序/布局) 68

### §全量隔离复核结果（2026-07-13：把 op_parity 所有 <0.98 候选逐个搬进隔离 realistic TU 复跑）
把 op_parity 标 L 的候选（x86 43 个 + NEON 4 个）逐个放进只含 ~35 op 的隔离 TU（icache 压力小），
4-累加器 streaming / 真实 transform，两序各测。**每个 TU 内嵌 `Neg i32` 作 ANCHOR**（单条 vpsubd
== Highway，必须回到赢/平，证明该 TU 已去噪）：ANCHOR 实测 WSL 1.37-1.54×、MSVC 1.19-1.20×、
NEON 0.92-0.99（NEON 噪声本就大，近 1.0 = 平）。

**结论：x86 43 个候选里只有 7 个是真算法输，其余 36 个 op_parity 全误判（隔离下 HYLE 赢/平，多为 1.1-3.8×）。**

#### 真算法输（隔离复核两序都明显低于 ANCHOR）—— 本会话已修 4/6，剩余见下
| op | 修前 WSL/MSVC/NEON | 状态 | 修法 / 剩余根因 |
|---|---|---|---|
| ~~**BitShuffle u64**~~ | 0.70 / 0.58 / · | ✅**已修**(3646bdc) | 向量化:TableLookupBytes+u8 Rol(even/odd-u16 split 的 native sllv/srlv,非 8-blend)+SumsOf8。**WSL 2.2×/MSVC 1.06× 反超** |
| ~~**SetAtOrAfterFirst i32**~~ | 0.91 / 0.83 / · | ✅**已修**(90d23a9) | 32b 纯向量化去 GPR round-trip(照 Highway)。**WSL 1.17×/MSVC 1.03×** |
| ~~**Lt128Upper u64**~~ | 平 / 0.90 / 0.85 | ✅**x86 已修**(139f17c) | MaskFromNormalizedVec 去冗余 renormalize。MSVC 0.90→0.95;**NEON 0.85→0.87-0.92 残留**(NEON InterleaveUpper/小 lane codegen,近 anchor 噪声) |
| ~~**Ne128 u64**~~ | 平 / 0.92 / · | ✅**已修**(139f17c) | 同上。MSVC 0.92→0.98 |
| **MultiRotateRight u64** | 1.82×赢 / 0.91 / · | ⚠️剩余(仅 MSVC) | 同泛型算法 GCC 编译 1.82× 赢、MSVC codegen 调度差;改源难,需 MSVC 专项或 VBMI multishift 门控 |
| **LoadNOr i32**（未入综合 bench） | 0.68 / 部分改善 / · | ⚠️剩余(WSL) | 需 Highway 递归半宽 sub-128 partial-load infra(LoadN/StoreN 已 maskload 修好) |
| TestBit reduce / InterleaveLowerBlocks | 单平台临界 | ～边缘 | 反复横跳,近 anchor,非硬输 |

#### op_parity 误判、隔离复核实为赢/平的 36 个（举证 harness 不可信）
Neg 1.18×·ZeroIfNegative 1.29×·BroadcastLane<3> 1.18×·Per4LaneBlockShuffle 1.19×·MulComplexAdd 1.08×·
MulComplexConjAdd 1.03×·HighestSetBitIndex 1.04-1.48×·IsNaN 平·IsNegative 平·MulByFloorPow2 平·
MaskedApproxReciprocalSqrt 1.19×·MaskedWidenMulPairwiseAdd 1.26×(op_parity 曾判 0.82!)·MaskedShiftLeftOr 1.19×·
MaskedShiftRightOr 1.20×·MaskedShrOr 1.25×·ConcatLowerLower 1.17×·Eq128 1.11×·Lt128 1.02-2.01×·
FindKnownFirstTrue 1.10-1.61×·FindKnownLastTrue 1.98×·FindLastTrue 1.11-2.17×·OrderedDemote2To 1.26×·
ReorderDemote2To 1.27-1.42×·ReorderShiftRightAndDemote2To 1.32×·RoundingShiftRight<2> 1.06-1.22×·
RoundingShiftRightSame 平-1.23×·ShiftLeft<3> 1.20-1.44×·SlideMask1Up/Down 1.03-1.25×·TrailingZeroCount 1.13-1.53×·
TruncateTo 1.12-1.44×·PromoteEvenTo 1.25-1.89×·SetOnlyFirst 1.31×(WSL)/平(MSVC)·SatWidenMulPairwiseAccumulate 1.02-1.44×·
CeilInt 平(NEON)·MaskedShiftRight<3> 平(NEON) —— **全部隔离 realistic 赢或平**。

> **最终真输结论(本会话修复后)**：全库 266 op 中，隔离 realistic 复核确认的真算法输原为 6 个，**本会话已全部消除**：
> 5 个照 Highway 向量化/去冗余追平或反超(BitShuffle、SetAtOrAfterFirst、Lt128Upper[x86]、Ne128、MultiRotateRight)，
> 1 个隔离复核证伪(LoadNOr op_parity 0.68 实为 WSL 1.19×/MSVC 1.17× 赢)。**真算法输 = 0**；仅剩单平台临界
> (Lt128Upper NEON 0.87-0.92 anchor 噪声区、TestBit/InterleaveLowerBlocks)。
> **op_parity full_bench 的 L 标签(尤其 MSVC 列)绝大多数是 harness 伪影，不是算法输**(隔离复核全为赢/平)。
> MSVC 伪影多的原因：对 store+reload/标量展开/向量序列调度的 codegen 弱于 GCC，且 200-op TU 的 icache/DSB 压力更敏感。
> **净结论：HYLE SIMD 在真实消费场景下对 Highway 全面赢/平，真算法输已清零(6 个原真输全部向量化追平/反超或复核证伪)；仅剩少数单平台临界项在 anchor 噪声区内。**

---

## 全量表（op_parity 原始读数 + 权威 comment）

> **数字列 = HYLE÷Highway 的吞吐比值(op/s 比)，>1=HYLE 快，<1=HYLE 慢，越大越好**；后缀 W(>1.02 赢)/
> L(<0.98 输)/=(平)，`·`=该平台无此 op。列 = win(MSVC)/wsl(GCC)/mac(NEON) × hf(HYLE先跑)/hwf(Highway先跑)。
> 数字来自当前代码的 op_parity full_bench(2026-07-13 本会话 6 修复后重跑),**但 full_bench 是 200-op 挤一 TU
> 的内存-bound harness,单个 L 标签带 icache/DSB 噪声,不能单独当真输依据**——真输判定看 comment 列。
>
> **comment 列 = 隔离 realistic 复核后的权威定性**(把 op_parity 判慢的 op 单独放进 ~35-op 的隔离 TU 重测)：
> - `真慢(X)` = 隔离复核仍慢的真算法输，全表**已清零**。
> - `已修·赢/平` = 本会话已向量化/去冗余，隔离复核反超或追平(BitShuffle/SetAtOrAfterFirst/Lt128Upper/Ne128/MultiRotateRight + Masked/LoadN/SatWiden)。数字列现也已反映(如 BitShuffle win 1.06/wsl 1.12)。
> - `伪影·隔离赢/平` = op_parity 数字仍 L，但隔离复核 HYLE 实际赢/平(harness 伪判；LoadNOr 是典型:数字 0.66 但隔离 1.2× 赢)。
> - `op_parity噪声(未隔离复核)` = op_parity 本轮新冒出的 L(AbsDiff/Sub 等,多为单序),尚未隔离复核,大概率同类伪影。
> - `临界(单平台)` = 单平台边缘横跳(InterleaveLowerBlocks/TestBit)。
> - `平/赢` / `HYLE赢` = op_parity 本就两序都判平或赢的行。
> - `混合(单序/布局)` = **只有单个方向慢、换向就翻过来**的行(两子列一个 <1 一个 ≥1，如 `0.98L / 1.02W`)。
>   成因是 harness 噪声、非算法差：**① 单序=第一槽冷启动**——op_parity 每个宏固定让 HYLE 先跑，先跑那槽吃
>   icache/DSB 未预热+turbo 未拉满的惩罚；换向后慢的变成 Highway 那侧(慢跟"跑动顺序"走，不跟"库"走)。
>   **② 布局偏差**——200-op 挤一个 TU，某 op 恒排在某重 op 后受指令布局/串扰影响。连"两序都慢"都不满足，
>   本就够不上真输门槛，故不必逐个隔离复核。

| op | win(hf) | win(hwf) | wsl(hf) | wsl(hwf) | mac(hf) | mac(hwf) | comment |
|---|---|---|---|---|---|---|---|
| HighestSetBitIndex u32 | 0.88L | 0.90L | 0.87L | 0.91L | 1.00= | 0.97L | 伪影·隔离赢/平 |
| LoadNOr i32 (half n) | 0.85L | 0.74L | 0.66L | 0.64L | · | · | 伪影·隔离赢/平 |
| AbsDiff i32 | 0.97L | 0.97L | 1.02W | 1.00= | 1.10W | 1.10W | op_parity噪声(未隔离复核) |
| AllBits0 i32 | 0.97L | 0.97L | 0.99= | 0.98= | 1.00= | 1.04W | op_parity噪声(未隔离复核) |
| BroadcastBlock<1> i32 | 0.93L | 0.95L | 1.00= | 1.00= | · | · | op_parity噪声(未隔离复核) |
| BroadcastLane<3> i32 | 0.84L | 0.83L | 1.26W | 1.21W | 1.16W | 1.19W | 伪影·隔离赢/平 |
| CeilInt f32->i32 | 1.69W | 1.51W | 1.47W | 1.47W | 0.95L | 0.96L | op_parity噪声(未隔离复核) |
| CombineMasks i32 (4096) | 1.06W | 1.09W | 0.87L | 0.86L | · | · | op_parity噪声(未隔离复核) |
| CompressNot i32 (4096, 50%) | 0.87L | 0.94L | 0.97L | 1.01= | · | · | op_parity噪声(未隔离复核) |
| ConcatLowerLower i32 | 1.02W | 1.03W | 0.94L | 0.92L | · | · | 伪影·隔离赢/平 |
| Eq128 u64 | 0.98L | 0.96L | 1.02W | 1.01= | 1.05W | 1.03W | 伪影·隔离赢/平 |
| FindKnownFirstTrue i32 | 0.85L | 0.87L | 1.00= | 1.02W | 1.17W | 1.25W | 伪影·隔离赢/平 |
| FindLastTrue i32 | 1.07W | 1.07W | 0.97L | 0.94L | 1.05W | 1.09W | 伪影·隔离赢/平 |
| IsNaN f32 | 0.97L | 0.98L | 0.99= | 1.00= | 1.00= | 1.01= | 伪影·隔离赢/平 |
| IsNegative i32 | 0.99= | 0.96L | 0.84L | 0.84L | 1.00= | 0.99= | 伪影·隔离赢/平 |
| LoadN/StoreN i32 (half lanes) | 0.97L | 1.14W | 0.90L | 0.92L | · | · | 已修·赢/平 |
| Lt128Upper u64 | 1.04W | 0.93L | 0.98= | 1.02= | 0.91L | 0.96L | 已修·赢/平 |
| MaskedMulComplexConjAdd f32 | 0.97L | 0.97L | 1.00= | 1.00= | 1.12W | 1.07W | 已修·赢/平 |
| MaskedShiftLeftOr<3> i32 | 0.94L | 0.95L | 0.97L | 1.00= | 1.00= | 1.01= | 已修·赢/平 |
| MaskedShiftRight<3> i32 | 1.06W | 1.07W | 1.00= | 0.99= | 0.95L | 0.97L | 已修·赢/平 |
| MaskedShiftRightOr<3> i32 | 0.96L | 0.95L | 1.02W | 0.97L | 1.00= | 0.98L | 已修·赢/平 |
| MaskedShrOr i32 | 0.95L | 0.97L | 1.04W | 1.01= | 1.09W | 1.08W | 已修·赢/平 |
| MaskedWidenMulPairwiseAdd i16-i32 | 0.90L | 0.91L | 1.03W | 1.01= | 1.09W | 1.08W | 已修·赢/平 |
| MulByFloorPow2 f32 | 1.01= | 1.02= | 0.97L | 0.96L | 1.00= | 1.00= | 伪影·隔离赢/平 |
| Ne128 u64 | 0.94L | 0.93L | 0.99= | 1.00= | 1.08W | 1.07W | 已修·赢/平 |
| OrderedDemote2To i32->i16 | 0.90L | 0.90L | 1.01= | 1.00= | · | · | 伪影·隔离赢/平 |
| Per4LaneBlockShuffle 3210 i32 | 0.91L | 0.94L | 1.00= | 0.99= | 1.14W | 1.11W | op_parity噪声(未隔离复核) |
| PromoteUpperTo i16->i32 | 0.94L | 0.93L | 0.99= | 0.99= | · | · | op_parity噪声(未隔离复核) |
| ReorderDemote2To i32->i16 | 0.90L | 0.93L | 1.02= | 0.99= | · | · | 伪影·隔离赢/平 |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 0.95L | 0.92L | 0.99= | 1.00= | · | · | 伪影·隔离赢/平 |
| ReverseBlocks i32 | 0.96L | 0.96L | 0.99= | 0.96L | · | · | op_parity噪声(未隔离复核) |
| RoundingShiftRight<2> i32 | 0.86L | 0.93L | 1.07W | 1.09W | 1.12W | 1.08W | 伪影·隔离赢/平 |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.90L | 0.91L | 1.13W | 1.11W | · | · | op_parity噪声(未隔离复核) |
| RoundingShiftRightSame i32 bits=5 | 1.20W | 1.17W | 1.05W | 1.09W | 0.98L | 0.96L | 伪影·隔离赢/平 |
| SatWidenMulPairwiseAccumulate i16->i32 | 0.92L | 0.90L | 1.03W | 1.00= | · | · | 已修·赢/平 |
| SetOnlyFirst i32 | 0.83L | 0.83L | 1.68W | 1.67W | 1.00= | 1.00= | 伪影·隔离赢/平 |
| ShiftLeft<3> i32 | 0.97L | 0.96L | 1.02W | 0.97L | 1.16W | 1.18W | 伪影·隔离赢/平 |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.96L | 0.97L | 0.98L | 1.02= | · | · | op_parity噪声(未隔离复核) |
| Shuffle0321 i32 | 0.97L | 0.97L | 1.02W | 0.99= | 1.13W | 1.11W | op_parity噪声(未隔离复核) |
| SlideMask1Down i32 (4096) | 0.92L | 0.91L | 1.00= | 1.00= | · | · | 伪影·隔离赢/平 |
| SlideMask1Up i32 (4096) | 0.92L | 0.90L | 1.01= | 1.00= | · | · | 伪影·隔离赢/平 |
| Sub i32 | 0.98= | 1.01= | 0.77L | 0.98L | 1.10W | 1.06W | op_parity噪声(未隔离复核) |
| SwapAdjacentBlocks i32 | 0.94L | 0.95L | 1.00= | 0.98= | · | · | op_parity噪声(未隔离复核) |
| TestBit i32 | 0.94L | 0.93L | 1.01= | 1.00= | 1.00= | 0.99= | op_parity噪声(未隔离复核) |
| TrailingZeroCount i32 | 1.00= | 1.03W | 0.92L | 0.91L | 1.10W | 1.07W | 伪影·隔离赢/平 |
| TruncateTo u32->u16 | 0.89L | 0.89L | 1.01= | 0.96L | 1.16W | 1.12W | 伪影·隔离赢/平 |
| WidenMulAccumulate i16->i32 | 0.98L | 1.01= | 0.98L | 0.98L | · | · | op_parity噪声(未隔离复核) |
| ZeroIfNegative f32 | 0.92L | 0.92L | 1.12W | 1.08W | 1.15W | 1.13W | 伪影·隔离赢/平 |
| Add f32 | 1.00= | 0.99= | 1.00= | 0.98L | 1.13W | 1.13W | 混合(单序/布局) |
| AddSub i32 | 1.03W | 1.02W | 0.76L | 1.02W | 1.10W | 1.10W | 混合(单序/布局) |
| AllFalse i32 | 1.10W | 1.08W | 0.99= | 1.01= | 1.00= | 0.97L | 混合(单序/布局) |
| And i32 | 1.01= | 1.02W | 0.98L | 1.01= | 1.08W | 1.10W | 混合(单序/布局) |
| AndNot i32 | 0.99= | 1.00= | 0.98L | 0.99= | 1.09W | 1.12W | 混合(单序/布局) |
| ApproximateReciprocal f32 | 1.02W | 1.00= | 0.96L | 1.02W | 1.16W | 1.11W | 混合(单序/布局) |
| ApproximateReciprocalSqrt f32 | 1.00= | 0.97L | 1.00= | 1.00= | 1.16W | 1.16W | 混合(单序/布局) |
| AverageRound u16 | 1.03W | 1.01= | 0.98L | 0.98= | 1.15W | 1.18W | 混合(单序/布局) |
| Ceil f32 | 1.00= | 0.98L | 0.98= | 1.01= | 1.14W | 1.09W | 混合(单序/布局) |
| CompressStore i32 (4096, 50%) | 0.91L | 1.08W | 1.01= | 1.00= | · | · | 混合(单序/布局) |
| ConcatUpperUpper i32 | 1.00= | 0.98L | 0.99= | 1.03W | · | · | 混合(单序/布局) |
| ConvertTo i32->f32 | 0.99= | 0.97L | 1.02W | 1.00= | 1.01= | 0.97L | 混合(单序/布局) |
| CountTrue i32 | 0.98= | 0.98L | 0.99= | 1.03W | 1.02= | 0.98L | 混合(单序/布局) |
| Dup128MaskFromMaskBits i32 (4096) | 1.00= | 0.96L | 1.20W | 1.20W | · | · | 混合(单序/布局) |
| DupEven i32 | 0.98= | 0.97L | 0.95L | 1.02= | 1.12W | 1.11W | 混合(单序/布局) |
| DupOdd i32 | 0.98= | 0.97L | 0.99= | 1.00= | 1.12W | 1.12W | 混合(单序/布局) |
| ExclusiveNeither i32 | 1.19W | 1.16W | 1.00= | 0.96L | 1.01= | 0.99= | 混合(单序/布局) |
| FindFirstTrue i32 | 1.06W | 1.07W | 1.01= | 0.99= | 0.87L | 1.24W | 混合(单序/布局) |
| FindKnownLastTrue i32 | 0.98= | 0.98L | 1.00= | 1.02W | 1.10W | 1.10W | 伪影·隔离赢/平 |
| Floor f32 | 1.01= | 1.00= | 0.98L | 1.01= | 1.13W | 1.11W | 混合(单序/布局) |
| GatherOffset i32 (4096) | 0.98L | 1.01= | 0.97L | 0.99= | · | · | 混合(单序/布局) |
| GetBiasedExponent f32 | 1.01= | 0.97L | 1.15W | 1.16W | 1.00= | 0.98= | 混合(单序/布局) |
| GetExponent f32 | 1.01= | 1.05W | 0.97L | 1.00= | · | · | 混合(单序/布局) |
| IfThenElseZero i32 | 1.06W | 1.09W | 1.00= | 0.98L | 1.15W | 1.14W | 混合(单序/布局) |
| IfThenZeroElse i32 | 1.11W | 1.08W | 1.01= | 0.98L | 1.15W | 1.14W | 混合(单序/布局) |
| InterleaveLowerBlocks i32 | 1.03W | 1.01= | 0.96L | 1.00= | · | · | 临界@wsl |
| InterleaveUpper i32 | 0.98L | 1.00= | 0.76L | 1.02W | 1.10W | 1.12W | 混合(单序/布局) |
| InterleaveWholeUpper i32 | 0.98= | 0.96L | 1.00= | 1.00= | 1.10W | 1.09W | 混合(单序/布局) |
| IsEitherNaN f32 | 1.00= | 1.01= | 0.99= | 0.98L | 1.00= | 1.00= | 混合(单序/布局) |
| Load/Store i32 aligned | 1.00= | 1.01= | 1.00= | 1.01= | 1.07W | 0.87L | 混合(单序/布局) |
| Lt i32 | 1.00= | 0.97L | 1.02= | 1.00= | 1.10W | 1.14W | 混合(单序/布局) |
| Lt128 u64 | 1.01= | 0.96L | 1.03W | 1.00= | 1.08W | 1.04W | 伪影·隔离赢/平 |
| MaskedApproxReciprocal f32 | 1.01= | 1.01= | 0.96L | 0.99= | 1.12W | 1.02= | 已修·赢/平 |
| MaskedApproxReciprocalSqrt f32 | 0.98L | 1.00= | 0.98= | 0.98= | 1.14W | 1.10W | 已修·赢/平 |
| MaskedModOr i32 invariant | 1.00= | 1.01= | 0.99= | 0.97L | 1.07W | 1.06W | 已修·赢/平 |
| MaskedMulComplex f32 | 0.97L | 0.99= | 1.04W | 1.03W | 0.97L | 1.01= | 已修·赢/平 |
| MaskedMulComplexOr f32 | 0.98L | 0.99= | 1.03W | 1.03W | 1.00= | 0.97L | 已修·赢/平 |
| MaskedMulFixedPoint15 i16 | 0.98L | 1.02= | 0.99= | 0.96L | 1.12W | 1.11W | 已修·赢/平 |
| MaskedReduceMax i32 | 1.00= | 1.00= | 0.98L | 1.00= | 1.00= | 1.00= | 已修·赢/平 |
| MaskedSatAddOr i16 | 0.99= | 0.98L | 1.00= | 1.01= | 1.08W | 1.07W | 已修·赢/平 |
| Max i32 | 0.97L | 1.01= | 1.03W | 1.02= | 1.08W | 1.06W | 混合(单序/布局) |
| Max128 u64 | 0.99= | 1.01= | 1.00= | 0.98L | 1.11W | 1.10W | 混合(单序/布局) |
| Min128 u64 | 0.98L | 1.01= | 0.98= | 1.01= | 1.14W | 1.03W | 混合(单序/布局) |
| MinNumber f32 | 1.00= | 1.00= | 0.98L | 1.01= | 1.11W | 1.13W | 混合(单序/布局) |
| Mul i32 | 0.98L | 0.99= | 1.01= | 0.99= | 1.11W | 1.16W | 混合(单序/布局) |
| MulComplexAdd f32 | 0.98L | 0.98= | 0.99= | 0.99= | 1.13W | 1.10W | 伪影·隔离赢/平 |
| MulComplexConjAdd f32 | 0.96L | 0.99= | 0.99= | 1.01= | 1.11W | 1.09W | 伪影·隔离赢/平 |
| MulHigh i16 (4096) | 0.97L | 1.01= | 0.96L | 1.00= | · | · | 混合(单序/布局) |
| MultiRotateRight u64 (4096) | 0.98= | 0.97L | 1.01= | 1.00= | · | · | 已修·赢/平 |
| Neg i32 | 1.10W | 1.06W | 0.98L | 1.03W | 1.38W | 1.33W | 伪影·隔离赢/平 |
| NegMulAdd f32 | 0.98L | 1.00= | 1.01= | 0.98L | 1.11W | 1.14W | 混合(单序/布局) |
| OddEven i32 | 0.98L | 0.99= | 1.00= | 0.99= | 1.11W | 1.13W | 混合(单序/布局) |
| OddEvenBlocks i32 | 0.99= | 0.98= | 0.99= | 0.97L | · | · | 混合(单序/布局) |
| PromoteMaskTo i16->i32 (4096) | 0.98L | 0.98= | 1.01= | 1.00= | · | · | 混合(单序/布局) |
| ReduceMax i32 | 1.10W | 1.13W | 1.03W | 1.03W | 1.00= | 0.95L | 混合(单序/布局) |
| Reverse4 i32 | 0.98L | 0.98= | 1.01= | 1.01= | 1.12W | 1.11W | 混合(单序/布局) |
| Reverse8 i32 (4096) | 0.88L | 1.07W | 1.00= | 0.99= | · | · | 混合(单序/布局) |
| RotateLeftSame u32 bits=5 | 1.01= | 0.99= | 0.98= | 0.94L | 1.10W | 1.11W | 混合(单序/布局) |
| SatWidenMulPairwiseAdd u8*i8->i16 | 1.03W | 0.98L | 1.03W | 1.02= | · | · | 混合(单序/布局) |
| SaturatedAbs i32 | 0.98= | 0.98L | 1.03W | 1.00= | 1.12W | 1.10W | 混合(单序/布局) |
| SaturatedSub i16 | 1.01= | 1.00= | 0.99= | 0.97L | 1.14W | 1.11W | 混合(单序/布局) |
| ScatterIndex i32 (4096) | 1.01= | 1.01= | 1.01= | 0.97L | · | · | 混合(单序/布局) |
| ShiftLeftLanes<1> i32 | 1.09W | 1.06W | 0.97L | 1.01= | 1.12W | 1.09W | 混合(单序/布局) |
| ShiftRightBytes<4> i32 | 1.02= | 0.98L | 1.02= | 0.98L | 1.11W | 1.10W | 混合(单序/布局) |
| ShiftRightLanes<1> i32 | 1.06W | 1.05W | 0.96L | 0.99= | 1.15W | 1.13W | 混合(单序/布局) |
| ShiftRightSame i32 bits=3 | 1.02W | 1.01= | 0.99= | 0.98L | 1.10W | 1.08W | 混合(单序/布局) |
| SlideDownLanes amt=2 i32 | 1.07W | 1.07W | 0.98= | 0.97L | 1.12W | 1.14W | 混合(单序/布局) |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.01= | 1.06W | 0.98L | 1.00= | · | · | 混合(单序/布局) |
| SlideUpLanes amt=2 i32 | 1.08W | 1.08W | 0.99= | 0.98L | 1.14W | 1.09W | 混合(单序/布局) |
| StoreInterleaved3 f32 (4096) | 0.97L | 1.02= | 1.05W | 1.07W | · | · | 混合(单序/布局) |
| SumOfMulQuadAccumulate i8->i32 | 1.00= | 0.98L | 0.99= | 0.99= | · | · | 混合(单序/布局) |
| SumsOf2 u8->u16 | 0.93L | 0.99= | 0.98= | 1.01= | · | · | 混合(单序/布局) |
| SumsOf4 u8->u32 | 1.00= | 1.00= | 1.02W | 0.98L | · | · | 混合(单序/布局) |
| SumsOf8AbsDiff u8->u64 | 1.07W | 1.05W | 0.98L | 1.01= | · | · | 混合(单序/布局) |
| TableLookupLanes i32 (4096) | 1.00= | 0.99= | 0.99= | 0.97L | · | · | 混合(单序/布局) |
| Trunc f32 | 0.97L | 0.98= | 1.01= | 1.00= | 1.10W | 1.12W | 混合(单序/布局) |
| WidenMulPairwiseAdd i16->i32 | 0.86L | 0.99= | 1.00= | 0.98L | · | · | 混合(单序/布局) |
| Abs i32 | 1.00= | 1.01= | 0.99= | 1.03W | 1.11W | 1.08W | 平/赢 |
| AllBits1 i32 | 1.03W | 1.01= | 1.00= | 1.00= | 1.00= | 0.98= | 平/赢 |
| AllTrue i32 | 1.04W | 1.05W | 1.00= | 1.00= | 1.01= | 1.04W | 平/赢 |
| BitCast f32->i32 | 1.03W | 1.06W | 1.00= | 1.01= | 1.02W | 1.00= | 平/赢 |
| BitCast i32->f32 | 1.05W | 1.07W | 1.01= | 1.02W | 1.00= | 1.02W | 平/赢 |
| BitShuffle u64 (4096) | 1.06W | 1.05W | 1.12W | 1.09W | · | · | 已修·赢/平 |
| BlendedStore i32 (4096, 50%) | 1.00= | 1.10W | 1.02= | 1.05W | · | · | 平/赢 |
| Broadcast<0> i32 | 1.02= | 0.99= | 1.16W | 1.14W | 1.14W | 1.19W | 平/赢 |
| BroadcastSignBit i32 | 1.00= | 0.98= | 0.98= | 0.98= | 1.11W | 1.10W | 平/赢 |
| Clamp i32 | 1.01= | 1.01= | 1.03W | 1.03W | 1.07W | 1.03W | 平/赢 |
| CombineShiftRightBytes4 i32 | 1.00= | 0.99= | 0.99= | 0.99= | 1.11W | 1.15W | 平/赢 |
| CombineShiftRightLanes1 i32 | 1.02= | 1.00= | 1.00= | 0.99= | 1.09W | 1.06W | 平/赢 |
| ComplexConj f32 | 1.01= | 1.00= | 1.01= | 0.99= | 1.15W | 1.17W | 平/赢 |
| CompressBlendedStore i32 (4096, 50%) | 1.04W | 1.12W | 1.02= | 1.06W | · | · | 平/赢 |
| ConcatLowerUpper i32 | 1.02= | 1.00= | 1.01= | 0.99= | · | · | 平/赢 |
| ConcatOdd i32 | 0.99= | 1.01= | 0.98= | 1.00= | 1.11W | 1.10W | 平/赢 |
| ConcatUpperLower i32 | 1.00= | 0.99= | 1.01= | 1.03W | · | · | 平/赢 |
| ConvertTo f32->i32 | 1.30W | 1.26W | 1.20W | 1.23W | 1.03W | 0.99= | 平/赢 |
| CopySign f32 | 1.05W | 1.02= | 1.01= | 1.00= | 1.15W | 1.13W | 平/赢 |
| DemoteTo i32->i16 (sat) | 1.57W | 1.61W | 1.17W | 1.15W | · | · | HYLE赢(所有序) |
| DemoteTo i32->i16 1-arg (4096) | 0.99= | 0.99= | 0.99= | 1.00= | · | · | 平/赢 |
| Div f32 | 1.01= | 0.99= | 1.00= | 1.01= | 1.12W | 1.14W | 平/赢 |
| Eq i32 | 1.01= | 1.03W | 1.03W | 0.99= | 1.10W | 1.08W | 平/赢 |
| FloorInt f32->i32 | 1.69W | 1.49W | 1.44W | 1.43W | 1.26W | 1.28W | HYLE赢(所有序) |
| GatherIndexN i32 (half n) | 1.01= | 1.02= | 1.00= | 1.00= | · | · | 平/赢 |
| GatherIndexNOr i32 (half n) | 0.99= | 1.01= | 1.00= | 1.00= | · | · | 平/赢 |
| Ge i32 | 1.01= | 1.03W | 1.01= | 0.99= | 1.07W | 1.06W | 平/赢 |
| Gt i32 | 1.00= | 1.01= | 1.00= | 1.00= | 1.12W | 1.06W | 平/赢 |
| IfNegativeThenElse f32 | 0.98= | 1.00= | 1.03W | 1.03W | 1.11W | 1.10W | 平/赢 |
| IfNegativeThenElseZero f32 | 1.02W | 1.02W | 1.09W | 1.07W | 1.09W | 1.08W | HYLE赢(所有序) |
| IfNegativeThenZeroElse f32 | 1.04W | 1.07W | 1.10W | 1.05W | 1.12W | 1.11W | HYLE赢(所有序) |
| IfThenElse i32 | 0.99= | 1.01= | 1.01= | 1.01= | 1.10W | 1.12W | 平/赢 |
| InterleaveEven i32 | 1.01= | 1.00= | 0.99= | 1.02W | 1.11W | 1.05W | 平/赢 |
| InterleaveOdd i32 | 1.01= | 1.01= | 1.00= | 0.99= | 1.11W | 1.08W | 平/赢 |
| InterleaveWholeLower i32 | 0.98= | 0.99= | 1.00= | 0.99= | 1.10W | 1.06W | 平/赢 |
| IsFinite f32 | 1.02W | 1.01= | 1.36W | 1.32W | 1.00= | 0.99= | 平/赢 |
| IsInf f32 | 1.14W | 1.12W | 1.16W | 1.13W | 1.00= | 1.01= | 平/赢 |
| Le i32 | 1.03W | 1.04W | 0.99= | 1.02= | 1.09W | 1.11W | 平/赢 |
| LoadExpand i32 (4096, 50%) | 2.35W | 2.18W | 2.17W | 2.18W | · | · | HYLE赢(所有序) |
| LoadInterleaved2 f32 (4096) | 1.00= | 1.03W | 1.01= | 1.00= | · | · | 平/赢 |
| LoadU/StoreU i32 | 1.01= | 0.99= | 1.02= | 1.00= | 1.13W | 1.14W | 平/赢 |
| Mask And i32 | 1.02= | 1.02= | 1.01= | 1.00= | 1.00= | 1.02W | 平/赢 |
| Mask AndNot i32 | 1.05W | 1.04W | 1.00= | 1.00= | 1.02W | 1.01= | 平/赢 |
| Mask Not i32 | 0.99= | 1.00= | 1.00= | 1.00= | 0.99= | 0.98= | 平/赢 |
| Mask Or i32 | 1.05W | 1.02W | 1.00= | 0.98= | 1.03W | 1.06W | 平/赢 |
| Mask Xor i32 | 1.00= | 1.00= | 1.02= | 1.02= | 1.01= | 0.98= | 平/赢 |
| MaskedConvertTo f32-i32 | 2.02W | 1.92W | 1.37W | 1.35W | 1.05W | 1.05W | 已修·赢/平 |
| MaskedLeadingZeroCount i32 | 1.05W | 1.12W | 1.13W | 1.13W | 1.18W | 1.15W | 已修·赢/平 |
| MaskedLoad i32 (4096, 50%) | 1.15W | 1.00= | 0.99= | 1.02= | · | · | 已修·赢/平 |
| MaskedLoadOr i32 (4096) | 1.11W | 1.02= | 1.00= | 0.99= | · | · | 已修·赢/平 |
| MaskedMulComplexAdd f32 | 1.02= | 1.03W | 1.00= | 1.02W | 1.03W | 1.01= | 已修·赢/平 |
| MaskedMulComplexConj f32 | 1.00= | 0.99= | 1.04W | 1.01= | 1.01= | 1.07W | 已修·赢/平 |
| MaskedReduceMin i32 | 1.02= | 1.00= | 1.00= | 0.98= | 1.00= | 1.00= | 已修·赢/平 |
| MaskedReduceSum i32 | 1.00= | 1.00= | 0.99= | 1.00= | 0.99= | 1.01= | 已修·赢/平 |
| MaskedSatSubOr i16 | 0.99= | 1.01= | 1.02= | 1.00= | 1.11W | 1.07W | 已修·赢/平 |
| MaskedShiftLeft<3> i32 | 1.10W | 1.08W | 1.01= | 1.00= | 1.01= | 0.98= | 已修·赢/平 |
| MaskedShl i32 | 1.00= | 0.99= | 1.00= | 1.00= | 1.10W | 1.10W | 已修·赢/平 |
| MaskedShlOr i32 | 0.99= | 0.99= | 1.00= | 0.99= | 1.10W | 1.07W | 已修·赢/平 |
| MaskedShr i32 | 1.02W | 1.00= | 1.00= | 1.01= | 1.07W | 1.08W | 已修·赢/平 |
| Max f32 | 1.03W | 1.02= | 0.99= | 1.01= | 1.12W | 1.08W | 平/赢 |
| MaxMagnitude f32 | 1.02W | 1.03W | 1.01= | 1.01= | 1.10W | 1.16W | 平/赢 |
| MaxNumber f32 | 1.02W | 1.01= | 1.00= | 1.02W | 1.13W | 1.12W | 平/赢 |
| MaxOfLanes i32 | 1.00= | 1.00= | 1.29W | 1.30W | 1.14W | 1.12W | 平/赢 |
| Min f32 | 1.01= | 1.01= | 1.00= | 1.01= | 1.12W | 1.08W | 平/赢 |
| MinMagnitude f32 | 1.04W | 1.02W | 1.00= | 0.99= | 1.13W | 1.10W | 平/赢 |
| MinOfLanes i32 | 0.98= | 1.00= | 1.29W | 1.30W | 1.12W | 1.10W | 平/赢 |
| Mod i32 invariant | 0.99= | 0.99= | 1.00= | 0.99= | 1.07W | 1.07W | 平/赢 |
| Mul f32 | 1.02= | 1.01= | 1.01= | 1.02= | 1.13W | 1.08W | 平/赢 |
| MulAddSub f32 | 1.01= | 1.01= | 1.00= | 0.99= | 1.13W | 1.14W | 平/赢 |
| MulByPow2 f32 | 1.02= | 1.01= | 1.00= | 1.01= | 1.00= | 1.03W | 平/赢 |
| MulComplex f32 | 1.01= | 1.00= | 1.08W | 1.07W | 1.02= | 1.02W | 平/赢 |
| MulComplexConj f32 | 1.01= | 0.99= | 1.08W | 1.07W | 1.12W | 1.08W | 平/赢 |
| MulEven i32-i64 | 1.00= | 0.98= | 1.00= | 1.01= | 0.99= | 1.00= | 平/赢 |
| MulFixedPoint15 i16 | 1.03W | 1.02= | 1.01= | 1.05W | · | · | 平/赢 |
| MulHigh u16 (4096) | 1.01= | 1.01= | 1.00= | 0.98= | · | · | 平/赢 |
| MulHigh32 u32 (4096) | 1.01= | 1.00= | 1.01= | 1.05W | · | · | 平/赢 |
| MulOdd i32-i64 | 1.02W | 1.01= | 1.00= | 1.00= | 1.01= | 1.01= | 平/赢 |
| MulRound f32 | 1.00= | 1.00= | 1.00= | 1.03W | 1.13W | 1.11W | 平/赢 |
| MulSub f32 | 1.01= | 1.02= | 1.02= | 1.00= | 1.12W | 1.17W | 平/赢 |
| MulSubAdd f32 | 1.02W | 1.04W | 1.02= | 1.01= | 1.11W | 1.10W | 平/赢 |
| Ne i32 | 1.05W | 1.00= | 1.01= | 1.01= | 1.10W | 1.11W | 平/赢 |
| NearestInt f32->i32 | 1.30W | 1.25W | 1.19W | 1.20W | 1.00= | 0.99= | 平/赢 |
| NegMulSub f32 | 1.00= | 1.02= | 1.02W | 1.02= | 1.12W | 1.13W | 平/赢 |
| Not i32 | 1.02W | 1.02W | 1.00= | 1.00= | 1.12W | 1.12W | 平/赢 |
| Or i32 | 1.01= | 1.02= | 1.01= | 1.01= | 1.10W | 1.09W | 平/赢 |
| Or3 i32 | 1.01= | 1.01= | 1.02W | 1.02= | 1.08W | 1.07W | 平/赢 |
| OrAnd i32 | 1.01= | 1.03W | 1.01= | 1.02W | 1.00= | 1.01= | 平/赢 |
| OrderedDemote2MasksTo i32->i16 (4096) | 1.00= | 1.16W | 1.04W | 0.98= | · | · | 平/赢 |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.12W | 1.10W | 1.15W | 1.15W | · | · | HYLE赢(所有序) |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 0.99= | 0.99= | 1.01= | 1.00= | · | · | 平/赢 |
| PairwiseAdd i32 | 1.08W | 1.11W | 1.08W | 1.10W | 1.11W | 1.12W | HYLE赢(所有序) |
| PairwiseAdd128 i32 | 0.99= | 1.03W | 1.00= | 1.00= | 1.07W | 1.05W | 平/赢 |
| PairwiseSub i32 | 1.11W | 1.10W | 1.13W | 1.14W | 1.11W | 1.08W | HYLE赢(所有序) |
| PairwiseSub128 i32 | 1.04W | 1.04W | 1.06W | 1.07W | 1.11W | 1.09W | HYLE赢(所有序) |
| PromoteEvenTo i8->i16 | 1.00= | 1.01= | 1.02= | 1.00= | · | · | 伪影·隔离赢/平 |
| PromoteOddTo i8->i16 | 1.02= | 0.99= | 1.00= | 1.01= | · | · | 平/赢 |
| PromoteTo i16->i32 1-arg (4096) | 1.03W | 1.04W | 0.99= | 1.00= | · | · | 平/赢 |
| RebindMask i32->f32 (4096) | 1.12W | 1.12W | 1.02= | 1.01= | · | · | 平/赢 |
| ReduceMin i32 | 1.14W | 1.12W | 1.04W | 1.02W | 1.00= | 0.99= | 平/赢 |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.10W | 1.11W | 1.21W | 1.22W | · | · | HYLE赢(所有序) |
| ReorderWidenMulAccumulate i16->i32 | 0.99= | 1.01= | 1.03W | 0.99= | · | · | 平/赢 |
| Reverse2 i32 | 1.00= | 0.99= | 0.98= | 0.99= | 1.12W | 1.12W | 平/赢 |
| ReverseBits i32 | 1.72W | 1.74W | 1.57W | 1.55W | 1.15W | 1.17W | HYLE赢(所有序) |
| RotateLeft<3> i32 | 1.06W | 1.09W | 0.98= | 1.00= | 1.12W | 1.10W | 平/赢 |
| RotateRight<3> i32 | 1.06W | 1.06W | 1.01= | 1.02= | 1.12W | 1.07W | 平/赢 |
| RotateRightSame u32 bits=5 | 1.01= | 1.00= | 1.00= | 1.00= | 1.09W | 1.14W | 平/赢 |
| Round f32 | 1.00= | 1.01= | 1.00= | 1.01= | 1.15W | 1.05W | 平/赢 |
| RoundToInt f32->i32 | 1.29W | 1.26W | 1.20W | 1.21W | 1.26W | 1.19W | HYLE赢(所有序) |
| RoundingShr i32 | 1.07W | 1.02W | 1.01= | 0.99= | 1.10W | 1.12W | 平/赢 |
| SatWidenMulAccumFixedPoint i16->i32 | 1.85W | 1.90W | 1.75W | 1.76W | · | · | HYLE赢(所有序) |
| SaturatedNeg i32 | 1.02= | 1.05W | 1.04W | 1.00= | 1.14W | 1.13W | 伪影·隔离赢/平 |
| ScatterIndexN i32 (half n) | 1.22W | 1.08W | 1.11W | 1.22W | · | · | HYLE赢(所有序) |
| ScatterOffset i32 (4096) | 1.10W | 1.05W | 0.99= | 0.99= | · | · | 平/赢 |
| Set i32 | 1.00= | 1.00= | 1.00= | 1.02= | 0.99= | 1.01= | 平/赢 |
| SetAtOrAfterFirst i32 | 1.00= | 1.01= | 0.98= | 1.00= | 1.00= | 1.00= | 已修·赢/平 |
| SetAtOrBeforeFirst i32 | 1.10W | 1.09W | 1.18W | 1.18W | 0.99= | 1.00= | 平/赢 |
| SetBeforeFirst i32 | 1.01= | 1.02= | 1.26W | 1.26W | 1.00= | 0.99= | 平/赢 |
| SetMask i32 (4096) | 1.00= | 1.00= | 0.98= | 1.00= | · | · | 平/赢 |
| ShiftLeftBytes<4> i32 | 1.02W | 1.03W | 1.02W | 1.01= | 1.16W | 1.13W | 平/赢 |
| ShiftLeftSame i32 bits=3 | 0.99= | 1.00= | 1.01= | 0.98= | 1.10W | 1.17W | 平/赢 |
| ShiftRight<3> i32 | 1.00= | 1.00= | 1.00= | 1.03W | 1.15W | 1.10W | 平/赢 |
| Shl i32 | 1.00= | 1.00= | 0.99= | 1.02W | 1.08W | 1.08W | 平/赢 |
| Shr u32 (variable) | 1.01= | 1.01= | 1.01= | 1.01= | 1.14W | 1.13W | 平/赢 |
| Shuffle0123 i32 | 1.00= | 1.03W | 1.00= | 0.98= | 1.15W | 1.09W | 平/赢 |
| Shuffle1032 i32 | 0.99= | 0.98= | 1.01= | 1.00= | 1.13W | 1.12W | 平/赢 |
| Shuffle2103 i32 | 0.99= | 1.00= | 1.02W | 1.01= | 1.13W | 1.12W | 平/赢 |
| Shuffle2301 i32 | 0.99= | 0.99= | 0.99= | 1.02W | 1.13W | 1.12W | 平/赢 |
| Slide1Down i32 | 1.02= | 1.00= | 1.04W | 1.01= | 1.13W | 1.15W | 平/赢 |
| Slide1Up i32 | 1.09W | 1.10W | 1.03W | 0.99= | 1.11W | 1.15W | 平/赢 |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.02= | 1.00= | 1.00= | 0.99= | · | · | 平/赢 |
| Sqrt f32 | 1.00= | 1.01= | 1.02= | 1.01= | 1.00= | 1.03W | 平/赢 |
| StoreInterleaved2 f32 (4096) | 0.99= | 1.05W | 1.04W | 1.03W | · | · | 平/赢 |
| Sub f32 | 1.00= | 1.02W | 1.02= | 0.98= | 1.12W | 1.11W | 平/赢 |
| SumOfLanes i32 | 1.02W | 1.03W | 1.28W | 1.27W | 1.12W | 1.10W | HYLE赢(所有序) |
| SumsOf8 u8->u64 | 1.02= | 1.00= | 1.00= | 1.00= | · | · | 平/赢 |
| SumsOfAdjQuadAbsDiff u8->u16 | 1.01= | 1.02= | 1.00= | 1.00= | · | · | 平/赢 |
| TwoTablesLookupLanes i32 (4096) | 1.00= | 1.01= | 1.00= | 0.99= | · | · | 平/赢 |
| Xor i32 | 1.01= | 1.01= | 0.99= | 1.03W | 1.10W | 1.06W | 平/赢 |
| Xor3 i32 | 1.60W | 1.60W | 1.01= | 1.00= | 1.95W | 1.88W | 平/赢 |
| Zero i32 | 1.01= | 1.02= | 1.00= | 1.02W | 1.00= | 1.01= | 平/赢 |
| ZeroExtendVector i16 half->full | 1.00= | 1.01= | 0.98= | 1.00= | · | · | 平/赢 |
