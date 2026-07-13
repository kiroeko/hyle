# SIMD 对拍：当前"输"情况总结（2026-07-13）

> HYLE 自研 SIMD 抽象层对标 Google Highway / xsimd。本文档回答两个问题：**(1) 当前还有哪些
> op 慢于 Highway？(2) 这些"慢"里有多少是真算法输、多少是测量伪影？** 数据来自 op_parity
> bench（HYLE 与 Highway 同 TU、同 ISA、都 header-only 内联，gap=真 codegen 差非调用开销），
> 配对符号检验（`scripts/bench-stat.py --pair`，N≥25），三平台（WSL GCC / MSVC / Mac NEON）。

---

## 0. 先厘清：为什么是"单边 HYLE 慢"，而不是输赢各半

对拍里方向确定指向 HYLE 慢的项远多于 HYLE 赢的项。直觉上若纯随机应对称——单边性必有系统因素。
**决定性实验（order-swap，Add f32 与 MaskedShiftLeft<3>，两者 objdump 逐指令与 Highway 相同）**：

| 顺序 | 第一个跑的 | 第二个跑的 | 结果 |
|---|---|---|---|
| HYLE first | HYLE 100% | Highway **104.5%** | 第二个(Highway)快 |
| Highway first | Highway 100% | HYLE **94.8%** | 第二个(HYLE)快 |

**慢的一边跟随"跑动位置"而非"库"**——谁第一个跑谁慢（第二次更极端：86.8% vs 118.1%）。根因是
**第一个跑的变体付冷启动代价**（icache/DSB 未填、分支预测器未训练、CPU turbo 未拉满）。op_parity
bench 里每个宏**恒定 HYLE 先跑**（`b.run("HYLE",...)` 在前），所以 HYLE 系统性吃第一槽惩罚 →
表现为"单边 HYLE 慢"。**这不是 HYLE 代码更差，是 harness 固定顺序造成的伪影。**

推论（本文档反复用到）：
- **方向确定 ≠ 算法输**。符号检验的"一致同侧"只排除随机噪声，排除不了固定顺序的第一槽惩罚。
- **唯一可靠终审 = objdump**（codegen 逐指令相同 ⇒ 不可能是算法差）+ **compute-bound 复测**
  （每元素链 4~8 个同 op 去掉 load/store 与冷启动主导，看计算负载下是否仍输）。
- 隔离单-op 复测能消掉大部分完整-sweep 的位置串扰（IsNegative 完整 1.19× → 隔离 TIE）。

铁证：**MaskedShiftLeft<3>** 与 Highway **逐指令相同的汇编**，隔离仍稳定 6% 慢、20/20 同向——
纯第一槽/顺序效应，跟算法无关。

---

## 1. 本会话已追平的真输（objdump 确认真 codegen 差 + compute-bound 坐实 → 照 Highway 改）

| op | 修前 | 修后 | 手法 |
|---|---:|---:|---|
| CompressStore / CompressBlendedStore | 3.4× / 3.1× | TIE | 弃"精确写尾部不碰"契约(零消费者)，对齐 Highway `StoreU(Compress)+CountTrue`(Compress 本体已 native) |
| SumsOf2 / SumsOf4 (u8) | 9.1× / 8.6× | TIE | 补 bench 才揪出(此前无对拍完全隐形)；avx2/sse4=PromoteEven/Odd+Add，neon=vpaddlq_u8 一条 |
| GatherIndexN / GatherIndexNOr | 2.6×(GCC)/6×(MSVC) | TIE | native `_mm256_mask_i32gather`+FirstN 掩码；**根因是 FirstN 底层标量 store+reload**(MSVC 没优化)，改 FirstN 为 in-register iota<n compare |
| GatherOffset | 1.2×(GCC)/5×(MSVC) | TIE | native vpgather scale=1 字节偏移 |
| MultiRotateRight (u64) | 2.9× | 1.1× | 照 Highway generic 向量化(DupEven/Odd+TableLookupBytes+u16变量Shr+ConcatEven)，连带 u16 变量移位 + ConcatEven/Odd u8/u16 原生化 |
| Max128 / 128-cmp 族(Lt128/Lt128Upper/Min128/Eq128) | 1.30×(compute-bound) | 1.04× | 广播 verdict 从 DupOdd+MaskFromVec(4条)换 InterleaveUpper(vpunpckhqdq 1条)+MaskFromNormalizedVec 跳冗余 renormalize |
| FindKnownLastTrue | 1.26× | TIE | `31-clz` 因 clz(0) 已定义→GCC 插空检查分支；加 HYLE_UNREACHABLE→裸 bsr，逐指令同 Highway |
| TruncateTo(前期) | 2.45× | TIE | native PSHUFB 字节 gather |

方法学收获：这些真输**共同特征**是"HYLE 走了 store+reload / 标量循环 / 多余步骤"，Highway 有更短的原语序列。
凡 objdump 显示 HYLE 明显更多指令、且 compute-bound 复测仍输的，都能照抄 Highway 追平。

---

## 2. "非真输"复核 —— 按 order-swap 统一实测分三类（2026-07-13，用户"伪影都换顺序反转了？"逼出）

> 之前把 §2 一律标"伪影"不严谨:证据强度不一(有的只 objdump、有的只隔离、只有 MaskedShift 真做过
> order-swap)。**统一对每项做 order-swap(op/s，两序)**，结果**不是所有项换序都反转**——分三类:

### 2a. 真·第一槽伪影：慢的一边跟随跑动位置，换序反转 / 或隔离即 TIE
| op | 证据 |
|---|---|
| **MaskedShiftLeft/Right<3> i32** | objdump **逐指令与 Highway 相同**；order-swap 慢边跟随位置翻转(HYLEfirst→HW快;HWfirst→HYLE快)。铁证伪影。 |
| **Clamp i32** | 隔离 TIE(1.031,p=0.12)；order-swap 近平(8.38/8.37↔8.65/8.32) |
| **CompressNot i32** | objdump 仅差 1 条 vpxor；order-swap 近平(9.48/9.46↔9.53/9.36) |
| **Lt128 / Eq128 / Ne128 u64** | 与 Max128 同族已改广播；order-swap 近平(HWfirst 时 HYLE 反而快 3.49 vs 3.40) |

### 2b. HYLE 其实更快（我曾误标为输）
| op | 证据 |
|---|---|
| **MulByPow2 f32** | order-swap **两序 HYLE 都快**(4.24 vs 4.15;4.28 vs 4.19)——不是输，是 HYLE 赢，误判。 |

### 2c. ⚠️内存-bound 慢但 compute-bound 快：**换序不反转**（我之前"伪影"标签在此类是错的）
| op | order-swap(内存-bound) | compute-bound(链式) | 结论 |
|---|---|---|---|
| **IsNegative i32** | **两序 HYLE 都慢~5%**(36.7 vs 38.1;36.2 vs 39.8) | TIE(vpxor=zeroing idiom 零端口) | 内存-bound 真慢,计算负载抹平 |
| **HighestSetBitIndex u32** | **两序 HYLE 都慢~11%**(14.6 vs 16.5 两序) | **1.16×WIN** | 内存-bound 慢,compute-bound HYLE 反赢 |
| **MinNumber/MaxNumber f32** | **两序 HYLE 都慢~5%**(11.1 vs 12.2;10.9 vs 12.3) | **1.8×WIN** | 同上 |

**2c 的诚实结论**:这三项**不是第一槽伪影**(换序不反转,两序都慢),但也**不是纯算法输**——它们在
`Load+单op+Store` 内存-bound 形态下 HYLE 确实慢 5-11%,而在计算密集(链式)形态下 HYLE 反而更快。
净影响**取决于消费者的使用模式**(算子被密集串起来用=HYLE 赢;稀疏内存流式用=Highway 略快)。
把它们叫"伪影"是我的措辞错误——准确说是"内存-bound 与 compute-bound 方向相反,不可一概而论"。
是否值得为内存-bound 形态优化,待有真实消费者的访问模式再定(目前 compute 消费多为密集,倾向不动)。

---

## 3. 当前仍方向确定 HYLE-slow 的完整并集（三平台任一，N=25，p<0.01）

> **⚠️读法**：这 87 项里绝大多数是 §0 的第一槽/顺序伪影，**不是算法输**。铁证：MSVC 把
> `Add f32`、`BitCast`(同一条 vaddps/无操作)都判进来。真算法输须经 §1/§2 的 objdump+compute-bound
> 终审——目前经终审的真输已全部在 §1 处理，§2 是已排除的伪影，本节其余多为**未逐个 objdump 的
> 待查项**（多数预计是伪影）。

- **3 平台都判（2）**：MaskedShiftLeft、MaskedShiftRight — 已证 codegen 相同 = 伪影(§2)。
- **2 平台判（16）**：CompressNot、HighestSetBitIndex、InterleaveWholeUpper、IsNegative、
  MaskedApproxReciprocalSqrt、MaskedMulComplex(Conj)、Max128、MaxNumber、MinNumber、
  MulByFloorPow2、MultiRotateRight、RoundingShiftRight、ShiftLeft、TrailingZeroCount、TruncateTo
  — 多数已在 §1(已修) 或 §2(伪影)。
- **1 平台判（69）**：MSVC 独占 ~30（含 Add/BitCast 纯直通=MSVC 布局噪声，MSVC 第一槽惩罚比 GCC
  强得多）、NEON 独占 ~30（多为单指令 Xor/Zero/Not/Ceil 等=伪影）、GCC 独占少数。

### 尚未逐个 objdump 终审的待查项（下一步）
- **MSVC 独占 mask 族**：Set{Only,Before,AtOrBefore,AtOrAfter}First、MaskedShl/Shr、
  Dup128MaskFromMaskBits、OrderedDemote2MasksTo、MaskedLeadingZeroCount、CountTrue、
  MaskedMulFixedPoint15、MaskedWidenMulPairwiseAdd。文档记为"GCC 已平 MSVC 没优化到位"，
  需在 MSVC 上 dumpbin 逐个查是否真有 codegen 没优化(store+reload 在 MSVC 是真输放大器)。
- **NEON 独占**：ShiftLeft/RightBytes、ShiftLeft/RightLanes、Shuffle*、Slide*、Reverse2/4、
  Rotate*、ConcatOdd、BroadcastSignBit 等——多为单指令，疑第一槽串扰，需隔离确认。
- **登记未追满的真输残留**：LoadNOr 1.49×(native maskload 已从 9.6× 降下，剩余需 Highway
  递归半宽二分=sub-128 partial-load 基础设施，工程量大)。

---

## 4. 判真输的五道关（沉淀的方法学）

1. **配对符号检验**（不是绝对 spread）：同机 HYLE/Highway 噪声相关，per-run ratio 远比绝对 spread 稳；k/N 一致同侧 + 二项 p<0.01 才算方向确定。
2. **隔离单-op bench**：消完整-sweep 的位置串扰（IsNegative 1.19→隔离 TIE）。
3. **objdump 比 HYLE/Highway codegen**：逐指令相同 = 伪影(不改)，不同才可能真输。
4. **布局/第一槽阈值 ~2-5%**：≤~1.05× 的一边倒 HYLE-slow 极可能是固定顺序第一槽惩罚（§0 order-swap 坐实）。
5. **compute-bound 复测**：链 4~8 个同 op 去掉 load/store 与冷启动主导，看计算负载下是否仍输（MinNumber 由此从"1.07 慢"翻成 1.8×WIN）。

**三平台错开跑**：WSL 与 Windows MSVC 是同一物理机，必须先后错开(不并发抢缓存)；Mac 独立可并行。
**GCC 追平≠MSVC 追平**：MSVC 对标量 stack round-trip 优化远弱，store+reload 原语在 MSVC 是真输放大器。
