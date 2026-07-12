# SIMD 配对符号检验揪出的真输清单（2026-07-12，2026-07-13 复扫更新）

## 2026-07-13 复扫（补全 bench 覆盖后重跑，含极性纠正）

**⚠️极性纠正（重要）**：`bench-stat.py --pair HYLE Highway` 的 ratio = **rival(Highway) op/s ÷ base(HYLE) op/s**，且 `[sign]WIN/LOSE` 是**从 rival 视角**：`[sign]WIN`(ratio>1, Highway op/s 更高)= **HYLE 慢=HYLE 真输**；`[sign]LOSE`(ratio<1)= Highway 慢 = **HYLE 赢**。（本轮一度读反，用原始 ns/op 核对纠正：如 LoadExpand HYLE 0.633 vs Highway 1.360=HYLE 快，工具标 LOSE=Highway 输。）

**补 bench 覆盖揪出的隐藏真输（directive 三点闭环范例）**：覆盖审计(377 op/276 已对拍/101 缺口)→ 补 15 高价值缺口 bench 行 → sign-sweep 立即揪出 **SumsOf2 9.1× / SumsOf4 8.6×**（三后端全 store+reload 标量循环，此前无 bench 完全隐形）→ 全部照 Highway 追平（avx2/sse4=PromoteEven/Odd+Add，neon=vpaddlq_u8 一条）→ TIE。**坐实"覆盖不足=真输隐形"，补 bench 是发现工具不只背书。**

**本轮全 avx2 sweep 的 HYLE 真输(rival WIN)完整清单 + 处置**：
| op | ratio(H/H) | 处置 |
|---|---:|---|
| CompressStore / CompressBlendedStore i32 | 3.4× / 3.1× | **契约差异非可修**：Compress 本体 native vpermd 已 TIE(0.108 vs 0.110，超 xsimd 7×)；CompressStore 慢=HYLE **精确写契约**(只写 CountTrue 个，尾部不碰)vs Highway 整向量写。AVX2 无 native compress-store，精确写只能标量 pack。契约优先，`StoreU(Compress(v,m))` 可取 Highway 语义的平。**保留强契约。** |
| GatherIndexN / GatherIndexNOr i32 | 2.57× | ✅ **已追平**→native masked gather(`_mm256_mask_i32gather`+FirstN 掩码 fault-safe)，0.46→0.18ns TIE |
| SumsOf2 / SumsOf4 u8 | 9.1× / 8.6× | ✅ **已追平**(见上，补 bench 揪出)，三后端 |
| LoadNOr i32 (half n) | 1.49× | native maskload 已用(9.6→1.5×)；残留需 Highway 递归二分(sub-128 partial-load 基础设施)，登记 |
| GatherOffset i32 | 1.18× | 可 native vpgather(scale=1 byte-offset)追，但内存-bound 收益小，登记 |
| FindKnownLastTrue 1.26 / IsNegative 1.19 / TrailingZeroCount 1.11 / HighestSetBitIndex 1.11 / FindLastTrue 1.07 / MinNumber 1.05 / MaxNumber 1.07 / Max128 1.04 / Eq128 1.02 / Lt128Upper 1.06 / MaskedShift 1.03 / MaskedModOr 1.03 / Mod 1.02 / RoundingShr 1.05 / BitShuffle 1.03 / CombineMasks 1.17 / MaskedApproxReciprocalSqrt 1.05 / ConcatLowerLower 1.08 / MultiRotateRight 1.09(已从 2.9×) | **内存-bound 掩盖 ALU 差 / 布局伪影**：`Load+单op+Store` 循环里多几条 ALU 藏在 load/store 延迟下(第五道关)；这些 objdump codegen 相同或 compute-bound 复测反 WIN(MinNumber compute-bound 1.8×WIN 铁证)。**低优先，多数不该动。** |

---

# SIMD 配对符号检验揪出的真输清单（2026-07-12 原始）

> 用户揪出旧 bench-stat "gap>spreadA+spreadB" 绝对判据把真输淹成平（同机噪声相关，ratio 远比绝对 spread 稳）。已加**配对符号检验**（`scripts/bench-stat.py` 的 `[sign]` 列：N 次 run 里 ratio 一致同侧、二项 p<0.01 即真输，无视绝对 spread）。以下是三平台全 sweep（op_parity full/parity/widen）在符号检验下的真输（HYLE 慢）清单，按 ratio 排序。
>
> **纪律**：WSL/Windows 同物理机必须错开跑（不并发）；Mac 独立可并行。每项修复方向由 WSL sign 判定，正确性四环境验，最终三平台 sign 一致才算追平。

## AVX2（WSL GCC14，符号检验）

### 大输（明确真输，优先）
| op | ratio | 性质 |
|---|---:|---|
| LoadNOr i32 (half n) | 9.64× | store+标量循环+reload；可用 native maskload(fault-safe) |
| CompressStore i32 | 3.42× | 契约差异(精确写)+AVX2 无 native compress-store —— 待重审 Highway |
| CompressBlendedStore i32 | 3.04× | 同上 |
| ~~MultiRotateRight u64~~ **已追平 2.9×→1.1×** | ~~2.90×~~ | ✅ 照 Highway generic 向量化(DupEven/Odd+TableLookupBytes+u16变量Shr+ConcatEven+InterleaveWholeLower)。补 3 子能力:16-bit 变量移位向量化(cvtepu16→srlv32→packus)、ConcatEven/Odd u8/u16 原生化(packus+permq，去 store+reload)、u16 Shr 去冗余 mask。1.037ns vs Highway 0.946ns。见下"已追平"区 |
| GatherIndexN / GatherIndexNOr i32 | 2.59× | 变长 gather harness |
| LoadInterleaved2 f32 | 1.34× | 待查 |
| GatherOffset i32 | 1.16× | gather harness |
| TwoTablesLookupLanes i32 | 1.11× | avx2 无 permutex2，回退 |
| MulHigh32 u32 | 1.10× | 待查 |

### 中/小输（1.02–1.25×，部分可能串扰，需隔离确认）
FindKnownLastTrue 1.25 / IsNegative 1.19 / CombineMasks 1.15 / Min·MaxNumber 1.07 / MaskedApproxReciprocal(Sqrt) 1.06 / FindLastTrue 1.07 / Lt128Upper 1.06 / Max128 1.04 / MulAddSub 1.04 / MaskedModOr 1.03 / RotateLeftSame 1.03 / ConcatOdd 1.03 / BitShuffle 1.03 / MaskedShiftLeft·Right 1.03–1.04 / Clamp 1.03 / MulByPow2 1.03 / Eq128 1.02

## NEON（Apple Silicon，符号检验）

### 明确真输
| op | ratio | 性质 |
|---|---:|---|
| Ceil f32 | 1.40× | 待查(vrndpq 单条?可能 harness) |
| Not i32 | 1.39× | 单指令，极可能 harness 串扰(隔离核) |
| TrailingZeroCount i32 | 1.38× | 待查 |
| RoundingShiftRightSame i32 | 1.14× | 待查 |

### 小输（1.01–1.07×，绝大多数是完整-bench 序列串扰）
Zero/CeilInt/FloorInt/AllBits0/MaskedReduceMax/DupEven/DupOdd/ShiftRight/IfThenElse*/Xor/And/Shuffle0321·2301/SaturatedAbs/Round/PairwiseAdd128/ZeroIfNegative —— 全是单指令或已知最优 op，1.01–1.02× 高度疑似 NEON 完整-bench 相邻 op 串扰（已坐实 Shuffle2301 隔离即平）。**须隔离单-op bench 逐个确认真输 vs 串扰**。

## 已在本轮修复追平（符号检验确认）
SaturatedNeg（位技巧 `~v-(v>MIN?-1:0)`）、SaturatedAbs i32（`Min_u32(Abs,INT_MAX)`）、PromoteMaskTo/OrderedDemote2MasksTo（向量域+PEXT+MaskFromNormalizedVec）、InterleaveEven/Odd（双 shuffle_ps）、TestBit（Eq(v&bit,bit)）、MulByFloorPow2 NEON（vcvtmq）。

## 三平台交集（铁真输，排除单平台 harness/编译器伪影）——2026-07-12 补

三平台(WSL GCC avx2 / MSVC avx2 / Mac NEON)各自 sign sweep 后求交集。**单平台独有的"输"多是 harness/编译器伪影**（NEON 22 输里 21 个单平台独有=串扰；MSVC 76 输含 Add/Max/Min f32 等单指令=MSVC harness），交集才是真算法输。

**三平台都输（1）→ objdump 证伪为 harness 伪影**：`MaskedApproxReciprocal f32`。HYLE 与 Highway 源码逐字相同（都是 `IfThenElseZero(m, ApproximateReciprocal(v))`），隔离 TU 编 `-S` 两函数体**逐指令完全相同**（vmovups→vbroadcastss→vcmpps→vrcpps→vandps→vmovups）→ 不可能有真算法差，三平台都显 1.05–1.27× 是各平台完整-bench 的一致 harness 串扰。**⚠️关键教训：符号检验"一致同侧"+三平台交集仍不足以定真输——一致方向也可能来自一致的 harness 偏差；必须 objdump 看 codegen 是否真不同才能终审。** 交集里每一项都要 objdump 分类（codegen 不同=真输可修 / 相同=伪影）。

**avx2 两编译器(GCC+MSVC)都输（20，x86 真算法输）**：Eq128 / Max128（128-cmp）、CompressStore / CompressBlendedStore / GatherOffset / LoadNOr / LoadInterleaved2 / BitShuffle / TwoTablesLookupLanes / MultiRotateRight（内存/复杂）、MaskedApproxReciprocal(Sqrt) / MaskedMulComplexConj / MaskedShiftLeft·Right<3>（Masked 族）、FindKnownLastTrue、MinNumber / MaxNumber、MulAddSub、ConcatOdd。

**⚠️ MSVC 独有的大输（GCC 已追平但 MSVC 没有，编译器 codegen 差异，需查）**：SetAtOrAfterFirst/OnlyFirst/BeforeFirst/AtOrBeforeFirst 5×、OrderedDemote2MasksTo 4.6×、Reverse2/4/8 2.8×、Dup128MaskFromMaskBits 5.4×、PairwiseAdd/Sub。这些我在 WSL(GCC) 用向量化追平了，但 **MSVC 对同一份 C++ 的 codegen 没优化到位**——说明"追平"必须三平台各自验证，不能只信 GCC。

**NEON 独有小输（1.01–1.07×，串扰待隔离确认）**：Not/Xor/And/Zero/DupEven/Shuffle 等单指令。

## "一边倒 HYLE 慢"根因调查（2026-07-12，用户观察驱动）——布局偏差非算法

用户注意到 sign sweep **一边倒 HYLE 慢**（MSVC 76 sign-WIN 几乎无 sign-LOSE），怀疑伪影/顺序问题。三步决定性实验：
1. **run 顺序不是根因**：同 op 各测 HYLE-first / Highway-first 两序（order_bias 探针）。MaskedApproxReciprocal **两序都 HYLE 慢 ~3.5%，不翻转** → 排除"先跑吃冷启动"。
2. **nanobench 两 slot 公平**：同一份代码放两个 run slot（selfsame 探针，G/G 和 H/H）→ **都干净 TIE**（p=0.61/0.42）→ harness 对两 slot 无系统偏差。
3. **codegen 相同却稳定慢**：mrecip 的 HYLE/Highway 内联后**逐指令相同**（objdump 证），却稳定 HYLE 慢 3.5% 且不随顺序翻转。

**结论**：根因是**代码布局偏差**——HYLE/Highway 是不同函数符号，链接后落在不同地址/对齐，icache/DSB/预测细微差异；每个 bench 宏里 HYLE 恒先定义/先实例化，系统性地让 HYLE 的布局吃 ~2–4% 的亏。**这是真实但与算法/代码质量无关、且不可控（取决于链接布局）的效应。**

**对真输判定的修正（第四道关）**：
- **≤ ~1.05× 的一边倒 HYLE 慢**：极可能是这 ~2–4% 布局偏差，**不是真输**。除非 objdump 显示 codegen 真不同，否则不动。
- **真输门槛上调**：只有 ratio 明显超出布局偏差量级（经验 ≥ ~1.10×）**且** objdump codegen 不同，才是真算法输。据此，交集 20 项里真正该修的收窄到大 ratio 的内存/复杂族（LoadNOr 9.6/Compress 3/Gather 2.6/MultiRotate 2.9/LoadInterleaved2 1.34/TwoTables 1.11/MulHigh32 1.10）+ 逐个 objdump 确认 codegen 不同的中 ratio 项。1.02–1.05× 的（Eq128/Max128/MulAddSub/ConcatOdd/Min·MaxNumber/MaskedShift 等）先 objdump，codegen 相同即布局伪影不修。

## 已追平（本轮，符号检验+objdump 确认真 codegen 差+照 Highway 改+三平台验）
SaturatedNeg i32（1.09×→位技巧 `~v-(v>MIN?-1:0)`）、SaturatedAbs i32（1.34×→`Min_u32(Abs,INT_MAX)`）、MulHigh32 u32（1.10×→shuffle-pick 5 条）、TwoTablesLookupLanes（1.11×→slli+blendv）、LoadInterleaved2 f32/i32（1.34×→shufps+permq）、ConcatEven/Odd 32-bit（→shufps+permq deinterleave，ConcatOdd 1.03→TIE）、PairwiseAdd/Sub 32-bit（两次 vpermq factor 成一次 `permq(x+y)`，4→3 p5-uop；试 native vphadd 隔离慢 10% 放弃）。**LoadNOr 9.6×→1.5×**（native maskload，剩余需 Highway 二分）。

**MultiRotateRight u64 2.9×→1.1×（大输，本轮照 Highway generic 向量化）**：旧=标量 N×8 双循环(2.83ns)。照 `generic_ops-inl.h:8484`：DupEven/DupOdd(vpermilpd)分离奇偶 u64 lane → TableLookupBytes(vpshufb)取每输出字节的相邻两源字节 → u16 变量 Shr 抽非对齐字节 → ConcatEven 打包低字节 → InterleaveWholeLower 重织。为此补 3 个原生子能力(均独立价值)：①**16-bit 变量移位向量化**(AVX2 无 sllv/srlv/srav_epi16=AVX512BW，照 Highway AVX2ShrU16Vec256：两半 u16→u32 cvt + native 32-bit 变量移位 + packus 回 u16 + permq D8；96 万随机含 count≥16 对拍绿)；②**ConcatEven/Odd u8/u16 原生化**(mask/srli+packus+permq，去 store+reload——u8 ConcatEven 的 store+reload 曾把向量版拖到 3.4ns 比标量还慢，是关键瓶颈)；③u16 Shr 去冗余 low-16 mask。objdump 81 insn/**0 mem**(旧 100/7) vs Highway 65。8 万随机 cross-check + gtest(rot=0/j*8 identity/splat 边界) + 221 avx2/220 sse4 test 全绿。最终 1.037ns vs Highway 0.946ns(1.1×，布局带内)。**教训：向量化"照抄"必须逐个确认子能力真的向量化了——u16 Shr / u8 ConcatEven 藏着 store+reload 时，"向量化版"反而比标量慢；objdump 数 `(%rsp)` 内存标记揪隐藏标量步。**

## 剩余中/小 ratio 候选：四道关分类终审（2026-07-12，本轮做完）——**结论：无真输**
按用户"最好把剩余分类做完"，对损失清单里剩余中小 ratio 项逐个走完整四道关（sign + 隔离单-op + objdump + 布局阈值），**全部证为 HYLE-WIN / TIE / 布局伪影，无一真输**：

| op | full-bench sign | objdump | 隔离/compute-bound 复测 | 定性 |
|---|---|---|---|---|
| MaskedShiftLeft/Right<3> i32 | ~1.03 "WIN" | HYLE 与 Highway **逐指令相同** | — | **布局伪影**（codegen 同） |
| RotateLeftSame u32 | 1.02 TIE | 功能等价（andl/negl+2 shift+por） | — | **布局伪影** |
| FindLastTrue i32 | 1.06 "WIN" | HYLE **更少**指令（Highway 多 `movq -1` 默认） | — | HYLE 本就不输 |
| MaskedMulComplexConj(Add) f32 | 1.02/1.01 TIE | HYLE 尾部多 `vpxor+vpcmpgtd` 重建 mask（18 vs 14） | — | ratio<gap 阈值，TIE |
| MaskedModOr i32 | 1.03 "WIN" | — | — | HYLE-WIN |
| Clamp i32 / MulByPow2 f32 / MulAddSub f32 | TIE~WIN | codegen 同（前轮已定） | — | 布局伪影/TIE |
| **HighestSetBitIndex u32** | 1.11 "WIN" | HYLE 17 vs Highway 10（HYLE 走 LZCnt 的 min-32 clamp；Highway 走 `NormalizeForUIntTruncConvToF32` 多个 AndNot） | 隔离 1.06×WIN；**compute-bound 1.16×WIN** | **HYLE-WIN**（LZCnt 复用比 Highway normalize 更省，勿改） |
| **Lt128Upper u64** | 1.05 "WIN" | HYLE 广播 3 条（vpermilpd+vpsrad+vpshufd）vs Highway 1 条（vpunpckhqdq） | 隔离 1.06×**WIN** | 内存 bound，ALU 差被 Load/Store 掩盖；sign 方向已翻成 WIN=布局 |
| **MinNumber/MaxNumber f32** | 1.07/1.09 "WIN" | HYLE 查**双**操作数 NaN（5 insn）vs Highway 单（3 insn） | **compute-bound 链式 1.8×WIN！** | **HYLE 大胜**；前轮"1.07 慢"纯内存-bound 布局噪声；曾拟"简化到 3 条对齐 Highway"会**反而变慢**，撤销 |

**关键教训（补第五道关）**：`Load+单op+Store` 循环是**内存带宽 bound**，objdump 里 HYLE 多几条 ALU 指令**不转化为真输**（藏在 load/store 延迟下），full-bench 甚至读成 HYLE-WIN。要判 ALU 差是否真实，**必须 compute-bound 复测**（每元素链 4~8 个同 op，去掉 load/store 主导）。MinNumber compute-bound 1.8×WIN 是铁证：更"重"的 codegen（双 NaN 检查）在真实计算负载下反而快——**指令数少 ≠ 更快，别拿 objdump 指令数当性能判据**。

## 128-cmp 族 objdump 定性（真 codegen 差但内存-bound 掩盖，隔离读 WIN）
Eq128/Ne128/Lt128/Lt128Upper/Min128/Max128（full-bench 1.02–1.06×）。objdump Max128：HYLE 11 insn（vpcmpgtq 后用 vpermq+vpermilpd+vpsrad+vpshufd 4 条广播 128-bit 比较结果），Highway 9 insn（vpslldq+vpunpckhqdq 2 条），Lt128Upper 同理 HYLE 3 条广播 vs Highway 1 条 vpunpckhqdq。**真 codegen 差（非布局伪影）**，但 `Load+cmp+Store` 是内存-bound，多的 2 条广播 ALU 藏在 load/store 延迟下：**Lt128Upper 隔离单-op sign 复测读 1.06× HYLE-WIN**（方向翻转）。即"codegen 更省"的 Highway 在真实内存负载下并不快。要坐实是否真差需 compute-bound 链式复测；ratio 小 + 128-bit 高低 64 位广播逻辑易错，**低优先**，收益远小于 MultiRotateRight 2.9×。

## 施工方针
1. 大输优先（内存族 LoadNOr/Compress/Gather、MultiRotateRight）——收益最大且明确。
2. 中输逐个对照 Highway 源码找多余步骤。
3. 小输（1.01–1.05×）先隔离单-op bench 确认是真输还是完整-bench 串扰，是串扰则不动（记录），是真输才修。
4. 每项：WSL sign 判方向 → 四环境正确性 → 三平台 sign 一致 → 提交。
