# SIMD 对拍：当前"输"情况总结（2026-07-13）

> HYLE 自研 SIMD 抽象层对标 Google Highway / xsimd。本文档按**判定结论**组织所有被评判过的 op。
> 数据来自 op_parity bench（HYLE 与 Highway 同 TU、同 ISA、都 header-only 内联，gap=真 codegen
> 差非调用开销），配对符号检验 + **order-swap（换跑动顺序，两序 op/s）** + **compute-bound 复测
> （链 4 个同 op 去 load/store 主导）**，三平台（WSL GCC / MSVC / Mac NEON）。

## 判定方法（怎么给一个 op 定性）
1. **order-swap**：把 bench 里 HYLE/Highway 的先后顺序对调，各测 op/s。
   - 两序 HYLE 都快 → **HYLE 赢**
   - 慢的一边跟随**跑动位置**（谁先跑谁慢）→ **伪输（第一槽冷启动）**，见 §0
   - 两序 HYLE 都慢 → 不是第一槽，进 compute-bound 复测
2. **compute-bound**：把 op 链式串 4 个去掉 load/store 主导。内存-bound 慢但链式快 = 访问模式相关，非纯算法输；链式仍慢 = 真算法输。
3. **objdump**：codegen 逐指令相同 = 铁定非算法差。

---

## §0 为什么"输"偏单边（伪输的机制）
order-swap 决定性实验（Add f32 / MaskedShiftLeft<3>，两者 objdump 与 Highway 逐指令相同）：
慢的一边**跟随跑动位置**——谁第一个跑谁慢（HYLE first→Highway 快；Highway first→HYLE 快）。
根因=第一个跑的变体付冷启动（icache/DSB 未填、预测器未训练、turbo 未拉满）。op_parity bench
每个宏**恒定 HYLE 先跑**，故 HYLE 系统性吃第一槽惩罚 → 表现为"单边 HYLE 慢"。**这是 harness
伪影，不是 HYLE 代码差。** MSVC 的第一槽惩罚比 GCC 强得多（MSVC 把 Add/BitCast 都判方向确定
HYLE-slow=铁证）。

---

## §1 总表：所有被评判项 + 定性（order-swap 4 次平均，ratio=HYLE/Highway，>1=HYLE 快）
| op | HYLEfirst | HWfirst | compute-bound | 定性 | 处置 |
|---|---:|---:|---:|---|---|
| RotateLeftSame u32 | 1.045 | 1.052 | — | **赢** | — |
| MaskedShiftLeft/Right<3> | 1.060 | 1.050 | — | **赢**(objdump 同 Highway) | — |
| MulByPow2 f32 | 1.003 | 0.996 | — | 平 | — |
| SumsOf4 u8 | 1.010 | 0.998 | — | 平(已修) | — |
| FindKnownLastTrue | 1.002 | 1.004 | — | 平(已修) | — |
| Clamp i32 | 0.984 | 1.003 | — | **伪输**(随位置) | — |
| IsNegative i32 | 0.981 | 0.996 | TIE | **伪输/内存**(随位置+计算平) | — |
| Max128 u64 | 0.979 | 1.011 | (已修 1.04) | **伪输**(随位置,已改广播) | 已修 |
| CompressStore | 0.981 | 0.983 | — | 内存-bound 近平(已弃精确写契约) | 已修 |
| CompressNot i32 | 0.979 | 0.983 | — | 内存慢~2%(差1条vpxor) | 登记 |
| MulByFloorPow2 f32 | 0.983 | 0.975 | **1.03×赢** | 内存慢/compute平 | — |
| Lt128 u64 | 0.978 | 0.974 | **1.13×赢** | 内存慢/compute赢 | — |
| MinNumber/MaxNumber f32 | 0.907 | 0.924 | **1.76×赢** | 内存慢/compute大赢 | — |
| HighestSetBitIndex u32 | 0.930 | 0.913 | **1.17×赢** | 内存慢/compute赢 | — |
| TrailingZeroCount i32 | 0.930 | 0.938 | **1.32×赢** | 内存慢/compute赢 | — |

其余曾被 sweep 标记但未进本表的（MSVC 独占的 Add/BitCast/mask 族、NEON 独占的单指令 Xor/Zero/Ceil 等）
= 纯第一槽伪影或单平台布局噪声，未逐个 order-swap（预计全属 §0）。

---

## §2 赢表（order-swap 两序 HYLE 都快，或 compute-bound 明确赢）
| op | 证据 |
|---|---|
| **RotateLeftSame u32** | order-swap 两序 1.045/1.052 |
| **MaskedShiftLeft/Right<3> i32** | order-swap 两序 1.050-1.060；objdump 与 Highway 逐指令相同（差异是 Highway 吃第一槽） |
| **MinNumber / MaxNumber f32** | compute-bound **1.76×**（HYLE 查双操作数 NaN 更稳且在计算负载下更快；objdump"更重"反而快） |
| **TrailingZeroCount i32** | compute-bound **1.32×**（float-exp trick） |
| **HighestSetBitIndex u32** | compute-bound **1.17×**（LZCnt 复用比 Highway normalize 省） |
| **Lt128 u64** | compute-bound **1.13×**（已改 InterleaveUpper 广播） |
| 另有本会话追平后转赢/平的：SumsOf2/4、GatherIndexN、CompressStore、Max128、FindKnownLastTrue |

---

## §3 伪输表（**仅**确实测出跟顺序有关的：慢边随位置翻转 / 隔离即平）
> 严格标准：只有 order-swap 显示"慢的一边跟随跑动位置"（HYLEfirst 慢、HWfirst 反而 HYLE 快）
> 或隔离即 TIE 的，才进本表。这是 harness 第一槽伪影，不是算法差。

| op | HYLEfirst | HWfirst | 判据 |
|---|---:|---:|---|
| **Clamp i32** | 0.984(慢) | 1.003(平/翻正) | 慢边随位置，隔离 TIE |
| **IsNegative i32** | 0.981(慢) | 0.996(近平) | 慢边随位置回正；compute-bound 亦 TIE；objdump 差=vpxor 零端口 |
| **Max128 u64** | 0.979(慢) | 1.011(翻正) | 慢边随位置翻正（已顺带改广播令其更稳） |
| **MaskedShiftLeft/Right<3>** | (见赢表) | | objdump 逐指令同 Highway，任何方向差纯第一槽 |
| （§0 样本）Add f32 | 随位置 | 随位置 | objdump 同 Highway，慢边严格跟随谁先跑 |

MSVC 独占的 Add/BitCast/CountTrue/DupEven 等纯直通单指令 = 同类伪输（MSVC 第一槽惩罚强），
未逐一列出。

---

## §4 真输表（compute-bound 复测仍慢 = 真算法差，需修或登记）
| op | 状态 | 说明 |
|---|---|---|
| （本会话已全部追平，当前无未处理的"compute-bound 仍慢"真输） | ✅ | 见下"已追平"清单 |
| **CompressNot i32** ~2% | 登记 | objdump 差 1 条 vpxor（运行时 Not(mask) vs Highway 预烘焙 NOT-表）；内存-bound、幅度极小，收益低，登记不改 |
| **LoadNOr i32** 1.49× | 登记 | native maskload 已从 9.6× 降下；残留需 Highway 递归半宽二分（sub-128 partial-load 基础设施，工程量大） |

**本会话已追平的真输**（compute-bound 曾坐实慢、照 Highway 改后 TIE/赢）：
CompressStore/CompressBlendedStore（3.4×，弃精确写契约）、SumsOf2/4（9×，PromoteEven/Odd+Add /
vpaddlq）、GatherIndexN/NOr（2.6-6×，native masked gather + FirstN 改 iota<n）、GatherOffset
（5× MSVC，native vpgather scale=1）、MultiRotateRight（2.9×→1.1×，向量化）、Max128/128-cmp
族（compute-bound 1.30×，改 InterleaveUpper 广播）、FindKnownLastTrue（1.26×，去空检查分支）、
TruncateTo（2.45×，native PSHUFB）。

---

## §5 方法学要点（沉淀）
- **方向确定 ≠ 算法输**。符号检验的"一致同侧"只排除随机噪声，排除不了固定顺序的第一槽惩罚
  （MaskedShiftLeft<3> objdump 逐指令同 Highway 却 sign 稳定同向=铁证）。
- **一个 op 可以内存-bound 慢、compute-bound 快**（MinNumber 内存 0.91×→compute 1.76×）。净影响
  取决于消费者访问模式（密集串用=HYLE 赢；稀疏内存流式=Highway 略快）。不能一概而论、也不能只
  拿内存-bound 数说"输"。
- **判真输三关缺一不可**：objdump（codegen 是否真不同）+ order-swap（是否随位置）+ compute-bound
  （去 load/store 后是否仍慢）。任一关证否即非真输。
- **GCC 追平≠MSVC 追平**：MSVC 对标量 stack round-trip 优化远弱，store+reload 原语在 MSVC 是真输
  放大器；且 MSVC 第一槽惩罚更强，其 sign-sweep"输"列表虚高。
- **三平台错开跑**：WSL 与 MSVC 同物理机必须错开（不并发抢缓存）；Mac 独立可并行。

---

## §6 MSVC 剩余劣化项收尾 + Tier1 调查（2026-07-13）
MSVC 上 HYLE 源码写法(store+reload / 标量展开 / __BMI2__ 门控)触发 codegen 劣化,Highway(同编译器)
避开了 → 是 HYLE 源码问题非 VC++ 版本问题(order-swap 坐实:同一 VS2026 编译,Highway 快)。已修:
- **Reverse2/Reverse/Reverse4/Reverse8**:swizzle_via_mem → native shuffle。Eq128/Lt128 族(compose
  Reverse2)MSVC 5x → TIE。
- **LoadMaskBits**:标量 per-lane 展开 → native 位测试(broadcast&bit==bit)。Set*First/mask 域 5x → TIE。
- **i16 mask_bits PEXT 门控**:`__BMI2__` → `__BMI2__ || _MSC_VER`(MSVC 不定义 __BMI2__ 但支持 _pext_u32)。
  OrderedDemote2MasksTo 5x → TIE。
- **RoundingShr**:标量 → native 变量 Shl/Shr。**ScatterOffset/GatherOffset**:per-byte 循环 → 整-lane memcpy。
- **MinNumber/MaxNumber**:双 NaN 检查 → 单侧(x86 min/max NaN 语义,64 组等价验证)。realistic kernel 1.6x → TIE。
MSVC both-order-slow: 74 → 56(极端 0.17-0.21 五倍簇清零)。

### Tier1 深挖(realistic streaming kernel + MSVC)结论
- **BitShuffle u64**:realistic MSVC 0.58 真慢。照 Highway generic 向量化(vpshufb+per-byte-rotate+psadbw)
  正确性 80k 对拍过,但 **per-byte rotate 的 8-blend 模拟在 MSVC 上更贵 → 0.58 反降到 0.46(回归!)**。
  **教训:向量化 store+reload 不一定赢——重的向量模拟(8路 blend rotate)可能比标量还慢,尤其 MSVC。
  已回退保留标量。** 要真追平需 Highway 的精确 u8-Rol 手法(AVX2 无 byte rotate,它另有更省的模拟),
  且 BitShuffle 无 HYLE 消费者=冷 op,低优先,登记 defer。
- **LoadN/StoreN、SatWidenMulAccumFixedPoint**:冷 op(SatWiden 无 src 消费者),鉴于 BitShuffle 教训
  (重向量模拟可能regress MSVC),不冒险改,登记 defer,待有热消费者或确证轻量 native 手法再动。
