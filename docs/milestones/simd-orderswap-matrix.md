# SIMD op parity 三平台矩阵（2026-07-16 校准仪器 v4；明确 LOSE：MSVC 0 / GCC 0 / NEON 0）

> **performance 导航（整理入口）**：[../performance/benchmarks-simd-ops.md](../performance/benchmarks-simd-ops.md)  
> （摘要 + 覆盖树 + 读法；**本文件仍是全量 case 表 / 修史 / ratio 全扫的权威正文**。）  
> 产品 kernel 不在此 → [../performance/benchmarks-simd.md](../performance/benchmarks-simd.md)。

> **simd-22 后再认证完成（2026-07-16）**：Windows/WSL/macOS 三个 TU 均以当前双轨源码、双序
> 各两份、4 replicas、layout bank 与执行审计重新出数；九份 report union 为 303/303、缺失 0。
> 旧表已由下方 320-case 新矩阵取代。所有 ratio `<0.98` 的行均完成源码/形状检查；真实回归逐项
> 修复，剩余低值全部为 `INCONCLUSIVE`，没有把它们偷换成 TIE。

> **权威计数(唯一口径 = `scripts/simd-op-coverage.py` 机械算 + 自校验,勿手数/勿拼快照)**。
>
> **统计方式(可复现)**:脚本解析 `vec.h` + 7 个后端头(avx2/sse4/avx512/neon/scalar/sve/rvv),用花括号
> 深度栈跳过 `namespace detail{}`,取 `HYLE_FORCE_INLINE` 的公开 op 名(CamelCase,跳过全大写宏与小写
> helper),跨头按 op 名去重。**自校验闭环**:已 bench 集必须 ⊆ 提取的全集,否则说明提取规则有盲区→报错
> 重跑(开发时抓出过 10 个盲区:`Load/Store` 复合标题、`Mask And` 前缀、run_reduce/cmp128 自定义 helper、
> 跨行字符串合并、大写宏泄漏)。已测集来自扫 `op_parity_{full,,widen}_bench.cc` + `crypto/gather_bench.cc`
> 的所有字符串字面量、映射到 op 名。SOTA 判决集 = 本文全量表里有 verdict 行的 op。
>
> **计数树**（2026-07-16 现跑 `scripts/simd-op-coverage.py`）：
> ```
> 387  公开 op 名(去重 8 后端头,剔 tag/trait/detail helper)
>  └─ 344  可测全集
>       │   ( − 26 不可测: constexpr 谓词(DivIsPow2)/常量产生器(Inf/NaN/Iota/Zero/Set)/
>       │     introspection(Lanes/Blocks)/单-lane 标量存取(Get·Extract·InsertLane)/纯 retag(BitCast)/
>       │     构造 helper(Combine/Lower·UpperHalf/LoadDup128/SetTableIndices/IndicesFromVec/FirstN)
>       │     − 13 内部选择器变体(*Native/*Emulated,由伞 op 覆盖)
>       │     − 4 别名(Rol→RotateLeft/Ror→RotateRight/ZipLower→InterleaveLower/ZipUpper→InterleaveUpper) )
>       ├─ 314  可测全集内有独立 source benchmark
>       └─  30  不独立测且已显式分类
>              (alias 6 / synthetic 19 / paired-info 2 / algorithm 3；truly unclassified = 0)
>
> 318  benchmark 源映射到的公开 op 名（= 上述 314 + 4 个虽被对拍但从 perf 分母剔除的
>      BitCast / Set / Zero / ZeroExtendVector）。旧“303”是第一轮重构后再认证快照；318 不是单个 executable 的行数，也不代表
>      每个 overload/类型/参数形状都已覆盖。
> ```
> 之前对话里的 16/30/41/245/8 是不同时刻/口径的快照,**不可相加**;唯一权威数是上面这棵树
> (或当下重跑 `python scripts/simd-op-coverage.py`)。

> **算法层/composite(非 op-vs-Highway 矩阵)**:`softmax`/`log_softmax` 等组合核不在本全量表
> (Highway 无对等公开 API)。P0-1 单次 exp 融合 multipass 的性能与弯路见下「算法层备忘」+
> [`simd-complexity-opportunities.md`](./simd-complexity-opportunities.md) P0-1。

> 本表由重建后的测量引擎产出(`bench/simd/op_parity_engine.h` + `scripts/op-parity-report.py`),
> 取代旧的"原始读数+人工伪影标签"体系。旧 op_parity 的数字被证明携带三层测量伪影(P/E 调度、
> 代码放置运气、闭包栈 4K 别名,合计可达 ±30~40%),单条 `vpsubd` 的 `Neg i32` 曾被它测成
> 0.75~1.49 之间的任意值。完整证据链与重建过程:[`../debug/op-parity-harness-artifacts-hunt.md`](../debug/op-parity-harness-artifacts-hunt.md)。

## 方法论(数字为何可信)
- **测量**:x86 钉一颗 P 核+高优先级，metadata 审计 affinity/实际 CPU/core class；macOS 无亲和
  API，要求 USER_INTERACTIVE QoS + 8× epoch。每 op 每侧 4 个独立实例化的代码副本(采样放置运气,防 ICF 折叠,
  且带确定性相位错位 stagger —— v2 修复:同侧副本若以 64 倍数步长排布会共享同一相位,
  使 best-of-replicas 偏向另一侧达 ~10%,同文本对拍曾读 0.898,修后读 1.000);
  H/G 交替轮次;kernel 指针外提(热循环零闭包访问);宏形状循环 4x 展开。
- **聚合**:每文件 anchor 归一；完整 run 是 bootstrap cluster，簇内 replica 是相关 placement 样本；
  判决线随该 op 实测副本散布自适应(`place_u>=3%`)。每平台喂双序(HYLE 先/Highway 先)。
- **校准闸门**:5 个源码级验证"两侧同指令"的锚点 op(真值=1.00)必须全部落在闸门内,否则整表拒绝输出。
  本表锚点读数:MSVC/WSL 0.99~1.03(闸门 ±4%),Mac 0.95~1.05(闸门 ±6%,无亲和性 API,列分辨率较低,
  已用 4 次运行合并 + 8x epoch 补偿)。
- **判读**:数字 = HYLE/Highway 配对中位比值(>1=HYLE 快)；只有整个 CI 落进 0.97–1.03 才叫
  TIE；CI/双序/place_u 未能证明输赢或等价时为 INCONCLUSIVE。Mac 当前已有 parity/full/widen 三个
  NEON target；不再用“只有 full”掩盖覆盖空洞。

## 真输清单(校准仪器判决,按比值升序)
### MSVC(AVX2): 0 个
- (`Mod i32 invariant` 的旧 0.931 不是可接受地板：修正逐-vector 外部 sink、对齐 `Divisor` byte
  layout，并让 x86 以 scalar invariant sign 选择 q/Neg(q) 后，当前 full 为 **1.147 WIN**，真实
  load→Mod→store 为 **1.104 WIN**；完整误诊/汇编链见猎人日志。)
- (BroadcastLane<3> 最终轮清零:0.93→**1.00 TIE**。旧"形状红利改不动"定论收回——可折叠与否取决于
  确切 intrinsic 写法:MSVC 对 Highway 的 `inserti128(v, low128(v), 1)` 自 dup 形态有折叠 peephole
  (load 输入→内存源 vbroadcasti128,shuffle 口免费),而 `broadcastsi128(cast)` 写法栈 spill——旧实验
  的 0.47 由此而来,当时误判为"该序列本质需要内存源"。已按编译器分叉:MSVC=inserti128 自 dup+vpshufd,
  GCC 保留 vpermd(1.13W);Slide1Up 金丝雀 1.00,无 ConcatLowerLower 类复合上下文回归。)

### WSL/GCC(AVX2): 0 个
- (FindLastTrue 最终轮清零:0.94→**1.00 TIE**。旧"仪器残余"定论收回——规范化逐条 diff 发现两侧循环
  14 vs 15 条:HYLE 多一条 movslq,因 `31 - std::countl_zero(b)` 的减法在 int 域做、进 64 位 sink 前
  需符号扩展;Highway 的 Num0Bits* 为 size_t 宽全程 64 位。改 ptrdiff_t 域减法(avx2+sse4)即消。
  方法教训:此前"objdump 逐指令等价"是**向量签名级**对比的假象——标量尾巴(寄存器宽度/扩展指令)
  没有逐条比,一条指令恰好就是那 6%。)

### Mac/NEON: 0 个（首次全 perf 覆盖暴露的欠账已清零）
>
> **⚠️ "漏了" vs "无指令"**：能进这张表 = 有 Highway NEON 对手可对拍、跑出了比值，
> 即两侧都在 NEON 上真跑,HYLE 只是走了 store+reload/标量而 Highway 向量化。**不是** "NEON 没这条指令"
> (那种根本没有 Highway NEON 侧、不会出现在对拍里)。当前 `·` 的 12 个 case 均为单 block NEON
> 不适用的 x86 multi-block 形状；九报告 union 仍为 303/303。
> **背景**:此前 NEON 只有 op_parity_full_bench 一个 target,且它自 P4 起加的 op 从没在 NEON 重跑;
> op_parity_bench_neon / widen_neon 到本会话才建(见 reference_simd_coverage_tool 记忆的"NEON 覆盖盲区")。
> 首次三 TU 全跑,暴露一批 HYLE NEON 走 store+reload/scalar-fallback 而 Highway 用 native intrinsic
> 的 op。它们是老 ARM 欠账；本轮按 Highway 源码/汇编形状全部收口。

| 最后收口项 | 历史 → 当前 | 修复 |
|---|---|---|
| ScatterIndexN i32 | 0.877 half / 0.916 full → **0.999 / 1.000 / 1.003 TIE**（full/one/half） | 固定四次循环 + `i<max` 谓词，使 AppleClang 与 Highway 同样 scalarize 为 `ldp/ldpsw/str` |
| Mod i32 invariant store | 0.932 LOSE → **1.087 INCONCLUSIVE**（Div 1.001、Mod 1.011） | scalar-lane `smull/smull2` + NEON vector sign；x86 继续用 scalar sign |

- **SatWidenMulAccumFixedPoint i16→i32 0.92 → 0.98 TIE(2026-07-15,C20)**:补 NEON 原生
  `SatWidenMulAccumFixedPointNative`(单条 `vqdmlal_s16`,对齐 Highway arm_neon-inl.h:7467),vec.h
  按宏门控路由。判定:双序 6 遍合并 best 0.982/med 0.988,同机同构建兄弟校准 PASS;276 gtest 绿。
  诚实注记:修复提交同时把 bench 内核改对称(旧 HYLE 侧步长 w16、store 数减半,与 HWY 侧形状不对称)
  ——历史 0.92 里 op 慢与内核形状各占多少已不可分,当前对称内核下为 TIE 是本判定的全部主张。

- **PromoteMaskTo i16→i32 0.575 → 1.00 TIE(2026-07-15,simd-19 表示层根治,轮⑩)**:根因是定宽后端
  表示层丢 N,已由 simd-19 D 档表示修复——证据链见 `simd-19-ledger.md` C12(单 op 双序 6 遍 99.5–100.0%
  + 反汇编 `ldr d;sshll.4s;cmgt.4s` 与 Highway 同形)与 C18(全量复合环境 1.000/1.000)。

- (第十四轮清零:MulAddSub/MulSubAdd 弃两 FMA 的 OddEven(MulAdd,MulSub) 改 Highway 单 FMA 形,见演进段。
  此前"NEON 0 输"是 NEON perf 覆盖不全的假象——只测过 full_bench 一部分。)
- **NEON 修复轮①(2026-07-15):AESKeyGenAssist 0.12→1.00 TIE**。弃 scalar software core(memcpy 到栈
  数组 + aes_keygen_assist_soft 逐字节),改 Highway arm_neon-inl.h 的向量合成:`DupOdd(u32)`(vtrn2q_u32
  广播两个 SubWord 源字)→`AESLastRound(kRconXorMask)`(SubBytes+ShiftRows+逐 lane rcon xor)→
  `TableLookupBytes(kRotWordShuffle)`(vqtbl1q 修正 ARM ShiftRows-先于-SubBytes 的字节序,落到
  _mm_aeskeygenassist 语义)。三原语都是 HYLE 自家已验 x86-bit-identical 的 NEON op,合成精确。仅
  `__ARM_FEATURE_AES` 路径改(需 vaese);无 AES 硬件时整个 crypto 家族本就软实现,保留 byte core。
  CryptoAES gtest 逐字节对 oracle(rcon=0x01)PASS。
- **NEON 修复轮②(2026-07-15):PromoteEvenTo/PromoteOddTo 整数 2× 加宽 0.46/0.48→TIE**。旧 vuzp+vmovl
  被编译器融成 `ld2.Nb` 结构化去交织 load(Apple 核 ~2× 代价);改 Highway inside-inl.h 的**寄存器内移位
  idiom**:把窄向量 reinterpret 成宽 lane 型(每宽 lane = [奇:偶],小端偶在低半)。EvenTo signed =
  `vshlq_n<w>` 把偶半移到高位再 `vshrq_n_s<w>`(算术右移符号扩展);OddTo signed = `vshrq_n_s<w>` 单条;
  unsigned even = `vandq` 低半掩码,odd = `vshrq_n_u<w>`。落 `shl+sshr`(2 条,普通 load)替 `ld2`。lane
  映射与旧式逐字节相同(dst[i]=src[2i]/2i+1),6 组宽度×符号(i8/u8→i16,i16/u16→i32,i32/u32→i64)
  standalone 对拍标量 bad=0 + 220 NEON gtest 全绿。
- **NEON 修复轮③(2026-07-15):TableLookupLanes 0.32→1.04 / TwoTablesLookupLanes 0.21→1.02 TIE**。两者旧
  实现 store+reload 标量 gather;改 Highway arm_neon-inl.h 的 byte-index 展开:把 lane 索引经
  BroadcastLaneBytes(vqtbl1q 广播每 lane 的索引字节)+ `<<log2(sizeof)` 缩放 + ByteOffsets({0..B-1})
  展成字节索引,单表走 vqtbl1q、双表走 vqtbl2q(32 字节 {a,b} 表)。整数 lane 型走此路;float/double 索引
  以浮点值存(位不 pun 成字节索引)保留 store+reload。契约 idx∈[0,N)(同 Highway);越界得 0(旧 &(N-1)
  回绕是防御性、非契约,取消)。CheckSwizzle<T> reverse-permute gtest 逐值 PASS(i8/i16/i32/i64)+ 220
  NEON gtest 全绿。
- **NEON 修复轮④(2026-07-15,改契约):MulFixedPoint15 0.70→1.00 TIE**。旧 NEON 为保 x86 `_mm_mulhrs`
  的 MIN×MIN 回绕语义,加 5 个向量 op 修正(disasm:movi+2 cmeq+mvn+orn+bsl 包 vqrdmulh)。用户拍板
  **契约放宽**:MIN×MIN(两 -1.0→积 +1.0 不可表示)为 impl-defined——x86/scalar 回绕到 MIN、NEON
  vqrdmulh 饱和到 MAX(+0.99997,数值上更接近真值),调用方不得依赖(scalar.h 契约注 + neon 注 + gtest
  lane0 放宽接受两者)。NEON 落 bare `vqrdmulhq_s16` 单条=TIE。220 NEON gtest 全绿。
- **NEON 修复轮⑤(2026-07-15,widen 乘加家族):SumOfMulQuadAccumulate 0.89→TIE、SatWidenMulPairwiseAdd
  0.575→TIE**。① SumOfMulQuadAccumulate:旧 vmull_s8 lo/hi + 双 vpaddl fold(5 op),而 Apple Silicon
  有 dotprod 扩展——Highway 用单条 `vdotq_s32(sum,a,b)`。加 `#if __ARM_FEATURE_DOTPROD` 走 vdotq(VNNI
  等价单指令),无 dotprod 退回 vmull fold。② SatWidenMulPairwiseAdd:旧 store+reload 标量循环,改 Highway
  向量合成(generic_ops-inl.h):even/odd 字节各自 widen 到 i16(a 零扩展 vand/vshr、b 符扩展 vshl+vshr),
  vmulq_s16 各自乘,vqaddq_s16 饱和加邻对。单积 i8*u8∈[-32640,32385] 不溢 i16,只和饱和=maddubs 语义。
  两者专用 gtest(含饱和例)+ 220 NEON gtest 全绿。
- **NEON 修复轮⑥(2026-07-15):DemoteToNearestInt f64→i32 0.57→TIE(契约不变)**。两处:①DemoteInRangeTo
  f64→i32 弃 store+reload 标量 cast,改 `vcvtq_s64_f64`(截断)+`vmovn_s64`(窄化)在寄存器内做;②新增
  NEON native DemoteToNearestIntNative(HYLE_SIMD_NEON_HAS_NATIVE_DEMOTE_NEAREST_INT 门控,vec.h 接入):
  用 IEEE minNum/maxNum 的 `vminnmq_f64`/`vmaxnmq_f64`——它们**抑制 NaN**(返回非 NaN 操作数),故
  `vmaxnmq(vminnmq(v,MAX),MIN)` 一举把 NaN 与 +溢出映射到 MAX、-溢出到 MIN,结果 finite in-range,
  再 `vcvtnq_s64_f64`(RNE)+vmovn 精确窄化。**NaN→MAX 契约不变**(与 x86 min_pd+cvt 同法),两条抑 NaN
  clamp 替原泛型的 IfThenElse(Lt)/IfThenElse(Le) cmp+bsl 对。DemoteToNearestIntFamily gtest(NaN/溢出/
  InRange==safe)全过 + 220 全绿;full 0.57→0.99、InRange fast 也归 1.00。
- **Per4LaneBlockShuffle<3,2,1,0> MSVC 0.88→1.00 TIE(2026-07-15,用户直觉揪出)**。此格曾判 UNSTABLE
  (best 0.88 稳定但 spread 1.17 淹过闸门),用户质疑"0.87 判平很奇怪"。**looped bench-equivalent dumpbin
  坐实是真输不是噪声**:`<3,2,1,0>` 是**恒等置换**(0xE4=_MM_SHUFFLE(3,2,1,0)),Highway 有 SizeTag<0xE4>
  特化直接 `return v`(只剩 load+store),HYLE 却照发 `vpshufd [mem],0xE4` 一条冗余恒等 shuffle。修:五后端
  (avx2/sse4/avx512/neon/scalar)Per4LaneBlockShuffle 顶部加 `if constexpr(i3==3&&i2==2&&i1==1&&i0==0)
  return v;`。MSVC 0.88→1.00;gtest 覆盖恒等(<3,2,1,0>==输入)+ 非恒等(<2,3,0,1>)305 GCC 全绿。
  **同轮验证 MulOdd i32-i64:WSL 0.91→复测 best 1.04(med 1.107,spread 大)= 实为 TIE/偏赢,旧 0.91= 是噪声;
  MSVC 0.997 TIE。非真输,矩阵不变。教训:单元格 `?`(UNSTABLE)或大 spread 的低 best 值必 looped-dump 核实,
  别当噪声放过——恒等/常量特例是 Highway 常见而自造易漏的一类。**
- **NEON 修复轮⑦(2026-07-15):LoadMaskBits 向量化 → Dup128MaskFromMaskBits 0.72→TIE**。LoadMaskBits 旧
  实现为标量循环写栈数组 + vld1q(store+reload);改 Highway arm_neon-inl.h 的向量 bit-test:`vdupq` 广播
  bits → 每 lane AND 位选择子 `{1,2,4,...}` → `vtstq`(test-nonzero)得 all-ones/0 掩码。8-bit 用 vqtbl1q
  把低/高字节各复制到 8 字节半区再按 {1..128} 测(同 Highway kRep8+kBit)。Dup128MaskFromMaskBits(NEON 单
  block 即 LoadMaskBits)0.72→1.00;220 NEON gtest 全绿。**PromoteMaskTo 未随之修好(仍 0.565)——它的瓶颈
  不在 LoadMaskBits 而在 Rebind 部分标签(4-lane i16)仍物化整 q 寄存器(disasm:cmgt.8h→xtn→zip1→ushll→
  shl→cmlt 一串),Highway 用 `d`(半寄存器 4-lane)load 后 sshll 全程 4-wide。这是 HYLE 部分标签 codegen
  的更深问题,单 op 改不动,登记待后续标签系统改进。**
- **NEON 修复轮⑧(2026-07-15,改契约):BlendedStore 0.18→TIE**。NEON 无 masked-store 指令,旧实现标量
  逐 lane 写(store+reload、fault-safe);改 Highway load-blend-store:`StoreU(IfThenElse(m,v,LoadU(p)),p)`。
  **契约变更(用户授权)**:NEON BlendedStore 现**读全 kN lane 的目标内存**(写回不变故无逻辑副作用),调用方
  须保证该区间可读——与 Highway NEON 同契约;x86(vpmaskmovd,真 fault-safe 且快)不变,故 fault-safety 成
  NEON-only 放宽。7 个 MaskedTail gtest(float/double/i8/i16/i32/i64/u8,k=0..n 全扫)全过。0.18→1.00 TIE。
- **NEON 修复轮⑨(2026-07-15):SumsOfAdjQuadAbsDiff u8→u16 0.24→TIE**。旧 store+reload 三重标量循环(SAD
  video quad)。改 Highway generic_ops-inl.h 向量合成:InterleaveLower/Upper(a,a) 复制每字节 →
  CombineShiftRightBytes<kA*8+1 / +5> 取滑窗对 a01/a23;Broadcast<kB*2 / +1>(u16 视图)取 b01/b23;
  SumsOf2(AbsDiff(a01,b01)) + SumsOf2(AbsDiff(a23,b23))。所有 op 都是 HYLE 自家 NEON。standalone 穷举对拍
  标量 oracle:kAOffset{0,1}×kBOffset{0..3}×8 seed bad=0;220 NEON gtest 全绿。0.24→1.00 TIE。
  (第一轮:ReverseLaneBytes 0.05->TIE、Broadcast<0>、TruncateTo、SumOfLanes、Min/MaxOfLanes;
  第二轮:TZC 三平台追平、HSBI、DemoteMaskTo 0.33->1.4W、LoadN/StoreN 与 LoadNOr 弃 vpmaskmovd
  ->WIN、CombineMasks、ConcatLowerLower)+ v2 stagger 仪器修相位相关后,收敛为上表数字。
- **NEON 修复轮⑩(2026-07-15,simd-19 表示层根治):PromoteMaskTo 0.575→1.00 TIE**。不是单 op 修复——
  定宽后端表示层补上 N 维:`RawNeonNT<T,N>`(`N*sizeof(T)<=8` → 真 D 寄存器类型)+ **常绿垫片迁移**
  (`to_q/from_q`;W0a 并行 traits → W0b 脚本包裹 650+ 处 raw 访问 → W0c 一行翻转,每步编译绿 + gtest 绿)
  + ~17 个热点 op 摘垫片换 native-D(Set/Load/Store、比较族、mask 桥、Promote/Demote、half/Combine、
  Concat)。终态反汇编:`ldr d ; sshll.4s ; cmgt.4s` 与 Highway 同形,7 指令掰回链消失。顺带清偿两个
  **既有** bug 家族(均由测试先行卡挖出,非 s19 引入):①partial LoadU/StoreU 满 16 字节读写(OOB,
  护栏测试 46 案全绿);②子满宽 UpperHalf/ZeroExtendVector/InsertIntoUpper 的 N-盲语义(oracle 13 案
  DISABLED→全绿)。全量复跑(三 TU × 双序 × 3,校准全 PASS)LOSE:parity 0/69、full 0/206、widen 1/28
  (唯一 LOSE = 上表 SatWidenMulAccumFixedPoint,既有 `0.94?` 边界项钉实,非回归)。neon.h 3297→4002 行
  (+21%,为 Highway arm_neon-inl.h 10982 行的 36%,未越"不 vendor"警戒线)。施工采用 AI 分派工单制
  (19 卡 + 铁律 + 台账 + 批复),全记录:`simd-19-ledger.md` / `simd-19-workorders.md`。
- v1 的假 LOSE 被 v2 消除的例子:TestBit / RebindMask / IsNegative(GCC)——同侧副本相位相关
  曾让它们稳定读 0.83~0.96,stagger 后回到 TIE/UNSTABLE。
- 第三轮(2026-07-14):RoundingShr 改 4-op bias=(1<<s)>>1(负 s 收窄 unspecified)MSVC 1.06W/GCC
  0.95->1.15W;NEON RoundingShiftRightSame 转发 vrshl(0.82->见上表);TableLookupBytesInRange 落地
  五后端+sve/rvv(InRange 家族惯例),对拍改为契约对等的 InRange-vs-Highway。
- DemoteToNearestInt 语义决策落地(2026-07-14):默认版弃 emu128 sign-based NaN(Highway 自己
  在任何硬件 target 都不遵守的参考语义,2.5x 代价无消费者),改为与 Highway x86 native 逐字节一致
  (Min+cvt;NaN 双符号->MAX、±溢出饱和,仍是全后端统一契约)。MSVC 0.61->1.00 / GCC 0.44->1.01 TIE;
  InRange 快路径不变(命名即 Highway 官方词汇)。7 个正确性闸门全绿(含 SDE avx512)。
- 本轮回归教训:ConcatLowerLower 的 vinsert 形态 GCC 赢但 MSVC 在复合上下文(Slide1Up)编成
  extract+栈 spill 掉到 0.099(单独 kernel 却 TIE)——复合上下文回归只有全表重测才抓得住,
  已按编译器分叉(MSVC=vperm2i128 / GCC=vinsert)修复,Slide1Up/SlideMask1Up 复位 TIE/WIN。
- 第四轮(2026-07-14,commit 7c856fa):CompressNot i32(曾唯一 x86 双编译器交集,MSVC 0.81/GCC 0.94)
  弃泛型 `Compress(v,Not(m))` 的向量 Not,改 Highway 的 bits 域补码(`mbits^0xFF` 复用同一 compress
  索引表,即 Highway IndicesFromNotBits);dumpbin 坐实 = Compress + 单条 `not eax`。GCC→TIE、
  MSVC→0.95(追到与 Compress 本身同速)。**x86 双编译器交集清零。**
- 第四轮续(commit 647d254):CompressStore i32 消重复 movemask —— 原 `StoreU(Compress(v,m))+CountTrue(m)`
  对同一 mask 双提取,改提一次 mbits 共享 compress+count(Highway 同款单提取)。MSVC 0.91→0.97 UNSTABLE、
  GCC TIE。
- **Compress 三兄弟收口定论(dumpbin 铁证)**:可修部分(向量 Not、重复 movemask)全消后,三者共享的
  `compress32_by_bits` 核在 MSVC 上**比 Highway 还干净**——同款 3 条(vpbroadcastd[表+idx*4] + vpsrlvd
  ymm,[shifts] + vpermd),HYLE 无 Highway 的 HWY_DASSERT cmp/jae→Abort 分支、无栈帧,shifts 已折成
  .rdata 内存操作数。残余 0.93~0.96 = bench 调用形状的一条 v-load + 绝对地址/布局级仪器抖动,非核可控、
  非算法输。**Compress/CompressNot/CompressStore 全部移出真输清单,此线到头不再追。**
- 第五轮(2026-07-14,commit 3f1b842):AddSub i32 改 Add(a,OddEven(b,Neg(b)))(avx2+sse4 整数路径,
  Highway 形态)。旧式 OddEven(Add(a,b),Sub(a,b)) 两算术都依赖 a、循环不变 b 无法外提——链式微基准
  GCC 旧式 164ns/新式 6.87ns(反超 Highway 13ns),24x 隐患。GCC op_parity 0.94→TIE。MSVC 仍 0.84 但
  链式(b 在寄存器)三者持平 17ns、dumpbin HYLE 4 条<Highway 5 条,判定 bench 调用形状特定(b 内存双读)
  非算法输,移出清单。**方法学:单编译器输 + 另一编译器 TIE/反赢时,先查链式(b 在寄存器)与 dumpbin——
  区分"真依赖链隐患(GCC 旧式 24x)"vs"bench 形状伪影(MSVC 内存操作数)",两者可同时存在于一个 op。**
- 第六轮(2026-07-14,commit abbd7db):avx2 IfNegativeThenElse 直用符号位喂 blendv 省 vpcmpgtd。
  原 IfThenElse(Lt(v,0),..) 先 vpcmpgtd 造全1掩码再 blendv;但 blendv 本按 lane MSB 选择,"v<0" 就是
  符号位,直接把 v 当 blend 控制即可(Highway 形态)。用 lane 宽 blend(vblendvps/pd)防 per-byte 误选。
  **广影响修复**——所有 IfNegativeThenElse 消费者受益。SatWidenMulPairwiseAccumulate MSVC 0.93→TIE
  (dumpbin 消 vpxor+vpcmpgtd 后与 Highway 逐指令一致),GCC 保持 TIE。MSVC 真输 10→9。
- 第七轮(2026-07-14,commit 3110d9d):RebindMask 改 MaskFromNormalizedVec 去冗余 srai。
  RebindMask=MaskFromVec(BitCast(VecFromMask)) 里 MaskFromVec 重跑符号广播(avx2 一条 vpsrad);
  VecFromMask→BitCast 已 canonical,改 MaskFromNormalizedVec 纯 re-tag(Highway avx2 RebindMask
  本就是 no-op)。补全 MaskFromNormalizedVec 到 sve/rvv(转发 MaskFromVec)使 7 后端齐备。MSVC
  0.92→TIE、GCC 保持 TIE。SVE(qemu VL=256)+RVV(qemu)RebindMask 探针 bad=0。MSVC 真输 9→8。
  (同族去冗余:DemoteMaskTo/CombineMasks/LowerHalfOfMask/RebindMask 均 MaskFromVec→MaskFromNormalizedVec。)
- 第八轮(2026-07-14,commit a5410df):SetBeforeFirst 32-bit 编译器分叉。=Not(SetAtOrAfterFirst)(补集),
  SetAtOrAfterFirst 32-bit 已纯向量。**但两编译器最优形态相反**:MSVC 向量域消 mask↔GPR↔LoadMaskBits
  往返→0.89→TIE;GCC scalar-bits(blsmsk 式)本就 1.30x WIN,向量域反退 0.91。故 #if _MSC_VER 分叉
  (同 ConcatLowerLower/第三轮 AddSub 印证的"GCC 反赢陷阱"处置法:分叉而非统一)。全 256 i32 模式验 bad=0。
  MSVC 真输 8→7。**方法学:GCC 反赢的 op 不是"不能改",而是"分叉改"——MSVC 追平不牺牲 GCC 反赢。**
- 第九轮(2026-07-14,commit 87d72ba):SetAtOrBeforeFirst 同法编译器分叉。=Not(SetAtOrAfterFirst(mask
  上滑 1 lane)),用 Slide1Up(全 256 上滑,跨 128-block 正确;per-block ShiftLeftLanes<1> 边界会错已验)。
  MSVC 向量域 0.90→0.99 TIE、GCC 保 scalar-bits 1.16 WIN。全 256 i32 模式 bad=0。MSVC 真输 7→6。
  至此 Set*First 族(SetBeforeFirst/SetAtOrBeforeFirst)MSVC 全追平、GCC 全保反赢。
- 第十轮(2026-07-14,commit 5e413c6):Sum/Min/MaxOfLanes 共享的 reduce256_broadcast 编译器分叉。
  HYLE 128-domain(extracti128→xmm reduce→inserti128 broadcast)GCC 反赢 1.17~1.19 但 MSVC 差(0.91~0.96)。
  加全-256 HWY shape(op(v,SwapAdjacentBlocks(v))+两轮 256 shuffle,每 lane 已得结果无需 broadcast)。
  #if _MSC_VER:MSVC 全-256 / GCC 保 128-domain。MSVC Sum 0.91→1.05 TIE、Max 0.95→1.13 WIN、Min 0.96→1.13 WIN;
  GCC 三个保持 1.17~1.19 WIN。一个共享核分叉,三个 op 全解决。MSVC 真输 6→4。
- 第十一轮(2026-07-14,commit 2fa1c2e→再分叉):RoundingShiftRight<k> 两种 bit-identical 形态编译器分叉。
  bias 常量式(v+2^(k-1))>>k 与自减形 sv=v>>(k-1);sv-(sv>>1)(Highway 形,消 vpbroadcastd+vpaddd)穷举
  证 bit-identical(4.3e9,diff=0,非 runtime RoundingShr 的契约差)。**GCC 把 bias 式编成 1.12W(外提
  bias splat),MSVC 偏好自减形(bias 式 0.91)**——先统一改自减(GCC 1.12W→TIE,丢 margin),后按"GCC
  反赢陷阱"分叉法复位(#if _MSC_VER:MSVC 自减 / GCC 保 bias)。RoundingShiftRightAndDemoteTo<3>
  MSVC 0.91→0.97 TIE、GCC 恢复 1.11 WIN。MSVC 真输 4→3。
- 第十二轮(2026-07-14):IsInf f32 整数/FP 域编译器分叉。HYLE 原 Eq(Abs(v),inf)(FP:vandps+vcmpeqps)
  GCC 反赢 1.13;Highway 泛型 AVX2 用整数域 Eq(vu+vu, 0xFF000000)(左移 1 清符号位、比 exp 全 1 尾 0
  = ±inf,x86_256 native 的 _mm256_fpclass 仅 AVX-512 故 AVX2 落泛型)。**又一"GCC 反赢陷阱"**:MSVC FP 式
  0.93、整数式追平;#if _MSC_VER:MSVC 整数域(u 掩码 castsi256_ps/pd 复回 float 掩码域)/ GCC 保 FP 域
  1.13W。gtest IsInf/Finite/NaN/Round 三平台(avx2/sse4/scalar)+ SDE avx512 全绿。MSVC 真输 3→2。
- 第十三轮(2026-07-14):Mod i32 invariant——Div 的 dsign 广播下沉。/FAs 反汇编(必须带 -DHYLE_SIMD_TARGET_AVX2,
  否则 ScalableTag 退化 128-bit vpmullq 路径,是错配陷阱)证 HYLE Fig5.1 与 Highway IntDivS **逐指令同算法**
  (HYLE 22 向量 ALU op < Highway 23);唯一差=HYLE 把 dsign_vec 在 signed 分支顶部就 Set,非 pow2 路径要到
  最后两 op 才用,活跃区跨越寄存器紧张的 MulHigh32→MSVC 把它塞进 callee-saved ymm6(sub rsp,40+xmm6 spill
  +3 reg 拷贝)。把 dsign 下沉到 MulHigh32 之后(只在 Xor/Sub 用),活跃区收窄→spill 消失(rsp 40→24,与
  Highway 同)。MSVC 0.93→0.97 TIE;GCC 0.99→1.00(本就 TIE,略升);姊妹 MaskedModOr 复合上下文 stack op
  6→4(共享同核)一并改善。gtest Div/Mod/Divisor 双平台 5 pass。纯 regalloc(非算法/非缺 Highway trick,
  MaskedModOr TIE + GCC TIE 双证)。MSVC 真输 2→1。**至此 MSVC 真输仅余 BroadcastLane<3>(bench 形状定论)。**
- 第十四轮(2026-07-14,Apple Silicon 真机):MulAddSub/MulSubAdd f32 NEON——弃两 FMA 改 Highway 单 FMA 形。
  HYLE 泛型层原 `OddEven(MulAdd(m,x,c), MulSub(m,x,c))` 发**两条 FMA**再选;Highway generic 先把加数在做减法
  的 lane 预取反 `OddEven(c, Neg(c))` 再单条 `MulAdd(m,x,·)`=**一条 FMA**(取反的是循环不变 c 可外提)。
  x86 AVX2 有 native vfmaddsub/vfmsubadd 覆盖泛型故不受影响,本改只动 neon/sve/rvv/scalar/sse4/avx512
  (均无 fused add-sub 指令)。符号约定逐 lane 核对不变(MulAddSub 偶=减/MulSubAdd 偶=加)。真机 NEON gtest
  220 pass + MSVC avx2/sse4/scalar MulAdd 族全 pass。校准仪器 Mac 3+3 run 合并 --gate 0.06 **CALIBRATION PASS**:
  MulAddSub 0.88→0.97 TIE、MulSubAdd 0.87→1.01 TIE。**Mac/NEON 真输清零。三平台真输终态 MSVC 1 / GCC 1 / NEON 0,
  x86 双编译器交集 0。**
- 最终轮(2026-07-14):用户对仅剩两个"定论地板"提出复审,**两个定论均被推翻并修成 TIE**——
  ① FindLastTrue(GCC 0.94→1.00):真凶=一条 movslq(int 域减法的符号扩展),非仪器残余;
  ② BroadcastLane<3>(MSVC 0.93→1.00):真凶=intrinsic 写法决定 MSVC 折叠与否,非"形状红利不可得"。
  复审教训:被"定论"关闭的项,证据等级要过硬——签名级 disasm 对比、单一形态的否定实验,都不足以
  盖棺;逐条 diff 与多形态尝试才算。
- 边缘带坐实轮(2026-07-14,锁核 3-run 双序):用户要求对 `?` 徘徊者按方法论多轮复测,**分清真 UNSTABLE
  vs 单轮误判的真输**。3-run 合并(HYLE_BENCH_CPU=2 锁核)结论:IsNegative(MSVC)/Compress/CompressNot/
  OrderedDemote2MasksTo 确属 UNSTABLE(单轮 0.84~0.95 是相位噪声,多轮判决翻转);**但 AddSub(MSVC)与
  IsNegative(GCC)是被 `?` 掩盖的真输,双双 3-run 稳定 0.84、spread 小,修成 TIE**:
  - **AddSub i32 MSVC 0.84→TIE**:/FAs 反汇编——泛型 `Add(a,OddEven(b,-b))` 在 **GCC 上 HYLE/Highway 逐字节
    相同(故 GCC TIE)**,但 **MSVC 把 b 折成内存操作数读进 vpsubd+vpblendd 两次**(b 从 [mem] 读两遍),
    Highway 先 vmovdqu 到寄存器只读一次。avx2/sse4 i32/u32 路径改显式 intrinsic(setzero+sub+blend_epi32
    0x55+add)强制 b 落单寄存器→MSVC 逐字节追平 Highway;GCC 仍逐字节相同保持 TIE。
  - **IsNegative i32 GCC 0.84→TIE**:HYLE `Lt(v,0)`=vpxor 造零+vpcmpgtd 两 op,Highway=BroadcastSignBit
    单条 vpsrad$31。改 signed-int 走 `MaskFromNormalizedVec(BroadcastSignBit(v))`(canonical -1/0 免
    MaskFromVec 的冗余二次 srai——MSVC 一度双 vpsrad 就是漏了这个)→两编译器逐字节=Highway 单 vpsrad。
    **仅限 signed-int**:float 保 `Lt(v,0)`(单 vcmpltps;BitCast-to-signed 往返会拖累 IfNegativeThen* 的
    float 消费者,实测反退 0.94),unsigned 保 Lt(恒 false 正确)。
  - 验证:MSVC+GCC 各 avx2/sse4/scalar gtest 9 pass;3-run 双序 CALIBRATION PASS,两 op 两编译器全 TIE,
    全表 LOSE 扫描归零(MaskedModOr 0.965 gate-edge 复测消失)。**方法学:`?` 不是"免检"——多轮锁核双序
    才能分清真 UNSTABLE 与被噪声淹没的真输;单轮 0.84 若 3-run 稳定就是真输,必查 disasm。**
- 全量回归轮(2026-07-14):动了泛型层(IsNegative/AddSub)后跑全量——MSVC 221/220/305 + GCC 221/220/
  scalar + avx512 SDE 221 + Mac NEON 220 gtest 全绿;GCC/Mac 全表 LOSE 归零。MSVC 全量 5-run 又暴出
  **Compress i32 稳定 0.94(此前"dumpbin 已≥Highway 移出"的定论又被推翻)**:/FAs 逐条 diff 现 HYLE 比
  Highway **多一条 `movzx edx,al`**——`compress32_by_bits` 里 `kTbl[mbits & 0xFFu]` 的 `& 0xFF` 让 MSVC
  在 `vmovmskps eax` 后补零扩展 al→edx;而 movemask_ps 结果本就 0..255,`& 0xFF` 冗余(Highway 直接
  `vmovmskps edx` 索引)。删掉 `& 0xFF`→两编译器逐字节=Highway,MSVC 0.94→0.98 UNSTABLE、CompressStore
  0.99 TIE、CompressNot 0.95 UNSTABLE、GCC 不变(它本就用全宽寄存器)。**又一"签名级对比漏看标量尾巴"
  (同 FindLastTrue 的 movslq):`movzx`/`movslq` 这类 0 成本外观的标量补齐指令恰好就是那 6%。**
- SOTA 覆盖补齐轮(2026-07-14):覆盖率复审(见 `../architecture/simd-vendor-surface.md`(vendor 借用面) + 自校验
  `scripts/simd-op-coverage.py`)揪出 14 个"可对拍却没 bench"的 op(旧文档假称覆盖或分母漏项),按
  methodology 补 bench 后三平台跑校准,**测出并修掉 4 个真 SOTA 输**(commit 5510f0a):
  - **U8FromU32 MSVC 0.088(11×!)**:store+reload 标量循环→Highway vpshufb 字节 gather + 两 128-block quad OR。
  - **MulEvenAdd/MulOddAdd bf16 0.855/0.857**:Bf16Even/OddToF32 的 PromoteEven/OddTo 隔离 vpand 冗余
    (u16 对 reinterpret u32 后 <<16 天然丢奇半)→改直接 shift/and,去 2 vpand(disasm HYLE 2vpand+2vpslld
    vs Highway 裸 2vpslld)。喂 bf16 点积/GEMM。
  - **CopySignToAbs 0.910**:转发全 CopySign(AndNot+And+Or),abs 无符号位故 AndNot 死码→Highway OrAnd 2-op。
  三平台坐实:MSVC+GCC 各 avx2/sse4/scalar gtest + Mac NEON gtest 全绿;MSVC+GCC 校准 3 轮双序全 TIE、
  Mac NEON 5-run gate-0.06 全 TIE,无 LOSE。新 17 行(掩码 FMA/位三元/掩码 gather-scatter/bf16/U8/
  CopySignToAbs)已入全量表。
  **口径澄清(30 个"有 bench 未测")**:①17 个是 HYLE-vs-Highway 校准对拍→已测已填;②5 个 HYLE 独创
  无 Highway 对手(ReduceMul/And/Or 对拍标量 fold、ReciprocalNewton/SqrtNewton 对拍 Div/裸估计)=无"输
  给 SOTA"之说,只查明显问题;③8 个 AES×6/CLMul×2 有 Highway 但在 crypto_bench(非校准仪器),要证
  SOTA 需移进 op_parity 校准框架(登记待办)。
- SOTA 收尾轮(2026-07-15):①**IfVecThenElse i32 MSVC 0.94→TIE**(用户直觉命中)——`MaskFromVec` 的防御性
  sign-broadcast(vpsrad$31)对下游 vpblendv 冗余(只读 MSB),Highway 契约本就"非全0/全1 结果 implementation-
  defined"故不广播;改 `MaskFromNormalizedVec`(avx2/sse4 跳 srai=逐字节同 Highway,其他后端转发 MaskFromVec
  不变)。三平台 gtest(全位/零选择子=契约域)全绿。②**ReduceMul/And/Or 32-bit 弃 store+reload 改 halving
  tree**(op128_2+reduce256_2,同 reduce256 shape)——vs 旧标量 fold:MSVC Mul 1.46×/And·Or 1.43×,GCC Mul
  1.32×/And·Or 2.2×;gtest 8 三平台+SDE 全绿。**独创 op(无 Highway 对手)仍按"比 baseline 快"判,发现 store+
  reload 就树化。**
- 窄宽 reduce 树化轮(2026-07-15,#145 收口):**ReduceSum/Min/Max i16+i64 弃 store+reload 改 halving tree**。
  先补 op_parity 窄宽 bench(此前 i16/i64 reduce 零 bench)暴出大输:**i16 Sum/Min/Max 0.078~0.118(比
  Highway 慢 8~13×!)、i64 0.67~0.87**。原注释"Highway 也 fallback"是**错的**——Highway 走 SumOfLanes/
  MinOfLanes 向量树。加 op128_narrow(i16 原生 min/max_epi16/epu16;i64 无原生 min/max_epi64→cmpgt_epi64+
  blendv 合成,无符号先 xor 符号位偏置)+reduce256_narrow。i16/u16/i64/u64 走树,i8/u8 留 fold(sum 需 widen,
  收益低,有意留)。i16 树在 16-bit lane 内累加,wrap 与标量 accumulator 逐位一致。**结果 i16 0.08→1.02
  (~13×)、i64 0.7→1.00,三平台(MSVC+GCC avx2/sse4/scalar + avx512-SDE + Mac NEON)gtest 8 全绿、全 TIE。**
  **教训:代码里"Highway 也这样 fallback"的自辩注释必须核实——这次就是错的,Highway 明明树化了;零 bench
  的能力面是盲区高发地(窄宽 reduce 慢 13× 一直没人知道)。**
- crypto 移进校准框架轮(2026-07-15,#146):8 个 AES/CLMul 从 crypto_bench(裸 nb::run 非校准)移进
  op_parity_bench 校准框架拿可信判决。过程中做了两处真改进:①**per-block worker 从裸函数名改 stateless
  lambda**——函数名 decay 成函数指针,MSVC 在热循环里不内联该间接调用(AESKeyGenAssist 因显式拆 lo/hi
  一直是唯一 TIE=铁证);AESInvMixColumns(map1 unary)因此 0.072→TIE。②**aes256_map recombine 从
  `_mm256_set_m128i`(MSVC 展开成 FP 域 vinsertf128)改 castsi128+vinserti128 整数域**(匹配 Highway)。
  但 MSVC 二元 AES/CLMul 起初仍读 0.09/0.17。**我一度误判为 harness 伪影——错了(用户坚持"明显有问题,追查")。**
  **真凶靠"带循环"的反汇编现形**(之前的单次调用 dump 干净,但循环体才和 bench 对等):map1/map2 template+
  lambda 间接层让 MSVC 每迭代把两个 256-bit 输入**溢出到栈 scratch**($T1/$T2:vmovdqu+vextractf128 到栈再
  reload)+ /GS cookie 帧;手写内联的 2-block 序列则与 Highway 逐字节相同、零溢出(AESKeyGenAssist 一直显式
  拆 lo/hi 故一直 TIE=铁证)。**修:弃 aes256_map1/map2,每 op 直接拆 lo/hi 调 block worker(整数域
  vinserti128 recombine)。MSVC 8 个全 0.09/0.17→TIE(AES 10×/CLMul 6× 消除);GCC 全 TIE/WIN。** gtest 双平台过。
  **教训:稳定+双序一致+校准通过的读数别轻易判伪影;单次调用 dump 会骗人,要 dump 带循环的 bench 对等体;
  用户"明显有问题"的直觉压过我的过早结论。**

## 全量表（2026-07-16，simd-22 后 v4 再认证）

这里的 **303** 是 source benchmark 标题映射到的 canonical 公开 op 名；实际报告有 **320 个
benchmark case**（同一 op 的类型/参数/形状会占多行）。Windows/WSL 各观察 320 case；NEON
观察 308 case，缺的 12 个都是 128-bit 单 block 平台不适用的 x86 multi-block 形状，不是漏跑。
九份报告的 union 机械审计为 `303 source-benched / 303 observed / 0 absent`。

| 平台 | 编译器 / ISA | case | LOSE | WIN | TIE | INCONCLUSIVE |
|---|---|---:|---:|---:|---:|---:|
| Windows | MSVC 19.50 / AVX2 | 320 | 0 | 39 | 103 | 178 |
| WSL | GCC 13.3 / AVX2 | 320 | 0 | 41 | 38 | 241 |
| MAC | AppleClang 21 / NEON (M4) | 308 | 0 | 7 | 54 | 247 |

记号：`W=WIN`、`T=TIE`、`?=INCONCLUSIVE`、`L=LOSE`、`·=该 target 不适用`；数字为
HYLE/Highway paired ratio。INCONCLUSIVE 不是 TIE，也不计输赢。

### ratio `<0.98` 全扫

| 平台/TU | op | ratio | 95% CI | H/G spread | verdict | 检查结论 |
|---|---|---:|---|---|---|---|
| win/parity | ScatterIndexN i32 (one lane) | 0.956 | [0.547, 1.044] | 1.348/1.111 | INCONCLUSIVE | native 修复后目标形状约 0.997–1.003；本格为 placement CI |
| win/parity | CombineMasks i32 (4096) | 0.971 | [0.937, 1.013] | 1.136/1.095 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| win/parity | LeadingZeroCount u32 (4096) | 0.971 | [0.949, 0.986] | 1.112/1.055 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| win/parity | ScatterIndexN i32 (full n) | 0.978 | [0.874, 1.038] | 1.057/1.071 | INCONCLUSIVE | native 修复后目标形状约 0.997–1.003；本格为 placement CI |
| win/full | ConcatUpperUpper i32 | 0.918 | [0.830, 1.015] | 1.254/1.228 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| win/full | ConcatUpperLower i32 | 0.936 | [0.852, 1.038] | 1.282/1.259 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| win/full | Not i32 | 0.961 | [0.953, 1.000] | 1.025/1.127 | INCONCLUSIVE | 两侧逐条同指令/同一 intrinsic；code placement |
| win/full | IfThenElseZero i32 | 0.968 | [0.945, 0.975] | 1.046/1.024 | INCONCLUSIVE | 两侧逐条同指令/同一 intrinsic；code placement |
| win/full | IfThenZeroElse i32 | 0.968 | [0.952, 0.977] | 1.078/1.133 | INCONCLUSIVE | 两侧逐条同指令/同一 intrinsic；code placement |
| win/full | TruncateTo u32->u16 | 0.972 | [0.923, 1.033] | 1.186/1.142 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| win/full | SwapAdjacentBlocks i32 | 0.977 | [0.927, 1.031] | 1.141/1.202 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| win/widen | U8FromU32 u32->u8 | 0.964 | [0.933, 0.999] | 1.143/1.143 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| wsl/parity | TwoTablesLookupLanes i32 (4096) | 0.978 | [0.949, 0.992] | 1.043/1.060 | INCONCLUSIVE | 实现与形状已复核；CI/双序证据不足，保留 INCONCLUSIVE |
| wsl/parity | MaskedGatherIndex i32 (4096, 50%) | 0.979 | [0.962, 1.009] | 1.109/1.053 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| wsl/full | MulOdd i32-i64 | 0.927 | [0.824, 1.045] | 1.344/1.198 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| wsl/full | MaskedReduceSum i32 | 0.974 | [0.954, 0.990] | 1.071/1.044 | INCONCLUSIVE | 实现与形状已复核；CI/双序证据不足，保留 INCONCLUSIVE |
| wsl/full | Eq i32 | 0.979 | [0.955, 1.012] | 1.061/1.062 | INCONCLUSIVE | 两侧逐条同指令/同一 intrinsic；code placement |
| wsl/widen | MulOddAdd bf16->f32 | 0.975 | [0.944, 1.002] | 1.087/1.074 | INCONCLUSIVE | 实现与形状已复核；CI/双序证据不足，保留 INCONCLUSIVE |
| mac/parity | StoreInterleaved3 f32 (4096) | 0.960 | [0.822, 1.075] | 1.388/1.458 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/parity | StoreInterleaved2 f32 (4096) | 0.970 | [0.906, 1.077] | 1.481/1.273 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Zero i32 | 0.957 | [0.904, 1.092] | 1.290/1.188 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | And i32 | 0.961 | [0.942, 1.027] | 1.135/1.103 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Ge i32 | 0.961 | [0.943, 1.032] | 1.150/1.160 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Broadcast<0> i32 | 0.966 | [0.919, 1.032] | 1.183/1.231 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | MaxOfLanes i32 | 0.966 | [0.867, 1.019] | 1.192/1.182 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Ne128 u64 | 0.967 | [0.920, 1.020] | 1.102/1.156 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | RoundingShr i32 | 0.967 | [0.931, 1.036] | 1.123/1.154 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | ZeroIfNegative f32 | 0.967 | [0.898, 1.013] | 1.113/1.109 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Floor f32 | 0.968 | [0.909, 1.032] | 1.267/1.113 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | DupEven i32 | 0.969 | [0.919, 1.032] | 1.286/1.207 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Shuffle2301 i32 | 0.969 | [0.888, 1.087] | 1.291/1.180 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Abs i32 | 0.972 | [0.922, 1.050] | 1.149/1.351 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Mul f32 | 0.973 | [0.936, 1.046] | 1.219/1.151 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Shuffle0321 i32 | 0.974 | [0.924, 1.027] | 1.179/1.120 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | FindKnownLastTrue i32 | 0.976 | [0.929, 0.995] | 1.110/1.071 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | ShiftRight<3> i32 | 0.976 | [0.912, 1.048] | 1.230/1.245 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |
| mac/full | Mul i32 | 0.979 | [0.950, 1.029] | 1.107/1.123 | INCONCLUSIVE | 高 replica spread，CI 跨等价带；无不同算法/额外 fallback |

本轮真正修复而不再出现在低值表中的项：`SatWidenMulAccumFixedPoint` 的 width-banded
store/reload 回归、`CompressNot` 的额外 xor、MSVC/NEON `ScatterIndexN`、MSVC `Mod`/sink、
MSVC `SetOnlyFirst`、MSVC `IsFinite`，以及 NEON `Mod` 的 scalar-multiplier/sign codegen。

<details>
<summary>展开 320-case 三平台矩阵</summary>

| TU | op / shape | Windows AVX2 | WSL AVX2 | Mac NEON |
|---|---|---:|---:|---:|
| parity | Add f32 (4096) | 1.000 ? | 1.001 ? | 0.998 ? |
| parity | AESInvMixColumns u8 | 0.994 ? | 1.127 W | 0.999 T |
| parity | AESKeyGenAssist u8 | 1.032 ? | 0.999 T | 1.000 T |
| parity | AESLastRound u8 | 0.999 T | 1.017 ? | 0.998 T |
| parity | AESLastRoundInv u8 | 1.004 T | 1.008 ? | 0.999 T |
| parity | AESRound u8 | 1.000 ? | 1.021 ? | 0.999 T |
| parity | AESRoundInv u8 | 1.005 T | 1.018 ? | 1.000 T |
| parity | BitShuffle u64 (4096) | 1.066 W | 1.103 W | · |
| parity | BlendedStore i32 (4096, 50%) | 1.003 T | 1.006 ? | 1.000 ? |
| parity | CLMulLower u64 | 0.997 ? | 0.990 ? | 0.992 ? |
| parity | CLMulUpper u64 | 1.001 T | 0.999 T | 1.004 ? |
| parity | CombineMasks i32 (4096) | 0.971 ? | 0.998 T | 1.002 T |
| parity | Compress i32 (4096, 50% dense) | 0.991 T | 0.983 ? | 0.993 ? |
| parity | CompressBlendedStore i32 (4096, 50%) | 1.032 ? | 1.175 W | 5.392 W |
| parity | CompressNot i32 (4096, 50%) | 1.016 ? | 1.024 ? | 0.993 ? |
| parity | CompressStore i32 (4096, 50%) | 1.006 ? | 1.003 ? | 1.004 T |
| parity | ConcatEven i32 (4096) | 1.024 ? | 0.997 ? | 0.993 ? |
| parity | DemoteMaskTo i32->i16 (4096) | 0.998 ? | 1.397 W | 1.001 T |
| parity | DemoteTo i32->i16 1-arg (4096) | 1.003 ? | 1.001 ? | 1.000 ? |
| parity | DemoteToNearestInt f64->i32 (4096) | 1.006 ? | 0.995 ? | 0.995 T |
| parity | Dup128MaskFromMaskBits i32 (4096) | 1.054 ? | 1.178 W | 1.000 T |
| parity | Expand i32 (4096, 50%) | 1.989 W | 2.059 W | 2.216 W |
| parity | GatherIndex i32 (4096) | 0.999 ? | 0.988 ? | 1.002 T |
| parity | GatherIndexN i32 (half n) | 0.987 ? | 1.002 ? | 0.996 ? |
| parity | GatherIndexNOr i32 (half n) | 1.002 ? | 0.983 ? | 1.002 ? |
| parity | GatherOffset i32 (4096) | 1.000 T | 0.999 ? | 1.004 ? |
| parity | InterleaveLower i32 (4096) | 1.006 ? | 1.011 ? | 0.995 T |
| parity | LeadingZeroCount u32 (4096) | 0.971 ? | 0.987 ? | 0.995 ? |
| parity | LoadExpand i32 (4096, 50%) | 1.989 W | 2.070 W | 2.247 W |
| parity | LoadInterleaved2 f32 (4096) | 0.996 T | 0.997 ? | 0.999 T |
| parity | LoadInterleaved3 f32 (4096) | 1.000 T | 0.993 T | 0.985 ? |
| parity | LoadInterleaved4 u8 (4096, RGBA) | 1.013 ? | 0.992 ? | 0.998 T |
| parity | LoadN/StoreN i32 (half lanes) | 1.567 W | 1.757 W | 1.001 T |
| parity | LoadNOr i32 (half n) | 1.202 ? | 0.994 ? | 1.003 ? |
| parity | MaskedGatherIndex i32 (4096, 50%) | 1.004 ? | 0.979 ? | 1.014 ? |
| parity | MaskedGatherIndexOr i32 (4096, 50%) | 1.008 T | 1.002 ? | 1.019 ? |
| parity | MaskedLoad i32 (4096, 50%) | 0.992 ? | 1.005 ? | 1.004 ? |
| parity | MaskedLoadOr i32 (4096) | 1.006 ? | 1.003 ? | 0.995 ? |
| parity | MaskedScatterIndex i32 (4096, 50%) | 1.063 ? | 0.994 ? | 1.022 ? |
| parity | Min i32 (4096) | 1.000 ? | 1.009 T | 1.001 ? |
| parity | MulAdd f32 (4096) | 0.999 ? | 0.995 T | 1.003 ? |
| parity | MulHigh i16 (4096) | 0.991 ? | 0.995 ? | 0.999 T |
| parity | MulHigh u16 (4096) | 1.005 T | 0.997 ? | 0.998 T |
| parity | MulHigh32 u32 (4096) | 1.000 ? | 1.000 ? | 1.002 ? |
| parity | MultiRotateRight u64 (4096) | 1.000 ? | 0.994 T | · |
| parity | OrderedDemote2MasksTo i32->i16 (4096) | 0.984 ? | 1.006 ? | 0.999 T |
| parity | PopulationCount u32 (4096) | 1.490 W | 1.500 W | 1.009 ? |
| parity | PromoteLowerTo i16->i32 (4096) | 0.995 ? | 0.993 ? | 1.000 T |
| parity | PromoteMaskTo i16->i32 (4096) | 1.001 ? | 0.992 T | 0.997 ? |
| parity | PromoteTo i16->i32 1-arg (4096) | 0.995 ? | 0.992 ? | 1.000 T |
| parity | RebindMask i32->f32 (4096) | 1.012 ? | 0.986 ? | 0.997 ? |
| parity | ReduceSum f32 (4096) | 0.990 T | 0.999 T | 1.000 T |
| parity | Reverse i32 (4096) | 0.998 ? | 0.995 ? | 1.003 T |
| parity | Reverse8 i32 (4096) | 1.001 ? | 0.999 ? | · |
| parity | ReverseLaneBytes u32 (4096) | 1.033 ? | 0.987 ? | 1.004 T |
| parity | RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.987 ? | 1.083 W | 1.004 T |
| parity | SaturatedAdd i16 (4096) | 1.003 T | 0.995 ? | 1.001 T |
| parity | ScatterIndex i32 (4096) | 0.991 ? | 1.013 ? | 1.000 T |
| parity | ScatterIndexN i32 (full n) | 0.978 ? | 1.000 T | 0.999 T |
| parity | ScatterIndexN i32 (half n) | 1.158 ? | 0.998 ? | 1.003 T |
| parity | ScatterIndexN i32 (one lane) | 0.956 ? | 0.993 ? | 1.000 T |
| parity | ScatterOffset i32 (4096) | 0.995 ? | 0.994 ? | 0.997 ? |
| parity | SetMask i32 (4096) | 0.980 ? | 0.998 T | 1.006 T |
| parity | ShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.992 ? | 0.999 T | 0.999 T |
| parity | SlideMask1Down i32 (4096) | 1.091 ? | 1.010 ? | 1.000 T |
| parity | SlideMask1Up i32 (4096) | 1.086 ? | 0.988 ? | 1.000 ? |
| parity | SlideMaskDownLanes i32 amt=2 (4096) | 1.066 ? | 0.992 ? | 1.000 T |
| parity | SlideMaskUpLanes i32 amt=2 (4096) | 1.036 ? | 0.990 T | 0.992 ? |
| parity | StoreInterleaved2 f32 (4096) | 1.002 T | 1.001 T | 0.970 ? |
| parity | StoreInterleaved3 f32 (4096) | 1.031 ? | 1.032 ? | 0.960 ? |
| parity | StoreInterleaved4 f32 (4096) | 1.488 W | 1.046 ? | 1.049 ? |
| parity | TableLookupBytes u8 (4096) | 1.002 ? | 0.994 ? | 0.999 T |
| parity | TableLookupLanes i32 (4096) | 1.004 ? | 1.009 ? | 0.995 ? |
| parity | TwoTablesLookupLanes i32 (4096) | 0.998 T | 0.978 ? | 1.002 ? |
| full | Abs i32 | 0.995 T | 0.996 ? | 0.972 ? |
| full | AbsDiff i32 | 0.999 T | 0.991 T | 1.003 ? |
| full | Add f32 | 0.996 T | 1.002 ? | 0.985 ? |
| full | AddSub i32 | 0.993 ? | 0.990 ? | 0.997 ? |
| full | AllBits0 i32 | 0.989 ? | 1.011 T | 1.001 ? |
| full | AllBits1 i32 | 0.998 ? | 1.004 ? | 0.990 T |
| full | AllFalse i32 | 0.996 ? | 0.990 T | 0.990 ? |
| full | AllTrue i32 | 0.994 ? | 0.986 ? | 0.993 ? |
| full | And i32 | 1.000 T | 1.003 ? | 0.961 ? |
| full | AndNot i32 | 0.995 ? | 1.000 ? | 1.002 ? |
| full | AndXor i32 | 1.001 T | 1.005 ? | 0.997 ? |
| full | ApproximateReciprocal f32 | 1.001 ? | 0.990 ? | 1.007 T |
| full | ApproximateReciprocalSqrt f32 | 1.002 T | 0.989 T | 1.008 ? |
| full | AverageRound u16 | 0.997 ? | 0.990 ? | 1.007 ? |
| full | BitCast f32->i32 | 0.987 ? | 1.006 ? | 1.004 T |
| full | BitCast i32->f32 | 1.005 ? | 0.993 ? | 1.001 T |
| full | BitwiseIfThenElse i32 | 1.000 T | 1.007 ? | 1.008 ? |
| full | Broadcast<0> i32 | 1.000 T | 0.998 ? | 0.966 ? |
| full | BroadcastBlock<1> i32 | 1.014 ? | 1.002 ? | · |
| full | BroadcastLane<3> i32 | 1.007 ? | 1.290 W | 1.006 ? |
| full | BroadcastSignBit i32 | 0.997 T | 0.992 ? | 0.988 ? |
| full | Ceil f32 | 0.998 ? | 0.999 T | 0.995 ? |
| full | CeilInt f32->i32 | 1.484 W | 1.747 W | 1.003 ? |
| full | Clamp i32 | 0.995 T | 0.998 ? | 1.007 ? |
| full | CombineShiftRightBytes4 i32 | 0.992 ? | 1.004 ? | 1.005 ? |
| full | CombineShiftRightLanes1 i32 | 1.005 ? | 0.993 ? | 1.008 T |
| full | ComplexConj f32 | 1.243 W | 1.007 ? | 0.997 ? |
| full | ConcatLowerLower i32 | 1.082 ? | 0.999 ? | · |
| full | ConcatLowerUpper i32 | 1.080 ? | 0.986 ? | · |
| full | ConcatOdd i32 | 0.999 T | 0.991 ? | 1.001 ? |
| full | ConcatUpperLower i32 | 0.936 ? | 0.993 ? | · |
| full | ConcatUpperUpper i32 | 0.918 ? | 0.999 ? | · |
| full | ConvertTo f32->i32 | 1.446 W | 1.714 W | 1.012 ? |
| full | ConvertTo i32->f32 | 0.995 ? | 0.987 ? | 1.008 ? |
| full | CopySign f32 | 0.995 T | 0.986 ? | 1.017 ? |
| full | CopySignToAbs f32 | 0.995 ? | 0.993 T | 0.995 ? |
| full | CountTrue i32 | 1.025 ? | 0.999 ? | 1.009 ? |
| full | Div f32 | 1.002 T | 0.990 ? | 1.006 ? |
| full | Div i32 invariant | 1.148 W | 1.190 W | 1.001 ? |
| full | DupEven i32 | 1.003 ? | 1.005 ? | 0.969 ? |
| full | DupOdd i32 | 0.997 T | 0.998 ? | 1.003 ? |
| full | Eq i32 | 0.997 T | 0.979 ? | 1.033 ? |
| full | Eq128 u64 | 1.000 ? | 1.004 T | 0.996 ? |
| full | ExclusiveNeither i32 | 1.036 ? | 0.980 ? | 1.001 ? |
| full | FindFirstTrue i32 | 0.994 ? | 0.991 ? | 1.034 ? |
| full | FindKnownFirstTrue i32 | 1.032 ? | 0.993 ? | 1.047 ? |
| full | FindKnownLastTrue i32 | 0.980 ? | 0.995 ? | 0.976 ? |
| full | FindLastTrue i32 | 1.027 ? | 0.996 T | 1.162 ? |
| full | Floor f32 | 0.997 ? | 0.997 ? | 0.968 ? |
| full | FloorInt f32->i32 | 1.483 W | 1.746 W | 1.000 ? |
| full | Ge i32 | 0.984 ? | 1.000 ? | 0.961 ? |
| full | GetBiasedExponent f32 | 0.996 T | 1.048 ? | 1.006 T |
| full | Gt i32 | 0.995 T | 0.991 ? | 1.032 ? |
| full | HighestSetBitIndex u32 | 1.026 ? | 1.006 ? | 1.005 ? |
| full | IfNegativeThenElse f32 | 1.001 ? | 1.007 ? | 1.009 ? |
| full | IfNegativeThenElseZero f32 | 1.298 W | 1.192 W | 0.982 ? |
| full | IfNegativeThenZeroElse f32 | 1.361 W | 1.196 W | 1.018 ? |
| full | IfThenElse i32 | 1.003 T | 0.996 ? | 0.996 ? |
| full | IfThenElseZero i32 | 0.968 ? | 1.004 ? | 1.002 ? |
| full | IfThenZeroElse i32 | 0.968 ? | 1.008 ? | 1.028 ? |
| full | IfVecThenElse i32 | 1.004 T | 0.996 ? | 1.002 ? |
| full | InterleaveEven i32 | 1.003 T | 1.004 ? | 0.987 ? |
| full | InterleaveLowerBlocks i32 | 0.982 ? | 1.004 ? | · |
| full | InterleaveOdd i32 | 0.998 T | 0.997 ? | 0.980 ? |
| full | InterleaveUpper i32 | 0.992 T | 1.003 ? | 0.985 ? |
| full | InterleaveWholeLower i32 | 0.994 T | 0.997 T | 1.030 ? |
| full | InterleaveWholeUpper i32 | 0.996 T | 0.997 ? | 1.004 ? |
| full | IsEitherNaN f32 | 1.001 ? | 0.995 ? | 0.989 ? |
| full | IsFinite f32 | 0.987 ? | 1.258 W | 0.999 ? |
| full | IsInf f32 | 1.002 ? | 1.084 ? | 1.007 ? |
| full | IsNaN f32 | 0.996 ? | 0.996 ? | 1.006 ? |
| full | IsNegative i32 | 1.010 ? | 1.009 ? | 1.005 ? |
| full | Le i32 | 0.985 ? | 1.004 ? | 1.007 ? |
| full | Load/Store i32 aligned | 0.997 ? | 1.000 ? | 1.055 ? |
| full | LoadU/StoreU i32 | 0.990 ? | 0.998 ? | 1.006 ? |
| full | Lt i32 | 1.000 T | 1.001 ? | 0.983 ? |
| full | Lt128 u64 | 0.997 ? | 0.995 ? | 0.985 ? |
| full | Lt128Upper u64 | 1.004 T | 1.004 T | 0.991 ? |
| full | Mask And i32 | 1.015 ? | 0.983 ? | 0.994 ? |
| full | Mask AndNot i32 | 0.999 T | 1.000 ? | 0.999 ? |
| full | Mask Not i32 | 1.004 ? | 0.989 ? | 1.007 ? |
| full | Mask Or i32 | 0.999 T | 1.014 ? | 1.007 ? |
| full | Mask Xor i32 | 1.000 ? | 0.999 ? | 1.015 ? |
| full | MaskedApproxReciprocal f32 | 0.994 ? | 1.002 ? | 1.007 ? |
| full | MaskedApproxReciprocalSqrt f32 | 1.002 ? | 0.997 ? | 1.005 ? |
| full | MaskedConvertTo f32-i32 | 2.057 W | 1.770 W | 0.997 ? |
| full | MaskedLeadingZeroCount i32 | 1.001 ? | 0.991 ? | 0.999 ? |
| full | MaskedModOr i32 invariant | 1.102 W | 1.118 W | 1.005 ? |
| full | MaskedMulAdd f32 | 0.995 T | 1.010 ? | 0.981 ? |
| full | MaskedMulAddOr f32 | 1.001 T | 0.986 ? | 1.002 ? |
| full | MaskedMulComplex f32 | 1.205 W | 1.099 W | 1.008 ? |
| full | MaskedMulComplexAdd f32 | 1.133 W | 1.034 ? | 1.006 ? |
| full | MaskedMulComplexConj f32 | 1.209 W | 1.101 W | 1.006 ? |
| full | MaskedMulComplexConjAdd f32 | 1.130 W | 1.031 ? | 1.005 ? |
| full | MaskedMulComplexOr f32 | 1.169 W | 1.071 W | 1.007 ? |
| full | MaskedMulFixedPoint15 i16 | 0.993 ? | 0.994 ? | 1.001 ? |
| full | MaskedNegMulAdd f32 | 0.997 T | 1.001 ? | 0.993 ? |
| full | MaskedReduceMax i32 | 1.000 T | 0.993 ? | 1.005 ? |
| full | MaskedReduceMin i32 | 1.000 ? | 0.991 ? | 1.000 T |
| full | MaskedReduceSum i32 | 1.000 ? | 0.974 ? | 1.008 ? |
| full | MaskedSatAddOr i16 | 1.000 T | 0.997 ? | 1.005 ? |
| full | MaskedSatSubOr i16 | 1.005 T | 1.004 ? | 1.005 ? |
| full | MaskedShiftLeft<3> i32 | 0.996 T | 0.987 ? | 0.986 ? |
| full | MaskedShiftLeftOr<3> i32 | 1.000 ? | 0.998 T | 1.006 ? |
| full | MaskedShiftRight<3> i32 | 0.996 T | 1.001 T | 1.021 ? |
| full | MaskedShiftRightOr<3> i32 | 0.999 ? | 0.998 ? | 1.003 T |
| full | MaskedShl i32 | 0.993 ? | 0.995 T | 1.011 ? |
| full | MaskedShlOr i32 | 0.993 T | 1.009 ? | 1.003 ? |
| full | MaskedShr i32 | 1.002 ? | 1.002 ? | 1.000 ? |
| full | MaskedShrOr i32 | 1.000 ? | 1.002 ? | 1.007 ? |
| full | MaskedWidenMulPairwiseAdd i16-i32 | 0.996 ? | 1.013 ? | 1.006 ? |
| full | Max f32 | 0.993 T | 1.000 ? | 1.000 ? |
| full | Max i32 | 1.000 T | 1.000 ? | 0.997 ? |
| full | Max128 u64 | 1.000 T | 1.005 ? | 1.005 T |
| full | MaxMagnitude f32 | 0.993 ? | 1.002 ? | 1.007 ? |
| full | MaxNumber f32 | 0.997 T | 0.995 ? | 1.019 ? |
| full | MaxOfLanes i32 | 1.137 W | 1.061 ? | 0.966 ? |
| full | Min f32 | 0.989 T | 0.991 ? | 1.002 ? |
| full | Min128 u64 | 0.995 ? | 0.998 ? | 1.005 ? |
| full | MinMagnitude f32 | 0.993 T | 1.001 ? | 1.001 ? |
| full | MinNumber f32 | 1.005 ? | 0.999 ? | 1.022 ? |
| full | MinOfLanes i32 | 1.129 W | 1.069 ? | 1.042 ? |
| full | Mod i32 invariant | 1.147 W | 1.150 W | 1.011 ? |
| full | Mod i32 invariant store | 1.104 W | 1.040 ? | 1.087 ? |
| full | Mul f32 | 1.003 T | 0.992 ? | 0.973 ? |
| full | Mul i32 | 1.002 ? | 1.000 ? | 0.979 ? |
| full | MulAddSub f32 | 1.008 T | 0.998 ? | 0.996 ? |
| full | MulByFloorPow2 f32 | 1.031 ? | 0.991 ? | 1.006 ? |
| full | MulByPow2 f32 | 1.002 T | 0.992 ? | 0.998 ? |
| full | MulComplex f32 | 1.427 W | 1.235 W | 1.006 ? |
| full | MulComplexAdd f32 | 1.019 T | 1.032 ? | 1.005 T |
| full | MulComplexConj f32 | 1.410 W | 1.234 W | 1.006 T |
| full | MulComplexConjAdd f32 | 1.017 T | 1.036 ? | 1.006 ? |
| full | MulEven i32-i64 | 1.000 ? | 1.002 ? | 1.002 ? |
| full | MulOdd i32-i64 | 0.983 ? | 0.927 ? | 1.001 ? |
| full | MulRound f32 | 0.998 ? | 0.993 ? | 1.017 ? |
| full | MulSub f32 | 0.999 T | 1.008 ? | 1.014 ? |
| full | MulSubAdd f32 | 1.011 T | 0.996 ? | 1.008 ? |
| full | Ne i32 | 0.987 T | 0.997 T | 1.009 ? |
| full | Ne128 u64 | 0.993 ? | 1.011 ? | 0.967 ? |
| full | NearestInt f32->i32 | 1.434 W | 1.754 W | 1.007 ? |
| full | Neg i32 | 0.999 ? | 0.991 ? | 1.009 ? |
| full | NegMulAdd f32 | 1.000 ? | 1.004 ? | 1.012 ? |
| full | NegMulSub f32 | 1.003 ? | 0.998 T | 0.995 ? |
| full | Not i32 | 0.961 ? | 0.991 ? | 0.985 ? |
| full | OddEven i32 | 1.001 T | 0.984 ? | 0.983 ? |
| full | OddEvenBlocks i32 | 0.990 ? | 0.999 ? | · |
| full | Or i32 | 0.997 T | 0.996 ? | 1.010 ? |
| full | Or3 i32 | 1.000 ? | 0.997 ? | 0.989 ? |
| full | OrAnd i32 | 0.996 ? | 1.000 ? | 1.009 ? |
| full | PairwiseAdd i32 | 1.085 W | 1.146 W | 0.992 ? |
| full | PairwiseAdd128 i32 | 1.000 T | 1.007 ? | 1.011 ? |
| full | PairwiseSub i32 | 1.094 W | 1.148 W | 1.000 ? |
| full | PairwiseSub128 i32 | 1.087 W | 1.153 W | 1.001 ? |
| full | Per4LaneBlockShuffle 3210 i32 | 0.991 ? | 0.986 ? | 1.005 ? |
| full | ReduceMax i16 | 1.029 ? | 1.021 ? | 1.004 T |
| full | ReduceMax i32 | 1.185 W | 1.045 ? | 1.005 T |
| full | ReduceMax i64 | 1.004 ? | 0.998 T | 1.005 ? |
| full | ReduceMin i16 | 1.021 T | 1.044 ? | 1.007 ? |
| full | ReduceMin i32 | 1.181 W | 1.027 ? | 1.005 ? |
| full | ReduceMin i64 | 0.998 T | 0.993 ? | 1.009 T |
| full | ReduceSum i16 | 1.020 ? | 1.041 ? | 1.007 ? |
| full | ReduceSum i64 | 1.042 ? | 1.093 ? | 1.006 T |
| full | Reverse2 i32 | 1.008 T | 0.999 ? | 0.986 ? |
| full | Reverse4 i32 | 0.999 T | 1.003 ? | 1.005 ? |
| full | ReverseBits i32 | 2.086 W | 2.225 W | 1.031 ? |
| full | ReverseBlocks i32 | 0.980 ? | 0.984 ? | · |
| full | RotateLeft<3> i32 | 1.000 T | 0.990 ? | 1.045 ? |
| full | RotateLeftSame u32 bits=5 | 0.991 ? | 1.008 ? | 0.985 ? |
| full | RotateRight<3> i32 | 0.988 T | 0.995 ? | 1.006 ? |
| full | RotateRightSame u32 bits=5 | 1.008 ? | 1.007 ? | 1.019 T |
| full | Round f32 | 1.001 ? | 1.007 ? | 0.995 ? |
| full | RoundingShiftRight<2> i32 | 1.010 ? | 1.150 W | 1.019 ? |
| full | RoundingShiftRightSame i32 bits=5 | 1.096 W | 1.009 ? | 1.461 W |
| full | RoundingShr i32 | 1.451 W | 1.442 W | 0.967 ? |
| full | RoundToInt f32->i32 | 1.390 W | 1.780 W | 1.005 ? |
| full | SaturatedAbs i32 | 0.994 ? | 1.000 ? | 1.007 ? |
| full | SaturatedNeg i32 | 0.993 T | 0.980 ? | 1.024 ? |
| full | SaturatedSub i16 | 1.003 T | 1.006 ? | 1.006 ? |
| full | Set i32 | 1.000 T | 1.001 ? | 1.007 ? |
| full | SetAtOrAfterFirst i32 | 0.997 T | 1.004 ? | 1.001 ? |
| full | SetAtOrBeforeFirst i32 | 1.193 W | 1.168 W | 1.006 T |
| full | SetBeforeFirst i32 | 1.040 ? | 1.263 W | 1.006 ? |
| full | SetOnlyFirst i32 | 0.998 T | 1.665 W | 1.009 T |
| full | ShiftLeft<3> i32 | 1.000 T | 1.003 ? | 1.029 ? |
| full | ShiftLeftBytes<4> i32 | 1.010 ? | 1.004 T | 1.004 ? |
| full | ShiftLeftLanes<1> i32 | 1.006 ? | 1.007 ? | 0.995 T |
| full | ShiftLeftSame i32 bits=3 | 0.999 T | 1.019 ? | 1.986 W |
| full | ShiftRight<3> i32 | 0.999 ? | 0.994 ? | 0.976 ? |
| full | ShiftRightBytes<4> i32 | 0.996 ? | 1.007 ? | 1.004 ? |
| full | ShiftRightLanes<1> i32 | 1.015 ? | 1.000 ? | 1.007 ? |
| full | ShiftRightSame i32 bits=3 | 1.006 ? | 0.983 ? | 1.197 W |
| full | Shl i32 | 0.993 T | 0.994 ? | 1.001 ? |
| full | Shr u32 (variable) | 1.003 T | 1.011 T | 0.990 ? |
| full | Shuffle0123 i32 | 0.997 T | 1.010 ? | 1.039 ? |
| full | Shuffle0321 i32 | 0.996 T | 1.006 ? | 0.974 ? |
| full | Shuffle1032 i32 | 0.998 T | 0.998 T | 1.002 ? |
| full | Shuffle2103 i32 | 0.996 T | 1.005 ? | 0.997 ? |
| full | Shuffle2301 i32 | 0.997 T | 0.998 ? | 0.969 ? |
| full | Slide1Down i32 | 1.003 ? | 0.997 ? | 1.020 ? |
| full | Slide1Up i32 | 1.002 ? | 1.001 ? | 1.018 ? |
| full | SlideDownLanes amt=2 i32 | 0.997 ? | 0.999 ? | 1.012 ? |
| full | SlideUpLanes amt=2 i32 | 0.998 ? | 0.994 ? | 1.014 ? |
| full | Sqrt f32 | 0.996 ? | 0.987 ? | 1.006 ? |
| full | Sub f32 | 0.994 T | 0.999 T | 1.007 ? |
| full | Sub i32 | 1.004 T | 0.995 ? | 0.995 ? |
| full | SumOfLanes i32 | 1.049 ? | 1.074 W | 1.047 ? |
| full | SwapAdjacentBlocks i32 | 0.977 ? | 0.996 T | · |
| full | TestBit i32 | 1.048 ? | 1.001 ? | 1.007 ? |
| full | TrailingZeroCount i32 | 1.001 T | 0.996 ? | 1.010 ? |
| full | Trunc f32 | 0.998 ? | 0.986 ? | 1.014 ? |
| full | TruncateTo u32->u16 | 0.972 ? | 0.995 ? | 1.009 ? |
| full | Xor i32 | 0.993 T | 1.005 T | 0.987 ? |
| full | Xor3 i32 | 1.606 W | 0.998 ? | 1.007 ? |
| full | XorAndNot i32 | 0.996 T | 0.994 ? | 1.011 ? |
| full | Zero i32 | 0.997 ? | 0.996 ? | 0.957 ? |
| full | ZeroIfNegative f32 | 1.013 ? | 1.037 ? | 0.967 ? |
| widen | Add f32 (4096) | 1.000 T | 1.000 ? | 1.000 ? |
| widen | DemoteTo i32->i16 (sat) | 1.872 W | 1.896 W | 1.980 W |
| widen | GetExponent f32 | 0.998 T | 1.059 ? | 1.022 ? |
| widen | MulEvenAdd bf16->f32 | 0.991 ? | 0.993 ? | 1.017 ? |
| widen | MulFixedPoint15 i16 | 0.995 T | 0.999 T | 1.019 ? |
| widen | MulOddAdd bf16->f32 | 0.990 ? | 0.975 ? | 1.019 ? |
| widen | OrderedDemote2To i32->i16 | 0.998 ? | 0.994 T | 1.017 ? |
| widen | OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 0.997 T | 1.226 W | 1.017 ? |
| widen | OrderedShiftRightAndDemote2To<2> i32->i16 | 0.999 T | 0.995 ? | 1.018 ? |
| widen | PromoteEvenTo i8->i16 | 0.999 ? | 1.003 ? | 1.016 ? |
| widen | PromoteOddTo i8->i16 | 0.998 ? | 0.990 ? | 1.018 ? |
| widen | PromoteUpperTo i16->i32 | 1.011 ? | 0.994 ? | 1.020 ? |
| widen | ReorderDemote2To i32->i16 | 0.997 ? | 1.010 ? | 1.019 ? |
| widen | ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.006 ? | 1.162 W | 1.017 ? |
| widen | ReorderShiftRightAndDemote2To<2> i32->i16 | 1.002 ? | 0.988 ? | 1.016 ? |
| widen | ReorderWidenMulAccumulate i16->i32 | 1.005 ? | 0.992 ? | 1.018 ? |
| widen | SatWidenMulAccumFixedPoint i16->i32 | 1.009 ? | 0.991 ? | 1.026 ? |
| widen | SatWidenMulPairwiseAccumulate i16->i32 | 0.999 T | 0.997 ? | 1.020 ? |
| widen | SatWidenMulPairwiseAdd u8*i8->i16 | 0.997 ? | 0.999 ? | 1.019 ? |
| widen | SumOfMulQuadAccumulate i8->i32 | 0.993 T | 0.991 ? | 1.020 ? |
| widen | SumsOf2 u8->u16 | 0.999 ? | 0.985 ? | 1.020 ? |
| widen | SumsOf4 u8->u32 | 0.986 ? | 1.012 ? | 1.019 ? |
| widen | SumsOf8 u8->u64 | 0.994 ? | 1.000 ? | 1.018 ? |
| widen | SumsOf8AbsDiff u8->u64 | 1.009 ? | 1.006 ? | 1.019 ? |
| widen | SumsOfAdjQuadAbsDiff u8->u16 | 0.998 ? | 0.995 ? | 1.019 ? |
| widen | U8FromU32 u32->u8 | 0.964 ? | 0.995 ? | 1.020 ? |
| widen | WidenMulAccumulate i16->i32 | 0.994 T | 0.983 ? | 1.028 ? |
| widen | WidenMulPairwiseAdd i16->i32 | 1.001 T | 0.988 ? | 1.020 ? |
| widen | ZeroExtendVector i16 half->full | 0.985 ? | 0.996 ? | 1.021 ? |

</details>

## HYLE 独有测试(无 Highway 对手的 op)

Highway/xsimd 没有这些同契约 op，无法做 HYLE-vs-Highway 对拍，故不在上方全量表。它们用 v4 的
replicated info pair 对拍**自家基线**（标量 fold / 裸估计 / Div）；这里只能说明相对该基线的成本，
不能把“没有直接对手”偷换成性能 SOTA。比值 = HYLE / baseline（>1 = HYLE 快）。

| op | 对拍基线 | MSVC | WSL GCC | Mac NEON | 备注 |
|---|---|---|---|---|---|
| ReduceMul i32 | 标量 fold | 1.55× | 1.44× | 1.01× | halving tree；x86 消 store+reload |
| ReduceAnd i32 | 标量 fold | 1.45× | 2.29× | 1.01× | 同上 |
| ReduceOr i32 | 标量 fold | 1.45× | 2.30× | 1.01× | 同上 |
| ReciprocalNewton f32 | 裸 ApproxRecip | 0.79× | 0.81× | 1.00× | x86 一步 Newton 提精度；Apple 路由原生 fdiv |
| ReciprocalNewton f32 | Div(1,x) | 2.42× | 3.07× | 1.00× | Apple fdiv 与基线等价 |
| ReciprocalSqrtNewton f32 | 裸 ApproxRSqrt | 0.64× | 0.52× | 0.52× | 一步 Newton 的精度成本 |
| ReciprocalSqrtNewton f32 | Div(1,Sqrt(x)) | 4.34× | 4.49× | 1.51× | 三平台均快于真 sqrt+div |

`Reciprocal*Newton/Approx` 小于 1 表示精修吞吐低于裸估计，是用速度换精度；另列 `/Div` 才回答它
是否仍比真除法快。旧表把 Apple generic Newton 误写成“远快于 Div”，v4 info pair 揭示它仅 0.52×；
NEON 已改原生 reciprocal/rsqrt Newton，Apple reciprocal 选择更快更准的 fdiv，最终约 1.00× Div。

## 算法层备忘(composite;非 op-vs-Highway)

这些是 `math.h` 组合核 / 算法件,不进上方全量表(Highway 无对等公开 API 或形状不是单 op)。
对拍基线 = 标量参考 / 自家旧实现。细节与弯路见 [`simd-complexity-opportunities.md`](./simd-complexity-opportunities.md)。

### softmax / log_softmax(P0-1,2026-07-15,`d9685ff`)

- **算法**:单次 exp 融合 multipass — max → 4K chunk 内 `(x-m)+exp_f32_batch+sum` → scale。
  整表只做 **一次** 向量 exp(曾试 online max+sum-exp 因双 exp 在 n=128K 慢 1.6× 已弃)。
- **正确性**:`Math.Softmax` 对拍 double 稳定参考 — Win MSVC / WSL GCC / Mac NEON 全绿。
- **性能(Mac NEON)**:

| 对比 | n=4K | n=32K | n=128K |
|---|---:|---:|---:|
| 融合 / 标量 3-pass | **0.41×** 时间(快) | 0.66× | 0.91× |
| 融合 / 旧 multipass(5 遍) | **0.81×** | 0.86× | ~1.00× |
| `activation_bench` vs scalar | **1.16×** 吞吐(115.9%) | — | — |

- **读法**:小 n 算力收益明显;大 n 逼近带宽墙与旧 multipass 持平,但不再双 exp 倒退。
  **不是** HYLE-vs-Highway 比值。

### f64 exp/log 表驱动 range reduction(P2-1,2026-07-15)

- **算法**:从 Arm Optimized Routines 的 AdvSIMD f64 核派生 N=128 方案。`exp` 以
  `round(x*128/ln2)` 选 `2^(j/128)` scale 表，做 split-ln2 reduction 与 4 次
  `exp(r)-1` 多项式；`log` 从 IEEE 位域精确拆 `x=2^k*z`，查 128 组 `{1/c,log(c)}`，
  做 5 次 `log1p(r)-r` 多项式。实现是 Highway 泛型形式，覆盖 AVX2、AVX-512 与 NEON；
  版权/许可与来源见 `src/simd/math/table_f64.h`。
- **迭代与闸门**:

  | 轮次 | 结果 |
  |---|---|
  | 0 | Highway 旧核正常有限域精度可用，但向量 special lane 不保 libm 语义；新增全域 ULP gate 后先在 wrapper 中将含 NaN/Inf/域外/边界 lane 的整向量回退 `std::exp`/`std::log`。 |
  | 1 | `exp` 的 Arm `vfmaq`→Highway `MulAdd` 参数次序误译，ULP gate 大幅失败，拒绝。 |
  | 2 | `log` 的多项式与 `hi` 重建也有同类 FMA 次序误译，ULP gate 失败，拒绝。 |
  | 3 | 按 `a*b+c` 语义重译并保留 split 常量，三平台全域 gate 均为 `exp ≤2 ULP`、`log ≤1 ULP`，接受。 |

- **直接 same-ISA 复测**:每平台重配 `HYLE_BENCH_HIGHWAY=ON`，各独立运行 5 次；下表是
  `4096` 正常有限输入的中位 `ns/op`。HYLE 行包含产品 wrapper 的 exceptional-vector 检查，
  Highway 行是无 special handling 的直接基准，故用于核本吞吐对比而不是替代产品语义。

  | 平台 | exp: HYLE / Highway | exp 判定 | log: HYLE / Highway | log 判定 |
  |---|---:|---|---:|---|
  | Win MSVC AVX2 | **0.43 / 0.83** | **1.93× 快** | **0.71 / 0.75** | **1.06× 快** |
  | WSL GCC14 AVX2 | **0.44 / 0.83** | **1.89× 快** | 0.80 / **0.76** | HYLE 慢 5% |
  | Mac AppleClang NEON | **0.70 / 1.35** | **1.93× 快** | **0.97 / 1.37** | **1.41× 快** |

- **WSL log 分流复核并拒绝**:直接 Highway 的 `0.76` 并不能直接装进产品；保留同一个
  exceptional-vector 检查后改走 `hn::Log` 的完整 HYLE wrapper 实测 **0.94–1.05 ns/op**，
  比表核的 `0.74–0.83` 更慢。故不做 Linux 特判：产品 `log_f64_batch` 全平台保留表核，
  WSL 的 5% 直接对手差距如实保留，不能拿没有 IEEE special 语义的基准行冒充可上架路径。

## 41 个待分类候选（不是 41 个待补 benchmark）

**2026-07-16 已闭环。** `scripts/simd-op-manifest.json` 将旧 41 项固定为 15 native、6 alias、
15 synthetic、2 paired/info、3 algorithm；新增 4 个 128-bit upper 合成接口后 manifest 为 45 项，
coverage 的未分类数仍为 0。15 个 native 新行在 Windows/WSL/Mac 各自可用 target 的 focused 双序认证均为 **0 LOSE**。

第二轮狩猎唯一发现的真回归是 `SumsOfShuffledQuadAbsDiff`：Windows 约 0.08x、Mac 约 0.11x，
原因是 HYLE SSE/AVX2/NEON 仍走 store/scalar/reload oracle。改为 Highway 同构的寄存器 recipe，补恒等
和非恒等 shuffle 两种测试形状后，WSL 认证约 1.49x/1.43x WIN，Mac 中心值均约 0.99、认证为 TIE；
SSE4 同步向量化，AVX-512 继续使用原生 DB-SAD。

`Stream` 另走 256 MiB 带宽 benchmark。三次外层复跑 HYLE/Highway 比值为 Windows
`1.001/1.021/1.005`、WSL `0.988/1.047/1.058`、Mac `3.787/3.316/3.468`。Apple 的大幅优势
来自 Highway 每 16B store 前的 write-prefetch，而 HYLE/普通 Store 直接写；它是算法/工作集证据，
不进入 L1 op-parity 分母。

旧脚本按 op 名与 benchmark 标题做启发式匹配，因此这 41 个名字只适合作为审计入口。它会把成对测量、
`run_third` 信息行、精确 alias 和合成 op 都报成“尚未写 bench”。simd-23 将以显式 manifest 取代
这个口径；**合成 op 不进入 benchmark 分母**。

原始候选分三档如下（保留快照以便逐项核账）：
- **结构性原语(在 CMP/mask 组合里被间接跑,但无独立 parity 行)**:BitsFromMask、MaskFromVec、
  MaskFromNormalizedVec、VecFromMask、SignBit、ShiftRightUniform、LowerHalfOfMask、UpperHalfOfMask、
  MaskFalse、LoadMaskBits、StoreMaskBits。
- **InRange / 契约快变体(消费方保证输入域→跳边界处理)**:ConvertInRangeTo、DemoteInRangeTo、
  DemoteToNearestIntInRange、PromoteInRange{Lower,Upper,Even,Odd}To、TableLookupBytesInRange、
  TableLookupBytesOr0。
- **真缺口(有 Highway 等价、值得补)**:MaskedSet、MaskedSetOr、MaskedIsNaN、InterleaveEvenBlocks、
  InterleaveOddBlocks、InterleaveUpperBlocks、SlideUpBlocks、SlideDownBlocks、Lt128Vec、
  SumsOfShuffledQuadAbsDiff、OrderedTruncate2To、TruncateStore、Shuffle01、RearrangeToOddPlusEven、
  SafeCopyN、SafeFillN、Stream、CompressBits、CompressBitsStore、CompressBlocksNot、
  IfNegativeThenNegOrUndefIfZero。

首轮核账：`TableLookupBytesInRange` 已在 `TableLookupBytes` 成对行中测量，
`DemoteToNearestIntInRange` 已是 `run_third` 信息行；约 14 项有独立 native/contract 性能价值，另将
`Stream` 放入大于 LLC 的带宽 benchmark。`SignBit`、`ShiftRightUniform`、`MaskedSet*`、
`PromoteInRange*`、`OrderedTruncate2To`、`TruncateStore`、`CompressBits*`、`SafeCopyN/SafeFillN`
等 alias/合成项只需正确性和零开销检查，不补孤立 nanobench。完整分类与验收见
[`simd-23`](./simd-23-cxx-facade-and-perf-closure.md)。

候选清单可用 `python scripts/simd-op-coverage.py --check` 复算；CI 会拒绝新增未分类项、缺 row 的
native 项，以及误进独立分母的 alias/synthetic 项。

## 语义缺口(对 Highway op 名的差集,与"41 待补 bench"正交)

上面 41 个是"HYLE 已有该 op 但没写 perf bench";**这一节是"HYLE 连该 op 名都还没有"**(对 Highway
全部 `HWY_API` 公开名做差集甄别,剔除内部积木/换名等价后,真缺 ≈40 名)。两者正交:前者补 bench,后者
补实现。(本节从原 `simd-gap-inventory.md` 收编,该文件已退役为指针。)

- **P1 旧误报已纠正**：基础 Masked 算术/逻辑/比较/一元家族早已在 `vec.h` 完整生成并有 oracle。
- **P2 已闭合**：Eq128Upper/Ne128Upper/Max128Upper/Min128Upper 已补；Highway 的
  WidenLow/HighMulAdd 是 internal helper，HYLE 已暴露更完整的 WidenMulAccumulate 消费者能力，不镜像名字。
- **P3 机会型(硬件门控)**:GFNI GaloisAffine(+GF2P8 乘,Icelake+/Zen4+,走 AES/CLMul 先例;GF(2^8) 仿射
  可反哺 ReverseBits/子字节置换的更优实现)。
- **明确不做**:ShuffleTwo*/PerBlock2x2MatMul/LoadTransposedBlocks*(缺能力类或无消费者;名字数不是 KPI)。

## 复现
```
# 每平台:构建 hyle_op_parity{,_full,_widen}_bench_<isa>,双序各跑一遍(Mac 建议 4 遍),然后
python scripts/op-parity-report.py --certify h1.txt h2.txt g1.txt g2.txt
python scripts/op-parity-report.py --gate 0.06 --certify mac_h*.txt mac_g*.txt
# 报告首段为锚点校准;CALIBRATION PASS 之后的判决才可引用。
```
