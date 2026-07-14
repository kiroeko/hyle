# SIMD op parity 三平台矩阵(2026-07-14 校准仪器 v2;全部修复轮次收口后三平台真输清单 **全零**:MSVC 0 / GCC 0 / NEON 0)

> 本表由重建后的测量引擎产出(`bench/simd/op_parity_engine.h` + `scripts/op-parity-report.py`),
> 取代旧的"原始读数+人工伪影标签"体系。旧 op_parity 的数字被证明携带三层测量伪影(P/E 调度、
> 代码放置运气、闭包栈 4K 别名,合计可达 ±30~40%),单条 `vpsubd` 的 `Neg i32` 曾被它测成
> 0.75~1.49 之间的任意值。完整证据链与重建过程:[`../debug/op-parity-harness-artifacts-hunt.md`](../debug/op-parity-harness-artifacts-hunt.md)。

## 方法论(数字为何可信)
- **测量**:钉一颗 P 核+高优先级;每 op 每侧 4 个独立实例化的代码副本(采样放置运气,防 ICF 折叠,
  且带确定性相位错位 stagger —— v2 修复:同侧副本若以 64 倍数步长排布会共享同一相位,
  使 best-of-replicas 偏向另一侧达 ~10%,同文本对拍曾读 0.898,修后读 1.000);
  H/G 交替轮次;kernel 指针外提(热循环零闭包访问);宏形状循环 4x 展开。
- **聚合**:best-of-replicas(放置理想值)+ median + 逐轮配对比值三估计器;判决需三者在每个输入
  运行中一致越线,且线宽随该 op 实测副本散布自适应(>= ±3%)。每平台喂双序(HYLE 先/Highway 先)。
- **校准闸门**:5 个源码级验证"两侧同指令"的锚点 op(真值=1.00)必须全部落在闸门内,否则整表拒绝输出。
  本表锚点读数:MSVC/WSL 0.99~1.03(闸门 ±4%),Mac 0.95~1.05(闸门 ±6%,无亲和性 API,列分辨率较低,
  已用 4 次运行合并 + 8x epoch 补偿)。
- **判读**:数字 = HYLE/Highway 配对中位比值(>1=HYLE 快);W=WIN,L=LOSE,==TIE,?=UNSTABLE(多运行判决不一致),
  ·=该平台无此 op。UNSTABLE 一律不算输赢。Mac 列只覆盖 full 形状集(NEON 无 parity/widen 目标)。

## 真输清单(校准仪器判决,按比值升序)
### MSVC(AVX2): 0 个
- (BroadcastLane<3> 最终轮清零:0.93→**1.00 TIE**。旧"形状红利改不动"定论收回——可折叠与否取决于
  确切 intrinsic 写法:MSVC 对 Highway 的 `inserti128(v, low128(v), 1)` 自 dup 形态有折叠 peephole
  (load 输入→内存源 vbroadcasti128,shuffle 口免费),而 `broadcastsi128(cast)` 写法栈 spill——旧实验
  的 0.47 由此而来,当时误判为"该序列本质需要内存源"。已按编译器分叉:MSVC=inserti128 自 dup+vpshufd,
  GCC 保留 vpermd(1.13W);Slide1Up 金丝雀 1.00,无 ConcatLowerLower 类复合上下文回归。)

### WSL/GCC(AVX2): 0 个(full 0/201 LOSE)
- (FindLastTrue 最终轮清零:0.94→**1.00 TIE**。旧"仪器残余"定论收回——规范化逐条 diff 发现两侧循环
  14 vs 15 条:HYLE 多一条 movslq,因 `31 - std::countl_zero(b)` 的减法在 int 域做、进 64 位 sink 前
  需符号扩展;Highway 的 Num0Bits* 为 size_t 宽全程 64 位。改 ptrdiff_t 域减法(avx2+sse4)即消。
  方法教训:此前"objdump 逐指令等价"是**向量签名级**对比的假象——标量尾巴(寄存器宽度/扩展指令)
  没有逐条比,一条指令恰好就是那 6%。)

### Mac/NEON: 0 个
- (第十四轮清零:MulAddSub/MulSubAdd 弃两 FMA 的 OddEven(MulAdd,MulSub) 改 Highway 单 FMA 形,见演进段。)

### 与旧报告的对照(演进:旧 harness -> v1 校准仪器 -> 修复 + v2 stagger)
- 旧 op_parity 报 ~40-50 个 0.xx(MSVC 列);v1 校准仪器判 MSVC 21 / WSL 11 / Mac 3;两轮后端修复
  (第一轮:ReverseLaneBytes 0.05->TIE、Broadcast<0>、TruncateTo、SumOfLanes、Min/MaxOfLanes;
  第二轮:TZC 三平台追平、HSBI、DemoteMaskTo 0.33->1.4W、LoadN/StoreN 与 LoadNOr 弃 vpmaskmovd
  ->WIN、CombineMasks、ConcatLowerLower)+ v2 stagger 仪器修相位相关后,收敛为上表数字。
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
  **至此 x86 两列真输清单双双归零(MSVC full 剩 AddSub 0.86 与 0.9x 边缘带徘徊者,详见全量表)。**
  复审教训:被"定论"关闭的项,证据等级要过硬——签名级 disasm 对比、单一形态的否定实验,都不足以
  盖棺;逐条 diff 与多形态尝试才算。

## 全量表

| op | MSVC(AVX2) | WSL GCC(AVX2) | Mac(NEON) |
|---|---|---|---|
| Abs i32 | 1.01= | 0.98= | 1.02= |
| AbsDiff i32 | 1.00= | 1.00= | 1.02= |
| Add f32 | 1.00= | 1.00= | 0.99= |
| Add f32 (4096) | 1.00= | 0.99= | · |
| AddSub i32 | 0.84? | 1.00= | 1.00? |
| AllBits0 i32 | 1.02= | 1.00= | 1.00= |
| AllBits1 i32 | 0.98= | 0.98= | 1.00= |
| AllFalse i32 | 0.99? | 1.01= | 1.00= |
| AllTrue i32 | 0.99= | 0.99= | 1.00? |
| And i32 | 1.00= | 0.99= | 0.97= |
| AndNot i32 | 1.00= | 1.01= | 1.02= |
| ApproximateReciprocal f32 | 1.00= | 1.01= | 1.00= |
| ApproximateReciprocalSqrt f32 | 1.00= | 0.99= | 1.00= |
| AverageRound u16 | 1.00= | 1.01= | 1.00= |
| BitCast f32->i32 | 0.99= | 1.00= | 1.00= |
| BitCast i32->f32 | 1.01= | 1.00= | 1.00= |
| BitShuffle u64 (4096) | 1.07W | 1.11W | · |
| BlendedStore i32 (4096, 50%) | 1.00= | 0.99= | · |
| Broadcast<0> i32 | 0.99= | 0.99= | 1.05= |
| BroadcastBlock<1> i32 | 1.03= | 1.01= | · |
| BroadcastLane<3> i32 | 0.93L | 1.14W | 1.00= |
| BroadcastSignBit i32 | 1.01= | 1.00= | 0.99= |
| Ceil f32 | 1.01= | 0.99= | 0.99= |
| CeilInt f32->i32 | 1.50W | 1.54W | 0.99= |
| Clamp i32 | 1.00= | 1.03? | 1.01= |
| CombineMasks i32 (4096) | 1.00= | 1.01= | · |
| CombineShiftRightBytes4 i32 | 1.04= | 0.99= | 1.00= |
| CombineShiftRightLanes1 i32 | 0.96= | 0.99= | 1.00= |
| ComplexConj f32 | 1.25W | 0.99= | 1.01= |
| Compress i32 (4096, 50% dense) | 0.94? | 1.00= | · |
| CompressBlendedStore i32 (4096, 50%) | 0.98? | 1.15W | · |
| CompressNot i32 (4096, 50%) | 0.95? | 1.00= | · |
| CompressStore i32 (4096, 50%) | 0.97? | 1.00= | · |
| ConcatEven i32 (4096) | 1.02= | 1.00? | · |
| ConcatLowerLower i32 | 0.97= | 0.99= | · |
| ConcatLowerUpper i32 | 0.92= | 0.98= | · |
| ConcatOdd i32 | 1.01= | 1.01= | 1.02= |
| ConcatUpperLower i32 | 1.04= | 1.01= | · |
| ConcatUpperUpper i32 | 0.96= | 0.99= | · |
| ConvertTo f32->i32 | 1.42W | 1.11W | 1.01= |
| ConvertTo i32->f32 | 0.99= | 0.99= | 1.01= |
| CopySign f32 | 1.00= | 1.00= | 0.98= |
| CountTrue i32 | 1.04= | 1.00= | 1.00= |
| DemoteMaskTo i32->i16 (4096) | 1.04= | 1.42W | · |
| DemoteTo i32->i16 (sat) | 1.88W | 1.89W | · |
| DemoteTo i32->i16 1-arg (4096) | 1.00= | 1.00= | · |
| DemoteToNearestInt f64->i32 (4096) | 1.01= | 0.99= | · |
| Div f32 | 1.00= | 1.01= | 1.00= |
| Dup128MaskFromMaskBits i32 (4096) | 1.02= | 1.19W | · |
| DupEven i32 | 1.00= | 1.00= | 0.99= |
| DupOdd i32 | 1.00= | 1.01= | 1.07= |
| Eq i32 | 0.99= | 1.02= | 1.04= |
| Eq128 u64 | 1.01= | 1.00= | 0.99= |
| ExclusiveNeither i32 | 1.01= | 0.98= | 1.00= |
| Expand i32 (4096, 50%) | 1.99W | 2.13W | · |
| FindFirstTrue i32 | 0.99= | 1.00= | 1.05= |
| FindKnownFirstTrue i32 | 0.97= | 1.00= | 1.18W |
| FindKnownLastTrue i32 | 1.08W | 1.01= | 1.12W |
| FindLastTrue i32 | 1.02= | 0.94L | 1.16W |
| Floor f32 | 1.01= | 1.00= | 1.01= |
| FloorInt f32->i32 | 1.49W | 1.55W | 0.99= |
| GatherIndex i32 (4096) | 1.00= | 1.01= | · |
| GatherIndexN i32 (half n) | 1.00= | 0.97= | · |
| GatherIndexNOr i32 (half n) | 1.00= | 1.00= | · |
| GatherOffset i32 (4096) | 1.00= | 1.00= | · |
| Ge i32 | 0.99= | 1.00= | 0.98= |
| GetBiasedExponent f32 | 1.00= | 1.03= | 1.00= |
| GetExponent f32 | 1.00= | 0.99= | · |
| Gt i32 | 1.00= | 1.02= | 1.01= |
| HighestSetBitIndex u32 | 1.02= | 1.01= | 1.00= |
| IfNegativeThenElse f32 | 1.05W | 1.10W | 1.01= |
| IfNegativeThenElseZero f32 | 1.35W | 1.08W | 0.97= |
| IfNegativeThenZeroElse f32 | 1.34W | 1.09W | 1.00= |
| IfThenElse i32 | 1.00= | 1.02= | 1.00= |
| IfThenElseZero i32 | 0.97? | 0.99= | 1.00? |
| IfThenZeroElse i32 | 0.97= | 1.01= | 0.95= |
| InterleaveEven i32 | 1.00= | 1.01= | 1.03? |
| InterleaveLower i32 (4096) | 1.00= | 1.00= | · |
| InterleaveLowerBlocks i32 | 0.92= | 0.99= | · |
| InterleaveOdd i32 | 1.01= | 1.00= | 0.98= |
| InterleaveUpper i32 | 1.00= | 1.02= | 1.00= |
| InterleaveWholeLower i32 | 1.00= | 1.00= | 0.98? |
| InterleaveWholeUpper i32 | 0.99= | 0.98= | 1.02= |
| IsEitherNaN f32 | 1.00= | 0.99= | 1.00= |
| IsFinite f32 | 1.00= | 1.33W | 1.00= |
| IsInf f32 | 1.00= | 1.13W | 1.00= |
| IsNaN f32 | 1.01= | 1.00= | 1.00? |
| IsNegative i32 | 0.97? | 0.94? | 1.00= |
| Le i32 | 1.00= | 1.00= | 0.98= |
| LeadingZeroCount u32 (4096) | 0.95? | 1.01= | · |
| Load/Store i32 aligned | 0.99= | 1.00= | 0.99= |
| LoadExpand i32 (4096, 50%) | 2.00W | 2.12W | · |
| LoadInterleaved2 f32 (4096) | 1.00= | 1.00= | · |
| LoadInterleaved3 f32 (4096) | 1.00= | 0.99= | · |
| LoadInterleaved4 u8 (4096, RGBA) | 1.00= | 1.00= | · |
| LoadN/StoreN i32 (half lanes) | 1.59W | 1.06W | · |
| LoadNOr i32 (half n) | 1.15? | 1.05? | · |
| LoadU/StoreU i32 | 1.01= | 0.99= | 1.00? |
| Lt i32 | 0.99= | 1.02= | 0.99= |
| Lt128 u64 | 0.99= | 1.00= | 1.00= |
| Lt128Upper u64 | 1.01= | 1.01= | 0.95? |
| Mask And i32 | 1.00= | 1.00= | 1.00= |
| Mask AndNot i32 | 1.00= | 1.01= | 1.00= |
| Mask Not i32 | 1.00= | 0.98= | 1.00= |
| Mask Or i32 | 1.00= | 1.00= | 1.00= |
| Mask Xor i32 | 1.00= | 1.00= | 1.00= |
| MaskedApproxReciprocal f32 | 1.00= | 1.00= | 1.00= |
| MaskedApproxReciprocalSqrt f32 | 1.00= | 0.99= | 1.00= |
| MaskedConvertTo f32-i32 | 2.06W | 1.50W | 0.99= |
| MaskedLeadingZeroCount i32 | 1.00= | 1.01= | 1.01= |
| MaskedLoad i32 (4096, 50%) | 0.98= | 0.99= | · |
| MaskedLoadOr i32 (4096) | 1.01= | 1.00= | · |
| MaskedModOr i32 invariant | 0.97= | 0.98= | 1.06W |
| MaskedMulComplex f32 | 1.20W | 1.09W | 1.00= |
| MaskedMulComplexAdd f32 | 1.14W | 1.02= | 1.00= |
| MaskedMulComplexConj f32 | 1.21W | 1.08W | 1.00= |
| MaskedMulComplexConjAdd f32 | 1.14W | 1.00= | 1.00= |
| MaskedMulComplexOr f32 | 1.16W | 1.07W | 1.00= |
| MaskedMulFixedPoint15 i16 | 0.99= | 1.00= | 1.00= |
| MaskedReduceMax i32 | 1.00= | 1.00= | 1.00= |
| MaskedReduceMin i32 | 1.00= | 1.00= | 1.00= |
| MaskedReduceSum i32 | 1.00= | 1.00= | 1.00= |
| MaskedSatAddOr i16 | 1.00= | 1.01= | 1.00= |
| MaskedSatSubOr i16 | 1.00= | 1.00= | 1.00= |
| MaskedShiftLeft<3> i32 | 0.99= | 0.99= | 0.99= |
| MaskedShiftLeftOr<3> i32 | 1.02= | 0.99= | 1.00? |
| MaskedShiftRight<3> i32 | 0.99= | 0.99= | 0.99= |
| MaskedShiftRightOr<3> i32 | 1.00= | 0.99= | 1.00? |
| MaskedShl i32 | 1.00= | 0.99= | 0.99? |
| MaskedShlOr i32 | 1.00= | 0.99= | 1.00= |
| MaskedShr i32 | 1.00= | 1.00= | 1.01= |
| MaskedShrOr i32 | 0.99= | 1.01= | 1.00= |
| MaskedWidenMulPairwiseAdd i16-i32 | 0.98= | 1.00= | 1.00= |
| Max f32 | 1.00= | 1.00= | 1.06? |
| Max i32 | 1.00= | 1.01= | 1.03= |
| Max128 u64 | 0.99= | 0.97= | 1.00= |
| MaxMagnitude f32 | 1.00= | 0.99= | 1.00= |
| MaxNumber f32 | 0.99= | 1.00= | 0.98= |
| MaxOfLanes i32 | 1.13W | 1.19W | 0.97? |
| Min f32 | 1.01= | 1.02= | 1.05= |
| Min i32 (4096) | 1.00= | 1.01= | · |
| Min128 u64 | 0.98= | 1.02= | 1.00= |
| MinMagnitude f32 | 1.00= | 1.00= | 1.00= |
| MinNumber f32 | 1.02= | 1.00= | 0.99= |
| MinOfLanes i32 | 1.13W | 1.16W | 1.07= |
| Mod i32 invariant | 0.97= | 1.00= | 1.07W |
| Mul f32 | 1.01= | 0.99= | 1.01= |
| Mul i32 | 0.99= | 1.02= | 0.98= |
| MulAdd f32 (4096) | 1.00= | 1.01= | · |
| MulAddSub f32 | 0.99= | 0.99= | 0.97= |
| MulByFloorPow2 f32 | 1.06W | 0.98= | 1.00= |
| MulByPow2 f32 | 1.00= | 1.00= | 1.00= |
| MulComplex f32 | 1.44W | 1.19W | 1.00= |
| MulComplexAdd f32 | 1.02= | 1.01= | 1.00= |
| MulComplexConj f32 | 1.46W | 1.20W | 1.00= |
| MulComplexConjAdd f32 | 1.02= | 1.03= | 1.00= |
| MulEven i32-i64 | 1.00= | 1.02= | 1.00= |
| MulFixedPoint15 i16 | 1.00= | 1.00= | · |
| MulHigh i16 (4096) | 1.00= | 0.99= | · |
| MulHigh u16 (4096) | 1.00= | 1.01= | · |
| MulHigh32 u32 (4096) | 1.00= | 1.01= | · |
| MulOdd i32-i64 | 0.97= | 0.91= | 1.00= |
| MulRound f32 | 1.00= | 0.99= | 1.01? |
| MulSub f32 | 1.00= | 0.99= | 0.98? |
| MulSubAdd f32 | 1.02= | 1.00= | 1.01= |
| MultiRotateRight u64 (4096) | 0.99= | 1.00= | · |
| Ne i32 | 1.00= | 1.00= | 1.00= |
| Ne128 u64 | 0.99= | 1.00= | 0.98? |
| NearestInt f32->i32 | 1.35W | 1.09? | 0.99= |
| Neg i32 | 0.99= | 0.99= | 1.05? |
| NegMulAdd f32 | 1.00= | 0.99= | 0.99? |
| NegMulSub f32 | 1.00= | 0.98= | 0.98? |
| Not i32 | 0.99= | 1.00= | 1.00= |
| OddEven i32 | 0.99= | 1.01? | 1.01= |
| OddEvenBlocks i32 | 1.02= | 0.99= | · |
| Or i32 | 0.99= | 1.03= | 1.01= |
| Or3 i32 | 0.99= | 1.00= | 1.00= |
| OrAnd i32 | 1.00= | 0.99= | 1.00= |
| OrderedDemote2MasksTo i32->i16 (4096) | 0.96? | 1.00= | · |
| OrderedDemote2To i32->i16 | 0.99= | 1.00= | · |
| OrderedRoundingShiftRightAndDemote2To<2> i32->i16 | 1.14W | 1.22W | · |
| OrderedShiftRightAndDemote2To<2> i32->i16 | 1.00= | 0.99= | · |
| PairwiseAdd i32 | 1.10W | 1.00= | 1.00= |
| PairwiseAdd128 i32 | 1.00= | 1.00= | 1.02= |
| PairwiseSub i32 | 1.11W | 0.98? | 1.01= |
| PairwiseSub128 i32 | 1.10W | 1.00= | 1.01= |
| Per4LaneBlockShuffle 3210 i32 | 0.87? | 1.00= | 1.00= |
| PopulationCount u32 (4096) | 1.48W | 1.49W | · |
| PromoteEvenTo i8->i16 | 0.96= | 0.99= | · |
| PromoteLowerTo i16->i32 (4096) | 1.00= | 1.00= | · |
| PromoteMaskTo i16->i32 (4096) | 1.00= | 1.00= | · |
| PromoteOddTo i8->i16 | 0.99= | 0.99= | · |
| PromoteTo i16->i32 1-arg (4096) | 0.99= | 0.99= | · |
| PromoteUpperTo i16->i32 | 1.01= | 0.99= | · |
| RebindMask i32->f32 (4096) | 1.00= | 0.99= | · |
| ReduceMax i32 | 1.17W | 1.02= | 1.00= |
| ReduceMin i32 | 1.17W | 1.03? | 1.00= |
| ReduceSum f32 (4096) | 0.98= | 1.00= | · |
| ReorderDemote2To i32->i16 | 1.00= | 0.99= | · |
| ReorderRoundingShiftRightAndDemote2To<2> i32->i16 | 1.05? | 1.17W | · |
| ReorderShiftRightAndDemote2To<2> i32->i16 | 1.00= | 1.00= | · |
| ReorderWidenMulAccumulate i16->i32 | 1.00= | 1.01= | · |
| Reverse i32 (4096) | 1.00= | 1.00= | · |
| Reverse2 i32 | 1.00= | 1.01= | 1.01? |
| Reverse4 i32 | 1.00= | 1.01= | 1.00= |
| Reverse8 i32 (4096) | 1.00= | 1.01= | · |
| ReverseBits i32 | 2.09W | 1.95W | 1.03= |
| ReverseBlocks i32 | 0.99= | 0.98= | · |
| ReverseLaneBytes u32 (4096) | 1.02= | 1.00= | · |
| RotateLeft<3> i32 | 0.99= | 1.00= | 1.02= |
| RotateLeftSame u32 bits=5 | 1.00= | 0.99= | 0.97= |
| RotateRight<3> i32 | 0.98= | 1.01= | 0.94= |
| RotateRightSame u32 bits=5 | 1.00= | 1.00= | 1.01= |
| Round f32 | 1.00= | 1.00= | 1.04= |
| RoundToInt f32->i32 | 1.38W | 1.14W | 1.00= |
| RoundingShiftRight<2> i32 | 1.08? | 0.99= | 0.99= |
| RoundingShiftRightAndDemoteTo<3> i32->i16 (4096) | 0.97= | 1.11W | · |
| RoundingShiftRightSame i32 bits=5 | 1.10W | 1.00= | 1.46W |
| RoundingShr i32 | 1.48W | 1.15W | 1.00? |
| SatWidenMulAccumFixedPoint i16->i32 | 1.90W | 1.78W | · |
| SatWidenMulPairwiseAccumulate i16->i32 | 1.00= | 1.00= | · |
| SatWidenMulPairwiseAdd u8*i8->i16 | 0.97= | 1.00= | · |
| SaturatedAbs i32 | 1.00= | 1.00= | 1.01= |
| SaturatedAdd i16 (4096) | 1.00= | 1.00= | · |
| SaturatedNeg i32 | 1.00= | 1.02= | 0.99= |
| SaturatedSub i16 | 1.00= | 1.00= | 1.00= |
| ScatterIndex i32 (4096) | 0.99? | 1.00= | · |
| ScatterIndexN i32 (half n) | 1.20W | 1.31W | · |
| ScatterOffset i32 (4096) | 0.99= | 0.99= | · |
| Set i32 | 1.00= | 1.00= | 1.00? |
| SetAtOrAfterFirst i32 | 1.02= | 0.99= | 1.00= |
| SetAtOrBeforeFirst i32 | 0.99= | 1.16W | 1.00= |
| SetBeforeFirst i32 | 1.01= | 1.26W | 1.00= |
| SetMask i32 (4096) | 1.00= | 1.01= | · |
| SetOnlyFirst i32 | 0.92= | 1.65W | 1.00? |
| ShiftLeft<3> i32 | 1.01= | 0.99= | 1.02= |
| ShiftLeftBytes<4> i32 | 1.02= | 0.97? | 1.00= |
| ShiftLeftLanes<1> i32 | 1.00= | 0.99= | 1.00= |
| ShiftLeftSame i32 bits=3 | 1.00= | 1.02= | 1.92W |
| ShiftRight<3> i32 | 1.00= | 0.99= | 0.97= |
| ShiftRightAndDemoteTo<3> i32->i16 (4096) | 1.00= | 0.99= | · |
| ShiftRightBytes<4> i32 | 0.99= | 1.01= | 1.00= |
| ShiftRightLanes<1> i32 | 1.02= | 1.00= | 1.00= |
| ShiftRightSame i32 bits=3 | 1.00= | 1.02= | 1.21W |
| Shl i32 | 1.00= | 1.02= | 0.99= |
| Shr u32 (variable) | 1.00= | 1.01= | 1.00= |
| Shuffle0123 i32 | 1.00= | 0.98= | 1.04= |
| Shuffle0321 i32 | 1.00= | 1.00= | 0.97= |
| Shuffle1032 i32 | 1.00= | 0.98= | 1.02= |
| Shuffle2103 i32 | 1.01= | 1.01= | 1.09? |
| Shuffle2301 i32 | 1.00= | 1.00= | 0.96? |
| Slide1Down i32 | 1.00= | 0.98? | 1.02? |
| Slide1Up i32 | 1.00= | 0.99= | 1.02= |
| SlideDownLanes amt=2 i32 | 0.95= | 1.01= | 1.00= |
| SlideMask1Down i32 (4096) | 1.08W | 1.01= | · |
| SlideMask1Up i32 (4096) | 1.06W | 1.00= | · |
| SlideMaskDownLanes i32 amt=2 (4096) | 1.11W | 0.99= | · |
| SlideMaskUpLanes i32 amt=2 (4096) | 1.10W | 1.00= | · |
| SlideUpLanes amt=2 i32 | 1.00= | 0.98= | 0.99= |
| Sqrt f32 | 1.00= | 1.00= | 1.00= |
| StoreInterleaved2 f32 (4096) | 1.00= | 1.01= | · |
| StoreInterleaved3 f32 (4096) | 1.03= | 0.91? | · |
| StoreInterleaved4 f32 (4096) | 1.48W | 1.01? | · |
| Sub f32 | 1.00= | 1.00= | 1.02? |
| Sub i32 | 1.01= | 1.03= | 1.03= |
| SumOfLanes i32 | 1.05= | 1.16W | 0.97= |
| SumOfMulQuadAccumulate i8->i32 | 1.00= | 1.02= | · |
| SumsOf2 u8->u16 | 0.98= | 1.00= | · |
| SumsOf4 u8->u32 | 0.99= | 0.98= | · |
| SumsOf8 u8->u64 | 1.00= | 1.01= | · |
| SumsOf8AbsDiff u8->u64 | 1.05? | 0.99= | · |
| SumsOfAdjQuadAbsDiff u8->u16 | 1.00= | 1.00= | · |
| SwapAdjacentBlocks i32 | 0.98= | 1.01= | · |
| TableLookupBytes u8 (4096) | 1.00= | 1.00= | · |
| TableLookupLanes i32 (4096) | 0.99= | 1.01= | · |
| TestBit i32 | 1.05W | 1.00= | 1.00? |
| TrailingZeroCount i32 | 0.99= | 1.00= | 1.04= |
| Trunc f32 | 1.00= | 1.01= | 1.07? |
| TruncateTo u32->u16 | 1.01= | 1.01= | 1.00= |
| TwoTablesLookupLanes i32 (4096) | 1.01= | 0.98= | · |
| WidenMulAccumulate i16->i32 | 1.00= | 0.99= | · |
| WidenMulPairwiseAdd i16->i32 | 1.00= | 0.98= | · |
| Xor i32 | 0.99= | 1.01= | 1.02= |
| Xor3 i32 | 1.63W | 1.00= | 1.04? |
| Zero i32 | 1.00= | 1.00= | 0.95= |
| ZeroExtendVector i16 half->full | 1.02= | 1.00= | · |
| ZeroIfNegative f32 | 1.02? | 1.09W | 1.01= |

## 复现
```
# 每平台:构建 hyle_op_parity{,_full,_widen}_bench_<isa>,双序各跑一遍(Mac 建议 4 遍),然后
python scripts/op-parity-report.py out_hf.txt out_hwf.txt          # x86(闸门 ±4%)
python scripts/op-parity-report.py --gate 0.06 mac_*.txt           # Mac(无亲和性,±6%)
# 报告首段为锚点校准;CALIBRATION PASS 之后的判决才可引用。
```
