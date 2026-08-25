# op_parity 大面积 0.xx 之谜（三层测量伪影 —— 已定位并重建基准）

> 调查日志（已结案）。关联:[`../milestones/simd-orderswap-matrix.md`](../../implementation/tasks/simd/simd-module-closure.md)（本次调查后全量重写）、`bench/simd/op_parity_engine.h`（重建后的测量引擎）、`scripts/op-parity-report.py`（聚合+校准闸门）。
> 结论先行:op_parity 老 harness 报的 ~40-50 个 "0.xx"（MSVC 列集中）**绝大多数是测量伪影,不是 HYLE 真慢**。伪影由三层独立因素叠加,全部用受控实验坐实:①不钉核时 13900K P/E 核迁移(±10~30% 随机);②同一 TU 内每个被测循环的**代码放置运气**(uop-cache/取指几何,±10~15%,尾部 ~30%,逐指令相同的循环也中招,换序不能抵消,每次重编译重掷);③kernel lambda 每迭代重载闭包捕获链 → **闭包栈地址与 store 流 4K 别名**(~25%,由启动器的环境块大小决定,换启动器就翻车)。修法不是贴标签,而是重建仪器:钉核+每侧 4 个独立代码副本采样放置+指针外提斩断捕获链+宏循环 4× 展开+**"已知真值=1.00"锚点校准闸门**。重建后锚点从 0.75~1.49 收敛到 **0.99~1.03(双启动器/双序一致)**,三平台重测:真输远少于旧报告(x86 两平台交集仅 3 个,详见重写后的矩阵文档)。

> **2026-07-16 仪器 v3**：保留上述已验证的 pin/replica/stagger/指针外提/anchor 机制，不推倒重来。
> 报告器新增 run metadata、R/R+8 双 layout-bank 认证、每文件 anchor 中位归一、paired replica
> bootstrap 95% CI，以及 `INCONCLUSIVE` 状态；校准失败会真正抑制所有 verdict。权威报告要求双序各
> 至少两份，避免再把一个 0.93 点估计强行解释成 TIE 或 LOSE。实现与规程见当前
> `op_parity_engine.h`、`op-parity-report.py` 和 benchmark methodology。

> **2026-07-16 v4 复审（simd-22 后）**：v3 的“16 个 paired placement samples”仍把同一
> 二进制内的 replica 当成独立样本，CI 会过窄。决定性反例是 MSVC `Not i32`：报告稳定给
> 0.962 LOSE，但 HYLE/Highway 两个热循环逐指令完全相同（同一 `vpcmpeqd` 常量和 4×
> `vpxor/load/store/loop-control`）。v4 改为按完整 run 分簇的层级 bootstrap，并以单侧 replica
> spread 推导 `place_u`；只有整个 CI 落入等价带才叫 TIE。引擎 metadata 同时记录 affinity/
> priority 返回值、实际逻辑 CPU 和 Windows efficiency class；绑定失败或混核会使认证失败。
> 复测还清掉一个 v3 残留：anchor gate 曾同时检查 paired 和 `best-of-replicas`，但 best 本身正是
> 固定代码放置的函数；Mac `ShiftLeft<3>` 出现 paired=1.028、median=1.014、best=1.075，且单侧
> replica spread=1.267。继续跑同一二进制不会重掷 placement。v4 现只用 raw paired anchor 检查
> run-wide H/G 偏斜；best/spread 继续显示，且每个 op 的 verdict 仍受 hierarchical CI、双序和
> `place_u` 三重约束。这是移除自相矛盾的校准条件，不是放宽 LOSE/TIE 判决。
>
> 本轮还抓出一个与计时无关但同样致命的构建伪影：Windows performance tree 曾在中文
> `/showIncludes` 下 configure、英文 DevShell 下 build，Ninja 的 `msvc_deps_prefix` 不匹配，导致
> SIMD 头文件改动不触发重编译。已规定新鲜 `build-bench-v3` 全程 `VSLANG=1033`，并核对 obj/exe
> 时间戳和真实编译日志。

## v4. simd-22 后复测：LOSE 必须经过形状、核心与代码生成三重拆解

- **`SatWidenMulAccumFixedPoint` 的 0.08 不是算法地板**：simd-22 表示层重构后 AVX2
  `ResizeBitCast` 退化为 store/memcpy/reload，修为寄存器 cast/extract/insert 后 Windows/WSL 回到
  TIE；测试 oracle 也加强为真正的定点饱和参考。
- **`FindKnownFirstTrue` 是测试形状错误**：旧 kernel 在闭包里反复 `.data()`，且测成
  `FirstN+Or` 的合成路径，不是直接 op。改为指针外提和直接调用后 TIE。
- **`CompressNot` 的 MSVC ~0.946 是可修的真实差距**：HYLE 用 `mbits ^ 0xFF` 复用 Compress
  LUT，Highway 用独立 Not LUT，少一条标量 xor；补独立 1 KiB LUT 后两 x86 平台 TIE。
- **`Mod i32 invariant` 的 0.93 最终不是地板**：HYLE `Divisor` 的字段布局与 Highway 不同，且
  signed `Div` 把不变的 `dsign` 广播成向量后做 xor/sub，扩大了 `MulHigh32` 周围的寄存器活跃区；
  MSVC 因而多一条 ymm copy/保存。字段顺序与 Highway 对齐，并利用 `dsign` 只可能为 0/-1，在标量
  invariant 分支中选择 `q`/`Neg(q)`，消掉广播和两条向量 op。另一个独立问题是旧“寄存器 sink”
  每个 vector 都调用一次外部 `doNotOptimizeAwaySink`，调用成本淹没 op；现改为两侧对称 xor 累积，
  循环后只 sink 一次，同时保留真实 load→Mod→store 形状。Windows 定点复测：Div **1.153**、Mod
  累积 **1.154**、Mod store **1.119**，均不再低于 Highway。教训是形状问题与真实 codegen 问题可以
  同时存在，不能因前者就停止逐条汇编调查。
- **`ScatterIndexN` 的 Windows/NEON 真输已清零**：泛型实现物化两个 aligned scratch 数组；MSVC
  拒绝内联后产生 208-byte 栈帧和调用，full 一度只有 0.24–0.33。AVX2 i32 full 改成 runtime count
  switch + compile-time lane extraction，热路径全寄存器，full **0.995**、half **1.260**。AArch64 没有
  scatter 指令，但机械 switch 仍不如 Highway；按其 `generic_ops-inl.h` 精确采用固定四次循环和
  `i < max_lanes` 谓词后，AppleClang 对 half 标量化成相同的 `ldp/ldpsw/str`，逐条汇编与 Highway
  同形，n=1/half/full 分别 **0.997/0.997/0.998**。无需 inline asm，也不再接受“编译器调度地板”。
- **`SetOnlyFirst` 0.933 是 MSVC 特有的表示往返**：原实现 mask→GPR→`countr_zero`→
  `LoadMaskBits`。在 MSVC 上改为 `suffix=SetAtOrAfterFirst(m)`、再以
  `suffix & ~Slide1Up(suffix)` 隔离第一个 true，结果 **0.998**。同一向量式在 GCC 反而约 0.96，
  所以只在 MSVC 选择它，GCC 保留原 scalar-bits 路径并维持 **1.63 WIN**。机械照搬 Highway 的
  half-vector Combine 形状在 HYLE width-banded 表示下曾退到 0.107，已回滚；参考的是算法/汇编形状，
  不是无条件复制源码。
- **`IsFinite f32` 0.964 是漏掉的 Highway 编译器特化**：HYLE 的 Abs+FP-compare 在 MSVC 较慢；
  按 Highway AVX2/MSVC 路径用整数左移清 sign、右移提 exponent、signed compare 后复测 **0.993
  TIE**。GCC 保留原 FP 路径，避免把另一编译器的赢面拉回去。
- **Mac `Mod i32 invariant store` 0.932 是两个 x86 修复不能跨平台照搬的反例**：逐条 probe 汇编
  显示 HYLE `MulHigh32(a, Set(multiplier))` 丢失标量广播信息，成为
  `ld1r+smull+ext+smull`，Highway 是 `ldr d+smull lane+smull2 lane`；同时 MSVC 受益的 scalar
  sign 分支在 AppleClang 变成 `cmeq+dup+neg+bit`，比 Highway vector sign 的 `dup+eor+sub` 多一条。
  NEON 现从 `Divisor::multiplier_` 地址安全加载相邻 64 bit、以 lane 0 做两次 widening multiply，
  并恢复 vector sign；x86 仍保留 scalar sign。定点双序四文件：Mod 累积 **0.993**、真实 store
  **1.080**、MaskedModOr **0.994**，均无 LOSE；279 个 NEON 测试通过。
- **所有最终报告中 ratio `<0.98` 的行都要过一眼，但不等于都要改**：本轮剩余低值分成两类：
  `Not`/`IfThenElseZero`/`IfThenZeroElse` 等两侧热循环逐条相同或同一条 intrinsic；Concat/MulOdd/
  Interleave 等自身 replica spread 很大且 CI 跨过等价带。它们记录为 placement/inconclusive，不能
  以点估计冒充 LOSE，也不能为了“好看”改报告口径。

## 0. 背景
op_parity 三个 bench(`op_parity_bench/full/widen`)逐 op 对拍 HYLE vs Highway(同 TU 同 ISA 双方都内联),报 ratio=HYLE÷Highway。旧数据显示 ~40-50 op 在 0.xx,集中在 MSVC 列。头号疑点:`Neg i32` 两侧都是单条 `vpsubd`(源码级确认 `_mm256_sub_epi32(zero,v)`),真值必须=1.00,却被判 0.91。一条指令不可能算法输 → harness 有系统性偏差,但根因从未被坐实。

## 1. 复现:数字不是代码的属性,是"那次编译+那次调度"的属性
同一 op_parity_full_bench 二进制(2026-07-13 HEAD),同机复跑:

| 条件 | Neg i32 ratio |
|---|---|
| 旧报告(历史二进制) | **0.91**(双序) |
| 今日二进制,不钉核,3 跑 | **1.33 / 1.27 / 1.49**(方向反转且乱跳) |
| 今日二进制,钉 P核+High,3 跑 | **1.27 / 1.32 / 1.30**(稳定但≠1.00) |

两个事实:①不钉核时逐跑乱跳(13900K 混合架构,P/E 迁移);②钉核后**稳定在 1.29**——既然稳定,就不是噪声,而是这个二进制里两个循环真的一快一慢。但真值必须是 1.00 → 二进制本身携带偏差。

## 2. 误诊诚实记录
- **前一 session 的"隔离复核"**(~45-op 小 TU + Neg 锚点 + streaming kernel)结论"36/43 是伪影、真输 7 个"——方向大致对,但**它自己的锚点读 1.19(MSVC)/1.38(WSL)**,离真值 1.00 差 19~38%,是一台没校准、反向偏 HYLE 的仪器。其 WIN/LOSE 标签在 ±20% 内不可信(例:LoadNOr 被它"证伪不是真输",本次校准仪器实测 WSL 0.54 真输;BroadcastLane 被它判 1.11-1.51 WIN,实测 MSVC 0.46 真输)。
- **本次调查中我自己的两次翻车**(都被校准闸门当场抓住):①PS 包装器在进程启动后设 `ProcessorAffinity`,把 exe 内 `pin_and_boost()` 刚设的线程亲和性**覆盖重置**(Windows 语义:设进程掩码会重置所有线程掩码),那一轮等于没钉核;②给宏循环做 4× 展开后直跑 bash 一切正常包装器却翻车——差点误诊为"展开破坏对称性",实为第三层伪影(§5)现形。
- **旧 harness 注释里的理论**("icache/DSB 压力"+"系统性偏向先跑的槽")半对半错:压力说法不准(测量时只有一个循环在热),"偏向先跑槽"被 SELF=1 实验直接证伪(见 §4)。

## 3. 反汇编铁证:两个循环逐指令等价,却稳定差 29%
dumpbin 全量反汇编,按"vpxor+vpsubd+vmovdqu"签名定位两个 Neg 循环(lambda 被外联成独立 `Bench::run<Op>` 实例,链接序≠源码序):

| | 循环 A @0x2AD60 | 循环 B @0x2E700 |
|---|---|---|
| 指令数 | 11 | 11 |
| 序列 | vpxor/双 mov 捕获链/vpsubd(折叠 load)/双 mov/vmovdqu/2×lea/cmp/jbe | 同(仅 vpxor 位置与目的寄存器号不同) |
| 循环顶对齐 | 32B 对齐 | 32B 对齐 |
| mod 64 相位 | 0x20(体跨 64B 线) | 0x00(体不跨) |

指令、对齐、mod-32 几何全同,唯 64B 相位不同 → 稳定 1.29 只能来自**代码放置**。

## 4. 决定性实验矩阵(一次一个变量)
| 实验 | 操作 | 结果 | 判读 |
|---|---|---|---|
| 换序 | 同二进制,`HYLE_BENCH_HWYFIRST=1`,钉核 | Neg 仍 1.31/1.26 | 偏差跟代码实例走,**不跟槽位/时序走** |
| MASM 相位探针 | 把 Neg 循环 11 条指令逐字节复制 8 份,唯一变量=循环顶 mod 64 相位(0,8,...,56),RDTSC 计时 | 1.65~1.90 tsc/iter(**±15%**) | 纯代码相位就能造成两位数吞吐差(Raptor Cove 前端对放置敏感);相位→惩罚映射非简单"跨线"规则 |
| SELF=1 | 两槽跑**同一个** lambda(同代码地址),全 TU 206 样本,钉核 | 中位 1.000,范围 [0.935,1.059],>10% 零个 | **槽位/顺序/warmup/时序无罪**(±2~3%) |
| SELF=2 | 两槽跑**文本相同但独立实例化**的两个 lambda(不同地址),203 样本 | 范围 [0.906,1.145],>5% 13 个,且 MulSub 0.92/0.91、NegMulSub 0.93/0.92 **跨运行可复现** | **代码放置注入稳定 ±10~15%,尾部更远**;每次重编译重掷(旧 0.91 vs 今 1.29 同源) |

## 5. 第三层伪影:闭包栈地址 4K 别名(换个启动器同一 exe 就翻车)
重建初版(钉核+副本+展开)在 PS 包装器下锚点全过,**直接从 bash 跑同一 exe 却 Neg=0.75~0.86**(双序一致、副本内散布小=系统性而非噪声),且探针证明 bash 路径钉核成功。差异只剩启动环境。机制:kernel lambda 每迭代执行 `[rsi+8]→[rcx]→data` 捕获链重载(MSVC 无法证明 store 不改写闭包,不敢外提);闭包在 main 栈上,**栈布局由环境块大小决定**(bash 与 PS 的 env 不同)→ 某启动器下某一侧闭包字段与 store 流页内偏移 4K 对齐冲突 → 该侧每迭代吃 store-forward 假依赖。四个副本共享同一闭包对象,所以副本间一致、只打一侧、换启动器重掷——全部对上。
**修**:所有 kernel 把 `X.data()` 外提为 lambda 内局部指针(局部变量地址不逃逸,编译器可证 store 打不到它 → 全程驻留寄存器)。修后 bash/PS 两启动器锚点同过(0.99~1.00),**启动器无关性达成**。

## 6. Mac(Apple Silicon)补丁:epoch 太短撞环境定时器
NEON 列初跑锚点 0.89~1.40、单测量内 err% 高达 28%(Windows 同指标 0.3~1.6%)。QoS 探针证明 USER_INTERACTIVE 生效且 CPU 稳(±3%),问题是 ~1.6ms 的 epoch 会被驻留应用的 ms 级定时器爆发打中(macOS 无亲和性 API 可躲)。修:引擎按平台放大 epoch(`__APPLE__` 默认 8×,env `HYLE_BENCH_EPOCH_SCALE` 可调)。

## 7. 修法(重建仪器,不贴标签)
`bench/simd/op_parity_engine.h` + 三个 TU 全量改造 + `scripts/op-parity-report.py`:
1. **进程内钉核**:`pin_and_boost()` = 一颗 P 核亲和 + HIGH 优先级(Linux sched_setaffinity;macOS QoS)。杀伪影①。
2. **放置采样**:每个 kernel 是泛型 lambda `[&](auto rep){...}`,引擎按 `RepTag<SITE,R>` 实例化 **每侧 4 个副本**(不同代码地址),H/G 交替轮次;`hyle_bench::mark(rep)` 把全局唯一立即数烙进每个实例(防链接器 ICF 把副本折叠回同一地址)。聚合取 best-of-replicas(≈放置理想值)+median 双估计。杀伪影②。
3. **指针外提**:kernel 内 `auto* p_X = X.data();`,热循环零捕获链访问。杀伪影③。
4. **宏循环 4× 展开**:3 指令级微循环处于前端放置超敏感区(±50%),展开后循环体离开该区,副本散布从 1.5 收敛到 ≤1.1。
5. **锚点校准闸门**:源码级验证同 intrinsic 的 op(Neg/Sub/Mul/Max/ShiftLeft i32、Add/MulAdd f32、Min i32、SaturatedAdd i16)真值=1.00,报告先验锚点,任一超出 [0.96,1.04] 即整表判 UNCALIBRATED 拒绝输出判决。资格教训:TableLookupBytes 曾按 bench 注释当锚点,实际 HYLE 多一条设计内的 vpand 索引掩码 → 不合格,已剔除(它是真实小代价,保留为被测 op)。
6. **判决口径**:LOSE/WIN 需 best 与 median 两个估计一致越过品位线,且品位线随该 op 实测副本散布放宽(`u=max(3%, (maxspread-1)/2)`);多输入文件(两序/多跑)判决不一致 → UNSTABLE。

## 8. 验证(校准闸门读数)
| 仪器状态 | Neg i32(真值 1.00) | 锚点整体 |
|---|---|---|
| 旧 harness 不钉核 | 0.91(历史)/1.27~1.49(今) | — |
| 旧 harness 钉核 | 1.29 稳定 | — |
| 前 session"隔离复核" | 1.19(MSVC)/1.38(WSL) | 未校准,反向偏 |
| **新引擎 MSVC** | **1.00**(直跑)/1.00(包装器) | 5/5 锚点 0.99~1.03 PASS(±4%),双序 |
| **新引擎 WSL/GCC** | **1.00** | 5/5 锚点 0.99~1.01 PASS(±4%),双序 |
| **新引擎 Mac/NEON** | **0.99** | 5/5 锚点 0.95~1.05 PASS(±6% 闸门;无亲和性 API,4 运行合并+8× epoch) |

## 9. 新结论(可信真慢清单)
见重写后的 [`../milestones/simd-orderswap-matrix.md`](../../implementation/tasks/simd/simd-module-closure.md)(唯一权威表)。要点:旧报告"40-50 个 0.xx"塌缩为 **MSVC 21 / WSL 11 / Mac 3 个 LOSE**;头部(ReverseLaneBytes 0.05、Broadcast 家族 0.46-0.52、TruncateTo 0.83-0.85)经源码核证=store+reload 兜底实现,真算法输;x86 双编译器交集 5 个(DemoteMaskTo、DemoteToNearestInt-safe、HighestSetBitIndex、TrailingZeroCount、TruncateTo),其中 TrailingZeroCount 三平台全输(0.87/0.87/0.68)。旧"隔离复核"对 LoadNOr 的免罪被推翻(WSL 0.54 真输)。

## 9b. 仪器 v2:副本相位相关漏洞(修复后 BroadcastLane 同文本 0.898→1.000)
v1 仪器交付后,修 op 时发现残余:BroadcastLane<3> 把 HYLE 改成与 Highway **逐字节相同**的序列后 full_bench 仍稳定 0.898 LOSE——"整表校准 PASS"竟不保证单 op 无偏。反汇编定位到精确根因:**同侧 4 个副本的放置是相关的**。同一 op 的副本实例尺寸相同,编译器以恒定步长连续排布;当步长是 64 的倍数(HYLE vpermd 副本步长 0x280),4 个副本的循环顶共享**同一个** mod-64 相位(实测全在 32)= 4 份同一次掷骰;而 Highway 侧步长 0x2D0(720 mod 64=16)恰好扫过 48/0/16/32 四个相位。best-of-replicas 于是系统性偏向"步长打破 64 对齐"的一侧(~10%),且副本内散布极小(同相位)骗过散布感知判决。锚点大multi数能过是因为两侧要么都扫要么都不扫——只在一扫一不扫时咬人,由实例尺寸 mod 64 抛硬币决定。
**修**:`hyle_bench::stagger(rep)`(强制内联)作为每个 kernel 的第一条语句,replica k 垫 ~16k 字节直线代码 → 两侧都确定性扫过 {0,16,32,48} 四相位(反汇编确认 vpermd 副本相位从 {32×4} 变为均匀 {0,16,32,48})。
**验证**:①锚点仍 0.99~1.02 PASS;②决定性——同文本双槽 BroadcastLane(用户实验的受控重演)从 0.898 → **1.000 TIE**;③相位敏感 op 的副本散布如实变大(如 1.33),散布感知判决自动拒绝为其发 LOSE。附带教训:scratch 复验构建漏 `/DNDEBUG` 会让 Highway 断言全开、锚点飙到 1.7-2.5——校准闸门连"构建旗标错了"也能当场抓住。
**残余分辨率**:4 副本×16B 步进覆盖 64B 相位空间的 4 点格;两侧基相位差非 16 倍数时格点不重合,单 op 残余 ≲4%,落入 TIE 带宽,不产生假判决。要更细可升 8 副本×8B(运行时×2,当前判为不值)。
**GCC 侧补记(2026-07-14)**:objdump 证实 GCC 把**所有**被测循环顶用 `.p2align` 钉到 64B 对齐(FindLastTrue 家族 32 个循环 phase64 全 =0)——stagger 在 GCC 上被对齐 padding 完全吸收,等效 no-op;但统一对齐本身就是放置均衡器(两侧同相位=公平),这解释了 GCC 列一贯比 MSVC 干净。
**上段的 FindLastTrue"仪器残余"定性后被推翻(同日复审)**:当时的"两侧指令相同"是**向量签名级**对比的假象——规范化逐条 diff 后发现两侧循环 14 vs 15 条,HYLE 多一条 movslq(`31 - std::countl_zero` 在 int 域做减法、进 64 位 sink 前的符号扩展;Highway 的 Num0Bits* 是 size_t 宽)。改 ptrdiff_t 域减法后 0.94→1.00 TIE。教训:**判"逐指令等价"必须规范化后逐条 diff(含标量尾巴的寄存器宽度),签名级对比会把一条指令的差距误判成测量噪声**;"稳定的单编译器 5-6%"值得默认怀疑是真的一条指令,而不是先怀疑仪器。

## 10. 教训
1. **仪器必须自带"已知真值"校准通道**:对拍基准里放几个逐指令相同的锚点 op,锚点不读 1.00 就没资格谈别的数字。前 session 的隔离复核有锚点却接受了 1.19——有锚点不看等于没有。
2. **代码放置是一等公民噪声源**:同一二进制里逐指令相同的两个循环能稳定差 29%,换序/多跑/中位数都救不了(它是确定性偏差不是噪声);唯一出路是放置采样(多副本)或让循环体离开超敏感区(展开)。
3. **微基准 kernel 必须指针外提**:让编译器把"每迭代重载捕获链"优化掉不是它的义务;栈-堆 4K 别名让"同一 exe 换个启动器数字就变"这种鬼故事成为现实。
4. **"稳定复现"≠"测得对"**:钉核后 1.29 三跑一致、双序一致,一切稳定指标都绿,但它测的是放置运气不是代码速度。稳定性检验永远替代不了真值检验。
5. 修仪器,别给脏数字贴解释标签——本次把标签体系连同旧数字全部推翻,数字直接可信后标签自然消失。
6. **副本采样必须保证采样点独立**:同尺寸实例等差排布会让"N 个副本"退化成"同一个放置抽 N 次"(步长 mod 64=0 时),副本内散布小反而是坏信号;确定性相位错位(stagger)比指望链接器"自然随机"可靠。"整表校准 PASS"只约束锚点集合,单 op 判死刑前若有疑问,用"同文本双槽"实验做该 op 的专属零假设(v2 后一般不再需要)。
