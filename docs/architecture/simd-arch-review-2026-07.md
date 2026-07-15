# HYLE SIMD 架构评审(2026-07)

> **结论先行**:架构没有系统性问题;有**两处真实的架构级债**——①定宽后端表示层丢 N(= simd-19,
> **2026-07-15 NEON 侧已根治**;x86 残部=脆弱 TIE,见路线 3)、
> ②手写 op 面的横切成本(simd-19 是它第一次真正收费)。两处都有不推倒重来的改善路径。**不需要推倒重来,
> 也不该走向 vendor Highway。**
> 背景:一个月 HYLE-vs-Highway 性能对齐(见 `docs/milestones/simd-orderswap-matrix.md`)后的架构侧复盘。

## 一、判断依据(架构质量怎么测,不靠感觉)

在校准仪器(锚点校准 + 双序判定 + 三平台,见 `docs/performance/simd-op-benchmark-methodology.md`)下:

- 299 个已 bench op 中 **291 个 SOTA**(与 Highway 平或胜);MSVC / GCC 真输**清零**;NEON 剩 2——
  一个是 PromoteMaskTo 0.575(即债①的量化样本),一个是 ScatterIndexN 0.87(编译器寄存器类微差,非算法)。
- 推断逻辑:**如果地基是歪的,逐 op 修不可能把表打平到这个程度**。~20 轮修复全部是 op 级(指令选择、编译器
  fork、契约收窄),没有一轮需要动 tag 系统、分派结构或测试基建——能打平,说明主体结构是对的。

## 二、两处架构级债

### 债① 表示层丢 N(simd-19;**NEON 侧已根治,2026-07-15 收口**)

类型系统携带 N(`Rebind`/`Repartition` 产生 partial tag),scalable 后端(SVE/RVV)用 `kPow2`+`vl` 正确
执行了它,但**定宽后端的物理表示曾不携带 N**——修复前 `VecNeon<int16_t,4>` 物理上仍是 `int16x8_t`,
op 被迫满宽做再掰回。这是设计时漏掉的一个维度,是缺陷不是权衡。

- **NEON:已根治**(simd-19,当日立项/施工/收口)。`RawNeonNT<T,N>` D 档表示 + 常绿垫片迁移 + ~17
  热点 native-D;PromoteMaskTo 0.575→TIE(C12 单 op 双序 + C18 全量双证,反汇编与 Highway 同形),
  全量复跑零回归,顺带修复两个既有 bug 家族(partial LoadU/StoreU OOB、子满宽 half/pad N-盲语义)。
  全记录:`docs/milestones/simd-19-partial-vector-repr.md` §八、`simd-19-ledger.md`。
- **x86**:同样丢 N,但 x86 没有比 xmm 更窄的寄存器,所以那边的正确形态**不是"D 表示"而是"partial 时选
  消费低半的指令"**(`cvtepi16_epi32` 天然只吃低半)。当前 x86 partial 路径的 TIE 很多是编译器碰巧优化出来
  的——属**脆弱 TIE 而非健康 TIE**,编译器升级可能悄悄回归。处置见 §四-3。

### 债② 手写 op 面的横切成本

每 op 手写 `if constexpr` + 硬编满宽 intrinsic,是当年拿扩展性换可读性/可断点/贴硬件的**有意权衡**(章程
"不 vendor Highway 宏体系")。simd-19 是这个权衡第一次真正收费:横切维度一变(加 D-form),270 op 全要碰。
今后每加一个新维度(f16/bf16 lane 类型、SVE2、AVX10)都会再收一次费。

这个债**不用还清**(还清 = 走宏生成 = 丧失可读性,违背章程),但需要一层收敛结构:**宽度分发 helper**
(`neon_bin<QFn,DFn>` 模式)——op 语义仍手写,"类型→intrinsic 选择"收敛到 helper,未来新维度只改 helper 层。
simd-19 阶段 1 正好是它的落地场。

### 债② 追问:Highway 宏体系是不是"正确答案"?HYLE 不走它会不会持续出问题?(2026-07-15)

数据(本仓 `third_party/highway/hwy/ops/arm_neon-inl.h`,10982 行):`HWY_NEON_DEF_FUNCTION*` 宏调用
286 处,`template <` 966 处——**Highway 的 NEON 后端也只有规整那部分由宏生成,其余大量仍是手写特化**。
宏解决的是"规整 op × (type×size) 矩阵的机械展开";难的那部分(本月 ~20 轮修复对应的指令选择、编译器
怪癖、契约边界)两家都得手写。

宏体系对 Highway 是对的,因为它的**结构**需要:一份 op 源码经 `foreach_target.h` 对每个 target 反复
re-include 展开(x86 128/256/512、NEON、SVE、WASM…),没有宏就是复制地狱。HYLE 结构不同:每 ISA 一个
后端头、每 target 一个 TU,不存在同源多 target 展开需求;C++20 `if constexpr` + 模板 helper 能以可读、
可断点的形式覆盖同样的分派。(行数对照:HYLE neon.h 3297 行 vs Highway arm_neon-inl.h 10982 行——后者
覆盖面更宽(armv7、f16/bf16、全 partial 档),不是纯赘肉,但说明宏并没有让后端变小。)

不走宏的真实风险 = **横切新轴时的一次性迁移成本**,且轴可枚举:
1. partial size——本里程碑(simd-19),一次垫片迁移 + ~20 热点升级;
2. 新 lane 类型(f16/bf16)——将来若做,helper 层落地后成本可控(只改"类型→intrinsic 选择"一层);
3. 新 ISA 代际(SVE2/AVX10)——两个世界成本相同,Highway 也要写新 -inl.h。

且有**检测器兜底**:校准 bench 会把轴缺失暴露为量化输局(PromoteMaskTo 0.575 正是这样被抓住的),
不会静默腐烂。结论:宏体系不是普适正确答案,是 Highway 结构约束下的正确答案;HYLE 的对应答案是
helper 层 + 校准闸门。持续风险有界、可检测、有着力点。

## 三、什么不是架构问题(防误诊)

- **MSVC 复合上下文劣化 / 寄存器类微差**:编译器层的事,fork 纪律(`#if _MSC_VER && !__clang__`,GCC-反赢
  陷阱教训)已是正确应对;不存在架构级修法(除非生成 asm)。
- **覆盖面缺口**(masked arith 族等 ~41 个待补 bench + P1 名单):广度债,清单在
  `docs/milestones/simd-gap-inventory.md`,与结构无关。
- **API 镜像 Highway**:是使能逐-op 对拍的**选择**,不是被动跟随的证据——InRange 契约族(TableLookupBytes
  等)证明能在镜像之上有节制地做得更好。代价是继承了 Highway 的 op 面设计决策(如 mask 宽度桥接族的存在),
  可接受。

## 四、改善路线(按 ROI 排;状态更新 2026-07-15 收口)

1. **simd-19 常绿落地** ✅ **完成**。表示层补 N 维(`RawNeonNT<T,N>` D 档),常绿垫片迁移全程兑现
   (W0a/b/c 每步绿,~17 热点 native-D);PromoteMaskTo 0.575→TIE(C12+C18 双证),全量复跑无回归,
   neon.h +21%(Highway 的 36%,警戒线未触发)。**债①在 NEON 侧根治**;x86 侧的对应形态(脆弱 TIE)
   见路线 3,不变。
2. **宽度分发 helper 层——被实践修正,未建且暂不建**。simd-19 实际用"垫片(to_q/from_q)兜底 + 热点
   op 内联 `if constexpr (kIsDBand)`"替代了设想中的 `neon_bin<QFn,DFn>` 层:宽度轴上两形指令仅差 q
   后缀,内联分支更贴文件风格、无额外抽象。**修正后的判断**:垫片只解宽度轴;未来的**类型轴**
   (f16/bf16——全新指令,垫片无从兜底)才是债②的真考验,收敛层在那时按量化需要再建,现在建属
   过度抽象。simd-19 留下的真资产是**已实证的横切迁移方法**(脚本包裹 + 常绿三步 + 工单制:270 op
   规模一天完成)——债②本金未还,利率大降。
3. **x86 partial 健康化(审计 + canary,不做表示改造)**。把 Rebind 产物路径上靠编译器运气的脆弱 TIE 找出
   来,显式改为低半消费指令,并加 bench canary 防编译器升级回归。成本远低于表示改造,收益是消掉"未量化的
   同类雷"。
4. **按 gap-inventory 补 P1 覆盖面**(masked arith 族 ~28 名)。新 op 一律过校准仪器 + oracle 闸门再进表。
5. **把测量体系当架构质量的常设准绳**。锚点校准、双序判定、三平台 gtest、复合上下文 canary——这套仪器是
   这轮对齐战最大的资产;"架构好不好"从此靠它读数,不靠感觉。

## 五、底线判断

simd-19 已落完(helper 层按路线 2 的修正暂不建,类型轴到来时再议),HYLE simd 的形态(**手写可读 op 面
+ 类型级 tag + 校准闸门**)长期站得住:
新增 op 有方法论文档护航,新增维度有 helper 层着力,性能回归有仪器兜底。推倒重来没有依据;vendor Highway
只在触发 §5.2 退出条件时才重新讨论。

## 六、参考

- 对齐战全记录与真输清单:`docs/milestones/simd-orderswap-matrix.md`
- 债①修复里程碑:`docs/milestones/simd-19-partial-vector-repr.md`
- 覆盖面清单:`docs/milestones/simd-gap-inventory.md`;复杂度机会:`docs/milestones/simd-complexity-opportunities.md`
- 测量方法论:`docs/performance/simd-op-benchmark-methodology.md`
- 仪器伪影狩猎记录:`docs/debug/op-parity-harness-artifacts-hunt.md`
- 章程(不 vendor / 行数警戒线):`docs/architecture/engineering-charter.md`,`simd-design.md` §229
