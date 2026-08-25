# simd-19: 定宽后端的部分向量表示(partial-vector representation)

> **状态**:**完成 ✅**(2026-07-15 当日立项/施工/收口;验收记录见 §八)。触发自 NEON perf 收口:`PromoteMaskTo i16→i32` 稳定 0.575 输
> Highway,根因不是单 op 的实现,而是**定宽后端把每个 `Vec<T,N>` 都当满宽寄存器**——部分标签(partial
> tag,`N < 满 lane`)的 op 被迫在满宽上做再掰回。属架构级欠缺,单点改不动,故单列里程碑。
> **决策:选方案 A(根治,VecNeon 按 N 选窄 D/Q raw 类型),分阶段施工——见 §五。**
> **施工形态修订(2026-07-15,探针定案)**:原"阶段 0 全 backend 编不过——预期"的红树迁移,改为
> **常绿垫片迁移**——树在每个提交点编译绿 + gtest 绿;必升 native-D 的热点仅 ~15-20 op,其余 250 个
> 永久留垫片(垫片跨界已实测零成本)。见 §5.1(修订版)与 §5.3(探针)。

## 一、问题陈述(what & why)

**现象**:`PromoteMaskTo i16→i32`(把 4 个 i16 mask 拓成 4 个 i32,输入是 `Rebind<i16, ScalableTag<i32>>`
= 4-lane i16 部分标签)在 NEON 上 0.575。dumpbin:

```
HYLE:    ldr q0,[x0] ; cmgt.8h ; xtn.8b ; zip1.8b ; ushll.4s ; shl.4s#31 ; cmlt.4s ; ... (7 op)
Highway: ldr d0,[x0] ; sshll.4s ; cmgt.4s ; ...                                          (2 op)
```

**根因**:`include/hyle/simd/backend/neon.h` 的 `RawNeonT<T>` 永远是满 128-bit 类型(`int16x8_t` 等),
与逻辑 lane 数 `N` 无关。于是 `VecNeon<int16_t, 4>`(4-lane 部分向量)**物理上仍是 8-lane `int16x8_t`**,高
4 lane 是垃圾。任何 op 都发满 q 寄存器指令;需要 4-wide 语义时(如 PromoteMaskTo 先 `cmgt.8h` 造 8-wide
mask 再 `xtn/zip1/ushll` 掰回 4-wide)就多出一串。Highway 的 `Vec128<T, N>`(N 可 < 满)用**半寄存器 D
(64-bit)load**(`ldr d0`)后全程 4-wide,零掰回。

**这不是 PromoteMaskTo 一个 op 的 bug**,而是四个定宽后端(sse4/avx2/avx512/neon)共有的表示缺失:部分
标签存在(经 `Rebind`/`Repartition` 产生,scalable 后端已用 `kPow2`+`vl` 正确执行——见 simd-15/simd-design
§212),但**定宽后端不携带"我只有 N 个有效 lane、可用窄寄存器"的信息**。

## 二、影响面(谁受损)

- **直接受损(已测)**:`PromoteMaskTo i16→i32` NEON 0.575(唯一因此项挂在真输清单的 op)。
- **潜在同类**:任何吃/返回部分标签的 op——`DemoteMaskTo`(反向,已 TIE 但可能有余量)、`PromoteLowerTo`/
  `PromoteUpperTo`/`DemoteTo` 单向量便利形(返回 partial 向量,simd-design §154 记为薄封装)、`Rebind` 后的
  `LoadU/StoreU`、掩码宽度桥接族。多数当前 TIE(编译器碰巧优化掉),但都在"满宽物化 + 掰回"的脆弱路径上。
- **x86 侧**:同样表示缺失,但 x86 的 `cvtep*`/`pack*` 半寄存器形常被编译器直接优化,加之 PromoteMaskTo
  x86 已 TIE,受损不明显。**NEON 因 D/Q 寄存器区分显式,受损最大且已量化。**

## 三、约束与已有基础

- **scalable 已解一半**:SVE/RVV 用非正 `kPow2` + 运行期 `vl` 表达 partial tag,predication 天然只算 active
  lane(simd-design §212/§220)。**共享层的 partial-tag 概念已存在**——本里程碑不是发明它,是让**定宽后端
  的物理表示**跟上。
- **不可回退正确性**:所有改动必须过现有 gtest(含 partial-tag 的 PromoteMaskTo/DemoteMaskTo/widen-demote
  用例)+ 三平台。
- **不过度抽象**:Highway 的 `Vec128<T,N>` 用 `Raw128<T,N>::type` 按 N 选 `float32x2_t`(N≤2)vs
  `float32x4_t`——一个 traits 分档。照搬这个**接口契约**,不 vendor 其宏体系(章程:贴硬件、给 vec.h 行数
  警戒线)。

## 四、方案选项(待决,出设计供拍板)

### 选项 A:VecNeon 按 N 选窄 raw 类型(Highway `Raw128<T,N>` 同构)
`RawNeon<T, N>` 增加 N 维度:`N*sizeof(T) <= 8` → 64-bit D 类型(`int16x4_t` 等),否则 128-bit Q。
- **收益**:根治。部分向量物理即窄,op 天然 4-wide,PromoteMaskTo 直落 `sshll.4s`+`cmgt.4s`。
- **代价**:`VecNeon<T,N>::raw` 类型随 N 变 → 每个 op 的 `if constexpr` 要按 raw 宽度分档(D vs Q intrinsic
  不同:`vadd_s16` vs `vaddq_s16`)。~200 op 的实现面翻倍分支。**这是主要工作量**。可用宏/模板收敛(一个
  `neon_binary<vaddq,vadd>` 分发器按 raw 宽度选),但仍是全后端触碰。
- **风险**:D/Q 混用的 `vcombine`/`vget_low` 边界;mask 类型同步加 N 维。

### 选项 B:只给"部分标签热点 op"特化窄路径(窄手术)
保持 `VecNeon<T,N>` 满宽,但对 PromoteMaskTo/DemoteMaskTo/Promote*/Demote* 这些**已知吃部分标签**的 op,
内部用 `vget_low` 取低半 + D 寄存器 intrinsic 做,再按需 `vcombine`。
- **收益**:PromoteMaskTo 等直接受益,工作量小(改个位数 op),不动全后端。
- **代价**:不根治——新加的吃部分标签的 op 仍要逐个特化;"满宽物化"的隐性脆弱性还在。
- **风险**:低(局部)。

### 选项 C:接受现状,文档登记(不做)
PromoteMaskTo 是 COLD op(mask 宽度桥接,~0.5ns 绝对,非内循环)。0.575 的**绝对**代价可忽略。
- **收益**:零风险零工作量。
- **代价**:留一个真输在清单;违背"有 Highway 参考只能平不能输"。

## 五、决策:选 A(根治),分阶段施工(2026-07-15 用户拍板)

### 5.0 调研结论(动手前必读——决定施工形态)

深读 `neon.h`(3297 行 / 270 op / 961 个 Q-form intrinsic / 650 处 `.raw`)与 Highway `arm_neon-inl.h` 后,
三个决定施工形态的事实:

1. **Highway 靠宏生成整个后端**:`Raw128<T,N>` 按 N 分档(满宽→`int16x8_t` Q,任何更窄→`int16x4_t` D),
   每个 op 由 `HWY_NEON_DEF_FUNCTION_*` 宏对每个 (type × size) 自动展开——size-8 发 `vaddq_s16`、size-4/2/1
   发 `vadd_s16`(`prefix##q` vs `prefix`)。**partial 向量能统一处理,正因为宏把 D/Q 两形都生成了。**
2. **HYLE 刻意没走宏路**:每 op 手写 `if constexpr(T)` + 硬编 Q-form(`vaddq_*`),换来可读、可断点、贴硬件
   (章程"不 vendor Highway 宏体系"的理由)。代价就是现在:没有 D-form,partial 只能满宽物化。
3. **HYLE 现有 `Half` 是"满寄存器存低 64"**(neon.h:2811 明确注释):partial 向量 = 满 128-bit 寄存器、
   高 64 位零/垃圾,`LowerHalf`/`UpperHalf`/`Combine` 用 `vget_low`+`vcombine` 在这个约定下搬运。**A 会打破
   这个不变量**——partial 变成真 D 寄存器,这三个 half↔full 边界 op 要从"低 64 约定"改成真 D↔Q 转换。

**故 A 的本质** = 给 HYLE 的 NEON op 面补上 D-form 维度。不照抄 Highway 的宏(违背章程),而是用**一个按 raw
宽度分发的模板 helper**(如 `neon_bin<Qfn, Dfn>(a,b)`)把 D/Q 选择收敛到一处,避免 270 op × 2 的手写爆炸。

### 5.1 分阶段施工计划(2026-07-15 修订:常绿垫片迁移)

> 修订理由:原计划阶段 0 落 `RawNeon<T,N>` 后"全 backend 编不过——预期",意味着 270 op 全部迁完前整棵树
> 长期不可编译:没有 gtest 安全网、没有可 bisect 的提交点。修订版用**垫片(shim)**让树在每个提交点都
> 编译绿 + gtest 绿,并把"必须 native-D"的面从 270 op 缩到 ~15-20 op。垫片零成本已由探针实测(§5.3)。

**阶段 0 — 表示层 + 垫片(同一提交,树保持绿)**
- `RawNeon<T,N>`(加 N 维):`N*sizeof(T) <= 8` → D 类型(`int16x4_t`/`int8x8_t`/`float32x2_t`/…),否则 Q。
  64-bit lane(i64/u64/f64)N=1 → `int64x1_t` 等;N=2 → Q。
- `MaskRawNeon<T,N>` 同步加 N 维(mask 是同宽无符号)。
- `VecNeon<T,N>::raw` / `MaskNeon<T,N>::raw` 类型随之变。
- **同一提交内**加两个 ~6 行垫片:`to_q(v)`(D→Q,`vcombine` 高半补零;满宽恒等)/ `from_q<T,N>(x)`
  (Q→D,`vget_low`;满宽恒等)。traits 正确性用 `static_assert` 矩阵验证(全 T × 全合法 N)。

**阶段 0.5 — 机械包裹(脚本化,一次线性源变换)**
- 所有既有 op 体:`x.raw` 读取包成 `to_q(x)`,结果赋值包成 `from_q<T,N>(…)`。不改任何指令选择——partial
  的物理类型变成 D,但所有 op 仍走今天的 Q 指令,语义与现状完全等价(满宽算 + 高位垃圾)。
- 用源变换脚本做(同 bench 三 TU refit 的做法),人工只审 diff 形态是否统一。
- **闸门(替代原阶段 0 闸门)**:此步完成后全树编译绿、gtest 绿、bench 无变化(垫片零成本,§5.3)。

**阶段 1 — 热点升级 native-D(~15-20 op,逐个独立提交)**
- 摘垫片换 D-form 的最小集(拿下 PromoteMaskTo 链所需):partial `LoadU/StoreU`(`ldr/str d`)、`Set/Zero/
  Iota`、比较族、`MaskFromVec`/`VecFromMask`、`PromoteMaskTo`/`DemoteMaskTo` 本体(`sshll`/`xtn`)、
  Promote*/Demote* 的输入侧。
- 宽度分发 helper `neon_bin<QFn,DFn>` / `neon_un<QFn,DFn>`(按 `sizeof(raw)==16` constexpr 选)在此落地,
  只服务升级过的 op;未升级 op 留垫片。
- 每 op 独立:编绿 → gtest 绿 → `HYLE_BENCH_ONLY` 单 op bench。随时可停,已升级的收益已入账。
- **简单算术/位/比较族不批量迁移**(与原计划相反):partial 算术非热点,留垫片;未来哪条 partial 路径被
  量化受损,再按本阶段模式逐个升级。

**阶段 2 — half↔full 边界 op 改真 D↔Q**
- `LowerHalf`/`UpperHalf`/`Combine`/`ZeroExtendVector`/`Concat*`:从"满寄存器存低 64"约定改成真 D 结果 /
  D→Q 组合。这是最容易出正确性 bug 的一批,逐 op 穷举对拍标量 oracle。

**阶段 3 —(修订后此阶段消失)**
- 无 D-form 的 op(跨 lane reduce、`vqtbl` 只有 Q 等)**永久留在垫片上**——垫片恰好就是原计划手工要写的
  "`vcombine` 满宽兜底",现在自动、统一、零 per-op 工作量。

**阶段 4 — 收口验证**
- PromoteMaskTo 0.575→TIE(校准);全量 op_parity 三 TU 复跑防复合回归;三平台 gtest(NEON + SDE avx512 +
  scalar);cloc 量增长写回;确认没越过"不 vendor Highway"的行数警戒线(逼近 Highway 每-target 量级就是丧失
  了不 vendor 的理由——见 simd-design §229)。

**x86 侧**:同样表示缺失,但 x86 partial 受损未量化(PromoteMaskTo x86 已 TIE)。**先只做 NEON**(受损已量化、
D/Q 区分显式收益最大);x86 的 partial 表示改造视 NEON 完成后是否发现量化受损再定,不在本里程碑强做。

### 5.2 风险与退出条件
- **若阶段 1 的 `neon_bin` 迁移使行数/复杂度逼近 Highway 宏体系量级** → 停下重估:要么接受"其实就该 vendor
  Highway NEON"的结论,要么退回选项 B(窄手术)。行数警戒线是硬约束(章程)。(常绿形态下触发概率大降:
  未升级 op 只多一层统一包裹,仅升级过的 ~15-20 op 有 D 分支。)
- **若阶段 2 的 half↔full 改造引入难缠正确性 bug** → 该边界可保守用 `vcombine` 满宽兜底(牺牲该 op 的 partial
  性能、保正确),不阻塞整体。

### 5.3 施工形态探针(2026-07-15,Mac 真机定案)

四变体同场对拍 PromoteMaskTo i16→i32 内核(`oplab/s19_probe.cc`,Apple clang -O2,3 遍稳定,四变体输出
位级一致 MATCH):

| 变体 | ns/elem | 读数 |
|---|---|---|
| HYLE 现状(真实 op 链) | 0.126~0.171 | ≈0.61×Highway,复现矩阵 0.575 |
| **垫片版**(D 类型;compare 借 Q:`vcombine`→`cmgt.8h`→`vget_low`) | **0.075~0.077** | **与 Highway 不可区分** |
| native-D 终态(`ldr d`+`vcmgt`+`sshll`) | 0.071~0.077 | 与 Highway 不可区分 |
| Highway 真实链路 | 0.073~0.077 | 基准 |

两个定案读数:
1. **垫片跨界零成本**。架构原因:AArch64 上 D 寄存器物理上就是对应 Q 的低半——`vget_low` 不产生指令;任何
   D 写入自动清零高 64 位,编译器据此消掉补零 `vcombine`。这是常绿迁移形态成立的前提,**已测,非猜**。
2. 诚实脚注:垫片变体里 PromoteMaskTo 本体已按升级后建模(`sshll` 一条),故该行证明的是"**未升级 op 经
   垫片不吃性能** + 终态到位",不是"不升级就自动变快"——旧 op 体的掰回链仍在,靠阶段 1 的热点升级消。
   本探针为快探针(best-of-30,无交错无副本),方向性证据;验收仍走校准仪器(§六)。

## 六、验收(无论选哪个)

1. `PromoteMaskTo i16→i32` NEON:0.575 → TIE(校准仪器,gate 0.06,双序 3+ 跑)。
2. 现有 gtest 全绿(重点:partial-tag 的 PromoteMaskTo/DemoteMaskTo/widen-demote/masked-load 用例);三平台
   (avx2/sse4/scalar + SDE avx512 + NEON;若动 scalable 共享层则加 qemu SVE/RVV)。
3. 无同类回归:全量 op_parity 三 TU 复跑,确认没有别的部分标签 op 因改动掉档(复合上下文回归教训,见矩阵)。
4. 若选 A:cloc 量实现面增长,写回本文;确认 vec.h/后端行数未越过"不 vendor Highway"的警戒线。

## 七、参考

- 症状与 dumpbin:历史 `docs/implementation/tasks/simd/simd-module-closure.md`(真输清单 PromoteMaskTo 行 + NEON 修复演进段)。
- scalable partial-tag 机制(已实现,可借鉴表示思路):`docs/architecture/simd-design.md` §212/§220,`simd-15`。
- Highway `Raw128<T,N>` / `Vec128<T,N>` 分档:`third_party/highway/hwy/ops/arm_neon-inl.h`。

## 八、收口记录(2026-07-15,C19)

对照 §六验收逐条:

1. **PromoteMaskTo 0.575 → TIE ✅**:C12 单 op 双序 6 遍 ×4 副本对 99.5–100.0%(校准 PASS)+ 反汇编
   `ldr d ; sshll.4s ; cmgt.4s` 与 Highway 同形;C18 全量复合环境 1.000/1.000 终态复确认。
2. **gtest 全绿 ✅**:NEON 276/276(基线 220 + 护栏 46 + oracle 10;此外 13 个子满宽 DISABLED 案由
   C14/C15 转绿计入 also_run 23/23);x86 构造性无关(全部代码提交仅触 neon.h/tests)+ Windows
   scalar/sse4/avx2 形式复跑(结果见台账 C19 行)。
3. **无同类回归 ✅**:C18 三 TU × 双序 ×3 全量复跑,校准全 PASS,LOSE:parity 0/69、full 0/206、
   widen 1/28——唯一 LOSE(SatWidenMulAccumFixedPoint 0.92)为矩阵既有 `0.94?` 边界项钉实,非 s19
   回归,已立 C20 修复卡。
4. **cloc 与警戒线 ✅**:neon.h 3297 → 4002 行(+705,+21%);新增测试 381 行(护栏 191 + oracle 190)。
   对照 Highway `arm_neon-inl.h` 10982 行,HYLE NEON 后端为其 **36%**——未越"不 vendor Highway"行数
   警戒线(§5.2),退出条件未触发。

**额外交付(立项时未计划)**:测试先行卡挖出并修复两个**既有** bug 家族——①partial LoadU/StoreU
满 16 字节读写(OOB);②子满宽 UpperHalf/ZeroExtendVector/InsertIntoUpper 的 N-盲语义。均非 s19 引入,
随本里程碑清偿。

**施工形态复盘**:常绿垫片迁移全程兑现——W0a/W0b/W0c 每步编译绿 + gtest 绿,W0c 翻转一次编译过零残错;
探针预测(垫片零成本、终态同 Highway)与实测一致。执行采用 AI 分派工单制(19 卡),全记录:
`simd-19-workorders.md`(卡片/铁律/批复)+ `simd-19-ledger.md`(台账/判定)。

**遗留**:C20(SatWidenMulAccumFixedPoint,独立卡,非本里程碑范围);x86 侧 partial 表示不做
(§5.1 既定),脆弱 TIE 健康化见 `simd-arch-review-2026-07.md` 路线 3。
