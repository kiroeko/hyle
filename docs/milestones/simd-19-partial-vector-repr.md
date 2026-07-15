# simd-19: 定宽后端的部分向量表示(partial-vector representation)

> **状态**:规划中(2026-07-15 编)。触发自 NEON perf 收口:`PromoteMaskTo i16→i32` 稳定 0.575 输
> Highway,根因不是单 op 的实现,而是**定宽后端把每个 `Vec<T,N>` 都当满宽寄存器**——部分标签(partial
> tag,`N < 满 lane`)的 op 被迫在满宽上做再掰回。属架构级欠缺,单点改不动,故单列里程碑。

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

## 五、建议

**倾向选项 B 先做(窄手术解 PromoteMaskTo),同时评估选项 A 的 ROI**:
- B 能立刻把最后一个"可修"NEON 真输清零(ScatterIndexN 是编译器寄存器类微差、非表示问题,另论),风险可控。
- A 是根治但翻倍实现面,ROI 取决于**未来还有多少吃部分标签的热点 op**。当前只有 PromoteMaskTo 一个已量化
  受损,其余同类都 TIE。若未来 partial-tag 消费者增多(如更多 mask 宽度桥接、half-vector widen 链),再上 A。
- 先做 B + 在本文登记 A 的完整设计,等消费者信号。

## 六、验收(无论选哪个)

1. `PromoteMaskTo i16→i32` NEON:0.575 → TIE(校准仪器,gate 0.06,双序 3+ 跑)。
2. 现有 gtest 全绿(重点:partial-tag 的 PromoteMaskTo/DemoteMaskTo/widen-demote/masked-load 用例);三平台
   (avx2/sse4/scalar + SDE avx512 + NEON;若动 scalable 共享层则加 qemu SVE/RVV)。
3. 无同类回归:全量 op_parity 三 TU 复跑,确认没有别的部分标签 op 因改动掉档(复合上下文回归教训,见矩阵)。
4. 若选 A:cloc 量实现面增长,写回本文;确认 vec.h/后端行数未越过"不 vendor Highway"的警戒线。

## 七、参考

- 症状与 dumpbin:`docs/milestones/simd-orderswap-matrix.md`(真输清单 PromoteMaskTo 行 + NEON 修复演进段)。
- scalable partial-tag 机制(已实现,可借鉴表示思路):`docs/architecture/simd-design.md` §212/§220,`simd-15`。
- Highway `Raw128<T,N>` / `Vec128<T,N>` 分档:`third_party/highway/hwy/ops/arm_neon-inl.h`。
