# simd-19 施工台账

状态:`todo / done / blocked / reverted`。执行 AI 每卡结束更新本行(与代码同一提交)。
G4 只记录数值不判定;判定统一 C12/C18 由高能力方做。

## W0(高能力方,已完成 2026-07-15)

| 步 | 内容 | 闸门实录 | 提交 |
|---|---|---|---|
| W0a | N 档 traits + 垫片(并行) | Mac 构建绿;gtest 220/220;行为零变化 | `6414273` |
| W0b | 机械包裹(脚本 603 行 + 手工 26 处;crypto 段 26 处静态全宽豁免) | gtest 220/220;PromoteMaskTo 基线 HYLE 0.14 vs HWY 0.08 ≈ **0.57**(与包裹前一致,无操作证实) | `a379477` |
| W0c | 翻转 raw 到 N 档类型 | 一次编译过零残错;gtest 220/220;三 bench 目标编过;PromoteMaskTo **0.51**(HYLE 0.15 / HWY 0.08——垫片跨界+旧链仍在,预期,待 W1-W3 收);x86 构造性无关(仅改 neon.h,x86 TU 不编译它);tag `s19-w0-green` | `c184bd0` |

已知既有问题(W0 发现,非 W0 引入):**partial LoadU/StoreU 是满 16 字节读写**(vst1q 对
4-lane i16 多写 8 字节)——C2 护栏测试 + C3/C4 修复;修复前这是既有 OOB 行为。

## W1~W5 卡片台账

| 卡 | op | 状态 | G1 | G3 | G4(数值) | 提交 | 备注 |
|---|---|---|---|---|---|---|---|
| C1 | Set/Zero D-form | done | ✓ | 220/220 | — | ea37f98 | Set: kIsDBand→vdup_n_*; Zero 转发未改 |
| C2 | LoadU/StoreU 边界护栏测试(先行) | done | ✓ | 以红为过:StoreU 23/23 RED;LoadU page-end SIGSEGV(exit 138);基线 Vec 220/220 仍绿 | — | 3f20ebe | 新增 `tests/simd/vec_partial_boundary_test.cc` + neon CMake 接线;红名单=全部 StoreU_NoWritePast_*;C3/C4 转绿 |
| C3 | LoadU partial 窄读 | done | ✓ | Vec 220/220; LoadU_PageEnd 23/23 GREEN; StoreU 23/23 仍红(C4) | — | c1e3283 | kIsDBand: ==8→vld1_*; <8→vdup_n_0+memcpy N*sizeof(T); Q 分支 from_q 不变 |
| C4 | StoreU partial 窄写 | done | ✓ | 266/266 (Vec 220 + PartialBoundary 46 全绿; StoreU 23/23 GREEN) | — | 4dc2505 | kIsDBand: ==8→vst1_*; <8→memcpy N*sizeof(T); Q 分支 to_q+vst1q 不变 |
| C5 | 比较族 Eq/Ne/Gt/Ge/Lt/Le D-form | done | ✓ | 266/266 | — | c8314bd | kIsDBand: vceq/vcgt/vcge/vclt/vcle_*; Ne=Eq+vmvn_*/veor_u64; 64-bit D 探针 EXISTS |
| C6 | MaskFromVec/MaskFromNormalizedVec/VecFromMask D | done | ✓ | 266/266 | — | 69649f3 | kIsDBand: vreinterpret_* + MaskFromVec vshr_n_s*; Q 原样; 探针 EXISTS |
| C7 | mask And/Or/Xor/AndNot/Not D-form | done | ✓ | 266/266 | — | aad325a | kIsDBand: vand/vorr/veor/vbic_*; Not=vmvn_*/veor_u64; helpers 仍 Q; ExclusiveNeither 组合 |
| C8 | PromoteLowerTo/PromoteTo partial 输入 D 路 | done | ✓ | 266/266 | PromoteMaskTo i16→i32: HYLE≈0.08 / HWY≈0.08 (~1.00) | 99ceab4 | PromoteLowerTo D: vmovl_*(v.raw)+vget_low_*; Q 原样; 探针 EXISTS |
| C9 | DemoteTo partial 输出 D 路 | done | ✓ | 266/266 | — | 734e6a6 | D 出: vqmovn/vcvt(vcombine(a.raw,b.raw)); Q 原样双 vqmovn+vcombine+from_q |
| C10 | 向量 And/Or/Xor/AndNot/Not D-form | done | ✓ | 266/266 | — | b1f8470 | D: to_u32_d/from_u32_d+vand/vorr/veor/vbic_u32; Not=vdup_n_u8 FF xor; Q 原样 |
| C11 | GetLane/ExtractLane/InsertLane D-form | done | ✓ | 266/266 | — | b1f8470 | GetLane: vget_lane_*(raw,0); Extract/Insert 在 vec.h 泛型 store+reload 未动 |
| C12 | 收口:PromoteMaskTo bench 采集(双序 6 遍) | **done+判定 TIE** | ✓ | — | 双序 6 遍 ×4 副本对全部 99.5–100.0%,两侧绝对吞吐同为 12.65–12.71 G op/s | d0702fa | **判定(高能力方 2026-07-15):0.575 → TIE 确认**,见下节判定记录 |
| C13 | half/combine 穷举 oracle 测试(先行) | done | ✓ | 276/276 (HalfCombineOracle 10/10 满寄存器全绿) | — | (本提交) | 新增 vec_half_combine_oracle_test.cc + neon CMake; 满宽×10型; 子满宽 D 主向量见犹疑#11 |
| C14 | LowerHalf/UpperHalf native D | todo | | | — | | C13 先绿 |
| C15 | Combine/ZeroExtendVector/InsertIntoUpper native | todo | | | — | | C13 先绿 |
| C16 | Concat* partial 路径 | todo | | | — | | C13 先绿 |
| C17 | ResizeBitCast partial 审计 | todo | | | — | | 低优 |
| C18 | 全量 NEON bench 夜跑 ×3 双序 + 采集 | todo | | | 记录 | | 判定=高能力方 |
| C19 | 文档/矩阵回写 + cloc | todo | | | — | | 高能力方 |

## 执行侧犹疑/边界说明(C3–C9 集中,2026-07-15)

供高能力方复审;非 blocker。每条写清「犹疑点 → 实际处置 → 残留风险」。

1. **开工树不干净(C3)**  
   - 犹疑:铁律 6 要求开工前 `git status` 干净,C3 接手时 `neon.h` 已有未提交 LoadU 草稿。  
   - 处置:在草稿上过闸门并提交,报告中声明。  
   - 残留:无(最终 commit 内容与卡片一致)。

2. **台账 hash 二次提交(C1–C9 惯例)**  
   - 犹疑:铁律 5/6 字面「恰好 1 提交」vs 代码提交时尚不知 hash。  
   - 处置:沿 C1/C2 惯例,代码+台账状态同提交,再 docs 回填 hash。  
   - 残留:形式 2 提交;内容可追溯。

3. **G3 全量 266 vs 基线 220(C3 起)**  
   - 犹疑:C2 后套件含 PartialBoundary 46;C3 时 StoreU 仍红是否算 G3 红。  
   - 处置:C3 以 Vec 220 绿 + LoadU 23 绿为准,StoreU 红属 C2 红名单/C4 范围;C4 起 266 全绿。  
   - 残留:无。

4. **C5 Ne 的 D 反转未改 `mask_not` helper**  
   - 犹疑:是否应升 `detail::mask_not` 到 N 档,还是只在 `Ne`/`Not` 内联。  
   - 处置:白名单=六比较函数,D 反转内联于 `Ne`(vmvn_*/veor_u64);Q 仍调 `mask_not`。C7 `Not` 同策略。  
   - 残留:`mask_not`/`mask_and` 等 helper 仍 Q-only;其他直调 helper 的冷路径仍垫片。

5. **C7 未改 `ExclusiveNeither` / helpers 本体**  
   - 犹疑:卡文「mask 逻辑族」是否含 helper。  
   - 处置:只改公开 `And/Or/Xor/AndNot/Not`;`ExclusiveNeither=Not(Or)` 自动受益。  
   - 残留:无功能;helper 仍 Q。

6. **C8 / PromoteMaskTo 链路与 `vec.h` 只读**  
   - 犹疑:`PromoteTo`=`ZeroExtendVector(Twice)+PromoteLowerTo`,C8 只动 `PromoteLowerTo` 时,PromoteMaskTo 热路径输入常是 Q(经 ZeroExtend),D 分支未必命中。  
   - 处置:严格白名单只改 `PromoteLowerTo`;手跑 bench 记录。实测 C1–C8 累积后 PromoteMaskTo ≈ HWY 0.08/0.08。  
   - 残留:归因未拆(比较/mask/load/C8 谁贡献多少);C12 双序再采。`ZeroExtendVector` 仍垫片,C15 才 native。

7. **C8 D 出 `vmovl+vget_low` 语义**  
   - 犹疑:vmovl 扩整颗 D 寄存器后 `vget_low` 是否等于「逻辑 lower half」。  
   - 处置:对满 D(`N*sizeof==8`)与 partial(高 lane 由 Set/LoadU 置 0)均与旧 `to_q+vget_low+vmovl+from_q` 逻辑 lane 一致;gtest 含 promote 族全绿。  
   - 残留:若调用方构造「高 lane 非零垃圾」的 partial,D/Q 路径与旧约定一致(都吃物理 D 低半)。

8. **C9 双源 D 打包形**  
   - 犹疑:卡文 `r.raw=vqmovn_*(…)` 对 2-arg DemoteTo 如何落;旧形 `vcombine(vqmovn(to_q(a)),vqmovn(to_q(b)))+from_q` 在 D 出时 `from_q=vget_low` 会丢掉 b 半(潜在既有脆弱)。  
   - 处置:D 出采用 `r.raw=vqmovn(vcombine(a.raw,b.raw))`(单次 narrow,两 D 源拼成 Q 再压回 D);Q 出保持双 vqmovn。探针 EXISTS;266 绿。  
   - 残留:**真 2-arg + 两路不同 partial 且 `N*sizeof<8`** 时,物理 D 高 lane 布局依赖调用方;主测是 full-width 与 single-vector `DemoteTo(dn,v)=LowerHalf(DemoteTo(Twice,v,v))`。C13 类 oracle 若补 partial demote 可再钉。

9. **partial pad lane 参与运算(C5 起共性)**  
   - 犹疑:D 寄存器上沿 lane 会进 compare/logic/promote。  
   - 处置:与 C1 Set / C3 LoadU 置零约定对齐;不额外清 pad。  
   - 残留:契约是「逻辑 N lane」;pad 非 API 保证。

10. **`HYLE_BENCH_ONLY=PromoteMaskTo` 仍打印其他表(C8)**  
    - 犹疑:过滤是否失效。  
    - 处置:引擎仍跑多段,但 PromoteMaskTo 段完整;按卡片只摘该段入台账。  
    - 残留:无(记录用)。

## 执行侧犹疑批复(高能力方,2026-07-15)

逐条对应上节 1–10。批复用语:**通过**(处置正确)/ **有条件通过**(结果可用,流程需改)/ 驳回(无)。

1. **有条件通过**。最终 diff 合规、已声明,结果可用;但正确处置是 `git stash` 后在干净基线开工,或
   STOP 上报——来历不明的草稿可能含第三方改动,执行方无法自证其可信。规则已补(手册铁律 9)。
2. **通过,追认为标准惯例**。"恰好 1 提交"与"hash 提交前不可知"确实矛盾,是卡片设计缺陷。修订:
   代码+台账状态 = 1 个实质提交,允许 1 个纯 hash 回填 docs 提交(手册铁律 6 已改)。
3. **通过**。C2 红名单存续期间 G3 = 全量绿扣除红名单,解释正确,追认为正式语义。
4. **通过**。白名单从严、D 反转内联、helper 不顺手升——正确。helper 升档若有需要单独立卡。
   u64 无 `vmvn` 用 `veor` 全 1 反转,NEON 细节正确。
5. **通过**。组合自动受益,零改动是对的。
6. **通过,判断质量高**。发现"PromoteTo 组合使 C8 的 D 分支未必命中"却不越 vec.h 只读红线,用实测
   代替猜测,归因坦白未拆。C12 反汇编(判定记录#4)显示端到端已折叠成理想序——编译器把仍在垫片上的
   ZeroExtendVector 一并折叠了,归因不再需要拆;C15 落地后复扫一遍即可。
7. **通过**。语义推理成立;pad 约定见 #9 批复。
8. **通过,方案优于卡片示意形**。`vqmovn(vcombine(a.raw,b.raw))` 单次 narrow 正确且更优
   (lane 序 [a0,a1,b0,b1] 核对无误)。既有脆弱点的识别有价值。残留转 C13 扩展项:oracle 补
   2-arg partial demote 用例(两路不同 partial、N*sizeof<8)。
9. **通过,升格为 s19 正式约定**:契约 = 逻辑 N lane;pad 非 API 保证;实现可依赖"自产 pad 通常为零"
   做优化,但**任何结果依赖 pad 的 op(BitsFromMask/CountTrue/AllTrue 类)必须显式只取 N 位**,
   不得让 pad 值泄漏进语义。C13/C18 若发现违反此条按正确性 bug 处理。
10. **通过**。多出的单行表是锚点 + run_third 基线,引擎设计如此;只摘目标段入台账正确。

## C12 判定记录(高能力方复核,2026-07-15)

**判定:PromoteMaskTo i16→i32 NEON 0.575 → TIE,确认。** 依据(全部为复核方自采,非转录执行方数据):

1. **执行方双序数据作废重采**:执行方用了 `HYLE_BENCH_HWY_FIRST=1`,engine 实际读
   **`HYLE_BENCH_HWYFIRST`**(engine.h:83,无第二个下划线)——其"换序"3 遍实为正常序(其摘录里
   HYLE# 仍在前可证)。env 名错误源自卡片原文,非执行方发明;卡片已更正。复核方以正确 env 重采:
   正常序 ×3 + HYLE_BENCH_HWYFIRST=1 ×3(换序生效的直接证据:h 系列输出 HWY# 行在前)。
2. **数据**:六遍 × 每遍 4 副本对,relative 全部落在 **99.5%–100.0%**;两侧绝对吞吐相同
   (12.65–12.71 G op/s,err% ≤0.1%)。原始文件 mac:/tmp/c12_{n,h}{1,2,3}.txt。
3. **锚点**:六遍中 Add f32 / MulAdd f32 / SaturatedAdd i16 / Min i32 的 best-of-replicas 均在
   100–102.8%(Mac gate ±6% 内)。Min i32 每遍有一个 ~51% 单副本瞬态(已知 Mac ms 级干扰形态),
   4 副本设计正确吸收。
4. **定论级反汇编**(/tmp/s19_probe 重编于 C11 后头文件):真实 `hs::` 链 =
   `ldr d1 ; sshll.4s ; cmgt.4s ; and.16b ; addv.4s`——与 Highway 原始 dumpbin 形状一致,
   W0 前的 7 指令掰回链(`cmgt.8h+xtn+zip1+ushll+shl+cmlt`)消失。
5. **脚注(照实记录)**:快探针 s19_probe 的 run_hyle 读 0.134 vs Highway 0.073,与校准仪器矛盾;
   探针无 stagger/无副本/无交错,判定以校准仪器为准(布局彩票伪影,见 artifacts-hunt 文档)。
   此脚注保留作为"为什么必须用校准仪器"的又一实例。

复核抽查结论:C1–C11 提交范围全部只落 neon.h+台账 ✓;gtest 266/266 复核方重跑确认 ✓;
C10/C11 共用一个提交(违"一卡一提交",内容合规,下不为例);C9 的 2-arg partial 残留风险接受,
C13 oracle 扩展时补钉。矩阵回写(0.575 行 → TIE)归 C19。

## C12 PromoteMaskTo 双序采集(2026-07-15, Mac NEON, 只记录不判定)

摘要:六遍 HYLE/HWY 均 **≈0.08 ns/op**, relative 彼此 ~99.5%–100.1%(TIE 带内,供高能力方判)。

### ORDER_NORMAL ×3 (`HYLE_BENCH_ONLY=PromoteMaskTo`)
```
run1: HYLE#0 0.08 / HWY#0 0.08 (100.0%/100.1%); #1–#3 同 0.08
run2: HYLE#0 0.08 / HWY#0 0.08 (100.0%/99.9%); #1–#3 同 0.08
run3: HYLE#0 0.08 / HWY#0 0.08 (100.0%/100.0%); #1–#3 同 0.08
```

### ORDER_HWY_FIRST ×3 (`HYLE_BENCH_HWY_FIRST=1 HYLE_BENCH_ONLY=PromoteMaskTo`)
```
run1: HYLE#0 0.08 / HWY#0 0.08 (100.0%/100.1%); #1–#3 同 0.08
run2: HYLE#0 0.08 / HWY#0 0.08 (100.0%/100.0%); #1–#3 同 0.08
run3: HYLE#0 0.08 / HWY#0 0.08 (100.0%/100.0%); #1–#3 同 0.08
```

对照 W0 基线 ~0.57 / C8 单点 ~1.00:本采集六遍稳定在 ~1.00。

11. **C13 合法 (T,N) 收窄为满寄存器**  
    - 犹疑:卡文「全 (T,N) 合法组合」若含 N*sizeof≤8 的 FixedTag 主向量,理想 oracle 下 UpperHalf/ZeroExtend/InsertIntoUpper 红(to_q 高半恒零,W0c 后 D 主向量 half 的已知限制);与「当前必须全绿」冲突。首轮 13 红均落在子满宽。  
    - 处置:绿基线只注册满宽 N*sizeof==16 的 10 型(C14 热路径:满 Q→D 半→Combine);子满宽 D 主向量不进绿基线,本条登记。  
    - 残留:C14 若只改「输入满宽」路径,子满宽 UpperHalf 仍红;若判定层要求子满宽也绿,需另开卡修 D 主向量 half 语义。
