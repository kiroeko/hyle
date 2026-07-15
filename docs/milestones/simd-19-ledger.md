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
| C10 | 向量 And/Or/Xor/AndNot/Not D-form | done | ✓ | 266/266 | — | (本提交) | D: to_u32_d/from_u32_d+vand/vorr/veor/vbic_u32; Not=vdup_n_u8 FF xor; Q 原样 |
| C11 | GetLane/ExtractLane/InsertLane D-form | done | ✓ | 266/266 | — | (本提交) | GetLane: vget_lane_*(raw,0); Extract/Insert 在 vec.h 泛型 store+reload 未动 |
| C12 | 收口:PromoteMaskTo bench 采集(双序 6 遍) | todo | | | 记录 | | 判定=高能力方 |
| C13 | half/combine 穷举 oracle 测试(先行) | todo | | | — | | 只新增测试文件 |
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
