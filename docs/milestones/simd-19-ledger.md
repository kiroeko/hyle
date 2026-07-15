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
| C3 | LoadU partial 窄读 | todo | | | — | | 修 OOB 读 |
| C4 | StoreU partial 窄写 | todo | | | — | | 修 OOB 写,C2 测试必须先绿 |
| C5 | 比较族 Eq/Ne/Gt/Ge/Lt/Le D-form | todo | | | — | | |
| C6 | MaskFromVec/MaskFromNormalizedVec/VecFromMask D | todo | | | — | | |
| C7 | mask And/Or/Xor/AndNot/Not D-form | todo | | | — | | |
| C8 | PromoteLowerTo/PromoteTo partial 输入 D 路 | todo | | | — | | 靶心 |
| C9 | DemoteTo partial 输出 D 路 | todo | | | — | | |
| C10 | 向量 And/Or/Xor/AndNot/Not D-form | todo | | | — | | |
| C11 | GetLane/ExtractLane/InsertLane D-form | todo | | | — | | |
| C12 | 收口:PromoteMaskTo bench 采集(双序 6 遍) | todo | | | 记录 | | 判定=高能力方 |
| C13 | half/combine 穷举 oracle 测试(先行) | todo | | | — | | 只新增测试文件 |
| C14 | LowerHalf/UpperHalf native D | todo | | | — | | C13 先绿 |
| C15 | Combine/ZeroExtendVector/InsertIntoUpper native | todo | | | — | | C13 先绿 |
| C16 | Concat* partial 路径 | todo | | | — | | C13 先绿 |
| C17 | ResizeBitCast partial 审计 | todo | | | — | | 低优 |
| C18 | 全量 NEON bench 夜跑 ×3 双序 + 采集 | todo | | | 记录 | | 判定=高能力方 |
| C19 | 文档/矩阵回写 + cloc | todo | | | — | | 高能力方 |
