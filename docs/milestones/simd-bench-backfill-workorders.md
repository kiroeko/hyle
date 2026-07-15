# simd bench 补全施工手册(41 个待补 bench,工单制,面向中上能力 AI)

> **任务**:把 `simd-orderswap-matrix.md`"41 个待补 bench"节的 op 全部纳入校准 op_parity 框架,产出可信
> parity 行。**这是写测量代码,不是修性能**:执行 AI 只负责让 op 被正确地测到,**不做任何胜负判定、
> 不改任何 op 实现**。判定统一在 B4 由高能力方按矩阵"复现"节协议做。
> 方法论法源:`docs/performance/simd-op-benchmark-methodology.md`(内核五要素、校准、判读)。执行 AI
> 开工前通读一遍,冲突时以本手册铁律为准。

## §1 分批

| 批 | 内容 | 数量 | 难点 |
|---|---|---|---|
| **B0(高能力方,前置)** | 配对决策表:每 op 一行——HYLE 调用形状 / Highway 对手名(或无对手时的 run_third 设计)/ 数据域 / sink 方式 / 归属 TU 与节 / minEpochIterations 档 | 41 行 | 这是全部判断密度所在;没有它执行 AI 必然退化成现场设计 |
| B1 | 结构性原语:BitsFromMask、MaskFromVec、MaskFromNormalizedVec、VecFromMask、SignBit、ShiftRightUniform、LowerHalfOfMask、UpperHalfOfMask、MaskFalse、LoadMaskBits、StoreMaskBits | 11 | 多数归 `op_parity_bench.cc` §25 mask 域一带;sink 走 BitsFromMask 前例 |
| B2 | InRange/契约快变体:ConvertInRangeTo、DemoteInRangeTo、DemoteToNearestIntInRange、PromoteInRange{Lower,Upper,Even,Odd}To、TableLookupBytesInRange、TableLookupBytesOr0 | 9 | 配对按 TableLookupBytes 前例:HYLE 变体 vs Highway 对应形,数据域限定在契约内,必要时 run_third 摆正体 |
| B3 | 真缺口:MaskedSet、MaskedSetOr、MaskedIsNaN、Interleave{Even,Odd,Upper}Blocks、Slide{Up,Down}Blocks、Lt128Vec、SumsOfShuffledQuadAbsDiff、OrderedTruncate2To、TruncateStore、Shuffle01、RearrangeToOddPlusEven、SafeCopyN、SafeFillN、Stream、CompressBits、CompressBitsStore、CompressBlocksNot、IfNegativeThenNegOrUndefIfZero | 21 | 按 B0 表归属:widen/truncate 类进 `op_parity_widen_bench.cc`,Compress 系挨现有 Compress 节,其余按家族 |
| B4(高能力方判读) | 三平台全量双序复跑 + `scripts/op-parity-report.py` + 判定 + 矩阵回写 | — | 执行 AI 只跑采集,不判读 |

注:AES/CLMul 8 个 op 迁入校准框架是矩阵单列的另一项欠账,不在本手册范围(需要 crypto_bench 的数据形
搬迁,判断密度高,留高能力方)。

## §2 铁律(违反任一 = 该卡作废)

1. **只准新增内核代码块,白名单 = B0 表指定 TU 的指定节**。`op_parity_engine.h`、
   `scripts/op-parity-report.py`、已存在的内核、op 实现(`include/`)一律只读。
   ——测不平**不准回头改 op 或改仪器**,记录数值即可,归因是 B4 的事。
2. **内核五要素一个不少**(模板见 §3):`stagger(rep)` 开头、指针提升(`auto* p_X = X.data()`)、
   主循环、`nb::doNotOptimizeAway(sink 或 p_X)`、`hyle_bench::mark(rep)` 收尾。两侧(HYLE/Highway)
   **形状必须逐行同构**——只有命名空间和 op 名不同。
3. **标题纯 ASCII**,格式 `"<OpName> <type> (4096)"`;`batch(kN)`;`minEpochIterations` 按 B0 档位,
   不自创数值。
4. **数据域按 B0 表**,尤其 InRange 变体:输入必须落在契约域内(越界 = 测了 UB,整行作废);
   数据生成用该 TU 现有的 `geni/genf` 辅助 + 固定种子,不引入新随机源。
5. **一卡一提交**,消息 `bench(simd): backfill <OpName> parity row`;开工前 `git status` 干净。
6. **完成 = 原始闸门输出粘贴 + 台账更新**(§4);"编译过了应该没问题"不算完成。
7. sink 必须真消费结果(mask 类走 `BitsFromMask` 异或进 `sink`,store 类 `doNotOptimizeAway(p_out)`),
   禁止让编译器可整体 DCE 的空转内核。
8. 同一错误修 ≤2 次,仍败 → revert + 台账记 `blocked` + 上报。

## §3 内核模板(法定形状,照抄改名)

```cpp
{
  nb::Bench bench; bench.title("<OpName> i16 (4096)").relative(true).batch(kN).minEpochIterations(<B0档>);
  HYLE_RUN_PAIR(bench, [&](auto rep){ hyle_bench::stagger(rep); auto* p_a = a.data();
    for (std::size_t i=0;i+hw<=kN;i+=hw) {
      /* HYLE 侧:一到两个 op 调用,结果进 sink */
    }
    nb::doNotOptimizeAway(&sink); hyle_bench::mark(rep); },
  [&](auto rep){ hyle_bench::stagger(rep); auto* p_a = a.data();
    for (std::size_t i=0;i+gw<=kN;i+=gw) {
      /* Highway 侧:同构 */
    }
    nb::doNotOptimizeAway(&sink); hyle_bench::mark(rep); });
}
```
无 Highway 对手的 op(B0 表标注)用 `run_third` 挂第三形,前例见 `op_parity_bench.cc` TableLookupBytes 节。

## §4 闸门(每卡)

| 闸门 | 内容 | 通过标准 |
|---|---|---|
| G1 | Windows 构建目标 TU(`hyle_op_parity_bench_avx` / `..._full_bench_avx` / `..._widen_bench_avx`,用 oplab 既有 build bat;失效则 `cmake --build build --target <t> -j`) | exit 0,exe mtime 新(stale-exe 教训) |
| G2 | 过滤冒烟:`HYLE_BENCH_ONLY=<OpName> ./<exe>` | 锚点行全 PASS 形态 + 新 op 行出现且无 `:wavy_dash:` 崩坏;粘贴原始输出 |
| G3 | 涉 NEON 的 op:同步 Mac 后 `HYLE_BENCH_ONLY=<OpName> ~/hyle/build-mac/bin/hyle_op_parity_bench_neon`(同步/构建命令逐字用 `simd-19-workorders.md` §3;**禁全量跑**,jetsam 会杀) | 同 G2 |
| G4 | 台账行:op、TU、冒烟 ratio(仅记录,不判定) | 台账与代码同提交 |

## §5 台账(`docs/milestones/simd-bench-backfill-ledger.md`,B0 时初始化)

```markdown
| op | 批 | TU/节 | 状态 | 冒烟 ratio(avx / neon) | 提交 | 备注 |
```
状态:`todo / done / blocked`。B4 判定结果由高能力方另行写回矩阵,不进本台账。

## §6 交接 prompt(逐字粘给执行 AI,替换 <op 名>)

```
你是 bench 补全施工 AI。唯一任务:按 docs/milestones/simd-bench-backfill-workorders.md 为 <op 名>
新增 parity 内核。先读该手册全文 + B0 配对决策表中你这一行 + methodology 文档的内核五要素节。
铁律:①只在 B0 指定的 TU/节新增代码,engine/报告脚本/既有内核/op 实现只读;②两侧内核逐行同构、
五要素齐全;③数据域按 B0 表;④测不平不改 op 不改仪器,只记录;⑤完成 = G1~G4 原始输出粘贴 +
台账更新 + 1 个提交;⑥同一错误修 2 次仍败即 revert 记 blocked。
产出:①diff 摘要 ②G1~G4 原始输出 ③提交 hash ④台账 diff。
```

## §7 Escalation

- B0 表某行拿不准(找不到 Highway 对手/数据域有疑) → 高能力方补行,执行 AI 不猜。
- G2/G3 出现锚点不 PASS → 立即停整批(仪器层问题),上报高能力方,不准继续出卡。
- B4 判读、UNSTABLE 复跑、矩阵回写 → 高能力方。
