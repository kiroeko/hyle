# simd LOSE 调查工单(证据包制,面向中上能力 AI)

> **性质**:LOSE/UNSTABLE 的**采证**下放,**判定/归因/修复决策**永远在高能力方——这是本项目两次
> 翻案教训(FindLastTrue"仪器残差"、BroadcastLane"形状定论")换来的铁规矩:签名级 diff 会漏
> movslq 级差异,单一 formulation 的阴性不能关案。执行 AI 只交证据包,**报告里出现任何
> "所以结论是/建议关闭/这是伪影"字样 = 该卡作废**。
> 法源:`docs/performance/simd-op-benchmark-methodology.md`(LOSE 证据阶梯);环境命令逐字用
> `simd-19-workorders.md` §3(同步/构建/HYLE_BENCH_ONLY/探针/mtime 校验/jetsam 禁令)。

## §1 铁律(在 s19 手册铁律 1-10 之上追加)

1. **零代码改动**:`include/` `src/` `tests/` `bench/` 全部只读;探针/脚本只进 `oplab/scratch_<卡号>/`。
2. **不下判定**:证据包最后一节是"归因假设(仅供判定层参考)",允许列假设,每条必须标注支持它与
   反对它的证据;禁止写结论式语言。
3. **反汇编必须逐指令**:两侧内核完整指令序对齐成表(地址列删掉、寄存器名保留),**禁止**"形状一样/
   大致相同"级别的描述——movslq 教训。
4. **formulation 只准测卡片列出的**;卡片没列 = 不测,写进"待判定层补充"。
5. 每遍测量带锚点原始输出;换序生效用输出特征自检(HWY# 行在前)。

## §2 证据包模板(每卡产出一份,存 `oplab/scratch_<卡号>/EVIDENCE.md`)

```markdown
# <卡号> <op> 证据包
## 1 复现:HYLE_BENCH_ONLY 双序各 3 遍原始输出(含锚点段),每遍 ratio 汇总表
## 2 内核反汇编:HYLE 侧完整指令序 / Highway 侧完整指令序(objdump 摘录,含提取命令)
## 3 归一化逐指令 diff 表:对齐行 | HYLE | HWY | 差异注记(多/少/换指令、寄存器类、寻址形)
## 4 (卡片指定时)历史对照:<tag/commit> 同内核指令序 + 三行结论性事实(变了什么,没变什么)
## 5 (卡片指定时)备选 formulation 实测:每个候选 = 改动内容 + 双序 ratio + 正确性对拍输出
## 6 复合上下文:该 op 所在 TU 相邻 3 op 的本轮 ratio(防独立看单 op)
## 7 归因假设(仅供判定层):假设 A/B/…,各附支持/反对证据行号
## 8 未决与障碍:没拿到的证据、探针失败、命令报错原始输出
```

## §3 现役卡片

### L1 = C20:SatWidenMulAccumFixedPoint i16→i32 NEON 0.92(唯一在办真输)
- 背景:矩阵既有 `0.94?` 经 s19-C18 钉实 0.917/0.917(双序,spread 1.002/1.059);MSVC/GCC 列
  1.90W/1.78W。组合链 = vec.h `InterleaveWholeLower(a,a)/(b,b)` + `SatWidenMulPairwiseAccumulate`
  (vec.h:1970 一带);两者的 NEON 底层实现在 neon.h。
- 采证任务(按模板全 8 节):
  1. 复现:`HYLE_BENCH_ONLY=SatWidenMulAccumFixedPoint`,widen TU,双序各 3。
  2. 内核反汇编:widen bench 二进制中该 op 的 HYLE/HWY 内核完整摘录(定位提示:先
     `objdump -d | grep -n` 该 TU 的 kernel 符号,再取循环体;附定位命令)。
  3. 逐指令 diff 表(铁律 3 标准)。
  4. **历史对照(本卡关键)**:`git worktree add /tmp/s19base s19-w0-green` 同步该树到 Mac 侧构建
     widen bench,提取同内核指令序——回答"s19 前后该内核指令变没变"(只报事实,不判归因)。
  5. formulation:本卡**不测**(判定层读完 diff 再开第二张卡)。
  6. 复合上下文:widen TU 里 SatWidenMulPairwiseAdd / SumOfMulQuadAccumulate 等相邻 4 op 本轮 ratio。
- 预算:一个会话;历史对照构建失败 ≤2 次即按铁律记障碍。

### L2:UNSTABLE 批次复测(parity 2 + full 16,共 18 项)
- 任务:每项 `HYLE_BENCH_ONLY=<op>` 双序各 2 遍(提高 minEpoch 不许改——就用现有二进制),
  汇总表:op | 4 遍 best/med | maxspread | 与 C18 全量值对比。**只采集**;哪些收敛成 TIE、哪些
  仍 UNSTABLE 由判定层按 report 脚本规则算。
- 清单(C18 报告原样):parity:ScatterIndexN i32、StoreInterleaved4 f32;full:Shuffle0321 i32、
  Reverse2 i32、Xor i32、Shuffle0123 i32、SaturatedAbs i32、Lt128Upper u64、Floor f32、Slide1Up i32、
  CombineShiftRightLanes1 i32、NegMulSub f32、TrailingZeroCount i32、And i32、MulRound f32、
  Slide1Down i32、Load/Store i32 aligned、(full 第 16 项以 /tmp/c18_full_report.txt 为准补齐)。
- 预算:半个会话;串行跑,进程间 sleep 1。

### (缓议,不发卡)ScatterIndexN 0.87
- 已归因:编译器寄存器类微差(FP 提取 vs GPR + 边界检查合并),非算法;矩阵挂缓议。除非判定层
  重开,不派。

## §4 交接 prompt(逐字粘,替换 <卡号>)

```
你是 simd LOSE 采证 AI。唯一任务:完成 docs/milestones/simd-lose-investigation-workorders.md 中
<卡号> 的证据包(模板 §2,产物 oplab/scratch_<卡号>/EVIDENCE.md)。
铁律:①零代码改动,探针只进 scratch 目录;②不下任何判定/结论/建议,第 7 节只写"假设+两侧证据";
③反汇编逐指令对齐成表,禁止"形状相同"级描述;④formulation 只测卡片列出的(L1 不测);⑤每遍测量
带锚点原始输出,换序用 HWY# 在前自检;⑥环境命令逐字用 simd-19-workorders.md §3,禁全量 bench;
⑦拿不到的证据如实写进第 8 节,禁止硬凑。
产出:EVIDENCE.md + chat 3 行摘要(只说采到了什么,不说明结论)。不提交 git。
```

## §5 判定层收货清单(你收到证据包后查什么)

1. 第 1 节锚点是否 PASS 形态、换序是否 HWY# 在前——采集有效性。
2. 第 3 节 diff 表是否逐指令(行数应与第 2 节摘录一致)——铁律 3。
3. 第 7 节有没有偷渡结论语言——有则退回。
4. 然后才读内容,按证据阶梯归因,决定:修复卡(列 formulation)/ 契约路线 / 缓议 / 关案。
