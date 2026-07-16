# SIMD op 面覆盖率清单(2026-07-14 实证审计)

> 结论先行（2026-07-16 复核）:HYLE **公开/selector op 名 387 个**(机械去重),非掩码面与 Highway
> 已基本齐平。旧差集声称缺失的 ~28 个掩码基础算术名其实已经由 `vec.h` 的泛型 `Masked*` 家族
> 完整实现并有 oracle；本轮又补齐 4 个 upper-only 128-bit key 接口。剩余值得跟踪的是硬件机会，
> 不是结构性 API 空洞。
> 变宽 SVE/RVV 追平定宽已在 simd-15~18 收口。经 op_parity 仪器重建(见
> [`../debug/op-parity-harness-artifacts-hunt.md`](../debug/op-parity-harness-artifacts-hunt.md)),
> 每个已测 op 均有三平台校准 parity 数据背书——**覆盖率已过"广度"阶段,当前真正的资产是"每个数字可信"**。
> 本文取代旧 gap-inventory(SVE/RVV 缺口版,已完成并吸收进 simd-15~18 文档)。
> 正交视角的姊妹文档:[`simd-complexity-opportunities.md`](./simd-complexity-opportunities.md)
> ——本文管"op 面缺什么名字",它管"算法件哪里还能在复杂度层面超过现状"。
>
> **⚠️ 计数口径统一(2026-07-16)**:早先本文用的 "255 个 op / HYLE 侧 393 名" 是差集甄别口径、含水分
> (混 detail helper、正则漏检),**已被自校验脚本 `scripts/simd-op-coverage.py` 的机械口径取代**:
> **公开/selector 名 387 → 可测全集 344**（− 26 不可测 − 13 内部选择器 − 4 别名）
> **→ benchable 内独立覆盖 314 + 30 显式非独立项**。旧 41 候选已经由
> `scripts/simd-op-manifest.json` 全量分类；新增的 4 个 128-bit upper 合成接口也进入同一 manifest。
> "缺什么名字"(本文,vs Highway 语义差集)与"哪些还没写 perf bench"(那 41 个,见
> [`simd-orderswap-matrix.md`](./simd-orderswap-matrix.md) "41 个待补 bench" 节)是两个正交问题——本文管前者。

## 方法

Highway 侧:`generic_ops/x86_256/x86_128/arm_neon-inl.h` 提取全部 `HWY_API` 公开名(407,含大量
重载/内部助手膨胀);HYLE 侧机械口径见 `scripts/simd-op-coverage.py`(383 公开/selector 名,花括号深度栈剔
`namespace detail`)。差集逐项 grep 甄别后剔除三类水分:①Highway 内部积木(LoadTransposedBlocks*/
Native*/AVX2Shl* 等,HYLE 以 LoadInterleaved 等直接暴露等价能力);②HYLE 已有但差集正则漏检的
(CompressBits/SumsOfShuffledQuadAbsDiff/PromoteInRangeTo 均在);③换名等价。**真实缺口 ≈ 40 名,聚类如下。**

## 现状总表

| 维度 | 状态 |
|---|---|
| 非掩码 op 面(算术/swizzle/mask 域/widen/gather/interleave/crypto) | 齐平,零散缺口个位数(见 P2/P3) |
| 变宽 SVE/RVV vs 定宽 | 已收口(simd-15~18;SVE 19 tests×4VL、RVV 18×4VLEN 全绿) |
| 数学函数面 vs xsimd/sleef | simd-09 已收(gamma/hypot best-of 等) |
| 掩码 op 面 | 完整：基础算术/逻辑/比较/一元 + complex/FMA/reduce/gather/shift 均已有 |
| 数字可信度 | op_parity 校准仪器 v2,三平台 CALIBRATION PASS,真输清单见 simd-orderswap-matrix.md |

## P1:掩码基础算术/逻辑/比较家族（✅ 已有，旧差集误报）

`vec.h:423` 起已经生成 MaskedAdd/Sub/Mul/Div/Min/Max、SaturatedAdd/Sub、And/Or/Xor 及全部
`*Or`；Eq/Ne/Lt/Le/Gt/Ge 返回掩码；Abs/Sqrt 与 `*Or` 也齐全。`Vec.MaskedExpandedOps`
正确性测试覆盖代表合同。它们是共享泛型层，编译器在 AVX-512/SVE/RVV 上可折成谓词指令；若未来
对拍证明某编译器不能融合，再按有数据的具体 op 加 native override，不预先复制七后端实现。

为什么这些“平凡组合”仍值得保留为公开名字（不是名字数 parity，而是能力面）：

1. **AVX-512 上它们是单条掩码指令**(`vpaddd {k}{z}`)。走 `IfThenElseZero(m, Add(a,b))` 组合
   等于赌编译器把 blend+op 融合成掩码指令——GCC/Clang 常融,**MSVC 不可靠**(op_parity 三轮
   调查中 MSVC 组合上下文劣化是惯犯)。HYLE 有 avx512 后端,却没有暴露这层直通面。
2. **SVE/RVV 的原生模型就是谓词化**(`svadd_z` 直接吃 predicate,RVV 同理)。刚追平的两个
   scalable 后端,这个家族是其天然表面,现在反而只能绕组合。
3. 分层成本低,正合 HYLE 两层策略:泛型层一份 IfThenElse 合成 covers scalar/sse4/avx2/neon,
   仅 avx512/sve/rvv 各写原生特化;oracle 直接由既有 Masked* 测试模式复制。

## P2:小缺口复核（✅ 消费者能力已闭合）

- **Eq128Upper / Ne128Upper / Max128Upper / Min128Upper** 已补齐并由 `Vec.Compare128` 覆盖；它们
  组合既有比较/选择积木，所以 manifest 归为 synthetic，不为凑数加 L1 benchmark。
- **WidenLowMulAdd / WidenHighMulAdd / WidenMulAdd** 在 Highway ARM 文件中是 `namespace detail`
  的内部积木，不是需要镜像的普通能力。HYLE 已公开更完整的 `WidenMulAccumulate`、
  `WidenMulPairwiseAdd`、`SatWidenMulPairwiseAccumulate`；没有消费者需要手工拆 low/high，故不新增名字。

## P3:机会型(更多硬件门控)

- **GFNI:GaloisAffine(+GF2P8 乘)**:Icelake+/Zen4+,走 AES/CLMul 既有先例(SDE 真跑验证)。
  收益不只是新 op:GF(2^8) 仿射可反哺 ReverseBits/子字节置换等现有 op 的更优实现。

## 明确不做(判据:缺能力类或有消费者才加,名字数不是 KPI)

- ShuffleTwo1230/2301/3012、PerBlock2x2MatMul、LoadTransposedBlocks*/StoreTransposedBlocks*:
  Highway 内部积木/特化助手,HYLE 已有等价组合面。
- 为了镜像 Highway 全名单加 op。**成本模型:每个 op = 永久的 7 后端实现 × scalar oracle ×
  三平台 parity 监控审计面**——op_parity 调查花三轮才把 255 个 op 的 parity 表洗到"数字直接
  可信",面越大洗得越贵。

## 复核方式

本文的差集可复现:对 Highway 头提取 `HWY_API` 名集,对 HYLE vec.h+后端头提取公开模板函数名集,
差集后逐名 `grep -rc` 甄别(警惕正则漏检:宏生成名/跨文件定义会假阴,逐项对源码核实——同
simd 缺口方法纪律)。
