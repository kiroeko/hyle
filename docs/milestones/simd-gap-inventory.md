# SIMD op 面覆盖率清单(2026-07-14 实证审计)

> 结论先行（2026-07-17 复核）:HYLE **公开/selector op 名 387 个**(机械去重),寄存器原语面与
> Highway 已基本齐平。旧差集声称缺失的 ~28 个掩码基础算术名其实已经由 `vec.h` 的泛型
> `Masked*` 家族完整实现并有 oracle；本轮又补齐 4 个 upper-only 128-bit key 接口。
> **但“寄存器 op 齐平”不等于“普通用户能力齐平”**：按 whole-buffer/C ABI 语义重新扫描
> Highway contrib 与 xsimd 后，仍有 search/reduction、混合精度 dot、排序类型和少量数学能力候选。
> 本文新增的 P0/P1 表是这些候选的唯一账本；不能再用 387-name 结论把它们误报为“没有 API 空洞”。
> 变宽 SVE/RVV 的 **P0~P4 正确性与 generic 实例化**已在 simd-15~18 收口；这不代表后端本地
> 名字、Runtime 产品 kernel 或真机性能证据与定宽同档。后续语义分类、`dtype/math/sort/bitpack`
> scalable 产品核和性能证据由 [`simd-25`](./simd-25-scalable-product-parity.md) 收口。经 op_parity 仪器重建(见
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
| 变宽 SVE/RVV vs 定宽 | 寄存器 P0~P4 正确性已收口；产品层/性能层由 simd-25 进行中 |
| 数学函数面 vs xsimd/sleef | simd-09 已收(gamma/hypot best-of 等) |
| 掩码 op 面 | 完整：基础算术/逻辑/比较/一元 + complex/FMA/reduce/gather/shift 均已有 |
| 数字可信度 | op_parity 校准仪器 v2,三平台 CALIBRATION PASS,真输清单见 simd-orderswap-matrix.md |

## 2026-07-17 Highway/xsimd whole-buffer 复扫

### 审计边界

- Highway：本地参考 checkout `83eff39371e5e6494b305b253f8a3c06f831faf4`；逐目录检查
  `hwy/contrib/{algo,dot,bit_pack,hash,image,intdiv,math,matvec,random,sort,thread_pool,unroller}`。
- xsimd：本地参考 checkout `5d2490f515cf773f984da9ec0a1aec103d00d238`；检查
  `types/xsimd_api.hpp` 和 arch/common 实现。该 checkout 的最新提交正是完整 64-bit
  `mul_hilo`，所以不能只复用此前的 Highway 名字差集。
- 筛选单位是“能否成为 pointer/count/POD C ABI 语义”或“能否改善现有 kernel”，不是名字相似度。
  generic lambda、STL 容器、第三方线程池和无消费者的内部 helper 不因上游存在就进入 HYLE。

### 候选结论

| 优先级 | 上游能力 | HYLE 现状 / 判断 | 动作 |
|---|---|---|---|
| P0 | Highway 2026 contrib/algo：`Count`、`Find`、`MinValue`、`MaxValue`、`IsSorted` | HYLE 的连续数组公共语义只有 add/dot/saxpy 等，缺最常见 search/reduction。它们天然是 pointer/count C ABI，普通用户价值高 | 增加 typed whole-buffer search/reduction core、动态/固定双轨和 Highway/stdlib benchmark；close-01 再封 C 名 |
| P0 | HYLE vqsort 大数组 partition | ✅ simd-24 已删除 HeapSort 保险丝；AVX2 u64/KV 根因是缺少 pair-vpermd Compress，已补真 partition | Windows、WSL ASan、macOS NEON 正确性已过；NEON KV64 仍约慢 Highway 27%，继续性能狩猎，详见 simd-24 |
| P1 | Highway contrib/dot：f16×f16、bf16×bf16、f32×bf16、i16×i16 | HYLE whole-buffer dot 只有 f32/f64；底层已有 widen/dtype 积木，但没有融合入口 | 按 AI/量化消费者优先补 bf16、mixed bf16、i16；f16 由真机收益决定，禁止“先转换整表再 dot”的额外内存 pass |
| P1 | Highway vqsort key 面：i16/u16/f16/u128 | HYLE public sort 只有 i/u32、i/u64、f32/f64、KV32/64；port 内部已含若干 traits，但 f16 目前明确是 placeholder | i16/u16 与 128-bit POD key 先做；f16 必须统一 HYLE dtype/排序 NaN 合同后再做 |
| ✅ | xsimd `mul_hi` / fused `mul_hilo`，含 i/u8、i/u32、i/u64 | 已统一为 HYLE `MulHigh` 全整数宽度 + `MulHighLow`；删 `MulHigh32` 公开特例，widen/64-bit 软件路径共享 partial，scalable 用原生 low/high 指令 | register/native 能力已完成并进入 parity bench；不进动态 C ABI |
| P2 | xsimd complex trig/hyperbolic 与 f64 complex | HYLE 只有 f32 `cexp/clog/csqrt/csin/ccos/ctanh/cpow`；旧注释误写已有 ctan/csinh/ccosh | 先纠正文档；有 FFT/solver 消费者后补 ctan/csinh/ccosh 与 f64，不把 trivial conj/real/imag 包装成独立 kernel |
| P2 | xsimd `frexp/ldexp`、mask predicates 等 | 对应寄存器积木多已存在（如 `MulByPow2`、分类和 mask reduce），但无普通 batch 语义 | 只有真实消费者或相对编译器循环的稳定收益才上 whole-buffer API |

### 明确不吸纳

- Highway contrib `thread_pool`：HYLE sched 是唯一 CPU 核仲裁者，不能在 SIMD 内再带一套线程池。
- Highway `matvec`：固定尺寸模板并耦合其 thread pool；HYLE 的 GEMM/matvec 属 compute/ggml 路线。
- Highway contrib `hash`：属于 `container` 的独立 SOTA 证据线，不能反转 `container -> simd` 依赖。
- Highway `image`：是图像所有权/边界扩展容器，不是 SIMD 基础语义。
- `Transform/Generate/Foreach/*If` generic-lambda 外观：适合源码模板库，不可能成为稳定 C ABI；HYLE
  只吸纳其中有稳定语义的具体算法，不发布 callback-in-hot-loop 万能入口。
- `Copy/Fill`：先与编译器 builtin/libc 的尺寸曲线对拍；没有稳定优势就不做第二套 memcpy/memset。
- xsimd 的 aligned allocator、batch/arch type 和运行时 arch list：HYLE 已有 mem、D-tag 与 dispatcher，
  吸纳会制造平行抽象。

### 是否可以阶段性收口

可以阶段性冻结的是 **387-name 寄存器原语扩面**：除有消费者证明的 `mul_hi/mul_hilo` 和 GFNI
机会外，不再按上游名字追加 op。还不能冻结的是 **whole-buffer SIMD 产品层**。至少要先完成：

1. 修复或正式替换 vqsort vector-partition-disabled 路径，并证明 sort/select 不落后正确对手；
2. 对 P0 search/reduction 做 API/性能裁决；
3. 对 mixed-precision dot 给出“实现”或“有证据拒绝”的结论。

三项闭环后，SIMD 算法核才适合暂时告一段落，把普通用户发布入口交给 close-01。P1 的排序扩型和
原先“P2 complex 等消费者”的延期裁决已被 simd-24 取代：HYLE 作为基础库，审计确认的通用候选全部
按阶段实现；优先级只决定顺序，不再决定是否落地。C ABI 仍在 close-01 统一封装。

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
