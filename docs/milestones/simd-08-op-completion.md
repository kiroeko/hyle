# simd-08：op 面补全至 Highway 能力面（A/B/C 三档 + 泛型层策略）

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **前置** | simd-07（op 穷尽 ~90）、simd-02/03（五后端）、simd-05（int lane 广度） |
| **状态** | ✅ 完成（2026-07-09，commit 086c5cb~b38776a + 修复 3022868）。Win MSVC vec_test 193 / 每 ISA 变体 163；WSL GCC14.2 sse4/avx2/sse4_aes 163 + ASan 零错误；mac AppleClang neon 变体 163（含 ARM crypto 硬件真跑）；AVX512 后续已由 Intel SDE 执行对拍并确认 kAvx512 派发，性能仍只接受真机 benchmark |

> **范围**：把 op 面从 simd-07 的 ~90 补到 ~217 个公开 op 名（对齐 Highway ~370 的typed变体能力面——我们的泛型模板一名多型，故名数更少但覆盖等价），补齐 A 档（结构性五后端）+ B 档（泛型层为主）+ C 档（Crypto/Complex）。**用户驱动**：simd-07 后发现"200 多个算子只做了 90 多个"，用户定"我们是基础库，不能总等消费者——做全"。
> **cloc**：simd-08 全量 +4,709 行（9 文件）。五后端头增至 scalar 1176 / sse4 1561 / avx2 1643 / avx512 1473 / neon 1871（均在 ~2k 设计上限内）+ crypto_soft.h 111 + crypto_x86.h 82（共享软件参照 + x86 worker，消除 512B 表五后端重复）+ vec.h 泛型层 +348 + vec_test +768（测试 138→163 每变体）。

## 为什么是它（基础库的能力面纪律）

simd-07 把 op 从 ~35 补到 ~90，但对照 Highway 的 ~370 typed op 仍是子集：掩码全家族（Masked*）、复数算术、字节表查、加宽点积的完整形态、crypto 原语都缺。**用户认知（2026-07-08）**：基础库的消费者是"整个世界"，能力要提供到位，不能拿"暂无消费者"当后置理由——那是应用层逻辑。simd-07 已确立此原则，simd-08 把它贯彻到底：**做全**。

## 核心决策：两层策略（结构性自造 vs 泛型层一份covers五后端）

simd-08 最重要的方法学产出，是把 op 分成两类分别处理：

### 1. 结构性 op = 自造 per-backend（A 档）
Set/Load/Add/Shuffle/Mask/BitCast/TableLookupBytes/widen-dot/Interleaved 这类**贴硬件**的 op，每后端照 Highway 的 per-op intrinsic 契约手写（抄契约不 vendor Highway——它的 ops/*.h 是万行重宏，违背"贴硬件、~2k 行/后端"上限）。scalar 后端是 oracle。

- **A-1 BitCast**（地基）：类型双关，后续掩码域/crypto 都依赖
- **A-2 Mask 域逻辑**：And/Or/Xor/AndNot/Not/MaskFromVec/BitsFromMask/LoadMaskBits/StoreMaskBits/前缀扫描/FindKnown*
- **A-3 FP 分类**：IsNaN/IsInf/IsFinite/IsEitherNaN（修了 VecFromMask float 潜伏 bug + IsEitherNaN NEON 编译顺序 bug——mac clang 揪出 x86 两阶段查找漏的前向引用）
- **A-4 TableLookupBytes**：PSHUFB 风格字节 LUT
- **A-5 widen-dot 量化核**：SatWidenMulPairwiseAdd/ReorderWidenMulAccumulate/SumOfMulQuadAccumulate（int8/16 GEMM）
- **A-6 Interleaved Load/Store 2/3/4**：AoS↔SoA
- **A-7 math/arith 增强**：ApproximateReciprocal(Sqrt)/GetExponent/MulByPow2/MinNumber/MaxNumber/AddSub/MulRound/SaturatedNeg/Abs（vrecpeq ARM 精度 ~0.06% 超 1e-3 容差——Highway 文档"~1% on Arm"，改 1e-2）

### 2. 泛型层 = 一份定义 covers 五后端（B 档 + Complex）
Masked*/lane-access/位融合/复数这类**能用已有结构 op 组合表达**的，在 vec.h 的 `#include <后端>` **之后**定义一次，靠重载决议对五后端全生效（Highway 的 generic_ops 模型）。避免五后端重复。

- **B-1 Masked 全家族**：~40 个 Masked* 变体（宏生成），一份覆盖五后端——泛型层策略的最强验证
- **B-2 swizzle 补全**：Reverse4/8/ReverseBits/ReverseLaneBytes/InterleaveEven/Odd/Slide1Up/Down/SlideUp/DownLanes/BroadcastLane/ZipLower/Upper（五后端，含硬件 shuffle）
- **B-3 conversion/lane 补全**：TruncateTo/PromoteEvenTo/PromoteOddTo/U8FromU32/SumsOf2/4（窄化 store+reload）
- **B-4 位/符号/reduce**：Xor3/Or3/OrAnd/BitwiseIfThenElse/IsNegative/CopySignToAbs/SumOfLanes/Min/MaxOfLanes/Rol/Ror/TestBit/RoundingShiftRight
- **B-5 compress/gather/masked-memory**：CompressNot/CompressBits(Store)/LoadExpand/GatherOffset/ScatterOffset/MaskedGatherIndex(Or)/LoadNOr/SafeFillN/SafeCopyN（store+reload 正确优先，vpgather/vpcompress 是 profiling 升级）
- **Complex**：ComplexConj/MulComplex/MulComplexConj/MulComplexAdd/MulComplexConjAdd

### 3. C 档 Crypto = 软件参照 + 硬件路径 feature-gated（借 best-of 语义）
AES 轮原语（AESRound/AESLastRound/AESRoundInv/AESLastRoundInv/AESInvMixColumns/AESKeyGenAssist<kRcon>）+ 无进位乘（CLMulLower/Upper），完整对齐 Highway 8 个 crypto op：

- **共享软件参照** `crypto_soft.h`：FIPS-197 S-box/GF(2^8)/round core，五后端复用一份（消除 512B 表重复），AES-NI 字节序 → 软硬 bit 一致
- **x86 硬件** `crypto_x86.h`：sse4/avx2/avx512 共享 128-bit AES-NI/PCLMUL worker；avx2/avx512 按 2/4 个 128-bit 块拆分（Highway x86_256/512 模型，**不依赖 VAES**——从常驻 AES-NI 构建，VAES 是 profiling 升级）
- **ARM 硬件**：neon 用 `vaesmcq_u8(vaeseq_u8(s,0))^key` **修正 AddRoundKey 顺序差异**（AESE 自带 AddRoundKey，与 AES-NI 末尾 XOR 语义不同——抄 Highway arm_neon 技巧），PMULL 走 vmull_p64/vmull_high_p64
- **feature-gating**：AES-NI/PCLMUL(x86 `__AES__`/`__PCLMUL__`) 与 ARM crypto(`__ARM_FEATURE_AES`) 是正交于 SSE/AVX/NEON tier 的独立扩展；GCC/Clang 需 `-maes -mpclmul`/`+crypto`，MSVC 无条件有。缺失时回落软件参照——**op 契约始终完整**（基础库）
- **契约**：AES 作用于 128-bit 块（sizeof(T)*N == 16/32/64 per backend）

## store+reload 正确优先原则

B/C 档的仿真路径（gather/compress/swizzle/quad-dot/crypto 软件回落/窄化转换）统一用 store-to-stack + scalar-loop + reload；原生 intrinsic（vpgather/vpcompress/vpshufb/vpdpbusd/AES-NI）注释标记为 profiling 升级。正确性先行，性能后调。

## 真机纪律 + 三平台硬件覆盖

| 平台 | 编译器 | crypto 路径 | 验证 |
|---|---|---|---|
| Win MSVC | cl 19.50 | AES-NI 无条件 | vec_test 193 / sse4·avx2·sse4_aes·scalar 163 全绿 |
| WSL Ubuntu | g++-14 | 基线=软件回落；sse4_aes 变体 `-maes -mpclmul`=硬件 | 163 全绿 + **ASan 零错误** |
| mac Apple Silicon | AppleClang | `__ARM_FEATURE_AES` 默认开=ARM crypto 硬件真跑 | neon 变体 163 全绿 |

新增 `sse4_aes` 测试变体专门坐实 GCC/Clang 的 AES-NI/PCLMUL 硬件路径（基线变体走软件回落，二者同对拍 in-test FIPS-197 oracle → 证软硬 bit 一致）。`hyle_isa_flags` 加 `SSE42_AES` tier。

## 修复：TruncateTo/Promote*/U8FromU32 栈越界读（既存 B-3 bug，commit 3022868）

**决定性发现**：档C 收口时 WSL GCC 跑全套，`Vec.TruncateAndStridedPromote` 段错误（exit 139），但 MSVC + scalar + 早前 mac 全绿。根因:store+reload 的 scratch 缓冲按逻辑 lane 数 `dst[N]` 开（如 Vec128<u8,4>=4 字节），但 Store/Load 搬整宽寄存器（`_mm_load_si128`=16B / `vld1q`=16B / `_mm256`=32B / `_mm512`=64B）→ 窄结果 reload 时整宽读 = **栈 OOB**。MSVC/scalar/mac 靠栈布局侥幸没崩。

修法:四后端 scratch 统一开到整寄存器宽（16/32/64B），dst 清零。**教训**:这是"单平台数当无条件结论"陷阱——B-3 当初只在 MSVC 验证。ASan 是根治闸门(修后 x86 三变体零错误),不靠"撞不撞崩"。

## 交付物

- `include/hyle/simd/vec.h`：泛型层 +348 行（Masked/lane-access/位融合/Complex/compress-memory/RebindTo* 别名）
- `include/hyle/simd/backend/{scalar,sse4,avx2,avx512,neon}.h`：A 档结构 op + B/C 档后端特化
- `include/hyle/simd/backend/crypto_soft.h`（111）+ `crypto_x86.h`（82）：crypto 共享参照 + x86 worker
- `tests/simd/vec_test.cc`：+768 行，每 ISA 变体 163 测试
- `cmake/HyleModule.cmake`：`SSE42_AES` tier；`tests/CMakeLists.txt`：`sse4_aes` 变体
