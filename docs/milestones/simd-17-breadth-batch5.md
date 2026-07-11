# simd-17 批5：复数超越 ARM 补全 + 偏门族评估落地

| 项 | 内容 |
|---|---|
| 状态 | 🔴 未开工（opus 出卡 / grok 实现 / opus 四查）；含评估分诊 |
| 前置 | simd-16 |
| 范围 | `src/simd/math/`（复数）+ 定宽后端（可做的偏门 op）+ 测试 |
| 目标 | ①`cexp/clog/csqrt` ARM SoA 补全 ②偏门族逐个分诊：能做的做、真硬件门控的登记 |

## 0. 分工

- grok 实现可做部分 / opus 四查 + 复核硬件门控分诊是否诚实。
- **本批含"评估"成分**：偏门族要逐个判"可做 vs 硬件门控"，grok 拿不准的标记等 opus。

## 1. A 部分：复数超越 ARM 补全（cexp/clog/csqrt）

### 现状（实测 math_dispatch.cc:172）
- **已有 hyle_neon SoA**：`csin/ccos/ctanh/cpow`（4 个，simd-13）。
- **落 scalar 的 ARM 缺口**：`cexp/clog/csqrt`——x86 走 xsimd AVX2，非 x86（ARM）落 `std::complex` scalar。这是 gap-inventory 记的"ARM 剩余 3 个复数超越"。

### 手法（抄已有 hyle_neon 的 SoA 编排风格）
grok 先读 `transcendental_xsimd_neon.cc` 里 `csin_f32_hyle_neon` 的 SoA 结构（拆实部/虚部数组，用 NEON f32 batch 数学核编排），照它加三个：

- **cexp(z) = e^(zr) · (cos(zi) + i·sin(zi))**：`exp_f32_batch(zr)` × `sincos(zi)`。复用已有 `exp`/`sin`/`cos` NEON 核。异常 lane（inf/NaN）逐 lane 回退 `std::complex`。
- **clog(z) = log|z| + i·arg(z)** = `0.5·log(zr²+zi²) + i·atan2(zi,zr)`。需 `log_f32_batch` + `atan2`。**若 NEON 无 atan2 batch 核**，用 `atan` + 象限校正合成，或该 lane 回退 scalar。grok 先查 NEON 有无 atan2/atan 核。
- **csqrt(z)**：标准公式 `sqrt((|z|+zr)/2) + i·sign(zi)·sqrt((|z|-zr)/2)`。复用 `sqrt`/`hypot`。branch-cut（负实轴）逐 lane 回退 scalar。

### 落点
- `src/simd/math/transcendental_xsimd_neon.cc`（虽名 xsimd 但含 hyle_neon SoA；或按现有文件组织）加 `cexp_f32_hyle_neon` 等。
- `math_dispatch.cc` 把 ARM 分支从 `_scalar` 改指 `_hyle_neon`（参照 csin 的 dispatch 结构 :200）。
- **契约**：异常/branch-cut lane 逐 lane `std::complex` 回退（同 ctanh/cpow 已有做法），主域向量化。

### 验收
- `math_test` 对拍 `std::complex<float>` 的 cexp/clog/csqrt（现有复数测试同款容差）。
- **mac NEON 真机**（Apple Silicon，reference_mac_ssh）——这是 ARM 补全，必须真机跑，非 skip。
- 覆盖 branch-cut（clog 负实轴、csqrt 负实轴）+ 异常（inf/NaN/0）。

## 2. B 部分：偏门族逐个分诊

> gap-inventory §2 OBSCURE 列的族，用户要求"明确不做的也做"。逐个判可做性——**能真跑验证的做，纯硬件门控无模拟器通道的登记**（同 simd-16 纪律：不拿"无硬件"当借口，但坐实工具链边界）。

| 族 | 分诊 | 依据 |
|---|---|---|
| `SumsOf8AbsDiff` / `SumsOfAbsDiff` | 🟢 **可做** | x86 `_mm256_sad_epu8`（SSE/AVX2 基线，非门控）；NEON `vabdq`+`vpaddlq`。SAD 是真实用途（运动估计/相似度）。**本批做** |
| `GaloisAffine` / `GaloisAffineTranspose` | 🔒 **GFNI 门控** | `_mm512_gf2p8affine_epi64_epi8` 需 `HWY_TARGET <= HWY_AVX3_DL`（GFNI）。SDE 可仿真验正确性 → **可做但走 SDE tier**，标记 |
| `BitShuffle` | 🔒 **AVX512-only** | `HWY_IF_V_SIZE_V(V,64)` + BITALG/VBMI。SDE 仿真 → 可做 SDE-gated |
| `MultiRotateRight` | 🔒 **AVX512-only** | 同上，512 位 + idx 向量。SDE-gated |
| `*Blocks` 族（BroadcastBlock/ExtractBlock/InterleaveWholeLower/SwapAdjacentBlocks/...） | ⚪ **512-block 机制** | Highway 512-bit block-lane 抽象。HYLE avx512 是单一 512 无 block 分层——**语义可定义**（每 128/256 子块），但**越 HYLE"贴硬件不过度抽象"scope**。逐个评估：BroadcastBlock/ExtractBlock 有实际用途可做；SwapAdjacentBlocks 等纯 Highway 内部机制登记不做 |
| `SumsOfShuffledQuadAbsDiff` / `SumsOfAdjQuadAbsDiff` | 🔒 **dbsad 门控** | `_mm256_maskz_dbsad_epu8`（AVX512-BW）。视频专用极窄。SDE-gated 或登记 |

### B 部分本批实际做的
1. **`SumsOf8AbsDiff`(a,b) / `SumsOf8AbsDiff`(v)（vs 0）**：五定宽后端。x86 SAD 指令、NEON vabd+vpaddl、scalar oracle。真实用途，双平台+SDE+mac 真跑。
2. **`GaloisAffine` + `BitShuffle` + `MultiRotateRight`**：AVX512 后端实现，**Intel SDE tier 验正确性**（同 AVX512 既有 SDE 先例）。GFNI/BITALG 用 `-march` 门控 + `#if` 保护，不污染非 DL 变体。
3. **`BroadcastBlock<k>` / `ExtractBlock<k>`**：avx512（512=4×128 block）+ avx2（256=2×128）。有用途（block 广播/提取），可做。其余 `*Blocks`（SwapAdjacent/InterleaveWhole/LoadTransposed）**登记不做**（纯 Highway 512 内部机制，越 scope，无消费者且非能力缺口）。

### B 部分登记不做（坐实理由，非借口）
| 项 | 理由 |
|---|---|
| `SwapAdjacentBlocks`/`InterleaveWholeLower/Upper`/`LoadTransposedBlocks` | Highway 512-block 内部编排机制，HYLE 无 block 分层抽象（贴硬件 scope），无消费者 |
| `SumsOfShuffledQuadAbsDiff` | dbsad 视频专用极窄，SDE 可验但无 HYLE 消费者，登记 SDE-gated 待需 |

## 3. 验收

- **A（复数）**：math_test 对拍 std::complex，mac NEON 真机 cexp/clog/csqrt 全绿。
- **B（SAD）**：vec_test 对拍 scalar SAD oracle，双平台+SDE+mac。
- **B（GaloisAffine/BitShuffle/MultiRotate）**：AVX512 SDE tier 正确性（对拍 scalar 位运算 oracle）。objdump 确认 `gf2p8affine`/`vpshufbitqmb` 真发射。
- **B（BroadcastBlock/ExtractBlock）**：多 block 独立性验证。

## 4. 输出

本批产出两类：**代码**（复数 ARM + SAD + block 提取 + AVX512-DL 族）+ **分诊登记**（哪些偏门做了、哪些坐实不做及理由）。收口时更新 gap-inventory §2 OBSCURE 分类为"已做 / SDE-gated / 坐实不做"。

## 5. clay 覆盖

复数超越是 HYLE 独有（Highway/xsimd 无完整复数路径）；SAD/block/GFNI 是定宽广度补全。均不改 clay 结论。
