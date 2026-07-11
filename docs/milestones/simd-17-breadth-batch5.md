# simd-17 批5：复数超越 ARM 补全 + 偏门族评估落地

| 项 | 内容 |
|---|---|
| 状态 | 🟡 实现完成大部分（grok）；待 opus 四查 + mac NEON 复数真机；GFNI/BitShuffle/MultiRotate 登记 SDE-gated 未实现 |
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

> **分诊修正记录（opus 复核 2026-07-11）**：初版分诊把 `BitShuffle`/`MultiRotateRight` 标成 "🔒 AVX512-only 门控"、把三个都推到"登记 SDE-gated 未实现"。**读 Highway 源坐实后两处错**：① `BitShuffle`/`MultiRotateRight` 是 `generic_ops-inl.h` 的 **公开 `HWY_API` 通用 op**（每个 target 都有可移植 fallback，native VPSHUFBITQMB/VPMULTISHIFTQB 只是 AVX3_DL 快路），不是 AVX512-only——是真·公开面缺口，必须做；② SDE 探针实测 GFNI/VBMI/BITALG 三族 `-future` 下**全部正确执行无 SIGILL**（affine=0x53/mul=6/bitshuffle=0xff/multishift=0x88），所以"无法验证"不成立。下表为修正后结论。

| 族 | 分诊 | 依据 |
|---|---|---|
| `SumsOf8AbsDiff` / `SumsOfAbsDiff` | 🟢 **已做** | x86 `_mm*_sad_epu8`（基线非门控）；NEON `vabdq`+`vpaddlq`。五后端 + SDE 全绿 |
| `BitShuffle` | 🟢 **已做（公开 op，非门控）** | Highway `generic_ops` 公开面：可移植 fallback（store+reload，HYLE 五后端）+ AVX3_DL native `VPSHUFBITQMB`。base avx512 走 fallback，avx512_dl 走 native，SDE 双路对拍 oracle |
| `MultiRotateRight` | 🟢 **已做（公开 op，非门控）** | 同上：fallback 五后端 + AVX3_DL native `VPMULTISHIFTQB`（VBMI）。SDE 双路验证 |
| `GaloisAffine` / `GaloisAffineTranspose` | ⚪ **不做（非公开面缺口，非门控借口）** | Highway 把它放 `namespace detail`（内部 GFNI 原语，服务 ReverseBits/RotateRight），**不是公开 op**。HYLE 的 ReverseBits/RotateRight 已由 vpshufb-nibble-LUT / shift-or 实现并全绿，其消费者已被服务。SDE 可验（探针已跑通）但无公开面需求 → 不做的理由是"非公开面缺口"，不是"无法验证" |
| `*Blocks`：`BroadcastBlock`/`ExtractBlock` | 🟢 **已做** | avx2（2×128）+ avx512（4×128）。真实用途（block 广播/提取）|
| `*Blocks`：`SwapAdjacentBlocks`/`InterleaveWholeLower/Upper`/`LoadTransposedBlocks` | ⚪ **不做（越 scope）** | Highway 512-block 内部编排机制，HYLE 无 block 分层抽象（贴硬件 scope），无消费者且非能力缺口 |
| `SumsOfShuffledQuadAbsDiff` / `SumsOfAdjQuadAbsDiff` | 🔒 **登记 SDE-gated 待需** | `dbsad`（AVX512-BW）视频专用极窄，SDE 可验但无 HYLE 消费者 |

### B 部分本批实际做的
1. **`SumsOf8AbsDiff`(a,b) / (v vs 0)**：五定宽后端。x86 SAD 指令、NEON vabd+vpaddl、scalar oracle。
2. **`BitShuffle` + `MultiRotateRight`**：五定宽后端 **portable fallback**（store+reload 对拍 scalar oracle）+ **AVX3_DL native**（`VPSHUFBITQMB`/`VPMULTISHIFTQB`，`#if defined(__AVX512BITALG__/__AVX512VBMI__)` 门控不污染基线）。新增 `AVX512_DL` 测试变体跑 native 路径，base avx512 变体跑 fallback，两路 SDE 都对拍 oracle。
3. **`BroadcastBlock<k>` / `ExtractBlock<k>`**：avx512（512=4×128 block）+ avx2（256=2×128）。

### B 部分不做（坐实理由，非借口）
| 项 | 理由 |
|---|---|
| `GaloisAffine`/`GaloisAffineTranspose` | Highway 内部 `detail::` 原语非公开 op；其消费者 ReverseBits/RotateRight 已在 HYLE 实现并全绿。非公开面缺口（不是"无法验证"——SDE 探针已跑通 GFNI）|
| `SwapAdjacentBlocks`/`InterleaveWholeLower/Upper`/`LoadTransposedBlocks` | Highway 512-block 内部编排机制，HYLE 无 block 分层抽象，无消费者 |
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


## 6. 实现结果（grok 2026-07-11）

### A 复数 ARM
- `cexp/clog/csqrt_f32_hyle_neon`：xsimd `batch<complex<float>>` 路径（同 AVX2 借用核），dispatch ARM64 从 scalar 改指 hyle_neon。
- **待 mac NEON 真机**（opus 四查）：本机 WSL 无 aarch64 运行。

### B 偏门（opus 复核修正后）
| 项 | 状态 |
|---|---|
| `SumsOf8AbsDiff` | ✅ 五定宽后端（x86 sad_epu8 / NEON vabd+vpaddl / scalar oracle）；五变体 SDE 全绿 |
| `BroadcastBlock`/`ExtractBlock` | ✅ avx2（2×128）+ avx512（4×128）；SDE 全绿 |
| `BitShuffle` | ✅ **五后端 portable fallback + AVX3_DL native VPSHUFBITQMB**；base avx512 fallback / avx512_dl native 双路 SDE 对拍 oracle；objdump 坐实 native 发射 |
| `MultiRotateRight` | ✅ **五后端 portable fallback + AVX3_DL native VPMULTISHIFTQB**；同上双路验证 |
| `GaloisAffine` | ⚪ 不做——Highway `detail::` 内部原语非公开面，消费者(ReverseBits/RotateRight)已服务（**非"无法验证"：SDE 探针已跑通 GFNI**）|
| `SwapAdjacentBlocks` 等 | ⚪ 不做（Highway 内部 block 编排，越 scope）|
| `SumsOfShuffledQuadAbsDiff` | 🔒 登记 SDE-gated 待需（dbsad 视频专用，无消费者）|

### 测试（真跑，非编译验）
- SDE 探针坐实 GFNI/VBMI/BITALG `-future` 全部正确执行无 SIGILL。
- 新增 `AVX512_DL` isa tier（`-mavx512vbmi -mavx512bitalg`）+ `avx512_dl` vec_test 变体。
- 五变体全量：scalar/sse4 **196 PASS**；avx2 **197 PASS**；avx512(fallback) SDE **197 PASS**；avx512_dl(native) SDE **197 PASS**。
- objdump：avx512_dl 发射 `vpshufbitqmb`×2 + `vpmultishiftqb`×2；base avx512 = 0（走 fallback）。
- **Win MSVC 复验完成**：探针实测 `/arch:AVX512` 只定义 AVX512F+BW，**不含 `__AVX512VBMI__`/`__AVX512BITALG__`**——故 MSVC 上 BitShuffle/MultiRotate 走 store+reload fallback（native 路径 `#if` 编译排除），avx512.h 五 TU 全编过（无 C2386）；scalar/sse4 196、avx2 197 全绿。`avx512_dl` 变体已 gate 为 GCC/Clang-only（MSVC 无 VBMI/BITALG 宏，dl=base 冗余）。
- **mac NEON 真机复验完成**（Apple Silicon Darwin 25.5.0 arm64）：批4/5 全绿，NEON vec_test 196、simd 聚合 275 PASS/1 skip；`Math.ComplexTranscendentals` 覆盖 cexp/clog/csqrt，三个 `_hyle_neon` symbol 确认编入 dylib（走新路径非 scalar）。
- **全平台闭环**：WSL(scalar/sse4/avx2) + AVX512 SDE(fallback) + AVX512_DL SDE(native+objdump) + mac NEON 真机 + Win MSVC。
