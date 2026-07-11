# simd-17 批2：运行期移位 *Same + Pairwise + Stream + LoadDup128 + 杂项

| 项 | 内容 |
|---|---|
| 状态 | 🔴 未开工（opus 出卡 / grok 实现 / opus 四查） |
| 前置 | simd-16 |
| 范围 | `backend/{scalar,sse4,avx2,avx512,neon}.h` + `vec_test.cc`。**不动** scalable、不动 vec.h 泛型层 |
| 目标 | 运行期(非编译期常量)移位/旋转 + 水平加减 + 非临时存 + 128 广播 + 浮点指数域 |

## 0. 分工 + 架构事实

- grok 实现 / opus 四查。**照抄卡里 intrinsic 全名不查名**。
- 架构事实同批1：定宽单一 native 宽度，`Vec→Vec` 全宽签名，256/512 的 Pairwise 是 per-128-block。
- **HYLE 已有的可复用件（grok 先 grep 确认）**：`Shl/Shr`（变量移位，vec 版）、`Rol/Ror`（vec.h 泛型旋转）、`ShiftLeft<k>/ShiftRight<k>`（编译期）、`InterleaveEven/Odd`、`Add/Sub`。本批多数 op 是它们的运行期标量参数包装。

## 1. 本批范围

| 组 | op | 语义 |
|---|---|---|
| A. 运行期移位 | `ShiftLeftSame(v,int)` `ShiftRightSame(v,int)` | 每 lane 移**运行期** bits（非模板常量）。整数 lane |
| A. 运行期旋转 | `RotateLeftSame(v,int)` `RotateRightSame(v,int)` | 每 lane 旋 bits。整数 lane |
| A. 舍入右移 | `RoundingShiftRightSame(v,int)` `RoundingShr(v,vAmt)` | 右移带四舍五入（+half 再移）。整数 |
| B. 水平加减 | `PairwiseAdd(d,a,b)` `PairwiseSub(d,a,b)` | 相邻对水平运算，交织 a、b 结果 |
| B. 128 水平 | `PairwiseAdd128(d,a,b)` `PairwiseSub128(d,a,b)` | per-128-block 的水平版 |
| C. 非临时存 | `Stream(v,d,aligned)` | 绕 cache 的对齐存 |
| C. 128 广播 | `LoadDup128(d,p)` | 载入 128 位并广播到全宽每个 128-block |
| D. 浮点指数 | `GetBiasedExponent(v)` | 提取**带偏置**指数域（不去偏置，返回 unsigned）|
| D. 浮点缩放 | `MulByFloorPow2(v,exp)` | v * 2^floor(exp)，float exp |

**op 数：6（A）+ 4（B）+ 2（C）+ 2（D）= 14 个公开名**。

## 2. A 组：运行期移位/旋转施工卡

> **与已有 `<k>` 版的区别**：`ShiftLeft<k>` 是模板常量；`*Same` 收 `int bits` 运行期参数。x86 的 `_mm_slli_epi32` 要求立即数——运行期版用 **`_mm_sll_epi32`（注意无 i）+ `_mm_cvtsi32_si128(bits)`** 把移位量放进向量低 64 位。

### sse4
- `ShiftLeftSame(v,bits)`：整数按宽分派——
  - i16/u16 → `_mm_sll_epi16(v.raw, _mm_cvtsi32_si128(bits))`
  - i32/u32 → `_mm_sll_epi32(...)`
  - i64/u64 → `_mm_sll_epi64(...)`
  - i8/u8 → **无硬件 8 位移位**：widen 到 16 位移位后 mask，或按 Highway 手法（`ShiftLeftSame` 用 16 位移 + `And` 清高位）。参照 Highway x86_128 i8 分支。
- `ShiftRightSame(v,bits)`：unsigned → `_mm_srl_epi*`；signed → `_mm_sra_epi*`（注意 `_mm_sra_epi64` 仅 AVX512，sse4 的 i64 算术右移需合成：`sra` 高 32 + 符号扩展，参照 Highway）。
- `RotateLeftSame/RightSame`：`= Shl/ShrSame 合成`——`RotateRightSame(v,b) = Or(ShiftRightSame(v,b), ShiftLeftSame(v,bits-b))`（b=0 特判返回 v）。或直接 `Ror(v, Set(d, b))` 若 HYLE Ror 收 vec 移位量。**grok 先查 HYLE 有无 vec-amount Ror**；有就一行转发，无则 Shl/Shr 合成。

### avx2 / avx512
- 同 sse4 换 `_mm256_sll_epi*` / `_mm512_sll_epi*`（256/512 的 `sll` 也是全 lane 用同一 count，天然正确）。
- i64 算术右移 avx512 有 `_mm512_sra_epi64` 原生；avx2 无需合成。

### neon
- NEON 移位量是**向量**（`vshlq_s32` 收 per-lane 有符号移位量向量，负=右移）。运行期版：`vshlq_*(v, vdupq_n_*(bits))`（左移）；右移用负量 `vdupq_n_*(-bits)` 或 `vshrq_n` 需常量→用 `vshlq` 负量。参照 neon.h 已有变量 Shl/Shr。
- 旋转同 sse4 用 Shl/Shr 合成。

### RoundingShr / RoundingShiftRightSame（generic 合成，抄 Highway generic_ops:5588）
- 语义：`(v + (1 << (shift-1))) >> shift`（无符号平均式舍入）；shift=0 特判返回 v。
- 实现：`RoundingShiftRightSame(v,s) = ShiftRightSame(Add(v, half), s)` 其中 half 由 s 构造。**照抄 Highway generic 手法**（它用 `ShiftRightSame` + `AverageRound` 思路）。这两个可做成 vec.h 之外的后端各自转发，或直接在每后端用已有 op 合成。

### scalar oracle
```
ShiftLeftSame(v,b): r[i] = v[i] << b  (unsigned 语义按类型)
ShiftRightSame(v,b): signed=算术右移, unsigned=逻辑右移
RotateRightSame(v,b): u=(unsigned)v[i]; r=(u>>b)|(u<<(W-b)), b=0→u
RoundingShiftRightSame(v,s): s=0→v[i]; else (v[i] + (1<<(s-1))) >> s
```

## 3. B 组：Pairwise 水平加减施工卡

> **语义（对齐 Highway generic:2458）**：`PairwiseAdd(d,a,b)` 结果 = 交织 [a 的相邻对和, b 的相邻对和]。Highway 用 `Add(InterleaveEven(a,b), InterleaveOdd(a,b))`——**HYLE 已有 InterleaveEven/Odd**，可直接这么合成，或用 x86 native `_mm_hadd_*`。

### x86 native（更快，优先）
- sse4：`PairwiseAdd` i16→`_mm_hadd_epi16`、i32→`_mm_hadd_epi32`、f32→`_mm_hadd_ps`、f64→`_mm_hadd_pd`。`PairwiseSub`→`_mm_hsub_*`。
- avx2：`_mm256_hadd_*`/`_mm256_hsub_*`（**天然 per-128-block**——正好是 `PairwiseAdd128` 语义！见下）。
- avx512：无 hadd 指令 → 用 generic 合成 `Add(InterleaveEven, InterleaveOdd)`。

### PairwiseAdd vs PairwiseAdd128 语义辨析（关键）
- `PairwiseAdd(d,a,b)`：**全向量**视角，结果 = [a0+a1, a2+a3, ..., b0+b1, ...]（a 的全部对在前，b 的在后）。
- `PairwiseAdd128(d,a,b)`：**per-128-block** 视角，每 128 内 = [a 块对和, b 块对和]。
- **x86 `_mm256_hadd` 天然是 128-block 语义** = `PairwiseAdd128`。`PairwiseAdd`（跨 block 连续）需要额外 permute 重排 block。**grok 必读 Highway x86_256 的 PairwiseAdd 看它如何从 hadd 重排到全向量语义**；拿不准标记等 opus。
- sse4（单 128）两者等价。

### scalar oracle
```
PairwiseAdd(a,b): 前 N/2 = a 相邻对和, 后 N/2 = b 相邻对和
  r[i]     = a[2i] + a[2i+1]   for i in [0,N/2)
  r[N/2+i] = b[2i] + b[2i+1]
PairwiseAdd128: per-128-block 内做上述, block 间独立
PairwiseSub: 换减法 (a[2i] - a[2i+1])
```

## 4. C 组：Stream + LoadDup128

### Stream（非临时对齐存）
- sse4：`_mm_stream_si128((__m128i*)aligned, rawi)`；f32→`_mm_stream_ps`；f64→`_mm_stream_pd`。
- avx2：`_mm256_stream_si256`/`_mm256_stream_ps`/`_mm256_stream_pd`。
- avx512：`_mm512_stream_si512`/`ps`/`pd`。
- neon：无非临时存 → **回退普通 Store**（Highway arm_neon:4213 也是回退）。诚实注释。
- **契约**：`aligned` 必须按向量宽对齐（调用方保证，`HYLE_FATAL_ASSERT` 可加对齐检查）。scalar：等同 Store。

### LoadDup128（载 128 广播到每 block）
- sse4（单 128）：`= LoadU(d,p)`（一个 block，直接载入）。
- avx2：载低 128 → 广播两半。`_mm256_broadcastsi128_si256` / `_mm256_broadcast_ps`（f32）/ `_mm256_broadcast_pd`（f64）。参照 Highway x86_256:4293。
- avx512：`_mm512_broadcast_i32x4` / `_mm512_broadcast_f32x4` 等（载 128 广播 4 份）。
- neon：`= LoadU`（单 128）。
- scalar oracle：`r[i] = p[i % (16/sizeof(T))]`（每 128-block 重复低 128 内容）。

## 5. D 组：浮点指数域

### GetBiasedExponent（返回 unsigned，**不去偏置**）
- 语义（Highway generic:1538）：f32 → `(bits >> 23) & 0xFF`（带偏置，未减 127）；f64 → `(bits >> 52) & 0x7FF`。返回同宽 unsigned 向量。
- **与已有 `GetExponent` 的区别**：`GetExponent`（simd-15 已有）返回**去偏置**的 float 指数；`GetBiasedExponent` 返回**带偏置**的整数域。两者不同，都要有。
- 实现：`BitCast` 到 unsigned → `ShiftRight<23/52>` → `And` 掩码。全后端用已有 op 合成（可提 vec.h，但本批留后端各自；grok 按后端现有 GetExponent 位置附近写）。

### MulByFloorPow2（v * 2^floor(exp)，float exp）
- 语义（Highway generic:1511）：`v * 2^floor(exp)`，exp 是 **float** 向量。特殊值：`MulByFloorPow2(v,NaN)=NaN`、`(0,inf)=NaN`、`(inf,-inf)=NaN`。
- 与 `MulByPow2`（simd-15，exp 是整数向量）区别：这里 exp 是 float，先 floor。
- 实现：`Floor(exp)` → 转整数 → 复用 `MulByPow2`（已有）。特殊值按 Highway 处理。**grok 先确认 MulByPow2 在定宽后端存在**（simd-15 只在 scalable 做了？若定宽无则本批一并补定宽 MulByPow2，或用 ldexp store+reload）。

> ⚠️ **grok 必查**：`MulByPow2` / `GetExponent` 在**定宽**后端是否已存在（simd-15 是 scalable P5）。若定宽缺，MulByFloorPow2 依赖它——需先补定宽 MulByPow2（store+reload ldexp，同 scalable RVV 手法）。标记依赖关系，拿不准问 opus。

## 6. 验收

- scalar oracle 先写。`vec_test.cc` 每 op 用例对拍。
- 运行期移位：测 bits=0/1/中间/W-1；旋转测回绕。
- Pairwise：多 block 独立性（avx2/avx512）；辨明 PairwiseAdd vs 128 语义各自正确。
- Stream：对齐 buffer，值等同 Store。
- GetBiasedExponent：1.0→127(f32)/1023(f64)、2.0→128、0.5→126。
- MulByFloorPow2：(3.0, 2.5)→3*4=12、特殊值 NaN。
- 双平台 + AVX512 SDE + mac NEON；objdump 抽验 `sll`/`hadd`/`stream`/`broadcast` 真发射。

## 7. 边界

- 不动 scalable。
- `RoundingShr(v, vAmt)`（向量移位量版）与 `RoundingShiftRightSame`（标量）都要；vAmt 版用已有变量 `Shr` 合成。
- 若 D 组依赖的定宽 MulByPow2/GetExponent 缺失，本批补定宽版（顺带，因是同族）。

## 8. clay 覆盖

净新增定宽广度，不改 clay 结论。
