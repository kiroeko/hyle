# simd-17 批1：Shuffle 定模式 + 字节/lane 移位族（定宽五后端）

| 项 | 内容 |
|---|---|
| 状态 | 🔴 未开工（opus 出卡 / grok 实现 / opus 四查） |
| 前置 | simd-16 |
| 范围 | `backend/{scalar,sse4,avx2,avx512,neon}.h` + `vec_test.cc`。**不动** scalable（SVE/RVV 归批3+）、不动 vec.h 泛型层 |
| 目标 | 追平 Highway 定宽广度：定模式 shuffle + 字节/lane 移位族 |

## 0. 协作分工 + 这份卡怎么用

- **grok（实现方）**：按 §2/§3 逐 op 卡填五个定宽后端 + scalar oracle，加 `vec_test.cc` 用例。intrinsic 全名、shuffle 立即数、oracle、验收都钉死，**照抄不查名**。
- **opus（校验方）**：四查（编译过 / 双平台+SDE+mac / 对拍 scalar oracle / objdump 无降级）。

> ⚠️ **opus 已探明的架构事实（grok 直接采信）**：
> - HYLE 定宽后端是**单一 native 宽度**：`Vec256<T,N>` 只含一个全宽 `raw`，`static_assert(kNativeWidth)`。**没有 partial 向量**——所以本批全是 `Vec→Vec` 全宽签名，比 Highway 的 partial 分支简单。
> - **256/512 位的 shuffle/字节移位是 per-128-bit-block**：`_mm256_shuffle_epi32` / `_mm256_slli_si256` / `_mm512_bslli_epi128` 天然按每个 128-bit lane 独立操作。**HYLE 必须对齐这个 Highway 契约**（scalar oracle 也按 128-bit block 建模）。
> - scalar `Vec<T,N>` = `T raw[N]`（oracle 直接按逻辑索引算）。
> - neon 无 256 位；`Vec128<T,N>`（16 字节）是其 native。字节移位用 `vextq_u8`。

## 1. 本批范围

| 组 | op | lane 限制 | 语义 |
|---|---|---|---|
| A. 32-bit shuffle | `Shuffle2301` `Shuffle1032` `Shuffle0321` `Shuffle2103` `Shuffle0123` | 仅 sizeof(T)==4（i32/u32/f32） | per-128-block 的 4-lane 定 permute |
| A. 64-bit shuffle | `Shuffle01` | 仅 sizeof(T)==8（i64/u64/f64） | per-128-block 交换两 64-lane |
| B. 字节移位 | `ShiftLeftBytes<k>` `ShiftRightBytes<k>` | 任意 lane（按字节） | per-128-block 移 k 字节，补 0 |
| B. lane 移位 | `ShiftLeftLanes<k>` `ShiftRightLanes<k>` | 任意 lane | = ShiftBytes<k*sizeof(T)> 的 lane 包装 |
| B. 组合移位 | `CombineShiftRightBytes<k>` `CombineShiftRightLanes<k>` | 任意 lane | per-128-block：`[lo,hi]` 拼接后右移 k 字节取低 128 |

**op 数：5（32-bit shuffle）+ 1（Shuffle01）+ 6（移位族）= 12 个公开名**。

## 2. Shuffle 组施工卡（A）

### 语义（务必对齐 Highway，scalar oracle 用它建模）
Highway 的 4-lane shuffle 命名 `ShuffleWXYZ` = 结果 lane 从低到高取源 lane `[Z,Y,X,W]` 反读（Intel 立即数惯例）。**直接抄 Highway 的立即数，不要自己推**：

| op | x86 立即数 | 语义（每 128-block 内，lane 0..3） | NEON(128) |
|---|---|---|---|
| `Shuffle2301` | `0xB1` | [1,0,3,2] 相邻对交换 | `vrev64q_*32` |
| `Shuffle1032` | `0x4E` | [2,3,0,1] 交换两 64 位半 | `CombineShiftRightBytes<8>(v,v)` |
| `Shuffle0321` | `0x39` | [1,2,3,0] 循环右移一 lane | 见下 |
| `Shuffle2103` | `0x93` | [3,0,1,2] 循环左移一 lane | 见下 |
| `Shuffle0123` | `0x1B` | [3,2,1,0] 全反 | 见下 |
| `Shuffle01`(64位) | `0x4E`(epi32) | [1,0] 交换两 64-lane | `CombineShiftRightBytes<8>(v,v)` |

### sse4（128 位，单 block）
- **签名**：`template <class V> HYLE_FORCE_INLINE V Shuffle2301(V v) noexcept`，`static_assert(sizeof(TFromV<V>)==4)`。
- **实现**：i32/u32 → `_mm_shuffle_epi32(v.raw, 0xB1)`；f32 → `_mm_shuffle_ps(v.raw,v.raw,0xB1)`。其余 op 换立即数。
- `Shuffle01`：`static_assert(sizeof(T)==8)`；i64/u64 → `_mm_shuffle_epi32(v.raw,0x4E)`；f64 → `_mm_shuffle_pd(v.raw,v.raw,1)`。

### avx2（256 位，per-128-block）
- i32/u32 → `_mm256_shuffle_epi32(v.raw, 0xB1)`（天然每 128 半独立同模式，正是要的 block 语义）；f32 → `_mm256_shuffle_ps(v.raw,v.raw,0xB1)`。
- `Shuffle01`：i64/u64 → `_mm256_shuffle_epi32(v.raw,0x4E)`；f64 → `_mm256_shuffle_pd(v.raw,v.raw, 0x5)`（每 128 内交换：立即数 0b0101）。
- **⚠️ 用 avx2.h 现有 `detail::bitcast_raw_avx2` 惯例取/塞 raw**（见 TableLookupBytes:1193 风格）。

### avx512（512 位，per-128-block）
- i32/u32 → `_mm512_shuffle_epi32(v.raw, (_MM_PERM_ENUM)0xB1)`；f32 → `_mm512_shuffle_ps(v.raw,v.raw,0xB1)`。
- `Shuffle01`：i64/u64 → `_mm512_shuffle_epi32(v.raw,(_MM_PERM_ENUM)0x4E)`；f64 → `_mm512_shuffle_pd(v.raw,v.raw,0x55)`（8 个 f64，每 128 交换 → 0b01010101）。
- 参照 avx512.h:1050 现有 `(_MM_PERM_ENUM)` cast 写法。

### neon（128 位）
- `Shuffle2301` → `vrev64q_u32/s32/f32`（按 T 分）。
- `Shuffle1032` / `Shuffle01`(64) → `CombineShiftRightBytes<8>(v, v)`（**依赖批内 B 组先实现**，或直接 `vextq_u8(vb,vb,8)` 内联）。
- `Shuffle0321`（循环右移1 lane=右移4字节循环）→ `vextq_u8(vb, vb, 4)` 后 BitCast 回。
- `Shuffle2103`（循环左移1 lane）→ `vextq_u8(vb, vb, 12)`。
- `Shuffle0123`（全反 4×i32）→ `vrev64q_u32` 后再 `vextq_u8(.,.,8)` 交换两半；或 store+reload 反索引兜底（正确优先，NEON 无单指令全反 4 lane）。**grok 若拿不准全反的单指令，用 store+reload 反索引，标记等 opus 复核**。

### scalar oracle
按 128-bit block 建模（`kBlockLanes = 16/sizeof(T)`，虽然 scalar N 通常≤16 一个 block，但为对齐 256/512 语义用 block 公式）：
```
Shuffle2301: r[base+0]=v[base+1], r[base+1]=v[base+0], r[base+2]=v[base+3], r[base+3]=v[base+2]
```
每 block 起点 `base` 遍历。其余 op 按 §2 表的 lane 映射写。

## 3. 字节/lane 移位组施工卡（B）

### 语义（per-128-block，对齐 Highway）
- `ShiftLeftBytes<k>(v)`：每 128-block 内向**高地址**移 k 字节，低位补 0。`0<=k<=16`。
- `ShiftRightBytes<k>(v)`：向低地址移 k 字节，高位补 0。
- `ShiftLeftLanes<k>` = `ShiftLeftBytes<k*sizeof(T)>`；`ShiftRightLanes<k>` 同理。
- `CombineShiftRightBytes<k>(hi,lo)`：每 128-block，把 `[hi_block:lo_block]` 256 位拼接右移 k 字节取低 128。`0<k<16`。
- `CombineShiftRightLanes<k>` = `CombineShiftRightBytes<k*sizeof(T)>`。

### sse4
- `ShiftLeftBytes<k>`：`BitCast` 到 u8 → `_mm_slli_si128(rawu8, k)` → BitCast 回。
- `ShiftRightBytes<k>`：`_mm_srli_si128(rawu8, k)`。
- `CombineShiftRightBytes<k>(hi,lo)`：`_mm_alignr_epi8(hi_u8, lo_u8, k)`（SSSE3+，sse4 有）。
- Lanes 版：转调 Bytes 版，`k*sizeof(T)`。

### avx2（per-128-block）
- `ShiftLeftBytes<k>`：`_mm256_slli_si256(rawi, k)`（**天然每 128 半独立**，正确 block 语义）。
- `ShiftRightBytes<k>`：`_mm256_srli_si256(rawi, k)`。
- `CombineShiftRightBytes<k>`：`_mm256_alignr_epi8(hi_i, lo_i, k)`（每 128 半独立对齐）。

### avx512（per-128-block）
- `ShiftLeftBytes<k>`：`_mm512_bslli_epi128(v.raw, k)`。
- `ShiftRightBytes<k>`：`_mm512_bsrli_epi128(v.raw, k)`。
- `CombineShiftRightBytes<k>`：`_mm512_alignr_epi8(hi.raw, lo.raw, k)`。

### neon（128 位单 block）
- `ShiftLeftBytes<k>`：`vextq_u8(vdupq_n_u8(0), rawu8, 16-k)`（零向量拼接左移）——**注意 k=0 特判返回 v**（vext 立即数须 1..15）。
- `ShiftRightBytes<k>`：`vextq_u8(rawu8, vdupq_n_u8(0), k)`，k=0 特判。
- `CombineShiftRightBytes<k>(hi,lo)`：`vextq_u8(lo_u8, hi_u8, k)`（Highway arm_neon:5751 手法：lo 在前 hi 在后取 k 偏移）。

### scalar oracle
```
ShiftLeftBytes<k>: 按字节 per-16-byte block, out_byte[base+i] = (i>=k) ? in_byte[base+i-k] : 0
ShiftRightBytes<k>: out_byte[base+i] = (i+k<16) ? in_byte[base+i+k] : 0
CombineShiftRightBytes<k>(hi,lo): 每16字节block, 取 [lo_block(16) : hi_block(16)] 32字节窗口从偏移 k 起的低16字节
  out_byte[base+i] = (k+i<16) ? lo_byte[base+k+i] : hi_byte[base+k+i-16]
```
用 `reinterpret_cast<uint8_t*>(v.raw)` 逐字节，block 大小固定 16。

## 4. 验收

- **scalar oracle 先写**（它是所有后端对拍参照，按 §2/§3 的 block 公式）。
- `vec_test.cc` 每 op 加用例：构造已知输入 → 每后端结果对拍 scalar → 逐 lane `EXPECT_EQ`。**shuffle 用 sizeof==4/8 的类型；字节移位覆盖 i8/i32/f32 至少三种；Combine 用两个不同向量**。
- **多 block 必须覆盖**：avx2(256=2 block)/avx512(512=4 block) 要验每 block 独立正确——用 lane 值 = 全局 index，检查 block 内 permute 且 block 间不串。
- 双平台（Win MSVC + WSL g++）+ **AVX512 Intel SDE** + **mac NEON 真机**。
- objdump 抽验关键 op 真发射（`vrev64`/`vext`/`alignr`/`bslli`），非 store+reload 降级（NEON 全反 Shuffle0123 除外，若用 reload 需 opus 确认）。

## 5. 明确边界

- **不动 scalable**：SVE/RVV 的这些 op 归批3+（scalable 语义不同，slide 已有部分覆盖）。
- **不做 partial**：HYLE 定宽无 partial 向量，`ShiftLeftBytes` 只全宽版。
- `ShrBytes`（Highway 别名）= `ShiftRightBytes`，登记为 naming-diff 不单列。

## 6. clay 覆盖

clay 无可移植 vector API，本批是定宽后端对齐 Highway swizzle/移位广度的净新增，不改 clay 结论。
