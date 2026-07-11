# simd-17 批4：半向量合并族（批0 之上，返回真实体）

| 项 | 内容 |
|---|---|
| 状态 | 🟢 施工卡就绪（批0 已铺地基；opus 出卡 / grok 实现 / opus 四查）|
| 前置 | **批0（分层多宽度类型）已完成**（commit 6c12369+73e58e3）——本批的使能器 |
| 范围 | `backend/{sse4,avx2,avx512}.h`（x86 层叠后）+ `backend/neon.h`（Vec64 半宽）+ `vec_test.cc`。不动 vec.h、不动 scalable |
| 目标 | `LowerHalf` `UpperHalf` `Combine` `ZeroExtendVector` `InsertIntoUpper` `ResizeBitCast`（~6 op）返回**真半宽实体** |

## 0. 批0 如何使能本批（读 simd-17-breadth-batch0-architecture.md）

批0 前：定宽后端单一 native 宽度，avx2 TU 无 `Vec128`，`LowerHalf(Vec256)` 无处返回。**曾被迫考虑"partial-tag 全宽存储"workaround（留永久债），已否决**。

批0 后（实测验证）：
- avx2.h 层叠 sse4.h → `Vec128` 与 `Vec256` 共存一 TU；avx512.h 层叠 avx2 → 三宽度共存。
- `ScalableTag<float>::Half`（16B）**已验证解析到 `Vec128<float,4>`**，`Zero(Half{})` 走 sse4 的 width-band `requires` 重载。full tag → `Vec256<float,8>`。
- **所以 `LowerHalf(Vec256)→Vec128` 现在返回真实体**（`_mm256_castps256_ps128`），不是打 tag 的全宽壳。这正是根本解法（研究文档方案①）。

## 1. 语义（对齐 Highway，scalar oracle 式）

设 full 向量 `v` 有 `N` lane，半宽有 `N/2` lane。`Half<D>` = `Simd<T, N/2, kPow2-1>`（vec.h:59，已有）。

| op | 签名 | 语义 |
|---|---|---|
| `LowerHalf(dh, v)` | `VFromD<Half<D>> LowerHalf(Half<D> dh, VFromD<D> v)` | 取 v 低 N/2 lane → 半宽向量 |
| `LowerHalf(v)` | `VFromD<Half<DFromV<V>>> LowerHalf(V v)` | 便利重载，dh 从 v 推 |
| `UpperHalf(dh, v)` | `VFromD<Half<D>> UpperHalf(Half<D> dh, VFromD<D> v)` | 取 v 高 N/2 lane → 半宽 |
| `Combine(d, hi, lo)` | `VFromD<D> Combine(D d, VFromD<Half<D>> hi, VFromD<Half<D>> lo)` | 两半宽拼全宽：[lo, hi] |
| `ZeroExtendVector(d, lo)` | `VFromD<D> ZeroExtendVector(D d, VFromD<Half<D>> lo)` | 低半=lo，高半=0 |
| `InsertIntoUpper(d, hi, v)` | `VFromD<D> InsertIntoUpper(D d, VFromD<Half<D>> hi, VFromD<D> v)` | v 的高半替换为 hi |
| `ResizeBitCast(d, v)` | `VFromD<D> ResizeBitCast(D d, V v)` | 位重解释到不同宽度 tag（可放大/缩小，多余位未定义/截断）|

**参数序（Highway 惯例，必对齐）**：`Combine(d, hi, lo)` = 结果 `[lo 在低半, hi 在高半]`。定宽 scalar.h 现有 Concat 是这个序，一致。

## 2. sse4 施工卡（128→64 半宽，需引入 Vec64）

sse4 的半宽是 **64 位**（`Vec128<T,N>` 的 half = `Vec64` 概念）。HYLE sse4 目前**无 Vec64 类型**——批4 sse4 部分需新增，或用 `Vec128<T,N/2>`（同 `__m128` 存储，低 64 位有效）表示半宽。

**决策（对齐批0 的 band 模型）**：sse4 半宽 = `Vec128<T, N/2>`（物理 `__m128`，逻辑 N/2 lane 在低 64 位）。`Half<sse4-tag>` 的 byte width 落 `(0,16]` 仍是 sse4 band——**sse4 内部半宽不跨后端**，只是更窄的 partial tag，批0 的 band 已天然支持（TruncateTo 已用 `Vec128<u8,4>`）。
- `LowerHalf(dh, Vec128<T,N> v)`：低 N/2 lane 已就位 → `return Vec128<T,N/2>{v.raw}`（同 raw，窄 tag）。
- `UpperHalf(dh, v)`：高 N/2 移到低 → `_mm_unpackhi_epi64` 或 `_mm_srli_si128(v.raw, 8)` 后 `Vec128<T,N/2>{...}`。
- `Combine(d, hi, lo)`：`_mm_unpacklo_epi64(lo.raw, hi.raw)`（lo 低 64 + hi 低 64 → 128）。
- `ZeroExtendVector(d, lo)`：`_mm_move_epi64(lo.raw)`（低 64 保留，高 64 清零）。
- `InsertIntoUpper(d, hi, v)`：`_mm_unpacklo_epi64(v.raw, hi.raw)`（v 低 64 + hi 低 64）。

## 3. avx2 施工卡（256→128 半宽，Vec128 已层叠共存）★核心

**批0 已让 Vec128 共存**，直接抄 Highway x86_256（已侦察）：
- `LowerHalf(dh, Vec256<T,N> v)`：
  - f32 → `Vec128<float,N/2>{_mm256_castps256_ps128(v.raw)}`
  - f64 → `_mm256_castpd256_pd128`；int → `_mm256_castsi256_si128`
  - 零成本 cast（只是"忘记"高 128）。
- `UpperHalf(dh, v)`：
  - f32 → `_mm256_extractf128_ps(v.raw, 1)`；f64 → `_mm256_extractf128_pd(.,1)`；int → `_mm256_extracti128_si256(.,1)`
- `Combine(d, hi, lo)`（抄 Highway:4893）：
  - `ZeroExtendVector(d, lo)` 得 lo256 → `_mm256_inserti128_si256(lo256.raw, hi.raw, 1)`（int）；f32/f64 用 `_mm256_insertf128_*`
- `ZeroExtendVector(d, lo)`（抄 Highway:4836）：
  - `_mm256_zextsi128_si256(lo.raw)`（int，GCC/Clang 有 zext intrinsic）；f32 `_mm256_zextps128_ps256`；f64 `_mm256_zextpd128_pd256`
  - **⚠️ MSVC 陷阱**（Highway 注释）：老 MSVC 的 `_mm256_inserti128` 不清零高位 → 用 `_mm256_zext*` 保证。grok 用 zext 系列即可，别用 insert-into-zero。
- `InsertIntoUpper(d, hi, v)`：`_mm256_inserti128_si256(v.raw, hi.raw, 1)`（int）/ `_mm256_insertf128_*`（float）。

## 4. avx512 施工卡（512→256 半宽，Vec256 已层叠共存）

批0 让 Vec256 共存，同 avx2 手法换 512 intrinsic：
- `LowerHalf` → `_mm512_castps512_ps256` / `castpd512_pd256` / `castsi512_si256`
- `UpperHalf` → `_mm512_extractf32x8_ps(v,1)` / `extractf64x4_pd(v,1)` / `extracti64x4_epi64(v,1)`
- `Combine` → `_mm512_insertf32x8`/`insertf64x4`/`inserti64x4`（lo 放 0，hi 放 1，先 zext lo）
- `ZeroExtendVector` → `_mm512_zextsi256_si512` / `zextps256_ps512` / `zextpd256_pd512`
- `InsertIntoUpper` → insert 到 index 1

## 5. neon 施工卡（128→64 半宽，Vec64）

NEON 独立不层叠（aarch64）。半宽 = 64 位（`float32x2_t` 等）。HYLE neon **可能已有 Vec64 或需新增**——grok 先 grep。
- `LowerHalf` → `vget_low_*`（`vget_low_f32(v)` 取低 64）
- `UpperHalf` → `vget_high_*`
- `Combine` → `vcombine_*`（`vcombine_f32(lo, hi)`）
- `ZeroExtendVector` → `vcombine_*(lo, vdup_n_*(0))`
- `InsertIntoUpper` → `vcombine_*(vget_low(v), hi)`

## 6. scalar oracle

```
LowerHalf(v):  r[i] = v[i]           for i in [0, N/2)
UpperHalf(v):  r[i] = v[N/2 + i]     for i in [0, N/2)
Combine(hi,lo): r[i] = lo[i] (i<N/2) ; r[i] = hi[i-N/2] (i>=N/2)
ZeroExtendVector(lo): r[i] = lo[i] (i<N/2) ; 0 (i>=N/2)
InsertIntoUpper(hi,v): r[i] = v[i] (i<N/2) ; hi[i-N/2] (i>=N/2)
ResizeBitCast: 按字节重解释（min(src,dst) 字节拷贝，dst 多余位 0/未定义）
```
scalar 后端半宽用 `Vec<T, N/2>`（scalar 无宽度约束，天然支持）。

## 7. 验收

- scalar oracle 先写。`vec_test.cc` 加 `HalfVectorFamily` 测试组：
  - 构造 full 向量 lane 值 = 全局 index（可辨）。
  - `LowerHalf`/`UpperHalf` 结果对拍低/高半；`Combine(UpperHalf, LowerHalf)` round-trip == 原向量；`ZeroExtendVector` 高半全 0；`InsertIntoUpper` 只改高半。
  - **round-trip 铁验**：`Combine(d, UpperHalf(dh,v), LowerHalf(dh,v)) == v`（拆了再拼回原样）。
- 四平台：Win MSVC + WSL g++（sse4/avx2）+ AVX512 SDE + mac NEON。
- **objdump 抽验**：`LowerHalf` 应是单条 `vextractf128`/`castsi256`（零或一指令），不是 store+reload。`Combine` 应是 `vinsertf128`。证明真实体路径，非 workaround。
- **批0 回归**：本批加半宽 op 后，vec_test 原 184 test 仍全绿（半宽是新增，不改现有）。

## 8. 关键陷阱（grok 必读）

1. **Half<D> tag 已由批0 正确解析**（实测 `ScalableTag<float>::Half`→`Vec128<float,4>`）。grok 直接用 `VFromD<Half<D>>` 作半宽返回/参数类型,不要自己造半宽 tag。
2. **ZeroExtendVector 必用 `_mm256_zext*` 系列**（不是 insert-into-zero）——MSVC 老版 insert 不清零高位（Highway 坑）。
3. **参数序**：`Combine(d, hi, lo)` = `[lo, hi]`（lo 低半）。测试 round-trip 时 `Combine(d, UpperHalf, LowerHalf)` 才等于原值。
4. **sse4/neon 半宽是 64 位**：若后端无 Vec64,用 `Vec128<T,N/2>`(sse4,低 64 有效)/ `float32x2_t`(neon)。grok 先 grep 现有半宽表示,拿不准标记问 opus。
5. **LowerHalfOfMask/UpperHalfOfMask**（mask 的半）跟随向量半——本批可一并做,或延批4b。grok 视工作量定,mask 半宽在 avx512 是 k-reg 移位(`_kshiftri`),定宽是 mask 向量的半,scalar 是 bool 数组半。

## 9. clay 覆盖

半向量族是窄化管线能力，clay 无对应；net 新增。批0 使能后返回真实体，接口对齐 Highway。不改 clay 结论。
