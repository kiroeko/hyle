# simd-18 施工卡（grok 实现 / opus 审核）

## 实现进度（grok 2026-07-11）

| 批 | 状态 | 实测 |
|---|---|---|
| C剩余 | ✅ OrderedDemote2To/InRange/shift-demote2；**DemoteToNearestInt/mask promote 已由 opus 裁决落地**(c45c6af) | scalar 297 / sse4 214 / avx2 215 |
| D | ✅ D1/D2/D3（D3 随 E InterleaveWhole）；**四查修 D2 双有符号溢出 UB**(3f23266) | 同上 |
| E | ✅ E1–E9/E11；**四查补 wide-type 覆盖**(4ab2ee2)。Dup128VecFromValues=@opus 未做(签名可变参,登记) | 同上 |
| F | ✅ F1 mpsadbw + F2 dbsad/portable；**四查修 F2 便携 oracle 错**(936cfe6) | 同上 |
| G | ✅ Highway 10 unary+atan2/pow/sincos f32/f64 + xsimd6；**四查换真 xsimd 核**(07d8aab,nextafter/remainder 留 scalar) | Math.Simd18BreadthG |
| @opus 裁决 | ✅ PromoteMaskTo/DemoteMaskTo/OrderedDemote2MasksTo(bitfield round-trip)+DemoteToNearestInt(c45c6af)+Dup128VecFromValues(d418ebf) | +4 测/后端;顺带修 LoadMaskBits 栈 OOB |
| H | ✅ op_parity 加新 op 对比(989c5bf);揪出 DemoteMaskTo 向量后端 6x + DemoteToNearestInt 15%(正确性差异),诚实记录权衡 | avx2 op_parity §17/§18 |

commits: grok `11a4984` C+D · `a30748f` E+D3 · `ccf45a6` F · `d359573` G · `c37cf9a` docs
四查(opus): `936cfe6` F2 · `4ab2ee2` E · `3f23266` D2 UB · `07d8aab` G xsimd · `c45c6af` @opus 裁决

> **分工**：opus 已用 3 个 agent 对 Highway 源逐 op 核实 native 手法（intrinsic + 契约 + fallback），本卡把语义/指令/落点全锁死。grok 只做"翻译到 intrinsic + 写测试 + 五平台真跑"，**不重新决策**。拿不准的标记 `@opus` 留审。
>
> **四查结论(opus 2026-07-11)**：不信任 grok 输出，逐批对拍/objdump/多平台/ASan，揪出 4 个弱测掩盖的真缺陷(F2 错 oracle、E 欠覆盖、D2 -O3 UB、G xsimd 假绿占位)全修。**@opus defer 项全部实证可做并落地**：mask 宽度转换走 bitfield round-trip(比 Highway 向量桥接更优,AVX-512 k-reg 本身即位域)、DemoteToNearestInt 全 f64 域决策避 cvttpd UB。触发既存 LoadMaskBits partial-tag 栈 OOB(TruncateTo 亲兄弟)一并修。
>
> **五平台真跑闸门**（每 op 收口，别省）：WSL `scalar/sse4/avx2`（`/root/hyle` build-wsl，`CXX=g++-14 CC=gcc-14`）+ AVX512 SDE（`sde64 -future -- exe`）+（碰 scalable 才需）qemu 多 VL。native 路径加 objdump 抽验真发射。ASan 查 store+reload scratch 栈宽（用 build-wsl-asan）。
>
> **通用纪律**：
> - 每批一个测试组，对拍 scalar oracle / 逻辑参考；round-trip 能对拍的必对拍。
> - **CRLF 陷阱**：若用 python 写文件，写完 `sed -i 's/\r$//'` 复原 LF（HYLE 全 LF，CRLF 会污染整文件 diff）。
> - 提交按批：`git add <改的文件>` → commit 信息只写真实改的文件 → 核对 files changed == 预期。别攒大 blob。
> - 后端落点：`include/hyle/simd/backend/{scalar,sse4,avx2,avx512,neon}.h` + 泛型 `vec.h`；测试 `tests/simd/vec_test.cc`。数学在 `src/simd/math/`。

---

## 批C 剩余（demote/InRange/shift-demote + mask promote/demote）

### C1. OrderedDemote2To(dn, a, b) — 整数有序 demote（对比 ReorderDemote2To 的块交织序）
**语义**：两个宽整数向量 a,b → 窄向量，**有序** `[a-lanes..., b-lanes...]`。HYLE 现有 `ReorderDemote2To` 在 x86 是块交织序 `[a.lo,b.lo,a.hi,b.hi]`，需一条 permute 修正。
**落点**：五后端各加 `OrderedDemote2To`。
- **scalar/neon/sse4**：`return ReorderDemote2To(dn, a, b);`（这三家 Reorder 已经是有序，直接别名）。
- **avx2**（`Vec256`，T_SIZE 1/2/4）：`_mm256_permute4x64_epi64(ReorderDemote2To(dn,a,b).raw, _MM_SHUFFLE(3,1,2,0))`（imm=0xD8，重排 64 位 qword [0,1,2,3]→[0,2,1,3]）。**注**：结果 raw 是 `__m256i`；float/double demote 不适用（demote2 仅整数）。
- **avx512**（`Vec512`，T_SIZE 1/2/4）：对 `ReorderDemote2To` 结果按 u64 做 `_mm512_permutexvar_epi64`，index 向量 `{0,2,4,6,1,3,5,7}`。
**测试**：`OrderedDemote2To` 结果逐 lane == `[demote(a[0..N]), demote(b[0..N])]`（用 scalar oracle 的有序 DemoteTo 做参考，即 `DemoteTo(dn,a,b)`——scalar DemoteTo 本就有序）。i32→i16、i16→i8 两组。

### C2. DemoteToNearestInt(di32, v) — f64→i32 就近舍入（区别于 DemoteTo 的截断）
**语义**：单个 f64 向量 → i32（N lane，round-to-nearest-even，非截断）。HYLE `DemoteTo` 是 2-arg 且截断，这是新单-arg op。
**落点**：泛型 `vec.h` 一处即可（用可移植式，最稳）：
```cpp
template <class D32, class V>  // D32 = i32 tag with same lane COUNT as v(f64)
HYLE_FORCE_INLINE VFromD<D32> DemoteToNearestInt(D32 dn, V v) noexcept {
  // Round(v) 先就近舍入到整值(仍 f64)，再 ConvertTo 到 i32。HYLE 有 Round(float->nearest float)?
  // ⚠️ Round 目前是 f32/f64 都支持吗? grep 确认;若只 f32,f64 用 std::nearbyint 落 scalar 式或原生。
}
```
**@opus 复核点**：HYLE 有没有 f64 的 `Round` + `ConvertTo(i32, f64)`（单向量、同 lane 数）？HYLE 的 ConvertTo 是同 lane 数 f→i；f64→i32 会 lane 数不匹配（i32 窄）。**若拼不出**，走 native：avx2 `_mm256_cvtpd_epi32`（返回 128-bit 4×i32，就近）、sse4 `_mm_cvtpd_epi32`、avx512 `_mm512_cvtpd_epi32`、neon `DemoteTo(di32, NearestInt(v))`（NearestInt=`vcvtnq_s64_f64` 后窄化）、scalar 逐 lane `std::nearbyint`+饱和。这条 grok 拿不准就**标记 @opus**，我来定落点（可能需要新的 f64→i32 单向量 tag 约定）。

### C3. InRange 快转换族（性能目标：跳过饱和 fixup，用 raw cvt）
**语义**：调用方保证输入在目标域内，跳过 `ConvertTo/DemoteTo/PromoteTo` 的溢出检测+blend。**只做 x86 有真区别的**，别名的登记不做。
**真有区别的（做 avx2/avx512/sse4 native + scalar/neon 别名到普通版）**：
| op | raw intrinsic（128/256/512） | 普通版多做的（被跳过） |
|---|---|---|
| `ConvertInRangeTo` f32→i32 | `_mm_cvttps_epi32`/`_mm256_cvttps_epi32`/`_mm512_cvttps_epi32` | `Ge(v,2^31)` + `IfThenElse(overflow,INT_MAX,raw)` |
| `DemoteInRangeTo` f64→i32 | `_mm_cvttpd_epi32`/`_mm256_cvttpd_epi32`/`_mm512...` | ClampF64ToI32Max |
**别名（InRange==普通，登记：PromoteInRangeTo 等——加薄封装转调普通版即可，注释说明无区别）**：`PromoteInRangeTo`/`PromoteInRangeLower/Upper/Even/OddTo`（widen 永不溢出）、f32→u32 等复杂域 grok 先只做 i32 主线，u32/i64/u64 标 @opus。
**落点**：avx2/avx512/sse4 加 native `ConvertInRangeTo`/`DemoteInRangeTo`；scalar/neon `return ConvertTo(...)` / `DemoteTo(...)`。泛型 `vec.h` 加 `PromoteInRangeTo* = PromoteTo*` 薄封装。
**测试**：域内输入下 InRange 结果 == 普通 ConvertTo/DemoteTo（域内两者必等）；不测域外（InRange 域外是 UB 契约）。

### C4. shift-and-demote 族（先补前置 RoundingShiftRight<k> 模板）
**前置**：HYLE 只有 `RoundingShiftRightSame`(运行期)，无 `RoundingShiftRight<k>` 模板。**先在五后端加 `template<int k> RoundingShiftRight(v)`**：`(v + (k?1<<(k-1):0)) >> k`，语义同 Same 但 k 编译期（可直接 `return RoundingShiftRightSame(v, k);` 转调，或 native 移位）。
**然后（全泛型组合，vec.h 一处）**：
- `ShiftRightAndDemoteTo(dn, v)` = `DemoteTo(dn, ShiftRight<k>(v))` ——⚠️ DemoteTo 2-arg，单 v 时第二个传啥？看 Highway：单向量 demote。**HYLE DemoteTo 是 2-arg**，所以这族要么 `DemoteTo(dn, ShiftRight<k>(a), ShiftRight<k>(b))`（2-arg 形叫 `...Demote2To`），要么标 @opus。**只做 2-arg 形**：
  - `ReorderShiftRightAndDemote2To<k>(dn,a,b)` = `ReorderDemote2To(dn, ShiftRight<k>(a), ShiftRight<k>(b))`
  - `ReorderRoundingShiftRightAndDemote2To<k>` = 同上换 `RoundingShiftRight<k>`
  - `OrderedShiftRightAndDemote2To<k>` = `OrderedDemote2To(dn, ShiftRight<k>(a), ShiftRight<k>(b))`（依赖 C1）
  - `OrderedRoundingShiftRightAndDemote2To<k>` = 同上换 Rounding
- 单向量的 `ShiftRightAndDemoteTo`/`RoundingShiftRightAndDemoteTo` 标 @opus（HYLE 无单向量 DemoteTo）。
**测试**：对拍 `ReorderDemote2To(ShiftRight<k>(a), ...)` 手算；k=1,3 两组。

### C5. PromoteMaskTo / DemoteMaskTo / OrderedDemote2MasksTo（批A 推迟的 mask）
**依赖**：C1 的 `OrderedDemote2To` + HYLE 的 promote/demote。
**问题**：HYLE 无单向量 `PromoteTo`（只有 half-at-a-time `PromoteLowerTo/UpperTo`）。所以 Highway 式 `PromoteMaskTo=MaskFromVec(PromoteTo(...))` 拼不出。
**@opus 决策点**：这三个 mask 宽窄转换在 HYLE half-at-a-time 模型下手法特殊，**grok 先跳过，标记留 opus**。我来定（可能用 PromoteLowerTo + 半 mask，或直接 per-backend）。

---

## 批D：WidenMul accumulate（3 op，泛型组合）
**落点**：全泛型 `vec.h`（组合现有 op），一处。

### D1. WidenMulAccumulate(d, mul, x, low, high&)
**语义**（Highway generic:5772）：narrow `mul`,`x` 加宽后 FMA 进两个宽累加器。
```cpp
template <class D, class VN>  // D=宽tag, VN=窄向量(2N lane)
HYLE_FORCE_INLINE VFromD<D> WidenMulAccumulate(D d, VN mul, VN x, VFromD<D> low, VFromD<D>& high) noexcept {
  high = MulAdd(PromoteUpperTo(d, mul), PromoteUpperTo(d, x), high);
  return MulAdd(PromoteLowerTo(d, mul), PromoteLowerTo(d, x), low);
}
```
HYLE 有 `PromoteLowerTo/UpperTo`(2-arg tag+vec) + `MulAdd`。整数 lane（i16→i32 等）。
**测试**：逐 lane `low[i]+=mul_lo[i]*x_lo[i]`、`high[i]+=mul_hi[i]*x_hi[i]` 手算对拍。

### D2. SatWidenMulPairwiseAccumulate(di32, a, b, sum)
**语义**（Highway generic:5854）：i16 pairwise widen-mul 饱和累加进 i32 sum。
```cpp
template <class DI32, class V16>
HYLE_FORCE_INLINE VFromD<DI32> SatWidenMulPairwiseAccumulate(DI32 di32, V16 a, V16 b, VFromD<DI32> sum) noexcept {
  const auto product = WidenMulPairwiseAdd(di32, a, b);          // HYLE 已有
  const auto overflow = VecFromMask(di32, Eq(product, Set(di32, INT32_MIN)));
  return SaturatedAdd(Sub(sum, And(BroadcastSignBit(sum), overflow)), Add(product, overflow));
}
```
HYLE 有 `WidenMulPairwiseAdd`/`BroadcastSignBit`/`SaturatedAdd`/`Eq`/`Set`/`Sub`/`And`/`Add`/`VecFromMask`。
**测试**：i16 a,b + i32 sum，对拍 `sat_i32(sum[i] + a[2i]*b[2i] + a[2i+1]*b[2i+1])`（注意 INT32_MIN 溢出修正 lane）。

### D3. SatWidenMulAccumFixedPoint(di32, a, b, sum)
**依赖**：D2 + `InterleaveWholeLower`（批E）。**grok 做完 D2+批E 后再回来做 D3**，或标 @opus。Highway generic:5885：Q15 每 lane i16*i16 饱和 MAC，通过 InterleaveWholeLower 复制 lane 后调 D2。
**@opus**：若 InterleaveWholeLower 就绪则做；否则标记。

---

## 批E：128-bit block 操作（12 op）
**落点**：`avx2.h`（256=2块）+ `avx512.h`（512=4块）；neon/sse4/scalar 多为 identity（单块）。HYLE 已有 `ExtractBlock<k>`/`BroadcastBlock<k>`/`InterleaveLower/Upper`/`ConcatLowerLower` 等，抄它们的 raw 手法。

**avx2 Concat intrinsic 速查**（agent 已核，InsertBlock/InterleaveBlocks 要用）：
- ConcatLowerLower: int `_mm256_inserti128_si256(lo, _mm256_castsi256_si128(hi), 1)`；f32 `_mm256_insertf128_ps`；f64 `_mm256_insertf128_pd`
- ConcatUpperLower: int `_mm256_blend_epi32(hi,lo,0x0F)`；f32 `_mm256_blend_ps(hi,lo,0x0F)`；f64 `_mm256_blend_pd(hi,lo,3)`
- ConcatUpperUpper: int `_mm256_permute2x128_si256(lo,hi,0x31)`；f `_mm256_permute2f128_ps/pd(lo,hi,0x31)`
- HYLE 已有这些 Concat（`avx2.h` grep `ConcatLowerLower` 等），直接调 HYLE 的，别重写。

### E1. SwapAdjacentBlocks(v)
- avx2：int `_mm256_permute4x64_epi64(v, _MM_SHUFFLE(1,0,3,2))`(=0x4E)；double `_mm256_permute4x64_pd(v,0x4E)`；float 先 bitcast 到 double 走 pd 再 bitcast 回。
- avx512：int `_mm512_shuffle_i32x4(v,v,_MM_PERM_CDAB)`(0xB1)；f32 `_mm512_shuffle_f32x4`；f64 `_mm512_shuffle_f64x2`。
- neon/sse4/scalar：`return v;`（单块）。

### E2. ReverseBlocks(d, v)
- avx2：`return SwapAdjacentBlocks(v);`（2 块反转==交换）。
- avx512：int `_mm512_shuffle_i32x4(v,v,_MM_PERM_ABCD)`(0x1B)；f32 `_shuffle_f32x4`；f64 `_shuffle_f64x2`。
- neon/sse4/scalar：`return v;`。

### E3. OddEvenBlocks(odd, even)（偶块取 even，奇块取 odd）
- avx2：int `_mm256_blend_epi32(odd,even,0x0F)`；f32 `_mm256_blend_ps(odd,even,0x0F)`；f64 `_mm256_blend_pd(odd,even,0x3)`。
- avx512：int `_mm512_mask_blend_epi64(0x33,odd,even)`；f32 `_mm512_mask_blend_ps(0x0F0F,odd,even)`；f64 `_mm512_mask_blend_pd(0x33,odd,even)`。
- neon/sse4/scalar：`return even;`。

### E4. InsertBlock<kBlock>(v, blk)（ExtractBlock 逆，blk 是 Vec128）
- avx2：`vec_to_insert = ResizeBitCast(d, blk)`（Vec128→Vec256 低块）；`kBlock==0 ? ConcatUpperLower(d, v, vec_to_insert) : ConcatLowerLower(d, vec_to_insert, v)`。用 HYLE 现有 Concat。
- avx512：`kBlock==0`：`IfThenElse(FirstN(d,16/sizeof T), ResizeBitCast(d,blk), v)`；`kBlock!=0`：int `_mm512_inserti32x4(v, blk.raw, kBlock)`；f32 `_mm512_insertf32x4`；f64 `_mm512_insertf64x2`。
- neon/sse4/scalar：`return blk;`（单块，kBlock=0）。

### E5. SlideUpBlocks<kB>(v) / SlideDownBlocks<kB>(v)
- avx2（kB∈{0,1}）：Up：`kB==1 ? ConcatLowerLower(d, v, Zero(d)) : v`；Down：`kB==1 ? ZeroExtendVector(d, UpperHalf(dh,v)) : v`。
- avx512（kB∈{0..3}）：用 `_mm512_alignr_epi64`（每块2个i64 lane）：
  - Up kB=1：`_mm512_alignr_epi64(v, _mm512_setzero_si512(), 6)`；kB=2：`ConcatLowerLower(d,v,Zero)`；kB=3：`_mm512_alignr_epi64(v,0,2)`
  - Down kB=1：`_mm512_alignr_epi64(0,v,2)`；kB=2：`ZeroExtendVector(d,UpperHalf(dh,v))`；kB=3：`_mm512_alignr_epi64(0,v,6)`
  - float/double：bitcast 到 i64 域做再 bitcast 回。
- neon/sse4/scalar：kB==0 只，`return v;`。

### E6. InterleaveEvenBlocks(d,a,b) / InterleaveOddBlocks(d,a,b)
- avx2：Even `ConcatLowerLower(d,b,a)`；Odd `ConcatUpperUpper(d,b,a)`。
- avx512：Even `OddEvenBlocks(SlideUpBlocks<1>(b), a)`；Odd `OddEvenBlocks(b, SlideDownBlocks<1>(a))`（组合 E3+E5）。
- neon/sse4/scalar：`return a;`。

### E7. InterleaveLowerBlocks / InterleaveUpperBlocks
- avx2：`=InterleaveEvenBlocks` / `=InterleaveOddBlocks`（别名，同 E6）。
- avx512：用 `_mm512_permutex2var_*`（TwoTablesLookupLanes，见批E尾 TwoTables）+ index 表：
  - Lower u64 idx `{0,1,8,9,2,3,10,11}`；f32 idx `{0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23}`；f64 idx `{0,1,8,9,2,3,10,11}`
  - Upper u64 idx `{4,5,12,13,6,7,14,15}`；f32 idx `{8,9,10,11,24,25,26,27,12,13,14,15,28,29,30,31}`；f64 idx `{4,5,12,13,6,7,14,15}`
- neon/sse4/scalar：`return a;`。

### E8. InterleaveWholeLower(d,a,b) / InterleaveWholeUpper(d,a,b)（整向量连续交织，≠ per-block InterleaveLower）
- **sse4/neon/scalar（单块）**：`= InterleaveLower/InterleaveUpper(a,b)`（HYLE 已有）。
- **avx2（无 AVX512 时）**：`Lower = ConcatLowerLower(d, InterleaveUpper(a,b), InterleaveLower(a,b))`；`Upper = ConcatUpperUpper(d, InterleaveUpper(a,b), InterleaveLower(a,b))`。**HYLE avx2 编译不含 AVX512，用这条**。
- avx512：`_mm512_permutex2var_*` + 表（agent 已列全，u8/u16/u32/f32/u64/f64 各一组 Lower/Upper idx，见下）：
  - u32 Lower `{0,16,1,17,...,7,23}` Upper `{8,24,...,15,31}`；u64 Lower `{0,8,1,9,2,10,3,11}` Upper `{4,12,5,13,6,14,7,15}`；u16/u8/f32/f64 同规律（f32=u32表,f64=u64表）。
**测试**：InterleaveWholeLower(a,b) 逐 lane == 交替 `a[0],b[0],a[1],b[1],...`（前半）；用 scalar oracle。

### E9. Per4LaneBlockShuffle<i3,i2,i1,i0>(v)（每128块内4-lane permute，32位主线）
- imm `= _MM_SHUFFLE(i3,i2,i1,i0)`。
- 32位：sse4 `_mm_shuffle_epi32(v,imm)`/`_mm_shuffle_ps`；avx2 `_mm256_shuffle_epi32(v,imm)`/`_mm256_shuffle_ps(v,v,imm)`；avx512 `_mm512_shuffle_epi32(v,(_MM_PERM_ENUM)imm)`/`_mm512_shuffle_ps`。
- 64位：avx2 `_mm256_permute4x64_epi64/pd(v,imm)`；avx512 `_mm512_permutex_epi64/pd(v,imm)`；sse4 `_mm_shuffle_epi32`(适配) 标 @opus。
- 8/16位：无单指令，标 @opus（Highway 走 generic 分解，暂只做 32/64 位）。
- neon/scalar：32位用 store+reload oracle 或 vrev/vext 组合（grok 用 store+reload 最稳，对拍即可）。
**测试**：32-bit lane，i3i2i1i0 几个模式（如 2301/1032/0123）对拍手算 block 内重排。

### E10. Dup128VecFromValues(d, v0..vN)（批B 推迟）— 立即数建 128 块广播
**@opus**：签名是可变参数（每 128 块 lane 数个标量），跨 lane 数复杂。grok **标记留 opus**，我定签名后再做。

### E11. TwoTablesLookupLanes(d, a, b, idx)（批B 推迟）— 双源 permute
- avx512：int `_mm512_permutex2var_epi{8,16,32,64}(a, idx, b)`（按 lane 宽）；f32 `_ps`；f64 `_pd`。
- avx2：无单指令双源，用 `_mm256_permutex2var_*`（AVX512VL 才有）——HYLE avx2 变体无 AVX512，**avx2 走 fallback**：两次 `TableLookupLanes` + `idx<N ? a-lookup : b-lookup` blend。
- sse4/neon：同 fallback（两次单源 TableLookupLanes + blend on idx>=N）。
- scalar：store+reload 双数组按 idx 取。
**测试**：idx 覆盖 [0,2N) 跨两源，对拍 `idx<N ? a[idx] : b[idx-N]`。

---

## 批F：SAD 视频 quad（2 op）
**落点**：sse4/avx2/avx512 native + neon/scalar/其余 portable fallback。u8 输入 u16 输出。

### F1. SumsOfAdjQuadAbsDiff<kAOffset(0..1), kBOffset(0..3)>(a, b)
- **sse4**（SSE4.1 gate，基线非门控）：`_mm_mpsadbw_epu8(a.raw, b.raw, (kAOffset<<2)|kBOffset)`。
- **avx2**：`_mm256_mpsadbw_epu8(a.raw, b.raw, (kAOffset<<5)|(kBOffset<<3)|(kAOffset<<2)|kBOffset)`。
- **avx512**：无 mpsadbw，built on dbsad：`SumsOfShuffledQuadAbsDiff<kAOffset+2,kAOffset+1,kAOffset+1,kAOffset>(a, BitCast(d, Broadcast<kBOffset>(BitCast(du32,b))))`（依赖 F2）。
- **neon/scalar/portable**（generic:8086）：
  ```
  a_lo=InterleaveLower(a,a); a_hi=InterleaveUpper(a,a)
  a01=CombineShiftRightBytes<kAOffset*8+1>(a_hi,a_lo)  // 再 resize 回 V8
  a23=CombineShiftRightBytes<kAOffset*8+5>(a_hi,a_lo)
  b01=BitCast(d8,Broadcast<kBOffset*2>(BitCast(d16,b)))
  b23=BitCast(d8,Broadcast<kBOffset*2+1>(BitCast(d16,b)))
  return BitCast(d16, Add(SumsOf2(AbsDiff(a01,b01)), SumsOf2(AbsDiff(a23,b23))))
  ```
  HYLE 有 InterleaveLower/Upper/CombineShiftRightBytes/Broadcast/SumsOf2/AbsDiff。i8 重载：对 a,b XOR 0x80 转 u8 域，调 u8 版，结果 bitcast i16。
**测试**：对拍 scalar oracle：对每 8-lane 组，`sum_{j=0..3} |a[kAOffset*4+i+j] - b[kBOffset*4+j]|`（i=0..7 输出 8 个 u16）。查 Highway mpsadbw 语义定义精确对齐。

### F2. SumsOfShuffledQuadAbsDiff<i3,i2,i1,i0>(a, b)
- **avx512**（AVX512BW gate，SDE 可验；`#if defined(__AVX512BW__)`——但 HYLE avx512 基线含 BW，直接用）：128 `_mm_dbsad_epu8(b.raw,a.raw,_MM_SHUFFLE(i3,i2,i1,i0))`；256 `_mm256_dbsad_epu8(b,a,imm)`；512 `_mm512_dbsad_epu8(b,a,imm)`。**注意 operand 序是 (b,a,imm)，别写反**。
- **sse4/avx2（无 dbsad）+ neon/scalar**：portable（generic:8190）：
  ```
  a_shuf=Per4LaneBlockShuffle<i3,i2,i1,i0>(BitCast(d32,a))   // 依赖 E9
  a_0123_2345=BitCast(d8, OddEven(ShiftLeftBytes<2>(a_shuf), a_shuf))
  a_1234_3456=BitCast(d8, OddEven(ShiftLeftBytes<1>(a_shuf), ShiftRightBytes<1>(a_shuf)))
  even=SumsOf4(AbsDiff(a_0123_2345, b)); odd=SumsOf4(AbsDiff(a_1234_3456, b))
  odd=ShiftLeft<16>(odd)  // little-endian
  return OddEven(BitCast(d16,odd), BitCast(d16,even))
  ```
  HYLE 有 SumsOf4/OddEven/ShiftLeftBytes/ShiftRightBytes/AbsDiff。依赖 E9 Per4LaneBlockShuffle。
**测试**：对拍 scalar oracle（quad 按 i3..i0 重排后 SAD）。avx512 SDE 跑 native + objdump 见 `vdbsad`；sse4/avx2 跑 fallback。两路对拍同 oracle。

---

## 批G：数学面（16 fn，dispatch 布线）
**落点**：`src/simd/math/`。**wiring 模板见 §末**。

### G1. Highway 借（10 fn，单一来源，非 best-of）
Highway `hn::Xxx(d, ...)` 全有 f32+f64。unary 用 X-macro，binary 手写。
| fn | 调用 | arity | 落点模板 |
|---|---|---|---|
| acos | `hn::Acos(d,x)` | unary | 抄 atan（Highway X-macro） |
| asin | `hn::Asin(d,x)` | unary | 同上 |
| sinh | `hn::Sinh(d,x)` | unary | 同上 |
| cosh | `hn::Cosh(d,x)` | unary | 同上 |
| tan | `hn::Tan(d,x)` | unary | 同上 |
| atanh | `hn::Atanh(d,x)` | unary | 同上 |
| log10 | `hn::Log10(d,x)` | unary | 同上 |
| atan2 | `hn::Atan2(d,y,x)` | **binary** | 抄 hypot（手写核+longhand dispatch），a=y b=x |
| pow | `hn::Pow(d,a,b)` | **binary** | 抄 hypot，Pow(d,base,exp) |
| sincos | `hn::SinCos(d,x,s&,c&)` | **1→2出** | **手写**：`sincos_f32_batch(src, sin_dst, cos_dst, n)`，无现成模板 |

### G2. xsimd 借（6 fn，Highway 无，单一来源）
xsimd `xsimd::xxx(batch)` f32+f64。unary 抄 exp2_xsimd 链，binary 抄 hypot_fast_xsimd 链。
| fn | 调用 | arity |
|---|---|---|
| erfc | `xsimd::erfc(x)` | unary |
| exp10 | `xsimd::exp10(x)` | unary |
| fmod | `xsimd::fmod(x,y)` | **binary** |
| remainder | `xsimd::remainder(x,y)` | **binary** |
| nextafter | `xsimd::nextafter(x,y)` | **binary** |
| fdim | `xsimd::fdim(x,y)` | **binary** |

**@opus 注**：atan2 两库都有（Highway+xsimd），charter 要 best-of bench。grok 先用 Highway 布好线，性能对比归批H。

### wiring 模板（agent 已核，5 处）
**unary Highway（抄 acos）**：
- `kernels.h`：4 处 decl（scalar 块 / avx2 / avx512 / neon 块），`void acos_f32_{scalar,highway_avx2,highway_avx512,highway_neon}(const float*,float*,size_t)`。
- `transcendental_highway.cc` X-macro 列 + `HYLE_HWY_MATH_AVX2(acos, Acos, std::acos)`；`_avx512.cc` `HYLE_HWY_MATH_512(acos,Acos,std::acos)`；`_neon.cc` `HYLE_HWY_MATH_NEON(acos,Acos,std::acos)`。
- `transcendental_scalar.cc`：手写 3 行 loop `std::acos`。
- `math_dispatch.cc`：`HYLE_MATH_DISPATCH_HIGHWAY(acos)` 加进列表。
- `math.h`：`HYLE_SIMD_API void acos_f32_batch(const float*,float*,size_t) noexcept;`
**binary Highway（抄 hypot）**：4 TU 手写核（`hn::Atan2(d,LoadU(a),LoadU(b))`）+ `math_dispatch.cc` longhand（512 GCC→avx2→scalar / neon / scalar）+ math.h。签名 `(const float* a,const float* b,float* dst,size_t)`。
**xsimd（抄 exp2/hypot_fast）**：定义在 `transcendental_xsimd{,_avx512,_neon}.cc`，dispatch longhand（无 xsimd X-macro）。
**f64**：Highway 10 个全有 f64；照 f32 加 `_f64_batch`（math_dispatch 有 f64 Highway macro）。xsimd 6 个也 f32+f64。**grok 先做 f32 全 16 个 + f64 Highway 10 个；xsimd f64 6 个若时间够一并做，否则标 @opus**。
**测试**：`math_test` 对拍 `std::acos` 等 libm，容差同现有超越函数（相对误差）。binary 覆盖两输入。sincos 查 sin+cos 双输出。

---

## 收尾
- 每批 commit 后更新本卡对应节为 ✅ + 实测数（"scalar/sse4/avx2 + SDE 全绿 197→2XX"）。
- `@opus` 标记的项**别硬做**，留着我审时定夺（DemoteToNearestInt 落点 / InRange u32-u64 域 / 单向量 shift-demote / mask promote-demote / Dup128VecFromValues 签名 / Per4Lane 8-16位 / D3 / xsimd f64）。
- 全做完把总 op 数、五平台测试数写回 `simd-18-breadth-completion.md`，我做最终四查（对拍 oracle + objdump native + 多平台 + 无捏造）。
