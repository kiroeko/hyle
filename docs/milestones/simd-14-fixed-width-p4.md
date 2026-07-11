# simd-14：P4 定宽 Mul / Mod / Magnitude

| 项 | 内容 |
|---|---|
| 状态 | ✅ 完成（2026-07-11） |
| 前置 | simd-13 |
| 后续 | `simd-gap-inventory.md` P5 |
| 公开 op 计数 | **77/92**；P4 新增 12 个，P5 剩 15 个 scalable 软件项 |
| 范围 | `vec.h` 泛型 API、定宽原生热路径、SVE/RVV 直接实例化与 Highway 同 ISA 对照 |

## 1. 公开语义

- FMA 家族：`MulSub(a,b,c)=a*b-c`、`NegMulSub(a,b,c)=-a*b-c`；`MulAddSub` 的偶 lane 为减、奇 lane 为加，`MulSubAdd` 相反。浮点复用 `MulAdd` 保留单次舍入；整数经无符号位表示完成模算术，`INT_MIN` 不触发未定义行为。
- `MinMagnitude` / `MaxMagnitude`：无符号等于 `Min` / `Max`；有符号按无符号模绝对值比较，`INT_MIN` 的 magnitude 正确为 `2^(N-1)`；浮点使用 Highway 同一 generic 组合。相等 magnitude 交由目标 ISA 的 `Min` / `Max` tie 语义决定：AArch64 NEON 使用 `FMINNM/FMAXNM`，异号零分别规范为 `-0/+0`，与 Highway NEON 一致。
- `MulEven` / `MulOdd`：从交替的 i/u8、i/u16、i/u32 source lane 各选择一半、提升至对应宽类型后相乘。定宽 i/u32→i/u64 有原生后端路径；其余类型复用 `PromoteEvenTo` / `PromoteOddTo`。
- `MulEvenAdd` / `MulOddAdd`：接受 packed `uint16_t` IEEE bf16 bit carrier，选择偶/奇 bf16、提升为 f32 后与 f32 accumulator 做 fused multiply-add；HYLE 不把 bf16 暴露为 native vector lane。
- `Mod` / `MaskedModOr`：仅覆盖既有 `Divisor` 的 invariant i32/u32 域。`Divisor(0)` fail-fast；signed `INT_MIN / -1` 沿用模 quotient，余数为零。mask false lane 返回 `no`，参数顺序保持 `(mask, no, dividend, divisor)`。

所有批量路径均为 Θ(n)：每个输出至少读取输入并写回一次。`Divisor` reciprocal 参数构造位于循环外，`Mod` 在循环内只执行既有 reciprocal-multiply、shift 与乘减余数路径。

## 2. 定宽热路径与性能裁决

`MulEven` / `MulOdd` 的 i32→i64 路径直接用各 ISA 的 widening multiply。`Div` 的 i32 high-product 在 SSE4、AVX2、AVX512 与 NEON 使用原生序列；NEON 采用 Highway 同形的 `vmull[_high] + vuzp2`，避免输入解交织和显式窄化的额外重排。

同 TU、同 ISA、双方 header-only、相同 noinline 边界，以 `scripts/bench-stat.py -n15` 判定。`Mod` 对照 Highway `contrib/intdiv` 的同一预计算 invariant divisor 与 `a - q*d` remainder；结果向量逐轮传给 `doNotOptimizeAway`，不以输出指针伪造 dead-code elimination 差异。首轮后，Windows AVX2 与 Apple NEON 各再独立执行三组 `N=15`：Windows 四项每组均为 TIE；NEON 每组均复现 `Mod` 1.073x、`MaskedModOr` 1.058x WIN，两个 widening multiply 每组为 TIE。

| 平台 | Mod i32 | MaskedModOr i32 | MulEven i32→i64 | MulOdd i32→i64 |
|---|---:|---:|---:|---:|
| Windows AVX2 | 0.169 vs 0.168 ns/op，TIE | 0.194 vs 0.193，TIE | 0.082 vs 0.081，TIE | 0.096 vs 0.096，TIE |
| Apple NEON | 0.204 vs 0.219，HYLE 1.073x WIN | 0.265 vs 0.280，HYLE 1.058x WIN | TIE | TIE |

FMA 符号变体与 magnitude 是既有 primitive 的直通组合，没有独立算法热路径，不为制造数字单列 benchmark。四个存在独立瓶颈价值的 P4 热路径均未落后 Highway。

## 3. 验收

- Windows MSVC：完整 SIMD 267 tests，266 passed + 1硬件 skip；SSE4、AVX2、SSE4-AES 各 184/184。
- WSL GCC14：完整 SIMD 267 tests，266 passed + 1硬件 skip；SSE4、AVX2、SSE4-AES 各 184/184；AVX512 由 Intel SDE 执行 184/184。
- WSL ASan：267 tests，266 passed + 1硬件 skip，零 sanitizer error。
- Apple Silicon：完整 SIMD 264 tests，263 passed + 1非 x86 skip；NEON 184/184。
- SVE QEMU：16 tests 在 VL 128/256/384/512 各通过；RVV QEMU：13 tests 在 VLEN 128/256/512/1024 各通过。

`vec_test.cc` 覆盖有符号/无符号 FMA 符号、`INT_MIN` magnitude、异号零 bit pattern、六种整数 widen、packed-bf16 accumulate、i32/u32 边界余数、稀疏与 all-off mask。`sve_test.cc` 与 `rvv_test.cc` 的 `P4MulModMagnitude` 直接实例化共享泛型层，避免 scalable API 仅靠未实例化模板看似可用。

## 4. 实际代码量

以 simd-13 后 P4 起点的 Git tracked 加未忽略 untracked 双树为基线，对 P4 目标文件执行 `cloc --diff`：生产头新增 **640** code LOC、测试新增 **486**、benchmark 新增 **90**，共 **1,216** 新增 code LOC；另有 28 code LOC 修改、4 code LOC 删除。规模来自跨 ISA 的 native i32 high-product / widen 路径、可扩展泛型实现和多宽度测试，而非 stub。

## 5. clay 覆盖

clay 没有可移植 vector API、invariant vector remainder 或 bf16 even/odd accumulation 的对应能力。P4 是 HYLE `simd/` 的净新增能力，不改变 `math/half.h` 已由硬件 dtype 路径取代的 clay 覆盖结论；coverage map 已明确该边界。
