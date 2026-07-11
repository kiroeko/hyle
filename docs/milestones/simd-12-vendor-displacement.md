# simd-12：自研热点核竞争与小数组调用税收口

| 项 | 内容 |
|---|---|
| 状态 | ✅ 完成（2026-07-11） |
| 前置 | simd-11 |
| 后续 | `simd-gap-inventory.md` P4 → P5 |
| 净增长 | **+348 production / +216 tests / +712 bench = +1,276 code LOC** |
| 结论 | u16 11/13 bit-pack 上架正确 SIMD 核；SoA `csin/ccos` 上架 HYLE 编排；整数/KV small-K `partial_sort` 改走实测更快的 std 专用核；失败的自研 sort 候选未进入生产 |

## 1. 裁决规则

所有比较使用同一 fixed-ISA TU 与显式 `noinline` 边界，分开解释：

- `public/direct`：公开 API、dispatcher、DLL/跨 TU 的固定调用税；
- `candidate/vendor`：算法差；
- `scalar/std`：正确性 oracle 和标量性能地板。

统计统一用 `scripts/bench-stat.py -n15`；gap 必须大于双方 spread 之和才判 WIN/LOSE。AVX2 与 Apple NEON 用真机时序；AVX512 SDE只验正确性和 tier 路由。候选先过独立 oracle、极值、tail、未对齐、guard、alias 与 ASan，未稳定获胜就不进入生产。

三个 benchmark 支持环境变量筛选，避免每次运行完整矩阵：

- `hyle_bitpack_bench`：`HYLE_BENCH_BITS/N/OP`；
- `hyle_math_scaling_bench`：`HYLE_BENCH_OP/N/DIST`；
- `hyle_sort_bench`：`HYLE_BENCH_TYPE/N/K/DIST/ORDER`。

规模覆盖 0/1、lane/block±1、L1/L2 与 1M 元素；sort 同时覆盖随机、少值、升序和降序。

## 2. u16 11/13-bit pack/unpack

### 2.1 根因与实现

Highway 问题不只在 inverse：

1. 11/13 inverse 的 rawF 在移位后才掩码，相邻组位会污染结果；
2. 13 inverse 的 raw8/raw9、rawD/rawE 读取了错误 packed vector；
3. 13 pack 把 `0x7C00 << 3` 当成 `0x8000`，16-bit 截断后实际为 `0xE000`，rawF 会覆盖合法组位。

`src/simd/bitpack/bitpack_hyle_u16-inl.h` 保持现有 tier-dependent block layout，修正上述 pack/inverse。11-bit pack 的既有布局正确，13-bit pack 与两种 inverse 由 HYLE 持有。`bitpack_body.inc` 仅对11/13切到该核，其余位宽继续借 Highway。

公开输入/输出范围必须不重叠。所有 SIMD tier 的最小整块都是128个 raw value；`count < 128` 时宽 TU 原本只执行 scalar tail，现由 baseline dispatcher 直接执行相同 bitstream，消除无效跨 TU 调用。

### 2.2 正确性

- 11-bit 0..2047、13-bit 0..8191 全值域；
- 独立标量 block-layout oracle，pack逐 packed word 对拍，unpack读取 oracle 流；
- block±1、多block、tail、未对齐输入输出与双端 guard；
- Windows/WSL/Apple NEON、AVX512 SDE、WSL ASan 全绿。

独立 oracle 在每个结构 lane 上重建 8+3 / 8+5 布局，不依赖生产 pack/unpack 互相自证。

### 2.3 性能

大数组相对原正确 scalar fallback：

| 平台 | 11-bit pack | 11-bit unpack | 13-bit pack | 13-bit unpack |
|---|---:|---:|---:|---:|
| Windows AVX2 | **22.1x** | **32.1x** | **22.7x** | **26.8x** |
| WSL AVX2 | **23.3x** | — | — | **14.0x** |
| Apple NEON | **19.8x** | — | — | **16.0x** |

正确 candidate 与错误 Highway 上界统计打平；public/direct 统计打平。`n=1` 的 public/direct 原先明确慢29.9%，dispatcher直跑tail后归于统计平局。

AVX512没有真机性能证据，不宣称速度；SDE已证明修正核和真实AVX512 dispatch正确。

## 3. SoA complex `csin/ccos`

### 3.1 编排

xsimd `batch<complex>` 的 `sin/cos` 分别计算 real `sin`、real `cos`、`sinh`、`cosh`，重复 argument reduction 和 exponential。HYLE编排改为：

```text
(sin_x, cos_x) = sincos(x)
t = expm1(abs(y)); e = 1 + t
sinh_y = copysign(0.5 * t * (e + 1) / e, y)
cosh_y = 1 + t*t / (2*e)
```

`expm1` 形式避免小虚部的灾难性消减。普通 lane 完全向量化；NaN/Inf、有限 overflow 与 real signed-zero 只逐 lane 回退 `std::complex`，不连带标量化同 block 的正常 lane。异常结果在写输出前保存，因此 exact in-place alias 仍正确。

`cexp` 未重写：源码核实 xsimd 已是 `exp(real)+sincos(imag)` 同款最优编排。`clog/csqrt` 的 branch cut 和缩放风险高，本轮不立项；后续 `ctanh/cpow` 的独立竞争与最终路由见 [`simd-13`](./simd-13-vendor-displacement-2.md)。

### 3.2 正确性与性能

测试覆盖随机有限值、±0、subnormal、tiny imaginary、Inf/NaN、有限 overflow、tail 和 exact in-place alias。

最终公平结果：

- Windows AVX2 random大数组：`csin` **1.29x**、`ccos` **1.28x** 胜 xsimd；
- tiny-imag 与 pure-real：和 xsimd 统计打平，不再有消减或整块标量回退；
- WSL GCC AVX2：统计打平，无回退；
- Apple NEON：和 xsimd direct 统计打平，但相对原生产 scalar 约 **2.2x**；
- 每block一个特殊值时结果保持完整语义，相对全scalar仍约 **9.0x**。

public/direct 在各平台统计打平。

## 4. small-N / small-K sort

Highway VQSort 已含完整 small-N sorting network，重造 full sort 没有依据。真正缺口是 `partial_sort` 的 small-K：VQSort small-N base case仍完整排序，没有利用 K。

尝试过固定有序前缀候选；它常赢 Highway，但在类型、K和重复键之间不能稳定赢 `std::partial_sort`，因此生产候选完整撤销，只在 benchmark 保留负结论。

最终路由：

- 整数与 `KeyValue32/64` 的 `k <= 16`：调用现有 std scalar 专用核；
- `k > 16`：保持 VQPartialSort；
- 浮点始终保持 VQSort 路径，避免 small-K 分流改变 NaN 语义；
- scalar浮点比较器补成与VQSort一致的严格弱序：有限值按方向排列，NaN始终在末尾。

N=16..1M、K=1/2/4/8/16、随机/少值、升/降序测量中，std 相对 Highway 最多快 **3.4x**；最差的 KV64 N=1M/K=16 统计打平，没有确认回退。生产 public 与同边界 std direct 统计打平；Windows/WSL/Apple NEON 的 KV32 N=4096/K=8 分别约 **2.8x / 2.0x / 4.8x** 胜 Highway（平台噪声不同，分别裁决）。

测试新增 small-K 整数、KV prefix oracle、全数组 pair multiset、重复键、双方向和浮点 NaN 末尾契约。

## 5. 验收

- Windows MSVC：SIMD **250 tests**，249 passed + 1硬件skip；
- WSL GCC14：SIMD **250 tests**，249 passed + 1硬件skip；
- Apple Silicon：SIMD **247 tests**，246 passed + 1非x86 skip；
- WSL ASan：250 tests，249 passed + 1硬件skip，零 sanitizer 错误；
- AVX512 SDE：本轮 Math/Sort/BitPack 41/41；
- shared vec/scalable层未改，不触发SVE/RVV多VL重跑；
- `git diff --check`通过。

LOC由会话起点版本逆向回放109次Write/Edit后用cloc差分核算，未混入此前P0-P3：**+348 production / +216 tests / +712 bench = +1,276 code LOC**。
