# simd-13：第二轮 vendor displacement（SoA `ctanh` / `cpow`）

| 项 | 内容 |
|---|---|
| 状态 | ✅ 完成（2026-07-11） |
| 前置 | simd-12 |
| 后续 | `simd-gap-inventory.md` P4 → P5 |
| 公开 op 计数 | **不变：65/92**；本轮只替换已有 API 的内部路由 |
| 裁决 | HYLE SoA `ctanh/cpow` 上架 AVX2/NEON；`cexp/clog/csqrt`、VQSort/Select 与其余 bit-pack 保留供应商 |

## 1. 范围与裁决纪律

本轮只竞争生产中直接调用 Highway/xsimd 的 SIMD 核；不把 P4/P5 的公开 API 缺口混进来，也不为“去 vendor”保留输家。

所有比较使用同 ISA、同 TU、同 `noinline` 边界：

- `public/direct` 分开公开 dispatcher 固定税和算法差；
- `candidate/vendor` 才是 HYLE 编排与 xsimd 算法的对照；
- `scripts/bench-stat.py -n15` 仅在 gap 大于双方 spread 之和时裁决 WIN/LOSE；
- scalar `std::complex` 是完整语义 oracle，不把 vendor 当 oracle；
- AVX512 SDE 仅验证 codegen、dispatch 和正确性，不参与时序结论。

## 2. 上架：SoA `ctanh`

对 regular lane：

```text
tanh(x + iy)
  = sinh(2x) / (cosh(2x) + cos(2y))
  + i sin(2y) / (cosh(2x) + cos(2y))
```

HYLE只计算一次 `sincos(2y)`，再用 `expm1(|2x|)`稳定推导双曲对。`|x| > 20` 时，整块均为安全 large-real lane 可直接写出 `(copysign(1,x), signed-zero)`；混合 block 用 `q = exp(-2|x|)` 渐近式，不让双曲中间量溢出。

regular lane 不等于“所有有限输入”：real/imaginary axis、subnormal、NaN/Inf、近极点，和 `abs(y) > FLT_MAX/2`（`2*y` 会溢出）的有限 lane 都逐 lane 用 `std::tanh(std::complex<float>)` 覆盖。异常结果在 vector store 前保存，因而 exact in-place alias 不被破坏。近极点 vendor 精度与极大有限虚部审计分别见 [`xsimd-ctanh-near-pole-hunt.md`](../../../developer/debugging/xsimd-ctanh-near-pole-hunt.md) 和 [`xsimd-ctanh-extreme-imaginary-hunt.md`](../../../developer/debugging/xsimd-ctanh-extreme-imaginary-hunt.md)。

最新 N=15、N=4096 真机 direct 对照：

| 平台 | random | large-real | 裁决 |
|---|---:|---:|---|
| Windows AVX2 | HYLE 1.632 vs xsimd 2.011 ns/op，**1.222x WIN** | 0.627 vs 1.970，**3.105x WIN** | 上架 |
| WSL AVX2 | 1.703 vs 1.687，0.986x，TIE | 0.604 vs 1.655，**2.663x WIN** | 上架 |
| Apple NEON | 3.483 vs 3.423，0.985x，TIE | 1.186 vs 3.393，**2.861x WIN** | 上架 |

near-pole 走完整 scalar oracle，吞吐显著低于 xsimd；这不是可替换的 regular-path 数据，而且 xsimd 在该点已被对拍发现精度不足，不能以其较快结果放宽公开契约。

## 3. 上架：SoA `cpow`

对 regular lane：

```text
rho = hypot(z); L = log(rho); phi = atan2(z_im, z_re)
A = w_re * L - w_im * phi
B = w_re * phi + w_im * L
z^w = exp(A) * (cos(B) + i sin(B))
```

相比 xsimd generic complex pow，HYLE只计算一次 `log(rho)`、一次 `exp(A)`和一次 `sincos(B)`。它竞争的是 SoA 编排和重复计算消除，不重写成熟实数 `hypot/log/atan2/exp/sincos` 原语。

base 或 exponent 的轴、subnormal、非有限、过大 base/exponent、危险 `A` 窗口与 branch-cut lane 逐 lane 落 `std::pow`；普通 lane 不会被异常邻 lane 标量化。exact base-plane in-place alias、tail、未对齐和 guard 全部经过测试。

最新 N=15、N=4096 真机 direct 对照：

| 平台 | random | large-base | 裁决 |
|---|---:|---:|---|
| Windows AVX2 | HYLE 3.064 vs xsimd 4.092 ns/op，**1.336x WIN** | 4.860 vs 8.652，**1.765x WIN** | 上架 |
| WSL AVX2 | 3.120 vs 4.001，**1.322x WIN** | 5.258 vs 8.793，**1.682x WIN** | 上架 |
| Apple NEON | 5.637 vs 7.441，**1.320x WIN** | 6.516 vs 7.793，**1.193x WIN** | 上架 |

near-cut/zero-base 等标量 oracle 路径的吞吐不参与 regular candidate 的胜负裁决；它们存在是为保住 signed-zero 和 branch-cut 语义。

## 4. 明确保留 vendor 的路径

| 路径 | 决定 | 原因 |
|---|---|---|
| `cexp` | 保留 xsimd | 已是 `exp(re) + sincos(im)` 同构编排，没有可消除的结构性重复工作。 |
| `clog` / `csqrt` | 保留 xsimd | 需要完整 overflow-safe magnitude、相位和象限体系；没有已证的局部冗余，不投机重写。 |
| real transcendentals、safe `hypot`、`tgamma` | 保留逐 op winner | 成熟多项式/缩放核无可证的局部超越点；`lgamma`继续 scalar，因 xsimd lane-mixing bug。 |
| VQSort / VQPartialSort / VQSelect / KV wide | 保留 Highway | full/small-N 重造和有序前缀 candidate 都没有成立。新增 float NaN partial/select、KV select、pair multiset 与 partition 契约测试。 |
| 除 u16 11/13 外的 bit-pack | 保留 Highway | 位宽专用 layout 核没有结构性候选；11/13 已在 simd-12 由 HYLE正确核接管。 |

## 5. 剩余直接借用核的复杂度审计

在**相同公开契约**下，剩余路径没有能靠手写获得渐近时间复杂度优势的候选：

| 类别 | 下界 / 结论 |
|---|---|
| 实数 `exp/log/tanh/...`、safe `hypot`、`tgamma` | 每个输出至少读取一个输入并写一个输出，都是 Θ(n)；多项式/缩放的差异只能是常数，且没有可消除的跨 lane 重复工作。 |
| `cexp` | 已是 `exp(re)+sincos(im)` 的Θ(n)最简编排；再写只会复刻 xsimd。 |
| `clog/csqrt` | 同为 Θ(n)，要赢只能重造完整 magnitude/phase/缩放/象限体系，当前不存在少算一次重型原语的结构性证据。 |
| VQSort / VQPartialSort / VQSelect | generic comparison full sort 有 Ω(n log n) 下界；select/partial 的通用输入为 Θ(n)。small-K 已按实际 workload 分流 std，不能把受限 key/radix 算法偷换成泛型契约。 |
| 除 u16 11/13 外的 bit-pack | 每值至少读写一次，Θ(n) I/O 下界；其他位宽是 Highway 专门 layout 核，未发现重复计算或错误实现。 |

因此本轮后不再为“也许能更快”重开 generic vendor 核。重新立项只能由**新事实**触发：公开输入域收窄并新增显式 fast API、消费者能批量复用中间量、key domain 改为可用 radix/计数排序，或新 ISA 提供此前不存在的指令。下一项回到 P4；这不是放弃常数优化，而是避免把没有渐近或结构性机会的成熟核重写成维护负债。

## 6. 正确性与验收

`Math`新增极大有限虚部、imaginary axis/subnormal、`ctanh` near-pole、large-real、NaN/Inf、tail、未对齐、guard和精确 alias；`cpow`新增 base/exponent signed-zero、subnormal、branch cut、危险指数和精确 alias。`Sort`增加浮点 `partial_sort/select` 的 NaN末尾契约、KV select 的 pivot partition 和 pair multiset。

最新源码验收：

- Windows MSVC：SIMD 255 tests，254 passed + 1硬件 skip；SSE4/AVX2/SSE4-AES 变体各 173/173；
- WSL GCC14：SIMD 255 tests，254 passed + 1硬件 skip；SSE4/AVX2/SSE4-AES 变体各 173/173；
- WSL ASan：255 tests，254 passed + 1硬件 skip，零 sanitizer 错误；
- Apple Silicon：SIMD 252 tests，251 passed + 1非 x86 skip；NEON 173/173；
- AVX512 SDE：`vec` 173/173，Math/Sort/BitPack 47/47；
- 本轮 `ctanh/cpow` candidate 未修改 shared `vec.h` 或 SVE/RVV后端，不扩大为 scalable QEMU 验收；
- `git diff --check`通过。

当前工作树含尚未提交的 simd-12 变更，按 Git 已跟踪变更加未跟踪文件建立临时双树后执行 `cloc --diff`。排除文档与构建脚本，仅 C/C++ 计数的二轮合并工作树为：3,681新增 code LOC、167 code LOC修改、10 code LOC删除。不能把未提交的 simd-12 与 simd-13 可靠拆分，故不伪造单独 LOC。simd-12 已记录其独立闭环的 +1,276 code LOC。
