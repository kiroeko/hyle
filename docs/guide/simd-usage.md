# SIMD 使用指南

> **当前状态（2026-07-17）**：本文先固定 SIMD 的用户语义、选择方法和调用合同。示例所用
> `hyle/simd/*.h` 是开发期 C++ pointer/count core，便于当前仓库测试，**不是发布 ABI**。
> close-01 落地后，外部用户的唯一稳定入口将改为纯 C 的 `include/hyle/c/simd.h`；届时本文会把
> 示例替换成正式 C 名称和 opaque handle，但下面的能力分类、数据布局和前置条件继续成立。

HYLE SIMD 不要求普通用户认识 300 多个寄存器原语。那些原语主要用于实现 kernel、移植 ISA 和做
Highway parity。应用通常只应调用处理整段连续数据的语义接口，让预编译库在运行时选择当前 CPU
可用的 scalar、SSE4、AVX2、AVX-512 或 NEON 实现。

## 先选择使用层

| 需求 | 应选层 | 是否运行时选 ISA | 稳定性 |
|---|---|---:|---|
| 发布一个二进制给未知 CPU | whole-buffer C ABI（close-01） | 是，每个模块缓存选择结果 | 未来唯一稳定发布面 |
| 当前参与 HYLE 开发、同一工具链调用 | `hyle/simd/{ops,math,...}.h` | 是 | 开发期 C++ core，不承诺 ABI |
| 整个 HYLE 只在一种已知 ISA 上运行 | `HYLE_SIMD_EXACT_TARGET` | 否，直接绑定目标实现 | 构建配置 |
| 自己写一个固定 ISA 的 kernel TU | `hyle/simd/native.h` / `native/*.h` | 否 | expert source surface |
| 实现或移植寄存器级 kernel | `hyle/simd/vec.h` | 否 | 内部 DSL，类型随 ISA 变化 |

```text
                         普通应用 / 任意语言
                                  |
                    close-01: hyle/c/simd.h
                      pointer + count + POD
                                  |
              +-------------------+-------------------+
              |                                       |
       未知客户 CPU                            已知目标的 HYLE 构建
   首次取得并缓存最佳实现                    HYLE_SIMD_EXACT_TARGET
              |                                       |
 scalar / SSE4 / AVX2 / AVX-512 / NEON          直接绑定一个实现
              \                                       /
               +---------- 共享 native kernel --------+
                                  |
                        vec.h 寄存器级 DSL
```

运行时选择不是每个元素做 CPUID，也不是在循环里反复判断 ISA。平台能力只探测一次；whole-buffer 调用
经缓存的函数入口进入一个 kernel。调用很小的数据时，函数边界本身仍可能比计算贵，因此应尽量批量调用。

## 常用能力速查

| 头（开发期） | 常用能力 | 适合做什么 |
|---|---|---|
| `ops.h` | `add`、同精度/混合精度 `dot`、`saxpy` | 连续数组的基础 BLAS-like 操作与量化存储 dot |
| `random.h` | `RandomState`、`random_fill` | 可复现、跨 ISA 位一致的批量随机数；非加密 |
| `scan.h` | `inclusive_scan`、`exclusive_scan` | 原地前缀和，支持 u/i32、u/i64、f32、f64 |
| `dtype_ops.h` | f16/bf16/fp8 与 f32 批量转换 | 模型权重、激活、KV cache 和混合精度数据搬运 |
| `math.h` | 实数/复数数学、激活函数、softmax | 数值计算、统计和 AI 热路径 |
| `sort.h` | sort、partial sort、select、radix、argsort | 排序、top-k、nth element、稳定整数/浮点 radix |
| `bitpack.h` | `pack`、`unpack`、`packed_size` | 同机内存中的定宽整数压缩 |

### 数学接口分组

- f32/f64 一元函数：`exp`、`log`、`sin`、`cos`、`tanh`、`exp2`、`log2`、`erf`、`atan`、
  `cbrt`、`expm1`、`log1p`、双曲函数、gamma 等，命名形如 `exp_f32_batch`。
- 二元函数：`atan2`、`pow`、`hypot`、`fmod`、`remainder`、`nextafter`、`fdim`。
- AI 语义：`gelu_f32_batch`、`gelu_tanh_f32_batch`、`sigmoid_f32_batch`、
  `silu_f32_batch`、原地 `softmax_f32` 和 `log_softmax_f32`。
- 复数 SoA：f32/f64 均提供 `cexp`、`clog`、`csqrt`、`csin`、`ccos`、`ctan`、`csinh`、
  `ccosh`、`ctanh`、`cpow`；实部和虚部放在两个数组中，避免把某种语言的 complex ABI
  固化到边界上。命名形如 `ctan_f32_batch` / `ctan_f64_batch`，输入输出可完全同址。
- `*_in_range_*` 是调用者承诺输入范围的快速合同，不是自动检查后回退的安全版本。

## 当前开发期调用示例

下面用 `std::vector` 只是展示 C++ 调用者如何提供内存。vector 不进入 HYLE 的函数签名或 ABI；换成
C 数组、C# `Span<T>` pin 后的地址、Rust slice 的 `as_ptr()/len()`，传入的仍是同一组
pointer/count 参数。

```cpp
#include "hyle/simd/ops.h"

std::vector<float> x(4096), y(4096), out(4096);
hyle::simd::add(x.data(), y.data(), out.data(), x.size());
const float inner = hyle::simd::dot(x.data(), y.data(), x.size());
hyle::simd::saxpy(0.25f, x.data(), y.data(), x.size());
```

模型权重或激活以窄类型存储时，不要先扩成一个临时 f32 数组；同一个入口会在寄存器中 widening：

```cpp
#include "hyle/simd/ops.h"

std::vector<hyle::simd::bfloat16_t> weights(n);
std::vector<hyle::simd::float16_t> left(n), right(n);
std::vector<float> activations(n);

const float score = hyle::simd::dot(activations.data(), weights.data(), n);
const float half_dot = hyle::simd::dot(left.data(), right.data(), n);
```

`bf16×bf16`、`f32×bf16` 和 `f16×f16` 都累加到 f32；`f32×bf16` 不会把 f32 输入降精度。
`i16×i16` 的乘积精确，以模 `2^64` 的二补码合同返回 i64，因此溢出行为不依赖 C++ UB。

```cpp
#include "hyle/simd/random.h"

hyle::simd::RandomState state{12345};
std::vector<float> unit(4096);
hyle::simd::random_fill(state, unit.data(), unit.size());  // [0, 1)
```

相同 seed 和相同的 fill 调用序列在不同 ISA 上位一致。它使用 xoshiro256++，不能生成 token、密钥、
nonce 或其他安全材料。单个普通随机数优先使用 `hyle/util/random.h` 的标量 `Rng`；批量填充才使用这里。

```cpp
#include "hyle/simd/scan.h"

std::vector<std::uint32_t> offsets = {3, 1, 4, 1, 5};
hyle::simd::exclusive_scan(offsets.data(), offsets.size());
// offsets == {0, 3, 4, 8, 9}
```

```cpp
#include "hyle/simd/dtype_ops.h"

std::vector<hyle::simd::float16_t> packed(n);
std::vector<float> values(n);
hyle::simd::f32_to_f16_batch(values.data(), packed.data(), n);
hyle::simd::f16_to_f32_batch(packed.data(), values.data(), n);
```

```cpp
#include "hyle/simd/math.h"

hyle::simd::exp_f32_batch(src, dst, n);
hyle::simd::gelu_f32_batch(src, dst, n);
hyle::simd::softmax_f32(dst, n);  // 原地
```

```cpp
#include "hyle/simd/sort.h"

hyle::simd::sort(keys, n);
hyle::simd::partial_sort(keys, n, 10);  // [0, 10) 是排好序的最小 10 个
hyle::simd::select(keys, n, k);         // keys[k] 到达完全排序后的位置
```

普通 key 支持 i/u16、i/u32、i/u64、f32/f64。没有可移植 `uint128_t` ABI 时使用
`UInt128 {lo, hi}`，按 `(hi, lo)` 的 unsigned 数值序排序。需要“分数排序时索引跟着走”时使用
`KeyValue32/64`；需要稳定排序时显式用 `radix_sort` 或 `argsort`，不要假设普通 `sort`
对相等 key 稳定。

```cpp
#include "hyle/simd/bitpack.h"

const unsigned bits = 11;
std::vector<std::uint16_t> packed(
    hyle::simd::packed_size<std::uint16_t>(raw.size(), bits));
hyle::simd::pack(raw.data(), raw.size(), packed.data(), bits);
hyle::simd::unpack(packed.data(), raw.size(), decoded.data(), bits);
```

bitpack 使用固定 128-bit logical block，跨 ISA/VL 保持同一块布局；格式仍应通过正式格式版本
管理，而不是把未版本化的内存缓冲直接当长期文件或网络 wire format。

## 必须遵守的调用合同

- HYLE 的 pointer/count core 不知道各容器长度。输入输出长度匹配、`n > 0` 时指针非空，由调用者保证。
- `n == 0` 在上述 batch/scan/sort 接口通常是 no-op；具体仍以对应头注释为准。
- 只有头文件明确写出的 exact in-place alias 才受支持；带偏移的重叠区间通常不受支持。
- `partial_sort` 要求 `k <= n`；`select` 在 `n > 0` 时要求 `k < n`。
- bitpack 的 `bits` 必须在元素位宽内，输入输出不得重叠。
- 浮点 scan 的结合顺序随 ISA 宽度变化；不要要求不同 ISA 得到逐位相同的浮点前缀和。
- 浮点 dot 也使用多累加器/vector reduction；结果遵守数值容差而非跨 ISA 逐位相同。
- 窄合同 `InRange` API 不做逐 lane 防御；违反范围前置条件的结果不受保证。
- 所有批量 RNG 都是非加密随机数。

## 已知目标：固定整库或固定 kernel

如果交付环境明确要求某个 ISA，可在配置 HYLE 时固定整个 SIMD 库：

```sh
cmake -S . -B build-avx2 -DHYLE_SIMD_EXACT_TARGET=AVX2
```

这不是“逃避动态税”，而是另一种产品合同：二进制本来就只允许运行在 AVX2 机器上，因此可以直接绑定
AVX2 symbol。未知客户 CPU 的发布包应使用默认 fat build。

只想固定自己的一两个 kernel TU 时，使用 CMake target `hyle_target_simd`，为该 TU 设置目标 ISA，
并调用 `hyle::simd::native`。`native` 返回的向量类型和 lane 数依赖目标；不能把它暴露到普通 TU、
动态库 ABI 或持久化格式中。直接使用 `vec.h` 也遵循同样边界，它是写 kernel 的 DSL，不是 387 个
普通用户 API 的集合。

## close-01 的迁移方式

正式 C ABI 的名字尚未冻结，概念映射预计如下：

| 当前开发期概念 | close-01 后的稳定形状 |
|---|---|
| `add(x, y, out, n)` 等无状态 batch | `hyle_simd_*` 纯 C 函数，pointer + count + POD enum |
| `RandomState` | `hyle_simd_random_o*` opaque handle + create/destroy/fill |
| `SortOrder` | 固定宽度 C enum/value |
| `float16_t`、`UInt128`、`KeyValue32/64` | 有尺寸、对齐和版本断言的 C POD |
| 模块批量入口 | 可选的 versioned `hyle_simd_api_t` 函数表 |

close-01 完成时，本文必须同步完成四件事：改用可被 C 编译器包含的正式头；增加纯 C 示例；写明
版本协商和 allocator/opaque 生命周期；删除“开发期 C++ core 可供外部集成”的过渡表述。C++、C#、
Java、Rust 的便利包装可由消费方或独立绑定项目提供，但不会成为 HYLE 核心的第二套发布契约。
