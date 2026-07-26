# SIMD 使用指南

> `include/hyle/c/simd.h` 是 whole-buffer 的跨语言/跨编译器稳定入口；完整 C 符号、POD 和
> FFI 说明见 [`c-simd.md`](./c-simd.md)。`hyle/simd/*.h` 是原生 C++ source surface，
> 其中 whole-buffer 算法供 C++ 用户使用，寄存器 DSL/native 头属于 expert tier；均不承诺
> 跨编译器 ABI。

HYLE SIMD 不要求普通用户认识 300 多个寄存器原语。那些原语主要用于实现 kernel、移植 ISA 和做
Highway parity。应用通常只应调用处理整段连续数据的语义接口，让预编译库在运行时选择当前 CPU
可用的 scalar、SSE4、AVX2、AVX-512 或 NEON 实现。

## 先选择使用层

| 需求 | 应选层 | 是否运行时选 ISA | 稳定性 |
|---|---|---:|---|
| 发布一个二进制给未知 CPU 或跨语言绑定 | `hyle/c/simd.h` whole-buffer C ABI | 是，每个模块缓存选择结果 | binary-stable C/FFI 面 |
| C++ 用户调用 whole-buffer 算法 | `hyle/simd/{ops,math,...}.h` | 是 | 原生 source API，不承诺跨工具链 ABI |
| 整个 HYLE 只在一种已知 ISA 上运行 | `HYLE_SIMD_EXACT_TARGET` | 否，直接绑定目标实现 | 构建配置 |
| 自己写一个固定 ISA 的 kernel TU | `hyle/simd/native.h` / `native/*.h` | 否 | expert source surface |
| 自己写一个 kernel，仍要发布给未知 CPU | `hyle_add_fat_kernel` + `HYLE_SIMD_SELECT_FAT_KERNEL` | 是，每个完整 kernel 选一次 | source integration |
| 实现或移植寄存器级 kernel | `hyle/simd/vec.h` | 否 | 内部 DSL，类型随 ISA 变化 |

```text
                         普通应用 / 任意语言
                                  |
                         hyle/c/simd.h
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
| `search.h` | count/find/min/max/minmax/is_sorted | 连续数组查找、归约和有序性检查 |
| `dtype_ops.h` | f16/bf16/fp8 与 f32 批量转换 | 模型权重、激活、KV cache 和混合精度数据搬运 |
| `math.h` | 实数/复数数学、激活函数、softmax | 数值计算、统计和 AI 热路径 |
| `sort.h` | sort、partial sort、select、radix、argsort | 排序、top-k、nth element、稳定整数/浮点 radix |
| `bitpack.h` | `pack`、`unpack`、`packed_size` | 同机内存中的定宽整数压缩 |
| `native/algorithm.h` | generate/transform/replace/find/count/copy-if | 自定义固定目标 kernel；expert source surface |

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

## 稳定 C ABI 调用示例

```c
#include "hyle/c/simd.h"

float x[4096], y[4096], out[4096];
hyle_simd_add_f32(x, y, out, 4096);
float inner = hyle_simd_dot_f32(x, y, 4096);
hyle_simd_saxpy_f32(0.25f, x, y, 4096);
```

模型权重或激活以窄类型存储时，不要先扩成一个临时 f32 数组；同一个入口会在寄存器中 widening：

```c
hyle_simd_bf16 weights[4096];
hyle_simd_f16 left[4096], right[4096];
float activations[4096];

float score = hyle_simd_dot_f32_bf16(activations, weights, 4096);
float half_dot = hyle_simd_dot_f16(left, right, 4096);
```

`bf16×bf16`、`f32×bf16` 和 `f16×f16` 都累加到 f32；`f32×bf16` 不会把 f32 输入降精度。
`i8×i8` 与 `i16×i16` 的乘积精确，以模 `2^64` 的二补码合同返回 i64，因此溢出行为
不依赖 C++ UB。i8 路径会在可用时自动采用 AVX-VNNI、AVX512-VNNI、NEON dotprod
或 SVE dot；默认 AVX2/AVX512 基线本身不要求这些可选扩展。

```c
hyle_simd_random_o* state = NULL;
hyle_status status = hyle_simd_random_create(&state, 12345);
if (!hyle_status_is_ok(status)) {
  fprintf(stderr, "%s\n", hyle_status_message_cstr(status));
  hyle_status_free(&status);
  return;
}
hyle_status_free(&status);
float unit[4096];
hyle_simd_random_fill_f32(state, unit, 4096);  // [0, 1)
hyle_simd_random_destroy(state);
```

相同 seed 和相同的 fill 调用序列在不同 ISA 上位一致。它使用 xoshiro256++，不能生成 token、密钥、
nonce 或其他安全材料。单个普通随机数优先使用 `hyle/util/random.h` 的标量 `Rng`；批量填充才使用这里。

```c
uint32_t offsets[] = {3, 1, 4, 1, 5};
hyle_simd_exclusive_scan_u32(offsets, 5);
// offsets == {0, 3, 4, 8, 9}
```

```c
hyle_simd_f16 packed[4096];
float values[4096];
hyle_simd_f32_to_f16(values, packed, 4096);
hyle_simd_f16_to_f32(packed, values, 4096);
```

如果是自己编译到固定 target TU 的 C++ kernel，并且数据要保持 f16/bf16 存储、在寄存器中以
f32 计算，使用 `narrow_float.h` 的显式格式 tag：

```cpp
#include "hyle/simd/narrow_float.h"

namespace hs = hyle::simd;
using DF = hs::ScalableTag<float>;
const DF df;
const hs::F16Tag<DF> dh;

const std::size_t lanes = hs::Lanes(dh);
std::size_t i = 0;
for (; i + lanes <= n; i += lanes) {
  auto a = hs::LoadNarrowU(dh, left + i);
  auto b = hs::LoadNarrowU(dh, right + i);
  auto y = hs::MulNarrow(dh, a, b);  // promote f16 -> f32, multiply, round once to f16
  hs::StoreNarrowU(y, dh, output + i);
}
if (i != n) {
  auto a = hs::LoadNarrowN(dh, left + i, n - i);
  auto b = hs::LoadNarrowN(dh, right + i, n - i);
  hs::StoreNarrowN(hs::MulNarrow(dh, a, b), dh, output + i, n - i);
}
```

`BF16Tag<DF>` 同形。还可用 `PromoteNarrowTo(df, dh, v)`、`DemoteNarrowTo(dh, v)`、
`AddNarrow/SubNarrow/DivNarrow/MulAddNarrow`。`Eq/Lt/Le/Gt/GeNarrow` 是 IEEE 数值比较：
NaN 无序，`-0 == +0`。`ReduceSumNarrow` 返回 f32；`ReduceMin/MaxNarrow` 遇任意
NaN 传播 canonical qNaN，`ReduceMin/MaxNumberNarrow` 忽略 NaN（全 NaN 时返回
canonical qNaN），并分别保留 `-0/+0`。

要处理 target-bound 整段数组，包含 `narrow_algorithm.h`：

```cpp
#include "hyle/simd/narrow_algorithm.h"

float sum = hs::SumNarrow(dh, values, n);
std::size_t first = hs::FindEqualNarrow(dh, values, n, needle); // IEEE ==
std::size_t exact = hs::FindBitsNarrow(dh, values, n, needle);  // exact 16-bit encoding
bool ordered = hs::IsSortedTotalOrderNarrow(dh, values, n);
```

`CountEqualNarrow/FindEqualNarrow` 把 signed zero 视为相等、NaN 永不匹配；
`CountBitsNarrow/FindBitsNarrow` 则区分 signed zero 和每个 NaN payload。
`narrow_total_order_key` / `TotalOrderKeyNarrow` 是可逆的 IEEE-754 `totalOrder`
key：negative qNaN < negative sNaN < -inf < ... < -0 < +0 < ... < +inf <
positive sNaN < positive qNaN。这是 bitwise deterministic order，不是“NaN 都相等”。

选择原则：

- 只搬运/转换整个数组：优先 `dtype_ops.h` 的 batch API，它处理 dispatch 和 tail；
- 写 target-bound 自定义计算循环：使用 narrow tag，x86 AVX2/AVX-512 f16 自动走 F16C，其他
  baseline 使用同语义 vector 位运算，不要求 SVE/RVV 的可选 half 扩展；
- carrier 的真实寄存器类型是后端 u16。不要对 `auto` 结果调用无 tag 的 `Add/Mul`，那是整数位
  carrier 语义；窄浮点操作必须使用带 `Narrow` 的名字；
- finite f32→f16 逐位一致；硬件 F16C 可以选择不同的 NaN payload，因此 NaN 只保证仍为 NaN。

这层是 C++ target-TU 接口，不把目标相关 vector ABI 暴露给 C。C/FFI 整段 narrow
search/reduction 还没有冻结 pointer/count 入口；当前只有 dtype convert/mixed dot 的 C ABI。
narrow sort 也尚未实施，不要拿 carrier 的 unsigned 顺序替代 total-order 语义。

```c
hyle_simd_exp_f32_batch(src, dst, n);
hyle_simd_gelu_f32_batch(src, dst, n);
hyle_simd_softmax_f32(dst, n);  // 原地
```

```c
hyle_simd_sort_f32(keys, n, HYLE_SIMD_SORT_ASCENDING);
hyle_simd_partial_sort_f32(keys, n, 10, HYLE_SIMD_SORT_ASCENDING);
hyle_simd_select_f32(keys, n, k, HYLE_SIMD_SORT_ASCENDING);
```

普通 key 支持 i/u16、i/u32、i/u64、f32/f64。没有可移植 `uint128_t` ABI 时使用
`UInt128 {lo, hi}`，按 `(hi, lo)` 的 unsigned 数值序排序。需要“分数排序时索引跟着走”时使用
`KeyValue32/64`；需要稳定排序时显式用 `radix_sort` 或 `argsort`，不要假设普通 `sort`
对相等 key 稳定。

```c
const unsigned bits = 11;
size_t packed_count = hyle_simd_packed_size_u16(raw_count, bits);
uint16_t* packed = allocate_u16(packed_count);
hyle_simd_pack_u16(raw, raw_count, packed, bits);
hyle_simd_unpack_u16(packed, raw_count, decoded, bits);
```

`packed_size` 与 tier 无关，但 packed bytes 只保证由同一 tier 解码；x86、NEON 与 scalar
布局可能不同。SVE/RVV 把 logical block 固定为 128 bit，因此其字节不随 VL/VLEN 改变并与
NEON 布局一致。当前内存编码不是跨 tier 的长期文件或网络 wire format。

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

多数自定义 kernel 不必自己重复写整向量循环和最后一个 partial vector。`native/algorithm.h` 接管
循环、tail 和 mask，lambda 只写寄存器表达式：

```cpp
#include "hyle/simd/native.h"

void bias_relu(float* values, std::size_t n, float bias) {
  hyle::simd::native::transform(values, n, [bias](auto d, auto v) {
    return hyle::simd::Max(hyle::simd::Add(v, hyle::simd::Set(d, bias)),
                           hyle::simd::Zero(d));
  });
}
```

这一层还提供 `generate`、二/三输入 `transform`、`replace[_if]`、`find_if`、`count_if` 和
stable `copy_if`。输入输出可 exact 同址（`copy_if` 也支持原地稳定压实），带偏移 overlap 不受支持。
不足一个 vector 时，transform/predicate 的 inactive input lane 为零，`generate` 的 index lane
继续递增；两者都只写/统计 active lane，lambda 不应把 inactive lane 当作可观察副作用。
`generate` 的 index lane 与输出 lane 等宽，窄整数自然按其位宽回绕。

要把自己的 pointer/count kernel 随 HYLE 一起多版本编译，则让 impl header 根据
`HYLE_SIMD_VARIANT_SUFFIX` 导出 suffixed entry，再用：

```cmake
hyle_add_fat_kernel(
  TARGET my_app NAME my_transform IMPL "src/my_transform-inl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
```

baseline 调用点用 `HYLE_SIMD_SELECT_FAT_KERNEL(...)` 选择并缓存入口。不要直接手写 CPUID，也不要让
vector/mask 出现在 entry 签名中。这个选择宏在默认 fat 构建中走一次平台探测，在
`HYLE_SIMD_EXACT_TARGET` 构建中只引用唯一生成的 exact symbol；在 macOS ARM64 上也不会引用未生成的
SVE symbol。fat `TARGETS` 必须列出本平台完整 canonical tier，CMake 会在配置期检查。

`native::transform` 不提供通用 C callback 版本：逐 lane function pointer 阻断内联和向量化，恰好破坏
这个 API 的目的。跨语言用户应调用已有具名 batch 产品；需要自定义语义时，在 fat-kernel impl 中
使用上面的 C++ helper，再导出自己的 pointer/count C entry。

## C++ source surface

同工具链源码用户仍可直接调用 `hyle::simd::dot/sort/...`；它们与 C shim 进入同一 dispatcher，
但不是跨编译器稳定 ABI。HYLE 没有 `hyle_simd_api_t`、`api_get` 或版本函数表：核心发布面直接导出
自由 C 符号。C++、C#、Java、Rust 的容器包装由消费方或独立 binding 项目提供。
