# SIMD 纯 C ABI

`include/hyle/c/simd.h` 是 SIMD 模块给外部程序、其他编译器和 FFI 的稳定入口。它只有
`extern "C"` 自由函数、pointer/count、固定布局 POD 和一个 RNG opaque handle；不暴露
C++ `Runtime`、STL、模板、D-tag 或 ISA 相关向量类型。

## 最小例子

```c
#include "hyle/c/simd.h"

float x[1024], y[1024], out[1024];
hyle_simd_add_f32(x, y, out, 1024);
float score = hyle_simd_dot_f32(x, y, 1024);
hyle_simd_exp_f32_batch(out, out, 1024);
```

默认 fat 库在进程内探测一次平台能力并缓存不可变 Runtime；每次调用只在 whole-buffer
边界进入已选 kernel，不在元素循环里执行 CPUID。调用者不应自行选 SSE/AVX/NEON。
要诊断实际选择可用：

```c
hyle_simd_runtime_info info = HYLE_SIMD_RUNTIME_INFO_INIT;
if (!hyle_simd_runtime_info_get(&info))
  fprintf(stderr, "%s\n", hyle_simd_last_error());
```

`info.isa` 是整张 Runtime 的 broad tier；各 `info.math`、`info.dtype` 等字段是模块主
产品 tier 和精细 feature bits，不承诺模块内每个冷门 op 都使用该 ISA。输入结构必须用
`HYLE_SIMD_RUNTIME_INFO_INIT` 初始化，不能把全零结构当作可演进协议。

`hyle_simd_module_info.reserved` 固定写零；`features` 用
`HYLE_SIMD_FEATURE_*` 具名位解释，包括 F16C/BF16/VNNI 等硬件选择和
exact-libm scalar、fp8 scalar、lower-tier winner、SVE capped network、logical-128 bitpack
等产品合同。绑定不要手抄位值。

`HYLE_SIMD_RUNTIME_INFO_V1_SIZE` 钉住首版可读前缀。库接受 `struct_size` 不小于该值的结构，
只写自己认识的字段；未来在尾部追加字段时，旧调用方仍可继续传 V1 尺寸。

## 类型、所有权和绑定

- `hyle_simd_f16`、`hyle_simd_bf16`、`hyle_simd_f8e4m3`、`hyle_simd_f8e5m2`
  是 2/2/1/1 字节 bit-carrier。
- `hyle_simd_u128` 是按 `(hi, lo)` 排序的 16 字节、16 对齐 unsigned key。
- `hyle_simd_kv32` / `hyle_simd_kv64` 布局为 `{value, key}`，按 key 排序。
- 数组始终由调用者拥有；HYLE 不接管、不返回 `std::vector`，也不要求专用 allocator。
- `hyle_simd_random_create` 返回的 `hyle_simd_random_o*` 由
  `hyle_simd_random_destroy` 释放；destroy 和三个 fill 都接受 NULL 句柄（no-op）。
  分配失败返回 NULL，并可从 `hyle_simd_last_error` 取线程局部诊断。
- C#、Rust、Java 等绑定把自己的数组 pin/borrow 后传 pointer/count；容器 facade 属于
  各语言绑定，不形成第二套 HYLE ABI。

## 能力索引

本头当前有 **247 个 C 入口**：244 个动态库符号和 3 个 typed header-inline
`hyle_simd_packed_size_{u8,u16,u32}`。以下公式是逐符号清单；花括号表示每个列出的后缀都存在一个具体函数。

| 组 | 入口 |
|---|---|
| 诊断 | `hyle_simd_active_isa`、`hyle_simd_supports_isa`、`hyle_simd_runtime_info_get`、`hyle_simd_last_error` |
| 基础 ops | `hyle_simd_{dot_f32,dot_f64,dot_bf16,dot_f32_bf16,dot_f16,dot_i16,dot_i8,saxpy_f32,saxpy_f64,add_f32,add_f64}` |
| RNG | `hyle_simd_random_{create,destroy,fill_u64,fill_f32,fill_f64}` |
| scan | `hyle_simd_{inclusive_scan,exclusive_scan}_{u32,i32,u64,i64,f32,f64}` |
| search | `hyle_simd_{count_equal,find_equal,min_value,max_value,minmax_value,is_sorted}_{i8,u8,i16,u16,i32,u32,i64,u64,f32,f64}` |
| dtype | `hyle_simd_{f16_to_f32,f32_to_f16,bf16_to_f32,f32_to_bf16,f8e4m3_to_f32,f32_to_f8e4m3,f8e5m2_to_f32,f32_to_f8e5m2,f16_to_f8e4m3,f16_to_f8e5m2}` |
| sort | `hyle_simd_{sort,partial_sort,select}_{u16,i16,u32,i32,u64,i64,f32,f64,u128,kv32,kv64}` |
| radix/argsort | `hyle_simd_radix_sort_{u32,i32,u64,i64,f32}`、`hyle_simd_argsort_{u32,i32,f32}` |
| bitpack | `hyle_simd_{pack,unpack}_{u8,u16,u32}`、`hyle_simd_packed_size_{u8,u16,u32}` |

数学一元入口为 `hyle_simd_<name>_<precision>_batch`：

- f32：`exp`、`log`、`tanh`、`sin`、`cos`、`exp2`、`log2`、`erf`、`atan`、
  `cbrt`、`expm1`、`log1p`、`asinh`、`acosh`、`acos`、`asin`、`sinh`、`cosh`、
  `tan`、`atanh`、`log10`、`erfc`、`exp10`、`lgamma`、`tgamma`、`gelu`、
  `gelu_tanh`、`sigmoid`、`silu`。
- f64：上述通用实数函数（不含四个 f32-only activation），另有
  `exp_in_range_f64_batch` 和 `log_in_range_f64_batch`。
- 二元：f32 有 `atan2/pow/fmod/remainder/nextafter/fdim/hypot/hypot_in_range`；
  f64 有前七项（无 `hypot_in_range_f64`）。
- 其他：`hyle_simd_sincos_{f32,f64}_batch`、
  `hyle_simd_{softmax_f32,log_softmax_f32}`。
- 复数 SoA：`hyle_simd_{cexp,clog,csqrt,csin,ccos,ctan,csinh,ccosh,ctanh,cpow}_{f32,f64}_batch`。

头、纯 C test 与导出表通过预处理/符号表机械核对为 **247/247 test references、
244/244 exported symbols**；测试源是 `tests/simd/c_simd_test.c`。

## 合同

- `n > 0` 时所有必需指针必须有效并覆盖相应范围；`n == 0` 通常为 no-op。
- `hyle_simd_supports_isa` 只接受本头列出的 canonical ISA 常量；未知数值返回 0。它回答
  CPU/OS 是否支持该档，不是让普通调用方自行挑 kernel。
- 除明确支持 exact in-place 的接口外，输入输出不得重叠；带偏移 overlap 不受支持。
- `select` 要求 `k < n`，`partial_sort` 要求 `k <= n`；非法 sort enum fail-fast。
- bitpack 的 `bits` 必须在元素位宽内，packed bytes 是同一 tier 的内存格式，不是长期 wire format。
- `*_in_range_*` 不检查前置范围。浮点 dot/scan 的归约次序可随 ISA 宽度变化。
- RNG 是确定性的 xoshiro256++，不是密码学随机源，禁止用于 token、key 或 nonce。

需要固定整库 ISA 或在源码 TU 中直接写寄存器 kernel 时，见
[`simd-usage.md`](./simd-usage.md)；那两条是 expert/source deployment，不改变本页 C ABI。
