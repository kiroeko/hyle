# SIMD 使用指南

> SIMD 当前只提供 C++ 产品面，可通过完整源码、headers + static libraries 或
> headers + shared libraries 使用。`hyle/simd/*.h` 中的 whole-buffer 算法供普通 C++ 用户
> 使用；三种交付都可通过显式 expert target 在 target-bound TU 中直接使用项目锁定的 Highway 或
> xsimd。旧 `vec.h`/`native/*`/`backend/*` 已删除；`target.h`、`fat_kernel.h` 随 expert SDK 交付，
> `provider/*` 仍是 HYLE 内部实现。预编译交付遵守其明确的 C++ 工具链合同，
> 本页适用于 C++23 source/static/shared 三种交付，不承诺任意编译器之间的 C++ ABI，
> 也不描述 C API 或跨语言调用层。
>
> 内部 route ledger 已对冻结的 387 个旧 operation 全部终态裁决。Highway 是同合同默认实现；xsimd/HYLE
> 只保留精确、具证据的 type × target × compiler 例外。expert consumer 可以取得两个锁定 vendor 和
> fixed/fat target helper；HYLE 内部 route contract 与有限 HYLE winner body 不因此成为用户 API。

HYLE SIMD 不要求普通用户认识数百个寄存器原语，也不再复制一套平行 Highway DSL。应用通常只应
调用处理整段连续数据的语义接口，让预编译库在运行时选择当前 CPU
可用的 scalar、SSE4、AVX2、AVX-512、NEON、SVE 或 RVV 实现。

## 先选择使用层

| 需求 | 应选层 | 是否运行时选 target | 稳定性 |
|---|---|---:|---|
| C++ 应用发布一个二进制给未知 CPU | `hyle/simd/{ops,math,...}.h` whole-buffer source API | 进程内解析一次；每个整批调用一次间接调用 | 同受支持工具链的 C++ source contract |
| C++ 用户调用 whole-buffer 算法 | `hyle/simd/{ops,math,...}.h` | 同上 | 原生 source API，不承诺跨工具链 ABI |
| HYLE 没有所需 whole-buffer 语义，只需在当前编译目标下使用 vendor 原语 | `hyle::simd_expert` + direct `hwy::`/`xsimd::` body | 无 HYLE runtime selector；采用 consumer target 的现有编译 flags | source/static/shared 同一锁定 header surface；vendor 类型不得跨 TU |
| 整个 HYLE 只在一种已知 ISA 上运行 | `HYLE_SIMD_EXACT_TARGET` | 无；宏生成并直接绑定目标 symbol | 构建配置 |
| 自己写一个固定 ISA 的 kernel target | 单独 OBJECT target + `hyle_target_simd(...)` + direct Highway/xsimd body | 无；宏/模板单态化并可内联 | helper 作用于整个 CMake target；vendor 类型不得跨 TU |
| 自己写一个 kernel，仍要发布给未知 CPU | `hyle_add_fat_kernel` + provider-coherent impl header | 进程内解析一次；每个整批调用一次间接调用 | source/static/shared 同一 expert surface；vendor 类型不得跨 TU |

```text
                              C++ 普通应用
                                  |
                    hyle/simd whole-buffer API
                                  |
              +-------------------+-------------------+
              |                                       |
       未知客户 CPU                            已知目标的 HYLE 构建
   首次解析并缓存实际 family 实现             HYLE_SIMD_EXACT_TARGET
              |                                       |
 scalar / SSE4 / AVX2 / AVX-512 / NEON          直接绑定一个实现
              \                                       /
               +------- HYLE pointer/count kernel -----+
                                  |
          locked Highway / xsimd / qualified HYLE body
```

运行时选择不是每个元素做 CPUID，也不是在循环里反复判断 ISA。平台能力只探测一次；之后**每次调用**
whole-buffer API 都经一个已缓存的函数入口进入 kernel。这里不是“同一 buffer 对象一生只路由一次”——
pointer/count API 不给用户内存增加路由 metadata。调用很小的数据时，函数边界本身仍可能比计算贵，
因此应批量调用；部署 ISA 已知或必须融合 tiny kernel 时，改用下文 fixed/exact 源码路径，其宏和模板会
直接绑定目标实现，运行时函数指针成本为零。

若只为日志或部署诊断查看 HYLE 实际选中的 tier，而不是原始 CPU 能力：

```cpp
#include <cstdio>

#include "hyle/platform/cpu.h"
#include "hyle/simd/runtime.h"

const auto math = hyle::simd::selected_target(hyle::simd::KernelFamily::kMath);
std::printf("math target: %s\n", hyle::simd::kernel_target_name(math.target));
const auto dtype = hyle::simd::selected_target(hyle::simd::KernelFamily::kDtype);
bool has_f16c = dtype.has(hyle::simd::TargetFeature::kF16c);
```

Platform 只公开无顺序的单项 `CpuFeature` facts，不存在一个诚实的“主机最佳 ISA”。
`selected_target()` 报告某个 HYLE family 实际采用的 compiled target；不同 family 可能因编译器
blocklist、可选扩展或已认证的 lower-target winner 而不同。`KernelTarget` 也不能跨架构排序，诊断
API 不能 pin 实现。

## 常用能力速查

| 头（开发期） | 常用能力 | 适合做什么 |
|---|---|---|
| `ops.h` | f32/f64 row 算术、归约、同精度/混合精度 `dot`、`saxpy` | 连续 row、归一化/激活前处理与量化存储 dot |
| `random.h` | `RandomState`、`seed/copy/advance/split`、`random_fill` | 可复现、跨 ISA 位一致的 worker-owned 批量随机数；非加密 |
| `scan.h` | `inclusive_scan`、`exclusive_scan` | 输入输出分离或原地前缀和；exclusive 可指定 init；支持 u/i32、u/i64、f32、f64 |
| `search.h` | count/find/min/max/minmax/argmin/argmax/is_sorted | 十种普通 scalar 数组的查找、归约、极值位置和有序性检查 |
| `dtype_ops.h` | f32/f16/bf16/E4M3/E5M2 间全部异类型转换 | 模型权重、激活、KV cache 和混合精度数据搬运 |
| `math.h` | 实数/复数数学、f32/f16/bf16 激活、split GLU、row norm、softmax | 数值计算、统计和 AI 热路径 |
| `sort.h` | 非破坏 score+index top-k、sort、partial sort、select、radix、argsort | MoE/词表选择、排序、nth element、稳定整数/浮点 radix |
| `bitpack.h` | canonical-v1 `pack`、`unpack`、`packed_size` | 模型载荷、文件或网络 framing 内的定宽整数 payload |
| `<hwy/highway.h>`（expert target + `hyle_target_simd`） | 锁定版本的 register/tag/load/store/arithmetic/reduction 等 | 三种交付中的自定义固定目标 kernel |
| `<xsimd/xsimd.hpp>`（expert target + `hyle_target_simd`） | 锁定版本的 batch、mask、数学与 load/store | 三种交付中的自定义固定目标 kernel |

### 数学接口分组

- f32/f64 一元函数：`exp`、`log`、`sin`、`cos`、`tanh`、`exp2`、`log2`、`erf`、`atan`、
  `cbrt`、`expm1`、`log1p`、双曲函数、gamma 等，命名形如 `exp_f32_batch`。
- 二元函数：`atan2`、`pow`、`hypot`、`fmod`、`remainder`、`nextafter`、`fdim`。
- AI 语义：exact/tanh/quick GELU、`sigmoid_f32_batch`、`silu_f32_batch`，以及
  `relu_f32_batch`、`leaky_relu_f32_batch`、`elu_f32_batch`、`hard_sigmoid_f32_batch`、
  `hard_swish_f32_batch`。GELU/SiLU、split ReGLU/GeGLU/SwiGLU 和 RMSNorm/LayerNorm 另有
  f16/bf16 storage 入口；`softmax_f32` / `log_softmax_f32` 同时提供 out-of-place 核心与原地重载。
- 复数 SoA：f32/f64 均提供 `cexp`、`clog`、`csqrt`、`csin`、`ccos`、`ctan`、`csinh`、
  `ccosh`、`ctanh`、`cpow`；实部和虚部放在两个数组中，避免把 `std::complex` 对象布局
  固化到算法边界上。命名形如 `ctan_f32_batch` / `ctan_f64_batch`，输入输出可完全同址。
- 普通实数入口保持对应 `std::` 函数的域和特殊值语义；内部直接采用中央锁定的 Highway，
  vendor 类型与版本不会进入用户接口。Highway 明确不覆盖的域由 HYLE 回退到标量 oracle。
- `exp/log_in_range_f32/f64_batch` 与 `hypot_in_range_f32_batch` 是调用者承诺输入范围的快速合同，
  不是自动检查后回退的安全版本。它们适合 softmax 等已经由上层证明范围的连续 row；不确定输入时
  使用普通入口。精确原位 `src == dst` 可用，任意部分重叠不可用。

### Activation 与 softmax 怎么选

- GELU 的 exact/erf、常见 tanh approximation 与 quick `x*sigmoid(1.702*x)` 是三个不同数值合同；
  模型指定哪一种就调用哪一种，不要因为名字相近互换。SiLU 就是这里提供的 swish 形状；移动端常用的
  分段近似则选 hard-swish。
- gate/value 已经是两个连续 row 时，直接使用 `reglu_*_batch`、`geglu_*_batch`、
  `geglu_tanh_*_batch`、`geglu_quick_*_batch` 或 `swiglu_*_batch`。它们计算
  `activation(gate[i]) * value[i]`，不会替用户解释 packed tensor、stride 或哪一半是 gate。
- 单行不减均值选择 `rms_norm_*`，需要减均值选择 `layer_norm_*`；不带 learned 参数的重载只做
  normalization，带参数的重载分别接受 weight 或 scale+bias。epsilon 必须有限且非负。
- leaky-ReLU 的 `negative_slope` 和 ELU 的 `alpha` 是调用者/模型参数；HYLE 不猜训练框架默认值。
- elementwise activation 与 softmax 核心都接受 `src, dst, n`；exact `src == dst` 或完全不相交均可，
  带偏移的部分 overlap 不受支持。`n == 0` 时输入输出都可以是空指针。
- softmax/log-softmax 是稳定的连续 f32 单 row 算法。至少有一个 finite lane 时，`-inf` lane 可作为 mask；
  row 含 NaN、`+inf`，或全为 `-inf` 时没有唯一受支持的 normalizer，目标整行写 quiet NaN。
- f16/bf16 入口按 f32 计算后只在最终 storage 写回时舍入；它们不是“全程半精度”的另一套数值合同。
- 这些入口不分配、不加锁，也不在元素循环里再次 dispatch。带 attention mask、tensor stride/batch、线程、
  fused residual/group norm 或 operator fusion 的形状属于 Compute；不要在 SIMD 外层逐元素反复调用
  whole-buffer API。

## C++ whole-buffer 示例

```cpp
#include "hyle/simd/dtype.h"
#include "hyle/simd/ops.h"

float x[4096]{}, y[4096]{}, out[4096]{};
hyle::simd::add(x, y, out, 4096);
float inner = hyle::simd::dot(x, y, 4096);
hyle::simd::saxpy(0.25f, x, y, 4096);
```

模型权重或激活以窄类型存储时，不要先扩成一个临时 f32 数组；同一个入口会在寄存器中 widening：

```cpp
hyle::simd::bfloat16_t weights[4096]{};
hyle::simd::float16_t left[4096]{}, right[4096]{};
float activations[4096]{};

float score = hyle::simd::dot(activations, weights, 4096);
float half_dot = hyle::simd::dot(left, right, 4096);
```

`bf16×bf16`、`f32×bf16` 和 `f16×f16` 都累加到 f32；`f32×bf16` 不会把 f32 输入降精度。
`i8×i8` 与 `i16×i16` 的乘积精确，以模 `2^64` 的二补码合同返回 i64，因此溢出行为
不依赖 C++ UB。i8 路径会在可用时自动采用 AVX-VNNI、AVX512-VNNI、NEON dotprod
或 SVE dot；默认 AVX2/AVX512 基线本身不要求这些可选扩展。

```cpp
#include "hyle/simd/random.h"

hyle::simd::RandomState state{12345};
float unit[4096];
hyle::simd::random_fill(state, unit, 4096);  // [0, 1)

state.seed(12345);       // 回到相同序列起点
auto worker0 = state.split();
auto worker1 = state.split();  // 与 worker0 位于不同的 2^192 segment
worker0.advance(1024);   // O(n) 跳过，不写临时 output
```

相同 seed 和相同的 fill 调用序列在不同 ISA 上位一致。它使用 xoshiro256++，不能生成 token、密钥、
nonce 或其他安全材料。copy construction/assignment 会复制当前位置，之后两份 state 会生成相同后续
序列；一个 mutable state 不可由多个线程并发调用。并行 worker 应在发布前各自取得 `split()` child，
而不是共享 state。单个普通随机数优先使用 `hyle/util/random.h` 的标量 `Rng`；批量填充才使用这里。

```cpp
#include "hyle/simd/scan.h"

const std::uint32_t counts[] = {3, 1, 4, 1, 5};
std::uint32_t offsets[5]{};
hyle::simd::exclusive_scan(counts, offsets, 5, 7u);
// offsets == {7, 10, 11, 15, 16}

hyle::simd::inclusive_scan(offsets, 5);  // exact in-place convenience
// offsets == {7, 17, 28, 43, 59}
```

选择 scan 形状时：已有只读 counts 且还要保留它，就用 `const src, dst, n` 核心；允许覆盖输入时才用
两参数原地 convenience。exclusive 的第四个参数是第一个输出值/初始 accumulator，不是“最后再整体
加一次”。输入输出可以完全分离或精确同址，带偏移的部分 overlap 不受支持。整数（包括 signed）按
模 `2^bits` 求和，不触发 signed-overflow UB；浮点结果的结合顺序可以随实际 SIMD tier 改变。

```cpp
#include "hyle/simd/search.h"

const float logits[] = {1.0f, 9.0f, -2.0f, 9.0f, 4.0f};
const std::size_t first_max = hyle::simd::index_of_max(logits, 5);  // 1
const std::size_t matches = hyle::simd::count_equal(logits, 5, 9.0f);  // 2
```

`find_equal` 和空 range 的 `index_of_min/index_of_max` 返回 `n`；相同极值取第一次出现的位置。浮点
equality 使用 IEEE `==`，所以 NaN 永不匹配；value extrema/minmax 遇到任意 NaN 会传播 quiet NaN，
index extrema 则返回第一个 NaN 的位置。`minmax_value` 的两个输出指针即使在 `n == 0` 时也必须有效且
互不相同。需要自定义 predicate 或 scan operation 时用标准算法或源码 expert kernel，不要为每个元素
增加 runtime callback。

```cpp
#include "hyle/simd/dtype_ops.h"

hyle::simd::float16_t packed[4096]{};
hyle::simd::bfloat16_t cache[4096]{};
float values[4096]{};
hyle::simd::convert(values, packed, 4096);
hyle::simd::convert(packed, cache, 4096);  // 一次遍历，不建立 f32 临时数组
hyle::simd::convert(cache, values, 4096);
```

五种公开 storage 间的全部 20 条异类型有向边都使用同一个可发现的 `convert(src, dst, n)`
重载名；同类型复制继续使用标准库。窄化遵守 `dtype.h` 的 RNE、饱和与 NaN 合同。`n == 0`
允许空指针；相同宽度的转换允许输入输出完全同址，不同宽度必须不重叠，任意部分重叠均不受支持。

整段 f16/bf16 查找、求和和 total-order 检查使用普通的 `narrow_ops.h`，不需要手写寄存器循环：

```cpp
#include "hyle/simd/narrow_ops.h"

namespace hs = hyle::simd;
std::size_t numeric = hs::count_equal(values, n, needle);      // IEEE ==
std::size_t exact = hs::find_equal_bits(values, n, needle);    // exact encoding
float total = hs::sum(values, n);                              // f32 accumulation
bool ordered = hs::is_sorted_total(values, n);
```

`count_equal/find_equal` 把 signed zero 视为相等、NaN 永不匹配；带 `_bits` 的入口则区分
signed zero 和每个 NaN payload。`narrow_total_order_key` 是可逆的 IEEE-754 `totalOrder`
key：negative qNaN < negative sNaN < -inf < ... < -0 < +0 < ... < +inf <
positive sNaN < positive qNaN。这是 bitwise deterministic order，不是“NaN 都相等”。

如果要在固定 target TU 内创作自己的窄类型 kernel，使用下文的
`hyle_target_simd()` 并直接调用锁定 Highway 的 `float16_t/bfloat16_t`、`PromoteTo/DemoteTo`
等接口。HYLE 不再提供一套平行的 narrow register DSL；vendor 类型不能跨 TU，且自定义 kernel
必须自行声明 NaN、rounding、tail 与 alias 合同。普通用户继续使用上述 HYLE whole-buffer API。

```cpp
#include "hyle/simd/math.h"

hyle::simd::exp_f32_batch(src, dst, n);
hyle::simd::gelu_f32_batch(src, dst, n);
hyle::simd::gelu_quick_f32_batch(src, dst, n);
hyle::simd::relu_f32_batch(src, dst, n);
hyle::simd::swiglu_f32_batch(gate, value, dst, n);
hyle::simd::rms_norm_f32(src, weight, dst, n, 1e-5f);
hyle::simd::softmax_f32(src, probabilities, n);  // out-of-place
hyle::simd::log_softmax_f32(dst, n);             // 原地 convenience
```

### 排序、top-k 与稳定索引怎么选

| 需求 | 使用 | 数据与资源合同 |
|---|---|---|
| 全量排序，不要求相等项稳定 | `sort(keys, n, order)` | 原地 comparison ordering；无隐藏的大数组分配 |
| 只要排好序的前 `k` 项 | `partial_sort(keys, n, k, order)` | 原地；前 `k` 项有序，尾部内容未指定 |
| 只要第 `k` 项及两侧 partition | `select(keys, n, k, order)` | 原地；不要为只取阈值付全排序成本 |
| 保留 f32 score，只取最高 `k` 个 score+原 index | `top_k_f32(scores, n, result, k)` | 输入不变；caller 提供 `k` 个结果槽，NaN-last、tie 按较小 index |
| 整数 key 和整数 payload 一起移动 | `partial_sort(KeyValue32/64*, ...)` | 按 `key` 排序，`value` 跟随；相同 key 不稳定 |
| 值本身要稳定排序 | `radix_sort(keys, n, scratch, order)` | base-256；显式 `n` 个同类型 scratch，不分配 |
| 保留原 key，只输出稳定索引 | `argsort(keys, indices, n, scratch, order)` | 输出 `uint32_t` permutation；需要 `n` 个 index scratch |

只需要 best `k` 个浮点 score 和原 index 时优先用 `top_k_f32`，避免破坏 score 或为全排列分配 O(n)
scratch；需要全部稳定 permutation 时才用 `argsort`。不要把浮点 bit pattern 塞入只按整数 key 比较的
`KeyValue32/64`。`top_k_f32` 的 `k <= 8` 是 MoE 常见 bounded-k 的 O(n) 实现，其余 k 走 heap/full-sort
复杂度；这不会改变公开排序、NaN 或 tie 合同。普通 `sort` 也不会按长度偷偷分配 radix workspace。

```cpp
#include "hyle/simd/sort.h"
#include <vector>

hyle::simd::sort(keys, n, hyle::simd::SortOrder::kAscending);
hyle::simd::partial_sort(keys, n, 10, hyle::simd::SortOrder::kAscending);
hyle::simd::select(keys, n, k, hyle::simd::SortOrder::kAscending);

std::vector<hyle::simd::ScoredIndexF32> selected(k);
hyle::simd::top_k_f32(read_only_scores, n, selected.data(), selected.size());

std::vector<float> value_scratch(hyle::simd::radix_sort_workspace_size(n));
hyle::simd::radix_sort(keys, n, value_scratch,
                       hyle::simd::SortOrder::kAscending);

std::vector<std::uint32_t> indices(n);
std::vector<std::uint32_t> index_scratch(hyle::simd::argsort_workspace_size(n));
hyle::simd::argsort(read_only_scores, indices.data(), n, index_scratch,
                    hyle::simd::SortOrder::kDescending);
```

普通 key 支持 i/u16、i/u32、i/u64、f32/f64。标准 C++ 没有可移植 `uint128_t` 类型，因此使用
`UInt128 {lo, hi}`，按 `(hi, lo)` 的 unsigned 数值序排序。需要“分数排序时索引跟着走”时使用
`KeyValue32/64`；需要稳定排序时显式用 `radix_sort` 或 `argsort`，不要假设普通 `sort`
对相等 key 稳定。stable 浮点排序把所有 NaN 放在有限值之后（升降序均如此），保留 NaN payload 间及
`-0/+0` 间的输入顺序。scratch 重载不会分配；省略 scratch 的 convenience 会分配恰好 `n` 个元素并可能
抛出 `std::bad_alloc`，不会在 OOM 时静默改用不稳定算法。`argsort` 使用 32 位 index，因此 `n - 1`
必须能由 `uint32_t` 表示。

```cpp
#include "hyle/simd/bitpack.h"

const unsigned bits = 11;
std::size_t packed_count = hyle::simd::packed_size<std::uint16_t>(raw_count, bits);
std::uint16_t* packed = allocate_u16(packed_count);
hyle::simd::pack(raw, raw_count, packed, bits);
hyle::simd::unpack(packed, raw_count, decoded, bits);
```

`kBitpackFormatVersion == 1` 的 payload 是 canonical little-endian wire layout：每个完整块有 128 个
value，不足一块的尾部按 value-order LSB-first 编码，最后 padding bits 为零；scalar、x86、NEON、
SVE 与 RVV 产生相同字节，VL/VLEN 不影响格式。codec 本身不写 framing，持久化或传输时调用者必须同时
记录 type、value count、`bits` 和 format version。只把 payload 存下来而丢掉这些字段无法可靠解码。

## 必须遵守的调用合同

- HYLE 的 pointer/count core 不知道各容器长度。输入输出长度匹配、`n > 0` 时指针非空，由调用者保证。
- `n == 0` 在上述 batch/scan/search/sort 接口通常允许空 data 指针；search 的 `minmax_value` 输出仍须有效。
- 只有头文件明确写出的 exact in-place alias 才受支持；带偏移的重叠区间通常不受支持。
- `partial_sort` 要求 `k <= n`；`select` 在 `n > 0` 时要求 `k < n`。
- `radix_sort` 的 scratch 与 keys、`argsort` 的 keys/indices/scratch 三者都不得重叠；workspace helper
  返回的是元素数，不是字节数。热路径应复用 caller-owned scratch，只有便利重载可能分配/抛异常。
- bitpack 的 `bits` 必须在元素位宽内，每个输入值必须已经能放入 `bits`；codec 不做隐式 mask pass。
  输入输出不得重叠，输出容量至少为 `packed_size<T>(count, bits)`。
- signed integer scan 明确定义为 modulo；浮点 scan 的结合顺序随 ISA 宽度变化，不要要求不同 ISA 得到
  逐位相同的浮点前缀和。
- 浮点 dot 也使用多累加器/vector reduction；结果遵守数值容差而非跨 ISA 逐位相同。
- 窄合同 `InRange` API 不做逐 lane 防御；违反范围前置条件的结果不受保证。
- 所有批量 RNG 都是非加密随机数；同一个 mutable `RandomState` 不可并发调用。

## 直接使用锁定的 Highway/xsimd 头

如果 ordinary whole-buffer API 没有所需语义，而当前 target 的编译 flags 已经是你想要的部署基线，可把
自定义 kernel 放在单独的 OBJECT target 中并显式链接专家层：

```cmake
add_library(my_kernel OBJECT my_kernel.cc)
target_link_libraries(my_kernel PRIVATE hyle::simd_expert)
target_sources(my_app PRIVATE $<TARGET_OBJECTS:my_kernel>)
```

`my_kernel.cc` 可直接包含 `<hwy/highway.h>` 或 `<xsimd/xsimd.hpp>`。这条路径只投影 HYLE 锁定的
provider headers、C++23 合同和必要定义，不探测运行时 CPU，也不会替 consumer 增加 ISA 编译 flags；
因此它适合已有 target policy 的 kernel 或基线实现，不会自动把未知客户 CPU 路由到最快 ISA。需要固定
ISA 时使用下一节的 `hyle_target_simd()`；同一二进制要覆盖未知 CPU 且利用多种 ISA 时使用
`hyle_add_fat_kernel()`。三种选择都应让公开 entry 保持 pointer/count/POD，vendor 类型只留在 kernel TU。

## 已知目标：固定整库或固定 kernel

如果交付环境明确要求某个 ISA，可在配置 HYLE 时固定整个 SIMD 库：

```sh
cmake -S . -B build-avx2 -DHYLE_SIMD_EXACT_TARGET=AVX2
```

这不是“逃避动态税”，而是另一种产品合同：二进制本来就只允许运行在 AVX2 机器上，因此可以直接绑定
AVX2 symbol。未知客户 CPU 的发布包应使用默认 fat build。

只想固定自己的一两个 kernel TU 时，把它们放在专用 CMake target 中，再用 `hyle_target_simd` 为整个
target 设置目标 ISA。源码嵌入、安装 static SDK 和安装 shared SDK 使用同一个 helper；它会显式链接
`hyle::simd_expert`，提供中央锁定 Highway/xsimd 的 include path、单目标宏和统一编译器 flags。普通
`hyle::simd` consumer 不继承这些 vendor 头，因此不用专家层时没有额外编译或运行成本。

```cmake
add_library(my_avx2_kernel OBJECT my_kernel.cc)
hyle_target_simd(TARGET my_avx2_kernel ISA AVX2)
target_sources(my_app PRIVATE $<TARGET_OBJECTS:my_avx2_kernel>)
```

helper 的粒度是 CMake target，不是单个 source 参数。除非整个应用本来就要求该 ISA，不要直接把它施加到
含大量普通源码的 executable/library target；使用上述单独 OBJECT target，能把 ISA 要求和 provider 类型
限制在 kernel TU 中。

kernel 内可以显式使用 Highway（xsimd 同理），而不是再经一套 HYLE 寄存器转发层：

```cpp
#include "hwy/highway.h"

void bias_relu(float* values, std::size_t n, float bias) {
  namespace hn = hwy::HWY_NAMESPACE;
  const hn::ScalableTag<float> d;
  const auto vbias = hn::Set(d, bias);
  const auto zero = hn::Zero(d);
  const std::size_t lanes = hn::Lanes(d);
  std::size_t i = 0;
  for (; i + lanes <= n; i += lanes) {
    const auto value = hn::LoadU(d, values + i);
    hn::StoreU(hn::Max(hn::Add(value, vbias), zero), d, values + i);
  }
  for (; i < n; ++i) values[i] = std::max(values[i] + bias, 0.0f);
}
```

Highway vector/mask 类型和 lane 数依赖目标，不能出现在跨 TU 的公开函数签名、shared-library ABI 或
持久化格式中。entry 保持 pointer/span/POD；tail、alias、异常与精度合同仍由你的 kernel 明确承担。
经典 cl.exe 被 Highway blocklist 禁止 AVX3，因此 helper 对 `ISA AVX512` 明确报错；使用 clang-cl 或
选择 AVX2，不能把静默降级当作 exact AVX-512。

`hyle::simd_expert` 不等于“把 Highway/xsimd 全部改名到 `hyle::simd`”。如果 ordinary API 没有你需要的
原语或 header-only contrib，在绑定该 target 的 TU 中直接使用 `hwy::...` / `xsimd::...`；这样可以取得
两家完整的锁定头文件能力，升级时也仍由上游维护。需要编译上游 `.cc` 的 thread pool、完整
VQSort/image/hash 等不在该 target 的承诺内，不能仅因头文件可见就假定已链接。

要把自己的 pointer/count kernel 随 HYLE 一起多版本编译，则让 impl header 根据
`HYLE_SIMD_VARIANT_SUFFIX` 导出 suffixed entry，再用：

```cmake
hyle_add_fat_kernel(
  TARGET my_app NAME my_transform IMPL "src/my_transform-inl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
```

baseline 调用点用 `HYLE_SIMD_SELECT_FAT_KERNEL(...)` 选择并缓存入口。不要直接手写 CPUID，也不要让
vector/mask 出现在 entry 签名中。impl header 可以显式固定一个 vendor，也可以让已资格化的 HYLE route
选择一套完整 provider body；不能在 element loop 中来回桥接两家 register 类型。生成 TU 会冻结 target，
Highway body 还会在编译期验证实际 `HWY_STATIC_TARGET`。这个选择宏在默认 fat 构建中首次使用时
解析平台事实并缓存函数指针，之后每个整批调用只有一次间接函数调用；它不会每个 buffer 重新探测 CPU。
在 `HYLE_SIMD_EXACT_TARGET` 构建中则只引用唯一生成的 exact symbol，运行时没有 selector 或函数指针；
在 macOS ARM64 上也不会引用未生成的 SVE symbol。fat `TARGETS` 必须列出本平台完整 canonical tier，
CMake 会在配置期检查。

不要为寄存器操作增加运行时 callback：逐 lane function pointer 会阻断内联和向量化。fixed source
由宏 + 模板单态化直接内联；source-fat 仅在 whole-kernel pointer/count 边界有一次可预测间接调用，
vector loop 内没有 HYLE wrapper、feature switch 或第二层 Highway dynamic dispatch。

HYLE canonical x86 tier 不承诺 AES/PCLMUL，因此 helper 使用 Highway 官方的
`HWY_DISABLE_PCLMUL_AES` 使 vendor target 与实际 Platform facts 一致。需要 crypto 指令的 specialty
kernel 必须单独声明/探测这些扩展。经典 cl.exe 的 source-fat 最高为 AVX2；fixed/exact AVX-512 会
fail-fast。需要 direct Highway AVX-512 时使用 clang-cl。

仓库的 `samples/02_simd` 同时演示 ordinary、fixed source 与 source-fat 三种路径：设置
`HYLE_SAMPLE_SIMD_ISA=AVX2` 构建 fixed path，或设置 `HYLE_SAMPLE_SIMD_FAT=ON` 构建 unknown-CPU
source-fat path；两者是不同部署合同，不能同时开启。

## C++ 产品面

源码用户以及符合交付工具链合同的 static/shared 用户，直接调用
`hyle::simd::dot/sort/...`。这是当前正式用户面；C++ shared library 不等于跨编译器稳定 ABI。
