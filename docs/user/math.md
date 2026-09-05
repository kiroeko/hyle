# Scalar math 使用指南

`hyle::math` 只补 C++23 缺失、又会在 shape/stride/byte sizing 中重复出错的少量 scalar integer
任务。普通标量数学用
`<cmath>/<numbers>/<bit>`；一段数组、lane 或 dtype 需要向量化时用
[`hyle::simd`](simd-usage.md)。

快速选择：

| 你要做的事 | 使用 |
|---|---|
| 普通标量三角、指数、常量或 bit primitive | C++ 标准库 |
| 检查或饱和处理整数 shape/count/stride/offset/byte-size | `hyle::math` |
| 对 lane、row 或 buffer 做 dtype、packing、activation 或向量数学 | `hyle::simd` |
| tensor/model 算子、layout 或融合 | Compute/消费层 |

Math 和 SIMD 不要求一起链接。`hyle::math` 当前只传递依赖 `hyle::core`；它不拉入 SIMD、Platform 或
第三方数值后端。若一个 SIMD/容器/内存实现需要这里的通用 checked scalar 合同，它应显式依赖 Math；
只定义该模块 layout 的专用 sizing 公式仍留在对应模块。

## 何时用

来自模型、文件或网络的尺寸先做 checked arithmetic，再分配或建立 tensor view：

```cpp
#include <hyle/math/scalar.h>

auto elements = hyle::math::try_multiply(rows, columns);
if (!elements) return invalid_shape();

auto bytes = hyle::math::try_multiply(*elements, element_size);
if (!bytes) return invalid_shape();

auto end = hyle::math::try_add(offset, *bytes);
if (!end) return invalid_shape();
```

`try_add`、`try_subtract`、`try_multiply` 成功时返回精确值，overflow/underflow 时返回
`std::nullopt`；不会返回 wrapped 或 saturated value。三个函数都不分配、不抛异常。可信且已经由
程序结构验证的普通算术继续直接写 `+/-/*`，不需要为了使用 HYLE 改成包装函数。

模型格式常用 signed 维度，而分配接口通常接收 `std::size_t`。用显式目标类型的 checked conversion，
再用 span product 或逐步 helper 检查，不依赖隐式 signed/unsigned promotion：

```cpp
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

std::vector<std::size_t> checked_dimensions;
for (std::int64_t dimension : dimensions) {
  auto checked = hyle::math::try_integer_cast<std::size_t>(dimension);
  if (!checked) return invalid_shape();
  checked_dimensions.push_back(*checked);
}
auto elements = hyle::math::try_product(std::span{checked_dimensions});
if (!elements) return invalid_shape();
```

`try_sum` 的 empty identity 是 0，`try_product` 是 1；两者单遍且不保留 span。需要计算
`base + count * stride` 时，可用 `try_multiply_add(count, stride, base)`；它先检查乘积，再检查加法，
不会让 unchecked intermediate 漏过。

普通除法和余数的 checked 形状是 `try_divide` / `try_remainder`。两者都拒绝零 divisor；signed
minimum / `-1` 的 quotient 不可表示而失败，remainder 的数学结果为零并安全返回零。

当业务明确要求“达到边界后保持边界”而不是拒绝输入时，才选择 saturating 路线：

```cpp
auto level = hyle::math::saturating_add(current_level, delta);
auto byte = hyle::math::saturating_cast<std::uint8_t>(sample);
```

`saturating_add/subtract/multiply/divide/cast` 返回 clamp 后的值，不返回 optional。除零不是 saturation，
`saturating_divide` 要求 divisor 非零；外部输入先用 `try_divide`。

分块与 padding 使用 ceil/round：

```cpp
const auto blocks = hyle::math::ceil_div(element_count, block_size);
const auto padded = hyle::math::round_up_to_multiple(element_count, block_size);
```

两个参数必须是同一个非 `bool` integral 类型。普通入口适合已经由程序结构保证的
shape/count：有符号值必须非负，除数/倍数必须为正，round-up 结果必须能由该类型
表示。违反这些程序不变量会 fail-fast。

来自模型、文件或网络的除数/倍数也先用 checked 形状：

```cpp
auto blocks = hyle::math::try_ceil_div(element_count, block_size);
auto padded = hyle::math::try_round_up_to_multiple(element_count, block_size);
if (!blocks || !padded) {
  // zero/negative divisor, negative signed value, or unrepresentable round-up
}
```

全部 checked 入口返回 `std::optional<T>`，不分配。对于 unsigned ceil-division，只要 divisor 非零，
结果总能表示；round-up 还可能因下一个倍数超过 `T::max()` 失败。

## 不要这样用

- 两个 arithmetic operand 必须是同一个非 `bool` integral 类型。需要组合 `uint32_t` shape 与
  `size_t` element size 时，先用 `try_integer_cast` 转到共同类型；不要依赖 implicit signed/unsigned
  promotion。整数 literal 也应写成目标类型，例如 `std::size_t{1}`。
- 不要在 overflow 后继续使用原输入计算，或把 `nullopt` 当作零。多维 shape product 应逐步检查每次
  `try_multiply`，这样能准确拒绝第一个不可表示的中间结果。
- `try_subtract` 的 unsigned `lhs < rhs` 是 underflow，返回 `nullopt`；它不是绝对差。
- 不要用 `round_up_to_multiple` 做 pointer/address alignment；power-of-two 地址对齐用 mem 的
  alignment 能力。
- 不要把 signed 负值当成“向上整除”输入；负数的 rounding direction 是另一种合同，
  math 不隐式猜测。
- 不要把 checked 与 saturating 结果混用：前者保持“精确或失败”，后者明确接受 clamp；饱和除法也
  不会为零 divisor 发明数值。
- 浮点 approximate equality 和死区 sign 不是公共 math 能力。请在消费算法中明确选择
  absolute/relative/ULP tolerance 及 NaN/infinity/zero 政策，或使用领域数值库。

## 交付

`math` 是 header-only 组件：源码组装或安装 SDK 都链接 `hyle::math`。static/shared 选项不会
制造一个没有符号的 math 库文件；它只决定所选闭包中编译型依赖的产物形状。
