# Scalar math 使用指南

`hyle::math` 只补 C++23 缺失、又会在 shape/stride/byte sizing 中重复出错的少量 scalar integer
任务。普通标量数学用
`<cmath>/<numbers>/<bit>`；一段数组、lane 或 dtype 需要向量化时用
[`hyle::simd`](simd-usage.md)。

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

- 两个 operand 必须是同一个非 `bool` integral 类型。需要组合 `uint32_t` shape 与 `size_t` element
  size 时，先显式验证并转换到共同类型；不要依赖 implicit signed/unsigned promotion。
- 不要在 overflow 后继续使用原输入计算，或把 `nullopt` 当作零。多维 shape product 应逐步检查每次
  `try_multiply`，这样能准确拒绝第一个不可表示的中间结果。
- `try_subtract` 的 unsigned `lhs < rhs` 是 underflow，返回 `nullopt`；它不是绝对差。
- 不要用 `round_up_to_multiple` 做 pointer/address alignment；power-of-two 地址对齐用 mem 的
  alignment 能力。
- 不要把 signed 负值当成“向上整除”输入；负数的 rounding direction 是另一种合同，
  math 不隐式猜测。
- 浮点 approximate equality 和死区 sign 不是公共 math 能力。请在消费算法中明确选择
  absolute/relative/ULP tolerance 及 NaN/infinity/zero 政策，或使用领域数值库。

## 交付

`math` 是 header-only 组件：源码组装或安装 SDK 都链接 `hyle::math`。static/shared 选项不会
制造一个没有符号的 math 库文件；它只决定所选闭包中编译型依赖的产物形状。
