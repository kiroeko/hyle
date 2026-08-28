# Scalar math 使用指南

`hyle::math` 只补 C++23 缺失的少量 scalar integer 任务。普通标量数学用
`<cmath>/<numbers>/<bit>`；一段数组、lane 或 dtype 需要向量化时用
[`hyle::simd`](simd-usage.md)。

## 何时用

```cpp
#include <hyle/math/scalar.h>

const auto blocks = hyle::math::ceil_div(element_count, block_size);
const auto padded = hyle::math::round_up_to_multiple(element_count, block_size);
```

两个参数必须是同一个非 `bool` integral 类型。普通入口适合已经由程序结构保证的
shape/count：有符号值必须非负，除数/倍数必须为正，round-up 结果必须能由该类型
表示。违反这些程序不变量会 fail-fast。

来自模型、文件或网络的数值先用 checked 形状：

```cpp
auto blocks = hyle::math::try_ceil_div(element_count, block_size);
auto padded = hyle::math::try_round_up_to_multiple(element_count, block_size);
if (!blocks || !padded) {
  // zero/negative divisor, negative signed value, or unrepresentable round-up
}
```

checked 入口返回 `std::optional<T>`，不分配。对于 unsigned ceil-division，只要 divisor 非零，
结果总能表示；round-up 还可能因下一个倍数超过 `T::max()` 失败。

## 不要这样用

- 不要用 `round_up_to_multiple` 做 pointer/address alignment；power-of-two 地址对齐用 mem 的
  alignment 能力。
- 不要把 signed 负值当成“向上整除”输入；负数的 rounding direction 是另一种合同，
  math 不隐式猜测。
- 浮点 approximate equality 和死区 sign 不是公共 math 能力。请在消费算法中明确选择
  absolute/relative/ULP tolerance 及 NaN/infinity/zero 政策，或使用领域数值库。

## 交付

`math` 是 header-only 组件：源码组装或安装 SDK 都链接 `hyle::math`。static/shared 选项不会
制造一个没有符号的 math 库文件；它只决定所选闭包中编译型依赖的产物形状。
