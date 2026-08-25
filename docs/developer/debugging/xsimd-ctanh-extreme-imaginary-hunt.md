# HYLE SoA `ctanh` 极大有限虚部 Hunt

> 调查日志（已结案）。触发：对新 SoA `ctanh` 的对抗式源码审查要求每个公开契约内 lane 都能保留 `std::complex` 语义，而不只验证常规随机值。

**结论先行：** 初版把 `abs(real) > 20` 的有限 lane 直接归入渐近向量路径，却没有限制 `abs(imag)`。当 `FLT_MAX / 2 < abs(imag) <= FLT_MAX` 时，内部 `2 * imag` 溢为 `Inf`，随后 `sincos(Inf)` 不能给出有限三角值；全-large block 还会绕过异常 lane 覆盖，直接把这个中间结果写出。此输入仍是有限的公开输入，不能以“罕见”排除。修复后，任一实/虚轴、subnormal、非有限、近极点，或会使 `2 * imag` 溢出的有限 lane 都逐 lane 走 `std::tanh(std::complex<float>)`；同 block 的正常 lane 仍保持向量化。

相关实现：[`complex_hyle-inl.h`](../../../src/simd/math/complex_hyle-inl.h)。回归：[`math_test.cc`](../../../tests/simd/math_test.cc)。

## 1. 现象与契约

`ctanh(x + iy)` 的 regular formula 使用 `sincos(2y)`：

```text
re = sinh(2x) / (cosh(2x) + cos(2y))
im = sin(2y)  / (cosh(2x) + cos(2y))
```

大实部采用 `q = exp(-2|x|)` 的等价渐近式，避免双曲函数溢出。初版已正确处理了 `x` 的大值，却遗漏了这一事实：即使 `y` 自身有限，`2*y` 仍可能溢出。完整 API 的 scalar floor 是 `std::complex`，所以候选必须将这种 lane 当作异常，而不是用 vector 中间值改变结果。

## 2. 先怀疑候选分类，不把随机通过当作证明

常规 random、large-real、near-pole 测试都通过，不能说明 full range 已正确：它们的虚部范围分别在 `[-2,2]` 或接近极点，不会触及 `FLT_MAX/2`。审查时同时重新核对了异常分类的其它边界：

- 初版只排除 real signed-zero，imaginary axis 和 subnormal 仍可进入向量公式；若 FP 环境启用 FTZ/DAZ，它们无法保证与 scalar oracle 一致。
- `cpow` 初版只把底数的轴/subnormal 排除，指数的 signed-zero/subnormal 仍进入 `log/atan2` 融合式；这违背了“轴和异常值由 oracle 定义”的候选边界。

这些不是 vendor 缺陷：是 HYLE candidate 的 regular-lane 判定不完整，必须先修候选而不是放宽测试或将问题归咎于 xsimd。

## 3. 决定性代码路径

初版的 all-large fast path只检查 `x` 与 `y` 是否有限，然后执行：

```cpp
const auto large_sc = xsimd::sincos(two * y);
```

`two * y` 对大于 `FLT_MAX/2` 的有限 `y` 已非有限；而该 fast path `continue`，不会执行稍后逐 lane `std::complex` 覆盖。混合 batch 的普通路径也有同一漏洞：大实部 lane 被无条件认定 regular。

这是一条确定的输入→中间溢出→跳过 oracle 路径，不依赖 benchmark 噪声或某个 ISA 的偶然代码生成。

## 4. 修法

`complex_ctanh_regular_lane` 与其 batch gate 现在共同要求：

1. real 和 imaginary 都是 normal finite（轴与 subnormal 回退）；
2. `abs(imag) <= FLT_MAX/2`，保证 `2*imag` 仍有限；
3. 小实部的 denominator 距极点大于 `0.01`；混合 block 的大实部 lane 才进入 `q = exp(-2|x|)` 渐近式，整块安全 large-real 则直接写出极限值。

全-large fast path复用同一 `imag` 容量门，因而不会绕过异常覆盖。`cpow` 的 lane gate 同步要求 base 与 exponent 四个分量都为 normal finite；zero/subnormal exponent 改走 `std::pow`。异常结果均在 vector store 前保存，之后覆盖对应 lane，因此精确原位 alias 保持成立。

## 5. 回归与验证

新增 `Math.ComplexTanhExtremeImaginaryFallsBack`：一个完整 AVX2 block 同时放入 `±FLT_MAX`、`±0.75*FLT_MAX` 和普通 large-real lane，覆盖旧 all-large 路径与 mixed-lane 不连坐。既有特殊值测试补入 imaginary signed-zero、subnormal；`cpow` 特殊值矩阵补入 exponent 的 signed-zero/subnormal。

修复后最新源码验收：

- Windows Math 21/21；Windows SIMD 255 tests，254 passed + 1硬件 skip；
- WSL GCC14 SIMD 255 tests，254 passed + 1硬件 skip；WSL ASan 同计数且零 sanitizer 错误；
- Apple Silicon SIMD 252 tests，251 passed + 1非 x86 skip，NEON 173/173；
- AVX512 SDE：`vec` 173/173，Math/Sort/BitPack 47/47。

## 6. 教训

1. **有限输入不等于所有中间量有限**：由 `2*y`、平方、指数等派生的范围必须出现在 regular-lane gate，不能只检查输入是不是 NaN/Inf。
2. **全-block 快路径也是语义路径**：它不能绕过异常 lane 的 oracle 覆盖；每条 `continue` 都要与一般路径逐项比契约。
3. **随机 benchmark 不是 full-range correctness 证据**：极大有限值、signed-zero、subnormal、轴、branch cut 和近极点应是显式测试维度。
4. **候选先接受更严的 oracle 分类**：罕见 lane 回退可牺牲局部吞吐，但不能静默缩窄公开语义；常规 lane 的性能再用统计 benchmark 裁决。
