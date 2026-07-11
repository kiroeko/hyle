# xsimd complex `tanh` 近极点精度 Hunt

> 调查日志（已结案）。触发：HYLE SoA `ctanh` 初版将近极点 lane 回退到 xsimd complex `tanh`，以避免 scalar 慢路径；`Math.ComplexTanhSpecialTailAndAlias` 对 `std::complex` oracle 的对拍立即证伪这条捷径。

**结论先行：** 在 `x=±0.01`、`y≈±(π/2-0.001)`，xsimd complex `tanh` 与 `std::complex<float>` 的误差远超本 API 的相对容差：xsimd 约为 `98.9872 + 9.89927i`，而 oracle 为 `99.0130 + 9.90184i`。该区域正是

```text
d = cosh(2x) + cos(2y)
```

很小、误差被除法放大的极点邻域。HYLE公开路径不能把 xsimd 当该区 oracle；最终实现以 `abs(d) <= 0.01` 为 lane gate，逐 lane 直接调用 `std::tanh(std::complex<float>)`。这条安全路径的局部吞吐输给 xsimd，但完整契约优先于局部 benchmark。

相关实现：[`complex_hyle-inl.h`](../../src/simd/math/complex_hyle-inl.h)。回归：[`math_test.cc`](../../tests/simd/math_test.cc)。

## 1. 现象

复数双曲正切可写为：

```text
tanh(x+iy) = sinh(2x)/d + i*sin(2y)/d
d = cosh(2x) + cos(2y)
```

当 `y` 接近 `π/2` 且 real part很小，`d` 接近零。初版 regular formula本身识别了这个区域，却错误地把异常 lane 交给 xsimd，而不是公共 scalar oracle。测试的近极点向量给出明显差异，普通 random 与 large-real 测试都不会放大到这个程度，因此不能覆盖该漏洞。

## 2. 先怀疑 HYLE fallback，不为 vendor 找借口

最初怀疑是 HYLE denominator 阈值过宽、SoA存取或 alias 覆盖次序有误。为隔离：

1. 对同一输入直接比较 HYLE formula、xsimd direct 和 `std::complex`；
2. 改为无条件 `std::complex` fallback 后，特殊值/alias/tail 对拍恢复；
3. xsimd direct 仍保留原误差。

因此问题不是阈值外的 HYLE vector 公式，也不是输出覆盖；是“用 vendor 当特殊值 oracle”的错误假设。借入核的普通 ULP 表现不能自动延伸到病态输入域。

## 3. 修法

- small denominator lane 不再调用 xsimd，直接构造 `std::complex<float>(real, imag)`并调用`std::tanh`；
- 正常 lane仍向量化，异常结果先存入 lane-local array，vector 写出后仅覆盖异常位置，保持 exact in-place alias；
- 极大有限虚部、axis和subnormal的另一类候选分类漏洞由 [`xsimd-ctanh-extreme-imaginary-hunt.md`](./xsimd-ctanh-extreme-imaginary-hunt.md) 记录并同样修正。

## 4. 性能与正确性的边界

Windows N=4096 near-pole 测量中，scalar-exception HYLE path约 **10.927 ns/op**，xsimd direct约 **1.755 ns/op**，是明确 LOSE。这个数字不是撤回修复的理由：xsimd在该输入的输出不满足 HYLE对`std::complex`的正确性门，不能作为等价替代物。regular random和large-real分布另有统计胜利，才是生产候选的性能裁决对象。

## 5. 验证与教训

`Math.ComplexTanhSpecialTailAndAlias`覆盖正负近极点、与普通/large lane混合、NaN/Inf、tail、未对齐、guard和原位 alias；Windows、WSL、WSL ASan、Apple NEON和AVX512 SDE均通过。

1. **vendor不是特殊值 oracle**：借核也必须按完整输入域对拍。
2. **条件数决定 fallback**：检测输入非有限不够；小 denominator 这种数值病态同样必须进入 gate。
3. **不要用“更快的错误路径”保住 benchmark**：safe/public 路径应保持语义；需要窄契约快入口时才另建显式 API，本例不值得增加一个近极点不正确的变体。
