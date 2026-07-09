# xsimd 向量化 lgamma lane-mixing Hunt(vendor 算法 bug —— HYLE 对拍揪出、三 target 独立坐实、已绕过)

> 调查日志(已结案)。触发:simd-09 借入 xsimd 独有超越函数(lgamma/tgamma)时,新增 `Math.XsimdGammaMatchesLibm` 对拍 libm,lgamma 在 in=1.29875 处偏差 0.83(不是精度级,是完全错值)。
> 关联:[`transcendental_xsimd.cc`](../../src/simd/math/transcendental_xsimd.cc)、[`math_dispatch.cc`](../../src/simd/math/math_dispatch.cc)(lgamma 强制走 scalar)、[`kernels.h`](../../src/simd/math/kernels.h)。上游:`third_party/xsimd/include/xsimd/arch/common/xsimd_common_math.hpp` 的 `lgamma_impl<batch<float>>::other()`(约 1292 行)。
> 结论先行:**xsimd 的向量化 `lgamma`(float)在 lane 值跨越参数归约分支边界时 blend 错**——一个落在 `[1.25,1.5)` 的 lane 若与一个落在 `[0.75,1.25)` 的 lane 同处一个向量,前者拿到错误的归约核结果。独立探针在 **GCC-x86 AVX2 + MSVC-x86 AVX2 + GCC-NEON(qemu-aarch64)** 三处复现同一错值 => 算法 bug 非 codegen。`tgamma` **不受影响**(同窗全 lane 正确)。**处置**:`lgamma` 整个走 scalar libm 参照(不向量化),`tgamma` 继续借 xsimd 向量核。API 两个都正确,只是 lgamma 未向量化,待 xsimd 上游修复再切回。

## 0. 为什么这篇值得独立成文:借库不等于免验(第二次坐实)

这是继 Highway `Pack16<11>/<13>` 之后,HYLE 对拍闸门揪出的**第二个成熟 vendor 数学 bug**。宪章是"借难且已解决的核不手搓",但"借"绝不等于"信"——借来的核**同样过 HYLE 对拍闸门**。这次的闸门是 libm 对拍:xsimd 借核的结果必须逐值匹配 `std::lgamma`(它是 oracle + tail + fallback,永远在场)。正是这个对拍在 in=1.29875 处抓到 0.83 的偏差——不是 3.5-ULP 精度问题,是**完全错值**,一眼是 bug 不是舍入。方法论沉淀再加一条:**vendor 数学核落地必配 libm 逐值对拍,且输入要用 distinct-per-lane 的 ramp——同值填满所有 lane 会掩盖 lane-mixing 类 bug(本例正是被它一度骗过)。**

## 1. 现象:ramp 对拍在孤立点炸,同值填充却全绿

新增测试 `check(lgamma_f32_batch, std::lgamma, ramp(0.05, 50, 600))`。MSVC 首跑:
```
lgamma in=1.29875 i=15: got=0.725872 want=-0.107962 (diff 0.833834)
```
偏差 0.83 远超 tol(~1e-5),且是错值不是精度。孤立、可复现。

## 2. 诚实记录的误诊排除 + 一个差点被骗过的陷阱

借来的成熟库出错先验概率低,第一嫌疑永远是自己;但这次还多一个**被同值探针骗过**的弯路,如实记录:

- **弯路(重要):第一个探针用同值填满 8 lane**(`b[i]=1.29875` 全部),结果 batch==scalar_xsimd==libm **全对**——差点据此误判"HYLE wiring 错"或"MSVC 特有"。**真相是探针没复现测试的 distinct-lane 输入**。教训:复现 SIMD bug 必须忠实复现原始输入形态(ramp 是 distinct-per-lane),同值填充是不同的执行路径(参数归约的 `any()`/`all()` 分支走向不同)。
- **疑① 我的 dispatch/wiring 错**?→ 若 wiring 错,tgamma(同一条 dispatch 形状)也该错,但 tgamma 全绿。且 scalar libm tail 也在同一函数里,不会只错向量段。排除。
- **疑② MSVC codegen bug**(数学核有 `Highway AVX3 MSVC HWY_BROKEN` 先例)?→ 最像的嫌疑,必须坐实排除(见 §3)。

## 3. 决定性实验:distinct-lane 探针,零 HYLE 代码,三 target 复现

写只碰 xsimd、不碰 HYLE 代码的 standalone 探针,忠实复现 ramp 的 distinct-lane 输入:
```cpp
// ramp(0.05,50,600),8-wide 逐块 xsimd::lgamma,对拍 std::lgamma,报 worst
// 关键:相邻 lane 值不同(ramp),不是同值填充
```
三 target 实测,**worst 偏差同为 0.833834 @ in=1.29875**:
```
GCC-x86  AVX2 (-mavx2 -mfma):        batch::size=8  worst=0.833834
MSVC-x86 AVX2 (/arch:AVX2):          batch::size=8  worst=0.833834
GCC-NEON (aarch64, qemu-aarch64):    batch::size=4  worst=0.833834
```
**坐实是 xsimd 算法本身**,非 HYLE、非 codegen:① 探针无一行 HYLE 代码;② 三个独立 target(两编译器 × 两 ISA,含 ARM NEON)复现**同一错值** —— codegen bug 不会跨这么多后端给出比特一致的同一错值,只有共享的算法源会。与数学核那次"MSVC HWY_BROKEN 是 codegen"相反:那次真 codegen,这次真算法,都靠"多 target 复现"区分。

**最小复现锁定触发条件**:lane7=1.29875 固定,填充其余 7 lane:
```
fill=0.716  -> lane7 ok    (-0.107962)
fill=1.0    -> lane7 BAD   ( 0.725872)   <- 同伴 lane 落在 [0.75,1.25)
fill=1.2    -> lane7 BAD   ( 0.725872)
fill=1.29875-> lane7 ok
fill=1.5    -> lane7 ok
fill=2.0    -> lane7 ok
```
=> 触发条件精确:**被测 lane∈[1.25,1.5),且同向量至少一个 lane∈[0.75,1.25)**。同值/同分支填充不触发(这就是 §2 弯路里同值探针漏网的原因)。

## 4. 根因:参数归约多分支的 select blend 覆盖

xsimd `lgamma_impl<float>::other()`(xsimd_common_math.hpp ~1292)按 x 落区间分派不同 Cody-Waite 归约核:
- `[1.5,∞)`:while 循环下拉 + `gammalnB`;
- `[1.25,1.5)`(`xge125t = xge125 && !xge150`):`r0x=x-1; r0s=-1`;
- `[0.75,1.25)`(`xge075t = xge075 && !xge125`):`kernelC=xge075t; r0x=x-1; p=gammalnC(r0x)`。

`p = gammalnC(r0x)` 对**所有 lane** 求值,但最终结果按 `kernelC` 谓词 select。当一部分 lane 在 `[1.25,1.5)`、另一部分在 `[0.75,1.25)` 时,`[1.25,1.5)` lane 的 `r0x`/`r0s` 被正确置了(第二分支),但后续合成把 `gammalnC`(第三分支核)的结果错误地 blend 进了 `[1.25,1.5)` lane —— select 掩码组合覆盖不全。所以**只有 distinct-lane 跨这两个相邻分支**才暴露:同分支填充时 `any()`/select 走向一致,不错位。上游测试八成都用单值或同分支样本,满分支交叉路径漏测发货 —— 与 Highway `& kBits` 漏测满位宽同类:**测试没覆盖到的路径没有正确性保证,哪怕来自成熟库。**

## 5. 修法:lgamma 整体退回 scalar libm oracle,tgamma 保持向量化

- `math_dispatch.cc`:`lgamma_f32_batch` 直接 `detail::lgamma_f32_scalar`(不进任何向量分支);`tgamma_f32_batch` 保持 512->avx2->scalar(x86)/ xsimd-NEON(arm64) 的向量路由。
- 三个 xsimd TU(avx2/avx512/neon)删掉 `lgamma_f32_xsimd_*`,只留 `tgamma_f32_xsimd_*`;kernels.h 同步删声明。
- **没有去"修" xsimd 的 `other()`** —— 借核出 bug 退回自己的 oracle,不改 vendored 源(改了下次更新 xsimd 又冲突);绕过点与原因就地注释登记。lgamma 的 API 仍在、结果仍正确(libm 精确),只是未向量化;xsimd 上游修复后一行 dispatch 即可切回向量核。

## 6. 验证

- HYLE `Math.XsimdGammaMatchesLibm`:lgamma(0.05..50) + tgamma(0.05..10) 对拍 libm 全净;simd 全套 MSVC 219(含新增 gamma 测试)。
- 独立探针(§3 逻辑,~30 行,只 include xsimd.hpp)可随时重建复跑,坐实上游 bug 仍在。

## 7. 教训

1. **借核必过对拍闸门(第二次)**:借不等于信。Highway Pack16 靠 round-trip 恒等式,xsimd lgamma 靠 libm 逐值对拍——两种自洽/对照闸门各揪出一个成熟 vendor bug。
2. **复现 SIMD bug 要忠实复现输入形态**:distinct-per-lane(ramp)≠ 同值填充。同值探针差点让我误判为 wiring/codegen 错——SIMD 的 lane-mixing/归约分支 bug 只在 distinct-lane 下暴露。对拍输入默认用 ramp,不用同值。
3. **多 target 区分 codegen vs 算法**:GCC-x86 + MSVC-x86 + GCC-NEON 三处同一错值 = 算法;只某 ISA 错 = codegen。别一上来甩锅 codegen(数学核那次是 codegen,这次不是,靠这招分开)。
4. **借核出 bug 退回自己的 oracle,不改 vendored 源**:就地注释登记绕过点 + 原因,保留独立复现器;上游修复后一行切回。
5. **能力完整性 ≠ 全部向量化**:lgamma 未向量化但 API 正确在场(scalar libm),这是诚实的"做全"——不是因为难就不提供,也不是明知有 bug 还硬上向量核给错值。

## commit 序
- lgamma/tgamma 借入 + lgamma 绕过 + 本 hunt 文档 + gamma 对拍测试:见 simd-09 xsimd-独有 提交。
