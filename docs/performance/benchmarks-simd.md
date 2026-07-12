# simd 实测铁证

> 本页是 [performance 总账](./README.md) 的 simd 详节:dot/saxpy、exp/log/tanh 数学核、fp16/bf16 dtype、reduce_max/scale。环境:WSL Ubuntu24.04 原生 ext4 GCC14.2 + Win MSVC19.50 Release,nanobench,4096 f32(L1 常驻)。`bench/simd/`。

## simd dot / saxpy（vs 标量 auto-vec + 同 ISA SOTA Highway，`simd_bench.cc`）

> **Highway = SOTA 标尺**(静态 dispatch,与 HYLE avx2 同 ISA、同 kernel 形状,只差抽象层)。**数字 = 15 遍进程级复测的中位数 op/s + run-to-run spread(`scripts/bench-stat.py` 驱动,沿 hashmap find-miss 那节的统计纪律)**;判定规则:差距 > 两变体 spread 之和才算赢/输,否则**统计平局**。
>
> **⚠️ 对照公平性(两条承重前置,踩过坑固化)**:
> 1. **对照必须补全 ISA baseline**:Highway `HWY_STATIC_TARGET` 从完整 baseline 宏链(`detect_targets.h`)推断向量宽度——AVX2 需 SSE4 baseline(`__PCLMUL__`/`__AES__`)+ BMI2+FMA+F16C。只给精简的 `-mavx2 -mfma -mf16c` 会让 GCC/Clang 下 Highway 静默降级到 SSSE3(128b,Lanes=4),objdump 见 xmm 无 ymm → 测成 HYLE-256b vs Highway-128b 的宽度差,非抽象层差异。`hyle_set_highway_isa`(`HyleModule.cmake`)给对照 TU 补全 baseline(`-mbmi -mbmi2 -maes -mpclmul`)选回真 256b AVX2。MSVC `/arch:AVX2` 本含完整 baseline。
> 2. **对照必须用库的优化路径**:Highway 的 SOTA dot 是 `contrib/dot::Compute`(4 独立累加器 + 4x 展开,吃满 2 FMA ports × 4-cyc 延迟),不是教学式 `ScalableTag` 单累加器写法。拿单累加器当"SOTA 标尺"会把两边都锁在 FMA 延迟链上得出假平局。

| kernel | scalar(auto-vec SSE2) | HYLE sse4(128b) | HYLE avx2(256b+FMA,4-acc) | Highway contrib/dot(**真 SOTA 4-acc**) | 结论(15 遍中位数) |
|---|---|---|---|---|---|
| **dot product** | 2.7 G op/s (1.0x) | 26 G (9.6x) | **42 G (15x)** | 42 G (15x) | **平真 SOTA**(ratio 0.984,gap 1.6% « spread 14%);4 累加器展开后 12.6→42 G |
| **saxpy** y=a*x+y | 17.4 G (1.0x) | 17.4 G (1.0x) | **39 G (2.2x)** | 40 G (2.3x) | **平 Highway**(element-wise 带宽受限,单累加器已够,未改) |

**当前实现要点**:
1. **dot 主循环 4 独立累加器 + 归约树**(objdump:4 条 `vfmadd231ps` 写 ymm0/1/2/3 + `vaddps` 树)。单累加器会被 FMA 延迟(~4 cyc)绑死;i9 双 FMA 端口下 4 链保持 ~8 FMA in-flight,转为 load 端口绑定。dot 42 G op/s,对拍 Highway 真 `contrib/dot::Compute`(同样 4-acc/4x 展开)统计平局(ratio 0.984,gap 1.6%«spread 14%)。dot 是 AI 热路径主力(注意力/GEMM 内积)。
2. **saxpy/scale 单累加器足够**:`y=a*x+y`、`y=a*x` element-wise 无跨迭代依赖,本就带宽绑定,多累加器无益(saxpy 39 G 平 Highway 40 G)。reduce_max 有归约依赖(理论同 dot 可展开)但**是 bench-only kernel 非公开 op**(`ops.h` 只导出 dot_f32/saxpy_f32),上架为公开 op 时再展开。
> **方法学铁律(承重)**:"平 SOTA" 若对照本身可疑,必追问对照是不是该库的**生产实现**——objdump 确认对照真用了预期宽度(非静默降 128b)且真用了库的优化路径(非教学式单累加器)。否则"平局"会掩盖真实差距(两边都没到硬件上限)。

## simd 数学核（exp/log/tanh vs 标量 libm + 同 ISA SOTA xsimd/Highway，`math_bench.cc`）

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4,各 15 遍中位数+spread**(`scripts/bench-stat.py`),nanobench,4096 f32(L1 常驻)。**两个标尺**:① 标量 `std::exp/log/tanh`(测向量化加速本身);② **同 ISA SOTA = xsimd + Highway contrib/math**(同 `-mavx2`)。HYLE 借 Highway/xsimd 验证过的核(已 vendored、无条件默认编进生产,不手搓多项式)。`active_math_variant()` 报当前层。

| kernel | 标量 libm | HYLE(best-of-breed) | xsimd | Highway | 结论(双平台 15 遍) |
|---|---|---|---|---|---|
| **exp f32**(→xsimd) | 2.06 ns | **0.23** | 0.22 | 0.30 | ✅ 平 xsimd(借其核)、反超 Highway 1.30x(双平台一致) |
| **log f32**(→Highway) | 2.0–2.6 ns | **0.27** | 0.37 | 0.27 | ✅ 平 Highway(借其核)、反超 xsimd 1.45x |
| **tanh f32**(→xsimd) | 6.3–8.7 ns | **0.44** | 0.45 | 0.54 | ✅ 平 xsimd(借其核)、反超 Highway 1.22x |

**核心诚实结论(同 ISA 对标 + 逐 op best-of-breed dispatch,全 op ≥ 两库 SOTA)**:
1. **⚠️ 揪出并修复一个真产品 bug——Highway 数学核在 Linux 出厂 128b**:`src/simd/math/transcendental_highway.cc`(log 借核)此前用 `hyle_set_isa(AVX2)`,缺 Highway 静态 dispatch 需要的完整 baseline(bmi2/aes/pclmul),GCC/Clang 下静默降级 SSSE3/128b(objdump:shipped 核 0 ymm/170 xmm)。**后果:出厂库的 log batch 在 Linux 跑半宽,0.53ns vs 应有 0.27ns,慢两库一半**——这不是文档错,是产品代码 bug(simd 那轮降级编译 bug 的产品侧遗漏,当时只修了 bench 对照)。改 `hyle_set_highway_isa` 补全 baseline 后:产品核 0→141 ymm,**Linux log 0.53→0.27ns,追平 Highway 核**。MSVC `/arch:AVX2` 自带完整 baseline,故 Win 一直正常(0.27)——是 Linux/GCC 特有出厂 bug。**这正是双平台审查的价值:文档"log 反超两库"是 Win 真相,却掩盖了 Linux 产品库半速。**
2. **"平 vs 反超"措辞校正**:HYLE 的 best-of-breed 是**借**对应库的核(exp/tanh 借 xsimd、log 借 Highway),所以对被借的库本质是**平局(同核)**,只对另一库反超。旧文档"log 反超两库"不确——借 Highway 核不可能反超 Highway,实测 log 0.27≈Highway 0.27 是平。诚实表述:**exp/tanh 平 xsimd + 反超 Highway;log 平 Highway + 反超 xsimd。每 op 都 ≥ 两库,但"≥"里对被借库是"平"非"超"。**
3. **逐 op best-of-breed dispatch**:同 ISA 对标证实没有单一库三 op 全胜(xsimd 的 exp/tanh 快、Highway 的 log 快),dispatcher 按 op 选最快核(`math_dispatch.cc`,编译期选,零运行时开销)。
4. **借核现状**:Highway + xsimd 均 vendored、无条件编进生产、默认全速,运行期 `cpu_features().avx2` 门控老 CPU 落 scalar 不 SIGILL。math_test 5/5、full simd 56/56。(注:SLEEF 已从项目移除,dispatcher 三 op 都没选它过——bench 顶部若仍有 SLEEF 注释属陈旧。)

### `hypot_f32` 双契约：safe 与 fast 由用户显式选择

xsimd 的 direct-formula `sqrt(a*a+b*b)` 吞吐约为 Highway exponent-scaling `Hypot` 的2倍，但不保护中间平方溢出/下溢。HYLE不再替用户二选一：`hypot_f32_batch` 保留完整 `std::hypot` 语义；`hypot_in_range_f32_batch` 显式采用受限输入域契约，借 xsimd AVX2/AVX512/NEON 核，scalar tail 同样计算 direct formula，热路径无逐 lane 检查。

| 真机 | safe | fast | `fast/safe` 判定 |
|---|---:|---:|---|
| WSL GCC14 AVX2 | 0.281 ns/op | **0.140 ns/op** | **2.03× WIN** (`bench-stat -n15`) |
| Apple Silicon NEON | 0.454 ns/op | **0.220 ns/op** | **2.07× WIN** (`bench-stat -n15`) |

f64 direct-formula 试验为0.554 vs safe 0.593 ns/op，6.7% gap小于24.7%合并spread，属于统计打平；没有收益就不增加受限API，故仅保留 `hypot_f64_batch`。输入域选择表见 `docs/guide/usage-guide.md` §2.2。

## simd f64 数学核（全 14 超越 + hypot,double 精度——CAE/求解器/几何路径）

> f32 math 补全后**铺全 f64 批 API**(exp/log/tanh + sin/cos/exp2/log2/erf/atan/cbrt/expm1/log1p/asinh/acosh + tgamma/lgamma + hypot + dot/saxpy)。CAE 求解器/几何核大量用 double。vec.h 后端层本就全 f64,缺口只在公开批 API + 借核层。**winner 逐精度 bench(WSL GCC AVX2,4-wide f64,nanobench 20k epoch),不从 f32 外推**。

| kernel | xsimd f64 | Highway f64 | winner | 对比 f32 winner |
|---|---|---|---|---|
| **exp f64** | 1.83 ns | **1.01 ns** | **Highway 1.8x** | f32 是 xsimd(**翻转**) |
| **log f64** | 1.72 ns | **0.89 ns** | **Highway 1.9x** | f32 也是 Highway |
| **tanh f64** | 3.17 ns | **1.44 ns** | **Highway 2.2x** | f32 是 xsimd(**翻转**) |
| **sin f64** | 1.27 | **0.86** | **Highway 1.5x** | f32 Highway |
| **cos f64** | 1.29 | **0.89** | **Highway 1.5x** | f32 Highway |
| **exp2 f64** | 1.81 | **0.87** | **Highway 2.1x** | f32 是 xsimd(**翻转**) |
| **log2 f64** | 1.86 | **0.93** | **Highway 2.0x** | f32 Highway |
| **erf f64** | **2.71** | 3.52 | **xsimd 1.3x** | f32 也是 xsimd(唯一 f64 xsimd 赢) |
| **atan f64** | 1.26 | **1.17** | **Highway 1.07x** | f32 Highway |
| **cbrt f64** | 5.88 | **2.33** | **Highway 2.5x** | f32 Highway |
| **expm1 f64** | 2.38 | **1.10** | **Highway 2.2x** | f32 Highway |
| **log1p f64** | 1.84 | **1.11** | **Highway 1.7x** | f32 Highway |
| **asinh f64** | 3.88 | **2.69** | **Highway 1.4x** | f32 Highway |
| **acosh f64** | 3.03 | **2.40** | **Highway 1.3x** | f32 Highway |

**核心方法学结论**:**f64 下 Highway 赢 13/14,xsimd 只赢 erf——而 f32 下 xsimd 赢 exp/tanh/exp2,winner 在 exp/tanh/exp2 上翻转**。坐实"best-of winner 必须逐精度 bench,不可从 f32 dispatch 表外推":若沿用 f32 表给 f64 exp/tanh/exp2 选 xsimd,会慢 ~1.8-2.2x。故 f64 dispatch 除 erf(xsimd)全走 Highway;tgamma xsimd 独有;lgamma scalar(vendor 向量化 lgamma lane-mixing bug,f64 假设同样受影响);hypot Highway(overflow-safe)。
**全后端铺完(非 follow-on)**:avx2(4-wide)+ **avx512(8-wide,GCC/Clang,SDE 真跑 7/7;MSVC-AVX3-broken 走 avx2)** + **neon(2-wide,qemu-aarch64 真跑 14 op vs libm ~1e-13)** + scalar floor。正确性:`Math.F64/F64Rest/HypotF64MatchesLibm` 对拍 long double libm ~1e-14 相对——四平台全绿(MSVC 232 + WSL GCC 232 + AVX512 SDE + NEON qemu)。dot_f64/saxpy_f64 亦全五后端。



> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4,各 15 遍中位数+spread**(`scripts/bench-stat.py`),nanobench,4096。三组:HYLE F16C/AVX2 硬件 dispatch / HYLE scalar 参考 / Highway(gated SOTA,baseline 齐已验 488 ymem、真 256 位未被降级编译)。`active_dtype_variant()` 报 `AVX2/F16C`。std 无 fp16/bf16 批量等价物。**dtype 是平台高度依赖项**:判定按双平台分列。

| 转换 | HYLE | vs scalar | vs Highway(SOTA) | 结论 |
|---|---|---|---|---|
| **f16→f32** | 0.025 ns | Win 26x / Lin 19x | **Win 1.88x / Lin 平局** | ✅Win 赢 Highway 1.88x、Linux 统计平局 |
| **f32→f16** | 0.030 ns | Win 80x / Lin 25x | **Win 1.61x / Lin 1.90x** | ✅**双平台都赢 Highway**(唯一双平台稳赢) |
| **bf16→f32** | 0.030/0.040 ns | Win 3.5x / Lin 平局 | **Win 1.87x / Lin 平局** | ✅Win 赢 Highway、Linux 平 |
| **f32→bf16** | 0.095(Lin)/0.098(Win) | Win 7x / Lin 1.9x | 平真 SOTA(OrderedDemote2To) | ➖平真 SOTA(2-input 生产形状,详见 ▼) |

**结论(双平台据实,Highway 对照 baseline 完整、真 256 位未被降级编译)**:
- **f16/bf16 F16C/AVX2 路径真兑现**——比 scalar Win 上 3.5–80x、Linux 上 1.8–25x(fp16 两平台都大赢 scalar,"fp16 dtype(F16C)"声明坐实)。
- **vs Highway 是平台相关,不是无条件赢**:**Win 上 HYLE 赢 Highway 1.6–1.9x**(f16→f32/f32→f16/bf16→f32 三项硬赢)——MSVC 把 Highway 的 PromoteTo/DemoteTo 编得不如 HYLE 直接用 `_mm256_cvtph_ps`/`cvtps_ph`(F16C 单指令)、`cvtepu16+slli`(bf16)紧凑;**Linux/GCC 上多数统计平局**——GCC 把 Highway 编得和 HYLE 一样好(亚-ns 量级 + Highway 那侧 spread 27–48% 噪声大)。**唯 f32→f16 双平台都赢 Highway**(Win 1.61x/Lin 1.90x,F16C 单指令 DemoteTo 的结构优势两平台都在)。诚实表述:**dtype 在 Win 赢 SOTA、在 Linux 平 SOTA,无虚假"无条件碾压"**。
- **bf16 Highway 对照已补(原"待补"已填)+ f32→bf16 改生产形状(2026-07 红队二轮)**:bf16<->f32 是纯整数位运算(bf16 = f32 高 16 位),Highway 同样用 AVX2 整数做。**f32→bf16 揪出与 FMA 案同构的漏接**:HYLE 旧核每 8 元素 `packus+permute4x64`(跨 lane 3c)+ **128b 半宽存**,正是 Highway **便利版 `DemoteTo`(8→8)** 的形状;Highway **生产版 `OrderedDemote2To`** 吃 2 个 f32 向量→1 个满 bf16 向量,每 16 元素只 1 packus+1 vpermq+**满 256b 存**,把跨 lane shuffle 和存端口摊到 2 倍元素。改用 2-input 生产形状(RNE 位语义不变,仍 bit-identical scalar oracle):**Linux 0.104→0.095 ~1.09x,双平台平真 SOTA `OrderedDemote2To`(Lin 0.095 vs 0.097 ratio 1.02、Win 0.098 vs 0.099),反超便利版 `DemoteTo`(0.107/0.117)**。收益比 FMA 案小(~1.09x 非 3x)因 bf16 demote 是 store-bound 非 compute-bound——摊销 shuffle 有限,存带宽才是墙。**bench 标尺同步升级**:旧标尺用便利版 `DemoteTo`(同 HYLE 旧形状=两个朴素核假平),加 `OrderedDemote2To` 真 SOTA 行。
- **bf16 vs scalar 也平台相关**:Win 赢 3.5x(bf16→f32)/6.7x(f32→bf16),但 **Linux 上 bf16→f32 与 scalar 统计平局**——GCC 把纯移位的 scalar 循环自动向量化到与 AVX2 核同速(MSVC 没有)。这不是 HYLE 退化,是 GCC 自动向量化够强;f32→bf16 因 RNE 比较复杂 GCC 没完全向量化,HYLE 仍赢 1.8x。**bf16 正确性**:纯位运算,AVX2 与 scalar bit-identical(NaN payload 也是),0x10000 全模式 + f32 宽扫 bit-exact 测试验证。

## simd 逐 op 扩展（reduce_max / scale，`simd_bench.cc`）

> dot/saxpy 之外补两个 distinct pattern,验证 vec.h 抽象在非 FMA 模式下是否退化。环境同上(15 遍中位数+spread,Highway 真 256b 对照——见上节公平性更正)。**记账口径**:reduce_max/scale 是 **bench-only kernel(`simd_kernels_*.cc`),不是 `ops.h` 导出的公开 op**(公开只有 dot_f32/saxpy_f32)——此处纯为验证 vec.h 抽象在 max-归约/element-wise map 上零成本,非"已交付的 SIMD 能力"。若将来上架为公开 op,reduce_max 有归约依赖可同 dot 做多累加器展开(现为单累加器)。

| op | scalar(auto-vec) | HYLE sse4 | HYLE avx2 | Highway(真 256b SOTA) | 结论(15 遍中位数) |
|---|---|---|---|---|---|
| **reduce_max**(非 sum 归约) | 0.73 ns | 0.17(4.3x) | **0.077(9.5x)** | 0.077 | **平 Highway**(ratio 1.00,gap 0.2% « spread 16%) |
| **scale** y=a*x(纯乘) | 0.044 ns | 0.043 | **0.025(1.8x)** | 0.025 | **平 Highway**(ratio 1.01,带宽受限) |

**结论(干净,无欺骗)**:**4 个 pattern(dot/saxpy/reduce_max/scale)HYLE avx2 全与 Highway 统计平局**——vec.h 抽象在 sum-归约之外的 max-归约、纯 element-wise map 上同样零成本,无隐藏退化。reduce_max 9.5x 赢 scalar;scale 是 1 读 1 乘 1 写的内存受限 op,scalar auto-vec(SSE2)已近饱和带宽,故 avx2 仅 ~1.8x(正常,非缺陷)。sse4 在 scale/saxpy 这类带宽受限 op ~1x(128b 与 scalar SSE2 同宽,无更宽 load 优势)。**注**:此前 reduce_max/scale 曾对 128 位瘸腿 Highway 显出 ~2x 假性领先(同上节根因);修正对照后归于平局——结论方向不变,但数字过程已诚实化。

## gather/compress 原生 intrinsic vs store+reload（`gather_bench.cc`，simd-09 #66/#69）

> "measure don't assume" 的落地:simd-08 的 gather/scatter/compress/expand 是 store+reload 仿真(能力优先)。simd-09 给常见宽度加原生路径(avx2 vpgatherd / avx512 vpcompress-expand),但**原生不是无条件更快**——vpgather 在老 uarch 臭名昭著地慢。这个 bench 在同 ISA 下把原生 vs `GatherIndexEmulated`/`CompressEmulated`(store+reload 参照)对拍,数据决定原生该不该上。native-relative-to-emulated:>100% 原生赢,<100% 原生输。

| op | AVX2 裸机(本机 Zen,i32 8-lane) | AVX512 SDE(i32 16-lane,**仅正确性,性能不可信**) | 处置 |
|---|---|---|---|
| **GatherIndex** | 原生 **1.31x 赢**(0.13 vs 0.17 ns) | 原生 0.30x(SDE 伪影) | **保留原生**(AVX2 有真实数据支撑) |
| **Compress** | ~持平(0.99x;AVX2 无 VBMI2,本就 store+reload) | 原生 0.50x(SDE 伪影) | AVX512 保留原生,**性能待真机** |

**诚实结论**:
- **AVX2 GatherIndex 原生 vpgatherd 本机实测赢 1.31x**——真实数据,该上,已上。
- **AVX2 Compress 无原生**(VBMI2 vpcompressb/w 非 AVX2,是 AVX-512),两行相同是预期(CompressEmulated 别名 Compress)。
- **AVX512 数不可信,只信正确性**:Intel SDE 是**功能模拟器**,对 vpgather/vpcompress 这类复杂指令的模拟开销远高于简单 load/store,故 SDE 上"原生慢 2-3x"是**模拟器伪影,不是真硬件时序**。真 AVX-512 硬件有专用 compress 单元,vpcompressd/vpexpandd 是公认快的。**avx512 原生路径正确性已验(SDE + 现有 Gather/Compress 测试全绿),性能待真 AVX-512 机复测**——不拿 SDE 性能数当结论,也不因 SDE 慢就撤掉真硬件会赢的原生路径。
- **方法学**:这条印证 best-of 判据里"原生不必更快、必须测"——AVX2 gather 测出真赢(上),AVX512 因无真硬件只能验正确性(原生留、性能挂真机)。SDE 验正确性可信、验性能不可信,是本轮明确的一条工具边界。

## op-parity: HYLE 自造 op vs Highway vs xsimd 同 op 同 ISA(`op_parity_bench.cc`)

> **覆盖率(2026-07-12 收口)**:op-parity 四文件(full/parity/widen/crypto)共 **281 bench 行**,覆盖 **245/255 真算子 = 96%**(分母=公开 op 面剔除 tag/trait/构造 helper + alias/纯直通-单指令-block变体等合理跳过后的真算子集)。**双平台真编真跑**:WSL GCC14 avx2 + Win MSVC /arch:AVX2 均编过跑通(full 190 行 / parity 61 / widen 22 / crypto 8)。剩 10 未覆盖全部有据:`DivIsPow2`(constexpr bool 非向量 op)、`Bf16EvenToF32`/`Bf16OddToF32`/`MulEvenAdd`/`MulOddAdd`/`U8FromU32`(bf16/dtype 域,dtype_bench 覆盖)、`CompressBits`/`CompressBitsStore`(预打包 byte-mask 指针,需特殊 harness)、`CompressBlocksNot`(block 变体)、`IfNegativeThenNegOrUndefIfZero`(Undef 语义难对拍)。
>
> 回答"HYLE 手写了 ~228 op/后端(抄 Highway 的 per-op intrinsic 序列,不 vendor Highway ops/),
> 这些自造件是否真不亏"。**同 TU 同 -march 两边都 header-only 可内联**(避内联不对称假差,
> feedback_benchmark_inlining_asymmetry);`.raw` 就是裸 `__m256`,HYLE/Highway/xsimd 三方桥接零成本。
> 判据:**看中位 ns/op 的稳定倍数**(亚纳秒 kernel 的 ratio-spread 易被调度离群值污染成假 TIE,
> 中位 ns/op 更可信)。WSL GCC14 AVX2 裸机,4096 f32/i32,`bench-stat.py -n15/31`。

### 结论表(med ns/op,越小越快)
| op | HYLE | Highway | xsimd | 判定 | 处置 |
|---|---|---|---|---|---|
| Add f32 | 0.082 | 0.081 | 0.081 | 三方打平 | 保留自造(=公开 add_f32 batch 底层核) |
| MulAdd f32 | 0.149 | 0.148 | — | 打平 | 保留 |
| Min i32 | 0.076 | 0.076 | — | 打平 | 保留 |
| ReduceSum f32 | 0.082 | 0.084 | 0.108 | HYLE/Highway 平,均胜 xsimd | 保留 |
| Reverse i32 | 0.050 | 0.050 | — | 打平 | 保留 |
| GatherIndex i32 | 0.182 | 0.183 | — | 打平(HYLE 已 native vpgatherd) | 保留 |
| SaturatedAdd i16 | 0.039 | 0.039 | — | 打平(都 native vpaddsw) | 保留 |
| TableLookupBytes u8 | 0.020 | 0.020 | — | 打平(都 native vpshufb) | 保留 |
| InterleaveLower i32 | 0.094 | 0.093 | 0.090 | 三方打平 | 保留 |
| ReverseLaneBytes u32 | 0.042 | 0.043 | — | 打平 | 保留 |
| PromoteLowerTo i16→i32 | 0.022 | 0.022 | — | 打平 | 保留 |
| **Expand i32** | **0.68** | 1.29 | — | **HYLE 赢 1.9x** | 保留(自造反超) |
| **Compress i32** | 0.66 | **0.11** | 0.74 | **HYLE 输 6x**(xsimd 也慢) | **抄 Highway** |
| **PopulationCount u32** | 0.52 | **0.10** | — | **HYLE 输 5x** | **抄 Highway** |
| **LeadingZeroCount u32** | 0.37 | **0.066** | — | **HYLE 输 5.5x** | **抄 Highway** |
| **ConcatEven i32** | 0.14 | **0.076** | — | **HYLE 输 1.8x** | **抄 Highway** |

### 定论
- **passthrough 单指令(~11 测,全打平)**:自造零损失——抄的就是同一条 intrinsic,包裹模板编译期全内联。保留(换来 HYLE dtype/D-tag/scalable 契约统一,桥接零成本)。
- **1 个自造反超(Expand 1.9x)**:HYLE store+reload 比 Highway 的 Expand 快,保留。
- **xsimd 结构 op 面窄**:仅 compress/reduce_add/zip_lo 可比;数学核才是 xsimd 主场(见上文 best-of 表)。
- **方法学**:亚纳秒 kernel 单次 % 是噪声(feedback_benchmark_inlining_asymmetry);必须 `-n15` 看中位倍数。曾单次误读"HYLE 102%"当微胜,15 次后真相是统计打平(min<1<max 互有胜负)。

### 全量扫补充(`op_parity_full_bench.cc`,23 clean-shape op)——又揪出 4 个大输家
抽样漏掉的,全量扫补上(宏生成 unary/binary/compare 批量,med ns/op):
| op | HYLE | Highway | 判定 |
|---|---|---|---|
| **DupEven / DupOdd i32** | 0.55 | **0.054** | **输 10x**(store+reload vs vpshufd) |
| **ReverseBits i32** | 1.23 | **0.139** | **输 9x**(标量 vs vpshufb-LUT) |
| **TrailingZeroCount i32** | 0.56 | **0.071** | **输 8x** |
| **ConcatOdd i32** | 0.136 | **0.080** | **输 1.7x**(同 ConcatEven 族) |
| Abs/Neg/Sub/Mul/Max/AbsDiff/OddEven/InterleaveUpper/Shl/ShiftLeft/ShiftRight/BroadcastSignBit/Eq/Lt/Gt/Sqrt/ApproxRecip×2 (~18) | ≈ | ≈ | 打平,保留 |

### 完整"该抄"清单(9 个自造 op 稳定慢 1.7-30x,全 store+reload/标量 vs Highway 向量指令)
**GetExponent**(30x!std::logb 标量)、**DupEven / DupOdd**(10x)、**ReverseBits**(9x)、**TrailingZeroCount**(8x)、**Compress / PopulationCount**(5-6x)、**LeadingZeroCount**(5.5x)、**ConcatEven / ConcatOdd**(1.7-1.8x)。(**MulByPow2** 同族 std::ldexp,几乎必输,归入。)共同根因:HYLE 栈 store+reload+标量循环/std:: 数学,Highway 用位操作/vpshufb-LUT/vpshufd/vpermd 纯向量。

**✅ avx2 后端已全修(2026-07-09,494a4b0)**——换原生 intrinsic,vec_test 169/169 + 隔离 bench 复测:
| op | 修前 | 修后 | vs Highway | 手法 |
|---|---|---|---|---|
| GetExponent | 30x | 0.052 | 打平 | 位提取指数域+cvtepi32_ps |
| DupEven/DupOdd | 10x | 0.059 | 打平 | moveldup/movehdup/vpshufd/permute_pd |
| ReverseBits | 9x | 0.103 | **反超** | 双 vpshufb nibble-LUT + 字节序反转 |
| PopulationCount | 5x | 0.090 | **反超** | vpshufb 半字节 LUT + fold |
| TrailingZeroCount | 8x | 0.112 | 1.4x(近) | PopCount((x&-x)-1) |
| LeadingZeroCount | 5.5x | 0.204 | 2.4x | bits-PopCount(smear-right) |
| Compress | 6x | 0.283 | 2.4x | vpermd 索引表+零尾 |
| ConcatEven/Odd | 1.7x | (原生) | ~平 | vpermd 偶奇收集+blend |

3 打平/反超,4 大幅改善(LZCnt/Compress/TZCnt Highway 的 float-trick/packed-table 更精巧仍略快,但已拿下绝大部分收益)。铁律仍立:store+reload+std 标量 op 优先全换。

**✅ 剩余 2.x 输家也拉到全平(2026-07-09,用户"返回全平,而不是比 highway 慢 2.x")**——上表 LZCnt/TZCnt/Compress 的"2.4x"残留全部消除,抄 Highway 的更精巧算法:
| op | 上一轮 | 本轮 | 手法(抄 Highway generic_ops / x86_128/256) | 统计判定(bench-stat -n15) |
|---|---|---|---|---|
| **LeadingZeroCount** | 2.4x | **打平** | float-exp trick:`cvtepi32_ps` 取偏置指数,LZCnt=min(158-be,32)。带 round-down 归一化 `v&~(v>>24)` 防跨 2 幂舍入 | avx2 ratio 1.13、sse4 1.09(gap«spread) |
| **TrailingZeroCount** | 1.4x | **打平** | 同上,TZCnt=min(be(x&-x)-127,32),unsigned-min 吸收 v=0 | (随 LZCnt 同核) |
| **Compress** | 2.4x(avx2)/5x(sse4) | **打平** | ①去零尾税(契约改 unspecified 尾部,对齐 Highway,见下)②avx2:1KB packed 表 + srlv 展开、**movemask_ps 直取 mask**(原 mask_bits 标量压位循环是主瓶颈)③sse4:256 B pshufb 字节索引表 | avx2 ratio 0.99、sse4 1.00 |

**Compress 契约变更(关键决策)**：HYLE 旧 Compress 保证不活跃尾部清零（强于 Highway `unspecified`）。实测该保证在 AVX2/SSE4（无清零-compress 指令）是纯软件税。经核实消费者只读取前 `CountTrue` 个 lane，故契约对齐 Highway/xsimd 为尾部 unspecified。各 target 尾部可不同。NEON 后续真机 benchmark 证明 store+标量循环+reload 并非“已够”：i32 Compress 慢 2.8×、Expand 慢 1.9×；现已改成 Highway 同类的 mask→byte-index table + `vqtbl1q_u8`，消除回退。

**固定宽后端当前性能证据**：三平台 `-n15` op-parity 全账（见下"2026-07-12 三平台定论总账"）显示 **95–97% 统计打平**，剩余显著输家已逐项归因——多数是冷路径 mask 往返、SSE4 无变量移位 vpsllvd 的纯 ISA 墙、或 AVX2 缺 srlv_epi16 的硬件墙；可用 native 指令消除的 store+reload 输家（TruncateTo 三平台、NEON RoundingShr）均已换原生指令追平。NEON 主 sweep 曾揪出并修复 Compress/Expand 回退，多 lane-width 复测又暴露并修复 u8/i16/i64 回退；P3 性能闸又揪出并修复 complex 组合式回退。AVX512 的 Intel SDE 只证明正确性，不能证明性能，因此不把它计入“统计打平”。HYLE 保留 dtype/scalable/契约统一；性能结论只覆盖有真机 benchmark 的 ISA/op。

**✅ 全四固定宽后端实现已铺完**——同 pattern 铺到 avx512/sse4/neon，各 vec_test 173/173；性能证据按下表区分真机与模拟器：
| 后端 | 验证闸门 | 手法要点 | 性能结论 |
|---|---|---|---|
| **avx2**(256b) | WSL+MSVC 173/173 + 隔离 bench | 见上表 | 255 pair 中 241 平(94%);11 显著输家已归因(冷路径/内存harness/契约/硬件墙;TruncateTo 已修追平),见定论总账 |
| **avx512**(512b) | Intel SDE 173/173 | 同 avx2 512-bit 化;MulByPow2 额外走原生 vscalef(512 有硬件)、GetExponent 用 vgetexp、Compress 用 maskz_compress | 无真机数据，不作性能结论 |
| **sse4**(128b) | WSL+MSVC 173/173 + op_parity sse4 bench | ConcatEven/Odd 用 shuffle_ps 双源选 lane;余同 avx2 单块 | 215 pair **全部统计打平(100%)**;原变量移位/TruncateTo/Set*First 输家全部向量化追平,见定论总账 |
| **neon** | Apple Silicon 173/173 + 主/full(190行)/widen/Q15/compact/P3 `bench-stat -n15` | PopCount=`vcntq_u8+vpaddlq`、LZCnt=`vclzq`、ReverseBits=`vrbitq_u8+vrevNq`、Concat=`vuzp1/2q`；Compress/Expand 按lane宽分档；complex用FMA友好直接公式 | RoundingShr/Set*First/CountTrue 已全部向量化追平；完整 bench 剩 5 项逐项隔离定性=4 串扰(隔离TIE)+1 真输 MulByFloorPow2(已改 vcvtmq 一条替 Floor+clamp+ConvertTo 追平)→真实算法层 Highway 显著更快 0，见定论总账 |

**MulByPow2 分档**:仅 avx512 有原生 vscalef,故 avx512 换原生;avx2/sse4/neon 无 scale 指令,保留 std::ldexp store+reload(与 Highway 同,非输家)。**SVE/RVV 无需改**:SVE 早用原生 svcnt/svclz/svuzp1/svuzp2;RVV 的 popcount/clz 属 Zvbb 扩展(rv64gcv 基础集无,已登记边界,同 Highway gate)。**定论:9 个 store+reload/std 标量慢 op 在全部固定宽后端(4)已换成原生或与 Highway 同类算法；scalable(2)本就原生或按扩展 gate。** 后续 `-n15` 全面对拍又在更大 op 面上揪出新一批显著项（见"三平台定论总账"）：多数是冷路径 mask 往返或 SSE4 无变量移位的纯 ISA 墙；可用 native 指令消除的 store+reload 输家（TruncateTo 三平台、NEON RoundingShr）均已换原生指令追平。

### 变宽 op 对照(`op_parity_widen_bench.cc`)——原 7 行全打平 + 2026-07-12 加宽 MAC 扩至 16 行
之前把变宽/SumsOf 标"疑似 store+reload 输家",实测(med ns/op)**原 7 行全打平**——主路径走原生指令,`alignas` 只在 fallback 分支:
| op | HYLE | Highway | 判定 |
|---|---|---|---|
| PromoteUpperTo i16→i32 | 0.024 | 0.024 | 打平(cvtep) |
| DemoteTo i32→i16 | 0.041 | 0.048 | HYLE 略快(packs) |
| ReorderDemote2To i32→i16 | 0.041 | 0.041 | 打平 |
| WidenMulPairwiseAdd i16→i32 | 0.036 | 0.036 | 打平(vpmaddwd) |
| SumsOf8 u8→u64 | 0.014 | 0.014 | 打平(vpsadbw) |

**2026-07-12 扩展**（WSL GCC14 AVX2 冒烟，待 `-n15`）：ReorderWidenMulAccumulate / SatWidenMulPairwiseAdd / WidenMulAccumulate / SatWidenMulPairwiseAccumulate / SumsOfAdjQuadAbsDiff / OrderedDemote2To / ZeroExtendVector **平**；**SumOfMulQuadAccumulate 输 ~10x**（HYLE store+reload vs Highway 紧循环/VNNI）；SatWidenMulAccumFixedPoint 初测失真（harness 栈拷贝，已改半宽 ZeroExtend，待复测）。

### 2026-07-12 三平台 `-n15` 定论总账（统合所有赢/平/输）

> 覆盖补全（crypto + Tier1–3 + 直通单指令 + 低优先 mask/Shuffle/Pairwise/单向量 Promote）后，用 `bench-stat.py -n15` 在三套真机 ISA 上逐 pair 定论（判据：`|ratio_med−1| > spreadA+spreadB` 才判赢/输，否则统计平）。这张表取代此前所有"冒烟单遍待复核"的零散记录。数据源 `_bench_out/{A,S,N}_*`。

| ISA | 真机 | pair 数 | 统计平 | Highway 显著更快 | HYLE 显著更快 |
|---|---|---:|---:|---:|---:|
| **AVX2**(256b) | WSL GCC14.2 | 255 | 242 (95%) | 11 | 2 |
| **SSE4**(128b) | WSL GCC14.2 | 215 | **215 (100%)** | **0** | 0 |
| **NEON**(128b) | Apple Silicon M-系 | 185 | 180→**184** | **0**（真值） | 2 |
| AVX512(512b) | Intel SDE | — | — | — | — |

> 数字为多轮 store+reload/bitfield-往返修复后（见下"已消除的算法输家"）。**SSE4 全部 215 对统计打平**——所有变量移位、TruncateTo、Set*First 输家均已向量化追平。**NEON 完整 bench 表面显示 5 项 Highway 更快，但逐项隔离 `-n15` 定性后：4 项是长 bench 序列串扰（隔离即 TIE）、1 项 MulByFloorPow2 是真输（已改 vcvtmq 追平）→ 真实算法层面 Highway 显著更快 = 0**（完整 bench 的 180/185 是串扰虚低，隔离真值 184/185，剩 1 为 ApproxRSqrt 单指令噪声边缘）。AVX512 只有 SDE 正确性（220 PASS），SDE 时序不可信，不计入性能账。sse4/neon 的 op_parity_bench（gather/mask 内存族）因 Highway 子-AVX2 无实现会编出 `hwy::Abort` 链接失败，那些行的子-AVX2 对照本就无意义（op 是 avx2-specific），故 sse4/neon 只统计 full+widen+crypto。

**已消除的算法输家（多轮修复，全部真机 -n15 追平 + 四环境正确性）**：
| op | 平台 | 原 gap | 根因 | 修法 |
|---|---|---:|---|---|
| TruncateTo u32→u16 | avx2+sse4 | 2.45×/2.06× | store+reload | native PSHUFB byte-gather（avx2 加 VPERMD+压实） |
| RoundingShr i32 | neon | 3.2× | store+reload | `vrshlq`（rounding shift）+ vmax clamp |
| Shl/Shr/MaskedSh* i32 | sse4 | 1.4–2.2× | store+reload（无 vpsllvd） | `v*Pow2` / `MulHigh(v,Pow2(32-b))`（GCC 折叠用内联汇编 cvttps2dq 规避） |
| RoundingShr i32 | sse4 | 1.7× | store+reload | 复用向量化 Shr + Pow2 bias |
| Set{Only,Before,AtOrAfter,AtOrBefore}First | sse4+neon | 2.2–2.57×/1.3–1.5× | mask bitfield 往返 | i64 位技巧 `x\|-x`/`x&-x` + 跨 64 位块符号传播 |
| CountTrue i32 | neon | 1.4× | mask_bits+popcount | `-vaddvq(mask)` 横向求和 |
| SumOfMulQuadAccumulate / PromoteEvenTo·OddTo | 全平台 | ~13× | store+reload gather | 寄存器内 shift/mask 展开 |

**AVX2 — 剩余 Highway 更快 11 项**（全部硬件墙/契约/内存 harness/亚纳秒噪声，非算法可修）：
| op | H/HYLE | 归因 |
|---|---:|---|
| OrderedDemote2MasksTo / PromoteMaskTo | 14.2× / 6.0× | mask bitfield 往返（avx2 是冷路径；sse4/neon 已按 Set*First 同类手法可攻，但 avx2 movemask 往返本就便宜，收益低） |
| LoadNOr / GatherIndexNOr / GatherIndexN | 9.6× / 2.6× / 2.5× | 变长内存 gather/load harness |
| CompressStore / CompressBlendedStore | 3.4× / 3.1× | **契约差异**：HYLE 精确写 p[count..) 不触碰（强于 Highway 弱契约）；AVX2 无 native compress-store |
| MultiRotateRight u64 | 2.9× | **AVX2 硬件墙**：Highway 核心 `srlv_epi16`=AVX512BW+VL，纯 AVX2 无 |
| LoadInterleaved2 f32 / FindKnownLastTrue / TestBit | 1.17–1.35× | 亚纳秒噪声边缘 |

**AVX2 — HYLE 更快 2 项**：LoadExpand i32（2.2×）、SetOnlyFirst i32（1.45×）。

**NEON — 完整 bench 显示的 5 项，逐项隔离 `-n15` 定性（4 串扰 + 1 真输，真输已修）**：完整 `op_parity_full_bench` 里 ConvertTo/Shuffle2301/TrailingZeroCount HYLE 恒 0.125ns、Highway 0.088ns（1.4×），MulByFloorPow2 1.11×、RoundingShiftRightSame 1.14×。**关键纪律：bench 有串扰缺陷，既不能拿它当"Highway 更快"的结论，也不能反过来用"串扰"当借口把没验证的项都判成打平**——逐项抽出单独 bench，用可信数据定性：
- **4 项 = 串扰伪影**（隔离后 TIE）：ConvertTo（0.979）、Shuffle2301（1.000）、TrailingZeroCount（1.005）、RoundingShiftRightSame（0.990）。objdump 佐证隔离循环两侧逐指令相同（`ldr/rev64/str`），换 run 顺序不翻转 → 完整 bench 里前序 op 的上下文串扰（频率/预测器/端口），单独跑回 0.089ns 打平。
- **1 项 = 真输，已修**：**MulByFloorPow2 f32** 隔离后仍 **1.118×**（spread 0.7–3.5% 干净）是真实算法差，非串扰——HYLE 用 `Floor+2×clamp+ConvertTo` 四条，Highway `FloorInt` 一条 `vcvtmq_s32_f32`。改用 vcvtmq（去掉多余 ±4096 clamp，MulByPow2 内部 ±3*bias clamp 兜饱和）→ **ratio 0.998 追平**，wide-exponent/饱和/NaN 特判红队全过。

**NEON — HYLE 更快 2 项**：RoundToInt f32→i32（1.23×）、TruncateTo u32→u16（1.06×）——同属完整 bench 串扰的反向体现（隔离收敛）。

**跨平台一致的输家类别**（历史，现多数已修）：
1. ~~**mask-domain 往返**（Set*First）~~ → **已修**（sse4+neon i64 位技巧向量化追平）；PromoteMaskTo/OrderedDemote2MasksTo 仅 avx2 剩（冷路径 movemask 往返，收益低）。
2. ~~**变量移位依赖**（Shl/Shr/RoundingShr/MaskedSh*）~~ → sse4 已 `v*Pow2` 向量化追平、neon RoundingShr 已 `vrshlq` 追平；**avx2 本就有 vpsllvd 全平**。
3. ~~**TruncateTo store+reload**~~ → **已修**（native PSHUFB，avx2+sse4 追平，neon 本就平）。

**Crypto（AES-NI + PCLMUL baseline）全平**：AES{Round,LastRound,RoundInv,LastRoundInv,InvMixColumns,KeyGenAssist} + CLMul{Lower,Upper} ratio 0.94–1.03。
**widen/MAC 全平**（-n15）：ReorderWidenMulAccum / SatWidenMulPairwiseAdd·Accum / WidenMulAccum / SumsOfAdjQuadAbsDiff / OrderedDemote2To / ZeroExtendVector；**SumOfMulQuadAccumulate 已从 ~13× 追平**（根因 PromoteEvenTo/OddTo store+reload，已寄存器内向量化）。
**Tier3 HYLE-only**：ReciprocalSqrtNewton ≈2.1× 快于 Div(1,Sqrt)；ReciprocalNewton ≈1.1× 快于 Div、≈1.3× 慢于裸 ApproxRecip（付 1 步 Newton，预期）。

### Apple-Silicon NEON 真机复测（2026-07-10，`bench-stat -n15`）

同TU、同Apple clang/NEON、HYLE与Highway均可内联；主sweep + full sweep + widen/Q15 + compact多宽sweep共覆盖71个结构/算术op：

- full sweep加入P3全部24个公开名后为47/47，全部统计打平。Highway没有同名zeroing `MaskedMulComplex/MaskedMulComplexAdd`，这两项用其等价`IfThenElseZero(MulComplex*)`组合对照，其余22项直接对同名API。
- `MaskedMulComplexConjAdd`初测0.249 vs 0.203 ns/op，ratio 0.815，18.5% gap大于3.4%合并spread，是明确回退；将`MulComplex/MulComplexConj`改为`DupEven/DupOdd + MulAdd/Sub + OddEven`直接公式后，完整24项复测中为0.204 vs 0.204，ratio 1.000、合并spread 6.9%，回到统计打平。
- widen/Q15：PromoteUpper、ReorderDemote2、WidenMulPairwiseAdd、Q15 均打平；DemoteTo 为 0.055 vs 0.103 ns/op，HYLE 约 1.87× 更快；SumsOf8 为 0.029 vs 0.033，约 1.15× 更快。
- 主 sweep 首次暴露 i32 `Compress` 0.489 vs 0.177 ns/op（HYLE 慢 2.8×）、`Expand` 0.370 vs 0.196（慢 1.9×）。根因是 HYLE store+标量循环+reload，而 Highway 使用 byte-index table + `vqtbl1q_u8`。
- 第一轮修复后 i32 `Compress` 0.179 vs 0.177 ns/op，统计打平；`Expand` 0.177 vs 0.196，HYLE 约 1.11× 胜。继续逐 lane-width 测出 u8/i16/i64 回退并分档实现：u8 用双 8-byte table lookup，i16 用 base-index table 扩成 byte indices，i64 用 conditional swap。
- 最终 compact sweep：u8 `Compress/Expand` 0.098/0.068 vs 0.103/0.069 ns/op，i16 为 0.089/0.089 vs 0.088/0.089，i64 为 0.310/0.312 vs 0.311/0.313；六项均统计打平。Q15 为 0.084 vs 0.084，`MIN*MIN` 契约修正未形成可判定回退。

**scalable 性能边界**：SVE/RVV 本轮只有 QEMU 多 VL 正确性证据；QEMU 时序不能作为性能数据。获得 SVE/RVV 真机前，不声称 P0-P2 与 Highway 性能打平。

### simd-12 vendor竞争与小数组税（2026-07-11）

三套scaling bench统一用`public/direct/vendor/candidate/scalar`同调用边界、`bench-stat.py -n15`，并支持环境变量筛选N/K/分布/方向。AVX512 public与fixed-AVX2 direct不同tier时主动省略public行，SDE不作性能证据。

- **bit-pack u16 11/13**：正确candidate相对原scalar fallback，Windows AVX2 pack为22.1/22.7x、unpack为32.1/26.8x；WSL 11-pack 23.3x、13-unpack 14.0x；Apple NEON 11-pack 19.8x、13-unpack 16.0x。candidate与错误Highway速度上界统计打平。`n=1` public/direct原慢29.9%，baseline直跑相同tail后打平。
- **SoA csin/ccos最终版**：稳定`expm1`公式 + 逐lane特殊值覆盖后，Windows AVX2 random N=4096分别约1.29/1.28x胜xsimd；tiny-imag、pure-real与xsimd打平；每block一个特殊值时仍约9.0x胜全scalar。WSL GCC对xsimd统计打平；Apple NEON对xsimd统计打平、相对原生产scalar约2.2x。完整语义修正削减了早期不稳定公式的更高数字，文档只保留最终核数据。
- **small-K sort**：整数/KV `partial_sort(k<=16)`的std核在N=16..1M最多3.4x胜Highway，最差KV64 N=1M/K16打平。KV32 N4096/K8：Windows约2.8x、WSL约2.0x、Apple NEON约4.8x胜Highway；升/降序均复测，public/std direct打平。自研有序前缀候选跨类型/重复键不稳定，未上架。

### simd-13：SoA `ctanh/cpow` vendor displacement（N=4096，N=15）

> 对照是同 TU、同 ISA、同 `HYLE_NOINLINE` 边界下的 xsimd direct；`HYLE public`单列可见 dispatcher 税。仅当 candidate/xsimd 的 gap 大于两边 run-to-run spread 之和时标 WIN。`ctanh` 的 near-pole/axis/subnormal/极大虚部及 `cpow` 的 branch-cut/zero/subnormal 路径故意走 scalar `std::complex`，不拿它们与不满足完整 oracle 的 xsimd 时序做替换裁决。

| op / 分布 | Windows AVX2 | WSL AVX2 | Apple NEON | 结论 |
|---|---|---|---|---|
| `ctanh` random | 1.632 vs 2.011 ns，**1.222x WIN** | 1.703 vs 1.687，0.986x TIE | 3.483 vs 3.423，0.985x TIE | 常规吞吐无确认回退；Windows胜 |
| `ctanh` large-real | 0.627 vs 1.970，**3.105x WIN** | 0.604 vs 1.655，**2.663x WIN** | 1.186 vs 3.393，**2.861x WIN** | 全平台胜；整块大实部极限快捷路径成立 |
| `cpow` random | 3.064 vs 4.092，**1.336x WIN** | 3.120 vs 4.001，**1.322x WIN** | 5.637 vs 7.441，**1.320x WIN** | 全平台胜；一次 `log/exp/sincos` 编排成立 |
| `cpow` large-base | 4.860 vs 8.652，**1.765x WIN** | 5.258 vs 8.793，**1.682x WIN** | 6.516 vs 7.793，**1.193x WIN** | 全平台胜 |

Windows/WSL/Apple 的 public/direct 都落在 candidate direct 附近，未见 dispatcher 固定税掩盖算法差。Apple nanobench 的表格列带前导空白且输出流会随平台变化；`bench-stat.py`现解析合并 stdout/stderr 并规范化末列空白，避免把真实基准误报为“无行”。

### 尚未逐个测的边界（风险已定性）
- **纯 passthrough(~34)**:AES/CLMul(单指令)、And/Or/Xor/Not(mask 域)、Le/IfThenElse* 等——同一 intrinsic,抽样已验打平。
- **需内存/mask harness的剩余项**:LoadInterleaved2/3/4+Store(引用参数)、Gather/Scatter/CompressStore/MaskedLoad(内存+mask)、SetAtOr*/FindKnown*(mask域)、TruncateTo/U8FromU32(变宽,与已测Promote/Demote同族=原生cvt)。P3的24个Masked*已逐项补进full sweep；GetExponent已换原生指数域实现，MulByPow2按ISA分档且与Highway同类，不再列为疑似输家。

### bench 覆盖补齐（directive3,2026-07-12）
覆盖审计揪出的零-bench 能力面已补，全部进 CI bench 目标:
- **f64 全 14 超越进 `math_bench`**:最高风险(f64 winner 与 f32 大面积翻转但无留档 bench)。HYLE 借核 ≥ xsimd/Highway 每 op → dispatch 选核正确,现有回归 bench。
- **f32 11 个 simd-09 op 接上 HYLE 行**(曾是 nullptr 未接线候选):sin/cos/exp2/log2/erf/atan/cbrt/expm1/log1p/asinh/acosh,回归 dispatch winner。
- **`activation_bench`(新)**:gelu 11.9x / gelu_tanh 18.5x / sigmoid·silu 6.2x / softmax 1.6x vs scalar;RNG VectorXoshiro 2.7x vs mt19937。**⚠️坑**:inline VectorXoshiro 吃 bench TU 的 vec.h 后端,不定义 `HYLE_SIMD_TARGET_AVX2` 会退 128-bit 后端反比 mt19937 慢(同 inlining-asymmetry 类,CMake 已 pin)。
- **`simd_bench` 补 dot_f64/saxpy_f64**:3.9x/2.4x vs scalar(CAE/求解器 double 热路径)。
- **`sort_bench` 补 select/nth_element**:HYLE 借 VQSelectStatic 6.4x 快过 std::nth_element。
- **`dtype_bench` 补 fp8 e4m3/e5m2↔f32**:**诚实发现 fp8 batch ≈ scalar loop(~100% 打平)**——分支型 per-element 位逻辑不自动向量化,当前无 SIMD 收益(印证"无硬件 fp8 convert";未来可 SIMD 化是优化机会)。
- **bitpack u8/u16/u32 public-API bench**(3baeedd):**纠正**——之前误标"登记不补 u8/u32",实际生产码 + round-trip 测试早全有(bitpack_body.inc 按 lane 宽选 Pack8/16/32,BitPack.U8/U16/U32 测),只是 bench 是 u16-only。补公开 pack/unpack 的 u8/u16/u32 bench vs scalar bitstream(u8 pack 77x/u32 pack 27x)。
