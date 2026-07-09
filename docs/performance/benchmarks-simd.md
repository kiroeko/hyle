# simd 实测铁证

> 本页是 [performance 总账](./README.md) 的 simd 详节:dot/saxpy、exp/log/tanh 数学核、fp16/bf16 dtype、reduce_max/scale。环境:WSL Ubuntu24.04 原生 ext4 GCC14.2 + Win MSVC19.50 Release,nanobench,4096 f32(L1 常驻)。`bench/simd/`。

## simd dot / saxpy（vs 标量 auto-vec + 同 ISA SOTA Highway，`simd_bench.cc`）

> **Highway = SOTA 标尺**(静态 dispatch,与 HYLE avx2 同 ISA、同 kernel 形状,只差抽象层)。**数字 = 15 遍进程级复测的中位数 op/s + run-to-run spread(`scripts/bench-stat.py` 驱动,沿 hashmap find-miss 那节的统计纪律)**;判定规则:差距 > 两变体 spread 之和才算赢/输,否则**统计平局**。
>
> **⚠️ 公平性更正(2026-06-30)**:此前此节标 dot"~1.10x 微超"、saxpy"~1.21x 微超" Highway——**是假象,源于对照被降级编译(对照方生成了低于预期 ISA 宽度的机器码)**。Highway 的 `HWY_STATIC_TARGET` 从完整 baseline 宏链(`detect_targets.h`)推断向量宽度:AVX2 baseline 需要 SSE4 baseline(`__PCLMUL__`+`__AES__`)再加 BMI2+FMA+F16C。bench 给 Highway 对照 TU 的是 HYLE 自家 kernel 的 AVX2 ISA 契约(`-mavx2 -mfma -mf16c`,刻意精简、关联 SIGILL-safety),在 **GCC/Clang 下让 Highway 静默降级到 SSSE3(128 位,Lanes=4)**——objdump 实证四个 `*_highway` kernel 全是 xmm 无 ymm。于是"parity"实际是 HYLE-256b vs Highway-128b,那 1.1-2x"赢"是宽度差不是抽象优势。修复:`hyle_set_highway_isa`(`HyleModule.cmake`)给对照 TU 补全 baseline(`-mbmi -mbmi2 -maes -mpclmul`),Highway 选回真 256 位 AVX2(`HWY_STATIC_TARGET=AVX2`、Lanes=8、四 kernel 全 emit ymm)。MSVC `/arch:AVX2` 本就含完整 baseline,故 Win 侧测的 dtype/math 节不受影响。下表是**修复后**的公平数据。

| kernel | scalar(auto-vec SSE2) | HYLE sse4(128b) | HYLE avx2(256b+FMA,4-acc) | Highway contrib/dot(**真 SOTA 4-acc**) | 结论(15 遍中位数) |
|---|---|---|---|---|---|
| **dot product** | 2.7 G op/s (1.0x) | 26 G (9.6x) | **42 G (15x)** | 42 G (15x) | **平真 SOTA**(ratio 0.984,gap 1.6% « spread 14%);4 累加器展开后 12.6→42 G |
| **saxpy** y=a*x+y | 17.4 G (1.0x) | 17.4 G (1.0x) | **39 G (2.2x)** | 40 G (2.3x) | **平 Highway**(element-wise 带宽受限,单累加器已够,未改) |

**结论(2026-07 红队揪出并修复第二个对照失真问题——dot 单累加器延迟链)**:
1. **旧"dot 平 Highway 12.6 vs 12.7 G"是两个都用了次优写法的假平局**。HYLE `dot_f32`(`kernel_impl.h`)和 bench 的"SOTA 标尺"`dot_highway` **都是单累加器**(`acc = MulAdd(.., acc)`)——每条 FMA 串进下一条,throughput 被 FMA **延迟**(~4 cyc)绑死,每 4 周期才退休一条。bench 拿的 Highway 参照是其**教学式 `ScalableTag` 写法,不是** Highway 真正的 `contrib/dot::Compute`(4 独立累加器 + 4x 展开,注释明写"2 FMA ports × 4 cycle latency = up to 8 in-flight, always unroll 4x")。**这是继"Highway 静态 dispatch 静默降 128b"之后的第二个同类对照陷阱:对照用了非优化路径。**
2. **修复=dot 主循环 4 独立累加器 + 归约树**(objdump 实证:主循环 4 条 `vfmadd231ps` 写 ymm0/1/2/3 + `vaddps` 树)。i9 双 FMA 端口下,4 链保持 ~8 FMA in-flight,从延迟绑定转 load 端口绑定。**dot 0.079→0.024 ns(3.3x,12.6→42 G op/s),对拍 Highway 真 `contrib/dot` 统计平局(ratio 0.984,gap 1.6%«spread 14%)**——真 SOTA 上限 42 G 而非旧标尺的 12.6 G。dot 是 AI 热路径主力(注意力/GEMM 内积),这是真收益。
3. **saxpy/scale/reduce 未改**:saxpy(`y=a*x+y`)、scale(`y=a*x`)是 **element-wise 无跨迭代依赖**,本就带宽绑定,多累加器无益(saxpy 已 39 G 平 Highway 40 G)。reduce_max 有归约依赖(理论同 dot 可展开)但**是 bench-only kernel,非公开 op**(`ops.h` 只导出 dot_f32/saxpy_f32),故不改;若将来上架为公开 op 再同 dot 展开。
> **意义 + 方法学印证**:这正是审查铁律"对标 SOTA 务必 objdump 确认对照真用了预期实现/宽度"的活例——第一次是对照被降级编译到 128b,这次是对照用了单累加器教学 kernel 冒充 SOTA。**两次都是"平局"掩盖了真实差距**:表面同速,实为两边都没到硬件上限。dot 真 SOTA 是 42 G,HYLE 现达到。**教训:'平 SOTA' 若对照本身可疑(如同形却都简单),要追问对照是不是该库的生产实现。**

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

## simd dtype 转换（fp16/bf16 <-> f32，`dtype_bench.cc`）

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
