# simd-09：数学函数 best-of 补齐 + 算法级能力(vqsort/RNG/intdiv)

| 字段 | 值 |
|---|---|
| **阶段** | P3 |
| **模块** | simd/ |
| **前置** | simd-06(数学核 best-of 机制:exp/log/tanh)、simd-08(op 面完备) |
| **状态** | ✅ 线① 数学函数 best-of(88bcc46,三平台)+ 线② 全 5 件 intdiv/cmp128/RNG(自实现)+ vqsort/bit_pack(vendor 薄封装 Highway)+ **收尾 4 缺口**(#66/#69 gather/compress 原生+bench 实测、#67 gelu/softmax、#68 rsqrt/rcp Newton)完成。四 ISA 全真跑绿(MSVC+WSL GCC 各 217/169×3 + **AVX512 Intel SDE 169/169**);NEON mac 真机已验 |

> **范围**:两条线。①**数学函数 best-of 批**——simd-06 的"三方 bench 选最快核"方法学只覆盖了 exp/log/tanh,把它扩到 sin/cos/exp2/log2/atan/erf/pow… 等两库都有真多项式核的超越函数。②**算法级能力补齐**——vqsort 向量排序 / 向量 RNG / intdiv 向量整除等 Highway contrib 招牌功能,HYLE 整块缺。
> **驱动**:用户 2026-07-09 "best-of 机会希望全面寻找,三者都有就用最快的;数学库很重要,缺了要补充"。

## best-of 判据(为什么是数学函数 + 这几个算法能力)

见 `backlog-map.md` 的"best-of 判据"块。核心:"三者都有选最快"只在 op **有真实算法内容 + 三家有差异 + 不贴单一硬件指令**时才有意义。据此:
- 数学函数(多项式逼近)= best-of 借库**主场**,simd-06 只覆盖 3 个是巨大遗漏。
- 结构 op 的性能空间大多是**自造 native intrinsic**(gather/compress 上 vpgather/vpcompress),非借库——因为参考库在窄 ISA 上和 HYLE 一样 store+reload emulation。
- 算法级能力(vqsort/RNG/intdiv)是**借库补缺**(只 Highway 有,非三方选优)。

## 线① 数学函数 best-of 批

复用 simd-06 的完整机制(见下"接入模式"),逐 op 在 `math_bench.cc` 跑同 ISA xsimd vs Highway,谁快借谁,scalar libm 永远当 oracle + 兜底。

**目标函数**(按 AI/数值价值排序):
| 批 | 函数 | 两库都有? | 用途 |
|---|---|---|---|
| 9a | sin / cos / sincos | ✅ | RoPE 位置编码、信号 |
| 9b | exp2 / log2 | ✅ | 量化、熵、对数域 |
| 9c | erf | ✅ | GELU 精确激活 |
| 9d | pow | ✅ | SwiGLU、幂激活 |
| 9e | atan / atan2 | ✅ | 几何、相位 |
| 9f | expm1 / log1p / cbrt | ✅ | 数值稳定、范数 |
| 9g | sinh/cosh/asinh/acosh/atanh/asin/acos | ✅ | 补全双曲/反三角 |

**xsimd 独有(单借,非 best-of)**:hypot / lgamma / tgamma、复数超越函数(`batch<complex>` exp/log/pow/sqrt/sincos)——Highway 无对手。按需。

**接入模式**(照 simd-06 葫芦画瓢,每函数动 6-7 处):
1. `include/hyle/simd/math.h`:加 `<fn>_f32_batch` 声明
2. `src/simd/math/kernels.h`:加各 tier detail:: 声明
3. `src/simd/math/transcendental_scalar.cc`:libm oracle + 兜底
4. 借核 .cc(`transcendental_{xsimd,highway}{,_avx512}.cc` + `_highway_neon.cc`):bench 决定借谁就写哪份
5. `src/simd/math/math_dispatch.cc`:加 dispatch(复制 exp 的 `#if HYLE_ARCH_X64` + `have_avx512/have_avx2` 分支)
6. `bench/simd/math_bench.cc`:扩 `enum Op` + `cases[]` 一行 → 三方对比
7. 精度测试:`tests/simd/math_test.cc` 加 ULP 对拍 libm

## 线② 算法级能力补齐(借 Highway contrib)

非 best-of(只 Highway 有),属"借难且已解决的"能力补缺。各自独立子里程碑,工作量大:
| 子 | 能力 | Highway 源 | 价值 | 备注 |
|---|---|---|---|---|
| 9-sort | vqsort 向量排序 | contrib/sort | 很高 | Highway 招牌;partial/select/KV/128-bit 键。借 vs 自造待评估 |
| 9-rng | 向量化 RNG | contrib/random | 高 | Xoshiro/AesCtr counter-based 可并行 |
| 9-intdiv | 向量整数除法 | contrib/intdiv | 高 | magic-number emulation,HYLE 只有浮点 Div |
| 9-bitpack | 任意位宽打包 | contrib/bit_pack | 中 | 量化存储 |
| 9-cmp128 | Lt128/Eq128/Min128/Max128 | ops/generic_ops | 中 | vqsort 基石 |

## 真机纪律
沿用 simd-06/08：x86 定宽档 WSL g++-14 真跑 + ASan，NEON mac 真机，AVX512 由 Intel SDE 做正确性与派发验证。数学函数额外跑 ULP 精度对拍 libm；best-of 决策必须由 `math_bench` 同 ISA 真机数据支撑，SDE 不作性能证据。

## 状态记录

### 线① 数学函数 best-of — ✅ 完成(2026-07-09,commit 88bcc46)
11 个函数接入,per-op winner 由 `math_bench` **GCC+MSVC 双平台交叉验证**定(防单平台陷阱):

| winner | 函数 | GCC 优势 | MSVC 优势 |
|---|---|---|---|
| **Highway**(9) | sin | 1.85x | 2.28x |
| | cos | 1.75x | 2.12x |
| | log2 | 1.23x | 1.44x |
| | atan | 1.30x | 1.52x |
| | cbrt | 1.07x | 3.26x(MSVC xsimd cbrt 极慢) |
| | expm1 | ~1.07x | 1.26x |
| | log1p | 1.12x | 1.43x |
| | asinh | ~平 | 1.10x |
| | acosh | 1.07x | 1.12x |
| **xsimd**(2) | exp2 | 1.29x | ~平(取 GCC 明显优势) |
| | erf | 1.14x | 1.07x |

**关键结论**:与 simd-06(exp/tanh→xsimd)**相反**,新批几乎全归 Highway(尤其 sin/cos ~1.8-2.3x)——**证明逐 op bench 必要,winner 不可外推**。唯二冲突(exp2 GCC-xsimd/MSVC-平、asinh GCC-平/MSVC-Highway)按"取有明显优势方、另一平台无损"解决,无需 per-platform 分叉。

**接入**:五 TU 按 simd-06 模式(scalar oracle 全 11 + x86 借核只写 winner 半避免死码 + NEON 全 Highway 一份验证核 + avx512 siblings GCC-only、MSVC-AVX3-broken 排除)。math_dispatch 宏生成 Highway 型 9 个 + longhand xsimd 型 2 个。复用现有 TU 无需改 CMake。
**执行验证**：Win MSVC AVX2 / WSL GCC AVX2 / mac NEON 真机下 `Math.Simd09MatchesLibm` 全 11 项对拍 libm（约 3.5 ULP）通过；AVX512 路径后续由 Intel SDE 通过正确性与 kAvx512 派发门。SDE 不作性能证据。

### 线①-b xsimd 独有借入:tgamma/lgamma(✅ 2026-07-09,揪出 xsimd vendor bug)

用户"xsimd 独有的我们也可以直接借过来"→ 借入 Highway contrib/math **没有**的 gamma 函数(单借,非 best-of,无第二实现竞速)。**做全所有 arch**(能力完整性,非"够本机"):x86 avx2/avx512 + **arm64 首个 xsimd-NEON TU**(`transcendental_xsimd_neon.cc`——此前 NEON math 全借 Highway,但 Highway 无 gamma)+ scalar libm oracle。

**纠错**:原以为 hypot 也是 xsimd 独有——实为 **Highway contrib/math 也有**(`math-inl.h:396`)且是 **binary**(两输入),属 best-of 候选非单借,留待补。真正 xsimd 独有的是 **lgamma/tgamma**。

**借核对拍揪出 xsimd 向量化 lgamma vendor bug**(第二次"借不等于信",继 Highway Pack16 后):新增 `Math.XsimdGammaMatchesLibm` 对拍 libm,lgamma 在 in=1.29875 偏差 **0.83**(错值非精度)。**三 target 独立坐实**(GCC-x86 + MSVC-x86 + GCC-NEON qemu 同一错值 0.833834)= 算法 bug 非 codegen。根因:xsimd `lgamma_impl<float>::other()` 参数归约多分支 select 覆盖不全——lane∈[1.25,1.5) 与 lane∈[0.75,1.25) 同向量时归约核 blend 错(distinct-per-lane 才触发,同值填充漏网,一度骗过我)。**tgamma 不受影响**。处置:**lgamma 整体退回 scalar libm**(API 正确、未向量化),tgamma 保持向量化,不改 vendored 源。详见 `docs/developer/debugging/xsimd-lgamma-lane-mixing-hunt.md`。

**四平台验证**:MSVC 219(+1 gamma 对拍)、WSL GCC AVX2 219、AVX512 SDE(tgamma-512 TU 链入)、**arm64 NEON qemu-aarch64 真跑**(lgamma worst 1.2e-5=libm 精确、tgamma 相对误差 ~7e-8、variant=kNeon)。

### 线①-c hypot binary 双契约:safe 与 fast 显式二选一(✅)

hypot 是 **binary best-of**(两库都有,非独有)。xsimd direct-formula 核约快2倍，但当 `a*a+b*b` 溢出或下溢时不具备 `std::hypot` 的缩放语义；Highway `Hypot` 做指数缩放覆盖完整范围。因此公开层提供两个不混淆的入口：`hypot_f32_batch` 借 Highway，保持完整语义；`hypot_in_range_f32_batch` 借 xsimd，由调用方保证普通输入域，热路径不做逐 lane 检查。前者的测试含3e19溢出用例，后者只对契约内输入对拍。

接入覆盖 scalar direct formula、x86 AVX2/AVX512 xsimd 与 arm64 xsimd-NEON。真机 `bench-stat -n15`：WSL AVX2 fast 0.140 vs safe 0.281 ns/op(**2.03×**)，Apple NEON 0.220 vs 0.454(**2.07×**)；Win/WSL/Mac 正确性门全绿。f64 direct-formula 仅统计打平，故不提供受限 fast 入口。具体输入限制见 usage guide §2.2。本次双契约增量经 cloc 核实为 **+43 production / +23 tests / +8 benchmark = +74 code LOC**。

### 线①-d 复数超越函数(✅ 2026-07-09,用户"xsimd 独有的复数超越函数抄过来")

复数 API 是 **SoA planar**（实部虚部分离数组 re[i]/im[i]）——匹配 xsimd `load_unaligned(re, im)` 的零交织构造，且是 FFT/信号的自然布局；精确原位 dst alias src 可用。这里记录的是 simd-09 落地时的旧路由；当前 f32/f64 完整集合及 AVX2/NEON/SVE/RVV 共用 typed D-tag body 见
[`simd-capability-fill-list`](simd-capability-fill-list.md) C3。

### 线③ fp8 dtype(E4M3/E5M2,✅ 2026-07-09,用户"fp8 推理常见精度还是要做的")

补 dtype 层 fp8。OCP 两格式:**E4M3FN**(bias 7/3 尾数/无 inf/唯一 NaN=0x7F/max 448,权重激活)、**E5M2**(bias 15/2 尾数/IEEE inf-NaN/max 57344,梯度)。**编码 transcribe 自权威 CUDA `cuda_fp8.hpp`**(ggml/Highway 均无 host fp8,不 vendor→自实现,同 fp16/bf16 抄 Highway 的做法);共享模板参数化两格式(字段宽度+特殊值规则),saturate-to-max-normal(OCP SATFINITE),E5M2 真 inf 输入保留 inf、finite 溢出才 saturate。

**关键 ISA 边界(非缓做)**:**无 CPU 硬件 fp8-convert**——硬件 fp8 转换是 AVX10.2 / GPU tensor core 专属,本代 AVX2/AVX-512 CPU 全无,Intel SDE 也不模拟。故 **scalar 位运算即当前完整实现**(不是"缓到有硬件",是根本无硬件可用);有 AVX10.2 时按 cvt_avx512bf16 先例接原生核。dtype.h 类型+转换 / dtype_ops 批处理 / cvt_scalar / dispatch(scalar 直调)。

**验证**:known values 锚 OCP 谱(1.0/448/57344/NaN/subnormal)+ **全 256 值 exhaustive round-trip**(fp8 仅 256 值→完备,强于 fp16 抽样)+ batch==element + 独立 hand-computed anchor 交叉验(揪出我一个手算错值,代码对)。三平台:MSVC 225 / WSL GCC 225 / **arm64 qemu 512-check bad=0**(scalar 无需 SDE)。

### 线② 算法级能力 — ✅ 完成(2026-07-09,全 5 件,四 ISA + SDE + mac 真机)

| 子 | 状态 | 说明 |
|---|---|---|
| **intdiv** u32/i32 | ✅ 完成(8114f40) | Granlund-Montgomery magic-number **自实现泛型层**(不 vendor Highway D-tag)。Divisor<T> 预算 + Div(V,Divisor) 应用;32-bit 无原生 MulHigh→u32→u64 widen+ShiftRight<32>,recombine 用 BitCast+ConcatEven(避 ReorderDemote2To 顺序陷阱)。MSVC+WSL GCC+ASan 净+mac NEON 真机。u8/u16/u64 后续(u64 需 128-bit 乘) |
| **cmp128** Lt128/Min128… | ✅ 完成(72bfb1f) | 每相邻 u64 对=128-bit 值。泛型层组合(Eq/Lt/Reverse2/DupOdd,block-safe),抄 Highway Lt128Vec 真值表。MSVC+WSL GCC+ASan 净+mac NEON 真机。vqsort 键比较基石 |
| **RNG** xoshiro256** | ✅ 完成(111a83b) | `include/hyle/simd/random.h` VectorXoshiro N 并行流,逐 lane 独立状态更新,**自实现泛型层**。SplitMix64 播种+逐 lane Jump 去相关;Uniform 用位模式技巧避 u64→f64 非原生转换。lane0 对拍标量 oracle。MSVC+WSL GCC+ASan 净+mac NEON 真机 |
| **vqsort** 排序 | ✅ 完成(5c1c884) | **vendor 薄封装** Highway vqsort(~10x std::sort),同数学核借法。sort/partial_sort/select(u32/i32/u64/i64/f32/f64 升降序)吃裸指针;`VQSortStatic` header-only + `-DVQSORT_ONLY_STATIC=1` 免链 vqsort.cc;dispatch avx512>avx2>scalar 门控。sort_test 对拍 std::sort/partial_sort/nth_element。MSVC+WSL GCC 全绿 + AVX512 SDE tier 真跑 + mac NEON。K32V32/K64V64 top-k 键值对已做(sort.h KeyValue32/64) |
| **bit_pack** 位打包 | ✅ 完成(35bab19) | **vendor 薄封装** Highway bit_pack。pack/unpack(u8 1..8/u16 1..16/u32 1..32)吃裸指针+运行期 bits(index_sequence 折编译期 kBits);block 走 Highway、tail 走自写 bitstream codec。**揪出并绕过 vendor bug**:Highway Pack16<11>/<13> 满位宽 round-trip 失败(独立探针 MSVC+GCC+EMU128 三处坐实=算法非 codegen;根因上游 `& kBits` 笔误漏测),这两宽度回退 HYLE bitstream codec。MSVC+WSL GCC 全绿(213)+ AVX512 SDE tier 真跑 + mac NEON。详见 docs/developer/debugging/highway-pack16-roundtrip-hunt.md |

**线②小结**:全 5 件落地。通用原语(intdiv/cmp128/RNG)**自实现泛型层**;重型件(vqsort/bit_pack)**vendor 薄封装 Highway**(与数学核借法同构——公开层吃裸指针不 include highway,借核 TU 内部用 Highway D-tag,dispatch 运行期门控 SIGILL 安全)。**关键收获**:借核同样过 HYLE 对拍闸门,bit_pack 的 round-trip 恒等式揪出 Highway 一个满位宽 bug(Pack16<11>/<13>)——"借不等于信"。K32V32 top-k 键值对已做(KeyValue32/64,四 ISA+SDE+mac 验)。

## AVX-512 首次真机(Intel SDE)验证 — 补齐历史空白(2026-07-09)

simd-01~09 全程 avx512.h 及各 avx512 tier 都标"写+objdump 验 zmm 但**从没跑过**"(无 AVX-512 机)。本次为 compute-07(会路由到 512 后端)收尾,用 **Intel SDE 9.48.0**(`sde64 -future`,WSL /root)首次真跑,补齐这个最大空白:
- **vec_test avx512 变体**(新增,`HYLE_SIMD_TARGET_AVX512`):avx512.h 全 op 面对拍 scalar oracle,**SDE 169/169 全绿**——含本会话新增的 crypto/cmp128/intdiv/RNG。
- **math/sort/bitpack 的 avx512 tier**:tier 探针确认 `sde64 -future` 下 `active_{math,sort,bitpack}_variant()==kAvx512`(裸机 kAvx2),即 dispatch **真的路由到 512 tier**,19/19 全绿。裸机正确留 kAvx2(SIGILL 安全双向坐实)。
- **SDE 揪出 2 真问题**(修复):① avx512.h **缺 16 个 simd-08 A-2 mask 域 op**(冻结在 simd-07,无 avx512 gtest 变体故一直不可见)→ 补齐 k-register 形式;② vec_test `kMaxLanes=32` 栈越界(AVX512 i8=64 lane,同 TruncateTo/SumsOf 类)→ 改 64。

**方法学**:"无硬件 → 理由化不验"是陷阱(simd-design 已记用户纠偏);Intel SDE 是 AVX-512 无机时的真验闸门(模拟 CPU 让 dispatch 真选 512 tier + 全 op 对拍),不该再让 avx512 停在"objdump 验"。AVX-512 从"写了没跑"升级为"SDE 真跑全绿"。

## 收尾:4 个 simd 缺口(2026-07-09,用户"这 4 个缺口先做了…simd 不要缓,都做了")

线①/② 之外,盘点 simd 面还剩 4 个非平凡缺口,一并做掉(而非缓到消费驱动):

| # | 缺口 | 做法 | 处置 |
|---|---|---|---|
| **#66/#69** | gather/compress 原生 intrinsic + performance.md 对比 | avx2 `GatherIndex`→`_mm256_i32gather_ps/epi32`;avx512 `Compress/Expand/CompressStore`→`vpcompress/vpexpand`、`GatherIndex`→`vpgather`。保留 `GatherIndexEmulated/CompressEmulated`(store+reload 参照)供 A/B。新增 `bench/simd/gather_bench.cc`(逐 ISA exe) | **数据驱动**:AVX2 GatherIndex 原生**本机实测 1.31x 赢→保留**;AVX2 Compress 无 VBMI2 本就 store+reload(~持平);**AVX512 SDE 数不可信**(功能模拟器对复杂指令模拟开销高),原生**正确性已验(SDE 169/169+Gather/Compress 测试全绿),性能待真机**。见 performance.md gather 段 |
| **#67** | AI 激活组合核 | `gelu`(精确 erf)/`gelu_tanh`(ggml 近似)/`sigmoid`/`silu`(逐元素) + `softmax`/`log_softmax`(归约:max 稳定化+exp+double 累加)。baseline C++ 调各 `*_f32_batch` 自派发 | ✅ 三平台+SDE 全绿;`math_test` Activations/Softmax 对拍。**注**:simd 层已做,compute-07 目前用 ggml 的算子非 simd 的(消费侧选择,非缺口) |
| **#68** | rsqrt/rcp Newton 精修 | 泛型层 `ReciprocalNewton` y₁=y₀(2−x·y₀)、`ReciprocalSqrtNewton` y₁=y₀(1.5−0.5x·y₀²)。~12-bit 估计→~23-bit f32,比 Div/Sqrt 便宜 | ✅ **HYLE 超越点**——xsimd/Highway 只出裸估计,加一步 Newton 可反超两家(best-of 判据里"自造超越"类);`vec_test` MathExtFloat 对拍。dtype int8/fp8:int8 lane 已有,fp8 无消费者→消费驱动缓 |

**收尾小结**:4 缺口 3 个全做(#67 激活、#68 Newton、#66/#69 原生+bench 实测)。**关键收获**:#66 印证 best-of 判据"原生不必更快、必须测"——AVX2 gather 测出真赢(上),AVX512 无真硬件只验正确性(原生留、性能挂真机);**SDE 验正确性可信、验性能不可信**是本轮明确的工具边界。dtype fp8 唯一留消费驱动(无消费者)。

**四缺口后全平台回归(2026-07-09,HEAD 06016b6)**:Win MSVC + WSL GCC 各 `simd_test 217/169×3`(1 skip=MSVC-AVX3-broken bf16,预期)+ **AVX512 SDE 169/169**。四 ISA 全真跑绿。

## 线⑤ op-parity 性能收口(✅ 2026-07-09,用户"自己做的性能不如抄 highway 就很愚蠢"→"返回全平")

能力面补全后,回答"自造 ~217 op/后端是否真不亏"。`bench/simd/op_parity_bench.cc`(+`_full`/`_widen`/`_sse4` 变体):HYLE vs Highway vs xsimd **同 TU/同 ISA/两边 header-only 都内联**(避内联不对称假差),`bench-stat.py -n15` 看中位倍数(gap>两变体 spread 和才判赢/输,防亚-ns 噪声假 TIE)。

**揪出 9 个 store+reload/std 标量慢 op**(GetExponent 30x/DupEven-Odd 10x/ReverseBits 9x/TZCnt 8x/Compress 6x/PopCount 5x/LZCnt 5.5x/ConcatEven-Odd 1.7x;都是 simd-05 批11"位计数留 std::"的欠账)→ **四固定宽后端全换原生**(avx2 494a4b0→avx512 925c689→sse4+neon db0352b)。**残留 2 个 2.x 输家(LZCnt/Compress)再追平**(c9390a1→b26279c):
- **LZCnt/TZCnt float-exp trick**:`cvtepi32_ps` 取偏置指数即 bit-length,LZCnt=min(158-be,32)、TZCnt=min(be(x&-x)-127,32),带 round-down 归一化 `v&~(v>>24)`。avx2/avx512/sse4 32-bit(avx512 tier 无 CD/vplzcnt、sse4 用 _mm_min_epu32);NEON 用原生 vclzq 无需改。
- **Compress**:①**契约改零尾为 unspecified**(对齐 Highway/xsimd——旧强契约在 AVX2/SSE4 无清零指令要 ~2x 软件税,核实 vqsort 不依赖 + 标准用法不碰尾部)②avx2 用 **movemask_ps 直取 mask**(原 mask_bits 标量压位循环才是真瓶颈)+1KB packed 表 ③sse4 用 **256B pshufb 字节索引表**(抄 Highway IndicesFromBits128)。
- **结果**：AVX2/SSE4 与 Apple-Silicon NEON 的已测 op 均 ≥ Highway 统计打平（`bench-stat -n15`）；NEON 真机复测另揪出并修复 Compress/Expand 的 store+reload 回退。AVX512 SDE 169/169 只作正确性证据，性能仍待真机。详见 benchmarks-simd.md op-parity 全表。
- **教训**:①"慢 2.x"先别归因表大小/算法天花板——真瓶颈常在具体细节(Compress 是 mask 提取的标量循环);②强契约(零尾)若无硬件支撑+无消费者依赖,是常年性能税,对齐参照库弱契约更值(但要核实消费者 + 改测试,不静默改语义)。

## 线④/⑥ SVE + RVV scalable 后端（✅；simd-10 继续补齐 P0-P2）

SVE/RVV 使用后端内 traits 承载 sizeless native type；`sizeof` 与结构体成员仍是硬禁区，op 的传值/返回值合法。共享 `vec.h` 以 `Rebind+kPow2+cap` 表达 partial tag 与 lane-width rebind，后端分别用 SVE predicate 和 RVV `vl` 保证只观察 `Lanes(d)`。

simd-10 已完成：
- **P0 泛型解锁**：补齐 complex/128-cmp/Newton/rotate 所需底层原语。
- **P1 scalable swizzle**：table/slide/interleave/concat/reverse 与 segmented I/O 覆盖 logical lanes，支持 partial tag。
- **P2 widen/quantize/reduce**：Promote/Truncate/Demote2、widen-dot、SumsOf2/4/8、ReduceMul、Q15、SaturatedAbs/Neg；Q15 明确保持 `INT16_MIN * INT16_MIN -> INT16_MIN`。

| 后端 | 最终执行门 | 关键边界 |
|---|---|---|
| **SVE** (`backend/sve.h`) | 15 tests × VL 128/256/384/512 全绿；P2/partial 另在 VL 2048 通过 | `PR_SVE_SET_VL` 设置并回读实际 VL；384-bit 门覆盖非 2 次幂 lane 数 |
| **RVV** (`backend/rvv.h`) | 12 tests × VLEN 128/256/512/1024 全绿 | QEMU 上限 1024；unsigned reduction、inactive tail 与 partial store 均有对抗测试 |

剩余 P3-P5 与 SVE2/RVV 扩展硬件 gate 以 [`simd-gap-inventory.md`](simd-gap-inventory.md) 为唯一施工清单。

**核心载荷面 + 常用 op 面对两 scalable ISA 已覆盖**,剩余是硬件扩展 gate 或语义受限项。
