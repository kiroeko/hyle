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
沿用 simd-06/08:x86 三档 WSL g++-14 真跑 + ASan;NEON mac 真机;AVX512 无机编译验。数学函数额外跑 ULP 精度对拍 libm。best-of 决策必须 `math_bench` 同 ISA 实测数据支撑(不凭记忆)。

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
**三平台验证**:Win MSVC AVX2 / WSL GCC AVX2(+highway_avx512 编译验)/ mac NEON 真机,`Math.Simd09MatchesLibm` 全 11 对拍 libm ~3.5ULP 通过;simd_test 195→196。AVX512 执行存疑(无硬件,objdump 模式已验 zmm)。

### 线①-b xsimd 独有借入:tgamma/lgamma(✅ 2026-07-09,揪出 xsimd vendor bug)

用户"xsimd 独有的我们也可以直接借过来"→ 借入 Highway contrib/math **没有**的 gamma 函数(单借,非 best-of,无第二实现竞速)。**做全所有 arch**(能力完整性,非"够本机"):x86 avx2/avx512 + **arm64 首个 xsimd-NEON TU**(`transcendental_xsimd_neon.cc`——此前 NEON math 全借 Highway,但 Highway 无 gamma)+ scalar libm oracle。

**纠错**:原以为 hypot 也是 xsimd 独有——实为 **Highway contrib/math 也有**(`math-inl.h:396`)且是 **binary**(两输入),属 best-of 候选非单借,留待补。真正 xsimd 独有的是 **lgamma/tgamma**。

**借核对拍揪出 xsimd 向量化 lgamma vendor bug**(第二次"借不等于信",继 Highway Pack16 后):新增 `Math.XsimdGammaMatchesLibm` 对拍 libm,lgamma 在 in=1.29875 偏差 **0.83**(错值非精度)。**三 target 独立坐实**(GCC-x86 + MSVC-x86 + GCC-NEON qemu 同一错值 0.833834)= 算法 bug 非 codegen。根因:xsimd `lgamma_impl<float>::other()` 参数归约多分支 select 覆盖不全——lane∈[1.25,1.5) 与 lane∈[0.75,1.25) 同向量时归约核 blend 错(distinct-per-lane 才触发,同值填充漏网,一度骗过我)。**tgamma 不受影响**。处置:**lgamma 整体退回 scalar libm**(API 正确、未向量化),tgamma 保持向量化,不改 vendored 源。详见 `docs/debug/xsimd-lgamma-lane-mixing-hunt.md`。

**四平台验证**:MSVC 219(+1 gamma 对拍)、WSL GCC AVX2 219、AVX512 SDE(tgamma-512 TU 链入)、**arm64 NEON qemu-aarch64 真跑**(lgamma worst 1.2e-5=libm 精确、tgamma 相对误差 ~7e-8、variant=kNeon)。

### 线①-c hypot binary best-of:correctness 压过速度选 Highway(✅ 2026-07-09)

hypot 是 **binary best-of**(两库都有,非独有)。`math_bench` 加 binary 对拍段,MSVC+GCC 一致:**xsimd 0.14ns vs Highway 0.30ns(xsimd ~2.2x 快)**。但 `HypotMatchesLibm` 对拍(含 3e19/1e20 溢出用例)揪出 **xsimd hypot 非 overflow-safe**——b²>FLT_MAX 直接返 inf(裸 `sqrt(a*a+b*b)`),而 Highway 的 `Hypot` 做指数缩放匹配 std::hypot 全域。**overflow-safe 正是 hypot 存在的意义(vs 裸 sqrt),故 correctness 压过 2.2x → 借 Highway**。这固化了 best-of 判据的关键前置:**"最快的正确实现"非"最快"**(见 backlog-map best-of 判据 ⚠️ 块)。

接入:全 arch(x86 avx2/avx512 + arm64 NEON + scalar),binary 签名(两输入数组)。MSVC AVX3-broken 故 512 排除同 log。四平台验证:MSVC 220 / WSL GCC 220 / AVX512 SDE 169 / arm64 NEON qemu(hypot bad=0/8 含溢出、variant=kNeon)。

**xsimd 独有仍未接**(按需):复数超越函数(`batch<complex>` exp/log/pow/sqrt——需复数类型面,Highway 无对手)。

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
| **bit_pack** 位打包 | ✅ 完成(35bab19) | **vendor 薄封装** Highway bit_pack。pack/unpack(u8 1..8/u16 1..16/u32 1..32)吃裸指针+运行期 bits(index_sequence 折编译期 kBits);block 走 Highway、tail 走自写 bitstream codec。**揪出并绕过 vendor bug**:Highway Pack16<11>/<13> 满位宽 round-trip 失败(独立探针 MSVC+GCC+EMU128 三处坐实=算法非 codegen;根因上游 `& kBits` 笔误漏测),这两宽度回退 HYLE bitstream codec。MSVC+WSL GCC 全绿(213)+ AVX512 SDE tier 真跑 + mac NEON。详见 docs/debug/highway-pack16-roundtrip-hunt.md |

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
