# SIMD 复杂度级优化机会清单(2026-07-15 实证审计)

> 结论先行:255-op parity 层的**常数因子之战已经打完**(三平台真输清单全零,见
> [`simd-orderswap-matrix.md`](simd-module-closure.md)),继续在 op-vs-op 层抠常数是负收益。
> 还能**超过现在表现**的机会全部在算法件层,且必须按"复杂度轴"逐一审:本文按五个轴
> (工作量 O / 依赖链深度 O / 内存流量 O / 误差 O / 指令类)盘点,每项附仓内证据、算法、预期
> 量级、SOTA 参照与消费者。姊妹文档:[`simd-gap-inventory.md`](simd-gap-inventory.md)(op 面
> 覆盖缺口,正交视角);立项后的测量纪律见
> [`../performance/simd-op-benchmark-methodology.md`](../../../developer/performance/simd-op-benchmark-methodology.md)。
>
> **审全面性的方法教训(本清单第一版漏了两个整类后补齐)**:逐组件扫(sort/math/bitpack/...)
> 不够,必须**逐复杂度轴扫**——第一版只扫了工作量 O 和深度 O,漏掉内存流量 O(pass 融合)和
> 指令类(VNNI),而后者恰好包含最大杠杆与现存缺陷。

---

## 立即可做:仓内已坐实的三处证据(P0)

### P0-1. softmax/log_softmax 的 pass 融合 — **已落地(单次 exp 融合 multipass,2026-07-15)**

**原状**:max → sub → exp → sum → scale,**4~5 遍**;中间 sum 单独扫。

**已做**:先 max;再按 4K chunk 做 `(x-m)→exp_f32_batch→累加 sum`(**整表只向量 exp 一次**);
最后 scale。log_softmax:max + 标量 sum-exp 得 logZ,再一遍 `x-(m+log s)`。
实现:`src/simd/math/activations.cc`(`d9685ff`)。

**曾试 online max+sum-exp(已弃)**:pass1 标量 exp 攒 s + pass2 再向量 exp = **双 exp**,Mac n=128K
比旧 multipass 慢 ~1.6×。CPU 有向量 exp 时「单次 exp + 少扫」才是正收益。

**正确性**:`Math.Softmax` 对拍 double 稳定参考 — **Win MSVC / WSL GCC / Mac NEON 三平台 PASSED**。

**性能(Mac NEON,2026-07-15;对拍对象=标量 3-pass / 旧 multipass,非 Highway——Highway 无 softmax 组合 API)**:

| 场景 | 结果 |
|---|---|
| `activation_bench` n=4096 vs scalar 3-pass | HYLE **1.16×**(115.9%; 1.42 vs 1.65 ns/op) |
| chrono AB n=4K 融合 / 旧 multipass | **0.81×**(更快) |
| chrono AB n=32K 融合 / 旧 multipass | **0.86×** |
| chrono AB n=128K 融合 / 旧 multipass | **~1.00×**(持平,逼近带宽墙) |
| chrono AB n=4K/32K/128K 融合 / scalar | **0.41× / 0.66× / 0.91×**(均不慢于标量) |
| 双 exp online(已弃) n=128K / 旧 multipass | **1.60× 更慢**(反例,写入档案) |

**机会残留**:正式 scaling bench(n=1e3..1e8 斜率写入 matrix/bench 文档);sum 累加可再向量化。

### P0-2. AVX-VNNI 通道 + 修 avx512 SumOfMulQuadAccumulate 的标量洞 — **已收口(2026-07-19)**

**原状,两处铁证**:
- avx2 注释称 "No AVX2 VNNI" 过时——Alder Lake+ 有 VEX **AVX-VNNI**(`vpdpbusd`);本机 13900K
  `/proc/cpuinfo` 含 `avx_vnni`。
- avx512 `SumOfMulQuadAccumulate` 曾是 **store+reload 标量 4-way 循环**(signedness 借口绕开 VPDPBUSD)。

**已做**:
1. **avx512 标量洞关闭**:非 VNNI 走 `PromoteEven/Odd + WidenMulPairwiseAdd`(与 avx2 generic 同构);
   **顺带**把 avx512 `PromoteEvenTo/OddTo` 从 store+reload 改成寄存器 shift/mask(同 avx2 形状,
   否则 "向量化路径" 仍被 promote 拖回内存)。
2. **VNNI 编译期路径**(Highway i8×i8 符号修正 `r0=dpbusd(sum,a,b); r1=dpbusd(0,a>>7,b)<<8; r0-r1`):
   - avx512:`#if __AVX512VNNI__`
   - avx2:`#if __AVX_VNNI__ || __AVXVNNI__`(默认 avx2 tier **不**带 `-mavxvnni`,仍走 madd 链)
3. **`cpu_features().avx_vnni`**:CPUID leaf 7,1 EAX bit 4 + OS YMM 门控(`cpu.h`/`cpu.cc`)。
4. **产品 runtime 覆盖**：新增稳定语义的 `i8×i8→i64` whole-buffer dot；独立
   `AVX2_VNNI`（VEX）与 `AVX512_VNNI`（EVEX）TU，分别由 `avx_vnni` /
   `avx512vnni` 精确门控，仅替换这一项函数指针。默认与 exact AVX2/AVX512
   继续使用无 VNNI 基线。MSVC 必须用 `_mm256_dpbusd_avx_epi32`，不能用会发
   EVEX 的无后缀 intrinsic；13900K 实机已抓到并关闭该 SIGILL 风险。

**正确性**(2026-07-15):
- Win MSVC `hyle_simd_avx2_test` 全量 221 + SumOfMulQuad/Promote* **PASSED**
- WSL GCC `hyle_simd_avx2_test` 全量 221 **PASSED**;`avx_vnni` 位在 cpuinfo 可见
- SDE `-skx` 跑 `hyle_simd_avx512_test` 的 SumOfMulQuad + Promote* **PASSED**(非 VNNI 向量路径)

**剩余证据项**：正式 op_parity `-n15` 重测 SumOfMulQuad（历史账已记
~10×→追平）。两档 VNNI 已是 per-op 产品核，不再是机制缺口。

### P0-3. 大 n 归约的误差档 — **调研后不做(2026-07-15 退回)**

**调研结论(时间主线)**:曾实现 `sum/dot_f32_accurate`(f64 中间累加)并真测:
`dot_f32_accurate` 相对快路径 `dot_f32` **~6× 更慢**(WSL 0.128 vs 0.022 ns/op)。
主线追的是**时间/吞吐**,不是误差阶 → **代码已 revert**(`6e31fcc`/`c63e7d1`)。
softmax 等仍可在算法内手写 double 累加(局部,不扩公开 API)。

**教训**:复杂度清单里「误差 O」项必须先问是否服务时间主线;纯精度档不做默认产品面。

---

## P1:真·工作量/深度复杂度类(需要新算法件)

### P1-1. Scan / PrefixSum 家族 — **已落地(2026-07-15)**

**目标**:深度 O(n) 串行链 → O(n/w·log w)(向量 Kogge-Stone + 跨向量 carry)。Highway/xsimd
无公开 scan → 能力面超越。

**已做**:`include/hyle/simd/scan.h` — `inclusive_scan` / `exclusive_scan`
(u32/i32/u64/i64/f32/f64),多 ISA 核 + dispatch;`Scan.*` 15 tests Win/WSL PASS;
`hyle_scan_bench`(1M u32 ones, in-place reseed)。

**跨 128b KS(已补全,非半吊子)**:
- **禁用** `ShiftLeftLanes` 做 KS(`slli_si256` 半寄存器,算错)。
- **AVX2**:`Slide1Up`=permute2x128+alignr(全宽);32-bit `SlideUpLanes`=vpermd。
- **AVX-512**:`Slide1Up`/`SlideUpLanes` 32/64-bit 改 **valignd/q**(`alignr_epi32/64`),SDE-skx
  手写 KS 检查 `AVX512_SLIDE_KS_OK`。
- carry 用 `BroadcastLane(last)+GetLane`,去掉每块 `ExtractLane` store+reload。

**时间(1M u32, inclusive vs 标量链)**:
| 平台 | scalar | HYLE | 结论 |
|---|---|---|---|
| Win MSVC | 1.48 ns/op | **0.39** | **~3.8× 更快**(SIMD KS) |
| WSL GCC | 0.26 ns/op | SIMD 曾 0.35(~0.74×) | **分平台**:`HYLE_OS_LINUX && HYLE_ARCH_X64`
  固定 pure scalar 链(与最快标量同级,不再亏) |

**分平台依据**(scan_wsl_probe):Linux x64 上 `a[i]+=a[i-1]` 已 ~1 cyc/elem;单流 SIMD KS+carry
测得 0.70–0.84×(更慢)。`scan_dispatch.cc` 用 `HYLE_OS_LINUX`+`HYLE_ARCH_X64` 钉 pure scalar;
Win 仍走 AVX2/512。Arm64 仍 NEON。

### P1-2. 整数键 radix sort + argsort — **已落地(2026-07-15)**

**已做**:
- `radix_sort` u32/i32/u64/i64/f32(LSD base-256,稳定);`argsort` u32/i32/f32
- `sort()` 整数键在 `n >= kRadixSortMinN`(4096) 自动走 radix,小 n 仍 vqsort
- 测试 `RadixSort.*` 9 tests **Win / WSL / Mac 全 PASS**;bench `hyle_radix_sort_bench`
- 实现:`src/simd/sort/radix_sort.cc`(portable)
- **多直方图**(4 路私有 `count[256]` 再 reduce):hist 微基准 WSL **1.08×**;端到端 Win 1M
  约 **+3–4%**(6.58→6.35 ms 量级),WSL 端到端基本持平。

**scatter 块化试过并弃(2026-07-15 同机 AB)**:per-bin `kScatterBuf=16` 暂存 + `memcpy` flush。
Win 1M u32 对照:**直写 6.28 ms / 9.2×** vs **缓冲 7.56 ms / 7.8×**(~20% 税:双写 + flush 分支)。
WSL/Mac 缓冲路径正确性过,但 e2e 不优于直写 → **保留直写 scatter**,多直方图留下。
scatter 仍是 pass 内主瓶颈(~74%),后续杠杆需别的形状(更大块/分区/并行),不是小缓冲。

**时间(u32 random, vs std::sort, 多直方图+直写 scatter;2026-07-15 三平台)**:
| n | WSL | Win | Mac NEON |
|---|---|---|---|
| 1K | ~0.7× | **~1.8×** | ~1.2× |
| 4K+ | **~5–6×** | **~6–9×** | **~2.5–4×** |
| 1M | **~5.6×**(8.2 ms) | **~9.2×**(6.3 ms) | **~3.8×**(4.1 ms) |

阈值 4096 合理:大 n 稳赢 std;小 n 仍给 vqsort。

### P1-3. 流式 top-k — **已落地(2026-07-15)**

**已做**:`partial_sort` 在 `k ≤ 64 && n ≥ 256 && k < n` 走流式阈值路径
(`src/simd/sort/stream_topk.cc`):
- 前 k 排序后阈值=`keys[k-1]`;扫剩余时单次比较拒绝;命中则把挤出的阈值写回 `keys[i]`
  (保持置换/multiset)再插入有序前缀。
- 覆盖全部标量键 + KV32/64;float 用与 scalar 相同的 NaN-after-finite 序。
- `k > 64` 仍走 VQPartialSort / 原 tier 梯。

**正确性**:`Sort.*` 16 tests **Win / WSL / Mac 全 PASS**(含 SmallK KV 置换对拍)。

**时间(u32 random n=1M partial, ns/key;越低越好)**:
| k | 平台 | std | Highway VQ | **HYLE public** | 相对 Highway |
|---|---|---|---|---|---|
| 10 | Win | 0.39 | 0.86 | **0.40** | **~2.2×** |
| 64 | Win | 0.40 | 0.93 | **0.40** | **~2.3×** |
| 100 | Win | 0.45 | 0.91 | 0.90(走 VQ) | ~1.0× |
| 10 | WSL | 0.30 | 0.31 | **0.29** | ~持平 |
| 64 | WSL | 0.31 | 0.34 | **0.32** | ~持平 |
| 10 | Mac | 0.69 | 1.56 | **0.37** | **~4×** |
| 64 | Mac | 0.30 | 0.96 | **0.30** | **~3×** |

Win/Mac 小 k 明显赢 VQ(VQ 仍整表分区写);WSL 上 VQ 已够快→持平不亏。
向量 compress 批过滤未做(标量阈值已吃到主收益);k>64 仍 VQ。

---

## P2:深度/度数级(常数大头,机制是降度数或降依赖链)

### P2-1. exp/log f64 表驱动 range reduction(多项式度数 d → ~d/2) — **已落地(2026-07-15)**

**实现**:`src/simd/math/table_f64.h` 是从 Arm Optimized Routines AdvSIMD f64 `exp`/`log`
派生的 Highway 泛型 N=128 表核(MIT OR Apache-2.0 WITH LLVM-exception，保留 SPDX/版权头)。
它不是移植 ARM intrinsic：AVX2/AVX-512 用 Highway gather，NEON 也走同一按 lane table lookup 语义。
`cloc` 实测新表头为 162 C/C++ code LOC；wrapper 与 ULP gate 保留在既有 TU/测试中，不另建运行时层。
- `exp`: `round(x*128/ln2)` → 128 项 `2^(j/128)` scale table → split-ln2 reduction → 4 次
  `exp(r)-1` 多项式。
- `log`: IEEE 位域精确拆 `x=2^k*z` → 128 组 `{1/c,log(c)}` → 5 次 `log1p(r)-r` 多项式。
- 含 NaN、Inf、log 非正/次正规或 exp 极端 lane 的向量整体回退 `std::exp`/`std::log`；这是 batch
  API 的 IEEE/libm 契约，不把缺 special handling 的 raw Highway row 当可部署路径。

**施工纠错**:先立 `Math.F64ExpLogFullDomainUlp`(特殊值、overflow/underflow 边界、`DBL_TRUE_MIN`、
16K 确定性随机)。两次把 Arm `vfmaq(a,b,c)` 错译成 Highway `MulAdd` 的非 `a*b+c` 形态：先后令 exp、
log ULP gate 失败，均已拒绝；按正确 FMA 参数次序重译后，**Win / WSL / Mac 全部 `exp ≤2 ULP`、
`log ≤1 ULP`**，`Math.*` 三平台全绿。

**五次 same-ISA 直接复测的中位时间**(4096 正常有限输入，`ns/op`；完整表和每轮原始读数见
[`simd-orderswap-matrix.md`](simd-module-closure.md)):

| 平台 | exp HYLE / Highway | log HYLE / Highway | 结论 |
|---|---:|---:|---|
| Win MSVC AVX2 | **0.43 / 0.83** | **0.71 / 0.75** | exp 1.93×、log 1.06× 快 |
| WSL GCC14 AVX2 | **0.44 / 0.83** | 0.80 / **0.76** | exp 1.89× 快；log 直接对手慢 5% |
| Mac AppleClang NEON | **0.70 / 1.35** | **0.97 / 1.37** | exp 1.93×、log 1.41× 快 |

**WSL log 的部署决定**:试过 `HYLE_OS_LINUX && HYLE_COMPILER_GCC` 改走 `hn::Log`，但带相同
exceptional-vector 检查的真实 HYLE wrapper 是 **0.94–1.05 ns/op**，反而慢于表核 `0.74–0.83`。
所以不做伪分流，所有平台保留表核；WSL 对 raw Highway 的 5% 差距留在证据中，不用不满足完整 IEEE
契约的 direct benchmark 掩盖。f32 best-of-breed 不动。

### P2-2. Estrin 多项式求值(依赖链深度 O(d) → O(log d)) — **调研收口:保留 bench、不改核(2026-07-16)**

**先测再决定**:`hyle_math_bench` 新增两层链式 latency mode。纯多项式每步把结果喂给下一步，
消除独立 lane/迭代的 ILP；产品行则把 serial state 广播到**恰好一个真实硬件向量**(不是 `n=1`
标量 tail)，取 lane0 作为下一步输入。链内只用仿射回映射保持 normal-finite，避免 overflow、
scalar libm 或第二个 transcendental 污染读数。`cloc` 实测 `math_bench.cc` 当前为 454 C++ code LOC；
P2-2 latency block 占本轮 185 行 source diff，不另建产品运行时层。

**纯 f64 FMA 多项式链**(Horner / Estrin，`ns/step`，4096-step chain):

| 度数 | Win MSVC AVX2 | WSL GCC14 AVX2 | Mac AppleClang NEON |
|---|---:|---:|---:|
| 7 | 5.31 / **2.75** (**1.93×**) | 5.33 / **2.83** (**1.88×**) | 4.84 / **2.84** (**1.70×**) |
| 11 | 8.23 / **3.48** (**2.36×**) | 8.24 / **3.47** (**2.37×**) | 7.63 / **3.55** (**2.15×**) |
| 15 | 11.18 / **4.07** (**2.75×**) | 11.41 / **3.99** (**2.86×**) | 10.41 / **3.95** (**2.64×**) |

**产品向量链(尺度锚点，不是新旧算法对拍)**:f64 `exp/log` 分别为 Win `18.60/20.25`、
WSL `16.44/18.04`、Mac `13.46/15.20 ns/step`；f32 同样量级。它证明 bench 进入了向量核，
但不能从这些绝对数推导单个多项式的收益。

**不改生产核的原因**:
- Highway f64 `ExpPoly` 已是 `Estrin` (`math-inl.h:1237-1251`)；f64 `LogPoly` 也已按 `x²/x⁴`
  分组，非线性 Horner 长链(`:1313-1328`)。
- P2-1 的上架表核更短：exp 是 4 次、log 是 5 次且已按 `r²` 分组
  (`src/simd/math/table_f64.h`)；没有度 7+ 的 Horner 热点可转换。
- 因此 1.7–2.9× 是**未来出现长自有多项式时的链式延迟上界证据**，不是对现有 exp/log
  的可实现收益。为追数字重写低阶/已分组核会增加寄存器压力与精度回归面，拒绝。

这条 bench 留作 P2-2 的回归闸门：只有新核先在链式行证明收益，才允许讨论 Estrin 改写。

### P2-3. f64 exp/log InRange 分层 — **已落地(2026-07-16)**

**公开面**:`exp_in_range_f64_batch` / `log_in_range_f64_batch`。前者要求每 lane 为
`[-700,700]` 内有限数；后者要求每 lane 为有限、normal、正数(`>= DBL_MIN`)；两者都支持
`src == dst`。同 `hypot_in_range_f32_batch`，这是调用方承诺更窄 domain 的 fast API；实现可在
codegen 获益时移除检查，WSL/GCC 的反例见下。

**实现/精度**:窄域 exp/log 在 ISA codegen 可获益时直接进 P2-1 N=128 表核、跳过
exceptional-vector 检查；scalar tier仍走 libm，WSL/GCC log 的例外见下。`Math.F64ExpLogInRangeUlpAndAlias`
覆盖 16K 正常有限随机、边界、ULP 与 in-place；Win/WSL/Mac 与 WSL SDE `-future` 的 AVX-512
派发均为 `exp ≤2 ULP`、`log ≤1 ULP`。

**五次 same-ISA 中位时间**(4096 正常有限输入，常规 / InRange，`ns/op`):

| 平台 | exp | exp 收益 | log | log 收益 |
|---|---:|---:|---:|---:|
| Win MSVC AVX2 | 0.45 / **0.30** | **1.50×** | 0.71 / **0.56** | **1.27×** |
| WSL GCC14 AVX2 | 0.44 / **0.28** | **1.57×** | 0.81 / **0.81** | 持平 |
| Mac AppleClang NEON | 0.70 / **0.48** | **1.46×** | 0.97 / **0.65** | **1.49×** |

**WSL log 的编译器分流**:直写 no-check table loop 连续五次为 `1.02–1.07 ns/op`，比完整核
`0.77–0.85` 慢约 25%。反汇编显示 GCC 把所有 table/poly 常量跨 gather loop 常驻，寄存器压力
抵消了省下的 check；Windows/Mac 不出现此形态。故 `HYLE_OS_LINUX && HYLE_COMPILER_GCC` 的
`log_in_range_f64_batch` 复用完整 table core：契约仍成立且零回退，其他平台走无检查快核。不是
把一个在 Linux 更慢的"fast" API 硬上架。

---

## P3:其他优化机会

P3 每项须先模拟消费者通常调用方式定出契约，再做端到端 scaling bench；AVX-512 SDE 仅可验证正确性，不作性能证据。

| 项 | 已有基础与机制 | 立项门槛 |
|---|---|---|
| 直方图/分组聚合 | radix 内部已有 4 路私有 256-bin histogram；它不是通用分组 API。`vpconflictd` 可处理 AVX-512 批内冲突 | graph/compute 出现 dense 字节聚合消费者；首项仅 `u8→u64[256]` histogram，验证 `sum(bins)==n` 并测输入规模斜率 |
| 小表 LUT | `TableLookupBytes[InRange/Or0]` 已在 AVX2/NEON/SVE 等后端映射 shuffle/table lookup；仓内没有码本、调色板或量化消费者 | HYLE 自有 int4 dequant 消费者出现；随它做 packed-nibble→16 项 int8 codebook 的端到端路径，对拍标量与 `GatherIndex` |
| 复数 split 布局 + FFT | 现有 SoA complex batch API；`math_scaling_bench` 只测 unary/pow，**未测复数链**。FFT 可将 DFT 的 O(n²) 降至 O(n log n) | DSP/卷积消费者出现后，先定 C2C/R2C、精度、长度、in/out-of-place、workspace、归一化及 CPU/GPU 合同；再以 DFT oracle 和 2^k scaling 验证 |
| crypto 多块流水 + 聚合 GHASH | HYLE 只有 AES/CLMul 原语；TLS 记录加密完全由 OpenSSL 负责。Gueron 聚合可减少 CLMul 归约次数 | 非 TLS 的 GCM 消费者出现，并先定 key schedule、nonce 唯一性、AAD/尾块、tag 验证与失败语义；NIST KAT 覆盖硬件及软件路径 |
| 计数器型 RNG(Philox) | `Rng` 有 `jump()`，SIMD 有按 lane 批量 `VectorXoshiro`；当前仅测试/bench 使用 | 出现 dropout、采样、Monte-Carlo 或 GPU 对齐流，且要求按元素 index、任务切分或 worker 数变化仍严格复现；届时由消费者定义 counter/key 合同 |

---

## 明确没有复杂度机会的(别投入)

- **dot/saxpy/逐元素单遍形状**:已在带宽墙(dot 42G 平真 SOTA)——单遍形状复杂度已最优;
  它们的机会在 P0-1 的**多遍组合形状**(pass 融合),不在核本身。
- **bitpack**:线性流式已最优(借 Highway 核)。
- **255-op parity 层**:常数仗已清零(矩阵三平台全零),回头抠是负收益;该层剩余价值是
  防回归监控。

## 落地纪律

1. **每项立项第一步是建 scaling bench(n 从 1e3 扫到 1e8),不是写核**——复杂度类主张只能
   用斜率/交叉点证明,单点 ratio 证明不了 O(n) vs O(n log n)。bench 骨架直接复用 op_parity
   引擎的钉核+副本+锚点纪律(方法论文档 §1/§2)。
2. radix/top-k/scan 的**交叉点与适用区间**要像 orderswap-matrix 一样成为文档里的可信数字;
   公开面若做自动分派,分派阈值即来自该实测。
3. 精度敏感项(P2-1/P0-3)先立 ULP/误差 oracle 再动核;误差档命名沿 `*_accurate`,
   前置条件档沿 `*InRange`(Highway 官方词汇)。
4. P0-2 注释纠错 + 标量洞关闭已落地(2026-07-15);残留 runtime VNNI tier 见上节。
