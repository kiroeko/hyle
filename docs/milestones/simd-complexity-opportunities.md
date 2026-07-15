# SIMD 复杂度级优化机会清单(2026-07-14 实证审计)

> 结论先行:255-op parity 层的**常数因子之战已经打完**(三平台真输清单全零,见
> [`simd-orderswap-matrix.md`](./simd-orderswap-matrix.md)),继续在 op-vs-op 层抠常数是负收益。
> 还能**超过现在表现**的机会全部在算法件层,且必须按"复杂度轴"逐一审:本文按五个轴
> (工作量 O / 依赖链深度 O / 内存流量 O / 误差 O / 指令类)盘点,每项附仓内证据、算法、预期
> 量级、SOTA 参照与消费者。姊妹文档:[`simd-gap-inventory.md`](./simd-gap-inventory.md)(op 面
> 覆盖缺口,正交视角);立项后的测量纪律见
> [`../performance/simd-op-benchmark-methodology.md`](../performance/simd-op-benchmark-methodology.md)。
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

### P0-2. AVX-VNNI 通道 + 修 avx512 SumOfMulQuadAccumulate 的标量洞 — **正确性/洞已落地(2026-07-15)**

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

**正确性**(2026-07-15):
- Win MSVC `hyle_simd_avx2_test` 全量 221 + SumOfMulQuad/Promote* **PASSED**
- WSL GCC `hyle_simd_avx2_test` 全量 221 **PASSED**;`avx_vnni` 位在 cpuinfo 可见
- SDE `-skx` 跑 `hyle_simd_avx512_test` 的 SumOfMulQuad + Promote* **PASSED**(非 VNNI 向量路径)

**机会残留**(未做,避免把 VNNI 指令编进无 VNNI 的 SKX 产品核):
- **runtime AVX-VNNI tier / 函数内分支**仍未做——现在只有 compile-time `#if` + 探测位;
  要 1-op native 吞吐需:独立 `AVX2_VNNI` 核 TU(`-mavxvnni`)或 ifunc/指针,再按 `avx_vnni` 选。
- 默认 `AVX512` 旗标仍无 `-mavx512vnni`(SKX 安全);VNNI 512 路径只在带该宏的 TU 生效。
- 正式 op_parity `-n15` 重测 SumOfMulQuad(历史账已记 ~10×→追平;本轮是正确性闭环)。

### P0-3. 大 n 归约的误差档 — **调研后不做(2026-07-15 退回)**

**调研结论(时间主线)**:曾实现 `sum/dot_f32_accurate`(f64 中间累加)并真测:
`dot_f32_accurate` 相对快路径 `dot_f32` **~6× 更慢**(WSL 0.128 vs 0.022 ns/op)。
主线追的是**时间/吞吐**,不是误差阶 → **代码已 revert**(`6e31fcc`/`c63e7d1`)。
softmax 等仍可在算法内手写 double 累加(局部,不扩公开 API)。

**教训**:复杂度清单里「误差 O」项必须先问是否服务时间主线;纯精度档不做默认产品面。

---

## P1:真·工作量/深度复杂度类(需要新算法件)

### P1-1. Scan / PrefixSum 家族(深度 O(n) → O(n/w·log w);Highway 没有的能力)

**现状**:op 面与算法层均无任何 scan(已 grep 确认);Highway/xsimd 也不提供公开 scan——
做成即**超越参考库的能力项**,不是追平。标量前缀和的本质瓶颈是串行依赖链(≈1 elem/cycle 封顶)。

**机会**:向量内 Kogge-Stone(log w 步 shift+add)+ 跨向量 carry 广播,实测这类实现对 u32/f32
有 **4~6×**。消费者现成:CompressStore 偏移流、CSR/直方图前缀、compute 的 cumulative 类。
建议面:`inclusive_scan/exclusive_scan`(u32/i32/u64/f32/f64),泛型层组合即可(Shift/Add/
Broadcast 全有),SVE/RVV 有原生倾向指令可后续特化。

### P1-2. 整数键 radix sort + argsort(O(n log n) → O(n·⌈k/8⌉) 线性)

**现状(include/hyle/simd/sort.h)**:sort/partial_sort/select 全部借 Highway vqsort =
**比较排序 O(n log n)**;argsort 无面(文档教用户"自己拼 key-index 对")。

**机会**:u32/u64/f32(键翻转技巧)的 LSD radix,u32=4 遍线性,**大 n 时按构造超过 vqsort**
——vqsort 论文自己承认的边界,Intel x86-simd-sort/IPS4o 在整数键大 n 同走 radix。附带
argsort(64 位 key|idx 打包 radix)与稳定性(radix 天然稳定,补上 vqsort 缺的稳定档)。
Highway 无 radix contrib → 自造(章程:借无可借)。**交叉点必须实测标注**(预期 n≈1e5~1e6
分界,小 n 仍归 vqsort,公开面做成按 n 自动分派或双入口)。代价:O(n) 暂存 + 直方图。

### P1-3. 流式 top-k(小 k 大 n:全数据 partial_sort → 均摊 O(n) 过滤)

**现状**:top-k 走 `partial_sort`(vqsort 分区,触碰并重排全部数据)。

**机会**:单遍"阈值向量比较 → 罕见命中才 compress 追加 → 缓冲满时重建阈值",随机数据下
绝大多数向量批一条比较跳过,均摊 O(n) 且只读不写主数据。k≪n(top-10~100 of 1e7,检索/
beam search 形状)预期数倍。KV top-k(sort.h 已有 KeyValue32/64)同法受益。

---

## P2:深度/度数级(常数大头,机制是降度数或降依赖链)

### P2-1. exp/log f64 表驱动 range reduction(多项式度数 d → ~d/2)

**现状**:f64 转录函数全列"借 Highway 赢 xsimd"(benchmarks-simd.md),Highway 核是纯多项式。
**机会**:2⁻ᵏ 段表 + 低阶多项式(SVML/vector-libm 路线),f64 exp/log 预期 1.2~1.5×。
按章程走:先建精度闸门(ULP 对拍 oracle)再自写;f32 列已是 best-of-breed 不动。

### P2-2. Estrin 多项式求值(依赖链深度 O(d) → O(log d))——先补 latency bench 再决定

吞吐 bench 看不见它(迭代间 ILP 掩盖);只有 **latency-bound 链式消费者**受益。立项前置:
给 math bench 加"链式延迟"模式(输出喂下一次输入),量出 Horner 深度差,有差再谈改核。
这也是把"要证 scaling/复杂度,非单点 ratio"的纪律落到 math 侧。

### P2-3. 数学函数 InRange 分层

`exp_in_range/log_in_range`(调用方保证无 overflow/denormal/非正数)砍掉边界处理 blend 层,
house 惯例(InRange 家族)现成,顺手级,预期 10~20%。

---

## P3:消费者/硬件驱动的候补

| 项 | 机制 | 触发条件 |
|---|---|---|
| 直方图/分组聚合 | AVX-512 `vpconflictd` 把批内冲突解决从最坏串行 O(w) 降到 O(批内 distinct);AVX2 字节切片私有桶 | graph/compute 出现聚合消费者;SDE 可验 |
| 小表 gather→vpshufb LUT | 表 ≤16/32 项时 shuffle 查表比 vpgatherdd 快 5~10×(gather 是慢 uop) | 出现小表 gather 消费者(量化码本/调色板) |
| 复数 split 布局 + FFT | split-complex 链式运算免逐 op shuffle(math_scaling_bench 正测复数链);FFT 是 O(n²)→O(n log n) 的终极项 | DSP/卷积消费者出现;FFT 无消费者不启动 |
| crypto 多块流水 + 聚合 GHASH | AES 4~8 块 in-flight 隐藏延迟;Gueron 聚合归约 O(n)→O(n/4) 次 CLMul 归约 | HYLE 只有 AES/CLMul 原语无 GCM 组合(io TLS 走 OpenSSL);等 GCM 消费者 |
| 计数器型 RNG(Philox) | 流创建 O(jump) → O(1),任意寻址 | 大规模并行流消费者;xoshiro 现状够用 |

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
