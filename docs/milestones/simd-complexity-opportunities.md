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

### P0-1. softmax/log_softmax 的 pass 融合(内存流量 5n → 2n)

**现状(src/simd/math/activations.cc:64-71,注释自认)**:softmax = 标量 max 折叠 → subtract
一遍 → exp 一遍(向量)→ 标量 sum 折叠 → divide 一遍,**共 4~5 次过数据,其中两次是标量折叠**;
注释同时写明消费形状 "vocab ~128k"——正是内存流量主导的尺寸。

**机会**:Online softmax(Milakov–Gimelshein 2018)把 max 与 sum 融合成单遍在线更新
(`m'=max(m,x); s'=s·exp(m-m')+exp(x-m')`),加一遍归一化输出 = **2 遍全向量**。
逐元素核都顶在带宽墙上,**减少遍数是唯一能"穿墙"的复杂度改进**:流量 5n→2n,预期 2~3×,
n 越大越接近流量比。log_softmax 同法。

**SOTA 参照**:oneDNN/FlashAttention 世系的 online softmax;标量折叠向量化部分可直接复用
ReduceMax/ReduceSum 已有核。消费者:activation_bench 在册 + compute 的 LLM 形状。

### P0-2. AVX-VNNI 通道 + 修 avx512 SumOfMulQuadAccumulate 的标量洞(指令类)

**现状,两处铁证**:
- `include/hyle/simd/backend/avx2.h:957` 注释 **"No AVX2 VNNI (native vpdpbusd is AVX-512-VNNI)"
  已过时**——Alder Lake 起存在 VEX 编码的 **AVX-VNNI(256-bit vpdpbusd/vpdpwssd)**,与 AVX-512
  无关;**开发机 13900K 的 cpuinfo 实测含 `avx_vnni`(原生与 WSL 双确认)**。当前 int8 点积族
  (SumOfMulQuadAccumulate/WidenMulPairwiseAdd 世系)在 avx2 tier 跑 2~3-op 模拟链
  (pmaddubsw+pmaddwd+add),native 是 **1 op**。
- `include/hyle/simd/backend/avx512.h:798-799`:SumOfMulQuadAccumulate 因 "vpdpbusd 是 u8×i8
  signedness 不对" 退到 **correctness-first store+reload 标量循环**——这是现存面上的洞,不是
  新增需求。标准修法(oneDNN/ncnn 同款):XOR 0x80 偏置转 u8 + 补偿项(`sum -= 128·Σb`,可预计算
  或用第二条 vpdpbusd 对全 1 向量),vpdpbusd 承载 i8×i8。

**机会**:量化点积吞吐的**指令类跃迁**(3-op→1-op;标量洞→native 差距更大)。需要:
cpu_features 加 avx_vnni 位 + dispatch tier(AVX2+VNNI 介于 avx2/avx512 之间,或作 avx2 内
运行时分支)+ SDE `-future` 与本机双验证。消费者:compute 的量化 kernel(vitamin-cuda 参照面
的 CPU 侧)。

### P0-3. 大 n 归约的误差档(误差 O(n·ε) → O(log n·ε),速度不变)

**现状**:ReduceSum/dot 直线累加,误差界 O(n·ε);softmax 里已经出现 double 累加兜底
(activations.cc:71 "double accum: n can be large (vocab ~128k)")——说明误差压力是真实的。

**机会**:pairwise(树形)求和误差 O(log n·ε)、块内 Kahan 误差 O(ε),**吞吐几乎不变**
(4-累加器结构天然就是 4 路树的第一层)。提供 `reduce_sum_accurate/dot_accurate` 变体,
沿 InRange 家族的"消费者自选"惯例。这是"误差增长阶"意义上的复杂度改进,不是速度。

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
4. P0-2 触发一次**注释纠错**:avx2.h:957 的 "No AVX2 VNNI" 需随修复一并更正(文档=当前真相)。
