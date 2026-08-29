# SIMD-08：Transcendental 与 activation

- ticket state：`ACTIVE / AUDITED`
- kind：product capability/performance
- depends on：SIMD-07
- budget：1 工程日；性能预算 60 分钟
- input commit：`0720f25432ccfa7697ad927f3d0bf37d72318615`

归属当前两个 activation 草稿头。审查 real/complex transcendental accuracy tier，以及 GELU、tanh-GELU、
sigmoid、SiLU、ReLU/leaky-ReLU、ELU、hard-sigmoid/hard-swish、softmax/log-softmax 的 allocation、alias、
out-of-place、NaN/Inf 和 reduction 合同。该有限前向集合由 ggml 与 oneDNN 的共同任务证明；Mish、Softplus、
backward 和 fused norm/GLU 不在本票。接受实现必须 O(n)、O(1) extra storage、无隐藏 heap/lock，并完整接入
target/runtime/build/test；否则删除草稿。

性能只测资格化的 exp、tanh-GELU、SiLU、softmax normal row；模型特化 fusion 仍由 compute 拥有。

SIMD-06 的 focused scalable dispatch 验证发现：现有 full SVE/RVV harness 中
`scalable_math_product_test.cc` 仍引用迁移前的 `tgamma/fmod/nextafter_*_hyle_{sve,rvv}` symbols，已不能
完成编译。该事实属于本票的 transcendental test inventory；先按当前 vendor route/实际 symbol 修正或
删除过时断言，再运行 full scalable family gate。`hyle_math_scaling_bench` 也尚未取得 direct Highway
math header 的 include/target contract，当前 `HYLE_BUILD_BENCH=ON` 会在
`hwy/contrib/math/math-inl.h` 缺失处停止；同在本票校准后再作性能证据。它们不授权 SIMD-06 改
transcendental 产品或扩大仪器。

## 资格化审计

- 当前四个 elementwise activation 都在 `noexcept` 热路径构造一个 O(n) `std::vector`，随后调用一次
  public transcendental batch，再做一遍 scalar combine；这带来 hidden heap、两到三次整段流量和第二次
  runtime dispatch，直接违反 closure card 的 O(n)/O(1)、no-allocation 和每 batch 一次选择合同。
- 当前 softmax 只有 in-place spelling，max 仍为 scalar，exp 又按 4096 元素人为切块并重复进入 public
  dispatcher；log-softmax 完全使用 scalar exp。它们不能承担 ggml/oneDNN 的普通 out-of-place row，也没有
  明确 NaN、Inf、all-masked、alias 或部分 overlap 合同。
- installed `math.h` 缺 ReLU、leaky-ReLU、ELU、hard-sigmoid 和 hard-swish；llama.cpp
  `7584430716ee229751771ed0d6bbcb780d105eeb` 的 CPU row 与 oneDNN
  `5dc67b7f37b1215d8c1f054e99c1b7cfd40317a9` 都包含这组前向 family。ggml 同时证明 out-of-place
  softmax/log-softmax、exact/tanh GELU、sigmoid/SiLU 的真实 row 形状；这不是 benchmark symbol 并集。
- 两个未跟踪 activation 头正确识别了“一个 target-bound loop”方向，但仍 include 已由 SIMD-04 删除的
  `math_xsimd_hyle-inl.h` / `math_hyle-inl.h`，并重新依赖 migration-only `vec.h`。因此裁决为
  `REWORK`：保留其无分配算法意图，代码不能原样进入候选。
- 锁定 Highway `f503c485cec6b4f2f596fa60d83484797c9eeb5c` 已直接提供 Exp/Expm1/Tanh/Erf 与
  load/store/reduction；现有 `highway_math_kernel-inl.h` 已是 AVX2/AVX-512/NEON/SVE/RVV 的唯一
  target-generated math seam。activation 应在同一 TU 直接组合这些原语，不再添加 vendor runtime 或第二套
  target DSL。
- SIMD-04 只在孤立 tanh valid-domain 行观察到 xsimd 约 4–9% 优势、1 ULP 对 Highway 2 ULP；这只是
  activation 的机制假设。只有本票同合同 tanh-GELU/SiLU/softmax consumer 复现 material 优势才允许重新
  引入第二产品 vendor，否则 xsimd 继续只作 accuracy/performance peer。
- 现有 `activation_bench` 把 RNG 混入本 family、只和 scalar loop 比较、没有 A/A/反序/布局控制；它是
  HISTORICAL/DIAGNOSTIC，不能签本票。`scalable_math_product_test` 仍引用已删除的 `*_hyle_sve/rvv`
  math port symbol，`math_scaling_bench` 没取得锁定 Highway include/target；两者是已复现 build 缺陷。

## 冻结合同与有限 FIX 集

1. 保留 exact GELU、tanh-GELU、sigmoid、SiLU，新增 f32 ReLU、leaky-ReLU(slope)、ELU(alpha)、固定
   `(x + 3) / 6` hard-sigmoid 与 hard-swish。全部采用 `src[, parameter], dst, n` pointer/count 形状，
   `n == 0` 可传空指针；exact `src == dst` 受支持，其他部分 overlap 未定义。Mish、Softplus、backward、
   fused norm/GLU 和 tensor stride/threading 仍不准入。
2. softmax/log-softmax 新增 `const float* src, float* dst, n` 核心，原地 spelling 只作 inline convenience；
   exact alias 或 disjoint，O(n)、O(1) extra storage、无 heap/lock/copy。finite row 采用稳定 max-shift；
   `-inf` mask 与至少一个 finite lane 正常工作；任意 NaN、`+inf` 或 all-`-inf` 使整行输出 quiet NaN，
   明确暴露未定义 normalizer，不静默制造概率。elementwise NaN 传播；ReLU 把 signed zero 归一为 +0。
3. 在现有 direct-Highway math target TU 中生成唯一 activation body，scalar libm 为总是可用 floor；普通
   façade 共用 math family 的已缓存 target，一次整段选择后直接进入 kernel。删除两个旧草稿头，不增加
   Runtime public surface、常态 metadata、逐元素 callback 或第二 vendor ABI。
4. 扩展 property/differential 覆盖 zero/tail、exact alias、disjoint out-of-place、参数、wide finite 域、
   NaN/±Inf/all-masked、概率和与 log-softmax identity；修正 full SVE/RVV math inventory 并让同一 scalable
   binary 跨多 VL/VLEN 验证 activation。修复 math scaling target 的 direct Highway include/ISA 合同，
   source/static/shared 和 public-header journey 同步验证。
5. P1 只测 normal f32 exp、tanh-GELU、SiLU、softmax 的 4096 与适用 streaming row。candidate 与 direct
   Highway/HYLE composition、ggml-shaped peer 或 xsimd 必须同 ISA、同 TU、同 accuracy/alias/dispatch/
   resource 计费，正反顺序且 A/A 合格；5% 为 material margin。若 xsimd 只在不同近似/精度合同获胜，写
   `NOT_COMPARABLE`，不得按一行 benchmark 重引产品依赖。

本票只允许上述五项。activation backward/fusion、半精度 activation、attention mask/scale/bias fusion、
online/blocked softmax 和端到端 tensor operator 均路由 Compute；不得在本票扩张。
