# CPU SDPA、Copy、RMSNorm 新版成熟 peer 复比

2026-10-07，Windows canonical；产品 baseline `d283aae73b5547fdd2833ef1bf43ab3d00c729ec`，
benchmark candidate `db84a6faf21646c8b17b5bed137efd1bf45cf5bd`。产品实现与最近关闭候选 c740 相同。
[冻结任务](../../../docs/implementation/tasks/compute/cpu-three-operator-recomparison.md)，
[逐格结果及四个 CI](receipt.json)，完整 manifest/observations/controls/validators/原始前置及实际 exe 在
[qualified.zip](qualified.zip)。SDK、全部 DLL 与 loaded `.text` 身份在冻结 benchmark 的 preflight/SDK 记录。

## 资格和边界

- W1-WIN / i9-13900K / MSVC19.50 / installed static / Release / LTO off / 单 P 核 CPU2 / above-normal。
- 同一公共操作、共享不可变输入与完整输出、每库一线程、完整 submit+wait；准备在计时外。
  普通无限预算，禁止 alias；没有把严格资源或动态整域保证混进性能。
- Copy 对当前 oneDNN3.13.2；F32 RMS 对 oneDNN，低精度 RMS 和 SDPA 对 CPU-only ggml
  `ffa4e8b80930029a35991f94e7c8a93cd67730ab+hyle-resource-iq1-bdae1e43311ee486`。
  这是当前默认已资格化接法的比较，不是所有潜在原库程序的全球冠军。
- 24/24 数值对照与 fresh workload A/A 通过；四执行/分配 strata，每格每 stratum 15 个独立进程。
  60 product + 30 controls，唯一 P1 总时间 595.578 秒，无 n31、无优化、无追加采样。
- controls AA/delta5/delta10/timer-floor 与 artifact/build/workload-AA validators 全 PASS。
  公共 manifest verify 通过；资格 **QUALIFIED**。18 WIN、6 LOSS，均在预固定 5% practical 门外，
  因确有六个 Native 劣势，聚合 performance **FAIL**；不改门，不把它写成正确性失败。

## 实际结论

表中倍数是四 strata 中位数范围；CI 和全部24个参数 ID 在 receipt。不能只用下面的点值做新路由。

| 子域 | 原/新明确参数 | Native 相对当前成熟完整调用 | 本轮处置建议 |
|---|---|---|---|
| dense Copy，10格 | F32/F16/BF16/两FP8 storage，`[1或32,2048]` | 快 6.85–7.31 倍 | 有保留依据；实际委托 Mem range-copy，不是另写 memcpy 算法 |
| strided Copy，2格 | F32，同shape，byte strides `[16384,8]` | 慢 4.26–7.21 倍 | 普通这些子域用 oneDNN；不以 dense 的 WIN 代签 strided |
| F32 weighted RMS，1格 | `[1,2048]`，gamma2048，epsilon1e-5 | 快 2.64–2.70 倍 | 有精确小形状保留依据 |
| F32 weighted RMS，1格 | `[32,2048]`，其它同上 | 慢 1.67–1.69 倍 | 普通这个子域用 oneDNN |
| F16/BF16 weighted RMS，6格 | rows1/32、hidden2048；epsilon1e-5，另BF16 epsilon1e-6 | 快 4.37–4.97 倍 | 有完整调用保留依据；低 dtype 原库转换/物化也已计费 |
| F32 equal-head SDPA，2格 | BSHD，B1/H4/D64/KV128，Q1/32，非causal、mask none、scale.125 | 慢 1.65–10.16 倍 | 原大查询约10倍劣势重现，小查询本次也明确落后 |
| F32 causal SDPA，1格 | B1/H8/D256/KV128/Q1，mask none、scale.0625 | 快 2.16–2.18 倍 | 仅签相对当前组合接法的 WIN；下述成熟融合接缝缺口阻止最终 KEEP_WIN |
| F32 causal SDPA，1格 | 同上但Q32 | 慢 3.63–3.72 倍 | 普通这个子域用成熟实现 |

## 有界源码终审：SDPA 的潜在成熟融合接缝

当前 `src/compute_provider_ggml/ggml_cpu_provider.cc::lower` 行824只有 `F16 && causal` 进入
`ggml_flash_attn_ext`；本批 F32 是 matmul→scale→causal mask→safe_softmax→V cont→matmul。
mask 持久准备行1627也限定 F16。当前比较确实完整计费了这套程序，但不能认为这就是原库最好程序。

锁定 CPU-only core 的 `src/ggml-cpu/ops.cpp::ggml_compute_forward_flash_attn_ext_f16`
行9509明确允许 F32/F16 K/V；入口行9602支持 `GGML_PREC_F32`，使用 F32 accumulators；
`src/ggml.c::ggml_flash_attn_ext` 行5542及 `ggml_prec_set_acc` 行3338提供原版 API。
符号带 `_f16` 不等于实际只支持 F16。这是可定位的 adapter 缺口，尚未资格化新公共程序；
不能仅凭源码将 fused 宣称正确性 PASS 或更快，也不能以本批2.17倍抢先签长期保留自研 SDPA。

后续必须有限接入/资格化原 F32 fused 程序、保持实际 mask/布局/精度/资源成本，再裁决 Native 的
该小子域。当前不删除 SDPA、不启用新默认 WIN 规则，不扩自研优化调查。这一缺口不抹掉本批
相对实际已交付组合接法的合格比较，也不自动使其它23格失格。

## 资源能力与解释

Native Copy/RMS 的非 exact compiled-root 只限当前已验形状域；成熟三个 factory 的本地接缝仍 exact-only。
Native invocation 的额外私有内存=0，但描述/编译制品及 caller workspace 非零；成熟 private bounds 未知。
这些差异不是“成熟库缺这个数学算子”，也不能替所有劣势子域签 UNIQUE。当前没有新资源实跑证明；
未变源码与原关闭资源证据复用。两侧 ordinary invocation 本批实际执行。

机制解释由源码支持，非硬件 counter 新归因：dense Copy 直接复用 Mem，而 oneDNN 使用字节维度 reorder
及映射/原库 handle 绑定；strided Native 逐元素求 logical offset。RMS 小行避开原库固定开销，低精度
Native tile decode/F32 math/最终 round 对比 ggml 显式 F32 图转换；大行 oneDNN 摊薄固定成本。
SDPA Native 的 dense Q/K 用 SIMD dot，逐 query 做 tiled softmax/value；成熟组合法以矩阵处理批量，
单 query 仍承担中间结果和V物化。成熟 F32 fused 尚未计时，不能把该解释当成其性能判定。

统一 Compute+Graph 实现仍 **61,499行 /250文件**，不含测试/vendor/benchmark；本轮无产品代码变化。
原始失败/成功输入保留，不修改旧187域证据或旧 GPU 状态，不宣称整模型/Graph融合片段收益或 SOTA。
