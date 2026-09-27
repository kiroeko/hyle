# D5-F47：CPU 官方混合精度片段边界

<!-- MODULE REVIEW: AUDITED -->

- baseline：`9e7830c291e6a2cc9a50a039e088a21ee35ce209`，Windows canonical。
- owner：当前 [D5](llm-moe-d5-real-fragment-acceptance.md) 的消费者验证；产品 FIX 为零，
  不重开 CPU/oneDNN 已关闭 owner，不改变默认路线、数学、格式或精度合同。
- 需求：GLM-5.3/Flash 官方 FP8、DeepSeek V4 Pro/Flash 官方 FP4/FP8 的完整 MoE。
  已有预编码 MatMul 拒绝与单库转换探针不足以证明这四个真实配方的普通入口边界。

## 有界源码复核

只读已经登记的固定源，不下载模型或变更依赖：

| 成熟来源 | 已确认事实 | 裁决范围 |
|---|---|---|
| oneDNN `71094df451909e95ceeefc8f64c275cdcbbcdff5` | `jit_uni_reorder.cpp` 的 FP8 资格要求 AVX512_CORE_FP16 或 AVX10.2；`float8.cpp` 的 F32 reference 经 F16 再转 E4M3 | 13900K 上不能靠打开既有快速路径解决 F45 的二次舍入反例；不外推所有 CPU 实现缺失 |
| SGLang `06992c92ad4494fe9f9636e75be3e35040512447` | `Fp8LinearMethod` 的 CPU 分支要求 AMX，调用 `fp8_scaled_mm_cpu`；CPU 实现使用未作 E4M3 动态量化的 BF16/F16 激活 | 不能原样绑定为本机的官方 A8W8 合同 |
| vLLM `51918e252af4dfff381001f42b7ebe6a0454501b` | `sgl-kernels/gemm_fp8.cpp` 接受 BF16/F16 activation 和 E4M3 weight；`cpu_attn_fp8.hpp` 的 scalar KV 编码取 mantissa 高位，并非所需 nearest-even | 两个具名程序不能代替本轮动态 FP8；不判定整个 vLLM 没有其它可用代码 |
| 已接入 ggml CPU / oneDNN adapter | 未绑定 `Float8BlockLinear`、`GroupedFloat8Linear`、`GroupedFloat4Linear` 的完整官方合同 | 需要普通入口记录实际首个拒绝，不能用 SIMD 冻结自研回填 |

## 有限验证与停止

1. 原样复用已在 CUDA 运行的 GLM `glm_moe`、DeepSeek `moe_chain` 完整配方，保留真实
   内宽、top-k、逐角色编码和原版激活量化；仅复用既定的 resident expert 缩容。
2. Direct 按原配方逐节点准备，记录第一个 `Unimplemented`，要求它确为官方 FP8/FP4
   语义节点；不能把其它错误算作预期拒绝。Graph 提交同一个完整图和真实常量绑定，
   记录 `compile` 返回值。所有先前准备的资源正常销毁，不执行未准备成功的图。
3. 一个 tests-only 候选，在现有 CPU 构建树低并行增量构建，两入口四画像一次批验。
   只在明确编译/测试接缝错误时允许一次纠正，不做性能采样或重复 F45 原生探针。
4. `BACKEND_REJECTED` 只表示当前具名版本/机器的该必要片段不能由普通路线完成；
   不是 `RUN_PASS`、不是所有上游缺失、不是整模型已运行。其它 region 的既有收据保留。

若真实拒绝确认，停止在 D5 中试拼这几种不等价程序；另列最小成熟承载需求：CPU 原生的
直接 RNE E4M3 编解码、group128 动态量化、按 K-block scale 累加，以及 DeepSeek A8W4
和 group32 FP4 floor。允许未来资格化具名成熟转换/完整程序，禁止为通过测试改为 W8A16、
改用 GGUF 画像或恢复 HYLE 数学。新 Provider/编译 runtime 需单独定界；本卡不授权接入。
这八个 CPU 格不能贡献支持完成率。已有可用 CPU 画像的 A5/A6 与其它路线继续独立推进。

预声明检查：本批两入口测试、模型输入/ledger 检查、selector 所选 docs/closure。
本卡不改公共头、产品、组件图或交付，故不因本卡重跑三交付或 P1。
