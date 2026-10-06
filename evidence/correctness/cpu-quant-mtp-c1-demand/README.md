# C1 实际权重需求与成熟 CPU 承载

这是存储/源码审计证据；不是 HYLE 接线、模型执行或性能 PASS。
当前完整度以 [矩阵](actual-storage-matrix.json) 的 `counts`、逐制品 `missing_files` 为准。
原始头记录与固定 URL 保存在同目录 `.json.gz`，不包含完整模型权重。

## 格式纠正

- Q09–Q11 实际是 Unsloth IQ1_XXXS/XXS/XS：type66/65/64，38/42/46 字节每256值。
  原始枚举、block、grid、dequantize 和 CPU dot 位于
  [固定原版扩展](https://github.com/unslothai/llama.cpp/tree/46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f/ggml)。
  不能映射为 stock Q1_0/TQ1_0/TQ2_0。扩展来源0.22，CPU core候选0.24；只借入具名推理代码。
- Qwen命名Q4_0的MTP含18 tensors，实际为Q3_K/Q4_K/Q6_K/F32；完整分支仍需C4资格。
- 官方DeepSeek新版本均有`mtp.*` DSpark权重。main projection为F8_E4M3＋F8_E8M0 scale；
  Markov与confidence源权重为BF16，confidence执行遵循官方F32边界。
- Flash MLX制品混用mxfp4/group32、mxfp8/group32、affine int4/group64；不能只接affine。
  完整index/header没有MTP/DSpark/Markov/confidence权重，不能制造该制品的分支资格。

## 有限成熟承载裁决

| 族 | 固定成熟来源 | 接入裁决与边界 |
|---|---|---|
| GGUF 11种格式 | stock ggml ffa4与Unsloth原IQ1扩展46cb | BORROW；压缩MatMul、MUL_MAT_ID及实际需要的GetRows，保留原字节 |
| NF4＋double quant | bitsandbytes 833649，原default/CPU实现 | BORROW原CPU解码或原ATen组合；高nibble在前，F32 codebook/scale计算后cast；二级scale＋offset独立输入 |
| AWQ GEMM int4 | AutoAWQ 88e4c76，原packing_utils | BORROW原列打包/逆排列；group128，保留原zero-point；不得把文件直接看成连续K方向int4 |
| compressed-tensors int4 | 0.18.0 ac8e2ba，原pack/decompress | BORROW原dense packing，unsigned存码减8；group128及真实g_idx约束；保留BF16未量化MTP |
| MLX混合量化 | MLX 2b2bb497，原QuantizedMatmul/GatherQMM CPU | BORROW原affine/fp_qmm CPU计算；保留scale/bias、浮点量化和原dtype边界；不能用有舍入损失的bias＋8scale冒充等价 |

具名来源的URL、SHA、完整小源码快照在 [来源身份](mature-source-identities.json) 与
`mature-source-snapshot.json.gz`。选择成熟承载不等于它已在HYLE具备交付、OOM或replay资格。
这些边界由C3一次批量实现并实际验证；目前没有批准自研数学核。

## 原库 int4 探针

固定LibTorch2.8 CPU的原 `_convert_weight_to_int4pack_for_cpu` /
`_weight_int4pack_mm_for_cpu` 在Windows通过24案例：3 dtype ×4 group ×2 M，N32/K256。
见 [执行日志](aten-3-original-aten-int4-execution.log) 与 [收据](receipt.json)。
只证明该原库接缝实际可运行；不授予NF4/AWQ/MLX制品或HYLE RUN_PASS，也不签SOTA。

## 实施约束

1. 存储、activation、accumulation、state、output分别记录；头只证明存储，forward和原执行程序证明计算。
2. 模型loader负责读文件/quant-state；HYLE接受checked typed tensor/scale/codebook/zero-point端口。
3. prepare绑定程序/配置；slot持有工作区、状态和完成责任；共享不可变权重保活至最后一次完成。
4. 不允许每轮展开全库专家；必要解包按原程序消费粒度及预声明资源合同执行并记录成本。
5. 单一合格成熟承载正确接入/P0；同合同多家成熟重叠域再批量比较，不恢复自研竞赛。

## CPU core 升级审计

候选 [stock身份](stock-ggml-cpu-identity.json)：ffa4e8b8，archive SHA
`2adae8335f09b75d05158ee32104f0b562929b8470e6d5858ecb513d325636f4`。
现有三文件资源补丁的所有13个before上下文在候选中各出现一次。
CPU-only用新版core＋原IQ1推理扩展；GPU-containing构建使用原456172源和原补丁。
一个SDK只选择一个core，baseline/AVX2完整CPU程序使用同源、同patch身份。
旧Native优势规则的软件peer改变时不继续应用旧WIN；只在准备期固定这一选择。
