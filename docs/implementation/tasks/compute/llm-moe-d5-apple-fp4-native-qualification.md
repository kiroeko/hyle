# D5-F46：Apple 官方 FP4 成熟原生边界

<!-- MODULE REVIEW: AUDITED -->

- baseline：`2c0c3f09d`（完整 commit 由 Git 解析）；Windows canonical。
- 当前 Apple owner 仍按 F45 AUDITED；本卡先做原生资格，产品 FIX 为零。
- 目标：DeepSeek 官方 E2M1/F32 group32 专家、group128 FP8 激活，以及 indexer
  group32 FP4 roundtrip。不能用额外 Q8 激活转换或 BF16 权重替代官方数值程序。
- 既有事实：F45 的 MLX MXFP4 普通256元素通过，极小256元素194个不符。
  该失败保留，不能外推全部原生 FP4 能力，也不能先扩大输出容差。

## 有限原生调查

1. 复用 MLX `1f8e74e3f12f31365464a6867c6579f0e9b29d85` 和现有 M4 build。
   核对 `fp_quantize/fp_dequantize/fp_quantize_dequantize`、E2M1 和 E8M0
   原版转换。与已登记 DeepSeek `fp4_quant_kernel` 的 floor、scale 向上取整、
   直接 RNE、F32/BF16 存储边界逐项对齐。
2. 一批原生程序分别检查：全部16个 FP4 code、低/高 nibble 次序、不同 E8M0
   scales 的解码、量化中点、普通/极小 BF16 组、原生 direct 与 compile 路径。
   分开记录 encoded codes/scales 与 roundtrip 输出，不能把 FTZ、scale 或舍入失败混为一谈。
3. 只有现有成熟组合/公开 API 确实能表达时才增加一个具名原生组合；不得写 HYLE
   转换/数学 kernel，不改第三方数学以过测，不加 host 量化热路径。
4. 资格通过后才固定批量产品 FIX、ordinary Direct/Graph、原版完整 DeepSeek 片段、
   资源和适用交付/性能。原生 probe 本身不增加完整画像格。

预算：源码审计与一次原生批次，最多一次明确 API/布局纠正；不建新大型 build、
不重新下载模型、不跑 P1。若公共调用仍不能覆盖具体域，保存精确拒绝/数值边界，
转向其它已有成熟候选或提出有证据的缺口，不能无界微调。同轮执行 docs/closure。

## 一次原生批次的裁决

本次有界调查已结束，结论为 `NATIVE_DOMAIN_MISMATCH`，详见
[原始证据和适用边界](../../../../evidence/correctness/llm-moe-d5-real-fragments/apple-fp4-native-boundary/README.md)。
普通与中点域通过；65280个有限BF16按固定group32分组时，普通及两种compile路径
各830个roundtrip输出不符，均位于参考极小scale域。独立FP4解码也确认±2^-127被清零。
源码确认MLX没有DeepSeek规定的floor；清零现象的具体编译/硬件责任尚未独立裁决。

产品FIX仍为零，当前owner和152/176 ledger不变。停止重复该原生批次；下一成熟候选
须明确解决上述两个边界，另行定界后才可接线。该失败不能外推所有Provider均缺失，
也不允许放宽公共语义或使用host数学回退。D5-A5/A6/D6继续待验。
