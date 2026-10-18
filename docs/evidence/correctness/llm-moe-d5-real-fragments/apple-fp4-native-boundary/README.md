# F46：Apple MLX FP4 原生边界

**结论：此固定 MLX/M4 原生调用未满足 DeepSeek 原版 FP4 roundtrip 的全部数值域。**
本批没有产品修改，也不增加完整画像验收格。152/176 的 ledger 保持不变。
MLX pin、调用命令、逐项统计、源码及产物哈希见 [receipt](receipt.json)、
[原始执行收据](run.json)、[日志](native.log) 和 [独立探针](probe.cpp)。
复用 F45 已有 MLX 构建；仅编译探针，未创建新大型构建树，未运行性能测试。

## 已运行的一次批次

| 检查 | 结果 |
|---|---|
| 16个 E2M1 code × E8M0 exponent 1..254，F32 与 BF16 解码 | 每种输出8128值，各4处不符；均为 scale=2^-126、code=±0.5，实际±0，期望±5.877471754e-39 |
| 单位尺度全部 code、nibble 次序 | 相符 |
| 全部65280个有限 BF16 值，按编码顺序分成32元素组 | 普通与 compiled 多输出路径各830个 code、28个 scale、830个输出不符 |
| 同输入，仅保留 roundtrip 输出的 compiled 路径 | 830个输出不符；code/scale 未观测，不能因日志中的0记为通过 |
| 固定单位尺度的 FP4 中点及 BF16 相邻值，共1344值 | 普通、compiled 多输出和输出单端口三路径的已观测项全部相符 |

全部830处输出差异的参考 E8M0 exponent 均≤8。该结论只限本次固定分组，
不等于穷尽所有32元素组合，也没有证明模型激活永远不会进入失败域。
日志末尾4214是跨模式、code/scale/output 的重复累计计数，不是独立失败输入数。
程序退出1表示数值不符；没有把它当作编译或环境故障重试。

## 源码已确认与未确认的部分

已登记 DeepSeek `third_party/deepseek_quantization/upstream_kernel.py:fp4_quant_kernel`
在生成 scale 前规定 `amax >= 6*2^-126`。固定 MLX 的
`mlx/backend/metal/kernels/fp_quantized.h:fp_quantize/fp_quantize_dequantize`
使用 `amax/6` 后向上取整，没有同一 floor。这是实际语义差异。

独立解码没有经过量化器，仍将可表示的极小输出变成零，因而不能只改 scale
就宣称全域对齐。该现象与次正规数清零一致；本批未独立裁决具体编译选项、转换
实现与硬件各自的责任，也未验证编译设置能否消除差异。

普通和中点结果支持继续评估成熟原生解码的适用域，但不足以资格化完整专家或
indexer。没有替换为 BF16 权重、额外 Q8 激活、host 数学回退，也没有放宽公共精度门。

## 停止与后续边界

本卡的一次原生批次已用完，调查在上述明确结论停止；不重复同一输入寻找通过结果。
下一步只能基于具体成熟 API/组合或具名候选，先明确如何同时处理 floor 与极小数值，
再按 owner 协议另行固定有限接线范围。若没有这样的候选，按精确域登记上游限制；
此收据本身不证明所有成熟 Provider 都缺失，也不将整幅画像标为 BACKEND_REJECTED。

源码/静态/动态交付、适用性能与最终收口仍待 D5-A5/A6/D6，未由原生探针代签。
