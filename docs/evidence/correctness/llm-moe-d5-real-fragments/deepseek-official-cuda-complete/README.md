# DeepSeek 官方 Pro/Flash：CUDA 完整画像正确性

候选 `9e846d87a80d4a0f8e8b220b4061fa3580232c86`，Windows RTX4070 / SM89 /
CUDA Driver API13040。**两个官方 FP4/FP8 混合画像的 Direct/Graph 必需 region 均有通过证据。**
新增4格，ledger为148/176；CUDA的22画像共44格完成正确性，三交付/P1仍未签署。

## 新完成的真实消费链

| 消费链 | 保留的模型要求 | 两入口实际执行 |
|---|---|---:|
| 压缩器 | ratio4重叠、ratio128、D128 indexer；F32投影/history/APE/pool；BF16 norm/RoPE及原FP8/FP4还原 | 72次 |
| 完整注意力 | 两种真实H/Qrank/head数；128 raw窗口、top1024/512真实索引、masked未来cache、sink、逆RoPE、冷解码wo_a、FP8 wo_b | 72次 |
| 完整MTP | 独立FP8 e_proj/h_proj与BF16相加；mHC attention→MoE→mHC；learned head与norm | 24次 |
| 全词表head | 每轮全部129280行，BF16 checkpoint无损扩宽为F32参数，F32 logits | 768个分片调用 |

六项GTest全部通过，零失败/零跳过。每个模型consumer六轮、两次reset，历史/缓存/hidden均由
实际输出回送；非reset输入的错误占位值用于暴露错误回灌。oracle独立推进，TopK成员和唯一性
单独精确比较。完整词表同时验证实际hidden输入的head局部门和独立decoder轨迹的整链门。

官方wo_a只在冷期由已固定成熟FP8解码程序生成真实BF16常量，再由两个热入口消费；旧程序与
原encoded buffer先销毁，禁止把host oracle常量上传代替解码。Pro/Flash的常驻BF16权重分别为
128MiB/64MiB；本批实际逐值检查。其余生命周期/失败能力复用未变的F41、F44收据。

前置完整score/hash MoE、原版FP4专家及资源见
[F44](../deepseek-original-fp4-entry/README.md)；R01 embedding/Gather复用
[已验词表](../moe-shared-core-batch/README.md)。旧通用BF16 head不代签本次官方F32 logits。
固定原始config/全层权重事实在仓库测试JSON中；合成非零payload保留真实格式和内维，
不宣称已加载、推理两份巨型checkpoint。

## 初始失败与有限纠正

- 初次编译为旧/新compressor的ADL歧义，显式限定命名空间后通过。
- 初次运行两入口的压缩器/MTP通过，注意力失败。把compressor赋给完整attention recipe时，
  丢失预先声明的quantized标记，BF16一个舍入步距被F32原子门判失败。恢复原整链分类，
  standalone压缩器的F32/BF16门保留；未调整通用数值阈值。原失败日志完整保留。
- 官方RoPE频率使用原版F32 power/reciprocal/YaRN表达式；独立于GGUF的逐项递推频率。
- 测试准备只转换合成权重的非零BF16项，零字节直接保留；上传矩阵与覆盖范围相同。
  测试批次由259.36秒降至42.59秒是夹具准备改进，不是产品性能成绩。

最终增量构建13.70秒，测试42.59秒；13项模型输入工具、closure、docs通过。
统一实现60,150行/240文件，本批没有新增产品算子数学或公共API。

## 未完成边界

仍有官方CPU/Apple/Vulkan的24格、DeepSeek GGUF Vulkan的4格，以及A5剩余selected gates/
源码静态动态交付、A6 P1和D6。F32诊断运输、BF16转换、冷常量容量及词表host staging成本
均保留，不宣称零拷贝或性能SOTA。当前GPU/驱动资格不外推。

[机器收据](receipt.json)逐region链接前置证据，记录候选、测试范围、日志hash与剩余项。
