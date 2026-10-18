# D5 Qwen GGUF 词表 carrier 批次收据

Windows canonical 候选 `76ebc05d5` 将 Qwen3.5-2B、Qwen3.8-27B、
Qwen3.8-2.4T-A95B 的六个固定 GGUF 画像合为一个 R01 接线批次。画像逐角色记录
embedding/output head 的实际 Q4_K、Q6_K、Q8_0、F16 或 BF16 格式，保留
H=2048/5120/8192、V=248,320 及 tied/untied 身份。不按 artifact 名称推断全局 dtype。

公共夹具用成熟 ggml 格式生成真实 codes/scales 的分片权重，独立解码后计算 oracle；
这不是下载的实际 GGUF 权重内容。输出头以两次变化输入执行全部 248,320 个逻辑
logits，Gather 验证跨片全局索引、两个全局越界值和尾片 Provider 失败不覆盖旧输出。
量化 MatMul 采用已有 GGUF 投影误差合同，计入 Provider 对 activation 的原生量化；
Apple/Vulkan 的量化 Gather 使用各自既有转换误差边界。

同一不可变候选上的八组普通入口定向运行全部通过：

| 路线 | Direct | Graph | 验证位置 |
|---|---:|---:|---|
| CPU | 12/12 | 12/12 | Windows canonical |
| CUDA | 12/12 | 12/12 | Windows RTX 4070 |
| Apple MLX/Metal | 12/12 | 12/12 | MacBook Air M4 验证副本 |
| Vulkan | 12/12 | 12/12 | Windows RTX 4070 |

合计 96/96。六个 Windows target 在既有隔离树低并行增量编译，Apple 两个 target
在从 Windows 候选同步的干净 Mac 验证树增量编译；重型运行串行。公共
`inputs_test.py` 13/13、文档和 closure 检查通过。Windows selector plan 为 `READY`；
本收据是 R01 批内接线检查点，未执行完整 selector owner suites、外部
source/static/shared 交付或 P1，也没有把八个模型 profile 的其它 region 代签。
`llm-moe-d5-acceptance-ledger.json` 仍为 0/176 `RUN_PASS`。
