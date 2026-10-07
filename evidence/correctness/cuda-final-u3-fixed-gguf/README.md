# CUDA U3：固定 GGUF 数值合同与当前参数批次

候选 `374bf4654f7a25f20932ee430395d64713780168` 在 Windows／WSL Linux 普通 Direct／Graph
各运行 **581 项，通过且无 skip**。包含全部固定 **559 个数值合同（433 packed／126 plain）**、
104 个真实制品身份、Qwen 实际 norm/attention/convolution／完整 NextN、当前混合 DeepSeek
main/cache/MoE/hyper/full-head 及当前 official／三量化 sidecar。另每平台 1 项重复专家／F32
乘数的独立源方程回归通过。精确命令、候选、输入 SHA、计数与旧失败见
[receipt](receipt.json)、[native batch](native-batch.zip)。

## 批量接入与纠正

共同 CPU C5 目录／物理 bytes／shape／角色／模型身份直接注册到普通 CUDA runner。CPU 默认
fixture 数学保持；GPU quantized reference 调用原 ggml CUDA，以实际 GPU activation codec
验证 adapter。plain 数学仍有独立源方程；原 primitive 的独立 codec/精度资格继续适用。
这不代表同载体差分是另一个性能 peer，也不拿 CPU vec-dot 当 GPU 量化误差参考。

一次完整批次检出 8 个失败测试，两入口相同：7 个 BF16 expert 合同的重复 ID，以及 Qwen
2.4T 完整 NextN 的 Q8 embedding 源大小。两个根因的具名源码审计及有限动作见
[R1 施工卡](../../../docs/implementation/tasks/compute/cuda-final-u3-model-regions.md)。

- plain GroupedMatMul 使用原 reshape，使每个公共 token/slot 成为原 MUL_MAT_ID 的独立 token，
  结果再 reshape 回原输出。保留 F32 activation、原 BF16→F32 权重转换、checked IDs、原
  kernel；不复制 selected 权重、不编写新数学。量化 expert 路径保持。
- 原量化 GET_ROWS 使用 size_t row byte offset，只放宽已 admitted source 的 byte 上限至
  UINT32；logical/stride/输出／其它 operation 原门保持。248320×8192 Q8 表与完整词表保留。

纠正后同一不可变候选集中验证，不逐算子构建。当前 official／sidecar 在本候选重新通过，
未继承修复前的聚合 PASS。旧失败原日志 213,152,530 bytes 已无损压缩并逐 SHA 校验，压缩包
随证据保存；释放约 203 MiB 工作空间，未删模型或 producer/cache。

## 范围与限制

这是当前 native 资格。其余 family-region 关联与本地 Qwen 实际权重程序、最终 U3 联合
CUDA-only/mixed 的 source/static/shared、U4 默认／best-of、U5 自研清退仍待完成；不签
完整 CUDA 收口、不声称全权重整模型常驻或推理 engine。

统一脚本 [LOC](loc.json)：Compute + Graph 全部实现 **60,658 行／253 文件**，不含测试、
bench、第三方或构建脚本，`--self-check` 通过。本批没有 P1 或性能赢家结论。
