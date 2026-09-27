# D5 CPU / Apple / Vulkan 性能消费者正确性

本批只资格化性能消费者，不签 A/A、P1 或整模型吞吐。产品 SDK 均复用
`9273bc71017a1febb69db240b6ea8970b88dd77c`；逐文件身份见各轮 `sdk-inputs.json`。

| 路线 | 通过的消费者提交 | 完整输出与完成检查 | CTest 墙钟 |
| --- | --- | --- | --- |
| CPU | `784819caece7eb650d72d73f889a4835b9ddeff6` | 20 / 20 | 5.860 s |
| Vulkan | `f74fa7a6e8e68adb9ae4311c04ea14ad2f9058c5` | 20 / 20 | 4.812 s |
| Apple | `7e77c863ef5d6a4cf892bf74876d14c8f24e4320` | 20 / 20 | 16.586 s |

每路线覆盖 Qwen3.5-2B 官方 BF16 完整 FFN 的两个变化输入，以及 DeepSeek Flash
UD-Q4_K_XL 完整 hash-MoE 的三个变化输入；Direct/Graph 各重复两遍。只发布最终输出，
Apple FFN 保持完整 MLX 区域，Vulkan Direct 使用公开 Recorder 一次提交完整调用序列。
编译低并行 2、重型验证串行；没有重建产品或重复三交付。

保留两次原始消费者失败：Vulkan 首轮把冷准备资源误作上游预热后资源，修正为完整槽位预热后
固定基线，并严格检查第二遍及后续重放；Apple 共用 helper 误用了 CUDA 的绝对容差，恢复原
Apple 消费者的 `0.008*abs(expected)+3e-8`，完整融合 FFN 实际通过。两项均未改产品。

[index.json](index.json) 记录归档文件和原日志的 SHA-256。日志/XML 以 gzip 保存原始字节；
各轮 receipt 记录原命令、提交、耗时和返回码。性能采样须绑定最终不可变消费者与输入后另行执行。
