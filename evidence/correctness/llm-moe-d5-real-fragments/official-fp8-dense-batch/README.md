# D5 官方 FP8 dense 边界与 Vulkan 准入修复

四个官方画像：DeepSeek V4 Pro/Flash、GLM-5.3/Flash；八个 gate/up、down 真实矩阵签名，
每个 M=1/3。结果与候选身份见[收据](receipt.json)，原始日志及锁定 bytecode 的反汇编保存在
同目录 gzip 文件，收据同时记录压缩前后 hash。

| 路线 | Direct / Graph 结果 | 准确边界 |
|---|---|---|
| CPU | 各16个冷拒绝已记录 | 当前普通入口不接纳具名 FP8 block128 storage |
| CUDA | 各16个冷拒绝已记录 | 当前普通入口不接纳该编码；不能用 SM89 rowwise FP8 代签 block128 |
| Apple | 各16个冷拒绝已记录 | 当前普通入口不接纳该编码；group32 与完整动态 block128 合同不同 |
| Vulkan | 各16个签名运行通过，各两次变化输入 | **所有 F32 输出逐值相等**；压缩输入进入原版 IREE 程序 |

CPU/CUDA/Apple 的记录来自 `3f8caf58`。后续修改仅涉及 Vulkan 冷准入及 Vulkan 执行夹具的
精确输出比较，复用三路线的拒绝证据。它们是**当前入口拒绝**，不证明所有上游方案皆不可能，
也没有被写成完整画像 PASS/终局缺失。旧
[成熟库资格](../../compute-llm-moe-m2-b6/qualification.json)说明本机 oneDNN FP8 与 SM89
cuBLASLt block128 scale 的拒绝；Marlin W8A16、SGLang rowwise、MLX group32 各有不同合同。
原始模型的具体 scale 角色取自
[共享 tensor 事实](../../../../tests/compute_models/d5_moe_execution_facts.json)。

## 发现并修复的产品缺陷

最初 Vulkan Direct 抛出夹具 `.value()` 异常，Graph 明确拒绝网格/workspace；初始失败日志保留。
原因是 HYLE 用65535个**元素**的保守门约束所有 FP8 矩阵。
同 pin `iree-dump-module` 直接反汇编仓库锁定 `fp8.vmfb`，无需恢复旧编译器或重编数学程序：
两个 scale 模式均依次执行 activation 转换、weight 转换、MatMul，其 X grid 分别为
`min(M*ceil(K/64),65535)`、`min(N*ceil(K/64),65535)`、`min(M*ceil(N/64),65535)`，Y/Z=1。
全宽测试通过后同时覆盖大于65535工作组的逻辑域和末尾矩阵行，不能把截断 dispatch 当正确执行。

有限修复 `652d8083` 按实际 grid 检查设备上限；同时检查合并临时 allocation
`align64(4*M*K)+align64(4*N*K)`、输出4*M*N 的 descriptor range，以及保守32位索引域。
原 bytecode、shader、算法、Scan 与 BF16 conversion 域均未修改。测试覆盖 grid 边界255/256、
单矩阵分别能装下但合并 workspace 超限、精确 workspace 边界及巨大形状的冷拒绝。

修复候选：Vulkan Direct6组（含 FP8 批次及 IREE5组）、Graph FP8批次1组通过；额外 Graph
混合/Scan2组、IREE Provider7组（含两个 quota 失败）及 route 分配失败1组通过。
随后 `4f60924b` 把本批 dyadic 稀疏 oracle 从通用绝对容差收紧为逐值相等，Direct/Graph 再各1组
通过。该夹具每个输出仅两项，选择的 codes/scales 乘积与和可精确表示为 F32。

## 范围与尚未完成项

GLM 使用原 F32 128×128 scale。DeepSeek 仅把原 UE8M0 scale 无损扩为 F32，维持128×128分组和
原 E4M3 codes；没有把它改为 group32，也没有在消费层常驻展开 F32 权重。IREE 自身仍使用片段级
F32 临时 workspace 和必要 device copy，成本必须计入最终 P1。

输入为合成、格式有效的非零 payload；激活已经是 E4M3FN/group128。本批不代签动态 activation
quantization、grouped FP8 expert、整模型加载或完整官方画像。完整画像 ledger 仍为72/176；
选择器已规划，本批执行有限 owner/consumer 回归，完整 D5 factor、三交付/P1 仍待最终批次。
没有新增 HYLE 数学；实现统计为 **58,588行 / 232文件**，不含测试和第三方源码。
