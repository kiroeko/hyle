# F40：GLM-5.3-Flash 官方 FP8 的 CUDA 完整片段

不可变候选 `f5d9abd5f58a0d5e7ccebbc27937cf0f8ead4a47`，Windows RTX 4070 / SM89。
普通 Direct／Graph 的完整必需 region 已齐备，新增 **2 个完整画像格**；ledger 为
**142/176 RUN_PASS、34/176 VALIDATION_NOT_RUN**。不代签其它路线、三交付、P1 或整模型推理。

## 已运行与复用

[机器收据](receipt.json)保留逐 region 对应证据。新批次构建 9.08 秒、测试 14.08 秒，
两入口各 2 组通过、零失败／跳过：官方混合片段与四轮 F32 累加舍入边界。
普通入口运行真实 hidden、rank、expert FFN 和154880全词表；非本地巨型模型使用格式合法的
非零合成权重，没有把 BF16/Q8 权重冒充官方 FP8。

| 所需 region | 已验边界 |
|---|---|
| R01 | 原完整词表 Gather／越界／全部 logits；本批真实 NextN 输出接全词表 head |
| R02–R03 | 官方 I12288 dense FFN、逐角色 FP8 投影、BF16 参数与 Flash 截断；KDA BF16 投影复用 |
| R06 | 已验完整 KDA、真实卷积历史与 recurrent 状态反馈 |
| R07–R09 | 混合 FP8/BF16 MLA 四轮 cache 反馈；512 个 pooled selection 展开2048个 token、0..3尾部；索引 carry/reset |
| R10–R11 | 原 SGLang router→grouped FP8 experts→combine及shared；补齐截断的回归已通过 |
| R12 | 已验真实 fn 权重、mHC/Sinkhorn20、四轮输出反馈 |
| R13 | EH→MLA→残差／norm→MoE→最终 norm，四轮真实 hidden/cache 回送，全部 head 输出 |
| R14–R15 | 原 block128 动态量化／dense与grouped执行；已验资源失败、并发、Completion保活、非法ID及重复提交 |

复用范围仅限未改变的对应模型／权重角色、程序、输入和环境。新增 ModelLinear 参考不被旧 KDA、
mHC 或词表 consumer 使用；新 Provider 条目按独立 operation 冷匹配，没有替换这些既有程序。
[初始批次](../glm-official-mixed-checkpoint/README.md)的聚合 FAIL 保留，其中已通过的共享／完整 MoE、
截断及 GGUF 回归按具体测试复用，未将整份初始收据改写成 PASS。

## 数值失败的原因和纠正

`Float8BlockLinear` 的既有合同要求 **F32 累加**。旧共享参考却把解码值在 double 中累加，最后
才转 BF16，省略原 SGLang 分块 F32 缩放与融合累加的舍入。首层输入 norm 的 F32 微小差别在
BF16 存储后全部消失；四轮 QA/QB/KVA/O 的16次真实投影都与独立 F32 累加参考逐元素一致。
使用同一声明的舍入边界、独立推进完整链而不读取GPU中间答案，四轮最终输出也全部一致。

修正只涉及测试参考：块内解码点积仍独立使用 double，显式恢复 F32 缩放／融合累加及 BF16
存储；保留旧数学参考，并用原 consumer 的四个输入证明这两种数值程序确会产生不同答案。
输入、权重、实际状态反馈、局部 BF16 1ULP 与整链 `abs2e-5 + rel0.025` 门均未改变。
完整 MLA／NextN 在最终不可变候选重新运行通过。没有修改上游数学、kernel 或产品 API。

[第一分歧统计](first-analysis.json)、[独立整链预测](first-chain-prediction.json)、
[捕获数组](first-projections.zip)及[哈希](first-manifest.json)保留；定位脚本与原始运行日志
随收据保存。诊断临时观察器已从产品工作树移除。

## 仍未完成

GLM-5.3 官方 DSA 的 FP8 kv_b 冷准备及完整 MLA／NextN仍待接入，本批只完成其 dense FFN、
indexer／selection部分；其它官方画像和路线、DeepSeek GGUF Vulkan四格、D5三交付／P1与D6待验。
全词表分片入口明确包含真实 hidden 的 host staging 和上传成本，没有零拷贝或性能通过声明。
统一[实现统计](loc.json)为 **59,645行／237文件**，不含测试和第三方代码。
