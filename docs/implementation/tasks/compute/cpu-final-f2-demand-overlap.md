# F2 固定需求与真正重叠域

F2只读需求/承载审计已固定，协调owner为当前CPU任务维护者，不新增模块或借旧CLOSED授权产品修改。
总合同[CPU最终收口](cpu-final-closure-plan.md)，baseline `4af5056aed00ceed415dd91a314f26414aee5c78`。
读取既有C1 header/原官方forward与C5合同，固定104 GGUF、官方混合、四非GGUF、三sidecar全部需求。
批量生成端口/精度/布局/state/region→每平台实际Provider接缝→独占/重复/缺失表。
以完整成熟融合region为候选，不拆库内kernel；文件Q标签/模型名字不替代端口精度。
保留未知形态和Mac缺口，不把metadata存在改成RUN_PASS。复用固定快照，不重新抓所有HF历史或下载巨权重。
每缺口最多两个具名成熟候选；矩阵签完后交F3/F4，不实现自研数学，不跑性能。
未来产品owner只按实际接缝确定，旧CLOSED不因本页重开；文档只执行closure/docs。

## 固定输出

[逐制品/端口/分支/候选表](cpu-final-demand-carriers.json)复用原C1/C5及D5官方身份，并保存输入SHA。
104主GGUF档位、559去重端口（433原packed）、34非GGUFpacked、45原plain与80参数role/shape；
八个family的完整semantic/state region、三sidecar、两新官方DeepSeek及其它六官方族/本地Qwen原版均保留。
旧DeepSeek preview只作未撤回能力的回归输入，不替代新版；metadata/旧CPU PASS不代签当前三平台。

| 当前真正承载/重叠 | 原接缝和有限动作 |
|---|---|
| 433原packed GGUF | ggml原codec/MatMul/Grouped/Gather；没有同合同oneDNN plain GEMM，不逐档位竞速 |
| 原plain矩阵 | ggml/ATen；oneDNN只dense rank2 F32/F32 accumulation、非empty，F16 cold cast真实单列 |
| oneDNN与ggml的重复 | 合格plain MatMul、RMSNorm、LayerNorm、Copy、Select五族；不是所有dtype/shape都重复 |
| oneDNN与ATen的重复 | 上述已准入交集与稳定Softplus；先核对资源/精度，再有界对比或工程默认 |
| ggml与ATen的其它交集 | Cast/Fill/Concat/Gather/Scatter/Reduce/Softmax/RoPE/SDPA的具体已准入合同；保留完整region |
| NF4/AWQ/CT/MLX affine/MX4/MX8及官版FP4/FP8 | 已借原codec/ATen CPU序列；当前单承载，cold转换/selected group/state边界必须保留 |
| 既有Native自研 | 35种/15族逐源码去留，旧187/17不继承新版；借入codec/模型程序不计自研 |
| Mac ARM64缺口 | ggml原CPU存在；现Torch平台/包/loader门关闭，必须接原ARM64 SDK，不用Metal代签 |

F3有限FIX固定为平台SDK/原程序承载与实际缺失接缝；当前无证据要求新增数学。
F4只覆盖真实候选交集与完整成本，近似/不稳定按用户授权工程默认快速裁决；未运行不签赢家。
F2完成的是需求与承载边界，不是CPU实现覆盖或三交付完成，不给整goal叠加旧CLOSED百分比。
