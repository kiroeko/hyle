# D5：十三画像路由及八个 GGUF MoE 批次（部分收据）

候选 `a11aac6a88a9a32265ce230e094f12d09299d061`，产品审计 baseline `b319d4b7`。
原始命令、日志、文件 hash 与未完成项见 [receipt.json](receipt.json)。本批不是完整画像验收；
ledger 仍为 **72/176 RUN_PASS、104/176 VALIDATION_NOT_RUN**。

| 路线 | Direct | Graph | 结果边界 |
|---|---|---|---|
| CPU | 5/5 | 5/5 | 全部本批用例通过 |
| Vulkan | 5/5 | 5/5 | 全部本批用例通过 |
| Apple Air M4 | 5/5 | 5/5 | 全部本批用例通过 |
| CUDA | 3/5 | 3/5 | 数值观察均通过；DeepSeek 普通 Softplus 落到冻结 HYLE，两个聚合测试的成熟 Provider 断言失败 |

五组为：原生 Clamp 有限值/±Inf 边界与重放；十三画像完整 router；十三画像 expert 激活；
八个 GGUF routed/shared 完整 MoE；Kimi 三画像三个真实 SiTU 宽度。公共 schema/独立头两项 ctest 通过。
这不是 36 个完整画像：每个 GTest 组批量消费真实角色表，整画像还缺其它 region 和交付/性能。

## 精确形状与接线

共享表固定 router 的全部896/384/256/288专家、top16/6/8、BF16/F32 角色、修正偏置、
原始分数归一化和1/1.5/2.5缩放。三个输入改变成员及顺序；expert ID 使用精确相等断言。
GGUF 混合 MXFP4、Q4_K/Q5_K、Q8_0、BF16，保留真实 H/N。只缩小常驻专家数至 top-k+1。
Kimi 包含 latent down → SiTU experts → 加权合并 → learned RMSNorm → latent up，独立共享分支
读取原 hidden。SiTU 的3072/6144/33792分别对应 expert/shared/dense，修正旧3584资格。
权重是按照真实 tensor header 生成的非零合成内容；不声称下载巨型模型或完整推理。

产品新增仅 `model_ops::Clamp` schema、四路线原版 ggml 调用与必要准入；没有 HYLE 数学/kernel。
普通 Binary/Softplus 不变；Clamp 的 NaN 数据和 signed-zero 限制在公开文档声明。
所有投影/激活中间输出和最终输出沿用原门，未为通过测试扩大误差阈值。

## 初始数值失败的明确原因

初始夹具为了生成 router logit，把少数共享 FFN 输入改成3–5，其余约0.06。原生 Q8_0 dot
会再次量化激活；同32元素块的 scale 因异常大的测试值而达到0.02362，弱信号明显损失。
独立 [q8-cause.py](q8-cause.py) 复现 GLM Flash shared-up row1921/column16：
未量化激活参考 `-0.009653848`，按原生 Q8 激活舍入预测 `-0.016981997`，与实际失败值一致。
这解释了该反例，不能泛化成任意真实模型精度保证；这种 outlier 风险确实属于原生量化成本。

修正为 router 权重64、选定输入除64，保持相同 logit、TopK 成员和归一化权重，同时让 FFN
输入处于声明的合成激活尺度。原始 FAIL、带中间输出的诊断 FAIL 和独立预测全部保留。
修正后 CPU/Apple/Vulkan 全部本批用例通过，CUDA 数值也通过。旧3584宽 SiTU 证据仅保留公式意义。

## 下一有限接缝与未完成

CUDA DeepSeek sqrt-softplus 普通算子不能用旧 HYLE 路径冒充成熟接入。已固定的 SGLang
`06992c92ad4494fe9f9636e75be3e35040512447` 在
`python/sglang/kernels/jit/csrc/moe/moe_fused_gate.cuh` 提供原版完整融合程序：稳定 sqrt-softplus、
bias、TopK、原分数归一化/缩放，准入最多512专家、top16，覆盖 DS 的256/384、top6。
它是下一次有界接入审计的候选；本收据未运行或签署该接入，也不抽出 helper 写新 HYLE 算子。

官方 expert packing、DeepSeek hash 层、其余 attention/MTP、完整 owner gate、D5 三交付与 P1 尚未完成。
[selector plan](plan.json) 为 READY（授权/选择通过），不等于 selected gates 已运行 PASS。
统一实现统计 **58,309行／228文件**，不含测试和 vendor，见 [loc.json](loc.json)。
