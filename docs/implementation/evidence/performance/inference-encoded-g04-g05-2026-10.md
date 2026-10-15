# G04/G05 编码页有限工程性能总账

这里只记录已合格同合同工程 P1；不代表全模型 token/s、跨路线速度排名、capture 或
整个 G04/G05。ratio大于1有利HYLE；方向相对1，非劣门独立使用原5%阈值。
原 FAIL/DIAGNOSTIC 和全部成本保留在对应施工卡及原始 evidence。

| route / candidate | workload | ratio范围 | 方向 | 5%非劣 | 原始证据 |
|---|---|---|---|---|---|
| CPU CE13 / 42a4c7cdcc | window D256 | 1.1716–1.1787 | WIN | PASS | [QUALIFIED/verify0](../../../../evidence/performance/inference-g04-g05-ce13-cpu-encoded-w1-p1/receipt.json) |
| CPU CE13 / 42a4c7cdcc | prefill D256 | 0.9899–0.9939 | INCONCLUSIVE | PASS | 同上；5%带内结论不替代方向 |
| CPU CE13 / 42a4c7cdcc | noncausal D64/Dv128 | 0.9654–0.9685 | LOSS | PASS | 同上；稳定小幅LOSS保留 |
| Vulkan / 982676cb05 | window D256 | 5.6315–5.6856 | WIN | PASS | [QUALIFIED/verify0](../../../../evidence/performance/inference-g04-g05-vulkan-encoded-m1-p1-attempt-1/receipt.json) |
| Vulkan / 982676cb05 | prefill D256 | 3.8990–4.0220 | WIN | PASS | 同上 |
| Vulkan / 982676cb05 | noncausal D64/Dv128 | 5.5225–5.5900 | WIN | PASS | 同上 |
| Apple CE14 / 3b5157532 | window D256 | 1.2279–1.2381 | WIN | PASS | [QUALIFIED/verify0](../../../../evidence/performance/inference-g04-g05-apple-encoded-m1-p1-attempt-5/receipt.json) |
| Apple CE14 / 3b5157532 | prefill D256 | 1.1443–1.1513 | WIN | PASS | 同上 |
| Apple CE14 / 3b5157532 | noncausal D64/Dv128 | 1.0765–1.0833 | WIN | PASS | 同上 |

CPU是W1-WIN，Apple/Vulkan是实际Air M1-MAC工程资格；各自contract/peer精度、准备、
生命周期和资源差异见原receipt，不能跨行推断绝对速度。CPU保原Q缩窄；GPU peer明确
使用原general Q8路线，不称原host最佳策略。native private-pool上界仍UNKNOWN。
CUDA原4ms/40ms A/A失格和后续时长/机制诊断均保留。新的episode交错与全量算术分组已
[通过完整16路A/A](../../../../evidence/performance/inference-g04-g05-cuda-episode-aa-qualification/receipt.json)，
原P1采样仍在运行；没有完整合格throughput裁决，继续UNPROVEN，不列虚构的平局或PASS。
