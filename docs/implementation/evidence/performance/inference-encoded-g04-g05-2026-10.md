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
CUDA原4ms/40ms A/A失格和后续时长/机制诊断、attempt6 ISA与attempt7 provenance绑定失格均保留。
[attempt8](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-8/receipt.json)
现为QUALIFIED/strict verify0，16AA及五controls PASS；三workload比率/方向/5%门为：
windowD256 0.953993–0.958642 / LOSS / UNRESOLVED；prefillD256 0.861386–0.872412 /
LOSS / FAIL；noncausal D64/Dv128 0.980057–0.994756 / INCONCLUSIVE / PASS。
整体工程性能FAIL，不能以留出PASS或旧未合格样本关闭CUDA或整个G04/G05。
当前仅作有限实际机制归因，不更改原5%/输入/统计或将压缩存储取舍改写为平局。

[attempt9](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-9/receipt.json)
对应CE21 candidate9c9ab88bb，QUALIFIED/strict verify0，16AA通过；120产品与30控制进程
按原n15完成，整体performance FAIL，不追加n31。windowD256为0.981006–0.985242 /
LOSS / PASS；prefillD256为0.883122–0.891172 / LOSS / FAIL；noncausal D64/Dv128为
0.914731–0.926564 / LOSS / FAIL。窗口5%门通过不能代签其它两项或全CUDA。
worker2635.632s和重叠外层2637.703s分别保留。当前只做同一冻结artifact的一次有限归因，
原8及所有历史失败均不覆盖，原输入/peer/统计/门槛不变。

[attempt10](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-10/receipt.json)
对应CE24 candidate6e57e28ad，QUALIFIED/strict verify0，16AA、五controls、三consumer与
三validator均PASS，原n15完成120产品与30控制进程。windowD256为0.977278–0.983388 /
LOSS / 5%PASS；prefillD256为0.852769–0.858133 / LOSS / FAIL；noncausal D64/Dv128为
0.976652–0.989723 / INCONCLUSIVE / PASS。整体performance FAIL，明确prefill失败不追加n31。
worker2679.859s与外层2715.844s重叠，分开列账。新1/1/4特化零栈读写和实际runtime身份
不代替吞吐；原8/9及全部失败、方向与成本保留，当前CUDA与整个G04/G05未关闭。
