# G04/G05 编码页有限工程性能总账

这里保留原合格工程P1及当前比较边界裁决；不代表全模型token/s、跨路线速度排名或capture。
ratio大于1有利HYLE；方向相对1，历史非劣门独立使用原5%阈值。
原 FAIL/DIAGNOSTIC 和全部成本保留在对应施工卡及原始 evidence。

**2026-10-16当前裁决**：[G04/G05终审](../../tasks/compute/inference-g04-g05-final-closure.md)
接受选定压缩驻留/F32数值合同，以P0结构成本结束本轮；无新计时或P1通过。CUDA原
comparator是预展开BF16的FlashInfer自然方案，具有额外驻留/中间精度差异，不是llama.cpp，
不签同约束FP8实现排名。attempt14窗口LOSS/5%PASS、prefill LOSS/5%UNRESOLVED、留出
INCONCLUSIVE/5%PASS以及原整体FAIL标签保留；下文“未关闭”均为各历史阶段当时状态。
精确未决数值不改成parity，C1私有候选拒绝也不计作产品退化。完整引擎tokens/s尚未实测。

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

[attempt11](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-11/receipt.json)
对应CE25 candidate0143102ae，QUALIFIED/strict verify0；16AA、五controls、三consumer与
采样前后三validator均PASS，artifact/input/host稳定。n15完成120产品与30控制；window
D256 ratio0.986884–0.993310 / LOSS / 5%PASS，prefillD256 0.913710–0.926685 / LOSS /
FAIL，noncausal D64/Dv128 0.981282–1.001226 / INCONCLUSIVE / PASS。整体performance
FAIL，明确prefill失败不追加n31。worker2629.221s和重叠outer2691.563s分别保留。
较原10改善不替代5%门；全部旧失败、方向、未知与成本继续保留。CUDA/G04/G05未关闭。

[attempt12](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-12/receipt.json)
对应CE26 candidate1e343ee3b，QUALIFIED/strict verify0，16AA、五controls、三consumer与
采样前后三validator均PASS，artifact/input/host稳定。n15为120产品/30控制；
windowD256 ratio0.987633–0.990970 / LOSS / 5%PASS；prefillD256
0.943718–0.950556 / LOSS / 5%FAIL（exact INCONCLUSIVE）；noncausal D64/Dv128
0.980925–0.993621 / INCONCLUSIVE / 5%PASS。整体performance FAIL，不追加n31。
worker2625.142s、outer2671.843s重叠不相加，全部旧失败/方向/未知与费用保留。
CUDA/G04/G05未关闭，不把较11的改善签为通过。

[attempt13](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-13/receipt.json)
对应CE27 candidated41431f82，QUALIFIED/strict verify0；16AA、五controls、三consumer与
采样前后三validator均PASS，artifact/input/host稳定。n15为120产品/30控制；
windowD256 ratio0.986590–0.993320 / LOSS / 5%PASS（exact PARITY）；
prefillD256 ratio0.940486–0.947439 / LOSS / 5%FAIL（exact INCONCLUSIVE）；
noncausal D64/Dv128 ratio0.982965–0.994006 / INCONCLUSIVE / 5%PASS（exact PARITY）；
整体performance FAIL，不追加n31。
worker2577.782s、outer2666.843s重叠不相加，全部旧失败/方向/未知和累计费用保留。
静态BAR减少不代签实际吞吐收益，CUDA/G04/G05仍未关闭。

[attempt14](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-14/receipt.json)
对应CE28 candidate830dd9e01，QUALIFIED/strict verify0；16AA、五controls、三consumer与
采样前后三validator均PASS，artifact/input/host稳定。[15,31]完成368产品/92控制。
window D256 ratio0.988927–0.992157 / LOSS / 5%PASS（exact PARITY）；
prefill D256 ratio0.952972–0.961375 / LOSS / 5%UNRESOLVED（exact INCONCLUSIVE）；
noncausal D64/Dv128 ratio0.982524–0.994022 / INCONCLUSIVE / 5%PASS（exact PARITY）；
整体performance标签FAIL；prefill为UNRESOLVED，非已证明material FAIL。八层median
均高于原5%耗时门1/1.05，两层CI跨界，不能据此签PASS或改用0.95门。
worker7262.945s、outer7477.109s重叠不相加；原2h检查点的同run900秒追加、全部旧失败/方向/未知和费用保留。
CUDA与整个G04/G05未关闭；不重复同候选P1、不追加n63或修改原合同。
