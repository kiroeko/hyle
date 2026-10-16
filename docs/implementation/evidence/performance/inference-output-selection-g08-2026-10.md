# G08 输出数值选择：有限性能分账（2026-10）

当前仅CPU正式P1完成；CUDA、Apple、Vulkan待同一串行批次完成。G08仍为CANDIDATE，
本页不改2026-09历史98个合同组，也不把S1片段外推为推理引擎吞吐。
[机器分账](inference-output-selection-g08-2026-10.json)保留全部合格方向；
[活动卡](../../tasks/compute/inference-g08-output-selection.md)持有其余验收。

比较对象为HYLE原`85bcfa4eb`，产品candidate为`5103c13d8`；CPU benchmark固定`5fbc139921`。
相同resident正常F32、V248320、values+I32 indices、checked完成、双输出读取和步内owner释放。
冷prepare/上传与末次保留结果/runtime/cache析构按[原时序账](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-boundary.json)分列。
CPU为W1-WIN/i9-13900K、MSVC19.50、原注册CPU2/ABOVE_NORMAL策略，八个执行/分配/placement
分层各n15，共120产品进程；另有原完整workload A/A与30公共控制。qualification、controls、
stock verify和固定5%非劣门全部通过，n15停止。

ratio大于1有利于新HYLE；下列范围覆盖全部八分层，不计算平均加速比。CI列为八分层外包络，
不是额外拟合的整体区间。严格方向相对1裁决，和5%工程门分别保留。

| 路线 / 工作负载 | 吞吐ratio范围 | 八分层CI外包络 | 严格方向 | 5%非劣门 |
|---|---:|---:|---|---|
| CPU / R1 K1 | 1.218660–1.255562 | 1.207310–1.262509 | WIN | PASS |
| CPU / R16 K1 | 1.218030–1.266141 | 1.209095–1.271683 | WIN | PASS |

[CPU原始凭据与完整归档](../../../../evidence/performance/inference-g08-output-selection/cpu/receipt.json)
包含原manifest/summary、全部raw logs、实际binary/编译命令及参考源码身份。新旧CPU算法都为
O(RV)；当前单选省去V宽Entry scratch/fill与partial_sort多次触达，不能声称旧实现为VlogV。
原失败与117.031s诊断保留，正式批次另508.265s；这些已记录阶段合计625.296s，不冒充全部历史费用。

S2/S3只签既定数值质量和P0；没有完整模型、prefill/decode token/s、其它CPU或其它dtype/shape的性能声明。
