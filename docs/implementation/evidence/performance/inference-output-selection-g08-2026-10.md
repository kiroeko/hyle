# G08 输出数值选择：有限性能分账（2026-10）

CPU、CUDA、Apple、Vulkan正式P1均通过原工程门，九个预声明工作负载均已归档。
[机器分账](inference-output-selection-g08-2026-10.json)保留全部合格方向；
[施工卡](../../tasks/compute/inference-g08-output-selection.md)持有其余验收。
本页不改2026-09历史98个合同组或G04/G05结果，不把S1片段外推为引擎token/s。

比较对象为HYLE原`85bcfa4eb`，产品candidate为`5103c13d8`；CPU benchmark固定`5fbc139921`，
其余路线固定`47b63229d7`。相同resident正常F32、V248320、values+I32 indices、checked完成、
双输出读取和步内owner释放。冷prepare/上传与末次保留结果/runtime/cache析构按
[原时序账](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-boundary.json)分列。
新旧公共framework及修复后的Mem相同，引用各原provider/lowerer完整实现；不是整仓新旧binary比较。

每路线八个执行/分配/placement分层各n15、共120产品进程，另有16路完整workload A/A与30公共控制。
qualification、五controls、stock verify和固定5%非劣门全部通过，n15停止。
CPU为W1-WIN/i9-13900K、MSVC19.50、原注册CPU2/ABOVE_NORMAL策略；CUDA为同机W1-WSL/RTX4070、
GCC14/CUDA13.3、SM89-real，仅guest affinity。Apple/Vulkan为Mac16,12 M4 Air/macOS27.0.1、
同AppleClang/arm64及interactive QoS，Vulkan沿实际MoltenVK。CUDA与Air证据明确为engineering角色，
不代表native Linux、可验证P/E硬亲和或Formal稳定machine身份绑定。

ratio大于1有利于新HYLE；范围覆盖全部八分层，不计算平均加速比。CI列为八分层外包络，
不是额外拟合的整体区间。严格方向相对1裁决，和原仪器exact verdict/5%工程门分别保留。

| 路线 / 工作负载 | 吞吐ratio范围 | 八分层CI外包络 | 严格方向 | 原exact / 5%门 |
|---|---:|---:|---|---|
| CPU / R1 K1 | 1.218660–1.255562 | 1.207310–1.262509 | WIN | WIN / PASS |
| CPU / R16 K1 | 1.218030–1.266141 | 1.209095–1.271683 | WIN | WIN / PASS |
| CUDA / R1 K1 | 2.358158–2.655978 | 2.252301–2.694195 | WIN | WIN / PASS |
| CUDA / R16 K1 | 3.123480–3.395567 | 3.107712–3.438778 | WIN | WIN / PASS |
| Apple / R1 K1 | 2.514905–2.533181 | 2.493961–2.554302 | WIN | WIN / PASS |
| Apple / R16 K1 | 9.529469–9.580187 | 9.411010–9.705648 | WIN | WIN / PASS |
| Apple / R16 K3 holdout | 1.043167–1.043839 | 1.042619–1.044343 | WIN | PARITY / PASS |
| Vulkan / R1 K1 | 4.400117–4.427783 | 4.379937–4.454940 | WIN | WIN / PASS |
| Vulkan / R16 K1 | 41.623223–41.915120 | 41.488354–42.548183 | WIN | WIN / PASS |

CPU新旧都O(RV)，当前单选省去V宽Entry scratch/fill与partial_sort触达。CUDA arg reduction替代
固定key宽度的完整radix sort；实际workspace R1由12039680B降到9728B，R16由192249344B降到512B，
provider persistent416B/private8B不变。Apple K1替代全argsort；Vulkan使用两次有界pair reduction dispatch，
完整完成与download staging/fences仍收费。Apple K3保留argsort/slice/gather算法族作为holdout，
观察到约4.3%差异，不把该小差异归因为K1的arg reduction。没有把旧CPU/CUDA排序统称VlogV。

[CPU原始凭据](../../../../evidence/performance/inference-g08-output-selection/cpu/receipt.json)、
[CUDA](../../../../evidence/performance/inference-g08-output-selection/cuda/receipt.json)、
[Apple](../../../../evidence/performance/inference-g08-output-selection/apple/receipt.json)、
[Vulkan](../../../../evidence/performance/inference-g08-output-selection/vulkan/receipt.json)
分别保留原manifest/summary、全部raw样本、actual binary/构建身份与参考源码身份。
CUDA提交raw.zip含全部样本/源码/身份，172MB完整binary ZIP在receipt明确标识的本地内容寻址缓存，
原WSL qualified binaries仍保留；这是依赖live artifacts的工程证据，未签Formal不可变长期保存。
其它三路线提交归档包含实际binary，四路线archive CRC/hash及对应binary身份均已核对。

| 路线 | 正式阶段秒 | 含原失败的诊断秒 | 已记录两阶段合计秒 |
|---|---:|---:|---:|
| CPU | 508.265 | 117.031 | 625.296 |
| CUDA | 650.609 | 186.199 | 836.808 |
| Apple | 589.012 | 66.685 | 655.697 |
| Vulkan | 663.901 | 88.363 | 752.264 |

已记录合计2870.065秒；正式串行controller墙时2455.485秒另留原记录。
这些不是整个G08历史费用；没有重置失败预算或把原预检n1的FAIL改成PASS。

[独立终审](../../../../evidence/architecture/inference-g08-output-selection-audit/independent-consumer-review.json)
未发现有限S1同合同MUST。
[实际artifact补证](../../../../evidence/architecture/inference-g08-output-selection-audit/p1-artifact-postreview.json)
确认真实调用、优化设置及纯布局变化，text仅增加32/0/20/20B；CUDA只证明计时callback移动16B，
不声称其GenericArm或GPU代码也移动。独立review、section大小和反汇编记录在正式采样后完成，
未在preflight附齐记录的流程偏差保留，作为工程终审补证接受，不回填为prospective Formal预签。
未收集hardware counters，不作cache/指令数归因。

S2/S3只签既定数值质量和P0。最终native/allocator/cache上界、末次retained析构、冷启动、其它dtype/
形状与机器均不由本表推断；没有完整模型、prefill/decode token/s或llama.cpp性能声明。
