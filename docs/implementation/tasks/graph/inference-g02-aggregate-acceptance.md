# G02 联合分页原定范围的整项裁决

状态：**G02 已完成，2026-10-15**。本记录聚合原 common CPU+CUDA、精确页守卫与
CPU+Apple/Vulkan 原生联合分页，以及原 Windows 未决门和补验发现的有限修正。
[聚合收据](../../../../evidence/correctness/inference-g02-aggregate-acceptance/receipt.json)
逐份绑定 13 项证据。它是 goal 工作项裁决；当前模块状态仍由
registry（已删除的历史文件：`docs/implementation/module-status.json`）决定，G04/G05 的活动候选不在此一并关闭。

## 原要求与实际证据

| 要求 | 已完成范围与证据 |
|---|---|
| 多 Context/Queue 的自然联合入口 | [原共同卡](inference-g02-multi-queue-batch.md)及[共同证据](../../../../evidence/correctness/inference-g02-common/receipt.json)：per-lane Context/Queue/staging，只有原共享资格成立才共享 metadata；真实 CPU RMS/Cast→CUDA page attention→CPU Cast/Neg，F16/BF16、两 requests、重排和旧结果；source/static/shared、宿主风险及 36 个 OOM 点 |
| 精确页权限与实际退休 | [宿主资格](../../../../evidence/correctness/inference-g02-exact-read-groups-host/receipt.json)和[三交付/直接消费者](../../../../evidence/correctness/inference-g02-exact-read-groups-delivery/receipt.json)：多个 ReadLease、共享块、未引用块可写、部分接纳回滚、两个未知原生 prefix 都退休后才释放；三 host sanitizer、64 次无热 allocator 调用 |
| CPU+Apple/Vulkan 真实分页数学 | [原生联合终审](inference-g02-native-paged-terminal.md)：Apple F16/BF16、Vulkan F16，D64/128/256、page16、单/混合两 requests、乱序/重复/尾页/sentinel、两个 attention 节点和固定 active 输出；原 source216、长程1152 episodes、四 SDK、六 host sanitizer、Windows compile/link及原拒绝/失败/取消/旧结果/退休风险 |
| 原三个 Windows 未完成命令对应的门 | [完整默认集合裁决](../../../../evidence/correctness/inference-g02-original-windows-acceptance/receipt.json)：CUDA804、Vulkan851 全部通过。实际执行同一原默认 GTest 集合，每例独立进程串行；不声称原单进程 CTest 命令实际跑过 |
| 补验发现的 Vulkan correctness 缺陷 | [CE15/16 施工卡](../compute/inference-g04-g05-cache-attention-domain.md)：重复导出根拥有独立 binding；原 computed-F32 phase 在绝对值不超过 2^20 时使用 split-2pi/FMA 归约。不可变产品 `2b60d9a34be12fda9a9e3fdc26f7a61b495c3be7` |
| 新修正的受影响资格 | [Windows 正常 source](../../../../evidence/correctness/inference-g02-vulkan-alias-phase-windows-stage/receipt.json)、[默认851与18项定向复验](../../../../evidence/correctness/inference-g02-vulkan-windows-default-acceptance/receipt.json)、[实际 Air 三交付](../../../../evidence/correctness/inference-g02-vulkan-alias-phase-air/receipt.json)：公开 Direct/Graph 独立 F64 相位方程、18 个位置、别名发布/旧结果/关闭；Air source 另有实际独立端口回归及 metadata 失败/move/清理 |

## 七门与成本

七门 `scope / architecture / correctness / delivery / usability_docs / red_team /
engineering_performance` 在上述原有限范围全部 PASS。

依赖、许可、公开组件和 C++ 用法沿原合同；复用既有 CPU/native 数学、F1/F2、G03、
LaneBinding、ResultSlot、LeaseSubmission 与实际原生完成，没有新 executor、controller、
队列或 FFI。[batch 指南](../../../user/batch-attention.md)保留完整调用、权限、失败与费用。

候选后独立反查了 native root 与每个 caller port 的对应、冲突输出的实际 copy、
metadata 容量与清理、相位域边界/零符号/域外行为、F32 乘数合同、fixture 编码承载、
SDK 消费路径及 unknown completion 退休。原 842 个不受测试修正影响的 Vulkan PASS
与九个实际修正后 PASS 合并；九个对应 Direct case 也通过，不缩减原集合或放宽容差。

工程性能为 **P0**。per-lane residency、metadata 上传/准备与等待继续收费；原
LeaseUsePool 去重最坏 O(guards²)，不由 G02 冒充 G06 低成本规划。CE15 只在冲突
导出增加对应输出字节的 native copy；CE16 增加有界归约的乘法/round/三次 FMA，
没有额外 tensor、dispatch、allocation 或等待。原 encoded P1 没有 Sin/Cos 或冲突
导出，按实际分支复用，不重复采样或新增性能结论。

## 停止与保留范围

原压力中断、缺 fixture/layer 环境、构建失败、数值失败和编码断言均保留。
[最终测试诊断](../../../../evidence/correctness/inference-g02-vulkan-final-test-diagnostics/receipt.json)
明确区分原 FAIL 与修正后 PASS；Air 首轮生成物 SHA 检查发现旧 unary 对象不可复用，
在测试前补编该对象，原配置/构建费用仍在收据中。

停止重复原 G02 有限门。仅已验旅程的数学、权限、版本、失败/退休、结构成本、交付
缺陷或新的具名消费者机制触发重开。Vulkan BF16 joint 不因此扩大准入；host sanitizer
不签 driver/shader/private pool；native private 上界仍 UNKNOWN。active/delta 与 decode
规划、packed 消费、输出选择、重叠流水和 CPU/private 预算继续归 G06–G10。

本裁决不声称所有模式完全对称、capture、单机多 GPU/多机或整引擎 token/s 已完成。
goal 整项计数更新为 **3/10（G01、G02、G03）**，不把它当作工作量百分比。
