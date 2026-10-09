# Vulkan：共同目录、成熟接线与最终收口

<!-- MODULE REVIEW: SCOPED -->

baseline `a19391c7efa84d1f0c42919e8192e4ce78d1ee32`，Windows工作树干净。
2026-10-09用户明确要求新建goal并执行。Windows是唯一产品真相；独立新goal已建立，
不恢复旧进度/施工队列。当前RTX4070/12GiB、NVIDIA617.14；实际Vulkan/SDK身份运行前记录。

## 范围与四包

需求只用[共同目录](../../../architecture/model-compute-catalogue.md)及
[逐制品端口](cpu-final-demand-carriers.json)：104 GGUF/559合同、80参数类、34packed/
45plain、八族，当前Pro0813/Flash0731、官方混合FP4/FP8及必要MTP/DSpark/Qwen3.5-2B。
自然Vulkan Direct/Graph、原buffer/command/fence/native完成、资源复用、冷默认/条件优选、
source/static/shared和文档均须收口。不是模型loader/engine/巨型权重驻留或算子大全。

| 包 | 权重 | 有限成果 |
|---|---:|---|
| V1 | 25% | 共同目录逐端口/原成熟source/现有接线/构建与self审计，固定有限FIX |
| V2 | 35% | 按成熟Provider批量接线，普通两入口真实数值/encoding/状态/失败与资源 |
| V3 | 25% | 合法未知条件default；真实重叠完整成本有限P1、self算子去留 |
| V4 | 15% | 三交付/离线搬迁/DLL邻置/SPIRV/许可/公开头、终审/LOC/临时清理 |

主纵切是固定模型实际量化投影/专家/embedding及完整状态分支；正交纵切是动态buffer与
布局/checked索引的常用数值管线。Provider管理其内部程序/融合/资源，HYLE管理边界，不
引入共享万能执行器。未知条件仍有合法成熟baseline或具名真实拒绝，不能假称全硬件SOTA。
平台/硬件/driver/SDK/compiler/provider、模型制品/phase/region和实际端口format/dtype/
shape均可冷绑定；hot不得查找/计时/裁决Provider。用户无需rank或timing profile。

## owner及旧阻塞

`compute_provider_ggml_vulkan`现AUDITED但旧卡只准入十格式/拒绝CPU扩展；本次依法重定界。
`compute_provider_iree_vulkan`现CANDIDATE，待明确原程序缺口后退回AUDITED，不能在
CANDIDATE下写产品。`compute_route_vulkan`/`graph_route_vulkan`及D5聚合`graph`原BLOCKED
保持：先只做source审计及独立Provider工作，不借新goal清空旧A/A预算/失败。
route产品改动前必须由具名测量修正/可说明环境变化或用户明示政策变更满足原
[resume_condition](llm-moe-d5-vulkan-performance-blocker.md)，SCOPED后审计有限FIX。
CPU/CUDA/Apple CLOSED不动；公共runtime等依赖只遇具名缺陷/新增source选择才定向重开。

已有[balanced-AA](llm-moe-d5-vulkan-measurement-balance.md)记录同轮paired与两侧独立
中位数可产生不同裁决；这是具体测量审计输入，不是旧P1 PASS。先固定同输入循环、
paired/时序/成本/边界的数学合同，必要公共统计接缝必须独立tool task，不能本轮无限造仪器。
具体唯一恢复资格按[同轮配对A/A独立卡](vulkan-parity-paired-aa-qualification.md)执行。
不恢复原n15/n31；不强求更严格桌面空闲/改系统模式/弹UAC。实现稳定candidate才一次
适用同拓扑A/A→固定n15、5% margin，正反/分配顺序和placement；不合格不能导入赢家。

## 原成熟路径与第一包

ggml Vulkan和IREE为当前两成熟Provider。所有HYLE数学/GLSL冻结；先接成熟库已有实现。
当前127合同/11格式无ggml Vulkan type绑定；其中119合同8格式在锁定456172ec原
backend/generator已有MUL_MAT/MUL_MAT_ID，另有8个压缩embedding需求未绑定。
这些是静态事实，尚未runtime PASS。[第一Provider批次](vulkan-parity-v1-ggml-native-bindings.md)
只补原8格式及实际embedding类型，保持原dtype/编码/资源判定。余8合同3种IQ1扩展、
34packed/官方新低精度/完整branch仍需具名source/API审计，不按库名认定有/没有实现。
无完整成熟路径时最多一次具名follow-up后列出具体取舍，不自动写新kernel或扩大目录。

## 批次与验收预算

每包先审计固定有限FIX，再批量接线，批末一个不可变候选集中build2/test1；不逐算子
重新配置/三交付/P1。每个新接线/fixture/ABI错误最多一次具名纠正，通过域按identity复用。
普通验收包括实际模型端口/完整region、changed input/weight/IDs、replay/state/reset、
数值/encoding/坏索引/OOM/close/cancel/lease，以及对应source/static/shared/许可/DLL
邻置/SPIRV/离线搬迁。有同合同成熟替代而无完整收益的self算子删除；必要checked/storage/
资源机制和SIMD本身保留，unique数学需真实需求/明确取舍。单carrier不造性能竞赛。

Windows RTX4070主正确性/P1；Air Mac16,12 M4默认SSH `kiro@172.22.10.198`仅MoltenVK
独立capability/portability，先读实际身份；Mac mini不可用。WSL是可丢弃副本，软件device/
编译不代签GPU运行。重型串行，复用受管sourcecache和少量producer；及时清理拥有者精确
临时文件，不删模型/user文件/共享cache，不改全局治理。selector按真实改动选门；
docs/state不build/P1。七门真实通过才CLOSED，适用未验不能PASS；预算到达具名裁决停止。

每包/重大变化/用户询问汇报已实现/已验/未完/阻塞和统一Compute+Graph非测试LOC。
采用[skip ci]提交推送，不用Actions。全部约定Vulkan能力和验收完成后通知并停止，
不自动扩算子大全。当前先V1审计/第一原库批次，不解除旧blocked或宣称Vulkan完成。

## 具名恢复条件已满足：重新SCOPED（2026-10-09）

[独立paired A/A](../../../../evidence/performance/vulkan-parity-paired-aa/README.md)唯一8process
PASS：相同原topology、双执行/分配顺序、24round×6调用、真实输出/资源/native终态，
原5%/10%门保持。源审计修正配对信息丢失，不靠更换目录/无限重测；原失败仍保留。
据旧resume_condition，`compute_route_vulkan`/`graph_route_vulkan`与D5聚合`graph`
依法BLOCKED→SCOPED绑定本卡，新覆盖改动须先AUDITED固定有限FIX；不直接CLOSED。
公共Graph仅协调覆盖/最终聚合，不因此授权重写共同Graph产品。旧正式P1仍未完成，
后续稳定candidate须同paired口径、相同完成/拥有服务与原controls；不重用旧未资格数据。
CPU/CUDA/Apple CLOSED保持，首次Provider原绑定工作及真实source记录继续复用。
