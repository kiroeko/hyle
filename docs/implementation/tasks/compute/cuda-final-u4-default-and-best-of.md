# CUDA U4：合理默认与有条件 best-of

<!-- MODULE REVIEW: SCOPED -->

baseline `c64185195d7e4968d802e8d53567a1095b0fa07f`，Windows工作树干净；
U2/U3已取得联合资格，按用户“继续完成goal，避免无用重复”授权推进本包。
只读审计完成后绑定当前AUDITED owner；本scope不授权立即修改产品或采样。
主需求为 [共同模型／格式／算子目录](../../../architecture/model-compute-catalogue.md)。

## 本轮有限范围

- consumer：普通CUDA Direct/Graph使用固定模型的原tensor/格式与完整region；正交消费为
  缺性能证书、模型/revision/driver/compiler不匹配时仍可绑定合法成熟程序。
- 实际反例：ggml Context.qualified由SM89 selector决定，supports整体拒绝其它SM；CUDA
  façade只有全局ProviderBinding优先级，无法逐operation/ports/完整成本解释默认。
  CCCL把SM89和有限runtime范围共同作为全能力门；真实AOT DeepSeek的SM89/Driver13040保持。
- owner候选：compute_route_cuda、compute_provider_ggml_cuda、graph_provider_ggml_cuda，
  必要时compute_provider_cuda的CCCL冷准入。现CU BLAS/Graph route CLOSED不因查看源码重开。
  Torch/FlashInfer只有实际必要变化才纳入；CPU及其它GPU产品不改。
- 有限FIX候选：D1同源冷默认装饰、D2原ggml能力与性能证书分离、D3CCCL真实能力/版本门
  与旧SM性能域分离、D4有限重复域资格化与固定规则/文档。不增加公共Router或热vtable。
- 私有默认策略复用ProviderAccess冷装饰与现RoutePolicyId，原资源准入先执行，热执行表保持。
  模型/制品/phase/真实ports/软件hardware事实沿既有selector，不增用户rank/timing参数。
- 性能先列真实重叠域/成本：原数学、precision、ownership、资源/完成边界相同才能计时；
  单provider不比；unique resource域不伪输赢。优先复用严格相同输入身份的既有证据，
  原Marlin/旧API benchmark不重跑、不签新规则。近似或未资格化采用具名成熟工程默认。
- 预算：本包一次源码审计、一次批量实现与影响验证、一次稳定candidate的适用P1批次；
  默认n15，无升级计划、至多一次具名接线纠正，不新增仪器。每30分钟报告实际产出。
  前两小时至少完成D1--D3实现/适用验证并裁决D4范围，出现新公开合同问题停止该项定界。
- 验收：冷匹配/unknown与失配、原precision/bounds/resources、普通两入口/provider/policy绑定、
  原native硬资格拒绝、source/static/shared适用小纵切/公开头、docs/closure/统一LOC。
  未影响的U2/U3完整数学及交付格按产品/依赖/合同身份复用，不因文档/测试小改重新全集。
- stop：固定全部默认及适用重复域裁决完成后进入U5，不扩算子目录；适用未验不PASS。

## 用户要求：缺性能证书不等于缺能力

已测 best-of 只覆盖有限 GPU／SM、工具链／编译 recipe、驱动、Toolkit/runtime、Provider 版本与 workload。
正常用户只选择 CUDA 路线，HYLE须按真实 operation/ports/precision/shape/resources/region 提供合理默认：

1. 命中适用已测规则时绑定已测优选。
2. 未命中或性能相关事实未知时，绑定具名 **ENGINEERING_DEFAULT** 成熟实现，依据原格式直算、
   完整融合、转换／同步／内存／准备与重放成本、原库原生设备选择及交付可靠性，不能标 WIN/SOTA。
3. 只有原程序／设备能力／编码／精度／资源／版本确实不支持时明确拒绝；无performance证书本身
   不能禁用成熟能力，不能要求用户排序 Provider／指定内部 kernel／提交 timing profile。

默认不是全域固定 provider order。一个默认可按 datatype、权重format/group/scale/codebook、
模型／制品身份、阶段与融合边界分支；未知模型仍按真实 descriptor 合法通用执行。
Direct与Graph用同一能力／默认依据；Graph保留成熟region资源/融合，不拆库内kernel来竞速。
同名型号或SM不自动继承旧WIN，编译flags/Debug/IPO/driver/Provider升级只撤受影响性能资格。
仍可执行的成熟默认不得被静默减去。已拒绝/OOM/失败不能伪fallback到CPU或发布错误输出。

## 能力资格与性能资格分开审计

| 现状／条件 | 本轮处理要求 |
|---|---|
| cuBLAS dense准入与原SDK设备/类型能力 | 原精度／资源成立即可成为成熟通用默认；有限性能规则另外匹配 |
| ggml CUDA 整个Provider目前限SM89 | 查明哪些是原程序/数值必要条件、哪些仅旧单机证据边界；补必要correctness/portability后接通合法原库基线，不凭性能缺证禁止所有GGUF |
| 原版SM89 AOT及Driver API13040限定 | 原二进制/数值能力条件继续保留，先找同合同成熟generic承载；不能为“default”盲删真实硬门 |
| Linux FlashInfer与Windows缺席 | 按实际平台成熟替代接齐共同能力，Provider数量不必相同；缺旧FlashInfer证书不禁用同合同ggml attention |
| Strict memory/capture／异步与checked IDs | 默认选择尊重实际资源／失败／完整完成边界；高成本或不支持需自然诊断，不能漏检查来获胜 |
| 自研CUDA算子 | 仍冻结至成熟接入完成；U5只保留完整优势或具名独有需求，默认不因先写了就优先 |

硬件资格扩展必须有原源码能力／数值合同依据与适用实际证据；不能用本用户要求给未跑硬件写PASS。
可用性域与性能证书域各自列明，不把历史SM89限制默认为最佳长期能力边界。

## 预声明的默认验收

- 硬件名／性能证书、编译器／flags／Debug／IPO、driver release/API、Toolkit/runtime／SDK／Provider版本
  不匹配时撤性能特化，命中合法成熟工程默认；真实设备/程序资格不满足则精确拒绝。
- 模型／制品身份缺失、模型名相同但revision不同、量化文件名相同而tensor格式不同、phase/region改变，
  按真实端口匹配默认，不猜dtype、不继承旧证书；普通prepare/compile仍自然。
- 默认冷绑定、复用独立Invocation、变化payload/replay，运行无搜索/测速/再裁决；支持集与实际三交付一致。
- 每个真实重复域固定少量真实normal/holdout；P0先查原接线/成本，同合同完整成本P1有界。
  单成熟carrier只正确性/资源/P0；接近或预算内不确定按原成本/可用性工程默认快速决断。
  未跑或preflight失效不签WIN/LOSS，不无界升级n或调仪器。

完整U4审计前不启动P1；当前原源码/窗口有限事实仅作具名输入，最终逐合同默认表与适用实测规则必须分开交付。
