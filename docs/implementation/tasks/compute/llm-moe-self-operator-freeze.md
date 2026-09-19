# 旧算子冻结、旧施工授权撤回与最后退场

## 当前约束

2026-09-19 用户要求停止默认自写，重写所有活动计划并重启 goal。冻结产品基线
`ebe5af5109afb01b029c1ba85484d03be61d6fe6`。本页是[新总计划](llm-moe-coverage-completion-plan.md)
的强制边界；原 C1 未完成，旧 FIX 不再允许续写。

## 冻结哪些实现

全体已有 HYLE 计算算法冻结，不限本轮新增文件、当前默认路径或某个平台：

| 范围 | 冻结内容 | 不受算法淘汰牵连的基础能力 |
|---|---|---|
| CPU Native / SIMD 算子 | tensor/cast/index/norm/reduce/activation/attention/MatMul 等自有计算与 SIMD 算子注册 | SIMD 模块本身、公共通用 API、非推理消费者 |
| CUDA | 所有 HYLE `.cu/.cuh` 算术/寻址/归约/排序/scan/scatter 算法，CUB 包装中的自有算法部分 | 原生 stream、资源与完成/错误设施 |
| Apple | 自有 Metal（含 C++ 字符串内 kernel）、有序 scatter、检查计算中的自有算法 | MLX 原生调用、array/region 的资源生命周期 |
| Vulkan | HYLE shader、原生自有数学/索引与替代候选路径 | ggml 完整原版后端、设备/资源/完成设施 |
| 修改上游代码 | 保留 provenance 区分原版算法与 HYLE 改写的数学部分；后者同样冻结 | 原版第三方代码不冒称 HYLE 自研 |

[冻结源码清单](llm-moe-legacy-operator-freeze.json)绑定完整 Compute/Graph 产品路径与 Git blob。
这是保守的定位全集，**不是声称每个文件都是自研算子**：任何算法嵌在 adapter/Graph/header 中也
不得漏掉；该清单只绑定文件身份，不研究自研算法。资源胶水后续只有在 M1/M2 新卡明确有限 FIX 后才能改，不能夹带数学扩域/优化。
同样禁止新增文件绕开冻结。算法数值语义适配必须优先表达为上游已有运算组合，不默写新 kernel。

既有能力保留供过渡，不因冻结立即删除或改变公开行为。发现真实正确性/安全缺陷时，只允许在
具名 owner 卡审计最小隔离/修复及回归；不得扩域或为性能跳过检查。未覆盖必需域先接成熟替代。

### SIMD 的基础能力与算子专用增量分开

SIMD 模块本身、公共通用能力与非推理消费者保留。Compute/Graph 对 SIMD 的算子封装、专用
计算与路由参加 M5 淘汰，不能因为调用 SIMD 就免检。用户另明确：**上个 goal 新扩到 SIMD 中、
专为算子服务且 Highway/xsimd 均无对应实现的部分，也留到 M5 一起裁决**，不因目录名保留。
届时必须同时给出新增提交、专用消费和上游缺席证据，区分通用 primitive 与算子算法；不能仅因
Highway/xsimd 没有某功能就删除 HYLE 通用能力，也不能顺带删除基础库。

本次只核对变更来源：[SIMD 增量记录](../../../../evidence/correctness/compute-llm-moe-restart/simd-origin.json)
显示从前 goal 起点 `2e85b894` 到冻结基线 `ebe5af51`，`include/hyle/simd` 与 `src/simd`
没有产品改动，也没有触及它们的提交；因此该范围的新增 SIMD 专用候选目前为0。
此结论不评价更早 SIMD 实现，不扩展为旧 SIMD 全库审计；第三方接入前不研究其算法或比性能。

## 最后的有限取舍（M5）

1. 先完成所有必需 Provider 接入及真实覆盖核对，具名缺口按 M4 裁决。全部准入第三方接入完成前禁止扩张、调研或性能比较自己的算子；此前不测试旧算法输赢、
   不优化旧核；能复用的原始合格证据保留。
2. 同输入、同数学/精度、同 checked/alias、同状态、同完成与结果寿命；转换/融合/同步/
   workspace/驻留全部计入。以真实片段分组，禁止每个旧数学原子开研究项目。
3. 每路线仅一次冻结 bundle；不超过现有预算及31个有效配对样本/比较，双顺序与实际 workload
   A/A 不省略。准备最多30分钟；仅一次具名测量缺陷纠正、最多30分钟；**算法优化预算0**。
   既有 CPU Q2 blocker 必须先满足其原条件，不能凭新阶段追加性能预算。
   全阶段新增分析总额2小时、采样 wall-clock 总额2小时；到点按现有证据及成熟基线退场规则裁决，
   不按算子叠加调查额度。必要删除回归不计作性能采样，也不能夹带新竞速。
4. 有效 **LOSS 立即删除对应可完整替代域**；不找优化翻盘，不加样本。迁移普通 Direct/Graph
   路由，完成所选回归再提交删除。WIN 仅保留已证明条件域；平局/INCONCLUSIVE 无保留优势时
   选合格成熟实现并删除冗余，准确记录其并非已证 LOSS。测量失格绝不伪造赢家。
5. 替代不完整不能丢必需功能：精确列出未覆盖域回 M4，不保留整个重复实现。没有模型/通用
   consumer 需求的旧域不制造 benchmark，直接按有限清理卡退场。删除专有注册、构建和过时说明，
   保留独立正确性 oracle 与必要回归。SIMD 基础库明确排除。

## 旧 owner 的终止与恢复

本轮用户改变了原施工范围，旧 C1 的 AUDITED/SCOPED 不能继续作为产品授权。先将这些活动 owner
按原状态记录为 BLOCKED（外部需求变更使旧轮无法按原卡关闭），不签 CLOSED；随后只有在新总计划、
本冻结合同及 M0 有限需求/provider 审计卡落盘后，才经 BLOCKED→SCOPED 绑定 M0。
新卡的产品 FIX 为空，不能修改算法。此两次迁移分别提交并用 `--previous-ref` 验证，不改 checker。

原 `compute_provider_ggml`、两个 oneDNN owner 的既有 BLOCKED 保持原 record/恢复条件。
所有 CLOSED 与 QUEUED 保持原状态。新计划不解除 runtime abort 或 Q2 测量资格阻塞。
旧 C1 子卡保存已发生事实，不再保存当前施工队列；完整旧 A–AB 记录可由 Git 基线读取。

## 已有最后一批事实

CUDA SGLang AB 在 `9ac3c91e` 实施，`ebe5af51` 修正测试调用已有 Operation 重载。
W1 static provider/default Direct/default Graph 及头/源码合同已实际通过；
[原始收据](../../../../evidence/correctness/compute-llm-moe-restart/sglang-w1-static/result.json)
绑定候选、artifact 和日志。shared/WSL/真实2B三交付/P1 未完成，不能签 AB 或旧 C1 完成。
此后产品冻结；本次只改文档、状态与已有证据归档。
