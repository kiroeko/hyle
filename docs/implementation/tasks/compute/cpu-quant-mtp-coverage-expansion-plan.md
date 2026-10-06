# CPU 热门量化、MTP 与 DeepSeek 新版接入计划

## 当前授权与基线

- 用户于2026-10-06已明确要求创建新goal并实际实施CPU扩覆盖；新的goal已创建，当前执行C1。
- baseline `5b0044e2a10d11fdfce54893020cfcaaed47b0c0`，执行时解析完整commit并核对HEAD/worktree；Windows canonical，不回退。
- 本计划对应[缺口清单](cpu-popular-quant-mtp-gap-list.md)及[固定元数据](cpu-popular-quant-mtp-gap-list.json)，
  是新的CPU扩覆盖计划，不恢复旧M/C/D施工队列、旧goal、旧进度或GPU阻塞项。
- 当前模块状态仍只在[module-status](../../module-status.json)。C1已激活为SCOPED，其余4张卡仍是**待激活计划卡**；
  开始每包时绑定其真实owner，合法进入SCOPED，审计后AUDITED固定有限FIX，再写产品。
  现有CLOSED不因本页自动重开；现有AUDITED范围改变必须重新定界；BLOCKED不自动恢复。

## 目标和数量

让普通 `compute::cpu::create_context()` / `graph::cpu::create_context()` 在Windows/Linux x86_64，
通过成熟Provider自然执行下列已准入计算，保留原量化格式、状态精度和低成本重复执行。
用户仍只选择CPU构建/路线，不枚举Provider、不提供timing profile。

| 范围 | 数量与边界 |
|---|---|
| 确定核心接入工作 | 11种GGUF格式 + 4种DSpark计算片段 = 15项；不是15个新数学算子 |
| 主GGUF档位盘点 | 八仓库104个固定档位；其中94个缺对应精确资格或需revision对账；逐档位是验收格，不是任务卡 |
| 新DeepSeek | Pro-0813、Flash-0731官方混合checkpoint及对应Unsloth量化主干 |
| 已发布sidecar | Qwen3.8-27B Q4_0 MTP；Flash-0731 DSpark Q8_0/BF16三份 |
| 条件待审范围 | NF4/double quant、AWQ/GPTQ/compressed-tensors、MLX int4三族；C1固定实际制品与CPU成熟承载 |
| 本轮自研数学 | 0；所有既有自研算子继续冻结，不优化、扩充、重开淘汰竞赛；SIMD基础保留 |

主纵切是上述LLM/MoE真实混合权重与MTP/DSpark；正交纵切是非模型名绑定的压缩矩阵乘、
量化表Gather和grouped expert路由/重复索引，用于反证接口不是按文件名或benchmark形状特化。
模型解析、权重加载、tokenizer、采样、draft token接受/回滚策略、服务与联合CPU+GPU推理仍属消费方。
Mac CPU/其它架构不由本轮代签；GPU功能、原验证和阻塞保持原scope，不宣称新版GPU资格。

## 五包顺序与任务卡

| 包 | 核心输出 | 包完成的具体条件 |
|---|---|---|
| [C1 需求/承载定界](cpu-quant-mtp-c1-demand-and-carriers.md) | 每个制品的真实tensor角色/格式/shape、MTP/DSpark适用性、成熟CPU承载与有限FIX | 104档位及3sidecar已对账；新官版和三非GGUF族有明确准入/边界，未知不冒充已接 |
| [C2 ggml升级/软件资格](cpu-quant-mtp-c2-ggml-upgrade.md) | 固定新版、资源patch/ABI/ISA、CPU构建隔离、旧Native规则peer版本约束 | 新CPU runtime真实执行；旧GPU构建/资格不被静默改写，SDK中没有未隔离的两版ggml对象 |
| [C3 格式批量接线](cpu-quant-mtp-c3-format-bindings.md) | Q01–Q11及C1已准入其它格式，一次参数化接普通Direct/Graph | 实际需要的MatMul/GroupedMatMul/Gather端口可用，原编码/精度/失败与资源合同正确 |
| [C4 MTP/DSpark组合接线](cpu-quant-mtp-c4-model-branches.md) | 四种DSpark片段、完整主干/分支和三sidecar计算消费者 | 数值、cache反馈、共享权重、Markov/confidence及精确制品资格真实通过 |
| [C5 批量验收/收口](cpu-quant-mtp-c5-batch-qualification.md) | 去重后的全部CPU准入格、两系统三交付、文档/状态/成本/LOC与清理 | 每个必需格实跑PASS；拒绝/未运行不计完成；完成后停止 |

C1按已有快照先批量读取所需header/index，不重新翻所有历史模型研究，不下载巨型data section。
C2/C3/C4可共用同一CPU producer和参数表；批末API/程序稳定后再冻结正式候选，构建并行2、
重型验证串行、test并行1。包内最小接缝检查不是新的全量验收轮，不人为逐算子暂停。
候选稳定后，各适用owner suite/交付组只集中跑一次；源码失败先修成本/接缝，最多一批定向纠正。
C5复核C2–C4已有完整证据并补齐总矩阵，不把交付/资源/普通入口首次接线留到最后。

## 准入与范围停止规则

1. Q01–Q11按真实tensor header确定实际使用端口；不强造11×3的未需求组合，也不撤掉已准入格式。
2. 104档位全部进入C1对账；同文件/blob/header/语义身份的资格可以复用，repo revision变化不自动
   签PASS，也不机械重测未变数学。主权重与MTP/DSpark revision、shared parameter和配置必须匹配。
3. 三个非GGUF族是必须完成的有限承载审计：每族最多核对两个具名成熟CPU候选及一条实际路径。
   若可借用/组合，列入C3；若只支持其它路线或确无合格CPU承载，列出精确制品/反例及取舍，
   请求用户决定是否改变范围。不能自行自研，也不能把DEFER/拒绝当全范围完成。
4. 新DSpark不是旧NextN；源码`mtp.*`名称不授予旧资格。Pro-0813未列独立sidecar不证明没有嵌入DSpark。
5. 新事实若需新数学、公共Graph执行机制或全局治理重写，停止该项并具名定界；其余独立接线可继续。
   不用一个有界疑问反复推翻整体架构，不重开旧Vulkan A/A或CPU自研竞赛。

## 升级与路由原则

现锁ggml `456172ec733a135778adcd32d00e576a58232e45`；已观察候选
`ffa4e8b80930029a35991f94e7c8a93cd67730ab`，llama.cpp源码参考候选
`d7a695ef679138c13d86359b84c1731d36213d32`。C2审计后固定准确源码、archive SHA/资源patch和ABI，
不在构建时追master，也不把llama.cpp模型引擎引入HYLE产品。

ggml现为四路线共用源。C2优先评估**按构建envelope选择固定版本、每个SDK只有一个ggml core**：
CPU-only采用新版；含GPU的既有SDK保留已验原版及patch，明确各SDK实际CPU能力。
若此方式无法保全已有公共组合/ABI，C2必须另给有证据的最小方案；不把两版未隔离的POD/符号混入
同进程，不静默撤回GPU能力或伪签GPU新版。方案在写产品前冻结，不能施工到一半靠重命名掩盖冲突。

旧187条Native WIN只相对原成熟peer：软件身份不匹配时成熟默认，或复用未变peer的精确资格。
本轮不重跑全部自研竞争比较来保留旧标签；不删除SIMD或恢复Native MatMul。
模型/制品身份与真实op/端口/精度/资源共同匹配，prepare绑定，submit不解析模型名或重新搜索。
Provider可保留完整融合片段，禁止为了HYLE逐核调度拆开成熟region。

## 验收和性能预算

- 正确性：原格式/scale与累加精度、动态A8和F32状态、边界/空/重复expert/越界、changed replay、
  MTP共享参数、DSpark cache wrap/reset、独立slot、完成失败/close/OOM；不把理论可表达签为RUN_PASS。
- 交付：Windows/Linux source、installed-static、installed-shared普通C++，DLL邻置、SDK搬移/offline
  与依赖资产完整；header/API/packaging影响由现有selector选择，适用风险/跳过理由保留。
- 性能以**P0成本结构**为主：不逐算子/档位进行赢家比赛；prepare一次，热路径无编译/搜索，
  无整库专家/权重每轮解包，保留融合和workspace/原库资源复用，必要转换、copy和同步真实记录。
- 若仅P0不能裁决具名接入成本，C4固定最多两个代表完整程序（量化MoE与DSpark），同原成熟调用
  比较完整完成边界；API固定后一次P1 bundle，沿原5%工程门和统一资格，算法优化0、最多一批接缝纠正。
  没有明确风险不自动启动P1；不以n31、换姿势或新仪器无限采样。必要但未运行仍为未验。
- docs/state/已证明格式变化不构建、不跑P1；不重写selector、closure或性能统计核心。
- 每包阶段变化用既有closure/docs检查；提交推送`[skip ci]`。Windows唯一产品真相，WSL仅前向副本。
  每包完成按统一29路径`loc-report.py`统计Compute+Graph实现代码（不含tests/vendor/bench），
  不把文档、研究、旧CLOSED或“接进target”算产品进度；清理本包有owner校验的临时消费者目录。

## 终局

启动时实现新增0、新运行资格0；当前只激活C1审计，registry按真实owner进入SCOPED。所有已准入CPU必需域/两入口/资源与交付
完成后，明确报告“CPU量化与MTP扩覆盖完成”，列出硬件/软件/制品限制并停止。
存在用户未裁决的非GGUF缺口、未通过格或升级ABI阻塞时不得宣布整个计划完成。
不会由本计划自动扩到GPU、模型引擎或自研算子大全。新goal只能按用户明确请求另建，旧goal保持已完成。

## 本次启动

实际HEAD `a09e9642d91e5c13227bd68462ab426f2e69c462`，工作树干净；保留计划基线后的提交。
新goal目标就是本页完整CPUscope；旧已完成goal不恢复，GPU阻塞不解除。
