# C1：CPU 制品需求与成熟承载定界

<!-- MODULE REVIEW: SCOPED -->

## 当前施工身份

- 状态：SCOPED，绑定compute_ggml_runtime、compute_provider_ggml、compute_provider_native；compute_ops只作源审计依赖，保留原AUDITED记录；当前只审计，未授权产品代码。
- baseline：`a09e9642d91e5c13227bd68462ab426f2e69c462`；已核对Windows HEAD/worktree干净，不回退计划基线后的提交。
- owner范围：compute_ops、compute_ggml_runtime、compute_provider_ggml；涉及已借用程序时compute_provider_native，仅审计，不授权产品修改。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- 真实trigger：Q01–Q11缺少公开格式/CPU映射，DSpark与新版/sidecar不在现有精确资格；旧CLOSED不能代签。
- 正常consumer：八目标模型真实混合权重、量化embedding/专家、MTP/DSpark；正交压缩矩阵/Gather/重复expert。
- 当前来源：gap-list JSON、既有header/source收据、本地Qwen3.5-2B；新Pro-0813/Flash-0731官方与GGUF固定revision。
- 工作范围：104主档位/94待对账实例、3sidecar、2新官版、3非GGUF族；角色/物理dtype/block/group/shape/精度/state与MTP适用性。
- 外部读取只取config/index/header及具名forward；核对所有实际分片，不能只用文件名或第一分片签全制品。
- 热门度只用仓库级信号辅助选择，不捏造逐文件下载排名；104快照为固定对账对象，不执行中追新发布。

## 待审计的有限输出

以下是计划输出，激活时须完成源码/生命周期审计并冻结实际FIX；不以本表假写AUDITED。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| D1 | 确认每个Q格式实际tensor/端口角色，归并重复format/program/signature；保留全部对象与跳过理由 | 每个制品都有精确身份、required regions与header证据级别 |
| D2 | Pro/Flash新版DSpark、Qwen Q4_0 MTP与sidecar/共享weight映射，区分旧NextN | 每个分支对应原forward及状态/完成边界 |
| D3 | 每个非GGUF族最多两个具名CPU成熟候选，查packing/scale/codebook/permutation与实际CPU路径 | BORROW/ROUTED可实施，或给具体反例和待用户裁决范围，不能签SELF |
| D4 | 为C2–C4绑定实际owner、API草图/生命周期与有限FIX | 在写产品前合法AUDITED，现有D5仅按需要的有限范围重新定界 |

## 验证、预算与停止

- 一轮metadata/来源审计；每个缺事实最多两条具名来源/取得方式，不无限代理/版本/peer循环。
- 不重建、不跑P1，不重新审查旧模型全部历史；真实counterexample才定向恢复必要证据。
- 若某三非GGUF族无合格CPU carrier，保留精确阻塞，不排除后假称全计划完成。
- 完成D1–D4后交接C2；当前卡只绑定有限审计，审计/metadata不能作运行PASS。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。

## 当前有限审计绑定

- compute_ops保留原D5 AUDITED记录，本包只读审计其格式/操作面；新CPU产品修改之前须完成新的有限审计并合法绑定新AUDITED卡，不从旧卡取得扩张授权。
- 三个CLOSED owner由新格式/DSpark/non-GGUF成熟承载及ggml版本事实触发合法新scope。
- Native只审计已借用ATen/原版CPU程序与旧规则软件边界，冻结全部HYLE自研算子；没有性能优化授权。
- compute/graph core、CPU facade以及全部GPU状态保持，不把审计顺手扩大到这些已关闭/阻塞模块。
- D1–D4一次审计，现有快照/metadata parser/cache优先复用；head/index只取必要metadata，不下载大权重。

## 实际发现（C1进行中，未签产品资格）

1. 复用12个完整GGUF画像/223分片；其余603个分片/sidecar在同一metadata批次补齐。
2. Qwen3.8-27B文件名Q4_0的MTP实际18 tensors，Q3_K/Q4_K/Q6_K/F32；不需要因为文件名额外
   重写Q4_0 MTP数学，仍需完整精确分支验收。
3. Qwen2.4T/Kimi的UD-Q1_0、UD-TQ1_0、UD-TQ2_0使用type66/65/64，不是标准Q1_0/TQ1_0/TQ2_0。
   固定Unsloth `iq1-narrow-grids-b10775` SHA `46cbf0e95786fe8f5b7c0e86d57aaf8f8eceea7f`
   定义为IQ1_XXXS/IQ1_XXS/IQ1_XS，实际block38/42/46 bytes每256值，CPU原vec_dot与解码源已找到。
   C2必须为该成熟扩展定版本/ABI，不能只升级stock core再猜映射。原Q09–Q11名称候选依法修正，
   仍为三个格式接缝；标准Q1/TQ仅在header真正出现时准入。
4. 已锁LibTorch2.8 CPU的packed-int4原操作在Windows实际24个案例通过：F32/F16/BF16，
   group32/64/128/256，M1/M5。它是原库carrier探针，不是HYLE接线/RUN_PASS；NF4/MLX的公式与
   packing/precision仍需独立核对，不凭“4-bit”外推。
5. 用户最新口径已更新总计划：单一合格成熟carrier正确接入/P0；多家合格成熟重叠域批量比较；
   不比较自研或逐档位重建/计时。
6. 用户明确每个模型算子有不同实际量化精度。D1输出必须按tensor角色分别保存权重storage、
   activation、accumulation、state和output精度；header只证明存储，forward证明执行边界。
   文件整体Q4/Q8标签、相同名字的MTP/DSpark和“4-bit”不得授予单算子资格。
7. Unsloth IQ1扩展源基于ggml0.22，不能整树替换当前0.24并声称升级。C2需在固定新版core上
   借入原版IQ1扩展或给同等明确的版本方案；原数学不改，不把65/66重解释为标准Q1/TQ。
