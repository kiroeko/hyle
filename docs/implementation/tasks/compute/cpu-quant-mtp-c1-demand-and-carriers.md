# C1：CPU 制品需求与成熟承载定界

## 计划身份

- 状态：PLANNED，未激活、未绑定registry；没有SCOPED/AUDITED产品授权或执行PASS。
- baseline：`5b0044e2a10d11fdfce54893020cfcaaed47b0c0`；激活时核对Windows实际HEAD/worktree，保留后续及用户修改。
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
- 完成D1–D4后交接C2；当前卡仍是计划，registry未绑定，审计结果不能作运行PASS。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。
