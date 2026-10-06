# C5：CPU 全批验收、文档与停止

## 计划身份

- 状态：PLANNED，未激活、未绑定registry；没有SCOPED/AUDITED产品授权或执行PASS。
- baseline：`5b0044e2a10d11fdfce54893020cfcaaed47b0c0`；激活时核对Windows实际HEAD/worktree，保留后续及用户修改。
- owner范围：汇总C1–C4真实owner，逐项复用其candidate/receipt；registry仍只有当前owner card，不另建状态系统。
- 总合同：[CPU扩覆盖计划](cpu-quant-mtp-coverage-expansion-plan.md)；事实：[缺口清单](cpu-popular-quant-mtp-gap-list.md)。
- 本卡是一个批次，不是逐算子/逐模型队列。CLOSED按真实trigger新开；AUDITED范围变化重新定界。
- 性能：P0为主，唯一可选完整程序P1只按总计划条件固定；无自研竞赛或新数学。

## 范围、真实消费者与来源

- scope：全部C1准入的104档位对账结果、3sidecar、2新官版及非GGUF已准入对象，两入口、资源与两系统三交付。
- 不新增算子/Provider/优化或第一次接线；只合并真实证据并补缺验收格。
- 新版当前能力表与旧22画像/P1历史分开，不能覆盖旧失败、被拒域或缩小范围后冒称完成。

## 待审计的有限输出

以下是计划输出，激活时须完成源码/生命周期审计并冻结实际FIX；不以本表假写AUDITED。

| ID | 本包动作 | 结束条件 |
|---|---|---|
| V1 | Metadata驱动完整CPU验收矩阵：匹配实际角色/格式/state/全部required region | 每格普通Direct/Graph实际输出/完成/失败通过；未运行/拒绝/数值失败不计完成 |
| V2 | Windows/Linux owning+headers、source/static/shared、DLL/loader/relocation/offline | 候选/source/toolchain/SDK身份匹配，复用理由明确，无全模型/未知硬件外推 |
| V3 | P0与已声明的唯一代表性成本证据汇总；用户CPU构建/default规则与文档 | 没新P1主张不强制竞赛；必要未验项不伪造PASS，旧Native规则不越过peer版本 |
| V4 | closure/docs/selector reasons、统一LOC、精确owner临时目录清理、skip-ci提交推送 | 最后红队无未决必需风险后ownerCLOSED；完成即停止 |

## 验证、预算与停止

- API/算法稳定后一个不可变验收候选；复用已完成包的匹配身份，不因docs/state重建或重跑P1。
- 只核对当前已准入scope，不扩大新发布/其它路线；无必要不重复owner suite。
- 存在未裁决非GGUF范围或任何必需拒绝/失败时保留具体阻塞；不得以有结果等于全PASS。
- 成功后明确报告CPU量化与MTP扩覆盖完成、限制和全部实现LOC，停止等待下一阶段；不开新goal/算子大全。

普通C++入口、所有权、并发/完成与失败语义在SCOPED/AUDITED阶段固定，不新增公共Provider排序或timing参数。
公共头/API/package变化遵守现有selector；P0查allocation/copy/lock/syscall/metadata/热冷分界。
产品改动前真实owner必须AUDITED；每轮适用检查及closure/docs实际运行，未运行不写PASS。
只有Windows可改产品，WSL/mac仅验证副本；构建j2、测试j1、重型串行；临时目录按owner校验清理。
