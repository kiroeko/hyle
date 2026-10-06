# C5：CPU 全批验收、文档与停止

<!-- MODULE REVIEW: SCOPED -->

## 计划身份

- 状态：SCOPED，C4有限资格已交接，开始全准入矩阵与证据审计；本卡尚未授权产品实施或签执行PASS。
- baseline：`400927ff7`；Windows实际HEAD与干净工作树已核对，保留C1–C4全部后续提交。
- owner范围：compute_ops、compute_provider_ggml、compute_provider_native。registry当前C4 AUDITED资格保持；
  本卡完成有限审计后再按AUDITED交接，不伪造AUDITED→SCOPED状态跳转或重开其它CLOSED/BLOCKED。
  CPU/Graph facade、公共头与交付直接消费者按实际未变身份复用；没有新增产品owner。
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

## 用户最新性能口径

单一合格成熟carrier完成正确接入与P0；多家成熟库具备同合同重叠域时按总计划批量比较固定默认规则。
不比较自研，不为每个算子/档位重建或计时，最多两个代表消费族共用一份稳定候选P1。

## 本包定界与审计输入

- C1完整header/index为准：104/104 GGUF主档位、826/826 shard/sidecar、152/152 Safetensors shard。
  存储表有25种实际GGUF类型；I32 hash/索引表按整数控制端口处理，不能当MatMul权重。
- C3/C4真实收据为执行输入：11新增GGUF及原格式、四artifact decode、三MLX格式，
  新官方主干/三阶段、三sidecar的普通source两入口；两系统安装SDK40项与资源20项。
  复用必须匹配产品/软件身份、语义、实际角色/shape/state及入口；头完整不等于RUN_PASS。
- 对全部角色按实际算子和端口去重，记录制品到资格的映射；同格式不同真实消费合同不能合并。
  保存尚未执行的格，先把有限差异固定为FIX，再一次参数化接验证消费者；不新增数学核。
- 源码/公开API/Provider没有本包升级授权；若发现产品接缝缺陷，先在V1–V4边界明确反例/FIX与owner。
  新数学或必须改变长期合同的缺口须具名定界，不能用自研或缩小必需集合补齐PASS。
- 每个系统一个稳定候选批次，构建j2、测试j1；只补缺失适用格。已有SDK与资源身份相同则复用，
  不逐档位构建、重复三交付或重跑性能。最终范围裁决、用户文档与红队属于本卡，不跳CLOSED。
