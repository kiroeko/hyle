# F5固定candidate分离红队核查

product25e，root在实现冻结后单独只读检查，未启子agent。按原六FIX/真实consumer/
公开入口与逐风险回映反证，不从实现顺序或测试数量推导完成；未修改产品/API/算法。

| 反证方向 | 实际核查 | 当前发现 |
|---|---|---|
| 未映射/benchmark-only入口 | schema/容量/offset/facts/version/export；typed两schema/infer；native program/token/report；arena writer/frame；reference recipe/version/controller；Graph coordinator/route factory的普通任务 | 全部回到公开准备/运行/资源版本/成本旅程，无只服务benchmark的新增API |
| ID/旧token/溢出 | generation CAS饱和前拒绝；slot epoch界；schema owns/complete node layout/shared input同version；update完整indptr key | 无地址/shape冒充身份；没有归零重用generation或run隐式plan |
| 失败prefix/metadata混合 | H2D/plan先失效、失败poison/清所有token、原region同slot/mirror/shared节点只相同版本 | actual fault/unknown/suffix证据与源码一致；不靠清cached status恢复failed slot |
| alias/权限/生命周期 | F1 readonly/frame精确页/额外CPU reader/copy/published capacity拒绝；旧output防覆写；store/arena不互相keep-cycle | 原独立use/native retirement与guard保活，raw arena不制造独立writer权限 |
| OOM/partial clone/发布 | common/layout/802 caller冷OOM与native/CPU实际失败F2→F5/F4不发布；expected-base/旧版本 | caller堆证据不覆盖driver私有堆；不称取消可回滚已写字节 |
| ordinary字段/费用 | 原Graph两个state声明、匹配GCC对象/六submit实际P0；固定workspace/metadata M清零/上传/reference gather和wait | 无新增默认tax；UNKNOWN保留，reference不叫native paged/同吞吐/capture |
| API可用性/交付 | source9 disabled头/4负API、18 SDK原raw/hash/候选映射重校验、actual native-only没有CPU fallback | PUBLIC FI shared反证已修；历史失败与不同平台数学资格未合并成虚假对称 |
| 字段/线程/仪器外推 | 控制外部串行；新native callback release/acquire必须TSan，宿主ASan不看shader/driver；NVIDIA LSan控制与完整诊断 | 不把CPU/common san代签native；整进程LSan未PASS，不改源/加suppression消报告 |

审查发现两项文档遗漏而非新产品缺陷：guide须明确F2消费writer后的保留/complete与
abandon close/recreate合同，已按公头/原单测补明；原引擎估计的“A已closed、F4/F5未实施”
不能描述当前状态，已保历史调查并增加当前事实/实测LOC，不提前称整个goal完成。

没有在上述有限源码/API/资格边界发现需要退AUDITED的新产品事实。审查结果不是全goal
完成：native TSan实际运行尚待、组件all聚合环境与原预算未PASS、原engine后评估仍待。
本red-team结论对已核源码与证据边界有效，最终terminal仍须将未完成工程门解决并逐项
裁决；若后续Tsan/运行反证产品，必须重新回AUDITED，不能借此文掩盖。

原输入/范围见[completion audit](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/project/foundation-f45-completion-audit.md)，
当前产品只以registry/候选卡给出。F1–F3/A/F4/R1–R7已签部分未因“再看一遍”重开。
