# CUDA 性能优选补齐与对应清退

<!-- MODULE REVIEW: SCOPED -->

baseline `3e9e71d0ab6a915d3682314e9102864638d1005e`，Windows工作树干净。
2026-10-08用户要求CUDA补至CPU同级，并明确“差不多就选主流，不无限性能比较”。
旧goal完成只代表覆盖/交付/工程默认；本轮重新纳入实测优选与对应清退，旧AA FAIL不改写。

## 四个有限工作包

| 包 | 权重 | 完成条件 |
|---|---:|---|
| C1 同合同候选与入口排查 | 20% | 全部实际重叠的能力/精度/资源/完成账单；一次具名AA成本排查，冻结有限矩阵和纠正 |
| C2 批量资格/比较与裁决 | 35% | 单承载不比，合格同合同请求一次固定n15；明显差距定winner，接近/噪声选成熟，不升级n/反复换姿势 |
| C3 冷规则及冗余清退 | 30% | 合格赢家有精确条件规则；无规则成熟默认仍可用；可完整替代且无优势的self物理删除，unique明确保留 |
| C4 差分交付与终审 | 15% | 受影响Direct/Graph/source/static/shared/失配/资源检查及七门、LOC、垃圾清理、提交推送后停止 |

当前consumer/需求目录与U23原数学和交付保持；不重新下载模型/重跑559合同，不改CPU或其它GPU。
scope owner候选仅compute_route_cuda、graph_route_cuda、compute_provider_cuda，原其它成熟库
只有具体接线反例才重开。它们旧CLOSED不授权本轮产品变化，先绑定SCOPED再AUDITED有限FIX。

## CPU 相同的快速工程取舍

- 支持/接线/原precision与全成本先成立；不同编码、舍入、完整region、ownership不伪同合同。
- 单成熟carrier只正确性/资源/P0，不造性能比赛。
- 同合同明显、合格且可复现的优势才能成为WIN冷规则；近似/不确定直接选成熟工程默认。
- 不为保住self无限优化；有完整成熟替代而无明确完整优势就清退。SIMD和路线控制/资源/完成
  与原借入库不误删；unique要有具名consumer/标准需求，不能只靠旧自研测试反证新设计。
- “同CPU级别”指真实候选审查、合格比较和规则/去留，不要求复制43条WIN或每个硬件都冠军。
  不能把仅工程默认标成全域best-of，也不能为增加WIN数扩算子或改用户形状。

## 有界执行和停止

C1调查上限30分钟：一遍已有失败/源码/原成本检查、一个明确机制诊断，不重测原8轮/升级样本。
C2适配器一次批量构建（至多一次具名setup纠正），一次短AA资格批及一批适用n15/四strata，
不升级31。AA未过的单项不进入正式采样，原FAIL保留；不能因少数无资格项阻塞已资格项。
模块adapter只映射workload/遥测，统计仍统一hyle-perf；公共工具如有真实必需缺陷另建工具卡。
首次四小时内必须有C1有限矩阵、批量准入和C2裁决产物；到预算按原门裁决，不继续研究。
源代码稳定再采样；批量接线/规则后集中验证，复用既有producer和最新U5 SDK，Windows DLL邻exe，
所有子进程隐藏、build2/test1、重型串行。文档/状态不build/P1。每包结束/重大变化报告实际结果
及统一Compute+Graph实现LOC；[skip ci]提交推送，不消耗Actions。

## 当前具体反例与预审

上轮adapter只有两次verify热身，nanobench warmup=0、minEpoch8ms、约12 paired round；
完整resident叶操作约10--30微秒，原日志同实现err达41.4%，包括没有ggml复杂管理的原native。
这些是已知入口事实，**不足以唯一归因GPU/系统噪声**，不应直接转成永久无比较默认。
先看原compare循环与native event/Completion，具名检验冷缓存/时钟稳定和CPU未运行成本，
完整终态等待仍收费，不能用纯CUDA event kernel时间替代完整consumer以获得漂亮比值。
本轮只在具体机制依据上纠正adapter；不提高margin、改pair/尺寸、拿更多n掩盖偏差。

完整C1审计及有限FIX前不改产品、不正式采样。三个本轮owner已依法CLOSED→SCOPED；其余CUDA owners保持旧CLOSED。

C1一次入口诊断已完成，见[原始结果](../../../../evidence/performance/cuda-best-of-c1/README.md)。
2048对称预热不足以消除同路径11/30us和7.4%差异，不反复增加预热/round或升级n。
已证adapter缺少D5已有的AA ready guard，补此窄guard，公共runner/统计不改；其验证待执行。

C1已冻结[206个有限请求](cuda-best-of-requests.json)的候选静态矩阵：32个语义variant、
三种实际float storage、主decode/小prefill holdout、6种storage cast和2个plain Gather。
这是8个重叠能力域的有界批量准入输入，不签206项已支持/已比较；每项仍须真实prepare与输出
门，单承载/没有同合同的域明确不比。Linux attention独立同合同region，Windows不伪造该Provider。

批量admission预编译发现adapter误用显式Operation构造，旧日志保留；只修adapter为既有
显式typed Operation，不改API或产品。初始诊断build加批量build两次已用，允许原同ID最后
一次这项具名纠正（累计3次），不再加预算/新ID；product/P1预算不变。

C1首次冷准入在ggml caller-workspace检查停止：zero-byte BufferView仍有owner，
不能传给provider-owned workspace；只修adapter在size0用默认空参数。首次HYLE Copy prepared
不代签206项，原FAIL/三次build预算保留，不启动第四次C1 build；C2冻结完整slot/输出批次后
统一编译/验收此必要接线及全部请求。C1明确交接此未验入口，不假签C1运行资格完成。
