# R3 Graph：独立 caller 的验收环境纠正

<!-- PROJECT UNIT REVIEW: BLOCKED -->

## Closure card

- owner：graph_route_cpu、graph_provider_onednn；baseline `a19977f7`，产品投影固定在 `d9712e68`。
- 前轮：[有限纠正卡](cpu-v1-candidate-corrections.md)，实际 workload A/A 超线后 BLOCKED。
- 恢复事实：原配置 caller=2/workers=0,2；16 次有界诊断证明共核带来长延迟；用户随后明确
  要求检查并使用性能模式，追加的 8 次 low-latency 诊断确认共核中位约 70.274ms、独立 caller
  约 30.780us。独立 caller 仍有一个同实现 ratio=1.10195，不能签 A/A PASS。
- 本轮明确决策：以用户要求的低延迟模式及已经复现的共核资源冲突为依据，将预声明资源合同
  从两物理核改为外部 caller + 两个实际 worker 的三个物理 P 核。双方对称计费。建立 v2 合同，
  不把它当原 v1 的随机重跑或 PASS；旧停止记录保留。新合同仍须先通过原 5% factor-median /
  10% gross A/A，才可开始产品计时。SCOPED 只恢复这次有限环境施工，不表示旧阻塞门已通过。
- 工具前置：[独立 caller 工具任务](../project/performance-external-caller-topology-2026-09.md)
  已完成，11 项工具/docs/closure 检查 PASS；新拓扑由统一机器选择器、launcher 和 raw replay 校验。
- 范围：benchmark 的既有 Sched idle policy、模块 A/A validator 与 v2 合同；不改产品库。
- 非目标：Sched 算法/等待协议、Compute/Graph 算法、Provider 规则/阈值、API、产品交付形状、
  统计规则、placement、8ms epoch、fixture 顺序、输入/输出、完成边界或算子目录。
- consumer/语义：沿父 R3 卡的完整 gated FFN，default Graph / direct fused / natural sequence /
  explicit Native；12 个原 workload，含 serial 与真实 two-worker；同 executor、同输入和终结点。
- P0：热工作不变；cold setup 选择 kLowLatency，caller 不参与 bulk 计算。总资源三个 P 核，
  two-worker 阶段两 worker 常驻；serial 阶段仍只有 caller 执行，两个 worker 尚未创建。
- P1：W1-WIN engineering；两个 placement × 执行/分配正反；原 5% 工程门、n=15→31。
  预先保留一次 16-process workload A/A；仅其通过后启动一次新合同完整采样，无产品性能纠正额度。
- 预算/停止：源码审计 20 分钟，定向接线 40 分钟；A/A 或适用资格再失败就 BLOCKED，不再
  改 idle/CPU/时长/形状重试，也不据未资格化点值重开 Sched。产品采样达 n=31 后必须裁决。
- 适用验证：selector、合同 validate、validator 的 cold policy/telemetry 检查、两个 benchmark
  target 构建、上述一次 A/A/P1、docs/closure/diff；产品文件未变，复用现有正确性、交付与 sanitizer。

## AUDITED：固定 E1～E3

| ID | 缺口 | 允许纠正 | 验收 |
|---|---|---|---|
| E1 | benchmark 复制 correctness sample 的 PowerSaving 且允许 caller 共核 | cold setup 固定 kLowLatency；拒绝 caller 与 worker 逻辑核重叠；上报实际模式 | W1 机器合同排除整个物理核/SMT；运行 telemetry 与两 worker pin 一致 |
| E2 | A/A validator 硬编码 caller2/workers0,2/mask0x5 | 从统一 W1 profile 与实时 probe 取得新拓扑；沿既有 environment/launcher 注入，完整核对结果 | A/A 与产品采用相同 policy/资源，不允许 CLI 任意覆盖 CPU；原停止线不变 |
| E3 | v1 的资源/idle 合同已不适用 | 新 graph-cpu-route-p1-v2，仅明确三核资源与低延迟；原样保留 workload/完成/统计 | validate；旧 v1 和失败 evidence 不覆盖 |

七问的产品答案继续沿父 R3 卡；本轮无第三方升级、API、布局、分配或新依赖。Sched 完成通知
末尾忙等仍只是具体嫌疑，尚无可复现的独立因果证明；不将配置纠正包装为 Sched 缺陷修复。

源码审计：`region_p1_bench.cc` 的 main 在启动 executor 前已经 pin caller；executor worker hook
逐线程验证 SetThreadAffinityMask 与当前 CPU。`ParallelExecutorRef` 的 admission/同步完成不变；
修改只在 setup。模块 validator 可直接复用 `probe_machine`、`expected_child_policy` 和 runner
的 `_child_environment` / `_apply_sampling_policy`，无需手写另一套 mask 或线程环境。v2 只复制
v1 并更新实验 ID、资源/模式说明、topology 及 validator argv，不改十二项数值/计费合同。
有限 FIX 到此冻结；没有发现允许产品算法或 Sched 改动的新事实。

## 不可变候选

benchmark/validator 固定在 `2b613413cd30af845f0b66911e75a2fb455acd2a`。相对 d9712e68，
include/src/samples/cmake/CMakeLists.txt 全部无差异，产品投影继续复用其普通验收。v2 合同
validate、validator Python 编译及记录 host probe 上的 cold policy/environment 接线检查通过；
selector 仅选 docs/closure，两门 PASS，其余 98 门 skipped，未触发产品重建或采样。

两个 benchmark target 构建和一次预声明 A/A/P1 仍待执行；CANDIDATE 不是资格 PASS。
Windows 三交付完成后按串行队列运行，使用全新 build/evidence 目录，不覆盖 v1 失败包。

## Terminal：资格通过，工程性能仍 BLOCKED

候选 `2b613413` 的两个 benchmark 构建通过；实际 workload A/A 16 进程 / 192 ratios
全部通过，范围 0.915824–1.065435。统一仪器的五个 controls、build/artifact 与完整原始数据
`verify` 均 PASS，engineering 资格为 QUALIFIED。完整 n=15→31 已执行，原始两阶段均保留。

12 个 workload 中 11 个非劣门 PASS；`two-worker-fused-boundary-17x129x257` 两个 primary
stratum 的 98.9326% CI 下界为 0.951462、0.950646，低于既定 1/1.05 = 0.952381。
此项为 UNRESOLVED，聚合 performance_gate 为 INCONCLUSIVE；不是 material FAIL，也不是 PASS。
八个分层 median ratio 为 0.970361–0.982654，即 Graph 相对同实现 direct fused 耗时多约
1.8%–3.1%。其余串行固定小幅 loss 原样登记，不以非劣门 PASS 改名为 parity。

[完整结果与证据](../../../../evidence/performance/graph-cpu-route-p1-v2-2b613413-w1-win/README.md)
保留全部 12 项方向、两阶段 raw、manifest、controls、driver 和 verify receipt。
本卡 n=31 预算已耗尽，停止 Graph 采样；graph_route_cpu / graph_provider_onednn 以及承担
该 replay 边界的 graph owner 进入 BLOCKED，已有正确性/三交付不失效。Direct 独立验收继续。

- blocker：合格同合同 Graph/direct fused 比较尚未证明两 worker 包装税落在工程预算内。
- blocker_owner：Graph CPU 重复执行与 region 边界；尚无 Sched 产品缺陷的独立因果证据。
- resume_condition：另立有限成本审计，证明具体多余工作并固定合法纠正；依法重新 SCOPED/AUDITED，
  产品/API 固定后才决定适用候选验证。不能通过追加原候选样本或更换 affinity/idle/阈值恢复。

用户在本次裁决后明确要求专门调研该阻塞项。此授权开启下一张有界源码成本卡，不覆盖本卡停止事实。
