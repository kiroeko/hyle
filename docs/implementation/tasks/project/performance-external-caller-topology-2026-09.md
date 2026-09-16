# 外部 caller 与固定 worker 的独立物理核拓扑

## Task card / SCOPED

- owner：公共性能工具；独立于 R3 模块的产品纠正，不重开 Sched。
- baseline：`0a183b21`。
- trigger：R3 完整 Graph/Sched consumer 的外部 caller 不执行 bulk 工作，却与一个 worker
  共用 P 核。已保存的 16 进程诊断及用户要求的 8 进程 low-latency 诊断显示，共核会产生
  毫秒级延迟；caller 独立后长延迟消失，但 A/A 尚未全部通过，不能据此签性能 PASS。
- 工具缺口：现有 `n-worker-physical-cores` 必须把 caller 包含在 N 个 worker 中。把 N 写成 3
  却只启动 2 个 worker 会伪报资源/线程数；模块不能靠手工 mask 绕过机器合同和 raw replay。
- 正常 consumer：外部同步 caller + N 个执行线程；现有 caller 参与工作的 N-worker 合同不变。
- 范围：Windows 增加显式 opt-in `caller-plus-n-worker-physical-cores`，N 仍是实际 worker 数。
  caller 使用机器登记核；worker 从实时 CPU Set 的同组、同效率级、其他物理核确定性选择。
  进程 mask 包含全部 N+1 个核，worker 不使用 caller 的 SMT sibling。资源记录明确保留 N+1。
- 非目标：Sched/Compute/Graph 产品、公共统计估计器、门槛、样本数、placement 策略、历史证据、
  多 processor group 或其他 OS 的 affinity 支持。工具通过不自动解除 Graph BLOCKED。
- tier：P0；这是 placement 合同支持，不对公共工具自身计时。
- 预算/停止：一次定向实现与回归，最多 45 分钟；若必须改变统计模型、证据 schema 或旧拓扑
  行为，停止并记录具体 blocker。模块接线/新不可变 benchmark 的 A/A 与 P1 另行定界。

## AUDITED：有限 FIX

| ID | 源码缺口 | 允许改动 | 验收 |
|---|---|---|---|
| T1 | machine.py 的 N-worker selector 保留 anchor 为 worker | opt-in 排除 caller 整个物理核；mask、CPU Set、物理核清单同时保留 caller 与 N worker | 实际 worker 数、N+1 资源账、同类/不同核、SMT 排除、核数不足拒绝；旧选择结果不变 |
| T2 | runner/raw replay 只识别原拓扑 | 新名称的合同校验、环境、product/control telemetry 与 replay 接线；新拓扑必须核对 caller_cpu | 遗漏或改变 caller/worker/mask 的日志拒绝；原拓扑回归仍通过 |
| T3 | 使用文档无法表达外部 caller 资源 | 补充机器/仪器文档的独立 caller 用法及限制 | 文档与实现一致，不宣称 P1 通过 |

审计结论：`bench/common/perf_process.h` 已能按环境 pin caller；无需改公共 C++ benchmark helper。
环境传递和 evidence 的 expected-child-policy 已复用 machine helper；不增加第二套 CPU 选择器。
模块将使用既有 `kLowLatency`，这属于下一张模块验收卡，不属于本工具 FIX。

## 预声明验证

先 change-impact selector，运行 `perf_machine_test.py`、`hyle_perf_runner_test.py`、
`hyle_perf_raw_replay_test.py`、`hyle_perf_evidence_runner_test.py` 和 `hyle_perf_build_replay_test.py`；
保留其他门 skipped reasons。机器选择、缺核拒绝、旧合同、caller telemetry 篡改及 raw replay
是必要回归。另运行 docs/closure checker 与 diff check。工具修改本身不启动产品构建或 P1。
