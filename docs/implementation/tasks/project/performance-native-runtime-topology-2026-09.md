# 原生 runtime 辅助线程的有限采集拓扑

## SCOPED / AUDITED

- owner：公共性能工具；baseline `c7397da4`；独立于 R4 产品实现，六个 CUDA owner 仍 BLOCKED。
- trigger：[独立 CUDA 现场](../../../../evidence/correctness/cuda-r4-context-causality/README.md) 在整个
  进程仅能使用 CPU2 时出现高优先级 caller 轮询、SDK helpers 无 CPU 时间，两种调度干预均恢复。
- 当前工具缺口：单核会限制 opaque SDK 辅助线程；现有 cross-core/N-worker 合同则要求应用逐个
  pin 并回报自有 worker。CUDA 的内部线程数量、创建时刻和角色由 SDK 拥有，不能把可用 CPU 集
  冒充逐线程 placement，也不为仪器接管私有 SDK 线程。
- consumer：固定 caller 执行原生 GPU API，SDK 在同进程内部辅助工作；两侧仍保有相同 native
  资源、完成边界与 CPU 资源预算。GPU 执行不是增加 CPU 模型执行器。
- 范围：Windows opt-in `native-runtime-pair`。复用既有 `_windows_cross_core_pair` 从实时 CPU Set
  选择 caller 与另一颗同组、同最高效率级、不同物理 P 核；process mask 只含这两颗 CPU，caller
  singleton pin。其余线程由 native runtime 管理，明确不声明 worker 数或 singleton pin。
- 非目标：产品、驱动设置、priority 修复、统计器、样本/门槛、全局机器 profile、其他 OS 拓扑、
  新并发执行器或重写仪器。无需改变公共 C++ affinity helper。历史拓扑字段与行为不变。
- tier：P0；预算为一次定向实现与必要回归，最多 45 分钟。若必须改统计模型、公共证据 schema
  或旧拓扑语义则停止。该工具完成不签 CUDA P1，模块的进度/A/A 接线另定界。

## 有限 FIX

| ID | 允许修改 | 验收 |
|---|---|---|
| N1 | machine.py 新 opt-in 拓扑复用现有双 P 核选择；保留真实 CPU Set/core/mask、caller、2 核预算和 runtime-managed 声明 | 缺核、跨类/SMT/非 Windows 拒绝；不产生虚假的 worker 数和 worker pin；旧政策字节等价 |
| N2 | runner 与 raw replay 接受该拓扑并核对 caller、parent mask、priority、affinity_ok | live 收集及 raw replay 均拒绝遗漏/篡改；control 使用既有 caller 与 parent telemetry |
| N3 | 机器/仪器文档说明精确资源与局限 | 不把“允许 helper 运行”写成已保证每线程独占核或已证明性能 |

预声明验证沿 selector：perf_machine、hyle_perf_runner、hyle_perf_raw_replay、evidence/build replay
相关 tooling suites；新用例覆盖选择/拒绝、旧合同稳定、环境传递与 telemetry 篡改。docs/closure
及 diff check 必跑。没有产品修改，本工具轮不重建、不运行 P1。

## 结果：工具任务 CLOSED

N1–N3 已完成。`native-runtime-pair` 复用既有双 P 核选择，记录 2 核 process budget 与固定 caller，
不产生 worker count/pin 字段；contract、live telemetry 与 raw replay 都支持同一声明。
新增回归证明：缺少另一物理 P 核时拒绝借用 SMT；非 Windows 拒绝；伪造 worker count 拒绝；
product/control 的 caller、mask、priority 或 affinity_ok 被删改时拒绝。既有拓扑回归全部保留。

selector 选中 9 个 tooling suites 与 docs/closure，11 门全部 PASS，其余 108 项有 skipped/reused
原因。[完整 selector receipt](../../../../evidence/correctness/performance-native-runtime-topology-2026-09/validation.zip)。
此轮实际没有产品构建或 P1。CUDA 模块尚须冻结接线、完成进度/A/A 后才可恢复正式资格。
