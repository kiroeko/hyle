# 多生产者 CPU 性能仪器拓扑

<!-- TOOL TASK: ACTIVE -->

## Finding

2026-08-28 的 obs `multi-producer-fanout` 需要 8 个真正并行、同 efficiency class、不同物理核心的
worker。`closure-perf-v1` 当前只登记 `single-logical-cpu` 和 `cross-core-pair`；把普通 obs
nanobench 放进单核 process mask 会产生漂亮但无效的“8 producer”数字。

有界模块 harness 已证明 W1-WIN 能把 8 个 worker 固定到逻辑 CPU
`0,2,4,6,8,10,12,14`，并验证它们来自不同 P-core、callback count 精确、1-sink 绝对预算 PASS。
但该外部 sampler 没有统一仪器的 A/A、5%/10% positive control、timer-floor、正式 n=15→31
升级和 raw deep replay，因此 baseline NI 只能写 `INCONCLUSIVE`。

## Required tool scope

- 在 machine/instrument contract 中声明通用 N-worker topology；CPU 集必须从 host probe 按 processor
  group、efficiency class、physical core 与 SMT sibling 机械选择，不能在模块脚本硬编码后冒充通用；
- parent process mask、每 worker singleton mask、priority、实际 observed CPU 与 callback/work count
  都进入可深重放 telemetry；
- structured concurrent rows进入现有 advantage-ratio、双顺序、A/A、5%/10% delta、timer-floor 和
  n=15→31 停止规则；`UNRESOLVED` 不得压成 `FAIL`，也不得靠追加样本跑到通过；
- 工具保持模块无关。obs 只提供真实 workload executable/mapping，不修改中央统计语义。

## Non-goals

- 不改变 5% margin、置信度、W1 registry 或现有单核/双核结果；
- 不为 obs 改产品 API、record layout、quiescent unregister、sink callback 或同步语义；
- 不把线程创建/barrier、分配或 benchmark 输出计入产品热路，也不新增第五种 workload 姿势；
- 不产生 Formal、多平台池化或完整 logging peer 的 SOTA 声明。

## Resume condition

只有中央工具在 fresh isolated W1-WIN build 上通过自身 tooling tests，并能对冻结的
8-producer × 1/4/16-sink workload 生成可深重放的 `closure-perf-v1` verdict，obs-v1 才可恢复其
performance gate。未满足前，不在 obs 模块内继续发明 placement、padding、order 或 sample 实验。

## 2026-08-29 implementation record

- `machine.py` 已增加通用 `n-worker-physical-cores` policy：从 live Windows CPU Set 按 group、
  efficiency class 与 physical core 选择一条 SMT sibling，并保留旧 `cross-core-pair` 的原选择语义；
- runner 已增加 `structured-paired-process-v1` adapter。candidate/peer 使用不同 product executable，
  但共享双顺序、A/A、5%/10% delta、batch-matched timer floor、15→31 停止规则和 Observation 统计；
- structured row 强制校验 worker/work/callback 守恒、requested/observed CPU、singleton masks 与稳定
  row set；raw verifier 从原始进程日志独立重建 topology、observations、controls 与停止决定；
- W1-WIN dirty-source `probe` 已实际配置、编译当前与 `f305b9d3` baseline，运行冻结的
  8-producer × 1/4/16-sink 双进程 mapping，并由 `hyle-perf verify` 深重放通过。probe 只证明 plumbing，
  按协议仍是 `DIAGNOSTIC / INCONCLUSIVE`；tooling tests 与 clean/fresh n=15→31 qualification 尚待完成。
