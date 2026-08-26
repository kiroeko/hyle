# 模块收口与统一性能仪器 v1 实施记录

> **候选验收（2026-08-26）。** 本页记录项目级工具施工事实，不是产品模块状态。engineering
> 仪器已完成本轮校准；Formal 长期保留层仍是独立 blocker。模块当前状态只读
> [`module-status.json`](../../module-status.json)。规则见
> [模块重构与收口协议](../../../developer/review/module-closure-methodology.md)，性能协议见
> [统一性能仪器 v1](../../../developer/performance/instrument-v1.md)，机器角色见
> [性能机器合同](../../../developer/performance/machines.md)。

## 为什么启动

`container_intrusive` 的产品实现不到千行，但旧审查把“产品形状冻结”“本地仪器校准”“三平台
Formal 证书”和模块当前状态写在不同页面，出现 `AUDITED`、`PRODUCT LOCKED`、`CANDIDATE` 等互相
冲突的视觉结论。专用 runner 又把机器探测、统计、controls、模块 codegen 与证据归档写在一起，
导致一个小模块也能无限增加 benchmark 姿势，后续模块无法机械复用。

本轮基线为 `79c644c2b7cef51786862a5bf0dd6b565305a308`。目标是先冻结收口状态机、机器合同和公共性能
仪器，再让 `container_intrusive` 与 `error` 成为两个不同消费者；本轮不借工具施工修改它们的公开
API 或产品算法。

## 已冻结的决策

- 产品状态固定为 `QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> VERIFIED -> CLOSED`，另有
  `BLOCKED` 与 `REOPENED`；只有中央 JSON 保存当前状态。
- 性能分为 P0/P1/P2。P1/P2 工程性能 `PASS` 是 `CLOSED` 硬门；P1 默认 material margin 为 5%。
- Formal 发布声明是独立轴，不把跨平台发布证书重复变成每次模块重构的默认门。
- 每个 workload 分开记录 comparability、qualification、exact performance；non-inferiority 单列。
- `advantage_ratio > 1` 永远有利于 HYLE。旧 `RIVAL/BASELINE` raw 必须由 adapter 归一化。
- 极短 CPU 公共 controls 为 A/A 双顺序、near/large known-delta 和 timer/empty floor；placement、
  DCE、concurrency 等按冻结风险选用，不机械做全排列。
- `probe` 固定每个 stratum/order n=1 且永远 diagnostic；P1 qualification 固定先跑 n=15。只有
  required controls 全通过、至少一个 NI gate 因功效 `UNRESOLVED` 且其余均 `PASS` 时，才按预声明
  完整重跑 n=31，不向 n=15 追加样本。
- timer floor 必须按 executable 声明的 `measurement_batch` 匹配；pair 较快一侧至少是两 order/
  两 floor row 中最保守地板的 20 倍。该阈值是 harness 资格门，不是可按模块调节的 margin。
- child 启动前清除 ambient benchmark、sanitizer/profiler、loader、allocator 和 affinity tuning，固定
  常见数学运行时为单线程，只恢复合同白名单环境；保留环境的事实由策略记录而非伪装成 sandbox。
- 每次 child 由 runner 注入唯一 128-bit nonce 并由 telemetry 回报；raw index 拒绝 nonce/hash 重复，
  所以复制 n=1 log 不能冒充多个独立进程。
- nanobench runner v1 只验静态链接。它会从实际 CMake cache/toolchain/artifact 复核配置；这不是
  HYLE 产品只支持 static，而是 shared dependency closure 尚未进入该 adapter。
- Formal 的 expected host/toolchain 必须在运行前登记，不能用本次 observed 值自认证；缺 exact
  compiler/linker/hash/SDK/stdlib 或稳定主机身份即 fail closed。
- 模块只能新增 contract、row mapping 和确有必要的 artifact validator；不能复制 runner 或改公共
  estimator、margin、verdict 与停止规则。
- 普通约千行以内模块的审查/取证默认两小时；一次反序/布局复现、一次机制检查、一个产品假设与
  重跑之后必须裁决，不能发明第五种 benchmark 姿势。

## 当前施工清单

| 项 | 当前事实 | 完成条件 |
|---|---|---|
| 状态合同 | `module-status.json`、人读投影、checker 和 tooling test 已加入 CTest；本地 checker 通过 | 整轮提交前复跑测试并确认投影未 stale |
| 机器合同 | W1-WIN 为 `REGISTERED_MATCH`；WSL engineering probe 通过且保持 unbound；M4 的 interactive QoS 应用/恢复实机通过且保持 unregistered | 当前 engineering 边界完成；WSL 不冒充 native Linux，未登记 M1 不签 Formal |
| 公共数据核心 | contract/model/stats/controls/evidence/nanobench adapter、schema、raw/build/host replay tests 已实现 | 本轮测试全绿；下一个普通 CPU consumer 不修改公共统计/controls |
| 公共 runner | `validate/probe/qualify/verify/certify`、互斥、环境清洗、fresh build、child nonce、raw→structured、n=15 stop、build/artifact/host 深重放已接通 | 当前 engineering 候选完成；最终红队无 HIGH |
| 公共 controls | A/A、5%、10%、1..4096 batch-matched floor 和纯函数重放已实现；synthetic 与 W1 n=15 实机校准通过 | 短 workload 不靠放宽 20x 过门；跨平台适用路径复验 |
| 首批消费者 | container v4 与 error v3 的真实 n=1 probe 均产生 nonce/raw/build/host 可深重放 manifest | adapter 验收完成；各模块的 P1 qualification 回到所属模块，不阻塞公共仪器 |
| Formal 边界 | 可精确核验 8 项 toolchain identity、登记机器与独立物理机；当前 bundle 仍引用 live artifact | [retention/signing 任务](perf-evidence-retention-v1.md)、M1 身份、exact Formal toolchain 和独立平台 manifests 完成前保持 PENDING |
| 文档 | 方法、仪器、机器、CLI 和本任务页已按当前实现校准 | `check-module-closure.py`、`check-docs.py`、`git diff --check` 通过 |

## Error plumbing pilot 暴露的仪器事实

W1 上一次早期 error plumbing pilot 在 `probe` 尚临时取 n=3 时跑通了四组 product strata、两种
order、公共 control 解析和证据中间文件；它不是 current qualification，也没有产生可签 manifest。
当前实现已经把 probe 收紧为 n=1，所以这批 n=3 数据只保留为调查事实：

- A/A、5% 和 10% control 因只有 3 个进程、没有最低 n=15 区间而 `FAIL`，不能从它们判断仪器校准；
- batch=1 的 conservative floor 约为 0.37 ns/op；`status-ok` 与 `status-ok-copy` 的较快一侧仅约
  1.50x floor，runtime-code/cancelled factory 约 2.03x，native query 约 3.05x，四帧 OK/result-code
  propagation 约 7.01x/9.80x；
- 较长的 message/context/format/update 路径有一部分超过 20x，但这不能替未过门的短路径签结论。

因此 pilot 的结论不是“error 输了”，也不是“20x 太严格”，而是这些单操作 nanobench rows 对当前
harness 而言是 **harness-dominated**。在门通过前，HYLE/Abseil 比率无资格影响 error 的 API、对象
布局或实现。下一步只能选择真实消费成立的自然 batch/多帧 replay、换用 counter/aggregate 仪器，
或把无法可靠分辨的 microbench 留作 diagnostic/P0 结构检查；禁止仅提高
`measurement_batch`、降低 20x、放宽 A/A/known-delta，或为过 benchmark 改产品形状。

加入最终 child nonce/raw/build replay 后，error v3 在 W1 用当前四个产品 executable 与公共 control
重新执行 n=1；`probe` 与独立 `verify` 都成功，资格仍按设计为 `DIAGNOSTIC`、performance gate 为
`FAIL`。这证明第二种消费者的 plumbing，不推翻上述 harness-dominated 结论，也不关闭 error 的后续
P1 性能任务。

## W1 公共 controls 实机校准

2026-08-26 在登记的 W1-WIN 上完成每个 order 15 个独立进程的公共 control 校准；运行期间主机事实
稳定，Windows child canary 验证继承的 logical CPU 2 / affinity `0x4` 与 above-normal priority。
分布无关 99.26% 区间结果如下；这次校准不运行任何产品 workload，也不签模块性能结论：

| control | candidate-first median / CI | peer-first median / CI | gate |
|---|---:|---:|---|
| A/A | 1.0010 / [0.9974, 1.0070] | 0.9992 / [0.9967, 1.0040] | PASS |
| artificial 5% | 1.0499 / [1.0458, 1.0526] | 1.0502 / [1.0462, 1.0540] | PASS |
| artificial 10% | 1.1000 / [1.0952, 1.1024] | 1.1001 / [1.0970, 1.1011] | PASS |

这是加入 per-child nonce 并重建公共 control binary 后的最终重采；原始 30 个进程日志位于本地
`.scratch/perf-controls-W1-WIN-n15-v2`，不提交仓库。5% 与 10% 在两个 order 都可辨认且单调。
另一个真实 consumer pilot 已证明 batch=4096 的
timer-floor/placement 均通过；首次运行还因 benchmark telemetry 把合同的 `candidate-first` 写成
`candidate-allocation-first` 而 fail closed，统一字段后重跑才生成可验证的 diagnostic manifest。

最终 container v4 又在 nonce/build/raw 深重放接通后重新执行；`probe` 与独立 `verify` 均成功，且
保持 `DIAGNOSTIC`。红队把一份真实 n=1 日志复制成 15 份时由重复 child nonce 机械拒绝；篡改实际
CMake cache、compiler hash、artifact 闭包、自报 builtin validator，以及用畸形 runner contract
进入 `certify` 均 fail closed。本地 63 项性能 tooling tests 加 machine/closure tests 全绿；在本文
声明的“unsigned bundle 只保证机械一致性”威胁边界内，最终红队无 HIGH。

## v1 验收与停止线

1. schema 缺字段、未知版本、重复/缺失 row、NaN 和证据 identity 漂移均 fail closed；
2. synthetic 与实机 controls 能机械重放 A/A、5%/10% 已知差、batch-matched floor 与顺序偏置；
3. raw index 精确覆盖所有 stage/run，child nonce 不重复；原始 log 可重建 observations、controls、
   summary 与 n=15→n=31 停止决策；
4. W1-WIN 与 W1-WSL 共享物理互斥键，WSL 不签独立硬件；M1-MAC 未登记稳定身份前 Formal fail
   closed，登记后仍明确 QoS 而不伪造 affinity；
5. `container_intrusive` 与 `error` 只通过 contract/plugin 接入；下一普通 CPU 模块不修改统计核心；
6. `probe` 固定 n=1 保持有界；单平台 `qualify-platform` 的耗时按合同 workload/strata 与资源预算记录，
   不删减 required evidence 来满足统一分钟数；
7. 完成 Windows、WSL、macOS 适用 probe/correctness 后，记录四项 LOC、独立红队结论和提交。

数据库、dashboard、远程调度、自动 margin/MDE、顺序贝叶斯、通用反汇编 DSL、全库 SOTA 分数和
一次迁移全部旧 benchmark 均不是 v1 目标。旧 module runners 只保留为 diagnostics 或可提取的
Formal/plugin 输入，不能成为下一模块的复制模板。

## 最终验证矩阵

| 范围 | 命令/证据 | 结果 |
|---|---|---|
| Python 统计、合同、runner、raw/build replay | `python -m unittest discover -s tests/tooling -p 'hyle_perf_*test.py'` | 63/63 PASS |
| CMake 注册后的公共 tooling | `ctest -R '^hyle_(perf|module_closure)'` | 10/10 PASS |
| 文档与状态投影 | `check-docs.py`、`check-module-closure.py`、`git diff --check` | 258 Markdown / 709 links，PASS |
| Windows C++ adapter | error 四个变体、container primary/placement/instrument、公共 control | Release 编译 PASS |
| 真实 bundle replay | container v4、error v3 的 `hyle-perf.py verify` | 两项 PASS，均保持 `DIAGNOSTIC` |
| W1 controls | 双顺序各 15 个独立进程 | A/A、5%、10% 全 PASS，host stable |
| WSL | tooling machine test + engineering probe | PASS；明确 WSL2/unbound |
| macOS M4 | tooling machine test + interactive QoS probe/恢复 probe | 应用时 PASS；退出恢复 default；无 QoS 参数按设计 FAIL |
| 独立红队 | 日志复制、build/cache/artifact、validator、certify 合同篡改 | 声明的机械一致性边界内无 HIGH |

候选提交使用主题 `build(perf): lock module closure instrument v1 [skip ci]`；本页不伪造尚未生成的
commit hash。staging 快照的四项 LOC 为：

1. 全库代码：**315,837**；
2. 自实现包含测试：**142,952**；
3. 自实现不包含测试：**96,244**；
4. 当前 workstream 实现：**7,377**（14 个 Python + 1 个 C++ + 1 个 C/C++ header；范围为
   `scripts/hyle_perf`、两个公共 CLI/checker、`bench/common/perf_process.h` 与公共 control bench）。

前三项使用 `loc-report.py` 的 canonical/self-check 口径。第四项不是 HYLE 产品模块，不能伪装成
`include/hyle + src` 的 native module boundary，因此明确用同一 staging 快照的 all-language `cloc`
口径；contracts、tests、module adapter 和文档未计入该实现数。

## 本工作流交付时的未完成事项

> 下列是 2026-08-26 工具任务结束时的历史快照，不声明产品模块当前状态；后续变化只见中央
> registry 和各模块当前 task/evidence。

- 两个首批消费者的 current diagnostic manifests 已深重放；它们尚无 engineering qualified
  manifests，这是各产品模块的性能 blocker，不再反向阻塞公共仪器；
- error 的若干短操作仍被 20x timer-floor 门拒绝，必须修正测量合同或保持 diagnostic；
- live artifact bundle 尚无不可变 retention/signing；pilot profiles 尚无 exact `formal_toolchain`，
  M1 也尚无稳定机器身份，Formal 当前按设计保持 PENDING；
- `container_intrusive` 当时仍为 `BLOCKED`，不能因产品形状已冻结提前改为 `VERIFIED/CLOSED`；
- push 记录将在候选提交实际推送后由 Git 历史提供；本文不制造循环自引用的 commit hash。
