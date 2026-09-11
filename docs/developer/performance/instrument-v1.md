# HYLE 统一性能仪器协议 v1

本协议是模块工程性能收口的唯一统计与证据入口。模块可以新增 workload contract、benchmark
adapter 和窄的结构 validator，但不得复制 runner，亦不得改变统计、margin、停止规则或 verdict
语义。发布级证据的附加要求见
[性能证据标准](benchmark-evidence-standard.md)。机器角色见
[性能机器合同](machines.md)。

## 1. 三个相互独立的结论

每项 workload 分别记录：

1. comparability：`COMPARABLE | NOT_COMPARABLE`；
2. qualification：`QUALIFIED | DIAGNOSTIC`；
3. material-band performance：`WIN | PARITY | LOSE | INCONCLUSIVE`。

同合同的单边 non-inferiority 门另记为
`PASS | FAIL | UNRESOLVED | N/A`。`NOT_COMPARABLE` 不是统计结果；缺失资格也不能伪装成
`PARITY`。

内部统一使用 `advantage_ratio`，且 **大于 1 永远有利于 HYLE candidate**：

- higher-is-better：`candidate / peer`；
- lower-is-better：`peer / candidate`。

给定预声明 margin `m`，等价带固定为 `[1/(1+m), 1+m]`：

- CI 完全高于上界：`WIN`；
- CI 完全位于带内：`PARITY`；
- CI 完全低于下界：`LOSE`；
- 其他：`INCONCLUSIVE`。

required strata 逐项裁决。exact verdict 只有在各 stratum 一致时合成；`WIN + PARITY` 所表达的
“没有输”由 non-inferiority gate 表达，不强行合成成某个 exact verdict。

`exact_verdict` 是 v1 已冻结的字段名，但它实际回答“置信区间相对 practical band 在哪里”，不能作为
ratio `1.0` 的方向标签。全库竞争性 scorecard 必须另从逐 stratum CI 导出 `directional_result`：全部
适用 CI 下界高于 1 为 `WIN`，全部上界低于 1 为 `LOSS`，其余为 `INCONCLUSIVE`；顺序或 placement
方向不一致同样为 `INCONCLUSIVE`。稳定 `LOSS` 不因仍在 5% band 内改名为 `PARITY`。旧 summary 与
manifest 保持原字节，不回写这一派生字段。

## 2. 产品阈值与仪器分辨率

每个合同在运行前声明 material margin `M`。仪器用 A/A、known-delta 和实际样本证明当前
machine/harness 的可检测差异 `D`；产品行动阈值是 `max(M, D)`。P1 默认 `M=5%`，但真实消费者
可以预声明其他值。P2 的 1%--3% 必须由端到端 profile/KPI 证明，不能由 microbench 自封。

CPU v1 的 timer-floor `20x` 门不是另一个 material margin，也不是“误差已被精确证明为 5%”。它是
更早的一道 harness 资格筛选：被测 pair 的较快一侧仍须至少达到同一 measurement batch 下保守空
harness floor 的 20 倍。未过该门时，仪器明确回答“当前计费边界太接近地板”，而不是回答谁快。

- 测不到低于 `M` 的差异不是工程门失败；单边最坏界仍在 `M` 内即可通过 non-inferiority，但方向仍
  必须独立报告。
- 允许超过 `M` 且 power 不足时才是 `UNRESOLVED`。
- 稳定 `LOSE` 只有同时具备机制证据和消费影响，才能改变产品 API、layout 或常态 metadata。

## 3. 机器合同

正式运行先选择 machine profile。profile 定义物理主机、OS/toolchain、filesystem、电源、affinity
或 QoS、允许的运行模式与互斥键。运行时 probe 必须证明实际环境满足 profile；CLI 不能覆盖
profile 的安全条件。

Windows 与同机 WSL 使用同一物理互斥键，不能并行采样。WSL2 结果只能声明 WSL2 环境，不能
冒充独立硬件或 native-Linux Formal。Formal 还必须匹配 registry 预先登记的稳定机器摘要与
CPU/内存/主板/OS/BIOS facts，不能由本次 probe 自报 expected identity。macOS 无受支持的 hard
affinity，合同必须明确 QoS；当前 M1 尚未登记稳定身份，所以只允许显式 unbound engineering，
Formal fail closed。

CPU adapter 默认使用 machine profile 的单逻辑 CPU policy。真实并发 workload 可以在 runner
extension 中声明 `sampling_topology=cross-core-pair`：Windows launcher 从实时 CPU Set 中保留登记的
caller CPU，并确定下一颗同 efficiency class、不同 physical-core 的 worker CPU；进程 affinity 只开放
这两颗 CPU，benchmark 还必须把 caller/worker 角色分别 pin 并由 telemetry 回报 CPU id 与成功状态。
只给进程一个双核 mask 不算证明，因为 OS 仍可能把两个线程同时放到一个核。当前该拓扑只在 Windows
adapter 准入；其他 OS 会 fail closed，而不是伪造 hard affinity。旧合同省略该字段时仍保持原有单核
policy 和 contract/manifest replay 语义。

固定 worker 数的并发 replay 可以声明
`sampling_topology=n-worker-physical-cores` 与 `sampling_worker_count=N`。Windows backend 从同一次
host probe 中，以登记 caller 的 processor group 和最高 efficiency class 为边界，每个 physical core
只选择一个 logical sibling；选择结果、parent process mask、每个 worker 的 singleton mask、CPU Set id、
core id 与 priority 全部进入 child policy。workload 必须逐 worker 应用 singleton mask，并在每个进程
回报 requested/observed CPU、worker/work/callback 守恒；只继承宽 parent mask、硬编码一台机器的 CPU
编号或仅回报“affinity 调用成功”均 fail closed。该拓扑目前同样只在 Windows 准入，不改变历史
`cross-core-pair` 的 caller/next-core 选择语义。它可与 structured separate-process 或默认同进程
paired-row adapter 组合；拓扑描述 worker placement，measurement adapter 描述 candidate/peer 如何成对，
两者没有从属关系。benchmark 仍须自行核对并回报 worker 与工作量守恒。

## 4. Experiment contract v1

正式 experiment contract 至少冻结：

- stable experiment/module/claim/workload ID、证据 `tier` 与正交的
  `gate_role=product|diagnostic`；
- candidate target、delivery、peer revision/config；
- semantic contract 与 comparability；
- metric、unit、direction、输入分布和规模；
- setup、steady-state、teardown 的计费边界；
- candidate/peer row mapping；
- material/equivalence/non-inferiority margin；
- 显式 required strata，不由 runner 擅自做笛卡尔积；
- estimator、固定样本数和唯一允许的升级计划；
- required controls、artifact validators 和 host requirements；
- compiler、ISA、LTO、linkage 与资源预算。
- 对并发负载，冻结 `single-logical-cpu`、`cross-core-pair` 或带固定 worker count 的
  `n-worker-physical-cores` sampling topology，并由角色 telemetry 证明实际落点；不得用进程允许的
  CPU 集代替线程角色归属。

运行前生成 immutable experiment-lock hash。CLI 只允许提供 checkout/build/output/peer 路径和
machine profile，不能覆盖 pair、margin、规模或停止条件。

`tier` 说明测量形状，`gate_role` 说明它是否参加产品裁决，两者不能互相冒充。当前新运行要求每个
workload 显式声明 `gate_role`：`mechanism` 只能是 `diagnostic`；`consumer/replay + diagnostic`
可以保留归因、保真度或绝对成本结果而不承担产品门；只有 `consumer/replay + product` 才进入产品
non-inferiority 聚合，且当前 NI adapter 要求这类 product 行为 `COMPARABLE`；无同合同 peer 的绝对
consumer budget 必须使用以后具名的 adapter，不能把 `NOT_COMPARABLE` 行塞进当前产品聚合。product
行还必须冻结以下 structured declarations：具名 `consumer_key`、由仓库
相对路径与 SHA-256 绑定的 retained provenance、分别陈述 candidate/peer/equivalence 的
`ownership_topology`，以及含 numeric `frequency/value/budget`、`unit`、`denominator` 和非空
`derivation` 的 `material_contribution`。这些字段使审查项不能被省略，但不机器证明语义真实；冻结前的
独立审查和适用 validator 仍负责确认 consumer、双边 ownership 与 material 推导有效。
四项 product declaration 是一个原子记录，不能只填其中几项；manifest 同时保存合同中的仓库相对路径
和 retained file hash，deep verify 会拒绝路径身份或内容漂移。
P1 的 denominator/value 可以表达组件 replay fidelity 与预冻结 non-inferiority 预算；这不会把 P1
自动抬成端到端消费者 KPI。P2 仍须按性能证据标准绑定真实 consumer/trace KPI 与贡献预算。

旧 v1 contract 可以缺少这些可选字段，以保持既有 contract hash 和 manifest 的只读 `verify`。新
`probe/qualify-platform` 总会写入 `consumer-evidence-v1` policy 及逐项 eligible/excluded workload；
缺少显式角色或任一 product 行消费声明时，qualification 为 `DIAGNOSTIC`，聚合
`performance_gate=INCONCLUSIVE`，不能写成产品 `FAIL`。

### Consumer provenance 的非实质字节变化

整文件 SHA-256 是默认的 fail-closed 地板，但“文件任何字节变化”不等于“冻结 workload 的消费者
语义变化”。若一个大 consumer TU 只修改了与 replay 无关的注释或相邻 API，不能手改旧 manifest，
也不应把 verifier 的停止误写成产品性能失败。当前工程回放使用
`bench/performance-provenance-renewals.json` 的版本化精确迁移账本处理这种情况：

- contract ID + semantic SHA-256、workload ID、repository-relative path、旧/新 SHA-256 必须全部
  精确匹配；禁止 wildcard、branch 和“当前 HEAD”一类移动身份；
- 每条 `NON_MATERIAL` 边绑定一份有自身 SHA-256 的 review record，逐项核对 semantic contract、
  ownership topology、material contribution、charged phases、算法和 delivery；
- verifier 只沿唯一无环 hash 链到达当前文件，并报告 `REVIEWED_NON_MATERIAL`、账本 hash 和实际
  边；exact 文件报告 `EXACT`，没有 product provenance 报告 `NOT_APPLICABLE`；
- 不存在精确迁移、审查记录被改、或上述任一产品事实发生变化时仍 fail closed；真实 material
  consumer drift 必须冻结新 contract ID 并重新 qualification；
- renewal 只允许历史 **engineering replay**。runner 生成新证据仍要求 byte-exact，Formal 也不接受
  renewal，不能把人工 materiality review 冒充发布证书。

这是一条可审计的证据兼容链，不是源文件 hash 的豁免表。它只说明旧统计结果仍对应同一个冻结消费
命题；模块 candidate 之后另有公开头、算法或 delivery 变化时，仍按模块自己的 reopen trigger 裁决。
首次实现与 Time 精确迁移事实见
[`perf-provenance-renewal-v1`](../../implementation/tasks/performance/perf-provenance-renewal-v1.md)。

## 5. 通用核心与插件边界

统一核心负责：

- contract/schema/lock 校验；
- process 与显式 strata 调度；
- Observation 规范化；
- paired process median CI 与 SIMD clustered estimator；
- 四态、non-inferiority 和 strata 合成；
- source/peer/build/artifact/host 的前后稳定性；
- `observations.jsonl`、`summary.json`、`manifest.json`；
- 任一必需记录缺失时 fail closed。

measurement adapter 只把 nanobench/结构化 telemetry 转为 Observation。默认 adapter 从一个
nanobench 进程中的 candidate/peer rows 配对；`structured-paired-process-v1` 则按冻结顺序分别启动
candidate 与 peer product executable，并要求两侧使用同一 row set、worker 拓扑与工作量守恒后再
配对。两者共用 Observation、统计、controls、停止规则和 raw replay，模块不得复制 runner 或自行
解释区间。platform backend 只负责 machine probe、affinity/QoS 与 artifact 身份。模块 plugin 只能
声明 target、row mapping 和确实特有的 codegen、DSE、资源守恒 validator。

两种 adapter 的 A/A 拓扑不能混用。公共 `hyle_perf_control_bench` 是同进程 paired-row control；它足以
校验默认 nanobench adapter，但不模拟 structured adapter 的两个独立 product process。新的 structured
合同若执行先后/启动热状态可能达到 material margin，必须在完整 qualification 前另有同启动拓扑的
same-artifact A/A gate；缺少该 gate 时只能诊断。若真实消费者本来就在一个 source/static/shared 进程
中调用两条公开路线，应直接使用默认 paired-row adapter，不得为了隔离而人为引入跨进程偏差。

runner validator 是整个 artifact bundle 的全局完整性门，不存在“只影响某一 diagnostic row”的隐含
scope；所以它只能检查 build/artifact/合同整体。某个 mechanism 或 diagnostic replay 特有的风险应放在
该 workload 的 required control，或拆到独立 diagnostic contract，不能用一个可能失败的全局 command
validator 间接否决无关产品行。

## 6. CPU v1 controls 与按风险扩展

当前 CPU v1 的 nanobench 与 structured paired-process adapter 对每个 workload 强制要求公共 A/A、
5% delta、10% delta 和 timer-floor，并分别执行 candidate-first 与 peer-first。正式 qualification
每个 order 至少 15 个独立进程：

- A/A 使用同实现的等价两侧，固定 5% reciprocal band；两个 order 都必须裁决为 `PARITY`；
- 5%/10% positive control 的区间下界都必须大于 1，中位数须分别落在 2.5%--7.5% 与
  5%--15% 的预注册容忍带，且每个 order 的 10% 响应必须大于 5%；
- timer-floor 同时采集 batch `1..4096` 的二次幂档；每个产品 executable 必须声明自己的
  `measurement_batch`，评价时只取同 batch 数据，不能拿大 batch 的地板替单操作辩护。

timer-floor 对两个 order、floor candidate/peer 四组中位数取最大值作为 conservative floor；产品 pair
则取 candidate/peer 中较快一侧的中位成本。只有
`faster_side_ns_per_op / conservative_floor_ns_per_op >= 20` 才通过。20x 是固定的 v1 harness-dominance
警戒线；不能因某个模块的短操作未通过就调低，也不能只修改 `measurement_batch` 来“过门”。只有真实
消费者本来就以该 batch/replay 为一次自然任务时，才建立对应合同；否则应改用更合适的 replay、
counter/aggregate 仪器，或把该 microbench 保持为诊断。

以下 control 只在合同风险适用时加入：

- opaque callback/observer：phase floor；
- DCE 风险：实际对象 codegen gate；
- 极短指令：一种 code-placement sensitivity；
- caller-owned 且地址敏感：一种 data/allocation sensitivity；
- 并发：caller/worker role affinity、physical-core 区分、migration 与工作量守恒；
- allocator、IO、GPU：各 measurement family 的资源/同步 control。

不得为每个 workload 机械加入所有风险维度。新增通用 control 或改变上述固定阈值必须升级
instrument version；模块任务内不能移动冻结终点。`probe` 的公共 controls 也会执行，但一次样本不足
以形成区间，出现 control `FAIL` 不等于一次正式校准失败；它本来就只能帮助排查 plumbing、row mapping
和显然的地板问题。

## 7. 三种运行模式

| 模式 | 用途 | 资格 |
|---|---|---|
| `probe` | dirty/可复用 build；每个 required stratum/order **恰好 n=1**，检查 plumbing、映射与明显地板问题 | 永远 `DIAGNOSTIC` |
| `qualify-platform` | 模块 P1/P2 工程收口；clean、fresh isolated Release、全部 required controls/strata；默认 n=15 | 可产生工程 verdict |
| `certify` | 发布时只校验并合成同 contract hash/candidate 的平台 manifests | 协议上产生 Formal claim；当前受 retention blocker 阻止发布 |

`validate` 只冻结并报告 contract hash。`verify` 则严格重放一个已有 manifest：复核 contract lock 与
全部文件 hash；从 `raw-log-index` 按 stage/stratum/executable/order/run 精确闭包重新解析 nanobench
或结构化 product 原始日志、topology/accounting telemetry、observations 与 controls；从实际
CMakeCache、compiler/linker 和 product artifact
重算 build/artifact gates；并从 registry 与 host probe 重算机器门。每次 child 都有 runner 生成的唯一
`run_nonce`，索引拒绝重复 nonce 或完全相同的日志字节，因此复制一份 n=1 输出不能伪装成 n=15。一个
`DIAGNOSTIC` manifest 可以完整通过 `verify`；这只证明证据未漂移且裁决可复现，不会把它升级为性能
产品资格。CLI 因此同时输出机械 `qualification`、`product_gate_policy`、
`current_certification_policy_eligible` 和 `product_interpretation`；历史 manifest 即使机械重放为
`QUALIFIED`，后两项仍明确为 `false` / `INCONCLUSIVE`。

n=1 probe 不读取合同的 n=15 作为缩小版统计，也不会因结果好看升级资格。`qualify-platform` 固定先跑
完整 n=15；升级 n=31 只允许合同预声明，而且 comparable `gate_role=product` workload 的
non-inferiority gates 至少一个为统计功效 `UNRESOLVED`、或一个 required common control 只缺 n=15
precision，其余 product gates 与 required controls 均为 `PASS`。precision-only control 仅包括：（1）
A/A 没有方向性，只因 CI 穿过预冻结等价边界；（2）known-delta 的两个 order median 均在预冻结接受带、
5%→10% 响应单调、样本/区间完整，但至少一个 CI 下界尚未证明大于 1。median 越界、响应不单调、
directional A/A、timer/placement/delegated failure 或畸形数据仍是 hard stop；不能借 n=31 重试。
`mechanism` 和其他 diagnostic 行仍完整采样、保存和报告，但它们的 verdict 与专属 controls 不触发
n=31，也不进入产品 `performance_gate`。n=31
必须重新运行完整产品 strata 与 controls，并以新 31 个样本替代统计输入；不得向原 15 个追加、挑样或
跑到刚好通过。证据仍保留 n=15 完整原始阶段；`verify` 会独立重算 n=15 controls、summary 与
内建 build/artifact gates，并把 command validator 的日志 hash、整数 exit code 和 gate 绑定后重放
停止决策，证明当时确实应当（或不应当）升级。跨平台样本永不池化。

## 8. 子进程环境、构建和 Formal 边界

runner 不继承未经审查的性能开关。启动每个 benchmark child 前，它会清除 sanitizer/profiler、
`LD_PRELOAD`/`DYLD_*`、allocator tuning、OpenMP affinity 等 blocklist，以及所有 ambient
`HYLE_BENCH_*`/`HYLE_PERF_*`；随后把常见数学/OMP 线程数固定为 1，只恢复合同声明并列入
`controlled_environment` 的 stratum 值和登记的 macOS QoS。该策略仍保留普通进程环境，并记录
`PATH` 的 SHA-256 而不是原始环境，所以它是明确的 sanitizer，不是假称完全 hermetic sandbox。
第三方 runtime 开关默认同样不能进入证据进程。v1 唯一精确 allowlist 是
`TRACY_ONLY_LOCALHOST=1`：runner 先清除 ambient 值，只有合同同时声明 name/value 时才恢复；它把
Tracy profiler listener 限制在 loopback，避免 Windows 为每个新 benchmark artifact 请求局域网防火墙
例外。`TRACY_NO_BROADCAST`、`TRACY_ON_DEMAND` 等会改变 runtime 机制的开关仍被拒绝，不能借此调分。
runner 还为每次启动加入 128-bit `HYLE_PERF_RUN_NONCE`；benchmark 只把它原样写回 telemetry，
它用于证明样本集合的进程边界与防止误复制，不是密码学签名。
macOS orchestrator 还会在整个加锁运行作用域内应用并恢复 profile 的 QoS，使 build 前后 host probe
与合同一致；每个 benchmark child 仍必须自行应用并通过 telemetry 证明同一 class。

### 历史证据与环境策略演进

`child_environment_policy` 是采样身份的一部分，但不能把“今天的 sanitizer 清单”误当成所有历史
artifact 当时都必须记录的事实。旧 v1 曾在加入 Tracy blocklist 时被错误地原地扩展，因此仓内保留了
“不含 Tracy”与“含 Tracy”两个真实 v1 指纹。runner 现在只生成不可变 v2；verifier 按 manifest 记录
的 schema/完整字段验证已知精确指纹，不再拿今天生成的记录冒充旧证据当时的策略。

遇到这种差异时必须：

1. 保留旧 manifest、原始日志与 hash，禁止手工补字段或把今天的策略伪装成当时记录；
2. 先比较唯一策略差异，再从 frozen contract、selected component、CMake option、链接产物和 runtime
   dependency 证明它是否可能影响被测 artifact；
3. 若 verifier 有显式兼容规则且 frozen contract 能证明差异与 artifact 无关，可继续深回放并单独报告
   compatibility；若不能证明，当前回放记为 instrument compatibility `INCONCLUSIVE`；历史 verdict
   仍绑定原 candidate/contract/instrument receipt；
4. 若差异相关或无法排除，使用新 contract/instrument identity 重新 qualification，不继承旧产品门；
5. 对活动收口轮，缺少所需 P1/P2 verdict 仍按硬门 `BLOCKED`；对已 `CLOSED` 模块的只读回放，按
   `METHOD_GAP` 处理，不能仅凭工具策略漂移追溯改写产品状态。

当前兼容规则只有一条：缺少 `TRACY_ONLY_LOCALHOST` 清理的原始 v1 只在 frozen runner cache 明确
`HYLE_TRACE=OFF`，并且 controlled environment/全部 strata 都未使用该变量时，报告
`LEGACY_COMPATIBLE_TRACE_DISABLED`。含 Tracy 的过渡 v1 报告 `LEGACY_SCHEMA_CURRENT_POLICY`，新 v2
报告 `CURRENT`。任何其他删项、fixed value/prefix 变化或未知 schema 继续 fail closed；不得为了让旧包
通过而放宽任意字段或重写 manifest。未来 CUDA、allocator 等策略变化必须新增不可变 schema 和明确
兼容规则，不能再次原地修改 v2。

nanobench runner v1 只接受 `machine-profile-bound` toolchain、fresh Release qualification 和
**静态链接**：合同必须是 `linkage=static`，实际 CMake cache 必须证明 `HYLE_STATIC=ON`，并同时核验
generator/architecture、LTO、ISA、compiler ID/version、实际 compiler/linker 路径与 hash、SDK/stdlib
记录和最终 artifact hash；engineering 会保留 SDK/stdlib 观测，只有 Formal 要求它们预注册且 exact
match。这里的 static 是 v1 仪器尚未实现 shared runtime dependency closure 的
边界，不是 HYLE 产品或交付系统只支持静态库；shared 性能证据需要未来的新 adapter/版本，不能在
当前合同中假装等价。

普通 `qualify-platform` 使用 engineering 机器角色和工具链 profile。`--formal` 还要求运行前已在
registry 登记并匹配的稳定主机摘要、CPU/内存/主板/OS/BIOS facts、独立物理机器，以及 profile 中
**预先钉死**的 compiler/linker ID、version、binary hash、SDK 和 stdlib 八项 exact
`formal_toolchain`。observed toolchain 不能反向成为 expected。当前 W1-WIN 已登记主机身份；W1-WSL
不是独立硬件，M1-MAC 尚未登记稳定身份，L1-LINUX 未 provision。首批 pilot contracts 也尚未钉死
Formal toolchain，因此它们当前只能走 engineering；这些缺项必须 fail closed。

`certify` 不重新跑 benchmark；它只接受 Formal manifests，复核相同 candidate/contract/peer 身份、
登记主机绑定与不同 `physical_host_key`，再覆盖合同要求的全部 certification platforms。工程
`QUALIFIED` manifest 不能被事后改名成 Formal。Formal comparable gate 只组合当前 policy 中明确
eligible 的 `consumer/replay + product` 行；mechanism/diagnostic 行即使是 `COMPARABLE` 也不能让证书
PASS 或 FAIL。没有当前 policy 标记的历史 manifest 仍可按原停止规则 `verify`，但新的 `certify`
明确拒绝它，避免旧的自报 replay 绕过消费钥匙。

### 8.1 信任与保留边界

v1 的本地 JSON、raw log 和 hash 是**机械一致性证据**，不是签名或硬件证明。它能拒绝缺文件、漂移、
重复样本、伪造 gate、错误停止规则和彼此不一致的记录；不能对抗一个有权同时重写全部日志、索引与
manifest 的恶意证据作者。需要这种真实性时，必须由受信 runner 把 bundle 签名并进入访问受控、
内容寻址的不可变存储。

当前 engineering bundle 仍引用 live build/toolchain artifact；后续 rebuild 会使 `verify` fail closed，
不会产生假 PASS，但它不是长期档案。产品发布的 Formal 流程在加入 artifact retention/signing 层前
保持未就绪；compiler/linker 本体若受再分发许可限制，应存内部 CAS 或以受信 toolchain provenance
替代，不能悄悄省略。

## 9. 调查与停止预算

每个 immutable candidate 只保留一份有效完整 P1 bundle；有效 material FAIL 最多进行一次
codegen/profile 机制检查，产品变化后必须形成新 candidate。验证 fixture/tool 缺陷使受影响 gate
失效，环境或 controls 失格使该 bundle 无资格：这两类可以在修正后按 task 预算重做，但不产生产品
PASS/FAIL，也不消耗产品修改名额。所有失败、失格与重跑理由必须写入 task；不得换顺序、cache 或追加
样本把有效失败跑到绿，也不得继续发明第五种 benchmark 姿势。通用仪器缺陷进入独立 instrument task；
除非证明会跨过 material 门，不追溯移动已完成模块的终点。

## 10. v1 非目标与验收

v1 不建设数据库、网页 dashboard、远程调度、自动 margin/MDE、顺序贝叶斯、通用反汇编 DSL、
模块 SOTA 总分，也不一次迁移全部旧 benchmark。

v1 完成必须满足：

- schema 缺字段、未知版本、重复/缺失 row、NaN 均 fail closed；
- synthetic A/A、5%、10%、batch-matched timer floor 和双顺序可机械重放与裁决；
- 每 child 唯一 nonce，raw logs 可重建 observations/controls/n=15 停止决策；相同 raw evidence
  确定地产生相同 summary；
- N-worker 结构化进程必须证明不同 physical core、逐 worker requested/observed CPU 与
  work/callback 守恒，且 candidate/peer row set 一致；
- build/host/artifact/environment 漂移以及未登记 Formal 身份/工具链均 fail closed；
- intrusive 与 error 两个不同消费者只通过 contract/mapping/窄 validator 接入；
- 新普通 CPU 模块无需修改统计核心或复制 runner；
- `probe` 由 n=1 机械限界；`qualify-platform` 的实际耗时按 workload/strata 数和合同资源预算报告，
  不用一个虚假的全局分钟数换取少跑必需证据。
