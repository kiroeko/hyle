# G09 native workload A/A 资格阻塞

状态：**AUDITED，2026-10-17；Windows 自动转储尝试已结束，当前只用现成调试器直接监视原诊断程序一次，区分其自身失败与采集环境；完成时延和原 A/A 原因仍未决，结果复核待 R26；产品/正式测量 FIX 集为空**。

本任务承接[活动 G09](../graph/inference-g09-joint-pipeline.md)的实际停止条件，不扩充基础库能力。
产品 candidate 固定为 `3524413272cb52d06af5e64d634edbfef09b9864`，实现为 `e09f30925`；
原 private consumer source 为 `35ec95dbfb7012c746c4a4114c5b3bcffaf3d12e`；
下述宿主入口只追加独立文件，以新的干净 private Git 提交冻结，不改该历史提交中的文件。
原仪器、源码/产物/平台和全部尝试由[检查点](../../../../evidence/architecture/inference-g09-p1-instrument-attempt/receipt.json)
及[独立 R19](../../../../evidence/architecture/inference-g09-p1-instrument-attempt/19.json)绑定。

## 已知事实与有限边界

Windows CUDA 的原 n=15 P1 已 QUALIFIED：四控制/三个 validator/输入及产物宿主稳定性通过，
两独立分支插入顺序 WIN，双 microbatch 与串行对照 PARITY，全部 5% NI 门通过。
它不能替其他平台签资格。另三组各自第一次 fresh 构建通过实际 text/路径/资源及完整 oracle，
但原 workload A/A 失败：

| profile | 失败 | 有效 A/B 样本 |
|---|---|---|
| Windows Vulkan | candidate 独立 CPU-first 的 peer-allocation 因子中位比 0.94695274，低于 1/1.05 | 0 |
| Air Apple | candidate 双 microbatch 单进程比 0.89615083，低于 1/1.10 | 0 |
| Air Vulkan | baseline/candidate 多个 gross/factor 拒绝；全体进程比 0.853571–1.259887 | 0 |

R19 独立重算 24 个 A/A 日志均与原 op/s 算法一致，完整 oracle/State 退休通过。
nanobench 的 paired relative 与两行各自 elapsed median 的倒数之比是不同统计量，不是解析错列；
不能换统计量消除失败。原 runner 的 validator FAIL 未立即抛出，但 private ready guard 在 compare
之前拒绝首个产品进程；该 fail-fast 报告问题不解释 A/A，也不恢复产品性能资格。

原静态阶段没有证实具体产品缺陷或可定向修复的工具/宿主机制。三组工程 NI 未决，方向 INCONCLUSIVE；
不把同实现 A/A 偏差记为产品 LOSS，也不降低门或丢样本。

## 阶段、验证与停止

初始 SCOPED 阶段只允许有限静态机制审计，先列出有源码/既有运行证据支持的控制、计时或绑定假设；
不能以“再采一次”作为定位方法。没有具体机制时不进入 AUDITED、不写产品或公共仪器代码，
不启动新的 GPU run、A/A 或正式 P1。确有机制后先固定独立的最小 FIX、输入/产物身份与有限
验证合同，再进入 AUDITED；任何公共性能核心修改必须留在这个独立任务。

合法恢复条件只有：发现并说明具体 instrument/host/contract 机制，独立审查修正依据；新测量前冻结
新入口，保留全部原失败与费用，在原 5%/10% A/A、四顺序及 n15→31 合同下重新取得所需资格，
再补该平台 P1。单纯追加 epochs、warmups、n、换尺寸/Provider/估计量/阈值或重复同配置不是恢复条件。

从该静态阶段恢复时仅可裁决具名机制及最小 FIX，或保留工具阻塞；不能扩成无期限测量框架工程。
G09 四产品 owner 暂记 BLOCKED-from-CANDIDATE，正确性/交付及 CUDA P1 证据保留；
线程预算 G10 作为本 goal 已授权的独立能力另卡审计，不在此任务修改，也不假定它会修复 A/A。

本次有限静态审计的[独立R20](../../../../evidence/architecture/inference-g09-p1-instrument-attempt/20.json)
反证错绑/顺序/漏计完成边界两类假说；宿主调度及idle worker竞争仍证据不足，未得到可定向修复的机制。
R20 本身没有得到新产品 FIX 或重跑依据，三组原 A/A 失败和 0 个有效 A/B 样本保持；
后续具名宿主条件单列如下。

## R21：一次具名宿主排除

后续[独立 R21](../../../../evidence/architecture/inference-g09-obs-host-attempt/21.json)确认：
Windows Vulkan 的原 9 个 verify/A/A 日志四阶段均加载 `graphics-hook64.dll`，当前 OBS manifest
声明 `DISABLE_VULKAN_OBS_CAPTURE=1` 可在进程启动时禁用该隐式层。它是可排除的外部宿主条件，
尚不证明 5% A/A 失败原因；官方 OBS 32.1 源码也不支持“每次 QueueSubmit/WaitForFences 被其拦截”
的推断。Air 系统模块名不能据此解释为捕获层。

有限工具 FIX 只有一项：专用 private launcher/validator/新合同冻结上述普通环境变量，
并验证原有四阶段实际模块快照无 OBS；不改注册表、不终止用户应用、不增加其他 overlay 开关。
旧 R18 源码、合同、facade/SDK/资源、原仪器及统计全部保留，新增入口经独立实际代码复核后才执行。

- source：`.scratch/inference-gaps/g09-p1-instrument-adapter`；新增 `validate_obs_host.py`、
  `contract-windows-vulkan-obs-free.json`、`obs-host-authorization.json`。
- launcher：`.scratch/inference-gaps/g09_run_obs_host.py`；冻结 manifest/入口/审查 SHA 和私有 Git。
- fresh build：`C:/h7/g09-p1-instrument/vulkan/qualified-build-obs-free-1`；
  output：`.scratch/inference-gaps/g09-p1-obs-free-1/evidence`；两者运行前必须不存在。
- 仅重建原入口 3 个 C++ TU，不重建产品/facade；普通构建最多 2 并发、父进程最多 4 核/BelowNormal，
  采样仍为原 `native-runtime-pair` 身份。1800 秒到期只延长同 PID，不重启/重置失败成本。
- 先一个 fresh verify：完整四形状 oracle、8 次 State 退休、正常退出、四阶段无 OBS。
  通过后才运行原 2 SDK × 2 执行顺序 × 2 分配顺序的 8 个 A/A，均检查实际模块快照。
- 保持 4M F32、2 worker、12 paired rounds/epochIterations1、8 warmups、完整完成/观察/退休/drain
  边界、5% factor/10% gross 门和原估计量。A/A 全通过才原 n15→31/P1；P1 后复查所有产品日志快照。
- 缺快照、OBS 仍加载、身份/oracle/退休错误或 A/A 失败即停止本次固定尝试，不自动试其他设置。
  `nvspcap64.dll` 只记录残余未知；不得声称全部捕获层已消失，或将新 PASS 回溯成历史因果证明。

本项只恢复一次 Windows Vulkan 宿主条件验证，不恢复 Air/CUDA 运行，不重开 G10；
G09 四产品 owner 保持 BLOCKED-from-CANDIDATE。旧失败、0 有效 A/B 和成本不覆盖。

### 本次宿主尝试终态

[R22 实际代码复核](../../../../evidence/architecture/inference-g09-obs-host-attempt/22.json)
修复一个 private launcher 异常退出后自有任务树可能继续运行的问题，随后冻结新 private source
`1757a6b3a4047790c94b43beb3492823fc0dcdf7`；产品、原 C++/仪器/R18 输入均未改。
[运行终态](../../../../evidence/architecture/inference-g09-obs-host-attempt/receipt.json)保存全部原件：
一个 verify 和八个 A/A 均正常退出，完整 oracle/退休与 36 个阶段模块快照全部通过，均无 OBS，
`nvspcap64.dll` 残余仍在。排除条件实际生效，但原 A/A 仍拒绝多项 factor/gross，
32 个同实现形状比范围为 0.886063–1.192072。它不能证明历史因果，也没有修复资格。

原 ready guard 在首个产品入口的 compare 前拒绝；有效 A/B 仍为 0，没有正式 P1 性能结论。
本次新增 case 34.843 秒，其中 A/A/verify 28.641 秒；产品/facade 构建为 0，原失败与费用完整保留。
[R23 结果复核](../../../../evidence/architecture/inference-g09-obs-host-attempt/23.json)独立核对原判据和原件。
这一次尝试已终止，不重复、不试其他 overlay 设置。当前阻塞 owner 为本工具任务；
恢复仍须新的具名 instrument/host/contract 机制、独立复核和新测量前冻结的有限修正，
不能继续复用 R21/R22 为同配置或另一开关授权。Air/CUDA、G10 和四产品 owner 状态不变。

## R24：一次完成观察的有限诊断

[独立 R24](../../../../evidence/architecture/inference-g09-completion-observation/24.json)确认：
Apple/Vulkan 的既有 pending 路径有未 signal 的 1ms 定时观察，新 frontier 同样没有 GPU 完成通知；
CUDA 原 prefix 有回调，但其新并行 fallback 仍会定时观察。baseline 已有同类成本，原卡也已披露。
TimerQueue 使用纳秒 deadline，没有固定 tick 舍入；未证 Windows 时钟错误、A/A 因果或产品 MUST。
原 exact1 确实绕过校准，但默认目标也是 1ms；不能把去掉 exact1 或加 epoch 时长预签成修正。

当前只审计一个诊断入口：在可丢弃的 Windows Vulkan candidate 副本中，对原 joint 的
poll/park/resume/complete-poll 及原公开完整 replay 边界追加预分配记录，保留原数学、ownership、
checked wait/结果观察/drain。只替换一个 hybrid invocation TU，并重链两份 private facade，
其余 SDK/叶实现、原 driver 主体与 nanobench 对象不变；driver wrapper 仅设置并核对真实
Windows AboveNormal/0x14，避免 profiler 子进程默认优先级冒充原身份。总计四个小 TU，
没有产品/API/layout、公共仪器、Scheduler 或新线程池改动。
实际代码/链接映射和有界记录归属须先独立审查，再冻结一次进程入口；未审查前不运行。
[R25 源码与产物终审](../../../../evidence/architecture/inference-g09-completion-observation/25.json)
确认：记录写者限 caller/Sched，flush 在原 close/drain 的独立 happens-before 之后；
计数器不冒充发布屏障。四个 TU 构建及离线检查通过，两 map/COFF 证明替换 invocation 与本地 recorder
绑定，旧 invocation archive member 未抽取；3 个 PE、8 个 runtime、42 个 facade 链接库、
原 driver 的 nanobench archive、237 个 SDK 头及 36 个 NVTX 头身份均已核对。
构建累计 70.890 秒（配置 60.046 秒、实际构建 8.484 秒）；此时尚未执行应用或 profiler。
一次入口及精确输入由[运行前封存](../../../../evidence/architecture/inference-g09-completion-observation/authorization.json)
固定；源码、原 help、helpers、map 与构建日志保存在同目录的两个 ZIP 中。

固定诊断边界如下：

- source：`.scratch/inference-gaps/g09-completion-observation-25/source`；
  build：`C:/h7/g09-completion-observation-25`；两者首次生成，不覆盖历史尝试。
- 原 Windows Vulkan 4M F32、四形状、两 candidate 映像、原 8 warmups/12 paired rounds/exact1，
  一个进程、candidate-first/candidate-allocation-first。这里只观察成本，任何数字都不签 A/A/P1 或排名。
- 单独固定 `DISABLE_VULKAN_OBS_CAPTURE=1`；最多 4 核、BelowNormal、build 2 jobs，重型任务串行。
  原 caller CPU2/parent0x14 采样身份保留；一个进程中额外 tracing 开销明确属于 diagnostic。
- 使用现有 Nsight Systems 2026.1.3 的 Vulkan/NVTX 应用 tracing，关闭 CPU sampling/context-switch/
  WDDM/CUDA tracing；不请求管理员、不启动 kernel ETW。只收集该 owned child，正常退出后离线导出。
- 记录计数溢出、源/对象/SDK/映像归属错误、oracle/退休或正常退出失败即停止；不重试、不换采样设置。
  1800 秒观察超时只延长同 PID；owned launcher 异常须终止自己的任务树，不放任后代采样。
- poll 驱动的 terminal 时间不充当独立 native readiness；GPU kernel/batch end 不充当公共 Completion。
  有明确同 submission 归属与共享时钟后才分析完成到观察的间隔，否则该项 UNKNOWN。
  park 很少或贡献/波动很小可反证本假说；park 多仍不证明等待可避免或可平均。

这不是恢复正式 P1，也不授权改 epoch/估计量/阈值/工作负载或基础库。
一次诊断后只接受可定位机制和可审的有限纠正，或保留未决；不扩成新的观测框架。

### 用户授权：实际管理员身份重试

上述首次入口的 Nsight 以 code=1、0.250 秒退出，累计 0.344 秒；报错为注册 Vulkan extension JSON
需要注册表写权限，未生成 trace，也无目标 policy/oracle/退出标记。原 `run` 与原授权保留。
用户随后明确要求“你已经在管理员终端了，重试”；新进程已实际核验 `IsUserAnAdmin=true`，
组身份为 High Mandatory Level。该外部状态变化和本次明确授权覆盖上面的不重试停止规则，
只允许再执行一次原配置，不把权限失败解释为 A/A 原因。

重试 launcher 为 `.scratch/inference-gaps/g09_run_completion_observation_admin_retry.py`，
只更换授权/输出路径并加入实际管理员断言；原源码、已审产物、四形状和 Vulkan/NVTX 采集参数均不变，
不重建。独立输出为 `g09-completion-observation-25/run-admin-retry-1`，原失败不覆盖；
输入由[重试封存](../../../../evidence/architecture/inference-g09-completion-observation/retry-authorization.json)固定。
采集之后仍须实际 oracle/退休/正常退出、记录完整度、时钟与 submission 归属裁决；不恢复 A/A/P1。

### 两次观察的实际结果

[结果及全部原件](../../../../evidence/architecture/inference-g09-completion-observation/receipt.json)
保留首次权限失败和管理员重试。重试已越过注册限制，实际应用 PID/策略/映像及 nonce 在 trace
内嵌 stderr 中可见；Nsight 的外层日志没有转发这些输出，不能据此说应用未启动。
前两个形状的完整数学 oracle、四次 State 退休及四块无溢出记录通过，共 3022 条记录、88 次完整 replay；
随后第三形状准备期间以 `0xC0000005` 退出，没有正常 main 退出标记或后两形状验收。
此为新诊断入口的失败，尚无栈证明崩溃由 profiler 或产品引起。

离线导出成功，实际表为 186 个 NVTX、1466 个 Vulkan API 记录；没有 GPU workload 表，
collector 还提示 NVTX/Vulkan 事件可能不完整。176 个已退休 marker 的 QPC/NVTX 对齐区间一致，
一个未退休 marker 无配对；它不补齐 native readiness 或同 submission 的完成归属。
仅对前两形状各映像的原 12 次计量 replay（排除 oracle/warmup）计算：native frontier park
并集的中位数为 3.196/3.803/3.937/4.302ms，占 replay 的中位比例为 51.4%/53.2%/54.7%/54.1%。
旧 unsignaled progress park 在这些记录中为零。park 包含实际 GPU/provider 工作及调度等待，
不能写成可消除开销，caller finished wait 也不得重复相加；原 A/A 因果仍 UNKNOWN。

新增构建和两次尝试累计 76.968 秒，另离线导出 0.047 秒；原失败与成本完整保留，0 个有效 A/B，
CUDA 原 P1 与三个未资格平台的结论均不变。两入口均已终止，没有追加采集。
当前具体障碍为诊断入口访问冲突且没有独立 native readiness；须先取得可定位的故障/工具机制并
复核有限纠正，才能冻结另一入口，不能由部分 host 记录直接修改产品等待或测量统计合同。

### 具名故障定位：一次 Windows local mini dump

上述实际 `0xC0000005` 是新诊断入口的可复现故障事实，当前工作限于取得该故障位置，
不据此恢复测量资格。既有日志未带异常地址/栈，有限本机查询未取得 dump，亦无现成调试器。
按 Windows 官方 LocalDumps 接口，仅对唯一 `g09_observed_driver.exe` 设置一次 mini dump；
无 automatic-debugging Auto 配置，当前管理员 token 已核实。未取得具体位置便不修产品/仪器。

- 复用 R25 已核对的完整 source、SDK、两 facade、driver 和原 Nsight 采集参数，无编译。
- 本轮 FIX 仅为可丢弃 launcher 增加按该 exe 的 `LocalDumps` 临时配置，`DumpType=1`、
  `DumpCount=1`，输出到 fresh `g09-completion-observation-25/wer-crash-1/dumps`。
  仅在应用专属键原本不存在时创建；finally 核对并删除自己的键，原父键若不存在只在恢复为空时删除。
  不改变系统全局 dump 值、JIT 配置、GPU/overlay 或统计设置。
- 运行仍仅一个原配置进程，独立路径 `run-wer-crash-1`；正常结束或首次异常均停止。
  原失败、授权、输出和费用保留；1800 秒只追加同 PID，异常清理仅限自有任务树。
- 只离线读取 dump 的异常/模块信息及实际保存的 trace；先确定异常地址归属，再决定有限纠正。
  profiler 内的故障地址也不自动证明其根因归属；缺 dump/缺归属写 UNKNOWN，本轮不再追加复现。

这是活动 goal 中修复自有诊断失败的有限工具动作，不是产品 FIX、性能重跑或新的采样框架。
应用使用自有 crash reporter 时 LocalDumps 可能不产出；此限制保留为真实失败结果。

[本次终态及原件](../../../../evidence/architecture/inference-g09-completion-observation/wer-receipt.json)：
一个进程再次以 `0xC0000005` 退出，仍在前两形状 oracle/四次 State 退休后、第三形状第二实例
的 cold-create 标记前中断；这只是可重复的边界，不是已定位的指令/模块。
保存 2996 条 source 记录及部分 trace，未生成 mini dump，没有正常 main 退出标记。
`registry_restored=true` 且实际复核专属键已不存在，父键原本存在且保留；417 项冻结输入均仍匹配。
新增尝试累计 6.547 秒（内层 4.906 秒），另离线导出 0.047 秒；无编译、产品改动或有效 A/B。
本轮按停止规则结束，不追加 native 复现，不把无 dump 推断为特定 crash reporter 的因果证明。

### 一次直接调试的区分检查

Windows 自动 dump 未能给出异常地址；新的有限工具动作只使用微软签名 ProcDump 12.02，
直接以 `-mt -e -n 1 -x` 启动同一已冻结诊断程序一次，不经 Nsight。它有现成的异常监视接口，
不编写产品异常处理器或新观测框架，不重编译。原四形状、source/SDK/两 facade/driver、数学、
等待、采样身份与 `DISABLE_VULKAN_OBS_CAPTURE=1` 不变；这个直接调试结果只用于区分故障范围。

- 入口 `.scratch/inference-gaps/g09_debug_original_observation.py`，唯一 fresh 输出
  `.scratch/inference-gaps/g09-av-debug-1/run`；工具 `C:/h7/g09-procdump-1/procdump64.exe`。
- 若异常复现，仅收一个 triage dump；若正常，则要求原 4+4 oracle、8 次 State 退休、8 块无溢出记录、
  实际策略/nonce、四阶段无 Nsight/OBS 和 main 正常退出。正常一次不证明 profiler 是根因。
- 不安装系统 postmortem debugger，不改变 AeDebug/LocalDumps，不启动新 P1/A/A 资格测试。
  失败清理复用已审的自有 PID/nonce/完整路径检查；原始混合 UTF16/narrow 日志保留，只给标记解析移除 NUL。
- 一个进程后停止；只离线检查已有 dump/记录。没有复现或归属仍 UNKNOWN，不扩成压力或 profiler 矩阵。

这是具名访问冲突的普通故障定位，前两个采集入口及 Windows dump 尝试的终态、费用与恢复约束均保留。
