# G09 native workload A/A 资格阻塞

状态：**AUDITED，2026-10-17；固定一次 Windows Vulkan OBS 宿主排除验证，产品 FIX 集为空**。

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
