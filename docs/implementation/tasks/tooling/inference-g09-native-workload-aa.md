# G09 native workload A/A 资格阻塞

状态：**SCOPED，2026-10-17；仅工具/宿主/合同机制定位，产品 FIX 集为空**。

本任务承接[活动 G09](../graph/inference-g09-joint-pipeline.md)的实际停止条件，不扩充基础库能力。
产品 candidate 固定为 `3524413272cb52d06af5e64d634edbfef09b9864`，实现为 `e09f30925`；
private consumer source 为 `35ec95dbfb7012c746c4a4114c5b3bcffaf3d12e`。
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

目前没有证实具体产品缺陷或可定向修复的工具/宿主机制。三组工程 NI 未决，方向 INCONCLUSIVE；
不把同实现 A/A 偏差记为产品 LOSS，也不降低门或丢样本。

## 阶段、验证与停止

本任务只允许有限静态机制审计，先列出有源码/既有运行证据支持的控制、计时或绑定假设；
不能以“再采一次”作为定位方法。没有具体机制时不进入 AUDITED、不写产品或公共仪器代码，
不启动新的 GPU run、A/A 或正式 P1。确有机制后先固定独立的最小 FIX、输入/产物身份与有限
验证合同，再进入 AUDITED；任何公共性能核心修改必须留在这个独立任务。

合法恢复条件只有：发现并说明具体 instrument/host/contract 机制，独立审查修正依据；新测量前冻结
新入口，保留全部原失败与费用，在原 5%/10% A/A、四顺序及 n15→31 合同下重新取得所需资格，
再补该平台 P1。单纯追加 epochs、warmups、n、换尺寸/Provider/估计量/阈值或重复同配置不是恢复条件。

下一阶段仅可裁决具名机制及最小 FIX，或保留工具阻塞；不能扩成无期限测量框架工程。
G09 四产品 owner 暂记 BLOCKED-from-CANDIDATE，正确性/交付及 CUDA P1 证据保留；
线程预算 G10 作为本 goal 已授权的独立能力另卡审计，不在此任务修改，也不假定它会修复 A/A。
