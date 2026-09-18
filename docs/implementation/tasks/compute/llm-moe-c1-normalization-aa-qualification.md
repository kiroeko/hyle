# C1-AA-Q1：归一化同实现对照资格恢复

## SCOPED

- owner：`/root`；模块专属 `bench/graph_provider_onednn/normalization_preflight.py` 的验证任务。
  不新增产品 factor，不重开 SIMD、Sched、Platform 或公共性能统计核心。
- baseline：`64e1e169`，F1 产品为 `ae270af9`；Windows 是唯一产品真相。
- 当前授权来源：[oneDNN 活动卡](llm-moe-c1-onednn-normalization.md)的 AA04 资格阻塞。
- trigger：F1 的采样前 workload A/A 通过，n=15 的120个产品进程及30个公共控制进程完成，
  五项公共控制均通过；采样后 workload A/A 在256个比值中有一个1.21435，资格 FAIL。
- 证据：[完整失败与限定诊断](../../../../evidence/performance/onednn-normalization-f1-64e1e169-w1-win/README.md)。
  失败项是 oneDNN 对自身的 Direct LayerNorm 1×2048、primary、执行与分配均 peer-first；
  两个角色虽然保留候选/peer 标签，但此 A/A 模式实际都调用 oneDNN，不能误称 Native 输了。
  该进程内部配对区间也排除1，不能把它直接解释为一个 timer tick。

## 已固定事实

1. source、artifact、host 身份未漂移；Release `/O2`、同一 P 核和自然完成边界保持。
2. 所有 A/A 聚合及 variant/order/allocation 中位数仍在原5%带内；失败来自单个进程比值的
   10% gross guard。不能据此擅自删点、放宽门或把整轮改写成 PASS。
3. 唯一追加的有限诊断已经结束：同一 primary binary、oneDNN/oneDNN 的四种顺序/分配组合，
   受影响项依次为1.00731、0.99646、0.95003、1.01484。没有复现21%偏差；这不证明具体 OS、
   JIT 或代码/数据地址原因，也不替代原失败。无未使用的原样完整重跑额度。
4. F1 只是冷融合准入修复，没有 SIMD、Sched 或热状态改动。该资格异常不能授权修改那些产品。

## 有限调查与 FIX 准入

追加静态调查最多30分钟，只允许两条具名路径：

- 对照合同：核对本模块 gross guard、独立进程/固定因素中位数及公共工程门分别声明什么；
  解释单实例差异与进程中位数结论的关系，不能因已看到失败就修改统计目标。
- 实例差异：核对两个真实 oneDNN plan 的 JIT、资源/代码地址、预热和完成责任是否同义；
  若缺少失败实例的必要观测，就明确证据缺失，不通过反复运行等待再次失败。

先把具体反例、拟修的有限验证代码和验收写成 AUDITED ledger，再改模块验证代码。没有具体缺陷时
不得改公共仪器、性能阈值、产品 API、常态 metadata 或源算法。原始失败和全部观测始终保留。
本卡不是额外产品优化轮；oneDNN 主卡仍只剩原定第二轮性能纠正预算。

若确有可修的模块验证缺陷，验收包括：同义对照、故意固定偏差必须被拒绝、原异常必须可追溯、
制品/合同/机器身份与失败传播、全部过程有界；使用原统一性能仪器。任何恢复后的完整比较必须
先冻结新候选和合同，最多一次，不能追加原样重试。若方案要求改变公共仪器或放松原工程门，
先形成独立、可审查的具体方案，不能在这里顺手实施。

## 停止与恢复

当前仅 SCOPED，尚无验证代码 FIX，也没有性能 PASS。到调查预算后必须给出原因/证据缺口及
明确修复或 blocker；不能继续无界研究。主卡两个 owner 暂为 BLOCKED；Q1 形成受审资格恢复
方案后，主卡按 BLOCKED→SCOPED→AUDITED 重新绑定剩余 AA04，再冻结候选验证。
默认规则、Native 取舍和 C1 整包仍未完成；其他已授权且独立的 C1 工作不由此自动变成 BLOCKED。

## 有限源码复核与 Q01 冷观测

2026-09-19，基于 `c0aa5ae7` 的第一遍复核：两边均由同一个 `Problem` 提供数据、参数和
独立 FP64 oracle；`measure<Route>` 的 selected A/A 确实创建两个 oneDNN Route。两边各有
Context/Queue、immutable plan、invocation、输出及 workspace，均在同一 submit/wait 边界完成。
`Bench::compare` 使用共同 iteration count 和交错配对；父进程使用
`SetProcessAffinityMask`，不是只设置无法由子进程继承的线程亲和性。没有发现这几处语义或
拓扑不对等，也不据此宣布失败实例没有地址效应。

两个 oneDNN plan 会分别生成 JIT 制品；当前日志只绑定 EXE，没有失败实例的 JIT bytes、代码
地址、数据地址或逐段运行干扰事实。不能从源码恢复已退出的异常实例，不能把四次未复现写成根因。

先固定唯一冷观测 Q01：使用已核验的 F1 primary binary，`HYLE_BENCH_AA_ROUTE=selected`、
`HYLE_BENCH_VERIFY_ONLY=1` 和 vendor 自带 `ONEDNN_JIT_DUMP=1`，只执行一次独立进程。
保留原 oracle、绑定/完成检查，收集并散列实际 JIT 制品，核对同 workload 两个实例是否生成相同
代码。沿既有 P 核策略、DLL邻置和禁弹窗设置；输出只在有界临时目录，核验归档后清理。
该动作没有 `Bench::compare` 或计时样本，不改产品/benchmark/公共仪器，不重签旧 P1；它最多
排除本次冷观测中的代码生成差异，不能重建历史实例地址或证明其性能稳定。

## Q1 调查裁决：BLOCKED，停止追加观测

上述一次冷进程退出0，16个 workload 的数值/绑定/完成检查通过。实际生成32个 JIT 文件，
按 benchmark 的逐 workload 构造次序及 vendor 加锁递增的 `unique_id` 对应16对；每对字节及
SHA256完全相同，包括历史异常所对应的 Direct LayerNorm 1×2048。约805KB原始制品、执行脚本、
日志和逐对哈希见[Q1记录](../../../../evidence/performance/onednn-normalization-aa-q1/README.md)。
这不证明历史失败实例也没有地址、缓存或执行干扰，也没有新增任何计时或性能资格。

两条静态路径已检查完，在30分钟上限前停止：本模块单实例10% gross guard 与各固定因素5%
中位数门是不同的预声明要求，后者通过不能覆盖前者失败；两个独立实例的实际工作与完成边界
未发现不等价。历史失败实例缺少代码/数据地址及干扰记录，现有材料不足以定位那一次21%差异。
Q1 **BLOCKED**，没有可由具体反例授权的产品或仪器 FIX，不修改 SIMD/Sched、不放宽门、不原样
追加完整重跑。主卡两个 owner 的 BLOCKED、原始失败及默认规则未签署状态保持。

恢复需要先审计具名、有限的模块验证改动：保留每次 preflight 的独立原始记录与实例观测，避免
只留下最终检查日志；证明它不改变数学、完成、拓扑、热路径或统计门后，再冻结唯一新实验。
仅增加日志本身不能写作资格恢复。公共核心、产品常态 metadata 或放松阈值均不在这张卡授权内。
独立的 C1-Y ggml Vulkan 能力绑定继续推进，不以该单项阻塞扩大为整包停工。
