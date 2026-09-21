# oneDNN A/A 差距：现有证据复核

2026-09-21，按[独立后续卡](../../../docs/implementation/tasks/compute/llm-moe-onednn-aa-cause-followup.md)
先复核原Q2包，后完成两组定向诊断；未进行正式产品性能采样或冻结自研比较。
**当前工程裁决**：用户接受机制与历史限制并授权翻案。旧Q2按环境干扰最可能的无效测量
处理，撤销产品FAIL含义和M2阻塞；原始FAIL日志保留，不记性能PASS。

- [逐进程分析](existing-evidence.json)：原包hash/CRC、八个oneDNN-only进程、输入/output/
  workspace身份、JIT hash、原始表格和已知观测缺口。
- [分析脚本与结果](existing-evidence-analysis-raw.zip)：将脚本解出后从仓库根目录执行
  `python -X utf8 <解出目录>/analyze-onednn-aa.py`，复核原包并在脚本旁输出分析JSON。
- [原失败证据](../onednn-normalization-aa-q2/README.md)：保持原样。分析未重新执行原资格脚本。

17×257 Direct LayerNorm的八个比值中一个为0.906388，另七个为0.981523–1.020131。
两个角色实际均走oneDNN，相关JIT字节相同，使用共同迭代数和ABBA/BAAB配对；显示的
`HYLE Native`只是A/A未改的旧标签。失败进程的配对区间仍明显偏离1，不能简单归为单点噪声。

现有证据没有每轮原始样本、role→实际JIT入口及逐臂干扰记录，不能唯一归因。实例布局与
进程内环境影响是待验证假说，不是已证明根因。后续只允许任务卡规定的有限单形状诊断。

## 已运行的定向诊断

- [同实例与对象交换](instance-diagnostic.json)、[原始记录](instance-diagnostic-raw.zip)：
  四进程、24组，没有复现固定10%实例差；同实例点值波动不能冒充稳定根因。
- [前置布局与执行窗口](window-diagnostic.json)、[原始记录](window-diagnostic-raw.zip)：
  另四进程、24组、288个配对round；只计时17×257，前置14个原版region不作性能采样。
  前缀恢复了失败时input/output/workspace的低22位相对布局。全部正确性通过，未修改产品。

### 已证实的机制及历史限制

第二组的`run-1 / post-AA / round-5`中，两臂是**同一个物理实例**：墙钟每调用分别约
2177.57ns、1971.99ns，差10.425%；线程被记账的运行tick只差0.0454%。两窗口的线程运行
tick/总TSC比例分别83.77%、92.46%，全程同一CPU2。因此该次墙钟差主要来自未记账为此线程
运行的时间，不能说同一个算子计算变慢了10%。288对中17对出现墙钟差超过9%、线程运行
tick差不足1.5%的现象；这是诊断观察，没有据此裁决任何实现输赢或替换正式wall-time指标。

此处`QueryThreadCycleTime`在x64上使用RDTSC；本机已核对invariant TSC，并保留busy/Sleep
控制。它**不是硬件指令数或核心执行cycle PMU**，不能泛化为任意机器的cycles-to-seconds
换算。计数器解释见Microsoft的[官方说明](https://devblogs.microsoft.com/oldnewthing/20160429-00/?p=93385)。

**旧Q2那一次的唯一原因仍不可证明**：原包没有保存逐轮thread-cycle/时段记录，原始EXE也
未保留；本次没有复现其12轮聚合的0.906388。已复现的是足以造成同量级假差的真实运行时
污染机制。不能进一步指定某个后台进程/驱动，不能把历史FAIL改成PASS，也不能声称物理
cache布局等其他历史因素全部排除。用户已接受这些边界；工程上不再把该样本视为产品失败，
M2恢复正常接入，原因调查结束。完整决策见[当前任务裁决](../../../docs/implementation/tasks/compute/llm-moe-onednn-aa-cause-followup.md#当前工程裁决旧样本无效恢复-m2)。
