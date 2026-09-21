# oneDNN A/A 差距：现有证据复核

2026-09-21，按[独立后续卡](../../../docs/implementation/tasks/compute/llm-moe-onednn-aa-cause-followup.md)
复核原Q2包。没有新性能采样，没有改变旧FAIL，没有比较冻结自研数学。

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
