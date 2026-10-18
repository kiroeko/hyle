# D5 Vulkan：中断恢复 A/A 未通过，停止采样

**工作负载 A/A FAIL；正式产品样本为0；P1 INCONCLUSIVE，不能判产品性能FAIL。**
候选 `d24360ca4fee884d1d765ffea840f54267f1f07e`，产品 SDK 保持
`9273bc71017a1febb69db240b6ea8970b88dd77c`；Direct reference
`c0de38c9b9d6fa32b19fb9791171ce5b8df3cb42`。Windows W1 / RTX4070，原Vulkan合同，
固定P核调用者和native-runtime-pair CPU集合，执行与分配双顺序。

## 实际结果

用户已尽量降低背景占用，并明确允许在仍有波动的普通桌面环境测试。本次是预先登记的唯一完整
中断恢复，SDK复用，仅编译外部consumer/control。8个A/A进程均完成，输出与资源检查未报错，
但同实现比较未达到原5%因素中位门、10%单进程粗不稳定门：

| 自比较 | Qwen BF16 FFN 各进程吞吐比范围 | DeepSeek GGUF MoE 各进程吞吐比范围 |
|---|---|---|
| Graph / Graph | 0.92085 ～ 1.12923 | 1.00549 ～ 1.08431 |
| Direct / Direct | 0.95435 ～ 1.22453 | 0.93156 ～ 1.00662 |

以上是同一执行方式的两个实例间比率，**不是Graph/Direct性能差**。完整因素中位数、错误与原日志见
[A/A收据](run/d5-workload-aa.json)。不能仅凭这些数据唯一归因于桌面应用、驱动、地址布局或调度；
更不能因CUDA本次通过而推定Vulkan必然是产品性能回归。

validator未创建ready marker。公共runner尝试的第一个产品子进程被已有guard立即拒绝，
日志只有`this artifact has no passing workload A/A receipt`；没有正式Graph/Direct样本、
没有公共control样本、没有n31，也没有完成manifest或严格重放PASS。
launcher已退出，采样mutex正常释放，核对无HYLE测试进程继续使用GPU；随即通知用户。

## 已有证据与停止

[上次中断收据](../graph-vulkan-d5-fragments-interrupted/README.md)的8进程A/A PASS、
完整n15和8个n31日志仍保留。其n15公共控制PASS、MoE诊断非劣PASS、FFN区间未裁决，
但缺少最终资格记录；不与本次数据合并，也不据此冒签完整P1。
CPU、Apple以及[本次CUDA](../graph-cuda-d5-fragments-p1/README.md)的合格P1继续有效。

本次恢复预算到此停止，**适用性能门BLOCKED、聚合INCONCLUSIVE**。不重试到通过，不再要求用户
关闭应用、不改电源、不加样本或放宽5%门。恢复须先有针对上述自比较偏差的具名测量修正/环境变化，
重新限定预算并通过相同拓扑A/A，或者由用户明确修订工程验收政策；仅文档更新或重复原命令不解除。
没有证据授权修改产品算法。正确性账本的Vulkan34通过/8拒绝/2数值失败保持原判。

[索引](index.json)绑定原路径与全部SHA256；15份原始文件包括8个A/A日志、拒绝日志、
一份validator日志、A/A JSON、三份构建输入和固定合同。日志/构建输入gzip解压后保持原字节。
