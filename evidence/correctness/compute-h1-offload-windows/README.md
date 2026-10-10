# H1 offload：Windows 已完成验收与内存中断

产品候选为 `7eaf67b8a496ee19b609533980cd69e7efb5aa49`。本记录保存 H1-07至09新合同下
实际运行结果，不替代原合同的历史证据，也不签整包 CLOSED。

| 范围 | 实际结果 | 证据 |
|---|---|---|
| Windows source，单并行编译/测试 | PASS，5/5：普通 Graph、普通 Compute、CPU+CUDA组合、Direct权重迁移、组合contracts | [receipt](source/receipt.json)、[测试日志](source/source-test.log) |
| Windows installed shared SDK | PASS，3/3：组合、Direct权重迁移、组合contracts；统一stage将DLL置于exe目录 | [receipt](shared-sdk/receipt.json)、[组合](shared-sdk/joint-test.log)、[Direct](shared-sdk/direct-test.log)、[contracts](shared-sdk/contracts-test.log) |
| Windows static producer | 构建及安装 PASS；同一producer切换static，复用未变vendor对象 | [receipt](static-sdk/receipt.json) |
| Windows installed static SDK consumers | INTERRUPTED；组合configure中用户报告内存告警，尚无消费者测试结果 | [中断记录](interruption.json)、[configure尾部](static-sdk/joint-configure.log) |
| 四个公开源码头各按shared/static宏独立语法检查 | PASS，8/8；仅源码header语法，不是installed header/链接/搬迁验收 | [header receipt](source-header-receipt.json) |
| WSL新候选、安装头/资产/搬迁与新的native trace | 未运行或未裁决，不能写PASS | [归档索引](archive.json) |

source及shared运行覆盖普通CPU主存权重参与真实MatMul、CPU↔CUDA激活、有界staging、完整FP16
gated FFN、共享资源预算的strict/partial、超额回滚、alias去重、双槽/旧lease、准备失效/host-wait
拒绝，以及原迁移/取消/错误/关闭/owner丢弃合同。具体消费代码以
[候选文件hash](candidate-files.json)为准；通过边界仅上述Windows实际入口。

原静态编译曾触发预声明的12GiB可用物理内存保护，自己的编译进程树被停止，见
[首次保护记录](static-sdk/memory-interruption.json)。内存回升后使用相同对象续编，producer及安装
完成。随后用户再次报告Windows内存告警，本轮root Python PID75544及其子进程被停止；会话
终态exit1，停止后没有本轮编译或验证残留。静态消费者不能因producer通过而代签。

停止后的只读观察：可用物理内存约46.5GiB，提交内存约35.3/75.3GiB（46%），页面文件分配
11,842MiB、当时使用2,956MiB。它们不是告警瞬间峰值，不能据此否定用户告警或唯一归因。
当前driver只检查可用物理内存；后续恢复前需兼顾提交内存余量、编译进程峰值及机器上其他工作，
不能以物理余量单独证明安全。重型工作继续停止，等待用户明确继续。

[原task-private driver](validation-driver.py)保留实际运行入口；它是证据，不是新的公共仪器。
归档均为小型文本、receipt及hash；没有复制producer、SDK或CUDA对象，没有启动P1，也没有
以中断刷新性能/失败预算。
文本日志仅剔除行尾空白；索引同时保存原始与归档sha256，未改日志内容或结果。
