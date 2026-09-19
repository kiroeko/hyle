# C1-Y34 所选工程验收

不可变候选 `ec240ac9392cb8144d326d8407d9af4b8456026a`，产品仍为 `05581df8`，测试编译修正为 `11804be7`。
[结果](selected-gates-result.json)与[原始日志](selected-gates-raw.zip)覆盖112项selected：
96项maintenance/configure/correctness/headers、1项component-contract、15项默认三交付。
8项未选理由保留；四个完整gate复用既有Y34证据，ggml Provider另复用其C++ suite，
源码合同及MASM组合本次实际执行。其余gate实际执行通过，执行与复用保存在本目录的独立收据中。

复用核对原归档CRC/SHA、不可变源码差异、EXE/DLL hash、设备/驱动与编译参数。
重新configure后163条原有编译命令逐条一致，相关编译选项不变；保留before/after证据。
Direct使用`11804be7`的91例，Provider/Graph分别使用产品候选的19/26例；公开头使用
`923895c5`的两个既有gate。没有因文档提交重跑已通过的GPU数值回归。

[交付](remaining-deliveries-receipt.json)与[交付日志](remaining-deliveries-raw.zip)证明15个
默认产品的45条source/static/shared消费，含重定位和接口/导出检查。header-only组件签
实际接口包消费，不虚构动态库。这里不替代此前W1/Air的实际归约/Softmax/L2六条GPU交付。
重型步骤串行、并行度2，Windows DLL邻exe、PATH隔离、SetErrorMode禁错误弹窗。

本次component-contract在初始化后的独立缓存放置已锁定Highway archive；无需修改产品
或公共工具。[清理](remaining-deliveries-cleanup.json)在核验归档后删除45个重复producer，
释放9,492,226,425逻辑字节；保留SDK、安装消费者、当前缓存和共享依赖。

[统一统计](current-loc.json)：Compute+Graph实现48,737行/197文件，不含测试/第三方。
本轮产品没有变化，没有P1、性能赢家或自有实现淘汰裁决。Y34本批适用工程门完成；C1其余
Provider覆盖、完整成本优选和CPU Q2 BLOCKED保持，不签C1、共同矩阵或整模型完成。
