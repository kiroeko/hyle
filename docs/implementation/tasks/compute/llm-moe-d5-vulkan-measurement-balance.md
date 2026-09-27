# D5 Vulkan：输入循环平衡与逐轮记录

<!-- MODULE REVIEW: BLOCKED -->

## SCOPED → AUDITED

baseline `416242f8ef2cd9a83d558b3b7879e4c2c1707be4`；Windows canonical。
owner为D5验收工具维护者；独立工具任务，产品FIX为零。`graph`和Vulkan路线仍按
[原停止记录](llm-moe-d5-vulkan-performance-blocker.md)BLOCKED；本卡不提前解除。
来源：[残余审计A4](../../../../evidence/research/llm-moe-d5-residual-contract-audit/README.md)。

真实问题：两侧独立校准可改变各自2/3槽的起始位置，epoch也未覆盖相同输入集合；
旧日志丢弃CompareResult，无法还原每轮计时与迭代数。没有证据认定它是历史偏差唯一原因。

## 有限 FIX V1–V3

1. 只在Vulkan消费者声明固定每轮6次、24配对轮。6为两个既有形状2/3槽的最小公倍数；
   从已有verify归零位置开始，禁止自适应校准改变起点。每次仍执行普通submit/wait和释放。
2. 使用nanobench公开CompareResult记录每侧每轮次数、每调用墙钟和精确paired ratio；
   输出在计时后。公共仪器、统计核、产品API和其它路线默认测量行为不变。
3. 同一不可变候选一次8进程A/A：Graph/Graph与Direct/Direct各覆盖双执行/双分配顺序，
   验证每轮6次和24轮、资源/正确性仍通过。主门保留两侧op/s比值的5%因素/10%gross门；
   paired值只作诊断，不选择更有利统计量。

## 预算与验收

先批量编辑，SDK/原数学/模型fixture复用；只编译一个Vulkan消费可执行文件，`-j2`。
编译或记录格式错误允许一次明确修正；正确性失败即停。保留DLL邻置与隐藏子进程。
8进程A/A预算一次，不自动n15/n31；失败即记录具体失败及原始轮次，不再加轮数求过门。
通过才重新SCOPED原性能owner并冻结后续P1候选；不能在工具卡里追认产品完成。
适用验证：消费verify、逐轮完整性、A/A、selector、docs/closure、既有D5 ledger。
不重建产品SDK，不跑无关路线，不修改数值门，不要求用户再退出应用。

## 有界执行结果与停止

不可变候选`3a2e1f01`完成V1–V2；8进程A/A耗时约42秒，正确性、资源稳定、
96条/进程逐轮记录（24轮×两侧×两形状）、每轮6次与2/3槽完整性均通过。
Graph/Graph两形状、Direct/Direct Qwen通过原门；Direct/Direct MoE出现1.11169单进程
比值及1.08040执行顺序因素中位数，原10%/5%门FAIL。产品正式采样0，原性能owner不恢复。

新逐轮数据的配对统计在四组均落入原幅度门；这是有用的机制证据，但本卡预先规定主门仍用
旧估计量，不能看完数据选择有利的一个。原失败进程两侧各自中位数比为1.11169，
同轮配对比仅1.01392；证明汇总方式确实能产生不同裁决，尚非原P1的替代PASS。
详见[完整收据](../../../../evidence/performance/graph-vulkan-d5-balanced-aa/README.md)。

本卡预算结束。下一次恢复只能有独立、预声明的配对估计量/资格任务或明确环境机制改变，
不能原命令重试、增加轮数、要求用户关程序或追认旧P1。需要统计接缝修正时绑定公共仪器owner，
不在本卡扩大统计核心。可独立推进CPU/Apple成熟来源资格和剩余数值能力；新goal保持活动。
Compute+Graph实现LOC仍60,373行/241文件；本次无产品代码改动。
