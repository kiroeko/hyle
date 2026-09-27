# D5 Vulkan 完整输入循环 A/A：原门仍未通过

候选`3a2e1f01`，产品SDK仍`9273bc71`；
[有限工具卡](../../../docs/implementation/tasks/compute/llm-moe-d5-vulkan-measurement-balance.md)。
W1-WIN / RTX4070 / 原native-runtime-pair双P核策略；没有要求用户进一步关闭桌面程序。

## 已实现和已验证

- 独立校准取消；每轮固定6次完整submit/wait调用，覆盖Qwen 2槽、MoE 3槽相同输入集合。
- 每侧24轮，计时后输出全部CompareResult；每进程96条记录，8进程共768条，完整性通过。
- Graph/Graph、Direct/Direct，双执行顺序、双分配顺序；每进程计时前后均通过原正确性检查。
- 复用SDK，仅编译Vulkan消费者；DLL邻置、隐藏子进程；本批A/A约42秒后结束。

## 预声明主门

| A/A分组 | 结果 |
|---|---|
| Graph/Graph Qwen | PASS |
| Graph/Graph MoE | PASS |
| Direct/Direct Qwen | PASS |
| Direct/Direct MoE | FAIL：单进程1.11169；执行顺序因素中位数1.08040 |

仍采用旧validator同样的两侧op/s比值、10%gross和5%因素门；没有新建ready marker，
正式Graph/Direct性能样本为0，没有n15/n31、没有产品性能FAIL。原BLOCKED不解除。

## 新机制证据（只作诊断）

`paired-offline.json`由本批精确逐轮输出提取，不是旧四舍五入表格。四个A/A分组的配对统计
均落在同样幅度门内。失败的`peer-0-0` MoE中，两个独立中位数分别1.005275ms、1.11755ms，
比值1.11169；逐轮配对后相对比值约1.01392。

这证明丢失配对关系会实质改变该批A/A裁决，不能把11.17%直接归因于HYLE路径。
但本次没有预声明用配对估计量作为主门，不能事后改主门给PASS。后续若修正该接缝，
必须独立冻结统计合同、资格和预算；本卡到此停止，不继续GPU采样。

## 文件和身份

`aa-receipt.json`含完整candidate、EXE/source/SDK/fixture SHA和执行策略；`raw/*.log.gz`
保留全部输出，解压SHA与收据一致。`run.py.gz`为本次有限入口，复用现有机器策略与解析器；
没有更改公共统计核心。`build.log.gz`保留消费编译/链接输出，包装脚本打印中文日志时出现
编码异常；编译已完成，后续冻结EXE的8次运行均成功。这不是产品编译失败或重测理由。
源代码审计及之前失败保留在[残余审计](../../research/llm-moe-d5-residual-contract-audit/README.md)。

本批只修验证消费者；固定22画像覆盖仍154通过、20拒绝、2数值失败。
统一脚本统计实现60,373行/241文件，不含测试和第三方源码；见`loc.json`。
