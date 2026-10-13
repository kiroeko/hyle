# CPU resident SDPA 的有限同合同比较

产品仍为CPU candidate `d91a523df9bce21eb81c13d9bf2a8f3f7d3ea80f`，比较适配器固定于
`901d926f4f9b14037d4586d0f3f170a199746516`，CPU原核pin为FFa4e8b80930。
[合同](contract.json)的canonical SHA为
`e270eea61921a1eacd3b12c5cb0c7cfd7d6f8964262781a59f30fd508cfdd686`。

三行均是BF16 BHSD、B1/GQA8/2、容量尾13：Q1/128、KV1024/D256 causal，和
Q17/KV257/D64 noncausal。同ready输入、readonly cache、两个invocation、一个native
thread、CPUQueue提交和checked Completion.wait，caller output回到消费者。query cast、
BF16输出转换及原native math在warm内；cold metadata、F16 mask与cplan workspace分列。
candidate与独立reference都复用公开PreparedOperation/typed kernel生命周期，reference
直接构造原flash graph，不调用candidate supports/lower，不写新数学。它们对相同
coherent映射、资源错误和close负责；不是裸kernel与完整队列的混层排名。

复杂度为原flash O(H×Q×KV×D)，不展开KV到Q/head维度。query/payload为O(QHD)，
causal mask为O(Q×KV)，workspace依原cplan。native/private pool上界两侧均UNKNOWN，
没有报告成0或总预算保证；G10另外收口。steady调用没有重建metadata/graph/plan，
两侧映射、检查与完成费用均计入。额外metadata、allocator/opaque费用可由原冷遥测
发现；本比较不承诺完整引擎代码量或token/s。

采样前按W1-WIN真实machine/affinity/priority验证。两个独立实现各在两种code placement、
双执行顺序/双allocation顺序实跑，共16个A/A process；沿原5%偏置/10%gross instability
规则，不用增加样本寻找PASS。A/A通过后才按八strata×n15和30个原公共controls裁决。
字段、源码和产物身份由原hyle-perf runner记录，公共工具不改。5%非劣与实际方向分开
报告；卡内原10%预算不抹除。

当前新三形状的两个placement/control已实际编译，独立double oracle、finite、readonly与
双slots预检通过。正式A/A和P1尚未运行；正在运行的Air Vulkan P1结束后才允许下一个
heavy任务。数学PASS不证明性能资格。变更选择器的共享bench归属缺口见
[独立工具记录](../../../docs/implementation/tasks/build/validation-selector-shared-bench-paths.md)。
