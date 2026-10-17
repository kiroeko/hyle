# G01 原 native ragged 实现阶段

活动卡[NR-01/02/03](https://github.com/axiomates/hyle/blob/d12f2204d23503256220e810385d6cb3e93027a8/docs/implementation/tasks/compute/inference-g01-native-ragged-layout.md)。
[receipt](receipt.json)与[raw](raw.zip)保存实际Air Mac16,12身份、全部build/run FAIL/PASS、
2020个canonical输入SHA和两产品TU/公开头/测试身份；只有source阶段，不代签SDK/san/P1。

Apple两tests PASS，F16/BF16三D、two/empty/reorder、两slots/旧result、readonly/tail、
zero cold metadata、8request输出。Vulkan四tests PASS，同F16域与实际KV roots/byte origin
仍指原capacity input1/2；原managed math之后拒绝原native Concat suffix，实际pending
保留至wait退休，无ready output，保原ResourceExhausted。大arity输出一次原Concat，
没有新增math/shader/runtime/diagnostic API，warm/driver/private费用不由结构test外推。

原Result<Lease>reset歧义、无reset成员、Vk未用desc、Apple测试池8192少于10752输入bytes
的失败均保留；已按原接口/真实总输入bytes修测试。最初Vulkan两request全managed路径
不经过HYLE QueueSubmit PFN，因此旧注入未接受计数FAIL也保留；修正为8request原
Concat suffix，真实managed prefix执行与原pending/retirement均实跑，未改产品失败机制。
Windows canonical只单向同步到可丢弃source副本；原CPU/Vulkan正式P1 git树不改。
