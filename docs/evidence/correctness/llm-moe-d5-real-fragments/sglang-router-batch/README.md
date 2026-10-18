# D5-F25：成熟 DeepSeek 融合路由与四路线 MoE 回归

候选 `e25c5188b3c61798fe96fc90cb12384086dd2d90`，审计提交 `bc4c3a48`。
[receipt.json](receipt.json) 保存命令驱动、日志 hash、范围及未完成门。
原始日志以 gzip 无损保存，收据同时记录压缩文件与解压内容的 hash。

| 路线 | Direct | Graph | 本批范围 |
|---|---|---|---|
| CPU | 2/2 PASS | 2/2 PASS | 十三画像得分路由、八个 GGUF 完整 MoE 链 |
| CUDA | 5/5 PASS | 3/3 PASS | 同上，另有原版路由边界及 Direct 资源/OOM |
| Vulkan | 2/2 PASS | 2/2 PASS | 同 CPU |
| Apple Air M4 | 2/2 PASS | 2/2 PASS | 同 CPU |

共20个测试组通过，无失败、无跳过；每组内部遍历共享画像表。公共 schema 55/55、受影响
SGLang 能力回归6/6、两个独立头检查均通过。完整画像 ledger 仍为 **72/176 RUN_PASS**：
本批是必要片段证据，不能代签其余 attention、官方权重格式等尚未完成的 region。

## 产品与数值边界

CUDA 普通 Direct/Graph 为 DeepSeek 使用固定 SGLang 的完整原生融合程序：稳定 sqrt-softplus、
加偏置选择 top6、原始分数归一化和缩放。E256/384、T1/513覆盖原小/大 token 分支，六种输入
覆盖大 logit、并列、下溢、NaN 非法数据安全、负尾部及近并列，分别两次变化输入重放。
非并列 ID 精确比较，并列检查合法且无重复的成员；原权重误差门不变。

保留原两个 kernels、SUM 归约和 launch policy；仅把上游小 token 分支的一处数组读取移入
既有 ID 检查，避免索引-1。没有新 HYLE 数学/kernel。公共 `SqrtSoftplusRouting` 是完整片段
描述，通用 Softplus 和稳定 TopK 合同不变。CPU/Apple/Vulkan 使用已有成熟组合；DeepSeek 的
零分母 floor 改为 F32 最小 normal，非零 F32 softplus 的平方根不会触及此 floor。
官方 BF16 bias 在消费者准备时显式转换为 F32，未伪造同一制品内的统一 dtype。

Direct 验证 host/错误资源、输入输出/输出输出别名、输入数量拒绝、拒绝后重用、两个独立
Queue/lane，以及逐次 metadata 分配失败回滚和重试。原生程序无设备 workspace、无热分配，
冷绑定后沿既有 shared stream/Completion 与 Graph capture 执行。

## 构建检查中的两个具体问题

首次 configure 的 dependency hash 失败保留在 [原始日志](initial-lock-order-FAIL.log.gz)。临时
Python 工具按 Windows Path 排序，与 CMake 的大小写敏感路径排序不同；改为 portable 字符串
排序后得到相同 hash，未改变锁定源码或绕过校验。

临时 Windows 驱动还缺少固定 `VSLANG=1033`：CMake 缓存英文 showIncludes 前缀，但编译器
输出中文，导致 Ninja 记录0个头依赖。三个构建树中57/83/12个 C/C++ 对象已精确失效；
只删除经绝对路径检查的对应 `.obj`，保留已通过的二进制、CUDA 对象、源缓存和其它产物。
修正环境后的 Vulkan 消费者记录93个头依赖，证明后续增量编译可跟踪公共头。CPU/CUDA 缺失
对象会在下次必要构建中恢复；诊断语言与缓存修复不改变本批已编译运行的产品语义，不重跑 P1。

## 尚未完成

官方专家 payload/scale 格式、DeepSeek hash 层、其它 attention/MTP、完整选中 owner 门、
D5 三种 C++ 交付及适用 P1 仍未完成。selector 为 READY，不等于全部 selected gates 已通过。
巨型模型权重是符合固定真实 header 的非零合成 payload，resident experts 为 top-k+1；
不声称整模型推理。统一实现统计 **58,561行／232文件**，不含测试和 vendor，见 [LOC](loc.json)。
