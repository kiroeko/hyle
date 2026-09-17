# R7 验收证据（进行中）

当前不可变候选为 `87ae583407cbb5ed13108c3837b020e6e343a3ca`。R7 仍在验收，不能由本页已完成的
P0 / configure 项代替剩余跨平台、sanitizer、TRACE OFF/ON 三交付及 fixture 检查。

## 已完成的结构与配置检查

- [P0 源码证明](p0-source-inheritance.json)：四份运行源只增加静态 trace hooks；十五个算法、
  路线和依赖输入保持不变。冷报告按节点线性复制，没有新增运行共享状态、锁或原子。
- [关闭跟踪的生成代码](off-codegen-final.json)：真实 MSVC Release shared 对象中的 29 个函数 /
  COMDAT 保持指令和调用目标；仅消除已逐项证明的匿名 namespace 路径 hash、同一 fatal assert
  的源码行号与路径差异。这是 P0，不是新的耗时测量，也不证明整个 binary placement 不变。
- [B1 配置结果](b1-configure.json)：九个实际 configure 正反例通过；absent factor 不泄漏实现、
  私有依赖或 Tracy。配置 probe 只验构建条件，不充当任何真实计算路线。
- [输入与原始归档身份](p0-configure-identity.json)保存 source archive 创建记录、脚本、配置日志和
  全部 hash；[原始归档](p0-configure-raw.zip)已逐条重算 hash。

产品代码来自 `e84a7ad3`。之后三个提交只修正测试：MatMul 原有语义是 declared-type accumulation；
custom operation descriptor 必须 trivially copyable，寿命测试把 shared owner 放在实际 lowering
callback 中。`src/include/cmake/samples/bench/scripts` 在两个候选间逐树一致。生成代码记录保留
当时的候选身份，未把旧二进制冒充重新构建的 `87ae5834`。

本轮没有新增 P1 采样。历史资格仅按已证明不变的计算路径及原机器、版本、格式和 shape 继承；
TRACE ON 的 profiler 成本不属于 OFF 性能证据。原 TSan 不记作本轮重跑。

统一 [LOC 原始结果](implementation-loc.json)统计整个 Compute + Graph 为 **37,035 行 / 130 个
代码文件**，不含测试、bench 或 vendor。它包含所有已登记 backend/provider，不是完成百分比。

## 待完成

Windows broad selector 的失败原始记录与定向测试修正后续跑分别保留，待全部 selected gates 完成后
归档覆盖表。其余有限验收按 [R7 当前卡](../../../docs/implementation/tasks/compute/four-route-foundation-final-review.md)
执行；未执行项保持待验，六个 owner 尚未 CLOSED。
