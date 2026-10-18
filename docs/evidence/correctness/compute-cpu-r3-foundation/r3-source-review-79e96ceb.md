# R3 候选的有限源码复核

候选 `79e96cebeb176c48d66f6b6a156fa9d910b89d82`；范围仅 C41–C46 的新默认组合及真实消费者。
这是源码/P0 记录，不替代尚未完成的 P1、三交付或 sanitizer。

## 生命周期和成本

- `src/compute_route_cpu/cpu_context.cc` 与 `src/graph_route_cpu/cpu_context.cc` 分别只构造至多
  两个/三个 Provider 的栈数组，随后调用已有 Context::create。没有全局 registry、线程、计时、
  用户输入试跑或新的逐 kernel 执行层。
- `src/compute/device.cc::create_context_state` 将 Provider 复制到 Context 自己的 slot，创建
  其具体状态；工厂返回后不借用栈数组。失败只增加已初始化 slot 计数后才承担销毁义务，已有
  release 路径按实际已创建数量回收。新冷限额测试覆盖部分组装失败与恢复。
- DefaultContextOptions 的 allocator/executor 仍为借用，公开头明确有效期必须覆盖派生执行物。
  PreparedOperation、Executable、Invocation、Queue/Completion 的布局和寿命没有新机制。
- hot path 的资源规划、packing、scratch 和 prepared replay 沿用原实现。普通 consumer 实际
  包含 Native RMSNorm 和 oneDNN gated FFN region；独立 Invocation、变化输入、嵌套 bulk、
  worker 排空和关闭均在 consumer 内检查，不能由 mock 调度器替代。

## 固定选择与能力边界

- Direct 工厂装配 oneDNN leaf 后接 Native。已有 select_route 只接 dense row-major rank-2 f32
  MatMul；shape 非法报错，零维和不适用格式明确拒绝。实际任务数为一且 rows < 16 时保留
  Native；有外部 executor 时仍按已有 N tiles 和真实并发度判定。
- Graph 工厂先装配完整 oneDNN region，再接 leaf。region 保留内部融合/packing；额外中间
  consumer、已发布中间结果及控制依赖边界仍由已有匹配器拒绝。未改共同 Graph planner。
- 此轮不改 Native 19-family 的数值、dtype/layout、alias 或内核；测试中的专家 Native 基准
  明确构建 Native Context，避免普通默认值变化偷偷改变 oracle/预算测试。CPU Q4 仍只有 schema。
- 编译条件裁去缺席实现；CPU OFF 由产品图移除新 route roots。实际 installed 头/target 隔离
  仍须由对应验证签字，不能仅靠本次源码判断记 PASS。

没有发现需重开 Mem、Sched 或共同执行机制的新反例。上述结构不预言实际完整片段性能；
默认规则最终仍受本卡的完整成本 P1 约束。
