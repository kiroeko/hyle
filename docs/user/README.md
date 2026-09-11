# HYLE C++ 用户文档

这里面向使用 HYLE 的 C++23 开发者：选择交付形式、找到正确模块、理解所有权与并发合同，并在
不阅读实现源码的前提下完成正常任务。

## 从这里开始

1. [使用指南](usage-guide.md)：最短接入路径和组件选择。
2. [三种交付形式](distribution.md)：完整源码、headers + static、headers + shared。
3. [调用安全合同](usage-contract.md)：热路径、线程、生命周期和内存边界。
4. [`samples/`](../../samples/README.md)：可编译的 C++ 消费者工程。

HYLE 尚未正式发布，并按统一标准逐模块取证。模块页描述当前真实 C++ surface，但页面存在不构成
稳定性或性能承诺；唯一当前状态见[施工状态](../implementation/status.md)。

## 模块指南

- [core](core.md)
- [error](error.md)
- [platform](platform.md)
- [mem](mem.md)
- [time](time.md)
- [obs](obs.md)
- [util](util.md)
- [container](container.md)：通用数据结构；稳定节点的零分配链表另见
  [Container 侵入式链表指南](container-intrusive.md)
- [math](math.md)：checked integral shape/byte arithmetic、非负 ceil-division 与可表示 multiple round-up
- [simd](simd-usage.md)
- [sched](sched.md)：CPU task/fiber runtime、结构化任务、同步与消息
- [io](io.md)
- [process](process.md)：受管本地 child、stdio、等待、tree termination 与可靠 reap
- [compute](compute.md)：显式 CPU execution、tensor binding、prepared operation 与 typed host kernel
- [graph](graph.md)：逻辑计算图、冷编译、可共享 Executable 与独占 Invocation

未列在上述模块指南中的组件可能还没有经过审查的正常用户 journey；以施工状态和实际用户页为准。
仓库中的 header、sample 或历史任务不能替代用户合同。模块进入 `CLOSED` 前必须补齐适用的正常
用户 journey。

当前产品没有面向原生 C 程序的独立 API，也没有可承诺的 Java、.NET、Python 或 Rust FFI。
早期公开 C schema/bridge 已移除；source/static/shared 都是 C++ 交付形式。未来只有在出现已命名
外语消费者后，才按
[Native FFI 准入策略](../architecture/native-ffi-policy.md)增加窄而真实的运行时边界。
