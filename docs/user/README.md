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
- [container_intrusive](container-intrusive.md)：稳定节点、零分配、auto/safe-link 双向链表
- [container](container.md)：完整数据结构模块；轻量 intrusive target 独立选择
- [simd](simd-usage.md)
- [io](io.md)

未列在上述模块指南中的组件可能还没有经过审查的正常用户 journey；以施工状态和实际用户页为准。
仓库中的 header、sample 或历史任务不能替代用户合同。模块进入 `CLOSED` 前必须补齐适用的正常
用户 journey。

当前产品没有面向原生 C 程序的独立 API，也没有可承诺的 Java、.NET、Python 或 Rust FFI。
早期公开 C schema/bridge 已移除；source/static/shared 都是 C++ 交付形式。未来只有在出现已命名
外语消费者后，才按
[Native FFI 准入策略](../architecture/native-ffi-policy.md)增加窄而真实的运行时边界。
