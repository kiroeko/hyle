# HYLE C++ 用户文档

这里面向使用 HYLE 的 C++23 开发者：选择交付形式、找到正确模块、理解所有权与并发合同，并在
不阅读实现源码的前提下完成正常任务。

## 从这里开始

1. [使用指南](usage-guide.md)：最短接入路径和组件选择。
2. [三种交付形式](distribution.md)：完整源码、headers + static、headers + shared。
3. [调用安全合同](usage-contract.md)：热路径、线程、生命周期和内存边界。
4. [`samples/`](../../samples/README.md)：可编译的 C++ 消费者工程。

HYLE 尚未正式发布，并正按新标准逐模块重审；Core 已完成首个签署，后续模块仍需分别取证。
模块页只描述当前真实 C++ surface；标为 `provisional` 或 `CANDIDATE` 的页面可以用于试验，但不
构成稳定性承诺。当前状态见
[施工状态](../implementation/status.md)。

## 模块指南

- [core](core.md)
- [error](error.md)
- [platform](platform.md)
- [mem](mem.md)
- [time](time.md)
- [obs](obs.md)
- [util](util.md)
- [container_intrusive](container-intrusive.md)：稳定节点、零分配、auto/safe-link 双向链表
- [container](container.md)：完整数据结构模块，并聚合轻量 intrusive target
- [simd](simd-usage.md)
- [io](io.md)

当前没有合格用户页的范围：sched 尚未做 v2 易用性审查；math 的模块边界仍待裁决；compute 与
graph 尚未完成。仓库中的 header、sample 或历史任务只供内部试验，不能替代用户合同。它们轮到
审查时必须先补正常用户 journey，再进入 `CLOSED`。

当前产品没有面向原生 C 程序的独立 API，也没有可承诺的 Java、.NET、Python 或 Rust FFI。
source/static/shared 都是 C++ 交付形式；未来只有在出现已命名外语消费者后，才按
[Native FFI 准入策略](../architecture/native-ffi-policy.md)增加窄而真实的运行时边界。
