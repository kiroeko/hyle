# 调用安全边界

本页汇总跨模块容易误用的生命周期、并发和编译约束。具体类型仍以公开头和模块指南为准。
`Status`/`Result` 表达可恢复失败；程序不变量、悬空借用和数据竞争不能靠忽略错误变得安全。

## 借用与异步完成

- `StringView`、`span`、`FunctionRef`、`AllocatorRef`、buffer/tensor view 等借用不延长 backing 的寿命。
  保存借用前先确定实际 owner，移动 owner 后检查其地址稳定性约束。
- 异步提交成功只表示工作被接受。等待成功 `Completion` 后再读结果；取消或返回错误不一定表示 GPU
  已停止使用资源。Context、Queue、资源、allocator 和外部 buffer 必须按接口合同活到真实退休。
- 同一可变对象一般需要外部串行化。共享只读计划不等于共享可变执行槽：并发 Compute/Graph 请求使用
  独立 Invocation，并避免未排序的外部输入输出别名。
- 错误后可能保留已经完成的写入；没有明确事务合同的接口不保证回滚。根据接口要求重置或重建失败槽。

具体 Compute/Graph 合同见 [Compute](compute.md#所有权失败和限制)、
[Graph](graph.md#所有权与常见陷阱)。

## 冷准备、实际能力与资源预算

Compute 的 `prepare()` 和 Graph 的 `compile()` 检查具体设备、dtype、编码、形状、布局与资源，
并绑定执行实现；公共描述能够构造或推导 shape，不表示所有后端都可执行。
准备成功后保留计划，为并发调用建立独立执行槽。普通 submit 不重新选 Provider；
改变形状、资源 identity、绑定或影响选择的配置时，按对应接口重新准备。

资源报告的 `unknown` 不是零。caller-owned workspace、计划常驻存储、每槽可变存储与原库/driver
私有内存是不同范围；有限预算只能约束接口明确计入的范围，不能当作整个进程的 RSS/显存硬上限。
需要拒绝未知内存、host 等待或热分配时，在准备期明确提出约束，并处理无可行实现的错误。
`kManaged` 程序可能在重复执行中建立原库 metadata 或临时数据；不能仅由“已编译”推断零开销。

`compute::ResourceBudget::create(options, nominated_resources, max_shared_bytes)` 可让同一预算中
指定的真实资源行共用已知请求字节总额，同时保留每行原限额。把这个预算的副本传给各个分配／
Graph／租约入口，才会共用总额与原 backing 去重；分别创建的预算不会自动合并。
`shared_usage()` 单列该总额的 live／peak／limit。指定 metadata 资源时，控制块与 metadata
分配也计入；未归属资源的 Provider 私有存储仍单列，未经过该预算的原始 Buffer 分配不会自动纳入。
额度随原 backing 及其实际在途使用释放，销毁业务句柄或公开 Completion 不等于退款。

## Fiber 与线程本地状态

可窃取的 Fiber 在 `yield()`、等待或挂起后可能换 OS 线程。不要把 `thread_local` 地址、当前 worker 的
arena/ring、线程身份或其它线程绑定句柄缓存跨越这些操作。栈切换不会切换 OS TLS，优化器也未必把
挂起调用视为“线程身份改变”；仅在源码中再写一次 TLS 访问不足以提供普遍的编译器保证。

优先把请求状态放在 Fiber 栈或显式 owner 中：

```cpp
Ctx request_context;                  // 本任务拥有，不是 thread_local
hyle::sched::Fiber::yield();
request_context.use();                // 随 Fiber 一起迁移
```

确实依赖某个 worker 的对象时，使用 [Sched](sched.md) 的 pinned 提交入口并保持全部唤醒在该 worker。
不要在可迁移 Fiber 中持有 OS 线程锁跨越挂起。普通 worker-stack callback 不得调用 Fiber wait/yield；
阻塞 syscall 和不可切分的长任务交给 `BlockingPool`。

## 分配与释放

`mem::system_allocator()` 使用 mimalloc，支持跨线程释放。自定义 `AllocatorRef` 遵守实际 backing
的并发合同；arena/pool 并不因为套上 ref 就变成线程安全。allocator 必须活过所有使用它的对象。

分配与释放必须配对原 logical size/alignment。`AllocatorRef`、`SystemAllocator` 和 PMR adapter
都是 sized-free 路线，不能省略 size 或把实际请求替换为猜测的 backing 大小。需要过对齐时才增加
alignment。静态链接和 LTO 定制不得把依赖线程状态的分配器实现跨 Fiber 挂起内联；保持实现边界，
或让相关任务固定在同一 worker。

共享固定池有两种容量语义：

| 选择 | 容量可见性 |
|---|---|
| `ConcurrentBlockPool` / `ObjectPool<T>` | completed free 立即对其它线程可用，适合严格容量与 handoff |
| `BlockPoolCache` / `ObjectPoolCache<T>` | 显式 worker cache；本地 free 块在 `publish()` 或析构前可能对其它使用者不可见 |

池 `reset()` 前必须没有 live block，全部 cache 已发布或销毁，并停止并发访问。
共享缓存的吞吐与软容量行为不能代替硬池合同。详见 [Mem](mem.md)。

## ISA 与 LTO

分发给未知 CPU 的程序保留 HYLE 的 baseline/ISA 编译隔离。CPU 探测位于 Platform，SIMD runtime
再从已编入且可执行的实现中分派；相应 CMake target 已处理 IPO/LTO 隔离。不要用全局编译参数
把高 ISA 指令重新带入 baseline 或探测代码。

固定 `-march=native`、exact ISA 或专家 kernel 的程序只能部署到满足其完整指令要求的机器。
`hyle_target_simd(... ISA AVX2)` 施加到整个 target，通常应为该 kernel 建单独编译单元/OBJECT target。
运行时判定与固定目标用法见 [SIMD 指南](simd-usage.md)。
