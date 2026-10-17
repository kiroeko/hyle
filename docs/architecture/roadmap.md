# 分层与组件关系

精确组件和依赖见 [`HyleComponents.cmake`](../../cmake/HyleComponents.cmake)。
当前工作安排见[工作主线](../developer/current-work.md)。

## 分层

- Core 提供极小的编译与布局原语，Error 统一失败，Platform 提供 OS 和硬件事实。
- Mem、Time、Obs、Util、Container、SIMD 和 Math 提供可组合基础能力。
- Sched 与 IO 组合任务、取消、唤醒、文件、网络和协议机制。
- Compute 承载设备、资源和算子执行；Graph 承载逻辑 DAG、编译、放置、搬运和可复用执行。
- 模型格式、tokenizer、采样、KV 策略和 serving 由引擎拥有。

底层能力避免重复线程池、allocator、时钟、事件循环和错误模型。
较高层组合拥有完整的取消、背压和关闭语义，不把业务策略下沉。

## Compute 与 Graph

```text
direct consumer -> Compute operation/kernel -> backend/provider
Graph IR -> compiler -> executable -> Compute regions
```

Compute 不依赖 Graph，单算子用户不取得图编译器。
Graph 拥有节点、依赖、生命周期、copy/sync、alias 和 workspace；Provider 拥有片段内计算。
不可变可共享制品与每次执行的可变状态分开。

CPU、CUDA、MLX/Metal、Vulkan 保留原生资源和执行差异，共享错误、资源寿命和计算合同。
CPU＋GPU 通过显式放置、迁移、依赖和完成机制组合；不暗中把失败的 GPU 计算重跑到 CPU。
共享内存也需要同步。取消不意味着 GPU 工作已经退休，关闭必须排空已接纳工作。

设备能力和性能规则在准备期匹配，执行期不现场计时选择 Provider。
普通消费者不需要手工排序实现或提供性能 profile。
详见 [Compute](compute-design.md)、[Graph](graph-design.md)、[后端路线](compute-routes.md)。

## 依赖与交付

内部组件用于隔离依赖，用户选择公开产品。依赖方向必须无环，公开依赖保持必要且可解释。
第三方算法不等于第二套 HYLE 语义模块，私有运行时也不能泄漏到普通入口。
Apple 路线不应拉入 CUDA SDK，CPU 路线不应需要 GPU；静态和共享安装包保持相同计算语义。

修改时直接检查受影响组件、公开头和真实消费者，并运行相关构建与测试。
不再使用重开、审批、固定调查预算或收口状态控制开发。
