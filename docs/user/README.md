# HYLE 用户手册

先选整体入口，跑通最小程序；局部模块和后端细节按需要查。当前引擎以`hyle`命令行交付，
基础库提供相同C++23 API的源码、静态SDK和动态SDK；尚无引擎应用API、产品C API或跨语言binding。

## 整体使用：先读这里

| 目的 | 入口 | 读完能做什么 |
|---|---|---|
| 运行模型 | [推理引擎](inference.md) | 构建、准备本地输入并运行普通／MTP生成，了解现有模型限制。 |
| 在自己的C++程序中用基础库 | [使用指南](usage-guide.md) | 选产品、构建安装、写最小消费者，再决定是否需要Compute／Graph。 |
| 制作或部署SDK | [交付与集成](distribution.md) | 选择源码／静态／动态形式，处理版本、运行库与后端资源。 |
| 管理异步和并发调用 | [调用安全边界](usage-contract.md) | 明确借用、完成、fiber、分配器和ISA的责任。 |

完整可编译工程见[样例](../../samples/README.md)。使用入口不要求先通读架构或实验记录。

## 计算子系统：按工作方式选择

| 需求 | 指南 |
|---|---|
| 准备一个算子，变化输入后重复提交 | [Compute](compute.md) |
| 编译多节点计算，规划存储并重复执行 | [Graph](graph.md) |

这两页的起步示例是主路径，算子、编码和高级资源章节是按需参考。

## 按需专题：有对应需求再读

| 局部问题 | 专题 |
|---|---|
| Apple或Vulkan原生接口、设备条件及资源 | [Apple MLX/Metal](apple-compute.md)、[Vulkan](vulkan-compute.md) |
| 多请求attention、KV编码、页更新与有效区 | [批次attention](batch-attention.md) |
| 只读快照、dirty clone和成功发布状态 | [状态版本](state-versions.md) |
| 模型编码及特殊数学 | [Compute模型格式入口](compute.md#cpu-与模型格式接入)；整模型范围查[模型覆盖](../implementation/model-operators.md)。 |

## 局部模块参考

| 领域 | 指南 |
|---|---|
| 基础与错误 | [core](core.md)、[error](error.md)、[math](math.md) |
| 系统、内存与时间 | [platform](platform.md)、[mem](mem.md)、[time](time.md) |
| 数据结构与工具 | [container](container.md)、[侵入式容器](container-intrusive.md)、[util](util.md) |
| 向量运算 | [simd](simd-usage.md) |
| 执行与通信 | [sched](sched.md)、[io](io.md)、[process](process.md) |
| 观测 | [obs](obs.md) |

接口签名以公开头为准，支持范围取决于具体系统、设备、数值和布局条件。
组件通过不能代替整模型或任意编码可用。缩写见[术语表](../glossary.md)。
