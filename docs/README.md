# HYLE 文档

HYLE 的产品目标是完整的单机推理引擎，基础库也可独立使用。当前开发重点仍是底层推理支持，
项目内的 Qwen 引擎用于执行真实模型、发现接口与性能问题；通用加载、tokenizer 和服务接口尚未完成。
`engine` 构建 `hyle` 命令行应用；基础库可以按产品独立构建、安装，并由 C++ 程序链接 `hyle::<product>`。

## 从哪里开始

第一次使用先读用户手册，跑通一个入口后再按需要查设计和开发资料。无需先通读全部架构或实验记录。

| 你想做的事 | 先读 |
|---|---|
| 初始化、构建、测试和安装 | [构建指南](developer/setup/README.md) |
| 运行推理引擎 | [推理引擎指南](user/inference.md) |
| 在自己的 C++ 程序中使用基础库 | [使用指南](user/usage-guide.md)，再按模块查[用户手册](user/README.md) |
| 理解推理引擎的长期开发目标 | [两项核心期许](architecture/inference-principles.md) |
| 看目前在哪、接下来做什么 | [推理进度](developer/inference-progress.md) |
| 理解项目和模块分工 | [架构入口](architecture/README.md)，按其中的阅读顺序深入 |
| 了解推理引擎的结构 | [推理引擎概览](architecture/engine-overview.md) |
| 修改代码与排障 | [开发与维护](developer/README.md) |
| 查性能数据 | 推理看 [decode 性能资料](developer/performance/decode.md)，基础模块看[模块性能数据](developer/performance/modules.md) |

文档里的缩写和固定说法（B2、N2／R2、pending、episode 等）见[术语表](glossary.md)。

## 文档分层

| 目录 | 内容 |
|---|---|
| [user](user/README.md) | 用户手册：引擎运行、C++ 集成、各模块用法与调用边界 |
| [developer](developer/README.md) | 开发维护：环境、构建、测试、依赖、排障、性能测量与当前进度 |
| [architecture](architecture/README.md) | 设计：模块职责、关键决策与理由、被放弃的方案 |
| [implementation](implementation/README.md) | 推理工程计划、模型与算子覆盖、测试仍在读取的结构化输入 |
| [evidence](evidence/README.md) | 实验原始数据：源码、日志、CSV/JSON、归档 |

## 维护约定

同一事实只在所属位置维护，其它页面链接引用。API 签名与局部约束以公开头为准；
用户手册讲怎么用，架构文档讲为什么，性能页面汇总数据并链接原始记录。

工作区与验证机器的约定见 [AGENTS.md](../AGENTS.md)。
当前引擎实现范围与后续顺序见[推理工程计划](implementation/inference-roadmap.md)，
正在进行的工作见[推理进度](developer/inference-progress.md)。
