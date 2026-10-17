# 从哪里开始

第一次使用不需要通读文档。按下面的路径完成一个任务，再按需查细节。

## 第一次构建和接入

1. [初始化与构建](developer/setup/README.md)：检查基础工具，选择后端，运行初始化、构建和测试。
2. [C++ 起步](user/usage-guide.md)：把 Hyle 接入自己的工程，选择模块，找到第一份可运行示例。
3. 运行后再按任务查[模块手册](user/README.md)。例如直接执行算子查 Compute，重复执行计算图查 Graph。

安装 SDK、静态/共享库、ABI 与部署资源的区别，只在[交付与集成](user/distribution.md)说明。
跨模块共用的所有权、并发和错误规则，只在[调用合同](user/usage-contract.md)说明。
具体 API 签名、参数和局部限制以公开头为准；文档中的示例帮助选择与组合接口，不复制一套 API 清单。

## 修改项目

[开发与维护](developer/README.md)提供相关构建测试、依赖、生成物、排障和性能测量的入口。
工作区与验证机器约定只见 [AGENTS.md](../AGENTS.md)；当前工作见[八项主线](developer/current-work.md)。

## 需要解释“为什么”时

- [architecture](architecture/README.md)：模块边界、资源与执行模型，以及无法从普通用法推导的设计取舍。
- [implementation](implementation/README.md)：仍有价值的实施背景和原始材料入口，不作为现行操作步骤。
- [evidence](../evidence/README.md)：历史源码、日志、数据与复现实验资产，不是产品能力表。

四层按职责分开：user 负责怎么用，developer 负责怎么维护，architecture 负责为什么，
implementation 负责有价值的实施背景。同一事实只在所属位置维护；其它页面链接过去。
