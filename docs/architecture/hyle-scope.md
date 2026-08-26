# HYLE 范围

## HYLE 是什么

HYLE 是面向 AI 推理与工程计算的 C++23 基础加速库。它把常用的硬件/OS 机制、高性能数据结构、
内存、SIMD、调度、I/O 与异构执行放在可组合的模块合同下，让消费方专注模型或领域逻辑。

首个需求锚是本地量化推理引擎；工程计算 workload 用于防止基础 API 被某个模型过拟合。HYLE
本身不是推理引擎，也不是面向任意业务的应用框架。

## 价值范围

HYLE 关注四类可以重复利用的加速：

1. 更好的算法与数据结构复杂度；
2. 减少 allocation、copy、syscall、同步和调度开销；
3. 贴合 cache、NUMA、SIMD、CPU/GPU backend 的数据与执行形状；
4. 让内存、错误、时间、取消、completion 和观测在模块之间正确组合。

性能与通用性同时是约束。一个模型专用 fused kernel 可以属于 compute 或推理引擎，但不能为了
它把 token/head/KV 语义塞进基础容器；反过来，通用 API 也不能用不必要的间接和防御抹平硬件
能力。

## 模块范围

| 层 | 模块 | 责任 |
|---|---|---|
| 最小地基 | core, error, util-core | 编译/布局 primitive、统一失败、极小无状态工具 |
| OS 与资源 | platform, time, mem, obs | 平台机制、时钟/定时、host memory、可观测机制 |
| 数据与计算 | container_intrusive, container, util, simd, math | 轻量侵入式与完整数据结构、通用 runtime 工具、向量与少量标量补位 |
| 执行与 I/O | sched, io | 任务/纤程/协程、取消/唤醒、文件/网络/protocol 机制 |
| 异构与编排 | compute, graph | device/backend/operator 执行与通用依赖/数据并行机制 |
| 贯穿合同 | build/package | 组件闭包、依赖、feature、三种 C++ 交付与验证 |

精确依赖与重审顺序见 [roadmap](roadmap.md)。模块名不是能力准入理由；每项能力仍需真实任务、
正确 owner 和 public/internal 裁决。

## 能力完整性的边界

作为基础库，HYLE 必须认真覆盖用户常见需求，但覆盖不等于自己实现所有类型。每项能力可以：

- 由 HYLE 直接提供；
- 由少量现有原语自然组合；
- 借成熟依赖并由 HYLE 提供正确 adapter；
- 明确属于 compute、graph、推理引擎或应用；
- 因缺少证据而延后，或因违反范围而拒绝。

多个成熟库反复出现的通用能力必须进入 capability ledger 评估。LOC、公开 symbol 数和 benchmark
case 数都不是完整度指标。

## 当前不属于 HYLE 的产品

- 模型文件解析、tokenizer、sampling、KV/prefix-cache 策略和 serving policy；
- 模型训练框架、自动微分、分布式训练或完整模型 zoo；
- CAD/BIM/CAE 领域对象、几何语义、求解策略和用户界面；
- 云部署、容器编排、数据库、消息系统或通用分布式 runtime；
- 为某个语言维护与 C++ 全量等价的 binding；当前 Native FFI 为 `N/A`；
- 替代操作系统、GPU driver、编译器或所有成熟数值/协议库。

这些能力可以由上层产品使用 HYLE 构建。若真实消费者表明某个可复用机制应该下沉，先通过
语义擦除与模块准入审查，而不是直接扩大 scope。

## 交付边界

HYLE 提供完整源码、headers + static 和 headers + shared 三种 C++ 形状。授权范围与技术交付
分开；特定闭源客户可以依法获得完整源码。C++ shared library 不承诺跨编译器/stdlib 的稳定 ABI。

支持的平台、ISA 和 backend 是逐模块、逐 candidate 的证据范围，不由一张全局愿望表自动承诺。
没有真机或可复现验证时应写未认证，而不是把可编译等同于可支持。

## 扩展准则

新增模块或公共能力必须满足：

1. 有真实消费者任务或成熟库反复证明的通用需求；
2. 不能由现有模块自然覆盖；
3. ownership、failure、concurrency、complexity 和负空间能独立说明；
4. 不造成依赖环或把上层策略下沉；
5. 有 correctness、三种交付、易用性与适用性能验证计划；
6. 有长期 owner，文档和机器合同能阻止边界漂移。

项目可以因为真实需求增长，但不以达到某个代码行数为目标。当前施工状态只见
[implementation/status](../implementation/status.md)。
