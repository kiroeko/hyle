# HYLE 范围

## HYLE 是什么

HYLE 是面向 AI 推理、异构渲染/资源流送、低延迟服务与工程计算的 C++23 基础加速库。它把常用的
硬件/OS 机制、高性能数据结构、
内存、SIMD、调度、I/O 与异构执行放在可组合的模块合同下，让消费方专注模型或领域逻辑。

本地量化推理引擎是首个可执行需求锚，但不是唯一产品定义；渲染/资源流送、批处理数值管线和低延迟
服务共同防止基础 API 被某个模型或框架过拟合。HYLE 本身不是推理引擎、渲染引擎，也不是面向任意
业务的应用框架。

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
case 数都不是完整度指标。完整判据与能力准入、已有能力反向审计规则见
[基础库能力完整性决策](capability-completeness.md)。特别地，“已经有一个公开 API”不等于该能力完整；
单向互操作、缺少正常关闭/销毁路径或只覆盖 benchmark 生命周期都必须标为 `PARTIAL`。

### 可执行的产品边界

一项新能力只有同时通过下面七道门，才进入 HYLE 产品范围：

1. **语义通用：**去掉 model、token、route、tenant、material 等领域名后，仍能用 ownership、failure、
   concurrency、cancel、backpressure、teardown 和复杂度完整描述；
2. **需求可复核：**它是定义性能力，或至少由“一个具名真实消费者 + 一个成熟实现”证明，或在两个
   独立消费者/纵切中重复出现；库名、LOC 和 API 数量本身不是证据；
3. **HYLE 有集成价值：**HYLE 能减少 material allocation/copy/syscall/同步成本，统一已有
   allocator/scheduler/IO/error 合同，或消除每个消费者都会重复手写的易错状态机；
4. **归属成立：**semantic owner、实现载体、决策 authority 和 delivery owner 唯一，能够落进无环
   factor DAG；当前依赖图不能反过来否决本来应该存在的能力；
5. **承载可持续：**明确裁决为 `SELF`、`BORROW`、`ROUTED` 或 `APPLICATION`。成熟实现更合适时不为
   symbol 数自造一套，HYLE 只拥有确有价值的 adapter；
6. **成本隔离：**可选能力不给默认热路径增加无关 metadata、allocation、锁、syscall 或动态分发；
7. **旅程自然：**C++ 用户能从公开 API 或指南完成创建、正常操作、取消/关闭和销毁，并能在 source、
   installed static、installed shared 三种交付中得到同一语义。

这里的最小单位不必是一条函数或一次 syscall。若拆开一个通用组合，会迫使每个消费者重复实现
child reap、取消后 drain、bounded backpressure、残留字节交接或资源树终止等竞态，它就是应独立审计的
**机制闭包**。反过来，listener 数量、route/middleware、鉴权、JSON schema、模型 batching、材质或部署
策略仍是上层产品责任。

`ROUTED` 只有在文档命名实际库/API、ownership 与线程模型、错误边界及构建接入后才算完整；“用户自己
实现”不是路线。未通过任一道门的想法只能成为带晋级条件的 `WATCH` 或明确的上层责任，不能新增产品
API、依赖或 benchmark。

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

新增能力先通过上面的七道产品边界，再按
[能力展开与组件质因数分解](capability-factorization.md)确定 factor 和依赖；只有随后建立的模块
closure card 才授权实现与验证。项目级扫描发现 `PRODUCT_TRIGGER` 不自动修改模块状态，也不把未经
SCOPED/AUDITED 的候选变成施工清单。

项目可以因为真实需求增长，但不以达到某个代码行数为目标。当前施工状态只见
[implementation/status](../implementation/status.md)。
