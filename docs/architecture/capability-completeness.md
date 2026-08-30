# 基础库能力完整性决策

> 决策状态：`ACCEPTED`，2026-08-28。本文定义 HYLE 所说的“功能完整”，并约束所有新模块、重构和
> 已有模块复检。施工步骤见
> [模块重构与收口协议](../developer/review/module-closure-methodology.md)。

## 问题

HYLE 是基础加速库。只实现当前仓库已经调用的最小集合，会在真实消费者出现时反复补洞；把 oneTBB、
Folly、Abseil、Seastar 等库的公开 API 求并集，又会复制业务策略、历史包袱和 benchmark-only 形状。
另一个更隐蔽的问题是：代码里已经有某项能力，不代表这项能力完整。只有一个构造入口、单向转换、
无法正常关闭的容器，或只覆盖 benchmark 生命周期的类型，都只能算部分实现。

因此，功能审计必须同时回答两个方向的问题：

1. 从真实用户任务和成熟问题域向下看，应该有的能力是否都有自然路线；
2. 从现有公开类型和函数向上看，它声称提供的能力是否覆盖完整的常用旅程，而不是只“有一个 API”。

## 决策

一个模块的功能完整，指该问题域内**所有已准入的常用用户旅程都有明确、自然、可验证的完成路线**。
路线可以由本模块直接实现、由少量 HYLE 原语自然组合、由受审成熟依赖承载，或明确交给相邻/上层
owner。它不等于 HYLE 自己实现所有功能，也不等于与任一参考库 API 对齐。

### 能力准入来源

能力先进入审计，再决定是否成为本模块的 `MUST`。以下来源按证据记录，不互相冒充：

1. **定义性能力**：缺少后，这个模块就不能自然完成其命名问题域的基本任务；由语言/平台标准、
   教科书式算法边界或成熟实现共同证明。
2. **当前真实消费者**：仓库内或已命名外部产品存在具体调用旅程、缺陷或性能需求。
3. **已声明北极星消费者**：当前以量化推理引擎为首；必须写出具体对象、生命周期和操作序列，不能只写
   “未来可能需要”。工程计算场景承担防止 API 被单一模型过拟合的反例职责。
4. **反复出现的通用基础需求**：至少两个独立成熟来源，或一个语言/平台标准加一个成熟来源，都提供
   同一用户任务；其边界稳定、owner 明确，且不会让默认热路径支付无关 metadata、allocation、锁或
   动态分发成本。
5. **现有公开承诺**：已经存在的公开入口必须反向审计，但存在本身不是保留理由；它可以被判为完整、
   部分、错误归属或无需求后删除。

参考库只负责发现能力族、成熟形状、算法档位和负空间。单库独有 feature、代码量、symbol 数、示例数
或 benchmark case 数不能独立形成准入理由。

### 先审能力族，再审函数

每轮 `AUDITED` 前必须形成一张自顶向下的 capability-family map。能力族按用户任务命名，例如
“checked OS timestamp interchange”，而不是按现有函数名命名。每个能力族列出：

- 正常用户旅程和证据来源；
- 必须覆盖的语义变体，例如 monotonic/realtime、checked/lossy、owner/view；
- 创建/取得、正常操作、组合/互操作、更新/取消/关闭/销毁等生命周期；
- zero/empty/min/max/overflow/OOM/system-failure 等边界；
- ownership、borrow invalidation、线程安全、重入和进展保证；
- 时间/空间复杂度，以及常态 allocation/copy/syscall/lock/metadata；
- source/static/shared 交付、可发现性、默认值和用户陷阱。

不适用的维度必须说明原因。随后再做一次自底向上的 public-surface trace，把每个现有公开入口映射回
能力族和具体旅程。这样既能发现完全缺失的能力，也能发现“已经做了，但只做了一半”。

### 覆盖状态与施工动作分离

当前覆盖状态只使用：

- `COMPLETE`：所有已准入旅程都有自然路线，关键语义和边界已有验证计划；
- `PARTIAL`：能力族存在，但至少一个已准入旅程、方向、生命周期或边界不完整；
- `MISSING`：已准入能力没有可用路线；
- `ROUTED`：有意由标准库、受审依赖或具名相邻/上层模块完整承载，用户路线明确；
- `OUT_OF_SCOPE`：该任务属于业务/执行策略或别的问题域，拒绝下沉。

覆盖状态与 `KEEP/FIX/BORROW/DEFER/REJECT` 是两个轴。`KEEP` 只能对应 `COMPLETE` 或已经证明完整的
`ROUTED`；“代码已经存在”不能写成 `KEEP`。`PARTIAL/MISSING` 的 `MUST` 必须 `FIX`、`BORROW`、
改为完整 `ROUTED`，否则本轮进入 `BLOCKED`。`GENERAL` 可以 `DEFER/REJECT`，但必须留下 owner、触发
条件和负空间；一旦通过上述准入规则，应提升为 `MUST`，不能用 `GENERAL` 长期躲避基础需求。

### 完整路线的最低要求

一条路线只有同时满足以下条件才算完整：

1. 用户能从 public C++ API 或用户指南自然发现并完成任务，不必模拟 benchmark 或读取内部源码；
2. 语义、精度、失败、ownership、lifetime 和 concurrency 合同明确；
3. 常见边界不会静默改变结果、泄漏资源或要求未记录的调用顺序；
4. 算法复杂度与成熟档位一致，默认路径没有无关成本；热点再接受真实 consumer/replay 性能门；
5. 依赖方向和 source/static/shared 交付成立；
6. correctness、用户指南和 negative space 能共同验证“支持什么、不支持什么”。

标准库或成熟依赖已经给出更自然路线时，HYLE 文档应直接指向组合方式，不为“功能数量”再造包装。
相反，若组合需要用户重复写易错的 overflow、lifetime、dispatch 或平台分支，这通常说明基础能力仍不完整。

### 成熟依赖、适配边界与唯一决策权威

`BORROW` 不是把产品责任交给依赖。每项借用能力必须分别写清四种 owner：

- **semantic owner**：向 HYLE 用户承诺语义、ownership、failure 和 concurrency 的模块；
- **mechanism/artifact owner**：实际提供算法、源码或二进制的 HYLE 模块或第三方项目；
- **decision authority**：运行时对同一件事作选择的唯一实现，例如选择可执行 SIMD target；
- **delivery owner**：负责组件边、锁定、安装、离线、许可证和装载的构建/package 合同。

同一运行时决策只能有一个 authority。两个模块可以观察不同层次的事实，但不能各自根据近似相同的
探测结果控制同一 hot path。例如 platform 可以拥有 OS、CPU、topology、NUMA 和 affinity 等通用
机器事实；若 SIMD 借用 Highway，具体可执行 target 与 kernel dispatch 由 Highway/HYLE SIMD adapter
唯一裁决，其他模块不能再读取一套 `CpuFeatures` 自行选择同一 kernel。面向诊断的重复展示只有在来源、
时点和可能不一致的语义写明后才可保留，不能成为第二条决策路径。

第三方类型、allocator handle、错误枚举和配置宏不得无意泄漏为 HYLE 的通用公开合同。HYLE 可以把
算法机制交给成熟依赖，但仍拥有稳定、自然的 C++ 任务形状和完整的依赖交付路线；依赖为可选项或目标
平台无法提供时，还必须有正确 fallback 或明确的 unsupported 路线。若公开接口必须暴露第三方类型，
应把它登记为有意的 interop capability，而不是适配层偶然泄漏。

第三方升级是一项有身份的产品输入变化，不能自动继承旧证书。升级前后按影响面裁决：

1. 只改变获取元数据且不改变源码、产物、语义、codegen、许可证或交付时，执行 build/package 维护门；
2. 改变 owner 模块使用的源码、ABI、算法、默认策略或 hot-path codegen 时，定向重开该能力族，绑定
   新 baseline，并回放适用 correctness、consumer、delivery 和性能门；
3. 影响公开合同、跨模块 ownership 或运行时决策时，同时复检直接消费者，但不借升级无界重审全库；
4. 新版本未通过时可以继续锁定旧版；“最新版”本身不是能力需求或性能结论。

因此，mimalloc 一类依赖可以拥有 allocator 核心实现，但 `mem` 仍拥有 HYLE allocation/PMR/pool 的
语义与生命周期；mimalloc 版本改变后只重签受影响的 host-memory 路线和交付闭包，不重新发明整个
Mem API，也不能沿用旧版性能证据。

## 收口判据

模块只有在以下事实同时成立时，才可声明功能面收口：

- capability-family map 覆盖当前消费者、北极星具体旅程、定义性能力、成熟来源差异和全部现有公开面；
- 每个 `MUST` 为 `COMPLETE` 或有完整 `ROUTED` 路线，不存在 `PARTIAL + KEEP`；
- 每个 `GENERAL` 已明确 `DEFER/REJECT` 的 owner、触发条件和负空间；
- 没有 benchmark-only 入口，也没有“有函数但缺反向转换/关闭/边界/文档”的半项能力；
- user 文档能按场景告诉用户选什么，architecture 记录边界原因，implementation 保存本轮证据。

结论绑定本轮范围与已知需求，不承诺未来永不增长。新的真实消费者、标准变化，或强 peer 证明了更好的
复杂度档和可传播收益时，按重开条件复审受影响能力族，而不是重做整个模块。

## 对已有模块和既有实现的适用方式

旧任务写过“功能完整”不能替代本决策所需的 capability-family map。轮到模块复检时，把当前实现当作
待审输入：先验证它覆盖了哪些完整旅程，再决定保留、补齐、路由或删除；不能只寻找全新的类型。依赖、
平台或工具链升级后，只重查可能改变语义、边界、复杂度或交付的能力族及其直接消费者。

本决策也不要求无新事实地同时重开所有 `CLOSED` 模块。只读方法回放若缺少旧轮没有记录的能力证据，
应记为 `METHOD_GAP`；发现真实缺失旅程、错误 owner、半项公开能力或更好的复杂度档时，才形成
`PRODUCT_TRIGGER`。这样既不让旧结论永久免检，也不把标准升级变成全库无限返工。

## 防止审计无限化

完整性不是逐行复刻所有参考库。每个能力族最多预选两个 strong peers；先枚举公共任务和算法档，再只读
支撑裁决所需的实现。每一行一旦拥有需求来源、覆盖状态、owner、动作、验证与负空间，就停止该项调查。
无法在冻结预算内裁决的 material `MUST` 是具名 blocker，不是继续无限研究的许可。
