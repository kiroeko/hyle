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

当全库的功能归属或依赖图本身仍有争议时，不能沿现有拓扑逐模块证明自己。必须先按
[能力展开与组件质因数分解](capability-factorization.md)完成所有产品问题域的只读功能超集统计；这一
阶段不以当前依赖拒绝能力。全部功能和自然形状冻结后，才统一确定 owner、组件因子和依赖。单模块方法
用于施工已确定边界的组件，不能替代这次全局展开。

### 能力准入来源

能力先进入审计，再决定是否成为本模块的 `MUST`。以下来源按证据记录，不互相冒充：

1. **定义性能力**：缺少后，这个模块就不能自然完成其命名问题域的基本任务；由语言/平台标准、
   教科书式算法边界或成熟实现共同证明。
2. **当前真实消费者**：仓库内或已命名外部产品存在具体调用旅程、缺陷或性能需求。
3. **已声明消费者纵切组合**：不能只选一个北极星产品。每个纵切都必须写出具体对象、生命周期和操作
   序列，不能只写“未来可能需要”；组合和选取规则见下节。
4. **反复出现的通用基础需求**：至少两个独立成熟来源，或一个语言/平台标准加一个成熟来源，都提供
   同一用户任务；其边界稳定、owner 明确，且不会让默认热路径支付无关 metadata、allocation、锁或
   动态分发成本。
5. **现有公开承诺**：已经存在的公开入口必须反向审计，但存在本身不是保留理由；它可以被判为完整、
   部分、错误归属或无需求后删除。

参考库只负责发现能力族、成熟形状、算法档位和负空间。单库独有 feature、代码量、symbol 数、示例数
或 benchmark case 数不能独立形成准入理由。

### 消费者纵切组合与反偏置

HYLE 的公开能力至少从以下真实产品纵切持续取样，任何一个都不能独占产品定义：

1. **量化推理与 serving**：模型映射、CPU/统一内存/GPU 执行、KV/临时存储、请求调度、流式响应、
   指标与关闭；主要参考 ggml/llama.cpp、ds4、vLLM 和 SGLang。
2. **异构渲染与资源流送**：资产发现/加载、host/device resource、动态 heap 子分配、staging/copy、
   completion/fence、稳定资源 identity、热重载、延迟销毁和交互帧优先级；主要参考 Cycles 类生产渲染
   路径、OpenVDB/NanoVDB 以及受审 GPU compute/resource 库。
3. **批处理数值与数据管线**：大文件、显式 scratch、并行分块、向量化变换、持久化和失败恢复；主要
   参考 oneDNN/oneTBB/OpenVDB 等真实批处理路径。
4. **低延迟异步服务**：监听、TLS/HTTP、连接与请求生命周期、deadline/cancel/backpressure、指标、
   reload 和 drain；主要参考 Seastar、libuv、nginx 及上述 serving 产品的真实宿主路径。

源码嵌入、installed static 和 installed shared 是覆盖所有纵切的交付旅程，不是第五种算法 workload。
一个模块任务只选择与其 owner 直接相关的一个主纵切，并在 API 容易被主纵切特化时增加一个正交纵切；
不机械执行“全部模块 × 全部产品”。项目级能力审计则必须逐条走通完整产品旅程，检查相邻模块之间的
胶水，而不能把一组孤立 API 的存在相加成“消费者可用”。

`ROUTED` 必须命名实际 owner 或成熟依赖，并给出可发现的组合路线；“上层处理”“用户自己写”或一个
尚为 `QUEUED` 的模块名称不能证明当前旅程完整。`OUT_OF_SCOPE` 只适用于模型、材质、协议业务规则等
领域策略；去掉领域名后仍在两个独立纵切重复出现的 ownership、lifetime、failure、concurrency 或
复杂度机制，必须进入通用能力审计，不能借 `OUT_OF_SCOPE` 隐藏。完整路线可以合法止于尚未施工的
具名 owner，但项目级纵切此时应如实记为 `BLOCKED_BY_OWNER`，而不是 `COMPLETE`。

### 原子机制、复合功能与应用策略

能力清单必须显式区分三个层级，不能把它们混成一张“已有 API”列表：

1. **原子机制**是可以独立拥有成本与语义的最小通用操作，例如一次 positional read、一次 map lookup、
   timer arm、cancel edge 或 file sync。它必须可组合、复杂度和隐藏成本透明，并只与同语义原子比较性能。
2. **复合功能**把多个原子组合成反复出现的完整用户任务，并拥有组合后的 ownership、failure、cancel、
   backpressure 与 teardown。例如 atomic file publication、structured task、managed subprocess、file-change
   stream 或 HTTP connection lifecycle。它可以是独立 factor，也可以是 product façade，但必须有自然的
   public journey；不能让每个消费者重新手写同一套易错状态机。
3. **应用策略**决定领域 schema、路由、鉴权、模型/脚本语义、保留周期或业务状态迁移。它留给消费方，
   除非去掉领域名称后仍由至少两个独立纵切证明为稳定通用机制。

原子齐全不证明复合功能完整；复合功能存在也不授权复制下层原子。每个复合能力必须列出其原子分解、
隐藏 allocation/copy/thread/syscall、异常与关闭路径，以及用户为完成正常任务仍需自写的通用 glue。
性能比较也必须同层：原子对原子，完整 journey 对拥有相同完成、持久化和 teardown 合同的完整 journey。
拿 raw syscall 与成熟库 façade 排名，或拿 HYLE 完整 lifecycle 与 peer 的轻量原子排名，都属于苹果对
橘子，不能形成产品 verdict。

发现缺口后不得直接按“缺什么就自写什么”施工，必须分别裁决准入和承载方式：

- `SELF`：HYLE 自己拥有实现。只适用于与既有 allocator/scheduler/IO/cancel/dispatch 紧密耦合、能提供
  明确性能或生命周期价值，并且项目能够长期维护的机制；
- `BORROW`：由受审成熟依赖提供算法或协议核，HYLE 只拥有必要的交付、错误、allocator 或异步组合边界；
- `ROUTED`：标准库或成熟库已经给出自然、完整的用户路线，HYLE 只在指南中说明组合方式，不制造仅改名
  或可能阻碍升级的包装；
- `APPLICATION`：schema、业务 identity、命令模板、保留策略等由最终产品拥有。

裁决至少比较跨纵切复用度、重复 glue 的错误风险、HYLE 能带来的集成/性能收益、成熟实现质量、平台
语义可移植性以及长期安全和升级成本。UUID、JSON、Base64、摘要算法或命令行 option 等名称本身不是
HYLE 准入理由；例如 JSON 可以完整借给 simdjson，UUID 可以继续由随机原子与消费方选定的成熟格式库
承载。反过来，若多个服务都重复实现 child spawn、stdio、异步 wait、cancel/timeout、owned process-tree
termination 与可靠 reap，这个**生命周期组合**值得进入独立能力审计，但不能因此复制某个语言框架的
`ProcessStartInfo` 全部选项。

这里的“原子”不是“一次 syscall”或“只有一个函数”。一个由多个 syscall/数据结构组成、但能完全用
领域无关的 ownership、failure、concurrency、cancel、backpressure、teardown 和成本合同描述的最小
生命周期单元，仍是 HYLE 的**机制闭包**。`FileWatcher`、HTTP framing/connection、atomic file publication
和 structured task 都可以是机制闭包；拆掉它们会迫使每个消费者重写同一套竞态状态机。若组合开始选择
listener 数量、route/middleware、JSON schema、鉴权、证书/模型 reload 策略或部署拓扑，它已经越过基础
库边界。

面向服务的易用层由独立的 `hyle-service` 消费库承载：它单向依赖 HYLE 的 IO/Sched/Error/Obs 等机制，
可按产品需要直接借成熟 JSON、TLS 或配置依赖，并提供 listener、router、request context、streaming、
graceful drain 等完整 server journey。它既是 HYLE 的真实纵切消费者，也不是 HYLE factor DAG 的反向
依赖；不能为了让 `hyle-service` API 更短而把服务策略下沉到 HYLE。

### 先审能力族，再审函数

每轮 `AUDITED` 前必须形成一张自顶向下的 capability-family map。能力族按用户任务命名，例如
“checked OS timestamp interchange”，而不是按现有函数名命名。每个能力族列出：

- 正常用户旅程和证据来源；
- 必须覆盖的语义变体，例如 monotonic/realtime、checked/lossy、owner/view；
- 创建/取得、正常操作、组合/互操作、更新/取消/关闭/销毁等生命周期；
- zero/empty/min/max/overflow/OOM/system-failure 等边界；
- ownership、borrow invalidation、线程安全、重入和进展保证；
- 时间/空间复杂度，以及常态 allocation/copy/syscall/lock/metadata；
- 可复用的现有 HYLE/标准库/compiler/OS/成熟依赖机制、加速设施适用性和唯一 decision authority；
- source/static/shared 交付、可发现性、默认值和用户陷阱。

不适用的维度必须说明原因。随后再做一次自底向上的 public-surface trace，把每个现有公开入口映射回
能力族和具体旅程。这样既能发现完全缺失的能力，也能发现“已经做了，但只做了一半”。

发现超集必须覆盖本轮选定纵切的具名真实实现，以及成熟基础/加速库反复出现的通用能力、按通常工程
划分定义上属于该域的能力和 HYLE 现有公开承诺。推理纵切仍须同时覆盖独立 ggml 的基础能力及
llama.cpp 的真实消费旅程、ds4 的紧凑 CPU/统一内存实现、vLLM 的 runtime/CPU/custom-op，以及
SGLang 的 serving/kernel；其它纵切按上一节选择版本化强来源。每轮只更新或确认实际引用的本地
reference，并记录 clean commit；无 Git 身份的本地快照只能作辅助证据。若某项只存在于 Python、
Triton、GPU/tensor 融合层或渲染领域层，也必须裁决到正确 owner，而不是机械复制成当前模块 API。
这些来源用于发现任务和语义档位，不自动成为同合同性能 peer；依赖、实现成本和 benchmark 不能在
这一阶段充当删项理由。

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
5. 已有同合同且依赖方向合法的机制优先复用；保留本地实现或判定某项加速设施不适用时，有明确的
   语义、ownership、复杂度、常态成本、codegen、平台或交付理由，且没有重复 runtime decision authority；
6. 依赖方向和 source/static/shared 交付成立；
7. correctness、用户指南和 negative space 能共同验证“支持什么、不支持什么”。

标准库或成熟依赖已经给出更自然路线时，HYLE 文档应直接指向组合方式，不为“功能数量”再造包装。
相反，若组合需要用户重复写易错的 overflow、lifetime、dispatch 或平台分支，这通常说明基础能力仍不完整。
“尽量复用”也不是最大化 target/include 数量：最低层 factor、冷失败路径、pointer-chasing、用户 callback
或不同 ownership/lifetime 合同可能没有可用加速路线。关键是逐项比较并留下可审计裁决，而不是默认手写，
也不是为了形式统一制造依赖环、额外 metadata 或动态分派。

自然路线不等于隐藏成本最重的路线。HYLE 默认 API 可以要求 owner affinity、显式容量、borrowed output
或 caller-owned storage，只要这些约束确实支撑正常热路径、能完成完整旅程，并在 public 声明、用户指南
与本轮限制中可发现。更方便但会稳定增加 allocation、copy、锁或 dispatch 的入口只有在真实消费者准入
后才并列提供，并必须拥有独立名称与成本说明；不能仅为缩短样例而让所有用户付费。

### 成熟依赖、适配边界与唯一决策权威

`BORROW` 不是把产品责任交给依赖。每项借用能力必须分别写清四种 owner：

- **semantic owner**：向 HYLE 用户承诺语义、ownership、failure 和 concurrency 的模块；
- **mechanism/artifact owner**：实际提供算法、源码或二进制的 HYLE 模块或第三方项目；
- **decision authority**：运行时对同一件事作选择的唯一实现，例如选择可执行 SIMD target；
- **delivery owner**：负责组件边、锁定、安装、离线、许可证和装载的构建/package 合同。

同一运行时决策只能有一个 authority。两个模块可以观察不同层次的事实，但不能各自根据近似相同的
探测结果控制同一 hot path。例如 platform 可以拥有 OS、CPU、topology、NUMA 和 affinity 等通用
机器事实，但不能给出跨架构“最佳 ISA”。SIMD、Container 或其它 kernel owner 可以分别读取同一个
`CpuFeatureSet`，只为**自己拥有的**生成 TU 声明完整 requirements 并作唯一选择；不能再由第二个模块
根据近似 facts 选择同一 kernel。面向诊断的重复展示只有在来源、时点和可能不一致的语义写明后才可
保留，不能成为第二条决策路径。

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

- capability-family map 覆盖当前消费者、选定产品纵切、必要反偏置纵切、定义性能力、成熟来源差异和
  全部现有公开面；
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

`CONFIRMED/METHOD_GAP/PRODUCT_TRIGGER` 只描述一次具名、定界、只读回放的结果，不是模块生命周期
状态。`CLOSED/BLOCKED/QUEUED` 的唯一真相仍是 `module-status.json`；回放只有形成合法 trigger 并建立
新 closure card 后，才授权改变产品。不得另维护一张“按新方法完成的模块”状态表，也不得把历史
`CLOSED` 自动写成 `CONFIRMED`。推进顺序可以写入 roadmap，但每一步开始前仍需读取 registry。

## 防止审计无限化

完整性不是逐行复刻所有参考库。每个能力族最多预选两个 strong peers；先枚举公共任务和算法档，再只读
支撑裁决所需的实现。每一行一旦拥有需求来源、覆盖状态、owner、动作、验证与负空间，就停止该项调查。
无法在冻结预算内裁决的 material `MUST` 是具名 blocker，不是继续无限研究的许可。
