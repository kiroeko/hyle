# HYLE 架构方向与组件关系

本文解释长期分层和依赖原则，不复制当前组件清单。机器可执行的 factor、产品 façade、公开/私有边和
`QUEUED` 属性只有一份，位于 [`cmake/HyleComponents.cmake`](../../cmake/HyleComponents.cmake)；当前
施工状态只有一份，位于 [`module-status.json`](../implementation/module-status.json)。

## 分层原则

HYLE 是推理引擎、异构渲染/资源流送、批处理数值管线和低延迟服务下面的 C++23 加速底座。能力应归到
能够完整表达 ownership、lifetime、failure、concurrency 和复杂度的最低合理层，但不能仅为“方便
复用”把上层策略下沉：

- 底层 factor 提供可组合、成本透明、可单独验证的高性能原子机制；
- 较高 factor 或产品 façade 把这些原子组合成自然的完整用户任务，拥有组合后的取消、背压和关闭，
  且不得隐藏第二套线程、事件循环、allocator 或不必要拷贝；
- 模型、协议业务规则和部署策略留给消费方。原子覆盖与复合旅程覆盖分别审计、分别做同层性能比较，
  不能把“零件存在”写成“用户功能已完成”。
- 容易启动 Web/推理服务的 listener/router/middleware/schema/drain 产品面由计划中的独立消费库
  `hyle-service` 组合；HYLE 只下沉领域无关、生命周期完整的机制闭包，依赖永远从 service 指向 HYLE。

- Core 只承载极小、稳定、跨模块必要的编译和布局 primitive；
- Error 统一可恢复失败，Platform 统一 OS/硬件事实与机制；
- Mem、Time、Obs、Util、Container、SIMD/Math 提供可组合基础能力；
- Sched 与 IO 组合底层机制；
- Compute 承载 device/backend/provider 与直接 op/kernel 执行；Graph 承载逻辑计算 DAG、编译、partition
  和可复用 Executable。one-op 只是可独立签发的纵切，后续能力按 registry 当前卡推进；旧实现只存在于
  Git 历史，不作现行需求或实现输入；
- 模型格式、tokenizer、KV/批处理和 serving policy 属推理引擎；材质、场景、渲染算法和窗口/present
  policy 属渲染产品；通用资源 ownership、异构 copy/completion 与调度机制仍须在正确 HYLE owner 审计；
- source、static、shared 都是 C++ 交付，Native FFI 只由真实外语消费者触发。

全库能力和边界变化先按[能力完整性](capability-completeness.md)与
[质因数分解](capability-factorization.md)确定自然用户任务，再改组件图；不能用现有目录或依赖反证一个
能力“不该存在”。

## 内部 factor 与用户产品

内部 factor 是依赖、构建、测试和状态 owner；用户选择的是较少、稳定的问题域 façade。façade 没有
自己的源码、状态或人工依赖表，只从中央声明投影一个或多个 factor root，再解析传递闭包。因此
`container_intrusive`、`timer_queue`、`platform_fs`、`mem_vm` 等小 factor 可以防环和减少内部闭包，
用户仍只需理解 Container、Time、Platform、Mem 等产品。`graph_plan` 当前只是 Graph 内部的热执行源码
边界，没有独立 component identity；用户取得的是 `graph::Executable/Invocation`，不会多出一个含义
模糊的 Plan 产品。只有以后形成独立依赖、测试 owner 和 consumer 才依法拆 factor。

中央声明同时投影：

1. factor 身份及 `PUBLIC/PRIVATE/BUILD_ONLY` 边；
2. target 构造的稳定拓扑序；
3. 用户产品 façade 与安装闭包；
4. 公开头 owner、选择器和验证图。

模块自己的 `CMakeLists.txt` 只列源文件和所属外部依赖，不再手写另一份内部 link graph。未知边、依赖
`QUEUED`、环、遗漏头 owner 和不完整安装闭包必须在 configure/合同测试中失败。新增 identity 通常只
追加；合法更名要有一对一迁移证据，不能为整理观感重写历史身份。

Highway、mimalloc、OpenSSL 等是模块拥有的外部实现，不是 HYLE factor。多个 factor 可以共享一个
build-only 外部 target；除非公开类型或链接合同泄露该依赖，否则不能据此制造内部模块边。

### Compute 与 Graph 的固定关系

Compute 和 Graph 是两个公开产品，不是三套平级 runtime：

```text
direct consumer -> Compute op/kernel -> backend/provider
Graph IR -> Graph compiler -> internal graph plan -> Compute compiled regions
```

Compute 永远不依赖 Graph；只运行单 op、kernel 或 Vulkan compute pass 的用户不取得图编译器。Graph
依赖 Compute 的 operation/provider 合同，但 provider 选择、全图 placement、copy/sync、liveness 与 alias
由 Graph compile 决定。编译结果公开为不可变可共享的 `graph::Executable`，每次运行状态放在独占的
`graph::Invocation`。详细合同见 [Compute 设计](compute-design.md)与 [Graph 设计](graph-design.md)。

推理采用 CPU、CUDA、MLX/Metal、Vulkan 四条独立计算路线，模型级 CPU+GPU 联合计算不属于 HYLE
范围，不保留为延期任务。host 提交/传输/同步及同 GPU 多 Provider 组合仍保留。共同的是图语义、
资源/完成责任与冷选路合同，不强制四套原生实现复用同一执行器；构建期裁剪、准备期绑定、执行期固定。

异构资源固定分成 execution target、storage domain/resource 和 execution lane 三条轴：node placement
选 target，值/workspace 选 storage，命令按 target-bound lane/Queue 排序；同一 target/storage 组合可以有
多条 compute/transfer/communication lane，capture 是 replay mode 而不是 lane。多 lane 的 happens-before
先于最终 workspace/alias 规划。`execution lane` 是 Compute core 与 Graph plan 的共同语义，不单独制造
用户产品或 factor。Provider 则
明确分为使用 HYLE Queue/resource 的 shared-runtime 路线，以及自有 allocator/stream/context、通过显式
boundary 与 Completion bridge 接入的 delegated-runtime island；每个 delegated provider 仍是独立 factor。

runtime ownership 与 replay 成本是两条轴：prepared 要求完整路线无热态建图/分配，managed 如实披露
Provider 内部执行准备。HYLE 自己的计划始终预绑定、无热态搜索/JIT；普通用户可选择 prepared-only。
typed custom execution 必须能从三种 C++ 交付接入，不以私有 Provider SPI 或仅有 schema identity 代替。
有限派生 shape、资源副作用域、跨 variant 内存共享与有界优化的合同分别由 Compute/Graph 设计持有，
不为这六项修正创建新公共产品、重复 runtime 或无真实依赖理由的 factor。

Provider 是 HYLE Operation/region 到实际制品的冷路径翻译器，不是第二套公共算子图。叶函数库产出
prepared op；自己 host 图的库产出 provider-native compiled region，Graph 仍是外层 partition/placement
owner。有公开 lowering/Primitive 扩展的 runtime 也可在编译前纳入 HYLE kernel，形成同 runtime 的完整
region 候选；不强制拆图或修改黑盒内部。Provider 生成输出按真实 owner 导出，不强制 HYLE 影子复制。
把第三方加入依赖锁或链接制品不算获得能力，adapter factor 必须实际完成翻译、资源和 Completion
合同。ggml 的 CPU、CUDA、Metal、Vulkan carrier 因 runtime owner 不同分别审计；当前 registry 中未带
backend 后缀的 `compute_provider_ggml` 只代表既有 CPU 审计身份，不能被解释为所有 ggml backend 或
通用 ggml 产品已经交付。

新 Provider 先验证具体成熟 API 的格式、精度、资源和失败边界，再扩大适配。Q4 冷展开到 F32 的
互操作程序不算压缩量化路线；输入临时量化与累加精度也不能混为一谈。接入方式与完整成本约束只在
[Compute 原始需求](compute-design.md#11-原始需求与架构取舍)和 [Graph 选路](graph-design.md#61-workload-intentroute-profile-与有限搜索)
维护；性能资料接口存在不等于真实 best-of 选路已资格化。

Apple Silicon 栈（MLX-first provider + 原生 Metal backend/custom-kernel escape hatch）和 NVIDIA 栈
（CUDA backend + 经审计的 vendor/ggml/HYLE provider）是首要优化目标。Vulkan 是面向广泛消费级设备与
renderer interop 的正式保底路线：Windows/Linux 走原生 Vulkan，Apple 走 MoltenVK portability；它不
是实验，也不替代 Apple 原生性能路线。HIP/ROCm 在没有 AMD 真机、明确消费者与维护预算的情况下
无限期不纳入当前产品，不创建空 backend 或构建开关；Vulkan 不冒充 ROCm 资格。

Compute/Graph 当前先通过一个共同的 **foundation implementation gate**，再扩展平台和算子广度：
leaf/可收缩 node-set region 同池路由与完整成本、exact + bounded/guarded shape domain、CPU one-op/oneDNN
shared region、CUDA 多 lane/exact capture + Q4 MLP、Apple MLX region + 原生 Metal kernel 组合，
以及 Vulkan 最小 SPIR-V/资源同步纵切都必须在同一对象模型上成立。真实 CPU executor、跨运行时失败
回收、性能 profile 消费和各 backend 交付也是基础资格，不全部拖到完整模型阶段。只有该 gate 的
candidate 完成，后续 provider、Vulkan 算子广度、Transformer 和系统算子目录才可横向推进。当前有限工作顺序只维护在
[Compute/Graph 路由审计](../implementation/tasks/compute/compute-graph-route-unification-audit.md)，本页不
复制一张容易过期的编号状态表。
本轮四路线[有限研究已形成重构基线](../implementation/tasks/compute/four-route-architecture-decision.md)，
待用户审视后另开实现卡；两个独立原型不是上述 foundation gate 通过，也不自动恢复旧大 goal。

## 依赖审查

每次模块 review 都检查：

1. 是否依赖了更高层，或无故扩大 `PUBLIC` 闭包；
2. 下游是否绕过已有 owner，私造 error、clock、allocator、CPU probe、queue 等同义机制；
3. 公开头和符号实际需要哪些直接依赖；
4. source、installed static、installed shared 是否都能只选择产品根并得到完整闭包。

较高 factor 需要较低 factor 时登记真实边并复用。较低 factor 看似需要较高 factor 时，依次检查能力
owner 是否放错、公开面是否泄漏机制，以及能否提取满足质因数判据的小 factor；都不成立时才使用
标准库、成熟依赖或有明确 layout/生命周期理由的专用实现。字段形似不等于语义重复，依赖顺序也不能
成为复制通用实现的理由。

底层边发生产品变化时只回审受影响的直接消费者和公开闭包，不顺手重开全库。公共构建变化由
[change-impact selector](../developer/review/change-impact-validation.md)扩展验证；build/package 是横向
合同，不伪装成具有运行时 candidate 的产品 component。

## 推进与停止

实际施工顺序由中央图的拓扑、registry 中的活动状态和真实 blocker 决定，不在 architecture 维护一份
容易过期的编号队列。对一个问题域的典型顺序是：底层 factor → 产品 façade 所需其余 factor → 直接
消费者回放。Compute 与 Graph 的每个新纵切都必须先有真实推理或异构资源任务、能力边界和 closure
card；它们采用“最小 Compute 合同 -> one-op Graph -> multi-op CPU -> 主要设备 -> Vulkan”的
共同纵切，不先各自造完大全套再尝试拼接。

`CLOSED` 模块只因可复现 correctness/security/lifetime 缺陷、真实消费者证明的常用能力缺口、相关
依赖升级、公开合同/算法/交付变化，或合格同合同证据的 material 回归而重开。形成 trigger 和新 record
后走 `CLOSED -> SCOPED`；只读观察、代码行数、命名偏好或单次微基准不改变状态。

模块每轮的具体能力表、验证、预算和停止规则见
[收口协议](../developer/review/module-closure-methodology.md)。项目规模不是完成度代理；自然 API、正确
依赖、真实消费者可用和可复现证据比达到某个代码行数更重要。
