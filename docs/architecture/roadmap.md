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
  和可复用 Executable。one-op 只是可独立签发的纵切，后续能力按 registry 当前卡推进；被退役实现从
  Git 追溯，工作树现有实现按需求决定复用/改写/删除，不把旧类或测试当新设计约束；
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

Compute 永远不依赖 Graph；只运行单 op、kernel 或 Vulkan compute pass 的用户不取得图编译器。
Graph 拥有逻辑图、region 边界、copy/sync、liveness 与 alias；每条路线可以独立实现 compiler 与原生
执行器，Provider 拥有片段内的实际计算。不可变可共享的制品与每次运行的可变状态分开，不把旧类名、
对象布局或通用 segment loop 当重构前提。详细合同见 [Compute](compute-design.md) 与 [Graph](graph-design.md)。

推理采用 CPU、CUDA、MLX/Metal、Vulkan 四条独立计算路线，模型级 CPU+GPU 联合计算不属于 HYLE
范围，不保留为延期任务。host 提交/传输/同步及同 GPU 多 Provider 组合仍保留。共同的是图语义、
资源/完成责任与冷选路合同，不强制四套原生实现复用同一执行器；构建期裁剪、准备期绑定、执行期固定。

谁计算、数据由谁持有、沿哪条时间线执行必须分清，但不是要求普通用户拼三套 ID 容器。真实资源依赖
先于 workspace/alias；共享内存不取消同步。Provider 可以借用原生资源，也可以自持 runtime；它的
replay 是 prepared 还是 managed 单独声明，不能从 runtime owner 推导。直接 Compute 同样可使用
独立 runtime，不为单个计算强制依赖 Graph。

原生调用、完整 Provider region、Provider 支持的内扩展都是合法接法。优先保留成熟完整实现，再按
完整性能增量替换；MLX 内的 HYLE Metal kernel 不强制切出一个额外 runtime。ggml-CUDA、ggml-Metal、
ggml-Vulkan 与 MLX-Metal 按各自 owner 分别准入，ggml 没有因此变成 MLX Provider。Provider 生成的
输出按真实 owner 导出，不为旧 out-buffer 形状强制复制。三种 C++ 交付都须有可用的 custom execution。

构建隔离一直到 vendor backend，不让 Apple 产品固定拉入 CUDA SDK 或 CPU/oneDNN 算子包。Apple
和 NVIDIA 是主优化方向；纯 CPU 保留完整路线；Vulkan/MoltenVK 是正式 portability 路线，不冒充
Apple 原生或未测硬件的性能资格。HIP/ROCm 当前不实现。细节只在[四路线设计](compute-routes.md)维护。

基础完成须分别通过路线隔离、真实计算/Provider 组合、资源/同步/错误寿命、三交付与公平完整性能。
有限架构实验不签这个 gate，也不锁定旧公共对象。执行顺序见
[四路线基础重构计划](../implementation/tasks/compute/four-route-foundation-refactor.md)，依据与未证明项见
[六项架构复核](../implementation/tasks/compute/six-point-architecture-review.md)；历史输入见
[研究索引](../implementation/tasks/compute/compute-graph-route-unification-audit.md)。本页不维护进度百分比
或第二份工作队列，研究完成也不自动恢复已暂停的大 goal。

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
card。Compute/Graph 的共同合同须接受四条原生路线的有限纵切检验，不能先把 CPU 对象写死再强迫
其它 runtime 适配；每条路线先完成真实计算/资源旅程，再扩算子目录，不先造大全套再尝试拼接。

`CLOSED` 模块只因可复现 correctness/security/lifetime 缺陷、真实消费者证明的常用能力缺口、相关
依赖升级、公开合同/算法/交付变化，或合格同合同证据的 material 回归而重开。形成 trigger 和新 record
后走 `CLOSED -> SCOPED`；只读观察、代码行数、命名偏好或单次微基准不改变状态。

模块每轮的具体能力表、验证、预算和停止规则见
[收口协议](../developer/review/module-closure-methodology.md)。项目规模不是完成度代理；自然 API、正确
依赖、真实消费者可用和可复现证据比达到某个代码行数更重要。
