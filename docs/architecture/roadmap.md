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

- Core 只承载极小、稳定、跨模块必要的编译和布局 primitive；
- Error 统一可恢复失败，Platform 统一 OS/硬件事实与机制；
- Mem、Time、Obs、Util、Container、SIMD/Math 提供可组合基础能力；
- Sched 与 IO 组合底层机制；
- Compute 与 Graph 最终承载计算后端和编排，但在 registry 为 `QUEUED` 时旧实现只作需求输入；
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
用户仍只需理解 Container、Time、Platform、Mem 等产品。

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
消费者回放。Compute 与 Graph 只有在真实推理或异构资源纵切、能力边界和新 closure card 成立后才从
`QUEUED` 进入施工。

`CLOSED` 模块只因可复现 correctness/security/lifetime 缺陷、真实消费者证明的常用能力缺口、相关
依赖升级、公开合同/算法/交付变化，或合格同合同证据的 material 回归而重开。形成 trigger 和新 record
后走 `CLOSED -> SCOPED`；只读观察、代码行数、命名偏好或单次微基准不改变状态。

模块每轮的具体能力表、验证、预算和停止规则见
[收口协议](../developer/review/module-closure-methodology.md)。项目规模不是完成度代理；自然 API、正确
依赖、真实消费者可用和可复现证据比达到某个代码行数更重要。
