# HYLE 工程章程

> 本章程规定维护者不能靠一次任务随意改变的工程原则。产品方向见
> [architecture](../../architecture/README.md)，具体代码规则见
> [coding conventions](coding-conventions.md)，模块验收见
> [closure methodology](../review/module-closure-methodology.md)。

## 1. 产品目标

HYLE 是 C++23 基础加速库，服务 AI 推理和工程计算。它提供可组合的内存、数据结构、SIMD、
调度、I/O 与异构计算机制，不拥有模型、业务协议或部署策略。

功能完整意味着常见基础需求都有自然路线，不意味着 HYLE 重写所有成熟库。能力可以直接实现、
组合、借用依赖、上移、延后或拒绝，但必须在 capability ledger 中可追溯。
能力准入与覆盖状态服从
[基础库能力完整性决策](../../architecture/capability-completeness.md)：审计既从用户任务查缺失能力，
也从现有公开面反查每项能力是否覆盖完整生命周期、边界和互操作；已有代码不能自动记为 `KEEP`。

性能目标是正确合同下的优秀复杂度与真实 workload 竞争力，不是赢每个微基准。实现和 API 不得
被 benchmark-only shape 塑造；产品性能改动必须同时有机制证据与 material 消费影响。

## 2. 语言与交付

- 语言基线：C++23。
- 当前工具链 floor：GCC 14+、Clang 18+、MSVC 19.43+；实际 package 记录精确版本与 stdlib。
- CMake target 是构建、依赖、feature 与发布合同的载体。
- 产品交付：完整源码、headers + static、headers + shared。
- C++ shared SDK 不承诺跨编译器、stdlib、runtime 或 build-mode 的稳定二进制 ABI。
- 当前没有原生 C 产品和 Native FFI。未来外语任务按
  [FFI 准入策略](../../architecture/native-ffi-policy.md)单独设计，不做 C++ parity。
- C++ modules 与尚未成熟的标准库能力不能成为必要构建前提；引入时需独立决策和迁移路径。

## 3. 模块与依赖

- 每项能力只有一个 owner module；公共与内部形状分开。
- 依赖必须无环，并指向能够完整表达机制的最低合理层，不把消费策略下沉。
- 中央组件声明同时投影选择闭包和真实 target link edge；模块不得维护第二份依赖表。
- 外部依赖由 owner module 一处登记，再投影到获取、构建、install、offline、license 与 CI。
- 下游不得私造已有的 error、clock、allocator、CPU/topology probe 等同义机制。
- 每轮模块审计都要反查已有 HYLE/标准库/compiler/OS/锁定成熟依赖的同合同机制，并逐项裁决可用的
  SIMD、allocator/pool、Container、reclamation、topology/NUMA、clock/wait、hash/text 等加速设施。
- “尽量复用”不以依赖数量计分；本地实现或不采用候选设施必须有语义、ownership、复杂度、常态成本、
  codegen、平台、依赖方向或交付理由。不得为了形式复用制造环，也不得绕开合法低层能力重复实现。
- 修改底层合同必须回审直接消费者与三种交付闭包。

## 4. API 与实现形状

- 公开 C++ API 以用户任务为中心，优先惯用 RAII、清楚的 owner/view、range/span、strong type、
  `Status/Result` 和合理默认值。
- 模板与 inline shell 可以提供类型安全；较重实现可放 compiled core，避免无界编译膨胀。
- internal fast path 可以特化，但不得要求用户模拟 benchmark 生命周期或拆散一次自然操作。
- 常态 metadata、allocation、indirection、branch 和对象大小都属于设计成本，必须登记。
- 可选 placement、diagnostics、quota 或极端优化显式 opt-in，不让普通用户缴税。

## 5. 失败与生命周期

- 可恢复的运营失败统一用 error 模块；程序不变量破坏使用 fail-fast。
- 默认不从 HYLE API 抛异常。包裹会抛异常的第三方 C++ 库时，只在 adapter 边界捕获并转成
  `Status`；不能吞掉 HYLE 自身 bug。
- ownership、borrow invalidation、move/destroy、callback 保活、cancel、drain 和晚到完成必须能
  从类型或附近文档发现。
- allocator 的创建与释放责任不能跨不兼容 runtime；异步资源的 owner 必须活到所有 completion
  quiescent。

## 6. 性能工程

- 先审复杂度、搬运、allocation、syscall、contention 和向量化，再审少数指令。
- 强 peer 是同问题域成熟实现；赢 `std` 不是在所有领域都足够。
- benchmark 先通过 A/A、known-delta、timer floor、order/placement 与最终 codegen 资格门。
- 真实 consumer 或保真 trace replay 是产品证据；microbench 只负责归因。
- consumer/replay 仍须区分自然产品路线与同语义实现比较；只有逐项通过语义等价门的后者才能签
  peer non-inferiority，前者只回答用户采用当前自然路线会发生什么。
- 性能结论绑定 capability、contract、hardware、delivery 与 candidate，不宣称模块全局 SOTA。
- 未知或仪器不合格时写 `INCONCLUSIVE/DIAGNOSTIC`，不能解释成平局或选择有利姿势。

## 7. Correctness 与验证

覆盖率是卫生地板，不是正确性证明。每项实现按风险组合：

- 正常、空值、容量、溢出、OOM/系统失败、move/destroy；
- property/differential/fuzz、fault injection 和 stress；
- 并发进展、取消、关闭与晚到 completion；
- ASan、UBSan、TSan；真 ISA、NUMA、GPU/backend；
- public-header self-contained 与 component-only build；
- source、installed static、installed shared 的独立 C++ consumer。

不适用的平台或 sanitizer 必须有理由和替代证据；未运行不能写通过。CI 是最小可重复回归门，
不替代本地高成本性能、真硬件和红队验证。

## 8. 第三方与可移植性

- 自造、借用、vendored/ported 与 adapter 边界必须明确；许可证和版本可追溯。
- 第三方类型不得无意泄露到 HYLE 通用公共边界。
- 可分发 CPU binary 使用安全 baseline probe 与明确 dispatch；本机 exact/native 是另一种交付合同。
- OS/硬件差异集中到 platform 或所属 backend，未知平台有正确 fallback 或显式 unsupported。
- 离线构建从锁定依赖或受审源码包复现，不依赖某台维护者机器的缓存。

## 9. 可维护性

- architecture、implementation、developer、user 四类文档各守职责；原始数据放根级 `evidence/`。
- 每个产品变更有 demand/defect、owner module、验证与 task 记录，不能只有 benchmark 名称。
- 候选冻结后进行独立红队；问题逐项裁决。
- 达到 `CLOSED` 后停止无证据重构，只按明确 reopen trigger 重新开启。
- 每轮运行适用的文档/构建机器合同并记录三项 LOC；LOC 只描述规模，不代表完成度。
