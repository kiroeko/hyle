# HYLE 架构方向与模块顺序

本文只记录长期模块边界、依赖原则和审查顺序。当前状态、任务和历史结果分别见
[implementation/status](../implementation/status.md) 与 [implementation/tasks](../implementation/tasks/README.md)。
组件图的机器真相是 [`cmake/HyleComponents.cmake`](../../cmake/HyleComponents.cmake)；本页解释它，
不另建一份可手工漂移的 manifest。

## 1. 分层原则

HYLE 是推理引擎与工程计算消费者下面的通用 C++ 加速底座。能力应归到能够完整表达其 ownership、
failure、concurrency 和复杂度的最低合理层，但不能把上层策略塞进底层：

- core 是 header-only target，只承载极小、稳定、跨模块必要的编译与布局 primitive；
- `container_intrusive` 单独承载零分配、header-only 的侵入式容器原语，避免底层消费者
  为一个链表类型拉入完整 container 实现；
- error 统一可恢复失败；platform 统一 OS/硬件机制；
- mem/time/obs/util/container/simd 提供可组合基础能力；
- sched/io/compute/graph 组合底层机制，但不反向渗漏策略；
- 模型格式、token/KV、continuous batching 和 serving policy 属推理引擎；
- source/static/shared 是 C++ 交付；Native FFI 只有真实外语任务才准入。

“放到底层以便复用”不是充分理由。公共基础类型必须有通用消费者形状；只有 HYLE 内部需要、
会暴露策略或仍在变化的能力留在 internal seam。

## 2. 组件图

下表是当前中央声明的直接边。`PUBLIC` 表示公开头/链接合同泄露该依赖；`PRIVATE` 只属于编译实现。
当前 io、compute 与 graph 的公开面都暴露 sched 类型，因此它们无条件选择 sched；目标平台缺少
fcontext 时配置会拒绝这些根，而不是生成删掉公开依赖的残缺 SDK。

| Component | Public deps | Private deps |
|---|---|---|
| core | — | — |
| container_intrusive | core | — |
| util_core | core | — |
| error | core | — |
| obs | core, error | — |
| math | — | — |
| time | core, container_intrusive | — |
| platform | core, error, time | — |
| mem | core, error, platform, util_core | — |
| simd | core, platform | — |
| container | core, mem | platform |
| util | core, error, mem, container | — |
| sched | core, error, platform, mem, container, util_core | time |
| io | core, error, mem, container, time, sched | — |
| compute | core, error, mem, sched | platform, simd |
| graph | core, error, mem, container, sched | util_core |

声明顺序必须是拓扑序，闭包必须无环。外部第三方依赖由所属模块登记，不出现在内部组件图中。
container 另有不进入产品闭包的 build-private `simd_vec_headers` seam：它让 per-ISA 实现复用
`hyle/simd/vec.h` 时留下真实 build-interface target 边，但不会为 container 用户构建或安装完整 simd。

`container_intrusive` 的公开头是 `<hyle/container/intrusive_list.h>`，公开类型位于
`hyle::container`；target 名称为 `hyle::container_intrusive`。`time` 的公开 `TimerEntry` 与
`TimerList` 暴露该类型，所以它是 `PUBLIC` 依赖；完整 `hyle::container` 不聚合这项独立能力。
sched waiter 在摘链后复用 link storage 作锁外 wake chain，保留专用实现且不登记虚假 direct edge。
这些消费者都不会把 `time` 拉入完整 container 的 mem/platform 闭包。组件自己的边界见
[`container-intrusive-design`](container-intrusive-design.md)。

`core`、`container_intrusive` 与 `util_core` 都是正式 `INTERFACE` 产品组件。公开头直接 include
`<hyle/core/**>` 或 `<hyle/version.h>` 的模块必须直接声明 `PUBLIC core`，即使没有 binary symbol
需要链接；这条边传播 include/build/package contract。当前除 `math` 外的组件都有真实 Core
公开消费者并显式登记该边，`math` 没有 include 就不增加伪依赖。配置合同会把公开头扫描与中央
声明对账，selected-header 安装或共享源码 include root 都不能充当隐藏依赖。

中央图和 target link 并不是两份人工事实：`HyleComponents.cmake` 的同一声明同时用于选择闭包和
向 `hyle_add_module()` 投影真实 `PUBLIC/PRIVATE` link edge。模块 CMakeLists 只声明源文件与模块
拥有的外部依赖；机器合同必须拒绝未知边、重复边、倒序和 package 漏链。

## 3. 依赖审查

每次模块 review 都同时检查四种问题：

1. 本模块是否依赖了更高层，或只为方便而扩大 `PUBLIC` 边；
2. 下游是否绕过本模块，私造 error、clock、allocator、CPU probe、queue 等同义机制；
3. 公开头是否真正暴露依赖类型，决定 `PUBLIC`/`PRIVATE` 是否正确；
4. source、installed static、installed shared 消费者能否只选择根组件并得到完整闭包。

修一条底层边后，必须回审直接消费者，不做无界的全库顺手重构。若问题属于另一个模块，登记
demand/任务并等待其轮次；只有 correctness 或构建阻断允许立即跨边修复。

## 4. 逐模块重审顺序

顺序综合机器依赖和推理消费者价值；build/package 是贯穿轴，不是最后补票：

1. core；
2. error；
3. container_intrusive；
4. util_core；
5. time；
6. platform；
7. mem；
8. obs；
9. container；
10. util-runtime；
11. simd 与 math 边界；
12. sched；
13. io；
14. compute；
15. graph。

前一个模块 `CLOSED` 后再冻结下一个，以免下层 API 反复击穿上层证据。研究成熟库、准备 workload
和独立红队可以并行；会修改同一依赖边或公开面的实现不得在共享分支并行合并。

compute 与 graph 的产品形状仍可能由真实推理引擎推翻，在其前置模块稳定前只修 correctness、
构建阻断和必要 adapter，不为旧 benchmark 形状做冻结式优化。

## 5. 每轮架构输出

每个模块按 [审查方法](../developer/review/module-closure-methodology.md)完成：

- 自顶向下 capability-family map 与现有公开面反向追踪，区分
  `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE`，再用 capability/demand ledger 裁决实现、组合、
  借用、上移、延后与拒绝；
- public/internal 与 owner module 裁决；
- ownership/lifetime/concurrency/failure/complexity 类型表；
- 消费者与依赖双向扫描；
- correctness、性能仪器、正常 workload、交付和易用性证据；
- 四类文档各自的唯一事实；
- 独立红队、已知限制、停止与 reopen trigger。

推理引擎提供首要 demand 与 workload，第二种工程计算形状作为泛化留出。基础 API 必须通过语义
擦除：去掉模型、token、服务和 backend 名称后仍然是自然的通用任务。详见
[推理引擎北极星](inference-engine-north-star.md)与
[能力完整性决策](capability-completeness.md)。

## 6. 停止规则

模块签署后停止基于 LOC、feature 数、命名偏好或单次微基准的反复重构。只有真实消费者缺口、
correctness/security、直接依赖语义变化、新平台/ISA/backend，或强 peer 在同合同正常 workload
证明更好复杂度或 material advantage 时才重开。

项目规模不是完成度代理。可维护的模块边界、无重复实现、真实消费者可用和可复现证据，比达到
某个总代码行数更重要。
