# HYLE 架构方向与模块顺序

本文只记录长期模块边界、依赖原则和审查顺序。当前状态、任务和历史结果分别见
[implementation/status](../implementation/status.md) 与 [implementation/tasks](../implementation/tasks/README.md)。
组件图的机器真相是 [`cmake/HyleComponents.cmake`](../../cmake/HyleComponents.cmake)；本页解释它，
不另建一份可手工漂移的 manifest。

当前图只描述当前可构建实现，不是下一版架构的先验。全库重构先按
[能力展开与组件质因数分解](capability-factorization.md)统计功能超集和自然形状，再由完整能力集合导出
owner、组件因子和新依赖图。

Stage C 的能力账本曾导出 **30 个候选 factor**，它们是下一版归属和拆分的设计输入，不是已经存在的
CMake target。当前机器合同只登记 **18 个真实 factor**，并从中投影 **14 个用户产品 façade**；只有进入
`module-status.json`、中央组件图和实际 target 的 factor 才算已经落地。Build/package v7 已实现这套
双层投影。内部 factor 只能依赖 factor，用户只需选择产品域；详见[构建系统](build-system.md)。

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

本节表格是**当前 18-factor 可构建现树**。下一版 30-factor 候选图保存在历史任务
[`capability-factorization-v1`](../implementation/tasks/project/capability-factorization-v1.md)，只用于后续
逐项准入；不得把候选记录写成当前 target、当前依赖或当前模块状态。

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
| math | core | — |
| time | core | — |
| timer_queue | core, time, container_intrusive | — |
| platform | core, error, time | — |
| mem_reclaim | core | — |
| mem | core, error, platform, util_core | — |
| simd | core, platform | — |
| container | core, mem | platform |
| util | core, error, mem, container | — |
| sched | core, container_intrusive, error, time, platform, mem, util_core | container, timer_queue |
| io | core, mem, container, time, timer_queue, sched | — |
| compute | core, error, mem, sched | platform, simd |
| graph | core, error, mem, container, sched | util_core |

用户看到的不是上述 17 个 factor，而是以下 14 个产品 façade。`Factors` 只列 façade 的直接 root；CMake
再从同一中央图解析其完整传递闭包。façade 没有源码、状态或新的依赖边。

| Product façade | Factor roots |
|---|---|
| core | core |
| error | error |
| time | time, timer_queue |
| platform | platform |
| mem | mem |
| obs | obs |
| simd | simd |
| math | math |
| container | container_intrusive, container |
| util | util_core, util |
| sched | sched |
| io | io |
| compute | compute |
| graph | graph |

因此，“真正的模块”按开发、源码 ownership、状态和依赖计算是 18 个 factor；按普通用户可选择的问题域
计算是 14 个产品。Build/package 是贯穿所有 factor 的横向开发单元，外部依赖 target 与 trace/testing
target 是 build-only mechanism，二者都不增加产品或 factor 数量。

声明顺序必须是拓扑序，闭包必须无环。外部第三方依赖由所属模块登记，不出现在内部组件图中。
SIMD 与 Container 的选定 per-ISA kernel 都可直接编译中央锁定的 Highway；两者共享 build-only
`hyle_highway_headers` 外部实现 target，但 Container 不因此依赖、构建或安装 `hyle::simd`。Highway 的
来源、许可证和产品使用由外部依赖账本登记，这不是一条 HYLE 内部组件边。

`container_intrusive` 的公开入口是 `<hyle/container/intrusive_list.h>`、
`<hyle/container/intrusive_forward_list.h>` 与 `<hyle/container/intrusive_hlist.h>`，公开类型位于
`hyle::container`；内部 factor target 是 `hyle_container_intrusive`，普通用户由 `hyle::container`
façade 取得该能力。Time 产品中的 `timer_queue` factor 公开 `TimerEntry` 与 `TimerList`，保留真实
`PUBLIC container_intrusive` 边；host-clock `time` 不因此取得 intrusive closure。Sched 的公开 cancel
record 嵌入 safe hlist hook，因而同样声明真实 `PUBLIC container_intrusive`；waiter 在摘链后复用 link
storage 作锁外 wake chain，仍保留专用实现。
这些消费者都不会把 `time` 拉入完整 container 的 mem/platform 闭包。组件自己的边界见
[`container-intrusive-design`](container-intrusive-design.md)。

`mem_reclaim` 是从下一版候选图登记的独立开发 factor。它只依赖 Core，拥有 read-side guard、intrusive
retire record 和 grace-period callback；在其性能门与真实 consumer migration 完成前，不进入 Mem façade，
也不改变已经关闭的 Container/Obs。具体可交付状态只以 registry 与该 factor 的 closure card 为准。

`core`、`container_intrusive`、`util_core` 与 `math` 都是正式 `INTERFACE` factor；只有 Core/Math
分别形成同名用户产品，intrusive/util-core 由 Container/Util façade 聚合。公开头直接 include
`<hyle/core/**>` 或 `<hyle/version.h>` 的模块必须直接声明 `PUBLIC core`，即使没有 binary symbol
需要链接；这条边传播 include/build/package contract。`math` 使用 Core 的 release-surviving fail-fast
primitive，因此同样声明真实 `PUBLIC core` 边。配置合同会把公开头扫描与中央
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

依赖顺序不得成为复制实现的理由。较高 factor 需要较低 factor 时，登记真实直接边并复用；较低 factor
若看似需要较高 factor，依次裁决 owner 是否放错、公开面是否泄漏机制类型，以及能否提取满足六项质因数
判据的更小 factor。都不成立时，使用标准库/成熟依赖，或保留有明确状态机、layout、生命周期和成本理由
的专用实现；不得只因“当前不能依赖”复制同义 error、clock、allocator、probe 或容器。`next/prev` 字段
本身不证明语义重复，例如 freed-slot free-list 和摘链后复用 link 的 wake chain 可以合法特化；普通
cancellation hlist、arena/deferred-work forward chain 等重复形状则应在其能力准入后回审公共质因数。

## 4. 全局功能展开顺序

下列顺序只用于有界地完成只读能力统计，不代表现有边界已正确，也不是未来实施拓扑。Build/package 是
贯穿每个候选组件的横向合同，不是最后补票：

1. core；
2. error；
3. container_intrusive；
4. util_core；
5. time；
6. platform；
7. mem；
8. obs；
9. simd 与 math 边界；
10. container；
11. util-runtime；
12. sched；
13. io。

这一遍必须先完成每个问题域的 ggml 能力、llama.cpp 真实需要、成熟库通用能力、定义性能力和现有 HYLE
承诺的超集，再统一做跨域去重、归属和质因数分解。不得边统计边实现，也不得以当前依赖冲突删除能力。
新组件图完成后，实际施工顺序只由它的拓扑序决定。

SIMD 是不可拆分的公开产品组件；Highway、xsimd、HYLE 自实现和 ISA 路由只是内部后端。Math 边界仍需
独立裁决。Compute 与 Graph 保持 `QUEUED` 和冻结：旧实现只用于发现需求，在新底座组件图与真实推理
consumer slice 成立前不做产品重构，也不为旧 benchmark 形状优化。

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
