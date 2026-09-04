# 能力展开与组件质因数分解

> 决策状态：`ACCEPTED`，2026-09-01。本文定义全库重构前怎样先确定功能，再设计组件和依赖。
> 单模块能力准入仍由[基础库能力完整性](capability-completeness.md)约束；施工和收口仍由
> [模块收口方法](../developer/review/module-closure-methodology.md)约束。

## 问题

当前模块名、目录和 CMake 边只能说明现有实现怎样组织，不能证明功能完整，也不能证明依赖正确。
如果先沿现有依赖图施工，会产生循环论证：某项常用能力因为当前下层无法依赖上层而被删除，随后又以
“下层没有这个能力”证明现有模块边界合理。

因此，本轮全库重构先回答“基础加速库应提供什么、用户怎样自然使用”，再回答“由谁拥有、怎样拆分、
依赖谁”。功能不完整时，不允许先冻结组件图。

## 第一阶段：只展开功能超集

先把下列 13 个名称当作**产品问题域**，而不是已经证明正确的原子组件：

1. Core；
2. Error；
3. Container-intrusive；
4. Util-core；
5. Time；
6. Platform；
7. Mem；
8. Obs；
9. SIMD 与 Math 边界；
10. Container；
11. Util-runtime；
12. Sched；
13. IO。

Compute 与 Graph 在这一阶段保持 `QUEUED`，其旧实现只用于发现真实需求和反例，不作为应保留的产品
形状。Build/package 是每个候选组件的横向交付约束，不作为第 14 个功能域。

每个域的发现超集为：

```text
ggml 已提供的基础能力
∪ 最新锁定 llama.cpp 真实消费旅程所需的能力
∪ 至少两个成熟基础/加速库反复提供的通用能力
∪ 按通常工程划分定义上属于该问题域的基础能力
∪ HYLE 当前全部公开承诺的反向追踪
```

这不是 API 名称并集。每个候选能力必须写成真实用户任务，并记录：

- 来源与具体消费旅程；
- 创建、正常操作、组合、关闭和销毁的完整生命周期；
- ownership、failure、concurrency、边界与复杂度；
- 对普通 C++ 下游自然的类型关系和最短调用草图；
- 当前 HYLE 为 `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE`；
- peer 独有历史包袱、业务策略或 benchmark-only 入口为何不准入。

第一阶段不得因为当前 CMake 边、目录位置、已经关闭的旧 task 或实现成本而删除已准入能力，也不得修改
产品代码、开始计时或设计最终依赖。现有 include/link 边只作为待审事实另行抄录，不能成为能力裁决的
前提。13 个域的能力表全部完成并经过跨域查漏后，才冻结发现超集。

## 自然形状硬门

能力存在不等于形状合格。每个 `MUST` 必须先有面向下游用户的普通 C++ journey：类型名表达语义而非
预期快慢；默认值安全；owner/view、sync/async、bounded/unbounded 等真实语义差异明确；用户无需模仿
benchmark fixture、拼内部类型或理解第三方实现才能完成任务。

性能不能反向制造 `fast/light/bench` 公共入口。发现超集和自然形状冻结以后，correctness 与性能只负责
检验或改进实现；只有正常消费者的 material 机制证据证明公共合同本身错误，才退回能力阶段。

## 第二阶段：跨域归属与质因数分解

只有第一阶段的全局能力清单冻结后，才把所有能力放在同一张账本中去重并确定 semantic owner。一个
产品问题域可以由多个组件组成，一个组件也可以服务多个问题域。组件只有同时满足下列条件，才算不能
继续有意义拆分的“质因数”：

1. **语义闭环：**能独立说明 ownership、lifetime、failure、concurrency 和复杂度；
2. **独立消费：**至少存在一个消费者只需要该能力，而不应被迫取得更大的产品闭包；
3. **依赖极小：**继续拆分不再减少真实公开依赖或消除循环；
4. **成本闭环：**默认路径不替别的策略支付 allocation、锁、动态分发或常态 metadata；
5. **交付闭环：**source/static/shared 下能作为内部 build/ownership 单元独立构建、测试和验证；用户交付
   可以由所属产品 façade 聚合，但不能丢失精确依赖、头 owner 或静态链接闭包；
6. **不含上层策略：**去掉具名模型、服务、backend 或 benchmark 后仍是自然基础能力。

`container_intrusive` 是现有范例：它属于 Container 产品域，却是零分配、header-only、依赖极小且可由
Time 单独选择的独立组件；Time 不应为侵入式链表拉入完整 Container、Mem 和 Platform 闭包。

分解不是追求最小文件或单函数 target。只有拆分能形成真实独立消费者、减少依赖或隔离成本时才成立；
否则留在所属产品组件内部。

### 内部 factor 与外部产品不是同一张菜单

质因数回答“源码、状态、直接依赖和唯一裁决权怎样组织”，不自动回答“用户愿意选择什么”。多数
`platform_host`、`mem_reclaim`、`container_queue` 一类 factor 是为 HYLE 自己精确复用、避免依赖环和
隔离成本而存在；普通用户不应为了调用 Mem 或 Container 先理解整张内部 DAG。

Build/package 因此从同一份中央记录生成两个正交投影：

- **内部 factor DAG**：供 HYLE target、模块状态、精确测试和并行施工使用；factor 只能依赖 factor；
- **外部 product façade catalog**：按 Core、Error、Time、Platform、Mem、Obs、SIMD、Math、Container、
  Util、Sched、IO 等完整问题域提供 `hyle::<product>`，纯 `INTERFACE` 地选择相应 factor roots。

façade 没有源码、模块状态或新的 decision authority，也不能成为内部 factor 的依赖。安装静态库时，
CMake 可能必须导出未承诺的 detail target 才能表达传递 archive 闭包；这些 target 不进入
`find_package(... COMPONENTS ...)`、用户指南或兼容承诺。source/static/shared 三种交付始终使用同名
产品 façade。可选 TLS/H2 等是产品 feature，不是内部 factor 名的公开化。

该投影的施工合同见
[`build-system-v7-factor-facade-projection`](../implementation/tasks/build/build-system-v7-factor-facade-projection.md)。

### SIMD 不可拆分例外

SIMD 明确保持一个公开产品组件，不参与第二阶段的组件拆分。Highway、xsimd、HYLE 自实现、不同 ISA、
fixed/fat target 与零成本编译期路由都是 `hyle::simd` 的内部后端或实现分区，不得变成多个公开组件、
不同用户产品或重复决策 authority。这样保证用户获得一个完整 SIMD 能力面，也保证后续可以逐项把实现
路由到成熟依赖或 HYLE 明显胜出的实现，而不破坏产品形状。

SIMD 内部仍可有私有 target、生成步骤和 build-only vendor target；这些不改变其不可拆分的公开语义。
Math 是否独立、哪些标量能力定义上不属于 SIMD，必须在第一阶段单独审计，不能借 SIMD 例外自动吞并。

## 第三阶段：从能力和因子导出依赖

依赖图是输出，不是输入。对每个质因数组件分别登记：

- public semantic/type edge：公开头或合同暴露另一个 HYLE 组件；
- private mechanism edge：实现使用但不泄漏给消费者；
- build-only vendor edge：第三方源码、生成器或工具链资产；
- consumer edge：谁提供了能力需求和正常 workload；
- reuse/acceleration edge：本 factor 需要的机制由哪个既有 factor/标准/成熟依赖承载，哪些加速设施经
  语义与成本比较后适用或明确不适用；
- decision authority：同一运行时选择由谁唯一裁决。

出现环时按顺序处理：先检查 owner 放错；再检查公开面是否泄漏机制类型；再寻找同时被双方需要且符合
上述六项判据的更小因子；最后才考虑 internal seam。不得用 forward declaration、隐藏 include 或重复
实现掩盖语义环。

每个保留的 factor-local 机制还必须解释为什么不能复用已有低层能力；“当前没有依赖边”不是理由，因为
依赖图本来就是本阶段的输出。如果候选设施会倒置依赖、引入循环、改变 ownership/失败语义、增加常态
metadata/dynamic dispatch，或对 pointer-chasing/冷路径没有可用加速，则明确记录 `LOCAL/NOT_APPLICABLE`
而不强行复用。

最终实现顺序由新因子图的拓扑序导出，不沿用现有目录顺序。每个因子随后仍按“能力/API 冻结 →
correctness/迁移 → candidate → 性能与交付”的模块方法施工。

## 阶段交付与停止规则

第一阶段固定产物是一张覆盖 13 个域的能力账本；第二阶段固定产物是跨域去重、owner 与因子账本；第三
阶段固定产物是无环组件图、迁移映射和拓扑实施顺序。每个能力行一旦已有来源、自然旅程、覆盖状态、
准入/拒绝理由和边界就停止研究，不按 symbol 数或代码行追求无限超集。

在第一阶段完成前，不得宣称某个现有模块“功能够了”，也不得因依赖不方便提前重写模块。真实
correctness/security/build blocker 可以独立处理，但不能顺手决定本轮产品架构。
