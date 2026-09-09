# 能力展开与组件质因数分解

> 决策状态：`ACCEPTED`。本文定义先确定用户能力、再确定 owner 和依赖的架构方法。能力准入见
> [基础库能力完整性](capability-completeness.md)，一次施工如何推进见
> [模块收口方法](../developer/review/module-closure-methodology.md)。

## 为什么不能从现有目录反推产品

目录、target 和 include 边只能说明当前实现怎样组织，不能证明功能完整或边界正确。若先接受现有依赖
再决定能力，常会形成循环论证：某项能力因当前下层不能依赖上层而被删掉，随后又拿“下层没有它”证明
边界合理。

因此顺序固定为：

```text
用户任务与能力超集 -> 自然 API/成本合同 -> semantic owner -> 最小 factor -> 无环依赖 -> 产品 façade
```

当前有哪些 factor、product 和施工状态，只查询
[`HyleComponents.cmake`](../../cmake/HyleComponents.cmake)与
[`module-status.json`](../implementation/module-status.json)；本文不复制数量、完成状态或施工顺序。
Build/package 是所有组件的横向交付约束，不伪装成功能域或稳定产品 component。

## 先展开能力，不先改实现

一次模块审计先把名称当作产品问题域，而不是已证明正确的原子组件。发现集合来自：

```text
真实 HYLE 消费者
∪ 本轮选定的推理、渲染/资源流送、批处理或低延迟服务具体消费旅程
∪ 至少两个独立成熟基础/加速库反复提供的通用任务
∪ 语言、平台标准或定义性算法
∪ HYLE 当前公开入口的反向追踪
```

纵切组合、参考来源和反偏置规则由[能力完整性决策](capability-completeness.md)统一定义。这些来源用于
发现任务和形状，不是 API 名称并集，也不自动成为同合同性能 peer。Python、Triton、模型格式、tensor
策略、材质/场景规则或服务编排必须归到正确 owner，不能按名字复制进底层库。

每个候选能力至少记录：

- 来源、具体用户操作序列和正常输入规模；
- 创建、使用、组合、关闭与销毁的生命周期；
- ownership、failure/OOM、concurrency、边界、复杂度和常态资源成本；
- 对普通 C++ 用户自然的类型关系和最短调用草图；
- 当前路线：`COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE`；
- peer 独有历史包袱、应用策略或 benchmark-only 入口为何不准入。

已有公开入口必须回指一项已准入任务；只能回指 benchmark fixture 的入口应删除或私有化。能力表形成
有限 `KEEP/FIX/BORROW/DEFER/REJECT` 集后停止扩张，才进入实现和性能阶段。跨多个产品域的边界问题可
建立一次横向架构任务，但不要求无关模块停工等待一场新的全库总清点。

## 自然形状是硬门

每项 `MUST` 先有普通 C++ journey。类型名表达语义，而不是预期快慢；默认值安全；owner/view、
sync/async、bounded/unbounded 等真正不同的合同明确；用户无需模仿 benchmark、拼内部类型或理解第三方
实现才能完成任务。

性能不能反向制造 `fast/light/bench` 公共入口。API 与能力冻结后，correctness 和性能负责改进实现；
只有真实消费者的 material 机制证据证明合同本身错误，才退回审计阶段。

## 何时拆成内部质因数

一个组件只有同时满足下列条件，才值得成为不能继续有意义拆分的内部 factor：

1. 能独立说明 ownership、lifetime、failure、concurrency 和复杂度；
2. 有消费者只需要该能力，不应被迫取得更大产品闭包；
3. 拆分确实减少依赖、消除环或隔离可选成本；
4. 默认路径不替其他策略支付 allocation、锁、dispatch 或常态 metadata；
5. 能作为源码 owner 和 build/test 状态单元独立验证；
6. 去掉具名模型、服务、backend 与 benchmark 后仍是自然基础能力。

分解不追求最小文件或单函数 target。没有独立消费者、依赖收益或成本隔离时，应留在原 factor 内。
`container_intrusive` 是典型例子：它属于 Container 产品域，却能让 TimerQueue/Sched 使用零分配链表而
不取得完整 Container 闭包。

SIMD 是明确例外：Highway、xsimd、HYLE 自实现、ISA variant 与编译期路由都是一个 `hyle::simd`
产品的内部 provider，不拆成用户必须选择的多个 SIMD 产品。私有 target 可以存在，但不建立第二套
能力或状态 authority。

## 内部 factor 与外部产品 façade

factor 回答源码、状态、直接依赖和裁决权怎样组织；product façade 回答用户选择哪个问题域。普通用户
只选择 `hyle::<product>`。façade 是无源码、无独立状态的 `INTERFACE` 聚合，不拥有第二份 API、算法或
verdict。安装静态库可能需要导出 detail factor target 来表达 archive 闭包，但它不进入用户组件目录。

一个能力可供多个产品复用，但只能有一个 semantic owner。模块内部应优先复用已关闭的下层能力；若
复用会倒置依赖、形成环、改变 ownership/failure 语义或增加常态成本，应明确记录 `LOCAL` 或
`NOT_APPLICABLE`，不能通过隐藏 include 或复制实现伪装无依赖。

## 从能力导出依赖

对每个 factor 只登记真实边：

- `PUBLIC`：公开头或合同暴露另一 HYLE factor；
- `PRIVATE`：实现使用但不泄漏；
- `PRIVATE_BUILD`：仅源码构造期需要；
- build-only vendor/provider：第三方实现或生成工具，不冒充 HYLE semantic edge；
- consumer/reuse：记录需求来源和可复用加速设施，不复制进 CMake 图。

遇到环时依次检查 owner 是否放错、公开面是否泄漏机制类型、双方是否真的共享一个符合上述判据的小
factor；最后才考虑内部 seam。依赖图是能力裁决的输出，由中央组件图唯一保存并投影真实 target link。

完成条件不是“符号最多”，而是：每项准入能力有 owner 和自然 journey，每个公开入口有需求，factor
边界有真实收益，依赖无环且与 target 一致。随后模块仍按 closure 协议完成 correctness、交付、性能、
红队和文档门。
