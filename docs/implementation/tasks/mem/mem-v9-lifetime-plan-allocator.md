# Mem v9：生命周期计划器的 metadata allocator

## Closure card

- component / owner：`mem` / Mem；baseline `03975179`；前一 terminal 为
  [v8](mem-v8-resource-type-visibility.md)，其它 Mem 能力沿用既有结论。
- trigger：Graph G106 的冷编译/运行期预算需要约束完整 metadata；`src/mem/lifetime_plan.cc` 的
  RawArray、PlanImpl 与 PlannedImpl 全部硬编码 `system_allocator()`，因此 Context allocator 的 quota
  和逐分配故障注入无法覆盖这一真实直接依赖。不是 benchmark 或重复 review 触发。
- 主旅程：Graph 用自己的 metadata allocator 规划并创建 workspace，再以 Mem Buffer 保有设备 backing。
  正交旅程：直接 Mem 用户将编译计划与运行实例分配给不同的 caller-owned quota/arena。
- scope：只补 LifetimePlanner/LifetimePlan 的显式 metadata allocator 与完整 sized-free；Graph 传入
  既有 allocator。不会让 payload 改走 host allocator，也不扩展 tensor/graph/runtime 职责。
- non-goals：best-fit/alias/lifetime 算法更改、常态 allocation 次数增加、device resource 虚表更改、
  全局 allocator setter、其它 Mem 能力、Native FFI、公共仪器或 CMake 改造。
- 依赖不变；真实产品调用者为 Graph，Mem-vm API surface 测试也消费该公开头；公开交付通过 Mem 及
  Graph source/static/shared 验证。
- 性能 P0：算法/分配次数/布局查询保持，检查新增 allocator identity 的冷 metadata 成本；不为冷绑定
  跑一轮纳秒 P1。既有 allocator/资源内核未改，不重新签其它能力的性能胜负。
- 预算：一次表示/实现及一轮纠正；错误注入覆盖每次 scratch/result/instance metadata 分配，验证
  payload OOM、move/reset、plan 释放后 instance、继承与覆盖 allocator、空 plan/非法 allocator。
- 验证：Mem correctness/逐头、Graph/oneDNN/CUDA 定向回归、source/static/shared；Windows、WSL GCC14、
  两台 Apple 的 Mem 定向正确性及适用 ASan/UBSan；独立于实现步骤的一遍源码反证；不跑全库/P1/CI。
- stop：必须改变规划算法或 payload ownership、allocator 生命周期无法自然表达，或新的常态工作；
  适用验证未通过不关闭。本轮成功只补依赖缺口，不代签 Graph 的完整内存 gate。

## SCOPED

现状缺口为 `PARTIAL/FIX`：payload 的 MemoryResource 已可选，但 planner scratch/result 与 instantiated
metadata 没有可注入路径。职责仍归 Mem；Graph 不复制 planner 或在 Mem 外猜测内部 allocation 数量。
下一阶段先固定公开调用与生命周期，再修改产品。

## AUDITED：固定接口与有限改动

| ID | FIX | 验收 |
|---|---|---|
| M90 | `LifetimePlanner::build(requests, resources, allocator = system_allocator())`；所有 scratch/result 经同一 allocator，保留 sized-free identity | 逐分配拒绝完全回滚；无 payload 分配；无新算法或额外 heap 次数 |
| M91 | `plan.instantiate(resources)` 继承 plan allocator；`plan.instantiate(resources, allocator)` 显式选择另一 metadata owner | instance 可超过 plan 生命周期；override 不借旧 allocator；payload 仍由 resources 分配/释放 |
| M92 | Graph 向 LifetimePlanner 传 `plan.allocator`，运行实例自然继承；guide/交付与故障测试同迁移 | Context fault injection 不再漏过 Mem scratch；三交付同义 |

所有 allocator 均为 borrowed `AllocatorRef`，必须活到其所分配的 plan/instance 最后释放。instantiate 的
默认继承与显式覆盖是两种真实 owner 需求，不是旧 ABI 兼容分支；无参数改变的 view/backing 查询保持
原样。显式 invalid allocator 拒绝，空计划也需要可用 metadata allocator。临时 RawArray 保有实际
allocator，转交结果后由 PlanImpl/PlannedImpl 独立保有 identity；用户可直接组合已有 `QuotaAllocator`，
不新增另一个 quota、全局 setter 或资源种类。

普通 C++ 形状：`LifetimePlanner::build(requests, resources, compile_quota.get())`，然后
`plan.instantiate(resources, run_quota.get())`。metadata quota 不约束 resource payload 或 allocator 内部
rounding；这一边界必须在头和用户指南同时说明。公开 owner 仍为单指针，新增 identity 只在 cold impl/
临时数组；分配数量、best-fit/alias、`O((N+E) log N)` 和 view 的 O(1) 路径不变。

至此 M90--M92 为有限 FIX。需要改变 payload/算法或增加其它公共功能时重新裁决，不顺手扩大本轮。

## 2026-09-15 实现检查点

M90--M92 已实现：全部 RawArray scratch/result 和 PlanImpl/PlannedImpl 使用显式 allocator；每个 owner
保存实际释放身份。默认实例继承，显式实例覆盖，payload 仍使用原 MemoryResource。Graph 唯一的
LifetimePlanner build 调用现传入 `plan.allocator`，没有复制 planner 或修改 placement 算法。

新增六项测试逐次拒绝 planner/instance metadata 分配，检查同 owner/size/alignment 释放、失败重试、
payload 第二次分配失败回滚、零配额/非法 allocator、计划先释放、实例 move/reset 和并发实例化。Mem-vm
API surface 与 Mem package sample 同步验证两个公开实例化形状；guide 区分 metadata 与 payload 预算。

Windows Release 增量：`compute-route-v8-static-20260914` 的 Mem/Graph/逐头 5/5，
`compute-route-onednn-v8-20260914` 的 Mem/Mem-vm/Graph/oneDNN/逐头 10/10；后者 Mem 237、Mem-vm 65、
Graph 基础 25、Graph 执行 227、oneDNN direct 1、region 17 个用例通过。最后仅整理新增声明的局部格式与
并发限制注释，不据此运行 P1。这些是增量正确性，不代签尚待候选验证的三交付、跨平台、sanitizer 和红队。

影响边界：公开头的保守 selector 会展开 Mem 的传递消费者；本卡实际代表集为直接使用 LifetimePlanner
的 Graph、Mem-vm API surface 与 Mem 三交付。其余 Container/IO/Process/Sched/Util 不调用本次变更的
planner API，AllocatorRef/MemoryResource/Buffer 的合同与实现没有变化，不额外运行那些模块的产品测试
或 package；该裁决遵循 correctness profile 的代表性直接消费者规则，不把未运行的自动选择项写成 PASS。

## Candidate

产品候选为 `8e4f096e801fda16aa3b8ab507556188e7883159`。M90--M92 的 API/算法/正常成本与直接消费者
迁移已固定，后续只验证并记录证据；Mem 进入 `CANDIDATE`，不是 `CLOSED`。Graph 保持原活动阶段，
完整内存预算仍待此依赖验证后继续。
