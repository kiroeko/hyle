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
- 依赖不变；直接消费者为 Graph，公开交付通过 Mem 及 Graph source/static/shared 验证。
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
