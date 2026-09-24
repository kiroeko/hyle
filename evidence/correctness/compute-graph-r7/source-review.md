# R7 候选后的有限源码复核

复核对象：`87ae583407cbb5ed13108c3837b020e6e343a3ca`，基线 `14e78c45`。在产品实现冻结后，
独立复核 F1–F5 的完整差异、调用方、失败边界及声明；没有重开未变模块或追加性能采样。

## 结论

源码阶段未发现需要新增产品 FIX 的问题。以下结论不替代仍在进行的 sanitizer、三交付或 fixture
构建；只有相应实际 receipt 完成后，R7 才能关闭。

| 风险 | 实际检查与结论 |
|---|---|
| 报告意外保活设备资源 | `CompileNodeReport` 仅有 ordinal/kind/traits；`OperationTraits` 是纯值。原生报告只存 Status、环境字符串、TensorDesc、计数和政策，不含 Operation、Tensor、Context 或回调。Apple capture 仅捕获内部 Plan；Vulkan 将 custom lowering 结果变成原生 kernel/字节，不保存 report 指针。 |
| reset、拒绝和 OOM | 两个原生 compile 包装层先 reset，失败再次清空并仅保留返回 Status；所有可能分配的元数据构造在既有 try/catch 中。Vulkan 在原生 Program 构成后若复制端口失败，局部 Result 销毁回收 Program；不发布半成功结果。普通运行测试及元数据 OOM 注入仍须实际通过。 |
| 共享报告与并发 | 输出 report 仅借用至 compile 返回，没有存入运行对象。两份指南明确同一 report 不得被并发写入；没有新增共享可变运行状态。旧 TSan 仅按不变路径继承。 |
| Generic Graph 分配和节点编号 | 冷 ReportState 通过原 plan allocator 分配；新增 nodes 使用同一 allocator，任一 resize 失败走原 release/error。节点 traits 复制自 GraphState，包括 symbolic View，查询检查空报告与越界。 |
| Vulkan 资源统计失真 | PlanFacts 在 packing、未引用常量释放和 storage plan 完成后读取；按真实 Buffer owner 去重，借入 owner 不当成 owned allocation，容量加法检查溢出。报告清楚区分逻辑容量、已知 VkDeviceMemory 与 unknown driver 成本。 |
| Apple 资源统计失真 | 只声明 logical constant views，重复绑定按声明计数；MLX 私有存储和融合保持 unknown，没有把逻辑字节当 RSS 或硬配额。 |
| 冷/热成本混淆 | nodes/ports 是线性冷复制；Vulkan 常量去重是 C 个 owner 指针排序，O(C log C) 时间、O(C) 临时空间，随后线性求和。默认无原生报告时不执行该分支。执行对象布局、算法和规则不变；OFF 宏与真实生成代码另有 P0 证据。 |
| 跟踪和交付边界 | 13 个静态名称 host zones 复用既有宏；OFF 不求值参数。真实 TRACE target 通过既有 PUBLIC contract 保持定义一致；B1 只豁免 absent 的中央 TRACE 必选条件，其余拒绝保持。ON 实际单 client 和交付依赖必须由运行门证明。 |
| 比较资格 | 活动 CPU Graph 仅 Default/Native Graph 同责入口；Sched 删除不等责 structured-spawn-join，产品头/实现树没有修改；六个裸 native cost target 及分支删除。保留完整服务 peer、独立数值 oracle 和原方向，不用删除入口制造 WIN/PASS。 |
| benchmark 污染 | 四路线 VERIFY_ONLY 在配对正确性检查后返回，GPU 还执行独立 oracle；返回点早于 nanobench warmup/compare。冷准备日志可能记录 prepare_ms，但不产生 P1 样本或执行性能裁决。Sched 本轮只构建保留入口，不运行计时。 |
| 历史数据与文档 | Graph v1–v3、Sched v1–v2 合同与基线逐 blob 一致。新合同明确尚无新 P1 资格。指南与 package consumer 使用同一新增报告接口，说明资源寿命、失败、unknown 和主机跟踪范围。 |

## 源码锚点

- [节点纯值](../../../include/hyle/graph/compile_report.h)、[操作 traits](../../../include/hyle/compute/operation.h)、
  [Generic 冷报告](../../../src/graph/executable.cc)与 [ReportState](../../../src/graph/plan_state.h)。
- [Apple lowering](../../../src/graph_route_apple/apple_lowering.cc)、
  [Vulkan lowering](../../../src/graph_route_vulkan/vulkan_lowering.cc)及
  [最终原生计划事实](../../../src/compute_route_vulkan/vulkan_program.cc)。
- [默认 OFF 宏](../../../include/hyle/obs/trace.h)、[构建门](../../../cmake/HyleModule.cmake)、
  [实际 P0 证据](README.md)。
- [当前比较边界](../../../bench/comparison-boundaries.md)、
  [CPU fixture](../../../bench/graph_route_cpu/region_p1_bench.cc)、
  CUDA 历史 fixture `bench/compute_route_cuda/route_p1_bench.cc`（已由
  [M3-D4](../../../docs/implementation/tasks/compute/llm-moe-d4-self-operator-retirement.md)
  随专用 Marlin 路线退场）、
  [Apple fixture](../../../bench/compute_route_apple/route_p1_bench.cc)、
  [Vulkan fixture](../../../bench/compute_route_vulkan/route_p1_bench.cc)。
- [Apple 指南](../../../docs/user/apple-compute.md)、[Vulkan 指南](../../../docs/user/vulkan-compute.md)、
  [普通 Graph consumer](../../../samples/package/graph/cpp/main.cc)。

本次只把冷复杂度表述精确到各分支；没有因报告排序改变运行成本或新增产品修改。
