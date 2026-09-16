# R3 Graph：同实现重复执行成本的定向审计

<!-- PROJECT UNIT REVIEW: SCOPED -->

## Closure card

- owner：graph、graph_route_cpu、graph_provider_onednn；baseline `5e9eb06a`。
- trigger：用户在 Graph v2 n=31 裁决后明确要求专门调研并解决阻塞；
  [前轮停止卡](cpu-graph-external-caller-qualification.md) 与全部失败/未决事实保留。
- 事实：qualification 与 workload A/A、公共 controls PASS；同实现双 worker FFN 的 Graph/direct
  median ratio 为 0.970361–0.982654，两个 CI 下界仍越过 1/1.05。其余 11 个工程门 PASS。
- 正常 consumer：同一个已编译的 CPU Graph，固定 shape/资源/队列，输入值变化，反复 submit/wait；
  两个独立 Invocation 仍必须可并发，关闭与失败仍必须按真实终结点回收。
- 对照：相同 oneDNN BRGEMM pair/down 与 HYLE SIMD SwiGLU 的直接融合调用；相同两 worker、
  三个物理 P 核、输入/精度/输出和完成边界。现有未融合与 Native 完整片段作为反偏置对照。
- 范围：沿 benchmark → Graph prepared replay → region enqueue → CPU Queue/completion 调用链
  建立逐阶段成本账单，检查冷准备已经证明的事实是否在热路径被重复计算。只按具体调用读依赖。
- 非目标：变更数学内核/默认 Provider 规则、扩充算子、重开 Sched 算法、放宽统计门或重新调核/idle；
  不新增公共 profile、通用执行器、用户手动排序或无需求的双轨 API。
- 当前仅 SCOPED 源码审计，不授权尚未定界的产品改动。确有多余工作后固定有限 FIX 并进入 AUDITED；
  如缺口属于 Compute 等其他 owner，先单独绑定精确依赖，不借本卡越权修改。
- 预算：源码/生成代码成本审计最多 40 分钟；只保留一个最小机制纠正方向。确需动态归因时，在现行
  Direct 验收结束后预先固定至多 8 个短 diagnostic 进程，使用真实片段，不以 probe 签性能 PASS。
  不允许重新跑原候选完整 P1 来寻找不同结果。若没有可证明的可消除成本，给出具体取舍并停止。
- 预声明验证：审计/文档只跑 selector、docs/closure/diff；进入产品 FIX 前再按其实际影响冻结
  correctness、生命周期/并发/失败、三交付与平台回放范围。API/算法稳定后，最多一份新候选的既有
  graph-cpu-route-p1-v2 工程 P1，原条件和 n=15→31 上限不变；适用未通过不能 CLOSED。

## 审计问题与不变量

1. 两侧各执行两次真实 bulk（gate/up 与 down），相同数学与 host mapping 是否确实等价？
2. Graph 的队列身份、lane/segment/step、依赖和 completion 选择，哪些随每次提交变化，哪些已经固定？
3. 可否在冷准备固定程序结构，同时保留队列关闭/重复接纳、Invocation 排他、provider 失败和资源保活？
4. 单片段、普通多算子、多个独立 Invocation 与异步/多段路径是否保持各自合同？
5. 修复必须来自真实重复执行成本，不为本次 M=17 shape 特判，也不移除正确性/生命周期保护来换数字。

## 初始源码锚点

- `bench/graph_route_cpu/region_p1_bench.cc`：GraphRoute::run 与 DirectFusedRoute::run；
  两侧完成后才返回，直接调用仍计 host mapping/close。
- `src/graph/executable.cc`：Invocation::submit、enqueue_segment_steps、完成组合与终结回调。
- `src/graph_provider_onednn/onednn_region_provider.cc`：region host 映射、BRGEMM pair/down 与激活。
- 具体热路径依赖按上述函数调用追踪，不重新审计整个 Compute、Mem 或 Sched。

有限 FIX 尚未冻结，产品文件尚未改动；本卡 SCOPED 不代表旧性能门已经解除或通过。
