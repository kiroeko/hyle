# 四路线基础架构验收索引

本页逐项对应已完成的四路线基础架构 MUST 与实际 evidence。R1–R7 均已结案；R7 不可变候选为
`87ae583407cbb5ed13108c3837b020e6e343a3ca`。各记录的资格限于其具名设备、版本、格式与 shape。

## 基础能力与实际证据

| 要求 | 已实现的入口 / 责任 | 验收记录 |
|---|---|---|
| 四条路线独立构建与交付 | public Compute/Graph 产品选择 + CPU/CUDA/Apple/Vulkan feature；vendor native backend 同步裁剪；无 CPU 模型 fallback | [R1](../build/compute-route-isolation-v1.md)、[隔离矩阵](../../../../evidence/correctness/compute-route-isolation-v1/README.md)；各路线下列三交付记录 |
| 自然 C++ Direct 与 Graph | CPU/CUDA 的自然 route factories 与固定规则；Apple/Vulkan 各自的 Context/Program/Invocation/results；Direct 无 Graph 依赖 | [R2](four-route-common-semantics-v1.md) 与四条路线的真实 package consumers |
| 共同操作语义、拥有结果与真实完成 | 有限 schema、生成结果 owner、接纳后错误仍保有实际工作、固定图边界；不恢复用户 timing-profile 搜索 | [R2](four-route-common-semantics-v1.md)、[Compute 结果合同](compute-v16-direct-owned-results.md)、[Graph 边界](graph-v11-four-route-compile-boundary.md) |
| CPU 默认实现与成熟 region | Native + oneDNN BRGEMM / whole region，固定适用条件；保留有证据的 HYLE 能力；资源和运行槽复用 | [R3](cpu-v1-four-route-foundation.md)、[正确性/交付](../../../../evidence/correctness/compute-cpu-r3-foundation/README.md)、[性能证据](../../../../evidence/performance/compute-cpu-route-p1-v3-d9712e68-w1-win/README.md) |
| CUDA 原生完整计算 | cuBLASLt、压缩驻留 Marlin、成熟 SGLang gated activation；同 GPU Provider 组合、stream/event、capture 和独立运行槽 | [R4](cuda-v1-four-route-foundation.md)、[最终默认与三交付](../../../../evidence/correctness/cuda-r4-mature-default/README.md)、[Direct P1](../../../../evidence/performance/compute-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md)、[Graph P1](../../../../evidence/performance/graph-cuda-ffn-p1-v3-755fb7c7-w1-win/README.md) |
| Apple 原生完整 region | MLX/Metal Q4/F32、同 runtime custom Metal；MLX 持有内部资源，HYLE 管 region 边界、FIFO owner 与结果租约 | [R5](apple-v1-four-route-foundation.md)、[Air 最终插桩](../../../../evidence/correctness/apple-r5-final-air/README.md)、[Mini correctness](../../../../evidence/correctness/apple-r5-final-mini/README.md)、[Air P1](../../../../evidence/performance/apple-r5-final-1636365b-m1-mac/README.md) |
| Vulkan 原生完整计算 | ggml strict-F32 shader 子集、同设备 custom SPIR-V、实际 VkDeviceMemory、immutable packing、liveness storage plan、原生命令与独立运行槽 | [R6](vulkan-v1-four-route-foundation.md)、[严格 F32 验证](../../../../evidence/correctness/vulkan-r6-f32-corrected/README.md)、[最终三交付](../../../../evidence/correctness/vulkan-r6-final-delivery/README.md)、R6 卡内四份实际 P1 |
| 开发时固定 best-of、准备时匹配 | 规则按 device/能力、格式、shape 及适用版本域设计；普通用户不排 Provider、不提供 timing profile；运行不重新计时或裁决 | R3–R6 普通入口、默认规则与留出 shape 验证；[路线合同](../../../architecture/compute-routes.md) |
| 准备/运行成本可解释 | HYLE 保留必要图规划、共享常量与可复用运行槽；第三方 whole region 不被拆成逐 kernel 通用 vtable；MLX managed 真实成本单独声明 | R3–R6 各自的资源、失败、并发与完整服务性能账单 |
| Mem 依赖尾项 | v10 Reservation / 外部字节配额，按原卡完成平台与失败回收验证 | [Mem v10](../mem/mem-v10-external-byte-reservation.md)，未扩为 Mem 全库重审 |
| 独立冷编译报告和 opt-in host 跟踪 | 共同节点数值语义；Apple/Vulkan owning metadata、明确 unknown 内存、reset/拒绝；默认 OFF 无新增热态工作，Tracy ON 复用单一 client | [R7](four-route-foundation-final-review.md)、[P0 / Windows / Mac host 插桩](../../../../evidence/correctness/compute-graph-r7/README.md)；Windows/Air/WSL 共 36 条 TRACE OFF/ON 三交付旅程通过 |
| 公平比较与旧入口清理 | 不等责裸 Direct / native 与 typed Sched 对照退出活动 A/B；历史数据和负向方向保留 | [比较边界](../../../../bench/comparison-boundaries.md)、[CPU 修正](cpu-r3-comparison-boundary-clarification.md)；Windows/Mac 共 20 个 fixture 构建、9 个独立数值 oracle 与退役入口负检查通过 |

## 资格限制

- 性能资格限具名机器、toolchain/driver、格式与 shape。Windows RTX 4070 的 CUDA SM89 / Toolkit
  13.3.73 / driver 610.62，以及 Air M4 的 Apple/MoltenVK 结果，不能外推为所有 GPU 的 SOTA。
  规则保留读取硬件/驱动/SDK 环境后固定适用域的能力。
- CPU 已接受的小片段例外是 Graph 生命周期服务的成本预算。八个裸 Direct 对照的服务不等价，
  已退出同合同排名；四个完整 Native Graph 对照保留原方向及非劣结果。
- Apple 是 MLX managed 路线；不承诺任意 MTLBuffer/Queue 互操作，也不把 MLX/driver 私有内存记为零。
  Vulkan 的首批 typed 内建范围是二维 F32 MatMul/SiLU 和 pure fixed-shape Graph；native custom 扩展另有合同。
- host ASan/UBSan 不覆盖 vendor/driver/GPU 指令；不故意制造设备挂死或丢失，也不为未注入的故障写 PASS。
  无新共享可变状态的路径按源码证明继承旧 TSan，明确不记为本轮重跑。
- 不包含算子目录扩张、CPU+GPU 模型联合推理、HIP、分布式、KV/batching 或推理引擎。
  ggml CPU 的旧 BLOCKED、compute_sched 的 QUEUED 保持原约束。

## 最终裁决

基础架构重构完成。[R7 terminal](four-route-foundation-final-review.md)七门 PASS，
[完整证据](../../../../evidence/correctness/compute-graph-r7/closure-identity.json)与当前 registry 已对应。
没有新增 P1 或改变原正负方向；实现共 37,035 行 / 130 个文件，不含测试。停止本轮施工，等待下一阶段讨论。
