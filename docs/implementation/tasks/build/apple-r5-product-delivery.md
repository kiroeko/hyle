# R5 A1/A6：Apple 公开产品与原生资源交付

## Closure card / SCOPED

- baseline：`9e41c15aab5a97206e82ebc70a0bee0d916e112f`；产品仍为 `d900a8ce`，后续仅 LOC 工具。
- 当前 owner：`compute`、`graph`、`compute_backend_cpu`、`compute_backend_cuda`、
  `compute_provider_native`、`compute_provider_onednn`、`compute_route_cpu`、`compute_route_cuda`、
  `graph_provider_onednn`、`graph_route_cpu`、`graph_route_cuda`。仅重开公开 roots 与交付闭包；
  两个 Apple owner 的 C++ 继续归 [R5 主卡](../compute/apple-v1-four-route-foundation.md) A1–A6。
- 工具 owner：构建包 registry 的 generated runtime asset 投影与 Apple 专属 staging helper。
  本卡是独立 build 任务，不在 Compute 产品实现中暗改公共工具。
- trigger：普通 `hyle::compute/graph` 尚未获得已经实现的 Apple route；MLX 即使 JIT 仍需要
  构建生成的 `mlx.metallib`。现有 registry 的 runtime assets 只接受锁定许可证，不能拿二进制产物
  冒充 license 或依赖开发机绝对 build 路径。
- 继承：十一 owner 的旧 terminal/candidate/performance 完整保存在
  [baseline](../../../../evidence/correctness/apple-r5-product-delivery/inherited-baseline.json)。
  CPU 例外、CUDA 的限定资格/方向和旧 runtime/算法/布局不变，不重跑它们的 P1。
- 主旅程：Apple ON、CPU/CUDA OFF → 普通 C++ Direct/Graph 的变化输入 Q4 FFN → checked result；
  同一自然代码分别 source、relocated static/shared 消费。正交旅程为 CPU-only、no-backend、
  unrelated core 和 Windows/WSL Apple OFF，保持原 DLL staging 合同。
- 范围：下表 D1–D4、真实资源/头/导出/许可证与普通样例/指南。不给用户新增 Provider 排名，
  不改共同执行器、selector 算法、closure checker 或性能仪器。
- 预算：15 分钟定向源码审计，固定 D1–D4；两轮具名 package seam 纠正，重型串行、parallel 2。
  发现其它基础组件缺陷另行定界，不扩全库重审。
- 验证：两 Mac 真实 source/static/shared Direct/Graph FFN、relocation 时撤去原 build 的 metallib
  fallback、共享 MLX runtime 唯一性、安装头/依赖/许可证、offline source cache；Windows/WSL
  Apple OFF 无 SDK 探测、CPU/无后端/core 裁剪；独立 package-registry 工具回归和 broad selector。
  源码身份相同的证据可依法复用；没运行的 applicable 项保持未证明。
- 性能：本卡 P0，旧 owner C++ 未变与冷交付成本；Apple 完整 P1 仍在主卡的稳定候选上执行。
- stop：有限交付项及适用门成立后 CLOSED；三交付失效、资源路径或公开 roots 改变才重新定界。
  本卡不签 R5 的其余风险/P1，更不签 R6/R7。

## 有限待审项

| ID | 缺口 | 允许动作与验收 |
|---|---|---|
| D1 | 公开产品缺少 Apple 实现 | 两个真实 route roots；OFF 保持 INTERNAL_ONLY_ABSENT，不装 SDK 头 |
| D2 | 生成 metallib 没有真实 package seam | 独立 generated runtime spec，绑定 producer target/build 路径，保留 license 锁校验；Apple helper 负责消费者邻置，shared native 库旁保留资源 |
| D3 | 普通 source/static/shared 消费尚未贯通 | 公开 target 的 Direct/Graph Q4 FFN 样例、变化输入和 oracle；真实 relocation/offline/链接闭包 |
| D4 | 新依赖与用户入口未形成说明 | 四项 license/notice、Apple CLI 构建要求、native 数组/结果/准备成本与 staging 指南 |

SCOPED 只允许审计；D1–D4 固定为 AUDITED FIX 后再改上述产品和 package seam。
