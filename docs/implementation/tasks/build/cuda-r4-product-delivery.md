# R4 C5：普通产品与 CUDA 三交付接线

## Closure card / SCOPED

- baseline：`d2e631b4e9eda584d94f5b9936f68e01f4a5a3c7`，工作树干净；C4 的三个真实 Provider、
  普通 factory 和 native captured FFN 已验证，公开产品 roots 尚未加入这两个路线 factor。
- owner：`compute`、`graph`、`compute_backend_cpu`、`compute_provider_native`、
  `compute_provider_onednn`、`compute_route_cpu`、`graph_provider_onednn`、`graph_route_cpu`。
  CUDA backend/三个 Provider/两个 CUDA route 仍归 [R4 主卡](../compute/cuda-v1-four-route-foundation.md)
  的 AUDITED C1–C6，不重复登记。这里仅将原 CLOSED roots 合法重开为交付范围。
- trigger：普通 `hyle::compute/graph` 用户应通过公开产品获得已实现路线。现有 selector 对 product
  roots 变化要求 old/new root union 的 owner 均已 AUDITED；不能为接线绕过该保护或顺手改工具。
- 正常旅程：选择 compute/graph 产品与 CUDA ON、CPU OFF → source 或 relocated static/shared
  消费 → 普通 Direct/Graph factory → 冷 pack/compile/prepare → 变化输入 FFN → 完成与回收。
  正交旅程为 CPU-only、no-backend、backend-only 与不相关 core 产品，保持各自真实裁剪边界。
- 范围：中央产品 roots、当前实际 CUDA factor 的导出/依赖/许可证和用户样例/指南、交付验证。
  本卡八个旧 factor 的 public C++、运行时/算法/对象布局/CPU 规则不改；若有具体缺陷另按 owner 定界。
- 继承：原 registry candidate、performance 字段和完整 claim 保存在
  [baseline receipt](../../../../evidence/correctness/cuda-r4-product-delivery/inherited-baseline.json)。
  旧 CPU 数据、批准的唯一例外、C7 失败槽语义保留；不把文档或交付接线当成新算法完成。
- 性能：本卡 P0，证明八个旧 factor 的 runtime/source/headers 与 baseline 未变。CUDA 完整 P1
  仍由主卡在 API/算法/交付固定后的不可变候选上执行；本卡不触发 CPU P1。
- 预算：15 分钟定向 roots/export/staging 审计，固定有限 D1–D3；实施/交付共用 R4 C5 的一个有限
  工作时段，重型串行、并行 2，同 exe 目录 staging 串行。遇到真实新错误只修具体机制，不扩全库重审。
- 预声明验证：Windows/WSL NVIDIA public source/static/shared 的实际 Direct/Graph FFN 与 relocation；
  CUDA-only 不带 CPU/oneDNN；CPU-only/no-backend/core 在 CUDA OFF 不查 Toolkit/不安装 CUDA 入口；
  原 backend-only 不取得默认 Providers；公共头与安装导出/依赖/Marlin license 一致。旧 CPU 代表性
  样例在受影响产品组合中消费；macOS 只检查 CUDA OFF 的配置/发布投影，无 CUDA 运行资格。
  Windows 运行前真实 DLL 邻置，非交互、有限 timeout。共同工具和性能核心不改。
- stop：有限 glue 与上述适用旅程成立后冻结/终审/CLOSED；未跑的适用门保留未证明。完成本交付卡
  仍不宣称 R4 的 P1 或 R5–R7 完成，不解除 ggml BLOCKED 或 compute_sched QUEUED。

## 有限待审项

| ID | 需求 | 允许动作 / 验收 |
|---|---|---|
| D1 | 普通 facade 尚未获得 CUDA 默认规则 | 添加 compute_route_cuda / graph_route_cuda 两个真实 roots；保持 feature 裁剪与已有 factor edges |
| D2 | 普通安装样例目前只 copy/runtime probe | 把 CUDA 分支迁移为默认 Direct/Graph 的真实 FFN、packing/capture 与检查结果；保持 CPU/no-backend 分支 |
| D3 | 三交付与系统 SDK/运行库边界需要实际验证 | 使用既有 source/package harness，补精确依赖/导出/许可与指南；必要修正限于 owning CUDA factor |

scope 已冻结；下一步按 D1–D3 审计实际 CMake 和自然消费，不在 SCOPED 写产品代码。
