# R4 C5：普通产品与 CUDA 三交付接线

## Closure card / AUDITED

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

## 审计冻结

现有 CMake 的 product 声明在 CPU feature 裁剪前形成 roots；增添两个 CUDA roots 后，CPU OFF 的
原有过滤仍只移除六个 CPU roots，CUDA 依赖随实际 backend 闭包进入。两个 route/三个 Provider
在 CUDA OFF 都使用已有 INTERNAL_ONLY_ABSENT，公共 factory/packing alias 受已有 capability
宏控制；不新增路线 enum、stub、SDK 下载或全局编译选项。普通 CUDA factory 的三实现覆盖互不
重叠的已支持操作域，来源和数值合同由 R4 主卡保持。

D1–D3 固定为 FIX。八个旧 owner 不改 C++ 产品输入，只有所属 facade 的交付闭包变化；baseline
receipt 用于验证原路径和原性能身份。系统 SDK 仍为 NVIDIA CUDA 13.3+ 的本机前提，Marlin 由
既定 pin/Apache notice 交付。源码/静态/动态 consumer 均只链接公开 hyle::compute/graph，不依赖
HYLE_DEV_FACTORS；Windows 验证显式复制实际运行库到各自 exe 目录，避免 PATH 偶然满足依赖。

普通消费采用已在 C4 证明的同一 FFN：Q4 gate/up → HYLE SwiGLU → dense down，既有 CUDA sample
分支从 runtime-only 替换成该真实计算旅程，Graph 要检查 kCaptured 与变化输入；CPU/no-backend
分支保持其原任务。Graph 自己管理中间 workspace，Direct 显式创建/复用 caller workspace；都在
真实 Completion 后读取结果。新的消费者代码是示例，不添加一套产品执行器或隐式 CPU fallback。

七问裁决：第三方与许可复用 R4；主/正交产品旅程如上；底层 owner 复用；P0 无旧 runtime 成本
变化、CUDA P1 独立；纯 C++；三交付同一公开 target；指南明确能力、失败、资源寿命和 capture
失效条件。D1–D3 完成后检查实际 install/export/依赖，适用门未完成之前不签 CLOSED。

### D1/D2 实现检查点

两个 CUDA roots 已接入公开 Compute/Graph 产品，selector 的 owner 检查为 READY；没有修改八个
旧 factor 的 C++。原 package 样例的 CUDA 分支改为普通入口下的 Q4 gate/up → SwiGLU → dense
 down，Graph 使用设备资源 compile/prepare_replay 并检查真实 captured 路线。CPU/no-backend
分支保留。Windows 源码父工程仅选择 compute/graph、CUDA ON/CPU OFF，两个公开产品样例 2/2
通过，每个执行三组变化输入；DLL 实际邻置。当前仍 AUDITED，安装包/WSL/隔离门待补，不记 PASS。
