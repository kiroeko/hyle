# R6 Vulkan：原生计算程序、资源与 Graph

## Closure card / SCOPED

- owner：`compute_route_vulkan`、`graph_route_vulkan`；身份已在 `9059ad30` 登记为 QUEUED。
- baseline：`9059ad30`，保留 R5 不可变候选 `1636365b` 及其仍在运行的串行验收。
- trigger：[四路线合同](four-route-foundation-refactor.md)的 R6 已获实施授权，当前无 Vulkan 产品路线。
- 当前仅定界、源码与环境准入审计；产品实现须先完成下列有限审计并进入 AUDITED，且不抢占 R5
  重型验证。登记身份和本卡不计为已实现路线，也没有创建空 target。
- 主纵切：常驻矩阵权重、变化 activation、门控 FFN 的 Direct 与 Graph；借用已登记 ggml Vulkan
  的成熟矩阵/门控实现。具体数值和物理格式在审计中按真实 shader 合同固定，不把其它路线的四位格式套入。
- 正交纵切：数值/资源管线中成熟计算与 HYLE 自定义 SPIR-V 片段组合，真实 upload/compute/download、
  ready native buffer handoff、两个运行槽、保留旧结果和关闭回收。
- 依赖方向：Compute 的操作/shape/Completion/ResultSlot、Error/Mem/Platform；Graph 只冷读取
  DAG 并降成 Vulkan Program。Direct 无 Graph 依赖，不复用 CPU/CUDA/Apple 的执行器或 Queue。
- 范围：原生 instance/device/queue、device memory 与 staging、pipeline/descriptor/command
  program、运行槽、失败完成和回收；成熟计算接入、同设备扩展、固定默认规则、Graph、三交付与文档。
- non-goals：算子大全、模型/KV/batching/serving、跨 GPU/跨设备模型执行、CPU 模型 fallback、
  HIP、任意动态图、旧 API 兼容、未知硬件的 SOTA 声明。多 queue/family 仅在实际公开入口承诺时纳入。
- 目标验证域：Windows RTX 4070 原生 Vulkan；Apple Silicon 的 MoltenVK portability。WSL/Linux
  先核对实际 loader/device，区分编译交付、原生 GPU、转换层与软件 device；无设备时如实保留资格边界。
  AMD/Intel 无当前实机，不从 NVIDIA 外推；Mini 只作正确性，不冒充 Air 性能机。
- 调查预算：60 分钟，限下面三个具名问题；最多一个确有未决原生约束的短 admission 程序。
  已有两 pass 探针不重跑、不计产品。两次具名 shader/build 接入纠正；新基础缺陷定向绑定 owner。
- 性能预算：P0；API/算法/规则冻结后的 Direct、Graph 各一次 P1，原 5% 非劣门与方向分开，
  n=15，仅资格成立而区间未决时最多 n=31；最多一次有源码归因的机制纠正。禁止按 benchmark
  创建公共能力、改统计核心或用不同数值/完成责任签输赢。
- 验证：有限 FIX 的真实 native 正确性、独立 oracle、非法域/OOM/部分接纳/晚到错误/取消/关闭、
  并发槽与保活、适用 host 插桩及 Vulkan validation、native handoff、两系统三交付/relocation、
  OFF 无 SDK、public headers、默认规则与完整成本 P1；不故意 GPU hang/device-loss。
- stop：有限问题裁决后固定 AUDITED FIX，不继续全架构研究；具体必需原生能力/环境不能兑现时
  记录精确 blocker 与恢复条件。任何适用未运行项不签 PASS。R7 不首次接线本路线的验收。

## 有限审计问题

1. **成熟实现边界**：已登记 ggml `456172ec733a135778adcd32d00e576a58232e45` 的 public
   Vulkan API 没有 device/queue/buffer import/export；完整 runtime 每次 graph_compute 编码。
   比较受审 shader 子集与窄 runtime bridge 的完整生命周期/融合/维护成本，选择一个默认承载。
   Vulkan 的 `vk::SystemError` 不能与旧 ggml CPU abort blocker 混为一谈；CPU owner 继续 BLOCKED。
2. **资源和执行合同**：固定 native owner、外部借用方保活、recorded/managed、descriptor 更新、
   pending fence、memory visibility、non-coherent flush/invalidate、cancel/close。普通入口必须
   自己处理必要回收，不要求用户重复手写 glue。Graph 中间资源按实际 lifetime 复用。
3. **实际能力与交付**：核对 loader/driver/device、shader compiler、Vulkan-Headers/Volk/glslang
   pins、MoltenVK 与启用 feature；compiler 识别扩展不等于设备可用。固定实际可维护的 shader
   变体与合法基线，按设备/架构/driver/SDK/数值/shape 冷绑定，热态不查库或计时。

## 能力承载待裁决

| 能力族 | 层级与来源 | 当前状态 | 审计方向 |
|---|---|---|---|
| 原生 buffer/memory、pipeline、descriptor、commands、fence | Vulkan 原子机制；数值与资源管线 | MISSING | BORROW Vulkan；SELF 必要 RAII/错误与组合边界 |
| MatMul、门控完整片段 | 成熟复合计算；ggml/llama.cpp FFN | MISSING | BORROW 受审成熟实现，保留适用融合/硬件专用分支 |
| 自定义 SPIR-V 与 native handoff | 原生组合；资源流送/数值管线 | MISSING | SELF 窄冷准备与保活合同，复用同 device/queue |
| Program/Invocation/Completion/owning result | 复合生命周期；Direct 与 Graph | 共同结果语义已可用，路线 MISSING | ROUTED 既有 Completion/ResultSlot；SELF Vulkan 原生状态 |
| Graph 冷规划、资源复用与低成本重放 | 复合功能；固定 shape DAG | 路线 MISSING | SELF 冷 lowering；不建统一热执行器 |
| 模型格式、调度与跨 GPU 策略 | 应用策略 | OUT_OF_SCOPE | APPLICATION |

AUDITED 前还须固定普通 C++ 调用、资源与失败边界、默认规则、有限 shader/格式域、交付 shape、
完整成本账单、代码/测试 owner 和有限 FIX。此卡未授权在 SCOPED 状态下写产品代码。

## 验证状态

所有 R6 产品门仍为 UNPROVEN。注册 QUEUED 身份的 selector 原始计划保存在
`.scratch/validation/vulkan-r6-identities-plan.json`；中央 manifest 触发的 broad build gates 未执行，
不计 PASS，也不为只有身份登记的改动另开一批重型构建。真实依赖/target 接入后按冻结候选的
实际影响面执行与复用证据。当前 docs/closure 与状态迁移检查通过；R5 的运行中验收保持独立。

## 有限审计检查点

[源码/编译器/环境记录](../../../../evidence/research/vulkan-r6-source-admission/README.md)已取得：
Windows RTX 4070 与 Air MoltenVK 的实际版本/能力；四个原样 F32 shader 变体编译成功。
GLSL 的扩展声明不等于最终 SPIR-V 必需能力，且部分 F32 输入的 cooperative-matrix 变体内部转为
FP16，须分开数值合同。两环境尚缺已枚举的 Khronos validation layer；Mini/WSL 原生条件继续按
本卡定向准备。这些不计产品完成，普通/native API 与生命周期及有限 FIX 尚未冻结，维持 SCOPED。
