# C1-Z：CPU MatMul 累加精度准入修复

## Closure card

- owner：`/root`；`compute_provider_onednn` 定向重开，`compute_provider_native` 沿
  [C1 活动卡](llm-moe-c1-tensor-state.md)追加有限 FIX，不重开 CPU/Graph route 或 SIMD。
- baseline：`55830705d95150ecfd71a4d49798c07457dc379c`。
- trigger：共同 `MatMul` schema 接受数值累加类型；oneDNN Direct `select_route` 与 Native
  `make_plan` 的 MatMul 分支却没有匹配 `accumulation_type`，实际实现只使用 F32。
  同样的 F32 `[16,5] * [5,4]` 描述配 `MatMul{DataType::kFloat64}` 因而可能被错误接纳。
  oneDNN Graph、cuBLASLt、Apple 与 Vulkan 的 MatMul 已拒绝非 F32 累加，不能靠下游掩盖 CPU
  普通入口的缺陷。先用回归重现接纳错误，再修准入；尚不声称已运行数值反例。
- 前轮 terminal：[R6 交付](../build/vulkan-r6-product-delivery.md)；本轮满足其可复现语义缺陷
  reopen trigger，原交付/P1 事实保留，不把旧 CLOSED 当本轮验收。
- 主 consumer：普通 CPU Direct/Graph 显式指定矩阵累加精度，候选必须保留该语义或明确拒绝。
  正交 consumer：专家限定 oneDNN/Native 或冷提案与精确准备分离；不能在 fallback 中降精度。
- scope：两个 CPU Provider 的 MatMul 冷准入与相关查询/准备/默认入口回归。
- non-goals：新增 F64 GEMM、改矩阵算法、扩 C2、改 SIMD、重排性能规则、重审其他算子或治理工具。
- 依赖/消费者：现有 Compute schema/preparation、CPU backend、oneDNN 3.13.2；直接消费者为
  CPU route 和 Graph oneDNN。只修实际拥有数值能力的 Provider，不在共同 schema 禁止合法类型。
- 性能：P0。一次冷 enum 判断；成功的 prepared invocation、内存布局、计算/同步/资源成本不变。
  此修复不进行 P1，不把缺检查实现的旧性能当作非 F32 域的合格证据。
- 预算：源码核对与回归设计最多20分钟，最多两轮具名产品纠正；不扩大为所有 Provider 全库复审。
- 预声明验证：不可变候选的 selector owner/真实一跳消费者；W1 Release/WERROR static/shared
  对应单测及普通 CPU Direct/Graph，WSL x64 对应定向语义回归；DLL 邻置、PATH unset。
  Native 的其他平台完整 C1 资格仍归主卡。公开头、ABI、CMake、vendor pin 均不变时复用原三交付
  形状及安装证据，不重复全 package。若 selector 因实际修改选择更广门，按其结果执行。
- 停止：有限修复与适用门完成后 oneDNN 定向轮 CANDIDATE→CLOSED；Native 保持 C1 AUDITED。
  未运行风险不签 PASS；外部条件确实阻断则给出具体 blocker 与恢复条件。

## 有限待审项

| ID | 具体问题 | 责任边界 |
|---|---|---|
| Z01 | oneDNN Direct 冷提案/准备没有匹配 MatMul 累加类型 | 共享准入函数拒绝本实现不支持的精度；保留 F32、shape/resource 和失败语义 |
| Z02 | Native fallback 存在同一遗漏 | 在 Native 自己的 MatMul 计划资格处检查；不能由 oneDNN 拒绝后静默降级 |

本卡进入 SCOPED 只授权审计。产品实现前提交 AUDITED 的有限 FIX、API/成本冻结及回归门。
