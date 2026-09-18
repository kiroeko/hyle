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

SCOPED 登记提交为 `17674262`。下列审计与有限 FIX 提交后才实施产品修改。

## AUDITED：有限 FIX 与冻结决定

2026-09-18，owner `/root`。共同 `validate_accumulation` 允许 Unknown 或数值类型，
`OperationPreparationAccess` 只核对 schema/资源，不代表具体 Provider 实现支持该精度。
oneDNN 的提案与准备共用 `select_route`；Native 两条路径共用 `make_plan`。本轮在各自共享
入口修复，避免只挡公开 `prepare` 而让 Graph 冷提案或精确准备旁路。

| FIX | 允许改动与验收 | 当前裁决 |
|---|---|---|
| Z01 | oneDNN `select_route` 只接纳显式 F32 accumulation；F64/F16/BF16/I32/Unknown 返回 Unimplemented。测试提案、普通 prepare 和复用 F32 提案但改变精度的精确 prepare 均拒绝 | FIX；第三方 BRGEMM 原样 BORROW |
| Z02 | Native MatMul 的共享计划入口执行同一资格判断；普通 CPU 默认工厂不能在 oneDNN 拒绝后落到不合格 Native。回归覆盖解码/预填充及转置权重域，既有 F32 数值测试必须继续通过 | FIX；既有 SIMD/计算实现 KEEP |

未知累加类型没有被这两个 Provider 声明为可自由选精度的域；默认 `MatMul{}` 已明确 F32。
普通 C++ API、布局、资源所有权、运行状态和错误类型不变：合法但未实现的语义返回
`Unimplemented`，非法描述仍由共同 schema 拒绝。只增加冷 enum 判断，不增加 payload 检查、
热路径分支、分配、锁、拷贝、同步或常驻字段；F32 计算与既有路由形状阈值保持原实现。

七问裁决：① oneDNN 3.13.2 pin/许可证/升级边界不变；② 具名显式精度请求不再静默降级，
不承诺新增 F64 实现；③ 复用共同 schema/准备与既有 SIMD，不复制基础设施；④ P0 冷检查，
没有同合同热路径变化，不跑 P1；⑤ 不增加 C/FFI；⑥ 私有 C++ 两处判断，无头/CMake/package
变化，原三交付形状证据复用，static/shared 对新实现重跑；⑦ 默认 CPU/专家入口都有回归，
用户 guide 明确有限 F32 域。测试先在未修产品上确认失败，再以同一测试验证修复。
