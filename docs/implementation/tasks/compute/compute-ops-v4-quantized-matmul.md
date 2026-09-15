# ComputeOps v4：R2 grouped Q4 MatMul 语义

<!-- PROJECT UNIT REVIEW: CANDIDATE -->

## Closure card

- component / owner：`compute_ops` / ComputeOps；baseline `25281c98`。
- trigger：[Compute Q4 合同](../../../architecture/compute-design.md#cuda-q4-matmul-的规范例)与
  [R2](four-route-common-semantics-v1.md) 要求 scale、可选 affine bias、group 和数值合同；旧 MatMul 只有两输入。
- 当前 v3 AUDITED 重新定界；[v3](compute-ops-v3-resource-access-schema.md) 的有效 shape/custom/resource
  机制保留，旧未完成交付尾项由本轮直接消费者覆盖，不能把历史 Windows 测试视为新终局。
- 主 consumer：已登记 MLX Q4 FFN 的 `[M,K] × [N,K]` 权重、`[N,K/group]` scale/bias。
  正交 consumer：signed symmetric weight 的无 bias 线性计算及批量 leading dimensions；不是模型运行器。
- 承载：SELF 最小中立 schema/校验；packing 和实际计算 BORROW 对应路线实现，图依赖/资源 ROUTED Graph。
- scope：Q41/Q42；non-goals：kernel、repack、FP4/GGUF 目录、混合精度隐式转换、g_idx/任意 group map。
- 依赖：既有 Core/Error/描述机制；直接消费者 Compute、Graph、Native/oneDNN。无新第三方或组件。
- P0：冷 schema 与 O(rank) 描述推导；无 tensor payload 访问/分配/搜索；既有提交算法和状态不改。
- 预算：定向审计 20 分钟，实施及普通验证 90 分钟，一次有据纠正；超出则明确退回 AUDITED/记录 blocker，
  不追加研究或全库验证。R2-Q 完成不代表 R2-D/G 或四路线完成。

## 有限 FIX 与签名

| ID | 缺口 / 允许改动 | 验收 |
|---|---|---|
| Q41 | 新增 `QuantizedMatMul` builtin 和 `QuantizedWeightMode`；只支持 grouped 4-bit signed scale / unsigned scale+bias，复用 StorageFormatId | typed Operation/traits/name、精确 arity、跨 DSO 值拷贝；不让普通 MatMul 自动解码 |
| Q42 | 冷验证 dtype/shape/group/packed format 并推导输出，更新真实 Graph 描述消费和指南 | 正常 M=1/7、batched leading dims、group/参数/格式/类型错误、overflow/空输出；合法 schema 不暗示 Native 已有 Q4 kernel |

API freeze：`QuantizedMatMul{.mode, .group_size, .accumulation_type}`，mode 默认 signed symmetric Q4，
group_size 默认 64，accumulation_type 默认 float32。logical weight 为 `[N,K]`，input 为 `[...,K]`
且 rank>=2，结果 `[...,N]`；scale 与可选 bias 为 `[N,K/group_size]`，K>0 且整除 group。
signed mode 的数学权重为 signed q（-8..7）乘 scale；affine mode 为 unsigned q（0..15）乘 scale 加
additive bias。physical format ID 单独定义字节编码，不能从位宽猜测 signedness、零点或 packing。

activation、weight 的 logical dtype 与参数 dtype 相同，限 float16/bfloat16/float32/float64；
累加类型显式为合法浮点类型。输出沿 activation dtype。合同允许该累加类型下的正常运算重排，
不承诺与先物化浮点权重的逐位结果相同；Provider 须另证数值域，不得隐式舍入 activation。
本轮不提供任意 dequantization/量化精度档位，无法兑现此合同的 carrier 必须拒绝或另卡扩展语义。

Schema 只借用 descriptor，不持有 payload/Queue/owner；失败在 prepare/Graph inference 前返回 Status。
推导不运行工作、不申请资源，无取消/晚到错误/并发回收机制；这些风险在执行 owner 验收。

## 预声明验证与七问

Windows：ComputeOps 单测/逐头与 Graph 描述消费，selector 选中的真实直接消费者；source/static/shared
及 relocation 复用现有 selected Compute/Graph package 入口。WSL/macOS 采用相同候选验证适用 schema/
直接消费，ComputeOps 的 v3 custom identity/resource tail 一并回放；不启动无关 all-components。
本 schema 没有热路径改动或新性能赢家，P1 不适用；R3–R6 的自然执行 P1 不得由本卡替代。
指针/长度/overflow 通过边界测试及适用 UBSan；不存在新并发共享状态，TSan 不为此冷值 schema 单独启动。

1. 第三方版本/许可不变；MLX/Marlin 只是既有具体语义依据，未在本卡接入 kernel。
2. 完整端口是实际量化消费所需；只扩一族必要 schema，不由 benchmark 行驱动目录。
3. 复用 TensorDesc、Error、typed Operation 和 Graph 推导；不重开已关闭底层。
4. O(rank) 冷验证，无 payload、常态分配/锁/同步；候选确认 Operation 大小及热态结构。
5. C++23 同工具链交付，无 C/Native FFI 或静态地址身份。
6. 仍由 ComputeOps owning factor 导出头/实现；三交付相同 descriptor。
7. 普通用户构造同一 typed operation，具体路线命名 packing factory；没有 Provider 优先级/profile 输入。

## 当前状态

SCOPED 已在 `9bf63d15` 提交；当前完成定向审计，Q41/Q42 与上述签名固定，进入 AUDITED 并绑定
compute_ops。检查 `operation.h` 的 enum/variant、`compute_ops/operation.cc` 四处语义分派以及
GraphBuilder 的通用 Operation/span 入口：只需改 schema owner 产品代码与直接消费者测试。
Native 对新 kind 明确 Unimplemented，oneDNN 只接收普通两输入 MatMul；无需改其已审计 runtime。
旧 Graph profile 对未知编码明确拒绝；不新增 Q4 计时编码，该过渡公开面归 R2-G 清理。
现有 TensorDesc checked 尺寸/字节数和 blocked/opaque 格式足够承载描述，不新增 storage/token。
审计未读/改其它历史任务或底层模块。最终七门、候选、实际限制及统一 LOC 在实施后记录。

## 候选验证检查点

`50753347` 已实现 Q41/Q42。Windows selector 的 14 个适用门通过（含复用的 13 门与最后 Graph
三交付），新增 Q4 consumer 的 source/shared/static 也通过。MSVC 显式环境、候选/合同/工具链/机器
身份与原始日志在本轮 scratch；最终归档在终局记录。Visual Studio 的嵌套 `/MP` 曾超过两编译进程，
已中止该 Graph 门并以 Ninja 两任务串行完成；中止日志保留，不作 PASS。

WSL GCC 14 的 Werror 首次回放发现五处测试告警：四处 range-loop copy（包括继承的 View/resource
测试）及一处 Graph 旧 GTest 宏外缺 braces。使用本卡一次纠正预算，只改测试引用/花括号；产品头及实现
保持相同。修正候选只回放受影响测试和未完成平台门，不重建已通过 SDK。当前仍 AUDITED，WSL/macOS/
UBSan 未完成前不签终局 PASS；Direct 的独立实现由 R2-D 继续。

## 终局验收准备

产品候选 `50753347`，修正测试候选 `210044ce0b8b4df0ab6b796e00331536d3f0263d`，原始材料见
[Q4 证据](../../../../evidence/correctness/compute-ops-v4-quantized-matmul/README.md)。各平台最后的适用门
已完成；保留所有失败/中止记录，未重建已通过 SDK、未运行 P1。进入 CANDIDATE，下一步只作状态终审。

| Gate | Verdict |
|---|---|
| scope | PASS：Q41/Q42 与继承的 custom/resource/View 验证尾项，未接入 kernel |
| architecture | PASS：完整数学/端口/数值/格式描述；复用 TensorDesc、Error 与 GraphBuilder |
| correctness | PASS：Windows/WSL/两 Mac 单测/逐头、shared producer，WSL UBSan；候选测试纠正已回放 |
| delivery | PASS：Windows source/shared/static 与 relocation；Unix 同候选 schema/Graph/custom/resource 直接消费 |
| usability_docs | PASS：公开 Q4 描述/拒绝边界明确；docs、closure、diff 检查通过 |
| red_team | PASS：错误端口、mode/group/K、布局/参数、zero/overflow、失败不发布输出；旧机制继续消费 |
| engineering_performance | PASS（P0）：新增冷 O(rank) 校验；W1 Operation 仍为 128 bytes/alignment 8，既有 submit 未改；本卡 P1 不适用 |

统一 LOC：Compute+Graph **31,220**；自实现不含测试 **126,405**；全库 **691,833**，自检通过。
限制：量化默认实现、native packing、执行及完整路线性能在 R3–R6 验收；描述通过不等于 kernel 可执行。
reopen：公开 schema 推导/格式/跨 DSO 发生真实回归，或路线实施发现当前数学/端口合同无法表达必要消费。
