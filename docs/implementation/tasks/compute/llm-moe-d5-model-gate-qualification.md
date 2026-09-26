# D5：原版模型门控的完整片段资格

<!-- MODULE REVIEW: AUDITED -->

## Closure card

- owner：`compute_ops`；接线候选 owner 为四路线现有 ggml Provider 与必要的 Apple/Vulkan 分类。
- baseline：`3570c943f4bc35d03156342a74919bd0d70918c7`，Windows 工作树干净。
- trigger：用户已明确选择[数值待决卡](llm-moe-d5-recurrent-gate-numeric-decision.md)的
  “完整片段误差资格化”。这是该卡恢复条件的实际满足，不是自动解除历史 BLOCKED。
- 正常 consumer：Qwen GDN 消费 raw a/b、dt_bias 和负 decay coefficient；准备模型专用 gate
  后，反复把结果输入成熟 GDN 更新，观察输出及连续状态。普通 Softplus 的精度合同保持不变。
- 主纵切：本地官方 Qwen3.5-2B layer0 的 A_log/dt_bias；反偏置：27B K16/V48、2.4T K16/V128，
  官方 grouped-head 与 GGUF cyclic-head，D128。巨大模型仅使用已批准的真实形状合成 payload。
- scope：原版 ggml gate -> GDN -> output RMSNorm/SiLU gate 的有限数值资格；通过后审定单个
  模型专用 gate schema、原生 lowering，以及九个 Qwen profile 的共用连通 R05 消费者。
- non-goals：修改普通 `Unary::kSoftplus` 精度、给同一请求偷偷降精度、新 HYLE 数学、升级 ggml、
  新 Provider、重构路由架构、整模型加载、视觉分支、其它模型的提前性能裁决。
- 依赖：当前固定 ggml CUDA 原版程序及既有四路线资源/完成/失败设施；不用新 kernel 包装 helper。
- 性能：本次数值资格 `P0`，不做计时优选；原生 gate region 的融合/中间量计入 D5 稳定后 P1。
- 预算：一次预声明数值批次；允许一次探针/接缝实现纠正，不能修改数值阈值。批量接线后一次
  受影响 owner/consumer 验证，旧不变证据按身份复用；三交付与 P1 随 D5 稳定批统一运行。
- stop：数值门失败且非探针接线问题时记录精确域并停止；通过仅授权相应已审计模型专用接线，
  不把 probe PASS 写作普通入口/完整画像 PASS。

## 运行前冻结的数值门

1. gate 的原生公式为 `g = negative_a * softplus(a + dt_bias)`，`beta = sigmoid(b)`。
   `negative_a` 是独立的 F32 参数；官方从 A_log 转换，GGUF 已存转换值。
2. 一组读取本地真实16头参数；形状反偏置使用16/48/128 V头，K16、D128，alpha强度在
   `{1/256, 1/16, 1/2, 1, 4, 16}` 中确定性轮换，不宣称读取其它巨型 checkpoint 的参数值。
3. 每组64次变化输入，覆盖 `a+dt_bias` 的 `-80,-40,-20,-15,-8,-4,-1,0,1,4,20,40` 及普通值，
   `b` 包含正/负饱和与中间值；保留官方与 GGUF 的 head 映射。原生程序反馈它自己的 F32 state，
   独立参考反馈参考 state，不能每次把 oracle state 填回被测程序而掩盖累积误差。
4. 参考使用稳定 `log1p` / sigmoid 与独立 F64 recurrent 更新，在声明的 F32 storage 边界舍入。
   Q/K 已 L2 归一化；同时观察 `g/beta`、GDN output、完整 state、RMSNorm/输出门控结果。
5. **所有有限观察沿用已有 R05 门：`abs_error <= 2e-6 + 4e-5 * abs(reference)`。**
   gate 接近零时不另加相对误差门；输出/状态仍按各元素判定。记录最大绝对误差与门限比值，
   任一超过1即 FAIL，不能只报告平均误差。NaN/Inf 不能被有限误差比较误签为通过。
6. 这是声明片段和输入族的工程数值资格，不是所有权重数值、所有上下文长度或整模型精度证明。
   未验证形状/精度域不能自动扩张；源自公开模型的精确身份仍由既有 `WorkloadContext` 冷绑定。

## 接线前审计项（原拟 gate 接口已否决）

资格通过后固定一个具名模型 gate 操作，显式区别于通用 Unary。候选自然调用是
`model_ops::recurrent_gate()`，四个 F32 端口 a/b/negative_a/dt_bias、两个输出 g/beta；
复用现有 Custom Operation schema，不新增普通路由 metadata 或逐 kernel vtable。
准备期检查 rank、shape、storage、输出数、容量和 alias；原生 Provider 保留组合/融合权。
候选端口与 owner 的最终有限 FIX 必须在产品修改前写入 AUDITED。

三交付、资源/失败检查与 D5-A5/A6 的适用门保留；本卡不会以数值探针代签这些门。

## 数值审计与最终公共边界

原拟独立 `recurrent_gate()` **REJECT**，不创建此接口：原版 CUDA gate 的少量中间 g 在反偏置
输入中超过上述门，最大门限比 `1.00884305`；原始探针整体 verdict 保持 FAIL，不能改写。
本地真实16头参数的全部观察通过；六组64步中的 GDN output、完整实际反馈 state、后续归一化/
输出 gate 均通过同一门，最大门限比分别约 `0.00335 / 0.02940 / 0.30907`。

最终选择用户已批准的**完整模型片段边界**：`model_ops::recurrent_gated_delta_net()`。
这是不同的公开语义，直接对应固定 SGLang `cutedsl_gdn.py` 已有的 raw gate + recurrent update
入口，以及 llama.cpp 的同一原生组合。它只返回 attended output 和 next state，g/beta 是 Provider
内部量；不公开一个没有合格数值保证的 gate。所有公开输出门限仍为 `2e-6 + 4e-5*abs(ref)`，
不放宽阈值；此前 probe 的中间量失败保留为不能独立发布 gate 的证据。

### AUDITED 有限 FIX

| ID | owner | 固定动作 | 必验边界 |
|---|---|---|---|
| MG1 | compute_ops | 添加一个 Custom Operation schema；八个 F32 输入 Q/K/V、raw a、raw b、negative_a、dt_bias、state；两个输出 attended/state | 复用既有 GDN shape inference；scalar gate、plain dense、参数 `[1,1,H,1]`；拒绝不匹配 shape/type/arity |
| MG2 | 四个 ggml Provider | 原版 Add/Softplus/Mul/Sigmoid 连接原版 GDN，复用现有输出拆分和执行资源 | 不改 Unary 准入、不补数学、不公开 g/beta；按各路线原生容量与失败规则准备 |
| MG3 | Apple Direct/Graph、Vulkan Graph | 已有原生分类增加此 schema；Vulkan 仅为此成熟 schema 允许两个 custom outputs | 未知 custom、多输出或错误 alias 仍拒绝；不重构公共路由 |
| MG4 | ggml Metal | 八输入加一个 packed output 的私有端口容量；四个 gate 中间 tensor 的原生冷期分配/完整释放 | 不在 submit 分配 workspace；backend 销毁前释放；沿用 checked completion |
| MG5 | 测试消费者 | 共享 R05 recipe 改用完整 region；批量扩展 Qwen 九画像并检查实际状态反馈 | 四路线 Direct/Graph、变化输入、原独立 oracle、资源和普通 Unary 拒绝；其它片段按身份复用 |

公共语义沿用 GDN 的 grouped-head 对应；官方与 GGUF 的 head 重排仍由已经明确的模型边界表达，
不隐藏转换。初始准入限定已资格化的 F32、D128、K16或等头、V16/48/128、单 token/sequence；
不以一个有限探针覆盖任意模型、token容量或低精度。参数为负 decay coefficient，Q/K 为模型已
归一化输入，数值合同的有限输入族见前节；普通通用 GDN 仍保留其原有功能域。

Metal 当前原生 one-op bridge 的七端口与无中间 tensor 假设是具名接缝：此 FIX 扩为九端口，
只为新 region 冷期取得四个 native scratch buffer，随 Program 保活/释放。CPU/CUDA/Vulkan 已有
region workspace 管理，直接复用。准备只绑定，不计时选择；重复执行不重建图。

本轮其他 owner 原有 AUDITED 卡在[D5 主卡](llm-moe-d5-real-fragment-acceptance.md)增加 F23
重新定界，既有未签三交付/P1 不丢失。先完成全组接线再增量编译；数值探针不是 profile RUN_PASS。
