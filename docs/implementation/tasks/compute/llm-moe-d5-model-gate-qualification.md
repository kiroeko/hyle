# D5：原版模型门控的完整片段资格

<!-- MODULE REVIEW: SCOPED -->

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

## 接线前审计项

资格通过后固定一个具名模型 gate 操作，显式区别于通用 Unary。候选自然调用是
`model_ops::recurrent_gate()`，四个 F32 端口 a/b/negative_a/dt_bias、两个输出 g/beta；
复用现有 Custom Operation schema，不新增普通路由 metadata 或逐 kernel vtable。
准备期检查 rank、shape、storage、输出数、容量和 alias；原生 Provider 保留组合/融合权。
候选端口与 owner 的最终有限 FIX 必须在产品修改前写入 AUDITED。

三交付、资源/失败检查与 D5-A5/A6 的适用门保留；本卡不会以数值探针代签这些门。
