# Compute v7：完整 typed operation façade

## Closure card

- component / owner：`compute` / Compute；round=`compute-v7-typed-operation-facade`；
- baseline：`0e55b5be`；v6 terminal 为
  [compute-v6-provider-runtime-ownership](compute-v6-provider-runtime-ownership.md)；
- reopen trigger：`compute_ops` 已有 19 个 typed family，Native 也已实现后续 family，但
  `prepared_operation.h` 的 convenience 仍停在早期目录。Gather/Scatter/Concat/RoPE/Softmax/
  TopK/SDPA 和 separate GatedActivation 要求普通用户手写 descriptor span；TopK 的双输出、
  SDPA 的可选第四输入和 variadic Concat 也没有自然 submit 形状。这是已实现能力的
  可复现 C++ 用户旅程缺口，符合 v6 的“公开 runtime 合同变化”重开条件；
- 正常任务：用户用 typed operation 和 TensorDesc 准备一个常用推理操作，让 schema 推导
  output，然后以自然 arity submit；只有真正 variadic 或动态的代码才使用通用 span 入口；
- scope：补齐 19 family 的 typed prepare convenience，消除 RMSNorm/MatMul/Gated 早期显式
  output 形状，补 TopK/SDPA/variadic 常用 submit convenience，迁移 HYLE 内部直接消费者；
- non-goals：修改 Operation schema/provider SPI/Graph IR、增加算子、保留未发布旧 overload、
  Native FFI、为 benchmark 增加入口；
- 依赖 / 消费者：只依赖已有 Core/Error/Mem/compute_ops/Compute runtime；直接消费者是
  Native direct Compute、Graph 示例/测试和未来推理 runtime；
- 性能：`P0`。所有 convenience 只在冷 prepare 路线构造有界 stack array 并调用同一
  generic prepare；submit overload 只构造 span 并调用同一 lane，不改 provider steady path；
- 预声明验证：所有 overload 的 output inference/resource 传播/arity/error；Native 全部 typed journey；
  Graph 直接消费；逐头；Windows/Linux/两台 Apple；sanitizer；source/static/shared 搬迁
  consumer；P0 源码账单；终局红队；
- 预算 / stop：一轮 API ledger、一轮实现、一轮红队；如果 output inference 需要新 owner/
  allocation 或 overload 产生不可解的 C++ 歧义，退回 `SCOPED`，不加 tag/legacy 兼容层掩盖。

`SCOPED` 阶段只做公开入口与消费者对账，不修改 Compute 产品头或实现。
