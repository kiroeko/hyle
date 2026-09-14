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
- scope：补齐 19 family 的 typed prepare convenience，消除 RMSNorm/Gated 冲突的早期显式
  output 形状，为 MatMul 增加自然推导形状并保留不冲突的显式-layout 高级入口，补
  TopK/SDPA/variadic 常用 submit convenience，迁移 HYLE 内部直接消费者；
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

## AUDITED 结论（2026-09-14）

逐项对账 19 family、现有 overload、Native direct tests 和用户 guide 后，底层通用
`prepare(context, Operation, input-span, output-span, resource)` 已能表达所有语义；缺口只在公开
C++ façade。输出推导是 `compute_ops` 已冻结的纯函数，不需要新 owner、provider 或分配器。

### 入口裁决

| 能力 | 现状 | 裁决 |
|---|---|---|
| View/Copy/Fill/Cast/Unary/Binary/Select/Reduce/LayerNorm | 已自然推导 output | `KEEP` |
| RMSNorm | 早期 overload 要求用户传 output，且与自然 `(input, weight)` 签名冲突 | `REPLACE`：只保留推导形状的 input 与 input+weight convenience；显式 output 走通用入口 |
| MatMul | 只有显式 output | `FIX`：增加 `(left,right)` 推导形状；显式 output 高级入口不冲突，保留 |
| GatedActivation | 只有 packed+显式 output，与自然 separate 签名冲突 | `REPLACE`：packed 一输入和 separate 两输入都推导 output |
| Gather/Scatter/RoPE | 只能手写 span | `FIX`：按固定 arity 补 typed prepare |
| Softmax/SDPA | 无 mask 与显式 mask 都只能手写 span | `FIX`：按 typed mask mode 提供 1/2 和 3/4 input overload，schema 仍检查 mode/arity |
| TopK | 两个 output 只能手写 span | `FIX`：prepare 在 stack 上推导两个 descriptor，submit 提供 input+values+indices |
| Concat | 真正 variadic | `FIX`：prepare 接收 input descriptor span，submit 接收 input-view span+单 output |
| SDPA 四输入 submit | 只有三输入 convenience | `FIX`：增加 Q/K/V/mask+output；不增加模型或 KV 字段 |

### 有限 FIX ledger

| ID | 允许改动 | 验收 |
|---|---|---|
| C70 | 在 `prepared_operation.h`/实现补齐上表 typed prepare，删除冲突的早期 RMSNorm/Gated 语义 | 19 family 都有与真实 arity 匹配的自然冷路入口；无 legacy tag/alias |
| C71 | 增加 variadic-single-output、single-input-two-output 和 four-input-single-output submit convenience | 直接 Concat/TopK/masked-SDPA 不需要用户构造 output span |
| C72 | 迁移库内 direct consumer 和 guide recipe | 无旧语义调用，清楚区分 single-resource convenience 与 heterogeneous-resource generic route |
| C73 | 补 overload 精确选择、output inference、error/resource 和 Native direct 回归 | Windows/Linux/Apple 及三交付不存在歧义/导出差异 |

P0 账单冻结：typed prepare 最多构造 rank-bounded descriptor 及 4 个 stack binding；只有用户
显式选择的 variadic Concat 超过通用 inline binding 容量时由冷路 metadata allocator 分配。submit
overload 只构造 `std::array`/`std::span` 并进入同一 `OperationInvocation::submit`；provider 选择、
mapping、kernel、Completion 和 steady allocation 账单不变。有限改动集已固定，可进入实现。

## Candidate（2026-09-14）

- immutable product commit：`18970739c5e0909cf968fecba9b96c9ea3560dfc`；C70--C73 的公开
  overload、内部 direct consumers、package sample 和用户 guide 已停止变化；
- 19 个 family 均可从真实 typed arity 推导 output；RMSNorm/GatedActivation 不再要求普通用户重复传
  output descriptor，MatMul 保留不冲突的显式 output 高级入口，异构 resource 仍使用既有 generic route；
- fixed-arity prepare/submit 只建立至多四项的 stack `array/span`，然后进入同一 generic runtime；没有新增
  steady allocation、provider/ISA search、锁、原子或复制，故本轮性能层级仍为 P0；
- Windows MSVC 19.50 Release/WERROR 的 Compute、ComputeOps、CPU backend、Native 与 Graph 共 6 个定向
  测试，以及 4 个受影响逐头测试通过；文档与 closure checker 通过。候选后只执行卡片已声明的
  Linux/两台 Apple、sanitizer 和 source/static/shared 交付回放；发现产品问题必须退回 `AUDITED`。
