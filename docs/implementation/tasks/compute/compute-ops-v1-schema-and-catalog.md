# Compute Ops v1：可扩展 typed schema 与推理算子目录审计

## Closure card

- component / owner：`compute_ops` / Compute；round=`compute-ops-v1-schema-and-catalog`；
- baseline commit：`322bc06e`；
- trigger 与证据：当前 [`operation.h`](../../../../include/hyle/compute/operation.h) 只有 `RmsNorm`、
  `MatMul`、`SiluAndMul` 三个内建操作，`Operation` 是封闭 kind 加单个参数字段，shape inference 只能返回
  一个输出。ggml `e91ded11` 的公共 op catalog、llama.cpp `e71b8051` 的模型图、vLLM `51da0ca6`、
  SGLang `dcebe8c4` 与 ds4 `f62ca29a` 的正常推理路线共同反复需要 tensor transform、reduction、
  embedding/RoPE、attention、quantized linear、MoE 和 recurrent/vision primitive；这是完整推理纵切的
  功能 blocker，不是由 benchmark 行数产生的需求；
- 正常 consumer 任务：普通 C++ 用户以 typed operation 描述模型无关张量计算，先检查/推导输入输出、
  dtype/layout/alias/workspace 约束，再交给 direct Compute 或 Graph；不接触 provider 字符串属性袋、模型
  名称、第三方 tensor 指针或热路径动态 registry；
- 主产品纵切 / 具名来源：量化 decoder 的 embedding/gather → norm → QKV/quantized linear → RoPE →
  attention → residual → gated MLP → logits/reduction/top-k；同时覆盖 MoE dispatch/combine 与 vLLM/SGLang/
  ds4 已出现的 recurrent/SSM primitive；
- 正交反偏置纵切：批处理数值/图像管线使用 copy/fill/cast、broadcast elementwise、layout/view/materialize、
  reduction、GEMM/batched GEMM、convolution/pooling 和 selection；验证公共 schema 不含 token、KV、expert
  policy、GGUF、PyTorch 或某一模型字段；
- 跨模块 public journey 与每段 owner：`compute_ops` 拥有 Tensor/storage-format schema、typed Operation、
  output/alias/effect/precision 合同与纯 shape validation；现有 `compute` 拥有 Context/provider/prepare/
  invocation/submit；Graph IR 直接消费 schema 并拥有跨节点 symbolic constraint 与 topology；Mem 拥有
  storage bytes，SIMD/provider/backend 拥有具体 kernel；模型、KV/batching/sampling policy 留在引擎；
- scope：完成全部现有 Tensor/Operation 入口的 `KEEP/MOVE/REPLACE/DELETE`，冻结可增长且无字符串属性袋的
  typed operation value、零/多输出 inference、alias/effect 与 precision metadata；建立版本化 capability-family
  map，并只实现该表最终标为 `MUST + FIX/BORROW` 的有限首批目录及 reference correctness；把 schema 从
  runtime factor 移到本 factor；
- non-goals：一次复制 ggml/PyTorch 的全部 op 枚举、训练/backward/optimizer、模型 layer/KV/page table/
  sampling/batching/serving policy、provider kernel 性能、Graph topology/partition、backend runtime、Native FFI、
  字符串或裸 `void *` custom-op 属性、为未实现能力预留成功 stub；
- 直接依赖 / 直接消费者：候选公开依赖 Core/Error/Mem，私有依赖 Math 做 checked shape arithmetic；不依赖
  Compute runtime、SIMD、Graph 或具体 provider。直接消费者为 `compute`、Graph IR、HYLE-native、oneDNN、
  ggml 及后续 backend/provider factors；
- 性能 tier 与原因：schema 本轮为 `P0`；要求 validation/inference 对 rank 不超过 8 为有界 O(rank)，常见
  built-in descriptor 不做 heap allocation、锁、全局 lookup 或字符串 dispatch。kernel/provider P1 由其实际
  owner 在固定 op 合同后签发，不能拿 schema 空调用冒充算子 SOTA；
- 适用机器 / 仪器 / material margin：三平台编译与 correctness、两台 Apple header/package；schema 不做
  纳秒 P1。后续 provider 卡的 material margin 独立冻结；
- 调查与纠正预算：四类推理来源各一次定向操作序列扫描；ggml op/schema 与 oneDNN primitive/descriptor 各
  一次 strong-peer 扫描；每个能力族最多两个 peer；schema/API 最多两轮；有限 FIX 每项最多两轮；
- 预声明验证：每个准入 op 的 normal/zero/shape/dtype/layout/stride/broadcast/overflow、output arity、alias/
  effect、precision、unsupported 与 OOM；reference/differential/property；逐头、三平台、sanitizers、source/
  static/shared、factor-only/product façade、Graph schema consumer、P0 layout/codegen/成本账单和终局红队；
- stop / blocker 条件：若一个候选 op 仍带模型或 backend 特有策略，降为 provider/private/application 而不是
  污染公共 catalog；若安全的开放 typed extension 必须让常见 built-in 支付常态 allocation/虚调用，则先保留
  编译期 built-in schema并把 extension 分离；任何未审计 kernel 不得因 enum 已存在被宣称支持。

## SCOPED 能力族

| 能力族 | 当前证据 | 当前覆盖 | 本轮待裁决 |
|---|---|---|---|
| tensor dtype/storage/view 与零拷贝变换 | HYLE v3、ggml tensor | `PARTIAL` | reshape/slice/broadcast 与 materialize 边界；host/device storage 由 runtime 卡拥有 |
| copy/fill/cast/layout/repack | 五个推理来源与 ggml | `MISSING` | descriptor-only view 与执行 op 分离，quantize/dequantize/repack 的 typed 参数 |
| unary/binary/broadcast/reduction | ggml、llama.cpp、vLLM/SGLang | `PARTIAL` | 通用 family 与具名数值语义，避免每个变体膨胀成字符串 |
| normalization/activation/residual/gated | 五个推理来源 | `PARTIAL` | LayerNorm、fused add+norm、GELU/SiLU/GLU 档位与 alias 语义 |
| GEMV/GEMM/batched/quantized linear | 五个推理来源、oneDNN | `PARTIAL` | bias/epilogue、accumulation、scale/zero、multi-output packing owner |
| embedding/gather/scatter/position/RoPE | 五个推理来源 | `MISSING` | index dtype、bounds、in-place Q/K、多轴与 position inputs |
| mask/softmax/attention primitive | 五个推理来源 | `MISSING` | causal/general mask、SDPA outputs、precision/workspace；KV policy不下沉 |
| selection/sort/top-k | ggml、llama.cpp、serving logits | `MISSING` | values+indices 多输出；sampling policy 不下沉 |
| MoE data movement/compute primitive | vLLM、SGLang、ds4、llama.cpp | `MISSING` | dispatch/permutation/combine 与 grouped matmul；expert policy 不下沉 |
| convolution/pooling/SSM/recurrent primitive | ggml multimodal、vLLM/SGLang、ds4 | `MISSING` | 只准入模型无关、跨来源重复的机制；状态调度留给引擎 |
| typed custom operation | accepted architecture、provider extension | `MISSING` | schema identity、copy/lifetime、inference/capability，无全局 registry |

## SCOPED 形状问题

1. 单输出 `infer_output_description()` 如何替换为 caller-owned、零/多输出合同而不在常态 heap allocation；
2. 封闭 `OperationKind` 与只保存 `RmsNorm` 参数的 `Operation` 如何支持大量 built-in 和 typed extension，
   同时让常见描述值保持小、可复制、无 process-global registry；
3. op descriptor、schema introspection、provider capability 与 prepared artifact 的边界；
4. view-only reshape/transpose/slice/broadcast 与真正 copy/repack operation 的自然区分；
5. quantized storage identity、scale/zero/bias tensor、accumulation/precision 和 provider-specific packed artifact
   各自 owner；
6. 多输出、in-place、partial alias、side effect、resource write 与 Graph dependency 所需的最小 schema；
7. 首批 `MUST` 的实际数量与分批顺序；未实现条目不得仅添加 enum 或文档承诺。

AUDITED 前不移动现有 Tensor/Operation 产品代码，不增加算子、依赖、公开头或产品投影。
