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
- 直接依赖 / 直接消费者：候选公开依赖 Core/Error，私有依赖 Math 做 checked shape arithmetic；不依赖 Mem、
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

## AUDITED 结论（2026-09-12）

### 来源对账与准入边界

定向扫描绑定 ggml `e91ded11bdcd`、llama.cpp `e71b80510c84`、vLLM `51da0ca66c80`、SGLang
`dcebe8c4733a`、ds4 `f62ca29a3087` 与 oneDNN `6b7f1fbbb18e`。共同高频机制是 tensor view/materialize、
elementwise/reduction、normalization/gated activation、dense/blocked linear、embedding/gather、RoPE、
softmax/attention、selection/top-k；MoE、vision 与 recurrent 也真实存在，但其 packed layout/state policy
尚不能在一张 v1 卡内自然定型。

准入规则冻结为：一个 public operation 必须有完整 typed schema、shape/arity/alias/effect/precision 合同，
并在同轮至少取得一个可执行 provider（本轮为 native reference）；只加 enum、名字或恒失败 stub 不算能力。
provider-specific packed artifact 可以由普通 operation + `StorageFormatId` 匹配；模型 layer、KV/page-table、
expert/sampling/batching policy 不进入本因子。

### Schema 入口 ledger

| ID | 当前形状 | 裁决 | 冻结合同 |
|---|---|---|---|
| O10 | TensorDesc/StorageFormat 在 runtime target | `MOVE` | `data_type.h`、`tensor_desc.h` 与 schema-only `operation.h` 归 `compute_ops`；include namespace 仍为 `hyle::compute` |
| O11 | dtype 缺 bool/fp8/常用 16/32/64 unsigned | `FIX` | 补 Bool、FP8 E4M3FN/E5M2、Int16/UInt16/UInt32/UInt64；sub-byte quantization 仍用 blocked format，不伪造 scalar byte size |
| O12 | 默认 Operation 实际是 RMSNorm，payload 只存一个字段 | `REPLACE` | 默认 invalid；built-in 为 enum + 固定 inline typed payload，普通构造/copy 无 heap、vtable、string 或 registry |
| O13 | 单一 `infer_output_description()` | `REPLACE` | `output_count()` + caller-owned exact-span `infer_output_descriptions()`；保留显式 `infer_single_output()` convenience |
| O14 | alias/effect/precision 分散在 kernel/Graph | `FIX` | `operation_traits()` 返回 arity、output alias allowance、purity/effect 和 accumulation policy；provider 只能收窄不能放宽 |
| O15 | descriptor view 与 material copy 混淆 | `FIX` | View operation 明确为零 kernel alias；Copy/Cast/Repack 是 material operation；TensorDesc convenience 只描述布局，不暗搬 bytes |
| O16 | 无 typed custom operation | `FIX` | opt-in inline、trivially-copyable custom descriptor + 稳定 OperationId/静态 vtable；shape/traits 只在冷路径调用，无全局注册和裸 callable |
| O17 | PreparedOperation/Invocation 混在 schema header | `MOVE` | runtime 类型移到 `prepared_operation.h` 并由 `compute` 拥有；Graph IR 只取得 schema factor |

Operation 使用无 heap 的 typed variant；尺寸由 rank-8 `TensorShape` 等自然 schema 决定，不为了微基准
强压到 64 bytes。真正大的 immutable metadata 仍必须作为显式 tensor/input，不能隐藏 allocation 或 owner。
Operation 只在 build/prepare 冷路径存取，prepared hot path 不复制它。rank 继续固定最多 8；本轮 static
shapes 足够形成按 signature specialization，symbolic algebra 留给有具体 dynamic consumer 的 Graph 后续卡。

### v1 必须交付的可执行目录

下表每行都是一个 operation family，而不是 benchmark 档位；family 的 typed mode 不膨胀成字符串 op。

| 批次 | family | v1 最小完整语义 / native reference |
|---|---|---|
| A | `View` | reshape、单轴 slice、permute、broadcast；输出显式 alias 输入，零 kernel |
| A | `Copy` / `Fill` / `Cast` | plain/strided copy，typed scalar fill，已登记 scalar dtype 的 out-of-place 转换；无隐藏 staging |
| A | `Unary` | Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid/Relu/Gelu/GeluTanh/Silu |
| A | `Binary` / `Select` | Add/Sub/Mul/Div/Min/Max 与 NumPy-style broadcast；typed boolean select |
| A | `Reduce` | Sum/Mean/Min/Max，axis bitset 与 keep-dim；明确 accumulation dtype |
| B | `RmsNorm` / `LayerNorm` | optional scale/bias，最后一轴逐 row；允许 provider 支持精确 in-place |
| B | `MatMul` | rank>=2、batch broadcast、stride 表达物理方向；bias/epilogue 由 Graph region 融合，不塞 flag |
| B | `GatedActivation` | packed halves 或两个显式输入；SiLU/GELU/GELU-tanh gate |
| B | `Gather` / `Scatter` | arbitrary axis，i32/i64 indices，checked 与显式 unchecked policy；Scatter assign/add |
| B | `Concat` | 同 dtype/layout 的任意输入数、显式 axis |
| C | `RoPE` | interleaved/half-split pair layout、rotary dimension、positions 与 frequency tensor 均显式 |
| C | `Softmax` | axis、scale、普通/additive/boolean mask；log-softmax 为 typed mode |
| C | `TopK` | values + i32 indices 两输出，largest/smallest、sorted 与稳定 tie 语义显式 |
| C | `ScaledDotProductAttention` | Q/K/V + optional mask，causal/scale/layout 显式；KV cache policy 和 paged storage 不下沉 |

以上共 **19 个 public OperationKind family（不含 invalid/custom）**，其中 family 内含多个 typed mode；
`SiluAndMul` 被自然的 `GatedActivation` 取代，原三项不是并列保留的兼容别名。A/B/C 是同一 v1 closure
内的施工顺序：只有相应 schema、native reference、差分测试和 direct/Graph journey 同时成立才对用户宣称。

### 明确后续而非 v1 stub 的目录

| family | 当前裁决 | 恢复 trigger |
|---|---|---|
| quantize/dequantize/repack 与 blocked MatMul | `DEFER` 到 ggml/provider storage 卡 | 锁定 format factory、scale/zero tensor 与 provider 可执行路线 |
| grouped MatMul、expert permutation/combine | `DEFER` 到 MoE slice | 至少两引擎的同一模型无关数据合同可复用 |
| convolution/pooling | `DEFER` 到 vision/audio slice | ggml + 第二个消费纵切冻结 layout/padding 语义 |
| SSM/selective scan/recurrent convolution | `DEFER` 到 recurrent slice | vLLM/SGLang/ds4 的 state owner 与 chunk contract 对齐 |
| sort/argsort 全集 | `DEFER` | TopK 不能覆盖的真实 Compute consumer；普通容器排序不搬入 Compute |

### P0 与构建边界

schema validation/inference 为 O(rank + inputs + outputs)，rank<=8；built-in Operation 构造/copy/name/traits
为 O(1)，不分配、不加锁、不查 hash/string。custom vtable 只在 build/prepare 冷路径一次。`compute_ops`
最终 `PUBLIC_DEPS core error`、`PRIVATE_DEPS math`；不依赖 Mem/Compute/SIMD/Graph/provider。公开头使用独立
visibility owner，但 Compute 产品 façade 聚合后，用户仍只 include `hyle/compute/*`。

## Implementation checkpoint（2026-09-12）

- `TensorShape`、`TensorDesc` 与 `Operation` schema 已物理归入 `compute_ops`；runtime artifact 留在
  `prepared_operation.h` / `compute`，Graph 公开边已显式声明 `compute_ops`，不存在经 `compute` 偶然取得
  schema 的隐式依赖；
- 19 个 built-in family、inline typed custom descriptor、exact output span、traits 和纯 shape inference 已实现；
  Windows factor-only、Compute 聚合与 Graph consumer 测试通过。广播 span 生命周期、RoPE 负轴、零 extent
  乘法顺序和未知 enum mode 的红队缺陷已在本 checkpoint 修复；
- 这不是 terminal record，也不改变 `AUDITED`：View、Copy、Fill、Cast、Unary、Binary、Select、RMSNorm、
  Reduce、RMSNorm、LayerNorm、rank-2 MatMul 和 packed SiLU GatedActivation 已取得 native executable route，
  其他 7 个 family 尚未取得；三平台/property/sanitizer 与完整
  direct/Graph journey 仍按 closure card 待执行。用户文档将“可描述”与“可执行”分开，不把 schema
  存在宣称成 kernel 支持。

2026-09-14 的 Native indexing checkpoint 已为 Gather、Scatter 与 Concat 增加可执行 CPU reference，目录
进度更新为 **15/19 executable**。`compute_ops` 自身 schema/API 没有因此变化；余下未执行 family 是 RoPE、
Softmax、TopK 与 ScaledDotProductAttention，仍不得把可描述冒充可执行。

同日的 transformer checkpoint 已为 RoPE、Softmax/LogSoftmax 与 TopK 增加 native executable route，目录
进度更新为 **18/19 executable**；只剩 ScaledDotProductAttention。TopK schema 同时补上 i32 index 输出的
axis 上界检查。该修正属于已有 typed contract 的完整性，不增加 family 或隐藏策略；`compute_ops` 仍保持
`AUDITED`，等待 19/19 direct/Graph journey 和终局验证。

同日的 attention checkpoint 已补齐 ScaledDotProductAttention，目录达到 **19/19 executable**。公开 schema
没有新增 family 或 provider-private knob：BSHD/BHSD、bottom-right causal、Bool/additive broadcast mask、
显式 scale 与 accumulation contract 继续由既有 typed descriptor 表达；Native 当前只接收 plain f32
Q/K/V、dense f32 output 与 f32 accumulation，不支持的 storage/layout 仍返回 `Unimplemented` 供有序 provider
fallback。`compute_ops` 保持 `AUDITED`，等待完整 direct/Graph、三平台和交付验证。
