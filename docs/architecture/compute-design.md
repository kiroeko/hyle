# HYLE Compute：异构执行与算子提供者架构

> 决策状态：`ACCEPTED`。本文只定义 Compute 的长期产品合同，不复制阶段性完成状态；当前可用能力查
> [用户指南](../user/compute.md)，施工资格只查
> [`module-status.json`](../implementation/module-status.json)。下文平台栈和开工顺序是目标边界，不表示
> 每个 provider/backend 已经交付。被删除的旧实现只可从 Git 历史取得调查事实；历史 task/evidence
> 不是当前设计输入、实现模板或用户承诺。

## 1. 产品承诺

Compute 是 HYLE 的**异构执行层**。它让 C++ 消费者可以：

1. 发现并选择 CPU、CUDA、Metal、Vulkan 等计算设备，建立显式执行上下文；
2. 在不经过 Graph 的情况下直接准备并执行一个算子或原生 kernel；
3. 让 Graph 编译器把一段逻辑计算交给最合适的 HYLE 或第三方 provider；
4. 显式管理 queue、storage、copy、同步、完成和 native interop，而不隐藏昂贵工作；
5. 在 provider 不支持某种形状时得到可解释的能力/失败结果，而不是静默复制、同步或退回慢路径。

Compute 不是推理引擎。模型格式、tokenizer、KV/prefix-cache 策略、continuous batching、sampling、
HTTP serving 和模型循环仍由消费方拥有。Compute 也不是 Graph：它执行已经决定的 op、kernel 或
compiled region，不拥有跨节点拓扑、全图 placement、liveness 和融合决策。

## 2. 两条同等重要的用户路径

### 2.1 直接执行

工程计算、渲染 compute pass、codec/图像处理和自定义推理 kernel 不应为了运行一个操作先造完整图。
概念上的正常路径是：

```text
Device -> Context -> Queue
TensorDesc + Mem BufferView -> TensorView
Operation -> PreparedOperation -> per-lane OperationInvocation
OperationInvocation -> submit(bindings) -> Completion

Module -> Kernel -> CommandList -> submit() -> Completion
```

- `Device` 是可复制的能力描述，不拥有 runtime；`Context` 拥有 backend/provider 会话。
- `Queue` 表达有序提交域；不能用全局默认 stream 隐藏 affinity 或同步。
- `TensorDesc` 把 logical dtype/shape 与 physical storage format/byte layout 分开；fixed block encoding
  使用开放数值 identity，无法通用解释的 packed layout 使用 opaque identity + exact extent。
  scale/zero/bias 和量化公式属于 typed operation/provider inputs，不进入 storage descriptor；
  `TensorView` 只把描述与 Mem storage 绑定，不拥有 payload。
- `PreparedOperation` 是可共享的不可变 artifact，把能力检查、provider 选择、layout/repack 决策和可能的
  编译移出热调用；`OperationInvocation` 拥有一次并发 lane 的 stream、binding metadata 与 workspace。
- `Module/Kernel/CommandList` 是原生或 HYLE kernel 的低层路线；Vulkan SPIR-V、CUDA
  module/kernel、Metal pipeline 和 CPU function target 都通过有类型的 backend extension 表达。
- `Completion` 是提交后的生命周期边界。等待、轮询、callback 和 coroutine bridge 必须共享同一个
  完成状态，不能制造互不相认的 event。

一次性 convenience 调用将来可以组合 prepare + invocation + submit，但必须在名称和文档中暴露它包含冷路径工作；
循环执行默认使用 prepared 形状。

### 2.2 Graph lowering

Graph 不逐节点调用一个通用 `execute(op)` 并在热路径重新路由。Compute provider 接收单 op 或一段
候选 region，返回绑定到设备、layout、shape signature 和 provider identity 的 compiled artifact。
Graph 的 `Executable` 只保存这些 artifact 与已经决定的 copy/sync 步骤，调用时直接提交。

因此 HYLE 既能借成熟算子库获得广度，也能保留 HYLE 自实现的胜出 kernel。二者竞争发生在
prepare/compile 阶段，不在 element、lane 或每次节点执行中反复裁决。

## 3. Backend 与 Provider 是两条轴

这两个概念不得再混成一个“大后端”枚举：

| 角色 | 回答的问题 | 例子 |
|---|---|---|
| backend | 设备、context、queue、memory、module、同步和 native handle 怎样工作 | CPU、CUDA、Metal、Vulkan；合同允许以后加入新的 backend |
| provider | 哪些 op/region 能以何种 layout、精度和成本在该 backend 上执行 | HYLE native、ggml、oneDNN、MLX、MPSGraph、vendor library |

Provider 必须声明而不能让调用方猜测：

- 支持的 op、dtype、shape/stride/layout、量化格式、in-place/alias 与精度；
- 编译和执行所需 workspace、常量 packing、storage domain 与 alignment；
- 是否启动或占用自有线程、是否阻塞 host、是否支持并发 invocation；
- copy、同步、host round-trip、动态分配和首次 JIT 等可观察成本；
- cancellation、device loss、迟到 completion 和销毁顺序；
- provider/version/device/ISA identity，供 plan cache 和诊断使用。

Provider 可以自行拥有 device allocation 或 graph allocator。HYLE 必须如实保留这个 owner；不能把
一个伪造的 `BufferView` 当成零拷贝。ggml CUDA 的历史调查只证明这种风险真实存在，见
[ggml provider 的 storage 约束](borrowing-ggml-memory-model.md)；adapter 开工时仍须对锁定版本复验。

## 4. 平台栈与优先级

“正式支持”与“首要优化目标”是两件事，设备 backend 与算子/图 provider 也不是同一层。CUDA 是一套
比 Metal 更宽的生态，但在 HYLE 分类中，二者首先都承担设备、queue、module、memory 和 completion
这一执行轴；CUDA libraries、MLX 等再作为上层 provider 审计。当前长期路线是：

| 优先级 | 设备 backend | 可审计的 provider/runtime | 产品要求 |
|---|---|---|---|
| 主要 | Apple 原生 Metal | **MLX（首个资格化 provider）**、HYLE Metal kernel、MPS/MPSGraph、Accelerate/BNNS | 推理纵切优先证明 MLX region；原生 Metal 保留 custom/best-of kernel、资源互操作和缺口路线；不同层不得混成一个 backend |
| 主要 | NVIDIA CUDA | HYLE CUDA kernel、cuBLAS/cuDNN/TensorRT、ggml 等 | 覆盖消费级 RTX 与可用数据中心形状，按完整 region 成本裁决 provider |
| 正式保底、次要优化 | 原生 Vulkan；Apple 上经 MoltenVK | HYLE SPIR-V kernel、ggml Vulkan 等 | 给广泛消费级 GPU 和 renderer interop 一条共同路线；MoltenVK 只签 Apple portability，不替代原生 Metal 性能路线 |
| 基线 | CPU | HYLE SIMD/native、oneDNN、ggml；Apple 可含 Accelerate/BNNS | 三平台 correctness、fallback、provider contract 和可重复性能基线；不是“只剩 CPU 时才用”的 stub |

AMD HIP/ROCm **无限期不进入当前产品线**：没有 component、CMake gate、空 target 或恒失败 stub，Vulkan
只提供跨厂商保底而不冒充 ROCm。只有同时具备可持续使用的 AMD 真机、明确消费者和维护预算，才以新的
closure card 恢复 HIP backend/provider 调研；通用 backend 扩展合同为这次恢复保留空间，但不构成承诺。

### 4.1 施工包不能隐藏 SIMD

进度清单必须把产品纵切与实现底座分开。`simd` 是已经独立关闭的 CPU 叶组件；Compute 不重新实现它，
但 **HYLE-native CPU provider** 仍是一项显式施工包：它负责把 Tensor/Operation 的 shape、layout、alias、
precision 和 completion 合同映射到 SIMD whole-row/fixed-kernel 路线，并拥有 tiling、tail、fallback 与
prepare-time route。把它只写成“Compute 算子目录”会错误隐藏 HYLE 自有算力路线。

| 层 | 施工包 | 直接实现底座 | 边界 |
|---|---|---|---|
| CPU 执行 | CPU backend + HYLE-native provider | Platform + Mem + **HYLE SIMD** | 内建的 reference/fallback/best-of 路线；当前归 `compute`，不因任务可见性虚构新 factor |
| CPU 借入 | oneDNN provider | oneDNN x64 BRGEMM ukernel | 可选 MatMul provider；不是 CPU backend，也不替代 native/SIMD |
| 跨后端借入 | ggml provider islands | ggml CPU/CUDA/Metal/Vulkan 等已准入 backend | 每种 backend 分别审计，不能把“ggml provider”写成 CPU-only 单项 |
| Apple | Metal backend + MLX/MPS/HYLE/ggml providers | Apple runtime 与 HYLE custom kernels | MLX-first，但保留 native Metal escape hatch |
| NVIDIA | CUDA backend + vendor/HYLE/ggml providers | CUDA runtime、vendor libraries、custom kernels | provider 选择按完整 region 成本 |
| 可移植 GPU | Vulkan backend + HYLE SPIR-V/ggml provider | Vulkan；Apple portability 使用 MoltenVK | 正式保底，不冒充原生 Metal 性能 |
| 图编译 | Graph IR/compiler/plan | 上述 prepared op/region | 横跨所有 backend/provider，拥有 partition、placement、copy/sync 和 workspace |

因此阶段清单应写成“CPU backend/HYLE-native-SIMD → oneDNN/ggml CPU region → Apple/CUDA/Vulkan
backend-provider 组合 → 跨 provider Graph”，而不是把 oneDNN、ggml 与整个 GPU backend 放在同一级。
当前状态仍只从 registry 和各 closure card 读取，本表不复制完成百分比。

### 4.2 Apple 各层的边界

- **Metal** 是 Apple GPU 的底层图形与计算 API。HYLE 只拥有其中的 compute pipeline、resource、queue、
  synchronization、profiling 和 native interop；raster、swapchain 和 present 不进入 Compute。
- **Metal Performance Shaders / MPSGraph** 分别提供优化 primitive 与符号计算图，更接近 operator/graph
  provider。若借入，artifact、workspace、stream 与 graph ownership 留在隔离的 provider island。
- **MLX** 是面向 Apple Silicon 的机器学习数组与计算图框架，具有 lazy evaluation、compile、stream、
  unified-memory 模型和完整 C++ API。对 HYLE 的 Apple 推理纵切，它是首个要资格化的 region provider、
  interop 对象和独立性能参考；它不是 Metal 的别名，也不是设备 backend。
- **Core ML** 是加载、编译和运行模型的更高层部署 runtime，并可调度 CPU、GPU 与 Neural Engine。
  HYLE v1 不把它伪装成低层 kernel provider；只有真实纵切证明模型级委托仍保留 HYLE 所需的 storage、
  scheduling、diagnostics 和 specialization 控制时，才登记 routed runtime adapter。

Apple 主路线因此是“**MLX-first provider + 原生 Metal backend/escape hatch**”，不是“MLX 对 CUDA”。
HYLE 优先把 MLX 能完整、高效承载的 region 交给 MLX；缺失算子、已证明胜出的 custom kernel、原生
resource/queue interop 和 provider 边界由 HYLE Metal 路线承接。选择发生在 prepare/compile 冷路径，
并计算 ownership、copy、graph capture、线程和 source/static/shared 交付成本；“MLX-first”不是未经
测量就强制所有 op 经 MLX。

Vulkan compute 不是只为 benchmark 保留的“可选图形 API”。它服务：

- Windows/Linux 上的跨厂商 compute；
- 与已有 Vulkan renderer 共用 device、queue、buffer 和 timeline semaphore 的资源管线；
- SPIR-V compute kernel、工程计算和不要求 CUDA 的部署；
- Apple 上经 MoltenVK 把 Vulkan 调用翻译到 Metal 的 portability 资格路径，但不能冒充原生 Metal 性能、
  Apple 专有能力或完整原生 Vulkan 支持。

Compute 的 Vulkan extension 必须覆盖 device/queue-family 选择、pipeline/module、descriptor/binding、
dispatch、barrier、timestamp 和 device-lost 语义。与 renderer 或其他 API 共享资源时，memory 与
semaphore/fence 必须分别导入/导出并查询 handle capability；共享地址或 native handle 不自动意味着
同步、coherence 或 ownership 正确。raster render pass、swapchain、窗口和 present policy 属渲染产品，
不进入 HYLE Compute。

## 5. Storage、同步与生命周期

Mem 是 storage-semantics owner；Compute 是 device/provider artifact 与异步使用 owner：

- Mem 提供 `MemoryResourceRef`、`Buffer`、views、mapping、range allocator 和 lifetime planner；
- Compute context/provider 暴露具名 storage resource，并解释 native allocation identity；
- Graph 推导 resource、liveness 和 alias facts，再交给 Mem planner；
- Queue/Completion 决定 Buffer、workspace、packed constants 和 staging slot 何时可复用。

任何异步 submit 的输入、输出、kernel argument backing 和 provider state 都必须活到 Completion 终态。
取消是“请求停止”，不是“立即可回收”；不支持抢占的设备仍需 drain 迟到完成。Context 的关闭是显式
`close/drain` 旅程，析构不能偷偷做无界等待，也不能在 work in flight 时释放 backend state。

Storage domain 至少区分普通 host、backend-visible/pinned host、device-local 和 managed/unified。
unsupported 必须返回明确能力错误；不能把普通 host RAM 冒充 pinned，把 unified address 冒充零复制，
也不能把共享物理内存的 host/device budget 重复相加。

Tensor 的 physical-format identity 不等于 backend memory identity。Plain Tensor 以 logical scalar width 和
byte stride 表达；单轴 fixed-block Tensor 记录 elements/block、bytes/block、alignment 与 block axis；更复杂
格式只暴露 opaque identity 和 exact bytes。通用层不得为了识别格式引入字符串查找、全局 registry、owner
或虚调用。格式专属 factory、alignment 终检与 scale/zero 等关联 tensor 由 typed provider extension 拥有。
未知但结构合法的 encoding 是 provider fallback 的 `Unimplemented`，不是 schema corruption。

## 6. Operator 面与可扩展性

公共 operator 合同以真实消费者反复出现的模型无关计算族为核心：

- copy/fill/cast、layout transform、gather/scatter 与 quantize/dequantize/repack；
- unary/binary、reduction、softmax、top-k 等张量运算；
- GEMV/GEMM/batched matmul 与必要的低精度 accumulation；
- normalization、activation、residual/gated 组合；
- embedding、RoPE/position transform、attention 的可复用计算 primitive；
- MoE dispatch/combine 所需的模型无关 tensor primitive。

这不是“看到 peer 有一个 op 就复制一个枚举”。精确准入集必须由 ggml/llama.cpp、ds4、vLLM、SGLang
的真实调用与至少一个独立成熟 provider 对账，并逐项登记精度、layout、alias 和 workspace。KV cell
管理、expert policy、mask 业务语义、sampling policy 与完整模型 layer 仍留在引擎。

未进入公共 catalog 的计算走 typed custom-kernel/provider extension。扩展不能依赖字符串属性袋、裸
`void*` 或 process-global 注册副作用；其 schema、owner 和 backend identity 必须在编译期或显式对象中
成立。这样后续可以逐个用 HYLE 胜出 kernel 替换 provider region，而不推翻 Graph 或普通用户 API。

## 7. 内部因子与依赖方向

长期分层如下；名称是中央 component graph 的目标形状，不表示当前已经登记或实现：

```text
HYLE::compute                         # 用户产品 façade
  compute_core                        # Device/Context/Queue/Completion/kernel/interop
  compute_ops                         # Tensor/Operation 描述与公共算子合同
  compute_sched                       # 可选 Scheduler/coroutine completion bridge
  compute_backend_cpu|cuda|metal|vulkan
  compute_provider_native|ggml|onednn|mlx|mpsgraph|vendor

graph_ir -> compute_ops
graph_compile -> graph_ir + compute provider contracts + mem
graph_plan -> compute_core + mem
graph_sched -> graph_plan + compute_sched + sched       # 可选运行时组合
```

`compute_core` 不因 coroutine convenience 强制公开依赖 Sched；需要 `co_await`/task 组合的用户取得窄的
bridge factor。可选 provider/backend 各自拥有外部依赖和 feature 开关，未选择时不下载、不编译、
不导出恒定失败 stub。Compute 永远不依赖 Graph。

内部 factor 是否按上图拆成独立状态 owner，要在开工 baseline 审计中逐项满足
[质因数判据](capability-factorization.md)；用户始终只选择 `hyle::compute`，不会被迫理解 provider target。

## 8. 性能合同

Compute 的目标是**端到端可组合的最佳路线**，不是包装层在一个空调用 benchmark 中少几纳秒：

1. prepare/compile 后的 submit 不做 provider 搜索、shape inference、JIT、常量 packing 或无关 allocation；
2. direct static kernel 可编译期路由，prepared op/region 至多支付一次预绑定调用边界，不按 element 分发；
3. provider 选择同时计入 kernel、layout/repack、copy、同步、thread handoff 和 workspace，不能只比 kernel；
4. host/device transfer 可批处理并与计算重叠时应显式表达，不通过隐藏 staging 制造假零拷贝；
5. provider 自有线程必须可配置或在 region 级隔离，禁止与 HYLE workers 无界嵌套造成 oversubscription；
6. compile latency、warm execution、steady throughput、tail、peak/resident memory、copy bytes 与能耗分别报告。

同合同对拍必须锁定 op 语义、dtype/精度、layout、shape、线程/queue、warmup、storage residence 和计费
边界。HYLE 原生只有在资格化 workload 上形成可复现 material advantage 才取代成熟 provider；无法区分、
汇编等价或结果随顺序/地址翻转时优先成熟实现。Graph 路线还必须与直接 Compute 路线做 one-op 税基线，
防止编排层吞掉 kernel 收益。

## 9. 交付、诊断与负空间

- source、installed static、installed shared 提供同一 C++ 语义；当前不设计 Native FFI。
- feature 集合必须确定产物 identity；正式构建不能依赖 `AUTO` 让机器环境悄悄改变 provider。
- capability query、compile report 和 tracing 能解释选了谁、为何 fallback、插入了哪些 copy/sync、需要
  多少 workspace；关闭 Obs 时热路径不支付诊断成本。
- native interop 使用 backend-specific typed extension；普通 API 不泄漏 CUDA/Vulkan/Metal/MLX/ggml 类型。
- 不提供训练、自动微分、分布式 collective runtime、模型 zoo、render graph、窗口/present 或 serving。

## 10. 开工顺序

Compute 与 Graph 采用共同纵切，但按独立 factor 小步验收：

1. **资产与身份审计**：固定现有源码/头/CMake baseline；把每个入口裁决为 KEEP、MOVE、REPLACE 或
   DELETE；按质因数判据登记必要 component ID。旧 API 不需要兼容。
2. **Compute core**：先完成 CPU Device/Context/Queue/Completion、Mem resource bridge、typed native
   kernel 和 explicit shutdown；清除对 Sched 的无条件公开依赖。
3. **Compute ops**：冻结最小 Tensor/Operation/PreparedOperation 合同，以一个 CPU reference op 和一个
   HYLE SIMD/native op 证明 direct path。
4. **立即接 Graph 纵切**：Graph 只做一个节点也能编译为 `Executable` 并通过 `Invocation` 运行；此时
   验出抽象税，而不是等 Compute 大全套完成后才发现形状不合。
5. **多节点 CPU slice**：以量化模型中的 normalization → matmul → gated activation 为首个真实片段，
   加 liveness/alias/workspace、ggml/oneDNN 候选 provider 和直接路线对拍。
6. **主要平台 slice**：NVIDIA 走 CUDA backend 与合格 CUDA providers；Apple 先资格化 MLX region，
   同轮建立原生 Metal custom-kernel/interop 路线并审计 MPSGraph/MPS；随后扩 attention、quantized
   matmul、repack 和异构 partition。Core ML 不进入首轮低层路线。
7. **Vulkan slice**：同一 graph/op 合同落到 SPIR-V，验证 native device、buffer、timeline completion、
   external memory/sync 与 Windows/Linux 消费级设备；MoltenVK 只签 portability 证据。
8. **Transformer block 与重复执行**：冻结后运行真实 prefill/decode、动态 shape specialization、并发
   invocation、取消/drain 和 source/static/shared 交付门。

每一步都先创建合法 closure card 并完成 `SCOPED -> AUDITED`，只实现冻结的有限 `FIX/BORROW/DELETE`
集合；未达到下一步所需的 lifecycle、correctness 或 P0 成本结构就停止为 `BLOCKED`，不靠扩 benchmark
或继续研究掩盖状态。
