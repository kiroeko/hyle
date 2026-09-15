# Compute/Graph：四条原生路线

本文持有路线隔离、native owner 和 Provider 接入决策；共同语义见 [Compute](compute-design.md) 与
[Graph](graph-design.md)。这是设计，不是已交付能力表。具名源码版本、实验覆盖和未证明项见
[六项复核](../implementation/tasks/compute/six-point-architecture-review.md)。

## 1. 隔离边界

CPU、CUDA、Apple MLX/Metal、Vulkan 各自实现冷编译、执行制品、运行状态与回收。公共层只共用同义的
计算/数值描述、错误、资源访问、结果/完成与必要计划事实，不强制一个 segment loop、统一 native
handle 或逐 kernel 虚调用。独立实现不是把 Error、Mem 等同义基础设施复制四份。

模型 CPU+GPU 联合计算不在范围内。用户选计算路线，HYLE 按研发时已验证的规则绑定路线内实现；
缺少 GPU 算子或资源时明确拒绝，不借 CPU fallback 绕过配置。host 控制、权重准备、上传下载和回收
不属于这种混合。

构建必须从公开产品、路线实现直到第三方 native backend 一直隔离：

- `hyle::compute/graph` 聚合所选真实实现，不固定拉入全部 CPU/oneDNN 或全部 GPU Provider。
- OFF 不探测 SDK、不编译/链接/安装原生实现；AUTO 只发现实际可用能力，ON 缺必要条件明确失败。
  系统 SDK 由用户安装，HYLE 不下载 CUDA Toolkit。没有实现不预造选项/空 target。
- 公共语义头不要求 CUDA/Metal/Vulkan SDK。native 专家入口归对应路线；flags、ISA、部署版本、
  vendor 宏归 owning target，不改变消费者可见的公共对象布局，不泄漏全局编译开关。
- vendor 源码是共享的版本化资源，构建产物按平台/编译器/backend 配置隔离。同一个源码仓库可提供
  多路线，但每个 native runtime 分别适配、资格化；source/static/shared 的能力边界一致。
- 裁剪必须传入 vendor 的 CMake，不能只关掉 HYLE 外层入口。vendor 必需的 host/CPU 基础代码不等于
  允许 CPU 模型计算；如果 vendor 无法关闭所需的隐式模型 fallback，该路径不能冒充纯 GPU。
- 多路线同 SDK 可选并存，不代表一个模型可跨 CPU/GPU联合执行。不增一套与 backend feature 重复的
  全局“路线 enum”开关；实际程序在冷准备期绑定，不每次运行检查构建配置。

MLX 有独立的 CPU/Metal/CUDA 开关，默认 tests/examples、文件格式功能和 nested Metal-C++ 下载也须
由适配构建明确处理。选择 Apple adapter 不能默认取得 MLX-CUDA；共享资源缓存应接管其所需源码，
不能靠关闭顶层 FetchContent 就声称离线完整。具体依赖锁/发行组合由产品准入确定，本页不签升级。
参见 [MLX CMake](https://github.com/ml-explore/mlx/blob/v0.32.2/CMakeLists.txt) 与
[HYLE build system](build-system.md)。

## 2. CPU：持久执行资源与真实 ISA 实现

CPU 不是 GPU 的失败接管路径。Platform 提供 OS 已公布的拓扑/NUMA/cache 事实；SIMD 拥有自身最佳
kernel 的 ISA 路由；Compute 不重新实现这两套探测。实现及线程/packing 策略的适用规则在开发时
确定，prepare 匹配 ISA/shape/用户预算并绑定资源；输入数据与明确运行参数才进入热路径。

普通计算保留 HYLE SIMD/Native、oneDNN/BLAS 等实际成熟实现，ggml-CPU 作为独立 runtime 候选，
不能因为 GPU carrier 可用就认为 CPU carrier 已合格。worker 数/亲和性依据 workload 与用户限制，
不把逻辑核心数直接当所有 shape 的最优线程数，也不按 cache 大小生造公共容器。

| 阶段 | 原生责任与限制 |
|---|---|
| prepare | 持久 executor/线程团队、ISA/primitive descriptor、权重 packing、每槽 scratch；明确 oneDNN/BLAS/ggml 是否使用自己的线程池 |
| run | 固定 block/调用，正常 bulk 并行；单线程、小任务或嵌套 worker 可走有界 inline，不能每 op 起线程、换池或重复 affinity |
| owner | caller 借出的 executor 保活到已接纳任务终态；immutable packing 共享，scratch/可变库状态按槽独立 |
| failure | 部分任务接纳后 drain 已接纳前缀；拒绝不丢任务，nested 调用不能因等同池 worker 自锁；全局进程线程开关不假装成对象局部设置 |

既有 Sched public bulk/executor 组合若已满足消费，就直接复用；不为架构整齐增一个空 bridge factor。
NUMA、物理/逻辑核、AMX 等是实现条件与用户资源约束，不把 vLLM 的环境变量/进程政策搬成 HYLE
公共 API。参考 [vLLM CPU platform](https://github.com/vllm-project/vllm/blob/51918e252af4dfff381001f42b7ebe6a0454501b/vllm/platforms/cpu.py)、
[ggml CPU threadpool](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-cpu/ggml-cpu.cpp)。

## 3. CUDA：vendor/native 程序与可选 capture

普通 GEMM 优先资格化 cuBLASLt/cuBLAS；量化、attention、融合等按实际 dtype/格式/device 选择
成熟专用 kernel 或 HYLE。CUDA runtime 与算子实现互补，不意味着 HYLE 要自写 vendor 已优化的全部
计算。推理引擎的 Torch/Triton 包装不是独立 C++ SDK，借 kernel 需窄、版本锁定的源码接入。

| 阶段 | 原生责任与限制 |
|---|---|
| prepare | device/context、stream、vendor handle、tactic/packing、module/code、每槽 scratch；capture 前完成允许的 JIT/warmup，明确 capture 支持与地址稳定域 |
| run | 已绑定 launcher/vendor call，或实例化 CUDA graph replay；有限参数更新不隐式重搜 tactic，不每 op 查询 device/设置函数属性 |
| owner | stream/context、graph executable、captured pointers/代码保活；cuBLAS handle 的 stream/workspace 等可变状态按并发要求独占或池化 |
| failure | enqueue 成功不等于 kernel 成功；记录 accepted prefix 与事件/stream 终态，事件创建/记录失败不能提前释放已使用 allocation；取消不立即回收 |

`cublasSetStream` 会重置用户 workspace，因此 stream/handle/workspace 的绑定次序和并发使用是
真实合同，不是统一 Queue 接口能抹平的细节。capture 不是所有输入/调用都可用；支持的 node 更新、
地址、shape 与操作限制按 API 验证。运行参数尽量进入稳定 data buffer，不每 token 创建新 capture。
参见 [cuBLAS workspace](https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace) 与
[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)。

vLLM、SGLang 的 finite graph keys、持久输入/输出与 decode bucket 说明冷准备/热 replay 的价值；
HYLE 借其机制，不拥有引擎 batch/KV/LoRA 策略。ggml-CUDA 自持 stream、cuBLAS handles、pool 和
graph 状态，走 delegated region 或被确实支持的 native integration，不把私有 stream 强制转换为
HYLE Queue。参考 [vLLM dispatcher](https://github.com/vllm-project/vllm/blob/51918e252af4dfff381001f42b7ebe6a0454501b/vllm/v1/cudagraph_dispatcher.py)、
[SGLang graph runner](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)、
[ggml CUDA context](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-cuda/common.cuh)。

压缩 MatMul 的渐进替换按 [Q4 规范例](compute-design.md#cuda-q4-matmul-的规范例)：研发时在相同格式、
精度和完整边界比较成熟 region、原生 kernel、HYLE 扩展，准备时只应用已验证规则。
GGUF、GPTQ/AWQ 与 FP4 不是同一种四位数。

## 4. Apple：优先在 MLX 的 Metal runtime 内组合

### 4.1 硬件和软件不是两条互斥路线

MLX 的 Apple GPU 后端本来就是 Metal。MLX 提供 array、graph/compile、allocator、kernel 选择等，
Metal 提供底层 GPU API 与 shader；“混用”首先是向 MLX 扩展计算，不是每个算子来回跨运行时。
Apple 另有 Neural Engine；M5 GPU 的 Neural Accelerators 又是 GPU 内的矩阵加速，不与独立 ANE
混为一谈，也不意味着 MLX 自动调用 Core ML/ANE。参见 [Apple 硬件说明](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)
与 [MLX/M5 说明](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)。

MLX 使用 Metal 4 TensorOps / Metal Performance Primitives 可获得新 GPU 矩阵加速；所以不能因
“图形 API”就认为 native Metal 没有推理硬件路径。硬件、SDK、deployment target 与数值模式一起决定
可用性，不能仅根据系统能运行 Metal 或设备名字决定。

具体准入以版本为准：MLX v0.32.2 的 NAX 构建需要 Metal 4、SDK 与 deployment target 至少 26.2，
运行时还检查 OS/device 架构；部分转置 QMM 路径要求 K 按 64 对齐，F32 是否允许 TF32 也影响选择。
不能以 M4 的严格 F32 实验签 M5 NAX 的 FP16/BF16/量化吞吐。参见
[Metal 构建](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/CMakeLists.txt)、
[NAX 能力判定](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp)、
[quantized dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp)。

### 4.2 三种接入，研发时验证完整收益

1. **MLX 完整 GPU region**：保留其 compile、packing、allocator 和内部融合，不逐 op 绕回 HYLE。
2. **MLX 内 HYLE Metal 扩展**：优先公共 `fast::metal_kernel` 或 C++ custom Primitive；沿相同
   stream/encoder 注册 inputs、outputs、temporaries，沿 MLX 原生 owner 与 hazard tracking 执行。
   可用预编译 metallib/窄 Primitive 降低 generic 扩展成本，但都须按版本验证。
3. **独立 native Metal / ggml-Metal region**：确有优势或能力缺口时独立候选。与 MLX 的边界须证明
   buffer、format、timeline 和 completion 互通；不能因统一内存就假定零 copy/零同步。

通用 custom kernel 可能分配输出、使输入 contiguous、查询 library/kernel cache，也可能形成融合
边界；“在同一个 Metal runtime”只避免不必要的外部桥，不保证无这些费用或比 MLX 内置计算更快。
研发时比较完整片段后固化默认替换规则。参见 [C++ 扩展](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
与 [custom Metal kernels](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)。

### 4.3 资源、线程和错误

| 阶段 | 原生责任与限制 |
|---|---|
| prepare | owned MLX stream/提交 owner、compile/finite shape warmup、constants；cache 与线程关系明确，不能在未知 worker 上假定已预热 |
| run | 固定 compiled region 或已选 native 程序；MLX cache hit 仍可能构造执行 arrays/metadata，因此先按 managed 资格化，不假装通用无分配 graph replay |
| output | MLX 生成 array 由 MLX owner 保活；支持 owning result，避免为旧 out-buffer 形状影子复制；保留旧结果期间不得重用其 storage |
| completion | CPU 可见/shared 不等于 GPU 成功结束；array ready/callback 不代替已检查的错误终态。公开路径若需 owned worker 上同步，明确成本，不能承诺全异步零 host wait |
| reclaim | 输入、输出、临时、metallib/Primitive 与 command 使用到终态才释放；跨 runtime import 借出方保活与失败 drain 单独证明 |

MLX 的 Python DLPack/MPS 互操作不自动成为 HYLE C++ 的承诺：v0.32.2 对应桥在 Python binding 中，
持有 nanobind owner 并处理相应 buffer 表示；不能把 Python 文档当独立 C++ 外部 Queue/import 已合格。
公开 C++ array/allocator 能力可作窄适配依据，但仍须说明同步与 owner。参见
[MLX 互操作说明](https://ml-explore.github.io/mlx/build/html/usage/numpy.html) 与
[实际 binding](https://github.com/ml-explore/mlx/blob/v0.32.2/python/src/convert.cpp)。

ds4 的原生 Metal 实现展示持久 pipeline/scratch、batched encoding 与明确 command-buffer 错误
检查；它不是 MLX adapter，其进程全局资源和可选 unretained command buffer 也不能原样成为基础库
API。借算法与原生约束，不借引擎全局生命周期。参考
[ds4 Metal](https://github.com/antirez/ds4/blob/9139e2ae58a41503968a500f36f75895c1ba63fc/ds4_metal.m)。

## 5. Vulkan：录制、资源可见性与 portability

Vulkan 是正式跨厂商保底计算路线。原生 shader/pipeline 或 ggml-Vulkan region 都可准入；Apple
上的 MoltenVK 属于 portability 组合，不替代 Apple 原生性能。NVIDIA 上 Vulkan 成功不证明 AMD、
Intel 或 MoltenVK 的 feature/limits、性能与正确性。

“保底”不等于所有 GPU 强制用最弱通用 shader：ggml 针对 cooperative matrix、NV cooperative
matrix2、integer dot、subgroup size 与 float controls 有专门分支。HYLE 保留按能力/数值/shape 的
已验证的选择规则和 specialized pipelines；shader 编译器、SPIR-V/headers 与运行 device 的支持须同时验证。
例如编译工具识别某扩展不能证明设备支持它，声明支持 integer dot 也不必然硬件加速。适用时借
成熟高效分支；无硬件资格的分支不宣称 SOTA。参考
[ggml shader 能力构建](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-vulkan/CMakeLists.txt)
与下文 native 实现。

| 阶段 | 原生责任与限制 |
|---|---|
| prepare | physical/device/queue family、memory type、descriptor layout/pool/set、pipeline/SPIR-V/specialization、每槽 command pool/buffer；查询 subgroup、storage、alignment 与 extensions |
| run | 可合法复用的 recorded commands 或声明为 managed 的 encoding；pipeline/cache lookup 不混入 prepared 热路径；录制的 push constants 不随 host 变量改变 |
| sync | execution dependency 与 memory availability/visibility 分别成立；shader write→read、transfer→compute 按 stage/access，跨 queue/family 需要真实 semaphore/ownership transfer |
| owner | descriptor/command pool 的 host 并发与 pending 限制明确；被引用 buffer、memory、pipeline、descriptor 与 commands 保活到 fence/timeline 终态 |
| failure | OOM、提交失败、device lost 分别处理；non-coherent host memory 做适用 flush/invalidate；取消不能 reset/free pending 资源 |

timeline semaphore、某种 queue 或 feature 不普遍存在；按真实 capability 选择合法 fence/提交方式
或拒绝，不伪造同成本功能。内存类型由真实访问模式决定，不为一个统一 Buffer 强迫所有计算走
host-visible backing。参见 [Khronos 同步示例](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
和 [command buffer 生命周期](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html)。

ggml-Vulkan 的 graph compute 可以逐次编码并按工作量分批提交，与 HYLE 录制一次的 native 程序
不是同一成本合同；底层也有自己的 device/command/descriptor pool 和 transfer/event owner。
接口叫 graph 并不证明 replay 无分配/无建图；borrow native objects 或同 GPU 跨 Provider device 依赖
只有真实公开桥成立才承诺。参考
[ggml Vulkan 实现](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-vulkan/ggml-vulkan.cpp)。

## 6. Provider family 不等于 backend

| 候选 | 当前设计中的接法 | 不可推导的能力 |
|---|---|---|
| HYLE Native/SIMD、oneDNN/BLAS | CPU 原生计算/片段，按真实线程 owner 接入 | oneDNN 家族有其它 backend 不等于已适配全部 |
| cuBLASLt/cuBLAS、成熟 CUDA kernel、HYLE CUDA | CUDA 原生调用/融合程序，合格时 capture | 任意 Q4 格式、所有 kernel 可 capture |
| ggml-CPU / ggml-CUDA / ggml-Metal / ggml-Vulkan | 每个 runtime 独立 region adapter，或受审窄源码 kernel | 一条成功不签另三条；不虚构 ggml-MLX backend |
| MLX-Metal | Apple 完整 region / 同-runtime HYLE 扩展 | MLX 的 CUDA backend 自动加入 Apple 路线；自动使用 ANE |
| 独立 HYLE Metal / HYLE Vulkan | 各 native 程序，与成熟 region 按真实边界比较 | 同 GPU/统一内存意味着可以免费交叉组合 |

ggml 在这些版本提供的是 **Metal，不是 MLX backend**；MLX 自己反而有可选 CUDA backend。
库名、硬件厂商和 native runtime 必须分开。共享版本锁、格式知识和源码缓存有价值，强制共享可变
handle/allocator/Queue 或同一个热态 adapter 没有依据。
[ggml backend 清单](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/CMakeLists.txt)
与 [MLX 构建选项](https://github.com/ml-explore/mlx/blob/v0.32.2/CMakeLists.txt) 是具体来源。

## 7. 如何获得 SOTA 而非宣布 SOTA

四路线都遵守 [Compute 的既定优选规则](compute-design.md#41-开发时验证-best-of准备时应用规则)：
研发时比较成熟完整程序、原生 leaf/融合与嵌入 HYLE 扩展的程序，固化有证据的 hardware/格式/shape/
精度域；准备时核对硬约束并绑定，不现场计时选优。无明确完整收益就保留合格成熟实现；不为一次
地址/顺序波动从头维护整套 kernel。新胜出实现只替换适用规则与局部代码，不重写公共 DAG 或全部后端。

热路径只调用绑定程序。静态调用可模板内联，动态计划的一次固定调用不是字面零成本；不能用宏
抹去 runtime owner、allocator、同步和不可融合边界。分别测 **同 Provider direct 的适配税** 与
**同合同完整片段对最强适用 peer 的成本**，含常驻/准备内存、packing、copy、launch、host submit 与
成功终态；CPU、decode/prefill、不同精度/资源模式分别资格化，不做苹果与橘子的对比。

小程序证明原生约束与接入机制；失败/并发实验、source/static/shared、实际 P1 证明正式实现资格。
没有做过的风险项保留未证明，不从引用成熟库、源码审查或正确性 PASS 推导四路线已 SOTA。
