# 四路线架构定案：研究卡与裁决

## 本轮范围

- baseline：`fac7c9c5`；2026-09-15 用户要求开始剩余架构调研。
- 只研究 CPU、CUDA、MLX/Metal、Vulkan 四条独立路线。模型 CPU+GPU 联合推理明确不在范围内。
- 输出：四条路线的资源/准备/执行/完成方案、共同合同、旧代码复用/改写/删除清单，以及有限未验证项。
- 不实现正式 Provider、不填算子目录、不更改 registry、依赖锁或公共治理工具。旧 goal 不自动恢复。
- 主纵切：真实 FFN 的第三方计算与自定义 kernel 组合；反偏置：两次 GPU compute pass 的资源依赖和重复执行。
- 参考：沿用主审计记录的具名 ggml/llama.cpp、vLLM、SGLang、ds4 pins；MLX v0.32.2，厂商公开 API。
- 预算：源码与必要实验最多 90 分钟；最多两个新独立探针，每个最多两次编译/夹具纠正，不扩大矩阵。
- Apple 探针：MLX GPU MatMul → C++ `fast::metal_kernel` 自定义 SwiGLU → MLX GPU MatMul；变化输入、
  六次重复调用、显式最终完成、double oracle，沿用绝对误差 `1e-4 * max(1, abs(expected))`。
  检查不需要中间 host read/wait，编译缓存 trace 与资源 owner；不据此签零分配、无 JIT 或性能胜负。
- Vulkan 探针：冷建 pipeline/buffers/descriptor/command buffer；两个 storage-buffer compute pass，
  中间明确 compute-write→compute-read barrier，最终 fence 后读取；三次变化输入复用录制命令，
  先等待上轮完成再修改 host input。小整数 F32 运算用精确 oracle；不用删除 barrier 后偶然正确证明无须同步。
- 验证只运行上述原型、docs/closure/diff。无 P1、无产品构建、无故意 GPU hang/device-loss 注入。
- 停止：报告架构能否定案；仍影响共同合同的未知量逐条点名。未完成的正式交付/性能不冒充架构问题，
  也不签产品通过。Windows 是原型来源，远程只运行对应完整 Git archive。

## 研究结果

**2026-09-15：本轮有限研究完成，建议采用下述重构基线，停下交用户审视。** 不是产品 CLOSED，不恢复旧
goal，不签四路线实现或 SOTA。新增两个探针各一次构建即通过，没有追加纠正、P1 或全库构建。

### 从零推导，而不是迁就旧实现

用户在本轮再次明确：**不要被旧 Compute/Graph 影响**。因此先假定产品目录不存在，只从需求和原生
运行时事实推导以下结构；后面的旧代码表只用于迁移，不构成保留授权。

| 独立设计输入 | 最少需要的职责；不因此继承的旧形状 |
|---|---|
| 用户选 CPU/CUDA/Apple/Vulkan，重复运行同一模型配置 | 一个显式部署配置与可准备的有限程序族；不要求用户手填 target/storage/lane ID 数组，也不需要跨 CPU/GPU 组合搜索 |
| C++ 可直接调用计算，也可提交逻辑图 | Compute 负责原生执行/结果/完成，Graph 负责语义图和冷编译；不要求沿用旧 Context/PreparedOperation/Invocation 类名、构造步骤或字段 |
| 成熟算子广度 + HYLE 渐进式性能优势 | 四条路线各自拥有 lowering/程序生成器；原生调用、成熟 region、受支持的内部扩展都可用，不强迫一个万能 Provider 执行 vtable |
| CUDA/Vulkan 稳定资源，MLX runtime 生成 array，CPU 持久 worker | 统一必要的逻辑/错误/借用规则，原生资源与可变状态分别实现；不把四种寿命压进同一公共对象布局 |
| warm replay 低成本 | 原生 Program 的固定入口，必要边界才协调；不要求逐节点经过旧 segment loop、QueueBinding 检查或隐式 prepare |
| 真实并发和异步失败 | 不可变可共享制品、每次运行独占状态、真实完成后的回收；这些来自硬件/运行时事实，不来自旧单测对象模型 |

execution、storage、lane 是必须辨清的事实，不意味着都要成为普通用户手工组装的公开类型。通用默认
旅程只选路线/设备、提供计算与输入、显式准备并重复运行；专家才控制 placement、lane、memory budget
等确有需求的细节。公共头/内部 factor/具体类在新实现卡中按此设计，不为让旧测试继续编译而定型。

当前搜索算法、capture 抽象、通用状态机、构建拆分和大量旧测试均没有预先保留资格。若最佳新设计
只需更短的直接调用链，就直接重写；已有正确性样例只可重新映射为消费者行为验证，不能反过来决定 API。
“复用”只在新职责确定后，以同义、成本合适且生命周期成立为条件；维护旧接口不是本次目标。

### 1. 七个问题的裁决

| 问题 | 采用；拒绝的替代方案 |
|---|---|
| 公共计算语义 | 一套 typed Operation/Graph、精度与副作用合同；各路线独立 lowering。不造四套用户 DAG，也不把第三方私有 op 枚举当公共语义 |
| 全局与局部调度 | HYLE 选择路线、region、边界和并发；Provider 管 region 内部融合/资源/提交。拒绝为了“全由 HYLE 调度”而强拆成熟融合 |
| 资源/执行表达 | target、storage、lane 仍区分；先限制为用户选定的计算路线。host buffer/上传线程不获得 CPU 模型算子资格 |
| 输出与可变状态 | 增加 Provider 生成结果的 owner/lease 合同；固定 caller output 是另一种要求。拒绝把 MLX 输出强制复制到 HYLE 预分配地址，或伪造 allocator 所有权 |
| 完成与回收 | 接纳前可拒绝，接纳后只完成/报错，不重跑；资源活到真实终态。原生完成对象各自实现，不把 callback 或“array ready”直接等同成功 |
| 渐进式 best-of | native leaf、compiled region、**在 Provider 编译前通过受支持扩展插入 HYLE lowering** 三种机制；完整片段有收益才替换。不修改已编译黑盒的内部节点 |
| 共用与特化 | 共用必要语义/验证和边界信息，冷编译器也可按路线特化；CPU/CUDA/MLX-Metal/Vulkan 独立 Program/RunSlot。CMake 裁剪、模板内联、冷绑定各司其职；没有新增 HIP 实现或模型联合路线 |

这里的 Program/RunSlot 是职责名，不预先新增公共类或 component。详细长期合同落在
[Compute](../../../architecture/compute-design.md#35-共同合同与四套可独立特化的实现)与
[Graph](../../../architecture/graph-design.md#63-路线内编译与特化执行)，本卡保留事实、取舍和施工处置。

### 2. 四路线的实际执行方案

| 路线 | 准备与长寿命 owner | 重复执行与性能关键点 | 完成/限制 |
|---|---|---|---|
| CPU | HYLE SIMD/Native、oneDNN 等实际可用 primitive；packed constants；显式持久 executor 与 worker scratch | CPU 专属 call/block 程序；预选 ISA、tile、线程额度；同一请求不每 op 建池。先用成熟完整块作基线，再比较 HYLE 融合 | 同步 primitive 或持久执行 episode；AMX 等每 worker 所需上下文不能随意移到准备线程。现有 ggml CPU 的 fallible pool blocker 不豁免 |
| CUDA | 每 lane 的 stream、可变 vendor handle/workspace、预分配 event；压缩权重及有限 capture variants | cuBLAS/cuBLASLt 与合格量化 kernel 为候选，HYLE 胜出部分进入同一 stream/program；只更新输入/参数，适用时 replay CUDA graph | 每真实提交前缀的 event/错误与回收；不每节点建事件、不在接纳后 fallback。capture 有真实地址/更新/调用限制 |
| MLX/Metal | GPU compiled function、array owners、固定提交线程；可选公开 CustomKernel/Primitive，独立 Metal 路线拥有自己的 pipeline/资源 | 默认尽量保持完整 MLX region；HYLE Metal lowering 可在编译前插入其中，避免先拆图再桥接。managed array/metadata/copy 成本如实计入 | MLX 最终错误必须检查；持久线程上的 checked synchronize 是可用但有成本的保底桥。任意外部 MTLBuffer/Queue 零拷贝未承诺；需要时单独资格化 |
| Vulkan | SPIR-V/pipeline、descriptor/command pools、每并发槽的 buffers、录制程序与 fence/semaphore | 独立 Vulkan 程序；稳定 binding 复用录制，动态数据进参数 buffer；显式 compute memory barrier。不能用 CUDA capture 接口假装实现 | pending command/descriptor/resource 不复用；按实际 coherence/queue-family 处理。原生 Windows 实验通过，不外推 MoltenVK 或其它 GPU |

GPU 路线的 host 工作仍可用 Sched/Platform/Mem，但不是 CPU 模型计算。MLX 中插入的是 **Metal GPU
kernel**，不是把已有 CPU SIMD 函数自动变成 GPU kernel。已有 SIMD 优化仍供 CPU Provider 直接使用。

### 3. 新实验与明确未证明项

完整源码、命令、环境与原始输出见[实验目录](../../../../evidence/correctness/compute-four-route-admission/README.md)。

- **MLX v0.32.2 / MacBook Air**：F32 FFN，`M=4, K=1024, H=256`；MLX gate/up MatMul → 公开
  `fast::metal_kernel` SwiGLU → MLX down MatMul。六次变化输入全部通过 double oracle，最大绝对误差
  `5.85924300814e-08`；同线程 trace 始终为 1；应用代码没有中间 host read/wait，只在最终输出处同步。
  证明同 runtime 的组合路径成立，不证明 kernel 胜出、无 JIT、零 metadata/allocator 成本或 HYLE adapter 已交付。
- **Vulkan / RTX 4070**：两个 storage-buffer compute pass，明确 write→read barrier；命令录制一次，
  三次变化输入各 1,024 个输出精确通过。重复阶段没有夹具资源创建或重新录制；driver 成本未测。
  这是资源/同步反偏置探针，不是模型性能用例，不证明 validation-layer、device-loss、跨队列或 MoltenVK。
- **复用而未重跑**：既有 CPU 真计算，以及 CUDA 压缩 Marlin 探针的九次变化输入、压缩权重与 workspace
  事实。旧独立 Marlin 并非最新版 vLLM Marlin，也不能冒充正式 Provider 或严格 FP32 数值资格。

MLX 源码进一步限制结论：`CustomKernel::eval_gpu` 会分配输出，默认可能整理输入布局，并查询
library/kernel cache；generic custom kernel 也可能成为融合边界。公开 C++ Primitive/metallib 扩展可供
更低层实现，但仍须按完整 region 验证，不因为使用了扩展就推定更快。
[公开扩展](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)、
[锁定版本实现](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp)。

### 4. 旧代码的处置依据

审计基线是 `fac7c9c5`；“保留”指保留有消费者依据的语义/实现候选，仍要验证重构后的调用链，不是本轮
产品 PASS。本轮没有执行下面的产品改动，也没有删除已有优化。

| 现有入口/事实 | 处置与理由 |
|---|---|
| `compute/tensor_desc.h` 的 logical dtype 与 Plain/Blocked/Opaque、稳定 StorageFormatId | **复用**编码分层；它不是量化数值 schema。禁止仅靠 Q4 字样把不同 scale/zero/packing 混用 |
| `compute/operation.h` 的 typed schema、resource access；普通 MatMul 只有两个输入 | **复用并补明确量化 schema**：activation、codes、scales、可选 zeros 与 group/rounding/accumulation 合同，不向属性塞指针 |
| `graph/region_provider.h::ProviderRegionPreparedVTable` 的 shared/delegated 都接收预先给定的 output TensorView；`OutputOwnership` 只有 caller/source/workspace/constant | **改写边界输出合同**：支持运行时 Provider 生成结果及 owner；Graph 不为这类结果预分配影子 payload。保留显式 caller-output/Copy 路线 |
| 旧 direct Compute 只接受共享 Queue，delegated 只能经 Graph 私有入口 | **删除这个架构限制**：直接计算也可绑定独立 runtime owner，返回其结果/Completion；不能因为旧参数列表而让单 op 用户依赖 HYLE Graph，不做忽略 Queue 的伪适配 |
| `graph/executable.cc::Invocation::submit` 在未准备或 Queue 改变时调用 `prepare_replay` | **移除低延迟入口中的隐式准备**；显式 prepare/bind 后固定执行。方便入口若保留须明确冷行为，不强迫严格路径经过它 |
| `RouteSearch/ResourceSearch` 枚举节点×target×Provider/存储，公共 InvocationState 保存通用队列/segment 状态 | **不保留为默认骨架**：从路线专属 compiler/program 重写；唯一可行 route 快路，成熟完整 region 作 incumbent、少量合格替代。只有具体图确需组合搜索才实现相应有界算法，不先把旧求解器搬过去 |
| `Invocation::submit(span<QueueBinding>)` 逐次验证队列并循环 segment；prevalidated op 仍有安全检查 | **生成路线专属入口**，把不变验证移到 bind/prepare，不把通用 segment 循环强加给原生整体程序；必要 in-flight/alias/接纳检查不能为测速删除 |
| `graph.cc`、`effect_order.cc`、shape program、常量 artifact 与 Mem lifetime planner | **复用必要语义**；lifetime planner 只规划自己拥有的 backing，Provider 结果/私有 storage 按实际 owner 记账。不能把 managed unknown 当已预留字节 |
| Native `parallel_for_ranges`、oneDNN BRGEMM 的冷 JIT/任务 scratch/ExecutorRef | **复用并组合 CPU episode**；真正持久 Sched/Platform bridge 单独完成，避免重复线程池/OS 探测。按完整块查过多 barrier/wakeup，源码计数不直接签性能 FAIL |
| CUDA backend 的 per-Queue event pool、sequence 接纳/完成、待完成回收 | **复用正确生命周期方向**；native Program 直接复用 stream/vendor 资源。event record 失败也要跟踪实际已接纳工作，不能提前归还 buffer |
| `CommandCaptureVTable` 的 begin/end/abort/launch | **限于真实支持的原生路线**；Vulkan 用录制程序，MLX 用自己的执行合同，不补一套假的 stream capture 兼容层 |
| “模型 CPU+GPU 自动分层/联合计划”的要求与旧进度 | **从当前范围移除**。现有通用 multi-target 字段不等于已实现 offload 引擎；不虚报删除不存在的功能。`kHybrid` 是 capture+commands，不机械删除 |

没有发现需要为了本轮架构去重写 Error、SIMD 或 Mem allocation primitive 的证据。Provider result lease
首先属于 Compute/Graph 的计算结果与 runtime lifetime 合同；如需表示实际原生 allocation，复用 Mem 的
typed resource/`Buffer::adopt`，但不能把尚不存在的 array storage 伪装成固定 allocation token。

### 5. CUDA Q4 MatMul 的具体路线

新的量化语义以 `QuantizedMatMul` 职责表达：显式 codes/scales/可选 zeros；scheme、group axis/size、
logical orientation、rounding 与 accumulation policy。名称/字段最终在对应 API 小卡锁定，不为某个
benchmark 的参数列表定公共形状。GGUF Q4_0/Q4_K 与 Marlin packing 是不同物理/数值合同；相同 bit 数
不保证可无损互转。仅允许语义等价的冷 repack，精度改变需要显式允许，不能悄悄 requantize 或展开整份权重。

准备期先查所选 CUDA device/精度/格式/shape，再决定原生 vendor/kernel 或完整 Provider region；只有
真实适配存在的候选才可选。不可变 packed weight/scales 可跨变体共享；output/scratch/locks 属于并发
run-slot。重复执行只填参数并提交，不查 GPU 属性、不重新选 tactic 或构造 Torch tensor façade。

当前 vLLM Marlin 有 raw-pointer launcher，但所在 host 层耦合 Torch，且 launcher 仍查询设备属性；
需要独立受审源码 factor 把配置工作移到 prepare，不能直接称“无成本导入最新版”。其
`use_fp32_reduce` 也不代表全部中间累加都为 FP32。不同数值政策的候选必须拒绝或分开比较。
[Marlin 源码](https://github.com/vllm-project/vllm/blob/51918e252af4dfff381001f42b7ebe6a0454501b/csrc/libtorch_stable/quantization/marlin/marlin.cu)。

### 6. 性能方法与有限后续施工

先据源码消除错误成本结构：强制输出 copy、每 op 资源创建/设备查询、隐式 prepare、反复路由、破坏
融合与不必要 host wait。保留有用并发/错误/lifetime，不把分支数量或四套宏当性能结论。
稳定候选再分别测 **同 Provider 的 direct 抽象税**和 **同语义完整片段竞争力**；准备、host submit、
最终完成、常驻/峰值内存分别报告。只测 kernel、不含对手/本方相同 copy/sync/精度的对比不能用于选路。
无可分辨优势用成熟默认；四条路线可有不同胜者，不进行整个 ops 目录的反复纳秒裁决。

用户确认后，按以下依赖顺序重新开有限实现卡，不复用旧 89% 当进度：

1. **共同边界迁移**：固定路线、Provider output owner/lease、显式 prepare 与冷/热状态分离；先用已有
   CPU/CUDA 对照完成绑定/失败/生命周期和 direct 税回归。新 lease 尚未在 HYLE 产品内实现或验证。
   同时使直接 Provider-runtime 执行不依赖 HYLE Graph；对象名和 target 拆分不从旧代码继承。
2. **四个原生执行物**：CPU 持久 episode；CUDA typed Q4/原生 façade；MLX 同-runtime lowering、输出
   owner 与 checked completion；Vulkan pipeline/program/同步，分别接入 build/package。每条只保留
   足以验架构的主纵切，不先扩大全目录。MLX/Vulkan 不互相冒充资格，MoltenVK 单列验证。
3. **基础终局**：各路线的完整片段、公平性能、并发槽、部分接纳/晚到错误/销毁与三交付；必要的
   CPU/GPU host transfer 计入，不恢复模型联合计算。全部基础通过后先报告，再谈算子广度。

未完成项是上述正式 adapter/执行器/交付/资格，不是“等实现后再决定谁拥有输出或谁调度”。不能承诺
未来绝无修改；若公开 runtime API 无法兑现已写明的合同，应先拒绝那条候选或采用已声明有成本的桥，
只有真实反例推翻共同语义才重开共同架构，不因新增 op 或局部优化回到全架构研究。

### 7. 参考与验证记录

- llama.cpp（含 ggml）`4c9233c034fc450dcf34c7c0988aebe6da5cdf1a`；standalone ggml
  `456172ec733a135778adcd32d00e576a58232e45`；vLLM `51918e252af4dfff381001f42b7ebe6a0454501b`；
  SGLang `06992c92ad4494fe9f9636e75be3e35040512447`；ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc`。
- MLX v0.32.2 `1f8e74e3f12f31365464a6867c6579f0e9b29d85`；实验与源码同版本，不将本地 main 混作稳定发行。
- CPU/线程、CUDA capture、Vulkan 生命周期的具体源码入口及厂商链接沿用
  [主审计记录](compute-graph-route-unification-audit.md#本轮架构分析统一合同与独立执行)。本轮不改依赖锁。
- 验证：两个预声明原型结果如上；以 `fac7c9c5` 为 baseline，change-impact selector 只选择 docs/closure，
  `check-docs.py`、`check-module-closure.py` 与 `git diff --check` 通过；其余 94 项有明确 skipped reason。
  产品正确性/P1/完整交付未运行，不签任何模块 PASS 或状态变化。
