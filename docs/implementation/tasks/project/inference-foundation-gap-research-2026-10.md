# 推理引擎实际需求与 HYLE 基础机制缺口调研

## 本轮合同

- 状态：`RESEARCH_COMPLETE`；本卡不是产品模块 closure card，不重开 CLOSED owner。
- 授权：2026-10-11 用户要求新建 goal，更新本地 SGLang/vLLM/llama.cpp 并调研
  **推理引擎实际需要、HYLE 尚未提供、且属于基础模块**的功能。
  用户随后明确纳入：HYLE 已有实现/接口与引擎实际消费形状对不上的部分；不能只查缺少的名字。
- HYLE baseline：`00fdc40c9311d862055591af3f0836d40e593dc7`，初始工作树干净；60 个组件均 CLOSED。
- 主纵切：本地量化推理的加载、准备、prefill/decode、状态更新、资源流送及取消/关闭。
- 正交纵切：异构渲染/资源流送和批处理数值管线；用于检查候选机制是否依赖 token/模型策略。
- 依据：[能力完整性](../../../architecture/capability-completeness.md)、
  [推理引擎北极星](../../../architecture/inference-engine-north-star.md)、
  [当前 Compute/Graph 边界](../../../architecture/compute-next-stage.md)。

### 四个有限工作包

| 包 | 工作 | 完成条件 |
|---|---|---|
| S0 | 核对、更新三个参考仓库 | 保留修改，使用 fast-forward；记录更新前后完整 SHA、分支、remote 和工作树 |
| S1 | 按真实调用链提取机制 | 六个旅程各列实际源码 owner、关键动作、生命周期与失败/并发边界 |
| S2 | 对照 HYLE 公开能力 | 区分已有完整路线、缺少组合层、缺少基础机制、引擎策略和未证实候选 |
| S3 | 汇总优先级与有限后续建议 | 有来源和成本边界的缺口表、可复用/借成熟库方式、负空间及文档验证 |

调查覆盖六个旅程：①文件/权重到计算资源；②冷准备与执行计划/实例复用；
③动态 batch 与分块/分页状态消费；④状态共享、分支、接受/回退；
⑤驻留/流送、预算与异步安全回收；⑥取消、排空、关闭及诊断。
不只验证上一条回复提出的四项；候选必须来自真实调用路径。

### 预算与停止

- 目标用一轮源码阅读完成：每库不超过 12 个主文件，至多 12 个跨库定向补读；
  主调查不超过 60 分钟。超预算或仅有领域专用证据的候选明确 DEFER，不追加无界扫描。
- 不编译这三个大项目、不下载模型、不运行推理/P1、不扩算子、不改变 HYLE 依赖锁。
- 不创建空组件或预设实现方案。通用增量图仍是已登记待办，不重复设计。
- 输出是调研事实与建议，不是 API/平台/性能完成证书；不能把原子存在当作完整复合旅程。
- 候选按 `KEEP / COMPOSE / INVESTIGATE / APPLICATION / DEFER` 裁决；
  新基础库实现须另行定界、登记 owner 和有限 FIX。
- 完成后执行 docs/closure 与 change-impact selector，文档修改不构建、不跑性能。

## 参考仓库身份

Windows 的以下参考 checkout 仅用于源码研究，与 HYLE 锁定产品依赖分离。
初始三个工作树均干净；更新只 fetch 当前默认分支并 fast-forward，不 reset/clean。

| 仓库 | 分支 | 更新前 HEAD | 更新后 HEAD |
|---|---|---|---|
| `C:\public\project-hyle\sglang` | main | `06992c92ad4494fe9f9636e75be3e35040512447` | `234ccba7f625e12639e621044496bcc6197c3758` |
| `C:\public\project-hyle\vllm` | main | `51918e252af4dfff381001f42b7ebe6a0454501b` | `276fbcff2717bd934cfa37c8a2e4c391f3e7237b` |
| `C:\public\project-hyle\llama.cpp` | master | `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a` | `23b0202a189c44a54625aadcb37a946dd1d6278d` |


## 1. 总结：三项优先缺口，两项后续候选

三个更新后的引擎都有大量模型/请求专用代码，但并不能因此把它们全部下沉到 HYLE。
按实际 ownership、完成边界和当前公开接口对照，本轮建议：

- **优先候选 F1–F3**：外部数据块的共享与完成后回收、准备后的批量范围搬运、有界文件到计算资源管线。
  它们缺的是完整组合入口，已有 Mem/Sched/IO/Compute 原子应直接复用。
- **后续候选 F4–F5**：显式状态集合的版本/分支，及分页/变长计算的成熟 Provider 接缝。
  已发现真实消费者，但还不能把模型回滚或某个 CUDA Attention 参数表直接当通用基础 API。
- 执行计划复用、基础预算、异构传输、取消/排空、mapping 和普通随机数已经存在。
  不再为这些名称另起一套基础设施。
- 通用增量图继续沿已有 DEFER；这五项不是新的产品准入或施工授权。

“缺口”指引擎目前仍需自写的通用生命周期组合或缺少的公开消费形状，
不表示已经完成的单模型数学目录失效，也不宣称本轮已测试所有 serving 形态。

## 2. 六个真实旅程与已读源码

来源 URL 固定完整 SHA 和行号；本地阅读关键实现片段及调用线索，没有运行三个引擎。
原始仓库身份和文件 SHA256 见
[源码身份收据](../../../../evidence/architecture/inference-foundation-gap-research/source-manifest.json)。

| 旅程 | SGLang | vLLM | llama.cpp | 可以提取的机制，及不能混入的策略 |
|---|---|---|---|---|
| 文件/权重到资源 | S5 分片滑动窗口；成熟 loader 的关闭 | V6 ParallelLoader 与有界 producer/consumer | L4 对齐读取、四 staging buffers、原生 event 保护复用和最后排空 | 有界管线、partial progress、backing lifetime；tensor 命名/文件 schema 和权重角色留在 loader |
| 冷准备、重复执行 | S6 预先 capture 的 bucket 与每步 staging | V5 预生成 descriptor/candidates、capture 与 replay | L5 reserve、L7 参数/输入复用判断 | 已编译程序与可变状态分离；模型分桶/padding 和批处理选择留给引擎 |
| 动态 batch/分块状态 | S4 paged allocator 与 S7 indptr/页索引 | V1 block pool、V2 不同状态类型、Attention adapter | L3 KV cell/stream 与 L1 memory context | 数据范围、边界与生命周期可通用；token/sequence/page-table policy 不属于 Mem |
| 共享、分支、回退 | S1 Mamba slot copy 对 checkpoint 的限制 | V2 CoW 及 retained copy source | L2 有限 recurrent rollback；L3 同 stream 共享、跨 stream copy | 保留具体状态版本、只写独占 backing；任意 token 回退不能由通用 copy 推导 |
| 驻留/流送和回收 | S2 producer event、layout move gate、临时索引保活 | V3 批量 CopyPlan、方向性屏障、event 后复用；V7 成功发布/失败回收 | L6 split 输入复制及 reuse event | 完成依赖、范围搬运和明确资源账单；命中、预取距离、LRU 和专家选择留给引擎 |
| 取消、关闭与诊断 | S3 stop/队列排空、线程 join，并检查未停线程 | V4 deferred-free fence；V3 shutdown drain | L5 context 析构同步；server queue 的 cancel 是领域动作 | 停止接纳、排空在途、原错误和安全回收；请求状态/重试/响应协议仍在引擎 |

### 关键源码索引

| ID | 固定源码 | 本轮核实的动作 |
|---|---|---|
| S1 | [memory_pool.py](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/mem_cache/memory_pool.py#L1148) | Mamba clone 涵盖 conv/temporal/sibling；ReplaySSM 源必须已 flush，不能通用撤销未保存状态 |
| S2 | [l2_transfer.py](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/mem_cache/l2_transfer.py#L103) | producer event 后解析索引，获得 layout protection；finish event 后释放并 record_stream 保活 |
| S3 | [cache_controller.py](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/managers/cache_controller.py#L489) | 停止 storage threads、唤醒/排空、join，仍存活时明确失败 |
| S4 | [allocator/paged.py](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/mem_cache/allocator/paged.py#L335) | 按页释放与 page-aligned segment；元数据不能在 deferred free 前被调用方改写 |
| S5 | [weight_utils.py](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/model_loader/weight_utils.py#L1327) | 分片并发滑动窗口，mmap/非 mmap，逐分片资源释放；文件窗口不等于字节硬预算 |
| S6 | [decode_cuda_graph_runner.py](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py#L1453) | raw/padded batch、ragged tier、输入 buffer 更新；不是运行期比较 Provider |
| S7 | [flashinfer_backend.py](https://github.com/sgl-project/sglang/blob/234ccba7f625e12639e621044496bcc6197c3758/python/sglang/srt/layers/attention/flashinfer_backend.py#L213) | 缓存的原生模块与 host-known indptr 避免每步 D2H；借私有内部 plan 是升级敏感接缝 |
| V1 | [block_pool.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/core/block_pool.py#L666) | block refcount、touch、独占写、free-list 与 reuse watcher；这些引用计数本身不证明 GPU 已完成 |
| V2 | [single_type_kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/core/single_type_kv_cache_manager.py#L425) | CoW copy 排队、source retention、boundary state；模型状态类型有不同回退/可写条件 |
| V3 | [cpu/gpu_worker.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/kv_offload/cpu/gpu_worker.py#L640) | compute/copy 依赖、原生 event、批量范围和 metadata 保活；完成后归还 stream/event/buffers |
| V4 | [sched/scheduler.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/core/sched/scheduler.py#L2758) | 请求结束/CoW source 释放延后到真实已处理 fence step；队列前端未完成允许保守延迟释放 |
| V5 | [gpu/cudagraph_utils.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/worker/gpu/cudagraph_utils.py#L104) | 有限兼容 descriptor、冷 capture、runtime lookup/replay，跨 offloader 时间线仍须同步 |
| V6 | [weight_utils.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/model_executor/model_loader/weight_utils.py#L1283) | 借 fastsafetensors ParallelLoader 的管线，获取错误的 fallback 只允许尚未 yield 时 |
| V7 | [cpu/manager.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/kv_offload/cpu/manager.py#L366) | pending store 的成功发布、失败清理与 eviction readiness |
| V8 | [topk_topp_sampler.py](https://github.com/vllm-project/vllm/blob/276fbcff2717bd934cfa37c8a2e4c391f3e7237b/vllm/v1/sample/ops/topk_topp_sampler.py#L318) | per-request generator 与设备抽样是两层；CPU 分支与不同 backend 不承诺同一随机序列 |
| L1 | [llama-memory.h](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-memory.h#L80) | 分型 memory context；prepare/apply/状态，明确 token/sequence 专用语义 |
| L2 | [llama-memory-recurrent.cpp](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-memory-recurrent.cpp#L168) | 有限已保存 snapshot 索引的 partial rollback；共享 cell 不允许单一 sequence 独自倒退 |
| L3 | [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-kv-cache.cpp#L469) | 同 stream 拷贝 sequence metadata，共享 payload；跨 stream 登记实际数据复制 |
| L4 | [llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-model-loader.cpp#L1509) | bounded staging、能力检查、event 后读取复用、异步上传，最终同步/销毁 |
| L5 | [llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-context.cpp#L624) | 最坏图资源 reserve，另见 492 行析构同步；模型 probe/批处理归 context |
| L6 | [ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/ggml/src/ggml-backend.cpp#L1808) | 保留原 scheduler split 的 copy/event 合同，输入借用和覆盖时序不同 |
| L7 | [llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-graph.cpp#L1463) | 参数与每个实际输入的 can_reuse；图身份不能只看尺寸 |
| L8 | [llama-sampler.cpp](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-sampler.cpp#L217) | CPU 标准 RNG/概率抽样；sampler clone 与领域处理不属于通用资源机制 |

## 3. 对照 HYLE：缺口与已有能力

本表核对公开头/实际承载，不以历史 CLOSED 卡代替接口。COMPLETE 只描述已声明的基础机制，
不代签整引擎集成或未验证平台。F1–F5 仍为建议，不修改 registry。

| ID / 能力族 | HYLE 现状及公开路线 | 状态 / 建议 | 引擎仍须拥有的部分 |
|---|---|---|---|
| F1 外部块共享、独占写与完成后回收 | [BufferPool](../../../../include/hyle/mem/buffer_pool.h) 租约销毁立即归还，明确要求 caller 先等完成；[RangeAllocation](../../../../include/hyle/mem/range_allocator.h) 管 offset 不管理设备同步；[Ref](../../../../include/hyle/util/ref_counted.h) 能共享 owner，但不绑定原生完成；WeightBatch/Graph Results 仅覆盖各自固定槽 | **PARTIAL / COMPOSE，优先级 1**：缺少任意外部数据块共享与异步安全回收的完整组合旅程 | prefix identity、页表、淘汰顺序、哪些块仍被业务引用 |
| F2 准备后的批量范围搬运 | [copy_async](../../../../include/hyle/compute/transfer.h) 每次一个范围/Completion；Vulkan [TransferSlot](../../../../include/hyle/compute/vulkan_memory.h) 是固定完整 tensor staging；原三路 [WeightSlots](../../../../include/hyle/compute/cuda_weights.h) 支持固定 catalog/两槽上传，不是通用页范围搬运 | **PARTIAL / COMPOSE，优先级 1**：缺少一批源/目标范围、固定 metadata 容量及一个批次完成/失败边界；不得靠 N 次 prepare/提交包装冒充低成本批量 | 运行时 block/expert IDs、数据放置和所选范围 |
| F3 有界文件到计算资源管线 | [File/read_exact_at](../../../../include/hyle/io/file.h)、[mapping](../../../../include/hyle/mem/mapping.h)、TaskScope、Channel、native copy/WeightSlots 都有；尚无 public 完整 read→stage→upload→ready→消费→回收 owner | **PARTIAL / COMPOSE，优先级 1**：引擎还需重复编写槽生命周期、partial read/error、背压、取消与 GPU 排空 | 文件格式/分片索引/权重角色、预取策略、哪些数据值得驻留 |
| F4 显式状态集合的版本/分支 | [GatedDeltaNet](../../../../include/hyle/compute/operation.h) 有 unchanged initial state 和 materialized final state；[Results](../../../../include/hyle/graph/cpu_cuda.h) 保护本 invocation 的结果；无外部 mutable 状态集合的共同版本发布/clone 入口 | **PARTIAL / INVESTIGATE，优先级 2**：普通 out-of-place 分支可用已有原语；是否新增集合级 helper 要由真实混合状态 consumer 决定，不准许任意回滚承诺 | 哪些 tensor/metadata 构成完整模型状态、有效 checkpoint、MTP 接受范围和 RNG/token 回退 |
| F5 分页/变长数据的成熟计算接缝 | [TensorDesc](../../../../include/hyle/compute/tensor_desc.h) 支持固定逻辑 shape、byte strides、实际 block encoding；[CompositeBuffer](../../../../include/hyle/mem/buffer.h) 已有分段字节存储，但普通 [TensorView](../../../../include/hyle/compute/tensor.h) 绑定单 BufferView；公开 SDPA 无 indptr/page-table，当前 [FlashInfer](../../../../include/hyle/compute/providers/flashinfer.h) 为有限 dense BSHD 域 | **PARTIAL / INVESTIGATE，优先级 2**：模型数学可 dense 运行，不代表已有 paged/ragged serving 的自然成熟 region。先裁决原 Provider region 的端口/动态 metadata，不建万能 Tensor/Queue | token/sequence 组批、逻辑 KV 页映射、padding 有效区、模型专用压缩状态 |
| K1 冷编译、shape specialization 和重复提交 | [Executable/Invocation](../../../../include/hyle/graph/executable.h) 已分离 immutable/shared plan 与独占 mutable slot；有限 exact variants/general 支持冷准备、固定复用 | **COMPLETE / KEEP**；容量受限的 warm Invocation 借还可先由消费层组合，不能把 ObjectPool 的内存复用当 workspace/capture 保留 | capture bucket、未命中时何时冷编译、哪些请求拼成一个 batch |
| K2 host/device/staging 与已知/未知账单 | [ResourceBudget](../../../../include/hyle/compute/resource_budget.h)、CPU/GPU 组合资源报告和独立普通 CPU 存储 | **COMPLETE / KEEP**，只在原明确预算范围成立 | 模型规模估计、显存 offload 比例、系统余量与其他进程争用 |
| K3 host/GPU 依赖、专家片段上传、异构粗片段 | [cpu_cuda](../../../../include/hyle/graph/cpu_cuda.h)、[cpu_apple](../../../../include/hyle/graph/cpu_apple.h)、[cpu_vulkan](../../../../include/hyle/graph/cpu_vulkan.h) 与三路 weights/native resources | **COMPLETE / KEEP**，CPU+一个 GPU 原合同；不等于任意可变 KV streaming | 层/专家放置、命中/驻留/淘汰、动态专家预测、多 GPU 拓扑 |
| K4 task 取消、deadline、背压和排空 | [TaskScope](../../../../include/hyle/sched/task_scope.h)、[Channel/Semaphore](../../../../include/hyle/sched/sync.h)、IO OperationOptions 与 native Completion | **COMPLETE / KEEP**；新 F1–F3 组合应复用它们，没有第二个 Scheduler | 请求 SLO、超时响应、continuous batching、抢占/优先级策略 |
| K5 分段 backing、mapping、NUMA/对齐和文件 IO | Mem/Platform/IO 现有公开路线 | **COMPLETE / KEEP**；动态 share/move 安全性仍看 F1，不因名字叫分页重写 Mem | KV/资产的业务布局、格式解释 |
| K6 普通 CPU 可复现随机源 | [Rng](../../../../include/hyle/util/random.h) 显式 seed、state 保存/恢复、jump/long_jump，组合标准概率分布；没有公共 counter-addressed GPU 随机算子 | **CPU COMPLETE / KEEP；GPU 扩展 DEFER**：GPU 抽样可由成熟 Provider/引擎层承担；未证明必须新增统一跨路线 RNG | sampling、top-p/min-p/penalty/grammar/停止条件；不能许诺跨 Provider 同 seed 同 token |
| A1 模型/schema 与 serving 规则 | README/北极星明确在消费层 | **OUT_OF_SCOPE / APPLICATION** | GGUF/Safetensors 解释、tensor 名称、tokenizer、模型结构、请求/batch/prefix/专家策略 |

特别纠正上一条建议：执行计划复用已经交付；“分块存储”也已经有范围/分段 backing。
真正缺口是**共享和完成后回收的组合**，不能据此重写已有分配器。
当前 RCU [ReclamationDomain](../../../../include/hyle/mem/reclamation.h) 等待 CPU reader quiescence；
没有证据表明它等待 CUDA/Metal/Vulkan completion，不能用它直接证明 GPU 安全回收。

## 4. 已有实现与引擎消费形状的适配差异

这部分追加只读实现核对，区分合同限制、组合缺口与实现缺陷。未运行真实引擎，
因此不把静态适配差异写成可复现 correctness/performance FAIL；也不自动重开 CLOSED owner。

| ID | 引擎实际消费形状 | HYLE 实际接口/实现 | 裁决和可行路线 |
|---|---|---|---|
| M1 每步 batch/有效长度/页索引变化 | S6/V5 每步更新真实有效区与固定 capture bucket；S7 分开 host-known metadata 与 GPU 索引 | 普通 Graph 能预编 declared-domain/有限 exact variants，但每个 Invocation 的具体 shape 固定；FlashInfer [propose](../../../../src/compute_provider_flashinfer/flashinfer_provider.cc) 明确 exact-only。CPU+Apple/Vulkan [compile](../../../../src/graph_route_cpu_apple/compile.cc) 与 [compile](../../../../src/graph_route_cpu_vulkan/compile.cc) 明确拒绝 symbolic/effects/control。CPU+CUDA [compile](../../../../src/graph_route_cpu_cuda/compile.cc) 使用共同 compiler，不能把前两条的限制套给 CUDA | **消费适配不对称，关联 F5**。固定 bucket+显式有效区仅在所选数学/Provider 有真实长度/索引语义时成立；不能仅改 shape/padding 冒充。有通用 domain 的现有路线复用原接口，其余用有限 exact plans；未来 paged/ragged native 接缝独立定界 |
| M2 单个请求终止，copy/compute 仍在引用外部块 | V4 把取消/结束请求的块从业务记录移出，晚于 fence 才回 pool | Graph Results 保留私有 workspace；[CPU+CUDA Results](../../../../include/hyle/graph/cpu_cuda.h) 明确外部输入/输出/resources 仍借用；[TensorView](../../../../include/hyle/compute/tensor.h) 不拥有 payload；RCU 是 CPU reader 安全 | **组合生命周期缺口，F1**。不能靠 Results/Ref/RCU 任一单项替代外部块最后使用协议；已有 owner 内部安全性没有因此被判错 |
| M3 已在 pinned/shared/device backing 的数据希望直接按范围流送 | V3 用源/目标 pointer ranges 和一个完成边界；S2 有明确页 layout gate | CUDA [WeightSlots::upload](../../../../src/compute_route_cuda/weights.cc) 校验完整 catalog TensorDesc，同步 storage.read 到自有 staging，之后一次 native H2D；两固定槽。Apple/Vulkan 也有各自真实 staging/ready。一个 upload 可选多个 fragment，**不是逐 fragment GPU submit** | **现有权重旅程正确但不能当任意可变页传输，F2/F3**。同步 host staging 是已声明成本；不能宣称全程异步/零拷贝，也不能仅因此判为 bug。保留权重入口，另用可受审的原生范围旅程覆盖实际场景 |
| M4 分页/分段 backing 已有，但成熟数学需要 indptr/page table | S4/S7、vLLM FlashInfer adapter 按物理页索引和 tail/indptr 消费，省去全量 gather | CompositeBuffer 可表达分段 bytes；普通 TensorView 只有一个 BufferView；SDPA/当前 FlashInfer route 没有页索引/变长端口。typed Gather/dense materialization 是另一种消费形状 | **计算消费接缝缺口，F5**，不是“Mem 没分页”。先保留成熟 Provider 原 region 与 metadata；不得强迫所有路线使用万能 paged Tensor 或用全量展开伪装同成本路线 |
| M5 GPU copy 异步，但文件/host 准备可能阻塞 | L4 等 slot event 后同步读下一文件块；S5 使用有界 loader window；V6 借成熟 producer/consumer | IO read_exact_at 可组合任务，WeightSlots upload 的同步 host 准备仍在调用线程；[IO guide](../../../user/io.md) 明确 epoll/kqueue blocking file fallback 不支持可靠逐操作 deadline/cancel | **进展形状需要完整组合，F3**。新管线复用现有 blocking handoff/TaskScope，关闭须等真实已接受 IO/native 退役；不能把外层 timeout 当底层已停止，也不扩大成另一套 IO runtime |
| M6 保存一组模型状态后做 speculative branch | S1 的 flushed checkpoint、L2 的 bounded rollback、V2 的 CoW 都与实际状态格式有关 | 纯 GatedDeltaNet 显式 initial/final state 本就可 out-of-place 分支；Apple/Vulkan hybrid 不接受显式 effects；普通 Graph 的 effects/alias 声明不会自动保存旧内容 | **集合级可选辅助，F4**。先沿纯输入/输出组合；只有必须克隆的显式外部状态集合才考虑 helper。token/RNG/metadata 由引擎纳入，不许诺没保存过的状态可回退 |
| M7 严格总内存上限与 Provider 私有成本 | 引擎须把 weights、KV、capture、workspace 和在途槽一起估计/接纳 | ResourceBudget 明确 requested managed bytes；Apple array 的私有 descriptor/allocator cache 可能 unknown/excluded，native 分配和系统占用也不是同一预算 | **已声明边界，KEEP**。不得向用户保证“传一个显存限额便绝不 OOM”；沿已知/未知/排除报告决策，部署余量与放置策略归引擎，不新写自动 placer |

M1/M3/M4 是接口/能力域不吻合，M2/M5 是尚缺完整组合，M6 是需要更具体消费者的可选范围，
M7 是合理且必须透明的边界。本轮没有发现足以独立裁决为实现 bug 的反例；这不等于已经证明
整引擎接入没有 bug。特别是 CUDA 与 Apple/Vulkan 的 symbolic/effect 接缝不能以同一张完成标签推断一致。

## 5. 三项优先候选的最小机制边界

### F1：外部资源租约与延迟回收

- 旅程：取得有限块/范围→共享只读→提交实际 native 使用→登记所有最后使用完成→业务引用释放→
  完成可证后归还；请求取消或丢弃 handle 不能提前覆盖 backing。
- 复用：Mem Buffer/RangeAllocator、Util Ref、既有 Completion 和各 native retirement owner。
  Mem 不反向依赖 Compute；组合层放在能合法拥有生命周期的较高 owner。
- 实现候选：HYLE 自己拥有少量 ownership/retirement glue，成熟 runtime 仍拥有原生完成。
  不复制 vLLM prefix hash/LRU 或 SGLang radix cache。
- 必须先明确：共享只读/独占写、旧 generation、多个使用者、accepted-failure/未知完成、
  有界 pending metadata、非阻塞回收与显式 drain；未证明完成不能自动释放。
- 正交任务：渲染资产块上传并被多个 in-flight frame 使用；数据管线同时消费同一分块。
- 成本：每个活跃块/在途使用的有限控制记录；只在选用组合时付费，不给所有 Buffer 添加原生 fence。
  引用计数为零和 GPU 完成是两条独立条件。

### F2：原生批量范围搬运与固定计划

- 旅程：冷准备固定容量和源/目标资源域→每次绑定有限范围/IDs→校验范围/别名/编码/对齐→
  原路线批量提交→一个可观察完成边界→重复执行或明确退休。
- 复用：现有 Queue/原生 command/copy 实现、ResourceBudget 与 Completion；Provider 数学仍原版。
  不要求三条 GPU 路线共享 executor/vtable；有 batching native API 时借它，无则在同一原生提交内
  发出有限 copy commands，不能以改名的逐范围 prepare/Completion 循环声称已解决。
- F1 保护 payload 和索引/control buffers；F2 不决定 cache、LRU 或移动目标。
- 不允许原编码块半截复制后伪装合法 tensor；单位是明确 byte range 或完整受审编码块，
  特定 repack/转换是已有 typed operation/Provider 的另一项工作。
- 验收应关注：N 个范围、重复/空/坏索引、overlap、部分已接受后失败、source/dest 保活、
  低成本 replay 和真实提交/分配计数。研究阶段没有性能 PASS 或平台完成承诺。

### F3：有界 read→stage→native-ready 管线

- 旅程：consumer 给已有文件句柄、范围和目标资源→最多 K 个槽及明确 byte budget→
  完整读取→native copy→ready→消费→最后使用后回收；失败/取消关闭接纳并排空已接受工作。
- 分离 host 文件数据、staging 和最终 backing。共享内存上也保留 producer/consumer 可见性与完成，
  不因为 Apple 同一物理内存就跳过 ready/同步。
- 复用 IO File、Mem mapping/pool、Sched TaskScope/Channel 和 F1/F2/native transfer；
  无私有线程池/事件循环。底层文件取消不适用的平台必须保持真实能力边界。
- 成熟格式核由消费层选 gguf/safetensors/fastsafetensors；HYLE 不解释模型、tensor 名称或下载 HF 资产。
  引擎提供已验证的数据位置，通用管线同样可流送渲染资产和数值矩阵。
- 按字节、槽、全部在途和旧 lease 计预算；参考库的“几个 shard”窗口不等于固定内存上限。
- 不承诺直接 NVMe→GPU/GDS、自动 IO fallback 或 CPU/GPU 全时重叠；各 native capability 单独表达。

## 6. 后续候选、负空间与顺序

### F4/F5 先在消费层验证

- F4 只考虑显式 tensor/resource 集合的版本、clone、成功发布及旧版本保活。
  SGLang flush 条件和 llama.cpp bounded recurrent snapshot 是反例：
  任意过去状态不可能由通用复制凭空重建。业务 metadata/RNG/位置由引擎纳入，
  完整模型回滚正确性必须由引擎验收；普通 out-of-place 算子状态无需新增事务 runtime。
- F5 先接原成熟 paged/ragged region 的真实物理端口、indptr/索引/last-page 与 host-known metadata，
  说明谁更新哪些数据、什么时候 plan、哪些成本每步发生。不得破坏原融合或把每步必要
  metadata 处理伪称一次冷编译解决；不复制 S7 对 FlashInfer 私有函数的优化补丁作为稳定 API。
- 四路线的自然消费能力与 native 限制须明确；CUDA 的 torch/FlashInfer/Python owner 不可
  原样复制成 CPU/Apple/Vulkan 的公共 ABI。没有受审承载时如实 pending，先查成熟 Provider。
- warm Invocation 借还可作为消费层小工具：预先建立有限 slot、预算、Semaphore admission，
  在最后结果/完成后 reset，失败 slot 退休；只有多个实际消费者重复这套状态机才考虑 HYLE helper。
  ObjectPool create/destroy 每次构造/析构 T，不是保留已编译 workspace/capture 的 pool。
- 普通 CPU 抽样可用现有 Rng+标准分布；GPU 随机/抽样优先借成熟实现。
  完整 sampling chain 不因可省引擎代码而进入 HYLE。

### 建议后续施工顺序

1. 若用户接受方向，**同一有限批次**定界 F1/F2 的共享 owner、原生完成与范围语义，集中接线后集中验收。
2. 用真实文件→GPU 和渲染/矩阵两个消费者兑现 F3，检查预算/取消/关闭，不先抽象一个大通用 cache。
3. 引擎最小 consumer 接入已有计划复用，并驱动 F4/F5 定向设计；只补实际缺口。
4. 通用增量图仍是独立待办；不把本轮五个候选、新模型调查和算子全目录合成无边界大 goal。

预估收益只按重复代码类型描述：F1 消除保活/回收竞态状态机，F2 消除逐范围编排与反复准备，
F3 消除加载/预取管线生命周期。没有实际引擎和测量，**不虚报节省代码行数、工期或吞吐提升**。

## 7. 调研终局与验证

四包 S0–S3 完成。核对三个参考仓库的 42 个文件关键片段/调用线索，以及 HYLE 的 44 个公开头/
实现文件，处于 36 个主文件+最多 12 个定向补读的预算内。调研及已有实现适配差异核对约 15 分钟，
没有启动编译、推理或性能比较；源码机制与建议边界已足够裁决，不追加全库扫描。

本轮没有产品代码、模块状态、依赖锁或 Provider 实现改动；
不把更新参考源码或调研结论计作基础能力完成。

- 验证：change-impact selector 只选择 docs/closure，138 项跳过且保留原因；文档和模块合同通过，
  git diff --check 通过。结果见[验证收据](../../../../evidence/architecture/inference-foundation-gap-research/validation-result.json)。
- 平台/正确性/性能：只有静态源码调研，没有实际引擎运行与 P1；不生成伪 PASS、性能赢家或新平台资格。
- 统一 LOC 脚本 self-check：[本轮 LOC](../../../../evidence/architecture/inference-foundation-gap-research/loc-current.json)
  的 Compute+Graph 非测试实现为 **68,204 行 / 301 文件**；全库非测试实现 **163,431 行 / 652 文件**。
  与前一产品候选相同，未扩产品代码；不把第三方源码、文档或研究记录计入自有实现。
- 后续：等待用户讨论/授权有限实施范围；本调研 goal 完成，不自动启动 F1–F5 或算子填充。
