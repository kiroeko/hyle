# Compute/Graph：第三方广度与 HYLE 全局路由统一审计

> 本页记录 2026-09-14 的架构需求协商、源码审计、foundation-v1 架构基线和当前总目标的工作顺序。它不是
> 单个 component 的 closure card，也不自行改变 `module-status.json`；产品修改前仍须让受影响 component
> 依法返回 `AUDITED` 并分别冻结有限改动集。

当前入口是[四路线研究裁决](four-route-architecture-decision.md)：有限研究完成，停下交用户审视，
不自动启动产品改造。下面保留范围与前轮分析。原总目标、
实现队列和 55%/89% 估算保留为历史施工背景，不授权继续正式实现，也不是本次研究进度。
架构文档不再复制第二套编号队列。

## 当前任务：四条独立路线的架构定案调研

`fac7c9c5` 后用户要求实际开始剩余调研；本轮已完成两个预声明实验和旧实现审计，结果见
[四路线研究裁决](four-route-architecture-decision.md)。MLX 同 runtime 的 Metal kernel 插入、Vulkan 原生
重复执行得到有限真机证据；旧 output-only buffer ABI 必须改为同时支持 Provider 生成结果的 owner。
这不签基础实现或性能通过。后续有限顺序也只见该裁决，不自动恢复旧 goal 或 Provider/算子扩充。
下述 30 分钟指前轮文档校准，90 分钟研究卡指本轮，二者不是互相覆盖的施工命令。

用户在 `e802e677` 后明确：当前任务是参考成熟推理引擎、按真实纵切进行必要调研/实验，把重大架构决策
定案；不是完成四条路线的正式产品实现。上一轮[准入研究](foundation-architecture-admission.md)只完成
局部机制验证，不等于整体架构已定案。旧大目标暂不执行，不创建新的并行状态或自动恢复实现。

2026-09-15 在 `6cb2dec3` 后，用户进一步裁决：只保留 CPU、CUDA、MLX/Metal、Vulkan 四条独立推理
计算路线；**模型级 CPU+GPU 联合计算不属于 HYLE 功能范围，不是延期 backlog**。此前列出的 CPU 层段
与 GPU 层段、CPU FFN/专家卸载纵切撤销；已有源码调查仍是历史事实，不再授权实现这些能力。
CPU-only 推理保留，GPU 路线所需的 host 提交、初始化/上传下载、同步与回收保留；同一 GPU 上多个
Provider 的组合和 MLX/Metal 边界也保留。不能用隐藏 CPU kernel fallback 把混合推理重新带入。

参考 ggml/llama.cpp、vLLM、SGLang、ds4 的具名源码与厂商 API。构建期选能力集合，准备期按实际设备/
格式/shape 绑定计划，执行期固定；用户接受四条路线各自实现，不以共享执行器或代码行数为目标。
目标仍是合理真实场景中的 SOTA 性能。已有实现只作审计输入，不能因已经存在或旧测试通过就默认保留；
调研后须按新架构决定复用、改写或删除，不维护无人消费的旧接口兼容层。
本轮只更新架构与任务边界、核对现有实现，不改产品/registry/依赖锁/治理工具，不启动构建或 P1。
复用已有有限实验；本次源码与文档校准预算 30 分钟，完成下述分析即报告，不自动恢复原大 goal。

固定要裁决的系统问题为：

1. logical operation/region、量化精度与物理格式怎样表达，成熟 leaf/opaque region 如何真正接入；
2. 全局/局部调度权、CPU executor、私有 runtime、shape specialization 与 capture 的责任边界；
3. execution/storage/lane 如何表达所选路线内的执行、共享/独立内存与输入输出传输，不把 host resource
   自动解释为 CPU 计算资格；
4. 常量、输出、scratch、KV 等外部可变资源的 lifetime/alias/并发/预算，哪些限制必须拒绝而不能假装兑现；
5. 部分接纳、异步错误、Completion、取消/回收能否跨 runtime 闭合；
6. 冷路由和渐进替换怎样计入真实完整成本，能否保留成熟实现的性能而不强制复制/同步/重建图；
7. 哪些共同语义值得保留，哪些 lowering/提交/资源/回收应独立实现；CMake、模板/宏、运行时冷绑定
   各负什么责任，以及未来 HIP 如何接入而不把当前 core 写成封闭的四分支枚举。

每项输出「真实消费/参考入口 → 可选方案及代价 → 必要实验与反例 → 采用/拒绝理由」。复用已有证据；
实验只针对会改变这些系统决策的未知量，不为了架构调研实现完整模型或完整交付矩阵。现有 CPU/CUDA
局部证据与本轮 MLX/Vulkan 独立实验分别记账；不得合并为 HYLE 四路线产品全部通过。

纵切以各路线自己的 FFN/资源流水线为基础。CPU 真计算、CUDA 压缩真计算、MLX GPU region 与原生
Metal 边界、Vulkan 真计算分别取证；GPU 上传下载不能冒充 GPU kernel，独立 MLX 程序也不能冒充
HYLE/MLX/Metal 全部交付。跨 Provider 组合是同一路线的局部替换，不再增加 CPU+GPU 模型联合计划。

**本阶段停止点是架构定案报告，而非 foundation implementation gate**：重大 owner/接口/资源/路由决策
有源码和必要实验支撑，不再留“等写完 Provider 才知道共同合同是否成立”的未知量；剩余工作可具体归属
为局部实现/配置/资格验证。若仍有会改变共同骨架的未决问题，报告精确问题，不能提前宣布定案。
定案后先报告并与用户确认，再恢复正式施工；不把全部算子、三交付和终局 P1 完成作为研究的结束条件。

### 本轮架构分析：统一合同与独立执行

采用「共同语义/冷编译 + 路线专属制品和运行状态」，拒绝以一个万能 GPU runtime 抹平差异；也不预先
复制四套 operation/schema/DAG/错误语义。长期决策见 [Compute §1.3/§3.5](../../../architecture/compute-design.md#13-四条独立路线与明确非目标)
和 [Graph §6.3](../../../architecture/graph-design.md#63-路线内编译与特化执行)。

研究与后续施工按四条路线分别组织，再做一次共同合同核对，不增加 CPU+GPU 组合矩阵。
每条路线只需确定算子/region 接入、原生资源与并发、准备与重复执行、同步/失败/回收，以及完整成本
验证方法。它们不必共用执行器；公共 DAG/精度/依赖语义不复制四份。后续局部厂商 API 差异先在所属
backend/provider 解决，只有具体反例推翻共同合同才重开共同架构，不因新增算子再次进行全架构调研。

本轮只读 pins：llama.cpp `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a`（含 vendored ggml）、
vLLM `51918e252af4dfff381001f42b7ebe6a0454501b`、SGLang `06992c92ad4494fe9f9636e75be3e35040512447`、
ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc`；MLX C++ 事实沿用 v0.32.2 与前轮真机证据。

| 源码/公开合同事实 | 采用的设计与不采用的推论 |
|---|---|
| ggml 的 CMake 分别选择 CUDA/Metal/Vulkan/HIP；HIP target 可编译部分 CUDA 源码并有 vendor 宏适配 | 构建期裁剪，target 彼此独立；宏可复用同义实现，但不能据此前缀替换就签 HIP，或声称所有执行选择已在编译期完成 |
| `ggml_backend_cuda_context` 按 device/stream 保存 cuBLAS handle/workspace；cuBLAS 明确 `SetStream` 会重置 workspace，共享可变 handle 有并发约束 | 冷准备 lane/run-slot 资源并复用；不每 op 建 handle，不强制所有 Provider 共用一个可变 handle |
| ggml Vulkan context 持有 command/descriptor pools、prealloc buffers、同步和回收状态；Vulkan command pool 有 host 外部同步要求 | Vulkan 独立录制/提交/回收；不模拟 CUDA capture，不把 pending command buffer 当可立即 reset 的普通对象 |
| MLX compile cache 是 thread-local，命中后仍实例化数组；`transforms.cpp` 按 primitive 的 device 选择 CPU/GPU eval | 持久提交 owner 与 managed GPU region；显式绑定/验证 GPU 计算，不把 MLX 同时具备 CPU/GPU 当作本库允许联合推理 |
| vLLM dispatcher 按 batch descriptor 选 FULL/PIECEWISE/NONE；SGLang 按 bucket/stream/variant 填持久输入并 replay | 部署固定但 shape/批次可以变化；有限已准备变体，不每 token 搜索设备或 Provider |
| ds4 decode islands 保存有限 key 与 CUDA graph exec，首见 warm、次见 capture、以后 replay | 采用稳定资源/分段 replay；HYLE 把 warm/capture 放显式准备，不照搬首 token 的隐式冷工作或把 peer 的错误处理注释视为 HYLE 失败保证 |

[ggml CUDA 资源](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/ggml/src/ggml-cuda/common.cuh#L1454)、
[HIP 构建](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/ggml/src/ggml-hip/CMakeLists.txt)、
[ggml Vulkan](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2506)、
[cuBLAS stream/workspace](https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace)、
[Vulkan command lifecycle](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html)、
[MLX primitive 执行](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/transforms.cpp)、
[vLLM dispatcher](https://github.com/vllm-project/vllm/blob/51918e252af4dfff381001f42b7ebe6a0454501b/vllm/v1/cudagraph_dispatcher.py)、
[SGLang replay](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)、
[ds4 islands](https://github.com/antirez/ds4/blob/9139e2ae58a41503968a500f36f75895c1ba63fc/ds4_cuda.cu#L899)。

### 现有 HYLE 的待审改造点与后续有限接入

下表明确后续重构的审计对象，不是本次源码阅读已经证明旧实现应当保留。语义需求与具体代码分开裁决。

| 现有实现 | 按新架构审计与改造 |
|---|---|
| `BackendVTable`、typed extensions、独立 CUDA target/SDK discovery | 核对 native owner、分发层数、target usage 与 SDK 隔离；独立原生 factor 的方向可复用，具体接口/布局不预先批准；共同层不扩成全局后端 switch |
| Operation/Graph、Executable/Invocation、资源三轴、shared/delegated、Completion | 核对共同语义是否必要、资源寿命是否准确、状态是否重复；可复用成立的语义，不保证当前类型/字段/分配方式原样保留；共享制品与可变 run-slot 分开 |
| `RouteSearch/ResourceSearch` | 按所选路线收紧候选与搜索；唯一可行路线直接准备；审查不再服务当前范围的 CPU+GPU 联合分支，而不是只在旧搜索外加一层过滤 |
| `Invocation::submit` 的共同 segment 循环 | 对照原生提交路径，查逐节点 wrapper、重复验证/分配/同步与事件成本；需要时改成路线专属执行物/入口；不把通用循环视为前提，也不从源码分支数量直接宣称性能 FAIL |
| `CommandCaptureVTable` | 核对是否给某后端增加模拟或额外生命周期成本；只保留适用的可选能力，必要时改写为独立原生程序接口，不要求 Vulkan/MLX 伪造 CUDA stream capture |
| 旧 CPU/GPU 模型组合计划和百分比 | 退出当前要求；不删历史证据、不伪造新通过。`ReplayRoute::kHybrid` 表示 capture+普通命令，不等于 CPU+GPU 联合推理，不能按单词机械删除 |

正式重构按每条路线交付「成熟参考调用链 → 新方案与成本理由 → 旧实现复用/改写/删除清单 → 验证」。
先用 P0/源码对照处理明显的重复规划、临时资源、强制复制/等待、融合破坏和错误线程归属，再对稳定
candidate 做有限实测：同一 Provider 的 direct 对照查抽象税；同语义/精度/shape/资源与完成边界的成熟
完整路径查端到端竞争力。冷准备、host submit 和最终完成成本分别报告，不能拿 kernel-only 胜利代替。
新的公共形状与执行器必须服务消费者和性能；「四套实现」本身不是 SOTA 证明，旧 PASS 也不是新架构的
性能资格。当前阶段只定案并报告，正式改造范围绑定后再实施，不在本轮顺手改产品或启动测量矩阵。

本节原列的未知量已由[本轮裁决](four-route-architecture-decision.md)逐项处理；不要把这段旧“剩余实验”
重新执行。CUDA typed scales/精度与源码 façade、MLX result owner/checked completion、Vulkan 原生程序
已能划定具体实现责任；正式 adapter、错误注入、交付和完整性能仍未资格化。MLX 可通过公开扩展在自己
runtime 中保留完整 region 并插入 Metal kernel，不强制为了替换而拆图；无支持的边界仍需实证费用。
模型 CPU+GPU 联合路线不再研究，未完成的正式工作不增加研究轮实现百分比。

本轮文档校准验证（2026-09-15）：以 `6cb2dec3` 为 baseline，validation selector 只选择 docs/closure，
`check-docs.py`、`check-module-closure.py` 与 `git diff --check` 通过。未改产品、CMake、依赖锁或 registry，
未运行构建、原生实验、交付测试或 P1。本次到文档与架构分析报告为止，不据此恢复正式实现或签发基础通过。

### 本轮 llama.cpp 混合计算源码核对

本节保留 `6cb2dec3` 的历史调研事实；用户之后已将该产品能力排除，以下不是当前待办或验收门。

只读同日 clean pin `4c9233c034fc450dcf34c7c0988aebe6da5cdf1a`，包括其 vendored ggml；未升级产品锁。

- `llama-model.cpp::load_tensors` 按 `n_gpu_layers` 与 device split 决定 layer buffer list，通常 CPU 前段、
  GPU 后段；该版本 output layer 也纳入 offload 计数，input layer 首选 CPU。权重在加载期分配到相应
  buffer，不是正常每 token 把整份层权重来回搬运。tensor overrides 和 backend 支持度仍可改变实际放置。
- `ggml_backend_sched_backend_id_from_cur` 优先尊重已有 buffer/view 及 weight locality，但在开启
  `op_offload` 且 backend 支持并愿意接纳时，可让 host 权重的算子在 GPU 执行。CUDA 的 offload 建议
  还检查 batch size，所以权重驻留不等于计算强制位置。
- `ggml_backend_sched_split_graph/compute_splits` 切出同 backend 子图，处理跨片段 tensor copy、event
  和同步，再调用 `ggml_backend_graph_compute_async`；没有异步 copy/事件能力时有同步路线。不能把
  async API 名称当作必然重叠，顺序 decode 的依赖链也不会因 CPU/GPU 都参加就自动并行。
- `llama_context::graph_get_cb` 显式约束部分 norm/层末 op 的 backend，源码理由就是避免归到邻层后
  增加传输。它是完整边界费用必须进入 HYLE 路由的真实例子，不能只做单 op 速度选择。
- 普通 `llama_kv_cache` 在 offload 开启时按 `model.dev_layer(il)` 选择 KV buffer，否则选 CPU；
  CLI 另有 KV offload、tensor override、CPU MoE/CPU dense FFN 选项。多 GPU `split-mode` 的 layer/
  row/tensor 是另一个维度，不能与 CPU/GPU 层卸载混为一个开关。

[层与权重放置](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/src/llama-model.cpp#L1496)、
[ggml 切图/执行](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/ggml/src/ggml-backend.cpp#L921)、
[层边界修正](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/src/llama-context.cpp#L2522)、
[KV 放置](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/src/llama-kv-cache.cpp#L214)、
[用户选项](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/common/arg.cpp#L2745)。

这些是参考实现事实，不是 HYLE 混合计算已经通过的证据。本轮仅 docs/closure/diff 验证，不构建、不采样。

## 两条需求与统一结论

1. HYLE 不可能经济地在所有平台独立实现所有算子，需要通过成熟 Provider 获得算子、融合和设备广度；
2. HYLE 又必须拥有 Graph、内存/放置、调度和最终路由，才能逐步用已证明更好的 HYLE 实现替换局部路线，
   并对端到端性能负责。

统一结论是**层级调度**：HYLE 唯一拥有逻辑 DAG、partition、target/storage/lane、跨 region copy/sync、
liveness/workspace 和最终 Completion；入选 Provider 只拥有一个 region 内的 artifact、私有 layout、局部
workspace 与内部 kernel 顺序。外包粒度可以是单 op、融合 region、block 或完整 engine，但每个粒度都是
HYLE 冷路径的候选。HYLE 不需要调度第三方 artifact 内每个 kernel，也不能在不透明 artifact 内偷偷替换
私有 leaf；渐进替换发生在明确的 HYLE boundary。

默认裁决不是“HYLE 优先”或“第三方优先”，而是：完整计划上存在合格、material 的 HYLE 胜利才替换；
没有可比证据、测不准或差异不足时使用已配置的成熟路线。HYLE steady replay 只运行不可变计划，不支付
搜索、计时、JIT、建图或常量 packing；Provider 内部执行成本按本页整体复核接受的 prepared/managed
合同分别披露，不能把 compiled runtime 自动写成零分配。

## Foundation v1 架构基线

下表是此前形成的工作基线，不把尚未完成的实现写成通过；当前按上面的四条独立路线研究验证。
已明确的产品目标保留，若关键实验反证某项系统合同，必须在定案前修正，不能仅因写过 ACCEPTED 就拒绝研究。

| 维度 | 工作基线 |
|---|---|
| 公开产品 | 只有 Compute 与 Graph；`graph_plan` 是 Graph 私有实现职责，不新增第三个 Plan 产品 |
| 生命周期 | Compute 固定为 Device/Context/Queue/Completion、Operation/PreparedOperation/per-run Invocation；Graph 固定为 Builder/immutable Graph/shareable Executable/per-run Invocation |
| IR 与 shape | 有限 typed DAG；独立 bounded symbol + 有限派生式/guards；Executable 声明 exact 或 bounded/guarded domain，Invocation 冷绑定 concrete shape；schema/provider 负责完整 subdomain，否则 exact-only；无界数据依赖输出和模型循环不进 v1 |
| 全局/局部 owner | HYLE 唯一拥有 region boundary、target/storage/lane、跨 region copy/sync、liveness/workspace 和最终 Completion；Provider 只拥有入选 region 内 artifact、私有 layout/workspace/顺序 |
| 资源模型 | execution target、exact storage resource/domain、execution lane 三轴分离；shared Queue 与 delegated island 都是一等路线，不伪造统一 stream/allocator |
| Provider 接入 | vendor factor 经版本化内部 SPI；消费者通过窄的公开 owning custom-kernel factory，不包含私有头。schema identity 与执行旅程分别验收；CPU/CUDA/Metal/Vulkan carrier 按 runtime owner 拆 factor |
| 路由空间 | 每个 node 的 leaf 与可收缩 node-set region 进入同一有界 `RouteCandidate` 空间；连续区间只是 fast path；完整 boundary 成本决定计划，覆盖数/dispatch/ordinal 只作最后 tie-break |
| workload | compile 显式区分 steady latency/throughput、预期并发与可选 replay 摊销；默认长寿命、单 Invocation、steady latency；memory/host-wait/compile 分别受硬约束，不混成魔法分数 |
| 性能证据 | 只消费同 HYLE boundary、完整 identity 的 qualified route profile；Provider 私有相对 ns 不可跨 Provider 排序；无证据或不可分时按成熟 Provider 顺序稳定回退 |
| tuning | 普通 `compile()` 不用 runtime input/output 执行 graph；Provider 可在私有隔离 scratch 内做有预算的局部 tactic/JIT 选择，immutable constant 只按显式 policy 读取。跨 Provider `graph_tune` 未来由调用方显式提供 bindings；两者都不是隐藏热路径 |
| identity/cache | OperationId 是 schema+version，不是 vtable 地址；constant 显式 borrow/materialize，兼容 variants 共享不可变常量，不能共享并发可变状态；全部 resident 与 cold peak 计费；跨 Executable cache 由消费方拥有 |
| capture | capture 与 compilation 正交；exact capture key 可与 bounded/general route 共存，业务 batch mode 由引擎选择，steady submit 不临时 capture/JIT |
| 分布式边界 | rank/topology/communicator lifecycle、全局 batching 与恢复归引擎；Graph 只调度带 external communicator 的本地 collective participation 与 Completion |
| 副作用 | 资源/副作用域 + read/write 合同；同域冲突有序，独立域可并行，未知副作用保守；KV/batch 等策略仍在引擎 |
| failure | selected-route prepare/compile 事务化；`Unsupported` 和明确 candidate-local、完整回滚的资源拒绝可在硬预算内重选；work 一经接纳只经 Completion 报错，不静默跨 Provider 重跑 |
| 热路径 | HYLE 不搜索/推 shape/建图/分配；prepared Provider 同样严格，managed 如实披露内部执行准备；两者均禁止热 JIT、隐藏 copy/sync 和接纳后 rerun，支持 prepared-only 冷约束 |
| 搜索预算 | 硬预算始终有效；无已验证完整可行解则失败，有可行解默认返回并报告 optimization_limited；严格调用方可拒绝受限结果，不能掩盖真实错误 |
| 交付 | Provider SPI 是 HYLE factor 间 C++ 合同，不是稳定动态 plugin/Native FFI；source/static/shared 语义一致，可选 SDK/provider 独立发现 |

本次复核保留大对象模型、依赖方向、资源模型和调度权；只有具体反证才修改这些边界。允许在实现卡中
局部选择的数据结构、beam/DP 细节、厂商 API 调用和 profile 存储表示，不得改变上表语义。架构基线要靠 CPU shared region、CUDA async+
Q4 route、Apple delegated region/Metal interop、Vulkan 资源同步的真实实现探针签发 foundation implementation candidate；
在该 gate 前不横向填算子或平台大全。

### 必须由纵切决定、不能在纸面伪造的部分

下表是剩余实验变量，不是未定义的产品边界。探针失败只先修正对应 seam；只有结果证明上表的 owner 或
lifetime 根本不可实现，才构成整体架构 reopen fact。

| 实验变量 | 已固定的不变量 | 签发探针 |
|---|---|---|
| node-set 的物理表示与选择算法 | Provider 语义是可收缩 node set；搜索有界、确定性，不宣称一般 DAG 全局最优 | 单节点、连续和非 insertion-order oneDNN region；重叠/离开后重入/预算耗尽反例；再裁决 interval fast path、small vector/bitset 与 DP/beam |
| `ShapeDomain` 的 C++ 表示与 guard dispatch | exact/bounded 与有限派生维度都能表达；schema/provider 负责完整 subdomain；HYLE submit 不求 shape/JIT | 动态 token general + exact variant；flatten/concat/head split、越界/非整除/overflow/workspace 反例 |
| constant materialize 的 owner 数据结构 | caller 明确 borrow/materialize；兼容 variants 共享、重复副本和全部内存可查，不静默改变 lifetime | mmap/device materialize、两 variant 共享、并发/capture 不安全复用拒绝、compile rollback 与 Completion lifetime |
| global selector 的实现和默认预算 | leaf/region/placement/boundary 同池；结构硬约束优先，只有 qualified profile 可跨 Provider 比较 | gated MLP 局部赢整体输、prepare decline/reselect、profile 缺失/不可分，以及有/无已验证可行解的预算耗尽 |
| CUDA capture 与 Q4 路由的具体 carrier | capture 与 compile 正交；exact key；接纳后不跨 Provider 重跑；HYLE 掌握 residence/lane/Completion | CUDA Q4 MLP 在 HYLE/成熟/unsupported、HYLE/Provider-owned storage、general/exact capture、多 lane 与 device error 上回放 |
| Apple delegated/shared 边界 | MLX island 与 native Metal 共用 HYLE boundary；ownership 与 prepared/managed 成本独立，不假称统一 allocator/stream | 两台 Apple 验证真实 copy/sync、Completion、元数据/runtime 成本、workspace、并发和 prepared-only 拒绝 |
| Vulkan 执行与内存可见性 | ordered Queue/Completion 不能省略 barrier、可见性或 queue-family ownership；typed backend 持有 native 细节 | 最小 SPIR-V copy/compute/copy、显式依赖与 native handoff；Windows/Linux 和 MoltenVK 分别取证 |
| 公共抽象与三交付成本 | direct 不取得 Graph；HYLE 热计划无搜索/分配；安装消费者可绑定 custom execution | CPU one-op 税、跨 shared custom identity、仅安装头/库的 CPU/CUDA custom Graph journey |
| 资源副作用域 | 同资源冲突有序，独立资源可并行，未知副作用保守；不新增模型策略 | 两独立资源/同域写冲突/绑定 alias/跨 region 顺序，复用现有 Queue/Completion |

既有产品边界、全局控制和渐进替换目标不变；数据结构和具体实现不必全部在研究阶段决定。但 owner、
资源、生命周期、shape/capture 或低成本路由若还没有真实可行依据，仍是本次必须解决的架构问题，
不能预先归类为“只剩实现细节”。

## 真实引擎与成熟 runtime 反证

本轮读取的是本地源码快照，不用项目名或二手总结代替实现事实：ggml `e91ded11bdcd`、llama.cpp
`e71b80510c84`、vLLM `51da0ca66c80`、SGLang `dcebe8c4733a`、ds4 `f62ca29a3087`。成熟 runtime 另以
ONNX Runtime、oneDNN、TensorRT、PJRT/IREE 与 MLX 的官方设计/API 交叉检查。结论只进入 HYLE 的 owner、
lifetime 和 seam，不复制 peer 的模型 API。

| 来源 | 可复核事实 | 对 HYLE 的约束 |
|---|---|---|
| ggml `src/ggml-backend.cpp` | 先按既有 residence、支持度、输入局部性与 backend 顺序放置 node，再形成连续 backend execution splits，插入显式 copy/event，并独立 reserve graph allocation | 连续 segment 是优秀执行 fast path，但不能反推 Provider capability 只能用 begin/count；HYLE 仍须拥有跨 segment residence/copy/sync |
| llama.cpp `src/llama-context.cpp`、`src/llama-graph.*` | `sched_reserve()` 用 prefill/token-generation worst cases 预留；graph 只在 `llm_graph_params` 可复用时重用，否则重建；用户配置拥有 offload/split 与 batch 语义 | Graph 要能预报/预留 bounded workspace，shape/拓扑 identity 要显式；模型 offload、batch 与 reuse policy 留给引擎 |
| vLLM `docs/design/torch_compile.md`、`cuda_graphs.md` | 同时保留 general symbolic compile 与 `compile_sizes` exact tuned kernels；CUDA Graph 的 FULL/PIECEWISE/NONE 与 compilation 解耦，并由 runtime mode + `BatchDescriptor` 选已捕获路线 | Executable 不能只支持 exact；bounded/general 与 exact variants 要共存；capture 是正交 replay 能力，业务 batch class 不下沉到 Graph core |
| SGLang `tc_piecewise_cuda_graph_backend.py` | attention 周围切 piecewise graph；启动期对 capture shape 做 dummy/JIT warmup，共享 graph pool，再捕获/重放 | compile/capture/warmup 必须是显式冷路径，有独立内存 owner；steady replay 不能临时 JIT 或分配 |
| ds4 `ds4_cuda.cu` decode-island path | 只捕获 position-independent islands；首见 key 先 eager warm，次见 capture，之后 replay；key 含 layer/island/variant 与 pointer identity，失败在未接纳 work 时回到 eager | exact capture key、分段 capture、pre-acceptance fallback 和 work-accepted 边界必须进入合同；不能要求整图都可 capture |
| ONNX Runtime / oneDNN Graph | EP/partition 用 node index/ID 集合描述可支持 subgraph/partition，而不是拓扑区间；compiled artifact 与每次执行 scratch 分离 | Region SPI 使用可收缩 node set + boundary ports；compiled artifact 与 Invocation/workspace 生命周期继续分离 |
| TensorRT / StableHLO/PJRT | optimization profile 用 min/opt/max shape；engine 与 execution context 分离；builder 可 timing tactics，StableHLO 区分 bounded/unbounded dynamism | shape domain 同时容纳 exact 与 bounded/guarded；允许 Provider 私有隔离调优，但 Graph 不因此执行用户 graph |

主要官方入口：[ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)、
[ONNX Runtime high-level design](https://onnxruntime.ai/docs/reference/high-level-design.html)、
[oneDNN graph partition](https://uxlfoundation.github.io/oneDNN/v3.13/class_dnnl_graph_partition.html)、
[TensorRT dynamic shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-dynamic-shapes.html)、
[TensorRT optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)、
[StableHLO dynamism](https://openxla.org/stablehlo/dynamism)、
[PJRT C++ API](https://openxla.org/xla/pjrt/cpp_api_overview) 和
[MLX compile](https://ml-explore.github.io/mlx/build/html/usage/compile.html)。

## 原始基线源码勘验

源码 baseline：`49d32deacf80fbc046e53be64de47042144edc5a`（架构文档提交未改变前一轮勘验的产品树）。

| 需要的机制 | 当前事实 | 裁决 |
|---|---|---|
| provider-neutral tensor/op | `TensorDesc` 已分离 logical dtype 与 blocked/opaque physical format；`Operation` 是 typed schema | `KEEP` |
| 冷 prepare 与热 artifact | Provider 先 `propose`，再对 exact resources `prepare`；`PreparedOperation`/Invocation 分离 | `KEEP` |
| shared/delegated runtime | 单 op 与 region 均声明 Queue ownership、workspace、threading、Completion endpoint | `KEEP` |
| 第三方 compiled region | versioned `RegionProviderVTable` 已能 propose/compile region，Executable 持有 artifact；但身份仍是连续 begin/count，不能表达成熟 peer 的 node-set partition | 生命周期 `KEEP`，proposal 表示 `FIX` |
| 全局 target/storage/lane/liveness | Graph 已显式规划三轴、copy、happens-before、workspace 与 report | `KEEP` |
| shape domain | Graph IR 有 `DimensionBounds`，但 `CompileOptions::shape` 要求每个 symbol exact-once，Executable/Invocation 只能固定一个 point | `FIX`：增加 closed exact 或 bounded/guarded domain；不在 submit JIT |
| capture 正交性 | 已有 `ReplayMode::kCommands/kPreferCapture/kRequireCapture` 与 hybrid route 骨架，但它只建立在 exact specialization 上，尚未证明 bounded/general route 与 exact capture variant 共存 | 骨架 `KEEP`，shape/capture 组合 `FIX` |
| constant lifetime | 当前 `ConstantBinding` 一律保守借到 Executable 与全部 Completion 结束，即使 Provider 已报告 packed copy；aggregate count 不能证明哪些 source 可释放 | `FIX`：显式 borrow/materialize，逐 constant 报 owner/release point |
| 单 op 与 region 同池竞争 | target placement 先因 `region_support` 布尔值偏向能提 region 的 target，operation fallback 再按 Context provider 顺序准备，region 随后独立做 interval selection | `FIX` |
| 完整成本选择 | selector 只使用 provider 自报 savings、覆盖数、dispatch 和 provider rank；已有 copy/sync/workspace/persistent/host-thread facts 主要只进 report，选中后得到的 exact artifact facts 不触发重选 | `FIX` |
| 可比成本来源 | `estimated_net_savings_ns` 相对各 Provider 自己的 fallback，不能跨 Provider 相加；没有资格化 profile identity | `FIX` |
| 单节点 graph-hosting Provider | region discovery 拒绝 `nodes.count < 2`；只能另走 delegated operation seam，不能统一比较 | `FIX` |
| route evidence seam | 没有显式 workload intent 或 qualified route profile；当前 compile 也不能安全取得代表性 payload 来调优 | `FIX`；compile 保持无执行副作用，未来 tuner 独立 |
| route 诊断 | CompileReport 描述胜出 step/fallback，但没有有界 shortlist、拒绝原因和所用 evidence identity | `FIX` |
| typed custom op 三交付身份 | `OperationId` 被保存但 `custom_as<T>()` 比较模板静态 vtable 地址；default-hidden shared factor 间地址不构成合同 | `FIX` |
| ggml backend 身份 | 当前无后缀 identity 实际只记录 CPU blocker；不能代表 CUDA/Metal/Vulkan | `FIX`（新 factor 时） |
| CUDA Q4 可执行能力 | CUDA backend runtime 已存在，但没有接受 Q4 `StorageFormatId` 的 CUDA operation/region Provider | `MISSING`，不是 runtime 缺陷 |

这说明现有 Compute/Graph 的大生命周期、storage、异步执行和 plan 骨架没有被反证；被反证的是**冷路径
route selection 层**、region/shape 表示上限和 custom-operation 的 shared 交付身份。继续填算子会把错误的
选择机制固化，整套推倒重写同样没有依据。

## 有限解决方案

### R1：修正 typed custom operation identity

`compute_ops` 以 `OperationId`（其中包含 schema/version identity）和 descriptor size 校验类型，不再以
inline vtable 地址判断。traits/infer 仍是冷路径函数指针，descriptor 仍保持有界、trivially-copyable、
heap-free。增加 source/static/shared 跨边界构造与 Provider 解码测试；不建立全局 registry 或 Native FFI。

### R2：把单 op 与 region 归一为冷路径 `RouteCandidate`

Graph 内部候选至少包含 canonical node set/fingerprint、显式 boundary ports、Provider/version、target、
resource/layout requirements、runtime mode、
workspace/persistent/packing、boundary copy/repack、sync/dispatch、host blocking/private threads、precision/
alias/failure/completion 与证据 identity。集合须无路径离开后重入，Graph 验证收缩后仍为 DAG；
普通 Provider proposal 产生单节点候选，region extension 产生一个或多个节点的显式 proposal。连续区间可
走 fast path，Graph 不枚举任意子集。只有选中后才构造重型 artifact，避免为所有候选 JIT/packing。

冷编译按有界闭环执行：先枚举 candidate skeleton 与资源要求；再为有限的 target/storage 组合计算 Graph
拥有的 boundary copy/repack/sync；随后选择候选计划并只 prepare 胜出 artifact。Provider proposal 对选择
所需结构成本给出 exact、保守上界或 unknown，artifact 必须验证该承诺。selected-route prepare 返回
结构化 success、无副作用 candidate decline 或真正 error；只有 `Unsupported` 或明确 candidate-local、
已完整回滚的预算 decline 才把候选排除并在预声明 retry budget 内重新选择。事实超过已接纳上界是合同
违约而不是普通 fallback；不能把不一致成本留到 report 后仍执行原计划，也不能预先 JIT 所有候选来换取
一个“完美”评分。

Direct Compute 继续使用显式 Context provider 顺序与 `Unimplemented` fallback，因为它没有全图边界信息；
Provider 应只在自己愿意按该顺序拥有的精确域内提案。Graph 则必须把这些 leaf proposal 与 region proposal
放入同一有界计划搜索。无需为此增加第三套公开 runtime 或 `Plan` 产品。

### R3：结构成本、资格化 profile 与显式 tuner 分层

1. 先按语义、精度、storage/import、host-wait、并发和内存预算排除不可行路线；
2. provider 只报告可审计结构事实，不提供可跨 Provider 比较的任意相对 ns；
3. 可比较执行估值来自绑定 provider/build、device/driver、dtype/format/layout、shape domain 与完成边界的
   evidence profile；差异不可分或不足 material margin 时选择已配置的成熟路线；
4. 普通 compile 不以 runtime input/output 执行 graph；Provider 可在私有隔离 scratch 中作有预算的局部
   tactic/JIT search，immutable constant 只按显式 policy 读取；但须报告开销、cache identity 和确定性政策，
   不能借此替 Graph 比较全局候选；
5. 未来独立 `graph_tune` 只有取得调用方代表性 bindings、允许执行的纯/确定性范围和独立预算后，才测量
   结构筛选后的少量**完整候选计划**并产出 profile；它不阻塞 foundation v1；
6. 一般 DAG 不宣称求全局最优。compile 有确定的 candidate/state/prepare-retry 预算；没有已验证完整
   可行计划时耗尽失败，已有可行计划则显式报告 optimization_limited 并按 policy 返回或拒绝，不能
   静默截断或在 warm path 继续试探；未来 tuner 另有独立预算。CompileReport 只保留
   有界 shortlist 的身份、拒绝原因、成本来源和最终裁决，不复制无限 proposal 日志。

### R4：按 runtime owner 拆第三方 carrier

CPU、CUDA、Metal、Vulkan 的 ggml adapter 使用独立 component、feature、依赖和验证。CPU 的 abort/threadpool
blocker 继续只约束 CPU；CUDA route 必须另证 buffer import/ownership、stream/event、device error、capture
和三交付。调用私有 ggml kernel 不是普通 adapter；必要时用独立、版本锁定的 source-patch factor 承担维护
成本。

### R5：用 CUDA Q4 MLP 纵切验收，而不是用空 selector 测试

同一 Graph 至少产生 HYLE Q4 leaf、成熟 CUDA/ggml leaf 或 region、以及显式 unsupported 三类路径。测试要
覆盖权重位于 HYLE CUDA storage 与 Provider-owned storage 两种 residence，并证明：局部 HYLE kernel 胜出
但拆分成本更大时保留 region；完整成本 material 胜出时切到 HYLE；无合法 Provider 时返回
`Unimplemented`；CompileReport 能解释选择；warm replay 无 route search、建图、packing 和 metadata allocation。

### R6：shape domain、capture、constant 与 cache owner

一个 Executable 对应一个 closed shape domain，可以是 exact，也可以是 bounded/guarded。后者可包含真正
接受范围的 artifact，或 immutable 有限 exact variant 表与已编译 general fallback；所有 guards 必须有限、
互斥并完整覆盖声明 domain，schema/provider 无法证明 bounded subdomain 时明确 exact-only。Invocation 在用户工作
接纳前绑定 concrete shape，warm submit 不 JIT。CUDA Graph 等 capture 只绑定 exact shape/storage/Queue 和
Provider 声明的稳定 key；compile 与 capture 正交，Executable 只共享 immutable recipe，address/workspace/
run state 属于 Invocation，业务 batch class 的选择留给引擎。跨 Executable specialization cache 也留给
引擎。compile-time constant 显式选择 borrowed immutable 或 materialize：前者
source 活到 Executable/Completion 终态，后者要求 compile 成功前 artifact 已拥有 replay 所需 bytes，随后
source 可释放；report 按 constant 披露，不以 aggregate packed count 缩短 lifetime。可变权重建模为 input 或
重新 compile。Graph 不读写进程全局 route/plan cache，qualified profile 由调用方显式传入。OperationId
只标识 schema/version；custom instance 没有稳定 fingerprint 时不得进入持久 profile。

### R7：只有接纳前可以回退

candidate decline、prepare failure 或 submit 的明确 pre-acceptance failure 可以在有限、已冻结路线中回退。
一旦 Queue/Provider 已接纳用户工作，错误只经 Completion 传播；Graph 不在另一 Provider 静默重跑，避免
重复 side effect、资源写入或部分完成。

## 原始基线的定向重构范围

以下记录从 `49d32dea` 启动的定向迁移；状态迁移已经发生，不能把这段当作今天仍待执行的重复迁移。
整体复核新增范围见下一节，不由这里的旧有限范围自动授权：

- `compute_ops`：从 `CANDIDATE` 返回 `AUDITED`，只修复 custom descriptor 的跨 DSO identity。后续所有
  bounded-domain 支持均是 Graph domain 与具体 provider 对完整 subdomain 的承诺，不能把 provider-specific
  可执行能力误写成 `OperationTraits` 的静态布尔属性；
- `compute`：从 `CANDIDATE` 返回 `AUDITED`，因为现有 leaf proposal 缺少与 region 同池所需的结构事实和
  shape-subdomain guard、事务化 decline 分类；Device/Context/Queue/Completion 生命周期不重写；
- `graph`：从 `CANDIDATE` 返回 `AUDITED`，统一 leaf/region route candidate、workload/profile、成本来源和
  有界重选，并引入 exact + bounded/guarded shape domain；
- `compute_provider_native`：从 `CANDIDATE` 返回 `AUDITED`，只迁移新版 proposal seam 并参与探针；
- `graph_provider_onednn`：当前虽为 `AUDITED`，其活动卡尚未包含 node-set region；产品修改前必须绑定新
  baseline，把该迁移和非 insertion-order node-set 探针写入有限集合；
- `compute_provider_onednn`：Graph 的 node-set 表示本身不会重开它，但 Compute v8 会修改它直接实现的
  私有 proposal/outcome SPI，且前轮尚欠真实 region consumer，因此以 v5 新卡保持 `AUDITED` 并共同迁移；
- CPU/CUDA backend、Tensor storage、Graph 生命周期/liveness/workspace 骨架没有重写依据；
- backend-qualified ggml CUDA/Metal/Vulkan factor 在各自 closure card 成立后从 `QUEUED` 开始，不恢复旧
  adapter，也不让当前 CPU blocker 冒充它们的状态。

在上述状态迁移和有限卡片完成前，不继续横向填充算子，也不把现有 `RegionSelectionPolicy` 的最大覆盖
结果称为自动 SOTA 路由。

## 整体架构复核与合同修正

2026-09-14，用户要求先 review 整套架构，随后确认更新架构设计与决策，并再次要求参考 vLLM、SGLang、
llama.cpp、ds4。复核产品基线为 `5802aa02c828505f83e791091b7491f96f6d58e6`：custom schema identity、
事务化 prepare outcome、可收缩 node-set 已有实现；统一 selector、bounded runtime、MLX adapter 等尚未
完成，不能把这些已登记的迁移工作再次统计为新发现。工作区另有未完成、未验证的 C80 shape-domain
草稿，本次保留，不作为候选或通过证据。

结论是保留 Compute/Graph 两产品、三轴资源、全局/局部调度权和 Executable/Invocation 生命周期，修正
以下六处合同。它们均为设计/实现准入结论，不是性能胜负；特别是 A5 尚无重复分配的实测结论。

| ID | 反证或未闭合之处 | 已接受的修正与权威位置 | 实现 owner / 最小验收 |
|---|---|---|---|
| A1 | MLX v0.32.2 `compile.cpp::compile_replace` 在 cache hit 后仍创建执行数组/容器；与一律零分配重放冲突 | Compute §3.4：prepared/managed 成本与 shared/delegated ownership 正交；HYLE 热计划不放宽，可要求 prepared-only；managed 内部成本如实披露，仍无热 JIT/隐藏同步 | Compute + Apple provider；两台 Apple 实测 replay/Completion/OOM/成本，不能从 compile 名称推断资格 |
| A2 | `graph.h::Dimension` 只有 fixed/symbol，独立 bounds/equality 不能表达 flatten 的乘积或 concat 的和 | Graph §4.3：有限 metadata shape program + guards，独立维度由用户绑定，中间维度派生；不做通用求解器 | Graph + ComputeOps/domain seam；flatten/concat/head split、overflow/非整除与 bounded/exact 组合 |
| A3 | `Operation::custom` 只有 traits/infer；Provider 的构造依赖私有 `ProviderAccess`，仅安装 SDK 不能自然加入自己的 Graph kernel | Compute §6.1：对象局部、owning typed custom-kernel factory；公共 schema 与执行扩展分开，不公开整个内部 SPI 或恢复 FFI | Compute/ComputeOps + backend consumer；仅安装头/库的 CPU/CUDA custom Graph，三交付/销毁/并发 |
| A4 | `graph.cc::validate_effect_order` 以单一 last_effect 要求所有外部副作用串链 | Graph §4.4：资源/副作用域 read/write，冲突有序、独立可并行、未知保守；绑定 alias 也验证 | ComputeOps + Graph；两独立资源、同域冲突、别名绑定和跨 region 顺序 |
| A5 | 生命周期已分层，但未明确 variants 的 packed constant 共享、重复副本与聚合预算 | Graph §3/§4.2：兼容不可变常量共享，可变并发状态独立；逐唯一 allocation 计 resident/cold peak，unknown 不能证明硬预算 | Graph + provider artifact；两 variant 共享或如实重复、并发/capture 不安全复用拒绝、预算/OOM |
| A6 | 旧 Graph §6.1 即使已有可行计划也一律把搜索耗尽变成编译错误 | Graph §6.1：无已验证可行解仍失败，有则默认返回并报告 optimization_limited；严格模式可拒绝，不吞掉真实错误 | Graph selector；有/无完整可行解两种耗尽、determinism、partial proposal 拒绝 |

### 四个引擎的具体依据与不照搬的部分

重新确认前述四个本地 reference 的 HEAD 与已记录 commit 一致，引用文件没有受管修改；这不是声称
它们等于当日 upstream 最新提交。仅比较本轮相关机制，不更新依赖锁、不扩算子目录、不复制引擎策略。

| 具名来源 | 实际机制 | 对本次决策的作用与界限 |
|---|---|---|
| vLLM `model_executor/custom_op.py::CustomOp`、`compilation/piecewise_backend.py` | 初始化绑定 `_forward_method`；general 与 exact/range runnable 预先编译，按 runtime shape 选已有路线 | 支持 A2/A3 与冷绑定、general/exact 并存；不复制 Python 全局注册，也不把 Python dispatch 的成本当作 C++ 零成本证据 |
| SGLang `srt/utils/custom_op.py`、`model_executor/cuda_graph_buffer_registry.py` | 实现与 fake/output shape 分开，声明 `mutates_args`；capture buffer owner 明确，可共享稳定地址的 buffer | 支持 A3/A4/A5 的执行/shape 分离、显式写资源与复用；HYLE 不复制进程全局 pool，仍单独证明并发安全 |
| llama.cpp `src/llama-graph.cpp`、`llama-context.cpp::sched_reserve` | attention 合并乘积维度，图输入按真实 shape/state 判断可复用，预留 worst-case graph | 证明 A2 不是 synthetic 需求，也支持显式资源预算与复用条件；模型/KV/rebuild policy 留在引擎 |
| ds4 `ds4_cuda.cu::cuda_decode_graph_find/begin` | 按 layer/island/variant/pointer key 保留有限 capture；槽位用尽返回 eager 路线，首次先 warm 再 capture | 支持固定 key、显式冷阶段及优化容量与执行能力分开；只为 A6 提供工程原则，不冒充同一种图搜索算法；不复制全局状态或未经接纳证明的错误重跑 |

可复核源码：[vLLM custom](https://github.com/vllm-project/vllm/blob/51da0ca66c8065619c79e35dff97aa99aeaf5644/vllm/model_executor/custom_op.py#L129)、
[vLLM variants](https://github.com/vllm-project/vllm/blob/51da0ca66c8065619c79e35dff97aa99aeaf5644/vllm/compilation/piecewise_backend.py#L99)、
[SGLang custom](https://github.com/sgl-project/sglang/blob/dcebe8c4733a5cb802fb679e0c8fbb886e2801b0/python/sglang/srt/utils/custom_op.py#L57)、
[SGLang buffers](https://github.com/sgl-project/sglang/blob/dcebe8c4733a5cb802fb679e0c8fbb886e2801b0/python/sglang/srt/model_executor/cuda_graph_buffer_registry.py#L831)、
[llama.cpp reshape](https://github.com/ggml-org/llama.cpp/blob/e71b80510c848c00175924ecf3c40333ccae8eb5/src/llama-graph.cpp#L2658)、
[llama.cpp reserve](https://github.com/ggml-org/llama.cpp/blob/e71b80510c848c00175924ecf3c40333ccae8eb5/src/llama-context.cpp#L582)、
[ds4 capture](https://github.com/antirez/ds4/blob/f62ca29a308724cde5bc99134ede19104b2a3260/ds4_cuda.cu#L961)、
[MLX compiled call](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp#L985)。
vLLM `compilation/cuda_graph.py` 还明确注释多 stream 共享 global pool 的安全性待处理；这恰好说明不能从
peer 使用了共享池就推断 HYLE 的并发复用已经正确。

### 实施边界

本节记录架构文档提交 `401d2969`，不是后续产品施工的当前状态。本次仅更新 architecture、说明和任务
入口，registry 的状态与 candidate 不变，也不修改公共治理或构建工具。当时 Compute v8、Graph v9、
ComputeOps v2 的有限卡不能自动扩大为全部 A1--A6；后续在
现有 `AUDITED` 阶段绑定新卡和本次架构基线，保留已经完成的事实并固定新增 public API/有限 FIX，再写
这些新合同的产品代码。Native/oneDNN 只迁移实际受影响的 SPI；已 CLOSED CPU/CUDA backend 没有新缺陷
不重开。不给六个决策各造一个 factor，也不重新计时未变化的产品候选。

预声明本次只运行 docs/closure/diff 检查与文档提交的 change-impact selection；无 C++ build/P1，不能
由文档 PASS 推导产品 architecture/correctness/performance gate 已通过。

本次维护验证：`check-module-closure.py`、`check-docs.py` 与 `git diff --check` 均通过。未运行 C++ 测试、
sanitizer 或性能采样；产品代码与 registry 未由本次文档工作改变。

后续实际实施已绑定 [Compute v9](compute-v9-shape-replay-substrate.md) 与
[Graph v10](graph-v10-reviewed-foundation-contracts.md)；具体进展、未验证项和剩余 FIX 只在这些活动卡
记录，组件状态仍以 registry 为准，不把本节文档验证当成产品关闭。

## 当前 Compute + Graph goal 的重排

以下保留原大目标的实施队列与历史进展，目前不自动执行。当前任务仅为本文开头的架构定案调研；
定案并取得后续继续指示后再校准和恢复本表，不能从研究原型通过直接跳进正式 adapter 施工。

| 顺序 | 工作包 | 当前裁决 / 完成条件 |
|---:|---|---|
| 0 | Foundation v1 架构基线 | 本页和两份 architecture 固定上表全部系统决策；docs/closure contract 通过，不改 registry/product |
| 1 | 合法状态迁移与小卡 | 原迁移已使相关 owner 处于 AUDITED；整体复核后先给受影响 owner 绑定新卡/架构基线与 A1--A6 有限 FIX，旧已验证事实保留，不重开无关 CLOSED factor |
| 2 | Identity + proposal/shape/custom substrate | 复用已完成 identity/outcome/node-set；补 replay 成本、有限派生 shape/domain、安装包 custom execution、资源访问与 constant seam，不先写 selector 大算法 |
| 3 | Graph route unification | leaf/region 同池、完整成本与有界重选、搜索受限可行解、资源域顺序、variant 常量/内存聚合和可解释 report |
| 4 | CPU foundation probe | direct/one-op tax + gated MLP；oneDNN 非 insertion-order region；exact/bounded 与派生维度、资源冲突/独立、variant 内存/预算反例；增加真实持久 executor 组合，不能由串行 callback 探针代替 |
| 4a | 成熟实现接入准入 | CUDA 独立压缩 kernel、MLX C++ FFN/输入借入/线程缓存已有有限实证；正式 CUDA 版本/精度/失败封装与 MLX 全局调度 bridge 仍须准入。MLX 不签为 prepared/shared 或有限私有内存；不先造宽适配 |
| 5 | CUDA foundation probe | 复用已通过的多 lane/capture；分开做正常浮点 vendor 调用与保留压缩权重的 Q4 decode/prefill FFN；正式 leaf/region/unsupported、residence/预算、general/exact 与完成边界。Q4→F32 预检不代替量化路线 |
| 6 | Apple foundation probe | MLX delegated managed replay + 原生 Metal custom-kernel/interop，实证成本分类、prepared-only 拒绝、island boundary/Completion，不横向填 Apple 算子 |
| 6a | Vulkan foundation probe | 最小 SPIR-V 计算、device/queue-family、buffer/barrier/timeline、native handoff；Windows/Linux 与 Apple MoltenVK portability；不填算子目录 |
| 6b | 基础组合资格 | delegated island 的失败/完成/回收、实测 profile 回灌、最小可选 Obs、各 backend 交付；原 CPU/GPU 模型计算边界已由后续用户范围裁决撤销，不再恢复为待办 |
| 7 | **Foundation implementation gate** | 上述全部基础探针、适用平台 correctness、one-op/MLP P0/P1、实测完整 profile 回灌、source/static/shared、红队通过；报告实际压缩量化/第三方能力与限制，Compute/Graph 基础 candidate 固定后暂停讨论 |
| 8 | NVIDIA/Apple provider 广度 | cuBLASLt/cuDNN/HYLE 与 MLX/MPS/Metal 按真实热路径扩展；不再改基础对象模型 |
| 9 | Vulkan provider/算子广度 | 复用 6a 已验证基础，扩正式 SPIR-V/成熟 provider 与跨 API 能力；不冒充原生 Metal 性能，不把基础实现留到本项 |
| 10 | Transformer prefill/decode | 多 region、引擎拥有的 specialization/cache 与 batch-class 选择、并发 Invocation、capture、cancel/drain 和长寿命计划纵切 |
| 11 | 系统性算子广度 | foundation lock 后按四个真实引擎热路径扩展，不用算子数量掩盖架构缺陷 |
| 12 | `compute_sched` convenience | 只有真实 executor/协程消费者证明重复 glue 或机制缺口才开卡；第 4 项先用现有 Sched public API 验证，不要求为测试制造新产品 |
| 13 | 终局 closure/package | 各 factor 独立 CLOSED 或精确 BLOCKED；source/static/shared、依赖锁、许可证、user/architecture/docs 一致 |

原目标中“ggml/oneDNN region”不再视为一个并列动作：oneDNN 是 route 层的首个真实验证者；当前 CPU ggml
保持独立 blocker；CUDA/Metal/Vulkan ggml carrier 各自随对应平台纵切开卡。原先尚未执行的 foundation P1
后移到 route 实现稳定之后，并在第 7 项只运行一次，避免给即将改变的 selector 重复签证。第 7 项以前
曾按这张表估计“基础完成”；该历史分母现已停用，不能在新范围下继续套用，算子 family 数量也不计入。

2026-09-15 总计划覆盖审计补入 6a/6b，之前不含 Vulkan 等资格工作的约 89% 口径停止沿用。已完成的
实现和证据不清零；新估算必须包含本表全部基础范围，并区分实现与终局验证，不把文档修改算成功能完成。

### 2026-09-15 原始需求重新对齐与进度基线

基线 `3ea0eced4eadb8ff18bd7fd5e4ea33bd54e2a3e4`。本轮只校准架构、计划、说明与估算，不改产品、
registry、依赖锁、构建脚本或公共治理，不跑 C++/P1。原始需求不是“框架能描述所有 Provider”，而是
让 C++ 消费者实际借到成熟算子广度，保留 HYLE 全局资源/计划权，以完整消费收益渐进替换局部实现。

| 原始需求 | 保留的设计 | 本次实际调整 / 完成证据 |
|---|---|---|
| 借第三方广度，不维护全平台大全 | 中立 Operation + 显式 Provider，leaf/region 两种载体 | 接入可行性前置；下载/schema/独立 sample 不计为已交付 route；CPU ggml blocker 不冒充全 ggml 已拒绝或已支持 |
| HYLE 拥有全局性能责任 | 分层调度、三轴资源、不可变计划和独占 Invocation | 只在明确 boundary 替换；允许成熟 runtime 的真实局部 owner，不强行拆每个 kernel |
| 性能优先且不被测试扭曲 | 冷选路，prepared/managed 如实分类 | 正常浮点与压缩量化分开；activation 舍入与 accumulation 分开；实测完整计划回灌后才能证明选路，不靠 synthetic timing |
| 普通量化模型能保持容量优势 | 物理格式与逻辑 dtype 分离、常量 owner/共享/配额 | 压缩权重的 decode/prefill 为主纵切；约 7.1 倍 F32 展开只作显式备选与内存反例，不作默认量化实现 |
| Apple/NVIDIA 主力、Vulkan 保底 | 同一对象模型、typed native interop | 三条基础路线均纳入分母；不为 HIP 建 stub；SDK 由用户安装、HYLE 按所选 feature 发现 |

#### 替代旧会话目标的任务说明

在现有 Compute/Graph foundation-v1 上完成真实可交付的基础执行与图计划，不重写已获证据的
Device/Context/Queue/Completion、Graph/Executable/Invocation 和三轴资源。先对成熟 CUDA 量化
carrier 与 Apple MLX 做有限接入准入，确认公开 shared/delegated 或独立窄补丁的可行边界；再按本表
完成正式 CUDA 浮点/压缩 Q4 FFN、Apple MLX/Metal、Vulkan SPIR-V，及实测选路、异步失败/回收、最小
Obs 与各 backend 的三交付。借成熟能力并允许 HYLE 在相同数值/资源/完成合同上渐进替换；无合格
胜利则保持成熟路线。算子目录、完整模型、KV/batching/serving 和 HIP 均不进入这次基础施工。新
factor/底层缺口按现有小卡与状态协议处理，已匹配证据复用，只测实际影响；基础实现和适用正确性、
性能、交付、红队全部通过后报告并暂停共同审视，不自动进入总目标第 8 项。

以上是先前为原大 goal 准备的实施说明，已被本文开头的“当前任务”暂停，不得作为当前 session 的启动
提示词。没有标记旧目标完成或恢复它；模块状态仍只看 registry，不能把会话目标与组件状态混同。

#### 更贴近消费的有限纵切

选型依据是已更新的同日 clean reference：ggml `456172ec`、llama.cpp `4c9233c0`、vLLM `51918e25`、
SGLang `06992c92`、ds4 `9139e2ae`；完整 commit 与版本化调查见
[Q4 preflight](cuda-q4-foundation-carrier-preflight.md#版本化来源与裁决)。本轮只读这些具名路径，未升级产品依赖。

- llama.cpp `src/llama-graph.cpp::build_ffn` 实际组合 up/gate、SiLU 和 down projection；
  ggml `src/ggml-cuda/mmvq.cu::ggml_cuda_mul_mat_vec_q` 与 `mmq.cu::ggml_cuda_mul_mat_q` 消费压缩权重，
  同时为 activation 建立 Q8_1 临时表示；Q4_0/Q4_K 均有 switch route，不能假定只有先全量展开再 GEMM。
- ds4 `ds4_cuda.cu` 的 dense quantized 路径直接把 Q4_K 权重交给 `ds4_mmq_q4_K_dense`；与其 Q8 的
  浮点缓存/临时展开备选是不同路线，不能从某条 fallback 推断整个引擎都展开权重。
- vLLM `model_executor/kernels/linear/mixed_precision/marlin.py` 和 SGLang
  `layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` 在加载后 repack packed
  weights/scales，执行时使用这些权重与 workspace；GPTQ/AWQ 的格式不等于 GGUF Q4_0/Q4_K。

源码入口：[ggml MMVQ](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-cuda/mmvq.cu)、
[ggml MMQ](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-cuda/mmq.cu)、
[llama.cpp FFN](https://github.com/ggml-org/llama.cpp/blob/4c9233c034fc450dcf34c7c0988aebe6da5cdf1a/src/llama-graph.cpp)、
[ds4 CUDA](https://github.com/antirez/ds4/blob/9139e2ae58a41503968a500f36f75895c1ba63fc/ds4_cuda.cu)、
[vLLM Marlin](https://github.com/vllm-project/vllm/blob/51918e252af4dfff381001f42b7ebe6a0454501b/vllm/model_executor/kernels/linear/mixed_precision/marlin.py)、
[SGLang WNa16](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)。

| 纵切 | 正常旅程 / 反偏置 | 要裁决的架构问题 |
|---|---|---|
| CPU（保留） | 现有 RMS+gated MLP，Native/oneDNN 共用持久 Sched executor | 真实 region、资源预算、嵌套/并发与抽象税；已有 Windows 证据不重跑 |
| CUDA 浮点 | GPU 常驻的正常浮点 MatMul，随后组合 FFN；使用 vendor 真正支持的 dtype/累加合同 | 同一个 vendor 的 direct 与 HYLE 路线隔离 adapter/Graph 成本，不把量化转换成本混入 |
| CUDA 压缩量化（主纵切） | 一次加载/必要 packed repack，device-resident Q4 FFN；M=1 decode 与一个正常 prefill chunk；重复变化 activation、两独立 Invocation | mature compressed route、HYLE/custom leaf 与 region 的真实可用性；压缩常驻、variant 共享、capture、边界费用与完整选路 |
| Apple | MLX 的真实 FFN region 与一段 native Metal 计算衔接；使用其原生支持的 dtype/format | delegated managed、copy/sync、最终 Completion、源释放与 prepared-only 拒绝；不强迫它假装支持 GGUF Q4 |
| Vulkan（正交） | 上传资源 → 最小 SPIR-V 数值变换 → 后续 native consumer/读回；重复复用 buffer | barrier/可见性、timeline/owner 与 native handoff；Windows/Linux 与 MoltenVK 分开资格 |

量化首个正式 carrier 在 4a 明确选定一个有实际公开/受审接入路线的格式；现有 Q4_0 是 codec/边界
回归资产，Q4_K 或 group-wise W4 是依据真实 carrier 的候选，不要求一次实现所有格式，也不强行把
同为“四位”的表示互换。normal shape 的 hidden/intermediate 来自具名模型配置/真实算子序列，至少保留
一个 decode 和一个 prefill；tiny/tail 只做 correctness。不用全模型权重下载/长上下文跑分证明基础。

“直接量化计算”只要求没有整份展开常驻权重；kernel 内解码、activation quantization、tile scratch 或
保持压缩的冷 repack 都可能合理，但必须分别计费。当前 `MatMul::accumulation_type` 不代表允许
F32 activation 变 Q8：4a 先确定数值合同及所需最小 schema/显式转换，再给实际 owner 固定 FIX；不能
靠放宽 oracle 容差准入。精度或资源合同不同只作 NATURAL_ROUTE，不签同合同 WIN/LOSS。

Q4→F32 样例的活动源码与用户入口已退役；已经完成的互操作与同步回归事实保留，不重写成历史错误，
也不继续把它发展成量化默认 Provider。cuBLASLt 的自然浮点用途与量化 carrier 分别准入；没有证据必须提供长期全量展开功能时，
不为这个测试建立独立 shipping capability。所选 vendor 的内部算法与 HYLE 包装成本分别测量，默认
成熟实现不等于它天然接受每一种 Q4 编码。

2026-09-15 的 [CUDA 压缩接入调查](cuda-compressed-provider-admission.md)在现有 Windows shared SDK 上
原样编译并执行 original Marlin W4A16/group128 kernel：HYLE buffers/Queue/Completion，九次变化输入，
无展开 device 权重。它证实独立 native kernel 路线可行，不需要为接入重写 Graph。建议首个正式压缩
carrier 选 Marlin family，但 2024 original pin 不直接成为 shipping/SOTA 默认；当前 vLLM/SGLang
版本的 native façade、冷配置、错误与数值合同须按该调查的有限清单准入。GGUF 格式/原 ggml blocker
和 MLX 不由该实验解决；4a 仍部分完成，无新的产品百分比或模块关闭。

同日[调度边界调查](foundation-architecture-admission.md)又完成一个独立 MLX v0.32.2 C++ FFN 实验：
MacBook Air 六次变化输入正确、页对齐 host 输入真实借入、六次释放，且同 compiled function 跨线程
再次 trace。公开 `array.wait()` 的 signal 快路不足以独立证明错误终态，调查 fixture 改为在读回前显式
同步并检查受控 stream。MLX 的线程缓存、全局 allocator、managed 元数据和显式完成 bridge 是局部
接入约束；不能因这个独立程序能运行就宣布 Apple backend/provider 或 foundation 已完成。

#### 决策与纵切的对应关系

不新增一批平行 epic；以下是既有 5/6/6a 的具体合格条件。各路径仍用同一 Graph/Executable/Invocation。

| 既有工作包 | 固定旅程与判据 | 会改变决策的反例 / 处理边界 |
|---|---|---|
| 5 / CUDA 压缩 FFN | 采用 Qwen3-0.6B 的 D=1024/H=3072 gate/up→SwiGLU→down 形状，M=1 与 M=64；所选 Marlin family 的 symmetric group128 W4A16，保持压缩权重。经 HYLE Graph 调用成熟 GEMM/custom 中间计算，再回成熟 GEMM；两个独立 Invocation、变化输入、共享不可变权重、普通与 capture 路线 | 不接受仅旧单 GEMM 或全量展开。若 layout/scale/累加语义不等，修正该格式/operation/provider 合同，不放宽 oracle；若替换造成完整成本变差，保留成熟组合。custom 的可调用性不等于已证明更快 |
| 6 / Apple 委托边界 | MLX 原生支持的 FFN region，与 HYLE native Metal 前/后处理衔接；固定提交 owner，明确输入/输出 copy 或 import、managed 成本、错误终态与回收；prepared-only/禁止等待/有限私有内存等不满足的要求在接纳前拒绝 | 不能把任意 worker 上的 compiled call 当已预热制品，不能把 ready bit 当成功。若只能公开 host bridge，先计入完整成本并保留较大片段；高效 device bridge 未成立就不能签为成立，不强制移植 MLX runtime |
| 6a / Vulkan 正交边界 | 上传→SPIR-V 计算→native 后继/读回；重复复用资源，明确 barrier、queue-family ownership、timeline 与真实完成；原生 Vulkan/MoltenVK 各取证 | Queue 顺序不自动证明内存可见性。失败修 typed backend 同步/resource seam，不因其 API 不叫 CUDA stream 就新造一套 Graph |

Qwen 仅提供真实 FFN 拓扑和矩阵规模；本纵切的 W4A16 是明确选择的量化部署合同，不宣称该模型官方
BF16 checkpoint 自带这种编码，不下载完整权重或用合成权重证明模型质量。
[官方 config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json)给出 hidden_size=1024、
intermediate_size=3072 与 SiLU。M=1/64 是明确的 decode/prefill 工作点，不代表所有 batch 最优。

全局调度的验收看实际证据：本轮 HYLE 选择的片段/队列、boundary copy、常量/临时驻留、输入就绪、
输出完成与失败后回收；没有依赖的片段可以提交到不同 lane，但不保证硬件必然并行。两个成本对照必须
分开：同一实现 direct vs HYLE 的抽象税；成熟完整计划 vs 混合计划的实际收益。只有不可变实现才跑
预声明 P1，文档/状态变更不重测；本轮研究没有新增 P1，也没有向未实现路径发放通过。

#### 执行约束和停止条件

已完成的 0--4 与 CUDA runtime/资源/capture 事实保留。以下 **4a → 5 → 6 → 6a → 6b → 7** 是原实施
阶段的顺序与限制；当前只运行能回答架构问题的研究子集，不恢复本链条的正式实现：

1. 4a 对 CUDA 压缩 carrier、Apple MLX 各做一次公开 API 源码核对和一个最小执行探针；每条最多再评估
   一个窄 source-patch 备选。单条先固定不超过 2 小时的调查预算（SDK/真机构建等待另记），不是新建
   通用调研框架。若须大规模接管依赖 runtime、无法安全恢复运营失败或无合法精度合同，提交精确方案
   与取舍供用户决策；受影响路线不能通过，独立已证实的路线不清零。
2. vendor handle、native context/stream、per-run workspace 等只在准入后由真实 owner 建最小 seam。
   当前 CUDA backend CLOSED 不自动重开；若内部 Provider 接入确实缺少能力，另卡记录 concrete trigger。
3. 5/6/6a 在正常 public C++ consumer 上实现，同一 API 复用 direct/Graph 或 region；必要 schema/kernel
   只服务所选纵切。5 完成压缩路线并不解除既有 CPU ggml blocker，也不宣称获得全部 ggml 算子。
4. 6b 复用这些具体程序做一次实测 profile 回灌与有限生命周期/Obs 组合，不造完整模型或通用 tuner。
   性能比较先过语义/源码成本 preflight；只对已固定不可变候选跑适用 P1，不为状态/文档/计数重测。
5. 7 汇总所有适用门并做实现后独立复核。新事实首先修改被反证的 seam；只有 owner、lifetime 或热成本
   根本无法兑现才重开核心架构。缺 SDK/硬件、未执行或精度不等价不能签为通过。

#### 进度重估（同一基线，不是关闭证明）

本表是本阶段一次工程估算，不是第二份 registry；权重按基础工作包而不是代码行、提交数或用例数。
架构文档更新不增加实现分数。总估算约 **55%（约 ±5 个百分点）**，旧约 89% 不可换名沿用；百分比
描述实现及具名定向旅程的进展，不能推出性能/交付/模块通过，也不是剩余墙钟时间。

| 基础工作包 | 固定权重 | 本包估算 | 证据与尚欠内容 |
|---|---:|---:|---|
| 通用 Compute + CPU/CUDA runtime | 20% | 90% | CPU/CUDA backend 独立 CLOSED；current Compute shape/custom substrate 已实现，仍需真实新 Provider 与总资格 |
| Graph 编译与固定执行计划 | 30% | 85% | node-set/shape/三轴/重选/常量共享已有真实增量；managed/跨 runtime 和最终资格未完成 |
| CPU 真实 region/executor 组合 | 10% | 80% | mixed MLP 与 Windows 三交付 executor 已过；跨平台、并行 P1/完整候选门尚欠 |
| CUDA 正式浮点/压缩 Q4 Provider 组合 | 15% | 20% | runtime/capture 与独立 codec/native/vendor 预检可复用；正式压缩 carrier、精度合同、region/residence 未交付 |
| Apple MLX/Metal 组合 | 10% | 0% | 有设计/源码调查，尚无正式 backend/provider 实现旅程 |
| Vulkan/SPIR-V/MoltenVK 基础 | 10% | 0% | 有需求，无当前 factor/实际旅程 |
| 实测选路、Obs、跨运行时组合 | 5% | 10% | profile matcher/模拟反例等前置可复用；真实回灌、事件和混合终态链仍缺 |

最后的 foundation gate **尚未通过**，不再用另一个猜测百分比代表验收。上述任务全部实现也必须满足
对应适用门，才能报告基础完成并暂停。若范围真有新事实，显式说明分母变化，不把未实现平台挪到
锁后以维持高百分比。之后按子模块完成、重大调整或约 10 个百分点变化报告，不为每次文档提交涨分。

同一 `3ea0eced` 源码快照、统一 `loc-report.py --module-path ... --self-check`：全库代码 **688,607**；
自实现不含测试 **126,320**；Compute 产品 **17,000**，Graph 产品 **14,135**，合计 **31,135**。
包括现有 backend/provider，不计测试、样例、第三方、生成物和依赖模块，不把 LOC 当完成度。

本轮维护验证：docs 309/1129 relative links、closure 39/schema 4、`git diff --check` 通过；change-impact
只选择 docs/closure、跳过 94 项。没有 C++ 构建、性能采样、SDK 更新或 registry 迁移；文档通过不增加
产品资格或本表进度。

### 基础计划覆盖审计与有限补项

审计基线 `405fd2892e9dff41d4155b591ae99a5dd0c07f22`，只读核对 architecture、活动卡、公开 API、
测试与 benchmark。结果不是全源码红队或新产品 candidate；没有发现必须推翻 Device/Context/Queue、
Graph/Executable/Invocation 或三轴资源模型的证据。以下是任务分解，不是第二套模块状态。

| ID / 工作包 | 当前事实与源码锚点 | 基础 gate 前的有限完成条件 / owner |
|---|---|---|
| F1 / 6a | 原第 9 项把 Vulkan 整条路线后移，registry 无 Vulkan factor | 新 backend 先依法登记/审计；一条 SPIR-V 资源流水线覆盖内存可见性、command/timeline 生命周期及 native handoff；实际支持分平台披露 |
| F2 / 4 | `tests/compute_provider_onednn/onednn_provider_test.cc::probe_parallel_for` 与 Graph region 同名 probe 实际串行循环；现有双 Invocation 线程测试不是单次 fan-out | [真实 CPU executor 旅程](cpu-executor-foundation-probe.md)：复用 Sched，1/多 worker、嵌套、并发独立 scratch、关闭；只在真实公共组合反证后开 `compute_sched` 或重开依赖 |
| F3 / 6b | `tests/graph/profile_fixture.h` 明确 synthetic；现有 bench 尚未形成 measured profile 回灌旅程 | Graph/已有性能工具：固定完整候选、同合同采样、导出 identity+interval、重新 compile 验证选路；缺失/过期/不可分保持成熟顺序；不新增通用 tuner/全局缓存 |
| F4 / 5、6、6b | `cuda_graph_test.cc::CapturesEligibleSegmentsAndReplaysAHybridPlan` 只有 CPU kernel+CUDA copy；public CUDA 样例为 GPU kernel；delegated 测试使用 RegionProbe | 在原 CUDA Q4/Apple card 增补真实计算衔接与边界失败：已接纳前缀、迟到完成、owner 关闭/释放、预算回收；复用已有 atomics/OOM 单测，不制造全矩阵 |
| F5 / 6b | Graph §7 承诺 Obs seam，但 `src/compute`/`src/graph` 无 trace 消费，中央图也无 TRACE | Compute/Graph 按各自小卡落实最小 opt-in compile/submit/region/completion 事件；区分 host enqueue 与 device terminal；OFF 无布局/热分支税，不以 CompileReport 冒充 |
| F6 / 各 backend、7 | CUDA 已有 feature/三交付实证，后续 backend 尚无对应 card；Graph→Math component-contract 尚待执行 | 各 owner 在基础候选时完成 OFF/AUTO/ON、缺 SDK、独立选择、source/static/shared、runtime staging/许可证；第 13 项是最终汇总，不替代此门 |

Vulkan 的依据是 [Khronos 同步规范](https://docs.vulkan.org/spec/latest/chapters/synchronization.html)：
execution dependency 本身不等于 memory visibility；因此不能从 CUDA stream 测试外推 Vulkan 已被验证。
平台没有某 queue family/extension 时记录真实 capability 与 unsupported，不用软件模拟器冒充原生 GPU。

正式 CUDA Q4 Provider/region/residence 与 Apple MLX/Metal 仍是原第 5/6 项未完成工作，不再次统计为新缺口。
CPU ggml 的既有 blocker 也不因该审计被解除。完整模型、KV/batching、分布式控制面、全算子目录、
全自动 tuner/cache 仍不进入基础范围。

F2 的首个实际检查点已由 [CPU executor probe](cpu-executor-foundation-probe.md#windows-检查点) 完成：
`f6c45efd` 的 Native/oneDNN 在 Windows source/static/shared 上均通过真实 worker、嵌套和并发旅程。
现有 Sched public 组合成立，未新增 `compute_sched`；跨平台/最终 P1 仍归第 7 项，不把本检查点当全局关闭。

本次只校准架构/任务文档；不改变 registry、重开依赖、跑 C++/P1 或改公共仪器。后续产品修改先给实际
owner 的当前卡固定有限 FIX，新增 factor 保持合法 QUEUED→SCOPED→AUDITED。补项验证优先复用现有
public journey；只运行受影响目标，文档/状态提交不触发重测。

维护验证：closure 39/schema 4、docs 307/1113 links、diff check 通过；selector 仅选 docs/closure，
跳过 94 项。上述结果只属于本次文档校准，不能签任何新产品能力通过。

2026-09-14 用户指定暂停点：第 7 项的基础实现与对应验证完成后，先报告实际能力、剩余限制和性能/交付
证据，暂停与用户共同审视。没有新的继续指示，不进入第 8 项及后续横向算子/后端扩张；局部修复或单个
平台测试通过不等于达到这个暂停点。

进度报告同时附 Compute、Graph 及合计的自实现不含测试 LOC，绑定同一源码快照。Compute 计入
`include/hyle/compute` 与全部 `src/compute*` 内现有 runtime、schema、backend/provider 自实现；Graph
计入 `include/hyle/graph`、`src/graph` 与 Graph provider 自实现。使用 `loc-report.py --module-path` 明确
这个产品口径，不把仅 `compute` runtime factor 当成全部 Compute；不计测试/bench、第三方实现、构建
脚本、生成产物、注释和空行，也不把依赖的 Mem/SIMD 等实现重复计入。LOC 不代替基础 gate 完成度。

### 迁移完成后的独立架构复核

本次前置整体复核不替代迁移后的实现复核。后者属于第 7 项已有的终局红队，不以某个局部迁移
通过代替全架构通过，也不新建平行状态系统。以实现和上述真实探针逐项核对：

- HYLE 全局 DAG/放置/生命周期与 Provider 内部调度是否各有唯一 owner，未形成双重调度或隐藏同步；
- borrowed/materialized constant、workspace、per-run state 与 Completion 的所有权能否在失败和并发下闭合；
- leaf/region/placement 是否真正共用完整成本路由，渐进替换是否有明确 boundary，warm replay 是否无搜索；
- bounded/general、exact variant 与 capture 是否自然组合，而非依赖单个测试 shape 的特殊分支；
- CUDA Q4、Apple MLX/Metal 与 Vulkan 是否能复用同一基础合同，新增 Provider 是否只需局部适配而非改核心对象；
- direct C++ 用法和 source/static/shared 交付是否一致，依赖方向、文档及公开示例是否与实现相符。

报告区分已证实通过、仍需局部实现/验证、以及有具体反证才成立的架构重开项。不以追加无限研究或横向
算子填充代替这次有限复核。
