# Compute/Graph 基础架构接入与调度边界调查

> 历史研究记录。下面当时的“保留现有对象”约束已由 2026-09-15 用户要求及
> [四路线研究裁决](four-route-architecture-decision.md#从零推导而不是迁就旧实现)取消；实验事实仍可引用，
> 旧类、接口、factor 和测试形状不构成新架构的保留要求。本卡不授权恢复产品施工。

## 本轮范围

- baseline：`0b5da87c`。用户要求把基础架构的大改风险查清，同时保留性能优先和渐进式 best-of。
- 保留现有 Device/Context/Queue/Completion、Graph/Executable/Invocation；本轮不写产品代码，
  不更改 registry、依赖锁或治理系统，不填算子目录，不创建新的大 goal。
- 固定问题：CUDA 压缩 kernel 的数值/冷配置边界；MLX 的公开资源、线程、异步完成/错误和重放；
  HYLE 能控制的全局调度与不能控制的第三方内部行为；局部替换的格式、融合和完整成本条件。
- 证据：前轮 CUDA 真机结果复用；核对当前 vLLM/SGLang/ggml/ds4 和 MLX 的具名入口，Vulkan
  只核对公开同步/队列合同。本轮只做一个 MLX C++ 接入实验，不签新 backend 或 Provider。
- MLX 调查预算至多两小时，一个实验、一轮 fixture 纠正。机器仅缺构建工具时可使用官方同版预编译
  runtime 作 C++ API 研究，不安装/修改全局开发环境；它不代替未来源码交付资格。
- 预声明实验：MLX v0.32.2 官方同版 headers/dylib；F32、M=4/K=1024/H=256 的
  gate/up MatMul → SwiGLU → down MatMul，两个提交线程各三次变化输入；host double oracle，
  逐元素 `abs(actual-expected) <= 1e-4 * max(1, abs(expected))`，不测完整模型精度；
  无复制输入的实际可用条件、完成前保活和完成后释放；同一编译函数跨提交线程是否再次 trace。
  不做 P1、不更换数值阈值跑到绿；不能主动制造 GPU hang/device loss。异步故障与未测场景按源码
  证据和未验证限制分别报告，不能冒称 fault-injection PASS。
- 该实验只裁决 MLX 自身的 import/lifetime/线程缓存，不冒充 HYLE Metal backend、跨 Provider
  FFN、量化、全局调度或性能纵切。正式纵切须从 HYLE Graph 进入并验证真实边界与失败回收。
- 输出：已有合同是否足够、真实矛盾需要的最小修正、后续实现门；文档/closure/diff 检查。
  调查完成不等于基础实现或性能已经通过，更不承诺永远不改任何内部实现。

## 结论 / 有限调查完成

没有查到必须推翻 Device/Context/Queue/Completion 或 Graph/Executable/Invocation 的反证。
但四条路线不能用同一种运行时所有权兑现：CPU/CUDA 的公开 leaf seam 与 MLX delegated runtime
不同，Vulkan 还有显式内存可见性。**本次不签基础实现完成或性能通过，不重估到 90%。**

长期合同已写回 [Compute](../../../architecture/compute-design.md#24-hyle-拥有计划与边界provider-拥有片段内执行)
与 [Graph](../../../architecture/graph-design.md#9-基础验收与停止点)。本轮历史纵切依据见
[总计划](compute-graph-route-unification-audit.md#决策与纵切的对应关系)，不复制第二套模块状态。

### 四条路线共用什么、不强行统一什么

| 路线 | HYLE 的整体控制 / 自然快路径 | 不能假定已成立 |
|---|---|---|
| CPU | HYLE Native/SIMD 或 oneDNN prepared primitive/region，显式共用持久 executor、绑定线程额度和 scratch；Graph 控制依赖/放置/片段与寿命 | “每 op 一个线程池”不是合理组合；ggml CPU 的既有 runtime blocker 未解除，不能假装全部第三方都能借 executor |
| CUDA | vendor/成熟压缩 kernel 借入 HYLE buffer/stream，冷配置 handle/tactic/packing；连续兼容片段可 capture；Graph 保留权重驻留、跨 stream 事件和独立 Invocation | cuBLAS 不等于任意 Q4 格式可用；Python/Torch host API 不等于无成本 native leaf；一个旧 Marlin kernel 接入不等于最新版或完整 FFN 已资格化 |
| MLX/Metal | MLX 作为 managed delegated region，固定提交 owner、明确输入输出与 Completion bridge；原生 Metal 是严格资源/捕获/custom 路线 | MLX compiled call 不是零分配 replay；当前公开 route 不自动接受任意 Metal queue、任意输出地址或 per-Invocation 硬内存配额 |
| Vulkan | backend 拥有 pipeline/descriptor/command buffer、queue family、barrier/semaphore；Graph 只给出资源依赖和片段；MoltenVK 单独验证 | 顺序提交不替代内存可见性，不能把 Vulkan queue 硬套成 CUDA capture；本轮只读官方同步合同，未新增 Vulkan 真机通过 |

SOTA 是精确 workload 的实现和完整成本资格，不是“全局 owner 在 HYLE”自动产生的属性。候选统一为
完整语义/端口/物理格式/资源/完成合同；同一边界可放入成熟 leaf、融合 region 和 HYLE 候选。冷选择后
只执行既定计划；完整混合计划没有合格收益时保留成熟路线。单 op 赢不能抵消未计入的 repack、copy、
同步、线程转交和 fusion 损失；也不承诺一般 DAG 搜索的全局最优。

主流引擎本身有分层：vLLM `v1/cudagraph_dispatcher.py` 用具名 batch descriptor 选择 FULL/PIECEWISE/NONE，
SGLang `runner/base_cuda_graph_runner.py` / `decode_cuda_graph_runner.py` 预备 capture bucket、输入
buffers、stream/variant 后 replay。它们支持 HYLE 将用户模型/batching policy 与片段执行分开，但不是
HYLE 实现已高效的证明。ggml/llama.cpp/ds4 的压缩 FFN 事实复用前轮具名源码调查，没有更新产品依赖。

[vLLM dispatcher](https://github.com/vllm-project/vllm/blob/51918e252af4dfff381001f42b7ebe6a0454501b/vllm/v1/cudagraph_dispatcher.py)、
[SGLang capture/replay](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)、
[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)、
[Vulkan 同步](https://docs.vulkan.org/guide/latest/synchronization.html)。

### MLX 实验和源码确认

[完整 fixture/日志](../../../../evidence/correctness/compute-mlx-admission/README.md)：MLX v0.32.2
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`，官方同版 Metal wheel；MacBook Air `Mac16,12`、
macOS 26.6.2、Apple clang 21.0.0、SDK 27.0。参考 main 更新到 `8f76a0aa`，但实验没有混用其 headers。
本机无离线 `metal` 编译工具，因此用 wheel 的 C++ dylib/metallib，只作公开 API 调查；未安装全局环境。

- F32 FFN M=4/K=1024/H=256，六次变化输入全部通过预声明 oracle；最大绝对误差
  `7.30185e-09`。这是很小的有限数据集，不是模型精度、完整 shape 域或性能资格。
- 六次 page-aligned 输入均保持相同 host pointer；caller array 活着时没有释放，drain/销毁后释放六次。
  只证明这台机器的这些资源，未证明任意指针/任意 native Metal buffer 无复制接入。
- 同线程三次调用 trace=1；同一个 compiled function 转到第二个线程后 trace=2。与 `compile.cpp`
  的 thread-local cache 一致。证明 retrace，不推断一定重新产生了设备机器码；compiled cache hit 本身
  仍经 `compile_replace` 创建数组/元数据，不能签 prepared。
- 初版 fixture `66dea696` 已通过上述六次。源码复核发现 `array::wait()` 可经 `is_available()` 的已 signal
  分支跳过 `Event::wait/check_error`；唯一 fixture 纠正 `eb94ae05` 把原本每轮末的显式 stream
  `synchronize()` 前移到读结果前，纠正注释，再次六次通过，未改阈值/矩阵。初版 clang 默认 target=26.0
  而 dylib 最低版本=26.2 的链接警告保留；纠正版显式 target=26.2，运行 OS 更高，未改产品平台合同。
- `CommandEncoder::commit` 的回调保存 Metal command-buffer failure、poison 并 signal event，
  `synchronize()` 则有明确错误检查。这是源码证据；本轮没有主动制造 GPU fault/OOM/hang，不能签发
  异步错误传播或跨 HYLE 运行时 drain 资格。ready bit 不足以签成功 Completion。
- Metal allocator 的 singleton 和全局 memory limit 不能证明 Context/Invocation 的私有内存上界；
  `array(void*, ...)` 明确允许无法借入时复制。未知成本继续未知，严格模式应在接纳前拒绝。

对应源码：[compile cache/replace](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/compile.cpp)、
[array import/wait](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/array.cpp)、
[event](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/event.cpp)、
[command completion](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp)、
[allocator](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/allocator.cpp)。

### 还没完成的接入，不应再算成基础完成

1. 正式 CUDA 压缩 Provider 的版本/native façade/数值封装仍须在既有 4a 内明确。当前 vLLM 的
   `marlin_template.h` 有 FP32 global reduce，但 SM75 某些实例仍有 `use_fp16_accum`；一个 flag 不能给
   所有 ISA/格式签 FP32。现有 HYLE builtin MatMul 恰好两个输入，不能直接传独立 scales/zeros。
   需用已有 typed operation 扩展机制补全量化语义/端口，不能靠 `void*` 偷渡或称 builtin 已完整。
   这些是 ComputeOps/Provider 的有限改动，不是重写 Graph 生命周期的理由。
2. MLX adapter 要在固定持久提交 owner 上完成准备/执行/清理，先按 managed、私有内存 unknown、
   显式受控 stream 完成 bridge 准入。没有证明的 device-only interop、并发、无热 JIT 或 source/static/
   shared 不签 PASS；只有完整 MLX→native Metal 纵切才能确定这些边界的实际成本。
3. Vulkan 的真实资源可见性、四路线共用的异步失败/并发/预算、完整 route profile 与交付资格，仍属于
   原有 5/6/6a/6b/7。原 CPU 证据复用，未实现路线不涨百分比；只修被具体反例推翻的 seam。

本轮只完成准入研究，没有创建新的大 goal、修改 registry/产品/依赖锁或启动 P1；亦未解除 ggml blocker。
维护验证：selector 仅选择 docs/closure、跳过 94 项；docs、39 components/schema 4 closure 与
`git diff --check` 通过。未触发无关产品重建/性能测试，registry 保持原样。
