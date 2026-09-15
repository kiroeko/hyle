# CUDA 压缩量化 carrier：有限接入调查

## 研究卡

- baseline：`6619da46`；对应总路由计划 4a，仅调查 CUDA 部分，不恢复暂停的大 goal。
- 问题：成熟量化实现能否保留压缩权重，借用 HYLE buffer/stream 执行，而不接管全局 Graph？
- 范围：ggml、ds4、vLLM、SGLang 的具名量化路径源码审计；一个独立 Marlin kernel 接入实验；
  清理容易被当成量化产品模板的旧 Q4→F32 sample。已有同步缺陷的修复、回归和历史结果不删除。
- 非目标：shipping Provider、完整 FFN、GGUF loader、全格式转换、Apple/Vulkan、P1 或性能排名。
  不改产品代码、registry、依赖锁、构建工具或治理合同；不会由本研究签发模块关闭。
- 预算：本路线调查至多两小时；一个实验、一轮 fixture 纠正。若需要重写 runtime 或修改内核算法，
  停止并报告具体边界，不继续扩建测试框架。
- 实验：只读原始 Marlin `1f25790bdd49fba53106164a24666dade68d7c90` 的自包含 CUDA TU（Apache-2.0），
  不引入 Python/PyTorch runtime；host 端按其公开 pack 规则生成压缩制品。使用现有 Windows CUDA
  shared SDK，HYLE 分配 device buffers、创建 Queue、返回 Completion。最多两编译任务，串行运行。
- 预声明验证：W4A16/group128，K=N=1024，M=1/7/64，各三次变化输入；权重只冷上传一次，workspace
  冷清零后重复使用，最终 Completion 后读回。host double oracle 基于实际 FP16 解量化值；沿用
  upstream test 的平均绝对误差 / 平均绝对参考值 `< 0.001`，同时输出最大绝对误差，不声称严格
  FP32 reduction、逐元素相对误差保证或模型精度资格。报告真实驻留字节，不计时。
- 维护验证：closure/docs/diff/change-impact selection。实验只证明本机 raw-kernel 接入；
  source/static、其他平台、capture、并发、fault injection 和数值极端值不在本次签发范围。
- 样例退役的交付检查：selector 选择 `delivery:compute`。只读比较确认 `be2d8b05..6619da46` 的
  include/src/cmake/根 CMake 无差异；复用该产品的现有 CUDA source/static/shared custom-kernel
  consumer 制品并定向回放，不重建无变化的 SDK 或 oneDNN。该回放不算新 Marlin 的三交付资格，
  也不声称执行了 selector 默认的全新 SDK 构建命令。

## 结论 / 本次调查完成

**压缩量化 native carrier 可接入，不需要为此重写 Compute/Graph 对象模型。** 实验原样编译 Marlin
kernel，成功使用 HYLE device allocation、Queue 与 Completion；未引入 PyTorch 或第三方 graph。
这不等于 ggml 全能力已接入，也不等于已经交付 Marlin Provider。大 goal 保持暂停，registry 不变。

### 先分清反量化的三种情况

| 路线 | 实际含义 | 本轮判断 |
|---|---|---|
| 压缩常驻，kernel 内按 tile 解码 | GPU 全局内存仍主要读取低位权重；寄存器/shared memory 内转为算术类型 | 正常量化主路径；不是浪费量化 |
| 压缩常驻，某次计算临时展开后 GEMM | 换取成熟 GEMM 的可用性或吞吐，承担额外 scratch、转换和内存流量 | 成熟引擎确实使用；必须按完整成本与资源预算判断 |
| 冷期整份展开并长期保留 | 文件仍小，但相应 device 权重副本已失去压缩容量优势；若压缩副本也保留，峰值还要相加 | 可以是显式特例，不能代表 HYLE 的默认量化方案 |

Q4 转 F32 不会找回量化时已丢失的信息；是否再发生舍入取决于格式与目标类型。旧实验的 Q4_0
scale×小整数可在 F32 中表达，但权重从 18 bytes/32 values 变成 128 bytes/32 values，约 7.1 倍。
问题是把互操作实验当成量化集成的代表，不是证明了“反量化计算永远错误”。

### 具名参考与真实接入边界

以下五个引擎 pin 沿用同日已更新的 [参考版本表](cuda-q4-foundation-carrier-preflight.md#版本化来源与裁决)，
没有升级 HYLE 产品锁。新增 original Marlin reference 的 HEAD 为 2024-09-04 提交；不能称为 2026 年
最新版 vLLM kernel，更不能拿其旧论文结果当本机性能资格。

| 参考与源码入口 | 查到的真实行为 | 对 HYLE 的含义 |
|---|---|---|
| ggml `ggml-cuda.cu::ggml_cuda_mul_mat`、`mmvq.cu`、`mmq.cu` | 优先适用的向量/量化 kernel，之后可退到 cuBLAS；MMVQ/MMQ 保留 Q4 权重，activation 使用 Q8_1 scratch；cuBLAS 路径可分配转换后的浮点临时权重 | GGUF Q4_0/Q4_K 有成熟压缩计算；F32 输入接口不等于严格 F32 activation 算术；不能直接偷调依赖 ctx/pool 的私有 kernel |
| llama.cpp `src/llama-graph.cpp::build_ffn` | gate/up、激活与 down 的多算子 FFN 由 ggml 后端承载 | 正式纵切应检验整段 FFN 的边界与复用，不用孤立 tiny dot 宣称承载推理 |
| ds4 `ds4_cuda.cu::ds4_mmq_q4_K_dense`、`cuda_q8_f32_ptr`、`cuda_matmul_q8_0_tensor_labeled` | 有保持压缩的 Q4_K 路线，也有受控 Q8 浮点缓存；另有 GLM opt-in 的 batched Q8→F16 GEMM，源码明确不是 DS4 默认路径 | 展开可以有真实用途，但不同 dtype/模型/开关不能混成一个默认路线 |
| vLLM `mixed_precision/marlin.py`、`libtorch_stable/quantization/marlin/marlin.cu` | 冷 repack；热用 packed weights/scales/workspace；raw-pointer `marlin_mm` 下接 CUDA kernel，但 host TU 和公开入口含 Torch Tensor、异常与 allocator | 不是一个可直接链接的无 Torch C++ runtime 库；可以评估窄 native façade，但维护与版本责任必须显式 |
| SGLang `compressed_tensors_wNa16.py` | 加载后 repack、执行时调用 Marlin WNa16 路线 | group-wise W4A16 是真实 serving 形状，不是为 HYLE benchmark 创造的算子 |
| original Marlin `marlin_cuda_kernel.cu::marlin_cuda` | 自包含 base-CUDA TU，直接接受 A/B/C/scales/workspace/device/stream；group128 或 per-column，FP16 输入输出、tensor-core 运算，部分跨 block reduction 经 FP16 | 可在 HYLE 拥有的内存/Queue 上作为 native leaf 工作；不是 GGUF Q4_0/Q4_K，也不承诺严格 FP32 reduction |

可定位来源：[ggml dispatch/转换](https://github.com/ggml-org/ggml/blob/456172ec733a135778adcd32d00e576a58232e45/src/ggml-cuda/ggml-cuda.cu)、
[ds4 CUDA](https://github.com/antirez/ds4/blob/9139e2ae58a41503968a500f36f75895c1ba63fc/ds4_cuda.cu)、
[vLLM Marlin host](https://github.com/vllm-project/vllm/blob/51918e252af4dfff381001f42b7ebe6a0454501b/csrc/libtorch_stable/quantization/marlin/marlin.cu)、
[SGLang WNa16](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py)、
[original Marlin kernel](https://github.com/IST-DASLab/marlin/blob/1f25790bdd49fba53106164a24666dade68d7c90/marlin/marlin_cuda_kernel.cu)。

### 实验事实与 P0

[完整源码和原始日志](../../../../evidence/correctness/compute-cuda-compressed-admission/README.md)：

- Windows、RTX 4070 SM89、CUDA 13.3.73、MSVC 19.50，已有 installed shared HYLE SDK。
- M=1/7/64、K=N=1024、group128；九次变化输入全部通过预声明误差检查。L1 误差比为
  0.0002763–0.0003879；最大绝对误差 0.02984。只证明这些有限数据，不是模型精度/极端值资格。
- device 权重 524,288 bytes、scales 16,384 bytes，合计 **528 KiB**；独占 lock workspace 512 bytes。
  没有 GPU 浮点权重副本。host oracle 的展开数组仅用于验证，不进入产品或 GPU。
- 首次编译缺少 NVCC relaxed-constexpr 选项而失败；唯一 fixture 纠正补编译选项及 MSVC UTF-8 后通过。
  第三方 kernel 源码未改、参考工作树干净；保留原始失败日志和未屏蔽的模板警告。
- 算术工作 O(MNK)，权重和 repack O(KN)，该配置的独占 workspace O(N)；没有每次重建第三方图、
  上传权重或全量展开。原 launcher 仍每次选择 launch shape、设置 function attribute；即使显式传入
  冷查询的 SM 数，也不能称为完整 prepared/zero-host-allocation route。CUDA runtime 隐式成本未测。
- 本次无 P1、无测速胜负；未修改 HYLE 的 MatMul 精度合同来容纳此 kernel。

### 固定下一步，而非再启动一轮大架构讨论

建议正式 CUDA 压缩首纵切采用 **Marlin family 的 symmetric group128 W4A16**，作为 native kernel
provider；输入输出 FP16、准确 packed format、scale 和 reduction policy 都必须显式。GGUF Q4_0/Q4_K
保持独立格式，不做隐式互转。正常 decode/prefill FFN 复用这个 leaf，随后才检验融合/region。

正式版本应优先核对当前 vLLM/SGLang 分支；本实验的 original Marlin 只证明低层接入边界，不自动成为
shipping 默认或 SOTA baseline。已审到的窄修改面是：

1. 版本锁定的 kernel-source factor，加一层 native launcher；不引入 PyTorch Tensor、allocator、scheduler。
   最新分支连 `core/scalar_type.hpp` 都引用 Torch exception header，因此不声称只换 include 路径即可。
2. 把设备属性查询、kernel/config selection、`cudaFuncSetAttribute` 前移到 prepare，捕获所有 CUDA
   返回码；原版只返回 shape 错误且忽略 attribute 返回码，不能原封不动签发 HYLE fallible Provider。
3. packed weights/scales 是共享不可变常量；workspace/输出归各 Invocation，禁止并发共享 lock workspace。
   C++ source/static/shared 走同一个 Provider factor，不向消费者暴露第三方私有头。
4. 数值语义不偷换：旧版部分 FP16 reduction 不能路由到严格 FP32 累加承诺。当前分支虽有
   `use_fp32_reduce`，仍需检查完整实现和所选 dtype/scale 舍入，不能只看参数名判等价。

这些是版本化内核接入和 adapter 的工作，不是修改 HYLE 全局 Graph owner 的理由。若最新版需要大规模
移植 runtime 或改内核算法，本研究的批准面到此为止，报告具体超界再决定，不静默退到旧版或继续扩范围。
后续有限验收仍是：正式 leaf＋FFN、capture/Completion、独立 Invocation、失败/回收、三交付，以及同
kernel direct vs HYLE 的完整成本；通过后才谈其他 format 或 provider 广度。

原 ggml CPU blocker、ggml CUDA runtime 的 import/fatal-error 边界与 Apple MLX 准入均未被本实验解除。
4a 不能整体写完成；基础 gate 仍未通过。

### 旧样例处置

移除 `samples/package/compute/cuda_q4_preflight` 五个文件及 samples/user 活动入口。它们不是错误数值
实现，但已完成的互操作探索不应继续引导量化产品设计。源代码可从 baseline `6619da46` 恢复；旧研究卡、
原始日志、CUDA v3 修复及 `SynchronousDeviceFillAndCopyDoNotReturnBeforeCompletion` /
`SynchronousResourceInitializationPrecedesIndependentQueue` 回归保留。新实验仅位于 evidence，
不增加新的用户展示样例或 shipping capability。

## 维护验证

closure：39 components/schema 4 通过；docs：310 Markdown/1144 相对链接通过；`git diff --check` 通过。
selector 选择 docs/closure/delivery:compute，跳过 93 项；交付按本卡约定只定向回放既有 CUDA
source/static/shared public consumer，三项通过，未执行默认的全新 SDK/oneDNN 构建。
未修改产品或 registry，未运行 P1，未新增模块 CLOSED，也不增加基础实现进度估值。
