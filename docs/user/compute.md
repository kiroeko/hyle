# HYLE Compute：何时直接执行一个计算

> 本页描述当前 Compute runtime foundation、CPU operation surface 与 CUDA runtime；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。普通 CPU 入口自动组装已构建的 Native/oneDNN 实现；
> 标准 CPU 构建使用 `HYLE_COMPUTE_ONEDNN=AUTO`，在支持的 x86-64 目标接入 oneDNN MatMul。
> CUDA factor 提供 device/context/queue、memory resource、stream interop、byte transfer、Completion 与
> command capture 基础运行时。R4 已接入 cuBLASLt、Marlin、SGLang 与 HYLE SwiGLU 的独立 Provider，
> 普通 CUDA Direct/Graph 入口、真实 FFN/capture、Win/WSL 三交付与当前 SM89 工程性能验收已完成。
> Apple MLX/Metal 的原生 Direct/Graph、三交付和 M4 限定性能验收已完成；
> Vulkan/SPIR-V 的 R6 也已完成真实计算、三交付与已测设备/版本域的有限性能验收。
> HIP/ROCm 当前无限期不纳入产品，也没有空实现或构建开关。

四路线目标、MLX 内 Metal 扩展与每个 native runtime 的边界见[路线设计](../architecture/compute-routes.md)。
研究程序不属于安装 SDK；本文的现有 API 也不保证在本次重构后兼容，不把目标设计当成已交付功能。
`cpu::create_context()` 在冷路径组装默认实现，prepare 只匹配和绑定固定规则，不运行计时比赛。
普通用户不排列 Provider 或提供性能资料；规则与性能验收状态由当前 CPU 任务卡记录。
路线原则见[开发期优选规则](../architecture/compute-design.md#41-开发时验证-best-of准备时应用规则)。

Apple 与 Vulkan 各自使用原生 Program、运行槽和资源；普通 Direct/Graph 用法分别见
[Apple MLX/Metal](apple-compute.md) 与 [Vulkan](vulkan-compute.md)。本文后续 Queue/Provider 的
细节属于 CPU/CUDA 接入，不要求这两条原生路线采用同一执行器。

构建时默认 `HYLE_COMPUTE_CPU=ON`。CUDA-only SDK 显式设 CPU OFF、CUDA ON；四条路线都 OFF 时仅有
计算语义/资源合同底座，无 CPU/Native 入口。安装包的实际 capability、头与 CMake 选项见
[交付配置](distribution.md#4-消费安装包)。

当程序已经知道要执行哪个 operation、在哪个 device 上执行，并希望复用固定 shape/layout 的路线时，
使用 Compute。跨节点拓扑、liveness 和可复用图执行物使用 [Graph](graph.md)；模型、KV cache、batching、
sampling 和 serving 留在推理引擎。

计算选择 CPU、CUDA、MLX/Metal、Vulkan 中的一条路线。CUDA 的普通入口和完整 FFN Graph
已完成 R4 正确性、三交付与适用性能验收；Apple R5 与 Vulkan R6 也已完成各自有限验收。
HYLE 不提供同一次模型计算由 CPU 与 GPU 分担层/FFN/专家的联合推理。纯 CPU 路线、GPU 的 host
提交/输入输出传输/同步，以及同 GPU 多 Provider 组合不属于该排除项；选择 GPU 后不应期待缺算子
自动交给 CPU。完整范围见 [架构边界](../architecture/compute-design.md#13-四条独立路线与明确非目标)。

direct `prepare()` 仍接收具体 TensorDesc，不要求用户构造 shape program。需要 batch × tokens、concat
输出轴等派生维度时，使用 [Graph 的独立与派生维度](graph.md#独立与派生维度)：默认 exact compile，
或显式声明完整域并在创建 Invocation 时绑定。后者目前支持固定 hidden width 的 Native RMSNorm；
oneDNN 仍 exact-only，可进入 Graph 的预编译 exact 变体；Graph 也已提供
[有限精确变体与通用回退](graph.md#有限精确变体与通用回退)。完整动态属性尚未交付。不要直接调用私有
factor seam，也不要把单个 family 的域支持当成整个算子目录都支持动态尺寸。

## 普通 CUDA 入口

### 执行前检查具体请求

CPU/CUDA 用描述符和 memory resource 调用普通 `prepare()`，即可检查所选设备上实际的
dtype、shape、layout 与资源是否可执行；此时无需分配 tensor payload 或创建 Queue。
准备可能编译并分配程序状态，但不读取或执行输入数据。保留成功结果供后续 invocation 使用，
无需另做一次查询或再次准备。`infer_output_descriptions()` 只检查共同数学描述，不证明路线支持。

准备失败保留具体原因：`InvalidArgument`/`OutOfRange` 表示请求不合法，`Unimplemented` 表示
当前资格条件不受支持，资源不足和编译错误按实际错误返回。所有实现都不支持时，错误包含 Provider
名字和其拒绝原因；已进入实际准备的明确拒绝优先于后续更宽泛的 proposal 拒绝。执行期不重新选择。

### 创建 CUDA 程序

选择 `HYLE_COMPONENTS=compute`、`HYLE_COMPUTE_CUDA=ON`；CUDA-only 再设
`HYLE_COMPUTE_CPU=OFF`。源码与安装包均链接 `hyle::compute`：

```cpp
#include <hyle/compute/cuda.h>
#include <hyle/compute/backends/cuda.h>

auto context = hyle::compute::cuda::create_context({.device_ordinal = 0});
if (!context) return context.error();
auto resource = hyle::compute::cuda::MemoryResource::create(
    *context, hyle::compute::cuda::MemoryKind::kDevice);
if (!resource) return resource.error();
auto queue = context->create_queue();
if (!queue) return queue.error();

auto prepared = hyle::compute::prepare(
    *context, hyle::compute::MatMul{}, activation_desc, weight_desc, resource->ref());
```

后续依 `workspace_requirement()` 分配 CUDA workspace，为每个并发槽创建 invocation；绑定同设备
tensor，submit 后显式检查 Completion。完整 Q4 gate/up → SwiGLU → dense down 样例见
[CUDA Direct FFN](../../samples/package/compute/cpp/cuda_ffn.h)。该样例连续执行三组变化输入，
没有 Provider 列表、计时资料或用户 kernel callback。

| 默认规则 | 支持域 |
|---|---|
| cuBLASLt dense MatMul | 非空 FP16/FP32 rank-2，row/column-major 输入、dense row-major 输出；FP32 accumulation，FP32 不降为 TF32 |
| Marlin QuantizedMatMul | `cuda::QuantizedWeights` 的 signed symmetric group128 Q4；FP16 activation/output、FP32 reduction；SM80+，M=1..64、K 为 128 倍数、N 为 64 倍数、N*K<=INT32_MAX |
| SGLang precise gated activation | dense FP16/BF16/FP32 packed halves；SiLU/GELU/GELU-tanh 在 FP32 计算及乘法后一次舍入；FP16/BF16 hidden%8、FP32 hidden%4；输入输出16字节对齐，输出元素数<=INT32_MAX，不重叠 |
| HYLE gated activation | separate 输入或不满足上述向量宽度的 packed halves；dense FP16/BF16/FP32，同三种激活、精度与元素数/不重叠限制 |
| HYLE 基本 tensor | plain View、Copy、Fill、Select；F32/F16/BF16 Cast、全部既有 Unary/Binary 模式；rank≤8，rank0/broadcast/非负及奇数 byte stride，非 View 输出 dense、元素数≤INT32_MAX |
| HYLE CUDA norms | dense F32/F16/BF16 RMSNorm、LayerNorm，最后一轴、FP32 accumulation；scale/bias 各自为输入类型或 half 输入的 F32 参数，元素数≤INT32_MAX；LayerNorm 使用先均值再 centered variance |

基本 tensor 使用同一普通 `prepare()` 和 CUDA Graph 入口，程序准备后无需 workspace 或槽私有
device storage。View 按共同 byte offset 必须别名原 storage；其余逐元素操作仅在输入与输出描述符
完全相同时允许精确原位，Copy/Cast 始终不重叠。Select 原样保留被选择值的位模式；浮点算术使用
FP32 中间值并在末尾舍入一次，Cast 窄化为 RNE。Norm 参数在计算前保持 F32，不先窄化；允许精确
input0 原位，其他参数和部分重叠拒绝。非连续 norm 输入先用普通 Copy materialize，并计入实际成本。
Norm 使用 CUDA SDK Cooperative Groups 的行归约，一次 native launch、零 workspace/槽私有状态。
Reduce 和 Softmax/LogSoftmax 支持 F32/F16/BF16、任意轴/普通 byte stride、dense output 和 F32
累加；半精度 Reduce 必须显式声明 F32。Reduce 不允许原位，空集合仍写约定 identity，Min/Max
传播 NaN；Softmax 可用广播 Bool/同类型 additive mask、scale 和精确 input0 原位，无有效值的行
产生 NaN。每个原子一次 native launch，无 caller workspace 或槽私有状态；Softmax 在≤8192列时
缓存 F32 行（最多32 KiB片上空间），更长行读取输入三遍。上述输入/输出非空元素数≤INT32_MAX。
这些新能力尚待 C1 完整片段的等责性能和全部交付验收，不从正确性推断最快规则。
TopK 使用同一普通入口，返回原始位模式的 F32/F16/BF16 values 与 I32 indices；支持任意轴和
byte stride，两个输出 dense，输入与输出及两输出之间都不重叠。同分（包括±0）按较小列号优先，
NaN 排在全部非 NaN 后；largest/smallest 均可，sorted=false 仍可返回有序结果。k=1 使用 SDK
选择，k>1 对≤4096列使用 CUB BlockRadixSort；都一次 native launch、无 caller workspace/槽
私有状态，片上排序 scratch 和 code storage 仍有实际成本。长序列 k>1 当前准备时明确拒绝。
Concat 支持同类型的任意数量 plain 1/2/4/8 字节输入、任意轴/byte stride 和 dense output，
逐位保留数据，无输入输出重叠，输出元素数≤INT32_MAX。每最多四个输入一组原生 kernel，常见
双输入一次提交；空组不取指针或启动 kernel，程序元数据按输入数量冷分配，无 caller workspace
或槽私有状态。普通 Graph 可直接组合 View→Concat 和窗口 append，并使用原生 capture。
Gather 支持 plain1/2/4/8字节 source、I32/I64 indices、任意axis/byte stride和dense output，逐位
保留数据。默认 checked 在 GPU 验证全部索引后才写结果；越界通过真实 Completion 返回 OutOfRange，
该 Gather 的 output 保持不变，失败的 Invocation 需重新创建。无需下载索引或模型数据到 host。
source/indices/output 不能重叠；非空 output/indices 元素数≤INT32_MAX，source 轴与地址计算保留64位。
连续后缀按16字节块或原标量宽度搬运，非16字节对齐的合法输入仍可执行；无 caller workspace 或
槽私有状态。checked prefix 的状态 reset/4字节回报，以及原生 capture 的状态合并仍有实际成本。
整数/FP8 Cast 与算术、Scatter/Scan 等
尚未由这批 CUDA 接线覆盖。新增 BF16/门控模式及基本 tensor 的完整性能资格仍待 C1 bundle；
下述 R4 性能结论只签原有 FFN 域。

权重用 `cuda::QuantizedWeights::pack()` 冷准备，owner 只保留 Q4 payload 与 FP16 scales；
`weights()/scales()` 借用 owner，搭配 `QuantizedWeights::operation()` 使用。普通用户无需认识
vendor permutation。其余 dtype/shape/operation 明确拒绝；prepare 绑定规则与 vendor tactic，
执行期不搜索、计时或重新 packing。向量域在 prepare 固定16字节对齐合同，运行绑定不满足时明确拒绝，
不会热切换或隐藏 copy；专家可用 `cuda_native::provider()` 保留2/4字节对齐的 HYLE 路径。
默认激活规则依据完整 FFN 比较：HYLE 未呈现跨实验顺序成立的明确优势，采用合格成熟实现。
硬件/版本条件仍可在冷匹配时使用；当前性能资格只覆盖 RTX 4070/SM89、driver 610.62、
Toolkit 13.3.73，不外推其它 GPU 或版本。
最终默认 Direct/Graph 的两种 FFN 形状均通过5%工程非劣门；这是同生命周期成熟实现的接入验收，
不代表 HYLE kernel 战胜 SGLang。完整结论、单个小幅负向分层与原生边界成本见
[R4 验收记录](../implementation/tasks/compute/cuda-v1-four-route-foundation.md#r4-终局closed)。

Context、MemoryResource、外部 tensor/压缩权重 owner 必须覆盖各自借用和在途工作。每槽的 workspace
与可变 vendor handle 独立，成功完成后复用；失败终态后重建槽。vendor/driver 私有存储可能无法
精确计费，不能把 unknown 当零。异步上传下载与 native stream 接入仍使用 backend/transfer API。

## 当前能力

| 任务 | 公开形状 | 当前合同 |
|---|---|---|
| 设备发现 | `Backend::device_count/device` | 每个显式 backend 只枚举自己的设备；`cpu::device()` 直接取得唯一 CPU |
| 执行域 | `Backend -> Device -> Context -> Queue` | CPU Queue 在调用线程同步、按序执行；没有全局 registry、隐藏 worker 或默认设备 |
| tensor schema | `dense/strided_bytes/blocked_dense/blocked_strided/opaque` | rank <= 8；logical dtype 与 physical encoding 分离；stride 明确以 byte 为单位；view 借用 Mem storage |
| operation schema | 21 个 typed family + inline custom | 含 grouped Q4 的完整 scale/bias schema；可做 arity/shape/layout/alias/effect/precision 检查，schema 存在不等于已有可执行 provider |
| 零拷贝 tensor 变换 | `View::reshape/slice/permute/broadcast_to` | submit 不映射或搬运数据；输出必须绑定到输入中由 view 推导出的精确起始位置 |
| materialize / 初始化 | `Copy` / `Fill` | Copy 支持 plain dense/strided 输入并产生 dense 输出；Fill 对全部已登记 scalar dtype 写入精确 bit pattern |
| 推理浮点 storage 转换 | `Cast` | f32/f16/bf16/E4M3/E5M2 的完整互转；dense 复用 SIMD whole-buffer 路线，strided 输入显式 materialize |
| 单目计算/激活 | `Unary` | f32/f16/bf16 Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid/ReLU/GELU/GELU-tanh/SiLU/Softplus；dense 复用 SIMD，半精度和 Softplus 使用固定大小栈上分块，plain stride 可 materialize |
| 双目计算 | `Binary` | f32/f16/bf16 Add/Sub/Mul/Div/Min/Max 与 NumPy-style broadcast；Min/Max 使用 `fmin/fmax` 语义 |
| 条件选择 | `Select` | 可广播 Bool condition；对任意 plain 1/2/4/8-byte scalar branch 原样选择 bits |
| reduction | `Reduce` | f32/f16/bf16 Sum/Mean/Min/Max，半精度显式 F32 accumulation；任意 axis 集、负 axis、keep-dim 与普通 byte stride；dense 常用路线复用 SIMD |
| 前缀求和 | `Scan` | i32/f32，inclusive 或从零开始 exclusive，正负 axis、普通 byte stride；整数溢出不写任何 output；dense input 可精确原地执行 |
| normalization | `RmsNorm` / `LayerNorm` | dense f32/f16/bf16 `[..., hidden]`，F32 accumulation；RMSNorm 可选 weight；LayerNorm 可选 scale/bias；半精度参数也可用 F32；允许精确 input/output in-place |
| matrix multiply | `MatMul` | f32 `[...,M,K] * [...,K,N]` 与 NumPy-style batch broadcast；两端 batch 都可广播，B 的末两维可 row-major 或物理 `[...,N,K]` 的转置 view |
| gated activation | `GatedActivation` | dense f32/f16/bf16 packed `[...,2*d]` 或 separate gate/value；SiLU、exact GELU、GELU-tanh 三种模式均产生 `[...,d]`；半精度在 F32 activation 与乘法完成后一次舍入 |
| 索引读写 | `Gather` / `Scatter` | plain scalar storage、i32/i64 indices、dense output；Scatter assign 支持任意标量位宽，add 支持 f32/f16/bf16，并可精确原地更新 input0 |
| 张量拼接 | `Concat` | 同 dtype/rank 的 variadic plain inputs，沿显式 axis 产生 dense output；dense 输入按连续后缀块复制 |
| 旋转位置编码 | `RoPE` | dense/strided f32 data、i32/i64 positions、显式 frequency tensor；支持 interleaved/half-split 与未旋转 tail，dense route 冷选择 fat kernel |
| 概率归一化 | `Softmax` | f32/f16/bf16 任意 axis、scale、LogSoftmax，以及可广播 Bool/同类型 additive mask；F32 accumulation，半精度使用显式 F32 行工作区；允许精确 input/output in-place；无有效值的 row 产生 NaN |
| 候选筛选 | `TopK` | f32/f16/bf16 任意 axis，largest/smallest、sorted/stable 语义，返回同类型 values + i32 indices；显式 caller-owned workspace，不隐藏分配 |
| 注意力 | `ScaledDotProductAttention` | f32 rank-4 BSHD/BHSD，Q/K depth 与 V/output depth 可不同；可选 bottom-right causal、Bool/additive 广播 mask，分块 online softmax 与显式 caller-owned workspace |
| 自定义 CPU 工作 | `cpu::submit_host(queue, callable, args...)` | `noexcept` callable，返回 `void` 或 `Status`，返回前完成 |
| 可准备、可进入 Graph 的用户 kernel | `make_custom_provider` + `make_custom_kernel` | owning typed factory；shared HYLE Queue，逐 Invocation 状态；见下文完整例子 |
| CUDA runtime | `cuda::backend/device/MemoryResource/submit` | 显式 device/pinned/managed storage、stream borrow/adopt、typed command、真实 async Completion；不冒充 CUDA 算子目录 |
| byte transfer | `transfer_properties/copy_async/fill_async` | 对 Mem `BufferView` 做显式 byte copy/fill；不隐藏 tensor repack、pageable staging 或 host wait |
| 完成边界 | `Queue::reserve_completions` / `Completion` | direct submit 或一个 Graph Queue segment 只产生一个 prefix Completion；销毁 pending handle 不等待 |
| provider 选择 | `Context::create(device, Provider)` / `ContextOptions::providers` | 显式且有序；冷能力查询不支持或 prepare 明确安全拒绝为 unsupported 才尝试下一项，真正错误不按 code 猜成回退 |
| 精确资源 prepare | `OperationResourceBindings` / TensorView overload | provider 先报告要求，再对 input/output/workspace 的 exact `MemoryResourceRef` 冷验证；submit 不重新选资源或隐藏 copy |
| provider 诊断 | `provider_identity` / `runtime_properties` / `persistent_bytes` | O(1) 读取 provider/version、shared/delegated owner、replay 成本类别、threading/capture/workspace owner；`persistent_bytes == nullopt` 表示 provider 无法给出完整精确值，不是零 |
| 稳定重复执行 | `prepare -> PreparedOperation -> OperationInvocation::submit` | prepared artifact 可共享；每个并发 lane 独占 invocation/workspace；submit 不重新推 shape/路由 |

公共 schema 包括上表的普通张量运算、Scan 和 QuantizedMatMul，共 21 个 family，并能做纯描述推导。
Native 承载其中除 QuantizedMatMul 外的 20 个 family，具体 dtype/layout 域见上表。这不表示每种 dtype、
layout 或 encoding 都由同一个 provider 执行：当前 SDPA 是 f32 reference/optimized fallback；block/opaque
storage、量化 kernel 与窄 storage attention 仍应由理解相应格式的 provider 接管，不会被隐藏 materialize。

CUDA runtime 与 CUDA operation provider 是两层。选择 `HYLE_COMPUTE_CUDA=AUTO` 时，CMake 仅在当前目标能
找到用户安装的 CUDA Toolkit 时启用；`ON` 找不到即配置失败，`OFF` 不探测。HYLE 不下载 CUDA Toolkit。
启用后可以创建 CUDA Context/Queue、device/pinned/managed `MemoryResource`，借入或接管同 primary context
的非默认 stream，并通过 `cuda::submit()` 录入 typed stream command。Graph 可在冷路径 capture 兼容的
CUDA command/copy segment。只创建 backend Context 不会自动装入计算 Provider；当前独立接入状态见下表。
同一 CUDA Context 的不同 Queue 可以用 device event 导入另一个 Queue 的 Completion prefix，不阻塞提交
线程；跨 context/backend 不会伪装成这条路线，Graph 只能依显式策略选择 host wait 或拒绝。
同 Context 只表示事件可以互通，不表示不同 stream 天然有序。Graph 根据值依赖为跨 Queue 路线插入
必要的 device wait；direct 调用方仍须建立完成依赖，不能仅靠 host 端先后调用保证另一 stream 已完成。

CUDA `Buffer/View` 的 `write/fill/copy_from/read` 是同步存储操作：返回成功后本次操作已完成，可用于
初始化权重后立即在独立 Queue 上消费。它们不是异步传输 API；设备写/填充/复制会等待自身默认流的完成，
但不替你等待其他 non-blocking Queue 对同一 buffer 的在途访问。先显式完成冲突工作，再调用同步方法。
需要传输与计算重叠时，用 `copy_async`/`fill_async` 和 Queue dependency；不要在 warm kernel 或 capture
回调里用同步存储方法。pinned/managed 的直接 host 操作同样要求先结束与其冲突的 GPU 访问。

第三方源码出现在 dependency cache 或最终链接中，不表示其算子已经自动成为 HYLE route。HYLE
`Operation` 只描述公共语义；一个已交付 Provider 必须在 `prepare()`/Graph compile 冷路径把它翻译成
第三方 plan、kernel 或 graph artifact。叶函数型 provider 可以产生 `PreparedOperation`；自己 host 图的
runtime 可经 delegated Direct 或 Graph region 保留原生执行。当前没有已交付 ggml provider，合法的
block-Q4 `TensorDesc` 只证明格式可表达；未绑定支持该格式的 Provider 时仍返回 `Unimplemented`，
不会把压缩 bytes 当普通 scalar、偷偷反量化或因为 ggml 已下载就调用私有 kernel。

CUDA 的 Marlin 接入使用具名 compressed carrier，普通入口通过 `cuda::QuantizedWeights` 提供冷
packing 与所有权。它供 `QuantizedMatMul` 使用，不能推断普通 `MatMul` 有量化 fallback。
不同 Q4 格式、activation 临时量化
与累加类型不是同一件事。当前
`MatMul::accumulation_type` 不能解释为允许 Provider 静默把 F32 输入量化成 Q8。

### CUDA Provider 接入状态

以下是 [R4 当前施工卡](../implementation/tasks/compute/cuda-v1-four-route-foundation.md) 下已实现并有
Windows/SM89 验证的专家接入点。普通入口按前述规则自动组成它们，专家可显式限定 Provider。
Direct/Graph FFN、完整 CUDA 三交付与当前支持域 P1 已验收；单个 Provider 的可编译域仍不等于
所有 GPU/驱动组合都已资格化。

| factor / 公共头 | 工厂 / owner | 当前有限域 |
|---|---|---|
| `compute_provider_cublas` / `compute/providers/cublas.h` | `cublas::provider()` | FP16/FP32 rank-2 MatMul；输入 row/column-major、输出 dense row-major；FP32 累加；冷绑定算法，每槽独立 handle/workspace |
| `compute_provider_marlin` / `compute/providers/marlin.h` | `marlin::provider()`、`marlin::QuantizedWeights::pack()` | FP16 activation/output、signed symmetric group128 Q4、FP32 累加；M=1..64、K%128=0、N%64=0、N*K<=INT32_MAX；冷 packing 后只常驻 Q4/scales |
| `compute_provider_cuda` / `compute/providers/cuda.h` | `cuda_native::sglang_provider()` | 锁定 SGLang precise packed 门控片段，支持域与16字节对齐要求见默认规则；不引入 TVM/Python runtime |
| `compute_provider_cuda` / `compute/providers/cuda.h` | `cuda_native::provider()` | dense FP16/BF16/FP32 SiLU/GELU/GELU-tanh 门控，packed halves 或 separate 输入；FP32 中间计算、输出一次舍入；输出元素数<=INT32_MAX；不允许输入输出重叠 |

这些门控不需要 caller workspace 或额外 native invocation 存储；空 output 不要求 device payload
allocation，也不发 kernel。Kernel 与 launch 配置在 prepare
绑定，热提交仅在已有 CUDA stream 上 launch。共同层仍为每个 invocation 管理准入与真实完成。
CUDA driver/code image 的全局开销不假报为零；payload 必须保活到 Completion 终态。所有 Provider
失败执行槽都须按本页执行槽合同重建；设备失效仍受 backend 恢复限制。

### Grouped Q4 的逻辑描述

`QuantizedMatMul{.mode, .group_size, .accumulation_type}` 描述 `[...,K]` activation 与逻辑 `[N,K]`
压缩权重相乘，输出 `[...,N]`；activation rank 至少为 2。默认 group 为 64、float32 累加。
`kSignedSymmetric4` 有 activation、weight、scale 三输入，权重公式为 signed q（-8..7）乘 scale；
`kUnsignedAffine4` 另有 additive bias 输入，公式为 unsigned q（0..15）乘 scale 加 bias。
scale/bias 的逻辑形状都是 `[N,K/group]`，K 必须为正且整除 group。输入共享 float16、bfloat16、
float32 或 float64 逻辑 dtype，输出保留该 dtype；累加类型显式声明，不授权偷偷转换 activation。

普通 Graph 可直接 `builder.add(QuantizedMatMul{...}, input, weight, scales, biases)`；signed 形态
省去最后一项。权重的 `StorageFormatId` 仍须由具体路线的 packing factory 给出；仅凭 4 bit 无法
推断物理编码或可否无损换格式。当前 CUDA Marlin 只解释前述 group128 carrier；其它路线/格式需要
具备相应 Provider，公共描述和校验不等于有可执行实现。
已有 19 族 CPU kernel 不因此多出 Q4 fallback；数值资格也不保证与先物化权重的逐位结果相同。

## Queue、byte transfer 与 Completion

Compute 的 Queue 是一个显式、有序的提交域。普通 direct operation、byte transfer 和 backend-specific
typed command 都在该顺序中执行；一个公开 direct submit 只在边界生成一个 `Completion`。Graph 会把同一
Queue 上连续的多个 step 录成一个内部 segment，也只在 segment 末尾生成一次 backend 终态，用户不需要
管理内部 command list 或 event。

`copy_async()` 和 `fill_async()` 接收 Mem `BufferView`，适合模型 byte staging、renderer upload/download
和已经确定 layout 的设备数据移动。它们不解释 Tensor dtype/stride，不做 repack，也不会为了伪造异步而
偷偷分配 pinned staging。`transfer_properties()` 可在冷路径确认当前 source/destination route 是立即完成还是
直接排入 Queue；不支持的资源组合会明确失败。

```cpp
#include <hyle/compute/transfer.h>

auto properties = transfer_properties(queue, device_view, host_view);
if (!properties) return properties.error();

// 异步 backend 可在建立并发 lane 时预留 event slot；CPU 上这是 O(1) no-op。
if (auto status = queue.reserve_completions(concurrent_lanes); !status.ok())
  return status;

auto uploaded = copy_async(queue, device_view, host_view);
if (!uploaded) return uploaded.error();
if (auto status = uploaded->wait(); !status.ok()) return status;

auto cleared = fill_async(queue, scratch_view, std::byte{0});
if (!cleared) return cleared.error();
return cleared->wait();
```

copy 两端必须同尺寸；非完全相同的重叠 range 会被拒绝，完全相同的 range 和零 byte 是成功 no-op。
CPU backend 当前立即执行，GPU backend 可以返回 pending Completion。无论哪种 backend，source、destination、
kernel arguments 及其 resource owner 都必须活到 Completion 真正终态。丢弃 pending Completion 不会隐式
等待；需要回收 Queue/Context 所有已接纳工作时显式调用 `drain()`。`request_cancel()` 只在 backend 能兑现时
取消工作，不能把“停止接纳后续命令”冒充设备抢占。

`Gather`/`Scatter` 默认使用 `IndexBoundsPolicy::kChecked`，会在写 output 前先验证全部 indices，因此越界失败
不会留下部分索引写入。`kUnchecked` 只适合调用方已由上游验证 indices 的热路径；越界值违反前置条件，
不能把它当作另一种裁剪或环绕语义。Scatter 对重复 index 按 updates 的逻辑行主序处理：assign 后项覆盖
前项，add 依次累加。F16/BF16 的每次加法在 F32 求值后立即舍入回 storage，不是先将全部重复更新求和再舍入。
需要并发原子 scatter 的算法应选择具备该明确合同的 provider，不能把本 CPU reference
路线误当原子操作。

## View、Copy、Fill 与 Cast

`View` 是显式零 kernel operation：prepare 推导新的 descriptor，submit 只验证输出 Mem view 是否指向输入
allocation 的正确 byte offset。reshape、permute 和 broadcast 从输入起点开始；slice 输出应使用由
`start * input.byte_strides()[axis]` 得到的 subview。不要给 slice 随便绑定一个“同样大小且位于输入内”的
区间，provider 会拒绝它。空 view 不触碰任何 byte，因此不要求制造虚假的非空 alias。

`Copy` 是 material operation。dense 输入直接复用 Mem resource 的 range-copy 路线；plain strided（包括
zero-stride broadcast）输入按 logical coordinates materialize 为 dense 输出。它不接收 block/opaque
encoding，也不隐藏 repack 或 staging。`Fill` 没有输入，根据 `TensorShape`、`DataType` 和精确 typed
`ScalarValue` 写输出；低精度浮点用 `ScalarValue::from_bits()` 明确传 storage bits。

`Cast` 当前执行推理常用的五种浮点 storage：f32、IEEE f16、bf16、OCP E4M3FN 和 E5M2，任意两种都可互转；
同 dtype Cast 等价于 materializing Copy。窄化继承 SIMD 的 round-to-nearest-even、FP8 finite saturation 和
NaN mapping 合同。连续输入只做一次 whole-buffer SIMD 调用；plain strided 输入使用相同 scalar oracle
materialize，byte stride 无须满足 C++ 类型对齐。Cast 始终是 out-of-place；即使输入输出 dtype 等宽，也不能
绑定重叠 storage，避免 strided 输入覆盖尚未读取的数据。整数/Bool/f64 Cast 的溢出、NaN-to-int 和越界语义尚未
冻结，因此 Native provider 明确返回 `Unimplemented`，不会借 C++ `static_cast` 偶然定义公共行为。

这四个单输出 operation 提供会自动推导输出 descriptor 的 typed prepare，Fill 也有自然的零输入 submit：

```cpp
auto slice = View::slice(/* axis = */ 1, /* start = */ 4,
                         /* count = */ 8);
if (!slice) return slice.error();
auto prepared_view = prepare(context, *slice, input_desc);

auto prepared_copy = prepare(context, Copy{}, strided_desc);

auto prepared_cast =
    prepare(context, Cast{.output_type = DataType::kBFloat16}, f32_desc);

auto shape = TensorShape::create(std::array<std::int64_t, 2>{2, 8});
if (!shape) return shape.error();
auto prepared_fill = prepare(
    context, Fill{.output_shape = *shape,
                  .output_type = DataType::kFloat32,
                  .value = ScalarValue(0.0F)});
if (!prepared_fill) return prepared_fill.error();
auto invocation = prepared_fill->create_invocation();
if (!invocation) return invocation.error();
auto done = invocation->submit(queue, output_view);
```

## Elementwise 与广播

`Unary`、`Binary` 与 `Select` 的 typed `prepare()` 都会推导 dense 输出，不要求调用者重复写 output descriptor。
Native Unary/Binary 支持 f32/f16/bf16，半精度在 F32 计算后一次 RNE 舍入回输入类型；Select 不做数值转换，因此可选择全部已登记的普通 1/2/4/8-byte
scalar storage。Binary 和 Select 使用 NumPy-style 尾轴对齐广播，输入也可携带普通 byte stride；输出始终
materialize 为 dense storage。

```cpp
auto activated = prepare(context, Unary{UnaryKind::kSilu}, logits_desc);
auto residual = prepare(context, Binary{BinaryKind::kAdd}, hidden_desc,
                        residual_desc);
auto chosen = prepare(context, Select{}, condition_desc, true_desc,
                      false_desc);
```

dense、等形路线可让 output 精确 alias 任一由 operation traits 允许的输入；provider 此时只建立一个
read-write mapping。广播或 strided 输入不能 alias 输出，因为逐坐标 materialization 可能覆盖尚未读取的
元素。dense f32 `Rsqrt`、`Minimum/Maximum` 与 f32 `Select` 的完整连续 row 直接复用 SIMD；partial
overlap 在通用 runtime 更早被拒绝。`Minimum/Maximum` 使用 `std::fmin/std::fmax` 语义：单侧
NaN 选择另一侧数值，并保留 IEEE signed-zero 规则；它们不是由某条 ISA 的偶然 min/max 行为定义。

## Reduction 与 LayerNorm

`Scan{.axis = -1, .exclusive = true}` 把 counts 转为前缀 offsets；默认 inclusive 也可计算 F32
gate 前缀。输出保持输入 shape/type 并 materialize 为 dense。I32 包括最后总和在内的全部前缀都须
可表示，任一溢出以 `OutOfRange` 完成且不写 output；F32 允许固定执行路线的重结合。Native dense
连续 row 复用 SIMD，其余按逻辑 axis 遍历；没有隐藏 heap scratch。

`Reduce` 的 typed `prepare()` 根据 axis 和 `keep_dimensions` 推导 dense output。当前 Native route 支持
f32/f16/bf16 Sum/Mean/Minimum/Maximum；半精度必须显式设置 `accumulation_type = DataType::kFloat32`，
在 F32 累加并仅在最终输出时舍入。任意普通 byte-strided input 都按逻辑坐标读取，dense reduce-all 和最后一轴
四种 operation 都走已有 SIMD reduction。空集合约定为 Sum=0、Mean=NaN、Minimum=+infinity、Maximum=-infinity；
Min/Max 遇到任一 NaN 会传播 NaN。Reduce 不允许 output alias input。

`LayerNorm` 对最后一轴逐 row 计算，可不带参数、只带 scale，或同时带 scale+bias；三种 typed prepare 都
自动推导与 input 同形的 dense output。output 可以精确 alias input，但不能 alias scale/bias。Native 的
RMSNorm/LayerNorm 支持 dense f32/f16/bf16 与 F32 accumulation。半精度输入的 scale/bias 可以各自使用
输入类型或 F32，计算保持 F32，affine 完成后一次 RNE 舍入回输入类型。固定大小栈上分块不会展开整张量。
Gemma 的 `1+gamma` 应通过普通 Binary/常量明确组成。

```cpp
auto reduced = prepare(context,
                       Reduce{.kind = ReductionKind::kMean,
                              .axes = Axes::all(),
                              .accumulation_type = DataType::kFloat32},
                       input_desc);
auto normalized = prepare(context, LayerNorm{.epsilon = 1.0e-5F},
                          input_desc, scale_desc, bias_desc);
```

Native 半精度 `Softmax`/`TopK` 的 workspace 按一行 F32 临时值复用；TopK 另有候选索引空间。
调用 `workspace_requirement()` 取得大小/对齐，并在 `create_invocation(workspace)` 时绑定；Graph
按同一要求管理每个 invocation 的独立工作区。空 output 的需求为零。数据和 mask 内容可在完成后更新，
重复执行不重新准备或重新选择 Provider。

## 普通、block 与 opaque storage

普通用户继续用 `TensorDesc::dense()`；它现在返回 byte stride，例如 dense f32 `[2,3]` 的 stride 是
`[12,4]`。非连续普通 storage 使用名称明确的 `strided_bytes()`。旧的 `strided()` /
`element_strides()` 已删除，没有把旧名称悄悄改成新单位的兼容层。

固定块压缩权重由 provider 的 typed factory 创建 `StorageFormat`，再传给 `blocked_dense()`。下面只是说明
通用 seam；产品代码应使用 provider 发布的具名 identity，而不是散落 magic numbers：

```cpp
constexpr StorageFormatId codec_id{vendor_namespace, format_value};
auto format = StorageFormat::blocked(codec_id,
                                     /* elements_per_block = */ 32,
                                     /* bytes_per_block = */ 18,
                                     /* alignment = */ 2);
if (!format) return format.error();

auto stored = TensorDesc::blocked_dense(
    DataType::kFloat32, std::array{n, k}, *format);
if (!stored) return stored.error();
auto logical_weight = stored->transposed();  // logical [k,n]，不搬数据
```

`DataType::kFloat32` 在这里表示解码后的数学 element，不表示每个 element 占四个物理 bytes。format 独立
给出 block elements、block bytes、alignment 与稳定的两字 identity。block axis 的 byte stride 每次推进
一个完整 block；其他轴推进一个逻辑坐标。该轴长度必须能整除 block size；`required_bytes()` 和
`transposed()/permuted()` 会一起处理 block axis。

不能由通用 Compute 解释的 vendor-packed、tile 或 sparse descriptor 使用 `StorageFormat::opaque()` 与
`TensorDesc::opaque()`，并显式给出精确 byte extent。Opaque tensor 没有可公开读取的 stride，也拒绝通用
`permuted()`；只有理解该 identity 的 typed provider 才能构造另一个被证明等价的 view。scale、zero、bias、
group size 和量化公式不是 storage format 属性，它们是具体 operation/provider 的独立 typed inputs。

`TensorView::create()` 会把传入 Mem view 收窄到 `required_bytes()`，并拒绝明显不满足格式 alignment 的
subview offset。普通 host route 映射后还会检查实际地址；设备 native handle 的最终 alignment 必须由能
解释它的 provider 检查。TensorDesc 不拥有 storage，不保存字符串、vtable、动态内存或进程全局 registry。
合法但当前 route 不理解的 encoding 返回 `Unimplemented`，从而允许有序 provider fallback；格式参数本身
畸形则在构造时返回 `InvalidArgument`/`OutOfRange`。

## MatMul 权重形状

不要增加 provider-specific `transpose_weight` 标志。若模型存储权重为连续 `[...,N,K]`，创建该 dense
desc 后调用 `transposed()`；它只交换末两轴并保留领先 batch axes，得到逻辑 `[...,K,N]` 和对应 byte
stride，数据不移动。Compute 会在 prepare 冷路径选择相应 route。普通连续 `[...,K,N]` 也直接受支持。
Block-compressed `[...,N,K]` 使用上一节相同的 transpose 方式；
当前 HYLE-native 与 oneDNN 不解释它，只有以后显式加入的量化 provider 会接收。

```cpp
auto stored = TensorDesc::dense(DataType::kFloat32, std::array{n, k});
if (!stored) return stored.error();
auto logical_weight = stored->transposed();
if (!logical_weight) return logical_weight.error();

auto prepared = prepare(context, MatMul{}, activation_desc, *logical_weight);
if (!prepared) return prepared.error();
auto output = prepared->output_description();
if (!output) return output.error();
```

## prepare、invocation 与 submit

`infer_output_descriptions()` 是 direct Compute 与 Graph 共用的 shape/layout 规则；单输出 operation 可用
`infer_single_output()`。通用 span 入口的 `prepare()` 再核对调用方给出的 output desc；19 个 family 的 typed
convenience 都按真实 arity 推导 output desc，其中 TopK 推导 values 与 i32 indices 两个输出，Concat 保留
真正 variadic 的 descriptor span。两者都把 provider/route 选择移出热路径。对同一描述反复执行时共享
`PreparedOperation`，为每个并发执行 lane 创建并复用一个 move-only `OperationInvocation`。当前故意
没有把 prepare 或 invocation 创建隐藏在一次性 convenience 中，因为 provider workspace 和并发状态必须
有明确 owner。

固定 arity submit 可直接传 0--4 个 input 与单 output；TopK 可直接传 input、values、indices，Concat
则传 input-view span 与单 output。这些 convenience 只在栈上组成有界 `std::array/std::span`，不分配，
随后进入同一通用 submit。极小 operation 的手写热循环若已经持有 descriptor/view arrays，可以直接使用
span 入口避免重复复制；Graph 会在编译/Invocation 创建时预建 binding，不在每个 node 重新组成数组。

只给 descriptor 的 typed convenience 默认绑定 `mem::system_resource()`，适合 CPU/host RAM 的普通路线。
使用 device、pinned、managed、NUMA 或应用自有 resource 时，不要让 provider 在第一次 submit 时猜测：把
同一个 `MemoryResourceRef` 作为 typed prepare 的最后一个参数；多资源 operation 使用
`OperationResourceBindings`；已经有实际 view 时，可直接调用接收 input/output TensorView spans 的 overload。
prepared artifact 会保存每个 binding 的 exact resource identity，之后换成“属性相同但不是同一 resource”
的 view 会在进入 provider 前返回 `InvalidArgument`。

```cpp
const std::array inputs{input_view};
const std::array outputs{output_view};
auto prepared = prepare(context, Operation(RmsNorm{.epsilon = 1.0e-5F}),
                        inputs, outputs);

// descriptor-only GPU/custom-resource route：显式给出 exact resource。
auto device_prepared = prepare(context, RmsNorm{.epsilon = 1.0e-5F},
                               input_desc, weight_desc, device_resource);
```

`runtime_properties()` 说明 artifact 是 `kSharedQueue` 还是 `kDelegated`，以及 workspace owner、host blocking、
私有线程、并发 invocation 和 capture 事实。公开 direct `prepare()` 只选择 `kSharedQueue` provider，因此其
invocation 总能提交到调用方 Queue。`kDelegated` artifact 只由 Graph 的内部精确路线创建并作为完整 island
提交；Compute 不会把一个无法 direct submit 的对象交给普通用户。caller-owned workspace 使用 prepare 时
固定的 resource；provider-owned workspace 则拒绝调用方传入 BufferView。

`runtime_properties().replay_class` 独立说明 Provider 成功重放的成本合同：

- `ProviderReplayClass::kPrepared`：已经准备命令，不在 Provider replay 内推 shape、建图、分配 heap、JIT
  或打包常量。当前 Native 与 oneDNN 路线显式声明这一类别。
- `kManaged`：已编译程序仍可能实例化执行 metadata/array、做已声明的内部 cache/shape 工作；这些成本
  必须算入完整调用，不得隐含新编译、copy/sync 或改变已选路线。
- `kUnknown`：未声明，**不是零成本**。`valid()` 只检查字段是否合法，不证明成本已资格化。

分类不能从 shared/delegated 或 capture flag 推断。调用方自带 executor/callback 的内部开销、失败时
Status 诊断分配仍由各自合同负责。direct 用户可在准备后、提交前检查该字段；Graph 用户可通过
[`CompileOptions::provider_replay`](graph.md#何时要求-prepared-only) 在创建 artifact 前筛选 leaf/region。
这不改变 direct `prepare()` 的 Provider 顺序，也不增加每次 submit 的政策查询。

Provider 的冷提案用 `ProviderRouteCostFacts` 声明结构成本，Graph region 也复用这个类型；workspace
大小/对齐、persistent/copy/repack bytes 与 synchronization/dispatch count 都区分 unknown 和 exact zero。
它不是时间估值或性能资格证：`valid()` 只检查对齐和非零 dispatch 等结构合法性，不证明某路线快于另一条。
普通 direct 用户仍使用 `prepare()`，不需要自己构造成本提案。

Provider 实现者使用共同的 `ProviderPrepareOutcome` 声明 prepared、unsupported decline、candidate-local
resource-budget decline 或 error；`valid()` 只检查 kind/status 合法性，安全拒绝还必须保证未接受用户工作、
没有外部副作用，且返回的部分 artifact 可由共同 owner 销毁。真正 error 即使带 Unimplemented 或
ResourceExhausted code 也不会被跳过。普通 direct 调用仍返回 `Result<PreparedOperation>`，只自动跳过
明确 unsupported；资源预算拒绝直接报告，不自行做 Graph 式资源重选。Graph 可以在有限预算内对两种
安全拒绝作回退。`submit()` 之后的失败属于 Completion，不能用这个冷编译结果重试已接受的工作。

需要区分同一家 Provider 的多个配置实例时，使用
`Context::provider_instance_id(index)`；`PreparedOperation` 和 `OperationInvocation` 的同名 accessor
返回实际选中的实例。`provider()` / `provider_identity()` 描述逻辑家族，不能区分同家族的多个 slot。
`ProviderInstanceId` 在 Context 移动或关闭后保持不变，重新创建 Context/slot 得到新值；默认对象返回
无效 ID。这个值不保活资源，也不证明资源仍可接受工作；只在同一个已加载 Compute runtime 的生命周期
内比较，不能持久化或跨进程使用，更不能单独作为合格性能 profile 的完整身份。

图编译器或手工 workspace planner 可在冷路径调用
`PreparedOperation::alias_support(output_index, input_index)`。公共 operation traits 只是语义允许的上界；
该查询给出当前 provider、shape 和 layout 真正支持的输入输出 alias 及相对 byte offset。普通调用者无需为
每次 submit 查询它，runtime 仍会在接收工作前验证实际绑定。

```cpp
auto context = cpu::create_context();
if (!context) return context.error();
auto queue = context->create_queue();
if (!queue) return queue.error();

auto prepared = prepare(*context, RmsNorm{.epsilon = 1.0e-5F},
                        input_desc, weight_desc);
if (!prepared) return prepared.error();
auto invocation = prepared->create_invocation();
if (!invocation) return invocation.error();
auto done = invocation->submit(*queue, input_view, weight_view, output_view);
if (!done) return done.error();       // admission failure
return done->wait();                  // terminal failure
```

## 准备时绑定实现

普通计算不需要用户提供 timing profile。四路线的默认优选规则由 HYLE 研发阶段验证并固定；
准备时匹配适用条件和资源约束，执行时复用已绑定制品，不计时、不重选。各路线当前能力以其
公开入口及验收记录为准；本节不把旧 Provider 顺序当作已完成的四路线优选规则。

`ProviderRouteKey` 仍标识 Provider 内部实现，供诊断报告使用；单个 key 不证明性能资格。
旧 `RouteProfile`、matcher 和 Graph 多计划性能竞赛接口已经移除。

## CPU 默认规则与显式实现约束

当前 oneDNN provider 只接收 **plain** dense row-major f32 rank-2 MatMul。它适合普通
`[M,K] * [K,N]`；模型常见的
物理 `[N,K]`、逻辑转置为 `[K,N]` 的连续-K 权重会诚实回落到 HYLE-native SIMD route。RMSNorm、
全部 GatedActivation、batched/零尺寸 MatMul 和其他未支持形状也走 native，不会在 submit 时重新选择。在默认串行
executor 下，`M < 16` 的 decode MatMul 同样选择结构更轻的 HYLE-native kernel；传入真实并行 executor
时，oneDNN 可以把 N 维拆成互不重叠的连续 tile。

```cpp
#include <hyle/compute/providers/onednn.h>

auto context = Context::create(cpu::device(), onednn::provider());
if (!context) return context.error();
auto prepared = prepare(*context, MatMul{}, a_desc, b_desc, c_desc);
if (!prepared) return prepared.error();

const auto need = prepared->workspace_requirement();
hyle::mem::Buffer scratch;
if (need.size != 0) {
  auto allocated = hyle::mem::Buffer::create(
      hyle::mem::system_resource(), need.size, need.alignment);
  if (!allocated) return allocated.error();
  scratch = std::move(*allocated);
}
auto invocation = prepared->create_invocation(scratch.view());
if (!invocation) return invocation.error();
```

上例是专家限定 oneDNN 的用法，不支持的形状返回 `Unimplemented`。普通使用选择 CPU 默认入口：

```cpp
#include <hyle/compute/cpu.h>

auto context = hyle::compute::cpu::create_context();
```

`Context::create()` 在冷路径把完整 provider 序列复制到 context 自己的一次 metadata allocation；输入
span 只需活到调用返回。序列没有人为四项上限，也不会自动追加 native。通过
`PreparedOperation::provider()` 返回的开放 `ProviderIdentity` 可以检查实际路线；是否发生组合回落由
所选默认规则和最终 identity 解释。确有实现约束时，专家 ContextOptions 仍可显式提供候选序列。

默认 Native 和 oneDNN executor 都是调用线程串行执行；HYLE 不偷偷创建线程池。已有同步线程池的程序可通过
`cpu::DefaultContextOptions::executor`、`native::Options::executor` 或 `onednn::Options::executor` 传入借用的
`ParallelExecutorRef`。其 `parallel_for` 必须在返回前完成全部任务、不得失败或抛异常，并在嵌套调用时
保证进展；executor backing 必须活过由 provider 创建的 Context 及其 retained artifacts。Native 在 prepare
冷路径按总工作量固定 task count，目前为足够大的 Norm、MatMul、dense gated activation 与 SDPA row 做
至多一次同步 fan-out；默认、小工作量与可检测的 nested episode 保持 caller-thread 串行。

并行分片由 HYLE 在 prepare 冷路径决定，不是 oneDNN 的隐藏线程池：正常 tile 宽度至少为 64 columns，
任务数不超过 executor concurrency，并且至多生成 base/base+1 两种预编译 kernel。每个任务使用独占、
64-byte-aligned scratch slice；调用方只需按 `workspace_requirement()` 分配一块总 workspace，不应自行
推导 tile 或依赖该内部阈值。

标准 CPU 配置的该选项默认 `AUTO`：支持的 x86-64 目标接入 oneDNN，AArch64 使用 Native。构建时
`HYLE_COMPUTE_ONEDNN=ON` 在受支持的 x86-64 目标强制启用，在 AArch64 等不支持目标会明确配置失败；
`AUTO` 在不支持目标关闭该实现。可复现的 x86-64 oneDNN 制品可显式选择 `ON`，不需要
它的制品保持 `OFF`。`PreparedOperation::provider()` 与 `workspace_requirement()` 可解释实际路线。
oneDNN 类型不会泄漏到 Tensor、Operation 或 Graph API。

## 所有权、失败和限制

- `Backend`、`Device` 和 `Provider` 是可复制的进程内值；它们以开放的双字 ID 标识实现，不要求
  修改中央枚举。内置 Provider 配置 inline；custom Provider 复制时保留其 owning factory（冷引用计数），
  identity 的 name/version view 借用这个 owner。`Context`、`Queue`、`OperationInvocation` 与 `Completion` 是 move-only；
  `PreparedOperation` 可复制共享。
- `Context::close()` 与 `Queue::close()` 原子地停止新 admission；`drain()` 显式等待已接纳工作。析构只
  close/release，不把等待藏进析构路径。CPU completion 当前立即终态；异步 backend 可返回真正的
  poll/wait/cancel completion。
- Context metadata allocator 必须活过从它保留的 handles，以及由该 Context 编译出的 Graph
  Executable/Invocation。
- `PreparedOperation` 可复制并跨线程共享；`OperationInvocation` 不可复制，也不得被两个并发 submit 共用。
  `in_flight()` 可观察该 lane 是否仍被一次提交占用，重叠 submit 明确失败。Completion 会保活 provider
  invocation，因此丢弃外层 `OperationInvocation` handle 不会提前销毁异步状态；Context 关闭后 retained
  artifact 仍可创建 invocation，但任何 submit 都会失败。
- `prepare` 选择 shared-Queue provider；`prepare_delegated` 在冷路径绑定独立 runtime，返回专用的
  `PreparedDelegatedOperation` / `DelegatedInvocation`，其 `submit(inputs, outputs)` 不接收 Queue。
  两种类型在准备时区分，执行不重新搜索 Provider。独立 runtime 仍须提供真实 dependency/Completion/drain 合同。
- Tensor views 不拥有 payload。Mem Buffer/resource、调用方 workspace 以及 Queue 必须活到 Completion 终态。
  direct op 和 Graph 都把自己的 invocation/provider state owner 附着到最终 Completion；这种内部附件不改变
  caller tensor、workspace 或 Queue 的借用合同。
- `StorageFormatId` 是跨 source/static/shared 稳定的数值 identity，不是 Native FFI 承诺。自定义 provider
  应以自己的 typed factory 统一发布 identity；同一 identity 不得在升级后改变物理解释。
- typed custom `Operation` 的 `OperationId` 同样标识完整 schema/version 和 inline descriptor layout；改变字段、
  大小或语义必须换 ID。`custom_id()` 可作冷路径分派/诊断，`custom_as<T>()` 以该 ID 和精确 descriptor
  字节数校验，因此 producer 与 consumer 可以位于不同 source/static/shared factor；不要复用 ID 模拟兼容。
  这只是 schema 描述/识别，不会自动提供执行实现；通过下面的 owning factory 单独提供执行。
  不要通过包含 `src/` 私有 Provider SPI 模拟公开扩展。
- admission 前的参数、shape、alias、context/queue mismatch 由外层 `Result` 报告；接纳后的 mapping、coherence
  等失败由 Completion terminal `Status` 报告，两层都要检查。
- 已接纳工作的 Completion 若以失败（含实际取消）终结，该 OperationInvocation/DelegatedInvocation
  退役。此时 `in_flight()` 为 false，但再次 submit 会失败；从原 Prepared 对象重新创建槽才能重新提交。
  未接纳工作的拒绝仍可修正参数后重试。设备或 Provider 自身失效可能还需重建对应 owner；重建槽不修复
  设备故障，也不缩短已提交 payload/workspace 的借用寿命。
- `DelegatedInvocationOptions::dependency_capacity` 在冷创建时预留依赖 scratch；零默认 input_count+1。
  `submit_after(span<const Completion* const>, inputs, outputs)` 校验容量及依赖。源 Completion 与输入/输出
  storage 必须活过消费完成，期间不能移动、销毁或 settle 源句柄的 native state；不支持的 native 依赖在接纳前拒绝，
  不暗中 host wait。若 runtime 不支持并行 invocation，调用者须将同一 Prepared 的执行串行到真实终态。
- `OwnedResult<T>` 持有原生 owning T 和真实 Completion；`wait()` 成功返回可复制的 `ResultLease<T>`，
  `result()` 只 poll，pending 明确返回 FailedPrecondition，失败/取消不产生 ready lease。保留 lease 后，
  原提交对象和生产槽可销毁；同槽仍有结果引用时不能覆盖，另一槽可继续使用。T 的 runtime 决定如何读其数据，
  HYLE 不把它伪装为 Mem allocation，也不隐式创建 host shadow copy。注入 allocator 必须活过所有结果引用。
  这些共同接口不描述 MLX/Metal 或 Vulkan 的原生 Program；其默认规则和验收范围见对应路线指南。
- RMSNorm 允许 input/output 精确同址；所有 operation 拒绝不安全 partial overlap 和第二输入/output
  overlap。CPU route 要求 host-accessible storage，并如实调用 non-coherent resource 的 invalidate/flush。
- 不同 Queue 可并发；同一 ordered Queue 是单 producer 提交域，两个线程同时录入会明确失败。Graph 会用
  一个不中断的内部 sequence 连续录入整份 plan，防止其他 producer 插入节点之间。成功的 provider
  Completion 必须代表截至本命令的完整 Queue prefix；同步 `Result` 错误则表示当前命令未接纳任何工作。
  调用方仍必须保证同一个 `OperationInvocation` 不被并发 submit，并让 payload/workspace 活到 Completion
  终态。
- 销毁 pending Completion 从不等待。异步 backend 必须让已接纳工作继续安全存活，并在真实终态后释放
  随 Completion 交接的内部 execution owner；不能把“用户没 wait”解释成工作已取消或 borrow 可回收。
- HYLE-native route 不隐藏 staging、copy、线程切换、packing 或 JIT。oneDNN BRGEMM handle 只在 prepare
  冷路径 finalize/generate；每个 Invocation 映射调用方 workspace，steady submit 不创建 descriptor、
  stream、memory handle、JIT artifact，不 packing/copy，也不执行 route search。当前闭包的六个推理形状
  曾证明 warm `submit()+wait()` 无 C++ 动态分配；这些是既有 exact 路线的历史证据，当前 foundation
  proposal/Graph 迁移仍由 registry 指向的活动卡验收，不能把它们推广为新合同或 MLX 已通过。
- oneDNN ukernel C API 在锁定版本中仍标记 experimental。它完全私有于 provider，并由精确版本断言保护；
  用户不应直接依赖其 header 或 ABI。升级 oneDNN 必须重做源码、符号、分配和正确性审计。

source、headers + static、headers + shared 都选择 `hyle::compute`。三者是同一 C++23 source contract，不是
Native FFI，也不承诺跨工具链 C++ ABI。

## 自定义执行：接入 direct 与 Graph

只执行一次 CPU callable 用 `cpu::submit_host`；需要冷准备、重复执行或放进 Graph 的 kernel 使用
`<hyle/compute/custom_kernel.h>`。可编译的完整路线见
[owning custom kernel 示例](../../samples/package/graph/custom_kernel/README.md)，其中 factory 在独立 shared library
内，消费方同时走 direct 与 Graph，不包含 `src/` 或手写函数表。

- `make_custom_provider(identity, backend_id, factory, allocator)` 拥有 factory，可 move-only；复制 name/version
  文本。失败不移动具名 factory。Context 保留其配置，直到全部派生执行对象释放。allocator owner 和用户
  实现所在 code image 仍必须存活；这不是支持任意 unload 的动态 plugin 系统。
- `const factory.propose(KernelRequest, input_requirements, output_requirements)` 返回 `Result<KernelProposal>`，
  只声明精确 shape 的支持、资源、runtime 和结构成本，不编译、不执行 payload；`ResourceRequirement`
  可指定属性、alignment、精确 resource/domain，未填写字段不施加额外约束。所有 span 只借到调用返回。
  成功查询不分配：使用调用方的输出 span 与有界栈 scratch；错误 Status 的诊断存储不属此成功合同。
  按算法真正需要的属性声明；静态链接到不同 code image 的 `system_resource()` 可能不是同一个实例，
  不能用它代替通用 host-accessible 要求。确需精确实例时从实际资源 owner 传入句柄。
- `KernelProposal::route` 是本次 propose→prepare 的临时 token；可选 `profile_key = ProviderRouteKey{1}`
  则是同一 build/configuration 下的稳定实现键，不得填指针、发现序号或随机值。默认 unknown 不影响
  正常执行。键仅是未来完整性能资料身份的一部分，不能代替 build、设备/驱动、语义、shape/layout、资源、
  workload 与完成边界；当前 Graph 只把它保留到 step report，不据此改顺序或声称自动择优。
- `const factory.prepare(KernelPrepareRequest, allocator)` 仅在选中路线之后创建 kernel，返回
  `KernelPreparation`。成功用 `make_custom_kernel(value, properties, allocator)`；明确 unsupported /
  resource-decline 才能授权对应政策回退，真正 error 即使错误码相同也不自动重跑。部分 kernel 由 RAII 回收。
- kernel 不可变，定义自己的 `Kernel::Invocation` 和 `create_invocation(workspace, allocator)`；每 lane
  一份独占可变状态，不能同时复用同一个 Invocation。可选 `alias_support(output, input)` 只能细化 schema
  已允许的 alias；不定义时拒绝 alias。factory 与 kernel 的 const 方法必须支持其声明的并发方式，全部
  callback 和 move/destructor 为 `noexcept`。
  prepare 与 create-invocation 收到的 allocator **可能不同**：Graph 可将编译制品与运行实例记到不同
  metadata owner。为本次实例分配的内存应使用本次参数并记录配对释放信息，不要固定回用 factory/
  kernel 创建时的 allocator。allocator 是借用，owner 须活到实例及其异步 Completion 完全释放；公开
  direct prepare/create 默认仍继承 Context 的 metadata allocator。allocator 本身不自动计 payload/workspace；
  Graph 的可选运行预算另行预留 workspace，并与 metadata 共享额度。
- `KernelProperties::invocation_private_bytes` / `PreparedOperation::invocation_private_bytes()` 是额外的
  每实例私有存储上界，排除 callback allocator 和 caller workspace 中的字节，覆盖创建、运行和销毁。
  未声明是 unknown；全部使用所供 allocator/workspace 时可明确写 0。不要把 prepared persistent 字节
  填到这里。Graph 的有限完整预算默认拒绝 unknown；普通 direct 用法不要求提供该事实。
  若 artifact 声明 `supports_concurrent_invocations=false`，direct 用户须将同一 PreparedOperation 的全部
  Invocation 串行到真实 Completion 终态，不能只等 enqueue 返回；direct 不额外加共享锁。Graph 会在同一
  Executable 内检查该限制；用户跨 Executable 借用的外部不可并发资源仍须自行同步。
- `KernelProposal::cost.compilation_storage` 声明所供 compilation allocator **以外**的私有存储上界。
  `resident_bytes` 覆盖制品保留到最终销毁（含 teardown scratch），`peak_bytes` 另覆盖 prepare/部分失败
  清理，且必须不小于 resident。全部编译内存走 callback allocator 时写 `{0, 0}`；不知道则保持 unknown。
  不能拿可能包含 allocator metadata 的 `persistent_bytes` 直接填进去，也不能等 prepare 后才给峰值。
  Graph 的 `compilation_storage` 配额在准备前预留这些字节；普通 direct 调用没有自动全局编译配额。
- `kernel.enqueue(invocation, queue, inputs, outputs)` 返回 `KernelCommandResult`，在已 admission 的 HYLE
  Queue 内执行/记录。不要再调用 `submit_host/cuda::submit` 嵌套 admission，也不要创建第二个 Completion。
  未接纳工作可 rejected；已经写入输出或提交任意命令后必须 accepted，即使随后报错，最终由 Completion
  覆盖已接纳的前缀。未知接纳状态应保守保活，而不是猜测可重跑。native handle 的 context/stream 限制仍适用。

本接口承载 shared HYLE Queue + caller workspace 的 kernel，不冒充 delegated runtime/region adapter。
配置、typed kernel、typed Invocation 各有冷分配；内置 Provider 仍用 inline 配置。HYLE 的 warm 调用
直接进入 typed trampoline，没有额外虚函数层；用户 kernel 的分配、同步和线程必须符合自己声明的
prepared/managed 成本。此 factory 当前为 exact-point；Windows CPU/CUDA 示例已验证三种交付，其中
CUDA 使用真实 native kernel、双 Queue 和 exact Graph capture。此 factory 的 exact-point 限制仍在；
Apple/Vulkan 使用各自原生 Program 与 custom lowering，见路线指南。四路线的正确性、三交付、
sanitizer 和有限性能证据各有适用范围，不能把 shared-Queue factory 外推为所有 native runtime 的接口。

## Custom operation 的外部资源访问

普通 tensor input/output 的数据流不需要另加资源端口。只有 custom operation 还观察或修改外部可变
状态时，descriptor 才使用 `EffectKind::kResourceAccess` 和非零 `resource_count`，逐端口声明
`ResourceAccess::kRead/kWrite/kReadWrite`。例如一个操作读取配置状态、更新另一个累积状态，其 descriptor
可包含下面两个成员（identity 与 output inference 仍按原 custom schema 提供）：

```cpp
compute::OperationTraits operation_traits() const noexcept {
  return {.inputs = {1, 1}, .outputs = 1,
          .effect = compute::EffectKind::kResourceAccess,
          .resource_count = 2};
}
compute::ResourceAccess resource_access(std::size_t port) const noexcept {
  return port == 0 ? compute::ResourceAccess::kRead
                   : compute::ResourceAccess::kReadWrite;
}
```

端口是 schema 中的有序语义位置，不是 tensor input 序号、Graph ID、地址或资源 owner；最多 255 个。
`operation_resource_access(op, index)` 返回按值结果，`validate_operation_effects(op)` 检查完整访问 schema；
inference/prepare 也会检查，非法枚举、缺方法、空资源声明或混用 pure 都会报错。声明方法必须为
`const noexcept`，确定、无分配且不读取 payload/执行工作；结果不能借用被临时解码 descriptor 的内存。
跨 shared 调用仍走原 producer 的 callback，承载代码的库必须保持加载，不提供卸载中的 plugin 支持。

只读外部状态不是 pure。无法列出完整访问集时，保留 `kExternalSideEffect`、零具名端口；它意味着保守
副作用，不是空资源集。direct Compute 的外部资源顺序/生命周期由调用方 Queue/Completion 管理。
Graph 使用 graph-local `EffectResourceId`、固定 backing footprint 与显式 data/control 依赖消费这些端口：
同域冲突有序，read/read 或独立域不加边，未知副作用保守排序。具体绑定、alias 与跨 Invocation 限制见
[Graph 外部资源指南](graph.md#外部可变资源声明访问与顺序)。schema 或资源绑定都不会自动安装 custom execution 实现。
