# HYLE Compute：何时直接执行一个计算

> 本页描述当前 Compute runtime foundation、CPU operation surface 与 CUDA runtime；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。普通 CPU 入口自动组装已构建的 oneDNN/ggml/Native 实现；
> 标准 CPU 构建使用 `HYLE_COMPUTE_ONEDNN=AUTO`，在支持的 x86-64 目标接入已支持的 oneDNN 能力。
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

## 可选模型与阶段信息

已知模型身份的消费者可在冷准备时传入 `WorkloadContext`。普通工程计算省略它即可。
HYLE 不加载模型文件；`family`、`revision` 和 `configuration` 是调用方声明的独立身份，
配置字段应覆盖可能影响计算规则的选项，例如消费方生成的完整配置摘要。

```cpp
auto identity = hyle::compute::ModelIdentity::create(
    {"my_model_family", "local-artifact-revision", "configuration-digest"});
if (!identity) return identity.error();
hyle::compute::WorkloadContext workload{
    .model = *identity,
    .phase = hyle::compute::WorkloadPhase::kDecode,
};
auto program = hyle::compute::prepare(
    context, hyle::compute::MatMul{}, left_desc, right_desc, resource, workload);
```

身份创建时一次拷贝拥有字符串，源字符串随后可销毁；多个程序共享同一个不可变身份。
自定义 allocator 的 backing 必须活到最后一个身份/程序释放。`program.workload()` 返回
保有所有权的冷诊断副本。阶段或版本变化需要重新准备，不能给运行中的程序换标签。
可选的 `region` 是带版本的片段语义标识，不替代实际 operation/DAG、精度、资源和状态检查。

固定的模型专用规则必须完整匹配身份及其余适用条件；缺失/未知信息只允许合法通用路径，
或返回明确拒绝。传入模型名本身不代表已有该模型专用优化。用户无需排列 Provider 或提供计时资料；
HYLE 的规则匹配在运行期不读取模型字符串、查询设备版本或重选实现；成熟 Provider 自身的
managed 派发成本另外声明。

CUDA 的 `cuda::environment(context)` 返回 Context 创建时采集的设备名、UUID、SM major/minor、
驱动支持的 CUDA API 版本、runtime 版本与编译 Toolkit 版本。`driver_cuda_version` 对应
`cuDriverGetVersion`，**不是 NVIDIA 驱动安装包版本号**。固定规则只能匹配实际具备的事实；
不能把该字段当作驱动补丁版本。Apple/Vulkan 使用各自原生 `Environment`，Provider 版本来自
对应实现/锁定依赖；它们不共用一个万能设备结构。

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
| FlashInfer causal GQA | Linux CUDA SM89；dense BSHD FP16/BF16，Q/KV heads=8/2、Q/K/V depth=256，正 batch/sequence、KV length≥Q length，无额外 mask，FP32 累加；head256 prefill/decode 使用锁定原版实现 |
| ggml CUDA 张量片段 | SM89；plain F32/F16/BF16，非空 dense rank0..8 或按元素对齐的 rank≤4 stride；Copy/Cast/Concat、Add/Sub/Mul/Div广播、Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh、带affine的RMSNorm/LayerNorm、任意轴Sum/Mean与无mask/浮点additive mask的Softmax；FP32中间计算、最终cast；端口对齐与容量限制见下文 |
| ggml CUDA causal GQA | Windows默认、Linux可显式组合；SM89，dense BSHD FP16/BF16，Q/KV heads=8/2、depth=256、KV length≥Q length、无额外mask、FP32累加；每tensor及score矩阵元素数≤INT32_MAX，batch≤65535，端口256字节对齐 |
| cuBLASLt dense MatMul | 非空 FP16/FP32 rank-2，row/column-major 输入、dense row-major 输出；FP32 accumulation，FP32 不降为 TF32 |
| Marlin QuantizedMatMul | `cuda::QuantizedWeights` 的 signed symmetric group128 Q4；FP16 activation/output、FP32 reduction；SM80+，M=1..64、K 为 128 倍数、N 为 64 倍数、N*K<=INT32_MAX |
| SGLang precise gated activation | dense FP16/BF16/FP32 packed halves；SiLU/GELU/GELU-tanh 在 FP32 计算及乘法后一次舍入；FP16/BF16 hidden%8、FP32 hidden%4；输入输出16字节对齐，输出元素数<=INT32_MAX，不重叠 |
| SGLang RMSNorm | dense 同类型 F16/BF16 输入和必需 scale，F32 累加、输出最终 RNE；宽度64/128/256、256倍数至8192、512倍数至16384；16字节对齐、允许精确 input0 原位；混合F32参数不先窄化，继续由下一合法候选承载 |
| SGLang Unary | dense F32/F16/BF16 SiLU/GELU/GELU-tanh，完整16字节向量、对齐16字节、允许精确原位；普通 Unary 的稳定尾值和 -Inf 极限合同由明确数学适配保留；输出元素数<=INT32_MAX |
| CCCL Scan / TopK / Gather | CUDA13.3 toolkit/runtime、CCCL3.3.4、SM89；Scan为F32/I32，TopK为F32/F16/BF16/I32，Gather为plain1/2/4/8字节storage与I32/I64索引；rank≤8、任意axis/byte stride、dense输出、不重叠；容量及caller workspace见下文，managed执行 |
| HYLE gated activation | separate 输入或不满足上述向量宽度的 packed halves；dense FP16/BF16/FP32，同三种激活、精度与元素数/不重叠限制 |
| HYLE 基本 tensor | plain View、Copy、Fill、Select；F32/F16/BF16 Cast、全部既有 Unary/Binary 模式；rank≤8，rank0/broadcast/非负及奇数 byte stride，非 View 输出 dense、元素数≤INT32_MAX |
| HYLE CUDA norms | dense F32/F16/BF16 RMSNorm、LayerNorm，最后一轴、FP32 accumulation；scale/bias 各自为输入类型或 half 输入的 F32 参数，元素数≤INT32_MAX；LayerNorm 使用先均值再 centered variance |

上述 FlashInfer 域复用普通 `ScaledDotProductAttention`，采用 bottom-right causal 对齐。
Direct 调用者按 `workspace_requirement()` 为每个并发 invocation 分配独立 CUDA workspace；
Graph 自动规划同一资源。输入、输出和 workspace 必须活到实际 Completion 结束。
原版派发包含 capability、occupancy 和 kernel attribute 查询，属于 managed 执行，当前不支持
外部 capture；普通 Graph 的 prefer-capture 会报告 commands 路线。失败后重建 invocation，
不在执行时改选其它实现。Windows 未准入此 FlashInfer pin，普通入口在上述域使用ggml。

M2 新增 ggml CUDA 张量映射仍在逐平台验收，具体门见
[M2-B1 CUDA 当前卡](../implementation/tasks/compute/llm-moe-m2-b1-cuda.md)。普通入口会在符合上述域时
绑定原版 ggml；这不是全21行或所有平台验证已完成的声明。其元素数上限为
`(INT32_MAX-1024)/sizeof(float)`，plain storage 的跨度和单stride不超过INT32_MAX；空张量、
高rank不可折叠stride、奇数字节stride、Select/min/max、稳定激活/gate与索引域尚不由这份映射承载。
Copy/Cast/Concat接受按storage类型对齐的subview；数学节点仍要求256字节。Graph对边界值的
所有读写者合并要求，发布的复制结果若被数学节点继续使用也须256字节，准备/提交不会跳过检查。
LayerNorm 用原版mean/sub/sqr/mean/epsilon/sqrt/div实现两遍中心化方差；gamma/beta遵守公共舍入点。
Softmax 保留原版scale/mask融合，先应用scale再加可广播同类型浮点mask；任意axis通过原版布局
变换转为行。无有效值行返回NaN。Direct允许相同dense描述符的精确input0原位，部分重叠或
mask/output重叠拒绝；Graph region仍要求不别名。Bool mask及LogSoftmax尚未由这份映射承载。
必要cast/repeat/cont和每slot冷常量上传属于完整成本，managed重放仍可能在原版pool中分配。
原版内部copy/repack在提案中仍为未知；Graph报告的copy_bytes=0仅表示适配边界没有额外搬运，
不能用它推断原版程序没有materialization成本。

ggml attention采用相同bottom-right causal合同。FP16保留原版融合attention；该版本的
SM89融合路径会缩窄BF16 K/V，因此BF16使用原版F32 MatMul/softmax组合与原版cast，
不将大指数BF16静默转成FP16。此组合持有score矩阵与转换临时值，成本随
`batch * query_heads * query_length * kv_length`增长，性能资格仍待M2。
ggml自管每invocation临时资源；`workspace_requirement().size == 0`时直接创建invocation，
不分配零字节workspace。因果mask只在冷准备时上传，重放只绑定地址。BF16域要求未设置
`GGML_CUDA_CUBLAS_COMPUTE_TYPE`，否则冷拒绝；Context绑定后不得修改影响上游数值的环境变量。

ggml域由普通CUDA入口冷绑定；expert工厂为`compute::ggml_cuda::provider()`，头为
`hyle/compute/providers/ggml_cuda.h`。每个invocation独立拥有原版backend、stream、scratch与
capture cache，输入输出借用到Completion。HYLE用入口/出口两个event连接现有Queue，ggml
保留片段内部调度和融合。该路径为managed，原版首轮pool/handle与后续capture/cache维护可能
分配或阻塞host；不在prepare偷跑payload，不支持外部capture。运行失败先收束已接受的原生
工作并退役该invocation，同一prepared program可创建新槽。上述限定域已完成Windows/WSL
源码、静态与动态安装消费；全需求域与成熟实现性能规则由M2裁决。

基本 tensor 使用同一普通 `prepare()` 和 CUDA Graph 入口，程序准备后无需 workspace 或槽私有
device storage。View 按共同 byte offset 必须别名原 storage；其余逐元素操作仅在输入与输出描述符
完全相同时允许精确原位，Copy/Cast 始终不重叠。Select 原样保留被选择值的位模式；浮点算术使用
FP32 中间值并在末尾舍入一次，Cast 窄化为 RNE。Norm 参数在计算前保持 F32，不先窄化；允许精确
input0 原位，其他参数和部分重叠拒绝。非连续 norm 输入先用普通 Copy materialize，并计入实际成本。
上述 SGLang norm 使用锁定原版 warp/CTA/double-load 算法，其他 norm 候选使用 CUDA SDK
Cooperative Groups；均一次 native launch、零 workspace/槽私有状态。新 SGLang norm/Unary
域的完整性能比较仍待 C1 资格，不能从普通入口接通推断为硬件上的性能赢家。
Reduce 和 Softmax/LogSoftmax 支持 F32/F16/BF16、任意轴/普通 byte stride、dense output 和 F32
累加；半精度 Reduce 必须显式声明 F32。Reduce 不允许原位，空集合仍写约定 identity，Min/Max
传播 NaN；Softmax 可用广播 Bool/同类型 additive mask、scale 和精确 input0 原位，无有效值的行
产生 NaN。每个原子一次 native launch，无 caller workspace 或槽私有状态；Softmax 在≤8192列时
缓存 F32 行（最多32 KiB片上空间），更长行读取输入三遍。上述输入/输出非空元素数≤INT32_MAX。
这些新能力尚待 C1 完整片段的等责性能和全部交付验收，不从正确性推断最快规则。
TopK 在上述已资格化环境通过普通入口绑定原版CCCL，返回原始位模式的 F32/F16/BF16/I32 values
与 I32 indices；支持任意轴和byte stride，两个输出dense，输入与输出及两输出之间都不重叠。
同分（包括±0）按较小列号优先，NaN排在全部非NaN后；largest/smallest均可，sorted=false仍可
返回有序结果。原版DeviceRadixSort对逻辑行与分数稳定排序，再通过库Transform提取原值/下标，
没有4096列限制。按`workspace_requirement()`分配每个invocation的独立CUDA工作区；其大小随
输入元素数增长，包括两份keys/indices及SDK临时区。默认managed执行不接受外部capture。
expert `cuda_native::provider()`和Graph明确要求prepared的既有路径仍保留原范围：F32/F16/BF16，
k=1或k>1且列数≤4096，无caller workspace，允许原生capture；这不是新增性能裁决。
Concat 支持同类型的任意数量 plain 1/2/4/8 字节输入、任意轴/byte stride 和 dense output，
逐位保留数据，无输入输出重叠，输出元素数≤INT32_MAX。每最多四个输入一组原生 kernel，常见
双输入一次提交；空组不取指针或启动 kernel，程序元数据按输入数量冷分配，无 caller workspace
或槽私有状态。普通 Graph 可直接组合 View→Concat 和窗口 append，并使用原生 capture。
Gather 在上述CCCL环境通过普通入口调用原版`thrust::gather`，支持plain1/2/4/8字节source、
I32/I64 indices、任意axis/byte stride和dense output，逐位保留数据。默认checked先在GPU用
原版Reduce Min/Max验证全部索引，只回传两个I64控制标量并等待结果；越界通过Completion返回
OutOfRange，整个Gather output保持不变，失败Invocation需重建。tensor计算与索引payload不下载到host。
output不能与source或indices重叠；output/indices元素数各≤INT32_MAX，存储跨度≤PTRDIFF_MAX，
source轴与地址计算保留64位。每个checked invocation需要独立caller workspace，按查询的大小和
对齐分配并保活到Completion；Graph自动规划。unchecked由caller保证索引有效，省略检查及工作区。
空output不访问payload；默认managed路径不接受外部capture。既有expert/prepared-only Gather
保留原生capture、连续后缀搬运与无caller workspace的原范围，冻结实现未作性能裁决。
CUDA `Scatter` 的 assign 保留1/2/4/8字节storage原始位，add支持F32/F16/BF16。当前过渡实现按
updates逻辑行主序处理、半精度每次F32求值后舍入；这是该实现特征，公共add合同允许重结合及
存储精度或更宽的累加。支持I32/I64、任意axis/byte stride、dense输出，
仅允许同描述input0精确原位更新；非空output/updates/indices各≤INT32_MAX。checked坏索引使整个
Scatter输出保持原值，通过Completion报告；失败槽退役后可建新槽恢复。
空indices只保留source，单索引无需workspace；多索引由CUB stable SortPairs对索引/ordinal分组，
显式caller workspace包含四个I32数组和SDK scratch。每个invocation在冷期固定并上传独立排序图，
普通Direct复用它，外层CUDA Graph接入原生child graph；执行期不重新调用排序准备。原位更新省去
整状态复制，opaque CUDA graph私有内存如实报告unknown，不能当成零内存。完整成本包括索引检查、
排序、必要source复制、逐目标有序更新与完成回报，不等同于无序atomic scatter。
CUDA `Scan` 在上述环境默认由原版CCCL承载I32/F32、inclusive/exclusive、任意axis/普通byte
stride与dense输出；非空总元素≤INT32_MAX，空域不访问payload。原版ScanByKey计算每个逻辑行
的inclusive前缀，再由库Transform提取所需输出；exclusive首项为零。I32先提升到I64扫描，原版
Reduce检查全部前缀（包括exclusive最后总和），仅两个验证标量返回host；任一越界通过Completion
返回OutOfRange，所有用户output保持原值。检查包含一次host等待，不能声称整条路径不阻塞host。
Direct必须提供独立caller workspace，不接受输入/输出别名；Graph规划同一工作区。caller的
tensor、workspace和Queue须活到Completion结束，即使PreparedOperation/Invocation句柄已释放。
既有expert/prepared-only Scan仍保留其原生capture和精确原位范围；默认CCCL不声明这些能力。
CCCL这些接线的Windows开发验证与尚未运行的跨平台/三交付/P1见
[当前接入卡](../implementation/tasks/compute/llm-moe-m2-b1-cuda-collectives.md)，不代表完整模型覆盖。
整数/FP8 Cast 与算术等
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

公共 SDPA 描述也接受 GQA/MQA：Q head 数必须是正 KV head 数的整数倍，K/V head 数相同；
Q head `h` 使用 KV head `h / (Qheads / KVheads)`，输出与 mask 的 head 轴仍按 Q。
Provider 单独声明计算域；当前 CPU Native 只支持同 head MHA，对 GQA/MQA 返回 unsupported。
| 自定义 CPU 工作 | `cpu::submit_host(queue, callable, args...)` | `noexcept` callable，返回 `void` 或 `Status`，返回前完成 |
| 可准备、可进入 Graph 的用户 kernel | `make_custom_provider` + `make_custom_kernel` | owning typed factory；shared HYLE Queue，逐 Invocation 状态；见下文完整例子 |
| CUDA runtime | `cuda::backend/device/MemoryResource/submit` | 显式 device/pinned/managed storage、stream borrow/adopt、typed command、真实 async Completion；不冒充 CUDA 算子目录 |
| byte transfer | `transfer_properties/copy_async/fill_async` | 对 Mem `BufferView` 做显式 byte copy/fill；不隐藏 tensor repack、pageable staging 或 host wait |
| 完成边界 | `Queue::reserve_completions` / `Completion` | direct submit 或一个 Graph Queue segment 只产生一个 prefix Completion；销毁 pending handle 不等待 |
| provider 选择 | `Context::create(device, Provider)` / `ContextOptions::providers` | 显式且有序；冷能力查询不支持或 prepare 明确安全拒绝为 unsupported 才尝试下一项，真正错误不按 code 猜成回退 |
| 精确资源 prepare | `OperationResourceBindings` / TensorView overload | provider 先报告要求，再对 input/output/workspace 的 exact `MemoryResourceRef` 冷验证；submit 不重新选资源或隐藏 copy |
| provider 诊断 | `provider_identity` / `runtime_properties` / `persistent_bytes` | O(1) 读取 provider/version、shared/delegated owner、replay 成本类别、threading/capture/workspace owner；`persistent_bytes == nullopt` 表示 provider 无法给出完整精确值，不是零 |
| 稳定重复执行 | `prepare -> PreparedOperation -> OperationInvocation::submit` | prepared artifact 可共享；每个并发 lane 独占 invocation/workspace；submit 不重新推 shape/路由 |

公共 schema 包括上表的普通张量运算、Scan、QuantizedMatMul 和 GatedDeltaNet，共 22 个 family，并能做纯描述推导。
Native 承载其中除 QuantizedMatMul 外的 20 个 family，具体 dtype/layout 域见上表。这不表示每种 dtype、
layout 或 encoding 都由同一个 provider 执行：当前 SDPA 是 f32 reference/optimized fallback；block/opaque
storage、量化 kernel 与窄 storage attention 仍应由理解相应格式的 provider 接管，不会被隐藏 materialize。

`GatedDeltaNet{}` 描述六个显式输入与两个输出：Q/K `[B,T,Hk,Dk]`、V `[B,T,Hv,Dv]`、
FP32 log-decay/beta `[B,T,Hv,1]`、FP32 初态 `[B,Hv,Dv,Dk]`；输出为 V 同形结果与新 FP32 状态。
Q/K/V 使用同一 F16/BF16/F32 类型，Hv 是 Hk 的正整数倍。它不修改初态，也不隐含 Q/K 归一化、
gate 激活或状态写回。目前只完成公共描述与推导；原版后端的实际准入由
[M1-A 接入卡](../implementation/tasks/compute/llm-moe-m1-apple-admission.md)记录，不能据此认为四路线已可执行。

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
当前 CPU Native 与 oneDNN MatMul 仅实现显式 F32 累加（也是 `MatMul{}` 的默认值）；
其他累加类型在准备期返回 `Unimplemented`，默认 Direct/Graph 不会通过 fallback 静默降精度。

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
不能把它当作另一种裁剪或环绕语义。Scatter assign 对重复index取最后逻辑更新并保留原始位。
浮点add是base与所有updates的归约，允许重结合、不同累加顺序及存储精度或更宽的中间值；
不承诺逐项舍入、行主序fold或跨调用逐位一致。模型要求F32中间值时，使用已有Cast与F32端口
明确表达，再在规定的存储边界转换。模型精度档和最终误差门仍适用，不能丢更新或静默降精度。
当前CPU reference采用逐次storage舍入，这是实现特征。Scatter本身不提供不同invocation之间
对同一状态的并发原子更新保证，调用方仍负责资源排序。

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

oneDNN 的 MatMul 路线接收 **plain** dense row-major f32 rank-2 MatMul。它适合普通
`[M,K] * [K,N]`；模型常见的
物理 `[N,K]`、逻辑转置为 `[K,N]` 的连续-K 权重会回落到 HYLE-native SIMD route。
全部 GatedActivation、batched/零尺寸 MatMul 和其他未支持形状也走 native，不会在 submit 时重新选择。在默认串行
executor 下，`M < 16` 的 decode MatMul 同样选择结构更轻的 HYLE-native kernel；传入真实并行 executor
时，oneDNN 可以把 N 维拆成互不重叠的连续 tile。

oneDNN 另接入原版 RMSNorm/LayerNorm primitive：dense 非空 `[..., hidden]`，显式 F32
accumulation，可选 scale/bias；scale 和 bias 同时存在时需同类型。F16/BF16 原生执行受
oneDNN 的 CPU ISA 要求限制；该 Provider 不隐式转换整个 tensor。不满足时普通 CPU context
继续匹配 ggml，再匹配现有 Native。F32 norm 后的最多4个 Add/Mul 可由 oneDNN region 绑定原生 post-ops，支持 scalar、
末轴向量及单行的完整同形状操作数。批量完整同形状 Add/Mul 保留为后续节点，使此前的归一化/
广播前缀继续使用优化内核；冷准备拒绝已识别的通用慢速融合实现。中间输出被发布或分叉时
保留边界。半精度跨节点链保留中间舍入。融合接入不代表每个形状已获性能优选资格，当前验收
状态见模块 registry 和对应任务记录。

归一化 primitive 每次执行有 oneDNN 内部 metadata 分配，报告 `ProviderReplayClass::kManaged`；
其数量不冒充 HYLE allocator 可计量的零字节。Graph 的 prepared-only 约束会排除该 managed
路线。每个 invocation 独立拥有 native stream/argument handles 和 caller scratchpad，可并发运行；
支持精确 input0/output 原地重叠。接受后的执行失败退休当前 invocation，需创建新槽，不能重放
到另一个 Provider。既有 BRGEMM 的 prepared replay 合同保持不变。

### oneDNN 张量接入检查点

[M2-B1 oneDNN](../implementation/tasks/compute/llm-moe-m2-b1-onednn.md)已接入原版`Select`与
`Copy`：普通CPU Direct/Graph自动绑定，支持plain rank0..8、非负及奇数字节stride、空输出，
以及已登记1/2/4/8字节scalar存储。Select将条件与两个分支共同广播，可用张量条件选择两个
标量常量；输出dense，保留所选值的原始位，包括NaN payload、signed zero与大整数。

实现用原版oneDNN字节primitive描述存储，无浮点转换、预先展开广播或HYLE数学kernel。
每个请求复用上述stream/argument handles与caller scratch；managed replay、vendor资源
未知项和失败退休语义保持。当前这两个映射均要求输出不与输入重叠。

`Unary{UnaryKind::kSoftplus}` 的 F32 dense 标量/rank1..8/空张量也已接入普通CPU入口及
Graph叶程序桥接。准备时固定匹配原版 `ref:any`：该版本默认JIT的小负值尾部不符合稳定精度
合同，因此不用于此域。执行不搜索或计时；输出不能与输入重叠。非连续布局、F16/BF16尚未由
这条成熟映射支持，准备时明确拒绝。Windows开发验证已通过；完整平台/三交付/成本资格仍按
任务卡验收，min/max及数值Cast的其余合格映射尚待完成。

### ggml CPU 接入检查点

[M2-B1](../implementation/tasks/compute/llm-moe-m2-b1-cpu.md)正在接入原版 ggml CPU 程序。
普通 CPU 工厂已组装这一 Provider；独立入口为 `compute/providers/ggml_cpu.h` 中的
`ggml_cpu::provider()`。当前映射包含 Copy、浮点 Cast、Concat、任意轴 Gather、Add/Sub/Mul/Div、
Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh、RMSNorm/LayerNorm 和任意轴集 Sum/Mean 的有限域。
half 计算显式经过原版 F32 cast/计算/输出舍入；转换属于真实执行成本。dense 可折叠高 rank，
非连续输入限 ggml 四轴且 stride 满足元素对齐；无 alias 资格。完整特殊值激活等仍等待成熟
承载，不能由名称相同推导已支持。

Gather 使用 I32 索引，支持标量或多维索引，以及上述已准入的非连续布局；每次运行都先按实际
索引 stride 检查边界，再调用原版 get_rows。不同 outer 行必要的索引 repeat 保存在每请求工作区。
二元运算支持公共推导确认的任意合法广播，仍受上述输入布局准入限制，保持左右操作数顺序。
同形、标量、行/列直接折叠，其余由原版逐轴 repeat 承载。Sum/Mean 支持正负轴集合与
keep-dim；必要的分组转置沿原版 view/cont 执行。空归约集合发布 Sum=0、Mean=NaN。
转置、广播与中间 F32 存储均在准备期确定并分配，执行时不重建程序。

同一 CPU 接入还组合 M0 已登记的 C++ 标准库 indexed 路径：F32/I32 `Scan` 调用原版
`std::inclusive_scan`，I32 在 I64 临时域检查全部前缀和最终总和，再发布 inclusive/exclusive
结果；越界不写输出。F32/F16/BF16/I32 `TopK` 调用 `std::partial_sort`，按 value 与原下标
的全序保留 ties，NaN 始终放在非 NaN 后，values 保留输入位模式，indices 为 I32。
这两个叶程序支持 rank1..8、任意合法 axis、非负 byte stride；不进入 ggml 数学图，也没有
私有 worker。每个 invocation 的 scratch 通过所供 allocator 冷分配，执行沿同一 CPU Queue
与同步 executor。两输出和动态失败边界也适用于普通 Graph。

每个 invocation 在准备期持有 graph、plan、工作区和独立 pool；重复执行只绑定当前数据。
`cpu::DefaultContextOptions::native_threads` 默认1，包括提交线程，显式大于1才创建持久 workers。
借用的 executor 接收一个完整原版程序，不把其 concurrency 自动变为 ggml worker 数量。
程序同步完成后返回终态，索引越界先于输出写入拒绝；一个 invocation 不允许并发运行或关闭。
当前完成 Windows CPU-only 的阶段开发验证，跨平台、三交付与性能资格仍以活动卡为准。

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

标准 CPU 配置的该选项默认 `AUTO`：支持的 x86-64 目标接入 oneDNN，AArch64 使用已构建的 ggml/Native 后备。构建时
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
