# Apple MLX/Metal：Direct Compute 与 Graph

Apple 路线保留 MLX 的原生 arrays、编译融合与内存管理；原版 ggml Metal region 保有自己的
原生程序与完成边界。普通 typed Direct 和 Graph 应用固定规则；用户不排序 Provider、不提交 timing profile。R5 已完成
[限定环境验收](../implementation/tasks/compute/apple-v1-four-route-foundation.md)，性能资格不外推其它硬件。

可选[模型与阶段信息](compute.md#可选模型与阶段信息)作为最后一个参数传给
`Context::prepare(operation, inputs, workload)` 或 `Context::compile(region, inputs, workload)`；
Graph 使用 `graph::apple::CompileOptions::workload`。原生 `Program::workload()` 保留绑定身份，
Graph 的 typed lowering 收到同一份冷信息。无注解继续使用合法 MLX baseline；这不声明新增了模型
专用算法。改变身份后重新编译，MLX replay 不读取模型信息或重选规则。

固定模型画像的当前片段支持、拒绝与数值限制统一见[四路线覆盖表](model-fragment-coverage.md)。

## 构建与三种 C++ 交付

只支持 macOS Apple Silicon。源码构建需要可用的 Apple C++/Objective-C/Objective-C++ 和 Metal 编译工具；
`xcrun -sdk macosx metal --version`、`xcrun -sdk macosx metallib --version` 必须成功。CLT 本身未必
包含 Metal CLI。HYLE 不代装系统 SDK；ON 缺条件失败，AUTO 不可用时关闭，OFF 不探测或下载 MLX。

```sh
cmake -S . -B build-apple -G Ninja \
  -DHYLE_COMPONENTS="compute;graph" \
  -DHYLE_COMPUTE_APPLE=ON -DHYLE_COMPUTE_CPU=OFF -DHYLE_COMPUTE_CUDA=OFF \
  -DHYLE_BUILD_TESTS=OFF -DHYLE_STATIC=ON -DHYLE_INSTALL=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-apple --parallel 2
cmake --install build-apple --prefix /your/sdk
```

完整源码嵌入使用相同选项和公开 target；安装 SDK 用 `find_package(Hyle ... COMPONENTS compute graph)`。
设置 `HYLE_STATIC=OFF` 交付 shared，使用独立 build/install prefix。shared HYLE 与消费者共用一份
shared MLX；不要另外链接另一版本或另一份 static MLX。

```cmake
target_link_libraries(my_app PRIVATE hyle::graph) # Direct-only 用 hyle::compute
hyle_stage_apple_runtime(my_app)
```

这个显式 helper 在 source 和 installed SDK 中相同。它把真实 `mlx.metallib` 放到 EXE 旁，并跟踪资源
更新；shared SDK 的 native dylib 旁也保有该文件。发布应用时保留邻置资源，不能只复制 executable。
MLX JIT 仍需要部分预编译基础 kernel，缺文件不是可以忽略的 warning。

运行时缺失该资源会通过`Context::create()`返回原native加载错误，不把缺文件误报为机器
没有GPU；调用方应检查结果中的Status消息。HYLE不会自动退到CPU或下载替代资源。

SDK 带锁定的 MLX v0.32.2、Metal-C++ 26 和对应许可证；其 CPU/CUDA/Python/GGUF/safetensors 功能
关闭。MLX 的 F32→BF16 转换带一个复用上游转换函数的窄补丁，保留次正规数和最近偶数舍入；
该转换保留一次实际存储边界，可能增加原生 dispatch 和中间数组，其余操作仍由 MLX 融合。
源码、静态与共享构建采用同一补丁及来源声明。源码构建还需要 Python 3，用于生成带哈希检查的
build-local 依赖源码，原始缓存保持不变。Apple 专属头公开 MLX C++ 类型；共同 Compute/Graph 头不要求这些 SDK。

## 普通 typed Direct

下例放在返回 `hyle::Status` 的函数内：

```cpp
#include <hyle/compute/apple.h>
#include <array>

namespace mx = mlx::core;
namespace apple = hyle::compute::apple;
HYLE_ASSIGN_OR_RETURN(auto context, apple::Context::create());
const std::array inputs{
    mx::array({1.f, 2.f, 3.f, 4.f}, {2, 2}),
    mx::array({2.f, 0.f, 0.f, 3.f}, {2, 2})};
HYLE_ASSIGN_OR_RETURN(auto program,
    context.prepare(hyle::compute::MatMul{}, inputs));
HYLE_ASSIGN_OR_RETURN(auto invocation, program.create_invocation());
HYLE_RETURN_IF_ERROR(invocation.bind(inputs));
HYLE_ASSIGN_OR_RETURN(auto work, invocation.submit());
HYLE_ASSIGN_OR_RETURN(auto result, work.wait());
const auto &native_output = result.get()->arrays()[0]; // 成功终态后的 lease
```

当前 typed 域还包括：

- F32/F16/BF16 的 Unary、广播 Binary、浮点 Cast、三种 GatedActivation，以及 RMSNorm/LayerNorm。
  gate/norm 在 F32 求值，最终一次回到输入存储类型；norm 可用相同类型或 F32 scale/bias。
- F32/F16/BF16/I32/I64/Bool 的 View、Copy、Fill、Select 和 Concat；Select 使用 Bool mask。
  支持 scalar、空维度、rank≤8、原生转置与广播，实际布局由 MLX 持有。
- F32/F16/BF16 的 Reduce Sum/Mean/Min/Max、Softmax/LogSoftmax。Reduce 支持任意
  axis 集/keep-dim，空集合为 0/NaN/+Inf/-Inf，Min/Max 传播 NaN 并保留 ±0 优先级；half Reduce
  和所有 Softmax 必须显式 F32 accumulation。Softmax 支持 scale、广播 Bool 或同类型 additive
  mask、任意 axis；未屏蔽的 NaN/+Inf 或全 -Inf 行产生全 NaN。
  计算由原版MLX Sum/Mean/Min/Max和precise Softmax承载；Min/Max另用原版布尔归约确定零符号。
  LogSoftmax采用MLX官方的先减max、再减shifted logsumexp组合，保留大公共offset下的精度。
  F32中间数组、mask转换及原版归约临时资源计入执行成本，由MLX管理。
- TopK 支持 F32/F16/BF16/I32，返回原始存储类型 values 与 I32 indices。同分含 ±0 按小索引
  优先，NaN 总在最后；largest/smallest 均可，sorted=false 仍返回有序结果。使用原版 MLX
  stable argsort 后取 k 项；完整排序、排序键和必要布局/精度转换计入成本。Graph 用 `add_node()` 与
  `result(node, 0/1)` 消费两个输出，Direct 从成功 result 的 `arrays()[0/1]` 读取。
- Gather 接收上述六类 source 存储与 I32/I64 indices，支持任意轴、scalar/多维索引和原生布局；
  输出保留原始位。默认 checked 在 GPU 检查索引，负值和越界通过完成结果返回 OutOfRange，
  不采用 MLX 的负索引 wrap。原版比较/where 为 take 提供安全索引，失败结果不会发布。
  空输出不读取 payload；unchecked 要求调用方保证所有索引合法。
- Scatter assign 接上述六类 storage，保留原始位，重复目标最后逻辑更新胜出；add 接F32/F16/BF16，
  公共语义允许重结合及存储精度或更宽的累加，不承诺逐次舍入或逐位重现。返回新状态数组，输入与旧结果
  保持不变；结果可绑定为下次调用的状态输入。任意轴、scalar/多维 I32/I64 indices 和原生布局
  均可使用，默认 checked 错误通过完成结果返回；空 output 跳过 payload，空 indices 保留 source。
  非空 output/indices/updates 的元素数≤INT32_MAX。assign 用原版 MLX I32 Scatter Max 记录每个
  目标的最后更新位置，再以原版take/where选取原始存储；临时映射按目标轴长度分配。
  add用原版MLX Scatter Sum，F16/BF16提升至F32累加后一次转回storage；F32使用原生F32累加。
  必要映射、布局整理、精度转换、原生临时、新结果及检查都计入资源和执行成本。
- Scan 支持 I32/F32 任意轴 inclusive/exclusive 前缀。I32 检查所有数学前缀，包括 exclusive
  未写出的最终总和；溢出返回 OutOfRange。F32 允许原生并行重结合。非空 Scan、Gather output/
  indices 的元素数≤INT32_MAX；整数 Scan 使用原版 I64 cumsum 与比较/any，exclusive 由原版
  slice/concatenate 移位，成本包含转换、检查及最终状态读取。
- 二维 F32 MatMul、二维 unsigned affine Q4/group64/F32。其它 typed 域明确拒绝。

Q4 weight 的 native dtype 是 uint32，shape
`[N,K/8]`，每 word 按 K 顺序从低 nibble 开始；scale/bias 是 F32 `[N,K/64]`，公式为 `q*scale+bias`。
Graph 描述用 `apple::affine_q4_description(N,K)`，不把 Marlin/GGUF/其它四位格式直接传入。

typed 准备也可以只给元数据，例如
`context.prepare(RmsNorm{}, std::array{apple::InputSpec{{1,2048}, mx::bfloat16},
apple::InputSpec{{2048}, mx::float32}})`；不需要制造用户 payload。精度或形状不合格时准备返回
具体拒绝。MLX 仍在冷期用隔离输入编译暖机。新增 typed 域的完整性能尚未验证，不能从既有
FFN 的性能结果外推。

checked Gather/Scatter 与整数 Scan 的低层 `apple::lower()` 必须在 `Context::prepare/compile` 的 managed
tracing 内调用，才能把设备状态绑定到完成结果；在外部直接调用会明确拒绝。准备不以隔离暖机
的状态裁决用户 payload。执行时，只有可达 checked 表达式的错误参与首错回报；没有此类表达式
的程序不添加状态输出。有检查的 region 增加一个隐藏 I32 native 数组，沿既有同步完成后读取，
它不会出现在 `Results::arrays()` 中。GPU 内部临时结果不会在失败时成为成功 lease；输入、共享
常量与此前结果仍保持不变。

`context.compile(callable, input_specs)` 表达更完整的原生 region。普通 `InputSpec` 的 shape/dtype
描述实际 MLX array；provider-native storage 还会保留一个 logical `TensorDesc`。也可给 ready
examples。callable 接受 native arrays 和 Context 的 GPU stream，返回数组列表或
`hyle::Result<std::vector<mlx::core::array>>`。在其中使用 MLX ops 或 `apple::lower(Operation,...)`
组合 typed 算子。完整 Direct Q4 FFN 见
[样例](../../samples/package/compute/cpp/apple_ffn.h)，没有 Graph 依赖。

### FP8 分块权重与动态激活

Apple 的普通 `Context::prepare()` 和 Graph 支持 canonical E4M3FN/F32 block128 权重：
预编码 activation 的 MatMul、`model_ops::float8_block_linear()`、
`float8_weight_decode()` 和 `grouped_float8_linear()`。逻辑描述使用共同的
`float8_activation_description()`、`float8_weight_description()` 或
`float8_grouped_weight_description()`；`apple::input_spec(desc)` 给出承载完整 bytes 的
一维 MLX uint8 array。Direct 按这些 InputSpec 准备，Graph 使用相同 TensorDesc。
仅给一个 uint8 array 无法推断其权重格式，应显式保留逻辑描述。

动态线性接受 BF16/F32 activation，按 K128 分组量化后执行各块 F32 点积、缩放和累加，
输出保留 activation 类型；存储为 BF16 不代表省略中间 FP8。默认普通 scale floor 为1e-10，
DeepSeek 的 power2 scale 和1e-4 floor 由操作描述显式选择。grouped 权重保留全部专家的
压缩存储，仅对本次选择的专家解码；负值或越界 ID 通过 Completion 返回 OutOfRange，
不发布输出。原生 F32 运算允许 fused/unfused 舍入差异，不承诺与 CUDA 逐位一致。

这些程序由原版 MLX 编解码、MatMul、ReduceMax、Take 等原语在冷期编译。
执行成本包括选中权重的 F32 解码、中间乘积、scale 广播和必要复制，不能视为零分配或
硬件原生 FP8 GEMM。Graph 捕获常量时的折叠由 MLX 管理，HYLE 不提供隐式权重缓存。
FP8 group64/128 roundtrip 也可用。FP4 group32 roundtrip接BF16二维输入，保留原E2M1
最近偶数舍入、power2 scale/floor及最终BF16存储；极小值和零符号由GPU位视图适配保真。
`model_ops::grouped_float4_linear()`消费canonical原E2M1 nibble/F32 group32专家权重、
BF16 activation和I32 IDs；activation按原group128 E4M3FN量化，F32逐block累加后存BF16。
重复ID合法，越界在Completion返回OutOfRange，不发布失败结果；它与MXFP4格式不同。
本批已完成普通入口、资源正确性及最终三交付。当前完整BF16 FFN与量化MoE的[Apple性能批次](../implementation/tasks/compute/llm-moe-d5-portable-fragment-performance.md)也已通过；该成绩不代签FP8所有形状的独立性能。

### GGUF/ggml 原生压缩权重

已经解析 GGUF tensor table 的调用方先用公共 `gguf_weight_description()` 或
`gguf_grouped_weight_description()` 建立逻辑描述，再用 `apple::input_spec()` 得到实际 MLX carrier：

```cpp
auto logical = gguf_weight_description(GgufQuantizedType::kQ4K,
                                       DataType::kFloat32, k, n);
HYLE_ASSIGN_OR_RETURN(auto packed, apple::input_spec(*logical));
std::array specs{apple::InputSpec{{m, k}, mx::float32}, packed};
HYLE_ASSIGN_OR_RETURN(auto program, context.prepare(MatMul{}, specs));
```

`packed.shape` 在总字节数≤`INT_MAX`时为一维 `uint8[logical->required_bytes()]`；更大payload
返回checked二维 `[encoded_rows,row_bytes]`，各axis须在native int范围内。上传和bind都使用
实际返回shape，不能把大字节数强制cast成int。bind 的 array 必须含原 GGUF bytes、
row-contiguous 且从自身 Metal buffer 起点开始。Invocation 保有 array owner 到 checked completion。
HYLE 不加载 GGUF、不量化、不解码成常驻浮点副本，也不在 submit 时重排或重新选 Provider。
examples overload 只能看到 `uint8`，无法猜出 Q4_K/IQ4_XS 等语义；压缩权重必须用上述 spec overload。

当前 ggml Metal 路线接入 Q4_0、Q4_1、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、
IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、IQ1_S、IQ1_M、MXFP4，
以及原Anemll IQ1_XS/IQ1_XXS/IQ1_XXXS的二维 MatMul 与 `[E,N,K]` GroupedMatMul。
后三类保留原MV程序；无原MM时按原MV承载，不调用不存在的kernel。checked embedding
接入上述前18类；窄IQ1三类没有原GetRows，不声明支持。Q4_K精确行选择使用原ggml
F32 row-decode公式的MLX GPU承载，只取所选行，避免原Metal half高nibble精度损失。
准备期调用固定上游 capability 检查；设备/形状不支持时明确拒绝，不展开到 F32 或转去 CPU。
这些格式与前述 MLX affine Q4 的 `uint32 + scale/bias` 合同不同，不能交换 carrier。

### 模型制品的 native packed 与冷准备

`model_ops::MlxQuantized`表达原MLX affine4/MXFP4/MXFP8线性及grouped，affine4另有
checked embedding；实际group、scale dtype、bias与端口以模型制品及公共schema为准。
普通prepare和Graph均接原MLX qmm/gather_qmm/take/dequantize，无需选择内部Provider。
U32 words按低bits在先存储；MX的U8 E8M0 scales与GGUF或官方F32 scales不同。

NF4/nested NF4、AWQ GEMM4和compressed-tensors symmetric4使用
`model_ops::artifact_weight_decode(format, group_size, output_type)`在GPU冷准备。
输入按[公共端口合同](../../include/hyle/compute/model_ops.h)给出真实packing/scales/zeros/
codebook；结果[N,K]保留为ready常量，再用于成熟MatMul或Gather。HYLE不加载模型、不按
文件名猜格式，也不把该准备放在每次decode内。原NF4 codebook/nested offset、AWQ列顺序
和zero、CT减8及源存储舍入均保留。

当前共同目录的559 GGUF、34packed、45plain合同已在两普通入口取得样本数值/replay证据；
大词表全行按4096-row shards验证，MoE仅共享有限专家witness，不代表巨型整模型驻留。
新官方DSpark与Qwen/Flash量化MTP片段的准入见[当前Apple收口卡](../implementation/tasks/compute/apple-cpu-cuda-parity-closure.md)。
官方F32-scale FP4的全部有限BF16存储值、正常/极小scale、中点邻域与符号零已通过精确保真
验证；Pro0813/Flash0731完整压缩attention/indexer/FP4 MoE与三stage DSpark也已两入口验证。
这些是实际算子/片段形状、状态与存储验收，不代表当前16GiB机器能驻留这些完整大模型。
默认为合法成熟承载，尚未实测的
硬件/driver/模型组合不声明best-of或SOTA。

## 原版 Gated DeltaNet

`context.prepare(GatedDeltaNet{}, inputs)` 冷绑定锁定的 ggml Metal 完整递归程序。当前接入域为
F32、相同 Q/K/V head 数、D=32/64/128、1≤T≤32767；输入及 packed 输出元素数不超过 INT32_MAX。
该有限域的原生/资源验证和源码、静态、动态三交付已通过，机器与限制见
[M1-A 验收卡](../implementation/tasks/compute/llm-moe-m1-apple-admission.md)。

六个输入按 Q、K、V、log-decay、beta、初态排列；Q/K/V 是 `[B,T,H,D]`，两种门是
`[B,T,H,1]`，初态是 `[B,H,D,D]`（最后两轴为 value/key）。Q/K 是否归一化、beta 是否激活由
调用方决定，操作不隐含这些变换。两个结果分别是 V 同形输出和新 FP32 状态，初态保持不变。

```cpp
const std::array<apple::InputSpec, 6> specs{{
    {{1,17,16,128}, mx::float32}, {{1,17,16,128}, mx::float32},
    {{1,17,16,128}, mx::float32}, {{1,17,16,1}, mx::float32},
    {{1,17,16,1}, mx::float32}, {{1,16,128,128}, mx::float32}}};
HYLE_ASSIGN_OR_RETURN(auto program,
    context.prepare(hyle::compute::GatedDeltaNet{}, specs));
// 与其它 Program 相同：create_invocation → bind(六个 ready arrays) → submit → wait。
```

Graph 使用 `add_node(GatedDeltaNet{}, inputs)` 和 `result(node, 0/1)`，自动组合递归与 MLX
区域。纯 MLX 的 `Context::compile(callable, ...)` / `apple::lower()` 不接受外部递归程序；需要
跨库组合时使用普通 Graph。各个 MLX 区域内部仍由 MLX 编译和融合。

每个 Invocation 独立持有 ggml 资源，输入保留原生 owner；输出由 MLX 分配一个 packed buffer，
两个 array view 共享 owner。跨库边界进行 checked host wait；非连续输入由原版 MLX 在 GPU
整理布局并计入执行成本。没有 CPU 数学 fallback，也不声明 event-only 或无分配 replay。
`Environment::ggml_revision` 与 MLX、设备、OS/SDK/Metal 编译器身份一起用于冷规则诊断。

## Graph 整段编译

用普通 `GraphBuilder` 的 typed operation 构图，再调用：

```cpp
HYLE_ASSIGN_OR_RETURN(auto program,
    hyle::graph::apple::compile(context, graph, ready_constants));
```

constants 按 `Graph::constant_id()` 的顺序，bind inputs 按 `input_id()` 顺序，结果按 publish 顺序。
其后复用同一 Apple Program/Invocation/OwnedResult 接口。可以释放 Graph 与原 constant 句柄，
Program 保有自己的原生 owner。HYLE 冷期检查并移除 dead pure nodes；纯 MLX 图保有整段融合，
含递归的图只在原生 region 边界切分，热期按固定区域顺序执行。完整例见
[Graph FFN](../../samples/package/graph/cpp/apple_ffn.h)。

此Graph lowering接受固定F32/F16/BF16/I32/I64/Bool及所需U8/U32 carrier逻辑schema、
View派生描述、上述MLX native packed/冷准备与GGUF MatMul/Grouped/checked embedding域。
Graph用同一`apple::input_spec()`映射constant/input carrier；GGUF bytes只进入对应已准入
native程序或精确所选行lowering，不作为普通浮点数组计算。
native arrays 的物理 stride、View 和必要布局整理仍由 MLX 管理；布局转换可能有成本，不能从
HYLE 逻辑描述推断实际 MLX 分配或声称任意 native View 都零拷贝。符号维度、资源效果、显式 control edges 和
未接入的 typed 操作会拒绝；需要这些能力时不能声称 Apple 已覆盖其它 Graph 路线的全部功能。

纯 custom Operation 可通过 `graph::apple::CompileOptions::custom` 在冷追踪时映射到
`mlx::core::fast::metal_kernel`。该回调检查完整 schema ID/版本，返回形状/dtype 匹配的 native arrays，
沿传入的同一 stream 执行。它支持同 runtime 扩展，不承诺任意外部 MTLBuffer/Queue 的零拷贝互通。

## 所有权、线程与成本

- 一个 Context 拥有一个固定提交线程与 GPU stream，按 FIFO 执行完整 region；独立 Context 可并发。
  冷 compile、隔离零输入暖机、native eval 和 checked synchronize 都在该 owner 上完成。
- callable 必须 pure，只使用传入的 GPU stream，捕获已成功完成且不可变的常量。不能在回调里调用
  同一 Context、读取 tensor payload 决定图结构或隐藏外部副作用；它会用隔离输入暖机。
- bind 只接收已经成功 ready 的 native arrays；pending 输入不隐式等待或复制。保活不允许其它别名
  同时写 payload。调用方原生 eval 应先做 checked stream synchronize，再把结果交给 HYLE。
- 同一 Invocation 在执行中或旧结果 lease 未释放时不能覆盖；不同槽可以保留旧结果。拥有型结果
  使用 MLX 生成的原始数组，不进行影子输出 copy；复制某个原生 array 句柄仍保有其独立原生 owner。
- Completion/OwnedResult 析构不隐式等待；取消仍等真实已接纳 GPU 工作排空后才回收。失败/取消槽
  不复用。checked 索引/整数溢出只使当前槽失败，可在同一 Context 创建新槽继续；原生执行错误
  会封闭 Context 的后续接纳，先 close 再创建新 Context。
- `close()` 停止接纳、排空并 join，Context 析构同样有 drain 成本。Program/Invocation/结果可比该
  Context 句柄长寿；关闭后新提交失败，成功结果仍可读。
- 这是 managed replay：每次一次外层提交、原生 MLX metadata/array/cache 成本、锁/原子/唤醒和一次
  checked region synchronization 都存在，不宣称零分配 recorded replay。常驻权重、编译和 slot
  可复用，形状改变需要重新准备。

`context.environment()` 提供冷创建时记录的 GPU 名称/架构、macOS runtime/build、编译 SDK/部署下限、
Metal compiler 和 MLX pin。规则可匹配这些事实；执行中不探测或计时选优。未知硬件采用对应
格式合法的成熟基线，不外推SOTA。默认deployment14.4的构建未启用NAX，不能拿M4验证声明M5/NAX资格。

## 可选编译诊断

需要保存编译依据时，传入一个独立报告：

```cpp
hyle::graph::apple::CompileReport report;
auto program = hyle::graph::apple::compile(
    context, graph, ready_constants, {.report = &report});
if (!program) {
  // report.status 与返回错误一致；没有上一次成功编译的残留统计。
  return program.error();
}
```

报告复制环境/版本、输入输出描述、live 节点原 ordinal 及数值 traits、固定实现规则、边界和并发
政策。它不保留 Graph、Program、常量、custom payload 或设备 owner，资源释放后仍可读。
每次 compile 先 reset；报告自身内存不足也会拒绝编译。默认不传报告，执行期不构造诊断信息。
同一个 report 不能被并发编译写入。

`declared_constant_count` 与 `logical_constant_bytes` 按声明 views 统计，重复引用也重复计数；
它们不是实际驻留内存。`private_storage_bytes` 保持 unknown，MLX 内部融合、私有 heap 和 driver
分配不能从逻辑 shape 精确推断。

需要主机调用跟踪时，构建时设置 `-DHYLE_TRACE=ON`，使用已有 Tracy client。
prepare/compile、submit、checked wait 或 native owner 的执行范围随实际入口记录；默认 OFF 不
求值跟踪参数、不创建 profiler 状态。这些是主机范围，不能解释为 GPU kernel 时间。
