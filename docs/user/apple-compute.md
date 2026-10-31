# Apple MLX/Metal：Direct Compute 与 Graph

Apple 路线保留 MLX 的原生 arrays、编译融合与内存管理；原版 ggml Metal region 保有自己的
原生程序与完成边界。普通 typed Direct 和 Graph 在准备期绑定实现，用户不必排序 Provider。

先完成[构建条件](#构建与运行条件)，再选[Direct](#普通-typed-direct)或[Graph](#graph-整段编译原生-expert)；
并发、借用与关闭前查[资源寿命](#所有权线程与成本)。模型阶段注解和编译诊断属于可选参考。

## 构建与运行条件

选择 Metal 后端，构建或安装 SDK：

```sh
python scripts/init.py --name metal --components compute,graph --backends metal
python scripts/build.py --install <安装目录>
```

编译器、Metal 工具与系统要求见[macOS 环境](../developer/setup/macos.md)。
只支持 macOS Apple Silicon；已选择后端却缺条件时明确失败，不自动关闭或回退 CPU。
源码嵌入与 SDK 集成见[交付指南](distribution.md)。

```cmake
target_link_libraries(my_app PRIVATE hyle::graph) # Direct-only 使用 hyle::compute
hyle_stage_runtime(my_app)
```

helper 在源码和安装 SDK 中相同，将需要的 `mlx.metallib` 放到应用旁并跟踪更新；
shared SDK 的 native dylib 旁也保留该资源。发布应用时保留资源布局，不能只复制 executable。
缺失资源时 `Context::create()` 返回实际加载错误；HYLE 不把它误报为“没有 GPU”，也不下载替代物。

SDK 提供匹配的 MLX 开发头、库和许可证。Apple 专属 API 公开 MLX C++ 类型，
应用不要另外链接不匹配版本或重复的 static MLX；共同 Compute/Graph 头不要求使用这些原生类型。
F32→BF16 转换保留次正规数和最近偶数舍入，实际转换 dispatch/中间数组仍有成本。

分页与不等长 attention 的 full kernel 按实际 Metal pipeline 的线程组共享内存需求选 tile，
也考虑 GPU 验证后的编译资源。默认 tile 超过设备上限时改用较小 tile，仍不满足则报原生错误。
它约束每个线程组的临时空间，不是模型、KV、GPU buffer 或整机内存上限。

低维 vector attention 的两阶段归约保留 F32 partial，最后才转回输出存储类型；普通、ragged 和分页调用共用该数学路径。
Q/K/V 不因此整体宽化。F16/BF16 的 partial 从每项 2 字节变为 4 字节，额外载荷为
`2 * B * Hq * Q * blocks * Dv` 字节；原生分配、布局整理及其他临时开销仍须另计。

不等长/分页 KV 的准备、布局更新和结果释放统一见[批次 attention](batch-attention.md)。
普通分页另支持E4 K/V和各自固定scale：`CacheEncode`复用精确字节编码，原Steel核直接读引用页；
普通页仅活跃Q／partial使用F32，不展开整套缓存。BF16 D64→128、多请求及Q1布局增长已实测。
紧凑E4 MLA另接BF16 Q／输出、512／576单缓存的paged/ragged路径；V为原CKV前512项的strided view。
576维保原BF16共享Q tile以满足32KiB限制，KV／QK／softmax／PV仍F32。数学与寿命范围见[紧凑MLA](batch-attention.md#紧凑-mla只存一份-cache)。
`prepare_ragged_batch()` 接 host-owned 布局；改变 Q 拓扑需重新准备，同拓扑更新 KV 长度后重新绑定。
变化长度可能在 MLX worker 上建立 managed 图，不意味着 capture 或零分配。

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
  count=1 arg reduction；浮点使用原IEEE storage的整数有序key，保留subnormal与源bits。
  count>1继续stable argsort后取k项，浮点共用该有序key；排序、key和必要布局转换计入成本。
  随机字与categorical的输入、数值差异和错误合同见[数值选择](compute.md#设备端单选与可定位随机数)。
  Graph 用 `add_node()` 与
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
DeepSeek 的 power2 scale 和1e-4 floor 由操作描述显式选择，dense 与 grouped 工厂均用
`(Float8ScalePolicy::kPowerOfTwo, 1e-4F)`。
`kMultiplyReciprocal` 则保 F32 `max*F32(1/448)`、`1/scale`、`A*inverse` 三步舍入，
供 GLM Quark block128 的参考使用；dense 与 grouped 均显式选择该策略。
grouped 权重保留全部专家的
压缩存储，仅对本次选择的专家解码；负值或越界 ID 通过 Completion 返回 OutOfRange，
不发布输出。原生 F32 运算允许 fused/unfused 舍入差异，不承诺与 CUDA 逐位一致。

FP8 点积通过 MLX Safe custom Metal 程序直接读取 packed codes/scales，在 tile 内解码并用
原生 SIMDgroup F32 矩阵运算累加。每128元素先点积，再乘 activation scale、weight scale，
最后按 K block 顺序加入 F32 结果；不生成完整 F32 权重或全部 block dots 临时张量。
dense和grouped均直接索引每128列共享的原F32 weight scale，不展开成逐列SF；
显式`Float8WeightDecode`仍输出调用者请求的完整浮点权重。
量化、选中 codes/scales、scale 广播、结果和必要复制仍由 MLX 管理并计入费用，
这不是硬件原生 FP8 GEMM 或零分配保证。编译/JIT 属于冷准备，Graph/native 完成与数组
寿命仍由原 MLX owner 管理。
Graph 捕获常量时的折叠由 MLX 管理，HYLE 不按模型文件或运行时地址建立全局权重缓存。
FP8 group64/128 roundtrip 也可用。FP4 group32 roundtrip接BF16二维输入，保留原E2M1
最近偶数舍入、power2 scale/floor及最终BF16存储；极小值和零符号由GPU位视图适配保真。
`model_ops::grouped_float4_linear()`消费canonical原E2M1 nibble/F32 group32专家权重、
BF16 activation和I32 IDs；activation按原group128 E4M3FN量化，F32逐block累加后存BF16。
重复ID合法，越界在Completion返回OutOfRange，不发布失败结果；它与MXFP4格式不同。
本批已完成普通入口、资源正确性及最终三交付。

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
IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、IQ1_S、IQ1_M、MXFP4、NVFP4，
以及原Anemll IQ1_XS/IQ1_XXS/IQ1_XXXS的二维 MatMul 与 `[E,N,K]` GroupedMatMul。
后三类保留原MV程序；无原MM时按原MV承载，不调用不存在的kernel。checked embedding
接入上述前19类；窄IQ1三类没有原GetRows，不声明支持。Q4_K精确行选择使用原ggml
F32 row-decode公式的MLX GPU承载，只取所选行，避免原Metal half高nibble精度损失。
准备期调用固定上游 capability 检查；设备/形状不支持时明确拒绝，不展开到 F32 或转去 CPU。
这些格式与前述 MLX affine Q4 的 `uint32 + scale/bias` 合同不同，不能交换 carrier。
GGUF NVFP4 保持原 64 值／36 字节编码，MatMul、GroupedMatMul 和 checked embedding 均直接读取
原压缩权重；激活、点积累加和输出为 F32。准备期编译实际原生变体，并检查线程数、SIMD 宽度与
线程组内存；设备资源不支持返回 `Unimplemented`，编译失败返回 `Unavailable`，分配失败返回
`ResourceExhausted`。不建立整库浮点权重副本。实际数值与寿命范围见
[本轮验证](../evidence/correctness/apple-gguf-nvfp4-2026-10-20.json)。

普通 `GroupedMatMul` 的共享激活合同见[Compute](compute.md)：`[T,1,K]` 或零 slot 步幅的
`[T,S,K]` View 都可供各专家读取，原独立分支输入仍为 `[T,S,K]`。用 `apple::input_spec(desc)`
保留实际描述；绑定 array 的 dtype、shape 和活动轴步幅须与准备时一致，原 array owner 保活到完成。
该共享 View 直接导入原 backing，不先物化 S 份激活；输出的原生附加存储由后端按真实需要分配。

### 模型制品的 native packed 与冷准备

`model_ops::MlxQuantized{format, kind, bits, group_size}`承载线性、选中专家和 affine checked embedding。
affine 支持 bits4/5/6/8、group32/64/128；MXFP4 固定 bits4/group32，MXFP8 固定 bits8/group32。
使用 `mlx_quantized_matmul(format,bits,group)`、`mlx_grouped_matmul(format,bits,group)` 或
`mlx_quantized_embedding(bits,group)`；实际 scale dtype、bias 与端口须匹配公共 schema。
每行 U32 words 是低位在先的连续位流，末轴为 `(K/32)*bits`，5/6 位码允许跨 word；
MX 的 U8 E8M0 scales 与 integer affine8、GGUF 或官方 F32 scales 不可互换。
普通 prepare 和 Graph 接原 MLX qmm/gather_qmm/take/dequantize，无需选择内部 Provider。

affine 点积将 activation/scales/bias 扩宽到 F32，保留 packed weights，最后按声明输出 dtype 舍入。
Graph中的不可变F16/BF16 scale／bias常量会在冷编译时提升一次，同一SSA来源去重；原logical类型与公开输出不变。
两种Graph入口共用该准备，普通输入／可换代WeightSlots仍按原就绪边界执行。公共Graph的派生存储计入constant预算，
实际数组owner保有reservation；[4项数值、预算与寿命检查](../evidence/performance/apple-affine-constant-preparation-reference-2026-10-21.json)通过，不代表吞吐或物理分配测量。
[Air 上 5 项 Direct 回归](../evidence/correctness/mlx-affine-widths-apple-2026-10-19.zip)覆盖定向位宽、
跨 word、分组及既有数值反例；本次没有运行 Apple Graph 或完整制品。
原生 decode 乘加可能收缩，QMV 可能重排 scale/bias 与归约；typed embedding 的一般两次独立舍入、
跨后端逐 bit 等价及完整模型数值意图仍未证明。

NF4/nested NF4、AWQ GEMM4和compressed-tensors symmetric/asymmetric4使用
`model_ops::artifact_weight_decode(format, group_size, output_type)`在GPU冷准备。
输入按[公共端口合同](../../include/hyle/compute/model_ops.h)给出真实packing/scales/zeros/
codebook；结果[N,K]保留为ready常量，再用于成熟MatMul或Gather。HYLE不加载模型、不按
文件名猜格式，也不把该准备放在每次decode内。原NF4 codebook/nested offset、AWQ列顺序
和zero、CT减8及源存储舍入均保留。
CT asymmetric4 的 zero 沿输出行打包为 I32 `[ceil(N/8),K/group]`，与 AWQ 的列排列不同；
源 dtype 的减法、乘法分别舍入，再转换到请求的输出类型。它仍是显式冷解码，不是 packed INT4 MatMul。

这些接口表达具体张量与模型片段，不负责完整模型常驻；准备前仍须核对内存容量与实际设备。

## 原版 Gated DeltaNet

`context.prepare(GatedDeltaNet{}, inputs)` 冷绑定锁定的 ggml Metal 完整递归程序。当前接入域为
F32、相同 Q/K/V head 数、D=32/64/128、1≤T≤32767；输入及 packed 输出元素数不超过 INT32_MAX。

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

## Graph 整段编译（原生 expert）

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

以下是原生 Direct `Program`／`Invocation` 的合同；普通 Graph 的槽、完成与结果合同见下节。

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
格式合法的实现。默认 deployment 14.4 的构建未启用 NAX；其它硬件特性按实际构建与设备检查。

## 可选模型与阶段信息

可选[模型与阶段信息](compute.md#可选模型与阶段信息)作为最后一个参数传给
`Context::prepare(operation, inputs, workload)` 或 `Context::compile(region, inputs, workload)`；
普通 Graph 使用 `graph::CompileOptions::workload`；原生整段编译使用
`graph::apple::CompileOptions::workload`。原生 `Program::workload()` 保留绑定身份，
Graph 的 typed lowering 收到同一份冷信息。无注解继续使用合法 MLX baseline；这不声明新增了模型
专用算法。改变身份后重新编译，MLX replay 不读取模型信息或重选规则。

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

报告的快照、寿命、复用与失败规则见 [Graph 原生路线报告](graph.md#原生路线报告)，
下列字段保留本后端特有的成本边界。

`declared_constant_count` 与 `logical_constant_bytes` 按声明 views 统计，重复引用也重复计数；
它们不是实际驻留内存。`prepared_constant_count`／`prepared_constant_bytes`另记去重后的F32派生常量及逻辑载荷，
不与原views混算；公共Graph报告使用已有prepared/packed常量字段。这两个原生报告字段扩展了结构布局，库和使用该结构的消费者须一起重新编译。
`private_storage_bytes` 保持 unknown，MLX 内部融合、私有 heap 和 driver
分配不能从逻辑 shape 精确推断。

主机调用跟踪的开启方式与解释边界见 [Graph 跟踪](graph.md#可选主机跟踪)。

## CPU/Apple组合与权重片段迁移

选择 `HYLE_COMPUTE_ROUTES="cpu;apple"`，使用 `hyle/graph/cpu.h`、`hyle/graph/apple.h`
和普通 Graph API。`graph::apple::execution_source(apple_context)` 保留实际 MLX Runtime，
`graph::ExecutionSource::acquire(cpu_context)` 保留实际 CPU Context；将二者及主存／Apple
存储域传给 `graph::compile`。需要固定 CPU／Apple 分工时显式设置 Required placement，
不能把未约束的默认选择当作 GPU 路由。

编译结果仍是原 `Executable`。只给报告中 `queue_required` 的 shared CPU lane 绑定真实
CPU Queue；Apple lane 使用原 Runtime/stream。需要异步主机推进时选择 Sched，冷调用
`prepare_progress`，Scheduler 推进同一槽的依赖前沿，BlockingPool 只执行实际阻塞的粗片段：

```cpp
namespace g = hyle::graph;
auto compiled = g::compile(graph, environment, options).value();
auto run = compiled.executable.create_invocation({
    .resources = g::ResourceOptions{}}).value();
HYLE_RETURN_IF_ERROR(g::prepare_progress(run, scheduler, workers));
HYLE_RETURN_IF_ERROR(run.bind(x, input_description, input_lease));
HYLE_RETURN_IF_ERROR(run.bind(y, output_description, std::move(output_writer)));
HYLE_RETURN_IF_ERROR(run.prepare_replay(queues));
auto done = run.submit(queues).value();
HYLE_RETURN_IF_ERROR(done.wait());
auto result = run.result(y).value(); // GraphOutput
auto view = result.tensor().value();
```

CPU caller 输出需绑定实际存储并保活；Apple generated 输出保留原生数组 owner，成功后才可观察，
不复制到影子输出。`GraphOutput` 保留原 Invocation；`output()` 返回借用视图，原生 expert
`OwnedResult<Results>` 则属于前述 Direct 接口。旧结果未释放时不能重绑、重置或重放同一槽。

权重仍用 `compute::apple::WeightSlots` 的固定 catalog／双槽上传；普通 Graph 调用
`graph::apple::bind(run, weight_id, batch, catalog_id)` 保留原 `WeightBatch` 与上传来源，
无需调用者先 wait。`WeightBatch::binding` 是另一个只发布 ready array 的原生访问入口。
修改绑定后显式重新 `prepare_replay`；固定布局和绑定可严格重放，不在热提交中重选实现。
分页布局更新按[批次 attention](batch-attention.md#同一分页图用于-cpu-与-gpu)执行冷准备。

`Completion::request_cancel()` 请求停止后继，已接纳工作仍须实际退休；`Invocation::close()`
只停止该槽的新接纳，`drain()` 等待该槽退休并返回首错，不关闭调用者的 Context/Queue。
Scheduler、BlockingPool 须活过已接纳工作和 drain；资源及 allocator 还须活到最后一个保留结果
或导出 lease 释放。关闭 Apple Context 会排空原 Runtime；成功结果仍可读。
原生 host frontier 可由 `HostWaitPolicy::kReject` 冷拒绝。

`ResourceBudget` 按 requested bytes 管理已知 backing、metadata 与旧 lease；MLX 私有分配
需要显式 `allow_unattributed=true`。原生 backing 观察须按真实 owner 去重，不能再加到 scope usage。
managed 图、数组和 Scheduler 任务仍有实际成本，不承诺零 warm allocation 或必然 CPU/GPU 重叠。

[Direct样例](../../samples/package/compute/apple_weights/README.md)和
[完整CPU→Q4 FFN→CPU/正交分块样例](../../samples/package/graph/cpu_apple/README.md)
使用同一源码/static/shared入口与`hyle_stage_runtime`，不手配Provider、metallib或动态库。
引擎管理model/KV、offload比例和专家cache/预测/淘汰；本接口只提供放置、预算、迁移和寿命。

## 原生只读输出到固定 copy slot

成功的原生输出可经 `MemoryResource::import_readonly` 导出为独立永久只读 lease。
冷准备 `CopyBatchProgram` 时以原 `memory.ref()` 与最大来源字节数声明 source endpoint，
每步可将新的导出 `ReadLease` 绑定到同一 `CopyBatchInvocation`，无需按导出的 resource
identity 重建 program/slot。`bind_source` 只接受原资源或受信 MLX readonly 导出家族；
仅伪造相同 domain、host accessible 或来自其它普通 resource 均不成立。

导出实际拥有者和精确 subview 会由批量搬运 binding 保留；每次 replay 仍校验容量、别名和
真实 Metal device，可写 endpoint 仍须精确资源身份。`reset`/重新绑定与关闭沿原实际
退休规则清理持有者。readonly import 自身仍有冷 metadata/opaque MLX descriptor 成本，
原 `MemoryResource` 仍需活到所有 program/slot/提交退休，不能经其它 MLX alias 改写源。
这里消费成功 ready 输出，没有隐式求值、producer 等待或 pending 结果转换。
