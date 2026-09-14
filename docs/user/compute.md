# HYLE Compute：何时直接执行一个计算

> 本页描述当前 Compute runtime foundation、CPU operation surface 与 CUDA runtime；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。默认 CPU façade 使用 HYLE-native；
> 选择 `HYLE_COMPUTE_ONEDNN=ON` 时，受支持的 x86-64 构建还提供显式 oneDNN MatMul provider。
> 已关闭的 CUDA factor 提供 device/context/queue、memory resource、stream interop、byte transfer、Completion 与
> command capture 基础运行时，但尚未签发通用 CUDA operation provider。Apple MLX/Metal、Vulkan/SPIR-V
> 与其它 provider 是后续独立纵切。HIP/ROCm 当前无限期不纳入产品，也没有空实现或构建开关。

当程序已经知道要执行哪个 operation、在哪个 device 上执行，并希望复用固定 shape/layout 的路线时，
使用 Compute。跨节点拓扑、liveness 和可复用图执行物使用 [Graph](graph.md)；模型、KV cache、batching、
sampling 和 serving 留在推理引擎。

direct `prepare()` 仍接收具体 TensorDesc，不要求用户构造 shape program。需要 batch × tokens、concat
输出轴等派生维度时，使用 [Graph 的独立与派生维度](graph.md#独立与派生维度)，在冷 compile 时解析；
当前 Native/oneDNN artifact 仍是 exact-only，不能从私有 shape-domain 接口推断已有整个动态范围的执行器。

## 当前能力

| 任务 | 公开形状 | 当前合同 |
|---|---|---|
| 设备发现 | `Backend::device_count/device` | 每个显式 backend 只枚举自己的设备；`cpu::device()` 直接取得唯一 CPU |
| 执行域 | `Backend -> Device -> Context -> Queue` | CPU Queue 在调用线程同步、按序执行；没有全局 registry、隐藏 worker 或默认设备 |
| tensor schema | `dense/strided_bytes/blocked_dense/blocked_strided/opaque` | rank <= 8；logical dtype 与 physical encoding 分离；stride 明确以 byte 为单位；view 借用 Mem storage |
| operation schema | 19 个 typed family + inline custom | 可做 arity/shape/layout/alias/effect/precision 检查；schema 存在不等于已有可执行 provider |
| 零拷贝 tensor 变换 | `View::reshape/slice/permute/broadcast_to` | submit 不映射或搬运数据；输出必须绑定到输入中由 view 推导出的精确起始位置 |
| materialize / 初始化 | `Copy` / `Fill` | Copy 支持 plain dense/strided 输入并产生 dense 输出；Fill 对全部已登记 scalar dtype 写入精确 bit pattern |
| 推理浮点 storage 转换 | `Cast` | f32/f16/bf16/E4M3/E5M2 的完整互转；dense 复用 SIMD whole-buffer 路线，strided 输入显式 materialize |
| 单目计算/激活 | `Unary` | f32 Abs/Neg/Exp/Log/Sqrt/Rsqrt/Tanh/Sigmoid/ReLU/GELU/GELU-tanh/SiLU；dense 一次 SIMD，plain stride 可 materialize |
| 双目计算 | `Binary` | f32 Add/Sub/Mul/Div/Min/Max 与 NumPy-style broadcast；Min/Max 使用 `fmin/fmax` 语义 |
| 条件选择 | `Select` | 可广播 Bool condition；对任意 plain 1/2/4/8-byte scalar branch 原样选择 bits |
| reduction | `Reduce` | f32 Sum/Mean/Min/Max；任意 axis 集、负 axis、keep-dim 与普通 byte stride；dense 常用路线复用 SIMD |
| normalization | `RmsNorm` / `LayerNorm` | dense f32 `[..., hidden]`；RMSNorm 可选 weight；LayerNorm 可选 scale/bias；逐 row、允许精确 input/output in-place |
| matrix multiply | `MatMul` | f32 `[...,M,K] * [...,K,N]` 与 NumPy-style batch broadcast；两端 batch 都可广播，B 的末两维可 row-major 或物理 `[...,N,K]` 的转置 view |
| gated activation | `GatedActivation` | dense f32 packed `[...,2*d]` 或 separate gate/value；SiLU、exact GELU、GELU-tanh 三种模式均产生 `[...,d]` |
| 索引读写 | `Gather` / `Scatter` | plain scalar storage、i32/i64 indices、dense output；Scatter assign 支持任意标量位宽，add 当前为 f32，并可精确原地更新 input0 |
| 张量拼接 | `Concat` | 同 dtype/rank 的 variadic plain inputs，沿显式 axis 产生 dense output；dense 输入按连续后缀块复制 |
| 旋转位置编码 | `RoPE` | dense/strided f32 data、i32/i64 positions、显式 frequency tensor；支持 interleaved/half-split 与未旋转 tail，dense route 冷选择 fat kernel |
| 概率归一化 | `Softmax` | f32 任意 axis、scale、LogSoftmax，以及可广播 Bool/additive mask；允许精确 input/output in-place；无有效值的 row 产生 NaN |
| 候选筛选 | `TopK` | f32 任意 axis，largest/smallest、sorted/stable 语义，返回 values + i32 indices；显式 caller-owned workspace，不隐藏分配 |
| 注意力 | `ScaledDotProductAttention` | f32 rank-4 BSHD/BHSD，Q/K depth 与 V/output depth 可不同；可选 bottom-right causal、Bool/additive 广播 mask，分块 online softmax 与显式 caller-owned workspace |
| 自定义 CPU 工作 | `cpu::submit_host(queue, callable, args...)` | `noexcept` callable，返回 `void` 或 `Status`，返回前完成 |
| CUDA runtime | `cuda::backend/device/MemoryResource/submit` | 显式 device/pinned/managed storage、stream borrow/adopt、typed command、真实 async Completion；不冒充 CUDA 算子目录 |
| byte transfer | `transfer_properties/copy_async/fill_async` | 对 Mem `BufferView` 做显式 byte copy/fill；不隐藏 tensor repack、pageable staging 或 host wait |
| 完成边界 | `Queue::reserve_completions` / `Completion` | direct submit 或一个 Graph Queue segment 只产生一个 prefix Completion；销毁 pending handle 不等待 |
| provider 选择 | `Context::create(device, Provider)` / `ContextOptions::providers` | 显式且有序；只有 `Unimplemented` 才尝试下一项，不存在隐式 fallback |
| 精确资源 prepare | `OperationResourceBindings` / TensorView overload | provider 先报告要求，再对 input/output/workspace 的 exact `MemoryResourceRef` 冷验证；submit 不重新选资源或隐藏 copy |
| provider 诊断 | `provider_identity` / `runtime_properties` / `persistent_bytes` | O(1) 读取 provider/version、shared/delegated owner、replay 成本类别、threading/capture/workspace owner；`persistent_bytes == nullopt` 表示 provider 无法给出完整精确值，不是零 |
| 稳定重复执行 | `prepare -> PreparedOperation -> OperationInvocation::submit` | prepared artifact 可共享；每个并发 lane 独占 invocation/workspace；submit 不重新推 shape/路由 |

公共 schema 已能描述 View、Copy/Fill/Cast、elementwise、reduction、normalization、batched MatMul、
gated activation、Gather/Scatter/Concat、RoPE、Softmax、TopK 和 SDPA 共 19 个 family，并能做纯描述推导。
**当前 HYLE-native provider 已签发上述 19 个 family 的 CPU executable route。** 这不表示每种 dtype、
layout 或 encoding 都由同一个 provider 执行：当前 SDPA 是 f32 reference/optimized fallback；block/opaque
storage、量化 kernel 与窄 storage attention 仍应由理解相应格式的 provider 接管，不会被隐藏 materialize。

CUDA runtime 与 CUDA operation provider 是两层。选择 `HYLE_COMPUTE_CUDA=AUTO` 时，CMake 仅在当前目标能
找到用户安装的 CUDA Toolkit 时启用；`ON` 找不到即配置失败，`OFF` 不探测。HYLE 不下载 CUDA Toolkit。
启用后可以创建 CUDA Context/Queue、device/pinned/managed `MemoryResource`，借入或接管同 primary context
的非默认 stream，并通过 `cuda::submit()` 录入 typed stream command。Graph 可在冷路径 capture 兼容的
CUDA command/copy segment；这不意味着 `prepare(cuda_context, MatMul{...})` 已有可执行 CUDA route。
同一 CUDA Context 的不同 Queue 可以用 device event 导入另一个 Queue 的 Completion prefix，不阻塞提交
线程；跨 context/backend 不会伪装成这条路线，Graph 只能依显式策略选择 host wait 或拒绝。

第三方源码出现在 dependency cache 或最终链接中，不表示其算子已经自动成为 HYLE route。HYLE
`Operation` 只描述公共语义；一个已交付 Provider 必须在 `prepare()`/Graph compile 冷路径把它翻译成
第三方 plan、kernel 或 graph artifact。叶函数型 provider 可以产生 `PreparedOperation`；自己 host 图的
runtime 通常只能由 Graph 作为 compiled region 使用。当前没有已交付 ggml provider，因此合法的 block-Q4
`TensorDesc` 只证明格式可表达；`prepare(cuda_context, MatMul{}, ...)` 仍返回 `Unimplemented`，不会把
压缩 bytes 当普通 scalar、偷偷反量化或因为 ggml 已下载就调用私有 kernel。

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
前项，add 依次累加。需要并发原子 scatter 的算法应选择具备该明确合同的 provider，不能把本 CPU reference
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
Unary/Binary 当前 arithmetic route 是 f32；Select 不做数值转换，因此可选择全部已登记的普通 1/2/4/8-byte
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

`Reduce` 的 typed `prepare()` 根据 axis 和 `keep_dimensions` 推导 dense output。当前 Native route 支持 f32
Sum/Mean/Minimum/Maximum；任意普通 byte-strided input 都按逻辑坐标读取，dense reduce-all 和最后一轴
四种 operation 都走已有 SIMD reduction。空集合约定为 Sum=0、Mean=NaN、Minimum=+infinity、Maximum=-infinity；
Min/Max 遇到任一 NaN 会传播 NaN。Reduce 不允许 output alias input。

`LayerNorm` 对最后一轴逐 row 计算，可不带参数、只带 scale，或同时带 scale+bias；三种 typed prepare 都
自动推导与 input 同形的 dense output。output 可以精确 alias input，但不能 alias scale/bias。当前只签发
dense f32 storage 与 f32 accumulation；f16/bf16 等窄 storage 由后续 provider 资格化，不发生隐藏 Cast。

```cpp
auto reduced = prepare(context,
                       Reduce{.kind = ReductionKind::kMean,
                              .axes = Axes::all()},
                       input_desc);
auto normalized = prepare(context, LayerNorm{.epsilon = 1.0e-5F},
                          input_desc, scale_desc, bias_desc);
```

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

## 何时选 oneDNN

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

上例只有 oneDNN；不支持的形状会返回 `Unimplemented`。需要回落到 native 时必须把顺序写出来：

```cpp
#include <array>
#include <hyle/compute/cpu.h>
#include <hyle/compute/providers/onednn.h>

const std::array providers{onednn::provider(), native::provider()};
auto context = Context::create(
    cpu::device(), ContextOptions{.providers = providers});
```

`Context::create()` 在冷路径把完整 provider 序列复制到 context 自己的一次 metadata allocation；输入
span 只需活到调用返回。序列没有人为四项上限，也不会自动追加 native。通过
`PreparedOperation::provider()` 返回的开放 `ProviderIdentity` 可以检查实际路线；是否发生组合回落由
调用方给定的顺序和最终 identity 共同解释，不再用固定枚举或含义模糊的 `used_fallback` 标志。

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

该选项默认 `OFF`，所以普通 Compute 配置不会下载或编译 oneDNN。构建时
`HYLE_COMPUTE_ONEDNN=ON` 在受支持的 x86-64 目标强制启用，在 AArch64 等不支持目标会明确配置失败；
`AUTO` 只适合本地探索并在不支持目标关闭能力。可复现的 x86-64 oneDNN 制品应显式选择 `ON`，不需要
它的制品保持 `OFF`。`PreparedOperation::provider()` 与 `workspace_requirement()` 可解释实际路线。
oneDNN 类型不会泄漏到 Tensor、Operation 或 Graph API。

## 所有权、失败和限制

- `Backend`、`Device` 和 `Provider` 是廉价、可复制的进程内值；它们以开放的双字 ID 标识实现，不要求
  修改中央枚举。`Context`、`Queue`、`OperationInvocation` 与 `Completion` 是 move-only；
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
- 公开 direct prepare 只考虑 shared-Queue provider。delegated provider 由 Graph 在 compile 冷路径选择，
  必须提供真实 endpoint/dependency/Completion/drain 合同；它不会假装在传入的 Queue 上执行。
- Tensor views 不拥有 payload。Mem Buffer/resource、调用方 workspace 以及 Queue 必须活到 Completion 终态。
  direct op 和 Graph 都把自己的 invocation/provider state owner 附着到最终 Completion；这种内部附件不改变
  caller tensor、workspace 或 Queue 的借用合同。
- `StorageFormatId` 是跨 source/static/shared 稳定的数值 identity，不是 Native FFI 承诺。自定义 provider
  应以自己的 typed factory 统一发布 identity；同一 identity 不得在升级后改变物理解释。
- typed custom `Operation` 的 `OperationId` 同样标识完整 schema/version 和 inline descriptor layout；改变字段、
  大小或语义必须换 ID。`custom_id()` 可作冷路径分派/诊断，`custom_as<T>()` 以该 ID 和精确 descriptor
  字节数校验，因此 producer 与 consumer 可以位于不同 source/static/shared factor；不要复用 ID 模拟兼容。
  这只是 schema 描述/识别，不会自动提供执行实现。目前安装包尚无通用的 owning custom-kernel factory
  可把用户实现接入 Graph；已接受的补齐方向见 [custom execution 设计](../architecture/compute-design.md#61-安装包消费者的-typed-custom-kernel-路线)，
  不要通过包含 `src/` 私有 Provider SPI 模拟公开扩展。
- admission 前的参数、shape、alias、context/queue mismatch 由外层 `Result` 报告；接纳后的 mapping、coherence
  等失败由 Completion terminal `Status` 报告，两层都要检查。
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
