# HYLE Compute：准备并执行计算

已知 operation、设备、形状和布局，需要对变化输入重复计算时使用 Compute。
需要编译多节点拓扑、规划中间存储或组合 CPU/GPU 片段时使用 [Graph](graph.md)。

普通入口负责组装所需实现，用户不排列 Provider 或第三方库。通用 CPU/CUDA 接口使用
Context、Queue、PreparedOperation 和 OperationInvocation；[Apple](apple-compute.md) 与
[Vulkan](vulkan-compute.md) 还提供各自的原生 Program/Invocation，遵守对应资源与完成合同。
构建后端的方法见[使用指南](usage-guide.md#开始使用-compute--graph)。

建议按任务阅读：

- 首次接入：下面的 Direct 例子，再看 [tensor/storage](#普通block-与-opaque-storage) 与
  [prepare/submit](#prepareinvocation-与-submit)。
- 需要具体算子：查看[能力与公共描述](#当前能力)，再核对后端支持的 dtype、布局与资源。
- 资源复用：看[租约](#外部资源租约与-direct-提交)、[批量搬运](#准备批量范围搬运)与
  [文件管线](#有界文件与资源管线)。
- KV/多请求：看[批次 attention](batch-attention.md)；快照和状态分支见[状态版本](state-versions.md)。
- 诊断或扩展实现：最后阅读显式 Provider、自定义 kernel 和资源访问章节。

## 第一次 Direct 计算

以下 CPU 例子对四个值做 RMSNorm。它演示公开样例使用的完整顺序：建立描述与存储、冷准备、
准备工作区和执行槽、提交、等待成功、读取结果。代码可放在返回 `hyle::Status` 的函数中：

```cpp
#include <array>
#include <span>
#include <utility>
#include <hyle/compute/cpu.h>
#include <hyle/compute/prepared_operation.h>
#include <hyle/error/result.h>
#include <hyle/mem/buffer.h>

namespace c = hyle::compute;
const std::array<std::int64_t, 1> shape{4};
const std::array<float, 4> values{1, 2, 3, 4};
std::array<float, 4> result{};
const auto resource = hyle::mem::system_resource();

HYLE_ASSIGN_OR_RETURN(auto desc, c::TensorDesc::dense(c::DataType::kFloat32, shape));
HYLE_ASSIGN_OR_RETURN(auto input, hyle::mem::Buffer::create(resource, sizeof(values), alignof(float)));
HYLE_ASSIGN_OR_RETURN(auto output, hyle::mem::Buffer::create(resource, sizeof(result), alignof(float)));
HYLE_RETURN_IF_ERROR(input.view().write(std::as_bytes(std::span(values))));
HYLE_ASSIGN_OR_RETURN(auto x, c::ConstTensorView::create(desc, input.view()));
HYLE_ASSIGN_OR_RETURN(auto y, c::TensorView::create(desc, output.view()));

HYLE_ASSIGN_OR_RETURN(auto context, c::cpu::create_context());
HYLE_ASSIGN_OR_RETURN(auto queue, context.create_queue());
HYLE_ASSIGN_OR_RETURN(auto operation, c::prepare(context, c::RmsNorm{}, desc, resource));
const auto required = operation.workspace_requirement();
hyle::mem::Buffer workspace;
if (required.size != 0) {
  HYLE_ASSIGN_OR_RETURN(workspace,
      hyle::mem::Buffer::create(resource, required.size, required.alignment));
}
HYLE_ASSIGN_OR_RETURN(auto slot, operation.create_invocation(workspace.view()));
HYLE_ASSIGN_OR_RETURN(auto done, slot.submit(queue, x, y));
HYLE_RETURN_IF_ERROR(done.wait());
HYLE_RETURN_IF_ERROR(output.view().read(std::as_writable_bytes(std::span(result))));
// result 现在可读；函数按自己的需要消费或返回它。
```

保持描述和 resource identity 不变时，复用 `operation`、`workspace` 和 `slot`，更新输入后再次 submit。
等上一轮完成后再覆盖它使用的存储；并发请求各有独立 slot 和工作区。共享计划不保证共享可变槽安全。
完整工程与 CPU/CUDA/Metal/Vulkan 示例见 [Direct sample](../../samples/package/compute/cpp/)。

## CPU 与模型格式接入

普通入口是 `compute::cpu::create_context()`；Graph 使用 `graph::cpu::create_context()`。
CPU 数学依赖自动装配，按实际请求选择合法实现；不要求传性能 profile 或排列 Provider。
模型加载与 tensor 角色、scale/codebook/zero 的解析由引擎完成，HYLE 不根据模型文件名或“Q4”标签猜格式。

`model_ops::dense_linear_f32()` 接受 F16／BF16／F32 激活 `[T,K]` 和按输出行存储的
F16／BF16／F32 权重 `[N,K]`，以 F32 乘积、累加与输出产生 `[T,N]`。
F32 激活保持来源精度，权重在读取时扩宽，不展开整份 F32 权重；具体类型组合与形状由各 Provider 冷准备检查。
它也接受同批数的 `[B,T,K]`、`[B,N,K]`，输出 `[B,T,N]`，不隐式广播批维。
CPU／CUDA 已检查批量 F32 激活路径，[Mac CPU与Metal下沉检查](../evidence/correctness/mtp-batched-f32-bfloat-apple-2026-10-24.json)也已通过；
这些既有记录覆盖 BF16 权重而不展开整份 F32 副本。CUDA多批权重目前要求`N<=512`、偶数`K`。
[Vulkan批量检查](../evidence/correctness/mtp-batched-f32-bfloat-vulkan-2026-10-24.json)已核F32低位、批权重偏移、输入保持及冷描述符；
ggml设备路径暂不承载新批量BF16激活域，Vulkan会在prepare时明确拒绝该域，
其它Provider仍以各自冷准备结果为准。

扩展后的 F16／BF16／F32 组合分别有[CPU／Metal／Vulkan](../evidence/correctness/mtp-dense-linear-native-domains-2026-10-24.json)和
[CUDA／cuBLAS](../evidence/correctness/mtp-dense-linear-cuda-2026-10-24.json)检查。
cuBLAS 接受相同输入类型；混合类型由其它实际支持的 Provider 承接。
普通 `MatMul` 继续遵守其原有类型规则。

- 普通张量算子见下方能力表；GGUF 原编码使用[原生压缩权重](#ggufggml-原生压缩权重)，
  NF4/AWQ/CT/MLX 使用[非 GGUF 接口](#cpu-非-gguf-原格式与压缩专家)。
- `Float8BlockLinear`、`GroupedFloat8Linear`、`GroupedFloat4Linear`、编码 `MatMul` 和 roundtrip
  保留其量化、scale、累加与输出 storage 合同。CPU 可以用 F32 表达已量化值并计算，不代表硬件原生 FP8 GEMM。
- 某些原格式需要显式冷解码或转换并缓存；这与直接读取压缩权重的成本不同，不能每次 decode 重做全库展开。
- 原库 managed 程序可能建立临时 Tensor、使用私有线程并阻塞 host。内部内存上界未知时，严格预算或
  prepared-only 请求可以拒绝；Context 的 `native_threads` 不统一控制所有第三方全局线程池。
- 完整 CPU 产品支持 Windows/Linux x86_64 和 macOS ARM64；依赖获取、外部 SDK 复用与离线方法见
  [依赖准备](../developer/setup/dependencies.md)。静态 HYLE 仍可能依赖 shared CPU runtime，部署见
  [交付说明](distribution.md#compute--graph-后端)。

算子和状态接口可用于 MTP、MoE 或模型片段；模型 loader、tokenizer、视觉管线、
草稿接受策略和请求循环归项目内的引擎层，当前实现范围见[推理指南](inference.md)。
片段可执行不意味着整个模型能在当前内存中常驻，也不意味着全部新制品编码自动兼容。

## 当前能力

按任务选择 descriptor；完整字段和 shape 推导见[公开 operation 头](../../include/hyle/compute/operation.h)。
这张表是接口导航，具体请求还要通过 `prepare()` 检查，不是四后端通用支持矩阵。

| 任务 | descriptor / 入口 | 先确定什么 |
|---|---|---|
| 改逻辑布局、物化、初始化、转换 | `View`、`Copy`、`Fill`、`Cast` | byte stride、alias、copy 与 numeric cast 的区别 |
| 逐元素与广播 | `Unary`、`Binary`、`Select`、`Concat` | 模型规定的激活、广播和存储舍入点 |
| 归约、前缀、归一化 | `Reduce`、`Scan`、`RmsNorm`、`LayerNorm` | axis、空集合、累加精度、workspace |
| 矩阵、量化权重、专家 | `MatMul`、`QuantizedMatMul`、`GroupedMatMul` | 逻辑 shape、实际编码、scale/group 与资源 |
| 索引、top-k | `Gather`、`model_ops::GatherElements`、`Scatter`、`TopK` | 笛卡尔或逐元素索引、checked 边界、重复写语义与临时存储 |
| 位置与注意力 | `RoPE`、`Softmax`、`ScaledDotProductAttention` | 位置类型、mask、GQA、causal 对齐与数值域 |
| ragged/paged KV | [批次 attention](batch-attention.md) | 容量与有效区、页身份、更新和发布 |
| 模型专用片段与状态 | [model_ops.h](../../include/hyle/compute/model_ops.h) | 显式输入状态、原始 gate 与反馈边界 |
| 一次 CPU callback / 可准备 kernel | `cpu::submit_host` / [自定义执行](#自定义执行接入-direct-与-graph) | callback 不抛异常，kernel 有自己的执行实现 |

`GatedDeltaNet{}` 的 Q/K 是 `[B,T,Hk,Dk]`、V 是 `[B,T,Hv,Dv]`，三者共享 F16、BF16 或 F32 类型。
log-decay 是 F32 `[B,T,Hv,1]` 或逐通道的 `[B,T,Hv,Dv]`，beta 是 F32 `[B,T,Hv,1]`，
初态是 F32 `[B,Hv,Dv,Dk]`；输出为 V 同类型同形结果和新 F32 状态。
值头 `h` 使用 Q/K 头 `h % Hk`；模型若采用连续头分组，需显式展开 Q/K 或重排相关权重与状态。
默认 `snapshot_count=1` 保持最终状态的四维形状；设置 `1<K<=T` 时，第二输出为
`[B,K,Hv,Dv,Dk]`，按时间正序保存最后 K 个 token 后的状态，不含原输入状态。
`output_scale` 留空使用 `1/sqrt(Dk)`，设为 `1.0F` 返回原始 `S*q`；显式值必须有限，可为零或负值。
缩放发生在 F32 点积归约之后，不改变 Q/K 输入、状态或快照；融合入口用 `recurrent_gated_delta_net(K, scale)` 接收相同设置。
它不隐含 Q/K 归一化、gate 激活或状态写回。`model_ops::recurrent_gated_delta_net()` 另接受
raw a/b、negative_a、dt_bias，组合 `g = negative_a * softplus(a + dt_bias)` 与 `beta = sigmoid(b)`。
该组合入口的 raw a/b 为 F32 `[B,T,Hv,1]`，negative_a/dt_bias 为共享逐头 F32 `[1,1,Hv,1]`；
Q/K/V、状态与 snapshot count 复用普通 GDN 的数学合同，不限定某个模型的 batch、头数或 D128。
它仍是 scalar-gate 变体，不能把逐通道 raw gate 直接代入。数学描述合法不代表所有后端均可执行：
真实类型、布局、维度和设备限制由所选实现检查，不支持时在准备阶段拒绝。
调用者显式决定何时以 next state 替换旧状态；后续节点失败不会替应用回滚已写数据。
MTP 接受前缀时的选择、卷积历史与 KV 联合提交见[状态版本](state-versions.md#mtp-接受前缀并继续计算)。

模型专用 MoE、压缩 attention、FP8/FP4 和 MTP 片段仍按真实端口、编码和形状准备，
不会因模型名相同而继承其它接口的数值或设备支持。常用低精度接入见下文；
整模型加载、视觉管线、草稿接受和请求调度属于引擎。

## 有界专家权重工作集

CUDA、Apple、Vulkan的`WeightSlots`提供两个预分配槽。catalog注册本地bank的ID与原始`TensorDesc`，
`upload_slices(slot, slices, ...)`把分立host tensor的原字节放入bank；每个`WeightSlice`指定bank ID和字节偏移，
选中的bank须完整覆盖，不能重叠或留洞。来源须具有`HostAccessible`和`HostAddressIdentity`，并在同步staging期间保持就绪、不可变。
每次最多接受`WeightSlotOptions::max_fragments`份切片，默认64；容量与元数据在创建时准备，上传不扩容。
容量包含bank间必要的对齐间隙，后端负责安排bank起点；不需要修改权重tensor形状或原打包。容量不足在source staging前拒绝。
plain dense bank的源保持同一scalar类型；encoded bank的源是显式dense U8物理字节片段，仍按原bank描述绑定。
上传不解释codec、不转换scale，也不自动重排分立tensor。例如原FP4 bank先放全部本地专家codes，再放全部F32 scale尾区；
各expert的codes和scale须给出各自目的偏移，不能把每份`[codes+scale]`依次拼接。

模型global expert ID与本地bank行号分开。调用者选择驻留专家，生成global→local表，将表与W、scale放入同一个batch，
再用`graph::<backend>::bind(run, span<const graph::WeightBinding>, batch)`原子绑定。普通checked Gather把global IDs映射为local IDs，
缺失映射或越界ID由原计算路径拒绝。WeightSlots不决定缓存淘汰、选专家或模型策略。
若编码以bit而非byte对齐，调用者须显式冷打包；例如CT各expert独立补齐的INT4 zero行，
使用`pack_int4_zero_rows()`去除每个expert的尾padding后再上传，不能盲拼。

`batch.report()`／`slots.report(slot)`区分logical、完整成功staging、原生已接纳与成功退休字节；
这是上传范围记账，不是总线流量测量。`copied_bytes_unknown`为真时，零completed不代表未搬运。
batch、绑定它的Invocation、旧GraphOutput或导出的lease仍存活时不能覆盖相应槽；Context、资源与allocator遵守原借用寿命。
[三种GPU的真实工作集验证](../evidence/correctness/expert-working-set-reference-2026-10-21.json)
覆盖512专家ID域中仅装载4个专家、NVFP4真实N2048/K8192、CT奇数行打包、双槽及失败恢复；
CPU另验共同装配、冷打包和原子绑定。其它codec的精确角色旅程、制品payload、整模型和吞吐仍需分别核实。
[原FP4编码分片旅程](../evidence/correctness/encoded-bank-slices-reference-2026-10-21.json)另覆盖三GPU实际N2048/K4096、7／256专家、同代I32映射、1字节拆分、双槽旧结果和失败恢复；host检查还覆盖FP8和block物理布局。[完整GA routed／shared FFN消费者](../evidence/correctness/ga-full-ffn-reference-2026-10-21.json)已四端沿同一WeightBatch执行普通／hash路由、down前route权重与旧结果寿命，共24次成功提交；范围为有限合成权重，不代表整制品执行。
[GLM Flash混合MLX消费者](../evidence/correctness/glm-flash-mixed-mlx-reference-2026-10-21.json)覆盖g64、原F16 metadata与4／5／6bit角色、每代92份分立片段、双槽三代、同代映射及失败恢复；四端共16次成功提交。另修CUDA whole／sliced bank对齐，保原容量与字节合同；未测整制品或性能。

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
CPU 低精度 Add 使用相应的累加程序；具体实现由准备期绑定。Scatter本身不提供不同invocation之间
对同一状态的并发原子更新保证，调用方仍负责资源排序。

MoE按每个token选择自己的路由分数时，用 `model_ops::gather_elements()`：输入为
scores `[T,E]` 和I32 indices `[T,K]`，输出 `[T,K]`，满足
`output[t,k]=scores[t,indices[t,k]]`。重复索引合法，K也可以大于E；不要用普通
`Gather{.axis=1}`，它会产生 `[T,T,K]` 的笛卡尔结果。该操作保留F32/F16/BF16或I32数据原码，
相同rank的非axis尺寸必须一致，不广播、不环绕负索引。CPU可使用任意合法axis，当前GPU承载为
rank2的末轴。默认Checked；若索引来自同一行的TopK，可显式使用
`gather_elements(-1, IndexBoundsPolicy::kUnchecked)`，沿Graph传递device IDs。
I32候选表可直接保存偏移后的ID或数据值`-1`，不经过浮点转换；Checked检查的是indices，data中的负数合法。
CUDA默认图已接通CCCL TopK→ggml GatherElements；带校验的ggml专家/embedding也在GPU上生成
safe IDs并在原Completion报告错误，避免读回整组ID。具体路线仍可能观察少量完成状态，不能据此推导整图完全异步。
默认CUDA路线的普通`Gather`也在GPU校验索引，越界由`Completion`终态报告；
调用方无需为了避免索引上下界回读而关闭边界检查。
ggml CUDA现可对普通F32／F16／BF16二维表按axis0（或−2）选行，支持连续行和leading row padding，
输出保持原类型及存储位；行内stride仍不在该实现范围。它仍检查任意输入ID，
不授予整个状态池的读取权限；[有限检查](../evidence/correctness/inference-cuda-slot-gather-2026-10-26.json)
覆盖NaN／零符号、重复选行、重放、padding及越界不改输出。

## 普通、block 与 opaque storage

普通用户继续用 `TensorDesc::dense()`；它现在返回 byte stride，例如 dense f32 `[2,3]` 的 stride 是
`[12,4]`。非连续普通 storage 使用名称明确的 `strided_bytes()`。旧的 `strided()` /
`element_strides()` 已删除，没有把旧名称悄悄改成新单位的兼容层。

`View::reshape` 是零拷贝操作：普通 strided tensor 可在连续 stride chunk 内拆分/合并维度，
例如 QKV 子行保留原 token pitch 后拆成 head/depth。跨存储间隙的 flatten 返回 `Unimplemented`；
确需连续化时先显式 `Copy`，再 reshape。此能力不扩大 Graph 的动态全域布局证明范围。

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

`RoPE` 可自然表达单head `[B,S,D]` 与多head `[B,S,H,D]`，序列轴分别可用1／-2与1／-3；
不必为压缩器人为增加head轴。只旋转尾部时先切出尾部再拼回；模型明确要求F32旋转时，
以`Cast`表达升至F32、旋转后回到原存储类型的数值边界。

## View、Copy、Fill 与 Cast

`View` 是显式零 kernel operation：prepare 推导新的 descriptor，submit 只验证输出 Mem view 是否指向输入
allocation 的正确 byte offset。reshape、permute 和 broadcast 从输入起点开始；slice 输出应使用由
`start * input.byte_strides()[axis]` 得到的 subview。不要给 slice 随便绑定一个“同样大小且位于输入内”的
区间，provider 会拒绝它。空 view 不触碰任何 byte，因此不要求制造虚假的非空 alias。

`Copy` 是 material operation。dense 输入直接复用 Mem resource 的 range-copy 路线；plain strided（包括
zero-stride broadcast）输入按 logical coordinates materialize 为 dense 输出。它不接收 block/opaque
encoding，也不隐藏 repack 或 staging。`Fill` 没有输入，根据 `TensorShape`、`DataType` 和精确 typed
`ScalarValue` 写输出；低精度浮点用 `ScalarValue::from_bits()` 明确传 storage bits。

CPU Native 与 CUDA Native 的 `Cast` 支持五种浮点 storage：f32、IEEE f16、bf16、OCP E4M3FN 和 E5M2，任意两种都可互转；
同 dtype Cast 等价于 materializing Copy。窄化继承 SIMD 的 round-to-nearest-even、FP8 finite saturation 和
NaN mapping 合同。CPU连续输入只做一次 whole-buffer SIMD 调用；plain strided 输入使用相同 scalar oracle
materialize，byte stride 无须满足 C++ 类型对齐。CUDA Native 对这五种浮点使用同一个准备后执行入口，
支持连续、plain byte-stride及zero-stride广播；没有Provider workspace或私有结果分配，可要求外部capture。
FP8有限转换借原CUDA转换接口，保留原输入NaN符号；BF16窄化保上层payload并设置quiet位，E5M2的真实Infinity保留。
大Cast使用64位索引，仍须满足实际设备内存、地址及绑定要求。CPU Native 与 CUDA Native 另支持 f32/f16/bf16 与 Bool 双向转换：
浮点的正负零为 false，其余位模式（含极小非零值、Infinity 和 NaN）为 true，输出 Bool byte 为 0 或 1；
Bool 输入的任意非零 byte 转成浮点 1，零转成 0。同 dtype Bool Cast 仍按 Copy 保留原始 byte。
Cast 始终是 out-of-place；即使输入输出 dtype 等宽，也不能绑定重叠 storage，避免 strided 输入覆盖尚未读取的数据。
CPU Native 另支持 I32→F32：完整 I32 范围按 nearest-even 舍入，不依赖调用线程的浮点舍入模式；
INT32_MAX 合法舍入为 2^31，INT32_MIN 可精确表示，连续及 plain byte-stride 输入沿用原执行器。
这不要求所有 Provider 承载该方向。其它整数/f64 Cast 的溢出、NaN-to-int 和越界语义尚未冻结，
相关 Native 转换仍返回 `Unimplemented`；
Bool 与 FP8 的直接转换也不在上述双向范围内。

Apple 另提供 E4M3FN 与 F32/F16/BF16 的 storage Cast 桥；Vulkan 直接接 F32↔E4M3FN，
F16/BF16 可以显式经过 F32 组合。这些入口保留 SATFINITE RNE、有符号零与 NaN 类别，
不开放任意 FP8 算术；绑定、布局和容量仍遵守各后端的实际合同。

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
Native Unary/Binary的保留域支持f32/f16/bf16；GELU/GELU-tanh三dtype及F32 SiLU的Native路径已退场，
普通CPU由已构建ATen接入稳定程序。半精度在F32计算后一次RNE舍入回输入类型；Select不做数值转换，因此可选择全部已登记的普通1/2/4/8-byte
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

Native 半精度 `Softmax` 的 workspace 按一行 F32 临时值复用。
调用 `workspace_requirement()` 取得大小/对齐，并在 `create_invocation(workspace)` 时绑定；Graph
按同一要求管理每个 invocation 的独立工作区。空 output 的需求为零。数据和 mask 内容可在完成后更新，
重复执行不重新准备或重新选择 Provider。

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
真正 variadic 的 descriptor span。TopK的`count=0`合法，即使输入所选轴非空，也返回该轴为0的values／I32 indices，
不读取或排序payload；其它axis、类型与范围检查照常。原生零字节结果可用`GraphOutput::description()`检查，不要求导出零字节lease。
两者都把 provider/route 选择移出热路径。对同一描述反复执行时共享
`PreparedOperation`，为每个并发执行 lane 创建并复用一个 move-only `OperationInvocation`。当前故意
没有把 prepare 或 invocation 创建隐藏在一次性 convenience 中，因为 provider workspace 和并发状态必须
有明确 owner。

固定 arity submit 可直接传 0--4 个 `ConstTensorView` input 与单 `TensorView` output；可写视图用
`as_const()` 作为输入。多输出统一传两个 span，例如 TopK 使用
`slot.submit(queue, std::array{input}, std::array{values, indices})`；Concat 可传 input-view span 与单 output。
不提供靠可写视图位置区分“两个输入”或“两个输出”的重载。这些 convenience 只在栈上组成有界 `std::array/std::span`，不分配，
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
invocation 总能提交到调用方 Queue。独立 runtime 通过 `prepare_delegated()` 返回另一种类型
`PreparedDelegatedOperation`，其 invocation 无 Queue 参数；Graph 也可将这类 runtime 作为完整 island 提交。
caller-owned workspace 使用 prepare 时
固定的 resource；provider-owned workspace 则拒绝调用方传入 BufferView。
ggml CUDA 也遵循 caller-owned 合同：按 `workspace_requirement()` 分配真实 workspace 后传给
`create_invocation()`，并保持到 Completion 退休及后续重放结束；零需求可传空 view。
普通 Graph 会自动规划、分配并持有这块空间，不要求引擎了解原生张量分配细节。

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

## 使用存储 scale 的静态 FP8 线性

对于 E4M3FN 权重 `W[N,K]`、F32/F16/BF16 的 `A[M,K]` 和独立 F32 标量 `SA`、`SW`（`[]` 或 `[1]`），
端口为 plain dense，M/N/K 为正；
用现有 `Cast`、`Binary`、transpose `View` 和 F32 累加的 `MatMul` 表达以下普通图：

```text
Q = CastE4M3FN(CastF32(A) * F32(1 / SA))
D = MatMul(CastF32(Q), transpose(CastF32(W)))
Y = CastOutput((D * SA) * SW)
```

`CastE4M3FN` 是保 NaN 类别的 SATFINITE nearest-even 转换。输出可显式取 F32/F16/BF16；
归一化、code-dot 和两次 scale 乘法保持 F32 边界，不预合并 `SA*SW`、不先还原成 BF16 操作数，
也不计算 per-token absmax。这是存储 scale 的软件参考，与动态 block128、CT channel/token QDQ
或硬件 `scaled_mm` 不自动等价。SA/SW 按普通 F32 的 IEEE 语义处理，包括零、负数、NaN/Inf 和子正规数。

CPU、CUDA、Apple、Vulkan 已接通该普通图及闭合子图的有界冷候选；中间值未发布、单读且无控制逃逸的
精确链可直接读取原 FP8 权重，避免整份 `N*K` F32 权重展开。发布中间结果时保留原分解与真实物化；
实际选择、workspace、capture 和设备范围仍以准备结果为准，没有新增公开线性算子。
[数值与资源寿命记录](../evidence/correctness/static-fp8-reference-2026-10-20.json)不代表所选制品 payload、
完整模型、质量或吞吐已经验证。

## E8M0 block128 权重导入

`<hyle/compute/weight_import.h>` 的 `import_float8_block128_weight(weight, scales, output)`
将原始 E4M3FN `weight[N,K]` 和 U8 E8M0 `scales[ceil(N/128),K/128]` 冷准备为
`float8_weight_description(K,N,kFloat32Block128)`。K 须整除128；输入允许合法的非负 byte stride。
调用者先分配输出，导入保留每个 E4 字节，仅将小块 scale 表扩宽为 F32，可缓存后交给现有
`Float8BlockLinear` 等消费者；它不执行激活量化，也不展开整份浮点权重。group32 MXFP4 是另一种格式。

这是同步 host 导入：三个 buffer 都须支持 `HostAccessible` 和 `HostAddressIdentity`，输出不能与任一输入重叠。
设备上传、同步与后续常量发布由调用者完成。scale 码0为 `2^-127`，255保留为 F32 原始 NaN 位
`0x7f800001`；后续算子的合法输入域仍按各自合同检查。映射、flush、unmap 错误会返回；写入后的错误不保证回滚，
失败输出不能发布为就绪常量。[四后端消费与导入单测](../evidence/correctness/e8m0-block128-import-reference-2026-10-21.json)
保留已测数值和寿命范围。

同一入口也接受上述二维grid的原始F32 scale，逐位复制而不再扩宽，保signed zero、tiny与NaN位。
外部`[NB,1,KB,1]`中的singleton轴由调用者显式去除，import不猜格式或计算精度；它不提前把SF舍入为BF16。
[原始F32导入与四端消费](../evidence/correctness/precise-role-consumers-reference-2026-10-21.json)
已核非power2 scale、双槽旧结果及CUDA capture。此消费者保留动态A128量化，有限见证与无激活量化结果不同；
`FP8_PB_WO`名字本身不能决定A16。若选用先SF→BF16再乘的另一参考，应显式用Cast／广播乘法表达，不能替换现F32解码数学。

## 普通 CUDA 入口

### 已编码 FP8 block128 投影

`float8_activation_description(M, K, Float8ScaleMode::kFloat32Block128)` 与
`float8_weight_description(K, N, Float8ScaleMode::kFloat32Block128)` 可以直接送入普通
`MatMul` 的 Direct/Graph 路径，输出为 F32 `[M,N]`。激活字节是 E4M3FN `[M,K]` 后接
F32 `[M,K/128]` scale；权重是 E4M3FN `[N,K]` 后接 F32 `[ceil(N/128),K/128]` scale。
K 须整除128，N允许最后一个不足128行的block；这些描述符不执行模型加载或动态量化。

CUDA 默认接入固定 SGLang 原版 SM89 AOT 程序；驱动兼容性由实际模块/函数加载检查，
不要求 `driver_cuda_version` 等于某个固定版本。
程序在准备时加载并绑定，重放为一次原生发射，没有额外 payload 工作区或权重展开。
源码/静态/动态构建都将程序嵌入 Provider，不依赖用户运行 Python/Triton。
其它硬件域、UE8M0 group32 编码及动态激活量化不能据此视为已支持。

### FP8 权重的一次性准备

`model_ops::float8_weight_decode()` 接受上述逻辑 `[K,N]` block128 权重，输出为明确的
物理 row-major BF16 `[N,K]`；参数 `DataType::kFloat32` 可保留 F32 结果。普通 Direct
`prepare()` 或 Graph 节点只绑定描述符与原生程序；真正解码发生在提交后，不隐式读取权重。

当模型需要吸收或拆分 K/V 权重时，消费者先执行这个准备操作，用既有 View/Copy 整理布局，
等 Completion 成功后将结果作为后续图的常量。消费者持有缓存并复用；不要在每个 token 的图里
重复解码。BF16 完整矩阵占 `2*N*K` 字节，另计布局整理时的临时矩阵和最终缓存峰值。

CUDA 默认绑定固定 DeepSeek 原版 SM89 程序，驱动兼容性同上；当前有限域为 K 整除128、
`N*K <= INT32_MAX`、`K/128 <= 65535`；输入需4字节对齐，输出按 BF16/F32 需2/4字节对齐。
一次原生发射，无额外工作区；支持独立 invocation 并发，输出禁止与输入重叠。编码中的 NaN
保留为 NaN；该入口不进行模型文件读取或有限权重扫描。AOT 程序随 Provider 嵌入，运行时无需 Python/Triton。

### 动态 FP8 模型投影

浮点激活尚未编码时，将 `model_ops::float8_block_linear()` 作为普通 Direct `prepare()` 或
Graph 节点的 operation，输入为 dense BF16/F32 `[M,K]` 和上述 block128 权重，输出保留激活
dtype `[M,N]`。它显式包含动态量化，普通 `MatMul` 不暗中量化输入。

默认 `Float8ScalePolicy::kDivide` 逐 token/group128 以 F32 除448生成 scale，
再用 F32 `A/scale` 归一化，absmax floor 为 `1e-10`。
`model_ops::float8_block_linear(model_ops::Float8ScalePolicy::kPowerOfTwo, 1e-4F)`
表达 DeepSeek 官方的 BF16 幂次 scale 语义。后者的 F32 激活域尚未资格化，准备时拒绝。
权重的128×128分组保持不变，不把幂次scale解释成group32存储。

GLM Quark block128 的参考使用
`model_ops::float8_block_linear(model_ops::Float8ScalePolicy::kMultiplyReciprocal, 1e-10F)`：
先计算 `scale = F32(max(absmax, floor) * F32(1/448))`，再分别计算
`inverse = F32(1/scale)`、`normalized = F32(A*inverse)`，最后转 E4M3FN。
这三步有各自的 F32 舍入，不能替换为直接除448或 `A/scale`，也不插入 BF16 中间值。
dense 与 grouped 工厂使用同一参数；两者都要求有限正 floor，且 `floor/448` 为正常 F32 数。

当前 CUDA 承载为 SM89 AOT，驱动兼容性由实际模块加载检查，激活域仍有限。准备绑定原版 SGLang 程序，独立线性执行为
量化和 GEMM 两次发射，使用每 invocation 独占的 caller workspace：`M*K + 4*M*(K/128)`
字节、4字节对齐；没有 host 同步或常驻展开权重。普通scale输入需要16字节对齐，BF16幂次
scale输入需要2字节对齐；Graph规划会应用这些要求，手动绑定的错位输入在提交前拒绝。
该接口不加载模型。

CUDA Graph 中，两个线性节点若读取同一激活值，且 floor、scale policy 与原实现的支持条件一致，
冷编译可以共享本次调用的量化：一次量化后执行两个原 GEMM，分别保留输出 dtype 和舍入。
两份权重不拼接，量化结果不跨调用缓存；不同激活值或量化规则继续独立执行。
这种共享减少重复量化，不代表两个 GEMM 合为一个 kernel，也不保证整个图的峰值工作区减半。

### CT BF16 block128 参考图

存储为 E4M3 权重和 BF16 block128 scale、激活按 BF16 group128 QDQ 的 CT 制品，使用普通
Graph 节点组合。将激活视为 `[T,K/128,128]`，Reduce 求每组 absmax，BF16 除448，
仅将恰好为零的 scale 替换为 BF16 epsilon `1/128`；随后保留 BF16 除法、对称零点加减、
E4 SATFINITE Cast 和 BF16 乘法。权重先 Cast BF16，再与原 BF16 scale 相乘。
最后用 F32 累加、BF16 输出的普通 MatMul。

选专家时先用受检 Gather 选择原 E4 权重及 BF16 scale，再还原选中行；共享激活使用 batch
广播，每 token 只做一次 QDQ。引擎沿同一语义图调用，无需选择 Provider。四后端的可执行
组合见[参考图](../../tests/compute_models/ct_block128_qdq_fixture.h)，
[数值与寿命检查](../evidence/correctness/ct-bf16-block128-reference-2026-10-21.json)覆盖有限源激活、
N/K整128、有界专家 bank；整制品工作集绑定、Linux及吞吐另行验证。

### 激活的组量化与还原

`model_ops::block_quantization_roundtrip(format)` 将 dense BF16 `[M,K]` 输入量化后还原为
相同形状的 BF16，供需要保留量化感知训练边界的消费者组合。两个显式格式为：

- `BlockQuantizationFormat::kFloat8E4M3Block64`：每64值取 absmax，floor为 `1e-4`，除448。
- `BlockQuantizationFormat::kFloat4E2M1Block32`：每32值取 absmax，floor为 `6*2^-126`，除6。

两者 scale 都向上取二次幂，代码值饱和最近偶数舍入；不等同于其它 MX 格式的 round-even scale。
输入必须有限，K须整组；量化还原后的极值仍可能溢出。输入输出不重叠，均需16字节对齐。
当前 CUDA 承载为原版 DeepSeek SM89 程序，驱动兼容性由实际模块加载检查。默认路线自动绑定，Graph使用同一
运算描述；不需要安装 Python/TileLang 或指定 Provider。每次执行一次原生发射，caller workspace
为 `4*M*(K/group)` 字节、4字节对齐；独立 invocation 使用独立空间，存储保持到 Completion。
模型如何切分 RoPE 尾部、更新 cache 或衔接 Hadamard 属于消费者；本操作不隐式管理这些状态。

### 逐行量化的历史存储

需要保留实际压缩历史时，可用 `RowCacheEncoding::float8_block64_tail64()` 或
`RowCacheEncoding::float4_block32()`，配合 `model_ops::row_cache_encode(encoding)` 和
`model_ops::row_cache_gather_decode(encoding)`。前者将新增的 BF16 `[R,D]` 行编码为 U8
`[R,row_bytes]`；后者从同格式的历史 bank 按 I32 IDs 只解码选中行，输出 BF16 `[IDs...,D]`。
`row_cache_carrier_description(encoding,R)` 构造实际存储描述，重复 ID 和多于历史行数的选择数均合法。

FP8格式固定D512：448项按group64量化，64项BF16尾部逐bit保留，共584B/行。
FP4格式固定D128：按group32量化，共68B/行；Indexer的Hadamard与BF16边界仍由上游明确调用。
两者使用逐组幂次scale和最近偶数舍入，有限输入的量化结果仍可能溢出。Checked模式检查全部ID，失败不发布结果，
不支持负索引回绕；输入含NaN/Inf时编码失败。实际字节格式与符号零规则见公共头。

这是Hyle显式定义的行内格式；官方DeepSeek参考实际保存BF16 QAT结果，没有规定此packed ABI。
CPU、CUDA、Apple、Vulkan均提供两项操作，可通过普通Direct/Graph入口使用。Apple当前单输入仍受
原INT_MAX字节承载限制；Vulkan仍检查设备单descriptor、U32字节寻址及dispatch限制。U8 carrier沿原
StateStore与F2的区间权限和发布使用，不会隐式更新可见长度、取得整arena权限或转换为原生分页attention布局；
`CacheStorage`原合同保持不变。独立页存储使用[typed行布局](batch-attention.md#逐行压缩历史的页存储)，
四后端均有ready frame的跨页选中行读取入口；CPU/CUDA Graph另可消费原CopyBatchUpdate正在
追加的行页。Apple/Vulkan使用各自native Context入口，详见同一指南；普通bank操作不能直接借用整arena权限。

CPU、CUDA、Apple、Vulkan另提供 `model_ops::row_cache_indexer_scores()`，四个输入依次为BF16 query `[T,64,128]`、
上述FP4 U8 bank `[L,68]`、已预缩放的BF16权重 `[T,64]` 和I32范围 `[T,2]`。
每个范围是bank中的半开区间 `[begin,end)`；所有范围在读取bank前检查，非法范围不发布成功结果。
区外score为负无穷。
输出F32 `[T,L]` 保留BF16 dot→ReLU→BF16乘权→BF16 head sum的边界，F32只承载累加及最终宽化；
上游仍显式执行Hadamard、query QAT和权重缩放。各路线不展开完整BF16 bank，仍需完整score输出；
CPU/CUDA/Apple按块共享历史解码，Vulkan直接计算可见的query/key对。
CUDA的caller workspace由原prepared operation报告并按slot独占；Apple还保留分块score与最终拼接结果。
上述四输入入口使用普通contiguous bank；typed分页评分有独立承载要求，跨页选中行读取不能代替TopK之前的评分。

CPU/CUDA的 `model_ops::row_cache_gather_decode_padded(encoding)` 用于首次prefill等历史不足的情况：
只有ID `-1` 表示padding，不读取bank并输出BF16正零；其他负数或越界ID仍失败，完整decoded输出保持不变。
普通checked decoder仍拒绝 `-1`。零向量还需要attention mask才能排除softmax贡献；
typed Graph宏会一并生成该mask，见[页存储指南](batch-attention.md#逐行压缩历史的页存储)。

### 动态 FP8 专家批次

`model_ops::grouped_float8_linear()` 的三个输入依次为：

- `float8_grouped_weight_description(E,N,K)`：全部专家的 E4M3FN `[E,N,K]` codes，
  后接 F32 `[E,ceil(N/128),K/128]` scales；两部分连续存放，不逐专家交错。
- dense BF16/F32 激活 `[T,A,K]`：`A=1` 时各专家共享一次量化的 token；`A=S` 时保留
  每个专家分支的独立激活，例如 down 投影。
- dense I32 专家 IDs `[T,S]`；输出为同激活 dtype 的 `[T,S,N]`，重复 ID 合法。

默认使用 `Float8ScalePolicy::kDivide` 与 floor `1e-10`。需要 GA power2 规则时调用
`model_ops::grouped_float8_linear(model_ops::Float8ScalePolicy::kPowerOfTwo, 1e-4F)`，
参数顺序与 dense 版本相同。BF16 输入先扩为 F32，按组求 absmax／floor，乘 F32 的 `1/448`，
再向上取二次幂；量化器中间不转 BF16。CPU、CUDA、Apple、Vulkan 均有该 BF16 路径的
[数值与寿命参考](../evidence/correctness/grouped-fp8-power2-reference-2026-10-21.json)。
CUDA 当前 power2 原生程序仅接受 BF16；F32 形状推导可以成功，但 prepare 会明确拒绝。
存储为 E8M0 的 macroblock128 scale 仍须准确导入原 F32 carrier。

当前 CUDA 支持条件为 SM89、`1<=E<=1022` 和完整 K128 分组；S为实际选中slot数，不固定为8，
并受原生索引和padding容量约束。使用普通 F32 scale、默认 floor `1e-10`。其它域在准备阶段明确拒绝。激活需16字节对齐，
权重/ID需4字节对齐；工作区大小与16字节对齐由 `workspace_requirement()` 给出。
每个 invocation 独占量化、专家排序和 padding 空间。准备绑定成熟程序，重放没有专家逐项
host 调度、ID读回或权重解码。负数及越界 ID 安全返回失败 Completion；失败输出不可消费。
每个K128代码点积先乘本组activation scale，再乘weight scale；两次乘法分别按F32 RNE舍入，随后F32累加。
不能先合并两份scale：例如零权重codes、activation scale为2、有限weight scale为`2^127`时，
预乘scale会溢出并把本应为零的点积变成NaN。此顺序不保证所有有限输入的非零结果都不会溢出。
单expert的 `N*K` 仍受原生I32寻址约束，完整bank的专家基址使用I64，不将 `E*N*K`
误限为I32。完整bank仍须满足真实buffer与设备内存限制；不自动转为按需换入。

Vulkan的官方grouped FP8/FP4也沿相同公开权重描述，完整压缩bank不再因超过2GB而一概拒绝；
仍受实际 `maxStorageBufferRange`、算子寻址及工作区限制。GLM的完整3.22GB FP8 bank已有
定向消费验证，不能外推为所有模型或更大FP4 bank均可驻留。用实际descriptor执行prepare检查资格。

### 执行前检查具体请求

CPU/CUDA 用描述符和 memory resource 调用普通 `prepare()`，即可检查所选设备上实际的
dtype、shape、layout 与资源是否可执行；此时无需分配 tensor payload 或创建 Queue。
准备可能编译并分配程序状态，但不读取或执行输入数据。保留成功结果供后续 invocation 使用，
无需另做一次查询或再次准备。`infer_output_descriptions()` 只检查共同数学描述，不证明路线支持。

准备失败保留具体原因：`InvalidArgument`/`OutOfRange` 表示请求不合法，`Unimplemented` 表示
当前资格条件不受支持，资源不足和编译错误按实际错误返回。所有实现都不支持时，错误包含 Provider
名字和其拒绝原因；已进入实际准备的明确拒绝优先于后续更宽泛的 proposal 拒绝。执行期不重新选择。

### 创建 CUDA 程序

Windows/Linux x86_64 选择 CUDA 后端即可取得 CUDA-only Compute 与 Graph，
自动组装路线所需的成熟 Provider。系统须预装受支持的 C++ 工具链和 CUDA Toolkit 13.3+；
缺 Toolkit 时初始化明确失败。

```text
python scripts/init.py --backends cuda --components compute,graph
python scripts/build.py --jobs 2
```

同时编入 CPU 与 CUDA 使用 `--backends cpu,cuda`，只编入 Vulkan 使用 `--backends vulkan`；
Windows/Linux x86_64 的 `--backends all` 同时提供 CPU、CUDA 和 Vulkan。
编译可用集合不替代引擎运行时的 CPU/GPU/混合执行选择。
静态库用 `init.py --static`，安装用 `build.py --install <prefix>`，无需内部 Provider flags 或排序。
源码与安装包均链接 `hyle::compute`，
应用链接后调用 `hyle_stage_runtime(app)`，Windows 把依赖 DLL 邻置 exe。
构建选择不代替具体输入的支持检查；prepare 按实际模型格式、dtype、布局、设备和资源约束绑定实现，
不满足条件时返回明确错误。

直接使用 CMake 时选择 `HYLE_COMPONENTS=compute`、`HYLE_COMPUTE_ROUTES=cuda`。自然 C++ 入口如下：

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
tensor，submit 后显式检查 Completion。完整 dense gate/up → SwiGLU → dense down 样例见
[CUDA Direct FFN](../../samples/package/compute/cpp/cuda_ffn.h)。该样例连续执行三组变化输入，
没有 Provider 列表、计时资料或用户 kernel callback。

普通 CUDA Context 在准备期选择实际可用的实现。普通张量、GGUF、FP8/FP4 和 managed
程序的执行边界不同；`prepare()` 成功后可用 `provider()` / `runtime_properties()` 诊断。

需要特别注意：

- GPU descriptor-only prepare 显式传 device resource；查询每个 invocation 的 workspace 大小与对齐，
  不让首次 submit 猜资源或隐式 staging。
- 普通 rank-2 F16/F32 GEMM 使用 F32 累加；F32 输入不默认为 TF32。FP8/FP4 模型接口保留各自
  量化与 scale 合同，不把输入 dtype 当作中间计算精度。
  连续 F32 三／四维批量 MatMul 也保留 F32 源精度和累加；具体广播与资源域由 `prepare()` 检查。
- Windows/Linux 的适用 attention 与原生批次可自动使用 FlashInfer；原生批次的 SM89、格式与形状域
  见[批次 attention](batch-attention.md)。
  managed 程序可能分配、阻塞 host，或只支持自己内部 capture。外部 capture 要显式要求并检查。
- SM80–SM121 的 dense BSHD BF16 attention，在无显式 mask、F32 accumulation、Q/K/V
  depth 相同且为8的倍数（8–256）时，可使用原生融合程序；支持 GQA、bottom-right causal
  和显式 scale。它保留 BF16 的指数范围，以 F32 累加和计算 softmax 统计，概率 tile 使用 BF16；
  与完整 F32 中间图不是逐位相同。该程序不物化 Q×KV scores/mask，但仍有 provider 私有分配、
  GQA 分组输出与拼接成本，属于 managed，不能用于要求外部 capture 的调用。
  其它 BF16/F32 路径仍可能物化 F32 score/softmax，其临时值随
  `batch * heads * query_length * kv_length` 增长，应检查实际程序并计入预算。
- 普通 BSHD attention 的第四口可接 dense F32 additive mask，按 `[B,H,Q,KV]` 广播；
  Q/K/V 保留原 F32/F16/BF16 合法域，mask 不收窄。该路径使用原 F32 scores/softmax/PV，
  长查询沿原分块执行，仍有临时存储与 managed 成本。全屏蔽或含未屏蔽 `+Inf` 的行遵循普通 softmax，
  不自动归零；causal 在 bias 后排除未来位置。
- checked Gather/Scan 在 GPU 验证索引或整数前缀，可回传少量控制标量并等待。
  失败输出不发布，重建失败 invocation；不把 checked 程序当作完全不阻塞 host 的 kernel。
- 原 ggml 数学节点有容量和对齐要求：通常为 256-byte 数学端口，对复制/转换可有更窄要求。
  不满足时准备或绑定明确失败，不在 submit 中偷偷复制或换 Provider。
- 不要在 Context 创建后改变影响数值的上游环境设置。未实现的 dtype/布局、设备失效与普通
  请求参数错误分别按实际错误返回；重建槽不能修复终态设备故障。

变化输入和完整 FFN 示例见 [CUDA Direct FFN](../../samples/package/compute/cpp/cuda_ffn.h)。
同 GPU 多 stream 的完成依赖、同步与异步传输见[Queue 与 Completion](#queuebyte-transfer-与-completion)。

## 可选模型与阶段信息

已知模型与固定部署制品的消费者可在冷准备时传入 `WorkloadContext`。普通工程计算省略它即可。
HYLE 不加载模型文件；调用方先核对实际 checkpoint/GGUF/safetensors，再提供模型 revision、配置摘要、
制品仓库/revision/variant、manifest 摘要和 container format。画像只绑定部署制品；每个 tensor 的真实
dtype、物理格式和 shape 仍由本次 operation 的 `TensorDesc` 证明。

```cpp
auto profile = hyle::compute::ModelArtifactProfile::create({
    .model_publisher = "Qwen",
    .model_repository = "Qwen/Qwen3.5-2B",
    .model_family = "qwen3_5",
    .model_name = "qwen3.5-2b",
    .model_revision = "15852e8c16360a2fea060d615a32b45270f8a8fc",
    .configuration_digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    .artifact_repository = "unsloth/Qwen3.5-2B-GGUF",
    .artifact_revision = "f6d5376be1edb4d416d56da11e5397a961aca8ae",
    .variant = "UD-Q4_K_XL",
    .manifest_digest = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    .format_family = "gguf",
});
if (!profile) return profile.error();
hyle::compute::WorkloadContext workload{
    .profile = *profile,
    .phase = hyle::compute::WorkloadPhase::kDecode,
};
auto program = hyle::compute::prepare(
    context, hyle::compute::MatMul{}, left_desc, right_desc, resource, workload);
```

画像创建时一次拷贝拥有字符串，源字符串随后可销毁；多个程序共享同一个不可变画像。完整画像的两个
digest 必须是规范的 64 位小写十六进制文本；空画像不分配，部分画像会被拒绝。自定义 allocator 的
backing 必须活到最后一个画像/程序释放。`program.workload()` 返回
保有所有权的冷诊断副本。阶段或版本变化需要重新准备，不能给运行中的程序换标签。
可选的 `region` 是带版本的片段语义标识，不替代实际 operation/DAG、精度、资源和状态检查。

### GGUF/ggml 原生压缩权重

公共描述器包含 Q4_0/Q4_1/Q2_K、IQ2_XXS/XS/S、IQ1_S/M 及 IQ1_XXXS/XXS/XS 等编码。
最后三种使用固定 Unsloth IQ1 narrow 扩展，不能从文件名中的 UD 档位猜编码。
描述器存在不代表每个后端的 MatMul、expert 和 embedding 均支持该格式；各操作分别检查原生实现。
CUDA/Vulkan 的原生选行 embedding 当前接通 19 种 GGUF 编码，尚不含这三个 IQ1 narrow 变体。
CUDA 的 IQ4_NL/MXFP4 embedding 还要求行宽为 256 的倍数；这与其 32 值物理 block 是不同限制。
`kNvfp4` 是独立的 GGUF 编码：每 64 个值占 36 字节，含四组内嵌 scale；K 必须为 64 的倍数。
CPU/CUDA/Vulkan/Apple 已接原编码 MatMul、GroupedMatMul 和所选行解码，不建立常驻浮点权重副本。
Apple NVFP4 的激活、累加和输出采用 F32，准备时检查实际设备与原生 pipeline 资源。
它与 GGUF MXFP4、HF ModelOpt/Quark 的 NVFP4 W4A4 不是同一存储和激活合同。

调用方已经解析 GGUF tensor table 时，用 `<hyle/compute/low_precision.h>` 描述原文件中的压缩字节，
不需要也不应先展开成常驻浮点权重。`gguf_weight_description()` 接收 `GgufQuantizedType` 声明的编码，
返回逻辑 `[K,N]`、物理连续 `[N,K]` 的
checked `TensorDesc`；`K` 必须包含完整原生 block。`logical_type` 是 MatMul activation/output 的
浮点类型，不是压缩 code 的位宽。

```cpp
#include <array>
#include <hyle/compute/low_precision.h>

auto weight_desc = hyle::compute::gguf_weight_description(
    hyle::compute::GgufQuantizedType::kQ4K,
    hyle::compute::DataType::kFloat32, k, n);
if (!weight_desc) return weight_desc.error();

// weight_bytes 是 loader 持有的原始 GGUF payload；HYLE 不读取 GGUF 文件。
auto weight = hyle::compute::ConstTensorView::create(
    *weight_desc, weight_bytes);
if (!weight) return weight.error();
std::array inputs{activation, *weight};
std::array outputs{output};
auto prepared = hyle::compute::prepare(
    context, hyle::compute::Operation(hyle::compute::MatMul{}), inputs, outputs);
```

`model_ops::GroupedMatMul` 使用权重 `[E,N,K]`、F32 激活 `[T,A,K]` 和 I32 IDs `[T,S]`，输出
F32 `[T,S,N]`。`A=1` 让同一 token 的专家共享激活，`A=S` 保留各分支独立激活；重复 ID 合法。
也可用 `View::broadcast_to` 的零 slot 步幅表达共享，实际 backing 只需覆盖被读取的行，无需先复制 S 份。
默认路线接 dense 输入及 K 连续的共享 View；具体行步幅仍由 prepare 检查，CUDA 多 token 的行间距须为
完整 K 行的倍数。普通 BF16 权重仍以原 F32 激活、F32 累加计算；CPU 会扩宽权重，Graph 的不可变常量可冷准备。
CUDA 保留原 BF16 bank，合资格小 T 域直接在寄存器中精确扩宽，不留整库 F32 副本；实际存储仍为 BF16。
Vulkan 非密集共享 View 还须满足原生缓冲地址对齐，偏移不合法时在绑定阶段拒绝；密集输入保留原对齐适配。
CUDA 在 NVIDIA 的合资格小 T 域使用原 MMVF，直接读取 GPU IDs 和共享／独立激活，不整理 T×S 份输入；
资格按原设备／形状偏好、pair 对齐、实际元素宽度／步幅和设备 grid 限制检查；严格 BF16W/F32A 域复用原 F32 计算偏好。
重复 ID 保留独立输出槽，
动态 ID 越界仍通过原 checked 流程由 Completion 报错，失败结果不会发布。
普通权重的分组回退也保持父运算的 F32 累加与源操作数精度请求。
大 T、奇数 K 或其他不合资格布局仍可能整理 T×S 行、读回 IDs 并同步；CUDA BF16 后备仅将实际使用的
单专家 N×K 权重临时扩为 F32，再走原 F32 cuBLAS。排序、checked 缓冲和原生池空间仍存在。
该严格 mixed 域的旧 indexed GLU epilogue 尚未融合。数值、寿命、空间及常量对照边界见
[CUDA BF16 分组记录](../evidence/performance/bf16-grouped-bank-2026-10-20.json)。

MoE 的连续专家权重使用 `gguf_grouped_weight_description(type, logical_type, experts, n, k)`，其逻辑与
物理轴均为 `[E,N,K]`，交给普通 `GroupedMatMul`。当前 CPU、CUDA、Apple 与 Vulkan 默认路线会在准备期把合格的
MatMul/GroupedMatMul 绑定到固定 ggml 原生程序；具体设备和上游版本仍可能拒绝某种格式或 shape。
Graph 使用相同描述器与 Provider recipe，重复 submit 不重新选路、不重新量化，也不复制压缩权重。
伪造已知格式 ID 但给出错误 block byte 数会在准备前被拒绝。Apple 用 `apple::input_spec()` 将逻辑描述
映射到 MLX `uint8` carrier；大 bank 用二维编码行保留单轴原生尺寸，消费者仍使用同一逻辑 descriptor。
Vulkan 使用其原生 buffer 与相应程序。
一种压缩格式可执行，不表示完整模型的其它必需片段也可执行。

CPU/CUDA 的压缩 `GroupedMatMul` 按原生block寻址检查完整bank，不把整个逻辑
`E*N*K` 误限为I32；单维、单expert stride和原native block索引边界仍保留。
Vulkan还受真实 `maxStorageBufferRange` 限制，Apple的carrier也有原生尺寸约束。
同一制品可能需分bank或按需驻留；库不在一次Grouped调用中暗中换入整个模型。

固定的模型专用规则必须完整匹配身份及其余适用条件；缺失/未知信息只允许合法通用路径，
或返回明确拒绝。传入模型名本身不代表已有该模型专用优化。用户无需排列 Provider 或提供计时资料；
HYLE 的规则匹配在运行期不读取模型字符串、查询设备版本或重选实现；成熟 Provider 自身的
managed 派发成本另外声明。

CUDA 的 `cuda::environment(context)` 返回 Context 创建时采集的设备名、UUID、SM major/minor、
驱动支持的 CUDA API 版本、独立 `driver_release` 文本、runtime 版本与编译 Toolkit 版本。
`driver_cuda_version` 对应 `cuDriverGetVersion`，**不是 NVIDIA 驱动安装包版本号**；
`driver_release` 来自可选系统 NVML，缺少时为空、不禁用 CUDA 路线。冷规则的 `kApiVersion` 为前者，
`kDriverVersion` 为实际 NVIDIA release 文本，`kDriverName` 为 nvidia。系统 driver DLL 不随 SDK 复制。
固定规则只能匹配实际具备的事实；
不能把该字段当作驱动补丁版本。Apple/Vulkan 使用各自原生 `Environment`，Provider 版本来自
对应实现/锁定依赖；它们不共用一个万能设备结构。

## CPU 非 GGUF 原格式与压缩专家

这些接口位于 `<hyle/compute/model_ops.h>`，普通 CPU Direct/Graph 自动绑定已构建的原库组合。
输入是调用方解析好的原 tensor；HYLE 不读取模型文件，也不把不同的 4-bit 制品视为相同编码。

| 原格式 | 接口与端口 | 输出或执行边界 |
|---|---|---|
| NF4 | `artifact_weight_decode(kNf4, group)`；U8 `[N,K/2]`、F32 absmax `[G]`、F32 codebook `[16]` | 高 nibble 在前，默认 BF16 |
| NF4 double quant | 同接口 `kNf4DoubleQuant`；U8 codes `[N,K/2]`、U8 absmax `[G]`、F32 nested absmax `[ceil(G/256)]`、F32 nested codebook `[256]`、F32 scalar offset、F32 codebook `[16]` | nested scale 与 offset 保留，默认 BF16 |
| AWQ GEMM int4 | 同接口 `kAwqGemm4`；I32 `[K,N/8]`、I32 zero `[K/group,N/8]`、floating scale `[K/group,N]` | 原 AWQ 列排列与 zero point，默认 scale dtype |
| compressed-tensors symmetric int4 | 同接口 `kCompressedTensorsSymmetric4`；I32 `[N,K/8]`、floating scale `[N,K/group]` | low-first pack、减 8 后按原 scale dtype 计算；无 g_idx 模式 |
| compressed-tensors asymmetric int4 | 同接口 `kCompressedTensorsAsymmetric4`；I32 codes `[N,K/8]`、I32 zero `[ceil(N/8),K/group]`、floating scale `[N,K/group]` | codes 沿 K、zero 沿 N 低位在先打包并减 8；zero 尾部填充不参与计算；仅自然列分组，无 g_idx |
| GPTQ-v1 int4 | 同接口 `kGptqV1Int4`；依次为 I32 Q `[K/8,N]`、I32 Z `[K/group,N/8]`、F16 scale `[K/group,N]`、I32 g_idx `[K]` | Q 沿 K、Z 沿 N 低位在先；zero nibble 加一，15→16；F16 乘法后转输出，默认 F16 |
| MLX affine 4/5/6/8 位；MXFP4/MXFP8 | `mlx_quantized_matmul(format, bits, group)`；A `[M,K]`、U32 words `[N,(K/32)*bits]`、scale、affine bias | 直接消费压缩权重，F32 累加；MX scale 为原 U8 E8M0 group32 |
| MLX 选中专家 | `mlx_grouped_matmul(format, bits, group)`；A `[T,S,K]`、words/scale/bias 带 expert 首轴、I32 ids `[T,S]` | 只展开当前 K group 与选中专家，不建立整个浮点专家库 |
| MLX affine embedding | `mlx_quantized_embedding(bits, group)`；原 words/scale/bias、I32 row ids | 只选择所需行，输出保持 scale dtype |

`G=N*K/group`。`artifact_weight_decode()` 显式输出 dense `[N,K]`；独立冷解码后可持有结果，
用其 transposed 描述作为普通 MatMul RHS，或绑定为 Graph constant，避免每步重复展开。
显式 `output_type` 可选 F16/BF16/F32；它不替代制品的原 scale 精度或 packing。
CT 非对称解码默认输出 scale dtype；减法和乘法先在该 dtype 各自舍入，最后才转换到显式输出类型。
例如 BF16 scale 配合 F32 输出仍保留 BF16 中间舍入，不能等同于全程 F32 解码。

GPTQ-v1要求上述plain dense端口、正K/N/group、K和N均整除8、K整除group。
`g_idx[k]`选择scale／zero的组，不重排激活或权重；每次提交先核全部ID位于`[0,K/group)`，再按ID寻址。
非法ID由Completion报告失败，不能读取失败结果。原始减法为精确整数，乘F16 scale后先舍入为F16，
显式输出类型或外层Cast随后执行；scale的NaN／Inf／有符号零不额外拒绝。不能按仓库名字套用AWQ或CT packing。

同图的CT symmetric/asymmetric与GPTQ-v1 INT4 `Decode→Transpose View→MatMul`可冷融合为packed计算。
转置前或后允许一处Cast，保scale dtype乘法、decode及operand舍入，MatMul为F32累加；
点积归约次序可不同。中间值未发布、单读者且无外向控制依赖时，才可省去整份dense W；
独立冷解码后将dense结果绑入另一张图会丢失codec身份。

CPU/CUDA共用识别与prepared适配；CUDA支持捕获，CPU以64元素栈内tile计算，packed workspace为0。
CPU默认单向量优先packed，多行先原矩阵；明确要求`ProviderReplayPolicy::kRequirePrepared`时可选prepared候选。
[CPU三系统验证](../evidence/correctness/ct-packed-cpu-portability-2026-10-20.json)和
[默认排序实测](../evidence/performance/ct-packed-cpu-policy-2026-10-20.json)区分能力与默认性能。

[Vulkan](../evidence/correctness/ct-packed-vulkan-2026-10-20.json)与
[Apple](../evidence/correctness/ct-packed-apple-2026-10-20.json)沿原native Program承载同一recipe：
Vulkan只对已测单行BF16 scale/decode/operand默认优先packed，多行和其他精度先原路线；
Apple支持正rank2 plain激活与实际stride，当前实测较慢，默认仍为原路线。
普通路线仍可能物化dense W，native private量以实际报告为准；workspace0不代表总分配0，也不承诺OOM后自动换路。
GPTQ-v1的四后端原decode与闭合packed候选已有[数值及寿命回归](../evidence/correctness/gptq-reference-2026-10-20.json)，
包含动态g_idx、非法ID、双槽／旧结果、逃逸原分解与CUDA实际捕获。packed路径直接读原四份权重端口，
不建立额外整份dense W；Vulkan／Apple的新GPTQ候选保守排序，不由功能测试宣称更快。
所选量化制品的payload／整模型、其它GPTQ packing、W4A4及FFN/专家覆盖见[模型总表](../implementation/model-operators.md)。

`model_ops::nvfp4_linear()`提供固定group16的ModelOpt软件参考线性：
`BF16 A[M,K], U8 W[N,K/2], E4M3FN scale[N,K/16], F32 weight_global, F32 input_global → BF16[M,N]`。
端口均为plain dense，两个global分别为scalar或长度1向量；M/N为正、K为正且整除16，bias另用Add。
A的当前块absmax经E4M3 scale、E2M1量化后存BF16；W先在F32乘两级scale、再乘原低位在先E2M1码并存BF16。
两端BF16操作数进入F32累加，最后写BF16；具体公式与特殊零规则见`Nvfp4Linear`声明。
输入A须finite、scale须finite且非负、两个global须finite且严格正；有限输入仍可产生IEEE非有限结果。
域错误由Completion报告，失败结果不可读；Graph失败Invocation须从同Executable重建并重新绑定／准备。
CPU/CUDA沿既有prepared入口执行，使用调用者的2×M×K字节激活工作区，只展开当前W组，不保留整bank浮点副本；
CUDA可RequireCapture；Apple/Vulkan沿各自原生program执行，也只物化A，不展开整份W。
Apple另有检查数组及managed状态成本，不能把总临时内存写成仅2×M×K。
[四后端数值及寿命回归](../evidence/correctness/nvfp4-cross-backend-reference-2026-10-20.json)
含非零subnormal权重；未验证Linux、实际制品质量或吞吐，也不证明与硬件native FP4 GEMM等价。

`model_ops::grouped_nvfp4_linear()`沿用上述数学，六个输入依次为
`BF16 A[T,1或S,K], U8 W[E,N,K/2], E4M3FN scale[E,N,K/16], F32 WG[E], F32 IG[E], I32 IDs[T,S]`，
输出为`BF16[T,S,N]`。各维须为正，K整除16；WG/IG为严格rank1的每专家向量，其余参数及IDs为dense。
A另支持真实零slot步幅和token行距View，重复ID保留独立slot。共享原始A也须按每个选中专家的IG分别QDQ，
不能共享已量化A。激活QDQ本体占2×T×S×K字节，直接读压缩W，不展开整E×N×K；各后端另有状态／运行时成本。
全部ID先检查，非法ID不会用于专家寻址；WG/IG/scale只检查选中专家，未选专家的无效metadata不使调用失败。
域错误及失败Invocation处理与dense入口相同。CPU Direct/Graph、CUDA capture、Apple Metal与Vulkan的
[数值及寿命回归](../evidence/correctness/grouped-nvfp4-reference-2026-10-20.json)已通过，保留原dense邻居检查；
这不代表执行了所选Q24整制品或证明了硬件FP4 GEMM、质量和吞吐。

`model_ops::local_nvfp4_linear(rounding)`与`grouped_local_nvfp4_linear(rounding)`分别提供
CT local-dynamic的dense／选中专家软件参考，端口、形状与上述两个入口相同，数学由独立descriptor确定。
传入原始WG／IG，调用者不预求倒数；重建组scale使用直接F32除法`sf/IG`和`SF/WG`。
A的absmax／6先舍入到BF16，再乘IG并量化E4M3，仅零scale替换为0.125；W的E2M1码8为负零。
`Fp4InputRounding::kFloat32`为默认值；`kBFloat16`在clamp后、FP4阈值量化前增加BF16舍入。
字段选择的是实际算术，不由checkpoint或producer版本推断；完整零规则及公式见`LocalNvfp4Linear`声明和
[固定制品／producer合同](../evidence/research/ct-local-nvfp4-contract-2026-10-20.json)。
输入值域、全ID先检查、只检查选中metadata及失败后重建Invocation的要求与上述入口一致。
令R为dense的M或grouped的T×S：CPU/CUDA使用调用者2×R×K字节激活工作区，CUDA支持capture；
Apple/Vulkan在原生program中准备该激活临时量，Apple另有逐组／逐输出检查数组，Vulkan另有状态缓冲及运行时成本。
各端均直接读取压缩W；2×R×K不代表总临时内存。索引域、workspace与设备单buffer／绑定容量仍按实际后端检查。
[27项数值／寿命及原ModelOpt邻居回归](../evidence/correctness/ct-local-nvfp4-reference-2026-10-20.json)
在Windows CPU/CUDA/Vulkan及macOS M4上通过，无跳过；未验证Linux、512专家分立tensor到工作集的整条资源旅程、
任意大bank、所选制品payload／整模型、质量或吞吐，也不等同硬件scaled FP4 GEMM。

`model_ops::mxfp4_linear()`／`Mxfp4Linear`提供group32软件参考，三个真实plain dense输入为
`BF16 A[M,K], U8 W[N,K/2], U8 SF[N,K/32] → BF16[M,N]`；各维为正，K整除32。
`grouped_mxfp4_linear()`／`GroupedMxfp4Linear`另加第四个`I32 IDs[T,S]`端口，
对应`A[T,1或S,K], W[E,N,K/2], SF[E,N,K/32] → BF16[T,S,N]`，没有WG／IG。
默认`Mxfp4ActivationMode::kFloat4Group32`是[研究合同选定的portable BF16 W4A4数学](../evidence/research/quark-mxfp4-contract-2026-10-20.json)：
A按group32的amax指数舍入确定power-of-two scale，经BF16除法、保留符号的E2M1 ties-to-even饱和量化及BF16重建；
W保留相邻偶K低nibble／奇K高nibble，码乘E8M0 scale先舍入到BF16，再作F32点积及BF16输出。
零与极端值按该合同定义，不照搬上游HIP／Triton的差异或未定义转换行为。
SF0为非零的2^-127；非finite A或SF255由Completion报错，合法有限输入产生的IEEE溢出仍是数值结果。
grouped先检查全部ID再形成专家地址，SF只检查选中专家；重复ID保留slot，A支持零slot步幅及token行距View。
CPU/CUDA使用调用者激活工作区：dense为2×M×K字节，共享A1／零slot步幅为2×T×K，独立AS为2×T×S×K。
Apple/Vulkan沿原生Program准备相同激活临时量；Apple的检查数组及managed状态、Vulkan的状态缓冲与运行时开销另计。
四后端复用既有FP4 family和执行机制，原ModelOpt／Local四个入口的数学保持不变。
[37项数值、寿命及FP4邻居回归](../evidence/correctness/quark-mxfp4-reference-2026-10-20.json)通过且无跳过；
该证据不覆盖所选制品payload／整模型、硬件FP4 GEMM、质量或吞吐。

两工厂也可传`Mxfp4ActivationMode::kBFloat16`选择W4A16：直接读取原BF16 A，
权重仍先按E8M0 scale解码并舍入到BF16，再作F32点积；不能替换为“分组点积后乘scale”。
它沿用上述端口、stride、ID和错误约定，但不分配激活QDQ暂存；状态与后端运行时开销另计。
原F32 micro32 scale可先用`encode_e8m0_scales(source, output)`冷转换成同shape的dense U8 E8M0，
再由调用者上传一次。此host入口要求稳定canonical地址、正finite F32、输出无alias；按GRS nearest-even数值转换，
SF0仍非零，源零／负值／NaN／Inf及舍入到255均在整grid写前拒绝。它只转换scale，不复制或展开W。
[固定reader依据](../evidence/research/inferact-micro32-contract-2026-10-21.json)与
[四端数值、捕获及寿命实测](../evidence/correctness/mxfp4-weight-only-reference-2026-10-21.json)
对应有限域的软件参考；未读取Inferact权重payload或取得原exporter／OAS规则，不代签整制品、硬件FP4或吞吐。

MLX 使用 `MlxQuantized{format, kind, bits, group_size, orientation}`；默认是 `kAffine/kLinear/4/64/kTransposeWeight`。
affine 的 bits 为 4/5/6/8、group 为 32/64/128；`kMxfp4` 固定 bits4/group32，
`kMxfp8` 固定 bits8/group32。每行按 K 顺序低位在先连续打包，5/6 位码可跨 U32，
不能用整数 `32/bits` 推导一行容量；scale/bias 为 `[N,K/group]`，grouped 另有 expert 首轴。
线性与专家输入 A 为 F16/BF16/F32；affine A/scale dtype 相同时保持该类型，混合时输出 F32，
解码权重及累加保持 F32，最后按输出 dtype 舍入。MX 输出保持 A dtype，其 E8M0 scale 先转为
A dtype，保留原低精度溢出或下溢。integer affine8 与浮点 MXFP8 是两个编码合同。

上述 MLX 基础算子已接通 CPU、CUDA、Apple 和 Vulkan；Vulkan 的设备特性、workspace 与容量资格仍按实际设备检查。
Apple的完整通用5/6/4bit MoE组合现也已[实际执行](../evidence/correctness/precise-role-consumers-reference-2026-10-21.json)，
覆盖两种专家数、双槽与更新后的旧结果；不代表某个GLM整层。

`mlx_quantized_matmul`／`mlx_grouped_matmul`的末参数可传`MlxMatmulOrientation::kWeightAsStored`，
表达affine的`A*W`：物理压缩W仍为`[N,K]`，scale/bias仍沿K分组，A末维变为N、输出末维为K。
默认`kTransposeWeight`表达`A*W^T`，A末维K、输出末维N。grouped的IDs、共享A与slot规则不变。
只有物理K要求按group对齐，N允许尾块；不对压缩bank作物理转置或整bank浮点展开。
MX及embedding不接受`kWeightAsStored`。同一个head-bank的两方向、显式小query Copy、
四端数值／旧结果及CUDA捕获边界见[本批实测](../evidence/correctness/mlx-weight-orientation-reference-2026-10-21.json)。
这是现有F32 affine解码／点积合同的有限回归。另有[GLM首层原payload双head角色](../evidence/correctness/mlx-head-payload-reference-2026-10-21.json)的四端消费、头序与结果寿命检查；未签上游原生低精度kernel全域逐位等价或整制品。
[CPU schema/Direct/Graph 17 项](../evidence/correctness/mlx-affine-widths-cpu-2026-10-19.zip)、
[CUDA 23 项及公开 Direct/Graph](../evidence/correctness/mlx-affine-widths-cuda-2026-10-19.zip)、
[Apple Direct 5 项](../evidence/correctness/mlx-affine-widths-apple-2026-10-19.zip)是定向算子回归；
Apple 的一般乘加收缩、typed embedding 两次独立舍入和跨后端逐 bit 等价仍未证明。
[Vulkan 18 项](../evidence/correctness/mlx-affine-widths-vulkan-2026-10-19.zip)含 Direct8/Graph3/private IREE7，另覆盖12个位宽/算子交叉域和8个F32解码反例；这些结果不表示已导入或运行任一完整 MLX 量化制品，也没有新增吞吐结论。

这组实现借用固定 bitsandbytes、AutoAWQ、compressed-tensors、MLX 的原计算组合，由 ATen CPU 承载。
它们属于 managed replay，存在原库临时分配与线程成本；不保证 capture、零热分配或未知私有内存的硬预算。
越界索引和分配失败通过 Completion 报错，已失败的执行槽需要重建；独立槽可以共享只读参数。该资格仍按真实packing/group/dtype和端口区分，
不把一个文件名的“4-bit”当作所有张量格式，也不代签全部混合模型完整片段。

CPU-only LibTorch启用时，`model_ops::causal_conv1d()`的F16/BF16输入现在由原ATen CPU程序承载。
samples为`[B,W,C]`、weights为`[C,K]`，二者dtype相同；F32累加后一次回原output dtype。
原`[C,1,K]`文件权重由调用方用View整理。该managed路径包含布局/精度转换、原库临时与结果copy，
不能要求capture、零热分配或已知严格私有内存界；失败slot需要重建。F32继续走原ggml路径。

CUDA 的相同公开操作支持 dense F32 输入。ggml 承载通道数为128倍数且 K 为3／4／5／9／15的形状；
其它合法通道数与卷积宽度自动选择 native CUDA 后备，无需调用方选择provider。
后备使用F32有序乘加、零workspace和原共享队列，支持prepared replay／capture；
地址范围及设备限制仍在冷准备时检查，不扩展公开F16／BF16输入的CUDA支持范围。

## 设备端单选与可定位随机数

确定性单选直接使用原 `TopK{.count = 1}`，仍返回原存储类型的 values 与 I32 indices，
同分取首索引、NaN 排在非 NaN 后。CPU、CUDA、Apple 和 Vulkan 的适用形状使用归约；
其它 count 和原先合法的 axis/stride 域沿各路线原合同执行，不需要额外的 greedy API。

`hyle/compute/sampling_ops.h` 提供两种独立数值操作，沿原 Direct prepare 或 Graph builder 使用：

```cpp
#include <hyle/compute/sampling_ops.h>
namespace c = hyle::compute;
auto words = c::sampling_ops::random_bits(8);
auto selected = c::sampling_ops::categorical();
auto scored = c::sampling_ops::categorical(true);
```

`random_bits(W)` 接收 dense U32[R,4]，每行依次是 key 的低/高32位、draw 的低/高32位，
输出 dense U32[R,W]。固定 Philox4x32-10 映射让相同 key/draw/word 地址在已验证路线间逐位
一致；改变请求行顺序或 W 不改变原行的字前缀。W≤INT32_MAX，零行/零W合法，draw 接受全部
64位值。调用方维护请求身份、draw 唯一性与溢出政策，操作不会隐式前进 counter 或持有全局 RNG。

`categorical()` 接收 dense F16/BF16/F32[R,V] log-weights 和同一 U32[R,4] counter，
V 为1..INT32_MAX，输出 I32[R]。`categorical(true)` 另返回 F32[R]：选中项归一化后的
log probability。-Inf 表示零质量；NaN、+Inf 或任一全-Inf行通过原 checked 完成报告
InvalidArgument，失败不发布可用结果。零行合法。先观察成功完成，再读取或向下游发布结果，
并按原 invocation 的失败恢复与结果寿命合同复用槽。

CPU 使用 SIMD exp、F64 累加的 CDF；GPU 使用居中 log-weights 与开放64位 midpoint 的
Gumbel 选择。同一路线、相同输入/counter可重放和重排；不承诺 CPU 与 GPU 选择相同 ID。
两者扫描词表，复杂度 O(RV)；可选 log probability 增加所选标量的归一化计算，GPU未请求时
省略对应normalizer。临时资源、native
完成和少量结果读取沿原路线 owner，不能由“设备端选择”推断零分配、零等待或新增 capture 支持。
temperature、top-p、penalty、grammar、EOS 等政策由应用组合，本数值域不接管这些政策。

没有据此声明完整推理引擎的 prefill/decode 吞吐。

## 各自选择 KV 的带 sink attention

`model_ops::sink_attention(scale)` 接收 dense F32 的 Q `[B,Q,H,512]`、K/V `[B,S,1,512]`、
additive mask，以及各 batch 共享的 sink logits `[H]`，其中 H 为64或128。mask 可为共享的
`[Q,S]`，或各 batch 独立的 `[B,Q,S]`；sink 只参与 softmax 分母，输出与 Q 同形。

多个 query 各自选择不同历史行时，可用普通 `Gather(axis=0)` 将共享 cache `[L,512]` 和
IDs `[T,S]` 组成 `[T,S,512]`，再用 View 整理成 `[T,S,1,512]`，与 Q `[T,1,H,512]` 一次计算。
无需为每个 query 建一个 Sink 节点再 Concat。选中 KV 仍会物化；F32 数学路径仍有 score 临时量，
Apple 的此 D512 域沿 MLX fallback，不是新的原生融合 kernel。这也不替代分页 arena 的读取权限或
压缩历史的物理 codec；分页联合承载见[批次边界](batch-attention.md)。

## CPU 多请求 ragged attention

`hyle/compute/batch_execution.h` 的 `BatchProgram` 在普通 `cpu::create_context()` 中冷选择
原 ggml flash。当前候选域为 F16/BF16、GQA、D64/128/256，packed NHD Q/K/V。
plain 同类型 K/V 另支持512→512、576→512，QK 与 V 深度分别由 schema 描述。
`BatchSchema::version()` 拥有 query/KV indptr，空 Q request 可跳过，非空 Q 必须有足够
的 KV。容量描述实际存储，indptr 描述这一次的有效请求区；未用尾部不参与数学。

```cpp
auto program = c::BatchProgram::prepare(context, schema,
    {.resource = hyle::mem::system_resource(),
     .max_float_workspace_bytes = 8u << 20,
     .max_int_workspace_bytes = 4096,
     .scale = scale, .causal = true}).value();
auto slot = c::BatchInvocation::create(program, queue).value();
auto prepared = slot.prepare_batch(layout).value();
auto done = prepared.run(std::array{query_read, key_read, value_read}).value();
HYLE_RETURN_IF_ERROR(done.wait());
auto output = slot.output().value();
```

先释放旧 output、prepared token 和 Completion，再调用 `update_batch(next_layout)`。
完整 query/KV topology 相同可复用原 native plan；改变则在这次显式更新中重建，run 不建图。
原 kernel 直接读取每个 request 的 K/V slice，没有按每 Q/head 物化历史；冷 mask、query
转换与每请求 metadata/workspace 仍计费。CPU `float_workspace_bytes` 暂为已知冷 plan
占用的保守合计，与外部预留 pool 分开，不能当作总 RSS；private pool 上界未知。
分页使用下节的同一 `BatchProgram` 入口；CPU 路线当前不支持 capture。

## CPU 准确页直接读取

普通 CPU `BatchProgram` 新增的分页路径沿原 ggml flash 读取物理 K/V 页，使用
`BatchStorage::kPaged`、`PagedArena` 和同一个 `BatchLayoutVersion`。当前验证域是
F16/BF16、GQA8/2、D64/128/256、page_size16；页表可重排、跨请求共享物理页，尾长
只表示最后一页的有效行数。
plain 高维另已验证512→512的H64/KV1与576→512的H96/KV1，两请求混合Q2/Q1；
Q8_0 仍保持其原有限几何。

K/V arena 按容量持有连续物理 NHD 页；发布每页 writer 后，用该 layout 所引用的
准确 ready page 集合构造 frame，不能传多余页或把 arena 当作一个普通 ReadLease：

```cpp
auto key_frame = keys.read(layout, ready_key_pages).value();
auto value_frame = values.read(layout, ready_value_pages).value();
auto prepared = slot.prepare_batch(layout).value();
auto done = prepared.run(query_read, key_frame, value_frame).value();
HYLE_RETURN_IF_ERROR(done.wait());
auto output = slot.output().value();
```

更新前释放旧 output、prepared token 和 Completion。CPU paged 每 request 为0或1 query、
完整 query indptr 相同时，KV indptr、页索引和尾长可通过显式 `update_batch()` 更新，复用
原 graph/pool/scratch；其它 paged 形状仍需完整布局相同，否则冷重建。frame
必须属于当前 schema/layout，所有 reader 和实际 native uses 退休后才能复用物理页。
CPU ragged 在相同 Q1 拓扑下也能更新实际 KV extent 与 packed 起点；仍绑定原三个
packed-capacity Q/K/V read ports，内部逐 request 检查实际子范围，不复制完整 KV。
热数学不按 query/head 复制整个 KV 历史；query 转换、冷 causal mask、页表节点与原
metadata/workspace 仍计费，private pool 上界未知。CPU capture、增量 producer、低成本
增长规划及 GPU/联合分页的资格各有独立范围，不由此 CPU 路径推导。

## 外部资源租约与 Direct 提交

`hyle/compute/resource_leases.h` 的 `LeasePool` 冷创建有限 backing/块 metadata，
`WriteLease` 表示独占写，成功发布后得到可复制的 `ReadLease`。两个读者、原生在途 use 或旧结果
存在时，不得把同一块重新取得为 writer。非重叠块可独立使用；池的 resource、Context 和 allocator
仍按公开声明的借用寿命保持有效。

`ReadLease::generation()` 标识租约块的分配身份，不是内容写入次数；
唯一读租约经 `try_write()`、修改、再发布后可以保留同一值。
缓存内容还须持有原不可变读权限，或另有真实内容代际与失效机制，不能只比较这个数字或buffer地址。

CPU/CUDA 的原 prepared operation 可用 `hyle/compute/leased_operation.h` 自动保护 bindings：

```cpp
auto lane = hyle::compute::LeasedOperationInvocation::create(
    prepared, std::move(workspace_write_lease)).value();
const std::array inputs{hyle::compute::LeasedInput(input_description, input_read_lease)};
const std::array outputs{hyle::compute::LeasedOutput(output_description, output_write_lease)};
auto done = lane.submit(queue, inputs, outputs).value();
HYLE_RETURN_IF_ERROR(done.wait());
auto ready = output_write_lease.publish().value();
```

这里的端口数量、描述和资源必须符合已准备的 `prepared`；零 workspace 时省略 workspace lease。
初始化 host bytes 后才 publish；异步 writer 必须实际成功退休后才 publish，失败不会变成 ready。
`LeasePool::adopt`/`ReadLease::adopt` 接管唯一 `mem::Buffer` 而不复制 payload，拒绝不消费输入；
readonly adopt 永久只读。原 native buffer/array/tensor 的受控导入见各路线的 memory 头。

固定状态项可用 `<hyle/compute/lease_slots.h>` 的 `LeaseSlotLayout` 描述同一池内的
item、槽数和字节 stride。`create(pool, item, slots, stride)` 不分配第二份 payload；
`locate(lease)` 只接受该原 domain 中、处在完整槽边界的普通整块租约，返回槽号和 generation。
部分视图、stride 中的 padding、其它池及 append 权限均被拒绝。
`bind_indices` 对实际 source reader 和连续 destination writer 集合先完整检查，再无分配地填入
调用者提供的地址数组；失败保持数组和租约不变。源可重复，独占目标不得与源或其它目标重叠。
地址不是权限，后续仍须保留真实租约并通过原 `LeaseUseBatch` 接纳和退休。
池关闭后仍可查询保留租约的地址事实，但不能恢复新 native 使用；该接口本身不执行 Gather／Store。

已经通过池完成初始化时，可对唯一、覆盖整个 backing 的普通 writer 调用
`pool.freeze(std::move(writer))`，直接得到永久只读的租约；成功同时清空 pool 和 writer，
不复制 payload，仍允许后续只读原生提交。尚有原生在途使用、部分块或失败写入时拒绝，
保留原句柄；resource、Context 和 allocator 的借用寿命不变。
Apple的`import_readonly`与原生Graph结果支持plain非连续只读视图：要求ready、shape／dtype／有效stride与完整地址span相符，保留原Array和结果寿命，不复制payload。
借用输入的视图保持借用权限，已有拥有者的视图沿原lease范围导出；编码carrier仍要求原精确dense布局。
所有未跟踪的 writable aliases 必须停止，不从裸 view 猜拥有权或就绪。

`lane.close()` 停本 adapter 接纳，`lane.drain()` 先永久停止接纳，再观察已接受的原 Completion；调用方可以仍持有
尚未观察的 `done`。一个冷完成槽要求先观察/丢弃旧 Completion 再 replay。控制与 Completion
观察须外部串行化。未知错误不证明 GPU 已停，保活仍交给原 runtime；取消也不能提前释放 backing。
池的 drain 也先永久停止接纳，再等待 admitted uses；保留 reader 仍有效，不等待这些 reader 全部销毁。
需要继续重放时等待本次 Completion，不调用 adapter 或池的 drain。

Apple/Vulkan 的 native Program 使用各自 memory 头里的 `LeasedInvocation` 与 `LeasedInput`；
bind 冷建立 readonly native view，submit 自动接纳 use，原 native slot 的实际退休归还 use。
结果/来源 alias 继续由原 native owner 保留。四路线接口独立，不需要通用 Queue 或自写 completion vector。不外推未测设备/工具链。

Graph 在原 `Invocation` 上准备资源绑定，使用 `<hyle/graph/executable.h>` 和
`<hyle/graph/result.h>`：

```cpp
namespace g = hyle::graph;
auto slot = executable.create_invocation({
    .resources = g::ResourceOptions{.allocator = metadata_allocator}}).value();
HYLE_RETURN_IF_ERROR(slot.bind(input_id, input_description, input_read_lease));
HYLE_RETURN_IF_ERROR(slot.bind(output_id, output_description,
                               std::move(output_write_lease)));
auto done = slot.submit(queue).value();
HYLE_RETURN_IF_ERROR(done.wait());
auto result = slot.result(output_id).value(); // GraphOutput
auto view = result.tensor().value();
```

声明 shape domain 的 Executable 在同一 `InvocationOptions` 中填写 `.shape`。
已有普通 slot 可在冷路径调用 `slot.prepare_resources(ResourceOptions{})`，随后使用相同的
bind、prepare_replay 和 submit；不另建执行器。`GraphOutput` 及其复制保留原 slot 和结果 backing，
阻止同槽 reset、rebind 或重放。`storage_lease()` 导出调用者拥有或执行来源明确允许导出的实际 backing，
范围仍受原租约约束；私有 workspace 和 Executable 常量不能凭地址导出独立租约。
独立 `ReadLease` 继续保护背存，固定输出程序在旧 lease 存活时会拒绝准备覆盖；`reset()` 成功不表示背存已可写。
同时保留多代状态时，冷预建独立 `Invocation` 承载所需代数，按实际读者退休复用，无需判断 Provider。
`slot.output(value)` 仍返回借用的 `ConstTensorView`；它不代替持有结果，寿命到下一次修改槽或槽销毁。

## 准备批量范围搬运

`hyle/compute/batch_transfer.h` 的 `CopyBatchProgram` 冷固定 CPU/CUDA Context、资源域、
范围/阶段上限和 metadata budget；每个 `CopyBatchInvocation` 是独立运行槽，借用一个地址稳定的
原 Queue。Apple/Vulkan 使用各自 memory 头中的同名类型及原 native Context，创建槽时省略 Queue。
resource、Context、Queue 和自定义 allocator 均须活到所有槽与原生在途工作退休。

```cpp
namespace c = hyle::compute;
const std::array resources{
    c::CopyResourceSpec{host_resource, 4096},
    c::CopyResourceSpec{device_resource, 4096}};
auto program = c::CopyBatchProgram::prepare(context, resources).value();
auto slot = c::CopyBatchInvocation::create(program, queue).value();
HYLE_RETURN_IF_ERROR(slot.bind_source({0}, source_read_lease));
HYLE_RETURN_IF_ERROR(slot.bind_mutable({1}, std::move(destination_write_lease)));
const std::array<c::CopyPhase, 1> phases{{{0, 7}}};
const std::array<c::CopyRange, 1> ranges{{{{0}, {1}, {0}, 0, 0, 4096}}};
HYLE_RETURN_IF_ERROR(slot.prepare_replay(phases, ranges));
auto copied = slot.submit().value();
HYLE_RETURN_IF_ERROR(copied.wait());
auto ready = copied.output({0}, {1}, 0, 4096).value();
// ready.tensor(description) 可交给已有 Direct/Graph 租约消费者。
```

同阶段的范围必须独立：重复读取 source 合法，交叠写及读写冲突被拒绝。A→B→C 使用两个连续阶段，
各后继的 `dependency_mask` 指向实际先行阶段；数组顺序不能代替依赖。`group_tag` 是调用方标识。
一个 block 的最后写必须实际成功退休，`output` 才返回被成功复制的 logical 范围；pending、失败、
取消与未知退役均不发布 ready。较早组可独立观察，后继 readonly 搬运仍保活它。
生产者有尚未退休的写时，用 `bind_producer` 交接同 native owner 的实际 Completion；未知外部
完成不能猜成 native fence。失败绑定不消费传入 lease/Completion。
Apple/Vulkan 的原 native dependency 接纳会拒绝已经请求取消且仍在途的来源，
返回 Cancelled 并保留传入 Completion。取消不证明退休；接纳之后才发生的取消仍
按原 native 完成合同处理。排序本身也不替消费者合并来源与后继的最终错误。

搬运保持原编码，不做 dtype 转换或量化重打包。`physical_bytes=0` 为 exact copy；显式 padding
必须属于源和目标的已分配范围，计入 alias 与 byte budget。Metal 的 offset/size 为 4-byte 对齐；
Vulkan 也必须符合原 buffer copy 约束，不能以向上取整越过邻块。
`CopyBatchReport` 分别给出 logical/physical bytes、实际 copy calls、native dependencies、阶段完成
和 host waits；`native_metadata_unknown` 表示 vendor 内部成本未知，不能据此宣称所有热分配为零。
Apple 沿原 MLX owner 同步实际 command buffer，每 native phase 的 worker host wait 会计入报告。

旧 submission 保留运行槽；观察、释放它后显式 `slot.reset()` 再绑定下一批。保留的 ready reader
仍拥有旧数据。`close()` 停止接纳，`drain()` 先永久停止接纳，再观察实际工作；两次重放间等待
`CopyBatchSubmission`。析构放弃观察时不会隐式 host wait，
原 native owner 继续保活资源。设备失败的槽不承诺自动恢复，应按返回错误和实际退役情况重建。
两组/两槽及原编码数学组合见[批量搬运样例](../../samples/package/graph/batched_transfers/)。

## 已接纳 copy 的单一观察所有权

需要把一组已接纳更新交给后继持有时，用 `CopyBatchUpdate::take(std::move(slot),
std::move(submission))` 合并两个原控制。CPU/CUDA 使用 `compute::CopyBatchUpdate`；
Apple/Vulkan 使用各自 namespace 的同名类型。两个控制必须来自同一实际运行槽；
失败不消费它们，成功同时移空 caller。该对象只保存两个原 handles，不创建队列、
Completion 或执行线程。Context、Queue、资源与 allocator 须活到所有实际工作退休。

同一个 update 只有一个串行观察者。`observe()` 非阻塞并返回原 Status；Unavailable
不等于成功，也不能只按错误码断言 native 工作仍在途。`wait()` 是显式 host 等待，
`request_cancel()` 沿原提交请求取消；析构仍沿原控制放弃观察，原 native owner 保活。
成功退休后 `take_mutable(id)` 返回原 caller-owned writer，保留未改写内容和 generation，
不扩大批量搬运 `output()` 的成功 logical coverage。`release_invocation()` 只在实际原 copy
退休后回收原 cold slot；失败/closed 状态保留，回收控制不意味着数据可以发布。

普通用户不能由 update 得到 pending ReadLease 或 host 可读 view。内部 receiver 借用的
原 native prefix 与 mutable geometry 必须先核实际 endpoint/ordering，并保持原 update
到后继实际退休；借用期间不能另观察、回收或销毁 source。

## 有界文件与资源管线

文件管线是显式的 `compute_stream` 服务产品。源码选择
`HYLE_COMPONENTS="compute;compute_stream"`（使用 Graph 时为 `"graph;compute_stream"`），
路线仍由 `HYLE_COMPUTE_ROUTES` 指定；安装包请求相同组件，并链接 `hyle::compute_stream`
与实际使用的 `hyle::compute` 或 `hyle::graph`。只用常驻计算的 Compute/Graph 不引入文件 IO。
该服务使用 `io_core`，不要求选择完整的 IO 协议产品。

`hyle/compute/file_stream.h` 的 `FileResourcePipeline` 把原 IO、资源租约与批量搬运组合成
分块文件旅程。另包含所选路线的 `hyle/compute/cpu_file_stream.h`、`cuda_file_stream.h`、
`apple_file_stream.h` 或 `vulkan_file_stream.h`，调用对应 namespace 的
`create_file_pipeline`；默认计算头不再声明文件工厂。CPU/CUDA
沿原 Context、Queue，Apple/Vulkan 沿各自 Context。创建时明确 stage resource、目标资源域、
槽数和有限 byte/request/group 容量。CUDA stage 使用原 pinned host resource；不把整个模型 pin。

`StreamFile::adopt` 接管已有 `io::File`，`borrow` 保留调用方地址稳定的 File；两者都需明确合法容量。
File 与原 IoContext 的 owner worker 合同保持不变。`StreamSource` 也可接管或借用稳定只读
`mem::FileMapping`，其范围偏移相对于映射的 logical bytes，期间不得修改或释放映射。
模型格式由消费者解释；管线只接受已核验的 offset、bytes、description、group/tag 与目标 WriteLease。

```cpp
namespace c = hyle::compute;
std::array<c::StreamReadRange, 1> ranges;
ranges[0].file_offset = checked_offset;
ranges[0].bytes = checked_bytes;
ranges[0].destination = std::move(weight_writer);
std::array<c::StreamReadGroup, 1> groups{{{layer_tag, ranges}}};
auto ticket = pipeline.read(c::StreamSource::file(source), groups).value();
auto group = ticket.wait_next_ready().value();
auto weight = group.output(0).value();
// weight 可直接绑定已有 Direct/Graph 租约入口；不必等待其他独立组。
auto result = ticket.wait();
HYLE_RETURN_IF_ERROR(result.status);
```

接纳失败不消费输入权限。一个 group 成功才返回 ready；后组 EOF/失败不会撤销前组结果。
旧 ReadyGroup、ReadLease 与 native use 继续占有限容量，需要释放并 `collect()` 后才能复用。
回写用 `StreamWriteRange::ready`，有原生生产者在途时用 `produced` 转交 WriteLease 与原 Completion；
实际 producer 退休之前不会读取资源。文件短写保留实际 prefix，原子文件发布由应用选择。
完整 read→math→write→reload→math 和本地 Qwen checked ranges 见
[普通 C++ 样例](../../samples/package/graph/file_streaming/)。

取消/deadline 在 chunk 边界停止后继，已接纳的 blocking IO/native 工作必须先退休，返回可能晚于 deadline。
析构放弃观察并停止后继；需要等待本管线实际工作时显式 `drain()`。Context、Queue、resource 与
allocator 必须活到所有原生 use 和保留结果结束；不会关闭调用方的整个 runtime。
未知原生退役保持隔离与 `retirement_proven=false`，不能以逻辑释放推断 GPU 已停止或驱动私有内存归零。

## 准备时绑定实现

普通计算不需要用户提供 timing profile。四路线的默认优选规则由 HYLE 研发阶段验证并固定；
准备时匹配适用条件和资源约束，执行时复用已绑定制品，不计时、不重选。各路线当前能力以其
公开入口与实际准备结果为准。

`PreparedOperation::implementation()` 返回 Provider 声明的稳定 `ProviderImplementationId`；
`route_policy()` 与 `route_preference()` 返回本次绑定的版本化规则及优先级。空 policy 表示通用、尚未具名的
基线路线。这些值用于解释实际绑定，不是可持久化的性能结论；普通调用不需要构造规则。

## CPU 默认规则与显式实现约束

普通 `cpu::create_context()` 组合合法的 oneDNN、ggml、ATen 与保留的 Native 实现。
准备期检查真实请求并固定路线；Graph 尽量保留合法完整 region，不为逐节点偏好拆散融合。
不要从底层库的能力目录推断 HYLE 已接入所有类型，也不要要求已移除的 Native 数学充当后备。

通常只需默认入口。需要诊断时检查 `PreparedOperation::provider()`、`runtime_properties()` 与
`workspace_requirement()`。确实必须限定某家实现时才创建专家 Context，例如：

```cpp
#include <hyle/compute/providers/onednn.h>
auto context = hyle::compute::Context::create(
    hyle::compute::cpu::device(), hyle::compute::onednn::provider());
if (!context) return context.error();
```

不支持的形状在准备期失败，专家限定不自动追加其它 Provider。精确 alias、stride 与数据类型
仍由该实现检查，名称相同不意味着资源或精度相同。

### 接入已有 executor

`cpu::DefaultContextOptions::executor` 借用 `ParallelExecutorRef`，默认使用调用线程。
executor 的 `parallel_for` 必须在返回前完成全部工作，不能失败或抛异常，并在嵌套调用时保证进展。
其 backing 活过 Context 及所有派生执行对象。分片与 scratch 由 prepare 决定，调用者按查询的
工作区要求分配，不依赖内部 tile 阈值。完整 Scheduler 组合见
[持久 executor 样例](../../samples/package/graph/cpu_executor/)。

`native_threads` 是 ggml 的独立预算，默认 1（含提交线程）；显式大于 1 才为 invocation 准备
持久 workers。executor concurrency 不会自动变成该预算，也不统一控制 ATen 的共享线程池。
并发多个 invocation 的线程/内存用量要一起考虑。

### 限制共享 CPU runtime 线程

在任何相关 HYLE Context 尝试创建或外部 ATen 首次使用前，应用可以配置共享 runtime：

```cpp
namespace cpu = hyle::compute::cpu;
HYLE_RETURN_IF_ERROR(cpu::configure_runtime_threads(2));
auto prefill = cpu::create_context({.native_threads = 2});
auto decode = cpu::create_context({.native_threads = 1});
auto runtime = cpu::runtime_thread_info();
if (!runtime) return runtime.error();
```

`configure_runtime_threads()` 控制共享 ATen runtime 的 intra-op 设置，非 Context 局部值。
`runtime_thread_info()` 只观察 backend、当前 caller 的 limits 和配置窗口；未初始化时 limits
可为 unknown，查询不会为观察而初始化 runtime。它不统计 live OS threads，也不限制多个并发
OpenMP team、executor、ggml worker、BLAS 或 GPU service thread 的总和。

首个相关 Context 尝试会关闭配置窗口，即使创建失败也不重开；Context 销毁不恢复设置。
相同显式值只在实际 getter 仍匹配时幂等。晚配置、不同值和 setter 部分失败明确拒绝，不回滚或
自动修复 foreign 漂移。应用须协调其它 ATen 用户与多个静态 HYLE 副本，不能假定每份静态库
拥有独立的共享运行时。

### 数值与内存的例外

- 稳定激活、mask、特殊值、低精度累加可能需要 ATen 或显式 F32/Double 转换，形成真实的
  packing/临时存储与融合边界；不能为了少一次转换换成另一种模型数学。
- 原库 norm 等 managed 程序可以在 replay 中建立 metadata；某些布局只有 managed 候选。
  prepared-only 或拒绝未知内存的要求可能没有可行实现，正确行为是报错。
- 原编码矩阵、paged KV 与普通 dense 矩阵各有资源及格式限制，不能用全量反量化结果证明
  直接压缩读取的性能。接口导航见前面的模型格式和批次章节。

## 所有权、失败和限制

- `Backend`、`Device` 和 `Provider` 是可复制的进程内值；它们以开放的双字 ID 标识实现，不要求
  修改中央枚举。内置 Provider 配置 inline；custom Provider 复制时保留其 owning factory（冷引用计数），
  identity 的 name/version view 借用这个 owner。`Context`、`Queue`、`OperationInvocation` 与 `Completion` 是 move-only；
  `PreparedOperation` 可复制共享。
- `Context::close()` 与 `Queue::close()` 原子地停止新 admission；这两种对象的 `drain()` 只显式等待已接纳工作，
  不停止后续接纳，需要终结时先 close 再 drain。租约 adapter 与池的 drain 则会先永久关闭接纳。析构只
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
重复执行成本由实际 `runtime_properties()` 声明，不能由库名或一次局部测试推断。
其它共同借用、失败与预算规则见[调用安全边界](usage-contract.md)。

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
- `KernelProposal::route` 是本次 propose→prepare 的临时 token；可选
  `implementation = ProviderImplementationId{1}`
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
- `kernel.enqueue(invocation, sequence, inputs, outputs)` 接收 `compute::detail::QueueSequence &` 并返回
  `KernelCommandResult`，复用已 admission 的原 HYLE Queue sequence 执行/记录。CUDA kernel 通过
  `cuda::stream(sequence)` 取实际 stream；不要再调用 `submit_host/cuda::submit` 嵌套 admission，也不要创建第二个 Completion。
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
