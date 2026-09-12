# HYLE Compute：何时直接执行一个计算

> 本页描述当前 CPU C++ surface；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。HYLE-native 始终可用；选择
> `HYLE_COMPUTE_ONEDNN=ON` 时可以在 x86-64 构建显式 oneDNN MatMul provider。该可选 factor 已
> `CLOSED`：BRGEMM ukernel 已通过 Windows/Linux 正确性与三交付、暖执行零分配、Linux sanitizers、
> 正式 P1、两台 Apple ARM 负能力、离线依赖/许可证和终局红队。Apple Silicon 等 AArch64 目标会关闭
> 该 factor 并
> 使用 HYLE-native；CUDA、Metal、Vulkan 与其他已准入 providers 仍是后续独立纵切。HIP/ROCm 当前
> 无限期不纳入产品，也没有可启用的空开关。

当程序已经知道要执行哪个 operation、在哪个 device 上执行，并希望复用固定 shape/layout 的路线时，
使用 Compute。跨节点拓扑、liveness 和可复用图执行物使用 [Graph](graph.md)；模型、KV cache、batching、
sampling 和 serving 留在推理引擎。

## 当前能力

| 任务 | 公开形状 | 当前合同 |
|---|---|---|
| 设备发现 | `enumerate_devices` / `query_device` | 当前只返回 CPU；短输出 span 不静默截断 |
| 执行域 | `Device -> Context -> Queue` | CPU Queue 在调用线程同步、按序执行；没有隐藏 worker/default device |
| tensor schema | `dense/strided_bytes/blocked_dense/blocked_strided/opaque` | rank <= 8；logical dtype 与 physical encoding 分离；stride 明确以 byte 为单位；view 借用 Mem storage |
| operation schema | 19 个 typed family + inline custom | 可做 arity/shape/layout/alias/effect/precision 检查；schema 存在不等于已有可执行 provider |
| normalization | `RmsNorm` | dense f32 `[..., hidden]`；可选 dense f32 weight `[hidden]`；逐 row |
| matrix multiply | `MatMul` | f32 rank-2 `A[M,K] * B[K,N]`；B 可 row-major 或物理 `[N,K]` 的转置 view |
| gated activation | `GatedActivation` 的 packed SiLU mode | dense f32 `[..., 2*d] -> [..., d]`，前半 gate、后半 value |
| 自定义 CPU 工作 | `submit_host(queue, callable, args...)` | `noexcept` callable，返回 `void` 或 `Status`，返回前完成 |
| provider 选择 | `Context::create(device, Provider)` / `ContextOptions::providers` | 显式且有序；只有 `Unimplemented` 才尝试下一项，HYLE-native 是最终 fallback |
| 稳定重复执行 | `prepare -> PreparedOperation -> OperationInvocation::submit` | prepared artifact 可共享；每个并发 lane 独占 invocation/workspace；submit 不重新推 shape/路由 |

公共 schema 已能描述 View、Copy/Fill/Cast、elementwise、reduction、normalization、batched MatMul、
gated activation、Gather/Scatter/Concat、RoPE、Softmax、TopK 和 SDPA 共 19 个 family，并能做纯描述推导。
**当前可执行 provider 仍只签发 RMSNorm、rank-2 MatMul 和 packed SiLU GatedActivation 三项。** Tensor
schema 能描述 block/opaque storage 也不等于已有量化 kernel；对其他 family 调用 `prepare()` 会诚实返回
`Unimplemented`，直至相应 native/provider 卡同时交付实现与正确性证据。

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

不要增加 provider-specific `transpose_weight` 标志。若模型存储权重为连续 `[N,K]`，创建该 dense desc 后
调用 `transposed()`，得到逻辑 `[K,N]` 和对应 byte stride；数据不移动。Compute 会在 prepare 冷路径选择
相应 route。普通连续 `[K,N]` 也直接受支持。Block-compressed `[N,K]` 使用上一节相同的 transpose 方式；
当前 HYLE-native 与 oneDNN 不解释它，只有以后显式加入的量化 provider 会接收。

```cpp
auto stored = TensorDesc::dense(DataType::kFloat32, std::array{n, k});
if (!stored) return stored.error();
auto logical_weight = stored->transposed();
if (!logical_weight) return logical_weight.error();

auto output = infer_single_output(
    Operation(MatMul{}), std::array{activation_desc, *logical_weight});
if (!output) return output.error();
auto prepared = prepare(context, MatMul{}, activation_desc,
                        *logical_weight, *output);
```

## prepare、invocation 与 submit

`infer_output_descriptions()` 是 direct Compute 与 Graph 共用的 shape/layout 规则；单输出 operation 可用
`infer_single_output()`。`prepare()` 再核对
调用方给出的 output desc，并把 provider/route 选择移出热路径。对同一描述反复执行时共享
`PreparedOperation`，为每个并发执行 lane 创建并复用一个 move-only `OperationInvocation`。当前故意
没有把 prepare 或 invocation 创建隐藏在一次性 convenience 中，因为 provider workspace 和并发状态必须
有明确 owner。

```cpp
auto context = Context::create(Device::cpu());
if (!context) return context.error();
auto queue = context->create_queue();
if (!queue) return queue.error();

auto prepared = prepare(*context, RmsNorm{.epsilon = 1.0e-5F},
                        input_desc, weight_desc, input_desc);
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
packed SiLU GatedActivation、零尺寸 MatMul 和其他未支持形状也走 native，不会在 submit 时重新选择。在默认串行
executor 下，`M < 16` 的 decode MatMul 同样选择结构更轻的 HYLE-native kernel；传入真实并行 executor
时，oneDNN 可以把 N 维拆成互不重叠的连续 tile。

```cpp
#include <hyle/compute/providers/onednn.h>

auto context = Context::create(Device::cpu(), onednn::provider());
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

默认 oneDNN executor 是调用线程串行执行；HYLE 不偷偷创建线程池。已有同步线程池的程序可通过
`onednn::Options::executor` 传入借用的 `ParallelExecutorRef`。其 `parallel_for` 必须在返回前完成全部
任务、不得失败或抛异常，并在嵌套调用时保证进展；executor backing 必须活过由 provider 创建的 Context
及其 retained artifacts。要多个 provider 时用 `ContextOptions::providers.add()` 明确顺序；列表最多四项，
`add()` 返回 `false` 表示无效或已满，不能忽略。

并行分片由 HYLE 在 prepare 冷路径决定，不是 oneDNN 的隐藏线程池：正常 tile 宽度至少为 64 columns，
任务数不超过 executor concurrency，并且至多生成 base/base+1 两种预编译 kernel。每个任务使用独占、
64-byte-aligned scratch slice；调用方只需按 `workspace_requirement()` 分配一块总 workspace，不应自行
推导 tile 或依赖该内部阈值。

该选项默认 `OFF`，所以普通 Compute 配置不会下载或编译 oneDNN。构建时
`HYLE_COMPUTE_ONEDNN=ON` 在受支持的 x86-64 目标强制启用，在 AArch64 等不支持目标会明确配置失败；
`AUTO` 只适合本地探索并在不支持目标关闭能力。可复现的 x86-64 oneDNN 制品应显式选择 `ON`，不需要
它的制品保持 `OFF`。`PreparedOperation::provider()`、`used_fallback()` 与
`workspace_requirement()` 可解释实际路线。oneDNN 类型不会泄漏到 Tensor、Operation 或 Graph API。

## 所有权、失败和限制

- `Device` 是廉价值；`Context`、`Queue`、`PreparedOperation`、`OperationInvocation` 与 `Completion` 是
  显式生命周期对象。Context metadata allocator 必须活过从它保留的 handles，以及由该 Context 编译出的
  Graph Executable/Invocation。
- `PreparedOperation` 可复制并跨线程共享；`OperationInvocation` 不可复制，也不得被两个并发 submit 共用。
  Context 关闭后 retained artifact 仍可创建 invocation，但任何 submit 都会失败。
- Tensor views 不拥有 payload。Mem Buffer/resource 以及 Queue 必须活到 Completion 终态。
- `StorageFormatId` 是跨 source/static/shared 稳定的数值 identity，不是 Native FFI 承诺。自定义 provider
  应以自己的 typed factory 统一发布 identity；同一 identity 不得在升级后改变物理解释。
- admission 前的参数、shape、alias、context/queue mismatch 由外层 `Result` 报告；接纳后的 mapping、coherence
  等失败由 Completion terminal `Status` 报告，两层都要检查。
- RMSNorm 允许 input/output 精确同址；所有 operation 拒绝不安全 partial overlap 和第二输入/output
  overlap。CPU route 要求 host-accessible storage，并如实调用 non-coherent resource 的 invalidate/flush。
- 不同 Queue 可并发；`Context::close()`/析构不得与另一 handle 的 submit 并发。
- HYLE-native route 不隐藏 staging、copy、线程切换、packing 或 JIT。oneDNN BRGEMM handle 只在 prepare
  冷路径 finalize/generate；每个 Invocation 映射调用方 workspace，steady submit 不创建 descriptor、
  stream、memory handle、JIT artifact，不 packing/copy，也不执行 route search。当前闭包的六个推理形状
  已证明 warm `submit()+wait()` 无 C++ 动态分配；适用性能与三交付证据已经由 registry 所指 terminal
  record 签发。
- oneDNN ukernel C API 在锁定版本中仍标记 experimental。它完全私有于 provider，并由精确版本断言保护；
  用户不应直接依赖其 header 或 ABI。升级 oneDNN 必须重做源码、符号、分配和正确性审计。

source、headers + static、headers + shared 都选择 `hyle::compute`。三者是同一 C++23 source contract，不是
Native FFI，也不承诺跨工具链 C++ ABI。
