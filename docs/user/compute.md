# HYLE Compute：何时直接执行一个计算

> 本页描述当前 CPU C++ surface；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。HYLE-native 始终可用；选择
> `HYLE_COMPUTE_ONEDNN=ON` 时可以在 x86-64 构建显式 oneDNN MatMul provider。该可选 factor 当前为
> `AUDITED`：BRGEMM ukernel 替换候选已通过定向正确性与暖执行零分配检查，但 source/static/shared 和
> 正式 P1 尚未全部签发，因此还不是 `CLOSED` 能力。Apple Silicon 等 AArch64 目标会关闭该 factor 并
> 使用 HYLE-native；CUDA、Metal、Vulkan、HIP 与其他 providers 仍是后续独立纵切。

当程序已经知道要执行哪个 operation、在哪个 device 上执行，并希望复用固定 shape/layout 的路线时，
使用 Compute。跨节点拓扑、liveness 和可复用图执行物使用 [Graph](graph.md)；模型、KV cache、batching、
sampling 和 serving 留在推理引擎。

## 当前能力

| 任务 | 公开形状 | 当前合同 |
|---|---|---|
| 设备发现 | `enumerate_devices` / `query_device` | 当前只返回 CPU；短输出 span 不静默截断 |
| 执行域 | `Device -> Context -> Queue` | CPU Queue 在调用线程同步、按序执行；没有隐藏 worker/default device |
| tensor schema | `TensorDesc::dense/strided/permuted/transposed` | rank <= 8；stride 以 element 为单位；view 借用 Mem storage |
| normalization | `RmsNorm` | dense f32 `[..., hidden]`；可选 dense f32 weight `[hidden]`；逐 row |
| matrix multiply | `MatMul` | f32 rank-2 `A[M,K] * B[K,N]`；B 可 row-major 或物理 `[N,K]` 的转置 view |
| gated activation | `SiluAndMul` | dense f32 `[..., 2*d] -> [..., d]`，前半 gate、后半 value |
| 自定义 CPU 工作 | `submit_host(queue, callable, args...)` | `noexcept` callable，返回 `void` 或 `Status`，返回前完成 |
| provider 选择 | `Context::create(device, Provider)` / `ContextOptions::providers` | 显式且有序；只有 `Unimplemented` 才尝试下一项，HYLE-native 是最终 fallback |
| 稳定重复执行 | `prepare -> PreparedOperation -> OperationInvocation::submit` | prepared artifact 可共享；每个并发 lane 独占 invocation/workspace；submit 不重新推 shape/路由 |

当前 operator catalog 有意只签发本轮真实 transformer 片段需要的三项，并非长期超集。f16/bf16、量化、
batched MatMul、attention、RoPE、softmax、embedding、repack 与 provider fusion 必须在各自有限卡中加入，
不能把 enum 预留写成已实现能力。

## MatMul 权重形状

不要增加 provider-specific `transpose_weight` 标志。若模型存储权重为连续 `[N,K]`，创建该 dense desc 后
调用 `transposed()`，得到逻辑 `[K,N]` 和对应 stride；数据不移动。Compute 会在 prepare 冷路径选择相应
route。普通连续 `[K,N]` 也直接受支持。

```cpp
auto stored = TensorDesc::dense(DataType::kFloat32, std::array{n, k});
if (!stored) return stored.error();
auto logical_weight = stored->transposed();
if (!logical_weight) return logical_weight.error();

auto output = infer_output_description(
    Operation(MatMul{}), std::array{activation_desc, *logical_weight});
if (!output) return output.error();
auto prepared = prepare(context, MatMul{}, activation_desc,
                        *logical_weight, *output);
```

## prepare、invocation 与 submit

`infer_output_description()` 是 direct Compute 与 Graph 共用的唯一 shape/layout 规则。`prepare()` 再核对
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

当前 oneDNN provider 只接收 dense row-major f32 rank-2 MatMul。它适合普通 `[M,K] * [K,N]`；模型常见的
物理 `[N,K]`、逻辑转置为 `[K,N]` 的连续-K 权重会诚实回落到 HYLE-native SIMD route。RMSNorm、
SiluAndMul、零尺寸 MatMul 和其他未支持形状也走 native，不会在 submit 时重新选择。在默认串行
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
- admission 前的参数、shape、alias、context/queue mismatch 由外层 `Result` 报告；接纳后的 mapping、coherence
  等失败由 Completion terminal `Status` 报告，两层都要检查。
- RMSNorm 允许 input/output 精确同址；所有 operation 拒绝不安全 partial overlap 和第二输入/output
  overlap。CPU route 要求 host-accessible storage，并如实调用 non-coherent resource 的 invalidate/flush。
- 不同 Queue 可并发；`Context::close()`/析构不得与另一 handle 的 submit 并发。
- HYLE-native route 不隐藏 staging、copy、线程切换、packing 或 JIT。oneDNN BRGEMM handle 只在 prepare
  冷路径 finalize/generate；每个 Invocation 映射调用方 workspace，steady submit 不创建 descriptor、
  stream、memory handle、JIT artifact，不 packing/copy，也不执行 route search。当前候选的六个推理形状
  已证明 warm `submit()+wait()` 无 C++ 动态分配；正式性能与三交付承诺仍以 registry 最终关闭为准。
- oneDNN ukernel C API 在锁定版本中仍标记 experimental。它完全私有于 provider，并由精确版本断言保护；
  用户不应直接依赖其 header 或 ABI。升级 oneDNN 必须重做源码、符号、分配和正确性审计。

source、headers + static、headers + shared 都选择 `hyle::compute`。三者是同一 C++23 source contract，不是
Native FFI，也不承诺跨工具链 C++ ABI。
