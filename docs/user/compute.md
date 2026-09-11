# HYLE Compute：何时直接执行一个计算

> 本页描述当前 CPU C++ surface；施工资格只以
> [`module-status.json`](../implementation/module-status.json) 为准。CUDA、Metal、Vulkan、HIP 与外部
> providers 仍是后续独立纵切，枚举词汇不代表实现可用。

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
| 稳定重复执行 | `prepare -> PreparedOperation::submit -> Completion` | prepare 固定 shape/provider route；submit 不重新推 shape/路由 |

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

## prepare 与 submit

`infer_output_description()` 是 direct Compute 与 Graph 共用的唯一 shape/layout 规则。`prepare()` 再核对
调用方给出的 output desc，并把 provider/route 选择移出热路径。对同一描述反复执行时复用
`PreparedOperation`；当前故意没有把 prepare 隐藏在一次性 convenience 中。

```cpp
auto context = Context::create(Device::cpu());
if (!context) return context.error();
auto queue = context->create_queue();
if (!queue) return queue.error();

auto prepared = prepare(*context, RmsNorm{.epsilon = 1.0e-5F},
                        input_desc, weight_desc, input_desc);
if (!prepared) return prepared.error();
auto done = prepared->submit(*queue, input_view, weight_view, output_view);
if (!done) return done.error();       // admission failure
return done->wait();                  // terminal failure
```

## 所有权、失败和限制

- `Device` 是廉价值；`Context`、`Queue`、`PreparedOperation` 与 `Completion` 是显式生命周期对象。
  Context metadata allocator 必须活过从它保留的 handles，以及由该 Context 编译出的 Graph
  Executable/Invocation。
- Tensor views 不拥有 payload。Mem Buffer/resource 以及 Queue 必须活到 Completion 终态。
- admission 前的参数、shape、alias、context/queue mismatch 由外层 `Result` 报告；接纳后的 mapping、coherence
  等失败由 Completion terminal `Status` 报告，两层都要检查。
- RMSNorm 允许 input/output 精确同址；所有 operation 拒绝不安全 partial overlap 和第二输入/output
  overlap。CPU route 要求 host-accessible storage，并如实调用 non-coherent resource 的 invalidate/flush。
- 不同 Queue 可并发；`Context::close()`/析构不得与另一 handle 的 submit 并发。
- 当前 route 不隐藏 staging、copy、线程切换、packing、JIT 或 provider fallback。

source、headers + static、headers + shared 都选择 `hyle::compute`。三者是同一 C++23 source contract，不是
Native FFI，也不承诺跨工具链 C++ ABI。
