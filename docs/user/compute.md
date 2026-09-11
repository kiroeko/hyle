# HYLE Compute：何时直接执行一个计算

> 当前页面描述正在收口的 CPU one-op C++ candidate，不签发完整 Compute 或 GPU 能力。精确施工状态见
> [`module-status.json`](../implementation/module-status.json)。

当程序已经知道要执行哪个操作、在哪个设备执行，并希望重复使用同一 shape 的已准备路线时，使用
Compute。若还需要跨节点拓扑、编译和可复用图执行物，使用 [Graph](graph.md)；模型加载、KV cache、
batching、sampling 和 serving 属推理引擎，不属于本模块。

## 当前能做什么

| 任务 | 公开形状 | 当前限制 |
|---|---|---|
| 发现执行设备 | `enumerate_devices` / `query_device` | 当前只枚举 CPU；枚举返回完整数量，短 buffer 不静默截断 |
| 建立执行域 | `Device -> Context -> Queue` | CPU queue 在调用线程同步、按序执行；没有隐藏 worker/default device |
| 描述和绑定 tensor | `TensorDesc` + `ConstTensorView` / `TensorView` | rank 至多 8；storage 借用 Mem `BufferView`；stride 以 element 为单位 |
| 重复执行一个算子 | `prepare(...) -> PreparedOperation::submit(...)` | 当前正常 route 只有 dense rank-1 f32 RMSNorm，可选 f32 weight |
| 直接工程 kernel | `submit_host(queue, callable, args...)` | CPU、`noexcept`、返回 `void` 或 `Status`，并在返回前完成 |
| 取得终态 | `Completion` | CPU candidate 已经 terminal；仍须检查 `wait()`/`status()` |

`BackendKind` 中的 CUDA、Metal、Vulkan 和 HIP 是稳定词汇，不表示对应实现已经编入产品。只以设备枚举、
构建配置和模块状态判断可用能力。

## 正常路线

```cpp
#include "hyle/compute/operation.h"
#include "hyle/mem/buffer.h"

using namespace hyle::compute;

constexpr std::array<std::int64_t, 1> shape{4096};
auto desc = TensorDesc::dense(DataType::kFloat32, shape);
if (!desc) return desc.error();
auto bytes = desc->required_bytes();
if (!bytes) return bytes.error();

auto input_storage = hyle::mem::Buffer::create(
    hyle::mem::system_resource(), *bytes, 64);
auto output_storage = hyle::mem::Buffer::create(
    hyle::mem::system_resource(), *bytes, 64);
if (!input_storage) return input_storage.error();
if (!output_storage) return output_storage.error();

auto input = ConstTensorView::create(*desc, input_storage->view());
auto output = TensorView::create(*desc, output_storage->view());
auto context = Context::create(Device::cpu());
if (!input) return input.error();
if (!output) return output.error();
if (!context) return context.error();

auto queue = context->create_queue();
auto op = prepare(*context, RmsNorm{.epsilon = 1.0e-5F}, *desc, *desc);
if (!queue) return queue.error();
if (!op) return op.error();

auto completion = op->submit(*queue, *input, *output);
if (!completion) return completion.error();
return completion->wait();
```

`prepare` 是冷路径：它验证 shape/layout 并固定 provider route。对同一描述反复执行时复用
`PreparedOperation`，不要每次重建它。一次性调用 convenience 当前故意不存在，以免把冷路径伪装为
廉价提交。

## 所有权、线程和失败

- `Device` 是廉价值；`Context`、`Queue`、`PreparedOperation` 和 `Completion` 是显式生命周期对象。
  Context 的 metadata allocator 必须活过从它保留的所有 handle。
- Tensor view 不拥有 payload。Mem Buffer/resource 以及 Context 必须活到 Completion 终态；CPU 当前同步
  完成也应按这条合同编写代码，才能迁移到异步 backend。
- 不同 Queue 可以并发提交；同一个 Context 的 `close()`/析构不得与提交并发。Queue 是 move-only，关闭后
  新提交返回 `FailedPrecondition`。
- 参数、shape、range、alias 或 provider 支持问题在 admission 前由 `Result` 返回；工作已接纳后的失败由
  Completion 的 terminal `Status` 返回。两层都要检查。
- RMSNorm 允许 input/output 精确同址；拒绝部分重叠和 weight/output 重叠。当前路线不会隐藏 staging、
  copy、线程切换或 provider fallback。
- non-coherent resource 的 invalidate/flush 仍遵守 [Mem](mem.md) 合同；Compute 不把 native handle 当作
  可解引用 host pointer。

source、headers + static 和 headers + shared 都链接 `hyle::compute`。三者都是 C++23 surface，不是
Native FFI，也不承诺跨工具链 C++ ABI。
