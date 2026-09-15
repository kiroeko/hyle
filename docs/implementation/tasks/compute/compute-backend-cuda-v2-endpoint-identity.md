# Compute CUDA backend v2：完成端点不能把 Context 当 Queue

## Closure card / AUDITED

- owner：`compute_backend_cuda`；baseline `a4dee153`；前轮为
  [CUDA v1](compute-backend-cuda-v1-async-runtime.md)。
- trigger：Graph 的真实 pinned → device kernel → pinned 双 lane consumer 在 `a4dee153` 失败。
  `cuda_backend.cc::completion_endpoint_impl` 返回 Context 指针；Graph 按既定 exact endpoint equality
  判为 ordered，于是不同 CUDA stream 的数据依赖没有得到 event wait。这是 correctness trigger，不是性能点值。
- 主旅程：同 Context 两个 CUDA Queue 在 Graph 中上传/计算/下载；两个独立 Invocation 重复 commands/
  capture。正交旅程为 direct Queue event import 和 borrowed native stream，不改变模型或算子语义。
- scope：只纠正私有 endpoint 身份及同 Context 的可导入判断；复用稳定 QueueState、现有 event wait、
  Queue/Completion 保活。公开 C++ API、layout、CMake、版本/许可证和 CUDA 内核不变。
- non-goals：Q4/provider 目录、跨 Context 互通、全设备同步、Graph workaround、热态 probe 或 endpoint registry。
- 性能：P0；endpoint/query 只在冷准备，O(1)、零新 allocation/lock/syscall/metadata；跨 stream 的真实
  event wait 是正确性必要成本，不能沿用错误省略同步的路线作性能基线。不为此冷修复单开 P1。
- 预声明验证：同 Queue 重复 query 相等、同 Context 不同 Queue 不等但可互相导入、另一 Context/invalid
  endpoint 拒绝、Queue move 身份稳定；现有真实 delayed producer/event/import、CUDA runtime/late error/
  逐头，Graph CUDA 定向回放，source/static/shared 的新双 lane kernel consumer；docs/closure/diff。
  CUDA 平台共享实现的 Linux 编译/运行与适用 sanitizer 按完整候选实际环境取证；Apple 不存在 CUDA，
  且本轮未改 AUTO/OFF/CMake，旧平台不存在路线不因私有函数纠正重测。
- 预算：一个表示/实现，一次纠正，候选后独立复核；无新指针 owner、逃逸生命周期或额外字段。
  如需改共用 endpoint 布局/生命周期或扩大 CUDA 上下文互通，停止并重新审计该 owner。

## 有限审计与修复集

`CompletionEndpoint` 的 exact ordering identity 与 compatibility 域是两层：公共内部头明确身份借到
owning Queue 存活期间；Graph equality 表达同一有序端点，可导入只是能建立依赖。CUDA 的
[`cudaStreamWaitEvent`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
才建立另一个 stream 的完成依赖。现有 `QueueSequence::dependency_route` 已按真实 Queue 区分，只有
backend endpoint 错用了 Context。这不是 Graph 应猜 CUDA 类型来补救的问题。

| ID | 允许修复 | 验收 |
|---|---|---|
| CU20 | endpoint identity 改为已存在、稳定的私有 QueueState 指针；先验证 authority/non-null，再通过该 QueueState 的 Context 判断可导入 | 不同 stream 不相等；同 Context 仍可 event import；move 稳定；invalid/其它 Context 拒绝 |
| CU21 | 实际 direct/Graph/三交付反例和指南说明 | 真实 copy/compute/copy 报告 device waits，无 host wait；commands/capture 两个独立 Invocation 数值正确 |

两处私有函数是完整产品 FIX；不新增字段、owner、引用计数或虚构通用 registry。相同 native stream 的
不同 borrowed Queue wrapper 可保守插入一次合法 event wait，不根据 stream 指针猜额外跨 Context 互通。
warm enqueue/copy/event/Completion 实现不改；Graph 自然消费正确身份，不添加 workaround。

能力族是“同执行域内跨队列依赖”；当前为 `PARTIAL/FIX`。正常 C++ 调用、错误与 async lifetime 沿用
v1，所有其它 v1 公开入口 `KEEP`。直接/间接依赖、CUDA Toolkit 13.3 与许可证、FFI=N/A、三个交付形状
均不变；复用 Compute endpoint/QueueSequence 与既有 backend wait。无新模块、开关或其它依赖重开。

`1def523a` 的最小反例已在 RTX 4070 上实际失败：不同 Queue 的 16-byte endpoint 完全相同；其余
compatibility/move/foreign/invalid 检查通过。对应安装源码 consumer 已在 `a4dee153` 因异步依赖报告
失败。两者与源码指向同一根因，接下来只实施 CU20，再按本卡回归；未达到 Candidate/关闭。
