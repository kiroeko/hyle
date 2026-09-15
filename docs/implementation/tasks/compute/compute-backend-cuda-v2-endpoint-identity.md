# Compute CUDA backend v2：完成端点不能把 Context 当 Queue

## Closure card / SCOPED

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

## 待审范围

确认 `CompletionEndpoint` 的 exact ordering identity 与 compatibility 域是两层，不能因 endpoint 可导入
就省略同步；Queue move/destruction 与 Graph 的借用范围沿用既有合同。其余 v1 功能不重做。
