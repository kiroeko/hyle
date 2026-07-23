# error-03：error 模块实际使用导向重收口

> 本文是 Jira 式任务记录。稳定设计见
> [`../architecture/error-design.md`](../architecture/error-design.md)，用户写法见
> [`../guide/usage-guide.md`](../guide/usage-guide.md) 与
> [`../guide/c-error.md`](../guide/c-error.md)，性能数字只见
> [`../performance/benchmarks-error.md`](../performance/benchmarks-error.md)。

## 目标

把最早按 clay 和测试形状形成的 error 模块，重构成基础加速库真实调用需要的失败值：
8B `Status`、`std::expected` Result、显式语义 context、完整 C owning handle，以及
OS/CUDA/Vulkan 叶子诊断。未发布，不保留未经审查的旧 ABI。

## 本轮实施

- Status 所有自身操作 `noexcept`；非法枚举归一为 `Unknown`；OOM 保留 code，context
  OOM 保留原消息。
- 新增一次分配、扁平文本的 `WithContext`；OK/空 context 快路径在消费方头内联。
- 传播宏收成 `HYLE_RETURN_IF_ERROR` / `HYLE_ASSIGN_OR_RETURN`，并迁入 compute/graph。
- C ABI 保留 Status 的 create/clone/move/free/query/format/context 全能力；删除无人使用的
  error handler、record、soft assert 和第二套日志 C API。
- CUDA、CUDA driver、NVRTC、Vulkan、Thread OS 边界保留 operation、原生名称、数值与
  可得的供应商说明；compute 语义边界增加 context。
- 当前 architecture、guide、performance 分别承接设计、recipe 和数字；error-01/02
  只保留为历史任务。

## 验收状态（2026-07-23）

- Windows MSVC Release：全新构建目录中的 error C++/纯 C、platform C++/纯 C、
  compute、5 个 Vulkan 变体与 graph 共 11 项测试通过；当前接口 benchmark n=15 已完成。
- Windows ASan error C++/纯 C 2 项通过；C ABI 14 个 Status 符号（含 context）及 C++
  context 符号已核对，旧 handler/report 符号不存在；diff 静态检查通过。
- WSL Linux GCC 14 已通过当前 error/platform C++/纯 C 4 项（含 POSIX translator）；
  Linux 当前 benchmark、Linux Clang 与 Apple M4 AppleClang 尚待远程复证。因此任务状态是
  **Windows 与 Linux GCC 实现完成、三平台性能冻结未完成**，不得在 architecture 或
  performance 写成全平台最终收口。

## 完成条件

Windows/Linux/macOS 当前 commit 构建与测试通过；三平台核心基准无未解释回退；C ABI
导出面与纯 C smoke 一致；届时只把本任务状态改为完成，不再复制 architecture 或
performance 的正文。
