# platform-error-convergence：platform 共享错误收敛

> Jira 式任务记录。稳定原则见
> [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)，当前 C 写法见
> [`../guide/c-platform.md`](../guide/c-platform.md)。

## 目标

让 platform 的可恢复资源失败复用 error 模块的 `Status` / `Result`，删除模块私有
last-error 分类和“失败指针 + 另查 TLS”的组合；未发布，不保留未经审查的旧 ABI。

## A 阶段：DynLib + MappedFile（2026-07-23）

- C++ `DynLib::{open,get,current,symbol,symbol_as}` 返回 `Result`；loader 的操作名、
  原生错误名、数值和系统说明进入 Status。
- C++ `MappedFile::open` 返回 `Result<MappedFile>`，`flush` 返回 `Status`；路径、模式、
  范围和 OS 失败不再坍缩为 invalid object / bool。
- C ABI 使用 `hyle_status` + out 参数，并明确 Status 与成功句柄的所有权；删除
  `hyle_dynlib_last_error`，MappedFile 不再借用 VM 的 TLS 通道。
- sched 显式声明对 error 的直接依赖，不依靠 mem/platform 的传递链接。

### A 阶段验收

- Windows MSVC Release 全库构建完成、CTest 51/51；error/platform/sched 的 C/C++
  六项在 MSVC ASan 下通过。
- WSL Linux GCC 14 的同六项通过；error/platform 四项在 UBSan 下通过，覆盖
  `dlopen`/`dlsym`、`mmap`/`msync` 与 GNU `strerror_r` 分支。
- macOS AppleClang 当前提交尚待远程复证，因此 A 阶段实现完成，但不把 platform
  或 error 写成三平台最终冻结。

## B 阶段：VM + NUMA（待执行）

VM/NUMA 仍使用 `hyle_vm_last_error`。它们直接穿过 `FiberStack`、`HugeRegion`、mem C
ABI 与若干热内存形状，须独立审计失败分层、底层依赖方向和 hot path，再决定哪些
资源构造返回 Result、哪些页原语保留零分配紧凑结果。A 阶段不宣称 platform 已全收口。

## 完成条件

- Windows/Linux/macOS 当前接口的 C/C++ tests 通过，ASan/TSan 按涉及路径复证。
- 所有公开可恢复资源构造均有结构化 code + message，不靠跨子系统 TLS 猜失败。
- C ABI 可覆盖 C++ 可恢复失败，不引入异常，不让 error 反向依赖 mem。
- architecture/guide 只写当前真相，旧 milestone 只保留历史并链接追记。
