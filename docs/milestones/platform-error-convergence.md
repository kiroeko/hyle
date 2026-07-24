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
- 2026-07-23 的 Apple M4 AppleClang 21 `hyle_platform_test` 148/148 通过，覆盖
  DynLib/MappedFile 的共享错误集成；A 阶段资源路径已完成三平台复证。

## B 阶段：VM + NUMA（2026-07-23，三平台已完成）

- VM 页原语返回 8-byte `Status`，reserve/`VmRegion` 工厂返回 `Result`；成功态零分配，
  原生诊断只在失败路径构造。删除 `hyle_vm_last_error` 与 setter。
- NUMA 返回 `NumaPlacement`（Applied/NotNeeded/Unavailable）：preferred 的可用降级不再
  冒充“已放置”，hard bind/interleave 无法满足时返回结构化错误。
- C ABI 全面采用 `hyle_status` + out，`VmRegion::adopt` 只在成功后转移所有权。
- `HugeRegion`、`FiberStack`、stack pool 与 sched 的 detached/task 路径传播共享错误；
  当时 mem C 仍保留分配语义的小枚举；error adoption 后续已把 mem 冷工厂迁到
  `hyle_status + out` 并删除该私有 TLS 通道，热分配 NULL 合同不变。
- Windows MSVC Release 的 platform/mem/sched C/C++ 与 mem header/surface 共 8 项通过；
  WSL GCC 14 与 Apple M4 AppleClang 21 的同链路各 8 项通过。macOS 非登录 SSH
  下的 header checker 现用 `xcrun` 注入 SDK sysroot，不再把工具链环境误报为
  `<cstddef>` 头缺陷。
- Windows MSVC ASan 的 platform/mem 全绿；sched 受影响的新 `Fiber::create` 与 VM stack
  路径通过。sched 全套仅既有 `RetiredThreadHandlesStayBoundedAcrossBursts` 首轮瞬时观测到
  2（阈值 1），隔离复跑通过，不冒充单轮 138/138。WSL UBSan 的 platform 2/2、mem
  4/4、sched 2/2 全绿。
- 当前形状的 Windows 定向 smoke：VM reserve/commit/decommit/release 为 1167 ns，raw
  为 897 ns；protect pair 为 460/439 ns；FiberStackPool 的 32-fiber batch p50 为
  7.3 us，直接 guarded VM 为 155 us。它们只证明接口重构没有离开既有统计带，不替代
  三平台正式性能证书。
- WSL TSan：platform 148/148、受影响 mem 27/27、sched 138/138 通过；WSL 的
  随机高地址映射会让 GCC TSan 启动前报 `unexpected memory mapping`，sched 正式运行
  用 `setarch x86_64 -R` 固定地址空间。它是 harness 环境要求，不是被测 race。

**B 阶段完成。**后续只有新的消费者需求、跨平台回归或正式性能证书失效才重开，
不再因 benchmark 形状自行扩张 VM/mem API。

## 完成条件

- Windows/Linux/macOS 当前接口的 C/C++ tests 通过，ASan/TSan 按涉及路径复证。
- 所有公开可恢复资源构造均有结构化 code + message，不靠跨子系统 TLS 猜失败。
- C ABI 可覆盖 C++ 可恢复失败，不引入异常，不让 error 反向依赖 mem。
- architecture/guide 只写当前真相，旧 milestone 只保留历史并链接追记。
