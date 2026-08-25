# platform-error-convergence：platform 共享错误收敛

> Jira 式任务记录。稳定原则见
> [`../architecture/c-abi-design.md`](../../../architecture/native-ffi-policy.md)，当前 C 写法见
> [`../guide/c-platform.md`](../../../architecture/native-ffi-policy.md)。

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

## C 阶段：Parker + 打开文件资源（2026-07-24，Windows 完成）

- C++ 文件资源从 `OwnedFd` 收束为跨平台名 `FileHandle`；`open_file` /
  `create_temp_file` 返回 `Result<FileHandle>`，read/write 返回 `Result<size_t>`，显式
  `close()` 返回 `Status`。析构仍兜底关闭，但需要观察延迟写回错误的用户应显式 close。
- POSIX read/write 处理 EINTR 和单次 I/O 上限；Windows/POSIX 失败保留 operation、原生
  code/name/message，并映射到共享 Status code。
- C ABI 的 Parker create 与打开文件 create/open/read/write/close 全部改为
  `hyle_status + out`；native file 值收束为 `hyle_file_handle`，不再把 Windows HANDLE
  命名为 fd；TLS last-error 只暂留给下一阶段迁移的 path/env 标量面。
- 增加 C++ `vm_allocate_rw` / `VmRegion::allocate_rw`，C `hyle_vm_alloc_rw` 也改走一次
  OS mapping，而非为了复用 staged API 付 reserve+commit 两次系统调用。
- Windows MSVC Release：platform C/C++ 与下游 mem C/C++、C surface/header 检查通过；
  MSVC ASan 下 platform/mem C/C++ 四项通过。VM 5 进程配对中位数：raw OS
  879.8 ns，HYLE 单步 885.2 ns（0.987x，统计平）；旧 staged 形状 1130.7 ns。

下一增量迁移 path/env 标量失败面，并在 Linux/macOS 镜像回归后关闭 C 阶段。

## D 阶段：filesystem path 操作（2026-07-24，Windows 完成）

- C++ stat/size/permissions/readlink 返回 `Result<T>`，path 变更返回 `Status`；
  `exists/is_directory/is_symlink` 将 NotFound 表达为成功的 false，但权限、编码和 I/O
  失败不会再被吞成 false。空文件 size=0 与失败、合法 mode=0000 与失败已可区分。
- 目录 visitor 提前停止返回成功的 `Result(false)`，打开/枚举/关闭失败才返回 error；
  recursive remove 传播首个结构化失败，并继续保持“不跟随 symlink/junction”的数据安全线。
- 对应 C ABI 全部改成 `hyle_status + out`，out 进入时清零；fs path 面不再读取 TLS。
  `hyle_platform_last_error` 只剩 cwd/temp/home/env buffer-fill、chdir/env 写这一组。
- Windows MSVC Release 的 platform/mem C/C++、mem C surface/header 六项通过；MSVC
  ASan 的 platform/mem C/C++ 四项通过。WSL/GCC 以 C++23、`-Wall -Wextra -Werror`
  编译当前 POSIX `fs.cc` 与 C bridge 通过（Linux 行为测试仍以真机复跑为准）。当前目录
  遍历 n=15：HYLE 40.23 us、std::filesystem 39.76 us，配对中位比 0.981，sign/spread
  均为统计平。

下一增量完成 cwd/temp/home/env，随后删除 `hyle_platform_last_error`。

## E 阶段：cwd / special directories / environment（2026-07-24，Windows 完成）

- C++ cwd/temp/home/env buffer-fill 返回 `Result<size_t>`，chdir/env 写返回 `Status`；环境
  变量未设置为 `NotFound`，与“已设置为空”的成功 size=0 明确分离。
- C ABI 使用 `hyle_status + out_size`，环境入口收束为对称的
  `hyle_env_get/set/unset`；删除 `HYLE_ENV_NOT_SET` 和模块私有
  `hyle_platform_last_error`。platform 的公开可恢复失败至此全部走共享 error。
- Windows 环境读写统一 UTF-8↔UTF-16；读侧对进程全局环境值的长度竞态做有界重试。
  POSIX `getcwd` / `getpwuid_r` 改为几何增长缓冲，不再把超过 4096/1024 字节吞成空串。
- 常见成功路使用 512B/1024B 栈缓冲；超长输入才几何增长，复杂度 O(输出长度)、额外
  空间 O(输出长度)，已经是必须复制 UTF-8 buffer-fill 的渐近下界，不为冷控制面制造
  无意义 microbenchmark。

Windows MSVC Release 的 platform/mem C/C++、mem C surface/header 六项通过，MSVC
ASan 的 platform/mem C/C++ 四项通过；WSL/GCC 14 Release 从当前源码构建并通过
platform C/C++ 2/2。Windows 是本机主证，WSL 只作为 POSIX 行为回归，不外推性能。

## 完成条件

- Windows/Linux/macOS 当前接口的 C/C++ tests 通过，ASan/TSan 按涉及路径复证。
- 所有公开可恢复资源构造均有结构化 code + message，不靠跨子系统 TLS 猜失败。
- C ABI 可覆盖 C++ 可恢复失败，不引入异常，不让 error 反向依赖 mem。
- architecture/guide 只写当前真相，旧 milestone 只保留历史并链接追记。
