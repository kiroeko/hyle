# Platform v6：同步文件控制与进程启动策略

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只把已经登记、且经 owner 明确准入的通用 Platform 缺口变成有限产品工作。
> 进入 `SCOPED`、完成 `AUDITED` 并冻结 FIX 集之前，不授权修改产品源码。

## Reopen evidence（`REOPENED`）

- baseline：`cba7392499c1eca2f0e961e0e2a45a89a2d6d88b`；
- trigger ID：`PLATFORM-COMMON-CONTROL-SUPERSET-2026-09-05`；
- 既有全局能力账本已将 `PLAT-A-18` 进程 priority/QoS 记为 `PARTIAL`，并以
  llama.cpp 的 startup `set_process_priority` 为直接 consumer；`PLAT-A-26` 已将同步文件
  lock/resize/preallocate 记为 `MISSING`；
- owner 决策：这些是基础库常见能力，应进入 Platform，而不是继续等未来 consumer；HYLE 尚未正式发布，
  不保留旧 API 兼容层；
- 当前实现的 `FileHandle` 已拥有 open/read/write/seek/sync/close，却不能在同一已打开对象上查询大小、
  调整 EOF、预留空间或做进程间建议锁。退回 path 查询会重新解析名字并暴露 rename/replace 竞态；
- `platform`、`platform_fs` 当前 `CLOSED` 记录没有新标准 `freeze_confirmation`。本轮只重开受影响的
  base/fs factor；`platform_thread` 保持只读，最终随聚合产品一起复核冻结，不借机修改其产品代码。

### 证据来源与用途

- 真实消费：当前 llama.cpp 的进程启动 priority helper；HYLE 的模型/缓存同步文件控制旅程；
- mature peers：Folly `File` 的 shared/exclusive whole-file lock，Boost/std filesystem 的 resize 能力；
- OS 合同：Windows process QoS、allocation/EOF 与 `LockFileEx`，POSIX/Linux
  `setpriority`、`ftruncate`、`posix_fallocate`/`fallocate` 与 `flock`，macOS `F_PREALLOCATE`；
- 书籍发现：Kerrisk《The Linux Programming Interface》的 file I/O、priority/scheduling、file locking，
  Stevens/Rago《Advanced Programming in the UNIX Environment》的 file I/O/process control/record locking，
  以及《Windows Internals》的 process/thread priority 与 power policy。书籍帮助发现能力和陷阱，
  具体行为仍由支持平台的官方 ABI 文档与测试裁决。

### 初步能力边界

| family | natural owner / shape | 初步 disposition |
|---|---|---|
| 已打开文件大小 | `Result<uint64_t> FileHandle::size()` | `MUST`；避免 path TOCTOU，并闭合 handle 控制旅程 |
| 精确逻辑大小 | `Status FileHandle::resize(uint64_t)` | `MUST`；只承诺 EOF，不假装物理分配 |
| 空间预留 | `Status FileHandle::preallocate(uint64_t)` | `MUST`；至少覆盖 `[0,size)`、不缩小较大文件 |
| 进程间整文件锁 | shared/exclusive `lock`、`try_lock`、`unlock` | `MUST`；建议锁、随 handle close 释放；不冒充线程 mutex |
| 进程启动 priority/QoS | 独立 `ProcessPriority` 与显式 setter | `MUST`；无 realtime；不自动改变应用策略 |
| async/direct/positional/vectored/watch/mapping | IO / Mem owner | `ROUTED`；本轮禁止吸收 |
| detach、range lock、lease、mandatory lock | 生命周期或平台特化 | `REJECT/DEFER`；没有进入本轮自然公共面 |

## 初步性能与比较合同

- tier：`P0 / closure-correctness-v1`。五项都是显式控制面 OS 操作，不在每次 read/write、thread task、
  CPU fact query 或其他常态热路径增加分支、对象字段、allocation 或 syscall；
- `size/resize/lock/unlock/process-priority` 应为固定次数 OS control，复杂度 O(1)；`preallocate` 的物理工作量
  由 filesystem 和申请范围决定，不用延迟点值冒充算法优劣；阻塞 `lock` 的等待时间由竞争者决定；
- 同合同硬门必须逐项对齐 logical EOF、physical allocation、blocking、shared/exclusive、advisory、
  failure/permission 和 cleanup 语义。任何 material 差异都标 `NOT_COMPARABLE`；不得把 path resize 与
  already-open handle、sparse resize 与 preallocate、try-lock 与 blocking lock 做“苹果对橘子”排名；
- 因而本轮工程性能主要由结构/codegen 与 syscall-count 证明；若审计发现自然同合同 peer 且存在 HYLE
  自造额外成本，才升级为有限 P1，不为生成 benchmark 行扩张 API。

## 下一状态入口

只有在 closure card 冻结真实 consumers、自然 API/失败语义、跨平台差异、依赖/交付、有限 FIX 和验证矩阵
后进入 `SCOPED`；只有完成 implementation/peer/文档/构建红队并冻结 FIX 后进入 `AUDITED`。

## Closure card（`SCOPED`）

- component / product / round / kind：`platform + platform_fs` / Platform /
  `platform-v6-file-control-and-process-policy` / `REOPEN`；
- baseline / trigger：`cba7392499c1eca2f0e961e0e2a45a89a2d6d88b` /
  `PLATFORM-COMMON-CONTROL-SUPERSET-2026-09-05`；
- consumers：推理程序在创建 workers 前显式设置 process policy；模型、缓存、索引与数据库控制面通过一个
  已打开 owner 查询/调整文件、预留空间并协调多个 cooperating processes；
- API 冻结候选：
  1. `enum class ProcessPriority : int8_t { kLowest, kLow, kNormal, kHigh, kHighest }` 与
     `Status set_process_priority(ProcessPriority)`，独立于 thread enum，避免 base factor 反向依赖
     `platform_thread`；不提供 realtime；
  2. `FileHandle::size() const`、`resize(size)`、`preallocate(size)`；size 为逻辑字节数，resize 只改变
     EOF，preallocate 保证文件至少达到 `size` 且请求 `[0,size)` storage、绝不缩小较大文件；
  3. `FileLockMode::{kShared,kExclusive}` 与 `lock(mode)`、`try_lock(mode)`、`unlock()`；锁随 handle
     close/析构释放，mode conversion 不承诺原子；
- 跨平台合同：
  - Windows process ladder 同时设置 priority class 与 execution-speed QoS；normal 恢复 system-managed
    QoS；两个 OS controls 不具备事务原子性；
  - Linux/macOS 使用 `setpriority(PRIO_PROCESS, 0, nice)`；Linux 的该调用作用于调用线程，故完整
    process-startup 效果要求在创建 workers 前调用；已有 workers 继续逐线程使用 Platform Thread；
  - POSIX lock 使用 `flock` 的 open-file-description/last-close 语义并重试阻塞调用的 `EINTR`；Windows
    使用从 offset 0 覆盖可表达全范围的 `LockFileEx`。前者是 advisory，后者会被内核 I/O 强制；公共
    承诺只有 cooperating processes 的互斥/共享协调；
  - POSIX resize 使用 `ftruncate`；Linux preallocate 使用 `posix_fallocate`；macOS 先
    `F_PREALLOCATE`（contiguous 失败后 all）再按需 `ftruncate`；Windows 分别使用
    `FileEndOfFileInfo` 与 `FileAllocationInfo` 后按需调整 EOF；超过 signed native offset 的值返回
    `OutOfRange`；preallocate 的 allocation 成功、EOF 更新失败可能留下不可见的已分配空间，明确为
    非事务操作；
- ownership / dependency：不改变一字 `FileHandle` layout，不保存 lock state；`platform` 仍公开依赖
  `core/error`，`platform_fs` 仍公开依赖 `core/error/time` 且私有依赖 `platform`；实现统一走已冻结
  Error 的 `status_from_errno/status_from_win32`；不引入 Container/Mem/第三方产品依赖；
- public delivery：C++23 source、static、shared 三种；Native FFI/C API 继续 `N/A`；新增头必须自包含、
  安装，并由 Platform package sample 消费；
- performance：`P0`。保持所有旧热路径字节/控制流不变；每个新方法只在显式调用时落最少 native
  control。禁止对不同 EOF/allocation/locking/QoS 语义做苹果对橘子计时；
- budget：一个 base source/header、现有 fs source/header、两套 factor tests、Platform sample、
  architecture/user/capability evidence、组件/三交付验证；不扩建统一性能仪器；
- stop：出现无法诚实统一的 supported-OS 语义、需要常态对象 metadata/新依赖、或必须改变 frozen
  Thread/Mem/IO 产品合同，退回 `AUDITED` 或 `BLOCKED`，不得用模糊文档掩盖。

### 预声明验证矩阵

| gate | 验证 |
|---|---|
| correctness | Windows Release + Werror；WSL GCC Release/ASan/UBSan；macOS AppleClang Release；closed handle、invalid enum/mode、offset overflow、grow/shrink/no-shrink、shared/exclusive/try/close-release |
| architecture | public-header closure；component graph/exact factors；无新 third-party；Platform/FS 只使用 Core/Error/Time 合法边 |
| delivery | Platform product source embedding、relocated static SDK、relocated shared SDK；安装 sample 编译/运行；C++ symbols only |
| consumers | Mem mapping、Sched、IO 只读构建回放；不因存在相邻同类 syscall 就非法重开其产品 |
| usability/docs | architecture 与 user guide 给出选择表、失败/权限、startup timing、lock/resize/preallocate 陷阱；过时 negative-space 清理 |
| performance | P0 structural/codegen：`sizeof(FileHandle)` 不变、旧 hot path 无 diff、新入口固定 native calls；任何 peer timing 先过完整语义等价表 |
| red team | path replacement、sparse≠preallocation、lock advisory/mandatory 差异、same-process/duplicate handle、close release、partial side effects、permissions/network filesystem、Windows two-control partial failure |

本卡已冻结边界；下一步只允许只读审计并形成有限 `FIX` 集，不能先写产品代码。
