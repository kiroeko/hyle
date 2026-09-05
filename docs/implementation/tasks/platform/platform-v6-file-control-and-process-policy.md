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
| 空间预留 | `Status FileHandle::preallocate(uint64_t)` | `MUST`；为增长到 minimum size 预留并按需增长，不缩小较大文件 |
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
  2. `FileHandle::size() const`、`resize(size)`、`preallocate(minimum_size)`；size 为逻辑字节数，
     resize 只改变 EOF；preallocate 在当前 EOF 小于 minimum 时先请求 backing 再增长 EOF，当前文件
     已经足够大则成功 no-op。它不缩小、不去稀疏化既有大文件，也不承诺连续 extent；
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

## 审计与冻结 FIX（`AUDITED`）

审计覆盖当前 Platform/FS 公开头与实现、中央组件图、package sample、三交付检查、Mem/Sched/IO
直接边；并复核 llama.cpp/ggml、Folly、Boost/std、TLPI/APUE/Windows Internals 与支持 OS 的官方合同。
没有发现需要第三个 factor、第三方产品依赖、对象 metadata 或公共性能仪器改动的理由。

### 语义与红队裁决

1. `size/resize/preallocate` 必须在已打开 handle 上工作，避免 path 二次解析；closed handle 为
   `FailedPrecondition`，超过 native signed offset 上限为 `OutOfRange`；resize 不移动 shared cursor；
2. preallocate 只在 `size() < minimum_size` 时工作：先请求 backing，再增长 EOF；OS 第二步失败可能
   保留已预留但不可见的空间。它不是 rollback transaction、不是 `resize` 的性能别名，也不承诺物理
   连续或消除已经存在的大 sparse file；
3. public lock 只承诺 cooperating handles/processes 的 shared/exclusive coordination。POSIX `flock`
   是 advisory/open-file-description lock；Windows `LockFileEx` 是 byte-range kernel lock，可能直接阻止
   冲突 I/O。network/distributed filesystem 可返回不支持或具有远端特有语义，不能宣传普遍分布式锁；
4. `try_lock` 只把明确的 would-block/lock-violation 映射为值 `false`，其它权限、I/O、closed-handle
   失败保留为 `Status`；`unlock` 不凭 HYLE metadata 猜测 ownership，close/destructor 由 OS 释放；
5. process ladder 不含 realtime。Windows normal 显式恢复 system-managed execution QoS；两个 Windows
   controls 任一失败都返回首个错误，但无法承诺原子 rollback。POSIX 提升优先级可能需要权限；Linux
   只有 startup-before-workers 才形成进程级效果；
6. 新 API 是用户/consumer 工作流，不进入现有 FileHandle read/write、Parker、CPU fact 或 scheduler
   hot path；`sizeof(FileHandle)` 必须保持一字。

### 同语义比较前置门

| candidate | 表面相似但不同合同的路线 | disposition |
|---|---|---|
| handle `resize` | `std::filesystem::resize_file(path)` | `NOT_COMPARABLE`：path lookup/replace race 与已打开 owner 不同；只作能力 shape 参考 |
| `preallocate` | sparse `resize`、Linux-only keep-size fallocate | `NOT_COMPARABLE`：physical-space、EOF 与 portability 不同 |
| blocking `lock` | `try_lock`、线程 mutex、lock-file creation | `NOT_COMPARABLE`：等待、进程范围、crash cleanup 不同 |
| process startup ladder | per-thread priority、realtime policy、只设 Windows class | `NOT_COMPARABLE`：scope/QoS/risk 不同 |
| whole-file cooperative lock | Folly whole-file shared/exclusive lock | shape/ownership 可比；本轮直接落同级 OS primitive，无额外算法层，故 P0 结构裁决足够 |

相同 checksum、最终文件大小或 enum 数量都不构成同合同。没有一行不同语义结果可以签 peer/SOTA
胜负；因此也没有为了赢 timing 增加 metadata、专用 overload 或 benchmark-only route。

### 冻结 FIX 集

| ID | factor | finite change |
|---|---|---|
| `P6-P01` | platform | 新增自包含 `process.h` 与 `process.cc`，实现五档显式 process startup priority/QoS，使用共享 Error translator |
| `P6-F01` | platform_fs | 在一字 `FileHandle` 上增加 size/resize/preallocate 和 shared/exclusive lock/try/unlock；三平台 native 实现，无状态字段 |
| `P6-T01` | tests | 补 invalid/closed/overflow、grow/shrink/no-shrink、cursor、shared/exclusive/try/close-release、process enum/安全 native 路线测试 |
| `P6-B01` | build/package | 登记 source 与 header-closure；Platform package sample 引用两个新增产品面；复放 exact factor 与 source/static/shared |
| `P6-D01` | docs | 更新 architecture、user guide、能力账本/watchlist/performance negative space，记录书籍与官方 ABI 依据及跨平台陷阱 |
| `P6-R01` | replay | 只读编译 Mem mapping、Sched、IO；若需要改 frozen consumer 产品，停止并形成独立 reopen trigger |

审计 finding 均已进入上述六项；无未处置 medium/high finding。FIX 集从此冻结并授权实现；出现范围外
产品需求必须退回 `AUDITED`，不能顺手添加。
