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
