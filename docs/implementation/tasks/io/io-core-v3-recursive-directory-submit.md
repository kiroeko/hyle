# IO Core v3：Windows 递归目录提交语义

<!-- MODULE CLOSURE RECORD: SCOPED -->

## Closure card

- component / owner / round：`io_core` / IO core maintainers /
  `io-core-v3-recursive-directory-submit`；kind=`REOPEN`；canonical checkout=
  `C:\public\project-hyle\hyle`；baseline=
  `6c0b877bab4ff574a6073d8f336423f8f94418dd`；前轮 terminal record 是
  [`io-core-v2-file-watch-submission`](io-core-v2-file-watch-submission.md)。
- trigger 与证据：正在 `SCOPED` 的 `io_watch` G05 源码预审证明，Windows 已关闭 detail seam 把
  `ReadDirectoryChangesW` 的 `bWatchSubtree` 固定为 `FALSE`，不能承载已准入的显式 recursive-tree
  target。绕过 Proactor 会复制 IOCP operation/cancel/completion 生命周期；前轮记录已把“io_watch
  证明 seam 不足”列为合法 reopen trigger。
- 正常 consumer 任务：`io_watch` 的 owner-pinned Windows pump 使用同一个 IOCP operation，在 direct
  target 传 `FALSE`、recursive-tree target 传 `TRUE`，buffer/token 继续借到 terminal completion。
- 主 / 反偏置纵切：主纵切是递归 model/asset tree；反偏置是既有 exact-file/direct-child target，必须
  保持原提交、完成与取消语义。
- scope：只让 Windows-only installed-detail `submit_directory_changes` 接受明确的 subtree 语义并透传给
  OS；补 true/false 直接机制测试与调用方迁移。不得增加另一种 operation、线程、队列或公开产品 API。
- non-goals：事件解析、相对路径、Linux inotify 树、macOS FSEvents、watch policy、公开 Native FFI、
  G03/G04/G06、其它 IO 算法或公共性能工具。
- 直接依赖 / consumer：中央依赖不变；唯一产品 consumer 是 `io_watch`。`io_core` 不得反向依赖它。
- 性能 tier：`P0`。新增值只是一次 native submit 的已有 syscall 参数；普通 IO operation、slab、对象 layout
  和 non-Windows binary 均不得变化。源码账单足以裁决，不运行孤立纳秒 benchmark。
- 适用机器 / 验证：Windows MSVC direct true/false mutation、rearm/cancel/admission；既有 io_core
  regression；Linux/macOS 条件编译；changed-header；exact source/static/shared IO consumer；P0、红队、
  closure/docs/diff。sanitizer 只覆盖可移植既有路径，不冒充 Windows syscall 插桩。
- 调查 / 纠正预算：一次源码与 Win32/libuv 契约审计，一个产品批次；若需要改变 completion layout、普通
  submit 成本或公开 IO API，立即 `BLOCKED`。
- stop：上述有限 seam 及调用方不能完整表达两种语义，或 Windows 递归需要第二 runtime 时 `BLOCKED`；
  否则通过预声明门后恢复 `CLOSED`。

## SCOPED 待审

1. 裸 `bool watch_subtree` 与 detail enum 二选一；公开 target 枚举不得泄漏进 `io_core`。
2. true/false 必须共享相同 operation kind、filter、buffer lifetime、cancel 和 completion 路线。
3. 在 `AUDITED` 冻结 API、有限 FIX 和成本账单前不改产品。

## 验证与裁决

尚未执行；`SCOPED` 阶段只允许只读审计，未运行项均为 `UNPROVEN`。
