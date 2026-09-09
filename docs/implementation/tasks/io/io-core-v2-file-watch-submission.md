# IO Core v2：文件观察原生提交原语

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round ID：`io_core` / IO core maintainers /
  `io-core-v2-file-watch-submission`；kind：`REOPEN`；baseline：
  `10c4e840beb5e6575065d484fcf0b7c3374ea1ed`。上一轮 terminal 事实见
  [`io-core v1`](io-v1-core-factorization.md)，本卡不改写其历史裁决。
- trigger：`io_watch` v1 的具名真实 consumer 审计证明，现有 Proactor 能异步 read Linux inotify fd，
  但 Windows 没有 overlapped `ReadDirectoryChangesW` 的 completion/cancel seam。把长期等待卸载到
  BlockingPool 会按 active native root 永久占用 OS thread，并与 IoContext 已有 IOCP 重复等待；这是
  新的通用机制/成本 trigger，符合 io_core v1 terminal reopen 条件。
- 正常任务：`io_watch` 的 owner-pinned Windows task 在既有 worker proactor 上提交一次目录变化读取，
  completion 仍经同一个 token/Waker 路由，并能由现有 `submit_cancel` 及时退休。
- 主/反偏置纵切：主路径是 Windows TLS exact-file atomic replace；反偏置是 Windows 单层 shader/asset
  directory burst。Linux 继续复用普通 async read；macOS 的 FSEvents 由 `io_watch` 拥有，不伪装成
  Proactor byte-read。
- scope：仅在 Windows installed-detail `Proactor` 增加一个 fixed-semantics directory-change submission；
  实现 overlapped `ReadDirectoryChangesW`，补直接机制测试、注释与三交付证据。只允许增加一个 operation
  kind；不改变 `Completion` layout/flags 或其它 backend。
- non-goals：公开 `FileWatcher`、path/target/event parsing、batch/overflow/rearm policy、递归 watch、轮询
  fallback、第二 event loop、普通 File/TCP/UDP 算法或 API、backend capability public bit、Native FFI、
  其它模块、公共 checker/性能仪器。
- 依赖 / consumer：中央 io_core 依赖不变；唯一直接产品 consumer 是正在 AUDITED 的 `io_watch`，机制
  测试是 pre-product proof。实现不得令 io_core 反向依赖 io_watch。
- tier：`P0`；该原语一次 native submit、一次既有 completion、零逐事件 allocation/lock/额外 thread。
  event parsing 与 end-to-end 性能由 io_watch 的 P1 承担。本轮不运行旧 HTTP P1，也不制造孤立微计时。
- 机器 / 验证：W1-WIN 直接 arm→native mutation→completion、cancel-before/during、queue-full/invalid
  handle 与 rearm；W1-WSL/M1-MAC 跑 existing io_core selected regression，证明条件编译和既有 backend
  不变；ASan/UBSan/TSan 中适用的非 OS-race 路径；header self-containment；source/static/shared exact
  io_core install consumer；P0 operation-slab/cancel/in-flight 账本；closure/docs/diff。
- 预算：只读审计 30 分钟、一个实现批次 60 分钟、取证 90 分钟；最多一次私有机制纠正，不改变
  `Completion` 既有 bit 语义或普通 operation layout 以追逐 benchmark。
- stop / blocker：三平台中任一路线不能经已有 completion/cancel 生命周期安全退休，或必须改变普通
  public IO API/常态成本，则 `BLOCKED` 并返回 io_watch 重新裁决；否则全部适用门通过即重新 CLOSED。

## 机制审计与有限 FIX（`AUDITED`，实现前纠正）

三 backend 的只读追踪已完成：Windows 每个真实 operation 使用 fixed-capacity `IoOp` slab，
`GetQueuedCompletionStatusEx` 后才 recycle，`CancelIoEx` 提供 terminal completion；Linux io_uring 与 epoll
已经能对 inotify fd 完成 read/cancel。进一步核对 macOS 语义发现，目录 `EVFILT_VNODE` 只报告目录入口
变化，不能可靠报告子文件原地内容修改，无法满足已冻结的 non-recursive directory 合同；libuv 也因此在
macOS 目录路径使用 FSEvents。故实现前删除错误的 neutral 三平台 seam：macOS FSEvents 由 `io_watch`
factor 直接拥有，Linux 复用 `submit_read`，`io_core` 只补 Windows 真正缺失的 IOCP operation。该纠正
收窄本卡改动面，不扩张公开能力，也没有产品代码需要撤回。

### Internal API freeze

```cpp
#if HYLE_OS_WINDOWS
bool Proactor::submit_directory_changes(std::intptr_t directory,
                                        void* buffer,
                                        std::uint32_t length,
                                        std::uint64_t user_data) noexcept;
#endif
```

- 从既有 slab 取得一个 `kDirectoryChanges` op，调用 overlapped `ReadDirectoryChangesW`；固定观察 file/dir
  name、attributes、size、last-write、creation 与 security，`watch_subtree=FALSE`。buffer 由 caller 借到
  terminal completion，结果为 native bytes，error 保留 Win32 negative domain；handle 必须先 bind IOCP。
- 方法仍位于 `hyle/io/detail` 且只在 Windows 声明；只有 factor 实现可依赖，用户语义由 `io_watch`
  拥有。普通 read/write/socket/file 操作没有新增 allocation、map、metadata 或 submit branch；Linux/macOS
  的 header/object 不产生伪 symbol。

| ID | 已证实问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `IOCORE-W01` | Windows 无 directory-change operation | Windows-only detail declaration、一个 slab kind 与 overlapped submit；reap/cancel/teardown 复用原路径 | mutation completion、queue full、cancel、buffer lifetime |
| `IOCORE-W02` | 新 detail symbol 的交付证据缺失 | Windows direct test；Linux/macOS regression；exact io_core source/static/shared/header 与 P0 账本 | 全部预声明门，terminal receipt |

### 七问

1. 无第三方或版本变化；直接调用 host OS。libuv/Seastar 只是机制证据。
2. 本卡只拥有 Windows 缺失的 internal submission；Linux 现有 read 和 macOS FSEvents 均由 io_watch
   组合，path/event/rearm 也属于 io_watch。
3. Windows 复用固定 slab、completion token、`submit_cancel` 与 worker park merge；没有 blocking pool、
   新 thread/loop、allocator、map 或 runtime decision authority。
4. P0：每次 arm 一个 syscall/native op；非 Windows 为零新增路径，Windows 普通 IO path 只多一个枚举
   case 且无常态 metadata。完整性能交 io_watch P1，故本卡计时 `NOT_APPLICABLE`。
5. detail C++ mechanism 不是 C/Native FFI；不导出稳定跨工具链 ABI 承诺。
6. 中央依赖不变；新增 symbol 随现有 io_core source/static/shared export，exact factor consumer 证明。
7. 普通用户不可见；detail 注释必须足以防止 buffer/token lifetime 与 macOS flag 误读，公开 recipe 只在
   io_watch guide 出现。

本卡至此保持 `AUDITED`，只授权 W01--W02 一个实现批次；没有 candidate，不能提前签 PASS。
