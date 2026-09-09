# IO Core v2：文件观察原生提交原语

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round ID：`io_core` / IO core maintainers /
  `io-core-v2-file-watch-submission`；kind：`REOPEN`；baseline：
  `10c4e840beb5e6575065d484fcf0b7c3374ea1ed`。上一轮 terminal 事实见
  [`io-core v1`](io-v1-core-factorization.md)，本卡不改写其历史裁决。
- trigger：`io_watch` v1 的具名真实 consumer 审计证明，现有 Proactor 能异步 read inotify fd，却没有
  Windows overlapped `ReadDirectoryChangesW` 或 macOS `EVFILT_VNODE` 的同一 completion/cancel seam。
  把长期等待卸载到 BlockingPool 会按 active native root 永久占用 OS thread，并与 IoContext 已有
  IOCP/kqueue 重复等待；这是新的通用机制/成本 trigger，符合 io_core v1 terminal reopen 条件。
- 正常任务：`io_watch` 的 owner-pinned background task 在既有 worker proactor 上提交一次目录/vnode
  watch，completion 仍经同一个 token/Waker 路由，并能由现有 `submit_cancel` 及时退休。
- 主/反偏置纵切：主路径是 Windows TLS exact-file atomic replace；反偏置是 macOS 单层 shader/asset
  directory invalidation。Linux inotify 是第三个平台的同一机制闭包。
- scope：仅在 installed-detail `Proactor` 增加 neutral file-watch submission 与必要 completion bits；实现
  Windows `ReadDirectoryChangesW`、Linux inotify-buffer read 路由、macOS vnode one-shot；补直接机制测试、
  注释与三交付证据。允许为 macOS 增加独立 pending-watch map，为 Windows 增加一个 operation kind。
- non-goals：公开 `FileWatcher`、path/target/event parsing、batch/overflow/rearm policy、递归 watch、轮询
  fallback、第二 event loop、普通 File/TCP/UDP 算法或 API、backend capability public bit、Native FFI、
  其它模块、公共 checker/性能仪器。
- 依赖 / consumer：中央 io_core 依赖不变；唯一直接产品 consumer 是正在 AUDITED 的 `io_watch`，机制
  测试是 pre-product proof。实现不得令 io_core 反向依赖 io_watch。
- tier：`P0`；该原语一次 native submit、一次既有 completion、零逐事件 allocation/lock/额外 thread。
  event parsing 与 end-to-end 性能由 io_watch 的 P1 承担。本轮不运行旧 HTTP P1，也不制造孤立微计时。
- 机器 / 验证：W1-WIN、W1-WSL、M1-MAC；各平台直接 arm→native mutation→completion、cancel-before/
  during、queue-full/invalid handle、one-shot rearm；existing io_core selected tests；ASan/UBSan/TSan 中适用
  的非 OS-race 路径；header self-containment；source/static/shared exact io_core install consumer；P0
  operation-slab/pending-map/cancel/in-flight 账本；closure/docs/diff。
- 预算：只读审计 30 分钟、一个实现批次 60 分钟、取证 90 分钟；最多一次私有机制纠正，不改变
  `Completion` 既有 bit 语义或普通 operation layout 以追逐 benchmark。
- stop / blocker：三平台中任一路线不能经已有 completion/cancel 生命周期安全退休，或必须改变普通
  public IO API/常态成本，则 `BLOCKED` 并返回 io_watch 重新裁决；否则全部适用门通过即重新 CLOSED。

## 机制审计与有限 FIX（`AUDITED`）

三 backend 的只读追踪已完成：Windows 每个真实 operation 使用固定-capacity `IoOp` slab，
`GetQueuedCompletionStatusEx` 后才 recycle，`CancelIoEx` 提供 terminal completion；Linux io_uring 与 epoll
已经能对 inotify fd 完成 read/cancel；macOS kqueue 的 `pending_in` 已是固定 admission map，vnode 可以作为
独立 opcode 共用该 map而无需给每个普通 IoContext 预分配第三张表。现有 completion bits 只占低三位，
可在高位登记 internal normalized watch bits，不改变 struct size/layout。

### Internal API freeze

```cpp
enum class FileWatchInterest : std::uint32_t { /* high-bit normalized mask */ };
bool Proactor::submit_file_watch(std::intptr_t handle, void* buffer,
                                std::uint32_t length,
                                FileWatchInterest interests,
                                std::uint64_t user_data) noexcept;
```

- Windows：从既有 slab 取得一个 `kFileWatch` op，调用 overlapped `ReadDirectoryChangesW`；buffer 由 caller
  借到 completion，结果为 native bytes，error 保留 Win32 negative domain。handle 必须先 bind IOCP。
- Linux：同名方法直接走已有 `submit_read`；interest 在 inotify add-watch 时已经冻结，因此 submission
  不产生第二套 mask decision。result 是 inotify buffer bytes。
- macOS：在既有 `pending_in` 插入 `kFileWatch`，以 `EVFILT_VNODE|EV_ONESHOT` arm；completion result 为 0，
  normalized content/metadata/rename/remove 位写入现有 flags 高位。cancel 根据 opcode 删除正确 filter。
- 无 backend capability/public selection：HYLE 支持的三 OS 都有路线。方法仍位于 `hyle/io/detail`，只有
  factor 实现可依赖；用户 API 由 io_watch 拥有。普通 read/write/socket/file 操作没有新增 allocation、
  map、branch-at-submit 或 layout；macOS reap 仅在收到 vnode filter 时走新分支。

| ID | 已证实问题 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `IOCORE-W01` | 缺 neutral interest/completion vocabulary | 在 detail header 增加高位 mask、operators 与一个 submit 声明 | existing bits/layout static checks、header self-contained |
| `IOCORE-W02` | Windows 无 directory-change operation | 一个 slab kind 与 overlapped submit；reap/cancel/teardown 复用原路径 | mutation completion、queue full、cancel、buffer lifetime |
| `IOCORE-W03` | Linux 需同名 portable seam | forwarding 到 existing read；不改 io_uring/epoll 算法 | inotify completion + forced epoll + cancel |
| `IOCORE-W04` | macOS 无 vnode operation | 复用 pending_in，增加 vnode arm/reap/cancel 分支；禁止新常驻 map | NOTE mapping、one-shot/rearm、cancel、ordinary socket regression |
| `IOCORE-W05` | 新 detail symbol 的交付证据缺失 | 三平台测试、exact io_core source/static/shared/header 与 P0 账本 | 全部预声明门，terminal receipt |

### 七问

1. 无第三方或版本变化；直接调用 host OS。libuv/Seastar 只是机制证据。
2. 本卡只拥有一个 internal submission family，三 OS 的 submit/completion/cancel/teardown 全生命周期由
   W01--W04 覆盖；path/event/rearm 属 io_watch。
3. 复用固定 slab、pending map、completion token、`submit_cancel`、worker park merge；没有 blocking pool、
   新 thread/loop、allocator 或 runtime decision authority。
4. P0：每次 arm 一个 syscall/native op；Windows/Linux 不新增常态 metadata，macOS 复用既有 map；普通
   IO path 代码结构不变。完整性能交 io_watch P1，故本卡计时 `NOT_APPLICABLE`。
5. detail C++ mechanism 不是 C/Native FFI；不导出稳定跨工具链 ABI 承诺。
6. 中央依赖不变；新增 symbol 随现有 io_core source/static/shared export，exact factor consumer 证明。
7. 普通用户不可见；detail 注释必须足以防止 buffer/token lifetime 与 macOS flag 误读，公开 recipe 只在
   io_watch guide 出现。

本卡至此进入 `AUDITED`，只授权 W01--W05 一个实现批次；没有 candidate，不能提前签 PASS。
