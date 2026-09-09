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

进入 `AUDITED` 前只读核对三 backend 的 operation admission、cancel、teardown、tag/bit 空间及测试 seam；
本卡当前不授权产品改动。
