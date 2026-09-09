# IO Watch v1：可取消的跨平台文件变更流

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

- component / owner / round ID：`io_watch` / IO maintainers / `io-watch-v1-file-change-stream`；
  task kind：`NEW`；canonical checkout：`C:\public\project-hyle\hyle`；baseline commit：
  `71f3d0368bc5f6878e8093bac37b767f9f468be4`。
- trigger 与证据：项目级纵切审计中的 `F03`。vLLM
  `51da0ca66c8065619c79e35dff97aa99aeaf5644` 与 SGLang
  `dcebe8c4733a5cb802fb679e0c8fbb886e2801b0` 都用可取消的异步文件观察完成 TLS
  certificate/key/CA reload；渲染资产与 shader 的变更失效提供正交需求。libuv
  `3e606bc30a842701f8d12007d3ade662d258b76b`、Asio
  `8806a6803cde7054c3049d3666d3ec36786568c5`、Seastar
  `78f64e2317b19c7e34ba61f5e8ea85f083eca1da` 和 Folly
  `65749da412d92fe27d87ae45a8ef670720712858` 只作成熟机制/形状证据；不自动成为产品依赖。
- 正常 consumer 任务：服务在一个 `IoContext` owner 上观察一组 certificate/config/model sidecar
  文件；编辑器以 write、rename 或 atomic replace 发布新版本；consumer await 下一批变化、重读并验证，
  服务关闭时用既有 deadline/cancel 合同立即停止等待。
- 主产品纵切 / 具名来源：低延迟异步服务的 TLS/config reload，具名实现为上述 vLLM/SGLang
  `watchfiles.awatch` 旅程；本轮只下沉通用 OS 文件变化与生命周期，不下沉证书解析或 reload policy。
- 正交反偏置纵切：渲染/资源流送中的 shader、asset manifest 与单层资源目录失效。它要求目录入口变化、
  rename/replace、burst 合并和 overflow 后显式 rescan，防止 API 只适合三个固定 TLS 文件。
- 跨模块 public journey 与 owner：`io_watch` 拥有异步流、event normalization、bounded buffering、
  cancel/close；`io_core` 拥有 `IoContext`、owner worker 与 `OperationOptions`；Sched 拥有 task/cancel/wake；
  Platform FS 拥有同步路径/文件机制；Error/Time/Mem/Container 只提供其已冻结的通用值、错误、时间、
  allocation 和 bounded storage。应用拥有 debounce 时长、重读/验证、证书/model/shader 发布策略。
- scope：一个新的内部 factor、公开 `<hyle/io/watch.h>`、其 source/test/package owner；支持多个 exact-file
  target 与 portable non-recursive directory target；文件 create/write/metadata/remove/rename/atomic replace；
  bounded batch/coalescing、overflow/rescan-required、root invalidation；single-owner awaitable `next`，显式
  deadline/cancel/close，错误与 teardown；Windows/Linux/macOS 原生机制与三种 C++ 交付；IO 产品 façade
  在 factor 关闭时聚合它。
- non-goals：递归目录树、network/remote-filesystem polling fallback、内容 diff/hash、glob/regex engine、
  application debounce/reload、watcher multiplex service、第二套 scheduler/event loop、Native FFI/C API、
  Compute/Graph 或其它已关闭 factor 的无证据重开。递归树只在具名 consumer 证明单层目录+rescan
  无法自然组合时另开能力轮。
- 直接依赖 / 直接消费者：依赖上限暂冻结为 public `core,error,io_core,mem,sched`，private
  `container,container_text,platform_fs,time`；AUDITED 必须删掉未被公开 layout/实现真实使用的边。
  直接消费者为 IO product façade、exact-factor package sample 与两个 compile-only TLS/resource reload recipe；
  当前没有产品实现反向依赖。
- 性能 tier 与原因：`P1`。等待是 OS 主导，但 idle polling、逐 event allocation、额外线程/loop、无界队列、
  多余 wake/syscall 会在长期服务和 burst 热更新中传播；需要 P0 成本账本和同合同 full-consumer P1，
  不能用 raw OS call 对完整 HYLE lifecycle。
- 适用机器 / 仪器 / material margin：W1-WIN、W1-WSL、M1-MAC；统一 process-isolated paired runner，
  primary 为 exact-file atomic replace，holdout 为 non-recursive directory burst；HYLE/libuv 都从 watcher
  已建立到 consumer 收到并分类同一批事件，均含自己的 loop/task dispatch，A/B 与 B/A、空白对照、
  path/placement 扰动。工程非劣 material margin 预声明为 `10%`；同时强制 idle CPU≈0、无 busy poll、
  无丢失且 overflow 可观察。AUDITED 只能收紧语义等价矩阵，不能因结果改 workload 或 margin。
- 调查与纠正预算：能力/peer/OS 审计最多 90 分钟；一个有限实现批次最多 180 分钟；correctness、三平台、
  三交付、P1 与独立红队最多 180 分钟。最多两轮只读审计、一次 API 纠正与一次私有机制纠正；每项门
  通过后不因文档或状态变化重跑。超过预算按已知事实裁决，不无限扩建公共仪器。
- 预声明验证：public header self-containment；exact file 与 directory 的 create/write/metadata/remove/
  rename/atomic-replace；多 target、unicode/长路径、empty/invalid/missing/root-disappear、overflow、OOM；
  cancel-before/during/after、deadline race、close-before/during/after、late native completion、move/destroy、
  single-owner/concurrent-next 误用；idle/burst/backpressure；ASan/UBSan/TSan；Windows/Linux/macOS；
  exact-factor 与 IO façade 的 source/static/shared/offline/install/relocation；用户/architecture/性能文档；
  P0 allocation/syscall/wake/thread 账本和上述 P1 primary+holdout。
- stop / blocker 条件：只有全部准入 MUST、适用门、自然 API、交付和工程性能通过才 `CLOSED`。任一 OS
  无法在不引入第二 scheduler/loop 或永久 dedicated thread 的情况下提供可取消、无 busy-poll 的合同，
  或同合同 peer/三平台/三交付无法取得，即进入 `BLOCKED`，记录 owner 与唯一恢复条件；不得用
  polling 冒充跨平台成功、把未运行门写成 PASS，或顺手扩入 recursive/network-FS。

## 进入 `AUDITED` 前的只读队列

- 双向 trace `io_core` ownership、`IoContextAccess`、blocking pool、CancelCallback 与现有 package/target
  生成规则，证明是否无需重开任何已关闭依赖；
- 对 libuv 的 Linux inotify、Windows `ReadDirectoryChangesW`/IOCP、macOS kqueue/FSEvents 生命周期，
  以及 Asio/Seastar/Folly 的取消与 ownership 逐项裁决；
- 冻结 event vocabulary、target identity/path ownership、batch lifetime、overflow/rescan、missing/root loss、
  concurrent-next 和 close 语义；
- 形成 capability-family map、public surface、七问、有限 FIX 集、调用草图、语义等价 P1 contract；
- 决定原生 fd/handle 是接入现有 proactor 还是经共享 blocking pool，必须有成本与取消证据；若需要修改
  `io_core` 产品/API/常态算法，停止并依法形成独立 reopen trigger，不能在本卡偷改。

当前只完成范围登记；尚未签发 `AUDITED`，不授权产品或测试实现改动。
