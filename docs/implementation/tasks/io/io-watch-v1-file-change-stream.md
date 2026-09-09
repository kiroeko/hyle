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
- 直接依赖 / 直接消费者：依赖冻结为 public `core,error,io_core,mem`，private
  `container,container_text,mem_vm,platform_fs,sched,time,util_core`；`sched` 没有公开类型泄漏，
  `mem_vm` 只提供已冻结的小型 fiber stack 档位，`util_core` 只承载实现内同步 primitive。
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

## 能力与形状审计（`AUDITED`）

只读队列已经完成。libuv 的跨平台 `uv_fs_event` 证明 portable baseline 只有 rename/change、filename
可为空、recursive 只在部分平台成立；它的 stat fallback flags 至今也未实现。Seastar
`experimental::fsnotifier` 证明 multiple watch + owning RAII token + batched await + explicit shutdown 是自然
形状，但它只包装 Linux inotify，且自己注明不会自动重建失效 watch。vLLM/SGLang 的真实调用则只需要
多个 exact file、异步迭代和 task cancellation。由此不能把 Linux mask 或 Windows rename-pair 偶然能力
伪装成 portable guarantee，也不能退化成 callback-only API。

| 能力族 | 准入旅程/来源 | 语义、生命周期与复杂度 | baseline | 裁决 / 动作 |
|---|---|---|---|---|
| W01 multiple target identity | vLLM/SGLang certificate/key/CA；Seastar token | create 时复制 UTF-8 path；稳定 input index；exact file 可暂不存在但 parent 必须存在；directory root 必须存在 | 无公开 owner | `MISSING / FIX` |
| W02 awaitable bounded batch | 两个 serving async iteration；Seastar batched wait | one outstanding `next`；返回 borrowed batch，直到下一次 next/close；bounded events/path bytes，O(events+bytes) parse | 无 | `MISSING / FIX` |
| W03 portable change vocabulary | libuv rename/change + 三 OS native events | content/metadata/create/remove/move-from/move-to/root-invalidated；rename correlation 尽力而为，0 表示未知 | 无 | `MISSING / FIX` |
| W04 atomic replace 与 coarse route | TLS editor replace；macOS FSEvents | exact file 观察 parent 才能跨 inode replace；无法给 filename 时 batch 明确 `rescan_required`，不能猜 identity | 无 | `MISSING / FIX` |
| W05 bounded backpressure/overflow | renderer burst；inotify/Windows buffer overflow | one sealed borrowed batch；producer 在消费前不覆写；native/library overflow 均返回 rescan-required，不静默丢失 | 无 | `MISSING / FIX` |
| W06 cancel/deadline/close | serving task cancellation；HYLE OperationOptions | next cancel 只结束该 wait；close 取消所有 native waits、join owner tasks、幂等；watcher 先于 IoContext/Scheduler 关闭 | 无 | `MISSING / FIX` |
| W07 cross-platform no-dedicated-thread mechanism | libuv/Seastar + HYLE shard-per-core | Linux one inotify fd/owner task；Windows one IOCP owner task per unique root；macOS one file-event FSEventStream over all roots，系统 dispatch callback 只发布 bounded batch；零 blocking-pool thread/busy poll | io_core 只缺 Windows submit | `PARTIAL / FIX`，依法重开 io_core v2 |
| W08 C++/delivery/docs | HYLE source/static/shared 产品 | `<hyle/io/watch.h>`；无 C/Native FFI；exact factor + IO façade；默认值、borrow、rescan 与 teardown recipe | factor 仅有 identity | `MISSING / FIX` |

public-surface 反向 trace 为空：此前没有 watcher API、兼容别名或 benchmark-only 入口需要保留。
`platform_fs` 的同步 stat/path/handle owner 不应承载 awaitable stream；`io_core` 只补 native submission，
`io_watch` 仍是 event semantics、lifetime、buffering 和用户文档的唯一 owner。

## API / usability freeze

```cpp
std::array targets{
    hyle::io::FileWatchTarget::file("tls/cert.pem"),
    hyle::io::FileWatchTarget::file("tls/key.pem"),
    hyle::io::FileWatchTarget::directory("shaders")};
auto made = hyle::io::FileWatcher::create(
    allocator, io, owner, targets, {.native_buffer_bytes = 16 * 1024});
if (!made) co_return;
auto watcher = std::move(*made);

for (;;) {
  auto changed = co_await watcher.next(operation);
  if (!changed) break;
  if (changed->rescan_required()) rebuild_authoritative_state();
  for (const auto& event : changed->events()) consume(event);
}
HYLE_RETURN_IF_ERROR(watcher.close());
```

- `FileWatchTarget::{file,directory}(std::string_view)` 只在 create 期间借用；watcher 复制完整 path，event 的
  `target_index` 是输入位置。exact-file event 的 relative path 为空；directory event 是一层 basename。
- `FileWatchEvent` 包含 bitmask change、borrowed relative path 与 optional nonzero rename correlation；
  `FileWatchBatchView` 包含 span、generation 和 batch-level `rescan_required()`。view 在下一次 next、close
  或 watcher 销毁时失效；用户要跨 await 保存必须复制。
- `next(OperationOptions)` 是零 coroutine-frame 的 awaitable；只允许一个在途 next。调用者运行在 watcher
  owner worker；event 已封存时 event 胜出，否则已触发 cancel/deadline 返回标准 typed status，watcher
  保持可继续使用。
- `close()` owner-fiber-aware、幂等：取消内部 token、等待全部 native pump 退休、关闭 handle；析构是
  防泄漏后备，但正常用户必须在 IoContext/Scheduler 停止前 close。move 转移唯一 owner；closed/invalid/
  concurrent-next 有明确 FailedPrecondition/Cancelled，不崩溃。
- 创建参数：至少一个 target；无 embedded NUL，长度/数量可表示；exact-file parent 存在即可，directory
  必须存在；buffer/event/path caps 非零并经 checked arithmetic。默认 16 KiB native buffer、256 events、
  16 KiB path bytes；Windows 每 unique native root 一个固定 buffer/小-stack pinned task，Linux 全部 root
  共用一个 inotify fd/task，macOS 全部 root 共用一个 FSEventStream 和系统 dispatch source。该成本在
  options/guide 可见，不随 event 数增长；FSEvents callback 只能向 bounded state 发布并唤醒 owner。
- 同批 `(target,path)` 使用预分配索引 O(1) expected merge flags；解析 O(native bytes + emitted events)，
  next 无逐事件 heap、无 busy poll/额外 OS thread。macOS directory/coarse parent notification，以及任意
  native/library overflow，必须置 rescan flag；用户重读 authoritative state，不依赖事件日志完整性。
- portable 保证 non-recursive；macOS 虽由递归 FSEvents source 提供机制，也必须在 factor 内过滤为 exact
  file/直接子项，深层事件不得泄漏到 API。root 按原生目录身份 canonicalize，子项按 native stream 返回的
  UTF-8 拼写 byte-exact 匹配；case-fold、Unicode normalization 与 symlink alias 等价不作跨平台承诺，用户
  应使用目录实际保存的名称，或观察 parent 后重读 authoritative state。rename pairing、每次 write 对应一个
  event、跨 network FS 可见性、递归、内容 diff 与 debounce 都不承诺。

## 冻结有限 FIX 集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| `IOW-F00` | Windows 无现成无线程 submit seam；macOS kqueue 不具备目录子文件 content 语义 | 先独立完成 `io-core-v2-file-watch-submission` 的 Windows IOCP op；Linux 复用 async read；本 factor 用 FSEvents file-events + dispatch lifecycle | io_core 重新 CLOSED；三平台机制/cancel/P0 |
| `IOW-F01` | factor/API 全缺 | 建立 watch header/source/test/export/target 与上面的 owning/borrowed shape | self-contained、invalid/OOM/move/lifetime |
| `IOW-F02` | target 与 native setup 全缺 | checked path copy/group；Linux inotify、Windows directory handle、macOS FSEvents roots/file-events | exact missing/replace、dir/root loss、Unicode/long path、nonrecursive filter、rollback |
| `IOW-F03` | event/batch/backpressure 全缺 | 固定 buffer/event/path storage、linear parse、expected O(1) coalesce、sealed generation、overflow/rescan | all kinds、burst/caps/overflow、view stability、无 silent loss |
| `IOW-F04` | async lifecycle 全缺 | 用 `spawn_task_on` + existing budget/cancel，single-owner next 与 idempotent close/join | cancel/deadline/late completion/concurrent-next/teardown/sanitizers |
| `IOW-F05` | 交付、指南、性能全缺 | exact factor + IO product、三交付/平台、guide/architecture/evidence、同合同 libuv P1 | 全部预声明门与独立红队 |

本表之外不加入 recursive watcher、polling fallback、callback façade、glob 或 application reload。任何公共
API、算法/常态成本变化在 candidate 前完成；P1 不反向塑造接口。

## 七问

1. **第三方：** 产品不链接第三方。libuv/Seastar/watchfiles 是需求与 peer evidence；inotify、IOCP 与
   Apple FSEvents/CoreServices 是系统机制，构建只增加适用平台的 system-framework link。
   libuv 只在 P1 harness 作同合同 peer，不泄漏其 handle/callback。
2. **能力：** 八族 MUST 均由 F00--F05 覆盖；recursive/network FS/debounce 有具名负空间。主 TLS 与正交
   renderer 单层目录都能由同一 API 完成。
3. **复用：** IoContext/proactor、Sched pinned task/cancel/CV、Error、Time、Mem allocator、Container
   bounded storage、Platform FS contract 均按 owner 复用；FSEvents dispatch source 是 host completion
   mechanism，不是第二套产品 scheduler；不新建 timer、thread pool 或 error。
4. **性能：** P0 固定 one native op/root、bounded storage、无 per-event allocation/busy poll；P1 以完整
   exact-replace 和 directory-burst consumer 对 libuv，双顺序/扰动/10% margin，不比 raw call。
5. **FFI：** 当前唯一消费者是 C++ source/static/shared；没有具名 Java/C# owner，N/A，不恢复伪 C API。
6. **交付：** `io_watch` 独立 compiled factor，关闭后由 `hyle::io` 聚合；header/target/export/offline/
   relocation 和 transitive runtime DLL 都要实测，不能借 io_core sample 冒充。
7. **易用性：** factory helper、typed batch、统一 OperationOptions、explicit rescan/close；guide 按 TLS 和
   resource reload 告诉用户如何选，突出 borrowed view、non-recursive、atomic replace 和 shutdown 顺序。

## Candidate / 验证 / terminal

尚未产生 candidate。先完成 F00；随后只允许 F01--F05 一个有限产品批次。验证与裁决沿用 SCOPED 卡的
预声明矩阵，任何未运行适用门均不得写 PASS。
