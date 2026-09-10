# IO Watch v2：递归目录树变化流

<!-- MODULE CLOSURE RECORD: SCOPED -->

## Closure card

- component / owner / round：`io_watch` / IO maintainers / `io-watch-v2-recursive-tree`；kind=`RESUME`；
  canonical checkout=`C:\public\project-hyle\hyle`；baseline=
  `5d88227ec193c02ef787ef4c99cfb0cec2be2b6e`；前轮 terminal record 是
  [`io-watch-v1-file-change-stream`](io-watch-v1-file-change-stream.md)。
- trigger 与证据：[基础能力扩展决策](../../../architecture/foundation-capability-expansion.md)已把 G05
  准入为独立能力：现有 exact-file/direct-child watcher 无法表达 model bundle、tokenizer/template tree、
  shader/include tree 或多层资产目录的统一失效流。libuv `3e606bc30a84` 在 Windows 与 macOS 提供
  `UV_FS_EVENT_RECURSIVE`，Windows `ReadDirectoryChangesW(..., TRUE, ...)` 与 macOS FSEvents 是原生
  recursive mechanism；Linux inotify 明确只观察单目录，必须维护每目录 watch。Seastar
  `78f64e2317b1` 的 RAII watch/batched wait 只作 Linux 生命周期反证，不能冒充 recursive owner。
- 正常 consumer 任务：服务或渲染 runtime 先建立一个树 watcher，再扫描 authoritative tree；随后等待
  bounded batch，把 root-relative path 映射到 model sidecar/tokenizer/template/shader/asset cache 条目；
  overflow、目录新建/移动造成的观察窗口或 native path 丢失时收到 `rescan_required` 并重新扫描。
- 主产品纵切 / 具名来源：推理服务热更新一个包含多层 tokenizer、chat template、adapter metadata 与
  model sidecar 的 bundle；vLLM `51da0ca66c80` 和 SGLang `dcebe8c4733a` 的 watchfiles reload 证明异步
  reload owner 真实存在，G05 只下沉通用树变化流，不下沉模型格式、reload/debounce 或发布策略。
- 正交反偏置纵切：渲染 runtime 观察 shader include 与 asset-manifest tree；新增/rename 子目录后，其后代
  仍必须进入同一 target 的 root-relative stream。该纵切防止 API 被 TLS 的少量 exact files 塑形。
- 跨模块 public journey 与 owner：`io_watch` 拥有 target depth、native watch set、relative-path
  normalization、bounded batch、rescan/failure 和 close/drain；`io_core` 拥有 IoContext/async native read；
  Sched 拥有 cancel/deadline/wake；Platform-fs 提供已关闭的路径/文件原子，Container/Mem 提供 bounded
  metadata；应用拥有初始/恢复扫描、内容解析、debounce、缓存失效和新对象发布。
- scope：为现有 `FileWatcher` 增加显式 recursive-tree target/option；事件路径相对 target root；保留 exact
  file 与 direct directory 默认语义；新建/移动/删除目录后维持后续覆盖；三平台原生 backend；Linux
  O(directory count) watch set、资源上限与 typed failure；overflow/path-loss/root-loss 的 rescan；best-effort
  rename correlation；既有 single-owner、one-next、bounded batch、deadline/cancel/close/drain 与三交付。
- non-goals：初始目录 snapshot、内容 diff/hash、glob/regex、network filesystem polling、debounce/reload、
  symlink/junction traversal、hard lossless event log、multi-consumer service、第二 scheduler/loop、Native FFI、
  G03/G04/G06、Compute/Graph，或为 benchmark 添加另一套 light/fast watcher。
- 直接依赖 / 直接消费者：依赖保持 public `core,error,io_core,mem`，private
  `container,container_text,mem_vm,platform_fs,sched,time,util_core`；不得令任何已关闭下层反向依赖
  `io_watch`。直接消费者是公开 IO façade、推理 bundle invalidation recipe 与渲染 asset-tree recipe。
- 性能 tier 与原因：`P0`。G05 的关键性能属性是 native blocking wait、零 busy poll、steady event 不逐项
  heap allocate、固定 batch bounds、Windows/macOS O(1) stream owner，以及 Linux O(D) watch metadata 和
  O(1) descriptor lookup；基础能力决策明确 watch overflow 以正确性/资源界为主。v1 已有 non-recursive
  同合同 P1 不重跑；只有源码账单留下无法解释的 material CPU 路径且存在同语义 peer 时，才另立 P1
  工具/产品任务，不能在本卡靠采样寻找设计。
- 适用机器 / 仪器 / margin：Windows/MSVC、WSL/GCC 14、macOS/AppleClang 21 做 correctness/P0；P0 核对
  allocation、watch/syscall、descriptor lookup、path copy、wake、queue、默认路径 branch/layout。无计时
  margin；必要短诊断不产生产品 WIN/LOSE，也不授权改 API。
- 调查与纠正预算：一次 libuv/Seastar/三 OS 源码与当前 HYLE 全路径审计；AUDITED 前冻结一个 API/backend
  方案和至多三个有限 FIX 批次（API/storage、Windows/macOS、Linux）；候选后一次红队。性能先做源码
  时序/成本账单，禁止以反复运行测试代替根因调查。
- 预声明验证：现有 exact/direct-child 全回归；预存在深树、空树、unicode/长路径、重叠 targets、同 root
  direct+recursive；新建/删除/rename/move-in/move-out 子树及随后后代变化；overflow/rescan、watch limit/
  OOM、root replace/loss、symlink 不跟随；cancel/deadline/close/late completion/move/single-owner；idle 无
  busy poll、bounded metadata；三平台、ASan/UBSan/TSan、逐头、公开 IO source/static/shared、直接 consumer；
  文档、build/component contract、候选后红队与 P0 账单。
- stop / blocker：若 Linux 不能在有限 O(D) metadata、无 dedicated polling thread 的前提下，在子目录结构
  变化后恢复持续覆盖并明确报告观察窗口；或任一 OS 只能通过隐藏无界队列/静默丢失实现，则本轮
  `BLOCKED`。native overflow、watch limit 或 rename 信息不足必须成为 rescan/typed error，不得降低合同、
  假装无损，也不得为得到 benchmark PASS 改默认 API。

## SCOPED 不变量与待审问题

1. recursive 必须显式选择；既有 `FileWatchTarget::directory()` 永远只表示 direct children。
2. recursive event 的 `relative_path` 是 target-root-relative UTF-8 路径，不暴露 native watch descriptor 或
   绝对路径；exact file 仍为空，direct directory 仍为 basename。
3. `rescan_required` 是正确性信号：consumer 收到后重读 authoritative state，不能把它记成普通日志后继续
   假定事件序列完整。
4. 不跟随 symlink/junction，避免越出 root、循环和平台语义分裂；root 本身的失效仍沿用 terminal 合同。
5. AUDITED 前必须裁决：公开形状采用 `tree(path)` 还是 depth option；Linux watch-capacity 形状；重叠 root
   的去重/routing；新目录和目录 rename 的无竞态恢复步骤；Windows/macOS 粗事件如何统一成 rescan。
6. 产品代码在 coverage map、OS 时序、复杂度账单和有限 FIX ledger 冻结前不得修改。

## 能力与形状审计

待只读审计完成后填写并把状态迁移到 `AUDITED`。

| 能力族 | 层级 | 当前路线 | 初始状态 | 待裁决 |
|---|---|---|---|---|
| explicit recursive target | 原子机制 | 只有 file/direct directory | MISSING | `tree(path)` 与 depth option 二选一 |
| root-relative descendant event | 原子机制 | backend 主动过滤 descendants | MISSING | path ownership、coalescing key 与 overlap routing |
| dynamic subtree coverage | 复合功能 | Linux 只有 root watch | MISSING | create/rename/delete 后安装、移除或重建 watch set |
| loss/resource signalling | 原子机制 | native overflow → rescan；无 tree capacity | PARTIAL | watch-limit/OOM/path-loss 的 batch/terminal 顺序 |
| lifecycle/backpressure | 复合功能 | bounded batch、single owner、cancel/close | COMPLETE | KEEP，并证明 recursive 不复制 owner |
| snapshot/debounce/reload | 应用策略 | 应用拥有 | OUT_OF_SCOPE | REJECT |

### 有限改动集

尚未冻结；AUDITED 只允许从上表导出至多三个产品批次。

### API / usability freeze

尚未冻结。普通调用必须保持“create watcher → 扫描 authoritative tree → await batches → rescan signal 后再扫 →
close before IoContext/Scheduler”的自然旅程；不得在实现或 benchmark 之后反推 API。

## 验证、红队与裁决

尚未执行。SCOPED 阶段只允许只读审计；任何未运行项均为 `UNPROVEN`。

