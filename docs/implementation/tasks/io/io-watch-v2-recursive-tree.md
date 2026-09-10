# IO Watch v2：递归目录树变化流

<!-- MODULE CLOSURE RECORD: AUDITED -->

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

## 审计裁决与不变量（`AUDITED`）

1. recursive 必须显式选择；既有 `FileWatchTarget::directory()` 永远只表示 direct children。
2. recursive event 的 `relative_path` 是 target-root-relative UTF-8 路径，不暴露 native watch descriptor 或
   绝对路径；exact file 仍为空，direct directory 仍为 basename。
3. `rescan_required` 是正确性信号：consumer 收到后重读 authoritative state，不能把它记成普通日志后继续
   假定事件序列完整。
4. 不跟随 symlink/junction，避免越出 root、循环和平台语义分裂；root 本身的失效仍沿用 terminal 合同。
5. 公开形状冻结为 `FileWatchTarget::directory_tree(path)` 与独立 `kDirectoryTree` kind；depth 是 target
   语义而不是整个 watcher 的 option，故一个 watcher 可自然混用 exact/direct/tree，也没有含混 `tree()`。
6. `FileWatchOptions::max_watched_directories` 默认 `65536` 且必须非零；它限制 native directory resource，
   Windows/macOS 按唯一 root、Linux 按唯一被观察目录计数，不预分配该上限。
7. 相同 canonical root 只建一个 native subscription；嵌套/重叠 roots 向每个匹配 target 独立发事件，batch
   的既有 `(target,path)` key 去重。direct/exact 只接收 immediate child，tree 接收完整相对路径。
8. recursive root 本身若是 symlink/junction 则 `FailedPrecondition`；遍历按 `platform_fs` entry type 不跟随
   descendant link。direct/exact 的既有 alias/canonicalization 行为不变。

## 能力与形状审计

下表已从 consumer journey、三平台机制和现有公开入口双向核对；实现只处理列出的 `FIX`。

| 能力族 | 层级 | 准入旅程/来源 | 语义与生命周期 | owner/分解 | 当前路线 | 覆盖 | 动作 |
|---|---|---|---|---|---|---|---|
| explicit recursive target | 原子机制 | 推理 bundle + shader tree；libuv recursive flag | `directory_tree(path)`，每 target 显式，direct 默认不变 | io_watch target kind | 只有 file/direct directory | MISSING | FIX `IOW-R01` |
| root-relative descendant event | 原子机制 | 两条纵切都按缓存 key 消费 | borrowed view 到下次 `next/close`；每 target coalesce | backend path → 公共 bounded batch | backend 主动过滤 descendants | MISSING | FIX `IOW-R01--R03` |
| native subtree mechanism | 原子机制 | Win32/libuv、FSEvents、inotify | one-shot/rearm、overflow、root loss | io_core 只拥有 Windows submit；io_watch 拥有语义 | Windows seam 固定 false；mac 过滤；Linux 单 root watch | PARTIAL | FIX；io_core v3 独立重开 |
| dynamic subtree coverage | 复合功能 | 新/移入/rename 目录后继续失效 | topology gap 必须先交 rescan，失败随后 typed terminal | Windows/mac native；Linux root-first rebuild | Linux 只有 root watch | MISSING | FIX `IOW-R02--R03` |
| bounded native resources | 原子机制 | inotify user-watch limit 与服务资源预算 | 非零显式 cap；不预分配；超限 ResourceExhausted | `max_watched_directories` | 无 tree capacity | MISSING | FIX `IOW-R01/R03` |
| loss/resource signalling | 原子机制 | 三 OS overflow/coarse/path-loss | 先发布 rescan batch，再在不可恢复时 terminal error | 复用既有双 bank/failure 优先级 | native overflow → rescan；无恢复 | PARTIAL | FIX `IOW-R02--R03` |
| lifecycle/backpressure | 复合功能 | 既有用户 journey | bounded batch、single owner、one next、cancel/close/drain | 复用 sched/io_core/mem | COMPLETE | COMPLETE | KEEP |
| snapshot/debounce/reload | 应用策略 | 模型/渲染发布策略不同 | 应用拥有扫描与发布 | 不属于通用 change stream | 无 | OUT_OF_SCOPE | REJECT |

### 有限改动集

| ID | 问题与源码证据 | 唯一允许改动 | 验收 |
|---|---|---|---|
| `IOW-R01` | 公开 target 只有 exact/direct；common router 只接受 basename | 新 kind/factory、native cap、root tree chain、root-relative router、symlink/root/option 校验及 API tests | 混合/重复/重叠 targets，bounded/OOM/UTF-8/长路径，direct 无行为漂移 |
| `IOW-R02` | Windows subtree 位固定 false；macOS 在 callback 主动丢弃 deeper descendant | 调用已独立审计的 io_core v3 bool；Windows 预分配转换 scratch 并 `/` 归一化；macOS 对所有 ancestor roots 路由 | 深树、新目录/rename 后后代、coarse/overflow/root loss、cancel/close |
| `IOW-R03` | Linux 一 root 一 wd，无法覆盖后代或重叠 roots | 一个 inotify fd；root-first 枚举唯一目录、wd/path O(1) map、O(D+path bytes) 有界状态；目录拓扑/overflow 后发 rescan 并重建 active roots | 深树/空树/重叠、动态 topology、cap/OOM、link 不跟随、无 busy poll/泄漏 |

不允许第四个产品批次；结构恢复需要新策略时退回 `SCOPED`，不能边测试边扩张。

### API / usability freeze

```cpp
std::array targets{
    hyle::io::FileWatchTarget::file("model/config.json"),
    hyle::io::FileWatchTarget::directory("certs"),
    hyle::io::FileWatchTarget::directory_tree("model")};
HYLE_ASSIGN_OR_RETURN(auto watcher,
    hyle::io::FileWatcher::create(alloc, io, owner, targets,
        {.max_watched_directories = 65536}));
// 建立后扫描 authoritative tree；每批消费 target-root-relative path。
// batch.rescan_required() 时先重扫，再继续 next()。
```

- exact path 可在 create 时不存在；direct/tree root 必须是现存真实目录，tree root 不得是 link。
- watcher、`next()`、`close()` 和析构仍由一个 owner worker 管理；同一时刻只有一个 `next()`；view 借到
  下一次 `next/close`。create 后先扫描是跨平台无丢失状态收敛合同，不承诺 event log 完整。
- 普通事件不 allocation；写入预留 batch path/event/index，溢出转 rescan。Windows/macOS 每唯一 root 一个
  原生递归 owner（macOS 是单 stream 中的 root）；Linux 一个 fd、每唯一目录一个 wd 与一个 owned path。
- Linux 初建 O(D+path bytes)；普通 event 为 O(1) wd lookup + O(R) root-prefix routing + O(path copy)，其中
  `R` 是唯一 roots；目录拓扑变化/overflow 是已显式要求 consumer 全树重扫的恢复路径，内部同步 O(D)
  重建，不把它伪装为常态热路径。Windows/macOS 普通 dispatch 同样 O(R) 最坏，root-local Windows 为 O(1)。
- `max_watched_directories` 是资源上限而非 reserve；触顶、native watch limit 或 OOM 明确
  `ResourceExhausted`。可恢复 loss 先产生 rescan；无法保持完整后续覆盖时下一次 `next()` terminal failure。
- 不提供 polling fallback、后台 reload、debounce 或 symlink traversal；这些限制写入 user guide。

## 源码时序与 P0 成本预审

| 路线 | syscall / owner | 常态 allocation / lock / copy | 复杂度与裁决 |
|---|---|---|---|
| Windows direct/tree | 每 root 一个 overlapped `ReadDirectoryChangesW(FALSE/TRUE)`，同 IOCP pump | create 时 buffer+UTF 转换 scratch；每 native batch 一次既有 SpinLock，事件只拷贝进预留 bank | subtree 是 syscall 语义而非第二实现；普通路径无新增 submit/wake |
| macOS direct/tree | 一个 FSEventStream + serial dispatch owner | callback 内既有锁；absolute path 只做 view/prefix 与 bounded bank copy | OS 本来递归；删除旧 deeper filter，不新增 stream/thread/allocation |
| Linux direct/tree | 一个 inotify fd；每唯一 directory 一个 wd；同一 async read pump | setup/recovery 分配 O(D+path bytes)；普通 event 不分配，锁内解析并拷贝有界路径 | setup O(D)，普通 O(1) lookup+O(R) route；topology/loss O(D) 与已要求的 consumer rescan 同阶 |

源码已把性能问题分成机制事实而不是点值：Windows 固定 `FALSE` 是明确能力缺陷，Linux 缺 O(D) watch set
是明确算法缺陷，macOS descendant filter 是明确路由缺陷。没有需要用计时猜测的剩余机制，故本轮 P1
`NOT_APPLICABLE`；测试只证明正确性、idle 无 busy poll、资源界和 default direct 路径没有复制 owner。

## 七问

1. 无产品第三方；libuv `3e606bc30a84` 与 Seastar `78f64e2317b1` 仅作 peer 证据，HYLE 直接用三 OS API。
2. 已覆盖显式 tree、相对路径、动态 topology、loss/cap 与完整 owner journey；snapshot/debounce/reload 是具名应用策略。
3. 复用 io_core IOCP/read、platform_fs 无分配 entry type、sched cancel/join、mem allocator 与 container bounded maps；不建第二 runtime。
4. P0 如上；没有苹果对橘子的计时，也不拿 direct-child 与 recursive 比速度。
5. 旧 C/Native FFI 不变；异步 view/owner 暂不承诺跨语言 runtime ABI。
6. 仍是独立 `io_watch` factor 并由 `hyle::io` 聚合；依赖图不变，io_core v3 只扩 detail seam；三交付需实测。
7. factory 名称直接表达三种 target；guide 必须写 create→scan→consume→rescan→close、path lifetime、cap、link 和 network-FS 限制。

## 验证、红队与裁决

产品实现只允许 `IOW-R01--R03`；任何未运行项仍为 `UNPROVEN`。
