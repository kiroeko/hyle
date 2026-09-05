# Platform v7：异构缓存事实与同步随机访问文件

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮只处理一次可复现的异构 CPU 事实丢失和三个 already-open file owner 缺口。
> `SCOPED/AUDITED` 前不修改产品源码；历史 Platform task 只作证据，不构成继续施工授权。

## Reopen evidence（`REOPENED`）

- baseline：`984eab5da3cae715984fd3a68a8e9299e23c0045`；该提交相对 Platform v6
  candidate `246bd182042f02c2b37457c21cea4ceebd8739c7` 没有 Platform 产品路径变化；
- trigger ID：`PLATFORM-HETERO-CACHE-RANDOM-ACCESS-2026-09-05`；
- M2-MAC（Apple M4、macOS 26.6.2）复现：全局 `hw.l2cachesize` 为 4 MiB，而
  `hw.perflevel0.l2cachesize` 为 16 MiB、`hw.perflevel1.l2cachesize` 为 4 MiB；L1I/L1D
  也分别为 192/128 KiB 与 128/64 KiB。现有 `cache_info()` 只返回一组全局值，实际把异构 class
  压扁并丢失 performance-core cache facts；这击中 v6 的 host-fact reopen trigger；
- 推理消费：强核选择之后的 tile、working-set 和 worker policy 需要查询对应 CPU class 的 cache
  capacity，不能把全局最小/默认值假装成每个 class 的事实；
- 当前 `FileHandle` 是唯一 owner，但 `FileMapping` 会消费它。调用者若要保留独立 owner，没有 portable
  duplicate；若用 path `stat()` 查询已打开对象，会重新解析名字并产生 rename/replace TOCTOU；
- 当前同步文件接口只有 shared-cursor read/write。模型、索引和 shard reader 的显式 offset 访问被迫进入
  coroutine IO，或自己写 `pread`/Windows offset I/O。Folly `File::dup` 与 `pread/pwrite`、Rust
  `FileExt` 证明这两类形状成熟；llama.cpp 的 model-load 路径直接对 opened descriptor 使用
  `fstat`/offset-oriented file controls；
- Windows 的同步 offset `ReadFile` 会更新并串行化该 handle 的 file pointer，而 POSIX `pread` 不移动
  cursor。因此不能给通用 single-cursor `FileHandle` 添加一个虚假的跨平台“cursor-independent”方法；
  需要独立 owner type，使内部 cursor 不可观察，并把并发/async 能力继续留给 IO。

### 初步能力边界

| family | natural owner / shape | disposition |
|---|---|---|
| heterogeneous cache class facts | base Platform 的 caller-buffered snapshot | `MUST`；保留现有全局 coarse convenience |
| owner duplication | `FileHandle::duplicate()` | `MUST`；新 owner，共享 native open-file cursor/lock state 的 OS 语义须显式说明 |
| opened-object metadata | `FileHandle::info()` | `MUST`；查询打开对象，不重新解析 path |
| synchronous explicit-offset file | 独立 move-only random-access owner + exact/all helpers | `MUST`；禁止 append，接口不暴露 cursor |
| direct/vectored/async IO | IO owner | `ROUTED`；本轮不吸收 |
| mapping/advice/residency | Mem owner | `ROUTED`；本轮不吸收 |
| filesystem capacity | `std::filesystem::space` | `REUSE`；不制造 HYLE 镜像 |
| cache associativity/complete CPU-to-cache membership | OS 不稳定或不可得的事实 | `DEFER`；不猜测、不伪造 |

## 初步性能与比较合同

- tier：`P0 / closure-correctness-v1`。新 cache snapshot 是显式冷路径 OS discovery；公开 buffer-fill 不
  要求 heap。既有 `cache_info()` 保持一次初始化、随后 O(1) cached query；
- duplicate、handle info、一次 offset read/write 都只允许最少 native controls；exact/all helper 仅按
  short transfer 前进，复杂度 O(bytes transferred / native transfer)；
- 不给 `FileHandle` 增加 mode/cache metadata，不让未调用新 API 的 read/write/seek/open 热路径新增分支；
- 只有 owner、cursor、blocking、offset、short-transfer、EOF 和 durability 语义全部相同才允许计时。
  `read` 对 `pread`、POSIX 可并发 pread 对 Windows 同步 handle、coroutine IO 对 blocking owner 均是
  `NOT_COMPARABLE`，不得做苹果对橘子排名。

## Source-archive 邻接发现

在 M2-MAC 从 commit archive 配置完整测试时，28 项中只有 `hyle_loc_report_test` 失败：测试无条件要求
源树存在 `.git`，而同一构建系统已经明确允许源码归档且不生成 `hyle_source_package` target。这是独立
Build/package test-selection 缺陷，不属于 Platform 产品 candidate；另立 task 修复并在归档上回放。

## 下一状态入口

只有在自然 API、失败语义、跨平台差异、有限 FIX、依赖/交付和验证矩阵冻结后进入 `SCOPED`；只有完成
implementation/peer/build/doc 红队并冻结 FIX 后进入 `AUDITED`。

## Closure card（`SCOPED`）

- component / product / round / kind：`platform + platform_fs` / Platform /
  `platform-v7-cache-class-and-random-access` / `REOPEN`；`platform_thread` 保持只读；
- baseline / trigger：`984eab5da3cae715984fd3a68a8e9299e23c0045` /
  `PLATFORM-HETERO-CACHE-RANDOM-ACCESS-2026-09-05`；
- locked peer snapshots：Folly `65749da412d92fe27d87ae45a8ef670720712858`、llama.cpp
  `4d9176092d00586775af140581bb0b558ddc4389`、ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`；三者均已 fetch 并确认本地 HEAD 等于 upstream；
- authoritative OS contracts：Apple “Determining system capabilities” 明确 `hw.nperflevels`、
  `hw.perflevelN.*` 且较小 N 表示更强 class，并明确全局 cache key 等价于最低性能 class；Windows
  `GetLogicalProcessorInformationEx(RelationCache)` 提供共享 cache 关系；Linux 使用稳定 sysfs
  `cpu*/cache/index*` 事实；
- normal consumers：推理 worker/cache blocking 在 CPU class 粒度取得容量与共享宽度；模型、权重、
  cache/index 通过 already-open owner 做 identity-stable metadata、owner cloning 和 caller-synchronized
  blocking explicit-offset transfer；
- public delivery：仅 C++23 source/static/shared；无 C/Native FFI、稳定 C++ ABI 或兼容层。

### 冻结 API 与语义

1. 新增 `CpuCacheClassInfo`：`performance_rank`、`CpuKind`、当前 process view 的
   `logical_cpus/physical_cores`、`cache_line_size`、`logical_cpus_per_l2/l3` 与 `CacheInfo`；未知事实为 0。
   `available_cpu_cache_classes(dst, cap)` 使用现有 buffer-fill 合同，返回完整记录数并按已知 rank 从强到弱
   排列；同 rank 但 cache 容量/共享宽度不同的记录不得合并。便利头提供同名 `vector` overload。
2. `cache_info()` 保留 cached coarse convenience；在 Apple 上它按官方合同是最低性能 class，不再被文档
   描述为整机每个 CPU 的 cache。需要调度/tiling 的调用者必须选择 class snapshot；不把 class 事实变成
   SIMD/provider winner。
3. `FileHandle::duplicate() const -> Result<FileHandle>` 创建独立 RAII owner。POSIX `dup`/`fcntl` 与
   Windows `DuplicateHandle` 都共享底层 open-file cursor/lock state；duplicate 不是独立 file position。
4. `FileHandle::info() const -> Result<FileInfo>` 对 already-open object 查询，不解析 path；`is_symlink=false`
   （handle 指向打开后的 object），类型、size、mtime 使用与 path `stat` 相同值域。
5. 新增 move-only、一字 owner `RandomAccessFile` 与 `open_random_access_file(path, flags)`；禁止
   `kOpenAppend`。它暴露 `read_at/write_at`、span overload、`read_exact_at/write_all_at`，以及完整 owner
   生命周期和 already-open controls：`valid/get/close/duplicate/info/size/resize/preallocate/sync/lock/
   try_lock/unlock`。不暴露 seek/position；raw handle 只作借用 escape hatch。
6. POSIX 使用 `pread/pwrite`，不移动 native cursor；Windows 使用同步 handle 的 offset `OVERLAPPED`
   `ReadFile/WriteFile`，其 native file pointer 可能移动，但该状态不在类型合同中可观察。每个 owner 及其
   duplicate aliases 均须由调用者同步；真正 concurrent/async/direct/vectored 访问继续归 IO，独立重新
   open 的 random-access owners 可由应用分配给不同 worker。

### Ownership、依赖与性能

- `platform` 继续只公开依赖 Core/Error；`platform_fs` 继续公开依赖 Core/Error/Time、私有依赖
  Platform；全部失败继续走共享 `Status/Result` 与 native-error adapters，不引入 Container/Mem/第三方；
- `FileHandle` layout 与旧 open/read/write/seek/sync 控制流不变；`RandomAccessFile` 以组合复用
  `FileHandle`，不保存 offset、mode、mutex 或 heap；
- tier 固定 `P0 / STRUCTURAL`：cache-class discovery 是显式动态 snapshot，复杂度为 OS topology/cache
  输入规模；common small-host scratch 有界，必要时失败可见地分配。duplicate/info/一次 at-transfer 各为
  固定次数 native control；exact/all 只按 short transfer 前进，不自造 payload copy；
- 性能比较硬门要求 owner、blocking、offset、EOF、short transfer、cursor observability、并发和 durability
  全同。不同项一律 `NOT_COMPARABLE`；不得为了点值给常态对象增加 metadata 或把 IO 能力吸入 Platform。

### 预算、停止条件与预声明验证

- finite FIX budget：`P7-CPU-01` class snapshot；`P7-FS-01` duplicate；`P7-FS-02` handle info；
  `P7-FS-03` random-access owner；`P7-T01` 三平台/失败语义测试；`P7-B01` header/exact factor/三交付；
  `P7-D01` architecture/user/capability/performance evidence；`P7-R01` Mem/Sched/IO 只读回放；
- correctness：W1-WIN MSVC Release/Werror；W1-WSL GCC Release/ASan/UBSan；M2-MAC AppleClang
  Release/Werror。覆盖 count/truncation/order/unknown、M4 双 class、closed/invalid/null/overflow/EOF/
  short transfer、offset 不依赖公开 cursor、duplicate lifetime/shared cursor、metadata path replacement；
- delivery：exact `platform`/`platform_fs` factors；Platform source embedding、relocated static/shared SDK；
  package sample 直接消费新增 API；header self-contained；
- architecture/red team：检查 no dependency drift、无 C symbols、Apple least-class trap、Windows cursor trap、
  duplicate aliasing、append、symlink/rename identity、special-file metadata、offset overflow、concurrent misuse；
- performance：旧 hot-path baseline diff、layout static assertions、新入口 syscall/code-shape 审计；只有发现
  HYLE 自造 material work 才升级有限 P1，不扩建公共仪器；
- stop：若 supported OS 无法诚实表达 frozen contract、需要常态 lock/heap/metadata、或必须修改 frozen
  Thread/Mem/IO 产品，退回 `AUDITED`/`BLOCKED`；不得用猜测填 unknown。

本卡已冻结产品边界。下一步只允许只读实现/peer/consumer 审计并形成上述有限 FIX 的具体处置；在
`AUDITED` 前仍不得修改产品源码。

## Audit 与冻结 FIX ledger（`AUDITED`）

### Public-surface / consumer trace

- `cpu.h -> cpu.cc`：`CacheInfo/cache_info()` 只有一份 cached value；Apple 官方合同证明它等价于最低性能
  class。`CpuInfo` 已有 rank/kind 与 cache-domain identity，却没有 class cache capacity/sharing facts；
  SIMD 只读取 feature facts，新增 snapshot 不改变 target authority；
- Windows 已在同一 baseline 中正确解析 variable-size `RelationAll` 与 CPU Sets；本轮只从
  `RelationCache` 提取 size/type/line/share mask，并与 process-available CPU/rank 相交；Linux 从既有
  `cpu_capacity`、topology 路径旁的 stable cache sysfs 读取；macOS 直接使用官方 `hw.nperflevels` 与
  `hw.perflevelN.*`，不伪造 logical-id affinity mapping；
- `fs.h -> fs.cc`：`FileHandle` 已是 move-only one-word owner，所有控制都落在 raw helper；duplicate/info
  可沿同一 seam 添加而不触碰旧 read/write。path `stat` 已有完整 `FileInfo` 转换逻辑，可与 handle query
  共享但不能二次解析 path；
- Folly 当前 `File::dup/dupCloseOnExec` 和 `pread/pwrite` 证明 owner/offset family；其 free-function shape
  不要求 HYLE 复制，因为 HYLE 的 structured Error 与 RAII owner 更适合 member API。Rust Windows
  `FileExt` 与 Microsoft 原生合同证明 synchronous offset call 会影响 native file pointer，支持独立 type；
- llama.cpp 当前 model loader 以 opened fd 做 `fstat`，并有分片/offset 读取；ggml 不提供更低层的
  portable owner。新增能力服务普通文件消费者和推理，不把某条 benchmark 变成 API；
- Mem `FileMapping` 消费 generic `FileHandle`；它无需改动。需要从 random access 转 mapping 的调用者应在
 设计时选择 owner，不增加双向转换；Sched/IO 也不需要新 public 类型。

### Capability / gap ledger

| ID | requirement | baseline | frozen disposition |
|---|---|---|---|
| `P7-CPU-01` | 不能丢失异构 class cache 容量与共享宽度 | `MISSING` | `FIX`：新增 snapshot；unknown 为 0，不改 target dispatch |
| `P7-FS-01` | 保留同一打开对象的第二个 owner | `MISSING` | `FIX`：native duplicate，non-inheritable；明确 shared open-file state |
| `P7-FS-02` | metadata 查询绑定 opened identity | `MISSING` | `FIX`：handle `info()`，不回退 path；symlink bit 为 false |
| `P7-FS-03` | caller-synchronized blocking explicit-offset transfer | `MISSING` | `FIX`：独立 `RandomAccessFile`；无 seek、无常态 metadata |
| `P7-T01` | 正确性与跨平台 failure contract | `PARTIAL` | `FIX`：新增 unit/integration tests 与 M4 双 class 真实断言 |
| `P7-B01` | factor/header/source/static/shared delivery | `PARTIAL` | `FIX`：exact roots、sample 与三交付回放 |
| `P7-D01` | architecture/user/current capability/performance 真相 | `STALE` | `FIX`：选择表、陷阱、owner/复杂度/NOT_COMPARABLE 同步 |
| `P7-R01` | frozen direct consumers 不被非法修改 | `UNPROVEN` | `FIX`：Mem/Sched/IO build-only replay |
| filesystem space/watch/direct/vectored/async/mapping | 相邻或标准能力 | `ROUTED` | `KEEP`：分别 std、IO、Mem；不进入本轮 |
| associativity、macOS per-id class mapping | 不稳定/不可得事实 | `UNAVAILABLE` | `KEEP`：返回 unknown 或不建字段，不做推断 |

### 实现约束与红队预判

- cache profile key 是 rank/kind + capacities + line/share widths；同 key 聚合 count，不同 key 不合并。
  Windows/Linux common <=256 CPU 使用 stack scratch，较大 process view 允许一次可失败 heap；排序
  O(P log P)，OS cache enumeration O(P*C) 或 O(relation bytes)，不在 warm fact query 中发生；
- Apple profile index 由官方定义决定 rank，`hw.nperflevels` 缺失时诚实回退一条 unknown/coarse record；
- POSIX duplicate 首选 `F_DUPFD_CLOEXEC`，fallback 必须在返回前设置 close-on-exec；Windows
  `DuplicateHandle(..., FALSE, DUPLICATE_SAME_ACCESS)`。公共承诺是同一 opened object 与 shared cursor；
  duplicate aliases 不能作为独立 lock contender，具体 lock association 保留 native 语义；
- handle info 在 Windows 区分 disk/directory/character/pipe，在 POSIX 映射完整 `st_mode`；regular size 与
  signed timestamp 做 checked conversion。rename/unlink 后仍查询 original opened object；
- `RandomAccessFile` 内部只含一个 `FileHandle`。POSIX `pread/pwrite`；Windows 同步 `ReadFile/WriteFile`
  传 offset `OVERLAPPED`，EOF 的 `ERROR_HANDLE_EOF` 映射 0。offset 超过 signed native 64-bit 为
  `OutOfRange`；exact/all 的 offset addition checked；append open 在 syscall 前拒绝；
- basic at-transfer 保留 short success，exact EOF 为 `DataLoss`，zero-progress write 为 `DataLoss`；对象及
  duplicate aliases caller-synchronized。若测试要求并发，必须路由 IO 或独立重新打开 owner，不能添加
  mutex/event/cache 来迎合测试。

只读审计没有发现需要新增 factor、内部依赖、第三方包、常态字段或额外 public family。有限 FIX 集冻结为
上表八项；现在允许实现，任何公共合同变化必须退回本节重新裁决。
