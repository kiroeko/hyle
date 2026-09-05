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
| heterogeneous cache class facts | base Platform 的 allocation-free buffer-fill snapshot | `MUST`；保留现有全局 conservative convenience |
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
