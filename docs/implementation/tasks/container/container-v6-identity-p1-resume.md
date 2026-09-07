# Container v6：Container Text 身份与 P1 证据恢复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`AUDITED`）

- components / owner / rounds：
  - `container` / Container / `container-v6-identity-p1-resume`；
  - `container_concurrent` / Container / `container-concurrent-v2-p1-resume`；
  - `container_queue` / Container / `container-queue-v2-p1-resume`；
  - `text -> container_text` / Container / `container-text-v2-identity-p1-resume`；
- task kind：`RESUME`；前轮为 `container-v5-product-factorization`、
  `container-concurrent-v1-factor`、`container-queue-v1-factor`、`text-v1-factor`；
- baseline：`d04710a33319e4f1088d81608bc5a7bd031b8eef`。产品实现仍是前轮第三候选
  `6fae3c11ebce9a0741f45780e63c9896e08bb2a2` / tree
  `8025ee111380c2600f804cf979ed59d289a8be76`；其 correctness/delivery 证据只在产品边界未变时复用；
- previous blocker：LRU、concurrent membership、unbounded MPMC、Unicode 四个 P1 cell 没有先等价化
  allocator、OOM、rollback、result 等合同，旧数字只是 `NATURAL_ROUTE` diagnosis；
- previous resume condition：逐 cell 冻结 `EQUAL/ADAPTER_EQUALIZED/INTENTIONALLY_DIFFERENT` 矩阵，
  用具名 adapter 真正等价后在 immutable candidate、W1-WIN、5% margin 下取得 qualified verdict；
- new owner decision `CONTAINER-TEXT-IDENTITY-01`：内部 factor 的真实 owner 是 Container，预发布时将
  `text/hyle_text` 完整迁移为 `container_text/hyle_container_text`，不保留 compatibility alias；受控
  registry 迁移合同见
  [`Container Text factor 身份迁移`](../build/container-text-factor-identity-migration-v1.md)；
- scope：迁移内部 factor/component/target/source/test/package detail 与所有直接边；保持公开
  `hyle/container/*`、`hyle::container` 和用户 `hyle::container` façade；修正四条 benchmark adapter 和
  comparison metadata；重放受影响 correctness、direct consumers、source/static/shared、symbols、docs、
  P1 与独立红队；
- non-goals：新 Container 能力、公开 API/layout/算法/默认 allocator、常态 metadata、推理策略、
  tokenizer/KV/scheduler/IO service、Native FFI/C API、稳定 C++ ABI、公共性能仪器或 margin 修改、Formal；
- direct dependencies：sequential `PUBLIC core,mem,container_intrusive / PRIVATE platform`；concurrent
  `PUBLIC core,mem,mem_reclaim,container`；queue `PUBLIC core,mem`；container_text
  `PUBLIC core,mem,container / PRIVATE platform`；
- direct consumers：Container façade；`util -> container_text`；`io -> container_text`；Sched 私有使用
  `container/container_queue`；package sample 与四 factor tests；
- frozen inference demand identities：llama.cpp
  `3ad1ba7336986d98592d3e28cafd1a406715351f`、standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`、ds4
  `9ab705347c1775e7599ede7eb81a6255ec7dccb5`、vLLM
  `52358e6e192aeae73dd3764046c98fc4b156ea83`、SGLang
  `28457f0dcab4ccf748d60f2387a1a4a7fdb5a110`。本轮只确认前轮 capability map 没有因身份迁移改变，
  不重新求 API 并集；
- strong peers：LRU 为 Abseil `38f8c1535de9ebc12c6340e5c1c8ebd72afde560`；membership/queue 为
  oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；Unicode mechanism 为 HYLE dependency
  lock 对应的 simdutf `2d4e249dcfd978b39dd567b156f20a43db97377b`。每个 cell 只有一个 peer；
- tier / machine / instrument / margin：P1；W1-WIN primary，W1-WSL/M1-MAC correctness/delivery；
  `closure-perf-v1`；5%；
- budget：identity/audit 45 分钟，迁移与 build 45 分钟，P1 90 分钟，三平台/交付 90 分钟，红队
  30 分钟；性能最多一次正式 qualification、一次机制检查、一次内部产品 correction 和一次重跑；
- stop：任一 comparison 仍非同合同、qualification 无 verdict、material regression 无法在一次机制检查/
  correction 内解决，立即以具名 blocker 终止；不得扩建第五个 workload、修改公共仪器或用有利
  natural-route 点值覆盖失败。

## 冻结 normal workloads 与语义矩阵（`FROZEN`）

四条 primary workload、输入规模、线程数、operation mix、计时 phase、双顺序/placement 与 checksum
沿用前轮 workload identity；本轮只修 adapter/metadata 中被红队证明不等价的格。以下矩阵于
2026-09-07 冻结；任何输入规模、operation mix、线程角色、计时 phase、产品 API 或产品算法变化都必须
退回 `SCOPED/AUDITED`，不得用新的有利姿势替代旧 blocker。

| Cell | comparison class | candidate minimum contract | peer adapter 必须补齐 | 不计入 steady phase |
|---|---|---|---|---|
| entry+weight LRU hit / insert+evict | `ADAPTER_EQUALIZED` | recoverable allocation result、失败不改变 cache、相同 touch/evict/checksum | 两侧使用 `SystemAllocator` 路由与同一 `StableHash`；peer 只捕获 `bad_alloc`，在 index failure 后回滚 list，并返回与 HYLE 同构的 stored/OOM/evicted 结果 | object/resource setup、reserve、input generation、final checksum |
| read-mostly membership | `ADAPTER_EQUALIZED` | concurrent contains/insert/erase result、相同 key/load 生命周期 | HYLE 与 oneTBB set 都使用同一 `StableHash`、equality 与 `SystemStdAllocator`；reserve/populate/warmup 相同且在计时外 | table/resource construction、thread creation、final verification |
| unbounded typed MPMC | `ADAPTER_EQUALIZED` | try push/pop、recoverable OOM、FIFO conservation、相同 producer/consumer participation | 两侧使用 `SystemStdAllocator`/`SystemAllocator` 的同一底层 owner；oneTBB throwing push 只捕获 `bad_alloc` 并映射为 false；相同 setup/warmup/drain | queue/resource construction、worker creation、final conservation |
| safe UTF-8→UTF-16 caller buffer | `ADAPTER_EQUALIZED` | validate、typed status/input offset/output count、exact caller buffer、相同 valid/error mix | peer 固定到 vendored simdutf 同一 revision，返回同构 status/input offset/output count 而不 abort；validator 覆盖 malformed 与 output-too-small；steady valid corpus 仍执行相同三趟 safe route | corpus construction、buffer allocation、final checksum |

若 peer 的 allocator template 或 API 使同一 resource 无法合法注入，该格必须写
`INTENTIONALLY_DIFFERENT` 并把 cell 降为 `NATURAL_ROUTE`；此时本轮没有预冻结独立 consumer budget，
工程 P1 不能因此通过。

## Capability-family carry-forward

前轮已经把四组推理需求和成熟 peer 差异映射为 sequential sequences/maps/hash/bitset/LRU、concurrent
map/set/vector、role-specific/bounded/unbounded queues、borrowed text/path/interner/Unicode。所有已准入
`MUST` 在第三候选为 `COMPLETE` 或完整 `ROUTED`，终局红队没有发现新缺口。本轮身份更名不改变
semantic owner、公开 surface、生命周期、复杂度或负空间；若审计发现任何新的 `PARTIAL/MISSING`，必须
在进入 `AUDITED` 前明确加入有限 FIX，否则本轮只保留：

| ID | class | coverage | action | verification |
|---|---|---|---|---|
| `CONT6-F01` | build/product identity | `PARTIAL`：factor owner 正确但内部 ID/target 过宽 | `FIX`：完整迁移为 `container_text`，无 alias | component graph、headers、direct consumers、三交付、symbols |
| `CONT6-F02` | P1 comparison contract | `PARTIAL`：产品完整，peer adapter/metadata 不等价 | `FIX`：按四行矩阵等价化；不改产品 API | independent pre-sample review、qualification、manifest |

当前没有第三个产品 FIX 授权。

## Audit verdict（`AUDITED`）

Top-down public trace 与 bottom-up ownership scan 得到以下冻结结论：

- `path.h`、`string_algo.h`、`string_interner.h`、`string_view.h` 的公开 include 与全部类型仍位于
  `hyle/container`、`hyle::container`；错误只存在于内部 factor/target/directory/package identity。更名
  不准移动公开头、改 namespace、API、layout、算法或 provider；
- 真实直接消费者是 Container façade、`util` 的 Registry 和 `io` 的 HTTP/TLS surface。中央图必须把
  `util/io -> text` 改为 `util/io -> container_text`；其他源码消费者只通过公开头或 façade，不新增边；
- 当前 LRU peer 已使用相同 hash 与 System allocator，但其 throwing insert 缺 OOM 映射和 index-failure
  rollback；membership peer 还使用不同 hash/allocator；queue peer 使用 oneTBB 默认 allocator 且把
  throwing push 无条件当成功；Unicode peer 终止而非返回同构 typed result，且 task 一度错误记录了
  本地 simdutf HEAD 而非产品 lock revision。这些都是比较器合同缺陷，不是产品缺陷；
- frozen workloads 的输入、角色、计时 phase 和产品实现不变。`CONT6-F02` 只允许 benchmark-only adapter、
  新版本 contract/shape metadata 与 validator；不得以异常处理、allocator 或 hash 差异继续做
  `SAME_CONTRACT` 声明，也不得因数字修改产品 API/layout/metadata；
- factor identity 执行完整一对一迁移：`text -> container_text`、`hyle_text -> hyle_container_text`、
  `src/text -> src/container_text`、`tests/text -> tests/container_text`、`HYLE_TEXT_* ->
  HYLE_CONTAINER_TEXT_*`、`text_simdutf -> container_text_simdutf`。不保留 alias、旧目录、旧构建开关或
  当前性文档引用；历史 task/evidence 仍按当时事实保留，并由迁移 receipt 连接；
- capability carry-forward 没有新的 `MUST/PARTIAL/MISSING`，因此有限 FIX 集仍严格只有
  `CONT6-F01` 和 `CONT6-F02`。任何产品能力、实现或公开 surface 变化都是新的 reopen trigger。

Audit verdict 为 **PASS**，四个 factor 进入 `AUDITED`。`SCOPED -> AUDITED` 前态 registry 快照位于
`C:\public\project-hyle\_validation\container-v6\transitions\module-status-before-audit.json`，SHA-256
`439AC47F594ED0B26963981C2BDE5132E3CA65F1354D976ECC179F54AE1DA4E4`。

## `CONT6-F01` implementation receipt

`43105dd5` 完成无 alias 的 `container_text` component/target/source/test/macro/provider/package 迁移，
并同步 Container façade、`util`、`io`、roadmap、build/container architecture 和用户指南。W1-WIN static
selected closure `36/36`、shared selected closure `35/35` 通过；旧 factor 选择明确失败；Container
source/shared/static relocated consumer 全部通过。公开 `<hyle/container/**>`、`hyle::container`、算法和
layout 没有变化。`CONT6-F01=PASS`；活动状态保持 `AUDITED`，只剩 `CONT6-F02` 与其后声明的回放门。
