# Container v6：Container Text 身份与 P1 证据恢复

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（`SCOPED`）

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
  oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f`；Unicode mechanism 为 simdutf
  `b91b70d7d7bf8609eec2612971c01f3c92499a4b`。每个 cell 只有一个 peer；
- tier / machine / instrument / margin：P1；W1-WIN primary，W1-WSL/M1-MAC correctness/delivery；
  `closure-perf-v1`；5%；
- budget：identity/audit 45 分钟，迁移与 build 45 分钟，P1 90 分钟，三平台/交付 90 分钟，红队
  30 分钟；性能最多一次正式 qualification、一次机制检查、一次内部产品 correction 和一次重跑；
- stop：任一 comparison 仍非同合同、qualification 无 verdict、material regression 无法在一次机制检查/
  correction 内解决，立即以具名 blocker 终止；不得扩建第五个 workload、修改公共仪器或用有利
  natural-route 点值覆盖失败。

## 冻结 normal workloads 与语义矩阵草案

四条 primary workload、输入规模、线程数、operation mix、计时 phase、双顺序/placement 与 checksum
沿用前轮 immutable contract；审计只允许修 adapter/metadata 中被红队证明不等价的格。采样前必须把
本节从草案签为 `FROZEN`。

| Cell | comparison class | candidate minimum contract | peer adapter 必须补齐 | 不计入 steady phase |
|---|---|---|---|---|
| entry+weight LRU hit / insert+evict | `SAME_CONTRACT` | recoverable allocation result、失败不改变 cache、相同 touch/evict/checksum | peer allocation failure 转为同一三态；异常边界 rollback；相同 allocator/resource 或预分配资源合同 | object/resource setup、input generation、final checksum |
| read-mostly membership | `SAME_CONTRACT` | concurrent contains/insert/erase result、相同 key/shard/load 生命周期 | 两侧注入同一可并发 resource/allocator；相同 reserve/setup 与 teardown | table/resource construction、thread creation、final verification |
| unbounded typed MPMC | `SAME_CONTRACT` | try push/pop、recoverable OOM、FIFO conservation、相同 producer/consumer participation | oneTBB throwing push 在 adapter 捕获并映射；两侧使用同一 resource ownership；相同 setup/token 和 drain | queue/resource construction、worker creation、final conservation |
| safe UTF-8→UTF-16 caller buffer | `SAME_CONTRACT` | validate、typed status/input offset/output count、exact caller buffer、相同 valid/error mix | simdutf peer 返回同构 adapter result，不终止；相同 size/validation/conversion passes 与 output capacity | corpus construction、buffer allocation、final checksum |

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

