# Container v3：并发 node map 的旧表寿命

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（CANDIDATE）

- component / owner / round：`container` / Container / `container-v3-node-table-reclamation`；
- task kind：`REOPEN`；baseline：`444b02c01afe00f4cd3b44433204e58db3c24712`；
- trigger ID：`CONTAINER-NODE-TABLE-RECLAIM-2026-09-03`；trigger evidence：本任务的 Reopen
  trigger 与 `mem-reclaim-v2-inference-resume`；
- demand：长寿命、持续增长的模型/backend/request registry 并发查询。`ConcurrentNodeHashMap`
  当前把每次 rehash 的旧 bucket array 保留到析构；虽然几何和小于当前 array，服务期仍无法归还旧表；
- normal journey：8 个读线程在预填充、发生过扩容的 4096-entry registry 中执行短 accessor lookup；
  writer 偶尔 reserve/rehash，扩容前已经取得旧 table 的 reader 离开后旧 bucket storage 可释放；
- 范围：仅修改 `ConcurrentNodeHashMap` 的 table hand-off/retirement、显式寿命策略、析构 quiescence、
  定向测试、直接 `container -> mem_reclaim` 边、package journey 和指南；
- 非目标：其他 Container family、node/value/accessor 语义、lock-free map、通用 hazard pointer、缓存/
  淘汰/模型策略、Native FFI/C API、Compute/Graph、公共仪器和 Formal；
- 直接依赖：新增 `PUBLIC mem_reclaim`，保留 `PUBLIC core/mem` 与 `PRIVATE platform`；直接产品消费为
  `hyle::container`，Mem façade 同轮聚合 `mem_reclaim`；
- tier：`P1`；W1-WIN / `closure-perf-v1` / 5% material margin；W1-WSL、M1-MAC 做 correctness 与
  delivery，不产生跨平台 timing claim；
- 总调查预算：协调轮 4 小时中的 90 分钟；性能 30 分钟；最多一次反序复现、一次 mechanism/codegen
  诊断、一个内部产品修复和一次重跑；
- strong peers：oneTBB `843162db8cf4b2531a4a6147b6ea3fd860d44668` 的
  `concurrent_hash_map` 用于 natural same-task route；Folly
  `46afdbec8f58fc21445741278f14d53ec7144676` 只证明 table publication 的 RCU lifetime 机制；
- provenance：`NORTH_STAR_JOURNEY` + owner authorization；不是已经存在的 HYLE inference trace；
- delivery gates：Container source embedding、installed static/shared relocation、header ownership、
  direct edge、package sample；
- stop：本卡有限 FIX 完成即结案；一次允许的修复后仍有 material P1 FAIL、sanitizer/lifetime FAIL 或
  delivery 断边则 `BLOCKED`，不换 workload、margin 或 API 名称追分。

## 预冻结语义档位与成本

| 档位 | 用户前提 | table 生命周期 | 常态读成本 |
|---|---|---|---|
| `RetiredTableLifetime::kMapDestruction`（默认） | 启动期已 reserve 或增长次数有界，owner 接受旧表保留 | 旧表随 map 析构释放 | 编译期移除 reclamation guard；保留现有 lookup 形状 |
| `RetiredTableLifetime::kAfterReaders` | 长寿命/持续增长，要求运行期归还旧表 | publish 后，在进入 hand-off 的旧 reader 离开后回收 | 每次 table hand-off 进入/退出一次 domain；取得 bucket lock 且确认 current 后立刻退出 |

两档的 node stability、accessor、failure、bucket lock、rehash stop-the-table 和析构外部 quiescence合同一致；
差异是明确的 memory lifetime/resource contract，不使用 `fast/light` 名称。显式 reclaiming alias 默认
使用进程 default domain；要求隔离长 reader、确定 teardown，或用外层 read region 摊薄一批 lookup 的
用户显式传入 `ReclamationDomain&`。每个 table 可支付一份冷
retire metadata；不能给每个 node、bucket 或 lookup 分配 callback storage。

## Capability / gap ledger（待 AUDITED 冻结）

| ID | requirement | current | action | verification |
|---|---|---|---|---|
| `CNTR3-01` | growing registry 可及时释放旧 table | `MISSING` | `FIX`：explicit reader-safe type | 强制交错、allocator balance、stress/sanitizer |
| `CNTR3-02` | 已 reserve/有界增长可拒绝 read tax | `COMPLETE`：现有默认 | `KEEP` + 显式 retained alias | codegen、同一 correctness suite |
| `CNTR3-03` | domain ownership/teardown 可发现 | `MISSING` | `FIX`：reclaiming default/explicit domain 构造、指南 | compile-only/package journey |
| `CNTR3-04` | 其他 Container 能力 | v2 `COMPLETE/ROUTED` | `REJECT` 重开 | v2 identity 不变 |

## P1 预声明

- primary normal task：4096 warmed successful lookups，读者取得同样的稳定 value、setup/teardown 不计时；
  HYLE 默认 retained 路线对 oneTBB `NATURAL_ROUTE` 验证 5% adoption non-inferiority；两者算法/资源合同
  不同，不得冒充 same-contract SOTA；
- holdout：`kAfterReaders` 对默认 `kMapDestruction`，只报告运行期内存回收换来的 guard 成本；它是
  产品取舍诊断，不决定 public API 是否存在；
- resource telemetry：table allocations/frees、live/peak bytes、lookup allocations、lock/atomic 路线；
- A/A、known-delta、timer floor、candidate/peer order 与 placement 必须通过统一仪器。

## SCOPED verdict

resume consumer key 已成立；本轮尚未进入 `AUDITED`，没有产品代码授权。下一状态只在 public-surface
trace、语义等价矩阵和有限 FIX 集确认后进入。

## Audit 与 API / usability freeze（AUDITED）

### Capability-family map / public trace

本轮反向追踪 `ConcurrentNodeHashMap` 的构造/valid/size/find/contains/insert/assign/erase/reserve/clear/
for_each/Accessor 全部公开面；只有 table publication lifetime 受影响。node lifetime 继续由 bucket lock
和 accessor 保证，管理操作继续要求外部 quiescence/不得持有同 map accessor。

| Family | journey | current | coverage | action |
|---|---|---|---|---|
| stable-node concurrent registry | lookup/update/erase + accessor | bucket RW lock | `COMPLETE` | `KEEP` |
| growth publication hand-off | reader load table 后等待 bucket lock | 旧表保留到析构 | `PARTIAL` | `FIX` guard 覆盖 load→lock→identity check |
| runtime old-table release | 长寿命增长 registry | 无 | `MISSING` | `FIX` explicit `kAfterReaders` alias |
| bounded-growth no-guard route | 启动期 reserve/接受旧表保留 | 隐式默认 | `COMPLETE` | `KEEP` + named alias |
| shrink/lock-free/snapshot transaction | 不属于本次 registry task | 无 | `OUT_OF_SCOPE` | higher owner/future demand |

API freeze：模板最后增加 `RetiredTableLifetime` 非类型参数，默认 `kMapDestruction`；提供
`RetainedTableConcurrentNodeHashMap` 与 `ReclaimingTableConcurrentNodeHashMap` 两个具名 alias。
`kAfterReaders` 的普通构造借用 process default domain，也可显式以 `ReclamationDomain&` 构造。policy 在编译期裁剪，retained route 不执行 guard/
domain branch；node/accessor/failure 公开形状不变。每张 table 最多一份 retire record 与 owner callback
状态，不能给 node/bucket/lookup 增加 allocation 或 metadata。

冻结 FIX：`CNTR3-01` hand-off guard 与 unlock-before-retire；`CNTR3-02` 两个寿命档位；`CNTR3-03`
domain/destructor/allocator tests；`CNTR3-04` direct edge、package journey、guide、P1。其他 Container
symbol 不改。

### Owner 六问

1. **第三方：**该模板产品路径不直接调用第三方；Container 其他算法的 Highway/xxHash/wyhash/simdutf
   闭包不在本轮变化。`mem_reclaim` 是 HYLE direct factor，不是第三方；oneTBB/Folly 仅为 peer；
2. **推理能力：**llama.cpp/ggml 锁定身份同 Mem reclamation card；backend/device/model metadata registry
   证明稳定-node/长寿命读取任务。通用 concurrent map 属 Container，模型/淘汰策略不上提到这里；
3. **性能：**oneTBB 采用 segmented lazy rehash，资源/算法与 HYLE table-swap material 不同，因此比较
   固定为 `NATURAL_ROUTE`，只回答完整用户路线；5% margin 是 adoption gate，不签 same-contract SOTA。
   两个 HYLE lifetime policy 的 holdout 只量化内存寿命取舍，不以结果增删 API；
4. **FFI：**`N/A/CLEAN`；header template 无 C surface，shared C++ decorated symbol 不等于 C ABI；
5. **交付：**新增 direct `PUBLIC mem_reclaim` 后必须重跑 source/static/shared/relocation、selected-header 和
   internal-factor alias leak；
6. **C++/文档：**默认保留既有零 guard 热路；只有持续增长且要求服务期回收的用户选择 reclaiming alias。
   guide 必须说明 reserve、domain isolation/outer batching、accessor release 和析构外部 quiescence。

### Performance comparison freeze

oneTBB 的 segment allocation、lazy bucket rehash 与 caller/accessor participation 和 HYLE 的
stop-the-table swap 不同，故不是 `SAME_CONTRACT`。normal task 固定为 4096-entry warmed successful
lookup，HYLE 默认 retained 路线对 oneTBB natural route 做 5% adoption non-inferiority；reclaiming policy
是同 HYLE API/result 下的 resource holdout，但 lifecycle 有意不同，只报告 guard cost 与旧表 live bytes。任何
结果都不得外推整个 Container 或全平台 SOTA。

### 实现反馈后的 audit amendment

首次同二进制机制分解得到稳定且 material 的结果：retained HYLE lookup 约 10.3 ns，oneTBB natural route
约 14.9 ns，而每次 lookup 独立进入 HYLE domain 后约 21.5 ns。约 11 ns 是服务期回收的真实 guard 成本，
不是地址/顺序噪声；把它设为所有用户默认会让普通 bounded/reserved registry 无条件付费。候选形成前因此
把默认值改回 `kMapDestruction`，并以具名 reclaiming alias 表达强 lifetime 需求；workload、5% margin、
两档实现和 direct consumer 均未删除。该 amendment 是产品语义反馈，不是为 benchmark 增加 metadata 或
移动计时边界。

Audit verdict：本轮仍只有四项有限 FIX；修正后的 API、默认值、负空间、performance class 和交付边冻结。

## Candidate freeze

产品候选为 `0f6ef8e83ee290ab565c0d53081a0acb75a46675`，tree
`8cb5ba0024cdb20af65347144f09e16f26147632`。默认 retained policy 在编译期裁掉 reclamation guard；具名
reclaiming policy 覆盖 table load 到 bucket-lock identity check，并在 unlock 后 retire。确定性交错、allocator
balance、package journey 与 `container-node-table-p1-v1` 均已进入候选。

## 候选验证结果

- correctness：W1-WIN Release、W1-WSL GCC 14 Release/ASan/UBSan/TSan、M1-MAC AppleClang 21 arm64
  全部通过定向 node-map suite；强制 stale-reader 证明旧表不会早释放，`domain.synchronize()` 后按
  table+buckets 两次 deallocation 精确回收；retained policy 延迟到析构且 allocator balance；
- stress：WSL 的 mixed read/update/growth 与 forced stale-reader 各重复 50 轮；无 race、UAF、double
  free 或 conservation 失败；
- delivery：三平台 Container source embedding、relocated shared/static consumer 全部通过；package
  sample 只链接 `hyle::container`，并核对真实内部 factors `container_intrusive,mem_reclaim`；
- P1：[reclamation evidence](../../evidence/performance/benchmarks-mem-reclamation.md) 的 oneTBB
  natural-route 四个 strata 均为 `WIN / PASS`，median advantage 1.3911--1.4028，最低 99.26% CI
  lower 为 1.3539。它只签稳定 node registry 完整路线，不冒充同算法 SOTA；
- 红队：guard 覆盖 table acquire-load 到 bucket lock + current identity check；成功取得 current bucket
  后先靠 bucket lock 保活再退出 guard；rehash publish 后先释放全部旧 bucket lock，再 retire table；
  map 析构要求外部静止并先 synchronize pending callbacks。显式 domain 必须长寿于 map，默认 domain
  为 process-image lifetime。retained 模板实例没有 runtime domain branch、guard、allocation 或常态
  metadata；没有 benchmark-only API 或伪 C ABI。

一次 delivery 复核发现 sample 错把无关 `util_core` 当成 Container factor；`f457babf` 已改为真实的
`container_intrusive,mem_reclaim` 并在 Windows/WSL/Mac 三交付重放通过。七项 aggregate gate 均有
`PASS` 证据，可以进入 `VERIFIED`。
