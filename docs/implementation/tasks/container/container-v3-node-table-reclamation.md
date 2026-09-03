# Container v3：并发 node map 的旧表寿命

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（SCOPED）

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
| `RetiredTableLifetime::kAfterReaders`（默认） | 长寿命/持续增长，要求运行期归还旧表 | publish 后，在进入 hand-off 的旧 reader 离开后回收 | 每次 table hand-off 进入/退出一次 domain；取得 bucket lock 且确认 current 后立刻退出 |
| `RetiredTableLifetime::kMapDestruction` | 启动期已 reserve 或增长次数有界，owner 接受旧表保留 | 旧表随 map 析构释放 | 编译期移除 reclamation guard；保留现有 lookup 形状 |

两档的 node stability、accessor、failure、bucket lock、rehash stop-the-table 和析构外部 quiescence合同一致；
差异是明确的 memory lifetime/resource contract，不使用 `fast/light` 名称。默认构造使用进程默认 domain；
要求隔离长 reader 或确定 teardown 的用户显式传入 `ReclamationDomain&`。每个 table 可支付一份冷
retire metadata；不能给每个 node、bucket 或 lookup 分配 callback storage。

## Capability / gap ledger（待 AUDITED 冻结）

| ID | requirement | current | action | verification |
|---|---|---|---|---|
| `CNTR3-01` | growing registry 可及时释放旧 table | `MISSING` | `FIX`：默认 reader-safe retirement | 强制交错、allocator balance、stress/sanitizer |
| `CNTR3-02` | 已 reserve/有界增长可拒绝 read tax | `PARTIAL`：只有隐式默认 | `FIX`：显式 compile-time lifetime route | codegen、同一 correctness suite |
| `CNTR3-03` | domain ownership/teardown 可发现 | `MISSING` | `FIX`：默认与显式 domain 构造、指南 | compile-only/package journey |
| `CNTR3-04` | 其他 Container 能力 | v2 `COMPLETE/ROUTED` | `REJECT` 重开 | v2 identity 不变 |

## P1 预声明

- primary normal task：4096 warmed successful lookups，读者取得同样的稳定 value、setup/teardown 不计时；
  HYLE 默认回收路线对 oneTBB natural route 验证 5% non-inferiority，若语义矩阵发现 material 差异则
  降为 `NATURAL_ROUTE`，改用预冻结 HYLE consumer budget `<= 1.05x` retained route；不得冒充
  same-contract SOTA；
- holdout：默认 `kAfterReaders` 对 `kMapDestruction`，只报告运行期内存回收换来的 guard 成本；它是
  产品取舍诊断，不决定 public API 是否存在；
- resource telemetry：table allocations/frees、live/peak bytes、lookup allocations、lock/atomic 路线；
- A/A、known-delta、timer floor、candidate/peer order 与 placement 必须通过统一仪器。

## SCOPED verdict

resume consumer key 已成立；本轮尚未进入 `AUDITED`，没有产品代码授权。下一状态只在 public-surface
trace、语义等价矩阵和有限 FIX 集确认后进入。
