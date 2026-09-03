# Mem reclamation v2：推理 snapshot 与 Container consumer 恢复轮

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

## Closure card（CLOSED）

- component / owner / round：`mem_reclaim` / Mem / `mem-reclaim-v2-inference-resume`；
- task kind：`RESUME`；前一轮：`mem-reclaim-v1`；blocker：`MR-P1-PEER-CONSUMER-01`；
- baseline：`444b02c01afe00f4cd3b44433204e58db3c24712`；可复用产品 candidate：
  `66671c39197765623f581dd872f4e626958880db`，但本轮必须重新绑定和验证；
- resume condition 证据：
  1. owner 在 2026-09-03 明确准入推理引擎所需的 read-mostly snapshot reclamation，并由独立
     `container-v3-node-table-reclamation` card 授权 `ConcurrentNodeHashMap` migration/holdout；
  2. 官方 vcpkg `folly 2026.02.23.00#2`（tag `v2026.02.23.00`，commit
     `46afdbec8f58fc21445741278f14d53ec7144676`，port tree
     `2d4ed2106b0c0a7ba6b7d4d04c5be835197443b5`，package ABI
     `6e9bfde81815e7e7f7bf7b5cc94944c9d2ac2452403aede371ee3d7db892b418`）在 W1-WIN / MSVC
     19.50 构建成功；`rcu_domain`、`std::scoped_lock`、`rcu_obj_base::retire` 与 `synchronize()`
     的 reader/publication smoke 返回 0。Folly static target 的 vcpkg config 漏传 Boost thread，资格化
     adapter 显式链接 `Boost::thread`；该 adapter 成本不进入 timed read path；
- user task：读线程在短 guard 内加载不可变模型/backend/request-routing snapshot；writer 原子发布
  replacement，旧 snapshot 或 Container table 在所有 pre-existing readers 离开后回收；
- 范围：复核并保留/修正现有 `ReclamationDomain/Guard/RetireRecord`，迁移一个 Container consumer，
  加入 Mem façade，完成三种交付、指南与 Folly same-contract P1；
- 非目标：hazard-pointer cohort、异步 executor、GPU fence retirement、自动 allocating retire node、
  lock-free container、Obs 迁移、Native FFI/C API、Compute/Graph、公共仪器与 Formal；
- direct dependency：`PUBLIC core` only；direct consumers：`container` 与 Mem product façade；
- tier：`P1`，W1-WIN / `closure-perf-v1` / 5%；W1-WSL、M1-MAC 为 correctness/sanitizer/delivery；
- 总预算：协调轮 4 小时中的 2 小时；P1 45 分钟；一次资格化、一次反序复现、一次 codegen/mechanism
  诊断、一个内部实现修复、一次重跑；
- strong peer：上述 Folly RCU；oneTBB 只用于 Container natural route，不冒充独立 reclamation API peer；
- provenance：`NORTH_STAR_JOURNEY` + 两个成熟 RCU/concurrent-container 形状；不是生产 trace；
- required gates：现有全部 correctness matrix、forced hand-off、allocator balance、stress、ASan/UBSan/
  TSan、W1-WIN/WSL/Mac、factor/product source/static/shared、guide/usability、red-team、P1；
- stop：一个允许的产品修复后仍无法通过 5% same-contract P1、direct consumer、sanitizer 或 delivery，
  则 `BLOCKED`；不放宽 margin、不换 peer、不新增 benchmark-only API。

## 预冻结 API 与成本

保留 `<hyle/mem/reclamation.h>` 的显式 `ReclamationDomain`、move-only `ReclamationGuard`、caller-owned
`RetireRecord`、`read/protect/retire/try_reclaim/synchronize/drain`、默认 domain 与 observation。普通路线：

```cpp
auto guard = domain.read();
const Snapshot* snapshot = guard.protect(published);
consume(snapshot);
guard.reset();

Snapshot* old = published.exchange(next, std::memory_order_acq_rel);
domain.retire(old->retire_record, old, &delete_snapshot);
```

- guard/domain/record 均不拥有 payload；domain 长寿于 guard 与 pending record；guard 同线程退出；
- read O(1)，普通 64 live-thread route 不分配、不锁、不 syscall；第 65 个并发参与线程后走固定 stripe
  RMW；retire O(1) 且不分配，callback 在 collector 调用线程同步执行；
- `try_reclaim` 不等待；`synchronize` 等待 pre-existing readers；`drain` 需要外部 quiescence；
- 默认 domain 适合短 reader；隔离长 reader 或需要确定 teardown 时显式 domain；hazard pointer 是
  stalled-reader 不应阻塞无关对象时的 deferred family。

## Capability / gap ledger（待 AUDITED 冻结）

| ID | requirement | current | action | verification |
|---|---|---|---|---|
| `MR2-01` | allocation-free guarded publication + retire | v1 candidate `COMPLETE` | `KEEP` after API audit | existing matrix + peer shape |
| `MR2-02` | admitted direct consumer | `MISSING` | `FIX` Container table hand-off | forced interleave/stress |
| `MR2-03` | ordinary Mem product delivery | `MISSING` | `FIX` facade/sample/docs | source/static/shared |
| `MR2-04` | qualified same-contract read evidence | `MISSING` | `FIX` official Folly peer | frozen P1 contract |
| `MR2-05` | hazard/executor/GPU retirement | no admitted current journey | `DEFER/ROUTE` | named future trigger |

## P1 预声明

- workload：单线程 warmed 4096 次 immutable snapshot pointer read；两侧均在 guard 内做 acquire load、
  读取相同 payload 并累积 checksum；对象构造、thread registration warm-up、retire 与 synchronize 不计入
  read-side timed phase；
- comparison：`SAME_CONTRACT`；HYLE `domain.read()+protect()` 对 Folly
  `std::scoped_lock<rcu_domain> + atomic.load(acquire)`；caller-owned immutable pointer、读者阻塞回收、
  同线程短 guard、无 cancel/OOM、同 setup/teardown charging；Folly 首次 TLS allocation在 warm-up；
- gate：HYLE paired median 不比 Folly 慢 5%；15/31 samples，candidate-first/peer-first 与 placement strata，
  A/A、known-delta、timer floor 全部资格化；
- allocation/bytes、thread-registration、atomic/fence/codegen 作为 telemetry；若 adapter 或平台使语义
  material 不等价则判 `NOT_COMPARABLE/INCONCLUSIVE`，不得产出 SOTA claim。

## SCOPED verdict

前一 blocker 的两把恢复钥匙均已取得，允许从 `BLOCKED -> SCOPED`。下一状态只在完整 public-surface
trace、六问、语义等价矩阵和有限 FIX 集冻结后进入 `AUDITED`；此前不改产品代码。

## Audit 与 API / usability freeze（AUDITED）

### Capability-family map

| Family | admitted journey | current route | coverage | frozen action / negative space |
|---|---|---|---|---|
| guarded immutable publication | 短读区加载模型/backend/request metadata | `read()` + `protect(atomic<T*>)` | `COMPLETE` | `KEEP`；不拥有或复制 payload |
| allocation-free deferred reclaim | owner 内嵌 record，allocator-aware callback | `RetireRecord` + raw callback | `PARTIAL` | `FIX` typed compile-time callback convenience；不增加 allocating free-function route |
| nonblocking pressure collection | writer 不等待 reader | `try_reclaim()` + threshold | `COMPLETE` | `KEEP`；无进度/FIFO 保证 |
| blocking grace period / teardown | 等待 pre-existing readers 并清空 owner work | `synchronize()` / quiescent `drain()` | `COMPLETE` | `KEEP`；guard 内调用 blocking route 是合同错误 |
| default / isolated domains | 普通短 reader；隔离长 reader/确定 teardown | default domain / explicit domain | `COMPLETE` | `KEEP`；不承诺跨 DSO 单例 |
| stalled-reader-independent reclaim | 一个停顿 reader 不阻塞无关对象 | 无 | `OUT_OF_SCOPE` | hazard pointer：真实触发后另立能力轮 |
| executor/GPU-affine retirement | 指定线程或 fence 后析构 | 无 | `ROUTED` | Sched/backend owner 组合，不下沉到本 factor |

### Public-surface trace 与有限 FIX

现有公开面只有 `ReclaimCallback`、`RetireRecord`、`ReclamationGuard`、`ReclamationDomain` 和
`default_reclamation_domain()`；每个入口均映射到上表，`pending/reclaimed/retire_threshold` 是 owner
pressure observation。没有 benchmark-only symbol。冻结 FIX 只有：

1. `MR2-01` 增加 `domain.retire<callback>(record, T*)` 的零分配 typed convenience，raw `void*` 路线继续
   服务 allocator-aware/internal owner；
2. `MR2-02` 完成 Container table hand-off consumer；
3. `MR2-03` 完成 Mem façade、三交付、package sample 与指南；
4. `MR2-04` 完成官方 Folly same-contract P1。

API freeze：typed overload 只在编译期生成 raw thunk，不改变 record/domain layout，不存 deleter，不增加
allocation/branch；Container 的 lifetime policy 属 Container card。冻结后只允许原子顺序、内部 route、
冷队列和无公开 layout 的实现优化。

### Owner 六问

1. **第三方：**产品路径无第三方；只用 C++ atomics/mutex/TLS。Folly vcpkg 身份仅为 `peer`，不会链接
   到 HYLE 产品或泄漏其类型；
2. **推理能力：**2026-09-03 远端确认 llama.cpp
   `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`、standalone ggml
   `d4716378882593333721eb33f153144b6885caf2`；llama 随仓 ggml tree 为
   `b91bd1c0eadf24e7d56143f36942e1a838c1dd8b`（最近同步 commit
   `159b741427337a2e9a58b08121001545d66b5825`）。它们提供 backend/model registry 与长寿命 metadata
   的真实形状，但没有通用 RCU API；本能力是 `NORTH_STAR_JOURNEY`，不是伪称现有 production trace；
3. **性能：**P1，冻结的短 snapshot read，与官方 Folly 同语义；5% non-inferiority；retire/synchronize
   只做复杂度与资源 telemetry，不外推 universal SOTA；
4. **FFI：**`N/A/CLEAN`；无 C header/schema、bridge、未装饰 export 或 managed sample；DLL 导出的是
   同工具链 C++ 方法；
5. **交付：**当前 exact factor 已验证，但 Mem product source/static/shared 为缺口，列入 `MR2-03`；
6. **C++/文档：**RAII guard、显式 domain、intrusive record 的 ownership 自然；typed overload 修掉常见
   `void*` cast 摩擦。指南必须解释 RCU vs lock/shared_ptr/hazard、短 guard、domain/record lifetime、
   callback 执行线程和 blocking self-wait。

### Same-contract equivalence freeze

| Dimension | HYLE | Folly adapter | verdict |
|---|---|---|---|
| work/result | guard 内 acquire-load immutable pointer 并读同 payload | 相同 | `EQUAL` |
| ownership/lifetime | caller owns pointer/domain；reader 阻塞 retire | 相同 | `EQUAL` |
| failure/cancel | warmed read 不分配、无结果失败/取消 | 首次 TLS 分配移出 timed phase | `ADAPTER_EQUALIZED` |
| ordering/progress | reader 不阻塞 writer publication；stalled reader 延后 reclaim | 相同 | `EQUAL` |
| setup/teardown | registration warm-up、retire/sync 不计入 read phase | 相同 | `EQUAL` |
| delivery/codegen | HYLE source/static；官方 Folly static peer | paired source benchmark，adapter link 不在 loop | `ADAPTER_EQUALIZED` |

Audit verdict：全部 `MUST` 已是 `COMPLETE` 或有限 `FIX`，API/成本/peer/workload 已冻结，允许进入实现。

## Candidate freeze

产品候选为 `0f6ef8e83ee290ab565c0d53081a0acb75a46675`，tree
`8cb5ba0024cdb20af65347144f09e16f26147632`。它完成 typed allocation-free retire、Container 旧表直接
consumer、Mem façade 与 package journey，并冻结 `mem-reclaim-p1-v2`。候选之后只允许补验证记录、
状态与文档事实；公开合同、算法或 normal workload 改动必须退回 `AUDITED`。

## 候选验证结果

- W1-WIN：MSVC Release 的 `hyle_mem_reclaim_test` 与 `hyle_container_test` 通过；Mem、Container
  selected-component 的 source、relocated shared、relocated static consumer 全部通过；
- W1-WSL：GCC 14 Release 全量 Mem-reclaim 13/13、Container 339/339；ASan、UBSan、TSan 下
  Mem-reclaim 13/13 与 node-map 9/9；两个并发/回收测试各重复 50 轮；Mem/Container 三交付通过；
- M1-MAC：AppleClang 21 arm64 下 Mem-reclaim 13/13、node-map 9/9；Mem/Container 三交付通过；
- delivery/FFI：`hyle_mem_reclaim.dll` 导出 7 个 MSVC 修饰的同工具链 C++ 符号，无未修饰 HYLE C
  symbol、C schema、bridge 或 managed fixture；公开头由三种消费者独立编译；
- P1：[reclamation evidence](../../evidence/performance/benchmarks-mem-reclamation.md) 的 Folly
  same-contract read 四个 strata 均为 `WIN / PASS`，median advantage 1.3649--1.3812，最低
  99.26% CI lower 为 1.3358；A/A、5%/10% known delta、timer floor、placement、身份与 provenance
  全部通过；这是 W1-WIN 工程结论，不是 Formal；
- 红队：复核 epoch flip/late reader、nested guard、callback reentrancy、record reuse、domain teardown、
  默认 domain image lifetime、超过 64 reader 的 fallback 与 Container load→lock hand-off。没有发现
  blocker；性能元数据未进入 payload/record/read path，typed convenience 只生成 compile-time thunk。

候选到采样 harness `c2ccc06f` 的产品路径无差异；之后 `f457babf` 只修正 Container package sample
对真实 factor 的断言。七项 aggregate gate 均已有 `PASS` 证据，可以进入 `VERIFIED`。

## Closure decision 与 terminal receipt

本 factor **`CLOSED`**：MR2-01--MR2-04 均为 `COMPLETE`，七项 aggregate gate 全为 `PASS`，无
blocker。保留而不删除它的理由是已经准入的推理 snapshot journey：模型/backend/request-routing/config
等 immutable metadata 极热读取、低频替换时，读侧无需 shared ownership RMW，writer 仍能在 grace
period 后释放旧版本；Container 的长期增长 node registry 也已成为直接产品 consumer。

已知限制：domain、payload 与 intrusive record 均由调用者拥有；domain 必须长寿于 guard/pending record，
guard 同线程退出，`drain`/析构需要外部静止。一个 stalled reader 会延迟同 domain 的回收；callback 在
collector 线程同步执行且必须 `noexcept`/短小；第 65 个并发 live thread 起使用固定共享 stripe。P1 只
签 W1-WIN 的短 immutable read，同步写侧、长 reader、fallback contention、跨平台与端到端 token latency
没有排名或 Formal 声明。

只有可复现的 reader/retire/lifetime/callback correctness 或易用性缺陷，支持 compiler/atomic/TLS 使
合同失效，公开 API/layout、算法、direct edge 或三交付改变，锁定同合同正常 consumer 出现 material
regression，或真实消费者正式要求 stalled-reader-independent hazard/GPU-fence/executor-affine 回收时
重开相应能力族。

immutable candidate 四项 LOC 为 `436,638 / 111,204 / 70,416 / 227`，由
`loc-report.py --module mem_reclaim --commit 0f6ef8e83ee290ab565c0d53081a0acb75a46675 --self-check`
重算通过。`227` 只计 factor-owned `src/mem_reclaim`；共享公开头计入 Mem conventional boundary。
registry 将在下一提交绑定本 terminal receipt 与首个实际保存 `CLOSED` state/candidate 的 anchor。
