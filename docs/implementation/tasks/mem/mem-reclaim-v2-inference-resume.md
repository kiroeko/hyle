# Mem reclamation v2：推理 snapshot 与 Container consumer 恢复轮

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Closure card（SCOPED）

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
