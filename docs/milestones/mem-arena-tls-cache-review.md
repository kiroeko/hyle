# MemoryArena TLS freelist — 审核包（2026-07-20/21）

> **历史名称边界：**本文记录的 `MemoryArena` TLS 实现现名为 `SharedPoolCached`，只代表软容量缓存池；当前硬容量默认是 `SharedPool`。旧 `memory_arena.h` 与 `hyle_memory_arena_*` 已删除，由 `shared_pool.h`、`hyle_shared_pool_*` / `hyle_shared_pool_cached_*` 取代。

> **当前闭环**：生产入口为 `include/hyle/mem/shared_pool.h` + `src/mem/shared_pool.cc`，测试为 `tests/mem/shared_pool_test.cc`，设计与三平台 fixed_pool 证书分别见 `docs/architecture/mem-design.md`、`docs/performance/benchmarks-mem.md`。`cloc`：双池核心 758 行生产代码；对应测试 + allocator/pool benchmark 1122 行；本轮相关八文件 bundle 从 2068 增至 2834 行代码（+766，其中生产 bundle +364）。
>
> 下文从 §1 起保留 2026-07-20/21 旧 TLS 实现的审计过程、旧路径与当时数字，用于解释为何最终拆成硬/软两套合同；不得当作现行 API 或性能证书。

## 1. 动机

2026-07-20 三平台重测后，**always-CAS 分片 Treiber** 的 MemoryArena 在 8 线程同线程 alloc/free churn 上对 oneTBB `scalable_allocator` **sign 输 0.72–0.91x**。旧账“赢 tbb ~3x”作废。

| 根因 | 说明 |
|---|---|
| 热路径形状 | MemoryArena 每次 pop/push 都是 `lock` CAS；tbb 同线程走 **非原子 TLS freelist** |
| 契约差 | MemoryArena = 有界共享固定 slab；tbb = 通用堆 + TLS + backend |
| 非 bug | 分片已消除 O(threads) 单 head 争用；输在 **同线程仍付 CAS 常数** |

## 2. 实现摘要

两层组成（同一类型，不同路径）：

1. **TLS freelist（owner-only，零原子）**  
   - 每线程、每 arena 一个 `ThreadCache`（`kTlsCacheCap = 32` 个 block index）  
   - `deallocate` → TLS push；满则 flush 半量到属主 shard  
   - `allocate` → TLS pop；空则走分片 Treiber  

2. **分片 Treiber（共享后备）**  
   - 与改前相同：pow2 shard、地址路由 free、ABA version pack  

3. **生命周期（2026-07-21 审计加固）**  
   - `ThreadCache` 节点由 **thread_local 链表**拥有；节点内存由 **`system_allocator()`** 分配/释放（**非** arena 注入 backing——节点可比 arena+backing 长寿）  
   - 进程级全局锁 `detail::tls_registry_lock`（`memory_arena.cc` 单定义，跨 DLL 唯一）管 attach/detach/reset/dtor；`~MemoryArena` / thread 退出 → 全局锁下 flush + `arena.store(nullptr, release)`  
   - **僵尸回收**：`tls_cache()` 遍历时 acquire-load `arena`，`==nullptr` 即释放节点（防长寿命线程 churn 多 arena 泄漏）  

4. **显式 handoff**  
   - `publish_thread_cache()`：本线程 TLS → 属主 shard（跨线程要用 free 块时）  

### 公开契约变化（审核重点）

| 项 | 改前 | 改后 |
|---|---|---|
| 同线程 free→alloc | 每次 CAS | **零原子** TLS |
| 硬容量 | free 立刻在 shard 可见 | **软容量**：他线程 TLS 内 free 对 `allocate` 不可见，窗口 ≤32×活线程 |
| 恢复硬容量 | — | `publish_thread_cache()` / overflow flush / 线程 exit / arena dtor |
| `reset()` | 重建 shard | 先清空所有已注册 TLS `count`，再重建（须 quiesce 并发） |

## 3. 红队结论（已试过的死路）

| ID | 问题 | 裁定 |
|---|---|---|
| **R1 软容量** | free 进他线程 TLS → 本线程可能 `nullptr` 而池未空 | **保留**。与 tbb/jemalloc tcache 同类。试 **热路径锁 + remote steal** → 硬容量恢复，但 Win vs tbb **1.74x → 0.98 TIE**，拒绝 |
| **R2 僵尸泄漏** | 池析构后 TLS 节点 `arena=null` 直到线程退出 | **已修**（遍历回收） |
| **R3 reset/dtor 碰 TLS** | 与并发 alloc/free 无锁同步 | **契约**：同 `reset()` 须 quiesce（与改前一致） |
| **R4 free 语义** | 不再总是立刻地址路由 CAS | overflow/publish/exit 才回流；`CrossThreadFree` 测 join→TLS dtor flush |
| **R5 节点生命周期（2026-07-21 审计新增，2 个真缺陷已修）** | ① C ABI：节点复制注入 backing `Allocator` 自释放，但 backing 随 wrapper 先死 → 线程退出 flush **确定性 UAF**（ASan 坐实）。② dtor vs 线程退出：per-arena `atomic_flag` 随 arena 先死，输方锁/读已释放对象 → **data race**（TSan 坐实 4 条） | **已修**：① 节点改 `system_allocator()` 分配/释放（与注入 backing 解耦）；② 换进程级全局锁 + `arena` 字段 atomic。回归 `InjectedBackingFreedBeforeWorkerExitNoUaf`；Win ASan + WSL TSan 双绿 |

## 4. 性能（审核用数字）

`scripts/bench-stat.py -n15` + pair sign，8 线程 shared 64B churn：

| 平台 | MemoryArena | oneTBB | ratio (HYLE/tbb) |
|---|---:|---:|---:|
| Win MSVC | ~1.19 ns | ~2.17 ns | **~1.74x WIN sign** (15/15) |
| WSL GCC14 | ~1.28 ns | ~1.80 ns | **~1.41x WIN sign** |
| Apple M4 | ~1.21 ns | ~1.97 ns | **~1.62x WIN sign** |

- vs `pmr::synchronized_pool`：仍 **~50–100x** 量级。  
- 红队 steal+锁后 Win：**~2.20 vs 2.15 → TIE**（故弃）。  
- ST 段 **不再** 把 MemoryArena 与 GrowablePool 同表（工具错配）。

## 5. 测试

`MemoryArena.*`（Win 全 mem 104 曾绿；MemoryArena 子集 12）：

| 测试 | 覆盖 |
|---|---|
| 原有 round-trip / 耗尽 / 对齐 / 并发 / reset / cross-thread free | 回归 |
| `SameThreadTlsCacheReusesPointer` | free→alloc 同指针（TLS LIFO） |
| `TlsCacheOverflowStillReclaimsAll` | cap>32 半量 flush 不丢块 |
| `ResetClearsTlsCache` | reset 后无双发 TLS 索引 |
| `RemoteTlsSoftThenPublishRestoresHardCapacity` | **钉死软容量 + publish 恢复** |
| `ManyArenaLifetimesDoNotBreakTls` | 多 arena 生命周期 / 僵尸路径 |

建议审核 CI：**Win Release `hyle_mem_test` + WSL TSan `MemoryArena.*`**（lock-free 面）。

## 6. 选型（给产品审核）

```text
ST / per-thread 无界同尺寸  → GrowablePool（别用 MemoryArena）
MT 共享有界定长块          → MemoryArena（TLS + 分片；接受软窗口或 publish）
通用 MT malloc              → SystemAllocator / mi / tbb
跨线程 free 后立刻硬容量    → free 方 publish_thread_cache() 或 join free 线程
```

## 7. 审核清单（勾选）

- [ ] 软容量是否可接受（fiber 栈池 / ObjectPool 固定 N 场景是否要求 publish）  
- [ ] `publish_thread_cache` 是否进入 C ABI / 文档 guide（当前仅 C++）  
- [ ] dtor / thread_local 退出顺序是否满足你们进程生命周期  
- [ ] 三平台 bench 数字是否要写入发布说明（见 `benchmarks-mem.md`）  
- [ ] 是否要求 WSL TSan 本轮再跑一记（实现无新共享写热路径，steal 已撤回）  

## 8. 非本包范围

- performance 目录其它模块拆页（error/platform/time/container/simd-ops）与 mem 同批提交，但**审核本文件只针对 MemoryArena TLS**。  
- GrowablePool / SystemAllocator / Arena bump：本轮未改算法；仅 ST bench 形状与文档。
