# mem 实测铁证

> 本页是 [performance 总账](./README.md) 的 **mem 模块**详节。  
> 源：`bench/mem/allocator_bench.cc` + `bench/mem/arena_pool_bench.cc`（`-DHYLE_BENCH_TBB`）。  
> **2026-07-20 三平台**：Win MSVC 19.50 / WSL GCC 14.2 / Apple M4 AppleClang 21，`bench-stat -n15`，信任 **[sign]**。  
> **同日加固**：`MemoryArena` 加 **TLS freelist 缓存**（同线程 free→alloc 非原子）；8 线程 shared 对 oneTBB 从 sign 输翻为 **三平台 sign 赢**。

## 1. 通用路径（`hyle_mem_bench`）

| 场景 | Win | Lin | M4 | 判定 |
|---|---|---|---|---|
| **SystemAllocator 内联** vs 裸 mi churn | 7.93 / 7.86 → **0.996 TIE** | 8.04 / 8.04 → **0.99 TIE** | 7.77 / 7.91 → **1.00 TIE** | ➖ **三平台平 mimalloc SOTA** |
| **vtable** vs 裸 mi churn | 8.67 / 7.86 → **0.91 LOSE sign** | 8.67 / 8.04 → **0.94 LOSE sign** | 7.93 / 7.91 → **1.00 TIE** | ⚠️ Win/Lin 固定 ~6–9% 多态税；M4 平 |
| **vtable** vs pmr(over mi) churn | 8.67 / 8.67 → **TIE** | 8.67 / 10.27 → **1.22 WIN sign** | 7.93 / 9.98 → **1.26 WIN** | ✅ 同类多态：平或赢 pmr |
| **vtable** touch / 8-thread vs mi | 多行 sign 略输 ~0.89–0.97 | 同左 | touch 平；8t 0.94 LOSE | ⚠️ 对裸 mi 常有个位数 % 税 |
| **reallocate** vs mi_realloc | 80 / 67 → **0.83 LOSE sign** | 83 / 77 → **0.92 LOSE sign** | 82 / 83 → **1.01 TIE** | ⚠️ Win/Lin 输裸 mi；仍远赢 std::realloc |

**结论**：默认内联路径 = **平 SOTA**。vtable 对 **裸 mi** 不是“统计全平”——Win/Lin churn **sign 稳输 ~9%**；对 **pmr** 仍赢/平。

## 2. 专用分配器（`hyle_arena_pool_bench`）

### Bump

| | Win | Lin | M4 |
|---|---|---|---|
| Arena ns | 0.55 | 0.37 | 2.23 |
| Stack ns | 0.55 | 0.37 | 2.23 |
| pmr monotonic | 1.46 | 0.62 | **1.49** |
| vs pmr | **~2.6x WIN** | **~1.7x WIN** | **~0.67x**（仍可输 pmr；赢裸 mi） |
| vs 裸 mi | **~7x WIN** | **~11x WIN** | **~2.1x WIN** |

⚠️ **平台翻转**：Win/Lin 大赢 pmr；**M4 上 pmr monotonic 仍可能更快**（仍赢裸 mi）。

### 单线程 pool（仅 GrowablePool）

| | Win | Lin | M4 |
|---|---|---|---|
| GrowablePool ns | 2.74 | 1.13 | 2.46 |
| pmr unsync | 5.81 | 11.93 | 3.74 |
| 裸 mi | 4.87 | 4.45 | 5.80 |
| vs pmr | **~2.1x WIN** | **~10x WIN** | **~1.5x WIN** |
| vs mi | **~1.8x WIN** | **~3.9x WIN** | **~2.4x WIN** |

✅ **三平台赢公平 ST 标尺**。

### 8 线程（TLS 后重测，`bench-stat -n15` + pair sign）

| | Win | Lin | M4 |
|---|---|---|---|
| GrowablePool×8 vs mi×8 | **~1.41x WIN sign** | **~2.0x WIN** | **~2.3x WIN** |
| MemoryArena shared vs pmr sync | **~106x WIN sign** | **~77x WIN sign** | **~57x WIN sign** |
| MemoryArena vs **oneTBB scalable** | 1.19 / 2.17 → **1.74x WIN sign** (15/15) | 1.28 / 1.80 → **1.41x WIN sign** (15/15) | 1.21 / 1.97 → **1.62x WIN sign** (15/15) |

✅ **三平台对 tbb 均 sign 赢**（owner-only TLS，零原子）。旧 always-CAS 0.72–0.91 **作废**。红队：热路径加锁做 remote-steal → Win 退成 **~0.98 TIE**，拒绝。

> ⚠️ **审计更正（2026-07-21）**：上面这组"8 线程 shared"数字的 workload（紧 `allocate→deallocate`）是**深度 1 的 TLS ping-pong**——首次 shard pop 后每 op 都命中本线程 TLS freelist，**从不回落共享 shard**。它只测 TLS 快路常数，**不测**它宣称的共享后备协调成本。故"赢 tbb 1.74x"= TLS 快路 vs tbb 的 tcache，不是共享 slab 竞争胜利。真正压到共享路径的数据见下 §2b（section D）。此外本机 Windows/WSL 当前**未构建 oneTBB**（缺 `tbbmalloc.dll`），上表 tbb 列无法在本机复现，留待重建 TBB 后重测。

### 2b. 共享后备路径实测（section D，2026-07-21 审计新增）

section C row(2) 的 ping-pong 测不到 lock-free 后备——MemoryArena 的存在理由。section D 两个 workload 真正驱动 TLS↔shard 机制，对同契约 SOTA（`pmr::synchronized_pool`，唯一同契约的"共享有界池"）：

- **D1 BURST**（每轮持有 64 块 > `kTlsCacheCap`=32 再全放 → 强制 TLS overflow-flush + shard refill，即真跑 Treiber push/pop）
- **D2 CROSS-THREAD handoff**（线程 t 分配、屏障轮转、线程 t 释放邻居 t+1 的批 → free 无法命中本线程 TLS，必须按地址路由回属主 shard 的 Treiber stack；即设计的"fiber 栈 core A 分配 / core B 释放"）

`bench-stat -n9` + 配对符号检验（section 过滤 `HYLE_BENCH_SECTIONS=D`，Win 与 WSL 错开跑）。**下表是 magazine 批量转移落地后（`kRefillBatch=16`，一次 CAS 搬 K 块）的数字**：

| workload | 对手 | Win MSVC | WSL GCC14 | 判定 |
|---|---|---:|---:|---|
| **D1 BURST** | `pmr::synchronized_pool`（同契约 std 池） | **27.0x** | **24.5x** | **WIN sign**（9/9，p=3.9e-3）|
| **D2 CROSS-THREAD** | `pmr::synchronized_pool` | **16.2x** | **7.9x** | **WIN sign**（9/9，p=3.9e-3）|
| **D1 BURST** | **oneTBB scalable** | **0.44x** | （本轮 WSL 未建 TBB）| **LOSE sign**（0/9）— TBB **2.3x 更快** |
| **D2 CROSS-THREAD** | **oneTBB scalable** | **0.90x** | — | **TIE sign**（1/9，p=3.9e-2）|
| **D2 CROSS-THREAD** | mimalloc（异契约：延迟原子 push） | ~1.0（≈tbb）| 1.008x TIE | 两平台打平 |

绝对值（magazine 后）：D1 MemoryArena Win ~5.0ns / WSL ~4.8ns（改前 ~21ns，**~4.4x 加速**）；D2 Win ~10ns（改前 ~22ns）/ WSL ~52ns。

**magazine 前后对比（Win MSVC，同一 bench）**：

| workload / 对手 | 逐块 CAS（改前） | magazine（改后） |
|---|---:|---:|
| D1 BURST 绝对 | 21.3 ns | **5.0 ns** |
| D1 BURST vs oneTBB | 0.106x（输 9.4x） | **0.44x（输 2.3x）** |
| D1 BURST vs pmr | 5.8x 赢 | **27x 赢** |
| D2 CROSS vs oneTBB | 0.363x（输 2.75x） | **0.90x（TIE）** |
| D2 CROSS vs pmr | 7.2x 赢 | **16x 赢** |

**诚实结论**：
1. 对**同契约** std 共享池（`pmr::synchronized_pool`）：**16–27x sign 稳赢**（两平台一致）。共享有界定长池对 std 同类的可靠 SOTA 优势。
2. 对 **oneTBB scalable**：**CROSS-THREAD 打平**（0.90x TIE，从改前输 2.75x 追平）；**BURST 仍输 2.3x**（从改前输 9.4x 大幅缩小）。BURST 残差是契约差价：tbb/mi 的 per-thread heap 同线程 free 是**非原子本地链**，HYLE 是真共享 slab，每 `kRefillBatch`=16 块仍付 1 次 CAS refill/flush。同契约下这是合理位置。
3. 平台翻转（历史）：改前 mi/tbb 跨线程 free 在 Win 比 HYLE 快 ~2.7x、WSL 打平；magazine 后 Win 也追平。单平台数不能当无条件结论。

**旧账更正**：旧文档"赢 tbb 1.74x"是 **ping-pong 假象**（测 TLS 快路 vs tbb tcache，非共享后备竞争）。真压共享路径：magazine 前对 tbb 完输，magazine 后 CROSS-THREAD 追平、BURST 缩到 2.3x。⚠️ Mac M4 待补。

## 3. MemoryArena vs tbb：输因、修法、红队

| 项 | always-CAS | TLS owner-only（现） | TLS+锁+steal（红队试，弃） |
|---|---|---|---|
| 同线程 free→alloc | 每次共享 CAS | **零原子** TLS | 本地 spinlock |
| 硬容量（他线程 TLS 有 free） | 是 | **软**：需 publish/join | 是（steal） |
| vs tbb 8t（Win） | 0.91 LOSE | **~1.7–2.0 WIN** | **~0.98 TIE** |

## 4. bench 单线程段形状

| 旧 | 问题 | 新 |
|---|---|---|
| ST 段含 MemoryArena + ObjectPool | 多线程工具；付 CAS；无公平 ST SOTA 分母 | **删除** |
| ST 表只留 GrowablePool + pmr + mi | — | 保留 |
| MemoryArena | — | **仅** 8 线程 shared 段，对 pmr sync + tbb |

## 5. 非 SOTA / 边界（2026-07-20 末）

1. ~~MemoryArena vs oneTBB 输~~ → **已翻为三平台 WIN**（TLS）。  
2. **Arena/Stack vs pmr @ M4**：仍可能输 pmr monotonic —— 平台翻转。  
3. **vtable vs 裸 mi**：Win/Lin 常 **~5–10% sign 输**（对 pmr 仍好）。  
4. **reallocate vs mi**：Win/Lin sign 输。  
5. 默认 SystemAllocator：三平台 **平 mi** ✅。  
6. GrowablePool ST + 8t per-thread：三平台 **赢** ✅。  
7. MemoryArena 并发软耗尽：他线程 TLS 内 free 不可见（≤32×T）；join 后全回收。

## 6. 选型（不同场景不同 mem 组成）

```text
默认小对象     → SystemAllocator（平 mi）
注入/mock      → vtable Allocator（接受对裸 mi 个位数 %）
ST bump        → Arena/Stack（M4 上 pmr 可能更快，需本机验）
ST 同尺寸无界  → GrowablePool
MT per-thread  → GrowablePool×N（赢 mi）
MT 共享有界定长 → MemoryArena（TLS + 分片；赢 tbb + 碾 pmr sync）
通用 MT malloc → mi / tbb，别硬套固定 slab
```

## 7. 复跑

```bash
# Win: PATH 含 build-bench/bin + tbbmalloc 所在 msvc_*_release
scripts/bench-stat.py build-bench/bin/hyle_mem_bench -n 15
scripts/bench-stat.py build-bench/bin/hyle_arena_pool_bench -n 15 \
  --pair "oneTBB scalable_allocator (8 threads, SHARED SOTA)" \
         "HYLE MemoryArena (8 threads, 1 SHARED pool)"
# -DHYLE_BENCH_TBB=ON -DHYLE_TBB_DIR=.../oneTBB
```
