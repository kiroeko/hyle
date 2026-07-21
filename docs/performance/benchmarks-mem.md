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
