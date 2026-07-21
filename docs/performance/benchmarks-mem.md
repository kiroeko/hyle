# mem 实测铁证

> 本页是 [performance 总账](./README.md) 的 **mem 模块**详节。
> 源：`bench/mem/allocator_bench.cc`、`bench/mem/arena_pool_bench.cc`。
> 通用分配器数据来自 2026-07-20 的 Win MSVC / WSL GCC 14 / Apple M4 AppleClang 实测；共享池数据来自 2026-07-21 的同三平台 Release shipping-like 构建（`HYLE_BUILD_TESTS=OFF`），`bench-stat.py -n 9`，性能判定信任配对 **[sign]**。

## 1. 通用路径（`hyle_mem_bench`）

| 场景 | Win | WSL | M4 | 判定 |
|---|---|---|---|---|
| **SystemAllocator 内联** vs 裸 mi churn | 7.93 / 7.86 → **0.996 TIE** | 8.04 / 8.04 → **0.99 TIE** | 7.77 / 7.91 → **1.00 TIE** | ➖ 三平台平 mimalloc |
| **vtable** vs 裸 mi churn | 8.67 / 7.86 → **0.91 LOSE sign** | 8.67 / 8.04 → **0.94 LOSE sign** | 7.93 / 7.91 → **1.00 TIE** | Win/WSL 有约 6–9% 多态税 |
| **vtable** vs pmr(over mi) churn | 8.67 / 8.67 → **TIE** | 8.67 / 10.27 → **1.22 WIN sign** | 7.93 / 9.98 → **1.26 WIN** | 同类多态对照平或赢 pmr |
| **reallocate** vs `mi_realloc` | 80 / 67 → **0.83 LOSE sign** | 83 / 77 → **0.92 LOSE sign** | 82 / 83 → **1.01 TIE** | Win/WSL 输裸 mi |

默认内联路径平 mimalloc；需要运行时注入的 vtable 路径对裸 mi 有可测税，但仍平或赢同为多态封装的 pmr。

## 2. Bump 与单线程定长对象

### Arena / StackAllocator

| | Win | WSL | M4 |
|---|---:|---:|---:|
| Arena ns | 0.55 | 0.37 | 2.23 |
| StackAllocator ns | 0.55 | 0.37 | 2.23 |
| pmr monotonic ns | 1.46 | 0.62 | **1.49** |
| vs pmr | **约 2.6x WIN** | **约 1.7x WIN** | **约 0.67x LOSE** |
| vs 裸 mi | **约 7x WIN** | **约 11x WIN** | **约 2.1x WIN** |

M4 上 pmr monotonic 更快，不能把 x86 结论外推到 ARM。

### GrowablePool

| | Win | WSL | M4 |
|---|---:|---:|---:|
| GrowablePool ns | 2.74 | 1.13 | 2.46 |
| pmr unsync ns | 5.81 | 11.93 | 3.74 |
| 裸 mi ns | 4.87 | 4.45 | 5.80 |
| vs pmr | **约 2.1x WIN** | **约 10x WIN** | **约 1.5x WIN** |
| vs mi | **约 1.8x WIN** | **约 3.9x WIN** | **约 2.4x WIN** |

单线程定长对象场景三平台都赢公平标尺；多线程独享池用 `GrowablePool × N`，不要把单个非线程安全实例共享。

## 3. 两种共享固定块池

| API | 容量契约 | 回收可见性 | 适用场景 |
|---|---|---|---|
| `SharedPool` / `ObjectPool<T>` | **硬容量** | `deallocate()` 后立即可被任意线程复用 | 服务配额、严格并发上限、容量紧张的跨线程共享 |
| `SharedPoolCached` / `ObjectPoolCached<T>` | **软容量** | 释放块可暂存在线程 TLS；`publish_thread_cache()` 或线程退出后公开 | 有容量余量的 request/context 批次、高频同线程复用 |

`SharedPoolCached` 的 TLS 上限为 128 块、refill magazine 为 32 块。这个窗口是明确的吞吐/容量交换，不是 `SharedPool` 的实现细节。

### 3.1 Workload 来自实际用户形状

基准先对齐成熟库和消费者，再扫描参数，不以某个内部阈值反推 workload：

- **request batch lifecycle**：worker 一次申请一批 request/context 对象，处理后整批释放；对应 Folly `IndexedMemPool` / Compression context cache 和 oneTBB fixed-pool 本地批次。
- **producer → neighbor consumer handoff**：生产者申请，邻居消费者释放；对应 oneTBB cross-thread-free、mimalloc Larson/xmalloc migration。轮次用 producer/consumer sequence 串成流水线，没有每轮全局 barrier。
- **service quota**：所有 worker 同时申请超过固定容量的对象；对应 bounded service/marl borrow 配额，验证的是准入和回收语义，不拿启动时间宣称吞吐。

直接对手是 oneTBB `fixed_pool`：双方都由调用者提供固定 buffer。fixed_pool 的 buffer 是与 HYLE 相同的 payload capacity，加 oneTBB 自己 fixed-pool 测试采用的 1 MiB bootstrap metadata allowance；不再额外给 2 倍 payload。每次 fixed-pool/HYLE/mimalloc/TBB 分配都检查非 NULL，任何少做工作会使基准直接失败。pmr resource 与其他池一样在计时 lambda 外构造。`scalable_allocator`、mimalloc 和 pmr 是 growable 参考，不能证明硬容量语义。

### 3.2 三平台基线

配置：8 workers、64B block、request batch=64、working set=容量的 75%；handoff batch=256。表内 ratio 是每轮配对的 **HYLE / oneTBB fixed_pool op/s** 中位数，不是两个绝对中位数相除。

#### Request batch lifecycle

| 池 | Win MSVC | WSL GCC 14 | Apple M4 | 判定 |
|---|---:|---:|---:|---|
| `SharedPool` | 8.315 / 4.200 ns，**0.515x** | 6.609 / 2.280 ns，**0.338x** | 11.881 / 2.024 ns，**0.170x** | 三平台 **LOSE sign** |
| `SharedPoolCached` | 3.540 / 4.200 ns，**1.171x** | 1.754 / 2.280 ns，**1.285x** | 1.537 / 2.024 ns，**1.316x** | Win/WSL **WIN sign**；M4 **TIE sign**（8/9 同侧） |

硬池为立即可见性付出共享原子与稳定空快照成本，不是本地 request churn 的吞吐选择。缓存池在 Win/WSL 稳赢；M4 中位数更快但有一次反转，按配对闸门只能判平。

#### Pipelined producer/consumer handoff

| 池 | Win MSVC | WSL GCC 14 | Apple M4 | 判定 |
|---|---:|---:|---:|---|
| `SharedPool` | 5.304 / 10.433 ns，**1.976x** | 3.807 / 9.239 ns，**2.380x** | 12.931 / 4.871 ns，**0.372x** | Win/WSL **WIN sign**；M4 **LOSE sign** |
| `SharedPoolCached` | 4.062 / 10.433 ns，**2.627x** | 2.414 / 9.239 ns，**3.807x** | 3.072 / 4.871 ns，**1.586x** | 三平台 **WIN sign** |

这里存在真实平台翻转：硬池在 x86 handoff 赢 fixed_pool，在 Apple M4 输约 2.69x。缓存池三平台都赢，但它仍不提供硬容量。

### 3.3 Windows 参数扫描

以下均为 `bench-stat.py -n 9`；扫描线程数、块大小和请求批次，避免用一个点代表整个能力面。

| 维度 | `SharedPoolCached` vs fixed_pool | `SharedPool` vs fixed_pool | 结论 |
|---|---|---|---|
| request，T=1/8/16，64B，batch=64 | **1.232/1.171/1.108 WIN** | **0.720/0.515/0.541 LOSE** | cached 随并发增加优势收窄但仍过 sign 闸门；hard 不适合本地批次 |
| request，T=8，block=32/64/4096B | **1.636/1.171/1.098 WIN** | **0.669/0.515/0.479 LOSE** | 三个数量级内均保持硬/软分工 |
| request，T=8，batch=8/64/128 | **0.993 TIE / 1.171 WIN / 0.898 TIE** | **0.969 TIE / 0.515 LOSE / 0.359 LOSE** | cached 只在中批次明确赢；不能把 batch=64 外推成无条件 SOTA |
| handoff，T=1/8/16，64B | **1.134/2.627/2.003 WIN** | **0.691 LOSE / 1.976 WIN / 1.677 WIN** | T=1 退化成同线程；真并发 x86 handoff 两者都强，M4 见上表 |
| handoff，T=8，block=32/64/4096B | **2.626/2.627/2.029 WIN** | **1.927/1.976/1.506 WIN** | x86 上跨线程迁移对块大小稳定 |

### 3.4 硬容量证据

配置：8 workers、64B block、capacity=1024、请求 110%（1128 次）。三平台得到相同计数：

| 池 | admitted | failed | publish 前可立即重获 | publish 后可重获 |
|---|---:|---:|---:|---:|
| `SharedPool` | **1024** | **104** | **1024** | **1024** |
| `SharedPoolCached` | 910 | 218 | 384 | **1024** |

硬池精确准入 capacity，释放后立即恢复完整容量。缓存池有 640 块暂存于 8 个活线程的 TLS，导致假耗尽；显式 publish 后才恢复 1024。因此容量紧张的 quota 场景必须选硬池，不能用缓存池的更快吞吐替代契约。

## 4. 选型

```text
默认小对象             → SystemAllocator
运行时注入/mock         → vtable Allocator（接受对裸 mi 的个位数 % 税）
单线程 bump             → Arena / StackAllocator（M4 上 pmr 可能更快）
单线程同尺寸可增长       → GrowablePool
多线程、每线程独享       → GrowablePool × N
共享固定块、严格配额     → SharedPool / ObjectPool<T>
共享固定块、吞吐优先且有余量 → SharedPoolCached / ObjectPoolCached<T>
通用多线程 malloc       → mimalloc / oneTBB；不要硬套固定 slab
```

## 5. 复跑

```bash
# shipping-like bench；oneTBB clone 默认位于仓库同级 ../oneTBB
cmake --preset bench -DHYLE_BENCH_TBB=ON
cmake --build --preset bench --target hyle_arena_pool_bench

export HYLE_BENCH_SECTIONS=DE
export HYLE_BENCH_THREADS=8
export HYLE_BENCH_BLOCK_SIZE=64
export HYLE_BENCH_BATCH=64
export HYLE_BENCH_CAPACITY_PCT=75
export HYLE_BENCH_HANDOFF_BATCH=256
export HYLE_BENCH_QUOTA=1024
export HYLE_BENCH_REQUEST_PCT=110

scripts/bench-stat.py build-bench/bin/hyle_arena_pool_bench -n 9 \
  --pair "oneTBB fixed_pool (fixed-buffer shared)" \
         "HYLE SharedPool (hard capacity)" \
  --pair "oneTBB fixed_pool (fixed-buffer shared)" \
         "HYLE SharedPoolCached (soft capacity)"

# quota 计数是单次语义证据；不把其中线程启动时间当吞吐比较
HYLE_BENCH_SECTIONS=E build-bench/bin/hyle_arena_pool_bench
```

Linux/WSL 必须从 ext4 clone 运行；Win、WSL 同机时必须错开，不得并发采样。Apple M4 使用本地 APFS clone。
