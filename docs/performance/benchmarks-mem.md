# mem 实测铁证

> 本页是 [performance 总账](./README.md) 的 **mem 模块**详节。
> 源：`bench/mem/allocator_bench.cc`、`bench/mem/scenario_bench.cc`。
> 通用分配器数据来自 2026-07-20 的 Win MSVC / WSL GCC 14 / Apple M4 AppleClang 实测；共享池数据来自 2026-07-21 的同三平台 Release shipping-like 构建（`HYLE_BUILD_TESTS=OFF`），`bench-stat.py -n 9`；request closure 数据来自 2026-07-22 的 Win、WSL GCC 14/Clang 20 `n=15` 与 Apple M4 每种 QoS `n=31`，mem 实现绑定 commit `7f270bb`，调度审计绑定 `803dae6`。新增 whole-request tail certificate 绑定 clean commit `4490c40`，FixedPool hot-path 实现为其父提交 `871d9e8`。性能判定信任配对 **[sign]**。

## 1. 通用路径（`hyle_mem_bench`）

| 场景 | Win | WSL | M4 | 判定 |
|---|---|---|---|---|
| **SystemAllocator 内联** vs 裸 mi churn | 7.93 / 7.86 → **0.996 TIE** | 8.04 / 8.04 → **0.99 TIE** | 7.77 / 7.91 → **1.00 TIE** | ➖ 三平台平 mimalloc |
| **vtable** vs 裸 mi churn | 8.67 / 7.86 → **0.91 LOSE sign** | 8.67 / 8.04 → **0.94 LOSE sign** | 7.93 / 7.91 → **1.00 TIE** | Win/WSL 有约 6–9% 多态税 |
| **vtable** vs pmr(over mi) churn | 8.67 / 8.67 → **TIE** | 8.67 / 10.27 → **1.22 WIN sign** | 7.93 / 9.98 → **1.26 WIN** | 同类多态对照平或赢 pmr |
| **reallocate** vs `mi_realloc` | 80 / 67 → **0.83 LOSE sign** | 83 / 77 → **0.92 LOSE sign** | 82 / 83 → **1.01 TIE** | Win/WSL 输裸 mi |

默认内联路径平 mimalloc；需要运行时注入的 vtable 路径对裸 mi 有可测税，但仍平或赢同为多态封装的 pmr。

## 2. Bump 与单线程定长对象

### PooledRegion / StackAllocator

| | Win | WSL | M4 |
|---|---:|---:|---:|
| PooledRegion ns | 0.55 | 0.37 | 2.23 |
| StackAllocator ns | 0.55 | 0.37 | 2.23 |
| pmr monotonic ns | 1.46 | 0.62 | **1.49** |
| vs pmr | **约 2.6x WIN** | **约 1.7x WIN** | **约 0.67x LOSE** |
| vs 裸 mi | **约 7x WIN** | **约 11x WIN** | **约 2.1x WIN** |

M4 上 pmr monotonic 更快，不能把 x86 结论外推到 ARM。

### Region / FixedRegion request scratch

section F 的同一 request workload 每轮执行 4096 次 mixed-size（24–320B）、16B 对齐分配，
然后 bulk clear/release。Win 与 WSL 同机错开运行；M4 为 macOS 26.5.2 / AppleClang 21，
分别显式请求 `USER_INTERACTIVE` 与 `UTILITY` QoS。数字是独立进程的中位 ns/op，表内 ratio
是逐进程配对 throughput ratio 的中位数。

| workload | Windows/MSVC，n=15 | Ubuntu GCC 14.2 / Clang 20.1，均 n=15 | Apple M4 interactive / utility，各 n=31 | 判定 |
|---|---:|---:|---:|---|
| external caller-owned storage：`FixedRegion` / PMR reconstruct | 1.210 / 2.072，**1.682x** | GCC 0.839 / 1.199，**1.432x**；Clang 0.915 / 1.736，**1.899x** | 1.010 / 1.551，**1.537x**（30/31）；1.007 / 1.552，**1.540x**（31/31） | 全平台/编译器 **sign WIN**；Win/Linux spread WIN |
| owned retained blocks：`Region` / PMR release | 1.674 / 1.923，**1.126x** | GCC 1.137 / 2.033，**1.779x**；Clang 1.239 / 1.838，**1.482x** | 2.282 / 2.444，**1.073x**；2.270 / 2.450，**1.083x**（均 31/31） | 全平台/编译器 **sign WIN**；Linux spread WIN |

`Region` 现在缓存当前 ordinary block 的 cursor/end；block 搜索、增长、backing allocation
隔离在 noinline slow path。相同校正 harness 下，Windows 旧实现为 2.196 ns/op、相对 PMR
ratio 0.949（sign LOSE）；新路径降至 1.674 ns/op。`clear()` 仍复用全部 ordinary blocks，
dedicated large block 仍归还 backing；优化没有删除逐次 alignment、hard reserved-byte limit
或 requested-byte accounting。benchmark 对 HYLE/PMR 都以只读指针左值做观测，避免纳秒路径
被 prvalue 临时对象或 GCC read-write asm operand 污染。

M4 是 4 个 Performance + 6 个 Efficiency core，Darwin 没有受支持的硬亲和 API，也不能从
benchmark 内证明实际 core class。harness 因此默认请求 `USER_INTERACTIVE`，并用
`HYLE_BENCH_APPLE_QOS=utility` 复核较低优先级调度；两组相对结论一致。额外的
`taskpolicy -b` background 审计出现 100% 以上绝对 spread，`FixedRegion` 仅 23/31 同侧，
不满足闸门。故这里签的是前台/application QoS 下的 M4 结论，不外推为所有 QoS 或硬绑定
P/E 核的结论。

WSL Clang 20.1 shipping-like 构建固定到 guest CPU 2；三组均 15/15 同侧。WSL 不公开该
virtual CPU 在 Windows 宿主上的实际 P/E core 映射，因此这只消除 guest 内迁移，不写成
物理 P-core 证书。

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

## 4. 外部固定 buffer 的变长分配

2026-07-22 section F；Win/WSL 在同机上错开运行、`n=15`，M4 显式 QoS、各 `n=31`：

| 平台 | HYLE FixedPool | oneTBB fixed_pool | HYLE/oneTBB throughput | 判定 |
|---|---:|---:|---:|---|
| Windows/MSVC Release | 10.314 ns/op | 10.094 ns/op | median 0.983，range 0.912–1.023 | **TIE spread / TIE sign** |
| Ubuntu 24.04/GCC 14.2 Release | 7.555 ns/op | 7.774 ns/op | median 1.027，range 0.948–1.088 | **TIE spread / WIN sign**（13/15，p=7.4e-3） |
| Ubuntu 24.04/Clang 20.1 Release，guest CPU 2 | 6.818 ns/op | 8.700 ns/op | median 1.289，range 1.142–1.373 | **WIN spread / WIN sign**（15/15，p=6.1e-5） |
| Apple M4 / AppleClang 21，interactive | 8.619 ns/op | 9.319 ns/op | median 1.083，range 1.046–1.339 | **TIE spread / WIN sign**（31/31） |
| Apple M4 / AppleClang 21，utility | 8.566 ns/op | 9.316 ns/op | median 1.095，range 1.057–1.185 | **TIE spread / WIN sign**（31/31） |

首轮 HYLE 20.45 ns/op 对 oneTBB 8.66 ns/op 的差距没有通过换 workload 掩盖。路径分解显示
同尺寸稳定复用并非瓶颈，主要成本是 mixed-size free→coalesce→bin 后紧接着
allocate→split→bin 的重复索引维护。实现现在保留立即邻块合并和精确 requested-byte 统计，
但用 one-entry recent-free cache 消除紧邻往返；recent block 不匹配时会先发布回普通 size bin。
alignment/overflow 也改为相对已验证 block end 的 checked arithmetic。benchmark 中 alloc 后还要
free 的指针以 `std::as_const` 作为只读左值观测：GCC 不再把指针变量当 read-write asm operand，
MSVC 也不为 const-pointer prvalue 增加临时对象。Windows 平局，Linux 与 M4 两种显式 QoS
均 sign 胜；M4 的结论边界同上，不冒充硬 P/E 核证书。

## 5. 装饰器与昂贵上下文复用

同一 section F；Windows/WSL Clang 为 `n=15`，M4 为 interactive QoS `n=31`：

| workload | Windows/MSVC | Ubuntu/Clang 20.1 | Apple M4/AppleClang 21 | 结论 |
|---|---:|---:|---:|---|
| strict requested-byte admission：`QuotaAllocator` / direct system | 22.181 / 9.144 ns/op | 19.529 / 6.004 ns/op | 9.498 / 7.680 ns/op | 可组合 hard admission 有明确原子 accounting 成本；不替代裸 allocator |
| expensive reusable context loan：`PreservingObjectPool` / reconstruct | 20.894 / 266.230 ns/op，**12.7x** | 18.173 / 93.788 ns/op，**5.2x** | 8.900 / 70.247 ns/op，**7.9x** | 构造昂贵、归还后保留状态的 context 值得复用 |

quota 行体现的是可组合 hard admission 的价格，不应拿来替代裸 allocator；Preserve 行对应
codec/session 等昂贵对象“构造一次、借还保留状态”的真实生命周期。

## 6. 选型

```text
默认小对象             → SystemAllocator
运行时注入/mock         → vtable Allocator（接受对裸 mi 的个位数 % 税）
单线程 borrowed/owned bump → PooledRegion / StackAllocator（M4 上 pmr 可能更快）
单线程同尺寸可增长       → GrowablePool
多线程、每线程独享       → GrowablePool × N
共享固定块、严格配额     → SharedPool / ObjectPool<T>
共享固定块、吞吐优先且有余量 → SharedPoolCached / ObjectPoolCached<T>
通用多线程 malloc       → mimalloc / oneTBB；不要硬套固定 slab
```

## 7. 复跑

```bash
# shipping-like bench；使用仓库同级 ../oneTBB 本地源码快照
cmake --preset bench -DHYLE_BENCH_TBB=ON
cmake --build --preset bench --target hyle_mem_scenarios_bench

export HYLE_BENCH_SECTIONS=DE
export HYLE_BENCH_THREADS=8
export HYLE_BENCH_BLOCK_SIZE=64
export HYLE_BENCH_BATCH=64
export HYLE_BENCH_CAPACITY_PCT=75
export HYLE_BENCH_HANDOFF_BATCH=256
export HYLE_BENCH_QUOTA=1024
export HYLE_BENCH_REQUEST_PCT=110

scripts/bench-stat.py build-bench/bin/hyle_mem_scenarios_bench -n 9 \
  --pair "oneTBB fixed_pool (fixed-buffer shared)" \
         "HYLE SharedPool (hard capacity)" \
  --pair "oneTBB fixed_pool (fixed-buffer shared)" \
         "HYLE SharedPoolCached (soft capacity)"

# Win/Linux 使用 n=15；M4 默认 USER_INTERACTIVE，正式复跑把这里改为 n=31
HYLE_BENCH_SECTIONS=F \
scripts/bench-stat.py build-bench/bin/hyle_mem_scenarios_bench -n 15 \
  --pair "std::pmr::monotonic_buffer_resource (external, reconstruct)" \
         "HYLE FixedRegion (external, no fallback)" \
  --pair "std::pmr::monotonic_buffer_resource (release upstream blocks)" \
         "HYLE Region (retained ordinary blocks)" \
  --pair "oneTBB fixed_pool (thread-safe, hard buffer)" \
         "HYLE FixedPool (single-thread, hard buffer)"

# Apple M4：默认 USER_INTERACTIVE；再以 UTILITY 做较低优先级调度复核
HYLE_BENCH_SECTIONS=F HYLE_BENCH_APPLE_QOS=utility \
scripts/bench-stat.py build-bench/bin/hyle_mem_scenarios_bench -n 31 \
  --pair "std::pmr::monotonic_buffer_resource (external, reconstruct)" \
         "HYLE FixedRegion (external, no fallback)" \
  --pair "std::pmr::monotonic_buffer_resource (release upstream blocks)" \
         "HYLE Region (retained ordinary blocks)" \
  --pair "oneTBB fixed_pool (thread-safe, hard buffer)" \
         "HYLE FixedPool (single-thread, hard buffer)"

# quota 计数是单次语义证据；不把其中线程启动时间当吞吐比较
HYLE_BENCH_SECTIONS=E build-bench/bin/hyle_mem_scenarios_bench
```

Linux/WSL 必须从 ext4 工作副本运行；Win、WSL 同机时必须错开，不得并发采样。Apple M4 使用本地 APFS 工作副本。

## 8. Request 尾延迟证书

section G 与 section F 使用相同的 4096 次 mixed-size request，但每次只在整个 request
边界读取一次 `steady_clock`。报告的是 **ns/request** 的 nearest-rank p50/p95/p99，不是给每次
allocation 插时钟，也不是把 nanobench epoch 当用户延迟。每个样本内 HYLE/对照交替先后顺序，
降低固定执行顺序与温漂偏差；正式证书至少 4096 samples、15 个独立进程，M4 延续 QoS 审计。

2026-07-22 clean commit `4490c40` 正式证书如下。数字均为
`HYLE / reference` 的 p50/p95/p99，单位 **µs/request**；ratio 是逐进程配对
`reference/HYLE` 的中位数，三项依次对应 p50/p95/p99：

| 平台 / QoS | request shape | HYLE p50/p95/p99 | reference p50/p95/p99 | paired ratio | 判定 |
|---|---|---:|---:|---:|---|
| Win / MSVC 19.50 | external `FixedRegion` / PMR reconstruct | 4.900 / 4.900 / 7.000 | 8.200 / 8.300 / 11.900 | 1.673 / 1.694 / 1.694 | **WIN / WIN / WIN** |
| Win / MSVC 19.50 | owned `Region` / PMR release | 6.800 / 6.800 / 9.100 | 8.300 / 8.400 / 11.700 | 1.221 / 1.235 / 1.316 | **WIN / WIN / WIN** |
| Win / MSVC 19.50 | `FixedPool` / oneTBB fixed_pool | 31.800 / 36.600 / 60.800 | 37.300 / 41.600 / 66.500 | 1.171 / 1.121 / 1.103 | **WIN / WIN / WIN** |
| Linux / Clang 20.1 CPU 2 | external `FixedRegion` / PMR reconstruct | 4.142 / 4.164 / 5.751 | 7.700 / 7.816 / 12.711 | 1.860 / 1.878 / 2.365 | **WIN / WIN / WIN** |
| Linux / Clang 20.1 CPU 2 | owned `Region` / PMR release | 5.408 / 5.435 / 8.568 | 8.188 / 8.907 / 14.505 | 1.514 / 1.633 / 1.864 | **WIN / WIN / WIN** |
| Linux / Clang 20.1 CPU 2 | `FixedPool` / oneTBB fixed_pool | 30.201 / 35.406 / 57.118 | 38.170 / 44.483 / 71.844 | 1.277 / 1.270 / 1.194 | **WIN / WIN / WIN** |
| Apple M4 / interactive | external `FixedRegion` / PMR reconstruct | 3.666 / 3.708 / 3.750 | 6.375 / 6.459 / 6.541 | 1.739 / 1.742 / 1.733 | **WIN / WIN / WIN** |
| Apple M4 / interactive | owned `Region` / PMR release | 5.667 / 5.791 / 5.833 | 9.959 / 10.084 / 10.167 | 1.757 / 1.741 / 1.750 | **WIN / WIN / WIN** |
| Apple M4 / interactive | `FixedPool` / oneTBB fixed_pool | 31.250 / 31.666 / 33.917 | 38.042 / 38.916 / 41.375 | 1.219 / 1.228 / 1.222 | **WIN / WIN / WIN** |
| Apple M4 / utility | external `FixedRegion` / PMR reconstruct | 3.666 / 3.708 / 3.750 | 6.375 / 6.458 / 6.500 | 1.739 / 1.742 / 1.733 | **WIN / WIN / WIN** |
| Apple M4 / utility | owned `Region` / PMR release | 5.667 / 5.791 / 5.833 | 9.958 / 10.084 / 10.208 | 1.757 / 1.741 / 1.750 | **WIN / WIN / WIN** |
| Apple M4 / utility | `FixedPool` / oneTBB fixed_pool | 31.167 / 31.583 / 35.625 | 38.167 / 39.041 / 42.917 | 1.224 / 1.236 / 1.207 | **WIN / WIN / WIN** |

原始逐进程 percentile：
[Windows CSV](./mem-request-latency-win.csv)、
[Linux CSV](./mem-request-latency-linux.csv)、
[M4 interactive CSV](./mem-request-latency-m4.csv)、
[M4 utility CSV](./mem-request-latency-m4-utility.csv)。四份文件的所有行均为同一 commit、
`dirty=0`、`qos_ok=1`；Windows/Linux 各 15 process，M4 每种 QoS 各 31 process。M4 raw
行分别记录 `qos=user-interactive/utility` 与 `affinity=unsupported`，不冒充硬 P/E 核绑定。

```bash
# Linux：固定同一逻辑 CPU
python3 scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench \
  -n 15 --samples 4096 --cpu 2 \
  --raw docs/performance/mem-request-latency-linux.csv

# macOS：无受支持的硬亲和 API，延续 USER_INTERACTIVE / UTILITY 两组 QoS 审计
python3 scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench \
  -n 31 --samples 4096 --raw docs/performance/mem-request-latency-m4.csv
HYLE_BENCH_APPLE_QOS=utility \
  python3 scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench \
  -n 31 --samples 4096 --raw docs/performance/mem-request-latency-m4-utility.csv

# Windows：oneTBB 的运行时 DLL 在 build tree 的配置目录中
python scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench.exe \
  -n 15 --samples 4096 \
  --dll-dir build-bench/msvc_19.50_cxx23_64_md_release \
  --raw docs/performance/mem-request-latency-win.csv
```

脚本保存每个 process/variant 的原始 percentile、HEAD 与 dirty 标记，并对每个 percentile 做
paired reference/HYLE ratio 与 sign test。dirty run 只允许调试 harness，不进入正式表。
pre-fix 诊断曾暴露 `FixedPool` 平台翻转；A/B 定位为编译器没有稳定展开 header-only
allocation/free 的嵌套 bin hot path。完整 hot graph 强制内联后，上述 clean 证书三平台、两种
M4 QoS 的 p50/p95/p99 已全部胜 oneTBB。

## 9. Request footprint / fragmentation 证书

section H 验证 owned request scratch 的空间行为，不把一个短循环的 allocator 调用次数冒充
真实 footprint。单次峰值 request 默认写入 128 MiB payload：约 80% 是 24B–64KiB 的普通
parser/graph scratch，约 20% 是偶发 2 MiB staging buffer；`clear/release` 后再写入 1/4 大小的
request，最后 full release。HYLE `Region` 与 `std::pmr::monotonic_buffer_resource` 都由同一个
bench-local mimalloc vtable backing，排除不同 system heap 的影响。

首轮 128 MiB dirty A/B 发现旧默认 1 MiB block 在 Linux 形成大量 backing allocation：虽然
`reserved/payload=1.007`，live RSS 却约 203 MiB；2/4/16 MiB 上限依次约 172/159/157 MiB，
同轮 PMR 约 151 MiB。4 MiB 已获得绝大部分 RSS 收益且 `reserved/payload` 仅为 1.031；16 MiB
只再省约 2 MiB RSS，却把精确 reservation 推到 1.062。Windows 4 MiB 的 live RSS 约 136 MiB，
PMR 约 135 MiB。因此产品默认上限改为 4 MiB；`--block-mib` 保留作诊断参数，正式证书只签
当前默认。以上 dirty A/B 是设计依据，不是发布表。

证书把两个概念分开：

- `peak_reserved_bytes / requested_bytes` 是 request live 时的 block slack、header 和 alignment
  总开销，可用于判断内部 fragmentation；HYLE 直接读取 `Region::reserved()`，PMR 通过
  `StatsAllocator -> PmrResource` 统计其 upstream block 请求。
- `retained_after_clear_bytes` 是生命周期策略：`Region::clear()` 有意保留普通块并归还 dedicated
  大块，PMR `release()` 归还全部 upstream block。保留量用于下一 request 复用，不等同于 live
  fragmentation；最后的 `released_bytes` 必须为 0。
- RSS 来自 Windows working set、Linux `/proc/self/statm` 和 Darwin task info。记录 baseline、
  live、clear、reuse、full release 与 OS high-water 原值；RSS 是 OS 页驻留观察，不替代精确
  backing 计数，也不要求每个平台在 free 后同步 trim working set。

每个子进程只运行一个 allocator variant，脚本逐轮反转 HYLE/PMR 子进程顺序，避免第一个实现的
页和 allocator cache 污染第二个实现。脚本绑定 HEAD/dirty、compiler/platform、Linux CPU 与
Apple QoS，并可把每个 child 的原始字段写入本目录 CSV。dirty 的 Win/Linux 8 MiB smoke 已验证
编译、解析、`released_bytes=0` 和 RSS 路径；正式三平台 128 MiB clean 数据尚未签入，因此这里
暂不发布比较表。

```bash
# Linux：每个 variant 是独立进程，并固定同一 guest CPU
python3 scripts/mem-footprint-stat.py \
  build-bench/bin/hyle_mem_scenarios_bench -n 15 --mib 128 --cpu 2 \
  --raw docs/performance/mem-request-footprint-linux.csv

# macOS：无硬亲和 API；和 latency 证书一样分别跑 interactive / utility
python3 scripts/mem-footprint-stat.py \
  build-bench/bin/hyle_mem_scenarios_bench -n 31 --mib 128 \
  --raw docs/performance/mem-request-footprint-m4.csv

# Windows：按实际 build tree 补 oneTBB DLL 目录
python scripts/mem-footprint-stat.py \
  build-bench/bin/hyle_mem_scenarios_bench.exe -n 15 --mib 128 \
  --dll-dir build-bench/msvc_19.50_cxx23_64_md_release \
  --raw docs/performance/mem-request-footprint-win.csv
```
