# mem 实测铁证

> 本页是 [performance 总账](./README.md) 的 **mem 模块**详节。
> 源：`bench/mem/allocator_bench.cc`、`bench/mem/scenario_bench.cc`、`bench/mem/fiber_stack_bench.cc`。
> 通用 churn 数据来自 2026-07-20 的 Win MSVC / WSL GCC 14 / Apple M4 AppleClang 实测；`SharedPool` / `SharedPoolCache` 绑定 clean commit `8cf52f9`。request closure、FixedPool、latency、fragmentation 和 footprint 按各节所列 commit。2026-07-23 的 MEM-FINAL-2 bump/realloc/adapter/fiber 证书统一绑定 clean product commit `2cbc307`：Win/WSL `n=15`；M4 为 `n=31`，bump/fiber 跑 interactive+utility，realloc/adapter 跑 interactive。同契约性能判定信任配对 **[sign]**。最终候选 `551be0b` 相对该提交的产品差异只有 `hugepage.h` 的直接标准头依赖，不改变受测实现；其余是文档/认证工具。raw/result 继续诚实绑定实际采样 commit，不重写 HEAD。

## 当前性能封口状态

本页独占 mem 的性能现状；architecture 和 guide 不复制点值。已签结果按各节标注的 commit
有效，不因 API-only commit 自动失效。MEM-FINAL-2 的四个有界证据缺口已关闭：

| 原缺口 | `2cbc307` 关闭证据 | 结论 |
|---|---|---|
| M4 PooledRegion/Stack | 对称 8192×64B batch；interactive 为 1.023x / 1.023x，utility 为 1.028x / 1.026x，均对 PMR sign WIN | 已关闭；旧 0.67x 作废 |
| realloc 分层 | direct `SystemAllocator` 对 raw mi：Win 1.005x TIE、Linux 0.989x TIE、M4 1.018x WIN；runtime vtable 单列 | 默认路径同档；x86 动态多态税显式可见 |
| adapter composition | direct、Quota、counters-only Stats、Quota→Stats 三平台独立量化 | 默认路径零装饰税；策略成本由用户按需购买 |
| FiberStackPool | 真实 `Scheduler` 每 batch 创建/运行/销毁 32 fibers，缓存与直接 guarded VM 比较 p50/p95/p99 | 三平台、M4 双 QoS 全 percentile sign WIN |

`SharedPool` hard core 的 immediate-global-free/精确容量合同强于 owner-local free list；它不以
击败较弱合同的 oneTBB local churn 为冻结条件。吞吐形状由显式 `SharedPoolCache` 对拍，hard
core 仍报告绝对成本和较弱合同参照。

## 1. 通用路径（`hyle_mem_bench`）

| 场景 | Win | WSL | M4 | 判定 |
|---|---|---|---|---|
| **SystemAllocator 内联** vs 裸 mi churn | 7.93 / 7.86 → **0.996 TIE** | 8.04 / 8.04 → **0.99 TIE** | 7.77 / 7.91 → **1.00 TIE** | ➖ 三平台平 mimalloc |
| **vtable** vs 裸 mi churn | 8.67 / 7.86 → **0.91 LOSE sign** | 8.67 / 8.04 → **0.94 LOSE sign** | 7.93 / 7.91 → **1.00 TIE** | Win/WSL 有约 6–9% 多态税 |
| **vtable** vs pmr(over mi) churn | 8.67 / 8.67 → **TIE** | 8.67 / 10.27 → **1.22 WIN sign** | 7.93 / 9.98 → **1.26 WIN** | 同类多态对照平或赢 pmr |
| direct **SystemAllocator reallocate** / raw `mi_realloc` | 66.979 / 67.691 → **1.005 TIE** | 79.279 / 78.995 → **0.989 TIE** | 81.643 / 83.577 → **1.018 WIN sign** | 默认编译期路径三平台同档 |
| runtime **Allocator vtable reallocate** / raw `mi_realloc` | 74.922 / 67.691 → **0.906 LOSE sign** | 86.344 / 78.995 → **0.921 LOSE sign** | 81.329 / 83.577 → **1.020 WIN sign** | x86 有约 8–9% 动态多态税；M4 无税 |

默认内联路径平 mimalloc；需要运行时注入的 vtable 路径对裸 mi 有可测税，但仍平或赢同为多态封装的 pmr。

## 2. Bump 与单线程定长对象

### PooledRegion / StackAllocator

| | Win | WSL | M4 |
|---|---:|---:|---:|
| PooledRegion ns | 0.568 | 0.345 | 1.176 / 1.176 |
| StackAllocator ns | 0.571 | 0.342 | 1.175 / 1.176 |
| pmr monotonic ns | 1.483 | 0.648 | 1.204 / 1.209 |
| Pooled / pmr throughput | **2.594x WIN** | **1.809x WIN** | **1.023x / 1.028x WIN sign** |
| Stack / pmr throughput | **2.638x WIN** | **1.849x WIN** | **1.023x / 1.026x WIN sign** |

M4 旧证书的约 0.67x 不是大小核结论，而是通用 alignment/overflow 检查留在每次已对齐
成功分配的串行依赖上，并且 `PooledRegion` 的 invalid-state 检查位于返回值之后。当前实现以
已对齐游标为常用 fast path、非对齐/溢出走完整 checked path，并把 invalid region 前置；
Windows/Linux 大幅胜 PMR，M4 interactive/utility 都与 libc++ PMR 同档并稳定同侧。Darwin
仍不冒充硬 P/E core 绑定，但双 QoS 已反证旧 lose 不是应保留的平台边界。

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
隔离在 noinline slow path。`clear()` 复用全部 ordinary blocks，dedicated large block 归还
backing；逐次 alignment、hard reserved-byte limit 与 requested-byte accounting 均保留。
benchmark 对 HYLE/PMR 都以只读指针左值做观测，避免纳秒路径被 prvalue 临时对象或 GCC
read-write asm operand 污染。

M4 是 4 个 Performance + 6 个 Efficiency core，Darwin 没有受支持的硬亲和 API，也不能从
benchmark 内证明实际 core class。harness 因此默认请求 `USER_INTERACTIVE`，并用
`HYLE_BENCH_APPLE_QOS=utility` 复核较低优先级调度；两组相对结论一致。额外的
`taskpolicy -b` background 审计出现 100% 以上绝对 spread，`FixedRegion` 仅 23/31 同侧，
不满足闸门。故这里签的是前台/application QoS 下的 M4 结论，不外推为所有 QoS 或硬绑定
P/E 核的结论。

WSL Clang 20.1 shipping-like 构建固定到 guest CPU 2；三组均 15/15 同侧。WSL 不公开该
virtual CPU 在 Windows 宿主上的实际 P/E core 映射，因此这只消除 guest 内迁移，不写成
物理 P-core 证书。

### Region 真实消费者：HTTP request lifecycle

2026-07-23 首个模块外消费者把 `Region` 接入 `http_serve`：连接 parser/buffer 继续使用
长生命周期 backing allocator，每个 handler 收到 request-scoped `Region::as_allocator()`；
response body/header view、stream generator state 和 handler coroutine frame 一直存活到发送
完成，随后 frame 销毁并 bulk clear。TCP/TLS 共用同一实现，Windows/MSVC 与 WSL/GCC 14
的 mem + HTTP + TLS 回归均通过。

真实场景同时暴露了默认 64 KiB/连接和异常请求高水位长期驻留的问题，因此 HTTP 没有
照搬通用默认值：`HttpServeOptions` 默认 4 KiB 起步、ordinary block 最大 64 KiB、
`>=64 KiB` dedicated allocation 在 clear 时立即归还；clear 后 ordinary backing 超过
256 KiB 则全部 release。四个值都可由用户调整，`retain_limit=0` 表示请求间不保留。

吞吐用 `hyle_http_bench` 的真实 64 connections × 64 keep-alive GET/response workload；
Windows MSVC 19.50 Release，同机交替运行 current Region 路径和 clean `2fd7187` 的
handler-direct-system-allocator 基线，各 `n=15`。ratio 是 baseline ns / Region ns，亦即
Region / baseline throughput。1/2/4 workers 全部在噪声范围内；8 workers 的 benchmark
本身仍标记 unstable，只记录数据、不签稳定结论。

| workers | direct baseline median | request Region median | throughput ratio median（range） | 判定 |
|---:|---:|---:|---:|---|
| 1 | 11,298.66 ns/request | **11,068.09 ns/request** | 1.0107x（0.9491–1.0743） | **TIE** |
| 2 | 7,803.30 ns/request | **7,748.29 ns/request** | 1.0043x（0.9629–1.0397） | **TIE** |
| 4 | **4,271.87 ns/request** | 4,352.44 ns/request | 0.9894x（0.9509–1.0146） | **TIE** |
| 8 | **4,208.84 ns/request** | 4,221.53 ns/request | 1.0137x（0.7965–1.2192） | unstable / 不判定 |

这项证据的结论不是“arena 微基准更快”，而是更强的产品结论：真实 request ownership、
streaming lifetime、keep-alive reuse 和有界 idle retention 可以同时成立，且当前小响应整链路
没有可测吞吐税。

#### 长连接高水位 / RSS 诊断

`hyle_http_bench` 另有 fresh-process footprint 模式：128 条真实 keep-alive 连接同步执行
small warmup → 每连接 1 MiB request-local spike（8 KiB ordinary allocations）→ small reuse →
close。每个阶段等所有连接到达、server 已 clear/release 且静置 25 ms 后，同时读取 backing
vtable 的精确 outstanding bytes 与 OS RSS。比较默认 256 KiB cap、同 cap 后显式
`system_collect(true)`、以及 unbounded retention；脚本交替三种子进程顺序并报告 p50/p99。

正式签表前的参数诊断否定了两个错误方向：提高 Region max block 到 256 KiB/1 MiB 会把该
workload 的 backing peak 从约 154 MiB 提高到约 162/258 MiB，且不改善 Linux RSS；仅调用
`Region::release()` 也不能保证 system heap 立刻 purge OS pages。

2026-07-23 clean `17ca3c3` 的正式证书为 Windows/MSVC 19.50 与 WSL/GCC 14 CPU 2 各
`n=15` fresh processes，所有 90 行 raw 数据均为 `dirty=0`：

| 平台 | 策略 | spike 后精确 backing | reuse RSS Δ vs warm p50 / p99 | 显式 collect p50 / p99 |
|---|---|---:|---:|---:|
| Windows | 256 KiB cap | **1.86 MiB** | 2.33 / 22.14 MiB | — |
| Windows | cap + force collect | **1.86 MiB** | **1.11 / 1.12 MiB** | 8.76 / 9.76 ms |
| Windows | unbounded | 153.37 MiB | 140.05 / 140.05 MiB | — |
| Linux CPU 2 | 256 KiB cap | **1.63 MiB** | 154.00 / 154.00 MiB | — |
| Linux CPU 2 | cap + force collect | **1.63 MiB** | **-1.03 / -1.03 MiB** | 0.252 / 0.297 ms |
| Linux CPU 2 | unbounded | 153.14 MiB | 154.00 / 154.00 MiB | — |

负 RSS delta 表示 force collect 顺带清掉了 warmup 前已有的当前线程 heap cache，不表示负内存。
三层结论必须分开：cap 把 HYLE/Region 的 ownership 确定地降低约 98.8%；OS RSS 还受 backing
allocator purge 时机控制；显式 force collect 可把 RSS 尾部收紧，但 Windows 约 8 ms 的成本
证明它只能用于用户选择的 idle/pressure 冷路径，不能偷偷插进每个 request。因此新增
`system_collect(bool force=false)`，保持热 `Allocator` 三函数 vtable 不变，并明确 worker pool
需要在各自 worker 上执行。

原始逐进程字段：[Windows CSV](./http-region-footprint-win.csv)、
[Linux CSV](./http-region-footprint-linux.csv)。两份都是 clean `17ca3c3`、每平台 15×3 行；
包含各阶段 RSS/backing 原值而不只保存聚合结果。

```bash
# Windows（fresh process，三 variant 交替）
python scripts/http-region-footprint-stat.py \
  build-bench/bin/hyle_http_bench.exe -n 15 --connections 128 \
  --spike-kib 1024 --block-kib 64 \
  --dll-dir build-bench/msvc_19.50_cxx23_64_md_release \
  --raw docs/performance/http-region-footprint-win.csv

# Linux / WSL（同一逻辑 CPU）
python3 scripts/http-region-footprint-stat.py \
  build-wsl-http-bench/bin/hyle_http_bench -n 15 --connections 128 \
  --spike-kib 1024 --block-kib 64 --cpu 2 \
  --raw docs/performance/http-region-footprint-linux.csv
```

### GrowablePool

| | Win | WSL | M4 |
|---|---:|---:|---:|
| GrowablePool ns | 2.74 | 1.13 | 2.46 |
| pmr unsync ns | 5.81 | 11.93 | 3.74 |
| 裸 mi ns | 4.87 | 4.45 | 5.80 |
| vs pmr | **约 2.1x WIN** | **约 10x WIN** | **约 1.5x WIN** |
| vs mi | **约 1.8x WIN** | **约 3.9x WIN** | **约 2.4x WIN** |

单线程定长对象场景三平台都赢公平标尺；多线程独享池用 `GrowablePool × N`，不要把单个非线程安全实例共享。

## 3. SharedPool hard core 与显式 owner-local cache

| API | 容量契约 | 回收可见性 | 适用场景 |
|---|---|---|---|
| `SharedPool` / `ObjectPool<T>` | **硬容量** | `deallocate()` 后立即可被任意线程复用 | 服务配额、严格并发上限、容量紧张的跨线程共享 |
| `SharedPoolCache` / `ObjectPoolCache<T>` | **软视图** | free indices 暂存在显式 owner cache；`publish()`/析构后公开 | 有容量余量的 request/context 批次、高频 owner-local 复用 |

cache capacity 与 transfer batch 都是构造参数；128/32 只是 convenience 默认值。caller-owned `span<uint32_t>` 可让 cache metadata 零隐藏分配。这个窗口是明确的吞吐/容量交换，不是 `SharedPool` 的实现细节。

### 3.1 Workload 来自实际用户形状

基准先对齐成熟库和消费者，再扫描参数，不以某个内部阈值反推 workload：

- **request batch lifecycle**：worker 一次申请一批 request/context 对象，处理后整批释放；对应 Folly `IndexedMemPool` / Compression context cache 和 oneTBB fixed-pool 本地批次。
- **producer → neighbor consumer handoff**：生产者申请，邻居消费者释放；对应 oneTBB cross-thread-free、mimalloc Larson/xmalloc migration。轮次用 producer/consumer sequence 串成流水线，没有每轮全局 barrier。
- **service quota**：所有 worker 同时申请超过固定容量的对象；对应 bounded service/marl borrow 配额，验证的是准入和回收语义，不拿启动时间宣称吞吐。

oneTBB `fixed_pool` 是成熟的**固定 buffer 吞吐参照**：双方都从调用者提供的有限
buffer 取 payload；fixed_pool 的 buffer 是与 HYLE 相同的 payload capacity，加 oneTBB
自己 fixed-pool 测试采用的 1 MiB bootstrap metadata allowance，不再额外给 2 倍 payload。
但它不是 `SharedPool` 的同契约对手。源码审计绑定 oneTBB `f48775ba`：64B 小对象从
tbbmalloc 的 owner-thread `freeList`/bump pointer 分配；同 owner free 非原子地回私有链，
remote free 才 CAS 到 `publicFreeList`，随后由 owner 批量 privatize。因此 fixed buffer
限制了后端扩容，却不保证 HYLE hard core 的 completed-free 全线程立即可见性。它更接近
`SharedPoolCache` 的局部性形状，只是 locality policy 隐藏在库内 TLS 中。

每次 fixed-pool/HYLE/mimalloc/TBB 分配都检查非 NULL，任何少做工作会使基准直接失败。
pmr resource 与其他池一样在计时 lambda 外构造。`scalable_allocator`、mimalloc 和 pmr
是 growable 参考；oneTBB fixed_pool 证明固定 buffer 下的成熟吞吐水平，但这些行都不能
证明 `SharedPool` 的硬可见性语义。hard core 的性能目标是把其额外同步成本压到合理范围，
而不是把 weaker-visibility 数字误标成同契约 SOTA。

### 3.2 当前显式 cache：Windows / WSL / M4 clean 证书

绑定 clean commit `8cf52f9`。三平台均为 Release shipping-like、8 workers、64B block、
request batch=64、working set=容量的 75%、handoff batch=256。Win MSVC 19.50 与 WSL
GCC 14.2 为 `n=15`，Apple M4 AppleClang 21 的 `USER_INTERACTIVE` / `UTILITY` 各
`n=31`。ratio 是每次进程内配对的 **HYLE / oneTBB op/s** 后再跨进程取中位数。
Darwin 无公开硬亲和 API，两组只证明 application QoS 下的结论。

| 平台 / 编译器 | 场景 | `SharedPoolCache` | oneTBB `fixed_pool` | HYLE/oneTBB throughput | 判定 |
|---|---|---:|---:|---:|---|
| Windows / MSVC 19.50 | request batch | **1.132 ns/op** | 2.060 ns/op | **1.837x**，range 1.641–2.005 | **WIN spread / WIN sign**（15/15，p=6.1e-05） |
| Windows / MSVC 19.50 | handoff | **1.414 ns/op** | 9.898 ns/op | **6.972x**，range 4.295–7.403 | **WIN spread / WIN sign**（15/15，p=6.1e-05） |
| WSL / GCC 14.2 | request batch | **1.186 ns/op** | 1.651 ns/op | **1.396x**，range 1.268–1.536 | spread TIE / **WIN sign**（15/15，p=6.1e-05） |
| WSL / GCC 14.2 | handoff | **1.785 ns/op** | 6.502 ns/op | **3.835x**，range 3.126–5.649 | **WIN spread / WIN sign**（15/15，p=6.1e-05） |
| Apple M4 / interactive | request batch | **0.888 ns/op** | 2.075 ns/op | **2.339x**，range 2.231–2.434 | **WIN spread / WIN sign**（31/31，p=9.3e-10） |
| Apple M4 / interactive | handoff | **1.791 ns/op** | 5.054 ns/op | **2.804x**，range 2.530–3.019 | **WIN spread / WIN sign**（31/31，p=9.3e-10） |
| Apple M4 / utility | request batch | **0.912 ns/op** | 2.116 ns/op | **2.311x**，range 2.212–2.402 | **WIN spread / WIN sign**（31/31，p=9.3e-10） |
| Apple M4 / utility | handoff | **1.819 ns/op** | 5.051 ns/op | **2.793x**，range 2.537–3.066 | **WIN spread / WIN sign**（31/31，p=9.3e-10） |

显式 cache 的 locality 形状与 oneTBB fixed_pool 接近；Windows、WSL 与 M4 两种 QoS
全部配对同侧。它不是硬容量视图，性能优势不能替代 3.4 的 quota 语义。

### 3.3 当前 hard core：三平台 clean 证书

绑定 clean commit `8cf52f9`，Release shipping-like（`HYLE_BUILD_TESTS=OFF`）；配置与
3.2 相同。Win MSVC 19.50、WSL GCC 14.2 为 `n=15`，Apple M4 AppleClang 21 的
interactive/utility 各 `n=31`。本次只收紧 Treiber 发布/消费所需的内存序：push 用
release 发布 relaxed link，pop 用 acquire 消费；pop 不发布写入，push retry 也不读取
旧节点内容，因此不需要 acq_rel/acquire。tag 饱和后的 serialized fallback 未改变。

hard 与 fixed_pool 的可见性合同不同，因此下表只量化更强合同的成本，不做 WIN/LOSE
或 SOTA verdict。request batch 中，oneTBB 的 allocate/free 都命中 owner 私有链，通常没有
共享 RMW；HYLE hard 的 allocate 与 deallocate 各做一次共享 CAS。handoff 中 oneTBB 通常是
本地 allocate + 一次 remote-free CAS，HYLE 仍是两端各一次 CAS。M4 对共享 atomic RMW
更敏感，所以差距比 x86 明显；x86 上 HYLE 的简单分片链又可能抵消 tbbmalloc 元数据成本。

| 平台 / 编译器 | 场景 | `SharedPool` hard | oneTBB `fixed_pool` | HYLE/oneTBB throughput（成本参照） |
|---|---|---:|---:|---:|
| Windows / MSVC 19.50 | request batch | 6.708 ns/op | 2.060 ns/op | 0.308x |
| Windows / MSVC 19.50 | handoff | **4.150 ns/op** | 9.898 ns/op | **2.326x** |
| WSL / GCC 14.2 | request batch | 6.506 ns/op | 1.651 ns/op | 0.257x |
| WSL / GCC 14.2 | handoff | **4.093 ns/op** | 6.502 ns/op | **1.569x** |
| Apple M4 / interactive | request batch | 8.307 ns/op | 2.075 ns/op | 0.251x |
| Apple M4 / interactive | handoff | 10.286 ns/op | 5.054 ns/op | 0.492x |
| Apple M4 / utility | request batch | 5.969 ns/op | 2.116 ns/op | 0.351x |
| Apple M4 / utility | handoff | 10.260 ns/op | 5.051 ns/op | 0.508x |

utility request 比 interactive 快，说明无公开硬亲和 API 时绝对数会受 macOS 调度影响；
不把其中任一组外推成固定 P/E core 结论。

如果用户不需要 completed-free 立即对全线程可见，应直接选择 3.2 的显式 cache；不能通过
给 hard core 偷加隐藏 TLS 来抹掉这项同步成本。

### 3.4 硬容量证据

当前显式 cache 的 Windows sanity 配置：8 workers、64B block、capacity=1024、请求 110%（1128 次）：

| 池 | admitted | failed | publish 前可立即重获 | publish 后可重获 |
|---|---:|---:|---:|---:|
| `SharedPool` | **1024** | **104** | **1024** | **1024** |
| `SharedPoolCache`（128/32） | 929 | 199 | 160 | **1024** |

硬池精确准入 capacity，释放后立即恢复完整容量。显式 cache 的 initial admission 与 publish 前可见数受 batch reservation/调度影响，表中只作语义 sanity，不是稳定性能计数；publish 后必须恢复 1024。因此容量紧张的 quota 场景必须选 hard core，不能用 cache 的更快吞吐替代契约。

## 4. 外部固定 buffer 的变长分配

2026-07-22 section F；Win/WSL 在同机上错开运行、`n=15`，M4 显式 QoS、各 `n=31`：

| 平台 | HYLE FixedPool | oneTBB fixed_pool | HYLE/oneTBB throughput | 判定 |
|---|---:|---:|---:|---|
| Windows/MSVC Release | 6.359 ns/op | 8.297 ns/op | median 1.305，range 1.161–1.356 | **WIN spread / WIN sign**（15/15） |
| Ubuntu 24.04/GCC 14.2 Release | 5.760 ns/op | 7.788 ns/op | median 1.383，range 1.283–1.516 | **WIN spread / WIN sign**（15/15） |
| Ubuntu 24.04/Clang 20.1 Release，guest CPU 2 | 5.787 ns/op | 9.656 ns/op | median 1.670，range 1.526–1.778 | **WIN spread / WIN sign**（15/15） |
| Apple M4 / AppleClang 21，interactive | 5.868 ns/op | 9.318 ns/op | median 1.607，range 1.543–1.721 | **WIN spread / WIN sign**（31/31） |
| Apple M4 / AppleClang 21，utility | 5.873 ns/op | 9.648 ns/op | median 1.668，range 1.544–1.798 | **WIN spread / WIN sign**（31/31） |

当前实现用 one-entry free cursor 同时承接 recent return 与 split remainder；相邻释放直接与
cursor 合并，非相邻或不匹配时才发布回普通 size bin，同时保留立即邻块合并和精确
requested-byte 统计。alignment/overflow 使用相对已验证 block end 的 checked arithmetic。
benchmark 中 alloc 后还要
free 的指针以 `std::as_const` 作为只读左值观测：GCC 不再把指针变量当 read-write asm operand，
MSVC 也不为 const-pointer prvalue 增加临时对象。clean `d3e4214` 上 Windows、GCC、Clang 与
M4 两种显式 QoS 均通过 spread/sign 双闸门；M4 的结论边界同上，不冒充硬 P/E 核证书。

## 5. 装饰器、昂贵上下文与 fiber stack

### Allocator decoration 与昂贵上下文复用

MEM-FINAL-2 adapter-only section F 绑定 clean `2cbc307`。每个 operation 是 64B
allocate + free；Windows/MSVC 与 Linux/GCC 14 固定 CPU 2 为 `n=15`，M4 interactive
为 `n=31`。单位 ns/op：

| workload | Windows/MSVC | Linux/GCC 14.2 | Apple M4/AppleClang 21 | 结论 |
|---|---:|---:|---:|---|
| direct system | **8.339** | **7.198** | **7.815** | 默认路径，不承担 adapter 成本 |
| `QuotaAllocator -> system` | 21.712 | 20.907 | 9.750 | strict requested-byte admission 的原子 accounting |
| counters-only `StatsAllocator -> system` | 30.015 | 29.291 | 10.518 | 计数/峰值；不含 leak map |
| `QuotaAllocator -> StatsAllocator -> system` | 47.160 | 46.181 | 12.031 | 正常可组合路径；成本约为各层叠加 |

三种 adapter 都是显式 opt-in，不应拿来替代裸 allocator。`StatsAllocator` 的 leak map 是
诊断慢路，涉及锁与 map，不进入正常 counters-only 性能承诺。先前 section F 的昂贵 context
证据仍有效：`PreservingObjectPool / reconstruct` 在 Windows、Linux Clang、M4 分别为
20.894/266.230（12.7x）、18.173/93.788（5.2x）、8.900/70.247 ns/op（7.9x）；对应
codec/session 等“构造一次、借还保留状态”的生命周期。

### FiberStackPool：真实 scheduler consumer

新 target `hyle_mem_fiber_stack_bench` 不单测一个孤立 cache API：每个样本通过真实
`Scheduler` 创建、运行、drain 并销毁 32 个 standard-tier fibers；pool 组使用 scheduler-owned
`FiberStackPool`，对照组每次直接 guarded VM reserve/release。表中单位为 **µs/32-fiber batch**，
ratio 为逐进程 `direct_vm / scheduler_pool` latency：

| 平台 / QoS | direct VM p50/p95/p99 | scheduler pool p50/p95/p99 | paired ratio p50/p95/p99 | 判定 |
|---|---:|---:|---:|---|
| Windows / MSVC 19.50 | 151.800 / 184.500 / 199.000 | **7.600 / 8.500 / 14.700** | **20.026 / 21.805 / 13.891** | 全部 15/15 **WIN** |
| Linux / GCC 14.2，CPU 2 | 182.621 / 408.191 / 482.049 | **10.192 / 247.533 / 280.977** | **17.892 / 1.634 / 1.716** | 全部 15/15 **WIN** |
| Apple M4 / interactive | 80.667 / 84.500 / 90.500 | **13.250 / 15.541 / 16.083** | **6.066 / 5.435 / 5.630** | 全部 31/31 **WIN** |
| Apple M4 / utility | 144.708 / 171.542 / 219.709 | **19.417 / 123.458 / 130.125** | **7.474 / 1.389 / 1.686** | 全部 31/31 **WIN** |

Linux/utility 的尾延迟收益小于 p50，但 p95/p99 仍稳定同侧；这正是报告整批生命周期 tail
而不是只报 cache hit 纳秒数的原因。原始逐进程数据：
[Windows](./mem-fiber-stack-win.csv)、[Linux](./mem-fiber-stack-linux.csv)、
[M4 interactive](./mem-fiber-stack-m4.csv)、[M4 utility](./mem-fiber-stack-m4-utility.csv)。
四份均为 clean `2cbc307`；M4 行记录 `qos_ok=1` 与 `affinity=unsupported`。

## 6. 选型

```text
默认小对象             → SystemAllocator
运行时注入/mock         → vtable Allocator（接受对裸 mi 的个位数 % 税）
单线程 borrowed/owned bump → PooledRegion / StackAllocator
单线程同尺寸可增长       → GrowablePool
多线程、每线程独享       → GrowablePool × N
共享固定块、严格配额     → SharedPool / ObjectPool<T>
共享固定块、吞吐优先且有余量 → SharedPoolCache / ObjectPoolCache<T> over hard core
通用多线程 malloc       → mimalloc / oneTBB；不要硬套固定 slab
scheduler fiber 栈       → 由 Scheduler 持有 FiberStackPool；用户不直接管理 raw stack cache
```

## 7. 复跑

```bash
# shipping-like bench；使用仓库同级 ../oneTBB 本地源码快照
cmake --preset bench -DHYLE_BENCH_TBB=ON
cmake --build --preset bench --target hyle_mem_bench hyle_mem_scenarios_bench hyle_mem_fiber_stack_bench

# bump：Win/Linux n=15；M4 分别以 interactive/utility 跑 n=31
HYLE_BENCH_SECTIONS=A \
scripts/bench-stat.py build-bench/bin/hyle_mem_scenarios_bench -n 15 \
  --pair "std::pmr::monotonic_buffer_resource (external, reconstruct)" \
         "HYLE PooledRegion (bump, pooled block)" \
  --pair "std::pmr::monotonic_buffer_resource (external, reconstruct)" \
         "HYLE StackAllocator (bump, own slab)"

# realloc：direct 默认路径和 runtime vtable 必须分开
HYLE_BENCH_SECTIONS=R \
scripts/bench-stat.py build-bench/bin/hyle_mem_bench -n 15 \
  --pair "raw mi_realloc (SOTA)" \
         "HYLE SystemAllocator::reallocate (compile-time)" \
  --pair "raw mi_realloc (SOTA)" \
         "HYLE Allocator::reallocate (vtable)"

# adapter-only：不重复 section F 的其他长项
HYLE_BENCH_SECTIONS=F HYLE_MEM_ADAPTER_ONLY=1 \
scripts/bench-stat.py build-bench/bin/hyle_mem_scenarios_bench -n 15 \
  --pair "HYLE system allocator (direct)" \
         "HYLE QuotaAllocator -> system allocator" \
  --pair "HYLE system allocator (direct)" \
         "HYLE StatsAllocator counters-only -> system allocator" \
  --pair "HYLE system allocator (direct)" \
         "HYLE QuotaAllocator -> StatsAllocator -> system allocator"

# scheduler-owned fiber stack 生命周期和 tail；Linux 另加 --cpu 2
python3 scripts/mem-fiber-stack-stat.py \
  build-bench/bin/hyle_mem_fiber_stack_bench -n 15 --samples 2048 --cpu 2 \
  --raw docs/performance/mem-fiber-stack-linux.csv

# macOS：无硬亲和 API，分别复核 application 的两种 QoS
python3 scripts/mem-fiber-stack-stat.py \
  build-bench/bin/hyle_mem_fiber_stack_bench -n 31 --samples 2048 \
  --raw docs/performance/mem-fiber-stack-m4.csv
HYLE_BENCH_APPLE_QOS=utility \
python3 scripts/mem-fiber-stack-stat.py \
  build-bench/bin/hyle_mem_fiber_stack_bench -n 31 --samples 2048 \
  --raw docs/performance/mem-fiber-stack-m4-utility.csv

export HYLE_BENCH_SECTIONS=DE
export HYLE_BENCH_THREADS=8
export HYLE_BENCH_BLOCK_SIZE=64
export HYLE_BENCH_BATCH=64
export HYLE_BENCH_CAPACITY_PCT=75
export HYLE_BENCH_HANDOFF_BATCH=256
export HYLE_BENCH_QUOTA=1024
export HYLE_BENCH_REQUEST_PCT=110

scripts/bench-stat.py build-bench/bin/hyle_mem_scenarios_bench -n 15 \
  --pair "oneTBB fixed_pool (fixed-buffer shared)" \
         "HYLE SharedPool (hard capacity)" \
  --pair "oneTBB fixed_pool (fixed-buffer shared)" \
         "HYLE SharedPoolCache (explicit soft capacity)"

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

2026-07-22 最新 clean commit `d3e4214` 正式证书如下。数字均为
`HYLE / reference` 的 p50/p95/p99，单位 **µs/request**；ratio 是逐进程配对
`reference/HYLE` 的中位数，三项依次对应 p50/p95/p99：

| 平台 / QoS | request shape | HYLE p50/p95/p99 | reference p50/p95/p99 | paired ratio | 判定 |
|---|---|---:|---:|---:|---|
| Win / MSVC 19.50 | external `FixedRegion` / PMR reconstruct | 4.900 / 4.900 / 7.400 | 8.200 / 8.300 / 13.000 | 1.673 / 1.694 / 1.756 | **WIN / WIN / WIN** |
| Win / MSVC 19.50 | owned `Region` / PMR release | 6.800 / 6.800 / 12.100 | 8.300 / 8.400 / 15.200 | 1.221 / 1.235 / 1.292 | **WIN / WIN / WIN** |
| Win / MSVC 19.50 | `FixedPool` / oneTBB fixed_pool | 26.600 / 33.100 / 56.200 | 36.100 / 42.500 / 67.700 | 1.357 / 1.280 / 1.196 | **WIN / WIN / WIN** |
| Linux / Clang 20.1 CPU 2 | external `FixedRegion` / PMR reconstruct | 4.119 / 4.145 / 7.046 | 7.651 / 7.784 / 13.968 | 1.858 / 1.877 / 2.094 | **WIN / WIN / WIN** |
| Linux / Clang 20.1 CPU 2 | owned `Region` / PMR release | 5.381 / 5.436 / 10.557 | 8.147 / 9.026 / 15.051 | 1.514 / 1.651 / 1.574 | **WIN / WIN / WIN** |
| Linux / Clang 20.1 CPU 2 | `FixedPool` / oneTBB fixed_pool | 24.428 / 30.947 / 56.817 | 37.857 / 48.009 / 73.706 | 1.554 / 1.484 / 1.317 | **WIN / WIN / WIN** |
| Apple M4 / interactive | external `FixedRegion` / PMR reconstruct | 3.667 / 3.750 / 3.834 | 6.333 / 6.417 / 6.500 | 1.727 / 1.711 / 1.696 | **WIN / WIN / WIN** |
| Apple M4 / interactive | owned `Region` / PMR release | 5.667 / 5.792 / 5.875 | 9.917 / 10.084 / 10.292 | 1.750 / 1.741 / 1.752 | **WIN / WIN / WIN** |
| Apple M4 / interactive | `FixedPool` / oneTBB fixed_pool | 23.584 / 23.834 / 30.167 | 40.917 / 42.916 / 48.875 | 1.735 / 1.784 / 1.636 | **WIN / WIN / WIN** |
| Apple M4 / utility | external `FixedRegion` / PMR reconstruct | 3.666 / 3.708 / 3.750 | 6.333 / 6.417 / 6.459 | 1.727 / 1.731 / 1.722 | **WIN / WIN / WIN** |
| Apple M4 / utility | owned `Region` / PMR release | 5.667 / 5.750 / 5.833 | 9.916 / 10.000 / 10.125 | 1.743 / 1.739 / 1.736 | **WIN / WIN / WIN** |
| Apple M4 / utility | `FixedPool` / oneTBB fixed_pool | 23.625 / 23.834 / 26.375 | 38.167 / 38.834 / 42.667 | 1.615 / 1.628 / 1.621 | **WIN / WIN / WIN** |

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
clean `d3e4214` 上 Win/Linux/M4 的 p50/p95/p99 全部通过 paired sign 闸门；75% mixed-fill
碎片证书未退化。

## 9. Request footprint / fragmentation 证书

section H 验证 owned request scratch 的空间行为，不把一个短循环的 allocator 调用次数冒充
真实 footprint。单次峰值 request 默认写入 128 MiB payload：约 80% 是 24B–64KiB 的普通
parser/graph scratch，约 20% 是偶发 2 MiB staging buffer；`clear/release` 后再写入 1/4 大小的
request，最后 full release。HYLE `Region` 与 `std::pmr::monotonic_buffer_resource` 都由同一个
bench-local mimalloc vtable backing，排除不同 system heap 的影响。

`Region` 当前默认 ordinary backing target 为 4 MiB，且 target 包含内部元数据；单个 user
allocation 放不下时才突破。`--block-mib` 仅作诊断参数，正式证书签当前默认。该定义不改变
`clear()` retention、dedicated-large 释放或 hard `size_limit` 合同。

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
Apple QoS，并把每个 child 的原始字段写入本目录 CSV。

2026-07-22 clean commit `02ec81e` 正式证书如下。reservation ratio 和 retained 是 allocator
精确字节；RSS 列为相对该 child baseline 的中位数，单位 MiB：

| 平台 / QoS | variant | reserved / payload | clear 后精确保留 | live RSS Δ | clear RSS Δ | full-release RSS Δ | full-release backing |
|---|---|---:|---:|---:|---:|---:|---:|
| Windows / MSVC 19.50 | HYLE Region | **1.031** | 107.938 | 129.109 | 104.363 | 0.035 | **0** |
| Windows / MSVC 19.50 | PMR monotonic | 1.443 | **0** | **128.590** | **0.031** | **0.031** | **0** |
| Linux / Clang 20.1 CPU 2 | HYLE Region | **1.031** | 107.938 | **138.047** | 106.672 | **0.047** | **0** |
| Linux / Clang 20.1 CPU 2 | PMR monotonic | 1.302 | **0** | 144.094 | **0.094** | 0.094 | **0** |
| Apple M4 / interactive | HYLE Region | **1.031** | 107.938 | **128.672** | 128.672 | 132.500 | **0** |
| Apple M4 / interactive | PMR monotonic | 2.000 | **0** | 128.719 | **128.719** | **128.750** | **0** |
| Apple M4 / utility | HYLE Region | **1.031** | 107.938 | **128.672** | 128.672 | 132.500 | **0** |
| Apple M4 / utility | PMR monotonic | 2.000 | **0** | 128.719 | **128.719** | **128.750** | **0** |

结论分三层：HYLE 的 allocator-visible live fragmentation 明确更低，且三套标准库实现都如此；
修正 size-class crossing 后，Linux live RSS 从比 PMR 高约 5.5% 变成低约 4.2%，M4 低约
0.04%，Windows 仍高约 0.4%，因此写成“三平台同档、Linux 缺口已关闭”，不夸成 RSS 全平台胜。
`clear()` 后 HYLE 精确保留 107.938 MiB ordinary blocks 是跨 request 复用策略，PMR
`release()` 则归零，两者不是同一个 retention policy。最终两者 backing 都精确归零；Win/Linux
RSS 回到 baseline 附近，Darwin 的 resident accounting 没有随 purge 同步下降，不能把该列误读
成泄漏。interactive/utility 完全一致，也没有证据表明 M4 QoS 改变空间行为。

原始逐进程字段：
[Windows CSV](./mem-request-footprint-win.csv)、
[Linux CSV](./mem-request-footprint-linux.csv)、
[M4 interactive CSV](./mem-request-footprint-m4.csv)、
[M4 utility CSV](./mem-request-footprint-m4-utility.csv)。全部为 `dirty=0`、`released_bytes=0`、
`qos_ok=1`；Win/Linux 各 15 对 fresh process，M4 每种 QoS 各 31 对。

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

## 10. FixedPool 外部碎片证书

section I 采用 oneTBB `test_malloc_pools` 相同的核心问题：池经历分配与释放后，最大还能满足多大
的连续对象，以及全部释放后能否恢复空池容量。HYLE 使用 8 MiB external buffer；oneTBB 使用
8 MiB payload budget + 其测试所需的 1 MiB bootstrap allowance。证书先二分探测各自
`empty_largest`，后续指标都按这个 user-visible limit 归一化，因此 bootstrap 不会成为容量优势。

workload 以 64B–64KiB mixed-size 对象填到 empty user capacity 的 75%，再释放每三个物理槽中的
一个，形成真实的 non-adjacent holes。报告：

- `fragmented_largest / (empty_largest - live_requested)`：理论剩余 user capacity 中，当前能作为
  一个连续对象满足的比例；越高说明外部碎片越低。
- `recovered_largest / empty_largest`：全部 live allocation 释放后的容量恢复率；应为 1。

2026-07-22 clean commit `d3e4214` 正式结果如下。Windows/Linux 各 `n=15` 独立进程；Apple M4
在 interactive / utility QoS 各 `n=31`。数值为 median `[min, max]`：

| 平台 / QoS | HYLE availability | oneTBB availability | HYLE / oneTBB | HYLE / oneTBB recovery |
|---|---:|---:|---:|---:|
| Windows / MSVC 19.50 | 0.6874 `[0.6874, 0.6874]` | 0.4843 `[0.4816, 0.4843]` | **1.4193x** `[1.4193, 1.4273]` | 1.0000 / 1.0000 |
| Linux / Clang 20.1，guest CPU 2 | 0.6874 `[0.6874, 0.6874]` | 0.4816 `[0.4816, 0.4816]` | **1.4273x** `[1.4273, 1.4273]` | 1.0000 / 1.0000 |
| Apple M4 / interactive | 0.6874 `[0.6874, 0.6874]` | 0.4816 `[0.4816, 0.4816]` | **1.4273x** `[1.4273, 1.4273]` | 1.0000 / 1.0000 |
| Apple M4 / utility | 0.6874 `[0.6874, 0.6874]` | 0.4816 `[0.4816, 0.4816]` | **1.4273x** `[1.4273, 1.4273]` | 1.0000 / 1.0000 |

因此，在这一个明确的 75% mixed-fill / selective-free workload 中，HYLE 留出的最大连续对象
约为 oneTBB 的 1.42 倍，且两者在全部释放后都完整合并回空池容量。这不是对任意 allocation
distribution 的普遍碎片保证；Windows 上 oneTBB 的 4 KiB 级变化也保留为原始 range，不用单点
中位数掩盖。M4 两种 QoS 一致只说明该非吞吐 workload 对调度不敏感，仍不声称 P/E 核绑定。

原始逐进程 CSV：
[Windows](./mem-fixed-pool-fragmentation-win.csv)、
[Linux](./mem-fixed-pool-fragmentation-linux.csv)、
[Apple M4 interactive](./mem-fixed-pool-fragmentation-m4.csv)、
[Apple M4 utility](./mem-fixed-pool-fragmentation-m4-utility.csv)。

```bash
# Linux；Windows 使用与 footprint 证书相同的 --dll-dir
python3 scripts/mem-fragmentation-stat.py \
  build-bench/bin/hyle_mem_scenarios_bench -n 15 --mib 8 --cpu 2 \
  --raw docs/performance/mem-fixed-pool-fragmentation-linux.csv

# macOS 延续 interactive / utility 两组 QoS 审计
python3 scripts/mem-fragmentation-stat.py \
  build-bench/bin/hyle_mem_scenarios_bench -n 31 --mib 8 \
  --raw docs/performance/mem-fixed-pool-fragmentation-m4.csv
HYLE_BENCH_APPLE_QOS=utility \
python3 scripts/mem-fragmentation-stat.py \
  build-bench/bin/hyle_mem_scenarios_bench -n 31 --mib 8 \
  --raw docs/performance/mem-fixed-pool-fragmentation-m4-utility.csv
```
