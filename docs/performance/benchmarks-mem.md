# mem 实测铁证

> 本页是 [performance 总账](./README.md) 的 **mem 模块**详节。
> 源：`bench/mem/allocator_bench.cc`、`bench/mem/scenario_bench.cc`、`bench/mem/fiber_stack_bench.cc`。
> 通用 churn、request closure、FixedPool、latency、fragmentation 和 footprint 按各节所列 commit。2026-07-25 的 `6213115` 把长寿命 Region 从 restart-from-head 收口为每 epoch 单调遍历，并在三平台 shipping 构建上重签正常 request 与 32→2048 retained-block scaling；此前 `2cbc307`、`1de939d`、`20f3676` 分别签署 bump/system-realloc/fiber、SharedPool/cache 和 O(1) FixedPool。同契约性能判定信任配对 **[sign]**；旧 raw/result 继续诚实绑定实际采样 commit，不把历史数据伪改成 HEAD。

## 当前性能封口状态

本页独占 mem 的性能现状；architecture 和 guide 不复制点值。已签结果按各节标注的 commit
有效，不因 API-only commit 自动失效。MEM-FINAL-2 的四个有界证据缺口已关闭：

| 原缺口 | `2cbc307` 关闭证据 | 结论 |
|---|---|---|
| M4 PooledRegion/Stack | 对称 8192×64B batch；interactive 为 1.023x / 1.023x，utility 为 1.028x / 1.026x，均对 PMR sign WIN | 已关闭；旧 0.67x 作废 |
| realloc 分层 | static/source direct `SystemAllocator` 对 raw mi：Win 1.005x TIE、Linux 0.989x TIE、M4 1.018x WIN；runtime vtable 单列 | 合并形状同档；x86 动态多态税显式可见 |
| adapter composition | direct、Quota、counters-only Stats、Quota→Stats 三平台独立量化 | 默认路径零装饰税；策略成本由用户按需购买 |
| FiberStackPool | 真实 `Scheduler` 每 batch 创建/运行/销毁 32 fibers，缓存与直接 guarded VM 比较 p50/p95/p99 | 三平台、M4 双 QoS 全 percentile sign WIN |

`SharedPool` hard core 的 immediate-global-free/精确容量合同强于 owner-local free list；它不以
击败较弱合同的 oneTBB local churn 为冻结条件。吞吐形状由显式 `SharedPoolCache` 对拍，hard
core 仍报告绝对成本和较弱合同参照。

`6213115` 关闭最后一个 arena 复杂度缺口：Region clear 后不再为每次 block transition 从链头
重扫，而是像 Folly Arena 一样单调推进 retained cursor；结构测试硬性证明每个 block 每 epoch
至多考察一次。热路径同时把 peak 折叠延迟到 clear/release/query，在保持 `peak_used()` 任意
时刻精确的前提下去掉每次 allocation 的 peak 依赖链。正常 request 与长寿命 scaling 证据见
§2/§8，正式数据全部来自 `HYLE_TESTING=OFF` 的 shipping 构建；测试诊断构建的临时数据未入表。
当前 scenario benchmark 还会输出 `HYLE_MEM_CERT build testing=0/1`，四个证书脚本在看到
`testing=1` 或缺失 build marker 时直接拒绝采样，避免以后再次把诊断成本签进产品表。

`1de939d` 还关闭了两个正常路径结构问题：SharedPool 将约 2 KiB 的 O(shards) stable-empty
observation 栈状态移入 noinline exhaustion 冷路（MSVC shipping codegen 的成功入口只保留
32B 栈帧）；FixedPool 将 retained-span/后一空闲邻块 grow 内联，搬移+copy 独立为 noinline
冷路。后者新增 oneTBB fixed_pool realloc 同形对拍，见 §4。

## 1. 通用路径（`hyle_mem_bench`）

### Shared SDK same-link standard adapter 证书（`ab92b49`）

旧表把 DLL/DSO 内 HYLE mimalloc 与可执行文件内第二份 static mimalloc 放在一起，适合作为绝对
诊断下界，却不能公平判定 shared API。section `L` 现在让正式行全部跨同一个 `hyle_mem` shared
seam、进入同一个 mimalloc backend；每个 variant 独立进程运行，外层按正反顺序交错，避免固定
执行顺序把 heap warmup 或频率漂移写成 adapter 差异。Windows MSVC 19.50 Release、testing=0、
n=15 的跨进程中位数如下（raw：`mem-shared-same-link-win-ab92b49.csv`）：

| 正常形状 | ns/op | 对同类直接形状 | 判定 |
|---|---:|---:|---|
| backend raw shared seam | 8.41 | baseline | 只作为相同链接/后端下界 |
| `SystemAllocator` | 8.60 | raw 的 **0.978x** | TIE（默认 heap） |
| `SystemStdAllocator<uint64_t>` | 8.74 | SystemAllocator 的 **0.984x** | TIE（默认 allocator-template） |
| runtime `Allocator` | 8.88 | raw 的 **0.947x** | 显式 policy 注入税，仍快于单层 PMR |
| runtime `StdAllocator<uint64_t>` | 9.11 | runtime Allocator 的 **0.975x** | TIE（注入 adapter） |
| direct-backend PMR rival | 9.49 | baseline | 同类单层 PMR 参照 |
| `system_memory_resource()` | 9.58 | rival 的 **0.991x** | TIE（默认 PMR） |
| runtime `PmrResource` | 10.19 | rival 的 **0.931x** | 双动态层的显式注入成本，不冒充默认形状 |

因此默认 standard-container 路径已有同链接 SOTA/TIE 证据；运行时注入路径仍诚实保留策略成本，
不为 benchmark 增加常态 cache/metadata。可执行文件内 `raw mi_*` 继续输出但不参与 shared 判定。
WSL GCC 14 shared shipping 构建与 section L 单轮逐项 smoke 已通过；正式 n=15 被本轮立即合入要求
中断，M4 也未重签，所以这两平台仍是证书补录项，不能引用本表冒充三平台完成。

2026-07-25 的底座依赖审计发现旧 benchmark 给可执行文件手工链接了 `mimalloc_static`，所以
“SystemAllocator 内联”数字真实代表 **static/source 合并形状**，不能冒充 shared SDK consumer。
shared 公开头现经 `hyle_mem` 的一次 direct-import seam 进入同一 mimalloc backend，不再要求用户
链接或加载第二份 vendor 实现。Windows shipping 诊断为 churn 8.22 ns vs executable 内 raw mi
7.75 ns（0.943x，约 6% DSO 边界上界），realloc 70.66 vs 67.64 ns（0.957x）；同轮 vtable 为
8.08/72.45 ns。这个单机诊断用于诚实揭示交付形状，不是三平台 sign 证书，也不是同为 shared
mimalloc 的对称对手；补齐三平台 shared-vs-shared 证书前，不宣称 shared 行已平 mi SOTA。

| 场景 | Win | WSL | M4 | 判定 |
|---|---|---|---|---|
| **SystemAllocator static/source direct** vs 裸 mi churn | 7.93 / 7.86 → **0.996 TIE** | 8.04 / 8.04 → **0.99 TIE** | 7.77 / 7.91 → **1.00 TIE** | ➖ 合并形状三平台平 mimalloc |
| **SystemAllocator shared direct-import** vs exe 内裸 mi churn | 8.22 / 7.75 → **0.943** | 待重签 | 待重签 | ⚠️ Windows 诊断：约 6% DSO seam 上界；待同链接模型三平台证书 |
| **vtable** vs 裸 mi churn | 8.67 / 7.86 → **0.91 LOSE sign** | 8.67 / 8.04 → **0.94 LOSE sign** | 7.93 / 7.91 → **1.00 TIE** | Win/WSL 有约 6–9% 多态税 |
| **vtable** vs pmr(over mi) churn | 8.67 / 8.67 → **TIE** | 8.67 / 10.27 → **1.22 WIN sign** | 7.93 / 9.98 → **1.26 WIN** | 同类多态对照平或赢 pmr |
| static/source direct **SystemAllocator reallocate** / raw `mi_realloc` | 66.979 / 67.691 → **1.005 TIE** | 79.279 / 78.995 → **0.989 TIE** | 81.643 / 83.577 → **1.018 WIN sign** | 合并形状三平台同档 |
| shared direct-import **SystemAllocator reallocate** / exe 内 raw `mi_realloc` | 70.66 / 67.64 → **0.957** | 待重签 | 待重签 | ⚠️ Windows 诊断：约 4.5% DSO seam 上界 |
| runtime **Allocator vtable reallocate** / raw `mi_realloc` | 74.922 / 67.691 → **0.906 LOSE sign** | 86.344 / 78.995 → **0.921 LOSE sign** | 81.329 / 83.577 → **1.020 WIN sign** | x86 有约 8–9% 动态多态税；M4 无税 |

static/source direct 路径平 mimalloc；shared 默认路径保持编译期 direct binding，但必须支付一次
库边界，不能再拿 benchmark-only vendor 补链消掉它。需要运行时注入的 vtable 路径对裸 mi 有
可测税，但仍平或赢同为多态封装的 pmr。

2026-07-26 strict-alignment 实现把验证放在**显式 alignment 的 C++ overload / C convenience**；
`Allocator::allocate(size)`、默认 `reallocate` 和编译期 `SystemAllocator` 默认参数的 shipping
汇编不含 alignment 分支，动态显式 allocate/free/reallocate 请求才付一次 fail-fast 检查。Windows shared implementation
smoke（MSVC 19.50，section A，`n=15`，未作为正式 clean 证书替换上表）为：shared
`SystemAllocator/raw mi=0.913`、runtime vtable/raw mi=0.904、runtime vtable/同类 PMR=0.982；
三者 paired sign 均在 lose 侧、absolute spread 判 TIE。它确认 strict check 没污染默认 runtime
热路，也再次说明 shared/vtable seam 尚不能宣称 SOTA；正式结论仍待同链接模型三平台重签。

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

section G 每个 request 执行 4096 次 mixed-size（24–320B）、16B 对齐分配，然后 bulk
clear/release；每个进程对 HYLE/reference 交替顺序采 10,001 个 request。`6213115` 的当前
shipping 结果在 §8 给出完整 p50/p95/p99：Windows/Linux 的 external/owned 两种 Region 与
M4 interactive/utility 全部 paired sign WIN。Win 与 WSL 同机错开；Linux 固定 guest CPU 2；
Darwin 不支持硬 P/E core 亲和，因此只签 application QoS，不冒充物理核心类别。

长寿命复杂度另由 section J 独立签署：先真实保留 N 个 ordinary blocks，再对同一 Region
执行 10,001 次 clear/reuse epoch，每个平台 31 个进程。表中是跨进程中位
`p50_ns_per_allocation`；末列是 2048/32 blocks 的单位成本比。若仍从链头重扫，单位成本会
随 block 数近似增长 64×；当前只反映 loop/cache/clock 的常数变化：

| 平台 / QoS | 32 blocks | 128 | 512 | 2048 | 2048/32 |
|---|---:|---:|---:|---:|---:|
| Windows / MSVC 19.50 | 3.125 | 3.906 | 3.711 | 3.955 | **1.266x** |
| WSL / GCC 14.2，CPU 2 | 3.250 | 3.125 | 3.070 | 3.174 | **0.977x** |
| Apple M4 / interactive | 2.594 | 2.930 | 3.174 | 4.679 | **1.804x** |
| Apple M4 / utility | 2.594 | 2.930 | 3.174 | 4.680 | **1.804x** |

M4 两种 QoS 的中位数几乎逐项相同；interactive 个别进程在 32-block 短 epoch 上被调度放大，
不影响 31-process 中位。1.804× 不是超线性遍历：硬结构门禁同时以 512 blocks×64 epochs
断言每个 epoch 恰好只考察 511 个后续候选且不增长 backing。这里签的是线性总工作、摊还
O(1) block transition，不承诺跨越 L1/L2 工作集后每次 allocation 的绝对 ns 完全不变。

原始逐进程行：
[Windows](./mem-region-scaling-win-6213115.csv)、
[Linux mirror](./mem-region-scaling-wsl-6213115.csv)、
[M4 interactive mirror](./mem-region-scaling-m4-6213115.csv)、
[M4 utility mirror](./mem-region-scaling-m4-utility-6213115.csv)。Windows 是 clean `6213115`；
WSL/M4 是 Windows 源码副本，按实际旧 HEAD/`dirty=1` 记录，不伪造镜像 Git 元数据。

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

绑定 clean product commit `1de939d`。三平台均为 Release shipping-like、8 workers、64B block、
request batch=64、working set=容量的 75%、handoff batch=256。Win MSVC 19.50 与 WSL
GCC 14.2 为 `n=15`，Apple M4 AppleClang 21 的 `USER_INTERACTIVE` / `UTILITY` 各
`n=31`。ratio 是每次进程内配对的 **HYLE / oneTBB op/s** 后再跨进程取中位数。
Darwin 无公开硬亲和 API，两组只证明 application QoS 下的结论。

| 平台 / 编译器 | 场景 | `SharedPoolCache` | oneTBB `fixed_pool` | HYLE/oneTBB throughput | 判定 |
|---|---|---:|---:|---:|---|
| Windows / MSVC 19.50 | request batch | **1.131 ns/op** | 2.244 ns/op | **1.951x** | **WIN spread / WIN sign**（15/15，p=6.1e-05） |
| Windows / MSVC 19.50 | handoff | **1.387 ns/op** | 8.894 ns/op | **6.419x** | **WIN spread / WIN sign**（15/15，p=6.1e-05） |
| WSL / GCC 14.2 | request batch | **1.058 ns/op** | 1.703 ns/op | **1.593x** | **WIN spread / WIN sign**（15/15，p=6.1e-05） |
| WSL / GCC 14.2 | handoff | **1.336 ns/op** | 7.305 ns/op | **5.493x** | **WIN spread / WIN sign**（15/15，p=6.1e-05） |
| Apple M4 / interactive | request batch | **0.897 ns/op** | 2.108 ns/op | **2.348x** | **WIN spread / WIN sign**（31/31，p=9.3e-10） |
| Apple M4 / interactive | handoff | **1.945 ns/op** | 4.994 ns/op | **2.601x** | **WIN spread / WIN sign**（31/31，p=9.3e-10） |
| Apple M4 / utility | request batch | **0.938 ns/op** | 2.471 ns/op | **2.580x** | **WIN spread / WIN sign**（31/31，p=9.3e-10） |
| Apple M4 / utility | handoff | **1.986 ns/op** | 5.152 ns/op | **2.593x** | **WIN spread / WIN sign**（31/31，p=9.3e-10） |

显式 cache 的 locality 形状与 oneTBB fixed_pool 接近；Windows、WSL 与 M4 两种 QoS
全部配对同侧。它不是硬容量视图，性能优势不能替代 3.4 的 quota 语义。

### 3.3 当前 hard core：三平台 clean 证书

绑定 clean product commit `1de939d`，Release shipping-like（`HYLE_BUILD_TESTS=OFF`）；配置与
3.2 相同。Win MSVC 19.50、WSL GCC 14.2 为 `n=15`，Apple M4 AppleClang 21 的
interactive/utility 各 `n=31`。home-shard 成功入口与 stable-empty 冷路已拆开；tag 饱和后的
serialized fallback、立即全局可见和稳定空证明均未改变。

hard 与 fixed_pool 的可见性合同不同，因此下表只量化更强合同的成本，不做 WIN/LOSE
或 SOTA verdict。request batch 中，oneTBB 的 allocate/free 都命中 owner 私有链，通常没有
共享 RMW；HYLE hard 的 allocate 与 deallocate 各做一次共享 CAS。handoff 中 oneTBB 通常是
本地 allocate + 一次 remote-free CAS，HYLE 仍是两端各一次 CAS。M4 对共享 atomic RMW
更敏感，所以差距比 x86 明显；x86 上 HYLE 的简单分片链又可能抵消 tbbmalloc 元数据成本。

| 平台 / 编译器 | 场景 | `SharedPool` hard | oneTBB `fixed_pool` | HYLE/oneTBB throughput（成本参照） |
|---|---|---:|---:|---:|
| Windows / MSVC 19.50 | request batch | 6.556 ns/op | 2.244 ns/op | 0.354x |
| Windows / MSVC 19.50 | handoff | **4.046 ns/op** | 8.894 ns/op | **2.176x** |
| WSL / GCC 14.2 | request batch | 6.182 ns/op | 1.703 ns/op | 0.270x |
| WSL / GCC 14.2 | handoff | **3.266 ns/op** | 7.305 ns/op | **2.132x** |
| Apple M4 / interactive | request batch | 8.260 ns/op | 2.108 ns/op | 0.258x |
| Apple M4 / interactive | handoff | 9.348 ns/op | 4.994 ns/op | 0.546x |
| Apple M4 / utility | request batch | 5.963 ns/op | 2.471 ns/op | 0.420x |
| Apple M4 / utility | handoff | 9.443 ns/op | 5.152 ns/op | 0.548x |

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

2026-07-24 产品提交 `20f3676`：Windows 权威 checkout 为 clean `n=15`；WSL 与 M4 是该
Windows 树的工作副本，`fixed_pool.h` 内容 SHA-256 同为
`9fdbd27001b5b109804f03804a212d570a4a796f19e8e4053198120071bd60d6`。镜像仓库保留长期
dirty/旧 HEAD，因此原始行如实记录 `dirty=1`，不伪称 clean；M4 显式 QoS 各 `n=31`。

| 平台 | HYLE FixedPool | oneTBB fixed_pool | HYLE/oneTBB throughput | 判定 |
|---|---:|---:|---:|---|
| Windows/MSVC 19.50 Release | 6.835 ns/op | 10.101 ns/op | median 1.491，range 1.395–1.588 | **WIN spread / WIN sign**（15/15） |
| Ubuntu 24.04/GCC 14.2 Release | 6.322 ns/op | 7.845 ns/op | median 1.293，range 1.071–1.469 | **TIE spread / WIN sign**（15/15） |
| Apple M4 / AppleClang 21，interactive | 6.012 ns/op | 9.406 ns/op | median 1.563，range 1.506–1.700 | **WIN spread / WIN sign**（31/31） |
| Apple M4 / AppleClang 21，utility | 6.028 ns/op | 9.407 ns/op | median 1.564，range 1.501–1.606 | **WIN spread / WIN sign**（31/31） |

当前实现保留 one-entry recent-free cursor 承接 recent return 与 split remainder；未命中后由
一级 logarithmic range、每级 8 个 subdivision 和两层 bitmap 直接定位一个满足大小的链表头，
不再扫描同 size bin 的候选。侵入式 free list 与 boundary tag 使 allocate/free/coalesce 的索引
工作均为 O(1)。`HYLE_TESTING` 的 96-hole 长寿命 arena 结构测试验证一次查找至多观察 cursor
和一个索引头；shipping 不保留该计数。Windows/WSL/M4 的正常 workload 均仍胜 oneTBB 的
成熟 fixed-buffer 参照；M4 双 QoS 一致，但不冒充硬 P/E 核绑定证书。

### 4.1 FixedPool retained-span realloc

`1de939d` 新增真实 external-buffer grow/shrink 循环：同一个 16-aligned allocation 在
256B/1024B 间反复 resize，每个 epoch 4096 cycles，HYLE 与 oneTBB `fixed_pool::realloc`
各执行相同的 grow+shrink 次数。两者的 shrink 都保留物理 span，后续 regrow 不复制；HYLE
还可在首次空间不足时 O(1) 合并后一空闲邻块，并把未消费尾部留在 pool。常用路径
force-inline，真正搬移+copy 为 noinline 冷路。

| 平台 / QoS | HYLE FixedPool | oneTBB fixed_pool realloc | HYLE/oneTBB throughput | 判定 |
|---|---:|---:|---:|---|
| Windows / MSVC 19.50 | **1.286 ns/op** | 1.649 ns/op | **1.280x**，range 1.248–1.329 | **WIN spread / WIN sign**（15/15） |
| WSL / GCC 14.2 | **1.207 ns/op** | 1.687 ns/op | **1.394x**，range 1.337–1.431 | **WIN spread / WIN sign**（15/15） |
| Apple M4 / interactive | **1.267 ns/op** | 2.334 ns/op | **1.829x**，range 1.601–1.874 | **WIN spread / WIN sign**（31/31） |
| Apple M4 / utility | **1.424 ns/op** | 2.358 ns/op | **1.657x**，range 1.616–1.864 | **WIN spread / WIN sign**（31/31） |

oneTBB 的 fixed_pool 是成熟但线程安全的 fixed-buffer 参照；它没有单线程特化，因此本表只证明
HYLE 的单线程产品形状胜过该可用成熟实现，不声称存在一个未测的完全同并发合同对手。
渐进复杂度上，retained-span/后一空闲邻块 resize 为 O(1)；无法原地容纳时进入普通两级
bitmap allocation，其索引工作仍为 O(1)，随后才复制 O(min(old,new)) bytes。

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

2026-07-23 `FiberStack::{allocate}` / pool `acquire` 改为 `Result<FiberStack>` 后，Windows
MSVC 19.50 用当前 benchmark 形状做 100-sample smoke：scheduler pool p50/p95/p99
**7.3/8.1/14.4 µs**，direct VM **155.0/196.1/230.7 µs**。与上表旧正式证书处于同一带，
没有观察到 Result 成功态回退；样本数不足以替换 n=15 正式表，只作为迁移哨兵。

2026-07-26 新增的 `capacity/trim/clear` 不进入 acquire/release 成功热路：capacity 是 immutable
query，trim 在显式 cold 控制点逐个把 VM release 放到 SpinLock 外。Scheduler shutdown 在 worker
join 后 clear 三档 cache。原有真实 scheduler hot-reuse 表继续有效；reclaim 的验收是 cached 从 N
降到 keep/0 且 shutdown 后为 0，不用一个孤立纳秒数字把冷 VM unmap 冒充性能能力。

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
         "HYLE FixedPool (single-thread, hard buffer)" \
  --pair "oneTBB fixed_pool realloc (thread-safe)" \
         "HYLE FixedPool (retained-span in-place resize)"

# Apple M4：默认 USER_INTERACTIVE；再以 UTILITY 做较低优先级调度复核
HYLE_BENCH_SECTIONS=F HYLE_BENCH_APPLE_QOS=utility \
scripts/bench-stat.py build-bench/bin/hyle_mem_scenarios_bench -n 31 \
  --pair "std::pmr::monotonic_buffer_resource (external, reconstruct)" \
         "HYLE FixedRegion (external, no fallback)" \
  --pair "std::pmr::monotonic_buffer_resource (release upstream blocks)" \
         "HYLE Region (retained ordinary blocks)" \
  --pair "oneTBB fixed_pool (thread-safe, hard buffer)" \
         "HYLE FixedPool (single-thread, hard buffer)" \
  --pair "oneTBB fixed_pool realloc (thread-safe)" \
         "HYLE FixedPool (retained-span in-place resize)"

# quota 计数是单次语义证据；不把其中线程启动时间当吞吐比较
HYLE_BENCH_SECTIONS=E build-bench/bin/hyle_mem_scenarios_bench

# 长寿命 Region：31 个独立进程，每进程/每尺寸 10,001 clear/reuse epochs
python3 scripts/mem-region-scaling-stat.py \
  build-bench/bin/hyle_mem_scenarios_bench -n 31 --samples 10001 --cpu 2 \
  --raw docs/performance/mem-region-scaling-wsl-6213115.csv
```

Linux/WSL 必须从 ext4 工作副本运行；Win、WSL 同机时必须错开，不得并发采样。Apple M4 使用本地 APFS 工作副本。

## 8. Request 尾延迟证书

section G 使用同一个 4096 次 mixed-size request，但每次只在整个 request
边界读取一次 `steady_clock`。报告的是 **ns/request** 的 nearest-rank p50/p95/p99，不是给每次
allocation 插时钟，也不是把 nanobench epoch 当用户延迟。每个样本内 HYLE/对照交替先后顺序，
降低固定执行顺序与温漂偏差；`6213115` 每进程使用 10,001 samples，Windows/Linux 至少 15 个
独立进程，M4 每种 QoS 31 个进程。

Region 行由当前 clean `6213115` 重签；Linux/M4 的同一 shipping build 也重签 FixedPool，
Windows FixedPool 保留未被本次产品改动影响的 clean `20f3676` 行。数字均为
`HYLE / reference` 的 p50/p95/p99，单位 **µs/request**；ratio 是逐
进程配对 `reference/HYLE` 的中位数，三项依次对应 p50/p95/p99：

| 平台 / QoS | request shape | HYLE p50/p95/p99 | reference p50/p95/p99 | paired ratio | 判定 |
|---|---|---:|---:|---:|---|
| Win / MSVC 19.50 | external `FixedRegion` / PMR reconstruct | 7.000 / 7.500 / 12.900 | 8.200 / 8.300 / 14.900 | 1.171 / 1.107 / 1.163 | **WIN / WIN / WIN** |
| Win / MSVC 19.50 | owned `Region` / PMR release | 5.400 / 5.800 / 10.200 | 7.600 / 7.700 / 14.200 | 1.407 / 1.328 / 1.430 | **WIN / WIN / WIN** |
| Win / MSVC 19.50 | `FixedPool` / oneTBB fixed_pool | 28.600 / 34.600 / 60.900 | 42.900 / 48.100 / 81.600 | 1.500 / 1.395 / 1.336 | **WIN / WIN / WIN** |
| Linux / GCC 14.2 CPU 2 | external `FixedRegion` / PMR reconstruct | 3.470 / 4.122 / 5.183 | 4.875 / 4.912 / 8.098 | 1.405 / 1.194 / 1.435 | **WIN / WIN / WIN** |
| Linux / GCC 14.2 CPU 2 | owned `Region` / PMR release | 5.689 / 5.942 / 8.554 | 8.254 / 9.006 / 14.289 | 1.451 / 1.524 / 1.709 | **WIN / WIN / WIN** |
| Linux / GCC 14.2 CPU 2 | `FixedPool` / oneTBB fixed_pool | 22.814 / 28.698 / 49.612 | 33.374 / 38.592 / 61.088 | 1.474 / 1.368 / 1.253 | **WIN / WIN / WIN** |
| Apple M4 / interactive | external `FixedRegion` / PMR reconstruct | 4.125 / 4.209 / 4.291 | 6.334 / 6.458 / 6.541 | 1.536 / 1.534 / 1.525 | **WIN / WIN / WIN** |
| Apple M4 / interactive | owned `Region` / PMR release | 5.500 / 6.125 / 6.375 | 9.958 / 10.084 / 10.250 | 1.803 / 1.664 / 1.608 | **WIN / WIN / WIN** |
| Apple M4 / interactive | `FixedPool` / oneTBB fixed_pool | 24.292 / 24.583 / 29.042 | 40.791 / 42.667 / 46.958 | 1.679 / 1.731 / 1.611 | **WIN / WIN / WIN** |
| Apple M4 / utility | external `FixedRegion` / PMR reconstruct | 4.125 / 4.209 / 4.291 | 6.334 / 6.458 / 6.542 | 1.535 / 1.525 / 1.524 | **WIN / WIN / WIN** |
| Apple M4 / utility | owned `Region` / PMR release | 5.500 / 6.084 / 6.375 | 9.958 / 10.125 / 10.292 | 1.811 / 1.664 / 1.617 | **WIN / WIN / WIN** |
| Apple M4 / utility | `FixedPool` / oneTBB fixed_pool | 24.458 / 24.958 / 29.542 | 40.750 / 42.583 / 47.292 | 1.666 / 1.701 / 1.592 | **WIN / WIN / WIN** |

`6213115` 原始逐进程 percentile：
[Windows clean](./mem-request-latency-win-6213115.csv)、
[Linux mirror](./mem-request-latency-wsl-6213115.csv)、
[M4 interactive mirror](./mem-request-latency-m4-6213115.csv)、
[M4 utility mirror](./mem-request-latency-m4-utility-6213115.csv)。Windows 为 clean HEAD；
Linux/M4 是同一产品源码副本并如实记录旧 HEAD/`dirty=1`。M4 raw 分别记录
`qos=user-interactive/utility` 与 `affinity=unsupported`，不冒充硬 P/E 核绑定。

Windows `20f3676` FixedPool 重签原始行继续有效：
[Windows](./mem-request-latency-fixed-pool-win-20f3676.csv)、
[Linux mirror](./mem-request-latency-fixed-pool-linux-20f3676.csv)、
[M4 interactive mirror](./mem-request-latency-fixed-pool-m4-20f3676.csv)、
[M4 utility mirror](./mem-request-latency-fixed-pool-m4-utility-20f3676.csv)。Windows 行记录
clean `20f3676`；两套镜像行按副本现状记录旧 HEAD/`dirty=1`，其 FixedPool 实现 hash 已在
§4 给出，不把副本 Git 元数据改写成权威提交。

```bash
# Linux：固定同一逻辑 CPU
python3 scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench \
  -n 15 --samples 10001 --cpu 2 \
  --raw docs/performance/mem-request-latency-wsl-6213115.csv

# macOS：无受支持的硬亲和 API，延续 USER_INTERACTIVE / UTILITY 两组 QoS 审计
python3 scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench \
  -n 31 --samples 10001 --raw docs/performance/mem-request-latency-m4-6213115.csv
HYLE_BENCH_APPLE_QOS=utility \
  python3 scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench \
  -n 31 --samples 10001 --raw docs/performance/mem-request-latency-m4-utility-6213115.csv

# Windows：oneTBB 的运行时 DLL 在 build tree 的配置目录中
python scripts/mem-cert-stat.py build-bench/bin/hyle_mem_scenarios_bench.exe \
  -n 15 --samples 10001 \
  --dll-dir build-bench/msvc_19.50_cxx23_64_md_release \
  --raw docs/performance/mem-request-latency-win-6213115.csv
```

脚本保存每个 process/variant 的原始 percentile、HEAD 与 dirty 标记，并对每个 percentile 做
paired reference/HYLE ratio 与 sign test。权威 checkout 的正式行必须 clean；只读镜像允许
保留 `dirty=1`，但必须像 §4 一样给出与权威提交一致的被测实现 hash，不能把镜像元数据改写
成 clean。`20f3676` FixedPool 在 Win/Linux/M4 的 p50/p95/p99 全部通过 paired sign 闸门，
75% mixed-fill 碎片证书也完整恢复。

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

`20f3676` 当前结果如下。Windows/Linux 各 `n=15` 独立进程；Apple M4 在 interactive /
utility QoS 各 `n=31`。数值为 median `[min, max]`：

| 平台 / QoS | HYLE availability | oneTBB availability | HYLE / oneTBB | HYLE / oneTBB recovery |
|---|---:|---:|---:|---:|
| Windows / MSVC 19.50 | 0.6858 `[0.6858, 0.6858]` | 0.4843 `[0.4816, 0.4843]` | **1.4159x** `[1.4159, 1.4239]` | 1.0000 / 1.0000 |
| Linux / GCC 14.2，guest CPU 2 | 0.6858 `[0.6858, 0.6858]` | 0.4816 `[0.4816, 0.4816]` | **1.4239x** `[1.4239, 1.4239]` | 1.0000 / 1.0000 |
| Apple M4 / interactive | 0.6858 `[0.6858, 0.6858]` | 0.4816 `[0.4816, 0.4816]` | **1.4239x** `[1.4239, 1.4239]` | 1.0000 / 1.0000 |
| Apple M4 / utility | 0.6858 `[0.6858, 0.6858]` | 0.4816 `[0.4816, 0.4816]` | **1.4239x** `[1.4239, 1.4239]` | 1.0000 / 1.0000 |

因此，在这一个明确的 75% mixed-fill / selective-free workload 中，HYLE 留出的最大连续对象
约为 oneTBB 的 1.42 倍，且两者在全部释放后都完整合并回空池容量。这不是对任意 allocation
distribution 的普遍碎片保证；Windows 上 oneTBB 的 4 KiB 级变化也保留为原始 range，不用单点
中位数掩盖。M4 两种 QoS 一致只说明该非吞吐 workload 对调度不敏感，仍不声称 P/E 核绑定。

`20f3676` 原始逐进程 CSV：
[Windows](./mem-fixed-pool-fragmentation-win-20f3676.csv)、
[Linux mirror](./mem-fixed-pool-fragmentation-linux-20f3676.csv)、
[Apple M4 interactive mirror](./mem-fixed-pool-fragmentation-m4-20f3676.csv)、
[Apple M4 utility mirror](./mem-fixed-pool-fragmentation-m4-utility-20f3676.csv)。

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
