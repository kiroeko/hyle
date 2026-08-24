# container 性能证据（导航 + freshness 真相）

> 本页是 [performance 总账](./README.md) 的 **container 模块**入口。  
> **历史模块收口**（正确性/C ABI，非性能全表）：[../milestones/container-module-closure.md](../milestones/container-module-closure.md)（上次签署 2026-07-27）。2026-08-25 的 hash/string API 修正与 simdutf 9.1 升级已重开受影响边；本页旧三平台点值在复跑前不是当前整模块证书。
> 调度队列的 SOTA 对标多在 sched 消费路径 + `queue_sota_bench`；并发关联容器已在本页按 shared-registry 正常合同单列。IntrusiveList 已归 core，见 [`benchmarks-core`](./benchmarks-core.md)。

## 0. 数据新鲜度（先看这个）

| 面 | 证据状态 | 说明 |
|---|---|---|
| **StringInterner / string 收口表** | 🗃️ **2026-07 三平台历史输入** | adaptive mutex + 16B cached-hash pointer index；Win/WSL 64B、Apple 128B core spacing。旧 M4 8t normal workload 为 14.678 vs 19.679 ns/op；本轮未重签 |
| **path** | 🗃️ 旧双平台 n=15 | 算法基线 + filesystem 分离；本轮未重签 |
| **FlatHashMap / FlatHashSet** | 🗃️ **2026-07 三平台历史输入** | 同合同 Abseil；旧 WSL map/set miss 与 M4 点值仍可定位回归，不是当前发布结论 |
| **BTreeMap / BTreeSet** | 🗃️ **2026-07-26 三平台历史输入** | ordered index、range scan、build/copy/churn、time expiry 十轴旧证据；本轮未重签 |
| **hash_bytes / mix64** | 🗃️ **上次三平台 n=15** | 算法未改，可作回归输入；17–128B 与宽核 crossover 尚未在本轮 source/peer/compiler 合同下重签 |
| **typed Hash&lt;T&gt; / StringHash** | 🟡 **受影响性能待三平台复跑** | 默认自定义 raw-object hash 已删除，float/double 改为 canonical bit mix，新增 owning/heterogeneous string helper；Windows float/double 点测不是三平台证书 |
| **UTF-8 validate/count** | 🟡 **simdutf 9.1 受影响项** | 一遍 validate-only 与 validate+count 已分开；Windows/GCC 14 WSL invalid/tail/ASCII/non-ASCII correctness 已过。Windows Release 单点约 `0.20 ns/B` validate、`0.21 ns/B` validate+count；当前配置未编同合同 rival，三平台 n=15 性能仍待刷新 |
| **SmallVector / PodVector** | 🗃️ **2026-07 三平台历史输入** | build+consume 防 DCE 的旧矩阵；本轮未重签 |
| **ConcurrentVector** | 🗃️ **2026-07 三平台历史输入** | append/range/prefix+writer 旧证据与 WSL 随机访问边界；本轮未重签 |
| **DynamicBitSet / SparseSet** | 🗃️ **2026-07-26 三平台历史输入** | graph frontier / frame-local active IDs 旧证据；本轮未重签 |
| **RingDeque / ByteRingBuffer** | 🗃️ **2026-07-26 三平台历史输入** | 固定容量 FIFO 与 stream staging 旧证据；本轮未重签 |
| **SpscQueue** | 🗃️ **2026-07-26 三平台历史输入** | exact-capacity 1P/1C 旧证据；本轮未重签 |
| **ConcurrentFlatHashMap / ConcurrentNodeHashMap** | 🗃️ **2026-07-26 三平台历史输入** | oneTBB 同 accessor 旧矩阵；本轮未重签 |
| **FlatMap / FlatSet** | 🗃️ **2026-07 三平台历史输入** | split-storage map / single-array set 旧矩阵；本轮未重签 |
| **并发队列** | 见 queue / sched 文 | 不在本页展开 |

2026-08-25 Windows/MSVC point（单进程方向检查，不是三平台证书）：float bit-mix 0.56 ns/hash
（`std::hash` 0.90），double 0.51（`std::hash` 1.64）；短串默认 process-seeded `StringHash`
1.89，显式 deterministic seed 2.02，`HardenedStringHash`/SipHash-2-4 9.99，`std::hash<string_view>`
6.05 ns/hash。真实 `FlatHashMap` 50% hit query 为 fast 7.24、hardened 17.61 ns/query。它量化了
正常 fast 与显式安全策略的代价边界；WSL/macOS、顺序互换与 n=15 尚未补齐，所以保持 🟡。

**读法**：🗃️ 只表示可用于回归定位的历史输入，不能当当前产品结论；🟡 表示本轮明确受影响且
尚未完成三平台重签。下文旧 ✅/➖/⚠️ 保留当时判定语义，统一受本 freshness gate 覆盖。

## 1. 上次总览判定（对真 SOTA，非只赢 std）

| 组件 | 真 SOTA 标尺 | 历史判定 / 当前待重签输入 |
|---|---|---|
| **mix64 / 整数 Hash、显式 BytewiseHash 16B key** | wyhash / abseil 形 | ✅ 赢或 SOTA 同款（自定义 key 不自动 opt-in） |
| **hash_bytes ≤16B** | XXH3-small / wyhash | ➖ **inline mix**（不同算法）；非「输 XXH3 wrapper」 |
| **hash_bytes 中长串** | 裸 XXH3 / wyhash | ✅ 17–128B 三平台同档或赢；宽输入按产品 crossover，三平台已签 |
| **FlatHashMap** | absl::flat_hash_map / Folly F14 | ✅ Win 五项胜；WSL 四项胜、miss 0.97x；M4 hit/miss/erase/iterate 赢、insert 同档 |
| **FlatHashSet** | absl::flat_hash_set / Folly F14 | ✅ 真 set slot/API；Win 四项胜；WSL hit 平、miss 0.94x、insert/iterate 胜；M4 hit/miss/iterate 赢、insert 同档 |
| **SmallVector** | absl::InlinedVector | ✅ Win/M4 赢；WSL build+consume inline 绝对 0.81x 但合并 spread 内同档，spill 1.22x 赢 |
| **PodVector** | std::vector | ➖ read 三平台平；growth Win/M4 赢，WSL 0.90x 但合并 spread 内同档 |
| **string_ref 搜索** | string_view / libc | ➖ 自研 SIMD 项平；子串/字符集 **转发 std**（刻意） |
| **StringInterner** | map+mutex / Absl 分片 peer | ✅ 8t/growth/lifecycle 三平台赢；single Win/M4 赢；⚠️ WSL 单条 0.894x，batch 赢 |
| **path 词法** | string_view 扫 / filesystem | ✅ 平算法基线；**碾** filesystem（零分配结构差） |
| **ConcurrentVector** | oneTBB `concurrent_vector` | ✅ Win/WSL：append、range iteration、并发 prefix 读同档或赢；⚠️ WSL `operator[]` 顺扫约 0.51x |
| **DynamicBitSet** | `std::bitset` + portable dynamic blocks | ✅ Win 1M-bit OR+count 1.40x；WSL 5.18x；M4 与 std 同档；⚠️ M4 单循环 fused floor 快 1.53x；⏳ Boost |
| **SparseSet** | dense bitmap + `std::unordered_set` | ✅ frame clear+rebuild Win 3.29x/28.6x，WSL 3.61x/11.6x，M4 1.86x/11.4x；membership 明确输 dense |
| **RingDeque** | Folly fixed-capacity ring shape / `std::deque` | ✅ Win/M4 快 floor 1.06x/1.15x，WSL 与 floor 同档；三平台快 std 1.28–9.20x |
| **ByteRingBuffer** | linear byte buffer+memmove / `std::deque<byte>` | ✅ stream staging Win/WSL/M4 快 linear 1.82x/1.94x/4.26x，快 deque 约118x/48x/121x |
| **SpscQueue** | Folly `ProducerConsumerQueue` shape | ➖ 三平台 n=15 **TIE**；同 exact-capacity sentinel / acquire-release 合同，不为测项增加 remote cache 或 power-of-two 特化 |
| **ConcurrentFlatHashMap / ConcurrentNodeHashMap** | oneTBB `concurrent_hash_map` | ✅ protected hit、8-reader、90/10 mixed 三平台均 15/15 胜；flat rebuild 6.25–7.05x；node rebuild M4 胜、Win/Linux 统计同档 |
| **BTreeMap / BTreeSet** | Abseil `btree_map/set` | ⚠️ 实际十轴混合：Windows 7 胜/1 平/2 负，WSL 6 胜/2 平/2 负，M4 6 胜/1 平/3 负；主要 time-index mutation 缺口已消除，但不宣称逐平台全轴 SOTA |
| **FlatMap / FlatSet** | C++23 `std::flat_map/set`（M4）/ sorted vector floor（x86） | ✅ M4 七轴同档或赢；x86 lookup/scan/mutation 同档或赢；AoS raw-copy bulk 只作跨布局下界 |

### BTreeMap / BTreeSet：有序索引真实矩阵（三平台 Release，n=15）

对手是本地 Abseil `btree_map/set`，key/value 均为 `uint64_t`，规模 64Ki entries。
矩阵不是只挑一条 find-hit：点查询同时包含 hit、稀疏 key 域内 miss 与“所有 query 大于
最大 key”的边界 miss；正常消费还覆盖 ordered set membership、lower_bound 后扫描 16 项、
确定性全遍历、乱序事件建索引、复制既有索引、单项 erase+insert churn，以及时间索引
`erase(begin, lower_bound(cutoff))` 后 refill。比值为 HYLE/Abseil throughput，`>`1 更快：

| workload | Windows HYLE / Absl ns | ratio/判定 | WSL HYLE / Absl ns | ratio/判定 | M4 HYLE / Absl ns | ratio/判定 |
|---|---:|---|---:|---|---:|---|
| find hit | **44.906 / 49.989** | **1.107x WIN** | **47.329 / 51.633** | **1.092x WIN** | **41.585 / 44.820** | **1.108x WIN** |
| in-domain miss | **46.298 / 49.782** | **1.078x WIN** | **49.539 / 53.048** | **1.065x WIN** | 45.730 / **41.348** | **0.887x LOSE** |
| above-maximum miss（边界探针） | **15.562 / 21.379** | **1.380x WIN** | 18.164 / **16.996** | **0.932x LOSE** | **17.067 / 19.426** | **1.137x WIN** |
| ordered set membership | 47.085 / 48.095 | 1.011x TIE | 47.503 / 47.444 | 0.994x TIE | **36.360 / 39.370** | **1.073x WIN** |
| lower_bound + scan 16 | 74.341 / **70.934** | **0.956x LOSE** | **69.977 / 74.755** | **1.079x WIN** | 60.342 / 62.839 | 1.029x TIE |
| deterministic iteration | 1.151 / **1.020** | **0.890x LOSE** | 1.021 / 1.021 | 0.995x TIE | **0.820 / 1.055** | **1.287x WIN** |
| build from shuffled events | **65.937 / 73.561** | **1.120x WIN** | **64.511 / 69.103** | **1.062x WIN** | 59.660 / **56.916** | **0.941x LOSE** |
| copy existing index | **2.936 / 11.715** | **3.927x WIN** | **2.803 / 4.330** | **1.523x WIN** | **1.305 / 4.563** | **3.490x WIN** |
| erase + insert churn | **48.064 / 55.442** | **1.159x WIN** | **45.455 / 49.487** | **1.096x WIN** | 41.890 / **41.196** | **0.983x LOSE** |
| time expiry range + refill | **31.641 / 33.293** | **1.051x WIN** | 32.198 / **31.497** | **0.978x LOSE** | **25.463 / 26.107** | **1.027x WIN** |

Windows 的 build/mutation absolute spread 较大，判定采用同进程成对 sign test；Windows
与 WSL 曾被误并行跑在同一物理 CPU 上的一批数据已废弃，没有进入本表。M4 为远程真机。

实现形状来自消费合同：约 256-byte leaf、entry 同时位于 leaf/internal node、unique-key、
双向 iterator 与 `[lower,upper)` range。copy 由逐项重插改为 O(N) 树形克隆后，三平台均
明确领先；范围删除按 leaf run 批量搬移并可直接卸下完全过期页；POD slot 用 memmove。
2026-07-26 profile 又定位出真实 time-index 瓶颈在单调 refill：满页 50/50 split 会让有序
流反复分配半空页，边界 deletion 也会把即将过期的 entry 借来再删。实现现与成熟 Abseil
形状一致地按插入位置偏置 split/rebalance，并在连续 begin/end 删除时延迟非必要 borrow。
这同时改善 append/prepend、churn 与 expiry，没有公开 node 容量、subtree rank、常态 metadata
或 benchmark 专用入口。

仍不能写“B-tree 全轴 SOTA”：Windows full iteration/lower-bound scan、WSL above-max miss/
expiry，以及 M4 domain miss/shuffled build/churn 是当前边界；其中 mutation 剩余差距只有
约 1.7–2.2%，但 paired sign test 仍如实记为 lose。不同平台的赢负方向并不一致，继续为
单条 miss 分布增加边界 cache 或调 node capacity 会扭曲普通用户形状；后续只由跨 workload
profile 证据重开。

### FlatMap / FlatSet：查多改少的排序连续索引（三平台 Release，n=15）

最终实现不再保存 AoS `Entry{key,value}`，而采用 C++23 `std::flat_map` 的 split storage：
keys 与 values 各自连续；FlatSet 则是独立单数组 core。这样 key lookup、只扫 mapped value
和 set 都不会装载无关字段。Windows/WSL 的标准库尚无 `std::flat_map`，因此用公开
sorted-vector wrapper 作为算法 floor；AppleClang 21 已提供真正 C++23 peer。

| 正常用法 | Windows / sorted vector | WSL / sorted vector | M4 / `std::flat_map/set` |
|---|---:|---:|---:|
| lookup hit | **61.234/65.159 ns，1.064x WIN** | **60.680/64.243，1.059x WIN** | **20.570/22.956，1.116x WIN** |
| in-domain miss | **59.513/63.747，1.071x WIN** | **59.834/63.817，1.067x WIN** | **19.827/22.835，1.151x WIN** |
| full key+value scan | 0.243/0.247，TIE | **0.128/0.150，1.17x WIN** | 0.132/0.132，TIE |
| mapped-only scan | 0.184/0.187，TIE | **0.093/0.187，2.01x WIN** | 0.070/0.069，TIE |
| set membership | **59.314/62.895，1.060x WIN** | **58.567/63.365，1.082x WIN** | **23.439/25.086，1.070x WIN** |
| 64 erase + ordered append | **3.122/4.536 us，1.453x WIN** | 3.062/3.054 us，TIE | 3.988/3.975 us，TIE |
| `sorted_unique` bulk | 0.476/0.287 ns，0.603x AoS floor | 0.289/0.213，0.737x AoS floor | **0.408/3.334，8.17x WIN** |

x86 bulk 对手可对 trivial AoS pair 直接 memcpy，而 split storage 必须拆成两列，因此该行是
跨布局 copy 下界，不是同合同完整 flat-map；它也与查找/单列扫描的收益互为取舍。真正
split-storage 的 M4 `std::flat_map` 同合同中，HYLE bulk 反而明显领先。实现通过 fresh
key/value vectors 一次 reserve、原位构造后一次发布 size；异常/OOM 回滚两列完成前缀，
不公开 uninitialized storage、不允许 mutable key，也不添加常态 benchmark metadata。

### DynamicBitSet / SparseSet：按实际生命周期验收（三平台 Release，n=15）

bitset workload 不是单条 intrinsic：预建两个 1M-bit graph frontier，每轮执行正常公开
`operator|=` 再 `count()`。`std::bitset<1M>` 是成熟固定尺寸同接口 peer；portable
`uint64_t` blocks 把 OR 与 popcount 融成一个手写循环，只是可达到的一遍扫描 floor，
不是同接口 peer。表中 ns/op 按 bit batch 归一，括号为每秒处理 bits：

| 1M-bit OR + count | median | paired 判定 |
|---|---:|---|
| HYLE DynamicBitSet（runtime ISA） | **0.004 ns/bit（279.79G bit/s）** | 基线 |
| `std::bitset<1M>` | 0.005 ns/bit（199.75G bit/s） | HYLE **1.40x**；HYLE 15/15 |
| portable `uint64_t` dynamic blocks | 0.019 ns/bit（53.64G bit/s） | HYLE **5.21x**；HYLE 15/15 |

WSL / GCC 14.2 同一合同：HYLE 0.003 ns/bit（286.61G bit/s）；`std::bitset` 0.018
ns/bit（55.06G，HYLE **5.18x**），portable blocks 0.016 ns/bit（63.87G，HYLE
**4.52x**），两组均 HYLE 15/15。Windows 的 std::bitset codegen 更强，所以对固定 peer
领先较小；产品 AVX2 path 在两个 shipping compiler 都成立。

M4 / AppleClang 21：HYLE 0.004 ns/bit（253.78G bit/s），`std::bitset` 0.004
ns/bit（253.97G），paired ratio 统计同档（std 10/15，median 1.008x）；fused portable
loop 0.003 ns/bit（389.42G），快 HYLE **1.53x**。第一版通用 u64-lane popcount 曾以
15/15 慢 std 约 12.3%；NEON count 改为 byte `CNT` + 四路 `UDOT` 累加后追平。
这个优化只改 backend，不增加常态 metadata 或 benchmark-only API。fused floor 的收益
来自一次访存同时 OR+count；在用户确有可复用的融合合同前，不把成熟库普遍分开的
`operator|=`/`count()` 扭成只服务这一测项的接口。

SparseSet workload 模拟每帧 active IDs：universe=1M，每轮 O(1) clear 后插入 4096 个 ID；
另以 64K mixed queries 单独测 membership。dense byte bitmap 是稠密 ID 查询的强 peer，
但每帧必须 O(U) 清零；`std::unordered_set<uint32_t>` 是没有 universe 约束的通用 peer。

| 操作 | HYLE SparseSet | dense byte bitmap | unordered_set | 判定（15/15） |
|---|---:|---:|---:|---|
| clear + insert 4096 | **1.145 ns/ID** | 3.705 ns/ID | 32.439 ns/ID | HYLE **3.29x / 28.6x** |
| 64K mixed membership | 1.380 ns/query | **0.363 ns/query** | 5.560 ns/query | dense **3.77x HYLE**；HYLE **4.03x hash** |

WSL / GCC 14.2：frame clear+insert 为 HYLE 0.959、dense 3.454、unordered 10.814
ns/ID（HYLE **3.61x/11.6x**，均 15/15）；membership 为 HYLE 1.327、dense 0.275、
unordered 2.895 ns/query（dense **4.74x HYLE**，HYLE **2.17x hash**，均 15/15）。
WSL absolute spread 较大，但 paired sign test 六组均同侧 15/15。

M4 / AppleClang 21：frame clear+insert 为 HYLE 1.402、dense 2.604、unordered 16.039
ns/ID（HYLE **1.86x/11.4x**；对 dense 15/15）；membership 为 HYLE 0.693、dense
0.477、unordered 0.945 ns/query（dense **1.45x HYLE**，HYLE **1.36x hash**；对
dense 15/15）。

这个 lose 是结构事实：dense bitmap 一次 indexed byte load，SparseSet 要读 sparse position
再反查 dense；SparseSet 的收益来自不清 U bytes、直接 O(active) 迭代和同时保存 active
ID。不得为赢 membership benchmark 给常态对象再塞一份 bitmap，否则会把本来明确的
内存/生命周期选择扭曲成三数组容器。2026-07 历史证据覆盖 Windows/MSVC、WSL/GCC14 AVX2
与 M4/AppleClang NEON；当时本地没有可编译的 Boost dynamic_bitset peer，因此只把
`std::bitset` 写作该轮成熟同接口证据，不外推成“所有对手 SOTA”。

### RingDeque / ByteRingBuffer：单线程常态形状（三平台 Release，n=15）

FIFO workload 先预留 4096 个 `uint64_t` slot 并保持半满，每轮做 65536 次正常
`try_push_back_bounded + front + pop_front`。`std::deque` 是熟悉通用 peer；portable
bounded ring 完整执行满检查、单调 head/tail 与 power-of-two mask，复现本地 Folly
`FixedCapacityRingQueue` 的算法形状，但不借 glog/build 依赖。它是最小 queue floor，
没有 HYLE 的 allocator、双端、非平凡生命周期、iterator 与 span 合同，不能单独当产品。

| 固定容量 FIFO | HYLE RingDeque | `std::deque<uint64_t>` | portable Folly shape | paired 判定 |
|---|---:|---:|---:|---|
| Windows / MSVC 19.50 | **1.307 ns/op** | 2.101 | 1.390 | HYLE 快 std **1.61x**、快 floor **1.06x**；两组 15/15 |
| WSL / GCC 14.2 | **0.610 ns/op** | 0.783 | 0.622 | HYLE 快 std **1.28x**（15/15）；与 floor 统计同档（10/15 HYLE） |
| M4 / AppleClang 21 | **0.524 ns/op** | 4.823 | 0.600 | HYLE 快 std **9.20x**、快 floor **1.15x**；两组 15/15 |

第一版在每次 pop 后检查 empty 并把两个游标重置为零，GCC 下慢 portable floor 约
12–20%。无符号单调游标本就允许 wrap；Folly 成熟实现也不在 hot pop 归零。删除这条
多余分支/双写后得到上表，不增加 metadata、特化 capacity 或 benchmark-only API。
两端 push/pop 与随机访问仍为 O(1)，只有 grow/reserve 为 O(N)。

Byte workload 模拟 parser/codec staging：保持 32KiB backlog，每轮写入并读出 4KiB，
共 64 轮。HYLE 使用公开 copy API（内部至多两次 memcpy）；linear peer 每轮 append 后
memmove 剩余 backlog；`std::deque<byte>` 使用逐元素 API。数值按处理 byte 归一：

| 32KiB backlog / 4KiB write+read | HYLE ByteRing | linear+memmove | `std::deque<byte>` | paired 判定 |
|---|---:|---:|---:|---|
| Windows / MSVC 19.50 | **0.009 ns/B（115.60 GB/s）** | 0.016（63.35） | 1.024（0.98） | HYLE **1.82x / 约118x**；均 15/15 |
| WSL / GCC 14.2 | **0.009 ns/B（115.14 GB/s）** | 0.017（59.48） | 0.419（2.39） | HYLE **1.94x / 48.2x**；均 15/15 |
| M4 / AppleClang 21 | **0.010 ns/B（99.35 GB/s）** | 0.043（23.33） | 1.216（0.82） | HYLE **4.26x / 约121x**；均 15/15 |

这不是“逐 byte deque”微基准塑形：公开类型同时提供 partial copy 与 two-span zero-copy，
正常网络/file/audio 用户可直接把 spans 接 scatter/gather I/O。该轮数字只证明 copy
staging；zero-copy 的价值取决于真实 syscall/codec 消费者，不用一个不读写 bytes 的
metadata benchmark 虚报无限带宽。2026-07 历史证据覆盖 Windows/MSVC、WSL/GCC14 与
M4/AppleClang；当时三平台 typed/C/surface/header gate 也均通过。

### SpscQueue：正常 1P/1C 持续传输（三平台 Release，n=15）

workload 保持一个 producer 与一个 consumer，传输 262,144 个连续 `uint64_t`，usable
capacity 恰为 1024；线程跨轮次常驻，计时不包含 thread creation。portable peer 复现
Folly `ProducerConsumerQueue` 的 exact-capacity sentinel、分离读写 index 与 acquire/release
合同，不含 HYLE 特化。每个实现单独创建 worker pair，避免 macOS 大小核机器按创建顺序
把两组同时存活的线程稳定分到不同核类。数值为 ns/item 中位数：

| 平台 | HYLE `SpscQueue` | portable Folly shape | paired sign 判定 |
|---|---:|---:|---|
| Windows / MSVC 19.50 | 8.823 | 8.601 | **TIE**，7/15 HYLE 更快 |
| WSL / GCC 14.2 | 9.699 | 11.428 | **TIE**，11/15 HYLE 更快 |
| M4 / AppleClang 21 | 32.362 | 49.083 | **TIE**，12/15 HYLE 更快；p=0.035，未达项目 p&lt;0.01 门槛 |

初版产品采用 power-of-two 单调 counter；这会让刚释放的 consumer slot 立即成为 producer
写目标，并增加 subtract/mask 指令。改为成熟 sentinel 形状后既接受任意正数 exact
capacity，也消除了多余 cache-line 追赶。它是公开合同与常态热路径的架构修正，不是
benchmark-only metadata。M4 曾在两个 rival worker pair 同时存活时出现约 1.58x 假领先；
该批因 P/E placement bias 已废弃，没有进入上表。三平台最终都只签为同档。

### Concurrent associative：同 accessor 合同的 shared registry（三平台 Release，n=15）

workload 使用 32K 个 `uint64_t` key/value 与同一 HYLE hash，三种实现都在查找期间持有
read/write accessor，避免拿无锁裸 pointer lookup 对比受保护引用。四个真实路径分别是：
单线程 protected hit、8 readers 查既有 key、8 threads 90% read/10% exclusive update，
以及 clear 后 8 个 disjoint writers 重建 registry。node/HYLE 与 oneTBB 都预留 2x bucket；
flat 使用 256 shards 并按总规模 reserve。表内均为 ns/op 中位数：

| 平台 / workload | HYLE node | HYLE flat | oneTBB | paired HYLE/oneTBB throughput |
|---|---:|---:|---:|---|
| Windows protected hit | **10.239** | 10.926 | 16.617 | node 1.615x；flat 1.540x；均 15/15 |
| Windows 8 readers | **3.379** | 3.746 | 7.048 | node 2.087x；flat 1.882x；均 15/15 |
| Windows 90/10 mixed | **4.609** | 5.515 | 7.120 | node 1.564x；flat 1.290x；均 15/15 |
| Windows 8-writer rebuild | 76.309 | **12.115** | 90.416 | node median 1.174x、9/15（TIE）；flat 6.250x、15/15 |
| WSL protected hit | 8.910 | **8.327** | 15.749 | node 1.765x；flat 1.893x；均 15/15 |
| WSL 8 readers | **3.133** | 3.528 | 5.857 | node 1.898x；flat 1.639x；均 15/15 |
| WSL 90/10 mixed | **4.258** | 5.054 | 6.434 | node 1.507x；flat 1.255x；均 15/15 |
| WSL 8-writer rebuild | 62.862 | **10.624** | 76.056 | node median 1.210x、12/15（TIE）；flat 7.052x、15/15 |
| M4 protected hit | 7.229 | **5.381** | 10.616 | node 1.451x；flat 1.959x；均 15/15 |
| M4 8 readers | **5.347** | 6.669 | 11.834 | node 2.218x；flat 1.777x；均 15/15 |
| M4 90/10 mixed | **5.576** | 7.034 | 10.165 | node 1.825x；flat 1.443x；均 15/15 |
| M4 8-writer rebuild | 61.529 | **12.690** | 79.669 | node 1.298x；flat 6.275x；均 15/15 |

node 在 read-heavy/accessor workload 更合适，并提供 rehash 后稳定 node address；flat 以
连续 Swiss storage 与 shard-local rehash 换来 compact/rebuild 吞吐。不能因 flat 重建数字
更高就删除 node，也不能把 node 的稳定地址成本塞给所有 cache 用户。Windows/Linux 的
node rebuild 绝对中位数领先，但 paired sign test 未达 p<0.01，因此明确记为 TIE，不写成
三平台全胜。

首版 adaptive RW lock 在每个“最后 reader”释放时都调用 `atomic::notify_all`，即使没有
waiter；libstdc++ 由此把独立 shard 汇入共享 waiter pool，Linux flat read 一度只有约
oneTBB 的 0.22x。当前状态增加 reader-waiting hint，只在真实 waiter/pending writer 存在时
通知；同时 writer unlock 保留竞争 reader 的瞬时计数，park 使用 acquisition loop 观察到
的旧 state，避免 missed transition。这些是锁正确性与常态多 shard 使用所需的机制修复，
没有增添 benchmark API 或每 entry metadata。GCC 14 UBSan 全绿；WSL 新内核下 TSan 需
`setarch -R` 避开 runtime shadow mapping 冲突，13 个 concurrent map/C ABI 用例无 race。

### FlatHashMap / FlatHashSet：真实 typed 形状（三平台 Release，n=15）

对手是本地 Abseil flat hash，不是较弱的 `std::unordered_*`。key 为 100k 个固定随机
`uint64_t`；hit/miss/iteration 使用预填表，insert 使用 fresh table，map erase 包含建表
与删除。HYLE 与 Abseil 分别使用各自默认 hash，因此结果代表用户默认形状，不是强行
共用 hasher 的探测器微测试。

Windows x64 / MSVC 19.50：

| 操作 | HYLE map | Abseil map | HYLE set | Abseil set | n=15 判定 |
|---|---:|---:|---:|---:|---|
| hit | **2.464 ns** | 3.010 ns | **1.891 ns** | 3.136 ns | HYLE 1.21x / 1.66x；均 15/15 |
| miss | **4.499 ns** | 4.801 ns | **4.419 ns** | 5.951 ns | HYLE 1.07x / 1.35x；均 15/15 |
| fresh insert | **11.742 ns** | 24.672 ns | **10.159 ns** | 19.475 ns | HYLE 2.07x / 1.92x；均 15/15 |
| iterate | **0.980 ns** | 1.857 ns | **1.004 ns** | 1.916 ns | HYLE 1.90x / 1.91x；均 15/15 |
| build+erase | **19.396 ns** | 34.748 ns | — | — | HYLE 1.80x；15/15 |

WSL / GCC 14.2：

| 操作 | HYLE map | Abseil map | HYLE set | Abseil set | n=15 判定 |
|---|---:|---:|---:|---:|---|
| hit | **1.669 ns** | 1.910 ns | **1.454 ns** | 1.461 ns | map HYLE 1.15x；set 统计同档（8/15） |
| miss | 3.860 ns | **3.730 ns** | 4.327 ns | **4.059 ns** | **Abseil 1.03x（map 13/15）/ 1.06x（set 14/15）** |
| fresh insert | **10.286 ns** | 14.354 ns | **10.084 ns** | 11.396 ns | HYLE 1.39x / 1.13x；均 15/15 |
| iterate | **0.736 ns** | 1.698 ns | **0.834 ns** | 1.835 ns | HYLE 2.30x / 2.20x；均 15/15 |
| build+erase | **16.668 ns** | 22.900 ns | — | — | HYLE 1.37x；15/15 |

Apple M4 / AppleClang 21：

| 操作 | HYLE map | Abseil map | HYLE set | Abseil set | n=15 判定 |
|---|---:|---:|---:|---:|---|
| hit | **1.613 ns** | 1.699 ns | **1.505 ns** | 1.634 ns | HYLE 1.05x / 1.09x |
| miss | **4.545 ns** | 4.822 ns | **4.437 ns** | 4.749 ns | HYLE 1.06x / 1.07x |
| fresh insert | 8.930 ns | **8.711 ns** | 8.390 ns | **8.187 ns** | 0.975x / 0.976x；统计同档 |
| iterate | **0.661 ns** | 1.500 ns | **0.527 ns** | 1.416 ns | HYLE 2.27x / 2.69x |
| build+erase | **15.166 ns** | 18.065 ns | — | — | HYLE 1.19x |

迭代器不是为 benchmark 另设的旁路：map/set 都提供标准 forward range，内部持有当前
control group 的 occupied mask，用户的正常 range-for 与测试走同一实现。set 共享 map
的 Swiss probe/resize 核心，但 empty mapped policy 不占 slot bytes。Windows 全胜；WSL
map/set miss 仍落后约 3%/6%；M4 查找、删除、迭代赢，fresh insert 与 Abseil 同档。

### ConcurrentVector：先校正合同，再谈速度（三平台 Release，n=15）

旧实现把所有 writer 的元素构造放进一个 mutex；第一版并行改造又为“实时 size 只含
完成连续前缀”增加全局 ordered commit，8 writer 仅为 oneTBB 的 0.17x。该 reader
合同没有生产消费者，也是被单 writer+reader benchmark 诱导出来的错误形状，已撤销。

当前合同与 oneTBB 对齐：append 返回后其 index 可用，或 writers quiesce 后遍历；并发
append 时只能读取事先完成的 prefix。普通 fail-fast append 用一次 `fetch_add` 取号；
可恢复 `try_*` 在领取 index 前完成 allocation，因此失败不会进入 oneTBB 文档允许的
broken-vector 状态。两种错误模型不再互相征税。

Windows x64 / MSVC 19.50：

| 正常用法 | HYLE median | oneTBB median | n=15 判定 |
|---|---:|---:|---|
| 单 writer push 100k | **5.952 ns** | 7.688 ns | HYLE 1.29x；15/15 |
| 8 writer，预留后 append 200k | **32.255 ns** | 35.370 ns | HYLE 约 1.10x；13/15 |
| quiesce 后 range iteration | **0.371 ns** | 1.242 ns | HYLE 3.35x；15/15 |
| quiesce 后逐次 `operator[]` 顺扫 | **0.916 ns** | 1.250 ns | HYLE 1.36x；15/15 |
| 4 readers 读既有 prefix + 1 writer | **0.476 ns** | 0.849 ns | HYLE 1.78x；15/15；peer 绝对方差高 |

WSL / GCC 14.2：

| 正常用法 | HYLE median | oneTBB median | n=15 判定 |
|---|---:|---:|---|
| 单 writer push 100k | **6.299 ns** | 7.359 ns | HYLE 1.17x；15/15 |
| 8 writer，预留后 append 200k | **32.646 ns** | 34.182 ns | HYLE 约 1.05x；14/15 |
| quiesce 后 range iteration | **0.280 ns** | 0.388 ns | HYLE 1.39x；15/15 |
| quiesce 后逐次 `operator[]` 顺扫 | 0.840 ns | **0.427 ns** | **oneTBB 1.97x；15/15，未收口** |
| 4 readers 读既有 prefix + 1 writer | **0.468 ns** | 0.866 ns | HYLE 1.85x；15/15 |

Apple M4 / AppleClang 21：

| 正常用法 | HYLE / oneTBB 比值 | n=15 判定 |
|---|---:|---|
| 单 writer push 100k | **1.093x** | HYLE 15/15 |
| 8 writer，预留后 append 200k | **0.997x** | 统计同档 |
| quiesce 后 range iteration | **1.373x** | HYLE 赢 |
| quiesce 后逐次 `operator[]` 顺扫 | **1.522x** | HYLE 赢 |
| 4 readers 读既有 prefix + 1 writer | **1.270x** | HYLE 赢 |

WSL 的逐次索引差距与 GCC 对运行时 segment 定位的代码生成有关；它没有被包装成赢。
成熟容器的随机访问 iterator/range 形状已补齐：`++` 缓存 segment，因此正常顺序遍历
在三平台都领先；`operator[]` 继续作为真正随机索引的独立证据，且仅 WSL/GCC14 保留
约 0.51x 的平台代码生成边界。

2026-07-26 又验证并**拒绝**了一版把 global index 加 first-capacity 后直接 `bit_width`
的 biased mapping。它把 Windows 索引降到 0.605 ns（oneTBB 1.256 ns），但同一
WSL/GCC14 原生 ext4 构建从旧 mapping 的 15-run median 0.85 ns 恶化到
2.27–2.33 ns（oneTBB 0.43–0.45 ns）。GCC 生成的 `lzcnt -> variable shift -> pointer`
形成更长的串行依赖；两个版本都不能自动向量化动态 segment lookup。因而产品保留旧
mapping：单平台点值改善不足以证明架构更好，也没有为顺扫 benchmark 增加 TLS cache、
每对象游标或常态 metadata。

### StringInterner：稳定 identity、回收与并发驻留（三平台 Release，n=15）

当前索引 slot 是 16B `{Entry*, cached_hash}`，字符串 bytes 只存一份，entry data 直接位于
header 后；growth/erase 不重扫 resident 字符串。对手按合同拆分：单条和生命周期使用
带 mutex、generation/direct-ID 的 peer；8-thread 使用同样自适应分片的 Abseil peer；
growth 的 Abseil 表是故意更弱的无锁 floor，因此只证明 cached-hash/稳定存储的结构收益。

| 正常用法 | Windows | WSL GCC 14 | Apple M4 | 判定 |
|---|---:|---:|---:|---|
| 单条 intern / 同合同 mutex peer | **1.273x** | **0.894x** | **1.533x** | 仅 WSL 单条边界 |
| batch intern / 同合同 mutex peer | **1.545x** | **1.360x** | — | Win/WSL 赢；M4 单条已赢 |
| 8-thread / sharded Abseil peer | **1.782x** | **1.428x** | **1.341x** | 三平台赢；M4 14.678 vs 19.679 ns/op，128B spacing 复证 |
| 1KiB unique growth / Abseil floor | **5.934x** | **1.543x** | **1.290x** | 三平台赢 |
| acquire/retain/release / direct-ID peer | **1.559x** | **1.273x** | **1.608x** | 三平台赢 |
| repeated lifecycle / retained peer | **1.626x** | **1.343x** | **1.649x** | 三平台赢 |

WSL 单条入口的成本来自 16/32/64 shard 的随机 table locality，而单锁 peer 只访问一个表；
把产品退回单锁会牺牲正常 batch/并发路径，或另加 TLS/调用模式元数据。静态链接复测仍为
0.867x，已排除 DSO seam，因此该点作为架构边界冻结，而不是继续按 microbenchmark 调参。

### SmallVector / PodVector：真实构建并消费（三平台 Release，n=15）

旧 benchmark 只观察 `data()`，GCC 能删掉 Abseil 的元素存储，曾制造约 4x 假差；当前每轮
都构建、遍历消费 checksum，并使 inline object 可观察。Windows 的 SmallVector inline/
spill 分别快 Abseil **1.225x/2.124x**；WSL inline 的绝对比值 0.803x，但 combined spread
46.1%，判为同档，spill 赢 1.218x；M4 inline/spill 赢 2.49x/2.67x。PodVector growth 在
Windows/M4 赢 1.545x/1.06x、WSL 与 std 同档；三平台 sequential read 均同档。

## 2. 非 SOTA / 冻结边界（诚实清单）

1. **StringInterner WSL 单条入口**
   相对同合同 single-mutex peer 为 **0.894x**；batch、8-thread、growth 与两种 lifecycle
   均赢。静态链接未改变结论，故不以破坏分片架构来追这个点。

2. **hash_bytes 17–128B 与长输入 crossover**
   17–128B 已改为产品 header-inline multiply stream，三平台对 raw wyhash/XXH3 属 SOTA
   同档或赢；>128B 才进入产品 dispatcher。Win/M4 从 1KiB、Linux 从 512B 选择宽 XXH3，
   阈值来自产品 API varied-input crossover，而非裸算法 microbenchmark。

3. **string_ref 短输入 split/rfind/count**
   4KB 点值可有 **0.83–0.96x**（DLL/dispatcher）；长输入摊平。

4. **FlatHashMap/Set 平台边界**
   WSL map/set miss 分别约 **0.97x/0.94x Abseil**，其余 map 项领先、set hit 同档；
   M4 fresh insert 约 0.975x/0.976x，统计同档，查找/删除/迭代领先。没有通过特化
   benchmark key 或增加常态 metadata 隐藏 WSL miss。

5. **并发队列类**
   SPSC 已在本页按 exact-capacity 1P/1C 正常合同完成三平台证书，结论是同成熟形状统计
   同档；调度/MPMC 性能叙事仍在 queue_sota_bench / sched。单线程 RingDeque/
   ByteRingBuffer 结果不能外推到原子队列。

6. **hash 的算法身份**
   `hash_bytes` 是不可持久化的进程内 table hash，不承诺与 XXH3/wyhash bit-identical；
   这允许按长度、ISA 和 shipping toolchain 选择最快产品核。需要稳定 fingerprint 的用户
   必须选择 codec/crypto 层的明确算法。

7. **SparseSet 纯 membership / 新集合类型平台证书**
   SparseSet 的 mixed membership 在 Win/WSL/M4 分别是 dense byte bitmap 的约
   0.27x/0.21x/0.69x，这是选择条件，不是待
   隐藏的小缺口；频繁 O(1) clear/O(active) iteration 才选择 SparseSet。DynamicBitSet /
   SparseSet 当前已有三平台 n=15；本地缺 Boost dynamic_bitset 可编译 peer，后续只在
   参考树具备同合同实现后补，不以网络临时 clone 改变参考集。

8. **ConcurrentNodeHashMap 并发重建**
   Windows/Linux 中位数分别快 oneTBB 1.174x/1.210x，但只有 9/15、12/15 同侧，未达到
   paired sign 门槛；按统计同档。稳定 node identity 是该类型的选择理由，纯 clear+rebuild
   优先 `ConcurrentFlatHashMap`，不为追单项数字扭曲 node storage。

9. **FlatMap/FlatSet 的剩余边界**
    M4 对真正 C++23 `std::flat_map/set` 七轴同档或赢。x86 的 `sorted_unique` bulk build
    对 raw AoS `vector<pair>` copy floor 分别为 0.603x/0.737x；这是 split key/value layout
    与跨布局 memcpy 的下界，不是完整 flat-map 对手。作为回报，mapped-only scan 在 WSL
    快 2.01x，查找/scan/mutation 同档或赢；不为单个 bulk floor 放弃 split storage。

## 3. 选型（不同场景不同 container 组成）

```text
整数 key 哈希              → Hash<T> / mix64（热路径）
representation-equality key→ 显式 BytewiseHash<T>；其他自定义 key 用语义 hasher
长字节哈希                 → hash_bytes（自动按发布平台/ISA 选择 wyhash 或宽 XXH3）
通用 hash map              → FlatHashMap
高频 membership / 去重     → FlatHashSet
查多改少的连续有序索引     → FlatMap / FlatSet
持续变更的 range / 时间窗口 → BTreeMap / BTreeSet
共享 compact map / 高频重建 → ConcurrentFlatHashMap
共享 registry / 稳定地址    → ConcurrentNodeHashMap
稠密 mask/frontier/集合代数 → DynamicBitSet
有界 ID、频繁 clear/活跃遍历 → SparseSet（纯 membership 优先 dense bitmap）
单线程双端/外部同步 ring    → RingDeque（固定容量使用显式 *_bounded）
跨线程恰好 1P/1C FIFO      → SpscQueue（exact capacity；满/空返回 false）
字节流 staging              → ByteRingBuffer（copy 或 two-span zero-copy）
栈上小数组                 → SmallVector
POD 堆动态数组             → PodVector（默认 SystemAllocator/mi）
只读字符串视图 + 搜索      → string_ref（子串族借 std）
驻留串 / intern            → StringInterner（WSL 单条极限优先时先批处理；growth/并发/生命周期赢）
路径词法（无 syscall）     → path（零分配；勿用 filesystem 当算法对手）
MPMC / stealing 队列       → bounded / sharded / WS deque（见 queue 文档）
```

## 4. 与 mem 的边界

- 容器**默认** `SystemAllocator`（平 mi）；注入 vtable 付多态税 → [benchmarks-mem.md](./benchmarks-mem.md)。  
- Swiss / StringInterner 内部用自有表与 slab，**不要**拿 `SharedPool` / `SharedPoolCache` 当通用 map 分配器。

## 5. 复跑（刷新 ⏳ 项时）

```bash
# hash
scripts/bench-stat.py <build>/bin/hyle_hash_bench -n 15 ...

# map/set vs absl（需 -DHYLE_BENCH_ABSL；分别复跑）
HYLE_ABSL_BENCH_FILTER=map scripts/bench-stat.py <build>/bin/hyle_absl_bench -n 15 ...
HYLE_ABSL_BENCH_FILTER=set scripts/bench-stat.py <build>/bin/hyle_absl_bench -n 15 ...

# ordered map/set actual-use matrix vs Abseil B-tree
scripts/bench-stat.py <build>/bin/hyle_container_btree_bench -n 15 \
  --pair "absl::btree_map" "HYLE BTreeMap" \
  --pair "absl::btree_set" "HYLE BTreeSet"

# sorted-contiguous map/set；无 std::flat_* 时对称 peer 是同布局排序 vector
scripts/bench-stat.py <build>/bin/hyle_container_flat_sorted_bench -n 15 \
  --pair "sorted std::vector peer" "HYLE FlatMap"

# vec
scripts/bench-stat.py <build>/bin/hyle_container_misc_bench -n 15 ...

# string / StringInterner
scripts/bench-stat.py <build>/bin/hyle_string_ref_bench -n 15 ...
scripts/bench-stat.py <build>/bin/hyle_string_buf_bench -n 15 ...

# path
scripts/bench-stat.py <build>/bin/hyle_path_bench -n 15 ...

# ConcurrentVector only（queue bench 其余 workload 不运行）
HYLE_QUEUE_BENCH_FILTER=concurrent-vector \
  scripts/bench-stat.py <build>/bin/hyle_queue_sota_bench -n 15 \
  --pair "HYLE ConcurrentVector" "tbb concurrent_vector"

# DynamicBitSet + SparseSet actual-use workloads
scripts/bench-stat.py <build>/bin/hyle_container_set_bench -n 15 \
  --pair "HYLE DynamicBitSet (runtime ISA)" "std::bitset<1M> (fixed-size peer)" \
  --pair "HYLE SparseSet<uint32_t>" "dense byte membership"

# single-thread ring + byte staging
scripts/bench-stat.py <build>/bin/hyle_container_ring_bench -n 15 \
  --pair "HYLE RingDeque bounded" "std::deque<uint64_t>" \
  --pair "HYLE RingDeque bounded" "portable bounded ring (Folly shape)" \
  --pair "HYLE ByteRingBuffer copy API" "linear byte buffer + memmove"

# bounded SPSC；benchmark 内让两个 rival worker pair 分时存活，避免异构核 placement 偏差
scripts/bench-stat.py <build>/bin/hyle_container_spsc_bench -n 15 \
  --pair "HYLE SpscQueue" "portable Folly ProducerConsumerQueue shape"

# concurrent associative vs oneTBB（其余 pair 见 benchmark 标签）
scripts/bench-stat.py <build>/bin/hyle_container_concurrent_map_bench -n 15 \
  --pair "oneTBB concurrent_hash_map 8R" "HYLE ConcurrentNodeHashMap 8R" \
  --pair "oneTBB concurrent_hash_map 8R" "HYLE ConcurrentFlatHashMap 8R" \
  --pair "oneTBB concurrent_hash_map mixed" "HYLE ConcurrentNodeHashMap mixed" \
  --pair "oneTBB concurrent_hash_map mixed" "HYLE ConcurrentFlatHashMap mixed"
```

定位单项时可再设 `HYLE_QUEUE_BENCH_VARIANT=hyle|tbb` 与
`HYLE_QUEUE_BENCH_CV_CASE=push|writers|index|iterate|read-write`；正式对比不设这两项。

刷新后：改本页 §0 新鲜度与对应表行；不要再建立第二份 container 当前性能表。

## 6. 当前表索引

| 主题 | 详节位置 |
|---|---|
| hash 公平性更正 + 长度表 | 本页 §hash / README 总账 |
| SwissTable vs absl | 本页 §FlatHashMap/FlatHashSet + README 总账 |
| BTreeMap/Set vs Abseil | 本页 §BTreeMap/BTreeSet + README 总账 |
| FlatMap/Set vs sorted-contiguous peer | 本页 §FlatMap/FlatSet |
| SmallVector / PodVector | 本页 §SmallVector/PodVector |
| string_ref / StringInterner 三平台收口 | 本页 §StringInterner + README 总账 |
| path/string-ref | README 总账；下一次刷新直接扩本页，不建立第二份当前表 |
