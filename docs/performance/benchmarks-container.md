# container 实测铁证（导航 + 当前真相）

> 本页是 [performance 总账](./README.md) 的 **container 模块**入口。  
> **详表与优化史**（长）：[benchmarks-mem-container.md](./benchmarks-mem-container.md)（mem 已迁出，该文件以 container 为主）。  
> **模块收口**（正确性/C ABI，非性能全表）：[../milestones/container-module-closure.md](../milestones/container-module-closure.md)（2026-07-19）。  
> 队列/并发容器的 SOTA 对标多在 sched 消费路径 + `queue_sota_bench`；本页聚焦 hash / map / vec / str / path。IntrusiveList 已归 core，见 [`benchmarks-core`](./benchmarks-core.md)。

## 0. 数据新鲜度（先看这个）

| 面 | 证书状态 | 说明 |
|---|---|---|
| **StringInterner / string 收口表** | ✅ **2026-07-19 三平台** | Win / WSL / M4；lifecycle 落后已诚实登记；2026-07-26 batch 分片锁改动待刷新 |
| **path** | ✅ 双平台 n=15 | Win+Lin；算法基线 + filesystem 分离 |
| **FlatHashMap / FlatHashSet** | ✅ Windows+WSL n=15；⏳ M4 | 同合同 Abseil；WSL map/set miss 是已登记小缺口 |
| **hash / mix64 / Hash&lt;T&gt;** | ✅ Windows/WSL n=15 长 hash SIMD；⏳ M4 | Win ≥1KiB、Linux ≥512B 产品派发；其余历史表待三平台统一刷新 |
| **SmallVector / PodVector** | ⚠️ 主 Win + 部分补测 | owning vector 形状待本轮统一三平台刷新 |
| **ConcurrentVector** | ✅ Windows+WSL n=15；⏳ M4 | append/range iteration 同合同 SOTA；WSL `operator[]` 是已知缺口 |
| **DynamicBitSet / SparseSet** | ✅ **2026-07-26 三平台 n=15** | actual-use graph frontier / frame-local active IDs；M4 bitset 与 std 同档；纯 SparseSet membership 明确输 dense bitmap |
| **RingDeque / ByteRingBuffer** | ✅ **2026-07-26 三平台 n=15** | 固定容量 FIFO 与 32KiB backlog/4KiB stream staging；WSL 与 Folly-shaped floor 同档，Win/M4 胜 floor |
| **并发队列** | 见 queue / sched 文 | 不在本页展开 |

**读法**：标 ✅ 的可当当前产品结论；标 ⚠️ 的方向大体可信，但**不要**当 2026-07-20 式三平台 sign 证书去对外硬报绝对 ns。

## 1. 总览判定（对真 SOTA，非只赢 std）

| 组件 | 真 SOTA 标尺 | 判定（当前文档共识） |
|---|---|---|
| **mix64 / Hash&lt;T&gt; 小 POD** | wyhash / abseil 形 | ✅ 赢或 SOTA 同款（整数 / 16B POD） |
| **hash_bytes ≤16B** | XXH3-small / wyhash | ➖ **inline mix**（不同算法）；非「输 XXH3 wrapper」 |
| **hash_bytes 中长串** | 裸 XXH3 / wyhash | ✅ Win：≥1KiB AVX2；WSL：≥512B AVX2；⏳ M4 |
| **FlatHashMap** | absl::flat_hash_map / Folly F14 | ✅ Win 五项胜；WSL 四项胜、miss 0.97x；⏳ M4 |
| **FlatHashSet** | absl::flat_hash_set / Folly F14 | ✅ 真 set slot/API；Win 四项胜；WSL hit 平、miss 0.94x、insert/iterate 胜；⏳ M4 |
| **SmallVector** | absl::InlinedVector | ✅ 小 N churn **赢** 同款 SOTA |
| **PodVector** | std::vector | ✅ growth 赢（mimalloc 质量）；read 平 |
| **string_ref 搜索** | string_view / libc | ➖ 自研 SIMD 项平；子串/字符集 **转发 std**（刻意） |
| **StringInterner** | map+mutex / Absl 分片 peer | ⚠️ **Win 多平**；WSL/M4 lifecycle **输**；growth 赢；8t 距 Absl ~5% |
| **path 词法** | string_view 扫 / filesystem | ✅ 平算法基线；**碾** filesystem（零分配结构差） |
| **ConcurrentVector** | oneTBB `concurrent_vector` | ✅ Win/WSL：append、range iteration、并发 prefix 读同档或赢；⚠️ WSL `operator[]` 顺扫约 0.51x |
| **DynamicBitSet** | `std::bitset` + portable dynamic blocks | ✅ Win 1M-bit OR+count 1.40x；WSL 5.18x；M4 与 std 同档；⚠️ M4 单循环 fused floor 快 1.53x；⏳ Boost |
| **SparseSet** | dense bitmap + `std::unordered_set` | ✅ frame clear+rebuild Win 3.29x/28.6x，WSL 3.61x/11.6x，M4 1.86x/11.4x；membership 明确输 dense |
| **RingDeque** | Folly fixed-capacity ring shape / `std::deque` | ✅ Win/M4 快 floor 1.06x/1.15x，WSL 与 floor 同档；三平台快 std 1.28–9.20x |
| **ByteRingBuffer** | linear byte buffer+memmove / `std::deque<byte>` | ✅ stream staging Win/WSL/M4 快 linear 1.82x/1.94x/4.26x，快 deque 约118x/48x/121x |

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
内存/生命周期选择扭曲成三数组容器。当前证书覆盖 Windows/MSVC、WSL/GCC14 AVX2
与 M4/AppleClang NEON；本地没有可编译的 Boost dynamic_bitset peer，因此只把
`std::bitset` 写作当前成熟同接口证据，不外推成“所有对手 SOTA”。

### RingDeque / ByteRingBuffer：单线程常态形状（Windows/WSL Release，n=15）

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
正常网络/file/audio 用户可直接把 spans 接 scatter/gather I/O。当前数字只证明 copy
staging；zero-copy 的价值取决于真实 syscall/codec 消费者，不用一个不读写 bytes 的
metadata benchmark 虚报无限带宽。当前证书覆盖 Windows/MSVC、WSL/GCC14 与
M4/AppleClang；三平台 typed/C/surface/header gate 也均通过。

### FlatHashMap / FlatHashSet：真实 typed 形状（Windows/WSL Release，n=15）

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

迭代器不是为 benchmark 另设的旁路：map/set 都提供标准 forward range，内部持有当前
control group 的 occupied mask，用户的正常 range-for 与测试走同一实现。set 共享 map
的 Swiss probe/resize 核心，但 empty mapped policy 不占 slot bytes。Windows 全胜、WSL
四个主要方向领先并不能抹掉 WSL map/set miss 的约 3%/6% 差距；M4 未刷，所以这里
不写“三平台 SOTA”。

### ConcurrentVector：先校正合同，再谈速度（Windows/WSL Release，n=15）

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

WSL 的逐次索引差距与 GCC 对运行时 segment 定位的代码生成有关；它没有被包装成赢。
成熟容器的随机访问 iterator/range 形状已补齐：`++` 缓存 segment，因此正常顺序遍历
在两平台都领先；`operator[]` 继续作为真正随机索引的独立证据与后续优化项。M4 尚未
刷新，不能把本表外推为三平台证书。

## 2. 非 SOTA / 冻结边界（诚实清单）

1. **StringInterner lifecycle（WSL/M4）**
   intern / acquire-retain-release / churn 相对同契约 peer 约 **0.73–0.86x**。
   产品价值在 cached-hash growth、稳定 identity、并发接近强 peer、batch——**不是**「全生命周期常数三平台无条件赢」。旧 1.24x churn 证书已撤回。

2. **StringInterner 8-thread**
   距 sharded Absl+mutex 约 **4–5%**（~0.95x），接近但非碾压。

3. **hash_bytes 默认 DLL 出线核**  
   与裸 XXH3 同算法时有固定 seam；小 typed key 走 `Hash&lt;T&gt;` inline mix。旧
   `hash_bytes_inline` 只服务微基准且泄漏私有 vendor 头，已从公开面删除；较长字节串
   接受动态库边界，真实热点应缓存 hash 或批处理。

4. **string_ref 短输入 split/rfind/count**  
   4KB 点值可有 **0.83–0.96x**（DLL/dispatcher）；长输入摊平。

5. **FlatHashMap/Set WSL miss 与 M4 证书**
   WSL map/set miss 分别约 **0.97x/0.94x Abseil**，其余 map 项领先、set hit 同档；
   map/set 的 M4 n=15 尚未刷新。这个差距没有通过增加常态 metadata 或特化 benchmark
   key 来隐藏。

6. **并发队列类**
   调度/MPMC 性能叙事仍在 queue_sota_bench / sched；单线程 RingDeque/ByteRingBuffer
   已在本页按实际 FIFO/staging 合同单列，不能拿其结果外推到原子队列。

7. **长 hash 的平台证书**
   Windows AVX2 已证明 ≥1KiB、WSL/GCC14 已证明 ≥512B 宽 XXH3 相对 wyhash 的收益；
   两个 shipping compiler 使用各自保守交叉点。M4 NEON 尚未按同一 varied-input 合同
   刷新；非 AVX2 产品保持 wyhash，不把 x86 结论外推成三平台 SOTA。

8. **SparseSet 纯 membership / 新集合类型平台证书**
   SparseSet 的 mixed membership 在 Win/WSL/M4 分别是 dense byte bitmap 的约
   0.27x/0.21x/0.69x，这是选择条件，不是待
   隐藏的小缺口；频繁 O(1) clear/O(active) iteration 才选择 SparseSet。DynamicBitSet /
   SparseSet 当前已有三平台 n=15；本地缺 Boost dynamic_bitset 可编译 peer，后续只在
   参考树具备同合同实现后补，不以网络临时 clone 改变参考集。

## 3. 选型（不同场景不同 container 组成）

```text
整数 / 小 POD key 哈希     → Hash<T> / mix64（热路径）
长字节哈希                 → hash_bytes（自动按发布平台/ISA 选择 wyhash 或宽 XXH3）
通用 hash map              → FlatHashMap
高频 membership / 去重     → FlatHashSet
稠密 mask/frontier/集合代数 → DynamicBitSet
有界 ID、频繁 clear/活跃遍历 → SparseSet（纯 membership 优先 dense bitmap）
单线程双端/外部同步 SPSC    → RingDeque（固定容量使用显式 *_bounded）
字节流 staging              → ByteRingBuffer（copy 或 two-span zero-copy）
栈上小数组                 → SmallVector
POD 堆动态数组             → PodVector（默认 SystemAllocator/mi）
只读字符串视图 + 搜索      → string_ref（子串族借 std）
驻留串 / intern            → StringInterner（接受跨平台 lifecycle 税；要 growth/并发）
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
```

定位单项时可再设 `HYLE_QUEUE_BENCH_VARIANT=hyle|tbb` 与
`HYLE_QUEUE_BENCH_CV_CASE=push|writers|index|iterate|read-write`；正式对比不设这两项。

刷新后：改本页 §0 新鲜度 + 总账对应行；详表回填 [benchmarks-mem-container.md](./benchmarks-mem-container.md)。

## 6. 详表索引

| 主题 | 详节位置 |
|---|---|
| hash 公平性更正 + 长度表 | mem-container §container·hash |
| SwissTable vs absl | mem-container §container·map + README 总账 |
| SmallVector / PodVector | mem-container §container·vec |
| string_ref / StringInterner 三平台收口 | mem-container §container·str |
| path | mem-container §container·path |
| util/obs 夹带 | 应看 util/obs 收口；mem-container「次要模块」勿当 container 主证书 |
