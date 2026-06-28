# HYLE 性能预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 实测铁证（持续回填,真跑非推算）

> 环境:Linux x86_64 GCC14.2 Release,nanobench。100k uint64 键。`bench/container/`。

### SwissTable hashmap

| 操作 | vs `std::unordered_map` | vs `absl::flat_hash_map`(**SOTA 标尺**) | 结论 |
|---|---|---|---|
| find-hit | **2.42x** 赢 | **~0.88x 输**(prefetch 后,曾 0.76x) | 赢 std,**差距砍半→~12%** |
| find-miss | **2.26x** 赢 | **~0.78x 输** | prefetch 不助(miss 不碰 slot) |
| insert | **5.3x** 赢 | **~1.30x 赢**(我 ~12.3ns / absl ~16.2ns) | **超 SOTA**(04c 融合之功) |
| erase | **3.5x** 赢 | **~1.25x 赢** | **超 SOTA** |

> **2026-06-28 原生 ext4 复测(跑 3 遍,非噪声)**:
> - **改前**:find-hit 72.6/77.1/75.6%(慢 25%)、find-miss 77.3/80.5/80.3%。
> - **学 absl 加 probe 预取后**(`HYLE_PREFETCH` slot,镜像 absl `find_large` 的 `PrefetchToLocalCache`):**find-hit 86.5/86.9/91.2%**(差距 25%→~12%,**砍半**);find-miss 78.0/77.8/77.8%(**不变**——miss 多在 match_empty 终止、不碰 slot,无 load 可藏);insert/erase 仍稳赢。双平台测试全过(Linux 133/133 + Win 10/10)。
> - **剩余 ~12% find-hit / ~22% find-miss**:需 profile/objdump 对比 codegen 进一步定位(候选:H1=identity 省一移位、match_empty codegen)。prefetch 是 find 优化的第一刀,已落地。

**这就是"对标 SOTA"的意义**:只对标 std 会得出"全面 2-5x 碾压"的虚假安心;对标 absl 才暴露 **find 有 ~25% 真实差距**。**已排除三个错误候选**(读码查实):① `eq_` 对 uint64 在 -O3 完全内联成裸 `==`,**无间接可去**;② `Hash<uint64>` 已是内联 `mix64`(splitmix64),**非 XXH3**,hash 不慢;③ **控制字节尾部镜像已实现**(`swiss_table.cc` core_set_ctrl 镜像前 kGroupWidth 字节 + 分配 `cap+kGroupWidth` + memset 尾),group near-end 加载已无回绕掩码。**find 循环结构已几乎等价 absl**——25% 来源**无法从源码结构看出**,需 **profile(perf/objdump 对比两个 find 内循环 codegen)** 定位真因;唯一已知结构差异是 absl 在 probe 起点 **预取槽数组**(100k 表常驻 cache 时收益小,待测)。**结论:这是一个需要 profile-guided 的真优化,非猜改**——留作下一步 container 性能待办。insert/erase 反超 absl 证明设计本身到位。

### sched（fiber 调度）

| 项 | 实测(Linux GCC14) | 参考 |
|---|---|---|
| fiber 上下文切换(resume/yield 对) | **~23 ns**(v2.3.2 实测 22.84ns) | fcontext 量级,对标 boost.context |
| spawn 2000 任务 + drain | **比 `std::async(launch::async)` 快 ~6-9x** | std::async 每任务起 OS 线程(~30μs);fiber ~3-4μs |

> sched bench(`bench/sched/scheduler_bench.cc`)**现已编入默认 build**(提交后解封)。
> 它曾被认为触发一个 P0 崩溃,但一轮穷尽调查(见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md)
> + [`known-issues.md`](./known-issues.md) KI-1)**排除了 HYLE 逻辑缺陷**:TSan 无源码级
> race、MI_DEBUG=3 全量堆校验无损坏、原始崩溃器本体在 Linux 原生 + Windows MSVC 本机
> 重负载下 + 强制迁移 687 万次跨线程 free 均**不复现**。证据(SIGILL = 代码页问题 +
> 全部 instrumented 净 + 原始崩溃全在 9p)指向 **9p/WSL demand-paging 环境假象**,非
> HYLE 缺陷。fiber 切换 ~23ns / spawn ~6-9x vs std::async 数值有效(mimalloc v2.3.2 复测一致)。

## 预期表（开工前推算目标,vs std）


## 先分两桶（混在一起说"快多少"会误导）

- **桶 A — 同操作更省开销（真 drop-in 可比）**：hashmap、分配器、容器。给"倍数"有意义。
- **桶 B — std 根本没有的能力**：fiber、async IO。"倍数"是错的框（无可比物），是「从不能到能」。

## 主表：时间 / 空间分列

### 桶 A：drop-in 可比

| 组件 vs std | 时间（speedup + 复杂度）| 空间（vs std）|
|---|---|---|
| **flat hashmap** vs `unordered_map` | 查找/插入 **2-4×**；均摊 O(1) 同，但常数小（连续布局 cache 命中 + SIMD 16 路探测）| **通常更省**：无 node 的指针 + 每节点分配头；代价是留负载因子（有空槽 + 1 控制字节/槽）。最坏（刚扩容）峰值略高 |
| **专用分配器** arena/bump vs new/delete | 分配 **5-10×+**；O(1) 指针自增 vs free-list 遍历+锁 | **更费**（以空间换时间）：块尾浪费 + 不单独释放，生命周期不齐时占用更久、峰值更高 |
| **通用分配器** mimalloc vs malloc | 多线程 **1.5-3×** 吞吐、p99 更好；单线程仅 10-30% | 碎片通常 ≤ glibc；大页会增驻留 |
| **SmallVector** vs `vector` | 小 N 省 malloc（大赢）；大 N 持平 | **更费**（以空间换时间）：inline 存 N 元素，sizeof 变大（没用满也占）；超 N 退化到堆 |
| **intrusive list/map** vs `list`/`map` | O(1) 摘除、无分配；查找同 | **更省**：hook 内嵌节点，无单独分配 + 无 node 包装 |
| `vector`/`string` 本身 | **≈ 持平** | ≈ 持平。std 这俩已很好，不白捡 |

### 桶 B：std 没有的能力（"倍数"仅作量级参考）

| 组件 vs std 最接近物 | 时间 | 空间 |
|---|---|---|
| **fiber 切换** vs `std::thread` | **~100×**（10-20ns vs 1-2μs）；仅海量任务兑现 | **更费**（以空间换时间）：每 fiber 独立栈（64KB-1MB × M）；vs 无栈协程只占 frame。故 mem/ 做栈池化 + 分级栈大小 |
| **fiber-aware mutex** vs `std::mutex` | 高竞争吞吐高数倍（不阻塞 OS 线程）；无竞争持平 | ≈ 持平（等待队列用 intrusive，零额外分配）|
| **async IO** io_uring/proactor vs 阻塞 IO | 并发连接 **10-100×**；批量提交省 syscall | **远省**：SQ/CQ ring 固定开销 vs thread-per-conn 的每线程栈 |
| **SIMD 数值核** vs 标量 | **4-16×**（AVX2=8/AVX512=16 宽）；vs 自动向量化仅 1.5-4×（transcendental 这类是大头）| 轻微增：对齐 padding + dtype 转换 scratch |
| **统一内存 MemorySpace** vs host/device 拷贝 | DGX/Mac 上 copy 塌缩 no-op（省一次 DMA）| **更省**：不需 host+device 双份镜像 |

## 整体系统层面（Amdahl，别期待"全面快 N 倍"）

局部再快也受非热路径拖累：

- **瓶颈不在这些点** → 整体可能仅 **1.2×**。
- **并发 / IO 密集** → **10×+**。
- **架构连贯性红利**（不打架的双调度器、统一内存零拷贝）不体现为微基准倍数，而是真实混合负载的系统级延迟/吞吐——Seastar/ScyllaDB 那种 shard-per-core 重写报过 **~2× 系统吞吐**，是"架构做对"的现实数字。

## 诚实边界

**很多情况下 HYLE 对 std 几乎零提升**：代码本就 cache 友好、分配少、单线程、低并发时，std 对常见场景真的好，HYLE 白搭。**赢点恰好集中在 AI infra 热路径**——海量并发（fiber）、高 IOPS（io_uring）、hash 密集（flat map）、分配 churn（arena）、数值核（SIMD）。这不是巧合，HYLE 就是为这个 workload 设计的。

## 一句话

不是"处处比 std 快"。是：**hashmap 稳赢 2-4×、专用分配器对的模式 5-10×、SIMD 数值核 4-16×、并发/IO 是数量级（仅高负载兑现），平凡单线程代码 ≈ 持平**；且 arena/fiber/SmallVector 的时间优势**以空间为代价**。HYLE 的价值是把 AI infra 那几条热路径顶到 std 摸不到的天花板，并让它们在同一套底座上不互相拖累。

## 验证（不是空头支票）

每个数字到 [`roadmap.md`](./roadmap.md) 的 benchmark gate 实测对标：
- 批2 mem：对标裸 mimalloc 不退化
- 批3 container：hashmap 对标 absl/F14；simd 数学函数 ULP 精度
- 批4 sched：百万次 fiber 切换基准
- 批5 io：echo/吞吐；compute：CPU+GPU 后端对拍

是骡子是马，gate 上遛。
