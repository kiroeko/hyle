# HYLE 性能预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 实测铁证（持续回填,真跑非推算）

> 环境:Linux x86_64 GCC14.2 Release,nanobench。100k uint64 键。`bench/container/`。

### SwissTable hashmap

| 操作 | vs `std::unordered_map` | vs `absl::flat_hash_map`(**SOTA 标尺**) | 结论 |
|---|---|---|---|
| find-hit | **2.42x** 赢 | **~1.10x 赢**(12 遍 avg 110.1%;prefetch+mix64+H1,曾 0.76x) | **反超 SOTA** |
| find-miss | **2.26x** 赢 | **≈1.00x 统计平局**(12 遍 avg 97.5%,spread 6.3% > 差距,2/12 遍赢) | **追平 SOTA** |
| insert | **5.3x** 赢 | **~1.40x 赢** | **超 SOTA** |
| erase | **3.5x** 赢 | **~1.39x 赢** | **超 SOTA** |

> **2026-06-28 原生 ext4 复测(每步跑 3 遍,非噪声)——两刀优化,学自 absl**:
> 1. **probe 预取**(`HYLE_PREFETCH` slot,镜像 absl `find_large` 的 `PrefetchToLocalCache`):find-hit 0.76x→**0.88x**(slot load 藏进 ctrl SIMD match);find-miss 不变(miss 不碰 slot)。
> 2. **mix64 换 absl 式 128 位乘折叠**(`(x^k)*k → hi^lo`,单乘短依赖链 vs splitmix64 的 2 串行乘):find-hit **0.88x→1.10x 反超**、find-miss **0.78x→0.92x**(hash 占 miss 成本大头,短链直接砍)。代价:整数 hash 从双射(零碰撞)降为统计级(~1/2⁶⁴),correctness 由 eq_ 兜底;insert/erase **未退化**(1.35x/1.32x)证明碰撞质量没崩。
> - 双平台真编真测:Linux 133/133(含雪崩/分布质量测试)+ Win MSVC `_umul128` 分支编译干净 + 测试过。`__extension__` 消 `__int128` 的 -Wpedantic。
> 3. **match_empty 加 SSSE3 快路径**(`_mm_sign_epi8(c,c)` 1 指令 vs SSE2 重建 kEmpty 广播 4 指令,`#ifdef __SSSE3__` gate + SSE2 兜底):**绝对增益/与 absl ISA 对等**,但**未缩 find-miss 相对差距**——`-mssse3` 下 absl 也用 `psignb`,两边同步提速,find-miss 仍 ~0.89x。**证伪了"match_empty 是 find-miss 差距来源"**。双 ISA 输出穷举验证一致(psignb 确认)+ SSSE3 激活下 134/134。注:默认 SSE2 ship 构建走兜底(no-op),仅 `-march`/`-mssse3` 生效。**[已登记]最终的"编译/运行期自动辨识硬件规格选 ISA 变体"与 simd 支柱统一做**(conventions §10 baseline+dispatch);现阶段 gated 着 + 文档留痕即可,认真部署可先 `-march=native`/提基线 `x86-64-v2`。
> - **find-miss 剩 ~10% 已排除三因**:非 hash(mix64 对齐)、非 prefetch(miss 不碰 slot)、非 match_empty(同 ISA 仍差)。**根源在别处**(probe_seq codegen / H1 的 `>>7` / BitMask 迭代),需逐指令 profile,留作 profile 引导任务。
> 4. **H1=identity**(objdump 逐指令对比发现:absl 用 hash 低位直接当 index、top 7 位当 H2,省掉我们的 `hash>>7` 一条移位):find/insert/erase 内循环各省 1 指令。find-miss **0.89x→~0.96x**、insert/erase 升到 ~1.40x。正确性:mix64 雪崩后低位(index)与高 7 位(H2)独立,探测链不变;SSE2 基线 134/134 + Win MSVC 通过。
> - **12 遍统计收口(回答"4% 差在哪")**:find-hit avg **110.1%**(min 102.5/max 119.4)、find-miss avg **97.5%**(min 94.9/max 101.2,**spread 6.3% > 差距 2.5%,12 遍有 2 遍赢**)、insert **146.6%**、erase **140.0%**。**find-miss 是统计平局**——那"4%"是测量噪声(微基准 ~2-4ns,find-hit 自身 spread 都 16.9%),非结构缺陷,无单点杠杆。**四项全部 ≥ 或追平 absl,SOTA 反例消除。**

**这就是"对标 SOTA"的意义**:只对标 std 会得出"全面 2-5x 碾压"的虚假安心;对标 absl 才暴露 **find 有 ~25% 真实差距**。**已排除三个错误候选**(读码查实):① `eq_` 对 uint64 在 -O3 完全内联成裸 `==`,**无间接可去**;② `Hash<uint64>` 已是内联 `mix64`(splitmix64),**非 XXH3**,hash 不慢;③ **控制字节尾部镜像已实现**(`swiss_table.cc` core_set_ctrl 镜像前 kGroupWidth 字节 + 分配 `cap+kGroupWidth` + memset 尾),group near-end 加载已无回绕掩码。**find 循环结构已几乎等价 absl**——25% 来源**无法从源码结构看出**,需 **profile(perf/objdump 对比两个 find 内循环 codegen)** 定位真因;唯一已知结构差异是 absl 在 probe 起点 **预取槽数组**(100k 表常驻 cache 时收益小,待测)。**结论:这是一个需要 profile-guided 的真优化,非猜改**——留作下一步 container 性能待办。insert/erase 反超 absl 证明设计本身到位。

### simd（向量后端 vs Highway + auto-vec，simd-02）

> 环境:WSL Ubuntu24.04 原生 ext4(非 9p)、GCC14.2 Release、nanobench。4096 个 f32(L1 常驻,隔离计算吞吐非内存带宽)。`bench/simd/`。**Highway = SOTA 标尺**(同 -mavx2 静态 dispatch,与 HYLE avx2 同 ISA、同 kernel 形状,只差抽象层)。**数字 = 4 遍均值(2026-06-29 复测,沿 hashmap 那节的多遍统计纪律,非单遍)。**

| kernel | scalar(auto-vec SSE2) | HYLE sse4(128b) | HYLE avx2(256b+FMA) | Highway avx2(**SOTA**) | 结论 |
|---|---|---|---|---|---|
| **dot product** | 2.75 G op/s (1.0x) | 11.2 G (4.07x) | **12.4 G (4.5x)** | 11.2 G (4.07x) | **追平/微超 Highway**(~1.10x,4 遍均赢);4.5x over auto-vec |
| **saxpy** y=a*x+y | 17.3 G (1.0x) | 17.5 G (1.01x) | **25.1 G (1.45x)** | 20.7 G (1.20x) | avx2 **~1.21x 微超** Highway;saxpy 内存带宽受限故倍数低 |

**结论(回答"比 SOTA 有下降么")**:**没有——HYLE 向量后端追平且小幅微超 Highway**。dot 上 HYLE avx2 ~1.10x、saxpy ~1.21x,**4 遍复测 HYLE avx2 两 kernel 均稳定领先**(非单遍噪声);但领先幅度小(dot 0.08 ns/op 量级,两边内循环都编成同一条 `vfmadd231ps`,objdump 可证),诚实表述是**追平/微超 SOTA**,非"碾压"。compute-bound 的 dot 上 **4.5x over 自动向量化标量**,落在性能文档"SIMD 4-16x / vs auto-vec 1.5-4x"区间(dot 含 reduction,略超 4x 合理)。saxpy 是 3 流内存操作(读 x/读 y/写 y),标量 auto-vec 已近饱和带宽,故 SIMD 倍数天然低——AVX2 靠更宽 load 仍取 1.45x。**SSE4 saxpy ~1.0x(与标量持平)**诚实留账:128 位宽度下该 kernel 已带宽受限,SIMD 无加速空间,非 bug。
>
> **意义**:这验证了"自造精简抽象不牺牲性能"——HYLE 的 vec.h+后端(~340+360 行)在热路径上与 vendor 整个 Highway(单后端 ~3k 行)同速或微超。抽象层零成本(全内联到 intrinsic),没有为"可移植"付性能税。
>
> **范围诚实**:此对标只覆盖 **dot/saxpy 两个 compute/memory-bound 数值核**(AI 热路径主力)。Highway 还有 ~200 op(swizzle/gather/scatter/compress/transcendental)+ 6 ISA,HYLE 现做 ~25 op + 2 ISA 实跑——**"追平"指已实现的核上追平,非功能对等**。广度补全(int 量化类型/更多 op/AVX512 启用)见 milestones simd 延期登记。**正确性**:Ne(NaN)/Neg(±0) 等边界经红队批量审计揪修(2026-06,见 milestones simd-02/03),已加 FloatEdgeCases 覆盖。

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

### sched-05 fiber-aware 同步原语(`bench/sched/sync_bench.cc`)

> 环境:Linux x86_64 GCC14.2 Release(原生 ext4)+ Win MSVC19.50 Release,nanobench。

| 项 | Linux GCC14 | Win MSVC | 说明 |
|---|---|---|---|
| **无争用 Mutex lock/unlock 对** | 9.88 ns(**std::mutex 2.41 ns 的 4.1x**) | 14.94 ns(std::mutex 11.22ns 的 1.33x) | 见下"诚实背景" |
| 争用 Mutex(8 fiber / 4 worker) | 196 ns/op(5.1M op/s) | 148 ns/op | HYLE 自有(无同模型 std 基线) |
| Channel 收发(cap 16,1p1c) | **66 ns/item(15.1M/s)** | 98 ns/item | fiber-aware 有界 MPMC |
| WaitGroup fork-join(64 worker) | 9.4 µs/批 | 1.9 µs/批 | 由 fiber 创建/调度主导,非 sync 本身 |

**waiter-count 门优化(本步实测驱动)**:基准揭示 Linux glibc `std::mutex` 无争用仅 2.5ns(单 CAS),HYLE Mutex 原 14.74ns(慢 **5.9x**)——根因每次 `unlock` 无条件 `notify` 取一次 CV 自旋锁。加 `waiters_` 原子门(无 waiter 停靠时跳过 notify 锁,靠 monitor 锁 handoff 的 happens-before 保证不丢唤醒)后:无争用 Mutex **14.74→9.88ns(1.5x)**、Channel **96→66ns(1.45x)**,**TSan 仍零 race**。这是"先正确、优化靠实测"的范例:门是实测暴露差距后才加的安全优化,非投机。

**诚实背景(为什么 4.1x 不等于"输")**:① `std::mutex` **根本不能用在 fiber 里**——它阻塞的是 worker OS 线程,会卡死该核所有其他 fiber、破坏 M:N。所以这不是同任务对标,std::mutex 是"裸锁成本"基线,4.1x 是**完全 fiber-aware(争用让出而非阻塞)的代价**(绝对值 ~10ns / 1 亿 op/s 仍快)。② marl 的 mutex 其实就是 `std::mutex`(只 CV 才 fiber-aware),故 marl mutex 无争用 = std 速度;HYLE Mutex 更强(本身 fiber-aware)但更贵。**剩余优化登记**:futex 式单原子快路径(`held_` 直接 CAS、3 态编码 waiter)可把无争用降到 ~1.5x std,但 lost-wakeup 面更微妙,需独立 TSan campaign——按"不在压力下赶 lock-free"纪律登记,不本步赶工。**SOTA parity vs marl**(fiber-sync 真标尺)需接 marl 自有 scheduler → 登记 bench-02d。




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
- P2 mem：对标裸 mimalloc 不退化
- P3 container：hashmap 对标 absl/F14；simd 数学函数 ULP 精度
- P4 sched：百万次 fiber 切换基准
- P5 io：echo/吞吐；compute：CPU+GPU 后端对拍

是骡子是马，gate 上遛。
