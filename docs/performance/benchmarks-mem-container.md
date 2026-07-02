# mem / container 实测铁证

> 本页是 [performance 总账](./README.md) 的 mem + container(vec/list/str/path) + 次要模块详节。环境:Win MSVC19.50 + Linux GCC14.2 Release(原生 ext4),nanobench,`scripts/bench-stat.py` 15 遍统计。

## mem 分配器（`Allocator` vtable vs 裸 mimalloc vs std::malloc，`allocator_bench.cc`）

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4 15 遍中位数+spread**(`scripts/bench-stat.py`),nanobench,200k/50k 小块(16..511B)。**两个标尺**:① 裸 `mi_malloc`(设为相对基线)——但它**非多态、全内联**,对它的差读的是"多态 vs 无多态"的物理代价,非接缝缺陷;② **同类 SOTA = `std::pmr::polymorphic_allocator` over mimalloc**——和 HYLE 运行时 vtable 同为"同一 backing 上的运行时多态"(各付一次 dispatch),这才是 apples-to-apples"我们的运行时路径是否 SOTA"。`std::malloc` = 弱基线。

| 场景 | 裸 mi(无多态) | HYLE vtable(运行时) | std::pmr(同类 SOTA) | HYLE SystemAllocator(编译期) | 结论(Linux 15 遍) |
|---|---|---|---|---|---|
| **churn 空循环**(alloc+free) | 7.65 ns | 8.14 ns(0.935x) | 9.87 ns(0.78x) | **7.84 ns(0.975x)** | **vtable 统计平局裸 mi**(gap 6.5%«spread 23%);赢同类 std::pmr 1.21x;编译期路径≈裸 mi |
| **alloc+touch+free**(真实) | 7.41 ns | 8.24 ns(0.906x) | — | — | 统计平局(gap 9.4%«spread 23%) |
| **alloc-all/free-all**(批) | 9.38 ns | 9.70 ns(0.970x) | — | — | 统计平局 |
| **8 线程并发** | 1.65 ns/op | 1.73 ns/op(0.954x) | — | — | 统计平局(per-thread heap) |
| **realloc grow chain** | 72 ns | 76 ns(0.946x) | — | — | 统计平局(gap 5.4%«spread 21%) |

**诚实结论(统计法修正:vtable 封装税在噪声内,不是"慢 10%")**:
- **运行时 `Allocator` vtable**(opt-in 路径,给 arena/mock/跨 DLL):**Linux 统计法下与裸 mi 全部统计平局**(5 个 workload 的 gap 3-9% 全小于 ~22% 的 run-to-run spread)——此前 Win 单遍读出的"慢 8-12%"在多遍统计下站不住,vtable 的一次间接调用在真实分配开销面前测不出来。**对同类 SOTA `std::pmr::polymorphic_allocator`(也一次虚调用、同 mimalloc backing)反超 1.21x**(HYLE 8.14 vs pmr 9.87ns)。即:在"运行时多态分配器"这个类里 HYLE 是 SOTA,且封装税相对裸 mi 在测量精度内为零。原 +21% 才是真缺陷(已修)——① 单 5 参 `ReallocFn` 在 Win64 压第 5 参 + 派发分支 → 拆 3 函数指针;② `normalize_alignment` 常开 FATAL_ASSERT → 默认对齐短路。
- **编译期 `SystemAllocator`**(默认路径,host RAM):`allocate` 内联直调 `mi_malloc`,**实测 0.975x 裸 mi(统计平局)**。前向声明 5 个 mi_* 不暴露 mimalloc.h(用户点破"内联≠暴露头"假两难)。**默认场景零接缝税,与裸 mimalloc 持平**——这就是"C-ABI 也能最强":默认编译期绑定 0 税,运行时注入是显式 opt-in,实测那一档也统计平局裸 mi、且反超 std::pmr。
- **结论:vtable 对裸 mi 统计平局(非"慢 10%"),且在多态这一类里反超 std::pmr 1.21x**;默认编译期路径同样平裸 mimalloc。**头内联容器簇已全量迁移**(SmallVector/BoundedQueue/ShardedQueue/WsFifoQueue/WorkStealingDeque/ConcurrentVector):默认参 `SystemAllocator`,`[[no_unique_address]]` 去掉每实例 32 字节运行时句柄(空间复杂度↓),运行时 `mem::Allocator` 降 opt-in(sched 等需注入者显式用)。type-erased 核容器(CArray/SwissTable/string_buf)保持运行时句柄(出线核服务所有 T,不内联,这是正确模型非缺口)。双平台 + WSL TSan 验证。

## mem 专用分配器簇（arena/pool/stack vs `std::pmr` + 裸 mimalloc，`arena_pool_bench.cc`）

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4 15 遍中位数+spread**(`scripts/bench-stat.py`),nanobench,8192×64B。**SOTA 标尺 = `std::pmr`**(标准库自带:bump arena=`monotonic_buffer_resource`、单线程 pool=`unsynchronized_pool_resource`、**多线程共享 pool=`synchronized_pool_resource`**,完美 drop-in,无需 clone)+ 裸 `mi_malloc`(=`new/delete` 的实际 backing)。**此前这簇零对标——本节真测揪出无锁变体的单点 CAS 争用(已分别根治:MemoryArena 分片 278x、Arena 去原子),并把每个工具放到正确的同契约对手上**:bump→monotonic、单线程池→GrowablePool vs unsync_pool、**多线程共享池→MemoryArena vs synchronized_pool(赢 44x)**。`MemoryArena` 单线程是选错工具(已下 ledger,见下警示)。

| bump 场景(alloc N + 批量释放) | ns/op(Linux) | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE Arena**(单线程 bump,池块) | **0.34** | **~12x** | ✅ pointer-style bump 后 0.74→0.34;赢 std::pmr monotonic 2.2x(见下) |
| **HYLE StackAllocator**(单线程自有 slab) | **0.34** | **~12x** | ✅ pointer-style bump 后 0.68→0.34,平 Arena |
| std::pmr::monotonic_buffer(SOTA) | 0.74(Lin)/1.49(Win) | — | **pointer-bump 后 HYLE(0.34)双平台赢**:Linux 赢 2.2x(pmr 0.74)、Win 赢(pmr 1.49)。此前 offset 版 HYLE 0.74ns 与 Linux pmr 打平/略输,改 pointer-style bump 后拉开——原"Linux pmr 反略快"结论已被 ④ 优化推翻 |
| 裸 mi_malloc(general,无批量) | 3.98 | 1.0x | — |

| pool churn 场景(**单线程**同尺寸 alloc/free) | ns/op(Linux) | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE GrowablePool**(单线程对象池) | **1.28** | **3.5x** | ✅ 单线程对象池的对工具(gap 252%»spread 23%) |
| 裸 mi_malloc/free(general) | 4.51 | 1.0x | — |
| std::pmr::unsync_pool(std 单线程池) | 11.61 | 0.39x | — |
| ⚠️ MemoryArena/ObjectPool(单线程) | 13.97/13.98 | 0.32x | **选错工具,非对标项**:无锁共享池单线程白付每 op 原子 CAS(详见下)——已下 ledger |

> **⚠️ "选错工具"明示**:`MemoryArena`/`ObjectPool` 是**为多线程共享有界池**设计的(无锁分片 Treiber 栈)。**单线程**同尺寸 churn 用它们是误用——每次 alloc+free 各付一次原子 CAS(两条 `lock cmpxchg`)+ tls homing + 地址路由,实测 13.97ns 慢裸 mi 3.1x、连 std 单线程池(`unsynchronized_pool` 11.6ns)都不如。这不是 HYLE 的缺陷,是拿错了工具——**单线程对象池请用 `GrowablePool`**(同 workload 1.28ns,3.5x 赢裸 mi)。故此误用对比**已从对标 ledger 移除**(留此行仅作选型警示),`MemoryArena` 的真实对标是多线程共享场景(见下,赢 std 共享池 44x)。usage 选型见 [`../guide/usage-guide.md` §3.1.1](../guide/usage-guide.md)。

**8 线程 pool churn——两个诚实问题分开:① 共享池 vs 共享池 SOTA(公平对标)② per-thread 模式参照:**

| 8 线程同尺寸 churn | ns/op(Linux) | 结论 |
|---|---|---|
| **HYLE MemoryArena**(共享固定池) | **2.5–2.7** | ✅ **赢真强 SOTA oneTBB scalable_allocator ~3x**(tbb ~7–9ns,per-thread cache+共享后备;HYLE 无锁分片) |
| oneTBB scalable_allocator(真强共享 SOTA) | ~7–9 | 真正的 MT-scaling 共享分配器——**这才是共享池天花板对手**(2026-07 补;tbb warm-up 敏感,单进程内 warm 值,cross-process 更高) |
| std::pmr::synchronized_pool(std 弱共享池) | 66–113 | std 唯一同契约但**实现弱**(一把 shared_mutex 串行)——HYLE 赢它 25–44x,但**这不是 SOTA 天花板**,tbb 才是 |
| **HYLE GrowablePool**(每线程一个池) | **0.63** | ✅ per-thread 模式赢裸 mi per-thread heap 2.04x(min 1.89x) |
| 裸 mi(per-thread heap) | 1.26 | per-thread 模式参照(**不同契约**,非共享池对手) |

**核心诚实结论(单点原子两处已修;并把每个工具放到正确的同契约对手上——红队纠错)**:

1. **"专用分配器 5-10x vs new/delete"全部坐实**:`StackAllocator`(pointer-style bump,0.34ns)+ `Arena`(0.34ns)**~12x 赢裸 mi**;`GrowablePool`(单线程 free-list,1.28ns)**3.5x 赢裸 mi、3.0x 赢 std::pmr pool**(gap 252%»spread 23%,铁证)。**vs std::pmr arena 由"平台相关"转"双平台赢"**:pointer-style bump(遗留优化④,offset→cur_/end_)把 HYLE 从 0.74→0.34ns,现赢 Linux monotonic(0.74)2.2x、赢 Win monotonic(1.49);此前 offset 版 0.74ns 与 Linux pmr 打平的结论已被 ④ 推翻。

2. **`MemoryArena` 曾是全场最慢——单点 CAS 争用,已分片根治,对真强 SOTA 反超 ~3x**:原 clay 继承的单 `head_` 让 8 线程 CAS 同一 cache line,O(threads) 串行化(8 线程慢 **278x**)。**修复**(`memory_arena.h`):内部分 N 片(片数=核数向上取 2 的幂,上限 128),每片独立 Treiber 头占独立 cache line,线程按 tls 序号 homing 到自己片(同线程 alloc/free 无争用),free 按地址 O(1) 路由回属主片。**复杂度落后(O(N)争用→O(1) per-shard)消除**,双平台 10/10 + WSL TSan 零 race。这同时补上了 mem-design 列为"硬约束"却从未实现的 per-core 池。**公平对标(2026-07 补强对手,回答"44x 是否虚高")**:对 **oneTBB scalable_allocator**(真正的 MT-scaling 共享分配器,per-thread cache+共享后备)8 线程 HYLE 2.5–2.7ns vs tbb ~7–9ns = **赢 ~3x**;对 `std::pmr::synchronized_pool`(std 唯一同契约但一把 shared_mutex 串行的**弱**池)是 25–44x。**44x 曾被当"共享池 SOTA 胜利"是虚高**——那只是碾 std 的弱实现,真正的天花板对手是 tbb,对它 HYLE 赢 ~3x(仍是真胜,但不是 44x)。**单线程用它是选错工具**(慢裸 mi 3.1x,该用 `GrowablePool`),那不是缺陷而是误用。

3. **`Arena` 无锁 bump 同源病已修(去原子化)+ pointer-style bump(④)**:`Arena` 的单 `offset_` 原子曾是单点病——单线程付每 alloc 一次 CAS(6.79ns vs 纯指针 bump,10x 税)。bump cursor 本质单游标无法分片,SOTA 模式是 **per-thread(各自一个 Arena,零原子)**。**修复**:`offset_` 改 plain,纯指针 bump(6.79→0.74ns);测试改 `PerThreadArenasNoOverlap`。**再经遗留优化④**(offset→cur_/end_ pointer-style bump)砍到 **0.34ns**——objdump 证 offset 版每 alloc 多一次 base load + base±offset 往返两遍,pointer 版省 ~5 指令。平 StackAllocator、赢裸 mi ~12x、双平台赢 std::pmr monotonic(见结论 1)。

4. **共享池对标的两次纠错——先纠"错比 per-thread heap",再纠"只比 std 弱池虚高 44x"**:第一次(此前):把"8 线程**共享一个**固定池"对"mimalloc 的 **8 个 per-thread heap**"比(0.51x)并写成"无法优化的诚实标价"——错在契约不同(共享有界池 vs 各线程私有堆),补同契约的 `std::pmr::synchronized_pool` 后 HYLE 反超。**第二次(2026-07,回应"44x 可疑")**:synchronized_pool 虽同契约但**是 std 里最弱的实现**(一把 shared_mutex 串行,66–113ns),44x 因此虚高、被当"共享池 SOTA 胜利"不诚实。**补真强对手 oneTBB scalable_allocator**(production MT-scaling 分配器):HYLE 2.5–2.7 vs tbb ~7–9ns = **赢 ~3x**——这才是共享池天花板对标。**教训(红队精神再进一步)**:"找到同契约对手"还不够,还要问"这个对手是不是该类的**强**实现"——synchronized_pool 同契约但弱,拿它得 44x 等于第一次错误的镜像(这次是对照太弱而非契约不符)。HYLE 赢 tbb ~3x 是真胜,但 44x 是虚高数,已改。mimalloc per-thread heap(1.26ns)仍仅作 per-thread 模式参照(那模式的对等答案是 `GrowablePool`,赢 mi 2.04x)。

5. **进一步逼近 mimalloc per-thread heap 的绝对延迟需 seastar 模型**:若拿共享池硬比 mimalloc 的 per-thread heap 绝对值(1.26ns),HYLE 共享池 2.51ns 的差距是"一块固定 slab 跨线程共享"的物理协调代价(每 op 一次原子 CAS,分片已消除 O(threads) 争用)。要把这个常数也逼到 per-thread heap 水平需 seastar 模型(线程钉死 per-core 终生、池生命周期长于线程),`MemoryArena` 的通用生命周期给不出该契约。但这**不是对标失分**——共享池的 SOTA 对手是 synchronized_pool(已赢 44x),per-thread heap 是另一契约(其对手是 GrowablePool,已赢 2.04x)。两个契约 HYLE 各自的对等答案都赢。

## container·vec / ·list —— SmallVector / IntrusiveList / CArray（vs std + 真 SOTA，`container_misc_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。声明承重容器。**标尺升级**:不止 std——SmallVector→`absl::InlinedVector`(small-vector SOTA,gated `-DHYLE_BENCH_ABSL`)、IntrusiveList→`boost::intrusive::list`(侵入容器 canonical SOTA,gated `-DHYLE_BENCH_BOOST`)、CArray→`std::vector`(动态数组的 drop-in SOTA,本身已是高度调优,always-on)。

| 场景 | HYLE | std | 真 SOTA(gated) | 结论 |
|---|---|---|---|---|
| **小 N churn**(建+填 8 int+析构) | SmallVector<int,16> **0.41 ns** | vector 16.8 / vector+reserve 3.06 | **absl::InlinedVector 0.84–1.0** | ✅ **赢 absl 2.0–2.4x**(absl 侧 spread 大)、赢 std 41x |
| **list churn**(push 4096 + pop 全) | IntrusiveList **3.15 ns** | list 25.0 | boost::intrusive **2.77**(平) | **8x 赢 std、平 boost SOTA**(2026-07 实测) |
| **动态数组 growth**(push_back 4096 int,无 reserve) | CArray<int> **0.51 ns** | vector **0.75** | (std::vector 即 SOTA) | ✅ **赢 std::vector 1.48x**(grow 用 reallocate) |
| **动态数组 read**(顺序求和 4096) | CArray<int> **0.18 ns** | vector 0.18 | (std::vector 即 SOTA) | ➖ **平 std::vector**(arr[i] 单 load 兑现) |

**诚实结论**:① **SmallVector 反超 absl::InlinedVector 2.0–2.4x**——不是"只赢 std":absl::InlinedVector 是 small-vector 的 SOTA(同 inline 存储设计),HYLE 0.41ns vs absl 0.84–1.0ns(absl 侧 nanobench 标 unstable、spread 大,故取范围),小 N churn 真赢同款 SOTA(也赢 std::vector 41x);溢出路径同样赢 absl 2.7x。② **IntrusiveList 8x 赢 std::list + 平 boost::intrusive SOTA(2026-07 实测,补 ⏳待补)**:零节点分配。checkout boost submodule(intrusive+config/assert/core/move/static_assert 依赖闭包)后实测:push/pop churn HYLE 3.15 vs boost 2.77(ratio 0.88,gap 12%«spread 44%=**统计平局**)、任意位置 O(1) remove HYLE 3.13 vs boost 3.15(平)。两者同 **circular-sentinel 双向链**形状(canonical intrusive),无抄漏——HYLE `unlink` 额外清 prev/next=nullptr 是 unlink-安全特性(支持 `linked()` 查询 + 安全重复 unlink),非性能差。list 项无可反哺的前轮优化(纯指针操作,不涉分配/hash/SIMD),测试项已覆盖 push/pop/front/back/任意 remove/empty 六种 + 两 bench 场景(churn + 乱序 unlink)。③ **CArray 反超 std::vector ~1.48x growth、平 read**:growth 赢的**真根因(2026-07 纠错)**——旧文档称"经 `reallocate()` mimalloc 原地扩展、无 copy"是**双重错**:① 当时代码根本没用 reallocate(是 `allocate`+`memcpy`+`free`,与 std::vector 同形);② 几何增长(4→8→16…)常跨 mimalloc size-class,**即便用 reallocate 也多是 relocate+copy,in-place 很少触发**。真正赢 std 的因是 **mimalloc 分配器质量**(比 std::vector 的 `operator new` 快)+ header-before-data 零索引开销,不是"原地无 copy"。现已把 grow **真改用 `mem::reallocate`**(兑现文档、且是更简 API：≤ 手动路径,偶尔省一次 copy),实测 growth 0.53→0.51ns、1.34x→1.48x(边际,证实 in-place 少触发);read 路径 arr[i] 单 load 与 vector 持平(header-before-data 布局零索引开销)。

## container·str —— string_ref / StringBuf / string_algo（`string_ref_bench.cc` / `string_buf_bench.cc` / `string_algo_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,~4KB 文本(L1 常驻),15 遍统计(`bench-stat.py`)。**标尺 = `std::string_view`**(string_ref 是其 drop-in 替代,conventions §7:同 16 字节 POD + null_terminated 位,同契约对照公平)+ `std::count` / simdutf。**至此 container·str 无真输项——全线追平或反超。** 分两条腿走(2026-07 定策):HYLE **自研**那些它能追平/反超的(find(char)/rfind(char)/count 走 SSE2+AVX2 运行期 dispatch、starts_with 走 memcmp、split/trim 零分配子视图);对 MSVC STL 手写向量化汇编(vector_algorithms.cpp:two-way 子串搜索 + PCMPESTRI 字符集扫描)**实测更强、自研 PCMPESTRI 只到 0.76x 追不动**的子串/成员族,**转发 `std::string_view`**——转发只在 `string_ref.cc` 内(include `<string_view>`),公开头不引任何 std string 头、不进 DLL 导出符号、不破 C-ABI seam。即"自研赢得过的留自研,std 更强的就借,不为纯粹性硬造打不过的轮子"。

| op | HYLE | std::string_view | vs SOTA | 结论 |
|---|---|---|---|---|
| **find(char) 全扫 miss** | 0.02/B | memchr 0.01 | ~1.0x | ➖平(自研 memchr;原 scalar 输 ~20x) |
| **starts_with(16B)** | 1.79 | memcmp 1.77 | ~1.0x | ➖平(自研 memcmp) |
| **split 全文按空格** | 0.44/B | find+remove_prefix 0.54 | 1.22x | ✅赢(自研,无分配子视图) |
| **trim 两端** | 13.6 | 16.85 | 1.24x | ✅赢(自研,组合 find_first/last_not_of) |
| **rfind(char) 后向扫** | 0.012/B | 0.012 | ~1.0x | ➖平(自研 AVX2 后向扫;SSE2-only 时 1.7x;原 scalar 输 ~20x) |
| **count(char) 全扫** | 0.008/B | std::count 0.008 | ~1.0x | ➖平(自研 AVX2 4累加器 psadbw;SSE2-only 时 3x;原 scalar 输 ~27x) |
| **find(needle) 6B 子串** | 0.08/B | 0.08 | ~1.0x | ➖平(转发 std;自研 memchr 候选曾输 2.5x) |
| **rfind(needle) 6B** | 0.12/B | 0.12 | ~1.0x | ➖平(转发 std;自研 region_eq 曾输 4x) |
| **find_first_of(4 字符集)** | 0.05/B | 0.05 | ~1.0x | ➖平(转发 std;自研 memchr-min 曾 0.62x、PCMPESTRI 0.76x) |
| **find_first_not_of 全字母** | 0.06/B | 0.06 | ~1.0x | ➖平(转发 std;自研 bool 表曾输 4.5x) |
| **find_last_of(digits)** | 0.06/B | 0.06 | ~1.0x | ➖平(转发 std;自研曾输 4.4x) |
| **StringBuf intern**(100k/2k 唯一) | 24.0 | map+mutex(同契约 thread-safe peer)28.1 | 1.17x | ✅赢同契约 peer(SwissTable 核) |
| **StringBuf add/retain/remove churn** | 17.9 | thread-safe map+mutex 22.3 / 无锁 floor 7.5 | 1.24x / 0.41x | ✅赢同契约 thread-safe peer 1.24x(min 1.17 保底,25 遍);对无锁裸 map 是 apples-to-oranges(见结论) |

**诚实结论(全线追平的路径)**:
- **自研追平的项(HYLE 自己的 SIMD)**:`find(char)`/`starts_with` 早已切 memchr/memcmp;`rfind(char)`/`count` 上运行期 ISA dispatch(SSE2 baseline + AVX2 变体)。**CPUID 探测下沉 platform**(`platform/cpu.{h,cc}`)后 container 首次能安全上 AVX2:`string_ref_avx2.cc` 是独立 `-mavx2` TU,只在 `cpu_features().avx2 && size>=32` 时进入(老 CPU 走 SSE2/scalar 无 SIGILL),count 用 4 独立累加器打破 psadbw 依赖链。`split`/`trim` 靠零分配子视图**反超** std。
- **转发 std 的项(借 SOTA,不硬造)**:`find(needle)`/`rfind(needle)`/`find_first_of`/`find_first_not_of`/`find_last_of`/`find_last_not_of`——MSVC STL 的 two-way 向量化子串搜索 + PCMPESTRI 字符集扫描是实测 SOTA,**自研 PCMPESTRI 尝试只到 0.76-0.78x 追不动**(PCMPESTRI 延迟高),遂转发 `std::string_view`。转发局限在 `string_ref.cc`(公开头不引 `<string_view>`),不破 C-ABI。**代价**:string_ref 在这些项上不是"纯自研替代",而是"借 std 的最优实现"——这是刻意取舍(不为纯粹性造打不过微软手写汇编的轮子),非缺陷。**结果:5 项从输 2.5–4.5x 全部统计平局(ratio 0.98–1.02,gap«spread)。**
- **前身 clay 的做法参照**:clay 的 `string_ref` 搜索族**全部**转发 `std::string_view`(且其头 `#include <iostream>`/`<string>`/`<functional>`)。HYLE 只转发 std 真正更强的几项、且只从 .cc 转发、公开头零 std string 头——比 clay 更克制。
- **StringBuf churn 换上同契约对手后 ✅赢 1.24x(红队待办已结)**:此前只有**无锁裸 map**(7.5ns)作对手,而 StringBuf 是 thread-safe(每 op 一把锁 + handle 是稳定 hash 非 iterator)——apples-to-oranges。2026-07 补上**同契约 thread-safe peer**(unordered_map + mutex,且每 op 按 key re-find,匹配 StringBuf 的 hash-handle 契约,不让对手靠复用 iterator + 免锁占便宜):HYLE 17.9ns vs peer 22.3ns = **赢 1.24x**(25 遍统计,ratio min 1.17 保底,从不 <1;判定卡在 TIE 仅因 unordered_map 的 add/erase churn rehash 抖动使 peer spread 65%,而 HYLE SwissTable 平坦槽+mimalloc free 稳定在 8%——又快又可预测)。对无锁 floor 的 0.41x 是选错对手,非真输。

## container·path —— 纯词法路径操作（`path_bench.cc`）

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4,各 15 遍 `bench-stat.py` 中位数+spread**。`path.h` 是 HYLE 的纯字符串路径层(无 syscall),`extension`/`base`/`directory`/`normalize`/`split` 在 string_ref 视图上原地走分量、**零分配**。**三方对标(2026-07 补公平性)**:① `std::filesystem::path` 词法操作——drop-in std SOTA,但每次操作**构造一个 path 对象(堆分配 + 存字符串副本)**;② **`std::string_view` 手写反向扫**——同零分配视图 + 同 back-to-front 扫描形状、仅 API 不同,是**纯算法基线**(隔离"算法" vs "结构");③ HYLE。**HYLE≈② 证扫描算法已最优;②③一起碾 ① 是"零分配 vs 必分配"的结构差,非更巧的算法。**

| op | HYLE(Win/Lin) | std::string_view 手写(算法基线,Win/Lin) | std::filesystem(Win/Lin) | vs 算法基线 | vs filesystem | 结论 |
|---|---|---|---|---|---|---|
| **extension** | **2.26/1.77** | 2.13/1.60 | 152/129 | 双平台**平**(Win 1.01/Lin 0.90,gap«spread) | 66/73x | ✅平算法基线(证扫描最优)+ 碾 fs(结构差) |
| **base** | **4.59/4.25** | 8.41/4.07 | 169/123 | **Lin 平**(0.96)/Win 数字 1.84x=MSVC 侧 string_view 构造税,非算法优势 | 37/29x | ✅平算法基线(Lin 为准)+ 碾 fs |
| **directory** | **10.08/4.66** | 9.94/4.71 | 172/249 | 双平台**平**(Win 0.99/Lin 1.01) | 17/53x | ✅平算法基线 + 碾 fs |
| **normalize 词法折叠** | 69 ns(Win) | (无零分配对手) | 401 | — | **5.8x** | ✅赢 fs(就地折叠 vs 堆 path;PRODUCE 型无 string_view 对手) |
| **join_array 6 部件** | 29 ns(Win) | (无零分配对手) | 618 | — | **21x** | ✅赢 fs(buffer-fill vs N 次 path 分配) |

**诚实结论(2026-07 re-audit:补算法基线 + extension 真优化 + 反哺检查,双平台)**:

1. **对 `std::filesystem` 的 17–73x 领先是结构差,不是巧妙算法**——补上 `std::string_view` 手写反向扫这个**零分配算法基线**后诚实分离:extension/base/directory **双平台均与算法基线统计平局**,证明 HYLE 的扫描算法本身 = 纯零分配标量扫的最优形状;对 fs 的巨大倍数纯粹来自"HYLE 在视图上原地扫、零分配" vs "fs 每次构造 path 对象(堆分配 + 拷贝字符串)"。**⚠️单平台数字陷阱(方法学)**:Win 上 base 看似"反超算法基线 1.84x"(4.59 vs 8.41),但 **Linux 平手(4.25 vs 4.07)**——那 1.84x 是 **MSVC 对 `std::string_view` 构造/返回的额外开销**,不是 HYLE 算法优势;双平台一跑就现原形(同 dtype/time 那几次"单平台数当无条件结论"的教训)。诚实表述:**扫描算法与零分配基线平手,不反超**。normalize/join_array 是 PRODUCE 型(产出新路径),std 无零分配对手,只两方对标 fs。

2. **extension 真优化:正向全扫 → 反向扫遇分隔符停,40.4→2.26ns(~18x)**——re-audit 揪出的真算法低效:原 `extension()` **正向全扫整条路径**(深路径 54 字符每次全扫),而 base/directory 早已**反向扫遇分隔符即停**(只碰最后一个 component ~12 字符)。这精确解释了旧账本"extension 仅 3.4x、远低于 base/directory 15x+"的异常。改成同样的反向扫(latch 最后一个点,遇第一个非点字符即返回——正是 Python splitext 语义),extension 从 40.4ns 追平到 2.26ns,与算法基线 string_view 手写版统计平局。新增 6 个反向扫边界用例(`..foo`/`...`/`a..b`/`x.`/深路径/点在分隔符前)守卫语义,18/18 过。

3. **反哺检查(全表 error→path,无可反哺项)**——① **string_ref 的 SIMD 反向扫核**(`str_rfind_char` SSE2、`str_find_last_of` PCMPISTRI)对 path **不适用**:path 组件短(basename ~12-20 字符,远低于 SIMD 收益阈值,账本"string_ref SIMD 仅 multi-KB 扫才划算"),且这些核是**导出符号**(有 ~1ns call seam)、会破坏 extension/base/directory 的 `constexpr`。实测 base/directory 平手 string_view **标量**反向扫已印证——短组件标量扫即最优,无 SIMD 空间。② **string_algo/hash/mem/simd** 与 path 零分配视图扫**正交**(path 不分配、不 hash、不做数值核)。结论:path 的赢点(零分配结构 + 最优标量扫)自成一体,总表其他模块的优化无一能反哺。

## 次要模块（util/random、time/clock、obs/log、SwissTable iterate，`misc_bench.cc` + `swiss_table_bench.cc`）

> 环境:Win MSVC19.50 + **Linux GCC14.2(time/clock + obs/log 双平台 15 遍统计,`bench-stat.py`)**,nanobench。此前零 bench 的性能相关原语。标尺 = std 等价物(同 OS 原语/同算法类),或 HYLE 自有(无 std 对等物时诚实标注)。**time 项平台差异大,见 monotonic_ns 结论③。**(util/Rng、SwissTable iterate 仍 Win 单平台,待补 Linux;log 已双平台。)

| 场景 | HYLE | 标尺 | vs 标尺 | 结论 |
|---|---|---|---|---|
| **Rng next_u64**(xoshiro256++) | **1.56 ns** | std::mt19937_64 2.41 | **1.54x** | ✅ 赢(同质量,256-bit state) |
| **Rng range_u64**(Lemire 无偏) | **1.75 ns** | std::uniform_int_distribution 2.75 | **1.57x** | ✅ 赢且**无偏**(std 也无偏,HYLE 更快) |
| **SwissTable iterate**(for_each) | **2.48 ns** | std::unordered_map 8.73 | **3.5x** | ✅ 赢(扁平 ctrl 扫 vs 节点指针链) |
| **realtime_ns** | 15.0(Win)/12.5(Lin) | system_clock 14.4/12.8 | **0.96x/1.02x** | ➖ 平双平台(同 OS 原语) |
| **monotonic_ns** | 12.4(Win)/12.6(Lin) | steady_clock 12.7/12.9 | **Win 1.03x/Lin 1.0x** | ➖平双平台(Win 曾输 1.48x,补频率特判后追平;见结论③) |
| **fast_monotonic_ns**(opt-in rdtsc) | 7.4(Win)/6.8(Lin) | HYLE monotonic_ns 12.4/12.6 | **1.67x/2.0x** | ✅快 1.7-2.0x(rdtsc 跳过 OS 调用壳;非默认,可能跳变;见结论③) |
| **log gated-out** | 1.41(Lin)/2.03(Win) | (无 std 对照) | — | HYLE 自有(1 relaxed 原子载 + 分支,跳过格式化;已最优) |
| **log emit**(format+fanout,无 I/O) | 54.7(Lin)/78–85(Win) | (无 std 对照) | — | HYLE 自有;**~90% 是 std::format(裸 52ns),锁+fanout 仅 Lin 6/Win 13ns** |

**诚实结论**:① **Rng 赢 std PRNG 1.54x**(xoshiro256++ vs mt19937_64,同统计质量,HYLE header-inline 无 DLL hop);**range_u64 改真 Lemire 无除法映射(2026-07 红队 util)**——原实现注释称 "Lemire's nearly-divisionless" 但实为经典 modulo-rejection(每次调用恒两次 64 位 idiv:`(MAX-n+1)%n` + `r%n`),名不副实。真 Lemire(absl/numpy/rust rand)用一次 128 位乘 `hi(x*n)` + 极罕见除法:常见路径拿 `n` 当 threshold 高估(`lo(x*n)<n` 检查,零除法),仅罕见时才算精确 `2^64%n`。改用 `detail::mul128_hi_lo`(MSVC `_umul128`/GCC `__uint128_t`),热路径 1 mul 替 2 idiv,无偏性不变(chi-square + 边界测试过)。**实测 shuffle 1000 ints 3.82→1.18ns = 3.2x(双平台一致)**——这是运行时变界的真实路径,idiv 真花。**教训(重要)**:`range_u64(0,999)` 直接微基准反而几乎没变(1.55→1.46ns)——**常量界 999 让编译器把旧 `%n` 折成魔数乘法,微基准从不暴露 idiv 成本**;只有 shuffle 那种运行时变界(`range_u64(0,i-1)`)才暴露真 idiv。同 monotonic_ns 那次"常量除数被折叠"的对偶——**测有除法的原语必须用运行时变量的除数,否则编译期折叠掩盖真成本**。Win 上 shuffle 赢 std::shuffle 5.5x(6.42→1.17)、Linux 平(GCC 的 std::shuffle 比 MSVC 强)。② **SwissTable iterate 赢 std::unordered_map 3.5x**——扁平 ctrl 字节扫 + 连续 slot(cache 友好)碾压节点指针链;`for_each` 是函数模板(内联、零分配)。③ **monotonic_ns 双平台均统计平局**:**双平台 15 遍统计**:Linux/GCC **统计平局**(HYLE 12.6 vs steady_clock 12.9,ratio 1.015,gap 1.5%«spread 11%)——POSIX 实现 `clock_gettime(CLOCK_MONOTONIC)`+一个乘加,与 glibc steady_clock 几乎同码。**Win MSVC 曾输 1.48x(18.7 vs 12.8),已修=追平 12.4 vs 12.7(ratio 1.03,统计平局)**。**根因(dumpbin 实锤,推翻旧"非可修接缝"结论)**:Win QPC 标度原来无条件做**两次 64 位除法**(`t/f`、`t%f`,f 是 `QueryPerformanceFrequency` 的运行时返回值,x86 `idiv` ~20-40 cyc);而 MSVC `steady_clock::now()`(`__msvc_chrono.hpp:656`)对**常见 QPC 频率特判**——10MHz(x86/x64 普遍)直接 `t*100`、24MHz(ARM64)走魔数乘法,只有罕见频率才除。除数变编译期常量后编译器把 idiv 折叠成魔数乘法。**修=照搬 MSVC 特判**(`clock.cc` Win 分支:`f==10MHz → t*100`;`f==24MHz → split`;其余保留运行时除法回退)。本机 QPC=10MHz,修后 objdump 确认快路径 `imul rax, rbx, 64h` 零除法,与 std 同形。旧结论曾误判"std 做完全相同的工作、算法无劣构、要更快得换 rdtsc"——**错在没 objdump 就信源码表面**:MSVC 的特判正是关键差异,纯算法对齐即可追平,无需 rdtsc。Linux POSIX 路径未动(本就平局)。**realtime_ns 双平台都平**(Win 0.96x/Lin 1.02x,std 也做 epoch 换算)。**另提供 opt-in `fast_monotonic_ns()`(rdtsc 直读)——双平台实测快 1.7-2.0x**(Win 7.4 vs 12.4=1.67x,DLL hop;Linux 6.8 vs 12.6=2.0x,静态无 hop;均 bench-stat 15 遍 gap»spread 真赢):QPC/clock_gettime 的调用壳本身吃 ~6ns,rdtsc + Q32 定点换算(`(tsc*mult)>>32`,无除法无 syscall)砍掉一半。校准参考 abseil sysinfo(配对读最低延迟样本抗抢占 + 倍增窗口到 1% 收敛),invariant-TSC 经 CPUID 探测(tracy 法),不支持/非 x86 透明回退 monotonic_ns。**不设默认**:可能在 sleep 唤醒/VM 迁移/跨核迁移时跳变(无 max 钳制省全局原子),不可用于 deadline——这正是 abseil 默认关其 cycle-clock 的理由;HYLE 把它作 profiling/高频采样专用工具,诚实标价(公开头 + usage-guide 选型表)。**校准优先 CPUID leaf 0x15(2026-07)**:leaf 0x15 枚举 TSC/晶振比 + 晶振 Hz,可零测量算出精确 TSC Hz(本机裸 Windows=2.9952 GHz 精确)。**注意收益边界:leaf 0x15 只在 `tsc_scale()` 首次校准时用一次(Meyers 单例),不进热路径**——`fast_monotonic_ns()` 的读路径恒为 `rdtsc + (tsc*mult)>>32`,mult 无论来自 leaf 0x15 还是计时法都一样,故 **7.4/6.8ns 读延迟不变**。它优化的是两件非调用延迟的事:① **启动校准成本**——省掉计时法的忙等(1ms+2ms+... 最坏几十 ms)→ 一次 CPUID;② **乘子精度**——1% 近似→精确,减小长时间累计漂移。回退计时法覆盖三类:hypervisor 屏蔽(WSL/Hyper-V 返回全 0)、pre-Skylake Intel(无此叶)、**AMD 全系(leaf 0x15 是 Intel-only,AMD 永远走计时法,非漏优化)**。crystal=0 时按 SDM 从 leaf 0x16 base MHz 推。⑤ **opt-in `fast_realtime_ns()`(coarse 时间源,realtime 的对称补,2026-07)**——实测 **Win 7.6x(1.8 vs 13.5ns)/Linux 4.2x(3.1 vs 13.0ns)**:Win 读 `GetSystemTimeAsFileTime`(KUSER_SHARED_DATA,无 syscall)、Linux `CLOCK_REALTIME_COARSE`(vDSO),跳过 `realtime_ns` 的高精度路径。**代价=分辨率降到 OS tick ~1-15ms(实测 ~4ms),同 tick 内多次调用返回同值**;值仍是真 epoch ns、免校准。用于日志/trace 时间戳(ms 分辨率足够、读成本要低),需真 sub-ms 墙钟才用 `realtime_ns`。同 jump-on-adjust caveat(不可用于 elapsed/deadline)。④ **log produce 路径(2026-07 红队 obs 拆解 + 双平台)**:gated-out 仅 1 relaxed 原子载 + 分支(Lin 1.41/Win 2.03ns,跳过 std::format,已最优);emit(format+fanout 无 I/O)Lin 54.7/Win 78–85ns。**成本拆解(微基准实锤,决定不优化锁)**:裸 `std::format_to_n("iter={} hex={:x}",x,x)` 单独就 **~52ns**——占 emit **~90%(Lin)**;`log_message` 的 mutex 锁 + sink 快照 memcpy + fanout **仅 Lin 6/Win 13ns**。**结论:produce 路径已贴 std::format 下限,锁不是瓶颈**。**SOTA 对标(spdlog 未 clone,量级参考)**:spdlog 同步模式禁 I/O 的 format+dispatch 用 fmtlib(与 std::format 同量级数十 ns),HYLE 在"同步 produce"这层无结构劣势——两者大头都是格式化本身;HYLE 明确不做 async 聚合(那是 ops 事,见 hyle-scope),故不与 spdlog async 比(不同架构)。**COW/RCU 优化(评估后未做)**:把每 emit 的 mutex 快照换成 `atomic<SinkArray*>` 写时复制(读路径单 acquire load,零锁零 memcpy,实测读 1.4ns « mutex 11.7/2.5ns)可省 Win ~12/Lin ~5ns,但相对整条 emit 仅 8–16%、且 sink I/O 才是真实部署大头——**非性能瓶颈,不为投机收益增结构复杂度**(注:log.cc:22-30 否掉的是 seqlock 读同一非原子数组=UB;COW 读写不同内存是标准 RCU、应 TSan-clean,注释"≈mutex 成本"只适用于 atomic<shared_ptr> 快照的引用计数 RMW,不适用裸指针 RCU 的纯 load。若将来日志 produce 成实测热点,COW 是干净正收益且顺带修 log.h:48-53 的 sink 移除 UAF 窗口——已登记备查,非即做)。
