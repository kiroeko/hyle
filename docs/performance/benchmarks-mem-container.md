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

> 环境:Win MSVC19.50 Release,nanobench,~4KB 文本(L1 常驻),历史 15 遍统计(`bench-stat.py`)；2026-07-19 container 收口又跑一遍完整当前二进制。**标尺 = `std::string_view`**(string_ref 是其 drop-in 替代,conventions §7:同 16 字节 POD + null_terminated 位,同契约对照公平)+ `std::count` / simdutf。旧标题“container·str 无真输、全线追平或反超”已经撤回：当前 4KB 点值里 split/rfind/count 有 **0.83–0.96x**，而 64KB/1MB 回到约 1.0–1.14x；短输入的 DLL/dispatcher seam 是真实产品边界，方差只能解释一部分。分两条腿走(2026-07 定策):HYLE **自研** find(char)/rfind(char)/count 和 ASCII lower/upper 的 scalar/SSE4/AVX2/AVX512/NEON 运行期 kernel；对 MSVC STL 手写向量化汇编实测更强的 substring/成员族，**转发 `std::string_view`**。公开头不引 std string 头。

### 2026-07-19 收口复测（当前接口形状）

比值大于 1 表示 HYLE 更快。三平台均为 Release、nanobench 3 repetitions
aggregate；macOS 未装 Abseil 的行使用同分片 std peer。

| 场景 | Windows | WSL GCC 14 | Apple M4 | 判定 |
|---|---:|---:|---:|---|
| 单次 intern / thread-safe std peer | **0.994x** | **0.772x** | **0.934x** | Win 平；WSL/M4 明确输 |
| `intern_many` / unlocked std floor | **0.990x** | **0.732x** | **1.045x** | Win 平、M4 小赢；WSL 输 |
| 1KiB unique growth / flat-set floor | **5.36x Absl** | **1.71x Absl** | **1.67x std** | cached hash 的复杂度优势三平台兑现 |
| 8-thread adaptive shards / strongest peer | **0.948x Absl** | **0.958x Absl** | **0.956x std** | 距强 peer 约 4–5%，不是无条件赢 |
| add/retain/remove / direct-ID+mutex peer | **0.990x** | **0.813x** | **0.785x** | Windows 平；Linux/M4 generation+slab 常数可见 |
| 16-cycle churn / retained peer | **0.998x** | **0.861x** | **0.808x** | 高水位复用成立，但非三平台常数 SOTA |
| find/substring/set-search/starts_with，4KB | 0.93–1.00x | ≈1.0x | ≈1.0x | 统计近似平；不宣传赢 |
| split/rfind/count，4KB | 0.83–0.96x | ≈1.0x | ≈1.0x | 短输入 DLL/dispatcher seam 可见 |
| count/rfind，64KB–1MB | 0.995–1.14x | ≈1.0x | ≈1.0x | 长输入摊平 |
| UTF-8 validate+count / simdutf | **1.004x** | — | — | 平 |
| ASCII lower / scalar branch | **1.52x** | — | — | 赢 scalar reference |

这轮不是只换报告口径：StringBuf 改成 16/32/64 自适应 shard、cached resident
hash、generation+shard+slot ID、短 entry slab、永久 intern 的 lazy ID、四类 batch
API；hash 用中间位选 shard，保留 Swiss H1/H2 熵且省掉第二次 avalanche。
`intern/add` 为预期 O(|input|)，growth/erase 不重扫 resident string；handle
lookup/retain/remove 为预期 O(1)，`size()` 只扫硬上限 64 shard。旧表保留为历史
优化证据，不再作为当前无条件证书；Linux/M4 lifecycle 的 14–22% 落后是已知冻结
边界，只有同契约真实 workload 再次证明它是瓶颈才重开。

| op | HYLE | std::string_view | vs SOTA | 结论 |
|---|---|---|---|---|
| **find(char) 全扫 miss** | 0.02/B | memchr 0.01 | ~1.0x | ➖平(自研 memchr;原 scalar 输 ~20x) |
| **starts_with(16B)** | 1.79 | memcmp 1.77 | ~1.0x | ➖平(自研 memcmp) |
| **split 全文按空格** | 0.44/B | find+remove_prefix 0.54 | 1.22x | ✅赢(自研,无分配子视图) |
| **trim 两端** | 13.6 | 16.85 | 1.24x | ✅赢(自研,组合 find_first/last_not_of) |
| **rfind(char) 后向扫** | 0.012/B | 0.012 | ~1.0x | ➖平 std(自研 vec.h 后向扫 FindLastTrue;**arm NEON 8x 赢原 scalar**;原 scalar 输 ~20x) |
| **count(char) 全扫** | 0.008/B | std::count 0.008 | ~1.0x | ➖平 std(自研 vec.h 4累加器 SumsOf8;**arm NEON 3.7x 赢原 scalar**;原 scalar 输 ~27x) |
| **find(needle) 6B 子串** | 0.08/B | 0.08 | ~1.0x | ➖平(转发 std;自研 memchr 候选曾输 2.5x) |
| **rfind(needle) 6B** | 0.12/B | 0.12 | ~1.0x | ➖平(转发 std;自研 region_eq 曾输 4x) |
| **find_first_of(4 字符集)** | 0.05/B | 0.05 | ~1.0x | ➖平(转发 std;自研 memchr-min 曾 0.62x、PCMPESTRI 0.76x) |
| **find_first_not_of 全字母** | 0.06/B | 0.06 | ~1.0x | ➖平(转发 std;自研 bool 表曾输 4.5x) |
| **find_last_of(digits)** | 0.06/B | 0.06 | ~1.0x | ➖平(转发 std;自研曾输 4.4x) |
| **StringBuf intern**(100k/2k 唯一) | 28.86(Win) | map+mutex 同契约 peer 28.70 | 0.994x | ➖Win 平；WSL 0.772x、M4 0.934x，见上方当前三平台表 |
| **StringBuf add/retain/remove churn** | 18.62(Win) | direct-ID+mutex 同契约 peer 18.44 | 0.990x | ➖Win 平；WSL 0.813x、M4 0.785x，旧 1.24x 证书撤回 |

**历史优化路径（当前结论以上述收口复测为准）**:
- **自研追平的项(HYLE 自己的 SIMD)**:`find(char)`/`starts_with` 早已切 memchr/memcmp;`rfind(char)`/`count` 走运行期 ISA dispatch。**2026-07 统一到 `simd/vec.h`**:count/rfind 的核从手写裸 intrinsic(`string_ref_avx2.cc` 256b + inline SSE2,**x86-only**)改写为 `str_search_{scalar,sse4,avx2,avx512,neon}.cc`(共享体 `str_search_impl.h`,写在 D-tag 抽象上),`string_ref.cc` 按 `best_isa()` 经缓存 fn 指针选最宽核——与 `simd/ops` 同变体-TU 派发模型。收益:**一份源码覆盖全 ISA,arm64 NEON + AVX-512 白得**(此前 arm 落 scalar)。为保住手写版的 psadbw 4 累加器计数技巧,给 vec.h 补了三原语 `SumsOf8`(SAD)/`VecFromMask`/`FindLastTrue`。**回归实测(WSL,apples-to-apples 两边都过 .so)**:新 vec.h 核 vs 旧手写 AVX2 核 **打平**(count 96-101%、rfind 99-100%,gap«spread;早先看到的 3-11% 是"手写核可内联进 bench TU、.so 调用不可内联"的测量不对称,非真回退——`__attribute__((noinline))` 对同一份核复现出同样 gap 坐实)。**arm 真机(Apple Silicon,vs 重构前 arm 落的 scalar 字节循环)**:count **3.7x**、rfind **8.0x**(4KB/64KB/1MB 全稳)。`split`/`trim` 靠零分配子视图**反超** std。
- **转发 std 的项(借 SOTA,不硬造)**:`find(needle)`/`rfind(needle)`/`find_first_of`/`find_first_not_of`/`find_last_of`/`find_last_not_of`——MSVC STL 的 two-way 向量化子串搜索 + PCMPESTRI 字符集扫描是实测 SOTA,**自研 PCMPESTRI 尝试只到 0.76-0.78x 追不动**(PCMPESTRI 延迟高),遂转发 `std::string_view`。转发局限在 `string_ref.cc`(公开头不引 `<string_view>`),不破 C-ABI。**代价**:string_ref 在这些项上不是"纯自研替代",而是"借 std 的最优实现"——这是刻意取舍(不为纯粹性造打不过微软手写汇编的轮子),非缺陷。**结果:5 项从输 2.5–4.5x 全部统计平局(ratio 0.98–1.02,gap«spread)。**
- **前身 clay 的做法参照**:clay 的 `string_ref` 搜索族**全部**转发 `std::string_view`(且其头 `#include <iostream>`/`<string>`/`<functional>`)。HYLE 只转发 std 真正更强的几项、且只从 .cc 转发、公开头零 std string 头——比 clay 更克制。
- **StringBuf 的旧 1.24x churn 证书已撤回**：旧 peer 每次按 key re-find，
  仍弱于当前 generation-tagged direct-ID+mutex 对手。新形状下 Windows 0.990x
  统计平，WSL 0.813x、M4 0.785x 明确输；另一方面 cached hash 的 1KiB growth
  三平台赢 strongest available floor（Win/WSL 对 Absl 5.36/1.71x，M4 对 std
  1.67x），8-thread 距 Absl 约 4–5%。因此产品价值落在复杂度、稳定 identity、
  并发接近强 peer 和 batch，而不是虚构“所有 lifecycle 常数都赢”。

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

> 环境:Win MSVC19.50 + Linux GCC14.2 + Apple M4 AppleClang（2026-07-19
> util/time/obs 收口复测）。nanobench。标尺 = std 等价物(同 OS 原语/同算法类),
> 或 HYLE 自有(无 std 对等物时诚实标注)。三平台数字按 Win/Linux/M4 排列；
> 高争用行只判断适用边界，不把高方差点值宣传成 SOTA。

| 场景 | HYLE | 标尺 | vs 标尺 | 结论 |
|---|---|---|---|---|
| **Rng next_u64**(xoshiro256++) | 1.59/1.38/2.26 ns | std::mt19937_64 2.32/1.44/3.47* | 1.46x/平/1.53x | Win/M4 赢，Linux 平；M4 std 行方差高 |
| **Rng runtime-varying range_u64** | 2.09/1.47/2.17 | std distribution 2.86/2.08/7.68 | 1.37/1.41/3.54x | ✅ 三平台赢且无偏 |
| **Rng C batch fill** | 4.99/0.67/0.65 | native C++ 5.16/0.68/0.65 | 全部平 | 类型擦除 ABI 被 batch 摊薄 |
| **SpinLock 无争用** | 3.71/4.79/1.01 | mutex 10.94/2.48/4.19 | 2.95x/0.52x/4.15x | 平台翻转，只限短低争用 |
| **SpinLock 8 threads** | 50.17/60.44/271.88 | mutex 50.59/39.08/20.98 | 平/0.65x/0.08x | ❌ 高争用选 mutex |
| **Ref fan-out** | 10.13/8.33/3.19 | shared_ptr 8.09/11.12/3.22 | 0.80x/1.34x/平 | 平台翻转；所有权形状而非普遍提速 |
| **SwissTable iterate**(for_each) | **2.48 ns** | std::unordered_map 8.73 | **3.5x** | ✅ 赢(扁平 ctrl 扫 vs 节点指针链) |
| **realtime_ns** | 15.0(Win)/12.5(Lin) | system_clock 14.4/12.8 | **0.96x/1.02x** | ➖ 平双平台(同 OS 原语) |
| **monotonic_ns** | 12.4(Win)/12.6(Lin) | steady_clock 12.7/12.9 | **Win 1.03x/Lin 1.0x** | ➖平双平台(Win 曾输 1.48x,补频率特判后追平;见结论③) |
| **fast_monotonic_ns**(opt-in rdtsc) | 7.4(Win)/6.8(Lin) | HYLE monotonic_ns 12.4/12.6 | **1.67x/2.0x** | ✅快 1.7-2.0x(rdtsc 跳过 OS 调用壳;非默认,可能跳变;见结论③) |
| **log gated-out** | 2.03–2.11(Win)/1.05–1.09(WSL)/0.91–0.92(M4) | (无 std 对照) | — | relaxed level gate + branch；参数不求值、不格式化 |
| **log preformatted C++**(1 sink,无 I/O) | 22.48/13.07/8.85 ns | (无 std 对照) | — | mutex snapshot + per-slot active refs + callback；换取 quiescent unregister |
| **log preformatted C ABI**(1 C sink,无 I/O) | 23.93/16.03/10.68 ns | native C++ 上行 | +1.45/+2.96/+1.83 ns | POD bridge + opaque trampoline 的真实 ABI 成本很小 |
| **log formatted emit**(1 sink,无 I/O) | 81.9–84.6/69.5–71.4/50.9–53.5 ns | (无 std 对照) | — | `std::format_to_n` + safe fan-out；真实 I/O sink 通常远大于此 |

**诚实结论**:① Rng 的 runtime-varying range 三平台赢；`next_u64` 在
MSVC/AppleClang 赢而 GCC 平，不能再写成无条件 1.54x。真 Lemire 用一次 128 位乘
和罕见 threshold 修正；基准必须让 bound 在运行时变化，避免编译器把常量除数折成
魔数。C ABI 单值调用有 0.09–0.95ns seam，但 fill batch 三平台与 C++ 平。
② SpinLock 的结果按平台甚至反向，高争用时 Linux/M4 均输 mutex；它是低争用
短临界区工具，不是通用锁。Ref 也随工具链翻转，其价值是侵入式共享所有权和省控制
块，而非原子 RMW 必然更快。完整 util 红队、ABI 和 sanitizer 证据见
[`../milestones/util-module-closure.md`](../milestones/util-module-closure.md)。
③ **SwissTable iterate 赢 std::unordered_map 3.5x**——扁平 ctrl 字节扫 +
连续 slot(cache 友好)碾压节点指针链；`for_each` 是函数模板(内联、零分配)。
④ **time 时钟收口（2026-07-19，三平台）**：`monotonic_ns`/`realtime_ns`
与相同转换工作的标准钟同档；x86 opt-in fast clock 获益，M4 正常回退。完整审计见
time-module-closure。⑤ **obs 收口后三层性能**：宏 gate 0.91–2.11ns；
preformatted native 8.85–22.48ns；C ABI trampoline 再加 1.45–2.96ns。
完整红队见 [`../milestones/obs-module-closure.md`](../milestones/obs-module-closure.md)。

## 模糊路径匹配（container-06c，`fuzzy_bench.cc`）

> 环境:Linux GCC14.2 Release,nanobench。净新设计,抄 fzf `src/algo/algo.go` V1 两趟算法(forward 找子序列结束位 → backward 收紧起点 → 收紧窗口评分)。**两件事要证明**(非单点 ratio):时间复杂度 + bonus 开销。

| 场景 | ns/candidate | candidate/s | 结论 |
|---|---|---|---|
| **find_path_matches N=1,000** | 45.8 | 21.9M | ✅ 基准点 |
| **find_path_matches N=10,000** | 47.9 | 20.9M | ✅ **O(N):跨 10× 持平** |
| **find_path_matches N=100,000** | 47.8 | 20.9M | ✅ **O(N):跨 100× 仅 +4%(cache)** |
| **path_match_score(完整评分)** | 44.8 | 22.3M | vs floor 2.35x |
| **std 子序列 floor(无评分)** | 19.0 | 52.6M | 算法下限(纯 membership 扫) |

**诚实结论**:① **O(N) 时间复杂度实证**——ns/candidate 在 N=1k/10k/100k = 45.8/47.9/47.8,跨 **100× 规模仅 +4%**(cache 效应,非算法项),线性 scaling 铁证(两趟均单遍扫,无隐藏 rescore)。这正是选贪心两趟而非 fzf Smith-Waterman DP 的理由:DP 每候选 O(pattern×len) + 分配 H/C 表,对"10万路径每键全量重评分"不可接受。② **bonus 成本 2.35x vs 子序列 floor(44.8 vs 19.0ns)——诚实标价**:floor 只做"是否有序子序列"(每个 fuzzy 匹配器的不可约最小工作),HYLE 多付的是 fzf 级排序质量(charClass 分类 + boundary/camel/consecutive bonus + backward 收紧 pass)。这 2.35x 买的是"能排序"而非"仅能过滤"——quick-open UI 的刚需。③ **绝对吞吐 2200 万候选/秒**:10万路径全量重评分 ~4.5ms,交互式按键延迟无感。**find_path_matches 零堆分配**(定长最小堆,scores 存调用方缓冲,O(N log K))——契约优先,K=20 时 100k 候选约 4.3N 次比较,远优于全排序 O(N log N)。
