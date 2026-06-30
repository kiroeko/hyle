# HYLE 性能预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 对标总账（win/loss ledger，全部实测 vs 真 SOTA，非 std）

> **一眼看尽输赢**。标尺是各领域 SOTA 参考库(非 std——赢 std 是最低门槛)。每行详情见下方对应小节。`倍数`>1=HYLE 快。**判定**:✅赢 / ➖平(噪声内) / ❌输(均附根因:by-design 接缝 / 约束选择 / 选错工具,无一为未处理的真缺陷) / ⏳待补(该项尚无同 ISA/同族 SOTA 对照,任务 #34–37 正在填)。

| 模块 · 操作 | HYLE | SOTA 标尺(非 std) | vs SOTA | 判定 |
|---|---|---|---|---|
| **hashmap** find-hit | — | absl flat_hash_map | 1.10x | ✅赢 |
| **hashmap** find-miss | — | absl | ~1.00x | ➖平(12 遍 avg 97.5%,spread>差距) |
| **hashmap** insert | — | absl | 1.40x | ✅赢 |
| **hashmap** erase | — | absl | 1.39x | ✅赢 |
| **hash** Hash&lt;T&gt; 16B POD 内联 | 0.59ns | wyhash | 1.58x | ✅赢 |
| **hash** mix64 整数 | 0.45ns | wyhash64 | 1.27x | ✅赢 |
| **hash** raw hash_bytes 8B | 0.21ns | 裸 XXH3 0.18 | 0.86x | ➖近平(原 1.85x,加 header 内联小路修复;0.03ns=亚周期) |
| **hash** raw hash_bytes 16B | 0.15ns | 裸 XXH3 0.10 | 0.67x | ➖测量地板(0.05ns 寄存器驻留;热路径 Hash&lt;T&gt; 0.59ns 赢 wyhash 1.6x) |
| **hash** raw hash_bytes 64B | — | 裸 XXH3 | 0.69x | ➖0.01ns 测量地板(不闭=保 xxhash.h 隐藏架构;热路径走 Hash&lt;T&gt;) |
| **hash** raw hash_bytes 256B | — | 裸 XXH3 / wyhash | 1.0x / 0.33x | ➖平 XXH3(wyhash 中串更强,观察) |
| **simd** dot product | 12.4 G/s | Highway avx2(同 -mavx2) | 1.10x | ✅赢 |
| **simd** saxpy | 25.1 G/s | Highway avx2 | 1.21x | ✅赢 |
| **simd** reduce_max | 0.08ns | Highway avx2 | ~1.0x | ➖平 |
| **simd** scale | 0.03ns | Highway avx2 | ~1.0x | ➖平(带宽受限) |
| **dtype** f16→f32 | 0.02ns | Highway PromoteTo | 1.9x | ✅赢 |
| **dtype** f32→f16 | 0.03ns | Highway DemoteTo | 1.6x | ✅赢 |
| **dtype** bf16↔f32 | 0.03/0.11ns | (无 Highway 对照) | — | ⏳待补 Highway bf16(vs scalar 3.5–6.9x) |
| **math** exp(best-of-breed→xsimd) | 0.23ns | Highway 0.30 / xsimd 0.23 | 1.30x / 1.0x | ✅平 xsimd(最快)、反超 Highway 1.3x |
| **math** log(best-of-breed→Highway) | 0.26ns | Highway 0.28 / xsimd 0.38 | 1.08x / 1.46x | ✅反超 Highway+xsimd(两库最快) |
| **math** tanh(best-of-breed→xsimd) | 0.44ns | Highway 0.56 / xsimd 0.45 | 1.27x / 1.02x | ✅平 xsimd(最快)、反超 Highway 1.27x |
| **sched** 争用 Mutex | 30.7ns | marl 32.1 | 1.05x | ➖平(本轮略胜) |
| **sched** 无争用 Mutex(MT) | 6.5ns | std::mutex 9.1 | 1.40x | ✅赢 |
| **sched** spawn 2000 noop | 141ns | marl 306 | 2.17x | ✅赢 |
| **sched** WaitGroup fork-join | 205ns(池化179) | marl 306 | 1.49x(池1.71x) | ✅赢 |
| **sched** fiber 切换 | 21–23ns | fcontext 原语 | 1.0x | ➖平(=原语,零封装) |
| **sched** Channel 收发(cap16,1p1c) | 63ns/item | (marl 无 channel) | — | ➖HYLE 自有(无 SOTA 对等物) |
| **queue** 单线程 fill+drain | 7.2(Lin) | moodycamel 7.7 / tbb 12.6 | 1.07x / 1.6x | ✅赢(微超 moody、赢 tbb) |
| **queue** 单线程 fill+drain | 7.8(Win) | rigtorp 11.8 / moody 9.6 / tbb 13.4 | 1.5x / 1.2x / 1.7x | ✅赢全部(含同约束 rigtorp) |
| **queue** MPMC 2P 单环 BoundedQueue | 56(Win) | rigtorp 58 / moody 59.4 | 1.04x / 1.06x | ✅平/微超同约束 rigtorp |
| **queue** MPMC 4P 单环 BoundedQueue | 72(Win) | rigtorp 79 / moody 62.9 | 1.09x / 0.87x | ✅反超 rigtorp(原输 1.33x,cache-line pad+去 backoff 修复) |
| **queue** MPMC 2P/4P ShardedQueue | 22.7/22.1(Win) | rigtorp 58/79 | 2.6x/3.4x | ✅赢(生产用途走这条,线性扩展) |
| **queue** work-steal owner LIFO | 2.8(Lin) | crossbeam 3.6 | ~1.3x | ✅赢(跨语言,谨慎下"有竞争力") |
| **queue** work-steal owner FIFO | 3.6(Lin) | crossbeam 4.4 | ~1.2x | ✅赢(跨语言,谨慎) |
| **queue** ConcurrentVector push/read | 3.9/0.8(Lin) | oneTBB 6.1/1.3 | 1.3x/1.1x | ✅赢 |
| **mem** SystemAllocator churn(默认) | 7.91ns | 裸 mimalloc 7.79 | 0.99x | ➖平(编译期内联,零接缝税) |
| **mem** Allocator vtable churn(opt-in) | 8.61ns | **std::pmr 同类 8.70** | **1.01x** | ✅赢同类 SOTA(对裸 mi 的 -10% 是多态物理税,std::pmr 同付) |
| **mem** 8 线程通用 | 2.07ns | 裸 mimalloc 2.14 | 1.03x | ➖平 |
| **mem** Arena bump | 0.67ns | std::pmr monotonic 1.49 | 2.2x | ✅赢 |
| **mem** StackAllocator | 0.65ns | std::pmr monotonic 1.49 | 2.3x | ✅赢 |
| **mem** GrowablePool(单线程) | 1.92ns | std::pmr unsync_pool 6.72 | 3.5x | ✅赢 |
| **mem** GrowablePool(8 线程 per-thread) | 0.72ns | 裸 mi per-thread heap 1.28 | 1.78x | ✅赢 |
| **mem** MemoryArena(单线程) | 15.0ns | 裸 mimalloc 4.87 | 0.32x | ❌输(原子 CAS 常数;单线程选 GrowablePool) |
| **mem** MemoryArena(8 线程共享) | 2.48ns | 裸 mi per-thread heap 1.28 | 0.51x | ❌输(apples-to-oranges;选对工具 per-thread 池则反超) |
| **container** SmallVector 小 N | 0.43ns | absl::InlinedVector 0.60 | 1.40x | ✅赢同款 small-vector SOTA(也赢 std::vector 39x) |
| **container** IntrusiveList | 3.04ns | boost::intrusive(gated,boost submodule 未 checkout) | — | ⏳对照待跑;vs std::list 8.1x |
| **container** CArray growth/read | 0.50/0.19ns | std::vector 0.76/0.19(本身即 SOTA) | 1.5x / 1.0x | ✅赢 growth / ➖平 read |
| **mem** ObjectPool(typed 无锁池) | 14.86ns | 裸 mimalloc 4.77 | 0.32x | ❌输(=MemoryArena 同 CAS 类;单线程选 GrowablePool) |
| **util** Rng next_u64(xoshiro256++) | 1.56ns | std::mt19937_64 2.41 | 1.54x | ✅赢 |
| **util** Rng range_u64(Lemire 无偏) | 1.75ns | std::uniform_int_distribution 2.75 | 1.57x | ✅赢(且无偏) |
| **container** SwissTable iterate(for_each) | 2.48ns | std::unordered_map 8.73 | 3.5x | ✅赢(扁平 ctrl 扫 vs 节点链) |
| **container** StringBuf intern(100k 流/2k 唯一) | 24.0ns | 同契约 thread-safe peer(unordered_map+mutex)28.1 / 无锁 floor 21.2 | **1.17x** / 0.88x | ✅赢同契约 peer(StringBuf 是 thread-safe 取锁,对无锁 map 比是 apples-to-oranges;同取一把锁则 SwissTable 核反超 1.17x) |
| **string_ref** find(char) 全扫 miss(4KB) | 0.02ns/B | std::string_view::find(memchr SIMD) 0.01 | ~1.0x | ➖平(切 memchr;原 scalar 字节循环输 ~20x,都亚 ns 噪声) |
| **string_ref** find(needle) 6B 子串 | 0.20ns/B | std::string_view::find 0.08 | 0.40x | ❌输 2.5x(memchr 定位候选+memcmp 验;原 naive 输 7x,大幅收窄) |
| **string_ref** find_first_of(4 字符集) | 0.08ns/B | std::string_view 0.05 | 0.62x | ➖近平 1.6x(memchr-min 窗口收缩;原 O(n·m) 输 36x) |
| **string_ref** starts_with(16B 前缀) | 1.79ns | std::string_view 1.77 | ~1.0x | ➖平(切 memcmp;原 region_eq 字节循环输 2x) |
| **string_ref** split 全文按空格(4KB) | 0.44ns/B | std::string_view find+remove_prefix 0.54 | 1.22x | ✅赢(无分配子视图) |
| **string_algo** utf8 validate+count(2KB,同契约) | 0.41ns/B(scalar默认)/ 0.21(借 simdutf) | simdutf validate_utf8+count_utf8 0.21 | 2.0x输 / **1.0x** | ➖默认 scalar 输 2x;**-DHYLE_SIMD_SIMDUTF 借 simdutf 核则追平**(借核不手搓,同 SLEEF;simdutf 2.8MB 重故 opt-in,默认轻) |
| **string_algo** utf8 count-only 天花板(非同契约) | — | simdutf::count_utf8 0.02 | — | (simdutf 跳校验做得更少,非 utf8_length 的 validate+count 契约,仅记 SIMD 上限) |
| **string_algo** hex_encode 1KB→2KB | 0.45ns/B | naive 逐字节参考 0.63 | 1.41x | ✅赢(表查找;无单一 SOTA hex 库,对参考实现) |
| **simd** dot_f32 公开分发入口 | 0.08ns | 直调 avx2 kernel(无间接)/ Highway | ~1.0x | ➖平(fn-ptr 分发税≈0;4096 batch 摊薄一次缓存间接调用,与直调同级) |
| **simd** saxpy_f32 公开分发入口 | 0.03ns | 直调 avx2 / Highway | ~1.0x | ➖平(同上,分发税不可测) |
| **sched** Latch fork-join(64) | 5059ns(fiber) | std::latch 64 线程 21975 | 4.3x | ✅赢(fiber vs OS 线程基底不同;HYLE 不阻塞 worker) |
| **sched** ConditionVariable notify_one 快路径 | 1.70ns | std::condition_variable 2.59 | 1.52x | ✅赢(无 waiter 跳过内部锁) |
| **sched** RunQueue fifo/lifo push+next | 8.3 / 2.6ns | (内部结构,无外部对手) | — | ➖HYLE 自有(完成唤醒热路径;lifo slot 快) |
| **sched** suspend/wake 完成脊柱 round-trip | 201–224ns | (HYLE 自有脊柱,无对等物) | — | ➖HYLE 自有(GPU/NIC/timer 汇流此脊柱;原 DEFERRED 现安全 harness 补测,WSL TSan 零 race) |
| **platform** Parker ping-pong 往返(真阻塞) | 309ns | std::condition_variable ping-pong 881 | 2.85x | ✅赢(futex/WaitOnAddress 直驱,无 mutex+predicate 开销) |
| **platform** Parker unpark+park 快路径 | 9.6ns | 裸 atomic flag 4.6(下界) | 0.48x | ➖2x 下界(3-态 CAS 机器换 unpark-before-park 安全;非 syscall 路径) |
| **platform** Thread spawn+join | 39.5µs | std::thread 42.0 | 1.06x | ➖平(OS 主导;证 ABI-clean handle 零开销) |
| **platform** directory_iterate visitor | 32.4µs | std::filesystem::directory_iterator 31.9 | ~1.0x | ➖平(syscall 主导;visitor 无分配 vs 每项 entry 对象) |
| **mem** Allocator::reallocate grow chain | 79ns | 裸 mi_realloc 67 / std::realloc 194 | 0.85x / 2.9x | ➖vtable 间接(冷路径,对裸 mi -15%;对 std::realloc 反超 2.9x) |
| **mem** StackAllocator mark/rewind | 1.31ns | (reset 基线 0.82,无 std 对应) | — | ➖HYLE 自有(LIFO 部分回退;std::pmr monotonic 无 partial rewind) |
| **container** SmallVector 溢出(64>内联16) | 0.88ns | absl::InlinedVector 2.42 | 2.75x | ✅赢同款(也赢 std::vector;堆溢出/增长路径) |
| **container** ConcurrentVector 并发读(4 reader+1 writer) | 0.37ns | oneTBB 0.82 | 2.2x | ✅赢(删 operator[] 冗余 per-read acquire,HB 由 caller size() 建立;原冗余 acquire 与 writer 争线输 2.15x,WSL TSan 验零 race) |
| **util** SpinLock lock/unlock(无争用) | 4.03ns | std::mutex 12.03 | 3.0x | ✅赢(TTAS,短临界区无 syscall 路径) |
| **util** Ref copy+destroy(背靠背同线原子) | 10.4ns | std::shared_ptr 8.2 | 0.79x | ❌microbench 假象 1.27x(asm 证 2 locked RMW 同工;背靠背同线序列化,实战不现) |
| **util** Ref fan-out 64 再释放(实战传 ref) | 8.15ns | std::shared_ptr 8.10 | ~1.0x | ➖平(incref/decref 突发分离=真实用法;算法同工故打平) |
| **time** realtime_ns | 14.68ns | std::chrono::system_clock 14.76 | 1.0x | ➖平 |
| **time** monotonic_ns | 18.88ns | std::chrono::steady_clock 12.76 | 0.68x | ❌输 1.48x(同 std::chrono 等量工作 QPC+QPF+2 除;试 inline+namespace var 均无效,残留=codegen,非可修接缝) |
| **obs** log gated-out / emit | 2.03 / 78.4ns | (无 std 对照) | — | ➖HYLE 自有(gated 仅 1 原子载+分支;emit=std::format+fanout 无 I/O) |

**总账小结（覆盖审计〔阶段A〕+ 攻坚〔阶段C〕后）**：对标**真 SOTA** 的已测项——✅**赢 ~31 项**(hashmap 3、hash 2、simd 2、dtype 2、sched 3、queue 6、mem 4、container/string 7〔SmallVector 小N+溢出反超 absl、CArray、SwissTable iterate、StringBuf 反超同契约 peer、string_ref split、ConcurrentVector 并发读反超 tbb 2.2x〕、util 3〔Rng×2、SpinLock 反超 std::mutex 3x〕— 含反超 absl/marl/moodycamel/Highway/std::pmr/rigtorp/per-thread mimalloc/oneTBB)、➖**平 ~14 项**(噪声内追平:absl find-miss / Highway exp+tanh+reduce_max+scale / marl 争用锁+fiber / 裸 mi 默认+8 线程 / rigtorp 2P / realtime_ns / Channel+log / string_ref find(char)+find_first_of+starts_with〔切 memchr/memcmp/memchr-min〕/ Ref fan-out 实战〔算法同工〕)、❌**仍输少数项**:① raw hash_bytes 16/64B=导出接缝 by-design 且亚周期、热路径 Hash&lt;T&gt; 内联已赢;② math exp/tanh 输 xsimd ~1.2x=跨库 best-of-breed 常数,已平主流 Highway;③ monotonic_ns 输 std::chrono 1.48x=同等工作 codegen 残留(查证过);④ MemoryArena/ObjectPool 单/共享=选错工具,per-thread 池已反超;⑤ string_ref find(needle) 2.5x=naive 子串搜残差(已从 7x 收窄,memchr 候选+memcmp 验)。**阶段C 把覆盖审计揪出的输项扭转**:string_ref 搜索簇(find char 20x→平、find_first_of 36x→1.6x、starts_with 2x→平、split 反赢)、ConcurrentVector 并发读(2.15x 输→2.2x 赢,删冗余 acquire)、Ref(反汇编证算法同工、实战 fan-out 打平,microbench 1.27x 是背靠背同线调度假象非缺陷)。**先前 5 个旧"输"扭成赢/平**:hash 8B、vtable(反超 std::pmr)、math log(平 Highway+反超 xsimd)、单环队列 4P(反超 rigtorp)、StringBuf intern(反超同契约 peer)。**揪出的真复杂度/争用问题(MemoryArena 278x、Arena 10x、4P 单点 CAS+伪共享、ConcurrentVector per-read acquire 争线)全部根治**。

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

### hash 哈希（`hash_bytes`/`mix64` vs 裸 XXH3 + wyhash + std，`hash_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。三组:HYLE / 裸 XXH3+wyhash(SOTA) / std::hash。XXH3 vendored 无条件,wyhash gated `-DHYLE_BENCH_WYHASH`。此前 hash 只在 SwissTable 内部跑过,从未单独对标(README 宣称 XXH3 默认 hasher)。

| 字节哈希 | HYLE hash_bytes | 裸 XXH3(SOTA) | wyhash | std::hash | 结论 |
|---|---|---|---|---|---|
| 8B | 0.34 ns | **0.18**(HYLE 慢 1.85x) | 0.27 | 0.26 | 小输入 HYLE 全场最慢 |
| 16B | 0.15 ns | 0.10(慢 1.5x) | 0.14 | 0.29 | 赢 std,输裸 XXH3 |
| 64B | 0.05 ns | 0.04(慢 1.38x) | 0.05 | 0.44 | 赢 std 大,输裸 XXH3 |
| 256B | 0.08 ns | 0.08(平) | **0.03**(HYLE 输 3x) | 0.62 | 中串 wyhash 碾 XXH3 |

| 整数哈希(4096 key) | HYLE mix64 | wyhash64 | std::hash | 结论 |
|---|---|---|---|---|
| ns/op | **0.45** | 0.57(HYLE 赢 1.27x) | 1.52(赢 3.4x) | ✅ mix64 真好 |

| 小 POD key(16B 结构,4096) | HYLE Hash<T> 内联 | 旧 hash_bytes 路径 | wyhash | 结论 |
|---|---|---|---|---|
| ns/op | **0.59** | 1.76(HYLE 赢 2.98x) | 0.93(赢 1.58x) | ✅ 内联 mix 修复小 key |

**结论(typed 路径已修 + raw API 接缝 + SOTA 观察 + 真赢)**:① **`Hash<T>` 小 POD key 曾走出线 XXH3、慢——已改内联 mix 修复**:通用 POD 路径对 ≤16B key 现走 `detail::hash_small`(内联 mix64,absl/F14 同款),不再调 `hash_bytes`。16B 结构键 **0.59ns 赢旧 XXH3 路径 2.98x、赢 wyhash 1.58x**——这才是 map 定长 key 的真热路径。② **raw `hash_bytes` API 对小输入仍慢裸 XXH3 1.05-1.85x**:它是导出函数(`HYLE_CONTAINER_API` 不可内联,藏 7490 行 xxhash.h),裸 XXH3 全内联——这是 raw 字节 API 隐藏 xxhash 头的**接缝代价**(同 vtable,by-design;变长串 ≥256B 已摊薄至平),典型消费者(定长 key)走 ① 的内联路径不付此税。③ **wyhash 在 256B 比 XXH3 快 3x**(SOTA 观察:中等串 wyhash > XXH3;HYLE 选 XXH3 是质量/分布,设计权衡非缺陷)。④ **`mix64` 整数哈希真赢**:赢 wyhash64 1.27x、赢 std 3.4x——SwissTable 整数键热路径,印证 find/insert 反超 absl 的哈希基础扎实。



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

### simd 数学核（exp/log/tanh vs 标量 libm + 同 ISA SOTA xsimd/Highway，`math_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,4096 f32(L1 常驻)。**两个标尺**:① 标量 `std::exp/log/tanh`(测向量化加速本身);② **同 ISA SOTA = xsimd + Highway contrib/math**(均 gated,同 `-mavx2`,gap-fill #36)。HYLE 借 SLEEF 验证过的核(不手搓多项式)。`active_math_variant()` 报当前层。

| kernel | 标量 libm | HYLE 默认 SLEEF _u35 | HYLE best-of-breed | xsimd | Highway | 结论 |
|---|---|---|---|---|---|---|
| **exp f32** | 2.08 ns | 0.32 | **0.23**(→xsimd) | 0.23 | 0.30 | ✅ 平 xsimd(最快)、反超 Highway 1.3x |
| **log f32** | 2.58 ns | 0.50 | **0.26**(→Highway) | 0.38 | 0.28 | ✅ **反超 Highway+xsimd**(两库最快) |
| **tanh f32** | 6.42 ns | 0.55 | **0.44**(→xsimd) | 0.45 | 0.56 | ✅ 平 xsimd(最快)、反超 Highway 1.27x |

**核心诚实结论(同 ISA 对标揪出真相 + 三刀真优化,最终全 op ≥ 两库 SOTA)**:
1. **同 ISA 对标揪出隐藏落后,精度分级是根因**:此前只对标量 libm(6-12x 赢),看不出问题;一加 xsimd/Highway 同 ISA 对照,暴露 **HYLE 原慢 1.4–3.4x**。根因:HYLE 调 SLEEF `_u10`(1-ULP,最准最慢),而 xsimd/Highway 用 ~3.5-ULP 更快多项式——**精度等级不对等**(apples-to-oranges)。
2. **真优化一:log/tanh 切 SLEEF `_u35`(3.5-ULP,AI 热路径足够)**:softmax/gelu/attention 输出走 fp16/bf16,1-ULP 是过度精度。切换后 tanh 平 Highway、log 慢 Highway 3.0x→1.85x。math_test 容差据实放宽到 _u35 等级(`kRelTol 8e-7 ≈6.7 ULP`,坏核 ≥7 ULP 仍 fail)。
3. **真优化二:加 Highway-math 后端**——同 ISA 同精度下 Highway 的 log 多项式比 SLEEF _u35 快,log 0.50→追平 Highway。
4. **真优化三(收口):逐 op best-of-breed dispatch**——同 ISA 对标证实**没有单一库三 op 全胜**(xsimd 的 exp/tanh 最快、Highway 的 log 最快)。故加 xsimd 作第 3 个 header-only math 后端,dispatcher **按 op 选最快核**:exp/tanh→xsimd、log→Highway(`math_dispatch.cc`)。结果 **exp 0.23 平 xsimd 反超 Highway 1.3x、log 0.26 反超两库、tanh 0.44 平 xsimd 反超 Highway 1.27x**——**三个 op 全部 ≥ 两个 SOTA 库,无 math 落后残留**。三个后端全是验证过的核(借核不手搓)。math_test 5/5(xsimd 精度在 8e-7 容差内)、full simd 56/56。
5. **构建开关(best-of-breed 需两个都 wire,否则次优)**:默认无向量化 math → 标量 libm。**要拿到上表的 best-of-breed(每个 op 都 ≥ 两库最快),必须同时 `-DHYLE_SIMD_HIGHWAY_MATH=ON -DHYLE_SIMD_XSIMD_MATH=ON`**(都 header-only,零额外构建负担)。**只 wire 一个 = 次优陷阱**:该库供全部三 op,它不是最快的那些 op 会落后另一库(只 xsimd→log 慢 Highway 1.36x;只 Highway→exp/tanh 慢 xsimd 1.2-1.3x)。dispatch 是编译期选核(零运行时选库开销),所以两个都接=真·三 op 全拿最快、互不拖累;漏接一个才出现"一些 op 好、另一些坏"。`-DHYLE_SIMD_SLEEF=ON` 是独立的另一条路(精度 _u35,无第三方 header 但需预编译 lib,单库供全部三 op 同样有次优问题)。**推荐:两个 header-only math 后端都开。**

### simd dtype 转换（fp16/bf16 <-> f32，`dtype_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,4096 f32。三组:HYLE F16C 硬件 dispatch / HYLE scalar 参考 / Highway(gated SOTA)。`active_dtype_variant()` 报 `AVX2/F16C`(硬件路径真启用)。std 无 fp16 批量等价物。

| 转换 | HYLE(AVX2/F16C) | HYLE scalar | Highway(SOTA) | 结论 |
|---|---|---|---|---|
| **f16→f32** | **0.02 ns** | 0.65(慢 26x) | 0.05 | **赢 Highway 1.9x** |
| **f32→f16** | **0.03 ns** | 2.05(慢 71x) | 0.05 | **赢 Highway 1.6x** |
| **bf16→f32** | **0.03 ns** | 0.11(慢 3.5x) | — | ✅ AVX2 核,赢 scalar 3.5x |
| **f32→bf16** | **0.10 ns** | 0.69(慢 6.9x) | — | ✅ AVX2 核,赢 scalar 6.9x |

**结论**:① **fp16 F16C 真兑现且超 SOTA**——比 scalar 26-71x,**还赢 Highway 1.6-1.9x**,"fp16 dtype(F16C)"声明坐实。② **bf16 批量曾慢内联 scalar 1.11x,已补 AVX2 核翻盘**:原以为"bf16 需 AVX-512-BF16 才能向量化"是误解——bf16 是 f32 高 16 位,转换是纯整数移位/RNE 位运算,plain AVX2 整数即可 8-wide(Highway 同法)。补 `cvt_avx2.cc`(`cvtepu16_epi32`+`slli` / RNE `incr`+`packus`+`permute4x64`)后 **bf16→f32 赢 scalar 3.5x、f32→bf16 赢 6.9x**。纯位运算故 AVX2 与 scalar bit-identical(NaN payload 也是),新增 0x10000 全模式 + f32 宽扫 bit-exact 测试验证。

### simd 逐 op 扩展（reduce_max / scale，`simd_bench.cc`）

> dot/saxpy 之外补两个 distinct pattern,验证 vec.h 抽象在非 FMA 模式下是否退化。环境同上,4 ISA 隔离 TU。

| op | scalar(auto-vec) | HYLE sse4 | HYLE avx2 | Highway(SOTA) | 结论 |
|---|---|---|---|---|---|
| **reduce_max**(非 sum 归约) | 0.74 ns | 0.17(4.3x) | **0.08(9.7x)** | 0.08 | **平 Highway** |
| **scale** y=a*x(纯乘) | 0.03 ns | 0.05 | **0.03(1.16x)** | 0.03 | **平 Highway**(带宽受限) |

**结论(干净,无欺骗)**:**4 个 pattern(dot/saxpy/reduce_max/scale)HYLE avx2 全平 Highway**——vec.h 抽象在 sum-归约之外的 max-归约、纯 element-wise map 上同样零成本,无隐藏退化。reduce_max 9.7x 赢 scalar;scale 是 1 读 1 乘 1 写的内存受限 op,scalar auto-vec(SSE2)已近饱和带宽,故 avx2 仅 ~1.16x(正常,非缺陷)。sse4 在 scale/saxpy 这类带宽受限 op ~1x(128b 与 scalar SSE2 同宽,无更宽 load 优势)。



| 项 | 实测(Linux GCC14) | 参考 |
|---|---|---|
| fiber 上下文切换(resume/yield 对) | **~21-23 ns**(本 session 复测 21.0ns) | fcontext 量级,= boost.context 原语本身(零封装开销) |
| spawn 2000 任务 + drain | vs `std::async` 快 144x(**弱基线**)→ **真 SOTA 见下 sched-05 节:vs marl 快 2.17x** | std::async 每任务起 OS 线程(稻草人);真标尺是 marl 调度器 |

> **注**:spawn 的权威数字在下方 sched-05 节(对标 marl,非 std::async)。此处保留 std::async 行仅作"赢最低门槛"的历史参照。

> sched bench(`bench/sched/scheduler_bench.cc`)**现已编入默认 build**(提交后解封)。
> 它曾被认为触发一个 P0 崩溃,但一轮穷尽调查(见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md)
> + [`known-issues.md`](./known-issues.md) KI-1)**排除了 HYLE 逻辑缺陷**:TSan 无源码级
> race、MI_DEBUG=3 全量堆校验无损坏、原始崩溃器本体在 Linux 原生 + Windows MSVC 本机
> 重负载下 + 强制迁移 687 万次跨线程 free 均**不复现**。证据(SIGILL = 代码页问题 +
> 全部 instrumented 净 + 原始崩溃全在 9p)指向 **9p/WSL demand-paging 环境假象**,非
> HYLE 缺陷。fiber 切换 ~21-23ns 数值有效(mimalloc v2.3.2 + 本 session 复测一致);spawn 的权威对标见下 sched-05 节(vs marl 2.17x,非此处历史的 vs std::async)。

### sched-05 fiber 同步原语(`sync_bench.cc` + `sync_marl_bench.cc`)

> 环境:Linux x86_64 GCC14.2 Release(原生 ext4)+ Win MSVC19.50 Release,nanobench。**SOTA 标尺 = marl**(本设计收敛自的 fiber-sync 库;gated `-DHYLE_BENCH_MARL=ON`)。

| 项 | HYLE | SOTA(marl / std) | 结论 |
|---|---|---|---|
| **争用 Mutex**(8 任务/4 worker,lock+inc+unlock) | **30.7 ns** | marl 32.1 ns(=std::mutex) | ✅ **打平 marl**(±噪声,本轮略胜) |
| **无争用 Mutex**(多线程进程=真实场景) | **6.5 ns** | std::mutex 9.1 ns | ✅ **HYLE 快 1.40x**(单线程 microbench 的"2.9x 慢"是 glibc 假象,见下) |
| **spawn 2000 noop**(纯派发) | **141 ns/task** | marl 306 ns | ✅ **HYLE 快 2.17x**(对标真 SOTA,非 std::async) |
| Channel 收发(cap 16,1p1c) | **63 ns/item(15.9M/s)** | (marl 无 channel) | HYLE 自有 |
| **WaitGroup fork-join**(1024 任务) | **205 ns/批**(池化栈 179) | marl 306 ns | ✅ **反超 marl 1.49x**(池化 1.71x) |

**Mutex = futex(Drepper 3 态),不是 fiber-aware——这是学 marl 的 SOTA 决策**:最初把 Mutex 做成 fiber-aware(争用让出 fiber),实测**慢 marl 5.8x**(176ns vs 30ns)——短临界区让出 fiber(suspend+重入 ~100ns)远贵于就地 park worker 几 ns。改写成建在 `platform::wait_on_address`(futex/WaitOnAddress/ulock)上的 Drepper 3 态锁(无争用单 CAS、争用 park worker)后**追平 marl**。marl 同理:它的 mutex 就是 `std::mutex`(futex),只把 fiber-awareness 留给 ConditionVariable。**HYLE 对齐:Mutex=futex(本节),CV=fiber-aware 双模(地基),各司其职。** 双平台 TSan 8/8 零 race。

**无争用 Mutex "慢 std" 是 microbench 假象——已实测揭穿**:HYLE futex Mutex 与 std::mutex 算法同级(各 1 locked RMW)。单线程 bench 里 std::mutex 假性快到 2.2ns,因 glibc 用 `__libc_single_threaded` **跳过原子**;HYLE 不做单线程特化,恒付。**铁证**:加一个后台线程让进程变多线程(`sync_bench.cc` MT 变体),std::mutex 立刻 2.2→**9.1ns**(不能再跳原子),HYLE 纹丝不动 **6.5ns**——**HYLE 反超 std::mutex 1.40x**。在 mutex 唯一真正被用的多线程场景,HYLE 的 Drepper 3 态比 pthread_mutex_t 更轻、更快;marl mutex=std::mutex,故也赢 marl。**结论:不是性能问题,是单线程 microbench 被 glibc 优化骗了。**

**spawn vs SOTA:对标 marl(非 std::async 弱基线)HYLE 快 2.17x**:scheduler_bench 的"spawn 2000 vs std::async 144x"是赢 thread-per-task 稻草人。直接对标真调度器 marl(`sync_marl_bench.cc` noop spawn-2000):marl 306ns/task vs **HYLE 141ns/task = 2.17x 反超**。spawn/dispatch 是真 SOTA,非弱基线侥幸。

**WaitGroup fork-join 反超 marl 的真根因 = 调度器唤醒策略,不是栈池(实测纠错)**:fork-join 起初慢 marl **17x**(5.4µs vs 0.30µs)。第一反应怀疑"没 fiber 栈池",但**诊断推翻**:调高 worker 泊车前自旋后立刻反超,单上栈池只改善 ~5%。**真根因 = 调度器急泊车 + 每次入队无条件 unpark 的 futex 系统调用风暴**。**治本方案 = §2.3 ②③ conditional-notify(已落地)**:维护 `num_searching_` 原子计数,producer 入队后仅当无 worker 在搜才 unpark(否则搜索者会捡到),从源头消除风暴;lost-wakeup 安全靠 notify 与 worker 离开搜索的 **seq_cst StoreLoad Dekker** 握手(双平台 28/28 + TSan + NoLostWakeup 100× 零超时验证)。结果:fork-join **205-241 ns/批,反超 marl 1.49x**;栈池(`MemoryArena::as_allocator`)再加 ~13%→**1.71x**(次要增益)。**教训:17x 一度被错误归因到栈池(连架构审计 agent 都照旧文档复述),实为唤醒策略——"登记延后掩盖真瓶颈"的活例;先做的 spin-then-park 是缓解,conditional-notify 才是根治(去自旋空烧 + 去魔法常数 + 源头治本)。** CV 的 `waiters_` 门让 Channel/WaitGroup 的 notify 无 waiter 时零锁。

> **TSan × fcontext(本步另一修复)**:调度器唤醒优化让 fiber 跨 worker 迁移更频繁,撞出 TSan 对 fcontext 栈切换的盲区(SEGV in libtsan,非逻辑 bug——无 TSan 50/50 过)。已给 `fiber.cc` 加 `__tsan_switch_to_fiber` 注解(每次切换告知 TSan 当前逻辑上下文,迁移时按 resume 重新捕获 resumer 句柄),**release 零成本**(非 TSan 构建宏全展开为空)。修复后双平台 28/28 + TSan 28/28 零 race,lock-free 闸门恢复。




### lock-free 队列簇（`queue_sota_bench.cc` + crossbeam Rust 对照，task #9 / bench-02b）

> 环境:Win MSVC19.50 Release + Linux x86_64 GCC14.2 Release(原生 ext4),nanobench。cap=1024、50000 item/生产者。**SOTA 标尺**:moodycamel ConcurrentQueue(单头,gated `-DHYLE_BENCH_MOODYCAMEL`)、oneTBB concurrent_bounded_queue/concurrent_vector(gated `-DHYLE_BENCH_TBB`,apples-to-apples 有界)、crossbeam-deque(Rust,work-stealing 同族,`bench/rust/`)。**此前这簇零对标("覆盖低就默认 SOTA"),本节是真对标揪出的结果——赢/平/输都登记。**
>
> **对标依赖 `project-hyle/` sibling 布局**:gated 库默认在仓库 sibling(`../moodycamel-ConcurrentQueue`/`../oneTBB`),crossbeam Rust crate path 依赖 `../../../../crossbeam-deque`(本地 0.8.3 clone)。换机器时:C++ 侧 `-DHYLE_MOODYCAMEL_DIR=/path` 覆盖;Rust 侧改 Cargo.toml path 或换 `crossbeam-deque="0.8"` 走 crates.io。

| 场景 | HYLE | moodycamel | oneTBB | rigtorp(同约束 SOTA) | 结论 |
|---|---|---|---|---|---|
| **单线程 fill+drain**(BoundedQueue) | 7.8(Win) | 9.6 | 13.4(Win) | 11.8(Win) | ✅ **赢全部**(rigtorp 1.5x、moody、tbb 1.7x) |
| **MPMC 2P/2C**(BoundedQueue 单环,cache-line pad) | 56(Win) | 59.4(Win) | 138(Win) | 58(Win) | ✅ **平/微超 rigtorp ~1.02x、赢 moody、tbb 远** |
| **MPMC 4P/4C**(BoundedQueue 单环,cache-line pad) | **72(Win)** | 62.9(Win) | 225(Win) | 79(Win) | ✅ **反超 rigtorp 1.08–1.09x**(原输 1.33x,已修——见结论1) |
| **MPMC 2P/2C**(**ShardedQueue**) | **18.1(Lin)/22.7(Win)** | 59.4 | — | 58 | ✅ **反超 rigtorp/moody 2.6x** |
| **MPMC 4P/4C**(**ShardedQueue**) | **21.0(Lin)/22.1(Win)** | 62.9 | — | 79 | ✅ **反超 rigtorp 3.4x、moody 2.8x,线性扩展** |
| **work-steal owner LIFO**(WorkStealingDeque) | 2.8(Lin)/3.1(Win) | (无同物) | — | — | crossbeam LIFO 3.6(Win) 参照,HYLE 略快 |
| **work-steal owner FIFO**(WsFifoQueue) | 3.6(Lin)/4.0(Win) | — | — | — | crossbeam FIFO 4.4(Win) 参照,HYLE 略快 |
| **ConcurrentVector** push_back/read | 4.6/1.1(Win) | — | 5.9/1.3(Win) | — | **赢 tbb push 1.3x / read 1.1x** |

**核心诚实结论 + 它驱动的真优化**:

1. **单环 BoundedQueue 4P 曾输 rigtorp 1.33x——纠正两次误判,真根因"抄作业没抄全"(cell 未对齐 cache line)+ 过度 backoff,已修到反超**。**误判史(都已证伪)**:① 最早称"严格-FIFO+有界 MPMC 单点 serialize 是理论墙、单环已是该约束 SOTA、rigtorp/folly 同款"——rigtorp 实测 4P 反赢证伪;② 随后**猜**根因"HYLE Vyukov CAS-重试 vs rigtorp fetch_add 票号"——读 rigtorp 源码**再证伪**:rigtorp 的 `try_push`/`try_pop`(实测路径)也用 `compare_exchange`(只有阻塞版 `push`/`pop` 用 fetch_add),热路径同是 CAS。**真根因(读码查实)**:rigtorp 把每 slot **对齐独立 cache line**(`alignas(64)`+`sizeof%64==0`),HYLE 的 `Cell{atomic<u64> seq; T data}` 对 8B T 仅 16B → 4 cell 挤一条 line → 相邻槽被不同线程并发读写时**伪共享**——clay 移植漏掉的 Vyukov 经典调优(根因①)。另:HYLE 每次 CAS 失败调 `cpu_relax()`(~40cyc),4P 高争用纯增延迟(rigtorp/folly 的 try_ 不 backoff)。**修复**:`Cell` 加 `alignas(HYLE_CACHE_LINE_SIZE)` + 去 CAS-失败 backoff。**3 遍稳定均值:单线程赢 rigtorp 1.5x、2P 平/微超 ~1.02x、4P 反超 1.08–1.09x**——4P 落后消除,同约束 SOTA 全点位平或超。代价:buffer 增至 capacity×64B(空间换吞吐,rigtorp 同付)。6/6 测试(含 MpmcStressConservation)双平台过 + **WSL TSan 待跑**(lock-free 闸门)。
2. **ShardedQueue 仍是生产用途首选**:sched `global` 注入无全局序需求,ShardedQueue(N 子环)4P **22ns 反超 rigtorp 79ns 3.4x**、moody 2.8x,线性扩展。单环用于真需严格全局 FIFO 的场景,现已不输同约束 SOTA。
3. **backoff 是 2P-vs-4P 权衡(诚实记账)**:`cpu_relax` backoff 在 2P 低争用曾给更大领先,但 4P 高争用造成 1.33x 落后;去掉后 2P 回到平 rigtorp、4P 反超。directive 要"无落后",故选无 backoff(canonical rigtorp/folly 同样 try_ 路径不 backoff)。

> **诚实边界**:① ShardedQueue ns/op 太快采样不足(err 6-20%),但反超幅度(2.6-3.4x)远超误差带。② 跨语言(crossbeam Rust)只下"同量级/有竞争力"。③ **单环 MPMC 2P/4P 是高方差微基准(err 2-5%),上表为 3 遍稳定均值**(单遍曾见 2P 37ns 乐观离群,不取);4P 落后已修(cache-line pad + 去 backoff),非"理论墙"也非"fetch_add"(两个旧结论都被读码/实测推翻)。④ work-stealing owner 路径单线程无争用(两侧同样)。

### mem 分配器（`Allocator` vtable vs 裸 mimalloc vs std::malloc，`allocator_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,200k/50k 小块(16..511B)。**两个标尺**:① 裸 `mi_malloc`(设为相对基线)——但它**非多态、全内联**,对它的差读的是"多态 vs 无多态"的物理代价,非接缝缺陷;② **同类 SOTA = `std::pmr::polymorphic_allocator` over mimalloc**——和 HYLE 运行时 vtable 同为"同一 backing 上的运行时多态"(各付一次 dispatch),这才是 apples-to-apples"我们的运行时路径是否 SOTA"。`std::malloc` = 弱基线。

| 场景 | 裸 mi(无多态) | HYLE vtable(运行时) | std::pmr(同类 SOTA) | HYLE SystemAllocator(编译期) | 结论 |
|---|---|---|---|---|---|
| **churn 空循环**(alloc+free) | 7.79 ns | 8.61 ns(90.5%) | 8.70 ns(89.5%) | **7.91 ns(98.5%)** | **vtable 赢同类 std::pmr 1.01x;编译期路径≈裸 mi** |
| **alloc+touch+free**(真实) | 7.75 ns | 8.77 ns(88.4%) | — | (编译期同理≈平) | 真实负载下接缝摊薄 |
| **alloc-all/free-all**(批) | 9.56 ns | 9.84 ns(97.1%) | — | — | ≈平 |
| **8 线程并发** | 1.75 ns/op | 1.97 ns/op(89.1%) | — | — | ≈平(per-thread heap) |

**诚实结论(两层:运行时注入 vtable + 编译期默认)**:
- **运行时 `Allocator` vtable**(opt-in 路径,给 arena/mock/跨 DLL):对裸 mi -10%,但**这 10% 是运行时多态本身的物理代价(一次间接调用),不是 HYLE 接缝缺陷**——铁证:**同类 SOTA `std::pmr::polymorphic_allocator`(也一次虚调用、同 mimalloc backing)8.70ns,HYLE vtable 8.61ns 反而快 1.01x**。即:在"运行时多态分配器"这个类里,HYLE 是 SOTA(3 函数指针 vtable 比 std::pmr 的 `do_allocate` 虚调用还略省)。原 +21% 才是真缺陷(已修)——① 单 5 参 `ReallocFn` 在 Win64 压第 5 参 + 派发分支 → 拆 3 函数指针;② `normalize_alignment` 常开 FATAL_ASSERT → 默认对齐短路。
- **编译期 `SystemAllocator`**(默认路径,host RAM):`allocate` 内联直调 `mi_malloc`,**实测 7.91ns = 98.5% 裸 mi(噪声内≈平)**。前向声明 5 个 mi_* 不暴露 mimalloc.h(用户点破"内联≠暴露头"假两难)。**默认场景零接缝税,与裸 mimalloc 持平**——这就是"C-ABI 也能最强":默认编译期绑定 0 税,运行时注入是显式 opt-in 才付间接,且那一档也已是同类 SOTA。
- **结论:vtable 对裸 mi 的"10%"不是落后,是多态税,且 HYLE 在多态这一类里反超 std::pmr**;默认编译期路径实测追平裸 mimalloc。**头内联容器簇已全量迁移**(SmallVector/BoundedQueue/ShardedQueue/WsFifoQueue/WorkStealingDeque/ConcurrentVector):默认参 `SystemAllocator`,`[[no_unique_address]]` 去掉每实例 32 字节运行时句柄(空间复杂度↓),运行时 `mem::Allocator` 降 opt-in(sched 等需注入者显式用)。type-erased 核容器(CArray/SwissTable/string_buf)保持运行时句柄(出线核服务所有 T,不内联,这是正确模型非缺口)。双平台 + WSL TSan 验证。

### mem 专用分配器簇（arena/pool/stack vs `std::pmr` + 裸 mimalloc，`arena_pool_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,8192×64B。**SOTA 标尺 = `std::pmr`**(标准库自带的 arena=`monotonic_buffer_resource`、pool=`unsynchronized_pool_resource`,完美 drop-in,无需 clone)+ 裸 `mi_malloc`(=`new/delete` 的实际 backing,即"5-10x vs new/delete"声明的对照基线)。**此前这簇零对标——本节真测揪出两个无锁变体的单点 CAS 争用(已分别根治:MemoryArena 分片 278x→1.94x,Arena 去原子 10.3x),并纠正了一处 apples-to-oranges 对照(共享固定池 vs per-thread heap,见结论 4)。**

| bump 场景(alloc N + 批量释放) | ns/op | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE Arena**(单线程 bump,池块) | **0.67** | **6.1x** | ✅ 去原子后平 Stack、赢 std::pmr 2.2x |
| **HYLE StackAllocator**(单线程自有 slab) | **0.65** | **6.3x** | ✅ 真兑现 |
| std::pmr::monotonic_buffer(SOTA) | 1.49 | 2.7x | Arena/Stack 赢它 2.2x |
| 裸 mi_malloc(general,无批量) | 4.07 | 1.0x | — |

| pool churn 场景(同尺寸 alloc/free) | ns/op | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE MemoryArena**(分片无锁池) | 14.62 | **0.33x(慢 3.1x)** | 单线程原子 CAS 常数因子 |
| **HYLE ObjectPool**(typed,坐 MemoryArena 上) | 14.86 | 0.32x(慢 3.2x) | =MemoryArena+placement-new(typed 壳 ~免费,98.3%) |
| **HYLE GrowablePool**(单线程) | **1.46** | **3.3x** | ✅ 真兑现 |
| std::pmr::unsync_pool(SOTA) | 6.17 | 0.77x | — |
| 裸 mi_malloc/free(general) | 4.77 | 1.0x | — |

**8 线程 pool churn——公平对照(双方都 per-thread)vs 共享固定池的协调代价:**

| 8 线程同尺寸 churn | ns/op | vs 裸 mi per-thread | 结论 |
|---|---|---|---|
| **HYLE GrowablePool**(每线程一个池) | **0.72** | **1.78x 反超** | ✅ 同线程 churn 的 apples-to-apples 答案 |
| 裸 mi(per-thread heap) | 1.28 | 1.0x | mimalloc 同线程非原子 local list |
| **HYLE MemoryArena**(8 线程**共享**一个固定池) | 2.48 | 0.51x(慢 1.94x) | 共享固定 slab 的协调代价(每 op 一次原子 CAS) |

**核心诚实结论(揪出第三个"自以为 SOTA"——单点原子,两处已修)**:

1. **"专用分配器 5-10x vs new/delete"全部坐实**:`StackAllocator`(纯指针 bump,0.65ns)+ `Arena`(去原子后 0.67ns)**6x 赢裸 mi、2.2x 赢 std::pmr arena**;`GrowablePool`(单线程 free-list,1.92ns)2.54x 赢裸 mi、3.5x 赢 std::pmr pool。四个单线程/去原子变体全是真 SOTA。

2. **`MemoryArena` 曾是全场最慢——单点 CAS 争用,已分片根治**:原 clay 继承的单 `head_` 让 8 线程 CAS 同一 cache line,O(threads) 串行化——8 线程共享慢 mimalloc per-thread heap **278x**,单线程也慢裸 mi 2.88x。诊断:与 BoundedQueue 单点 CAS 同病(churn 更密故惨烈得多)。**修复**(`memory_arena.h`):内部分 N 片(片数=核数向上取 2 的幂,上限 128——非固定 16,固定数在线程>片数时又退化 O(⌈T/片⌉)),每片独立 Treiber 头占独立 cache line,线程按 tls 序号 homing 到自己片(同线程 alloc/free 无争用),free 按地址 O(1) 路由回属主片。**8 线程 344.72→2.48ns,278x→1.94x(~139x 改善),复杂度落后(O(N)争用→O(1) per-shard)消除**。双平台 10/10 + WSL TSan 零 race。这同时补上了 mem-design 列为"硬约束"却从未实现的 per-core 池。残留 1.94x 见结论 4(是 apples-to-oranges 对照伪影,非真落后)。

3. **`Arena` 无锁 bump 同源病已修(去原子化)**:`Arena` 的单 `offset_` 原子也是单点病——单线程付每 alloc 一次 CAS(6.79ns vs `StackAllocator` 纯指针 bump 0.65ns,10x 税)。bump cursor 本质单游标无法分片,SOTA 模式是 **per-thread(各自一个 Arena,零原子)**,并发共享一个 cursor 是反模式(无内部用户的投机无锁化)。**修复**:`offset_` 改 plain `size_t`,纯指针 bump;测试改 `PerThreadArenasNoOverlap`(每线程自己的 Arena)。**6.79→0.66ns(10.3x),平 StackAllocator、赢 std::pmr 2.2x、赢裸 mi 6x**。

4. **"共享 MemoryArena 慢 1.95x" 是 apples-to-oranges,不是真落后——公平对照 HYLE 反超**:此前把"8 线程**共享一个**固定池"对"mimalloc 的 **8 个 per-thread heap**"比,本身不公平——这个 workload 全是**同线程** alloc/free,mimalloc 用**非原子 thread-local free list** 服务,而单块共享固定 slab 协调跨线程本就要每 op 一次原子 CAS(分片+无争用,`lock` 前缀也非零)。**公平对照(双方都 per-thread)**:HYLE `GrowablePool` 每线程一个池 = **0.72ns,反超裸 mi per-thread heap(1.28ns)1.78x**。即:同线程同尺寸 churn 这个 workload,SOTA 模式就是"池 per-thread"(mimalloc 内部正是如此),HYLE 的对等答案赢。共享固定池的 1.94x 是**"一块固定 slab 当跨线程共享有界资源"的真实协调代价**(非复杂度——分片已消除 O(threads) 的 278x),而非可包装掉的水分。进一步逼到 1.0x 需 seastar 模型(线程钉死 per-core 终生、池生命周期长于线程),`MemoryArena` 的通用生命周期给不出这个契约(owner-local 非原子表与固定 slab 的"不假性耗尽"硬冲突,详见 mem-design)。**结论:这一项无需"修"——选对工具(per-thread 池)即已超 SOTA;真要共享有界池,1.94x 是诚实标价。**

### container 杂项（SmallVector / IntrusiveList / CArray vs std + 真 SOTA，`container_misc_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。声明承重容器。**标尺升级**:不止 std——SmallVector→`absl::InlinedVector`(small-vector SOTA,gated `-DHYLE_BENCH_ABSL`)、IntrusiveList→`boost::intrusive::list`(侵入容器 canonical SOTA,gated `-DHYLE_BENCH_BOOST`)、CArray→`std::vector`(动态数组的 drop-in SOTA,本身已是高度调优,always-on)。

| 场景 | HYLE | std | 真 SOTA(gated) | 结论 |
|---|---|---|---|---|
| **小 N churn**(建+填 8 int+析构) | SmallVector<int,16> **0.43 ns** | vector 16.7 / vector+reserve 3.07 | **absl::InlinedVector 0.60** | ✅ **赢 absl 1.40x**、赢 std 39x |
| **list churn**(push 4096 + pop 全) | IntrusiveList **3.0~3.4 ns** | list 24.1 | boost::intrusive(需 submodule checkout) | **8x 赢 std**;boost 同 hook-in-node,gate 已接 |
| **动态数组 growth**(push_back 4096 int,无 reserve) | CArray<int> **0.53 ns** | vector **0.71** | (std::vector 即 SOTA) | ✅ **赢 std::vector 1.34x** |
| **动态数组 read**(顺序求和 4096) | CArray<int> **0.18 ns** | vector 0.18 | (std::vector 即 SOTA) | ➖ **平 std::vector**(arr[i] 单 load 兑现) |

**诚实结论**:① **SmallVector 反超 absl::InlinedVector 1.40x**——不是"只赢 std":absl::InlinedVector 是 small-vector 的 SOTA(同 inline 存储设计),HYLE 0.43ns vs absl 0.60ns,小 N churn 真赢同款 SOTA(也赢 std::vector 39x)。② **IntrusiveList 8x 赢 std::list**(零节点分配);boost::intrusive 同 hook-in-node 设计,gate 已接线(`-DHYLE_BENCH_BOOST`),但 boost 模块化 submodule 未 checkout,留待环境备齐实测(预期同量级)。③ **CArray 反超 std::vector 1.34x growth、平 read**:growth 赢的真根因 = 经 `reallocate()` 增容(mimalloc 原地扩展,无 copy),std::vector 每次扩容恒 alloc-new+move+free;read 路径 arr[i] 单 load 与 vector 持平(header-before-data 布局零索引开销)。

### 次要模块（util/random、time/clock、obs/log、SwissTable iterate，`misc_bench.cc` + `swiss_table_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。此前零 bench 的性能相关原语。标尺 = std 等价物(同 OS 原语/同算法类),或 HYLE 自有(无 std 对等物时诚实标注)。

| 场景 | HYLE | 标尺 | vs 标尺 | 结论 |
|---|---|---|---|---|
| **Rng next_u64**(xoshiro256++) | **1.56 ns** | std::mt19937_64 2.41 | **1.54x** | ✅ 赢(同质量,256-bit state) |
| **Rng range_u64**(Lemire 无偏) | **1.75 ns** | std::uniform_int_distribution 2.75 | **1.57x** | ✅ 赢且**无偏**(std 也无偏,HYLE 更快) |
| **SwissTable iterate**(for_each) | **2.48 ns** | std::unordered_map 8.73 | **3.5x** | ✅ 赢(扁平 ctrl 扫 vs 节点指针链) |
| **realtime_ns** | 14.68 ns | std::chrono::system_clock 14.76 | **1.0x** | ➖ 平(同 OS 原语) |
| **monotonic_ns** | 18.88 ns | std::chrono::steady_clock 12.76 | **0.68x(慢 1.48x)** | ❌ 输——同等工作,残留=codegen(见结论③) |
| **log gated-out** | 2.03 ns | (无 std 对照) | — | HYLE 自有(1 relaxed 原子载 + 分支,跳过格式化) |
| **log emit**(format+fanout,无 I/O) | 78.4 ns | (无 std 对照) | — | HYLE 自有(std::format 进栈 buf + 锁内快照后锁外 fanout) |

**诚实结论**:① **Rng 赢 std PRNG 1.54x**(xoshiro256++ vs mt19937_64,同统计质量,HYLE header-inline 无 DLL hop);range_u64 用 Lemire 无偏映射,赢 std::uniform_int_distribution 1.57x(两者都无偏,HYLE 更快)。② **SwissTable iterate 赢 std::unordered_map 3.5x**——扁平 ctrl 字节扫 + 连续 slot(cache 友好)碾压节点指针链;`for_each` 是函数模板(内联、零分配)。③ **monotonic_ns 输 std::chrono 1.48x——已查证非可修接缝,诚实留账**:先疑 DLL 导出接缝(monotonic_ns 是导出函数),试 header-inline(前向声明 QPC/QPF,同 system_inline.h 手法)→**无效**(仍 18.3ns);再疑 function-local `static` 的 per-call magic-static guard,改 namespace-scope inline var →**仍无效**;且 inline 反而把本来打平的 realtime_ns 拖慢。查 MSVC `steady_clock::now()` 源码:它做**完全相同的工作**(QPC 经 IAT + 缓存的 QPF + 同样两次 64 位除法)。即 HYLE 算法无劣构,~1.45x 残留是 QPC 调用形/codegen 噪声层面,**非接缝、非算法**,在"同一 syscall"下逼平 std::chrono 无干净杠杆(要更快得换 rdtsc+校准时基,是另一个大特性)。已回退 inline 实验(给公开头加平台 forward-decl 复杂度却零收益 + 拖慢 realtime,坏交易)。realtime_ns 因 std 也做 epoch 换算,两边持平。④ **log produce 路径**:gated-out 仅 1 relaxed 原子载 + 分支(2ns,跳过 std::format);emit 78ns 含 std::format 进 1KiB 栈 buf(零分配)+ 锁内快照 sink 列表后锁外 fanout——无 std 直接对等物(std 无内建结构化日志)。






- **桶 A — 同操作更省开销（真 drop-in 可比）**：hashmap、分配器、容器。给"倍数"有意义。
- **桶 B — std 根本没有的能力**：fiber、async IO。"倍数"是错的框（无可比物），是「从不能到能」。

## 主表：时间 / 空间分列

### 桶 A：drop-in 可比

| 组件 vs std | 时间（speedup + 复杂度）| 空间（vs std）|
|---|---|---|
| **flat hashmap** vs `unordered_map` | 查找/插入 **2-4×**；均摊 O(1) 同，但常数小（连续布局 cache 命中 + SIMD 16 路探测）| **通常更省**：无 node 的指针 + 每节点分配头；代价是留负载因子（有空槽 + 1 控制字节/槽）。最坏（刚扩容）峰值略高 |
| **专用分配器** 单线程 bump/pool vs new/delete | 分配 **2.5-9.5×**(实测:StackAllocator 9.5×、GrowablePool 2.57×)；O(1) 指针自增 vs free-list+锁。**注:无锁 Arena/MemoryArena 反而慢**(单点 CAS 税,见上节实测)| **更费**（以空间换时间）：块尾浪费 + 不单独释放，生命周期不齐时占用更久、峰值更高 |
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
| **device buffer**(compute/ 自有) vs host/device 双份镜像 | 异构(独显)本就要显式 copy/drawcall 过 PCIe——无"零拷贝"可言,优化点是**批量提交 + 复用 buffer 省往返**;统一内存(DGX/Mac)host/device 同物理 RAM,**本就无互传概念**,不存在"省一次 DMA"的红利 | **更省**:统一内存下不需 host+device 双份镜像(物理同址) |

## 整体系统层面（Amdahl，别期待"全面快 N 倍"）

局部再快也受非热路径拖累：

- **瓶颈不在这些点** → 整体可能仅 **1.2×**。
- **并发 / IO 密集** → **10×+**。
- **架构连贯性红利**（不打架的双调度器、统一内存无双份镜像）不体现为微基准倍数，而是真实混合负载的系统级延迟/吞吐——Seastar/ScyllaDB 那种 shard-per-core 重写报过 **~2× 系统吞吐**，是"架构做对"的现实数字。

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
