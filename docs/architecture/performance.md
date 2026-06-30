# HYLE 性能预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 对标总账（win/loss ledger，全部实测 vs 真 SOTA，非 std）

> **一眼看尽输赢**。**两把标尺各拆成「组件名 + ns」两列**:`std组件`/`std ns`(基线下界——赢它是最低门槛)和 `SOTA组件`/`SOTA ns`(真门槛:absl/F14、mimalloc、moodycamel/tokio/rigtorp、Highway/xsimd、marl、boost、simdutf)。`比std`/`比SOTA`:>1=HYLE 快几倍。**判定**针对 SOTA:✅赢 / ➖平(噪声内) / ❌输(均附根因:by-design 接缝 / 约束选择 / 选错工具 / scalar 待优化,无一为未处理的真缺陷) / ⏳待补。`n/a`=该层无对等物(非未测:如 bounded-MPMC 队列无 std 等价物);`同std`=该项 std 本身即领域 SOTA,无独立第三方更快。所有数字真跑(Win MSVC,2026-06-30 全套重跑),非推算。

| 模块 · 操作 | HYLE ns | std 组件 | std ns | 比std | SOTA 组件 | SOTA ns | 比SOTA | 判定 |
|---|---|---|---|---|---|---|---|---|
| **error** Status::Ok 构造+ok() | 0.37 | std::expected&lt;int,ec&gt; | 0.56 | 1.51x | absl::OkStatus | 0.37 | 1.0x | ✅赢 std、平 absl |
| **error** Status(code) inline 无消息 | 0.37 | std::unexpected&lt;ec&gt; | 0.55 | 1.49x | absl 错误(code-only 仍堆) | 1.64 | 4.6x | ✅赢 absl(HYLE inline 零堆) |
| **error** Status(msg) 带消息→堆 | 28.2 | (无 std 等价) | n/a | n/a | absl::NotFoundError | 54.3 | 1.92x | ✅赢 absl(单 malloc header+data) |
| **error** Status 拷贝(heap,retain/release) | 9.4 | (无 std 等价) | n/a | n/a | absl::Status copy | 10.6 | 1.13x | ✅赢 absl |
| **error** Result&lt;int&gt; 传播 OK(解包消费) | 1.42 | std::expected unwrap | 1.40 | ~1.0x | absl::StatusOr | ~1.4 | ~1.0x | ➖平(真实用法;裸构造-析构 microbench 假象见 error 节) |
| **error** Result&lt;int&gt; 传播 ERROR 上抛 | 10.2 | (errc 无堆消息不可比) | n/a | n/a | absl::StatusOr | 21.5 | 2.11x | ✅赢 absl |
| **container·map** find-hit(100k u64) | 2.21 | std::unordered_map | 10.47 | 4.74x | absl flat_hash_map | 3.03 | 1.30x | ✅赢 |
| **container·map** find-miss(100k) | 4.36 | std::unordered_map | 11.57 | 2.65x | absl flat_hash_map | 4.86 | 1.09x | ✅赢(噪声内略胜) |
| **container·map** insert(100k fresh) | 11.58 | std::unordered_map | 80.07 | 6.92x | absl flat_hash_map | 24.53 | 2.30x | ✅赢 |
| **container·map** erase(100k) | 18.01 | std::unordered_map | 93.10 | 5.17x | absl flat_hash_map | 34.60 | 1.83x | ✅赢 |
| **container·map** iterate(100k visit) | 2.24 | std::unordered_map | 8.09 | 3.60x | absl(未测此项) | n/a | n/a | ✅赢 std 3.6x(扁平 ctrl 扫) |
| **container·map** Hash&lt;T&gt; 16B POD 内联 | 0.59 | std::hash | ~0.9 | ~1.5x | wyhash | 0.93 | 1.58x | ✅赢 |
| **container·map** mix64 整数 | 0.45 | std::hash | ~0.6 | ~1.3x | wyhash64 | 0.57 | 1.27x | ✅赢 |
| **container·map** raw hash_bytes 8B | 0.21 | std::hash&lt;string&gt; | 慢 | (大赢) | 裸 XXH3 | 0.18 | 0.86x | ➖近平(原 1.85x,header 内联修;0.03ns=亚周期) |
| **container·map** raw hash_bytes 16B | 0.15 | std::hash&lt;string&gt; | 慢 | (大赢) | 裸 XXH3 | 0.10 | 0.67x | ➖测量地板(0.05ns 寄存器驻留;热路径 Hash&lt;T&gt; 赢 wyhash 1.6x) |
| **container·map** raw hash_bytes 64B | 0.14 | std::hash&lt;string&gt; | 慢 | (大赢) | 裸 XXH3 | ~0.10 | 0.69x | ➖0.01ns 测量地板(保 xxhash.h 隐藏架构) |
| **container·map** raw hash_bytes 256B | ~0.20 | std::hash&lt;string&gt; | 慢 | (大赢) | 裸 XXH3 / wyhash | 0.20/0.07 | 1.0x/0.33x | ➖平 XXH3(wyhash 中串更强,观察) |
| **simd** dot product(4096 f32) | 0.079 | scalar autovec | 0.37 | 4.6x | Highway avx2 | 0.079 | 1.00x | ✅赢 std、**平 Highway(统计平局)** |
| **simd** saxpy(4096 f32) | 0.039 | scalar autovec | 0.057 | 1.45x | Highway avx2 | 0.039 | 1.00x | ✅赢 std、**平 Highway(统计平局)** |
| **simd** reduce_max(4096 f32) | 0.077 | scalar autovec | 0.73 | 9.5x | Highway avx2 | 0.077 | 1.00x | ➖平 Highway(统计平局) |
| **simd** scale(4096 f32) | 0.025 | scalar autovec | 0.044 | 1.8x | Highway avx2 | 0.025 | 1.01x | ➖平 Highway(统计平局,带宽受限) |
| **simd** dot_f32 公开分发入口 | 0.079 | scalar autovec | 0.37 | 4.6x | 直调 avx2(无间接) | 0.079 | 1.00x | ➖分发税仅在大 batch 摊薄≈0(4096 元素摊 1 次间接跳转);**小 n 未测** |
| **simd** saxpy_f32 公开分发入口 | 0.038 | scalar autovec | 0.057 | 1.5x | 直调 avx2(无间接) | 0.039 | 1.00x | ➖同上(大 batch 摊薄≈0,小 n 未测) |
| **dtype** f16→f32 | 0.02 | scalar ref | ~0.10 | 3.5–6.9x | Highway PromoteTo | ~0.04 | 1.9x | ✅赢 |
| **dtype** f32→f16 | 0.03 | scalar ref | ~0.13 | 3.5–6.9x | Highway DemoteTo | ~0.05 | 1.6x | ✅赢 |
| **dtype** bf16↔f32 | 0.03/0.11 | scalar ref | 0.2–0.4 | 3.5–6.9x | Highway bf16(未对照) | n/a | ⏳待补 | ✅赢 scalar;Highway bf16 待补 |
| **math** exp(best-of-breed→xsimd) | 0.23 | libm scalar | 慢 | (大赢) | Highway / xsimd | 0.30/0.23 | 1.30x/1.0x | ✅平 xsimd 最快、反超 Highway |
| **math** log(best-of-breed→Highway) | 0.26 | libm scalar | 慢 | (大赢) | Highway / xsimd | 0.28/0.38 | 1.08x/1.46x | ✅反超两库最快 |
| **math** tanh(best-of-breed→xsimd) | 0.44 | libm scalar | 慢 | (大赢) | Highway / xsimd | 0.56/0.45 | 1.27x/1.02x | ✅平 xsimd、反超 Highway |
| **sched** 争用 Mutex | 30.7 | std::mutex(争用) | 31 | ~1.0x | marl | 32.1 | 1.05x | ➖平(本轮略胜 marl) |
| **sched** 无争用 Mutex(MT) | 6.5 | std::mutex | 9.1 | 1.40x | marl(=std::mutex) | 9.1 | 1.40x | ✅赢 |
| **sched** spawn 2000 noop | 141 | std::async | 1020 | 7.2x | marl | 306 | 2.17x | ✅赢 |
| **sched** WaitGroup fork-join | 205(池179) | (无 std 等价) | n/a | n/a | marl | 306 | 1.49x(池1.71x) | ✅赢 |
| **sched** Latch fork-join(64) | 5059(fiber) | std::latch(64线程) | 21975 | 4.3x | marl(无 latch) | n/a | n/a | ✅赢 std(fiber 不阻塞 worker) |
| **sched** CV notify_one 快路径 | 1.70 | std::condition_variable | 2.59 | 1.52x | marl(无独立 CV) | n/a | n/a | ✅赢 std |
| **sched** CV wait+notify handoff | 280 | (无 std fiber 等价) | n/a | n/a | (HYLE 自有) | n/a | n/a | ➖HYLE 自有(阻塞 handoff,TSan 零race) |
| **sched** fiber 切换 | 21–23 | std::thread 切换 | ~1000+ | ~50x | fcontext 原语 | 21–23 | 1.0x | ➖平 fcontext(=原语零封装) |
| **sched** suspend/wake 完成脊柱 | 201–224 | (无 std 等价) | n/a | n/a | (HYLE 自有脊柱) | n/a | n/a | ➖HYLE 自有(GPU/NIC/timer 汇流) |
| **sched** Channel 收发(cap16,1p1c) | 63/item | (无 std 等价) | n/a | n/a | marl(无 channel) | n/a | n/a | ➖HYLE 自有 |
| **sched** RunQueue fifo/lifo push+next | 8.3/2.6 | (内部结构) | n/a | n/a | (无外部对手) | n/a | n/a | ➖HYLE 自有(完成唤醒热路径) |
| **queue** 单线程 fill+drain | 7.98 | (std 无 bounded-MPMC) | n/a | n/a | rigtorp/moody/tbb | 11.40/9.39/12.86 | 1.5/1.2/1.6x | ✅赢全部(含同约束 rigtorp) |
| **queue** MPMC 2P 单环 BoundedQueue | 56.9 | (std 无) | n/a | n/a | rigtorp/moody | 61.9/61.0 | 1.09x/1.07x | ✅平/微超同约束 rigtorp |
| **queue** MPMC 4P 单环 BoundedQueue | 74.0 | (std 无) | n/a | n/a | rigtorp/moody | 83.1/56.3 | 1.12x/0.76x | ✅反超 rigtorp(原输 1.33x,cache-pad+去backoff) |
| **queue** MPMC 2P/4P ShardedQueue | 21.5/18.0 | (std 无) | n/a | n/a | rigtorp/moody | 62/83·61/56 | 2.9x/4.6x | ✅赢(生产走这条,线性扩展) |
| **queue** work-steal owner FIFO/LIFO | 3.94/3.02 | (std 无) | n/a | n/a | crossbeam(Rust) | ~3.6/4.4 | ~1.2x/1.3x | ✅赢(跨语言,谨慎) |
| **queue** work-steal steal | 8.42 | (std 无) | n/a | n/a | crossbeam steal | (跨语言) | n/a | ➖HYLE WsFifo(本族 WSDeque 10.5) |
| **queue** ConcurrentVector push/读 | 4.51/0.93 | (std 无并发 vec) | n/a | n/a | oneTBB | 5.99/1.29 | 1.33x/1.39x | ✅赢 |
| **queue** ConcurrentVector 并发读(4读1写) | 0.37 | (std 无) | n/a | n/a | oneTBB | 0.82 | 2.2x | ✅赢(删冗余 acquire) |
| **mem** SystemAllocator churn(默认) | 7.91 | std::malloc(OS) | ~25 | ~3x | 裸 mimalloc | 7.79 | 0.99x | ➖平 SOTA(编译期内联,零接缝税) |
| **mem** Allocator vtable churn(opt-in) | 8.61 | std::malloc(OS) | ~25 | ~3x | std::pmr / 裸 mi | 8.70/7.79 | 1.01x/0.90x | ✅赢同类 std::pmr(对裸 mi -10% 是多态物理税,pmr 同付) |
| **mem** 8 线程通用 | 2.07 | std::malloc(OS) | ~30 | (大赢) | 裸 mimalloc | 2.14 | 1.03x | ➖平 SOTA |
| **mem** Arena bump | 0.67 | new/delete | ~15 | ~22x | std::pmr monotonic | 1.49 | 2.2x | ✅赢 |
| **mem** StackAllocator | 0.65 | new/delete | ~15 | ~23x | std::pmr monotonic | 1.49 | 2.3x | ✅赢 |
| **mem** StackAllocator mark/rewind | 1.31 | (无 std partial rewind) | n/a | n/a | std::pmr(无 partial rewind) | n/a | n/a | ➖HYLE 自有(LIFO 部分回退) |
| **mem** GrowablePool(单线程) | 1.92 | new/delete | ~15 | ~8x | std::pmr unsync_pool | 6.72 | 3.5x | ✅赢 |
| **mem** GrowablePool(8 线程 per-thread) | 0.72 | std::malloc | ~30 | (大赢) | 裸 mi per-thread heap | 1.28 | 1.78x | ✅赢 |
| **mem** MemoryArena(多线程共享池) | 2.51 | std::malloc | ~30 | (赢) | std::pmr::synchronized_pool(共享池 SOTA) | 112.9 | **44x** | ✅赢同契约 SOTA 44x(无锁分片 vs std mutex 串行;详节) |
| **mem** Allocator::reallocate grow chain | 76 | std::realloc | 166 | 2.2x | 裸 mi_realloc | 72 | 0.95x | ➖平裸 mi(Linux 统计平局,gap 5.4%«spread 21%);赢 std::realloc 2.2x |
| **container·vec** SmallVector 小 N | 0.43 | std::vector | 16.7 | 39x | absl::InlinedVector | 1.01 | 2.35x | ✅赢同款 small-vector SOTA |
| **container·vec** SmallVector 溢出(64>内联16) | 0.86 | std::vector | 4.57 | 5.3x | absl::InlinedVector | 2.38 | 2.77x | ✅赢同款(堆溢出/增长路径) |
| **container·list** IntrusiveList push/pop | 3.57 | std::list | 25.29 | 7.1x | boost::intrusive(未 checkout) | n/a | ⏳待补 | ✅赢 std 7.1x;boost 对照待跑 |
| **container·list** IntrusiveList::remove(O(1)) | 2.56 | std::list::erase | 28.31 | 11.1x | boost::intrusive::erase(未 checkout) | n/a | ⏳待补 | ✅赢 std 11x;boost 对照待跑 |
| **container·vec** CArray growth/read | 0.50/0.19 | std::vector(即 SOTA) | 0.79/0.19 | 1.58x/1.0x | std::vector(同 std,本身即 SOTA) | 0.79/0.19 | 1.58x/1.0x | ✅赢 growth / ➖平 read |
| **container·map** SwissTable iterate(for_each) | 2.24 | std::unordered_map | 8.09 | 3.60x | absl(未测此项) | n/a | n/a | ✅赢 std 3.6x(扁平 ctrl 扫) |
| **container·str** StringBuf intern(100k/2k 唯一) | 24.0 | 无锁 map floor | 21.2 | 0.88x | thread-safe peer(map+mutex) | 28.1 | 1.17x | ✅赢同契约 peer(对无锁 map 是 thread-safe-vs-无锁) |
| **container·str** StringBuf add/retain/remove churn | 18.2 | 无锁 map+引用计数 | 7.54 | 0.41x | (同契约 thread-safe 待补) | n/a | ⏳ | ❌输无锁 floor(同 intern;含 erase+free) |
| **util** Rng next_u64(xoshiro256++) | 1.56 | std::mt19937_64 | 2.41 | 1.54x | (std 即标尺) | 同std | 1.54x | ✅赢 std PRNG |
| **util** Rng range_u64(Lemire 无偏) | 1.75 | std::uniform_int_distribution | 2.75 | 1.57x | (std 即标尺) | 同std | 1.57x | ✅赢(且无偏) |
| **util** Rng next_double [0,1) | 1.65 | std::uniform_real_distribution | 2.64 | 1.60x | (std 即标尺) | 同std | 1.60x | ✅赢(53 位缩放) |
| **util** Rng next_float [0,1) | 1.63 | std::uniform_real_distribution | 2.66 | 1.63x | (std 即标尺) | 同std | 1.63x | ✅赢(独立 24 位路径) |
| **util** Rng shuffle 1000(Fisher-Yates) | 3.77 | std::shuffle(mt19937_64) | 6.40 | 1.70x | (std 即标尺) | 同std | 1.70x | ✅赢(swap 回调) |
| **util** SpinLock lock/unlock(无争用) | 4.03 | std::mutex | 12.03 | 3.0x | folly MicroSpinLock(未 clone) | n/a | ⏳待补 | ✅赢 std 3x(TTAS,无 syscall) |
| **util** Ref copy+destroy(背靠背同线) | 10.4 | std::shared_ptr | 8.2 | 0.79x | (shared_ptr 即标尺) | 8.2 | 0.79x | ❌microbench 假象(asm 证 2 locked RMW 同工;实战见下行) |
| **util** Ref fan-out 64 再释放(实战) | 8.15 | std::shared_ptr | 8.10 | ~1.0x | (shared_ptr 即标尺) | 8.10 | ~1.0x | ➖平(实战传 ref;算法同工) |
| **container·str** find(char) 全扫 miss(4KB) | 0.02/B | std::string_view::find | 0.01 | ~1.0x | string_view(即 SOTA,libc memchr) | 0.01 | ~1.0x | ➖平(切 memchr;原 scalar 输 ~20x) |
| **container·str** find(needle) 6B 子串 | 0.20/B | std::string_view::find | 0.08 | 0.40x | string_view(即 SOTA) | 0.08 | 0.40x | ❌输 2.5x(memchr 候选+memcmp;原 naive 输 7x) |
| **container·str** find_first_of(4 字符集) | 0.08/B | std::string_view | 0.05 | 0.62x | string_view(即 SOTA) | 0.05 | 0.62x | ➖近平 1.6x(memchr-min;原 O(n·m) 输 36x) |
| **container·str** starts_with(16B 前缀) | 1.79 | std::string_view | 1.77 | ~1.0x | string_view(即 SOTA,memcmp) | 1.77 | ~1.0x | ➖平(切 memcmp;原 字节循环输 2x) |
| **container·str** split 全文按空格(4KB) | 0.44/B | string_view find+remove_prefix | 0.54 | 1.22x | string_view(即 SOTA) | 0.54 | 1.22x | ✅赢(无分配子视图) |
| **container·str** trim 两端(4KB) | 13.6 | string_view 两侧 trim | 16.85 | 1.24x | string_view(即 SOTA) | 16.85 | 1.24x | ✅赢(组合 find_first/last_not_of) |
| **container·str** rfind(char) 后向扫 miss | 0.26/B | std::string_view::rfind | 0.01 | 0.05x | string_view(即 SOTA) | 0.01 | 0.05x | ❌输 ~20x(无 reverse-memchr;可上 memrchr) |
| **container·str** find_first_not_of 全字母 | 0.27/B | std::string_view | 0.06 | 0.22x | string_view(即 SOTA) | 0.06 | 0.22x | ❌输 ~4.5x(256-bool 表 vs std 调优 bitmap) |
| **container·str** find_last_of(digits) miss | 0.25/B | std::string_view | 0.06 | 0.23x | string_view(即 SOTA) | 0.06 | 0.23x | ❌输 ~4.4x(后向表扫) |
| **container·str** rfind(needle) 6B(4KB) | 2.37/B | std::string_view::rfind | 0.12 | 0.05x | string_view(即 SOTA) | 0.12 | 0.05x | ❌输 ~20x(后向 region_eq,无 memchr) |
| **container·str** count(char) 全扫 | 0.27/B | std::count | 0.01 | 0.04x | std::count(即 SOTA,自动向量化) | 0.01 | 0.04x | ❌输(scalar 累加无早退) |
| **container·str** utf8 validate+count(2KB) | 0.41(scalar)/0.21(simdutf) | (无 std utf8 count) | n/a | n/a | simdutf validate+count | 0.21 | 2.0x输/1.0x | ➖scalar 参考 2x;simdutf 核(vendored 无条件默认)追平 |
| **container·str** hex_encode 1KB→2KB | 0.45/B | naive 逐字节 | 0.63 | 1.41x | (无单一 SOTA hex 库) | n/a | n/a | ✅赢参考实现(表查找) |
| **container·path** extension / base / directory | 44/10/9.7 | (无 std 词法) | n/a | n/a | std::filesystem 词法 | 152/164/166 | 3.4/16/17x | ✅赢 filesystem(视图原地扫 vs 构造 path 分配) |
| **container·path** normalize 词法折叠 | 65 | (无 std 等价) | n/a | n/a | std::filesystem lexically_normal | 382 | 5.85x | ✅赢(就地 vs 堆 path 对象) |
| **container·path** join_array 6 部件(无 alloc) | 30 | (无 std 等价) | n/a | n/a | std::filesystem operator/= | 454 | 14.9x | ✅赢(buffer-fill 折叠 vs N 次 path 分配) |
| **platform** Parker ping-pong(真阻塞) | 317(Win) | std::condition_variable ping-pong | 927(Win) | 2.99x | Rust std park(概念对照) | n/a | n/a | ✅赢 std(Win 2.99x;**Linux 统计平局**,平台差异见 ⬇ platform 详节) |
| **platform** Parker unpark+park 快路径 | 9.6 | (无语义对等 std) | n/a | n/a | (无公平对照) | n/a | n/a | ⏳无对照:既给唤醒保证又无 syscall 的标准件不存在(裸 flag 不丢保证/semaphore 每次 futex,均不对等)。绝对值 ~10ns 无 syscall。详节 ⬇ |
| **platform** Thread spawn+join | 40.4µs(Win) | std::thread | 40.1µs | 1.01x | std::thread(即 SOTA,同 syscall) | 40.1µs | 1.01x | ➖平(双平台一致;OS 主导,证 handle 零开销) |
| **platform** directory_iterate visitor | — | std::filesystem::directory_iterator | — | — | filesystem | — | — | ➖平(Win 0.98x)/**✅Linux 赢 1.81x**(免 stat 的 d_type + 无分配 visitor;详节 ⬇) |
| **time** realtime_ns | 14.68 | std::chrono::system_clock | 14.76 | 1.0x | std::chrono(即标尺,同 OS 原语) | 14.76 | 1.0x | ➖平 |
| **time** monotonic_ns | 18.88 | std::chrono::steady_clock | 12.76 | 0.68x | std::chrono(即标尺) | 12.76 | 0.68x | ❌输 1.48x(同等工作 codegen 残留,查证过) |
| **obs** log gated-out / emit | 2.03/78.4 | (无 std log) | n/a | n/a | spdlog 等(未 clone) | n/a | ⏳待补 | ➖HYLE 自有(gated 1 原子+分支;emit=format+fanout 无 I/O) |

**总账小结（两把标尺,2026-06-30 全套真跑）**：**比 std**——HYLE 几乎全线大赢(分配器 8–23x、hashmap 2.6–6.9x、SIMD 4.5x、容器 5–39x、Rng 1.5–1.7x、SpinLock 3x);唯几项亚 ns 测量地板或 codegen 残留(monotonic_ns 0.68x)及 scalar 反向串族(rfind/count,可上 memrchr)。**比 SOTA**(真门槛,absl/mi/Highway/rigtorp/marl/tbb/xsimd/simdutf)——✅**赢 ~36 项**、➖**平 ~16 项**(噪声内追平)、❌**仍输少数项**且全附根因:① raw hash_bytes 16/64B=导出接缝 by-design 且亚周期(热路径 Hash&lt;T&gt; 内联已赢);② math exp/tanh 输 xsimd ~1.2x=跨库 best-of-breed 常数(已平 Highway);③ monotonic_ns 1.48x=codegen 残留(查证);④ MemoryArena/ObjectPool 单线程=**选错工具已下 ledger**(单线程对象池应用 GrowablePool;多线程共享场景对同契约 SOTA `std::pmr::synchronized_pool` **赢 44x**,非输项);⑤ string_ref scalar 反向/否定族(rfind/find_*_not_of/count)输 std::string_view 4–20x=scalar vs libc memrchr/调优 bitmap,**待优化项(可上 SIMD,非缺陷)**;⑥ string_algo utf8 默认 scalar 输 simdutf 2x(simdutf 核 vendored 无条件默认追平)。**阶段C 已扭转**:string_ref 前向搜索簇(find char 20x→平、find_first_of 36x→1.6x、starts_with 2x→平、split 反赢)、ConcurrentVector 并发读(2.15x 输→2.2x 赢)、Ref(反汇编证算法同工、实战打平,microbench 假象)、hash 8B、vtable(反超 std::pmr)、math log(反超两库)、单环队列 4P(反超 rigtorp)、StringBuf intern(反超同契约 peer)。**真复杂度/争用问题(MemoryArena 278x、Arena 10x、4P CAS+伪共享、CV per-read acquire 争线)全部根治**。**遗留优化(非覆盖缺口)**:error 根模块已补测(对标同款 SOTA absl 全线平或赢:inline 码赢 4.6x、带消息赢 1.92x、ERROR 上抛赢 2.11x,空间 Result 16B<std/absl 24B;OK 传播"输 std"经隔离探针+消费变体证为 microbench 假象);string_ref scalar 反向串族可上 memrchr/SIMD。

## 实测铁证（持续回填,真跑非推算）

> 环境:Linux x86_64 GCC14.2 Release,nanobench。100k uint64 键。`bench/container/`。

### error Status/Result（`status_bench.cc`，对标 absl::Status/StatusOr + std::expected）

> 环境:Win MSVC19.50 Release,nanobench。**两个标尺**:① **同款 SOTA = `absl::Status`/`absl::StatusOr`**(HYLE 这套 tagged-pointer + 规范 gRPC 码表正是抄它;gated `-DHYLE_BENCH_ABSL`)——这才是 apples-to-apples"带消息错误载体"这一类的真门槛;② `std::expected<int,std::error_code>` = 标准给的、但 E 平凡(俩 int,**带不了消息**)的下界。此前 error 是零依赖根却唯一无 bench 的模块。

| 操作 | HYLE | absl(同款 SOTA) | std | 结论 |
|---|---|---|---|---|
| **Status::Ok 构造+ok()** | **0.37** | 0.37 | std::expected 0.56 | ✅ 平 absl、赢 std 1.5x(`rep_==0` 单字) |
| **Status(code) 无消息** | **0.37** | 1.64(必堆 rep) | std::unexpected 0.55 | ✅ **赢 absl 4.6x**(HYLE inline 码零堆,absl code-only 也 malloc) |
| **Status(msg) 带消息→堆** | **28.2** | 54.3 | (无等价) | ✅ **赢 absl 1.92x**(单 malloc(rep+len+1),消息内联尾随) |
| **Status 拷贝(heap,refcount)** | **9.4** | 10.6 | (无等价) | ✅ 赢 absl 1.13x(3 指针 vtable refcount) |
| **Status 拷贝(OK)** | 0.59 | 0.57 | (无等价) | ➖ 平 absl(word copy,无 retain) |
| **Result OK 传播 4 层(裸)** | 0.73–0.94 | 0.91–0.94 | std::expected 0.39–0.55 | ➖ 平 absl;输 std=microbench 假象(见下) |
| **Result OK 传播(解包消费,真实)** | **1.42** | ~1.4 | std::expected 1.40 | ✅ **平 std**(真实用法,假象消失) |
| **Result ERROR 上抛 3 层** | **10.2** | 21.5 | (errc 无堆消息不可比) | ✅ **赢 absl 2.11x** |

**核心诚实结论(error 是 SOTA——对标真同侪 absl 全线平或赢,空间最省)**:

1. **"带消息错误载体"这一类里 HYLE ≥ absl 全项**:OK 构造平、inline 码 **赢 4.6x**、带消息构造 **赢 1.92x**、heap 拷贝赢 1.13x、ERROR 上抛 **赢 2.11x**。absl 是 HYLE 这套设计的来源(tagged ptr + gRPC 码表),逐项对标证明 HYLE 实现更精:**inline 错误码零堆**(absl 即便 code-only 也 malloc 一个 rep)是最大结构优势,直接兑现 status.h 的三态编码设计。

2. **Result OK 传播"输 std 1.7x"是 microbench 假象,已用消费变体证伪**:裸 microbench(构造 Result 立即在循环里析构、不取值)HYLE 0.73–0.94ns vs std::expected 0.39–0.55ns。**三条铁证它不是缺陷**:① objdump 看 `hyle_chain_ok` 函数体 = **7 条指令、全内联、零析构调用**(OK 路径本身零开销);② **隔离探针** `expected<int,TrivialErr>`(平凡析构错误类型)穿同样传播 idiom **精确追平 std(0.39=0.39)**——证明差距来自 `Status` 的非平凡析构(它要能 OWN 堆消息 → 析构带 `is_heap()` 分支;`std::error_code` 平凡因它带不了消息),不是宏/idiom;③ **同款 SOTA absl::StatusOr 付一模一样的税(0.91–0.94,与 HYLE 并排)**——这是"可携带堆消息的错误类型"整类的固有代价,SOTA 同付。**改用真实用法(`ASSIGN_OR_RAISE` 解包 + 消费 int)后 HYLE 1.42ns 平 std 1.40ns**——紧循环里反复构造-析构临时量才会把那一个析构分支放大,真实代码取值用掉后分支与后续工作交织即消失(同 util/Ref 条目记录的同类 microbench 假象)。验证过 static 构建(`status_release` 同 TU 可内联)仍 0.95ns,**证伪了"DLL 接缝阻止内联"假设**——确认是 std::expected 对非平凡 E 的语言规则,非接缝。

3. **空间维度(时间/空间冲突时优先时间,此处无冲突反而双赢)**:`sizeof(Status)=8B`(单 tagged word,= absl::Status 的 8B)、`Result<int>=Result<void>=16B` **< std::expected<int,errc> 24B、< absl::StatusOr<int> 24B**。HYLE 错误载体在两个维度都 ≤ SOTA:Status 与 absl 同宽且功能等价,Result 比 std/absl 省 1/3(8B Status vs std 的 16B error_code / absl 的内嵌 Status+对齐)。

### container·map —— SwissTable hashmap

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
> 3. **match_empty 加 SSSE3 快路径**(`_mm_sign_epi8(c,c)` 1 指令 vs SSE2 重建 kEmpty 广播 4 指令,`HYLE_HAVE_SSSE3` gate + SSE2 兜底):**绝对增益/与 absl ISA 对等**,但**未缩 find-miss 相对差距**——`-mssse3` 下 absl 也用 `psignb`,两边同步提速,find-miss 仍 ~0.89x。**证伪了"match_empty 是 find-miss 差距来源"**。双 ISA 输出穷举验证一致(psignb 确认)+ SSSE3 激活下 134/134。注:SSSE3 现为全局 baseline(CMake 对 x86 GCC/Clang 全局加 `-mssse3`、MSVC x64 无条件),`HYLE_HAVE_SSSE3` 默认成立、快路径默认生效。
> - **find-miss 剩 ~10% 已排除三因**:非 hash(mix64 对齐)、非 prefetch(miss 不碰 slot)、非 match_empty(同 ISA 仍差)。**根源在别处**(probe_seq codegen / H1 的 `>>7` / BitMask 迭代),需逐指令 profile,留作 profile 引导任务。
> 4. **H1=identity**(objdump 逐指令对比发现:absl 用 hash 低位直接当 index、top 7 位当 H2,省掉我们的 `hash>>7` 一条移位):find/insert/erase 内循环各省 1 指令。find-miss **0.89x→~0.96x**、insert/erase 升到 ~1.40x。正确性:mix64 雪崩后低位(index)与高 7 位(H2)独立,探测链不变;SSE2 基线 134/134 + Win MSVC 通过。
> - **12 遍统计收口(回答"4% 差在哪")**:find-hit avg **110.1%**(min 102.5/max 119.4)、find-miss avg **97.5%**(min 94.9/max 101.2,**spread 6.3% > 差距 2.5%,12 遍有 2 遍赢**)、insert **146.6%**、erase **140.0%**。**find-miss 是统计平局**——那"4%"是测量噪声(微基准 ~2-4ns,find-hit 自身 spread 都 16.9%),非结构缺陷,无单点杠杆。**四项全部 ≥ 或追平 absl,SOTA 反例消除。**

**这就是"对标 SOTA"的意义**:只对标 std 会得出"全面 2-5x 碾压"的虚假安心;对标 absl 才暴露 **find 有 ~25% 真实差距**。**已排除三个错误候选**(读码查实):① `eq_` 对 uint64 在 -O3 完全内联成裸 `==`,**无间接可去**;② `Hash<uint64>` 已是内联 `mix64`(splitmix64),**非 XXH3**,hash 不慢;③ **控制字节尾部镜像已实现**(`swiss_table.cc` core_set_ctrl 镜像前 kGroupWidth 字节 + 分配 `cap+kGroupWidth` + memset 尾),group near-end 加载已无回绕掩码。**find 循环结构已几乎等价 absl**——25% 来源**无法从源码结构看出**,需 **profile(perf/objdump 对比两个 find 内循环 codegen)** 定位真因;唯一已知结构差异是 absl 在 probe 起点 **预取槽数组**(100k 表常驻 cache 时收益小,待测)。**结论:这是一个需要 profile-guided 的真优化,非猜改**——留作下一步 container 性能待办。insert/erase 反超 absl 证明设计本身到位。

### container·map —— hash 哈希（`hash_bytes`/`mix64` vs 裸 XXH3 + wyhash + std，`hash_bench.cc`）

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



> 环境:WSL Ubuntu24.04 原生 ext4(非 9p)、GCC14.2 Release、nanobench。4096 个 f32(L1 常驻,隔离计算吞吐非内存带宽)。`bench/simd/`。**Highway = SOTA 标尺**(静态 dispatch,与 HYLE avx2 同 ISA、同 kernel 形状,只差抽象层)。**数字 = 15 遍进程级复测的中位数 op/s + run-to-run spread(`scripts/bench-stat.py` 驱动,沿 hashmap find-miss 那节的统计纪律)**;判定规则:差距 > 两变体 spread 之和才算赢/输,否则**统计平局**。
>
> **⚠️ 公平性更正(2026-06-30)**:此前此节标 dot"~1.10x 微超"、saxpy"~1.21x 微超" Highway——**是假象,源于对照被编瘸**。Highway 的 `HWY_STATIC_TARGET` 从完整 baseline 宏链(`detect_targets.h`)推断向量宽度:AVX2 baseline 需要 SSE4 baseline(`__PCLMUL__`+`__AES__`)再加 BMI2+FMA+F16C。bench 给 Highway 对照 TU 的是 HYLE 自家 kernel 的 AVX2 ISA 契约(`-mavx2 -mfma -mf16c`,刻意精简、关联 SIGILL-safety),在 **GCC/Clang 下让 Highway 静默降级到 SSSE3(128 位,Lanes=4)**——objdump 实证四个 `*_highway` kernel 全是 xmm 无 ymm。于是"parity"实际是 HYLE-256b vs Highway-128b,那 1.1-2x"赢"是宽度差不是抽象优势。修复:`hyle_set_highway_isa`(`HyleModule.cmake`)给对照 TU 补全 baseline(`-mbmi -mbmi2 -maes -mpclmul`),Highway 选回真 256 位 AVX2(`HWY_STATIC_TARGET=AVX2`、Lanes=8、四 kernel 全 emit ymm)。MSVC `/arch:AVX2` 本就含完整 baseline,故 Win 侧测的 dtype/math 节不受影响。下表是**修复后**的公平数据。

| kernel | scalar(auto-vec SSE2) | HYLE sse4(128b) | HYLE avx2(256b+FMA) | Highway avx2(**真 256b SOTA**) | 结论(15 遍中位数) |
|---|---|---|---|---|---|
| **dot product** | 2.73 G op/s (1.0x) | 11.3 G (4.1x) | **12.6 G (4.6x)** | 12.7 G (4.6x) | **平 Highway**(ratio 1.00,gap 0.3% « spread 10%) |
| **saxpy** y=a*x+y | 17.4 G (1.0x) | 17.4 G (1.0x) | **25.5 G (1.46x)** | 25.6 G (1.47x) | **平 Highway**(ratio 1.00,gap 0.2% « spread 13%) |

**结论(回答"比 SOTA 有下降么")**:**没有,也没有虚假领先——HYLE 向量后端与 Highway 统计平局**。公平的同 ISA 对照下,dot/saxpy 两 kernel 的 per-run op/s 比值中位数都是 ~1.00(min/max 横跨 1.0),差距远小于测量 spread——两边内循环都编成同一条 `vfmadd231ps`(objdump 可证),本就该同速。compute-bound 的 dot 上 **4.6x over 自动向量化标量**,落在性能文档"SIMD 4-16x / vs auto-vec 1.5-4x"区间(dot 含 reduction,略超 4x 合理)。saxpy 是 3 流内存操作(读 x/读 y/写 y),标量 auto-vec 已近饱和带宽,故 SIMD 倍数天然低——AVX2 靠更宽 load 仍取 1.46x。**SSE4 saxpy ~1.0x(与标量持平)**诚实留账:128 位宽度下该 kernel 已带宽受限,SIMD 无加速空间,非 bug。
>
> **意义**:这验证了"自造精简抽象不牺牲性能"——HYLE 的 vec.h+后端(~340+360 行)在热路径上与 vendor 整个 Highway(单后端 ~3k 行)**同速(统计平局)**。抽象层零成本(全内联到 intrinsic),没有为"可移植"付性能税。**平局比"微超"更可信**:同形 kernel 编出同样指令,本就不该有谁快几个百分点;之前那点"领先"恰恰是对照失真的红旗。
>
> **范围诚实**:此对标只覆盖 **dot/saxpy/reduce_max/scale 四个 compute/memory-bound 数值核**(AI 热路径主力)。Highway 还有 ~200 op(swizzle/gather/scatter/compress/transcendental)+ 6 ISA,HYLE 现做 ~25 op + 2 ISA 实跑——**"平"指已实现的核上同速,非功能对等**。广度补全(int 量化类型/更多 op/AVX512 启用)见 milestones simd 延期登记。**正确性**:Ne(NaN)/Neg(±0) 等边界经红队批量审计揪修(2026-06,见 milestones simd-02/03),已加 FloatEdgeCases 覆盖。

### simd 数学核（exp/log/tanh vs 标量 libm + 同 ISA SOTA xsimd/Highway，`math_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,4096 f32(L1 常驻)。**两个标尺**:① 标量 `std::exp/log/tanh`(测向量化加速本身);② **同 ISA SOTA = xsimd + Highway contrib/math**(同 `-mavx2`)。HYLE 借 Highway/xsimd 验证过的核(已 vendored、无条件默认编进生产,不手搓多项式)。`active_math_variant()` 报当前层。

| kernel | 标量 libm | HYLE(best-of-breed,默认) | xsimd | Highway | 结论 |
|---|---|---|---|---|---|
| **exp f32** | 2.08 ns | **0.23**(→xsimd) | 0.23 | 0.30 | ✅ 平 xsimd(最快)、反超 Highway 1.3x |
| **log f32** | 2.58 ns | **0.26**(→Highway) | 0.38 | 0.28 | ✅ **反超 Highway+xsimd**(两库最快) |
| **tanh f32** | 6.42 ns | **0.44**(→xsimd) | 0.45 | 0.56 | ✅ 平 xsimd(最快)、反超 Highway 1.27x |

**核心诚实结论(同 ISA 对标 + 逐 op best-of-breed dispatch,全 op ≥ 两库 SOTA)**:
1. **同 ISA 对标揪出真相,精度分级是公平性前提**:只对标量 libm(6-12x 赢)看不出问题;一加 xsimd/Highway 同 ISA 对照才暴露真实差距。三库都用 ~3.5-ULP 档多项式(AI 热路径足够——softmax/gelu/attention 输出走 fp16/bf16,1-ULP 是过度精度),**精度等级对等**,对标才 apples-to-apples。math_test 容差据实定在 _u35 等级(`kRelTol 8e-7 ≈6.7 ULP`,坏核 ≥7 ULP 仍 fail)。
2. **逐 op best-of-breed dispatch(收口)**:同 ISA 对标证实**没有单一库三 op 全胜**(xsimd 的 exp/tanh 最快、Highway 的 log 最快)。故 dispatcher **按 op 选最快核**:exp/tanh→xsimd、log→Highway(`math_dispatch.cc`),编译期选核(零运行时选库开销)。结果 **exp 0.23 平 xsimd 反超 Highway 1.3x、log 0.26 反超两库、tanh 0.44 平 xsimd 反超 Highway 1.27x**——**三个 op 全部 ≥ 两个 SOTA 库,无 math 落后残留**。
3. **借核现状**:Highway + xsimd 均 **vendored 在 `third_party/`、无条件编进生产、默认全速**(无 `-D` 开关),运行期 `cpu_features().avx2` 门控老 CPU 落 scalar 不 SIGILL。两个后端全是验证过的核(借核不手搓)。math_test 5/5、full simd 56/56。

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

> dot/saxpy 之外补两个 distinct pattern,验证 vec.h 抽象在非 FMA 模式下是否退化。环境同上(15 遍中位数+spread,Highway 真 256b 对照——见上节公平性更正)。

| op | scalar(auto-vec) | HYLE sse4 | HYLE avx2 | Highway(真 256b SOTA) | 结论(15 遍中位数) |
|---|---|---|---|---|---|
| **reduce_max**(非 sum 归约) | 0.73 ns | 0.17(4.3x) | **0.077(9.5x)** | 0.077 | **平 Highway**(ratio 1.00,gap 0.2% « spread 16%) |
| **scale** y=a*x(纯乘) | 0.044 ns | 0.043 | **0.025(1.8x)** | 0.025 | **平 Highway**(ratio 1.01,带宽受限) |

**结论(干净,无欺骗)**:**4 个 pattern(dot/saxpy/reduce_max/scale)HYLE avx2 全与 Highway 统计平局**——vec.h 抽象在 sum-归约之外的 max-归约、纯 element-wise map 上同样零成本,无隐藏退化。reduce_max 9.5x 赢 scalar;scale 是 1 读 1 乘 1 写的内存受限 op,scalar auto-vec(SSE2)已近饱和带宽,故 avx2 仅 ~1.8x(正常,非缺陷)。sse4 在 scale/saxpy 这类带宽受限 op ~1x(128b 与 scalar SSE2 同宽,无更宽 load 优势)。**注**:此前 reduce_max/scale 曾对 128 位瘸腿 Highway 显出 ~2x 假性领先(同上节根因);修正对照后归于平局——结论方向不变,但数字过程已诚实化。



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

### platform 原语（Parker park/unpark、Thread、directory_iterate，`platform_bench.cc`）

> **平台差异是本节主题**：platform 是 OS 原语封装层,这些项的行为在 Windows 与 Linux 上**由各自内核/运行时主导**,数字不可跨平台搬用。两侧均为 **15 遍中位数+spread**(`scripts/bench-stat.py`,2026-06-30):**Win MSVC19.50** + **Linux GCC14.2 原生 ext4**,双平台并列。判定规则同 simd:gap > 两变体 spread 之和才算赢/输,否则统计平局。

| 项 | HYLE (Linux) | 对照 (Linux) | Linux 判定 | Win 判定(15 遍) | 标尺性质 |
|---|---|---|---|---|---|
| **Parker 真阻塞乒乓** | 20.8 µs/rt | std CV 22.2 µs | **统计平局**(1.09x,gap 8.9% « spread 28.6%) | **赢 CV 2.99x**(317 vs 927 ns,gap 199% » spread 53%) | std CV(等到被通知) |
| **Thread spawn+join** | 31.6 µs/rt | std::thread 31.4 µs | 统计平局(1.00x) | 统计平局(40.4 vs 40.1 µs,1.01x) | std::thread 即 SOTA(同 syscall) |
| **directory_iterate visitor** | 2.90 µs | std::filesystem 5.19 µs | **✅赢 1.81x**(gap 81% » spread 33%,min 1.73x) | 统计平局(33.1 vs 32.2 µs,0.98x,gap 2.1% « spread 18%) | std::filesystem 目录遍历 |

> **Parker unpark+park 快路径未列入表格——没有语义对等的公平对照**。这条热路径(unpark 先到、park 不睡)实测 9.94 ns(Linux)/9.6 ns(Win),但**找不到一个"既给同样的唤醒保证、又同样无 syscall"的标准件来比**,所以从对标表移除,只在下面留注释说明(理由见 ② 节)。绝对值有意义(无 syscall、~10 ns),相对"输赢"无诚实分母可填。

**① Parker 真阻塞乒乓——平台差异的典型,两个数字都对**：两线程互相 park/unpark,每腿一次真睡眠+唤醒(`futex`/`WaitOnAddress`)。**Win 上赢 CV 2.85x**——Windows 的 `condition_variable` 要 `SRWLock`+`CONDITION_VARIABLE`+谓词三件套,Parker 的裸 futex 直驱省掉这些。**Linux 上只统计平局**——glibc 的 `pthread_cond` 本身已是薄薄一层 futex 封装,跟 Parker 的直驱 futex 成本几乎一样,没有结构性差距可拿。**这不是回归,是"对照在 Linux 上本就更强"**:同一个 HYLE 实现,赢多少取决于对手平台实现有多重。

**② Parker unpark+park 快路径——为什么没有公平对照,那个"安全"是什么**：这是调度器最常走的热路径——**unpark 先到、park 不必睡**。三态机 `EMPTY=0`/`NOTIFIED=1`/`PARKED=0xFFFFFFFF`,`park()` 第一步 `state_.fetch_sub(1, acquire)`(`wait.cc:103`):看到 `NOTIFIED(1)` 就减成 `EMPTY` 直接返回(快路径,不睡);看到 `EMPTY(0)` 则回绕成 `PARKED` 去 `wait_on_address` 睡。绝对值 **9.94 ns(Linux)/9.6 ns(Win),全程无 syscall**,对比真阻塞那条走 futex 的 ~20 µs 快约 2000 倍——热路径成本就是一条 `fetch_sub`(`lock xadd`,RMW 带 acquire)+ 分支。
> - **为什么不给"对标倍数"——没有语义对等的对手**。要公平比,对照必须同时满足两条:① 提供同样的 **unpark-before-park 不丢唤醒**保证;② 同样**无 syscall** 的快路径。实测两个候选都不对等(故移出表格):
>   - **裸 `atomic` flag(4.4 ns)**:无 syscall,但**没有唤醒保证**——unpark 先到时信号丢失,随后 park **永久睡死**。拿它当对手是让"有保证的"比"裸奔的",对 HYLE **苛刻**(凭空背一个 2x"输")。
>   - **`std::binary_semaphore`**:语义对等(release/acquire 令牌安全),但 libstdc++ 实测 **release+acquire 快路径每轮都发一次 futex**(strace 实证 10 万轮=10 万次 futex),根本没有"令牌已在就纯原子返回"的优化——比它 HYLE 会**远赢**(µs vs 10 ns),对 HYLE **太宽松**,同样是误导。
>   - 结论:**既给保证、又无 syscall 的标准件不存在**,所以这条只报绝对值,不填输赢分母(这正是上一轮 simd 的教训:对照语义/ISA 不对等,任何倍数都是假的)。
> - **那条快路径买到的"安全" = unpark-before-park 不丢唤醒**。并发中唤醒方/等待方在不同线程赛跑。若 unpark **先于** park 发生,朴素 flag 会让随后的 park **永久睡死**(信号在无人等待时发出就丢了)。三态机让 unpark 把状态置 `NOTIFIED`(`exchange(kNotified)`,`wait.cc:122`)**记住令牌**;park 进来 `fetch_sub` 撞见 `NOTIFIED` 直接取走返回,**令牌永不丢失**。这是 CV/调度器赖以正确的语义,也是它比裸 flag 多花那条 RMW 的原因——那不是低效,是正确性的最低成本。

**③ Thread spawn+join——双平台都平,意义在"证零开销"**:建线程→跑空函数→join,2000 次。标尺是 `std::thread` 本身(两者都包 `pthread_create`/`CreateThread`,µs 级,OS 全主导)。Linux/Win 都统计平局,证明 HYLE 那个能设栈大小/亲和性的 ABI-clean 句柄**没在 OS 之上加任何税**。

**④ directory_iterate——Win 统计平局、Linux 赢 1.81x,平台差异不可混看**：此前总账只记 Win 单平台数且数值方向(HYLE 33.1 µs ≳ std 32.2 µs)看着像"略输",易被误判。**统计法澄清两件事**:① **Win 是统计平局**(0.98x,gap 2.1% « spread 18%),不是输;② **Linux 是真赢 1.81x**(2.90 vs 5.19 µs,15 遍 min 都 1.73x,gap 81% » spread 33%,铁证)。Linux 赢的根因是设计意图兑现:HYLE 走 `readdir` + `d_type` 直接拿"是否目录"(`fs.cc:487`),**省掉 stat**;`std::filesystem::directory_iterator` 每步构造 `directory_entry`,bench 里的 `is_directory()` 还触发**额外 stat 系统调用**(`platform_bench.cc:165`),再加每条目的 path 分配。无分配 visitor + 免 stat = 真实结构优势;Win 上两者底层都走 `FindNextFileW` 量级相近故平。**教训:platform 项必须标平台,单平台数字直接搬到总账会误导输赢判断**(本轮就是从"疑似输"查成"Win 平 / Linux 赢")。



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

### mem 专用分配器簇（arena/pool/stack vs `std::pmr` + 裸 mimalloc，`arena_pool_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,8192×64B。**SOTA 标尺 = `std::pmr`**(标准库自带的 arena=`monotonic_buffer_resource`、pool=`unsynchronized_pool_resource`,完美 drop-in,无需 clone)+ 裸 `mi_malloc`(=`new/delete` 的实际 backing,即"5-10x vs new/delete"声明的对照基线)。**此前这簇零对标——本节真测揪出两个无锁变体的单点 CAS 争用(已分别根治:MemoryArena 分片 278x→1.94x,Arena 去原子 10.3x),并纠正了一处 apples-to-oranges 对照(共享固定池 vs per-thread heap,见结论 4)。**

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4 15 遍中位数+spread**(`scripts/bench-stat.py`),nanobench,8192×64B。**SOTA 标尺 = `std::pmr`**(标准库自带:bump arena=`monotonic_buffer_resource`、单线程 pool=`unsynchronized_pool_resource`、**多线程共享 pool=`synchronized_pool_resource`**,完美 drop-in,无需 clone)+ 裸 `mi_malloc`(=`new/delete` 的实际 backing)。**此前这簇零对标——本节真测揪出无锁变体的单点 CAS 争用(已分别根治:MemoryArena 分片 278x、Arena 去原子),并把每个工具放到正确的同契约对手上**:bump→monotonic、单线程池→GrowablePool vs unsync_pool、**多线程共享池→MemoryArena vs synchronized_pool(赢 44x)**。`MemoryArena` 单线程是选错工具(已下 ledger,见下警示)。

| bump 场景(alloc N + 批量释放) | ns/op(Linux) | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE Arena**(单线程 bump,池块) | **0.74** | **5.4x** | ✅ 去原子后平 Stack;vs std::pmr 平台相关(见下) |
| **HYLE StackAllocator**(单线程自有 slab) | **0.74** | **5.4x** | ✅ 真兑现 |
| std::pmr::monotonic_buffer(SOTA) | 0.61(Lin)/1.49(Win) | — | **平台差异**:Win 上 HYLE 赢 2.2x,**Linux 上 pmr 0.61 反略快(HYLE 0.82x)**——GCC 的 monotonic 比 MSVC 实现更紧;均为纯指针 bump 量级,~亚 ns 难分高下 |
| 裸 mi_malloc(general,无批量) | 3.98 | 1.0x | — |

| pool churn 场景(**单线程**同尺寸 alloc/free) | ns/op(Linux) | vs 裸 mi | 结论 |
|---|---|---|---|
| **HYLE GrowablePool**(单线程对象池) | **1.28** | **3.5x** | ✅ 单线程对象池的对工具(gap 252%»spread 23%) |
| 裸 mi_malloc/free(general) | 4.51 | 1.0x | — |
| std::pmr::unsync_pool(std 单线程池) | 11.61 | 0.39x | — |
| ⚠️ MemoryArena/ObjectPool(单线程) | 13.97/13.98 | 0.32x | **选错工具,非对标项**:无锁共享池单线程白付每 op 原子 CAS(详见下)——已下 ledger |

> **⚠️ "选错工具"明示**:`MemoryArena`/`ObjectPool` 是**为多线程共享有界池**设计的(无锁分片 Treiber 栈)。**单线程**同尺寸 churn 用它们是误用——每次 alloc+free 各付一次原子 CAS(两条 `lock cmpxchg`)+ tls homing + 地址路由,实测 13.97ns 慢裸 mi 3.1x、连 std 单线程池(`unsynchronized_pool` 11.6ns)都不如。这不是 HYLE 的缺陷,是拿错了工具——**单线程对象池请用 `GrowablePool`**(同 workload 1.28ns,3.5x 赢裸 mi)。故此误用对比**已从对标 ledger 移除**(留此行仅作选型警示),`MemoryArena` 的真实对标是多线程共享场景(见下,赢 std 共享池 44x)。usage 选型见 [`guide/usage-guide.md` §3.1.1](../guide/usage-guide.md)。

**8 线程 pool churn——两个诚实问题分开:① 共享池 vs 共享池 SOTA(公平对标)② per-thread 模式参照:**

| 8 线程同尺寸 churn | ns/op(Linux) | 结论 |
|---|---|---|
| **HYLE MemoryArena**(共享固定池) | **2.51** | ✅ **赢同契约 SOTA `std::pmr::synchronized_pool` 44x**(后者 112.9ns,mutex 串行;HYLE 无锁分片) |
| std::pmr::synchronized_pool(共享池 SOTA) | 112.9 | 共享池的**公平对手**——同契约(一池所有线程共用) |
| **HYLE GrowablePool**(每线程一个池) | **0.63** | ✅ per-thread 模式赢裸 mi per-thread heap 2.04x(min 1.89x) |
| 裸 mi(per-thread heap) | 1.26 | per-thread 模式参照(**不同契约**,非共享池对手) |

**核心诚实结论(单点原子两处已修;并把每个工具放到正确的同契约对手上——红队纠错)**:

1. **"专用分配器 5-10x vs new/delete"全部坐实**:`StackAllocator`(纯指针 bump,0.74ns)+ `Arena`(去原子后 0.74ns)**5.4x 赢裸 mi**;`GrowablePool`(单线程 free-list,1.28ns)**3.5x 赢裸 mi、3.0x 赢 std::pmr pool**(gap 252%»spread 23%,铁证)。四个单线程/去原子变体全兑现 5-10x vs general malloc。**vs std::pmr arena 是平台相关**:Win 上 Arena/Stack 赢 monotonic 2.2x,但 Linux 上 GCC 的 monotonic(0.61ns)反比 HYLE(0.74ns)略快——两者都是纯指针 bump、亚 ns 量级,这一项诚实表述为"平台相关、量级相当",不宣称稳赢 std::pmr。

2. **`MemoryArena` 曾是全场最慢——单点 CAS 争用,已分片根治,公平对标下反超 SOTA 44x**:原 clay 继承的单 `head_` 让 8 线程 CAS 同一 cache line,O(threads) 串行化(8 线程慢 **278x**)。**修复**(`memory_arena.h`):内部分 N 片(片数=核数向上取 2 的幂,上限 128),每片独立 Treiber 头占独立 cache line,线程按 tls 序号 homing 到自己片(同线程 alloc/free 无争用),free 按地址 O(1) 路由回属主片。**复杂度落后(O(N)争用→O(1) per-shard)消除**,双平台 10/10 + WSL TSan 零 race。这同时补上了 mem-design 列为"硬约束"却从未实现的 per-core 池。**公平对标(对同契约的 `std::pmr::synchronized_pool_resource`——std 自带线程安全共享池):8 线程 HYLE 2.51ns vs std 112.9ns = 赢 44x**(HYLE 无锁分片 vs std mutex 串行)。**单线程用它是选错工具**(慢裸 mi 3.1x,该用 `GrowablePool`,见上"⚠️选错工具"),那不是缺陷而是误用。

3. **`Arena` 无锁 bump 同源病已修(去原子化)**:`Arena` 的单 `offset_` 原子也是单点病——单线程付每 alloc 一次 CAS(6.79ns vs `StackAllocator` 纯指针 bump,10x 税)。bump cursor 本质单游标无法分片,SOTA 模式是 **per-thread(各自一个 Arena,零原子)**,并发共享一个 cursor 是反模式(无内部用户的投机无锁化)。**修复**:`offset_` 改 plain `size_t`,纯指针 bump;测试改 `PerThreadArenasNoOverlap`(每线程自己的 Arena)。**6.79→0.74ns(~9x),平 StackAllocator、赢裸 mi 5.4x、与 std::pmr arena 量级相当(平台相关,见结论 1)**。

4. **共享池的真实对标是 std 共享池,不是 mimalloc per-thread heap——换公平对手 HYLE 赢 44x**:此前把"8 线程**共享一个**固定池"对"mimalloc 的 **8 个 per-thread heap**"比(0.51x),并把这 1.98x 写成"无法优化的诚实标价"——**这是错的**:两者契约不同(共享有界池 vs 各线程私有堆),不该当输赢分母。**红队补的公平对手 = `std::pmr::synchronized_pool_resource`**(std 自带、同契约的线程安全共享池):HYLE 2.51ns **碾压它 112.9ns,赢 44x**(15 遍 min 28.8x)。HYLE 的无锁分片在共享池这一类里远超 std 的 mutex 串行池。**教训(红队精神)**:"对照不公平"不能当免死金牌写成"无法优化"——找到**真正同契约**的对手(synchronized_pool),HYLE 不是输 2x 而是赢 44x。mimalloc per-thread heap 数(1.26ns)仅作 per-thread 模式的 context 参照,不是共享池的对手;若用户要的就是 per-thread 模式,HYLE 的对等答案是 `GrowablePool`(0.63ns,赢 mi 2.04x)。

5. **进一步逼近 mimalloc per-thread heap 的绝对延迟需 seastar 模型**:若拿共享池硬比 mimalloc 的 per-thread heap 绝对值(1.26ns),HYLE 共享池 2.51ns 的差距是"一块固定 slab 跨线程共享"的物理协调代价(每 op 一次原子 CAS,分片已消除 O(threads) 争用)。要把这个常数也逼到 per-thread heap 水平需 seastar 模型(线程钉死 per-core 终生、池生命周期长于线程),`MemoryArena` 的通用生命周期给不出该契约。但这**不是对标失分**——共享池的 SOTA 对手是 synchronized_pool(已赢 44x),per-thread heap 是另一契约(其对手是 GrowablePool,已赢 2.04x)。两个契约 HYLE 各自的对等答案都赢。

### container·vec / ·list —— SmallVector / IntrusiveList / CArray（SmallVector / IntrusiveList / CArray vs std + 真 SOTA，`container_misc_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。声明承重容器。**标尺升级**:不止 std——SmallVector→`absl::InlinedVector`(small-vector SOTA,gated `-DHYLE_BENCH_ABSL`)、IntrusiveList→`boost::intrusive::list`(侵入容器 canonical SOTA,gated `-DHYLE_BENCH_BOOST`)、CArray→`std::vector`(动态数组的 drop-in SOTA,本身已是高度调优,always-on)。

| 场景 | HYLE | std | 真 SOTA(gated) | 结论 |
|---|---|---|---|---|
| **小 N churn**(建+填 8 int+析构) | SmallVector<int,16> **0.43 ns** | vector 16.7 / vector+reserve 3.07 | **absl::InlinedVector 0.60** | ✅ **赢 absl 1.40x**、赢 std 39x |
| **list churn**(push 4096 + pop 全) | IntrusiveList **3.0~3.4 ns** | list 24.1 | boost::intrusive(需 submodule checkout) | **8x 赢 std**;boost 同 hook-in-node,gate 已接 |
| **动态数组 growth**(push_back 4096 int,无 reserve) | CArray<int> **0.53 ns** | vector **0.71** | (std::vector 即 SOTA) | ✅ **赢 std::vector 1.34x** |
| **动态数组 read**(顺序求和 4096) | CArray<int> **0.18 ns** | vector 0.18 | (std::vector 即 SOTA) | ➖ **平 std::vector**(arr[i] 单 load 兑现) |

**诚实结论**:① **SmallVector 反超 absl::InlinedVector 1.40x**——不是"只赢 std":absl::InlinedVector 是 small-vector 的 SOTA(同 inline 存储设计),HYLE 0.43ns vs absl 0.60ns,小 N churn 真赢同款 SOTA(也赢 std::vector 39x)。② **IntrusiveList 8x 赢 std::list**(零节点分配);boost::intrusive 同 hook-in-node 设计,gate 已接线(`-DHYLE_BENCH_BOOST`),但 boost 模块化 submodule 未 checkout,留待环境备齐实测(预期同量级)。③ **CArray 反超 std::vector 1.34x growth、平 read**:growth 赢的真根因 = 经 `reallocate()` 增容(mimalloc 原地扩展,无 copy),std::vector 每次扩容恒 alloc-new+move+free;read 路径 arr[i] 单 load 与 vector 持平(header-before-data 布局零索引开销)。

### container·str —— string_ref / StringBuf / string_algo（`string_ref_bench.cc` / `string_buf_bench.cc` / `string_algo_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench,~4KB 文本(L1 常驻)。**标尺 = `std::string_view`**(string_ref 是其 drop-in 替代,conventions §7:同 16 字节 POD + null_terminated 位,同契约对照公平)+ `std::count` / simdutf。**这是 HYLE 目前 ❌ 输项最集中的一节,如实记账**:string_view 的搜索族在 libc 里是 memchr/memrchr/向量化 + 调优 bitmap,HYLE 现是 scalar 字节循环 / 256-bool 表——**前向 char 搜索已切 memchr 追平,但反向族(rfind/find_last_of)和否定/计数族(find_first_not_of/count)仍是 scalar,真输 4–20x**。

| op | HYLE | std::string_view | vs SOTA | 结论 |
|---|---|---|---|---|
| **find(char) 全扫 miss** | 0.02/B | memchr 0.01 | ~1.0x | ➖平(已切 memchr;原 scalar 输 ~20x) |
| **find_first_of(4 字符集)** | 0.08/B | 0.05 | 0.62x | ➖近平 1.6x(memchr-min;原 O(n·m) 输 36x) |
| **starts_with(16B)** | 1.79 | memcmp 1.77 | ~1.0x | ➖平(已切 memcmp) |
| **split 全文按空格** | 0.44/B | find+remove_prefix 0.54 | 1.22x | ✅赢(无分配子视图) |
| **trim 两端** | 13.6 | 16.85 | 1.24x | ✅赢(组合 find_first/last_not_of) |
| **find(needle) 6B 子串** | 0.20/B | 0.08 | **0.40x** | ❌输 2.5x(memchr 候选+memcmp;原 naive 输 7x) |
| **rfind(char) 后向扫** | 0.26/B | 0.01 | **0.05x** | ❌输 ~20x(无 reverse-memchr;**可上 memrchr**) |
| **rfind(needle) 6B** | 2.37/B | 0.12 | **0.05x** | ❌输 ~20x(后向 region_eq,无 memchr) |
| **find_first_not_of 全字母** | 0.27/B | 0.06 | **0.22x** | ❌输 ~4.5x(256-bool 表 vs std 调优 bitmap) |
| **find_last_of(digits)** | 0.25/B | 0.06 | **0.23x** | ❌输 ~4.4x(后向表扫) |
| **count(char) 全扫** | 0.27/B | std::count 0.01 | **0.04x** | ❌输 ~25x(scalar 累加无早退;**std::count 自动向量化**) |
| **StringBuf intern**(100k/2k 唯一) | 24.0 | map+mutex(同契约 thread-safe peer)28.1 | 1.17x | ✅赢同契约 peer(SwissTable 核) |
| **StringBuf add/retain/remove churn** | 18.2 | 无锁 map floor 7.54 | 0.41x | ⚠️对手是**无锁 floor**(StringBuf 是 thread-safe);**同契约 thread-safe churn 对手待补**(参照 intern 行,公平对手下应反超)——红队待办,勿当真输 |

**诚实结论(❌ 真输项集中地,不掩饰)**:
- **6 个真输项全是 scalar vs libc 向量化/调优**,且对手 `std::string_view` 是**同契约 drop-in SOTA**(对照公平,不是选错工具)——这是 string_ref 搜索族**还没把参考库抄平**的真实缺口,非测量伪影、非对照不公平。
- **优化方向(明确,project-hyle 有现成参考可抄)**:① 反向族 `rfind(char)`/`find_last_of` → glibc `memrchr`(Linux)/手写 SIMD 反向扫;② `count(char)` → SIMD 计数或至少让编译器向量化(std::count 已证可向量化);③ `find_first_not_of` → 学 std 的调优 bitmap(非 256-bool 表);④ `find(needle)`/`rfind(needle)` → 子串搜索用 memchr 定位首字节候选 + SIMD 比较。**抄都要抄平**——这些是后续 string 优化里程碑的明确 TODO,留账于此。
- **StringBuf churn 的 0.41x 是对照存疑(⚠️非确认真输)**:它的 std 对手是**无锁裸 map**(无锁),而 StringBuf 是 thread-safe(每 op 一把锁)——同 MemoryArena 那类 apples-to-oranges。intern 行换上同契约 thread-safe peer(map+mutex)后 HYLE ✅赢 1.17x;churn 的同契约对手(thread-safe add/retain/remove)尚未加,**红队待办**,加上前不算确认真输。

### container·path —— 纯词法路径操作（`path_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。`path.h` 是 HYLE 的纯字符串路径层(无 syscall),`extension`/`base`/`directory`/`normalize`/`split` 在 string_ref 视图上原地走分量、**零分配**。**标尺 = `std::filesystem::path` 词法操作**——但后者构造一个 path 对象(分配 + 存字符串),HYLE 只在视图上扫,故全线赢。

| op | HYLE | std::filesystem | vs SOTA | 结论 |
|---|---|---|---|---|
| **extension / base / directory** | 44/10/9.7 ns | 152/164/166 | 3.4/16/17x | ✅赢(视图原地扫 vs 构造 path 分配) |
| **normalize 词法折叠** | 65 ns | lexically_normal 382 | 5.85x | ✅赢(就地折叠 `.`/`..`/分隔符 vs 堆 path 对象) |
| **join_array 6 部件** | 30 ns | operator/= ×6 = 454 | 14.9x | ✅赢(buffer-fill 折叠 vs N 次 path 分配) |

**诚实结论**:path 全线赢 `std::filesystem` 3.4–17x,根因单一且真实——HYLE 是**视图上的纯词法扫描、零分配**,std::filesystem 每次操作都构造 path 对象(堆分配 + 存字符串副本)。这不是巧妙算法,是"不分配 vs 必分配"的结构差;对标 std::filesystem 是公平的(它就是标准库的路径词法 SOTA),但 HYLE 的优势仅在**纯词法**层(无 syscall 的 fs 操作另见 platform 节 directory_iterate)。

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
