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
| **simd** dot product(4096 f32) | 0.024 | scalar autovec | 0.37 | 15x | Highway contrib/dot(真 SOTA 4-acc) | 0.024 | 1.00x | ✅**4 累加器展开后 0.079→0.024(3.3x)平真 SOTA**;旧"平 Highway 0.079"是两个都单累加器的假平局(对照用了 Highway 教学 kernel 非 contrib/dot),见 simd 详节 |
| **simd** saxpy(4096 f32) | 0.026 | scalar autovec | 0.057 | 2.2x | Highway avx2 | 0.025 | 1.01x | ✅赢 std、平 Highway(element-wise 带宽受限,多累加器无益,未改) |
| **simd** reduce_max(4096 f32) | 0.077 | scalar autovec | 0.73 | 9.5x | Highway avx2 | 0.077 | 1.00x | ➖平 Highway(bench-only,非公开 op;单累加器,若上架公开 op 可同 dot 展开) |
| **simd** scale(4096 f32) | 0.025 | scalar autovec | 0.044 | 1.8x | Highway avx2 | 0.025 | 1.01x | ➖平 Highway(bench-only 非公开 op;带宽受限) |
| **simd** dot_f32 公开分发入口 | 0.024 | scalar autovec | 0.37 | 15x | Highway contrib/dot(真 SOTA) | 0.024 | 1.00x | ✅出厂 dispatch 入口 4 累加器,平真 SOTA(分发税大 batch 摊薄≈0) |
| **simd** saxpy_f32 公开分发入口 | 0.026 | scalar autovec | 0.057 | 2.2x | 直调 avx2(无间接) | 0.025 | 1.01x | ➖同上(大 batch 摊薄≈0,小 n 未测) |
| **dtype** f16→f32 | 0.025 | scalar ref | 0.66 | 26x | Highway PromoteTo | 0.048 | Win 1.88x/Lin 平 | ✅Win 赢 Highway/Linux 统计平局(平台差异) |
| **dtype** f32→f16 | 0.030 | scalar ref | 2.07 | 25–80x | Highway DemoteTo | 0.048 | Win 1.61x/Lin 1.90x | ✅赢 Highway 双平台(唯一双平台稳赢) |
| **dtype** bf16→f32 | 0.030 | scalar ref | 0.10 | Win 3.5x/Lin 平 | Highway PromoteTo | 0.050 | Win 1.87x/Lin 平 | ✅Win 赢 Highway/Linux 平(纯位运算 GCC 也向量化) |
| **dtype** f32→bf16 | 0.095(Lin)/0.098(Win) | scalar ref | 0.18/0.70 | 1.9–7x | Highway OrderedDemote2To(真 SOTA 2-vec) | 0.097/0.099 | 1.02x/1.01x | ➖平真 SOTA(改 2-input 生产形状后 Lin 0.104→0.095 ~1.09x,平 OrderedDemote2To;旧对标的 DemoteTo 便利版是 8→8 朴素形状,HYLE 现反超它 0.107/0.117);**AVX-512-BF16 native `vcvtneps2bf16` 已上架**(待真机,见 AVX-512 节①) |
| **math** exp(best-of-breed→xsimd) | 0.23 | libm scalar | 2.06 | 9x | Highway / xsimd | 0.30/0.22 | 1.30x/1.0x | ✅平 xsimd 最快、反超 Highway(双平台一致) |
| **math** log(best-of-breed→Highway) | 0.27 | libm scalar | 2.0–2.6 | 4–8x | Highway / xsimd | 0.27/0.37 | 1.0x/1.45x | ✅平 Highway、反超 xsimd(修产品核 128b bug 后 Linux 同样成立) |
| **math** tanh(best-of-breed→xsimd) | 0.44 | libm scalar | 6.3–8.7 | 14–20x | Highway / xsimd | 0.54/0.45 | 1.22x/1.0x | ✅平 xsimd、反超 Highway(双平台一致) |
| **platform** Parker ping-pong(真阻塞) | 317(Win) | std::condition_variable ping-pong | 927(Win) | 2.99x | Rust std park(概念对照) | n/a | n/a | ✅赢 std(Win 2.99x;**Linux 统计平局**,平台差异见 ⬇ platform 详节) |
| **platform** Thread spawn+join | 40.4µs(Win) | std::thread | 40.1µs | 1.01x | std::thread(即 SOTA,同 syscall) | 40.1µs | 1.01x | ➖平(双平台一致;OS 主导,证 handle 零开销) |
| **platform** directory_iterate visitor | — | std::filesystem::directory_iterator | — | — | filesystem | — | — | ➖平(Win 0.98x)/**✅Linux 赢 1.81x**(免 stat 的 d_type + 无分配 visitor;详节 ⬇) |
| **time** realtime_ns | 14.68 | std::chrono::system_clock | 14.76 | 1.0x | std::chrono(即标尺,同 OS 原语) | 14.76 | 1.0x | ➖平 |
| **time** monotonic_ns | 12.4(Win)/12.6(Lin) | std::chrono::steady_clock | 12.7/12.9 | Win 1.03x/Lin 平 | std::chrono(即标尺,同 OS 原语) | 12.7/12.9 | Win 1.03x/Lin 1.0x | ➖平双平台(Win 曾输 1.48x,补常见频率特判后追平;详节③) |
| **time** fast_monotonic_ns(opt-in rdtsc) | 7.4(Win)/6.8(Lin) | HYLE monotonic_ns | 12.4/12.6 | 1.67x/2.0x | (std 无 opt-in TSC 钟) | n/a | n/a | ✅快 1.7-2.0x(跳过 OS 调用壳;非默认,可能跳变;leaf 0x15 只优化启动校准+乘子精度,**不改读延迟**,详节③) |
| **time** fast_realtime_ns(opt-in coarse) | 1.8(Win)/3.1(Lin) | HYLE realtime_ns | 13.5/13.0 | 7.6x/4.2x | (std 无 opt-in coarse 钟) | n/a | n/a | ✅快 4-7.6x(coarse 时间源,~1-15ms 分辨率;日志时间戳用,详节③) |
| **util** Rng next_u64(xoshiro256++) | 1.56 | std::mt19937_64 | 2.41 | 1.54x | (std 即标尺) | 同std | 1.54x | ✅赢 std PRNG |
| **util** Rng range_u64(真 Lemire 无除法) | 1.46(microbench)/shuffle见下 | std::uniform_int_distribution | 2.75 | — | absl/numpy/rust rand(真 Lemire) | 同法 | ~1.0x | ✅改真 Lemire(128 位乘替 2 idiv);常量界 microbench 编译折除法故 1.46ns 平,**真实运行时界见 shuffle 3.2x** |
| **util** Rng next_double [0,1) | 1.65 | std::uniform_real_distribution | 2.64 | 1.60x | (std 即标尺) | 同std | 1.60x | ✅赢(53 位缩放) |
| **util** Rng next_float [0,1) | 1.63 | std::uniform_real_distribution | 2.66 | 1.63x | (std 即标尺) | 同std | 1.63x | ✅赢(独立 24 位路径) |
| **util** Rng shuffle 1000(Fisher-Yates+Lemire) | 1.18(Lin)/1.17(Win) | std::shuffle(mt19937_64) | 1.21(Lin)/6.42(Win) | Lin平/Win 5.5x | (std 即标尺) | 同std | — | ✅**改 Lemire 后 3.82→1.18ns(3.2x)**:运行时变界暴露旧 modulo 的 2 idiv,真 Lemire 消除;Win 赢 std 5.5x/Lin 平(GCC std::shuffle 比 MSVC 强) |
| **util** SpinLock lock/unlock(无争用) | 4.03 | std::mutex | 12.03 | 3.0x | folly MicroSpinLock | ~同 | ~1.0x | ✅赢 std 3x(TTAS+pause+cache-pad);**无争用快路径已核对 = folly 形状**(单原子 RMW + release store),⏳结案;有争用无 sleep-fallback 是刻意设计(短临界区专用) |
| **util** Ref copy+destroy(fan-out 64,实战) | 8.15 | std::shared_ptr | 8.10 | ~1.0x | (shared_ptr 即标尺) | 8.10 | ~1.0x | ➖平(算法同工:各 2 locked RMW;实战传 ref,incref/decref 突发分离)。注:紧循环背靠背 copy+destroy 的 microbench(HYLE 10.4 vs std 8.2)是**不公平对照已下表**——同 cache line 两原子串行化放大、无法与实际工作重叠,asm 证同工;真实用法即本行。引导见 usage-guide §3.7 |
| **obs** log gated-out / emit | 1.4–2.0 / 54.7(Lin)/78–85(Win) | (无 std log) | n/a | n/a | spdlog 同步(未 clone,量级参考) | ~同档 | ➖ | ➖HYLE 自有;**emit ~90% 是 std::format(裸 52ns),锁+fanout 仅 Lin 6/Win 13ns**——produce 路径已贴 std::format 下限,SOTA 对标本质=同步 format 成本,无结构劣势(详节) |
| **mem** SystemAllocator churn(默认) | 7.91 | std::malloc(OS) | ~25 | ~3x | 裸 mimalloc | 7.79 | 0.99x | ➖平 SOTA(编译期内联,零接缝税) |
| **mem** Allocator vtable churn(opt-in) | 8.61 | std::malloc(OS) | ~25 | ~3x | std::pmr / 裸 mi | 8.70/7.79 | 1.01x/0.90x | ✅赢同类 std::pmr(对裸 mi -10% 是多态物理税,pmr 同付) |
| **mem** 8 线程通用 | 2.07 | std::malloc(OS) | ~30 | (大赢) | 裸 mimalloc | 2.14 | 1.03x | ➖平 SOTA |
| **mem** Arena bump | 0.34 | new/delete | ~15 | ~44x | std::pmr monotonic | 0.74 | 2.2x | ✅赢(pointer-style bump 后 0.67→0.34,vs 裸 mi 4.0ns~12x;见 AVX-512 表④) |
| **mem** StackAllocator | 0.34 | new/delete | ~15 | ~44x | std::pmr monotonic | 0.74 | 2.2x | ✅赢(pointer-style bump 后 0.65→0.34) |
| **mem** StackAllocator mark/rewind | 0.43 | (无 std partial rewind) | n/a | n/a | std::pmr(无 partial rewind) | n/a | n/a | ➖HYLE 自有(LIFO 部分回退;pointer-bump 后 1.31→0.43) |
| **mem** GrowablePool(单线程) | 1.92 | new/delete | ~15 | ~8x | std::pmr unsync_pool | 6.72 | 3.5x | ✅赢 |
| **mem** GrowablePool(8 线程 per-thread) | 0.72 | std::malloc | ~30 | (大赢) | 裸 mi per-thread heap | 1.28 | 1.78x | ✅赢 |
| **mem** MemoryArena(多线程共享池) | 2.5–2.7 | std::malloc | ~30 | (赢) | oneTBB scalable_allocator(真强 SOTA) | ~7–9 | ~3x | ✅赢 tbb 真 scalable ~3x(无锁分片 vs tbb per-thread cache+共享后备);对 std::pmr::synchronized_pool(66–113ns mutex 串行弱池)25–44x——**后者非天花板,tbb 才是**(2026-07 补强对手,详节) |
| **mem** Allocator::reallocate grow chain | 76 | std::realloc | 166 | 2.2x | 裸 mi_realloc | 72 | 0.95x | ➖平裸 mi(Linux 统计平局,gap 5.4%«spread 21%);赢 std::realloc 2.2x |
| **container·hash** Hash&lt;T&gt; 16B POD 内联 | 0.43 | std::hash | ~0.9 | ~2x | wyhash | 0.94 | 2.2x | ✅赢(2026-07 改 abseil 单 128 位乘 mix64_2:0.59→0.43,vs wyhash 1.6x→2.2x) |
| **container·hash** mix64 整数 | 0.44 | std::hash | 1.50 | 3.4x | wyhash64 | 0.57 | 1.27x | ✅赢(已 SOTA=abseil 同款单乘折叠) |
| **container·hash** hash_bytes 8B(inline mix,非 XXH3) | 0.23/B | std::hash&lt;string&gt; | 0.29 | 1.3x | 裸 XXH3-small / wyhash | 0.21/0.30 | ~平/赢 wyhash | ➖**≤16B 走 inline mix 非 XXH3**(两算法比,非"输 XXH3");赢 wyhash-small、近平 XXH3-small |
| **container·hash** hash_bytes 16B(inline mix,非 XXH3) | 0.15/B | std::hash&lt;string&gt; | 0.35 | 2.3x | 裸 XXH3-small / wyhash | 0.12/0.15 | ~平 wyhash | ➖同上(inline mix vs XXH3-small,不同算法;热路径 Hash&lt;T&gt; 见上行赢 wyhash 2.2x) |
| **container·hash** hash_bytes 64B | 0.055/B | std::hash&lt;string&gt; | 0.45 | 8x | 裸 XXH3(同 hasher) | 0.043 | **~1.0x(同链接模型)** | ➖**平裸 XXH3**:同为内联时 `hash_bytes_inline` ≈ 裸 XXH3(都全内联,同 hasher 同算法)。旧 "0.78x" 是**不公平对比**——HYLE 默认出线核(隔离 xxhash.h,DLL 可调)vs 裸 XXH3 全内联,不同链接模型;那 ~1ns 是 seam 非算法(固定量,64B 占 37% 可见/256B 摊平),短串热点用 opt-in `hash_bytes_inline` 即消。map key 走 ≤16B inline 本不付 |
| **container·hash** hash_bytes 256B(best-of-breed→wyhash) | 0.029/B | std::hash&lt;string&gt; | 0.073 | 2.5x | 裸 XXH3 / wyhash | 0.075/0.029 | **XXH3 2.6x / 平 wyhash** | ✅**best-of-breed 按长度选 wyhash**(>128B),256B 0.082→0.029=2.8x,精确追平 raw wyhash;详节⑥ |
| **container·map** find-hit(100k u64) | 2.21 | std::unordered_map | 10.47 | 4.74x | absl flat_hash_map | 3.03 | 1.30x | ✅赢 |
| **container·map** find-miss(100k) | 4.36 | std::unordered_map | 11.57 | 2.65x | absl flat_hash_map | 4.86 | 1.09x | ✅赢(噪声内略胜) |
| **container·map** insert(100k fresh) | 11.58 | std::unordered_map | 80.07 | 6.92x | absl flat_hash_map | 24.53 | 2.30x | ✅赢 |
| **container·map** erase(100k) | 18.01 | std::unordered_map | 93.10 | 5.17x | absl flat_hash_map | 34.60 | 1.83x | ✅赢 |
| **container·map** iterate(100k visit) | 0.96 | std::unordered_map | 8.37 | 8.7x | absl flat_hash_map | 1.94 | 2.0x | ✅赢 std 8.7x、**反超 absl 2.0x**(2026-07 for_each 改 SIMD 组扫:2.32→0.96=2.4x,补 absl 对标揪出旧逐字节扫曾输 absl 1.2x) |
| **container·map** SwissTable iterate(for_each) | 0.96 | std::unordered_map | 8.37 | 8.7x | absl flat_hash_map | 1.94 | 2.0x | ✅同上(match_full 组扫 + for_each 模板无迭代器对象开销) |
| **container·vec** SmallVector 小 N | 0.41 | std::vector | 16.8 | 41x | absl::InlinedVector | 0.84–1.0 | 2.0–2.4x | ✅赢同款 small-vector SOTA(absl 侧 spread 大取范围) |
| **container·vec** SmallVector 溢出(64>内联16) | 0.88 | std::vector | 4.70 | 5.3x | absl::InlinedVector | 2.40 | 2.7x | ✅赢同款(堆溢出/增长路径) |
| **container·vec** CArray growth/read | 0.51/0.18 | std::vector(即 SOTA) | 0.75/0.18 | 1.48x/1.0x | std::vector(同 std,本身即 SOTA) | 0.75/0.18 | 1.48x/1.0x | ✅赢 growth(2026-07 grow 真改 reallocate;赢因 mimalloc 分配器质量非"原地无copy",详节纠错)/ ➖平 read |
| **container·list** IntrusiveList push/pop | 3.15 | std::list | 25.0 | 7.1x | boost::intrusive::list | 2.77 | ~1.0x 平 | ✅赢 std 7.1x、**平 boost SOTA**(2026-07 checkout boost submodule 实测:ratio 0.88 gap12%«spread44%=统计平局) |
| **container·list** IntrusiveList::remove(O(1)) | 3.13 | std::list::erase | 55.7 | 11.1x | boost::intrusive::list::erase | 3.15 | ~1.0x 平 | ✅赢 std、**平 boost SOTA**(任意位置 O(1) 摘除,两者同 circular-sentinel 形状) |
| **container·str** StringBuf intern(100k/2k 唯一) | 24.0 | 无锁 map floor | 21.2 | 0.88x | thread-safe peer(map+mutex) | 28.1 | 1.17x | ✅赢同契约 peer(对无锁 map 是 thread-safe-vs-无锁) |
| **container·str** StringBuf add/retain/remove churn | 17.9 | thread-safe map+mutex(同契约 peer) | 22.3 | 1.24x | thread-safe peer(map+mutex re-find/op) | 22.3 | 1.24x | ✅赢同契约 peer 1.24x(min 1.17 保底;对无锁 floor 7.5 是 apples-to-oranges) |
| **container·str** find(char) 全扫 miss(4KB) | 0.02/B | std::string_view::find | 0.01 | ~1.0x | string_view(即 SOTA,libc memchr) | 0.01 | ~1.0x | ➖平(切 memchr;原 scalar 输 ~20x) |
| **container·str** find(needle) 6B 子串 | 0.08/B | std::string_view::find | 0.08 | ~1.0x | string_view(即 SOTA,two-way 向量化) | 0.08 | ~1.0x | ➖平(转发 std;自研 memchr 候选曾输 2.5x) |
| **container·str** find_first_of(4 字符集) | 0.05/B | std::string_view | 0.05 | ~1.0x | string_view(即 SOTA) | 0.05 | ~1.0x | ➖平(转发 std;自研 memchr-min 曾 0.62x) |
| **container·str** starts_with(16B 前缀) | 1.79 | std::string_view | 1.77 | ~1.0x | string_view(即 SOTA,memcmp) | 1.77 | ~1.0x | ➖平(切 memcmp;原 字节循环输 2x) |
| **container·str** split 全文按空格(4KB) | 0.44/B | string_view find+remove_prefix | 0.54 | 1.22x | string_view(即 SOTA) | 0.54 | 1.22x | ✅赢(无分配子视图) |
| **container·str** trim 两端(4KB) | 13.6 | string_view 两侧 trim | 16.85 | 1.24x | string_view(即 SOTA) | 16.85 | 1.24x | ✅赢(组合 find_first/last_not_of) |
| **container·str** rfind(char) 后向扫 miss | 0.012/B | std::string_view::rfind | 0.012 | ~1.0x | string_view(即 SOTA) | 0.012 | ~1.0x | ➖平(AVX2 后向扫,统计平局 gap2%«spread11%;原 scalar 输 ~20x) |
| **container·str** find_first_not_of 全字母 | 0.06/B | std::string_view | 0.06 | ~1.0x | string_view(即 SOTA) | 0.06 | ~1.0x | ➖平(转发 std;自研 bool 表/PCMPESTRI 曾输 4.5x/0.31x) |
| **container·str** find_last_of(digits) miss | 0.06/B | std::string_view | 0.06 | ~1.0x | string_view(即 SOTA) | 0.06 | ~1.0x | ➖平(转发 std;自研曾输 4.4x) |
| **container·str** rfind(needle) 6B(4KB) | 0.12/B | std::string_view::rfind | 0.12 | ~1.0x | string_view(即 SOTA,two-way 向量化) | 0.12 | ~1.0x | ➖平(转发 std;自研 region_eq 曾输 ~4x) |
| **container·str** count(char) 全扫 | 0.008/B | std::count | 0.008 | ~1.0x | std::count(即 SOTA,自动向量化) | 0.008 | ~1.0x | ➖平(AVX2 4累加器 psadbw,统计平局 gap7%«spread10%;原 scalar 输 ~27x) |
| **container·str** utf8 validate+count(2KB) | 0.41(scalar)/0.21(simdutf) | (无 std utf8 count) | n/a | n/a | simdutf validate+count | 0.21 | 2.0x输/1.0x | ➖scalar 参考 2x;simdutf 核(vendored 无条件默认)追平 |
| **container·str** hex_encode 1KB→2KB | 0.45/B | naive 逐字节 | 0.63 | 1.41x | (无单一 SOTA hex 库) | n/a | n/a | ✅赢参考实现(表查找) |
| **container·path** extension / base / directory | 44/10/9.7 | (无 std 词法) | n/a | n/a | std::filesystem 词法 | 152/164/166 | 3.4/16/17x | ✅赢 filesystem(视图原地扫 vs 构造 path 分配) |
| **container·path** normalize 词法折叠 | 65 | (无 std 等价) | n/a | n/a | std::filesystem lexically_normal | 382 | 5.85x | ✅赢(就地 vs 堆 path 对象) |
| **container·path** join_array 6 部件(无 alloc) | 30 | (无 std 等价) | n/a | n/a | std::filesystem operator/= | 454 | 14.9x | ✅赢(buffer-fill 折叠 vs N 次 path 分配) |
| **queue** 单线程 fill+drain | 7.98 | (std 无 bounded-MPMC) | n/a | n/a | rigtorp/moody/tbb | 11.40/9.39/12.86 | 1.5/1.2/1.6x | ✅赢全部(含同约束 rigtorp) |
| **queue** MPMC 2P 单环 BoundedQueue | 56.9 | (std 无) | n/a | n/a | rigtorp/moody | 61.9/61.0 | 1.09x/1.07x | ✅平/微超同约束 rigtorp |
| **queue** MPMC 4P 单环 BoundedQueue | 74.0 | (std 无) | n/a | n/a | rigtorp/moody | 83.1/56.3 | 1.12x/0.76x | ✅反超 rigtorp(原输 1.33x,cache-pad+去backoff) |
| **queue** MPMC 2P/4P ShardedQueue | 21.5/18.0 | (std 无) | n/a | n/a | rigtorp/moody | 62/83·61/56 | 2.9x/4.6x | ✅赢(生产走这条,线性扩展) |
| **queue** work-steal owner FIFO/LIFO | 3.94/3.02 | (std 无) | n/a | n/a | crossbeam(Rust) | ~3.6/4.4 | ~1.2x/1.3x | ✅赢(跨语言,谨慎) |
| **queue** work-steal steal | 8.42 | (std 无) | n/a | n/a | crossbeam steal | (跨语言) | n/a | ➖HYLE WsFifo(本族 WSDeque 10.5) |
| **queue** ConcurrentVector push/读 | 4.51/0.93 | (std 无并发 vec) | n/a | n/a | oneTBB | 5.99/1.29 | 1.33x/1.39x | ✅赢 |
| **queue** ConcurrentVector 并发读(4读1写) | 0.37 | (std 无) | n/a | n/a | oneTBB | 0.82 | 2.2x | ✅赢(删冗余 acquire) |
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

**总账小结（两把标尺,2026-06-30 全套真跑；simd dot 2026-07 更新）**：**比 std**——HYLE 几乎全线大赢(分配器 8–23x、hashmap 2.6–6.9x、SIMD dot 15x/saxpy 2.2x、容器 5–39x、Rng 1.5–1.7x、SpinLock 3x);唯几项亚 ns 测量地板或 Win-only codegen 残留(monotonic_ns Win 0.68x/Linux 平)及 scalar 反向串族(rfind/count,可上 memrchr)。**比 SOTA**(真门槛,absl/mi/Highway/rigtorp/marl/tbb/xsimd/simdutf)——✅**赢 ~36 项**、➖**平 ~16 项**(噪声内追平)、❌**仍输少数项**且全附根因:① raw hash_bytes 64B=导出接缝 by-design(同算法 XXH3 vs 裸 XXH3 ~1.28x,>16B 出线核不可内联;≤16B 走 inline mix 无 seam,改 abseil 单乘后 16B key 赢 wyhash 2.2x);② math exp/tanh 输 xsimd ~1.2x=跨库 best-of-breed 常数(已平 Highway);③ monotonic_ns **Win-only 输 1.48x、Linux 平**(QPC split 算术 vs MSVC steady_clock,已查证非接缝非算法;POSIX clock_gettime 与 glibc steady_clock 同码故 Linux 平);④ MemoryArena/ObjectPool 单线程=**选错工具已下 ledger**(单线程对象池应用 GrowablePool;多线程共享场景对真强 SOTA `oneTBB scalable_allocator` **赢 ~3x**(对 std 弱池 synchronized_pool 25–44x,后者非天花板),非输项);⑤ string_ref scalar 反向/否定族(rfind/find_*_not_of/count)输 std::string_view 4–20x=scalar vs libc memrchr/调优 bitmap,**待优化项(可上 SIMD,非缺陷)**;⑥ string_algo utf8 默认 scalar 输 simdutf 2x(simdutf 核 vendored 无条件默认追平)。**阶段C 已扭转**:string_ref 前向搜索簇(find char 20x→平、find_first_of 36x→1.6x、starts_with 2x→平、split 反赢)、ConcurrentVector 并发读(2.15x 输→2.2x 赢)、Ref(反汇编证算法同工、实战打平,microbench 假象)、hash 8B、vtable(反超 std::pmr)、math log(反超两库)、单环队列 4P(反超 rigtorp)、StringBuf intern(反超同契约 peer)。**真复杂度/争用问题(MemoryArena 278x、Arena 10x、4P CAS+伪共享、CV per-read acquire 争线)全部根治**。**遗留优化(非覆盖缺口)**:error 根模块已补测(对标同款 SOTA absl 全线平或赢:inline 码赢 4.6x、带消息赢 1.92x、ERROR 上抛赢 2.11x,空间 Result 16B<std/absl 24B;OK 传播"输 std"经隔离探针+消费变体证为 microbench 假象);string_ref scalar 反向串族可上 memrchr/SIMD。

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
| find-miss | **2.26x** 赢 | **≈1.0x 平/微胜**(最新 15 遍 4.41 vs 4.86,1.12x,gap«spread) | **追平 SOTA(见下 profile 收口)** |
| insert | **5.3x** 赢 | **~1.40x 赢** | **超 SOTA** |
| erase | **3.5x** 赢 | **~1.39x 赢** | **超 SOTA** |
| iterate(for_each) | **8.7x** 赢 | **2.0x 赢**(0.96 vs 1.94) | **反超 SOTA(2026-07 组扫)** |

> **2026-07 补 iterate 对标 + 组扫优化(揪出真输并反转)**:此前 iterate 只对 std(赢 3.6x)、absl 标"未测此项"——补 absl 对标**揪出旧 for_each 逐字节扫(load ctrl_[i]+每 slot 分支)输 absl 迭代器 1.2x**(2.32 vs 1.94)。改**组扫**:每 kGroup 字节一次 SIMD `match_full`(SSE2 `movemask^0xFFFF`)得占用 lane 位掩码,只迭代 set bit(无每 slot 分支),= absl 迭代器形状。**iterate 2.32→0.96ns=2.4x,反超 absl 2.0x、赢 std 8.7x**——组扫 + `for_each(fn)` 模板无迭代器对象/校验开销双重优势(absl range-for 每步构造迭代器)。加 `Group::match_full()` + `ForEachVisitsExactlyLiveSet` 测试(空/插/删 tombstone/growth,守卫组扫;此前 for_each 零直接测试)。双平台 container 148/148。**教训同 dtype/dot:"只赢 std 未对 SOTA"会掩盖真输——补对标才暴露,补上就反超。**
> **find-miss profile 待办收口(2026-07)**:历史留的"find-miss ~10% 待 profile"已过时——最新 15 遍 SwissTable 4.41 vs absl 4.86 = **HYLE 微胜 1.12x(gap«spread,统计平局偏赢)**,那 ~10% 差距在后续 mix64/H1/组 baseline 优化后已消失,HYLE 现不落后 find-miss,profile 待办关闭。
> **hash 层提升是否反哺 map(2026-07 评估)**:SwissTable 默认 `Hash<K>`,所以 hash 层优化按 **key 类型**自动流入。① **整数/指针 key**(如本表 uint64 bench):走 `mix64`,本轮**未动**(它早是 abseil 单乘折叠、已 SOTA)——故 uint64 map 数字不变,符合预期,非遗漏。② **9-16B POD 复合 key**(如 `{u64,u64}`):走 `hash_small`,本轮 `mix64_2`(单乘替两串行乘)使其 hash 1.38x,map 用此类 key 时 find/insert 直接受益。③ **长 string key**(>128B):走 best-of-breed wyhash,map 用长串 key 时受益。**结论**:反哺是"按 key 类型条件生效"的——最热的整数 key 本就 SOTA 无可反哺,复合/长串 key 才吃到本轮红利;无需额外改 map 代码(hash 是模板参数,自动继承)。

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

> 环境:Win MSVC19.50 Release,nanobench,15 遍中位数。三组:HYLE / 裸 XXH3+wyhash(SOTA) / std::hash。XXH3 vendored 无条件,wyhash gated `-DHYLE_BENCH_WYHASH`。**2026-07 红队修 bench 公平性 + 改 abseil 单乘**——数字与结论已重做,见下。
>
> **⚠️ bench 公平性更正(2026-07,红队 hash)**:旧 bench/ledger 有三处误导,已修:① **≤16B 行标签 "XXH3 wrapper" 错**——`hash_bytes(len≤16)` 是 `HYLE_FORCE_INLINE` 且短路走 **inline mix(mix64 折叠)根本不调 XXH3、不跨导出 seam**;所以 8B/16B 是 HYLE-inline-mix vs **完整 XXH3**(两个不同算法),旧结论"输裸 XXH3 0.86x/0.67x"是拿 inline-mix 比 XXH3-small,非真输。**真正的 seam/wrapper 税只在 64B/256B**(那里才是同算法 XXH3 vs XXH3)。② 旧 bench **单固定串重复 hash 百万次**→数据常驻 L1、长度分派分支完美预测,小串读出亚周期(0.03ns 假地板);已改**每长度轮转 256 个随机串**(smhasher 式),8B 从假地板 0.03→真实 0.23ns。③ `batch(len)`=ns/byte,小串固定开销主导,per-byte 低≠算法快。

| 字节哈希(ns/op per byte,15 遍) | HYLE | 裸 XXH3 | wyhash | std | 结论(算法归属) |
|---|---|---|---|---|---|
| 8B | 0.23 | 0.21 | 0.30 | 0.29 | **inline mix**(非 XXH3):赢 wyhash-small、近平 XXH3-small |
| 16B | 0.15 | 0.12 | 0.15 | 0.35 | **inline mix**(非 XXH3):平 wyhash、近平 XXH3-small |
| 64B | 0.055 | 0.043 | 0.052 | 0.45 | **真 XXH3 wrapper**:同算法 vs 裸 XXH3 ~1.28x 接缝税(出线核不可内联) |
| 256B | 0.082 | 0.075 | **0.029** | 0.63 | 平裸 XXH3(接缝);**wyhash 碾 XXH3 2.6x=SOTA 观察** |

| 整数哈希(4096 key) | HYLE mix64 | wyhash64 | std::hash | 结论 |
|---|---|---|---|---|
| ns/op | **0.44** | 0.57(赢 1.27x) | 1.50(赢 3.4x) | ✅ mix64 已 SOTA(= abseil 同款单 128 位乘折叠) |

| 小 POD key(16B 结构,4096,真热路径) | HYLE Hash<T> | wyhash | 结论 |
|---|---|---|---|
| ns/op(改 mix64_2 前→后) | 0.595 **→ 0.43** | 0.94 | ✅ **改 abseil 单乘后 1.38x,vs wyhash 从 1.6x→2.2x** |

**结论(bench 已诚实化 + 9-16B 改 abseil 单乘 + SOTA 观察)**:
1. **9-16B key 改 abseil 单 128 位乘(mix64_2)——16B POD key 热路径 1.38x**:HYLE 原对 9-16B 用**两次串行 mix64**(各一个 128 位乘);abseil `CombineContiguousImpl9to16` 只用**一次** `Mix(a,b)=fold(a*b)`(两 word 进单乘的两操作数)。加 `detail::mix64_2` 照此改,16B POD key **0.595→0.43ns(1.38x)、vs wyhash 赢 2.2x**(原 1.6x)。短依赖链(1 乘替 2 串行乘)的真收益;质量经 `SmallStructInlineQuality`(40000 distinct + 每 bit 雪崩≥12)+ StringView 10k 无碰撞门(10/10 Hash 测试双平台过)。已知 abseil 文档的 1/2^64 corner(操作数恰 0,仅 16B 可达,XOR 常数+重叠字节 practically 不可达)。
2. **≤16B "输 XXH3" 是伪命题(两算法比)**:纠旧结论——`hash_bytes(≤16)` 走 inline mix 非 XXH3,拿它比完整 XXH3 不是同物;正确对手是 wyhash-small(HYLE 赢/平)。旧文档"接缝代价"归因也错:≤16B 全内联无 seam,**真接缝税只在 >16B 的 XXH3 出线核**。**64B 那 0.78x 已实测隔离出根因(2026-07)**:① `XXH3_64bits_withSeed(64B,0)` vs plain `XXH3_64bits(64B)` 同 TU 全内联时 **1.01x 同速**——排除"withSeed 是不同算法"这个候选;② 同一 XXH3 经 HYLE 导出核 `hash_bytes_xxh3`(`HYLE_CONTAINER_API` 不可内联,DLL/`.so` hop):inline 1.9ns → 导出 2.9ns = **+1ns 纯调用边界税**(函数调用 + 跨 seam 不内联 + 共享库 PLT/import 间接)。这是**故意的设计取舍**——出线核让 7490 行 xxhash.h 不进任何 consumer TU(编译期 + ABI 洁净),换约 1ns 固定开销;map 定长 key 走 ≤16B inline 路径不付此税。**这 ~1ns 是固定量,随串长摊薄**:64B(单 hash ~2.5ns)时占比大→可见 ~1.28x;**256B(单 hash ~12ns)时已摊薄到统计平局**(Linux 15 遍 ratio1.02 gap2%«spread13%,导出核甚至因独立 TU 的 codegen 偶尔反超)。旧 ledger "256B 0.91x 输" 是 Win 单遍噪声,实为平局。**短串 seam 逃生口已提供(2026-07,应用户要求)**:opt-in header `hyle/container/hash_inline.h` 的 `hash_bytes_inline()` 把 hasher 内联进调用者 TU(无出线调用=无 seam),bit-identical `hash_bytes`(`InlineVariantBitIdenticalToOutOfLine` 测试验全长度带+多 seed)。**代价=该 TU 吃 xxhash.h/wyhash.h 的编译期开销**,故 opt-in per-TU、默认 `hash.h` 绝不 include 它(隔离不破)。只对短串(17~128B)热点有意义:≤16B 默认已内联无 seam、长串已摊平。**两个设计决策(记以防复问)**:① **不违反 C-ABI 分层**——`hash_bytes_inline` 是 header-inline、非导出符号(无 `HYLE_CONTAINER_API`),不参与 C-ABI;默认 `hash_bytes` 仍是导出/DLL 可调/隔离 xxhash.h 的稳定符号。inline 变体要求**源码内嵌/静态链接**(把 xxhash.h 编进调用 TU),纯 DLL/C-FFI 消费者用默认 `hash_bytes`。这是分层(默认 C-ABI 洁净 + opt-in 源码内嵌换极致),非破例。② **只做一个全长度函数,不按长度拆对称 API**——≤16B inline 变体与默认字节级同码(零差别)、长串 seam 已摊平(省 ~1ns 噪声级),真收益只在 17-128B;为"对称"给每长度带配 inline 入口=制造与默认相同的冗余 API + 选择困惑,违"不为对称加复杂度"。用户按"要不要消 seam + 能否源码内嵌"决策,非按长度。usage-guide §3.2 讲清何时用。
3. **mix64 整数哈希已 SOTA**:与 abseil `Mix` 逐字同款(单乘折叠),赢 wyhash64 1.27x(后者两次 `_wymum` 更长链)、赢 std 3.4x。SwissTable 整数键热路径的哈希基础扎实。
4. **best-of-breed byte hasher:按长度选 XXH3/wyhash(2026-07 落地,256B 2.8x)**:长度谱实测(15 遍 median,per-hash)定出交叉点——**XXH3 短中串(17-128B)赢 wyhash 1.1-1.26x,wyhash 长串(≥192B)赢 1.45-2.6x**(XXH3 在 256B 附近有 midsize-path 劣化谷)。既然两者质量同档(都过 SMHasher,wyhash 是 Go1.17+/Zig/Nim 默认),就"谁快用谁":vendor wyhash(`third_party/wyhash`,单头 295 行,public domain)出线核 `hash_bytes_wyhash`,阈值 `kWyhashThreshold=128`(保守,tie 点归 XXH3 侧,wyhash 只接管明显赢的区间)。**两条分派各用其所**:① `hash_bytes(ptr,len)` 变长——`len` 运行时才知,用**运行时分支**(≤16 inline / ≤128 XXH3 / >128 wyhash,一条可预测 if,真"热切换");② `Hash<T>` 定长——`sizeof(T)` 是编译期常量,用 **`if constexpr` 三档全展开**,每 `Hash<T>` 实例化只编译进一个 hasher、**零运行时分支**。**实测 256B 0.082→0.029ns=2.8x**,精确追平 raw wyhash;64B 仍走 XXH3(0.060,不误切)。按长度切换是确定性的(hash 不跨进程持久化,同 build 内一致即可)。质量:新增测试覆盖 wyhash 长串路径(10k 无碰撞 + seed 敏感)+ 128/129 边界确定性;container 双平台 146/146。**真实收益边界**:HYLE 内部 >16B hash 的调用者是 string 驻留(string_buf)+ string_ref map key,多为短中串(XXH3 侧),长串是冷门——best-of-breed 主要是"谁快用谁"的正确性与未来长串负载兜底,当前热路径受益有限。
5. **未被 mem/simd/util 加速(确认)**:hash 不分配(mem 无关);批量 SIMD hash 需 64×64→128 高位乘,AVX2 无(要 AVX-512 `vpmullq`),本机不划算=新特性非漏接;util 的 `mul128_hi_lo` 与 hash 的 128 位乘语义略异(hi+lo out-param vs hi^lo 折叠),可后续收敛为单一 helper(纯代码复用,非性能)。



> 环境:WSL Ubuntu24.04 原生 ext4(非 9p)、GCC14.2 Release、nanobench。4096 个 f32(L1 常驻,隔离计算吞吐非内存带宽)。`bench/simd/`。**Highway = SOTA 标尺**(静态 dispatch,与 HYLE avx2 同 ISA、同 kernel 形状,只差抽象层)。**数字 = 15 遍进程级复测的中位数 op/s + run-to-run spread(`scripts/bench-stat.py` 驱动,沿 hashmap find-miss 那节的统计纪律)**;判定规则:差距 > 两变体 spread 之和才算赢/输,否则**统计平局**。
>
> **⚠️ 公平性更正(2026-06-30)**:此前此节标 dot"~1.10x 微超"、saxpy"~1.21x 微超" Highway——**是假象,源于对照被编瘸**。Highway 的 `HWY_STATIC_TARGET` 从完整 baseline 宏链(`detect_targets.h`)推断向量宽度:AVX2 baseline 需要 SSE4 baseline(`__PCLMUL__`+`__AES__`)再加 BMI2+FMA+F16C。bench 给 Highway 对照 TU 的是 HYLE 自家 kernel 的 AVX2 ISA 契约(`-mavx2 -mfma -mf16c`,刻意精简、关联 SIGILL-safety),在 **GCC/Clang 下让 Highway 静默降级到 SSSE3(128 位,Lanes=4)**——objdump 实证四个 `*_highway` kernel 全是 xmm 无 ymm。于是"parity"实际是 HYLE-256b vs Highway-128b,那 1.1-2x"赢"是宽度差不是抽象优势。修复:`hyle_set_highway_isa`(`HyleModule.cmake`)给对照 TU 补全 baseline(`-mbmi -mbmi2 -maes -mpclmul`),Highway 选回真 256 位 AVX2(`HWY_STATIC_TARGET=AVX2`、Lanes=8、四 kernel 全 emit ymm)。MSVC `/arch:AVX2` 本就含完整 baseline,故 Win 侧测的 dtype/math 节不受影响。下表是**修复后**的公平数据。

| kernel | scalar(auto-vec SSE2) | HYLE sse4(128b) | HYLE avx2(256b+FMA,4-acc) | Highway contrib/dot(**真 SOTA 4-acc**) | 结论(15 遍中位数) |
|---|---|---|---|---|---|
| **dot product** | 2.7 G op/s (1.0x) | 26 G (9.6x) | **42 G (15x)** | 42 G (15x) | **平真 SOTA**(ratio 0.984,gap 1.6% « spread 14%);4 累加器展开后 12.6→42 G |
| **saxpy** y=a*x+y | 17.4 G (1.0x) | 17.4 G (1.0x) | **39 G (2.2x)** | 40 G (2.3x) | **平 Highway**(element-wise 带宽受限,单累加器已够,未改) |

**结论(2026-07 红队揪出并修复第二个对照编瘸——dot 单累加器延迟链)**:
1. **旧"dot 平 Highway 12.6 vs 12.7 G"是两个都编瘸的假平局**。HYLE `dot_f32`(`kernel_impl.h`)和 bench 的"SOTA 标尺"`dot_highway` **都是单累加器**(`acc = MulAdd(.., acc)`)——每条 FMA 串进下一条,throughput 被 FMA **延迟**(~4 cyc)绑死,每 4 周期才退休一条。bench 拿的 Highway 参照是其**教学式 `ScalableTag` 写法,不是** Highway 真正的 `contrib/dot::Compute`(4 独立累加器 + 4x 展开,注释明写"2 FMA ports × 4 cycle latency = up to 8 in-flight, always unroll 4x")。**这是继"Highway 静态 dispatch 静默降 128b"之后的第二个同类对照陷阱:对照用了非优化路径。**
2. **修复=dot 主循环 4 独立累加器 + 归约树**(objdump 实证:主循环 4 条 `vfmadd231ps` 写 ymm0/1/2/3 + `vaddps` 树)。i9 双 FMA 端口下,4 链保持 ~8 FMA in-flight,从延迟绑定转 load 端口绑定。**dot 0.079→0.024 ns(3.3x,12.6→42 G op/s),对拍 Highway 真 `contrib/dot` 统计平局(ratio 0.984,gap 1.6%«spread 14%)**——真 SOTA 上限 42 G 而非旧标尺的 12.6 G。dot 是 AI 热路径主力(注意力/GEMM 内积),这是真收益。
3. **saxpy/scale/reduce 未改**:saxpy(`y=a*x+y`)、scale(`y=a*x`)是 **element-wise 无跨迭代依赖**,本就带宽绑定,多累加器无益(saxpy 已 39 G 平 Highway 40 G)。reduce_max 有归约依赖(理论同 dot 可展开)但**是 bench-only kernel,非公开 op**(`ops.h` 只导出 dot_f32/saxpy_f32),故不改;若将来上架为公开 op 再同 dot 展开。
> **意义 + 方法学印证**:这正是审查铁律"对标 SOTA 务必 objdump 确认对照真用了预期实现/宽度"的活例——第一次是对照被编瘸降 128b,这次是对照用了单累加器教学 kernel 冒充 SOTA。**两次都是"平局"掩盖了真实差距**:表面同速,实为两边都没到硬件上限。dot 真 SOTA 是 42 G,HYLE 现达到。**教训:'平 SOTA' 若对照本身可疑(如同形却都简单),要追问对照是不是该库的生产实现。**

### simd 数学核（exp/log/tanh vs 标量 libm + 同 ISA SOTA xsimd/Highway，`math_bench.cc`）

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4,各 15 遍中位数+spread**(`scripts/bench-stat.py`),nanobench,4096 f32(L1 常驻)。**两个标尺**:① 标量 `std::exp/log/tanh`(测向量化加速本身);② **同 ISA SOTA = xsimd + Highway contrib/math**(同 `-mavx2`)。HYLE 借 Highway/xsimd 验证过的核(已 vendored、无条件默认编进生产,不手搓多项式)。`active_math_variant()` 报当前层。

| kernel | 标量 libm | HYLE(best-of-breed) | xsimd | Highway | 结论(双平台 15 遍) |
|---|---|---|---|---|---|
| **exp f32**(→xsimd) | 2.06 ns | **0.23** | 0.22 | 0.30 | ✅ 平 xsimd(借其核)、反超 Highway 1.30x(双平台一致) |
| **log f32**(→Highway) | 2.0–2.6 ns | **0.27** | 0.37 | 0.27 | ✅ 平 Highway(借其核)、反超 xsimd 1.45x |
| **tanh f32**(→xsimd) | 6.3–8.7 ns | **0.44** | 0.45 | 0.54 | ✅ 平 xsimd(借其核)、反超 Highway 1.22x |

**核心诚实结论(同 ISA 对标 + 逐 op best-of-breed dispatch,全 op ≥ 两库 SOTA)**:
1. **⚠️ 揪出并修复一个真产品 bug——Highway 数学核在 Linux 出厂 128b**:`src/simd/math/transcendental_highway.cc`(log 借核)此前用 `hyle_set_isa(AVX2)`,缺 Highway 静态 dispatch 需要的完整 baseline(bmi2/aes/pclmul),GCC/Clang 下静默降级 SSSE3/128b(objdump:shipped 核 0 ymm/170 xmm)。**后果:出厂库的 log batch 在 Linux 跑半宽,0.53ns vs 应有 0.27ns,慢两库一半**——这不是文档错,是产品代码 bug(simd 那轮编瘸 bug 的产品侧遗漏,当时只修了 bench 对照)。改 `hyle_set_highway_isa` 补全 baseline 后:产品核 0→141 ymm,**Linux log 0.53→0.27ns,追平 Highway 核**。MSVC `/arch:AVX2` 自带完整 baseline,故 Win 一直正常(0.27)——是 Linux/GCC 特有出厂 bug。**这正是双平台审查的价值:文档"log 反超两库"是 Win 真相,却掩盖了 Linux 产品库半速。**
2. **"平 vs 反超"措辞校正**:HYLE 的 best-of-breed 是**借**对应库的核(exp/tanh 借 xsimd、log 借 Highway),所以对被借的库本质是**平局(同核)**,只对另一库反超。旧文档"log 反超两库"不确——借 Highway 核不可能反超 Highway,实测 log 0.27≈Highway 0.27 是平。诚实表述:**exp/tanh 平 xsimd + 反超 Highway;log 平 Highway + 反超 xsimd。每 op 都 ≥ 两库,但"≥"里对被借库是"平"非"超"。**
3. **逐 op best-of-breed dispatch**:同 ISA 对标证实没有单一库三 op 全胜(xsimd 的 exp/tanh 快、Highway 的 log 快),dispatcher 按 op 选最快核(`math_dispatch.cc`,编译期选,零运行时开销)。
4. **借核现状**:Highway + xsimd 均 vendored、无条件编进生产、默认全速,运行期 `cpu_features().avx2` 门控老 CPU 落 scalar 不 SIGILL。math_test 5/5、full simd 56/56。(注:SLEEF 已从项目移除,dispatcher 三 op 都没选它过——bench 顶部若仍有 SLEEF 注释属陈旧。)

### simd dtype 转换（fp16/bf16 <-> f32，`dtype_bench.cc`）

> 环境:**双平台**——Win MSVC19.50 + **Linux GCC14.2 原生 ext4,各 15 遍中位数+spread**(`scripts/bench-stat.py`),nanobench,4096。三组:HYLE F16C/AVX2 硬件 dispatch / HYLE scalar 参考 / Highway(gated SOTA,baseline 齐已验 488 ymem 非编瘸)。`active_dtype_variant()` 报 `AVX2/F16C`。std 无 fp16/bf16 批量等价物。**dtype 是平台高度依赖项**:判定按双平台分列。

| 转换 | HYLE | vs scalar | vs Highway(SOTA) | 结论 |
|---|---|---|---|---|
| **f16→f32** | 0.025 ns | Win 26x / Lin 19x | **Win 1.88x / Lin 平局** | ✅Win 赢 Highway 1.88x、Linux 统计平局 |
| **f32→f16** | 0.030 ns | Win 80x / Lin 25x | **Win 1.61x / Lin 1.90x** | ✅**双平台都赢 Highway**(唯一双平台稳赢) |
| **bf16→f32** | 0.030/0.040 ns | Win 3.5x / Lin 平局 | **Win 1.87x / Lin 平局** | ✅Win 赢 Highway、Linux 平 |
| **f32→bf16** | 0.095(Lin)/0.098(Win) | Win 7x / Lin 1.9x | 平真 SOTA(OrderedDemote2To) | ➖平真 SOTA(2-input 生产形状,详见 ▼) |

**结论(双平台据实,Highway 对照 baseline 完整非编瘸)**:
- **f16/bf16 F16C/AVX2 路径真兑现**——比 scalar Win 上 3.5–80x、Linux 上 1.8–25x(fp16 两平台都大赢 scalar,"fp16 dtype(F16C)"声明坐实)。
- **vs Highway 是平台相关,不是无条件赢**:**Win 上 HYLE 赢 Highway 1.6–1.9x**(f16→f32/f32→f16/bf16→f32 三项硬赢)——MSVC 把 Highway 的 PromoteTo/DemoteTo 编得不如 HYLE 直接用 `_mm256_cvtph_ps`/`cvtps_ph`(F16C 单指令)、`cvtepu16+slli`(bf16)紧凑;**Linux/GCC 上多数统计平局**——GCC 把 Highway 编得和 HYLE 一样好(亚-ns 量级 + Highway 那侧 spread 27–48% 噪声大)。**唯 f32→f16 双平台都赢 Highway**(Win 1.61x/Lin 1.90x,F16C 单指令 DemoteTo 的结构优势两平台都在)。诚实表述:**dtype 在 Win 赢 SOTA、在 Linux 平 SOTA,无虚假"无条件碾压"**。
- **bf16 Highway 对照已补(原"待补"已填)+ f32→bf16 改生产形状(2026-07 红队二轮)**:bf16<->f32 是纯整数位运算(bf16 = f32 高 16 位),Highway 同样用 AVX2 整数做。**f32→bf16 揪出与 FMA 案同构的漏接**:HYLE 旧核每 8 元素 `packus+permute4x64`(跨 lane 3c)+ **128b 半宽存**,正是 Highway **便利版 `DemoteTo`(8→8)** 的形状;Highway **生产版 `OrderedDemote2To`** 吃 2 个 f32 向量→1 个满 bf16 向量,每 16 元素只 1 packus+1 vpermq+**满 256b 存**,把跨 lane shuffle 和存端口摊到 2 倍元素。改用 2-input 生产形状(RNE 位语义不变,仍 bit-identical scalar oracle):**Linux 0.104→0.095 ~1.09x,双平台平真 SOTA `OrderedDemote2To`(Lin 0.095 vs 0.097 ratio 1.02、Win 0.098 vs 0.099),反超便利版 `DemoteTo`(0.107/0.117)**。收益比 FMA 案小(~1.09x 非 3x)因 bf16 demote 是 store-bound 非 compute-bound——摊销 shuffle 有限,存带宽才是墙。**bench 标尺同步升级**:旧标尺用便利版 `DemoteTo`(同 HYLE 旧形状=两个朴素核假平),加 `OrderedDemote2To` 真 SOTA 行。
- **bf16 vs scalar 也平台相关**:Win 赢 3.5x(bf16→f32)/6.7x(f32→bf16),但 **Linux 上 bf16→f32 与 scalar 统计平局**——GCC 把纯移位的 scalar 循环自动向量化到与 AVX2 核同速(MSVC 没有)。这不是 HYLE 退化,是 GCC 自动向量化够强;f32→bf16 因 RNE 比较复杂 GCC 没完全向量化,HYLE 仍赢 1.8x。**bf16 正确性**:纯位运算,AVX2 与 scalar bit-identical(NaN payload 也是),0x10000 全模式 + f32 宽扫 bit-exact 测试验证。

### simd 逐 op 扩展（reduce_max / scale，`simd_bench.cc`）

> dot/saxpy 之外补两个 distinct pattern,验证 vec.h 抽象在非 FMA 模式下是否退化。环境同上(15 遍中位数+spread,Highway 真 256b 对照——见上节公平性更正)。**记账口径**:reduce_max/scale 是 **bench-only kernel(`simd_kernels_*.cc`),不是 `ops.h` 导出的公开 op**(公开只有 dot_f32/saxpy_f32)——此处纯为验证 vec.h 抽象在 max-归约/element-wise map 上零成本,非"已交付的 SIMD 能力"。若将来上架为公开 op,reduce_max 有归约依赖可同 dot 做多累加器展开(现为单累加器)。

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
| **directory_iterate visitor** | 2.90 µs | std::filesystem 5.19 µs | **✅赢 1.81x**(gap 81% » spread 33%,min 1.73x) | 统计平局(32.20 vs 32.22 µs,1.002x,gap 0.2% « spread 11.9%;已切 FindFirstFileExW) | std::filesystem 目录遍历 |

> **Parker unpark+park 快路径未列入表格——没有语义对等的公平对照**。这条热路径(unpark 先到、park 不睡)实测 9.94 ns(Linux)/9.6 ns(Win),但**找不到一个"既给同样的唤醒保证、又同样无 syscall"的标准件来比**,所以从对标表移除,只在下面留注释说明(理由见 ② 节)。绝对值有意义(无 syscall、~10 ns),相对"输赢"无诚实分母可填。

**① Parker 真阻塞乒乓——平台差异的典型,两个数字都对**：两线程互相 park/unpark,每腿一次真睡眠+唤醒(`futex`/`WaitOnAddress`)。**Win 上赢 CV 2.85x**——Windows 的 `condition_variable` 要 `SRWLock`+`CONDITION_VARIABLE`+谓词三件套,Parker 的裸 futex 直驱省掉这些。**Linux 上只统计平局**——glibc 的 `pthread_cond` 本身已是薄薄一层 futex 封装,跟 Parker 的直驱 futex 成本几乎一样,没有结构性差距可拿。**这不是回归,是"对照在 Linux 上本就更强"**:同一个 HYLE 实现,赢多少取决于对手平台实现有多重。

**② Parker unpark+park 快路径——为什么没有公平对照,那个"安全"是什么**：这是调度器最常走的热路径——**unpark 先到、park 不必睡**。三态机 `EMPTY=0`/`NOTIFIED=1`/`PARKED=0xFFFFFFFF`,`park()` 第一步 `state_.fetch_sub(1, acquire)`(`wait.cc:103`):看到 `NOTIFIED(1)` 就减成 `EMPTY` 直接返回(快路径,不睡);看到 `EMPTY(0)` 则回绕成 `PARKED` 去 `wait_on_address` 睡。绝对值 **9.94 ns(Linux)/9.6 ns(Win),全程无 syscall**,对比真阻塞那条走 futex 的 ~20 µs 快约 2000 倍——热路径成本就是一条 `fetch_sub`(`lock xadd`,RMW 带 acquire)+ 分支。
> - **为什么不给"对标倍数"——没有语义对等的对手**。要公平比,对照必须同时满足两条:① 提供同样的 **unpark-before-park 不丢唤醒**保证;② 同样**无 syscall** 的快路径。实测两个候选都不对等(故移出表格):
>   - **裸 `atomic` flag(4.4 ns)**:无 syscall,但**没有唤醒保证**——unpark 先到时信号丢失,随后 park **永久睡死**。拿它当对手是让"有保证的"比"裸奔的",对 HYLE **苛刻**(凭空背一个 2x"输")。
>   - **`std::binary_semaphore`**:语义对等(release/acquire 令牌安全),但 libstdc++ 实测 **release+acquire 快路径每轮都发一次 futex**(strace 实证 10 万轮=10 万次 futex),根本没有"令牌已在就纯原子返回"的优化——比它 HYLE 会**远赢**(µs vs 10 ns),对 HYLE **太宽松**,同样是误导。
>   - 结论:**既给保证、又无 syscall 的标准件不存在**,所以这条只报绝对值,不填输赢分母(这正是上一轮 simd 的教训:对照语义/ISA 不对等,任何倍数都是假的)。
> - **那条快路径买到的"安全" = unpark-before-park 不丢唤醒**。并发中唤醒方/等待方在不同线程赛跑。若 unpark **先于** park 发生,朴素 flag 会让随后的 park **永久睡死**(信号在无人等待时发出就丢了)。三态机让 unpark 把状态置 `NOTIFIED`(`exchange(kNotified)`,`wait.cc:122`)**记住令牌**;park 进来 `fetch_sub` 撞见 `NOTIFIED` 直接取走返回,**令牌永不丢失**。这是 CV/调度器赖以正确的语义,也是它比裸 flag 多花那条 RMW 的原因——那不是低效,是正确性的最低成本。

**③ Thread spawn+join——双平台都平,意义在"证零开销"**:建线程→跑空函数→join,2000 次。标尺是 `std::thread` 本身(两者都包 `pthread_create`/`CreateThread`,µs 级,OS 全主导)。Linux/Win 都统计平局,证明 HYLE 那个能设栈大小/亲和性的 ABI-clean 句柄**没在 OS 之上加任何税**。

**④ directory_iterate——Win 统计平局、Linux 赢 1.81x,平台差异不可混看**：此前总账只记 Win 单平台数且数值方向(HYLE 33.1 µs ≳ std 32.2 µs)看着像"略输",易被误判。**统计法澄清两件事**:① **Win 是统计平局**(FindFirstFileExW+FindExInfoBasic 后 32.20 vs std 32.22 µs,1.002x,gap 0.2% « spread 11.9%),不是输;② **Linux 是真赢 1.81x**(2.90 vs 5.19 µs,15 遍 min 都 1.73x,gap 81% » spread 33%,铁证)。Linux 赢的根因是设计意图兑现:HYLE 走 `readdir` + `d_type` 直接拿"是否目录"(`fs.cc:487`),**省掉 stat**;`std::filesystem::directory_iterator` 每步构造 `directory_entry`,bench 里的 `is_directory()` 还触发**额外 stat 系统调用**(`platform_bench.cc:165`),再加每条目的 path 分配。无分配 visitor + 免 stat = 真实结构优势;Win 上两者底层都走 `FindNextFileW` 量级相近故平;`FindFirstFileExW+FindExInfoBasic` 只跳过一次性短名工作,被逐项 `FindNextFileW` + UTF-16→UTF-8 visitor 转换淹没,故**只保留为严格更少工作/对齐 STL,不宣称 bench win**。**教训:platform 项必须标平台,单平台数字直接搬到总账会误导输赢判断**(本轮就是从"疑似输"查成"Win 平 / Linux 赢")。



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

> **⚠️ "选错工具"明示**:`MemoryArena`/`ObjectPool` 是**为多线程共享有界池**设计的(无锁分片 Treiber 栈)。**单线程**同尺寸 churn 用它们是误用——每次 alloc+free 各付一次原子 CAS(两条 `lock cmpxchg`)+ tls homing + 地址路由,实测 13.97ns 慢裸 mi 3.1x、连 std 单线程池(`unsynchronized_pool` 11.6ns)都不如。这不是 HYLE 的缺陷,是拿错了工具——**单线程对象池请用 `GrowablePool`**(同 workload 1.28ns,3.5x 赢裸 mi)。故此误用对比**已从对标 ledger 移除**(留此行仅作选型警示),`MemoryArena` 的真实对标是多线程共享场景(见下,赢 std 共享池 44x)。usage 选型见 [`guide/usage-guide.md` §3.1.1](../guide/usage-guide.md)。

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

### container·vec / ·list —— SmallVector / IntrusiveList / CArray（SmallVector / IntrusiveList / CArray vs std + 真 SOTA，`container_misc_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。声明承重容器。**标尺升级**:不止 std——SmallVector→`absl::InlinedVector`(small-vector SOTA,gated `-DHYLE_BENCH_ABSL`)、IntrusiveList→`boost::intrusive::list`(侵入容器 canonical SOTA,gated `-DHYLE_BENCH_BOOST`)、CArray→`std::vector`(动态数组的 drop-in SOTA,本身已是高度调优,always-on)。

| 场景 | HYLE | std | 真 SOTA(gated) | 结论 |
|---|---|---|---|---|
| **小 N churn**(建+填 8 int+析构) | SmallVector<int,16> **0.41 ns** | vector 16.8 / vector+reserve 3.06 | **absl::InlinedVector 0.84–1.0** | ✅ **赢 absl 2.0–2.4x**(absl 侧 spread 大)、赢 std 41x |
| **list churn**(push 4096 + pop 全) | IntrusiveList **3.15 ns** | list 25.0 | boost::intrusive **2.77**(平) | **8x 赢 std、平 boost SOTA**(2026-07 实测) |
| **动态数组 growth**(push_back 4096 int,无 reserve) | CArray<int> **0.51 ns** | vector **0.75** | (std::vector 即 SOTA) | ✅ **赢 std::vector 1.48x**(grow 用 reallocate) |
| **动态数组 read**(顺序求和 4096) | CArray<int> **0.18 ns** | vector 0.18 | (std::vector 即 SOTA) | ➖ **平 std::vector**(arr[i] 单 load 兑现) |

**诚实结论**:① **SmallVector 反超 absl::InlinedVector 2.0–2.4x**——不是"只赢 std":absl::InlinedVector 是 small-vector 的 SOTA(同 inline 存储设计),HYLE 0.41ns vs absl 0.84–1.0ns(absl 侧 nanobench 标 unstable、spread 大,故取范围),小 N churn 真赢同款 SOTA(也赢 std::vector 41x);溢出路径同样赢 absl 2.7x。② **IntrusiveList 8x 赢 std::list + 平 boost::intrusive SOTA(2026-07 实测,补 ⏳待补)**:零节点分配。checkout boost submodule(intrusive+config/assert/core/move/static_assert 依赖闭包)后实测:push/pop churn HYLE 3.15 vs boost 2.77(ratio 0.88,gap 12%«spread 44%=**统计平局**)、任意位置 O(1) remove HYLE 3.13 vs boost 3.15(平)。两者同 **circular-sentinel 双向链**形状(canonical intrusive),无抄漏——HYLE `unlink` 额外清 prev/next=nullptr 是 unlink-安全特性(支持 `linked()` 查询 + 安全重复 unlink),非性能差。list 项无可反哺的前轮优化(纯指针操作,不涉分配/hash/SIMD),测试项已覆盖 push/pop/front/back/任意 remove/empty 六种 + 两 bench 场景(churn + 乱序 unlink)。③ **CArray 反超 std::vector ~1.48x growth、平 read**:growth 赢的**真根因(2026-07 纠错)**——旧文档称"经 `reallocate()` mimalloc 原地扩展、无 copy"是**双重错**:① 当时代码根本没用 reallocate(是 `allocate`+`memcpy`+`free`,与 std::vector 同形);② 几何增长(4→8→16…)常跨 mimalloc size-class,**即便用 reallocate 也多是 relocate+copy,in-place 很少触发**。真正赢 std 的因是 **mimalloc 分配器质量**(比 std::vector 的 `operator new` 快)+ header-before-data 零索引开销,不是"原地无 copy"。现已把 grow **真改用 `mem::reallocate`**(兑现文档、且是更简 API：≤ 手动路径,偶尔省一次 copy),实测 growth 0.53→0.51ns、1.34x→1.48x(边际,证实 in-place 少触发);read 路径 arr[i] 单 load 与 vector 持平(header-before-data 布局零索引开销)。

### container·str —— string_ref / StringBuf / string_algo（`string_ref_bench.cc` / `string_buf_bench.cc` / `string_algo_bench.cc`）

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

### container·path —— 纯词法路径操作（`path_bench.cc`）

> 环境:Win MSVC19.50 Release,nanobench。`path.h` 是 HYLE 的纯字符串路径层(无 syscall),`extension`/`base`/`directory`/`normalize`/`split` 在 string_ref 视图上原地走分量、**零分配**。**标尺 = `std::filesystem::path` 词法操作**——但后者构造一个 path 对象(分配 + 存字符串),HYLE 只在视图上扫,故全线赢。

| op | HYLE | std::filesystem | vs SOTA | 结论 |
|---|---|---|---|---|
| **extension / base / directory** | 44/10/9.7 ns | 152/164/166 | 3.4/16/17x | ✅赢(视图原地扫 vs 构造 path 分配) |
| **normalize 词法折叠** | 65 ns | lexically_normal 382 | 5.85x | ✅赢(就地折叠 `.`/`..`/分隔符 vs 堆 path 对象) |
| **join_array 6 部件** | 30 ns | operator/= ×6 = 454 | 14.9x | ✅赢(buffer-fill 折叠 vs N 次 path 分配) |

**诚实结论**:path 全线赢 `std::filesystem` 3.4–17x,根因单一且真实——HYLE 是**视图上的纯词法扫描、零分配**,std::filesystem 每次操作都构造 path 对象(堆分配 + 存字符串副本)。这不是巧妙算法,是"不分配 vs 必分配"的结构差;对标 std::filesystem 是公平的(它就是标准库的路径词法 SOTA),但 HYLE 的优势仅在**纯词法**层(无 syscall 的 fs 操作另见 platform 节 directory_iterate)。

### 次要模块（util/random、time/clock、obs/log、SwissTable iterate，`misc_bench.cc` + `swiss_table_bench.cc`）

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






- **桶 A — 同操作更省开销（真 drop-in 可比）**：hashmap、分配器、容器。给"倍数"有意义。
- **桶 B — std 根本没有的能力**：fiber、async IO。"倍数"是错的框（无可比物），是「从不能到能」。

## AVX-512 上架状态（①②③ 已接线+双平台编译验真，待真机执行对拍；④ pointer-bump 已完成）

> 逐模块性能审查(error→simd→math→dtype→platform→mem→time)后，把"已 ✅赢 or ➖平 SOTA、但本机硬件够不到、或仍差一条指令"的优化**显式记账**于此。**本开发机 = i9-13900K，AVX2-only 无 AVX-512**(P-core 屏蔽 512)，故 ①②③ 的 512-bit 路径在本机**无法执行**。**用户指令(2026-07)**:照 NEON/avx512 backend 先例**实现+接线+双平台编译 objdump 验真发宽指令**,执行对拍待真机/SDE——"不做用户根本无从验证"。①②③ 现已全部上架(代码+dispatch+CMake tier+测试接线),双平台编译验真完成,执行对拍 `存疑` 待真机。④ 与硬件无关、纯 codegen,**已完成**(双平台测试过)。

| # | 方向 | 状态(2026-07 已上架) | 验真证据 | 待真机对拍 |
|---|---|---|---|---|
| ① | **dtype `f32→bf16` native `VCVTNEPS2BF16`** | ✅上架:`cvt_avx512bf16.cc`(`_mm512_cvtneps_pbh` 16-wide 硬件 RNE,把 AVX2 ~7-op RNE 折成单条),`dtype_dispatch` f32→bf16 优先 `avx512bf16` 档(`cpu_features().avx512bf16` 门控)回退 AVX2/scalar;新 `AVX512_BF16` CMake tier | 双工具链编译+指令验:GCC14 `vcvtneps2bf16`(EVEX.512)、MSVC `/arch:AVX512` `vcvtneps2bf16 ymm,zmmword`。测试 `F32ToBf16Avx512MatchesScalar`(NaN equal-by-class、finite bit-exact)本机正确 SKIP | 真机/SDE 跑该测试即对拍(纯硬件 RNE,finite 与 oracle bit-identical) |
| ② | **simd ops AVX-512 派发档** | ✅已接线(此前里程碑):`kernels_avx512.cc`(共享 `kernel_impl.h` 16-wide)+ `dispatch_ops.cc` 默认路由 AVX-512 host;本轮复核接线完整 | objdump:GCC 7 zmm/MSVC 33 zmm(含 `vfmadd213ps zmm`)。`ops_test` 宽度无关对拍 oracle,真机 `active_variant()` 自动选 kAvx512 | 真机跑 `ops_test`/`vec_test` 即对拍(共享体与已实跑 avx2/sse4 同源,仅宽度宏不同) |
| ③ | **math exp/log/tanh 512b 变体** | ✅上架:`transcendental_{xsimd,highway}_avx512.cc`(借同核 512b/16-lane),`math_dispatch` 加 avx512 档优先(exp/tanh→xsimd-512、log→highway-512)回退 AVX2;新 `hyle_set_avx512_math_isa`(全 AVX-512 组+cd) | objdump:GCC xsimd512 92zmm/highway512 43zmm、MSVC xsimd512 581zmm。`Math.*MatchesLibm` 对 libm 对拍,本机走 AVX2 路径过 | 真机跑 `math_test` 即对拍。**MSVC log-512 已排除**(Highway AVX3=`HWY_BROKEN_MSVC` 退 ymm,故 highway512 仅 GCC/Clang;MSVC log 留 avx2 highway,同速不假装512) |
| ④ | **Arena/StackAllocator pointer-style bump**(硬件无关) | ✅完成:offset-based(存 `base_`+`top_`/`offset_`,每 `allocate` 重算 `base+offset`→对齐→减回 offset)改 canonical `cur_`/`end_` 双指针纯指针 bump,直接对齐活指针。Marker 从 size_t 偏移改 byte*(API 不变) | objdump(GCC -O2 inlined):offset 版 ~15 指令含 3 load + base+offset 往返(对齐一次、overflow 记账再算一次);pointer 版 ~10 指令 2 load 无 +base/-base。**bench-stat 15 遍实测(Linux):Arena 0.68→0.34ns、Stack 0.68→0.34ns、mark/rewind 0.63→0.43ns=~2x**(gap 50%»spread ~12%,真赢非平局) | — |

**诚实定性**:①②③ 已从"留账非即做"转为**已上架+双平台编译 objdump 验真**(用户指令:写完接线才能在真机验证,否则永远落不了地);三项的 512-bit 执行正确性是 `存疑`,待 Intel SDE/真 AVX-512 机跑既有测试(全部宽度无关/对 oracle,真机一跑即自证,无需新码)。**关键发现**:Highway 在 MSVC 把整个 AVX3 标 `HWY_BROKEN`(已知 codegen bug),故 ③ 的 highway-512 log 仅 GCC/Clang 上是真 512,MSVC 退 ymm 已显式排除——不假装 512。④ 已完成(pointer-style bump);**实测意外——不是预期的"亚-ns 平局"而是 ~2x 真赢**(Arena/Stack 0.68→0.34ns、mark/rewind 0.63→0.43ns,15 遍 gap 50%»spread 12%):offset 版每 alloc 多一次 `base_` load + base±offset 往返算两遍(对齐一次、overflow 记账再算一次),objdump ~15 指令 vs pointer 版 ~10 指令;在 0.68ns(仅 ~3-4 指令成本)的量级上省 5 条指令直接砍半。**"亚-ns 天花板不值得做"的预判被实测推翻**——bump 成本本身太小,任何多余指令占比都放大。**没有其它"本机低成本高收益"项**——time/`monotonic_ns` 那种(频率特判换 1.48x)的机会,逐模块扫描后未再发现(④ 这个 2x 是本轮附带发现的第二个)。

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
