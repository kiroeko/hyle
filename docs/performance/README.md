# HYLE 性能：预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 本目录结构（原单文件已拆分，按模块分档）

| 文档 | 内容 |
|---|---|
| 本页（对标总账） | 全模块 win/loss ledger 一览 + 总账小结（最常看的一张表） |
| [error / hash 实测](./benchmarks-error-hash.md) | error Status/Result（对标 absl）、hash 哈希（XXH3/wyhash/mix64）、SwissTable hashmap |
| [simd 实测](./benchmarks-simd.md) | dot/saxpy、exp/log/tanh 数学核、fp16/bf16 dtype、reduce_max/scale |
| [simd op 对拍方法论](./simd-op-benchmark-methodology.md) | **给 op_parity 新增条目/跑数/判真输的操作规程**:kernel 强制形状、逐平台跑法、校准闸门读法、LOSE 定罪证据阶梯、修复模式库、症状速查 |
| [sched 实测](./benchmarks-sched.md) | fiber 切换、platform 原语（Parker/Thread/dir_iterate）、sched 同步（marl 对标）、调度器层（work-stealing 扩展/spawn_on 亲和/协程胶水/fiber sleep）、lock-free 队列簇 |
| [io 实测](./benchmarks-io.md) | echo 吞吐 + shard-per-core 扩展 + vs 裸 epoll/liburing/asio/libuv + 连接生命周期 + TLS/HTTP + multishot/DEFER/fixed-files/direct-accept 机制项 |
| [mem / container 实测](./benchmarks-mem-container.md) | mem 分配器 vtable + 专用分配器簇、SmallVector/IntrusiveList/CArray、string_ref/StringBuf、path、次要模块 |
| [预期与边界](./expectations.md) | AVX-512 上架状态、主表（时间/空间分列）、Amdahl 整体观、诚实边界、验证 gate |

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
| **dtype** f32→bf16 | 0.095(Lin)/0.098(Win) | scalar ref | 0.18/0.70 | 1.9–7x | Highway OrderedDemote2To(真 SOTA 2-vec) | 0.097/0.099 | 1.02x/1.01x | ➖平真 SOTA(改 2-input 生产形状后 Lin 0.104→0.095 ~1.09x,平 OrderedDemote2To;旧对标的 DemoteTo 便利版是 8→8 朴素形状,HYLE 现反超它 0.107/0.117);**AVX-512-BF16 native `vcvtneps2bf16` 已上架**(待真机,见 expectations AVX-512 节①) |
| **math** exp(best-of-breed→xsimd) | 0.23 | libm scalar | 2.06 | 9x | Highway / xsimd | 0.30/0.22 | 1.30x/1.0x | ✅平 xsimd 最快、反超 Highway(双平台一致) |
| **math** log(best-of-breed→Highway) | 0.27 | libm scalar | 2.0–2.6 | 4–8x | Highway / xsimd | 0.27/0.37 | 1.0x/1.45x | ✅平 Highway、反超 xsimd(修产品核 128b bug 后 Linux 同样成立) |
| **math** tanh(best-of-breed→xsimd) | 0.44 | libm scalar | 6.3–8.7 | 14–20x | Highway / xsimd | 0.54/0.45 | 1.22x/1.0x | ✅平 xsimd、反超 Highway(双平台一致) |
| **platform** Parker ping-pong(真阻塞) | 317(Win) | std::condition_variable ping-pong | 927(Win) | 2.99x | Rust std park(概念对照) | n/a | n/a | ✅赢 std(Win 2.99x;**Linux 统计平局**,平台差异见 sched 详节) |
| **platform** Thread spawn+join | 40.4µs(Win) | std::thread | 40.1µs | 1.01x | std::thread(即 SOTA,同 syscall) | 40.1µs | 1.01x | ➖平(双平台一致;OS 主导,证 handle 零开销) |
| **platform** directory_iterate visitor | — | std::filesystem::directory_iterator | — | — | filesystem | — | — | ➖平(Win 0.98x)/**✅Linux 赢 1.81x**(免 stat 的 d_type + 无分配 visitor;详节见 sched) |
| **platform** Parker park_timeout(sched-04b 底座) | ~底座 | (无语义对等 std) | n/a | n/a | (Rust std park_timeout 概念对照) | n/a | n/a | ➖时间轮执行底座(futex 相对 timeout;TSan 5×净) |
| **time** realtime_ns | 14.68 | std::chrono::system_clock | 14.76 | 1.0x | std::chrono(即标尺,同 OS 原语) | 14.76 | 1.0x | ➖平 |
| **time** monotonic_ns | 13.22/13.07/12.93(Win/Lin/M4) | std::chrono::steady_clock | 13.30/13.07/12.88 | 全部统计平 | std::chrono(即标尺,同 OS 原语) | 同左 | 1.0x | ➖header-inline 后三平台与 std 同档 |
| **time** fast_monotonic_ns(opt-in rdtsc) | 7.37/7.13/12.93 | HYLE monotonic_ns | 13.22/13.07/12.93 | 1.79x/1.84x/1.0x | (std 无 opt-in TSC 钟) | n/a | n/a | ✅x86 快约1.8x；M4 透明回退(非默认,可能跳变) |
| **time** fast_realtime_ns(opt-in coarse) | 3.20/3.90/13.64 | HYLE realtime_ns | 15.77/13.09/13.70 | 4.94x/3.36x/1.0x | (std 无 opt-in coarse 钟) | n/a | n/a | ✅Win/Linux 真赢；macOS 无 coarse 原语而回退 |
| **time** timer wheel churn(arm+70%cancel+expire) | 12.87/13.63/7.62 | (std 无定时器结构) | n/a | n/a | indexed eager-erase heap(libevent/libuv 核心形状) | 20.85/16.01/11.03 | 1.62x/1.17x/1.45x | ✅**三平台赢**；对手按真实 keyed erase 建模、无墓碑 |
| **time** timer wheel arm+cancel / arm+expire | 6.86·21.34/5.95·22.05/4.91·11.83 | (std 无) | n/a | n/a | indexed eager-erase heap | 10.42·49.80/6.73·40.64/8.36·23.74 | 1.52·2.33x/1.13·1.84x/1.70·2.01x | ✅三种工作量三平台同向；8×64 覆盖完整 uint64 ns 域 |
| **time** CommonTimeoutQueue(固定时长)arm+expire/cancel | 3.28·2.70/3.14·1.99/3.52·3.06 | (std 无定时器结构) | n/a | n/a | 裸 TAILQ / checked TAILQ | 4.22·3.13/3.95·2.59/2.73·3.06；checked expire 3.61/3.21/4.21 | 裸 1.29·1.14x/1.29·1.30x/**0.77·平**；checked expire 1.10x/平/**1.16x** | ✅同语义三平台不输；⚠️ M4 对无检查裸下界仍输 |
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
| **mem** Arena bump | 0.34 | new/delete | ~15 | ~44x | std::pmr monotonic | 0.74 | 2.2x | ✅赢(pointer-style bump 后 0.67→0.34,vs 裸 mi 4.0ns~12x;见 expectations AVX-512 表④) |
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
| **container·path** extension / base / directory | 2.3/4.6/10(Win)·1.8/4.3/4.7(Lin) | std::string_view 手写反向扫(纯算法基线,零分配) | 双平台各项 | 双平台**平**算法基线 | std::filesystem 词法 152/169/172(Win) | — | 66/37/17x(Win) | ✅**双平台平算法基线**(证扫描已最优,非更巧算法)+ 碾 filesystem(结构差:零分配 vs 构造 path);extension 2026-07 正向全扫→反向扫遇分隔符停 40→2.3(18x);base Win"1.84x"是MSVC的string_view构造税,Lin平 |
| **container·path** normalize 词法折叠 | 69 | (无零分配 std 词法等价) | n/a | n/a | std::filesystem lexically_normal | 401 | 5.8x | ✅赢(就地折叠 vs 堆 path 对象;PRODUCE 型无 string_view 零分配对手) |
| **container·path** join_array 6 部件(无 alloc) | 29 | (无零分配 std 等价) | n/a | n/a | std::filesystem operator/= | 618 | 21x | ✅赢(buffer-fill 折叠 vs N 次 path 分配;PRODUCE 型无 string_view 对手) |
| **queue** 单线程 fill+drain | 7.77/7.79 | (std 无 bounded-MPMC) | n/a | n/a | rigtorp/moody | 11.54·11.84/9.48·8.41 | 1.5/1.2x·平 | ✅赢 rigtorp 双平台1.5x;moody Win赢1.23x·Lin平 |
| **queue** MPMC 2P 单环 BoundedQueue | 56.08/55.97 | (std 无) | n/a | n/a | rigtorp/moody | 61.04·57.88/73.90·61.69 | gap3.5-7.5% | ➖双平台平 rigtorp+moody(gap«spread26-29%,15遍) |
| **queue** MPMC 4P 单环 BoundedQueue | 69.26/69.10 | (std 无) | n/a | n/a | rigtorp/moody | 78.41·77.89/68.49·67.01 | gap12-15% | ➖双平台平 rigtorp+moody(原输1.33x,cache-pad+去backoff消除落后;gap«spread23-29%,不判反超) |
| **queue** MPMC 4P ShardedQueue | 25.31/14.60 | (std 无) | n/a | n/a | rigtorp | 78.41/77.89 | 3.27x/5.3x | ✅赢(生产走这条;spread21-43%大=采样不足,取量级) |
| **queue** work-steal owner FIFO/LIFO | 3.93/3.06 | (std 无) | n/a | n/a | crossbeam(Rust) | ~3.6/4.4 | ~1.2x/1.3x | ✅赢(跨语言,谨慎) |
| **queue** work-steal steal | 8.42 | (std 无) | n/a | n/a | crossbeam steal | (跨语言) | n/a | ➖HYLE WsFifo(本族 WSDeque 10.5) |
| **queue** ConcurrentVector push/读 | 4.51/0.93 | (std 无并发 vec) | n/a | n/a | oneTBB | 5.99/1.29 | 1.33x/1.39x | ✅赢 |
| **queue** ConcurrentVector 并发读(4读1写) | 0.37 | (std 无) | n/a | n/a | oneTBB | 0.82 | 2.2x | ✅赢(删冗余 acquire) |
| **sched** 争用 Mutex(15遍双平台) | 15.4(Win)/31.3(Lin) | marl=std::mutex | 19.0/31.2 | — | marl | 19.0/31.2 | Win1.23x/Lin平 | ✅**Win 反超**(SRWLOCK 修,gap22%>spread18%)/➖Lin 平 |
| **sched** 无争用 Mutex(MT) | 6.5(Lin) | std::mutex | 9.1 | 1.40x | marl(=std::mutex) | 9.1 | 1.40x | ✅赢(**单平台单遍,待双平台统计**) |
| **sched** spawn 2000 noop(15遍双平台) | 347(Win)/317(Lin) | marl | 302/309 | — | marl | 302/309 | 0.83/0.98 | ➖**双平台统计平局**(旧"2.17x"是单遍噪声,gap«spread) |
| **sched** WaitGroup fork-join(15遍双平台) | 341(Win)/305(Lin) | marl | 299/317 | — | marl | 299/317 | 0.87/1.05 | ➖**双平台统计平局**(旧"1.49x 反超"是单遍;conditional-notify 把 17x 落后修到平级) |
| **sched** Latch fork-join(64) | 5059(fiber) | std::latch(64线程) | 21975 | 4.3x | marl(无 latch) | n/a | n/a | ✅赢 std(fiber 不阻塞 worker) |
| **sched** CV notify_one 快路径 | 1.70 | std::condition_variable | 2.59 | 1.52x | marl(无独立 CV) | n/a | n/a | ✅赢 std |
| **sched** CV wait+notify handoff | 280 | (无 std fiber 等价) | n/a | n/a | (HYLE 自有) | n/a | n/a | ➖HYLE 自有(阻塞 handoff,TSan 零race) |
| **sched** fiber 切换 | 21–23 | std::thread 切换 | ~1000+ | ~50x | fcontext 原语 | 21–23 | 1.0x | ➖平 fcontext(=原语零封装) |
| **sched** suspend/wake 完成脊柱 | 163(Win)/121(Lin) | (无 std 等价) | n/a | n/a | (HYLE 自有脊柱) | n/a | n/a | ➖HYLE 自有(GPU/NIC/timer 汇流);**LIFO fast-slot 后 Win 240→163(−32%)/Lin 186→121(−35%)**——同 worker 唤醒走 cache-hot 慢槽,跳过 global CAS+notify futex(见 sched 详节) |
| **sched** Channel 收发(cap16,1p1c) | 63/item | (无 std 等价) | n/a | n/a | marl(无 channel) | n/a | n/a | ➖HYLE 自有 |
| **sched** RunQueue fifo/lifo push+next | 8.3/2.6 | (内部结构) | n/a | n/a | (无外部对手) | n/a | n/a | ➖HYLE 自有(完成唤醒热路径) |
| **sched** remove_sleeper(park 退出) | O(1) | (内部结构) | n/a | n/a | tokio idle.rs(容忍 O(n)扫描) | O(n) | 复杂度赢 | ➖复杂度机会②:per-worker position 索引 swap-remove 消 O(worker数)扫描,缩短 idle_mu 临界区(HYLE 每次 park 退出都调,故值得;tokio 只罕见路径扫故容忍) |
| **sched** global-queue 反饥饿 | 修 bug | (正确性) | n/a | n/a | tokio global_queue_interval=61 | 同款 | 修真 bug | ➖复杂度机会③:探针实证本地 firehose 饿死全局注入真实存在(yield-hog 1700万次而 victim 从不运行),每 61 tick 先查 global 修复(回归测试双向验证 pre-fail/post-49ms) |
| **sched** fiber sleep(sched-04b) | — | (std 无 M:N fiber sleep) | n/a | n/a | tokio time driver(概念对照) | n/a | n/a | ➖SOTA 成本模型:无 sleeper 时 wheel.empty() 单比较=零税;arm O(1) 无分配;park_timeout 不空转(非逐 ms tick) |
| **sched** CV wait_for notify-before-deadline round-trip | 580(Win)/~(Lin) | std::condition_variable::wait_for | 1165(Win) | 2.0x | (std 是直接 SOTA) | n/a | n/a | ✅**Win 赢 2.0x**(gap99.8%»spread47%,15遍)/ **Lin 平**(glibc cv 已薄 futex 封装,gap3.4%«spread21%)——又一单平台数陷阱,同 sched-05 Parker |
| **sched** CV wait_for 纯超时(1ms) | 15.58ms | std::condition_variable::wait_for | 15.58ms | ~1.0x | (OS 定时器地板) | n/a | n/a | ➖双平台平(gap0.1%«spread):HYLE 超时路径零额外开销,两者同被 OS ~15ms 定时器粒度地板 |
| **sched** spawn_task+join 每任务(sched-10,批量摊销) | 390(Win)/423(Lin) | 裸 Fiber+WaitGroup(1)+slot | 338/415 | 0.87x/1.02x | (糖 vs 自有裸机制) | n/a | n/a | ➖**层开销 Win ~52ns/Lin 平局**:糖多 2 次堆分配换调用方零生命周期管理;初测单任务 ~1.3x/~2x 是 OS 唤醒抖动非层开销,批量摊销后现原形 |
| **sched** when_all 64 任务(sched-10) | 372(Win)/467(Lin) ns/任务 | (= 逐个 join) | — | ~1.0x | (HYLE 自有) | n/a | n/a | ➖群等待 vs 逐 join 零额外;when_any(4) ~2x 快于 when_all(4)(取第一即返) |
| **io** echo 吞吐 shard-per-core 4w(io-04) | ~1.63M/s(611ns/rt) | (无 std 异步 IO) | n/a | n/a | 裸 epoll 1 线程就绪循环 / asio-1w | 743K / 618K/s | 2.25x / 2.74x | ✅**赢裸 epoll 2.25x·asio 2.74x**(n=15 within-run,gap 125/174%»spread27%)。**扩展 1w→4w ~2.2x**(2w 1.48x);shard-per-core 多核开箱免费,单 io_context 框架够不着 |
| **io** echo 吞吐 shard-per-core 1w(io-04) | ~748K/s(1337ns/rt) | (无 std 异步 IO) | n/a | n/a | asio(io_uring)/libuv/裸 epoll | 618K/505K/743K/s | 1.25x/1.52x/平 | ✅**单核赢成熟抽象框架 asio 1.25x·libuv 1.52x**(reinject 同核短路+COOP 后 n=15 within-run:gap 25/52%>spread)/ ➖**平裸 epoll**(1.026,gap2.6%«spread20%:零抽象就绪循环无 fiber 固定开销,追平即好,单跑"反超"是噪声)。优化前 vs asio 本是统计平局,-33% 推成单核真赢 |
| **io** pipelined 高扇出 512 流(io-04) | 328K→1.71M/s(1w→8w) | (无 std 异步 IO) | n/a | n/a | (自身扩展性)| n/a | 5.2x | ✅**扩展 5.2x**。此负载(512»ring 深度 256)揪出并修复一个 **P0 hang**:worker 空闲的 reap(0) 从不带 GETEVENTS 进内核→>ring 深度在途 op 的 completion 永不收割;修=park 边界有在途 op 阻塞 GETEVENTS reap——详见 [debug 专篇](../debug/io-pipelined-hang-hunt.md)|
| **io** 连接生命周期 accept+serve+close(真实短连接) | ~240µs/conn(4162 conn/s) | (无 std 异步 IO) | n/a | n/a | (自身:ephemeral→persistent 谱系) | n/a | n/a | ➖**每连接固定成本 ~240µs(accept+握手+close)主导**,R=1→16 round-trip 仅 +~2.6µs/轮=echo 稳态往返(填 accept_bench 即弃/echo_bench 预建之间的真实使用缺口;正是 keep-alive/连接池要摊薄的那块)。双平台(Win IOCP ~2ms)+ TSan 净 |
| **io** multishot recv awaitable vs one-shot(io-07b) | 165,679 ns/dgram | (无 std) | n/a | n/a | HYLE one-shot RecvOp(自身对照) | 166,985 | 0.998 | ➖**统计平局**(gap0.2%«spread13.4%,n=15):同核 lockstep ~1 op 在途,fiber 往返主导、multishot 拿不到"一批完成同到"来摊薄省下的 submit。**非回归、待承重**(需突发/高扇出负载显优势,登记未来;正确性+TSan+回归已验) |
| **io** DEFER_TASKRUN vs COOP_TASKRUN(io setup flag) | 1.35-1.43µs(1w) | (无 std) | n/a | n/a | COOP_TASKRUN(自身对照) | 1255ns(1w) | ~同量级 | ➖**同量级无显著提升**:DEFER 收益=省完成 task_work 的 IPI,单 worker loopback 本就每次 drain 都 enter ring、几乎无 IPI 可省。**正确接入(R_DISABLED 生命周期修 + kCancelToken latent bug)、不回归、机制就位待高 syscall 压力/多核争用显收益**(opt-in 默认 COOP) |
| **io** registered/fixed files(IOSQE_FIXED_FILE,io-07c) | +0.8-3.3% | (无 std) | n/a | n/a | 裸 fd(自身对照) | 基线 | 1.008-1.033x | ➖**边际正收益**(socketpair ping-pong 3 跑一致为正、淹在 1-3% err% 边缘):省内核每 op fget/fput 是真的,但 loopback 上 fd 查表只占往返极小一块 |
| **io** direct-accept vs normal accept(io-07c) | 合并慢~1%/accept 相快~10-30ns | (无 std) | n/a | n/a | normal accept(自身,拆相分离) | — | accept 段 direct 快 | ➖**非回归**(拆相追根 0.7%问题):**accept 原语本身 direct 反而快 ~10-30ns/conn**(省 fget/fput,合理);合并 bench 那~1%全是 teardown 不对称伪影(direct close_direct+reap+FILES_UPDATE 2 syscall 其一冗余 vs normal ::close 1 syscall),生产不即建即关故不出现。primitives-only 不改 TcpStream |

**总账小结（两把标尺,2026-06-30 全套真跑；simd dot 2026-07 更新）**：**比 std**——HYLE 几乎全线大赢(分配器 8–23x、hashmap 2.6–6.9x、SIMD dot 15x/saxpy 2.2x、容器 5–39x、Rng 1.5–1.7x、SpinLock 3x);唯几项亚 ns 测量地板或 Win-only codegen 残留(monotonic_ns Win 0.68x/Linux 平)及 scalar 反向串族(rfind/count,可上 memrchr)。**比 SOTA**(真门槛,absl/mi/Highway/rigtorp/marl/tbb/xsimd/simdutf)——✅**赢 ~36 项**、➖**平 ~16 项**(噪声内追平)、❌**仍输少数项**且全附根因:① raw hash_bytes 64B=导出接缝 by-design(同算法 XXH3 vs 裸 XXH3 ~1.28x,>16B 出线核不可内联;≤16B 走 inline mix 无 seam,改 abseil 单乘后 16B key 赢 wyhash 2.2x);② math exp/tanh 输 xsimd ~1.2x=跨库 best-of-breed 常数(已平 Highway);③ monotonic_ns **Win-only 输 1.48x、Linux 平**(QPC split 算术 vs MSVC steady_clock,已查证非接缝非算法;POSIX clock_gettime 与 glibc steady_clock 同码故 Linux 平);④ MemoryArena/ObjectPool 单线程=**选错工具已下 ledger**(单线程对象池应用 GrowablePool;多线程共享场景对真强 SOTA `oneTBB scalable_allocator` **赢 ~3x**(对 std 弱池 synchronized_pool 25–44x,后者非天花板),非输项);⑤ string_ref scalar 反向/否定族(rfind/find_*_not_of/count)输 std::string_view 4–20x=scalar vs libc memrchr/调优 bitmap,**待优化项(可上 SIMD,非缺陷)**;⑥ string_algo utf8 默认 scalar 输 simdutf 2x(simdutf 核 vendored 无条件默认追平)。**阶段C 已扭转**:string_ref 前向搜索簇(find char 20x→平、find_first_of 36x→1.6x、starts_with 2x→平、split 反赢)、ConcurrentVector 并发读(2.15x 输→2.2x 赢)、Ref(反汇编证算法同工、实战打平,microbench 假象)、hash 8B、vtable(反超 std::pmr)、math log(反超两库)、单环队列 4P(消除 1.33x 落后到平 rigtorp+moody)、StringBuf intern(反超同契约 peer)。**真复杂度/争用问题(MemoryArena 278x、Arena 10x、4P CAS+伪共享、CV per-read acquire 争线)全部根治**。**遗留优化(非覆盖缺口)**:error 根模块已补测(对标同款 SOTA absl 全线平或赢:inline 码赢 4.6x、带消息赢 1.92x、ERROR 上抛赢 2.11x,空间 Result 16B<std/absl 24B;OK 传播"输 std"经隔离探针+消费变体证为 microbench 假象);string_ref scalar 反向串族可上 memrchr/SIMD。

**io 段补记(2026-07-05,reinject+COOP 优化后 n=15 重测)**:HYLE echo-1w **单核反超成熟抽象框架 asio 1.25x / libuv 1.52x**(优化前是统计平局,-33% 推成真赢)、**平裸 epoll**(零抽象就绪循环,追平即好);多核 shard-per-core 开箱大赢。新增**真实连接生命周期**基准(accept+serve+close):每连接固定成本 ~240µs 主导、每 round-trip 仅 +~2.6µs——量化了 keep-alive/连接池的价值。io 机制项(multishot awaitable / DEFER_TASKRUN / registered-fixed-files / direct-accept)均为**正确接入、loopback 上统计平局或边际**、"就位待承重"——它们的收益需 loopback lockstep 造不出的突发/高扇出/高建连率负载,已诚实登记为待承重(非回归、非缺陷);direct-accept 的"慢 1%"经拆相追根证为 teardown microbench 伪影(accept 原语本身 direct 更快)。
