# sched 实测铁证

> 本页是 [performance 总账](./README.md) 的 sched + platform + queue 详节:fiber 切换、platform 原语、fiber-aware 同步(marl 对标)、lock-free 队列簇。环境:Win MSVC19.50 + Linux GCC14.2 Release(原生 ext4),nanobench。

## fiber 切换与 spawn（`bench/sched/scheduler_bench.cc`）

| 项 | 实测(Linux GCC14) | 参考 |
|---|---|---|
| fiber 上下文切换(resume/yield 对) | **~21-23 ns**(本 session 复测 21.0ns) | fcontext 量级,= boost.context 原语本身(零封装开销) |
| spawn 2000 任务 + drain | vs `std::async` 快 144x(**弱基线**)→ **真 SOTA 见下 sched 同步节:vs marl 快 2.17x** | std::async 每任务起 OS 线程(稻草人);真标尺是 marl 调度器 |

> **注**:spawn 的权威数字在下方 sched 同步节(对标 marl,非 std::async)。此处保留 std::async 行仅作"赢最低门槛"的历史参照。

> sched bench(`bench/sched/scheduler_bench.cc`)**现已编入默认 build**(提交后解封)。
> 它曾被认为触发一个 P0 崩溃,但一轮穷尽调查(见 [`../architecture/debug/ki-1-crash-hunt.md`](../architecture/debug/ki-1-crash-hunt.md)
> + [`../architecture/known-issues.md`](../architecture/known-issues.md) KI-1)**排除了 HYLE 逻辑缺陷**:TSan 无源码级
> race、MI_DEBUG=3 全量堆校验无损坏、原始崩溃器本体在 Linux 原生 + Windows MSVC 本机
> 重负载下 + 强制迁移 687 万次跨线程 free 均**不复现**。证据(SIGILL = 代码页问题 +
> 全部 instrumented 净 + 原始崩溃全在 9p)指向 **9p/WSL demand-paging 环境假象**,非
> HYLE 缺陷。fiber 切换 ~21-23ns 数值有效(mimalloc v2.3.2 + 本 session 复测一致);spawn 的权威对标见下 sched 同步节(vs marl 2.17x,非此处历史的 vs std::async)。

## platform 原语（Parker park/unpark、Thread、directory_iterate，`platform_bench.cc`）

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

## sched fiber-aware 同步（CV/Mutex/WaitGroup/Channel vs marl，`sync_bench.cc` / `sync_marl_bench.cc`）

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

## lock-free 队列簇（`queue_sota_bench.cc` + crossbeam Rust 对照，task #9 / bench-02b）

> 环境:Win MSVC19.50 Release + Linux x86_64 GCC14.2 Release(原生 ext4),nanobench。cap=1024、50000 item/生产者。**SOTA 标尺**:moodycamel ConcurrentQueue(单头,gated `-DHYLE_BENCH_MOODYCAMEL`)、oneTBB concurrent_bounded_queue/concurrent_vector(gated `-DHYLE_BENCH_TBB`,apples-to-apples 有界)、crossbeam-deque(Rust,work-stealing 同族,`bench/rust/`)。**此前这簇零对标("覆盖低就默认 SOTA"),本节是真对标揪出的结果——赢/平/输都登记。**
>
> **数字 = 双平台各 15 遍 `bench-stat.py` 中位数(2026-07 升级)**:此前是"3 遍手动均值"(SOTA 库未 clone);clone rigtorp+moody 到 sibling(Win)/`/root`(WSL ext4)后按统计纪律重测——**gap > 两变体 spread 之和才判赢/输,否则统计平局**。这修正了旧账"4P 反超 rigtorp"的过度声明(见结论 1)。
>
> **对标依赖 `project-hyle/` sibling 布局**:gated 库默认在仓库 sibling(`../MPMCQueue`/`../moodycamel-ConcurrentQueue`/`../oneTBB`),crossbeam Rust crate path 依赖本地 clone。换机器时:C++ 侧 `-DHYLE_RIGTORP_DIR=/path`/`-DHYLE_MOODYCAMEL_DIR=/path` 覆盖;Rust 侧改 Cargo.toml path 或换 `crossbeam-deque="0.8"` 走 crates.io。

| 场景 | HYLE (Win/Lin) | rigtorp(同约束 SOTA) | moodycamel | 判定(15 遍,双平台) |
|---|---|---|---|---|
| **单线程 fill+drain**(BoundedQueue) | 7.77/7.79 | 11.54/11.84 | 9.48/8.41 | ✅ **赢 rigtorp 双平台 1.49x/1.51x**(gap»spread);moody **Win 赢 1.23x、Lin 平**(gap7%«spread18%) |
| **MPMC 2P/2C**(BoundedQueue 单环) | 56.08/55.97 | 61.04/57.88 | 73.90/61.69 | ➖ **双平台平 rigtorp+moody**(vs rigtorp gap 7.5/3.5%«spread 26/29%) |
| **MPMC 4P/4C**(BoundedQueue 单环) | 69.26/69.10 | 78.41/77.89 | 68.49/67.01 | ➖ **双平台平 rigtorp+moody**(vs rigtorp gap 14.6/12.1%«spread 23/29%;vs moody 双平台亦平,HYLE≈moody) |
| **MPMC 2P/2C**(**ShardedQueue**) | 29.53/~ | 61.04/57.88 | 73.90/61.69 | ✅ **Win 赢 rigtorp 1.93x**(gap48%»spread13%);drops 全局 FIFO |
| **MPMC 4P/4C**(**ShardedQueue**) | 25.31/14.60 | 78.41/77.89 | 68.49/67.01 | ✅ **赢 rigtorp Win 3.27x/Lin 5.3x**(gap»spread),但 spread 21/43% 大(采样不足,见诚实边界③) |
| **work-steal owner LIFO**(WorkStealingDeque) | 3.06/2.99 | (无同物) | crossbeam 3.6(Win) | crossbeam 参照,HYLE 略快(跨语言,仅量级) |
| **work-steal owner FIFO**(WsFifoQueue) | 3.93/3.92 | — | crossbeam 4.4(Win) | crossbeam 参照,HYLE 略快 |
| **ConcurrentVector** push_back/read | 4.51/0.93 | — | tbb 5.9/1.3 | ✅ 赢 tbb push 1.3x / read 1.1x |

**核心诚实结论 + 它驱动的真优化**:

1. **单环 BoundedQueue 4P 曾输 rigtorp 1.33x——纠正两次误判,真根因"抄作业没抄全"(cell 未对齐 cache line)+ 过度 backoff,已修到平**。**误判史(都已证伪)**:① 最早称"严格-FIFO+有界 MPMC 单点 serialize 是理论墙、单环已是该约束 SOTA、rigtorp/folly 同款"——rigtorp 实测证伪;② 随后**猜**根因"HYLE Vyukov CAS-重试 vs rigtorp fetch_add 票号"——读 rigtorp 源码**再证伪**:rigtorp 的 `try_push`/`try_pop`(实测路径)也用 `compare_exchange`(只有阻塞版 `push`/`pop` 用 fetch_add),热路径同是 CAS。**真根因(读码查实)**:rigtorp 把每 slot **对齐独立 cache line**(`alignas(64)`+`sizeof%64==0`),HYLE 的 `Cell{atomic<u64> seq; T data}` 对 8B T 仅 16B → 4 cell 挤一条 line → 相邻槽被不同线程并发读写时**伪共享**——clay 移植漏掉的 Vyukov 经典调优(根因①)。另:HYLE 每次 CAS 失败调 `cpu_relax()`(~40cyc),4P 高争用纯增延迟(rigtorp/folly 的 try_ 不 backoff)。**修复**:`Cell` 加 `alignas(HYLE_CACHE_LINE_SIZE)` + 去 CAS-失败 backoff。**双平台 15 遍统计:单线程赢 rigtorp 1.49x/1.51x(gap»spread,铁赢)、2P/4P 统计平局**(4P vs rigtorp gap Win14.6%/Lin12.1%«spread 23-29%;vs moody 双平台亦平,HYLE 69.3≈moody 68.5/67.0——高争用 MPMC 微基准方差大,不足以判"赢")。**诚实结论:单环 = 单线程赢 rigtorp、多线程平 rigtorp+moody**(不再声称"反超"——旧账 3 遍手动均值的 1.08x 落在噪声带内,15 遍统计法归为平局)。4P 落后已消除,同约束 SOTA 全点位平或超。代价:buffer 增至 capacity×64B(空间换吞吐,rigtorp 同付)。**6/6 单元 + WSL TSan 32/32 双平台过(lock-free 闸门,含 MpmcStressConservation 4P/4C 守恒,cache-line pad 布局改动零 data race)**。
2. **ShardedQueue 仍是生产用途首选**:sched `global` 注入无全局序需求,ShardedQueue(N 子环)4P **赢 rigtorp Win 3.27x/Lin 5.3x**、moody 同量级倍数,线性扩展(15 遍 gap»spread,但 spread 21-43% 偏大——采样不足,见诚实边界③)。单环用于真需严格全局 FIFO 的场景,现单线程赢、多线程平同约束 SOTA。
3. **backoff 是 2P-vs-4P 权衡(诚实记账)**:`cpu_relax` backoff 在 2P 低争用曾给更大领先,但 4P 高争用造成 1.33x 落后;去掉后 2P/4P 都回到平 rigtorp。directive 要"无落后",故选无 backoff(canonical rigtorp/folly 同样 try_ 路径不 backoff)。

> **诚实边界**:① ShardedQueue 赢幅(Win 1.93-3.27x/Lin 5.3x)远超误差带(gap»spread,双平台 15 遍)。② 跨语言(crossbeam Rust)只下"同量级/有竞争力"。③ **单环 MPMC 2P/4P 是高方差微基准(spread 23-40%),双平台各 15 遍 `bench-stat.py` 统计**:单线程铁赢 rigtorp、2P/4P 平 rigtorp+moody(中位数偏赢但 gap«spread,不判"赢")。旧账"3 遍手动均值 4P 反超 1.08x"已被 15 遍统计法修正为平局——"反超"是噪声带内的乐观读数。cache-line pad + 去 backoff 确实消除了原来的 1.33x 落后(那是真结构缺陷),但没到"反超",到"平"。**ShardedQueue 本身 spread 偏大(Win 21%/Lin 43%):`bench_mpmc` 的 `minEpochIterations(50)` 对快 20 倍的 sharded 采样不足(nanobench 单遍即自报 "Unstable ~54 iters, 建议 543"),故 sharded 倍数取"量级铁赢"而非精确点值——这是待修的 bench 采样缺陷,非算法问题。** ④ work-stealing owner 路径单线程无争用(两侧同样)。⑤ **lock-free 闸门:WSL TSan(setarch -R,GCC14 RelWithDebInfo)队列族 32/32 零 data race**——cache-line pad 改了 Cell 布局,TSan 复验 MpmcStressConservation(4P/4C 守恒)等压力测试确认无 race。

## 分层时间轮 vs min-heap（`bench/time/timer_wheel_bench.cc`，time-02）

> HYLE `TimerWheel`(6 层 × 64 槽 tokio 移植)对标 **min-heap**(Go/libev/libuv 的定时器选择)=同职算法对照。folly HHWheelTimer 焊死在 EventBase/libevent + glog + boost 上,链进 microbench 测的是事件循环不是轮,故对标算法本身(设计文档争论的正是"轮 vs 堆")。堆按真实行为建模:无 O(1) keyed 删除 → **惰性 cancel(墓碑标记)** + expire 跳墓碑。N=4096 timer/op,双平台各 15 遍 `bench-stat.py`(gap > 两变体 spread 之和才判赢/输)。

| 工作负载 | HYLE 轮(Win/Lin ns/op) | min-heap(Win/Lin) | 判定(15 遍,双平台) |
|---|---|---|---|
| **churn(arm N + 取消 70% + expire)= 真实 agent 画像** | 12.60/11.83 | 26.28/38.66 | ✅ **双平台铁赢 2.09x/3.27x**(gap 108/226%»spread 10.5/27%) |
| arm+expire(全触发) | 23.09/16.97 | 25.51/38.73 | Win 统计平局(1.09,gap9%«spread12%)/ Lin **赢 2.25x**(gap125%»spread16%) |
| arm+cancel(纯取消,无 expire) | 7.39/5.57 | 7.42/2.94 | Win 统计平局(1.00)/ Lin **输 0.53x**(gap47%»spread30%) |

**核心诚实结论**:

1. **真实负载(churn)双平台铁赢 2-3x**。这是北极星画像——百万 agent"arm 超时 → 响应先到 → 取消 timeout → 少数真到期"。堆的惰性墓碑要活着穿过 expire(文档预言的 add-then-cancel churn 堆膨胀),轮在 cancel 时 O(1) 摘链已清走。

2. **`arm+cancel` 纯取消 Linux 输 2x 是微基准假象,如实记账不藏**(对齐 mem 红队方法论③"对照不公平绝不当免死金牌")。根因:此 bench 每 op 丢弃一个全新堆,堆的墓碑 `vector<bool>` 位标记比轮的摘链 + 反算 slot 便宜——**但 bench 在墓碑必须被走之前就把膨胀的堆扔了**。真实负载墓碑要穿过 expire(churn 行),轮双平台就赢。这是"切片微基准掩盖对手真实成本"的典型:纯取消孤立看堆便宜,但那是它把 cleanup 债留到了 expire(churn 里还)。堆唯一真优势"O(1) 拿最近到期"对我们不重要(单调钟批量 advance、不需精确 min)。

3. **平台分歧(又一"单平台数陷阱"印证)**:arm+expire 与 arm+cancel Win/Lin 判定不同(Win 更平、Lin 分化更大),MSVC 与 GCC 对 `std::priority_queue`/`countl_zero`/`rotr` codegen 差异所致——**单平台数搬总账会误判**(同 dtype/time/path/queue)。churn 方向双平台一致(轮赢),是稳健结论。

> **诚实边界**:① 时间轮**单线程数据结构**(per-worker,无锁),非 lock-free——无 TSan 闸门需求;新增的并发面是 `Parker::park_timeout`(见 platform 节,TSan 5×净)。② 未对 folly HHWheelTimer 实测(EventBase 耦合,非同职对照;算法层已对 min-heap)。③ tokio 是 Rust,未跨语言实测,取其算法参数(6×64/1ms)移植 + 对 min-heap 验证同族优势。④ arm+cancel 微基准的堆"赢"是采样切片,churn 才是负载真相——ledger 不把纯取消孤立数当"堆更优"结论。
