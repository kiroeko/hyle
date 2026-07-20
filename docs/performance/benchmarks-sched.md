# sched 实测铁证

> 本页是 [performance 总账](./README.md) 的 sched + platform + queue 详节:fiber 切换、platform 原语、fiber-aware 同步(marl 对标)、lock-free 队列簇。环境:Win MSVC19.50 + Linux GCC14.2 Release(原生 ext4),nanobench。

## fiber 切换与 spawn（`bench/sched/scheduler_bench.cc`）

| 项 | 实测(Linux GCC14) | 参考 |
|---|---|---|
| fiber 上下文切换(resume/yield 对) | **~21-23 ns**(本 session 复测 21.0ns) | fcontext 量级,= boost.context 原语本身(零封装开销) |
| spawn 2000 任务 + drain | vs `std::async` 快 144x（**弱基线**）→ **真 SOTA 见下 sched 同步节：worker-stack `schedule` 三平台对 marl 为 Win/Lin 明确赢、M4 统计平局** | std::async 每任务起 OS 线程（稻草人）；真标尺是 marl 调度器 |

> **注**:spawn 的权威数字在下方 sched 同步节(对标 marl,非 std::async)。此处保留 std::async 行仅作"赢最低门槛"的历史参照。

> sched bench(`bench/sched/scheduler_bench.cc`)**现已编入默认 build**(提交后解封)。
> 它曾被认为触发一个 P0 崩溃,但一轮穷尽调查(见 [`../debug/ki-1-crash-hunt.md`](../debug/ki-1-crash-hunt.md)
> + [`../architecture/known-issues.md`](../architecture/known-issues.md) KI-1)**排除了 HYLE 逻辑缺陷**:TSan 无源码级
> race、MI_DEBUG=3 全量堆校验无损坏、原始崩溃器本体在 Linux 原生 + Windows MSVC 本机
> 重负载下 + 强制迁移 687 万次跨线程 free 均**不复现**。证据(SIGILL = 代码页问题 +
> 全部 instrumented 净 + 原始崩溃全在 9p)指向 **9p/WSL demand-paging 环境假象**,非
> HYLE 缺陷。fiber 切换 ~21-23ns 数值有效(mimalloc v2.3.2 + 本 session 复测一致);spawn 的权威对标见下 sched 同步节（拆分 worker-stack 与 stackful 合同，三平台对 marl），非此处历史的 vs std::async 弱基线。

## LIFO fast-slot：同 worker 唤醒 cache-hot 恢复（复杂度机会①，`scheduler.cc`）

> 移植 tokio `lifo_slot`。**背景**:sched-02 的 `RunQueue` 本就有 LIFO 槽(`kMaxLifoPolls=3`),但迁到 steal-safe `WsFifoQueue` 时丢了——生产 scheduler 每次跨 fiber 唤醒(`reinject`)都经 global 分片 CAS + `notify()` futex 弹到别的核,即便唤醒方就在某个 worker 上跑。给 `Impl::Worker` 加回单 fiber LIFO 槽(纯字段,仅本 worker 线程碰→天然 steal-exempt + 无原子):`reinject` 在 worker 线程上唤醒非 pinned fiber 时投进本 worker 槽,下轮 `find_task` 在本核 cache-hot 立即恢复,跳过 CAS+futex。优先级 `pinned→lifo槽→local→global→steal`,`kMaxLifoPolls=3` 反饥饿(超额降级 local ring)。

| 项(1 worker suspend/wake ping-pong) | 基线(无槽) | LIFO 槽 | 判定 |
|---|---|---|---|
| Win MSVC Release(各 3 轮) | ~240 ns/往返 | ~163 ns/往返 | **延迟 −32%** |
| WSL2 GCC Release(各 3 轮) | ~186 ns/往返 | ~121 ns/往返 | **延迟 −35%** |

- **收益来源**:一次唤醒省掉 global-shard 的 CAS enqueue + `notify()` 的 seq_cst fence(常态还免了一次 futex 系统调用),且被唤醒 fiber 在**同核**恢复——生产者刚写的 buffer 还在 L1/L2。ping-pong / channel send / RPC reply / io echo 这类"A 唤 B、B 唤 A"的往返模式最受益。
- **无回退**:LIFO 槽只拦截 `reinject`(唤醒路径),spawn 走 `local.push_back`(不变)——spawn 吞吐仍 339ns/任务(vs std::async 69x)。work-stealing 扩展仍近线性(1→2→4→8 = 1.9x/3.5x/5.4x),槽是 worker 私有、偷取路径看不到,不动 Dekker/CAS 反羊群数学。
- **正确性**:槽是单线程字段(仅本 worker 填/取),worker 绝不会带着非空槽去 park(`find_task` 先清空),故无跨线程访问。**WSL `setarch -R` TSan:102/102 全绿、0 warning**。cooperative yield 走 `reschedule` 不经此路,tokio"yield 不入槽"公平规则自动满足。


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

> 环境：Win MSVC19.50、WSL GCC14.2 原生 `/root/hyle`、Apple M4
> AppleClang21 Release，均为 `bench-stat.py -n15`。**SOTA 标尺 = marl**。
> 最终形状同时修正了两个旧错误：HYLE 完整有栈 spawn 不再和 marl 短 callback
> 混比；marl 不再 `bind()` benchmark caller（否则“4 worker”实际允许第 5 个
> caller fiber 执行），改为直接 `Scheduler::enqueue`，严格 4 对 4。

| 同合同项 | Windows：HYLE / marl | WSL：HYLE / marl | M4：HYLE / marl | 判定 |
|---|---:|---:|---:|---|
| **争用 Mutex**（8 task/4 worker） | 18.687 / 24.372 ns | 28.179 / 28.283 ns | 12.819 / 24.560 ns | Win/M4 赢；Linux 统计平局 |
| **WaitGroup + schedule**（1024 task） | 192.510 / 297.790 ns | 220.770 / 337.420 ns | 253.980 / 252.750 ns | 配对中位 Win **1.553× WIN**、Lin **1.509× WIN**；M4 **1.003×，统计平局** |
| **schedule 2000 noop** | 192.600 / 297.530 ns | 218.440 / 327.030 ns | 252.120 / 252.720 ns | 配对中位 Win **1.534× WIN**、Lin **1.480× WIN**；M4 **1.007×，统计平局**（12/15 HYLE 更快） |
| **spawn_detached 2000** | 342.276 ns | 355.471 ns | 396.352 ns | 完整有栈腿绝对成本，不与短 callback 裁决 |
| **spawn_task 2000** | 415.130 ns | 413.570 ns | 434.416 ns | 再含 coroutine + Join result state |

旧的无争用 Mutex、Channel、CV 专项行仍保留在各自 bench；本表只放本轮重新跑过
的三平台 SOTA 主证书。

**Mutex = 平台快锁，不是 fiber-aware。** 短临界区让出 fiber 曾慢 marl 5.8×，
因此 Windows 使用 SRWLOCK，POSIX 使用紧凑的三态 address-wait 锁。收口狩猎又
发现竞争解锁旧路径先 `2→1` 再写 `0`，每次 handoff 写两次 cache line；改成一次
`exchange(0)` 后按旧值决定 wake。最终 Windows/M4 明确赢、Linux 与
`std::mutex` 平。仍不加 userspace 预自旋：M:N 下 holder 可能是同 worker 队列
里的另一 fiber，spinning 会饿死它。Mutex 持有期间不得 yield/阻塞；长等待用
fiber-aware CV。

**无争用 Mutex "慢 std" 是 glibc 单线程假象——双平台 15 遍实测揭穿(又一"同现象双平台方向相反"陷阱)**:HYLE futex Mutex 与 std::mutex 算法同级(各 1 locked RMW)。双平台 15 遍 bench-stat 后现原形——
- **Linux 单线程 bench 输 0.35x**(HYLE 6.9ns vs std 2.4ns):glibc 用 `__libc_single_threaded` 让 std::mutex 在单线程进程里**跳过原子**,假性快到 2.4ns;HYLE 不做单线程特化,恒付一个 RMW。
- **但这是 microbench 骗局,不是真差距**:MT 变体加一个后台线程让进程变多线程,std::mutex 立刻**不能再跳原子** → 2.4→**10.1ns**,HYLE 纹丝不动 **6.9ns**——**Linux MT 下 HYLE 反超 std::mutex 1.47x**(gap47%>spread29%)。mutex 唯一真正被用的场景就是多线程,HYLE 的 Drepper 3 态比 pthread_mutex_t 更轻;marl mutex=std::mutex,故也赢 marl。
- **Windows 无此假象**:MSVC 的 std::mutex(SRWLOCK)没有 `__libc_single_threaded` 式单线程特化,单线程 bench 里 HYLE 就已**反超 1.15x**(10.6 vs 12.1ns)。Win 的 MT 变体因 keepalive 后台线程抢核使 spread 爆到 79%/64%(测量噪声,非算法),判平局——**Win 看单线程行即可,MT 行在 Windows 无诊断价值**。
- **结论 + 教训**:不是性能问题,是"glibc 单线程 microbench 假象",且**同一现象双平台方向相反**(Lin 单线程假输/MT 真赢;Win 单线程真赢/MT 噪声)——正是反复出现的"单平台数陷阱":只看 Linux 单线程会误判"输 std",只看 Win 又看不到 glibc 这个坑。双平台 + 单线程/MT 双变体一起看才是真相。

**派发 vs SOTA：拆合同后 schedule 明确赢，完整有栈腿单报成本。**
`scheduler_bench` 的 std::async 仍只是弱基线，不参与 SOTA 结论。公平的短 callback
对照是 HYLE `schedule` vs marl enqueue。`InlineJob` 的用户 payload 仍是
`{fn,ctx}` 两字并按值入队，不再每任务 malloc/free owning record；内部另带一个
accounting-shard tag。Fiber 与 InlineJob 分开计数，close 竞态仍由
open→increment→recheck 保证。

M4 初测的 0.904× 真输最终定位为**单一完成计数 cache-line 争用**：一个提交者
`fetch_add`，四个 worker `fetch_sub`，在 ARM 上反复抢同一行。把计数按 worker
分片并独占 cache line、每个 shard 从 0→1 时才 notify，再把不可挂起 callback
以最多 8 个为公平预算批量消费，M4 15 遍回到 1.007× 统计平局；Windows/Linux
仍明确赢。ShardedQueue 同时把 power-of-two shard 的取模改为 mask，pop 扫描改为
自增回绕。对象池和短锁 ring 两个候选实测更差并撤回；最终保留的是有根因、有
三平台和 TSan 证据的改动。

**WaitGroup fork-join：conditional-notify 根治真瓶颈，schedule 腿现已明确赢。**
初版 5.4µs vs 0.30µs 的 17x 真输来自急泊车 + 无条件 unpark 的 futex 风暴，
不是栈池；conditional-notify 的 Dekker 握手修复仍成立。旧测试用完整有栈腿得到
统计平局；新同合同 worker-stack shape 在 Win/Lin 赢、M4 统计平局。`done()`
也改为非末次纯原子，只有 1→0 完成者进入 waiter-publication 锁。教训从“单遍
不可判”进一步升级为“统计纪律之前先统一任务合同和实际参与 worker 数”。

> **TSan × fcontext(本步另一修复)**:调度器唤醒优化让 fiber 跨 worker 迁移更频繁,撞出 TSan 对 fcontext 栈切换的盲区(SEGV in libtsan,非逻辑 bug——无 TSan 50/50 过)。已给 `fiber.cc` 加 `__tsan_switch_to_fiber` 注解(每次切换告知 TSan 当前逻辑上下文,迁移时按 resume 重新捕获 resumer 句柄),**release 零成本**(非 TSan 构建宏全展开为空)。修复后双平台 28/28 + TSan 28/28 零 race,lock-free 闸门恢复。

## 定时 CV 等待 wait_for/wait_until（sched-04c thread-mode，`sync_bench.cc` 块 8/9）

thread-mode(裸线程 waiter,外部提交者/跨进程联邦)。对标 `std::condition_variable::wait_for`=直接 SOTA(两者都 futex-park OS 线程到 deadline)。双平台各 15 遍 `bench-stat.py`。

| 场景 | HYLE(Win/Lin) | std::cv::wait_for(Win/Lin) | 判定(15 遍,双平台) |
|---|---|---|---|
| **notify 早于 deadline 的唤醒往返** | 580 ns / ~ | 1165 ns / ~ | **Win 赢 2.0x**(gap99.8%»spread47%)/ **Lin 平**(ratio1.03,gap3.4%«spread21%) |
| **纯超时(1ms,无 notify)** | 15.58 ms / — | 15.58 ms / — | 双平台**平**(gap0.1%«spread):被 OS 定时器粒度地板 |

- **唤醒往返 Win 赢 2.0x / Lin 平——又一"单平台数陷阱"(同 Parker/monotonic)**:Windows 的 `std::condition_variable` 是 SRWLOCK+CONDITION_VARIABLE 往返较重,HYLE 的 SpinLock monitor + 精简 `park_timeout` 路径胜出;Linux glibc 的 cv 本就是薄 futex 封装,HYLE 打平(与 sched-05 "Parker Linux 平 CV" 同源)。**只报单平台会误判,双平台才是真相**。
- **纯超时双平台平**:HYLE 与 std 都被 OS ~15ms(Win)定时器粒度地板,gap 0.1%——**证 HYLE 超时路径零额外开销**(不是"赢/输",是"没引入税")。这正是想验证的:双向链表 O(1) 自摘 + `linked` 仲裁不给热路径加负担。
- **正确性优先于跑分**:此件核心是 notify-vs-timeout 竞态无 lost-wakeup(红队 6 类审计 + TSan×3 含 NotifyVsTimeoutRace 压力),性能是"不劣于 std 且 Win 更优"的附带确认。

## sched-09 并发原语补全：Semaphore / RwLock / oneshot / watch / broadcast（`bench/sched/sync_ext_bench.cc`）

sched-05 的 `sync_bench` 覆盖 Mutex/Channel/WaitGroup/Latch/CV;sched-09 新增的 5 个 tokio 语义原语(~1.5k 行)此前零性能测试,这里补齐。都建在 sched-05 的 fiber-aware CV 地基上,故热路径成本应与那层同量级。双平台 Release(Win MSVC / Lin GCC14 io_uring 宿主)。

| 场景 | HYLE(Win / Lin) | 对照 | 判定 |
|---|---|---|---|
| **Semaphore try_acquire+release**(无争用快路径) | 10.1 / 10.6 ns | — | 限流器热路径:单 SpinLock 下的 permit 增减,~10ns |
| **Semaphore acquire/release**(8 fiber/4 worker 争用,permits=1) | 418 / 470 ns | — | 含阻塞 acquire→release-唤醒-队头 的 fiber 往返 |
| **RwLock read()**(无争用) | 11.1 / 10.8 ns | std::shared_mutex 9.8 / 11.8 ns | **Lin 微赢 std 10%、Win 88%**:fiber-aware 读锁不输线程级锁(建在 Semaphore permit 上) |
| **RwLock read() x8**(8 reader fiber/4 worker) | 85 / 80 ns | — | 并发读:permit 共享不互斥 |
| **oneshot make+send+recv**(请求/响应槽往返) | 166 / 168 ns | — | 全程建 channel+发+收+析构;send 不阻塞、recv 拿已置值不让出 |
| **watch send+borrow**(最新值路径) | 56 / 55 ns | — | 版本号 bump + borrow 最新值(配置热重载形态) |

**broadcast 扇出(1 send → N receiver 各 recv,扫 N)**:

| receivers | ns/delivery(Win / Lin) | delivery/s |
|---|---|---|
| 2 | 853 / 1059 | ~1M |
| 8 | 221 / 419 | ~2-4M |
| 32 | 82 / 101 | ~10M |

- **ns/delivery 随 N 下降是健康的**:一次 send 唤醒 N 个游标,send 的固定成本摊到 N 个 delivery——边际投递成本远低于固定成本(2→32 receiver,per-delivery 从 853 降到 82ns),扇出线性无超线性惩罚。
- **RwLock 读侧不输 std::shared_mutex**:HYLE 的 RwLock 是 fiber-aware(建在 Semaphore=写抢全 permit/读抢 1),但无争用读锁开销与线程级 `std::shared_mutex` 同量级(Lin 甚至微赢),证"写优先公平"不以读侧性能为代价。与 sched-05 "Mutex 用 futex 打平 std/marl" 同源:fiber-aware 只在真需让出时才贵。
- **正确性已由 sched-09 的 sync_test + WSL TSan 覆盖**(14 测试零 race);本 bench 是性能测量,复用同一套已验证的 acquire/release/send/recv 驱动。

## sched-11 spawn_blocking：阻塞卸载池 vs marl 每次建线程（`bench/sched/blocking_bench.cc`）

把阻塞 syscall / CPU 密集闭包从 worker 卸载到专用线程池,让 worker 继续跑本核其它 fiber。sched-11 落地时 7 测试无性能数,这里补。SOTA 对标 marl `blocking_call`——**同目标但 marl 每次 spawn 全新 `std::thread` 并 join(无池,marl/blockingcall.h `OnNewThread`)**,HYLE 是 tokio 式懒创建+复用池。故对标恰好量出池的价值。

| 场景 | HYLE(Win / Lin) | 判定 |
|---|---|---|
| **spawn_blocking 顺序往返(暖池)** | 1,025 ns / 12,772 ns | 见下"双平台 12x 差"解读 |
| **fan-out 8/64/256 并发 job** | ns/job 随 N 降(Win 64k→18k、Lin 99k→30k) | 池并发生效:更多 job 摊薄冷启动建线程 |
| **HYLE 池 vs marl 每次建线程** | HYLE 快 **26x**(Win 1769 vs 46846)/ **2.6x**(Lin 13863 vs 36328) | 池复用 vs 每次 `std::thread` create+join |

- **池 vs 每次建线程 = spawn_blocking 的核心价值(双平台都赢)**:marl 每次 `blocking_call` 建一个新 OS 线程再 join,HYLE 复用池线程。Windows 建线程贵(~47μs)故 HYLE 领先 26x;Linux 建线程相对便宜(~36μs)但 HYLE 仍快 2.6x。**诚实框定**:marl 的设计假设 blocking 调用低频(建线程可接受),HYLE 假设高频(池摊薄)——高频 offload 场景池优势明确,低频两者都够用。
- **暖池往返双平台 12x 差(Win 1μs / Lin 12.7μs)——根因坐实是 WSL2 虚拟化放大的线程唤醒地板,非 HYLE**:一次 spawn_blocking 往返要**唤醒一个睡眠的池线程**(submit 的 `condition_variable::notify_one`)+ 池线程跑完再经 JoinState CV **唤醒 join 方**。**真调查(非臆断)**:①判别实验——给闭包加固定 spin,Win 1025→2023ns、Lin 12772→13929ns,**都只 +1μs(spin 工作量)无质变**,证两平台都在走完整往返、join 确实挂起(排除"join 侥幸不挂起"假设);②脱离 HYLE 的纯 `std::condition_variable` 两线程 ping-pong 微基准实测**单次唤醒地板:Windows 原生 ~330ns、WSL2 ~12,015ns(35x 差,稳定复现)**——HYLE 的 12.7μs 几乎全是这个 OS 地板(+~700ns HYLE 开销)。**结论:差异是 WSL2(Hyper-V 轻量 VM)跨线程 futex-wake + vCPU 调度经虚拟化层被放大**(原生 Linux 通常 1-3μs),不是 HYLE 逻辑、也不是 Linux vs Windows 的库差异。裸 Linux 上此数应回落到微秒级。且揭示一个真结论:**spawn_blocking 是为真阻塞工作(syscall/CPU 密集,本身 ≥ 数十 μs~ms)设计的**,卸载微秒琐事时唤醒地板会主导(本 bench 的 trivial 闭包是放大唤醒成本的极端情形)。
- 正确性由 sched-11 的 blocking_test(7 测试,含 WorkerNotStalledByBlockingCall 证单 worker 不被阻塞拖死)+ WSL TSan 覆盖。

## sched-07 取消树：is_cancelled() 热路径零税 + cancel() 传播 O(N)（`bench/sched/cancel_bench.cc`）

CancelToken 树是 shard-local、(几乎)无锁结构——唯一原子是 per-node `cancelled` 标志(fiber 跨 worker 迁移后的跨核可见性)。sched-07 落地时 16 测试 + TSan 无性能数,这里补,证三主张。纯用户态(无需 scheduler)。

| 场景 | HYLE(Win / Lin) | 对照 | 判定 |
|---|---|---|---|
| **is_cancelled()** | 1.36 / 0.28 ns | 裸 std::atomic<bool> acquire load 1.18 / 0.26 ns | **86.7% / 93.8%**:只比裸 load 多一个 null 检查,热路径实质免费 |
| **derive() 每子** | 8.6 / 7.5 ns | — | 一次分配 + O(1) 前插,请求 fan-out 到 N 子任务的每子成本 |

**cancel() 传播 O(N)(建 N 节点树 + cancel + 拆,batch=N → ns/node)**:

| 树形 | N=64 | N=256 | N=1024 | 判定 |
|---|---|---|---|---|
| **WIDE**(root + N 直接子,Win) | 13.87 | 12.98 | 13.68 ns/node | 平坦 => O(N) |
| **WIDE**(Lin) | 11.97 | 11.13 | 11.49 ns/node | 平坦 => O(N) |
| **DEEP**(N 深链,Win) | 18.05 | 18.09 | 18.83 ns/node | 平坦 => O(N) |
| **DEEP**(Lin) | 15.17 | 15.20 | 15.93 ns/node | 平坦 => O(N) |

- **is_cancelled() 热路径实质免费**:每个 await 点 / `HYLE_CHECK_CANCEL()` 都 poll 它,故必须 ~一条 load。实测只比裸 `atomic<bool>` 多那个 `node_!=nullptr` 检查(Win 86.7%、Lin 93.8%,Lin 更贴近因编译器把 null 检查优化进去)。这坐实"每个让出点查取消"零负担。
- **传播 wide/deep 双形态都平坦 = iterative DFS O(N)**:`cancel_subtree` 迭代遍历(非递归,防深 fan-out 爆栈)。wide(兄弟扫描)与 deep(parent→child 下降)两条遍历路径都测,ns/node across N 平坦证全链路(建+cancel+拆)线性,无隐藏 re-walk。**deep 比 wide 略贵**(~18 vs ~13ns/node):深链是指针追逐(每节点在不同 cache line),wide 兄弟链更局部。
- 正确性由 sched-07 的 cancel_test(16 测试:析构过继/born-cancelled/宽深树/回调健壮性)+ net_cancel/http_cancel + WSL TSan 零 race 覆盖。

## JoinHandle / spawn_task / when_all / when_any（sched-10，`bench/sched/join_bench.cc`）

结构化并发糖层的对标——**没有外部对手**(tokio/folly 是别的 runtime),标尺是"糖 vs 它抽象的裸机制":`spawn_task+JoinHandle::join` vs `裸 Fiber + WaitGroup(1) + 手动结果 slot`(sched-10 前测试代码手搓的那套)。目标是证明这层封装不比手写贵多少。

| 场景 | HYLE 糖(Win/Lin) | 裸机制基线(Win/Lin) | 判定 |
|---|---|---|---|
| **spawn+join 每任务**(批量 100 摊销) | 390 / 423 ns | 338 / 415 ns | Win **~1.15x**(~52ns 层开销)/ Lin **统计平局**(0.98x) |
| **when_all 64 任务** | 372 / 467 ns/任务 | (= 逐个 join) | 群等待 vs 逐 join **零额外**(≈单任务 spawn+join) |
| **when_all(4) 组调用** | 1859 / 6264 ns | — | 4 任务全等待的组安装+等待+卸载 |
| **when_any(4) 组调用** | 1728 / 3040 ns | — | 取第一即返,故 ~2x 快于 when_all(4) |
| **when_any(8) 全完成** | 503 / 527 ns/完成 | — | 8 任务全走 signal 路径(loser 真完成),测 group 锁竞争 |

- **层开销每任务 Win ~52ns / Lin 平局——本质"几乎免费"**:糖路径比裸多 **2 次堆分配**(JoinState 结果 slot + TaskDriver{Task 帧+Fiber}),换来**调用方零生命周期管理**(driver 经 Fiber finish 钩子自析构);裸路径复用调用方栈对象但**调用方必须保活 + drain()**。~52ns 买"啥都不用管"+ 安全(无悬空 fiber 陷阱)。spawn 非最紧热路径(那是 23ns fiber 切换 / 141ns spawn),付这点启动一个带结果句柄的自持任务是对的取舍。
- **⚠️ 方法学教训(重要,已内化)**:初测**单任务往返** spawn_task ~869ns/Win、~9908ns/Lin,看似 ~1.3x/~2x"重开销"——**全是 OS 跨线程唤醒抖动**(spawn 在主线程→worker 跑→futex 唤醒主线程),不是层开销。改**批量 100/task 摊销 handoff + 迭代 2000** 后,Win 收敛到 1.15x、Lin 到平局,err% 从 7% 降到 <3%。**亚 μs 的 sched 数字必须摊销 + 够迭代,否则线程唤醒噪声冒充"开销"**(同 queue/CV 的"单平台数陷阱"一族)。
- **when_any 快于 when_all 符合语义**:when_any 一个完成就返回(其余协作取消/detach),when_all 等全部——4-pack 下 when_any ~2x 快,证组引擎的"取第一即卸载"路径生效。
- **when_any UAF 修复 A/B 实测(2026-07-05,回答"选 A 更彻底会不会更快")**:红队审出 loser-vs-栈上 GroupSignal 的 UAF(见 [`../debug/when-any-loser-uaf-hunt.md`](../debug/when-any-loser-uaf-hunt.md)),两个修法——**B**(已采用,on_complete 移进 state.mu、靠锁序保活 group 栈对象,零分配)vs **A**(folly 式,GroupSignal 改 `util::RefCounted` 堆分配+引用计数,on_complete 可锁外调)。完整实现 A、双平台各 3 轮跑上面那条 when_any(8) 行:**Win B~503 vs A~511、Lin B~527 vs A~533,统计打平**(err% 内互换排序)。根因=8 次跨 worker fiber 唤醒主导耗时,淹没了"锁内多一次 on_complete"(B)与"堆 alloc+free+refcount 原子"(A)的微差。A 更"纯"(生命周期显式、不依赖锁序论证)但**不更快**,还给零分配路径加了每次 alloc+free → **保留 B**,删 A 脚手架(只留这条 bench 行,它照测 B 的 signal 竞争)。附:A 首跑 TSan 报 2 warning 经判别实验(decref 临时改 acq_rel → 0 warning)坐实是 **TSan 对 `RefCounted` fence 型释放建模不足**(§3.1 mimalloc 同族),非 A 真竞态。
- **正确性优先**:核心是 UAF(自捕获 fiber 自析构,已改 dispatch 收割)+ GroupSignal 销毁前卸载(folly WhenN 防护)+ loser 并发完成的栈对象保活(sched-10 UAF,option B 锁内 signal),双平台 53/53 + **WSL TSan 零 race**(mimalloc×TSan 地址复用误报已用可见分配器隔离,见 milestones sched-10)。

## 调度器层缺口 bench：work-stealing 扩展 / spawn_on 亲和 / 协程胶水 / fiber sleep（`bench/sched/scheduler_ext_bench.cc`）

> 补 sched 此前四个缺 bench 的行为。**边界(避免与既有重复)**:`scheduler_bench` 测原语(yield/spawn/suspend-wake spine),`queue_sota_bench` 已测 work-stealing **数据结构层**(WsFifoQueue steal_into 微观 ns),本节测**调度器层行为**——三者互补。双平台各 15 遍 `bench-stat.py`(Win MSVC19.50 + Linux GCC14.2 原生 ext4),gap>两变体 spread 之和才判赢/输。

| 项 | HYLE(Win/Lin) | 基线 | 判定(15 遍,双平台) |
|---|---|---|---|
| **work-stealing 扩展 2 worker** | 1.78x / 1.66x / 1.74x (Win/Lin/M4) | 1 worker 为 100% | ✅ 三平台同向 |
| **work-stealing 扩展 4 worker** | 2.85x / 2.47x / 2.95x | 1 worker | ✅ 三平台同向 |
| **work-stealing 扩展 8 worker** | 3.95x / 3.11x / 4.34x | 1 worker | ✅ 三平台同向，硬件拓扑决定边际 |
| **spawn_on 亲和 vs spawn** | 平局 / **1.33x** | 普通 spawn(全局注入) | **Win ➖平局**(gap27%«spread50%)/ **Lin ✅WIN**(gap33%>spread31%) |
| **协程胶水 co_await vs 裸 fiber** | 平局 0.99 / 1.007 | 裸 suspend()/Waker::wake() | **双平台➖平局**(gap 1.0/0.7% « spread 57/23%)=零开销铁证 |
| **fiber sleep 1 vs 64 并发 sleeper** | 平局 0.998 / 0.995 | 1 sleeper vs 64(同单 worker) | **双平台➖平局**(gap 0.2/0.5% « spread 9/3%)=不串行化铁证 |

- **work-stealing 扩展三平台 WIN——sched-03 核心引擎兑现**：单个非
  worker 线程点注入 512 个 compute fiber，全落 global 注入队列；N worker 靠
  pop 自己分片 + 随机 victim steal 并行跑。2026-07-19 util 收口重建仪器时发现
  benchmark 仍调用已删除的旧 Fiber 构造器，且原 2,000-round 计算被每个 guarded
  stack 的 VM 构造淹没，曲线只剩约 1.45x；修构造器并提高到 20,000 rounds 后，
  Win/Lin/M4 的 8-worker 分别恢复 3.95/3.11/4.34x。A/B 把 victim hint 从
  xorshift 换成用户级 Rng 未改变曲线，证明瓶颈是负载形状，不是随机数质量。
- **spawn_on 亲和 = 又一"单平台数陷阱"**:pinned 路由(私有 per-worker 队列 + `unpark_specific` 无条件唤醒,**绕过** `notify()` 的 `num_searching_` anti-herd gate,且永不被偷)对比普通 spawn(全局注入 + 条件 notify)。**Linux 真赢 1.33x**(261 vs 355ns,直达目标 worker 省掉 anti-herd 判断 + steal 竞争),**Windows 判平局**(spawn_on 中位仍偏快 26%,但 spread 50% 吞掉 gap)。只报单平台会误判——同 dtype/time/path/queue 的老陷阱。**注:spawn_on 的正职是 io shard-per-core(连接 fiber 钉在拥有其 io ring 的核),此 bench 顺带证它的派发路径不比普通 spawn 贵、Linux 还更省。**
- **协程胶水零开销——"糖 vs 自家裸原语"**:`co_await SuspendFiber{}`/`CaptureWaker{}` ping-pong 对比**形状完全相同**的裸 `suspend()`/`current_waker()`/`Waker::wake()` ping-pong(即 scheduler_bench 的 PingPong)。**双平台统计平局**(ratio 0.99-1.007):协程帧冻结在 fiber 栈上,一次 fiber suspend 覆盖整个 co_await(await_suspend 返回 false 让协程在再注入后原地续跑,不二次挂起),所以协程语法层**没在裸 fiber spine 上加可测开销**。无外部对手(tokio/folly 是别的 runtime),标尺是自家裸原语。
- **fiber sleep 不串行化——sched-04b 时间轮脊**:1 个 vs 64 个并发 sleeper **睡同样 2ms、都在单 worker 上**。若 sleep 烧 worker(像 `std::this_thread::sleep_for` 阻塞整条 OS 线程),64 个就得 64×2ms 串行;实测 **1 与 64 几乎相等**(ratio 0.998/0.995)=一个 worker 扛住 64 个并发 sleep 不串行(定时轮 arm O(1) + 一次 park 到最近 deadline,tick 到期批量 re-inject)。绝对值被 OS 定时器地板主导(Win ~15.6ms/Lin ~3.3ms,2ms 请求被抬到粒度整数倍),故**看两行相等这个对比,不看绝对值**。
- **方法学**：亚 μs + fork-join 数受 OS 跨线程唤醒抖动主导。scaling 必须让
  fiber 内计算显著超过 guarded-stack 建立成本；当前为 512 fiber × 20,000 rounds，
  三平台 fresh build 复核。coro 用批量摊销，sleep 用 batch=1（每 epoch 是真睡眠）。

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

旧基线把 heap 的取消实现成 `vector<bool>` 墓碑、把 common-timeout 实现成
`std::list + unordered_map`；这分别少算了真实 heap 的 keyed erase 能力，又给链表基线
人为加了节点分配和 hash，不能用于 SOTA 结论。2026-07-19 收口时已按本地
libevent `min_heap_erase_` 和 libuv `heap_remove` 重写为**预分配 indexed
eager-erase binary heap**：节点保存 heap index，取消立即 sift/remove，不留墓碑。
N=4096 timer/op，Windows / WSL / Apple M4 各 15 个独立进程。节点在 epoch
外预分配并跨 epoch 复用，arm 与 consume 之间有 compiler memory barrier；因此
数字不包含节点生命周期分配，也不能被编译器把整轮 arm+consume 融合删除。

| 工作负载 | Windows：HYLE / heap | WSL：HYLE / heap | M4：HYLE / heap |
|---|---:|---:|---:|
| arm + cancel | 6.86 / 10.42 ns（**1.52x**） | 5.95 / 6.73 ns（**1.13x**） | 4.91 / 8.36 ns（**1.70x**） |
| arm + expire | 21.34 / 49.80 ns（**2.33x**） | 22.05 / 40.64 ns（**1.84x**） | 11.83 / 23.74 ns（**2.01x**） |
| churn（arm + 70% cancel + expire） | 12.87 / 20.85 ns（**1.62x**） | 13.63 / 16.01 ns（**1.17x**） | 7.62 / 11.03 ns（**1.45x**） |

**核心诚实结论**:

1. 三种形状在三个平台同向，时间轮的领先不依赖对手墓碑；对 eager-erase heap
   仍成立。cancel 领先来自 O(1) 已知槽摘链对 O(log n) heap 修复，expire 领先来自
   分槽批量转移。最新 WSL churn 为约 1.17x，仍是三平台中最窄的一档，不把它
   外推成所有 Linux/所有负载的结构性大胜。
2. 收口后的 wheel 是 **8×64**，覆盖完整 `uint64_t` 纳秒域；旧 6 层轮在约 2.18
   年外会丢失高位并可能提前触发，这是正确性修复，不应用旧 6 KB footprint 解释新数据。
3. 这仍是数据结构 microbenchmark，不代表所有 timer workload。timer 很少、heap 已嵌入
   event loop，或需要其它 heap 操作时，整体系统成本可能不同。

> **诚实边界**：① 时间轮是 per-worker 单线程结构，不是 lock-free；跨线程唤醒由 sched
> completion spine 负责。② 没有把 folly EventBase 或完整 libevent event loop 链进来，
> 对比的是其核心 heap 形状。③ 70% cancel 是合成压力参数，不是生产观测值。④
> `TimerEntry` 现在记录精确 owner 并执行链接状态检查；这些安全合同已包含在 HYLE 数字中。

## 两种定时器结构:TimerWheel vs CommonTimeoutQueue(`timer_wheel_bench.cc` 块 4/5)

同一时长的 deadline 随单调钟天然有序，`CommonTimeoutQueue` 可以只做尾插、头摘和
O(1) unlink。基线直接复刻 libevent `struct event` 内嵌 TAILQ link 的形状：
预分配节点、零 hash、零每次分配。现在同时报告两条线：完全不检查的裸 TAILQ
是理论常数下界；checked TAILQ 加入与 HYLE 相同的 reschedule、exact-owner、
unlinked-node、deadline-order 合同，是实现质量的同语义标尺。

| 固定时长负载 | Windows：HYLE / rival | WSL：HYLE / rival | M4：HYLE / rival |
|---|---:|---:|---:|
| arm + expire / 裸 TAILQ | 3.28 / 4.22 ns（**1.29x**） | 3.14 / 3.95 ns（**1.29x**） | 3.52 / 2.73 ns（**0.77x LOSE**） |
| arm + cancel / 裸 TAILQ | 2.70 / 3.13 ns（**1.14x**） | 1.99 / 2.59 ns（**1.30x**） | 3.06 / 3.06 ns（**统计平局**） |
| arm + expire / checked TAILQ | 3.28 / 3.61 ns（**1.10x**） | 3.14 / 3.21 ns（统计平局） | 3.52 / 4.21 ns（**1.16x WIN**） |

- Common queue 没有“9–15x 超越 libevent”的证据；旧数字测的是不同分配模型，已废止。
- 二轮 M4 狩猎把 HYLE 从 4.86 降到 3.52ns（约 -28%）：`pop_front` 利用已知
  sentinel、`consume_all` 一次 detach 后寄存器遍历、free-entry 先看 hook 再在
  uncommon path 读取 owner。order-swap 把 TAILQ 放前仍为 2.75/3.62ns，
  排除固定顺序与睿频解释；background QoS 也不改变输赢方向。
- M4 对 checked TAILQ 是 14/15 同向胜（paired ratio 1.16，`p=9.8e-4`），说明
  链表实现损耗已修平；对裸 TAILQ 仍 0/15 胜（0.77x），剩余是生产安全合同成本。
  两条都报告：不拿不同语义冒充实现胜利，也不删除防跨-source 腐化的合同换数字。
- 混合时长不满足 append-order 前提，必须用 `TimerWheel`。拒绝型
  `try_schedule` 是事务性的：失败不会丢掉旧 arm。
- `CommonTimeoutQueue` 结构已完成并验证，但 io reactor 尚未以它路由固定时长连接超时；
  当前在役 timed I/O 仍由每 worker `TimerWheel` 承担。
