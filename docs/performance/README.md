# HYLE 性能：预期与实测（核心目的：做到 SOTA）

> **第一性目的**：HYLE 是高性能 infra 库，**唯一存在理由是性能做到 SOTA**。标尺不是 `std`(赢它是最低门槛),而是**各领域 SOTA 参考库**(hashmap→absl/F14、分配器→mimalloc、队列→moodycamel/tokio、SIMD→Highway)。**做到等同或超过参考库才算数;只赢 std 不算价值。**
>
> **两段式**：① 下面"预期表"是开工前基于参考库公开基准的**推算目标**;② 实现后必须有**对标 SOTA 的真实 benchmark**(非推算),数字诚实印出——赢/平/输都登记,差距是优化目标。

## 本目录结构（原单文件已拆分，按模块分档）

| 文档 | 内容 |
|---|---|
| 本页（对标总账） | 全模块 win/loss ledger 一览 + 总账小结（最常看的一张表） |
| [error 实测](./benchmarks-error.md) | **error 2026-07-20 三平台证书**（Status/Result vs absl + std::expected） |
| [platform 实测](./benchmarks-platform.md) | **platform 2026-07-20 证书**（Parker/Thread/dir/vm/MappedFile；OS 段 n=15 Win/M4、WSL n=5） |
| [time 实测](./benchmarks-time.md) | **time 2026-07-19 三平台证书**（clock / TimerWheel / CommonTimeoutQueue） |
| [mem 实测](./benchmarks-mem.md) | **mem 2026-07-22 证书**：Region/FixedRegion request closure + hard core/显式 owner cache + external FixedPool；显式 cache 已签 Windows clean `n=15`，Linux/M4 待重签；request latency、footprint 与 FixedPool 外部碎片有 raw CSV |
| [container 实测](./benchmarks-container.md) | **container 导航证书**：新鲜度、非 SOTA、选型；详表 → [benchmarks-mem-container.md](./benchmarks-mem-container.md) |
| [container 详表](./benchmarks-mem-container.md) | hash/SwissTable/vec/list/str/path/fuzzy 长表 + 史；**入口见上** |
| [simd 产品 kernel](./benchmarks-simd.md) | dot/saxpy、math、dtype、gather、vendor 竞争…；**op 全量不在此** |
| [simd op 对拍证书](./benchmarks-simd-ops.md) | **逐 op vs Highway**：三平台 LOSE=0 摘要 + 覆盖树；全表 → [orderswap 矩阵](../milestones/simd-orderswap-matrix.md)（最新最全） |
| [simd op 对拍方法论](./simd-op-benchmark-methodology.md) | op_parity 规程：形状、双序、锚点闸门、LOSE 证据阶梯 |
| [sched 实测](./benchmarks-sched.md) | fiber 切换、sched 同步（marl 对标）、调度器层；**platform 原语已拆出** → [benchmarks-platform.md](./benchmarks-platform.md) |
| [io 实测](./benchmarks-io.md) | echo 吞吐 + shard-per-core 扩展 + vs 裸 epoll/liburing/asio/libuv + 连接生命周期 + TLS/HTTP + multishot/DEFER/fixed-files/direct-accept 机制项 |
| [预期与边界](./expectations.md) | AVX-512 上架状态、主表（时间/空间分列）、Amdahl 整体观、诚实边界、验证 gate |

## 对标总账（win/loss ledger，全部实测 vs 真 SOTA，非 std）

> **一眼看尽输赢**。**两把标尺各拆成「组件名 + ns」两列**:`std组件`/`std ns`(基线下界——赢它是最低门槛)和 `SOTA组件`/`SOTA ns`(真门槛:absl/F14、mimalloc、moodycamel/tokio/rigtorp、Highway/xsimd、marl、boost、simdutf)。`比std`/`比SOTA`:>1=HYLE 快几倍。**判定**针对 SOTA:✅赢 / ➖平(噪声内) / ❌输(均附根因:by-design 接缝 / 约束选择 / 选错工具 / scalar 待优化,无一为未处理的真缺陷) / ⚠️平台翻转 / ⏳待补。`n/a`=该层无对等物(非未测:如 bounded-MPMC 队列无 std 等价物);`同std`=该项 std 本身即领域 SOTA,无独立第三方更快。**error 行 = 2026-07-20 三平台 bench-stat n=15**；其余模块日期不一，未刷新前勿当当前证书。

| 模块 · 操作 | HYLE ns | std 组件 | std ns | 比std | SOTA 组件 | SOTA ns | 比SOTA | 判定 |
|---|---|---|---|---|---|---|---|---|
| **error** Status::Ok 构造+ok() | 0.55/0.05/0.23(Win/Lin/M4) | std::expected&lt;int,ec&gt; | 0.56/0.18/1.07 | 平/3.4x/4.7x | absl::OkStatus | 0.55/0.05/0.23 | ~1.0x 三平台 | ➖平 absl；Lin/M4 赢 std |
| **error** Status(code) inline 无消息 | 0.55/0.05/0.22 | std::unexpected&lt;ec&gt; | 0.37/0.05/0.22 | 三平台平 | absl code-only(仍堆) | 1.88/1.01/1.19 | **3.4/19.8/5.3x** | ✅三平台赢 absl(inline 零堆) |
| **error** Status(msg) 带消息→堆 | 26.1/8.9/16.4 | (无 std 等价) | n/a | n/a | absl::NotFoundError | 54.8/22.2/32.4 | **2.1/2.5/1.9x** | ✅三平台赢 absl |
| **error** Status 拷贝(OK) | 0.59/0.24/0.36 | n/a | n/a | n/a | absl::Status copy OK | 0.56/0.28/0.26 | 0.95/1.26/0.72x | ⚠️**平台翻转**(Win/M4 sign 输，Lin 赢) |
| **error** Status 拷贝(heap,retain/release) | 9.40/10.34/3.17 | n/a | n/a | n/a | absl::Status copy | 10.70/9.32/4.15 | 1.14/0.90/1.31x | ⚠️**平台翻转**(Lin sign 输 0.90x；Win/M4 赢) |
| **error** Result&lt;int&gt; 传播 OK(解包消费) | 1.45/0.33/1.59 | std::expected unwrap | 1.42/0.33/1.59 | ~1.0x 三平台 | (消费变体未对 absl) | n/a | n/a | ➖真实用法平 std |
| **error** Result&lt;int&gt; 传播 ERROR 上抛 | 10.2/9.4/3.2 | (errc 无堆消息不可比) | n/a | n/a | absl::StatusOr | 22.0/22.2/6.8 | **2.1/2.4/2.1x** | ✅三平台赢 absl |
| **simd·ops** 可对拍 op 全量(矩阵 v4) | — | (无 std) | n/a | n/a | Highway 同 ISA 同 op | — | **LOSE=0** | ✅**Win/WSL/NEON 明确 LOSE 均为 0**(320/320/308 case)；?≠TIE；全表 [benchmarks-simd-ops.md](./benchmarks-simd-ops.md)→[orderswap-matrix](../milestones/simd-orderswap-matrix.md) |
| **simd** dot product(4096 f32) | 0.024 | scalar autovec | 0.37 | 15x | Highway contrib/dot(真 SOTA 4-acc) | 0.024 | 1.00x | ✅**4 累加器展开后 0.079→0.024(3.3x)平真 SOTA**;旧"平 Highway 0.079"是两个都单累加器的假平局(对照用了 Highway 教学 kernel 非 contrib/dot),见 [benchmarks-simd.md](./benchmarks-simd.md) |
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
| **platform** Parker ping-pong(真阻塞) | 341/22.5k/2053(Win/Lin/M4) | std::condition_variable | 638/23.5k/2130 | **1.75x/1.05x/1.05x** | (std CV 即实用标尺) | 同左 | 同左 | ⚠️**平台翻转**(Win 大赢;Lin/M4 平或微赢)。旧 2.99x 作废。见 [platform](./benchmarks-platform.md) |
| **platform** Thread spawn+join | 43.4k/33.6k/10.6k | std::thread | 44.4k/33.6k/10.7k | ~1.0x 三平台 | std::thread(即 SOTA) | 同左 | ~1.0x | ➖平(OS 主导,证零封装税) |
| **platform** directory_iterate visitor | 40.5k/6.5k/16.5k | std::filesystem | 32.0k/13.7k/18.4k | **0.93/2.15/1.12x** | filesystem | 同左 | 同左 | ⚠️**平台翻转**(Win sign 输;Lin/M4 赢 d_type+visitor) |
| **platform** vm_reserve+commit vs raw 一步 | 1144/1075/846 | raw 一步分配 | 877/848/332 | **0.77/0.78/0.39x** | raw 一步 | 同左 | 同左 | ❌**by-design 两步**输一步 raw;非漏优化 |
| **platform** MappedFile vs ifstream | 43.7k/12.7k/14.1k | ifstream 拷贝 | 62.3k/**8.7k**/21.7k | **1.39/0.68/1.53x** | — | — | — | ⚠️**平台翻转**(Win/M4 赢;Linux 64KiB ifstream 更快) |
| **platform** MappedFile vs raw mmap | 43.7k/12.7k/14.1k | raw mmap | 42.0k/12.2k/13.7k | **0.96/0.96/0.97x** | raw 地板 | 同左 | ~平 | ➖近地板(~3–4% 封装税) |
| **platform** Parker 快路径 / park_timeout(0) | 9.7·3740 / 10.6·59k / 2.0·3.7 ns | (无公平对手) | n/a | n/a | atomic floor 非对等 | n/a | n/a | ➖快路径绝对值;0-timeout **平台极差** |
| **time** realtime_ns | 14.68 | std::chrono::system_clock | 14.76 | 1.0x | std::chrono(即标尺,同 OS 原语) | 14.76 | 1.0x | ➖平 |
| **time** monotonic_ns | 13.22/13.07/12.93(Win/Lin/M4) | std::chrono::steady_clock | 13.30/13.07/12.88 | 全部统计平 | std::chrono(即标尺,同 OS 原语) | 同左 | 1.0x | ➖header-inline 后三平台与 std 同档 |
| **time** fast_monotonic_ns(opt-in rdtsc) | 7.37/7.13/12.93 | HYLE monotonic_ns | 13.22/13.07/12.93 | 1.79x/1.84x/1.0x | (std 无 opt-in TSC 钟) | n/a | n/a | ✅x86 快约1.8x；M4 透明回退(非默认,可能跳变) |
| **time** fast_realtime_ns(opt-in coarse) | 3.20/3.90/13.64 | HYLE realtime_ns | 15.77/13.09/13.70 | 4.94x/3.36x/1.0x | (std 无 opt-in coarse 钟) | n/a | n/a | ✅Win/Linux 真赢；macOS 无 coarse 原语而回退 |
| **time** timer wheel churn(arm+70%cancel+expire) | 12.87/13.63/7.62 | (std 无定时器结构) | n/a | n/a | indexed eager-erase heap(libevent/libuv 核心形状) | 20.85/16.01/11.03 | 1.62x/1.17x/1.45x | ✅**三平台赢**；对手按真实 keyed erase 建模、无墓碑 |
| **time** timer wheel arm+cancel / arm+expire | 6.86·21.34/5.95·22.05/4.91·11.83 | (std 无) | n/a | n/a | indexed eager-erase heap | 10.42·49.80/6.73·40.64/8.36·23.74 | 1.52·2.33x/1.13·1.84x/1.70·2.01x | ✅三种工作量三平台同向；8×64 覆盖完整 uint64 ns 域 |
| **time** CommonTimeoutQueue(固定时长)arm+expire/cancel | 3.28·2.70/3.14·1.99/3.52·3.06 | (std 无定时器结构) | n/a | n/a | 裸 TAILQ / checked TAILQ | 4.22·3.13/3.95·2.59/2.73·3.06；checked expire 3.61/3.21/4.21 | 裸 1.29·1.14x/1.29·1.30x/**0.77·平**；checked expire 1.10x/平/**1.16x** | ✅同语义三平台不输；⚠️ M4 对无检查裸下界仍输 |
| **util** Rng next_u64(xoshiro256++) | 1.59/1.38/2.26 (Win/Lin/M4) | std::mt19937_64 | 2.32/1.44/3.47* | 1.46x/平/1.53x | std 即算法类标尺 | 同左 | 同左 | ✅ Win/M4 赢；GCC 标准库很强而平（M4 std 行方差高） |
| **util** Rng varying range_u64 | 2.09/1.47/2.17 | std::uniform_int_distribution | 2.86/2.08/7.68 | 1.37/1.41/3.54x | Lemire 同法实现 | 同档 | — | ✅ 三平台赢；运行时 bound 避开常量除数折叠伪影 |
| **util** Rng shuffle 1000 | 2.05/1.21/0.81 | std::shuffle | 6.71/1.19/7.35 | 3.28x/平/9.07x | std 即组合标尺 | 同左 | 同左 | ✅ Win/M4 赢、GCC 平；平台实现差异显著 |
| **util** C ABI RNG fill / C++ fill | 4.99/0.67/0.65 | 原生 C++ batch | 5.16/0.68/0.65 | 全部平 | 原生 C++ | 同左 | ~1.0x | ➖ batch 摊薄 ABI seam；单值 C 调用仍多约 0.09–0.95ns |
| **util** SpinLock lock/unlock(无争用) | 3.71/4.79/1.01 | std::mutex | 10.94/2.48/4.19 | 2.95x/**0.52x**/4.15x | OS mutex | 同左 | 平台翻转 | ⚠️ 仅短低争用；不能从 Windows 外推到 Linux |
| **util** SpinLock 8-thread contention | 50.17/60.44/271.88 | std::mutex | 50.59/39.08/20.98 | 平/**0.65x**/**0.08x** | OS mutex | 同左 | 同左 | ❌ 高争用应选 mutex；M4 自旋尤其差 |
| **util** Ref copy+destroy fan-out | 10.13/8.33/3.19 | std::shared_ptr | 8.09/11.12/3.22 | 0.80x/1.34x/平 | shared_ptr | 同左 | 平台翻转 | ⚠️ 价值是侵入式所有权/省控制块，不承诺每平台拷贝更快 |
| **obs** log gate / preformatted / formatted | 0.91–2.11 / 8.85–22.48 / 50.9–84.6（三平台） | (无 std log) | n/a | n/a | 同步 sink 产出原语 | n/a | ➖ | ✅ gated 极轻；preformatted 含 quiescent-unregister active ref；C ABI 仅多 1.5–3.0ns；formatted 由 `std::format` 主导。详见 obs closure |
| **mem** SystemAllocator churn(默认) | 7.93/8.04/7.77 | malloc OS | 25/4.7/20 | 大赢 OS | 裸 mimalloc | 7.86/8.04/7.91 | **~1.00x 三平台** | ➖**平 mi SOTA**(2026-07-20 n=15) |
| **mem** Allocator vtable churn | 8.67/8.67/7.93 | pmr over mi | 8.67/10.3/9.98 | 平/1.22/1.26 | 裸 mi | 7.86/8.04/7.91 | **0.91/0.94/1.00** | ⚠️对 mi Win/Lin **sign 输~6–9%**；✅对 pmr 平/赢 |
| **mem** PooledRegion/Stack bump | 0.57·0.55/0.36·0.36/2.23 | pmr monotonic | 1.46/0.62/**1.24** | **2.65/1.70/0.56x** | 裸 mi | 3.9/3.9/4.8 | 7–11x/2x | ⚠️**平台翻转**(Win/Lin 赢 pmr；**M4 输 0.56x**) |
| **mem** Region owned request scratch | 1.674/1.137/2.282 | pmr monotonic release | 1.923/2.033/2.444 | **1.126/1.779/1.073** | Folly Arena（shape 参考，未同 harness 实测） | n/a | n/a | ✅三平台 sign 赢；WSL Clang 另为 1.482x；M4 interactive/utility 各 n=31 均赢，无硬核心亲和 |
| **mem** GrowablePool 单线程 | 2.76/1.38/2.46 | pmr unsync | 5.57/11.3/3.47 | **2.0/8.3/1.4x** | 裸 mi | 4.75/4.31/5.80 | **1.7/3.1/2.4x** | ✅三平台赢公平 ST 标尺；固定共享池不是这条单线程合同的对手 |
| **mem** GrowablePool 8t per-thread | 0.93/0.61/0.60 | mi 8t | 1.35/1.28/1.42 | **1.49/2.07/2.38x** | mi per-thread | 同左 | 同左 | ✅三平台 sign 赢 |
| **mem** SharedPool request batch=64（硬容量） | 8.315/6.609/11.881 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 4.200/2.280/2.024 | **0.515/0.338/0.170x** | ❌三平台 sign 输；为 completed-free 立即共享和精确 quota 付费，不冒充本地 churn 工具 |
| **mem** SharedPool pipelined handoff（硬容量） | 5.304/3.807/12.931 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 10.433/9.239/4.871 | **1.976/2.380/0.372x** | ⚠️平台翻转：Win/WSL sign 赢，M4 sign 输 |
| **mem** SharedPoolCache request batch=64（显式 owner cache） | **1.203/—/—** | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 2.208/—/— | **1.846x/待签/待签** | ✅Windows clean `5f35ec1`、n=15：spread/sign WIN（15/15）；Linux/M4 不继承旧 TLS 数字 |
| **mem** SharedPoolCache pipelined handoff（显式 owner cache） | **1.431/—/—** | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 7.288/—/— | **5.004x/待签/待签** | ✅Windows clean `5f35ec1`、n=15：spread/sign WIN（15/15）；Linux/M4 待重签 |
| **mem** SharedPoolCached request batch=64（已删除，历史） | 3.540/1.754/1.537 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 4.200/2.280/2.024 | **1.171/1.285/1.316x** | ⚠️只记录旧 TLS 实现；当前显式 `SharedPoolCache` 已重签 Windows，Linux/M4 待签 |
| **mem** SharedPoolCached pipelined handoff（已删除，历史） | 4.062/2.414/3.072 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 10.433/9.239/4.871 | **2.627/3.807/1.586x** | ⚠️只记录旧 TLS 128/32 实现；不可改名继承到当前产品 |
| **mem** FixedPool external mixed-size window | 6.359/5.760/5.868 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 8.297/7.788/9.318 | **1.305/1.383/1.607x** | ✅ clean `d3e4214`：Win/GCC/Clang/M4 interactive+utility 全部 spread/sign WIN；Clang 1.670x、M4 utility 1.668x |
| **mem** FixedPool whole-request p50（Win/Lin/M4，µs） | 26.600/24.428/23.584 | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 36.100/37.857/40.917 | **1.357/1.554/1.735x** | ✅ clean `d3e4214`：Win/Linux/M4 p50/p95/p99 全 WIN，M4 utility 同结论；split-tail cursor 关闭 Windows tail 缺口 |
| **mem** Region 128MiB request reservation / payload（Win/Lin/M4） | **1.031/1.031/1.031** | pmr monotonic | 1.443/1.302/2.000 | **更低 28.6%/20.8%/48.5%** | 同 backing live RSS Δ (MiB) | 129.11/138.05/128.67 vs 128.59/144.09/128.72 | — | ✅ clean `02ec81e`：exact fragmentation 三平台更低；RSS Win +0.4% / Linux **-4.2%** / M4 -0.04%，三平台同档且 Linux 缺口关闭；full release backing 均为 0 |
| **mem** FixedPool 75% mixed-fill 后 fragmented availability（Win/Lin/M4） | **0.6874/0.6874/0.6874** | （无同合同 std） | n/a | n/a | oneTBB fixed_pool | 0.4843/0.4816/0.4816 | **1.419/1.427/1.427x** | ✅ clean `d3e4214`：此 selective-free workload 下最大连续对象约 1.42x；两者 full recovery=1.000；不外推到任意分布 |
| **mem** reallocate grow chain | 80/83/82 | std::realloc | 200/176/204 | ~2.5x | mi_realloc | 67/77/83 | **0.83/0.92/1.01** | ⚠️Win/Lin 输裸 mi；M4 平 |
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
| **container·str** StringBuf intern(100k/2k 唯一) | 28.86(Win) | 无锁 map floor | 19.78 | 0.685x | thread-safe peer(map+mutex) | 28.70 | **0.994x 平** | Win 平；WSL 0.772x、M4 0.934x，旧 1.17x 证书撤回 |
| **container·str** StringBuf `intern_many` | 19.97(Win) | 无锁 map floor | 19.78 | **0.990x 平** | 同左 | 19.78 | 0.990x | M4 1.045x；WSL 0.732x，非全平台无条件赢 |
| **container·str** StringBuf 1KiB unique growth | 101.58(Win) | std::unordered_set | 1813.93 | 17.86x | absl::flat_hash_set | 544.05 | **5.36x** | ✅ cached resident hash；WSL vs Absl 1.71x，M4 vs std 1.67x |
| **container·str** StringBuf 8-thread adaptive shards | 23.56(Win) | sharded std+mutex | 56.96 | 2.42x | sharded Absl+mutex | 22.33 | **0.948x** | 距 Absl 约 5%；WSL 0.958x，M4 vs std 0.956x |
| **container·str** StringBuf add/retain/remove churn | 18.62(Win) | direct-ID+mutex 同契约 peer | 18.44 | **0.990x 平** | 同左 | 18.44 | 0.990x | Win 平；WSL 0.813x、M4 0.785x |
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
| **sched** schedule 2000 noop（worker-stack，15遍三平台） | 192.60(Win)/218.44(Lin)/252.12(M4) | marl enqueue（严格4 worker） | 297.53/327.03/252.72 | 1.53/1.48/1.007x | marl | 同左 | 同左 | ✅ Win/Lin 明确赢；M4 统计平局。旧 stackful 混比与 marl caller bind 形状已作废；ARM 全局完成计数争用已修 |
| **sched** WaitGroup fork-join（worker-stack，15遍三平台） | 192.51(Win)/220.77(Lin)/253.98(M4) | marl | 297.79/337.42/252.75 | 1.55/1.51/1.003x | marl | 同左 | 同左 | ✅ Win/Lin 明确赢；M4 统计平局；conditional-notify + per-worker completion accounting |
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
| **io** echo shard-per-core 4w（2026-07-20 收口） | Win 3,937 / WSL 1,138 / M4 6,151 ns | (无 std 异步 IO) | n/a | n/a | 自身 1w | 8,811 / 1,515 / 3,082 ns | **2.24x / 1.34x / 0.50x** | ⚠️ **Win/Linux 扩展赢，M4 4w 真输**。旧 2.2x headline 只属于历史 Linux run；当前三平台证书见 [IO 详节](./benchmarks-io.md) |
| **io** echo 1w（2026-07-20 收口） | WSL 1,515 ns | (无 std 异步 IO) | n/a | n/a | raw epoll | 1,375 ns | **0.90x** | ❌ **当前 Linux 单 worker 比 raw epoll 慢约 10%**；Win/M4 raw peer 未同形重跑，不沿用旧的 asio/libuv headline |
| **io** pipelined 高扇出（2026-07-20 收口） | WSL 1w 3,567 → 4w 1,754 ns | (无 std 异步 IO) | n/a | n/a | 自身扩展性 | n/a | **2.06x** | ✅ 实际完成数校验后的当前 Linux 扩展证据；旧 5.2x 留在详节作优化历史，不作发布证书 |
| **io** 连接生命周期 accept+serve+close(真实短连接) | ~240µs/conn(4162 conn/s) | (无 std 异步 IO) | n/a | n/a | (自身:ephemeral→persistent 谱系) | n/a | n/a | ➖**每连接固定成本 ~240µs(accept+握手+close)主导**,R=1→16 round-trip 仅 +~2.6µs/轮=echo 稳态往返(填 accept_bench 即弃/echo_bench 预建之间的真实使用缺口;正是 keep-alive/连接池要摊薄的那块)。双平台(Win IOCP ~2ms)+ TSan 净 |
| **io** multishot recv awaitable vs one-shot(io-07b) | 165,679 ns/dgram | (无 std) | n/a | n/a | HYLE one-shot RecvOp(自身对照) | 166,985 | 0.998 | ➖**统计平局**(gap0.2%«spread13.4%,n=15):同核 lockstep ~1 op 在途,fiber 往返主导、multishot 拿不到"一批完成同到"来摊薄省下的 submit。**非回归、待承重**(需突发/高扇出负载显优势,登记未来;正确性+TSan+回归已验) |
| **io** DEFER_TASKRUN vs COOP_TASKRUN(io setup flag) | 1.35-1.43µs(1w) | (无 std) | n/a | n/a | COOP_TASKRUN(自身对照) | 1255ns(1w) | ~同量级 | ➖**同量级无显著提升**:DEFER 收益=省完成 task_work 的 IPI,单 worker loopback 本就每次 drain 都 enter ring、几乎无 IPI 可省。**正确接入(R_DISABLED 生命周期修 + kCancelToken latent bug)、不回归、机制就位待高 syscall 压力/多核争用显收益**(opt-in 默认 COOP) |
| **io** registered/fixed files(IOSQE_FIXED_FILE,io-07c) | +0.8-3.3% | (无 std) | n/a | n/a | 裸 fd(自身对照) | 基线 | 1.008-1.033x | ➖**边际正收益**(socketpair ping-pong 3 跑一致为正、淹在 1-3% err% 边缘):省内核每 op fget/fput 是真的,但 loopback 上 fd 查表只占往返极小一块 |
| **io** direct-accept vs normal accept(io-07c) | 合并慢~1%/accept 相快~10-30ns | (无 std) | n/a | n/a | normal accept(自身,拆相分离) | — | accept 段 direct 快 | ➖**非回归**(拆相追根 0.7%问题):**accept 原语本身 direct 反而快 ~10-30ns/conn**(省 fget/fput,合理);合并 bench 那~1%全是 teardown 不对称伪影(direct close_direct+reap+FILES_UPDATE 2 syscall 其一冗余 vs normal ::close 1 syscall),生产不即建即关故不出现。primitives-only 不改 TcpStream |

**历史总账小结（2026-06-30 快照，不作为当前证书）**：当时按两把标尺统计约
36 win / 16 tie，并推动了 allocator、hashmap、SIMD、queue、string-ref 等多轮
优化。此后接口、对手和三平台仪器都已升级，不能继续用这组数量证明当前
“全线赢”。尤其 StringBuf 的旧 1.17–1.24x 证书已被本页当前 direct-ID peer
三平台表替代。发布判断以各模块最新 closure 文档和本页逐行当前数据为准，不再
维护一个会把不同日期、不同合同混在一起的总 win 数。

**io 段补记(2026-07-05,reinject+COOP 优化后 n=15 重测)**:HYLE echo-1w **单核反超成熟抽象框架 asio 1.25x / libuv 1.52x**(优化前是统计平局,-33% 推成真赢)、**平裸 epoll**(零抽象就绪循环,追平即好);多核 shard-per-core 开箱大赢。新增**真实连接生命周期**基准(accept+serve+close):每连接固定成本 ~240µs 主导、每 round-trip 仅 +~2.6µs——量化了 keep-alive/连接池的价值。io 机制项(multishot awaitable / DEFER_TASKRUN / registered-fixed-files / direct-accept)均为**正确接入、loopback 上统计平局或边际**、"就位待承重"——它们的收益需 loopback lockstep 造不出的突发/高扇出/高建连率负载,已诚实登记为待承重(非回归、非缺陷);direct-accept 的"慢 1%"经拆相追根证为 teardown microbench 伪影(accept 原语本身 direct 更快)。
