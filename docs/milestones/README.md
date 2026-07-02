# HYLE 里程碑索引

> 实现的真正"步骤"在这里——约 47 个里程碑，每个 1-4k 行、可独立编/测/基准。`roadmap.md` 是依赖层(阶段)的拓扑序与决策；本目录是**可执行的逐步计划**，每步一文档。
>
> **粒度原则**：一个里程碑 = 一次"写完→编过→单测(覆盖达 Tier 标准)→基准→提交"的闭环。
>
> **生长方式（JIT）**：最先执行的 `00-scaffolding` 与 walking-skeleton 几步已写完整；其余**临到该模块开工前再补全细节**——后期里程碑依赖前面实现学到的东西，提前写=空想。索引行先占位（ID/依赖/目的/LOC），细节随推进生长。
>
> **clay 吸纳完整性标尺**：见 [`../architecture/clay-coverage-map.md`](../architecture/clay-coverage-map.md)——clay 每个文件 → HYLE 对应物/状态/LOC 的逐文件映射。**每个吸纳 clay 的里程碑闭环前必须 diff clay 同名文件 API 表面**，缺的补或显式登记，不许默默略过(红队核账 2026-06 立此纪律)。

## Walking Skeleton（最先做，纵切验脊）

打通"**一个 fiber 跑起→让出→被唤醒**"。= `00` + 各模块**第一个最小里程碑**串起来，**先不做 work-stealing/多后端**。验掉最高风险的 fcontext+调度脊，之后逐里程碑加厚。

骨架路径：`00-scaffolding` → `error-01` → `platform-01` → `mem-01` → `container-01` → `sched-01`(单 worker 单队列)。

## 预测 vs 真实 LOC（防玩具化）

### 整体进度(诚实,2026-06-30 实测)—— 三个分母别混

> **实测代码 8,543 行**(生产,cloc `include`+`src` 的 header+C++,不含 CMake/测试/文档/vendored)。三个"进度"说法分母不同,**混用会得出误导数字**:
>
> | 口径 | 分母 | 当前 | 说明 |
> |---|---|---|---|
> | **clay 吸纳权重** | 被重构的旧 clay 库 | ~32%(coverage-map) | 基础层(sys/utils/container 主体)已基本吸纳;**分母是旧库,不是新愿景** |
> | **自有生产 LOC 比** | 投影完成时 ~30-45k 自有 | **~19-28%** | 8.5k/(30-45k);但 LOC≠完成度(广度缺口不体现在比值里) |
> | **对完整愿景的功能完成度** | 完整 HYLE 自有(全广度 + io/compute/graph,**不含借入的 ggml**) | **≈22%**(band 19-28%) | 见下方算式;此前模糊的"个位~十几%"是误把借入的 ~100k ggml 算进分母所致,已纠正 |
>
> **"≈22%" 怎么算出来的(两法交叉验证,非拍脑袋)**:
> - **法一 · LOC 比**:8.54k(已写生产,cloc 实测)÷ 37.5k(完成投影中值,band 30-45k)= **22.8%**(45k 下界 19%,30k 上界 28%)。
> - **法二 · 能力加权**(按投影 LOC 占比加权各支柱完成度):foundation 13%×80% + simd/math 17%×25% + sched 13%×60% + **io 17%×0% + compute 17%×0%** + graph/T3+延期 20%×5% ≈ **22%**。两法收敛 → 取 **≈22%**。
> - **为何不是更低的"个位%"**:那个数把**借入的 ggml ~100k 当成 HYLE 要写的**塞进分母(8.5/(37.5+100)≈6%)。但 ggml 是 compute 支柱**借核不造**(scope 明列),不计入 HYLE 自有完成度——剔除后回到 ≈22%。
>
> **22% 但"最难的一半未动工"(关键定性)**:io / compute / graph 三大支柱 = **0 行**,且占**剩余自有工作量的 ~58%**(17k / 29.4k 剩余),还是"异构混合极致性能"的卖点本体。已写的 22% 是**地基 + 一条窄缝**(mem/container/sched 核/simd f32 核/platform/util/time/error/obs),质量高、SOTA 对标、双平台+WSL TSan 验证(非 stub);但**广度薄**(simd 仅 f32 9 op,无 i8/i16 量化;AVX512/NEON 未硬件实跑)、**三大支柱未开工**。故:**按代码量 ≈22%,但按"差异化能力"主观更早——剩余的过半工作含全部头牌支柱。**

> **真实LOC = 该里程碑自写代码的 cloc 实测（生产+测试，不含文档、不含 vendored 第三方）。**每步闭环后回填。
> **预测/真实差距是"防玩具"信号**：差距大必须在该里程碑文档解释清楚——是"骨架版刻意精瘦、重活在后续里程碑"（合理），还是"偷懒 stub/空实现"（不可接受）。
> 已完成几步真实普遍 < 预测，因走的是 **walking skeleton 路径**（每步只做最小纵切，模块重头留加厚里程碑）；非玩具的证据是每步的真测 + TSan + 对抗式自审，详见各文档。

### 系统性 ~10x 高估的根因(2026-06-28 用户提出,数据调研)

观察:已完成里程碑真实 LOC 普遍是预测的 **1/5 ~ 1/20**(util/math 类小件最甚 ~10-18x)。**调研结论:主因是预测方法,非玩具**——三条证据:

1. **预测锚错基准**:roadmap "每里程碑 1-4k 行" 是按 **clay C++17 体量**估的。clay 满是 SFINAE/手写 Result/per-type allocator/C++17 polyfill;HYLE C++23 合法地更精简——`<bit>`/`<numbers>`/`std::expected`/concepts/`std::pmr` 把 clay 几百行样板**直接吃掉**(功能在 std 里,非没实现)。math-01 铁证:clay 458→28 行。
2. **倍率 ∝ 可压缩性**:gap 最大的恰是样板最密的 util/math(std 吃得最多);gap 最小的是 error/platform/SwissTable 等**不可约**真逻辑(~3-7x)。**反向证明非均匀偷工**——若玩具,大件会同样缩水。
3. **完整性真标尺 = clay API diff(coverage-map),非 LOC**。每里程碑闭环 diff clay 同名文件 API,缺的补或显式登记(container-04b/06b 即可见的登记延期)。"吸纳 ≠ 追平 LOC"。

**校准规则(此后预测按"可约性"分档,非一律 clay 体量):**
- **std 现代化重的小件**(util/math/纯类型设施):预测 **0.03-0.2k**(真实多 30-200 行)。
- **真算法/容器/平台原语**(hashmap/pool/parker/clock):预测 **0.1-0.5k**。
- **不可约大件**(path/fs 字符串处理、SIMD kernel、sched asm/调度、io 后端、compute):预测**接近 clay 体量**(path clay 1372 行→HYLE 估 0.6-1k;simd/sched/io/compute 仍 2-10k 级)。**这些若也缩成 1/10 才是真玩具信号,届时必须补实现而非改预测。**

### 完成时总量再投影 + "高估 vs 玩具"双向校准(2026-06-29,simd 支柱做完后)

> simd 支柱 6 里程碑做完 = **实测 ~2.3k 生产**(旧表预测 10-16k)。用户质疑:"确定高估,不是玩具?" 实测核验后的诚实拆解(两半,别只甩一边):
>
> 1. **不是玩具(铁证)**:simd 后端每个 **33 个真 op**(真 intrinsic)、**零 stub/TODO**、vec/dtype 测试断言 43/41 条 + 全 65536 fp16 模式硬件逐位对拍 + dot/saxpy 实测追平 Highway + 红队揪出真 ODR/SIGILL bug。玩具撑不住这些。
> 2. **但"5-7x 纯高估"是误导**:simd 2.3k vs Highway 100k **非功能对等**——HYLE 现做 **4 lane 类型 + ~25 op + 2 ISA 实测**;Highway 是 **12 类型 + ~200 op(swizzle/gather/scatter/compress/table-lookup,HYLE 全无)+ 6+ ISA**。dot/saxpy 追平只证"**已做的 op 上**追平"。i8/u8/i16/u16/i64/u64 量化类型等是**登记延期的广度**,补齐后 simd 涨到 **~5-8k**(非停在 2.3k)。
>
> **故 simd 真实高估约 2x(非 6x)**;foundation 的 5-6x 高估是真(std 取代)。校准规则 #3 对 simd "2-10k 级"基本对,只是要明确**2.3k 是"核心数值子集做好",非"全广度做完"**。
>
> **完成时总量(实测密度再投影,含广度补全)**:HYLE 自有 **~30-45k 生产 / ~48-72k 含测试**(旧 76-120k);**第三方 vendored ~150-250k+(ggml 100k+ 主导)**;项目总足迹 ~200-300k+。详见 [`../architecture/hyle-scope.md` 规模节](../architecture/hyle-scope.md)。

## 全部里程碑（阶段 × 模块）

| ID | 阶段 | 模块 | 目的(一行) | 前置 | 预测LOC | 真实LOC | 状态 |
|---|---|---|---|---|---|---|---|
| **00-scaffolding** | P-1 | 构建 | CMake 多模块结构+GoogleTest/nanobench/sanitizer/CI+HYLE_宏骨架 | — | 1-2k | 294 | ✅ 完成 |
| error-01 | P0 | error | Status + std::expected + ASSIGN_OR_RAISE 宏 | 00 | 1-2k | 456 | ✅ 完成(进程内 Status;error_handler→error-02) |
| error-02 | P0 | error | error_handler vtable(跨 DLL 错误上报) | error-01 | 1k | 127 (+74 测试) | ✅ 完成(非变参 vtable ABI 安全;干净 HYLE_TRAP fatal;补 charter §5 HYLE_ASSERT;record 无分配) |
| obs-01 | P0 | obs | 结构化日志产出:level 门控 + sink 注册表 + 默认 stderr 彩色 sink | 00 | 1-2k | 133 (+143 测试) | ✅ 完成(吸纳 clay log;新建 obs 模块;非变参 sink+caller-side std::format 避 clay 变参 fn-ptr ABI 坑;atomic level 无锁门+mutex 快照锁外 I/O 修 clay 裸数组 race;双平台 12/12 + 并发压测过 WSL TSan;曾试 seqlock 无锁读被 TSan 判 UB 撤回。instrumentation hook=obs 后续步) |
| platform-01 | P1 | platform | thread + park(futex/WaitOnAddress/ulock) | error-01 | 2-3k | 527 | ✅ 完成 |
| platform-02 | P1 | platform | virtual_memory + dynlib | platform-01 | 2-3k | 158 (dynlib) | 🔶 dynlib ✅(DynLib RAII,Win/Linux 真测);virtual_memory→platform-02b |
| platform-03 | P1 | platform | path + fs + env + os_info | platform-01 | 1-2k | — | 🔶 拆分 |
| platform-03a | P1 | container | path 字符串操作(join/dirname/basename/extension/normalize/split/is_absolute) | container-06,container-03 | 0.6-1k | 232 (+166 测试) | ✅ 完成(吸纳 clay path_api_t 16 函数纯字符串半;落 container 非 platform——避 P1 反向依赖 P3;3 处语义升级:splitext extension/真 is_absolute/normalize native;修 directory 的 //x OOB;clay path.cc 1372 行无美化空函数,vtable 外 ~1100 行 fs/blend/Win 域分流登记) |
| platform-03b | P1 | platform | fs/env/os_info syscall:stat/mkdir/remove/rename/copy/iterate/cwd/temp/home + getenv | platform-01,platform-03a | 0.5-1k | 379 (+162 测试) | ✅ 完成(file_system_api 13 fn + helper;buffer-fill+visitor 保 P1 零 mem/container 依赖,比 clay 分配式更精简;改进:mkdir 幂等/temp 认环境变量/copy 可移植替 sendfile/env 读净增;双平台 14/14 fs,platform 19→33;file_io/mmap→io-05、avx→simd、absolute 组合登记) |
| [platform-03c](./platform-03c-win-utf8.md) | P1 | platform | Win UTF-8(...A→...W)+ 长路径 \\?\ + copy_file 内核零拷贝 | platform-03b | 0.1k | fs.cc 320→475 | ✅ 完成(WidePath/fill_w 自包含 UTF-8↔UTF-16,12 个 ...A 全转 ...W;揪出+修 touch() 双重 ANSI;\\?\ 扩展前缀 >260 长路径(drive 测过/UNC 写不验);Linux copy_file_range 内核零拷贝失败回退。双平台 36/18,含 UTF-8/长路径/大文件拷贝测试。UNC 实跑待网络共享) |
| mem-01 | P2 | mem | Allocator vtable + 修 alignment（骨架版含最小 mimalloc 包装）| platform-01 | 1-2k | 182 | ✅ 完成(vtable+裸alloc+typed层) |
| mem-01b | P2 | mem | allocator typed 便利层(allocate<T>/construct/make_unique/destroy) | mem-01 | 1k | 并入allocator.h | ✅ 完成(补漏:clay allocator.h typed层) |
| mem-02 | P2 | mem | mimalloc backing 完整接入(封 vtable 后) | mem-01 | 1-2k | vendored(system.cc+~30) | ✅ 完成(透明替换31测试不变;Win 4-12x>默认堆/Linux真实模式5.2x;快路径修复) |
| mem-03 | P2 | mem | MemoryArena 无锁块池 + 修 64 槽上限 | mem-01 | 2-3k | 209 | ✅ 完成(块池+Arena bump) |
| mem-03b | P2 | mem | Arena bump 分配器 + ArenaAllocator 适配 + make_arena_allocator | mem-03 | 1k | 并入memory_arena.h | ✅ 完成(补漏:clay memory_arena.cc 另半) |
| mem-04 | P2 | mem | ObjectPool + StackAllocator 推倒重写(修 UB) | mem-02 | 2-3k | 197 | ✅ 完成(坐 mem-03 上;修 bug#3/#4) |
| mem-05 | P2 | mem | ~~MemorySpace 组合抽象 + xcpu_freelist~~ | mem-03 | — | — | ⛔ 双双否决:MemorySpace=异构内存过度设计(device 边界=显式 copy,统一内存无互传,见 mem-design);xcpu_freelist=同线程常数优化,与固定 slab"不假性耗尽"硬冲突+生命周期耦合,评估后不采用(跨核 free 正确性已由地址路由原子 push 给,理由见 mem-design) |
| mem-06 | P2 | mem | NUMA + 大页 + stats(修下溢) | mem-05 | 2-3k | — | 占位 |
| mem-07 | P2 | mem | GrowablePool(growable/单线程池) + PmrResource(std::pmr 适配) | mem-03 | 1-2k | 145 (+116 测试) | ✅ 完成(吸纳 clay memory_pool/stl_allocator/generic_allocator;修 clay move-assign bug;generic_allocator⛔=Allocator 已是它) |
| time-01 | P2 | time | 单调钟(CLOCK_MONOTONIC/QPC) | platform-01 | 1k | 103 (+62 测试) | ✅ 完成(加 clay 缺的单调钟;ns精度;Instant) |
| [time-02](./time-02-timer-wheel.md) | P2 | time | 分层时间轮 | time-01 | 1-1.5k | 167 轮 + 90 platform timeout (+213 测试 +115 bench) | ✅ 完成(6层×64槽 tokio 移植:occupied 位图不空转 + O(1) 摘链 cancel 无墓碑 + cascade 降级;platform 加 park_timeout 执行底座;clay 无时间轮=net-new;双平台 15+9 测试绿 + TSan 净;churn 负载对 min-heap 双平台赢 2.1x/3.3x。**解锁 sched-04b**) |
| container-01 | P3 | container | vendor 有界 ring(Vyukov/rigtorp,骨架用) | mem-01 | 1-2k | 220 | ✅ 完成 |
| container-02 | P3 | container | XXH3 接入(默认 hasher) | 00 | 1k | 133 | ✅ 完成(XXH3 封 .cc 不泄;小键内联) |
| container-03 | P3 | container | carray header-before-data + RAII 壳 | mem-02 | 1-2k | 224 | ✅ 完成(type-erased核;修over-align) |
| container-04 | P3 | container | SwissTable 探测层重写(concept 模板)【大】 | container-02 | 3-4k | 474 | ✅ 完成(find/insert/erase/grow;type-erased→04b) |
| container-04b | P3 | container | SwissTable type-erased core(防per-T膨胀)+ erase空槽回收 + NEON | container-04 | 1-2k | 68 (+62 probe +84 shell重构) | ✅ 完成(type-erase 仅冷路径 resize/free 进 DLL,hot find/insert/erase 留 inline 零回归;erase abseil reclaim-to-empty 还 growth;NEON aarch64 GCC14 编过+指令验证、无 ARM 机未跑;双平台 11/11,132→133) |
| [container-04c](./container-04c-find-or-prepare.md) | P3 | container | SwissTable insert find_or_prepare 融合(单趟探测) | container-04b | 0.05k | +50(shell) | ✅ 完成(红队点名:新键原 2×哈希+2×探测→单趟同时查命中+记空/删槽,哈希算一次;附带修 growth_left 复用墓碑双扣 bug;双平台 11/11 含 oracle 对拍 50k/200k) |
| container-05 | P3 | container | SmallVector + intrusive list/slist | mem-02 | 1-2k | 255 | ✅ 完成(新增,抄 folly 设计) |
| container-06 | P3 | container | string_ref + string_buf 驻留池 | container-02 | 1-2k | 363 (+341 测试) | ✅ 完成(string_ref 类型核心+零拷贝算法;StringBuf pimpl 复用 SwissTable;高层工具→06b) |
| container-06b | P3 | container | string_algo 字节串算法(hex/escape/capitalize/UTF-8/ASCII case) | container-06 | 1-2k | 175 (+92 测试) | ✅ 完成(吸纳 clay 真实现部分;重大发现 format_si/find_path_matches 等 clay 只声明无实现→06c;修 hex_to_int 静默0 bug) |
| container-06c | P3 | container | 模糊路径匹配(score_similarity/path_match/find_path_matches) | container-06b | 0.5-1k | — | 占位(**clay 只声明无实现→净新设计**,非吸纳;需要时做) |
| container-07 | P3 | container | 有界 Chase-Lev work-stealing deque(owner LIFO/DFS,供 graph fork-join) | mem-01 | 1-2k | 249 | ✅ 完成(用途:graph-01,非调度器本地队列) |
| container-08 | P3 | container | 有界 work-stealing FIFO ring(tokio queue.rs:owner push免同步/pop+thief CAS packed-head/偷半) | mem-01 | 1-2k | 279 | ✅ 完成(sched-03 真前置) |
| container-09 | P3 | container | ConcurrentVector 读优化并发向量(分段/读 lock-free/写 SpinLock) | mem-01,util-05 | 1-2k | 87 (+98 测试) | ✅ 完成(吸纳 clay parallel/concurrent_vector;修五问题:零拷贝1x内存/move-only/结构性不变量;WSL TSan 零 race) |
| [container-10](./container-10-sharded-queue.md) | P3 | container | ShardedQueue 分片有界 MPMC(N 个 Vyukov 子环,生产者 hash 分片消单点争用) | container-01 | 0.5-1k | 58 (+171 测试) | ✅ 完成(bench-02b 揪出单环 4P 输 moody 1.7x=单点 CAS 争用→分片根治;**双平台15遍反超 rigtorp Win3.27x/Lin5.3x**;复用 BoundedQueue placement-new 避 std 拥有型成员;保留有界弃全局 FIFO;双平台+WSL TSan 零 race。**已接入 sched global 注入**) |
| util-01 | P3 | util | ref_counted(Ref<T> CRTP) + scope_guard(HYLE_DEFER) | 00 | 1k | 168 (+186 测试) | ✅ 完成(修 clay destroy CRTP bug;decref release+fence;concepts 替 SFINAE) |
| util-02 | P3 | util | 编译期 type_name<T> + type_hash<T> | 00 | 0.5k | 44 (+50 测试) | ✅ 完成(probe 派生偏移>clay 硬编码;FNV-1a constexpr) |
| util-03 | P3 | util | registry 自注册 + DECLARE/DEFINE 宏 | util-02,container-04 | 1k | 94 (+92 测试) | ✅ 完成(SwissTable 替3map;函数指针 Entry 内联;fail-fast 替 exit;三套收一套) |
| util-04 | P3 | util | random(xoshiro256++ PRNG) | 00 | 0.5k | 99 (+103 测试) | ✅ 完成(升级 clay xoroshiro128+;Lemire 无偏 range 修 clay `%` 偏差;jump 并行流) |
| util-05 | P3 | util | spin_lock(TTAS+pause+padding) | 00 | 0.5k | 36 (+69 测试) | ✅ 完成(升级 clay 裸 TAS;满足 Lockable;TSan 互斥验证) |
| util-06 | P3 | util | demangle 运行期符号反修饰(`__cxa_demangle`/`UnDecorateSymbolName`) | util-02 | 0.5k | 65 (+46 测试) | ✅ 完成(buffer-fill 无分配 vs clay Allocator+std::function;MSVC 真反修饰;type_name 运行期补集) |
| math-01 | P3 | math | scalar 标量助手(div_ceil/round_up_to_multiple/sgn/approx_eq) | 00 | 0.5k | 28 (+51 测试) | ✅ 完成(clay consts/utils 458 行绝大多数♻️ C++20/23 `<bit>`/`<numbers>`/`<cmath>`;只自写 std 真缺的4个;div_ceil 改溢出安全) |
| bench-01 | P3 | bench | nanobench 设施 + SwissTable/StringBuf 对标 std | container-04/06 | — | bench设施 | ✅ 完成(SwissTable find2.7x/insert6.1x/erase4.1x>std;intern基准驱动优化0.94→1.31x;absl/F14对标→bench-02) |
| bench-02 | P3+ | bench | 对标 SOTA(absl/F14/moodycamel/oneTBB/highway/marl,不止 std) | 各模块 | — | — | 📋 规划(用户提:对比 project-hyle 高性能库;拆 02a hashmap/02b 队列/02c hash+wyhash/02d simd-sched) |
| simd-01 | P3 | simd | vec D-tag 抽象 + scalar 后端 + 运行期 CPUID dispatch 骨架 | mem-01 | 2-3k | 496 (+210 测试) | ✅ 完成(dispatch.h/cc 运行期 CPUID+XGETBV 探测,探测器独立 TU+禁 LTO 防自身 SIGILL;vec.h 照搬 Highway D-tag 契约 Simd<T,N,kPow2>/Lanes(d) 可运行期求值留 SVE 口;scalar 地板后端核心 op 全集作后续后端对拍参照;双平台 22/22。用户铁律"一开张就搭 dispatch 骨架"已落实。3 红队 agent+oracle 核账揪修 4 硬伤(0x80000001/LZCNT 真补、tier 真对标 Highway group、死码、过度承诺注释诚实化)。SSE2/AVX2/AVX512/NEON 各后端→simd-02/03,变体 DLL 选择→simd-04) |
| simd-02 | P3 | simd | SSE4.2 + AVX2 后端 | simd-01 | 2-3k | 730 (后端) +161 bench | ✅ 完成(sse4.h 128b + avx2.h 256b,lane f32/f64/i32/u32 native 全宽;每 op 对拍 scalar,双平台三后端各 24/24;**对标 Highway 统计平局**——dot/saxpy/reduce_max/scale 四 kernel 15 遍中位数 ratio 全 ~1.00(gap«spread),详见 performance.md。**2026-06-30 更正**:此前标 dot"~1.10x"、saxpy"~1.21x 微超"是假象,源于 bench 给 Highway 对照 TU 的 ISA flag 不全(`-mavx2 -mfma -mf16c` 缺 BMI2/AES/PCLMUL),GCC 下 Highway 静态 dispatch 降级 SSSE3/128b,实为 256b vs 128b 不公平对照;补 `hyle_set_highway_isa` 后归于平局——印证 vec.h 抽象零成本;intrinsic 逐个对照 Highway x86_128/256;AndNot 折进 bitop 宏。i8/u8/i16/u16→simd-05,i64/u64+partial 宽→待需,变体 DLL 选择→simd-04。**红队 2026-06 揪修真 bug:`Ne(NaN,NaN)` 用有序谓词 `_CMP_NEQ_OQ`(NaN 返 false)与 scalar/SSE4/NEON 发散→改 `_CMP_NEQ_UQ`(无序),AVX2 真机 FloatEdgeCases 测试覆盖**) |
| [simd-03](./simd-03-neon.md) | P3 | simd | AVX512 + NEON 后端 | simd-01 | 2-3k | 312+330 | 🔶 NEON + AVX512 均**写+编+objdump 验,未跑存疑**。NEON:128b aarch64 对照 Highway arm_neon,交叉编译+objdump(fmla/faddp/fcmgt/addv);AVX512:512b F+BW+VL+DQ,自定义 Mask512 包 __mmask k 寄存器,原生编+objdump(vfmadd zmm/vcmpgt→k1/vpcmpud→k0+popcnt)。**写 AVX512 证明 vec.h per-backend mask 抽象够用**(纠先前"需改造"误判)。两者未接 ops/dtype 运行期(降级跑 AVX2 安全),待 ARM 机/qemu + AVX512 机/SDE 真跑。**红队 2026-06 揪修 AVX512 两 bug(objdump 确认):`Ne` 同 AVX2 的 `_CMP_NEQ_OQ`→`_CMP_NEQ_UQ`(立即数 0x04);`Neg(+0.0)` 用 `0-a` 得 +0.0 与 SSE4/AVX2 发散→改 `vxorps` 符号位翻转** |
| simd-04 | P3 | simd | 运行期变体 dispatch(fn-ptr 表,best_isa 选 kernel) | simd-02 | 1-2k | 111 (+67 测试) | ✅ 完成(ops.h dot_f32/saxpy_f32/active_variant 派发入口;fn-ptr 表模型像 Highway HWY_DYNAMIC_DISPATCH——每 ISA kernel 独立 -march TU,baseline dispatcher 按 best_isa 选 fn-ptr Meyers 缓存;**闭合 simd-01 审计的 best_isa 死代码**;红队**代码分析**核 SIGILL 安全成立(老 CPU 永不取址 avx2 符号+FMA gating;未在非 AVX2 机实跑,待 QEMU/sde 补真验);修掉 kNativeVectorBytes inline→static 的 IFNDR ODR bug;双平台 36/36。变体 DLL 零间接模型→后续) |
| simd-05 | P3 | simd | dtype fp16/bf16(F16C/AVX512-FP16/BF16) | simd-02 | 2-3k | 187 (+151 测试) | ✅ 完成(float16_t/bfloat16_t + 标量转换核逐行照搬 Highway base.h;fp16 批量走 F16C 硬件变体 TU(_mm256_cvtph_ps,门控 f16c&&avx)+ scalar 兜底;bf16 移位 RNE。**全 65536 fp16 位模式硬件-vs-标量逐位对拍 + RNE tie-to-even oracle**;双平台 47/47。红队揪修 2 个本机测不到的 SIGILL bug(YMM 缺 os 门→f16c&&avx;F16C TU 误编 avx2→降 -mavx,防 Ivy Bridge SIGILL)+ 测试漏洞(RNE 未测/非 F16C tautology→GTEST_SKIP)。AVX512-FP16/BF16+int8/fp8→后续,待 Ivy Bridge/QEMU 补真验) |
| [simd-06](./simd-06-sleef-math.md) | P3 | simd | 数学核 adapter(exp/log/tanh) | simd-04 | 1-2k | 103 (+53 测试) | ✅ 完成(math.h exp/log/tanh_f32_batch + active_math_variant;**scalar libm 核双平台永远在**(ULP oracle+兜底)+ **向量化核 Highway+xsimd**(借核不手搓,逐 op best-of-breed:exp/tanh→xsimd、log→Highway)。**现状**:Highway/xsimd 已 vendored 在 third_party/、无条件默认编进生产(运行期 avx2 门控落 scalar);最初的 SLEEF 后端已退役(非 header-only 需预编库+被 xsimd/Highway 全盖,见 simd-06 篇)。三 op 全 ≥ 两库 SOTA(exp 0.23 平 xsimd、log 0.26 反超两库、tanh 0.44 平 xsimd),详见 performance.md。math_test 收紧到 4e-7(≈3.4 ULP);log/tanh tail 补测) |
| sched-01 | P4 | sched | fcontext 6 平台 + 栈池(骨架:单 worker 单队列跑一 fiber) | container-01,mem-01,platform-01 | 2-3k | 226 (+737 vendored asm) | ✅ 完成(仅证切换,队列/worker 留 sched-02) |
| sched-02 | P4 | sched | per-core 有界 run-queue + LIFO slot | sched-01,container-01 | 2-3k | 205 | ✅ 完成(单worker驱动,steal留sched-03) |
| sched-03 | P4 | sched | work-stealing + spin-then-park + unpark 接力 | sched-02,container-08 | 2-3k | 340 | ✅ 完成(多worker偷取+park;§2.3②③防惊群当时诚实延迟,**后于 sched-05 以 conditional-notify 落地**) |
| sched-04 | P4 | sched | Waker completion 脊 + time 接入(deadline) | sched-02,time-02 | 1-2k | — | 🔶 拆分 |
| sched-04a | P4 | sched | Waker completion 脊(suspend/wake,race-free) | sched-03 | 1k | ~114 (+96 测试) | ✅ 完成(架构拱顶:io/compute/timer 共用的唤醒接缝;wake-before-suspend 对称原子 rendezvous;双平台 4/4 + WSL TSan 8/8 零 race。deadline 半→04b 待 time-02) |
| sched-04b | P4 | sched | fiber sleep via timer wheel(deadline 到期 wake) | sched-04a,time-02 | 0.5-1k | 113 (+156 测试) | ✅ 完成(per-worker TimerWheel;sleep_for_ns/sleep_until_ns arm 本 worker 轮 O(1) 无分配 + suspend;worker 循环顶 fire_due_timers(wheel.empty() 门=无 timer 零税)+ park 用 next_deadline 喂 park_timeout 不空转;到期走 sched-04a Waker rendezvous reinject。双平台 33/33 + TSan×3 净;红队 7 类竞态审计无 bug。wait_for/wait_until→sched-04c) |
| sched-04c | P4 | sched | 定时 CV 等待(wait_for/wait_until)+ 跨 worker timer cancel | sched-04b | 0.5-1k | — | 📋 就绪(sched-04b 解锁。核心=timeout-vs-notify 竞态:败者取消对方;fiber 可能在异于 timer 归属 worker 的核恢复→需 per-worker MPSC 命令队列做跨核 cancel=time-02 头号分歧点。落地即补 sync.h wait_for/wait_until) |
| [sched-05](./sched-05-sync-primitives.md) | P4 | sched | fiber-aware CV + mutex/waitgroup/channel/latch | sched-04a | 2-3k | 290 (+297 测试) | ✅ 完成(marl 收敛:一个双模 `ConditionVariable` 地基[FIFO waiter 链表+SpinLock] + 其上薄封装 Mutex/WaitGroup/Latch/Channel[有界 MPMC];**双模**:fiber waiter→`suspend()`重入 run-queue、裸线程 waiter→`platform::Parker`(跨进程/外部线程提交,非 std::cv)。**直接复用 sched-04a Waker rendezvous**,wake-before-suspend race-free 非新机制。双平台 28/28(原20+新8 Sync)+ **WSL TSan 零 data race**;clay async_value/barrier 见下。`wait_for`定时变体依赖 time-02→sched-04b 登记延后。**benchmark + SOTA 对标 marl**(`sync_bench.cc`/`sync_marl_bench.cc`,gated):初版 fiber-aware Mutex 实测**慢 marl 5.8x**(短临界区让出 fiber 太贵)→**重写为 futex(Drepper 3 态,建在 platform::wait_on_address)→打平 marl**(30.7 vs 32.1ns),学 marl"mutex 用 futex、fiber-aware 只留 CV"。CV 的 waiter-count 门让 Channel notify 无 waiter 时零锁(63ns/15.9M/s)。**WaitGroup fork-join 反超 marl 1.49x(205ns vs 306ns)、spawn-2000 反超 2.17x**:起初慢 17x,诊断推翻"无栈池"猜想——真根因是**调度器急泊车的 futex 唤醒风暴**,落地 **conditional-notify**(`num_searching_` 计数门 + seq_cst Dekker,有 worker 在搜就不 unpark,源头消除风暴;spin-then-park 是中间缓解版,已被取代)后反超;栈池(MemoryArena::as_allocator)再加 ~13%(→179ns,1.71x)。**附带修 TSan×fcontext 盲区**:fiber 跨 worker 迁移增多撞出 TSan SEGV(非逻辑 bug,无 TSan 50/50 过),给 fiber.cc 加 `__tsan_switch_to_fiber` 注解(release 零成本)→ 双平台 28/28 + TSan 28/28 绿。详见 performance.md) |
| sched-06 | P4 | sched | 协程胶水(co_await→fiber 让出) | sched-04 | 1-2k | — | 占位 |
| sched-07 | P4 | sched | 取消：CancelToken 树 + await 取消点 + 结构化并发 | sched-04,sched-06 | 2-3k | — | 占位 |
| sched-08 | P4 | sched | runtime 优雅关闭(两阶段 drain + 逆序清理) | sched-03 | 1-2k | — | 占位 |
| io-01 | P5 | io | proactor 抽象(Waker 接 sched) | sched-04 | 1-2k | — | 占位 |
| io-02 | P5 | io | io_uring 后端(借 liburing) | io-01 | 2-3k | — | 占位 |
| io-03 | P5 | io | epoll + kqueue + IOCP 后端 | io-01 | 3-4k | — | 占位 |
| io-04 | P5 | io | net(tcp/udp socket) | io-02 | 2-3k | — | 占位 |
| io-05 | P5 | io | file + 跨进程共享内存 | io-02 | 2-3k | — | 占位 |
| compute-01 | P5 | compute | device/stream 薄包 ggml-backend | sched-04 | 2-3k | — | 占位 |
| compute-02 | P5 | compute | device buffer(ggml buffer_type 式,compute 自有所有权+stream) + 显式 copy(src,dst,stream) | compute-01 | 2-3k | — | 占位(device 内存归 compute/,走显式 copy;统一内存无互传,mem/ 不参与) |
| compute-03 | P5 | compute | GPU completion→fiber(直连原生 handle) | compute-01,sched-04 | 1-2k | — | 占位 |
| compute-04 | P5 | compute | kernel 启动抽象 | compute-01 | 2-3k | — | 占位 |
| compute-05 | P5 | compute | ggml_bridge + 注入 HYLE 线程池 | compute-01 | 2-3k | — | 占位 |
| compute-06 | P5 | compute | Vulkan compute 自定义 kernel | compute-04 | 2-3k | — | 占位 |
| graph-01 | P6 | graph | taskgraph 剥特化 + 绑 sched | sched-03 | 3-4k | — | 占位 |
| [close-01](./close-01-c-abi-shim.md) | P6 | 收口 | C-ABI shim(对外纯 C 头) | 各模块稳定 | 2-3k | — | 📋 **承诺已锁定,实现待排期**。产品铁律:发布给用户绑定的唯一稳定 ABI 面=纯 extern "C" C shim;C++ 公开头不承诺跨编译器/跨语言。未做此前 HYLE 不具"可发布给 FFI"资格。详见里程碑文档 |
| close-02 | P6 | 收口 | serialize(simdjson/flatbuffers) + hash + T3 便利 | container | 4-6k | — | 占位 |
| close-03 | P6 | 收口 | FFI 绑定(TS N-API / Python ctypes) | close-01 | 2-3k | — | 占位 |

约 47 个里程碑。状态：✅ 完成 / 进行中 / 占位(JIT 补全)。真实LOC 闭环后回填，差距大见对应文档解释。

> 依赖列只标**直接前置**；完整依赖图见 `roadmap.md` 第一节。`00-scaffolding` 是所有里程碑的隐含前置（万物在它上面编译）。
