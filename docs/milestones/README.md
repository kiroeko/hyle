# HYLE 里程碑索引

> 实现的真正"步骤"在这里——约 47 个里程碑，每个 1-4k 行、可独立编/测/基准。`roadmap.md` 是依赖层(阶段)的拓扑序与决策；本目录是**可执行的逐步计划**，每步一文档。
>
> **粒度原则**：一个里程碑 = 一次"写完→编过→单测(覆盖达 Tier 标准)→基准→提交"的闭环。
>
> **生长方式（JIT）**：最先执行的 `00-scaffolding` 与 walking-skeleton 几步已写完整；其余**临到该模块开工前再补全细节**——后期里程碑依赖前面实现学到的东西，提前写=空想。索引行先占位（ID/依赖/目的/LOC），细节随推进生长。
>
> **clay 吸纳完整性标尺**：见 [`../architecture/clay-coverage-map.md`](../architecture/clay-coverage-map.md)——clay 每个文件 → HYLE 对应物/状态/LOC 的逐文件映射。**每个吸纳 clay 的里程碑闭环前必须 diff clay 同名文件 API 表面**，缺的补或显式登记，不许默默略过(红队核账 2026-06 立此纪律)。

## 每个里程碑文档的模板

见 [`_TEMPLATE.md`](./_TEMPLATE.md)。固定章节：**前置依赖 / 目的 / 在架构中的位置 / 具体事务 / 交付物 / 验收标准(含覆盖 Tier) / 参考**。

## Walking Skeleton（最先做，纵切验脊）

打通"**一个 fiber 跑起→让出→被唤醒**"。= `00` + 各模块**第一个最小里程碑**串起来，**先不做 work-stealing/多后端**。验掉最高风险的 fcontext+调度脊，之后逐里程碑加厚。

骨架路径：`00-scaffolding` → `error-01` → `platform-01` → `mem-01` → `container-01` → `sched-01`(单 worker 单队列)。

## 预测 vs 真实 LOC（防玩具化）

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

## 全部里程碑（阶段 × 模块）

| ID | 阶段 | 模块 | 目的(一行) | 前置 | 预测LOC | 真实LOC | 状态 |
|---|---|---|---|---|---|---|---|
| **00-scaffolding** | P-1 | 构建 | CMake 多模块结构+GoogleTest/nanobench/sanitizer/CI+HYLE_宏骨架 | — | 1-2k | 294 | ✅ 完成 |
| error-01 | P0 | error | Status + std::expected + ASSIGN_OR_RAISE 宏 | 00 | 1-2k | 456 | ✅ 完成(进程内 Status;error_handler→error-02) |
| error-02 | P0 | error | error_handler vtable(跨 DLL 错误上报) | error-01 | 1k | 127 (+74 测试) | ✅ 完成(非变参 vtable ABI 安全;干净 HYLE_TRAP fatal;补 charter §5 HYLE_ASSERT;record 无分配) |
| obs-01 | P0 | obs | instrumentation hook API(编译期 no-op 空实现) | 00 | 1-2k | — | 占位 |
| platform-01 | P1 | platform | thread + park(futex/WaitOnAddress/ulock) | error-01 | 2-3k | 527 | ✅ 完成 |
| platform-02 | P1 | platform | virtual_memory + dynlib | platform-01 | 2-3k | 158 (dynlib) | 🔶 dynlib ✅(DynLib RAII,Win/Linux 真测);virtual_memory→platform-02b |
| platform-03 | P1 | platform | path + fs + env + os_info | platform-01 | 1-2k | — | 🔶 拆分 |
| platform-03a | P1 | container | path 字符串操作(join/dirname/basename/extension/normalize/split/is_absolute) | container-06,container-03 | 0.6-1k | 232 (+166 测试) | ✅ 完成(吸纳 clay path_api_t 16 函数纯字符串半;落 container 非 platform——避 P1 反向依赖 P3;3 处语义升级:splitext extension/真 is_absolute/normalize native;修 directory 的 //x OOB;clay path.cc 1372 行无美化空函数,vtable 外 ~1100 行 fs/blend/Win 域分流登记) |
| platform-03b | P1 | platform | path/os syscall 半:getcwd/absolute/temp_dir/directory_entries + 宽字符↔UTF-8 + Win 盘符/UNC normalize + safe_filename | platform-01,platform-03a | 0.5-1k | — | 占位(clay path.cc vtable 外 + os.*.cc syscall) |
| mem-01 | P2 | mem | Allocator vtable + 修 alignment（骨架版含最小 mimalloc 包装）| platform-01 | 1-2k | 182 | ✅ 完成(vtable+裸alloc+typed层) |
| mem-01b | P2 | mem | allocator typed 便利层(allocate<T>/construct/make_unique/destroy) | mem-01 | 1k | 并入allocator.h | ✅ 完成(补漏:clay allocator.h typed层) |
| mem-02 | P2 | mem | mimalloc backing 完整接入(封 vtable 后) | mem-01 | 1-2k | vendored(system.cc+~30) | ✅ 完成(透明替换31测试不变;Win 4-12x>默认堆/Linux真实模式5.2x;快路径修复) |
| mem-03 | P2 | mem | MemoryArena 无锁块池 + 修 64 槽上限 | mem-01 | 2-3k | 209 | ✅ 完成(块池+Arena bump) |
| mem-03b | P2 | mem | Arena bump 分配器 + ArenaAllocator 适配 + make_arena_allocator | mem-03 | 1k | 并入memory_arena.h | ✅ 完成(补漏:clay memory_arena.cc 另半) |
| mem-04 | P2 | mem | ObjectPool + StackAllocator 推倒重写(修 UB) | mem-02 | 2-3k | 197 | ✅ 完成(坐 mem-03 上;修 bug#3/#4) |
| mem-05 | P2 | mem | MemorySpace 组合抽象 + xcpu_freelist | mem-03 | 2-3k | — | 占位 |
| mem-06 | P2 | mem | NUMA + 大页 + stats(修下溢) | mem-05 | 2-3k | — | 占位 |
| mem-07 | P2 | mem | GrowablePool(growable/单线程池) + PmrResource(std::pmr 适配) | mem-03 | 1-2k | 145 (+116 测试) | ✅ 完成(吸纳 clay memory_pool/stl_allocator/generic_allocator;修 clay move-assign bug;generic_allocator⛔=Allocator 已是它) |
| time-01 | P2 | time | 单调钟(CLOCK_MONOTONIC/QPC) | platform-01 | 1k | 103 (+62 测试) | ✅ 完成(加 clay 缺的单调钟;ns精度;Instant) |
| time-02 | P2 | time | 分层时间轮 | time-01 | 1-1.5k | — | 占位 |
| container-01 | P3 | container | vendor 有界 ring(Vyukov/rigtorp,骨架用) | mem-01 | 1-2k | 220 | ✅ 完成 |
| container-02 | P3 | container | XXH3 接入(默认 hasher) | 00 | 1k | 133 | ✅ 完成(XXH3 封 .cc 不泄;小键内联) |
| container-03 | P3 | container | carray header-before-data + RAII 壳 | mem-02 | 1-2k | 224 | ✅ 完成(type-erased核;修over-align) |
| container-04 | P3 | container | SwissTable 探测层重写(concept 模板)【大】 | container-02 | 3-4k | 474 | ✅ 完成(find/insert/erase/grow;type-erased→04b) |
| container-04b | P3 | container | SwissTable type-erased core(防per-T膨胀)+ erase空槽回收 + NEON | container-04 | 1-2k | — | 占位(显式补漏) |
| container-05 | P3 | container | SmallVector + intrusive list/slist | mem-02 | 1-2k | 255 | ✅ 完成(新增,抄 folly 设计) |
| container-06 | P3 | container | string_ref + string_buf 驻留池 | container-02 | 1-2k | 363 (+341 测试) | ✅ 完成(string_ref 类型核心+零拷贝算法;StringBuf pimpl 复用 SwissTable;高层工具→06b) |
| container-06b | P3 | container | string_algo 字节串算法(hex/escape/capitalize/UTF-8/ASCII case) | container-06 | 1-2k | 175 (+92 测试) | ✅ 完成(吸纳 clay 真实现部分;重大发现 format_si/find_path_matches 等 clay 只声明无实现→06c;修 hex_to_int 静默0 bug) |
| container-06c | P3 | container | 模糊路径匹配(score_similarity/path_match/find_path_matches) | container-06b | 0.5-1k | — | 占位(**clay 只声明无实现→净新设计**,非吸纳;需要时做) |
| container-07 | P3 | container | 有界 Chase-Lev work-stealing deque(owner LIFO/DFS,供 graph fork-join) | mem-01 | 1-2k | 249 | ✅ 完成(用途:graph-01,非调度器本地队列) |
| container-08 | P3 | container | 有界 work-stealing FIFO ring(tokio queue.rs:owner push免同步/pop+thief CAS packed-head/偷半) | mem-01 | 1-2k | 279 | ✅ 完成(sched-03 真前置) |
| container-09 | P3 | container | ConcurrentVector 读优化并发向量(分段/读 lock-free/写 SpinLock) | mem-01,util-05 | 1-2k | 87 (+98 测试) | ✅ 完成(吸纳 clay parallel/concurrent_vector;修五问题:零拷贝1x内存/move-only/结构性不变量;WSL TSan 零 race) |
| util-01 | P3 | util | ref_counted(Ref<T> CRTP) + scope_guard(HYLE_DEFER) | 00 | 1k | 168 (+186 测试) | ✅ 完成(修 clay destroy CRTP bug;decref release+fence;concepts 替 SFINAE) |
| util-02 | P3 | util | 编译期 type_name<T> + type_hash<T> | 00 | 0.5k | 44 (+50 测试) | ✅ 完成(probe 派生偏移>clay 硬编码;FNV-1a constexpr) |
| util-03 | P3 | util | registry 自注册 + DECLARE/DEFINE 宏 | util-02,container-04 | 1k | 94 (+92 测试) | ✅ 完成(SwissTable 替3map;函数指针 Entry 内联;fail-fast 替 exit;三套收一套) |
| util-04 | P3 | util | random(xoshiro256++ PRNG) | 00 | 0.5k | 99 (+103 测试) | ✅ 完成(升级 clay xoroshiro128+;Lemire 无偏 range 修 clay `%` 偏差;jump 并行流) |
| util-05 | P3 | util | spin_lock(TTAS+pause+padding) | 00 | 0.5k | 36 (+69 测试) | ✅ 完成(升级 clay 裸 TAS;满足 Lockable;TSan 互斥验证) |
| util-06 | P3 | util | demangle 运行期符号反修饰(`__cxa_demangle`/`UnDecorateSymbolName`) | util-02 | 0.5k | 65 (+46 测试) | ✅ 完成(buffer-fill 无分配 vs clay Allocator+std::function;MSVC 真反修饰;type_name 运行期补集) |
| math-01 | P3 | math | scalar 标量助手(div_ceil/round_up_to_multiple/sgn/approx_eq) | 00 | 0.5k | 28 (+51 测试) | ✅ 完成(clay consts/utils 458 行绝大多数♻️ C++20/23 `<bit>`/`<numbers>`/`<cmath>`;只自写 std 真缺的4个;div_ceil 改溢出安全) |
| bench-01 | P3 | bench | nanobench 设施 + SwissTable/StringBuf 对标 std | container-04/06 | — | bench设施 | ✅ 完成(SwissTable find2.7x/insert6.1x/erase4.1x>std;intern基准驱动优化0.94→1.31x;absl/F14对标→bench-02) |
| bench-02 | P3+ | bench | 对标 SOTA(absl/F14/moodycamel/oneTBB/highway/marl,不止 std) | 各模块 | — | — | 📋 规划(用户提:对比 project-hyle 高性能库;拆 02a hashmap/02b 队列/02c hash+wyhash/02d simd-sched) |
| simd-01 | P3 | simd | vec D-tag 抽象 + scalar 后端 | mem-01 | 2-3k | — | 占位 |
| simd-02 | P3 | simd | SSE4.2 + AVX2 后端 | simd-01 | 2-3k | — | 占位 |
| simd-03 | P3 | simd | AVX512 + NEON 后端 | simd-01 | 2-3k | — | 占位 |
| simd-04 | P3 | simd | 运行期变体 dispatch(探测器隔离+score) | simd-02 | 1-2k | — | 占位 |
| simd-05 | P3 | simd | dtype fp16/bf16(F16C/AVX512-FP16/BF16) | simd-02 | 2-3k | — | 占位 |
| simd-06 | P3 | simd | sleef 数学核 adapter | simd-04 | 1-2k | — | 占位 |
| sched-01 | P4 | sched | fcontext 6 平台 + 栈池(骨架:单 worker 单队列跑一 fiber) | container-01,mem-01,platform-01 | 2-3k | 226 (+737 vendored asm) | ✅ 完成(仅证切换,队列/worker 留 sched-02) |
| sched-02 | P4 | sched | per-core 有界 run-queue + LIFO slot | sched-01,container-01 | 2-3k | 205 | ✅ 完成(单worker驱动,steal留sched-03) |
| sched-03 | P4 | sched | work-stealing + spin-then-park + unpark 接力 | sched-02,container-08 | 2-3k | 340 | ✅ 完成(多worker偷取+park;§2.3②③防惊群优化诚实延迟) |
| sched-04 | P4 | sched | Waker completion 脊 + time 接入(deadline) | sched-02,time-02 | 1-2k | — | 占位 |
| sched-05 | P4 | sched | fiber-aware CV + mutex/waitgroup/channel | sched-04 | 2-3k | — | 占位 |
| sched-06 | P4 | sched | 协程胶水(co_await→fiber 让出) | sched-04 | 1-2k | — | 占位 |
| sched-07 | P4 | sched | 取消：CancelToken 树 + await 取消点 + 结构化并发 | sched-04,sched-06 | 2-3k | — | 占位 |
| sched-08 | P4 | sched | runtime 优雅关闭(两阶段 drain + 逆序清理) | sched-03 | 1-2k | — | 占位 |
| io-01 | P5 | io | proactor 抽象(Waker 接 sched) | sched-04 | 1-2k | — | 占位 |
| io-02 | P5 | io | io_uring 后端(借 liburing) | io-01 | 2-3k | — | 占位 |
| io-03 | P5 | io | epoll + kqueue + IOCP 后端 | io-01 | 3-4k | — | 占位 |
| io-04 | P5 | io | net(tcp/udp socket) | io-02 | 2-3k | — | 占位 |
| io-05 | P5 | io | file + 跨进程共享内存 | io-02 | 2-3k | — | 占位 |
| compute-01 | P5 | compute | device/stream 薄包 ggml-backend | sched-04 | 2-3k | — | 占位 |
| compute-02 | P5 | compute | MemorySpace 注册 + copy(同 coherent 塌缩) | compute-01,mem-05 | 2-3k | — | 占位 |
| compute-03 | P5 | compute | GPU completion→fiber(直连原生 handle) | compute-01,sched-04 | 1-2k | — | 占位 |
| compute-04 | P5 | compute | kernel 启动抽象 | compute-01 | 2-3k | — | 占位 |
| compute-05 | P5 | compute | ggml_bridge + 注入 HYLE 线程池 | compute-01 | 2-3k | — | 占位 |
| compute-06 | P5 | compute | Vulkan compute 自定义 kernel | compute-04 | 2-3k | — | 占位 |
| graph-01 | P6 | graph | taskgraph 剥特化 + 绑 sched | sched-03 | 3-4k | — | 占位 |
| close-01 | P6 | 收口 | C-ABI shim(对外纯 C 头) | 各模块稳定 | 2-3k | — | 占位 |
| close-02 | P6 | 收口 | serialize(simdjson/flatbuffers) + hash + T3 便利 | container | 4-6k | — | 占位 |
| close-03 | P6 | 收口 | FFI 绑定(TS N-API / Python ctypes) | close-01 | 2-3k | — | 占位 |

约 47 个里程碑。状态：✅ 完成 / 进行中 / 占位(JIT 补全)。真实LOC 闭环后回填，差距大见对应文档解释。

> 依赖列只标**直接前置**；完整依赖图见 `roadmap.md` 第一节。`00-scaffolding` 是所有里程碑的隐含前置（万物在它上面编译）。
