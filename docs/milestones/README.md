# HYLE 里程碑索引

> 实现的真正"步骤"在这里——约 47 个里程碑，每个 1-4k 行、可独立编/测/基准。`roadmap.md` 是依赖层(阶段)的拓扑序与决策；本目录是**可执行的逐步计划**，每步一文档。
>
> **粒度原则**：一个里程碑 = 一次"写完→编过→单测(覆盖达 Tier 标准)→基准→提交"的闭环。
>
> **生长方式（JIT）**：最先执行的 `00-scaffolding` 与 walking-skeleton 几步已写完整；其余**临到该模块开工前再补全细节**——后期里程碑依赖前面实现学到的东西，提前写=空想。索引行先占位（ID/依赖/目的/LOC），细节随推进生长。

## 每个里程碑文档的模板

见 [`_TEMPLATE.md`](./_TEMPLATE.md)。固定章节：**前置依赖 / 目的 / 在架构中的位置 / 具体事务 / 交付物 / 验收标准(含覆盖 Tier) / 参考**。

## Walking Skeleton（最先做，纵切验脊）

打通"**一个 fiber 跑起→让出→被唤醒**"。= `00` + 各模块**第一个最小里程碑**串起来，**先不做 work-stealing/多后端**。验掉最高风险的 fcontext+调度脊，之后逐里程碑加厚。

骨架路径：`00-scaffolding` → `error-01` → `platform-01` → `mem-01` → `container-01` → `sched-01`(单 worker 单队列)。

## 预测 vs 真实 LOC（防玩具化）

> **真实LOC = 该里程碑自写代码的 cloc 实测（生产+测试，不含文档、不含 vendored 第三方）。**每步闭环后回填。
> **预测/真实差距是"防玩具"信号**：差距大必须在该里程碑文档解释清楚——是"骨架版刻意精瘦、重活在后续里程碑"（合理），还是"偷懒 stub/空实现"（不可接受）。
> 已完成几步真实普遍 < 预测，因走的是 **walking skeleton 路径**（每步只做最小纵切，模块重头留加厚里程碑）；非玩具的证据是每步的真测 + TSan + 对抗式自审，详见各文档。

## 全部里程碑（阶段 × 模块）

| ID | 阶段 | 模块 | 目的(一行) | 前置 | 预测LOC | 真实LOC | 状态 |
|---|---|---|---|---|---|---|---|
| **00-scaffolding** | P-1 | 构建 | CMake 多模块结构+GoogleTest/nanobench/sanitizer/CI+HYLE_宏骨架 | — | 1-2k | 294 | ✅ 完成 |
| error-01 | P0 | error | Status + std::expected + ASSIGN_OR_RAISE 宏 | 00 | 1-2k | 456 | ✅ 完成 |
| obs-01 | P0 | obs | instrumentation hook API(编译期 no-op 空实现) | 00 | 1-2k | — | 占位 |
| platform-01 | P1 | platform | thread + park(futex/WaitOnAddress/ulock) | error-01 | 2-3k | 527 | ✅ 完成 |
| platform-02 | P1 | platform | virtual_memory + dynlib | platform-01 | 2-3k | — | 占位 |
| platform-03 | P1 | platform | path + fs + env + os_info | platform-01 | 1-2k | — | 占位 |
| mem-01 | P2 | mem | Allocator vtable + 修 alignment（骨架版含最小 mimalloc 包装）| platform-01 | 1-2k | 182 | ✅ 完成 |
| mem-02 | P2 | mem | mimalloc backing 完整接入(封 vtable 后) | mem-01 | 1-2k | — | 占位 |
| mem-03 | P2 | mem | MemoryArena 无锁块池 + 修 64 槽上限 | mem-02 | 2-3k | — | 占位 |
| mem-04 | P2 | mem | ObjectPool + StackAllocator 推倒重写(修 UB) | mem-02 | 2-3k | — | 占位 |
| mem-05 | P2 | mem | MemorySpace 组合抽象 + xcpu_freelist | mem-03 | 2-3k | — | 占位 |
| mem-06 | P2 | mem | NUMA + 大页 + stats(修下溢) | mem-05 | 2-3k | — | 占位 |
| time-01 | P2 | time | 单调钟(CLOCK_MONOTONIC/QPC) | platform-01 | 1k | — | 占位 |
| time-02 | P2 | time | 分层时间轮 | time-01 | 1-1.5k | — | 占位 |
| container-01 | P3 | container | vendor 有界 ring(Vyukov/rigtorp,骨架用) | mem-01 | 1-2k | 220 | ✅ 完成 |
| container-02 | P3 | container | XXH3 接入(默认 hasher) | 00 | 1k | — | 占位 |
| container-03 | P3 | container | carray header-before-data + RAII 壳 | mem-02 | 1-2k | — | 占位 |
| container-04 | P3 | container | SwissTable 探测层重写(concept 模板)【大】 | container-02 | 3-4k | — | 占位 |
| container-05 | P3 | container | SmallVector + intrusive list/slist | mem-02 | 1-2k | — | 占位 |
| container-06 | P3 | container | string_ref + string_buf 驻留池 | container-02 | 1-2k | — | 占位 |
| container-07 | P3 | container | 有界 Chase-Lev work-stealing deque(owner LIFO/DFS,供 graph fork-join) | mem-01 | 1-2k | 249 | ✅ 完成(用途:graph-01,非调度器本地队列) |
| container-08 | P3 | container | 有界 work-stealing FIFO ring(tokio queue.rs:owner push免同步/pop+thief CAS packed-head/偷半) | mem-01 | 1-2k | 279 | ✅ 完成(sched-03 真前置) |
| simd-01 | P3 | simd | vec D-tag 抽象 + scalar 后端 | mem-01 | 2-3k | — | 占位 |
| simd-02 | P3 | simd | SSE4.2 + AVX2 后端 | simd-01 | 2-3k | — | 占位 |
| simd-03 | P3 | simd | AVX512 + NEON 后端 | simd-01 | 2-3k | — | 占位 |
| simd-04 | P3 | simd | 运行期变体 dispatch(探测器隔离+score) | simd-02 | 1-2k | — | 占位 |
| simd-05 | P3 | simd | dtype fp16/bf16(F16C/AVX512-FP16/BF16) | simd-02 | 2-3k | — | 占位 |
| simd-06 | P3 | simd | sleef 数学核 adapter | simd-04 | 1-2k | — | 占位 |
| sched-01 | P4 | sched | fcontext 6 平台 + 栈池(骨架:单 worker 单队列跑一 fiber) | container-01,mem-01,platform-01 | 2-3k | 226 (+737 vendored asm) | ✅ 完成(仅证切换,队列/worker 留 sched-02) |
| sched-02 | P4 | sched | per-core 有界 run-queue + LIFO slot | sched-01,container-01 | 2-3k | 205 | ✅ 完成(单worker驱动,steal留sched-03) |
| sched-03 | P4 | sched | work-stealing + spin-then-park + unpark 接力 | sched-02,container-08 | 2-3k | — | 占位 |
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
