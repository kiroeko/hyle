# HYLE 参考库映射

> 参考项目均已 clone 到 `C:\public\project-hyle\`（与 hyle 并列）。本表把每个库映射到 HYLE 模块，标注**借**（vendored 依赖/adapter 对接）还是**参考**（读设计、不依赖），及开工优先级。
>
> 角色定义：**借** = 进 HYLE 构建（封 adapter 或 vendored）；**参考** = 读其设计/代码，自己实现，不引入依赖。

## 按模块映射

### simd/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **highway** | 借 | transcendental 数学核（log 等）+ dtype 对照；vendored 在 `third_party/`、无条件默认编进生产。其 scalable-friendly 接口 + HWY_DYNAMIC_DISPATCH 亦是 vec 抽象的范本 |
| **xsimd** | 借 | transcendental 数学核（exp/tanh），vendored 在 `third_party/`、无条件默认编进生产；逐 op best-of-breed 与 highway 互选最快核 |
| **simdutf** | 借 | utf8 validate/count SIMD 核，vendored 在 `third_party/`、无条件默认编进生产（自带内部 ISA dispatch）|

### compute/（Tier 1，含 Intel 计算加速）
| 库 | 角色 | 说明 |
|---|---|---|
| **ggml** | 借（主力）| 算子 + 多后端（cuda/hip/metal/vulkan/sycl/webgpu/opencl…），`ggml_bridge` adapter 对接上游 pin 版；同时参考 ggml-backend 接口、dev_caps、CPU 多变体 dispatch |
| **llama.cpp** | 参考 | 多变体 DLL+score dispatch、CMake ISA 处理、探测器隔离（-fno-LTO）、推理引擎工程范式 |
| **oneDNN** | **借（算子库）** | **Intel DL 原语 + AMX/VNNI 加速正主**；运行时 cpuid dispatch + Xbyak JIT。定位是**算子库（ggml 同级 CPU 算子供应商，接 bridge）**，非 device/stream 后端——它自带 JIT 不暴露 raw kernel launch |
| **oneMath** | **仅参考（不借）** | Intel 数学（前 oneMKL）：BLAS/FFT。**SYCL 重依赖**（引它=引整条 SYCL/DPC++ 工具链），HYLE 已有 ggml BLAS；仅参考其 domain×device×library 三级分发表 |
| **oneDAL** | **边界外**（深读校正）| 经典 ML（kmeans/PCA/knn/forest）是 L1/L2 应用层，非 infra 原语 |

### sched/（Tier 0）+ 并发
| 库 | 角色 | 说明 |
|---|---|---|
| **marl** | 参考（精读）| **最贴 HYLE 的 fiber + work-stealing 调度库**，第一参考 |
| **tokio** | 参考 | work-stealing/LIFO slot/unpark 接力/per-core 队列的设计源 |
| **boost**(context) | **借（仅 3 汇编）** | **fcontext** 是 fiber 切换原语来源；`fcontext_t`/`transfer_t` 纯 POD 零依赖，只取 `jump/make/ontop` 三例程 .S/.asm（**本地 context 是空 submodule，须从上游拉**）+ 自写 ~10 行头，不要 config/封装层。boost.intrusive 仅参考 |
| **seastar** | 参考 | per-core 局部性、reactor_backend completion 抽象;**不学** share-nothing 调度;**xcpu_freelist 评估后否决**(其 owner-local 非原子表依赖 share-nothing+线程钉死+池长于进程,HYLE 通用生命周期固定池给不出,且跨核 free 正确性已由地址路由原子 push 解决,见 mem-design) |
| **oneTBB** | **多为边界外**（深读校正）| task scheduler 边界外（自带 work-stealing 抢权，同拒 boost.fiber）；`concurrent_bounded_queue` **不借**（票号+分页+链 libtbb，非定长 ring，用 Vyukov/rigtorp）；仅 `scalable_allocator`/rml pool policy 设计**参考** mem/，不 vendor |
| **oneDPL** | **边界外**（深读校正）| 并行 STL 架在 TBB/OpenMP 之上——而那层正是 HYLE 自造调度器要替换的；算法层非 infra 原语 |

### container/（Tier 0）
| 库 | 角色 | 说明 |
|---|---|---|
| **abseil-cpp** | 参考 | SwissTable 设计（控制字节 + SIMD 探测），hashmap 头号参考 |
| **folly** | 参考 | F14 hashmap + 并发原语/SmallVector 等 |
| **moodycamel-ConcurrentQueue** | 参考 | lock-free MPMC 队列，sched 的 global 注入队列直接参考 |
| **boost**(intrusive) | 参考 | 侵入式 list/map（sched run-queue/waiter 链）|

### time/（Tier 0）
| 库 | 角色 | 说明 |
|---|---|---|
| **abseil-cpp** | 参考 | TSC 校准算法（`sysinfo.cc`：GetTimeTscPair 取最低延迟样本抗抢占 + MeasureTscFrequency 倍增窗口到 1% 收敛）；`fast_monotonic_ns` 的频率标定直接照搬其思路。另：abseil 默认关闭 cycle-clock 当 wall-time（唤醒可能错），正是 HYLE 不把 rdtsc 设默认钟的依据 |
| **tracy** | 参考 | invariant-TSC 探测（`CheckHardwareSupportsInvariantTSC`：CPUID leaf1 EDX bit4 + leaf 0x80000007 EDX bit8）+ rdtsc 仅作测量用途的范式 |

> 均**读设计自实现、不引依赖**（time/ 零 HYLE 依赖定位）。MSVC `__msvc_chrono.hpp` steady_clock 的常见频率特判（10/24MHz→乘法）亦是 `monotonic_ns` 补特判的参考。

### hash/（Tier 3）
| 库 | 角色 | 说明 |
|---|---|---|
| **xxHash** | **借（默认 hasher）** | **XXH3 设为 hashmap 默认 hasher + 内容寻址**（30+ GB/s、SMHasher 满分、自带 x86 dispatch、大小端一致）。**替换 clay 的 murmur**（murmur 仅 3.9 GB/s，留编译期 type-ID 用）|
| **wyhash** | **不借** | 作者自承非 64-bit 抗碰撞（~62bit）、长数据 PractRand 失败；无优势盖过 XXH3 |

### serialize/（Tier 3）
| 库 | 角色 | 说明 |
|---|---|---|
| **simdjson** | **借（vendoring）** | **JSON 读+写都归它**（two-stage SIMD 解析 + on-demand；写侧 `string_builder`/`to_json` + C++26 reflection——"只读"已过时）。x64 需 `-march` 满速 |
| **flatbuffers** | **借（vendoring）** | 权重/张量 mmap 直读；轻、footprint 小、贴 ggml/gguf 生态 |
| **capnproto** | **借核，RPC 边界外** | 跨进程消息/shared-mem 一等公民；只取序列化核（`capnp`），不取 `capnp-rpc`（L1）|

### obs/（Tier 2 hook / Tier 3 重机器）
| 库 | 角色 | 说明 |
|---|---|---|
| **tracy** | **参考设计，不整库引入** | obs/ hook 照搬三点：①宏埋点+编译期 no-op（零开销关闭，开 zone 仅 2.25ns）②TLS 无锁 SPSC 队列 ③scoped zone(RAII)。**支持 fiber 追踪**（`TRACY_FIBERS`，正中 HYLE）但**百万 fiber 不能 always-on**（≤65534 source location），只采样/调试子集。整库带 server/UI 是 L2 工具、非 infra |

### io/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **seastar**(reactor_backend) | 参考 | epoll/aio 也对上呈现 completion 语义的后端抽象范本 |
| liburing | 借 | io_uring syscall 薄封装（仅 Linux；非本目录，构建期拉）|

## 开工优先级（按 roadmap 阶段）

- **P3-P4 即用**：marl（sched 精读）、moodycamel（队列）、boost.context（fcontext）、abseil/folly（container）、highway/xsimd/simdutf（simd，vendored 数学核 + utf8）。
- **P5 即用**：ggml + llama.cpp（compute 主力借/参考）、oneDNN/oneMath（Intel 加速，CPU/数学后端）、seastar（io reactor_backend）。
- **P6 / 按需**：simdjson/flatbuffers/capnproto（serialize）、xxHash（hash，设默认 hasher）、tracy（obs，借设计）。oneDPL/oneTBB/oneDAL **边界外**（见下）。

## 借 vs 造的总原则（呼应 charter）

借**多年解决的通用难题**（highway/xsimd 数学核、ggml 算子、boost.context 汇编、liburing syscall）；造**贴 HYLE 需求或差异焦点**（container 全自造、sched 调度自造、simd 抽象自造）。所有借入第三方一律封 adapter / 不外露其类型（同 ggml/mimalloc 解耦原则）。

### 关键区分：加速库分两类，只借第一类（这是 oneTBB 为何边界外的原因）

| 类 | 是什么 | 例 | HYLE |
|---|---|---|---|
| **指令/kernel 级加速**（让单次计算更快）| 优化的算子/数学核/SIMD intrinsic，**不自带调度** | oneDNN(AMX/VNNI JIT)、oneMath(BLAS)、highway/xsimd 数学核、ggml 算子、XXH3 | ✅ **借**（喂它 HYLE 的线程，结果回灌）|
| **并行运行时级加速**（把任务铺到多核）| **自带 work-stealing 调度器 / task arena / 线程池** | **oneTBB**、oneDPL、OpenMP、asio io_context、boost.fiber | ❌ **不借**（这是 HYLE 自己的活）|

**oneTBB 确实是加速库——但它加速的是"运行时层"，恰是 HYLE 要独占的那一层。** 它的 `parallel_for` 离不开 task arena（自带 work-stealing 调度器）；引它 = HYLE fiber 调度器 + TBB 调度器**两套在同一批核上抢线程、过度订阅、cache 争抢**——正是 HYLE 存在要消灭的"双调度器阻抗失配"（同拒 boost.fiber/asio 的理由）。所以 oneTBB scheduler/queue、oneDPL **边界外**；只 oneTBB 的 allocator pool 设计作 mem/ 参考。

**那 CPU 数据并行（matmul 等铺满多核）靠谁？** 三层，都不需要 TBB：
1. **借来的算子自带并行 + 可注入 HYLE 调度器**：ggml CPU 后端有 `ggml_backend_cpu_set_threadpool`（`ggml-cpu.h:135`）——**HYLE 把自己的线程池喂给 ggml，而非让 ggml 自起线程**，避免双调度器。oneDNN 同理（内部 JIT，并行跑在 HYLE 给的线程上）。
2. **HYLE 自有 kernel 的数据并行**：work-stealing 调度器做 fork-join / parallel-for——**一个调度器同时干 IO 并发(fiber)和 CPU 数据并行**，这正是 work-stealing 的本职。
3. **指令级加速**：借 highway/xsimd 数学核、oneDNN 的 AMX kernel（第一类），不借 TBB。

### 隐藏运行时清单（引依赖前必查，防"悄悄混进第二个运行时"）

凡库**自带线程/事件循环/调度器/控制流接管**的，都是红线——只借它的"叶子计算"，不借它的运行时。检测法：源码搜 `pthread_create`/`std::thread`；看它有没有要你 `run()`/`loop()`/`executor`/`pool`；看它装不装 signal handler/timer（撞 HYLE time/）。

| 隐藏运行时来源 | 处置 |
|---|---|
| 并行框架（TBB/OpenMP/oneDPL）| 边界外；数据并行用 HYLE 调度器 |
| 异步/IO 运行时（asio/libuv/seastar 整框架）| 只借 syscall 层（liburing），事件循环自造 |
| 分配器后台线程（mimalloc/jemalloc 的 purge 线程）| 配置关掉或纳管，别让它自起线程 |
| 算子库内部线程（ggml/oneDNN）| **注入 HYLE 线程池**（`set_threadpool`），不让其自起 |
| 日志/profiler 异步刷盘（spdlog async / tracy server）| 借设计不借库，或用 HYLE 调度器驱动 |
| 协程库自带调度器 | 只用语言级 `co_await`，调度归 HYLE sched |

**铁律**：HYLE 必须独占所有 OS 线程与控制流——每个 OS 线程要么是 HYLE worker、要么归 HYLE blocking 池管。加任何依赖前先问：**它起线程吗？它想接管控制流吗？** 是 → 只借计算、运行时留在门外。
