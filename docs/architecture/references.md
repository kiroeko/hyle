# HYLE 参考库映射

> 参考项目均已 clone 到 `C:\public\project-hyle\`（与 hyle 并列）。本表把每个库映射到 HYLE 模块，标注**借**（vendored 依赖/adapter 对接）还是**参考**（读设计、不依赖），及开工优先级。
>
> 角色定义：**借** = 进 HYLE 构建（封 adapter 或 vendored）；**参考** = 读其设计/代码，自己实现，不引入依赖。

## 严格收口对照矩阵

每轮模块复审先从真实场景出发，再去本地 clone 查成熟形状、算法和负空间。下表是最低
对照集，不表示要复制其全部功能，也不允许把参考库的 benchmark API 搬进 HYLE 产品面。

| HYLE 模块 | 首要本地参考 | 本轮主要核对什么 |
|---|---|---|
| core | C/C++ 标准库、Boost.Intrusive、folly | 版本/宏最小面、intrusive 生命周期与零额外所有权 |
| error | `std::expected`、Abseil Status、folly Expected | 值大小、context、传播、OS/GPU 叶错误与 C owning handle |
| platform | folly portability、libuv、Abseil、llama.cpp | thread/affinity、wait、VM/fs/dynlib、CPU dispatch 与平台忠实语义 |
| mem | mimalloc、oneTBB allocator/pool、folly、seastar | 通用/专用分配边界、arena/pool/cache/NUMA/huge page、复杂度与跨线程 free |
| time | Abseil、folly、libevent、seastar | clock 语义、deadline/sleep、arbitrary/common timer queue 与取消复杂度 |
| util | folly、Abseil、Boost | scope/ref/random/registry/spin 的模块归属、生命周期、并发和 header seam |
| obs | Tracy、folly | OFF 零成本、单 backend client、结构化 log、sink 摘除与 fiber trace |
| math | clay math、C++23 `<bit>/<cmath>/<numbers>`、Eigen、HYLE simd | 把迁移残余逐项分给 std/scalar/SIMD/消费域，先裁决模块边界，再谈 API/C ABI |
| container | Abseil、folly、oneTBB、moodycamel、rigtorp | SwissTable/SmallVector/string、并发/侵入式容器、allocator 与迭代失效 |
| sched | marl、folly fibers、tokio、seastar、oneTBB/taskflow | fiber/work-stealing、同步/取消/关闭、blocking 与 oversubscription |
| io | liburing、tokio-uring、seastar、Asio、libuv、libevent | proactor 生命周期、四后端语义、deadline/cancel、文件/网络与 backpressure |
| simd | Highway、xsimd、simdutf、SLEEF、oneDNN | 用户 batch/kernel 面、dispatch/type/dtype/math/sort/search、精度与原生复杂度 |
| compute | ggml、llama.cpp、oneDNN、Kompute、vitamin-cuda | device/stream/buffer/completion、CPU/GPU 算子、Vulkan/CUDA 和隐藏运行时 |
| graph | oneTBB、taskflow、Eigen、OpenVDB | DAG 增量重算、fork-join/scan/sort、确定性与 CAD/CAE 真消费者 |

对照结论必须落在对应 architecture/guide/performance；这里仅维护“应该读谁、读什么”。

## 按模块映射

### simd/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **highway** | 参考 + bench 对照 | D-tag/scalable/dynamic-dispatch 架构范本与 SOTA oracle；simd-20 后不进产品构建，外部 pin checkout 仅用于 benchmark |
| **xsimd** | 参考 + bench 对照 | 数学 best-of 与精度/性能 oracle；simd-20 后不进产品构建 |
| **simdutf** | 可选薄借 | 仅 `HYLE_SIMD_SIMDUTF` 门控 utf8 validate/count；未定义时不 include，不属于 Runtime/native 核心面 |
| **sleef** | **已退役(clone 留存)** | 曾评估作 transcendental 核,选定 highway+xsimd 后退役;仍 clone 供对照,不进构建 |
| **nanobench** | 借(测试期)| 微基准框架,vendored 在 `third_party/`;`scripts/bench-stat.py` 在其上做 n=15 统计判定(gap>spread 才判赢) |

### compute/（Tier 1，含 Intel 计算加速）
| 库 | 角色 | 说明 |
|---|---|---|
| **ggml** | 借（主力）| 算子 + 多后端（cuda/hip/metal/vulkan/sycl/webgpu/opencl…），`ggml_bridge` adapter 对接上游 pin 版；同时参考 ggml-backend 接口、dev_caps、CPU 多变体 dispatch |
| **kompute** | 参考（compute-06 Vulkan 后端骨架）| Vulkan compute 封装，**纯 Vulkan 零隐藏运行时**（无 std::thread/事件循环，唯一同步=可选 submit mutex + 每 Sequence 一个 vk::Fence），合"不引入第二调度器"铁律。抄其 Manager 初始化链 / Algorithm 的 SPIR-V→pipeline→descriptor / Tensor 的 memory-type 选择+staging 双缓冲+barrier 编排；**丢**阻塞 `waitForFences`（换非阻塞 `getFenceStatus` 轮询接 fiber，= Event::query 语义，fence=Vulkan Event 的 native handle）、shared_ptr 所有权、logger、OpBase 虚基类。详见 compute-design「compute-06 kompute 映射」节 |
| **vitamin-cuda** | **抄（compute-04/06 自造 kernel）** | 教学级 CUDA kernel 库（`C:\public\project-hyle\vitamin-cuda`），SGEMM/HGEMM/flash_attn/softmax/rope/reduce 等，含中英"beating cuBLAS"逐步优化 writeup（`posts/`）。**正是 ggml 显存边界之外 HYLE 自造 kernel 的领域**（compute-05 定案：借 ggml 用 ggml 显存；自造 kernel 才在 HYLE Buffer 上真零拷贝）——② CAD/CAE 数值核的 SOTA 抄写源。SGEMM 优化阶梯：naive → block tiling+共享内存 → a 转置写 smem+float4 → swizzle 消 bank conflict → C 写回事务合并 → double buffer 流水线（超 cuBLAS）。抄其 tiling/向量化/swizzle/双缓冲手法进 compute-04 NVRTC kernel |
| **llama.cpp** | 参考 | 多变体 DLL+score dispatch、CMake ISA 处理、探测器隔离（-fno-LTO）、推理引擎工程范式；compute-05 显存归属边界（host 归调用方/device 归 ggml gallocr）的权威范本 |
| **oneDNN** | **借（算子库）** | **Intel DL 原语 + AMX/VNNI 加速正主**；运行时 cpuid dispatch + Xbyak JIT。定位是**算子库（ggml 同级 CPU 算子供应商，接 bridge）**，非 device/stream 后端——它自带 JIT 不暴露 raw kernel launch |
| **oneMath** | **仅参考（不借）** | Intel 数学（前 oneMKL）：BLAS/FFT。**SYCL 重依赖**（引它=引整条 SYCL/DPC++ 工具链），HYLE 已有 ggml BLAS；仅参考其 domain×device×library 三级分发表 |
| **oneDAL** | **边界外**（深读校正）| 经典 ML（kmeans/PCA/knn/forest）是 L1/L2 应用层，非 infra 原语 |

### sched/（Tier 0）+ 并发
| 库 | 角色 | 说明 |
|---|---|---|
| **marl** | 参考（精读）| **最贴 HYLE 的 fiber + work-stealing 调度库**，第一参考 |
| **tokio** | 参考 | work-stealing/LIFO slot/unpark 接力/per-core 队列的设计源;io-04 批量 submit 的 tick/park flush 节奏直接照搬 `tokio-uring` 前身 tokio runtime |
| **tokio-uring** | 参考 | io_uring 运行时:submit/flush 模型(on_thread_park flush + SQ 满触发)、recv_from 的 `Box<msghdr>` 生命周期契约(io-04 udp 照搬)。读本地源码挖出 io P0 hang 真病根 |
| **crossbeam-deque** | 参考(bench 对照)| Chase-Lev work-stealing deque(Rust);container-07/queue bench-02b 的 work-stealing 同族 SOTA 对照(独立 cargo crate,gated 编译) |
| **boost**(context) | **借（仅 3 汇编）** | **fcontext** 是 fiber 切换原语来源；`fcontext_t`/`transfer_t` 纯 POD 零依赖，只取 `jump/make/ontop` 三例程 .S/.asm（**本地 context 是空 submodule，须从上游拉**）+ 自写 ~10 行头，不要 config/封装层。boost.intrusive 仅参考 |
| **seastar** | 参考 | per-core 局部性、reactor_backend completion 抽象;**不学** share-nothing 调度;**xcpu_freelist 评估后否决**(其 owner-local 非原子表依赖 share-nothing+线程钉死+池长于进程,HYLE 通用生命周期固定池给不出,且跨核 free 正确性已由地址路由原子 push 解决,见 mem-design) |
| **oneTBB** | **多为边界外**（深读校正）| task scheduler 边界外（自带 work-stealing 抢权，同拒 boost.fiber）；`concurrent_bounded_queue` **不借**（票号+分页+链 libtbb，非定长 ring，用 Vyukov/rigtorp）；仅 `scalable_allocator`/rml pool policy 设计**参考** mem/，不 vendor |
| **oneDPL** | **边界外**（深读校正）| 并行 STL 架在 TBB/OpenMP 之上——而那层正是 HYLE 自造调度器要替换的；算法层非 infra 原语 |

### container/（Tier 0）
| 库 | 角色 | 说明 |
|---|---|---|
| **abseil-cpp** | 参考 | SwissTable 设计（控制字节 + SIMD 探测），hashmap 头号参考 |
| **folly** | 参考 | F14 hashmap + 并发原语/SmallVector 等 |
| **moodycamel-ConcurrentQueue** | 参考(bench 对照)| lock-free MPMC 队列，sched global 注入队列参考;queue bench-02b 的 SOTA 对标基线 |
| **rigtorp-MPMCQueue** / **MPMCQueue** | 参考(bench 对照)| 定长环形 MPMC(Vyukov 序号法);BoundedQueue 设计参考 + queue bench-02b 对标 |
| **boost**(intrusive) | 参考 | 侵入式 list/map（sched run-queue/waiter 链）|

### time/（Tier 2 必要基质）
| 库 | 角色 | 说明 |
|---|---|---|
| **abseil-cpp** | 参考 | TSC 校准算法（`sysinfo.cc`：GetTimeTscPair 取最低延迟样本抗抢占 + `nanosleep` 后按实际 elapsed 测频 + 倍增窗口到 1% 收敛）；ordered measurement 使用其 LFENCE/RDTSC/RDTSCP 序列。另：abseil 默认关闭 cycle-clock 当 wall-time，正是 HYLE 只把硬件 counter 暴露为 profiling clock 的依据 |
| **folly** | 参考 | `hardware_timestamp` 的 x86 TSC / ARM64 CNTVCT 平台形状，以及 raw sampling 与 ordered measurement start/stop 的语义分离 |
| **tracy** | 参考 | invariant-TSC 探测（`CheckHardwareSupportsInvariantTSC`：CPUID leaf1 EDX bit4 + leaf 0x80000007 EDX bit8）+ rdtsc 仅作测量用途的范式 |

> 对参考库均**读设计自实现、不引入外部依赖**。time native 核只借 core 头；整个 `hyle_time` target 因 C adapter 的统一 `hyle_status` 合同依赖 error，但不链接 platform。MSVC `__msvc_chrono.hpp` steady_clock 的常见频率特判（10/24MHz→乘法）亦是 `monotonic_ns` 补特判的参考。

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
| **tracy** | **后端 + 参考设计** | native hook 的编译期 no-op、scoped zone、plot/counter/fiber/frame/message/memory/thread-name 由 Tracy 承载；HYLE 只保留自己的 seam，并保证 shared build 全进程一个 client。server/UI 是外部 profiler 工具，不复制进 obs 产品 API |
| **abseil-cpp** | **参考 + benchmark 机制项** | 同步 `LogSink`、全局注册/摘除、flush、sink 内日志不递归回注册图、process-lifetime registry 是 log runtime 的直接形状参考；其完整 LogEntry 带更多 metadata，点值只能标载荷差异，不能冒充同合同胜负 |
| **folly** | **参考** | handler 同步线程语义、flush、category/handler 分层及默认转义不可信控制字符。HYLE 借安全边界与 adapter 形状；hierarchical category、async writer/file/upload 属于应用日志系统，不塞进低层 producer seam |

### io/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **seastar**(reactor_backend) | 参考 | epoll/aio 也对上呈现 completion 语义的后端抽象范本;HTTP 栈是"自有运行时→自写 HTTP(Ragel parser)"的直接先例(io HTTP 定案参考) |
| **liburing** | **参考(不链)** | io_uring 机制权威:barrier/SQE 布局/多 op(multishot accept/recv、provided buf_ring、ASYNC_CANCEL)。**HYLE 用裸 syscall 自写,不链 liburing**——精确复刻其 v2.15 fast-path barrier + buf_ring 协议(io-01/02/07 已落地) |
| **tokio-uring** | 参考 | 见 sched/ 段:submit/flush 节奏 + msghdr 生命周期(io-04 udp) |
| **libevent** | 参考 | reactor 级:common_timeout 有序链算法(time CommonTimeoutQueue 同款基线)、evdns 异步 DNS(io-09 待)、bufferevent 水位(应用层不做);对 HYLE proactor 价值小于以上 |
| **nghttp2** | **借核(io-11 HTTP/2，已落地)** | 固定 v1.69.0 commit；借 HTTP/2 frame/HPACK/stream/flow-control core，自写 allocator、caps、over-co_await transport 与 C codec，不取其 event loop/应用层 |

### graph/（Tier 6，北极星② 编排 + 数据并行）
| 库 | 角色 | 说明 |
|---|---|---|
| **oneTBB** | **参考（设计,不借运行时）** | graph-02 数据并行的 SOTA 设计源:`blocked_range`(grainsize + 递归 split)、`parallel_reduce`(reduction_tree_node + **deterministic** 双档)、`parallel_scan`(两趟)。**不借其 task arena/调度器**(双调度器红线);跑 HYLE sched。见 graph-02-design |
| **taskflow** | 参考 | 现代 C++ 的 DAG + 数据并行一体框架(`tf::Taskflow` 依赖图 + for_each/reduce/scan + subflow 动态图)——**graph-01(DAG 编排)+ graph-02(数据并行)合起来的最贴 SOTA 对照**,纯 header 比 oneTBB 更贴 HYLE 编排语义。对拍其 API 形态;调度仍归 HYLE sched |
| **eigen** | **参考（CAE 求解器,按需）** | 北极星② CAE 的稠密/稀疏线代 SOTA(matmul/分解/迭代求解器 CG/BiCGSTAB);motivation 里"CAE 把 compute/simd 压到稀疏线代深度"的头号参考。**并行可注入外部池**:`Parallelizer.h` 除 OpenMP 外有 `EIGEN_GEMM_THREADPOOL`(`pool->NumThreads()`,src:38/81)——同 ggml `set_threadpool` 模式,能喂 HYLE 调度器而非用其 OpenMP(避双运行时)。**等 graph-02 之后真做 CAE 求解器消费者时读**,现在超前 |
| **openvdb** | **参考（CAD/CAE 网格/体数据,按需）** | 工业级稀疏体素/网格数据结构 + 其上的并行遍历(自带 TBB 并行,只读其数据结构 + 分块遍历设计,并行归 HYLE graph-02)。北极星② 几何核/网格 stencil 的参考。**等真做几何/体数据消费者再读**,现在超前 |

## 开工优先级（按 roadmap 阶段）

- **P3-P4 即用**：marl（sched 精读）、moodycamel（队列）、boost.context（fcontext）、abseil/folly（container）、highway/xsimd（SIMD 架构与外部 benchmark 对照）、simdutf（可选 UTF-8 薄借）。
- **P5 即用**：ggml + llama.cpp（compute 主力借/参考）、oneDNN/oneMath（Intel 加速，CPU/数学后端）、seastar（io reactor_backend）。
- **P6 / 按需**：simdjson/flatbuffers/capnproto（serialize）、xxHash（hash，设默认 hasher）、tracy（obs，借设计）、**oneTBB/taskflow（graph-02 数据并行设计对照）、nghttp2（io-11 HTTP/2 借核）、eigen/openvdb（北极星② CAE/几何,更后置按需）**。oneDPL/oneDAL **边界外**（见下）。

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
