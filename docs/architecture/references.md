# HYLE 参考库映射

> 参考项目均已 clone 到 `C:\public\project-hyle\`（与 hyle 并列）。本表把每个库映射到 HYLE 模块，标注**借**（vendored 依赖/adapter 对接）、**参考**（读设计、不依赖）或 **ROUTED**（消费方直接采用成熟路线），及开工优先级。
>
> 角色定义：**借** = 进 HYLE 构建（封 adapter 或 vendored）；**参考** = 读其设计/代码，自己实现，不引入依赖；**ROUTED** = 不进 HYLE 构建，由消费方拥有依赖和领域合同。

## 严格收口对照矩阵

每轮模块复审先从真实场景出发，再去本地 clone 查成熟形状、算法和负空间。下表是最低
对照集，不表示要复制其全部功能，也不允许把参考库的 benchmark API 搬进 HYLE 产品面。

| HYLE 模块 | 首要本地参考 | 本轮主要核对什么 |
|---|---|---|
| core | C++ 标准库、oneTBB、Abseil、Highway、folly | compiler/invariant、cache layout/prefetch 的最小可移植面；build-owned 编译期版本边界 |
| container_intrusive | Boost.Intrusive、folly、oneTBB | caller-owned node、auto/safe hook、counted/slist/hlist、sequence algorithms 与零分配形状 |
| error | `std::expected`、Abseil Status、folly Expected | 值大小、context、传播与 OS/GPU 叶错误 |
| platform | folly portability、libuv、Abseil、llama.cpp | thread/affinity、wait、VM/fs/dynlib、CPU dispatch 与平台忠实语义 |
| mem | mimalloc、oneTBB allocator/pool、folly、seastar | 通用/专用分配边界、arena/pool/cache/NUMA/huge page、复杂度与跨线程 free |
| time | Abseil、folly、libevent、seastar | clock 语义、deadline/sleep、arbitrary/common timer queue 与取消复杂度 |
| util | folly、Abseil、Boost | scope/ref/random/registry/spin 的模块归属、生命周期、并发和 header seam |
| obs | Tracy、folly | OFF 零成本、单 backend client、结构化 log、sink 摘除与 fiber trace |
| math | clay math、C++23 `<bit>/<cmath>/<numbers>`、Eigen、HYLE simd | 把迁移残余逐项分给 std/scalar/SIMD/消费域，先裁决模块边界，再谈 C++ API |
| container | Abseil、folly、oneTBB、moodycamel、rigtorp | SwissTable/SmallVector/string、并发/侵入式容器、allocator 与迭代失效 |
| sched | marl、folly fibers、tokio、seastar、oneTBB/taskflow | fiber/work-stealing、同步/取消/关闭、blocking 与 oversubscription |
| io | liburing、tokio-uring、seastar、Asio、libuv、libevent | proactor 生命周期、四后端语义、deadline/cancel、文件/网络与 backpressure |
| simd | Highway、xsimd、simdutf、SLEEF、oneDNN | 用户 batch/kernel 面、dispatch/type/dtype/math/sort/search、精度与原生复杂度 |
| compute | ggml、llama.cpp、ds4、vLLM、SGLang、oneDNN、Kompute、vitamin-cuda | 真实推理 shape、device/stream/buffer/completion、CPU/GPU 算子、Vulkan/CUDA 和隐藏运行时 |
| graph | oneTBB、taskflow、Eigen、OpenVDB | DAG 增量重算、fork-join/scan/sort、确定性与 CAD/CAE 真消费者 |

对照结论必须落在对应 [architecture](./)、[用户指南](../user/) 或
[实现证据](../implementation/evidence/)；这里仅维护“应该读谁、读什么”。参考库的名声、代码量
或单个 benchmark 结果都不能替代 HYLE 自己的同合同证据。

Native FFI 当前为 `N/A`，不存在 C 或 managed 用户面；旧 C/managed fixture 已从当前产品树移除，
其名称和形状不构成兼容合同。未来只有满足 [Native FFI 准入策略](native-ffi-policy.md)后，才为
已命名消费者选择并验证相应运行时参考。

### 跨模块消费者纵切

| 纵切 | 主要来源 | 用来发现什么 |
|---|---|---|
| 量化推理与 serving | ggml/llama.cpp、ds4、vLLM、SGLang | 模型映射、量化 kernel、KV/scratch、调度、流式服务与关闭 |
| 异构渲染与资源流送 | Cycles 源码快照、OpenVDB/NanoVDB、Kompute | host/device ownership、opaque range、staging/copy/completion、资源 identity、热重载与延迟销毁 |
| 批处理数值/数据管线 | oneDNN、OpenVDB、oneTBB 的算法形状 | 大文件、显式 scratch、并行分块、向量化变换、持久化与失败恢复 |
| 低延迟异步服务 | Seastar、libuv、nginx、vLLM/SGLang serving | listener/session、deadline/cancel/backpressure、文件 watch、reload、drain 与指标导出 |

这些来源跨越多个 HYLE owner，用于发现完整旅程中的胶水缺口；它们不是把整套引擎或 framework
引入产品构建的依赖清单。无 Git identity 的 Cycles 本地快照只能作辅助证据，正式裁决还须绑定可复核
版本或第二个独立来源。

## 按模块映射

### simd/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **highway** | 借（锁定产品源码）+ 参考 + bench 对照 | 当前直接承载 SIMD 数学实现，后续 foundation/dispatch 仍按 capability 裁决；HYLE façade 不泄漏 vendor 类型。独立 peer checkout 只用于同身份 benchmark，不能与产品 source override 混用 |
| **xsimd** | 参考 + bench 对照 | 当前只作数学精度/性能 oracle，不进产品构建；只有 Highway 无法满足的具名 capability/accuracy gap 才重新准入 |
| **simdutf** | 借（container_text 直接编入） | vendored v9.1 amalgamation 无条件编入 `hyle_container_text`；`utf8_validate` 借其单遍 validate，`utf8_length` 借 validate/count，二者复用内部 runtime ISA dispatch，公开头不泄露 vendor 类型 |
| **sleef** | **已退役** | 曾评估作 transcendental 核；当前不进产品或基准默认闭包，只有新的具名缺口才重新准入 |
| **nanobench** | 借（测试期） | 微基准框架，vendored 在 `third_party/`；统一性能仪器在其上做跨进程同进程配对：预声明 margin、分布无关中位数 CI、comparability/qualification/exact-performance 分轴、同合同 non-inferiority gate、逐进程 raw JSON 与正反序/shape 合成；sign p 值只作诊断 |

### compute/（Tier 1，含 Intel 计算加速）
| 库 | 角色 | 说明 |
|---|---|---|
| **ggml** | 借（主力）| 当前中央 lock 固定 `v0.21.0`；HYLE 实际只构建私有 PIC 静态 CPU + 可选 CUDA 闭包，全部类型封在 `ggml_bridge`。其自有 threadpool 不可替换成 HYLE scheduler；按合批性选择 ggml 自有并行或 `n_threads=1`，阻塞提交走 HYLE blocking pool |
| **kompute** | 参考（compute-06 Vulkan 后端骨架）| Vulkan compute 封装，**纯 Vulkan 零隐藏运行时**（无 std::thread/事件循环，唯一同步=可选 submit mutex + 每 Sequence 一个 vk::Fence），合"不引入第二调度器"铁律。抄其 Manager 初始化链 / Algorithm 的 SPIR-V→pipeline→descriptor / Tensor 的 memory-type 选择+staging 双缓冲+barrier 编排；**丢**阻塞 `waitForFences`（换非阻塞 `getFenceStatus` 轮询接 fiber，= Event::query 语义，fence=Vulkan Event 的 native handle）、shared_ptr 所有权、logger、OpBase 虚基类。详见 compute-design「compute-06 kompute 映射」节 |
| **vitamin-cuda** | **参考（compute 自造 kernel）** | 教学级 CUDA kernel 库（`C:\public\project-hyle\vitamin-cuda`），SGEMM/HGEMM/flash_attn/softmax/rope/reduce 等，含中英“beating cuBLAS”逐步优化 writeup（`posts/`）。它是 ggml 显存边界之外 HYLE 自造 kernel 的优化参考，不是 HYLE kernel 的性能证明。SGEMM 展示 naive → block tiling+共享内存 → 转置写 smem+float4 → swizzle → 合并写回 → double buffer 的阶梯；采用前仍须在 HYLE 的真实合同上复证 |
| **llama.cpp** | 参考 | 多变体 DLL+score dispatch、CMake ISA 处理、探测器隔离（-fno-LTO）、推理引擎工程范式；compute-05 显存归属边界（host 归调用方/device 归 ggml gallocr）的权威范本 |
| **ds4** | 真实消费参考 | 紧凑 CPU/统一内存量化推理路径；用于发现行算子、内存、top-k 和数据布局需求，不自动作性能 peer |
| **vLLM** | 真实消费参考 | runtime、CPU offload/custom-op、paged KV、sampling 与 serving 需求；GPU/Triton/tensor fusion 形状必须归到正确 owner |
| **SGLang** | 真实消费参考 | serving/kernel registry、attention、MoE、sampling、speculative decoding 需求；用于发现语义族，不机械复制 Python/GPU API |
| **OpenAI Jalapeño / InferenceX 结果** | 行业方向证据，不是 HYLE peer | inference-only 全栈 co-design 对 prefill/decode、KV locality、memory/network placement 与 latency-throughput-power Pareto 的公开案例；只使用其公开 workload/口径反查需求，不把厂商自测或封闭硬件当 HYLE SOTA 证明。见 [`motivation`](motivation.md#2026-08-的外部事实推理正在成为全栈系统问题) |
| **Apple M5 Ultra** | 可获得统一内存平台方向，不是仅凭规格成立的性能 peer | 最高 512 GB unified memory、1.2 TB/s memory bandwidth 与 quad-die interconnect 使超出普通离散显存的本地量化模型成为正式 workload；HYLE 仍须实测 residency、跨 die locality、Metal/kernel 与端到端 token 指标。见 [`motivation`](motivation.md#2026-08-的外部事实推理正在成为全栈系统问题) |
| **oneDNN** | **借（算子库）** | **Intel DL 原语 + AMX/VNNI 加速正主**；运行时 cpuid dispatch + Xbyak JIT。定位是**算子库（ggml 同级 CPU 算子供应商，接 bridge）**，非 device/stream 后端——它自带 JIT 不暴露 raw kernel launch |
| **oneMath** | **仅参考（不借）** | Intel 数学（前 oneMKL）：BLAS/FFT。**SYCL 重依赖**（引它=引整条 SYCL/DPC++ 工具链），HYLE 已有 ggml BLAS；仅参考其 domain×device×library 三级分发表 |
| **oneDAL** | **边界外**（深读校正）| 经典 ML（kmeans/PCA/knn/forest）是 L1/L2 应用层，非 infra 原语 |

### sched/（Tier 0）+ 并发
| 库 | 角色 | 说明 |
|---|---|---|
| **marl** | 参考（精读）| **最贴 HYLE 的 fiber + work-stealing 调度库**，第一参考 |
| **tokio** | 参考 | work-stealing/LIFO slot/unpark 接力/per-core 队列的设计源;io-04 批量 submit 的 tick/park flush 节奏直接照搬 `tokio-uring` 前身 tokio runtime |
| **tokio-uring** | 参考 | io_uring 运行时:submit/flush 模型(on_thread_park flush + SQ 满触发)、recv_from 的 `Box<msghdr>` 生命周期契约(io-04 udp 照搬)。读本地源码挖出 io P0 hang 真病根 |
| **crossbeam-deque** | 参考(bench 对照)| Chase-Lev work-stealing deque(Rust)；work-stealing 同族的成熟对照（独立 cargo crate，gated 编译） |
| **boost**(context) | **借（仅 3 汇编）** | **fcontext** 是 fiber 切换原语来源；`fcontext_t`/`transfer_t` 纯 POD 零依赖，只取 `jump/make/ontop` 三例程 .S/.asm（**本地 context 是空 submodule，须从上游拉**）+ 自写 ~10 行头，不要 config/封装层。boost.intrusive 仅参考 |
| **seastar** | 参考 | per-core 局部性、reactor_backend completion 抽象;**不学** share-nothing 调度;**xcpu_freelist 评估后否决**(其 owner-local 非原子表依赖 share-nothing+线程钉死+池长于进程,HYLE 通用生命周期固定池给不出,且跨核 free 正确性已由地址路由原子 push 解决,见 mem-design) |
| **oneTBB** | **多为边界外**（深读校正）| task scheduler 边界外（自带 work-stealing 抢权，同拒 boost.fiber）；`concurrent_bounded_queue` **不借**（票号+分页+链 libtbb，非定长 ring，用 Vyukov/rigtorp）；仅 `scalable_allocator`/rml pool policy 设计**参考** mem/，不 vendor |
| **oneDPL** | **边界外**（深读校正）| 并行 STL 架在 TBB/OpenMP 之上——而那层正是 HYLE 自造调度器要替换的；算法层非 infra 原语 |

### container/（Tier 0）
| 库 | 角色 | 说明 |
|---|---|---|
| **abseil-cpp** | 参考 | SwissTable 设计（控制字节 + SIMD 探测），hashmap 头号参考；seed-aware default hash 与 table seed 用来打散跨启动/跨表布局，但其 ASLR seed 同样明确不作安全承诺 |
| **folly** | 参考 | F14 hashmap + 并发原语/SmallVector 等；透明 string policy 与 seeded fast hash 形状用于易用性/正常 workload 对账 |
| **moodycamel-ConcurrentQueue** | 参考(bench 对照)| lock-free MPMC 队列，sched global 注入队列参考与同族成熟对照 |
| **rigtorp-MPMCQueue** / **MPMCQueue** | 参考(bench 对照)| 定长环形 MPMC(Vyukov 序号法);BoundedQueue 设计参考 + queue bench-02b 对标 |
| **boost**(intrusive) | 参考 | 通用 caller-owned 稳定节点的 intrusive list/map；auto/safe link、member hook 与 O(1) relink 用于 `container_intrusive` 对账。sched waiter 曾作为迁移候选，后因 link-storage 复用与锁内写/footprint 退化而保留专用链 |

### time/（Tier 2 必要基质）
| 库 | 角色 | 说明 |
|---|---|---|
| **abseil-cpp** | 参考 | TSC 校准算法（`sysinfo.cc`：GetTimeTscPair 取最低延迟样本抗抢占 + `nanosleep` 后按实际 elapsed 测频 + 倍增窗口到 1% 收敛）；ordered measurement 使用其 LFENCE/RDTSC/RDTSCP 序列。另：abseil 默认关闭 cycle-clock 当 wall-time，正是 HYLE 只把硬件 counter 暴露为 profiling clock 的依据 |
| **folly** | 参考 | `hardware_timestamp` 的 x86 TSC / ARM64 CNTVCT 平台形状，以及 raw sampling 与 ordered measurement start/stop 的语义分离 |
| **tracy** | 参考 | invariant-TSC 探测（`CheckHardwareSupportsInvariantTSC`：CPUID leaf1 EDX bit4 + leaf 0x80000007 EDX bit8）+ rdtsc 仅作测量用途的范式 |

> 对参考库均**读设计自实现、不引入外部依赖**。time native 核只借 core 头；target 依赖应服从当前
> C++ 实现图，不能为历史 C adapter 保留产品依赖。MSVC `__msvc_chrono.hpp` steady_clock 的常见
> 频率特判（10/24MHz→乘法）亦是 `monotonic_ns` 补特判的参考。

### hash/（Tier 3）
| 库 | 角色 | 说明 |
|---|---|---|
| **xxHash** | **借（宽输入 runtime path）** | `hash_bytes` 的超 crossover 输入在 x86 AVX2/AArch64 NEON 安全路径使用 XXH3；不是默认 typed hash，也不承诺内容寻址或持久 bit pattern |
| **wyhash** | **借（baseline long-input core）** | 17–128B 先走库内 inline multiply stream；其上默认进入 vendored wyhash baseline，达到平台实测 crossover 且 ISA 可用时再转宽 XXH3 |
| **SipHash-2-4** | **按公开算法/known vectors 自实现** | 显式 `HardenedStringHash` 与 stateless keyed primitive；128-bit secret 必须由调用方安全注入，默认 fast process seed 不冒充 CSPRNG |

### 格式与序列化（消费方路线；当前没有 HYLE serialize factor）
| 库 | 角色 | 说明 |
|---|---|---|
| **simdjson** | **ROUTED / 参考** | JSON 读写的成熟候选；当前不在 HYLE dependency lock、target 或安装闭包中。需要 JSON 的 service/engine 自己选择并链接，未来只有具名 adapter 价值才重新准入 |
| **flatbuffers** | **ROUTED / 参考** | mmap-friendly schema/serialization 候选；当前不是 HYLE vendored 依赖，权重或张量 schema 仍由 loader/engine 拥有 |
| **capnproto** | **ROUTED / 参考** | 跨进程消息候选；当前不构建 `capnp` 或 RPC。若未来进程/IPC 旅程准入，先分别裁决 serialization core 与 runtime，而不是预先借入 |

`ROUTED` 不表示 HYLE 已把这些库下载、编译或导出；它只说明 HYLE 不复制该成熟问题域。真正使用时由
消费方拥有版本、构建和 schema，直到新的 closure card 证明一个 HYLE adapter 能提供 allocator、error、
IO 或交付价值。

### obs/（Tier 2 hook / Tier 3 重机器）
| 库 | 角色 | 说明 |
|---|---|---|
| **tracy** | **后端 + 参考设计** | 中央 lock 固定 `v0.14.1`；启用 trace 时默认由共享源码 cache 取得并验证完整 archive/tree/license，显式 `HYLE_TRACY_DIR` 也必须通过相同 identity 校验。native hook 的编译期 no-op、scoped zone、plot/counter/fiber/frame/message/memory/thread-name 由 Tracy 承载；HYLE 只保留自己的 seam，并保证 shared build 全进程一个 client。server/UI 是外部 profiler 工具，不复制进 obs 产品 API |
| **abseil-cpp** | **参考 + benchmark 机制项** | 同步 `LogSink`、全局注册/摘除、flush、sink 内日志不递归回注册图、process-lifetime registry 是 log runtime 的直接形状参考；其完整 LogEntry 带更多 metadata，点值只能标载荷差异，不能冒充同合同胜负 |
| **folly** | **参考** | handler 同步线程语义、flush、category/handler 分层及默认转义不可信控制字符。HYLE 借安全边界与 adapter 形状；hierarchical category、async writer/file/upload 属于应用日志系统，不塞进低层 producer seam |

### io/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **seastar**(reactor_backend) | 参考 | epoll/aio 也对上呈现 completion 语义的后端抽象范本;HTTP 栈是"自有运行时→自写 HTTP(Ragel parser)"的直接先例(io HTTP 定案参考) |
| **liburing** | **参考(不链)** | io_uring 机制权威:barrier/SQE 布局/多 op(multishot accept/recv、provided buf_ring、ASYNC_CANCEL)。**HYLE 用裸 syscall 自写,不链 liburing**——精确复刻其 v2.15 fast-path barrier + buf_ring 协议(io-01/02/07 已落地) |
| **tokio-uring** | 参考 | 见 sched/ 段:submit/flush 节奏 + msghdr 生命周期(io-04 udp) |
| **libevent** | 参考 | reactor 级:common_timeout 有序链算法(time CommonTimeoutQueue 同款基线)、evdns 异步 DNS(io-09 待)、bufferevent 水位(应用层不做);对 HYLE proactor 价值小于以上 |
| **nghttp2** | **借核(io HTTP/2)** | 中央 lock 固定 `v1.70.0` commit；只构建静态 libnghttp2 protocol core，借 HTTP/2 frame/HPACK/stream/flow-control，自写 allocator、caps 与 over-co_await transport，不取其 event loop/应用层 |

### graph/（Tier 6，编排 + 数据并行）
| 库 | 角色 | 说明 |
|---|---|---|
| **oneTBB** | **参考（设计,不借运行时）** | 数据并行的成熟设计参考：`blocked_range`(grainsize + 递归 split)、`parallel_reduce`(reduction_tree_node + **deterministic** 双档)、`parallel_scan`(两趟)。**不借其 task arena/调度器**(双调度器红线)；运行仍归 HYLE sched |
| **taskflow** | 参考 | 现代 C++ 的 DAG + 数据并行一体框架(`tf::Taskflow` 依赖图 + for_each/reduce/scan + subflow 动态图)，适合对拍 HYLE DAG 编排与数据并行的 API 形态；调度仍归 HYLE sched |
| **eigen** | **参考（数值/几何消费,按需）** | 稠密/稀疏线代（matmul/分解/迭代求解器 CG/BiCGSTAB）的成熟参考。`EIGEN_GEMM_THREADPOOL` 暴露 `pool->NumThreads()` 路径，但未来 HYLE adapter 仍须实证任务提交接口和所有权，不能类比 ggml 的不可注入 `set_threadpool`。由具名数值或几何消费者触发，不以库名代替性能结论 |
| **openvdb** | **参考（批处理/稀疏网格）** | 工业级稀疏体素/网格数据结构和 NanoVDB host-buffer 资源形状；只读其数据结构、分块遍历与 buffer ownership，TBB 并行不进入 HYLE runtime。完整几何/体数据消费者仍由 Graph/Compute 后续任务绑定 |

## 阅读优先级（按问题域）

- **底层运行时**：marl（sched 精读）、moodycamel（队列）、boost.context（fcontext）、abseil/folly（container）、xxHash/wyhash（table byte hash）、Highway（SIMD 产品机制与架构参考）、xsimd（独立 benchmark peer）、simdutf（UTF-8 validate/count 借核）。
- **计算与 IO**：ggml + llama.cpp、ds4、vLLM、SGLang（推理需求/工程参考）、Cycles/OpenVDB/Kompute（渲染与资源形状）、oneDNN/oneMath（Intel CPU/数学后端参考）、Seastar/libuv/nginx（异步服务与 IO）。
- **按真实消费者裁决**：simdjson/flatbuffers/capnproto 当前是消费方路线；tracy 与 nghttp2 是已经审查的
  产品机制；oneTBB/taskflow、Eigen/OpenVDB 继续作设计/消费参考。oneDPL/oneDAL **边界外**（见下）。

## 借 vs 造的总原则（呼应 charter）

借**多年解决的通用难题**（Highway 数学/向量机制、ggml 算子、boost.context 汇编、liburing syscall）；造**贴 HYLE 需求或差异焦点**（container 自有容器算法/API、sched 调度、HYLE SIMD 产品合同与缺口组合）。container 的 byte-hash core、UTF-8 validate/count 以及 target-bound string/bitset register lowering 分别借 wyhash/XXH3、simdutf 与锁定 Highway，并不等于把容器 ownership/布局交给外部 runtime。所有借入第三方一律封 adapter / 不外露其类型（同 ggml/mimalloc 解耦原则）。

### 关键区分：加速库分两类，只借第一类（这是 oneTBB 为何边界外的原因）

| 类 | 是什么 | 例 | HYLE |
|---|---|---|---|
| **指令/kernel 级加速**（让单次计算更快）| 优化的算子/数学核/SIMD intrinsic；部分库仍自带不可替换的内部并行 | oneDNN(AMX/VNNI JIT)、oneMath(BLAS)、Highway 数学/向量机制、ggml 算子、XXH3 | ✅ **借**（每个 adapter 明确线程所有权；不能一概声称注入 HYLE pool）|
| **并行运行时级加速**（把任务铺到多核）| **自带 work-stealing 调度器 / task arena / 线程池** | **oneTBB**、oneDPL、OpenMP、asio io_context、boost.fiber | ❌ **不借**（这是 HYLE 自己的活）|

**oneTBB 确实是加速库——但它加速的是"运行时层"，恰是 HYLE 要独占的那一层。** 它的 `parallel_for` 离不开 task arena（自带 work-stealing 调度器）；引它 = HYLE fiber 调度器 + TBB 调度器**两套在同一批核上抢线程、过度订阅、cache 争抢**——正是 HYLE 存在要消灭的"双调度器阻抗失配"（同拒 boost.fiber/asio 的理由）。所以 oneTBB scheduler/queue、oneDPL **边界外**；只 oneTBB 的 allocator pool 设计作 mem/ 参考。

**那 CPU 数据并行（matmul 等铺满多核）靠谁？** 三层，都不需要 TBB：
1. **借来的算子必须逐库明确并行所有权**：ggml 的 `set_threadpool` 只接受 ggml 自己创建的
   threadpool，不是 HYLE scheduler 注入点。HYLE 对不可合批的小任务用 `n_threads=1` 后自行铺并发；
   对能合批的大 graph 才让 ggml 自有并行独占该次计算，并从 blocking pool 提交。oneDNN 等未来 adapter
   也必须实证其注入接口，不能从函数名类推。
2. **HYLE 自有 kernel 的数据并行**：work-stealing 调度器做 fork-join / parallel-for——**一个调度器同时干 IO 并发(fiber)和 CPU 数据并行**，这正是 work-stealing 的本职。
3. **指令级加速**：当前借锁定 Highway 数学/向量机制，按需借 oneDNN 的 AMX kernel（第一类），不借 TBB；xsimd 保持独立 peer。

### 隐藏运行时清单（引依赖前必查，防"悄悄混进第二个运行时"）

凡库**自带线程/事件循环/调度器/控制流接管**的，都是红线——只借它的"叶子计算"，不借它的运行时。检测法：源码搜 `pthread_create`/`std::thread`；看它有没有要你 `run()`/`loop()`/`executor`/`pool`；看它装不装 signal handler/timer（撞 HYLE time/）。

| 隐藏运行时来源 | 处置 |
|---|---|
| 并行框架（TBB/OpenMP/oneDPL）| 边界外；数据并行用 HYLE 调度器 |
| 异步/IO 运行时（asio/libuv/seastar 整框架）| 只借 syscall 层（liburing），事件循环自造 |
| 分配器后台线程（mimalloc/jemalloc 的 purge 线程）| 配置关掉或纳管，别让它自起线程 |
| ggml 内部线程 | 不伪造注入：小任务 `n_threads=1` + HYLE 外层并发；合批大 graph 才使用 ggml 自有并行并由 blocking pool 隔离 |
| 其他算子库内部线程（如未来 oneDNN） | adapter 落地前逐项验证外部 pool 接口；无证据不得宣称已注入 HYLE scheduler |
| 日志/profiler 异步刷盘（spdlog async / tracy server）| 借设计不借库，或用 HYLE 调度器驱动 |
| 协程库自带调度器 | 只用语言级 `co_await`，调度归 HYLE sched |

**铁律**：HYLE 必须独占所有 OS 线程与控制流——每个 OS 线程要么是 HYLE worker、要么归 HYLE blocking 池管。加任何依赖前先问：**它起线程吗？它想接管控制流吗？** 是 → 只借计算、运行时留在门外。
