# HYLE 参考库映射

> 有本地源码的参考项目位于 `C:\public\project-hyle\`（与 hyle 并列）；其余链接官方设计/API。
> 裁决必须记录所读 commit 或文档版本，不能把 sibling checkout 的当前 HEAD 当永久事实。本表把每个库
> 映射到 HYLE 模块，并标注**借**（vendored 依赖/adapter 对接）、**参考**（读设计、不依赖）或
> **ROUTED**（消费方直接采用成熟路线）。
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
| compute | ggml、llama.cpp、ds4、vLLM、SGLang、MLX、oneDNN、CUDA/ROCm/Vulkan、Kompute | device/context/queue/completion、operator provider、storage/interop、kernel 与隐藏运行时 |
| graph | ggml/llama.cpp、oneDNN Graph、ONNX Runtime、TensorRT、PJRT/IREE、vLLM/SGLang | logical graph、provider partition、compiled artifact、workspace、shape specialization 与 replay |

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
| **simdutf** | 借（container_text 直接编入） | vendored v9.1 amalgamation 无条件编入 `hyle_container_text`；当前 UTF validate/count 与已准入 G01 Base64 共用其 runtime ISA dispatch 和 caller-buffer algorithm core，公开头不泄露 vendor 类型或 option bitmask |
| **sleef** | **已退役** | 曾评估作 transcendental 核；当前不进产品或基准默认闭包，只有新的具名缺口才重新准入 |
| **nanobench** | 借（测试期） | 微基准框架，vendored 在 `third_party/`；统一性能仪器在其上做跨进程同进程配对：预声明 margin、分布无关中位数 CI、comparability/qualification/exact-performance 分轴、同合同 non-inferiority gate、逐进程 raw JSON 与正反序/shape 合成；sign p 值只作诊断 |

### compute/（Tier 1，含 Intel 计算加速）
| 库 | 角色 | 说明 |
|---|---|---|
| **ggml** | Compute 需求/机制输入 | `compute` 当前为 `QUEUED`，旧 bridge 与 dependency lock 只作重设计输入，不是可选择的产品能力。未来可借其 CPU/backend/operator 核，但 HYLE 必须重新裁决 threadpool、allocator、device lifetime 与 source/static/shared 交付 |
| **kompute** | Compute 参考 | Vulkan compute 的无隐藏 event-loop 形状可参考；`compute` 尚未开工时，不从旧 task 编号继承 backend、fence 或 ownership 决策 |
| **vitamin-cuda** | **参考（compute 自造 kernel）** | 教学级 CUDA kernel 库（`C:\public\project-hyle\vitamin-cuda`），SGEMM/HGEMM/flash_attn/softmax/rope/reduce 等，含中英“beating cuBLAS”逐步优化 writeup（`posts/`）。它是 ggml 显存边界之外 HYLE 自造 kernel 的优化参考，不是 HYLE kernel 的性能证明。SGEMM 展示 naive → block tiling+共享内存 → 转置写 smem+float4 → swizzle → 合并写回 → double buffer 的阶梯；采用前仍须在 HYLE 的真实合同上复证 |
| **llama.cpp** | 参考 | 多变体 dispatch、CMake ISA 处理、探测器隔离和推理引擎工程范式；host/device 资源、backend 与 operator 形状只作未来 Compute 审计输入 |
| **ds4** | 真实消费参考 | 紧凑 CPU/统一内存量化推理路径；用于发现行算子、内存、top-k 和数据布局需求，不自动作性能 peer |
| **vLLM** | 真实消费参考 | runtime、CPU offload/custom-op、paged KV、sampling 与 serving 需求；GPU/Triton/tensor fusion 形状必须归到正确 owner |
| **SGLang** | 真实消费参考 | serving/kernel registry、attention、MoE、sampling、speculative decoding 需求；用于发现语义族，不机械复制 Python/GPU API |
| **Metal** | Apple 主要 backend 参考 | Apple 的低层 graphics/compute API；HYLE 只借 compute resource/queue/pipeline/sync/interop，不承接 raster/present。见 [Metal overview](https://developer.apple.com/metal/) |
| **MPS / MPSGraph** | Apple provider 候选 | MPS 是优化 primitive，MPSGraph 是[符号计算图](https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph)；按 region 能力、artifact/workspace owner 与 command queue 互操作逐项审计 |
| **MLX** | Apple 首个资格化的 provider/interop 候选 | 审计 Apple Silicon [unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)、Metal stream 与 [compiled execution](https://ml-explore.github.io/mlx/build/html/usage/compile.html)；MLX 有完整 C++ API，是上层 runtime/provider，不能与 Metal backend 混成一个枚举或直接泄漏其类型 |
| **Core ML** | 高层 routed runtime 参考 | [Core ML](https://developer.apple.com/documentation/coreml/) 以模型为单位并可使用 CPU/GPU/Neural Engine；首轮不把它伪装成低层 op provider，只有模型级委托保留所需控制且有真实纵切时才建 adapter |
| **CUDA / NVIDIA libraries** | NVIDIA 主要 backend/provider 参考 | device/context/stream/event、graph capture、external memory、kernel/module 与 vendor op；每项借用分别登记 allocation、threading、license 和 delivery，不用“CUDA available”代替算子证据 |
| **ROCm / HIP** | 必须补齐的 AMD backend/provider 参考 | 当前 HYLE 没有合格 HIP 实现；按 device/module/queue/completion、VMM、graph 和 operator 能力建立独立资格，不把 CUDA 源码宏替换或 Vulkan fallback 冒充 ROCm 支持 |
| **Vulkan / MoltenVK** | 正式保底 backend + renderer interop 参考 | llama.cpp 当前把 Vulkan 作为可构建、可运行的 GPU backend；HYLE 长期合同要覆盖 SPIR-V compute、queue/barrier/timeline、external memory/sync 和消费级设备。Windows/Linux 优先原生驱动；Apple 经 MoltenVK 翻译到 Metal，只签 portability，不替代 Apple 原生 Metal/MLX/MPS 路线 |
| **OpenAI Jalapeño / InferenceX 结果** | 行业方向证据，不是 HYLE peer | inference-only 全栈 co-design 对 prefill/decode、KV locality、memory/network placement 与 latency-throughput-power Pareto 的公开案例；只使用其公开 workload/口径反查需求，不把厂商自测或封闭硬件当 HYLE SOTA 证明。见 [`motivation`](motivation.md#2026-08-的外部事实推理正在成为全栈系统问题) |
| **Apple M5 Ultra** | 可获得统一内存平台方向，不是仅凭规格成立的性能 peer | 最高 512 GB unified memory、1.2 TB/s memory bandwidth 与 quad-die interconnect 使超出普通离散显存的本地量化模型成为正式 workload；HYLE 仍须实测 residency、跨 die locality、Metal/kernel 与端到端 token 指标。见 [`motivation`](motivation.md#2026-08-的外部事实推理正在成为全栈系统问题) |
| **oneDNN** | Compute 已锁定 provider + 后续参考 | v3.13.2 的 x64 BRGEMM ukernel 以独立可选 factor 借入 dense row-major f32 MatMul；HYLE 在 prepare 冷路径生成 immutable full/tail kernel，以显式同步 executor 划分 N tile，并让每个 Invocation 使用 caller-owned scratchpad。oneDNN runtime 为 SEQ，不暴露 vendor 类型或隐藏线程池；AArch64 ukernel 因 warm vector allocation 未采用。量化、其他 primitive 与 Graph partition 仍须逐卡裁决，不能从依赖存在推断已支持 |
| **oneMath** | **仅参考（不借）** | Intel 数学（前 oneMKL）：BLAS/FFT。**SYCL 重依赖**（引它=引整条 SYCL/DPC++ 工具链），HYLE 已有 ggml BLAS；仅参考其 domain×device×library 三级分发表 |
| **oneDAL** | **边界外**（深读校正）| 经典 ML（kmeans/PCA/knn/forest）是 L1/L2 应用层，非 infra 原语 |

### sched/（Tier 0）+ 并发
| 库 | 角色 | 说明 |
|---|---|---|
| **marl** | 参考（精读）| **最贴 HYLE 的 fiber + work-stealing 调度库**，第一参考 |
| **folly** | 参考 | `AsyncScope/CancellableAsyncScope` 的动态 child owner、显式 join/drain，以及 Executor 多 priority queue；只借语义，不引入其 executor/runtime |
| **tokio** | 参考 | work-stealing、LIFO slot、unpark 接力、per-core queue，以及 IO submit/park 协作的成熟机制参考；不引入第二 runtime |
| **tokio-uring** | 参考 | io_uring submit/flush 节奏和跨异步边界保持 kernel argument backing 的生命周期参考 |
| **crossbeam-deque** | 参考(bench 对照)| Chase-Lev work-stealing deque(Rust)；work-stealing 同族的成熟对照（独立 cargo crate，gated 编译） |
| **boost**(context) | **借（仅 3 汇编）** | **fcontext** 是 fiber 切换原语来源；`fcontext_t`/`transfer_t` 纯 POD 零依赖，只取 `jump/make/ontop` 三例程 .S/.asm（**本地 context 是空 submodule，须从上游拉**）+ 自写 ~10 行头，不要 config/封装层。boost.intrusive 仅参考 |
| **seastar** | 参考 | per-core 局部性、reactor_backend completion 抽象;**不学** share-nothing 调度;**xcpu_freelist 评估后否决**(其 owner-local 非原子表依赖 share-nothing+线程钉死+池长于进程,HYLE 通用生命周期固定池给不出,且跨核 free 正确性已由地址路由原子 push 解决,见 mem-design) |
| **oneTBB** | **多为边界外**（深读校正）| task scheduler 边界外（自带 work-stealing 抢权，同拒 boost.fiber）；只参考 low/normal/high 非抢占 priority 语义。`concurrent_bounded_queue` **不借**（票号+分页+链 libtbb，非定长 ring，用 Vyukov/rigtorp）；`scalable_allocator`/rml pool policy 只作 mem 设计参考，不 vendor |
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

### 相邻格式、标识、数据处理与 IPC（消费方路线）
| 库 | 角色 | 说明 |
|---|---|---|
| **simdjson / Boost.JSON** | **ROUTED / 参考** | simdjson 承载 hot read/DOM；需要 owning mutation/write 时使用 Boost.JSON 等完整 JSON owner。它们当前不在 HYLE dependency lock、target 或安装闭包中；service/engine 自己选择并链接，只有具名 adapter 价值才重新准入 |
| **flatbuffers** | **ROUTED / 参考** | mmap-friendly schema/serialization 候选；当前不是 HYLE vendored 依赖，权重或张量 schema 仍由 loader/engine 拥有 |
| **capnproto** | **ROUTED / 参考** | 跨进程消息候选；当前不构建 `capnp` 或 RPC。若未来进程/IPC 旅程准入，先分别裁决 serialization core 与 runtime，而不是预先借入 |
| **Boost.URL / Boost.UUID** | **ROUTED** | URL parse/resolve 与 UUID generate/format 的消费方路线；Transfer 内部窄 parser 和 HYLE random primitive 都不冒充完整公开产品 |
| **OpenSSL EVP / BLAKE3** | **ROUTED** | stable/cryptographic digest 路线；`TransferVerifier` 接受 incremental adapter，Container `hash_bytes` 只作进程内 table hash |
| **zlib / Zstandard / Brotli / libarchive** | **ROUTED** | HTTP/content、artifact 与 archive codec 由消费方拥有格式、limits、版本和安全更新；HYLE 提供 bounded buffer 与 IO，不再造薄 codec façade |
| **ICU4C / utf8proc** | **ROUTED** | normalization、casefold 与 grapheme 路线；HYLE ContainerText 只拥有 byte/ASCII/UTF validation/count/conversion 快路径 |
| **Boost.Interprocess / ZeroMQ** | **ROUTED / 参考** | 一般 local IPC 的 named sync/message/shared-memory 以及复杂消息 topology 路线；HYLE `SharedMapping` 只承诺 storage，scheduler-native zero-copy channel 仍需独立证据 |
| **RE2 / Hyperscan** | **ROUTED** | 通用 linear-time regex 与大规则集 bulk scan 路线；HYLE SIMD 的 literal/value search 不复制 pattern compiler 或 automaton runtime |
| **CLI11 / cxxopts / toml++ / yaml-cpp** | **ROUTED** | 命令行与配置格式由应用拥有 option/schema/default/reload；未来 Process 机制只消费已经决定的 argv/env |
| **fmt / ICU** | **ROUTED** | 通用 formatting/localization 路线；HYLE Error 的 bounded/no-throw formatter 只服务失败路径成本与可用性 |
| **Eigen / GLM / Apache Arrow** | **ROUTED / 参考** | matrix/geometry 与 columnar schema 路线；HYLE Math/SIMD/Mem 只提供可组合原语和 backing，不复制领域数据模型 |

`ROUTED` 不表示 HYLE 已把这些库下载、编译或导出；OpenSSL 只有在选择 IOTLS 时才由 HYLE package
拥有，其它场景仍是消费方依赖。路线只说明 HYLE 不复制该成熟问题域。真正使用时由消费方拥有版本、
构建和 schema，直到新的 closure card 证明一个 HYLE adapter 能提供 allocator、error、IO 或交付价值。

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
| **liburing** | **参考(不链)** | io_uring barrier、SQE 布局、multishot、provided buffers 与 async cancel 的机制参考；HYLE 的 Linux backend 直接使用 kernel ABI，不把 liburing 变成公开或链接依赖 |
| **tokio-uring** | 参考 | submit/flush 节奏与 `msghdr` 等 kernel argument backing 的异步生命周期参考 |
| **libevent** | 参考 | common-timeout 有序链、异步 DNS 与水位控制的 reactor 参考；HYLE 对外仍保持自己的 completion、Time 与 backpressure 合同 |
| **libuv** | 参考 | recursive file watch、sendfile 和 child-process 的跨平台语义/退化边界；不借其 event loop、thread pool 或 process runtime |
| **Boost.Beast** | 参考 | WebSocket 的预读 HTTP handoff、message/partial I/O、control frame、close 与并发约束；不引入 Boost.Asio 第二 runtime |
| **Boost.Process v2** | 参考 | argv/env/cwd、stdio、async wait/cancel 和 process group 的成熟生命周期；其 Asio 绑定不进入 HYLE process factor |
| **OpenSSL** | 借核(io TLS) | provider 机制承载证书、会话、ALPN 与已准入 file/borrowed PEM/password credential source；HYLE 拥有 opaque context、secret lifetime、错误和 Scheduler transport，不泄漏 provider 类型 |
| **nghttp2** | **借核(io HTTP/2)** | 中央 lock 固定 `v1.70.0` commit；只构建静态 libnghttp2 protocol core，借 HTTP/2 frame/HPACK/stream/flow-control，自写 allocator、caps 与 over-co_await transport，不取其 event loop/应用层 |

### graph/（计算 DAG、编译与可复用执行物）
| 库 | 角色 | 说明 |
|---|---|---|
| **ggml / llama.cpp** | 真实 C++ 实现与消费参考 | backend graph plan、scheduler reserve/split/allocate/async compute、ubatch graph rebuild/reuse 与 buffer ownership；用于验证轻量本地推理形状，不把 ggml IR 直接改名为 HYLE Graph |
| **oneDNN Graph** | 成熟 provider/partition 参考 | [partition](https://uxlfoundation.github.io/oneDNN/v3.9/group_dnnl_graph_api_partition.html) 是 op 与 logical tensor 的编译/执行单元，compiled partition 与 [caller scratchpad](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_scratchpad.html) 分离；用于约束 region、artifact 与 workspace 生命周期 |
| **ONNX Runtime** | provider partition 参考 | [Execution Provider](https://onnxruntime.ai/docs/reference/high-level-design.html) 先声明 capability、再把支持的 subgraph 编译为 fused node；HYLE 同样在冷路径 partition，但不复制 ONNX schema/session 产品 |
| **TensorRT** | compiled artifact / invocation 参考 | [NetworkDefinition -> Engine -> ExecutionContext](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/coreConcepts.html) 区分逻辑网络、不可变可共享 engine 与每次运行 context；用于约束 Executable/Invocation，不作为唯一 NVIDIA 路线 |
| **PJRT / IREE** | runtime 边界参考 | [PJRT](https://openxla.org/xla/pjrt/cpp_api_overview) 区分 executable、loaded executable、buffer/device；[IREE invocation model](https://iree.dev/developers/design-docs/invocation-execution-model/) 区分 program/executable/context 与 HAL command；用于防止 Graph 吞掉 backend runtime |
| **vLLM / SGLang** | shape specialization/capture 消费参考 | compile cache、piecewise/full CUDA graph、attention backend 与 breakable region 证明同一逻辑模型需要多个 shape/provider specialization；模型 batching/capture policy 仍归引擎 |
| **MLX / MPSGraph** | Apple compiled graph 候选参考 | 对账 lazy/symbolic graph、compile/replay、unified-memory 与 Metal command/stream；若作为 HYLE provider，其 artifact 留在 provider island，不把其数组 owner 冒充 Mem Buffer |
| **oneTBB / taskflow** | 仅作相邻领域参考 | parallel-for/reduce/scan 与 callable task DAG 属未来 `parallel`/`graph_incremental` 候选，不再定义 HYLE 计算 Graph；只有独立非计算消费者才能重新准入 |
| **OpenVDB / renderer compute pipeline** | 泛化纵切 | 稀疏网格、资源 upload、Vulkan compute pass 与 delayed completion 防止 Graph 只适配 Transformer；raster/present/render-graph policy 留渲染产品 |

## 阅读优先级（按问题域）

- **底层运行时**：marl（sched 精读）、moodycamel（队列）、boost.context（fcontext）、abseil/folly（container）、xxHash/wyhash（table byte hash）、Highway（SIMD 产品机制与架构参考）、xsimd（独立 benchmark peer）、simdutf（UTF-8 validate/count 借核）。
- **计算与 IO**：ggml + llama.cpp、ds4、vLLM、SGLang（推理需求/工程参考），Apple MLX-first + 原生
  Metal custom-kernel/interop 与 NVIDIA CUDA + vendor providers（两条主要平台栈），Vulkan/MoltenVK（正式消费级
  保底/renderer 路线），HIP/ROCm（待补齐覆盖），oneDNN、ONNX
  Runtime、TensorRT、PJRT/IREE（provider/compiled graph 参考），Seastar/libuv/nginx（异步服务与 IO）。
- **按真实消费者裁决**：本页 `ROUTED` 表中的格式、标识、codec、Unicode 与 IPC 库由消费方直接选择；
  Tracy 与 nghttp2 是已经审查的产品机制；oneTBB/taskflow、Eigen/OpenVDB 继续作设计/消费参考。
  oneDPL/oneDAL **边界外**（见下）。

## 借 vs 造的总原则（呼应 charter）

借**多年解决的通用难题**（Highway 数学/向量机制、合格 provider 算子、boost.context 汇编、
liburing syscall）；造**贴 HYLE 需求或差异焦点**（container 算法/API、sched、Graph 编译组合、HYLE
胜出 kernel）。ggml、MLX、oneDNN 等只是候选 provider，必须连同 copy、layout、线程和生命周期在真实
region 上裁决，不能按库名预先写成“必借”。所有借入第三方一律封 adapter、不外露其类型。

### 关键区分：叶子计算与运行时接管

| 类 | 是什么 | 例 | HYLE |
|---|---|---|---|
| **指令/kernel 级加速**（让单次计算更快）| 优化的算子/数学核/SIMD intrinsic；部分库仍自带不可替换的内部并行 | oneDNN、MLX provider、Metal/CUDA kernels、Highway、ggml、XXH3 | ✅ **逐能力裁决后借**（每个 adapter 明确线程、storage 与 completion；不能一概声称注入 HYLE pool）|
| **并行运行时级加速**（把任务铺到多核）| **自带 work-stealing 调度器 / task arena / 线程池** | **oneTBB**、oneDPL、OpenMP、asio io_context、boost.fiber | 默认 `ROUTED`；只有不可拆的 provider region 有真实 material 价值时才隔离接入，不成为第二个公共 runtime |

**oneTBB 确实是加速库——但它加速的是"运行时层"，恰是 HYLE 要独占的那一层。** 它的 `parallel_for` 离不开 task arena（自带 work-stealing 调度器）；引它 = HYLE fiber 调度器 + TBB 调度器**两套在同一批核上抢线程、过度订阅、cache 争抢**——正是 HYLE 存在要消灭的"双调度器阻抗失配"（同拒 boost.fiber/asio 的理由）。所以 oneTBB scheduler/queue、oneDPL **边界外**；只 oneTBB 的 allocator pool 设计作 mem/ 参考。

**CPU 数据并行怎样组合？** HYLE-native kernel 使用 HYLE Sched；第三方 provider 必须声明能否注入外部
线程池、限制线程数或只能自管。Graph compiler 按 region 大小、外层并发和 handoff 成本选择路线，禁止
默认把 provider 多线程嵌入 HYLE 多 worker。`n_threads=1`、独占 region 或 blocking isolation 都是需由
workload 裁决的策略，不写成适用于所有 shape 的永久规则。指令级仍使用 SIMD/ISA provider，不因线程
选择而改变 operator 语义。

### 隐藏运行时清单（引依赖前必查）

凡库自带线程、event loop、scheduler 或 graph runtime，都必须进入 capability/cost/teardown 合同。优先
借叶子计算；若成熟 provider 只有连 runtime 才能提供 material 优势，可以作为 region island 接入，但
必须由 HYLE Sched 进行资源 admission/隔离，禁止无界嵌套和隐藏 shutdown。检查源码中的线程创建、
`run/loop/executor/pool`、signal/timer、全局 cache 与 allocator。

| 隐藏运行时来源 | 处置 |
|---|---|
| 并行框架（TBB/OpenMP/oneDPL）| 边界外；数据并行用 HYLE 调度器 |
| 异步/IO 运行时（asio/libuv/seastar 整框架）| 只借 syscall 层（liburing），事件循环自造 |
| 分配器后台线程（mimalloc/jemalloc 的 purge 线程）| 配置关掉或纳管，别让它自起线程 |
| ggml 内部线程 | 不伪造注入；adapter 复验当前 threadpool API，按 region/外层并发选择限线程、独占或隔离，并把决策写入 compile report |
| 其他算子库内部线程（如未来 oneDNN） | adapter 落地前逐项验证外部 pool 接口；无证据不得宣称已注入 HYLE scheduler |
| 日志/profiler 异步刷盘（spdlog async / tracy server）| 借设计不借库，或用 HYLE 调度器驱动 |
| 协程库自带调度器 | 只用语言级 `co_await`，调度归 HYLE sched |

**铁律**：HYLE 必须能解释并约束所有线程与控制流。HYLE-native 路线使用 HYLE Sched；provider-owned
线程必须显式登记、配置或隔离，并在 plan/report 中可见。无法解释资源上限、嵌套、取消和 shutdown 的
runtime 不得进入产品。
