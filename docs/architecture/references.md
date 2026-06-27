# HYLE 参考库映射

> 参考项目均已 clone 到 `C:\public\project-hyle\`（与 hyle 并列）。本表把每个库映射到 HYLE 模块，标注**借**（vendored 依赖/adapter 对接）还是**参考**（读设计、不依赖），及开工优先级。
>
> 角色定义：**借** = 进 HYLE 构建（封 adapter 或 vendored）；**参考** = 读其设计/代码，自己实现，不引入依赖。

## 按模块映射

### simd/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **highway** | 参考 | 可移植 vector + 运行时多变体 dispatch 的头号范本（scalable-friendly 接口、HWY_DYNAMIC_DISPATCH）|
| **xsimd** | 参考 | 定宽 SIMD wrapper + 数学函数实现参考 |
| **sleef** | 借 | transcendental 数学核（exp/log/tanh），封 adapter，HYLE 不手搓多项式 |

### compute/（Tier 1，含 Intel 计算加速）
| 库 | 角色 | 说明 |
|---|---|---|
| **ggml** | 借（主力）| 算子 + 多后端（cuda/hip/metal/vulkan/sycl/webgpu/opencl…），`ggml_bridge` adapter 对接上游 pin 版；同时参考 ggml-backend 接口、dev_caps、CPU 多变体 dispatch |
| **llama.cpp** | 参考 | 多变体 DLL+score dispatch、CMake ISA 处理、探测器隔离（-fno-LTO）、推理引擎工程范式 |
| **oneDNN** | 借/参考 | **Intel DL 原语 + AMX/VNNI 加速正主**；CPU 后端可借其 primitive，或参考其能力门控 |
| **oneMath** | 借/参考 | Intel 数学（**前 oneMKL 改名**）：BLAS/FFT/向量数学；需稠密线代时的工业级实现 |
| **oneDAL** | 参考 | Intel 数据分析/经典 ML 算法；多属上层，按需参考 |

### sched/（Tier 0）+ 并发
| 库 | 角色 | 说明 |
|---|---|---|
| **marl** | 参考（精读）| **最贴 HYLE 的 fiber + work-stealing 调度库**，第一参考 |
| **tokio** | 参考 | work-stealing/LIFO slot/unpark 接力/per-core 队列的设计源 |
| **boost**(context) | 借 | **fcontext** 是 fiber 上下文切换原语来源（sched 第一节定）；boost.intrusive 参考 |
| **seastar** | 参考 | per-core 局部性、reactor_backend completion 抽象、xcpu_freelist；**不学** share-nothing 调度 |
| **oneTBB** | 参考 | task 并行/并发容器参考；HYLE 用 fiber，不直接用其调度器 |
| **oneDPL** | 参考 | 并行 STL 算法参考（若 algorithm 层要并行原语）|

### container/（Tier 0）
| 库 | 角色 | 说明 |
|---|---|---|
| **abseil-cpp** | 参考 | SwissTable 设计（控制字节 + SIMD 探测），hashmap 头号参考 |
| **folly** | 参考 | F14 hashmap + 并发原语/SmallVector 等 |
| **moodycamel-ConcurrentQueue** | 参考 | lock-free MPMC 队列，sched 的 global 注入队列直接参考 |
| **boost**(intrusive) | 参考 | 侵入式 list/map（sched run-queue/waiter 链）|

### hash/（Tier 3）
| 库 | 角色 | 说明 |
|---|---|---|
| **xxHash** | 借/参考 | 非加密快速哈希 |
| **wyhash** | 借/参考 | 更快的非加密哈希，可作 hashmap 默认 hasher 候选 |

### serialize/（Tier 3）
| 库 | 角色 | 说明 |
|---|---|---|
| **simdjson** | 参考 | **SIMD JSON 解析**（替代 rapidjson），贴 HYLE 极致哲学 |
| **flatbuffers** | 参考 | 零拷贝序列化 + mmap 直读 |
| **capnproto** | 参考 | 零拷贝序列化 + RPC（RPC 部分边界外）|

### obs/（Tier 2 hook / Tier 3 重机器）
| 库 | 角色 | 说明 |
|---|---|---|
| **tracy** | 参考 | 游戏级 profiler，instrumentation hook 设计参考 |

### io/（Tier 1）
| 库 | 角色 | 说明 |
|---|---|---|
| **seastar**(reactor_backend) | 参考 | epoll/aio 也对上呈现 completion 语义的后端抽象范本 |
| liburing | 借 | io_uring syscall 薄封装（仅 Linux；非本目录，构建期拉）|

## 开工优先级（配合 roadmap 批次）

- **批3-4 即用**：marl（sched 精读）、moodycamel（队列）、boost.context（fcontext）、abseil/folly（container）、highway/sleef（simd）。
- **批5 即用**：ggml + llama.cpp（compute 主力借/参考）、oneDNN/oneMath（Intel 加速，CPU/数学后端）、seastar（io reactor_backend）。
- **批6 / 按需**：simdjson/flatbuffers/capnproto（serialize）、xxHash/wyhash（hash）、tracy（obs）、oneDPL/oneTBB/oneDAL（并行/分析，多属参考）。

## 借 vs 造的总原则（呼应 charter）

借**多年解决的通用难题**（sleef 数学核、ggml 算子、boost.context 汇编、liburing syscall）；造**贴 HYLE 需求或差异焦点**（container 全自造、sched 调度自造、simd 抽象自造）。Intel 系（oneDNN/oneMath）作为 CPU 计算加速的**可选后端**接入，走能力门控 dispatch，不强绑。所有借入第三方一律封 adapter / 不外露其类型（同 ggml/mimalloc 解耦原则）。
