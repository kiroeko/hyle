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
| **oneDNN** | **借（算子库）** | **Intel DL 原语 + AMX/VNNI 加速正主**；运行时 cpuid dispatch + Xbyak JIT。定位是**算子库（ggml 同级 CPU 算子供应商，接 bridge）**，非 device/stream 后端——它自带 JIT 不暴露 raw kernel launch |
| **oneMath** | **仅参考（不借）** | Intel 数学（前 oneMKL）：BLAS/FFT。**SYCL 重依赖**（引它=引整条 SYCL/DPC++ 工具链），HYLE 已有 ggml BLAS；仅参考其 domain×device×library 三级分发表 |
| **oneDAL** | **边界外**（深读校正）| 经典 ML（kmeans/PCA/knn/forest）是 L1/L2 应用层，非 infra 原语 |

### sched/（Tier 0）+ 并发
| 库 | 角色 | 说明 |
|---|---|---|
| **marl** | 参考（精读）| **最贴 HYLE 的 fiber + work-stealing 调度库**，第一参考 |
| **tokio** | 参考 | work-stealing/LIFO slot/unpark 接力/per-core 队列的设计源 |
| **boost**(context) | **借（仅 3 汇编）** | **fcontext** 是 fiber 切换原语来源；`fcontext_t`/`transfer_t` 纯 POD 零依赖，只取 `jump/make/ontop` 三例程 .S/.asm（**本地 context 是空 submodule，须从上游拉**）+ 自写 ~10 行头，不要 config/封装层。boost.intrusive 仅参考 |
| **seastar** | 参考 | per-core 局部性、reactor_backend completion 抽象、xcpu_freelist；**不学** share-nothing 调度 |
| **oneTBB** | **多为边界外**（深读校正）| task scheduler 边界外（自带 work-stealing 抢权，同拒 boost.fiber）；`concurrent_bounded_queue` **不借**（票号+分页+链 libtbb，非定长 ring，用 Vyukov/rigtorp）；仅 `scalable_allocator`/rml pool policy 设计**参考** mem/，不 vendor |
| **oneDPL** | **边界外**（深读校正）| 并行 STL 架在 TBB/OpenMP 之上——而那层正是 HYLE 自造调度器要替换的；算法层非 infra 原语 |

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

## 开工优先级（配合 roadmap 批次）

- **批3-4 即用**：marl（sched 精读）、moodycamel（队列）、boost.context（fcontext）、abseil/folly（container）、highway/sleef（simd）。
- **批5 即用**：ggml + llama.cpp（compute 主力借/参考）、oneDNN/oneMath（Intel 加速，CPU/数学后端）、seastar（io reactor_backend）。
- **批6 / 按需**：simdjson/flatbuffers/capnproto（serialize）、xxHash/wyhash（hash）、tracy（obs）、oneDPL/oneTBB/oneDAL（并行/分析，多属参考）。

## 借 vs 造的总原则（呼应 charter）

借**多年解决的通用难题**（sleef 数学核、ggml 算子、boost.context 汇编、liburing syscall）；造**贴 HYLE 需求或差异焦点**（container 全自造、sched 调度自造、simd 抽象自造）。Intel 系（oneDNN/oneMath）作为 CPU 计算加速的**可选后端**接入，走能力门控 dispatch，不强绑。所有借入第三方一律封 adapter / 不外露其类型（同 ggml/mimalloc 解耦原则）。
