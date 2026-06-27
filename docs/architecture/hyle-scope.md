# HYLE Scope 定义

> 本文档定义 HYLE 的范围边界：包含什么、不包含什么、有多大。资产盘点依据见 [`clay-asset-liability.md`](./clay-asset-liability.md)。

## HYLE 是什么

**一个 C++ 基础设施库**（编译成带 C-ABI 的共享库），给上层（AI agent、训练/推理框架）当地基。把四样东西做在**同一套底座**上：

1. **调度** — 一个调度器同时管 线程 / 进程 / 纤程 / 协程
2. **内存** — 分配器 / arena / cache-line 对齐 / NUMA
3. **计算** — CPU SIMD + GPU（CUDA / Vulkan），算子借 ggml
4. **IO** — 网络（proactor）+ 文件 + 跨进程共享内存

**差异点在"同一套底座"**：GPU 算完、网卡收包、fiber 让出，都回到同一个调度器、唤醒同一个 fiber，而非四个子系统各开线程池、各跑事件循环互相打架。HYLE 不是"又一个 hashmap/日志库"，而是让 计算/并发/内存/IO 无阻抗拼在一起的那层地基。

**刻意约束**：
- **不特化、不往应用层叠**。不做引擎、不做偏应用的东西。
- 集中在性能、贴近硬件。
- 不止重构 clay 那 2 万行，还要"加东西"撑野心——但**加的是 infra 柱子，不是应用**。

## 两条主轴

HYLE 在 clay 地基上扩张的方向是两条，不是往应用爬：

### 主轴 A — 极致性能（横向加宽、纵向做深）
五个焦点：
1. **SIMD** — 可移植 vector 抽象 + 运行时 dispatch + AVX512 + bf16
2. **CUDA 混合计算** — 自有异构计算原语（device/stream/memory 抽象）
3. **L1/L2 cache 命中率** — cache-conscious 数据布局，cache-line 对齐
4. **极致内存管理** — 分配器家族 + arena + NUMA/大页
5. **超越 std 的高性能容器** — flat/intrusive/lock-free

### 主轴 B — 并发执行模型 + IO
6. **统一调度** — 把「线程 / 进程 / 纤程 / 协程」统一成一张调度表（HYLE 最核心的野心）
7. **fiber** — clay 承诺却从未实现的核心，亲手补上
8. **网络 IO** — proactor 模型，与调度一体（fiber 遇 IO 让出）

## 模块划分

```
HYLE infra（通用、不特化、贴硬件）         [Tier：0=核心 1=兑现承诺 2=必要基质 3=便利]

核心计算/并发（论点本身）
├─ mem/       极致内存管理  [T0] ← 继承 clay 分配器家族+arena，修 5 个 bug，加 NUMA/大页/cache-line
├─ container/ 超越 std 容器 [T0] ← 继承 hashmap/carray/cstring + 核心 hasher，加 SwissTable SIMD/SmallVector/intrusive/lock-free
├─ sched/     统一调度【核心】[T0] ← 线程/进程/纤程/协程一张表：fiber 上下文 + per-core 无锁 run-queue + work-stealing（含 completion 脊）
├─ simd/      SIMD 抽象     [T1] ← 推倒重写：可移植 vector 类型 + 运行时 dispatch + AVX512 + bf16
├─ io/        网络 IO       [T1] ← 从零：proactor（io_uring/IOCP/epoll），与 sched 一体（fiber 遇 IO 让出）
├─ compute/   异构计算原语   [T1] ← device/stream/memory 抽象，先包装 ggml 不替算子
└─ graph/     并行任务编排   [T3] ← 继承 clay taskgraph，剥离特化，跑在 sched 之上

支撑层（为上面服务的基质，进主图因被广泛依赖）
├─ platform/  OS 抽象       [T2] ← thread/virtual_memory/dynlib/env/path/fs，按 clay *_api_t 细分；mem 要 vm、sched 要 thread、compute 要 dynlib
├─ time/      时间/定时器    [T2] ← 单调钟 + timer wheel；sched 的 deadline/超时、io 的 timeout 依赖（单列：sched 强依赖 + 有设计分量）
├─ error/     错误模型       [T2] ← status + std::expected + assert，万物用
└─ obs/       可观测        [T2/3] ← instrumentation hook（计数器/trace point）T2 第一天嵌热路径；sink/聚合 T3 缓做
```

> **不进 runtime 主图**：`ffi/`（TS/Python 语言绑定）是 build-time 工具、归"对外沟通"工具链（C-ABI 边界本身已是 T0）；`serialize/`、外围 hash(md5/uuid/base64)、`plugin/registry`、`random/`、`text/`、`compress/`、`config/` 是 T3，列举不画。Tier 详解见下方"模块分层"。

## 模块分层：哪些是核心，哪些可借可缓

不是按"重要不重要"排，而是按一个能操作的判据分：

> **判据：把这个模块换成现成库（std / folly / asio / ggml），HYLE 还是不是 HYLE？**
> 换了就不是 → 核心，自己造；换了照样是 → 可借、可缓。
> **补一刀**：是"数据结构"不等于"可换"。若①全 infra 都依赖它 ②边界上没 drop-in（C++ 类型过不了 C-ABI），那它即便是个容器也是地基（如下方 `container/`）。

| Tier | 模块 | 换成现成库会怎样 | 怎么做 |
|---|---|---|---|
| **0 核心** | `sched/`、completion 脊（io/gpu/fiber 完成都唤醒同一 fiber）、`mem/`、`container/`（flat hashmap + carray + 核心 hasher murmur/xxhash）、C-ABI 边界 | **HYLE 就没了** | 自己造、深拥有、测试/性能预算最高 |
| **1 兑现承诺** | `compute/`(GPU)、`io/`(网络)、`simd/` | 还像 HYLE，但"混合/极致性能"的卖点落空 | 自造，可借起步（compute 先包 ggml、simd 可参考 xsimd）|
| **2 必要基质** | atomics、`platform/`(thread/vm/dynlib/env/path/fs)、`time/`(单调钟+timer wheel)、`error/`(status+expected+assert)、`obs/` instrumentation hook、构建/ABI 骨架 | **照样是 HYLE**——只是没它编不过/跑不起来 | 优先用 std / 借，别在这层过度发明；但 `obs/` hook 要第一天嵌热路径 |
| **3 便利** | `obs/` 重机器（sink/格式化/聚合）、`serialize/`、外围 hash（md5/uuid/base64）、`plugin/registry`、`random/`、`text/`、`compress/`、`config/`、`graph/` | 在不在都是 HYLE | 能缓则缓、薄封装别自造 |

**为什么 `container/` 在 Tier 0 而 `simd/` 在 Tier 1**（同是"性能焦点"，凭什么分开）：
- `container/`（flat hashmap）是**地基 ∩ 差异**：clay 里 `string_buf`/`type_name`/`cstring`/`taskgraph` 等 13 处依赖它，抽掉整个 infra 裂开；且 C-ABI 边界上 abseil 的 C++ 类型过不去，没 drop-in。**核心 hasher 跟容器一起 Tier 0：运行期/hashmap/内容寻址用 XXH3（借 xxHash，替换 clay murmur），编译期类型 ID 留 murmur（constexpr 友好）**。
- `simd/` 是**差异但非地基**：是焦点，但 `string_buf`/`registry`/类型系统不依赖它才能工作（只有 compute/math 热路径用），抽掉 HYLE 仍能站立。故 Tier 1。

**为什么分这个**（不是排座次，是让每个决策有依据）：
- **造 vs 借**：Tier 0 必造；Tier 1 自造可借起步；Tier 2 优先借/std；Tier 3 薄封装或缓做。
- **力气往哪使**：性能预算、设计心力、测试深度按 0→3 递减。**别在 Tier 2 的 atomics 上手搓汇编，却把 Tier 0 的 completion 脊做含糊**——那是本末倒置。
- **scope 反向看门**：凡不强化"同一套底座"的，都是出 scope / 该上移到 L1 的候选（与下方"边界外"同一把尺子）。

注意：`time/`、`os/dynlib` 看着不起眼，但 `sched/` 的超时/deadline 依赖 `time/`、`compute/` 加载 GPU 后端依赖 `os/dynlib`——它们是**为 Tier 0 服务的仆从**，所以归 Tier 2 而非可缓的 Tier 3。`graph/` 虽在原 7 模块里，本质是建在 sched 之上的便利，归 Tier 3、贴近 L1。

## 完整性评估

**结论：scope 结构上完整——不缺顶层模块**（核心计算/并发 7 + 支撑层 4 已覆盖 计算/并发/内存/IO + 基质 全部）。但有**两个模块内的缺口**，补在既有模块里、不新增顶层：

1. **`sched/` 缺 fiber-aware 同步原语**（fiber mutex/condvar/channel/waitgroup/latch）。**必补**：`std::mutex` 会阻塞 worker 线程、破坏 M:N 调度——fiber 间同步必须"让出 fiber 而非阻塞线程"。归 sched/，补进 sched-design 待办。
2. **`simd/` 缺 dtype 层**（fp16/bf16/int8/fp8 ↔ fp32 转换）+ SIMD 化数学函数（exp/tanh/gelu/softmax）。AI 数值路径刚需，归 simd/。

其余（math 标量、quant）都能并进既有模块，无需新顶层。

## 规模（全模块深挖后确认）

> 早先"50-70k"估低了。**12 份设计全部深挖完后**，自底向上逐模块确认：sched/simd/io/compute 四个净新子系统每个 10-16k。下表为**生产代码**（不含测试）。**这已是基于完整设计的确认值，非粗估。**

| 模块 | Tier | 继承(clay) | 生产 LOC | 性质 |
|---|---|---|---|---|
| mem/ | 0 | allocator+arena ~2k | 6-10k | 改造+MemorySpace 组合+fiber栈+NUMA/大页+修5bug |
| container/ | 0 | hashmap/carray/cstring ~3k | 7-11k | 改造+SwissTable(concept模板)+SmallVector+intrusive+lock-free |
| sched/ | 0 | ~0（thread_pool 不可用）| 10-15k | **净新**：fcontext(6平台)+work-stealing+Waker+协程胶水+fiber-sync |
| C-ABI shim | 0 | — | 3-5k | 净新 |
| simd/ | 1 | ~0 | 10-16k | **净新**：scalable-friendly vector+多ISA+dispatch+dtype+数学函数 |
| io/ | 1 | ~0 | 9-14k | **净新**：proactor+4后端+net/file/shmem |
| compute/ | 1 | host_device 宏 | 10-16k | **净新**：device/stream/kernel+后端adapter+ggml bridge+MemorySpace 注册 |
| platform/ | 2 | os 层 ~3k | 5-8k | 改造继承 |
| time/ | 2 | ~0 | 1.5-2.5k | 净新：单调钟+时间轮 |
| error/ | 2 | status/result ~1k | 1.5-2.5k | 改造（迁 std::expected）|
| obs/ hook | 2 | log ~1k | 1.5-3k | 改造+埋点 |
| graph/ | 3 | taskgraph ~5k | 3-5k | 改造（剥特化+绑 sched）|
| 其余 T3 | 3 | 散落 ~3k | 8-12k | serialize/hash/plugin/random/text/compress/config |

| | 生产 LOC | 含测试(×1.6~1.8) |
|---|---|---|
| **核心(T0+T1+T2)** | **~65-100k** | ~105-185k |
| **完整(+T3)** | **~76-120k** | ~120-215k |

**对标校准**（非虚高）：Tokio（调度+io+sync+time）≈80k、Highway（仅 SIMD）≈100k、Seastar≈150k。HYLE 横跨 mem+container+sched+simd+io+compute+platform，**核心 65-100k 生产代码合理甚至偏保守**。

**一句话**：clay 给了 内存+容器+原子+编排骨架 四块地基（约 1/4），而 **sched+io+simd+compute 四根支柱从零造、每根 10-16k**——这才是 HYLE 真正体量，比"重构 2 万行"大一个量级（**核心约 3-4 倍、完整含测试约 5-7 倍**）。

## 边界外（明确不做）

> 核心原则：**HYLE 止于「原语 / 机制」，不碰「策略 / 协议 / 领域」。** socket 不 HTTP、序列化不 IDL、哈希不加密、fiber 不 actor、产出日志不聚合。守住这条线，HYLE 就不会从 infra 滑成框架。

### 分层模型：「在边界外」≠「不存在」，而是「上层/另一个项目做」

HYLE 是 L0 底座，协议/框架/应用都在它**之上**，由消费 HYLE 的上层项目实现。clay→slate/iron 就是活证据：

| 层 | 性质 | clay 世界（实证）| HYLE 世界 |
|---|---|---|---|
| **L0 infra** | 机制/原语 | clay = `slate/infra/`、`iron/infra/` | **HYLE 本身**（mem/sched/io/compute…）|
| **L1 中间件/框架** | 协议/框架 | （slate 渲染器直接用 clay，未单独拆）| HTTP/gRPC/actor/RPC——**消费 HYLE 的另一个项目** |
| **L2 应用** | 产品 | `slate/src/`（光追渲染器 kernel/bvh/scene/hydra）、`iron/src/services`（AI 推理服务）| agent(TS)、训练/推理(Python) |

clay 从未把光追逻辑、USD/Hydra 协议、service 端点吸进 `infra/`——都在 `src/`、在 infra 之上。**HYLE 继承这个纪律：底座永远是底座。** HYLE 的 C-ABI 边界（见 charter §6）正是 L0 与上层之间那道缝。

### 四类不做

**① 应用 / 领域层（L2，消费方的活）**：Tensor/autograd/层抽象、optimizer/loss/训练循环、tokenizer/sampler/推理循环/KV-cache、模型格式语义（GGUF/safetensors 怎么解读）、agent 逻辑/prompt、渲染。→ HYLE 给 `compute/` 的 device/stream/kernel 和 `mem/` 分配器，**怎么拼成神经网络是上层的事**。

**② 借不造（第三方已解决，自造=浪费或危险）**：GPU 算子实现（借 ggml）、压缩算法（借 lz4/zstd，只做封装）、**加密原语（AES/RSA/TLS 必借 vetted lib 如 BoringSSL，绝不自 roll——自造加密是安全事故）**、JSON 等解析器（借 nlohmann）。

**③ 比 infra 高一层的协议/框架（L1，另一个项目）**：HTTP/gRPC/WebSocket 协议、protobuf/IDL schema 编译器、RPC 框架/服务网格/服务发现、数据库/存储引擎/KV store、分布式共识（Raft/Paxos）、actor 框架/语言运行时、GUI/窗口系统。→ HYLE 给 socket+proactor+fiber+序列化原语，**协议/框架在 L1 拼**。

**④ 非库职责（工具链/运维）**：构建系统/包管理（用 CMake）、测试框架本身（用 GoogleTest）、日志聚合/监控后端（`obs/` 只产出结构化日志+metrics，采集/聚合/告警是运维栈）、GC（做分配器不做垃圾回收）。

### 容易模糊的边界线

| 在内（HYLE 给原语）| 在外（上层给策略/协议）|
|---|---|
| socket + proactor（io/net）| HTTP / gRPC 协议 |
| 二进制序列化原语 | IDL / schema 编译器 |
| 哈希函数（murmur/xxhash）| 加密（借 vetted lib）|
| DAG 执行器（graph）| 数据流编程语言/框架 |
| 分配器 + arena | GC / 自动内存管理 |
| fiber + 调度原语 | actor 模型 / 语言运行时 |
| 结构化日志产出（obs）| 日志聚合服务 |
| device/stream/kernel（compute）| Tensor/autograd/optimizer |

### 另外已明确不纳入

- DataModel（undo/序列化/协作）——clay 是空位，HYLE 也不纳入。
- 替换 ggml 算子 / 训练推理引擎本身——`compute/` 只做抽象层，包装而非替换。

## 待办（下一步）

完整资产负债表已出（见 `clay-asset-liability.md`）。下一步二选一：
1. **定路线与优先级**：8 个空位排依赖序（建议 sched 先行——io 的 proactor、taskgraph 的后端都挂在它上面），出重构路线图。
2. **先深挖某根支柱**：如 sched 的统一调度模型设计（fiber 上下文切换方案、协程与 fiber 如何统一、IO 让出机制），或 simd 抽象层选型。
