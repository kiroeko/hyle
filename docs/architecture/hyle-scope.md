# HYLE Scope 定义

> 本文档定义 HYLE 的范围边界：包含什么、不包含什么、有多大。资产盘点依据见 [`clay-asset-liability.md`](./clay-asset-liability.md)。

## HYLE 是什么

**一个 C++ 基础设施库**（编译成带 C-ABI 的共享库），给上层（AI 推理引擎、CAD/BIM/CAE 工程计算管线、训练框架、agent）当地基。把四样东西做在**同一套底座**上：

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
2. **CUDA 混合计算** — 自有异构计算原语（device/stream/buffer 抽象）
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
├─ compute/   异构计算原语   [T1] ← device/stream/buffer 抽象，先包装 ggml 不替算子
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

**结论：scope 结构上完整——不缺顶层模块**（核心计算/并发 7 + 支撑层 4 已覆盖 计算/并发/内存/IO + 基质 全部）。曾识别**两个模块内的缺口**,补在既有模块里、不新增顶层——**两者现均已落地**:

1. ✅ **`sched/` fiber-aware 同步原语已落地（sched-05）**:CV 地基 + Mutex(futex)/WaitGroup/Latch/Channel,双模 fiber/裸线程,对标 marl 全线追平/反超。`std::mutex` 会阻塞 worker 破坏 M:N 的隐患由"短锁用 futex、长等待用 fiber-aware CV"分工解决。
2. ✅ **`simd/` dtype/math 层已落地并完成双轨收口（simd-05/06/20/22）**：fp16/bf16 ↔ fp32 硬件路径、fp8、HYLE 自有数学核与组合核；普通 Runtime、整库 exact、用户 TU native 三种部署面已统一，产品零 Highway/xsimd。

其余（math 标量、quant）都能并进既有模块,无需新顶层。

## 规模

> **信源**：体量以 `cloc` 实测为准；能力进度以 [`../milestones/README.md`](../milestones/README.md) 为准。
> **口径**：生产 = `src` + `include` 的 code 列（不含 tests/bench/docs、不含 third_party）。
> **2026-07-24 锚定**：旧投影「完成时生产 ~28–40k / 中值 34k」**已作废**——当前生产已 **78.0k**，单 simd **37.5k**。当前判断见 [`../milestones/pre-release-readiness.md`](../milestones/pre-release-readiness.md)；更早数字只作历史快照。

### 当前实测（2026-07-24）

| 口径 | code | 说明 |
|---|---:|---|
| **生产 `src`+`include`** | **78,023** | 库本体；不含测试 |
| tests | **35,059** | 由同语言口径 113,082 - 78,023 得出 |
| bench | **15,012** | |
| docs | **22,239** | Markdown + CSV |
| 自有工程（`--vcs=git --exclude-dir=third_party,.deps`） | **156,142** | 含测/文档/脚本 |
| git 全跟踪（含 in-tree third_party） | **238,969** | 含 simdutf/mimalloc/fcontext 等 |
| 再加 ggml 等大依赖足迹 | **~200–350k+** | 由 ggml 主导，不计入「自有生产」 |

**生产按模块（src+include，约）：**

| 模块 | code | 占生产 |
|---|---:|---:|
| **simd** | **~37.5k** | **~48%** |
| io | ~9.0k | ~12% |
| compute | ~4.1k | ~5% |
| sched | ~5.1k | ~7% |
| container | ~4.9k | ~6% |
| platform | ~5.7k | ~7% |
| mem | ~4.3k | ~6% |
| graph | ~0.6k | |
| 其余基质及顶层/C ABI 头 | ~6.4k | ~8% |

### 完成时投影（2026-07-19 重估，取代 28–40k）

| 剧本 | 生产 code | 条件 |
|---|---:|---|
| **A 纪律收口（主预期）** | **~75–90k**（中值 **~82.5k**） | 八底层模块冻结；sched/io 收口；上层 C ABI/install；compute/graph 必要能力尾 |
| **B 能力做满仍守边界** | **~85–105k** | compute 后端/算子尾更满；graph-02 完整；发布工具链完整 |
| **C 失控上限（避免）** | **120k+** | 继续镜像 Highway contrib / 自研算子帝国 / 协议层吸入 infra |

**含测试粗乘**：生产 ×1.4–1.6 → A 约 **95–125k**、B 约 **120–160k**（tests only）。
**自有工程（+bench+docs）**：A 约 **130–160k**、B 约 **160–200k**。

**剩余增量直觉（相对今日 77.8k 生产，非承诺）：**

| 方向 | 再长（生产） |
|---|---:|
| 基础模块收口复审 | 原则上小增量；mem 功能族已封闭，只剩有界 API/perf/认证任务 |
| sched/io 强收口 + C ABI | +1～4k |
| compute 能力尾 | +2～10k（主弹性） |
| graph-02 后两步/后续真实消费者 | +1～5k |
| close-01 install/export/FFI | +1～4k |

### 第三方（足迹，非自有生产 KPI）

| 库 | 约行数 | 状态 |
|---|---|---|
| mimalloc | ~15k | in-tree vendored |
| simdutf | ~60k（整树 cloc） | 可选 UTF-8 薄借，门控宏 |
| fcontext | ~0.8k + asm | vendored |
| xxhash 等 | 小 | vendored / 薄封 |
| highway/xsimd | 0 产品 | 仅 bench 外部 pin |
| **ggml** | **100k+** | compute 算子供应商——单独可压过 HYLE 自有 |
| **第三方合计（含 ggml）** | **~150–300k+** | 由 ggml 主导 |

### 读数纪律

- **LOC 不是 SOTA 标尺**；完成度看能力清单 + 真测，不看是否碰到 70k/100k 整数。
- **里程碑计数 ≠ 完成度**（颗粒不等重）。
- **git 全跟踪 238.4k ≠ 自写 238.4k**（含 third_party）。
- 早期「simd 完成 ~5–8k / 全库完成 ~34k」是 **广度未展开时的低估**，不是当前目标。
- 实现粒度：里程碑表见 milestones/README；拓扑序见 roadmap。

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

### 边界重审（2026-07-03，参考库对照 + 北极星检验）

> **背景**：上面的"四类不做"是项目早期写的,用户指出"都不靠谱、后面都要审"——早期负空间可能把**通用原语**误判成 out-of-scope。本节逐条重审。
>
> **判据不是"参考库有没有",而是两问**:① 它是**原语/机制**还是**策略/协议/领域**?(scope 铁律)② **双北极星(推理引擎 ＋ CAD/BIM/CAE 工程管线底座,见 motivation)需不需要它作为可复用底座件?** 两问都"是原语 + 北极星需要"→ `PULL IN`(补漏);是原语但成熟库已解决→ `BORROW`(借不造);真策略/协议/领域→ `KEEP OUT`;拿不准→ `DISCUSS`(待你拍)。

| 原判"不做"项 | 重审 | 理由（北极星视角）|
|---|---|---|
| **① 应用/领域层**:Tensor/autograd/optimizer/训练循环、tokenizer/sampler/KV-cache、模型格式语义、agent 逻辑/prompt;**光追/光栅化算法、几何核(布尔/NURBS/网格)、BIM 语义(IFC/构件);CAE 的物理模型/本构方程/网格划分器/求解器策略(FEA/CFD 具体算法)** | ✅ **KEEP OUT** | 真 L2 领域。北极星是引擎/管线的**底座**不是引擎本身;KV-cache/sampler 是推理引擎的活,光追 kernel/几何算法/IFC 是 CAD/BIM 引擎的活,本构方程/湍流模型/网格划分是 CAE 求解器的活。HYLE 给 compute/simd/mem/**graph 编排**,拼成渲染器/建模器/求解器是上层事。**线画在"依赖 DAG 重算/BVH·矩阵内存布局/几何·数值 SIMD kernel(SpMV/stencil) 调度"(底座,HYLE 给) vs "光追积分器/布尔运算/IFC 解析/有限元离散化·湍流模型"(应用,消费方写)**。判对。 |
| **② 加密/TLS**(借 BoringSSL)| 🔄 **BORROW→已登记** | 原判"借不造"对,但**漏了登记**——公网 agent/服务必需。已改判 C9/io-08(借 shim,不自 roll)。 |
| **② GPU 算子/压缩/JSON**(借 ggml/lz4/simdjson)| ✅ **BORROW**(判对)| 成熟库解决,自造=浪费/危险。北极星需要但"借"就够。JSON 借 simdjson 已明确。 |
| **③ HTTP/1.1 协议层(client+server 骨架)** | 🔄 **改判 PULL IN(有条件)** | **北极星重排后此条翻案**:类 llama.cpp 推理引擎**必然要 RESTful serving**(llama.cpp 自己 vendor cpp-httplib)——AI 服务器不可能不提供 REST。故 HTTP over fiber/proactor 是底座一等能力,不是 L1。**但成本大、且不能抄 asio**→见下"HTTP 专项:成本与借-造抉择",借-造未定前不动手。 |
| **③ HTTP/2·gRPC·WebSocket** | 🔄 **HTTP/2 PULL IN；gRPC/WebSocket KEEP OUT** | **io-11 已落地**：借固定版本 nghttp2 的 HPACK/frame/stream/flow-control 核，自写 allocator/caps、HYLE coroutine transport 与 transport-neutral C codec；不引入第二 event loop。gRPC 的 protobuf/IDL 与 WebSocket 应用协议仍更靠 L1，继续留给消费方。 |
| **③ 路由/中间件/handler DSL/业务端点** | ✅ **KEEP OUT** | 这才是 HTTP 框架/应用(L1/L2)。llama.cpp 的 `/v1/chat` 端点是 L2;它底下的 httplib 那层才是 HYLE 该给的。**线画在"协议骨架 vs 路由业务"之间**。 |
| **③ IDL/schema 编译器** | ✅ **KEEP OUT** | 编译器是工具链;但**序列化运行时原语**(builder/reader)是另一回事→ DISCUSS-2。 |
| **③ RPC 框架/服务网格/服务发现** | ✅ **KEEP OUT** | 真框架(L1)。agent 用得上但那是消费方拼。 |
| **③ 数据库/存储引擎/KV store** | 🟡 **KEEP OUT(存疑)** | 引擎是 L1。但 agent 要**持久化状态/记忆**——嵌入式 KV(如借 RocksDB/lmdb 封装)是否算底座件?倾向 KEEP OUT(借现成),记此存疑。 |
| **③ 分布式共识(Raft/Paxos)** | ✅ **KEEP OUT** | 真分布式协议(L1)。 |
| **③ actor 框架/语言运行时** | ✅ **KEEP OUT** | actor 是 fiber+channel 之上的**模型**(L1)。HYLE 给 fiber+channel 原语,actor 上层拼。判对。 |
| **③ GUI/窗口系统** | ✅ **KEEP OUT** | 领域,与北极星无关。 |
| **④ 构建/包管理/测试框架**(CMake/GoogleTest)| ✅ **KEEP OUT** | 工具链,判对。 |
| **④ 日志聚合/监控后端** | 🔄 **部分 PULL IN** | **边界已收口（2026-07-19）**：日志/trace 的“产出原语”属于 L0，聚合、轮转、上传、查询、告警和 viewer 属于运维栈，继续 KEEP OUT。日志是运行期 level gate + 同步结构化 sink；trace 是另一条编译期面，默认 OFF 时参数不求值/零后端依赖，ON 时接单一共享 Tracy client。真实 hook 已覆盖 mem/sched/io/compute（含 fiber 身份、线程名、zone/counter/memory），不再只是 fiber 单点。C/FFI 有稳定 runtime trace seam，但函数 no-op 不宣称零指令；纳秒热路径用原生宏。架构、生命周期和三平台验收见 [`obs-design.md`](./obs-design.md) / [`../milestones/obs-module-closure.md`](../milestones/obs-module-closure.md)。 |
| **④ GC** | ✅ **KEEP OUT** | 做分配器不做 GC,判对。 |
| **DataModel(undo/序列化/协作)** | 🟡 **拆开看** | undo/协作=L2 领域(KEEP OUT);内含的**序列化**已单独在 DISCUSS-2。原"整块不纳入"太粗。 |
| **替换 ggml 算子/训练推理引擎** | ✅ **KEEP OUT** | compute 只做抽象层,判对(借 ggml 是明确决策)。 |

**新发现的真缺口(原负空间根本没提到,审计补)**——均已登记为能力/里程碑,属 `PULL IN`:
- **Semaphore / RwLock / oneshot / broadcast·watch**(C7/sched-09)、**JoinHandle + select/when_all**(C7/sched-10)、**spawn_blocking**(C8/sched-11)、**带超时 io**(C2/io-06)、**multishot+buffer ring**(C1/io-07)、**异步 DNS + signal**(C10/io-09)。全在**原语侧**、全是双北极星(推理引擎服务底座 + CAD/BIM/CAE 协同·HPC 后端)刚需。
- **graph 支柱(taskgraph 绑 sched)**:CAD/BIM/CAE 北极星把它从"P6 收口件"提为**一等消费者驱动的支柱**——参数化/仿真依赖 DAG 的增量重算(改一参数只重算下游脏节点/单元)是工程管线核心热路径,graph 节点跑在 work-stealing sched 上、几何/求解 kernel 走 compute。**已在 roadmap 轴 C 登记为 C12(参数化/仿真依赖 DAG 增量重算)驱动 graph-01/02**,P6 graph 行改标"北极星②驱动、非纯收口"——悬空点已闭环。

**待你拍板(DISCUSS)**:
- **DISCUSS-1 HTTP —— ✅ 已定案**(见下"HTTP/网络能力:最终结论"):三层分工=IO 自有 / 解析借纯 parser(picohttpparser 起步)/ 集成自写 / 路由业务留上层。结合完整库(asio/beast/httplib)三条死路已论证排除(C++ 无中立异步契约)。
- **DISCUSS-2 序列化运行时原语**:借 flatbuffers/capnproto 的 builder+reader 当原语(不含 schema 编译器),还是全推 L1?agent 间消息/持久化状态要序列化。
- **DISCUSS-3 数据并行层**(parallel_for/reduce/scan):原语无疑,有消费者(大 buffer kernel)故非投机。**✅ 归属已定案(2026-07-10,见 [`graph-design.md`](./graph-design.md) §2)=归 graph(graph-02)**:compute-design 立铁律"数据并行铺多核靠 graph-02 fork-join、host kernel 单线程内联",graph-02 归 compute 会破 compute"不起线程池"边界;且消费者(CAE 网格单元并行/场量归约)与 graph-01 仿真依赖重算同源。graph-02 消费 sched(workers)+ compute/simd(每块 kernel),是编排者非算力本体。

> **重审结论**:早期"四类不做"的**大框对**(领域/框架/工具链确实在外),但犯了两类错——① **漏登记该借的原语**(TLS、trace 埋点),已补;② **把原语和它上面的策略捆在一起判出界**(序列化 vs IDL、trace 产出 vs 聚合、HTTP 协议骨架 vs 路由业务)。修正方式=**拆细粒度**:原语侧拉进来,策略/框架侧留在外。

### HTTP / 网络能力:最终结论(2026-07-03 定案)

> 一路问题收敛:自己实现所有 RFC 成本太高、完全不管网络又不对(推理引擎必需 REST)、结合现成完整库(asio/beast/httplib)在 C++ 里不干净。定案如下。

**定位**:HTTP/网络是 HYLE **一等能力**(类 llama.cpp 推理引擎必然要 RESTful serving),但**只做到"协议骨架 over 自有 io",不做路由/框架/业务**。

**做法——三层分工(业界自有运行时的库一致选择:seastar/tokio/folly)**:

| 层 | 谁做 | 说明 |
|---|---|---|
| 路由 / 中间件 / handler / 业务端点(`/v1/chat`)| **上层(L1/L2)** | 真框架/应用,不是底座 |
| **HTTP 集成层**:读循环→喂 parser→出 Request→组 Response→write | **HYLE 自写(~1-2k/协议)** | ★HYLE 的价值 + 性能所在:HTTP over fiber/proactor,非阻塞线程池 |
| **协议解析器**(请求行/头/chunked/HPACK 等纯状态机)| **借纯 parser** | picohttpparser/llhttp(HTTP/1.1)、nghttp2(HTTP/2);零 IO 零线程,vendor 一个文件 |
| TCP / TLS / 超时 / DNS + proactor + fiber | **HYLE 自有** | 性能命根子,io-04✅ / io-06 / io-08 / io-09 |

**为什么不结合现成完整网络库(调研实证,三条死路)**:
1. **HYLE 用 asio/libevent** → 它们的协议层焊死自己的 `io_context`/event_base,只能 `poll_one()`/`EVLOOP_NONBLOCK` **单步驱动**=**双重 poll**(asio/libevent 自己还在 epoll/io_uring,HYLE proactor 也在)→ 直接废掉 io_uring 优势。cpp-httplib 更是阻塞线程池,和 fiber 冲突。
2. **C-ABI 让完整库反向用 HYLE 的 io** → 也不行。障碍是**运行时控制反转**(谁拥有事件循环、完成怎么通知),不是调用约定;完整库编译期就焊死"等我自己的循环",C++ 库又几乎没有"可插拔 IO 后端"插槽,C-ABI 插头再标准也无处可插(除非改库源码=fork 重写,比借 parser 还贵)。
3. **根因:C++ 没有中立异步契约**。Rust 能干净结合(hyper 只依赖 `AsyncRead/AsyncWrite` trait,tokio-uring 换后端 hyper 零改)是因为 `Future/Poll/Waker` 是语言级契约;C++ 无等价物(asio concept 绑 executor、seastar 绑 future/reactor、P2300 sender/receiver 还没库用)。**所以"用别人的完整协议库 + 自己的运行时"在 C++ 里本质做不到干净结合。**

> **活路只有一条**(seastar/tokio/folly 全走):**借纯 parser(不碰 IO)+ 自造集成层(接 HYLE io_uring/fiber)**。seastar 是最直接先例——自有 reactor→自己写 HTTP 栈(~3.5k 行)但解析交给 Ragel 生成;不手写协议。C-ABI 留给其正确用途:**把 HYLE 整体(含自集成 HTTP)对外暴露给上层引擎/TS/Py**,不是内部结合网络库。

**性能怎么保证(为什么这条路能快)**:性能不来自"用了某个快库",来自**网络字节全程待在 HYLE 自己的 io_uring/shard-per-core 上,不引入第二套运行时**——① 无双重 poll(fd 只被 HYLE proactor 管);② 无阻塞线程池(连接跑 fiber,10-20ns 切换,百万连接是设计目标);③ parser 是纯用户态状态机,零 syscall,且是被千锤百炼的借来件;④ 集成层就是 `co_await stream.read/write` + 喂 parser,和 echo bench 跑出 shard-per-core 扩展性的同一条路径。**即 HTTP 性能 = HYLE 自有 io_uring/fiber 性能 + 一个借来的解析器,无额外运行时税。**

**成本/现状**:HTTP/1.1 server+client 已借 picohttpparser + 自有集成层完成；
HTTP/2 已按相同边界借 nghttp2 协议核完成 h2c/TLS、C++ connection 与 C codec。
WebSocket、gRPC（依赖 protobuf/IDL，更靠 L1）仍后置于范围外。

**vs nginx 的诚实标尺**(避免跑偏,详见 motivation):
- **纯静态文件伺服** PK nginx:能打平、极难超过(nginx 20 年 sendfile/内存池调优贴内核天花板),**且是伪目标**——那不是 HYLE 的战场,赢了没意义。
- **高并发动态 / AI serving**(与 compute/GPU 同进程零阻抗协作):这里 HYLE 有**结构性架构优势**(io_uring + fiber + 同进程),"超过 nginx 类方案"可信——但真正对手是"**nginx + AI 引擎用胶水拼装**"的组合,HYLE 赢在消除那层阻抗失配,不是单挑纯 HTTP。
- **诚实预期**:第一版大概率多项负载不如 nginx(同 io echo 初版输裸 epoll 0.61x),靠迭代 + 老实 benchmark 追。**别把底座降维成"又一个 HTTP 服务器"去和 nginx 单挑**——那放弃了唯一差异化。

**parser 选型**:HTTP/1.1 起步用 **picohttpparser**(极简/最快/无依赖,~1 文件,推理引擎 REST 够用),需要更全特性再评 llhttp;HTTP/2 用 nghttp2。均属 vendor 纯 parser。

## 进度

路线图见 [`roadmap.md`](./roadmap.md);逐里程碑实时状态见
[`milestones/README.md`](../milestones/README.md)。当前：八个底层模块完成强收口；
sched/io 已完成同等级模块审计并带公开性能尾进入预发布冻结；compute 的 CPU/CUDA/Vulkan/ggml 主干和
graph-01/graph-02 step-1 已实现，二者都不是 stub，但仍有产品能力尾。
