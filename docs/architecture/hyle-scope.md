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
2. ✅ **`simd/` dtype 层已落地（simd-05/06）**:fp16/bf16 ↔ fp32 转换(F16C 硬件路径)+ SIMD 数学核(exp/log/tanh,借 Highway/xsimd vendored)。int8/fp8 + gelu/softmax 登记后续广度。

其余（math 标量、quant）都能并进既有模块,无需新顶层。

## 规模

> HYLE 自有 **~22-30k 生产 / ~40-55k 含测试**(2026-07-02 第二轮下修,io 破土后按实测密度分锚外推);含第三方 vendored 项目总足迹 **~200-300k+**，由 ggml 主导。LOC 不是 SOTA 标尺——靠精简自有 + 借成熟件，不靠堆行数。

**完成时投影（已完成 = 9,932 生产 cloc 实测,其余按已完成模块的实测密度分锚外推）:**

| 部分 | 生产 LOC | 依据 / 置信度 |
|---|---|---|
| 已完成（foundation + simd f32 核 + sched 全支线 + io 收割层） | **~9.9k** | cloc 实测(硬锚) |
| simd 广度补全（int 量化类型 + 更多 op + AVX512 启用） | +~2-3k | 有 f32 核密度锚,中 |
| sched 后段（协程胶水/取消/优雅关闭/栈 guard page） | +~1-1.5k | 核心已 1k,后段更薄,中 |
| io/ net·socket·file（proactor+双后端已 done,net/ 未做） | +~2-3.5k | 参照 platform 1.3k,低(net/ 未开工) |
| compute/（device/stream + ggml bridge + Vulkan） | +~3-5k | 借 ggml 薄层,很低(0 行无锚) |
| graph + mem-05/06 + C-ABI shim + T3（serialize/hash） | +~4-6k | 多借 simdjson/flatbuffers,低 |
| **HYLE 自有 生产合计** | **~22-30k**(中值 26k) | 第二轮下修;三大支柱落地即复核 |
| **HYLE 自有 含测试**（test/prod≈0.65 实测,16.3k 含测试/9.9k 生产） | **~40-55k** | — |

**第三方（vendored，计入项目足迹）——大头，由 ggml 主导：**

| 库 | 行数 | 状态 |
|---|---|---|
| mimalloc | ~15k | 已 vendored |
| xxhash | ~6k | 已 vendored |
| fcontext | ~0.8k | 已 vendored（仅 3 汇编） |
| highway/xsimd（数学核子集） | ~10-30k | 已 vendored（simd-06，无条件默认编进生产） |
| simdutf | ~10k | 已 vendored（utf8 SIMD，无条件默认） |
| liburing | ~12k | 将借（io） |
| **ggml** | **100k+** | **将借（compute）——单独就压过 HYLE 自有全部** |
| simdjson/flatbuffers/capnproto | tens of k | 将借（serialize） |
| **第三方合计** | **~150-250k+** | 由 ggml 主导 |

**当前 LOC 低 ≠ 玩具，也 ≠ 已精简到位**——两条都要看：
- **非玩具**：已实现部分零 stub/TODO、测试密（simd 全 65536 fp16 硬件逐位对拍、红队揪真 bug），玩具撑不住。
- **scope 仍会长回**：simd 实测 2.3k 是 Highway 数值子集（4 lane 类型 + ~25 op + 2 ISA），非功能对等（Highway 12 类型 + ~200 op + 6 ISA）。量化类型/更多 op 是显式延期，补齐后涨到 ~5-8k。**别把"广度未铺开"误读成"已精简完毕"。**

**实现粒度**：上表是体量、**不是步骤**。落地为**约 55 个里程碑**（milestones/README 为准），每个 1-4k 行、可独立编/测/基准。6 个"阶段 P0-P6"是依赖层拓扑序，不是 6 步。先走 walking skeleton（纵切各层最小片）打通调度脊、验最高风险，再逐里程碑加厚。

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

### ⚠️ 边界待复核（2026-07-03 参考库审计后，用户提出）

> 上面的"边界外/四类不做"是**项目早期写的**,本身可能"漏点"——把某些真该做的原语误判成 out-of-scope。**这些边界不当已决,留作后续讨论。** 一次参考库审计(拿 tokio/seastar/folly/liburing/asio/libevent/tracy/oneTBB 逐条对照)已触发若干修正与待决:

- **TLS**:已从"隐性排除"改判为 **C9 真能力**(io-08)。scope §153"借不造"本就写明 TLS 借 BoringSSL——所以它是**借入的原语**不是 out-of-scope 协议,之前没登记是漏,不是有意不做。
- **序列化原语**:边界表说"二进制序列化原语"IN、"IDL/schema 编译器"OUT——原则清楚,但 **close-02 无设计、无接口形状**,是个"看着像计划的占位"。**待决**:借 flatbuffers/capnproto 的 builder+reader 当原语落地,还是把 wire-format 全推 L1。需拍板。
- **数据并行层**(parallel_for/reduce/scan over range,oneTBB/oneDPL):不同于 graph 的 DAG 编排,是 sched+simd 的天然消费者(大 buffer compute kernel)。**不被现边界排除**(是原语不是协议),但归 graph 还是 compute 未定。待与序列化一起复核。
- **原则(暂)成立,但也待审**:HYLE 止于原语/机制,不碰策略/协议/领域——审计的真缺口(Semaphore/JoinHandle/select/spawn_blocking/multishot/异步DNS/TLS)确实都在原语侧,没越界 L1。**但"明确 out-of-scope"的每一条(HTTP/gRPC/RPC 框架/IDL 编译器/log 聚合/actor/DataModel)都不当已决**——早期负空间可能把"其实是通用原语"的东西误判出界。**下一轮要逐条重审整个"边界外/四类不做/已明确不纳入",不是只审这里点到的几个。**

### 待复核的"不纳入"项(原标"已明确",现降级=后面都要审)

> ⚠️ 用户明确:以下两条(及上文"四类不做")都**不靠谱、后面都要审**,别当已决。

- DataModel（undo/序列化/协作）——clay 是空位,曾据此判 HYLE 也不纳入。**存疑**:undo/序列化里可能藏通用原语(如序列化已在复核)。待审。
- 替换 ggml 算子 / 训练推理引擎本身——`compute/` 只做抽象层,包装而非替换。**这条相对稳**(借 ggml 是明确决策),但仍随 compute 支柱落地复核。

## 进度

路线图见 [`roadmap.md`](./roadmap.md);逐里程碑实时状态见 [`milestones/README.md`](../milestones/README.md)。当前:mem/container/sched/simd/platform/time/util/error/obs 主体已落地,io/compute/graph 三大净新增支柱未开工。
