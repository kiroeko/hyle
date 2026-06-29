# HYLE 动机：为什么是现在，为谁做

> 这份回答一个绕不开的问题：HYLE 的祖先 clay 是 19 年的通用 infra，地基不错却停更了——「scope 不足以支撑野心」。HYLE 凭什么不是 clay 2.0？答案是：**clay 死的时候，那个能拉动「同一套底座」的 workload 还不存在；现在存在了。**

## clay 真正的死因，和现在的不同

clay 没死于技术差——内存/SIMD/线程池都是真的。它死于**没有一个非它不可的上层应用拉动**。更深一层：**2019 年没有任何大众 workload 会在单进程里同时压满「并发 + 内存 + 计算 + IO」四样**。四样分开用现成库就够，"统一成一套底座"优雅但无溢价，于是 clay 成了博物馆。

**AI-native workload 是第一个四样全压的大众负载。** 这是"一套底座"从"没人需要"变成"刚需"的临界点——也是 HYLE 区别于 clay 的根本外部条件。不是我们更聪明，是时代变了。

## 北极星：蜂群 agent（v1 主攻）

三个 AI-native 场景按契合度排序，**蜂群 agent 是最尖的刀**：

| workload | 压哪几根支柱 | std/拼装货为什么不行 | 契合度 |
|---|---|---|---|
| **蜂群 agent** | fiber + io + channel（百万并发 agent，几乎都在等 LLM/工具/网络）| `std::thread` 几千个就崩；thread-per-agent 是灾难 | **最强** |
| **推理引擎** | io+sched+mem+compute 全压（高并发服务 + paged KV-cache + 批调度 + GPU kernel）| serving 层和 compute 层用胶水粘，阻抗失配 | 强（HYLE 当引擎下面的 infra，不与 vLLM 正面竞争）|
| **训练引擎** | io+simd 数据管线 + host 编排 | 核心循环是 PyTorch/JAX/NCCL 地盘 | 较弱 |

**为什么蜂群 agent 最对**：
- 一个 agent 大部分时间在**等**（LLM 返回、工具、网络），CPU 几乎闲着。
- 要**百万个**并发跑在少数核上。`std::thread`（每个 1-2MB 栈 + 内核切换）几千个到顶；**fiber（64KB-1MB 池化栈 + 10-20ns 切换）百万级是设计目标**。
- agent 间要轻量消息（channel）、共享状态（shmem）、IO 让出（proactor）——**全是 HYLE 的 Tier 0/1**。
- 2026 没有一个 C++ infra 是冲着「百万 IO-bound agent」设计的——**真空地带**。

这把 HYLE 从「通用 infra 赌有人用」重定义为「**为 clay 死后才出现的 AI-native 负载（尤其蜂群 agent）做的 infra**」。采用逻辑也变了：**不需要广泛采用，只需成为一个有说服力的 AI-native 引擎的底座**——那个引擎可以是自己的。

## 比 Seastar 好在哪

Seastar 是参考标杆：per-core + DPDK/SPDK 内核旁路 IO、completion 驱动事件循环，io/mem 侧做得极好，**该学**。但要分清学什么、不学什么（核对真实 seastar 源码）：

- **Seastar 是 share-nothing + future/continuation + 线程钉核 + 无 work-stealing**。它的 "shard-per-core" 本质就是 share-nothing 调度——**和 HYLE 的 work-stealing 是对立哲学**。
- **HYLE 选 work-stealing，不选 share-nothing**：北极星是"百万 IO-bound agent、唤醒突发且不可均分"，share-nothing 的负载不均在此是致命伤（热 shard 无法卸载、agent 无法干净分区）。
- **Seastar 完全没有异构加速维度**（实测全库 0 个 `.cu`、无 GPU/AMX/特化注册），且是 2019 前设计、诞生在 AI-native 负载之前。

"做得比它好" = **借它的 per-core io/mem 局部性 + completion 驱动事件循环 + reactor_backend 后端抽象;不借它的 share-nothing 调度（HYLE 用 work-stealing）;补上它缺的异构加速维度（能力门控 + GPU 统一进 completion 脊）**。后两者正是 AI-native 负载需要、Seastar 时代不存在的。(注:seastar 的 xcpu_freelist 评估后未借——它依赖 share-nothing+线程钉死,HYLE 的通用生命周期固定池给不出该契约;跨核 free 的正确性 HYLE 用地址路由原子 push 已解决,详见 mem-design。)

## 极致 = 吃尽各家硬件独家加速（HYLE 的硬要求）

"为 AI 大放异彩"的前提是**真做到极致**，而极致 = 吃尽各家独家加速（Intel AMX/VNNI/DSA、Blackwell FP4/tcgen05、Vulkan cooperative matrix）。HYLE 的框架必须优雅支持这点：

- **编译期优先**：默认按架构重新编译（llama.cpp 式 `-march=native`/feature flag），运行期零 dispatch、零间接——这是极致主路径。
- **运行期一次冷跳兜底**：仅发布单一二进制吃异构硬件时才要。
- **组合优于包含**：每个特化是独立组合单元，加一个 = 改 1 文件、0 调用点、0 热路径开销。
- 机制细节见 [`coding-conventions.md` → 能力 dispatch](./coding-conventions.md)；厂商 kernel 多数借 ggml，HYLE 负责"插得进、选得对、有地板"。

## 诚实边界（聚焦，不是唱衰）

- **仍要选一个北极星**，否则又摊薄成「三场景都做、都 70 分」。v1 以蜂群 agent 为锚——最差异化、最契合 fiber+io、对 GPU 依赖最轻（compute 可后接）。
- **顺序**：让北极星定义需求、HYLE 跟着砍，而非先做完通用全套再找用户。这也反过来锐化 scope——agent 北极星下，fiber/io/channel/mem 绝不妥协，simd/compute 可稍后。
- **采用现实**：作为「广泛采用的通用库」前景仍难（迁移成本/风险厌恶）；作为「一个具体 AI-native 引擎的专用底座」完全合理。**定位决定一切**。

## 一句话

clay 死于生不逢时——它要的 workload 还没来。HYLE 生于 AI-native 负载把「并发+内存+计算+IO」四样同时压满的时代，**以蜂群 agent 为北极星**，靠「同一套底座 + 吃尽硬件独家加速」做到 std/拼装货摸不到的极致。不求广泛采用，求成为一个有说服力的 AI-native 引擎的地基。
