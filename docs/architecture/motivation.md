# HYLE 动机：为什么是现在，为谁做

> 这份回答一个绕不开的问题：HYLE 的祖先 clay 是 19 年的通用 infra，地基不错却停更了——「scope 不足以支撑野心」。HYLE 凭什么不是 clay 2.0？答案是：**clay 死的时候，那个能拉动「同一套底座」的 workload 还不存在；现在存在了。**

## clay 真正的死因，和现在的不同

clay 没死于技术差——内存/SIMD/线程池都是真的。它死于**没有一个非它不可的上层应用拉动**。更深一层：**2019 年没有任何大众 workload 会在单进程里同时压满「并发 + 内存 + 计算 + IO」四样**。四样分开用现成库就够，"统一成一套底座"优雅但无溢价，于是 clay 成了博物馆。

**AI-native workload 是第一个四样全压的大众负载。** 这是"一套底座"从"没人需要"变成"刚需"的临界点——也是 HYLE 区别于 clay 的根本外部条件。不是我们更聪明，是时代变了。

## 北极星：类 llama.cpp 推理引擎 + 通用加速库（v1 主攻）

> **2026-07-03 重排(用户纠偏)**：早期把"蜂群 agent"列为最尖的刀——**太窄、会误导**(它只压 fiber+io+channel,不压 compute/simd,且把 HYLE 引向"agent 专用"而非"通用底座")。真正第一位的是 **类 llama.cpp 的推理引擎 + 通用加速库**：它**四样全压**(io serving + sched 批调度 + mem paged-cache + compute/simd kernel),是"同一套底座吃满四样"最直接的证明。蜂群 agent 降为**消费者之一**,不再是唯一锚。

三个 AI-native 场景按契合度排序：

| workload | 压哪几根支柱 | std/拼装货为什么不行 | 契合度 |
|---|---|---|---|
| **推理引擎 / 加速库**(类 llama.cpp)| **四样全压**：io(RESTful serving,**HTTP 是标配**)+ sched(批调度/连续 batching)+ mem(paged KV-cache)+ compute/simd(GPU/CPU kernel)| serving 层和 compute 层用胶水粘、阻抗失配;llama.cpp 自己得 vendor cpp-httplib+ggml 拼 | **最强**——四样全压=底座价值最直接 |
| **蜂群 agent** | fiber + io + channel（百万并发 agent，几乎都在等 LLM/工具/网络）| `std::thread` 几千个就崩；thread-per-agent 是灾难 | 强（fiber 最契合,但只压半边;是引擎的**消费者**）|
| **训练引擎** | io+simd 数据管线 + host 编排 | 核心循环是 PyTorch/JAX/NCCL 地盘 | 较弱 |

**为什么"推理引擎/加速库"是对的锚**：llama.cpp 是活证据——一个真推理引擎必然**同时**要 ①高并发 RESTful serving(它 vendor 了 cpp-httplib)②GPU/CPU kernel(它就是 ggml)③KV-cache 内存管理 ④请求批调度。这四样正是 HYLE 四支柱。**HYLE 定位=做 llama.cpp 脚下那层它现在只能东拼西凑(httplib+ggml+自写线程)的统一底座**。**RESTful API 因此是底座一等能力,不是可选 L1**——AI 服务器不可能不提供 REST。(HTTP 实现成本/借-造抉择见 scope 的 HTTP 专项评估。)

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

- **仍要选一个北极星**，否则又摊薄成「三场景都做、都 70 分」。v1 以**类 llama.cpp 推理引擎 + 通用加速库**为锚——四样全压、最能证明"同一套底座"的价值；蜂群 agent 是其消费者之一（fiber 最契合但只压半边）。
- **顺序**：让北极星定义需求、HYLE 跟着砍，而非先做完通用全套再找用户。这也反过来锐化 scope——引擎北极星下，io serving/sched 批调度/mem cache/compute kernel 都要压实。
- **采用现实**：作为「广泛采用的通用库」前景仍难（迁移成本/风险厌恶）；作为「一个具体 AI-native 引擎的专用底座」完全合理。**定位决定一切**。

## 网络能力 + "超 nginx" 的诚实定位

推理引擎必然要 RESTful serving（llama.cpp 自己 vendor cpp-httplib），故 **HTTP/网络是 HYLE 的一等能力**，不是可选 L1。但做法是"借纯协议 parser + 自造集成层 over 自有 io_uring/fiber"（详见 [`hyle-scope.md` HTTP 专项](./hyle-scope.md)），不结合完整网络库（C++ 无中立异步契约，asio/beast/httplib 结合都不干净）、也不自研 RFC。

**"能不能建个超过 nginx 的 HTTP 服务器"——诚实拆**：
- **纯静态文件伺服** PK nginx：能打平、**极难超过**（nginx 20 年 sendfile/内存池调优贴内核天花板），**且是伪目标**——那不是 HYLE 的战场。
- **高并发动态 / AI serving**（HTTP serving 与 compute/GPU/KV-cache **同进程零阻抗协作**）：这里 HYLE 有**结构性架构优势**（io_uring 完成模型 + fiber co_await + 同进程），"超过 nginx 类方案"可信——但真正对手是"**nginx + AI 引擎用胶水拼装**"这个组合，HYLE 赢在**消除那层阻抗失配**，不是单挑纯 HTTP 吞吐。
- **诚实预期**：第一版大概率多项负载不如 nginx（一如 io echo 初版输裸 epoll 0.61x，迭代后才追平），靠迭代 + 老实 benchmark 追。**别把底座降维成"又一个 HTTP 服务器"去和 nginx 单挑**——那等于放弃唯一差异化。HYLE 的北极星不是"打败 nginx"，是"做 llama.cpp 那种引擎脚下、nginx+ggml 拼不出来的统一底座"。

## 一句话

clay 死于生不逢时——它要的 workload 还没来。HYLE 生于 AI-native 负载把「并发+内存+计算+IO」四样同时压满的时代，**以类 llama.cpp 推理引擎 + 通用加速库为北极星**（蜂群 agent 是其消费者之一），靠「同一套底座 + 吃尽硬件独家加速」做到 std/拼装货摸不到的极致。不求广泛采用，求成为一个有说服力的 AI-native 引擎的地基。
