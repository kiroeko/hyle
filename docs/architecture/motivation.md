# HYLE 动机：为什么是现在，为谁做

> 这份回答一个绕不开的问题：HYLE 的祖先 clay 是 19 年的通用 infra，地基不错却停更了——「scope 不足以支撑野心」。HYLE 凭什么不是 clay 2.0？答案是：**clay 死的时候，那个能拉动「同一套底座」的 workload 还不到临界压力；现在到了。**

## clay 真正的死因，和现在的不同

clay 没死于技术差——内存/SIMD/线程池/**taskgraph 节点图执行引擎**都是真的（它本就服务 slate-render 的 CAD/渲染管线：`cycles` 渲染器 + `taskgraph` 参数化依赖重算）。它死于**那时的 workload 还没把「并发 + 内存 + 计算 + IO」四样同时压满**——CAD/渲染管线当年主要压 compute + graph 两根，io/并发是本地单机规模，"统一成一套底座"优雅但无溢价，于是 clay 成了博物馆。

**两类 workload 现在把四样同时推到临界**：① **AI-native 推理引擎**（serving + 批调度 + KV-cache + GPU/CPU kernel）；② **现代 CAD/BIM/CAE 工程计算管线**（参数化建模 + 渲染 + 数值仿真：云端协同/实时参数化把 io+并发拉起来，叠加它本来就重的 compute+graph；CAE 的 FEA/CFD 求解更把 compute/simd 压到极致）。这是"一套底座"从"没人需要"变成"刚需"的临界点——也是 HYLE 区别于 clay 的根本外部条件。不是我们更聪明，是时代变了，**且工程管线这一支正是 clay 血统里就在做、如今被重新激活的老本行**。

## 北极星：两个「四样全压」的具体 workload——推理引擎 ＋ CAD/BIM/CAE 工程管线（v1 主攻）

> **2026-07-03 演进**：① 早期把"蜂群 agent"列为最尖的刀——**太窄、会误导**(只压 fiber+io+channel,不压 compute/simd,把 HYLE 引向"agent 专用"而非通用底座),降为**消费者之一**。② 一度写成"推理引擎 + **通用加速库**"——但"通用加速库"太含糊,听着像 clay 2.0 摊大饼,**给不出验收标尺**。现改为**两把有血统、有压力的具体实刀**并列为锚:**类 llama.cpp 推理引擎** ＋ **CAD/BIM/CAE 工程计算管线**(参数化建模 + 渲染 + 数值仿真)。二者都**四样全压**,且合起来**压满 HYLE 全部五个支柱**——推理引擎拉满 io/sched/mem/compute,工程管线额外拉满 **graph**(参数化/仿真依赖重算,推理引擎不怎么碰),其 CAE 分支还把 compute/simd 从"kernel"推到"稀疏线代/迭代求解器"深度。"通用"不再靠形容词,靠"两个正交的具体 workload 都跑得动"来证明。

按契合度排序（前两个是并列北极星，其余是消费者/较弱场景）：

| workload | 压哪几根支柱 | std/拼装货为什么不行 | 契合度 |
|---|---|---|---|
| **推理引擎**(类 llama.cpp)| io(RESTful serving,**HTTP 标配**)+ sched(连续 batching)+ mem(paged KV-cache)+ compute/simd(GPU/CPU kernel)| serving 层与 compute 层胶水粘、阻抗失配;llama.cpp 得 vendor cpp-httplib+ggml 拼 | **北极星①**——四样全压、外部热度最高 |
| **CAD/BIM/CAE 工程管线**(参数化建模+渲染+仿真) | **graph**(参数化/仿真依赖 DAG 重算,clay taskgraph 老本行)+ compute/simd(几何核/网格/BVH/光追;**CAE 再加稀疏线代/迭代求解器/网格 stencil**)+ mem(大场景图/BVH/**刚度矩阵·网格**)+ io+sched(云端协同/**HPC 分布式求解**)| 参数化引擎(几何核)+ 求解器(FEA/CFD)+ 渲染器(光追)+ 协同/HPC 后端各用一套 runtime 拼,graph 重算与 GPU/求解 kernel 跨子系统打架 | **北极星②**——**有 clay 血统直接背书**(taskgraph/cycles),额外拉满 graph 支柱;CAE 把 compute/simd 压到最深 |
| **蜂群 agent** | fiber + io + channel（百万并发,几乎都在等 LLM/工具/网络）| `std::thread` 几千个就崩 | 强（fiber 最契合但只压半边;是引擎的**消费者**）|
| **训练引擎** | io+simd 数据管线 + host 编排 | 核心循环是 PyTorch/JAX/NCCL 地盘 | 较弱 |

**为什么"推理引擎"是对的锚**：llama.cpp 是活证据——一个真推理引擎必然**同时**要 ①高并发 RESTful serving(它 vendor 了 cpp-httplib)②GPU/CPU kernel(它就是 ggml)③KV-cache 内存管理 ④请求批调度。这四样正是 HYLE 四支柱。**HYLE 定位=做 llama.cpp 脚下那层它现在只能东拼西凑(httplib+ggml+自写线程)的统一底座**。RESTful API 因此是底座一等能力,不是可选 L1(HTTP 借-造抉择见 scope HTTP 专项)。

**厘清 ggml vs llama.cpp 的分层（定位关键，别混）**：**ggml = 张量计算库（算力）**——固定 ~90 个 AI 算子(matmul/softmax/rope…)的多后端(CPU/CUDA/Vulkan/Metal…)实现，只管"把张量算子跑上硬件"。**llama.cpp = 建在 ggml 之上的 LLM 推理引擎**——tokenizer/model-loader/KV-cache/sampler/连续批调度/RESTful server 都是它在 ggml 算力之上东拼西凑(httplib+自写线程池+自管 KV 内存)搭出来的。**HYLE 的定位不是替代 ggml（算子借它，compute-05），而是替代 llama.cpp 脚下那层拼装底座**：把 io(serving)/sched(批调度)/mem(KV-cache)/completion 统一成一套脊，让"类 llama.cpp 引擎"不再各子系统各自为政。**且 HYLE ⊋ llama.cpp 的底座**——它还要服务北极星② CAD/CAE，那需要 ggml **结构上给不了**的通用 GPU 计算（几何核/网格 stencil/CAE 稀疏求解器；ggml 只有固定 AI 算子、自定义 op 仅 CPU 路径，见 compute-design 发现 9），故 HYLE 自造 kernel 抽象(compute-04 NVRTC / 06 SPIR-V)补上 ggml 的能力边界外那块。**一句话：算子借 ggml，底座替 llama.cpp，通用 GPU 计算补 ggml 之不能。**

**为什么"CAD/BIM/CAE 管线"是对的第二锚**：它不是新赌注,是 **clay 的老本行被重新激活**——clay 的 `taskgraph` 插件(Task/schema/socket/kernel/graph_io)就是**节点图执行引擎**,宿主 slate-render 带 `cycles` 渲染器。现代工程软件的变化在于:①参数化建模/仿真流水=**依赖 DAG 增量重算**(改一个参数,只重算下游受影响节点/单元)——这是 HYLE **graph 支柱**的教科书消费者,而推理引擎几乎不碰 graph;②几何核/网格布尔/BVH 遍历/光追=**compute+simd 重载**,**CAE 的 FEA/CFD 求解更把 compute/simd 推到"稀疏线代 SpMV/迭代求解器/网格 stencil"的深度**(比 CAD 的几何 kernel 更吃向量化+带宽);③大装配体/场景图/BVH/**刚度矩阵·大网格**=**mem 大对象管理**;④云端协同/实时多人编辑/**HPC 分布式求解**把 **io+sched** 从"本地单机"拉到"高并发网络"。四样全压 + 独占拉满 graph,**正好补上推理引擎压不到的那根支柱**——两个北极星合起来才覆盖 HYLE 全部五支柱。CAE 与推理引擎在 compute 维度还高度同源(都是 GEMM/SpMV/stencil 类数值 kernel,共用 simd+GPU 后端)。

**为什么蜂群 agent 降为消费者**：一个 agent 大部分时间在**等**(LLM/工具/网络),要**百万个**跑在少数核上——`std::thread`(每个 1-2MB 栈)几千个到顶,**fiber(64KB-1MB 池化栈 + 10-20ns 切换)百万级是设计目标**。它最契合 fiber,但只压 io+sched 半边,不压 compute/graph,故是引擎的消费者而非独立锚。

这把 HYLE 从「通用 infra 赌有人用」重定义为「**为两个四样全压的具体 workload(AI 推理 + CAD/BIM/CAE 工程管线)做的统一底座**」。采用逻辑也变了：**不需要广泛采用,只需成为一个有说服力的引擎/管线的底座**——那个引擎/管线可以是自己的。

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

- **锚是两个具体 workload，不是一句形容词**。v1 以**推理引擎 ＋ CAD/BIM/CAE 工程管线**双锚——都四样全压、合起来压满全部五支柱（推理拉 io/sched/mem/compute，工程管线独占拉满 graph、CAE 再把 compute/simd 压到求解器深度）；蜂群 agent 是消费者之一（fiber 最契合但只压半边）。**双锚不是摊大饼**：它们正交（一个压 KV-cache/batching，一个压依赖 DAG 重算/几何核/稀疏求解），逼 HYLE 把"同一套底座"做到真通用，而非只喂一种负载。
- **顺序**：让北极星定义需求、HYLE 跟着砍，而非先做完通用全套再找用户。这也反过来锐化 scope——两锚下，io serving/sched 批调度/mem cache/compute kernel/**graph 依赖重算**都要压实。
- **采用现实**：作为「广泛采用的通用库」前景仍难（迁移成本/风险厌恶）；作为「一个具体引擎/管线（AI 推理或 CAD/BIM/CAE）的专用底座」完全合理。**定位决定一切**。

## 网络能力 + "超 nginx" 的诚实定位

推理引擎必然要 RESTful serving（llama.cpp 自己 vendor cpp-httplib），故 **HTTP/网络是 HYLE 的一等能力**，不是可选 L1。但做法是"借纯协议 parser + 自造集成层 over 自有 io_uring/fiber"（详见 [`hyle-scope.md` HTTP 专项](./hyle-scope.md)），不结合完整网络库（C++ 无中立异步契约，asio/beast/httplib 结合都不干净）、也不自研 RFC。

**"能不能建个超过 nginx 的 HTTP 服务器"——诚实拆**：
- **纯静态文件伺服** PK nginx：能打平、**极难超过**（nginx 20 年 sendfile/内存池调优贴内核天花板），**且是伪目标**——那不是 HYLE 的战场。
- **高并发动态 / AI serving**（HTTP serving 与 compute/GPU/KV-cache **同进程零阻抗协作**）：这里 HYLE 有**结构性架构优势**（io_uring 完成模型 + fiber co_await + 同进程），"超过 nginx 类方案"可信——但真正对手是"**nginx + AI 引擎用胶水拼装**"这个组合，HYLE 赢在**消除那层阻抗失配**，不是单挑纯 HTTP 吞吐。
- **诚实预期**：第一版大概率多项负载不如 nginx（一如 io echo 初版输裸 epoll 0.61x，迭代后才追平），靠迭代 + 老实 benchmark 追。**别把底座降维成"又一个 HTTP 服务器"去和 nginx 单挑**——那等于放弃唯一差异化。HYLE 的北极星不是"打败 nginx"，是"做 llama.cpp 那种引擎脚下、nginx+ggml 拼不出来的统一底座"。

## 一句话

clay 死于生不逢时——它服务的 CAD/渲染管线当年只压 compute+graph 两根，四样没到临界。HYLE 生于两类 workload 把「并发+内存+计算+IO」四样同时压满的时代，**以两个具体的四样全压 workload 为双北极星：类 llama.cpp 推理引擎 ＋ CAD/BIM/CAE 工程计算管线（参数化建模+渲染+数值仿真）**（前者拉满 io/sched/mem/compute，后者额外拉满 graph、CAE 再把 compute/simd 压到求解器深度，合起来覆盖全部五支柱；蜂群 agent 是消费者之一），靠「同一套底座 + 吃尽硬件独家加速」做到 std/拼装货摸不到的极致。不求广泛采用，求成为一个有说服力的引擎/管线的地基。
