# HYLE 动机：为什么做，怎样避免再做偏

## 问题

高性能 C++ 应用通常同时需要内存管理、数据结构、SIMD、调度、I/O、异构 backend 和诊断。
每一项都有成熟的专门库，但消费方仍要自行解决它们之间的 allocator、ownership、错误、取消、
completion、线程和构建边界。局部优秀不自动形成可组合的系统。

HYLE 希望提供这层基础加速底座：让消费者能够按模块选择能力，同时共享清楚的资源与执行合同，
并在需要时贴近 OS 和硬件，而不是退化成只包装标准库的 façade。

## 为什么必须有真实消费者

项目早期按孤立 benchmark 发展，反复出现三类问题：

- 为了某个计时姿势暴露不自然的类型或入口；
- 把代码位置、地址分布和计时器噪声误判为产品差异；
- 因为没有端到端任务，既会遗漏成熟库已有的重要能力，也会实现没人需要的东西。

因此 HYLE 不再以“模块功能很多”或“微基准多数获胜”作为方向。每项公共能力必须回到用户任务、
所有权、正常数据分布和组合方式；微基准只解释机制。

## 首要北极星：本地量化推理引擎

一个真实推理引擎会同时压到 HYLE 的关键边界：模型加载与 mapped I/O、权重和临时内存、量化
数据结构与 SIMD、CPU/GPU 执行、请求调度、取消、观测和三种发布形状。它能把底层优化映射到
TTFT、token latency、throughput、峰值内存和复制量，而不是停在一次 allocator 或 atomic 的纳秒数。

HYLE 不以复制 llama.cpp 为目标。llama.cpp/ggml 是成熟的行为、能力和性能参照，也是可提取真实
workload 的来源；未来 HYLE-based 引擎是否更快，要由相同模型、量化、输出、backend、内存与
调度合同下的端到端证据回答。

### 2026-08 的外部事实：推理正在成为全栈系统问题

AI 的模型、硬件和部署形状仍在快速变化，HYLE 不能把某一代 GPU 或某一套 kernel 当作永久前提。
2026-08，OpenAI 公布首代 inference-only Jalapeño 的实测结果：在公开 InferenceX 的
GPT-OSS 120B、DeepSeek R1 670B 与 Kimi K2.5 1T workload 上，相对当时商用 GB200/GB300 系统，
报告了约 1.5--1.9 倍的 peak mixed tokens/s/kW、1.7--3.6 倍更低的端到端延迟，并在低 TBT
工作点取得更高的单用户 token rate。它仍是厂商自测、尚在量产资格阶段的推理芯片，不是训练
替代品，也不能把某个 matched-latency 点的巨大比例外推为所有 workload 的通用倍数；但它提供了
一个重要、可复核的方向证据：现代推理的优势来自模型、kernel、compiler、memory、network、
placement、scheduler 与 rack 一起设计，而不只来自峰值 FLOPS。

OpenAI 对机制的解释同样约束 HYLE 的需求锚：prefill 更偏 compute，decode 更偏 memory
bandwidth；权重和 KV state 的局部性、跨 chip 通信和不同阶段的资源平衡决定实际利用率。
Jalapeño 是封闭的第一方数据中心平台，HYLE 不以复刻其硬件或与它直接对拍为目标；HYLE 的机会是
让开放模型在用户能够取得的 CPU、GPU、NPU 与统一内存系统上，也能用清楚的机制表达相同问题，
并让上层引擎做模型特化。

因此本项目把以下内容视为长期背景，而不是一次新闻驱动的 feature list：

- 优先减少权重、activation、KV 与 completion 的搬运、同步和跨拓扑通信；
- 同时覆盖 prefill、decode、低并发交互和高并发吞吐，不用一个姿势代签另一姿势；
- 用 TTFT、TBT/ITL、tokens/s/user、总 tokens/s、joules/token、峰值 host/device memory 与复制量
  描述端到端 Pareto 面；package TDP、实测 wall power 和单 chip/rack 口径必须分开；
- 允许 compute/引擎按模型家族提供 kernel、placement 和执行计划特化，但基础模块只接收通过
  语义擦除测试的可复用机制；
- 把可预测、机器可审计的合同/codegen/workload 留给人与 AI 共同优化，但选定 block 的提升不能
  冒充整模型或整系统收益。

方向输入以 OpenAI 的
[首轮测量及完整口径](https://openai.com/index/jalapeno-first-results/)和
[芯片/系统联合设计公告](https://openai.com/index/openai-broadcom-jalapeno-inference-chip/)为准；
项目裁决仍必须由 HYLE 自己在可获得硬件上的可复现证据完成。

同一天出现的另一条事实覆盖了不同市场：Apple 公布的 M5 Ultra 是可由开发者直接购买的桌面
统一内存系统，采用 quad-die UltraFusion，最高 512 GB unified memory、1.2 TB/s memory
bandwidth，die 间互连超过 4.4 TB/s。它不等同于 Jalapeño 的 inference ASIC，也没有由这些规格
自动推出的 HYLE/LLM 性能结论；但它使“在一台本地机器内完整容纳并运行数百 billion 参数级
量化模型”从边缘设想变成需要认真服务的产品形状。容量决定模型能否运行，持续带宽、inter-die
locality、kernel 利用率与同步决定实际 tokens/s。

这为 HYLE 增加了一个长期且可获得的消费者平台方向：同一套本地引擎既要利用离散 GPU，也要能
利用 CPU/GPU/NPU 共享大容量 backing 的统一内存机器。统一地址/内存池不代表自动零复制或自动
高效；buffer storage mode、resource residency、page/memory pressure、CPU/GPU handoff、cache
coherency 与跨 die placement 仍须由 platform/mem/compute 和引擎明确表达。公共基础 API 不能写成
Apple 专有形状，但 workload 矩阵必须包含“模型超过普通离散显存、能够完整驻留统一内存”的真实
档位。规格来源见 Apple 的
[M6 与 M5 Ultra 公告](https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/)。

### 2026-09 的模型架构信号：推理深度可能与输出长度解耦

截至 2026-09-03，OpenAI 已确认 Astra 是即将发布、相对 GPT-5.6 Sol 能用更少输出 token 完成更强
网络安全任务的模型，并表示完整 system card 将在发布时提供；官方材料尚未公开网络结构。因此，外界
关于 Astra 使用 recurrent depth 的说法只能作为**待确认的架构报道**，不能写成 HYLE 已验证的产品
事实。事实边界以 OpenAI 的
[Astra 官方说明](https://openai.com/index/path-to-astra/)为准。

recurrent-depth Transformer 的一般机制已有公开研究：前置层把输入映射到隐空间，一个共享参数的
计算块在深度方向重复若干次，最后由输出层映射到 token。推理时增加循环次数，可以增加隐空间计算而
不同比例增加参数量或必须先生成更多思维 token。这不是 2026 年才出现的概念：
[Universal Transformer](https://arxiv.org/abs/1807.03819) 已探索跨深度共享与动态停止，
[Huginn recurrent-depth 研究](https://arxiv.org/abs/2502.05171)进一步把可变推理深度扩展到
3.5B 参数、800B 训练 token 的概念验证模型。若同类机制在前沿产品中成立，新的系统轴将不只是
“更大模型 / 更多输出 token / 更多候选搜索”，还包括“同一隐状态经过多少轮共享计算”。

中文评论作者**风音**从雅可布森的语内、语际与符际翻译区分出发，指出把模型高维内部计算转成自然
语言解释本身是一种有损的符际转译，而常见 CoT 监控主要处理模型已经外化成语言的部分；若更多推理
留在 latent workspace，自然语言记录不能自动等同于完整内部过程。该观点适合作为可观察性与安全边界
的风险提示，不是 Astra 架构或效率数据的第一手证据。出处：
[风音，知乎回答](https://www.zhihu.com/question/2078440102733337893/answer/2078609210632413968)；
转载与使用遵守原作者标注的授权条件。

这一方向可能提高参数复用和局部性、减少为了“思考更久”而产生的中间 token、词表投影与部分 KV
增长，但并不自动提高效率：共享块若不能驻留在更近的存储中，每一轮仍会读取权重；循环引入串行延迟，
量化误差可能累积，按 token 选择不同深度还会破坏 batch 规整性。latent 推理也不会让可解释性问题
消失，只会要求行为监控、状态探针和授权边界承认自然语言 CoT 不是完整内部状态。

HYLE 将 recurrent/hybrid decode 作为 Compute/推理引擎的一个**泛化压力形状**，而不是新闻驱动的
feature list 或重开已关闭底层模块的理由。执行层应允许已编译计算区域低开销重复、循环不变量常驻、
loop-carried state 复用和动态 active-set 整理；具体模型的循环次数、停止策略与 latent-state 语义仍由
推理引擎拥有。只有真实模型或至少两个独立成熟消费者证明通用形状后，相应机制才可通过语义擦除进入
公共基础 API。具体 workload 与模块边界见
[推理引擎北极星](inference-engine-north-star.md#31-固定深度之外的执行形状)。

模型格式、tokenizer、sampling、KV/prefix-cache 策略、continuous batching 和 serving policy
属于推理引擎。HYLE 只下沉去掉模型语义后仍自然的通用机制。详细边界见
[inference engine north star](inference-engine-north-star.md)。

## 泛化留出：工程计算

参数化建模、仿真、几何与数值计算能提供与推理不同的数据和执行形状：稳定节点与依赖 DAG、
稀疏/网格访问、长任务、交互式重算和不同精度。它当前是泛化留出，而不是要求 HYLE 同时制造
一套 CAD/CAE 产品。

这条留出用于发现 API 是否只适合 dense tensor、某种量化块或某个 backend。若一个“通用”类型
在去掉 token/KV 名称后仍只能服务单一模型，它应留在引擎；若工程任务证明相同机制成立，公共
基础形状才更可信。

## 与 clay 的关系

HYLE 继承 clay 中仍有价值的高性能思想和部分实现资产，但不以签名、文件名或 LOC 覆盖为目标。
clay 的历史价值是提示成熟问题与失败模式；每项资产仍需按当前 C++、真实消费者、依赖、正确性
和性能标准重新裁决，可以保留、现代化、借库替代或删除。

## 成功标准

HYLE 成功不是因为仓库达到某个行数，也不是因为每个操作都比所有库快。成功意味着：

1. 推理消费者能用自然 C++ API 组合所需机制，不再重复发明底层；
2. 常见基础需求有明确路线，模块边界和依赖无环且可独立组装；
3. 关键正常 workload 达到合理复杂度，并相对强 peer 没有未解释的 material 劣势；
4. correctness、OOM、取消、并发、平台 fallback 和资源成本可复现；
5. source/static/shared 三种 C++ 交付都能在独立工程中使用；
6. 文档与机器合同让后来维护者知道为什么、做过什么、怎样验证和用户怎样使用；
7. 模块达到证书后能停止关注，只被明确的新事实重新打开。

## 主要风险

- 没有真实消费者时范围无限增长；
- 为追逐微小 benchmark 差异破坏产品形状；
- 复制成熟库能力却没有维护预算；
- 底层模块过大、依赖环和跨模块同义实现；
- compute/graph 等未稳定层过早冻结；
- 把 shared C++ library 误当跨语言稳定 ABI；
- 文档保存大量旧结论却没有可信度和当前状态。

当前的应对是：推理需求锚、工程留出、逐模块收口审查、仪器资格、独立红队、四类文档和机器
合同。具体方法见 [module closure methodology](../developer/review/module-closure-methodology.md)。
