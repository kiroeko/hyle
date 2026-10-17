# 推理引擎对基础库的要求

HYLE 是推理引擎的基础加速库。真实引擎用于检查基础能力是否齐全、公共接口是否自然，
以及优化是否改善实际延迟、吞吐、能耗和内存。
当前目标是多个受支持系统上的单机 CPU、CPU＋一张 GPU、单 GPU，不包含跨机器分布式推理。

通用机制应能描述所有权、寿命、失败、并发和复杂度；模型格式、tokenizer、采样、KV 策略和
请求政策属于引擎。真实消费者暴露问题时直接调整相应接口和实现，不需要重新运行全库审查。

## 基础模块与引擎边界

| 领域 | HYLE 应提供的机制 | 应留在相邻 owner / 推理引擎的策略与语义 |
|---|---|---|
| core | 编译器合同、稳定的编译期 cache-layout、极小的无所有权 primitive、产品身份 | 模型常量、token/layer 元数据、为单个 kernel 定制的公共入口 |
| error | 统一 `Status/Result`、context、OS/backend 错误映射 | 模型格式错误分类、对用户的恢复与重试策略 |
| platform | CPU/ISA/topology、线程亲和、NUMA、VM、动态库和 OS 机制 | 所选路线使用哪些核、模型/KV 的业务放置 |
| mem | host allocator/resource、arena/pool、对齐、大页、file/shared mapping 与 NUMA placement 机制 | KV cell/page 表、prefix cache、sequence copy/evict/defrag 策略 |
| time | 单调时钟、deadline、timer queue | 请求 SLO、batch 等待窗口和超时策略 |
| container/util | 通用连续、flat、queue、bitset、registry 等数据结构与工具 | 词表、token 表、模型 registry 语义和模型专用索引 |
| sched | task/fiber/coroutine、waker、取消、work stealing、blocking handoff | continuous batching、admission、优先级、公平与多模型仲裁策略 |
| simd | 通用 lane、dtype conversion、bitpack 和可复用向量原语 | 与具体量化块绑定的 fused dequant-dot/GEMM 和模型布局 |
| compute | device/context/queue/completion、storage bridge、backend/provider、直接算子/kernel、量化计算与 repack | 跨节点 topology/placement/liveness 归 Graph；模型、KV、采样和服务策略归引擎 |
| io | 文件、socket、proactor、异步读写和取消机制 | GGUF 等格式语义、HTTP endpoint、路由、请求 schema 与 serving 策略 |
| obs | 低成本 log/trace/counter 机制及 sink 接缝 | 指标命名、告警、采样策略和产品 dashboard |
| graph | 模型无关计算 DAG、shape/layout、provider partition、fusion/copy/sync、liveness/alias、Executable/Invocation | 模型 layer 构图、KV/batch/sampling/循环停止归引擎；task runtime 和通用 parallel algorithms 不因同名进入 Graph |
| build/package | 组件闭包、依赖锁、ISA/backend 组合、source/static/shared 交付 | 模型资产发布、服务部署和应用配置 |

两个边界尤其不得混淆：

- device storage artifact、queue 和算子执行属于 Compute；host allocator 不能假装统一管理第三方
  backend 的 device memory。借 ggml 时继续遵守
  [`borrowing-ggml-memory-model`](./borrowing-ggml-memory-model.md) 的所有权边界。
- KV-cache 是推理状态管理策略。mem 可以提供适合它的 backing、pool、NUMA 和 lifetime 原语，
  但不得公开 `KVCache`、sequence 或 token 形状。

## 正常推理旅程

下表帮助查漏，按实际修改选择相关用例；不要求每次修改运行完整矩阵。

| Workload | 必须覆盖的形状 | 首要指标 | 主要拉动 |
|---|---|---|---|
| 模型加载 | 冷/暖文件页；mapped/read；成功、损坏、OOM；适用时单/多 NUMA node | load-to-ready、page fault、峰值 RSS、复制字节、失败清理 | platform/mem/io/error |
| Prefill | 短/中/长 prompt；小/大 batch 与 ubatch；CPU 和可用 GPU backend | prompt tokens/s、TTFT、峰值临时内存、带宽/算力利用率 | compute/simd/mem |
| Decode | 单序列与并行序列；浅/深 context；短/长生成 | generation tokens/s、TPOT/ITL 的中位与 tail、内存带宽 | compute/mem/sched |
| Recurrent/hybrid decode | 固定/自适应循环深度；共享计算块；同批不同 active depth；循环取消 | 质量—时间/能耗曲线、loops/token、bytes/token、状态峰值、batch 利用率 | compute/mem/sched/obs |
| 连续批处理 | 混合 prompt/output 长度；稳态与突发到达；不同并发度 | requests/s、tokens/s、TTFT/ITL tail、公平性、deferred/backpressure | sched/mem/compute |
| KV 生命周期 replay | 会话加入/退出、共享前缀、fork/copy、回收、容量不足 | bytes/token、峰值 committed、碎片、分配/回收 tail、泄漏 | 引擎策略消费 mem 原语 |
| Backend 资源交接 | 单机 CPU、CPU＋一张 GPU、单 GPU；输入输出传输、跨 Provider 资源与完成边界 | copy bytes、同步次数、overlap、CPU busy、device idle | compute/sched/platform |
| Graph compile/replay | one-op、多 region、shape specialization、并发 Invocation 与重复 decode | compile time、steady submit 税、workspace、copy/sync、cache hit、fallback | graph/compute/mem |
| 取消与关闭 | 排队、prefill、decode 中取消；backend failure；有界 drain | cancel latency、迟到 completion、泄漏、drain 时间、错误保真 | error/time/sched/compute |
| 构建与交付 | source embed、installed static/shared；baseline、native/exact ISA；可选 backend | 可构建/运行、组件闭包、artifact 依赖、可复现性 | build/package + 被测模块 |


完整 Qwen3.5-2B CUDA 文本引擎是当前集成锚，但不能用一个模型证明底座覆盖完整。
还需要检查变长与跨页增长、其他 dtype/布局、MoE、recurrent 状态、分块 prefill 和多请求组合。
MTP 验证后的状态恢复及图片 encoder/projector 另行补全，底座现有接口不应封死这些组合。

## 公平比较与实际验证

与 llama.cpp 等参考比较时固定模型与权重、精度、输入 token、上下文、输出数量、设备、线程和
计时边界。分开报告加载与冷准备、prefill、decode、显存和主存；质量与数值先成立。
能耗使用实际测量并注明范围，TDP 不能冒充整机功耗。

验证正常路径、错误、OOM、并发 invocation、取消、迟到完成和关闭。
真机计算、编译安装与模拟结果分别报告；未运行不写通过。
性能实验先检查计时分辨率、噪声和漂移，确认存在的输项继续定位。
公共接口和布局的改动需要能解释其对真实消费者的作用，不能由单次微基准决定。

当前八项工作见[工作主线](../developer/current-work.md)，
后端差异见[四路线](compute-routes.md)，组合机制见[CPU/GPU 设计](compute-next-stage.md)。
