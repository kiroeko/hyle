# 推理引擎北极星：用真实消费者约束基础库

> 本文定义本地量化推理引擎如何作为 HYLE 的需求与性能锚。它不把推理引擎并入 HYLE，
> 也不替代各模块设计、收口方法或性能统计标准。模块审查仍遵守
> [`module-closure-methodology`](../developer/review/module-closure-methodology.md)，性能判定仍遵守
> [`benchmark-evidence-standard`](../developer/performance/benchmark-evidence-standard.md)。

## 1. 定位

HYLE 是推理引擎下面的基础加速库；推理引擎是第一个可执行、可观测、能同时压到内存、调度、
计算和 IO 的真实消费者。北极星的作用是回答三件事：

1. 哪个用户任务要求某项基础能力存在；
2. 该能力应属于哪个模块，正常数据、生命周期和并发形状是什么；
3. 底层改动是否改善真实推理结果，而不只是改善一个人工微基准。

北极星不是 API 命名空间。`token`、`layer`、`head`、`expert`、`sequence`、`KV`、模型格式和
采样策略不能因为推理引擎需要它们就下沉到基础模块。HYLE 提供可复用机制；推理引擎拥有模型
语义和产品策略。

一项候选能力进入公共基础 API 前必须通过**语义擦除测试**：去掉模型、token 和服务名称后，
它仍能以明确 ownership、lifetime、concurrency、failure 和复杂度描述一个通用任务。否则它应
留在推理引擎、compute adapter 或私有 seam。

## 2. 需求等级与 API 准入

每项候选能力先登记到版本化需求账本，并且只能处于以下一种等级：

| 等级 | 定义 | 允许动作 |
|---|---|---|
| `ENGINE_BLOCKER` | 没有它，最小真实引擎不能正确完成模型加载、prefill 或 decode | 在正确归属模块实现并优先验证 |
| `COMPETITIVE` | 已有真实路径，且会实质影响延迟、吞吐、内存或可靠性 | 以消费者 workload 和强 peer 驱动优化 |
| `GENERALIZATION_GATE` | 用于防止只适配一个模型、量化格式、ISA 或 backend | 进入留出验收，不自动扩大公开面 |
| `INTERNAL_PROBE` | 只有一个尚未稳定的内部消费者 | 先保持私有，允许实验和删除 |
| `DEFERRED` | 没有当前消费者，只有未来猜测 | 不实现、不增加公开类型或常态成本 |

公共 API 还必须至少满足以下一项：

- 有两个独立的正常消费者；或
- 有一个真实消费者，并且同一模型无关形状在多个成熟库中反复出现。

只有一个消费者、形状仍会随模型或 backend 改变时，默认留在私有 seam。成熟库有某个 feature
只是查漏信号，不是准入依据。Native FFI 同样只由已命名的 foreign-runtime 任务驱动；C++ 的
source/static/shared 交付本身不要求制造 FFI parity。

每项保留能力必须有一张可核对合同卡：消费者任务、需求等级、owner 模块、公开或私有、数据
与生命周期、并发与失败、正常规模、强 peer、计费边界、成功指标、负空间。缺任一字段不得进入
实现阶段。

## 3. 基础模块与推理引擎的边界

| 领域 | HYLE 应提供的机制 | 应留在 compute / 推理引擎的策略与语义 |
|---|---|---|
| core | 编译器合同、稳定的编译期 cache-layout、极小的无所有权 primitive、产品身份 | 模型常量、token/layer 元数据、为单个 kernel 定制的公共入口 |
| error | 统一 `Status/Result`、context、OS/backend 错误映射 | 模型格式错误分类、对用户的恢复与重试策略 |
| platform | CPU/ISA/topology、线程亲和、NUMA、VM、mapped file、动态库和 OS 机制 | 用哪些核、如何放权重/KV、模型级 offload 策略 |
| mem | host allocator/resource、arena/pool、对齐、大页、NUMA placement 机制 | KV cell/page 表、prefix cache、sequence copy/evict/defrag 策略 |
| time | 单调时钟、deadline、timer queue | 请求 SLO、batch 等待窗口和超时策略 |
| container/util | 通用连续、flat、queue、bitset、registry 等数据结构与工具 | 词表、token 表、模型 registry 语义和模型专用索引 |
| sched | task/fiber/coroutine、waker、取消、work stealing、blocking handoff | continuous batching、admission、优先级、公平与多模型仲裁策略 |
| simd | 通用 lane、dtype conversion、bitpack 和可复用向量原语 | 与具体量化块绑定的 fused dequant-dot/GEMM 和模型布局 |
| compute | device/buffer/stream/event、backend adapter、算子/kernel 执行、量化算子与 repack | 模型拓扑、layer 构图、KV 语义、采样和推理循环 |
| io | 文件、socket、proactor、异步读写和取消机制 | GGUF 等格式语义、HTTP endpoint、路由、请求 schema 与 serving 策略 |
| obs | 低成本 log/trace/counter 机制及 sink 接缝 | 指标命名、告警、采样策略和产品 dashboard |
| graph | 有独立消费者时保留通用任务 DAG/数据并行机制 | 推理 operator graph 的构建、融合和执行计划归 compute/引擎；不能因同名而下沉 |
| build/package | 组件闭包、依赖锁、ISA/backend 组合、source/static/shared 交付 | 模型资产发布、服务部署和应用配置 |

两个边界尤其不得混淆：

- device buffer、stream 和算子执行属于 compute；host allocator 不能假装统一管理第三方 backend
  的 device memory。借 ggml 时继续遵守
  [`borrowing-ggml-memory-model`](./borrowing-ggml-memory-model.md) 的所有权边界。
- KV-cache 是推理状态管理策略。mem 可以提供适合它的 backing、pool、NUMA 和 lifetime 原语，
  但不得公开 `KVCache`、sequence 或 token 形状。

## 4. 版本化 workload 矩阵

稳定文档只规定维度和资格；具体模型 hash、量化格式、prompt/token 序列、规模、线程数、backend、
编译参数和机器写入版本化 workload manifest。候选冻结后不得改 manifest 来迎合结果。

| Workload | 必须覆盖的形状 | 首要指标 | 主要拉动 |
|---|---|---|---|
| 模型加载 | 冷/暖文件页；mapped/read；成功、损坏、OOM；适用时单/多 NUMA node | load-to-ready、page fault、峰值 RSS、复制字节、失败清理 | platform/mem/io/error |
| Prefill | 短/中/长 prompt；小/大 batch 与 ubatch；CPU 和可用 GPU backend | prompt tokens/s、TTFT、峰值临时内存、带宽/算力利用率 | compute/simd/mem |
| Decode | 单序列与并行序列；浅/深 context；短/长生成 | generation tokens/s、TPOT/ITL 的中位与 tail、内存带宽 | compute/mem/sched |
| 连续批处理 | 混合 prompt/output 长度；稳态与突发到达；不同并发度 | requests/s、tokens/s、TTFT/ITL tail、公平性、deferred/backpressure | sched/mem/compute |
| KV 生命周期 replay | 会话加入/退出、共享前缀、fork/copy、回收、容量不足 | bytes/token、峰值 committed、碎片、分配/回收 tail、泄漏 | 引擎策略消费 mem 原语 |
| Backend handoff | CPU-only、部分/全部 offload；同步与可重叠传输 | copy bytes、同步次数、overlap、CPU busy、device idle | compute/sched/platform |
| 取消与关闭 | 排队、prefill、decode 中取消；backend failure；有界 drain | cancel latency、迟到 completion、泄漏、drain 时间、错误保真 | error/time/sched/compute |
| 构建与交付 | source embed、installed static/shared；baseline、native/exact ISA；可选 backend | 可构建/运行、组件闭包、artifact 依赖、可复现性 | build/package + 被测模块 |

模块微基准不能代替这张表。它只能隔离 allocator、原子、dispatch、数据结构或 codegen 的根因；
最终产品结论必须回到使用正常公开入口或正常内部 seam 的消费者 workload。

每个适用 workload 还必须记录能耗与拓扑口径：package/TDP 只可作明确标注的归一化输入，不能冒充
整机实测功耗；能够取得 wall/rack power 时报告 joules/token。低延迟交互与峰值 batch 吞吐分别画
Pareto 点，至少包含 TTFT、TBT/ITL、tokens/s/user 和 aggregate tokens/s，不能用峰值吞吐掩盖
单用户停顿，也不能把某一 matched-latency 点的比例外推到整条曲线。权重、KV 和 activation 的
host/device/NUMA placement、复制字节、跨 device 通信及 idle time 属于必记 workload 身份。
Backend 矩阵至少区分离散 device memory 与大容量 unified-memory 系统；后者加入一个超过普通离散
显存、但可完整驻留统一内存的模型档位，同时记录 resident/committed memory、memory pressure、
CPU/GPU handoff 和跨 die 行为。统一地址空间不得被记作“天然零复制”或“天然单 NUMA domain”。

## 5. 泛化与留出协议

开发集和资格留出集必须分离：

- 开发集允许用于 profiler、参数选择和实现迭代；
- 留出集在候选冻结前只登记合同，冻结后才运行；
- 留出集至少改变模型家族、计算阶段、数据规模或 backend 中的两个维度；
- 首个 dense 量化模型不能是唯一证据；资格集至少包含另一种 dense 形状，并在能力适用时加入
  MoE、sliding-window 或 recurrent/hybrid 形状；
- 至少覆盖一种未用于开发调参的量化格式或 block/layout 形状；
- 至少覆盖 x86 与 ARM CPU；GPU 结论只在对应真机 backend 上成立；
- workload 参数是测试输入，不得固化成基础对象字段、公共枚举或默认策略。

留出失败必须归因到算法、实现、必要语义税、平台能力或证据无效。不得调换留出项、改变计费
边界、扩大事前预算或把失败模型宣布为“不典型”。若失败暴露模型专用假设，先修分层和形状，
而不是给基础 API 增加模型分支。

## 6. 可验证的成功门

推理引擎北极星不产生“感觉更通用”或“总体差不多”的结论。每个模块候选必须同时通过：

1. **功能门**：所有 `ENGINE_BLOCKER` 和本轮选择的 `COMPETITIVE` 任务有成功、边界和失败测试；
   每个公开入口都能反向指向合同卡。
2. **边界门**：语义擦除通过；模型/服务策略没有下沉；下游没有复制或绕过本模块的同义机制。
3. **复杂度门**：非平凡操作登记正常/最坏复杂度、常态元数据和进展保证；相对强 peer 没有
   未解释的算法档落后。
4. **正确性门**：固定 token 输入下的 logits/token oracle、所有权与释放、OOM/取消/迟到完成、
   sanitizer 和三平台适用门全部明确通过或登记不适用。
5. **性能门**：同一模型文件、量化、token 输入、精度、backend、线程/亲和、NUMA、batch、KV、
   冷热状态和计费边界与强 peer 对拍；按性能证据标准分别保存 comparability、qualification、
   exact-performance 与 non-inferiority 结论及原始数据。
6. **泛化门**：开发集和所有适用留出项分别通过预声明 non-inferiority/SLO；不得池化有利项
   掩盖单项 material regression。
7. **资源门**：吞吐不能代偿峰值 host/device memory、碎片、复制量、tail latency、公平性、
   错误率或泄漏；合同登记的每个资源指标独立守门。
8. **交付门**：受影响的 component closure、source/static/shared、ISA/backend 组合和安装后
   consumer 描述同一产品边界。
9. **文档门**：architecture 记录稳定边界，user 记录任务选择与坑，implementation 记录施工与
   结果，developer 记录可重复方法，原始点值进入 `evidence/`；各自没有互相冒充。

基础库的 SOTA 表示：所需操作采用同档最佳复杂度，关键正常 workload 相对已登记强 peer 排除
实质落后，并在用户关心的延迟、吞吐和资源维度形成可解释的 Pareto 位置。它不表示每个微基准、
每个地址布局或每个平台都必须获胜。亚纳秒、短 aggregate 或 code-placement 敏感结果若不能在
真实消费者中复现，只能作为机制诊断，不能驱动产品类型和 API。

## 7. 逐模块使用方式

重构某一模块时只改变该模块及必要的直接边；其他模块作为消费者或证据源，不顺手扩建。固定
顺序为：

1. 从 workload manifest 抽取本模块消费者 trace；
2. 完成需求等级、合同卡、语法消费者和语义重复账本；
3. 裁决 public/internal/compute/engine/不做，再对账成熟库；
4. 先修类型、ownership、复杂度和依赖，再做 codegen/常数优化；
5. 冻结 candidate，运行模块门、真实消费者门和留出门；
6. 只有全部适用门通过才签署；未知或未解释 material loss 保持黄色。

对于 core 这类底层源码 seam，完整推理引擎尚未存在时，可以使用从真实推理路径抽取的量化块
遍历、stable-node 生命周期、cache 争用和编译器 codegen replay。抽取必须保留原数据分布、
lifetime 和计费边界；人为单指令循环不能取代消费者门。

## 8. 停止与重开

模块通过上述门后停止无证据的反复重构。只有以下事件重开受影响项：

- 推理引擎真实消费暴露缺失能力、错误 ownership、难用接口、重复机制或 material regression；
- 新模型家族、量化布局、ISA/backend 或部署形状使既有合同失效；
- correctness、安全、OOM、取消、并发生命周期或构建/交付缺陷；
- 强 peer 在同合同正常 workload 证明更好的复杂度档或可复现 material advantage；
- 直接依赖升级改变实际算法、线程模型、内存归属、错误语义或 backend 行为；
- 公共边界、组件依赖或发布形状改变。

新库出现、feature 数量、LOC、命名偏好、单次跑分、亚纳秒差异或没有消费者的设想都不是重开
理由。新增需求先进入需求等级与 workload manifest，不能越过消费者证据直接修改产品。
