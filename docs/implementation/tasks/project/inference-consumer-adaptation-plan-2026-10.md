# HYLE 一期收口与推理引擎启动方案

目标是让 HYLE 在有限的一期消费合同下收口，并开始实现推理引擎。已有机制按真实消费
形状检查组合与结构成本；十项疑点是裁决输入，四个工作包是可选组织方式，不是全部完成
以后才允许写引擎的串行前置条件。优先解决阻断已选模型自然调用和资源安全的具体接缝。

用户已于 2026-10-13 选择：一个 dense/GQA 模型作一期完整交付验收，MoE/recurrent 片段用于
反证通用性；三种部署形态保留。用户同时明确，端到端 prefill/decode token/s 要在引擎闭环
后才能与其他引擎比较。因此 HYLE 收口不以尚不存在的完整引擎性能作为前置门。

文档状态：**设计提案**，2026-10-13；当前产品输入为
`6a41c49b261eacf06af4a10b86861ab42d062044`，其产品源码与前次审计 baseline `67192441c` 相同。
本页细化[单机基础能力审计](inference-single-machine-foundation-audit-2026-10.md)，承接
[原缺口研究](inference-foundation-gap-research-2026-10.md)。事实与源码身份见
[固定输入证据](../../../../evidence/architecture/inference-single-machine-foundation-audit/README.md)。
本页不是活动 closure card，不改变 module-status、不表示以下能力已实现或取得性能资格。

## 1 目标与完成标准

目标是单机高性能推理，覆盖单请求与多请求，以及三种部署形态：

| 部署 | 必须表达的消费形状 | 具体路线 |
|---|---|---|
| 单 GPU 主计算 | 模型数学在 GPU；CPU 承担提交、IO 和应用控制 | CUDA、Apple、Vulkan 分别签资格 |
| 一个 CPU 与一个 GPU 联合计算 | 同一旅程中包含真实 CPU 与 GPU 数学节点、资源搬运和依赖 | CPU＋CUDA、CPU＋Apple、CPU＋Vulkan |
| 单 CPU | 一颗 CPU 的多核执行；无 GPU 数学依赖 | CPU 与实际 Provider、线程池、ISA 域 |

联合计算不能只用 host staging 代替 CPU 数学；单 GPU 路线不能通过未声明的 CPU fallback
填补 GPU 支持域。分布式、多 GPU、多 socket 和完整 serving 政策不进入本方案的必做范围。

消费者层面的对称性以完成路线判断：相同的已准入数学、位置和有效区，可以通过公开 C++
完成加载、prefill、多步 decode、输出、取消及关闭；权限、错误、状态发布和退休合同完整。
允许不同原生对象、数据布局、规划与 capture 资格，但差异可发现且费用可核。
不要求所有 Provider 拥有相同 API、相同编码域或相同速度。

功能补齐与高性能资格分别验收。正确结果若依赖每步展开历史 KV、复制整个词表、临时重建
权重或不必要 host wait，应保留为成本问题，不能因有示例通过就写成高性能目标完成。

### 一期冻结合同

一期模型范围已经选定为一个 dense/GQA；具体 checkpoint、格式/精度、硬件与 Provider 在
首个引擎消费者卡冻结，选择来自目标部署与实际支持域，不为补齐清单随意换模型。
MoE/recurrent 片段只触发与共享机制有关的反证，不自动增加完整模型、格式或策略交付。

HYLE 一期库收口和引擎一期验收分成两个不同完成点：

| 完成点 | 必须完成 | 不以前置条件阻塞它的事项 |
|---|---|---|
| HYLE 一期库收口 | 冻结模型所需机制有自然 public 路线；shape/encoding、producer/权限、错误、取消/退休与三交付完整；P0 结构成本已裁决，适用局部验证通过 | 完整 engine/server、端到端 token/s、所有模型域、所有候选优化、二期分布式 |
| 引擎一期交付 | 同一 dense/GQA 模型加载→prefill→连续 decode→输出/取消/关闭；三部署形态；真实单/多请求；正确性与资源账 | MoE/recurrent 完整模型、二期模型切分/多机控制 |
| 引擎一期性能资格 | 完整固定候选在同模型/精度/硬件/请求合同下与成熟引擎比较，公开实际限制和差距 | 对源码或 isolated op 的数字外推端到端结果 |

有明确错误、生命周期漏洞、缺少所需消费路径，或结构复杂度已破坏选定目标的项不能简单
称“优化留以后”。反之，尚未在冻结模型中出现、已有成本可接受的替代，或只影响未选域的
候选不阻断库收口。性能收益大小未知不等于继续扩大库范围；在引擎完成后通过 trace 裁决。

库收口应输出一份有限交接清单：固定可用域、公开 recipe、已知 copy/wait/scratch、明确
不支持域、对应 owner 和触发条件。清单有未选功能不等于库永远无法收口；但所需 MUST
没有裁决完时，也不能把整个一期写成完成。

### 引擎何时开始

现在即可开始模型格式/tokenizer 接入、权重索引、已有数学域的模型图、请求与输出控制、
日志及资源观测；先使用一个已合格路线贯通端到端，再接三部署形态。缺少接缝的段明确
保留为未完成，不拿临时拆图或参考档签高性能。库修改只有在活动 owner 卡完成 SCOPED/
AUDITED 后实施，引擎消费代码不会绕过库的权限协议。

这一安排允许“库的有限接缝纠正”和“引擎正常产品实现”按实际依赖推进；不要求等待所有
G 项优化，也不将原 CLOSED 模块反复全局复检。创建活动 goal/施工卡仍以实际授权和冻结
范围为准，本页仅记录收口与启动方案。

### 新发现问题的准入与退出

先固定一期模型/路线合同，对现有 G01–G10 做一轮裁决：`当期所需 FIX/BORROW`、
`已有完整 ROUTED/recipe`、`引擎待验证`、`后续模型域/二期`。只有当期所需有限集进入
owner 卡；任何新增项附具体 trigger，并说明替换/增加了哪条已冻结旅程及原有限预算。
清单定界后不因阅读另一个 peer feature 自动扩范围。

| 新发现 | 当期处理 | 对一期完成点的影响 |
|---|---|---|
| 固定模型的错误结果、资源安全/退休错误、所需 public 路线缺失 | 提交具体调用和最小反例到真实 owner；依法 reopen、冻结有限 FIX | 阻断相应功能/路线，其他已签能力不重跑全局 |
| 已证明 warm 路径有额外全历史展开、完整权重重建等结构代价，破坏目标 | 用 P0 成本反例纳入有限修正，保可用参考档 | 裁决所需路线能否满足库交接；不能静默把参考档签高效档 |
| 可能影响吞吐/延迟，但自然路线完整且成本大小待实测 | 进入具名引擎待验证项，保 trace 与局部限制 | 不阻断基础库收口；端到端测出 material 瓶颈后再触发原 owner |
| 未选模型/编码/策略或二期拓扑 | 记录需求、owner、触发条件和负空间 | 不扩一期，进入后续独立 scope |

原范围被新事实改变时，只有活动卡可以将它纳入；审计记录中的“建议做”不会自动变成
新 MUST。一次 owner 轮到达有限修正集和预算后按协议裁决，不能把停止条件写成
“直到不再发现任何问题”。

## 2 差异如何归属

依据[能力完整性决策](../../../architecture/capability-completeness.md)，先列真实对象与调用，
再裁决覆盖和承载；API 名称、代码行数或 benchmark 行不能独立成为准入理由。

| 差异 | 处理方式 | 具体例子 |
|---|---|---|
| 普通公开 API 接线 | 保留自然调用，补 recipe 与失败用法 | F3 文件 stage 接 F2；已有 ready lease 接 Direct Compute |
| 模型和应用政策 | 引擎拥有，借现有机制实现 | tensor 名称、RoPE 配置、EOS、prefix key、LRU、请求合批 |
| 重复通用组合胶水 | 在真实 owner 补窄接缝，复用原子机制 | 多 Queue bindings、page footprint、producer 依赖、原生输入保活与释放 |
| 数学或存储域不匹配 | 明确 dtype、shape、encoding、layout；借成熟实现扩域或明确拒绝 | quant KV、window、不同 QK/V 维度、packed expert 消费 |
| 可运行但成本影响目标 | 先记结构账，再以同合同消费者做定向验证 | 全 KV gather、完整 metadata upload、selected IDs 的设备到 host 检查 |

承载方式分别记录为 `SELF / BORROW / ROUTED / APPLICATION`。HYLE 主要拥有集成、权限、
寿命、错误和交付接缝；成熟数学核优先 `BORROW`。不因某条旅程调用繁琐而另建 Queue、
allocator、scheduler、copy runtime 或模型级 cache manager。

一个真实可复现消费者、标准或成熟 peer 可以触发审计。通用 API 容易被推理特化时，用
第二条独立或正交旅程反证；不是必须先存在两个引擎产品才允许修正问题。

## 3 三条具体消费旅程

### 3.1 驻留 KV 的单请求与变长多请求

固定数学域先使用已准入 causal attention；具体 dtype、D、GQA ratio 与 Provider 由施工卡
逐路线确认。示例容量以 `page_size=16`、两条请求为起点，连续运行至少 64 个 decode step，
覆盖从第 15 个位置到下一页、请求加入/结束和 batch 顺序交换。数字用于保证事件发生，
不是新的公共 API 限制，也不是吞吐达标阈值。

当前公开对象关系是：

1. 冷阶段创建 Context、资源、Queue、`BatchSchema`、`PagedArena`、有限 invocation slots。
2. 引擎决定请求位置和物理页；`BatchSchema::version(HostBatchLayout)` 保存 owned layout。
3. `PagedArena::acquire(page)` 取得 `PagedPageWriter`；producer 或 F2 完成写入，沿现有
   `publish()` 或 `complete(ready_output)` 协议产生合法页 reader。
4. `PagedArena::read(layout, ready_pages)` 形成准确页面权限；Q 是普通 ready lease。
5. CUDA 合格档通过 `BatchInvocation::prepare_batch/update_batch` 和
   `PreparedBatch::run(query, keys, values)` 消费；CPU/Apple/Vulkan 当前 F5 参考档沿原
   materialized/native controller 消费。具体接口见[batch 指南](../../../user/batch-attention.md)。
6. 数学完成后消费 output，选择 token；释放不再需要的读者，实际 native uses 退休后复用页。

第 3–5 步之间目前有 ready 边界，不能把未完成 writer 当 reader。参考档还会物化重复 KV。
当前调用能证明正确路线存在，但不能据此证明低等待、低 scratch 的 decode 路线已完整。

目标组合语义为：

```text
已接纳 QKV producer
    → 精确 dirty 范围的 KV 编码/追加
    → 原生依赖与准确 page/layout 权限
    → 直接消费驻留 KV 的 attention
    → logits 数值选择
    → 必要的小结果返回与应用输出
```

这段是拟议旅程，不是现有新 API。新增接缝必须同时规定写失败、部分 accepted prefix、取消、
旧 reader、共享页、slot 复用和关闭；只有全部成功的结果才能沿原 F4 发布需要的 checkpoint。
旧 checkpoint 仍被引用时改写共享页需要独占证明或显式 dirty-page clone；禁止改写旧版本。

### 3.2 CPU 与一个 GPU 的联合图

使用一条公开图：CPU 计算 Q 或辅助数值 → GPU 消费驻留 KV 做 attention → CPU 消费结果。
另加独立 CPU 支路，检验有依赖的必要等待和无依赖支路的并行推进是否被混淆。
GPU→CPU 的结果搬运属于这条联合旅程本身，不要求所有跨设备 copy 消失。

当前接缝缺口：`graph::batch::Invocation` 保存单 Queue，最后只调用单 Queue submit；原
`LeasedInvocation` 的多 lane 图需要完整 `QueueBinding` 集。CPU＋CUDA 有两条 lane；
CPU＋Apple/Vulkan 的公开 bind 也没有 `PagedArenaRead`。详见审计 G02，以及
[当前 graph batch](../../../../include/hyle/graph/batch.h)、
[原多 Queue leased graph](../../../../include/hyle/graph/leased_invocation.h)。

拟议补齐分两层：

- common graph batch coordinator 借原完整 bindings/replay，保原 region、slot 和 completion；
- native hybrid owner 将准确 page footprint、layout generation 与自己的数学输入接起来。

API 名称与签名在 owner 的 AUDITED 卡冻结；本页不预先发明一个跨 CUDA/MLX/Vk 的万能
Invocation。只扩多 Queue overload 也不能宣告完成，因为原生数学消费、权限与 metadata
组合仍需证明。拆成独立图可作为现有可用对照，逐段寿命胶水和额外等待要计入适配账。

### 3.3 量化权重与 selected expert

引擎借成熟格式库读取 artifact 与 tensor 名称，确定编码、shape、layout 和模型角色；
HYLE 接收实际物理描述，通过 F3/F2 放到合格资源。冷转换若需要，明确何时发生和容量。
prefill/decode 直接消费驻留权重；MoE 的路由 ID 来源、host-known/device-produced 域明确。
专家 miss 只搬运所需编码块，再由合格 grouped/selected 数学消费。

当前有 `ArtifactWeightDecode`、多种低精度格式、QuantizedMatMul、grouped 数学和 weights
slot；冷解码到 dense、Apple 某些 block decode，以及 CUDA selected IDs 的 host 检查，
会形成不同消费成本。原生 packed 路径也已存在，不能把全部量化操作判成展开缺口。

目标是逐个锁定“artifact 编码 → 原生驻留表示 → 所选 row/expert → 数学”的自然路线。
LRU、prefetch 预测、模型兼容表和专家选择政策留引擎；编码转换、原生 owner 互操作、
检查与 actual retirement 的重复通用 glue 才下沉。保留 checked IDs 的错误合同。

## 4 十项具体处理清单

G 编号沿用前次审计，下面是处理方案与最低验收，不将所有候选一次列为实现 MUST。

| 项目 | 当前差异和引擎负担 | 拟议处理与 owner | 最低验收 |
|---|---|---|---|
| G01 高效驻留 attention | F5 参考 recipe 按查询容量/heads/KV 长度重复 gather，再 cast/MatMul；引擎可能另找 dense/view 路径或重写 batch glue | 各 route/Provider 与 compute_batch 借成熟直接消费；先证明现有单请求 dense/view，再补确有需求的 paged/varlen 等价档 | 同数学/位置/有效区；长 prefill、短 decode、混合长度均保成本账；不得额外按每 Q/head 展开完整 KV |
| G02 联合分页接缝 | single Queue coordinator 与 hybrid bind 无法直接组合完整分页权限；拆图重复接 owner/completion | graph_batch 与三个 hybrid owner 复用原 multi-queue/native region，接准确页权限与 generation | CPU→GPU paged→CPU，完整 bindings；重复/错误绑定、旧 output、两个独立 slots、取消与三交付 |
| G03 增量更新到读取 | F1 exclusive、F2 producer 已有；F5 ready-only，原生 input imports 可能保活至重绑定/销毁 | compute_batch 及原 F1/F2/route seam；先证现有页 copy/双 buffer，再按真实 carrier 接 producer、写足迹与 idle release | 连续 64 step、跨页、共享旧页、producer failure、取消；只写 dirty 域，仍在途 owner 不提前释放 |
| G04 量化 KV | batch 只有共同 F16/BF16 与一个 D；roundtrip 返回 dense 不等于驻留压缩 cache | compute_ops/batch 与 Provider 分离 payload/scales/layout 资格，借 mature quantize-append 与压缩读取 | 一个真实选定 KV 编码，实际 backing/scales 核计，误差域与尾部/边界有 oracle；不能每步解码全部历史 |
| G05 attention 语义域 | FI 当前 causal、有限 D/GQA，同 D、无 window/sinks/softcap；其他具名 model ops 也是有限域 | 目标模型先触发精确参数需求，再由原 Provider 窄扩域；RoPE 配置、模型政策不下沉 | 对绝对位置、窗口首尾、mask 和无效区作独立参照；不支持组合具名拒绝，不静默改数学 |
| G06 metadata/decode 规划 | 完整 snapshot/H2D 与显式 host waits；当前 batch native 使用 prefill plan；引擎要自己管理 bucket/重复准备 | compute_batch 与 FI/native owner 分离 host-known/device-produced、active/delta 与 topology；借成熟 decode 档 | stable tail、跨页、加入/结束、重排、同 shape 不同 layout；逐层 upload bytes、plan、host waits，错误不留旧 token |
| G07 packed 权重消费 | 现有编码与原生承载不均；选中权重临时展开、IDs 检查等待可能进 warm 路径 | 目标格式/数学 owner 借原生 packed 或冷 repack；明确 checked IDs 的 host-known/device 域 | resident 与 temporary 分开，cold/warm 分开，重复 decode 不重解码全部权重；保越界错误和已接纳 suffix |
| G08 logits 数值选择 | TopK/Softmax/CPU RNG 已有，GPU 随机完整路径未成 public consumer；部分 TopK 走全 argsort | compute_ops/Provider 评估 row-wise argmax/selection/随机原语；seed/counter 契约明确，sampling policy 留引擎 | greedy 只回必要小结果；随机路径检验分布、request seed、重排与错误；有限 TopK 不冒充任意精确 top-p |
| G09 联合流水资格 | Apple/Vulkan 单 invocation 按 fragment await；独立 invocations 可 overlap | 原 hybrid owner 先借 slots/Sched 验证 microbatch，只有真实无依赖串行化才补有限推进 | 同图独立 CPU/GPU 支路与至少两 microbatch，trace 区分必要依赖和额外 host wait，关闭仍正确 |
| G10 CPU 预算与观测 | cpu options 控 ggml native_threads；ATen private pool 未统一控制，引擎要协调嵌套线程 | compute_route_cpu/实际 Provider 先报告 executor 与 private pools，再裁决 phase contexts/池控制 | prefill/decode 与两个并发 invocation，计实际线程/oversubscription；不把局部参数解释为全进程约束 |

G02 为源码可定位的公开组合缺口；G01/G04/G05/G06 为明确参考成本或缺席域，其实际准入和
严重程度依目标模型/机器确定。G03/G07–G10 必须由最小消费者收窄；现有路线若满足目标，
可裁决为 recipe、完整 ROUTED 或只补资格，不必新增 API。

### G01 的具体成本界线

原 F5 materialized 路径令 N=max_query_rows、H=query_heads、L=max_kv_per_request、D=head_dim，
索引为 4NHL bytes、mask 为 4NL bytes、F16/BF16 gathered K/V 为 4NHLD bytes，后面还有 F32
cast 等临时值。[现有 recipe](../../../../src/compute_batch/batch_materialize.cc#L58)对此显式计费。
N=32、H=32、L=4096、D=128 时，仅 gathered K/V 即 2 GiB；不是峰值 resident 实测。

高效档验收要区分实际 KV backing、必要数学读取、native 必需 scratch 与额外物化。
attention 本身需要读取有效历史 KV；禁止的是适配层因每个 Q/head 重复构造全历史 payload。
单请求 dense view 若可直接满足，应保留原 native 路径；多请求无须强制所有平台使用相同分页
对象，但必须保真实有效区、权限和可核成本。原参考 recipe 继续作为显式正确性/有限成本档。

### G03 的具体权限界线

不是取消 ready 检查，而是组合“接纳 producer → 原生 wait → 成功读取/输出”与原 use 记录。
`ReadLease::try_write`、page writer、F2 producer 和完成后 ready output 都已有合同。
不能给任意 GPU kernel 开放可写 whole-arena view，也不能因为 public Completion 不再 pending
就忽略未裁决的 actual retirement。新 helper 如存在，必须由原 mutable carrier 和实际写范围
证明；失败后不能发布新有效长度或将部分更新当成完整版本。

### G06 的具体复用界线

同 shape 不代表同 page/layout，更不自动代表同 native topology。复用 key 必须覆盖 Provider
要求的全部事实，generation 与输入权限同源。尾长更新、页增长、请求重排分别判资格。
规划 reuse 不能隐藏完整 metadata upload，也不能宣告 capture：上传、更新和运行的实际
capture 域及 fallback 成本必须分别核定。多 layer 共享 host snapshot 不等于可共享有 mutable
native state 的 workspace；只有 Provider 明确支持时才复用。

## 5 四个有限工作包

这些包按需求选择和定界。G02 只有在冻结消费者确需同一个 paged hybrid 图时才成为对应
路线的 MUST；若现有独立图组合经完整权限/错误/成本裁决满足目标，保留该公开 recipe 并
记录差距。G01 优先排除已确认的物化成本；G03/G06 只修被选消费旅程实际阻断的组合。
G04/G05/G07 由一期模型触发，未选编码/数学域后移；G08/G09/G10 的收益候选随引擎 trace
验证，不默认挡住引擎启动。正确性与生命周期问题始终优先，不受优化优先级豁免。

| 包 | 范围与先后依赖 | 交付物 | 停止条件 |
|---|---|---|---|
| W1 联合分页与高效读取 | G02＋G01；先冻结原数学/有效区及公开联合调用 | 多 Queue/page seam 的 owner 卡；CPU 与三 GPU 的 native/等价读取资格表；上述三部署 consumer | 每个已准入格有自然调用、正确性/生命周期证据及明确成本；参考档与高效档区分；不顺手扩所有 attention 域 |
| W2 增量更新与低等待 decode | G03＋G06；沿 W1 已选原生消费档 | dirty 更新/producer 旅程、metadata/plan 复用表、长序列 warm 成本账 | 已准入稳定 tail/跨页/重排/请求结束都裁决，旧结果及失败 prefix 受保护；host wait/copy 达冻结目标或如实留差距 |
| W3 目标编码和语义 | G04＋G05＋G07；以具体目标模型触发，不依赖所有 W2 优化完成 | 精确格式/数学域、BORROW 输入与升级边界、quant KV/packed weights 路径 | 只完成被冻结模型/编码/语义；其他域明确负空间，不用“继续适配”无限加项 |
| W4 输出与联合流水 | G08＋G09＋G10；复用已有稳定数学及所有生命周期合同 | 数值选择 consumer、CPU/private pool 观测、microbatch dependency trace | greedy/选定随机域、联合支路和 CPU 预算分别裁决；不扩成 serving scheduler 或完整 Sampler 产品 |

每包先绑定各真实 registry owner 的 baseline/reopen trigger，进入 SCOPED，再在 AUDITED 冻结
有限 FIX、API、预算与适用验证。owner 分批实现，不把十项做成一次全库重构。新 component
只有自然机制确需独立身份时登记，不能按 G 编号机械拆十个模块。

构建只按[change-impact selector](../../../developer/review/change-impact-validation.md)选择
owner 和真实直接消费者；公共 API/package seam 才扩展对应验证。分批复用依赖树与缓存，
不因每个 op 或每次文档更新重建全库。适用 source/static/shared、sanitizer/原生风险写入
owner 卡；数学/API 固定的不可变 candidate 才按同合同启动定向 P1/P2。

## 6 共同验收数据

### 用例和事件

| 用例 | 必须发生的事件 | 主要问题 |
|---|---|---|
| 单请求长上下文 decode | prefill 后至少 64 step、尾页增长和跨页、保旧输出一次 | G01/G03/G06，warm 路径复制与等待 |
| 两请求不同长度 | 请求重排、加入/结束、一个跨页另一个不跨页、无效 capacity 区填不同值 | active 区和 layout identity、G02/G06 |
| prefill 与 decode 混合 | 长 Q 的 request 与单 Q request 同轮消费 | 不能只优化 N=1，G01/G05 |
| 共享 prefix/分支 | 旧 reader/版本仍在、dirty 页独占失败或显式 clone、新候选失败 | 不破坏旧版本，F1/F4 与 G03 |
| 联合与并发 | CPU→GPU→CPU、独立 CPU 支路、两个 slots、多 microbatch | G02/G09/G10，完整 bindings 与实际 overlap |
| 错误和关闭 | 越界 IDs/页、stale token、接纳前 OOM、producer 失败、接纳后取消、持旧 output 关闭 | 原错误、suffix 与 actual retirement 合同 |
| 编码与语义 | 一期模型实际使用的编码和数学域；quant KV/window 等仅在选定模型需要时加入 | G04/G05/G07，独立 oracle 和物理 bytes；未选域不要求凑用例 |

容量参数、真实模型/精度、数学误差、性能 material margin 在活动卡中冻结；本方案不把任意
示例数字签成所有 Provider 必须支持的域。数值 oracle 独立于被测同一实现，不以跨 Provider
逐 bit 相同或跨 RNG 后端同 seed 同 token作为未声明合同。

每个适用路线格保存：真实 OS/硬件/toolchain/Provider、固定 candidate、数学域、公开调用、
正确性、权限、错误、关闭、成本、资格限制与验收状态。Windows CUDA、Linux FI、Apple、
Vulkan 分别按实际 OS 域记录；不能借其他 OS 的 PASS 填格。Apple 验证只用用户指定 Air，
运行前读取实际身份。未运行门保未完成，source-only 判断不写 PASS。

### 成本账

| 项目 | 记录内容 | 判定方式 |
|---|---|---|
| resident | KV、weights、layout/mirror、输出、slots、已知 native backing；private UNKNOWN 单列 | 实际物理表示；不把 requested bytes 当驱动全部内存 |
| temporary | gather/cast/repack、native scratch 的 bytes、容量依赖和寿命 | extra adaptation 与数学必需费用分开 |
| 数据移动 | H2D/D2H/device copy 的实际 bytes、cold/warm、dirty/active/capacity | 无用历史复制/完整词表返回与必要跨设备结果分别计费 |
| 控制 | host waits、native waits、plan/replan、metadata 校验/扫描、cold/warm 分配 | 必要依赖不能当缺陷；host 观察边界与 native ordering 不混算 |
| 线程与执行 | Sched executor、native private pools、实际同时活跃工作和 dependency trace | 不用 pending 数量代替 overlap，不由局部参数推全进程预算 |
| 局部消费结果 | 同语义算子/region/片段的延迟、数据量、完成与关闭 | 只签局部档位，不外推 engine token/s；沿原仪器资格 |
| 引擎消费结果 | 完整引擎的 prefill/decode token/s、TTFT、ITL 分布、吞吐、模型/精度/请求合同 | 在引擎闭环后比较；不作为库收口的循环前置门 |

优先做 P0 复杂度与结构账：如 full KV 的 Q/H 重复物化可直接定位，未必需要先跑全模型
性能。定向 P1/P2 用于结构纠正后是否产生 material 收益，不以一个未资格点值决定公共 API。

### 端到端性能由引擎验收

prefill 与 decode 分开记录。prefill 按实际处理的 prompt tokens 和明确计时窗口计算；decode
区分单请求 ITL/生成速度与多请求 aggregate output token/s，warmup、首 token、EOS、取消、
排队和请求是否完成的规则一致。CPU/GPU 联合路线分别保实际 placement 与线程预算。

比较 llama.cpp、vLLM、SGLang 时，固定模型权重/数学语义、量化精度、上下文与输出长度、
请求数/到达条件、batching、prefix/capture 开关、硬件、版本和计时边界。不同引擎对某路线/
编码不支持时，明确不可同合同比较；不得改成另一精度或设备后仍声称直接输赢。
只给 token/s 会掩盖排队和首 token 代价，因此同时保 TTFT/ITL 分布与资源账。

端到端瓶颈回流 HYLE 时附：固定模型和候选、调用链/owner、正确性复现、trace、copy/wait/
scratch 证据、同合同影响与最小允许改动。应用 batch/策略问题由引擎处理；原生接缝、数学
承载或结构代价问题回对应库 owner。修一项不授权再做全库完整性审计。

## 7 二期多 GPU 与多机边界

二期保留为明确产品阶段，不要求 HYLE 一期先实现所有二期机制。当前单卡的 `Completion`、
Queue 或“collective”名称不能证明已有跨 GPU/跨机完整路线；例如设备内 block collective
不等于 NCCL/RCCL 类跨设备通信。[Compute 边界](../../../architecture/compute-design.md#13-四条独立路线与明确非目标)
也不由四路线推导多 GPU 支持。本节是需求分解，未对二期作完整源码资格审计。

| 二期拓扑 | HYLE/成熟依赖需要审计的基础机制 | 引擎拥有的策略 |
|---|---|---|
| 单机多 GPU | device/topology 与资源身份、跨设备 peer copy 能力/host fallback 成本、event/stream ordering、跨 GPU collectives、group 生命周期/错误与实际退休 | TP/PP/其他模型切分、rank 到设备映射、microbatch、参数/KV 分布、admission |
| 多机每机一个 GPU | 传输/通信 group 与 buffer ownership、注册内存、异步通信 completion、timeout/cancel、故障后在途使用退休；成熟 transport/collective 的交付与资格 | rendezvous 配置、角色/分片、pipeline、路由、故障后请求重试或终止政策 |
| 多机每机多 GPU | 上述机制组合、节点内/间 topology、跨域 ordering、group 资源及错误传播、实际扩展成本 | 分层 TP/PP、rank/topology 规划、调度、负载与部署控制 |

数学/通信核优先借成熟依赖；HYLE 只拥有真实必要的资源、执行与失败接缝。跨设备/跨机
collective 与 transport 需单独版本、错误和生命周期合同，不能隐藏在单卡 Queue 中。
跨机故障可能导致资源无法立即证明可复用；明确隔离/失效与待退休状态，不能用 host 超时
冒充设备已停止或远端已释放，也不要求不可能的任意故障回滚。

二期路线不能机械扩大成“三拓扑×全部 GPU 后端必须一次达标”。先冻结实际首个设备/通信
环境、真实模型切分及完成/故障旅程，再审承载；CUDA 之外的 GPU backend 逐项判实际原生
能力，未选资格不能伪 PASS。选择首个二期 backend 不取消三种拓扑的长期目标。

一期应保资源/device/queue 身份、显式 producer/retirement、有限 slots 和原 ownership 合同，
不给模型 layer/token/rank policy 塞进基础 API；不预造空 transport target、万能分布式
executor 或通信接口。二期由真实消费者证明无法组合时按新 scope 补，而非为了未来可能性
提前改完整单卡 runtime。这样保留发展空间，也给一期一个可到达的结束点。

## 8 引擎代码量如何重新评估

固定一份最小但真实的引擎 consumer 基线：加载真实权重、prefill、连续 decode、单/多请求、
输出及取消关闭。主模型先选 dense/GQA，另留一个 MoE 或 recurrent consumer 防过拟合。
两份代码均使用同一模型范围、同一产品能力和同一交付方式，不能一份含 server 另一份不含。

将引擎自己的代码分成三个并列桶：

1. 模型/应用代码：权重角色、模型拓扑、tokenizer 接入、位置/输出语义、cache/request 政策。
2. 可下沉通用适配：跨 route layout、owner/import、producer/page guards、metadata/plan、
   错误 suffix、取消及退休。按 G01–G10 关联到具体文件和调用。
3. 普通接线：Context/资源创建、有限 graph/batch 绑定、错误传播、必要应用控制。

分别统计产品代码、测试和工具；native dependency、生成代码及 vendor 不算引擎自有代码。
保留统一 physical LOC 与非空非注释统计口径、固定 revision、文件集合和跨桶共享计数规则。
一个 HYLE helper 搬走 200 行但引擎又新增 150 行迁移/适配，应按实际净变化记录；只减少
调用行数而内部仍完整 gather/wait，不记成性能缺口解决。

每个工作包完成后记录同功能 prototype 的 before/after：引擎自有产品 LOC、通用适配 LOC、
平台分支数、状态机/寿命 owner 数、copy/wait/plan 账，以及测试维护量。只有完整旅程替代
重复机制时才报告节省；HYLE 新增 LOC 不能直接从此前 14–30k 条件估计中相减。
原估计见[引擎工作量评估](foundation-post-goal-engine-effort-2026-10.md)，尚未由完整 prototype
校准；本方案不给十项完成后的确定新行数。

## 9 单项适配记录模板

后续 owner 卡可复制下面的字段，每项结论都能反查成本与验收。

```text
关联 G 项 / consumer / 固定 baseline：
实际 shape、dtype、encoding、strides/layout、active 与 capacity：
原生对象与 public 调用序列：
producer、读写/alias 权限、版本、错误、取消、actual retirement：
当前引擎适配文件/产品 LOC/平台分支/状态机：
cold 与 warm 的 resident、temporary、copy bytes、allocation、plan、host waits：
现有机制反证与可用 recipe：
覆盖结论及 SELF/BORROW/ROUTED/APPLICATION：
真实 owner / 最小允许修改 / 不支持域：
正确性与失败用例 / 三交付 / 定向性能合同：
完成或停止条件 / 实际结果与证据：
```

只有现有自然路线、具体 owner 和完整验收均明确时，才能把适配项从缺口清单移出。写了
wrapper、补了名字或只有数学输出，并不足以证明完整消费形状已交付。
