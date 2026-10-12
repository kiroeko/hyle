# A/F4/F5 完成后的引擎工作量与消费适配评估

2026-10-12，RESEARCH_ESTIMATE。用户在活动 goal 中要求评估：少量模型、四路线、量化、流式输出，
并明确最终目标是**单机高性能推理，单请求及多请求都覆盖**。此记录是按既定 goal **全部通过后的条件预测**，不是引擎实现、当前能力证书或
新模块施工卡。原估计调查时A已有限范围CLOSED、F4/F5未实施，原产品候选546b0d2468e5de29031c34668f4afa4bc31b9a17、HEAD41d8f60784b358822a4b1a7abe5482d2a53d593f。
当前A与F4已CLOSED，F5实现固定25e4fb398 CANDIDATE，完整goal尚未收口；状态只以registry为准。
以下原条件估计不冒充最终goal后统计，完成定义及剩余门见[completion audit](foundation-f45-completion-audit.md)。

## 评估口径

目标假设为两种结构相近且已有成熟算子承载的 text-only decoder 模型；例如一个普通 Transformer
及一个已覆盖 recurrent/conv 的模型。支持 CPU、CPU+CUDA、CPU+Apple、CPU+Vulkan，自然原生模式
按公开资格使用；有限形状冷准备、小量并发会话、现有量化格式、CLI/嵌入式流式 token 输出、取消和关闭。
具体模型名尚未选定，不能由算子目录直接推定整模型正确。

复用受审格式读取与 tokenizer 库，算子、执行、异步搬运和资源寿命实际走 HYLE。
若直接借另一个引擎的整套 context/执行循环，虽然 glue 更少，却不能用于证明 HYLE 基础能力完善。
不从零自研 tokenizer/kernel，不承诺全部 GGUF/量化跨路线均可，不纳入多模态、分布式、任意 LoRA、
生产级连续组批、完整 OpenAI API、grammar/工具调用、所有模型和性能对齐 SGLang/vLLM。

估计“产品行”指引擎自有 C++ 非测试有效代码，包含必要 API/头和适配；排除 HYLE、第三方/vendor、
生成文件、文档、空白与测试。下列区间是自底向上的设计估计，没有真实 engine prototype 校准，
不是统一 LOC 仪器测出的未来结果。风险主要是端到端 layout/权重适配和跨路线共同能力域。

## 做完 goal 后的对称边界

| 模式 / 能力 | 预计共同合同 | 保留的实际差异 |
|---|---|---|
| 有限 exact shapes | 相同 Graph 身份/guards/coverage；冷表、表外拒绝、独立槽与旧结果 | 不保证任意动态 general；每个 Invocation 仍固定已选择形状；MLX shapeless 未由本 goal 自动准入 |
| pure control | 前驱活性、顺序和失败停止后继 | 原生 barrier/managed waits/materialized outputs 不同，禁止 host wait 的图可拒绝 |
| 外部 effects | 合法端口、footprint、alias、成功/失败前缀的共同裁决；合格 CPU 岛可实际执行 | 没有合格可写 GPU carrier 时原生 effect 仍拒绝；不保证任意外部 mutable native 输出 |
| 状态 share/branch/publish | F4 同 schema/版本、显式 dirty clone、expected-base 发布、旧版本/native uses 保活 | native copy/readonly carrier 不同；不能回到未保存 checkpoint，业务 RNG/token/位置由引擎恢复 |
| paged/ragged 消费 | F5 同数学端口、容量/有效区、metadata generation、prepare_batch/run 与退休合同 | 首批原生 paged 是 Linux CUDA；Windows CUDA及Apple/Vulkan的 native 承载仍按准入事实，显式 materialize 含副本与等待 |
| capture、预算、性能 | 可查询资格、报告真实已知/未知成本，失败明确 | 不保证所有模式 capture、零复制、相同 native memory 上界或相同吞吐 |

因此本 goal 补齐的是已声明消费旅程的语义和接缝。所有性能档位、任意动态形状和所有 GPU mutation
仍完全一致，不是此 goal 的验收条件。F5 的 PENDING_BY_PROVIDER 不能改名 NOT_APPLICABLE，
也不能用偷偷 CPU fallback 给四路线凑满表。

## 对实用本地引擎是否足够

结论为 **机制边界预计足够，端到端尚未证明**。少量模型、小量并发、量化和流式输出不要求
所有native具有相同capture/general算法。只要shape、layout、编码与更新操作落在合格域，
在冷准备明确选择route/capacity/artifact，普通生成循环可以共用模型与会话逻辑。

不应为追求签名或性能档位完全相同继续扩万能API。优先完成以下具名consumer证据，
再以实际重复glue/缺失数学决定是否增加机制；本记录不授权实现整引擎：

| 验收问题 | goal已有覆盖 | 仍须真实模型证明 |
|---|---|---|
| 所选模型的prefill与decode是否能在共同数学/编码域运行 | 既有算子承载与A有限shape；F5新数学接缝 | 位置/RoPE、模型mask、权重排列与量化布局组合，独立数值/logits参照 |
| KV/state是否可连续多步更新并读取 | F1/F2/F4/F5已有或计划的append→read→branch→publish寿命闭环 | 所选模型真实KV encoding和更新域；不能每步重建全历史KV且隐藏副本 |
| 跨路线是否只需窄适配 | 原生独立artifact/typed端口/资格报告，CPU islands显式 | 同一个生成consumer能复用，backend/state glue不重复自写last-use或整套执行循环 |
| 容量/取消/旧结果是否能完成正常产品任务 | A/F1–F5故障与退休合同 | 长prompt、逐步增长、满容量、多会话退出、正在生成时取消/关闭的完整旅程 |

实用性还取决于上下文与内存目标。若显式materialize在目标prompt长度下每步产生不可接受的
KV副本/等待，则该平台native paged或成熟等价native消费成为具名竞争性缺口，需要另行准入；
不能因为数学能跑就写“足够”。同理，若所选模型必须原地KV更新而当前qualified carrier不能
承载，或输入/输出只靠host来回搬才能接上，该具体consumer可触发窄seam审计。

高并发serving比本地少请求更容易触及这些限制：native paged、有效区、低成本append、
稳定batch/capture topology和workspace容量需要目标机器上的真实资格。当前goal明确保留的
Provider pending不等于这些工作以后永远无需做；是否升级为MUST由目标consumer决定。
未有反例前先完成固定A/F4/F5，而不是把可能的所有serving能力纳入本轮。

## 原 M1–M7 的逐项消化

原 [gap 研究](inference-foundation-gap-research-2026-10.md) 固定在00fdc40，保留当时事实。
这七行大多是“已有原子/实现与引擎消费形状不一致”，不是七个已复现 correctness bug。

| 原项 | 具体差异 | 本轮/已有包覆盖 | goal 后引擎仍要做 / 仍保留的限制 |
|---|---|---|---|
| M1 每步 batch/长度/页表变化 | 固定 Invocation shape 不等于每批 layout；早期 Apple/Vulkan symbolic/control/effect 全拒绝 | A 的 finite exact/control/合格CPU effects；F5 的容量与版本化批准备 | 选择 bucket、有效长度、模型/LoRA key、未命中冷准备；general/capture 逐原生资格；只补 padding 不足以证明数学 |
| M2 请求已结束、GPU仍引用块 | Result 私有 workspace 保活不自动拥有 caller 外部 KV；Ref/CPU RCU 不证明 GPU完成 | F1 已 CLOSED，原最后使用/lease/退休；F4 集合版本继续借用 | 业务 prefix/session 引用和淘汰策略；所有提交通过真实lease/last-use接线，裸 TensorView 不自动成为 owner |
| M3 已驻留 backing 的范围搬运 | WeightSlots 固定 catalog/staging 正确，但不是任意页范围；不该重新 upload 完整权重 | F2/F3 已 CLOSED：有界原生批量范围与资源流送 | 计算选中的 page/expert byte ranges、放置/预取；编码转换须借原数学，不能按 bytes 伪 repack |
| M4 字节分页与分页数学 | CompositeBuffer 有分段 bytes，但 SDPA 的 Q/K/V 无页索引/indptr | F5 接 typed paged/ragged 与真实 native plan/run | 引擎建立逻辑KV→物理页表；格式/arena适配；原生paged未准入平台显式参考档或拒绝，不能当同成本 |
| M5 host 文件准备可能阻塞 | 原 upload 和 file fallback 的进展边界，外层 cancel/timeout 不等于已接受底层IO停止 | F3 已 CLOSED：有界 read/stage/native-ready、背压、取消/实际排空 | 解析格式和分片索引、预取优先级、驻留策略；平台不支持可靠逐操作文件取消的边界仍存在 |
| M6 speculative branch 是完整状态集合 | initial/final 只覆盖该tensor；conv/temporal/sibling/RNG/位置未必被保存 | F4 明确schema/集合版本/share/dirtyclone/发布；A 补CPU effect接缝 | 定义完整模型 checkpoint，MTP接受范围、token/RNG恢复、ring已flush前提；无法恢复没保存的过去 |
| M7 总显存上限与私有成本 | managed requested bytes 不等于driver、allocator cache、capture、系统总量 | KEEP 已知/未知账，F4/F5明确新目录/backing/workspace账 | 留安全余量、admission/offload策略；传一个预算不保证整进程绝不OOM，严格未知项可拒绝 |

F1/F2/F3 的历史 CLOSED 范围不需要因这张表重复验证；新 consumers 对它们的实际接线仍在当前
F4/F5 卡定向验证。普通 WeightSlots、TensorView、Graph Invocation 保留其原有任务，不强行改成
模型级资源管理器。原 reference 研究也明确 native 返回类型、external writable output、general、
capture 和 Linux-only FlashInfer 是不同能力档位。

## 本地引擎剩余自有代码估计

下表8,000–16,000档先估可用local core；用户随后要求的单机高性能/单多请求增量在表后另列。
不能拿较小档的数字回答最终高性能目标。

| 工作 | 预计产品行 | HYLE/成熟依赖已负责什么 | 引擎要实现什么 |
|---|---:|---|---|
| 模型配置、文件/权重/量化适配 | 1,000–2,000 | IO/mapping/F3、已有计算编码/repack、成熟格式核 | tensor名/角色、shape/排列、格式元数据、缺权重/损坏诊断 |
| 两模型构图与位置/state语义 | 1,500–3,000 | typed ops、Provider、Graph编译/partition/copy | layer结构、RoPE/attention位置、模型差异、initial/final完整接线 |
| tokenizer/模板/编码适配 | 500–1,000 | 成熟 tokenizer/Unicode/JSON | model vocab选型、特殊token、模板、增量decode、错误处理 |
| prefill/decode/会话/生成循环 | 1,000–2,000 | 可复用artifact、独立Invocation、F1/F4 | prompt分块、position更新、EOS/停止、取消、会话状态机 |
| KV/state/bucket资源策略 | 1,200–2,400 | F1–F5寿命/搬运/版本/paged接缝 | 页表、会话引用、容量准入、bucket选择、回收/驻留策略 |
| 采样与输出流 | 700–1,400 | RNG/SIMD/通用输出与调度 | greedy/top-k/top-p/penalties、token停止、流式解码与背压 |
| 四路线选择与残余消费适配 | 800–1,800 | 各原生Context/算子/图/交付与报告 | 能力交集选择、layout/权重桥、显式参考档策略、错误上下文 |
| CLI/嵌入API/配置/诊断 | 600–1,200 | Error/Obs/IO/Sched/容器原语 | 产品配置、会话接口、指标命名、进度与终端输出 |
| **分项合计** | **7,300–14,800** | | |

四路线这一行只计跨路线补充 glue；不再重复计已在模型或KV行中的代码。模型/布局尚未最终
指定，规划宜取整为 **8,000–16,000 行产品代码**。其中领域代码预计占主要部分；不应为了
让这一总数更小，把模型命名、tokenizer、KV策略或采样塞进基础库。
额外测试/模型差分/故障/四路线集成夹具初估6,000–12,000行，不和产品行混算。
测试量同样是预测，不以行数替代数学/寿命证明；模型资产和第三方依赖不计入。

参考档位：单模型/单路线/单请求的最小可用引擎约2,000–5,000产品行；上述实用本地引擎
8,000–16,000；在已有local core上增加有界continuous batching、prefix cache、paged策略和
服务接入的单机serving总量约20,000–40,000。后者不代表SGLang/vLLM feature parity；
分布式/多模态/大量模型/生产运维没有由此估出可靠上界。

按一位熟悉 HYLE/C++ 的工程师、复用成熟 tokenizer/格式核、所选数学与量化均已准入，
local core实现及四路线验收规划约6–12人周；不是当前agent的完成时间承诺。
模型端到端正确性、布局、native资格和硬件验收比录入这些行更可能决定日历时长；
goal后prototype如暴露新基础机制或必须新native provider，此估计需重新审计。

### 用户最终目标：单机高性能，单请求及多请求

在上述local core基础上，增加真实连续组批/公平与admission、prefix/page策略、稳定bucket与slot
管理、prefill/decode并存、每路线成本诊断及targeted性能纠正。这里仍无分布式、多模态、全模型
或生产HTTP产品，故不能拿整个SGLang/vLLM的规模当比较对象。

| 高性能增量 | 预计新增产品行 | 属于谁 |
|---|---:|---|
| 单机batch调度、准入、公平、prefill/decode队列与请求生命周期 | 2,000–4,500 | 引擎策略，复用Sched/Time/IO，不再写底层线程池 |
| prefix/page/cache策略、fragmentation、压力与offload选择 | 1,500–3,500 | 引擎策略，复用F1–F5，不再写异步最后使用协议 |
| 稳定bucket/slot/plan复用与路线成本适配 | 1,000–3,000 | engine key/choice与窄适配；缺失通用native消费由真实counterexample进入owner |
| 性能相关产品诊断/配置/退化处理 | 500–1,500 | 产品观测，复用Obs/实际报告；benchmark工具不计产品行 |
| **增量合计** | **5,000–12,500** | |

分项与local core合计12,300–27,300，规划取整为 **14,000–30,000行自有产品代码**；
tests/数值与多会话/容量/取消/性能验收夹具另估10,000–20,000行。
在数学/编码/目标native路径已具备并复用成熟tokenizer/格式核的条件下，熟悉项目的一位工程师
规划约12–24人周用于实现与跨路线验收。数据仍未被真实engine prototype校准；若需要新增
native paged/append/general carrier，这部分基础库工作单列，不能偷偷放进此范围并保证期限。

当前goal是这项高性能目标的重要基础，**不足以单独签高性能完成**：其A门为P0结构裁决，
F5也不能以explicit materialize正确或Linux CUDA原生资格代签所有路线吞吐/延迟。
新的P1/P2必须在具体目标模型/工作量/API固定后由单独有限卡启动，不能为“高性能”无限跑全库。

单请求的自然成熟路线不一定使用FlashInfer页表。llama.cpp固定f2cc7282的
[get_k/get_v](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/llama-kv-cache.cpp#L1296)
按当前n_kv和stride创建原native views，更新通过原set_rows/copy。因此要验证的是**驻留KV可由
原数学直接消费、只更新实际新增/dirty范围、真实有效长度与低成本复用**；不必要求所有模式
拥有同一种paged对象。多请求则必须验证不同长度/页布局的真实原生或成熟等价承载，不能用
每步全历史gather/host回传当高性能默认。

成本红线来自机制而非纳秒点值：单步新增token的数据只应更新其实际范围；如果decode每步
因接口适配而再复制整段历史KV，这是额外O(context_bytes)搬运，与attention原本读取历史
的数学工作不是同一项。固定容量若强迫对max_context计算而非实际长度，也需记录多做工作。
host-authoritative布局更新允许显式规划/上传，但稳定layout不应重复冷编译/无界分配；
capture是可能的优化，缺capture本身不等于不能高性能，须按原native提交成本比较。

后续最高优先级是窄纵切：一个模型的单请求长上下文decode与多请求变长连续组批，
实际记录KV写入/额外copy bytes、host/device往返、planning/allocation次数、等待和旧结果寿命，
先以同数学正确性证明排除误比较，再按合格仪器签prefill吞吐、首token、逐token与并发吞吐。
只有该证据确认库消费接缝造成重复glue或material退化时，才新增最小native seam。

## 估计根据与不确定性

固定源码尺度见 [source-scale.json](../../../../evidence/architecture/foundation-engine-effort/source-scale.json)。
其“nonblank_lines”是简单物理非空行，含注释，仅作尺度诊断，不冒充HYLE统一LOC口径。

- llama.cpp 固定 f2cc7282 的 context 3,771非空行、loader1,547、vocab4,048、sampler3,547、
  KV2,267、recurrent1,143；[Qwen3模型文件](https://github.com/ggml-org/llama.cpp/blob/f2cc7282ce0042641df55e0e563b75e824e2b647/src/models/qwen3.cpp)
  只有120行却依赖大量共同engine/backend machinery，不能用单模型文件估完整引擎。
- vLLM 固定 cc9b0f09 的 [scheduler](https://github.com/vllm-project/vllm/blob/cc9b0f097d49abfa6fc94d1d773dca1cc2b9619c/vllm/v1/core/sched/scheduler.py)
  3,108非空行、block pool789、single-type KV manager2,500、FlashInfer adapter2,664；
  SGLang固定e6072935的runner2,215、memory pool5,583、FlashInfer adapter2,364。
  这些文件混合策略、模型种类与已有基础/runtime，不是可直接移植的净engine行数；不按整个仓库扣除HYLE代码。
- HYLE已有 CPU+CUDA inference片段100非空行、Qwen文件/资源consumer340、公开effects consumer180；
  它们分别证明局部接线，不含完整生成模型。623行的 inference_service 样例运行mock model，不能当真实引擎。

F1–F5/A能移走反复手写的资源寿命、异步范围搬运、文件管线、集合版本和batch native planning
glue；现阶段没有goal前后同一engine的实测代码，故不给“减少了精确多少行/百分比”的伪统计。

## 如何把代码量变成基础库完善度验收

用户进一步确认代码量应作为基础库完善度的一维：在**同一固定引擎目标与能力矩阵**下，
支持更多具备稳定共同语义的通用机制，可以降低引擎重复实现的产品行和平台适配行。
F1–F5/A正是这种方向，且不必给普通调用者增加未选能力的成本。
但8,000–16,000与14,000–30,000是不同目标档位，不能据此得出基础库扩展后引擎LOC上涨。

可下降部分包括：last-use/退休记录、共享与独占权限、范围copy批次/槽复用、文件到resource的
背压与错误排空、显式版本目录/clone/发布、metadata版本/batch plan协调、重复native布局接线。
模型layer/权重命名、tokenizer模板、位置/采样、KV/prefix/batch/admission选择仍属engine。
自然普通接线也不会变成零行；单纯把完整引擎第三方化虽然让本仓库更短，不能证明HYLE更完整。

同scope的净效果应比较：移出的重复机制行−新增绑定/适配行，并同时核对隐藏成本、失败、
取消、旧结果、三交付与公共依赖。两个consumer反复自写同一寿命/权限/批规划就是可准入证据；
大量行仅来自不同模型策略或参数表，不构成基础库MUST。更通用但更难消费的接口可能增加glue，
因此不能保证每新增feature都会降低每个engine的总行数。

当前没有同一真实engine迁移前后基线，不给固定减少比例。第一次prototype应锁定功能目标并
保存代码分类/基线；随后新增第二模型、第二route与single/multi-request分别记录增量。
若领域代码复用而通用glue显著下降且hotpath成本不恶化，才能量化证明这一维的改善。

除了总产品行，后续真实引擎应分别记：领域行、HYLE普通接线、重复通用寿命/权限/批规划glue、
为绕过缺能力的layout/materialize/platform workaround、借第三方整引擎runtime的行。
重点应是后两项的需求和成本，而非把自然模型代码清零。

第一步选择一普通Transformer和一recurrent/conv模型，各走加载→prefill→多步decode→流式输出→
cancel/close及独立模型数值参照；第二步迁到另两种native路线，统计新增glue并保持同消费者合同。
如果模型代码基本复用，而第二种模型/路线仍要重写相同ownership/retirement/batch glue，
这才是新基础库准入证据。是否存在此问题须prototype验证，不能仅凭此次估计关闭所有engine blocker。

本次只增加研究记录和输入尺度，不改变活动goal的16项ledger，不启动整引擎产品或新增owner。
到goal收口时按最终真实支持矩阵再校正此条件估计；F4/F5未实施的当前结果不能被写成已完成。

2026-10-13阶段实测更新：F5产品对F4的native净增2610行，compute_batch/graph_batch含公开头
分别1437/400；F5自有native测试净增1862，测试登记/本轮工具修复净增5。口径和固定输入见
[LOC清单](../../../../evidence/architecture/foundation-f5-loc-inventory/README.md)。18 SDK及普通公开
consumer已真实取证，CPU/Apple/Vulkan参考与Linux FI路径的额外copy/wait/prepare账见
[P0结构](../../../../evidence/performance/foundation-f5-cost-structure/README.md)。这些是基础库实测，
不是同一engine迁移前后差值，也不证明14–30k区间已经被实际engine校准。最终收口后仍须
以具体单机高性能目标逐类核残留glue、Provider资格和未验证模型接缝，不能直接减去2610。

用户追加要求：当前goal完成后必须统计一轮，不停在本文条件预测。验收使用最终registry/
candidate/actual consumer证据，按单机高性能单/多请求目标逐项分列：已通过的基础机制、
尚缺通用机制、未验证消费接缝/平台Provider、合理的引擎策略。汇总实际适配代码/额外copy与
wait/planning成本、明确剩余基础库owner/工作包和代码量估计的可信边界；不给未经prototype
校准的精确代码量下降百分比。不为统计本身启动全库构建或扩完整推理引擎。
