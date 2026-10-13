# 单机推理基础库完整性与收口终点

状态：**只读补充审计与收口提案**，2026-10-13。输入为当前产品
`9087db46c5ce13038d53680c1d1ce23cb2453be5`、[原十八旅程族审计](inference-single-machine-foundation-audit-2026-10.md)、
[消费适配方案](inference-consumer-adaptation-plan-2026-10.md)、
[Qwen3.5 实物范围](qwen35-2b-command-consumer-scope-2026-10.md)与
[G01–G10 执行总卡](inference-gaps-goal-execution-2026-10.md)。本页不改变 registry、不授权产品重开，
也不替已有 G 项或未运行门签完成。

**目前没有定位到十项之外、已经证明必须新增的独立基础机制。** 原十八旅程族已经覆盖加载、
数学、状态、混合流水、内存、locality、关闭与交付；剩余风险既有 G 项的子任务，也有组合和
资格尚未证明的部分。不能据此宣称绝对穷尽，也不能把每个 UNKNOWN 重新命名成 G11。

收口应证明一组固定消费合同已经完整，而不是承诺以后再无缺陷。保留全部 G01–G10，做一次
补充扫描和一次疑点复核，冻结施工清单；完成全部已准入项及适用资格后交接引擎。完整模型的
token/s 与模型政策仍在引擎阶段验证，不能成为永远阻止基础库交接的循环前置条件。

## 已有覆盖与真正待定之处

部署范围保持单 CPU、一个 GPU 主数学、CPU 与一个 GPU 真数学联合，单请求与多请求均保留。
路线保持 CPU、CUDA、Apple、Vulkan、CPU+CUDA、CPU+Apple、CPU+Vulkan；OS/native 资格
分别记录，不能用 CPU fallback 或另一 OS 的 PASS 填格。三种部署不是要求相同原生对象、
capture、私有分配器或完全相同的性能。

| 疑点 | 当前可定位事实 | 分类与处理 |
|---|---|---|
| 全进程内存与私有池是否全部可控 | [ResourceBudget](../../../../include/hyle/compute/resource_budget.h)明确约束 managed requested bytes；私有上界/未知 owner 单列。driver、其他进程不在该限额内 | 原族15的边界与验收，不是已证缺少“总显存绝不 OOM”机制。engine 负责容量、安全余量和 admission；若已声明账漏报实际 owner，才形成真实缺陷 |
| 编译峰值、变体及并发 invocation 是否会失控 | [Graph storage limits](../../../../include/hyle/graph/executable.h)已有 compilation resident/peak、compiled family、live invocation 上限与 UNKNOWN 拒绝；[ProviderCompilationStorage](../../../../include/hyle/compute/prepared_operation.h)声明 prepare/failure scratch 范围 | 原族09/15已存在机制，组合压力待核；不能仅因某 native 上界未知再造预算层 |
| CPU 混合核、cache、NUMA 与 locality | [CPU](../../../../include/hyle/platform/cpu.h)已有动态可用 CPU、P/E 类、cache class、topology、quota、frequency；[NUMA](../../../../include/hyle/platform/numa.h)与[Thread](../../../../include/hyle/platform/thread.h)提供明确能力/失败语义 | G10 与原族14。未知 native pool、进程 setter、线程亲和及每阶段策略须在 G10 定界；不是新 G11 |
| 跨设备等待、copy/compute 是否真的重叠 | 原 F2 具有 producer/native waits；G02/G03 接精确页和更新依赖，G09 验独立支路与 microbatch | G02/G03/G09 和原族05/10/16。调用名 async 或 pending 数量不能证明 overlap，也不能由必要 CPU→GPU 数据依赖推导机制缺陷 |
| 长期 decode 是否仍有热分配、全历史转换与同步 | [ProviderRouteCostFacts](../../../../include/hyle/compute/prepared_operation.h)已有 copy/repack/sync/dispatch/workspace 事实，Graph report 保留未知；G01/G06 已针对物化和 metadata/plan | G01/G03/G06/G07/G09 的热成本验收。若现有事实与真实账冲突，回真实 owner 修复；不默认重写 profiler |
| 单算子对拍是否足以保证全层数值 | [operation schema](../../../../include/hyle/compute/operation.h)与[Compute 用法](../../../user/compute.md)声明精度、累加、NaN/舍入与有限 route 域；组合会累积误差 | 原族03/07/12/13/18：库片段需独立 oracle；完整24层/model logit 漂移仍由引擎整链对拍校准。未做整模型不是已证新数学缺口 |
| 运行观测是否需要额外产品 | [CompileReport](../../../../include/hyle/graph/executable.h)、runtime_properties、资源 usage 与 native 环境身份已提供冷诊断；实际 trace 可借 native 工具 | G06/G09/G10 和原族14/15/18。先补成本/身份/trace 证据，只有普通消费确实无法发现承诺的必要事实，才考虑窄观测 seam；不为审计制造全局仪器 |

这些事实说明机制存在与边界可见，**不证明当前全部组合已经通过**。当前 G01 的 CPU/Vulkan
有限驻留 candidate 与 G02 common candidate 也不能代表 G01/G02 全域完成，更不能覆盖 G03–G10。

固定 peer 源码保持 SGLang `7224fce1b5e488c9916f20ec0556d696e595d512`、vLLM
`98f721116227f0f48690deb05c815c057bac0668`、llama.cpp
`cbe28bb3cb0758f0c566a9a3561766821aeb0748`，归档身份在
[原 source manifest](../../../../evidence/architecture/inference-single-machine-foundation-audit/source-manifest.json)。
本次复核不移动参考 checkout、不追最新 feature。

固定 vLLM 的 [GPU memory profiling](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/worker/gpu_worker.py#L634)
测 activation/capture 与可用量，明确依赖其他进程不改变内存的前提；
[CPU sizing](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/worker/cpu_worker.py#L204)
在 warmup 后读取 node 可用量和进程 RSS。这证明 profile/admission 属于消费者整链工作，
不是统一 requested quota 可以保证全系统不 OOM。
[prefetch](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/model_executor/offloader/prefetch.py#L283)
用每层 event，也明确特定情况的 stream wait；
[ggml async copy](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/ggml/src/ggml-backend.cpp#L651)
会在缺原生能力时同步 fallback。要求应落到实际等待和生命周期，而不是求这些 API 的并集。

## 一次有界补充扫描

沿原十八旅程族×七路线矩阵补齐导航，不重新生产126套测试。每格回指一个活动 G 卡、既有
terminal/evidence、明确负空间或待核反例；共有机制可复用证据，native/OS 特性只在实际适用
格验证。保未证明状态，索引覆盖率不能代替正确性或资格。

追加扫描只聚焦下面五组交叉反例。它们是既有族的验收组合，不是五个新施工项；已有活动卡
已覆盖的用例直接引用，不重跑。尚未覆盖者在相关 G owner 定界时加入有限验证，独立文档
维护不触发产品 reopen 或全库构建。

| 交叉检查 | 固定反例/事件 | 回映与通过依据 |
|---|---|---|
| 资源压力与原生退休 | 接纳前 quota 不足；accepted producer 失败/取消；保旧结果关闭；partial compile/prepare 失败；两个 slots 与别名共享 backing | 族01/15/16/17，F1–F4及G02/G03/G09/G10。已知账只收费一次、错误不发布成功、原生退休后才释放；真实 private 未知保留。受控 quota/fault 替代把桌面推至真 OOM |
| 冷启动与有限 shape family | general+已准入 exact variants；shape 切换；同形不同布局；一个 prepare 失败；编译到 invocation 的限额分界 | 族02/09/15，原A及G06/G07。报告与生命周期一致，warm 不暗中重新 JIT/repack，UNKNOWN 按声明政策处理 |
| 持续热复用 | 原方案至少64步：tail、跨页、两请求重排/加入/结束、混合 prefill/decode、invalid capacity sentinel、旧输出保活一次 | 族04–09/16/17，G01–G07。独立数值/有效区、dirty bytes、热分配/plan/wait 账；不能复制完整历史或只验 N=1 |
| 跨算子数值链 | 固定 dense/GQA 的 norm→QKV/RoPE→驻留 attention→gate/projection；recurrent 或 MoE 留出链；选定低精度与输出选择 | 族03/06/07/11–13/18，G01/G04/G05/G07/G08。独立 oracle/精度边界，给累计误差而非简单逐bit一致。短片段不替完整模型质量或 token/s |
| CPU locality 与混合流水 | 普通 public CPU→GPU→CPU；独立 CPU 支路；两 slots/microbatch；prefill/decode contexts；有界 CPU executor+native pool；actual copy/compute trace | 族10/14–16，G02/G09/G10。真实 CPU数学与GPU数学、不用fallback填资格；预算/身份可发现，必要等待与多余host观察分别计费，UMA 不按两个独立物理池相加 |

每个仍有疑点的格只记录五件事：具体 public 调用、真实 owner、最小反例、归属裁决、所需
最小验收。源码缺信息先标 `UNPROVEN`；不得把没有跑的门写 PASS，也不得把 UNKNOWN
直接转成新增 MUST。建新 API 前先反证现有 public/native 路线与成熟承载。

## 冻结与退出协议

1. **补扫一次**：固定上面的产品/peer/模型范围，对原矩阵和五组交叉反例完成回映；每个疑点
   归到 G 子任务、已有完整路线、机制反例、资格缺失、引擎整链或二期。扫描不持续刷新参考库。
2. **疑点复核一次**：只复核补扫留下的有限疑点，用自然 public consumer 与源码反证；保留
   “现有路线已完整、不需要改”的结论。不再发起新的全库特性目录或“再看一遍”。源码无法
   裁决的资格格进入其适用验收，不能靠追加研究消耗代替实际验证。
3. **冻结有限施工清单**：保全部 G01–G10；在每项进入实现前冻结 owner、精确编码/数学域、
   输入/权限/依赖/错误/关闭、成本目标、normal/holdout、适用 OS/native/三交付与风险门。
   未定 G04/G05/G07/G08 域必须作真实有限选择，不能通过标“模型不需要”取消这些 G 项。
   新增独立机制只有具体反例通过下面准入门才增加；未知费用与未选优化进入限制账。
4. **施工与验收**：按原 SCOPED→AUDITED→CANDIDATE→CLOSED，只运行变更影响及预声明
   适用门。真正的验证失败持续处理到裁决；“只复核一次”限制重复扫描，不限制修 bug 的
   合法验证，也不把仍未通过的项强制关闭。时间追加保原失败和累计投入。
5. **基础库交接终点**：十项全部有真实 FIX/BORROW 完成或完整 ROUTED 反证；所有已准入
   路线格有数学/有效区、权限/producer、错误/取消/退休、三交付、结构成本及适用局部性能
   verdict；相关 owner 有 terminal；矩阵没有影响该合同的未裁决缺陷，限制/UNKNOWN 有明确
   范围，普通 C++ 调用与费用账可交接。此时结束一期基础库施工，进入引擎，不追加第三轮
   全局调查。不能用先生成一个 token、当前候选数量或减少 G 项来达到这个终点。

完成的是**固定合同的功能和工程资格**。数学不变的原生 capture、驱动内部存储和各设备
吞吐可以不同；原成本/资格不足若落在已准入必需合同内必须修复，若是未选域、二期或引擎
政策则保留明确边界。不能使用“仍可能存在未知问题”否定一切已签证据，也不能使用
“清单冻结”掩盖已知的一期错误。

## 后续发现的准入门

冻结后出现的发现先保最小反例，再决定是否影响库交接；不要自动增加下一 G 项。

| 发现 | 当前动作与停止影响 |
|---|---|
| 已准入旅程错误结果、越界/寿命/退休错误、必要 public 路线确实不可完成，或已冻结结构成本目标被反例推翻 | 定向回真实 owner，按协议回退/重开有限 FIX，阻断该格交接。不是重新调查整个HYLE |
| 既有 G 项的漏验事件、编码或正常域子任务 | 合并该 G 卡有限清单并依法回退，不制造新独立机制项；不能靠改名排除原目标 |
| 仅缺实际 OS/native/performance 证据 | 进入对应适用验证，当前格保未证明；不是由证据缺失自动推导 API 缺失 |
| private/driver 费用未知但原合同允许未知、完整模型瓶颈尚无 trace | 保限制与具体补证位置；先用引擎 trace/原生工具校准，不能签全局安全保证，也不新增万能预算/profiler |
| 新模型、新编码、未准入 attention 特例、驱动新版本、二期多GPU/多机，或纯 batching/cache/sampling政策 | 进入后续需求账；不阻断现有固定合同交接。已有承诺若覆盖了该反例，则仍按第一行处理 |
| 更快的 peer、局部微优化、便利包装 | 只有同语义/生命周期的合格证据表明已冻结 material 目标不满足时才回 owner；单次慢点或符号名不能触发重开 |

新独立机制的最小准入记录是：一期具体旅程与可复现反例、当前自然 public/native 路线为何
不足、真实 owner 与成熟承载、语义/生命周期/成本影响、normal 与正交留出验收。一个真实
反例可以成立，不机械要求已有两个完整产品；纯应用政策和参考库特例求并集不能成立。

交接后引擎完成加载→prefill→多步decode→小结果输出→取消/关闭整链、模型质量与
token/s 对比；新增真实库缺陷以候选/owner/trace/最小复现定向回流。这个反馈环属于正常
产品维护，不撤销“上一固定合同已收口”，也不意味着必须在写引擎前预先证明所有未来模型。

## 接口与实际消费的距离

当前距离不是均匀的：**普通加载、冷准备、数学及显式关闭较小；持续更新、多请求和原生
联合分页仍较大。** 不能用“有 Graph/lease/attention API”证明这些链自然，也不能将所有
模型代码都当接口缺口。下面评估自然公开组合与附加机制成本，不能替 native 验收签 PASS。

分级为小（已有直接路线或少量 ordinary connection）、中（可用但需显式多个对象/阶段与
成本管理）、大（公开组合或目标数学/存储域尚缺，不能用上层薄 wrapper 补齐）。
`Q` 表示还需要实际资格；数学/存储域不足与易用性不足分别记录。

| 消费旅程 | CPU | CUDA | Apple | Vulkan | CPU+CUDA | CPU+Apple | CPU+Vulkan | 主要距离 |
|---|---|---|---|---|---|---|---|---|
| 文件→有界stage→ready权重/资源 | 小 | 小 | 小 | 小 | 中 | 中 | 中 | F3已有完整ticket/ready group；联合只是选择资源/放置，不应重复读/拷贝/退休状态机 |
| 冷prepare/compile→基础prefill | 小/中 | 小/中Q | 中Q | 中Q | 中Q | 中Q | 中Q | shape/encoding、native资源与各lane需声明；模型图和权重角色由engine写。新数学域不足另归G05/G07 |
| 单请求驻留attention | 小Q | 小/中Q | 中Q | 小/中Q | 中Q | 大Q | 大Q | CPU/Vk G01有限stride驻留已有candidate；CUDA需分OS资格；Apple原dense/native存在，长期有效长度及warm组合仍待证；typed hybrid分页接缝未补 |
| KV producer→增量更新→读取 | 中/大 | 中/大 | 大 | 大 | 大 | 大 | 大 | 原F1/F2 writer/producer已有，F5只接ready页；长期更新、input imports释放和少host边界尚属G03/G06 |
| 两请求变长/重排→多步decode | 大 | 中/大Q | 大 | 大 | 中/大Q | 大 | 大 | CUDA native paged/ragged存在；CPU/Apple/Vk参考物化可算，但不满足G01结构目标。共通协调已修不代表native数学/metadata全域完成 |
| recurrent状态→成功发布/分支 | 中Q | 中Q | 中Q | 中Q | 中Q | 中Q | 中Q | F4已有version/draft/head；哪些conv/recurrent成员组成完整checkpoint是模型代码，原目录O(E)不能当每token默认路径 |
| logits→必要小结果 | 小/中 | 中/大 | 中/大 | 中/大 | 中/大 | 中/大 | 中/大 | greedy/TopK现有路线与随机完整域分开；G08设备数值选择和少D2H尚待补。EOS/temperature/penalty/grammar是engine政策 |
| 错误→取消→旧输出/退休→关闭 | 小/中 | 小/中Q | 中Q | 中Q | 中Q | 中Q | 中Q | 每个原owner已有错误/取消/retirement合同；跨多个owner仍需有界scope和明确关闭顺序，不能在失败时destroy资源冒充drain成功 |

表中“大”主要回映当前 G01–G10，并非本次新增项。CPU+CUDA 的 G02 common 已将
[graph batch](../../../../include/hyle/graph/batch.h)改为完整 `LaneBinding`，各native batch slot
使用自己的 Queue/host workspace；原适配文档中的“仅单Queue”是修改前事实。
[公开联合消费者](../../../../tests/compute_batch/public_hybrid_batch_consumer.cc)现在直接采用
`prepare_batch→bind Q/精确页/输出→submit→wait→output`，无需自写页guard/native退休。
但 [CPU+Apple](../../../../include/hyle/graph/cpu_apple.h) 与
[CPU+Vulkan](../../../../include/hyle/graph/cpu_vulkan.h)的typed hybrid `bind`仍只有native
tensor、ready lease与weights，没有 `PagedArenaRead`；不能拿common修正代签这两条组合。

### 公共对象序列与成本

普通消费可先分成四个长期owner：运行资源（Context/Queue/allocator/IoContext）、不可变
program/executable、有限invocation slots、每次提交的ticket/completion/result。运行资源
必须包住所有accepted work与结果，这是必须兑现的borrow合同，不是应隐藏的细节。

| 阶段 | 已有自然公开序列 | 消费者必要工作与当前危险接缝 |
|---|---|---|
| 加载/放置 | [StreamSource/StreamFile→FileResourcePipeline→StreamTicket→StreamReadyGroup](../../../../include/hyle/compute/file_stream.h)；ready output再bind数学 | engine解析权重文件/tensor角色，列实际范围和放置；pipeline已管stage/backpressure/取消/accepted prefix。不要为每个模型重写read/copy retirement，也不要每组等待全部文件完成 |
| 冷数学 | `TensorDesc→prepare/compile→workspace与Invocation→bind`；[LeasedOperationInvocation](../../../../include/hyle/compute/leased_operation.h)自动保accepted lease | engine列shape/算子与模型位置；资源、artifact与slots应复用。跨native迁移改typed资源/compile/binding，保数学schema；不假装所有backend一种对象 |
| 页更新与读取 | [PagedArena.acquire→writer→F2 bind_mutable/bind_producer→成功output→writer.complete→arena.read(layout)→batch](../../../../include/hyle/compute/paged_arena.h) | 当前ready-only组合要求host观察；engine不得自行把pending改ready。共享页/旧result防写保护必须保持，必要clone只针对dirty范围。受控pending组合归G03 |
| 每批规划 | [BatchInvocation.update_batch→PreparedBatch.run→Completion→output](../../../../include/hyle/compute/batch_execution.h) | [prepare实现](../../../../src/compute_batch/batch_execution.cc)每批copy完整metadata→upload.wait，再规划wait；reuse通常仍一个host wait，未复用两个。`update`名字不承诺delta或零等待，G06负责费用纠正 |
| 结果和关闭 | `OwnedResult.wait/ResultLease→必要读取→释放旧reader→close/drain`；随后再销毁runtime资源 | `Completion`成功/失败和actual retirement不同；可持结果跨下一业务步骤，但slot复用会拒绝。不同owner的close语义按头文档，不能一律当非阻塞cancel；非阻塞析构也不等于可释放borrowed资源 |

[F2 report](../../../../include/hyle/compute/batch_transfer.h)已区分logical/physical/staging bytes、
native wait与host wait；batch report提供generation、metadata bytes、native-plan reuse与
host waits。这些足以构成现有成本账入口，不需先扩建公共仪器。
原 Apple hybrid [fragment推进](../../../../src/graph_route_cpu_apple/invocation.cc)逐段await并
在CPU/GPU边界搬运，必要依赖正确，但独立支路是否多付wait归G09。复制schema/owned
metadata、K/V capacity展开、F16/BF16转换、权重冷repack、native private scratch必须分别
计费；一层 wrapper 若仍隐藏这些费用，不算体验改善完成。

### 消费者代码量的源推估

以下是**接口连接量的设计估计，未由同目标引擎前后LOC实测校准**；置信度低到中。只计
consumer自有产品代码，排除测试、oracle、准备工具、成熟tokenizer/格式库与HYLE源码。
范围受选域和native资格影响，不能把基础库新增行数直接从引擎估计相减。

依据是上面实际公开对象/调用序列，以及
[普通F3消费](../../../../samples/package/graph/file_streaming/main.cc)、
[F2与leased数学消费](../../../../samples/package/graph/batched_transfers/main.cc)、
[G02公开联合分页](../../../../tests/compute_batch/public_hybrid_batch_consumer.cc)。这些fixture
含oracle、多route分支和负例，格式有压缩，原文件总行数不能直接当引擎glue实测。

| 自有代码类别 | 当前源推估 | 依据与预期取舍 |
|---|---|---|
| 单route普通连接：resource/context、冷prepare、绑定、错误返回与scope关闭 | 约150–400行/一个复用runtime适配域，中置信度 | 通常10余个冷对象/配置动作和每step数次bind/submit/结果动作；没有理由每层复制完整适配器。属于合理连接，不应全下沉 |
| 多请求更新/页权限/metadata与slot复用的重复通用协调 | 约300–900行/一套有界协调，低置信度 | writer→producer→ready、layout generation、旧reader阻止reuse、failure suffix与close的数个状态分支；目标是G03/G06把通用机制收进owner，engine留下页选择/请求策略 |
| CPU+GPU跨typed原生路线的适配与失败/退休组合 | 约300–800行/一个native联合适配域，低置信度 | complete bindings/资源/placement、各片段结果、copy dependency、取消与runtime borrow。G02common已降低CUDA组合负担；Apple/Vk未补的数学/权限不能以这点LOC冒充可实现 |
| 明知无public目标数学/存储域而另写kernel/backend的workaround | 不给“glue上界” | G04/G05/G07/G08域不足是基础承载工作，不是几百行consumer连接能解决；必须在已授权G项内真正补齐或证明完整ROUTED |

前三行存在重叠，**不能逐route相加当总引擎报价**。以三部署、有限模型和七路线的一套
复用实现规划，当前通用平台/生命周期glue可先按约1500–4500产品行作低置信度风险账；
十项完成后的合理目标是让剩余部分主要为约600–1800行的资源/route适配和声明式接线。
这只是待prototype校准的设计目标，不是承诺减少了某个百分比，也不证明其他所有机制已具备。
如果补齐后仍要自写native retirement、页permission或每route一套batch错误状态机，应判为
接口消费验收未过，而不是合理模型代码。

模型结构/权重映射、位置与完整状态集合、自回归/request循环、prefix/LRU/admission、采样
政策、tokenizer/文本协议是引擎应写或借成熟库的部分。首个限定CLI与实用多请求引擎仍沿
[原条件估计](qwen35-2b-command-consumer-scope-2026-10.md#5-工作量粗估)：2500–6000与
14000–30000产品行，尚无整链校准；这些总数含普通连接，不能再叠加上表。易用性目标是
减少错误易发的重复机制和昂贵workaround，不是让整个引擎只剩main函数。

### 通用性与实际体验的平衡

1. **先保持原子自然组合**：已有F3 ticket/ready、F2 producer/native wait、F1权限、F4版本与
   原Graph/route的完整生命周期直接用，普通资源声明和少量bind不另造wrapper。recipe应
   呈现cold一次、warm复用、失败/关闭与成本，不仅给成功的十行demo。
2. **重复机制才补窄复合接口**：当推理主纵切和资源流送/批处理数值正交consumer都重写
   同一owner/producer/精确范围/失败退休协调时，在原owner补有界组合；暴露范围、依赖、
   ready与retirement事实，不能暗含token/page淘汰政策。两个consumer是反偏置设计证据，
   一个真实可复现缺陷仍可准入，不机械等两个完整产品出现才修。
3. **保原生能力与共同合同**：共同数学、有效区、权限、错误/取消/结果可统一验收；MLX
   array、VkTensor、CUDA Queue/stream及capture差异保typed入口。不能为统一调用降到
   最慢参考档、隐藏CPUfallback，或建立第二套scheduler/allocator。
4. **体验有具体判据**：普通用户从指南能发现route与已准入shape/编码；冷prepare能给
   selected Provider与成本/未知；同layout的绑定/producer状态和错误阶段可诊断；scope
   的borrowed资源关闭顺序明确；warm复用费用可查；迁移route保模型图，改有限资源/绑定。
   暂无证据的资格需明确失败或未签，不让用户通过大量bench或private源码猜支持域。
5. **避免“一神接口”**：不添加兼容转发、多轨旧API或万能Engine/Sampler；用户本轮已
   不要求兼容。若recipe解决发现问题就只补recipe，若公开机制真实缺失就修G owner；
   不能把重复依赖/退休glue归“上层自己组合”，也不能把tokenizer/模型政策下沉来刷LOC。

接口消费章节的退出依据仍是十项及已准入旅程完成。source估计与设计目标不产生新的产品
授权；剩余glue的真实量在一个同目标prototype上计数并标职责，届时有反例才定向回owner。

### 已识别的接口形状调整是否在当前goal内

**是：目前已识别、一期必需的形状调整均由当前G01–G10承接。** “没有定位到十项外新
独立机制”不表示现有接口形状都不用改，也不允许只补测试、recipe或薄wrapper代签完成。
G02已将单Queue入口改成完整LaneBinding；G03仍需producer/精确写域/原生依赖/成功readable
的受控组合；G06仍需host-known/device-produced、active/delta更新、完整topology/plan复用
形状；Apple/Vulkan联合分页的准确页权限与generation属于G02/G01 native桥。G04/G05/G07/
G08按准入编码/数学的有限payload、scale、layout或输出端口定界，G09/G10按实际流水/预算
consumer定界。该goal的完成条件包含这些真实FIX/BORROW或完整公开ROUTED反证，不只是
“能算出结果”。实现前仍按真实owner固定有限API/寿命/成本，直接迁移当前consumer，不保
用户已明确不需要的旧接口兼容。未来具体反例若证实清单漏项，按上面的有限准入门处理。

## 本次边界

本次只读复核已核对上述公开合同、实现导航与固定 peer 调用，未修改产品/registry、未启动
构建、性能或模型运行。没有签发新的 native/模型 PASS。尚未证明全部组合，尚未完成本页
提出的动态验收；本页给出补扫边界和交接规则，不是十项已完成的证书。
