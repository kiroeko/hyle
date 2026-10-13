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

## 本次边界

本次只读复核已核对上述公开合同、实现导航与固定 peer 调用，未修改产品/registry、未启动
构建、性能或模型运行。没有签发新的 native/模型 PASS。尚未证明全部组合，尚未完成本页
提出的动态验收；本页给出补扫边界和交接规则，不是十项已完成的证书。
