# 单机推理基础缺口补齐执行总卡

状态：**执行中，首批定界**，2026-10-13。用户明确要求先补前次统计的缺口，并在清除旧 goal
后要求创建新 goal；本轮覆盖 G01–G10 的修正、成熟承载接入或已有完整公开路线的反证验收。
不能以延后、引擎自己写胶水或一个小模型出字代替完成。

baseline：`5d46db5fd99322e2f75d3fe07553c61fadff8eb9`。创建时工作树干净，64 个 registry owner
为 CLOSED；原有限验收与失败/未知费用保留。总卡是调度入口，**不代替 owner 的活动收口卡**，
也不授权在 CLOSED 状态写产品代码。模块真实状态仍只看
[module-status.json](../../module-status.json)。

用户本轮补充执行偏好：**不保旧接口兼容、增加合理编译并行、时间不要限制过死、治理轻量**。
直接设计最终 API 并迁移仓库现有消费者，不添加旧签名转发/双轨实现/兼容层；必要旧入口
可以删除，权限、数值、失败与退休合同仍需兑现。批次共用一张短 owner 卡与实际验证记录，
不为每个函数/测试另建文档或审批；原状态机与选择器用作最小授权/影响核验，不扩建治理工具。

## 输入与完成目标

- [十项源码审计](inference-single-machine-foundation-audit-2026-10.md)：固定三 peer、七实际路线、
  十项发现与反证；[输入证据](../../../../evidence/architecture/inference-single-machine-foundation-audit/README.md)。
- [一期收口与引擎启动方案](inference-consumer-adaptation-plan-2026-10.md)：消费形状、有限包、
  原子/组合/政策边界及局部/端到端性能分界。
- [Qwen3.5-2B 实物与消费者范围](qwen35-2b-command-consumer-scope-2026-10.md)：dense FFN＋
  hybrid attention，已有片段与本轮外部参考；完整 HYLE 模型执行尚未签发。
- [完整性补充审计与收口终点](inference-single-machine-completeness-and-exit-2026-10.md)：
  复核原18旅程族的横切风险，区分G子任务、组合/资格、真实新机制与引擎/二期；补扫一次、
  疑点复核一次后冻结有限清单，保留完整G01–G10与全部已准入域，不以继续研究替代验收。
- 三 peer 固定身份仍为 SGLang7224fce1b5e488c9916f20ec0556d696e595d512、
  vLLM98f721116227f0f48690deb05c815c057bac0668、
  llama.cpp cbe28bb3cb0758f0c566a9a3561766821aeb0748。

目标覆盖单机单 GPU、一个 CPU＋一个 GPU 真数学联合、单 CPU，单/多请求及 CPU/CUDA/
Apple/Vulkan 的实际 OS/native 资格。共同验收位置、有效区、shape/encoding、权限、producer、
错误、取消、旧结果与 actual retirement；保不同 native 对象、plan、capture 和成本事实。

不纳入完整引擎/server、tokenizer/模型格式产品、请求/cache/采样政策、多 GPU/多机二期。
整引擎 prefill/decode token/s 在完整引擎后验收，不作为本 goal 的循环前置门；结构 P0 与
已冻结同合同片段的定向 P1/P2 仍适用。具名模型片段/命令消费者可作为验收入口，不把
外部 Transformers 执行当 HYLE PASS。

## 十项跟踪

下表是当前 goal 工作项，不是第二份模块 registry。完整项完成数 **0/10**；已完成
G01 CPU resident/native ragged有限轮、Vulkan单请求有限轮与G02 common实现，不能由
子批替代全部路线。Apple native ragged有限轮已CLOSED；Vulkan原280短D64合格性能
FAIL保留；原region分段修正候选0e9e011b8随后同合同P1也失败。
0e9失败原因为固定记录只接单flash，合并图反而每次managed编码。RF-04已复现/修正，
最终bc0的64步记录/失效、三交付/三san/原消费者及同合同fresh4 P1全部PASS、verify0；
Vulkan有限NR/RF轮已CLOSED，原两次FAIL仍保留。整个G01分页等范围仍未完成。

| 项 | 当前状态 | 有限目标 | 预计涉及的机制 owner，需在活动卡精确定界 |
|---|---|---|---|
| G01 高效驻留读取 | CPU973、Apple280与Vulkanbc0有限轮均CLOSED。Vulkan最终原1region＋安全记录通过十source tests/两SDK/三san/原联合回归及Win定向编译，multi P1 QUALIFIED/5%非劣PASS、verify0；mixedD256 WIN，prefill/D64方向INCONCLUSIVE。原两次qualified FAIL与全部限制保留，整个G01其它准入域与分页未完成 | 单请求先核 dense/view；多请求接 native/等价直接驻留消费，避免按每 Q/head 重复物化全 KV | compute_batch、原 route/Provider、graph_batch 必要桥 |
| G02 联合分页 | common与CPU+CUDA实跑、三交付/宿主风险通过；Windows余门及Apple/Vulkan联合分页待验 | 原 multi-queue bindings 接 batch coordinator；native hybrid 接准确 page/layout 与原 region/slot，公开 CPU→GPU→CPU | graph_batch、三个 graph hybrid、原 compute/native 权限桥 |
| G03 增量 producer 组合 | 待 owner 定界 | 原 exclusive/F2/native waits到更新与读取；精确 dirty 域、成功 ready、idle input 生命周期 | compute_batch、compute/F2、原 native route与graph桥 |
| G04 量化 KV | 待有限编码选择 | 按成熟实际 cache 编码冻结 payload/scales/layout、quantize append与直接读取，不每步解码完整历史 | compute_ops/batch、实际 Provider/native route |
| G05 attention 语义域 | 待有限数学选择 | 固定真实消费者所需 window/mask/维度等域及失败边界，成熟承载；未准入未来特例不无限扩张 | compute_ops/batch、实际 Provider/native route |
| G06 metadata/decode | 待 owner 定界 | host-known/device-produced分界，active/delta上传、完整 topology复用与decode档，实际capture资格 | compute_batch、FI/native Provider、graph_batch |
| G07 packed 权重 | 待有限编码选择 | 锁定 artifact→resident→selected math，借原生packed/冷repack；checked IDs保错误，排除warm完整展开/多余host检查 | 实际低精度 Provider/native route及资源桥 |
| G08 数值输出选择 | 待有限数值域选择 | row-wise greedy/selection与选定随机数值域，counter/state、质量/分布和少量结果返回 | compute_ops、实际 Provider/native route；模型采样政策不下沉 |
| G09 联合流水 | 待最小consumer | 原有限slots/依赖推进，真实无依赖CPU/GPU支路及microbatch trace；不重复scheduler | 三个graph hybrid及必要原native执行桥 |
| G10 CPU预算 | 待最小consumer | executor/native private pools明确报告和约束边界，阶段与并发invocation组合 | compute_route_cpu及实际CPU Provider |

编码、数学与数值域由固定 peer/native 实际支持、当前模型和留出消费者共同裁决。总卡不
预先宣称 FP8/Q8/window/MLA/random 的任意组合全部准入，也不以 BF16 单请求不需要为由
删除上述 G 项。确需调整已冻结域时按 owner 协议回退，不在实现途中扩成所有 peer API。

G01 [驻留SDPA子批](../compute/inference-g01-resident-sdpa.md)的单请求验证已完成：CPU P1
2598.985s、Vulkan P1 3669.387s，均合格且5%非劣PASS；失败/STOP及原合同原样保留。
Vulkan实际port Copy3→0与原生精度/固定record资格见有限terminal；CPU当前owner按
[有限terminal](../compute/inference-g01-cpu-resident-ragged-terminal.md)CLOSED，原flash复用
与packed输出不复制KV；三交付/宿主风险/独立multi P1通过，原失败记录保留。
多请求其它路线与联合分页仍属原G01/G02有限清单，不新增全局调研轮。
CPU公开工厂声明与产品root交付已补齐，普通consumer无需内部target或fake宏，实际
source/relocated static/shared Direct＋两个dependent Graph regions全部PASS；CPU未选
时header/library仍不进入安装。正式CPU multi合同8751dfe5在clean9cfa副本完成，固定
三真实长/短形状、原16 paired rounds与n15/31，A/A16、QUALIFIED/5%非劣与verify0。
当前native批按
[原native ragged卡](../compute/inference-g01-native-ragged-layout.md)固定原View/SDPA/
Program消费而不另建controller；Apple280有限轮已CLOSED，直接联合回归两平台通过。
Vulkan原qualified FAIL已封存并按[有限RF卡](../compute/inference-g01-vulkan-ragged-region-fusion.md)
修冷region分段；新candidate交付/宿主风险与原消费者通过，原同合同P1串行运行。
最终bc0已按[有限terminal](../compute/inference-g01-vulkan-ragged-terminal.md)关闭，原同合同
P1资格/非劣/verify通过；停止重复这一已签域，保两次FAIL和全部原后续范围。
分页/G03–G10和全部已准入域
继续保留，不以本批冷入口签growth/capture完成。

## 推进顺序

1. **首批 G02/G01**：先核 single/multi-queue coordinator、native hybrid bind 与原权限边界，
   固定最小 owner 卡；将可独立签发的 common 接缝与 native 数学承载分开。
2. **G03/G06**：沿选定驻留数学加入真实 producer/dirty 更新、metadata/topology与warm成本。
3. **G04/G05/G07**：每项先冻结一个或有限个真实编码/数学消费档，再接成熟承载；必要 native
   适配属于本轮工作，不用未选/拒绝假装跨路线对称。
4. **G08/G09/G10**：数值选择/随机、联合依赖推进与CPU pools，以具体消费者及同合同费用裁决。
5. **终局**：十项对照输入审计，三部署/七实际路线的已准入格、源码/头 LOC 与剩余引擎
   接线账；所有未支持/未知/capture限制明确，不签全模型吞吐或二期完成。

实现顺序可按真实依赖调整，但不能省略工作项。可复核完整 ROUTED 路线必须给具名 owner/
成熟依赖、自然 public 调用、生命周期、费用/资格和验收；写 convenience wrapper 或“上层
处理”不足以结束。没有真实需要的应用政策剥离后，还须证明其通用机制覆盖。

## 状态、预算与验证

每个实际产品 owner 由本轮新事实创建小卡，合法 CLOSED→SCOPED→AUDITED，再写实现。
SCOPED冻结baseline、触发、主/正交消费者、预算和预声明门；AUDITED固定有限FIX、API/
owner/依赖、identity、失败/完成/并发、冷热成本和negative space。稳定不可变candidate
以后做适用正确性/风险/三交付与定向性能，全部适用门真实裁决后CLOSED。

总卡不是全局 token/时间预算，也不给新 goal 设置用户未要求的 token budget。活动卡的
估计时间作为检查点；预计仍有有效进展的实现/构建/验证自动继续并记录实际耗时，不因
机械超时结束正确工作或反复请求追加。类似超时在同配置/树/合同下自动延长，保原
FAIL/TIMEOUT、elapsed与累计投入，不以新日志目录重置。停止依据是有限问题已裁决、
同一方法确无进展、资源实际耗尽或真实外部阻断，而不是短时间上限。
发生工具/环境问题先查实际根因和替代合法验证；未运行不得写PASS，模块与goal的blocked
裁决分别遵守原协议，不将缓慢或一次失败当成自动停止理由。

验证按[selector](../../../developer/review/change-impact-validation.md)取 owner 与真实直接
消费者。公共头/API/package seam才扩对应门，不每个op全局构建。库依赖按实际CPU和可用
内存提高并行并复用tree/cache；当前Win为24核/32线程、约42GiB可用，普通编译先用12 jobs，
按观测峰值可提高到16或更多；大型CUDA/LibTorch编译按内存实际调节，不套旧j1/j2/j4限制。
链接和heavy运行避免彼此争用，consumer构建也按自身规模选并行；测试/性能合同的运行
并发与编译jobs分开。用户后续反馈桌面卡顿，当前Windows编译上限改为4 jobs，按内存
继续下调；重型GPU/全模型运行暂不恢复，Windows/WSL/Air重型任务串行，已有FAIL和
未完成项照实保留。公共工具、构建系统或
性能统计核心不在模块内无限扩建。

Windows `C:\public\project-hyle\hyle` 是产品真相；WSL/Air仅一次性验证副本，不能反向
覆盖源文件。Apple只用MacBook Air，执行前读取实际hardware/OS/toolchain；Mini不可用。
所有普通轮至少docs/closure及适用验证；registry改变带改前明确Git revision，状态摘要
由原checker生成，保无关既有修改。

## 完成条件

十项有限机制均有实际纠正/成熟承载和验收，或已有完整公开路线的可核反证；每项给出
最终owner、支持域/负空间、三部署路线、正确性/寿命/错误/关闭、结构成本及适用资格。
所有当期适用风险未完成不得通过，所有产品owner应有terminal，真实产品输入固定且
工作树可交接。goal只在这些条件完成时标complete；启动文档或第一次模型输出不算完成。
