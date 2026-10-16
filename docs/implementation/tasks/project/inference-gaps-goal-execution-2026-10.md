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

下表是当前 goal 工作项，不是第二份模块 registry。完整项完成数 **3/10**（G01、G02、G03）。
G01 原 plain 驻留读取范围已按各 finite CLOSED terminal 与原 Linux FI ROUTED/P0
[聚合验收](../compute/inference-g01-aggregate-acceptance.md)；原八份声明 P1 的最终结果已核对，
早先 FAIL/STOP/DIAGNOSTIC 和方向限制保留。G04/G05 新编码与数学域、
G06–G10 不由该聚合裁决代签。

| 项 | 当前状态 | 有限目标 | 预计涉及的机制 owner，需在活动卡精确定界 |
|---|---|---|---|
| G01 高效驻留读取 | [整项已验收](../compute/inference-g01-aggregate-acceptance.md)：CPU/Apple/Vulkan 原 resident/ragged/page16 子批均 CLOSED，八份声明 P1 已核最终通过；Linux FI 原完整公开路线沿用已签 P0，不伪造新 P1。只接受原 plain 有限域，其他 G 项继续 | 单请求先核 dense/view；多请求接 native/等价直接驻留消费，避免按每 Q/head 重复物化全 KV | compute_batch、原 route/Provider、graph_batch 必要桥 |
| G02 联合分页 | [整项已验收](../graph/inference-g02-aggregate-acceptance.md)：原common CPU+CUDA、精确页守卫及Apple/Vulkan native联合域完成；原Windows CUDA804/Vulkan851默认门及补验Vulkan alias/phase的实际Air三交付均通过。按原P0保留metadata/等待/去重和private UNKNOWN | 原 multi-queue bindings 接 batch coordinator；native hybrid 接准确 page/layout 与原 region/slot，公开 CPU→GPU→CPU | graph_batch、三个 graph hybrid、原 compute/native 权限桥 |
| G03 增量 producer 组合 | [CLOSED](../compute/inference-g03-final-closure.md)，candidate11b2403e7：四原路线/三联合、Direct/Graph、多独立native来源、完整CPU+CUDA源码/两SDK、生命周期与P0通过；原失败/checked waits/费用保留 | 原 exclusive/F2/native waits到更新与读取；精确 dirty 域、成功 ready、idle input 生命周期 | compute_batch、compute/F2、原 native route与graph桥 |
| G04 量化 KV | CPU CE13、Vulkan与Apple CE14有限P1均QUALIFIED/PASS/verify0，Apple已CLOSED。CUDA CE27前置门通过，但[attempt13](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-13/receipt.json)QUALIFIED/strict0，prefill ratio0.940486–0.947439仍非劣FAIL，窗口与留出PASS。CE28加载时转换的新17math/22device及同1/1/4 attention+merge零栈codegen通过，FI为CANDIDATE830；两SDK各33测试/12头/16TU PASS；新两variant各24 oracle/设备检查及实际merge身份PASS；三host各16测试/五风险模式PASS，[attempt14](../../../../evidence/performance/inference-g04-g05-cuda-encoded-w1-wsl-p1-attempt-14/receipt.json)已QUALIFIED/strict0，原[15,31]完成；窗口与留出非劣PASS，prefill ratio0.952972–0.961375但两层CI跨原5%耗时门，非劣UNRESOLVED。有限成本检查已终止，无新FI修正钥匙；FI及共同批聚合门BLOCKED，保留原候选与已过门；按下文调度更正转G06定界，整项未验收。旧失败、未知和累计费用见[活动卡](../compute/inference-g04-g05-cache-attention-domain.md)，整项未验收 | 按成熟实际 cache 编码冻结 payload/scales/layout、quantize append与直接读取，不每步解码完整历史 | compute_ops/batch、实际 Provider/native route |
| G05 attention 语义域 | 同卡已实现 causal-window/full noncausal/有限 Dqk≠Dv，与 G04 共用数学、数值/生命周期和交付证据；适用性能及剩余风险未全部签发 | 固定真实消费者所需 window/mask/维度等域及失败边界，成熟承载；未准入未来特例不无限扩张 | compute_ops/batch、实际 Provider/native route |
| G06 metadata/decode | [共同、CPU Q1及Apple/Vulkan runtime lengths source通过](../compute/inference-g06-metadata-decode.md)：active/delta、首上传共享、准确页准入/旧import释放、CPU Q1原graph/pool/scratch复用；两native固定Q拓扑运行时更新KV indptr/tail，保独立Invocation长度。新D3 Windows91＋18场景、Air实际Direct/联合/pending等27次消费者和Vulkan provider10/alias-phase2项PASS；CPU D2原32×64步/两mixed core各25项保留。owner仍AUDITED；ragged动态extent、FI Q1/capture与最终san/header/SDK未完成；native managed/driver费用UNKNOWN，G04/G05未决保留 | host-known/device-produced分界，active/delta上传、正确 topology/runtime复用与decode档，实际capture资格 | compute_batch、FI/native Provider、graph_batch |
| G07 packed 权重 | 待有限编码选择 | 锁定 artifact→resident→selected math，借原生packed/冷repack；checked IDs保错误，排除warm完整展开/多余host检查 | 实际低精度 Provider/native route及资源桥 |
| G08 数值输出选择 | 待有限数值域选择 | row-wise greedy/selection与选定随机数值域，counter/state、质量/分布和少量结果返回 | compute_ops、实际 Provider/native route；模型采样政策不下沉 |
| G09 联合流水 | 待最小consumer | 原有限slots/依赖推进，真实无依赖CPU/GPU支路及microbatch trace；不重复scheduler | 三个graph hybrid及必要原native执行桥 |
| G10 CPU预算 | 待最小consumer | executor/native private pools明确报告和约束边界，阶段与并发invocation组合 | compute_route_cpu及实际CPU Provider |

编码、数学与数值域由固定 peer/native 实际支持、当前模型和留出消费者共同裁决。总卡不
预先宣称 FP8/Q8/window/MLA/random 的任意组合全部准入，也不以 BF16 单请求不需要为由
删除上述 G 项。确需调整已冻结域时按 owner 协议回退，不在实现途中扩成所有 peer API。

2026-10-15 G06 有限只读预审仅核已有旅程成本：`batch_layout.cc` 每版完整容量分配/清零；
`batch_execution.cc` 每步整 metadata staging/upload 及一或两次 host wait；`graph_batch/batch.cc`
逐节点准备后才共享 mirror；逐页普通 lease read 重复线性查重，最坏 O(G²)。CPU 更新比较
全部 metadata，Apple/Vulkan controller 重建原生 Program/Invocation；CUDA 原 plan 只依赖
Q/KV indptr，页 IDs/tails 不参与该计划是正确的。原生编译期长度/切片不能只靠放松 reuse key。
G06 后续有限 FIX 应覆盖 active/delta 与 immutable generation、Graph 一次上传与原依赖、
实际 lease identity 去重后复用内部 read_unique，以及四路线 topology/runtime 边界与 decode
成本资格。验收沿稳定 tail、跨页、加入/结束、重排、同 shape 不同 layout 记上传字节、
plan/replan、host wait、分配与权限账；复用未变 G03 生命周期/G04–G05 数学，不扩大能力域。
本段是只读预审结论；原活动 owner 仍按 registry，G06 未进入 SCOPED/AUDITED 或取得 PASS。

### G06 有限实施约束与验收草案

本节补齐下一批的输入和反证，尚不切换活动 owner。固定 peer 源码复核没有发现第五组机制；
只保留上述四组，具体接口在活动卡进入 AUDITED 前冻结。引用均使用本总卡的固定版本：

| 有限组 | 成熟承载或现有机制 | 实施约束 |
|---|---|---|
| active / changed metadata | [SGLang host-known 与 active buffers](https://github.com/sgl-project/sglang/blob/7224fce1b5e488c9916f20ec0556d696e595d512/python/sglang/srt/layers/attention/flashinfer_backend.py#L203)；原 BatchSchema / BatchLayoutVersion | SELF，归 compute_batch；按实际有效区与改变区计费。peer 只证明 active buffer 更新，不证明任意 delta 零成本；保 immutable generation、旧快照与 inactive 区合同 |
| 一次共享上传与原依赖 | [llama attention input 的层外创建、逐层消费](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/models/llama.cpp#L113)；原 F2/G03 upload / dependency | SELF，归 graph_batch / compute_batch；同 layout 的共享发生在各 node 上传之前。native workspace、reader 退休和必要跨队列顺序仍由原 owner 管理 |
| 准确 lease 准入 | 原 resource_leases 的 read_unique 和 batch 页权限证明 | BORROW 原内部入口，归 compute_batch 及必要原 controller；先对完整 lease identity 去重，不能用 page ID、地址或 shape 代替。保 permission、poison、writer 冲突和 pending ordering，不重写公共 F1 |
| topology / runtime 与 Q1 decode | [vLLM decode wrapper 与 tensor-core 默认](https://github.com/vllm-project/vllm/blob/98f721116227f0f48690deb05c815c057bac0668/vllm/v1/attention/backends/flashinfer.py#L1310)；原四路线 native plan / Program / Invocation | BORROW 原数学、SELF 必要状态桥，归实际 Provider / native route；按真实 plan 输入、workspace、split 和更新成本选档。不能因 PrefillPlan 名称判缺 decode，也不能只放松 key 而继续使用编译期旧长度 |

正常消费固定为一个请求和多个请求各自一次 Q1：`Q=[B,Hq,Dqk]`、
`q_indptr=[0,1,...,B]`，每请求有独立 KV indptr、page IDs 和 tail，输出 `[B,Hq,Dv]`；
混合 prefill/decode 仍使用现有一般 q_indptr。请求选择、bucket 和调度政策归引擎。
host-known 路线不得暗中 D2H；device-produced 来源须经真实 checked boundary 取得可依赖
的布局事实，不能伪造 host snapshot。capture 按实际路线单独签发，不由 plan reuse 推定。

沿现有 public64 旅程批量验收，新增断言仅限本次四组成本和生命周期：

| 输入或变化 | 必须反证 | 必须保留的账 |
|---|---|---|
| 稳定 tail、跨页 | 不读旧长度/旧页；无改变区不冒充上传 | active / changed / uploaded bytes、plan / replan、host waits、warm allocation |
| 两请求重排、加入/结束、同 shape 不同 layout | shape 相同不代表 identity 相同；capacity 超限仍拒绝 | 完整 topology key、实际 native 对象重建与 workspace |
| 两个真实 attention node 使用同 metadata | 只共享同 generation 且队列/依赖合格的上传；不共享可变 workspace | 上传次数、各 node 必要等待与实际退休 |
| K/V 重复 readonly lease、不同 generation、pending writer→reader | 完整 identity 去重；权限、poison、失败 suffix 与旧结果不变 | permission admission 次数与复杂度，不以去重绕过安全检查 |
| host-known / device-produced 来源 | 无隐式 D2H；未完成或失败来源不能发布 ready | checked boundary、实际 host / native waits 与未知成本 |

四路线和三个联合消费者继续承担已准入域；未变数学与 G03 生命周期证据按实际输入借用，
变动部分补对应 source / sanitizer / static / shared / header / 直接消费者门，不启动全库验证。
先批量完成这四组，再集中检查并针对实际失败修复；本草案没有运行测试，也不计作 G06 完成。

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


2026-10-14 CPU精确page16读取[有限轮终审](../compute/inference-g01-cpu-paged-terminal.md)已
CLOSED：provider1e9/runtime42f，source/三san/relocated static3+shared3/实际Air mixed CPU
全部PASS；正式3d05合同1657.156s QUALIFIED/三个5%非劣/全部controls/verify0，D256
小幅LOSS如实保留。G02共同exact-read group接缝MQ-LP-01已实现5ce；首fixture默认
256字节对齐下两块backing不足的FAIL保留，修fixture后Windows八source tests PASS。
graph仍AUDITED，风险/交付与native分页输入数学/权限桥仍须完成。全项计数继续0/10，
GPU高性能页/G02native/G03–G10不由这两个有限进展代签。


2026-10-14 Apple/Vulkan原生page16 [有限域终审](../compute/inference-g01-native-paged-terminal.md)
已CLOSED，candidate520；两route真实数学/风险/SDK/宿主san/定向Win链接及固定同合同
P1全部完成。Apple938.810s、Vulkan1493.406s，两者QUALIFIED/三个5%非劣PASS/verify0；
原FAIL及Metal宿主TSan显式模型、private UNKNOWN、Win未run限制保。全项仍0/10。
G02新hybrid三owner仍AUDITED：公开联合草稿Apple编译已过，首次actual CPU/native/CPU
运行FAIL，readonly Q导入资源身份与原controller strict query resource校验不一致；
须在合法owner审计后修正该真实接缝，不能删除资源校验或复制controller来取得PASS。

2026-10-14 G03完整有限组合已按[终局](../compute/inference-g03-final-closure.md)CLOSED，
正式计数1/10，整体阶段粗估约28%。G01/G02旧未决门仍原样保留，不由新G03回填为PASS；
G04–G10继续。下一实现批依原总卡转G04/G05编码与数学域定界，避免重复已闭合G03。


2026-10-15 当前 CPU CE13 性能已通过；停止重复该 immutable-history 有限合同。
Vulkan 已在原 CPU 真正终态后从干净源码启动实际 Air P1；Apple 数值→P1、CUDA 数值→P1
继续沿真实进程/远程终态串行。一次 canonical-clean 接力拒绝发生在 CUDA benchmark 文件
提交期间，未创建 Vulkan run/远程子进程；原5.5秒失败保留，源提交完成后才启动首个真实 run。
G01/G02 表项已按原 finite terminal 修正过时摘要，完整项仍仅 G03，不由摘要更新代签收口。


2026-10-15 G01 原 plain 有限目标已聚合完成，完整项更新为2/10（G01、G03）。
这不表示整体工作量20%；各项大小不同。原 G02 Windows 三个未完成命令继续保持未证明，
当前 G04/G05 owner 状态仍按 registry，不因 goal 工作项聚合回写 CLOSED。

### 2026-10-15 调度纠正：先补 G02 验收尾项

用户指出G02未终审却继续G03–G05。原Windows单CUDA测试在2026-10-13桌面资源压力下
中断，Vulkan构建/测试未运行；后续独立工作不补签这些门。当前Apple attempt5真实结束
并封存后，优先本次重新纳入的G02原3门，再处理CUDA失稳及G04/G05终审；G06–G10
保持原完整目标，G06暂不实施。G01–G05即使全部验收也只有5/10整项，不宣称全库收口。

有限验证仅原`factor:graph_route_cuda` command1与`factor:graph_route_vulkan` commands0/1，
见[原施工卡](../graph/inference-g02-multi-queue-batch.md)和
[保留证据](../../../../evidence/correctness/inference-g02-common/receipt.json)。Windows原树
`C:/h7/inference-g02-impacted`已落后当前源码；先正常重新配置并核实际source/native输入，
按当前不可变输入增量更新两测试target所需闭包。CUDA旧build PASS不作为当前artifact资格，
必要增量构建是输入重绑定，不是复跑已完成门。只构建两原target，max4jobs/CUDA TU pool1，
不启动全库、模型/server或P1。CMake/native产物变化如导致额外大规模重编，先记录具体
差异及可复用的相同输入，不因路径改变就重建未变库，不修改公共构建系统。

两原CTest correctness recipe的完整默认用例集合保持；为避免同进程vendor缓存累计到
11.5GiB，允许scratch编排先枚举同一exe全部默认GoogleTest用例，再按单case独立进程
串行执行并保存每项exit/XML/raw。这是资源隔离的等价测试覆盖，不伪造未运行的原ctest
command为PASS，不缩减形状/断言/用例；disabled、真实skip与失败逐项保留并裁决。每个
进程终态后才启动下一个，记录GPU/主存观测；不改全局驱动/电源/线程设置。原压力中断、
错误和elapsed全部保留，普通/SDK/风险的原已过资格按匹配输入与实际影响借用，不重跑。
本阶段为已有有限G02的证据补完，无产品FIX和模块状态跳转；若发现真实产品缺陷，按当前
owner退回AUDITED并固定修正后才改产品。2h检查点/自动追加，不以墙钟到点假签验收。

2026-10-15 G02补验：CUDA完整默认804项先797PASS/7缺fixture环境skip，核对既有真实
Qwen3.5-2B fixture全部25文件SHA后仅补7项，实际7PASS/0skip；原记录保留。Vulkan完整
默认集合851项在第79项终止：78PASS/0skip，24T attention对旧F16参考的误差约9.18e-5
超过原门。scratch同输入诊断确认当前public SDPA的全F32程序对原独立F32方程最大仅
8.20e-8，strict-F32误差/原门最大0.00101；产品与原exe/source未改。有限测试FIX仅
Direct/Graph两wrapper按明确F32输入/累加的SDPA操作选择原F32参考，保原8e-5+4e-5门，
不放宽容差、不改公共fixture/其它native参考/产品或已合格P1。先定向两入口验证该例，
再续原Vulkan未运行集合与本次受影响ID；原78已过且参考不受影响的案例不重复。G02仍未
验收，不因诊断PASS代签851集合；原失败、原skip及全部成本保留。


2026-10-15 G02补验已完成原Windows两个默认集合：CUDA804通过；Vulkan851先842PASS/
9FAIL/零skip，完成两组有限测试参考/codec fixture纠正后，九Graph及九Direct实际PASS，
完整默认851聚合通过。新Vulkan产品修正的正常source四项也已通过，详见
[Windows原默认门证据](../../../../evidence/correctness/inference-g02-vulkan-windows-default-acceptance/receipt.json)。
仅本FIX对应Air源码/static/shared与alias/phase/metadata尾验继续串行；原所有失败和费用
保留。完成项仍2/10，G02尚未聚合验收；不提前CUDA P1或G06，不重测已签域。


2026-10-15 G02原有限范围已[整项验收](../graph/inference-g02-aggregate-acceptance.md)。
原CUDA804/Vulkan851默认门和CE15/16的Windows正常source、实际Air源码/static/shared均
通过；Air source另过独立输出端口及metadata失败/move/清理。13份证据逐项核SHA与原
requirement；历史FAIL、预算/费用和private UNKNOWN保留。goal更新3/10（G01/G02/G03），
停止重复原G02门，继续已声明CUDA测量诊断与G04/G05资格；G06–G10完整目标不缩减。
本次仅goal聚合，不批量关闭G04/G05的registry候选。

### 2026-10-16 调度更正：结束 CE28 搜索，推进 G06

[当前有限检查终局](../compute/inference-g04-g05-cache-attention-domain.md#ce28-有限成本检查终局与-g06-调度更正)
没有发现新FI material消费与源码机制双钥匙。原attempt14终态与UNRESOLVED保留；
按原门将FI及受阻共同批明确裁决BLOCKED，goal保持ACTIVE，G04/G05不签完成。
已冻结G06四组由原独立成本事实准入新卡，允许在G04/G05聚合验收前先定界和施工；
实施前仍逐owner完成合法阶段，不借调度更正改未授权产品。不保证共有费用下降会
改善CUDA相对比率，不为重跑旧候选而改消费者。G01–G10完整目标、旧失败/未知/费用
均保持，完整完成数3/10。

2026-10-16 G06 两共同 owner 已合法恢复 SCOPED，见[单卡](../compute/inference-g06-metadata-decode.md)；
四组完整范围、原生与联合资格、冷热成本/权限/失败边界和预声明验收均保留，当前未改产品。
