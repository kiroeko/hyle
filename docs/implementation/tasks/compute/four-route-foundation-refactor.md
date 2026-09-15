# Compute/Graph 四路线基础重构执行计划

2026-09-15 用户批准将七个工作包落成计划并绑定首包；baseline
`98778b555f25acc172722efa7ab8a9911032c520`。依据是已完成的[六项架构复核](six-point-architecture-review.md)，
长期合同仍只由 [Compute](../../../architecture/compute-design.md)、[Graph](../../../architecture/graph-design.md)
与[四路线设计](../../../architecture/compute-routes.md)持有。本页只负责执行顺序、产物和完成判据。

上轮只建立计划和 R1 施工卡。2026-09-15 用户明确授权实际实施并创建新的四路线 goal；该 goal 已建立，
不恢复旧 goal/百分比。每包仍须绑定 owner 的当前卡，计划、旧 AUDITED 卡或独立探针不是产品修改权限。
首包入口：[R1：路线选择与交付隔离](../build/compute-route-isolation-v1.md)，实施、适用 native 验证
与七门已完成，R1 CLOSED；R2–R7 尚未完成。当前进入 [R2 有限审计与共同语义](four-route-common-semantics-v1.md)。

## 目标与不做项

交付 CPU、CUDA、Apple MLX/Metal、Vulkan 四条各自原生的基础计算路线：普通 C++ 用户既可直接准备/
执行计算，也可用 Graph 冷编译并重复运行；真实成熟实现与 HYLE 自定义实现可自然组合、渐进替换，
不因包装被迫复制、同步、改精度或逐 kernel 动态选路。四路不必共享 compiler、执行器或对象布局。

本计划完成不代表全平台算子大全或完整推理引擎完成。模型 CPU+GPU 联合推理不在产品范围；host
控制/传输/同步保留。HIP、模型格式/KV/batching/serving、多 GPU 模型切分和任意动态控制流不纳入。
不为兼容旧 API 保留双轨实现；也不因新架构删除有需求的既有能力或已证明有效的 HYLE 算法。

CPU 是硬件执行路线，oneDNN 是该路线可选择的算子实现，不是第五条平级路线。CPU 路线可组合
HYLE Native/SIMD 与 oneDNN；不用 oneDNN 仍有 Native 计算能力。当前 HYLE 的 oneDNN 接入限于
x64 CPU BRGEMM，使用 SEQ 配置，由 HYLE 在外层组织并行；这不是对 oneDNN 本身能力的完整描述，
也不表示它总比 Native 快。研发时按证据固定适用域的优选规则，冷准备只匹配规则并绑定；正常用户
不排序 Provider，也不提供 benchmark/profile，见 [Compute 的规则合同](../../../architecture/compute-design.md#41-开发时验证-best-of准备时应用规则)。

## 七个工作包

| 包 / 顺序 | 有限产物与责任 | 完成判据 |
|---|---|---|
| R1 迁移与构建隔离 | 核对下表旧资产；按实际 feature 投影路线与默认实现闭包，区分正常发行和专家裁剪；首轮处理已有 CPU/CUDA/oneDNN，不预造 Apple/Vulkan 空开关 | CPU-only、CUDA-only、无 backend 底座与三交付闭包可解释；规则不能指向未交付实现；未选路线不泄漏 SDK/flags/运行库；新 API 草图按 owner 分配 |
| R2 必要共同语义 | 在各 owner 卡落实生成结果 owner、直接 delegated 执行、真实完成/失败与量化格式；固定自然 C++ 入口与既定优选规则的接入边界，保留必要 shape/resource/alias 机制 | caller/provider-output 均可表达；Direct 不依赖 Graph；普通用户不排列 Provider 或提供 profile；四路线具体映射、CPU 与已证实 MLX 机制反证，不强制共用旧搜索器/vtable |
| R3 CPU 完整纵切 | Native/SIMD、oneDNN + public Sched；对应 Graph lowering；在已支持域验证并交付默认实现/融合规则；盘点既有 19-family，不填大全 | 正常 Direct/Graph 入口自动应用既定规则；变化输入 FFN/批处理、worker/嵌套/并发 scratch/关闭；同合同 direct 税与完整性能、平台/三交付及下文规则门通过 |
| R4 CUDA 完整纵切 | vendor 浮点 GEMM、正式 W4A16 compressed carrier、HYLE 中间 kernel；对应 Graph lowering/capture；交付各适用域已验证规则 | 默认入口选择已验证实现而非用户点名；压缩常驻、冷 packing、有限 shape、两槽/变化输入；接纳/晚到错误/回收；Windows/Linux 三交付、性能与规则门，不用旧 Marlin 探针签现代实现 |
| R5 Apple 完整纵切 | MLX-Metal region、同-runtime HYLE 扩展、对应 Graph lowering、结果 lease/提交 owner/完成；必要 Metal 专家入口；完整片段优选规则 | 普通量化 FFN 和资源/自定义旅程自然应用规则，无影子输出 copy；并发/保留结果/失败关闭、成本/三交付与规则门；M4 不替 M5/NAX 签证 |
| R6 Vulkan 完整纵切 | 原生 pipeline/program/生命周期、成熟计算片段、对应 Graph lowering、已验证的设备/格式/shader 规则；MoltenVK portability | 正常 Direct/Graph 规则绑定、重复执行、计算/传输依赖、native handoff、回收、性能/三交付与规则门；单机 smoke 不替 AMD/Intel/MoltenVK 签证 |
| R7 基础终审与清理 | 汇总前六包 receipts，核对各路线已经交付的默认规则与普通消费者；最小 opt-in Obs；旧入口/无用途实现清理和指南校准 | 本轮 MUST/适用门和依赖尾项有结果；默认 best-of 不留到本包首次接线；无隐藏 CPU fallback、双重规划、未计费桥或失效文档；报告后暂停，不扩算子 |

R1 → R2 → R3 → R4 → R5 → R6 → R7 是默认工作顺序；R3–R6 各自含 **Direct + Graph + build +
correctness + performance**，不是最后才拼 Graph。R2 提供可用共同语义，不提前实现一个万能热循环。
每条路线在自己的包内交付默认规则与证据，R7 只复核和收尾；没有明确优势的域采用合格成熟基线，
不为完成规则表虚造 HYLE 胜出阈值。首轮允许只覆盖实际已交付计算，不能声称完成全部未来算子选型。
某路线实际 API 的局部问题由所属卡处理，不能让独立路线的已证实结果清零。需要跨卡修改时记录精确
依赖与回放，不顺手改公共仪器或关闭的底层模块。

## 旧代码去留与 owner

以下是具体迁移范围，不是未经复验的 KEEP verdict。删除与替代按包原子完成；无可用替代时先报告
能力损失，不能只为减行数删掉功能。历史 runtime/算法证据可定向复用，不变成新架构整体证书。

| 现有资产 | 新计划中的处置 | owner / 包 |
|---|---|---|
| `compute_ops` 的 tensor/typed operations、custom identity；shape/overflow/布局验证 | 保留需求与可用算法，完整量化输入/数值不能由旧两输入 MatMul 冒充；现有公开入口逐族映射，新形状无需兼容旧签名 | compute_ops、compute；R2/R3 |
| `src/compute/operation.cc`、prepared callbacks/state、custom kernel、completion/backend extensions | 改写只认 caller output/Queue 的限制；直接 delegated runtime 与 owning result 不依赖 Graph；不用一个回调表强迫所有 native 操作同形 | compute；R2，各路线 |
| `src/graph/executable.cc`、plan state、公共 placement/queue arrays | 保留必要 DAG/资源事实；拆开冷逻辑与路线专属执行；普通用户不手拼三套 ID；撤销模型 CPU+GPU 联合路径，保留 host transfer | graph；R2，各路线 |
| shape program、resource effects、partial-order lifetime、constant sharing、bounded search/profile | 保留必要语义与资源规划；旧 profile 接收/多计划性能重选不成为默认路径，超出本轮规则绑定需求的公开面/实现随迁移清理；历史数据仍是研发证据，不强制生成兼容层或新 tuner | compute/graph；R2，各路线，R7 检漏 |
| CPU backend、Native/SIMD kernels、oneDNN SEQ ukernel 与 region | 底层机制和胜出算法是复用候选，API/包装按新合同迁移；没有新机制问题不重做 kernel；公开 persistent executor 已有可用组合 | CPU/native/oneDNN owners；R3 |
| CUDA backend memory/stream/event/capture | 复用符合新合同的 native 机制，因实际接口/交付变化才重开其 CLOSED 卡；不按单词删除 capture+普通命令的 `hybrid replay` | CUDA owner；R1/R4 |
| `compute_provider_ggml` BLOCKED，`compute_sched` QUEUED | CPU ggml 恢复条件保持；不强制它成为 CPU 必经路径。现有 public Sched 组合成立，不为预留名称实现冗余 pool；其身份是否退役另按组件合同处理 | 原 owner；不自动解锁/建空实现 |
| `mem` v10 Reservation / Graph quota 消费 | 已实现未终局验收的定向依赖尾项，纳入资源回收回放与本轮结案；不复制 quota，也不把 Mem 误记为已完成该轮 | Mem 当前卡 + Graph；R2/R7 |
| research 程序、旧 Q4 展开案例、历史证据、bench fixtures | 原生研究可转成具名 consumer，但不是 shipping；旧全量展开路线不恢复为量化默认；模拟测试只覆盖协议分支，不能代替 native lifetime/性能 | 对应包；R7 检漏 |

当前 owner/record 仍只看 registry。R2 开始前，把需改动的现有 AUDITED owner 依法重新定界/更新活动卡，
不能继续旧 FIX 后声称满足新目标。新增 backend/provider 在开始实现时才登记；CLOSED 因交付/API
变化重开时走合法状态流，非相关底层不重开。Mem 尾项见[当前 v10 卡](../mem/mem-v10-external-byte-reservation.md)。

## C++ 消费形状与路由界限

这是调用顺序/owner 约束，不是已发布代码或固定类名；R2 的真实可编译 sample 再锁定具体拼写。

```text
Direct: 选择明确 native route → 准备计算 → 建运行槽/绑定输入输出策略 → 提交 → checked 完成/取结果
Graph:  构造 typed DAG → 为所选 route 冷编译 → 建运行槽/绑定 → 重复固定程序 → checked 完成/取结果
```

- 普通用户不用供应 Provider vtable、storage/lane ID 表或自写通用失败回收 glue；专家 native interop
  位于明确路线头/target。HYLE Graph 不是调用 MLX/ggml runtime 的前提。
- caller destination 与 Provider-generated result 是两种真实输出合同；后者保有原 owner。接纳后即便
  提交前缀失败，也要保有真实 Completion 与依赖资源；错误返回不能丢掉已经在执行的工作。
- source/static/shared 同样支持 typed custom execution，跨 DSO 不靠静态地址做 identity，不恢复 C ABI。
- 默认按合格成熟完整路线执行；新 HYLE 实现仅替换有证据的 device/格式/shape 域。内扩展能保留融合时
  在研发时比较完整收益，不强制把 opaque region 切成小段。准备期只按既定规则绑定，热路径无查库/
  ISA 探测/搜索/首次 JIT；不把原始 timing profile 变成普通用户输入。
- MLX-managed 内部真实成本计入完整路线；宏/模板只消除静态可消除的层，不能掩盖动态分发/同步。

## 验证、性能和停止

每包先固定 consumer/peer 的正常调用及成本账单，correctness 包括真实 owner、非法输入、OOM/部分
接纳、并发、晚到错误、取消/关闭。CPU、CUDA、Apple、Vulkan 分别按可用的真实 API 做有限故障检查；
不故意制造 GPU hang/device loss，也不把没有注入的路径写 PASS。测试适用性由各卡说明。

R3–R6 各自的默认规则门：普通 public C++ consumer 不排序 Provider、不提供性能资料，也能取得研发
确定的选择；检验实际适用条件、裁剪/缺席实现、合法基线和明确拒绝。若有不同域的真实赢家，覆盖相应
边界和留出 shape；不为测试制造规则。源码/P0 确认准备不启动跨 Provider 性能试跑、重复执行不重选；
source/static/shared 在相同能力配置下得到一致选择。完整 P1 测自然默认入口，不能只测强制 Provider
的专家样例。规则接口与算法固定后才进入候选；性能纠正按原预算处理，不能因裁决规则而无限开新实验。

性能不集中到最后：R3–R6 各自给出同实现 direct 包装税与相同数值/格式/shape/资源/完成边界下的完整
片段比较。CPU 使用已选 Native/oneDNN direct，CUDA 使用选定 vendor/kernel direct，Apple 使用 MLX
原始 region，Vulkan 使用对应 native/成熟 carrier；再选一个最强适用完整 peer。不同格式/数值的自然
路线另报，不能签 WIN/LOSS。没有微小可分辨优势就保留成熟实现，但不能把固定小输改写成 parity。

只有 API/算法稳定的不可变 candidate 跑预声明 P1；先 P0/源码归因，再短 preflight，通过才正式测。
纠正预算在各卡固定，禁止为了换方向无限重测；文档/状态/格式不重跑。R7 优先复用匹配的 receipts，
只新增真实组合缺口，不做无理由全库重建。公共工具缺少必需 verdict 时另记工具任务，不扩仪器工程。

CUDA 资格对应 Windows/Linux 真 NVIDIA；CPU 对应 Windows/WSL/macOS 的实际支持组合；Apple 对应
原生 macOS；Vulkan 原生与 MoltenVK 分别取证。Mac mini 仅 correctness/portability，正式 Apple P1
使用已登记 Air profile。没设备的 AMD/Intel/M5 等保持未资格化，不从单机外推 SOTA，也不要求无设备
就永远停在研究。需要调整已承诺平台范围时必须显式裁决，不写成 NOT_APPLICABLE 偷过门。

完成点是本轮明确支持设备上的基础能力、七门与实际限制报告，不是“代码都写了”。未解决的必需功能/
性能/失败/交付问题按协议留给精确 owner；局部问题不抹掉其它路线证据。只有反例推翻共同语义、所有权
或成本责任时重开共同设计，不因新 op 或 local kernel tuning 重排全部计划。

## 进度口径

不沿用旧 55%/90%，也不把七个包等权平均成实现百分比。计划和卡建立不计产品完成；每包报告
“已实现 / 已验证 / 未证明”及证据，R7 才汇总基础是否完成。源码规模按同一 commit 的 Compute 全部
backend/provider 与 Graph 全部 provider 统计，不仅计 runtime factor；LOC 不是完成度。
后续只在工作包完成、重大范围变化或用户询问时报告，不为文档提交增加实现分数。

本轮 baseline `98778b55` 的 `loc-report.py --self-check` 统计：Compute **17,000**、Graph
**14,135**，合计 **31,135** 行自实现，不含测试。统计包含各自 `include/hyle` 公共头及 `src` 下
全部已有 backend/provider，包含待迁移代码，不表示已有这么多代码满足新架构。同期全库代码
688,842 行，自实现不含测试 126,320 行；本轮只有文档改动，产品实现行数未增长。

## 默认优选决策校准（2026-09-15）

文档变更 baseline：`92a5cf875f91fcefb125c6f5a301c88479c19c96`。按用户确认，把研发时验证、交付
规则、准备时匹配、执行时固定分开；同步 Compute/Graph/四路线、build/roadmap、首卡与当前用法说明。
R3–R6 各自验收默认规则和普通入口，R7 不再首次接线；旧计时 profile API 按需求迁移，不强制保留。

上轮实施就绪判断：可以按上述七包进入有界施工，无需再等待一轮全架构研究或全部未来算子性能证明。
当时 R1 为 SCOPED，先完成卡内定向审计和有限 FIX 再进入 AUDITED；R2 的具体 C++ 签名/规则接入和各
路线的 native API 细节仍须在各 owner 实现前固定。未知设备性能不外推，实际反例按所属卡处理，
不能把此判断当成四路线实现/生命周期/性能已经验收。

本次 `validation-select.py run --base 92a5cf87 --candidate WORKTREE` 选择 docs/closure 并 PASS，
其余 94 项跳过或复用；`git diff --check` PASS。产品代码/registry/原始证据不变，未构建、未运行 P1，
上述实现 LOC 未变。研究记录仅修正文档锚点，不改写历史实验结论。
